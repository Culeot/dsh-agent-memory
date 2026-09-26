"""把检索模型导出成 transformers.js 能直接加载的 ONNX 目录。

为什么要这一步
--------------
插件是 TypeScript(Node)。要让它**在进程内**算语义向量,只有三条路:

  A) Python 侧车进程 —— 多一个进程要管,启动、崩溃、端口、生命周期全是新故障点;
  B) Node 里跑 ONNX —— transformers.js 直接加载,离线、无外部依赖(**选这条**);
  C) 调远端 API —— 记忆内容是私密的(路径、凭据痕迹、项目决策),不走网络。

已有一个同生态的先例证明 B 可行:`dsh-skill-router` 用同一套
(transformers.js + onnxruntime-node)在 DSH 插件里跑 bge-m3。

关键坑:不要导出 pooling
------------------------
训练时的向量 = **mean pooling + L2 归一化**(见 mem_train.py 的 BiEncoder.forward)。
而 transformers.js 的 feature-extraction pipeline **自己会做池化**:
    pipeline('feature-extraction', dir, { pooling: 'mean', normalize: true })

所以这里导出**纯 encoder**(输出 last_hidden_state),把池化交给 pipeline 做。
如果图省事把带池化的整个模型导出去,pipeline 会再池化一次——
向量仍然"有值"、余弦仍然"像那么回事",但已经与训练时不一致,而且**不会报错**。
这类错误最难查,所以宁可在这里多写两行注释。

产物
----
    <out>/
      config.json  tokenizer.json  tokenizer_config.json …   ← 从源模型复制
      onnx/model.onnx               ← fp32
      onnx/model_quantized.onnx     ← int8 动态量化(约 1/4 体积,CPU 更快)

用法
----
    .venv\\Scripts\\python.exe src\\mem_export_onnx.py \\
        --model models\\mem-retriever --out models\\mem-retriever-onnx
    .venv\\Scripts\\python.exe src\\mem_export_onnx.py --skip-quant   # 只要 fp32
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

# --- 路径:全部从环境变量读,脚本放哪都能跑 --------------------------------- #
import os

WORK = Path(os.environ.get("MEM_TRAIN_WORK", "work"))
MEMORY_JSON = Path(os.environ.get("MEM_TRAIN_MEMORY",
                                  str(Path.home() / ".dsh" / "storages" / "memory.json")))
BASE_MODEL = Path(os.environ.get("MEM_TRAIN_BASE", "models/chinese-roberta-wwm-ext"))
LLM_MODEL = Path(os.environ.get("MEM_TRAIN_LLM", "models/Qwen2.5-0.5B-Instruct"))
COPY_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
)


def mean_pool_normalize(hidden, mask):
    """复刻训练时的池化:mask 加权平均 + L2 归一化(用于一致性验证)。"""
    m = mask.unsqueeze(-1).float()
    vec = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)
    return vec / vec.norm(dim=-1, keepdim=True).clamp(min=1e-9)


class EncoderOnly(torch.nn.Module):
    """Thin wrapper exposing only `last_hidden_state`.

    Why the wrapper is needed: transformers 5.x changed BertModel's positional
    parameter order, so handing `(input_ids, attention_mask)` to onnx export as
    positional args collides with `use_cache`
    (measured: "BertModel.forward() got multiple values for argument 'use_cache'").
    Calling through with **keywords** sidesteps the signature entirely and also
    pins the graph output to exactly one tensor, which is what transformers.js
    expects from a feature-extraction model.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def main() -> None:
    ap = argparse.ArgumentParser(description="导出 transformers.js 可用的 ONNX 模型")
    ap.add_argument("--model", default=str(WORK / "model"))
    ap.add_argument("--out", default=str(WORK / "model-onnx"))
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--skip-quant", action="store_true")
    args = ap.parse_args()

    src, out = Path(args.model), Path(args.out)
    if not (src / "config.json").exists():
        print(f"[错误] 源模型不存在:{src}")
        sys.exit(1)
    (out / "onnx").mkdir(parents=True, exist_ok=True)

    from transformers import AutoModel, AutoTokenizer

    print(f"[加载] {src}")
    tok = AutoTokenizer.from_pretrained(str(src))
    model = AutoModel.from_pretrained(str(src)).eval()
    encoder = EncoderOnly(model).eval()

    probes = ["npm 发布 token 过期", "把包推到仓库时凭证失效怎么办"]
    dummy = tok(probes, return_tensors="pt", padding=True, truncation=True,
                max_length=args.max_len)

    fp32_path = out / "onnx" / "model.onnx"
    print(f"[导出] fp32 → {fp32_path}")
    t0 = time.perf_counter()
    torch.onnx.export(
        encoder,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"        {time.perf_counter() - t0:.1f}s,{fp32_path.stat().st_size / 1e6:.1f} MB")

    for name in COPY_FILES:
        p = src / name
        if p.exists():
            shutil.copy2(p, out / name)
    print(f"[复制] tokenizer/config → {out}")

    q8_path = None
    if not args.skip_quant:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        q8_path = out / "onnx" / "model_quantized.onnx"
        print(f"[量化] int8 动态量化 → {q8_path.name}")
        quantize_dynamic(str(fp32_path), str(q8_path), weight_type=QuantType.QInt8)
        print(f"        {q8_path.stat().st_size / 1e6:.1f} MB")

    # ---- 一致性验证:ONNX 必须复现 PyTorch 的池化向量 ----
    import numpy as np
    import onnxruntime as ort

    with torch.inference_mode():
        ref_hidden = model(**dummy).last_hidden_state
        ref = mean_pool_normalize(ref_hidden, dummy["attention_mask"]).numpy()

    feeds = {
        "input_ids": dummy["input_ids"].numpy().astype(np.int64),
        "attention_mask": dummy["attention_mask"].numpy().astype(np.int64),
    }
    for label, path in (("fp32", fp32_path), ("int8", q8_path)):
        if path is None:
            continue
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        hidden = sess.run(None, feeds)[0]
        got = mean_pool_normalize(torch.from_numpy(hidden), dummy["attention_mask"]).numpy()
        cos = [float(np.dot(ref[i], got[i]) / (np.linalg.norm(ref[i]) * np.linalg.norm(got[i]) + 1e-9))
               for i in range(len(ref))]
        verdict = "一致" if min(cos) > 0.999 else ("可用" if min(cos) > 0.99 else "!! 偏差过大")
        print(f"[验证] {label:5} 与 PyTorch 向量的余弦:{[round(c, 5) for c in cos]}  → {verdict}")

    (out / "export_report.json").write_text(json.dumps({
        "source": str(src),
        "fp32_mb": round(fp32_path.stat().st_size / 1e6, 1),
        "int8_mb": round(q8_path.stat().st_size / 1e6, 1) if q8_path else None,
        "pooling": "mean + L2 (applied by the consumer pipeline, not baked into the graph)",
        "probes": probes,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[完成] 产物目录 {out}")
    print("       插件配置 semanticModelDir 指向它即可(Node 侧用 dtype='q8')")


if __name__ == "__main__":
    main()
