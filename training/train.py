"""检索模型训练 —— 把 encoder 训成「记忆检索专用」。

用户要什么
----------
不是「懂语义的模型」，是「一个专门做检索本职的模型」：
喂一个查询,它给每条记忆打分,挑出最该推送的那条。

为什么之前不行
--------------
评测实测（本目录的 eval.py / mem_eval.py）：

    纯词面基线（插件现有）       v1 MRR 0.5965   v2 MRR 0.7708
    chinese-roberta 句向量       v1 MRR 0.3129
    训练版句向量（v1 数据）      v1 MRR 0.6309   v2 MRR 0.7121  ← 在可信评测上输给词面
    融合（词面 + 语义）                            v2 MRR 0.8432  ← 真正有效的用法

结论两条,都很关键:
1. 语义**单独**不该替代词面 —— 它的价值在融合里（v2: 0.7708 / 0.7121 → 0.8432）;
2. v1 数据是**问句式**合成查询,而真实查询是**关键词堆叠式**,
   分布错了模型就在真实场景吃亏(v2 的 kw 类:语义 0.8533 vs 词面 0.9505)。
   → 所以有了 `gen_data.py`,训练时优先用分布对齐的数据。

怎么训
------
标准的对比学习（bi-encoder）：

    正样本对 (query, 它对应的记忆)   → 向量拉近
    同批次内其他记忆 = 负样本        → 向量推远

这是 sentence-transformers 的 MultipleNegativesRankingLoss 思路,
手写实现（不引额外依赖）。批越大、负样本越多、效果越好——
所以 GPU 吃满就是「把 batch 拉满」。

用法
----
    # 默认(读 work/train_pairs.jsonl,产物落 work/model)
    python training/train.py --epochs 30 --batch 32

    # 另存一份,不覆盖老模型,便于对照
    python training/train.py --pairs work/train_pairs.jsonl \
        --out work/model-v2 --epochs 30 --batch 32
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

# CPU 让路规则(项目铁律):本机常驻量化训练(约 2 核)与进化论舰队(约 6 核),
# 训练脚本一律只吃 2 线程。Jev 的 CPU 占用靠这两行从 0.65 核降到 0.11 核 —— 别删。
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
PAIRS = WORK / "train_pairs.jsonl"
BASE_DIR = BASE_MODEL
OUT_DIR = WORK / "model"
LOG_DIR = WORK / "logs"


# ---------------------------------------------------------------------- #


class BiEncoder(nn.Module):
    """共享权重的双塔：查询和记忆用同一个 encoder + mean pooling。

    bi-encoder（而不是 cross-encoder）的原因：记忆向量可以**离线预计算**,
    查询时只算一次余弦相似度——毫秒级。cross-encoder 每条候选都要过一遍模型,
    376 条记忆每次查询要过 376 次,不可接受。
    """

    def __init__(self, base: str):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(base)

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        summed = (h * mask).sum(1)
        cnt = mask.sum(1).clamp(min=1e-6)
        vec = summed / cnt
        return F.normalize(vec, dim=-1)          # L2 归一化 → 点积即余弦


def load_pairs(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def make_batch(rows: list[dict], tok, device: str, max_len: int = 192):
    """一个 batch = 若干 (查询, 正样本记忆) 对。

    批次内其他记忆自动成为负样本 —— 所以 batch 越大越强。
    """
    qs = [r["query"] for r in rows]
    ms = [r["content"] for r in rows]
    qe = tok(qs, return_tensors="pt", padding=True, truncation=True,
             max_length=max_len).to(device)
    me = tok(ms, return_tensors="pt", padding=True, truncation=True,
             max_length=max_len + 128).to(device)
    return qe, me


def main() -> None:
    ap = argparse.ArgumentParser(description="训练记忆检索模型")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64, help="越大负样本越多(吃显存)")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--temp", type=float, default=0.05, help="InfoNCE 温度")
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=1)
    # v2 新增:数据与产物路径可指定 —— v1/v2 要能对照跑,不能互相覆盖
    ap.add_argument("--pairs", default=None, help="训练数据 jsonl(默认 work/train_pairs.jsonl)")
    ap.add_argument("--out", default=None, help="模型输出目录(默认 work/model)")
    ap.add_argument("--log-dir", dest="log_dir", default=None, help="日志目录")
    args = ap.parse_args()

    pairs_path = Path(args.pairs) if args.pairs else PAIRS
    out_dir = Path(args.out) if args.out else OUT_DIR
    log_dir = Path(args.log_dir) if args.log_dir else LOG_DIR

    if not pairs_path.exists():
        print(f"[错误] 没有训练数据 {pairs_path}，先跑 gen_data.py")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = load_pairs(pairs_path)
    # 同一记忆的多个查询留作训练，但要保证 batch 内不出现「同一记忆的两个查询」
    # 否则它们会互相成为假负样本。按记忆分组后每组只取一个进 batch。
    by_mem: dict[str, list[dict]] = {}
    for r in rows:
        by_mem.setdefault(r["id"], []).append(r)
    mem_ids = list(by_mem.keys())
    print(f"[数据] {pairs_path.name}:{len(rows)} 个样本 / {len(mem_ids)} 条记忆", flush=True)
    if len(mem_ids) < 8:
        print("[错误] 记忆种类太少，无法做批内负样本")
        sys.exit(1)

    print(f"[加载] {BASE_DIR.name} ...", flush=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    model = BiEncoder(str(BASE_DIR)).to(device)
    # 注意：**不能** 手动 .half()。
    # 混合精度训练要求权重保持 fp32、由 autocast 自动处理前向的 fp16 计算；
    # 把整个模型转半精度会让梯度也变成 fp16，GradScaler.unscale_ 会直接报错
    # （实测：ValueError: Attempting to unscale FP16 gradients）。
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[模型] {n_par:.1f}M 参数 device={device} dtype={next(model.parameters()).dtype}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = max(1, len(mem_ids) // args.batch)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup))

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    # 新版 AMP API（torch 2.6+ 里 torch.cuda.amp.* 已弃用）
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"\n{'=' * 72}")
    print(f"训练检索模型   epochs={args.epochs} batch={args.batch} "
          f"steps/epoch={steps_per_epoch} 总步数={total_steps}")
    print(f"输出目录 {out_dir}")
    print(f"{'=' * 72}\n", flush=True)

    log_path = log_dir / f"train_log_{out_dir.name}.jsonl"
    gstep = 0
    t_all = time.perf_counter()

    for ep in range(1, args.epochs + 1):
        order = mem_ids[:]
        rng.shuffle(order)
        ep_loss = 0.0
        ep_n = 0
        t0 = time.perf_counter()

        for s in range(steps_per_epoch):
            chunk = order[s * args.batch:(s + 1) * args.batch]
            if len(chunk) < 8:
                continue
            # 每条记忆随机取一个查询（多查询 -> 每轮看到的问法不同）
            batch_rows = [rng.choice(by_mem[i]) for i in chunk]

            qe, me = make_batch(batch_rows, tok, device, args.max_len)
            with torch.amp.autocast("cuda", enabled=use_amp):
                qv = model(qe["input_ids"], qe["attention_mask"])
                mv = model(me["input_ids"], me["attention_mask"])
                # InfoNCE：对角线是正样本，其余是批内负样本
                logits = (qv @ mv.T) / args.temp
                labels = torch.arange(len(chunk), device=device)
                loss = F.cross_entropy(logits, labels)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            gstep += 1
            ep_loss += float(loss.item())
            ep_n += 1

        avg = ep_loss / max(1, ep_n)
        el = time.perf_counter() - t0

        # 训练集内的「批内 top-1 命中率」——正样本是否被排在第一
        with torch.inference_mode():
            model.eval()
            hits = 0
            tot = 0
            for s in range(0, min(len(order), 64 * 4), 64):
                chunk = order[s:s + 64]
                if len(chunk) < 8:
                    continue
                br = [rng.choice(by_mem[i]) for i in chunk]
                qe, me = make_batch(br, tok, device, args.max_len)
                qv = model(qe["input_ids"], qe["attention_mask"])
                mv = model(me["input_ids"], me["attention_mask"])
                pred = (qv @ mv.T).argmax(dim=1)
                hits += int((pred == torch.arange(len(chunk), device=device)).sum().item())
                tot += len(chunk)
            model.train()
        acc = hits / max(1, tot)

        print(f"[ep{ep}/{args.epochs}] loss={avg:.4f}  批内top1={acc:.1%}  "
              f"lr={sched.get_last_lr()[0]:.2e}  {el:.1f}s", flush=True)

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": ep, "loss": round(avg, 5),
                                "batch_top1": round(acc, 4), "sec": round(el, 1),
                                "ts": time.strftime("%H:%M:%S")}, ensure_ascii=False) + "\n")

        if ep % args.save_every == 0:
            model.encoder.save_pretrained(str(out_dir))
            tok.save_pretrained(str(out_dir))
            print(f"        [存盘] {out_dir}", flush=True)

    model.encoder.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"\n[完成] {args.epochs} 轮,{(time.perf_counter() - t_all) / 60:.1f} 分钟")
    print(f"[产物] {out_dir}")
    print(f"\n下一步：python training/eval.py --run   # 在独立评测集上重新评测")


if __name__ == "__main__":
    main()
