"""记忆检索评测 —— 先有度量，再谈优化。

为什么要先做这个
----------------
上次训练优化时我「先跑训练、后定义指标」，白跑几十轮才发现指标选错。
这次反过来：先把「检索好不好」量化出来，再决定改什么。

评测集怎么来（全自动，零人工标注）
----------------------------------
对每条记忆，让 Qwen 读它、生成一个**本该召回它**的自然查询：

    记忆：「用户偏好 PowerShell 里用 python -m pip ...」
    ↓ Qwen
    查询：「pip 安装命令怎么写」

然后测：用这个查询去检索，目标记忆能不能排进前 K 名。
指标：Recall@1 / Recall@5 / MRR。

**查询是合成的，所以绝对分数不代表真实场景，但用来横向比较两种方案足够。**

两种待比方案
------------
A. 基线：记忆插件现有的纯词面打分（bm25 + 词重叠 + 标签命中）
B. 语义：用训练过的 encoder 算句向量余弦相似度

用法
----
    python src/mem_eval.py --build          # 生成评测集（调 Qwen）
    python src/mem_eval.py --run            # 跑评测对比
    python src/mem_eval.py --build --run --n 60
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

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
MEM_JSON = MEMORY_JSON
EVAL_PATH = WORK / "eval_set.json"
LLM_DIR = LLM_MODEL
ENC_BASE = BASE_MODEL
ENC_RETRIEVER = WORK / "model"


# ---------------------------------------------------------------------- #
# 读记忆
# ---------------------------------------------------------------------- #


def load_memories() -> list[dict]:
    d = json.loads(MEM_JSON.read_text(encoding="utf-8"))
    table = d["tables"]["records"]
    rows = list(table.values()) if isinstance(table, dict) else table
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if r.get("expiresAt"):          # 跳过已过期的
            continue
        c = (r.get("content") or "").strip()
        if len(c) < 20:
            continue
        out.append(r)
    return out


# ---------------------------------------------------------------------- #
# 评测集：让 Qwen 为每条记忆生成一个「本该召回它」的查询
# ---------------------------------------------------------------------- #


def build_eval_set(n: int, seed: int = 0) -> list[dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = random.Random(seed)
    mems = load_memories()
    rng.shuffle(mems)
    mems = mems[:n]
    print(f"[生成] 从 {len(load_memories())} 条记忆里采样 {len(mems)} 条", flush=True)

    tok = AutoTokenizer.from_pretrained(str(LLM_DIR))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(str(LLM_DIR))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        model = model.half()
    model = model.to(dev).eval()

    SYS = (
        "你是一个记忆检索系统的测试用例生成器。"
        "给定一条记忆，写出一个用户可能会问的、自然简短的问题——"
        "这个问题应该能检索出这条记忆。"
        "只输出问题本身，不要解释，不要引号，不超过 25 个字。"
    )
    out = []
    B = 8
    for i in range(0, len(mems), B):
        batch = mems[i:i + B]
        prompts = []
        for m in batch:
            content = (m["content"] or "")[:400]
            prompts.append(
                tok.apply_chat_template(
                    [
                        {"role": "system", "content": SYS},
                        {"role": "user", "content": f"记忆内容：\n{content}"},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            gen = model.generate(
                **enc, max_new_tokens=40, do_sample=True, temperature=0.9,
                top_p=0.9, repetition_penalty=1.1, pad_token_id=tok.pad_token_id,
            )
        for m, g in zip(batch, gen):
            q = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            q = q.strip('"“”').split("\n")[0][:40]
            if len(q) >= 4:
                out.append({"id": m["id"], "query": q})
        print(f"  {len(out)}/{len(mems)}", flush=True)

    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[落盘] {EVAL_PATH}  ({len(out)} 条)")
    return out


# ---------------------------------------------------------------------- #
# 两种打分
# ---------------------------------------------------------------------- #


def tokenize_bigram(text: str) -> list[str]:
    """复刻插件 search.ts 里的 bigram 分词（中文双字 + ascii 词）。"""
    lower = text.lower()
    toks = re.findall(r"[a-z0-9_]+", lower)
    for run in re.findall(r"[\u4e00-\u9fff]+", lower):
        if len(run) == 1:
            toks.append(run)
        else:
            for i in range(len(run) - 1):
                toks.append(run[i:i + 2])
    return toks


def bm25_signal(query: str, content: str, avg_len: float = 40.0) -> float:
    qt = list(set(tokenize_bigram(query)))
    if not qt:
        return 0.0
    ct = tokenize_bigram(content)
    clen = max(1, len(ct))
    freq: dict[str, int] = {}
    for t in ct:
        freq[t] = freq.get(t, 0) + 1
    k1, b = 1.2, 0.75
    s = 0.0
    for t in qt:
        tf = freq.get(t, 0)
        if tf == 0:
            continue
        norm = clen / avg_len
        s += (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * norm))
    return min(2.0, s * 0.4)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def score_lexical(query: str, mem: dict) -> float:
    """基线：复刻插件的词面打分（去掉时效/重要性的乘积，做纯相关性对比）。"""
    q = query.strip().lower()
    if not q:
        return 0.0
    content = (mem.get("content") or "").lower()
    base = 0.0
    if q in content:
        base += 3.0
    for tag in (mem.get("tags") or []):
        if q in str(tag).lower():
            base += 1.5
            break
    for w in set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-z][a-z0-9_]{2,}", q)):
        if w in content:
            base += 1.2
    base += jaccard(set(tokenize_bigram(query)), set(tokenize_bigram(content))) * 2
    qc = set(re.findall(r"[\u4e00-\u9fff]", query))
    cc = set(re.findall(r"[\u4e00-\u9fff]", content))
    base += jaccard(qc, cc) * 0.8
    base += bm25_signal(query, content)
    return base


class SemanticScorer:
    """用 encoder 算句向量余弦相似度。

    池化用 mean pooling（比取 [CLS] 稳，因为这不是 sentence-transformer）。
    向量做 L2 归一化后点积即余弦。
    """

    def __init__(self, path: Path, device: str):
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModel.from_pretrained(str(path))
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def encode(self, texts: list[str], batch: int = 16) -> torch.Tensor:
        outs = []
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            enc = self.tok(
                chunk, return_tensors="pt", padding=True,
                truncation=True, max_length=256,
            ).to(self.device)
            h = self.model(**enc).last_hidden_state          # [B, L, H]
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (h * mask).sum(1)
            cnt = mask.sum(1).clamp(min=1e-6)
            vec = summed / cnt
            vec = torch.nn.functional.normalize(vec, dim=-1)
            outs.append(vec.float().cpu())
        return torch.cat(outs, 0) if outs else torch.zeros(0, 768)


# ---------------------------------------------------------------------- #
# 评测
# ---------------------------------------------------------------------- #


def evaluate(scorer_name: str, score_fn, mems: list[dict], eval_set: list[dict], topk=(1, 5, 10)):
    id2idx = {m["id"]: i for i, m in enumerate(mems)}
    ranks = []
    t0 = time.perf_counter()
    for item in eval_set:
        q, target = item["query"], item["id"]
        if target not in id2idx:
            continue
        scores = score_fn(q, mems)
        order = sorted(range(len(mems)), key=lambda i: -scores[i])
        try:
            rank = order.index(id2idx[target]) + 1
        except ValueError:
            rank = len(mems)
        ranks.append(rank)
    el = time.perf_counter() - t0
    n = len(ranks)
    if n == 0:
        return {}
    res = {"n": n, "mrr": round(sum(1.0 / r for r in ranks) / n, 4), "sec": round(el, 1)}
    for k in topk:
        res[f"R@{k}"] = round(sum(1 for r in ranks if r <= k) / n, 4)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="记忆检索评测")
    ap.add_argument("--build", action="store_true", help="生成评测集")
    ap.add_argument("--run", action="store_true", help="跑评测")
    ap.add_argument("--n", type=int, default=50, help="评测集条数")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(WORK))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    mems = load_memories()
    print(f"[数据] 记忆 {len(mems)} 条")

    # ---- 构建评测集 ----
    if args.build or not EVAL_PATH.exists():
        eval_set = build_eval_set(args.n, args.seed)
    else:
        eval_set = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
        print(f"[数据] 评测集 {len(eval_set)} 条（已有）")

    if not args.run:
        print("\n加 --run 跑评测")
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'=' * 70}\n检索评测   设备={dev}\n{'=' * 70}")

    # 方案 A：词面
    print("\n[A] 词面基线 ...", flush=True)

    def fn_lex(q, ms):
        return [score_lexical(q, m) for m in ms]

    rA = evaluate("lexical", fn_lex, mems, eval_set)
    print(f"    {rA}")

    # 方案 B：语义
    print("\n[B] 语义（encoder 句向量）...", flush=True)
    best = None
    candidates = [
        ("检索专用 mem-retriever", ENC_RETRIEVER),
        ("原版 chinese-roberta", ENC_BASE),
    ]
    results_all = {}
    for label, path in candidates:
        if not (path / "config.json").exists():
            print(f"    [跳过] {label} 不存在")
            continue
        try:
            sc = SemanticScorer(path, dev)
            mem_vecs = sc.encode([(m.get("content") or "")[:400] for m in mems])
            q_vecs = sc.encode([e["query"] for e in eval_set])

            id2idx = {m["id"]: i for i, m in enumerate(mems)}
            ranks = []
            t0 = time.perf_counter()
            for i, e in enumerate(eval_set):
                sims = (q_vecs[i:i + 1] @ mem_vecs.T).squeeze(0)
                order = torch.argsort(sims, descending=True).tolist()
                try:
                    ranks.append(order.index(id2idx[e["id"]]) + 1)
                except ValueError:
                    ranks.append(len(mems))
            el = time.perf_counter() - t0
            n = len(ranks)
            rB = {
                "n": n,
                "mrr": round(sum(1.0 / r for r in ranks) / n, 4),
                "R@1": round(sum(1 for r in ranks if r <= 1) / n, 4),
                "R@5": round(sum(1 for r in ranks if r <= 5) / n, 4),
                "R@10": round(sum(1 for r in ranks if r <= 10) / n, 4),
                "sec": round(el, 3),
            }
            print(f"    {label}: {rB}")
            results_all[label] = rB
            if best is None or rB["mrr"] > best[1]["mrr"]:
                best = (label, rB)
        except Exception as e:  # noqa: BLE001
            print(f"    [失败] {label}: {e}")

    # ---- 对比 ----
    print(f"\n{'=' * 70}\n对比\n{'=' * 70}")
    print(f"  {'方案':<28} {'MRR':>8} {'R@1':>8} {'R@5':>8} {'R@10':>8}")
    print(f"  {'A 纯词面基线':<28} {rA.get('mrr',0):>8} {rA.get('R@1',0):>8} {rA.get('R@5',0):>8} {rA.get('R@10',0):>8}")
    for label, r in results_all.items():
        print(f"  {'B ' + label:<28} {r.get('mrr',0):>8} {r.get('R@1',0):>8} {r.get('R@5',0):>8} {r.get('R@10',0):>8}")
    if best:
        d = best[1].get("mrr", 0) - rA.get("mrr", 0)
        print(f"\n  最好的语义方案: {best[0]}")
        print(f"  相对词面基线的 MRR 变化: {d:+.4f}  "
              f"({'语义更好' if d > 0 else '语义更差' if d < 0 else '持平'})")

    (out_dir / "eval_result.json").write_text(
        json.dumps({"lexical": rA, "semantic": results_all, "best": best},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[落盘] {out_dir / 'eval_result.json'}")


if __name__ == "__main__":
    main()
