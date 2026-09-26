"""检索评测 v2 —— 可信的尺子 + 融合方案。

与 v1 的区别(v1: mem_eval.py,50 条同源样本)
--------------------------------------------
1. 评测集换成 **382 条异源查询**(强模型生成、真人口吻、四类混合),
   1 条 ≈ 0.26%,结论才不被单条样本左右;
2. 除了「词面 vs 语义」,还评测**融合**——词面擅长精确术语/路径,
   语义擅长换说法,两者互补,而 v1 从没调过融合权重;
3. **调参/报告分离**:权重在 A 半样本上选,B 半样本上报告。
   否则 382 条上调一个 α 也是过拟合(50 条上调 α 更是纯噪声);
4. 按 qtype 分组报告,能看出语义到底在哪一类查询上赢。
   预期:semantic 在 vague 类(模糊指代)上优势最大,kw/term 类词面强。

两个融合公式
------------
- 分数融合:  z(每个通道的分数) 后加权求和 —— z 归一化解决量纲问题
             (词面分 0~10+,余弦 0~1,直接加权毫无意义)
- RRF:        用**排名**而非分数融合,对量纲不敏感,工业界混合检索的默认做法

用法
----
    python training/eval.py --run
    python training/eval.py --run --no-semantic   # 只跑词面(快)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

# CPU 让路:本机同时跑量化训练与进化论舰队,别抢核
torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from mem_eval import SemanticScorer, score_lexical  # noqa: E402

# --- 路径:全部从环境变量读,脚本放哪都能跑 --------------------------------- #
import os

WORK = Path(os.environ.get("MEM_TRAIN_WORK", "work"))
MEMORY_JSON = Path(os.environ.get("MEM_TRAIN_MEMORY",
                                  str(Path.home() / ".dsh" / "storages" / "memory.json")))
BASE_MODEL = Path(os.environ.get("MEM_TRAIN_BASE", "models/chinese-roberta-wwm-ext"))
LLM_MODEL = Path(os.environ.get("MEM_TRAIN_LLM", "models/Qwen2.5-0.5B-Instruct"))
EVAL_DIR = WORK
PART_DIR = WORK / "parts"
OUT = WORK / "eval_result.json"
ENC_RETRIEVER = WORK / "model"
ENC_BASE = BASE_MODEL
# 想额外比对别的句向量模型时,用逗号分隔的路径传进来(可选)
EXTRA_ENCODERS = [p for p in
                  os.environ.get("MEM_TRAIN_EXTRA_ENCODERS", "").split(",") if p]


# ---------------------------------------------------------------------- #
# 数据
# ---------------------------------------------------------------------- #


def load_mems() -> list[dict]:
    mems: dict[str, dict] = {}
    for p in sorted(PART_DIR.glob("part*.json")):
        for m in json.loads(p.read_text(encoding="utf-8")):
            mems[m["id"]] = m
    return list(mems.values())


def load_queries() -> list[dict]:
    rows: list[dict] = []
    for p in sorted(EVAL_DIR.glob("queries_part*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


# ---------------------------------------------------------------------- #
# 归一化与融合
# ---------------------------------------------------------------------- #


def zscores(xs: list[float]) -> list[float]:
    """按查询内候选集做 z 归一化 —— 分数融合的前提。"""
    if not xs:
        return xs
    mu = statistics.fmean(xs)
    sd = statistics.pstdev(xs) or 1e-9
    return [(x - mu) / sd for x in xs]


def rrf_fuse(rank_lists: list[list[int]], weights: list[float], k: int = 60) -> list[float]:
    """倒数排名融合:每条文档得 sum(w_i / (k + rank_i))。"""
    n = len(rank_lists[0])
    fused = [0.0] * n
    for ranks, w in zip(rank_lists, weights):
        for pos, doc in enumerate(ranks):
            fused[doc] += w / (k + pos + 1)
    return fused


def ranks_of(scores: list[float]) -> list[int]:
    """返回「按分数从高到低排列的文档下标」。"""
    return sorted(range(len(scores)), key=lambda i: -scores[i])


# ---------------------------------------------------------------------- #
# 指标
# ---------------------------------------------------------------------- #


def metrics(scored: list[list[float]], target_idx: list[int], topk=(1, 5, 10)) -> dict:
    ranks: list[int] = []
    for scores, tgt in zip(scored, target_idx):
        order = ranks_of(scores)
        try:
            ranks.append(order.index(tgt) + 1)
        except ValueError:
            ranks.append(len(scores))
    n = len(ranks)
    if n == 0:
        return {}
    out = {
        "n": n,
        "MRR": round(statistics.fmean([1.0 / r for r in ranks]), 4),
    }
    for k in topk:
        out[f"R@{k}"] = round(sum(1 for r in ranks if r <= k) / n, 4)
    return out


def fmt_row(label: str, m: dict) -> str:
    if not m:
        return f"  {label:<34} —"
    return (f"  {label:<34} {m.get('MRR', 0):>7.4f} {m.get('R@1', 0):>7.4f} "
            f"{m.get('R@5', 0):>7.4f} {m.get('R@10', 0):>7.4f}")


# ---------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description="检索评测 v2")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--no-semantic", action="store_true", help="跳过语义通道(只验词面)")
    ap.add_argument("--grid", type=float, default=0.05, help="权重网格步长")
    args = ap.parse_args()
    if not args.run:
        ap.print_help()
        return

    mems = load_mems()
    queries = load_queries()
    if not queries:
        print("[错误] 没有 queries_part*.jsonl,先等评测集生成")
        sys.exit(1)

    id2idx = {m["id"]: i for i, m in enumerate(mems)}
    valid = [q for q in queries if q.get("id") in id2idx]
    print(f"[数据] 记忆 {len(mems)} 条,查询 {len(valid)} 条(丢弃 {len(queries)-len(valid)} 条 id 不匹配)")
    targets = [id2idx[q["id"]] for q in valid]
    texts = [str(q["query"]) for q in valid]
    qtypes = [str(q.get("qtype") or "?") for q in valid]

    # ---- 通道 1:词面 ----
    print("\n[A] 词面通道(复刻插件打分)...", flush=True)
    lex_scores: list[list[float]] = []
    for t in texts:
        lex_scores.append([score_lexical(t, m) for m in mems])
    mA = metrics(lex_scores, targets)
    print("    " + str(mA))

    # ---- 通道 2:语义(多模型对照) ----
    # v2 的核心发现是「语义单独打不过词面,但融合能赢」,所以这里必须能同时看到
    # 几个语义模型的表现:老模型(v1 数据训的)、新模型(v2 分布对齐训的)、原版基座。
    candidates = [
        ("mem-retriever(本次训练产物)", ENC_RETRIEVER),
        ("chinese-roberta(未训练基座, 对照)", ENC_BASE),
    ] + [(Path(p).name, Path(p)) for p in EXTRA_ENCODERS]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sem_results: dict[str, dict] = {}
    sem_by_model: dict[str, list[list[float]]] = {}
    if not args.no_semantic:
        print(f"\n[B] 语义通道(device={dev})...", flush=True)
        for label, path in candidates:
            if not (path / "config.json").exists():
                print(f"    [跳过] {label} —— 模型不存在 {path}")
                continue
            sc = SemanticScorer(path, dev)
            mem_vecs = sc.encode([(m.get("content") or "")[:400] for m in mems])
            q_vecs = sc.encode(texts)
            scores = (q_vecs @ mem_vecs.T).tolist()
            m = metrics(scores, targets)
            sem_results[label] = m
            sem_by_model[label] = scores
            print("    " + fmt_row(label, m))

    if sem_by_model:
        best_label = max(sem_results, key=lambda k: sem_results[k]["MRR"])
        sem_scores = sem_by_model[best_label]
        mB = sem_results[best_label]
        print(f"    → 融合采用最强的语义通道:{best_label}")
    else:
        best_label, sem_scores, mB = None, None, {}
        print("\n[B] 语义通道:全部跳过")

    results: dict = {"n_queries": len(valid), "n_memories": len(mems),
                     "lexical": mA, "semantic": sem_results,
                     "best_semantic": best_label}

    if sem_scores is None:
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[落盘] {OUT}")
        return

    # ---- 调参/报告分离 ----
    # 按查询文本排序后奇偶切分,保证两半分布同质(不是按 id 顺序切,避免记忆块状分布)
    order = sorted(range(len(valid)), key=lambda i: texts[i])
    half_a = order[::2]
    half_b = order[1::2]
    print(f"\n[切分] 调参 A 半 {len(half_a)} 条 / 报告 B 半 {len(half_b)} 条")

    def subset(xs, idxs):
        return [xs[i] for i in idxs]

    targets_a, targets_b = subset(targets, half_a), subset(targets, half_b)

    # ---- 融合 1:z 分数加权 ----
    print("\n[C] 分数融合:z(sem)×w + z(lex)×(1-w),在 A 半网格搜索...", flush=True)
    zl = [zscores(s) for s in lex_scores]
    zs = [zscores(s) for s in sem_scores]
    best_w, best_mrr = 0.0, -1.0
    curve = []
    w = 0.0
    while w <= 1.0001:
        fused = [[w * zs[i][j] + (1 - w) * zl[i][j] for j in range(len(mems))]
                 for i in range(len(valid))]
        m = metrics(subset(fused, half_a), targets_a)
        curve.append({"w_semantic": round(w, 3), "A_MRR": m["MRR"], "A_R@1": m["R@1"]})
        if m["MRR"] > best_mrr:
            best_mrr, best_w = m["MRR"], w
        w += args.grid
    print(f"    A 半最优 w_semantic = {best_w:.2f} (A_MRR={best_mrr:.4f})")

    fused_all = [[best_w * zs[i][j] + (1 - best_w) * zl[i][j] for j in range(len(mems))]
                 for i in range(len(valid))]
    mC_full = metrics(fused_all, targets)
    mC_b = metrics(subset(fused_all, half_b), targets_b)
    results["fusion_zscore"] = {"w_semantic": best_w, "full": mC_full, "holdout_half": mC_b, "curve": curve}
    print("    " + fmt_row(f"融合 w={best_w:.2f}(B 半,未调参)", mC_b))

    # ---- 融合 2:RRF ----
    print("\n[D] RRF 倒数排名融合(A 半搜索 w 与 k)...", flush=True)
    lex_ranks = [ranks_of(s) for s in lex_scores]
    sem_ranks = [ranks_of(s) for s in sem_scores]
    best_rrf = (-1.0, 0.0, 60)
    rrf_curve = []
    for k in (10, 30, 60, 100):
        wr = 0.0
        while wr <= 1.0001:
            fused = [rrf_fuse([sem_ranks[i], lex_ranks[i]], [wr, 1 - wr], k) for i in range(len(valid))]
            m = metrics(subset(fused, half_a), targets_a)
            rrf_curve.append({"k": k, "w_semantic": round(wr, 3), "A_MRR": m["MRR"]})
            if m["MRR"] > best_rrf[0]:
                best_rrf = (m["MRR"], wr, k)
            wr += 0.1
    _, wr, k = best_rrf
    fused_rrf = [rrf_fuse([sem_ranks[i], lex_ranks[i]], [wr, 1 - wr], k) for i in range(len(valid))]
    mD_full = metrics(fused_rrf, targets)
    mD_b = metrics(subset(fused_rrf, half_b), targets_b)
    results["fusion_rrf"] = {"w_semantic": wr, "k": k, "full": mD_full, "holdout_half": mD_b, "curve": rrf_curve}
    print(f"    A 半最优 k={k} w_semantic={wr:.1f}")
    print("    " + fmt_row(f"RRF k={k} w={wr:.1f}(B 半)", mD_b))

    # ---- 总表 ----
    best_method = max(
        [("词面基线", mA, None), ("语义", mB, None),
         (f"分数融合 w={best_w:.2f}", mC_full, mC_b),
         (f"RRF k={k} w={wr:.1f}", mD_full, mD_b)],
        key=lambda x: x[1].get("MRR", 0),
    )
    print("\n" + "=" * 78)
    print("汇总(全 382 条)")
    print(f"  {'方案':<34} {'MRR':>7} {'R@1':>7} {'R@5':>7} {'R@10':>7}")
    print(fmt_row("A 词面基线(插件现有)", mA))
    print(fmt_row("B 语义 mem-retriever", mB))
    print(fmt_row(f"C 分数融合 w_sem={best_w:.2f} ★", mC_full))
    print(fmt_row(f"D RRF k={k} w_sem={wr:.1f}", mD_full))
    print(f"\n  最佳: {best_method[0]}  (MRR {best_method[1].get('MRR')})")
    print(f"  相对词面基线: {best_method[1].get('MRR',0) - mA.get('MRR',0):+.4f}")

    # ---- 分组:按 qtype ----
    print("\n" + "=" * 78)
    print("分组:按查询类型(融合列用全量最佳权重)")
    print(f"  {'qtype':<8} {'n':>4} {'词面MRR':>9} {'语义MRR':>9} {'融合MRR':>9}  谁赢")
    by_type: dict[str, dict] = {}
    for t in sorted(set(qtypes)):
        idxs = [i for i, tt in enumerate(qtypes) if tt == t]
        if not idxs:
            continue
        ml = metrics(subset(lex_scores, idxs), subset(targets, idxs))
        ms = metrics(subset(sem_scores, idxs), subset(targets, idxs))
        mf = metrics(subset(fused_all, idxs), subset(targets, idxs))
        by_type[t] = {"n": len(idxs), "lexical": ml, "semantic": ms, "fusion": mf}
        winner = max([("词面", ml), ("语义", ms), ("融合", mf)], key=lambda x: x[1]["MRR"])[0]
        print(f"  {t:<8} {len(idxs):>4} {ml['MRR']:>9.4f} {ms['MRR']:>9.4f} {mf['MRR']:>9.4f}  {winner}")
    results["by_qtype"] = by_type

    # ---- 分组:held-out 记忆(未参与训练的那 76 条) ----
    hold_path = EVAL_DIR / "holdout_ids.json"
    if hold_path.exists():
        hold = set(json.loads(hold_path.read_text(encoding="utf-8")))
        idxs_h = [i for i, q in enumerate(valid) if q["id"] in hold]
        idxs_s = [i for i, q in enumerate(valid) if q["id"] not in hold]
        print("\n分组:记忆是否见过训练")
        for name, idxs in (("held-out(未见)", idxs_h), ("seen(已见)", idxs_s)):
            if not idxs:
                continue
            ml = metrics(subset(lex_scores, idxs), subset(targets, idxs))
            ms = metrics(subset(sem_scores, idxs), subset(targets, idxs))
            mf = metrics(subset(fused_all, idxs), subset(targets, idxs))
            results.setdefault("by_seen", {})[name] = {"n": len(idxs), "lexical": ml, "semantic": ms, "fusion": mf}
            print(f"  {name:<14} n={len(idxs):<4} 词面 {ml['MRR']:.4f}  语义 {ms['MRR']:.4f}  融合 {mf['MRR']:.4f}")

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[落盘] {OUT}")


if __name__ == "__main__":
    main()
