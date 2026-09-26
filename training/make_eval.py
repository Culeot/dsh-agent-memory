"""评测查询集 —— 单独造,因为它必须是「模型没见过的东西」。

为什么评测集不能从训练数据里拿
------------------------------
训练对 `train_pairs.jsonl` 是给模型学的,拿它评测等于自己判自己的卷子。
所以这里重新生成一批查询,区别有三点:

  1. 换种子。生成查询的随机过程与训练对不同,不会撞出同一批串;
  2. 覆盖全部记忆,**包括 holdout_ids.json 里留出来的那批**——
     那批记忆模型从没见过,在它们上面的分数才反映泛化能力;
  3. 只留 id/query/qtype 三个字段,eval.py 直接吃。

诚实说清局限
------------
这批查询仍然是模型生成的,分布上比真实用户查询窄。最可信的评测集是从
真实会话里挖出来的查询(本项目当初从 271 个会话里挖了 157 条)。如果你有
这样的日志,优先用它们;没有,用这里生成的至少能横向比较两种方案。

分片产出 `queries_part1.jsonl ... queries_partN.jsonl`,与 eval.py 的读取约定一致。

用法
----
    python training/make_eval.py                     # 关键词轨 + LLM 轨,每条记忆 2 条查询
    python training/make_eval.py --kw-only           # 零 GPU,秒级
    python training/make_eval.py --llm-per 2         # 多造几条,评测更稳
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gen_data import keyword_pool, make_kw_queries, run_llm_track  # noqa: E402

# --- 路径:全部从环境变量读,脚本放哪都能跑 --------------------------------- #
import os

WORK = Path(os.environ.get("MEM_TRAIN_WORK", "work"))
PART_DIR = WORK / "parts"

N_PARTS = 4
# 与 gen_data.py 的默认种子刻意错开:两批查询不该是同一批
EVAL_SEED = 20261001


def main() -> None:
    ap = argparse.ArgumentParser(description="生成评测查询集(与训练数据分开)")
    ap.add_argument("--kw-per", type=int, default=1, help="每条记忆造几条关键词式查询")
    ap.add_argument("--llm-per", type=int, default=1, help="每条记忆造几条 LLM 查询")
    ap.add_argument("--batch-mems", type=int, default=8)
    ap.add_argument("--parts", type=int, default=N_PARTS)
    ap.add_argument("--seed", type=int, default=EVAL_SEED)
    ap.add_argument("--kw-only", action="store_true", help="只造关键词轨(零 GPU)")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    mems: dict[str, dict] = {}
    for p in sorted(PART_DIR.glob("part*.json")):
        for m in json.loads(p.read_text(encoding="utf-8")):
            mems[m["id"]] = m
    if not mems:
        print(f"[错误] 没有记忆分片 {PART_DIR}/part*.json,先跑 prep.py")
        sys.exit(1)

    all_mems = sorted(mems.values(), key=lambda m: m["id"])
    hold_path = WORK / "holdout_ids.json"
    hold = set(json.loads(hold_path.read_text(encoding="utf-8"))) if hold_path.exists() else set()
    print(f"[数据] 记忆 {len(all_mems)} 条(其中 held-out {len(hold)} 条 —— 这批是泛化的关键)")

    rows: list[dict] = []

    # ---- kw 轨 ----
    for mem in all_mems:
        pool = keyword_pool(mem)
        for q in make_kw_queries(mem, pool, args.kw_per, rng):
            rows.append({"id": mem["id"], "query": q, "qtype": "kw"})
    print(f"[kw 轨] {len(rows)} 条")

    # ---- LLM 轨 ----
    if not args.kw_only:
        llm_rows = run_llm_track(all_mems, args.llm_per, args.batch_mems, rng)
        for r in llm_rows:
            rows.append({"id": r["id"], "query": r["query"], "qtype": r["qtype"]})

    # ---- 去重 + 丢掉空查询 ----
    uniq: list[dict] = []
    seenq: set[tuple[str, str]] = set()
    for r in rows:
        q = (r["query"] or "").strip()
        if len(q) < 4:
            continue
        k = (r["id"], q)
        if k in seenq:
            continue
        seenq.add(k)
        uniq.append({"id": r["id"], "query": q, "qtype": r["qtype"]})
    rng.shuffle(uniq)

    # ---- 分片落盘(轮询分配,各片条数尽量均匀)----
    shards: list[list[dict]] = [[] for _ in range(args.parts)]
    for i, r in enumerate(uniq):
        shards[i % args.parts].append(r)
    for i, shard in enumerate(shards, 1):
        p = WORK / f"queries_part{i}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in shard:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"        {p}  {len(shard)} 条")

    dist = Counter(r["qtype"] for r in uniq)
    n_hold = sum(1 for r in uniq if r["id"] in hold)
    print(f"\n[完成] {len(uniq)} 条查询 → {WORK}/queries_part*.jsonl")
    print(f"[分布] " + "  ".join(f"{k}={v}" for k, v in sorted(dist.items())))
    print(f"[覆盖] 命中 held-out 记忆的查询 {n_hold} 条")
    print(f"[示例]")
    for r in uniq[:5]:
        tag = "held-out" if r["id"] in hold else "seen"
        print(f"    [{r['qtype']:<6} {tag}] {r['query']}")
    print(f"\n下一步:python training/eval.py --run")


if __name__ == "__main__":
    main()
