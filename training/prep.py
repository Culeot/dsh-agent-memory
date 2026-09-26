"""评测集 v2 准备 —— 把记忆稳定切分,供并行生成「异源」评测查询。

为什么要 v2(不是小题大做)
--------------------------
v1 评测集(out/mem_eval/eval_set.json,50 条)有两个致命缺陷:

1. **与训练数据同源** —— 查询都是同一个 Qwen2.5-0.5B 生成的,连提示词角度都一样,
   记忆还重叠。测出来的分数很大程度上是「模型记住了这套生成器的话术」。
2. **查询质量差** —— 里面混进了记忆片段而不是提问,例如:
       「重启脚本已写」「请在图片路径或 URL 内输入图片名,然后在…
   这些不是用户会问的问题,拿它们当测试集,等于用一把不准的尺子量东西。

最关键的一条:50 条样本上,MRR 每条的贡献在 0~1 之间(标准差约 0.3),
标准误 ≈ 0.3/sqrt(50) ≈ 0.042。而 v1 报告的差距只有 0.034 ——
**小于标准误,统计上不显著**。一条样本翻面,结论就能反向。
所以「训练有效」目前只能标为「未证实」。

v2 的设计
---------
- 覆盖**全部**有效记忆(380 条),而不是抽样 50 条 → 1 条 ≈ 0.26%,差距才可信
- 查询由**强模型人工口吻生成**(不是 0.5B 小模型),与训练数据异源
- 切分成分片文件,便于多个 agent 并行生成,互不干扰、结果可拼接
- 额外划一份 **held-out 记忆**(20%,固定 seed):这部分记忆完全不参与训练,
  专门用来测「模型面对没见过的新记忆能不能检索」——真实场景里记忆是不断新增的,
  这个能力比在见过的记忆上刷分重要得多

产物
----
    out/mem_eval/eval_v2/parts/part1.json ... part4.json   (每片约 95 条记忆)
    out/mem_eval/eval_v2/holdout_ids.json                  (75 条 held-out 记忆 id)
    out/mem_eval/eval_v2/prep_report.json                  (统计)

用法
----
    python training/prep.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# --- 路径:全部从环境变量读,脚本放哪都能跑 --------------------------------- #
import os

WORK = Path(os.environ.get("MEM_TRAIN_WORK", "work"))
MEMORY_JSON = Path(os.environ.get("MEM_TRAIN_MEMORY",
                                  str(Path.home() / ".dsh" / "storages" / "memory.json")))
BASE_MODEL = Path(os.environ.get("MEM_TRAIN_BASE", "models/chinese-roberta-wwm-ext"))
LLM_MODEL = Path(os.environ.get("MEM_TRAIN_LLM", "models/Qwen2.5-0.5B-Instruct"))
MEM_JSON = MEMORY_JSON
OUT_DIR = WORK
PART_DIR = WORK / "parts"

N_PARTS = 4
HOLDOUT_RATIO = 0.2
HOLDOUT_SEED = 20260925


def load_memories() -> list[dict]:
    """读全部有效记忆(未过期、内容 ≥ 20 字),按 id 排序保证可复现。"""
    d = json.loads(MEM_JSON.read_text(encoding="utf-8"))
    table = d["tables"]["records"]
    rows = list(table.values()) if isinstance(table, dict) else table
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if r.get("expiresAt"):
            continue
        content = (r.get("content") or "").strip()
        if len(content) < 20:
            continue
        out.append(
            {
                "id": r["id"],
                "content": content,
                "tags": r.get("tags") or [],
                "kind": r.get("kind") or "note",
                "importance": r.get("importance") or 1,
            }
        )
    out.sort(key=lambda m: m["id"])
    return out


def main() -> None:
    PART_DIR.mkdir(parents=True, exist_ok=True)

    mems = load_memories()
    print(f"[数据] 有效记忆 {len(mems)} 条")

    # ---- held-out 记忆(固定 seed,任何时候重跑结果一致)----
    rng = random.Random(HOLDOUT_SEED)
    ids = [m["id"] for m in mems]
    n_hold = max(1, round(len(mems) * HOLDOUT_RATIO))
    holdout = sorted(rng.sample(ids, n_hold))
    (OUT_DIR / "holdout_ids.json").write_text(
        json.dumps(holdout, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"[held-out] {len(holdout)} 条记忆不参与训练(seed={HOLDOUT_SEED})")

    # ---- 分片(全部记忆,供并行生成查询)----
    per = (len(mems) + N_PARTS - 1) // N_PARTS
    stats = []
    for i in range(N_PARTS):
        chunk = mems[i * per:(i + 1) * per]
        if not chunk:
            continue
        p = PART_DIR / f"part{i + 1}.json"
        p.write_text(json.dumps(chunk, ensure_ascii=False, indent=1), encoding="utf-8")
        avg = sum(len(m["content"]) for m in chunk) // max(1, len(chunk))
        stats.append({"part": i + 1, "n": len(chunk), "avg_chars": avg, "path": str(p)})
        print(f"[分片] part{i + 1}: {len(chunk)} 条  平均 {avg} 字  → {p.name}")

    (OUT_DIR / "prep_report.json").write_text(
        json.dumps(
            {
                "total": len(mems),
                "n_parts": len(stats),
                "holdout": len(holdout),
                "holdout_seed": HOLDOUT_SEED,
                "parts": stats,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[完成] 产物目录 {OUT_DIR}")
    print("下一步:python training/make_eval.py --kw-only   # 生成评测查询 queries_partN.jsonl")


if __name__ == "__main__":
    sys.exit(main())
