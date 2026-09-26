"""训练数据 v2 —— 把分布对齐到真实查询。

为什么必须重造数据
------------------
v1 的 2154 个训练样本全部是 Qwen2.5-0.5B 生成的**问句式**查询:
    「agnes-ai 脚本在哪里找到?」
但真实查询(从 271 个会话里挖出的 157 条实测)是**关键词堆叠式**:
    「语音插件 语音播报 speak tts」
    「token 明文 命令 日志 泄漏 github」
    「edit 工具 缩进 匹配失败 yaml 配置 修改 preset」
平均 29.3 字,实词堆叠,中英混排,几乎不用虚词。

分布错了,训出来的模型就在真实场景里吃亏——这在 v2 评测集上已经看到:
语义通道对 kw 类只有 0.8533,而词面是 0.9505。

本脚本干什么
------------
双轨生成,按比例混合:

  kw 轨(模板化,不花 GPU,40%)
      从记忆里抽关键词池(tags + ASCII 词 + 高频中文短语),
      组合成 2~6 个词的检索词串——**模拟真实 agent 的查法**。
      为什么要模板化:让 0.5B 小模型"写成关键词式"它写不像,
      而关键词恰好是权重最大的那一类,不能交给它糊弄。

  LLM 轨(Qwen2.5-0.5B 生成,60%)
      问句 / 模糊指代 / 术语式。这几类需要语义理解,模板造不出来。
      模糊指代尤其重要:它逼模型学"查询里没有的词,靠意思找"。

数据纪律
--------
- 只用 **seen 记忆**(排除 holdout_ids.json 里的 76 条)→ held-out 才测得出泛化;
- 同一记忆的多条查询,训练时靠 `mem_train.py` 的批内去重避免假负样本,这里不动它;
- 输出 `out/mem_train/train_pairs_v2.jsonl`,字段与 v1 完全一致(id/query/content/tags),
  这样 `mem_train.py` 不用改就能直接吃。

用法
----
    python training/gen_data.py --kw-per 4 --llm-per 6
    python training/gen_data.py --kw-only      # 只造 kw 轨(零 GPU)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

# CPU 让路:量化训练与进化论常驻本机,别抢核
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
PART_DIR = WORK / "parts"
HOLDOUT = WORK / "holdout_ids.json"
OUT_PATH = WORK / "train_pairs.jsonl"
LLM_DIR = LLM_MODEL

# 中文停用词/虚词:关键词串里不该出现这些
STOP_CN = set("的了和与或是把被给向从到于对就都也还但而则且这那哪什么怎为何"
              "可以能好不别请先再又很太真挺会想要帮看说做弄搞整我你他她它咱您们"
              "一条这个那个因为所以如果但是然后现在已经一个没有我们他们以及")

# 英文停用词:从正文里捞 ASCII 词时,最常见的噪音就是这些普通词。
# 实测反例:"could 修改 〇 网页 保存 文件"、"server contents PowerShell" ——
# 这些词对检索没有任何区分力,却会占掉关键词槽位。
EN_STOP = set("""
the and for with this that from have been will would could should about into over after
before between during without within your their them then than there here what when where
which while who whom whose how why all any both each few more most other some such only
own same too very can just don now not are was were is be to of in on at by as it its if
or an we you they he she his her our out up down server content contents file files data
main source host body store label value name type code text string list item key user
agent model config settings default example python node windows linux error issue fixed
version update change support using used make made take taken give given need needs
full part case point line form side end start first last next new old good bad
""".split())


def good_ascii(word: str) -> bool:
    """这个 ASCII 词值不值得当检索词?

    实测噪音三类:①普通英文词(would/server);②纯编号与版本号(R73/e12/x80070570);
    ③过短词。术语的特征是:带分隔符、带数字、含驼峰大写、或足够长。
    """
    lw = word.lower()
    if lw in EN_STOP or len(lw) < 3:
        return False
    if re.fullmatch(r"[a-z]?\d+([.\-_]\d+)*", lw):      # 纯编号/版本
        return False
    if any(c in word for c in "-_."):                    # dsh-agent-memory / onnxruntime-node
        return True
    if any(c.isdigit() for c in word):                   # bge-m3 / utf-8
        return True
    if any(c.isupper() for c in word[1:]):               # PowerShell / InfoNCE
        return True
    return len(word) >= 7                                # 长词更像术语

ASCII_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-\.]{2,}")
CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")


# ---------------------------------------------------------------------- #
# 关键词池
# ---------------------------------------------------------------------- #


def keyword_pool(mem: dict) -> list[str]:
    """给一条记忆造关键词池:标签 + ASCII 术语 + 高频中文短语。"""
    content = mem.get("content") or ""
    pool: list[str] = []

    # 1) 标签:最像"检索词"的东西,优先
    for t in mem.get("tags") or []:
        t = str(t).strip()
        if 2 <= len(t) <= 16:
            pool.append(t)

    # 2) ASCII 术语
    for w in ASCII_RE.findall(content):
        if good_ascii(w):
            pool.append(w)

    # 3) 中文短语:只取 3~4 字的高频切片
    #    2 字切片看着能用,实则大量噪音(实测:「智慧 智慧收 智慧收敛」)。
    #    3~4 字更像一个真词,且要求出现≥2 次。
    cnt: Counter = Counter()
    for run in CJK_RUN_RE.findall(content):
        for n in (3, 4):
            for i in range(len(run) - n + 1):
                frag = run[i:i + n]
                if any(ch in STOP_CN for ch in frag):
                    continue
                cnt[frag] += 1
    for frag, c in cnt.most_common(30):
        if c >= 2:
            pool.append(frag)

    # 去重保序
    seen: set[str] = set()
    out: list[str] = []
    for w in pool:
        k = w.lower()
        if k not in seen:
            seen.add(k)
            out.append(w)

    # 去掉被别的片段包住的碎片(「智慧收」是「智慧收敛」的子串,留着只会制造噪音)
    out = [w for w in out if not any(w != o and w in o for o in out)]
    return out


def make_kw_queries(mem: dict, pool: list[str], n: int, rng: random.Random) -> list[str]:
    """把关键词池组合成"真实代理会打出的"检索词串。

    真实查询的形状:2~6 个实词、空格分隔、中英混排、顺序不讲究语义通顺。
    这里刻意**不**保证每个词都出现在同一条记忆的关键位置,允许少量"想当然的词",
    因为在真实场景里,人也会打错词——那正是模型该靠语义补上的地方。
    """
    if len(pool) < 2:
        return []
    out: list[str] = []
    for _ in range(n * 2):
        k = rng.randint(2, min(7, len(pool)))
        # 偏向池子前部(标签和术语更靠前)
        idxs = sorted(rng.sample(range(len(pool)), k), key=lambda i: i + rng.random() * 3)
        words = [pool[i] for i in idxs]
        q = " ".join(words).strip()
        if 4 <= len(q) <= 40:
            out.append(q)
    # 去重、取前 n 条
    uniq: list[str] = []
    for q in out:
        if q not in uniq:
            uniq.append(q)
    return uniq[:n]


# ---------------------------------------------------------------------- #
# LLM 轨
# ---------------------------------------------------------------------- #

ANGLES = [
    ("q", "用一句很短的口语提问(12 字以内,带问号)"),
    ("q", "像跟助手随口确认一样问一句"),
    ("vague", "用模糊指代问同一件事——**不要出现上面记忆里的任何专有名词、文件名、路径、版本号**"),
    ("vague", "假设你忘了具体名字,只能凭印象问"),
    ("term", "只用该领域的技术术语组合成检索式(不要问句、不要虚词)"),
    ("q", "从「怎么操作」的角度问一句"),
]


def build_prompts(tok, mem: dict, angles: list[tuple[str, str]]) -> list[tuple[str, str]]:
    content = (mem.get("content") or "")[:400]
    tags = ", ".join(mem.get("tags") or [])
    SYS = (
        "你在为记忆检索系统造测试查询。给定一条记忆,写出一个用户可能会问的问题——"
        "这个问题应该能检索出这条记忆。只输出查询本身,不要解释、不要引号、不超过 30 个字。"
    )
    out = []
    for qtype, a in angles:
        out.append((qtype, tok.apply_chat_template(
            [
                {"role": "system", "content": SYS},
                {"role": "user", "content": f"标签:{tags}\n记忆内容:\n{content}\n\n要求:{a}"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )))
    return out


def run_llm_track(mems: list[dict], per: int, batch_mems: int, rng: random.Random) -> list[dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(LLM_DIR))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(str(LLM_DIR))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        model = model.half()
    model = model.to(dev).eval()
    print(f"[LLM 轨] {LLM_DIR.name} device={dev}", flush=True)

    rows: list[dict] = []
    i = 0
    t0 = time.perf_counter()
    while i < len(mems):
        chunk = mems[i:i + batch_mems]
        prompts: list[str] = []
        plan: list[tuple[dict, list[str]]] = []
        for mem in chunk:
            angles = rng.sample(ANGLES, min(per, len(ANGLES)))
            ps = build_prompts(tok, mem, angles)
            plan.append((mem, [t for t, _ in ps]))
            prompts.extend(p for _, p in ps)
        try:
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=640).to(model.device)
            with torch.inference_mode():
                gen = model.generate(**enc, max_new_tokens=40, do_sample=True, temperature=0.95,
                                     top_p=0.92, repetition_penalty=1.1, pad_token_id=tok.pad_token_id)
            cur = 0
            for mem, types in plan:
                got: list[str] = []
                for j, g in enumerate(gen[cur:cur + len(types)]):
                    q = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                    q = q.strip('"“”「」').split("\n")[0].strip()[:40]
                    if len(q) >= 4:
                        rows.append({"id": mem["id"], "query": q, "qtype": types[j],
                                     "content": (mem.get("content") or "")[:512],
                                     "tags": mem.get("tags") or []})
                        got.append(q)
                cur += len(types)
        except Exception as e:  # noqa: BLE001
            print(f"  [跳过] {chunk[0]['id']}: {e}", flush=True)
        i += len(chunk)
        if i % 40 < batch_mems:
            el = time.perf_counter() - t0
            print(f"  {i}/{len(mems)} 记忆  样本 {len(rows)}  {el:.0f}s", flush=True)
    return rows


# ---------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description="生成分布对齐的检索训练数据 v2")
    ap.add_argument("--kw-per", type=int, default=4, help="每条记忆造几条关键词式查询")
    ap.add_argument("--llm-per", type=int, default=6, help="每条记忆造几条 LLM 查询")
    ap.add_argument("--batch-mems", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--kw-only", action="store_true", help="只造关键词轨(零 GPU,秒级)")
    ap.add_argument("--out", default=None, help="输出 jsonl(默认 work/train_pairs.jsonl)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out) if args.out else OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mems: dict[str, dict] = {}
    for p in sorted(PART_DIR.glob("part*.json")):
        for m in json.loads(p.read_text(encoding="utf-8")):
            mems[m["id"]] = m
    holdout = set(json.loads(HOLDOUT.read_text(encoding="utf-8"))) if HOLDOUT.exists() else set()
    seen = [m for mid, m in mems.items() if mid not in holdout]
    print(f"[数据] 记忆 {len(mems)} 条 → 训练用 {len(seen)} 条(排除 held-out {len(holdout)} 条)")

    rows: list[dict] = []

    # ---- kw 轨 ----
    kw_rows: list[dict] = []
    for mem in seen:
        pool = keyword_pool(mem)
        for q in make_kw_queries(mem, pool, args.kw_per, rng):
            kw_rows.append({"id": mem["id"], "query": q, "qtype": "kw",
                            "content": (mem.get("content") or "")[:512],
                            "tags": mem.get("tags") or []})
    print(f"[kw 轨] {len(kw_rows)} 条")
    rows.extend(kw_rows)

    # ---- LLM 轨 ----
    if not args.kw_only:
        rows.extend(run_llm_track(seen, args.llm_per, args.batch_mems, rng))

    # ---- 去重落盘 ----
    uniq: list[dict] = []
    seenq: set[tuple[str, str]] = set()
    for r in rows:
        k = (r["id"], r["query"])
        if k in seenq:
            continue
        seenq.add(k)
        uniq.append(r)
    rng.shuffle(uniq)

    with out_path.open("w", encoding="utf-8") as f:
        for r in uniq:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dist = Counter(r["qtype"] for r in uniq)
    print(f"\n[完成] {len(uniq)} 条 → {out_path}")
    print(f"[分布] " + "  ".join(f"{k}={v}" for k, v in sorted(dist.items())))
    print(f"[示例]")
    for r in uniq[:5]:
        print(f"    [{r['qtype']}] {r['query']}")


if __name__ == "__main__":
    main()
