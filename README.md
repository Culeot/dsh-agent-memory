# dsh-agent-memory

Cross-session long-term memory plugin for [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (DSH).

![Memory panel](https://raw.githubusercontent.com/Culeot/dsh-agent-memory/main/docs/memory-panel.png)

[中文文档](README.zh-CN.md)

## What it does

Remembers facts, preferences, decisions, and lessons across sessions, so a new session starts with what earlier sessions already learned. Storage lives in `$DSH_HOME/storages/memory.json` (plain JSON — inspectable, git-friendly). No MCP servers, no vector database, no embedding API, no extra runtime dependencies.

## Features

- **Cross-session persistence** — write in session A, recall in session B; new sessions pick up where old ones left off.
- **Relevance-threshold injection** — before each model step, the plugin recalls memories matching the user's current message, **filters by minimum relevance score** (`injectMinScore`), and injects up to `injectCount` that pass the threshold. Low-relevance memories are discarded — you only pay for what actually matters.
- **Multi-dimensional relevance** — scoring combines semantic similarity, task relevance, pattern matching, causal linkage, and recency (not just keyword overlap). A memory is used when ≥2 dimensions match, reducing noise from coincidental keyword hits.
- **Anti-decay rules** — the protocol section includes self-reinforcing rules: task-start recall re-reads core rules, recurring mistakes auto-solidify as lessons, and language drift triggers self-correction.
- **Self-correction loop** — when the same error (same code/message) fires repeatedly (`lessonizeAfter`, default 2), the plugin nudges the agent to write it as an importance-3 lesson; the lesson then auto-injects on related topics, preventing recurrence. User corrections are covered by the memory protocol (write the lesson right away).
- **Built-in hygiene** — capacity cap with lowest-value eviction first; near-duplicate entries merge instead of piling up (Jaccard ≥ 0.7); optional TTL expiry; deleting importance-3 records requires an explicit confirm.
- **Chinese-friendly search** — Chinese text is indexed by bigrams plus single-char fallback plus a BM25 term-frequency signal, English by words, plus exact substring matching. Works without a tokenizer or any ML dependency.
- **No unrelated association** — weak matches (pure single-char coincidence, filler-only queries like "ok了吗") score zero and are never injected; only substring/tag/bigram-strength matches surface. Saves tokens on short chatter.
- **Explainable recall** — every recall hit carries `reasons` (substring/tag/bigram/BM25/importance/recency/access signal breakdown), so both you and the agent can audit *why* a memory surfaced.
- **Memory panel (Web UI)** — a top-level "记忆" entry in Settings with stats, search, kind filters, and direct **create / edit / delete** of memories; changes apply immediately, dark mode included.
- **memory_sediment** — batch-persist facts/decisions/lessons at session wind-down (≤3 entries per call, cooldown-guarded); the agent summarizes what it already has in context, so there is zero extra model cost.
- **Native integration** — uses DSH's own storage domain (`ctx.storageDomain`), tool registry, and agent lifecycle hooks, so it stays compatible with official releases.

## Tools

| Tool | Purpose |
|---|---|
| `memory_remember` | Store a durable memory (content, kind, tags, scope, importance, optional TTL). |
| `memory_recall` | Search memory by keywords; ranks by relevance, importance, recency, past usage. |
| `memory_index` | Browse the inventory with kind/tag/scope filters, paginated, title-level. |
| `memory_forget` | Delete by id or tags; importance-3 records need `confirm: true`. |
| `memory_import` | Import memories from a JSONL/JSON file through the write chain (not by editing the store file) — safe bulk loading. |
| `memory_reload` | Reopen the store from disk after an external edit; merges external changes without a restart. |
| `memory_sediment` | Batch-persist several memories at once (session wind-down), with an entry cap and cooldown guard. |

Kinds: `fact | preference | decision | lesson | todo | note`. Scopes: `user` (applies everywhere) or `project` (this project only).

## External-modification protection

The store file is loaded once at startup and written as a whole by the running process (single-writer model). To prevent an in-process write from silently wiping external edits:

- every write checks the file fingerprint first — a mismatch (another process or a script edited it) **refuses the write** with a clear error instead of overwriting;
- `memory_import` is the supported way to bulk-load data (goes through the write chain, file and memory stay in sync);
- if you still edit `memory.json` by hand or copy it in, call `memory_reload` to merge it back (or restart).

## Troubleshooting: panel reports HTTP 405

A `transport failure for /dsh-memory-read/…: HTTP 405` in the memory panel means the RPC
channel was **never mounted** — the request fell through to the SPA static fallback, which
only serves GET/HEAD. Root cause is in DSH 0.1.5-rc.2 itself: `connection.rpc.handle()`
resolves `webServer` from the *calling* context to register its route, and throws
`cannot get property "webServer" without inject` when that lookup fails, silently dropping
the channel (still unfixed in 0.1.6-alpha.2). This plugin therefore tries the official
`handle()` first and, on failure, registers its own prefix routes through
`ctx.inject(['connection', 'webServer'])`, reusing `connection.requestRejection()` for the
Host/Origin fence and browser authentication.

**A DSH restart is required** — routes are mounted at startup. Verify with:

```bash
curl -i -X POST http://127.0.0.1:3080/dsh-memory-read/stats   # 401 = route mounted (no cookie), 405 = stale process
```

## Repeat suppression

Per-step injection skips when the recalled set is unchanged from the previous step, so consecutive messages about the same topic don't re-inject the same block — the「相关记忆」notice appears on topic change, not on every message.

## Install & enable

```bash
# 1. Add the dependency to your profile
cd ~/.dsh/profiles/<name>
npm install dsh-agent-memory
```
```bash
# 1. add the dependency to your profile
cd ~/.dsh/profiles/<name>
pnpm add dsh-agent-memory@file:/path/to/dsh-agent-memory
```

```yaml
# 2. add one row to your agent preset (~/.dsh/.agent-presets/<preset>/agent.cordis.yml)
- id: memory
  name: 'dsh-agent-memory'
```

```bash
# 3. restart DSH — the four tools appear in new sessions
```

No preset? Mount it on the host plane instead, in `~/.dsh/profiles/<name>/cordis.patch.yml`:

```yaml
- insert:
    - id: memory
      name: 'dsh-agent-memory'
```

Requires the storage trio already present in the profile (`dsh-storage`, `dsh-storage-json`, `dsh-storage-domain` — the web profile ships with it).

## Configuration

All options are optional:

| Option | Default | Meaning |
|---|---|---|
| `maxRecords` | 400 | Capacity cap; lowest-value records evicted first. |
| `maxContentChars` | 2000 | Max content length per record. |
| `injectEnabled` | true | Per-step injection of relevant memories (via `agent/pre-step`). |
| `injectCount` | 3 | Max memories injected per step (0 disables). |
| `injectMinScore` | 1.0 | Minimum relevance score to inject (0 = no threshold, just rank). |
| `injectMaxChars` | 120 | Max chars per injected memory summary. |
| `lessonizeEnabled` | true | Auto-nudge to solidify repeated errors as lessons. |
| `lessonizeAfter` | 2 | Same error fingerprint occurrences before nudging. |
| `recencyHalfLifeDays` | 90 | Freshness half-life. |
| `mergeSimilarity` | 0.7 | Near-duplicate merge threshold. |
| `protocolSection` | true | Inject the memory protocol prompt section. |

## Uninstall & troubleshooting

- Uninstall: `pnpm remove dsh-agent-memory` in the profile, delete the preset/patch row. Data stays in `memory.json` and is restored on reinstall.
- Tools missing: check the row exists, the dependency is installed, and DSH was restarted.
- Storage errors: the memory plugin needs the storage trio; add it to the patch if your profile lacks it.
- Corrupted `memory.json`: it is plain JSON — fix it by hand or delete it (deleting resets memory).

## Development

```bash
npm install && npm run build && npm test   # build + 31 unit tests
npm run smoke                              # real-machine headless round-trip check
```

## License

MIT
