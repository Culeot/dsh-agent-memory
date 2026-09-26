/**
 * dsh-agent-memory: cross-session long-term memory for DeepSeek Harness.
 *
 * One cordis plugin, three surfaces:
 *  1. a schema-validated `memory` domain opened on ctx.storageDomain
 *     (durable through the stock json backend → $DSH_HOME/storages);
 *  2. four model-facing tools — memory_remember / memory_recall /
 *     memory_index / memory_forget — with governance (capacity eviction,
 *     near-duplicate merge, TTL, access tracking) inside;
 *  3. a `memory-protocol` system-prompt section plus a dynamic
 *     `memory-state` context snapshot so the model knows the system exists
 *     and what it currently holds.
 *
 * Mount it inside an agent preset (recommended) or on a host plane row:
 *
 *   - id: memory
 *     name: 'dsh-agent-memory'
 *
 * @module dsh-agent-memory
 */
import { randomUUID } from 'node:crypto';
import { statSync, readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import type { Context } from '@deepseek-ai/cordis';
import z from '@deepseek-ai/schemastery';
import { HarnessError, createUserMessage, boundContextSummary } from '@deepseek-ai/dsh-llm';
import { defineTool } from '@deepseek-ai/dsh-tools';
import type { JsonValue } from '@deepseek-ai/dsh-tools';
import type { PreStepDecision } from '@deepseek-ai/dsh-agent';
import type { UserMessage } from '@deepseek-ai/dsh-session';
import {
  memoryDomain,
  MEMORY_KINDS,
  MEMORY_SCOPES,
  MEMORY_DEFAULTS,
  isValidKind,
  isValidScope,
  type MemoryRecord,
  type MemoryKind,
  type MemoryScope,
} from './spec.ts';
import {
  rankRecords,
  rankRecordsHybrid,
  countHybridMatches,
  hotRecords,
  isExpired,
  tokenSet,
  jaccard,
  scoreRecord,
  hasMeaningfulQuery,
  DEFAULT_SEMANTIC_WEIGHT,
  DEFAULT_SEMANTIC_MIN,
  IMPORTANCE_BOOST_SLOPE,
  LEXICAL_SATURATION,
} from './search.ts';
import { SemanticIndex, type SemanticConfig } from './semantic.ts';
import { registerRpc } from './rpc.ts';

export * from './spec.ts';
export * from './search.ts';
export * from './semantic.ts';
export * from './rpc.ts';

export const name = 'memory';
export const inject = ['tools', 'storageDomain', 'systemPrompt'];

/** Schemastery config (same convention as the official dsh-tool-* plugins). */
export const Config = z.object({
  maxRecords: z.number().min(1).default(MEMORY_DEFAULTS.maxRecords),
  maxContentChars: z.number().min(50).default(MEMORY_DEFAULTS.maxContentChars),
  recencyHalfLifeDays: z.number().min(1).default(MEMORY_DEFAULTS.recencyHalfLifeDays),
  mergeSimilarity: z.number().min(0.5).max(1).default(MEMORY_DEFAULTS.mergeSimilarity),
  /** Register the memory-protocol system-prompt section. */
  protocolSection: z.boolean().default(true),
  /**
   * Per-step dynamic injection: before each model step, recall the most
   * relevant memories for the user's current message and append them to the
   * step input. Replaces the old every-turn hot-memory broadcast (stateContext)
   * — relevant-on-demand instead of broadcast.
   */
  injectEnabled: z.boolean().default(true),
  /** Maximum memories to inject per step (0 disables injection). Actual count depends on relevance. */
  injectCount: z.number().min(0).max(10).default(3),
  /**
   * Budgeted injection: keep the top memory plus any scoring within this
   * fraction of it. 0 = off (historical behaviour: score floor + fixed cap).
   * A clear winner then injects one memory instead of three mediocre ones.
   */
  injectBudgetRatio: z.number().min(0).max(1).default(0.5),
  /** Minimum score threshold for injection (0 = no threshold, just use rank). */
  injectMinScore: z.number().min(0).max(100).default(1.0),
  /** Max content chars per injected memory summary. */
  injectMaxChars: z.number().min(40).max(400).default(120),
  /** Default max chars of a recalled record's content shown to the model. */
  recallContentMax: z.number().min(100).max(2000).default(MEMORY_DEFAULTS.recallContentMax),
  /** memory_sediment: max entries per call. */
  sedimentMaxEntries: z.number().min(1).max(10).default(MEMORY_DEFAULTS.sedimentMaxEntries),
  /** memory_sediment: min ms between calls; 0 disables the cooldown. */
  sedimentCooldownMs: z.number().min(0).default(MEMORY_DEFAULTS.sedimentCooldownMs),
  /**
   * Lesson auto-solidification: when the SAME error fingerprint (code/message)
   * fires this many times, inject a nudge telling the agent to write it as an
   * importance-3 lesson. The lesson then auto-injects on related topics,
   * preventing recurrence. 0 disables.
   */
  lessonizeEnabled: z.boolean().default(true),
  lessonizeAfter: z.number().min(2).max(20).default(2),
  /**
   * Semantic channel: fuse a local sentence-embedding model with the lexical
   * scorer. Needs @huggingface/transformers (an OPTIONAL peer dependency) plus
   * a model directory; when either is missing, recall silently falls back to
   * lexical-only — an optional accelerator must never break the plugin.
   *
   * Measured on a real 382-record bank: lexical 0.7248 MRR → fused 0.8432.
   */
  semanticEnabled: z.boolean().default(true),
  /** Model directory (config.json / tokenizer.json / onnx/*.onnx). '' disables. */
  semanticModelDir: z.string().default(''),
  /** Vector-cache directory. '' = $DSH_HOME/storages/memory-semantic. */
  semanticCacheDir: z.string().default(''),
  /** Semantic weight in fusion: relevance = w*semantic + (1-w)*lexical. */
  semanticWeight: z.number().min(0).max(1).default(DEFAULT_SEMANTIC_WEIGHT),
  /** Minimum cosine for a record to count as a match with no lexical hit. */
  semanticMin: z.number().min(0).max(1).default(DEFAULT_SEMANTIC_MIN),
  /**
   * Slope of the importance boost. The historical 0.75 measured -0.046 MRR
   * versus no boost at all (most records sit at importance 3, so the boost
   * separates nothing while punishing the rest). 0 disables it.
   */
  boostSlope: z.number().min(0).max(2).default(IMPORTANCE_BOOST_SLOPE),
  /**
   * "Say nothing rather than something wrong" gate, on the 0..1 relevance scale.
   * When the best candidate scores below this, recall reports `noMatch` and the
   * injection path stays silent — offering an unrelated memory is worse than
   * offering none: it spends the attention budget and misleads.
   *
   * Calibrated against 382 queries that DO have an answer and 20 that do NOT
   * (src/mem_calibrate.py). On the lexical path the equivalent base threshold
   * was 0.735 (85% of real queries admitted, 5% of the negatives leaking).
   */
  noMatchThreshold: z.number().min(0).max(1).default(0.18),
});

function makeId(): string {
  const rand = randomUUID().replaceAll('-', '');
  return `mem_${rand.slice(0, 16)}`;
}

function iso(now: number): string {
  return new Date(now).toISOString();
}

function truncate(content: string, max: number): string {
  const chars = Array.from(content);
  if (chars.length <= max) return content;
  return chars.slice(0, max - 1).join('') + '…';
}

/** Extract a search query from the user messages claimed by this step. */
export function queryFromMessages(messages: readonly UserMessage[]): string {
  const parts: string[] = [];
  for (const msg of messages ?? []) {
    const content = (msg as { data?: { content?: unknown } }).data?.content ?? (msg as { content?: unknown }).content;
    if (typeof content === 'string') parts.push(content);
    else if (Array.isArray(content)) parts.push(content.map((b) => (typeof (b as { text?: unknown })?.text === 'string' ? (b as { text: string }).text : '')).join('\n'));
  }
  return parts.join('\n').slice(0, 600).trim();
}

/** Render recalled memories into a compact injection block. */
export function renderInjection(results: RecallResult['results'], maxChars: number): string {
  if (results.length === 0) return '';
  const lines = results.map((r) => {
    const text = truncate(r.content, maxChars);
    const flags = [r.importance === 3 ? '重要' : null, r.scope === 'user' ? '全局' : null].filter(Boolean).join('/');
    return `- ${text}${flags ? `(${flags})` : ''}`;
  });
  return `相关记忆(自动检索 ${results.length} 条,可能与当前任务相关):\n${lines.join('\n')}\n需要细节用 memory_recall,过时用 memory_forget。`;
}

/**
 * Pick the memories worth spending attention on.
 *
 * The old rule was "everything above a score floor, capped at N" — which treats a
 * clear winner and a pack of marginal matches identically. Budgeted selection
 * looks at the *shape* of the score curve instead: keep the top record, then only
 * those within `budgetRatio` of it. A query with one obvious answer injects one
 * memory; a query with several comparably relevant memories injects several.
 *
 * Why this matters: attention is the scarce resource (see docs/BUILD-PLAN.md).
 * Injecting a marginal memory is not neutral — it spends budget and can mislead.
 *
 * `budgetRatio` 0 disables the budget (historical behaviour: floor + cap only).
 */
export function selectForInjection<T extends { score: number }>(
  results: readonly T[],
  options: { minScore?: number; budgetRatio?: number; limit: number },
): T[] {
  const minScore = options.minScore ?? 0;
  let kept = minScore > 0 ? results.filter((r) => r.score >= minScore) : [...results];
  const ratio = options.budgetRatio ?? 0;
  if (ratio > 0 && kept.length > 1) {
    const top = kept[0]!.score;
    if (top > 0) kept = kept.filter((r) => r.score >= top * ratio);
  }
  return kept.slice(0, Math.max(0, options.limit));
}

/**
 * Stable fingerprint of an error for same-mistake counting. Prefer a typed
 * code; fall back to the first meaningful line of the message.
 */
export function extractErrorFingerprint(error: unknown): string {
  if (typeof error === 'string') return error.trim().slice(0, 80);
  const e = error as { code?: unknown; message?: unknown; name?: unknown } | null;
  const code = typeof e?.code === 'string' && e.code !== '' ? e.code : null;
  if (code) return code;
  const msg = typeof e?.message === 'string' ? e.message.trim().slice(0, 80) : '';
  if (msg) return msg;
  return typeof e?.name === 'string' ? e.name : 'unknown-error';
}

function normalizeTags(tags: readonly string[] | undefined): string[] {
  if (!tags) return [];
  const out: string[] = [];
  for (const raw of tags) {
    const tag = String(raw).trim().toLowerCase().slice(0, 64);
    if (tag !== '' && !out.includes(tag)) out.push(tag);
  }
  return out.slice(0, 16);
}

export type RememberResult = {
  id: string;
  merged: boolean;
  evicted: number;
  content: string;
  /** Ids of older records this write marked as superseded (relation data). */
  superseded?: string[];
};

export type RecallResult = {
  results: Array<{
    id: string;
    content: string;
    kind: MemoryKind;
    tags: string[];
    scope: MemoryScope;
    importance: number;
    updatedAt: string;
    score: number;
    /** Set when a newer memory supersedes this one (it is shown, but down-weighted). */
    supersededBy?: string | null;
    /** Human-readable hit reasons (query recall only; absent for hot set). */
    reasons?: string[];
  }>;
  /** Records that matched (after filters), before the limit cut. */
  totalMatched: number;
  /**
   * True when nothing cleared the relevance gate. Callers should say
   * "no relevant memory" instead of presenting the top-ranked record — a wrong
   * memory is more harmful than an absent one.
   */
  noMatch: boolean;
  /** Fused relevance (0..1) of the best candidate; 0 when nothing was found. */
  matchStrength: number;
  /** Records actually returned (== results.length). */
  returned: number;
};

export type IndexResult = {
  total: number;
  expired: number;
  byKind: Record<string, number>;
  entries: Array<{
    id: string;
    content: string;
    kind: MemoryKind;
    tags: string[];
    scope: MemoryScope;
    importance: number;
    updatedAt: string;
  }>;
};

export type ForgetResult = {
  deleted: number;
  /** importance-3 records skipped (they need confirm: true). */
  skippedImportant: number;
};

export type MemoryTable = {
  get(key: string): MemoryRecord | undefined;
  entries(): IterableIterator<[string, MemoryRecord]>;
  put(key: string, value: MemoryRecord): Promise<void>;
  update(key: string, fn: (current: MemoryRecord) => MemoryRecord): Promise<MemoryRecord>;
  delete(key: string): Promise<boolean>;
  readonly size: number;
};

/**
 * Pure governance/read/write core over an opened domain table. Split out so
 * tests can drive it without a live cordis context.
 *
 * `beforeWrite` is an optional external-modification guard: called before every
 * mutating operation. Throw to refuse the write (the caller turns it into a
 * clear error) — the fail-safe that prevents an in-memory state from silently
 * overwriting a file another process edited.
 */
export class MemoryCore {
  /**
   * Optional semantic channel. Null when disabled or when no model directory is
   * configured; recall then behaves exactly as it did before this existed.
   */
  private readonly semantic: SemanticIndex | null;

  constructor(
    private readonly table: MemoryTable,
    private readonly config: {
      maxRecords: number;
      maxContentChars: number;
      mergeSimilarity: number;
      recencyHalfLifeDays: number;
      semantic?: SemanticConfig | null;
      semanticWeight?: number;
      semanticMin?: number;
      boostSlope?: number;
      /** Relevance gate (0..1); below it recall reports noMatch. */
      noMatchThreshold?: number;
    },
    private readonly beforeWrite?: () => void,
    /**
     * Called after EVERY durable write lands (each put/update/delete). The
     * external-modification guard compares fingerprints before a write, so the
     * baseline must be refreshed after each real write — not after a whole
     * multi-write operation (import) — otherwise the second write of a batch
     * is falsely rejected as "externally modified".
     */
    private readonly onWritten?: () => void,
  ) {
    // Constructed eagerly, loaded lazily: starting a plugin must not block on
    // a 100 MB ONNX runtime, and a missing model must not be an error.
    this.semantic = config.semantic ? new SemanticIndex(config.semantic) : null;
  }

  /** Semantic-channel diagnostics, surfaced to the UI/RPC layer. */
  semanticStatus(): {
    enabled: boolean;
    ready: boolean;
    reason: string | null;
    backend: string | null;
    vectors: number;
  } {
    return {
      enabled: this.semantic !== null,
      ready: this.semantic?.isReady ?? false,
      reason: this.semantic?.unavailableReason ?? null,
      backend: this.semantic?.backendName ?? null,
      vectors: this.semantic?.vectorCount ?? 0,
    };
  }

  /**
   * Cosine scores for the pool, or null when the semantic channel is off.
   * Never throws: an optional channel that fails must degrade, not break.
   */
  private async semanticScores(query: string, pool: readonly MemoryRecord[]): Promise<number[] | null> {
    if (!this.semantic) return null;
    try {
      return await this.semantic.scores(query, pool);
    } catch {
      return null;
    }
  }

  private all(): MemoryRecord[] {
    return [...this.table.entries()].map(([, v]) => v);
  }

  private guardWrite(): void {
    if (this.beforeWrite) this.beforeWrite();
  }

  private async put(key: string, value: MemoryRecord): Promise<void> {
    await this.table.put(key, value);
    if (this.onWritten) this.onWritten();
  }

  private async update(key: string, fn: (current: MemoryRecord) => MemoryRecord): Promise<MemoryRecord> {
    const next = await this.table.update(key, fn);
    if (this.onWritten) this.onWritten();
    return next;
  }

  private async delete(key: string): Promise<boolean> {
    const existed = await this.table.delete(key);
    if (existed && this.onWritten) this.onWritten();
    return existed;
  }

  /**
   * Mark older records as replaced by `newId`.
   *
   * The protocol has always told the model to note "已由 X 更新/覆盖" in the body
   * when a decision changes — but that is discipline, not mechanism, and it left
   * the outdated record ranking exactly as high as the new one. Turning the
   * relation into data lets the ranker down-weight it (see SUPERSEDED_PENALTY).
   *
   * The old record keeps its content: auditable, and reversible by clearing
   * `supersededBy`. Unknown ids and self-references are ignored.
   */
  private async markSuperseded(newId: string, ids: readonly string[] | undefined): Promise<string[]> {
    if (!ids || ids.length === 0) return [];
    const marked: string[] = [];
    for (const oldId of ids) {
      if (oldId === newId) continue;
      if (!this.table.get(oldId)) continue;
      await this.update(oldId, (current) => ({ ...current, supersededBy: newId }));
      marked.push(oldId);
    }
    return marked;
  }

  /** Delete every TTL-expired record. Returns the number removed. */
  async purgeExpired(now: number = Date.now()): Promise<number> {
    let removed = 0;
    for (const [key, record] of [...this.table.entries()]) {
      if (isExpired(record, now)) {
        await this.delete(key);
        removed += 1;
      }
    }
    return removed;
  }

  /**
   * Evict lowest-value records until the table fits maxRecords. importance-3
   * records are eviction-proof unless the whole store is full of them — in
   * that case the oldest importance-3 record goes, so the store can never
   * deadlock while still honoring "critical survives space pressure".
   */
  private async evict(): Promise<number> {
    const excess = this.table.size - this.config.maxRecords;
    if (excess <= 0) return 0;
    const all = this.all();
    const sortKey = (r: MemoryRecord) => r.accessedAt ?? r.updatedAt;
    const nonCritical = all.filter((r) => r.importance < 3).sort((a, b) => {
      if (a.importance !== b.importance) return a.importance - b.importance;
      return sortKey(a) < sortKey(b) ? -1 : sortKey(a) > sortKey(b) ? 1 : 0;
    });
    const critical = all.filter((r) => r.importance === 3).sort((a, b) =>
      sortKey(a) < sortKey(b) ? -1 : sortKey(a) > sortKey(b) ? 1 : 0,
    );
    const victims = [...nonCritical, ...critical].slice(0, excess);
    let removed = 0;
    for (const victim of victims) {
      if (await this.delete(victim.id)) removed += 1;
    }
    return removed;
  }

  /** Insert or merge one memory. */
  async remember(input: {
    content: string;
    kind: string;
    tags?: readonly string[];
    scope: string;
    project?: string | null;
    importance?: number;
    ttlDays?: number;
    /** Ids of older records this new memory replaces (they get marked). */
    supersedes?: readonly string[];
    now?: number;
  }): Promise<RememberResult> {
    this.guardWrite();
    const now = input.now ?? Date.now();
    const kind = isValidKind(input.kind) ? input.kind : 'note';
    const scope = isValidScope(input.scope) ? input.scope : 'project';
    const importance = Math.min(3, Math.max(1, Math.round(input.importance ?? 2)));
    const content = truncate(input.content.trim(), this.config.maxContentChars);
    if (content === '') throw new HarnessError('memory content is empty', 'MEMORY_EMPTY_CONTENT');
    const tags = normalizeTags(input.tags);
    const project = scope === 'project' ? (input.project ?? null) : null;
    const expiresAt = input.ttlDays && input.ttlDays > 0
      ? new Date(now + Math.min(input.ttlDays, 3650) * 24 * 3600 * 1000).toISOString()
      : null;

    // Near-duplicate merge: same kind/scope/project, token Jaccard >= threshold.
    const incomingTokens = tokenSet(content);
    let merged = false;
    for (const existing of this.all()) {
      if (isExpired(existing, now)) continue;
      if (existing.kind !== kind || existing.scope !== scope || existing.project !== project) continue;
      if (jaccard(incomingTokens, tokenSet(existing.content)) < this.config.mergeSimilarity) continue;
      await this.update(existing.id, (current) => {
        const tagsMerged = [...new Set([...current.tags, ...tags])];
        return {
          ...current,
          content: content.length >= current.content.length ? content : current.content,
          tags: tagsMerged,
          importance: Math.max(current.importance, importance),
          updatedAt: iso(now),
          accessedAt: iso(now),
          accessCount: current.accessCount + 1,
          // Only an explicit ttl_days refreshes/clears expiry; otherwise the
          // merged record keeps its existing TTL instead of silently extending
          // or clearing it.
          expiresAt: input.ttlDays !== undefined && input.ttlDays !== null
            ? expiresAt
            : current.expiresAt,
        };
      });
      merged = true;
      const supersededByMerge = await this.markSuperseded(existing.id, input.supersedes);
      return { id: existing.id, merged: true, evicted: 0, content, superseded: supersededByMerge };
    }

    const id = makeId();
    const record: MemoryRecord = {
      id,
      content,
      kind,
      tags,
      scope,
      project,
      importance,
      createdAt: iso(now),
      updatedAt: iso(now),
      accessedAt: null,
      accessCount: 0,
      expiresAt,
    };
    await this.put(id, record);
    const superseded = await this.markSuperseded(id, input.supersedes);
    const evicted = await this.evict();
    return { id, merged: false, evicted, content, superseded };
  }

  /**
   * Ranked recall with kind/tag/scope filters and access tracking.
   * `touch: false` skips the access-count write-back entirely — used by the
   * per-step injection path so passive recall never triggers a disk write.
   */
  async recall(input: {
    query: string;
    kinds?: readonly string[];
    tags?: readonly string[];
    scope?: string;
    project?: string | null;
    limit?: number;
    now?: number;
    touch?: boolean;
    /** Max content chars per result; longer content is truncated (cheaper calls). */
    contentMax?: number;
  }): Promise<RecallResult> {
    const now = input.now ?? Date.now();
    if (input.touch !== false) {
      this.guardWrite();
      await this.purgeExpired(now);
    }
    const kinds = (input.kinds ?? []).filter(isValidKind);
    const tags = normalizeTags(input.tags);
    const scope = input.scope && isValidScope(input.scope) ? input.scope : undefined;
    const project = input.project ?? undefined;

    let pool = this.all().filter((r) => !isExpired(r, now));
    if (kinds.length > 0) pool = pool.filter((r) => kinds.includes(r.kind));
    if (tags.length > 0) pool = pool.filter((r) => tags.every((t) => r.tags.includes(t)));
    if (scope) pool = pool.filter((r) => r.scope === scope);
    if (project) pool = pool.filter((r) => r.scope === 'user' || r.project === project);

    const limit = Math.max(1, Math.min(50, Math.round(input.limit ?? MEMORY_DEFAULTS.recallLimit)));
    const query = (input.query ?? '').trim();
    const rankOptions = {
      now,
      recencyHalfLifeDays: this.config.recencyHalfLifeDays,
      boostSlope: this.config.boostSlope,
    };
    const hybridOptions = {
      ...rankOptions,
      semanticWeight: this.config.semanticWeight,
      semanticMin: this.config.semanticMin,
    };
    // Computed once per query over the filtered pool. Null means the channel is
    // off (no package / no model), and the hybrid ranker then degrades to the
    // pure-lexical behaviour that existed before.
    const semantic = query === '' ? null : await this.semanticScores(query, pool);
    const ranked = query === ''
      ? hotRecords(pool, limit, rankOptions)
      : rankRecordsHybrid(pool, query, limit, { ...hybridOptions, semantic });
    // True match count before the limit cut (the rankers filter internally).
    const totalMatched = query === ''
      ? pool.length
      : countHybridMatches(pool, query, { ...hybridOptions, semantic });

    // Touch the winners atomically so access tracking feeds future
    // scoring/eviction without racing concurrent merge/remember updates.
    // Skipped entirely when touch: false (passive injection path).
    if (input.touch !== false) {
      for (const { record } of ranked) {
        try {
          await this.update(record.id, (current) => ({
            ...current,
            accessedAt: iso(now),
            accessCount: current.accessCount + 1,
          }));
        } catch {
          // Record deleted concurrently; its touch is meaningless. Keep going.
        }
      }
    }

    // 「宁可不给,也不硬凑」闸门:最高相关性低于阈值时明确报告没有匹配,
    // 调用方据此保持沉默,而不是推一条无关记忆出去。
    // 纯词面路径没有 fusion relevance,用归一化的 base 顶上,保证两种模式同尺度。
    const topRelevance = ranked.length > 0
      ? (ranked[0].relevance ?? Math.min(1, (ranked[0].base ?? 0) / LEXICAL_SATURATION))
      : 0;
    // 阈值 0 表示关闭闸门(gate > 0 才判断),这样旧配置与不传该参数的调用方行为不变。
    const gate = this.config.noMatchThreshold ?? 0;
    const noMatch = query !== '' && gate > 0 && topRelevance < gate;

    return {
      totalMatched,
      noMatch,
      matchStrength: Number(topRelevance.toFixed(4)),
      returned: ranked.length,
      results: ranked.map(({ record, score, reasons }) => ({
        id: record.id,
        content: truncate(record.content, input.contentMax ?? MEMORY_DEFAULTS.recallContentMax),
        kind: record.kind,
        tags: record.tags,
        scope: record.scope,
        importance: record.importance,
        updatedAt: record.updatedAt,
        score: Number(score.toFixed(3)),
        ...(record.supersededBy ? { supersededBy: record.supersededBy } : {}),
        reasons,
      })),
    };
  }

  /** Read-only single record lookup (used by the UI RPC layer). */
  getById(id: string): MemoryRecord | undefined {
    return this.table.get(id);
  }

  /**
   * Read-only inventory view for the UI panel: never purges/writes, so a
   * passive page load can't mutate the store. Mirrors index() minus the
   * purge and without touching access stats.
   */
  inspect(input: {
    kinds?: readonly string[];
    tags?: readonly string[];
    scope?: string;
    limit?: number;
    offset?: number;
    now?: number;
  }): { total: number; expired: number; byKind: Record<string, number>; entries: Array<{
    id: string;
    content: string;
    kind: MemoryKind;
    tags: string[];
    scope: MemoryScope;
    importance: number;
    updatedAt: string;
    expiresAt: string | null;
  }> } {
    const now = input.now ?? Date.now();
    const kinds = (input.kinds ?? []).filter(isValidKind);
    const tags = normalizeTags(input.tags);
    const scope = input.scope && isValidScope(input.scope) ? input.scope : undefined;

    let pool = this.all();
    if (kinds.length > 0) pool = pool.filter((r) => kinds.includes(r.kind));
    if (tags.length > 0) pool = pool.filter((r) => tags.every((t) => r.tags.includes(t)));
    if (scope) pool = pool.filter((r) => r.scope === scope);

    const expired = pool.filter((r) => isExpired(r, now)).length;
    const byKind: Record<string, number> = {};
    for (const r of pool) byKind[r.kind] = (byKind[r.kind] ?? 0) + 1;

    pool.sort((a, b) => (a.updatedAt < b.updatedAt ? 1 : a.updatedAt > b.updatedAt ? -1 : 0));
    const offset = Math.max(0, Math.round(input.offset ?? 0));
    const limit = Math.max(1, Math.min(200, Math.round(input.limit ?? 50)));
    const entries = pool.slice(offset, offset + limit).map((r) => ({
      id: r.id,
      content: truncate(r.content, 161),
      kind: r.kind,
      tags: r.tags,
      scope: r.scope,
      importance: r.importance,
      updatedAt: r.updatedAt,
      expiresAt: r.expiresAt,
    }));

    return { total: pool.length, expired, byKind, entries };
  }

  /** Full inventory with stats (title-level listing to bound token cost). */
  async index(input: {
    kinds?: readonly string[];
    tags?: readonly string[];
    scope?: string;
    limit?: number;
    offset?: number;
    now?: number;
  }): Promise<IndexResult> {
    const now = input.now ?? Date.now();
    await this.purgeExpired(now);
    const kinds = (input.kinds ?? []).filter(isValidKind);
    const tags = normalizeTags(input.tags);
    const scope = input.scope && isValidScope(input.scope) ? input.scope : undefined;

    let pool = this.all();
    if (kinds.length > 0) pool = pool.filter((r) => kinds.includes(r.kind));
    if (tags.length > 0) pool = pool.filter((r) => tags.every((t) => r.tags.includes(t)));
    if (scope) pool = pool.filter((r) => r.scope === scope);

    const expired = pool.filter((r) => isExpired(r, now)).length;
    const byKind: Record<string, number> = {};
    for (const r of pool) byKind[r.kind] = (byKind[r.kind] ?? 0) + 1;

    pool.sort((a, b) => (a.updatedAt < b.updatedAt ? 1 : a.updatedAt > b.updatedAt ? -1 : 0));
    const offset = Math.max(0, Math.round(input.offset ?? 0));
    const limit = Math.max(1, Math.min(100, Math.round(input.limit ?? 20)));
    const entries = pool.slice(offset, offset + limit).map((r) => ({
      id: r.id,
      content: truncate(r.content, 121),
      kind: r.kind,
      tags: r.tags,
      scope: r.scope,
      importance: r.importance,
      updatedAt: r.updatedAt,
    }));

    return { total: pool.length, expired, byKind, entries };
  }

  /** Delete by id, or by tag set (optionally scoped). importance-3 ids need confirm. */
  async forget(input: {
    id?: string;
    tags?: readonly string[];
    scope?: string;
    confirm?: boolean;
    now?: number;
  }): Promise<ForgetResult> {
    this.guardWrite();
    const now = input.now ?? Date.now();
    const tags = normalizeTags(input.tags);
    const scope = input.scope && isValidScope(input.scope) ? input.scope : undefined;

    let deleted = 0;
    let skippedImportant = 0;
    if (input.id) {
      const record = this.table.get(input.id);
      if (!record) return { deleted: 0, skippedImportant: 0 };
      if (record.importance === 3 && input.confirm !== true) {
        throw new HarnessError(`memory ${input.id} is importance 3; pass confirm: true to delete`, 'MEMORY_CONFIRM_REQUIRED');
      }
      deleted = (await this.delete(input.id)) ? 1 : 0;
    } else if (tags.length > 0) {
      for (const [key, record] of [...this.table.entries()]) {
        if (isExpired(record, now)) continue;
        if (scope && record.scope !== scope) continue;
        if (!tags.every((t) => record.tags.includes(t))) continue;
        if (record.importance === 3 && input.confirm !== true) {
          skippedImportant += 1;
          continue;
        }
        if (await this.delete(key)) deleted += 1;
      }
    } else {
      throw new HarnessError('forget requires an id or at least one tag', 'MEMORY_FORGET_TARGET_REQUIRED');
    }
    return { deleted, skippedImportant };
  }

  /** Manual edit from the UI panel: overwrite content/tags/importance directly. */
  async updateContent(input: {
    id: string;
    content?: string;
    tags?: readonly string[];
    importance?: number;
    now?: number;
  }): Promise<{ id: string; updated: boolean }> {
    this.guardWrite();
    const now = input.now ?? Date.now();
    const record = this.table.get(input.id);
    if (!record) return { id: input.id, updated: false };
    const content = input.content !== undefined ? truncate(input.content.trim(), this.config.maxContentChars) : record.content;
    if (content === '') throw new HarnessError('memory content is empty', 'MEMORY_EMPTY_CONTENT');
    const tags = input.tags !== undefined ? normalizeTags(input.tags) : record.tags;
    const importance = input.importance !== undefined ? Math.min(3, Math.max(1, Math.round(input.importance))) : record.importance;
    await this.update(input.id, (current) => ({
      ...current,
      content,
      tags,
      importance,
      updatedAt: iso(now),
    }));
    return { id: input.id, updated: true };
  }

  stats(): { total: number; byKind: Record<string, number> } {
    const byKind: Record<string, number> = {};
    for (const r of this.all()) byKind[r.kind] = (byKind[r.kind] ?? 0) + 1;
    return { total: this.table.size, byKind };
  }
}

const PROTOCOL_SECTION = `## Long-term memory (dsh-agent-memory)

You have a cross-session long-term memory, and relevant memories are auto-injected into each step based on the user's current message (look for a "相关记忆" block — treat it as context, not as the user speaking).

Rules:
- At the START of a task, run memory_recall with keywords from the request before diving in; use memory_index when you need the big picture of what is already known. The auto-injected block is a preview — recall for details.
- DURING work, actively memory_remember anything the user will need in LATER sessions: facts about them or their environment, stated preferences, concluded decisions and their rationale, lessons from mistakes ("never do X again"), and cross-session todos. Quality over quantity — do not log noise.
- When the user corrects outdated knowledge, fix memory right away: pass the stale record's id in \`supersedes\` when storing the new one. The old record stays stored (auditable) but is heavily down-weighted, so it stops being presented as the current conclusion. Use memory_forget only when a memory is *wrong*, not merely *outdated*.
- When the user corrects you, or the same mistake recurs twice, immediately memory_remember it as a lesson (kind=lesson, importance=3) — the lesson then auto-injects on related topics, preventing recurrence.
- Write lessons DIALECTICALLY — never absolute. Distinguish: (1) what went wrong and how to avoid it; (2) the CONDITIONS that caused the failure (would it have worked under different conditions? note the applicability boundary — a lesson's value depends on its conditions); (3) anything that actually worked — keep the salvageable part. Prefer "under condition A, X failed because B; the C part worked" over "X is impossible". When new evidence contradicts an old lesson (e.g. it worked under changed conditions), UPDATE the old lesson rather than stacking a new one.
- Treat the memory store dialectically too: (1) contradicting memories are not necessarily wrong — separate conditions and timing before judging; two records can both be true under different conditions; (2) when a new decision supersedes an old one, mark the old record as superseded ("已由 <X> 更新/覆盖") instead of leaving silent contradictions; (3) before forgetting a memory, confirm its conditions are truly gone — do not discard it merely because it feels outdated.
- kind: one of fact / preference / decision / lesson / todo / note. tags: short lowercase words. importance: 1 nice-to-know, 2 useful, 3 critical (eviction-proof). scope: user (applies everywhere) or project (this project only).
- Near-duplicate entries merge automatically; do not re-create the same memory on purpose.
- **Memory retrieval relevance is multi-dimensional — never single-indicator:** (1) semantic similarity (not just keyword overlap), (2) task relevance (does it directly help complete the current task?), (3) pattern match (same scenario pattern?), (4) causal link (prerequisite/cause/consequence?), (5) recency (recent may be better, but old memories have value too). Apply if ≥2 dimensions match; otherwise discard. Hot-count(auto-injected) memories are a safety net only — always run memory_recall for task-specific retrieval.
- **Anti-decay rules:** The memory system itself can "decay" (forget rules over long conversations). Countermeasures: (1) memory_retrieval at task start acts as a fresh re-read of core rules; (2) the SAME mistake recurring twice triggers an automatic lesson-importance-3 nudge; (3) if you notice yourself drifting from your configured behavior, use memory_recall to retrieve your core rules and re-ground yourself.`;

/** Shared canonical-output shape: author-only JSON node, pretty-printed. */
function jsonOutput(): { schema: { type: 'json' }; render: (args: unknown, value: JsonValue) => Array<{ type: 'text'; text: string }> } {
  return {
    schema: { type: 'json' },
    render: (_args, value) => [{ type: 'text', text: JSON.stringify(value, null, 2) }],
  };
}

export async function apply(ctx: Context, config: Schemastery.TypeT<typeof Config>): Promise<void> {
  // The memory domain is a process-wide singleton: "DomainFacility.open" rejects
  // a second open of the same name (already-open). A preset mounts per session,
  // so the first session opens the domain and every later session reuses it —
  // only the opener registers the close effect.
  const existing = ctx.storageDomain.get(memoryDomain.name);
  let domain = existing ?? (await ctx.storageDomain.open(memoryDomain));
  if (!existing) {
    ctx.effect(() => () => {
      void domain.close();
    });
  }

  // External-modification guard: the domain holds the process-authoritative
  // in-memory copy and writes the whole file on every change. If another
  // process edited memory.json behind our back, a write would silently destroy
  // those edits. Before every write we compare the file fingerprint; on a
  // mismatch we refuse the write (never overwrite) and tell the caller to
  // memory_reload (merge) or restart. Our own writes refresh the fingerprint.
  const storePath = join(process.env.DSH_HOME ?? join(homedir(), '.dsh'), 'storages', 'memory.json');
  const fingerprint = (): string | null => {
    try {
      const st = statSync(storePath);
      return `${st.mtimeMs}:${st.size}`;
    } catch {
      return null; // file absent (never written yet) — nothing to guard
    }
  };
  let baseFingerprint = fingerprint();
  const guard = () => {
    const cur = fingerprint();
    if (cur !== null && baseFingerprint !== null && cur !== baseFingerprint) {
      throw new HarnessError('memory.json 已被进程外修改(可能脚本或另一个 DSH 实例改过)。为防止覆盖,本次写入已拒绝。请调用 memory_reload 合并外部修改,或重启 DSH。', 'MEMORY_EXTERNAL_MODIFIED');
    }
  };
  const refreshFingerprint = () => { baseFingerprint = fingerprint(); };

  // Semantic channel wiring. The model directory is explicit configuration —
  // the plugin never downloads anything on its own, and an empty value means
  // "stay lexical-only", which is also what happens when the optional
  // @huggingface/transformers package is absent.
  const semanticConfig: SemanticConfig | null = config.semanticEnabled && config.semanticModelDir !== ''
    ? {
        modelDir: config.semanticModelDir,
        cacheDir: config.semanticCacheDir !== ''
          ? config.semanticCacheDir
          : join(process.env.DSH_HOME ?? join(homedir(), '.dsh'), 'storages', 'memory-semantic'),
      }
    : null;
  const coreConfig = { ...config, semantic: semanticConfig };
  const makeCore = () => new MemoryCore((domain.table('records') as unknown) as MemoryTable, coreConfig, guard, refreshFingerprint);
  let core = makeCore();

  // UI-facing RPC (web only): browse/manage memories from the settings panel.
  // 必须同时注入 connection 与 webServer:0.1.5-rc.2 的 connection.rpc.handle() 会从
  // **调用方 ctx** 解析 webServer,只注入 connection 时它静默丢掉路由(面板报 HTTP 405)。
  // 注册与自注册兜底见 src/rpc.ts 顶部说明。
  ctx.inject(['connection', 'webServer'], (webContext: any) => {
    if (webContext?.connection === undefined) return;
    registerRpc(
      { connection: webContext.connection, webServer: webContext.webServer },
      () => core,
      (message) => ctx.logger?.warn?.(message),
    );
  });

  ctx.tools.register(defineTool({
    name: 'memory_remember',
    description:
      'Store one durable cross-session memory. Use proactively for user facts, preferences, decisions, lessons, and todos that matter beyond the current session. Near-duplicate content merges into the existing record; the store evicts lowest-value records when full.',
    parameters: {
      content: { type: 'string', required: true, description: 'What to remember. One clear statement; write it so a future session understands it without this conversation.' },
      kind: { type: 'string', enum: [...MEMORY_KINDS], description: 'Memory kind: fact / preference / decision / lesson / todo / note. Defaults to note when omitted.' },
      tags: { type: 'array', items: { type: 'string' }, description: 'Short lowercase lookup tags, e.g. ["deploy","sensitive"]. Max 16.' },
      scope: { type: 'string', enum: [...MEMORY_SCOPES], description: 'user = applies in every project; project = this project only.' },
      project: { type: 'string', description: 'For scope=project: the project path this memory belongs to. Fill from your known working directory ({{cwd}}).' },
      importance: { type: 'number', description: '1 nice-to-know, 2 useful, 3 critical (never evicted for space). Defaults to 2.' },
      ttl_days: { type: 'number', description: 'Optional: auto-expire after N days (e.g. temporary credentials, short-lived decisions). Max 3650.' },
      supersedes: {
        type: 'array',
        items: { type: 'string' },
        description:
          'Ids of older memories this one REPLACES (e.g. a decision that changed). They stay stored — auditable — but are heavily down-weighted so they stop being presented as the current conclusion. Use whenever a previously stored conclusion no longer holds.',
      },
    },
    output: jsonOutput(),
    execute(args) {
      return core.remember({
        content: String(args.content ?? ''),
        kind: String(args.kind ?? 'note'),
        tags: Array.isArray(args.tags) ? args.tags.map(String) : undefined,
        scope: String(args.scope ?? 'project'),
        project: typeof args.project === 'string' && args.project !== '' ? args.project : undefined,
        importance: typeof args.importance === 'number' ? args.importance : undefined,
        ttlDays: typeof args.ttl_days === 'number' ? args.ttl_days : undefined,
        supersedes: Array.isArray(args.supersedes) ? args.supersedes.map(String) : undefined,
      });
    },
  }));

  ctx.tools.register(defineTool({
    name: 'memory_recall',
    description:
      'Search long-term memory. Run at task start with keywords from the user request; scores by relevance, importance, recency, and past usefulness. Also updates access stats so frequently used memories rank higher.',
    parameters: {
      query: { type: 'string', required: true, description: 'Keywords or a phrase, e.g. "deployment pipeline" or "user prefers".' },
      kinds: { type: 'array', items: { type: 'string', enum: [...MEMORY_KINDS] }, description: 'Optional kind filter.' },
      tags: { type: 'array', items: { type: 'string' }, description: 'Optional: only records carrying ALL these tags.' },
      scope: { type: 'string', enum: [...MEMORY_SCOPES], description: 'Optional scope filter.' },
      project: { type: 'string', description: 'Optional: only project-scoped records of this project (plus all user-scoped ones).' },
      limit: { type: 'number', description: 'Max results, default 3.' },
      content_max: { type: 'number', description: 'Optional: max chars of each result content shown to the model (default 400). Truncated results are enough to judge relevance; lower = cheaper calls.' },
    },
    output: jsonOutput(),
    execute(args) {
      return core.recall({
        query: String(args.query ?? ''),
        kinds: Array.isArray(args.kinds) ? args.kinds.map(String) : undefined,
        tags: Array.isArray(args.tags) ? args.tags.map(String) : undefined,
        scope: typeof args.scope === 'string' ? args.scope : undefined,
        project: typeof args.project === 'string' && args.project !== '' ? args.project : undefined,
        limit: typeof args.limit === 'number' ? args.limit : undefined,
        contentMax: typeof args.content_max === 'number' ? args.content_max : undefined,
      });
    },
  }));

  ctx.tools.register(defineTool({
    name: 'memory_index',
    description:
      'List the memory inventory (title-level entries, newest first) with counts by kind. Use to see what is already known without burning context on full recall.',
    parameters: {
      kinds: { type: 'array', items: { type: 'string', enum: [...MEMORY_KINDS] }, description: 'Optional kind filter.' },
      tags: { type: 'array', items: { type: 'string' }, description: 'Optional: only records carrying ALL these tags.' },
      scope: { type: 'string', enum: [...MEMORY_SCOPES], description: 'Optional scope filter.' },
      limit: { type: 'number', description: 'Max entries, default 20.' },
      offset: { type: 'number', description: 'Pagination offset, default 0.' },
    },
    output: jsonOutput(),
    execute(args) {
      return core.index({
        kinds: Array.isArray(args.kinds) ? args.kinds.map(String) : undefined,
        tags: Array.isArray(args.tags) ? args.tags.map(String) : undefined,
        scope: typeof args.scope === 'string' ? args.scope : undefined,
        limit: typeof args.limit === 'number' ? args.limit : undefined,
        offset: typeof args.offset === 'number' ? args.offset : undefined,
      });
    },
  }));

  ctx.tools.register(defineTool({
    name: 'memory_forget',
    description:
      'Delete memories: by exact id, or every record carrying ALL given tags (optionally scoped). Importance-3 records require confirm: true. Use to retire outdated or wrong memories.',
    parameters: {
      id: { type: 'string', description: 'Exact memory id from recall/index.' },
      tags: { type: 'array', items: { type: 'string' }, description: 'Delete all records with ALL these tags (used when id is omitted).' },
      scope: { type: 'string', enum: [...MEMORY_SCOPES], description: 'Optional scope filter for tag deletion.' },
      confirm: { type: 'boolean', description: 'Required to delete importance-3 records.' },
    },
    output: jsonOutput(),
    execute(args) {
      return core.forget({
        id: typeof args.id === 'string' && args.id !== '' ? args.id : undefined,
        tags: Array.isArray(args.tags) ? args.tags.map(String) : undefined,
        scope: typeof args.scope === 'string' ? args.scope : undefined,
        confirm: args.confirm === true,
      });
    },
  }));

  // memory_sediment: batch-persist worthy memories (session wind-down, after a
  // correction). The agent IS the LLM — it summarizes what it already has in
  // context, so there is zero extra model cost. Guardrails: max N entries per
  // call + a process-wide cooldown window keep noise from polluting the pool
  // (which would otherwise waste injection tokens on irrelevant hits).
  let lastSedimentAt = 0;
  ctx.tools.register(defineTool({
    name: 'memory_sediment',
    description:
      'Batch-persist durable memories at once (facts/decisions/lessons/todos), e.g. at session wind-down or right after a user correction. The agent summarizes what it already knows — no extra LLM call. Guardrails: max entries per call and a cooldown window prevent noise pollution.',
    parameters: {
      entries: {
        type: 'array', required: true,
        description: `1-${config.sedimentMaxEntries} memory entries. Each: {content (required), kind: fact|preference|decision|lesson|todo|note, tags: string[], scope: user|project, project: string, importance: 1-3}.`,
        items: {
          type: 'object',
          additionalProperties: false,
          properties: {
            content: { type: 'string' },
            kind: { type: 'string' },
            tags: { type: 'array', items: { type: 'string' } },
            scope: { type: 'string' },
            project: { type: 'string' },
            importance: { type: 'number' },
          },
        },
      },
    },
    output: jsonOutput(),
    execute: async (args) => {
      const now = Date.now();
      const cooldownMs = config.sedimentCooldownMs;
      const elapsed = now - lastSedimentAt;
      if (cooldownMs > 0 && elapsed < cooldownMs) {
        return { stored: 0, merged: 0, cooldown: true, nextInMs: cooldownMs - elapsed };
      }
      const rawEntries = Array.isArray(args.entries) ? args.entries : [];
      let stored = 0;
      let merged = 0;
      for (const item of rawEntries.slice(0, config.sedimentMaxEntries)) {
        const e = (item ?? {}) as {
          content?: unknown; kind?: unknown; tags?: unknown;
          scope?: unknown; project?: unknown; importance?: unknown;
        };
        const content = String(e.content ?? '').trim();
        if (content === '') continue;
        const r = await core.remember({
          content,
          kind: String(e.kind ?? 'note'),
          tags: Array.isArray(e.tags) ? e.tags.map(String) : undefined,
          scope: String(e.scope ?? 'project'),
          project: typeof e.project === 'string' && e.project !== '' ? e.project : undefined,
          importance: typeof e.importance === 'number' ? e.importance : undefined,
        });
        if (r.merged) merged += 1;
        else stored += 1;
      }
      if (cooldownMs > 0) lastSedimentAt = now;
      return { stored, merged, cooldown: false, nextInMs: 0 };
    },
  }));

  ctx.tools.register(defineTool({
    name: 'memory_reload',
    description:
      'Reopen the memory domain from disk. Use after memory.json was modified by something outside this DSH process (script, another instance, manual edit): the external changes are loaded and the in-memory store is replaced. Never loses data — but check the returned count.',
    parameters: {},
    output: jsonOutput(),
    execute: async () => {
      await domain.close();
      domain = await ctx.storageDomain.open(memoryDomain);
      core = makeCore();
      refreshFingerprint();
      const { total, byKind } = core.stats();
      return { reloaded: true, total, byKind };
    },
  }));

  ctx.tools.register(defineTool({
    name: 'memory_import',
    description:
      'Import memories from a JSONL or JSON file through the domain write chain (not by editing memory.json), so the file and the in-memory store stay in sync. Record shape per line: {"content":"...","kind":"fact|preference|decision|lesson|todo|note","tags":["..."],"scope":"user|project","importance":1-3}.',
    parameters: {
      file: { type: 'string', required: true, description: 'Absolute path to a .jsonl or .json file with memory records.' },
    },
    output: jsonOutput(),
    execute: async (args) => {
      const file = String(args.file ?? '');
      const raw = readFileSync(file, 'utf8').replace(/^\uFEFF/, '');
      const entries = raw.trimStart().startsWith('[')
        ? JSON.parse(raw)
        : raw.split(/\r?\n/).filter((l) => l.trim()).map((l) => JSON.parse(l));
      let added = 0; let skipped = 0;
      for (const entry of entries) {
        const content = String(entry?.content ?? entry?.summary ?? '').trim();
        if (!content) continue;
        const kind = isValidKind(String(entry?.kind ?? '')) ? String(entry?.kind) : 'note';
        const scope = isValidScope(String(entry?.scope ?? '')) ? String(entry?.scope) : 'project';
        const importance = Math.min(3, Math.max(1, Math.round(Number(entry?.importance) || 2)));
        const tags = normalizeTags(Array.isArray(entry?.tags) ? entry.tags.map(String) : undefined);
        const id = `mem_${randomUUID().replaceAll('-', '').slice(0, 16)}`;
        const now = iso(Date.now());
        if ((domain.table('records') as unknown as MemoryTable).get(id)) { skipped += 1; continue; }
        await core.remember({ content, kind, tags, scope, importance });
        added += 1;
      }
      const { total, byKind } = core.stats();
      return { imported: added, skipped, total, byKind };
    },
  }));

  if (config.protocolSection) {
    ctx.systemPrompt.section({ name: 'memory-protocol', order: 110, text: PROTOCOL_SECTION });
  }

  // Per-step dynamic injection: before each model step, recall the most
  // relevant memories for the user's current message and append them to the
  // step input. Replaces the old hot-memory broadcast (stateContext) —
  // relevant-on-demand instead of every-turn broadcast.
  //
  // Repeat suppression: consecutive steps about the same topic recall the same
  // top set; injecting it every time would spam identical blocks. We skip the
  // injection when the recalled id set is unchanged from the previous step.
  if (config.injectEnabled && config.injectCount > 0) {
    let lastInjectedKey = '';
    ctx.on('agent/pre-step', async (payload, next) => {
      const decision: PreStepDecision = await next();
      if (decision.kind !== 'enter') return decision;
      const query = queryFromMessages(payload.messages);
      if (query === '' || !hasMeaningfulQuery(query)) return decision;
      let rec: RecallResult;
      try {
        // Read-only recall: passive injection must not write (touch) the store.
        // Retrieve more than injectCount so we can filter by relevance.
        rec = await core.recall({ query, limit: Math.max(config.injectCount, 10), touch: false });
      } catch {
        return decision; // never break the agent loop over memory
      }
      if (rec.results.length === 0) return decision;
      // 「宁可不给,也不硬凑」:相关性没过闸门就保持沉默。
      // 硬凑一条无关记忆比不注入更糟 —— 它占预算,还把判断带偏。
      if (rec.noMatch) return decision;
      // 预算化选择:先过分数门槛,再按分数曲线决定"几条值得占注意力",
      // 最后才受 injectCount 上限约束(见 selectForInjection 的说明)。
      const filtered = selectForInjection(rec.results, {
        minScore: config.injectMinScore,
        budgetRatio: config.injectBudgetRatio,
        limit: config.injectCount,
      });
      if (filtered.length === 0) return decision; // nothing relevant enough
      const key = filtered.map((r) => r.id).sort().join(',');
      if (key === lastInjectedKey) return decision; // same topic, already injected
      lastInjectedKey = key;
      const text = renderInjection(filtered, config.injectMaxChars);
      const memMessage = createUserMessage({
        content: [{ type: 'text', text }],
        source: {
          kind: 'plugin',
          plugin: 'memory',
          form: 'notice',
          summary: boundContextSummary(`相关记忆 ${rec.results.length} 条`),
        },
      });
      return { ...decision, messages: [...decision.messages, memMessage] };
    });
  }

  // Lesson auto-solidification: the same error fingerprint recurring N times
  // means the agent keeps making the same mistake. Nudge it (via agent.inject,
  // queued for the next step) to write the mistake as an importance-3 lesson —
  // the lesson then auto-injects on related topics, closing the loop.
  if (config.lessonizeEnabled && config.lessonizeAfter > 0) {
    const errorCounts = new Map<string, number>();
    ctx.on('agent/error', async (payload) => {
      const fingerprint = extractErrorFingerprint(payload.error);
      if (fingerprint === 'unknown-error' || fingerprint === '') return;
      const n = (errorCounts.get(fingerprint) ?? 0) + 1;
      if (n < config.lessonizeAfter) { errorCounts.set(fingerprint, n); return; }
      errorCounts.delete(fingerprint); // nudged once per streak; reset after
      try {
        payload.agent.inject(createUserMessage({
          content: [{
            type: 'text',
            text: `检测到同类错误第 ${n} 次:${fingerprint}。请用 memory_remember 把这条经验固化为教训(kind=lesson, importance=3),并辩证总结:①具体错在哪、怎么规避 ②失败的条件是什么(换条件是否可行,标注适用边界) ③这次有没有其实有效的部分值得保留。防止再次发生,也避免把条件性失败记成绝对结论。`,
          }],
          source: {
            kind: 'plugin',
            plugin: 'memory',
            form: 'notice',
            summary: boundContextSummary(`同类错误 ${n} 次,建议固化教训`),
          },
        }));
      } catch {
        // injection is best-effort; never break the loop
      }
    });
  }
}
