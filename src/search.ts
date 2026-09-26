/**
 * Zero-dependency retrieval for memory records. Memory entries are short
 * (<= 2000 chars), so the scoring model is deliberately simple and
 * explainable: exact substring + token overlap, weighted by importance,
 * recency (half-life decay), and access count. No vector database, no LLM
 * call, no embedding — works offline on every platform, and the whole index
 * is recomputed per query (400 records max keeps this cheap).
 *
 * @module dsh-agent-memory/search
 */
import type { MemoryRecord } from './spec.ts';

/** Split text into lowercase tokens: ASCII word runs + individual CJK chars. */
export function tokenize(text: string): string[] {
  const lower = text.toLowerCase();
  const tokens: string[] = [];
  const ascii = /[a-z0-9_]+/g;
  let match: RegExpExecArray | null;
  while ((match = ascii.exec(lower)) !== null) tokens.push(match[0]);
  for (const ch of lower) {
    const code = ch.codePointAt(0)!;
    if (code >= 0x4e00 && code <= 0x9fff) tokens.push(ch); // CJK unified
  }
  return tokens;
}

/** Unique token set. */
export function tokenSet(text: string): Set<string> {
  return new Set(tokenize(text));
}

/**
 * Bigram tokenizer: ASCII word runs + CJK bigrams (sliding windows of two
 * consecutive CJK chars; a CJK run shorter than 2 chars contributes single
 * chars). Bigrams carry far more information than single chars, which makes
 * Chinese recall noticeably more precise — "苹果" no longer partially matches
 * "水果摊" through the shared char "果".
 */
export function tokenizeBigram(text: string): string[] {
  const lower = text.toLowerCase();
  const tokens: string[] = [];
  const ascii = /[a-z0-9_]+/g;
  let match: RegExpExecArray | null;
  while ((match = ascii.exec(lower)) !== null) tokens.push(match[0]);
  const cjkRuns = lower.match(/[\u4e00-\u9fff]+/g) ?? [];
  for (const run of cjkRuns) {
    if (run.length === 1) {
      tokens.push(run);
    } else {
      for (let i = 0; i < run.length - 1; i += 1) tokens.push(run.slice(i, i + 2));
    }
  }
  return tokens;
}

export function tokenSetBigram(text: string): Set<string> {
  return new Set(tokenizeBigram(text));
}

/** Loose matching signal: pure CJK unigrams (never used alone). */
export function cjkUnigrams(text: string): Set<string> {
  const set = new Set<string>();
  const runs = text.toLowerCase().match(/[\u4e00-\u9fff]+/g) ?? [];
  for (const run of runs) for (const ch of run) set.add(ch);
  return set;
}

/** Jaccard similarity of two token sets, 0..1. */
export function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) return 0;
  let intersection = 0;
  for (const token of a) if (b.has(token)) intersection += 1;
  return intersection / (a.size + b.size - intersection);
}

export interface RankOptions {
  /** Current time as epoch ms; injectable for tests. */
  now?: number;
  /** Recency half-life in days. */
  recencyHalfLifeDays?: number;
  /**
   * Slope of the importance boost: boost = 1 + (importance - 1) * slope.
   *
   * Measured on a 382-query held-out set built from a real memory bank:
   * the historical slope 0.75 cost **-0.046 MRR** versus no boost at all
   * (0.7248 vs 0.7708), because 7 of 10 records sit at importance 3 — the
   * boost stops separating anything and only drags importance-2 records down.
   * Slope 0.25 recovers most of the loss (0.7968 within the fused pipeline)
   * while keeping the product intent. Pass 0 to disable entirely.
   */
  boostSlope?: number;
}

export interface ScoredRecord {
  record: MemoryRecord;
  score: number;
  /** Human-readable hit reasons (query ranking only; absent for hot set). */
  reasons?: string[];
  /** Lexical base before boosts (query ranking only; absent for the hot set). */
  base?: number;
  /**
   * Fused relevance before boosts, on a 0..1 scale (hybrid path only).
   * The "say nothing rather than something wrong" gate compares against this,
   * because boosts must not be able to talk the system into a match.
   */
  relevance?: number;
}

const ACCESS_BOOST_CAP = 1.5;

/** Most-recent activity time: updates and recall hits both refresh freshness. */
function lastActive(record: MemoryRecord, fallbackNow: number): number {
  const updated = Date.parse(record.updatedAt);
  if (record.accessedAt !== null) {
    const accessed = Date.parse(record.accessedAt);
    return accessed > updated ? accessed : updated;
  }
  return updated;
}

/**
 * Minimum matching strength (base score) for a record to be considered a hit.
 * Base contributions: substring hit = 3, tag hit = 1.5, bigram Jaccard ≤ 2,
 * unigram Jaccard ≤ 0.8 (pure single-char coincidence can never reach 1.0).
 * A threshold of 1.0 therefore keeps every strong match (substring/tag/decent
 * bigram) while discarding pure single-char noise — the source of "unrelated
 * association" on short queries like "ok了吗".
 */
export const MATCH_BASE_MIN = 1.0;

/**
 * Default slope of the importance boost (see {@link RankOptions.boostSlope}).
 *
 * Calibrated on a 382-query held-out set, sweeping w × slope together
 * (`src/mem_tune_final.py` in the jev-as-llm project):
 *
 *     slope 0.00 → full MRR 0.8358   R@1 0.7513
 *     slope 0.10 → full MRR 0.8320   R@1 0.7539   ← chosen (near-lossless, R@1 best)
 *     slope 0.25 → full MRR 0.8032   R@1 0.7042
 *     slope 0.75 → full MRR 0.7570   R@1 0.6466   (the historical value)
 *
 * 0.1 keeps the product intent ("important memories rank a little higher")
 * at a cost of 0.004 MRR; anything above 0.25 starts replacing relevance with
 * importance rather than nudging it.
 */
export const IMPORTANCE_BOOST_SLOPE = 0.1;

/** Lexical base at which the lexical channel saturates when fused (maps base→0..1). */
export const LEXICAL_SATURATION = 4.0;

/**
 * Default weight of the semantic channel: relevance = w*semantic + (1-w)*lexical.
 *
 * Swept 0..1 on the 382-query set — the optimum sits near **0.3**, not the 0.65
 * an earlier evaluation suggested. The two evaluations are NOT comparable:
 * this module normalizes linearly (`min(1, base/4)` and a clamped cosine) while
 * that script used z-scores, so the same model at the same w yields different
 * numbers under each formula. Always compare within one formula.
 *
 *     w 0.0  → 0.7006   (lexical only, truncated by the saturation cap)
 *     w 0.3  → 0.8358   ← chosen
 *     w 0.5  → 0.8348
 *     w 0.65 → 0.8079
 *     w 1.0  → 0.7121   (semantic only)
 */
export const DEFAULT_SEMANTIC_WEIGHT = 0.3;

/** Minimum cosine for a record with no lexical hit to still count as a match. */
export const DEFAULT_SEMANTIC_MIN = 0.5;

/**
 * Score multiplier for a record that a newer memory has superseded.
 *
 * Superseded records are kept — auditable, and reversible by clearing
 * `supersededBy` — but they must stop being presented as the current
 * conclusion. 0.25 pushes them below fresh records in a normal candidate pool
 * without hiding them entirely.
 */
export const SUPERSEDED_PENALTY = 0.25;

function clamp01(x: number): number {
  if (!Number.isFinite(x)) return 0;
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

/** Common filler words that carry no retrieval information. */
const STOPWORDS = new Set([
  '的', '了', '吗', '呢', '啊', '哦', '嗯', '哟', '吧', '呀', '哈', '嘿', '喂',
  '是', '在', '和', '与', '或', '及', '把', '被', '给', '向', '从', '到', '于', '对', '就', '都', '也', '还', '但', '而', '则', '且',
  '我', '你', '他', '她', '它', '咱', '您', '们',
  '这', '那', '哪', '什', '么', '怎', '为', '何', '啥',
  '可', '以', '能', '好', '不', '别', '请', '先', '再', '又', '很', '太', '真', '挺', '会', '想', '要', '帮', '看', '说', '做', '弄', '搞', '整',
  'ok', 'ok了', '好的', '嗯嗯', '哈哈', '谢谢', '感谢', '谢了', '可以了', '明白了', '懂了', '知道', '看看', '请问',
]);

/**
 * Whether a query carries enough retrieval information to bother searching.
 * Short/filler queries ("ok了吗", "可以吗") would otherwise surface weak
 * matches — pure token waste. Requires ≥2 meaningful CJK chars or ≥1 ASCII
 * word of length ≥3.
 */
export function hasMeaningfulQuery(query: string): boolean {
  let meaningfulCjk = 0;
  for (const ch of query) {
    const code = ch.codePointAt(0)!;
    if (code >= 0x4e00 && code <= 0x9fff && !STOPWORDS.has(ch)) meaningfulCjk += 1;
  }
  const asciiWords = query.toLowerCase().match(/[a-z]{3,}/g);
  return meaningfulCjk >= 2 || (asciiWords !== null && asciiWords.length >= 1);
}

/**
 * BM25-style term-frequency signal (single-document collection, no idf).
 * Query bigram tokens are matched against content bigram frequencies with
 * length normalization: short content is rewarded for hits, long content is
 * not penalized for wordiness. Scaled and capped so it can only add up to
 * +2.0 to the base score.
 */
export function bm25Signal(query: string, content: string, avgLen = 40): number {
  const qTokens = [...new Set(tokenizeBigram(query))];
  if (qTokens.length === 0) return 0;
  const cTokens = tokenizeBigram(content);
  const contentLen = Math.max(1, cTokens.length);
  const freq = new Map<string, number>();
  for (const t of cTokens) freq.set(t, (freq.get(t) ?? 0) + 1);
  const k1 = 1.2;
  const b = 0.75;
  let sum = 0;
  for (const t of qTokens) {
    const tf = freq.get(t) ?? 0;
    if (tf === 0) continue;
    const norm = contentLen / avgLen;
    sum += (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * norm));
  }
  return Math.min(2.0, sum * 0.4);
}

export interface ExplainResult {
  score: number;
  /** Human-readable reasons, best-contributing first. */
  reasons: string[];
  /** Lexical base before boosts (0 when the match is below MATCH_BASE_MIN). */
  base: number;
}

/**
 * Score one record against a query and explain why it scored. 0 when the
 * query is empty or the match is too weak to be meaningful.
 * score = base * importanceBoost * recencyBoost * accessBoost
 * where base = 3*substring + 1.5*tag + 1.2*qwordHits + 2*bigramJaccard +
 * 0.8*unigramJaccard + bm25, and base < MATCH_BASE_MIN is no match.
 * Reasons mirror the exact contributions, so both the user and the agent can
 * audit "why did this memory surface".
 */
export function explainRecord(record: MemoryRecord, query: string, options: RankOptions = {}): ExplainResult {
  const q = query.trim();
  if (q === '') return { score: 0, reasons: [], base: 0 };

  const contentLower = record.content.toLowerCase();
  const queryLower = q.toLowerCase();
  const reasons: string[] = [];
  const baseParts: string[] = [];
  let base = 0;

  if (contentLower.includes(queryLower)) {
    base += 3;
    baseParts.push('substring(3.0)');
    reasons.push('substring:整串命中(+3.0)');
  }
  const matchedTags = record.tags.filter((tag) => tag.toLowerCase().includes(queryLower));
  if (matchedTags.length > 0) {
    base += 1.5;
    baseParts.push('tag(1.5)');
    reasons.push(`tag:${matchedTags.join('/')} 命中(+1.5)`);
  }

  const qWords = new Set(q.match(/[\u4e00-\u9fff]{2,}|[a-z][a-z0-9_]{2,}/gi) ?? []);
  let qwordHits = 0;
  for (const w of qWords) {
    if (contentLower.includes(w.toLowerCase())) {
      base += 1.2;
      qwordHits += 1;
      reasons.push(`qword:${w}(+1.2)`);
    }
  }
  if (qwordHits > 0) baseParts.push(`qword(${qwordHits}×1.2)`);

  const qBigrams = tokenSetBigram(q);
  const cBigrams = tokenSetBigram(record.content);
  const bigramJ = jaccard(qBigrams, cBigrams);
  if (bigramJ > 0) {
    base += bigramJ * 2;
    baseParts.push(`bigram(${(bigramJ * 2).toFixed(2)})`);
    reasons.push(`bigram:${bigramJ.toFixed(2)}(+${(bigramJ * 2).toFixed(2)})`);
  }

  const qUnis = cjkUnigrams(q);
  const cUnis = cjkUnigrams(record.content);
  const uniJ = jaccard(qUnis, cUnis);
  if (uniJ > 0) {
    base += uniJ * 0.8;
    baseParts.push(`unigram(${(uniJ * 0.8).toFixed(2)})`);
    reasons.push(`unigram:${uniJ.toFixed(2)}(+${(uniJ * 0.8).toFixed(2)})`);
  }

  const bm = bm25Signal(q, record.content);
  if (bm > 0) {
    base += bm;
    baseParts.push(`bm25(${bm.toFixed(2)})`);
    reasons.push(`bm25:${bm.toFixed(2)}(+${bm.toFixed(2)})`);
  }

  if (base < MATCH_BASE_MIN) return { score: 0, reasons, base: 0 };

  const slope = options.boostSlope ?? IMPORTANCE_BOOST_SLOPE;
  const importanceBoost = 1 + (record.importance - 1) * slope;
  const halfLifeDays = options.recencyHalfLifeDays ?? 90;
  const now = options.now ?? Date.now();
  const ageMs = now - lastActive(record, now);
  const recencyBoost = Math.pow(0.5, ageMs / (halfLifeDays * 24 * 3600 * 1000));
  const accessBoost = Math.min(ACCESS_BOOST_CAP, 1 + Math.log(1 + record.accessCount) * 0.15);

  const supersededBoost = record.supersededBy ? SUPERSEDED_PENALTY : 1;
  const score = base * importanceBoost * recencyBoost * accessBoost * supersededBoost;
  if (importanceBoost > 1) reasons.push(`importance:${record.importance}(×${importanceBoost.toFixed(2)})`);
  const ageDays = ageMs / (24 * 3600 * 1000);
  reasons.push(`recency:${ageDays.toFixed(1)} 天前活跃(×${recencyBoost.toFixed(2)})`);
  if (record.accessCount > 0) reasons.push(`access:${record.accessCount} 次(×${accessBoost.toFixed(2)})`);
  if (baseParts.length > 0) reasons.unshift(`base:${baseParts.join('+')}`);
  // 「被取代」必须排在最前面:它是最该被看见的事实,而 reasons 只保留前 8 条 ——
  // 排在末尾等于被丢掉(实测:它正是被 slice 砍掉的那一条,测试因此失败)。
  if (record.supersededBy !== null && record.supersededBy !== undefined) {
    reasons.unshift(`superseded:已被 ${record.supersededBy} 取代(×${SUPERSEDED_PENALTY})`);
  }
  return { score, reasons: reasons.slice(0, 8), base };
}

/** Compatibility wrapper: plain score (existing API/tests keep working). */
export function scoreRecord(record: MemoryRecord, query: string, options: RankOptions = {}): number {
  return explainRecord(record, query, options).score;
}

/** Retired records: expired by TTL. */
export function isExpired(record: MemoryRecord, now: number = Date.now()): boolean {
  return record.expiresAt !== null && Date.parse(record.expiresAt) <= now;
}

/**
 * Hot-memory score (no query): how strongly one record belongs in the small
 * always-visible working set. recency (last activity: update OR recall hit) +
 * importance + access count. Recalling a record refreshes accessedAt, so
 * frequently used memories genuinely rise into the hot set on their own.
 */
export function hotnessScore(record: MemoryRecord, options: RankOptions = {}): number {
  const halfLifeDays = options.recencyHalfLifeDays ?? 90;
  const now = options.now ?? Date.now();
  const ageMs = now - lastActive(record, now);
  const recencyBoost = Math.pow(0.5, ageMs / (halfLifeDays * 24 * 3600 * 1000));
  const importanceBoost = 1 + (record.importance - 1) * (options.boostSlope ?? IMPORTANCE_BOOST_SLOPE);
  const accessBoost = Math.min(ACCESS_BOOST_CAP, 1 + Math.log(1 + record.accessCount) * 0.15);
  return recencyBoost * importanceBoost * accessBoost;
}

/**
 * The hot working set: top records by hotness, excluding expired ones.
 * This is the "hot memory" tier — the few entries worth injecting into every
 * session — while the full table remains the "memory bank" queried on demand.
 */
export function hotRecords(records: readonly MemoryRecord[], limit: number, options: RankOptions = {}): ScoredRecord[] {
  const now = options.now ?? Date.now();
  const scored: ScoredRecord[] = [];
  for (const record of records) {
    if (isExpired(record, now)) continue;
    scored.push({ record, score: hotnessScore(record, { ...options, now }) });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, Math.max(0, limit));
}

/**
 * Rank non-expired records against a query and return the top `limit`.
 * Records with score 0 are dropped; ties keep insertion order (stable).
 */
export function rankRecords(records: readonly MemoryRecord[], query: string, limit: number, options: RankOptions = {}): ScoredRecord[] {
  const now = options.now ?? Date.now();
  const scored: ScoredRecord[] = [];
  for (const record of records) {
    if (isExpired(record, now)) continue;
    const explained = explainRecord(record, query, { ...options, now });
    if (explained.score > 0) scored.push({ record, score: explained.score, reasons: explained.reasons, base: explained.base });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, Math.max(0, limit));
}

/**
 * Business boosts (importance × recency × access), applied AFTER relevance.
 * Kept separate so the fusion path can scale a relevance score that came from
 * either channel — in the lexical path the same product is folded into
 * {@link explainRecord}.
 */
export function recordBoost(record: MemoryRecord, options: RankOptions = {}, now: number = Date.now()): number {
  const slope = options.boostSlope ?? IMPORTANCE_BOOST_SLOPE;
  const importanceBoost = 1 + (record.importance - 1) * slope;
  const halfLifeDays = options.recencyHalfLifeDays ?? 90;
  const ageMs = now - lastActive(record, now);
  const recencyBoost = Math.pow(0.5, ageMs / (halfLifeDays * 24 * 3600 * 1000));
  const accessBoost = Math.min(ACCESS_BOOST_CAP, 1 + Math.log(1 + record.accessCount) * 0.15);
  return importanceBoost * recencyBoost * accessBoost;
}

/** Options for hybrid (lexical + semantic) ranking. */
export interface HybridOptions extends RankOptions {
  /** Per-record cosine similarity in the same order as `records`; null/absent = lexical only. */
  semantic?: readonly number[] | null;
  /** Weight of the semantic channel: relevance = w*semantic + (1-w)*lexical. */
  semanticWeight?: number;
  /** Minimum cosine for a record with no lexical hit to still count as a match. */
  semanticMin?: number;
}

/**
 * Rank records with the lexical and semantic channels fused.
 *
 * Why fuse instead of crowning a winner — measured on this memory bank
 * (382 held-out queries): neither channel wins alone. Lexical dominates
 * keyword-shaped and term-shaped queries (MRR 0.9505 / 0.9504 versus the
 * model's 0.8853 / 0.8181), the model dominates conversational ones
 * (0.7396 vs 0.6267). Fused: **0.8432**, above both.
 *
 * The channels are normalized before mixing because their raw scales are
 * unrelated (lexical base runs 0..~10, cosine -1..1): lexical saturates at
 * {@link LEXICAL_SATURATION}, cosine is clamped to 0..1. A record counts as a
 * match when either channel finds it — requiring a lexical hit would throw
 * away exactly the paraphrases the semantic channel exists to catch.
 *
 * When `semantic` is absent or the wrong length this degrades to
 * {@link rankRecords}, so callers can pass a channel that failed to load.
 */
export function rankRecordsHybrid(
  records: readonly MemoryRecord[],
  query: string,
  limit: number,
  options: HybridOptions = {},
): ScoredRecord[] {
  const semantic = options.semantic;
  if (!semantic || semantic.length !== records.length) {
    return rankRecords(records, query, limit, options);
  }
  const now = options.now ?? Date.now();
  const w = clamp01(options.semanticWeight ?? DEFAULT_SEMANTIC_WEIGHT);
  const semMin = options.semanticMin ?? DEFAULT_SEMANTIC_MIN;
  const scored: ScoredRecord[] = [];
  for (let i = 0; i < records.length; i += 1) {
    const record = records[i]!;
    if (isExpired(record, now)) continue;
    const explained = explainRecord(record, query, { ...options, now });
    const cos = clamp01(semantic[i]!);
    const lexicalHit = explained.base >= MATCH_BASE_MIN;
    if (!lexicalHit && cos < semMin) continue; // neither channel finds it
    const lexPart = clamp01(explained.base / LEXICAL_SATURATION);
    const relevance = w * cos + (1 - w) * lexPart;
    const penalty = record.supersededBy ? SUPERSEDED_PENALTY : 1;
    const score = relevance * recordBoost(record, options, now) * penalty;
    const reasons = [
      `fuse:sem=${cos.toFixed(2)} lex=${lexPart.toFixed(2)} w=${w.toFixed(2)}`,
      ...(record.supersededBy ? [`superseded:已被 ${record.supersededBy} 取代(×${SUPERSEDED_PENALTY})`] : []),
      ...explained.reasons,
    ].slice(0, 8);
    if (score > 0) scored.push({ record, score, reasons, base: explained.base, relevance });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, Math.max(0, limit));
}

/**
 * How many records the hybrid ranker would admit as matches, before the limit
 * cut. Mirrors {@link rankRecordsHybrid}'s admission rule exactly: a record
 * counts when EITHER channel finds it (requiring a lexical hit would discard
 * the paraphrases the semantic channel exists to catch).
 */
export function countHybridMatches(
  records: readonly MemoryRecord[],
  query: string,
  options: HybridOptions = {},
): number {
  const semantic = options.semantic;
  const now = options.now ?? Date.now();
  const semMin = options.semanticMin ?? DEFAULT_SEMANTIC_MIN;
  let count = 0;
  for (let i = 0; i < records.length; i += 1) {
    const record = records[i]!;
    if (isExpired(record, now)) continue;
    if (explainRecord(record, query, { ...options, now }).base >= MATCH_BASE_MIN) {
      count += 1;
      continue;
    }
    if (semantic && semantic.length === records.length && clamp01(semantic[i]!) >= semMin) count += 1;
  }
  return count;
}
