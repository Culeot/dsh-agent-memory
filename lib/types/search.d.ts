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
export declare function tokenize(text: string): string[];
/** Unique token set. */
export declare function tokenSet(text: string): Set<string>;
/**
 * Bigram tokenizer: ASCII word runs + CJK bigrams (sliding windows of two
 * consecutive CJK chars; a CJK run shorter than 2 chars contributes single
 * chars). Bigrams carry far more information than single chars, which makes
 * Chinese recall noticeably more precise — "苹果" no longer partially matches
 * "水果摊" through the shared char "果".
 */
export declare function tokenizeBigram(text: string): string[];
export declare function tokenSetBigram(text: string): Set<string>;
/** Loose matching signal: pure CJK unigrams (never used alone). */
export declare function cjkUnigrams(text: string): Set<string>;
/** Jaccard similarity of two token sets, 0..1. */
export declare function jaccard(a: Set<string>, b: Set<string>): number;
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
/**
 * Minimum matching strength (base score) for a record to be considered a hit.
 * Base contributions: substring hit = 3, tag hit = 1.5, bigram Jaccard ≤ 2,
 * unigram Jaccard ≤ 0.8 (pure single-char coincidence can never reach 1.0).
 * A threshold of 1.0 therefore keeps every strong match (substring/tag/decent
 * bigram) while discarding pure single-char noise — the source of "unrelated
 * association" on short queries like "ok了吗".
 */
export declare const MATCH_BASE_MIN = 1;
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
export declare const IMPORTANCE_BOOST_SLOPE = 0.1;
/** Lexical base at which the lexical channel saturates when fused (maps base→0..1). */
export declare const LEXICAL_SATURATION = 4;
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
export declare const DEFAULT_SEMANTIC_WEIGHT = 0.3;
/** Minimum cosine for a record with no lexical hit to still count as a match. */
export declare const DEFAULT_SEMANTIC_MIN = 0.5;
/**
 * Score multiplier for a record that a newer memory has superseded.
 *
 * Superseded records are kept — auditable, and reversible by clearing
 * `supersededBy` — but they must stop being presented as the current
 * conclusion. 0.25 pushes them below fresh records in a normal candidate pool
 * without hiding them entirely.
 */
export declare const SUPERSEDED_PENALTY = 0.25;
/**
 * Whether a query carries enough retrieval information to bother searching.
 * Short/filler queries ("ok了吗", "可以吗") would otherwise surface weak
 * matches — pure token waste. Requires ≥2 meaningful CJK chars or ≥1 ASCII
 * word of length ≥3.
 */
export declare function hasMeaningfulQuery(query: string): boolean;
/**
 * BM25-style term-frequency signal (single-document collection, no idf).
 * Query bigram tokens are matched against content bigram frequencies with
 * length normalization: short content is rewarded for hits, long content is
 * not penalized for wordiness. Scaled and capped so it can only add up to
 * +2.0 to the base score.
 */
export declare function bm25Signal(query: string, content: string, avgLen?: number): number;
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
export declare function explainRecord(record: MemoryRecord, query: string, options?: RankOptions): ExplainResult;
/** Compatibility wrapper: plain score (existing API/tests keep working). */
export declare function scoreRecord(record: MemoryRecord, query: string, options?: RankOptions): number;
/** Retired records: expired by TTL. */
export declare function isExpired(record: MemoryRecord, now?: number): boolean;
/**
 * Hot-memory score (no query): how strongly one record belongs in the small
 * always-visible working set. recency (last activity: update OR recall hit) +
 * importance + access count. Recalling a record refreshes accessedAt, so
 * frequently used memories genuinely rise into the hot set on their own.
 */
export declare function hotnessScore(record: MemoryRecord, options?: RankOptions): number;
/**
 * The hot working set: top records by hotness, excluding expired ones.
 * This is the "hot memory" tier — the few entries worth injecting into every
 * session — while the full table remains the "memory bank" queried on demand.
 */
export declare function hotRecords(records: readonly MemoryRecord[], limit: number, options?: RankOptions): ScoredRecord[];
/**
 * Rank non-expired records against a query and return the top `limit`.
 * Records with score 0 are dropped; ties keep insertion order (stable).
 */
export declare function rankRecords(records: readonly MemoryRecord[], query: string, limit: number, options?: RankOptions): ScoredRecord[];
/**
 * Business boosts (importance × recency × access), applied AFTER relevance.
 * Kept separate so the fusion path can scale a relevance score that came from
 * either channel — in the lexical path the same product is folded into
 * {@link explainRecord}.
 */
export declare function recordBoost(record: MemoryRecord, options?: RankOptions, now?: number): number;
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
export declare function rankRecordsHybrid(records: readonly MemoryRecord[], query: string, limit: number, options?: HybridOptions): ScoredRecord[];
/**
 * How many records the hybrid ranker would admit as matches, before the limit
 * cut. Mirrors {@link rankRecordsHybrid}'s admission rule exactly: a record
 * counts when EITHER channel finds it (requiring a lexical hit would discard
 * the paraphrases the semantic channel exists to catch).
 */
export declare function countHybridMatches(records: readonly MemoryRecord[], query: string, options?: HybridOptions): number;
//# sourceMappingURL=search.d.ts.map