import type { Context } from '@deepseek-ai/cordis';
import z from '@deepseek-ai/schemastery';
import type { UserMessage } from '@deepseek-ai/dsh-session';
import { type MemoryRecord, type MemoryKind, type MemoryScope } from './spec.ts';
import { type SemanticConfig } from './semantic.ts';
export * from './spec.ts';
export * from './search.ts';
export * from './semantic.ts';
export * from './rpc.ts';
export declare const name = "memory";
export declare const inject: string[];
/** Schemastery config (same convention as the official dsh-tool-* plugins). */
export declare const Config: z<Schemastery.ObjectS<{
    maxRecords: z<number, number>;
    maxContentChars: z<number, number>;
    recencyHalfLifeDays: z<number, number>;
    mergeSimilarity: z<number, number>;
    /** Register the memory-protocol system-prompt section. */
    protocolSection: z<boolean, boolean>;
    /**
     * Per-step dynamic injection: before each model step, recall the most
     * relevant memories for the user's current message and append them to the
     * step input. Replaces the old every-turn hot-memory broadcast (stateContext)
     * — relevant-on-demand instead of broadcast.
     */
    injectEnabled: z<boolean, boolean>;
    /** Maximum memories to inject per step (0 disables injection). Actual count depends on relevance. */
    injectCount: z<number, number>;
    /**
     * Budgeted injection: keep the top memory plus any scoring within this
     * fraction of it. 0 = off (historical behaviour: score floor + fixed cap).
     * A clear winner then injects one memory instead of three mediocre ones.
     */
    injectBudgetRatio: z<number, number>;
    /** Minimum score threshold for injection (0 = no threshold, just use rank). */
    injectMinScore: z<number, number>;
    /** Max content chars per injected memory summary. */
    injectMaxChars: z<number, number>;
    /** Default max chars of a recalled record's content shown to the model. */
    recallContentMax: z<number, number>;
    /** memory_sediment: max entries per call. */
    sedimentMaxEntries: z<number, number>;
    /** memory_sediment: min ms between calls; 0 disables the cooldown. */
    sedimentCooldownMs: z<number, number>;
    /**
     * Lesson auto-solidification: when the SAME error fingerprint (code/message)
     * fires this many times, inject a nudge telling the agent to write it as an
     * importance-3 lesson. The lesson then auto-injects on related topics,
     * preventing recurrence. 0 disables.
     */
    lessonizeEnabled: z<boolean, boolean>;
    lessonizeAfter: z<number, number>;
    /**
     * Semantic channel: fuse a local sentence-embedding model with the lexical
     * scorer. Needs @huggingface/transformers (an OPTIONAL peer dependency) plus
     * a model directory; when either is missing, recall silently falls back to
     * lexical-only — an optional accelerator must never break the plugin.
     *
     * Measured on a real 382-record bank: lexical 0.7248 MRR → fused 0.8432.
     */
    semanticEnabled: z<boolean, boolean>;
    /** Model directory (config.json / tokenizer.json / onnx/*.onnx). '' disables. */
    semanticModelDir: z<string, string>;
    /** Vector-cache directory. '' = $DSH_HOME/storages/memory-semantic. */
    semanticCacheDir: z<string, string>;
    /** Semantic weight in fusion: relevance = w*semantic + (1-w)*lexical. */
    semanticWeight: z<number, number>;
    /** Minimum cosine for a record to count as a match with no lexical hit. */
    semanticMin: z<number, number>;
    /**
     * Slope of the importance boost. The historical 0.75 measured -0.046 MRR
     * versus no boost at all (most records sit at importance 3, so the boost
     * separates nothing while punishing the rest). 0 disables it.
     */
    boostSlope: z<number, number>;
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
    noMatchThreshold: z<number, number>;
}>, Schemastery.ObjectT<{
    maxRecords: z<number, number>;
    maxContentChars: z<number, number>;
    recencyHalfLifeDays: z<number, number>;
    mergeSimilarity: z<number, number>;
    /** Register the memory-protocol system-prompt section. */
    protocolSection: z<boolean, boolean>;
    /**
     * Per-step dynamic injection: before each model step, recall the most
     * relevant memories for the user's current message and append them to the
     * step input. Replaces the old every-turn hot-memory broadcast (stateContext)
     * — relevant-on-demand instead of broadcast.
     */
    injectEnabled: z<boolean, boolean>;
    /** Maximum memories to inject per step (0 disables injection). Actual count depends on relevance. */
    injectCount: z<number, number>;
    /**
     * Budgeted injection: keep the top memory plus any scoring within this
     * fraction of it. 0 = off (historical behaviour: score floor + fixed cap).
     * A clear winner then injects one memory instead of three mediocre ones.
     */
    injectBudgetRatio: z<number, number>;
    /** Minimum score threshold for injection (0 = no threshold, just use rank). */
    injectMinScore: z<number, number>;
    /** Max content chars per injected memory summary. */
    injectMaxChars: z<number, number>;
    /** Default max chars of a recalled record's content shown to the model. */
    recallContentMax: z<number, number>;
    /** memory_sediment: max entries per call. */
    sedimentMaxEntries: z<number, number>;
    /** memory_sediment: min ms between calls; 0 disables the cooldown. */
    sedimentCooldownMs: z<number, number>;
    /**
     * Lesson auto-solidification: when the SAME error fingerprint (code/message)
     * fires this many times, inject a nudge telling the agent to write it as an
     * importance-3 lesson. The lesson then auto-injects on related topics,
     * preventing recurrence. 0 disables.
     */
    lessonizeEnabled: z<boolean, boolean>;
    lessonizeAfter: z<number, number>;
    /**
     * Semantic channel: fuse a local sentence-embedding model with the lexical
     * scorer. Needs @huggingface/transformers (an OPTIONAL peer dependency) plus
     * a model directory; when either is missing, recall silently falls back to
     * lexical-only — an optional accelerator must never break the plugin.
     *
     * Measured on a real 382-record bank: lexical 0.7248 MRR → fused 0.8432.
     */
    semanticEnabled: z<boolean, boolean>;
    /** Model directory (config.json / tokenizer.json / onnx/*.onnx). '' disables. */
    semanticModelDir: z<string, string>;
    /** Vector-cache directory. '' = $DSH_HOME/storages/memory-semantic. */
    semanticCacheDir: z<string, string>;
    /** Semantic weight in fusion: relevance = w*semantic + (1-w)*lexical. */
    semanticWeight: z<number, number>;
    /** Minimum cosine for a record to count as a match with no lexical hit. */
    semanticMin: z<number, number>;
    /**
     * Slope of the importance boost. The historical 0.75 measured -0.046 MRR
     * versus no boost at all (most records sit at importance 3, so the boost
     * separates nothing while punishing the rest). 0 disables it.
     */
    boostSlope: z<number, number>;
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
    noMatchThreshold: z<number, number>;
}>>;
/** Extract a search query from the user messages claimed by this step. */
export declare function queryFromMessages(messages: readonly UserMessage[]): string;
/** Render recalled memories into a compact injection block. */
export declare function renderInjection(results: RecallResult['results'], maxChars: number): string;
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
export declare function selectForInjection<T extends {
    score: number;
}>(results: readonly T[], options: {
    minScore?: number;
    budgetRatio?: number;
    limit: number;
}): T[];
/**
 * Stable fingerprint of an error for same-mistake counting. Prefer a typed
 * code; fall back to the first meaningful line of the message.
 */
export declare function extractErrorFingerprint(error: unknown): string;
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
export declare class MemoryCore {
    private readonly table;
    private readonly config;
    private readonly beforeWrite?;
    /**
     * Called after EVERY durable write lands (each put/update/delete). The
     * external-modification guard compares fingerprints before a write, so the
     * baseline must be refreshed after each real write — not after a whole
     * multi-write operation (import) — otherwise the second write of a batch
     * is falsely rejected as "externally modified".
     */
    private readonly onWritten?;
    /**
     * Optional semantic channel. Null when disabled or when no model directory is
     * configured; recall then behaves exactly as it did before this existed.
     */
    private readonly semantic;
    constructor(table: MemoryTable, config: {
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
    }, beforeWrite?: (() => void) | undefined, 
    /**
     * Called after EVERY durable write lands (each put/update/delete). The
     * external-modification guard compares fingerprints before a write, so the
     * baseline must be refreshed after each real write — not after a whole
     * multi-write operation (import) — otherwise the second write of a batch
     * is falsely rejected as "externally modified".
     */
    onWritten?: (() => void) | undefined);
    /** Semantic-channel diagnostics, surfaced to the UI/RPC layer. */
    semanticStatus(): {
        enabled: boolean;
        ready: boolean;
        reason: string | null;
        backend: string | null;
        vectors: number;
    };
    /**
     * Cosine scores for the pool, or null when the semantic channel is off.
     * Never throws: an optional channel that fails must degrade, not break.
     */
    private semanticScores;
    private all;
    private guardWrite;
    private put;
    private update;
    private delete;
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
    private markSuperseded;
    /** Delete every TTL-expired record. Returns the number removed. */
    purgeExpired(now?: number): Promise<number>;
    /**
     * Evict lowest-value records until the table fits maxRecords. importance-3
     * records are eviction-proof unless the whole store is full of them — in
     * that case the oldest importance-3 record goes, so the store can never
     * deadlock while still honoring "critical survives space pressure".
     */
    private evict;
    /** Insert or merge one memory. */
    remember(input: {
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
    }): Promise<RememberResult>;
    /**
     * Ranked recall with kind/tag/scope filters and access tracking.
     * `touch: false` skips the access-count write-back entirely — used by the
     * per-step injection path so passive recall never triggers a disk write.
     */
    recall(input: {
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
    }): Promise<RecallResult>;
    /** Read-only single record lookup (used by the UI RPC layer). */
    getById(id: string): MemoryRecord | undefined;
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
    }): {
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
            expiresAt: string | null;
        }>;
    };
    /** Full inventory with stats (title-level listing to bound token cost). */
    index(input: {
        kinds?: readonly string[];
        tags?: readonly string[];
        scope?: string;
        limit?: number;
        offset?: number;
        now?: number;
    }): Promise<IndexResult>;
    /** Delete by id, or by tag set (optionally scoped). importance-3 ids need confirm. */
    forget(input: {
        id?: string;
        tags?: readonly string[];
        scope?: string;
        confirm?: boolean;
        now?: number;
    }): Promise<ForgetResult>;
    /** Manual edit from the UI panel: overwrite content/tags/importance directly. */
    updateContent(input: {
        id: string;
        content?: string;
        tags?: readonly string[];
        importance?: number;
        now?: number;
    }): Promise<{
        id: string;
        updated: boolean;
    }>;
    stats(): {
        total: number;
        byKind: Record<string, number>;
    };
}
export declare function apply(ctx: Context, config: Schemastery.TypeT<typeof Config>): Promise<void>;
//# sourceMappingURL=index.d.ts.map