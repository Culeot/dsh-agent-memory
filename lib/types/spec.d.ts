import { z } from 'zod';
/** Memory kinds: what a record is FOR. Drives recall filters and governance. */
export declare const MEMORY_KINDS: readonly ["fact", "preference", "decision", "lesson", "todo", "note"];
export type MemoryKind = (typeof MEMORY_KINDS)[number];
export declare const MEMORY_SCOPES: readonly ["user", "project"];
export type MemoryScope = (typeof MEMORY_SCOPES)[number];
/** One durable memory record. */
export declare const MemoryRecordSchema: z.ZodObject<{
    id: z.ZodString;
    content: z.ZodString;
    kind: z.ZodEnum<{
        fact: "fact";
        preference: "preference";
        decision: "decision";
        lesson: "lesson";
        todo: "todo";
        note: "note";
    }>;
    tags: z.ZodArray<z.ZodString>;
    scope: z.ZodEnum<{
        user: "user";
        project: "project";
    }>;
    project: z.ZodNullable<z.ZodString>;
    importance: z.ZodNumber;
    createdAt: z.ZodString;
    updatedAt: z.ZodString;
    accessedAt: z.ZodNullable<z.ZodString>;
    accessCount: z.ZodNumber;
    expiresAt: z.ZodNullable<z.ZodString>;
    supersededBy: z.ZodOptional<z.ZodNullable<z.ZodString>>;
}, z.core.$strip>;
export type MemoryRecord = z.infer<typeof MemoryRecordSchema>;
/** The domain spec. `version` stamps the medium; bump on schema changes. */
export declare const memoryDomain: {
    name: string;
    version: number;
    tables: {
        records: import("@deepseek-ai/dsh-storage-domain").DomainTableSpec<`mem_${string}`, {
            id: string;
            content: string;
            kind: "fact" | "preference" | "decision" | "lesson" | "todo" | "note";
            tags: string[];
            scope: "user" | "project";
            project: string | null;
            importance: number;
            createdAt: string;
            updatedAt: string;
            accessedAt: string | null;
            accessCount: number;
            expiresAt: string | null;
            supersededBy?: string | null | undefined;
        }>;
    };
};
/** Governance limits. Bumped values are validated against these in tests. */
export declare const MEMORY_DEFAULTS: {
    /** Hard cap on stored records; evicts lowest-value records first. */
    readonly maxRecords: 400;
    /** UTF-16 code units per record content; longer input is truncated. */
    readonly maxContentChars: 2000;
    /** Jaccard token similarity at/above which a new record merges into an old one.
     *  0.7 fits single-char CJK tokenization (each added char dilutes the set). */
    readonly mergeSimilarity: 0.7;
    /** Half-life in days for recency scoring. */
    readonly recencyHalfLifeDays: 90;
    /** Default recall result count (fewer hits = cheaper per call). */
    readonly recallLimit: 3;
    /** Default max chars of a recalled record content shown to the model.
     *  Keeps on-demand recall cheap; 400 chars is enough to judge relevance. */
    readonly recallContentMax: 400;
    /** memory_sediment: max entries per call (token noise guard). */
    readonly sedimentMaxEntries: 3;
    /** memory_sediment: min ms between calls across the process; 0 disables. */
    readonly sedimentCooldownMs: 300000;
};
export declare function isValidKind(value: string): value is MemoryKind;
export declare function isValidScope(value: string): value is MemoryScope;
//# sourceMappingURL=spec.d.ts.map