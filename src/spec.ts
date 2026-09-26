/**
 * Memory domain declaration: the single source of the `memory` domain's
 * identity, layout, and record schemas. Persisted through the harness storage
 * hub (ctx.storage → ctx.storageDomain), so any configured backend works
 * (the stock `json` backend keeps everything in plain JSON files under
 * $DSH_HOME/storages — human-inspectable, git-friendly, zero dependencies).
 *
 * @module dsh-agent-memory/spec
 */
import { defineDomain, domainTable } from '@deepseek-ai/dsh-storage-domain';
import { z } from 'zod';

/** Memory kinds: what a record is FOR. Drives recall filters and governance. */
export const MEMORY_KINDS = [
  'fact', // verifiable statement about the world or the user's environment
  'preference', // what the user likes/wants/how they want things done
  'decision', // a concluded choice with its rationale (checkpoint, not a todo)
  'lesson', // a mistake/insight worth keeping ("don't do X again")
  'todo', // cross-session follow-up item
  'note', // anything else
] as const;
export type MemoryKind = (typeof MEMORY_KINDS)[number];

export const MEMORY_SCOPES = ['user', 'project'] as const;
export type MemoryScope = (typeof MEMORY_SCOPES)[number];

/** One durable memory record. */
export const MemoryRecordSchema = z.object({
  id: z.string(),
  content: z.string(),
  kind: z.enum(MEMORY_KINDS),
  tags: z.array(z.string()),
  scope: z.enum(MEMORY_SCOPES),
  /** Project root path this record belongs to; null for user-scope records. */
  project: z.string().nullable(),
  /** 1 = nice-to-know, 2 = useful, 3 = critical (protected from eviction). */
  importance: z.number().int().min(1).max(3),
  createdAt: z.string(),
  updatedAt: z.string(),
  accessedAt: z.string().nullable(),
  accessCount: z.number().int().min(0),
  /** ISO timestamp after which the record is retired from recall; null = never. */
  expiresAt: z.string().nullable(),
  /**
   * Id of the record that replaces this one, when a newer memory supersedes it.
   *
   * This is the "不腐烂" mechanism: the protocol has always *asked* the model to
   * write "已由 X 更新" into the body when a decision changes, which relies on
   * discipline and leaves the stale record ranking as high as the fresh one.
   * Making the relation data lets the ranker down-weight it instead.
   *
   * Optional so records written before this field existed still parse.
   */
  supersededBy: z.string().nullable().optional(),
});
export type MemoryRecord = z.infer<typeof MemoryRecordSchema>;

/** The domain spec. `version` stamps the medium; bump on schema changes. */
export const memoryDomain = defineDomain({
  name: 'memory',
  version: 1,
  tables: {
    records: domainTable<`mem_${string}`, MemoryRecord>(MemoryRecordSchema),
  },
});

/** Governance limits. Bumped values are validated against these in tests. */
export const MEMORY_DEFAULTS = {
  /** Hard cap on stored records; evicts lowest-value records first. */
  maxRecords: 400,
  /** UTF-16 code units per record content; longer input is truncated. */
  maxContentChars: 2000,
  /** Jaccard token similarity at/above which a new record merges into an old one.
   *  0.7 fits single-char CJK tokenization (each added char dilutes the set). */
  mergeSimilarity: 0.7,
  /** Half-life in days for recency scoring. */
  recencyHalfLifeDays: 90,
  /** Default recall result count (fewer hits = cheaper per call). */
  recallLimit: 3,
  /** Default max chars of a recalled record content shown to the model.
   *  Keeps on-demand recall cheap; 400 chars is enough to judge relevance. */
  recallContentMax: 400,
  /** memory_sediment: max entries per call (token noise guard). */
  sedimentMaxEntries: 3,
  /** memory_sediment: min ms between calls across the process; 0 disables. */
  sedimentCooldownMs: 300_000,
} as const;

export function isValidKind(value: string): value is MemoryKind {
  return (MEMORY_KINDS as readonly string[]).includes(value);
}
export function isValidScope(value: string): value is MemoryScope {
  return (MEMORY_SCOPES as readonly string[]).includes(value);
}
