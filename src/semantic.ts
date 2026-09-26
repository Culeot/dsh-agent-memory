/**
 * Semantic retrieval channel — an OPTIONAL enhancement to the lexical scorer.
 *
 * Why optional
 * ------------
 * `@huggingface/transformers` pulls in `onnxruntime-node` (~100 MB of native
 * binaries). Most users of a memory plugin never need it, so it is a peer
 * dependency: when it is missing (or the model directory is absent), this
 * module reports {@link SemanticIndex.unavailableReason} and the plugin keeps
 * working exactly as before, on lexical scoring alone. A memory plugin that
 * refuses to start because an optional accelerator is missing would be worse
 * than no accelerator at all.
 *
 * Why it is worth the weight (measured, see the project README)
 * ------------------------------------------------------------
 * On a 382-query held-out set built from this machine's real memory bank:
 *
 *     lexical only                        MRR 0.7248   R@1 63.4%
 *     semantic only                       MRR 0.7121   R@1 61.3%
 *     fused (0.65 semantic + 0.35 lexical) MRR 0.8432   R@1 76.4%
 *
 * Neither channel wins alone — semantic is *worse* than lexical by itself.
 * The gain comes entirely from fusing two channels that disagree in different
 * places: lexical owns exact terms and paths, semantic owns paraphrase. That
 * is also why this file does NOT try to make the model "look like" the
 * lexical scorer: a semantic channel trained toward keyword overlap stops
 * being an independent signal and the fusion gain collapses (measured: MRR
 * 0.8432 -> 0.8265 when keyword-style pairs dominated training).
 *
 * Pooling contract
 * ----------------
 * Vectors MUST be produced the same way the model was trained: mean pooling
 * over tokens + L2 normalization (the training script's `BiEncoder.forward`).
 * A different pooling (e.g. CLS) silently degrades quality without erroring,
 * so it is pinned here explicitly.
 *
 * @module dsh-agent-memory/semantic
 */
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import type { MemoryRecord } from './spec.ts';

/** Something that turns text into normalized vectors. */
export interface SemanticBackend {
  readonly name: string;
  readonly dimensions: number;
  embed(texts: readonly string[]): Promise<Float32Array[]>;
}

export interface SemanticConfig {
  /** Directory holding config.json / tokenizer.json / onnx/*.onnx. */
  modelDir: string;
  /** Directory for the precomputed vector cache. */
  cacheDir: string;
  /** Characters of record content to embed (training used 400). */
  contentChars?: number;
  /** ONNX intra-op threads. The host machine runs other heavy jobs; stay small. */
  threads?: number;
}

const CACHE_VERSION = 2;
const DEFAULT_CONTENT_CHARS = 400;
/** Default ONNX threads. The training machine runs other work — never take all cores. */
const DEFAULT_THREADS = 2;

function hashOf(text: string): string {
  return createHash('sha1').update(text).digest('hex').slice(0, 16);
}

/**
 * Load the transformers.js runtime through a non-analyzable dynamic import.
 *
 * A literal `await import('@huggingface/transformers')` would make the bundler
 * (and `tsc`) resolve a package that is intentionally not a hard dependency,
 * turning "optional" into "required at build time". Indirection keeps the
 * dependency genuinely optional while failing cleanly at runtime.
 */
async function importTransformers(): Promise<unknown> {
  const dynamicImport = new Function('specifier', 'return import(specifier)') as
    (specifier: string) => Promise<unknown>;
  return dynamicImport('@huggingface/transformers');
}

/** Mean-pooled, L2-normalized embeddings through transformers.js. */
async function loadTransformersBackend(config: SemanticConfig): Promise<SemanticBackend> {
  const mod = (await importTransformers()) as {
    pipeline: (task: string, model: string, options?: Record<string, unknown>) => Promise<unknown>;
    env?: { backends?: { onnx?: { wasm?: { numThreads?: number } } } };
  };
  if (typeof mod.pipeline !== 'function') {
    throw new Error('@huggingface/transformers has no pipeline() export');
  }
  const threads = config.threads ?? DEFAULT_THREADS;
  if (mod.env?.backends?.onnx?.wasm) mod.env.backends.onnx.wasm.numThreads = threads;

  const extractor = (await mod.pipeline('feature-extraction', config.modelDir, {
    dtype: 'q8',
    session_options: { intraOpNumThreads: threads, interOpNumThreads: 1 },
  })) as (
    texts: string[],
    options: { pooling: string; normalize: boolean },
  ) => Promise<{ tolist: () => number[][] }>;

  const probe = await extractor(['probe'], { pooling: 'mean', normalize: true });
  const dimensions = probe.tolist()[0]?.length ?? 0;
  if (dimensions === 0) throw new Error('embedding probe returned an empty vector');

  return {
    name: `${config.modelDir} (${dimensions}d, mean-pooled)`,
    dimensions,
    async embed(texts: readonly string[]): Promise<Float32Array[]> {
      if (texts.length === 0) return [];
      const out = await extractor([...texts], { pooling: 'mean', normalize: true });
      return out.tolist().map((row) => Float32Array.from(row));
    },
  };
}

/**
 * Lazily-built semantic index over the memory bank.
 *
 * The whole design is "never make things worse": if anything fails (missing
 * package, missing model, broken cache) the index reports why and the caller
 * falls back to lexical scoring for the rest of the process lifetime.
 */
export class SemanticIndex {
  private backend: SemanticBackend | null = null;
  private vectors = new Map<string, Float32Array>();
  private hashes = new Map<string, string>();
  private attempted = false;
  private reason: string | null = null;
  private ready = false;

  constructor(private readonly config: SemanticConfig) {}

  /** Why semantic scoring is off, or null when it is on. */
  get unavailableReason(): string | null {
    return this.reason;
  }

  get isReady(): boolean {
    return this.ready;
  }

  get backendName(): string | null {
    return this.backend?.name ?? null;
  }

  /** Vector count held in memory (diagnostics / RPC). */
  get vectorCount(): number {
    return this.vectors.size;
  }

  private loadCache(): void {
    const indexPath = join(this.config.cacheDir, 'index.json');
    const binPath = join(this.config.cacheDir, 'vectors.bin');
    if (!existsSync(indexPath) || !existsSync(binPath)) return;
    try {
      const parsed = JSON.parse(readFileSync(indexPath, 'utf8')) as {
        version?: number;
        model?: string;
        dimensions?: number;
        entries?: Record<string, { hash: string; offset: number }>;
      };
      if (parsed.version !== CACHE_VERSION || parsed.model !== this.config.modelDir) return;
      const dims = parsed.dimensions ?? 0;
      if (dims <= 0) return;
      const buffer = readFileSync(binPath);
      for (const [id, entry] of Object.entries(parsed.entries ?? {})) {
        const start = entry.offset * dims * 4;
        const end = start + dims * 4;
        if (end > buffer.byteLength) continue;
        const slice = buffer.subarray(start, end);
        // Copy: subarray shares the underlying buffer, and Buffer.allocUnsafe
        // semantics would let a later read observe mutated bytes.
        this.vectors.set(id, new Float32Array(slice.buffer.slice(slice.byteOffset, end)));
        this.hashes.set(id, entry.hash);
      }
    } catch {
      // A corrupt cache is never fatal — it just means everything is recomputed.
      this.vectors.clear();
      this.hashes.clear();
    }
  }

  private saveCache(): void {
    try {
      mkdirSync(this.config.cacheDir, { recursive: true });
      const dims = this.backend?.dimensions ?? 0;
      if (dims <= 0) return;
      const entries: Record<string, { hash: string; offset: number }> = {};
      const ids = [...this.vectors.keys()];
      const bin = Buffer.alloc(ids.length * dims * 4);
      ids.forEach((id, i) => {
        const vec = this.vectors.get(id)!;
        Buffer.from(vec.buffer, vec.byteOffset, vec.byteLength).copy(bin, i * dims * 4);
        entries[id] = { hash: this.hashes.get(id) ?? '', offset: i };
      });
      // Write-then-rename so a crash mid-write cannot leave a half cache behind.
      const tmpIndex = join(this.config.cacheDir, 'index.json.tmp');
      const tmpBin = join(this.config.cacheDir, 'vectors.bin.tmp');
      writeFileSync(tmpIndex, JSON.stringify({
        version: CACHE_VERSION, model: this.config.modelDir, dimensions: dims, entries,
      }));
      writeFileSync(tmpBin, bin);
      renameSync(tmpBin, join(this.config.cacheDir, 'vectors.bin'));
      renameSync(tmpIndex, join(this.config.cacheDir, 'index.json'));
    } catch {
      // Cache writes are best-effort; a read-only profile must still work.
    }
  }

  /** Prepare the backend and (re)compute vectors for the given records. */
  async ensure(records: readonly MemoryRecord[]): Promise<boolean> {
    if (this.ready) return true;
    if (!this.attempted) {
      this.attempted = true;
      try {
        if (!existsSync(join(this.config.modelDir, 'config.json'))) {
          throw new Error(`no model at ${this.config.modelDir}`);
        }
        this.backend = await loadTransformersBackend(this.config);
        this.loadCache();
      } catch (error) {
        this.reason = error instanceof Error ? error.message : String(error);
        this.backend = null;
        return false;
      }
    }
    if (!this.backend) return false;

    const chars = this.config.contentChars ?? DEFAULT_CONTENT_CHARS;
    const stale = records.filter((r) => {
      const h = hashOf(r.content);
      return this.hashes.get(r.id) !== h || !this.vectors.has(r.id);
    });
    if (stale.length > 0) {
      try {
        const vectors = await this.backend.embed(stale.map((r) => r.content.slice(0, chars)));
        stale.forEach((r, i) => {
          this.vectors.set(r.id, vectors[i]!);
          this.hashes.set(r.id, hashOf(r.content));
        });
        this.saveCache();
      } catch (error) {
        this.reason = error instanceof Error ? error.message : String(error);
        this.backend = null;
        return false;
      }
    }
    this.ready = true;
    return true;
  }

  /**
   * Cosine similarity of the query against each record, in input order.
   * Returns null when the channel is unavailable (caller falls back).
   */
  async scores(query: string, records: readonly MemoryRecord[]): Promise<number[] | null> {
    const ok = await this.ensure(records);
    if (!ok || !this.backend) return null;
    const q = query.trim();
    if (q === '') return null;
    let qVec: Float32Array;
    try {
      [qVec] = await this.backend.embed([q]);
    } catch (error) {
      this.reason = error instanceof Error ? error.message : String(error);
      return null;
    }
    if (!qVec) return null;
    return records.map((r) => {
      const vec = this.vectors.get(r.id);
      if (!vec || vec.length !== qVec.length) return 0;
      // Both sides are L2-normalized, so the dot product is the cosine.
      let dot = 0;
      for (let i = 0; i < qVec.length; i += 1) dot += qVec[i]! * vec[i]!;
      return dot;
    });
  }
}
