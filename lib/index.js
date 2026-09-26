// src/index.ts
import { randomUUID } from "node:crypto";
import { statSync, readFileSync as readFileSync2 } from "node:fs";
import { homedir } from "node:os";
import { join as join2 } from "node:path";
import z2 from "@deepseek-ai/schemastery";
import { HarnessError, createUserMessage, boundContextSummary } from "@deepseek-ai/dsh-llm";
import { defineTool } from "@deepseek-ai/dsh-tools";

// src/spec.ts
import { defineDomain, domainTable } from "@deepseek-ai/dsh-storage-domain";
import { z } from "zod";
var MEMORY_KINDS = [
  "fact",
  // verifiable statement about the world or the user's environment
  "preference",
  // what the user likes/wants/how they want things done
  "decision",
  // a concluded choice with its rationale (checkpoint, not a todo)
  "lesson",
  // a mistake/insight worth keeping ("don't do X again")
  "todo",
  // cross-session follow-up item
  "note"
  // anything else
];
var MEMORY_SCOPES = ["user", "project"];
var MemoryRecordSchema = z.object({
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
  supersededBy: z.string().nullable().optional()
});
var memoryDomain = defineDomain({
  name: "memory",
  version: 1,
  tables: {
    records: domainTable(MemoryRecordSchema)
  }
});
var MEMORY_DEFAULTS = {
  /** Hard cap on stored records; evicts lowest-value records first. */
  maxRecords: 400,
  /** UTF-16 code units per record content; longer input is truncated. */
  maxContentChars: 2e3,
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
  sedimentCooldownMs: 3e5
};
function isValidKind(value) {
  return MEMORY_KINDS.includes(value);
}
function isValidScope(value) {
  return MEMORY_SCOPES.includes(value);
}

// src/search.ts
function tokenize(text) {
  const lower = text.toLowerCase();
  const tokens = [];
  const ascii = /[a-z0-9_]+/g;
  let match;
  while ((match = ascii.exec(lower)) !== null) tokens.push(match[0]);
  for (const ch of lower) {
    const code = ch.codePointAt(0);
    if (code >= 19968 && code <= 40959) tokens.push(ch);
  }
  return tokens;
}
function tokenSet(text) {
  return new Set(tokenize(text));
}
function tokenizeBigram(text) {
  const lower = text.toLowerCase();
  const tokens = [];
  const ascii = /[a-z0-9_]+/g;
  let match;
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
function tokenSetBigram(text) {
  return new Set(tokenizeBigram(text));
}
function cjkUnigrams(text) {
  const set = /* @__PURE__ */ new Set();
  const runs = text.toLowerCase().match(/[\u4e00-\u9fff]+/g) ?? [];
  for (const run of runs) for (const ch of run) set.add(ch);
  return set;
}
function jaccard(a, b) {
  if (a.size === 0 && b.size === 0) return 0;
  let intersection = 0;
  for (const token of a) if (b.has(token)) intersection += 1;
  return intersection / (a.size + b.size - intersection);
}
var ACCESS_BOOST_CAP = 1.5;
function lastActive(record, fallbackNow) {
  const updated = Date.parse(record.updatedAt);
  if (record.accessedAt !== null) {
    const accessed = Date.parse(record.accessedAt);
    return accessed > updated ? accessed : updated;
  }
  return updated;
}
var MATCH_BASE_MIN = 1;
var IMPORTANCE_BOOST_SLOPE = 0.1;
var LEXICAL_SATURATION = 4;
var DEFAULT_SEMANTIC_WEIGHT = 0.3;
var DEFAULT_SEMANTIC_MIN = 0.5;
var SUPERSEDED_PENALTY = 0.25;
function clamp01(x) {
  if (!Number.isFinite(x)) return 0;
  return x < 0 ? 0 : x > 1 ? 1 : x;
}
var STOPWORDS = /* @__PURE__ */ new Set([
  "\u7684",
  "\u4E86",
  "\u5417",
  "\u5462",
  "\u554A",
  "\u54E6",
  "\u55EF",
  "\u54DF",
  "\u5427",
  "\u5440",
  "\u54C8",
  "\u563F",
  "\u5582",
  "\u662F",
  "\u5728",
  "\u548C",
  "\u4E0E",
  "\u6216",
  "\u53CA",
  "\u628A",
  "\u88AB",
  "\u7ED9",
  "\u5411",
  "\u4ECE",
  "\u5230",
  "\u4E8E",
  "\u5BF9",
  "\u5C31",
  "\u90FD",
  "\u4E5F",
  "\u8FD8",
  "\u4F46",
  "\u800C",
  "\u5219",
  "\u4E14",
  "\u6211",
  "\u4F60",
  "\u4ED6",
  "\u5979",
  "\u5B83",
  "\u54B1",
  "\u60A8",
  "\u4EEC",
  "\u8FD9",
  "\u90A3",
  "\u54EA",
  "\u4EC0",
  "\u4E48",
  "\u600E",
  "\u4E3A",
  "\u4F55",
  "\u5565",
  "\u53EF",
  "\u4EE5",
  "\u80FD",
  "\u597D",
  "\u4E0D",
  "\u522B",
  "\u8BF7",
  "\u5148",
  "\u518D",
  "\u53C8",
  "\u5F88",
  "\u592A",
  "\u771F",
  "\u633A",
  "\u4F1A",
  "\u60F3",
  "\u8981",
  "\u5E2E",
  "\u770B",
  "\u8BF4",
  "\u505A",
  "\u5F04",
  "\u641E",
  "\u6574",
  "ok",
  "ok\u4E86",
  "\u597D\u7684",
  "\u55EF\u55EF",
  "\u54C8\u54C8",
  "\u8C22\u8C22",
  "\u611F\u8C22",
  "\u8C22\u4E86",
  "\u53EF\u4EE5\u4E86",
  "\u660E\u767D\u4E86",
  "\u61C2\u4E86",
  "\u77E5\u9053",
  "\u770B\u770B",
  "\u8BF7\u95EE"
]);
function hasMeaningfulQuery(query) {
  let meaningfulCjk = 0;
  for (const ch of query) {
    const code = ch.codePointAt(0);
    if (code >= 19968 && code <= 40959 && !STOPWORDS.has(ch)) meaningfulCjk += 1;
  }
  const asciiWords = query.toLowerCase().match(/[a-z]{3,}/g);
  return meaningfulCjk >= 2 || asciiWords !== null && asciiWords.length >= 1;
}
function bm25Signal(query, content, avgLen = 40) {
  const qTokens = [...new Set(tokenizeBigram(query))];
  if (qTokens.length === 0) return 0;
  const cTokens = tokenizeBigram(content);
  const contentLen = Math.max(1, cTokens.length);
  const freq = /* @__PURE__ */ new Map();
  for (const t of cTokens) freq.set(t, (freq.get(t) ?? 0) + 1);
  const k1 = 1.2;
  const b = 0.75;
  let sum = 0;
  for (const t of qTokens) {
    const tf = freq.get(t) ?? 0;
    if (tf === 0) continue;
    const norm = contentLen / avgLen;
    sum += tf * (k1 + 1) / (tf + k1 * (1 - b + b * norm));
  }
  return Math.min(2, sum * 0.4);
}
function explainRecord(record, query, options = {}) {
  const q = query.trim();
  if (q === "") return { score: 0, reasons: [], base: 0 };
  const contentLower = record.content.toLowerCase();
  const queryLower = q.toLowerCase();
  const reasons = [];
  const baseParts = [];
  let base = 0;
  if (contentLower.includes(queryLower)) {
    base += 3;
    baseParts.push("substring(3.0)");
    reasons.push("substring:\u6574\u4E32\u547D\u4E2D(+3.0)");
  }
  const matchedTags = record.tags.filter((tag) => tag.toLowerCase().includes(queryLower));
  if (matchedTags.length > 0) {
    base += 1.5;
    baseParts.push("tag(1.5)");
    reasons.push(`tag:${matchedTags.join("/")} \u547D\u4E2D(+1.5)`);
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
  if (qwordHits > 0) baseParts.push(`qword(${qwordHits}\xD71.2)`);
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
  const recencyBoost = Math.pow(0.5, ageMs / (halfLifeDays * 24 * 3600 * 1e3));
  const accessBoost = Math.min(ACCESS_BOOST_CAP, 1 + Math.log(1 + record.accessCount) * 0.15);
  const supersededBoost = record.supersededBy ? SUPERSEDED_PENALTY : 1;
  const score = base * importanceBoost * recencyBoost * accessBoost * supersededBoost;
  if (importanceBoost > 1) reasons.push(`importance:${record.importance}(\xD7${importanceBoost.toFixed(2)})`);
  const ageDays = ageMs / (24 * 3600 * 1e3);
  reasons.push(`recency:${ageDays.toFixed(1)} \u5929\u524D\u6D3B\u8DC3(\xD7${recencyBoost.toFixed(2)})`);
  if (record.accessCount > 0) reasons.push(`access:${record.accessCount} \u6B21(\xD7${accessBoost.toFixed(2)})`);
  if (baseParts.length > 0) reasons.unshift(`base:${baseParts.join("+")}`);
  if (record.supersededBy !== null && record.supersededBy !== void 0) {
    reasons.unshift(`superseded:\u5DF2\u88AB ${record.supersededBy} \u53D6\u4EE3(\xD7${SUPERSEDED_PENALTY})`);
  }
  return { score, reasons: reasons.slice(0, 8), base };
}
function scoreRecord(record, query, options = {}) {
  return explainRecord(record, query, options).score;
}
function isExpired(record, now = Date.now()) {
  return record.expiresAt !== null && Date.parse(record.expiresAt) <= now;
}
function hotnessScore(record, options = {}) {
  const halfLifeDays = options.recencyHalfLifeDays ?? 90;
  const now = options.now ?? Date.now();
  const ageMs = now - lastActive(record, now);
  const recencyBoost = Math.pow(0.5, ageMs / (halfLifeDays * 24 * 3600 * 1e3));
  const importanceBoost = 1 + (record.importance - 1) * (options.boostSlope ?? IMPORTANCE_BOOST_SLOPE);
  const accessBoost = Math.min(ACCESS_BOOST_CAP, 1 + Math.log(1 + record.accessCount) * 0.15);
  return recencyBoost * importanceBoost * accessBoost;
}
function hotRecords(records, limit, options = {}) {
  const now = options.now ?? Date.now();
  const scored = [];
  for (const record of records) {
    if (isExpired(record, now)) continue;
    scored.push({ record, score: hotnessScore(record, { ...options, now }) });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, Math.max(0, limit));
}
function rankRecords(records, query, limit, options = {}) {
  const now = options.now ?? Date.now();
  const scored = [];
  for (const record of records) {
    if (isExpired(record, now)) continue;
    const explained = explainRecord(record, query, { ...options, now });
    if (explained.score > 0) scored.push({ record, score: explained.score, reasons: explained.reasons, base: explained.base });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, Math.max(0, limit));
}
function recordBoost(record, options = {}, now = Date.now()) {
  const slope = options.boostSlope ?? IMPORTANCE_BOOST_SLOPE;
  const importanceBoost = 1 + (record.importance - 1) * slope;
  const halfLifeDays = options.recencyHalfLifeDays ?? 90;
  const ageMs = now - lastActive(record, now);
  const recencyBoost = Math.pow(0.5, ageMs / (halfLifeDays * 24 * 3600 * 1e3));
  const accessBoost = Math.min(ACCESS_BOOST_CAP, 1 + Math.log(1 + record.accessCount) * 0.15);
  return importanceBoost * recencyBoost * accessBoost;
}
function rankRecordsHybrid(records, query, limit, options = {}) {
  const semantic = options.semantic;
  if (!semantic || semantic.length !== records.length) {
    return rankRecords(records, query, limit, options);
  }
  const now = options.now ?? Date.now();
  const w = clamp01(options.semanticWeight ?? DEFAULT_SEMANTIC_WEIGHT);
  const semMin = options.semanticMin ?? DEFAULT_SEMANTIC_MIN;
  const scored = [];
  for (let i = 0; i < records.length; i += 1) {
    const record = records[i];
    if (isExpired(record, now)) continue;
    const explained = explainRecord(record, query, { ...options, now });
    const cos = clamp01(semantic[i]);
    const lexicalHit = explained.base >= MATCH_BASE_MIN;
    if (!lexicalHit && cos < semMin) continue;
    const lexPart = clamp01(explained.base / LEXICAL_SATURATION);
    const relevance = w * cos + (1 - w) * lexPart;
    const penalty = record.supersededBy ? SUPERSEDED_PENALTY : 1;
    const score = relevance * recordBoost(record, options, now) * penalty;
    const reasons = [
      `fuse:sem=${cos.toFixed(2)} lex=${lexPart.toFixed(2)} w=${w.toFixed(2)}`,
      ...record.supersededBy ? [`superseded:\u5DF2\u88AB ${record.supersededBy} \u53D6\u4EE3(\xD7${SUPERSEDED_PENALTY})`] : [],
      ...explained.reasons
    ].slice(0, 8);
    if (score > 0) scored.push({ record, score, reasons, base: explained.base, relevance });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, Math.max(0, limit));
}
function countHybridMatches(records, query, options = {}) {
  const semantic = options.semantic;
  const now = options.now ?? Date.now();
  const semMin = options.semanticMin ?? DEFAULT_SEMANTIC_MIN;
  let count = 0;
  for (let i = 0; i < records.length; i += 1) {
    const record = records[i];
    if (isExpired(record, now)) continue;
    if (explainRecord(record, query, { ...options, now }).base >= MATCH_BASE_MIN) {
      count += 1;
      continue;
    }
    if (semantic && semantic.length === records.length && clamp01(semantic[i]) >= semMin) count += 1;
  }
  return count;
}

// src/semantic.ts
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { join } from "node:path";
var CACHE_VERSION = 2;
var DEFAULT_CONTENT_CHARS = 400;
var DEFAULT_THREADS = 2;
function hashOf(text) {
  return createHash("sha1").update(text).digest("hex").slice(0, 16);
}
async function importTransformers() {
  const dynamicImport = new Function("specifier", "return import(specifier)");
  return dynamicImport("@huggingface/transformers");
}
async function loadTransformersBackend(config) {
  const mod = await importTransformers();
  if (typeof mod.pipeline !== "function") {
    throw new Error("@huggingface/transformers has no pipeline() export");
  }
  const threads = config.threads ?? DEFAULT_THREADS;
  if (mod.env?.backends?.onnx?.wasm) mod.env.backends.onnx.wasm.numThreads = threads;
  const extractor = await mod.pipeline("feature-extraction", config.modelDir, {
    dtype: "q8",
    session_options: { intraOpNumThreads: threads, interOpNumThreads: 1 }
  });
  const probe = await extractor(["probe"], { pooling: "mean", normalize: true });
  const dimensions = probe.tolist()[0]?.length ?? 0;
  if (dimensions === 0) throw new Error("embedding probe returned an empty vector");
  return {
    name: `${config.modelDir} (${dimensions}d, mean-pooled)`,
    dimensions,
    async embed(texts) {
      if (texts.length === 0) return [];
      const out = await extractor([...texts], { pooling: "mean", normalize: true });
      return out.tolist().map((row) => Float32Array.from(row));
    }
  };
}
var SemanticIndex = class {
  constructor(config) {
    this.config = config;
  }
  config;
  backend = null;
  vectors = /* @__PURE__ */ new Map();
  hashes = /* @__PURE__ */ new Map();
  attempted = false;
  reason = null;
  ready = false;
  /** Why semantic scoring is off, or null when it is on. */
  get unavailableReason() {
    return this.reason;
  }
  get isReady() {
    return this.ready;
  }
  get backendName() {
    return this.backend?.name ?? null;
  }
  /** Vector count held in memory (diagnostics / RPC). */
  get vectorCount() {
    return this.vectors.size;
  }
  loadCache() {
    const indexPath = join(this.config.cacheDir, "index.json");
    const binPath = join(this.config.cacheDir, "vectors.bin");
    if (!existsSync(indexPath) || !existsSync(binPath)) return;
    try {
      const parsed = JSON.parse(readFileSync(indexPath, "utf8"));
      if (parsed.version !== CACHE_VERSION || parsed.model !== this.config.modelDir) return;
      const dims = parsed.dimensions ?? 0;
      if (dims <= 0) return;
      const buffer = readFileSync(binPath);
      for (const [id, entry] of Object.entries(parsed.entries ?? {})) {
        const start = entry.offset * dims * 4;
        const end = start + dims * 4;
        if (end > buffer.byteLength) continue;
        const slice = buffer.subarray(start, end);
        this.vectors.set(id, new Float32Array(slice.buffer.slice(slice.byteOffset, end)));
        this.hashes.set(id, entry.hash);
      }
    } catch {
      this.vectors.clear();
      this.hashes.clear();
    }
  }
  saveCache() {
    try {
      mkdirSync(this.config.cacheDir, { recursive: true });
      const dims = this.backend?.dimensions ?? 0;
      if (dims <= 0) return;
      const entries = {};
      const ids = [...this.vectors.keys()];
      const bin = Buffer.alloc(ids.length * dims * 4);
      ids.forEach((id, i) => {
        const vec = this.vectors.get(id);
        Buffer.from(vec.buffer, vec.byteOffset, vec.byteLength).copy(bin, i * dims * 4);
        entries[id] = { hash: this.hashes.get(id) ?? "", offset: i };
      });
      const tmpIndex = join(this.config.cacheDir, "index.json.tmp");
      const tmpBin = join(this.config.cacheDir, "vectors.bin.tmp");
      writeFileSync(tmpIndex, JSON.stringify({
        version: CACHE_VERSION,
        model: this.config.modelDir,
        dimensions: dims,
        entries
      }));
      writeFileSync(tmpBin, bin);
      renameSync(tmpBin, join(this.config.cacheDir, "vectors.bin"));
      renameSync(tmpIndex, join(this.config.cacheDir, "index.json"));
    } catch {
    }
  }
  /** Prepare the backend and (re)compute vectors for the given records. */
  async ensure(records) {
    if (this.ready) return true;
    if (!this.attempted) {
      this.attempted = true;
      try {
        if (!existsSync(join(this.config.modelDir, "config.json"))) {
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
          this.vectors.set(r.id, vectors[i]);
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
  async scores(query, records) {
    const ok2 = await this.ensure(records);
    if (!ok2 || !this.backend) return null;
    const q = query.trim();
    if (q === "") return null;
    let qVec;
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
      let dot = 0;
      for (let i = 0; i < qVec.length; i += 1) dot += qVec[i] * vec[i];
      return dot;
    });
  }
};

// src/rpc.ts
var MEMORY_READ_CHANNEL = "/dsh-memory-read";
var MEMORY_WRITE_CHANNEL = "/dsh-memory-write";
var MAX_BODY_BYTES = 1 << 20;
function ok(value) {
  return { ok: true, value };
}
function fail(code, message) {
  return { ok: false, error: { code, message, details: {} } };
}
function str(payload, key) {
  const v = payload[key];
  return typeof v === "string" && v !== "" ? v : void 0;
}
function arr(payload, key) {
  const v = payload[key];
  return Array.isArray(v) ? v.map(String) : void 0;
}
function num(payload, key) {
  const v = payload[key];
  return typeof v === "number" && Number.isFinite(v) ? v : void 0;
}
function notReady() {
  return fail("internal", "memory core not ready");
}
function createReadHandler(getCore) {
  return async (method, payload) => {
    try {
      const core = getCore();
      if (core === void 0) return notReady();
      switch (method) {
        case "stats": {
          const { total, byKind } = core.stats();
          const expired = core.inspect({}).expired;
          return ok({ total, byKind, expired });
        }
        case "list":
          return ok(core.inspect({
            kinds: arr(payload, "kinds"),
            tags: arr(payload, "tags"),
            scope: str(payload, "scope"),
            limit: num(payload, "limit"),
            offset: num(payload, "offset")
          }));
        case "get": {
          const id = str(payload, "id");
          if (!id) return fail("bad-request", "id required");
          const record = core.getById(id);
          return record ? ok(record) : fail("not-found", `no memory ${id}`);
        }
        case "search": {
          const query = str(payload, "query");
          if (!query) return fail("bad-request", "query required");
          const result = await core.recall({
            query,
            limit: num(payload, "limit") ?? 10,
            touch: false,
            contentMax: num(payload, "content_max")
          });
          return ok(result);
        }
        default:
          return fail("bad-request", `unknown method ${method}`);
      }
    } catch (error) {
      return fail("internal", error instanceof Error ? error.message : String(error));
    }
  };
}
function createWriteHandler(getCore) {
  return async (method, payload) => {
    try {
      const core = getCore();
      if (core === void 0) return notReady();
      switch (method) {
        case "forget": {
          const id = str(payload, "id");
          if (!id) return fail("bad-request", "id required");
          const out = await core.forget({ id, confirm: payload.confirm === true });
          return ok(out);
        }
        case "update": {
          const id = str(payload, "id");
          if (!id) return fail("bad-request", "id required");
          const out = await core.updateContent({
            id,
            content: typeof payload.content === "string" ? payload.content : void 0,
            tags: arr(payload, "tags"),
            importance: num(payload, "importance")
          });
          return ok(out);
        }
        case "remember": {
          const content = typeof payload.content === "string" ? payload.content.trim() : "";
          if (content === "") return fail("bad-request", "content required");
          const out = await core.remember({
            content,
            kind: typeof payload.kind === "string" && payload.kind !== "" ? payload.kind : "note",
            tags: arr(payload, "tags"),
            scope: typeof payload.scope === "string" && payload.scope !== "" ? payload.scope : "user",
            importance: num(payload, "importance")
          });
          return ok(out);
        }
        default:
          return fail("bad-request", `unknown method ${method}`);
      }
    } catch (error) {
      return fail("internal", error instanceof Error ? error.message : String(error));
    }
  };
}
function writeJson(res, status, body) {
  const text = JSON.stringify(body);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(text)
  });
  res.end(text);
}
async function readBody(req) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    size += buf.length;
    if (size > MAX_BODY_BYTES) return void 0;
    chunks.push(buf);
  }
  return Buffer.concat(chunks).toString("utf8");
}
function createHttpRoute(channel, handler, connection) {
  return {
    kind: "prefix",
    path: channel,
    async handler(req, res) {
      const rejection = connection.requestRejection?.(req);
      if (rejection !== void 0) {
        res.writeHead(rejection);
        res.end(rejection === 401 ? "unauthorized" : "forbidden");
        return;
      }
      const pathname = new URL(String(req?.url ?? "/"), "http://localhost").pathname;
      const endpoint = pathname.startsWith(`${channel}/`) ? pathname.slice(channel.length + 1) : void 0;
      if (req?.method !== "POST" || endpoint === void 0) {
        res.writeHead(404);
        res.end("not found");
        return;
      }
      const contentType = String(req?.headers?.["content-type"] ?? "").split(";")[0]?.trim().toLowerCase();
      if (contentType !== "application/json") {
        res.writeHead(415);
        res.end("content type must be application/json");
        return;
      }
      const raw = await readBody(req);
      if (raw === void 0) {
        res.writeHead(413);
        res.end("body too large");
        return;
      }
      let message;
      try {
        message = JSON.parse(raw);
      } catch {
        res.writeHead(400);
        res.end("body is not JSON");
        return;
      }
      const rpcId = typeof message?.rpcId === "string" ? message.rpcId : "invalid-request";
      if (message?.type !== "client-request" || typeof message?.method !== "string") {
        writeJson(res, 200, { type: "server-response", rpcId, result: fail("bad-request", "invalid client-request message") });
        return;
      }
      if (message.method !== endpoint) {
        writeJson(res, 200, {
          type: "server-response",
          rpcId,
          result: fail("bad-request", `method ${JSON.stringify(message.method)} does not match endpoint ${JSON.stringify(endpoint)}`)
        });
        return;
      }
      const payload = message.payload !== null && typeof message.payload === "object" ? message.payload : {};
      let result;
      try {
        result = await handler(endpoint, payload);
      } catch (error) {
        result = fail("internal", error instanceof Error ? error.message : String(error));
      }
      writeJson(res, 200, { type: "server-response", rpcId, result });
    }
  };
}
function registerRpc(webContext, getCore, warn) {
  const connection = webContext?.connection;
  if (connection === void 0) return;
  const channels = [
    [MEMORY_READ_CHANNEL, createReadHandler(getCore)],
    [MEMORY_WRITE_CHANNEL, createWriteHandler(getCore)]
  ];
  const webServer = webContext.webServer;
  for (const [channel, handler] of channels) {
    let viaOfficial = false;
    let officialError;
    try {
      if (typeof connection.rpc?.handle !== "function") throw new Error("connection.rpc.handle unavailable");
      connection.rpc.handle(channel, handler, { authority: "trusted-host" });
      viaOfficial = true;
    } catch (error) {
      officialError = error;
    }
    if (viaOfficial) continue;
    if (webServer === void 0) continue;
    warn?.(`memory: connection.rpc.handle(${channel}) \u4E0D\u53EF\u7528(${officialError instanceof Error ? officialError.message : String(officialError)}),\u6539\u7528\u81EA\u6CE8\u518C\u8DEF\u7531`);
    try {
      webServer.register(createHttpRoute(channel, handler, connection));
    } catch (error) {
      warn?.(`memory: \u81EA\u6CE8\u518C\u8DEF\u7531 ${channel} \u5931\u8D25: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
}

// src/index.ts
var name = "memory";
var inject = ["tools", "storageDomain", "systemPrompt"];
var Config = z2.object({
  maxRecords: z2.number().min(1).default(MEMORY_DEFAULTS.maxRecords),
  maxContentChars: z2.number().min(50).default(MEMORY_DEFAULTS.maxContentChars),
  recencyHalfLifeDays: z2.number().min(1).default(MEMORY_DEFAULTS.recencyHalfLifeDays),
  mergeSimilarity: z2.number().min(0.5).max(1).default(MEMORY_DEFAULTS.mergeSimilarity),
  /** Register the memory-protocol system-prompt section. */
  protocolSection: z2.boolean().default(true),
  /**
   * Per-step dynamic injection: before each model step, recall the most
   * relevant memories for the user's current message and append them to the
   * step input. Replaces the old every-turn hot-memory broadcast (stateContext)
   * — relevant-on-demand instead of broadcast.
   */
  injectEnabled: z2.boolean().default(true),
  /** Maximum memories to inject per step (0 disables injection). Actual count depends on relevance. */
  injectCount: z2.number().min(0).max(10).default(3),
  /**
   * Budgeted injection: keep the top memory plus any scoring within this
   * fraction of it. 0 = off (historical behaviour: score floor + fixed cap).
   * A clear winner then injects one memory instead of three mediocre ones.
   */
  injectBudgetRatio: z2.number().min(0).max(1).default(0.5),
  /** Minimum score threshold for injection (0 = no threshold, just use rank). */
  injectMinScore: z2.number().min(0).max(100).default(1),
  /** Max content chars per injected memory summary. */
  injectMaxChars: z2.number().min(40).max(400).default(120),
  /** Default max chars of a recalled record's content shown to the model. */
  recallContentMax: z2.number().min(100).max(2e3).default(MEMORY_DEFAULTS.recallContentMax),
  /** memory_sediment: max entries per call. */
  sedimentMaxEntries: z2.number().min(1).max(10).default(MEMORY_DEFAULTS.sedimentMaxEntries),
  /** memory_sediment: min ms between calls; 0 disables the cooldown. */
  sedimentCooldownMs: z2.number().min(0).default(MEMORY_DEFAULTS.sedimentCooldownMs),
  /**
   * Lesson auto-solidification: when the SAME error fingerprint (code/message)
   * fires this many times, inject a nudge telling the agent to write it as an
   * importance-3 lesson. The lesson then auto-injects on related topics,
   * preventing recurrence. 0 disables.
   */
  lessonizeEnabled: z2.boolean().default(true),
  lessonizeAfter: z2.number().min(2).max(20).default(2),
  /**
   * Semantic channel: fuse a local sentence-embedding model with the lexical
   * scorer. Needs @huggingface/transformers (an OPTIONAL peer dependency) plus
   * a model directory; when either is missing, recall silently falls back to
   * lexical-only — an optional accelerator must never break the plugin.
   *
   * Measured on a real 382-record bank: lexical 0.7248 MRR → fused 0.8432.
   */
  semanticEnabled: z2.boolean().default(true),
  /** Model directory (config.json / tokenizer.json / onnx/*.onnx). '' disables. */
  semanticModelDir: z2.string().default(""),
  /** Vector-cache directory. '' = $DSH_HOME/storages/memory-semantic. */
  semanticCacheDir: z2.string().default(""),
  /** Semantic weight in fusion: relevance = w*semantic + (1-w)*lexical. */
  semanticWeight: z2.number().min(0).max(1).default(DEFAULT_SEMANTIC_WEIGHT),
  /** Minimum cosine for a record to count as a match with no lexical hit. */
  semanticMin: z2.number().min(0).max(1).default(DEFAULT_SEMANTIC_MIN),
  /**
   * Slope of the importance boost. The historical 0.75 measured -0.046 MRR
   * versus no boost at all (most records sit at importance 3, so the boost
   * separates nothing while punishing the rest). 0 disables it.
   */
  boostSlope: z2.number().min(0).max(2).default(IMPORTANCE_BOOST_SLOPE),
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
  noMatchThreshold: z2.number().min(0).max(1).default(0.18)
});
function makeId() {
  const rand = randomUUID().replaceAll("-", "");
  return `mem_${rand.slice(0, 16)}`;
}
function iso(now) {
  return new Date(now).toISOString();
}
function truncate(content, max) {
  const chars = Array.from(content);
  if (chars.length <= max) return content;
  return chars.slice(0, max - 1).join("") + "\u2026";
}
function queryFromMessages(messages) {
  const parts = [];
  for (const msg of messages ?? []) {
    const content = msg.data?.content ?? msg.content;
    if (typeof content === "string") parts.push(content);
    else if (Array.isArray(content)) parts.push(content.map((b) => typeof b?.text === "string" ? b.text : "").join("\n"));
  }
  return parts.join("\n").slice(0, 600).trim();
}
function renderInjection(results, maxChars) {
  if (results.length === 0) return "";
  const lines = results.map((r) => {
    const text = truncate(r.content, maxChars);
    const flags = [r.importance === 3 ? "\u91CD\u8981" : null, r.scope === "user" ? "\u5168\u5C40" : null].filter(Boolean).join("/");
    return `- ${text}${flags ? `(${flags})` : ""}`;
  });
  return `\u76F8\u5173\u8BB0\u5FC6(\u81EA\u52A8\u68C0\u7D22 ${results.length} \u6761,\u53EF\u80FD\u4E0E\u5F53\u524D\u4EFB\u52A1\u76F8\u5173):
${lines.join("\n")}
\u9700\u8981\u7EC6\u8282\u7528 memory_recall,\u8FC7\u65F6\u7528 memory_forget\u3002`;
}
function selectForInjection(results, options) {
  const minScore = options.minScore ?? 0;
  let kept = minScore > 0 ? results.filter((r) => r.score >= minScore) : [...results];
  const ratio = options.budgetRatio ?? 0;
  if (ratio > 0 && kept.length > 1) {
    const top = kept[0].score;
    if (top > 0) kept = kept.filter((r) => r.score >= top * ratio);
  }
  return kept.slice(0, Math.max(0, options.limit));
}
function extractErrorFingerprint(error) {
  if (typeof error === "string") return error.trim().slice(0, 80);
  const e = error;
  const code = typeof e?.code === "string" && e.code !== "" ? e.code : null;
  if (code) return code;
  const msg = typeof e?.message === "string" ? e.message.trim().slice(0, 80) : "";
  if (msg) return msg;
  return typeof e?.name === "string" ? e.name : "unknown-error";
}
function normalizeTags(tags) {
  if (!tags) return [];
  const out = [];
  for (const raw of tags) {
    const tag = String(raw).trim().toLowerCase().slice(0, 64);
    if (tag !== "" && !out.includes(tag)) out.push(tag);
  }
  return out.slice(0, 16);
}
var MemoryCore = class {
  constructor(table, config, beforeWrite, onWritten) {
    this.table = table;
    this.config = config;
    this.beforeWrite = beforeWrite;
    this.onWritten = onWritten;
    this.semantic = config.semantic ? new SemanticIndex(config.semantic) : null;
  }
  table;
  config;
  beforeWrite;
  onWritten;
  /**
   * Optional semantic channel. Null when disabled or when no model directory is
   * configured; recall then behaves exactly as it did before this existed.
   */
  semantic;
  /** Semantic-channel diagnostics, surfaced to the UI/RPC layer. */
  semanticStatus() {
    return {
      enabled: this.semantic !== null,
      ready: this.semantic?.isReady ?? false,
      reason: this.semantic?.unavailableReason ?? null,
      backend: this.semantic?.backendName ?? null,
      vectors: this.semantic?.vectorCount ?? 0
    };
  }
  /**
   * Cosine scores for the pool, or null when the semantic channel is off.
   * Never throws: an optional channel that fails must degrade, not break.
   */
  async semanticScores(query, pool) {
    if (!this.semantic) return null;
    try {
      return await this.semantic.scores(query, pool);
    } catch {
      return null;
    }
  }
  all() {
    return [...this.table.entries()].map(([, v]) => v);
  }
  guardWrite() {
    if (this.beforeWrite) this.beforeWrite();
  }
  async put(key, value) {
    await this.table.put(key, value);
    if (this.onWritten) this.onWritten();
  }
  async update(key, fn) {
    const next = await this.table.update(key, fn);
    if (this.onWritten) this.onWritten();
    return next;
  }
  async delete(key) {
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
  async markSuperseded(newId, ids) {
    if (!ids || ids.length === 0) return [];
    const marked = [];
    for (const oldId of ids) {
      if (oldId === newId) continue;
      if (!this.table.get(oldId)) continue;
      await this.update(oldId, (current) => ({ ...current, supersededBy: newId }));
      marked.push(oldId);
    }
    return marked;
  }
  /** Delete every TTL-expired record. Returns the number removed. */
  async purgeExpired(now = Date.now()) {
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
  async evict() {
    const excess = this.table.size - this.config.maxRecords;
    if (excess <= 0) return 0;
    const all = this.all();
    const sortKey = (r) => r.accessedAt ?? r.updatedAt;
    const nonCritical = all.filter((r) => r.importance < 3).sort((a, b) => {
      if (a.importance !== b.importance) return a.importance - b.importance;
      return sortKey(a) < sortKey(b) ? -1 : sortKey(a) > sortKey(b) ? 1 : 0;
    });
    const critical = all.filter((r) => r.importance === 3).sort(
      (a, b) => sortKey(a) < sortKey(b) ? -1 : sortKey(a) > sortKey(b) ? 1 : 0
    );
    const victims = [...nonCritical, ...critical].slice(0, excess);
    let removed = 0;
    for (const victim of victims) {
      if (await this.delete(victim.id)) removed += 1;
    }
    return removed;
  }
  /** Insert or merge one memory. */
  async remember(input) {
    this.guardWrite();
    const now = input.now ?? Date.now();
    const kind = isValidKind(input.kind) ? input.kind : "note";
    const scope = isValidScope(input.scope) ? input.scope : "project";
    const importance = Math.min(3, Math.max(1, Math.round(input.importance ?? 2)));
    const content = truncate(input.content.trim(), this.config.maxContentChars);
    if (content === "") throw new HarnessError("memory content is empty", "MEMORY_EMPTY_CONTENT");
    const tags = normalizeTags(input.tags);
    const project = scope === "project" ? input.project ?? null : null;
    const expiresAt = input.ttlDays && input.ttlDays > 0 ? new Date(now + Math.min(input.ttlDays, 3650) * 24 * 3600 * 1e3).toISOString() : null;
    const incomingTokens = tokenSet(content);
    let merged = false;
    for (const existing of this.all()) {
      if (isExpired(existing, now)) continue;
      if (existing.kind !== kind || existing.scope !== scope || existing.project !== project) continue;
      if (jaccard(incomingTokens, tokenSet(existing.content)) < this.config.mergeSimilarity) continue;
      await this.update(existing.id, (current) => {
        const tagsMerged = [.../* @__PURE__ */ new Set([...current.tags, ...tags])];
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
          expiresAt: input.ttlDays !== void 0 && input.ttlDays !== null ? expiresAt : current.expiresAt
        };
      });
      merged = true;
      const supersededByMerge = await this.markSuperseded(existing.id, input.supersedes);
      return { id: existing.id, merged: true, evicted: 0, content, superseded: supersededByMerge };
    }
    const id = makeId();
    const record = {
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
      expiresAt
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
  async recall(input) {
    const now = input.now ?? Date.now();
    if (input.touch !== false) {
      this.guardWrite();
      await this.purgeExpired(now);
    }
    const kinds = (input.kinds ?? []).filter(isValidKind);
    const tags = normalizeTags(input.tags);
    const scope = input.scope && isValidScope(input.scope) ? input.scope : void 0;
    const project = input.project ?? void 0;
    let pool = this.all().filter((r) => !isExpired(r, now));
    if (kinds.length > 0) pool = pool.filter((r) => kinds.includes(r.kind));
    if (tags.length > 0) pool = pool.filter((r) => tags.every((t) => r.tags.includes(t)));
    if (scope) pool = pool.filter((r) => r.scope === scope);
    if (project) pool = pool.filter((r) => r.scope === "user" || r.project === project);
    const limit = Math.max(1, Math.min(50, Math.round(input.limit ?? MEMORY_DEFAULTS.recallLimit)));
    const query = (input.query ?? "").trim();
    const rankOptions = {
      now,
      recencyHalfLifeDays: this.config.recencyHalfLifeDays,
      boostSlope: this.config.boostSlope
    };
    const hybridOptions = {
      ...rankOptions,
      semanticWeight: this.config.semanticWeight,
      semanticMin: this.config.semanticMin
    };
    const semantic = query === "" ? null : await this.semanticScores(query, pool);
    const ranked = query === "" ? hotRecords(pool, limit, rankOptions) : rankRecordsHybrid(pool, query, limit, { ...hybridOptions, semantic });
    const totalMatched = query === "" ? pool.length : countHybridMatches(pool, query, { ...hybridOptions, semantic });
    if (input.touch !== false) {
      for (const { record } of ranked) {
        try {
          await this.update(record.id, (current) => ({
            ...current,
            accessedAt: iso(now),
            accessCount: current.accessCount + 1
          }));
        } catch {
        }
      }
    }
    const topRelevance = ranked.length > 0 ? ranked[0].relevance ?? Math.min(1, (ranked[0].base ?? 0) / LEXICAL_SATURATION) : 0;
    const gate = this.config.noMatchThreshold ?? 0;
    const noMatch = query !== "" && gate > 0 && topRelevance < gate;
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
        ...record.supersededBy ? { supersededBy: record.supersededBy } : {},
        reasons
      }))
    };
  }
  /** Read-only single record lookup (used by the UI RPC layer). */
  getById(id) {
    return this.table.get(id);
  }
  /**
   * Read-only inventory view for the UI panel: never purges/writes, so a
   * passive page load can't mutate the store. Mirrors index() minus the
   * purge and without touching access stats.
   */
  inspect(input) {
    const now = input.now ?? Date.now();
    const kinds = (input.kinds ?? []).filter(isValidKind);
    const tags = normalizeTags(input.tags);
    const scope = input.scope && isValidScope(input.scope) ? input.scope : void 0;
    let pool = this.all();
    if (kinds.length > 0) pool = pool.filter((r) => kinds.includes(r.kind));
    if (tags.length > 0) pool = pool.filter((r) => tags.every((t) => r.tags.includes(t)));
    if (scope) pool = pool.filter((r) => r.scope === scope);
    const expired = pool.filter((r) => isExpired(r, now)).length;
    const byKind = {};
    for (const r of pool) byKind[r.kind] = (byKind[r.kind] ?? 0) + 1;
    pool.sort((a, b) => a.updatedAt < b.updatedAt ? 1 : a.updatedAt > b.updatedAt ? -1 : 0);
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
      expiresAt: r.expiresAt
    }));
    return { total: pool.length, expired, byKind, entries };
  }
  /** Full inventory with stats (title-level listing to bound token cost). */
  async index(input) {
    const now = input.now ?? Date.now();
    await this.purgeExpired(now);
    const kinds = (input.kinds ?? []).filter(isValidKind);
    const tags = normalizeTags(input.tags);
    const scope = input.scope && isValidScope(input.scope) ? input.scope : void 0;
    let pool = this.all();
    if (kinds.length > 0) pool = pool.filter((r) => kinds.includes(r.kind));
    if (tags.length > 0) pool = pool.filter((r) => tags.every((t) => r.tags.includes(t)));
    if (scope) pool = pool.filter((r) => r.scope === scope);
    const expired = pool.filter((r) => isExpired(r, now)).length;
    const byKind = {};
    for (const r of pool) byKind[r.kind] = (byKind[r.kind] ?? 0) + 1;
    pool.sort((a, b) => a.updatedAt < b.updatedAt ? 1 : a.updatedAt > b.updatedAt ? -1 : 0);
    const offset = Math.max(0, Math.round(input.offset ?? 0));
    const limit = Math.max(1, Math.min(100, Math.round(input.limit ?? 20)));
    const entries = pool.slice(offset, offset + limit).map((r) => ({
      id: r.id,
      content: truncate(r.content, 121),
      kind: r.kind,
      tags: r.tags,
      scope: r.scope,
      importance: r.importance,
      updatedAt: r.updatedAt
    }));
    return { total: pool.length, expired, byKind, entries };
  }
  /** Delete by id, or by tag set (optionally scoped). importance-3 ids need confirm. */
  async forget(input) {
    this.guardWrite();
    const now = input.now ?? Date.now();
    const tags = normalizeTags(input.tags);
    const scope = input.scope && isValidScope(input.scope) ? input.scope : void 0;
    let deleted = 0;
    let skippedImportant = 0;
    if (input.id) {
      const record = this.table.get(input.id);
      if (!record) return { deleted: 0, skippedImportant: 0 };
      if (record.importance === 3 && input.confirm !== true) {
        throw new HarnessError(`memory ${input.id} is importance 3; pass confirm: true to delete`, "MEMORY_CONFIRM_REQUIRED");
      }
      deleted = await this.delete(input.id) ? 1 : 0;
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
      throw new HarnessError("forget requires an id or at least one tag", "MEMORY_FORGET_TARGET_REQUIRED");
    }
    return { deleted, skippedImportant };
  }
  /** Manual edit from the UI panel: overwrite content/tags/importance directly. */
  async updateContent(input) {
    this.guardWrite();
    const now = input.now ?? Date.now();
    const record = this.table.get(input.id);
    if (!record) return { id: input.id, updated: false };
    const content = input.content !== void 0 ? truncate(input.content.trim(), this.config.maxContentChars) : record.content;
    if (content === "") throw new HarnessError("memory content is empty", "MEMORY_EMPTY_CONTENT");
    const tags = input.tags !== void 0 ? normalizeTags(input.tags) : record.tags;
    const importance = input.importance !== void 0 ? Math.min(3, Math.max(1, Math.round(input.importance))) : record.importance;
    await this.update(input.id, (current) => ({
      ...current,
      content,
      tags,
      importance,
      updatedAt: iso(now)
    }));
    return { id: input.id, updated: true };
  }
  stats() {
    const byKind = {};
    for (const r of this.all()) byKind[r.kind] = (byKind[r.kind] ?? 0) + 1;
    return { total: this.table.size, byKind };
  }
};
var PROTOCOL_SECTION = `## Long-term memory (dsh-agent-memory)

You have a cross-session long-term memory, and relevant memories are auto-injected into each step based on the user's current message (look for a "\u76F8\u5173\u8BB0\u5FC6" block \u2014 treat it as context, not as the user speaking).

Rules:
- At the START of a task, run memory_recall with keywords from the request before diving in; use memory_index when you need the big picture of what is already known. The auto-injected block is a preview \u2014 recall for details.
- DURING work, actively memory_remember anything the user will need in LATER sessions: facts about them or their environment, stated preferences, concluded decisions and their rationale, lessons from mistakes ("never do X again"), and cross-session todos. Quality over quantity \u2014 do not log noise.
- When the user corrects outdated knowledge, fix memory right away: pass the stale record's id in \`supersedes\` when storing the new one. The old record stays stored (auditable) but is heavily down-weighted, so it stops being presented as the current conclusion. Use memory_forget only when a memory is *wrong*, not merely *outdated*.
- When the user corrects you, or the same mistake recurs twice, immediately memory_remember it as a lesson (kind=lesson, importance=3) \u2014 the lesson then auto-injects on related topics, preventing recurrence.
- Write lessons DIALECTICALLY \u2014 never absolute. Distinguish: (1) what went wrong and how to avoid it; (2) the CONDITIONS that caused the failure (would it have worked under different conditions? note the applicability boundary \u2014 a lesson's value depends on its conditions); (3) anything that actually worked \u2014 keep the salvageable part. Prefer "under condition A, X failed because B; the C part worked" over "X is impossible". When new evidence contradicts an old lesson (e.g. it worked under changed conditions), UPDATE the old lesson rather than stacking a new one.
- Treat the memory store dialectically too: (1) contradicting memories are not necessarily wrong \u2014 separate conditions and timing before judging; two records can both be true under different conditions; (2) when a new decision supersedes an old one, mark the old record as superseded ("\u5DF2\u7531 <X> \u66F4\u65B0/\u8986\u76D6") instead of leaving silent contradictions; (3) before forgetting a memory, confirm its conditions are truly gone \u2014 do not discard it merely because it feels outdated.
- kind: one of fact / preference / decision / lesson / todo / note. tags: short lowercase words. importance: 1 nice-to-know, 2 useful, 3 critical (eviction-proof). scope: user (applies everywhere) or project (this project only).
- Near-duplicate entries merge automatically; do not re-create the same memory on purpose.
- **Memory retrieval relevance is multi-dimensional \u2014 never single-indicator:** (1) semantic similarity (not just keyword overlap), (2) task relevance (does it directly help complete the current task?), (3) pattern match (same scenario pattern?), (4) causal link (prerequisite/cause/consequence?), (5) recency (recent may be better, but old memories have value too). Apply if \u22652 dimensions match; otherwise discard. Hot-count(auto-injected) memories are a safety net only \u2014 always run memory_recall for task-specific retrieval.
- **Anti-decay rules:** The memory system itself can "decay" (forget rules over long conversations). Countermeasures: (1) memory_retrieval at task start acts as a fresh re-read of core rules; (2) the SAME mistake recurring twice triggers an automatic lesson-importance-3 nudge; (3) if you notice yourself drifting from your configured behavior, use memory_recall to retrieve your core rules and re-ground yourself.`;
function jsonOutput() {
  return {
    schema: { type: "json" },
    render: (_args, value) => [{ type: "text", text: JSON.stringify(value, null, 2) }]
  };
}
async function apply(ctx, config) {
  const existing = ctx.storageDomain.get(memoryDomain.name);
  let domain = existing ?? await ctx.storageDomain.open(memoryDomain);
  if (!existing) {
    ctx.effect(() => () => {
      void domain.close();
    });
  }
  const storePath = join2(process.env.DSH_HOME ?? join2(homedir(), ".dsh"), "storages", "memory.json");
  const fingerprint = () => {
    try {
      const st = statSync(storePath);
      return `${st.mtimeMs}:${st.size}`;
    } catch {
      return null;
    }
  };
  let baseFingerprint = fingerprint();
  const guard = () => {
    const cur = fingerprint();
    if (cur !== null && baseFingerprint !== null && cur !== baseFingerprint) {
      throw new HarnessError("memory.json \u5DF2\u88AB\u8FDB\u7A0B\u5916\u4FEE\u6539(\u53EF\u80FD\u811A\u672C\u6216\u53E6\u4E00\u4E2A DSH \u5B9E\u4F8B\u6539\u8FC7)\u3002\u4E3A\u9632\u6B62\u8986\u76D6,\u672C\u6B21\u5199\u5165\u5DF2\u62D2\u7EDD\u3002\u8BF7\u8C03\u7528 memory_reload \u5408\u5E76\u5916\u90E8\u4FEE\u6539,\u6216\u91CD\u542F DSH\u3002", "MEMORY_EXTERNAL_MODIFIED");
    }
  };
  const refreshFingerprint = () => {
    baseFingerprint = fingerprint();
  };
  const semanticConfig = config.semanticEnabled && config.semanticModelDir !== "" ? {
    modelDir: config.semanticModelDir,
    cacheDir: config.semanticCacheDir !== "" ? config.semanticCacheDir : join2(process.env.DSH_HOME ?? join2(homedir(), ".dsh"), "storages", "memory-semantic")
  } : null;
  const coreConfig = { ...config, semantic: semanticConfig };
  const makeCore = () => new MemoryCore(domain.table("records"), coreConfig, guard, refreshFingerprint);
  let core = makeCore();
  ctx.inject(["connection", "webServer"], (webContext) => {
    if (webContext?.connection === void 0) return;
    registerRpc(
      { connection: webContext.connection, webServer: webContext.webServer },
      () => core,
      (message) => ctx.logger?.warn?.(message)
    );
  });
  ctx.tools.register(defineTool({
    name: "memory_remember",
    description: "Store one durable cross-session memory. Use proactively for user facts, preferences, decisions, lessons, and todos that matter beyond the current session. Near-duplicate content merges into the existing record; the store evicts lowest-value records when full.",
    parameters: {
      content: { type: "string", required: true, description: "What to remember. One clear statement; write it so a future session understands it without this conversation." },
      kind: { type: "string", enum: [...MEMORY_KINDS], description: "Memory kind: fact / preference / decision / lesson / todo / note. Defaults to note when omitted." },
      tags: { type: "array", items: { type: "string" }, description: 'Short lowercase lookup tags, e.g. ["deploy","sensitive"]. Max 16.' },
      scope: { type: "string", enum: [...MEMORY_SCOPES], description: "user = applies in every project; project = this project only." },
      project: { type: "string", description: "For scope=project: the project path this memory belongs to. Fill from your known working directory ({{cwd}})." },
      importance: { type: "number", description: "1 nice-to-know, 2 useful, 3 critical (never evicted for space). Defaults to 2." },
      ttl_days: { type: "number", description: "Optional: auto-expire after N days (e.g. temporary credentials, short-lived decisions). Max 3650." },
      supersedes: {
        type: "array",
        items: { type: "string" },
        description: "Ids of older memories this one REPLACES (e.g. a decision that changed). They stay stored \u2014 auditable \u2014 but are heavily down-weighted so they stop being presented as the current conclusion. Use whenever a previously stored conclusion no longer holds."
      }
    },
    output: jsonOutput(),
    execute(args) {
      return core.remember({
        content: String(args.content ?? ""),
        kind: String(args.kind ?? "note"),
        tags: Array.isArray(args.tags) ? args.tags.map(String) : void 0,
        scope: String(args.scope ?? "project"),
        project: typeof args.project === "string" && args.project !== "" ? args.project : void 0,
        importance: typeof args.importance === "number" ? args.importance : void 0,
        ttlDays: typeof args.ttl_days === "number" ? args.ttl_days : void 0,
        supersedes: Array.isArray(args.supersedes) ? args.supersedes.map(String) : void 0
      });
    }
  }));
  ctx.tools.register(defineTool({
    name: "memory_recall",
    description: "Search long-term memory. Run at task start with keywords from the user request; scores by relevance, importance, recency, and past usefulness. Also updates access stats so frequently used memories rank higher.",
    parameters: {
      query: { type: "string", required: true, description: 'Keywords or a phrase, e.g. "deployment pipeline" or "user prefers".' },
      kinds: { type: "array", items: { type: "string", enum: [...MEMORY_KINDS] }, description: "Optional kind filter." },
      tags: { type: "array", items: { type: "string" }, description: "Optional: only records carrying ALL these tags." },
      scope: { type: "string", enum: [...MEMORY_SCOPES], description: "Optional scope filter." },
      project: { type: "string", description: "Optional: only project-scoped records of this project (plus all user-scoped ones)." },
      limit: { type: "number", description: "Max results, default 3." },
      content_max: { type: "number", description: "Optional: max chars of each result content shown to the model (default 400). Truncated results are enough to judge relevance; lower = cheaper calls." }
    },
    output: jsonOutput(),
    execute(args) {
      return core.recall({
        query: String(args.query ?? ""),
        kinds: Array.isArray(args.kinds) ? args.kinds.map(String) : void 0,
        tags: Array.isArray(args.tags) ? args.tags.map(String) : void 0,
        scope: typeof args.scope === "string" ? args.scope : void 0,
        project: typeof args.project === "string" && args.project !== "" ? args.project : void 0,
        limit: typeof args.limit === "number" ? args.limit : void 0,
        contentMax: typeof args.content_max === "number" ? args.content_max : void 0
      });
    }
  }));
  ctx.tools.register(defineTool({
    name: "memory_index",
    description: "List the memory inventory (title-level entries, newest first) with counts by kind. Use to see what is already known without burning context on full recall.",
    parameters: {
      kinds: { type: "array", items: { type: "string", enum: [...MEMORY_KINDS] }, description: "Optional kind filter." },
      tags: { type: "array", items: { type: "string" }, description: "Optional: only records carrying ALL these tags." },
      scope: { type: "string", enum: [...MEMORY_SCOPES], description: "Optional scope filter." },
      limit: { type: "number", description: "Max entries, default 20." },
      offset: { type: "number", description: "Pagination offset, default 0." }
    },
    output: jsonOutput(),
    execute(args) {
      return core.index({
        kinds: Array.isArray(args.kinds) ? args.kinds.map(String) : void 0,
        tags: Array.isArray(args.tags) ? args.tags.map(String) : void 0,
        scope: typeof args.scope === "string" ? args.scope : void 0,
        limit: typeof args.limit === "number" ? args.limit : void 0,
        offset: typeof args.offset === "number" ? args.offset : void 0
      });
    }
  }));
  ctx.tools.register(defineTool({
    name: "memory_forget",
    description: "Delete memories: by exact id, or every record carrying ALL given tags (optionally scoped). Importance-3 records require confirm: true. Use to retire outdated or wrong memories.",
    parameters: {
      id: { type: "string", description: "Exact memory id from recall/index." },
      tags: { type: "array", items: { type: "string" }, description: "Delete all records with ALL these tags (used when id is omitted)." },
      scope: { type: "string", enum: [...MEMORY_SCOPES], description: "Optional scope filter for tag deletion." },
      confirm: { type: "boolean", description: "Required to delete importance-3 records." }
    },
    output: jsonOutput(),
    execute(args) {
      return core.forget({
        id: typeof args.id === "string" && args.id !== "" ? args.id : void 0,
        tags: Array.isArray(args.tags) ? args.tags.map(String) : void 0,
        scope: typeof args.scope === "string" ? args.scope : void 0,
        confirm: args.confirm === true
      });
    }
  }));
  let lastSedimentAt = 0;
  ctx.tools.register(defineTool({
    name: "memory_sediment",
    description: "Batch-persist durable memories at once (facts/decisions/lessons/todos), e.g. at session wind-down or right after a user correction. The agent summarizes what it already knows \u2014 no extra LLM call. Guardrails: max entries per call and a cooldown window prevent noise pollution.",
    parameters: {
      entries: {
        type: "array",
        required: true,
        description: `1-${config.sedimentMaxEntries} memory entries. Each: {content (required), kind: fact|preference|decision|lesson|todo|note, tags: string[], scope: user|project, project: string, importance: 1-3}.`,
        items: {
          type: "object",
          additionalProperties: false,
          properties: {
            content: { type: "string" },
            kind: { type: "string" },
            tags: { type: "array", items: { type: "string" } },
            scope: { type: "string" },
            project: { type: "string" },
            importance: { type: "number" }
          }
        }
      }
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
        const e = item ?? {};
        const content = String(e.content ?? "").trim();
        if (content === "") continue;
        const r = await core.remember({
          content,
          kind: String(e.kind ?? "note"),
          tags: Array.isArray(e.tags) ? e.tags.map(String) : void 0,
          scope: String(e.scope ?? "project"),
          project: typeof e.project === "string" && e.project !== "" ? e.project : void 0,
          importance: typeof e.importance === "number" ? e.importance : void 0
        });
        if (r.merged) merged += 1;
        else stored += 1;
      }
      if (cooldownMs > 0) lastSedimentAt = now;
      return { stored, merged, cooldown: false, nextInMs: 0 };
    }
  }));
  ctx.tools.register(defineTool({
    name: "memory_reload",
    description: "Reopen the memory domain from disk. Use after memory.json was modified by something outside this DSH process (script, another instance, manual edit): the external changes are loaded and the in-memory store is replaced. Never loses data \u2014 but check the returned count.",
    parameters: {},
    output: jsonOutput(),
    execute: async () => {
      await domain.close();
      domain = await ctx.storageDomain.open(memoryDomain);
      core = makeCore();
      refreshFingerprint();
      const { total, byKind } = core.stats();
      return { reloaded: true, total, byKind };
    }
  }));
  ctx.tools.register(defineTool({
    name: "memory_import",
    description: 'Import memories from a JSONL or JSON file through the domain write chain (not by editing memory.json), so the file and the in-memory store stay in sync. Record shape per line: {"content":"...","kind":"fact|preference|decision|lesson|todo|note","tags":["..."],"scope":"user|project","importance":1-3}.',
    parameters: {
      file: { type: "string", required: true, description: "Absolute path to a .jsonl or .json file with memory records." }
    },
    output: jsonOutput(),
    execute: async (args) => {
      const file = String(args.file ?? "");
      const raw = readFileSync2(file, "utf8").replace(/^\uFEFF/, "");
      const entries = raw.trimStart().startsWith("[") ? JSON.parse(raw) : raw.split(/\r?\n/).filter((l) => l.trim()).map((l) => JSON.parse(l));
      let added = 0;
      let skipped = 0;
      for (const entry of entries) {
        const content = String(entry?.content ?? entry?.summary ?? "").trim();
        if (!content) continue;
        const kind = isValidKind(String(entry?.kind ?? "")) ? String(entry?.kind) : "note";
        const scope = isValidScope(String(entry?.scope ?? "")) ? String(entry?.scope) : "project";
        const importance = Math.min(3, Math.max(1, Math.round(Number(entry?.importance) || 2)));
        const tags = normalizeTags(Array.isArray(entry?.tags) ? entry.tags.map(String) : void 0);
        const id = `mem_${randomUUID().replaceAll("-", "").slice(0, 16)}`;
        const now = iso(Date.now());
        if (domain.table("records").get(id)) {
          skipped += 1;
          continue;
        }
        await core.remember({ content, kind, tags, scope, importance });
        added += 1;
      }
      const { total, byKind } = core.stats();
      return { imported: added, skipped, total, byKind };
    }
  }));
  if (config.protocolSection) {
    ctx.systemPrompt.section({ name: "memory-protocol", order: 110, text: PROTOCOL_SECTION });
  }
  if (config.injectEnabled && config.injectCount > 0) {
    let lastInjectedKey = "";
    ctx.on("agent/pre-step", async (payload, next) => {
      const decision = await next();
      if (decision.kind !== "enter") return decision;
      const query = queryFromMessages(payload.messages);
      if (query === "" || !hasMeaningfulQuery(query)) return decision;
      let rec;
      try {
        rec = await core.recall({ query, limit: Math.max(config.injectCount, 10), touch: false });
      } catch {
        return decision;
      }
      if (rec.results.length === 0) return decision;
      if (rec.noMatch) return decision;
      const filtered = selectForInjection(rec.results, {
        minScore: config.injectMinScore,
        budgetRatio: config.injectBudgetRatio,
        limit: config.injectCount
      });
      if (filtered.length === 0) return decision;
      const key = filtered.map((r) => r.id).sort().join(",");
      if (key === lastInjectedKey) return decision;
      lastInjectedKey = key;
      const text = renderInjection(filtered, config.injectMaxChars);
      const memMessage = createUserMessage({
        content: [{ type: "text", text }],
        source: {
          kind: "plugin",
          plugin: "memory",
          form: "notice",
          summary: boundContextSummary(`\u76F8\u5173\u8BB0\u5FC6 ${rec.results.length} \u6761`)
        }
      });
      return { ...decision, messages: [...decision.messages, memMessage] };
    });
  }
  if (config.lessonizeEnabled && config.lessonizeAfter > 0) {
    const errorCounts = /* @__PURE__ */ new Map();
    ctx.on("agent/error", async (payload) => {
      const fingerprint2 = extractErrorFingerprint(payload.error);
      if (fingerprint2 === "unknown-error" || fingerprint2 === "") return;
      const n = (errorCounts.get(fingerprint2) ?? 0) + 1;
      if (n < config.lessonizeAfter) {
        errorCounts.set(fingerprint2, n);
        return;
      }
      errorCounts.delete(fingerprint2);
      try {
        payload.agent.inject(createUserMessage({
          content: [{
            type: "text",
            text: `\u68C0\u6D4B\u5230\u540C\u7C7B\u9519\u8BEF\u7B2C ${n} \u6B21:${fingerprint2}\u3002\u8BF7\u7528 memory_remember \u628A\u8FD9\u6761\u7ECF\u9A8C\u56FA\u5316\u4E3A\u6559\u8BAD(kind=lesson, importance=3),\u5E76\u8FA9\u8BC1\u603B\u7ED3:\u2460\u5177\u4F53\u9519\u5728\u54EA\u3001\u600E\u4E48\u89C4\u907F \u2461\u5931\u8D25\u7684\u6761\u4EF6\u662F\u4EC0\u4E48(\u6362\u6761\u4EF6\u662F\u5426\u53EF\u884C,\u6807\u6CE8\u9002\u7528\u8FB9\u754C) \u2462\u8FD9\u6B21\u6709\u6CA1\u6709\u5176\u5B9E\u6709\u6548\u7684\u90E8\u5206\u503C\u5F97\u4FDD\u7559\u3002\u9632\u6B62\u518D\u6B21\u53D1\u751F,\u4E5F\u907F\u514D\u628A\u6761\u4EF6\u6027\u5931\u8D25\u8BB0\u6210\u7EDD\u5BF9\u7ED3\u8BBA\u3002`
          }],
          source: {
            kind: "plugin",
            plugin: "memory",
            form: "notice",
            summary: boundContextSummary(`\u540C\u7C7B\u9519\u8BEF ${n} \u6B21,\u5EFA\u8BAE\u56FA\u5316\u6559\u8BAD`)
          }
        }));
      } catch {
      }
    });
  }
}
export {
  Config,
  DEFAULT_SEMANTIC_MIN,
  DEFAULT_SEMANTIC_WEIGHT,
  IMPORTANCE_BOOST_SLOPE,
  LEXICAL_SATURATION,
  MATCH_BASE_MIN,
  MEMORY_DEFAULTS,
  MEMORY_KINDS,
  MEMORY_READ_CHANNEL,
  MEMORY_SCOPES,
  MEMORY_WRITE_CHANNEL,
  MemoryCore,
  MemoryRecordSchema,
  SUPERSEDED_PENALTY,
  SemanticIndex,
  apply,
  bm25Signal,
  cjkUnigrams,
  countHybridMatches,
  explainRecord,
  extractErrorFingerprint,
  hasMeaningfulQuery,
  hotRecords,
  hotnessScore,
  inject,
  isExpired,
  isValidKind,
  isValidScope,
  jaccard,
  memoryDomain,
  name,
  queryFromMessages,
  rankRecords,
  rankRecordsHybrid,
  recordBoost,
  registerRpc,
  renderInjection,
  scoreRecord,
  selectForInjection,
  tokenSet,
  tokenSetBigram,
  tokenize,
  tokenizeBigram
};
//# sourceMappingURL=index.js.map
