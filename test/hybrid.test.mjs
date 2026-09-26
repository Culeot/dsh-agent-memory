// Unit tests for the hybrid (lexical + semantic) ranking path and the optional
// semantic channel's degradation behaviour.
//   npm run build && npm test
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  rankRecordsHybrid,
  rankRecords,
  countHybridMatches,
  recordBoost,
  SemanticIndex,
  MemoryCore,
  MATCH_BASE_MIN,
} from '../lib/index.js';

const FIXED_NOW = Date.parse('2026-09-25T00:00:00Z');

function rec(id, content, extra = {}) {
  return {
    id,
    content,
    kind: 'note',
    tags: [],
    scope: 'project',
    project: null,
    importance: 2,
    createdAt: '2026-09-01T00:00:00Z',
    updatedAt: '2026-09-20T00:00:00Z',
    accessedAt: null,
    accessCount: 0,
    expiresAt: null,
    ...extra,
  };
}

// "npm 发布" matches mem_exact lexically; mem_para only matches by meaning.
const RECORDS = [
  rec('mem_exact', 'npm 发布 token 过期 处理'),
  rec('mem_para', '把包推到仓库时凭证失效该怎么办'),
];

describe('hybrid ranking', () => {
  it('degrades to pure lexical ranking when no semantic channel is given', () => {
    const hybrid = rankRecordsHybrid(RECORDS, 'npm 发布', 5, { now: FIXED_NOW });
    const lexical = rankRecords(RECORDS, 'npm 发布', 5, { now: FIXED_NOW });
    assert.deepEqual(hybrid.map((x) => x.record.id), lexical.map((x) => x.record.id));
    assert.ok(hybrid.length > 0, 'lexical hit must survive the degradation path');
  });

  it('treats a wrong-length vector array as "channel unavailable"', () => {
    const ranked = rankRecordsHybrid(RECORDS, 'npm 发布', 5, { now: FIXED_NOW, semantic: [0.9] });
    const lexical = rankRecords(RECORDS, 'npm 发布', 5, { now: FIXED_NOW });
    assert.deepEqual(ranked.map((x) => x.record.id), lexical.map((x) => x.record.id));
  });

  it('lets a strong paraphrase win with zero lexical overlap', () => {
    // Weight is explicit on purpose: this asserts the fusion MECHANISM, not
    // whatever the calibrated default happens to be.
    const ranked = rankRecordsHybrid(RECORDS, 'npm 发布', 2, {
      now: FIXED_NOW,
      semantic: [0.2, 0.95],
      semanticWeight: 0.95,
    });
    assert.equal(ranked[0].record.id, 'mem_para');
    assert.match(ranked[0].reasons[0], /^fuse:sem=0\.95/);
  });

  it('admits records found by EITHER channel and drops those found by neither', () => {
    const semantic = [0.9, 0.1];
    const ranked = rankRecordsHybrid(RECORDS, 'npm 发布', 5, { now: FIXED_NOW, semantic });
    assert.deepEqual(ranked.map((x) => x.record.id), ['mem_exact']);
    const looser = rankRecordsHybrid(RECORDS, 'npm 发布', 5, {
      now: FIXED_NOW,
      semantic,
      semanticMin: 0.05,
    });
    assert.equal(looser.length, 2, 'a lower semantic floor re-admits the paraphrase');
  });

  it('countHybridMatches agrees with what rankRecordsHybrid admits', () => {
    for (const semantic of [null, [0.9, 0.1], [0.1, 0.9], [0.0, 0.0]]) {
      const opts = { now: FIXED_NOW, semantic };
      const admitted = rankRecordsHybrid(RECORDS, 'npm 发布', 999, opts).length;
      assert.equal(countHybridMatches(RECORDS, 'npm 发布', opts), admitted);
    }
  });

  it('applies business boosts after fusion, so importance cannot flip relevance', () => {
    // The historical bug in one assertion: a critical-but-irrelevant record used
    // to outrank a relevant one through a 2.5x importance multiplier.
    const records = [
      rec('mem_critical', '完全无关的内容', { importance: 3 }),
      rec('mem_relevant', 'npm 发布 token', { importance: 1 }),
    ];
    const ranked = rankRecordsHybrid(records, 'npm 发布', 2, { now: FIXED_NOW, semantic: [0, 0] });
    assert.equal(ranked[0].record.id, 'mem_relevant');
  });

  it('falls back to hotness ordering for empty queries (semantic not consulted)', () => {
    const ranked = rankRecordsHybrid(RECORDS, '', 2, { now: FIXED_NOW, semantic: [0, 0] });
    assert.equal(ranked.length, 0, 'empty query with no lexical match produces nothing');
  });
});

describe('importance boost slope', () => {
  it('slope 0 removes importance from the score entirely', () => {
    const low = rec('mem_x', 'same content', { importance: 1 });
    const high = rec('mem_x', 'same content', { importance: 3 });
    assert.equal(
      recordBoost(low, { boostSlope: 0, now: FIXED_NOW }, FIXED_NOW),
      recordBoost(high, { boostSlope: 0, now: FIXED_NOW }, FIXED_NOW),
    );
  });

  it('the default slope is gentler than the historical 0.75', () => {
    // Pin recency at 1 (updatedAt == now) so this asserts the importance slope
    // alone; otherwise the recency factor leaks into the expected number.
    const fresh = rec('mem_x', 'content', {
      importance: 3,
      updatedAt: new Date(FIXED_NOW).toISOString(),
    });
    const gentle = recordBoost(fresh, { now: FIXED_NOW }, FIXED_NOW);
    const historical = recordBoost(fresh, { now: FIXED_NOW, boostSlope: 0.75 }, FIXED_NOW);
    assert.ok(Math.abs(gentle - 1.2) < 1e-9, `default boost at importance 3 must be 1.2x, got ${gentle}`);
    assert.ok(Math.abs(historical - 2.5) < 1e-9, 'slope 0.75 must still reproduce the historical 2.5x');
    assert.ok(gentle < historical, 'the default must be gentler than the old behaviour');
  });

  it('a relevant record survives a critical-but-irrelevant neighbour in the lexical path', () => {
    const records = [
      rec('mem_critical', '完全无关的内容', { importance: 3 }),
      rec('mem_relevant', 'npm 发布 token 流程', { importance: 1 }),
    ];
    const ranked = rankRecords(records, 'npm 发布', 2, { now: FIXED_NOW });
    assert.equal(ranked[0].record.id, 'mem_relevant');
    assert.ok(ranked[0].base >= MATCH_BASE_MIN);
  });
});

describe('optional semantic channel degradation', () => {
  it('reports a reason instead of throwing when the model is absent', async () => {
    const index = new SemanticIndex({
      modelDir: 'Z:/definitely/not/a/model',
      cacheDir: 'Z:/definitely/not/a/cache',
    });
    const ok = await index.ensure([rec('mem_1', 'anything')]);
    assert.equal(ok, false);
    assert.match(String(index.unavailableReason), /no model at/);
    assert.equal(index.isReady, false);
    // And scores() keeps the contract: null means "caller, fall back".
    assert.equal(await index.scores('anything', [rec('mem_1', 'anything')]), null);
  });

  it('returns null for an empty query rather than spending an embedding', async () => {
    const index = new SemanticIndex({
      modelDir: 'Z:/definitely/not/a/model',
      cacheDir: 'Z:/definitely/not/a/cache',
    });
    assert.equal(await index.scores('   ', [rec('mem_1', 'x')]), null);
  });
});

// ---- 元记忆闸门:宁可不给,也不硬凑 ------------------------------------------
//
// 记忆系统最坏的行为不是"没想起",而是硬凑一条无关的:它花掉注入预算、
// 误导判断,而且用户不会察觉那是错的。这组测试钉住闸门的两侧 ——
// 库里没有时要敢说没有,库里有答案时不许误报。

function makeTable() {
  const map = new Map();
  return {
    map,
    get(key) { return map.get(key); },
    entries() { return map.entries(); },
    async put(key, value) { map.set(key, value); },
    async update(key, fn) {
      if (!map.has(key)) throw new Error('missing-key');
      const next = fn(map.get(key));
      map.set(key, next);
      return next;
    },
    async delete(key) { return map.delete(key); },
    get size() { return map.size; },
  };
}

function makeCore(configOverrides = {}) {
  return new MemoryCore(makeTable(), {
    maxRecords: 500,
    maxContentChars: 2000,
    mergeSimilarity: 0.7,
    recencyHalfLifeDays: 90,
    ...configOverrides,
  });
}

const BANK = [
  { content: 'npm 发布 token 过期 处理流程:先去 vault 换新 token 再重试', kind: 'lesson', tags: ['npm'] },
  { content: 'dsh 插件构建用 esbuild,产物是 lib/index.js', kind: 'fact', tags: ['dsh'] },
];

describe('元记忆闸门(no-match gate)', () => {
  it('库里没有答案时报告 noMatch,而不是硬凑一条', async () => {
    const core = makeCore({ noMatchThreshold: 0.18 });
    for (const entry of BANK) await core.remember({ ...entry, scope: 'project', now: FIXED_NOW });
    const rec = await core.recall({ query: '佛卡夏面包怎么做', touch: false, now: FIXED_NOW });
    assert.equal(rec.noMatch, true);
    assert.ok(rec.matchStrength < 0.18, `matchStrength=${rec.matchStrength}`);
  });

  it('库里有答案时不误报 noMatch', async () => {
    const core = makeCore({ noMatchThreshold: 0.18 });
    for (const entry of BANK) await core.remember({ ...entry, scope: 'project', now: FIXED_NOW });
    const rec = await core.recall({ query: 'npm 发布 token 过期', touch: false, now: FIXED_NOW });
    assert.equal(rec.noMatch, false);
    assert.ok(rec.matchStrength >= 0.18, `matchStrength=${rec.matchStrength}`);
    assert.ok(rec.results.length > 0);
  });

  it('阈值 0 关闭闸门(向后兼容)', async () => {
    const core = makeCore({ noMatchThreshold: 0 });
    for (const entry of BANK) await core.remember({ ...entry, scope: 'project', now: FIXED_NOW });
    const rec = await core.recall({ query: '佛卡夏面包怎么做', touch: false, now: FIXED_NOW });
    assert.equal(rec.noMatch, false);
  });

  it('空查询(hot 集合)不受闸门影响', async () => {
    const core = makeCore({ noMatchThreshold: 0.9 });
    await core.remember({ content: '随便一条记忆内容', kind: 'note', scope: 'project', now: FIXED_NOW });
    const rec = await core.recall({ query: '', touch: false, now: FIXED_NOW });
    assert.equal(rec.noMatch, false);
  });

  it('融合路径给出 0..1 尺度的 relevance', () => {
    const ranked = rankRecordsHybrid(RECORDS, 'npm 发布', 5, { now: FIXED_NOW, semantic: [0.2, 0.95] });
    assert.ok(ranked[0].relevance > 0 && ranked[0].relevance <= 1);
  });
});
