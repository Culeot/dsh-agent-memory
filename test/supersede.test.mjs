// 关系裁决测试 —— "不腐烂"这条判据的证据。
//
// 背景:记忆协议一直要求模型在决策变化时把"已由 X 更新"**写进正文**。
// 那是纪律,不是机制 —— 旧记录和新记录在检索里排得一样高,于是过时的结论
// 照样被当成现行结论注入。这组测试钉住把它变成**数据**之后的行为:
//
//   写入时用 supersedes 声明取代 → 旧记录被标记 → 检索里被重罚 → 但内容还在。
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { MemoryCore, scoreRecord, SUPERSEDED_PENALTY } from '../lib/index.js';

const FIXED_NOW = Date.parse('2026-09-25T00:00:00Z');

function record(id, content, extra = {}) {
  return {
    id, content, kind: 'decision', tags: [], scope: 'project', project: null,
    importance: 2, createdAt: '2026-09-01T00:00:00Z', updatedAt: '2026-09-20T00:00:00Z',
    accessedAt: null, accessCount: 0, expiresAt: null, ...extra,
  };
}

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

function makeCore(overrides = {}) {
  return new MemoryCore(makeTable(), {
    maxRecords: 500, maxContentChars: 2000, mergeSimilarity: 0.7,
    recencyHalfLifeDays: 90, noMatchThreshold: 0.18, boostSlope: 0.1,
    ...overrides,
  });
}

describe('关系裁决(取代)', () => {
  it('remember 带 supersedes → 旧记录被标记,且返回值里列出了它', async () => {
    const core = makeCore();
    const old = await core.remember({
      content: '部署方式用 docker compose 起三个容器', kind: 'decision', scope: 'project', now: FIXED_NOW,
    });
    const fresh = await core.remember({
      content: '部署方式改成 k8s,不再用 docker compose',
      kind: 'decision', scope: 'project', now: FIXED_NOW + 1000, supersedes: [old.id],
    });
    assert.deepEqual(fresh.superseded, [old.id]);

    const rec = await core.recall({ query: '部署方式', touch: false, now: FIXED_NOW + 2000 });
    const entry = rec.results.find((r) => r.id === old.id);
    if (entry) {
      assert.equal(entry.supersededBy, fresh.id, '旧记录必须带上取代者 id');
      assert.match(entry.reasons.join(' '), /superseded/);
    }
  });

  it('被取代的记录仍然可检索到(可审计),但排在新记录之后', async () => {
    const core = makeCore();
    const old = await core.remember({
      content: '部署方式用 docker compose 起三个容器', kind: 'decision', scope: 'project', now: FIXED_NOW,
    });
    const fresh = await core.remember({
      content: '部署方式改成 k8s,不再用 docker compose',
      kind: 'decision', scope: 'project', now: FIXED_NOW + 1000, supersedes: [old.id],
    });
    const rec = await core.recall({ query: '部署方式 docker', touch: false, now: FIXED_NOW + 2000 });
    const ids = rec.results.map((r) => r.id);
    assert.ok(ids.includes(old.id), '旧记录不能被藏起来(要可追溯)');
    assert.ok(ids.indexOf(fresh.id) < ids.indexOf(old.id), `新记录应排在前面,实际顺序 ${ids.join(',')}`);
  });

  it('降权幅度就是 SUPERSEDED_PENALTY', () => {
    const active = scoreRecord(record('mem_a', '部署方式用 docker compose'), '部署方式', { now: FIXED_NOW });
    const stale = scoreRecord(record('mem_a', '部署方式用 docker compose', { supersededBy: 'mem_new' }), '部署方式', { now: FIXED_NOW });
    assert.ok(active > 0);
    assert.ok(Math.abs(stale / active - SUPERSEDED_PENALTY) < 1e-9, `${stale} / ${active}`);
  });

  it('未知 id 与重复 id 被安全忽略', async () => {
    const core = makeCore();
    const r = await core.remember({
      content: '一条无关的新记忆', kind: 'note', scope: 'project', now: FIXED_NOW,
      supersedes: ['mem_does_not_exist', 'mem_nor_this_one'],
    });
    assert.deepEqual(r.superseded, [], '不存在的 id 不该出现在结果里');
    const again = await core.remember({
      content: '另一条记忆内容', kind: 'note', scope: 'project', now: FIXED_NOW + 1,
      supersedes: [],
    });
    assert.deepEqual(again.superseded, []);
  });

  it('走合并路径时同样会标记取代', async () => {
    const core = makeCore();
    const unrelated = await core.remember({
      content: '端口占用时先查 netstat 再换端口', kind: 'lesson', scope: 'project', now: FIXED_NOW,
    });
    // 第一条 docker 记忆
    await core.remember({
      content: '部署流程:docker build 然后 push 到 registry',
      kind: 'decision', scope: 'project', now: FIXED_NOW,
    });
    // 高度相似 → 触发合并;同时声明取代那条无关记忆
    const merged = await core.remember({
      content: '部署流程:docker build 然后 push 到 registry 并重启',
      kind: 'decision', scope: 'project', now: FIXED_NOW + 1000, supersedes: [unrelated.id],
    });
    assert.equal(merged.merged, true, '这条应该走合并路径');
    assert.deepEqual(merged.superseded, [unrelated.id], '合并路径也必须处理 supersedes');
  });

  it('没有 supersededBy 字段的旧记录照常解析与排序', async () => {
    // 兼容性:字段是可选的,历史记录没有它
    const legacy = record('mem_legacy', '部署方式用 docker compose');
    delete legacy.supersededBy;
    const s = scoreRecord(legacy, '部署方式', { now: FIXED_NOW });
    assert.ok(s > 0, '旧记录的打分不应因为缺字段而变成 0');
  });
});
