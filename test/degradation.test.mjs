// 降级路径测试 —— "部署之后不会用不了"这件事的证据。
//
// 一个可选加速器最危险的失败方式不是报错,而是**让宿主用不了**。
// 所以这里把每一种"缺一环"的情形都钉住:
//
//   1. 装了模型、但没装可选依赖(部署到干净 profile 的最常见情形)
//   2. 缓存文件损坏 / 是垃圾数据
//   3. 模型目录根本不存在
//   4. 上面任何一种发生时,recall 仍然给出正常的词面结果
//
// 这些测试跑在 dsh-memory 自己的 node_modules 下 —— 那里**本来就没有**装
// @huggingface/transformers,所以第 1 条测的不是模拟,而是真实情形。
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { SemanticIndex, MemoryCore } from '../lib/index.js';

const FIXED_NOW = Date.parse('2026-09-25T00:00:00Z');
const REAL_MODEL = 'C:/ai项目/jev-as-llm/models/mem-retriever-onnx';

function record(id, content, extra = {}) {
  return {
    id, content, kind: 'note', tags: [], scope: 'project', project: null,
    importance: 2, createdAt: '2026-09-01T00:00:00Z', updatedAt: '2026-09-20T00:00:00Z',
    accessedAt: null, accessCount: 0, expiresAt: null, ...extra,
  };
}

function tempDir() {
  return mkdtempSync(join(tmpdir(), 'mem-degrade-'));
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

describe('降级路径(部署安全)', () => {
  it('模型存在但可选依赖未装:报出原因,不抛异常', async (t) => {
    if (!existsSync(join(REAL_MODEL, 'config.json'))) {
      t.skip('本机没有导出的 ONNX 模型,跳过');
      return;
    }
    const cacheDir = tempDir();
    const index = new SemanticIndex({ modelDir: REAL_MODEL, cacheDir });
    const ok = await index.ensure([record('mem_1', '任意内容')]);
    assert.equal(ok, false, '依赖缺失时必须报告不可用,而不是抛异常');
    assert.equal(index.isReady, false);
    assert.ok((index.unavailableReason ?? '').length > 0, '必须给出可读的原因');
    assert.equal(await index.scores('任意查询', [record('mem_1', '任意内容')]), null);
    rmSync(cacheDir, { recursive: true, force: true });
  });

  it('缓存文件是垃圾数据时不崩溃(会当作无缓存处理)', async () => {
    const cacheDir = tempDir();
    writeFileSync(join(cacheDir, 'index.json'), '{ 这不是合法 JSON');
    writeFileSync(join(cacheDir, 'vectors.bin'), Buffer.from([1, 2, 3, 4]));
    const index = new SemanticIndex({ modelDir: 'Z:/definitely/not/here', cacheDir });
    const ok = await index.ensure([record('mem_1', '内容')]);
    assert.equal(ok, false, '模型不存在 → 不可用,但绝不能抛');
    assert.match(String(index.unavailableReason), /no model at/);
    rmSync(cacheDir, { recursive: true, force: true });
  });

  it('依赖缺失只尝试一次,不会每一步都重试', async () => {
    const cacheDir = tempDir();
    const index = new SemanticIndex({ modelDir: 'Z:/nope', cacheDir });
    const first = await index.ensure([record('mem_1', 'x')]);
    const reason1 = index.unavailableReason;
    const second = await index.ensure([record('mem_1', 'x')]);
    assert.equal(first, false);
    assert.equal(second, false);
    assert.equal(index.unavailableReason, reason1, '原因应保持一致,不应每次重算');
    rmSync(cacheDir, { recursive: true, force: true });
  });

  it('语义通道不可用时,recall 依然给出正常的词面结果', async () => {
    // semantic 传 null 正是"没有可用语义通道"时插件内部的形态
    const core = new MemoryCore(makeTable(), {
      maxRecords: 500, maxContentChars: 2000, mergeSimilarity: 0.7,
      recencyHalfLifeDays: 90, semantic: null, semanticWeight: 0.3,
      boostSlope: 0.1, noMatchThreshold: 0.18,
    });
    await core.remember({ content: 'docker 部署流程与回滚步骤', kind: 'decision', scope: 'project', now: FIXED_NOW });
    const rec = await core.recall({ query: 'docker 部署', touch: false, now: FIXED_NOW });
    assert.ok(rec.results.length > 0, '词面路径必须照常工作');
    assert.equal(rec.noMatch, false);
    assert.equal(core.semanticStatus().enabled, false);
    assert.equal(core.semanticStatus().reason, null);
  });

  it('语义通道配置存在但模型缺失时,recall 也不受影响', async () => {
    const core = new MemoryCore(makeTable(), {
      maxRecords: 500, maxContentChars: 2000, mergeSimilarity: 0.7,
      recencyHalfLifeDays: 90,
      semantic: { modelDir: 'Z:/definitely/not/here', cacheDir: tempDir() },
      semanticWeight: 0.3, boostSlope: 0.1, noMatchThreshold: 0.18,
    });
    await core.remember({ content: 'docker 部署流程与回滚步骤', kind: 'decision', scope: 'project', now: FIXED_NOW });
    const rec = await core.recall({ query: 'docker 部署', touch: false, now: FIXED_NOW });
    assert.ok(rec.results.length > 0, '语义通道坏掉不该影响词面召回');
    assert.equal(rec.noMatch, false);
    const status = core.semanticStatus();
    assert.equal(status.enabled, true, '配置存在 → 通道标记为启用');
    assert.equal(status.ready, false, '但实际未就绪');
    assert.ok((status.reason ?? '').length > 0, '必须能报出为什么没就绪');
  });
});
