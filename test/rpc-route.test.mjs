// 回归测试:面板通道的 HTTP 路由注册与信封处理。
// 背景:DSH 0.1.5-rc.2 的 connection.rpc.handle() 会抛
//   cannot get property "webServer" without inject
// 导致 /dsh-memory-read 通道静默丢失(面板报 HTTP 405)。这里锁住两条路径:
//   ① handle() 抛错 → 自己经 webServer 注册 prefix 路由,信封/鉴权行为与官方对齐
//   ② handle() 可用 → 走官方实现,不重复注册
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { registerRpc, MEMORY_READ_CHANNEL, MEMORY_WRITE_CHANNEL } from '../lib/index.js';

function fakeCore(overrides = {}) {
  const calls = [];
  return {
    calls,
    stats: () => ({ total: 3, byKind: { lesson: 2, fact: 1 } }),
    inspect: (options) => { calls.push(['inspect', options]); return { total: 3, expired: 0, entries: [] }; },
    getById: (id) => (id === 'mem_x' ? { id: 'mem_x', content: 'hi' } : undefined),
    recall: async (request) => { calls.push(['recall', request]); return { totalMatched: 1, returned: 1, results: [] }; },
    forget: async (request) => { calls.push(['forget', request]); return { deleted: 1, skippedImportant: 0 }; },
    updateContent: async (request) => { calls.push(['update', request]); return { updated: true }; },
    remember: async (request) => { calls.push(['remember', request]); return { id: 'mem_new', merged: false, evicted: 0, content: request.content }; },
    ...overrides,
  };
}

/** Node-style request: async-iterable body plus method/url/headers. */
function makeReq({ method = 'POST', url, body, contentType = 'application/json' } = {}) {
  const chunks = body === undefined ? [] : [Buffer.from(body, 'utf8')];
  return {
    method,
    url,
    headers: contentType === null ? {} : { 'content-type': contentType },
    async *[Symbol.asyncIterator]() {
      for (const chunk of chunks) yield chunk;
    },
  };
}

function makeRes() {
  const out = { status: undefined, headers: undefined, body: undefined };
  return {
    out,
    writeHead(status, headers) { out.status = status; out.headers = headers; },
    end(body) { out.body = body; },
  };
}

/** Collect the routes a registerRpc() call mounts through a stub webServer. */
function mount({ handleThrows = true, rejection } = {}) {
  const routes = [];
  const connection = {
    requestRejection: rejection === undefined ? () => undefined : rejection,
    rpc: {
      handle(channel) {
        if (handleThrows) throw new Error('cannot get property "webServer" without inject');
        routes.push({ kind: 'official', path: channel });
      },
    },
  };
  const webServer = {
    register(route) { routes.push({ kind: 'self', route }); return () => {}; },
  };
  const warnings = [];
  const core = fakeCore();
  registerRpc({ connection, webServer }, () => core, (message) => warnings.push(message));
  return {
    routes,
    warnings,
    core,
    route: (channel) => routes.find((entry) => entry.kind === 'self' && entry.route.path === channel)?.route,
    call: (channel, req, res) => (routes.find((entry) => entry.kind === 'self' && entry.route.path === channel)?.route.handler(req, res)),
  };
}

const envelope = (body) => JSON.parse(body);

describe('panel channel registration', () => {
  it('falls back to self-registered routes when connection.rpc.handle throws', () => {
    const { routes, warnings } = mount();
    assert.equal(routes.filter((entry) => entry.kind === 'self').length, 2);
    assert.deepEqual(
      routes.map((entry) => entry.route.path).sort(),
      [MEMORY_READ_CHANNEL, MEMORY_WRITE_CHANNEL].sort(),
    );
    assert.ok(routes.every((entry) => entry.route.kind === 'prefix'));
    assert.equal(warnings.length, 2, '两个通道各留一条降级告警');
  });

  it('prefers the official handle() when it works (no duplicate route)', () => {
    const { routes } = mount({ handleThrows: false });
    assert.deepEqual(
      routes.map((entry) => entry.path).sort(),
      [MEMORY_READ_CHANNEL, MEMORY_WRITE_CHANNEL].sort(),
    );
    assert.equal(routes.filter((entry) => entry.kind === 'self').length, 0);
  });

  it('answers stats with a server-response envelope echoing rpcId', async () => {
    const { call } = mount();
    const res = makeRes();
    await call(MEMORY_READ_CHANNEL, makeReq({
      url: `${MEMORY_READ_CHANNEL}/stats`,
      body: JSON.stringify({ type: 'client-request', rpcId: 'r1', method: 'stats', payload: {} }),
    }), res);

    assert.equal(res.out.status, 200);
    const body = envelope(res.out.body);
    assert.equal(body.type, 'server-response');
    assert.equal(body.rpcId, 'r1');
    assert.deepEqual(body.result, { ok: true, value: { total: 3, byKind: { lesson: 2, fact: 1 }, expired: 0 } });
  });

  it('routes write-channel methods to the core', async () => {
    const { call, core } = mount();
    const res = makeRes();
    await call(MEMORY_WRITE_CHANNEL, makeReq({
      url: `${MEMORY_WRITE_CHANNEL}/forget`,
      body: JSON.stringify({ type: 'client-request', rpcId: 'r2', method: 'forget', payload: { id: 'mem_x' } }),
    }), res);

    assert.deepEqual(envelope(res.out.body).result, { ok: true, value: { deleted: 1, skippedImportant: 0 } });
    assert.deepEqual(core.calls[0], ['forget', { id: 'mem_x', confirm: false }]);
  });

  it('applies the connection fence before doing any work', async () => {
    const { call, core } = mount({ rejection: () => 401 });
    const res = makeRes();
    await call(MEMORY_READ_CHANNEL, makeReq({
      url: `${MEMORY_READ_CHANNEL}/stats`,
      body: JSON.stringify({ type: 'client-request', rpcId: 'r3', method: 'stats', payload: {} }),
    }), res);

    assert.equal(res.out.status, 401);
    assert.equal(res.out.body, 'unauthorized');
    assert.equal(core.calls.length, 0);
  });

  it('rejects non-POST, wrong content type, bad JSON and mismatched method', async () => {
    const { call } = mount();

    const notPost = makeRes();
    await call(MEMORY_READ_CHANNEL, makeReq({ method: 'GET', url: `${MEMORY_READ_CHANNEL}/stats` }), notPost);
    assert.equal(notPost.out.status, 404);

    const wrongType = makeRes();
    await call(MEMORY_READ_CHANNEL, makeReq({ url: `${MEMORY_READ_CHANNEL}/stats`, body: '{}', contentType: 'text/plain' }), wrongType);
    assert.equal(wrongType.out.status, 415);

    const badJson = makeRes();
    await call(MEMORY_READ_CHANNEL, makeReq({ url: `${MEMORY_READ_CHANNEL}/stats`, body: 'not json' }), badJson);
    assert.equal(badJson.out.status, 400);

    const mismatch = makeRes();
    await call(MEMORY_READ_CHANNEL, makeReq({
      url: `${MEMORY_READ_CHANNEL}/stats`,
      body: JSON.stringify({ type: 'client-request', rpcId: 'r4', method: 'list', payload: {} }),
    }), mismatch);
    const body = envelope(mismatch.out.body);
    assert.equal(body.result.ok, false);
    assert.equal(body.result.error.code, 'bad-request');
  });

  it('reports a not-ready core instead of throwing', async () => {
    const routes = [];
    const connection = { requestRejection: () => undefined, rpc: { handle() { throw new Error('boom'); } } };
    const webServer = { register(route) { routes.push(route); return () => {}; } };
    registerRpc({ connection, webServer }, () => undefined);

    const res = makeRes();
    await routes[0].handler(makeReq({
      url: `${MEMORY_READ_CHANNEL}/stats`,
      body: JSON.stringify({ type: 'client-request', rpcId: 'r5', method: 'stats', payload: {} }),
    }), res);

    const body = envelope(res.out.body);
    assert.equal(body.result.ok, false);
    assert.equal(body.result.error.message, 'memory core not ready');
  });

  it('stays a no-op when the profile has no web server (headless)', () => {
    const { routes } = mount();
    assert.ok(routes.length > 0);
    const headless = [];
    registerRpc({ connection: { rpc: { handle() { throw new Error('boom'); } } } }, () => undefined, (m) => headless.push(m));
    assert.deepEqual(headless, [], '没有 webServer 时既不注册也不刷告警');
  });
});
