// 端到端诊断:在 DSH 的真实装载拓扑上验证面板通道。
//
// 与 probe-rpc-mount.mjs 的区别:那个只证明"哪种拓扑能挂路由";这个把
// **编译产物 lib/index.js 里真实的 registerRpc** 放进真实 cordis +
// 真实 HostConnectionService(webServer 由同级 entry 提供,即 DSH 的实际结构),
// 然后发一个真实的 HTTP 形态请求,检查信封、鉴权围栏与错误分支。
//
// 用法:node test/probe-rpc-e2e.mjs
import { pathToFileURL } from 'node:url';

const NM = 'C:/Users/李弘毅/.dsh/profiles/web/node_modules';
const PLUGIN = new URL('../lib/index.js', import.meta.url).href;

const { Context } = await import(pathToFileURL(`${NM}/@deepseek-ai/cordis/lib/index.js`).href);
const { HostConnectionService } = await import(pathToFileURL(`${NM}/@deepseek-ai/dsh-client-connection/lib/index.js`).href);
const { registerRpc, MEMORY_READ_CHANNEL, MEMORY_WRITE_CHANNEL } = await import(PLUGIN);

const stubAuth = { isAuthenticated: () => true, authorizeIndex: () => false, authenticatedUrl: (u) => u };

const core = {
  stats: () => ({ total: 139, byKind: { lesson: 4, fact: 9, decision: 3 } }),
  inspect: () => ({ total: 139, expired: 0, entries: [] }),
  getById: () => undefined,
  recall: async () => ({ totalMatched: 0, returned: 0, results: [] }),
  forget: async () => ({ deleted: 0, skippedImportant: 0 }),
  updateContent: async () => ({ updated: false }),
  remember: async (r) => ({ id: 'mem_probe', merged: false, evicted: 0, content: r.content }),
};

const root = new Context();
const mounted = [];
const routes = new Map();

// webServer 由同级 entry 提供 —— 复刻 DSH 的装载结构(probe C/D 场景的 D)。
root.inject([], (c) => {
  c.provide('webServer', {
    register(route) {
      mounted.push(route.path);
      routes.set(route.path, route);
      return () => {};
    },
  });
});
root.inject([], (c) => { new HostConnectionService(c, ['127.0.0.1:3099'], stubAuth); });

const warns = [];
await new Promise((resolve) => {
  const fiber = root.inject(['connection', 'webServer'], (c) => {
    console.log('[dbg] typeof c.connection =', typeof c.connection, '| typeof c.webServer =', typeof c.webServer);
    console.log('[dbg] ctx.get("webServer") =', typeof c.get?.('webServer'), '| ctx.get("connection") =', typeof c.get?.('connection'));
    try {
      console.log('[dbg] 同步读 c.webServer 是否 === root 提供的那份:', c.webServer === undefined ? 'undefined' : 'present');
    } catch (error) {
      console.log('[dbg] 同步读 c.webServer 抛错:', error.message.split('\n')[0]);
    }
    try {
      c.effect(() => {
        console.log('[dbg] effect 内读 c.webServer =', typeof c.webServer, '| c.connection =', typeof c.connection);
        return () => {};
      }, 'probe:effect-check');
    } catch (error) {
      console.log('[dbg] effect 注册抛错:', error.message.split('\n')[0]);
    }
    try {
      registerRpc({ connection: c.connection, webServer: c.webServer }, () => core, (m) => warns.push(m));
    } catch (error) {
      console.log('registerRpc 抛错:', error.message.split('\n')[0]);
    }
    resolve();
  });
  fiber?.then?.(() => {}, (e) => { console.log('子 fiber 失败:', e.message.split('\n')[0]); resolve(); });
  setTimeout(resolve, 400);
});

console.log('挂载的通道:', JSON.stringify(mounted));
console.log('降级告警条数:', warns.length, warns.length ? `(首条: ${warns[0].slice(0, 60)}…)` : '');
console.log('读通道已挂:', routes.has(MEMORY_READ_CHANNEL), ' 写通道已挂:', routes.has(MEMORY_WRITE_CHANNEL));

const makeReq = (url, body, method = 'POST', contentType = 'application/json') => ({
  method,
  url,
  headers: { 'content-type': contentType, host: '127.0.0.1:3099', origin: 'http://127.0.0.1:3099' },
  async *[Symbol.asyncIterator]() { yield Buffer.from(body, 'utf8'); },
});
const makeRes = () => {
  const out = { status: undefined, body: undefined };
  return { out, writeHead(s) { out.status = s; }, end(b) { out.body = b; } };
};
const call = async (channel, req) => {
  const res = makeRes();
  await routes.get(channel).handler(req, res);
  return res.out;
};
const envelope = (b) => JSON.parse(b);

console.log('\n--- 真实请求 ---');
const stats = await call(MEMORY_READ_CHANNEL, makeReq(
  `${MEMORY_READ_CHANNEL}/stats`,
  JSON.stringify({ type: 'client-request', rpcId: 'probe-1', method: 'stats', payload: {} }),
));
console.log('POST /dsh-memory-read/stats →', stats.status, JSON.stringify(envelope(stats.body)));

const forget = await call(MEMORY_WRITE_CHANNEL, makeReq(
  `${MEMORY_WRITE_CHANNEL}/forget`,
  JSON.stringify({ type: 'client-request', rpcId: 'probe-2', method: 'forget', payload: { id: 'mem_x' } }),
));
console.log('POST /dsh-memory-write/forget →', forget.status, JSON.stringify(envelope(forget.body)));

const notPost = await call(MEMORY_READ_CHANNEL, makeReq(`${MEMORY_READ_CHANNEL}/stats`, '{}', 'GET'));
console.log('GET 同路径 →', notPost.status, '(期望 404)');

// --- headless 拓扑:没有 webServer,必须是安静的 no-op ---
const headlessWarns = [];
const headlessRoot = new Context();
headlessRoot.inject([], (c) => new HostConnectionService(c, ['127.0.0.1:3099'], stubAuth));
await new Promise((resolve) => {
  headlessRoot.inject(['connection'], (c) => {
    registerRpc({ connection: c.connection }, () => core, (m) => headlessWarns.push(m));
    resolve();
  });
  setTimeout(resolve, 300);
});
console.log('\n--- headless 拓扑 ---');
console.log('告警条数:', headlessWarns.length, '(期望 0 —— 无 webServer 时保持安静)');
