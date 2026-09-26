// 诊断:dsh-agent-memory 的浏览器面板报
//   Error: transport failure for /dsh-memory-read/stats: HTTP 405
// 的根因探针。
//
// 背景(2026-09-19 实测):DSH 0.1.5-rc.2 里 connection.rpc.handle(channel, handler)
// 内部走 `owner.webServer` 取路由注册口,而 cordis 的 tracker 让 `this.ctx` 解析成
// **调用方的 ctx**;当调用方解析不到 webServer 时,handle() 直接抛
//   Error: cannot get property "webServer" without inject
// 自定义通道于是从未挂上,POST 落到 SPA 静态兜底(frontend-static 对非 GET/HEAD 回 405)。
// DSH 自己的 /api 通道不受影响——它先 `ctx.inject(['webServer'], …)` 再注册。
//
// 用法:node test/probe-rpc-mount.mjs
// 判据:只有 self-register 两种拓扑能挂上路由;rpc-handle 在 C(同级提供)失败。
import { pathToFileURL } from 'node:url';

const NM = 'C:/Users/李弘毅/.dsh/profiles/web/node_modules';
const { Context } = await import(pathToFileURL(`${NM}/@deepseek-ai/cordis/lib/index.js`).href);
const { HostConnectionService } = await import(pathToFileURL(`${NM}/@deepseek-ai/dsh-client-connection/lib/index.js`).href);

const stubAuth = { isAuthenticated: () => true, authorizeIndex: () => false, authenticatedUrl: (u) => u };
const makeWebServer = (mounted) => ({ register(route) { mounted.push(route.path); return () => {}; } });

/**
 * @param name 场景名(同时作通道名,必须 ASCII)
 * @param globalWebServer true = webServer 注册在根 ctx(全局可见);false = 由同级 entry 提供
 * @param mode 'rpc-handle' 走 DSH 的 connection.rpc.handle;'self-register' 自己 inject 后注册
 */
async function scenario(name, { globalWebServer, mode }) {
  const root = new Context();
  const mounted = [];
  if (globalWebServer) {
    root.provide('webServer', makeWebServer(mounted));
  } else {
    root.inject([], (c) => { c.provide('webServer', makeWebServer(mounted)); });
  }
  // connection 服务放在自己的 entry fiber 里(与 DSH 装载结构一致)
  root.inject([], (c) => { new HostConnectionService(c, [], stubAuth); });

  await new Promise((resolve) => {
    const fiber = root.inject(['connection', 'webServer'], (c) => {
      try {
        if (mode === 'rpc-handle') {
          c.connection.rpc.handle(`/probe-${name}`, () => ({ ok: true, value: {} }));
        } else {
          c.effect(() => c.webServer.register({ kind: 'prefix', path: `/probe-${name}`, handler: () => {} }), 'probe');
        }
        console.log(`[${name}] 注册未抛错`);
      } catch (error) {
        console.log(`[${name}] 抛错: ${error.message.split('\n')[0]}`);
      }
      resolve();
    });
    fiber?.then?.(() => {}, (e) => { console.log(`[${name}] 子 fiber 失败: ${e.message.split('\n')[0]}`); resolve(); });
    setTimeout(resolve, 300);
  });
  console.log(`[${name}] 挂载结果: ${JSON.stringify(mounted)}\n`);
}

await scenario('A-global-rpc-handle', { globalWebServer: true, mode: 'rpc-handle' });
await scenario('B-global-self-register', { globalWebServer: true, mode: 'self-register' });
await scenario('C-sibling-rpc-handle', { globalWebServer: false, mode: 'rpc-handle' });
await scenario('D-sibling-self-register', { globalWebServer: false, mode: 'self-register' });
