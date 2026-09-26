/**
 * UI-facing RPC layer: lets the web client panel browse and manage memories
 * without going through the model tools. Two channels:
 *   - `memory-read`  — stats / list / search / get
 *   - `memory-write` — forget / update / remember
 * Read channel never mutates the store (uses MemoryCore.inspect).
 *
 * ## 为什么这里自己注册 HTTP 路由(2026-09-19)
 *
 * DSH 0.1.5-rc.2 的 `connection.rpc.handle(channel, handler)` 是坏的:它内部用
 * `owner.webServer.register(route)` 挂路由,而 cordis 的 tracker 让 `this.ctx`
 * 解析成**调用方的 ctx**;调用方解析不到 `webServer` 时,handle() 抛
 *   Error: cannot get property "webServer" without inject
 * 于是通道静默丢失——面板的 POST /dsh-memory-read/stats 落到 SPA 静态兜底
 * (frontend-static 对非 GET/HEAD 一律 405),前端报
 *   Error: transport failure for /dsh-memory-read/stats: HTTP 405
 * 上游 0.1.6-alpha.2 的 register() 与 0.1.5-rc.2 逐字相同,尚未修复。
 *
 * 对策:优先尝试官方 handle()(DSH 修好后自动回到官方实现),失败则自己经
 * `ctx.inject(['connection','webServer'])` 拿到的 webServer 注册 prefix 路由,
 * 并复用 `connection.requestRejection()` 的 Host/Origin 围栏与浏览器鉴权。
 * 行为与官方 rpcFetchHandler 对齐(信封、content-type、method↔endpoint 校验)。
 *
 * @module dsh-agent-memory/rpc
 */
import type { MemoryCore } from './index.ts';

export const MEMORY_READ_CHANNEL = '/dsh-memory-read';
export const MEMORY_WRITE_CHANNEL = '/dsh-memory-write';

/** Body cap for panel requests: they only carry queries and short text edits. */
const MAX_BODY_BYTES = 1 << 20;

export interface RpcResult<T = unknown> {
  ok: boolean;
  value?: T;
  error?: { code: string; message: string; details: Record<string, unknown> };
}

type RpcHandler = (method: string, payload: Record<string, unknown>) => Promise<RpcResult> | RpcResult;

/** Core is read lazily so a `memory_reload` swap is seen by later panel requests. */
type CoreGetter = () => MemoryCore | undefined;

/** The slice of Connection's Host service this module touches. */
export interface RpcConnection {
  rpc?: {
    handle(channel: string, handler: RpcHandler, options?: { authority?: string }): unknown;
  };
  /** Host/Origin fence + browser auth: 403 / 401, or undefined when allowed. */
  requestRejection?(request: unknown): number | undefined;
}

/** The slice of the Web server service this module touches. */
export interface HttpRoute {
  kind: 'prefix';
  path: string;
  handler: (req: any, res: any) => void | Promise<void>;
}

export interface WebServer {
  register(route: HttpRoute): () => void;
}

/** Context shape handed to {@link registerRpc} by the injected plugin scope. */
export interface RpcWebContext {
  connection?: RpcConnection;
  webServer?: WebServer;
}

function ok(value: unknown): RpcResult {
  return { ok: true, value };
}

function fail(code: string, message: string): RpcResult {
  return { ok: false, error: { code, message, details: {} } };
}

function str(payload: Record<string, unknown>, key: string): string | undefined {
  const v = payload[key];
  return typeof v === 'string' && v !== '' ? v : undefined;
}

function arr(payload: Record<string, unknown>, key: string): string[] | undefined {
  const v = payload[key];
  return Array.isArray(v) ? v.map(String) : undefined;
}

function num(payload: Record<string, unknown>, key: string): number | undefined {
  const v = payload[key];
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined;
}

function notReady(): RpcResult {
  return fail('internal', 'memory core not ready');
}

function createReadHandler(getCore: CoreGetter): RpcHandler {
  return async (method, payload) => {
    try {
      const core = getCore();
      if (core === undefined) return notReady();
      switch (method) {
        case 'stats': {
          const { total, byKind } = core.stats();
          const expired = core.inspect({}).expired;
          return ok({ total, byKind, expired });
        }
        case 'list':
          return ok(core.inspect({
            kinds: arr(payload, 'kinds'),
            tags: arr(payload, 'tags'),
            scope: str(payload, 'scope'),
            limit: num(payload, 'limit'),
            offset: num(payload, 'offset'),
          }));
        case 'get': {
          const id = str(payload, 'id');
          if (!id) return fail('bad-request', 'id required');
          const record = core.getById(id);
          return record ? ok(record) : fail('not-found', `no memory ${id}`);
        }
        case 'search': {
          const query = str(payload, 'query');
          if (!query) return fail('bad-request', 'query required');
          const result = await core.recall({
            query,
            limit: num(payload, 'limit') ?? 10,
            touch: false,
            contentMax: num(payload, 'content_max'),
          });
          return ok(result);
        }
        default:
          return fail('bad-request', `unknown method ${method}`);
      }
    } catch (error) {
      return fail('internal', error instanceof Error ? error.message : String(error));
    }
  };
}

function createWriteHandler(getCore: CoreGetter): RpcHandler {
  return async (method, payload) => {
    try {
      const core = getCore();
      if (core === undefined) return notReady();
      switch (method) {
        case 'forget': {
          const id = str(payload, 'id');
          if (!id) return fail('bad-request', 'id required');
          const out = await core.forget({ id, confirm: payload.confirm === true });
          return ok(out);
        }
        case 'update': {
          const id = str(payload, 'id');
          if (!id) return fail('bad-request', 'id required');
          const out = await core.updateContent({
            id,
            content: typeof payload.content === 'string' ? payload.content : undefined,
            tags: arr(payload, 'tags'),
            importance: num(payload, 'importance'),
          });
          return ok(out);
        }
        case 'remember': {
          const content = typeof payload.content === 'string' ? payload.content.trim() : '';
          if (content === '') return fail('bad-request', 'content required');
          const out = await core.remember({
            content,
            kind: typeof payload.kind === 'string' && payload.kind !== '' ? payload.kind : 'note',
            tags: arr(payload, 'tags'),
            scope: typeof payload.scope === 'string' && payload.scope !== '' ? payload.scope : 'user',
            importance: num(payload, 'importance'),
          });
          return ok(out);
        }
        default:
          return fail('bad-request', `unknown method ${method}`);
      }
    } catch (error) {
      return fail('internal', error instanceof Error ? error.message : String(error));
    }
  };
}

function writeJson(res: any, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(text),
  });
  res.end(text);
}

/** Read a bounded request body; undefined means "too large". */
async function readBody(req: any): Promise<string | undefined> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk as string);
    size += buf.length;
    if (size > MAX_BODY_BYTES) return undefined;
    chunks.push(buf);
  }
  return Buffer.concat(chunks).toString('utf8');
}

/**
 * Build the prefix route for one channel. Mirrors DSH's own rpcFetchHandler:
 * fence first, POST + application/json only, `method` must equal the endpoint,
 * and the reply is a `server-response` envelope echoing the request's rpcId.
 */
function createHttpRoute(channel: string, handler: RpcHandler, connection: RpcConnection): HttpRoute {
  return {
    kind: 'prefix',
    path: channel,
    async handler(req: any, res: any) {
      const rejection = connection.requestRejection?.(req);
      if (rejection !== undefined) {
        res.writeHead(rejection);
        res.end(rejection === 401 ? 'unauthorized' : 'forbidden');
        return;
      }
      const pathname = new URL(String(req?.url ?? '/'), 'http://localhost').pathname;
      const endpoint = pathname.startsWith(`${channel}/`) ? pathname.slice(channel.length + 1) : undefined;
      if (req?.method !== 'POST' || endpoint === undefined) {
        res.writeHead(404);
        res.end('not found');
        return;
      }
      const contentType = String(req?.headers?.['content-type'] ?? '').split(';')[0]?.trim().toLowerCase();
      if (contentType !== 'application/json') {
        res.writeHead(415);
        res.end('content type must be application/json');
        return;
      }
      const raw = await readBody(req);
      if (raw === undefined) {
        res.writeHead(413);
        res.end('body too large');
        return;
      }
      let message: any;
      try {
        message = JSON.parse(raw);
      } catch {
        res.writeHead(400);
        res.end('body is not JSON');
        return;
      }
      const rpcId = typeof message?.rpcId === 'string' ? message.rpcId : 'invalid-request';
      if (message?.type !== 'client-request' || typeof message?.method !== 'string') {
        writeJson(res, 200, { type: 'server-response', rpcId, result: fail('bad-request', 'invalid client-request message') });
        return;
      }
      if (message.method !== endpoint) {
        writeJson(res, 200, {
          type: 'server-response',
          rpcId,
          result: fail('bad-request', `method ${JSON.stringify(message.method)} does not match endpoint ${JSON.stringify(endpoint)}`),
        });
        return;
      }
      const payload = message.payload !== null && typeof message.payload === 'object'
        ? (message.payload as Record<string, unknown>)
        : {};
      let result: RpcResult;
      try {
        result = await handler(endpoint, payload);
      } catch (error) {
        result = fail('internal', error instanceof Error ? error.message : String(error));
      }
      writeJson(res, 200, { type: 'server-response', rpcId, result });
    },
  };
}

/**
 * Mount both channels on the Web server.
 *
 * Prefers DSH's own `connection.rpc.handle`, which carries its own authority
 * handling; when that throws (0.1.5-rc.2 — see the module header) each channel
 * falls back to a self-registered prefix route guarded by the same browser fence.
 *
 * @param webContext - injected scope holding `connection` and `webServer`.
 * @param getCore - lazy accessor for the live MemoryCore.
 * @param warn - optional sink for a non-fatal registration problem.
 */
export function registerRpc(webContext: RpcWebContext, getCore: CoreGetter, warn?: (message: string) => void): void {
  const connection = webContext?.connection;
  if (connection === undefined) return;

  const channels: Array<[string, RpcHandler]> = [
    [MEMORY_READ_CHANNEL, createReadHandler(getCore)],
    [MEMORY_WRITE_CHANNEL, createWriteHandler(getCore)],
  ];

  const webServer = webContext.webServer;
  for (const [channel, handler] of channels) {
    let viaOfficial = false;
    let officialError: unknown;
    try {
      if (typeof connection.rpc?.handle !== 'function') throw new Error('connection.rpc.handle unavailable');
      connection.rpc.handle(channel, handler, { authority: 'trusted-host' });
      viaOfficial = true;
    } catch (error) {
      officialError = error;
    }
    if (viaOfficial) continue;
    // Headless (no web server) is an expected shape: the panel cannot exist, so stay quiet.
    if (webServer === undefined) continue;
    warn?.(`memory: connection.rpc.handle(${channel}) 不可用(${officialError instanceof Error ? officialError.message : String(officialError)}),改用自注册路由`);
    try {
      webServer.register(createHttpRoute(channel, handler, connection));
    } catch (error) {
      warn?.(`memory: 自注册路由 ${channel} 失败: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
}
