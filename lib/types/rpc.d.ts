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
export declare const MEMORY_READ_CHANNEL = "/dsh-memory-read";
export declare const MEMORY_WRITE_CHANNEL = "/dsh-memory-write";
export interface RpcResult<T = unknown> {
    ok: boolean;
    value?: T;
    error?: {
        code: string;
        message: string;
        details: Record<string, unknown>;
    };
}
type RpcHandler = (method: string, payload: Record<string, unknown>) => Promise<RpcResult> | RpcResult;
/** Core is read lazily so a `memory_reload` swap is seen by later panel requests. */
type CoreGetter = () => MemoryCore | undefined;
/** The slice of Connection's Host service this module touches. */
export interface RpcConnection {
    rpc?: {
        handle(channel: string, handler: RpcHandler, options?: {
            authority?: string;
        }): unknown;
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
export declare function registerRpc(webContext: RpcWebContext, getCore: CoreGetter, warn?: (message: string) => void): void;
export {};
//# sourceMappingURL=rpc.d.ts.map