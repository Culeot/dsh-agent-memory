// One-command real-machine smoke test for dsh-agent-memory.
// Verifies the full chain in a separate headless DSH process: plugin load,
// domain open, remember, recall, persistence file, then cleans up.
//
// Prereqs: dsh on PATH; headless profile patched with storage trio + dsh-agent-memory
// (see presets/README.md and ~/.dsh/profiles/headless/cordis.patch.yml).
// Usage: node scripts/smoke.mjs
import { execFileSync } from 'node:child_process';
import { copyFileSync, existsSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';

const DSH_HOME = process.env.DSH_HOME ?? join(homedir(), '.dsh');
const STORAGE = join(DSH_HOME, 'storages', 'memory.json');
// dsh CLI entry, resolved from the shared profile node_modules so the script
// does not depend on PATH or platform-specific launchers (.ps1/.cmd).
const DSH_BIN = join(DSH_HOME, 'profiles', 'node_modules', '@deepseek-ai', 'dsh', 'lib', 'bin.js');
const MARKER = `冒烟验证${Date.now()}`;

function fail(msg) {
  console.error(`[smoke] FAIL: ${msg}`);
  process.exit(1);
}

function run() {
  console.log('[smoke] launching headless session (remember + recall)...');
  const prompt = `用 memory_remember 记住一条记忆:内容='${MARKER} 冒烟测试 通过',kind='fact',scope='user',tags=['smoke']。然后用 memory_recall 查询'冒烟测试' 并原样报告返回的 JSON。`;
  let out;
  try {
    out = execFileSync(process.execPath, [DSH_BIN, '--profile', 'headless', prompt], {
      encoding: 'utf8',
      timeout: 240_000,
      maxBuffer: 64 * 1024 * 1024,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
  } catch (err) {
    const detail = err.stderr ? String(err.stderr).slice(0, 800) : String(err.message).slice(0, 800);
    fail(`headless process failed: ${detail}`);
  }
  return out;
}

function assertPersisted() {
  if (!existsSync(STORAGE)) fail(`persistence file missing: ${STORAGE}`);
  const raw = readFileSync(STORAGE, 'utf8');
  if (!raw.includes(MARKER)) fail('persisted record does not contain the smoke marker');
  const stats = statSync(STORAGE);
  console.log(`[smoke] persistence ok (${stats.size} bytes, marker found)`);
}

// 1. preconditions
if (!existsSync(join(DSH_HOME, 'profiles', 'headless', 'package.json'))) {
  fail('headless profile not initialized — run: dsh --profile headless "hi"');
}
const patched = readFileSync(join(DSH_HOME, 'profiles', 'headless', 'cordis.patch.yml'), 'utf8');
if (!patched.includes("dsh-agent-memory")) {
  // headless 的 patch 层默认是**刻意留空**的(见其文件头注释:dsh-base 已内置挂载
  // storage 三件套,重复挂会报 "duplicate loader entry id")。所以在没手动把插件
  // 挂到 headless 的机器上,这条真实链路冒烟本来就跑不了 —— 这是"环境不适用",
  // 不是"插件坏了"。用 SKIP + 退出码 0 表达,但把话说清楚;需要强制失败时加 --strict。
  const msg = 'headless profile 未挂载 dsh-agent-memory(该 profile 的 patch 刻意留空),真实链路冒烟无法执行';
  if (process.argv.includes('--strict')) fail(msg);
  console.log(`[smoke] SKIP: ${msg}`);
  console.log('[smoke]       要跑真实链路:把插件挂到 headless profile,或用 web profile 的部署自检:');
  console.log('[smoke]         node scripts/check-deploy.mjs web');
  process.exit(0);
}
if (!existsSync(join(DSH_HOME, 'profiles', 'node_modules', 'dsh-agent-memory')) &&
    !existsSync(join(DSH_HOME, 'profiles', 'headless', 'node_modules', 'dsh-agent-memory'))) {
  fail('dsh-agent-memory not installed into headless profile');
}

// 2. run
const output = run();
if (!output.includes(MARKER)) fail('recall did not surface the remembered content');
if (!/mem_[0-9a-f]{16}/.test(output)) fail('recall result missing a memory id');
console.log('[smoke] remember+recall round trip ok');

// 3. persistence
assertPersisted();

// 4. cleanup — ONLY remove the probe record written by this run.
// 绝不能整文件删除:该文件是所有会话共享的生产记忆库(web/headless 同源),
// 早期版本的 rmSync(STORAGE) 一旦在已挂载记忆的机器上跑通,会清空全部历史记忆。
const store = JSON.parse(readFileSync(STORAGE, 'utf8'));
const records = store?.tables?.records ?? {};
const probes = Object.keys(records).filter(
  (id) => typeof records[id]?.content === 'string' && records[id].content.includes(MARKER),
);
if (probes.length !== 1) {
  fail(`cleanup aborted: expected exactly 1 probe record, found ${probes.length} — store left untouched`);
}
copyFileSync(STORAGE, `${STORAGE}.smoke-backup`);
for (const id of probes) delete records[id];
writeFileSync(STORAGE, JSON.stringify(store, null, 2), 'utf8');
console.log(
  `[smoke] PASS — full chain verified, probe ${probes[0]} removed, ${Object.keys(records).length} records kept (backup: ${STORAGE}.smoke-backup)`,
);
