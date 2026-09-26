// 存储兼容性自检 —— 部署前必须过的一关。
//
// 每一次改 MemoryRecord schema 都在赌一件事:磁盘上那些**老记录**还能不能解析。
// 如果新增字段被写成必填,或者校验比之前更严,插件在重启后会读不出记忆库 ——
// 这正是"部署之后用不了"最危险的一种形态:不是崩溃,而是数据读不出来。
//
// 这个脚本拿**真实的 memory.json** 逐条过一遍当前 schema,报告任何一条失败。
// 只读,不写。
//
// 用法:node scripts/check-schema.mjs [memory.json 路径]
import { readFileSync, existsSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { MemoryRecordSchema, MEMORY_KINDS, MEMORY_SCOPES } from '../lib/index.js';

const DSH_HOME = process.env.DSH_HOME ?? join(homedir(), '.dsh');
const storePath = process.argv[2] ?? join(DSH_HOME, 'storages', 'memory.json');

if (!existsSync(storePath)) {
  console.log(`[schema] 找不到记忆库:${storePath}(没有数据就没有兼容性问题)`);
  process.exit(0);
}

const raw = JSON.parse(readFileSync(storePath, 'utf8'));
const table = raw?.tables?.records ?? {};
const rows = Array.isArray(table) ? table : Object.values(table);
console.log(`[schema] 记忆库 ${storePath}`);
console.log(`[schema] 记录 ${rows.length} 条,逐条校验...`);

const failures = [];
let withSuperseded = 0;
const kindCounts = new Map();
const scopeCounts = new Map();

for (const row of rows) {
  if (!row || typeof row !== 'object') {
    failures.push({ id: '(not an object)', issue: 'record is not an object' });
    continue;
  }
  const parsed = MemoryRecordSchema.safeParse(row);
  if (!parsed.success) {
    const first = parsed.error?.issues?.[0];
    failures.push({
      id: row.id ?? '(no id)',
      issue: first ? `${first.path?.join('.') || '(root)'}: ${first.message}` : 'unknown validation error',
    });
    continue;
  }
  if (row.supersededBy) withSuperseded += 1;
  kindCounts.set(row.kind, (kindCounts.get(row.kind) ?? 0) + 1);
  scopeCounts.set(row.scope, (scopeCounts.get(row.scope) ?? 0) + 1);
}

console.log(`[schema] 通过 ${rows.length - failures.length} / ${rows.length}`);
console.log(`[schema] 已知 kind:${MEMORY_KINDS.join(', ')}`);
console.log(`[schema] 已知 scope:${MEMORY_SCOPES.join(', ')}`);
console.log(`[schema] 实际分布 kind=${JSON.stringify(Object.fromEntries(kindCounts))} scope=${JSON.stringify(Object.fromEntries(scopeCounts))}`);
console.log(`[schema] 带 supersededBy 的记录:${withSuperseded}(新增的可选字段,老记录没有它是正常的)`);

if (failures.length > 0) {
  console.error(`\n[schema] FAIL:${failures.length} 条记录过不了当前 schema —— 部署后这些记忆会读不出来`);
  for (const f of failures.slice(0, 10)) console.error(`[schema]   ${f.id}  ${f.issue}`);
  if (failures.length > 10) console.error(`[schema]   ...还有 ${failures.length - 10} 条`);
  process.exit(1);
}

console.log('\n[schema] PASS —— 当前 schema 能读全部历史记录,重启后不会出现"数据读不出来"');
