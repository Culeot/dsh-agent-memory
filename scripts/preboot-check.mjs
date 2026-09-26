// 重启前验证 —— 能在这里查的，现在就查掉。
//
// 不启动 DSH(那会中断会话)，只验证两件事:
//   1. 插件模块能被 Node 加载(语法/依赖有没有断)
//   2. 配置文件语法正确、新字段都在、类型对
//
// 用法: node scripts/preboot-check.mjs
import { readFileSync, existsSync } from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const fail = (msg) => { console.error(`✗ ${msg}`); process.exit(1); };

// ---- 1. 模块能加载 ----
try {
  const mod = await import('../lib/index.js');
  console.log(`✓ 插件模块能加载  name=${mod.name}  version=${mod.version ?? '(读不到)'}`);
  for (const key of ['rankRecordsHybrid', 'countHybridMatches', 'recordBoost', 'selectForInjection',
                    'SemanticIndex', 'MemoryCore', 'MemoryRecordSchema']) {
    if (!(key in mod)) fail(`导出缺失: ${key}`);
  }
  console.log('✓ 关键导出都在(融合/闸门/预算化/语义通道/记忆核心)');
} catch (e) {
  fail(`插件模块加载失败: ${e.message}`);
}

// ---- 2. 配置 ----
const DSH_HOME = process.env.DSH_HOME ?? `${process.env.USERPROFILE}\\.dsh`;
const patchPath = `${DSH_HOME}/profiles/web/cordis.patch.yml`;
if (!existsSync(patchPath)) fail(`配置文件不存在: ${patchPath}`);
const yaml = readFileSync(patchPath, 'utf8');

// 2a. 语法:能解析成 YAML 数组
let parsed;
try {
  const { parse } = await import('yaml').catch(() => import('js-yaml')).catch(() => ({}));
  if (parse) {
    parsed = parse(yaml);
    if (!Array.isArray(parsed)) fail('配置顶层不是数组');
    console.log('✓ 配置语法正确(可解析为 YAML 数组)');
  } else {
    console.log('~ 没有 yaml 解析器,跳过语法检查(用关键字检查代替)');
  }
} catch (e) {
  fail(`配置语法错误: ${e.message}`);
}

// 2b. memory 段与新字段
const memMatch = yaml.match(/-\s*id:\s*memory[\s\S]*?(?=\n\s*-\s*id:|\s*$)/);
if (!memMatch) fail('配置里找不到 memory 段');
const section = memMatch[0];
for (const key of ['semanticEnabled', 'semanticModelDir', 'semanticCacheDir',
                   'semanticWeight', 'boostSlope', 'noMatchThreshold']) {
  if (!section.includes(key)) fail(`memory 段缺少新配置项: ${key}`);
}
console.log('✓ memory 段包含全部 6 个新配置项');

// 2c. 类型与取值(用文本检查,不依赖 YAML 解析器)
const expect = {
  semanticEnabled: 'true',
  semanticModelDir: 'C:/ai项目/jev-as-llm/models/mem-retriever-onnx',
  semanticCacheDir: 'C:/Users/李弘毅/.dsh/storages/memory-semantic',
  semanticWeight: '0.3',
  boostSlope: '0.1',
  noMatchThreshold: '0.18',
};
for (const [k, v] of Object.entries(expect)) {
  const re = new RegExp(`${k}:\\s*([^\\s#]+)`);
  const m = re.exec(section);
  if (!m) { fail(`${k} 读不到值`); continue; }
  if (m[1] !== v) fail(`${k} = ${m[1]},预期是 ${v}`);
}
console.log('✓ 新配置项的取值全部正确');

// 2d. 模型目录
for (const p of ['config.json', 'tokenizer.json', 'onnx/model_quantized.onnx']) {
  if (!existsSync(`${expect.semanticModelDir}/${p}`)) fail(`模型文件缺失: ${p}`);
}
console.log('✓ 模型文件齐全(config / tokenizer / int8 ONNX)');

// 2e. 备份还在吗
const backups = readFileSync(patchPath, 'utf8'); // 确认文件可读
console.log(`✓ 配置文件可读(${DSH_HOME}/profiles/web/cordis.patch.yml)`);
const bak = `${patchPath}.bak-20260926-105518`;
if (existsSync(bak)) console.log(`✓ 备份还在: ${bak.split('\\').pop()}`);

console.log('\n结论:能在不启动 DSH 的前提下查的,全过了。');
console.log('仍不能确定的:真实 DSH 进程加载插件 + 配置解析(那是启动时才发生的)。');
