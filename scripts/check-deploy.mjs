// 部署自检 —— 在重启 DSH 前后回答同一个问题:这次部署会怎样工作?
//
// 只读,不写:不启动进程、不改配置、不碰记忆库。
// 输出一张诊断表,并给出**结论**:重启后会走"语义融合"还是"降级为纯词面",
// 以及为什么。部署之后如果发现是降级路径,这张表会直接告诉你缺哪一环。
//
// 用法:
//   node scripts/check-deploy.mjs            # 默认检查 web profile
//   node scripts/check-deploy.mjs headless
import { existsSync, readFileSync, statSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';

const DSH_HOME = process.env.DSH_HOME ?? join(homedir(), '.dsh');
const profile = process.argv[2] ?? 'web';
const PROFILE_DIR = join(DSH_HOME, 'profiles', profile);
const PLUGIN_DIR = join(PROFILE_DIR, 'node_modules', 'dsh-agent-memory');
const PLUGIN_DIR_SHARED = join(DSH_HOME, 'profiles', 'node_modules', 'dsh-agent-memory');

const rows = [];
const notes = [];
let fatal = null;

function say(label, value, ok = null) {
  const mark = ok === null ? ' ' : ok ? '✓' : '✗';
  rows.push([mark, label, String(value)]);
  return value;
}

// ---- 1. profile 与插件安装 ----
if (!existsSync(PROFILE_DIR)) {
  fatal = `profile 不存在:${PROFILE_DIR}`;
} else {
  say('profile', PROFILE_DIR, true);
  const dir = existsSync(PLUGIN_DIR) ? PLUGIN_DIR : (existsSync(PLUGIN_DIR_SHARED) ? PLUGIN_DIR_SHARED : null);
  if (!dir) {
    fatal = '该 profile 里没有安装 dsh-agent-memory';
  } else {
    say('插件目录', dir, true);
    const pkgPath = join(dir, 'package.json');
    let version = '(读不到 package.json)';
    if (existsSync(pkgPath)) {
      try { version = JSON.parse(readFileSync(pkgPath, 'utf8')).version ?? '?'; } catch { /* ignore */ }
    }
    say('插件版本', version, true);
    const bundlePath = join(dir, 'lib', 'index.js');
    if (!existsSync(bundlePath)) {
      fatal = `缺少构建产物 ${bundlePath}`;
    } else {
      const bundle = readFileSync(bundlePath, 'utf8');
      const hasHybrid = bundle.includes('rankRecordsHybrid');
      const hasSemantic = bundle.includes('SemanticIndex');
      const hasGate = bundle.includes('noMatch');
      say('构建产物', `${(statSync(bundlePath).size / 1024).toFixed(1)} KB  构建于 ${statSync(bundlePath).mtime.toISOString().slice(0, 16)}`, true);
      say('含融合排序', hasHybrid ? '是' : '否', hasHybrid);
      say('含语义通道', hasSemantic ? '是' : '否', hasSemantic);
      say('含元记忆闸门', hasGate ? '是' : '否', hasGate);
      if (!hasHybrid || !hasSemantic || !hasGate) {
        notes.push('构建产物是旧版:profile 里是拷贝,需要把插件重新同步过去才会生效。');
      }
    }
  }
}

// ---- 2. 可选依赖 ----
const depPaths = [
  join(PROFILE_DIR, 'node_modules', '@huggingface', 'transformers'),
  join(DSH_HOME, 'profiles', 'node_modules', '@huggingface', 'transformers'),
];
const depFound = depPaths.find((p) => existsSync(p)) ?? null;
say('可选依赖 @huggingface/transformers', depFound ?? '未安装', depFound !== null);
if (!depFound) {
  notes.push('没装可选依赖 → 语义通道会在加载时失败并**自动降级为纯词面**(不报错、不影响使用)。');
}

// ---- 3. 配置 ----
let modelDir = null;
let weight = null;
let slope = null;
const patchPath = join(PROFILE_DIR, 'cordis.patch.yml');
if (existsSync(patchPath)) {
  const patch = readFileSync(patchPath, 'utf8');
  const memSection = /-\s*id:\s*memory[\s\S]*?(?=\n\s*-\s*id:|\Z)/.exec(patch);
  if (memSection) {
    const sec = memSection[0];
    const grab = (key) => {
      const m = new RegExp(`${key}\\s*:\\s*([^\\n#]+)`).exec(sec);
      return m ? m[1].trim() : null;
    };
    modelDir = grab('semanticModelDir');
    weight = grab('semanticWeight');
    slope = grab('boostSlope');
    say('配置 memory 段', '已找到', true);
    say('semanticModelDir', modelDir ?? '(未设置 → 语义关闭)', modelDir !== null && modelDir !== '');
    say('semanticWeight', weight ?? '(默认 0.3)', null);
    say('boostSlope', slope ?? '(默认 0.1)', null);
  } else {
    say('配置 memory 段', 'cordis.patch.yml 里没有 memory 行', false);
    notes.push('插件可能由 bundle 层挂载(不在 patch 里),配置需写到对应层。');
  }
} else {
  say('cordis.patch.yml', '不存在', false);
}

// ---- 4. 模型文件 ----
if (modelDir) {
  const cfg = join(modelDir, 'config.json');
  const tok = join(modelDir, 'tokenizer.json');
  const onnxQ8 = join(modelDir, 'onnx', 'model_quantized.onnx');
  const onnxFp32 = join(modelDir, 'onnx', 'model.onnx');
  const onnx = existsSync(onnxQ8) ? onnxQ8 : (existsSync(onnxFp32) ? onnxFp32 : null);
  say('模型 config.json', existsSync(cfg) ? '存在' : '缺失', existsSync(cfg));
  say('模型 tokenizer.json', existsSync(tok) ? '存在' : '缺失', existsSync(tok));
  say('ONNX 图', onnx ? `${onnx.endsWith('quantized.onnx') ? 'int8 量化' : 'fp32'}  ${(statSync(onnx).size / 1e6).toFixed(1)} MB` : '缺失', onnx !== null);
  if (!existsSync(cfg) || !onnx) {
    notes.push('模型文件不全 → 语义通道报告不可用并降级;补齐后重启即可。');
  }
}

// ---- 输出 ----
const w = Math.max(...rows.map((r) => r[1].length));
console.log(`\n部署自检 · profile=${profile}\n${'─'.repeat(w + 34)}`);
for (const [mark, label, value] of rows) {
  console.log(`  ${mark} ${label.padEnd(w)}  ${value}`);
}
console.log('─'.repeat(w + 34));

if (fatal) {
  console.log(`\n结论:**部署不完整** —— ${fatal}`);
  process.exit(1);
}

const allGood = rows.every(([mark]) => mark !== '✗');
if (allGood && modelDir && depFound) {
  console.log('\n结论:**重启后会走「语义融合」路径**(词面 + 语义 + 融合 + 元记忆闸门)。');
} else if (allGood) {
  console.log('\n结论:**重启后会走「纯词面 + 元记忆闸门」路径** —— 插件可用,只是没有语义通道。');
} else {
  console.log('\n结论:**部分能力不可用**(见上面的 ✗)。插件仍可加载并降级运行,不会导致用不了。');
}

if (notes.length > 0) {
  console.log('\n说明:');
  for (const n of notes) console.log(`  · ${n}`);
}
console.log('');
