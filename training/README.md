# 训练你自己的记忆检索模型

## 第一件事:仓库里没有模型

这个仓库**不含**任何训练好的模型文件。不是忘了放,是故意不放,三个原因:

1. **太大**。基座 412 MB,训完 410 MB,导出的 ONNX 目录 510 MB。塞进 npm 包和 git 仓库都不合适。
2. **别人的模型对你没用**。检索模型学的是「查询长什么样、你的记忆长什么样」。我们这边是中文技术笔记、关键词堆叠式的查法(`语音插件 语音播报 speak tts`),你要是记英文文档,那套权重帮不上忙。
3. **不训也完全能用**。插件默认走词面打分(子串 + 标签 + 双字 + BM25),功能是完整的。语义通道是**加分项**,不是必需品。

所以下面这条路是「你自己动手,把它变成加分项」。

## 训了能好多少

本机 382 条查询(每条记忆配一条,从真实记忆库里挖出来造的),三个口径:

| 方案 | MRR | Recall@1 |
|---|---|---|
| 只用词面 | 0.7248 | 63.4% |
| 只用语义(训好的模型) | 0.7121 | 61.3% |
| **两路融合** | **0.8432** | **76.4%** |

第三行才是重点,**也是这套东西唯一值得投入的地方**。

注意第二行:语义单独跑**打不过**词面。这不是训坏了,是必然的——词面精确吃路径名、命令、变量名这些「一字不差」的东西,语义管的是「换个说法还认得出」。两个通道在不同地方各错各的,合起来才赢。

所以别指望训完换个模型就变强;训完要**开着融合**(插件默认就是融合)。另外这三个数的口径是「同一台机器、同一个查询集」,换融合公式绝对值会变——横向比可以,跨论文比不上。

## 你需要什么

**硬件**

- 有 NVIDIA GPU 最好。本机 RTX 4080 Laptop(12 GB),382 条记忆里拿 306 条训练,跑 30 轮约 2.4 分钟。显存是瓶颈不是算力——batch 到 32 就快顶到 12 GB 了,再大容易爆。
- 纯 CPU 也能跑,慢十几倍,几百条记忆的话喝杯咖啡就完了。
- 磁盘留 3 GB:基座 412 MB + Qwen 1.0 GB + 训练产物 410 MB + ONNX 目录 510 MB。

**软件**

```
Python 3.10+
torch            # 2.6 实测可用,cu124 版能吃到 GPU
transformers     # 5.x 实测可用(脚本已绕过 5.x 的 use_cache 签名坑)
onnxruntime      # 导出量化 + 验证要用,1.30 实测可用
numpy
huggingface_hub  # 下模型用
```

**数据**

一份 DSH 记忆库 `memory.json`(就是 `~/.dsh/storages/memory.json`)。记忆条数少于二三十条的话,训出来不会比词面强多少,建议先攒攒再训。

## 目录约定

七个脚本都在这个目录里,互相 import,别拆开。所有路径从环境变量读,脚本放哪都行:

| 环境变量 | 默认值 | 含义 |
|---|---|---|
| `MEM_TRAIN_WORK` | `work` | 工作目录,所有中间产物和模型都落这儿 |
| `MEM_TRAIN_MEMORY` | `~/.dsh/storages/memory.json` | 你的记忆库 |
| `MEM_TRAIN_BASE` | `models/chinese-roberta-wwm-ext` | 基座模型目录 |
| `MEM_TRAIN_LLM` | `models/Qwen2.5-0.5B-Instruct` | 造数据用的本地小模型 |
| `MEM_TRAIN_EXTRA_ENCODERS` | 空 | 评测时想额外比对的模型,逗号分隔(可选) |

跑完以后 `work/` 长这样:

```
work/
  parts/part1.json ... part4.json   记忆分片
  holdout_ids.json                  抽出来不参与训练的记忆 id
  prep_report.json                  切分统计
  train_pairs.jsonl                 训练对(gen_data.py 产出)
  queries_part1.jsonl ...           评测查询(make_eval.py 产出)
  model/                            训好的 sentence encoder
  model-onnx/                       量化后的 ONNX(插件真正加载的)
  logs/                             每轮 loss
  eval_result.json                  评测结果
```

## 完整流程

下面每一步都单独跑,按顺序来。命令假设你在仓库根目录,并且 `MEM_TRAIN_WORK` 没改过(默认落 `./work`)。

脚本清单(用哪个的时候到了会讲):

| 脚本 | 干什么 |
|---|---|
| `prep.py` | 读记忆库、切片、抽 held-out |
| `gen_data.py` | 造训练对(关键词轨 + LLM 轨) |
| `make_eval.py` | 造评测查询(**和训练数据分开**) |
| `train.py` | 对比学习训练 |
| `export_onnx.py` | 导出 + 量化 + 验证 ONNX |
| `eval.py` | 跑评测,对比词面/语义/融合 |
| `mem_eval.py` | 另一套轻量评测入口(50 条快速自测),本文流程用不到,`eval.py` 会 import 它的打分函数 |

### 0. 装依赖

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install torch transformers onnxruntime numpy huggingface_hub
```

要 GPU 的话,按 PyTorch 官网给你的机器挑对应的 CUDA 版命令装 torch,别用默认的 CPU 轮子。

### 1. 下基座模型

用 `hfl/chinese-roberta-wwm-ext`(中文 RoBERTa,412 MB)。它只是个「中文读得懂」的起点,你要的是在上面继续训出检索能力。

新建一个文件 `download_models.py`,内容:

```python
from huggingface_hub import snapshot_download

snapshot_download("hfl/chinese-roberta-wwm-ext",
                  local_dir="models/chinese-roberta-wwm-ext")

# 造数据用的小模型,不想用 LLM 轨可以注释掉(见第 3 步)
snapshot_download("Qwen/Qwen2.5-0.5B-Instruct",
                  local_dir="models/Qwen2.5-0.5B-Instruct")
```

然后 `python download_models.py`。

国内网络下不动的话,先把镜像环境变量设上:

```powershell
# Windows PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```
```bash
# Linux/macOS
export HF_ENDPOINT=https://hf-mirror.com
```

Qwen2.5-0.5B(1.0 GB)是用来造训练数据的,0.5B 这个尺寸在 CPU 上也能跑。不想下就在第 3 步加 `--kw-only`,代价见下。

### 2. 切分数据

```bash
python training/prep.py
```

它干三件事:读记忆库(丢掉过期的、正文不满 20 字的)、切 4 片、按固定种子抽 20% 当 held-out。本机 382 条记忆里,抽出 76 条 held-out。

**held-out 不参与训练**,这是后面评测能不能信的前提。种子写死在脚本里(`HOLDOUT_SEED = 20260925`),重跑结果一致,别改。

### 3. 造训练对

```bash
# 完整版:关键词轨 4 条/记忆 + LLM 轨 6 条/记忆(要 GPU,或者有耐心)
python training/gen_data.py --kw-per 4 --llm-per 6

# 零 GPU 版:只造关键词轨
python training/gen_data.py --kw-only
```

默认输出 `work/train_pairs.jsonl`——就是第 5 步训练默认读的那个文件,不用手动传路径。

两条轨分开造是有原因的:

- **关键词轨**(模板拼的):从 tags、ASCII 词、高频中文短语里抽词,拼成 2~6 个词的串。为什么要模板而不是让模型写——真实场景里 agent 就是这么查的,而 0.5B 小模型「写成关键词式」写不像,恰好这类权重最大,不能交给它糊弄。
- **LLM 轨**:问句、模糊指代、术语式。这几类需要真的懂语义,模板造不出来。模糊指代最重要,它逼模型学「查询里没出现过的词,靠意思也能找到」。

`--kw-only` 的代价是丢掉 LLM 轨那几类样本,模型对模糊指代的处理会弱一截,但关键词类样本照样有,不至于白训。

一个反向经验:**别把训练数据全做成关键词风格**。我们试过让关键词样本主导训练,融合后的 MRR 从 0.8432 掉到 0.8265。原因是语义通道一旦被训得「像词面」,它就不再是独立信号了,两路一起错。让语义通道保持它自己的毛病,融合才有意义。

### 4. 造评测集

```bash
python training/make_eval.py            # 每条记忆 2 条查询(1 条关键词 + 1 条 LLM)
python training/make_eval.py --kw-only  # 零 GPU
```

**这步不能省。** 训练对是给模型学的,拿它评测等于自己判自己的卷子。`make_eval.py` 用另一个随机种子重新生成查询,而且**目标覆盖全部记忆,包括那 76 条 held-out**——那批记忆模型从没见过,在它们上面得的分才反映泛化能力。

覆盖率有个坑:LLM 轨对每条记忆都能造出查询,关键词轨则会漏掉「关键词池凑不满两个词」的记忆。本机实测 `--kw-only` 只覆盖 301/382 条记忆、54/76 条 held-out。想让评测集覆盖全,就别省 LLM 轨。

产出 `work/queries_part1.jsonl ... queries_part4.jsonl`,格式是 `{"id", "query", "qtype"}`,`eval.py` 直接吃。

诚实说清局限:这批查询仍然是模型生成的,分布比真实用户查询窄。最可信的评测集是从真实会话日志里挖出来的查询(本项目当初从 271 个会话里挖了 157 条)。你要是有这样的日志,优先用它们。

### 5. 训练

```bash
python training/train.py --epochs 30 --batch 32
```

标准对比学习(bi-encoder + 批内负样本):同一批里,query 和它对应的记忆向量拉近,和同批其他记忆推远。思路就是 sentence-transformers 的 `MultipleNegativesRankingLoss`,这里手写实现,没引额外依赖。

`bi-encoder` 的意思是 query 和记忆各自过同一个编码器、进同一个向量空间,然后比余弦相似度——跟「先把两者拼起来再过一个模型」(cross-encoder)不是一回事,后者准但慢到没法给几百条记忆排序。

**batch 越大越好,但别贪**。负样本数就是 batch 大小,这是效果的主要来源。脚本默认 `--batch 64`,而本机 12 GB 显存下 64 会爆,所以上面命令显式传了 32。你显存富裕就往上加,加到崩了退一档;不够就往下调,或者降 `--max-len`。

常用参数(默认值都来自脚本的 `--help`):

| 参数 | 默认 | 说明 |
|---|---|---|
| `--epochs` | 8 | 训练轮数 |
| `--batch` | 64 | 批大小 = 负样本数,吃显存 |
| `--lr` | 3e-5 | 学习率 |
| `--warmup` | 0.1 | 学习率预热比例 |
| `--temp` | 0.05 | InfoNCE 温度:越小越「严」,把正样本拉得更紧 |
| `--max-len` | 192 | 训练时序列截断长度 |
| `--save-every` | 1 | 几轮存一次盘 |
| `--pairs` | `work/train_pairs.jsonl` | 训练对路径 |
| `--out` | `work/model` | 模型输出目录 |
| `--log-dir` | `work/logs` | 日志目录 |

训练日志按轮追加到 `work/logs/`,每行一个 JSON,能看 loss 降没降、批内 top-1 命中率涨没涨。

### 6. 导出 ONNX

插件跑在 Node 进程里,不跑 Python,所以模型得转成 ONNX:

```bash
python training/export_onnx.py
```

这一步做四件事:导出 fp32 图(407 MB)、复制 tokenizer 和 config、把权重动态量化成 int8(102 MB,约四分之一大)、**验证 ONNX 的输出和 PyTorch 对得上**(余弦 > 0.999 才算过)。最后那步别跳过——量化偶尔会把模型弄坏,而验证会直接告诉你。

产物在 `work/model-onnx/`:

```
model-onnx/
  config.json  tokenizer.json  tokenizer_config.json
  onnx/model.onnx              407 MB fp32,插件不用
  onnx/model_quantized.onnx    102 MB int8,插件加载的就是它
  export_report.json
```

导出参数:

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model` | `work/model` | 源模型目录 |
| `--out` | `work/model-onnx` | 导出目录 |
| `--max-len` | 512 | 导出时的序列长度上限 |

**关于 `--max-len` 两个默认值不一样**(训练 192、导出 512):训练时截得短只影响显存和速度,导出的是推理上限,两者不必相等,**但导出值不能小于训练值**——比训练短会切掉训练时见过的长度,白丢信息。想统一就两边都传同一个数。

### 7. 评测

```bash
python training/eval.py --run           # 需要第 4 步的 queries_part*.jsonl
python training/eval.py --run --no-semantic   # 只跑词面,快
```

不加 `--run` 它只打印帮助就退出。跑之前确认 `work/queries_part*.jsonl` 存在,否则它会提示「没有 queries_part*.jsonl」。

它跑四组:词面基线、语义单通道、分数融合(会网格搜权重)、RRF 倒数排名融合,再按查询类型和「记忆是否见过训练」分组拆开看。结果写 `work/eval_result.json`。

**重点看 held-out 那一组**:seen 高分说明模型记住了训练数据,held-out 高分才说明真会检索。

想对比自己别的模型,用 `MEM_TRAIN_EXTRA_ENCODERS` 传路径进来,逗号分隔。

### 8. 接进插件

给 profile 装上可选依赖:

```bash
cd ~/.dsh/profiles/<你的 profile>
pnpm add @huggingface/transformers
```

然后在 `cordis.patch.yml` 里把模型目录指过去(绝对路径):

```yaml
- insert:
    - id: memory
      name: 'dsh-agent-memory'
      config:
        semanticModelDir: /absolute/path/to/work/model-onnx
        semanticWeight: 0.3
```

重启 DSH。插件会自己把记忆向量算一遍并缓存到 `$DSH_HOME/storages/memory-semantic`,第一次启动慢几十秒(几百条记忆),之后走缓存。

装没装上,看插件状态里的语义信息;或者看日志有没有 `no model at ...`。**装错了不会崩**——语义通道是「尽力而为」的:依赖没装、模型目录不存在、向量算一半失败,插件都记下原因然后退回纯词面打分。一个记忆插件因为可选加速器没配好就起不来,那比没有加速器还糟。

## 参数怎么调

`semanticWeight` 是融合权重,默认 0.3,公式是:

```
relevance = w × semantic + (1 − w) × lexical
```

往上调,语义的话事权更大。调之前先想清楚:语义单独是打不过词面的(0.7121 vs 0.7248),它的价值全在「和词面不一致的地方」。权重给太高,等于把词面的精确性稀释掉。本机实测 0.3 附近是好的,你换数据集要重测——`eval.py` 的融合那组会网格搜一遍权重,直接看它报的最佳值。

## 常见问题

**量化的时候报错说找不到 `model-inferred.onnx`**

onnxruntime 的量化会把中间文件写到临时目录,你的临时目录路径里如果有中文,它会找不到。把 `TMP`/`TEMP` 指到纯英文路径:

```powershell
$env:TMP = "C:\tmp"; $env:TEMP = "C:\tmp"
```

**导出时报 `BertModel.forward() got multiple values for argument 'use_cache'`**

transformers 5.x 改了 `BertModel` 的位置参数顺序,按位置传 `(input_ids, attention_mask)` 会撞上 `use_cache`。脚本里已经用一层薄包装 `EncoderOnly` 改成关键字调用绕过去了。你如果自己写导出代码,记得照做。

**训练完效果还不如词面**

先看评测怎么跑的。如果语义单独比词面低——那是正常的,看融合那一行。如果融合也低,挨个查:是不是用了全关键词风格的训练数据(前面说过会掉点)、batch 是不是太小、轮数够不够、训练用的记忆是不是太少(少于几十条基本别指望)。

**GPU 利用率只有 30%**

正常。这条流水线是显存受限,不是算力受限——batch 32 时显存先到顶,GPU 在等数据。想压榨只能加 batch,加不了就认了。

**池化方式能不能换?**

不能。向量是**均值池化 + L2 归一化**,训练时就这样,推理时必须一致。换成 CLS 不会报错,但效果会悄悄掉下去——这种不报错的错误最费时间,所以特意写在这儿。

**记忆库在别的地方 / 想换个工作目录**

用环境变量:

```powershell
$env:MEM_TRAIN_WORK = "D:\memtrain"
$env:MEM_TRAIN_MEMORY = "D:\backup\memory.json"
python training/prep.py
```

## 最后:什么情况下不用训

- 记忆库还没攒起来(几十条以内):先攒,训了也是白训。
- 记忆全是英文 / 非中文:基座模型挑得不对,换个英文的,流程一样。
- 你觉得现在的检索够用:那就是够用。语义通道带来的是若干个百分点,不是「能不能用」的分界线。
