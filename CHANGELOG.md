# CHANGELOG

dsh-memory 迭代记录。每轮:改进 → 测试 → 验证 → commit。

## v0.8.0 (2026-08-16) — 可解释召回 + 记忆面板 + 可视化编辑

- **可解释召回**:`memory_recall` 每条结果附 `reasons`(子串/标签/bigram/unigram/bm25/importance/recency/access 信号分解),可审计"为什么命中"
- **BM25 信号**:词频×长度归一化(零依赖,cap +2.0),提升中文长文本召回
- **token 优化**:recall 默认 limit 5→3、result 内容默认截断 400 字(可配 `content_max`),按需调用成本下降
- **memory_sediment 工具**:批量沉淀事实/决策/教训(≤3 条/次 + 进程级 5 分钟冷却,agent 自己即 LLM,零额外成本),防噪纪律
- **记忆面板(Web UI)**:设置侧边栏一级导航「记忆」——统计(总数/类型/过期)、关键词搜索、类型过滤、**新建/编辑/删除**(带确认),深色模式适配
- **RPC 双通道**:`/dsh-memory-read`(trusted-host 只读)+ `/dsh-memory-write`(loopback 写),面板经 RPC 直连 `MemoryCore`
- `MemoryCore.updateContent`(可视化编辑写回);peerDependencies 补 client 三件套 + react(第三方可独立安装)
- 测试 50 → 52 全绿;tsc 绿

## v0.6.0 (2026-08-14) — 系统瘦身优化(不重不漏)

架构审视后的删除与提效(功能零变化,内部更干净高效):
- **删 wrapWrite 恒等包装**(v0.2.1 指纹刷新已下沉到 onWritten 后,包装纯冗余,6 处调用点直连)
- **删 hotSummary 死方法**(v0.2 停用热记忆广播后无调用者,仅测试引用;hotRecords/hotnessScore 保留——空查询浏览仍用热度排序)
- **注入改只读**:pre-step 的 recall 加 `touch: false`——被动注入不再触发访问计数写盘,**每轮省一次全量 JSON 写盘 IO**
- 测试适配(hot 测试改用 hotRecords 断言),45/45 全绿,tsc 绿

## v0.5.0 (2026-08-14) — 防无关联想 + 省 token

用户反馈:"ok了吗"这类短查询注入一堆无关记忆(0.048 分),浪费 token。两层优化:
- **弱匹配剔除**:`MATCH_BASE_MIN = 1.0`——纯单字重合(unigram Jaccard ≤0.8)永远达不到阈值 → 分数归 0 不命中;子串(3)/标签(1.5)/双字强匹配(≥2)保留。实测"苹果"vs"水果摊"不再命中,"ok了吗"查任何记忆都是 0
- **查询质量门槛**:`hasMeaningfulQuery`——纯虚词/填充语查询(ok了吗/可以吗/嗯)直接跳过注入;有效中文 ≥2 字或英文词 ≥3 字母才检索
- 测试 40 → 45(弱匹配 3 + 查询质量 2)
- README 补"无无关联想"特性

## v0.4.0 (2026-08-14) — 自我纠错闭环(lesson 自动固化)

用户需求:agent 弄错事情多次 → 自动总结为教训入库 → 保证不再犯 → 教训动态注入防本对话再犯。
**设计(不重不漏)**:
- **自动检测**:监听 `agent/error`(官方错误事件),按错误指纹(code/message)计数——同指纹第 2 次触发 `agent.inject()` 提示模型固化教训
- **高质量固化**:提示让模型自己用 memory_remember 写 lesson(importance 3,附场景与规避)——模型总结比自动生成更懂语境
- **防复发闭环**:lesson 入库后由 pre-step 注入自动召回(已有机制,不重复造注入通道)
- **用户纠正**:记忆协议加规则(被纠正/同类错误两次 → 当场固化),覆盖"工具没报错但结果不对"的场景
- 不重:复用 remember 写、复用 pre-step 注入、复用 lesson kind;不新增工具/存储/注入通道
- 配置:lessonizeEnabled(true)/lessonizeAfter(2)
- 指纹函数 extractErrorFingerprint 可测;测试 37 → 40(指纹 3 例)
- README 双语补自我纠错闭环

## v0.3.0 (2026-08-14) — 注入重复抑制 + 仓库产品化

- **重复抑制**:pre-step 注入在检索结果 id 集与上一轮完全一致时跳过(连续同话题不刷屏,话题变化才注入);用户实测反馈"每句话都有相关记忆块"后加的优化
- **仓库产品化**:GitHub 只留产品资产(源码/测试/脚本/README/设计提案),内部文件(STATE/CHANGELOG/research 含记忆提取)全部本地化+.gitignore;推送清单改用 git ls-files(曾误推 research/zcode-memory-extract.jsonl 隐私记忆,被密钥扫描拦下);教训已写入记忆(lesson/imp3/user)
- 新记忆:推送 GitHub 别放内部文件/内部对话(lesson/imp3/user)
- README 双语补:memory_import/memory_reload 工具、外部修改防护、重复抑制
- 37/37 测试绿;tsc 绿

## v0.2.1 (2026-08-14) — 外部修改防护(防覆盖三件套)

踩坑(3 条内存覆盖 41 条)后的治本方案,杜绝"外部写文件 → 进程全量覆盖":
- **① 写前指纹校验**:每次写操作(remember/recall-touch/forget/import)前 stat memory.json,发现外部修改 → 拒绝写入并报 `MEMORY_EXTERNAL_MODIFIED`(绝不静默覆盖)
- **② memory_import 工具**:JSONL/JSON 导入走 domain 写链(不碰文件),导入后内存=文件,天然一致;真机验证 imported 2/2 ✓
- **③ memory_reload 工具**:close+reopen domain,把外部修改合并进内存,免重启
- 指纹刷新下沉到每次写盘(onWritten),修复批量导入第二次写被误判为外部修改的 bug(实测 alpha 进 beta 拒 → 修复后 2/2 全进)
- 测试 34 → 37(guard 调用/拒绝/只读豁免)
- 记忆库恢复 67 条(测试数据已清理)

## v0.2.0 (2026-08-14) — 热记忆广播 → 按需注入

用户反馈:多项目场景下"每轮广播全局最活跃 8 条"命中率低(闲鱼活跃但你在做教辅),token 浪费;新会话仍不知道旧项目存在。
**方案 C 落地**:取消 stateContext 热记忆广播,改为 `agent/pre-step` 动态注入——
每轮模型执行前,用用户当前消息做 recall,注入 top-N(默认 3)条相关记忆摘要。
- 说"继续教辅"→ 自动注入教辅决策/踩坑;说"闲鱼"→ 注入闲鱼方案
- token:按需付费(3 条×120 字≈0.4K/轮),条条相关,替代原来固定 1K/轮
- 真机验证:headless 会话中模型明确看到"相关记忆"注入块并能引用其内容 ✓
- 配置:injectEnabled / injectCount(默认 3)/ injectMaxChars(默认 120);移除 hotCount / stateContext
- 测试 31 → 34(新增 queryFromMessages / renderInjection)
- 文档:README 双语 / DESIGN 三层架构 / CHANGELOG 同步更新

## v0.1.0 (2026-08-14 基础版,已完成)

- 三层架构:底层逻辑(preset persona/AGENTS.md,已有)+ 热记忆(hotCount 工作集常驻上下文)+ 记忆库(domain 全量按需检索)
- memory domain v1(records 表,zod 4.4.3 校验,JSON 后端持久化到 $DSH_HOME/storages)
- 4 工具:memory_remember / memory_recall / memory_index / memory_forget
- 治理:容量淘汰(maxRecords)、近重复合并(Jaccard 0.7)、TTL 过期、importance=3 删除确认、访问计数
- 检索:零依赖(子串命中 + Jaccard,加权重要性/90 天半衰新鲜度/访问)
- 协议注入:工具 description + systemPrompt.section(order 110)+ 热记忆 context 快照(order 150)
- 21 例单元测试全绿;tsc 全绿;真机 headless 冒烟(写入/同进程检索/跨进程检索/UTF-8 持久化)全过
- 文档:DESIGN.md / README.md / PROPOSAL.md / docs/AGENTS.template.md / presets/README.md / LICENSE(MIT)

## 迭代计划(2026-08-14 下午,持续到 18:00)

- [x] R0: 代码审查 + 文档审查意见处理(评审反馈闭环)
  - Critical: domain 幂等 open(多 preset/多会话共存,修复 single-open 冲突)
  - H1: project 参数注入(remember/recall 支持项目隔离)
  - H2: 原子 update(并发安全,recall touch/merge 无覆盖竞态)
  - 热机制修正:新鲜度基准改为 lastActivity(recall 命中真的能让记忆变热)
  - 检索精度:中文 bigram 索引 + 单字宽松兜底(苹果 vs 水果摊 精确区分)
  - evict 保护 importance=3(仅全库 3 级时淘汰最老,不死锁)
  - merge 保留旧 TTL(显式 ttl_days 才刷新);ttl 上限 clamp 3650 天
  - forget 返回 skippedImportant;recall 返回 totalMatched+returned
  - 截断码点安全(emoji 不切碎代理对)
  - 测试 21 → 27 例
- [x] R1: 冒烟自动化(scripts/smoke.mjs 一键 headless 回归,PASS)+ 文档修订(依赖版本/insert 语法/热机制语义/卸载排查/术语表)
- [x] R2: 压力/容量基准(scripts/benchmark.mjs → research/benchmark.md):400 条写入 45ms、recall 3.1ms、index 0.11ms、容量上限与 3 级保护验证通过
- [x] R3: 迁移脚本(scripts/migrate.mjs:MEMORY.md → dsh-memory,标题关键词→kind、scope 提示、--apply 直写 store;自测+CLI 实测通过)
- [x] R4: 并发测试补强(test/concurrency.test.mjs 4 例:并发写入/recall touch 计数不丢/并发 merge 字段合并/并发 delete+recall;31/31 全绿)+ DESIGN 记录 merge 并发边界
- [x] R5: 检索质量评测(scripts/eval-retrieval.mjs → research/retrieval-eval.md):真实风格语料 10 条 + 9 查询,**top-1 命中 100%**
- [ ] R6: 文档最终打磨(humanizer 自查)+ 全量回归 + 收尾
