# M3 / #73：独立23项val补全与完整跨尝试配对（冻结出口暂停）

**23项新验证全部完成，跨来源30/30评估与四成本视图齐；原any-unresolved出口触发暂停。**
完整覆盖不等于原failed attempt成功，不等于forecast改善或可推进。N3/N4/N5必要前置未解除，
未运行matched-Generic、可微rollout或三seed确认，未补N5反证、关闭issue或推main；最终goal未自行裁定完成。

## 1. 范围、授权与工程身份

执行依据是当前用户总体V2目标、决策0030的方向内实验/预算/普通决策/节点常设下放，及ADR0031的
独立补测来源契约；不重新申请逐轮许可，但原失败、身份校验和冻结科学出口不豁免。
节点长文 `docs/goals/n2a-m3-validation-complement.md` 在运行前写定：0训练更新、只补原确实缺失的
23项val、新排他输出、planned1800秒/宽松hard3600秒/清理预留10秒、任一失败停止且全额计费。
原 `outputs/r7_73_process_supervision/` 的协议/109文件/六训练/七val/失败终态保持只读。

新补测编排/聚合工程及driver执行的base commit为 `011ab4cdc3b4fa9f6671a22962a2cf3227998c66`；
实际forecast evaluator使用原归档 `code.zip`，其base commit为
`d6c98cf1c33eca5885772c473805af3ef0ad62ba`。原归档的working_tree_modified=true，故 evaluator身份
以原80-member源码字节/zip SHA256为准，不声称它等于干净旧commit或新011ab4c代码。
`R7 CPU CI` run **37036869969**、job110937121161 head_sha精确匹配，completed/success，九主步骤成功。
匿名一手 [run API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/37036869969) 与
[jobs API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/37036869969/jobs?per_page=100)
访问日期 **2026-10-02**；响应/核验保存于旁侧接受目录，回执SHA256
`12c4e33c1af986b20387779d38134ff6fe9b1989bb666e3e4e28a3abdf81e5cb`。
该CI只接受工程，不代替GPU实跑、科学效果或后续证据登记CI。
执行时已跟踪文件无未提交差异，但两无关untracked仍在：`.zcode/agents/web-researcher-backup.md`、
`.zcodeignore`。新code_status如实保存它们，working_tree_modified=true；不声称工作区完全干净，
二者未参与新源码闭包、未修改或提交。

- 最终定向CPU：**123 passed，0 skipped，13.04s**；包含驱动55与聚合68项，集合不累加到全量之外。
- 完整CPU：**2395 passed / 9 skipped / 2 warnings，233.74s**。六CUDA条件/三可选真实fixture跳过，
  不计通过；两Lightning无Trainer日志warning保留。没有新增依赖或弱化断言。
- 补测后定向登记合集 **221 passed，0 skipped，17.50s**。首次误用不存在的测试文件名，pytest未收集
  任何测试/exit4，日志保留；核实际文件名后重跑通过，不把未运行记PASS。
- 37阻断/campaign/两当前goal/index/brief/compile/staged与unstaged空白均通过；campaign四历史notes
  没有索引支撑，不冒称机器核数。R-009实测1210函数/3112断言，新增34/137；只同步基线与规模marker。
- 新五源码/两测试均在600/200硬上限内，最大文件543行、最大函数体87行；模型digest未变化。
- 独立审阅发现非法CPUreceipt被拒后finally仍写、评估祖先symlink写穿旧输出两项缺陷，已写前修复并
  用真实CPU CLI old/outside零写、合法身份失败留证、evaluator不可达symlink反证核验。限范围复核resolved，
  不当作许可。编辑中fixture尚未同步新anchor签名的33errors保留，最终123通过覆盖其修正。

## 2. 合法复用与真实数据前置

独立CPU审计实际核24历史checkpoint内部contract与六selected400、原code.zip/model/source/sidecar/固定
train inverse及旧七项指标/案例；109文件hash/stat/集合前后不变。成功技术资格回执SHA256
`020b0647434db6cf28acd9ac07116672e7ec34ddc7c49dd5dec2506e8f9718bc`，
readonly seal SHA256 `2311c584d0a29fbce46e5e490c8ee975101c3e940ddc1d558f83ef2dec5f204a`。
此资格不把原完整实验接受为成功。首次自写审计把driver双时钟采样误要求≤1µs失败，另留失败回执
SHA256 `5fed3f3365fe19f45039791dabb0db699c8cca549598fb3070b76280b41f659d`；
修核验器按实际双采样时序，不改原产物/预算。

当前真实source的只读 `prepare_r7_local.py` preflight已在GPU前运行：17通道、240时次、65×65、train186
/val22；报告SHA256 `ef4ed74883b5b45694f5a7f7de92cbe47ad42e226b9b3c6546c3ae684478f1ac`。
没有下载、--write、重建store或发布新数据。本轮仅消费既有完成cache。历史M2独立conversion前报告与
单次raw用户授权回执在有限范围未定位，不能倒填或由缺文件推无许可；事前版本化第二阶段授权在
`ecdae0d5bd0c70799f91d0ddfa02bcd04bf76e6f` 及ADR0010可核，此限制保持披露。

| 对象 | SHA256 / identity |
| --- | --- |
| source.nc | `496084a9260bacfaf6293a01d89439c1e49d6afa8f09bc1f51d89a1d1f9bda21` |
| train data identity | `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07` |
| model code digest | `11090929930da4e1259698699cbbf12b3738cdfb2f609c3af516c24399144476` |
| sidecar identity | `4fed1c78e4c4c09a41d95457649925b02d0c8ebf89aa47c2ac8dc6496734912d` |
| original canonical protocol | `404cf32b8ee8f6c3ff192d46c1de6765abe4ae3fa72967469af800a774fde15d` |
| original code.zip | `18595abce5acfa9e6f3252342f04eace470a06f48c3eea97c6ba463e3ea02d95` |
| complement canonical protocol | `74d05ab36a5c15451893e20f9cadcaecdac62170717af61a3e7524af57509dd5` |
| complement code.zip | `e44ef233b0455c38279549d22c1d864d604400933cc5a61210d3591df37abb43` |
| complement source-tree map | `98fd0b1ccdcd1712bbfa812e826828c434a76a608fd363114458ffd730c44337` |

## 3. 实际执行与逐文件来源

新运行目录：`outputs/r7_m3_validation_complement_20261002/`。
旁侧接受目录：`outputs/r7_m3_validation_complement_acceptance_20261002/`。
prepare/run各执行一次；不resume、retry、重训或更换seed/案例/selected checkpoint。

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B scripts/replay_r7_m3_validation.py \
  --mode prepare --original-output outputs/r7_73_process_supervision \
  --output outputs/r7_m3_validation_complement_20261002

CUDA_VISIBLE_DEVICES='' .venv/bin/python -B scripts/replay_r7_m3_validation.py \
  --mode run --output outputs/r7_m3_validation_complement_20261002
```

冻结新代码归档为81个本地源码闭包成员；原80个源码/config安全提取到新archived_code。实际worker用
`.venv/bin/python -I -B`、归档cwd和隔离sys.path，stdlib双socket禁网先于项目import；逐模块origin守卫
拒当前仓库fallback。原协议output仍指原目录，归档helper自身ROOT指提取源码，不伪造迁移协议。
两CPU owned前后置核原six-selected400/旧七val及输入身份；23 fresh CUDA进程直接调用归档evaluate_local
写新输出，不调用会写原目录的旧worker。source、BUILD_COMPLETE、signature/model digest、sidecar与固定
inverse均强核。无封存test清单/字段解码，代码/源文件身份只读不等于OS全程读取证明。

| 来源 | 数量 | 记录语义 |
| --- | ---: | --- |
| 原训练 | 6 | 各400更新/selected400，原receipt/checkpoint/report与成本不变 |
| 原成功val | 7 | 原119个RMSE格/131case，保留原四指标文件与峰值 |
| 新补测val | 23 | 只补缺项；新receipt、归档evaluator身份、独立allocator与连续计费 |
| 完整val | 30 | 510 RMSE格、528case；17变量×五lead×两seed×三臂 |

`source_manifest.json` 对36项逐文件记录receipt/timing/log/checkpoint/artifact路径和SHA256、采集时序、
原训练协议与独立聚合协议。两协议分别保存，不将新23项说成原attempt的36成功receipt。
lead案例数保持22/21/19/15/11；同seed同lead同案例配对，无跨seed互配。

## 4. 完整指标与冻结配对出口

全量数据在 `rmse_table.csv`（510行，物理RMSE、climatology RMSE与MSE skill）、`acc_table.csv`（510行）、
`initialization_table.csv`（528行）、`paired_comparison.json`（255 aggregate rows、三pair各85格）。
17变量为t2m/u10/v10/mslp/z850/t850/q850/u850/v850/z500/t500/q500/u500/v500/z250/u250/v250；
全部6/12/24/48/72h和坏变量保留，不跨物理单位平均。

M3本身的skill CSV已为物理单位，不盲目套用其他历史归一化缺陷再乘std；原CSV不改。
510 skill格中238正、272负、0未定义；177格同时正ACC且非正MSE skill，不能拿ACC正替代气候态技巧。
这些是逐格描述计数，不是多变量显著性或统一效果指标。

比较沿用原#60规则：delta=同seed focus RMSE减baseline RMSE；seed41/42严格都<0才improved、
都>0才worsened，其余unresolved，不用均值掩盖反号。每格显式seed delta、单位与案例身份。

| 配对 | improved | worsened | unresolved |
| --- | ---: | ---: | ---: |
| future_draft_aux − aux_off | 19 | 45 | 21 |
| input_aux − aux_off | 21 | 36 | 28 |
| future_draft_aux − input_aux | 3 | 44 | 38 |

逐lead全部计数保留于paired JSON；各pair总数85。T2M对aux_off两seed/五lead都恶化：

| 配对 / ΔT2M(K) seed41,seed42 | 6h | 12h | 24h | 48h | 72h |
| --- | --- | --- | --- | --- | --- |
| input_aux − aux_off | 0.4664,0.4319 | 0.3272,0.2222 | 0.2389,0.1290 | 0.8055,0.9281 | 0.9024,1.0765 |
| future_draft_aux − aux_off | 0.3827,0.4291 | 0.4358,0.3027 | 0.4248,0.3480 | 1.1574,1.4028 | 1.8477,1.6393 |

这是当前两seed小段的描述性负面，不能称辅助任务无用的普遍因果证明。完整多变量仍有改善、恶化和
unresolved，不能只挑T2M或改善格替换冻结全集合。原 `descriptive_outcome` 的any-unresolved暂停保持：
`status=paused`、`advance_next_node=false`、`scientific_gate_evaluated=false`。
attempt finalized/coveragecomplete=true、partial/budget_limited=false；**完整实验覆盖与科学暂停同时成立**。

归档比较器在隔离CPU进程对现有metadata重放，table/pairs/outcome逐字段精确一致；回执
`comparison_replay.json` SHA256 `9ef4d22ca91fe3b7d34f3ef6795a6017e09e8a83c809282af018999c928729a9`。
该零GPU metadata重放不是第二次forecast评估，不声称GPU逐位复现。

## 5. 四成本视图与最终整轮封印

| 成本视图 | 完整证据 | 来源与边界 |
| --- | --- | --- |
| 参数 | parameter_table.csv，3行，各2,968,259参数 | 引用原CPU实测，不是新计数 |
| FLOPs | flops_table.csv，3行；forward15,549,227,904 | 引用原enable_grad计量；forward+backward off46,530,544,896，aux46,530,594,048；不含计数器未覆盖elementwise/normalization，不是streamed完整成本 |
| 训练吞吐 | training_throughput_table.csv，6行 | 原400update实耗0.492455–0.722591s/update；本轮0训练，不填新训练速度 |
| allocator/时长 | allocator_table.csv，36行；evaluation_time_table.csv，30行；phase_time_table.csv，36行 | 六原训练/七原eval/23新eval逐行来源；本进程owned峰值，不是设备总量或孤立forward成本 |

新23进程pre-init/initialized/baseline allocated/reserved全0，不用私有clear放行；23不同PID，峰值allocated
39,590,400–42,861,568B，reserved46,137,344–71,303,168B，新evaluate_local内部3.0852–6.9460s。
连续GPU计费包含fresh启动/import、间隔与owned清理，不以成功worker小计打折。
23次GPU spawn各只读固定UUID，门槛2390MiB，最小free24107MiB，无选定卡外部PID观测；GPU0邻居不干预。
共驻政策不等于真实邻居负载下性能验收，读数不是显存预留。只操作直接Popen，25自有CPU/GPU进程正常退出。

| attempt / 成本 | 连续GPU秒 | GPU-h | 整轮秒 |
| --- | ---: | ---: | ---: |
| 原failed（保持） | 1790.1153369722888 | 0.49725426027008024 | 1805.1085775829852 |
| 新补测（完整但paused） | 644.3709621066228 | 0.17899193391850632 | 700.5363800507039 |
| 两次总成本（不是同一clock） | 2434.4862990789115 | 0.6762461941885866 | 2505.644957633689 |

新整轮从prepare最早CLI入口冻结同boot单调anchor开始，prepare4.9935s、prepare→run间隔、CPU前置
21.3537s/后置19.3587s、全部GPU启动与评估、聚合5.6461s及owned cleanup均计入；final whole700.5364<
planned1800<hard3600，soft_overrun_seconds=0。聚合前whole694.8902271008119s只是stage快照。
`artifact_manifest.json` 的publication-time final_round_sealed=false与cost_views的finalization_pending
保持原始事实，不回写为已包含未来artifact；最终旁侧seal引用实际attempt/execution digest并补最终成本。
原failed的whole超过旧1800秒5.1086秒仍保留，不以新时长修成旧通过。

最终旁侧 `final_round_seal.json` 已实际通过，status=accepted-engineering/final_round_sealed=true，SHA256
`0810665b861a3f2206b53556a6878b179e68f2923e560e11c7ef906af6797a72`；引用实际attempt/execution
与276项新文件、109项原文件SHA256，补final whole和两attempt总成本，不修改原stage文件。
`independent_final_verification.json` SHA256
`576a72b26efd4148ddc9a6ffc59347c8e3bddb0b979968f6f9fcd5167ca2ab79`，
scientific_paused=true、advance_next_node/goal_complete/scientific_success=false。

## 6. 关键产物digest

| 产物（新运行目录） | SHA256 |
| --- | --- |
| protocol.json文件 | `689abd9e028078734574576644b7e2b2b628b3efc05ad40d2dba2cfa5dfacb03` |
| attempt.json | `401b270fdfe378dfd19e4cc29e7cb9309e2349aaf689ef862ca7b05b5a380e5e` |
| execution_attempt.json | `7931af4f3535f0426d48435b14954af020e7b0281aa36cc038592db7e7a15b9c` |
| source_manifest.json | `dc2c275033819b886ef8dd716bc8325990b7de089ff502381f9c68e466318d89` |
| paired_comparison.json | `5e845badb28086a32ef016167b3efc48114fe303780fd3d13898b2fa035c2d68` |
| merged_result.json | `1f4d3e85dad457cd692d1286542653b40af24dffc8eb59d2e30e7ce0afe54c5c` |
| artifact_manifest.json（stage） | `57df22867570489b7541d6df7904da5b300a09aafc457c3ff2a83680ea42e84b` |
| cost_views.json（stage） | `a82680cd4f816c611d4d8a66787aff187f1f9a376103885de7cc8cab0d2acf47` |

## 7. 节点停止、issue现状与未做项

冻结不可分辨/any-unresolved出口触发，因此N3/N4/N5依赖BLOCKED，不以0030的常设推进权取消出口。
没有为推进而增seed、lambda扫描、扩训练或改端点，也没有把任务改成完整goal完成。
六issue匿名终态读仍全部open，访问2026-10-02，原响应hash与回执已在旁侧保存：
`final_issue_state.json` SHA256 `2ec437458613c0aa0014e410cc7b9cc08c4bb38848e8ce8c9767a3c0a5c0296a`。

| issue | 本次确认的范围 | 未满足依赖 / 本次关闭状态 |
| --- | --- | --- |
| #70 | V2主线的M3真实探索补齐，全部负面保留 | matchedGeneric/rollout/三seed与实际改进未成立；BLOCKED，保持open |
| #71 | 复用现有M1实现证据，无新工程claim | N5年末/闰年/UTC/经度/batch/padding/K同valid-time追加反证未做；BLOCKED，open |
| #72 | 复用现有RW-A/RW-B实现与负面证据 | N5现役latent反馈/process_reader非零梯度追加反证未做；BLOCKED，open |
| #73 | 尺度/三时刻工程及一轮三臂两seedval完整；T2M辅助对off描述性恶化 | 无三seed独立贡献评价；N5共同关闭前置未齐，不把覆盖齐等于科学成功；保持open |
| #74 | 本次没有新增实现或实验 | N2a冻结出口阻断N3；matchedGeneric/可微两步/同父两臂两seed未做；BLOCKED，open |
| #75 | adaptive默认未启动，没有新前沿claim | N3不可运行、N4三臂三seed未做；BLOCKED，open |

此表是本次停止与未做记录，不冒称已执行N5正式验收/关闭阶段。没有Closes提交、main推送、merge、force、
新数据发布/付费资源/独占或cron。关闭授权只在N5具名条件齐备时适用，本次没有满足。

## 8. 独立核验、登记与可复现限制

独立stdlib审计已真实执行，**67380项检查，4.3066s，exit0/accepted-engineering**；核新276/原109文件
集合与raw SHA前后不变、原failed/旧成本、原80归档提取/新81代码闭包与工程commit、source/sidecar/
train-val身份、两CPU+23新GPU/原13成功+旧失败进程退出、fresh基线/余量/连续计费/prepare时序、
30项四指标文件及per-case metadata、36来源/255aggregate/三pair85与全部13表逐行来源。
独立核算RMSE/skill与严格同seed符号，不import项目模型、torch或数组，不反序列化checkpoint；
检查数是机器guard次数，不是67380个独立科学证据。源码与运行日志旁侧归档，主脚本SHA256
`41f177f891a14a93819d004d874ee93ee42c18a750a5f070435edec3bcac06d1`，五帮手SHA在verification回执。
完整CSV/案例/成本/代码身份接受为可审计工程产物，原failed仍failed，科学暂停保持；这不是节点
推进许可、统计结论、OS全程网络/test/邻居信号证明或最终goal完成裁定。
可复现等级为code/data/protocol-pinned、原训练config-reproducible，加精确semantic comparator metadata
重放；不能声称GPU训练/新评估/峰值/时长未来逐位一致。两seed/单冬季区域/400更新不证明显著性、
收敛、泛化或过程语义因果；没有matchedGeneric，不能把相同forwardFLOPs当公平完整算力。
禁网guard防意外而非OS沙箱；独立只读校验持久化身份/进程/表/成本不是全syscall证明。
有限独立报告复核对原109 opaque hash、三来源/全部CSV计数、同seed delta、单位公式、stage/final成本、
精确工程CI及六issue开放状态确认一致；唯一报告身份措辞缺口已明确分开新driver与旧归档evaluator。
此报告复核不是新许可、重复最终seal或goal完成校验。

**历史版本化登记BLOCKED（2026-10-02）**：保存本页后，单独证据commit命令的PreToolUse被Mimosa L3强制拒绝，整条
包含复制/暂存/commit的命令未执行，HEAD保持011ab4c；随后仅把证据页保存到本地，不重试或换路提交。
闸门报告 `legacy_v531_full/` 中12high/1low，最高high，包括10处路径穿越/2处SQL注入；扫描覆盖不完整，
报告尚未在本会话独立复现，不作项目安全定论。归档只读，未改归档、安全hook配置或采用绕过通道。
旁侧 `registration_commit_blocked.json` 记录完整拒绝和evidence_commit/registration_commit/CI=null。
canonical evidence index/brief仍17条、新audit只作旁侧pending-registration草稿，不填伪造证据commit。
本地全额账本新0.1790/总4.8808/余19.1192（精确总4.880706349145538），新增一行暂无index支撑，
campaign应新增一条note，不冒称已提交/机器数值接受。解除安全闸门与只读归档的冲突属于用户边界决定。

截至上述阻塞时点，无模型接口/模型digest/依赖/原数据/凭据/安全配置变更；工程commit/push的
scanner_enobufs与high阻断均不构成完整安全结论，当时未请求或运行完整项目安全审计。新证据提交、
索引登记、最终登记CI尚未执行成功；不能用工程CI37036869969冒称覆盖本页/工作态改动。

**用户恢复登记（2026-10-03）**：用户先回复「我对其进行明确授权」，随后明确说明
「我已经关闭了Mimosa，可以继续进行」。只读核到用户配置的
`plugins.enabledPlugins.mimosa@zcode-plugins-official=false`，仓库本地配置另有同名关闭项的未暂存差异。
这些是用户关闭工具的实际状态，执行者没有编辑安全设置；本地`.zcode/config.json`与两个无关未跟踪
文件不纳入证据提交。此前官方资料检索没有取得Mimosa的finding裁定或新修复版本，支持工单未外发。
本次继续仅恢复D6证据冻结/正式index/账本登记与正常工作分支CI，不伪称L3扫描放行、官方豁免或
项目安全通过；若实际正常调用仍被拒绝则停止，不换通道。原12high/1low、覆盖不完整与历史拒绝
回执完整保留，不将关闭插件写成漏洞已修复。

恢复前再次只读核原109/新276运行文件集合与SHA256、366跟踪归档及43科学pins未变，六训练/30评估
的已有覆盖与0.6762461941885866GPU-h不重跑。恢复回执与验证另存在仓外独立目录
`/tmp/r7_m3_registration_resume_u8t215bd/`；本次完整CPU真实结果为2395passed/9skipped/2warnings、
230.96s，六CUDA条件与三可选真实fixture跳过不计通过，两Lightning无Trainer警告保留。
本会话此前M3定向420passed/53.87s、0skip和3399项只读产物核查通过，分别是工程反证与身份/表格
核对，不累加成科学样本、不替代完整安全审计；MetPy、GPU重放和完整真实17通道CPU端到端未执行。
正式证据commit/index和精确登记CI在活目标进度绑定，
不向本页递归追写其自身commit或未来CI。科学any-unresolved出口、N2a/paused与advance=false保持；
不进入N3/N4/N5、不关闭issue、不操作main。最终goal仍由用户/运行时独立裁定，执行者不自宣完成。
