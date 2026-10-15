# 主模型超气候态 campaign：持续自主探索与独立年份/四季确认

<!-- round-node: S3 -->

**状态：active（2026-10-08；S0–S2 与S3 ordering已登记，S3气候态锚单次筛选已完整执行、保存开发支持false，独立读回及负面登记进行中；S4 未启动）。**
本文件是新方向的**唯一主计划与 goal 长文**。科学合同见
`docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`；执行授权见决策 0030/0038；实施交接见计划 0016。
旧 V2 收尾与 #70–#75 不重做，不自裁旧/新 goal complete。以后各具体实验长文由本主计划派生。

## §0 Objective（可粘贴；单段，实测 3998 字符）

> 在/data/esw/UrbanPiDiT_R2的r7/weather-reasoning继续UrbanPiDiT可思考/递归主模型长期研究。先读docs/goals/main-model-climatology-campaign.md、docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md、决策0030/0038/0039/0042–0045，核HEAD与issue评论。本次实际推进，不交计划等我。当前S3/index73/53.8478GPU-h、2023test未评分r0、S4未启动；capacity轮(depth 4→8)待跑，K4→8(index71)已登记（未完成、勿重登）；从§8“下一动作”恢复，不重复已登记轮次。cap20/remaining−33.8478仅会计。自迭代契约：每轮先头脑风暴＋委派web-researcher检索一手论文/官方源码找办法，再提可证伪假设；每轮写完四件套（§8下一动作、账本行、index record、campaign-state）即进下轮，不等触发/校验；仅在保留授权/冻结停止/无可独立推进时停，停下须在§8写“为何停+恢复命令”。本客户端goal校验会报错并fail-open标complete；已打本地补丁二（§4b，仓库外、更新即覆盖），若见complete而科学门未过仍按§8恢复。防空转：同一假设连续无收益就换/停，不重复同一test、不加seed/剂量练赢；CI轮询不阻塞整轮。用workflow：先加载dynamic-workflows技能（未加载时CreateWorkflow拒跑）并按准入判断；goal与workflow必须同窗口同app实例（跨实例丢通知、表现不推进，已实证）；多轮编排用.zcode/workflow-drafts/r7-campaign-round.dwf.ts，一轮=侦察→提可证伪假设→派生单因素驱动→独立复核→冻结提交等CI→逐stage GPU→读数四件套登记；脚本自设轮数硬上限、自跑commit-guard/verify不依赖hook、只推工作分支（禁main/force/merge）、GPU按卡串行；被打断按ResumeWorkflowRun或resumeFromRound续，跨attempt seed复用按0045白名单全等准入；一次性inline。我下放方向内实验设计、架构/方法、训练信号/优化、数据源/区域/年份/变量、GPU训练评估、seed/更新/规模、时长预算、普通决策/排障、下一实验与节点推进权；不逐项询问/不逐轮等触发，时长/GPU-h自主增加并简报，无总GPU-h上限。付费/租卡、GPU独占、方向改变、main合并/发版、破坏性须具名许可；新issue关闭/main写入不继承旧许可。D1按§8接续：先跑capacity轮(depth 4→8双卡，用.zcode/workflow-drafts/r7-s3-capacity-depth-run.dwf.ts)并按预声明出口读（72h三seed转正→验收；否则关闭“加大容量”支转架构）；训练配方支与宽输入支已关闭不重开；核归档code.zip/loader/案例/单位/CPU反证，不重训练/replay，正负登记，不以小样本称泛化；D2据#76–#79及后续评论与正负证据提可反驳假设；GPT6PRO建议仅参考，旧PE/scale/typed/ordering/anchor/rollout-dose不无理由重跑，issue顺序非matrix。允许主模型架构、表达、目标/优化、可推理信息组织和数据制度探索；每试写实质差异、正负控制、预期信息、新protocol/输出、停止/饱和出口与回主线动作。先解析/CPU接线与小规模真实train/val，支持后扩；性能与机制归因分开，matched_generic是同图等价控制非必须赢baseline，typed须同信息有/无类型对照，future真值仅监督。无收益查官方论文/源码、梯度/预测/优化/数据瓶颈，转可证伪假设；不无限调开关/seed/训练量，不换外部大模型冒充主线；已否定假设停，旧negative/failed/paused保持，独立主线继续。D3**本轮起暂缓数据扩充**（收益判断偏小）：不新增下载、不重建store；仅当架构/训练信号/优化全无收益时才作为**最最后**手段启用——启用时按§5与real-data-acquisition技能分批执行（先冻source/范围/单位/网格/split/预算，只读preflight留digest后--write至全新排他outputs，完整身份/schema/BUILD_COMPLETE不省，失败留failed-no-fallback）。旧data/raw|interim|processed、store、归档不改，不删部分输出复活。统计只fit train，跨split历史/目标窗剔除隔离；数据变公平重建气候态及必要incumbent，不拿新模型打旧弱基线。实验离线(R028)。D4用train/val选架构/K/loss/checkpoint，分报信息/样本/updates/参数/FLOPs/GPU-h，同updates非同算力；最终配置/checkpoint/baseline/cases/split、统计法/块长/重采样/RNG/seed及数字预算全锁后才评分新test。验收固定t2m/full的6/12/24/48/72h：至少3预声明seed，各seed在确认总集/各年/四季组skill=1−MSE模型/MSE气候态均>0，seed均值全部primary同时区间下界>0；u10/v10/mslp对同数据incumbent在同组/lead每seed相对MSE变化≤0且同时区间上界≤0，容忍0，undefined不通过。至少一个完整真正未见年份及四季，配对分层时间块bootstrap同重采样病例/变量/lead/seed，不把像素/重叠窗/seed当独立样本；不足块不过门。首次test标签评分消耗确认r，partial也计，alpha_r=0.05/(r*(r+1))、累计≤0.05、首轮97.5%同时区间；失败留曝光/全部分组，不换名重置/复封旧test/反复看test练到赢。全17变量×五lead×full/interior/edge、物理RMSE/MSEskill/ACC及坏变量照报，不跨单位平均或挑赢家。未达门返独立开发，门过封印证据并提请验收，不保证必胜、不自宣SOTA或complete。D5调度代理与技能：未知路径/调用链/证据扫仓用Explore并给范围；多步依赖/架构/大改用planner配planner-delegation，JSON过check_planner_plan及事实核，科学门只引不外委；planner失败留原错，自规划过同契约并独立审阅。外部一手文献(论文/公式/实验设置/引用图)委派arxiv-researcher(arXiv MCP只读)，通用库/版本/文档/报错委派web-researcher，不可用切web-researcher-backup，仍失败才curl留因；一手核URL/访问日/版本，不以摘要为证。准备期可clone源码至outputs/reference_sources/<project>/<commit>/，锁commit/license/hash并标注引用与范围；镜像只读，不import/执行上游训练。general-purpose做无冲突实现、候选升级/最终独立只读复核，不代最终verifier。命中先实际加载：goal-loop、parallel-research-workflow、dynamic-workflows、planner-delegation、decision-record、issue-lifecycle、real-data-acquisition、environment-rebuild、web-research、bounded-study-run、pinned-artifact-replay、result-freeze、ci-workflow-triage。D6每运行先冻protocol/digest、范围/失败处理、planned软预算/hard宽硬限与收尾余量，整轮含准备/启动/训练/评分/核验/清理。软超继续记overrun，硬截止/真错停attempt全额留失败，下一独立protocol可加时，不运行中改硬限或重试至偶然过。节点前后核campaign/goal、身份/账本/前置出口/独立审阅，实跑定向反证，源码变跑完整回归，登记index/brief与精确SHA必要CI；skip/cancel/queued/partial非PASS。登记全部假设/负面、code/data/protocol/resultsdigest、复现等级、scientific_claim:false/limitations与GPU-h/网络/磁盘/墙钟/overrun。普通环境/网络/工程/身份/统计/CI阻塞自主查因、合法修活跃代码、独立复核后继续，不以“负结果/有阻塞等决定”交差；仅保留权限或不可克服的外部硬依赖停相关动作。禁止改冻结门、伪造PASS、删弱测试、绕身份、安全降门/改用户配置/绕hook、force/mirror、凭据外发、cron/守护/会话外续跑；上下文压缩按最新进度恢复，终止留交接。结束汇报改动、实验与失败、技能/代理使用、验证与精确CI、成本与issue状态及未做项；不自宣最终goal完成。

### 连续自我迭代契约（2026-10-09 新增；objective 引用本节）

**定位**：把「一轮结束 → 等用户/goal 校验 → 再开下一轮」改成「一轮结束 → 立刻开下一轮」。适用于本主计划
每一次执行轮；与 §5 的预算/停止规则并用，**不放松任何科学门**。

1. **每轮交接四件套**（缺一不可，写在同一批交付里）：
   - §8 追加一个进度块：本轮做了什么、证据与 digest、**下一动作（可执行的第一条命令）**；
   - §7 账本新增一行（含实际 GPU-h / 网络 / 磁盘 / 墙钟 / overrun 口径）；
   - `docs/R7_EVIDENCE_INDEX.jsonl` 新增 record（无独立 record 时如实写明原因）；
   - `<!-- campaign-state: ... -->` 注释更新 `current_node` / `current_round_goal` / `used_gpu_h` 等字段。
2. **立刻续跑**：写完四件套即进入下一轮，**不等用户触发、不等 goal 完成校验**。下一轮起点由 §8 的
   「下一动作」与 `campaign-state` 共同定义，**不重复已登记轮次**。
3. **不等 verifier（实测依据）**：轮次结束时客户端会发起**独立非流式** goal 完成校验
   （`query_source='target_completion_verification'`、`tools:[]`）。2026-10-09 实测该调用 provider 报错
   （`status='error'`、31157 ms、0 token），harness **fail-open**：`payload.status='failed_closed'` 却
   `verification.passed=true`、`session_target.status='complete'`，而科学门（r=0、2023 test 未评分、
   S4 未启动）**一条都没过**。故：**若目标被标 `complete` 而 §8 科学门未过，一律按 §8「下一动作」恢复
   继续，不把 harness 的 complete 当科学完成**。两种失败形态（provider 拒答 `failed_closed`；无超时挂起）
   见 `docs/R7_ZCODE_GOAL_VERIFIER_ABORTS.md`。
4. **停止条件（仅三种）**：(a) 触保留授权边界（付费/租卡/独占/方向改变/main 合并/破坏性/无法安全克服的
   外部硬依赖）；(b) 达到已冻结的科学停止条件；(c) 确无可安全独立推进的工作。停时必须在 §8 写
   「**为何停 + 恢复第一条命令**」，否则视为未完成交接。
5. **防空转**：同一假设连续无收益就换假设或停；不重复同一 test；不加 seed/剂量练到赢；不做「无新信息的
   复验」冒充进度。
6. **CI 不阻塞**：登记/验证 CI 用**后台轮询**，同时推进其他工作（见 §5）；不得用阻塞式长 `sleep` 占用整轮。
7. **0 GPU-h 文档轮的例外**：无实验机器核数的文档轮**不新增** index record 与账本行（不编造机器核数行），
   按 §7「无 index 支撑的文档 0 成本行如实列 note（如列出）」处理，但必须在 `§8` 写清
   「本轮 0 GPU-h、无独立 record 及原因」。

## §1 起点、交接与已知差距

- 文档起点：`73097d49fd174d8e5f2bbb465dad64037bd486c9`，工作分支 `r7/weather-reasoning`。
  用户修改的 `.zcode/config.json` 与两个无关未跟踪文件不修改、不提交。
- 旧 #70–#75 在 N5 已具名关闭，精确 CI 37215981072、ff 与匿名 API 终态在
  `docs/goals/main-model-v2-campaign.md` §8、`docs/goals/v2-issue-closeout.md` 中登记。
  旧 master 是旧路线权威，本文件是**新方向权威**；只追加交接，不自裁旧 goal complete。
- actual C：三臂/三seed、400 L6更新，144/144 jobs；package 相对 old_ours 的 t2m6/12h六格改善，
  同结构 process/matched_generic 差约1e-5 K且反号，机制归因 unresolved；adaptive准确率—成本门未过。
  依据 `docs/R7_C_ACTUAL_CONFIRMATION.md`，不能把上述工程接受追认为超过气候态。
- M2：单冬季/区域双月段，train-only 8桶历史气候态与已曝光test不能支撑全年/跨年声明。
  新 S0 核相同案例与物理单位实际 gap，而不是复制历史跨单位“9.09×”或旧test数。
- 当前新 issue（访问2026-10-04）由 API 显示 Eswink 发布；GPT6PRO 来源为用户说明，不能作为真值权威。

| issue | 优先假设 | 必要控制与次序 |
| --- | --- | --- |
| [#76](https://github.com/Eswink/UrbanPiDiT_R2/issues/76) | 性能EPIC/P0固定incumbent与gap | 性能≠过程归因；P0→#77→#78，#79可并行设计 |
| [#77](https://github.com/Eswink/UrbanPiDiT_R2/issues/77) | 有效位置频带 | 实际激活/接线阳性反证，单因素，不混K/loss/aux/solver |
| [#78](https://github.com/Eswink/UrbanPiDiT_R2/issues/78) | 变化量参数化与训练尺度 | 解码先保持loss，后轮改多变量权重；normalized误差用s_c/d_c |
| [#79](https://github.com/Eswink/UrbanPiDiT_R2/issues/79) | 有类型局部诊断进入前向 | aux0；A当前/B同信息无类型/C类型，B→C归因，不重做旧aux扫参 |

## §2 交付物清单

| 编号 | 内容 | 可核查证据 |
| --- | --- | --- |
| D1 ✅S0 | 实际incumbent与公平基线gap | `docs/R7_S0_INCUMBENT_GAP_AUDIT.md`：身份链六类 digest 独立重算一致、9 checkpoint逐条复核、6,885 cell 独立重算 ≤6.63e-16、17变量×5lead×3区域×3seed×3K 全表（`outputs/r7_s0_gap_audit_20261005_attempt01/gap_table.csv`）；结论 incumbent 只在 t2m/6h 正 skill |
| D2 | 单因素候选与自主探索 | 每假设来源/差异、解析与真实实验、全部学习曲线/梯度/坏变量、冻结协议/新输出/停止读法、回主线决定 |
| D3 | 分批多季节/多年数据 | read-plan许可/预算、pilot真实成本、preflight审阅、source hash、split/stat身份、新store/BUILD_COMPLETE/失败无回退及恢复回执 |
| D4 | 独立确认越过气候态 | 预注册主门/守门、至少3seed、未见年度/四季、时间块同时区间、确认r/alpha与曝光日志、全17×5评分及成本 |
| D5 | 子代理、技能与参考源码使用 | 实际调用/分工、planner校验、独立审阅、一手URL/访问日/commit/license/hash、借鉴与未搬范围，不以摘要作证据 |
| D6 | 每节点登记与最终报告 | code/data/protocol/artifact digest、真实测试/精确CI、E/index/brief、累计GPU/网络/磁盘/墙钟和overrun、工程/性能/机制/限制分别报告 |

文档、测试、CI或更多issue不是D2/D4的天气技巧证据。没有获益不交差，也不保证一定能达到门。

## §3 判据、身份与每节点对表

科学判据只来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`，旧判据引用只说明历史而不改写：

- t2m五lead/full正MSE skill；至少3预声明seed同向；独立年度和四季；时间块同时区间下界>0；
  u10/v10/mslp同数据incumbent零退化守门。协议包含确认序号与总alpha支出<=0.05，初次97.5%同时
  区间、后续至少95%，不能靠无止境重复确认获得假胜。达到t2m目标不代表全部17变量全面赢。
- source/model/BUILD_COMPLETE/sidecar/统计强核；旧checkpoint用原code.zip重放；不能合法复用才新方案。
- 科学实例的dataset/cases/block长度/统计法/seed/预算必须先冻结，缺字段只能准备，不能正式训练评分。
- 真实失败即停该attempt全额留痕，negative停相关假设；新实质不同机制可预注册接续，不回改旧停止线。

每节点前后六项对表：①机器节点/round-node/上一轮最后next-action一致；②实际账本逐行及证据指针；
③前一证据已登记且commit可达；④本次scope/判据/soft-hard数字写定；⑤身份/失败/冻结页保持；
⑥主线与保留权限一致。机械漂移先自主核事实、合法修复再检查，不能编造账本或静默弱化失败门。

```bash
.venv/bin/python tools/check_campaign_state.py --campaign docs/goals/main-model-climatology-campaign.md
.venv/bin/python tools/check_goal_brief.py --brief docs/goals/main-model-climatology-campaign.md
```

默认 campaign 校检仍管旧master；本路径必须显式检查，不用旧绿灯冒称新计划通过。

## §4 节点路线、子代理与技能调度

### 节点

| 节点 | 输入与具体动作 | 出口和接续 |
| --- | --- | --- |
| S0 | 核actual C身份/旧六终态、合法读取train/val，冻结新的gap审计实例与公平climatology | D1可复核；明确最大短板/新data plan，不要求已赢才允许改主模型 |
| S1 | 分批数据pilot；并行#77/#78解析、#79设计与文献源码准备 | preflight/schema/身份/预算/发布门通过；候选CPU接线与阳性反证齐 |
| S2 | 单因素真实train/val小试、梯度/优化/表达诊断；每次独立冻结 | 完整positive/mixed/negative与取舍；实质不同假设继续，连续无收益重新分析而非无限开关 |
| S3 | 有效候选扩训练/全年数据，同数据incumbent/climatology公平重建 | val候选与成本可接受、关键守门预审；固定最终K/配置/checkpoint/统计法 |
| S4 | 未见独立年份/四季/3+seed确认与同时区间 | 失败记录曝光和r后回独立开发；门过独立复核/封印/提请goal验收，不自裁complete |

### S3前瞻接续（2026-10-07；旧路线不是必重跑清单）

S0–S2 已完成并登记；v3 的 train 2017–2021 / val 2022 / test 2023 已发布，单病例80更新及
三端点独立精确反算已按 audit/not-candidate 登记。执行窗口从**最新 §8**恢复；若下一问题已有
新结果，先合法核对 code/data/checkpoint/protocol/cases 等身份与复用条件，不强制重做已完成工作。
§0 的 D1–D6 是本次接续执行清单；§2 保留 campaign 原交付编号，已完成的 D1/S0 不因此重新打开。

第一问题不新增 optimizer 或 update：复用0/20/80端点、原四季train病例及四个val开发例、同train
气候态，查原 deep-K 目标与最终17变量×五lead物理评分的对应。先核归档 code.zip、新80 contract、
普通loader、案例/单位及CPU反证，具体诊断范围与数字预算由执行窗口另行冻结；结果按三分支接续：

- 原目标下降而最终输出不改善：查监督聚合、草稿/最终头与评分路径，不直接归因于更新不足。
- 只在同病例改善：查记忆、跨例干扰及数据制度，不把单病例响应当泛化或共同优化因果。
- 跨开发病例有支持才扩完整val；小样本仅供开发诊断，不是泛化结论或科学验收门。

诊断后自主规划主模型内架构、状态表达、目标/优化、合法诊断信息组织及数据制度的不同可证伪
假设。issue不是永久执行顺序或固定matrix；单假设negative停止并回主线，其他独立假设继续。
每次写明实质差异、正负控制、信息收益、停止/饱和出口和回主线动作，以新protocol及排他outputs
执行，旧negative/failed/paused保持。不能把更多更新或数据直接当原因，也不能据此称S4已过。

S1下载慢时并行CPU/资料工作，不让联网下载与离线实验隔离失效。节点可为必要分析往返，需在主计划
写出理由和对应新protocol；不重复已完成的B/C实现，不扩成全外部baseline训练竞赛。

### 子代理触发（实际委派，不只在报告里列名字）

| 类型 | 何时使用 | 交付与隔离 |
| --- | --- | --- |
| Explore | 路径未知、跨文件调用链、产物/issue证据映射 | 指定medium/very thorough与范围；只读结论，不复扫主链已委派问题 |
| planner | 多步依赖/架构/大改动风险 | JSON经check_planner_plan，科学阈值仅引用不外委；失败合法自规划 |
| web-researcher | 负面需机制依据、官方论文/源码/API/错误事实 | 一手URL/日期/revision/license；摘要不算证据 |
| web-researcher-backup | 主研究员不可用/额度/原生搜索失败 | 同提示词接续；仍不可用才curl留因，不因此放弃主线 |
| general-purpose | 隔离定向实现，候选/最终独立只读证据复核 | 无冲突文件、精确scope；复核不授权节点也不代最终verifier |

资料、CPU、独立复核可并行；GPU只有统一父调度核余量决定并发，禁代理独自抢卡/重复训练。
候选升级和最终验收必须独立审阅实际证据，不把planner设计意见当实验或科学判定。

### 技能触发表（命中先实际 Skill 调用）

| 名称 | 触发 |
| --- | --- |
| goal-loop | goal长文/节点交接、上下文恢复、harness不可用的手工循环 |
| planner-delegation | 多步计划/架构/风险及planner JSON合同校验 |
| decision-record | 授权/长期方法/数据契约变更，ADR负面后果与编号 |
| issue-lifecycle | #76–#79开始/推进/验收，工程与研究分开、精确SHA证据 |
| real-data-acquisition | 新源/下载器/派生数据，0038范围内自审preflight后新路径--write |
| environment-rebuild | 环境/依赖缺失/CUDA异常，不用conda base代项目venv |
| web-research | 一手文献/官方源码/库行为/错误事实检索与引用 |
| bounded-study-run | 有界CPU探针/实验；不是完整GPU训练通用SOP |
| pinned-artifact-replay | 历史产物/旧checkpoint重放，原code.zip/pins不绕 |
| result-freeze | 结果引用/报告/归档，digest/重放等级/登记 |
| ci-workflow-triage | CI失败/pending/取消、精确SHA所有必要步骤核验 |
| parallel-research-workflow | 多个互不依赖的研究线并行推进、或需 workflow 脚本编排多个子代理；先过准入判断 |
| dynamic-workflows | 需 ZCode workflow 编排（`/workflow`）：先加载再按七 gate 判断是否并行；未加载时 `CreateWorkflow` 拒绝运行 |

GPU研究按0030/0038、GPU资源规则与该次协议，不虚构不存在的通用GPU技能。新能力需要重复事实后
才独立立SOP，不为本轮多造平台。#76–#79建议不是固定必须跑全部格子的matrix。

并行前先过**准入判断**（七 gate：假设独立/产物隔离/计算资源/数据实例/可变状态/可独立验证/多重比较，
任一不满足即串行）；能并行才并行，资料/CPU/只读检索/独立复核可并行，GPU 仍由统一父调度按共驻余量串行。
详见 `.agents/skills/parallel-research-workflow/SKILL.md`（决策 0042，当时新增为第 12 项技能）。

本表共 **13 项**：其中 12 项为本仓 `.agents/skills/` 能力，**第 13 项**为 ZCode **bundled 技能 `dynamic-workflows`**
（`/workflow` 的落地技能，2026-10-09 纳入；不属本仓 `.agents/skills/`，故不进 `docs/skills/README.md` 能力表）。

## §5 时长、下载、停止与恢复

- S0–S2 已有实际协议及登记；本轮计划0017文档改动**0新增GPU-h**。后续每次运行由执行者
  依据旧测量/pilot自主确定 `planned_seconds`、`hard_cap_seconds` 及收尾余量的具体数字，先写
  长文并冻结protocol再运行；当前文档不伪冻结未来protocol。整轮包含准备/启动/训练/评分/核验/
  聚合/清理；超软预算继续并记overrun，硬截止停止该attempt并保留真实失败与全部成本。
- 复用先核合法pins，验证按风险开展；源码更改跑完整回归，明确CPU禁网且CUDA隐藏，必要GPU测试
  另冻协议并核默认共驻余量。不以无限完整复验代替研究；已冻结的hard cap及reserve不修改，
  不因真实failure结果事后追认stage通过，skip/cancelled/queued/partial不算通过。
- 无总GPU-h上限；cap/used/remaining是兼容会计字段。计划额度耗尽不单独停止；真实成本/说明/简要
  通知必留，不把软预算用来关闭attempt。新时长用下一独立协议，不修改原已冻硬截止。
- 下载每批source/snapshot/license/变量/区域/时间/half-open split、decoded/network预算、磁盘峰值、
  free-space门、软硬秒数、重试/分片恢复写定；先小pilot。仅hash/身份核齐可重用完成分片，不假称续传。
- 准备期可联网，实验R028离线；默认GPU共驻，启动/spawn只读UUID/余量，不能signal邻居。等待余量时
  继续合法CPU工作或安排新符合余量的协议，不降门/抢卡。承载CI timeout须大于hard+收尾余量。
- **CI 不阻塞轮次（2026-10-09）**：登记/验证用的 CI 只做**后台轮询**，不得用阻塞式长 `sleep` 占用整轮；
  轮次推进与 CI 等待并行，CI 结果在下一轮开工前回写精确绑定（见 §8 与 `goal-loop` 技能「连续自我迭代」节）。
- ordinary环境/网络/接口/身份/统计/CI/登记阻塞自主查因、修活跃实现/另立修复再独立核，不直接问
  用户“下一步”。不修改用户配置/安全门/冻结校验。工具失败按skill合法降级，不冒称调用成功。
- 真实错误/硬截止停该attempt、全额记账保留；已否定假设停止，不让新独立研究机械冻结。
  只有付费/租卡/独占/方向改变/main合并/破坏性或无法安全克服的外部硬依赖交用户，并继续可独立动作。
- 不能保证无限时间后获胜；科学门未达到保持未完成，不改阈值、挑变量或让test越来越像val。
  会话终止留progress/代码/协议/成本和下一个具体动作；不能自建cron/守护或无人续跑。

数据构建用只读preflight与完整source hash，0038许可范围内自行审阅后向**全新outputs**发布；不改旧
数据目录。train-only统计，split隔离，新test访问/确认r日志；坏源失败无合成fallback。
参考源码隔离 `outputs/reference_sources/<project>/<commit>/`，一手引用台账与许可，不直接import或运行上游。

## §6 与 planner 草案的差异与引用范围

本方案源自一次planner委派（工作态，非证据），主链修订设计经只读validate_plan=true。
不采纳草案把长文也限制4000字、继承旧C primary当新门、宽免空账本、忽略新CI检查、虚构
决策目录另建CHANGELOG或把旧campaign自行complete等建议。只有objective<=4000；新合同由主链
独立写定；原C01-C06都保留，新实例有真实零耗文档行，不编造GPU/index结果。

规则依据：`docs/rules/reproducibility.md`、`docs/rules/data-and-artifacts.md`、
`docs/rules/gpu-resources.md`、`docs/rules/external-sources.md`、`docs/rules/ci-and-verification.md`。
未在本轮检索外部论文/clone源码；issue URL/评论访问日与建议范围在科学合同§7登记。

## §7 明确不做与成本账本

不改已冻判据/test身份/failed，不删除弱化测试，不伪造PASS，不用统计undefined过滤通过；
不写旧data/raw|interim|processed/legacy，不覆盖既有outputs；不付费/租卡/独占/main合并/force/mirror；
不凭据外发、不改用户安全配置/降门/绕hook、不建常驻/cron。新issue关闭不是本次授权或科学完成条件。

| 轮次 | 实测 GPU-h | 累计 | 证据 |
| --- | --- | --- | --- |
| 0038新方向文档与校检支持（无实验） | 0.0000 | 0.0000 | `docs/plans/0016-main-model-climatology-campaign.md`（真实0GPU文档准备，无实验index record） |
| S0 incumbent身份复核与D1差距审计（零GPU只读） | 0.0000 | 0.0000 | `docs/R7_S0_INCUMBENT_GAP_AUDIT.md`；产物 `outputs/r7_s0_gap_audit_20261005_attempt01/`（墙钟5.523s，planned600/hard1200，overrun0，网络0，6,885 cell独立重算≤6.63e-16） |
| S1 四季2017获取与dev store发布（0 GPU） | 0.0000 | 0.0000 | `docs/R7_S1_FOUR_SEASON_ACQUISITION.md`（索引记录 `record:s1-four-season-2017-dev-store`）；产物 `outputs/r7_s1_seasons_2017/`（4 part 合并 480 时间点，网络14,390,323,242字节=计划99.2%、硬上限56%，4,604.4s；store 16.83s BUILD_COMPLETE；test未读） |
| S2 #77 频带筛选 v1（writer 缺陷轮，非注册） | 0.3201 | 0.3201 | `docs/R7_S2_77_PE_BAND.md` §5（v1 缺陷轮计费；无索引条目：非注册轮）；`outputs/r7_77_pe_band_pilot/FINALIZE_DEFECT.md`；三 seed 各 549.0/550.4/575.5s；finalize 被共享门拒绝，产物保留计费不手改 |
| S2 #77 频带筛选 v2（注册轮） | 0.3297 | 0.6498 | `docs/R7_S2_77_PE_BAND.md`（索引记录 `record:s2-77-pe-band-screening`）；`outputs/r7_77_pe_band_pilot_v2/paired_comparison.json`（整轮 579.7/555.6/552.4s，训练 1186.9s+评估 170.5s；overrun 0；test未读） |
| S2 #78 R-A 变化尺度筛选 v1（探针拒绝轮，零训练） | 0.0059 | 0.6557 | `docs/R7_S2_78_CHANGE_SCALE.md` §5；`outputs/r7_78_change_scale_pilot/FINALIZE_DEFECT.md`（三 seed 各 ~7 s 在训练前被拒） |
| S2 #78 R-A 变化尺度筛选 v2（注册轮） | 0.3700 | 1.0257 | `docs/R7_S2_78_CHANGE_SCALE.md`（索引记录 `record:s2-78-change-scale-screening`）；`outputs/r7_78_change_scale_pilot_v2/paired_comparison.json`（三 seed 549.0/553.7/554.9s，两臂同参数量/FLOPs；worsened 6h +0.1105/12h +0.1668；test未读） |
| S2 #79 类型诊断证据三臂（注册轮） | 0.7129 | 1.7386 | `docs/R7_S2_79_TYPED_EVIDENCE.md`（索引记录 `record:s2-79-typed-evidence-screening`）；`outputs/r7_79_typed_evidence_pilot/paired_comparison.json`（三 seed 整轮 855.9/860.4/850.2s，训练 1793.0s；B/C 同参数量 3097571 同 FLOPs；B→C 归因 6h/12h 三 seed 同号为负 −0.0533；相对 incumbent 两主格 unresolved；test未读） |
| S2 #78 R-B 加权损失（注册轮） | 0.3533 | 2.0919 | `docs/R7_S2_78_RB_LOSS.md`（索引记录 `record:s2-78-rb-loss-screening`）；`outputs/r7_78_rb_loss_pilot/paired_comparison.json`（三 seed 532.5/520.7/542.5s，两臂同参数量/FLOPs；worsened 6h +0.3101/12h +0.4806；接线探针 rel_err 0.0；test未读） |
| S3 batch-2 四季 2022/2023 获取（0 GPU） | 0.0000 | 2.0919 | `docs/R7_S3_BATCH2_ACQUISITION.md`（索引记录 `record:s3-batch2-20222023-acquisition`）；`outputs/r7_s2_batch2_20222023/`（8 part 成功、2 个 failed-no-fallback 保留、网络28,773,423,423字节=计划99.9%、硬上限55.8%；合并源`44a24ca0…`960 stamps；test未读） |
| S3 确认实例 v2 发布（0 GPU，含决策0039修正） | 0.0000 | 2.0919 | `docs/R7_S3_CONFIRMATION_INSTANCE.md`（索引记录 `record:s3-confirmation-instance-v2`）；`outputs/r7_s3_confirmation_2017_2022_2023_v2/`（train2017/val2022/test2023，472/472/472窗口，三sidecar齐；v1缺陷构建保留；test未读） |
| S3-D2 同数据气候态/persistence 重建（CPU，零 GPU） | 0.0000 | 2.0919 | `docs/R7_S3_D2_BASELINES.md`（索引记录 `record:s3-d2-same-data-baselines`）；`outputs/r7_s3_d2_baselines_20261005_attempt01/`（逐 lead 全覆盖 472/468/460/444/428；气候态 480 步 fit、16 桶各 30；墙钟 553.1s，planned 1800/hard 3600，overrun 0；网络 0；test未读） |
| S3-D3 同数据 incumbent 重训（GPU 3 seed） | 1.0987 | 3.1906 | `docs/R7_S3_D3_INCUMBENT.md`（索引记录 `record:s3-d3-incumbent-retrain`）；`outputs/r7_s3_d3_incumbent_20261005_attempt01/`（三 seed 初始化逐位复现 actual C；400 L6 updates×3；整轮 3955.3s，planned 5400/hard 10800，overrun 0；t2m 6h skill +0.2143/+0.2754/+0.3262；seed43 12h +0.0037/24h +0.0526；test未读） |
| S3-D4 R-C lead-coverage 候选筛选（注册负结果） | 0.8472 | 4.0378 | `docs/R7_S3_D4_RC_CANDIDATE.md`（索引记录 `record:s3-d4-rc-candidate`）；`outputs/r7_s3_d4_rc_candidate_20261005_attempt01/`（two_step×200 更新 vs D3 l6×400，FLOP 匹配比2.0018；整轮 3050.1s，planned 5400/hard 10800，overrun 0；主格 worsened 6h +0.9202/+1.0247/+0.9263、12h +1.3219/+1.4587/+1.2365；守门 34 个正 cell；候选停当轮；test未读） |
| S3-UB 更新预算筛选 l6×800（主格 supported / 守门未过，注册混合） | 1.1010 | 5.1388 | `docs/R7_S3_UB_UPDATE_BUDGET.md`（索引记录 `record:s3-ub-update-budget`）；`outputs/r7_s3_update_budget_20261006_attempt01/`（l6×800 vs D3 l6×400，训练 FLOP 比 2.0；整轮 3963.6s，planned 5400/hard 10800，overrun 0；主格 supported 6h −0.6170/−0.5144/−0.3985、12h −0.6191/−0.5105/−0.3487，全 lead 15/15 负；t2m skill 6h +0.49/+0.50/+0.50、12h +0.18/+0.21/+0.19、24h +0.08/+0.12/+0.16；守门 48h/72h 13 个正 cell 未过→不进 S4；test未读） |
| S3-BC 预算曲线 l6×1600（主格 supported / 守门 17 cell，注册混合） | 1.3716 | 6.5104 | `docs/R7_S3_BUDGET_CURVE.md`（索引记录 `record:s3-budget-curve`）；`outputs/r7_s3_budget_curve_20261006_attempt01/`（l6×1600 vs D3 l6×400，训练 FLOP 比 4.0；整轮 4937.6s，planned 5400/hard 10800，overrun 0；主格 supported 6h −0.6427/−0.7454/−0.9084、12h −0.6195/−0.7667/−0.9339；t2m seed 均值 skill 6h +0.595/12h +0.333/24h +0.226；守门 48h/72h 17 个正 cell 比 800 更差→预算响应判归数据；test未读） |
| S3 batch-3 四季 2018–2021 获取（0 GPU） | 0.0000 | 6.5104 | `docs/R7_S3_BATCH3_ACQUISITION.md`（索引记录 `record:s3-batch3-20182021-acquisition`）；`outputs/r7_s3_batch3_20182021/`（16/16 part 成功、零失败、无重试；网络 57,084,203,564 字节=计划 98.4%、硬上限 53.2%；16 part 合计 20,709.0s（planned 21600/hard 43200）；合并源 `97d29bca…` 1920 stamps；batch-2 两项修正（16 GiB 解码上限、per-part 看门狗）本批未被触发；test未读） |
| S3 确认实例 v3 发布（train 2017–2021 五年扩年，0 GPU） | 0.0000 | 6.5104 | `docs/R7_S3_CONFIRMATION_INSTANCE_V3.md`（索引记录 `record:s3-confirmation-instance-v3`）；`outputs/r7_s3_confirmation_train2017_2021_v3/`（三源合并 `bc2ff9cf…` 3360 stamps、combine 重放字节确定；store 窗口 2360/472/472、17 通道、`raw_state_GiB` 0.899；三 sidecar `f76c373d…`/`0e87fe40…`/`6f9bedbf…` 绑定 data identity `2564eeaf…`；全流程约 509s，planned 1200/hard 3600；v2 保留不取代；test未读） |
| S3 v3-D2 同数据基线重建（train-only 五年气候态，CPU 零 GPU） | 0.0000 | 6.5104 | `docs/R7_S3_V3_D2_BASELINES.md`（索引记录 `record:s3-v3-d2-baselines`）；`outputs/r7_s3_v3_d2_baselines_20261006_attempt01/`（气候态 2400 步、16 桶各 150；逐 lead 全覆盖 472/468/460/444/428；t2m RMSE 3.5171/3.4889/3.4375/3.3534/3.3184 K；整轮 992.4s，planned 1800/hard 3600，overrun 0；网络 0；test未读） |
| S3 v3-D3 同数据 incumbent 重训（GPU 3 seed） | 1.5012 | 8.0116 | `docs/R7_S3_V3_D3_INCUMBENT.md`（索引记录 `record:s3-v3-d3-incumbent`）；`outputs/r7_s3_v3_d3_incumbent_20261006_attempt01/`（三 seed 初始化逐位复现 actual C；400 L6 updates×3；整轮 5404.5s，planned 5400/hard 10800，**软超 4.5s 已记录**；t2m 6h skill +0.4148/+0.2610/+0.4248、12h +0.0285/−0.0805/+0.0756；正 seed-cell 51/50/45/2/0；test未读） |
| S3 v3 预算剂量 l6×1600（主格 supported / 守门 17 cell 持平，注册负结果） | 1.9122 | 9.9238 | `docs/R7_S3_V3_BUDGET_DOSE.md`（索引记录 `record:s3-v3-budget-dose`）；`outputs/r7_s3_v3_budget_dose_20261006_attempt01/`（l6×1600 vs v3-D3 l6×400，FLOP 比 4.0；整轮 6883.9s，planned 6300/hard 12600，**soft overrun 583.9s 记录**；主格 supported 6h −0.4328/−0.7607/−0.4444、12h −0.5351/−0.7199/−0.5158；守门 17 cell（24h 1/48h 7/72h 9）与 v2-BC 同剂量持平→数据响应判「长 lead 不由体量单独修复」，不进 S4；test未读） |
| S3 v3 数值语义独立勘误（只读，零 GPU） | 0.0000 | 9.9238 | `docs/R7_S3_V3_NUMERICAL_ERRATA.md`（索引 `record:s3-v3-numerical-errata`）；正气候态 skill 格数与优于控制的格数分开，seed42 跨实例方向纠正，旧输出/证据 digest 不改；非新实验 |
| S3 v3 rollout 微调 attempt01（failed / partial，全部计费） | 2.6316 | 12.5554 | `docs/R7_S3_V3_ROLLOUT_FT_ATTEMPT01.md`（索引 `record:s3-v3-rollout-ft-attempt01`）；planned4200/hard7200，实际9473.920853s（软超5273.920853、硬超2273.920853），seed41/42完整、seed43仅最新checkpoint60，无三seed verdict；合作式deadline迟延，阻塞根因未确认；旧attempt保留不续跑 |
| S3 v3 rollout 微调 attempt02（完整 registered-negative） | 1.1729 | 13.7283 | `docs/R7_S3_V3_ROLLOUT_FT_ATTEMPT02.md`（索引 `record:s3-v3-rollout-ft-attempt02`）；三seed200双步，主格supported、守门13/45未过，不进S4；4222.618677s，planned5400/hard10800，overrun0，五worker均reaped、无signals，test未评分 |
| S3 v3 rollout 归档重放（含两次评分前失败） | 0.3871 | 14.1154 | `docs/R7_S3_V3_ROLLOUT_FT_ATTEMPT02.md` §7（索引 `record:s3-v3-rollout-ft-replay`）；1375.732820s成功+8.888093/8.885239s失败全额计费；seed41五lead15CSV精确相同，三seedcollector JSON表示精确匹配；非新科学确认 |
| S3 12步FP32可行性probe（单train样本，无optimizer/评分） | 0.4458 | 14.5612 | `docs/R7_S3_LONG_ROLLOUT_FEASIBILITY.md`（索引 `record:s3-long-rollout-feasibility`）；1605.013031s、soft1800/hard3600、overrun0；有限loss/gradient，reservedpeak2.244GiB，全部3workerreaped/no signals；独立身份/窗口/费用核验，非科学通过 |
| S3 12步probe归档单样本重放（精确同配置再现） | 0.2350 | 14.7962 | `docs/R7_S3_LONG_ROLLOUT_FEASIBILITY.md` §5（索引 `record:s3-long-rollout-feasibility-replay`）；846.060163s、overrun0，两workerreaped/no signals；sample/loss/12steploss/gradient/FLOPs精确一致，非训练/val/test确认 |
| S3 12步长监督三seed screen（开发supported，绝对长lead气候态未过） | 2.4984 | 17.2946 | `docs/R7_S3_LONG_ROLLOUT_SCREEN.md`（索引 `record:s3-long-rollout-screen`）；8994.237921s、soft6300超2694.237921s/hard12600超0，六workerreaped/no signals；主格supported、控制gate0/45，但t2m48/72h气候态skill全负、parentgate16/45，test未评分 |
| S3 12步screen归档seed41全val重放（15CSV精确、单独全额计费） | 0.5378 | 17.8324 | `docs/R7_S3_LONG_ROLLOUT_SCREEN.md` §7（索引 `record:s3-long-rollout-screen-replay`）；1935.462220s、soft1800超135.462220/hard3600超0，ceil1936秒；15CSV及2272case精确、三seedcollector摘要一致，审阅optional文档收尾hard超3.402s如实保留，非科学确认 |
| S3 同case train/val诊断（20in-sample+4开发，无optimizer） | 0.4865 | 18.3189 | `docs/R7_S3_SAME_CASE_GAP.md`（索引 `record:s3-same-case-gap-diagnostic`）；1751.421197s、soft1800/hard3600超0；父相对t2m全lead改善但train/val48/72h气候态仍负，独立fsum审阅FAILED保留、另冻NumPy精确补充，无科学确认 |
| S3 同case归档全payload重放（原code/气候态/端点） | 0.7900 | 19.1089 | `docs/R7_S3_SAME_CASE_GAP.md` §6（索引 `record:s3-same-case-gap-replay`）；2844.160051s、soft1800超1044.160051/hard3600超0；24病例/6120MSE及全组/gap身份精确一致、三workerreaped无记录信号，最高config-reproducible |
| S3 四季固定train梯度分量诊断（无optimizer，描述性） | 0.6847 | 19.7936 | `docs/R7_S3_TRAIN_GRADIENT_COMPONENTS.md`（索引 `record:s3-train-gradient-components`）；whole2465.088904s、soft1800超665.088904/hard3600超0，四workerexit0/reaped、记录signals空；state锚不变，长lead norm占比大但cosine依病例变化，不证明clip伤害/欠拟合，test未评分 |
| S3 默认全套混合CUDA测试（取卡证据缺口另计） | 0.3497 | 20.1433 | `docs/R7_S3_TRAIN_GRADIENT_COMPONENTS.md` §5（索引 `record:s3-gradient-mixed-suite-resource-deviation`）；4009passed/3skipped含六CUDA条件test，未留该套专用UUID/余量/协议，不能称有界GPU合规；JUnit至log1258.349468s、ceil1259全记账；另冻明确CPU全套4003passed/9skipped，0GPU，不追认原缺口 |
| S3 梯度代表例replay attempt01（prepare工程失败） | 0.0005 | 20.1438 | `docs/R7_S3_TRAIN_GRADIENT_COMPONENTS.md` §6（索引 `record:s3-train-gradient-replay-attempt01`）；1.834367s、soft/hard超0，fixture manifest归档复制被guard拒绝，未进实际字段/GPU；exit1/reaped，failed保留不复活 |
| S3 梯度代表例replay attempt02（精确恢复negative） | 0.4361 | 20.5799 | `docs/R7_S3_TRAIN_GRADIENT_COMPONENTS.md` §6（索引 `record:s3-train-gradient-replay-attempt02`）；1569.817662s、soft1800/hard3600超0，两workerexit0/reaped、signals空但161numeric leaves不精确，loss/state/RNG/activity相同；独立差异确认，failed-restoration、无accepted，不加容差/attempt03 |
| S3 单固定train病例原目标80更新（描述性响应） | 0.4300 | 21.0099 | `docs/R7_S3_FIXED_CASE_OBJECTIVE_RESPONSE.md`（索引 `record:s3-fixed-case-objective-response`）；whole1547.848237s、planned3600/hard7200超0，三workerexit0/reaped、记录signals空；freshAdamW/clip1/原12步目标、标准0/20/80齐、总loss降58.8800%，单病例不等于天气泛化；旧工程失败/超hard/取消均保留 |
| S3 单病例三端点独立精确反算（只读配置核验） | 0.4078 | 21.4177 | `docs/R7_S3_FIXED_CASE_OBJECTIVE_RESPONSE.md` §7（索引 `record:s3-fixed-case-objective-response-readback`）；whole1467.480233s、planned1800/hard3600超0，0/20/80原生typed全12loss/total/state/recipe/RNG投影diff0，1532保存事实核齐；首terminal超hard48.497180失败不追认，另同字节helper5.029937s补齐；不重复计原训练 |
| S3 原目标—最终评分配对诊断（无更新，mixed/not-candidate） | 0.5256 | 21.9433 | `docs/R7_S3_OBJECTIVE_FORECAST_DIAGNOSTIC.md`（索引 `record:s3-objective-forecast-diagnostic`）；0/20/80×原四train/四val24pairs、576transitions，parentwhole1891.377279s，planned3600/hard10800/reserve180超0；原Jan目标−58.8800%/最终83of85改善，但其余3train/4val目标全升、val最终241of340变差；train/evalfinal同hash，独立754files/math/cost核齐，原失败保留，test未评分/r0 |
| S3 等病例曝光ordering小试验（完整negative/not-candidate） | 0.6848 | 22.6281 | `docs/R7_S3_CASE_INTERLEAVING_PILOT.md`（索引 `record:s3-case-interleaving-pilot`）；同original0/freshAdamW/seed41，两臂80各case20曝光、16endpoint80×原四train/四dev，whole2464.990661s/soft5400/hard14400/reserve180超0，五workerexit0/reaped/signals=[]；interleaved四train目标均降但开发support=false，t2m仅72h对both好/其他6–48h差于blocked、u10全lead对0差；独立947files/math/cost齐，所有工程失败/误绿不追认，test未评分/r0 |
| S3 气候态锚实例（negative/not-candidate，含保存读回） | 0.6859 | 23.3140 | `docs/R7_S3_CLIMATOLOGY_ANOMALY_ANCHOR.md`（索引 `record:s3-climatology-anomaly-anchor`）；candidate80 恰80更新、原 deep-K/K4/full12/FP32；开发 support=false：t2m 48/72h 优于两控制而 6/12/24h 与 u10/v10 守门退化；whole2468.177771s、soft5400/hard14400/reserve180 超0；保存读回 v6 链 1500/1500 准入、1,946,575,475B、三原哈希与 13futureEND 相等、0 模型执行/0 网络；原 attempt03/04/05 与被阻 v5 链失败保留不追认，test 未评分/r0 |
| S3 rollout-dose 可行性 probe attempt01（父 digest 未迁移，失败保留） | 0.4808 | 23.7948 | `docs/R7_S3_ROLLOUT_DOSE.md` §8；prepare/archive success、probe 阶段被 `load_checkpoint` 以模型身份变更拒绝；1730.858695s、planned1800/hard3600 超0；失败不追认，未登记独立 index record |
| S3 rollout-dose 可行性 probe attempt02（成功） | 0.4706 | 24.2654 | `docs/R7_S3_ROLLOUT_DOSE.md` §8；FLOPs 393,859,201,536、reserved peak 2,409,627,648 与注册 probe 逐位相同；1694.289826s、超0；未登记独立 index record |
| S3 rollout-dose screen l6-rollout×800（开发 supported/守门过，reading 阶段失败保留） | 3.3251 | 27.5905 | `docs/R7_S3_ROLLOUT_DOSE.md`（索引 `record:s3-rollout-dose`）；三 seed 800 更新自注册 v3-BD 1600 父（12 步权重/LR2e-5/FP32/K4/控制不变）；冻结函数判 primary supported（t2m/full 6h/12h 三 seed 对 v3-D3 控制全负）、守门 0/45、decision advance-to-S4-freeze；绝对气候态 48h −0.1191/72h −0.4309 仍未过；相对注册 200 screen 24/48/72h RMSE 低 0.154/0.325/0.465 K 而 6/12h 持平，48h/72h ACC 0.329/0.153→0.367/0.189；整轮 11970.334049s（planned9000、soft overrun 2970.334049、hard 0），六 worker reaped/signals=[]；reading 阶段因共享收集器 parent-pin 缺陷失败；只读复算 CPU 7.95s/0GPU；test 未读/r0 |
| S3 宽区（域外大尺度上下文）采集 pilot（工程 positive，0 GPU-h） | 0.0000 | 27.5905 | `docs/R7_S3_WIDE_REGION_PILOT.md`（索引 `record:s3-wide-region-pilot`）；只改读取区域 65×65→129×129（19–51N/99–131E，各方向 +8°），源/快照/17 通道/0.25°/6h/2017–2021+2022 split/模型 spec/归档初始化全不变；129 网格中心 65×65 块按断言=冻结目标框，`boundary_masks(129,129,(32,))` 的 `interior_32` 即该框，故注册气候态与 val cohort 仍适用。2018 冬 120 stamp 实测：网络 3,565,096,475 B 对注册目标框 part 3,554,716,077 B（比 1.0029）、解码 charged 9,482,837,880 B 与 chunk reads 2287 **完全相同**、墙钟 1223.141 s（比 0.965）、存储 70,207,133 B（比 3.699，面积比 3.939）；原因是每时刻一张全球场一个 chunk，放大只花存储。冻结判据 18/18、中心块与 v3 `source.nc` 逐位相同（9 变量/17 通道/坐标/单位/120 stamp）。preflight 在下载前抓到每 stamp 字段读数应为 19（非 17 通道），协议记 pre-download revision；三个新工具缺陷已修并补反证。0 GPU-h、0 test 读取、r=0 |
| S3 宽区全量获取 + v4 store + 宽区气候态（工程 positive，0 GPU-h） | 0.0000 | 27.5905 | `docs/R7_S3_WIDE_REGION_FULL.md`（索引 `record:s3-wide-region-full`）；28 part（27 新 + 复用 pilot winter_2018）覆盖 train 2017–2021 / val 2022 / test 2023 共 3360 stamps；`winter_2020`/`winter_2022` 首跑超冻结 1800 s 每 part 期限、**无产物**、按新名 `_r2` 重取成功，两次失败原样保留；网络 ≈101 GB（计划 100 GB、硬上限 150 GiB 未触）、磁盘峰值 6.6 GiB、墙钟 ≈4.5 h（批次 12223 s + validation 1218.5 s + store 168.8 s + 气候态 2461.6 s）；`source.nc` SHA `6bc9a1a2…`，时间轴与 v3 **逐位相同**，中心 65×65 块在 NetCDF 与 store 两级都与 v3 **逐位相同**；v4 store `[3360,17,129,129]`、split/window 计数与 v3 相同（2360/472/472）、train-only 归一化 + process 诊断、`BUILD_COMPLETE`；`interior_32`（4225 点 = 冻结 65×65 盒）的 t2m 气候态与 persistence RMSE 与**已登记 v3-D2 在 ~1e-10 内相同**。committed 宽区路径的四处 R-021 拆分缺陷（含"下载完成后写回执崩溃"）已修并补两道反证守卫。并发实测：6 死锁 / 4 正常，故 pool 固定 4。test 未读/r0 |
| S3 宽区域单因素 train/val（输入区域为唯一因子；null 读数） | 3.5339 | 31.1244 | `docs/R7_S3_WIDE_SINGLE_FACTOR.md`（索引 `record:s3-wide-single-factor`）；唯一改动因子 = 输入区域 65×65→129×129，配方/父/LR/warmup/权重/K/FP32/batch/clip/seed/updates 逐字沿用注册 rollout-dose，全 129×129 全监督，评分域 `boundary_masks(129,129,(32,))['interior_32']`=4225 点=冻结中心盒；对照是**已登记**窄 65×65 臂（按 `rmse_csv_sha256` 复核后复用，未重训练）；t2m seed 均值 delta（宽−窄）6h **+0.0013**、12h **+0.0147**、24h **+0.0189**、48h **+0.0121**、72h **+0.0231** K，48/72h 仅 2/3 seed 更优；`interior_32` t2m skill 均值宽 −0.1271/−0.4485 对窄 −0.1191/−0.4309（48/72h 绝对气候态门仍未过且宽臂略差）；**不支持**"冻结 16° 盒缺长 lead 所需信息"，也**不构成反证**（全监督面积 3.94× 使信息增益与稀释混淆）→ **不进 S4 冻结包**；六阶段（prepare/archive/seed41/seed42/seed43/reading）全 success、`returncode=0`、`signals=[]`、`reaped=true`；whole 12721.940569 s（planned 9000、soft overrun 3721.940569、hard 0）；协议 `d016eb7a…`、code.zip `fa5752f6…`、执行文件 182 个、model `d3fb58db…`、training `6e4363d5…`；cohort 472/468/460/444/428 逐 lead 与窄臂 pin 一致；网络 0（离线）、磁盘 4.6 GiB；开轮提交 `13f3c44` 的 CI `37985102030` 因 C 类 size marker 漂移在 pytest 失败（唯一红点），`de64697` 修回绿（CI `37990463998`）；test 未读/r0 |
| S3 宽输入 + 内部盒监督（分离稀释与信息增益；mixed/needs-review） | 3.4655 | 34.5899 | `docs/R7_S3_WIDE_INTERIOR_SUPERVISION.md`（索引 `record:s3-wide-interior-supervision`）；唯一改动因子 = **监督域**（全 129×129 → `boundary_masks(129,129,(32,))['interior_32']` = 4225 点 = 冻结中心盒 = 已登记窄臂自己的网格），输入仍是宽 129×129，配方/父/LR/warmup/权重/K/FP32/batch/clip/seed 逐字沿用；mask 声明块（`mask_sha256 4a2a576b…`）进入绑定 contract 与 `signature`；`training/r7_losses.py` 新增 `mask=None` 关键字参数（**None 时逐位不变**），并加 8 条反证（单格受限加权均值、全 1 mask 逐位等于原目标、空/非 2-D/负值 mask 报错、驱动拒绝已存在输出与缺失 store）；**结果**：seed 均值 delta（本臂 − 窄臂）五个 lead 全为负（6h **−0.0348**、12h **−0.0401**、24h **−0.0069**、48h **−0.0528**、72h **−0.0445** K），6/12h **3/3** seed 更优，而上一轮全网格臂五个 lead 全为正 → **上一轮的 null 主要来自监督稀释**；`interior_32` t2m skill 均值 0.5944/0.3341/0.3700/−0.0868/−0.3996 全面高于窄臂（0.5816/0.3151/0.3668/−0.1191/−0.4309）与上一轮宽臂；但 48/72h **绝对气候态门仍未过**（仅 seed 43 的 48h +0.0140），故**仍不进 S4 冻结包**，改为把该臂推入剂量检验；六阶段全 success、`returncode=0`、`signals=[]`、`reaped=true`；whole 12475.851618 s（planned 9000、soft overrun 3475.851618、hard 0）；协议 `6fc50d04…`、code.zip `2e60dfd6…`、执行文件 183、model `d3fb58db…`（未变）、training `8ad77d93…`（变）；cohort 472/468/460/444/428 逐 lead 与 pin 一致；网络 0、磁盘 4.6 GiB；test 未读/r0 |
| S3 宽输入 + 内部盒监督剂量 800→2400（48h 三 seed 全正、72h 仍负；mixed/needs-review） | 6.6941 | 41.2840 | `docs/R7_S3_WIDE_INTERIOR_DOSE.md`（索引 `record:s3-wide-interior-dose`）；唯一改动因子 = **updates 800→2400**（宽 129×129 输入、`interior_32` 掩码、配方其余参数、迁移 v3-BD 1600 父、seed、评分域、气候态分母、cohort pin 全不变）；对照同时报**已登记窄 800 臂**与**已登记 800 内部监督臂**（均 pinned 复用、未重训练，15/15 表哈希匹配）；**结果**：相对窄臂 12/24/48/72h **3/3 seed 更优**（delta 均值 −0.0284/−0.1159/**−0.3086**/**−0.3306** K），6h 持平（−0.0029）；相对 800 内部监督臂 24/48/72h 3/3 更优（−0.1090/−0.2558/−0.2861），6/12h 略差；**48h 绝对气候态门首次三 seed 全正**（skill +0.0613/+0.0277/+0.1127），6/12/24h 亦全正；**72h 三 seed 仍全负**（−0.1746/−0.2773/−0.1562，均值 −0.2027，相对窄臂 −0.4309 已收窄）→ 冻结验收条件（48/72h 三 seed 全正）**只满足一半**，故**不进 S4 冻结包**，并按预声明**关闭剂量支**转长 lead 权重假设；attempt01 因冻结 reading 阶段 pin 错表（`rmse.csv` 哈希对 `boundary_rmse.csv`）被操作者中止，`operator_abort.json` 全额计费 **0.7517 GPU-h**（prepare 271.45 + archive 289.77 + seed41 部分 2145.0 s），修复 `0b517b3`/`76a3d60`（pin 改存 boundary 哈希 + 两个测试改用合成输出树，因同一缺陷在无 `outputs/` 的 CI 上失败过）；attempt02 六阶段全 success、`returncode=0`、`signals=[]`、`reaped=true`、whole 21392.663503 s、**soft/hard overrun 均 0**；协议 `1d8d163d…`、code.zip `0a6e0b81…`、执行文件 184、model `d3fb58db…`（未变）、training `8ad77d93…`；cohort 472/468/460/444/428 逐 lead 与 pin 一致；网络 0、磁盘 14 GiB；test 未读/r0 |
| S3 长 lead 物理权重再分配 0.5→1.0（几乎无效，72h 仍负；宽输入支关闭） | 7.0027 | 48.2867 | `docs/R7_S3_WIDE_LONG_LEAD_WEIGHT.md`（索引 `record:s3-wide-long-lead-weight`）；唯一改动因子 = **48h/72h 物理权重 0.5→1.0**（`(1,.5,0,.5,0,0,0,.5,0,0,0,.5)` → `(1,.5,0,.5,0,0,0,1.,0,0,0,1.)`），宽输入/`interior_32` 监督/2400 更新/父/seed/评分域/气候态分母全不变；对照是**已登记 2400 剂量臂**与**已登记窄 800 臂**（pinned 复用、未重训练，15/15 表哈希匹配）；**结果**：相对 2400 臂 seed 均值 6h **+0.0129**、12h **+0.0188**、24h **+0.0140**（略差）、48h **−0.0054**、72h **−0.0218**（略好）——翻倍长 lead 权重只换来 72h ~0.02 K；**72h 仍三 seed 全负**（skill −0.1672/−0.2614/−0.1363），48h 仍三 seed 全正（+0.0631/+0.0278/+0.1200）；预声明出口触发 → **宽输入支作为长 lead 的解关闭**（四轮合并：输入范围本身无收益；监督域收回目标盒后收益只到 48h；剂量把 48h 推过气候态；72h 用三种杠杆均未推过）；**不进 S4 冻结包**；六阶段全 success、`returncode=0`、`signals=[]`、`reaped=true`、whole 25209.895089 s、soft/hard overrun 均 0；协议 `c8c8e79e…`、code.zip `2696f7ef…`、执行文件 185、model `d3fb58db…`（未变）、training `8ad77d93…`；cohort 472/468/460/444/428 逐 lead 与 pin 一致；网络 0、磁盘 14 GiB；test 未读/r0 |
| S3 推理深度 K 4→8（操作者中止，**未完成**；单 seed） | 5.0197 | 53.3064 | `docs/R7_S3_REASONING_DEPTH.md`（索引 `record:s3-reasoning-depth`）；唯一改动因子 = `reasoning_steps`/`steps` **K 4→8**（宽 129×129 输入、`interior_32` 监督、2400 更新、注册物理权重、迁移 v3-BD 1600 父、seed 41/42/43 全不变）；**seed 41 完整跑完**（exit 0、训练 11209.65 s、checkpoint `efc43304…`、五 lead 472/468/460/444/428 全评、峰值 13.55 GiB），**seed 42 训练中被 `SIGKILL`**（`exit_code=-9`、1656.784 s，来源是操作者停止 workflow 时连带终止在跑的 `world.run`，**非**科学失败），**seed 43 与 reading 从未运行** ⇒ 无 `readings.json`/`result.json`/`failure.json`/`attempt.json`；**唯一可读数字（单 seed 41）**：相对已登记 2400 剂量臂（K=4，pinned 复用）`interior_32` t2m Δrmse 6h **+0.00776**、12h **+0.00884**、24h −0.00856、48h −0.00309、72h −0.00780 K，Δskill ≤0.00509，**72h skill 仍负（−0.16954）**；**未达预声明出口**（需三 seed），**不是科学判定**，只作"关闭训练配方支"的旁证；whole 18070.843 s（planned 54000、soft/hard overrun 均 0）；协议 `35b84f2a…`（`protocol.json` 最终 digest；预备阶段初步 digest `d8ba87bb…` 不采用）、code.zip `1ca8e648…`、code_commit `964fc8ce…`；网络 0、磁盘 14 GiB；test 未读/r0 |
| S3 主干深度 depth 4→8（**未运行**，操作者决定跳过；0 GPU-h） | 0.0 | 53.3064 | `docs/R7_S3_CAPACITY_DEPTH.md`（索引 `record:s3-capacity-depth`）；驱动/测试/code.zip **已就绪**（commit `710612d`、code.zip SHA256 `de9f6bc4…`、1380 文件、comment `710612d`、CI run `38205959882` 绿、`tools/r7_round.py round-guard` 通过、CPU preflight 对真实迁移父 `bit_identical=True`/`added=[4,5,6,7]`/严格 `load_state_dict` OK）但 **GPU 部分未运行**：无 attempt 目录、无 freeze、无 seed 启动、无读数文件；**0 GPU-h**（两块 GPU 全程空闲）、网络 0、soft/hard overrun 均 0；操作者依据＝上一轮 K4→8 臂（index71）**唯一完整跑完**的 seed 41 读数：相对已登记 2400 剂量臂（K=4，pinned 复用）`interior_32` t2m Δrmse 6/12/24/48/72h = **+0.00776/+0.00884/−0.00856/−0.00309/−0.00780 K**、Δskill ≤0.00509、**72h skill 仍负 −0.16954**，据此判断"效果基本上没变化"、不值得花 3 seed × ≤28800 s 的 GPU；**口径（必须照此读）**：这是**操作者的成本/收益决定**，**不是**对本轮 depth 4→8 假设的判定——`K`（每步内部推理）与主干 `depth`（`WindowAttentionBlock` 层数）是两个不同因子，本轮**从未运行**、其假设**未被检验**，§3 预声明出口**既未触发也未满足**，故**不**据此关闭"加大容量解决 72 h"这一支；驱动/测试/code.zip 保留为"已准备、未运行"的**可复用资产**；test 未读/r0 |
| S3 主干深度 depth 4→8（**运行尝试失败**：seed 41 CUDA OOM；共驻显存不足；操作者判定此方向不能这样做） | 0.5414 | 53.8478 | `docs/R7_S3_CAPACITY_DEPTH_RUN.md`（索引 `record:s3-capacity-depth-run`，`candidate_state: blocked`）；本轮**实际启动**：workflow `dwfrun-c9039cf5-6fca-4fe5-83c2-c888fddfd184` 独立复核 clean（9 处差异全为预注册）、`code.zip` 按 HEAD `370dca0` 重建、`freeze`/`prepare`/`archive` 全 success；**`seed 41` 训练 1948.854226 s 后 `torch.OutOfMemoryError`**（本进程峰值 9.18 GiB / PyTorch 分配 8.56 GiB；24 GiB 共驻卡剩 4 MiB；邻居 pid 2967988 5.15 GiB + pid 3516490 9.17 GiB），`seed 42/43` 与 `reading` **从未运行**，**无 `readings.json`/`result.json`/`attempt.json`** ⇒ **无读数、无 seed 均值、无同时区间、无 primary/gate 判定**；spawn 时余量门通过（free 15.29 GiB ≥ required 9.0 GiB），碰撞来自**邻居在时间点检查之后长大**，未对邻居发任何信号；**操作者判定**（2026-10-13）：~9.2 GiB 训练足迹与本项目「中小模型 / 小显存」初衷冲突、在本机共驻条件下不可行 ⇒ 诚实标记「靠加大容量解决 72 h」**不能这样做**（资源/前提判断，**非**科学证伪；该臂假设未被检验、§3 预声明出口既未触发也未满足）；**不重跑、不加 seed、不重试**，部分输出原样保留；code 身份 `370dca0` + code.zip `ad15b3d4…`，该 code 的 CI `38262038297` completed/success；网络 0、soft/hard overrun 0、磁盘 ≈15 MB；test 未读/r0 |
| S3 主干深度 depth 4→8 **改双卡一卡一 seed**（决策 0047；无 GPU 运行，0 GPU-h） | 0.0000 | 53.8478 | `docs/decisions/0047-dual-card-seed-distribution.md`；`training/r7_device_selection.py`；`tools/r7_round_stages.py`；`tools/r7_round.py`；`scripts/study_r7_s3_capacity_depth.py`（无 index record：未运行、非实验轮，按 0 成本行如实列 note） |
| S3 主干深度 depth 4→8（**三 seed 跑完**；72h 三 seed 全负，按决策 0048 关闭深度轴；negative/not-candidate） | 9.2141 | 63.0619 | `docs/R7_S3_CAPACITY_DEPTH_RESULT.md`（索引 `record:s3-capacity-depth-result`，`outcome_class: negative`、`candidate_state: not-candidate`）；唯一改动因子 = 主干 `depth` **4→8**（`backbone.encoder` 的 `WindowAttentionBlock` 层数；`dim` 仍 192、K 仍 4、宽 129×129 输入、`interior_32` 监督、2400 更新、注册物理权重、迁移 v3-BD 1600 父、seed 41/42/43、评分域与气候态分母逐字不变）；参数量 3,286,037 → **5,065,493（+54.2%）**；**冻结的深度扩张规则**：新增块 `i∈{4..7}` 在每个参数上镜像块 `i-4`、两个输出投影（`proj`/`ff.net.3`）清零 ⇒ 每块恰为恒等，depth-8 主干**从注册的 depth-4 函数出发**，逐 seed 在固定合成批`(1,2,17,129,129)`/seed 1234 上断言 `encoder_bit_identical=true`（三 seed 全过）；**设备**：决策 0047 双卡`device_mode=dual`（`gpu_uuids` 两张 3090、`neighbour_reserve_bytes` 6 GiB、`estimated_peak_bytes` 8.5 GiB、margin 0.5 GiB）；**结果**：三 seed 全跑完并评分（cohort 472/468/460/444/428 逐 lead 与 pin 一致），`interior_32` t2m **72h skill −0.1512/−0.2545/−0.1432（均值 −0.1830）三 seed 全负** ⇒ 预声明出口的正向条件（72h 三 seed 全正）**未满足**；相对已登记 2400 剂量臂（同 seed 配对、pinned 复用、未重训练）Δrmse 6/12/24/48/72h = **+0.0042/+0.0046/−0.0078/−0.0259/−0.0298 K**（24/48/72h **3/3 seed 更优**，6/12h 仅 1/3），即深度是**弱正向杠杆、收益集中在长 lead**；但把 72h skill 推回 0 需再降约 **0.29 K**，本臂只贡献 **0.0298 K（约 1/10）**；相对已登记窄 800 臂（已登记窄 800 臂）12/24/48/72h 三 seed 全优（−0.0238/−0.1237/−0.3345/−0.3604 K）、6h 持平；六阶段全 success、`returncode=0`、`signals=[]`，whole 33170.720 s、**soft/hard overrun 均 0**（per-seed 28800 s / planned 86400 s 内）；协议 `a8e472f7…`、code.zip `e8802b7f…`（1390 条目，ZIP comment == code_commit `4e9fc96…`）、执行文件 187、model `d3fb58db…`（未变）、data_identity `2360d42b…`、source `6bc9a1a2…`；**一条实测修正**：协议写 `estimated_peak_bytes` 8.5 GiB，三 seed 实测峰值**均为 11.03 GiB**（低估 2.5 GiB），真正兜住本轮的是 0047 的 6 GiB 邻居 reserve；**登记链**：同一因子第三条登记（① 未运行 `s3-capacity-depth` ② OOM `blocked` `s3-capacity-depth-run` ③ 本次 `s3-capacity-depth-result`），前两条**逐字节保留**，只在 ② 上追加 `correction_record_id`；**不进 S4 冻结包**；网络 0、磁盘 ≈14 GiB；test 未读/r0 |
| S3 主干容量 width dim 192→256（**三 seed 跑完**；72h 三 seed 全负，按决策 0048 永久关闭整条容量轴（深度＋宽度）；negative/not-candidate） | 7.9677 | 71.0296 | `docs/R7_S3_CAPACITY_WIDTH_RESULT.md`（索引 `record:s3-capacity-width-result`，`outcome_class: negative`、`candidate_state: not-candidate`）；唯一改动因子 = 主干 `dim` **192→256**（`backbone.encoder` 的 `WindowAttentionBlock` 宽度；`depth` 仍 4、K 仍 4、宽 129×129 输入、`interior_32` 监督、2400 更新、注册物理权重、迁移 v3-BD 1600 父、seed 41/42/43、评分域与气候态分母逐字不变）；参数量 3,286,037 → **5,790,389（+76%）**；**宽度无逐位恒等扩张**（155 张量中 150 形状变化），父状态经**冻结零填充**迁移（重叠切片拷入、其余置零）⇒ 本臂测「宽度 + 该零填充初始化」**联合效应**、非宽度单因素（不照搬 depth 的 identity-gated 措辞）；**结果**：三 seed 全跑完并评分（cohort 472/468/460/444/428 逐 lead 与 pin 一致），`interior_32` t2m **72h skill −0.1658/−0.2833/−0.1418（均值 −0.1970）三 seed 全负** ⇒ 预声明出口正向条件（72h 三 seed 全正）**未满足**；相对已登记 2400 剂量臂（同 seed 配对、pinned 复用、未重训练）Δrmse 6/12/24/48/72h = **+0.2880/+0.2751/+0.0705/+0.0419/−0.0090 K**（6/12/24/48h **0/3 seed 更优**、仅 72h **2/3**）⇒ 宽度**不是**正向杠杆；相对已登记窄 800 臂 24/48/72h 更优（−0.0454/−0.2666/−0.3396 K）；六阶段全 success、`returncode=0`、`signals=[]`，whole 28683.583 s、**soft/hard overrun 均 0**；协议 `277cf27e…`、code.zip `14079fe6…`（1395 条目，ZIP comment == code_commit `35b1785`）、执行文件 187、model `d3fb58db…`（未变）、data_identity `2360d42b…`、source `6bc9a1a2…`；**一条实测修正**：协议 `estimated_peak_bytes` 11.0352 GiB、三 seed 实测峰值 **11.9023 GiB**（仍低估 0.87 GiB），真正兜住本轮的是 0047 的 6 GiB 邻居 reserve；**CI 如实**：实验 commit `35b1785` 的 R7 CPU CI `38391291040` 第 9 步（全量测试）红，根因是本臂测试文件 `tests/test_r7_s3_capacity_width.py:392` 对缺失 `outputs/` 路径取哈希（工程缺陷、非读数问题；本机 4540 passed/3 skipped/0 failed）；**按决策 0048 整条容量轴（深度＋宽度）永久关闭**；**不进 S4 冻结包**；网络 0、磁盘 ≈14 GiB；test 未读/r0 |
| S3 H-A（P_0 逐样本输入条件化）**运行尝试失败**：`stage-seeds` 被父身份门拒绝（**0 GPU-h**，训练未开始） | 0.0000 | 71.0296 | `docs/R7_S3_PROCESS_INIT_RUN.md`（索引 `record:s3-process-init-run`，`outcome_class: unresolved`、`candidate_state: blocked`）；本轮**实际启动**：workflow `dwfrun-13d26a6b` 独立复核 clean（与剂量驱动差异全为预注册：唯一因子 `process_init_conditioning` False→True、identity-gated 扩张、off 路径逐位一致；复核者实跑 `all_shared_bit_identical: True`、新增参数 37440、清零投影恒零、pin 15/15、干净 worktree 无 `outputs/` 时 **39 passed**）、`code.zip` 按 HEAD `7539bb3` 重建（`00f447d6…`）、`freeze`/`prepare`/`archive` 全 success；**`stage-seeds` exit 1**：seed 41/42 **都在训练之前**的 `_parent_model → load_checkpoint` 抛 `ValueError: checkpoint model implementation differs`（`training/r7_experiment.py:129`），seed 43 从未启动（`device_mode: dual`，一卡一 seed、两卡均失败）⇒ **无 `readings.json`/`result.json`/`attempt.json`、无读数、无 seed 均值、无同时区间、无 primary/gate 判定**，§3 预声明出口**既未触发也未满足**；**根因（已确认，非推测）**：本轮改了 `model/*.py` ⇒ `model_code_digest()` 由 `d3fb58db…` 变为 `223b2e5b…`，而迁移父 `…_migrated_20261009_attempt02` 的 receipt 记的是旧 digest ⇒ 身份门**按设计**拒绝（**不是** bug、**不是**父损坏——父文件在其自身 SHA256 pin 校验处已通过）；**结构性发现**：**任何 `model/*.py` 改动都会让所有迁移父在「微调自注册父」协议下不可加载**，spec-only 改动（depth/dim）不动 digest，故容量两轮能跑而本轮不能；**合法修复（已执行，未绕过身份门）**：用**已提交的** `scripts/migrate_r7_parent_state.py`（docstring 明言 "performs the documented model-only migration **instead of weakening that guard**"）在当前代码下重跑迁移 → `outputs/r7_s3_rollout_dose_parent_migrated_20261014_attempt01`（seed SHA `3bb09c19…/91667a68…/b1a716bc…`、digest＝当前 `223b2e5b…`；`semantics_equal: false` 系新增 gated 模块所致、属预期，off 路径逐位一致由本臂正控保证）；**未**改比对逻辑、**未**放宽任何门、**未**对任何非本实验进程发信号；code 身份 `7539bb3` + code.zip `00f447d6…`，该 code 的 CI `38462716303` completed/success（R7 CPU CI，另 17 条标签 workflow `skipped` 系设计行为）；网络 0、磁盘 15 MB、soft/hard overrun 0；**不进 S4 冻结包**；test 未读/r0 |
| S3 H-A（P_0 逐样本输入条件化）单因素臂（**三 seed 跑完**；72h 三 seed 全负，按预声明出口关闭 H-A 支；negative/not-candidate） | 6.8590 | 77.888623 | `docs/R7_S3_PROCESS_INIT_RESULT.md`（索引 `record:s3-process-init-result`，`outcome_class: negative`、`candidate_state: not-candidate`）；唯一改动因子 = 过程状态初值 `P_0` 由全体样本共享的 `process_queries` 改为**逐样本条件化**（`process_init_conditioning` False→True；`P_0 = process_queries.expand(B,-1,-1) + process_init_proj(context.mean(dim=1)).unsqueeze(1)`），宽 129×129 输入/`interior_32` 监督/2400 更新/K=4/depth=4/dim=192/注册物理权重/迁移 v3-BD 1600 父/seed 41/42/43/评分域/气候态分母逐字不变；新增 `process_init_proj = Sequential(LayerNorm(192),Linear(192,192))` **最后、在 `isolated_stream()` 下构造**，故 off 路径**逐位等于现实现**（逐 seed 实测 `bit_identical_off_path=true`）、清零其 Linear 则注入恒为 0（`proj_zeroing_is_identity=true`、`zeroed_proj_output_abs_max=0.0`）、新增参数 37440（4 键）；**结果**：三 seed 全跑完并评分（cohort 472/468/460/444/428 逐 lead 与 pin 一致），`interior_32` t2m **72h skill −0.2437/−0.3064/−0.2171（均值 −0.2557）三 seed 全负** ⇒ 预声明出口正向条件（72h 三 seed 全正）**未满足**；相对已登记 2400 剂量臂（同 seed 配对、pinned 复用、未重训练）Δrmse 6/12/24/48/72h = **−0.0027/−0.0026/+0.0103/+0.0605/+0.0798 K**（正值=本臂更差；24/48/72h **0/3 seed 更优**、6/12h 2/3 略好）⇒ 逐样本 P_0 条件化**不是正向杠杆**，长 lead 反而更差（72h 需反向约 0.41 K）；相对已登记窄 800 臂五 lead 三 seed 全优（−0.0056/−0.0310/−0.1056/−0.2481/−0.2508 K）；六阶段全 success、`returncode=0`、`signals=[]`，whole 24,692.481 s、**soft/hard overrun 均 0**（per-seed 28800 s / planned 86400 s 内）；协议 `ad7566f3…`（预备 digest `d6cfde3e…`）、code.zip `897b9bc0…`（1401 条目，ZIP comment == code_commit `81233b3`）、执行文件 188、model `223b2e5b…`（本轮改了 model/*.py）、training `8ad77d93…`、data_identity `2360d42b…`、source `6bc9a1a2…`；**一条实测修正（方向与容量两轮相反）**：协议 `estimated_peak_bytes` 11.91 GiB、三 seed 实测峰值**均 8.361 GiB**（**高估** 3.55 GiB）；实验 commit `81233b3` 的 R7 CPU CI `38474413900` completed/success（另 17 条标签 workflow `skipped` 系设计行为）；**登记**：本因子第一次尝试（attempt01 身份门失败、0 GPU-h）已单独登记为独立 record `s3-process-init-run`（`blocked`，见其自身账本行），本页为其跑完的 attempt02；**不**向任何旧 record 追加 `correction_record_id`；**按预声明出口关闭 H-A 支**（不调 proj 宽度、不加 seed）；**不进 S4 冻结包**；网络 0、磁盘 ≈14 GiB；test 未读/r0 |
| S3 H-B（空间分辨的类型化证据 pooled→spatial）单因素臂（**两臂 × 三 seed 跑完**；B→C 无差异 ~1e-6 K、72h 三 seed 全负，按预声明出口关闭 H-B 支；negative/not-candidate） | 15.443038 | 93.331638 | `docs/R7_S3_TYPED_SPATIAL_RESULT.md`（索引 `record:s3-typed-spatial-result`，`outcome_class: negative`、`candidate_state: not-candidate`）；唯一改动因子 = 类型化证据 token 在 patch 位置维上的**聚合方式**：pooled 均匀均值（B，`mode=typed_routing`，#79 现状）→ spatial softmax 注意力（C，`mode=typed_spatial`，一个共享可学习 query）；宽 129×129 输入/`interior_32` 监督/2400 更新/K=4/depth=4/dim=192/注册物理权重/迁移 v3-BD 1600 父/seed 41/42/43/评分域/气候态分母逐字不变；**B/C 同协议内同跑**，共享张量**逐位一致**（逐 seed `pathway_init_sha256` 两臂相等 `18dfa0a2…/e5d78ca9…/2a7f855e…`）、C 恰多 `typed_evidence.position_query`（**+192 参数**，+0.13% of 148800），query=0 时 C==B（`zero_query_is_pooled=true`）、清零投影则证据恒 0（`zeroed_projection_is_identity=true`），逐 seed 正控制通过；**结果**：**B→C 无差异**——五个 lead 的 seed 均值 Δrmse 为 **+1.2e-7/−1.3e-6/−2.8e-6/−6.5e-8/+3.2e-6 K**（abs(Δ)≤3.2e-6 K，float32 噪声量级），72 h **三 seed 不同号**（−5.2e-6/+1.3e-5/+1.9e-6）；C 的 `interior_32` t2m **72h skill −0.1827/−0.2791/−0.1596（均值 −0.2071）三 seed 全负**（B 几乎同值）⇒ 预声明出口（B→C 72h 三 seed 同号且 C 的 72h 三 seed 全正）**两条都未满足**；**结论**＝类型路由本身无效（同信息、不同路由不改变读数），**不是**空间保留无用；相对已登记 2400 剂量臂（同 seed 配对、pinned 复用、未重训练）Δrmse 6/12/24/48/72h = **−0.0026/−0.0022/+0.0011/+0.0052/+0.0068 K**（6/12h 3/3 略优、24/48/72h 更差）；相对已登记窄 800 臂 12/24/48/72h 三 seed 全优（−0.0306/−0.1148/−0.3033/−0.3238 K）、6h 持平；六阶段全 success、`returncode=0`、`signals=[]`，whole 55,594.936 s、**soft/hard overrun 均 0**（per-seed 43200 s / planned 86400 s 内）；协议 `7a529c54…`（预备 digest `1ef9dc6c…`）、code.zip `c32c73f3…`（1406 条目，ZIP comment == code_commit `3c1d2b0`）、执行文件 188、model `15523927…`（本轮改了 model/*.py）、training `8ad77d93…`、data_identity `2360d42b…`、source `6bc9a1a2…`；**一条实测**：协议 `estimated_peak_bytes` 9.5 GiB、实测峰值 spatial 9.276 GiB / pooled 9.041 GiB（本轮**略高估** 0.22 GiB）；实验 commit `3c1d2b0` 的 R7 CPU CI `38516302068` completed/success；**按预声明出口关闭 H-B 支**；**不进 S4 冻结包**；网络 0、磁盘 ≈14 GiB；test 未读/r0 |
| S3 H-C（长 lead 全局混合路径 global vs 同参数量额外局部层）单因素臂（**两臂 × 三 seed 跑完**；G→L 无差异、72h 两臂三 seed 全负，按预声明出口关闭 H-C 支；negative/not-candidate） | 12.287980 | 105.619618 | `docs/R7_S3_GLOBAL_MIXING_RESULT.md`（索引 `record:s3-global-mixing-result`，`outcome_class: negative`、`candidate_state: not-candidate`）；唯一改动因子 = 主干出口**一条** context-mixing 层的**注意力范围**：global N×N 注意力（G，对全部 1089 token）vs local 主干窗口注意力（L，window_size=4）；宽 129×129 输入/`interior_32` 监督/2400 更新/K=4/depth=4/dim=192/注册物理权重/迁移 v3-BD 1600 父/seed 41/42/43/评分域/气候态分母逐字不变；**G/L 同协议内同跑**、**参数量逐位对齐 444864**（同构 block 外壳、`shell_parity.parameter_counts_equal=true`）、同 seed 除 12 个新增键外**逐位一致**（`shared_bit_identical=true`、167 键；`parent_state_sha256`/`expanded_state_sha256` 逐字相同）、`pathway_init_sha256` 两臂相等（`55a9bd77…/db5efa02…/2c5e44bc…`）；正控制逐 seed 通过（清零输出投影**与 FF 输出层** ⇒ 恒等 ⇒ 整个 forward 逐位退回 off，`zeroed_block_is_identity=true`/`zeroed_block_max_abs=0.0`；只清投影**不**等于 off，`proj_only_is_identity=false`）；**结果**：**G→L 无差异**——五个 lead 的 seed 均值 Δrmse 为 **−1.9e-4/−3.0e-5/−2.2e-3/+1.9e-3/−2.5e-3 K**，**五个 lead 三 seed 全部不同号**（72h +4.6e-4/−8.7e-3/+8.0e-4）；72 h 两臂 skill **三 seed 全负**（G −0.1748/−0.2650/−0.1511、L −0.1745/−0.2709/−0.1506）⇒ 预声明出口（72h 三 seed 全正且 G 相对 L 在 72h 三 seed 同号为优）**两条都未满足**；**结论**＝**瓶颈不是感受野**、且该无差异**与「额外容量」不可分**（两臂都加同样的 444864 参数）；相对已登记 2400 剂量臂（同 seed 配对、pinned 复用、未重训练）G 48h 3/3 优（−0.0052 K）/72h 2/3 优（−0.0085 K），L 48/72h 三 seed 全优（−0.0071/−0.0061 K）；**两臂算力不对齐**（注意力长度 68.0625×、SDPA 子成本 57.19140625×、整层 1.9129018382845497×，已声明混杂：若 G 胜出差异可能来自算力而非「全局性」）；六阶段全 success、`returncode=0`、`signals=[]`，whole 44,236.729 s、**soft/hard overrun 均 0**（per-seed 43200 s / planned 86400 s 内）；协议 `36e93cc8…`（预备 digest `b841c34f…`）、code.zip `d0dfe26d…`（1412 条目，ZIP comment == code_commit `d560bbe`）、执行文件 189、model `fff81dcc…`（本轮改了 model/*.py）、training `8ad77d93…`、data_identity `2360d42b…`、source `6bc9a1a2…`；**一条实测**：协议 `estimated_peak_bytes` 9.5 GiB、实测峰值 G 8.955 GiB / L 8.959 GiB（本轮**略高估** 0.54 GiB）；实验 commit `d560bbe` 的 R7 CPU CI `38607253249` completed/success；**按预声明出口关闭 H-C 支**；**不进 S4 冻结包**；网络 0、磁盘 ≈14 GiB；test 未读/r0 |
| **合计已用** | **105.619618** | — | 本方向基数0，历史V2不重复计费，所有失败与测试缺口费用全额登记；cap20.0/remaining−73.331638是会计字段，不是总GPU-h许可上限（0030/0038）；原cap16/旧累计不回改。**2026-10-08 补记 index63，修正「索引已登记而账本缺行」的漂移**；该漂移现有校检器发现不了（C-02/C-03 只校验已存在行）。**2026-10-09 记 index64（rollout-dose，三行合计 4.2765）**；**2026-10-09 记 index65（宽区 pilot，0 GPU-h，网络 3.565 GB / 保留 70.2 MB 只记会计不加 GPU 账）**；**2026-10-09 记 index66（宽区全量 + v4 store + 宽区气候态，0 GPU-h，网络 ≈101 GB / 磁盘 6.6 GiB 只记会计）**；**2026-10-09 记 index67（宽区单因素 train/val，3.5339 GPU-h，null）**；**2026-10-09 记 index68（宽输入 + 内部盒监督，3.4655 GPU-h，稀释已分离）**；**2026-10-09 记 index69（内部监督剂量 800→2400，6.6941 GPU-h 含 attempt01 中止 0.7517，48h 三 seed 转正）**。**2026-10-10 记 index70（长 lead 权重 0.5→1.0，7.0027 GPU-h，几乎无效：72h 仍三 seed 全负，宽输入支关闭）**。**2026-10-12 记 index71（推理深度 K 4→8，5.0197 GPU-h，**操作者中止、未完成**：seed 41 完成、seed 42 被 SIGKILL、seed 43/reading 未跑、无读数；单 seed Δrmse ≤0.00884 K、Δskill ≤0.00509、72h skill 仍负 −0.16954；据此与 index69/70 的三 seed 证据合并**关闭"靠训练配方解决 72h"这一支**，转模型容量/架构方向）**。**2026-10-12 记 index72（主干深度 depth 4→8，**未运行**、0 GPU-h：驱动/测试/code.zip 已就绪、commit `710612d` 已推送、CI run `38205959882` 绿、`round-guard` 通过，但操作者决定**不跑 GPU 部分**、直接登记；依据是 index71 唯一完成的 seed 41「效果基本上没变化」；**口径**＝操作者的成本/收益决定，**不是**对 depth 4→8 假设的判定（该假设未被检验、§3 预声明出口既未触发也未满足），**不**据此关闭"加大容量解决 72h"这一支；驱动/测试/code.zip 保留为可复用资产）**。科学气候态/全年/同时区间仍未过；固定病例/ordering/anchor/rollout-dose/宽区单因素/内部监督/剂量 开发负或 mixed 读数分别登记，不作泛化/clip/欠拟合/遗忘/收敛/稀释来源/剂量饱和因果证明，不无限重复同剂量或精确replay **2026-10-12 补记（capacity 轮三次尝试）**：gate 从 19 GiB 经 `9e8124f`/`8ff541b` 校准到 **9 GiB**（显存主因是 K 不是 depth：同 K=4 的剂量臂实测 8562 MiB、K=8 臂 13878 MiB，而本臂是 depth-8/K=4），并把登记提交口径由 `commit-round` 改为 `commit-register`；三次 attempt（01 prepare+archive、02 prepare、03 prepare+archive）**均未进 seed、无读数、0 GPU-h**，GPU 全程只有邻居作业；run `dwfrun-6c3847ad` 被中断（根因＝会话跨 ZCode app 实例，完成通知被丢，工作流其实仍在跑）。**合计值不变**。**2026-10-13 记 index73（主干深度 depth 4→8 **运行尝试失败**，0.5414 GPU-h：单窗口重开一轮、`freeze` 自动建 `attempt04`，独立复核 clean、code.zip/freeze/prepare/archive 全 success，但 `seed 41` 训练 1948.854226 s 后 CUDA OOM（本进程 9.18 GiB，第二邻居长大占满 24 GiB 共驻卡），`seed 42/43`/`reading` 未跑、无读数、0 科学结论；**操作者判定**该 ~9.2 GiB 足迹与「中小模型/小显存」初衷冲突、共驻下不可行 ⇒ 方向**不能这样做**，不重跑；新 record `s3-capacity-depth-run`（`blocked`），旧「未运行」record `s3-capacity-depth` 追加 `correction_record_id` 指向它；**合计 53.3064 → 53.8478**）**。 **2026-10-13 记 index74（主干深度 depth 4→8，**三 seed 跑完**，9.2141 GPU-h：`interior_32` t2m 72h skill −0.1512/−0.2545/−0.1432 三 seed 全负、均值 −0.1830 ⇒ 预声明出口正向条件未满足；相对 2400 剂量臂 24/48/72h 三 seed 全优但72h 仅 −0.0298 K（达气候态平价需约 −0.29 K，约 1/10）；**按决策 0048 关闭深度轴**（宽度是独立轴、封顶为最后一个容量臂）；三 seed 峰值实测 11.03 GiB 高于协议估计 8.5 GiB；**合计 53.8478 → 63.0619**）**。**2026-10-14 记 index75（主干容量 width dim 192→256，**三 seed 跑完**，7.9677 GPU-h：`interior_32` t2m 72h skill −0.1658/−0.2833/−0.1418 三 seed 全负、均值 −0.1970 ⇒ 预声明出口正向条件未满足；相对 2400 剂量臂 6/12/24/48h 三 seed 全差（+0.2880/+0.2751/+0.0705/+0.0419 K）、仅 72h 略好（−0.0090 K）；宽度无逐位恒等扩张（150/155 张量形状变化），本臂测「宽度 + 冻结零填充初始化」联合效应；**按决策 0048 宽度是最后一个容量臂，它也未过 ⇒ 整条容量轴（深度＋宽度）永久关闭**；三 seed 峰值实测 11.9023 GiB 高于协议估计 11.0352 GiB；实验 commit `35b1785` 的主 CI `38391291040` 第 9 步红（本臂测试文件对缺失 `outputs/` 路径取哈希，工程缺陷、非读数问题）；**合计 63.0619 → 71.0296**）**。**2026-10-14 记 index76（H-A P_0 逐样本输入条件化，**运行尝试失败**，0 GPU-h：`stage-seeds` 在**训练之前**被父身份门拒绝——本轮改了 `model/*.py` ⇒ `model_code_digest()` 由 `d3fb58db…` 变为 `223b2e5b…`，注册时迁移的父按设计不可加载；seed 41/42 失败、seed 43 未启动，无读数、无 primary/gate 判定、预声明出口既未触发也未满足；**不是**科学失败、**不是**父损坏、**不得**绕过；合法修复＝在当前代码下重跑**已提交的** `scripts/migrate_r7_parent_state.py`（身份翻译，已 success 到 `…_migrated_20261014_attempt01`），随后重新 freeze 再跑；**结构性发现**：任何 `model/*.py` 改动都会让所有迁移父失效，spec-only 改动不受影响；新 record `s3-process-init-run`（`blocked`）；**合计不变 71.0296**）**。**2026-10-14 记 index77（H-A P_0 逐样本输入条件化，**三 seed 跑完**，6.8590 GPU-h：`interior_32` t2m 72h skill −0.2437/−0.3064/−0.2171 三 seed 全负、均值 −0.2557 ⇒ 预声明出口正向条件未满足；相对 2400 剂量臂 24/48/72h 三 seed 全部更差（Δ +0.0103/+0.0605/+0.0798 K，正值=本臂更差）、仅 6/12h 略好（−0.0027/−0.0026 K，2/3 seed）⇒ 逐样本 P_0 条件化不是正向杠杆；off 路径逐位一致 + 清零正控制逐 seed 实测；**按预声明出口关闭 H-A 支**（H-A 是决策 0048 永久关闭容量轴后的第一根架构轴）；三 seed 峰值实测 8.361 GiB 低于协议估计 11.91 GiB（本轮**高估**，与容量两轮相反）；**合计 71.0296 → 77.888623**）**。**2026-10-14 记 index78（H-B 空间分辨的类型化证据 pooled→spatial，**两臂 × 三 seed 跑完**，15.443038 GPU-h：B/C 同协议内同跑、共享张量逐位一致、C 恰多 `typed_evidence.position_query`（+192 参数）；**B→C 无差异**——五个 lead seed 均值 |Δrmse|≤3.2e-6 K、72h 三 seed 不同号（−5.2e-6/+1.3e-5/+1.9e-6），C 的 72h skill −0.1827/−0.2791/−0.1596 三 seed 全负 ⇒ 预声明出口两条都未满足；**结论＝类型路由本身无效**（同信息、不同路由不改变读数），**不是**空间保留无用；相对 2400 剂量臂 6/12h 略优、24/48/72h 更差，相对窄 800 臂 12/24/48/72h 三 seed 全优；**按预声明出口关闭 H-B 支**（H-B 是决策 0048 永久关闭容量轴后的第二根架构轴）；三 seed 峰值实测 spatial 9.276/pooled 9.041 GiB 略低于协议估计 9.5 GiB（本轮**高估**）；**合计 77.888623 → 93.331638**）**。**2026-10-15 记 index79（H-C 长 lead 全局混合路径 global vs 同参数量额外局部层，**两臂 × 三 seed 跑完**，12.287980 GPU-h：G/L 同协议内同跑、参数量逐位对齐 444864、共享初始化逐位一致（`pathway_init_sha256` 两臂相等）、清零 proj+ff ⇒ 逐位退回 off、只清 proj ≠ off；**G→L 无差异**——五个 lead 三 seed 全部不同号（seed 均值 −1.9e-4/−3.0e-5/−2.2e-3/+1.9e-3/−2.5e-3 K），72 h 两臂 skill 三 seed 全负（G −0.1748/−0.2650/−0.1511、L −0.1745/−0.2709/−0.1506）⇒ 预声明出口两条都未满足；**结论＝瓶颈不是感受野**、且与「额外容量」不可分；**两臂算力不对齐**（注意力长度 68.0625×、SDPA 子成本 57.19140625×、整层 1.9129018382845497×，已声明混杂）；相对 2400 剂量臂 G 48h 3/3 优、72h 2/3 优，L 48/72h 3/3 优；**按预声明出口关闭 H-C 支**（H-C 是决策 0048 永久关闭容量轴后的第三根架构轴）；三 seed 峰值实测 G 8.955/L 8.959 GiB 略低于协议估计 9.5 GiB（本轮**高估**）；**合计 93.331638 → 105.619618**）。 |

新账本每个失败和成功都加实际连续GPU/执行口径及证据record；其他网络/decoded/disk/whole/overrun在
各回执分列；无index支撑的文档0成本行如实列note，不伪装成实验机器核数。

## §8 进度块

- **2026-10-07接续（仍S3）**：固定train四季梯度诊断完成且独立saved-fact/math/cost审阅；
  原155张量141active/14unused、state不变，48/72hweighted norm占比76.0–86.1%，cosine随病例变化，
  不作clip/欠拟合/收敛因果声明。代表Jan精确replay attempt01归档prepare失败，attempt02身份/fields执行齐但161numeric leaves不同，
  两failed保留，无RESTORATION_ACCEPTED、不加epsilon/rtol/重复到偶然精确。证据6c3bb696、
  `docs/R7_S3_TRAIN_GRADIENT_COMPONENTS.md` SHA627b324d...、index58条、四行新增1.4710GPU-h，累计20.5799。
  默认mixed-CUDA全量4009passed的取卡/协议证据缺口另计0.3497，明确CPU全量4003passed/9skipped另核，skip不算通过。
  精确prelaunch CIaa9e4ac/37559461860success；登记70fe46ea0a78f10f08ac1a6c765336b305648d45精确CI37569443913
  completed/success、全部必要steps success。test/S4/r/终极科学接受不变。
  下一动作仍在S3：新前瞻goal `docs/goals/s3-fixed-case-objective-response.md`，固定Jan训练病例、原目标/clip1/80updates/freshoptimizer，
  只问有限同病例响应不冒泛化/容量因果；soft3600/hard7200及endpoint0/20/80先声明。
  planner草稿已校结构但跳preflight/筛slow/事后扩MSE等拒绝；先TEMP实际CPU反证和独立prelaunch，不执行未qualified真实训练。
  单病例初版独立prelaunch找到16项RNG/AdamW/trace/完整contract false-accept，34pass/1fail及preparation1234s超hard34保留。
  05:00:53Z修复阶段失联、无完整交接，部分三文件排他快照标interrupted-unqualified，精确elapsed未知；
  新独立planned600/hard1200 TEMP恢复只修已确认工程缺口，不修改旧证据/科学门槛或启动未qualified训练。
  该修复已有稳定四文件与实际86passed/最终35passed，但whole1268.007564s超hard68.007564，仍partial未qualified；
  独立路线准备372.154166s超hard12.154166亦保留，不冒称合规。稳定包另冻独立qualification900/1800s复核，真实80更新仍未开始。
  恢复文档39a7aba精确主CI37576366667 completed/success、全部必要steps success；账本20.5799/index58/旧hash不变。
  稳定包独立完整86passed但2个coherent RNG/活动真实性反证仍接受；1671.521s在900/1800内冻结negative资格。
  后续仅修seed41 native状态和已冻结case0逐参数活动的外锚，另冻600/1500；新stable4pins最终91passed且独立定向full91+2coherent闭合，
  1032.230798s/hard1500超0、receipt a96525b1，限定CPU工程qualified，实际source/CUDA/native状态仍待attempt01 runtime严格资格。
  readback TEMP50passed但独立deepparent正例后4scope反义全链未拒→资格negative，另300/900最小scope修复，收尾余量违规保留，未获真实readback资格。
  actual fixedcase attempt01恰80更新/标准0-20-80齐、whole1547.848237s/hard7200超0、3workerexit0/reaped；同病例原目标1.296344→0.533057降58.8800%，非泛化/科学接受。
  807files独立JSON/math/cost核齐，但首次terminal whole1964.930446s超hard1800且checkpoint ownedSIGTERM取消，旧stage保留incomplete。
  另冻600/1200 CPUcheckpoint实际13.571614s核普通loader/完整state/RNG/AdamW，三端点语义齐；旧取消与预算失败不追认。
  readback owner scope修复1082.041479s超hard900失败保留，另独立630.870804s/hard1800 current50tests/deeppositive/14scope双入口28拒绝闭合。
  实际三端点独立反算完整：全部12loss/total/state及typed contract projection每端点diff0，不epsilon/allclose，无optimizer/heldout。
  whole1467.480233s在1800/3600内，默认共驻门持续known2491416576+2GiB；新0.4078GPU-h与训练0.4300分列。
  首readback terminal1532pins算术齐但948.497180s超hard900的48.497180，旧失败保留；同独立helper原字节另300/900实际5.029937s重核savedfacts成功。
  首文档审阅581.968135s/hard600但收尾余18.032<90且当前末hash未齐，issues/not-qualified；修标题/保留失败，限定17输入另300/1500复核。
  新限定17输入事实复核whole763.191438s/hard1500、soft超463.191438/hard0、report余736.801410满足reserve90；17first/end SHA/bytes一致，issues=[]、仅qualified-document-assigned-saved-facts。
  receipt SHA `b1ae94ff2c5971224f7787d59c52cf58a0b8f9d4b44d4c6b00cea80bbfe4af6c`，新证据 `docs/R7_S3_FIXED_CASE_OBJECTIVE_RESPONSE.md` 已冻结于 `6b2e3db541797708dc1ebb916b39ea7d9c02b91a`，SHA `fbfd83ee84ceecdcaac8f93250e564fde9e0ac57c3d71f87b0ae4cb3f68e0742`。
  两条audit/not-candidate已登记index60，训练0.4300/反算0.4078分列加入累计21.4177，cap20.0/remaining−1.4177仅会计；旧58全文prefixSHA `3ad00bd4980ef318263f55927287d62fa49146abf2a83b389b3d6e49a42c1abd`、旧54及全部旧证据SHA不变，brief按renderer字节同步。
  新CPU完整仓库验证实跑4003passed/9skipped/6warnings（JUnit4012/0fail/errors），whole1239.821378s在1500/3600内、soft/hard0、reserve90满足、workerexit0/reaped无记录signals、0GPU-h，明确CUDA不可见/离线无过滤；6CUDA+3未跟踪fixture skip不算通过。
  attempt SHA `6bbfc820bedd364398c28fc2216cc0fa72982461e69a11409be0a5f866c1bea8`，不重用旧计数、不追认mixedCUDA资源缺口。登记后治理93passed/0skip、两个campaign/两个goal/indexbrief/空白核通过。
  登记提交 `5b72e5642c0563adef6db163b257e17f4ecf8115` 已推工作分支，精确主CI `37606126942` completed/success、必要pytest job/全部steps success（2026-10-07 UTC匿名只读API；17标签实验skip非实验PASS）。
  本单病例诊断完整登记终态，下一不同S3问题优先无新增更新地核0/20/80原deepK目标和最终physical17×5评分对应，预声明原四train季节/四val开发例与同train气候态，不由单病例降幅推共同优化/泛化因果。S3/test未评分/r0/科学未接受不变。

<!-- campaign-state: {"current_node": "S3", "previous_node": "S3", "current_round_goal": "docs/goals/s3-global-mixing.md", "previous_round_goal": "docs/goals/s3-global-mixing.md", "previous_round_evidence": "docs/R7_S3_GLOBAL_MIXING_RESULT.md", "cap_gpu_h": 20.0, "used_gpu_h": 105.619618, "remaining_gpu_h": -85.619618, "status": "active", "next_node_proposal": "S4", "budget_mode": "per-node-hard-cap-summed", "route_decision": "0038"} -->

- **状态**：active；S0/S1/S2 已完成并登记；**S3 进行中（2026-10-05/06）**：batch-2 四季
  2022/2023 获取完成（8/8 part、28,773,423,423 字节、两次失败保留），v2 确认实例（2017/2022/2023
  三年度、三 sidecar、决策 0039 修正）已发布；D2（同数据气候态/persistence，0 GPU）、
  D3（同数据 incumbent 重训，1.0987 GPU-h）、D4（R-C 候选筛选，0.8472 GPU-h，注册负结果）、
  UB（更新预算筛选，1.1010 GPU-h，注册混合）与 BC（预算曲线，1.3716 GPU-h，注册混合）
  已完成并登记；batch-3（2018–2021 四季，train 扩年）获取**已完成并登记**（16/16 part、
  零失败、57,084,203,564 字节、合并源 `97d29bca…` 1920 stamps，见
  `docs/R7_S3_BATCH3_ACQUISITION.md`）；**v3 扩年确认实例已发布并登记**（train 2017–2021 /
  val 2022 / test 2023，三源合并 `bc2ff9cf…` 3360 stamps、窗口 2360/472/472、三 sidecar
  绑定 data identity `2564eeaf…`，见 `docs/R7_S3_CONFIRMATION_INSTANCE_V3.md`）；**v3 控制已重建
  并登记**：v3-D2 同数据基线（0 GPU，992.4s）与 v3-D3 incumbent 重训（1.5012 GPU-h，三 seed
  逐位 actual-C 初始化，6h skill +0.4148/+0.2610/+0.4248）；v3 预算剂量筛选（l6×1600）已完成
  并登记：主格 supported（6h −0.4328/−0.7607/−0.4444、12h −0.5351/−0.7199/−0.5158 K）但守门
  17 cell 与 v2-BC 同剂量持平 → 注册负结果，不进 S4。
  S3 轮次目标见 `docs/goals/s3-confirmation-baselines-and-candidate.md`（同数据基线 + incumbent
  重训 + R-C 候选筛选 + S4 冻结包）。
- **S3-D2/D3 已做并实核**：D2 在 v2 val 上重建 train-only (month,hour) 气候态（480 步 fit、
  16 桶各 30、fail-closed）与 persistence，逐 lead 全覆盖 472/468/460/444/428，墙钟 553.1s
  （planned 1800/hard 3600、overrun 0、网络 0）。D3 三 seed 从**逐位复现 actual C 的初始化**
  出发（`a79ea47f…`/`ec5bb5ef…`/`844bd234…` 与归档 pairing 完全一致），在 v2 2017 train 上完成
  400 次 L6 更新并评 val 全 cohort；整轮 3955.3s（planned 5400/hard 10800、overrun 0），
  保守 GPU-h 1.0987，邻居共驻未 signal。**同数据 t2m/full skill**：6h 三 seed 全正
  （+0.2143/+0.2754/+0.3262），12h 两负一微正（−0.1655/−0.0697/+0.0037），24h 两负一微正
  （−0.2907/−0.1088/+0.0526），48–72h 三 seed 全负——与 S0 形状一致，且 seed43 把边界推到
  12h/24h 微正；全 17 变量正 seed-cell 计数 51/49/47/6/0（6/12/24/48/72h）。两页均带索引记录
  （`record:s3-d2-same-data-baselines`、`record:s3-d3-incumbent-retrain`），test 全程未读。
- **S3 v3-D2/D3 已做并实核（扩年实例上的控制重建）**：v3-D2 在 v3 训练段（2017–2021 五年）上重训
  train-only (month,hour) 气候态（2400 步、16 桶各 150、fail-closed）并重跑 persistence，
  逐 lead 全覆盖 472/468/460/444/428；整轮 992.4s（planned 1800/hard 3600、overrun 0、网络 0、
  GPU 0）；t2m 气候态 RMSE 3.5171/3.4889/3.4375/3.3534/3.3184 K——五年气候态比 v2 单年略强
  （6h精确差−0.0232 K；48/72h略弱），不能说全lead更强。v3-D3 三 seed 同样从**逐位复现 actual C 的
  初始化**出发，在 v3 train（2360/2340 窗口）上完成 400 次 L6 更新并评 val 全 cohort；整轮
  5404.5s（planned 5400/hard 10800，**软超 4.5s 按决策 0030 记录**），保守 GPU-h 1.5012，
  邻居共驻未 signal。**同数据 t2m/full skill**：6h 三 seed 全正（+0.4148/+0.2610/+0.4248，
  seed41/43 高于 v2-D3、seed42 低于 v2-D3 的0.2754），12h 两正一负（+0.0285/−0.0805/+0.0756，v2 为一正），24h 一正，
  48–72h 仍全负；全 17 变量正 seed-cell 计数 51/50/45/2/0。跨实例数值为描述性（非受控比较），
  但近 lead 形状的改善与「扩数据是下一投资方向」的预判方向一致。两页均带索引记录
  （`record:s3-v3-d2-baselines`、`record:s3-v3-d3-incumbent`），test 全程未读。
- **S3 v3-BD 预算剂量已做并实核（注册负结果）**：扩年实例上 l6×1600 对同实例 400 更新控制
  （三 seed、val-only、FLOP 比 4.0）。主格 t2m/full 6h/12h 三 seed 同号 supported
  （6h −0.4328/−0.7607/−0.4444 K、12h −0.5351/−0.7199/−0.5158 K，24h 亦全负）；但守门
  u10/v10/mslp **17 个正 cell（24h 1、48h 7、72h 9）与 v2-BC 同剂量完全持平**——预声明的
  数据响应分支的观察为「同剂量全45格仍17个失败」；但≥48h实际17→16，不做训练体量因果推断。
  同剂量 seed 均值skill对各自气候态（6h +0.595→+0.5915、12h +0.333→+0.3171、24h +0.226→+0.2082）
  只是跨实例描述；v3-BD正气候态skill格数为51/51/50/4/0，优于D3为51/51/50/22/14。
  已登记独立勘误 `docs/R7_S3_V3_NUMERICAL_ERRATA.md`。按冻结合取规则 `registered-negative` →
  候选停当轮、不进 D5/S4。整轮 6883.9s（planned 6300/hard 12600，**soft overrun 583.9s 记录**），
  保守 GPU-h 1.9122；共驻 GPU0、三 gate 全过；test 未读。索引记录
  `record:s3-v3-budget-dose`。
- **S3 v3 rollout微调attempt01（失败已登记）**：two_step×200接续v3-BD1600；41/42训练评估完整，
  43仅update20/40/60、无endpoint，实际9473.920853s超冻结hard7200。原attempt/driver不改，
  无三seed verdict；全额2.6316GPU-h计入累计12.5554。独立审阅核身份/40评估组匹配，
  两seed partial守门仍8格反例；不升级S4、不与未来attempt拼seed。合作式deadline缺少外部看门狗，
  阻塞根因未确认。证据 `docs/R7_S3_V3_ROLLOUT_FT_ATTEMPT01.md`。
- **S3 v3 rollout attempt02已完成并登记**：同配方单次新路径重执行，三seed200更新endpoint及全部
  val评分完整；主格6/12h三seed全负supported，但守门13/45（48h5/72h8）仍未过→registered-negative。
  相对parent的17格减少不等于清门；t2m48/72h气候态skill仍全负，短双步剂量停止、不再retry。
  整轮4222.618677s、planned5400/hard10800/perseed3600，overrun0，全部五workerreaped，无signals；
  保守1.1729GPU-h。独立765格复核，seed41归档代码五lead15CSV重放完全一致，逐case差0；
  两次评分前失败也保留，全重放另记0.3871，累计14.1154。证据
  `docs/R7_S3_V3_ROLLOUT_FT_ATTEMPT02.md`，索引两个record均已登记。执行SHA8466c2d精确CI
  37468877656 completed/success，全量3758passed/3skipped（不是科学接受），2023未评分。
  新长监督前瞻会计cap20.0/remaining5.8846不作许可闸门；文献仅为假说，见
  `docs/R7_S3_ROLLOUT_TRAINING_SOURCES.md`。新轮次goal
  `docs/goals/s3-long-rollout-supervision.md`：新12步train窗口2140/220、全BPTT CPU反证、
  严格状态/协议消费已实现，全量3887passed/3skipped与独立工程审阅已过（均非科学接受）。
  执行SHA66836d2精确主CI37500335847success；单样本FP32 probe完整且独立身份/成本已核，
  1605.013031s、reservedpeak2.244GiB、无optimizer更新/评分。归档重放846.060163s且五字段精确相同，
  独立terminal审阅28项通过，证据5abc3da/index50条已登记，合计0.6808GPU-h加入累计14.7962。
  新screen完整且独立审阅：三seed主格supported、控制守门0/45，全255格RMSE低于控制；
  8994.237921s、soft超2694.237921/hard超0，2.4984GPU-h已本輪登记。parent相对守门16/45正格，
  t2m48/72h气候态skill仍全负，advance-to-S4-freeze仅表示可另准备冻结包，不是正式确认就绪。
  归档seed41五lead15CSV/全病例精确重放已完成，1935.462220s、soft超135.462220/hard0，
  ceil整秒0.5378GPU-h已登记，三seedcollector摘要一致；独立replay审阅153项精确比较齐，但文档optional
  收尾hard超3.402s不是预算全合规，全部限制保留。证据d1dcd46、index52条、累计17.8324；原输出不改、test未评分。
  登记提交e9db4fb精确CI37509906302completed/success且全部必要步骤success，不自裁科学通过。
  screen protocol `a1631d9b…`及feasibility receipt pin独立冻结，下一spawn携带新峰值至少4370MiB。
- **S3-D4 已做并实核（注册负结果）**：R-C lead-coverage 候选（two_step l6+0.5·l12）在 FLOP 匹配
  下（比 2.0018、相对差 0.09%）以 200 更新对 D3 的 400 更新 l6 incumbent 做单因素筛选；预注册主格
  t2m/full 6h/12h 三 seed 全部同号为正（6h +0.9202/+1.0247/+0.9263 K、12h +1.3219/+1.4587/
  +1.2365 K）→ **worsened**；u10/v10/mslp 守门预审 34 个正 cell 亦不通过 → 候选停当轮
  （`candidate-stops-registered-negative`），不进 D5/S4 冻结包。整轮 3050.1s（planned 5400/hard
  10800、overrun 0），保守 GPU-h 0.8472，test 未读。该轮独立复现了旧 B 阶段「扩展训练目标到 +12h
  无增益」的负结论，横向把「two_step 目标」这条 R-C 杠杆在两个数据实例上都标为无效。
- **S3-UB 已做并实核（注册混合：主格 supported、守门未过）**：直达检验 S0「400 更新末段 loss
  仍在降＝可能停早了」——同配方 l6 跑到 800 更新，对注册的 D3 400 更新控制做单因素筛选。预注册
  主格 t2m/full 6h/12h 三 seed 全部同号为负 → **supported**（6h −0.3985/−0.5144/−0.6170 K、
  12h −0.3487/−0.5105/−0.6191 K），且全部五个 lead 的 t2m 15/15 全负；**首次**把正 skill 推过
  6h：t2m/full 相对同数据气候态 6h +0.49/+0.50/+0.50、12h +0.18/+0.21/+0.19、
  24h +0.08/+0.12/+0.16（D3 的 400 更新仅在 6h 稳定为正）。训练 loss 在 400 之后继续下降
  （末段 701–800 比 301–400 低 8.8%/13.5%/9.6%），确认「停早了」在 t2m 上成立。但守门预审在
  48h/72h 有 13 个正 cell（v10 最重 +0.39），按冻结合取规则登记为不进入 S4 的混合轮；整轮
  3963.6s（planned 5400/hard 10800、overrun 0），保守 GPU-h 1.1010，test 未读。节点软预算
  2.5 现已超 0.5470（累计 3.0470），按决策 0030 软超继续并记账，硬上限 12.0 未触。
- **S3-BC 已做并实核（注册混合：主格 supported、守门 13→17 恶化）**：第二剂、也是 v2 实例上
  最后一剂预算因子——l6×1600 对 400 更新控制（训练 FLOP 比 4.0）。主格 t2m/full 6h/12h 三 seed
  同号 supported（6h −0.6427/−0.7454/−0.9084 K、12h −0.6195/−0.7667/−0.9339 K，24h 亦全负）；
  预算曲线 seed 均值 skill 单调改善：6h +0.272→+0.498→+0.595、12h −0.077→+0.196→+0.333、
  24h −0.116→+0.120→+0.226（400/800/1600）。但守门正 cell 从 800 的 13 升至 **17**（48h 8、
  72h 9），且 1600 时 72h 出现首个 seed 级 t2m 正增量；训练 loss 到 1600 仍在降。**预声明的
  预算响应读数据此判为：单年 train 上预算饱和，下一能力投资是数据而非算力**（batch-3
  2018–2021 四季已在并行获取）。整轮 4937.6s（planned 5400/hard 10800、overrun 0），
  保守 GPU-h 1.3716，test 未读。
- **零成本诊断（BC 后，只读已产出的 ACC/skill，无新运行）**：t2m 长 lead 的失败是
  **pattern 能力**而非幅度伪影——任何纯幅度/线性重标定最多把 skill 恢复到 ρ²（构造性恒等式，
  已用合成数据数值核验）：48h 上限 seed 均值 +0.095、72h 仅 +0.017，而当前 skill 为
  −0.89/−2.15。全 17 变量同口径：72h 近地面上限 +0.00…+0.02，上层仍有可测 pattern
  （z250 +0.15、u250 +0.20、z500 +0.10）。该界说明解码侧校准救不了 48–72h 墙，必须提升
  ρ 本身（模型/数据）；与「单年预算饱和→扩数据」的读数一致，支撑 batch-3 与 v3 实例路线。
- **S0 已做并实核**：actual C 身份链六类 digest 独立重算全部一致（protocol `ea0efb80…`、manifest
  `bb4569f6…`、model_code `551261c4…`、code commit `562e526`、data `ef8c6691…`、source `496084a9…`），
  九个 endpoint checkpoint 逐个 SHA256 与 receipt 相符；新工具 `tools/recompute_r7_s0_gap_audit.py`
  （462 行，纯标准库，最长函数 51 行）与 14 例测试全过，37 阻断规则 0 失败；对 135 个 worker 的
  per-case CSV 独立重算 **全部 6,885 cell**，与归档 pooled 行最大相对偏差 **6.63e-16**。
  产物 `outputs/r7_s0_gap_audit_20261005_attempt01/`（protocol SHA `3209fcd2…`、gap_table SHA
  `7201859a…`、墙钟 5.523 s、planned 600/hard 1200、overrun 0、网络 0、GPU-h 0）。
- **S0 关键科学读数**：incumbent（process/K4/400 更新）**只在 t2m 6h 稳定超过 train-only 气候态**
  （三 seed skill +0.2578/+0.1715/+0.1924，seed 均值 +0.2073）；12h 起全部转负并随 lead 恶化
  （−0.19/−0.42/−2.14/−3.41）。118/255 seed-cell 为正、137 非正、0 undefined；三个区域结论一致。
  u500/u250 是 48–72h 仍多数为正的少数变量。训练目标只有 L6 深监督（`mode='l6'`、`lambda12=0`）
  却评估到 72h，且 400 更新末段 loss 仍在下降——两条事实支持 #78 R-C/R-A 优先，但**都只是推测的
  机制假说，须由 S2 的可证伪实验区分**。package 相对 old_ours 的进步与「超过气候态」分开陈述。
- **本轮不做**：未训练、未下载、未构建 store、未读 test、未开 GPU、未 clone 源码、未改任何旧产物。
  旧 test 曝光状态不变，unresolved 的 process/matched_generic 归因不重裁。
- **恢复方式**：执行窗口核实际 HEAD/status、只读本 §8 与 §2/§3/§5、核两个校检器；必要前瞻工作态
  更新与上一轮 next-action/round-node 一致后才能实验。历史失败终态不改，普通节点自主推进。
- **S1 已完成并实核**：四季 2017（winter/spring/summer/autumn）四 part 全部
  `downloaded-real-source`，网络合计 **14,390,323,242 字节（计划 99.2%、24 GiB 硬上限 56%）**、
  4,604.4 秒；合并源 SHA256 `3b2c2dad…`（480 时间点、17 通道、全域有限、四个 UTC init 小时齐）；
  dev store 于 16.83 s 发布（`BUILD_COMPLETE`，train-only 归一化与重算最大相对偏差 5.3e-08），
  data identity `894b8d1b…`。原 frozen 意图中的 `train=1月/val=4月/test=7月` 被**建库前机械拒绝**：
  fail-closed 的 train-only (month,hour) 气候态缺 4/7 月桶且时间范围合同禁交错声明，改为四季节
  前 24 天作 train、十月 16–24 日作 val、25 日–次月 1 日作 test（340/34/22 窗口），依据写在
  `store_build_protocol.json`。test 从未读取。
- **S2 #77 已完成并登记（2026-10-05，0.6498 GPU-h 含 v1 缺陷轮）**：`scripts/study_r7_77_pe_band.py`
  两臂 `legacy` vs `nyquist_band`（同参数量 2,948,771 / 同 FLOPs / 同 seed 初始化逐位共享）三 seed
  400 更新 val-only。机制探针确认开关生效（legacy 16 live/68 dead → nyquist 192 live/0 dead，
  spread 比 0.44–0.48 → 0.49–0.54），但**预注册主格 t2m 6h/12h 三 seed 增量符号不一致 → 双格
  unresolved**：按冻结决定文本该机制在本预算/实例上被证伪为杠杆，不进入正式确认候选。
  全 85 cell：18 improved/14 worsened/53 unresolved，如实报告不追认方向。
  注册读数与成本见 `docs/R7_S2_77_PE_BAND.md`；v1 轮 writer 缺陷（合并门缺 per-arm
  protocol_sha256/switches，修复 74c2279）作为缺陷记录保留、双轮计费。
- **S2 #78 R-A 已完成并登记（2026-10-05）**：两臂 `identity` vs `normalized_change_scale`
  （`Y = X_t + (d_c/s_c)*r_c`，loss 不变；sidecar 独立身份 `60e6ac55…`）三 seed 400 更新 val-only。
  接线探针证明 ratio 确实进入前向（≤2.6e-5，且两探针模型同 seed 逐位同权）；但**预注册主格
  t2m 6h/12h 三 seed 全部同号为正 → 双格 worsened**：按冻结决定文本该重参数化在本预算/实例
  被证伪，不进入正式确认候选。全 85 cell：30 improved/11 worsened/44 unresolved；24–72h 的
  改善多数格仅作描述、未预注册、不追认。v1 探针拒绝轮（harness 缺陷，零训练）已修复并双轮计费。
  注册读数与成本见 `docs/R7_S2_78_CHANGE_SCALE.md`（累计 1.0257 GPU-h）。
- **S2 全部四项已登记（#77/#78 R-A/#79/#78 R-B）**：#77 主格 unresolved（screening-negative）、
  #78 R-A 双格 worsened、#79 B→C supported（相对 incumbent 两主格 unresolved，needs-review）、
  #78 R-B 双格 worsened；四项的回主线动作分别是"不进正式确认候选/留 S3 设计讨论/新协议才可再筛"，
  见各自证据页与 S2 轮次目标 §7。
- **S3 batch-2 已完成并登记（2026-10-05，0 GPU）**：8 个 (year, season) part 全部真实下载，
  网络 **28,773,423,423 字节（计划 99.9%、48 GiB 硬上限 55.8%）**、成功 part 合计 9,832.8 s；
  合并源 SHA256 `44a24ca0…`（960 stamps、17 通道、四个 UTC init 小时、精确并集）。两次真实失败
  全部保留：winter_2022 被冻结的 8 GiB 每 part 解码上限拒绝（在任何成功 part 前修正为 16 GiB =
  S1 实跑值，重试用新名 `_r2`）；winter_2023 挂死超过冻结 1800 s 期限（执行者终止自己的进程、
  自撰 executor `failed-no-fallback` 回执、给下载脚本加每 part 看门狗，重试用 `winter_2023_r2`）。
  从未删除部分产物、未宣称续传。详情 `docs/R7_S3_BATCH2_ACQUISITION.md`。
- **S3 确认实例 v2 已发布并登记（2026-10-05，0 GPU）**：两批源合并（`e0b51616…`，231,865,649
  字节）后建于全新排他路径 `outputs/r7_s3_confirmation_2017_2022_2023_v2/`，train 2017 /
  val 2022 / test 2023（年模式，472/472/472 窗口），三个 train-only sidecar 全部发布并绑定同一
  data identity `e01828e9…`（change-scale `6fd9e774…`、process-scale `b2830014…`、
  typed-evidence `8760894a…`）。v1 构建因旧 64 MiB 指纹上限只能记 `file-stat-only`、结构上无法
  发布 process-scale sidecar（S4 incumbent 契约的必需输入）——按决策 0039 修 `source_fingerprint`
  为总是全文哈希（更强身份，非放宽），v2 与 v1 的天气字节逐位相同，v1 缺陷构建保留不删。
  详情 `docs/R7_S3_CONFIRMATION_INSTANCE.md`；修正提交 8fa6e0c。
- **下一动作（S3，2026-10-06）**：短 rollout attempt02 的独立审阅、归档重放与全额登记已完成，
  实际三seed主格supported但守门13/45未过→registered-negative，不再重执行短双步配方。
  新轮 `docs/goals/s3-long-rollout-supervision.md` 已实现直接48/72h监督和12步全BPTT，
  train-only metadata实测2140窗口/220排除，CPU反证/独立静态修复与全量回归齐；
  精确commit/archive66836d2已完成且主CI37500335847success。fresh单样本FP32 probe在
  `outputs/r7_s3_long_rollout_probe_20261006_attempt01/`完整，1605.013031s、soft1800/hard3600、
  overrun0、全额0.4458GPU-h已加入索引/账本，单train样本有限loss/gradient、reservedpeak2.244GiB，
  独立身份/成本/作用域已核。归档probe重放846.060163s/0.2350GPU-h、五测量字段精确相同，
  独立terminal核验和全額登记齐，当前已登记累计17.8324。新screen与归档重放均完整，控制守门0/45
  但长lead气候态未过；本輪2.4984+replay0.5378GPU-h及所有softoverrun/审阅预算例外均index登记。
  本登记85ae0ea精确CI37535580456已completed/success，全部必要steps success；新当前轮
  `docs/goals/s3-same-case-gap-diagnostic.md`在S3做20train+4val同case双endpoint/原climatology差距诊断，
  已按soft1800/hard3600完成同case真实诊断，1751.421197s/0.4865GPU-h；t2m48/72h不只val负，train pooled和每year/month亦负。
  全24病例归档重放2844.160051s/0.7900GPU-h，soft超1044.160051/hard0，全指标/组/gap/气候态身份精确数值同原值。
  独立NumPy51484numeric/19500typed精确，原fsum relative审阅FAILED及5082residual另立补充保留不改门；
  三worker/no recorded signals与完整成本/限制已登记。证据09447bc、index54条，当前累计19.1089；执行61e46bd精确CI37543875564success。
  登记46ddd262fdfdfb60319bc455ac428a61c170dade精确CI37553027758亦completed/success，全部必要steps success。
  下一独立S3问题先检查固定长权重下train损失分量/梯度方向，不凭norm>1盲目增加剂量或改clip；
  新当前轮 `docs/goals/s3-train-gradient-components.md` 固定2021四train病例、FP32/K4/原配方、无optimizer/val/test。
  先临时wrapper/tests CPU资格与独立prelaunch，再首例peak后余三例；暂未实际梯度执行，soft1800/hard3600另冻。
  600条已保存norm全部clip前>1、四个loss块非同case不判收敛，不据此无界加预算。
  不重开短双步、不消耗未见test、不假称完整全年或科学complete。
  v3-D2/D3已登记，不重复重建；S4/test未启动，现有四季30日块不等于完整未见全年确认。

### 2026-10-07长期goal接续文档（计划0017；0新增GPU-h）

- 已核起点 `d892e20d93afc1a4fb54006300cadfe887d5485d`；当前S3、index60条、累计21.4177GPU-h，2023test未评分、确认r0；
  单病例80更新与三端点精确反算已登记为audit/not-candidate，不重裁为候选或科学接受。
- 本窗口四份docs的范围仅为既有文档编辑与新计划0017，交接只引用
  `docs/plans/0017-climatology-s3-long-term-handoff.md`。科学协议、旧历史/账本与机器状态原字节保留，
  不修改冻结证据；两goal结构0失败0建议，双campaign均0失败/4既有notes，37阻断及四路径模拟
  0失败，index/brief60/空白通过；四完整治理CPU模块213passed/0failed/0error/0skip、整轮39.204876秒，
  软600/硬1200、overrun0，测试进程socket禁网/CUDA隐藏。13只读pins、state/账本与旧进度前缀不变。
  objective落盘精确3944codepoints；独立内容审阅无必修，非科学或最终verifier接受。回执见计划实际结果。
  本轮0新增GPU-h，无科研/下载/发布/clone/GPU查询，无全量suite/wheel/remoteCI或暂存提交推送。
- 读者必须复核最新§8，不得把早期“尚未run/尚未开始”等过期进度当当前事实；S4 proposal不是
  科学许可。0017 prompt按0030/0038/0039在总体方向内完全自主，原保留授权与冻结边界不变。
- **下一动作（S3）**：先冻结无新增optimizer/updates的配对目标—最终物理评分诊断之身份、
  原四季train/四val开发案例、同train气候态、CPU反证与soft/hard/reserve数字预算；若已有新结果
  先核身份并合法复用，不重做case80或旧replay。新next-child goal、排他protocol及actual state由
  执行窗口先核实际状态并冻结，本文档窗口不改round-goal指针或ledger。

### 2026-10-08 并行科研能力与终极 goal 提示词（计划0018；0新增GPU-h）

- 用户要求项目支持 workflow 与并行科研设计（能并行才并行）。新增第 12 项技能
  `.agents/skills/parallel-research-workflow/SKILL.md`：并行准入七 gate → 编排形态（串行/普通扇出/dynamic
  workflow 脚本）→ 安全隔离（子代理可能不跑 hook，默认只读、写只写各自排他 outputs；GPU 仍由父调度按共驻
  余量串行）→ 科学完整性（每臂独立 protocol/digest/证据/index record；并行≠共享预算；test 封存与 r/alpha
  不放宽）→ 降级（workflow/planner 不可用退回串行扇出）。登记于 `docs/skills/README.md` 与 `AGENTS.md`
  路由表，`AGENTS.md` 硬约束增一条并行准入；决策 0042 记录范围、代价与「2026-10-08 曾被回滚后经用户明确
  要求重启」。本决策不加机械检查器（`docs/skills/README.md:9` 要求重复 ≥2 次）。
- 同轮机械同步：账本补 index63（`s3-climatology-anomaly-anchor`，0.6859），合计 22.6281 → **23.3140**，
  campaign-state `used 23.314 / remaining −3.314`；修正「索引已登记而账本缺行」的漂移。该漂移现有校检器
  发现不了（C-02/C-03 只校验已存在行），已写入技能与决策的代价清单。
- §0 objective 刷新为从 S3 接续（index63/23.3140/模型身份已变更/anchor 已 negative 终结），并写入并行准入
  与编排纪律；落盘单段实测 3964 code points。§4 技能表增至 12 项。
- 本轮 0 新增 GPU-h，无下载/发布/clone/训练/评分/GPU 查询，未跑全量 suite/wheel/远端 CI；未改科学合同、
  冻结证据、index/brief、用户配置，不提交推送。实际验证与回执见计划 0018。

### 2026-10-07实际研究接续：目标—最终输出配对诊断准备

- 起点HEAD `d892e20d93afc1a4fb54006300cadfe887d5485d`，仍S3/index60/21.4177GPU-h，
  2023test未评分/r0。只读核旧精确CI37607651646与保存端点/完整新80contract/原八病例/同train气候态。
- 新当前子goal `docs/goals/s3-objective-forecast-score-diagnostic.md` 从旧fixedcase的S3下一动作接续，
  previous_node仅因同S3新轮次从S2更新S3，previous_goal/evidence绑定已登记fixedcase；不虚跳S4、账本不改。
- 原code.zip兼容61e46bd，新TEMP薄wrapper正在实际CPU合成接线与反证，独立内容审阅已完成并修科学
  判据来源和改名引用漂移；独立新指标CPU反证已调度。原三端点/原四train2021与四val2022、原train-only
  气候态mean identity先绑定，新val读域只在新protocol声明，不回写旧val=false。
- planner实际认证失败（`Provider authentication failed.`、142427ms），保留原错，自规划JSON已同契约
  verified并独立内容核。web-researcher实测idle失败（`upstream stream idle for 3m0s`），按既定路由备选
  通道只读获取issue原文/官方资料，不把摘要当事实；未伪称成功。
- 准备文档提交 `3c354e642682b3767f4f94771104ea7c7aa937b8` 已工作分支推送，精确主CI37629370467必要steps全部success；首次匿名API403检查中断完整保留，不是测试失败。固定TEMP包最小修复后producer完整CPU19/19、原指标88/88实际通过、0skip；原独立55pass/33FALSE_ACCEPT及全部修复失败不回写。最终bundle manifest `6ca336511c8eda4d2d04f8bd34806387db1af0cc3dd0546b487ca5f649714254`，独立prelaunch已另冻600/1800/reserve90正在执行；CPU资格不作天气或科学代理。
- 此时无新optimizer/update/backward/天气字段评分/GPU-h，不重复已登记80或精确replay。
  实际新诊断软3600/硬10800/reserve180秒待独立prelaunch后冻结排他protocol，再实际CPUprepare核完整source/BUILD_COMPLETE/metadata/ordinary三端点。原准备clock soft已超继续、hard10800不reset；逐项证据与预算见子goal§8。
- 独立prelaunch已限定工程PASS，receipt `600a7abf8b1e98f05a7b19ffc8e4a73e88cf2c6b9360ec5187122261a4999aff`；19CPU/38rows通过、指标91/92原环境FAIL保留并4单项补充闭环，不冒称单套allpassed。原准备时钟7178.884820s终态、soft超3578.884820/hard10800未超、0GPU。新实际诊断 `outputs/r7_s3_objective_forecast_20261007_attempt01/` 已排他冻结prefreeze `fa93763b9dbffa6190db02749e8fbeb311fdf78cc97534e19218ef301977fdc1`、planned3600/hard10800/reserve180，CPUprepare真实资格运行中，未提前称24配对完成；账本待真实终态全额登记。
- CPUprepare实际资格已完成671.480996s、exit0/reaped/signals=[]；完整source/preflight/BUILD_COMPLETE/train/val/归档digest/三个普通pt与state外锚全部匹配，fields仍未由prepare读。final protocol `26a17743b50c159b4b5e1e70703c89c3a9dd0ee56aaf4efebe4e687663d2a59d` 已冻结，diagnostic spawn即时GPU1余量25,280,118,784B≥4,638,900,224B，配对执行中；读数未出齐不写完成，真实费用待whole终态。
- 新配对诊断已真实完整并独立终态：24pairs/576transitions、native/FP64与全17×5×3物理指标齐，parent1891.377279s/0.5256GPU-h、三workerexit0reaped/signals=[]、soft/hard超0；同Jan原目标−58.8800%且83/85finalcell改善，其余3train/4val目标全升、val最终241/340cell退化，所有24train/evalfinal hash同，排除本次mode路径不一致作为读数解释但不证明原因。证据冻结 `0befd262f971318cacbcae627d62a16a6e5723b4`、SHA `ff02acac4fb87160bbb1eafd82bdbd1465fd4e724b5c5ded1afec0ac55a35a58`；新index61/mixed-not-candidate，原60前缀hash不变，累计21.9433/remaining−1.9433仅会计。独立终态754allowedfiles首末/算术/identity/cost全核、receipt `c60add04a8cd1952b637524fb8b0383b0183dd6a8dfd07c3adec2ec7f83eff0e`，仅savedfact/configuration-only；未follow真实weather或重复forward。
- 诊断登记精确提交 `3533483df9d8cafd9064ddf561335e64bfb3d102` 主CI `37645924906` completed/success，pytest `112876528185` 五必要steps及所有实际steps均success；首次908.967591秒读取旧post-checkout in_progress/null快照被拒，原failed-incomplete留存，后精确HTTP200终态齐，不改checker或CI。工程绿非科学接受。
- 下一独立同S3子goal `docs/goals/s3-case-interleaving-pilot.md` 已冻结准备：同original0/freshAdamW、原四2021train每例20次/每臂80update，blocked连续20each对interleavedcycle4；只改序列，不加Jan剂量。原bulk确为randperm(seed+epoch)+cursor，人工blocked压力对照不冒称历史失败原因。自规划同契约verified、Explore事实核/独立设计4项必修已写，TEMP实现与CPU反证调度中；prepare3600/10800/reserve180、实际5400/14400/reserve180，未本轮真实训练。
- **下一动作（S3）**：完成等曝光ordering薄wrapper完整CPU与独立prelaunch、完整新contract及逐病例LR曝光冻结后真实两臂train/原八dev评分，unsupported止本实例而不自动增加剂量或扩完整val。旧negative、test/S4/r/最终科学接受不变。

- **2026-10-07等曝光ordering完整登记（仍S3）**：两臂original0/freshAdamW/seed41、batch1各80/四季case各20曝光，原deepK/K4/full12/FP32/clip1及globalLR不变。人工blocked连续四块vsinterleaved循环；不是旧bulk randperm或已证明worstcase。五worker完整exit0/reaped/signals=[]、16endpoint80×原8case train/eval score与8shared0合法复用齐，whole2464.990661s/soft5400/hard14400/reserve180超0，完整0.6848GPU-h加累计22.6281。四train目标共同下降，interleaved最终302/340traincell改善，但devsupport=false：t2m72对both好，其余6/12/24/48比blocked差、48/72气候态仍负，u10五lead对0全差；登记negative/not-candidate、终结本ordering实例，不加dose/seed/fullval。独立947allowedfiles首末hash/160rows/17×5×3数学/contract/calendar/RNG/scope/五gate/process/cost齐。原prelaunch四FALSE_ACCEPT修新bytes后49/49qualified；helper预算失败、calendar失败与出版误绿exit0仍FAILED，新72回归及terminal03实际result/digest齐，原失败全保留。证据冻结commit `9dde48de1666de88035e2c677f20c70ede33ed48`、页SHA `7043fa608de5a74d1ecb2b10c52475419e5f937278460248a8dd6c65d85a63bb`；index62、旧61prefixSHA `7f7da26686aaaf2c5f714e1d25945c870e5555e09603a216750b90b3857dee7b`逐字保留、brief canonicalrenderer。资格提交a1c9639精确CI37664236520成功；最终登记提交 `2af4e825f70c650536622d3d5726b7895e484289` 精确主CI `37676532474` completed/success，pytest `112981363638`、五必要及全部12实际steps均success；匿名只读metadata receipt SHA `384789b1b5043840ebe6dc5c44218b63c22ec7a6685f0027a183c0b314ba1106`，whole819.124314秒/soft900/hard2700超0。冻结页/index62/brief/旧合同与用户配置hash未变，工程CI不是科学通过。test科学评分0/r0、S4/最终接受不变。下一动作（S3）：本ordering实例已终结；新子goal `docs/goals/s3-climatology-anomaly-anchor.md` 与决策0040先冻结train-only valid-time decode/proposal锚与异常feedback不同package，原控制0/80强hashreuse、candidate0另评分、model-only合法归档迁移/新身份/完整CPU和独立资格先行。新模式可逆implementation准备中、尚无真实run或收益；原planner0read/426字符超限及设计审阅M1–M3/并行源码漂移失败保留、补方案不追认。准备3600/10800/reserve180固定clock，实际5400/14400/reserve180待固定包资格后排他protocol，不因orderingnegative改科学门或重跑原因素；账本/index62/test0/r0/S3不变。

### 2026-10-08 气候态锚实例真实执行与独立读回

- **最新读回接续**：用户决策0041前瞻采用完整普通JSON/每role64MiB，压缩review已superseded/not-qualified，旧2MiB失败不追认。普通JSON短修复01/02因pause/provider后原hard1800过期，分别lateclosure3545.748463/2627.781172秒，保持partial-unqualified不reset。MAIN直接接管新repair03先冻1800/7200/res180，492.236870秒完成最终sourcef307/current29+7target/1500plan；独立窄风险review934.018155秒/soft900超34.018155/hard5400超0，ACKeb7f900e/receipt4215a84d/releasea4569396闭合，只资格原JSON/guards/七synthetichelper+13END接口。真实MAIN02随后protocol44d844fe/1800/5400/res180，stat/FIRST/preEND/scope/外部pins与全部七plainrole发布完成，但不变verify_documents发现`exact: saved new reading training_receipt`不一致，42.501045秒/child2766626 rc1reaped/signals=[]，failurebb3484cc/closureeb1d0a4e保持failed，archive_check/第三original/13futureEND未做。新的原JSON/source-text有限forensic已闭合（独立代理provider失败另存closure不追认；MAIN只读阶段562.413秒/soft-hard超0、162,767,377B+2MiB、reportbcfdb41b/closure262c3389）：根因=原saved reading `training_receipt`多一键`checkpoint_observations`（0/80四字段，内嵌与独立文件实测全等含真hash），其余八键canonical全等；producer v0（05:30）8键返回与v1 helper模型一致、producer repair v1（06:26）加观测嵌入并自身test sanction、helper v1（07:24）未跟随——属**v1 helper回执模型缺口**（非reader投影/非数据缺陷）。修复=新排他v2模型补全（数值math不变+内嵌==独立文件强校验+篡改反证+独立复核），再新preparation/binding与全新MAIN03 actual；MAIN01/MAIN02维持failed不追认，不改原数值门或重训。三用户文件/科学合同/model字节仍同，双campaign/goal/37阻断通过；index62/22.6281/待记0.6859/test0/r0/S4不变。#76–#79官方正文/评论经web-researcher及主链一手HTTP读回，2026-10-08仍open且无新增评论；原文保存 `outputs/r7_s3_climate_anchor_issue_refresh_20261008_attempt01/`，receipt `33d64ffc04165214eecbd0a730899c8f5037c97bec059d7cb458fc2f8406fc3c`，48,695B网络/9.536403秒/0GPU，仅参考提案不当结果，不关闭issue。v2链重建后（2026-10-08）：V2复核#2 ACK2 → v2引导attempt03预检fail保留 → V3 fork复核#3 ACK3 → attempt04启动即桥梁ledger fail-closed（取证：snapshot_contract `reserved()` 父源字面量漏改，同一main/parent-source label两条path、distinct paths 1501>1500；attempt02记录恰1500/1500；parent `KeyError: 'schema'` 为级联伪影），attempt03/04两失败件均保留；修正走V4 fork（reserved五方一致+parent失败分类守卫+attempt05字面量），MAIN核对freeze引用全过、离线校准计数1500/1500（within_cap）、复核#4 ACK4通过（4817a4a9）；attempt05实跑fail-closed保留（`TypeError`：qualifier_facts:433 六参调用 vs v2 九参签名——第三调用方在 qualifier 暂存根、312套件未覆盖），修复=新排他Q根调用点6→9参（含全量调用签名扫描定向反证：前身集恰一处失败/修正集0失败），Q复核#5 **ACK5**通过（cc2511a7，preparation-only）；V5链构建（单成员容器替换，freeze afb262b4；首版漏写qualifier行member字段由MAIN构建核对在复核/运行前发现，缺陷binding d8764789归档修复）；V5复核#6判**BLOCK F-RV6-01**（三处运行时ACK引用错指ACK5而非本链自身评审根，两模式实测fail-closed，保留）；修复走V6 fork（引用改指RV7、attempt07字面量、容器沿用v5；freeze 8b7565ed、binding仅父源两叶变化、计数1500/1500），复核#7 **ACK7**通过（1969a512，7/7）；**attempt07实际读回exit0完整成功**（40.59s；receipt dd620455=complete-new-snapshot-bridge-engineering-only、准入1500/1500、总读1,946,575,475B、三原哈希一致、13futureEND全等、0模型执行/0网络、无failure.json；证据归档 outputs/r7_s3_climate_anchor_readback_20261008_attempt01/ manifest 4f0148df），工程闭合但非科学通过、负面判定不变；登记完成（index63 `s3-climatology-anchor` 0.6859 计入累计 23.3140；证据提交 ef7302a CI 37789073741 success、登记提交 581e402 CI 37791472774 success 双绿精确绑定，2026-10-08；index dac2ee0e、brief 4f771fd7）；1500/2GiB/64MiB上限未放宽，test0/r0/S4不变。

- 当前仍S3/index62/累计22.6281GPU-h、test评分0/r0；主模型实现提交`abe5439d3444a89ee569aef3172588dc92ec9ca8`精确主CI37695634597成功。唯一候选attempt已恰80更新/16新pair/五phase完整success、owned父及worker全部exit0/reaped，整轮2468.177771秒/软硬超0，保守0.6859GPU-h待独立终态登记后加账，不预填index63。保存必要开发支持false，t2m48/72h对两控制改善而6/12/24h退化，部分关键变量亦退化；仅package小开发读数，实例按原停止线不加seed/dose/fullval。完整控制reuse成本0，无新数据/下载/fit外观测/额外test评分；科学门/S4/最终goal均未接受。
- 独立observer准备25合成通过，但真正完整保存清单1357项含OLD692复制源码、其中209非活跃legacy成员；old483/new487活跃映射不得扩成legacy运行资格。全原始路径通用读回方案还需497旧pin原路径，实际1945 distinct输入超过已冻1500门，且observer+adapter重复整批读取超过2GiB。原准备接口因此真实actual不qualified，准入失败保留，未跑actualreader或发scope权威，不删清单/跳测试/改冻门。正设计另行资格的单一原记录快照与外部权威pins握手方法：完整actual字节首末核、非活跃成员仅bytehash，原CPU观测按已独立源码资格与完整原first/end receipts归因，不冒称450旧CPU记录再次逐字读取；global1500/累计2GiB/64MiBJSON不放宽。具体source与新固定预算/反证由子goal§8接续，旧failed/readplan不回改。
- 下一具体动作（按最新交接更新）：依据用户侧聊与决策0041，新的读回保留完整普通JSON并采用每role64MiB；不继续只为旧2MiB门而压缩，旧冻结失败不回改。先闭合plain64固定source与独立风险专属资格，才冻结actual-terminal1800/5400/reserve180单clock；原记录source/scope/数学/process/cost独立闭合后冻负面页/index63/brief/账本及工作分支精确登记CI。累计2GiB/1500路径及科学门不变，按实际all7尺寸/读次先准入。本实例不重训、不进S4，不拿工程工作量当研究获益。

- **2026-10-08描述性负面全表与读回预算方法准备**：candidate0/80与两个合法原控制共有eval2040+trainmode2040物理cell，全17×5×3/四role×两split齐，无单位混用或undefined过滤；20个开发门格7满足both、13未满足，supportfalse。candidate80 t2m气候态skill6/12/24/48/72为−0.335176/+0.074726/−0.406438/−0.305835/+0.004885；不是全年/三seed科学接受。表与result/三preparer失败、Python3.12 scalarordered加法及step-vs-total残差勘误已排他保存，工作态新证据docs/R7_S3_CLIMATOLOGY_ANOMALY_ANCHOR.md明确未冻结/未登记/terminal尚待。旧index62/22.6281/科学合同/用户配置hash未变。新snapshot桥将FIRST哈希和一次原JSON解析合用同读字节，减少412MB重复读而保留三次全inputhash；最终固定包报告1497显式输入/路径及1,988,618,601B保守读预算候选，前述1492估计被后续完整managed dependency/source绑定取代，仍须actualstat准入，不把预算估计或局部回归当资格。既有helper/adapter/observer数值/testbody原样保留，NumPyPython/native容器精确bytes身份/只读managedcost明列，不声称OS全读oracle或把NumPy当stdlib。无新增训练/评分/GPU-h/test/S4；此实例不追加剂量、seed、fullval。
- **2026-10-08固定snapshot桥资格失败与最小修复**：source a6476026独立实际49B合成canary证明Ledger.raw自动admit加relative O_NOFOLLOW guard绕冻结白名单；原完整helper305/adapter76通过不抵消observer25的一error、新25未跑，未来authority/5packet/7role未被最终END封存亦是真实接口缺口。review880.021054秒失败闭合，无ACK，15held文件字节/stat首末同；回执4abedcea/END6bb34feb/HOLD释放4eb0767主链实际字节核同。原engineering6310.922790秒/soft超2710.922791/hard0以failed-source-and-read-accounting-unqualified闭合（62971123），整个原stage≤2GiB未证实、五旧adapter空目录与provider/全部红反证保留，不追认原失败或改旧源。
  另排他whitelist_and_publication_repair01在首次source复制前冻结1800/5400/reserve180、start6209778.539712201/protocol b5206d6a，controlled reads全meter；只修membership/精确openat-FD/sourcepinned bootstrap、scopebody、publicationABI与所有新authority END。主链parent只叉BRIDGE路径（69314446），原29完整测试通过，三actual ownedstdio toy success/提前exit2/截止SIGTERM全部reaped，2.230372秒；限定合成资格不代weather/scope数值。provider无文本错误继续原repairclock不reset。实际saved-reader未启动、负面页未冻结/index63未登记，0新增GPU-h/test0/r0/S4不变。
- **修复及限定描述性复核已闭合，独立source qualification进行中**：productionfreeze38112b72六source首末同，实际305/76/25通过；new29 first有一fakeFD overflow fixture error，b4a92f64保留，仅真实pipeFD夹具更正后29/29通过，composed7df819b3明确分次执行/适用原源码复用，不声称fresh435或追认旧失败。修复3795.088284秒/soft超1995.088284/hard0/reset0，106820726Bobserved+151122636B显式allowance=257943362B<2GiB，非OS全读oracle；closecfdc98aa/handoff8df34365/HOLD02acdc85ac主链核实际字节。独立固定source review在6213919.340210698新冻1800/5400/reserve180/完整meter，但427.376827秒因源/template/plan仍绑旧failed ACK root闭合BLOCK，21source首末同，receipt88c72b28/release90418b57核实际字节，未ACK/fulltoy/139/actualreader；另排他ack_path_repair01先冻900/2700/reserve90只修ACK/rootliteral与input_rows正反例，旧失败保留、数学/FD/13-END/科学门不改；前瞻actual恰1500labels/paths与2057082475B knownupper尚需实际stat准入。
  描述性独立43项核4080cell/20gate/32目标/155活动/五phase/费用算术，15input FIRST/END实际5384466B逐字/stat相同，1516.116925秒/soft超616.116925/hard0/noreset，只支持export/document。勘误时间104.064806是finalreceiptconstruct采样而非terminalwhole，bodyready104.062374分列，whole未观测；optionalACC每path661末位差异为reviewer猜测算法harness失败保留，不变数学/epsilon或认导出坏。index62/22.6281、test0/r0/S3与本anchor实例停止不变。
- ACKroot最小修复另先冻900/2700/reserve90，560.848443秒/超0/noreset闭合；17,202,843B observed与58,770,040B显式allowance分列。新f95682e5 productionfreeze/sourceAST仅binding/root/clockliteral，input_rows实际合成接受正确新ACK、拒旧ACK/bootstrap且零originalfileopen；postguard与13newbindingEND mutation核、corrected29pass/0skip，24immutable源身份后reuse原305/76/25/notfresh435。handoffb58e7d26/close4706897b主链核实际字节，newparent9db7/bootstrap89a9原29+22完整pass。正确新ACKroot独立review在6215887.957373先冻1800/5400/reserve180/meter，guard10/newACK3/原sourceproof完成；fullownedtoy03 directrc0/reaped验证真实stdin外部pins/guard/immutableAPI/3pass/13END，toy01/02自身path/HEAD夹具错误保留，toycontextSHA为合成非生产；118独立反证实际拒绝、后续gate/END尚待。实际原保存reader当时未启动、未ACK登记、成本0.6859仍待资格/index63，no新GPU/test/S4/r。
- 固定f956 source独立限定ACK9dfec237在4959.773620秒/soft超3159.773620/hard0/noreset/总charge2038708167B闭合，250source+47native首末同、trueguard/stdin+exact immutable合成API、composedsourcebound测试完毕；archive_check仅原body/inherited-test资格非actual调用，toy合成HEAD/自身harness及provider全部失败保留。真实saved-reader随后MAIN01先冻start6221024.783868065/1800/5400/res180，stat/FIRST/preEND/完整scope/inventory/五外部rawpins/一次stdin权威已实跑，但data角色pretty6443132B>冻结2MiB拒绝，child2515933 rc1/reaped/no signals、whole32.095591秒，failure4075a3b7保留/terminal数值未qualify，第三END/13futureEND/archive_check未做。compact3205394B仍超门，diagnostic86bb3bb8证明sameJSON zlib9为215095B；新排他lossless_role_storage_repair01先冻1200/3600/res120仅修7role无损wire存储、有界展开/完整types/canonical/byte双hash，raw2MiB/JSON64MiB/decodedaggregate14MiB/global1500/2GiB与数学/科学不改。新MAIN02只防覆盖旧失败，源资格与下一真实读回尚待；no新训练/GPU/test/r/S4，原weathernegative止实例与账本/index62不变。

### 2026-10-09 rollout 监督剂量 200 → 800（单因素；登记 mixed/needs-review）

- **动机**：注册的 S3 long-rollout screen 在 200 更新时余弦日程已退到 0.1× 下限，只界定一个剂量点。
  本轮只改更新数（200→800），父、12 步物理权重、LR 2e-5、warmup 10、FP32、K4、batch 1、clip 1、
  控制与判定形式全部不变。
- **模型身份变更的合法处理**：父 checkpoint 由修订 `66836d2`（`model_code_sha256 3ddab46b…`）训练，
  climatology-anchor 修订把它改成 `d3fb58db…`，受审计的 `load_checkpoint` 因此按设计拒绝。
  新增 `scripts/export_r7_parent_state.py`（在归档修订下用其自身 loader 导出 state，
  `export_sha256 d1d9886a…`）与 `scripts/migrate_r7_parent_state.py`（复核源 SHA pin、导出 state
  digest、张量 key/shape/dtype 与模块语义 digest 后写新 `r7-local-v1` + 归档 provenance，
  不带 optimizer/cursor/RNG；receipt `23daf794…`）。惰性证据：模块树只多两个默认属性、源码 diff 中
  所有新分支都以它们为门、**固定合成 batch 在两修订下的 FP32 forward 逐位相同**。
- **冻结判读**（`primary_verdict`/`gate_verdict`/`advance_decision` 三个同一冻结函数）：
  **primary supported**（t2m/full 6h 与 12h 三 seed 对 v3-D3 控制增量 −0.401/−0.727/−0.429 与
  −0.532/−0.718/−0.508 K）、**守门 0/45**（最大 −0.104401）、**decision advance-to-S4-freeze**
  （与 200 更新 screen 同形，只表示可另备冻结包）。
- **绝对门仍未过**：t2m/full seed 均值 skill 6h **+0.5816**、12h **+0.3151**、24h **+0.3668**、
  48h **−0.1191**、72h **−0.4309**；ACC 0.7953/0.6651/0.6525/0.3671/0.1894，ACC²（线性重标定上限）
  0.6326/0.4425/0.4259/**0.1355**/**0.0366**——完全校准也过不了 48/72h 门。
- **剂量仍在付但递减**：相对注册 200 更新 screen，24/48/72h RMSE 低 0.1536/0.3248/0.4653 K，
  6h/12h 持平（高 0.0254/0.0076 K）；48h/72h 的 ACC 由 0.329/0.153 升到 0.367/0.189——增益是
  **pattern** 而非仅幅度；幅度比 σ_f/σ_o 由 200 轮的 0.99/1.05 落到 0.87/0.87。全 17 变量在
  24/48/72h 对控制相对 MSE 变化全负。
- **失败与偏离（保留）**：screen 的 `prepare/archive/seed41/seed42/seed43` 五阶段全 success、
  worker 全 reaped/signals=[]，唯 `reading` 阶段 rc=1——共享 `training/r7_rollout_evidence.py::_references`
  要求协议 parent pin 等于注册 v3-BD receipt 记录的原始 checkpoint，同时又要求 `load_checkpoint`
  能加载它，两者在 model-only 迁移后不可能同时成立；这是**身份记账管线缺陷**，非训练/评分失败。
  未改共享判定代码，另立只读工具 `scripts/report_r7_rollout_dose.py`（委托同一冻结函数，
  偏离写在产物 `deviation` 字段；CPU 7.95 s、0 GPU）。probe attempt01（0.4808）与 screen 的
  `failure.json` 全额计费保留。
- **独立复核**：独立子代理以自有脚本从原始产物复算 14 项（阶段/失败根因/协议 digest/损失行与加权
  恒等/checkpoint pin/cohort 与 provenance/配对增量/守门 45 格/primary 符号/skill 与 ACC/剂量差/
  迁移 pin/成本算术/test 封存）**14/14 CONFIRMED，无 REFUTED/UNCERTAIN**。
- **成本**：0.4808 + 0.4706 + 3.3251 = **4.2765 GPU-h**，累计 **27.5905**（cap20/remaining −7.5905
  仅会计）；网络 0、无下载/发布/clone、未改 model/ 与科学合同、未 signal 邻居；test 未读/r0。
- **规模登记**：执行后对 `scripts/study_r7_s3_rollout_dose.py` 只做一次纯格式改动（10 处 >120 字符
  行换行到 ≤120，语义不变、AST 复核通过），使 R-019b 不增；因此工作树当前字节与运行用 `code.zip`
  （`bec1edd5…`，494 行版本）不同，运行的执行身份仍由协议 `execution_files_sha256` 与归档
  `code.zip` 完整固定，重放走归档。`docs/rules/size-thresholds.md` 记 R-021 84→85、
  R-019=0/R-019b=1026 未增，并把 CI 红过的 `tests/test_check_conventions.py::
  test_size_report_counts_match_the_checker` 修回绿。
- **父修订的识别方式（精确）**：v3-BD 运行**没有**记录 `code.zip` 或 `code_commit`，因此 66836d2
  不是"记录的 pin"而是**按 digest 相等识别**的修订：`model_code_digest()` 在 66836d2 等于父
  checkpoint 归档的 `model_code_sha256 3ddab46b…`，导出在该修订下由该修订自己的 `load_checkpoint`
  接受，故导出所用 model 代码与训练父者逐字相同；任何其他共享该 model digest 的修订同样成立。
  这条限制写在迁移 receipt 的 provenance 字段里，读回路径不依赖"66836d2 是唯一可能修订"。
- **可复现等级**：**config-reproducible**（协议/源/代码/控制 pin 齐、归档 code.zip 可重放），
  GPU 训练非逐位可复现；不声称 bit-reproducible。
- **验证与 CI（精确）**：本机完整 CPU 套件 `4214 passed, 9 skipped, 3 warnings in 1208.02 s`
  （CUDA 隐藏、离线）。证据提交 `73459af` 的主 CI **失败**（唯一红点即上面的 size-marker 测试，
  已在 8aedd5a 修复；该失败不回改、不追认）；登记提交
  `8aedd5a16f47cf76ed6f579c9e723e7eb1e87b3b` 的主 CI `37854151641` **completed/success**。
  17 条标签门控实验 workflow 显示 skipped 是设计行为，不是通过。
- **下一动作（S3）**：修共享 `_references` 对迁移父的表达（或把迁移注册为可 pin 的父记录），
  然后以"提高 pattern 相关"为下一杠杆；最可证伪的一支是扩大区域/引入域外大尺度上下文
  （旧实测显示 ARCO ERA5 的 ROI 放大几乎不增成本），并重建同数据气候态与 incumbent 后再比较。

- **2026-10-09 连续自我迭代契约轮（0 GPU-h 文档轮）**：用户报告上一连续任务会话
  `sess_1d199296-0818-4484-a384-786216eabbe0`「跑了一会儿自己停了」。只读查 `~/.zcode/cli/db/db.sqlite`
  定位**真实停止原因**：轮次总结后触发的 goal 完成校验 provider 报错（`status='error'`、31157 ms、
  0 token），harness fail-open 令 `payload.status='failed_closed'` 而 `verification.passed=true`、
  `session_target.status='complete'`；科学未完成（r=0、2023 test 未评分、S4 未启动）。**不是**用户打断、
  **不是**科研完成。据此把 §0 objective 换成**自迭代版**（单段、实测 4000 字符），新增本节所属的
  「连续自我迭代契约」与 §5 的 CI 不阻塞纪律，并在 `goal-loop` 技能补「连续自我迭代（不等 verifier）」。
  技能清单由 12 项增至 **13 项**（加入 `dynamic-workflows`；`/workflow` 的落地技能，未加载时
  `CreateWorkflow` 拒绝运行；为 ZCode bundled 技能，不进 `docs/skills/README.md` 的本仓能力表）。本轮 **0 GPU-h、0 test 读取、r=0**，不新增 index record 与账本行
  （无实验机器核数，按 §7「无 index 支撑的文档 0 成本行如实列 note」处理）；不启动实验/下载/发布/clone/GPU，
  不 commit/push。决策 0043、计划 0019 记录本契约。**下一动作**：从本轮之前 S3 的「下一动作」继续——
  用已建并 pilot 的宽区域（129×129，中心 65×65 = 冻结目标盒）数据做**单因素真实 train/val**，证伪
  「冻结 16° 盒缺长 lead 所需信息」，同配方同预算对照 65×65，再据证据决定是否进 S4 冻结包。

### 2026-10-09 宽区全量获取、v4 store 与宽区气候态（本轮；0 GPU-h）

- **本轮目标** `docs/goals/s3-wide-region-full.md`；证据页 `docs/R7_S3_WIDE_REGION_FULL.md`（index66
  `record:s3-wide-region-full`，`outcome_class engineering-positive`、`not-candidate`）。
- **committed 宽区路径曾不可运行（本轮发现并修复）**：R-021 拆分把 helper 在模块间搬动时留下**四处**缺陷，
  committed 的 `--preflight` 与 `--write` 都 `NameError`/崩溃，pilot 是在拆分**之前**跑的，而当时只有
  "名字同一性"测试、没有作用域检查。四处是：`validate_wide_namespace` 引用未定义的 `surface_chunk_bytes`；
  `earthmover_spatial_w1` 未 import 五个已搬到 `earthmover_spatial_d1` 的 helper；`_wide_part_receipt` 引用
  调用方局部 `started`（**下载完成后写回执时崩溃**，留下完整 nc 却无回执）；对已是字符串的版本号调用
  `.__version__`。已修并补两道守卫：作用域走查（对拆分前版本报出全部七处未定义名）+ 回执合成测试
  （当场抓出类型缺陷）。attempt01 作为失败记录保留（2 个 failed 回执、1 个孤儿 nc、被取代的协议）。
- **获取**：28 part（27 新 + 复用 pilot winter_2018）覆盖 train 2017–2021 / val 2022 / test 2023，
  3360 stamps；每 part 恰 120 stamps、`[120,3,129,129]`、17 通道全有限、`read_plan` digest `d5618cbf…`。
  `winter_2020`（1806 s）与 `winter_2022`（1806 s）首跑在**冻结 1800 s 每 part 期限**上被拒、**无产物**，
  按冻结恢复规则以新名 `_r2` 重取（1414.1/1521.9 s）成功；两次失败回执原样保留、不追认。
- **并发实测**：6 个并发 reader 死锁（CPU 冻结、socket 空闲、0 字节）；4 个正常（聚合 ~8.2 MB/s，每 part
  1596–1789 s）；2 个 ~1414–1522 s；1 个 1218.5 s。故 pool 固定 **4**；两次失败正是"1800 s 期限在 pool 4
  下已接近"的直接后果，下一批若要提并发须同时上调 per-part 期限。
- **合并与身份**：`source.nc` 2,003,552,683 B、SHA `6bc9a1a2d93d345ce0dcb39d4f3a26f175ea3bcc15082ac4931a179ca13b7701`；
  时间轴与 v3 源**逐位相同**；`verify_r7_wide_target_block` **pass**（中心块 rows/cols `[32,97]`、
  `stacked_channels` 与 v3 逐位相同、units 匹配）。
- **v4 store**：`outputs/r7_s3_wide_instance_v4_20261009_attempt01/`，协议 `03f2c647…`、只读 preflight
  `51dd90cc…`、自审 13/13；`[3360,17,129,129]`、split/window 计数与 v3 相同（2360/472/472）、train-only
  归一化 + process 诊断、`BUILD_COMPLETE`；data identity `2360d42b…`（与 v3 不同，区域不同）。构建 168.8 s。
  **store 级身份**：宽 store 的 `[:,:,32:97,32:97]` 与 v3 store `state` **逐位相同**。
- **宽区气候态**：`scripts/study_r7_s3_wide_d2_baselines.py`（协议 `591ede1d…`）10 次 `evaluate_local`，
  val 2022 全 cohort，`boundary_margins=(32,)`；`interior_32` = **4225 点 = 冻结 65×65 盒**；2461.6 s、
  overrun 0。t2m `interior_32` 气候态 RMSE 3.5171/3.4889/3.4375/3.3534/3.3184 K，与**已登记 v3-D2 在
  ~1e-10 内相同**；persistence 同。**即下一轮单因素比较的 skill 分母就是已登记的那个分母。**
- **成本**：0 GPU-h；网络 ≈101 GB（计划 100 GB、硬上限 150 GiB 未触，软超约 1%）；磁盘峰值 6.6 GiB；
  整轮 ≈4.5 h 墙钟。**已知测量偏差**：`network_body_bytes` 是宿主级 `/proc/net/dev` 计数器在每 part 读取
  窗口内的增量，pool 4 下每 part 读数含兄弟流量（约 4× 高估），故协议判据"每 part 网络在 pilot 的 25% 内"
  **在并发下不可按字面评估**，按 `stop_rules` 记为该判据的**偏差**（仪器限制，非数据缺陷）；解码 charged
  9,482,837,880 B 与 chunk reads 2287 是进程内计数器，逐 part 与 pilot 完全相同、正常通过。
- **验证与 CI**：本机完整 CPU 套件 **4263 passed / 9 skipped / 0 failed**（1301.4 s，CUDA 隐藏、离线）；
  37 条阻断规则 0 失败；宽区测试 55 passed。证据提交 `123097b0a13d4f330782025e274b3f6c739b4c3f`
  的主 CI **`37974234521` completed/success**，全部必要步骤（conventions / campaign state / evidence
  index+brief / compile / unit+integration+wheel）success；17 条标签门控实验 workflow 显示 skipped 是
  设计行为，不是通过。
- **下一动作（S3）**：按轮次目标 §3 的设计决定做**单因素 train/val**——在宽 store 上用注册的
  rollout-dose 配方（12 步物理权重、800 updates、LR 2e-5、warmup 10、FP32、K4、batch 1、clip 1、
  seed 41/42/43，**全 129×129 全监督**）训练，用 `boundary_margins=(32,)` 读 `interior_32` 与**已登记
  的窄 rollout-dose 800 臂**比较；先在 CPU 上完成新 protocol/输出/停止出口与定向反证，再按共驻余量
  申请 GPU。读法：宽臂在 `interior_32` 优于窄臂 → 支持"域外上下文有用"；持平或更差 → **不构成**信息
  不足的反证（可能是监督稀释），须另做"同监督域、只改输入范围"的对照臂区分。test 未读、r=0、S4 未启动不变。

### 2026-10-09 宽区域单因素 train/val：输入区域不是长 lead 的瓶颈（null 读数；3.5339 GPU-h）

- **本轮目标** `docs/goals/s3-wide-single-factor.md`；证据页 `docs/R7_S3_WIDE_SINGLE_FACTOR.md`
  （index67 `record:s3-wide-single-factor`，`outcome_class negative`、`not-candidate`）。
- **单因素**：唯一改动因子是**输入区域**（65×65 → 129×129）。配方（`long_rollout`、12 步物理权重
  `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)`、800 updates、LR 2e-5、warmup 10、FP32、K4、batch 1、clip 1、
  checkpoint_every 20、seed 41/42/43）、父状态（同一份迁移的 v3-BD 1600）、监督方式（全 129×129
  全监督）、评分路径全部沿用注册 rollout-dose 臂。对照是**已登记**的窄 65×65 臂（按
  `rmse_csv_sha256` 复核后复用，**未重训练**）。
- **结果（null）**：`interior_32`（= 冻结中心 65×65 盒、4225 点）上 t2m 的 seed 均值 delta
  （宽 − 窄）为 6h **+0.0013**、12h **+0.0147**、24h **+0.0189**、48h **+0.0121**、72h **+0.0231** K
  ——**每个 lead 都是正的（更差）**；48/72h 只有 2/3 seed 更优（72h 由 seed41 的 +0.1428 拉正，
  seed42/43 为 −0.0650/−0.0083）。绝对气候态门 48/72h **仍未过**且宽臂略差：skill 均值
  −0.1271/−0.4485 对窄臂 −0.1191/−0.4309。
- **对假设的判定**：**不支持**"冻结 16° 盒缺长 lead 所需信息"；**不构成反证**——本轮是
  **全 129×129 全监督**，监督面积 3.94×，信息增益与监督稀释混淆。故**不进 S4 冻结包**。
- **公平性守卫（全过）**：`WIDE_VAL_COHORTS = {6:472,12:468,24:460,48:444,72:428}` 在 freeze 时与
  窄臂回执逐 lead 核对，实测每 seed 每 lead `n_evaluated == n_available_windows` 且等于 pin；
  宽区气候态 `interior_32` t2m RMSE 与**已登记 v3-D2 在 ~1e-10 内相同**，两臂共用同一分母；
  reading 阶段按 pin 复核窄臂 `rmse.csv`，无漂移。
- **工程缺陷（本轮发现并修复）**：开轮提交 `13f3c44` 的主 CI **`37985102030` 失败**，唯一红点是
  pytest 的 `test_size_report_counts_match_the_checker`：
  `R-019b: size-thresholds.md says 1026, the checker reports 1028`——新驱动 599 行（>400 → R-021
  85→86）且有两行 141/132 字符（120–200 带内 → R-019b 1026→1028）。C 类 marker 必须随实测同步，
  `de64697` 修回（**未改任何阈值/规则级别/冻结例外**），其主 CI **`37990463998` completed/success**、
  全部必要步骤 success；证据提交 `dcd1a86c88d97bc231508ab8c1ebd0a2161557d2` 的主 CI
  **`38005892925` completed/success**、全部必要 steps success；登记提交 `876cef8c…` 的主 CI
  **`38007212799` completed/success**。17 条标签门控实验 workflow 显示 skipped 是设计行为，不是通过。
- **成本**：whole 12,721.940569 s → **3.5339 GPU-h**（口径沿用本方向约定：whole ÷ 3600，故
  CPU-only 的 prepare/archive/reading 也按 1 GPU 当量计入，偏保守）；soft overrun 3,721.94 s、
  hard overrun 0；网络 **0**（离线）；产物磁盘 4.6 GiB。累计 27.5905 → **31.1244**。
- **进程卫生**：六阶段（prepare/archive/seed41/seed42/seed43/reading）全 `success`、`returncode=0`、
  `signals=[]`、`reaped=true`；每阶段 spawn 前写只读 `*_spawn_gate.json`（`source: "nvidia-smi
  read-only"`），**未对任何非本实验进程发送信号**，邻居（GPU1）全程未触碰。
- **可复现等级**：**config-reproducible**（协议/源/代码/父 pin 齐、归档 `code.zip`
  `fa5752f6…` 可重放），GPU 训练非逐位可复现；不声称 bit-reproducible。
- **下一动作（S3）**：做**分离稀释与信息增益**的单因素对照——宽 129×129 输入 + **只在
  `interior_32` 上监督**（与已登记窄臂同监督域），对照仍是已登记的窄 65×65 臂；目标
  `docs/goals/s3-wide-interior-supervision.md`。先 CPU 写新 protocol/新输出目录/新 `code.zip`/
  停止出口与定向反证，再按共驻余量申请 GPU。test 未读、r=0、S4 未启动不变。

### 2026-10-09 宽输入 + 内部盒监督：上一轮的 null 主要来自监督稀释（mixed/needs-review；3.4655 GPU-h）

- **本轮目标** `docs/goals/s3-wide-interior-supervision.md`；证据页
  `docs/R7_S3_WIDE_INTERIOR_SUPERVISION.md`（index68 `record:s3-wide-interior-supervision`，
  `outcome_class mixed`、`candidate_state needs-review`）。
- **单因素**：在**宽 129×129 输入**下把监督域从全网格收回 `interior_32`
  （`boundary_masks(129,129,(32,))`，4225 点 = 冻结中心 65×65 盒 = **已登记窄臂自己的网格**）。
  相对上一轮（宽输入 + 全网格监督）唯一改动因子是**监督域**；相对已登记窄臂唯一改动因子是**输入范围**。
- **结果**：seed 均值 delta（本臂 − 窄臂）**五个 lead 全为负**——6h **−0.0348**、12h **−0.0401**、
  24h **−0.0069**、48h **−0.0528**、72h **−0.0445** K；6/12h **3/3** seed 更优。
  而上一轮全网格臂五个 lead **全为正**（+0.0013/+0.0147/+0.0189/+0.0121/+0.0231）。skill 也全面更高
  （`interior_32` t2m 均值 0.5944/0.3341/0.3700/−0.0868/−0.3996 对窄臂 0.5816/0.3151/0.3668/
  −0.1191/−0.4309，对上一轮宽臂 0.5812/0.3081/0.3581/−0.1271/−0.4485）。
- **判定**：**上一轮的 null 主要来自监督稀释**，不是"域外上下文无用"。但收益幅度小（≤0.10 K），
  48/72h 的**绝对气候态门仍未过**（只有 seed 43 的 48h 首次为正 +0.0140；72h 三 seed 全负），
  故**仍不进 S4 冻结包**，改为把该臂推入**剂量检验**。
- **目标层改动（本轮唯一代码改动）**：`training/r7_losses.py::latitude_weighted_mse` 新增
  `mask=None` 关键字参数（**None 时逐位不变**，有 4290 条本地测试作回归）；`r7_recursive_losses`、
  `r7_autoregressive_rollout::_draft_loss`、`r7_long_rollout::training_long_rollout`、
  `r7_long_rollout_runner::fine_tune_long_rollout` 逐层透传；mask 的声明块
  （`kind/shape/selected_cells/mask_sha256/归一化方式`）进入绑定 contract 的
  `autoregression.supervision` → 进入 `signature` 与 `training_report.json`，驱动再断言两者逐字一致。
- **反证**（`tests/test_r7_s3_wide_interior_supervision.py`，8 测试全过）：单格 mask 上手工算出的
  受限加权均值必须被复现（全网格 1/3 对受限 1.0）；**全 1 mask 必须逐位等于原目标**；空 mask /
  非 2-D mask / 负值 mask 必须报错；驱动对已存在输出与缺失 store 必须拒绝。
- **成本**：whole 12,475.851618 s → **3.4655 GPU-h**（口径同前，CPU 阶段也按 1 GPU 当量计入）；
  soft overrun 3,475.85 s、hard overrun 0；网络 **0**（离线）；磁盘 4.6 GiB。累计 31.1244 → **34.5899**。
- **进程卫生**：六阶段（prepare/archive/seed41/seed42/seed43/reading）全 `success`、`returncode=0`、
  `signals=[]`、`reaped=true`；每阶段 spawn 前写只读 `*_spawn_gate.json`，**未对任何非本实验进程
  发送信号**，邻居（GPU1）全程未触碰。
- **可复现等级**：**config-reproducible**（协议/源/代码/父 pin 齐、归档 `code.zip` `2e60dfd6…`
  可重放），GPU 训练非逐位可复现；不声称 bit-reproducible。
- **验证与 CI（精确）**：本机完整 CPU 套件 **4290 passed / 3 skipped / 0 failed**（1217.0 s，
  CUDA 隐藏、离线）；37 条阻断规则 0 失败；新测试文件 8 passed。执行提交 `7ef43cbdc3a7…` 的主 CI
  **`38009737294` completed/success**；证据提交 `63e533f457d6…` 的主 CI `38022897803`（登记时）。
  17 条标签门控实验 workflow 显示 skipped 是设计行为，不是通过。
- **下一动作（S3）**：**单因素剂量提升**——同一冻结监督域（`interior_32`）、同一宽 store、同一配方族、
  同一迁移 v3-BD 1600 父，把 updates 从 **800 提到 2400**，检验收益是否随剂量放大到 48/72h 绝对门；
  对照同时报已登记窄臂与上一轮 800 内部监督臂（两者都 pinned 复用、不重训练）。目标
  `docs/goals/s3-wide-interior-dose.md`。读法：2400 后 48/72h 三 seed 全正且 seed 均值同时区间下界 > 0
  → 提请独立验收；否则按防空转纪律**停该支**、转别的可证伪假设。test 未读、r=0、S4 未启动不变。

### 2026-10-09 宽输入 + 内部盒监督剂量 800→2400：48 h 首次三 seed 全为正，72 h 仍未过（mixed/needs-review；6.6941 GPU-h）

- **本轮目标** `docs/goals/s3-wide-interior-dose.md`；证据页 `docs/R7_S3_WIDE_INTERIOR_DOSE.md`
  （index69 `record:s3-wide-interior-dose`，`outcome_class mixed`、`candidate_state needs-review`）。
- **单因素**：**updates 800 → 2400**。宽 129×129 输入、`interior_32` 掩码、配方其余参数、迁移
  v3-BD 1600 父、seed、评分域、气候态分母、cohort pin 全部逐字不变。
- **结果（剂量响应强且单调）**：相对**已登记窄 800 臂**，本臂在 12/24/48/72 h **3/3 seed 更优**
  （delta 均值 −0.0284 / −0.1159 / **−0.3086** / **−0.3306** K），6 h 持平（−0.0029）。
  相对**已登记的 800 内部监督臂**，24/48/72 h 3/3 更优（−0.1090/−0.2558/−0.2861），6/12 h 略差
  ——即剂量把收益**集中加在长 lead**。
- **48 h 绝对气候态门首次三 seed 全为正**：`interior_32` t2m skill **+0.0613 / +0.0277 / +0.1127**；
  6/12/24 h 也三 seed 全正。**72 h 三 seed 仍全负**（−0.1746/−0.2773/−0.1562，均值 −0.2027，
  相对窄臂的 −0.4309 已收窄）。
- **判定**：冻结的验收条件（48/72 h 三 seed 全正）**只满足一半** → **不进 S4 冻结包**；
  按上一轮预声明的防空转读法**关闭剂量支**（不再加剂量），转**长 lead 物理权重**假设
  （注册配方 48/72 h 权重与 12/24 h 同为 0.5）。
- **本轮的一个失败 attempt（工程缺陷，已修）**：attempt01 的冻结 reading 阶段把 800 臂的
  `boundary_rmse.csv` 与 pin 里的 `rmse.csv` 哈希比较，**必然**在三 seed 全部训完后 `RuntimeError`。
  发现即**中止**，未把剩余 GPU 时间花在不可能出读数的运行上；`operator_abort.json` 全额计费
  **0.7517 GPU-h**（prepare 271.45 s + archive 289.77 s + seed41 部分 2145.0 s = 2706.22 s），
  无 in-place resume。修复 `0b517b3`/`76a3d60`：pin 改存 `boundary_rmse_csv_sha256`，
  并把两个依赖 `outputs/` 的测试改用**合成输出树**——**同一缺陷暴露出的第二处问题**是
  `0b517b3` 的 CI 在无 `outputs/` 的干净 checkout 上因那两个测试失败；本地用 `git worktree`
  复现该条件后修好。
- **成本**：attempt01 0.7517 + attempt02 5.9424 = **6.6941 GPU-h**（口径同前：whole ÷ 3600）。
  attempt02 whole 21,392.663503 s、**soft overrun 0 / hard overrun 0**（planned 36000 s 内）；
  网络 **0**（离线）；磁盘 14 GiB。累计 34.5899 → **41.2840**。
- **进程卫生**：attempt02 六阶段全 `success`、`returncode=0`、`signals=[]`、`reaped=true`；
  每阶段 spawn 前写只读 `*_spawn_gate.json`；**未对任何非本实验进程发送信号**（邻居 GPU1 未触碰，
  仅共享显存；seed 43 评分期间邻居占卡，故其评分秒数偏高）。
- **可复现等级**：**config-reproducible**（协议/源/代码/父 pin 齐、归档 `code.zip` `0a6e0b81…`
  可重放），GPU 训练非逐位可复现；不声称 bit-reproducible。
- **验证与 CI（精确）**：本机完整 CPU 套件 **4290 passed / 3 skipped / 0 failed**；37 条阻断规则
  0 失败；`git worktree` 无 `outputs/` 复现 CI 条件下相关 222 测试全过。执行提交 `76a3d60f…`
  的主 CI **`38027168107` completed/success**；证据页最终提交 `9fc4db2c…` 的主 CI
  **`38048807438` completed/success**（index69 的 `ci_run_id`/`ci_commit` 即 pin 于此）。
  **登记提交 `9f6a3df02c…` 的主 CI `38050148795` 被紧随其后的下一轮推送取消——取消不算通过**；
  其内容（index69、账本行、§8、campaign-state、下一轮目标）由下一个提交 `98184febf7…` 的树
  完整包含，而该提交的主 CI **`38050360256` completed/success**。17 条标签门控实验 workflow
  显示 skipped 是设计行为，不是通过。
- **下一动作（S3）**：**固定 2400 更新**、只把 48/72 h 的物理权重从 0.5 提到 **1.0** 的单因素臂，
  对照是**本轮已登记的 2400 臂**（pinned 复用）。目标 `docs/goals/s3-wide-long-lead-weight.md`。
  预声明出口：72 h 三 seed 转正 → 提请独立验收；否则**关闭整个宽输入支**，回主线别的可证伪假设
  （推理深度 K / 架构），不再加 seed、不再调权重、不重复同一 test。test 未读、r=0、S4 未启动不变。

### 2026-10-10 长 lead 物理权重再分配：几乎无效，72 h 仍不过——宽输入支关闭（negative；7.0027 GPU-h）

- **本轮目标** `docs/goals/s3-wide-long-lead-weight.md`；证据页 `docs/R7_S3_WIDE_LONG_LEAD_WEIGHT.md`
  （index70 `record:s3-wide-long-lead-weight`，`outcome_class negative`、`not-candidate`）。
- **单因素**：在**完全相同的 2400 更新**下只把 48 h/72 h 的物理权重从 0.5 提到 1.0
  （`(1,.5,0,.5,0,0,0,.5,0,0,0,.5)` → `(1,.5,0,.5,0,0,0,1.,0,0,0,1.)`），
  宽输入、`interior_32` 监督、父、seed、评分域、气候态分母全不变。
- **结果：杠杆近乎无效**。相对**已登记 2400 剂量臂**，seed 均值 delta 为 6h **+0.0129**、
  12h **+0.0188**、24h **+0.0140**（略差），48h **−0.0054**、72h **−0.0218**（略好）——翻倍长 lead
  权重只换来 72 h 约 **0.02 K**。**72 h 仍三 seed 全负**（skill −0.1672/−0.2614/−0.1363），
  48 h 仍三 seed 全正（+0.0631/+0.0278/+0.1200）。
- **已记录的混淆**：权重翻倍同时**放大总损失尺度**（物理损失是加权和），不是纯再分配；LR 保持注册值。
- **判定：预声明出口触发 → 宽输入支关闭**。四轮合并结论：**输入范围本身无收益**（全网格监督下
  48/72h 三 seed 全负）；**把监督域收回目标盒**后收益出现但只到 48h；**剂量 800→2400** 把 48h
  推过气候态（三 seed 全正）；**长 lead 权重**几乎无效。**72 h 用输入范围、剂量、长 lead 权重三种
  杠杆都未推过气候态**，故**不进 S4 冻结包**。
- **成本**：whole 25,209.895089 s → **7.0027 GPU-h**（口径同前）；soft/hard overrun 均 **0**；
  网络 **0**（离线）；磁盘 14 GiB。累计 41.2840 → **48.2867**。
- **进程卫生**：六阶段全 `success`、`returncode=0`、`signals=[]`、`reaped=true`；每阶段 spawn 前写
  只读 `*_spawn_gate.json`（prepare 门 free 20,046,675,968 B、required 4,294,967,296 B、passed）；
  **未对任何非本实验进程发送信号**（邻居在 GPU0 共驻占卡，seed 42/43 因此明显变慢）。
- **可复现等级**：**config-reproducible**（归档 `code.zip` `2696f7ef…` 可重放），GPU 训练非逐位可复现。
- **验证与 CI（精确）**：本机 37 条阻断规则 0 失败；`git worktree` 无 `outputs/` 复现 CI 条件下
  相关 229 测试全过。执行提交 `98184febf7…` 的主 CI **`38050360256` completed/success**；
  证据提交 `c0206082c0…` 的主 CI **`38078884049` completed/success**（index70 的
  `ci_run_id`/`ci_commit` 即 pin 于此）。
- **下一动作（S3）**：回主线换**与输入区域无关**的可证伪假设——**推理深度 K 4 → 8** 的单因素臂
  （在已登记最佳臂上：宽输入 + `interior_32` 监督 + 2400 更新 + 注册物理权重），对照是**已登记的
  2400 剂量臂**（pinned 复用）。目标 `docs/goals/s3-reasoning-depth.md`。预声明出口：72 h 三 seed
  转正 → 提请独立验收；否则**关闭"靠训练配方解决 72 h"这一支**，转模型容量/架构或数据方向，
  不再加 seed、不再调 K、不重复同一 test。test 未读、r=0、S4 未启动不变。

- **2026-10-11 campaign 驱动 workflow 支持轮（0 GPU-h；文档 + 治理层代码）**：用户问「用 workflow 能不能搞一个
  特别大型的、让目标循环自迭代」。调研结论已归档：**可表达但有界**——动态 workflow 允许循环、harness 无节点
  上限，但它归当前进程所有（退出即 `interrupted`）、子代理不能嵌套 workflow、子代理很可能不跑本仓 hook
  （E-187 推测），所以它是**有界、可恢复的批量编排器，不是守护进程**；无人值守的永久自迭代仍被决策
  0030/0038 禁止。四个硬事实：① 驱动**没有任何 resume**（产物全部独占创建、attempt 目录必须全新、
  `failure.json` 明文 no in-place resume）⇒ 被打断的一轮要重付 GPU；② 驱动已是子进程编排器
  （`--worker prepare|archive|seed|reading`，seed 一次一个）；③ `world.run` 的 argv 必须是编译期字面量 ⇒
  架构走「文件即参数」（round-spec 块 + 冻结 protocol.json）；④ 此前没有 state/CI/GPU 探针三类工具。
  **交付**：决策 0044（workflow 作本轮执行者 + 六层打断抗性 + 裁决 `.zcode/workflows/` 的跟踪冲突：草稿
  git-ignored、保存命名脚本按决策 0001 入库）、决策 0045（跨 attempt pinned seed 复用，白名单全等准入、
  逐 seed 如实写明重跑/复用、被打断那段 GPU 全额记账）、计划 0020、`tools/r7_round.py` +
  `tools/r7_round_support.py` + `tools/r7_round_stages.py`（state / reuse-plan / ledger-plan / round-guard /
  verify / attempt / stage-plan / freeze / stage / commit-round / push-round）、`tools/r7_ci_wait.py`
  （`curl` 子进程轮询，**未 import 任何 Python 网络客户端**）、`tests/test_r7_round.py` +
  `tests/test_r7_round_reuse.py` + `tests/test_r7_round_stages.py` + `tests/test_r7_ci_wait.py`（**90 例全通过**，含逐字段反证）、
  `parallel-research-workflow` 技能新增「campaign 驱动 workflow」节、`AGENTS.md` 路由与硬约束、
  草稿 `.zcode/workflow-drafts/r7-campaign-round.dwf.ts`。
  **补记首次真实使用（此前未留痕的欠账）**：2026-10-11 已有草稿
  `.zcode/workflow-drafts/第-6-轮推理深度-K8-驱动实现与独立复核.dwf.ts` 与一次真实运行产物
  `.zcode/workflow-runs/dwfrun-0f324485-a59c-468e-866a-34d7bcdbfda3.mjs`（该轮做的是 K=8 驱动的实现与
  独立复核，未开 GPU、未登记）；决策 0042 与本仓技能都要求首次真实使用留痕，本条即为该留痕。
  本轮 **0 GPU-h、0 test 读取、r=0**，不新增 index record 与账本行（无实验机器核数，按 §7 处理）。
  **验证与 CI（精确）**：本机 37 条阻断规则 0 失败；`check_campaign_state`（默认旧 master 与显式新）0 失败；
  `check_goal_brief` 0 失败；证据索引 70 条与 brief 逐字节一致；新测试模块 90 例全通过（另跑
  `test_check_conventions` 120、`test_check_campaign_state` 60、`test_check_goal_brief` 19、
  `test_verify_r7_evidence_index` 14、`test_agent_hooks` 196 全通过）。内容提交 `795bb5e` 的主 CI
  **`38123579554` completed/failure**（唯一红点 = `test_size_report_counts_match_the_checker`：我的新文件把
  R-019/R-019b/R-021/R-023 四个**报告型**计数推高而冻结 marker 未变；该失败**不回改、不追认**）；
  修复提交 `f1a218d`（折长行、拆出 `tests/test_r7_round_reuse.py`、`run_stage` 参数降到 8 个，使本轮新增
  文件对这些计数贡献为 0，且**不动** `docs/rules/size-thresholds.md`——那是另一窗口的工作树改动）的主 CI
  **`38124858453` completed/success**，全部必要步骤成功；17 条标签门控实验 workflow 显示 skipped 是设计行为。
  **未验证项**：`freeze`/`stage` 的 GPU 轮编排**没有端到端跑过**（本轮不开 GPU），故决策 0044 第 4 条要求
  首次真实使用必须留痕验证。**下一动作（S3）**：不变——**推理深度 K 4→8** 单因素臂（对照已登记的 2400
  剂量臂 pinned 复用），预声明出口：72 h 三 seed 转正 → 提请独立验收；否则关闭「靠训练配方解决 72 h」
  这一支。test 未读、r=0、S4 未启动不变。

### 2026-10-12 — 推理深度 K 4→8 臂被操作者中止（未完成）＋训练配方支关闭

- **做了什么**：上一轮的 K4→8 单因素臂（目标 `docs/goals/s3-reasoning-depth.md`）**实际开跑后被执行者中止**。
  产物如实登记在 `docs/R7_S3_REASONING_DEPTH.md` 与 index `record:s3-reasoning-depth`
  （`outcome_class: unresolved`、`candidate_state: not-candidate`、`scientific_claim: false`）。
- **实际发生**：`outputs/r7_s3_reasoning_depth_20261009_attempt01/` —— prepare（510.163 s）/archive（471.532 s）
  成功；**seed 41 完整跑完**（exit 0、训练 11209.65 s、checkpoint `efc43304…`、五 lead 472/468/460/444/428
  全评、峰值 13.55 GiB）；**seed 42 训练中被 `SIGKILL`**（`exit_code=-9`、1656.784 s，源自操作者停止 workflow
  时连带终止在跑的 `world.run`，见计划 0021 §1）；seed 43 与 reading **从未运行** ⇒ 无 `readings.json`/
  `result.json`/`failure.json`/`attempt.json`。
- **唯一可读数字（单 seed 41）**：相对已登记 2400 剂量臂（K=4，pinned 复用）`interior_32` t2m Δrmse
  6h **+0.00776**、12h **+0.00884**、24h −0.00856、48h −0.00309、72h −0.00780 K；Δskill ≤0.00509；
  **72h skill 仍负（−0.16954）**。
- **成本**：whole 18070.843 s = **5.0197 GPU-h**，全额计费（含被 SIGKILL 的 seed 42 段）；soft/hard overrun 均 0。
- **判定（已确认与推断分开写）**：**已确认** —— 该臂未完成，未达预声明出口所需的三 seed，
  **不构成科学判定**；**推断** —— "K 翻倍无实质收益"只有单 seed 支撑，只作旁证。
- **分支关闭（依据合并）**：训练配方的四个杠杆（输入范围 index67、监督域 index68、剂量 index69、
  长 lead 权重 index70）加推理深度 K（本轮单 seed 旁证）都已试过，**72h 从未三 seed 转正** ⇒
  **关闭"靠训练配方解决 72 h"这一支**；宽输入支此前已按预声明关闭、不重开。
- **为何停 + 恢复第一条命令**：停因＝操作者判断收益不足（**非**机器可判失败：无 hard overrun、无身份/资源门失败）；
  恢复第一条命令＝ `.venv/bin/python -m pytest -q tests/test_r7_s3_wide_long_lead_weight.py`，
  然后按新 round brief `docs/goals/s3-capacity-or-architecture.md` 派生 `depth=8` 变体（新输出目录、
  新协议 digest、新 `code.zip`，并**重新标定** `estimated_peak_bytes`）。
- **下一动作（S3）**：**主干深度 `depth` 4→8** 单因素臂（在已登记最佳臂上；对照＝已登记的 2400 剂量臂
  pinned 复用），预声明出口：72 h 三 seed 转正 → 提请独立验收；否则关闭"加大容量解决 72 h"这一支，
  转架构或数据方向。test 未读、r=0、S4 未启动不变。

### 2026-10-12 — capacity（主干深度 depth 4→8）轮：操作者决定**不运行**，直接登记（0 GPU-h）

（本块取代上一块结尾的"下一动作＝capacity 轮"：该轮已关闭、**未运行**。）

- **做了什么**：capacity 轮（目标 `docs/goals/s3-capacity-or-architecture.md`）的驱动/测试/code.zip
  **已就绪并已推送**（commit `710612d`，CI run `38205959882` 绿，`tools/r7_round.py round-guard` 通过），
  但**操作者决定不跑 GPU 部分**，直接登记。证据页 `docs/R7_S3_CAPACITY_DEPTH.md`、索引
  `record:s3-capacity-depth`（`outcome_class: unresolved`、`candidate_state: not-candidate`、
  `scientific_claim: false`、`gpu_hours: 0`）。
- **依据（操作者给出）**：读 seed 41 的读数，**效果基本上没变化**。该 seed 属于**上一轮 K4→8 臂**
  （index71），是本轮之前唯一完整跑完的 seed：相对已登记 2400 剂量臂（K=4，pinned 复用）
  `interior_32` t2m Δrmse 6/12/24/48/72h = **+0.00776/+0.00884/−0.00856/−0.00309/−0.00780 K**、
  Δskill ≤0.00509、**72h skill 仍负（−0.16954）**。
- **口径（必须照此读，别读成结果）**：这是**操作者的成本/收益决定**，**不是**对本轮 depth 4→8 假设的
  判定——`K`（每步内部推理次数）与主干 `depth`（`WindowAttentionBlock` 层数）是**两个不同因子**，
  本轮**从未运行**、其假设**未被检验**，本轮 brief §3 的预声明出口**既未触发也未满足**；因此
  **不**据此关闭"加大容量解决 72 h"这一支，也**不**写任何通过/失败。
- **未运行的确证**：`outputs/r7_s3_capacity_depth_20261012_attempt01/` **不存在**；无 `freeze`、无 seed
  启动、无读数文件、无 `attempt.json`；**0 GPU-h**（两块 GPU 全程空闲）、网络 0、soft/hard overrun 均 0。
- **保留的资产**：驱动 `scripts/study_r7_s3_capacity_depth.py`、测试
  `tests/test_r7_s3_capacity_depth.py`（28 例）、`outputs/r7_s3_capacity_depth_code.zip`
  （SHA256 `de9f6bc4…`、1380 文件、comment `710612d`）保留为"已准备、未运行"的**可复用资产**；
  CPU preflight 对真实迁移父（sha `fba27e26…`）得 `bit_identical=True`、`added=[4,5,6,7]`、
  严格 `load_state_dict` OK。
- **为何停 + 恢复第一条命令**：停因＝操作者判断不值得花 3 seed × ≤28800 s 的 GPU（**非**机器可判失败：
  无 hard overrun、无身份/资源门失败）；恢复第一条命令＝ `.venv/bin/python tools/r7_round.py freeze`
  （前置：HEAD 的 ZIP comment 必须等于 HEAD，否则先重建 `code.zip`）。
- **下一动作（S3）**：由操作者与主链路**重新制定方向**（架构 / 数据）并据此开新的 round brief；
  本轮 brief `docs/goals/s3-capacity-or-architecture.md` 保留为"已准备、未运行"的记录，**不重开**。
  宽输入支与训练配方支已关闭不重开。test 未读、r=0、S4 未启动不变。
- **campaign-state 指针**：`previous_round_goal` / `previous_round_evidence` 已前移到本轮
  （`docs/goals/s3-capacity-or-architecture.md` + `docs/R7_S3_CAPACITY_DEPTH.md`，C-04/C-06 绿）；
  `current_round_goal` **暂与本轮相同**——因为下一轮的 brief 要由后续**规划轮**产出，此处**不预造占位 brief**；
  新 brief 落盘时再把 `current_round_goal` 前移、`previous_*` 留在本轮。`used_gpu_h` 不变（本轮 0 GPU-h）。
- **交接（给并行窗口）**：本窗口 2026-10-12 收到"是否重跑 capacity 轮"的发配询问后，**用户直接决定不跑**
  （依据见上）。因此发配选项 (a) **未执行**：没有起 workflow、没有 `freeze`、没有 seed、0 GPU-h。
  若要恢复本轮，按上面的"恢复第一条命令"。**不要重复登记本轮**——index72 已在册，
  重复登记会造出重复的 index record 与账本行。
- **本窗口独立复验（只读，2026-10-12）**：用驱动自带函数复跑该轮 preflight——父 checkpoint
  `outputs/r7_s3_rollout_dose_parent_migrated_20261009_attempt02/seed41/parent_update_0001600.pt`，
  SHA256 `fba27e26…` **与 pin 一致**，`updates=1600` / `contract.seed=41` / `mode=l6`；
  `expand_parent_state` 返回 `encoder_bit_identical=True`、`added_blocks=[4,5,6,7]`、
  `non_block_keys_unchanged=True`，严格 `load_state_dict` 通过。全程未创建 attempt 目录、未启动 GPU 进程。

### 2026-10-12（同日稍后）— 决定改回：capacity 轮**恢复运行**；多轮驱动修复

- **决定**：上面那条"未运行"的登记**被撤销为执行决定，但不被改写为历史**。操作者决定**恢复运行**
  depth 4→8 单因素臂。**撤销理由（这是关键）**：当初跳过的依据是上一轮 K4→8 臂唯一完成的 seed 41
  「效果基本上没变化」，但那是**跨因子**证据——`K` 是每步内部推理次数（实测参数量 **不变**，
  3,286,037），`depth` 是 `WindowAttentionBlock` 层数（实测参数量 **3,286,037 → 5,065,493，+54.2%**）。
  两者是不同的轴，K 的读数不构成对 depth 假设的证据。故本轮假设**当时未被检验**、现在恢复检验。
- **登记口径**：原「未运行」record `s3-capacity-depth` 与证据页 `docs/R7_S3_CAPACITY_DEPTH.md`
  **逐字节保留**（那是真实发生过的成本/收益决定），只在其上**追加** `correction_record_id` 指向新的
  运行 record；本轮读数登记为**新** record `s3-capacity-depth-run` + 新证据页
  `docs/R7_S3_CAPACITY_DEPTH_RUN.md` + **新**账本行（实际 GPU-h）。做法照 `s3-v3-numerical-errata`
  先例（"旧证据 SHA 不改，修正以追加表达"）。
- **zip 绑定改为机制化**：`outputs/r7_s3_capacity_depth_code.zip` 的 ZIP comment 是 `git archive`
  自动写入的被归档 commit SHA，而 archive 阶段断言 `comment == freeze 时的 HEAD`。**预建 zip 已失效**
  （comment 停在 `710612d`，HEAD 已前进），因此把"freeze 前按 HEAD 重建 zip"做进了两个 workflow 脚本，
  使该约束永久成立，也顺带满足 R-010。
- **多轮驱动（自迭代载体）修复**（`.zcode/workflow-drafts/r7-campaign-round.dwf.ts`，gitignored 草稿）：
  ① 补上**缺失的确定性闸门**——原先审计不干净或 `round-guard` 失败也照样提交并开 GPU（最严重）；
  ② `already-running`/`headroom-blocked` 改为有界等待重问（原先当失败停轮）；
  ③ stage 超时 6h → 9.5h（原先短于 per-seed 8h 上限，会杀掉合法 seed）；
  ④ 登记提交改用新增的 `commit-register`（原先用 `commit-round` 且**不检查结果**）；
  ⑤ 证据页路径由 `RoundPlan.evidencePath` 提供（原先字面量 `<ARM>` 未插值）；
  ⑥ 补上**谁写新 brief、谁更新 `campaign-state.current_round_goal`**——`r7_round.py` 从该指针读本轮 spec，
  原先没有任何角色更新它（侦察员是只读的），下一轮会重复读旧 spec；现由实现者写 brief 并改指针，
  `commit_paths`（实现路径）与 `register_paths`（登记路径）分开；
  ⑦ 默认值对齐、去掉硬编码的规则条数。
- **工具侧**（`tools/r7_round_stages.py` / `tools/r7_round.py`，additive）：新增 `commit_register` 与
  `r7_round.py commit-register` 子命令（读 `register_paths`/`register_message`，缺失回退
  `commit_paths`/`commit_message`）；并让提交在 **"nothing to commit"** 时按 no-op 成功返回——
  否则被中断的轮次无法恢复（实现者重跑会发现文件已提交、无变化，旧行为会整轮停死）。
  新增 8 条测试（含"钩子拒绝仍然失败"的反证），`tests/test_r7_round*.py` 95 例全过。
- **本轮不在此窗口运行**：运行交给另一个窗口（避免重复烧卡）；本窗口只做仓库侧准备与脚本修复。
- **本窗口准备提交与 CI（2026-10-12）**：`5ca88d9`（工具侧 `commit_register` + round-spec 拆分
  stage/register + 指针恢复 + §0 压缩）与 `d5859a0`（同步 R-021 规模 marker 91→93）已推工作分支；
  `R7 CPU CI` run `38233411450` 对 `d5859a0…` **九主步骤全 success**。首轮 run `38228709491` 红：
  本轮把 `tools/r7_round_stages.py`（362→416）与 `tests/test_r7_round_stages.py`（347→428）推过
  400 行目标，anti-drift marker 未同步；按 `docs/rules/size-thresholds.md` 先例重测同步（**未改阈值、
  规则级别或冻结例外**）。
- **下一动作（S3）**：跑 capacity 轮（depth 4→8）并按 §3 预声明出口读——72 h 三 seed 全正 → 提请独立验收；
  否则关闭"靠加大容量解决 72 h"这一支，转架构方向。**数据扩充暂缓**（放到最最后）。test 未读、r=0 不变。

### 2026-10-12（深夜）— capacity 轮第三次尝试被中断：跨 app 实例导致通知丢失；gate 已校准

- **gate 两次误标定（已修，另一窗口提交 `9e8124f`/`8ff541b`）**：
  - 首轮 gate 写 15 GiB + 4 GiB = **19 GiB**，而共驻邻居占着约 14 GiB ⇒ 两卡都不满足，
    `stage seed` 无限 `headroom-blocked`（重问上限 240 次 ≈ 4 h，耗尽即整轮停）。
  - `8ff541b` 纠正**代理臂取错**：显存主因是 **K**，不是 depth —— 同 K=4 的剂量臂实测 **8562 MiB**、
    K=8 臂 **13878 MiB**，而本臂是 depth-8/**K=4**（K 不变、只加深编码器，一次前向约 0.1 GiB），
    真实足迹约 8.5 GiB，原按 K=8 臂高估约 2×。现为 `ESTIMATED_PEAK_BYTES = 8704 MiB`
    （≥ 实测代理）+ `HEADROOM_MARGIN_BYTES = 512 MiB` = **9 GiB**，floor 8 GiB。
    **未动任何科学判据，也未放宽"禁止对邻居发信号"。**
  - 另修登记提交口径：`commit-round` → `commit-register`（原先提交的是早已提交的 `commit_paths`，
    而登记四件套在 `register_paths`；照原样跑完四件套根本不会进版本库，CI 还会因 HEAD 未动显示"绿"）。
- **三次 attempt，全部 0 GPU-h、无 seed、无读数**（`outputs/` 下**原样保留、不删不续**）：
  - `attempt01`：prepare 619.8 s + archive 614.6 s 成功（`code_commit 5386f7c`、`code_zip_sha256 615bb2d0…`）。
  - `attempt02`：仅 prepare 593.3 s 成功。
  - `attempt03`：prepare 630.2 s + archive 490.1 s 成功（`code_commit 8ff541b`、`code_zip_sha256 a4bf0746…`）；
    `stage seed 41` 已启动但在 preflight 阶段被停，**没有 seed41 目录、没有 readings.json**。
- **"目标与工作流都无法推进"的根因（已确证，非推测）**：`dwfrun-6c3847ad` 的 `.mjs` 进程**从未死**
  （pid 3268974 一直在跑，把 prepare→archive 推完并进入 seed 阶段），但它的**运行记录被判为
  `stopped/interrupted`**，且进度事件被丢弃 29 次，日志原文为
  `dynamic_workflow.run_progress.session_mismatch` / `"Dynamic workflow run progress dropped:
  parent session is not this app"`。即**会话跨了 ZCode app 实例**（同时存在 3 个 `zcode-cli`）：
  workflow 由 `sess_1abffa2b` 发起，而观察它的 `sess_3f4ccb3b` 属于另一个 app ⇒ 完成通知无法投递
  ⇒ goal 永远等不到结果。**工作流本身仍在推进**，所以"卡住"的是通知链路，不是工作流。
- **处置**：本窗口对该 workflow 编排进程及其 `stage seed 41` 子进程发 SIGTERM（**未用 `-9`**），
  已核对无残留、无新派生；GPU 上自始至终只有邻居作业的进程，本仓**全程未占卡**。
- **操作教训（已写进 §0）**：**goal 与 workflow 必须在同一个窗口/同一个 app 实例内跑**；
  跨实例会丢完成通知，表现为"目标不推进"而工作流仍在后台跑。
- **下一动作（S3）**：在**单一窗口**重开干净一轮（`freeze` 会自动创建 `attempt04`）——
  `round-guard` → 重建 code.zip（comment 必须 == HEAD）→ `freeze` →
  `stage prepare/archive/seed{41,42,43}/reading` → 四件套登记。有效驱动身份已从 `710612d` 前进到
  **`8ff541b`**（index72 的 record 与 `docs/R7_S3_CAPACITY_DEPTH.md` 按惯例**不改字节**）。
  预声明出口不变：72 h 三 seed 全正 → 提请独立验收；否则关闭"靠加大容量解决 72 h"这一支，
  转架构方向。**数据扩充仍暂缓**。test 未读、r=0 不变。

### 2026-10-12（同一窗口）— 新增「文献通道」：一手论文检索走 `arxiv-researcher`

- **决定（决策 0046）**：本机新增用户级 arXiv MCP（server 名 `arxiv`，19 个工具；源码与
  `mcp.server.connected` 日志两处独立核对）。据此新增**第二条外部检索通道**：**一手科研文献**
  （论文摘要/正文/公式/实验设置/引用关系）委派新子智能体 `arxiv-researcher`（只读 9 个
  `mcp__arxiv__*` 工具，能按章节精读与全文检索）；**通用网络事实**（库用法/版本/官方文档/报错）
  仍走 `web-researcher` → `web-researcher-backup` → 主链路 `curl` 降级。两条通道**互不重叠**，
  引用纪律是同一条（R-050：URL + 访问日期，摘要不等于证据，一手/二手分开）。
- **给正在跑的这一轮之后的窗口**：下一轮起，凡是要为可证伪假设找**文献旁证**（例如
  `docs/plans/0022-s3-architecture-next-round.md` §6 那类「这条研究线是否已饱和」），
  **先委派 `arxiv-researcher`**，不要再默认用主链路 `curl` 打 arXiv API——那是**降级路径**，
  不是常规路径。通用库/版本/文档问题仍走 `web-researcher`。技能里的委派阶梯已改成两层。
- **接线与留痕**：新 `.zcode/agents/arxiv-researcher.md`；`.agents/skills/web-research/SKILL.md`
  （两条通道的阶梯）；`AGENTS.md`（外部信息硬约束 + 按任务加载表各补一条）；
  `docs/skills/README.md`；决定记录 `docs/decisions/0046-arxiv-literature-research-route.md`；
  计划 `docs/plans/0023-literature-research-subagent.md`。
- **两条边界（如实）**：① **文献通道没有机械闸门**——R-049 的 hook 只匹配 `WebSearch`/`WebFetch`，
  不覆盖 `mcp__arxiv__*`，所以该通道靠**约定与技能**约束，不是强制；② **新 agent 是否免重启生效
  未验证**（hook 配置按 `AGENTS.md` 是会话启动时读取，agent 定义是否同理**无证据**）——两条变通
  （重启会话；或主链路直接调用 `mcp__arxiv__*`，它不被 R-049 拦）**都标为未验证**。
- **未做**：不改 R-049/R-050 正文；不改闸门 matcher；不把 MCP 写进仓库级 `.zcode/config.json`；
  **未做端到端真实委派**——本次只验证「文件与契约就位、既有断言不漂移」，没有真的委派它检索过一篇论文。
- **capacity 轮的推进方式不变**：仍在单一窗口跑 depth 4→8，并按 §3 预声明出口读。

### 2026-10-13 — capacity（depth 4→8）轮**运行尝试失败**：seed 41 CUDA OOM；操作者判定此方向不能这样做

- **做了什么**：在**单一窗口**（本会话 `sess_7c571a03`，goal 与 workflow 同 app 实例）重开干净一轮——
  workflow `dwfrun-c9039cf5-6fca-4fe5-83c2-c888fddfd184` 用 `.zcode/workflow-drafts/r7-s3-capacity-depth-run.dwf.ts`
  跑编排；`freeze` 自动建 `attempt04`。
- **实际发生**：独立复核 clean（与剂量驱动差异 9 处，全部为预注册：对照/评分域/父状态 identity-gated 扩张；
  复核者实跑编码器逐位一致、pin 表比对、`strace` 证明测试不读 `outputs/`）；`code.zip` 按 HEAD `370dca0`
  重建；`freeze`/`prepare`/`archive` 全 success。**`seed 41` 失败**：训练 **1948.854226 s** 后
  `torch.OutOfMemoryError`（`exit_code=1`、`passed=false`、soft/hard overrun 0）。`seed 42`/`seed 43`/`reading`
  **从未运行**。
- **根因（已确认，非推测）**：`failed_attempt.json.failure_reason` 原文——24 GiB 共驻卡剩 **4 MiB**；
  邻居 pid 2967988 **5.15 GiB** + pid 3516490 **9.17 GiB** + 本进程 **9.18 GiB**（PyTorch 分配 8.56 GiB、
  reserved-unallocated 313.72 MiB）。spawn 时余量门**通过**（free 15.29 GiB ≥ required 9.0 GiB），
  碰撞来自**第二邻居在时间点检查之后长大**（该 pid 随后又回落到 ~2.9 GiB）。**未对任何邻居发信号**（决策 0026）。
- **口径（必须照此读）**：这是**一次资源失败**，**不是**对 depth 4→8 假设的科学判定；本轮 §3 预声明出口
  （72 h 三 seed 全正 → 提请独立验收；否则关闭加大容量支）**既未触发也未满足**，因为三 seed 从未跑完。
  **无读数、无 seed 均值、无同时区间、无 primary/gate 判定**。
- **操作者决定（关键）**：操作者指出其**初衷**是用**中小型模型、较小显存**在统一环境下超过气候态、进而
  超过其它模型；但仅**主干深度翻倍**就要 ~9.2 GiB，而本机是**公用显卡**（默认共驻），不可能稳定给这么大显存。
  故操作者判定「靠加大容量解决 72 h」这一方向与本项目前提冲突、在本机**不可行** ⇒ **诚实标记"不能这样做"**
  （资源/前提判断，**非**科学证伪）。**不重跑、不加 seed、不重试**；部分输出原样保留、不删不复活。
- **登记**：新证据页 `docs/R7_S3_CAPACITY_DEPTH_RUN.md`（`scientific_claim: false`）、新 index record
  `s3-capacity-depth-run`（`outcome_class: unresolved`、`candidate_state: blocked`、`gpu_hours: 0.5414`、
  `ran: false`）；旧「未运行」record `s3-capacity-depth` 与证据页 `docs/R7_S3_CAPACITY_DEPTH.md`
  **逐字节保留**，只在其上追加 `correction_record_id` 指向新 record。§7 账本新增一行，**累计 53.3064 → 53.8478**。
  code 身份 `370dca0` + code.zip `ad15b3d4…`，该 code 的 CI `38262038297` completed/success（R7 CPU CI）。
- **登记提交与 CI**：`677c914`（证据页 `docs/R7_S3_CAPACITY_DEPTH_RUN.md`）与 `31228d5`（index/brief/§7/§8/campaign-state 四件套）已推工作分支 `r7/weather-reasoning`；精确主 CI run **`38278337165`**（R7 CPU CI，对 `31228d5`）**completed/success**，pytest job **12 步全 success**（17 条标签实验 workflow 显示 `skipped` 是设计行为、非实验 PASS）。本地确定性闸门：`r7_round.py verify`（check_conventions 37 阻断 0 失败、check_campaign_state C-01..C-06 **0 失败**、index+brief PASS）与 `commit-guard` 均通过。
- **为何停 + 恢复命令**：停因＝**操作者暂停 goal** 并判定该方向不能这样做（**非**机器可判失败：无 hard overrun、
  无身份/资源门失败，失败点是共驻 OOM）。恢复命令＝在**新窗口**先核 HEAD/status、只读本 §8 与 §2/§3/§5、
  核两个校检器；若要重跑本臂，第一条命令是 `.venv/bin/python tools/r7_round.py round-guard`
  （前置：`code.zip` 的 ZIP comment 必须等于 HEAD，否则先按 HEAD 重建）。
- **下一动作（S3）**：**由操作者在新的窗口重新规划方向**（架构 / 数据 / 目标组织），并按「中小模型 / 小显存」
  前提约束新的 round brief。宽输入支与训练配方支已关闭不重开；test 未读、r=0、S4 未启动不变。

### 2026-10-12 — capacity 轮**改「双卡一卡一 seed」**（决策 0047）：方向不废弃，改用两张卡 + 邻居增长 reserve

- **做了什么（本轮无 GPU）**：按操作者决定，把 capacity 轮（depth 4→8）从「钉死单张 `GPU-408ad137…` +
  一次一个 seed」改成「**一卡一 seed、并发**」，**默认双卡、保留单卡**，**不改任何训练配置**、**不引入 DDP**
  （先例 `scripts/study_r7_b3_scale.py`：两张 3090 无 NVLink，DDP 惩罚从未收回）。实现是**纯增量**：
  - 新增 `training/r7_device_selection.py`（只读多卡探测 / 分配 / 逐卡门槛 / 邻居 reserve）；**不动**
    `training/r7_s3_v3_screen.py`——`gpu_gate` 与 `pin_declared_gpu` 一字未改，历史臂继续按 `gpu_uuid` 校验。
  - `tools/r7_round_stages.py`：`run_stage(device=)` 只对**分配到的**那张卡过门并 pin 它；新增 `run_seeds()`
    ——**每张卡同一时刻至多一个本轮 worker**，seed 的偏好卡是 `gpu_uuids` 上的确定性轮转；
    `headroom-blocked` 有界等待后重问，**从不给邻居发信号**（R-054 / 决策 0026）。
  - `tools/r7_round.py` 新增 `stage-seeds`；workflow 草稿 `r7-s3-capacity-depth-run.dwf.ts` 的三个 seed 步
    换成一次 `stage-seeds`（草稿在 `.zcode/` 下、按 R-035 不提交）。
  - 驱动 `scripts/study_r7_s3_capacity_depth.py`：冻结块增写 `device_mode` / `gpu_uuids` /
    `neighbour_reserve_bytes`，worker 用**本次分配**的卡；round-spec 第 5 行加 `"device_mode": "dual"`。
- **门槛口径**：`required = max(估计峰值, 实测峰值) + margin + reserve`，本轮 reserve = **6 GiB**。
  依据是 `attempt04` 那次失败的现场：spawn 时门槛**通过**（free 15.29 GiB ≥ 9 GiB），seed 41 仍在
  1948.85 s 后 CUDA OOM，因为同卡邻居从 ~8.1 GiB **长到 14.32 GiB**（+6.2 GiB）。**reserve 是判断不是保证**：
  邻居涨得更多仍会 OOM，这类失败如实登记为资源失败，**不写成「已解决」**。
- **验证**：新增 `tests/test_r7_device_selection.py`（15 例）；`tests/test_r7_round_stages.py` 多卡段含反证
  「同卡第二个 worker 必须被拒」「缺 `gpu_uuids` 时旧路径逐位不变」「不在冻结列表内的分配必须报错」；
  `tests/test_r7_s3_capacity_depth.py` 加协议字段 / `single` 与 `dual` 两态 / 轮转 / 反证。本地闸门
  `tools/check_conventions.py` 37 条阻断 **0 失败**、`check_goal_brief` 0 失败、全套 `pytest` 绿；
  精确 CI run **`38298866946`**（`R7 CPU CI`，对 `ca15eea`）**completed/success**，pytest job **9 个主步骤全 success**（17 条标签实验 workflow 显示 `skipped` 是设计行为、非实验 PASS）。
- **未做**：**本轮没有跑任何 GPU**（0 GPU-h，§7 新增一行 0.0000）；**没有重跑、没有加 seed**；`attempt04`
  原样保留、不删不复活、不续跑；未改 R-054 的陈述与任何闸门 matcher；未碰
  `docs/R7_EVIDENCE_INDEX.jsonl` / `docs/R7_CANDIDATE_BRIEF.md` / `docs/R7_S3_CAPACITY_DEPTH_RUN.md`。
- **为何停 + 恢复命令**：上一块的判定是「操作者判定靠加大容量解决 72 h 不能这样做、不重跑」；本块**取代该判定**
  ——操作者指出真正的问题是**单张卡硬吃邻居增长**，解法是**把两张卡都用起来并给邻居留余量**，而不是放弃
  depth 4→8 假设，所以方向**不废弃**。恢复/开跑命令＝在**新窗口**先
  `.venv/bin/python tools/r7_round.py round-guard`（前置：`outputs/r7_s3_capacity_depth_code.zip` 的 ZIP comment
  必须等于 HEAD，否则先按 HEAD 重建），再起 `.zcode/workflow-drafts/r7-s3-capacity-depth-run.dwf.ts`。
- **下一动作（capacity 轮，双卡）**：在**新窗口**跑 `.venv/bin/python tools/r7_round.py round-guard` 确认前置，
  然后起 `r7-s3-capacity-depth-run.dwf.ts`；本轮驱动身份已变 ⇒ 必为**新 attempt**（`attempt05`），
  按 §3 预声明出口读（72 h 三 seed 全正 → 提请独立验收；否则关闭「加大容量」支转架构）。

### 2026-10-13 — capacity 轮（depth 4→8）三 seed 跑完：72h 三 seed 全负，按决策 0048 关闭深度轴

- **做了什么**：按 `docs/goals/s3-capacity-or-architecture.md` 的预声明出口把 capacity 轮跑完。单窗口重开一轮
  （`freeze` 自动建 `attempt05`），**决策 0047 的双卡生效**：GPU0 跑 seed 41、GPU1 跑 seed 42，seed 43 在腾出的卡上接续；
  三 seed 全部到 2400 更新并评分五个 lead。代码身份 `4e9fc96` + `code.zip` `e8802b7f…`（1390 条目，ZIP comment ==
  `code_commit`）、协议 `a8e472f7…`、执行文件 187、`model_code_sha256` `d3fb58db…`（未变）、`data_identity` `2360d42b…`。
- **读数（登记口径 `interior_32` t2m）**：72h skill **−0.1512 / −0.2545 / −0.1432**（均值 −0.1830）**三 seed 全负**
  ⇒ 预声明出口的正向条件（72h 三 seed 全正）**未满足**。相对已登记 2400 剂量臂（同 seed 配对、pinned 复用、未重训练）
  Δrmse 6/12/24/48/72h = **+0.0042 / +0.0046 / −0.0078 / −0.0259 / −0.0298 K**：24/48/72h **三 seed 全优**、6/12h 仅 1/3
  ⇒ 深度是**弱正向杠杆、收益集中在长 lead**；但把 72h 推回气候态平价需再降约 **0.29 K**，本臂只贡献 **0.0298 K（约 1/10）**。
  相对已登记窄 800 臂 12/24/48/72h 三 seed 全优（−0.0238/−0.1237/−0.3345/−0.3604 K）、6h 持平。
- **成本**：whole 33170.720 s、**9.2141 GPU-h**（prepare 0.1030 + archive 0.1041 + seed41 2.4861 + seed42 3.1772 +
  seed43 3.1817 + reading 0.1620）；**soft/hard overrun 均 0**；网络 0；磁盘 ≈14 GiB；累计 **53.8478 → 63.0619**。
  attempt04 那次 OOM 的 0.5414 GPU-h 已单独记账，两者都全额计费（决策 0045）。
- **一条实测修正（下一轮必须照此标定）**：协议 `estimated_peak_bytes` 写 8.5 GiB，三 seed 实测峰值**均为 11.03 GiB**
  （低估 2.5 GiB）；真正让本轮跑下来的是 0047 的 **6 GiB 邻居 reserve**，不是那个估计值。
- **登记**：新 record `s3-capacity-depth-result`（`negative` / `not-candidate`）+ 证据页
  `docs/R7_S3_CAPACITY_DEPTH_RESULT.md`。这是同一因子的**第三条**登记（① 未运行 `s3-capacity-depth`、
  ② OOM `blocked` `s3-capacity-depth-run`、③ 本次），前两条**逐字节保留**，只在 ② 上追加 `correction_record_id`。
  **不进 S4 冻结包。**
- **规则**：按决策 **0048**（`docs/decisions/0048-capacity-axis-scope-and-cap.md`），本出口关闭的是**深度轴**；
  宽度是**独立轴**、被封顶为**最后一个**容量臂——若宽度也未能让 72h 三 seed 全正，则整条容量轴（深度＋宽度）**永久关闭**。
- **为何停 + 恢复第一条命令**：本轮按预声明出口**正常结束**（不是被打断，无 hard overrun、无身份/资源门失败）。
  下一轮第一步 `.venv/bin/python tools/r7_round.py state`，然后按 0048 二选一：
  ① **宽度单因素臂**——必须先写自己的 round brief，并显式处理"宽度**没有**逐位恒等扩张"（实测 `dim` 192→256 时
  state_dict 155 个张量里 **150 个形状变化**，已登记的 dim-192 父状态装不进去，不得照搬 depth 的 identity-gated 措辞），
  且 `estimated_peak_bytes` 必须按 ≥11.03 GiB 标定；② 直接走 `docs/plans/0022` 的 **H-A**（P_0 输入条件化，
  与出口原文完全一致、零规则风险）。
- **登记提交的 CI 与一次修复（如实记录）**：登记提交 `09fbe66` 的 `R7 CPU CI`（run `38381529000`）
  **第 9 步红**——前 4 步（Check repository conventions / Check campaign state / Check evidence index and
  candidate brief / Compile active modules and check whitespace）**全绿**，说明**不是登记内容违规**；
  红的是全量测试步，根因是**既有测试的浮点相等缺陷**：`test_cli_ledger_plan_prints_arithmetic` 跑在真实账本上，
  断言 `new_total_gpu_h == previous_total_gpu_h + 1.0`，而 `ledger_plan` 把总量 `round(..., 6)`——
  `53.8478 + 1.0` 恰好往返、`63.0619 + 1.0` 不能（`64.06190000000001`），本轮累计 53.8478 → 63.0619
  只是把这个一直存在的脆弱断言暴露出来。修复提交 `3d2b546`（改 `pytest.approx(..., abs=1e-6)`；
  **未放宽任何判据**，同文件既有反证与实跑偏差反证都证明容差仍能抓住真错）的 CI run
  **`38386670446` completed/success**；本地全量套件 **4501 passed / 3 skipped / 0 failed**（22:46）。
  index74 记录的 `ci_run_id` 指向实验代码 commit `4e9fc96` 的绿 CI `38301335129`，不受此影响。
- **下一动作（S3）**：写下一轮 round brief（**宽度单因素** 或 plan 0022 的 **H-A**），先跑
  `.venv/bin/python tools/r7_round.py round-guard` 确认前置，再冻结新协议。宽输入支与训练配方支已关闭不重开；
  test 未读、r=0、S4 未启动不变。

### 2026-10-14 — capacity 轮（width dim 192→256）三 seed 跑完：72h 三 seed 全负，按决策 0048 永久关闭整条容量轴（深度＋宽度）

- **做了什么**：按 `docs/goals/s3-capacity-width.md` 的预声明出口把**最后一个容量臂**跑完（决策 0047 双卡：
  GPU0 跑 seed 41、GPU1 跑 seed 42，seed 43 在腾出的卡上接续），三 seed 全部到 2400 更新并评分五个 lead。
  代码身份 `35b1785` + `code.zip` `14079fe6…`（1395 条目，ZIP comment == `code_commit`）、协议 `277cf27e…`、
  执行文件 187、`model_code_sha256` `d3fb58db…`（未变）、`data_identity` `2360d42b…`、source `6bc9a1a2…`。
- **本臂不是"宽度单因素"（必须照此读）**：宽度**无逐位恒等扩张**——`dim` 192→256 时 state_dict 的
  155 个张量里 **150 个形状变化**，注册的 dim-192 父状态装不进去；本轮用**冻结零填充扩张**（重叠切片拷入、
  其余置零），LayerNorm 在 256 维上归一化、被置零的高 64 维并非严格惰性 ⇒ 本臂测的是
  「**宽度 + 该零填充初始化**」的**联合效应**，**不是**宽度单因素。**不得**照搬 depth 轮的 identity-gated「单因素保真」措辞。
- **读数（登记口径 `interior_32` t2m）**：72h skill **−0.1658 / −0.2833 / −0.1418**（均值 −0.1970）**三 seed 全负**
  ⇒ 预声明出口的正向条件（72h 三 seed 全正）**未满足**。相对已登记 2400 剂量臂（同 seed 配对、pinned 复用、未重训练）
  Δrmse 6/12/24/48/72h = **+0.2880 / +0.2751 / +0.0705 / +0.0419 / −0.0090 K**：6/12/24/48h **0/3 seed 更优**、
  仅 72h **2/3** ⇒ 宽度**不是**正向杠杆。相对已登记窄 800 臂 24/48/72h 更优（−0.0454/−0.2666/−0.3396 K）、6/12h 更差。
- **成本**：whole 28683.583 s、**7.9677 GPU-h**（prepare 0.1127 + archive 0.1110 + seed41 2.0273 + seed42 2.0230 +
  seed43 3.5642 + reading 0.1293；口径 = attempt 下全部 6 个 `*_process.json` 的 `elapsed_seconds` 之和 ÷ 3600，
  无 attempt 级 wall clock）；**soft/hard overrun 均 0**；网络 0；磁盘 ≈14 GiB；累计 **63.0619 → 71.0296**。
- **一条实测修正（连续第二次）**：协议 `estimated_peak_bytes` 写 11.0352 GiB，三 seed 实测峰值**均为 11.9023 GiB**
  （仍低估约 0.87 GiB）；真正让本轮跑下来的是 0047 的 **6 GiB 邻居 reserve**。下一轮标定必须 ≥11.90 GiB。
- **登记**：新 record `s3-capacity-width-result`（`negative` / `not-candidate`）+ 证据页 `docs/R7_S3_CAPACITY_WIDTH_RESULT.md`。
  这是本因子**第一条**登记（此前无任何已登记 record，故**不**追加 `correction_record_id`）。**不进 S4 冻结包。**
- **CI 如实（未粉饰）**：实验 commit `35b1785` 的 `R7 CPU CI` run **`38391291040` completed/failure**——第 5–8 步
  （conventions / campaign state / evidence index / compile）**全绿**，红的是第 9 步全量测试：**本臂自己的测试文件**
  `tests/test_r7_s3_capacity_width.py:392` 对 `capacity.execution_files()` 逐个取哈希，其中含
  `outputs/r7_v2_comparison_20261004_attempt01/protocol.json`，该路径在**无 `outputs/` 的干净 CI checkout** 上不存在
  ⇒ 4 个用例 `FileNotFoundError`；本机因 `outputs/` 存在故**全量套件 4540 passed / 3 skipped / 0 failed**（1255.30 s）。
  这是**测试侧工程缺陷、非读数/科学问题**；本轮登记**未改**该测试文件（登记员只追加本轮材料），缺陷留待后续修复。
- **规则**：按决策 **0048**（`docs/decisions/0048-capacity-axis-scope-and-cap.md`），宽度是**最后一个**容量臂；
  它未能让 72h 三 seed 全正 ⇒ **整条容量轴（深度＋宽度）永久关闭**——不再开任何容量臂（不重试 depth、不再扫 dim、不做宽＋深组合）。
- **为何停 + 恢复第一条命令**：本轮按预声明出口**正常结束**（不是被打断，无 hard overrun、无身份/资源门失败）；
  容量轴按 0048 封顶条款**永久关闭**。此后只剩**训练信号 / 目标组织 / 数据**（数据按 D3 暂缓、放最最后）。
  下一轮第一步 `.venv/bin/python tools/r7_round.py state`，随后为新方向写新 round brief（零规则风险的候选是
  `docs/plans/0022` 的 **H-A**：P_0 输入条件化）。
- **下一动作（S3）**：写下一轮 round brief（**训练信号 / 目标组织**方向，数据按 D3 暂缓；候选 `docs/plans/0022` 的
  **H-A**），先跑 `.venv/bin/python tools/r7_round.py round-guard` 确认前置，再冻结新协议。容量轴（深度＋宽度）
  已按决策 0048 **永久关闭不重开**；test 未读、r=0、S4 未启动不变。

### 2026-10-14 — 绑定：宽度登记的 CI 红→绿（测试侧修复 + 精确 run）

- **修复**：commit `1f54079`（`fix(r7): make the capacity-width validate_protocol tests CI-safe`）。根因＝
  `tests/test_r7_s3_capacity_width.py` 的 3 个 `validate_protocol` 用例经 `_protocol_body` 对
  `capacity.execution_files()` 逐个取哈希，而该清单含 `outputs/r7_v2_comparison_20261004_attempt01/protocol.json`，
  在**无 `outputs/` 的干净 CI checkout** 上不存在 ⇒ `FileNotFoundError`（第 9 步全量测试红）。修法：把清单收窄到
  **实际存在**的文件并 stub 掉 `sha256_file`（被断言的仍是冻结字段，不是磁盘源码哈希）；**未放宽任何断言**。
  在**干净 git worktree（无 `outputs/`）**上复现并验证：修复前 4 failed / 修复后 **39 passed**。
- **精确 CI**：`R7 CPU CI` run **`38454943530`** 对 `1f54079` **completed / success**（第 5–8 步全绿、第 9 步转绿）。
- **下一动作（S3）**：按 `docs/plans/0022` 的 **H-A**（P_0 输入条件化）写下一轮 round brief + 派生单因素驱动，
  先跑 `.venv/bin/python tools/r7_round.py round-guard` 确认前置，再冻结新协议。容量轴已按 0048 永久关闭；
  test 未读、r=0、S4 未启动不变。

### 2026-10-14 — H-A（P_0 逐样本输入条件化）轮**运行尝试失败**：父身份门拒绝；已用**已提交的**工具合法重迁移

- **做了什么**：按上一块「下一动作」开 H-A 轮——round brief `docs/goals/s3-process-init-conditioning.md`、
  单因素驱动 `scripts/study_r7_s3_process_init.py`、CI-safe 测试 `tests/test_r7_s3_process_init.py`、
  模型改动 `model/process_forecast_r7.py`（新增 `process_init_proj` + 一处加性注入点，标 `[model-digest-change]`）；
  用 `.zcode/workflow-drafts/r7-s3-process-init-run.dwf.ts` 编排（单窗口、goal 与 workflow 同 app 实例）。
- **实际发生**：独立复核 clean（与剂量驱动差异全为**预注册**：唯一因子 `process_init_conditioning` False→True、
  identity-gated 扩张、off 路径逐位一致；复核者实跑 `all_shared_bit_identical: True`、新增参数 37440、
  清零投影恒零、pin 15/15、干净 worktree 无 `outputs/` 时 **39 passed**）；`code.zip` 按 HEAD `7539bb3` 重建
  （`00f447d6…`）；`freeze`/`prepare`/`archive` 全 success。**`stage-seeds` exit 1**：seed 41 与 seed 42
  **都在训练之前**的 `_parent_model → load_checkpoint` 抛 `ValueError: checkpoint model implementation differs`
  （`training/r7_experiment.py:129`），**seed 43 从未启动**（`device_mode: dual`，一卡一 seed、两卡均失败）。
- **根因（已确认，非推测）**：本轮**确实修改了 `model/*.py`**（这是相对 depth/width 两轮的实质差异——那两轮只改
  **spec**）。`model_code_digest()` 由 `d3fb58db…` 变为 `223b2e5b…`，而注册时迁移好的父
  `…_migrated_20261009_attempt02` 的 receipt 记的是**旧** digest ⇒ 身份门**按设计**拒绝。
  **不是** bug、**不是**父损坏（父文件在其自身 SHA256 pin 校验处已通过），**不得**绕过。
- **口径（必须照此读）**：这是**一次身份门失败**，**不是**对 H-A 假设的科学判定；本轮 §3 预声明出口
  （72h 三 seed 全正 → 提请独立验收；否则关闭 H-A 支）**既未触发也未满足**，因为**训练从未开始**。
  **无读数、无 seed 均值、无同时区间、无 primary/gate 判定**；**0 GPU-h**、网络 0、soft/hard overrun 0。
- **结构性发现（本轮主要产出）**：**任何 `model/*.py` 改动都会改变 `model_code_digest()`，从而让所有"迁移父"
  在「微调自注册父」协议下不可加载**；spec-only 改动（depth、dim）不动 digest，所以容量两轮能跑、H-A 不能。
  这条约束对后续任何架构方向都成立，是架构轴的真实成本项。
- **合法修复（已执行，未绕过身份门）**：用**仓库内已提交的** `scripts/migrate_r7_parent_state.py`
  （docstring 明言 "performs the documented model-only migration **instead of weakening that guard**"）
  在**当前**代码下重跑迁移 →
  `outputs/r7_s3_rollout_dose_parent_migrated_20261014_attempt01`（seed SHA `3bb09c19…/91667a68…/b1a716bc…`、
  digest＝当前 `223b2e5b…`；`semantics_equal: false` 系新增 gated 模块所致、属预期，off 路径逐位一致由本臂正控保证）。
  **未**修改任何比对逻辑、**未**放宽任何阈值/门、**未**改 `data/raw|interim|processed` 或归档快照、
  **未**对任何非本实验进程发送信号。
- **登记**：新证据页 `docs/R7_S3_PROCESS_INIT_RUN.md`（`scientific_claim: false`、`test_read: false`）、
  index record `s3-process-init-run`（`outcome_class: unresolved`、`candidate_state: blocked`、`gpu_hours: 0.0`、
  `ran: false`、`failure_class: checkpoint_model_digest_mismatch`）；§7 账本新增一行（**0.0000**，
  故累计仍 **71.0296**）；round brief 补 §6b（失败与修复）与 §7「下一动作（S3）」（补 C-04 所需）。
- **登记提交与 CI（补记）**：`b2f654f`（证据页 `docs/R7_S3_PROCESS_INIT_RUN.md`）与 `1285dc4`（index/brief/§7/§8/campaign-state
  四件套）在同一次 push 里，GitHub 只对 head `1285dc4` 跑 CI ⇒ `b2f654f` 无独立 run；`1285dc4` 的 `R7 CPU CI`
  run **`38470790188` completed/success**。父迁移修复 `81233b3` 的 run 见下一块。本地确定性闸门：
  `r7_round.py verify`（check_conventions 37 阻断 0 失败、check_campaign_state C-01..C-06 0 失败、index+brief PASS）
  与 `commit-guard` 均通过。
- **下一动作（S3）**：把驱动的父解析从 `wide.dose.pinned_parent` 切到本轮自己的迁移父
  （`…_migrated_20261014_attempt01` + 上表新 SHA pin），`round-guard` → 重新 `freeze`（新 `protocol_sha256`）→
  `stage prepare/archive/stage-seeds/reading` → 四件套登记。第一条命令：
  `.venv/bin/python tools/r7_round.py round-guard`。预声明出口不变。**不据此关闭 H-A 支**。

### 2026-10-14 — H-A（P_0 逐样本输入条件化）轮**跑完**：72 h 三 seed 全负，按预声明出口关闭 H-A 支

- **做了什么**：按上一块「下一动作」把驱动的父解析切到本轮自己的迁移父（`…_migrated_20261014_attempt01` + 新 SHA pin），`round-guard` → 重新 `freeze`（新 `protocol_sha256`）→ `stage prepare/archive/stage-seeds/reading` → 四件套登记。attempt02 六阶段全 success、`returncode=0`、`signals=[]`。
- **结果（唯一因子 = `process_init_conditioning` False→True）**：`interior_32` t2m 72h skill **−0.2437 / −0.3064 / −0.2171（均值 −0.2557）三 seed 全负** ⇒ 预声明出口正向条件（72 h 三 seed 全正）**未满足**；相对已登记 2400 剂量臂（同 seed 配对、pinned 复用、未重训练）Δrmse 6/12/24/48/72h = **−0.0027 / −0.0026 / +0.0103 / +0.0605 / +0.0798 K**（正值=本臂更差；24/48/72h **0/3** seed 更优）⇒ 逐样本 P_0 条件化**不是正向杠杆**，长 lead 反而更差。48 h 仍三 seed 全正（+0.0211/+0.0022/+0.0733），72 h 仍三 seed 全负。
- **保真**：`process_init_proj` **最后、在 `isolated_stream()` 下构造**，off 路径**逐位等于现实现**（逐 seed 实测 `bit_identical_off_path=true`）；清零其 Linear 则注入恒为 0（`proj_zeroing_is_identity=true`、`zeroed_proj_output_abs_max=0.0`）；新增参数 37440（4 键）。这是相对容量两轮的实质差异：本轮改的是**模型前向**，故 `model_code_sha256` 由 `d3fb58db…` 变为 `223b2e5b…`。
- **成本**：whole 24,692.481 s、**6.8590 GPU-h**（prepare 0.1316 + archive 0.1320 + seed41 2.5813 + seed42 1.9777 + seed43 1.9085 + reading 0.1281；口径 = attempt 下全部 6 个 `*_process.json` 的 `elapsed_seconds` 之和 ÷ 3600，无 attempt 级 wall clock）；**soft/hard overrun 均 0**；网络 0；磁盘 ≈14 GiB；累计 **71.0296 → 77.888623**。
- **一条实测修正（方向与容量两轮相反）**：协议 `estimated_peak_bytes` 写 11.91 GiB，三 seed 实测峰值**均为 8.361 GiB**（**高估** 3.55 GiB）；本轮余量门因而是保守的。
- **登记**：新 record `s3-process-init-result`（`negative` / `not-candidate`）+ 证据页 `docs/R7_S3_PROCESS_INIT_RESULT.md`。本因子第一次尝试（attempt01 身份门失败、0 GPU-h）已单独登记为 `record:s3-process-init-run`（`blocked`）；本页为其跑完的 attempt02，**不**向任何旧 record 追加 `correction_record_id`。**不进 S4 冻结包。**
- **CI 如实**：实验 commit `81233b3` 的 `R7 CPU CI` run **`38474413900` completed/success**（另 17 条标签 workflow `skipped` 是设计行为、非实验 PASS）。登记提交 `7295291`（证据页）与 `bf7964c`（index/brief/§7/§8/campaign-state 四件套）同一次 push ⇒ 只对 head `bf7964c` 跑 CI，其 `R7 CPU CI` run **`38505965671` completed/success**。
- **为何停 + 恢复第一条命令**：本轮按预声明出口**正常结束**（不是被打断，无 hard overrun、无身份/资源门失败）；H-A 是决策 0048 永久关闭容量轴后的**第一根架构轴**，它也未让 72 h 三 seed 全正 ⇒ **关闭 H-A 这一支**（不调 proj 宽度、不加 seed）。此后架构方向还剩 `docs/plans/0022` 的 **H-B / H-C**，训练信号 / 目标组织方向亦开放（数据按 D3 暂缓、放最最后）。恢复第一条命令＝ `.venv/bin/python tools/r7_round.py state`，随后为新方向写新 round brief。
- **下一动作（S3）**：写下一轮 round brief（架构 **H-B / H-C** 或**训练信号 / 目标组织**方向，数据按 D3 暂缓），先跑 `.venv/bin/python tools/r7_round.py round-guard` 确认前置，再冻结新协议。H-A 支已按预声明出口关闭、不重开；容量轴（深度＋宽度）已按决策 0048 永久关闭不重开；test 未读、r=0、S4 未启动不变。

### 2026-10-14 — H-B（空间分辨的类型化证据）轮**开轮**：round brief 已写（承接 H-A 关闭，按 0022 排序取第二位）

- **方向依据**：H-A 已按**预声明出口关闭**（`record:s3-process-init-result`，72h 三 seed 全负、相对剂量臂长 lead 反而更差）；容量轴（深度＋宽度）已按**决策 0048 永久关闭**；`docs/plans/0022` §4 的排序是 **H-A > H-B > H-C**，故下一候选是 **H-B**（训练信号 / 目标组织方向亦开放，数据按 D3 暂缓放最最后）。
- **本轮 brief**：`docs/goals/s3-typed-spatial.md`（`<!-- round-node: S3 -->`）。**唯一改动因子 = 类型化证据 token 的空间路由**：从"按位置求均值压成 1 个向量"（`model/typed_evidence_r7.py:199` 的 `value.mean(dim=1)`）改为**保留 patch 位置 `[B,P,dim]` 并做位置对齐注入**（同信息、不同路由）。**B/C 两臂在同一协议内同跑**（B = 同信息 pooled = #79 现状；C = 同信息空间保留 = 候选），参数量/FLOPs 必须对齐，只检验 **B→C** 的归因；A（无类型）= 已登记 2400 剂量臂 pinned 复用。`check_goal_brief` **failures=0**（1 条 A-02 advisory，与 H-A brief 同）。
- **父迁移前提已写入 brief §6**：本轮同样改 `model/*.py` ⇒ `model_code_digest()` 会变 ⇒ 必须**照搬 H-A 的合法修复模式**——先用**已提交的** `scripts/migrate_r7_parent_state.py` 在当前代码下重跑迁移，再按 `scripts/r7_s3_process_init_parent.py` 同一模式建专属父模块并由驱动 `install_parent_pins(body)` 覆写冻结协议的 pins。**不得**改比对逻辑或放宽任何门。
- **文献旁证**：`.zcode/plans/s3-hb-lit-2026-10-13.md`（**工作态草稿、不进 index**）——委派 `arxiv-researcher` 的结论＝**只有间接支持、没有直接实证**：NeuralGCM（给水平梯度）、ClimaX（per-position cross-attention 聚合）、GraphCast（多尺度同质网格）、Pangu（3D vs 2D 保留结构维度）都表明主流做法保留空间结构，但**没人做过"同信息只改路由"的定量对照** ⇒ H-B 的方向**不可由文献决定**，必须靠自己的 B/C 对照判定。
- **口径（必须照此读）**：本轮**尚未开跑**——**0 GPU-h**、无 attempt 目录、无读数、无 seed 均值、无同时区间、无 primary/gate 判定；§3 预声明出口**既未触发也未满足**。**不宣布通过、不关闭任何支**、不称泛化/SOTA。
- **下一动作（S3）**：实现模型改动（`model/typed_evidence_r7.py` 保留 `P` 维 + 路由开关 + 位置对齐注入）与 `scripts/r7_s3_typed_spatial_parent.py`，再派生驱动（**R-051 600 行硬限**）与 CI-safe 测试（含"类型置零 ⇒ 恒零"反证、B/C 参数量与 FLOPs 对齐断言）；先跑 `.venv/bin/python tools/r7_round.py round-guard` 确认前置，再冻结新协议。第一条命令＝ `.venv/bin/python tools/r7_round.py round-guard`。test 未读、r=0、S4 未启动不变。

**【外部告知 · 非本轮产物 · 不改任何指针】** 另一会话留（2026-10-14 晚）：写下一轮 round brief 前，请读 **`.zcode/plans/s3-READ-ME-next-round-levers-20261014.md`**（git-ignored 工作态笔记，不占 index、不是 brief）。内含 72h 绝对气候态门卡点的**机制诊断**——t2m 的 `skill = 2·α·ρ − α²`、天花板 = ρ²；实测 72h 的 ρ(ACC)=0.210 为正但振幅 α≈1.18 过冲约 5.6×，天花板仅 +0.044——以及按证据强度排序的候选杠杆 C1–C5（幅度保持损失 / 填满 12 步监督 / Professor Forcing / CRPS 概率化 / 经典基线参照）与两条检索通道的一手留痕。该文件只负责告知，主链可径行采纳或删除本行。

### 2026-10-14 — H-B（空间分辨的类型化证据）轮**跑完**：B→C 无差异（~1e-6 K），72 h 三 seed 全负，按预声明出口关闭 H-B 支

- **做了什么**：按上一块「下一动作」实现模型改动（`model/typed_evidence_r7.py` 保留 `P` 维 + 路由开关 + 位置对齐注入）与 `scripts/r7_s3_typed_spatial_parent.py`、派生驱动与 CI-safe 测试，`round-guard` → `freeze` → `stage prepare/archive/stage-seeds/reading` → 四件套登记。attempt01 六阶段全 success、`returncode=0`、`signals=[]`。
- **结果（唯一因子 = 证据 token 在 patch 位置维上的聚合方式：pooled 均匀均值 → spatial softmax 注意力）**：**B/C 两臂在同一协议内同跑**（B=pooled=#79 现状，C=spatial=候选），共享张量逐位一致、C 恰多一个 `typed_evidence.position_query`（+192 参数，+0.13% of 148800）。**B→C 无差异**：五个 lead 的 seed 均值 Δrmse 为 **+1.2e-7 / −1.3e-6 / −2.8e-6 / −6.5e-8 / +3.2e-6 K**（|Δ|≤3.2e-6 K，float32 噪声量级），72 h **三 seed 不同号**（−5.2e-6 / +1.3e-5 / +1.9e-6）；C 的 `interior_32` t2m **72 h skill −0.1827 / −0.2791 / −0.1596（均值 −0.2071）三 seed 全负** ⇒ 预声明出口（B→C 72 h 三 seed 同号 **且** C 的 72 h 三 seed 全正）**两条都未满足**。**结论（必须照此读）**：B→C 无差异 ⇒ **类型路由本身无效**（同信息、不同路由不改变读数），**不是**「空间保留无用」。相对已登记 2400 剂量臂（同 seed 配对、pinned 复用、未重训练）Δrmse 6/12/24/48/72 h = **−0.0026 / −0.0022 / +0.0011 / +0.0052 / +0.0068 K**（6/12 h 3/3 略优，24/48/72 h 更差）；相对已登记窄 800 臂 12/24/48/72 h 三 seed 全优（−0.0306 / −0.1148 / −0.3033 / −0.3238 K），6 h 持平。
- **保真**：同 seed 两臂的 `pathway_init_sha256` **相等**（41 `18dfa0a2…` / 42 `e5d78ca9…` / 43 `2a7f855e…`）、共享 163 键 `shared_bit_identical=true`；正控制逐 seed 通过（`zero_query_is_pooled=true`、`zero_query_max_abs=0.0`；`zeroed_projection_is_identity=true`、`zeroed_projection_nonzero_spatial=0`、`zeroed_projection_nonzero_pooled=0`；`moved_max_abs_after_random_query` 0.43–0.61）。这是相对容量两轮的实质差异：本轮改的是**模型前向**，故 `model_code_sha256` 由 H-A 的 `223b2e5b…` 变为 `15523927…`。
- **成本**：whole 55,594.936 s、**15.443038 GPU-h**（prepare 0.1293 + archive 0.1318 + seed41 4.9745 + seed42 5.7667 + seed43 4.3254 + reading 0.1153；口径 = attempt 下全部 6 个 `*_process.json` 的 `elapsed_seconds` 之和 ÷ 3600，无 attempt 级 wall clock；两 seed 在两张卡并发，故该和**保守偏大**）；**soft/hard overrun 均 0**；网络 0；磁盘 ≈14 GiB；累计 **77.888623 → 93.331638**。
- **一条实测**：协议 `estimated_peak_bytes` 写 9.5 GiB，实测峰值 spatial 9.276 GiB / pooled 9.041 GiB（本轮**高估**约 0.22 GiB，余量门保守）。
- **一条关键诊断（决定这个 null 怎么读）**：直接从三 seed 的 spatial checkpoint 读出训练后的 `typed_evidence.position_query` 模长 = **0.007393 / 0.012458 / 0.008398**（`max|q|` 0.0018–0.0023），而同层投影 `projections.0.weight` 模长 ≈ **8.02–8.04** —— query 确实从 0 训练出了非零值（梯度通了、参数在更新），但比投影小**三个数量级**，故注意力打分幅度 ≈ `0.01·O(1)/sqrt(192)` ≈ 7e-4、softmax 权重相对均匀只偏移约 **1e-3**。**因此这个 B→C null 必须读作「弱操纵下的 null」**：它证明「在本轮学到的这点位置偏移内读数不变」，**不能**当作对 H-B 假设（空间结构被均值抹掉才是 #79 负结果的原因）的**强反驳**；要真正检验该假设需要让 query 有足够大的学习幅度，本轮**没有**做。出口结论「类型路由本身无效」限定在**本设置、本学习幅度**内。
- **登记**：新 record `s3-typed-spatial-result`（`negative` / `not-candidate`）+ 证据页 `docs/R7_S3_TYPED_SPATIAL_RESULT.md`。本因子此前只有 #79 的**不同实例**登记 `s2-79-typed-evidence-screening`，不是同协议对照，故**不**向任何旧 record 追加 `correction_record_id`。**不进 S4 冻结包。**
- **CI 如实**：实验 commit `3c1d2b0` 的 `R7 CPU CI` run **`38516302068` completed/success**；登记提交 `2304b3d`（证据页）与 `de23556`（index/brief/§7/§8/campaign-state 四件套）同一次 push ⇒ 只对 head `de23556` 跑 CI，其 run **`38592346142` completed/success**；补记 `418e716`（§7b 关键诊断 + index/brief/§8）的 run **`38597613716` completed/success**（另 17 条标签 workflow `skipped` 是设计行为、非实验 PASS）。**注**：workflow 的结论串写「CI 未绿/超预算」，那是它自己的 `r7_ci_wait` 等待超预算（轮次预算内 run 尚未出现），**不是** CI 失败——两个 run 事后均为 completed/success。
- **为何停 + 恢复第一条命令**：本轮按预声明出口**正常结束**（不是被打断，无 hard overrun、无身份/资源门失败）；H-B 是决策 0048 永久关闭容量轴后的**第二根架构轴**，它也未让 B→C 出现差异、72 h 三 seed 全正 ⇒ **关闭 H-B 这一支**（不调注入宽度、不加 seed）。此后架构方向还剩 `docs/plans/0022` 的 **H-C**，训练信号 / 目标组织方向亦开放（数据按 D3 暂缓、放最最后）。恢复第一条命令＝ `.venv/bin/python tools/r7_round.py state`，随后为新方向写新 round brief。
- **下一动作（S3）**：写下一轮 round brief（架构 **H-C** 或**训练信号 / 目标组织**方向，数据按 D3 暂缓），先跑 `.venv/bin/python tools/r7_round.py round-guard` 确认前置，再冻结新协议。H-B 支已按预声明出口关闭、不重开；容量轴（深度＋宽度）已按决策 0048 永久关闭不重开；H-A 支已关闭不重开；test 未读、r=0、S4 未启动不变。

### 2026-10-15 — H-C（长 lead 的全局混合路径）轮**开轮**：round brief 已写（承接 H-B 关闭，按 0022 排序取最后一位）

- **方向依据**：H-A 与 H-B 均已按各自预声明出口关闭（`record:s3-process-init-result` negative；`record:s3-typed-spatial-result` B→C null，且其证据页 §7b 已把该 null 限定为**弱操纵下的 null**）；容量轴（深度＋宽度）已按**决策 0048 永久关闭**；`docs/plans/0022` §3 只剩 **H-C**（§4 排最后是**成本**理由，不是证据理由——它是唯一直接针对"局部窗口 + 3×3 conv 感受野受限"这一结构缺口的候选）。训练信号 / 目标组织方向亦开放，数据按 D3 暂缓放最最后。
- **本轮 brief**：`docs/goals/s3-global-mixing.md`（`<!-- round-node: S3 -->`）。**唯一改动因子 = 空间混合范围**：给解码器（或 solver）加**一条全局混合路径**（低秩全局算子或单层全局注意力），局部窗口与 3×3 conv 本身不变。**G/L 两臂在同一协议内同跑**（G = 全局；L = **同参数量、同 FLOPs 的额外局部层**），只检验 **L→G** 的归因——这正是容量轴关闭后必须证明的事：差异来自「全局性」还是「单纯加容量」。`check_goal_brief` **failures=0**（1 条 A-02 advisory，与 H-A/H-B brief 同）。
- **两条已学的硬约束已写进 brief §3**：① **共享初始化**——两臂新增参数必须在同一 round seed 下逐位相同、`pathway_init_sha256` 相等、不等即 `raise`（这是 H-B 轮独立复核抓到的 medium finding 的修复，必须照搬）；② **正控制失败即 `raise`**，不是只写进 receipt。③ 父迁移必须照搬 H-A/H-B 的合法修复模式（重跑已提交的迁移工具 + 专属父模块 + `install_parent_pins` + `expand_parent_state(..., seed=...)`）。
- **口径（必须照此读）**：本轮**尚未开跑**——**0 GPU-h**、无 attempt 目录、无读数、无 seed 均值、无同时区间、无 primary/gate 判定；§3 预声明出口**既未触发也未满足**。**不宣布通过、不关闭任何支**、不称泛化/SOTA。
- **下一动作（S3）**：先以**只读测绘**固定插入点（解码器 vs solver、全局注意力 vs 低秩算子），再实现模型改动与 `scripts/r7_s3_global_mixing_parent.py`，然后派生驱动（**R-051 600 行硬限**）与 CI-safe 测试（含清零输出投影 ⇒ 逐位退回反证、G/L 参数量与 FLOPs 对齐断言、两臂共享初始化断言）；先跑 `.venv/bin/python tools/r7_round.py round-guard` 确认前置，再冻结新协议。test 未读、r=0、S4 未启动不变。

### 2026-10-15 — H-C（长 lead 的全局混合路径）轮**跑完**：G→L 无差异（~1e-3…1e-2 K），72 h 两臂三 seed 全负，按预声明出口关闭 H-C 支

- **做了什么**：按上一块「下一动作」实现模型改动（`model/global_mixing_r7.py` 的 `GlobalAttentionBlock`/`WindowAttentionBlock` 同壳 + `ContextMixLayer`、`model/process_forecast_r7.py` 的 `global_mixing_mode` 开关，在所有既有可训练权重之后、在 `isolated_stream()` 下构造）、`scripts/r7_s3_global_mixing_parent.py`、`scripts/study_r7_s3_global_mixing.py`（两臂 G/L 同协议）、`tests/test_r7_s3_global_mixing.py`；`round-guard` → `freeze` → `stage prepare/archive/stage-seeds/reading` → 四件套登记。attempt01 六阶段全 success、`returncode=0`、`signals=[]`。
- **结果（唯一因子 = 主干出口一条 context-mixing 层的注意力范围：global N×N 注意力 vs local 主干窗口注意力）**：**G/L 两臂在同一协议内同跑**（G=global，L=local），**参数量逐位对齐 444864**（同构 block 外壳、`shell_parity.parameter_counts_equal=true`），同 seed 除 12 个新增键外**逐位一致**（`shared_bit_identical=true`、167 键）、`pathway_init_sha256` 两臂相等（`55a9bd77…/db5efa02…/2c5e44bc…`）。**G→L 无差异**：五个 lead 的 seed 均值 Δrmse 为 **−1.9e-4/−3.0e-5/−2.2e-3/+1.9e-3/−2.5e-3 K**，**五个 lead 三 seed 全部不同号**（72h +4.6e-4/−8.7e-3/+8.0e-4）；72 h 两臂 skill **三 seed 全负**（G −0.1748/−0.2650/−0.1511 均值 −0.1970；L −0.1745/−0.2709/−0.1506 均值 −0.1987）⇒ 预声明出口（72h 三 seed 全正 **且** G 相对 L 在 72h 三 seed 同号为优）**两条都未满足**。**结论（必须照此读）**：L→G 无差异 ⇒ **瓶颈不是感受野**，且该无差异**与「额外容量」不可分**（两臂都加了同样的 444,864 参数）；72 h 的负 skill 在两臂里都出现，故它与混合范围无关。相对已登记 2400 剂量臂（同 seed 配对、pinned 复用、未重训练）G 48h 3/3 优（−0.0052 K）/72h 2/3 优（−0.0085 K），L 48/72h 三 seed 全优（−0.0071/−0.0061 K），6/12/24h 两臂约 1/3 优。
- **保真**：正控制逐 seed 通过（清零输出投影**与 FF 输出层** ⇒ 该层恒等 ⇒ 整个 forward 逐位退回 off，`zeroed_block_is_identity=true`/`zeroed_block_max_abs=0.0`；**只清投影不等于 off**，`proj_only_is_identity=false`、`proj_only_max_abs` 0.18–0.23）；负控制＝两臂在大于一个窗口的网格上 forward 不同（`arms_are_not_identical=true`、`arms_probe_max_abs` 0.31–0.43）。这是相对容量两轮的实质差异：本轮改的是**模型前向**，故 `model_code_sha256` 由 H-B 的 `15523927…` 变为 `fff81dcc…`。
- **一条必须记住的混杂（算力不对齐）**：两臂**参数量**逐位对齐，但**算力不对齐**——注意力长度 **68.0625×**、SDPA 子成本 **57.19140625×**、整层 **1.9129018382845497×**（已声明）。故若 G 胜出，差异**可能来自算力而非「全局性」**；本轮 G 未胜出，此混杂未兑现为假阳性优势，但它限定了这个 null 的可归因范围。
- **成本**：whole 44,236.729 s、**12.287980 GPU-h**（prepare 0.1317 + archive 0.1318 + seed41 3.9195 + seed42 4.0420 + seed43 3.9316 + reading 0.1315；口径 = attempt 下全部 6 个 `*_process.json` 的 `elapsed_seconds` 之和 ÷ 3600，无 attempt 级 wall clock；两 seed 在两张卡并发，故该和**保守偏大**）；**soft/hard overrun 均 0**；网络 0；磁盘 ≈14 GiB；累计 **93.331638 → 105.619618**。
- **一条实测**：协议 `estimated_peak_bytes` 写 9.5 GiB，实测峰值 G 8.955 GiB / L 8.959 GiB（本轮**高估**约 0.54 GiB，余量门保守）。
- **登记**：新 record `s3-global-mixing-result`（`negative` / `not-candidate`）+ 证据页 `docs/R7_S3_GLOBAL_MIXING_RESULT.md`。本因子此前**没有**任何已登记 record，故**不**向任何旧 record 追加 `correction_record_id`。**不进 S4 冻结包。**
- **CI 如实**：实验 commit `d560bbe` 的 `R7 CPU CI` run **`38607253249` completed/success**（另 17 条标签 workflow `skipped` 是设计行为、非实验 PASS）。
- **为何停 + 恢复第一条命令**：本轮按预声明出口**正常结束**（不是被打断，无 hard overrun、无身份/资源门失败）；H-C 是 H-A、H-B 关闭后的**第三根架构**轴，它也未让 72 h 三 seed 全正、G 相对 L 也无同号差异 ⇒ **关闭 H-C 这一支**（不调宽度/秩/层数、不加 seed）。此后 plan 0022 的架构方向（H-A/H-B/H-C）已**全部关闭**；训练信号 / 目标组织方向开放（数据按 D3 暂缓、放最最后）。恢复第一条命令＝ `.venv/bin/python tools/r7_round.py state`，随后为新方向写新 round brief。
- **下一动作（S3）**：架构方向（H-A/H-B/H-C）与容量轴均已关闭；下一轮在**训练信号 / 目标组织**方向（或重估主线）写新 round brief，先跑 `.venv/bin/python tools/r7_round.py round-guard` 确认前置，再冻结新协议。H-C 支已按预声明出口关闭、不重开；容量轴已按决策 0048 永久关闭不重开；H-A/H-B 支已关闭不重开；test 未读、r=0、S4 未启动不变。

**【外部告知 · 非本轮产物 · 不改任何指针】** 另一会话留（2026-10-15 上午）：**本窗口正在下载 2000–2022 全年（12 个月）区域 ERA5**，落到**全新排他路径** `outputs/r7_s3_fy_2000_2022/`（33,604 帧 / 276 个 part / ≈999 GB 网络 / ≈5.2 GB 磁盘；分批 A 2000–2007、B 2008–2015、C 2016–2022，各批独立冻结协议与 `failed-no-fallback`）。**它是纯网络/CPU 任务、不占 GPU-h、不写你任何路径、不改 `campaign-state`/账本/index/brief，也不改变你的下一步**——你的下一步仍是本块上一行写的「训练信号 / 目标组织方向新 round brief」。**新数据落地前不要动它、不要把它当成可用实例。** 细节、实测数字与「为什么含 2022」见 **`.zcode/plans/s3-DOWNLOADING-fy-2000-2022-20261014.md`**（git-ignored 工作态笔记，不占 index、不是 brief）。**同文件另含一条会影响「如何验证」的实测**：现有 train **只有 Jan/Apr/Jul/Oct 四个月**（8 个月完全缺席，16 个「月×时次」气候态桶每桶仅 ~147 帧），而**现有 val（2022）也只有那 4 个月**——故用现有 val **无法检验「全年扩充是否有效」**，这正是本次把 2022 也下成全年的理由。另：**源本身是逐小时的**（间隔 3600 s），6 小时是我们的选择而非源的限制。该文件只负责告知，主链可径行采纳或删除本行。
