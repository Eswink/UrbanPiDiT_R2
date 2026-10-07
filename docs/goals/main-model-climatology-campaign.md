# 主模型超气候态 campaign：持续自主探索与独立年份/四季确认

<!-- round-node: S3 -->

**状态：active（2026-10-06；S0–S2 已登记，S3 开发筛选与长 lead 机制研究进行中，S4 未启动）。**
本文件是新方向的**唯一主计划与 goal 长文**。科学合同见
`docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`；执行授权见决策 0030/0038；实施交接见计划 0016。
旧 V2 收尾与 #70–#75 不重做，不自裁旧/新 goal complete。以后各具体实验长文由本主计划派生。

## §0 Objective（可粘贴；单段，实测 3974 字符）

> 你在 /data/esw/UrbanPiDiT_R2 的 r7/weather-reasoning 分支执行新的长期主模型研究goal，先通读docs/goals/main-model-climatology-campaign.md、docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md、最新自主研究/数据扩围ADR及#76–#79的正文与评论，核HEAD/status与既有证据。本次是实际端到端研究，不是交计划后等待我。总体方向是沿UrbanPiDiT可思考/递归主模型持续迭代优化，直到按预先冻结的公平协议稳定超过train-only climatology并完成独立确认；旧#70–#75已结题，不重做已完成阶段或把关闭/CI当科学成功。我明确把此方向内的实验设计、架构/方法/训练策略选择、假设取舍、数据源及区域/时间/变量选择、免费真实数据下载、preflight审阅与新路径派生发布--write、本地GPU训练评估、种子/更新数、每次实验和下载时长预算、普通工程决策、故障排查、下一实验及节点推进权全部下放给你。无需逐项询问、逐轮等我触发；时长与GPU-h可自行增加、记账并简要通知，通知不是等待批准，没有总GPU-h上限，但不无限重试或加算力练到赢。付费数据/租卡、GPU独占、改变总体研究方向、main合并和破坏性操作仍须额外具名许可；#76–#79的GitHub关闭/main写入不从旧六issue授权自动继承，不以全关作为本goal完成条件。D1先核actual C的真实incumbent配置/代码/数据/checkpoint身份，重建同case/变量/lead/单位/区域的模型、当前train-only climatology及persistence差距表，旧分数只在身份兼容时引用。同图matched_generic用于等价性核验，不当必须击败的竞争baseline；性能改善与过程机制归因分开，不把package相对old_ours的进步说成超过气候态。新验收门：t2m/full的6/12/24/48/72h，至少3预声明seed，每seed在确认总集、每预注册年及四季组的MSE skill=1−MSE模型/MSE气候态均>0，seed均值的全部primary同时区间下界>0；u10/v10/mslp对同数据incumbent在相同组/lead每seed相对MSE变化≤0、同时区间上界≤0，容忍0，undefined不通过。至少一个完整真正未见年份及四季，配对时间块bootstrap保留病例/lead/seed相关性；确认r自1累加，首次test标签评分即占r，partial也计，alpha_r=0.05/(r*(r+1))、总≤0.05（首轮97.5%同时区间），不可换名重置。全部17变量×6/12/24/48/72h、全域/边界与坏变量照报，不事后换primary、挑赢格子或直接平均不同物理单位。D2参考#76的P0→#77→#78顺序，#79可并行设计；issue与GPT6PRO建议是待验证假设，不是权威结论。#77先测实际激活路径和有限PE频带，单因素控制；#78先train-only连续6h变化尺度重参数化解码、保持loss，再另轮改多变量权重，normalized误差换算用s_c/d_c，不把物理尺度直接除进归一化误差；#79验证初始/自生成草稿诊断实际进入前向，首轮aux=0，当前模型/同信息无类型融合/类型路由三臂，只有后两者可归因类型偏置，不再重复同图无语义控制或无效aux扫参。旧negative/failed/paused保持；新机制用独立协议与输出，不能以全权探索绕原停止线，也不能让已否定单假设冻结其他独立主线。每次新试都说明可反驳机制、与旧试实质差异、信息收益和回主线动作。D3按数据支持需要分批扩到更长训练窗口、全年季节与独立年份，不一次拉全球全集。允许免费合法公开的区域多年度、多季节数据；准备期冻结source/snapshot/license、区域/分辨率/变量/单位/时间、split、解码/网络字节预算、磁盘峰值、planned/hard时长与恢复策略，先小pilot核真实速率和schema再扩。复用已有下载器和prepare_r7_local只读preflight；你自行核报告并留审阅记录后可向全新outputs路径--write建store，强核完整source hash、单位/坐标/有限性/BUILD_COMPLETE及派生身份，不写改旧data/raw|interim|processed或已有归档。网络慢/中断自主诊断、分片、校验重用或合法备用源；不能宣称未实现的续传，失败保留failed-no-fallback与费用，不合成替代，不删旧部分产物复活。数据改变就重新训练同样信息/统计/样本下的气候态和必要控制，不能混用旧8桶弱气候态；归一化、变化尺度、proxy统计、climatology只fit train，所有跨split历史/目标窗口剔除并留足隔离，新test真正未见，旧曝光test不重新封存。D4用train/val选型，先低成本解析/CPU接线和有界小规模真实训练，证据支持再扩数据/训练；更新数、计算预算/参数/输入信息分别对表，不把同updates称同算力。选择最终候选后冻结配置、checkpoint、baseline、cases/端点/统计法，再一次进入未见test并做跨年/季节/seed确认；若不达门，保留失败与曝光记录，返回独立开发方案，后续确认须新冻结并控制多重探索，不反复看test练到赢。无收益不交差；主动检索、定位信号/优化/前向瓶颈和数据制度，开展不同可证伪实验；不扩成全baseline竞赛或改叫别的大模型来冒充主线。D5明确调度子代理与技能：扫仓/调用路径/证据位置用Explore，多步依赖/大改动用planner配合planner-delegation并校验JSON，科学阈值由主链按协议预注册不外委；外部论文/官方源码/API用web-researcher配合web-research，不可用切web-researcher-backup，再不可用才curl留因，一手核查URL/访问日/版本，不拿摘要当证据。需要公开源码可联网准备期git clone到outputs/reference_sources/<project>/<commit>/，固定commit/license/hash，提交引用台账，注明实际借鉴文件/改写与未搬范围；归档只读，不直接import或执行上游安装训练脚本。general-purpose承担独立只读证据复核或隔离定向实现，分配不冲突文件；并行资料/CPU准备，不重复主链已委派搜索，GPU并发由统一父调度核余量。维护goal/节点交接用goal-loop，长期授权/契约用decision-record，issue推进/验收用issue-lifecycle，新增/派生数据用real-data-acquisition，缺依赖/CUDA异常用environment-rebuild，有界CPU实验用bounded-study-run（不冒充GPU通用SOP），历史产物重放用pinned-artifact-replay，结果引用/冻结用result-freeze，CI失败/pending/取消/精确SHA核对用ci-workflow-triage；每逢触发先实际加载对应技能，不能只列名称。D6每实验先冻结protocol/digest、范围/产物/失败skip处理、planned_seconds软预算与hard_cap_seconds宽松硬上限，整轮计时含CPU前置/启动间隔/训练/评估/清理；软超继续记overrun，硬截止或真实错误停止该attempt并全额记账，下一独立协议可自主增加时长，不运行中改已冻hard cap。下载也有分批数字预算和磁盘余量门，空间不足自主减少本次范围/分片，不删历史数据或干扰其他作业；无授权租卡/收费不允许。GPU默认共驻，每启动/spawn只读核UUID/余量，禁止signal邻居；准备可联网，实验R028离线，训练评估真实数据身份不绕过。每节点前后对表、实跑conventions/campaign/goal结构、定向反证及全量测试、index/brief/空白与精确SHA CI；skip/cancelled/queued/partial不算通过，科学比较不由CI代替。记录全部假设、源码/数据/协议/结果digest、负面与累计GPU-h/网络/磁盘/墙钟，scientific_claim:false和limitations保留。普通工程、网络、身份、环境、统计或CI阻塞由你自主查因、修合法活跃代码、独立复核后继续，不以“有阻塞/负结果，等你决定”结束；只触保留授权/真实不可安全克服的外部依赖才停相关动作并报告，其余独立工作继续。禁止改冻结判据、伪造PASS、删弱测试、降安全门/改用户配置/绕hook、force/mirror、凭据外发、cron/常驻守护或会话外自建续跑；上下文压缩从进度恢复，会话终止留可审计交接。只有协议定义的气候态超越、独立确认、科学限制及可复核证据齐备才提请最终goal验收，不保证一定成功、不自行宣告SOTA或complete。结束一次性汇报模型与数据改动、子代理/技能实际使用、全部实验/负面、验证与精确CI、接口/安全凭据/兼容依赖影响、成本/overrun、issue状态与真正未做项；不要把单次局部改善当作终极目标完成。

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

GPU研究按0030/0038、GPU资源规则与该次协议，不虚构不存在的通用GPU技能。新能力需要重复事实后
才独立立SOP，不为本轮多造平台。#76–#79建议不是固定必须跑全部格子的matrix。

## §5 时长、下载、停止与恢复

- 文档交付0GPU-h；**S0/S1尚无具体运行协议**。执行窗口按旧实测/新pilot自定每次
  `planned_seconds` 和 `hard_cap_seconds` 两个数字（通常约2倍），长文和protocol中写定再启动。
  时长计CPU前置/启动间隔/训练/评估/聚合/清理；超软继续记overrun，仅硬截止因时长截断。
- 无总GPU-h上限；cap/used/remaining是兼容会计字段。计划额度耗尽不单独停止；真实成本/说明/简要
  通知必留，不把软预算用来关闭attempt。新时长用下一独立协议，不修改原已冻硬截止。
- 下载每批source/snapshot/license/变量/区域/时间/half-open split、decoded/network预算、磁盘峰值、
  free-space门、软硬秒数、重试/分片恢复写定；先小pilot。仅hash/身份核齐可重用完成分片，不假称续传。
- 准备期可联网，实验R028离线；默认GPU共驻，启动/spawn只读UUID/余量，不能signal邻居。等待余量时
  继续合法CPU工作或安排新符合余量的协议，不降门/抢卡。承载CI timeout须大于hard+收尾余量。
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
| **合计已用** | **19.1089** | — | 本方向基数0，历史V2不重复计费，所有失败全额登记；cap20.0/remaining0.8911是会计字段，不是总GPU-h许可上限（0030/0038）；原cap16/旧累计不回改。长监督200剂量已完整且只开发门清零，正式气候态/全年/同时区间未过；同case失败亦见train，下个独立train信号诊断另冻，不无限重复同剂量 |

新账本每个失败和成功都加实际连续GPU/执行口径及证据record；其他网络/decoded/disk/whole/overrun在
各回执分列；无index支撑的文档0成本行如实列note，不伪装成实验机器核数。

## §8 进度块

<!-- campaign-state: {"current_node": "S3", "previous_node": "S2", "current_round_goal": "docs/goals/s3-same-case-gap-diagnostic.md", "previous_round_goal": "docs/goals/s2-climatology-mechanism-screening.md", "previous_round_evidence": "docs/R7_S2_79_TYPED_EVIDENCE.md", "cap_gpu_h": 20.0, "used_gpu_h": 19.1089, "remaining_gpu_h": 0.8911, "status": "active", "next_node_proposal": "S4", "budget_mode": "per-node-hard-cap-summed", "route_decision": "0038"} -->

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
  下一独立S3问题先检查固定长权重下train损失分量/梯度方向，不凭norm>1盲目增加剂量或改clip。
  600条已保存norm全部clip前>1、四个loss块非同case不判收敛，不据此无界加预算。
  不重开短双步、不消耗未见test、不假称完整全年或科学complete。
  v3-D2/D3已登记，不重复重建；S4/test未启动，现有四季30日块不等于完整未见全年确认。
