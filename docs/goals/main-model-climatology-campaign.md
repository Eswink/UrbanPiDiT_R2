# 主模型超气候态 campaign：持续自主探索与独立年份/四季确认

<!-- round-node: S1 -->

**状态：active（S0 已完成并登记，2026-10-05；进入 S1 数据 pilot 与候选解析设计）。**
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
| **合计已用** | **1.0257** | — | 本方向起始会计基数0.0000；历史V2的9.9139保留在旧master，不复制重复计费 |

新账本每个失败和成功都加实际连续GPU/执行口径及证据record；其他网络/decoded/disk/whole/overrun在
各回执分列；无index支撑的文档0成本行如实列note，不伪装成实验机器核数。

## §8 进度块

<!-- campaign-state: {"current_node": "S2", "previous_node": "S1", "current_round_goal": "docs/goals/s2-climatology-mechanism-screening.md", "previous_round_goal": "docs/goals/main-model-climatology-campaign.md", "previous_round_evidence": "docs/R7_S1_FOUR_SEASON_ACQUISITION.md", "cap_gpu_h": 6.0, "used_gpu_h": 1.0257, "remaining_gpu_h": 4.9743, "status": "active", "next_node_proposal": "S2", "budget_mode": "accounting-only", "route_decision": "0038"} -->

- **状态**：active；S0 已完成并登记；S1 数据侧已完成（2026-10-05，0 GPU）：四季节 2017 区域段已
  下载、合并、发布为 dev store，并登记进账本与证据索引；#77 频带接线与 11 项反证已随 62414bd 过
  CI；进入 S2 单因素机制筛选（#77 → #78）。
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
- **下一动作（S2 续）**：#79 有类型局部诊断证据三臂设计已冻结
  （`docs/R7_S2_79_TYPED_DIAGNOSTICS_DESIGN.md`，首轮 aux=0、B→C 才归因 typed 结构）；
  实现顺序为先 CPU 反证测试与逐类到达探针，再写 protocol.json 后上 GPU。
  dev store 上只做筛选，不产生科学声明；正式确认门在未见年份实例上判定。
