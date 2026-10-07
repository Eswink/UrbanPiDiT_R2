# 主模型超气候态 campaign：持续自主探索与独立年份/四季确认

<!-- round-node: S3 -->

**状态：active（2026-10-07；S0–S2 已登记，S3 固定病例响应诊断及证据收尾，S4 未启动）。**
本文件是新方向的**唯一主计划与 goal 长文**。科学合同见
`docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`；执行授权见决策 0030/0038；实施交接见计划 0016。
旧 V2 收尾与 #70–#75 不重做，不自裁旧/新 goal complete。以后各具体实验长文由本主计划派生。

## §0 Objective（可粘贴；单段，实测 3944 字符）

> 在/data/esw/UrbanPiDiT_R2的r7/weather-reasoning分支继续UrbanPiDiT可思考/递归主模型长期研究。先加载goal-loop，读docs/goals/main-model-climatology-campaign.md、docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md、决策0030/0038/0039和最新§8，核HEAD/status、代码/数据/checkpoint身份及issue正文评论。这是实际端到端研究，不是交计划后等我；按最新证据恢复。当前S3/index60/累计21.4177GPU-h，2023test未评分、r0，启动时复核；不重启已登记S0–S2、旧#70–#75、单病例80更新或精确反算。cap20/remaining−1.4177仅会计非授权限。目标是持续优化本主模型，公平稳定超过train-only climatology并独立确认；工程绿、单例loss下降和关闭issue非科学成功。我明确下放方向内实验设计、架构/方法/训练信号/优化/信息路由、数据源/区域/年份/变量/分辨率、免费合法真实下载、自审preflight与新outputs发布--write、本地GPU训练评估、seed/更新/规模、每次时长预算、普通决策/排障、下一实验及节点推进权。不逐项询问、不逐轮等我；时长/GPU-h自主增加、记账并简报，通知不是等批准，无总GPU-h上限。付费/租卡、GPU独占、方向改变、main合并/发版和破坏性仍须具名许可；新issue关闭/main写入不继承旧六issue许可。D1先用既有0/20/80端点、原四季train病例/四val开发例/同train气候态，冻结无新optimizer或更新的原deep-K目标与最终17变量×五lead物理评分对应诊断。核归档code.zip/新80contract/普通loader/案例/单位/CPU反证，不重训练或失败replay，正负登记，不以小样本称泛化。目标降而最终输出不改善，查监督聚合、草稿/最终头及评分路径；只同病例好，查记忆、跨例干扰和数据制度；开发例有支持再扩完整val，不默认加剂量。D2据#76–#79及后续相关issue正文评论、现有正负证据提可反驳假设；GPT6PRO建议仅参考，旧PE/scale/typed无理由不重跑，issue顺序不是永久固定matrix。允许主模型架构、表达、训练目标/优化、可推理诊断信息组织和数据制度探索；每试写实质差异、正负控制、预期信息、新protocol/输出、停止/饱和出口及回主线动作。先解析/CPU接线与小规模真实train/val，支持后扩；性能与机制归因分开，matched_generic是同图等价控制非必须赢baseline，typed归因须同信息无类型/有类型对照，future真值仅监督不进前向。无收益主动查官方论文/源码、梯度信号/最终预测/优化/数据瓶颈，转不同可证伪假设；不无限调开关、seed或训练量，不换外部大模型冒充主线。已否定假设停，旧negative/failed/paused保持，独立主线继续。D3允许分批下载免费合法公开区域天气数据，含更长训练期、多年度/季节和独立确认年；先核已有完整源复用，不默认全球全集。每批先冻sourceURL/snapshot/license、范围/单位/网格/half-open split、网络/解码字节/磁盘峰值余量、planned/hard时长和重试恢复策略，pilot测速/schema再扩。复用带预算下载器，慢网自主排障、分片、核hash复用完成片或换合法真实源；不假称续传、不合成回退，失败留failed-no-fallback。自审prepare_r7_local只读preflight留digest/逐项结论后才--write至全新非嵌套排他outputs，完整sourceSHA256/单位/schema/finite/sidecar/BUILD_COMPLETE不省。旧data/raw|interim|processed、store、归档不改，不删部分输出复活失败。统计只fit train，跨split历史/目标窗剔除隔离；数据变公平重建气候态及必要incumbent，不能拿新模型打旧弱基线。准备联网、实验R028离线；下载慢时并行独立CPU/资料，不混边界。D4用train/val选架构/K/loss/checkpoint，分报信息、样本、updates、参数、FLOPs/GPU-h，不把同updates当同算力。最终配置/checkpoint/baseline/cases/split、统计法/块长/重采样/RNG/seed及数字预算全锁后才评分新test。验收固定t2m/full的6/12/24/48/72h：至少3预声明seed，各seed在确认总集、每预注册年及四季组的skill=1−MSE模型/MSE气候态均>0，seed均值全部primary同时区间下界>0；u10/v10/mslp对同数据incumbent在同组/lead每seed相对MSE变化≤0且同时区间上界≤0，容忍0，undefined不通过。至少一个完整真正未见年份及四季，配对分层时间块bootstrap共同重采样病例/变量/lead/seed，不把像素、重叠窗或seed当独立天气样本；不足块不能过门。首次test标签评分消耗确认r，partial也计，alpha_r=0.05/(r*(r+1))、累计≤0.05、首轮97.5%同时区间；失败留曝光/全部分组，不换名重置、不复封旧test、不反复看test练到赢。全17变量×五lead×full/interior/edge、物理RMSE/MSEskill/ACC及坏变量照报，不跨单位平均或挑赢家。未达门返回独立开发，门过封印证据并提请验收，不保证必胜、不自宣SOTA或complete。D5实际调度代理与技能：未知路径/调用链/证据扫仓用Explore并给范围，主链不重复搜索；多步依赖/架构/大改用planner配planner-delegation，JSON过check_planner_plan及事实核，科学门只引用不外委；服务端/额度失败留原错，自规划过同契约并独立内容审阅，不算委派完成也不停整条研究。外部论文/官方源码/API/报错用web-researcher配web-research，不可用切web-researcher-backup，仍失败才curl留因；一手核URL/访问日/版本，不拿摘要作证据。准备期可git clone源码至outputs/reference_sources/<project>/<commit>/，锁commit/license/hash，留官方引用、借鉴文件/改写与未搬范围；镜像只读隔离，不直接import或运行上游安装训练。general-purpose做无冲突定向实现、候选升级/最终独立只读复核，不代最终verifier。资料/CPU可并行，GPU统一父调度，每启动/spawn核共驻UUID/余量，不signal邻居。命中先实际加载：goal/交接用goal-loop，授权/长期契约用decision-record，issue推进/验收用issue-lifecycle，新数据/派生发布用real-data-acquisition，环境/缺依赖/CUDA异常用environment-rebuild，有界CPU用bounded-study-run（非GPU通用SOP），历史重放用pinned-artifact-replay，引用/归档用result-freeze，CI失败/pending/取消/精确SHA用ci-workflow-triage；不能只列名字。D6每运行先冻protocol/digest、范围/失败处理、planned_seconds软预算/hard_cap_seconds宽硬限与收尾余量，整轮含准备/启动/训练/评分/核验/聚合/清理。软超继续记overrun，硬截止/真错停attempt全额留失败，下一独立protocol可加时，不运行中改硬限或重试到偶然通过。复用先核合法pins，不以反复全套复验代研究；节点前后核campaign/goal、身份/账本/前置出口/独立审阅，实跑定向反证，源码变跑完整回归，登记index/brief/空白与工作分支精确SHA必要CI；skip/cancel/queued/partial非PASS。登记全部假设/负面、code/data/protocol/resultsdigest、复现等级、scientific_claim:false/limitations与GPU-h/网络/解码/磁盘/墙钟/overrun分口径成本。普通环境/网络/工程/身份/统计/CI阻塞自主查因、合法修活跃代码、独立复核后继续，不以“负结果/有阻塞等你决定”交差；仅保留权限或无法安全克服的外部硬依赖停相关动作报告，其余继续。禁止改冻结门、伪造PASS、删弱测试、绕身份、安全降门/改用户配置/绕hook、force/mirror、凭据外发、cron/守护或会话外续跑；上下文压缩按最新进度恢复，会话终止留可审计交接。结束汇报实际模型/数据改动、全部实验/失败、代理技能使用、科学/机制限制、验证/精确CI、接口/安全凭据/兼容依赖影响、全部成本/issue状态及未做项；不要自行宣布最终goal完成。

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

GPU研究按0030/0038、GPU资源规则与该次协议，不虚构不存在的通用GPU技能。新能力需要重复事实后
才独立立SOP，不为本轮多造平台。#76–#79建议不是固定必须跑全部格子的matrix。

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
| **合计已用** | **21.4177** | — | 本方向基数0，历史V2不重复计费，所有失败与测试缺口费用全额登记；cap20.0/remaining−1.4177是会计字段，不是总GPU-h许可上限（0030/0038）；原cap16/旧累计不回改。科学气候态/全年/同时区间仍未过；固定病例原目标有限响应已确认但非泛化/clip/欠拟合因果证明，不无限重复同剂量或精确replay |

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

<!-- campaign-state: {"current_node": "S3", "previous_node": "S3", "current_round_goal": "docs/goals/s3-objective-forecast-score-diagnostic.md", "previous_round_goal": "docs/goals/s3-fixed-case-objective-response.md", "previous_round_evidence": "docs/R7_S3_FIXED_CASE_OBJECTIVE_RESPONSE.md", "cap_gpu_h": 20.0, "used_gpu_h": 21.4177, "remaining_gpu_h": -1.4177, "status": "active", "next_node_proposal": "S4", "budget_mode": "per-node-hard-cap-summed", "route_decision": "0038"} -->

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
- 此时无新optimizer/update/backward/天气字段评分/GPU-h，不重复已登记80或精确replay。
  实际新诊断软3600/硬10800/reserve180秒待exact包CPU资格与独立prelaunch后冻结排他protocol。
  **下一动作（S3）**：完成新薄wrapperCPU反证/独立资格，首次实跑0/20/80×原八病例的原deep-K目标及
  最终全17×5×full/interior_1/edge_1物理评分，正负登记后按证据继续不同可证伪主模型假设。
