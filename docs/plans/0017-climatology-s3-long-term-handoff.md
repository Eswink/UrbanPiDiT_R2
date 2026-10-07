# 0017 主模型超气候态：从真实 S3 接续的长期 goal 交接

**状态：文档交付与本地验证完成（2026-10-07）；本轮科研未执行、未提交推送。**
起点 `d892e20d93afc1a4fb54006300cadfe887d5485d`，分支 `r7/weather-reasoning`。
唯一主计划与可复制 objective：`docs/goals/main-model-climatology-campaign.md`；本计划不是第二个 master。

## 1. 本轮范围与真实起点

用户要求持续自主优化 UrbanPiDiT 可思考/递归主模型，直到公平稳定超过 train-only climatology，允许
继续获取真实数据、增加实验时长、主动排障与文献/源码检索，并要求显式说明子代理及技能使用时机。
本轮落实为四份文档交付；**不下载、不发布 store、不 clone 源码、不训练、不评分天气、不查询 GPU，
不暂存/commit/push，不修改用户配置，不关闭 issue 或写 main**。

已核起点事实：

- S0–S2 已完成并登记；当前 active/S3，S4 未启动。
- v3 数据实例 train2017–2021 / val2022 / test2023 已发布；现有四季时间块不代表完整未见全年。
- 固定 January train 病例恰80更新与0/20/80精确反算已登记 `audit / not-candidate`，不是候选或科学接受。
- 索引60条，旧58条前缀逐字节保持；本轮之前新增训练0.4300/反算0.4078，共0.8378GPU-h，累计21.4177。
  cap20.0 / remaining−1.4177 仅会计字段，0030/0038没有总GPU-h授权上限。
- 冻结证据 `docs/R7_S3_FIXED_CASE_OBJECTIVE_RESPONSE.md` SHA256
  `fbfd83ee84ceecdcaac8f93250e564fde9e0ac57c3d71f87b0ae4cb3f68e0742`，freeze commit6b2e3db。
- 登记5b72e56/CI37606126942和交接d892e20/CI37607651646的精确SHA、pytest job及全部12steps success
  已在本会话只读核验；这是既有工程事实，不覆盖本轮未提交文档，不升级科学结论。
- 2023 test尚未科学评分，confirmation r=0；科学主门、同时区间与最终goal接受未完成。

现有旧objective仍要求从S0重建gap并重走初始issue顺序，§5仍写S0/S1无运行协议，索引停在早期v2。
本轮替换过时活入口并追加前瞻交接，历史阶段的当时记录保留；不能据旧“未执行”文字重做已登记工作。

## 2. 授权继承与边界

沿决策0030/0038/0039：方向内方法/架构、训练信号、优化、信息组织、实验设计、数据源/区域/年份/
变量/分辨率、免费合法区域真实下载、自审preflight后向全新非嵌套排他outputs路径--write、本地训练
评估、种子/更新/规模、逐次预算、普通排障、下一实验与节点推进自主决定，不逐轮问人。

无总GPU-h上限，通知与记账不构成等批准。每次先冻planned_seconds软预算、hard_cap_seconds宽硬限
及收尾余量，计整轮；软超继续记overrun，硬截止/真错误停该attempt，全额保留失败。下一独立协议可
增加时长，不运行中延长旧硬限，不追认超时/取消/partial阶段为通过。

免费合法公开区域数据可多年度、多季节；先核旧完整源复用，pilot后分批扩围。源身份/许可、单位/schema、
网络/解码/磁盘预算与恢复策略先冻结；完整sourceSHA256、finite、sidecar、BUILD_COMPLETE不省。
失败留下failed-no-fallback，不合成，不删除部分输出复活。统计只fit train，跨split窗口剔除并隔离。

付费/租卡、GPU独占、总体方向改变、main合并/发版、破坏性仍需具名许可；新issue关闭/main写入不继承
旧六issue许可。受保护数据、旧store、归档与已登记产物只读；不改安全门/用户配置/凭据，不force/mirror，
不signal邻居，不自建cron/守护或会话结束后续跑。原授权充分，本轮不新增ADR或平行SOP。

## 3. 从 S3 到终极确认的研究路径

### D1 当前最有信息价值的目标—最终评分诊断

执行窗口先核最新进度；若该问题已有合法新结果，应核身份并复用，不按旧提示重复运行。尚未执行时，
沿当前交接冻结无新optimizer或训练更新的审计：既有0/20/80端点，原四季train病例、四val开发例，
同train气候态，原deep-K目标与最终17变量×五lead物理评分。先归档code.zip/新80contract/普通loader/
案例/单位资格与CPU反证，再执行。小样本只定位，不作为泛化或候选通过。

| 观察 | 后续自主研究分支 |
| --- | --- |
| 原目标降但最终输出未改善 | 检查监督聚合、草稿/最终预测头与评分路径，不直接加剂量 |
| 只同病例改善 | 检查局部记忆、跨例干扰与数据制度，保留其他季节/val坏结果 |
| 开发例存在同向支持 | 预注册完整val验证，不直接解封test或升级S4 |

### D2 可反驳探索而非永久 issue 矩阵

#76–#79的历史提案是来源而非科学权威。S2已有PE/scale/typed真实正负结果，按证据复用；新架构、状态
表达、训练目标/优化、可推理诊断信息组织与数据制度均可在主模型方向内自主设计。每次列实质差异、
阳性/负控制、预期新增信息、新protocol/输出、停止/饱和出口和回主线动作。

先解析/CPU接线与有界真实train/val，小试有支持再扩大。连续无收益主动查一手论文/官方源码和训练
信号/最终预测/优化/数据瓶颈，停止已否定单假设，不冻结全部独立主线；也不无限开关/seed/加量至赢。
同图matched_generic只作等价控制；typed机制归因须同信息无类型/有类型对照，性能与机制归因分写。

### D3 数据和参考源码

按科学需要继续真实区域数据分批获取并公平重建同数据气候态及必要incumbent，不能新模型打旧弱基线。
准备期可联网，实验R028离线。慢网时继续独立CPU/资料工作，不宣称未实现的续传，不混网络边界。

必要公开官方源码可clone至 `outputs/reference_sources/<project>/<commit>/`，固定URL/访问日/commit/
license/hash和借鉴文件/改写/未搬范围。镜像隔离只读，不直接import或执行上游安装训练；合法移植仍需
许可证、测试与新模型身份，不把外部大模型改名冒充主线。

### D4 未见年份/四季独立确认

科学成功只按 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`，不修改该文件或事后放宽门。
objective内直接携带可核条目：t2m/full五lead，3+预声明seed各seed/确认总集/各年/四季正MSEskill，
seed均值全部primary同时下界>0；u10/v10/mslp同组lead对同数据incumbent每seed相对MSE变化≤0，
同时上界≤0，容忍0、undefined不通过；完整真正未见年份与四季、配对分层时间块bootstrap，块长等
在test前冻结，不把格点/窗口/seed当独立天气。确认r首次标签评分消耗，partial也计，
alpha_r=0.05/(r*(r+1))、总≤0.05、首轮同时97.5%，失败不复封/重置r/重复test练到赢。

train/val选型，配置/checkpoint/baseline/cases/统计/seed及数字预算全锁后才test。全17变量×五lead×
full/interior/edge与坏变量照报；未过返回独立开发，过门封印证据并提请独立最终验收，不保证必胜、
不自行宣告SOTA或complete。工程资格、开发支持、独立越过与思考机制归因分别判断。

## 4. 子代理与技能触发（实际调用，不只列名）

| 代理 | 何时使用与边界 |
| --- | --- |
| Explore | 未知路径/调用链/证据定位；只读有界，不重复主链已委派搜索 |
| planner | 多步依赖/架构/大改风险；JSON契约校验及事实复核，科学门仅引用不外委 |
| web-researcher | 外部论文/官方源码/API/报错事实，一手URL/访问日/版本 |
| web-researcher-backup | 主研究不可用同任务接续；仍失败才curl留因 |
| general-purpose | 无冲突定向实现，候选升级/最终独立只读证据复核，不代最终verifier |

planner服务端/额度失败保留原错，自规划过同契约并独立内容审阅；不伪称委派成功、不停止整条研究。
资料/CPU可并行；GPU由统一父调度共驻余量门，每启动/spawn只读核UUID/余量，禁代理各自抢卡。

| 技能 | 命中时先加载 |
| --- | --- |
| goal-loop | goal、节点、上下文交接或harness降级 |
| planner-delegation | 多步方案、架构风险与planner JSON |
| decision-record | 授权、长期方法/契约决策 |
| issue-lifecycle | issue推进/验收，工程与科研区分 |
| real-data-acquisition | 新真实源/下载器/派生发布 |
| environment-rebuild | 环境、缺依赖、CUDA异常 |
| web-research | 外部一手事实及引用 |
| bounded-study-run | 有界CPU实验，不是GPU通用SOP |
| pinned-artifact-replay | 旧产物/归档代码重放 |
| result-freeze | 结果引用、归档、digest与复现等级 |
| ci-workflow-triage | CI失败/pending/取消/精确SHA与必要steps |

## 5. 四文档改动面与验证

- 主计划：§0从S3接续单段objective；§4前瞻分支；§5修过时预算状态、收尾余量和风险导向复验；
  §8追加本轮0GPU交接。机器state/账本与旧进度原字节保留，下一子goal/protocol由执行窗另冻。
- 新本计划；plans/README登记0017；goals/README同步v3/S3与固定病例入口，梯度诊断终态按证据更新。
- 不改科学合同、旧计划/ADR、证据页/index/brief、代码/测试/依赖、安全配置和用户无关改动。

实际验证范围：goal单段/自指针/≤4000codepoints、D1–D6/5代理/11skills/科学门；默认旧campaign与新主计划
显式--campaign；37阻断及--paths四路径模拟，不暂存；index/brief60与原字节；引用/空白/编号；四完整
CPU模块test_check_goal_brief/test_check_campaign_state/test_check_conventions/test_verify_r7_evidence_index。
使用venv -B、socket出网拒绝、CUDA隐藏、仓外tmp、无cache；此测试整轮软600/硬1200秒，不预写PASS。
独立只读审阅有必修就修后复核。未跑全量suite/wheel/新remoteCI，不引用既有4003pass为本轮实跑。

本轮只读身份基线 `/tmp/r7_s3_goal_handoff_20261007_b7xbl09b/baseline.json` 保存13个只读文件hash/bytes、
master state和账本hash。新增治理文档未提交时干净克隆无0017，不能冒称版本化或新CI。

## 6. 规划来源与纠错

本方案来自两项Explore侦察与planner委派；planner草稿不是证据。原草稿objective超400、把提示词4000
限制误作UTF8字节、错记本轮新费用1.4710、引用不存在的索引/工具/测试、用环境变量冒称禁网、把本轮
实际结果留后续窗口填等均不采用。主链按真实身份/0.8378费用/既有校检工具修订；纯内存validate_plan
返回verified=true，无文件写入。主链首次只读Python调用被plan guard拒绝，未执行且不算通过。

原提示词草稿4057codepoints超限，独立内存压缩候选3700；主链进一步改善可读性后实测3944，仍单段。
该长度不是目标长文上限。落盘版必须再核全部科学门及名称，不能以压缩删掉独立验收要求。

当前open问题一手列表于2026-10-07经web-researcher并回原JSON核到#76–#79，未见更高open编号；
这不是“无更高closed问题”的证明，不把旧issue建议的TODO替代本仓实际登记：
[open issues API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/issues?state=open&sort=created&direction=desc&per_page=20)。
#76/comment与#78单位评论的来源已在科学合同登记；本轮不另检索论文或clone源码。

## 7. 风险与停止边界

- 旧入口重做已完工作：从最新§8和身份恢复，已有合法结果先核后复用；state不虚跳S4。
- 压缩漏科学门/代理技能：机械长度与显式清单、独立内容审阅并行，只有结构通过不足以科学接受。
- 持续探索成本与多重选择：每协议数字预算与停止/饱和出口，test曝光/r-alpha和全部负面保留。
- 排障反复吞掉研究：复用已核合法pins，按改动风险验证；source变化全回归，必要GPU回归另冻共驻协议，
  不把CPU检查当GPU接受，不无限全套复验代实际科学问题。
- 普通文档/验证错误自主修；触保留权限或无法安全解释的冻结证据矛盾停相关动作，不编造或降门。

## 实际结果

- **完成情况**：四文档交付已完成：唯一master的§0/前瞻路线/预算说明/零GPU交接、本计划、两索引。
  修正旧S0/S1和gradient preparation过时活入口；§0接续D1–D6与§2历史交付编号区别明确，已完S0不重开。
  既定科学门、机器state、账本和历史进度原字节保持；科研/下载/发布/源码clone/GPU查询未执行。
- **与计划的差异**：可读objective3944而非内存压缩候选3700，单段≤4000且与落盘§0精确相同；科学门不改。
  四文件边界未扩，新计划0017可用；不新建ADR/checker/能力或未来实验protocol。
- **实际验证**：两个goal结构0failure/0advisory；旧/新campaign均0failure/4既有notes，notes不冒核数。
  37阻断0失败，四交付路径--paths模拟0失败且未暂存；60条index/canonical brief和git diff --check通过。
  四完整CPU模块test_check_goal_brief/test_check_campaign_state/test_check_conventions/test_verify_r7_evidence_index
  实跑 **213passed/0failed/0error/0skip**，整轮39.204876秒，planned600/hard1200、softoverrun0。
  项目venv -B，pytest -q -p no:cacheprovider、插件autoload关闭、测试进程socket拒绝出网/CUDA隐藏、
  全部临时产物仓外；未过滤模块内用例。不是天气/GPU验收或本轮fullsuite/remoteCI。
- **回执**：`/tmp/r7_s3_goal_handoff_20261007_b7xbl09b/cpu_verification_ihmcnyb9/verification_receipt.json`，
  SHA256 `2e3c8e2ab782fcf12628a53b65e32c0a46feef4a8b32bfb28795000f8d665ba1`，内含JUnit/log摘要、
  模块、时间/预算/费用及limitations；本地tmp回执不是新科学索引record。
- **只读身份**：13/13预存文件hash/bytes一致，含科学合同、固定病例证据、index/brief、旧0016/三ADR、
  两历史goal及三用户无关文件。master machine state/完整§7账本与旧§8全文前缀精确保持，§1/§2/§3/§6
  原文不改。HEAD仍d892e20d93afc1a4fb54006300cadfe887d5485d，无暂存/commit/push；累计21.4177不加费用。
- **独立审阅与实际调度**：Explore完成状态/改动面定位，planner原草稿内容缺陷拒绝后主链修订并纯内存
  校契约；web-researcher获取issue一手资料并回原JSON核。general-purpose隔离修改唯一master，另一个
  general-purpose独立读四项内容审阅无必修；主链实际核落盘提示词/历史prefix/全部身份，审阅不代最终verifier。
  实际使用goal-loop/planner-delegation及本会话相关检索/核验技能；11项表为将来命中触发，不伪称本轮全使用。
- **影响与未做**：仅文档活入口和索引变化，公开接口/默认参数/模型digest、数据/结果、依赖、安全hook/
  配置/凭据均无改动。未跑fullsuite/wheel/新remoteCI，不关闭issue/main/租卡/独占。新0017未跟踪、四文档
  未提交时干净克隆不可用，不能把旧两greenCI说成覆盖本轮改动；本轮0新增GPU-h。
- **遗留与下一项**：目标—最终评分实验尚未执行，其具体protocol/预算/子goal由执行窗口核最新状态自主冻结；
  已有新结果先合法核后复用，尚无时从0/20/80端点无新增更新审计开始。最终科学目标未通过，仍S3，
  2023评分/r与保留授权边界不变；文档交付完成不升级独立确认或goal complete。
