# v2-remaining-stages-exploration：独立自回归暴露主线与 V2 剩余交付

<!-- round-node: N3 -->

**状态：active（2026-10-03）。** 用户明确选择新路线并下放方向内全部普通执行权；适用0030/0032。
M3规定交付已完成，旧failed、新complete-paused、any-unresolved/advance=false不改；不再补23项。
本目标承接主计划B/N3、C/N4、D/E/N5，不把辅助监督暂停作独立路线总前置，不自判最终goal complete。

## §0 Objective（单段；字符数由 check_goal_brief 实测）

> 在 docs/goals/v2-remaining-stages-exploration.md 按0030/0032接续主模型V2：M3辅助监督已交付且终结假设，原failed/negative/any-unresolved/paused全部保留，不补23val；独立autoregressive-exposure主线完成D1=B/N3，同结构同已知输入无过程语义matched-Generic，核aux_off K4或旧RW-A/K3父model/source/data/sidecar与归档身份，合法复用或显式可审计导入/必要有界重训，真实可微两步L6对L6+0.5L12、同父同seed两臂两预声明seed及可负担equal-compute，t+12精确窗口/时间/K/梯度/poison/resume/finite/FP32/BF16实测；D2遇负面先证据/可反驳假设/一手来源/最小探针/独立预登记对照/登记取舍，不机械加seed算力unroll；D3=C/N4旧Ours/matched-Generic/Process三臂三预声明seed，运行前冻结primary退化容忍/配对单位选择，全17变量五lead的RMSE/climatology MSE skill/ACC、全域边界、同checkpoint K1/2/4非独立训练与四成本视图，adaptive仅新预登记有效前沿门满足才另立协议否则不启动；D4=D/E/N5补#71日历经度batch/padding/K同valid-time与#72现役latent反馈/process_reader非零梯度及反证，逐#70–#75给工程/实验/科学正混负、未做与复现等级，#70无实际改善不DONE-positive，充分负面注明重开条件。每节点conventions/campaign/goal/index/brief/空白、定向反证及全量测试实跑，证据绑定代码/协议/产物digest、精确工作分支SHA主CI必要步骤success后才Closes非force ff推main并匿名核六终态；main分歧不合并强推。实验前独立协议新输出冻结整轮planned/hard，B初轮5400/10800秒，C初轮10800/21600秒，探索另写数值；计时含CPU前置至清理，软超继续记overrun，仅硬截断与真实错误停本attempt并全额保留，新修复用新协议不复活旧失败。无总GPU-h闸门，通知不等批准，独立任务继续。默认GPU共驻每spawn只读UUID/余量，真实数据只读preflight与原身份强核，实验禁网，scientific_claim:false/limitations必写。必要公开官方源码可clone隔离固定URL/日期/commit/license/hash，不import归档、不运行上游脚本；不读封存test或据曝光test选择、不改旧判据/删弱测试、不改用户安全配置绕hook。付费租卡、新多年度天气数据、发布--write、独占、main合并与破坏性仍需具名许可，只停相关动作；无cron/守护/会话后自建续跑。持续实际执行与审计，不以计划CI文档当实验进展；最终分报工程/实验结题/科学增益/关闭、全负面成本overrun与真正未做，不自行宣告SOTA或最终goal完成。

## §1 现状、由来与授权

- 起点：`edcc33539178f95646983eca3084f7c16c241692`，`r7/weather-reasoning`。用户本地
  `.zcode/config.json`差异及两无关未跟踪项保留，不纳入提交、不改配置。
- 开工campaign N2a/paused、0fail/4历史未索引notes；账本显示4.8808 GPU-h，精确累计
  4.880706349145538。旧cap24只是会计基数，没有余额许可门。
- 阶段A证据 `docs/R7_73_VALIDATION_COMPLEMENT.md:94-135`，尺度/三时刻/反证、六训练及30val，
  对future−off为19改善/45恶化/21未决。原attempt仍失败；新补测完整但paused。辅助假设终结
  是用户本次方向决定，非更改旧实验接受结果。0032记录独立路线，0031保持历史有效。
- 原B/C/closeout任务清单通读：`docs/plans/0009-r7-v2-completion-and-closeout.md`，
  `docs/goals/n3-m4-autoregressive-rollout.md`、`n4-m5-confirmation.md`、`v2-issue-closeout.md`。
  它们是旧prepared/预算/父名称时点，不复制RW-B误称、0021问许可、≤30min或余额闸门。
- 父资格待独立实核：M3 aux_off K4；旧process_spacetime_rwa RW-A/K3。归档源身份、checkpoint
  contract与参数映射优先于名称。既有store/source/sidecar只读，不新发布--write。
- 用户明确下放实验设计、架构/方法取舍、编排、训练评估、seed/updates、时长/预算、排障及节点
  推进；保留总体方向与付费/新数据/发布/独占/main合并/破坏性权限。预算增加留痕通知不等批准。

## §2 交付物清单与 prompt-to-artifact 对表

| 交付 | 实际要求 | 证据形态 / 落点 |
| --- | --- | --- |
| D0 | 起点、指定文档通读、0032、主计划/round-node/next-action同步 | git输出、目标/ADR索引、campaign/goal结果 |
| D1a | 同结构、同输入、无过程语义matched-Generic，复用已有组件 | 代码与参数/结构/信息白名单反证；model-digest-change |
| D1b | 父实际K/模型/source/data/sidecar/zip资格，显式导入若必要 | read-only identity报告、旧接受/新映射/数值等价、新checkpoint新身份 |
| D1c | 真可微L6+0.5L12；第二history为预测，未来truth只监督；两个时间轴 | 精确t+12/缺测split拒绝、时间/K、梯度、poison、resume/finite/精度测试 |
| D1d | 两臂两seed同父额外更新，长时效全表及不等算力/equal-compute | 独立B协议、六臂次或已声明缺项、全部指标/案例/四成本、失败与预算回执 |
| D2 | 有必要才做独立机制探索；不是无限延长旧假设 | 证据/假设/一手原文/hash/最小探针/独立protocol/正混负取舍与回B/C动作 |
| D3a | 三臂三seed公平确认，预先冻结primary/容忍/配对/单位/选择 | 独立C协议及digest、9训练与全部固定K评估、全域/边界指标 |
| D3b | 17变量×6/12/24/48/72、RMSE/skill/ACC、K1/2/4与坏案例 | CSV/逐案例、计数/单位拒绝反证；K1明确非独立训练 |
| D3c | params、forward/backward、训练/评估/真实latency、allocated/reserved | 四成本视图，进程独立峰值、计数覆盖局限、实耗GPU-h/whole/overrun |
| D3d | adaptive前瞻门禁与独立决定 | 新前沿固定gate/证据，满足才新协议；不满足“不启动”合法 |
| D4a | #71年末/闰年/UTC/东西经/batch多日期/odd padding/内部K同时间 | 直接断言与故障注入反证、暴露缺陷记录/活跃修复复验 |
| D4b | #72现役反馈到latent/process_reader非零梯度 | 真实现役路径测试与阻断路径反证，不用默认off开关替代 |
| D4c | 六issue逐项工程/实验/科学verdict、未做、复现等级及#70重开条件 | 新证据页、comments、queue最终快照、E/index/canonical brief |
| D4d | 精确关闭SHA九主步success，再合法ff/main与六终态 | commit/CI run/jobs URL日期响应digest、git祖先/推送输出、匿名issue终态 |
| D5 | 每节点完整验证与最后逐要求审计 | 全量/定向pass与skip、conventions/campaign/goal/index/brief/空白日志、独立复核 |

## §3 判据与证据来源

- 设计单一权威：`docs/R7_MAIN_MODEL_V2_DESIGN.md` §4/5/8（两轴、未来白名单、身份与共享step），
  `docs/decisions/0023-main-model-first-baseline-freeze.md`（matched-Generic与基线冻结）。
- B清单与λ0.5：`docs/goals/n3-m4-autoregressive-rollout.md` §3、计划0009 B；继承真实可微、精确
  t+12/第二步预测、同父两seed、全部变量长lead、不等算力与可负担equal-compute，不改旧判据。
- C范围/科学层级：`docs/plans/0004-r7-main-model-v2.md` Evaluation protocol/Scientific gates，
  `docs/goals/n4-m5-confirmation.md` §3与issue#75一手正文。primary和退化容忍是新实验运行前
  冻结字段，不拿旧结果事后选。三seed只描述一致性，代表性/独立test缺失不科学支持。
- 日历/现役梯度与逐issue验收：`docs/goals/v2-issue-closeout.md` §2/3、issue#70–#75正文。
  #70实际改进不成立时只能充分负面/混合结题，列重开条件，不记DONE-positive。
- 暂停旧出口的适用范围：0032用户前瞻路线；M3页/0031不改。不因某cell未决停止全部独立任务。
- 机械与运行：`docs/rules/testing.md`、`docs/rules/ci-and-verification.md`、
  `docs/rules/gpu-resources.md`、0030。skip/failed/cancelled/queued/partial不接受为通过。

## §4 实施顺序与依赖

1. 只读起点/父身份与来源；0032/本目标及主计划同步，six recheck。
2. 并行工程：matched-Generic/显式导入，可微训练数据/loss/runner，D4 CPU日历/现役梯度；
   代码探查与一手文献独立委派，摘要不作证据。各文件所有权不重叠。
3. 定向反证、全量CPU及真实FP32/BF16/resume小探针；源码/协议/数据强核、工作分支精确工程CI。
4. B预声明seed41/42，同父K4优先；200额外L6更新对200两步更新，equal-compute采用400 L6
   更新作为约两倍优化计算的前瞻控制（以实测FLOPs/时长报告偏差，不宣称严格墙钟相等）。
   每个新微调优化器/学习率端点在协议显式冻结；不得把新训练解释成旧schedule的resume。
5. B全量结果冻结/index/账本/独立核验，按结果结束对应假设；无收益不扩unroll，独立C继续。
6. C预声明seed41/42/43与三臂，合法可比父/初始化及训练配方先冻结；候选获支持才用它，否则
   明示负向固定结构控制。全K/边界/四成本后判adaptive，不能强开。新机制探索仅有实质差异才立。
7. 每节点登记与对表后自主推进N4/N5。D4/逐issue实证足够后Closes，精确CI绿，核main祖先，
   非force ff及匿名核六issue。真正保留权限只停相关动作，独立任务继续。
8. 最终逐条D0–D5/本次用户要求实证审计，分开工程完成、实验结题、科学增益与GitHub关闭。

## §5 预算与停止条件

| 新独立轮 | planned_seconds（软） | hard_cap_seconds（硬） | 范围 |
| --- | ---: | ---: | --- |
| B初轮 | 5400 | 10800 | 父/数据CPU前置、6训练臂次、完整val、计量、聚合/清理 |
| C初轮 | 10800 | 21600 | 三臂三seed完整确认、固定K/全域边界/成本、聚合/清理 |
| CPU/精度工程探针 | 900 | 1800 | 独立protocol、小模型反证/FP32/BF16/resume，不冒充预报实验 |

每轮从最早prepare入口冻结同boot单调anchor，含CPU准备、归档、启动/间隔、训练/评估、聚合和
owned清理；超软预算继续，`soft_overrun_seconds=max(0,whole-planned)`。不得运行中改硬上限；
下一独立修复按实测自定新预算/新输出。连续GPU区间全额记账，失败不能仅计成功worker小计。
账本仅会计，无总GPU-h上限；本地运行不新增GPU CI，若以后承载CI必须timeout大于硬上限并留收尾。

真实错误/身份拒绝/资源余量不足/不完整集合停止本attempt保留失败，不原地复活。负面终结相关假设，
不无限重训、增seed或扩4/8/12训练unroll；D2仅实质不同机制/对照。付费/租卡、新多年度天气数据、
发布--write、独占、main合并/破坏性操作停止相关动作请求具名权限，不妨碍其它独立任务。main
分歧停main动作、不force/merge；安全hook拒绝不绕、不改用户设置。最终goal状态不自行complete。

## §6 与planner草稿的差异

只读planner返回围栏JSON；主链修正版 `/tmp/r7_v2_remaining_plan_20261003.json` 实跑
check_planner_plan verified=true/fence_stripped=false/0失败。抽查process_step:95与Generic构造、
r7_store:16–48及scheduled transfer后，纠正草稿把solver/reader误当可省诊断部件、父误称RW-B、
不存在父路径、protocol需stage入git、余额闸门、负面停止全部任务、重复0032关闭ADR和自行complete。
实际matched-Generic保留完整reader/solver，只省诊断readout；aux_off RW-A/K4优先、full新子实验
身份明确，预算由主链5400/10800与10800/21600写定。规划不是证据，科学判据不外委。

## §7 明确不做

- 不补M3、回改旧failed/协议/暂停、清理旧产物、伪造digest或绕旧加载比较。
- 不写data/raw/interim/processed或legacy，不import归档；不下载天气数据/发布--write，不读封存test。
- 不据已曝光test选配置，不改变旧端点/阈值、不删弱测试，不用no_grad评估rollout训练。
- 不新增solver家族/外部baseline竞赛，不机械加seed/更长unroll，不称三seed显著或SOTA。
- 不signal邻居、独占/付费，不force/mirror/main合并、不改用户配置，不cron/后台会话后续跑。
- 不把更多文档/CI绿/关闭六issue当模型增益或最终goal完成。

## §8 进度块

- **状态**：active；起点已核，指定材料通读，父身份与代码探查/规划/一手参考、D4工程并行进行。
- **六项recheck**：①新路线N3由用户0032选择，主计划与上一轮next-action前瞻同步；②旧账本
  4.8808/19.1192不改、四历史无索引notes保留；③N2a证据index已登记可达；④本objective派生
  B/C/D/E清单并引用冻结来源；⑤开工rules/science无执行者未记录diff；⑥实验尚未启动，先过同步对表。
- **已实跑**：campaign新N3为0fail/4notes，目标结构0失败/0建议；planner修正版verifiedtrue。
  只读四父核actual K/131 tensors/signature与HEAD model map，M3 codezip80/source/sidecar实际hash齐；
  最新并行工作树改变digest，必须显式导入新子实验，不直接旧load。官方GraphCast clone固定原版
  commit/Apache2许可/五文件hash、原文已核，无上游安装/训练。D4修前年界探针实际2fail/15pass，
  optional init_calendar_year前瞻修复中；这些是工程/机制准备，不是预报结果。无新增GPU/新CI。
- **工程实际进度（2026-10-03）**：matched Generic完整577参数映射、显式导入、exact185窗口、真实grad-enabled两步与原initial/allK深监督、calendar修正、单lead/K评估和薄驱动初版已实测；主链首次全量2614pass/10fail/9skip，不能当通过。输入/Generic/calendar复跑169pass。独立归档父四合成case final/all5drafts严格torch.equal，109产物hash主链再核无差，回执SHA56acdc7e…c5a794；这不是天气结果。
- **独立核验缺口已修复**：actual嵌套heads/dropout合约、roles/noFeedback、direct draft query、累计K列表、零训练baseline、baseline两kind、worker主动硬截止归因均有实际反证。0034默认关query234定向passed，四default case前后output/grad/weights/RNG相同；独立小探针也确认fixedP/C/pos局部影响与branch-cut失败、L12全部reader梯度、roles/语义拒绝。最终model digest0cc9c16e…a223e的旧父四合成case再次严格逐位相同，109pins核齐。driver77、results81、主链runner/driver160实际passed各自保留；不是预报结果。C前瞻primary/0退化与adaptive完整门已写 `docs/R7_V2_CONFIRMATION_PREREGISTRATION.md`，仍待独立protocol冻结。
- **最终完整CPU**：3107passed/9skipped/2warnings，767.02s，0fail；六CUDA与三未跟踪optional真实fixture跳过不算通过，两Lightning警告保留。full期间仅去driver测试末空行，bytes proof及最终160项复跑接受；380源快照/完整stdout/receipt已留存，37阻断/两goal/campaign/index18/compile/空白pass。不是预报结果或CI。
- **精确CI/真实probe**：工程616b029dce569b92bd08295737981512180a1ad3、CI37133487340九principal+三post全部success。独立precisionattempt01 protocol0829da02…5f4ba实际FP32/BF16各4updates、weights/optimizer/RNG/losses resume严格相等，L12-only第一步与reader梯度finite>0、poison/calendar通过；allocator0/0、连续0.04711594580465721GPU-h/whole242.4378195s/0overrun。新证据页R7_V2_PRECISION_ACCEPTANCE保持工程边界，独立产物复核在收尾，不是预报收益。
- **B实际执行（2026-10-03）**：协议9f76e6e3…5e62b02、616b029/source5de6b16d/codezipd3d10a74，
  六训练1600updates与三十val全部success，但聚合training objective校验失败，attempt failed/finalizedfalse；
  不恢复原attempt，不重复GPU。连续3069.380453112535s=0.8526056814201487GPU-h，全轮3200.912431293167s，
  overrun0/非budget-limited，全成本保留。361原文件pins已封印；400两步loss按FP32逐运算重构严格一致，
  原binary64校验拒绝99/88条，0036前瞻修正与独立CPU统计补全600soft/1200hard，不放宽物理指标容差。
- **原验收补齐工作态**：0035/0015的known-context/source-position仅隔离活跃复制树实施；原B期间
  runtime与HEAD未变。known owner工程attempt01实现耗时超1800硬上限，pytest尚未spawn，明确budget-limited
  非通过；新验证协议独立，不回改旧预算。原#72参照确为K3，不冒称M3 K4；28model成员一致的M3归档
  合法strict-loader源码桥已只读核，独立0train10eval专用工程准备中，旧calendar差异必须披露。
- **独立机械闸门实际失败**：统计补全38项合成测试通过，但独立反证确认提前validation失败未封印、原attempt/execution未直接绑定原protocol/stage、最后hash跨硬限仍finalized三缺陷；旧审阅门FAIL，未消费真实B统计，修复另冻新审阅。known/source定向220通过之后，独立反证另发现活跃batch未切片anchor及合法20分钟offset舍入误拒，真实修复中。旧与新RW-B的BF16 fixed/streamed提案梯度差异保留为兼容性限制，不放宽断言，不冒称新包全部通过。
- **原失败/精度证据正式本地追加**：A=42ecb890ce8f8f1d2925f8841a74859b2decd8f7两冻结页Git blob实际强hash；index18→20单增precision engineering-positive/needs-review与B failed audit/blocked两record，旧18条字节不变。canonical brief/账本record refs齐，14索引定向0.08s通过；工程CI只绑定616b/37133487340，登记提交/其精确CI仍待。此登记不接受尚未重聚合的天气比较。
- **B实际零GPU补全（2026-10-03）**：新protocol `ecd8ecdc33151537cd11c50c08036d19dfbaeec712742cf09732ce696fd721af`、独立排他 `outputs/r7_74_stats_complement_20261003_attempt01` 完成，whole51.719秒/软超0/新增GPU0；原B failed不改。运行前58named+28独立反证接受三项机械修复及旧sealed source写前保护，不作实际结果验收。实际冻结primary两lead/两seed对200L6均改善，但对400L6均恶化，规则 `negative_or_mixed / selected_mode=l6`，只终结rollout假设、独立C继续；完整产物独立复核已启动。
- **两模型修复复验**：独立46passed仅接受合法cadence/selected-anchor及有限FP32baseline，原FAIL/BF16gap保留；主链完整定向249passed/51.03秒、whole52.933秒，源码无漂移、含两checkpoint序列化case。不是fullsuite或新包GPU接受。
- **登记精确CI尾核**：`adb28667759e8d1aabd3bc0bf7bc07f7b2bbd591` / CI37144555634 / job111265708281 completed/success，九principal和三post全部success。匿名jobs API访问2026-10-03，响应SHA `9b5fc6c68c0f24bd9964475b747fd578bce179d3b9ca7cfde6036b1fc90e7026`；仅登记原失败/旧精度，不覆盖未提交新源码。
- **B补全正式受限接受**：统计attempt01因113文件/110 pins遗漏worker.log被独立拒绝，数值一致不升级整包；旧审计whole1396.540345秒/soft-overrun496.540345、未跑qualification.py保留。日志pin及reap后hash修复有59named+3独立反证；新attempt02 canonical `076f28dfe88cd6e600d0ed37089f40b8e0562008e5d254facb99f93bb9a449e9` 实际52.027686852秒/软超0/新增GPU0。独立完整361旧+113新 before/after、111 pins+2 exclusions、93/94/5/28归档成员、6train30eval、1530/765/3×255配对与全部11 CSV row hashes通过；audit SHA `0f965f2e7fd8dff509637835866737f3961003eb56b434704abf515676bf4e8b`，final closure442.525976秒/900软1800硬/overrun0，仅 ACCEPTED-METADATA-COMPLEMENT。原B failed/finalizedfalse不改，negative_or_mixed/l6，独立C继续；新定稿页 `docs/R7_74_STATISTICS_COMPLEMENT.md` 不替代尚未正式索引登记。
- **稳定工程收尾**：K3四模块103passed/0skip、active项目命名空间保留；治理实测1550 test函数/4272 assert，166passed/0skip，checker1929→1927仅baseline常量/原注释变化，37阻断/两goal/campaign/index20/brief通过；治理最终whole748.818秒/软超148.818/硬1200未截断。新stable完整CPU/wheel1800软3600硬已启动，终态未取得，不能称通过。
- **实际UTC首轮失败**：B-only官方CPU统计protocol `f367d7cadd56a7f9e0c0924e79a8bf0eae78f6650b1d7ed818ddca0e51443867` 已执行并failed/finalizedfalse/0行0组，whole374.24秒/软超0/新增GPU0。首个climatology CSV保存skill0及undefined_ACC，重算binary64却为−2.22e−16，数值在原核验容差内但重造negative标签误拒；原sealed失败不复活，活跃未来统计修正须直接反证及新协议。
- **最终稳定工程轮**：源码无漂移，完整CPU/wheel实际3467passed/9skipped/3warnings，903.06秒，整轮905.674715秒/1800软3600硬/overrun0。六CUDA与三optional真实fixture跳过不算通过；zip重复成员反证及两Lightning警告保留。protocol `3703be5ccf2eaf7788742f7e0c108117793060b1510194301cca3be1fd31388b`、source tree `5b6699bb78249d6251770c5d5b78e582b9ba341b55183bf001700ff11b475956`、stdout `f0d2e99bdfedfb9ebcea6dbe48fa2988a3ede837fa19ebf2514d3055b8e0fc86`；原5fail/3412pass失败保留。261活跃文件独立compile成功，37阻断、campaign、两goal与两侧空白通过；结构wrapper误用index CLI参数退出2保留，独立纠正CLI后21record canonical通过，不改checker。最终基线1574函数/4299assert，166治理测试通过，整轮288.226秒/软超108.226/硬360未截断。
- **独立窄修复封印**：UTC保存metric标签域修复26named及24 scalar调用接受，receipt `fd2f8a663b58ffe8335c96bae4825366ca5838e19a3e95f5766c5b9173552358`，whole535.720716秒/软超235.720716/硬600未截断。C全库存owner118pass后独立24+28反证通过，原review因index-only暂存变化not-accepted不改；新固定归档byte补充验收receipt `1319e732ae852fc9e8ddc6404a4ceb1f79a203b5ad6160b688f7c4f0fa6c6969`，fresh24pass/whole401.507秒/软超101.507，只接受前瞻工程不接受实际C。
- **新实际运行**：genuine原#72 K3 protocol `8c42e5031e88c04f4c9514236d31bb4bc89da5b1fb96aa86bb9700ab63d8bdbc` 已10/10评估/0训练、176案例、1530区域行/170全域model cells，whole176.678382秒/continuous GPU173.477160秒=0.0481881000453GPU-h/overrun0；candidate complete-not-accepted，独立全库存/数值复核尚进行。旧365.25 init clock及累计lead保留，不是C Gregorian同因素参照。CorrectedB-only UTC新protocol `9a82c1357750b41b9a930879543757dfe6ca537228ea862b2f5ba5d6bc57c5e8` 官方实际20400行/400组、empty0、CLI147.193569秒，初次公式自核封印snapshot712.884945秒，含后续历史log-prefix核证与最终publication closure整轮949.854352秒/900软1800硬/软超49.854352；日志snapshot与最终full-hash分别声明，不回改旧receipt。closure `68e06fa10d403e2014fc45624ab831f2f1b3eedf69eabad440b11596c4392c46`；owner receipt `f1c301ac604d3d69530fb9734f1d56126c6881b93c3063f2da2d0c9169435383`。独立不同审阅仍待，不改原UTC失败或B失败/negative_or_mixed/l6，不据tiny climatology负号判科学退化。
- **未做**：新工程精确CI、K3/UTC不同审阅终态与正式登记、新包GPU精度、C三臂训练评估/UTC、adaptive裁定、六issue结题/关闭。#71单输入因素历史三臂完整资格还在核；C整包不自动代替M1归因。不将并行委派或更多文档当完成。
- **当前具体动作**：提交已实际通过完整CPU的工程修复并核精确CI，完成K3/UTC受限独立接受及登记，新包precision后绑定实际前置、冻结L6 C并实跑9train/135eval，按原条款补齐必要M1单因素/UTC，再N5实证结题和有条件关闭。不等普通许可，不回写N2a科学出口。

### 实际 N3 出口与 N4 接续（2026-10-03；不改上述历史事实）

- 本页仍 `<!-- round-node: N3 -->`。实际N3/B02完整统计独立接受只为metadata complement，`negative_or_mixed / selected_mode:l6`；原record:n3-b-statistics-complement-negative/evidence1615795/页SHA7ea1保留，B原failed/finalizedfalse与B01库存FAIL不改。rollout假设按冻结规则终结，独立C继续，不要求M3 any-unresolved/paused/advancefalse改成通过。
- K3与UTC新独立final接受已登记：真正K30train10eval/176case/1530行/170full/136库存/24inputs，reference metadata-only、原complete-not-accepted不改、旧365.25/accumulated/interior_2不作Gregorian/architecture-matched paired C；原GPU0.04818810004533993h只记一次。UTC20400/400/empty0、248files/80784源行/全公式，仅official metadata-only；owner旧whole-package NOT ACCEPTED/full-log-prefix mismatch、negative/undefined及旧失败保留。K3audit719.8785105217248无软超、UTCowner949.854352/over49.854352、independent1340.701317/over440.701317，各900软1800硬、0GPU。
- 两final冻结页actual61db2725f6fda6abf986d94b7dbd0c6dd0bf12a4/final Git bytes已append新audit/not-candidate记录/E-243/E-244，index23canonical/旧21原字节保留。主计划现在current=N4/previous=N3、current_round_goal为新distinct `docs/goals/v2-l6-confirmation-current.md`，previous_round_goal仍本页/previous_round_evidence=B02，不重用旧prepared N4为actual。显示账本5.8287/18.1713与精确总5.828616076415684/18.171383923584315分开。
- 精确523c819/CI37155293949实际FAILED step9/post17 skipped，全部12必要步骤未通过，不能以旧工程CI/本地全量绿追认。主链纠正clean523/Py3.12独立全量3462passed/14skipped/6warnings，whole926.8922247411683/1800软3600硬/overrun0，sourcechanged[]仅本地工程；初nested-basetemp17failed是诊断harness缺陷，不归远端。新tempPy3.11 full进行中，未accepted；新包GPUprecision未run、C未prepare/未protocol/全9train135eval未run、adaptive unevaluated非failed。新C严格package/CI/precision前置不因路由变化豁免。
- M1#71历史allProcess不能替当前完整单输入因素。实际C独立接受先行，随后old_ours41/42的10K4score资格复用与4fresh scratch M1-only400L6train/20K4eval、185窗/fullcohorts、5400软10800硬另冻；不混RW-A/query/source-position/roles，不阻独立C。临时M1 owner1816.622756秒硬超NOT accepted、独立审阅进行中。此登记无新GPU/source/tests/数据/配置/commit/main/issue动作，无科学或最终goal完成。
- 下一动作：N4按 `docs/goals/v2-l6-confirmation-current.md` 先取得新工程包精确CI全部12必要步骤success及独立新包precision，再冻结同anchor scratch/selected400的L6三臂三seed C协议，实际9train/135eval后独立整包核验；未齐不得先启动GPU，不逐次问普通许可。后续必要M1-only仅C独立接受后执行，不复活B/M3或回写本页N3 marker。

- 最后登记工程实测：37阻断0失败；campaign N4/0失败/4历史notes；current/remaining/master三个goal0失败0建议；index23/canonical brief与git diff --check均exit0；五相关治理CPU模块 **213 passed/33.15秒**，socket先禁网、CUDA hidden、无skip，actual AST仍1574测试函数/4299assert。回执 `/tmp/r7_governance_registration_audit_20261003_NRQYhoBF/verification_receipt.json` 记录whole软900/硬1800、实际软超，不把工程PASS当remoteCI、科学或最终goal通过。
- 主链独立Py3.11.15/torch2.14.1+cpu的精确523 full已实跑 **3462passed/14skipped/0fail/0error、pytest906.85秒/child+reap909.57秒/attempt whole909.92秒、exit0、425sourcepins前后不变**，tree `cb81953cebc6bc032ac619cfa144c0da28101656`；六warnings经主链实际读取为三record_property+duplicatezip+两Lightning；attempt whole采用909.92秒权威值，不拿pytest或child时间代替。主链stdlib JUnit对比两环境skip identities精确同14，14skip为6CUDA+缺M2/D1及optional真实fixtures，与独立clean3.12同缺项，不能算passed/依赖错误或合成回退。此本地绿仍不能归因或追认实际remote523失败，新包precision/C仍未run。
- 主链另改 `.github/workflows/ci.yml` 仅step9失败诊断（保持原pytest全量选择/原exitstatus，RUNNER_TEMP junit及有限escaped annotations）；这是外部Main非owned变动，非本登记agent改source/model/tests，不能称remote bug已修。最晚通报workflow SHA `aacaca7ababd01e071a48fd70a7ae04151c0a52951bad8382707eba43bbf7e5a`、独立review尚待；初PATH python缺失counterproof失败127保留，随后诊断反证独立通过是工程范围，不是新CI。当前remote cause未复现/未确认。

- 下一动作：N4按新current目标完成精确新包CI12必要步骤及独立newprecision后，才prepare并实际执行独立C；本页N3 marker和原事实不改。
