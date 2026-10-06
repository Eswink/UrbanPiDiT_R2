# S3：同 case train/val 长 lead 差距诊断

<!-- round-node: S3 -->

状态：active / implementation，2026-10-06。唯一主计划
`docs/goals/main-model-climatology-campaign.md`，当前节点仍S3，不是S4或新方向。

## §0 Objective（单段，实测1090字符）

> 沿 docs/goals/main-model-climatology-campaign.md 的S3接续已登记12步长监督screen，执行 docs/goals/s3-same-case-gap-diagnostic.md 的小型同case误差诊断，不把控制gate清门当超气候态。D1只读metadata为2017–2021 train每年月份1/4/7/10和2022val四块各选完整t−6,t,t+6..72h窗口的chronological lower median，共20train+4val，全部索引/时间/digest在天气字段读取前冻结；D2严格seed41父BD1600与长监督200endpoint模型身份复载，FP32/K4无optimizer，用原train-only month/hour climatology、相同case/五lead6/12/24/48/72h/全17变量，报告纬度面积加权物理MSE/RMSE/skill、逐case和year/month/split组及误差增长，不平均不同单位；D3CPU阳性/断言反证及完整回归、独立prelaunch审阅、精确commit/archive后，在新outputs冻结soft1800/hard3600秒有界诊断，所有准备/归档/评分/判读/清理全额计费，soft超继续记账、hard或真错误停该attempt不resume；D4独立身份/数值/成本复核与兼容代码临时再现，证据/index/brief/主账本和精确SHA CI登记。0030/0038方向内自主推进无总GPUh上限，GPU1共驻每spawn free≥max2048MiB估计/known2434793472bytes+2048MiBmargin，仅直接拥有Popen可截止清理，禁止邻居信号/cron/付费/独占/main/安全配置变更。train为in-sample，val为开发且24case稀疏；误差符号不证明优化欠拟合、泛化因果或收敛，不换弱气候态、不设百分比阈值、不做梯度/训练剂量首阶段、不读2023test、不启动S4，不改旧model/heldout evaluator/recipe/data/failed或科学门。既有科学判据只来自docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md与docs/goals/s3-confirmation-baselines-and-candidate.md，所有未做/负面/预算例外和限制保留，不自行宣布goal complete。

## §1 承接与事实

- 起点 `85ae0ea827614749fe4712339470626cfeb85023`，工作分支r7/weather-reasoning；用户config/.zcodeignore不动。
- 前轮 `docs/R7_S3_LONG_ROLLOUT_SCREEN.md`，证据d1dcd46/登记85ae0ea：三seed开发主格supported，
  控制gate0/45，全255RMSE低于D3400；但t2m48/72hskill三seed全负，parent相对gate16/45。
  screen+replay3.0362GPU-h全额，方向累计17.8324；模型源码/数据不变，2023未评分。
- 保存training600条preclipnorm全部>1，但各块样本不同，后期下降/反弹混合，不判收敛或clipping因果。
- 冻结候选路径seed41 `outputs/r7_s3_long_rollout_20261006_attempt01/seed41/training/candidate/update_0000200.pt`，
  SHA449bc4ce6d9c15178fecb75a954a265ffb43f7ea8cea92575d95387047ee066b；parent从原pinned_parent(41)严格引用。
- 现有evaluate_local/ZarrRolloutDataset保持heldout-only；独立diagnostic允许train仅明确in-sample，不弱化正式接口。

## §2 交付物

| # | 内容 | 证据形态 |
| --- | --- | --- |
| D1 | 24case精确metadata/选择digest、泄漏与边界CPU反证 | 冻结plan和tmp_path测试；metadata不读state |
| D2 | strict双endpoint/原climatology/target-free rollout/physical per-variable指标 | 契约/源码/单位与analytic tiny CPU positive/counterproof |
| D3 | 独立审阅后fresh有界诊断 | code.zip/protocol/四phaseworker/measurements/费用与failure retained |
| D4 | 独立复核/再现/登记和接续 | artifact hash、E/index/brief/master/cost/精确CI，不给科学PASS |

## §3 判据与证据来源

- 不新增科学阈值；`docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`为终极合同，S3旧screen判读来自
  `docs/goals/s3-confirmation-baselines-and-candidate.md`。本diagnostic不产生candidate通过判定。
- 模型与气候态同case/lead/grid/纬度权重/物理单位，全部17变量单列；climate只fittrain原2400/16×150，不重拟弱baseline。
- metadata选lower median不由label/metric反馈；train自参照乐观、val开发且24case稀疏，必须报告而不因果外推。

## §4 顺序与依赖

1. 完整登记前轮并核精确CI；审读实际evaluation/rollout/store代码，独立case/preflight/core与CPU反证。
2. 新父worker driver所有spawn统一headroom、原owned bounded_process；协议scope/输出/失败及soft/hard先写定。
3. 独立prelaunch、全量tests、commit+exactarchive；boundedprepare核source/norm/build/plan/端点与库存。
4. 发布protocol后CPUarchive再GPUmeasurement，只推理两模型同case与原climate；boundedreading核全case/指标。
5. 独立hash/metric/cost审阅及代码临时再现、E/index/brief/账本/CI，才选择后续实质不同假说。

## §5 预算与停止

| 项 | 数值/边界 |
| --- | --- |
| 整轮软/硬 | planned1800 / hard3600 s，包括CPU准备/归档/测量/判读/清理 |
| GPU | GPU1 UUID GPU-9d1624af-9d77-aa7c-0620-b6cb778f4ced，共驻非独占 |
| 每spawnfree | max估计2GiB/known2434793472bytes+2GiBmargin；至少4370MiB |
| scope | seed41parent/candidate，20train+4val，12步推理，五lead全17变量；无optimizer/test |
| 停止 | 真identity/finite/executionerror或hard截止停此attempt、保留全额，不原地retry/resume |

软超继续记录，没有总GPUh授权上限；当前cap20/used17.8324/remaining2.1676只是会计。
没有证据时不自动扩数据/剂量，不为了获胜放宽scope/冻结门。

## §6 planner草稿与降级

planner两次返回无实际toolreads的错误草稿：不存在climatology/evaluation路径、220排除当val数量、
source/window/codecommit混当模型/数据/checkpoint身份，lead列表变更和跨17变量物理单位平均、
百分比“判定”及虚构截止。全部拒绝，不采用这些阈值/事实。
执行者已根据实际 `training/r7_evaluate.py:198-221`、`model/r7_rollout.py:53-75`、
`data/r7_evaluation.py:90-142`自规划；工作JSON
`/tmp/r7_same_case_gap_reviewed_plan_20261006.json` verified:true/fence_stripped:false，两个引用实读。
计划不是科学证据，不外委科学判据；实现代理只编辑独立core/test两文件，主链driver/test/goal不冲突。

## §7 明确不做

不读2023test/启动S4/optimizer更新/新剂量/gradientfirststage；不改oldmodel/recipe/evaluator/store/archive/failed；
不下载/付费/独占/租卡/main/issue关闭/force/credential/config/cron/守护或邻居信号；不弱化测试/身份/科学门。

## §8 进度与交接

- 状态active，implementation，S3不变。
- 前轮所有screen/replay费用与52index已登记，85ae0ea精确CI37535580456已completed/success，全部必要steps success；非科学接受。
- 父driver已补齐固定train/val身份、ordered24case绑定、原climatology与逐case/汇总/gap算术重算，
  全源scope/path/bytes及每endpoint modelcode一致性补齐；稳定core+driver完整122项CPU实跑通过（63.50s），
  core单套60passed（63.49s）。installed-wheel新增模块成员/clean-directory来源核验1passed（24.75s）。
  原scope-refreeze四缺口独立target17passed/20deselected（1.61s）已关闭。
- Core首轮CPU测试因误用Process的`forecaster`属性失败，未完成suite保留为未通过；实际模块属性是`backbone`，
  仅修constructor接口、严格checkpoint/模型/单位门未放宽。修后真实tiny24case CPU阳性1passed（50.22s），
  独立core→reading同fixture接续亦通过（39.116s）；这些均非天气或科学证据。完整回归仍待实跑终态。
- 独立prelaunch审阅无剩余essential blocker；最终新header反证6passed/56deselected（5.04s）。
  审阅983s，soft600超383、hard1200未超；中途停止suite及并行旧fixture13failed/20passed保持非PASS。
  临时test-only metadata快照不改production读法，独立阳性使用真实本地tinyZarr而非快照缓存。
- 尚无真实metadata24case冻结/天气字段读取/新GPU诊断/结果/费用，不预写全量工程资格或科学通过。
- 下一动作：完成核心CPU反证与独立prelaunch终态、全量回归，commit/archive再冻结fresh有界diagnostic。
