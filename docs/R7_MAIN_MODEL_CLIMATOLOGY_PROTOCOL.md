# 主模型超气候态研究：公平比较与独立确认合同

**状态：合同写定 / 数据实例待预注册（2026-10-04）；研究未执行。**
`scientific_claim: false`。本合同对应 `docs/goals/main-model-climatology-campaign.md` 和决策 0038。
它由主链按用户批准的方案写定，不由 planner、检索摘要或 issue 作者代定阈值。
不是旧 C 协议的修订，也不回改任何历史通过、失败、negative/mixed/unresolved。

## 1. 目标与证据起点

终极目标是沿 UrbanPiDiT 可思考/递归主模型迭代，在公平条件下稳定超过 **train-only climatology**，
并完成真正未见年份/四季确认。它不是承诺全部变量 SOTA，也不是通过关闭 issue 完成研究。

- actual C 的 9 train / 135 eval 已完整执行，三 seed、17 变量、五时效、三区域；package 对 old_ours
  的 primary 六格改善，但过程语义归因 unresolved：`docs/R7_C_ACTUAL_CONFIRMATION.md` §1/§3/§6。
- 旧 C 的 primary 是 t2m 6/12h、对 old_ours 的 package 比较：
  `docs/R7_V2_CONFIRMATION_PREREGISTRATION.md`。本合同是**独立的新目标**，不追认旧 C 超气候态。
- 旧 M2 test 已曝光，旧 8 个 month/hour 桶只是当时 train 的覆盖；历史尺度错误或不兼容分数
  不可复用为新公平基线：`docs/R7_69_BUCKET_EXPANSION.md`、`docs/R7_67_SEALED_TEST_REPORT.md`。

结果分为工程资格、开发期候选支持、独立确认越过、过程机制归因四层。后两层不能由 CPU/CI/关闭代理。

## 2. 科学实例冻结与“不齐不能启动”清单

合同固定下述目标、正/负判定与确认序列规则。每次科学实例另写排他的 `protocol.json`；必须在
训练/评估之前完成并记录 digest，不能拿本合同的待定实例字段直接启动正式运行。

| 字段组 | 运行前必须写出的具体值 |
| --- | --- |
| 身份 | source URL/snapshot/license/hash、data/BUILD_COMPLETE、manifest、训练统计/sidecar、模型版本、代码 SHA/zip/dirty 状态、checkpoint/导入映射 |
| 数据 | 地域/网格/压力层/变量顺序与物理单位、train/val/test 半开时间段、gap/embargo、每 lead 的 init/valid cases、缺测/地下压力面/区域 mask |
| 构建 | preflight 及审阅回执、新 store 路径、单位转换的显式实现/测试、finite 校验、归一化及派生统计来源 |
| 臂与预算 | incumbent/候选/控制的完整开关、Ktrain/Kinfer、精度、种子、更新/样本、选择规则、warm-start、每臂 planned/hard 两秒数、计算定义 |
| 基线 | climatology estimator 和桶/平滑/覆盖配置、仅 train 拟合的身份、persistence 初始化定义、incumbent 训练/选择身份 |
| 评分 | 以下固定 primary/守门，全变量/全 lead、full/interior/edge、空间权重与病例权重、undefined 处理、统计实现版本 |
| 独立确认 | 确认序号 r、年度及四季定义、所有 seed、block 长度/边界、重采样数及统计 RNG、同时区间实现、alpha_r、blind 访问/解封规则 |
| 终态 | attempt 失败/截断/partial/skip 分类、所有产物/指标/成本、排他输出、修复与新协议关系、停止/饱和出口 |

选源/块长/训练量等普通选择由执行者在运行前按依据自主确定，不逐项等用户；未写定时只能准备。
改变实例就新建协议/输出，不运行中改冻结常量。没有完整案例集或合法加载身份时不得启动科学评分。

## 3. train-only 基线与共同信息预算

对每个数据实例重新构建气候态。estimator 至少支持同网格逐位置、按合法 valid-time 日历分组的
train 统计，保留训练覆盖和有效样本数；扩为全年后不能只复用冬季 8 桶或缺桶默认零来让模型获胜。

- 首选复用已有单位已修复的 climatology/skill 路径。若季节平滑、日历分组需要扩展，先在 train/val
  按预声明候选选择，最终实例固定 estimator、平滑及稀疏/空桶处理，不查看 test 再选择更弱基线。
- train 内部做统计交叉检查；缺桶、NaN、有效样本不足明确拒绝或按**先冻结**合法 train-only 估计
  处理，不能读 val/test 补拟合。取得年度训练覆盖后再冻结年度气候态版本。
- 模型和气候态共享 init/valid cases、变量、目标网格、mask、物理单位、空间/病例权重；每个 lead
  的所有配对严格一致。允许不同 lead 合法病例数不同，但逐 lead 指定同 cohort；完整五 lead 报告。
- persistence 来自最后一帧合法历史，不需要训练。incumbent 是 S0 确认的 actual C 结构；数据改变时
  在同数据与合理预算重新训练/选择，无法复用旧值就不填旧值。每候选必须有同数据 incumbent 控制。
- 同图 matched_generic 只钉住等价路径，不当必须击败的竞争架构。类型归因须同信息不同路由控制。
- 参数、输入、更新、样本、FLOPs/算子覆盖、GPU-h分别报；同 updates 不等于同算力。

## 4. primary 与主目标越过门

令 `MSE_{m,s,g,l}` 与 `MSE_{c,g,l}` 为模型 seed s 与气候态在分组 g、时效 l 的共同加权物理 MSE：

`skill_{s,g,l} = 1 - MSE_{m,s,g,l} / MSE_{c,g,l}`。

- **Primary**：t2m，lead `{6,12,24,48,72}` h，region `full`，物理单位 K；不是挑 K 后的 GT oracle。
  最终固定 K 在 val 选定并预注册，推理 K 不能按 test RMSE 选择。
- **点估计**：至少三个预声明 seed；每个 seed 在确认总集、每个预声明确认年份、每个四季组的
  五 lead 均 `skill > 0`。同 seed/同案例逐格配对，均值不能掩盖反号 seed 或某失败季节。
- **同时不确定性**：对 seed 平均的成对误差/skill，以下时间块方法所得全部 primary 确认格的同时
  区间下界均 `>0`。总集、预声明年份与四季组均进同一确认 family，不仅报告 pooled 好结果。
- **关键变量退化守门**：u10/v10/mslp 在同数据 incumbent 比较，五 lead/full、上述各确认组，每 seed
  的相对 MSE 变化均 `<=0`；同时区间上界 `<=0`。默认容忍为 **0.0**，不事后放宽。若守门不成立，
  可报告 t2m 获益但不达到本终极 package 门，继续在独立开发实例排查或优化。
- 分母零、负、非有限或缺病例无法构成 skill；单元标 `undefined` 及原因。primary/守门有 undefined
  就不能宣布通过，不删格、不替换分母 epsilon 获胜。其余变量全部保留 undefined，不均值略去。

“稳定超过”是上述合同门，不意味着所有 17 变量胜出。最终结果必须同时给所有 17 变量 × 五 lead、
full/interior/edge，物理 RMSE、MSE skill、ACC、各病例/UTC段/季节与坏变量。不同单位不直接平均，
正 ACC 不自动等于正 MSE skill；t2m 主目标通过也不改 `scientific_claim:false` 为发布科学认证。

## 5. 时间相关不确定性与重复确认控制

确认不能把像素、重叠窗口、五 lead 或三个 seed 当独立天气样本。

1. 样本单位为 init-time 的**成对病例误差序列**。同一病例的两臂、所有变量/lead与模型 seed 按同一
   时间块一起重采样，保留配对与 lead 相关性；采用季节/年份分层的 moving-block bootstrap。
2. block 长度先用 train/val 的误差相关性或保守物理时窗确定，至少覆盖最长 72h rollout 与历史依赖
   所需的时间跨度；确认实例写成明确秒数和病例数量，不在 test 结果出来后缩短。不得跨 split/缺测
   /年份季节边界拼块。分组内不能形成足量完整块时该确认标 `insufficient-evidence`，不是通过。
3. 统计实现使用同一成对重采样给所有 primary/守门格生成 **simultaneous** 区间（max-statistic
   或预注册等价强控制），必须有解析/合成反证核 family 覆盖、错配病例及相关序列的拒绝。
   具体 bootstrap 数、方法/转换/随机 seed 与不足样本条件在该实例冻结，不作为结果后调参项。
4. 初次确认 r=1 的 family-wise alpha 为 0.025（同时 97.5%）；后续确认
   `alpha_r = 0.05 / (r*(r+1))`，因此总 alpha 支出不超过 0.05，各次至少是同时 95%。
   r 在首次真实 test 评分或任何标签驱动比较时消耗；不能只记录成功确认，partial 读过也占序号。
   metadata-only source/schema/hash 验证不占评分序号，但读取范围须独立记录。
5. 确认账本列全部 r、冻结 hash、候选、真未见窗口、曝光/评分开始时点、alpha、全部结果和成本。
   test 失败或反复探索带来的选择偏差不得通过换一个文件名重置 r。

这是本次前瞻确认纪律，不把旧“三 seed 同号”追认成统计显著性。bootstrap 不能消除有限地域、
时序非平稳或潜在泄漏的偏差；其科学限制必须与区间一起报告。

## 6. train/val/盲测与全年泛化

- 数据 pilot 可以验证真实源/单位/schema/缺测/有限性，正式 split 必须在模型训练前冻结。
  所有 train 统计（normalization、变化量、proxy、climatology）不得消费 val/test 天气标签。
- 源数据发布方可以在隔离构建中按固定 train 统计转换及 finite 检查 test，但不向开发返回 test
  统计、评分、个案/标签分析或配置反馈；读取日志写清 metadata/工程构建/科学评分三种范围。
  下载/存在/校验 source hash 不等于已经科学评分，不能伪造“从未读取任何字节”的宽泛声明。
- 排除任何历史/目标跨 split 的窗口；gap 至少覆盖历史依赖与最长目标 rollout。更长相关隔离由
  train/val 依据前置决定。缺精确时刻明确拒绝，不跨缺测或跨 split 拼接。
- val 用于选结构、K、loss、climatology 配方和 checkpoint。模型/基线/实例/统计法/seed 全锁后，
  test 才评分。最终确认至少覆盖一个完整未见年份和全部四季；区域半球与四季划分先登记，所有
  预注册确认年份都报，不能将只合并好年份替代失败分组。更多独立年份按冻结实例可自主扩。
- 旧 M2 test 已看过，只能标历史开发/已曝光，不能以新 read_count 复封。确认失败后保持其曝光，
  任何后续正式确认要新独立窗口与第 5 节累积控制，禁止反复看同 test 训练到赢。

## 7. 探索、停假设与 incumbent 选择

优先参考新 issue（访问 2026-10-04）：

- [#76](https://github.com/Eswink/UrbanPiDiT_R2/issues/76)：P0 固定 actual C 开关/身份/差距与公平基线。
- [#77](https://github.com/Eswink/UrbanPiDiT_R2/issues/77)：实际激活/有效 PE 频带，单因素低成本试验。
- [#78](https://github.com/Eswink/UrbanPiDiT_R2/issues/78)：解码变化尺度保持原 loss，后轮再变多变量权重；
  [单位评论](https://github.com/Eswink/UrbanPiDiT_R2/issues/78#issuecomment-5982409045)
  指归一化误差应乘 `s_c/d_c`，而非直接除物理 `d_c`。连续 train-only 6h 对，不跨 split/缺测。
- [#79](https://github.com/Eswink/UrbanPiDiT_R2/issues/79)：aux=0 的诊断入前向，A当前/B同信息无类型/C类型
  路由；B→C 才检类型偏置。地下压力面 mask、球面微分/单位与 FP32 敏感差分需要直接测试。

issue 作者 API 为 Eswink；GPT6PRO 来源按用户说明记录，提案不等于事实或冻结判据。旧确认相同图的
Generic 不重复当结构性能竞赛；旧负面假设停下，不能伪装重开同样因素。新增机制必须写明不同点。

每试验在开始前列假设、阳性/负控制、配对选择、新增信息、饱和/停止出口与回 S2/S3 主线动作。
连续无收益转向训练信号/优化/表达/数据制度分析和一手检索，不无限 PE/loss/seed 网格；不把简单
参数化重写冒称新增物理机制。ordinary failure 自主排障、另立新身份，真实失败停止当前 attempt。

## 8. 身份、资源、成本与证据接受

- 旧产物按其归档 code.zip/pins 重放；新代码/输入/统计改变模型身份，显式导入/迁移先核与测试，
  不篡改 digest 或删除 checkpoint 校验。不能合法复用时才预注册有界重训，并写原因。
- R-006/R-028/R-054/R-009及 0030/0038：准备可联网，实验离线；默认共驻，启动/spawn前只读UUID/余量，
  不信号邻居；planned/hard 先冻结，软超继续、硬截断失败全额记账。GPU-h无总授权上限但不能干扰邻居。
- 四成本视图、params、forward/backward计量覆盖、样本/update、训练/评估墙钟与实际推理延迟分别报；
  allocated/reserved 为本进程，不能将邻居负载波动删作坏点；不把 K 次数直接说成加速。
- 结果带 scientific_claim:false、limitations与可复现等级。完整失败/取消/partial/skip不算通过，
  完整 negative/mixed 是有效研究结题但不满足本终极正门；CI不代替真实天气评分。
- 接受链：真实运行和源/代码/协议/产物 digest → 独立只读复核 → result-freeze/E/index/brief →
  工作分支精确SHA CI。最终提请验收时分别写工程、开发、独立越过、机制归因，不自裁goal complete。

## 9. limitations 与当前未做

当前未做 S0 gap 重建、新数据、PE/scale/typed 新实验、新统计实现/反证和未见 test 确认。
本合同和计划不是可运行数据实例，不产生任何获胜数字，也不保证可思考模型一定超越气候态。
严格守门可能长期未过；执行者应自主寻找不同可证伪主线，不弱化合同来“完成”。

本方向仍未授权付费/租卡/独占/main合并/破坏性操作；#76–#79 的关闭不包含在旧六issue授权里。
旧归档证据只读，新数据/参考源码/试验目录排他，磁盘和下载每批预算与 failed-no-fallback 始终保留。
