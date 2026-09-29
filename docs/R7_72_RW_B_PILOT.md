# #72 M2-B 第四轮：RW-B 局部门控求解状态的一轮有界真实对照（E1 / D5）

**状态：一轮有界实验已执行并已判决；本文件不声称任何预报收益，判决是负向的。**
本文件**不覆盖**前三轮：`docs/R7_71_72_M1_AND_RWA.md`、`docs/R7_71_72_ROUND_TWO_ATTRIBUTION.md`、
`docs/R7_71_72_ROUND_THREE.md` 仍是各自唯一的证据；四份文档的 `protocol_sha256`、
`model_code_sha256` 都不同，**数字不可相加、不可并列**。实现与测试的正面结论在
`docs/R7_72_RW_B.md`，本文件只记这一轮实验做了什么、读到什么。

| 项 | 值 |
| --- | --- |
| 起点 SHA | `21387a025941c843967332479160c8b09b407090`（开工 `git rev-parse HEAD`） |
| 授权 | 决策 0021 的会话内通道；用户于本轮执行前明确授权「按上述范围跑」（2 seed × 4 臂 × 400 updates，≤1.0 GPU-h） |
| 数据 | M2 双月段 `outputs/r7_m2_segment/store/manifests`：train 186 / val 22 / test 26（**test 全程未读**） |
| 协议 digest | `6f48874296352d660c2a5d07c3de7c3bceac16619e54adec4a505ce8ff5d9ed9`（8 个 run 全同，逐 run 核对） |
| `model_code_sha256` | `9ddd2660820dc95e89ca19d78b36765d22805c6462c30e23b58ae62aa1f5a21e` |
| 实测 GPU | **0.8356 GPU-h**（两个完整尝试之和，见 §6；单次成功尝试 0.4154） |
| 本轮自设上限 | ≤1.0 GPU-h（实测 0.8356，**未超**）；第二批授权余 21.537 → **20.701**（重算见 §6） |
| 墙钟 | seed41 14:33:40Z→（cuda:0）/ seed42（cuda:1）并行；两卡各一个 seed 进程 |
| 实验产物 | `outputs/r7_72_rw_b_pilot/`（protocol、逐 seed、合并、配对、五张表） |
| 失败尝试产物 | `outputs/r7_72_rw_b_pilot_attempt1_probe_defect/`（**保留未删**，见 §6） |
| 判决 | primary **未获支持**：5 个时效中 3 个 sign-consistent 恶化（6/48/72h），2 个改善（12/24h，幅度小一个数量级） |

---

## 1. 本轮问的是什么，四臂怎么摆

设计契约 `docs/R7_MAIN_MODEL_V2_DESIGN.md` §3 的 RW-B 有四件：Z（每 patch 工作状态）、
锚定 `X_t` 的绝对提案、逐位置门控、来源 role 标记。§3.2 把 role 标记单列为**较弱候选**，
并要求一轮实验「要么支持它，要么**保留 role 标记与否的对照**作为负结果」。把 role 打进
RW-B 再一起判，负结果就不可读——所以本轮为这条对照单花一臂，而不是把它并进主臂。

**四臂，一个协议，全部从零训练，全部打开时空条件通路**（这样「读/解结构」是唯一变量）：

| 臂 | 开关 | 角色 |
| --- | --- | --- |
| `process_spacetime_only` | 全关（池化读） | **旧 mean-Ours 参照**：`P.mean` 广播，上一代 solver |
| `process_spacetime_rwa` | `positional_process_readout` | **RW-A**：每位置 query 读 process token，加性、无状态、无门控（＝第二轮 C 臂） |
| `process_local_solver` | + `local_solver_state` | **RW-B**：同一读取 + Z + 锚定提案 + 逐位置门控 |
| `process_local_solver_roles` | + `source_role_markers` | **RW-B + roles**：再加两个可学 role 向量 |

与计划 `docs/plans/0004-r7-main-model-v2.md` 的 E1 行（三臂：旧 mean-Ours / RW-A / RW-B）相比，
本轮的**唯一偏离是增加了第四臂**（role 对照），依据是设计契约 §3.2 的明文要求；其余
（同一 M2 store、同一 400-update 协议族、2 seed、≤1.0 GPU-h、只读 val）不变。

**实测代价**（协议冻结前测量，`arm_table.csv`）：

| 臂 | 参数量 | 前向 FLOPs | 前向+反向 FLOPs | 训练峰值 allocated | 训练墙钟（2 seed 合计） |
| --- | --- | --- | --- | --- | --- |
| `process_spacetime_only` | 2,819,267 | 12,927,412,608 | 38,665,111,296 | 219.7 MiB | 324.4 s |
| `process_spacetime_rwa` | 2,968,259（+5.28%） | 13,904,603,520（×1.0756） | 41,596,684,032 | 226.4 MiB | 329.0 s |
| `process_local_solver` | 3,283,157（+16.45%） | 16,820,126,592（×1.3011） | 54,656,320,512 | 254.3 MiB | 351.3 s |
| `process_local_solver_roles` | 3,283,541（+384） | 16,820,126,592（×1.0000） | 54,656,320,512 | 254.3 MiB | 356.7 s |

相对 **RW-A**，RW-B 是 **+314,898 参数（+10.61%）**、前向 FLOPs **×1.2097**、训练峰值 allocated
**+27.9 MiB（+12.3%）**、墙钟 ×1.068。`process_spacetime_rwa` 的参数量 `2,968,259` 与第二轮
C 臂归档数字**逐位相同**，这是共享 step 重构没有改变既有路径的一个独立佐证。

**四臂不是等容量的**：RW-A 与 RW-B 各自增加参数与算力，凡涉及参照臂的对都是容量混杂的；
`RW-B+roles − RW-B` 是**唯一**参数几乎相同（+384）、FLOPs 逐位相同的一对。

## 2. 协议冻结与运行资格

- `protocol.json` 在每 seed **第一次 `optimizer.step()` 之前**以 `'x'` 排他写入，写完**回读**并
  重新计算 digest（JSON 往返无损），再核对 primary 登记项与脚本常量一致；不一致即抛错。
- 8 个 run（2 seed × 4 臂）的 `protocol_sha256` **全部相同**（`merged_result.json` 的
  `run_protocol_sha256` 逐 run 列出）。
- 全部评估 `split == val`；`test_read: false` 写在协议、逐 seed 结果与合并结果三处；
  **没有为方法选择打开过 test**。
- 臂配对**实测而非假定**：四臂由同一 seed 构造，共享张量逐位相同；参照臂的 `state_dict`
  经 trainer 自己的 transfer 规则载入其余三臂，`applied=119 / ignored=0`，载入后逐张量
  `torch.equal`。每条新开关都被断言**确实新增了张量**（RW-A +12、RW-B +20、roles +2），
  否则该轮拒绝运行。
- 四个臂**都没有早停**，全部选中 update 400（`training_table.csv`）。

## 3. 登记的主端点读数（t2m）

登记原文见 `protocol.json` 的 `primary_registration.decision_text`，跑后未改一字：
t2m、5 个时效、对 `process_local_solver - process_spacetime_rwa`（负 = RW-B RMSE 更低）、
只读 comparator 逐 seed 同号才算数。

| 时效 | RW-B − RW-A 逐 seed | 均值 | 三态 |
| --- | --- | --- | --- |
| 6h | seed41 **+0.1986** / seed42 **+0.0372** | **+0.1179 K** | worsened |
| 12h | seed41 **−0.0436** / seed42 **−0.1148** | **−0.0792 K** | supported |
| 24h | seed41 **−0.0221** / seed42 **−0.1274** | **−0.0748 K** | supported |
| 48h | seed41 **+1.0617** / seed42 **+1.0684** | **+1.0650 K** | worsened |
| 72h | seed41 **+1.3440** / seed42 **+1.8168** | **+1.5804 K** | worsened |

`verdict_counts = {supported: 2, worsened: 3}`；headline（脚本按登记文字自动生成）：

> the sign-consistent leads disagree: supported x2, worsened x3 - the per-lead table is the result

**判读**：登记的模态读数是 **worsened**。两个时效上的改善（12/24h，−0.075…−0.079 K）比三个时效上的
恶化（6h +0.118、48h +1.065、72h +1.580 K）**小一个数量级**；48h 的恶化在两个 seed 上几乎相同
（1.0617 与 1.0684，相差 0.6%），72h 相差 30%——方向一致但幅度不稳。登记文字里写的两条
**证伪条件都没有触发**：6h 恶化但 12h 支持（不是「6h 与 12h 双双恶化」），且五个时效全部
sign-consistent（不是「全部 unresolved」）；所以严格按登记文字，这不是「被证伪」，而是
**模态不支持且长时效明确受损**。两者都必须照写。

## 4. role 标记对照（第二格，不并入主格）

`process_local_solver_roles - process_local_solver`，同样的读法：

| 时效 | 逐 seed | 均值 | 三态 |
| --- | --- | --- | --- |
| 6h | −0.0018 / −0.0096 | −0.0057 K | supported |
| 12h | −0.0070 / −0.0153 | −0.0112 K | supported |
| 24h | −0.0047 / −0.0343 | −0.0195 K | supported |
| 48h | −0.0137 / −0.0427 | −0.0282 K | supported |
| 72h | −0.0014 / −0.0400 | −0.0207 K | supported |

五个时效在两个 seed 上**全部同号为改善**，但幅度 0.006–0.028 K，比 RW-B 的恶化小两个数量级；
全 17 变量的聚合计数是 30 改善 / 21 恶化 / 34 未决（48h 那一行是 2 改善 / 10 恶化）。
**读法**：这是「role 标记不是零效应」的证据，**不是**「role 标记有用」的证据——它没有可测的算力
代价（＋384 参数、FLOPs 逐位相同），方向一致且很小，且在一个时效上聚合计数反向。按设计契约
§3.2 的要求，本轮**把这条对照保留下来**（无论正负）。不得据此宣称 role 标记解决了任何问题。

## 5. 其余三对与探针

**聚合三态计数（全 17 变量 × 5 时效 = 85 格）**：

| 对 | improved | worsened | unresolved |
| --- | --- | --- | --- |
| `local_solver − spacetime_only` | 28 | 37 | 20 |
| `local_solver_roles − spacetime_only` | 28 | 36 | 21 |
| `local_solver − rwa` | 29 | 39 | 17 |
| `local_solver_roles − rwa` | 30 | 21 | 34 |
| `rwa − spacetime_only` | 16 | 20 | **49** |

最后一行与第二轮一致：**位置化读取本身在上述池化读取上没有可分辨的效应**（85 格中 49 格未决）。
逐时效计数在 `paired_comparison.json` 的 `per_lead` 里全表照报，长时效（48h）的恶化集中在
6h 与 48h 的 6/8/12 变量上。

**K=1/2/4 深度探针**（同一 K=3 checkpoint 上重跑，**不是**独立训练的 K=1/2/4；t2m）：

| seed | 臂 | K=1 6h | K=2 6h | K=4 6h | K=1 24h | K=4 24h |
| --- | --- | --- | --- | --- | --- | --- |
| 41 | RW-A | 2.770 | 2.742 | 2.908 | 3.965 | 4.646 |
| 41 | RW-B | 2.947 | 2.969 | 3.006 | 4.455 | 4.275 |
| 42 | RW-A | 2.581 | 2.522 | 2.609 | 4.315 | 4.712 |
| 42 | RW-B | 2.546 | 2.546 | 2.691 | 4.239 | 4.650 |

两个 seed 上 K 从 1 加到 4 都不单调改善（6h 上 K=4 一致最差），且 RW-B 在 6h 上**每个 K 都比
RW-A 差**。这与 #66/S4「自适应的前置 gate 不成立」相容，也是 M5 的 adaptive 部分**默认不启动**的
又一条同向观察。

**修正幅度/角度**（`training/r7_correction_diagnostic.collect_correction_terms`，每 seed 8 个
val 窗口，归一化单位）：

| seed | 臂 | 步1 \|d\| | 步1 cos(e,d) | 步3 cos(e,d) | 步3 worsening |
| --- | --- | --- | --- | --- | --- |
| 41 | RW-A | 0.0359 | −0.233 | −0.078 | 0.412 |
| 41 | RW-B | **0.0665** | −0.137 | **+0.070** | **0.610** |
| 42 | RW-A | 0.0297 | −0.203 | −0.007 | 0.500 |
| 42 | RW-B | **0.0543** | −0.109 | **+0.057** | **0.566** |

两条一致的事实：**(a)** RW-B 的第一步修正幅度约为 RW-A 的 **1.8 倍**（能量约 3.4 倍）；
**(b)** RW-B 的误差—修正余弦在第 2–3 步**转正**（修正指向误差同侧＝方向错），而 RW-A 全程为负，
且 RW-B 的第 3 步恶化比例超过一半。这与 E0 的 E-196（「加读取通路会放大修正幅度」）同向，但
**多出一条**：E0 在 C/D 上没有看到余弦变差，RW-B 这里看到了。本轮**没有**做能把这归因到
「门控」还是「锚定提案」还是「Z」的对照，所以这条只是**相容的机制线索，不是归因**。

## 6. 实测 GPU、失败尝试与账本

**权威数字取自产物本身**（`elapsed_seconds` 求和），不是估计：

| 尝试 | 训练 | 评估 | 合计 |
| --- | --- | --- | --- |
| attempt 1（**失败**，`..._attempt1_probe_defect`） | 1376.7 s = 0.3824 h | 136.3 s = 0.0379 h | **0.4203 h** |
| attempt 2（本文件的数字来源） | 1361.4 s = 0.3782 h | 134.0 s = 0.0372 h | **0.4154 h** |
| **本轮合计** | | | **0.8356 GPU-h** |

**attempt 1 为什么失败（如实记录）**：训练与 40 次评估**全部完成**后，脚本在**新的修正探针**里
抛 `KeyError: 'atmos_target'`——`ZarrRolloutDataset` 的样本字段名是 `rollout_targets`（且带一个
时效轴），我在探针里直接沿用了 `collect_correction_terms` 期望的 `atmos_target`。**这是本轮
新写代码的实现缺陷，不是数据或方法问题**。处置：修字段映射并加类型/深度守卫；补一条对照
reader 的回归钉住测试；**保留** attempt 1 的整个产物目录（未删、未改写）；重跑一整个干净尝试，
使所有数字来自**同一次运行、同一份代码修订**。attempt 1 的 RMSE **没有**被采用，也没有被
与本轮数字并列——重跑是修复缺陷，不是「重试凑数」，但它确实花掉了 0.42 GPU-h，这笔账照记。

**账本重算**（开工时从证据文档重算，未沿用提示词数字）：
第二批授权 ≤24 GPU-h；三轮文档起点已用 2.463；此后无新 GPU 运行；
本轮 0.8356 → **已用 3.2986，余 20.701 GPU-h**。

## 7. limitations（协议冻结原文，逐条照抄）

`scientific_claim: false` 写在协议、逐 seed 结果、合并结果与配对结果四处。

- one bounded four-arm run at 400 updates, not a convergence or SOTA comparison
- two seeds: sign agreement across two seeds is consistency, not significance, and no
  significance threshold is introduced or relaxed; two seeds are weaker than the three
  the previous rounds used
- the arms are NOT capacity-matched: RW-A and RW-B each add parameters and FLOPs, so
  every pair here is capacity-confounded and only the measured counts say by how much
- the K=1/2/4 depth probe runs on a checkpoint trained at K=3: not an independently
  trained K=1/2/4 model
- the correction probe reports error/update geometry on validation windows only, and
  its retrospective damping uses future truth: a diagnostic, never a deployable rule
- one winter segment of one year in one region: no seasonal, cross-year or cross-region
  conclusion is testable
- the space-time conditioning pathway is on in every arm, so this round says nothing
  about the pathway itself; round three answered that under its own digest
- the role markers add no measured FLOPs (an additive constant on each half of the key),
  so that contrast is a parameter contrast, not a compute one
- validation-split only; the test split stays sealed and was never opened for selection
- the val climatology is the store's 8-bucket (month, hour) train-only mean, not a
  strong seasonal climatology
- the round-two and round-three numbers are different protocol digests and different
  model code digests: a stability observation only, never pooled with this round's

**本轮特有的三条**：

1. 判决是**在这个 400-update 预算、这个容量档、这个种子对**下的判决；RW-B 的负结果
   不能外推成「局部门控求解状态这个机制不可能有用」。
2. 主端点的**改善**只在 12/24h 且极小（−0.075…−0.079 K，与 48/72h 的恶化不在同一量级），
   而**恶化**在 6/48/72h 且大得多；把 12/24h 单独摘出来讲就是选择性报告。
3. 修正探针的窗口数只有 **8**，per-step 比例是这 8 个窗口上的比例，不是分布估计。

## 8. 未做的事

- 未做**匹配容量/匹配算力**的 RW-B 对照（决策 0023 要求的 matched Generic 前置仍未满足），
  因此本轮**不能**把 RW-B 与 RW-A 之差归因到「门控/Z/锚定」中的任何一项；这条归因需要
  另一个专门设计的控制臂，本轮**没有**做。
- 未做 Z 的消融（有 Z 无门控 / 有门控无 Z / 锚定 vs 累加各自单独开关）——**这是本轮之后
  最该做的一件事**，也是唯一能把 §5 的机制线索变成归因的动作。
- 未跑第三 seed；未跑确认轮（≤2.5 GPU-h，未动用）；未做 E2（过程监督）、E3（真自回归）、E4（确认）。
- 未读 test，未新增数据下载，未租 GPU，未合并 main，未关任何 issue。
- 未做 M3/M4/M5。
- 未把 role 标记的 +0.006…0.028 K 做成任何机制声明。
- 未重新规划已冻结的路线；未改动前三轮的产物与文档。

## 9. 下一项的第一个具体动作

**给 RW-B 做减法而不是加法**：在同一个 400-update 协议族下，把 `local_solver_state` 拆成
三个**各自单独可开关**的部件——(a) 门控 + 锚定提案、(b) Z 的递推本身、(c) role 标记——
用**预声明的单一对比**回答「负结果是这三件里的哪一件造成的」。本轮 §5 的修正几何
（幅度 ×1.8、余弦转正、第 3 步 60% 恶化）指向 (a) 或 (b)，但**本轮的数据不能区分**；
在做出这个拆解对照之前，**不得**再给 solver 加新部件，也**不得**把这轮的 12/24h 改善
当作继续加码的理由。

判据与预算：沿用 `protocol.json` 的 `scientific_claim: false`、只读 val、`'x'` 冻结、逐 seed
同号三态计数；**执行前按决策 0021 重新取授权**并重读账本（本轮结余 20.701 GPU-h）。
若拆解对照仍为负，按目标 §5 的停止条件 3，停止发明新模块，转去重新检查
forecast state / training objective / data regime。

## 10. 本机 conventions 状态与测试计数

`python tools/check_conventions.py`：37 条阻断 **0 违规**；`git show --check` 干净。
R-009 报告基线随本轮从 859/2114 抬到 **885/2180**，同步更新了 `docs/rules/MIGRATION.md` 与
`docs/rules/testing.md`（两处的数字由 `tests/test_check_conventions.py` 直接对 checker 常量断言）；
规模报告标记同步为 R-020=41 / R-021=26 / R-023=17（三个新模块分别 393/145/599 行，
**全部在 R-051 的 600 行硬上限内**，未新增任何 R-051/R-052 例外）。

`pytest -q` 两处口径**都要报**（本地多出来的那些是本地才有的产物）：

| 口径 | 结果 | skip 原因 |
| --- | --- | --- |
| 本机（有 `outputs/` 产物） | **1484 passed, 3 skipped** | 三个 `test_real_data_pipeline` 用例：可选真实数据 fixture 不在版本控制内，且**不允许**用合成数据兜底 |
| 干净检出（`git clone` 后 checkout 本提交） | **1479 passed, 8 skipped** | 上述 3 个，加上 5 个因本地产物缺失而跳过的用例：本轮两个探针测试（M2 段不存在）、`test_arco_regional_bounded`（可选 ARCO 子集）、`test_r7_budget_audit`（已发布区域 store 不在）、`test_r7_time_range_readers`（D1 store 不在） |

**干净检出这一步不是形式**：本轮第一次推送时 CI 在 pytest 步失败，而本机是全绿的——原因是
`docs/rules/size-thresholds.md` 的机器核对标记只统计**未被容忍**的命中，而新增的四个文件在本地
还是未跟踪状态、命中被容忍，标记因此**只在干净检出上才对不上**。这条与 E-193/E-194 是同一类
缺陷（只在 CI 暴露的钉住值），已按 E-200 记录，并在推送前用 `git clone` 复现干净检出才发现。

## 11. 提交与 CI

| 提交 | 内容 | `R7 CPU CI` run | 结果 |
| --- | --- | --- | --- |
| `d8c828d` | 本轮的代码、模块、测试、证据文档与规则文档 | `36587955914` | **失败**，第 6 步 `Check evidence index and candidate brief` |
| `36292c8` | 把索引记录指到 RW-B 文档的新 digest，并登记本轮的负向记录 + 重生成 brief | `36588423695` | **失败**，第 8 步 `Run unit, integration and installed-wheel tests` |
| `b4407c4` | 同步 `size-thresholds.md` 的机器核对标记（R-020 39→41、R-021 24→26、R-023 15→17） | **无独立 run** | 与 `de1e87d` 在同一次 `git push` 里被推送，GitHub 只对推送的 head SHA 触发 workflow，因此该 SHA 没有自己的 run——这一格是「没有证据」而不是「通过」 |
| `de1e87d` | 两处 pytest 口径 + 两次失败的原因 + 用 `git clone` 复现干净检出 | **`36591213197`** | **success**，八步全绿 |

`36591213197` 是**覆盖本轮代码、索引与文档的绿 run**：第 5 步 `Check repository conventions`、
第 6 步 `Check evidence index and candidate brief`、第 7 步 `Compile active modules and check
whitespace`、第 8 步全量 pytest 全部 success；同 SHA 下 18 条 workflow 中 17 条实验 workflow
按 commit-message 标签门控 **skipped**（设计行为，**不算失败，也不当通过**），本轮提交信息
**未**带任何实验标签，因此没有任何实验 workflow 被触发。

两次失败**都是真实原因、都能在本地复现**，不是 flake：

1. 第 6 步失败：本次提交给一个**已经登记在索引里**的证据文档
   （`docs/R7_72_RW_B.md`）加了一段带日期的更新块，而索引仍钉着旧 digest。闸门要求的正是
   「改了已登记的证据就要同步索引」——修法是重指 digest 并如实写明原因，不是放宽闸门。
2. 第 8 步失败：见 §10 最后一段，是只在干净检出上暴露的钉住值。

18 条 workflow 中 17 条实验 workflow 按 commit-message 标签门控 **skipped**（设计行为，
**不算失败，也不当通过**）；本轮提交信息**未**带任何实验标签，因此没有任何实验 workflow 被触发。

## 12. 仍未做的事（明确写下，不靠「应该没问题」）

- **本轮没有重跑 `mimosa` 深度安全扫描**：会话内的 commit/push hook 报
  `scanner_enobufs`（扫描未取得完整结论），因此**不得**把本轮说成「已通过完整安全审计」。
  这是一条交付欠账，不是「应该没问题」。
- 本文件 §11 之外还有一个只改目标进度块的收尾提交；它的 `R7 CPU CI` run 号记在
  `docs/goals/main-model-v2-rw-b-round.md` 的进度块里，**不在本文件**——本文件是已登记的证据页，
  再改一次就要同步索引 digest，因此收尾提交刻意不再改这一页。
