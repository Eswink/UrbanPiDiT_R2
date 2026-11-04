# 0089 — Stage B 第二点（1 epoch / 8,033 updates）：把曲线从一条臂改成一条**共享引擎**的点阶梯

- **日期**：2026-11-04
- **状态**：accepted
- **范围**：`scripts/r7_s3_curve_point_protocol.py`、`scripts/r7_s3_curve_point_driver.py`、
  `scripts/study_r7_s3_throughput_physics_b_p1000.py`、
  `tests/test_r7_s3_curve_points.py`，以及执行源 `docs/goals/s3-throughput-physics-b-p1000.md`
- **依据**：决策 0084（六 Stage 与判据拆分）、0086/0087（同步守卫与 checkpoint 策略）、
  0088（第一点的执行设计）、`docs/goals/s3-throughput-and-physics-campaign.md` §3、
  `docs/plans/0028-throughput-physics-campaign.md` §2

## Context

**① 第二点的预算必须按第一点的实测重算，不能沿用任何估计。**
第一点实测训练 **2,358.3 s / 2,008 updates = 1.174 s/update**、五 lead 评分 **2,633.8 s**，
合计 **1.3867 GPU-h**。此前的层估计是约 12 GPU-h，**被实测推翻 8.7 倍**（0088 负面后果）。
因此本点的 `train_seconds_estimate` 由 `8,033 × 1.174 = 9,430.742 s` **推导**，
`scoring_allowance_seconds` 直接取第一点实测的 `2,633.8 s`（不是 3,600 的圆整数），
`point_seconds_estimate = 12,064.542 s ≈ 3.35 GPU-h`。测试断言这三个数是乘积关系而非巧合。

**② 第一点冻结的模块不能被「统一」掉，只能被并列。**
`scripts/r7_s3_throughput_physics_b_protocol.py` 与其驱动的 SHA256 已随第一点证据登记进
`execution_files_sha256`（90 项具名闭包）。把它改成「带 POINT 参数」会让那份已登记的身份
在事后悄悄不再可复算——这正是 R-006 与决策 0067 要防的事。所以**第一点一个字节不动**，
第二至四点改用新的参数化模块，每个点是一条只写死 label 的薄绑定。

**③ 剩余三点若各自复制 633 行，会把 R-051 的 600 行上限与维护成本同时踩穿。**
第一点的协议层与驱动各 308/325 行，是**单点**的量。参数化后协议层约 330 行、驱动约 300 行，
每个点只需约 40 行的绑定（`POINT`、预算常量、`freeze` 转发、`main` 守卫）。
三条测试直接钉住「绑定必须有自己的具名入口文件」（决策 0067 不许 glob）。

**④ 第一点实际踩过、并已修的两个缺陷必须在共享引擎里成为结构性约束。**
(a) 驱动缺 `if __name__ == "__main__"` 守卫 ⇒ `stage prepare` 曾 0.06 s「成功」却没产物；
(b) 协议声明 `checkpoint_every 5000` 但驱动没传参 ⇒ 实写 101 个 checkpoint。
两者的守卫现在是测试：`test_the_binding_is_runnable_as_a_script` 与
`test_the_runner_is_told_the_frozen_checkpoint_cadence`（源码级断言）。
共享引擎模块**不是**入口点，所以它的 `__main__` 守卫**抛 `SystemExit`** 说明自己是引擎，
而不是静默挑一个点跑——「静默成功」正是 (a) 的形态。

**⑤ 一个点仍然不能判定饱和，这一条在每个点上都要重新写明。**
本点的预声明出口与第一点同构，是工程性的 (a)–(d)；饱和规则逐字进协议、
`saturation_verdict` 恒为 `null`。第三/四点以及饱和判定是**各自的**独立工作。

## Decision

**1. 第二点跑 `8,033 updates × batch 4`（1.000031 epoch）、单 seed 41、单臂。**
`POINT = "p1000"`，`UPDATES` 由 `updates_for(POINT)` 取得；未知 label **拒绝**而不是取默认值
（`test_an_unknown_point_is_refused_not_defaulted`）。`epochs` 记精确比 `32132/32131`，
`nominal_epochs` 记 `1.0`，两者并存以免把标签当成算术。

**2. 预声明出口是工程性的 (a)–(d)，与第一点同构**：
(a) 精确 `8,033` updates、batch `4`；(b) 五 lead × 17 变量全有限、cohort 逐字相符；
(c) 72 h t2m skill 有限且分母是注册 D2 值；(d) 无判据被改。
任一假 ⇒ 该点作废、**从头重跑**（无断点续跑，不从上一点续）。

**3. 时间包络由第一点的实测推导，并留 3 倍余量。**
`per_seed 36,000 s`（约 `point_seconds_estimate` 的 2.98 倍，覆盖第一点实测的 +4.1% 训练方差
与更慢的卡）、`planned 39,600 s`、`hard 79,200 s`（保持 1:2）。软预算超了继续并记 overrun，
只有硬上限因时长截断。测试断言 `per_seed > point_seconds_estimate`，否则本点会被时长截断。

**4. 峰值显存估计取第一点**实测**的 `9,372,172,288 B`（8.729 GiB），不是 `6,270 MiB` 的旧猜值。**
同一 batch/配方/loader，故这是测量而不是估计；协议里 `estimate_provenance` 明写出处，
`test_the_peak_memory_estimate_is_point_one_s_measurement` 钉住这一点。

**5. 十七变量清单逐字取自 store 的 `channels`，只有 t2m 有 skill 分母。**
其余 16 个变量报原始 RMSE，`skill = null`。缺 key 时 `climatology_denominator` **raise**
（不发明分母）——这是第一点 `KeyError: 'leads'` 事故之后的结构性修法。

**6. 共享引擎的读法与 Stage A 一致**：`fine_tune_long_rollout(..., loader=(4, 4))`、
`checkpoint_every` 显式传参、`deny_network()` 在任何阶段分支之前安装。
因此本点的 `training_code_sha256` 与第一点**相同**（同一套 Stage A 代码），
与 R1/R2 及更早轮次**不同**——这一点写进 `limitations`。

## Consequences

**负面后果（代价，必须如实列出）**

1. **多了一套并列的协议层。** 第一点用旧模块、第二至四点用新模块，仓库里因此同时存在两份
   几乎相同的冻结逻辑。这是**刻意**的：统一它们会让第一点已登记的执行闭包 SHA256 失效。
   代价是后续若改配方，两处都要改，且有漂移风险——缓解是两套都跑同一批测试
   （`test_r7_s3_curve_points.py` 与 `test_r7_s3_throughput_physics_b.py`），
   但**没有**任何自动断言能证明两者逐行等价。
2. **本点的估计仍是估计。** `9,430.742 s` 训练来自第一点的 `1.174 s/update`，若卡况不同
   （邻居占用、时钟降频）真实值会偏离。`per_seed 36,000 s` 的 3 倍余量是为此留的，
   不是对精度的声明。
3. **n=1 依旧不给区间。** 与第一点同一理由：加 seed 会把「曲线形状」偷换成「单点显著性」，
   而那在四点各 n=1 的前提下本就不成立。
4. **`training_code_sha256` 与历史臂不同**，故不能与 R1/R2 作代码身份层面的对照；
   能对照的只有「同配方、同数据、同父、不同 updates」这一层。
5. **本点跑完也只是一个纵坐标。** 不得据此判饱和、不得自宣 Stage B 完成。
6. **第三/四点仍各需一次独立 attempt。** 32,132 / 128,528 updates 分别约 10 / 40 小时训练
   （按 1.174 s/update 推算），中途失败即整轮作废、从头重跑。

**正面后果**：第二至四点共用一份机器，每点新增约 40 行；第一点的冻结身份**原封不动**；
第一点踩过的两个缺陷在共享引擎里有会失败的反证测试；预算与显存估计都从**实测**推导，
并可被测试复算。

**否决**：把第一点的模块改成参数化（会让已登记的执行闭包 SHA256 事后失效）；
给共享引擎一个默认点以便「直接可跑」（静默跑错点，正是第一点那个 0.06 s 静默成功的形态）；
沿用 `6,270 MiB` 的旧峰值估计（第一点已实测 8.729 GiB，继续用猜值会低报显存门）；
把 16 个变量也编 skill 分母（自造判据）；加 seed 到 3（改判据的另一种形式）。
