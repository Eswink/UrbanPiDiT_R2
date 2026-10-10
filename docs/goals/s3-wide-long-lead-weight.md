# S3：长 lead 物理权重再分配（固定 2400 更新，只改 48/72 h 的权重）

<!-- round-node: S3 -->

状态 **未开始**（2026-10-09 交接，本轮自迭代契约的下一轮）。唯一主计划
`docs/goals/main-model-climatology-campaign.md`；承接 `docs/goals/s3-wide-interior-dose.md`
§9 的"下一动作"与主计划 §8 的最后一条"下一动作"。

## §0 Objective（单段）

> 在 docs/goals/s3-wide-long-lead-weight.md 接续S3/index69/41.2840GPU-h、2023test未评分r0、S4未启动。上一轮把宽输入 + `interior_32` 监督臂的 updates 从 800 提到 2400，剂量响应强且单调：相对已登记窄臂在 12/24/48/72h **3/3 seed 更优**，48h 均值低 0.3086 K、72h 低 0.3306 K，且 **48h 的绝对气候态门首次三 seed 全为正**（skill +0.0613/+0.0277/+0.1127）；但 **72h 三 seed 仍全负**（−0.1746/−0.2773/−0.1562），冻结的验收条件（48/72h 三 seed 全正）只满足一半，故**不进 S4 冻结包**。按上一轮预声明的防空转读法，**不再继续加剂量**（避免无限加剂量练到赢），改测另一个可证伪假设且仍只改一个因子：**72h 的残余误差是否由长 lead 训练权重不足主导**。注册配方的 12 步物理权重是 `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)`，48/72h 与 12/24h 同为 0.5。本轮在**完全相同的 2400 更新、同一宽 129×129 输入、同一 `interior_32` 监督、同一迁移 v3-BD 1600 父、同一 seed 41/42/43、同一评分域与气候态分母**下，把 48h 与 72h 的物理权重从 0.5 提到 **1.0**（其余不动）；对照是**上一轮已登记的 2400 臂**（index `record:s3-wide-interior-dose`，pinned 复用、不重训练）与已登记窄臂。读法：72h 三 seed 转正 → 长 lead 权重是有效杠杆，该支值得提请独立验收；仍不全正 → 按预声明**关闭整个宽输入支**，回主线别的可证伪假设（推理深度 K / 架构）。本轮只报读数，**不宣布科学通过**、不称泛化/SOTA、不自行宣布最终goal完成。不paid/rental/exclusive/main/release/issueclose/force/mirror/破坏性/cron，不改科学合同/冻结证据/用户配置，不读 test。

## §1 起点、证据与实质差异

- 起点工作分支 `r7/weather-reasoning`，HEAD 为上一轮登记提交（累计 41.2840 GPU-h、index69）。
- 起点证据：`docs/R7_S3_WIDE_INTERIOR_DOSE.md`（剂量响应单调、48h 三 seed 全正、72h 仍负）。
- 与上一轮的实质差异：**只有 48h/72h 的物理权重**（0.5 → 1.0）。updates 固定 2400，
  其余逐字不变。相对已登记窄臂与 800 内部监督臂则同时有剂量与权重两处差异，故对照以
  **2400 臂**为主。
- 权重改动为何是单因素：物理权重只出现在损失加权和里（`physical_weights`），
  它同时进入绑定 contract 与 `signature`，因此新臂有新身份、可逐位区分。

## §2 交付物清单

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 本轮目标与冻结设计决定 | 本文件 §3 |
| D2 | 新协议/新输出/新 code.zip | `preparation_protocol.json`、`prepared_protocol.json`、`code.zip` + `archive_receipt.json` |
| D3 | 三 seed 真实训练与评分 | `seed{41,42,43}_receipt.json`、每 seed 五 lead 的 `boundary_rmse.csv` |
| D4 | 配对读数 | `readings.json`（本臂 vs 2400 臂 vs 窄臂 vs 800 内部监督臂） |
| D5 | 登记与精确 CI | `docs/R7_S3_WIDE_LONG_LEAD_WEIGHT.md`、index record、brief、账本行、§8、campaign-state |

## §3 判据与设计决定（本轮冻结）

科学门与验收形式只来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` 与主计划 §8；本轮
**不触碰**任何阈值，也**不宣布**通过。

- **臂**：宽 129×129 输入 + `interior_32` 监督 + **2400 updates** + 物理权重
  `(1,.5,0,.5,0,0,0,1.,0,0,0,1.)`（只把第 8、12 个非零项从 .5 提到 1.0，即 48h/72h）。
- **对照**：**上一轮已登记的 2400 臂**（pinned 复用）为主对照；窄臂与 800 内部监督臂作为
  次级参照。三者都不在本协议内重训练。
- **实现约束**：不改动上一轮的驱动与证据（已登记身份）；本轮在上一轮驱动上派生
  `--long-lead-weight` 变体或新驱动文件，写进**新输出目录**，`code.zip` 与 `protocol_sha256` 都是新的。
- **预算**：与上一轮同量级（2400 更新实测 ≈ 21392 s whole），per-seed 12600 s、
  planned 36000 s、hard 54000 s；先冻后跑，不在运行中改。
- **预声明出口**：72h 三 seed 全正 → 提请独立验收；否则**关闭整个宽输入支**，
  回主线别的可证伪假设，不再加 seed、不再调权重、不重复同一 test。
- **test 策略**：`test.jsonl` 从不读取；`test_read=false`、`scientific_claim=false`。

## §4 数字预算与停止

- planned 36000 s（软预算）、hard 54000 s（宽松硬上限）、每 seed 12600 s；软超继续并记 overrun，
  hard 截止即停 attempt 并全额留 `failure.json`，不运行中改硬限、不重试至偶然通过。
- 每阶段 spawn 前做**只读** `gpu_gate`（`GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b`）；
  **禁止**对任何非本实验进程发送信号或做冻结/终止自动化。
- 失败 attempt 原样保留、全额计费、不做 in-place resume。

## §5 明确不做

不重训练已登记的窄臂/800 内部监督臂/2400 臂、不重跑更早的已登记轮次、不读 test、不改
`data/raw|interim|processed`、不改归档快照、不合成替代数据、不删部分输出复活、不改科学合同与
冻结证据与判据、不 main 合并/发版/关 issue、不 paid/rental/exclusive/force/mirror/破坏性/cron。

## §6 进度与交接

- 本轮尚未开始。**第一条命令**（CPU 准备）：
  `.venv/bin/python -m pytest -q tests/test_r7_s3_wide_interior_dose.py`，
  然后在上一轮驱动上派生 `--long-lead-weight 1.0` 变体（新输出目录、新协议 digest、新 `code.zip`）。
- 若 72h 仍非三 seed 全正，按 §3 的预声明出口**关闭宽输入支**并在 §8 写
  "为何停 + 恢复第一条命令"。
