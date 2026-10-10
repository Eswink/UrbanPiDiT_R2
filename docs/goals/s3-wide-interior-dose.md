# S3：宽输入 + 内部盒监督的剂量提升（800 → 2400 updates）

<!-- round-node: S3 -->

状态 **未开始**（2026-10-09 交接，本轮自迭代契约的下一轮）。唯一主计划
`docs/goals/main-model-climatology-campaign.md`；承接 `docs/goals/s3-wide-interior-supervision.md`
§9 的"下一动作"与主计划 §8 的最后一条"下一动作"。

## §0 Objective（单段）

> 在 docs/goals/s3-wide-interior-dose.md 接续S3/index68/34.5899GPU-h、2023test未评分r0、S4未启动。上一轮把监督域从全 129×129 收回 `interior_32`（= 冻结中心 65×65 盒 = 已登记窄臂自己的网格）后，宽输入的收益第一次出现：seed 均值 delta（本臂 − 窄臂）五个 lead 全为负（6h −0.0348、12h −0.0401、24h −0.0069、48h −0.0528、72h −0.0445 K），6/12h 三 seed 全优，skill 也全面高于窄臂与上一轮全网格臂；但 48/72h 的**绝对气候态门仍未过**（只有 seed 43 的 48h 为正 +0.0140），因此**仍未进 S4 冻结包**。本轮做**单因素剂量提升**：在同一冻结监督域（`interior_32`）、同一宽 store、同一配方族（`long_rollout`、12 步物理权重、LR 2e-5、warmup 10、FP32、K4、batch 1、clip 1、seed 41/42/43）、同一迁移 v3-BD 1600 父下，把 updates 从 800 提到 **2400**，检验收益是否随剂量放大到 48/72h 绝对门；剂量是该族唯一已登记有正响应的杠杆（rollout-dose 200→800）。对照同时报告**已登记窄臂**（index `record:s3-rollout-dose`，pinned 复用、不重训练）与**上一轮 800 内部监督臂**（index `record:s3-wide-interior-supervision`）。读法：2400 更新后 48/72h 三 seed 全为正且 seed 均值同时区间下界 > 0 → 该支值得提请独立验收；仍不过绝对门 → 按防空转纪律**停该支**、转别的可证伪假设，不再加 seed、不重复同一 test、不无限加剂量。本轮只报读数，**不宣布科学通过**、不称泛化/SOTA、不自行宣布最终goal完成。不paid/rental/exclusive/main/release/issueclose/force/mirror/破坏性/cron，不改科学合同/冻结证据/用户配置，不读 test。

## §1 起点、证据与实质差异

- 起点工作分支 `r7/weather-reasoning`，HEAD 为上一轮登记提交（累计 34.5899 GPU-h、index68）。
- 起点证据：`docs/R7_S3_WIDE_INTERIOR_SUPERVISION.md`（稀释已分离、收益小、绝对门未过）。
- 与上一轮的实质差异：**updates 800 → 2400**（唯一改动因子）。监督域、输入区域、配方其余参数、
  父、seed、评分域、气候态分母、cohort pin 全部不变。
- 剂量依据：`docs/R7_S3_ROLLOUT_DOSE.md`——同一配方族里 200 → 800 更新把 24/48/72h 的 RMSE
  降低 0.154/0.325/0.465 K，是该族唯一有正响应的杠杆。

## §2 交付物清单

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 本轮目标与冻结设计决定 | 本文件 §3 |
| D2 | 新协议/新输出/新 code.zip | `preparation_protocol.json`、`prepared_protocol.json`、`code.zip` + `archive_receipt.json` |
| D3 | 三 seed 真实训练与评分 | `seed{41,42,43}_receipt.json`、每 seed 五 lead 的 `boundary_rmse.csv` |
| D4 | 配对读数 | `readings.json`（本臂 `interior_32` vs 窄臂 `full` 与 vs 上一轮 800 臂的逐 seed 逐 lead delta） |
| D5 | 登记与精确 CI | `docs/R7_S3_WIDE_INTERIOR_DOSE.md`、index record、brief、账本行、§8、campaign-state |

## §3 判据与设计决定（本轮冻结）

科学门与验收形式只来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` 与主计划 §8；本轮
**不触碰**任何阈值，也**不宣布**通过。

- **臂**：宽 129×129 输入 + `interior_32` 监督 + **2400 updates**（其余逐字沿用上一轮）。
- **对照**：已登记窄 65×65 rollout-dose 800 臂（pinned 复用）+ 上一轮 800 内部监督臂（pinned 复用）；
  两者都**不在本协议内重训练**。
- **实现约束**：不改动上一轮的驱动与证据（已登记身份）；本轮在上一轮驱动上新增
  `--updates` 变体或新驱动文件，写进**新输出目录**，`code.zip` 与 `protocol_sha256` 都是新的。
- **预算**：updates ×3 → 每 seed 期限相应放大（先按 800 臂实测 3765 s × 3 ≈ 11,300 s 估，
  取 per-seed 12600 s、planned 36000 s、hard 54000 s；先冻后跑，不在运行中改）。
- **读法**：2400 后 48/72h 三 seed 全正且 seed 均值同时区间下界 > 0 → 提请独立验收；
  否则停该支、转别的可证伪假设（防空转）。
- **test 策略**：`test.jsonl` 从不读取；`test_read=false`、`scientific_claim=false`。

## §4 数字预算与停止

- planned 36000 s（软预算）、hard 54000 s（宽松硬上限）、每 seed 12600 s；软超继续并记 overrun，
  hard 截止即停 attempt 并全额留 `failure.json`，不运行中改硬限、不重试至偶然通过。
- 每阶段 spawn 前做**只读** `gpu_gate`（`GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b`）；
  **禁止**对任何非本实验进程发送信号或做冻结/终止自动化。
- 失败 attempt 原样保留、全额计费、不做 in-place resume。

## §5 明确不做

不重训练窄臂、不重训练上一轮 800 臂、不重跑更早的已登记轮次、不读 test、不改
`data/raw|interim|processed`、不改归档快照、不合成替代数据、不删部分输出复活、不改科学合同与
冻结证据与判据、不 main 合并/发版/关 issue、不 paid/rental/exclusive/force/mirror/破坏性/cron。

## §6 进度与交接

- 本轮尚未开始。**第一条命令**（CPU 准备）：
  `.venv/bin/python -m pytest -q tests/test_r7_s3_wide_interior_supervision.py`，
  然后在上一轮驱动上派生 `--updates 2400` 变体（新输出目录、新协议 digest、新 `code.zip`）。
- 若 2400 更新后仍不过绝对门，按 §3 的读法**停该支**并在 §8 写"为何停 + 恢复第一条命令"。
