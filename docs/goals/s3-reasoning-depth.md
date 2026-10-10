# S3：推理深度 K 4 → 8（在已登记最佳臂上，单因素）

<!-- round-node: S3 -->

状态 **未开始**（2026-10-10 交接，本轮自迭代契约的下一轮）。唯一主计划
`docs/goals/main-model-climatology-campaign.md`；承接 `docs/goals/s3-wide-long-lead-weight.md`
§9 的"下一动作"与主计划 §8 的最后一条"下一动作"。

## §0 Objective（单段）

> 在 docs/goals/s3-reasoning-depth.md 接续S3/index70/48.2867GPU-h、2023test未评分r0、S4未启动。前三轮把宽输入支走完：输入 65×65→129×129 本身**没有**收益（全网格监督下 48/72h 三 seed 全负）；把监督域收回目标盒（`interior_32`）后收益出现但只到 48h（skill −0.16/−0.12/+0.01）；再把 updates 800→2400，**48h 首次三 seed 全正**（+0.06/+0.03/+0.11）而 72h 仍全负（−0.17/−0.28/−0.16）；最后把 48/72h 物理权重 0.5→1.0，**几乎无效**（72h 仅改善 ~0.02 K，仍三 seed 全负），预声明出口触发 → **宽输入支作为"长 lead 的解"关闭**。本轮回主线换一个**与输入区域无关**的可证伪假设：**72h 的残余误差是否由每次预报的推理深度不足主导**（12 步自回归里每一步只做 K=4 次内部推理）。单因素：在**已登记的最佳臂**（宽 129×129 输入 + `interior_32` 监督 + 2400 更新 + 注册物理权重 `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)` + 迁移 v3-BD 1600 父 + seed 41/42/43）上，把 `reasoning_steps`/`steps` K 从 **4 提到 8**，其余全部逐字不变；对照是**已登记的 2400 剂量臂**（index `record:s3-wide-interior-dose`，pinned 复用、不重训练）与已登记窄臂。K 同时进入训练与推理，因此这是"每步更多内部推理"的单因素检验。读法：72h 三 seed 转正 → 提请独立验收；否则**关闭"靠训练配方解决 72h"这一支**，转模型容量/架构或数据方向，不再加 seed、不再调 K、不重复同一 test。本轮只报读数，**不宣布科学通过**、不称泛化/SOTA、不自行宣布最终goal完成。不paid/rental/exclusive/main/release/issueclose/force/mirror/破坏性/cron，不改科学合同/冻结证据/用户配置，不读 test。

## §1 起点、证据与实质差异

- 起点工作分支 `r7/weather-reasoning`，HEAD 为上一轮登记提交（累计 48.2867 GPU-h、index70）。
- 起点证据：`docs/R7_S3_WIDE_LONG_LEAD_WEIGHT.md`（§8 三轮合并结论 + §9 下一动作）。
- 与上一轮的实质差异：**只有 K**（4 → 8）。输入区域、监督域、updates、物理权重、父、seed、
  评分域、气候态分母、cohort pin 全部不变。
- 为何这是"回主线"：宽输入支已按预声明关闭；K 是**与输入区域无关**的推理侧因子，
  且直接对应 72h 的误差累积机制（12 步链式预报，每步 K 次内部推理）。

## §2 交付物清单

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 本轮目标与冻结设计决定 | 本文件 §3 |
| D2 | 新协议/新输出/新 code.zip | `preparation_protocol.json`、`prepared_protocol.json`、`code.zip` + `archive_receipt.json` |
| D3 | 三 seed 真实训练与评分 | `seed{41,42,43}_receipt.json`、每 seed 五 lead 的 `boundary_rmse.csv` |
| D4 | 配对读数 | `readings.json`（本臂 vs 2400 剂量臂 vs 窄臂） |
| D5 | 登记与精确 CI | `docs/R7_S3_REASONING_DEPTH.md`、index record、brief、账本行、§8、campaign-state |

## §3 判据与设计决定（本轮冻结）

科学门与验收形式只来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` 与主计划 §8；本轮
**不触碰**任何阈值，也**不宣布**通过。

- **臂**：最佳已登记臂 + **K=8**（`steps=8` 训练、`reasoning_steps=8` 评分）。
- **对照**：**已登记的 2400 剂量臂**（K=4，pinned 复用）为主对照；窄臂为次级参照。两者都不重训练。
- **实现约束**：不改动已登记的驱动与证据；本轮在剂量驱动上派生 `--reasoning-steps 8` 变体或新驱动，
  写进**新输出目录**，`code.zip` 与 `protocol_sha256` 都是新的。
- **预算**：K 翻倍使每步前向变慢，per-seed 期限相应放大（先按上一轮实测 2400 步 × 2 ≈ 12,600 s 估，
  取 per-seed **18000 s**、planned **54000 s**、hard **72000 s**；先冻后跑，不在运行中改）。
- **预声明出口**：72h 三 seed 全正 → 提请独立验收；否则关闭"训练配方解决 72h"这一支，
  转容量/架构或数据方向，不再加 seed、不再调 K、不重复同一 test。
- **test 策略**：`test.jsonl` 从不读取；`test_read=false`、`scientific_claim=false`。

## §4 数字预算与停止

- planned 54000 s（软预算）、hard 72000 s（宽松硬上限）、每 seed 18000 s；软超继续并记 overrun，
  hard 截止即停 attempt 并全额留 `failure.json`，不运行中改硬限、不重试至偶然通过。
- 每阶段 spawn 前做**只读** `gpu_gate`（`GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b`）；
  **禁止**对任何非本实验进程发送信号或做冻结/终止自动化。
- 失败 attempt 原样保留、全额计费、不做 in-place resume。

## §5 明确不做

不重训练任何已登记臂（窄/800 内部监督/2400 剂量/长 lead 权重）、不重跑更早的已登记轮次、不读 test、
不改 `data/raw|interim|processed`、不改归档快照、不合成替代数据、不删部分输出复活、不改科学合同与
冻结证据与判据、不 main 合并/发版/关 issue、不 paid/rental/exclusive/force/mirror/破坏性/cron。

## §6 进度与交接

- 本轮尚未开始。**第一条命令**（CPU 准备）：
  `.venv/bin/python -m pytest -q tests/test_r7_s3_wide_long_lead_weight.py`，
  然后在剂量驱动上派生 `--reasoning-steps 8` 变体（新输出目录、新协议 digest、新 `code.zip`）。
- 若 72h 仍非三 seed 全正，按 §3 的预声明出口**关闭"训练配方解决 72h"这一支**并在 §8 写
  "为何停 + 恢复第一条命令"。
