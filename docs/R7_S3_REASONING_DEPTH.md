# S3 推理深度 K 4→8 单因素臂：操作者中止，未完成（单 seed，未达预声明出口）

**状态：未完成（2026-10-12 操作者中止）；`scientific_claim: false`；2023 test 未读；r 不消耗。**
本页对应 `docs/goals/main-model-climatology-campaign.md` §8 与 `docs/goals/s3-reasoning-depth.md`。
它**不是**该臂的正式读数：三 seed 只完成了一个，reading 阶段从未运行，**没有任何 seed 均值**。
登记它的目的是**不留空轮**——把"跑过、被中止、花了多少、看到什么"如实写下来，而不是把它当结论。

## 1. 结论摘要

- **单因素**：`reasoning_steps` / `steps` **K 4→8**（训练与评分同时），其余逐字沿用已登记最佳臂
  （宽 129×129 输入、`interior_32` 监督、2400 更新、注册物理权重 `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)`、
  迁移 v3-BD 1600 父、seed 41/42/43）。
- **中断**：seed 41 **完整跑完**（exit 0），**seed 42 在训练中被 `SIGKILL`**（`exit_code=-9`），
  seed 43 与 reading 阶段**从未运行** ⇒ **无 `readings.json`、无 `result.json`、无 `failure.json`**，
  `attempt.json` 也不存在（驱动在收尾前被打断）。
- **唯一可读的原始数字（seed 41）**：K=8 相对已登记 2400 剂量臂（K=4，pinned 复用、未重训练）的
  `interior_32` t2m 变化为——6 h **+0.00776 K**、12 h **+0.00884 K**、24 h **−0.00856 K**、
  48 h **−0.00309 K**、72 h **−0.00780 K**；skill 变化 |Δ| **≤ 0.00509**。**72 h skill 仍为负**
  （**−0.16954**，K=4 为 −0.17463）。
- **判定**：预声明出口要求"72 h **三 seed** 全正 → 提请独立验收；否则关闭训练配方支"。
  本臂**只完成一个 seed**，**既没有达到出口的正向条件，也不能据此宣布负结论**。
  它是**操作者按成本/收益决定中止**的未完成轮，不是科学判定。
- **对下一轮的作用**：与**已登记**的剂量臂（`record:s3-wide-interior-dose`，72 h 三 seed 全负）
  和长 lead 权重臂（`record:s3-wide-long-lead-weight`，72 h 三 seed 全负）合起来，
  支持**关闭"靠训练配方（输入范围 / 剂量 / 长 lead 权重 / 推理深度）解决 72 h"这一支**；
  但本臂自身的单 seed 数字**只作旁证**，不单独构成该支关闭的充分证据（见 §6）。

| lead | K8 rmse | K4 rmse（已登记） | Δrmse | K8 skill | K4 skill（已登记） | Δskill |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 h | 2.289701 | 2.281941 | **+0.00776** | 0.57618 | 0.57905 | −0.00287 |
| 12 h | 2.877077 | 2.868233 | **+0.00884** | 0.31998 | 0.32415 | −0.00417 |
| 24 h | 2.618599 | 2.627157 | −0.00856 | 0.41969 | 0.41589 | +0.00380 |
| 48 h | 3.245894 | 3.248979 | −0.00309 | 0.06311 | 0.06133 | +0.00178 |
| 72 h | 3.588657 | 3.596453 | −0.00780 | **−0.16954** | −0.17463 | +0.00509 |

（单 seed 41；无 seed 42/43 数字，故**没有** seed 均值、没有同时区间、没有 primary/gate 判定。）

## 2. 身份与协议

| 项 | 值 |
| --- | --- |
| attempt 目录 | `outputs/r7_s3_reasoning_depth_20261009_attempt01/` |
| 协议 digest（`protocol.json`，最终；驱动 `validate_protocol` 自洽校验通过） | `35b84f2a25816c25f5bc0382ecf38dd23a3d2c5cc69938ffd175fa261fa69ca7` |
| （预备阶段 `preparation_protocol.json` 的初步 digest，合并身份字段前） | `d8ba87bbe62cd83ad4b198e1a0fa68c2f19e6a62cd444c119a3de1fdc070b8e0` |
| 归档 code.zip SHA256 | `1ca8e648a87f5f7fcd8155d8bcfc534606a70599858536833fee4a7355eb6406` |
| 代码 commit | `964fc8ce4c4e64b085b297a056962406b381c4e8` |
| 预算 | planned 54000 s / hard 72000 s / per-seed 18000 s |
| GPU | `GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b`（只读余量门，未向任何非本实验进程发信号） |
| 评分域 / 气候态分母 | `boundary_masks(129,129,(32,))['interior_32']`（4225 点）；分母与已登记臂逐字相同 |
| cohort pin | 472 / 468 / 460 / 444 / 428（逐 lead，与已登记臂一致） |

## 3. 逐阶段回执

| 阶段 | 文件 | exit_code | status | elapsed_seconds |
| --- | --- | ---: | --- | ---: |
| prepare | `prepare_process.json` | 0 | success | 510.163 |
| archive | `archive_process.json` | 0 | success | 471.532 |
| seed 41 | `seed41_process.json` | 0 | success | 15432.364 |
| seed 42 | `seed42_process.json` | **−9** | **failed** | 1656.784 |
| seed 43 | —（不存在） | — | — | — |
| reading | —（不存在） | — | — | — |

- **seed 41 回执**（`seed41_receipt.json`）：`updates 2400`、`reasoning_steps 8`、训练 `elapsed_seconds`
  11209.65、checkpoint `efc43304…`、`owned_cuda_reserved_peak_bytes 14552137728`（≈13.55 GiB）、
  五 lead 评分目录与哈希齐全。
- **seed 42**：训练停在 `seed42/training/candidate/update_0000160.pt`（2400 步里的第 160 步），
  **无 `seed42_receipt.json`**。`exit_code=-9` = SIGKILL，来源是**操作者停止 workflow 时连带终止了在跑的
  `world.run` 子进程**（该机制见 `docs/plans/0021-round-interruption-hardening.md` §1）；**不是**科学失败。

## 4. 成本（全额计费，含被中止段）

| 项 | 秒 | GPU-h |
| --- | ---: | ---: |
| prepare | 510.163 | 0.1417 |
| archive | 471.532 | 0.1310 |
| seed 41（完成） | 15432.364 | 4.2868 |
| seed 42（被 SIGKILL） | 1656.784 | 0.4602 |
| **本轮合计（whole 18,070.843 s）** | — | **5.0197** |

网络 0（离线，R-028）；磁盘按产物目录计；soft/hard overrun 均 0（未触硬上限）。被中止的那段 GPU 时间
**全额记账、产物不复活**（决策 0045 / 计划 0021）。

## 5. 已确认与推测

**已确认（读到的产物）**：上面所有 exit_code / elapsed / receipt / 五 lead 评分 CSV / cohort 计数；
seed 41 的 K8 与已登记 K4 的逐 lead RMSE（两者评分域、气候态分母、cohort pin 逐字相同，可直接相减）。

**推测（未证）**：① "K 翻倍对本任务无实质收益"——只有 **1 个 seed**，且 Δ 量级（≤0.009 K）与
已登记的 seed 间波动同量级，**不能排除**另两个 seed 给出更大或反号差异；② 中止的直接原因是
操作者判断收益不足，**没有**任何机器可判的失败条件被触发（无 hard overrun、无身份/资源门失败）。

## 6. limitations

- **未完成**：三 seed 只完成一个；**无读数聚合、无 seed 均值、无同时区间**；reading 阶段从未运行。
- **不是科学判定**：本页不产生 primary/gate 结论，不改变 r、不读 test。
- **旁证而非证据**：关闭"训练配方解决 72 h"支主要依据**已登记**的 `record:s3-wide-interior-dose`
  与 `record:s3-wide-long-lead-weight`（两者 72 h 均**三 seed**全负）；本臂的单 seed 数字只作旁证。
- **失败不追认**：seed 42 的 SIGKILL 段保留、计费、不重跑、不 in-place resume（决策 0045）。
- **可复现等级**：产物与协议 digest 齐备，但**未产出**结果文件，故无法做结果级重放；只能按
  `protocol_sha256` + `code.zip` SHA 复现"同一协议"，不能复现"同一结果"。

## 7. 下一动作

按 `docs/goals/main-model-climatology-campaign.md` §8：**关闭"靠训练配方解决 72 h"这一支**
（依据已登记的两臂三 seed 读数 + 本臂旁证），转**模型容量/架构**方向（首候选：已登记最佳臂上的
容量单因素、同 updates 对照），预声明出口与停止条件写进新的 round brief
`docs/goals/s3-capacity-or-architecture.md`。宽输入支已关闭不重开。test 未读、r=0、S4 未启动不变。
