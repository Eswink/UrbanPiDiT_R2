# S3 主干容量 depth 4→8 单因素臂：**运行尝试失败**（seed 41 CUDA OOM；共驻显存不足）

**状态：本轮 GPU 部分**已实际启动**、在 seed 41 因 `torch.OutOfMemoryError` 失败（2026-10-13）；
`scientific_claim: false`；2023 test 未读；r 不消耗。** 本页记录一次**真实发生过的失败运行**：
驱动/测试/code.zip 就绪、`freeze`/`prepare`/`archive` 全 success、seed 41 训练到 1948.854 s 时 OOM，
**seed 42/43 与 reading 从未运行**，**无 `readings.json`/`result.json`/`attempt.json`** ⇒ **无任何读数、
无 seed 均值、无同时区间、无 primary/gate 判定**。本页**不是**该臂的读数，也**不是**负结果。

> 口径（必须照此读）：这是**一次资源失败**（共驻显卡显存不足），**不是**对该臂科学假设的判定；
> 本轮 §3 预声明出口（"72 h 三 seed 全正 → 提请独立验收；否则关闭加大容量支"）**既未触发也未满足**，
> 因为三 seed 从未跑完。据此**不**写任何通过/失败，也**不**据此单独关闭"加大容量解决 72 h"这一支——
> 该方向由**操作者另行按资源与初衷判断**关闭（见 §4）。

## 1. 实际发生了什么

- **运行载体**：`.zcode/workflow-drafts/r7-s3-capacity-depth-run.dwf.ts`（workflow run
  `dwfrun-c9039cf5-6fca-4fe5-83c2-c888fddfd184`，本会话内启动、通知链路正常）。
- **各阶段**（`outputs/r7_s3_capacity_depth_20261012_attempt04/`）：
  - **独立复核**（clean，与剂量驱动差异 9 处，全部为预注册改动：对照/评分域/父状态 identity-gated 扩张；
    复核者实跑了编码器逐位一致 `torch.equal=True`、pin 表比对、`strace` 证明测试不读 `outputs/`）。
  - `code.zip` 重建（`git archive HEAD`，comment = `370dca0`）、`freeze`、`prepare`、`archive`：**全 success**。
  - **`seed 41`：失败**（`exit_code=1`、`status=failed`、`passed=false`、`elapsed_seconds=1948.854226`、
    soft/hard overrun 均 **0**）。spawn 时余量门**通过**（free 16,412,311,552 B = 15.29 GiB ≥ required
    9,663,676,416 B = 9.0 GiB）。
  - `seed 42`/`seed 43`/`reading`：**从未运行**（前一 seed 失败即停）。
- **产物**：`attempt04/seed41/training/candidate/failed_attempt.json`（`resume_permitted: false`）、
  `fine_tune_started.json`；**无** `readings.json`、**无** `seed41_receipt.json`。

## 2. 根因（已确认，非推测）

`failed_attempt.json.failure_reason` 原文：

> `OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of
> 23.56 GiB of which 4.00 MiB is free. Process 2967988 has 5.15 GiB memory in use. Process 3516490 has
> 9.17 GiB memory in use. Including non-PyTorch memory, this process has 9.18 GiB memory in use. Of the
> allocated memory 8.56 GiB is allocated by PyTorch, and 313.72 MiB is reserved by PyTorch but unallocated.`

- **共驻碰撞**：spawn 余量门是**时间点**检查（通过时 free 15.29 GiB），但训练进行中**第二个邻居作业**
  （pid 3516490，另一用户 `breast_us ... native_competence.py --worker nn2d`）**长大**到 9.17 GiB，
  与既有邻居（pid 2967988，`train_hsi_finetune.py`，5.15 GiB）加本进程（9.18 GiB）合计 ≈23.5 GiB，
  把 24 GiB 卡占满 ⇒ OOM。**本仓全程未对任何邻居发信号**（决策 0026）。
- **本臂真实足迹 = 9.18 GiB**，高于冻结估计 8.5 GiB（`ESTIMATED_PEAK_BYTES = 8704 MiB`）约 0.7 GiB。
  深度 4→8 相对已登记剂量臂（K=4，实测 reserved peak 8562 MiB）确有增长，但**主因是邻居长大**，
  不是本臂单独超限——只靠调高门槛无法防止邻居在运行中长大（门槛够高就等于回到"无限等待"）。

## 3. 这条失败能支持什么、不能支持什么

- **已确认**：seed 41 确实启动并训练了 1948.854 s 后 OOM；三 seed 未跑完；无读数；本次运行**无科学结论**。
- **不能支持**：把本页当作 depth 4→8 的**科学读数**（没有读数）；也不能据此说"加大容量无效"——
  该假设**未被检验**（运行因资源失败中断）。
- **可支持一条工程事实**：在**本机共驻**、邻居作业可长大的条件下，depth-8（K=4）单因素臂的 ~9.2 GiB
  足迹**无法稳定容纳**——这是**资源不可行**，不是科学结论。

## 4. 操作者决定（2026-10-13）：这个方向**不能这样做**

- 操作者的**初衷**是：用**中小型模型、较小的 GPU 显存占用**，在与其它模型**统一的环境水平**下超过
  气候态并进而超过它们（先超过气候态，再谈 SOTA）。
- 实测：仅**主干深度翻倍**这一支，训练就要 **~9.2 GiB** 显存；本机是**公用显卡**（默认共驻，邻居
  可占 ~14 GiB），**不可能**稳定给出这么大的显存。
- 因此操作者判定：**"靠加大容量解决 72 h"这一方向与本项目"小模型 / 小显存"的前提冲突，且在本机
  共驻条件下不可行** ⇒ **诚实标记为"不能这样做"**。这**不是**科学证伪，是**资源与方向前提**的判断；
  后续由操作者在**新窗口重新规划**（架构 / 数据 / 目标组织等其它方向）。
- 因此本臂**不再重跑**、不加 seed、不重试至偶然通过；已产出的部分输出**原样保留、不删不复活**。

## 5. 成本

| 项 | 值 |
| --- | --- |
| 实测 GPU-h | **0.5414**（attempt04 seed 41 训练 1948.854226 s；保守口径，含失败全额） |
| 网络 | 0 |
| soft / hard overrun | 0 / 0 |
| 磁盘 | attempt04 ≈15 MB（`code.zip` 10.4 MB + 协议 2.3 MB ×2 + seed41 小文件）；attempt01–03 为早期 prepare/archive（0 GPU-h，原样保留） |
| test | 未读（`test_read=false`）；r=0 |

## 6. 身份与可核查证据

| 项 | 值 |
| --- | --- |
| 驱动 | `scripts/study_r7_s3_capacity_depth.py`（本轮未改） |
| 本次运行 code 身份 | `code_commit 370dca0aa12aaabdbf96438e3a0e697140416135`、`code.zip SHA256 ad15b3d4a9cb0beacfd327f4c6069455b302b642c80e0e2fd194331c8b65bf7f` |
| 该 code 的 CI | `R7 CPU CI` run `38262038297` completed/success（对 `370dca0`） |
| attempt 目录 | `outputs/r7_s3_capacity_depth_20261012_attempt04/`（失败产物保留） |
| 失败回执 | `attempt04/seed41/training/candidate/failed_attempt.json`（`resume_permitted: false`） |

## 7. limitations

- **未完成**：三 seed 未跑完（仅 seed 41 启动并失败），无 `readings.json`，本轮 §3 预声明出口
  **既未触发也未满足**；**无任何科学读数**。
- **失败性质是资源**（共驻显存 OOM），**不是**科学失败；本页不作 primary/gate 结论，不改 r、不读 test。
- **可复现等级**：GPU 训练本就非逐位可复现；本页无结果产物可复现，只有 code 身份
  （`370dca0` + `ad15b3d4…`）与失败回执可核查。
- **余量门是时间点检查**：无法防止邻居在运行中长大；本页不把"调高门槛"当作本轮的修复。
- **邻居身份为只读观测**（`nvidia-smi` + `ps`），未对邻居发任何信号。

## 8. 下一动作（S3）

**goal 已由操作者暂停**，本臂**不重跑**。下一动作：操作者在**新窗口重新规划方向**
（架构 / 数据 / 目标组织），并按 §4 的"小模型 / 小显存"前提约束**新的 round brief**。
恢复方式：新窗口核实际 HEAD/status、只读 master §8 与 §2/§3/§5、核两个校检器；
若仍要重跑本臂，第一条命令是 `.venv/bin/python tools/r7_round.py round-guard`
（前置：`code.zip` 的 ZIP comment 必须等于 HEAD，否则先按 HEAD 重建）。test 未读、r=0、S4 未启动不变。
