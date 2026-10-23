# 0055 — 训练年份覆盖轮的预算更正：按本轮实测把每 seed 预算 21,600 s 提到 43,200 s

- **状态**：accepted
- **日期**：2026-10-23
- **范围**：`docs/goals/s3-year-coverage.md` 的 S3 年份覆盖轮（决策 0054 的预算条款更正，不新开轮）

## Context

决策 0054 第 6 条把本轮的预算沿用决策 0053 在**月份覆盖轮**上实测的刻度：per-seed **21,600 s**（软）、
round planned **43,200 s**、hard **86,400 s**。那个刻度来自 2,140 / 7,302 窗的两臂实测
（单臂 7,169.6–7,201.4 s）。

本轮的两臂规模**不是同一个量级**：控制臂 **7,291** 窗（与月份轮候选臂同一批窗口），
候选臂 **32,142** 窗（4.4×）；store 也从 9,244 帧涨到 **34,084** 帧（3.7×）。

`attempt01`（协议 `740eae7b…`，code commit `64248e5`）的 `prepare` 阶段给出了本轮第一个**实测**：
该阶段跑的是与两臂训练**同一条校验路径**——`dataset_identity()` 与
`preflight_long_rollout_windows()` 都要构造 `ZarrAtmosWindowDataset` / 逐窗 `validate_record`
（每个窗 12 个 rollout 目标，共约 41.8 万次记录校验）。实测该阶段已运行 **>95 min 仍在进行**，
遂停止该 attempt；随后一次独立只读探针把同一条路径跑完并计时：**123 min 12.216 s**
（user 115 min 54 s、sys 13 min 31 s），而月份轮同阶段只需数分钟。同一探针给出两臂的冻结身份：
控制 `input 7,291 / usable 7,291 / excluded 0`、identity `71bb33e0…`、`window_sha256 b778893f…`；
候选 `input 32,142 / usable 32,131 / excluded 11`、identity `81307008…`、`window_sha256 e8f22c30…`；
val identity `e5fb694f…`。

据此估计**每 seed 的清单校验开销**约 3,700 s（单进程、两臂合计；探针做了约两倍于训练的工作），
双卡并发时再翻倍到约 7,400 s；加上月份轮实测的 train+eval 14,370 s，每 seed 约 **18,100–24,000 s**，
已经贴住甚至越过 21,600 s 的上限。

**这不是「放宽判据」**：(a)–(d) 出口、两臂单因子约束、评分域、气候态分母、updates/LR/K/depth/dim、
seed、val 划分全部不动。改的只有**时间预算**，而 D6 明确允许「下一独立 protocol 可加时」。
也**不**在运行中改硬限、**不**重试以求偶然通过：`attempt01` 原样保留为失败 attempt，
本轮以**新协议**重跑。**没有任何 seed 运行过，因此 `attempt01` 的 GPU-h 为 0。**

## Decision

1. **新预算（`attempt02` 起生效）**：`PER_SEED_SECONDS = 43,200 s`（12 h）、
   `PLANNED_SECONDS = 86,400 s`（24 h）、`HARD_CAP_SECONDS = 172,800 s`（48 h）。
   比例结构（per-seed : planned : hard = 1 : 2 : 4）与 0050/0053 一致。
   43,200 s 是上面估计上界 24,000 s 的 **1.8×**，与月份轮的 1.5× 同量级——因为本轮的校验开销
   虽已实测，但双卡并发会同时抬高两个 worker 的墙钟，且 store 是本轮首次真实使用。
2. **峰值估计不变**：仍写 `estimated_peak_bytes = 4,096 MiB` + margin 512 MiB + neighbour
   reserve 6 GiB。本轮不新增参数、不改模型，显存需求与月份轮同量级。
3. **`attempt01` 如实保留**：只有 `preparation_protocol.json`，无 seed、无读数、**0 GPU-h**；
   不做 in-place resume，不删。本轮读数全部来自新协议。
4. **科学门、单因子约束、评分域、气候态分母与 seed 一律不动**；不放宽任何已冻结判据，
   也不因为「跑得慢」而减少 seed 数或窗口数。

## Consequences

- **代价**：本轮的墙钟上限从 24 h 抬到 48 h，per-seed 从 6 h 抬到 12 h。软预算超了继续并记
  overrun；只有 hard 截止会截断 attempt。预算抬高的**唯一**后果是跑得久，它**不能**让读数变好，
  因此不构成对结果的偏置。
- 若 43,200 s 仍被撞到，按同一纪律再记一次失败 attempt 与新的更正决策，**不**在运行中改硬限。
- `attempt01` 的 prepare 阶段实测（>95 min）作为本轮校验开销的**第一手成本证据**登记进账本说明；
  它不是科学读数，也不是 PASS。
- `scientific_claim: false`、`test_read: false`；不宣布科学通过、不称泛化/SOTA、不自宣最终 goal 完成。
