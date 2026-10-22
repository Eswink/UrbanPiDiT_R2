# 0053 — 全年覆盖轮的预算更正：按首次真实使用实测把每 seed 预算 9,000 s 提到 21,600 s

- **状态**：accepted
- **日期**：2026-10-22
- **范围**：`docs/goals/s3-fy-annual-coverage.md` 的 S3 数据覆盖轮（决策 0050 的预算条款更正，不新开轮）

## Context

决策 0050 第 4 条把本轮的预算冻为「per-seed **9,000 s**（软）、round planned **21,600 s**、
hard **43,200 s**」，并在同一条里注明这些数字是**估计**（按 H-D 宽臂实测峰值与 token 格点比例
缩放），首次 seed 后按实测修正。

`attempt02`（协议 `23235448…`，code commit `27cfc39`）是第一次真实使用，实测结果：

| 项 | 冻结值 | 实测 |
| --- | --- | --- |
| 单臂 2,400 updates 训练 | — | **4,522.2 s**（seed41）/ **4,427.2 s**（seed42） |
| 单臂五 lead 评分合计 | — | **2,679.2 s**（seed41）/ **2,742.4 s**（seed42） |
| 单臂合计（训练 + 评分） | — | **7,201.4 s / 7,169.6 s** |
| 一 seed 两臂 | 9,000 s（软） | **约 14,400 s** |
| 峰值保留显存 | 4,096 MiB（估计） | **2,322 MiB** |

因此每 seed 预算被低估约 **1.6×**。后果是 `attempt02` 的 seed41/seed42 在完成 **control 臂**
（训练 + 五 lead 评分，receipt 已落盘）后、treatment 臂训练中途被 `run_stage` 的
`per_seed_seconds + 60` 超时杀掉：

```
subprocess.TimeoutExpired: ... '--worker', 'seed', '--out', ... '--deadline', '984484.741111612',
'--seed', '41']' timed out after 9059.999915540917 seconds
```

同时暴露一个**工具缺陷**：该超时以未捕获异常的形式从 `default_runner` 逃逸，整个
`stage-seeds` 调用带着 traceback 崩溃、连一行 JSON 都没输出。两者都在本决策里更正。

**这不是「放宽判据」**：(a)–(d) 出口、两臂单因子约束、评分域、气候态分母、updates/LR/K/depth/dim
全部不动。改的只有**时间预算**，而 D6 明确允许「下一独立 protocol 可加时」。也**不**在运行中改
硬限、**不**重试以求偶然通过：`attempt02` 原样保留为失败 attempt，本轮以**新协议**重跑。

## Decision

1. **新预算（`attempt03` 起生效）**：`PER_SEED_SECONDS = 21,600 s`（6 h，为实测 14,400 s 的
   1.5 倍）、`PLANNED_SECONDS = 43,200 s`、`HARD_CAP_SECONDS = 86,400 s`。三者的比例结构
   （per-seed : planned : hard = 1 : 2 : 4）与 0050 一致。
2. **峰值估计不变**：仍写 `estimated_peak_bytes = 4,096 MiB` + margin 512 MiB + neighbour
   reserve 6 GiB。实测 2,322 MiB 低于该估计，属于**保守**（高估）方向，不是门失败，故不改；
   证据随本决策与证据页登记。
3. **工具缺陷更正**：`tools/r7_round_support.py::default_runner` 捕获 `subprocess.TimeoutExpired`
   并返回 `exit_code = 124`、`timed_out = True` 的结构化失败，而不是让异常逃逸。退出码仍非 0，
   所以所有调用方照旧停止该轮；改变的只是失败的形状（有 JSON、有记录，而不是整轮崩溃）。
   反证测试在 `tests/test_r7_round.py::test_default_runner_turns_a_timeout_into_a_structured_failure`。
4. **`attempt02` 的处理**：原样保留，**不删**、**不做 in-place resume**；它已花的 GPU 时间按
   实际计费登记。已完成的 control 臂 receipt 只作为**预算实测证据**引用，**不**作为本轮读数——
   本轮的科学比较要求同 seed 的 F 与 N 都在同一协议、同一 attempt 下跑完。
5. **其余条款全部沿用 0050 + 0052**：唯一因子（训练清单月份覆盖 4 → 12）、(a)–(d) 出口、桥 (e)
   非科学、65×65 限制、双卡 seed 分配、`test_read=false` / `scientific_claim=false`。

## Consequences

- **收益**：预算与实测对齐，六次训练不再被调度器提前杀掉；工具缺陷修好后，未来的超时会留下
  结构化失败记录而不是整轮崩溃。
- **代价**：本轮的 wall-clock 上限从 12 h 提到 24 h；`attempt02` 已消耗的 GPU 时间无法回收，
  如实登记。
- **风险与缓解**：若 treatment 臂因 7,302 条清单而比 control 慢得多，21,600 s 仍可能不足；
  届时按同一规则再开一个新协议并再次更正预算，而不是在运行中改硬限。

## References

- 决策 0050（原始预算条款）、0052（改绑当前修订）
- `docs/goals/s3-fy-annual-coverage.md`（本轮 brief §4 预算表）
- `outputs/r7_s3_fy_annual_coverage_20261021_attempt02/seed{41,42}_control_receipt.json`（实测）
- `tools/r7_round_support.py::default_runner`、`tests/test_r7_round.py`
- D6（每运行先冻预算；下一独立 protocol 可加时；不运行中改硬限）
