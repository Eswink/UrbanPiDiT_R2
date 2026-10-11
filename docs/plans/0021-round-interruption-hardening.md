# 0021 编排加固：打断后的孤儿 worker 护栏

**状态：已落地（2026-10-12）；不跑实验、不开 GPU、不下载。**
起点 `964fc8ce4c4e64b085b297a056962406b381c4e8`，分支 `r7/weather-reasoning`。
承接计划 0020 的「六层打断抗性」与决策 0044/0045；本计划不是第二个 master。

## 1. 触发

用户把「尽可能降低打断风险」列为首要目标。2026-10-12 一次真实打断给出了硬证据：**暂停
workflow 时，在跑的 `world.run` 子进程被 `SIGKILL`**——seed 42 的 `seed42_process.json` 记
`exit_code=-9`、`elapsed_seconds=1656.78`，该段 GPU 时间**全额计费**、产物不复活。

顺带暴露第二个缺陷：**恢复时会重放 stage**，而 `run_stage` 在 spawn 前不写任何东西、
`stage_plan` 只看 receipt。于是"孤儿仍在训练 + 恢复重放"会 spawn 出**第二个** worker，
两个进程争同一 attempt 目录与同一块 GPU。余量门拦不住：协议写 `estimated_peak_bytes=2**31`
（2 GiB）、margin 也是 `2**31`，`required=4 GiB`，而实测需求 **13.55 GiB**
（seed 41 回执 `owned_cuda_reserved_peak_bytes=14_552_137_728`）。

## 2. 交付物

| # | 交付物 | 位置 |
| --- | --- | --- |
| D1 | 只读 stage 存活探测（扫 `/proc/<pid>/cmdline`，**不发任何信号**） | `tools/r7_round_liveness.py` |
| D2 | `stage_plan` 报 `already_running`/`running_pids`；`run_stage` 在 receipt 缺失但 worker 存活时返回 `status="already-running"` 且**拒绝 spawn** | `tools/r7_round_stages.py` |
| D3 | 余量门取「协议值」与「同 attempt 内已完成 seed 的实测峰值」的较大者 | `tools/r7_round_stages.py` + `tools/r7_round_liveness.py:measured_peak_bytes` |
| D4 | 只读孤儿视图：新子命令 `tools/r7_round.py orphans` | `tools/r7_round.py` |
| D5 | 反证测试（合成 `/proc` 树 + AST 无信号扫描） | `tests/test_r7_round_liveness.py` |
| D6 | 打断恢复 SOP + 两条实测教训 | `docs/plans/0020-campaign-driver-workflow.md` §8 |
| D7 | 登记：本计划、计划索引、`docs/rules/CHANGELOG.md`、R-009 基线同步 | 本页、`docs/plans/README.md`、`docs/rules/CHANGELOG.md`、`tools/check_conventions.py`、`docs/rules/testing.md`、`docs/rules/MIGRATION.md` |

## 3. 设计取舍

- **探测用 `/proc` 扫描而不是 PID claim**：不依赖编排器是否活着（孤儿恰恰是编排器死后的形态），
  且完全只读。匹配的是 `run_stage` 自己拼的完整 argv 形状（driver + `--worker <phase>` +
  `--out <attempt>` + `--seed <n>`），所以另一个 seed、另一个 attempt、另一个 driver 都不会误伤。
- **存活探测拆成独立模块**：`run_stage` 已经用满 R-023 的 8 参数预算（上一轮专门降过），
  且 `r7_round_support.py` 会越过 R-021 的 400 行目标。拆出 `tools/r7_round_liveness.py` 让
  两个报告计数**都保持原值**（R-021=89、R-023=45），并把 `PROC_ROOT` 做成模块属性而非第 9 个参数。
- **只拒绝、不等待**：`already-running` 是一个可区分的终态（`passed=false`），不是轮询等待——
  常驻轮询守护被 `docs/rules/gpu-resources.md:11` 明文禁止。
- **不改已登记轮次的驱动**：`estimated_peak_bytes` 的标定是**下一轮派生驱动**时的动作（写进 SOP），
  不回头改历史驱动（R-006 与各轮 §5）。

## 4. 验证

- `pytest tests/test_r7_round_liveness.py tests/test_r7_round_stages.py tests/test_r7_round.py tests/test_r7_round_reuse.py` → **87 passed**。
- **全量 CPU 套件** `pytest -q` → **4439 passed / 3 skipped / 0 failed**（2191.34 s；当机负载 41.8，3 个 skip 是"干净 checkout 无真实数据 fixture、禁止合成回退"的设计行为）。
- `python tools/check_conventions.py --quiet` → **37 条阻断 0 失败**；尺寸计数 R-021=89 / R-023=45 未变。
- `pytest tests/test_check_conventions.py` → **120 passed**（含 R-009 基线与 `size-thresholds.md`
  标记的防漂移断言）。
- 反证：① 有匹配存活 worker → `already-running` 且 `calls == []`；② 不同 seed / 不同 attempt /
  只提到 driver 的 shell 都不匹配；③ 无 worker → 照常 spawn；④ AST 扫描断言四个模块内
  **无任何信号调用**，且该扫描器本身有"能失败"的反证；⑤ `measured_peak_bytes` 在 receipt
  缺失/损坏时为 0。

## 5. 明确不做

不向任何进程发送信号；不实现冻结/终止/抢占；不建常驻守护或定时任务；不把编排搬出会话
（决策 0044:15-16「禁会话外续跑」）；不做 GPU 预留/锁；不删改任何记录/证据/index；
不改已登记轮次的驱动与证据；不接 main、不 force push、不跑 GPU。

## 实际结果

- **已落盘**：`tools/r7_round_liveness.py`（130 行）、`tools/r7_round_stages.py` 增 `already_running`
  判定与实测峰值门、`tools/r7_round.py` 增 `orphans` 子命令、`tests/test_r7_round_liveness.py`（277 行）、
  `docs/plans/0020-...md` §8 SOP、本计划、计划索引与 CHANGELOG。
- **实测**：本地四模块 **87 passed**；**全量 CPU 套件 4439 passed / 3 skipped / 0 failed**（2191.34 s）；
  `check_conventions.py --quiet` **37 阻断 0 失败**；
  自测 **120 passed**；尺寸计数与改动前一致（R-019=0 R-019b=1028 R-020=64 R-021=89 R-022=65 R-023=45）。
- **R-009 基线同步**：按 `docs/rules/testing.md:12` 的明文要求，把 `TEST_FUNCTION_BASELINE`/
  `ASSERT_BASELINE` 由 1912/5116 更新为实测 **2103/5607**，并同步 `testing.md` 与 `MIGRATION.md`
  引用的数字（两处由 `tests/test_check_conventions.py` 直接对 checker 常量断言）。
- **未做**：未推送、未绑定 CI（本计划 §6 只授权提交到工作分支）；GPU 轮编排仍未端到端验证
  （本轮不开 GPU）；科学目标未通过，仍 S3、test 未评分、r=0、S4 未进。
