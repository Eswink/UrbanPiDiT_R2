# S3：宽区域单因素真实 train/val（输入区域为唯一改动因子）

<!-- round-node: S3 -->

状态 **进行中**（2026-10-09）：本轮在已建成的 v4 宽 129×129 store 上用**注册的 rollout-dose
配方原样**训练三 seed，并只在冻结中心 65×65 盒（`boundary_masks(129,129,(32,))['interior_32']`）
上评分，与**已登记**的窄 65×65 rollout-dose 800 臂配对比较。唯一主计划
`docs/goals/main-model-climatology-campaign.md`；承接 `docs/goals/s3-wide-region-full.md` §6 的
"下一动作"与主计划 §8 的最后一条"下一动作"。不重跑已登记的 rollout-dose 与宽区 pilot 轮，
不重训练已登记的窄臂，不进入 S4，不读 2023 test，不自行宣布最终 goal 完成。

## §0 Objective（单段）

> 在 docs/goals/s3-wide-single-factor.md 接续S3/index66/27.5905GPU-h、2023test未评分r0、S4未启动。唯一问题：48/72h 的墙是 pattern 相关本身（ACC 0.367/0.189），而 10 m/s 系统 72h 走 ~2600 km（23°）远大于冻结 16° 目标框，因此"域内可用信息不足"是可反驳假设。本轮做**单因素真实 train/val**：唯一改动因子是**输入区域**（65×65 → 129×129，中心块 = 冻结目标盒），配方逐字沿用注册的 rollout-dose（`long_rollout`、12 步物理权重 `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)`、800 updates、LR 2e-5、warmup 10、FP32、K4、batch 1、clip 1、checkpoint_every 20、seed 41/42/43），监督域为**全 129×129 全监督**（与注册配方逐字相同，只换数据），评分域为 `interior_32`（= 中心 65×65），气候态分母用宽 store train-only 气候态（其 `interior_32` t2m RMSE 与已登记 v3-D2 在 ~1e-10 内相同）。对照是**已登记**的窄 65×65 rollout-dose 800 臂（index `record:s3-rollout-dose`），不重训练。读法：宽臂在 `interior_32` 上优于窄臂 → 支持"域外上下文有用"；持平或更差 → **不构成**信息不足的反证（全监督使监督面积扩大 3.94×，稀释是可能原因），须另做"同监督域、只改输入范围"的对照臂区分。本轮只报配对 delta 与 `interior_32` t2m skill，**不宣布科学通过**、不称泛化/SOTA。不paid/rental/exclusive/main/release/issueclose/force/mirror/破坏性/cron，不改科学合同/冻结证据/用户配置，不读 test，不自行宣布最终goal完成。

## §1 起点、证据与实质差异

- 起点工作分支 `r7/weather-reasoning`，HEAD `9c45341`（index66 已登记、累计 27.5905 GPU-h）。
- 本轮准备提交 `968f48c`（驱动 `scripts/study_r7_s3_wide_single_factor.py` 599 行 + 测试 119 行，7 测试过）。
- 冻结墙证据：`docs/R7_S3_ROLLOUT_DOSE.md` §4/§10——窄臂 48/72h t2m/full skill
  −0.1191/−0.4309、ACC 0.367/0.189。
- 宽 store 证据：`docs/R7_S3_WIDE_REGION_FULL.md`——`[3360,17,129,129]`、split/window 计数与 v3
  完全相同、中心 65×65 与 v3 `source.nc` 逐位相同、`interior_32` t2m 气候态 RMSE 与 v3-D2 同。
- 与已登记窄臂的实质差异：**只有输入区域**（65×65 → 129×129）。父状态是同一份迁移的 v3-BD 1600
  （模型 spec 与区域无关），LR/warmup/权重/步数/K/FP32/batch/clip/seed/updates 全部逐字相同。
- 与已登记轮次的实质差异：rollout-dose 轮改的是**训练信号**（rollout 剂量），pilot/全量轮改的是
  **数据制度**（读取区域并建成 store）；本轮第一次把"区域"这个因子送进**真实 train/val**。

## §2 交付物清单

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 本轮目标与冻结设计决定 | 本文件 §3 |
| D2 | 冻结的三阶段协议与代码归档 | `preparation_protocol.json`、`prepared_protocol.json`（`protocol_sha256`）、`code.zip` + `archive_receipt.json`、`execution_files_sha256` |
| D3 | 三 seed 真实训练与评分 | `seed{41,42,43}_receipt.json`、`seed{41,42,43}_process.json`、每 seed 五个 lead 的 `boundary_rmse.csv` |
| D4 | 配对读数 | `readings.json`（宽 `interior_32` t2m RMSE vs 窄 full t2m RMSE 的逐 seed 逐 lead delta；宽 `interior_32` t2m skill） |
| D5 | 共驻门与进程记录 | 每阶段 `*_spawn_gate.json`（只读余量核对，绝不对邻居进程发信号） |
| D6 | 登记与精确 CI | `docs/R7_S3_WIDE_SINGLE_FACTOR.md`、index record、brief、账本行、§8、campaign-state、工作分支精确 SHA CI |

## §3 判据与设计决定（本轮冻结）

科学门与验收形式只来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` 与主计划 §8；本轮
**不触碰**其中任何阈值，也**不宣布**通过。设计决定写进 `prepared_protocol.json` 的
`design_decision`/`decision_rule`/`hypothesis`/`difference` 四个字段，缺字段算失败。

- **臂**：宽 store 上跑注册 rollout-dose 配方（`long_rollout`、12 步物理权重、800 updates、
  LR 2e-5、warmup 10、FP32、K4、batch 1、clip 1、seed 41/42/43），父状态为迁移的 v3-BD 1600。
- **监督域**：**全 129×129 全监督**（与注册配方逐字相同，只把数据换成宽区），不做 loss 裁剪。
  理由：rollout 自回归，只监督内部会让外圈无约束、回灌后可能伤及内部；代价是监督面积 3.94×，
  属**已记录混淆**（limitations）。
- **评分域**：`evaluate_local(..., boundary_margins=(32,))` 在 129 网格上取 `interior_32`
  （= 中心 65×65，4225 点），逐变量/逐 lead RMSE；气候态分母来自宽 store train split 拟合的
  train-only 气候态，`skill = 1 − (rmse_wide/rmse_clim)²`。
- **对照**：**已登记**的窄 65×65 rollout-dose 800 臂，逐 seed 逐 lead 的 `rmse.csv` 以其 index
  record 的 `rmse_csv_sha256` 重新核对后才使用（漂移即失败），**不在本协议内重训练**。
- **可证伪读法**：宽臂 `interior_32` 优于窄臂 → 支持"域外上下文有用"；持平或更差 →
  **不构成**"域内信息充足"的反证，须另做"同监督域、只改输入范围"的对照臂。
- **cohort 守卫**：`WIDE_VAL_COHORTS = {6:472,12:468,24:460,48:444,72:428}` 在 freeze 时与窄臂
  回执逐 lead 核对，不等即失败（保证配对比较在同一病例集上）。
- **test 策略**：`test.jsonl` 从不读取；`test_read=false`、`scientific_claim=false`。

## §4 数字预算与停止

- planned 9000 s（软预算）、hard 18000 s（宽松硬上限）、每 seed 4200 s；软超继续并记 overrun，
  hard 截止即 `RuntimeError` 停 attempt 并全额留 `failure.json`，不运行中改硬限、不重试至偶然通过。
- 每阶段 spawn 前做**只读** `gpu_gate`（`GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b`，
  estimated_peak 2 GiB、headroom 2 GiB），余量不足即拒绝启动；**禁止**对任何非本实验进程
  发送信号或做冻结/终止自动化。
- 失败 attempt 原样保留、全额计费、不做 in-place resume；下一 attempt 用新输出目录与新协议。

## §5 明确不做

不重训练已登记的窄臂、不重跑 rollout-dose/pilot/全量获取轮、不读 test、不改
`data/raw|interim|processed`、不改归档快照、不合成替代数据、不删部分输出复活、不改科学合同与
冻结证据与判据、不 main 合并/发版/关 issue、不 paid/rental/exclusive/force/mirror/破坏性/cron。

## §6 进度与交接

- 已完成（CPU 准备）：驱动 `scripts/study_r7_s3_wide_single_factor.py`（599 行）+ 测试
  `tests/test_r7_s3_wide_single_factor.py`（7 测试全过、37 条阻断规则 0 失败），提交 `968f48c`。
- 宽 val cohort 实测 = 注册 v3 cohort（472/468/460/444/428），已在驱动内 pin。
- **下一动作**：见 §7（本轮交接）。

## §7 本轮交接

待本轮 attempt 完成后填写：实际 GPU-h / 网络 / 磁盘 / 墙钟 / overrun 口径、配对读数与可证伪读法、
以及**下一轮可执行的第一条命令**。
