# S3：宽输入 + 内部盒监督（分离"监督稀释"与"信息增益"的单因素对照）

<!-- round-node: S3 -->

状态 **未开始**（2026-10-09 交接，本轮自迭代契约的下一轮）。唯一主计划
`docs/goals/main-model-climatology-campaign.md`；承接 `docs/goals/s3-wide-single-factor.md` §7 的
"下一动作"与主计划 §8 的最后一条"下一动作"。

## §0 Objective（单段）

> 在 docs/goals/s3-wide-interior-supervision.md 接续S3/index67/31.1244GPU-h、2023test未评分r0、S4未启动。上一轮把输入区域从 65×65 扩到 129×129（唯一改动因子）后发现：宽臂在 `interior_32`（= 冻结中心 65×65 盒）上五个 lead 的 seed 均值 delta 全部为正（更差），48/72h 的绝对气候态门仍未过且略差于窄臂，因此**不支持**"冻结 16° 盒缺长 lead 所需信息"。但该读数有一个已记录的混淆：上一轮是**全 129×129 全监督**，监督面积比窄臂大 3.94×，"域外信息有用"与"监督被稀释"无法区分。本轮做**分离这两者**的单因素对照：**宽 129×129 输入 + 只在 `interior_32` 上监督**（与已登记窄臂的 65×65 `full` 监督域逐点相同），配方（`long_rollout`、12 步物理权重、800 updates、LR 2e-5、warmup 10、FP32、K4、batch 1、clip 1、seed 41/42/43）、父状态（同一份迁移的 v3-BD 1600）、评分域（`interior_32`）、气候态分母与 cohort 守卫全部沿用上一轮；唯一改动因子回到**输入范围**。读法：内部监督臂优于已登记窄臂 → 域外上下文在**同监督域**下确有信息；持平或更差 → 上一轮的 null 不能归因于稀释，该假设在更强的对照下仍不被支持。本轮只报配对 delta 与 `interior_32` t2m skill，**不宣布科学通过**、不称泛化/SOTA、不进入 S4 冻结包。新 protocol/新输出目录/新 `code.zip`；先在 CPU 完成协议冻结、身份与定向反证，再按共驻余量申请 GPU。不paid/rental/exclusive/main/release/issueclose/force/mirror/破坏性/cron，不改科学合同/冻结证据/用户配置，不读 test，不自行宣布最终goal完成。

## §1 起点、证据与实质差异

- 起点工作分支 `r7/weather-reasoning`，HEAD 为上一轮登记提交（累计 31.1244 GPU-h、index67）。
- 起点证据：`docs/R7_S3_WIDE_SINGLE_FACTOR.md`（null 读数与 dilution 混淆的完整记录）。
- 与上一轮的实质差异：**监督域**。上一轮 `objective` 是 `deep_supervised_latitude_area_mse` 覆盖
  全 129×129；本轮把监督限制到 `interior_32`（中心 65×65），输入仍是 129×129。
- 与已登记窄臂的实质差异：**只有输入范围**（65×65 → 129×129），监督域、配方、父、评分域全同。

## §2 交付物清单

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 本轮目标与冻结设计决定 | 本文件 §3 |
| D2 | 新驱动/新协议/新 code.zip | `preparation_protocol.json`、`prepared_protocol.json`、`code.zip` + `archive_receipt.json` |
| D3 | 三 seed 真实训练与评分 | `seed{41,42,43}_receipt.json`、每 seed 五 lead 的 `boundary_rmse.csv` |
| D4 | 配对读数 | `readings.json`（内部监督臂 `interior_32` vs 已登记窄臂 `full` 的逐 seed 逐 lead delta；skill） |
| D5 | 登记与精确 CI | `docs/R7_S3_WIDE_INTERIOR_SUPERVISION.md`、index record、brief、账本行、§8、campaign-state |

## §3 判据与设计决定（本轮冻结）

科学门与验收形式只来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` 与主计划 §8；本轮
**不触碰**任何阈值，也**不宣布**通过。

- **臂**：宽 store 输入（129×129），损失只在 `boundary_masks(129,129,(32,))['interior_32']` 上累加；
  其余（配方、父、seed、评分域、气候态分母、cohort pin）逐字沿用上一轮。
- **实现约束**：不得改动上一轮的驱动与证据（那是已登记身份）；本轮新增 `--supervision interior`
  变体或新驱动文件，写进**新输出目录**，`code.zip` 与 `protocol_sha256` 都是新的。
- **对照**：**已登记**的窄 65×65 rollout-dose 800 臂（index `record:s3-rollout-dose`），
  按 `rmse_csv_sha256` 复核后复用，**不重训练**。
- **可证伪读法**：内部监督臂优于窄臂 → 支持"域外上下文有用"；持平或更差 → 上一轮 null
  不是稀释造成的，该假设在更强对照下仍不被支持。
- **cohort 守卫**：仍按 `{6:472,12:468,24:460,48:444,72:428}` 与窄臂回执逐 lead 核对。
- **test 策略**：`test.jsonl` 从不读取；`test_read=false`、`scientific_claim=false`。

## §4 数字预算与停止

- planned 9000 s（软预算）、hard 18000 s（宽松硬上限）、每 seed 4200 s（与上一轮同）；
  软超继续并记 overrun，hard 截止即停 attempt 并全额留 `failure.json`，不运行中改硬限。
- 每阶段 spawn 前做**只读** `gpu_gate`（`GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b`）；
  **禁止**对任何非本实验进程发送信号或做冻结/终止自动化。
- 失败 attempt 原样保留、全额计费、不做 in-place resume。

## §5 明确不做

不重训练已登记的窄臂、不重跑上一轮或更早的已登记轮次、不读 test、不改
`data/raw|interim|processed`、不改归档快照、不合成替代数据、不删部分输出复活、不改科学合同与
冻结证据与判据、不 main 合并/发版/关 issue、不 paid/rental/exclusive/force/mirror/破坏性/cron。

## §6 进度与交接

- 本轮尚未开始。**第一条命令**（CPU 准备）：
  `.venv/bin/python -m pytest -q tests/test_r7_s3_wide_single_factor.py`，
  然后在上一轮驱动上派生 `--supervision interior` 变体（新输出目录、新协议 digest、新 `code.zip`）。
- 若本轮完成后仍未出现可辨收益，按防空转纪律**换假设或停**，不再加 seed/剂量、不重复同一 test。
