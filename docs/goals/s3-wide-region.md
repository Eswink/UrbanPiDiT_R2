# S3：扩大区域（域外大尺度上下文）采集与 pilot（单因素，数据制度）

<!-- round-node: S3 -->

状态 **进行中**（2026-10-09）：pilot 已完成并登记，全量采集尚未开始。唯一主计划
`docs/goals/main-model-climatology-campaign.md`；承接 `docs/goals/s3-rollout-dose.md`
"下一动作（S3）"与 `docs/R7_S3_ROLLOUT_DOSE.md` §10 的"扩大区域/引入域外大尺度上下文"。
不重跑已停止的 anchor/ordering/剂量实例，不进入 S4，不自行宣布最终 goal 完成。

## §0 Objective（单段）

> 在 docs/goals/s3-wide-region.md 接续S3/index64/27.5905GPU-h、2023test未评分r0。唯一问题：48/72h 的墙是 pattern 相关本身——ACC 0.367/0.189，线性重标定上限 ACC² 仅 +0.1355/+0.0366，即便完全校准也过不了绝对气候态门；而 10 m/s 的系统 72h 走 ~2600 km（23°），远大于冻结 16° 目标框，因此"域内可用信息不足"是可反驳假设。本轮只改一个因素：把读取区域从 27–43N/107–123E 65×65 扩为 19–51N/99–131E 129×129（各方向 +8°，面积 3.94×），其中心 65×65 块按断言**就是**冻结目标框，因此冻结气候态与 val cohort 不变、评分可用 `boundary_masks(129,129,(32,))` 的 interior 掩码精确取得，与旧 65×65 读数逐格可比。源/快照/17 通道/节奏/split/模型 spec/归档初始化全部不变，只换数据制度，是真正的单因素。先写独立身份的冻结读计划（新 format，不覆盖 read_plan_frozen 的归档 digest）、新下载器与新输出路径，再自审只读 preflight 后 --write；pilot 只用 2018 冬 120 stamp 量真实网络/解码/墙钟成本，并与注册的目标框同季 part 逐项对照，另用只读工具核中心块与 v3 source.nc 逐位相同。不做训练、不评分、不读 test、不宣称收益；全量批次要等 pilot 实测成本出来后另冻协议。不paid/rental/exclusive/main/release/issueclose/force/mirror/破坏性/cron，不改科学合同/冻结证据/用户配置，不自行宣布最终goal完成。

## §1 起点、证据与实质差异

- 起点工作分支 `r7/weather-reasoning`，HEAD `43e58e7`（index64 已登记、累计 27.5905）。
- 冻结墙：`docs/R7_S3_ROLLOUT_DOSE.md` §4/§10——48/72h 的 t2m/full skill −0.1191/−0.4309，
  ACC 0.367/0.189，ACC² 上限 +0.1355/+0.0366，且剂量 200→800 只把 ACC 抬到 0.367/0.189。
- 与已停止实例的实质差异：**只改读取区域**（65×65 → 129×129，中心块 = 原框）。
  模型 spec、归档 actual-C 初始化、17 通道、0.25°、6h 节奏、2017–2021/2022 split、
  `l6`/12 步物理权重、LR、K、FP32、batch、clip 全部不变。
- 不重跑 anchor/ordering/剂量；不复用任何已否定假设。

## §2 交付物清单

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 独立身份的冻结宽区读计划 | `data/download/read_plan_wide.py`（format `r7-era5-wide-read-plan-v1`、digest `d5618cbf…`）、`read_plan_frozen` 归档 digest 不变 |
| D2 | 宽区下载器（复用受审计读机制） | `data/download/earthmover_wide_io.py`、`data/download/earthmover_spatial_w1.py` |
| D3 | 先冻协议 + 自审只读 preflight | `outputs/r7_s3_wide_region_pilot_20261009_attempt01/acquisition_protocol.json`（`bfdcfa89…`）、`preflight_wide.json`、`preflight_review.json` |
| D4 | pilot 实跑与两项只读核验 | `parts/part_winter_2018_receipt.json`、`criteria_check.json`（18/18）、`target_block_check.json`（中心块逐位相同）、`pilot_result.json` |
| D5 | 登记与精确 CI | `docs/R7_S3_WIDE_REGION_PILOT.md`、index65、brief、ledger、工作分支精确 SHA CI |

## §3 判据与开发读法

科学门与验收形式只来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`（本合同第 4/5 节）
与 `docs/goals/main-model-climatology-campaign.md` §8；本轮**不触碰**其中任何阈值。pilot 的
工程判据在下载前写进 `acquisition_protocol.json` 的 `success_criteria`，由只读工具
`scripts/check_r7_wide_pilot_receipt.py` 逐条对照实测回执执行（缺字段算失败，不算跳过），
另由 `scripts/verify_r7_wide_target_block.py` 核中心块身份；两者的读法与结果记在
`docs/R7_S3_WIDE_REGION_PILOT.md`。两个工具都有"故意违规必须报错"的反证测试。
本轮不产生任何 skill/ACC/气候态读数：它是**数据制度可行性**，不是模型结果。

## §4 数字预算与停止

per-part soft 1800 s / watchdog TERM 1860 s、`--max-decoded-gib 16`、planned network 4.0e9 B、
planned disk peak 6.0e8 B、disk margin 2.0e10 B、cleanup reserve 120 s。
soft 超继续并记 overrun；watchdog 触发即该 part 为保留失败，绝不当通过。
全量批次预算待 pilot 实测后另立协议，不沿用本轮数字。

## §5 明确不做

不训练、不评分、不读 test、不做全量下载（本轮只 1 part）、不改 `read_plan_frozen`/`earthmover_spatial_d1`/
`earthmover_spatial_s1` 的字节、不写 `data/raw|interim|processed`、不删部分输出复活、不合成替代数据、
不改科学合同与冻结证据、不 main 合并/发版/关 issue。

## §6 进度与交接

- 已完成：宽区读计划与下载器（三个测试文件共 45 个用例）→ 冻结 pilot 协议 → 只读 preflight（129×129、单位/层
  全部通过）→ 发现并**在下载前**修正每 stamp 字段读数为 19（原估 17）→ 自审 9/9 → 1 part 实跑成功
  → 冻结判据 18/18 → 中心块与 v3 `source.nc` 逐位相同（9 变量/17 通道/坐标/单位，120 stamp）。
- 实测：网络 3,565,096,475 B（目标框同季 part 3,554,716,077 B，比 1.0029）、解码 charged
  9,482,837,880 B（**完全相同**）、chunk reads 2287（相同）、墙钟 1223.141 s（比 0.965）、
  存储 70,207,133 B（比 3.699，面积比 3.939）。GPU-h **0.0000**（纯 CPU/网络）。
- 下一动作（S3）：按实测成本冻结全量批次（train 2017–2021 + val 2022，20+4 part），
  并行分片但受网络/磁盘预算约束；随后用 `prepare_r7_local.py` 建 v4 store 与 sidecar，
  重建同数据气候态与 incumbent，再以 `interior_32` 掩码与旧 65×65 读数逐格比较。
  test 未读、r=0、S4 未启动不变。
