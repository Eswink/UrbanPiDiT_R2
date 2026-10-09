# S3：宽区域全量获取、v4 store 与宽区气候态（单因素数据制度轮）

<!-- round-node: S3 -->

状态 **进行中**（2026-10-09）：pilot 已登记（index65），全量获取正在执行。唯一主计划
`docs/goals/main-model-climatology-campaign.md`；承接 `docs/goals/s3-wide-region.md` §6 的
"下一动作"与主计划 §8 的最后一条"下一动作"。不重跑已登记的 rollout-dose 与宽区 pilot 轮，
不进入 S4，不自行宣布最终 goal 完成。

## §0 Objective（单段）

> 在 docs/goals/s3-wide-region-full.md 接续S3/index65/27.5905GPU-h、2023test未评分r0。唯一问题：48/72h 的墙是 pattern 相关本身（ACC 0.367/0.189，线性重标定上限 ACC² 仅 +0.1355/+0.0366），而 10 m/s 的系统 72h 走 ~2600 km（23°），远大于冻结 16° 目标框，因此"域内可用信息不足"是可反驳假设。本轮只改数据制度：把读取区域从 65×65（27–43N/107–123E）扩为 129×129（19–51N/99–131E），其中心 65×65 块按断言**就是**冻结目标框，因此冻结气候态与 val cohort 不变、评分可用 `boundary_masks(129,129,(32,))` 的 `interior_32` 掩码精确取得。源/快照/17 通道/节奏/split/模型 spec/归档初始化全部不变。本轮交付：28 个 (year, season) part 的真实获取与合并（train 2017–2021 / val 2022 / test 2023，3360 stamps，与 v3 完全同时间轴）、`prepare_r7_local.py` 只读 preflight 后 `--write` 的全新排他 v4 wide store（含 process 诊断与 BUILD_COMPLETE）、以及 train-only 宽区气候态。不做训练、不评分、不读 test、不宣称收益；单因素 train/val 是**下一轮**的事。不paid/rental/exclusive/main/release/issueclose/force/mirror/破坏性/cron，不改科学合同/冻结证据/用户配置，不自行宣布最终goal完成。

## §1 起点、证据与实质差异

- 起点工作分支 `r7/weather-reasoning`，HEAD `9c45341`（index65 已登记、累计 27.5905）。
- 冻结墙：`docs/R7_S3_ROLLOUT_DOSE.md` §4/§10——48/72h 的 t2m/full skill −0.1191/−0.4309、
  ACC 0.367/0.189、ACC² 上限 +0.1355/+0.0366。
- 与 pilot 的实质差异：pilot 只取了 1 个 part（2018 冬）做成本/schema 测量；本轮取
  **全部 28 个 part**（2017–2023 × 四季）并合并成可建 store 的 source。
- 与已停止实例的实质差异：**只改读取区域**（65×65 → 129×129，中心块 = 原框）。模型 spec、
  归档 actual-C 初始化、17 通道、0.25°、6h 节奏、split、`l6`/12 步物理权重、LR、K、FP32、
  batch、clip 全部不变。

## §2 交付物清单

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 冻结的全量获取协议与只读 preflight | `outputs/r7_s3_wide_region_full_20261009_attempt02/acquisition_protocol.json`（`be50afbc…`）、`preflight_train.json`/`preflight_valtest.json`、`preflight_review.json` |
| D2 | 28 part 真实获取与合并 | `parts/part_<season>_<year>_receipt.json`（27 新 + 复用 pilot winter_2018）、`source.nc`、`merge_receipt.json` |
| D3 | 中心块身份核验 | `verify_r7_wide_target_block.py` 报告：宽源中心 65×65 与 v3 `source.nc` 逐位相同（3360 stamps、17 通道） |
| D4 | v4 wide store + sidecar | `outputs/r7_s3_wide_instance_v4_20261009_attempt01/store/`（`cache.zarr`、`manifests/`、`BUILD_COMPLETE.json`）、`write_report.json`、store 构建协议 |
| D5 | train-only 宽区气候态 | 宽区 D2 基线产物（气候态 2400 步、逐 lead 全覆盖） |
| D6 | 登记与精确 CI | `docs/R7_S3_WIDE_REGION_FULL.md`、index record、brief、ledger、工作分支精确 SHA CI |

## §3 判据与设计决定

科学门与验收形式只来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` 与主计划 §8；本轮
**不触碰**其中任何阈值。本轮的工程判据在下载前写进 `acquisition_protocol.json` 的
`success_criteria`（缺字段算失败，不算跳过）。

**下一轮单因素 train/val 的设计决定（本轮冻结，供下一轮执行）**：

- **臂**：在宽 store 上用**注册的 rollout-dose 配方**（`long_rollout`、12 步物理权重
  `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)`、800 updates、LR 2e-5、warmup 10、FP32、K4、batch 1、
  clip 1、seed 41/42/43）训练；对照是**已登记**的窄 65×65 rollout-dose 800 臂。
- **监督域**：**全 129×129 全监督**（与注册配方逐字相同，只把数据换成宽区），不做 loss 裁剪。
  理由：rollout 是自回归的，若只监督内部 65×65，外圈输出无约束、回灌后可能伤及内部预测；
  全监督保持 rollout 自洽。代价是监督面积扩大 3.94×，属**已记录的混淆**（见 limitations）。
- **评分域**：`evaluate_local(..., boundary_margins=(32,))` 在 129 网格上给出
  `interior_32`（= 中心 65×65）的逐变量/逐 lead RMSE；气候态由宽 store 的 train split 拟合
  （同一 `interior_32` 掩码裁剪），据此算 `skill = 1 − MSE_model/MSE_clim`。
- **可证伪读法**：宽臂在 `interior_32` 上优于窄臂 → 支持"域外上下文有用"；持平或更差 →
  **不构成**信息不足的反证（可能是监督稀释），须另做"同监督域、只改输入范围"的对照臂区分。
- 该决定与备选（只监督内部）都写进下一轮 protocol 的 `design_decision` 字段；本轮不实现训练代码。

## §4 数字预算与停止

- per-part soft 1800 s / watchdog TERM 1860 s、`--max-decoded-gib 16`、planned network
  100,000,000,000 B、network 硬上限 150 GiB、planned 14400 s、hard 64800 s、disk peak 12 GiB、
  free-space gate 50 GiB、cleanup reserve 180 s、**parallel pool 4**。
- **并发实测**：6 个并发 reader 死锁（CPU 冻结、socket 空闲）；4 个并发推进（聚合 ~8.1 MB/s，
  每 part ~2.0 MB/s，单 part ~1756 s，在 1800 s 内）；2 个 ~4.5 MB/s；1 个 ~2.9 MB/s。
  因此 pool 固定 4。
- soft 超继续并记 overrun；watchdog 触发即该 part 为保留失败，绝不当通过；失败 part 只以**新
  artifact 名**重试，不删除部分文件。
- store 构建预算与气候态预算在各自 protocol 里另冻。

## §5 明确不做

不训练、不评分、不读 test、不改 `read_plan_frozen`/`earthmover_spatial_d1`/`earthmover_spatial_s1`
的字节、不写 `data/raw|interim|processed`、不删部分输出复活、不合成替代数据、不改科学合同与
冻结证据、不 main 合并/发版/关 issue。

## §6 进度与交接

- 已完成：全量协议 + 两次只读 preflight + 自审 13/13 → 28 part 获取（27 新 + 1 复用 pilot；
  两次真实超期失败 `winter_2020`/`winter_2022` 原样保留、以 `_r2` 新名重取成功）→ 合并
  `source.nc`（3360 stamps、SHA `6bc9a1a2…`、时间轴与 v3 逐位相同）→ 中心块与 v3 `source.nc`
  逐位相同（`verify_r7_wide_target_block` pass）→ v4 wide store（`BUILD_COMPLETE`、
  `[3360,17,129,129]`、split/window 计数与 v3 完全相同、train-only 归一化 + process 诊断）→
  宽区 train-only 气候态与 persistence（10 次 `evaluate_local`、2461.6 s、0 overrun），且
  `interior_32` 的 t2m 气候态 RMSE 与**已登记 v3-D2 在 ~1e-10 内相同**。
- **下载器缺陷（本轮发现并修复，全部在 attempt02 之前）**：R-021 拆分把 helper 在模块间搬动时
  留下四处缺陷，committed 路径的 `--preflight` 与 `--write` **都不可运行**，pilot 是在拆分前跑的：
  1. `earthmover_wide_io.validate_wide_namespace` 引用未定义的 `surface_chunk_bytes`；
  2. `earthmover_spatial_w1` 未 import 五个已搬到 `earthmover_spatial_d1` 的 helper；
  3. `_wide_part_receipt` 引用其调用方的局部变量 `started`（**下载完成后写回执时崩溃**）；
  4. `_wide_part_receipt` 对已是字符串的版本号调用 `.__version__`。
  已修复并加两道回归守卫（作用域走查 + 回执合成），两者都带"故意违规必须报错"的反证。
  attempt01（`outputs/r7_s3_wide_region_full_20261009_attempt01/`）作为失败记录保留。
- **并发实测**：6 个并发 reader 死锁；4 个正常（每 part ~1600–1789 s）；2 个 ~1414–1522 s；
  1 个 1218.5 s。冻结的 1800 s 每 part 期限在 POOL=4 下已接近，两次失败由此产生。
- **下一动作（S3）**：按 §3 的设计决定做**单因素 train/val**——在宽 store 上用注册的
  rollout-dose 配方（全 129×129 全监督）训练三 seed，用 `boundary_margins=(32,)` 读 `interior_32`
  与**已登记的窄 rollout-dose 800 臂**比较；先在 CPU 上完成新 protocol/输出/停止出口与定向反证，
  再申请 GPU。test 未读、r=0、S4 未启动不变。
