# S3：全年覆盖单因素轮（4 个月 → 12 个月，年份固定 2017–2021）——65×65 实例

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-fy-annual-coverage", "driver": "scripts/study_r7_s3_fy_annual_coverage.py", "output_stem": "r7_s3_fy_annual_coverage_20261021", "control_record": "s3-v3-budget-dose", "seeds": [41, 42, 43], "device_mode": "dual", "test_path": ["tests/test_r7_s3_fy_annual_coverage.py"], "commit_paths": ["scripts/r7_s3_fy_annual_parent.py", "scripts/study_r7_s3_fy_annual_coverage.py", "scripts/build_r7_s3_fy_control_manifest.py", "tests/test_r7_s3_fy_annual_coverage.py", "docs/goals/s3-fy-annual-coverage.md", "docs/decisions/0050-fy-annual-coverage-round.md"], "commit_message": "feat(r7): implement the FY annual-coverage two-arm data round", "register_paths": ["docs/R7_S3_FY_ANNUAL_COVERAGE_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md"], "register_message": "feat(r7): register the FY annual-coverage run as a new evidence record", "code_archive": "outputs/r7_s3_fy_annual_coverage_code.zip"} -->

状态 **CPU 数据准备与正式 brief 已定、驱动/测试待实现、GPU 未开跑**（2026-10-21）。
唯一主计划 `docs/goals/main-model-climatology-campaign.md`；承接 **H-D 关闭**
（`record:s3-anchor-isolation-result`）、**决策 0048**（容量轴永久关闭）与**决策 0050**
（本轮授权边界）。用户 2026-10-21 明确选择数据方向并要求「用我们最新下载的数据」。

## §0 Objective（单段）

> 在 docs/goals/s3-fy-annual-coverage.md 接续 S3/index80/120.775625 GPU-h、2023 test 未评分 r0、S4 未启动；容量轴（0048）、架构轴 H-A/H-B/H-C、训练信号/目标组织各单因素臂均已按各自预声明出口关闭或未越过气候态门，用户 2026-10-21 明确选择数据方向并授权用已下载 FY 源做「全年 vs 4 个月」单因素轮：在同一个全新 65×65 全年 store（2017–2021 train / 2022 val / 2023 四个月 test 封存，source SHA 449470415aa8b845a6d8c3870140bd47e43bb5e3dc7b6a87c302cd26976afbc8，9,244 帧，48 桶 train-only 气候态）上，只把训练清单的月份覆盖从 Jan/Apr/Jul/Oct（2,140 条 72h 完整窗）扩到 12 个月（7,302 条），年份固定 2017–2021；两臂同模型 spec、同迁移父（attempt04，model_code_sha256 3fe747b2…）、同 2400 updates、K=4、depth=4、dim=192、同物理权重、同 seed 41/42/43、同 2022 全年 val、同 48 桶分母，只差训练清单；本轮只在 65×65 上成立，不新增下载、不改模型代码、不做 129×129 上采样、不把结果外推为宽实例结论；预声明出口：(a) F 的 72h t2m skill 三 seed 全 >0、(b) F−N 72h ΔRMSE 三 seed 全 <0、(c) F 48h skill 三 seed 全 >0、(d) F−N 6/12/24h seed 均值 ΔRMSE ≤0，全真只提请独立验收，任一假则永久关闭「全年覆盖」支；桥 (e)（控制臂 vs 已登记 s3-v3-budget-dose）只用于身份/工程核对，不作科学判定；本轮只报读数、不宣布科学通过、不称泛化/SOTA、不自行宣布最终 goal 完成；不 paid/rental/exclusive/main/release/issueclose/force/mirror/破坏性/cron，不改科学合同/冻结证据/用户配置，不读 test。

（objective 实测字符数由 `tools/check_goal_brief.py` 在提交前核验。）

## §1 起点与实质差异

- 起点工作分支 `r7/weather-reasoning`；累计 **120.775625 GPU-h**、index **80**。
- 起点证据：`docs/R7_S3_ANCHOR_ISOLATION_RESULT.md`（H-D，negative）、
  `docs/R7_S3_GLOBAL_MIXING_RESULT.md`（H-C，negative）、
  `record:s3-wide-interior-dose`（宽实例 2400 更新外部对照，pinned）。
- **为什么是数据覆盖**：架构/容量/训练信号/优化方向的单因素臂已全部按各自预声明出口关闭或
  未越过气候态门；用户点名选择数据方向。这是唯一能在**不新增下载、不改模型代码**的前提下，
  直接检验「现有模型是否被 4 个月训练覆盖限制」的一轮。
- 与上一轮的**实质差异**：H-D 改的是推理循环的反馈表征；本轮**不改任何模型代码**，只改训练清单
  覆盖的月份数。因子互不重叠。
- **65×65 先决约束**：FY 源是 65×65，已登记的宽臂是 129×129；仓库没有 65→129 上采样管线，
  store 契约 `spatial_resampling: false`。本轮只在 65×65 上成立，**不能**冒充宽实例结论。
  宽实例全年需新采集（约 34–36 h、~1 TB 网络、~27 GiB store），不在授权内。

## §2 交付物清单

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 本轮目标与冻结设计决定 | 本文件 §3、决策 0050 |
| D2 | 全年 65×65 store + 两套清单 + 48 桶气候态 | `outputs/r7_s3_fy_annual_20261021_attempt01/`（`source.nc`、`store/cache.zarr`、`store/manifests_{control,treatment}/`、`BUILD_COMPLETE`）；`outputs/r7_s3_fy_annual_d2_20261021_attempt01/reference.json` |
| D3 | 新协议/新输出/新 code.zip | `preparation_protocol.json`、`prepared_protocol.json`、`code.zip` + `archive_receipt.json` |
| D4 | **两臂**三 seed 真实训练与评分 | `seed{41,42,43}_{control,treatment}_receipt.json`、每 seed 每臂五 lead 的 `boundary_rmse.csv` |
| D5 | 配对读数与预声明判定 | `readings.json`（F−N、skill、桥 (e)、出口 (a)–(d)） |
| D6 | 登记与精确 CI | `docs/R7_S3_FY_ANNUAL_COVERAGE_RESULT.md`、index record、brief、账本行、§8、campaign-state |

## §3 判据与设计决定（本轮冻结；科学门只引用冻结文档）

科学门形式只来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` 与
`docs/plans/0024-fy-annual-coverage-round.md`；本轮**不触碰**任何阈值，也**不宣布**通过。

- **唯一因子：训练覆盖的月份数，4 → 12；年份固定 2017–2021。**
- **控制臂 N（4 个月）**：全年 store 的 train 时间轴取 Jan/Apr/Jul/Oct、每月 1–30 日、
  UTC 00/06/12/18，6h 清单 2,360 条；再按 **72h 完整窗**过滤——`t+6..t+72` 的每个目标都必须
  落在同一子集（否则长 rollout 会从全年 store 读到 31 日/下个月的帧，控制臂就不再是纯 4 个月），
  得到 **2,140** 条（丢弃 220 条）；dataset identity `04693129…`。长 rollout 完整窗数由
  `preflight_long_rollout_windows` 在 freeze 时写入协议（该过滤后应为 2,140、0 排除）。
- **候选臂 F（12 个月）**：2017–2021 全部日期；**7,302** 条 6h 清单，dataset identity
  `c17494ab…`。长 rollout 完整窗数同样在 freeze 时冻结。
- **两臂共用**：同一个 store（`store_path=../cache.zarr`）、同全年归一化、同模型 spec
  （K=4、depth=4、dim=192）、同迁移父（attempt04，三 seed pin `f9f12e…/8d4e05…/5af646…`）、
  同 2400 updates、LR 2e-5、warmup 10、物理权重 `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)`、FP32、
  batch 1、clip 1、同 seed 41/42/43、同 2022 全年 val（**1,458** 条，identity `94a0e509…`）、
  同 48 桶 train-only 气候态分母。**只差训练清单。**
- **评分域**：65×65 整张网格，`boundary_margins=(1,)`，读 `boundary_rmse.csv` 的 `full`
  区域（4,225 点）；不用 `interior_32`（在 65×65 上只剩 1×1）。
- **气候态分母**：48 个 `(month, hour)` 桶，每桶 141–155 帧，`n_selected_steps=7,304`，
  `training_years=[2017..2021]`；t2m RMSE 6h **3.8878542160913172 K**、12h
  **3.8887477466606346 K**、24h **3.8906948018885616 K**、48h **3.8946928171358484 K**、
  72h **3.8970789101854089 K**。缺桶即报错，不允许 held-out fallback。
- **预声明出口（冻结后不放宽）**：
  - (a) F 的 72h t2m skill 在 seed 41/42/43 **全部 > 0**；
  - (b) F−N 的 72h t2m ΔRMSE **三 seed 全部 < 0**（负值 = F 更优）；
  - (c) F 的 48h t2m skill **三 seed 全部 > 0**；
  - (d) F−N 在 6/12/24h 的 **seed 均值** ΔRMSE **≤ 0**。
  全真 → 只提请独立复核/验收，不自动进 S4；任一 (a)–(d) 假 → **永久关闭「全年覆盖」这一支**：
  不扫月份组合、不调 2400 updates、不加 seed、不把 2000–2016 塞进同一支、不与已关轴组合。
- **桥 (e)，非科学判定**：控制 N 的 5-lead t2m RMSE seed 均值与已登记窄 v3 四个月臂
  （`record:s3-v3-budget-dose`，1600 updates，2022 四个月 val）的 seed 均值比较；差值不得超过
  该登记臂在对应 lead 的 seed 极差或均值的 5%（取较大者）。登记读数（t2m RMSE，seed 41/42/43）：
  6h 2.2577/2.2628/2.2229、12h 2.9037/2.9067/2.8385、24h 3.0762/3.2123/2.8785、
  48h 5.0497/4.9808/4.0660、72h 7.0448/6.4577/4.8957。**归一化、case 集与 updates 都不同，
  桥不能当作等价性证明**；失败只记 `identity/engineering unresolved`，不进入 (a)–(d)。
- **正/负控制**：两臂由同一 seed、同一父状态、同一 spec 构造，初始化逐位相同；清单 identity
  是唯一允许的协议差异（测试断言 `set(diff) == {"train_manifest","train_data_identity",
  "train_windows"}`）。无新增参数，故无参数量混杂。
- **预算**：两臂 × 三 seed = 六次 2400-update 训练 + 每臂五 lead 评分。per-seed **9,000 s**
  （软），round planned **21,600 s**，hard **43,200 s**；软超继续并记 overrun，hard 截止即停
  attempt 并全额留 `failure.json`。峰值估计 **4,096 MiB** + margin **512 MiB** +
  neighbour reserve **6 GiB** = 10.75 GiB 门槛；按 H-D 宽臂实测峰值 9,170 MiB 以
  token 格点比例（65×65→约 1,089 token vs 129×129→约 4,225）缩放，并保留余量；
  **这是估计，不是实测**，首次 seed 后按 `owned_reserved_peak_bytes` 提升。
- **双卡（决策 0047）**：`device_mode: "dual"`，一卡一 seed，冻结 `gpu_uuids` 与
  `neighbour_reserve_bytes = 6 GiB`；每阶段 spawn 前只读余量门，**不对邻居发信号**。
- **test 策略**：`test.jsonl` 从不读取；`test_read=false`、`scientific_claim=false`。

## §4 数字预算与停止

| 项 | 冻结值 |
| --- | --- |
| 训练次数 | 2 臂 × 3 seed = 6 |
| 每 seed 软预算 | 9,000 s |
| round planned（软） | 21,600 s |
| round hard（宽松硬上限） | 43,200 s |
| 峰值估计 / margin / reserve | 4,096 MiB / 512 MiB / 6 GiB |
| 评分窗口（全 65×65） | 6h 1,458；12h 1,457；24h 1,455；48h 1,451；72h 1,447 |

- 每阶段 spawn 前做**只读** `gpu_gate`；失败 attempt 原样保留、全额计费、不做 in-place resume。
- 两 seed 并发（决策 0047），每卡同时至多一个 worker；GPU 由统一父调度串行。
- 墙钟不是科学判据；GPU-h 按 spawn 到 reap 连续计费。

## §5 明确不做

不新增下载；不建 129×129 全年源；不碰 `data/raw|interim|processed`；**不改模型代码**
（因此无需父重迁移）；不复用 16 桶气候态/锚表；不重开容量轴（0048）、H-A/H-B/H-C、PE/scale/
typed/ordering/anchor/rollout-dose/EMA/dense/final-only/spatial-gradient/zero-base decoder/
gradient-surgery/tendency-input；不读 test；不 main/force/merge/发版/关 issue；不删 `outputs/`；
不合并另一窗口的 tendency 改动。

## §6 进度与交接

- **已完成（CPU，2026-10-21）**：
  1. 合并源：FY 2017–2022 + 2023 四个月块，坐标/变量/单位逐位一致；9,244 帧，
     `source.nc` 2,036,671,757 B，SHA256 `44947041…`。
  2. store：shape `[9244,17,65,65]`，windows `7302/1458/472`，`BUILD_COMPLETE`，
     `validate_store` 通过。
  3. 两套清单：control 6h 2,360 → **72h 完整窗 2,140**（丢弃 220 条跨月/31 日窗）、
     treatment train 7,302（长 rollout 可用 7,291，11 条年末排除）、val 1,458（逐字节相同）、
     test 472（封存）；各自 `BUILD_COMPLETE`。
  4. 48 桶 train-only 气候态：5 个 lead 满覆盖，`reference.json` 已写。
  5. 决策 0050 与本 brief。
- **待做（开工前）**：① `scripts/r7_s3_fy_annual_parent.py`（钉 attempt04 迁移父，不改比对逻辑）；
  ② `scripts/study_r7_s3_fy_annual_coverage.py`（两臂同协议，≤600 行）；③
  `tests/test_r7_s3_fy_annual_coverage.py`（协议/单因子/预声明出口/CI-safe，不依赖 `outputs/`）；
  ④ 独立复核、`round-guard`、`commit-guard`、提交推工作分支并等 CI 绿。
- **阻塞项（2026-10-21 更新）**：另一窗口已把 tendency-input 轮提交为 `15ba518`
  （`model_code_sha256` 从 `3fe747b2…` 变为 `60de3b2d…`），并把 campaign-state 的
  `current_round_goal` 指向 `docs/goals/s3-tendency-input.md`。因此：
  1. 本轮的迁移父 attempt04（`3fe747b2…`）在 HEAD 上会被 `load_checkpoint` **按设计拒绝**；
     本轮不能把数据因子叠加到 tendency 架构上（那会同时改变模型输入表示与数据覆盖，破坏单因子）。
  2. 本轮不是当前 campaign 轮次；GPU 由统一父调度串行，**不**与 tendency 轮并发开跑。
  3. 恢复路径：等 tendency 轮完成/登记后，在**干净的 `fd92268` checkout**（最后一个
     `3fe747b2…` 修订）上应用本轮数据驱动提交并跑 GPU；若要在 `60de3b2d…` 上跑，必须先
     重新迁移父并另立决策，且明确本轮不再是「注册架构上的纯数据单因素」。
- **本轮 GPU 未开跑**：0 GPU-h 新增；store/manifest/climatology 是 CPU 派生数据，不是训练读数。
- **父迁移前提**：本轮不改 `model/**/*.py`，所以 `model_code_sha256` 应保持 `3fe747b2…`；
  迁移父用 attempt04 的三 seed checkpoint（`f9f12e…/8d4e05…/5af646…`），
  由 `scripts/r7_s3_fy_annual_parent.py` 在加载时核验 receipt 的
  `current_model_code_sha256` 与 `model_code_digest()` 相等。**不得**修改比对逻辑或放宽任何门。
- **CI 安全**：测试不得依赖 `outputs/`；纯函数 `frozen_constants()` 与 `arm_protocol()` 在
  tmp 路径上构造，`sha256_file`/`dataset_identity` 用 stub。提交前在干净 worktree 实跑。
- **第一条命令**：`.venv/bin/python tools/r7_round.py round-guard` → 重建 code zip →
  `.venv/bin/python tools/r7_round.py freeze` → `stage prepare/archive/stage-seeds/reading`。

## §7 下一动作（S3）

- 实现父模块、两臂驱动与测试；独立复核；`round-guard`；提交推 `r7/weather-reasoning` 并等
  `R7 CPU CI` 绿。
- 阻塞解除后：`freeze` → `stage prepare/archive/stage-seeds/reading`；两臂 × 三 seed。
- 预声明出口见 §3：**(a)–(d) 全真 → 提请独立验收；任一假 → 永久关闭「全年覆盖」支**。
- 本轮**未产生任何训练读数**；不宣布通过、不关闭任何支。
