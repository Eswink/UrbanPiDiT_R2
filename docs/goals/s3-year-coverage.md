# S3：训练年份覆盖单因素轮（2017–2021 → 2000–2021，月份固定 12 个月）——65×65 实例

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-fy-year-coverage", "driver": "scripts/study_r7_s3_fy_year_coverage.py", "output_stem": "r7_s3_fy_year_coverage_20261023", "control_record": "s3-fy-annual-coverage-result", "seeds": [41, 42, 43], "device_mode": "dual", "test_path": ["tests/test_r7_s3_fy_year_coverage.py"], "commit_paths": ["scripts/study_r7_s3_fy_year_coverage.py", "scripts/build_r7_s3_fy_year_manifests.py", "scripts/build_r7_s3_fy_multiyear_source.py", "scripts/build_r7_s3_fy_year_d2_reference.py", "tests/test_r7_s3_fy_year_coverage.py", "docs/goals/s3-year-coverage.md", "docs/decisions/0054-fy-year-coverage-round.md", "docs/decisions/README.md", "docs/goals/main-model-climatology-campaign.md", "docs/rules/size-thresholds.md"], "commit_message": "feat(r7): implement the FY training-year coverage two-arm data round", "register_paths": ["docs/R7_S3_FY_YEAR_COVERAGE_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md"], "register_message": "feat(r7): register the FY year-coverage run as a new evidence record", "code_archive": "outputs/r7_s3_fy_year_coverage_code.zip"} -->

状态 **CPU 数据准备进行中、驱动/测试待实现、GPU 未开跑**（2026-10-23）。
唯一主计划 `docs/goals/main-model-climatology-campaign.md`；承接 **FY 全年覆盖轮**
（`record:s3-fy-annual-coverage-result`，candidate、独立复核已完成、**等用户验收是否进 S4**）
与**决策 0050 后果 4**（「2000–2016 未使用；更多年份是后续独立因子」）。

## §0 Objective（单段）

> 在 docs/goals/s3-year-coverage.md 接续 S3/index89/202.359827 GPU-h、2023 test 未评分 r0、S4 未启动；FY 全年覆盖轮已按决策 0050 的出口 (a)–(d) 全真登记为 candidate（record:s3-fy-annual-coverage-result）并完成独立复核，其自声明 limitation 明确「预算固定 2400 updates 时候选臂只看到约三分之一 epoch，因此正结果定价的是该预算下的数据多样性，而非任意预算下的数据量」，决策 0050 后果 4 又预声明「2000–2016 未使用，更多年份是后续独立因子，不能与本轮合并解释」；本轮据此在同一个全新 65×65 多年 store（2000–2022 连续 6 h 源 + 2023 四个月封存块；train 2000–2021 / val 2022 / test 2023 封存）上只改一个因子：训练清单覆盖的**年份**从 2017–2021（5 年）扩到 2000–2021（22 年），**月份固定 12 个月**；两臂同 store、同 2000–2021 train-only 归一化、同 48 桶 train-only 气候态分母、同模型 spec（K=4/depth=4/dim=192）、同当前修订迁移父、同 2400 updates、同 seed 41/42/43、同 2022 全年 val（1,458 条），只差训练清单的年份范围；预声明出口：(a) F 的 72h t2m skill 三 seed 全 >0、(b) F−N 72h ΔRMSE 三 seed 全 <0、(c) F 48h skill 三 seed 全 >0、(d) F−N 6/12/24h seed 均值 ΔRMSE ≤0，全真只提请独立验收，任一假则永久关闭「训练年份覆盖」支；桥 (e)（控制臂 vs 已登记 FY 候选臂）只用于身份/工程核对，不作科学判定；本轮只在 65×65 上成立、不新增下载、不改模型代码、不读 test、不宣布科学通过、不称泛化/SOTA、不自行宣布最终 goal 完成；不 paid/rental/exclusive/main/release/issueclose/force/mirror/破坏性/cron，不改科学合同/冻结证据/用户配置。

（objective 实测字符数由 `tools/check_goal_brief.py` 在提交前核验。）

## §1 起点与实质差异

- 起点工作分支 `r7/weather-reasoning`；累计 **202.359827 GPU-h**、index **89**。
- 起点证据：`docs/R7_S3_FY_ANNUAL_COVERAGE_RESULT.md`（candidate，出口 (a)–(d) 全真，
  独立复核已完成，**按决策 0050 出口等用户验收是否进 S4**）。
- **为什么是年份覆盖**：决策 0050 后果 4 把「更多年份」预声明为后续**独立**因子，并禁止把它
  塞进全年覆盖轮解释；FY 轮自己的 limitation 也把「正结果定价的是 2400-update 预算下的数据
  多样性」写成未决问题。本轮正面检验这个未决问题。
- **一手检索（2026-10-23，`web-researcher` 通道；arXiv MCP 当次不可用）**：在数据驱动天气
  预测里**没有**任何「训练记录长度 / 年数」的受控消融。GraphCast（arXiv:2212.12794）训练用
  1979–2015，论文报告加入 1959–1978 这 20 年「对性能几乎没有影响」故排除，**未给数字**；
  Pangu-Weather（2211.02556）1979–2017、FuXi（2306.12873）1979–2015、FourCastNet
  （2202.11214）1979–2015 都**未做**年数消融；Aurora（2405.13063）只报数据**多样性**（跨数据集）
  的收益，未隔离单一再分析内的年数效应。天气社区也**没有**「固定 updates vs 固定 epochs」
  双口径的数据量消融，也无人讨论本轮的混杂。**最接近的先例方向与我们的 FY 结果相反**
  （GraphCast 是「加更早的 20 年，收益很小」），因此本轮是一个真正有区分力的检验。
- 与上一轮的**实质差异**：FY 轮改的是训练覆盖的**月份数**（4 → 12，年份固定）；本轮改的是
  **年份数**（5 → 22，月份固定 12）。两个因子互不重叠，且本轮的两臂月份覆盖相同，
  因此 FY 轮里「训练月份与 val 月份不匹配」这一成分在本轮**不存在**，不构成混杂。
- **65×65 先决约束**：FY 源是 65×65；仓库没有 65→129 上采样管线，store 契约
  `spatial_resampling: false`。本轮只在 65×65 上成立，**不能**冒充宽实例结论。

## §2 交付物清单

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 本轮目标与冻结设计决定 | 本文件 §3、决策 0054 |
| D2 | 多年 65×65 store + 两套清单 + 48 桶气候态 | `outputs/r7_s3_fy_multiyear_20261023_attempt01/`（`source.nc`、`merge_receipt.json`、`store/cache.zarr`、`store/manifests_{control,treatment}/`、`BUILD_COMPLETE`）；`outputs/r7_s3_fy_year_coverage_d2_20261023_attempt01/reference.json` |
| D3 | 新协议/新输出/新 code.zip | `preparation_protocol.json`、`prepared_protocol.json`、`code.zip` + `archive_receipt.json` |
| D4 | **两臂**三 seed 真实训练与评分 | `seed{41,42,43}_{control,treatment}_receipt.json`、每 seed 每臂五 lead 的 `boundary_rmse.csv` |
| D5 | 配对读数与预声明判定 | `readings.json`（F−N、skill、桥 (e)、出口 (a)–(d)） |
| D6 | 登记与精确 CI | `docs/R7_S3_FY_YEAR_COVERAGE_RESULT.md`、index record、brief、账本行、§8、campaign-state |

## §3 判据与设计决定（本轮冻结；科学门只引用冻结文档）

科学门形式只来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`；本轮**不触碰**任何阈值，
也**不宣布**通过。

- **唯一因子：训练清单覆盖的年份数，2017–2021（5 年）→ 2000–2021（22 年）；月份固定 12 个月。**
- **控制臂 N（5 年）**：多年 store 的 train 时间轴取 **2017–2021 全部 12 个月**的窗口，
  即 FY 轮候选臂的同一时间覆盖；窗口数由 `preflight_long_rollout_windows` 在 freeze 时写入协议。
- **候选臂 F（22 年）**：**2000–2021 全部 12 个月**的窗口；窗口数同样在 freeze 时冻结。
- **两臂共用**：同一个 store（`store_path=../cache.zarr`）、同 2000–2021 train-only 归一化、
  同模型 spec（K=4、depth=4、dim=192）、同当前修订迁移父、同 2400 updates、LR 2e-5、warmup 10、
  物理权重 `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)`、FP32、batch 1、clip 1、同 seed 41/42/43、
  同 2022 全年 val（**逐字节相同**）、同 48 桶 train-only 气候态分母。**只差训练清单的年份范围。**
- **评分域**：65×65 整张网格，`boundary_margins=(1,)`，读 `boundary_rmse.csv` 的 `full`
  区域（4,225 点）；不用 `interior_32`（在 65×65 上只剩 1×1）。
- **气候态分母**：48 个 `(month, hour)` 桶，在 store 的 train 年份（2000–2021）上拟合；
  缺桶即报错，不允许 held-out fallback。
- **预声明出口（冻结后不放宽）**：
  - (a) F 的 72h t2m skill 在 seed 41/42/43 **全部 > 0**；
  - (b) F−N 的 72h t2m ΔRMSE **三 seed 全部 < 0**（负值 = F 更优）；
  - (c) F 的 48h t2m skill **三 seed 全部 > 0**；
  - (d) F−N 在 6/12/24h 的 **seed 均值** ΔRMSE **≤ 0**。
  全真 → 只提请独立复核/验收，不自动进 S4；任一 (a)–(d) 假 → **永久关闭「训练年份覆盖」这一支**：
  不扫年份子集（例如只加 2010–2016）、不调 2400 updates、不加 seed、不与已关轴组合。
- **桥 (e)，非科学判定**：控制臂 N 的 5-lead t2m RMSE seed 均值与已登记 FY 轮候选臂
  （`record:s3-fy-annual-coverage-result`，12 个月 / 2017–2021 / 2022 全年 val）的 seed 均值比较；
  差值不得超过该登记臂在对应 lead 的 seed 极差或均值的 5%（取较大者）。**归一化与气候态分母
  都不同**，桥只用于发现身份/工程错误，失败只记 `identity/engineering unresolved`，
  不进入 (a)–(d)。
- **正/负控制**：两臂由同一 seed、同一父状态、同一 spec 构造，初始化逐位相同；清单 identity
  是唯一允许的协议差异（测试断言 `set(diff) == {"train_manifest","train_data_identity",
  "train_windows"}`）。无新增参数，故无参数量混杂。
- **预算**：两臂 × 三 seed = 六次 2400-update 训练 + 每臂五 lead 评分。per-seed **21,600 s**（软）、
  round planned **43,200 s**、hard **86,400 s**（沿用 FY 轮实测：单臂训练+评分 7,169.6–7,201.4 s，
  一 seed 两臂约 14,400 s；决策 0053）。软超继续并记 overrun，hard 截止即停 attempt 并全额留
  失败记录。峰值估计 **4,096 MiB** + margin **512 MiB** + neighbour reserve **6 GiB** = 10.75 GiB
  门槛；**这是估计，不是实测**，首次 seed 后按 `owned_reserved_peak_bytes` 提升。
- **双卡（决策 0047）**：`device_mode: "dual"`，一卡一 seed，冻结 `gpu_uuids` 与
  `neighbour_reserve_bytes = 6 GiB`；每阶段 spawn 前只读余量门，**不对邻居发信号**。
- **test 策略**：`test.jsonl` 从不读取；`test_read=false`、`scientific_claim=false`。

## §4 数字预算与停止

| 项 | 冻结值 |
| --- | --- |
| 训练次数 | 2 臂 × 3 seed = 6 |
| 每 seed 软预算 | 21,600 s |
| round planned（软） | 43,200 s |
| round hard（宽松硬上限） | 86,400 s |
| 峰值估计 / margin / reserve | 4,096 MiB / 512 MiB / 6 GiB |
| 训练帧（store train 年份 2000–2021） | 32,144 |
| 评分窗口（全 65×65） | 6h 1,458；12h 1,457；24h 1,455；48h 1,451；72h 1,447 |

- 每阶段 spawn 前做**只读** `gpu_gate`；失败 attempt 原样保留、全额计费、不做 in-place resume。
- 两 seed 并发（决策 0047），每卡同时至多一个 worker；GPU 由统一父调度串行。
- 墙钟不是科学判据；GPU-h 按 spawn 到 reap 连续计费。

## §5 明确不做

不新增下载（多年源已在盘上，只做本地合并）；不建 129×129 多年源；不碰 `data/raw|interim|processed`；
**不改模型代码**（因此无需父重迁移）；不复用 FY 轮的 48 桶气候态/锚表（本轮在 2000–2021 上重算）；
不重开容量轴（0048）、H-A/H-B/H-C/H-D、PE/scale/typed/ordering/anchor/rollout-dose/EMA/dense/
final-only/spatial-gradient/zero-base decoder/gradient-surgery/tendency-input/SAM/全年覆盖（月份轴）；
不扫年份子集；不读 test；不 main/force/merge/发版/关 issue；不删 `outputs/`；不合并另一窗口的改动。

## §6 进度与交接

- **已完成（CPU，2026-10-23）**：
  1. 合并源：FY 源 `2000-01-01..2022-12-31`（`b5a8dba0…`，33,604 帧，连续 6 h）+
     2023 封存块 `2023-01-01..2023-10-30`（`44a24ca0…`，480 帧），坐标/变量/单位/dtype 逐位一致。
     合并后 **34,084 帧**、`2000-01-01 00:00 .. 2023-10-30 18:00`；
     receipt `merge_receipt.json`（`r7-fy-multiyear-merged-source-v1`，`scientific_claim=false`）。
     2023 内部有 59–62 天的月间空档（封存块只有四个月），如实记录。
- **待做（开工前）**：① store 只读 preflight → `--write`；② 两套清单；③ 48 桶气候态
  `reference.json`；④ 独立复核、`round-guard`、`commit-guard`、提交推工作分支并等 CI 绿；
  ⑤ `freeze` → `stage prepare/archive/stage-seeds/reading`；⑥ 登记四件套。
- **CI 安全**：测试不得依赖 `outputs/`；纯函数在 tmp 路径上构造，`sha256_file`/`dataset_identity` 用 stub。
- **第一条命令**：`.venv/bin/python tools/r7_round.py round-guard` → 重建 code zip →
  `.venv/bin/python tools/r7_round.py freeze` → `stage prepare/archive/stage-seeds/reading`。

## §7 下一动作（S3）

- **开工顺序**：先补齐 store/manifest/气候态（CPU，只读 preflight 后 `--write` 到全新排他路径），
  再写驱动与测试、独立复核、`round-guard`/`commit-guard`、提交并等 CI 绿，然后 `freeze` 与 GPU。
- **与 S4 的关系**：本轮**不推进到 S4**。FY 轮的 candidate 仍**等用户验收**；S4（封存 2023 test
  的首次评分）是用户的决定，本轮不做、也不以本轮结果代替该验收。
- 预声明出口见 §3：**(a)–(d) 全真 → 提请独立验收；任一假 → 永久关闭「训练年份覆盖」支**。
- `test` 未读、r=0；不宣布科学通过、不称泛化/SOTA、不自宣最终 goal 完成。
