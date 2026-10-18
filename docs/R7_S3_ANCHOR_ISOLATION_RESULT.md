# S3 H-D 气候态锚的异常反馈表征（`draft − C_valid` vs `draft`）单因素臂：F→N 五个 lead 三 seed 全为配对改善（72 h −0.321 K），但 72 h 绝对 skill 三 seed 未全正（2/3），按预声明出口关闭「气候态锚」族

**状态：已完成（2026-10-18，两臂 × 三 seed 全部跑完并评分）；`scientific_claim: false`；2023 test 未读；r 不消耗。**
本页对应 `docs/goals/main-model-climatology-campaign.md` §8 与 `docs/goals/s3-anchor-isolation.md`。
它是**开发筛选读数**，不是科学确认，也不是 S4 就绪声明；`primary`/`gate` 形式在这里只是**报告**，
因为外部对照（A）是已登记臂的 pinned 复用，而协议内对照（N）是本轮同协议内重训练的。

**同一因子的登记序号**：这是「K 步推理循环的**反馈表征**（绝对草稿 `draft` vs 异常草稿 `draft − C_valid`）」
这一因子的**第一次**登记。本因子此前**没有**任何已登记 record
（`record:s3-climatology-anomaly-anchor` 是**不同的、package 级的**记录：它同时改变输出先验、proposal 先验
与 feedback 表示，故不是单一机制；本页**不**向它追加 `correction_record_id`，也不改写它）。

## 1. 结论摘要

- **单因素**：两臂装**同一张 16 桶、129×129 的 train-only 气候态锚表**
  （`table_sha256=da2f28c9b3de0e175dafd6ee02b656cbdbeb9898d7fb94243796a45370ebd2d0`，
  `table_shape=[16,17,129,129]`），两臂都显式设 **`anchor_output_prior=False`**、**`new_params=0`**；
  唯一差异是 `anomaly_feedback`：控制臂 N 为 `False`（反馈 = `draft`），候选臂 F 为 `True`
  （反馈 = `draft − C_valid`，实现见 `model/process_step_r7.py:108`）。其余（宽 129×129 输入、
  `interior_32` 监督、2400 更新、K=4、depth=4、`dim`=192、注册物理权重 `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)`、
  迁移 v3-BD 1600 父、seed 41/42/43、评分域与气候态分母）逐字不变。
- **两控制逐 seed 实测通过**（探针为合成 tiny spec/表、seed 1234，**不是天气真值**）：
  ① **控制臂 N 与无锚注册模型逐位相同**——`unanchored_max_abs=0.0`、`unanchored_equals_off=true`、
  `bit_identical_outside_anchor=true`、`n_shared_compared=127`、`added_keys` 恰为
  `backbone.climatology_anchor.{ready,table}` 两键、`lost_keys=[]`；
  ② **锚表清零 ⇒ 候选臂逐位塌回控制臂**——`zeroed_table_delta_max_abs=0.0`、
  `zeroed_table_is_identity=true`、`zeroed_control_equals_off=true`（`draft − 0 = draft`）。
  另实测 `feedback_is_read=true`（`feedback_delta_max_abs` **2.56e-4…4.81e-4**）与
  `output_prior_delta_max_abs` **2.531…2.791**（输出先验开关本身可被观测；本轮两臂都关）。
- **F→N 五个 lead 全部为配对改善**：F−N 的 `interior_32` t2m Δrmse seed 均值为
  **−0.041398 / −0.108964 / −0.088920 / −0.206355 / −0.321199 K**，**五个 lead 的 3/3 seed 全为负**
  （负值 = F 更优），15/15 个 seed×lead 单元同号。
- **72 h 绝对读数**：F 臂 `interior_32` t2m skill 为 **+0.010101 / −0.038377 / +0.030003**
  （seed 均值 **+0.000576**），N 臂 **−0.174635 / −0.277247 / −0.156169**（均值 **−0.202684**）。
  **预声明出口的第一条「72 h 三 seed 全正」未满足**（seed 42 为 −0.038377）；
  **第二条「F 在 6/12/24 h 不差于 N」满足**（6/12/24 h 的 F−N 在 3/3 seed 均为负）。
  ⇒ **预声明出口整体未满足 ⇒ 按预声明永久关闭「气候态锚」这一族**（不调表粒度、不减锚、不加 seed）。
- **必须照此读（不要误读为「反馈表征无效」）**：F−N **不是 null**——它在本轮是**大而一致**的配对改善
  （72 h 均值 −0.321 K），故计划里「若 F−N 无差异 ⇒ 结论是异常反馈表征无效」的**条件分支未触发**。
  本族的关闭原因是**绝对 72 h 门在三 seed 上未全过**，不是「反馈表征没有效果」。
- **相对已登记 2400 剂量臂 A**（`record:s3-wide-interior-dose`，pinned 复用、未重训练）：
  F 在五个 lead 上 3/3 seed 全优（Δrmse **−0.0414 / −0.1090 / −0.0889 / −0.2064 / −0.3212 K**）；
  N 与 A 的差在 **−4.8e-6…+1.8e-5 K**（训练非逐位可复现的噪声级），即控制臂在行为上复现了注册臂。
- **判定**：按**决策 0048**（`docs/decisions/0048-capacity-axis-scope-and-cap.md`）与预声明出口，
  H-D（H-C 关闭后转训练信号/目标组织方向的**第一根**轴）**未满足出口** ⇒ **关闭「气候态锚」这一族**
  （不调表粒度、不减锚、不加 seed），转主线重估或另一条可证伪假设。本臂**不进 S4 冻结包**。

**表 A：F→N 配对 delta（F − N，同协议同 seed 同父同锚表）**

| lead | Δrmse F−N（K，seed 均值） | 范围 | 三 seed 同号 | F skill 均值 | F skill(41) | F skill(42) | F skill(43) |
| ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: |
| 6 h | −0.041398 | −0.044518 … −0.035530 | **是** | +0.597781 | +0.595310 | +0.591666 | +0.606366 |
| 12 h | −0.108964 | −0.127229 … −0.098582 | **是** | +0.378767 | +0.382779 | +0.364626 | +0.388896 |
| 24 h | −0.088920 | −0.109723 … −0.076319 | **是** | +0.458220 | +0.463661 | +0.443160 | +0.467838 |
| 48 h | −0.206355 | −0.216710 … −0.200783 | **是** | +0.182327 | +0.182373 | +0.142661 | +0.221947 |
| 72 h | −0.321199 | −0.368822 … −0.294899 | **是** | **+0.000576** | +0.010101 | **−0.038377** | +0.030003 |

**表 B：两臂相对已登记 2400 剂量臂 A 的配对 delta（本臂 − 剂量臂；负值 = 本臂更优）**

| lead | Δrmse F−剂量（K，seed 均值） | 范围 | 更优 seed | Δrmse N−剂量（K，seed 均值） | 范围 | 更优 seed |
| ---: | ---: | ---: | :---: | ---: | ---: | :---: |
| 6 h | −0.041398 | −0.044518 … −0.035530 | **3/3** | −0.000000 | −0.000000 … +0.000000 | 1/3（噪声级） |
| 12 h | −0.108964 | −0.127229 … −0.098581 | **3/3** | +0.000000 | +0.000000 … +0.000001 | 0/3（噪声级） |
| 24 h | −0.088917 | −0.109718 … −0.076315 | **3/3** | +0.000003 | −0.000001 … +0.000005 | 1/3（噪声级） |
| 48 h | −0.206350 | −0.216704 … −0.200771 | **3/3** | +0.000005 | −0.000003 … +0.000012 | 1/3（噪声级） |
| 72 h | −0.321193 | −0.368827 … −0.294894 | **3/3** | +0.000006 | −0.000005 … +0.000018 | 1/3（噪声级） |

**表 C：两臂逐 lead `interior_32` t2m skill（1 − (rmse/rmse_climatology)²）**

| lead | F skill 均值 | F(41) | F(42) | F(43) | N skill 均值 | N(41) | N(42) | N(43) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 h | +0.597781 | +0.595310 | +0.591666 | +0.606366 | +0.582700 | +0.579045 | +0.575467 | +0.593588 |
| 12 h | +0.378767 | +0.382779 | +0.364626 | +0.388896 | +0.328557 | +0.324151 | +0.317598 | +0.343921 |
| 24 h | +0.458220 | +0.463661 | +0.443160 | +0.467838 | +0.419468 | +0.415889 | +0.407563 | +0.434953 |
| 48 h | +0.182327 | +0.182373 | +0.142661 | +0.221947 | +0.067267 | +0.061329 | +0.027735 | +0.112736 |
| 72 h | **+0.000576** | +0.010101 | **−0.038377** | +0.030003 | **−0.202684** | −0.174635 | −0.277247 | −0.156169 |

（Δ 为「本臂 − 对照」，负值 = 本臂更优；F 与 N 的评分域、气候态分母、cohort pin 逐字相同，A 为同 seed 配对的
pinned 复用，故可直接相减。逐 lead 的绝对值与 Δ 见
`outputs/r7_s3_anchor_isolation_20261015_attempt01/readings.json`。）

## 2. 身份与协议

| 项 | 值 |
| --- | --- |
| attempt 目录 | `outputs/r7_s3_anchor_isolation_20261015_attempt01/` |
| 协议 digest（`protocol.json`，最终；嵌入字段；与 `readings.json` 逐字一致） | `c90159761f7f7efede98bde39f3ceeb9e618becec359fe02bd51cb9c22676ada` |
| 准备/预备 digest（`freeze` 阶段报告，`preparation_protocol.json` 的嵌入 digest） | `126b55a9450f877a5ac239a1a0455081c7f382703dfc2c001c007ed2808a70c8`（文件 SHA256 `e48e6d68e31089fd81100fef2e2cf643eec285288b5fb35ca58128df198695d6`） |
| 归档 `code.zip` SHA256 | `6639ce6111f3b423ecaebbf4670dc76c212d24b821dd3b484f979b591f7191b9`（1426 条目） |
| ZIP comment / 代码 commit | `0becf246d5d027d33bf9cd325004e9f5e734985d`（comment 与 `code_commit.txt`、`archive_receipt.json` 逐字一致） |
| `model_code_sha256` | `3fe747b28e7d0ef53febc07739f210f103514a73fd19bfba6643670db65ffeb1`（本轮改了 `model/*.py`，故与 H-C 的 `fff81dcc…` 不同） |
| `source_sha256` | `6bc9a1a2d93d345ce0dcb39d4f3a26f175ea3bcc15082ac4931a179ca13b7701` |
| `training_code_sha256` | `8ad77d9349bdc4930e6bbba05f0c417814fe3093fe39aa921e816af197494974` |
| 执行文件身份 | 192 个（`execution_files_sha256` 的键数） |
| 监督块（冻结） | `interior_32`；runner block `spatial_mask`、`[129,129]`、4225 点、`mask_sha256=4a2a576bf948b92924720fa28a44c128a61445dd29b4906805930430e0a0919a`（与对照臂相同） |
| **锚表（两臂完全相同）** | `table_sha256=da2f28c9b3de0e175dafd6ee02b656cbdbeb9898d7fb94243796a45370ebd2d0`；`spec_sha256=09dec42cada57807dcf6a29440b681a1c45428497dc35defeadf638f92b524db`；文件 `anchor_table.pt` SHA256 `29f6f9deccc6798f221288e07a763f7a614968e9cf638ed0d8512ad5cf42913f`；`table_shape=[16,17,129,129]`、16 个 `(month,hour)` 桶、每桶 150 帧、train 2017–2021、`n_selected_steps=2400` |
| 气候态口径 | **4 个月季节均值**：train/val 只覆盖 **Jan/Apr/Jul/Oct**（16 桶 = 4 月 × 4 个 init hour），**不是 12 个月年循环**（`outputs/r7_s3_anchor_table_20261015_attempt02/build_receipt.json` 的 `limitations` 与 `bucket_keys`） |
| **声明差异（冻结）** | 两臂**装同一张表**、**`anchor_output_prior=False`**、**`new_params=0`**（仅两个 persistent buffer `backbone.climatology_anchor.{ready,table}`，不参与参数计数）；唯一因子为 `anomaly_feedback`（`feedback=draft` vs `draft − C_valid`） |
| 父（迁移 v3-BD 1600，逐 seed pin） | 41 `f9f12e063c30cb0b471def01b67d7333d5aae95ecfba1ff0feb824a2a6f9ac45` / 42 `8d4e05ed5baa285113db353eede34e883812092085b8f86716fbee93fb13770f` / 43 `5af64655402cb16f0148b692ef7a84eb8934230cf43ed6ac9306501a7756cafa`（`outputs/r7_s3_rollout_dose_parent_migrated_20261015_attempt04/`） |
| 设备（决策 0047 双卡） | `device_mode=dual`；`gpu_uuids=[GPU-408ad137…, GPU-9d1624af…]`；`neighbour_reserve_bytes=6 GiB`；`estimated_peak_bytes=9.5 GiB`；margin 0.5 GiB |
| 预算 | planned 86400 s / hard 115200 s / per-seed 43200 s（**soft/hard overrun 均 0**） |
| cohort pin | 472 / 468 / 460 / 444 / 428（逐 lead，两臂三 seed 一致） |
| 外部参考 A | 2400 剂量臂（`record:s3-wide-interior-dose`，pinned，**未重训练**） |
| test | `test.jsonl` 从不读取；`test_read=false` |

## 3. 训练与评分回执

| seed | 臂 | checkpoint SHA256 | 训练 s | 五 lead 评分 s | 父状态 `parent_state_sha256` | 扩张 `expanded_state_sha256` | `anchor_init_sha256` | 峰值 reserved |
| ---: | --- | --- | ---: | ---: | --- | --- | --- | ---: |
| 41 | control (N) | `61e8f07af0ddf241601f33e5e1dfaf7376e81734dbb374ad127b73c30c0bcaeb` | 6155.9 | 1680.5 | `7c416ccb4054cf7e077788a43b02917e6a2dcf2503058152a7f0ab4b0ab70707` | `00ada5e25c4e7ef16acfb24768408f06e5e6ab091d4a91bdc7505c99229de612` | `de0e76beb548881b282fbd8c6f194055286ec7d0dd1d7ba88dad476fc2aa1112` | 8.379 GiB |
| 41 | candidate (F) | `3f6ac0e60f4701d33678fe7a9b3035a229892538122be14b34c4cbc38d01a9e6` | 6662.8 | 1478.3 | `7c416ccb4054cf7e077788a43b02917e6a2dcf2503058152a7f0ab4b0ab70707` | `00ada5e25c4e7ef16acfb24768408f06e5e6ab091d4a91bdc7505c99229de612` | `de0e76beb548881b282fbd8c6f194055286ec7d0dd1d7ba88dad476fc2aa1112` | 8.379 GiB |
| 42 | control (N) | `b3c071714c4df51be0bcd2f786151546bdd4c0cbe771f45cf3b0cdbaec0ca33d` | 6640.0 | 1829.9 | `857d880466f76500ed74c534e757732ae9eff4c2b3f0bbc0fa74b895fcae3623` | `baed2ff9ca898af7c6590f5ba269411d6bd660bd9601b5bcee7ae2c2a2b660a1` | `de0e76beb548881b282fbd8c6f194055286ec7d0dd1d7ba88dad476fc2aa1112` | 8.379 GiB |
| 42 | candidate (F) | `21ab420c18784705e57f754406c5939cfd8493f8f30e2feb948fbef79d899888` | 7364.9 | 989.3 | `857d880466f76500ed74c534e757732ae9eff4c2b3f0bbc0fa74b895fcae3623` | `baed2ff9ca898af7c6590f5ba269411d6bd660bd9601b5bcee7ae2c2a2b660a1` | `de0e76beb548881b282fbd8c6f194055286ec7d0dd1d7ba88dad476fc2aa1112` | 8.379 GiB |
| 43 | control (N) | `5d8240f34baae3f6f5c11cdb6f602292609b142cd5d65c028b7f17447aaf54ed` | 6549.9 | 1005.2 | `61fdac58b7bfacf828e1eae4968817b1cbd46616442c6d16ac0fbbb63e294e1b` | `4dda647f86ffee6befcab749e04153e7dbf09c6034e62d03a8f7d00bf066cce7` | `de0e76beb548881b282fbd8c6f194055286ec7d0dd1d7ba88dad476fc2aa1112` | 8.379 GiB |
| 43 | candidate (F) | `e85b15fe130a635c5baa0748c291726d06122b075db08df3112c7465a68eb270` | 5061.5 | 982.8 | `61fdac58b7bfacf828e1eae4968817b1cbd46616442c6d16ac0fbbb63e294e1b` | `4dda647f86ffee6befcab749e04153e7dbf09c6034e62d03a8f7d00bf066cce7` | `de0e76beb548881b282fbd8c6f194055286ec7d0dd1d7ba88dad476fc2aa1112` | 8.379 GiB |

- 六个训练都到 `total_updates=2400`；每 seed 每臂每 lead `n_evaluated == n_available_windows` 且等于冻结
  cohort（472/468/460/444/428）；每 seed 每臂的 `supervision` 与冻结 `mask_sha256` 逐字相同（驱动断言）。
- **同 seed 两臂的父状态与扩张状态逐位相同**：`parent_state_sha256` 与 `expanded_state_sha256` 在两臂
  **逐字相等**（41 `7c416ccb…`/`00ada5e2…`；42 `857d8804…`/`baed2ff9…`；43 `61fdac58…`/`4dda647f…`），
  `anchor_init_sha256` 三 seed 两臂全为 `de0e76be…`——这正是「两臂只差 `anomaly_feedback` 一个布尔」的意思。
  不同 seed 的父状态不同（符合三 seed 各自的迁移 pin）。
- **正控制逐 seed 通过**（`scripts/r7_s3_anchor_isolation_parent.py` 在控制失败时 `raise`）：
  `unanchored_equals_off=true`、`unanchored_max_abs=0.0`、`bit_identical_outside_anchor=true`、
  `n_shared_compared=127`、`zeroed_table_is_identity=true`、`zeroed_table_delta_max_abs=0.0`、
  `zeroed_control_equals_off=true`、`feedback_is_read=true`：
  `feedback_delta_max_abs` 2.5606e-4 / 3.0381e-4 / 3.1435e-4 / 3.2556e-4 / 3.3939e-4 / 4.8126e-4（六份回执），
  `output_prior_delta_max_abs` 2.5306…2.7910（说明输出先验开关本身可被观测；两臂均关）。
- **控制臂与已登记剂量臂的行为复现**：N−剂量 Δrmse 在五个 lead 上为
  −4.8e-6…+1.8e-5 K（见**表 B**），即重训练噪声级；这不是逐位相等（GPU 训练不可逐位复现），
  但与「N 初始化 ≡ 无锚注册模型」的逐位控制互相印证。
- seed 42 的 F 臂训练墙钟较长（7364.9 s），源于与邻居共驻——**墙钟不是科学判据**，GPU-h 记账见 §6。

## 4. 逐阶段回执

| 阶段 | 文件 | exit_code | status | elapsed_seconds |
| --- | --- | ---: | --- | ---: |
| prepare | `prepare_process.json` | 0 | success | 466.804 |
| archive | `archive_process.json` | 0 | success | 468.404 |
| seed 41（两臂） | `seed41_process.json` | 0 | success | 18339.620 |
| seed 42（两臂） | `seed42_process.json` | 0 | success | 19082.440 |
| seed 43（两臂） | `seed43_process.json` | 0 | success | 15775.460 |
| reading | `reading_process.json` | 0 | success | 428.899 |

六阶段全 `success`、`returncode=0`、无信号；每阶段 spawn 前只读 `gpu_gate`（`nvidia-smi read-only`）；
**未对任何非本实验进程发送任何信号**（邻居全程只共享显存，R-054 / 决策 0026）。

## 5. 公平性守卫

- **单因素**：两臂同 seed、同预算、同父状态、同锚表；初始化状态除两个 anchor buffer 外逐位相同
  （`parent_state_sha256`/`expanded_state_sha256` 两臂逐字相等）；唯一差异是 `anomaly_feedback` 布尔，
  它只改变 `model/process_step_r7.py:108` 的 `feedback_draft = draft - climate`。
- **负控制（N 本身）**：`anomaly_feedback=False` 时锚不被任何前向路径读取；实测控制臂与无锚模型
  `unanchored_max_abs=0.0`、`bit_identical_outside_anchor=true`（`n_shared_compared=127`，
  新增键恰为两个 anchor buffer）。
- **正控制（表清零）**：`table.zero_()` 后 `climate=0` ⇒ `feedback_draft=draft`；实测候选臂逐位塌回控制臂
  （`zeroed_table_delta_max_abs=0.0`、`zeroed_table_is_identity=true`），逐 seed 断言失败即 `raise`。
- **锚表同一性**：每 seed 驱动断言两臂 `anchor_table_sha256` 相同（不同即 `raise`），
  六份回执均为 `da2f28c9…`；三 seed 的 `anchor_init_sha256` 全为 `de0e76be…`。
- **cohort 与监督**：两臂三 seed 的逐 lead cohort 计数与已登记对照臂**逐字相同**（472/468/460/444/428）；
  `mask_sha256=4a2a576b…` 与冻结监督块一致。
- **pinned 外部参考**：reading 阶段按 `boundary_rmse_csv_sha256` 复核已登记 2400 剂量臂的逐 seed 逐 lead 表
  （`scripts/study_r7_s3_anchor_isolation.py:418-426` 的 `_reference_delta`，drift 即 `raise`）；
  reading 阶段 exit 0 ⇒ 15 张表全部匹配其 pin，A **未重训练**。

## 6. 成本

| 阶段 | whole 秒 | GPU-h | 说明 |
| --- | ---: | ---: | --- |
| prepare（CPU） | 466.804 | 0.129668 | |
| archive（CPU） | 468.404 | 0.130112 | |
| seed 41（两臂 2400 更新 + 五 lead） | 18339.620 | 5.094339 | |
| seed 42（两臂 2400 更新 + 五 lead） | 19082.440 | 5.300678 | 与 41 并发（双卡） |
| seed 43（两臂 2400 更新 + 五 lead） | 15775.460 | 4.382072 | |
| reading（CPU） | 428.899 | 0.119139 | |
| **本轮合计** | 54,561.627 | **15.156007** | 累计 105.619618 → **120.775625** |

- **GPU-h 口径（保守）**：attempt 目录下全部 6 个 `*_process.json` 的 `elapsed_seconds` 之和 ÷ 3600
  = 54,561.626982 s ÷ 3600 = **15.156007 GPU-h**；attempt 目录下**无** attempt 级 wall clock 文件，故直接取
  该和。因 seed 41 与 seed 42 在两张卡上**并发**，逐阶段求和大于真实墙钟——这是**保守（偏大）**口径。
  （上表逐阶段 GPU-h 为 6 位小数显示，四舍五入后相加为 15.156008，与 15.156007 差 1e-6，属显示舍入。）
- **soft overrun 0、hard overrun 0**（per-seed 43200 s 内，planned 86400 s 内）。
- 网络 **0**（离线，R-028）。磁盘 ≈40 GiB（两臂 × 三 seed 的训练检查点各约 13 GiB）。
- **峰值口径**：协议 `estimated_peak_bytes` 写 9.5 GiB，六份回执实测 `owned_cuda_reserved_peak_bytes`
  均为 8,996,782,080 B = **8.379 GiB**——本轮估计**偏高约 1.12 GiB**，余量门是保守的。

## 7. 已确认与推测

**已确认（读到的产物）**：上表全部 exit_code / elapsed / receipt / 逐 lead 评分 CSV / cohort 计数；
六个 checkpoint SHA256；逐 seed 两臂 `parent_state_sha256`/`expanded_state_sha256` 逐字相同、
`anchor_init_sha256` 与 `anchor_table_sha256` 全同；正控制四项（无锚逐位、表清零逐位、反馈被读取、输出先验可观测）
逐 seed 实测；`readings.json` 的逐 lead rmse / Δ / skill（F−N、两臂 vs 剂量臂）；
reading 阶段 exit 0 表明 2400 剂量臂的 15 张 `boundary_rmse.csv` 匹配其 pin。

**推测（未证）**：① F−N 的改善是否能在**其他实例/区域/年份**复现——本轮只覆盖一个宽实例、一个 ROI、
2017–2021 train / 2022 val；② 72 h 残余负 skill（seed 42）的主因仍未定位，候选有自回归误差累积、
目标/训练信号组织、数据覆盖，三者本轮**未分离**；③ 若把 `C_valid` 换成全年 12 个月气候态、
或改变表粒度，F 的 72 h 读数会怎样——**本轮未测**，且按预声明出口**不得**在本族内再调；
④ 输出先验与 proposal 先验各自的贡献**未测**（两者本轮都关，且仓库规则使二者无法在本仓分开）。

**未做**：不重训练任何已登记臂（A pinned 复用）、不读 test、不做显著性检验、不报泛化/SOTA/收敛因果、不加 seed、
不调表粒度、不减锚、不重开架构/容量/加权损失支。

## 8. limitations

- **输出先验与 proposal 先验未分离（必须记住）**：二者被仓库的校验规则与「锚存在」耦死——
  `model/process_forecast_r7.py:65-66` 规定 `anomaly_feedback` 必须有锚；
  `model/process_forecast_r7.py:252-253` 规定 `anchor_output_prior=False` 也必须有锚；
  而锚一旦存在，输出头锚（`model/weather_forecaster_r7.py:122`，`anchor_output_prior` 决定
  `climate` vs `base`）与 proposal 锚（`model/process_step_r7.py:104`，同一开关决定 `climate` vs
  `anchor`）**同时**被切换。本轮把两者**都关掉**、只孤立反馈表征，故 **package 的 48/72 h 增益不能
  从本轮归因到那两个先验**；本轮也**不能**分离这两个先验彼此的贡献。若将来要测它们，必须再加开关并
  重新预声明，不能借用本轮读数。
- **气候态是 4 个月季节均值**：train/val 只覆盖 Jan/Apr/Jul/Oct，锚表是 16 个 `(month,hour)` 桶的
  train-only 均值（每桶 150 帧），**不是 12 个月年循环**；现有 val 也只能检验这 4 个月，
  不能检验「全年扩充是否有效」。
- 单实例、单 ROI、17 通道、2017–2021 train / 2022 val 的开发筛选；两臂三 seed 只是一致性证据，
  **无显著性**、无收敛/SOTA/泛化声明。
- **F−N 不是 null，故本族关闭不是「反馈表征无效」的证明**：配对改善在 15/15 个 seed×lead 单元同号为负，
  72 h 均值 −0.321 K；关闭的原因是**绝对 72 h 门**在三 seed 上未全正（seed 42 −0.038377）。
  不能把本轮读成「异常反馈表征无效」，也不能把 F 的配对改善升级为科学确认。
- 外部参考 A（2400 剂量臂）是 **pinned 复用**，不是同协议内同批重训练；协议内对照 N 是本轮重训练的，
  故 **F→N 是 in-protocol**，而 F/N vs A 是跨协议比较。
- 两臂都**与注册父的权重联合训练**，故每臂价格的是「从注册父出发的一条反馈表征路由」，
  不是该机制本身的孤立效应。
- `interior_32` 监督使外圈在训练中无约束；只评分 `interior_32`。
- 控制臂与无锚模型的逐位相同是**初始化/前向**层面的断言；训练后与剂量臂的差在训练噪声级
  （−4.8e-6…+1.8e-5 K），不是逐位相同。
- 三 seed 是一致性证据，不是显著性检验；GPU 运行共驻，seed 42 的部分墙钟因此较长。
- 可复现等级：`protocol_sha256` + `code.zip` SHA + 逐 seed×两臂 receipt（含 `expanded_state_sha256`）+
  `readings.json` 齐备，可按同一协议复现；但 **GPU 训练非逐位可复现**（同机同 seed 也不保证逐位一致），
  故只能复现「同一协议」，不能复现「同一结果」。

## 9. 下一动作

按**决策 0048**（`docs/decisions/0048-capacity-axis-scope-and-cap.md`）与本轮预声明出口：
**永久关闭「气候态锚」这一族**——不调表粒度、不减锚、不加 seed、不重跑。H-D 是 H-C 关闭后转
训练信号/目标组织方向的**第一根**轴；它未满足出口，故转主线重估或另一条可证伪假设
（数据按 D3 暂缓；FY 全年数据由另一会话持有，本轮**不得**使用）。

**结论如实写清**：F−N 是**大而一致**的配对改善（72 h −0.321 K），不是 null，故「异常反馈表征无效」
的结论**不成立**；本族关闭是因为**72 h 绝对 skill 三 seed 未全正**（+0.010101/−0.038377/+0.030003），
即候选仍未把 72 h 推过气候态。由于本轮两臂都关掉输出先验与 proposal 先验，**package 的 48/72 h 增益
不能从本轮归因到那两个先验**；若要检验它们，须另立开关、另写 round brief、另冻预声明出口。

test 未读、r=0、S4 未启动不变。
