# S3 H-C 长 lead 的全局混合路径（global N×N 注意力 vs 同参数量额外局部层）单因素臂：G→L 无差异（~1e-3…1e-2 K），72 h 两臂三 seed 全负，按预声明出口关闭 H-C 支

**状态：已完成（2026-10-15，两臂 × 三 seed 全部跑完并评分）；`scientific_claim: false`；2023 test 未读；r 不消耗。**
本页对应 `docs/goals/main-model-climatology-campaign.md` §8 与 `docs/goals/s3-global-mixing.md`。
它是**开发筛选读数**，不是科学确认，也不是 S4 就绪声明；`primary`/`gate` 形式在这里只是**报告**，
因为外部对照（A）是已登记臂的 pinned 复用，而协议内对照（L）是本轮同协议内重训练的。

**同一因子的登记序号**：这是「主干出口**一条** context-mixing 层的**混合范围**（全局 N×N 注意力
vs 主干窗口注意力）」这一因子的**第一次**登记。本因子此前**没有**任何已登记 record
（H-A 是 `record:s3-process-init-result`、H-B 是 `record:s3-typed-spatial-result`，均为**不同因子**），
故本页**不**向任何旧 record 追加 `correction_record_id`。

## 1. 结论摘要

- **单因素**：在**已登记的最佳臂**上，只在主干出口加**一条** context-mixing 层（`global_mixing_mode`
  `None`→`global`），并把它的**注意力范围**作为唯一因子：`global`=对全部 1089 个 token 做 **N×N** 注意力
  vs 协议内对照 `local`=主干同款**窗口注意力**（`window_size`=4）。其余（宽 129×129 输入、`interior_32` 监督、
  2400 更新、K=4、depth=4、`dim`=192、注册物理权重 `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)`、迁移 v3-BD 1600 父、
  seed 41/42/43、评分域与气候态分母）逐字不变。**唯一因子是该层的混合范围**。
- **G/L 同协议内同跑**：global（G）与 local（L）在**同一协议、同一 seed、同一预算、同一通路初始化**下各跑一遍。
  两臂由**同构 block 外壳**声明**同名同形状的 12 个张量**，故**参数量逐位对齐 = 444,864**
  （`parameter_matched_by_construction=true`；`shell_parity.parameter_counts_equal=true`）。同 seed 下两臂
  **除这 12 个新增键外逐位一致**（`shared_bit_identical=true`、`n_shared_compared=167`；两臂
  `parent_state_sha256`、`expanded_state_sha256` 逐字相同，见 §3），两臂 `pathway_init_sha256` **相等**
  （41 `55a9bd77…` / 42 `db5efa02…` / 43 `2c5e44bc…`）。**这是 L→G 可归因到混合范围的前提。**
- **正控制逐 seed 通过**：把混合块的**输出投影与 FF 输出层同时清零** ⇒ 该层退化为恒等 ⇒ **整个 forward
  逐位退回 off 路径**（`zeroed_block_is_identity=true`、`zeroed_block_max_abs=0.0`）；**只清零输出投影不够**
  ——实测 `proj_only_is_identity=false`、`proj_only_max_abs` 0.18–0.23（残差之后还有 FF）；负控制＝两臂在
  大于一个窗口的 token 网格上 forward **必须不同**（`arms_are_not_identical=true`、`arms_probe_max_abs`
  0.31–0.43）。
- **G→L 无差异**：G−L 的 `interior_32` t2m Δrmse 在五个 lead 上 seed 均值为
  **−0.00019 / −0.00003 / −0.00217 / +0.00186 / −0.00247 K**，**五个 lead 三 seed 全部不同号**
  （6h +,+,−；12h +,−,−；24h +,−,−；48h +,+,−；72h +,−,+）。即混合范围**改变不了**读数。
- **72 h 两臂三 seed 全负**：G 臂 `interior_32` t2m skill 为 **−0.1748 / −0.2650 / −0.1511**（均值 −0.1970），
  L 臂 **−0.1745 / −0.2709 / −0.1506**（均值 −0.1987）。**预声明出口**（**72 h 三 seed 全正** **且** G 相对 L
  在 72 h 三 seed 同号为优）**两条都未满足**——72 h 不是全正，G−L 也不是同号为优。
- **结论（必须照此读）**：**L→G 无差异** ⇒ 结论是「**瓶颈不是感受野**」（即窗口/局部混合范围不是长 lead 的
  约束），**且该无差异与「额外容量」不可分**——两臂都加了同样的 444,864 参数，故「加一层」本身是否有效
  本轮**未**与「全局性」分离。72 h 的负 skill 在两臂里都出现，故它与混合范围无关。
- **相对已登记 2400 剂量臂 A**（`record:s3-wide-interior-dose`，pinned 复用、未重训练）：G 臂 48 h 三 seed 全优
  （Δ −0.0052 K）、72 h 2/3 优（−0.0085 K）；L 臂 48/72 h 三 seed 全优（−0.0071 / −0.0061 K）；6/12/24 h 两臂
  均约 1/3 seed 更优、seed 均值 +0.0003…+0.0033 K（略差）。
- **判定**：按**决策 0048**（`docs/decisions/0048-capacity-axis-scope-and-cap.md`），H-C 是 H-A、H-B 关闭后的
  **第三根架构**轴，容量轴（深度＋宽度）已**永久关闭**。本轮预声明出口未满足 ⇒ **关闭 H-C 这一支**——
  不调宽度/秩/层数、不加 seed、不重跑。本臂**不进 S4 冻结包**。

**表 A：G→L 配对 delta（G − L，同协议同 seed 同通路初始化）**

| lead | Δrmse G→L（K，seed 均值） | 范围 | 三 seed 同号 | G skill 均值 | G skill(41) | G skill(42) | G skill(43) |
| ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: |
| 6 h | −0.00019 | −0.00461 … +0.00218 | 否 | +0.5823 | +0.577990 | +0.577014 | +0.592034 |
| 12 h | −0.00003 | −0.00221 … +0.00229 | 否 | +0.3270 | +0.322437 | +0.318835 | +0.339796 |
| 24 h | −0.00217 | −0.00549 … +0.00151 | 否 | +0.4203 | +0.419341 | +0.407133 | +0.434437 |
| 48 h | +0.00186 | −0.00200 … +0.00630 | 否 | +0.0703 | +0.068156 | +0.028499 | +0.114141 |
| 72 h | −0.00247 | −0.00867 … +0.00080 | **否** | **−0.1970** | −0.174751 | −0.264991 | −0.151148 |

**表 B：两臂相对已登记 2400 剂量臂 A 的配对 delta（本臂 − 剂量臂；负值 = 本臂更优）**

| lead | Δrmse G−剂量（K，seed 均值） | 范围 | 更优 seed | Δrmse L−剂量（K，seed 均值） | 范围 | 更优 seed |
| ---: | ---: | ---: | :---: | ---: | ---: | :---: |
| 6 h | +0.0010 | −0.0042 … +0.0043 | 1/3 | +0.0012 | −0.0060 … +0.0089 | 1/3 |
| 12 h | +0.0033 | −0.0026 … +0.0089 | 1/3 | +0.0033 | −0.0024 … +0.0111 | 1/3 |
| 24 h | −0.0019 | −0.0078 … +0.0012 | 1/3 | +0.0003 | −0.0093 … +0.0067 | 1/3 |
| 48 h | −0.0052 | −0.0118 … −0.0013 | **3/3** | −0.0071 | −0.0181 … −0.0005 | **3/3** |
| 72 h | −0.0085 | −0.0180 … +0.0002 | 2/3 | −0.0061 | −0.0094 … −0.0003 | **3/3** |

**表 C：两臂逐 lead `interior_32` t2m skill（1 − (rmse/rmse_climatology)²）**

| lead | G skill 均值 | G(41) | G(42) | G(43) | L skill 均值 | L(41) | L(42) | L(43) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 h | +0.5823 | +0.577990 | +0.577014 | +0.592034 | +0.5823 | +0.578796 | +0.577702 | +0.590359 |
| 12 h | +0.3270 | +0.322437 | +0.318835 | +0.339796 | +0.3270 | +0.323519 | +0.318751 | +0.338768 |
| 24 h | +0.4203 | +0.419341 | +0.407133 | +0.434437 | +0.4193 | +0.420012 | +0.406001 | +0.432032 |
| 48 h | +0.0703 | +0.068156 | +0.028499 | +0.114141 | +0.0713 | +0.071779 | +0.029254 | +0.113015 |
| 72 h | **−0.1970** | −0.174751 | −0.264991 | −0.151148 | **−0.1987** | −0.174450 | −0.270872 | −0.150632 |

（Δ 为「本臂 − 对照」，负值 = 本臂更优；G 与 L 的评分域、气候态分母、cohort pin 逐字相同，A 为同 seed 配对的
pinned 复用，故可直接相减。逐 lead 的绝对值与 Δ 见
`outputs/r7_s3_global_mixing_20261015_attempt01/readings.json`；G 与 L 的逐 lead Δ 见**表 A**。）

## 2. 身份与协议

| 项 | 值 |
| --- | --- |
| attempt 目录 | `outputs/r7_s3_global_mixing_20261015_attempt01/` |
| 协议 digest（`protocol.json`，最终；嵌入字段；与 `readings.json` 逐字一致） | `36e93cc8d023b8a0f627651f72f48f4023e4be236de95d10c17498926b5133b6` |
| 准备/预备 digest（`freeze` 阶段报告，`preparation_protocol.json`） | `b841c34fe9038caf3a16faae326cbb5ed87199a3505eb47e568f64c94d09d2f2` |
| 归档 `code.zip` SHA256 | `d0dfe26d8e33e068628fd6fe2010e45f1d7fec956ab95556366e80ac60231d9f`（1412 条目） |
| ZIP comment / 代码 commit | `d560bbeef31826bb5afc2e8440b34b1d462bbf0a`（comment 与 `code_commit.txt` 逐字一致） |
| `model_code_sha256` | `fff81dcc184373cc9253ea6a24ce228d32cc0d0406f3a1766f62fd270b4c02df`（本轮改了 `model/*.py`，故与 H-B 的 `15523927…` 不同） |
| `source_sha256` | `6bc9a1a2d93d345ce0dcb39d4f3a26f175ea3bcc15082ac4931a179ca13b7701` |
| `training_code_sha256` | `8ad77d9349bdc4930e6bbba05f0c417814fe3093fe39aa921e816af197494974` |
| 执行文件身份 | 189 个（`execution_files_sha256` 的键数） |
| 监督块（冻结） | `interior_32`；runner block `spatial_mask`、`[129,129]`、4225 点、`mask_sha256=4a2a576bf948b92924720fa28a44c128a61445dd29b4906805930430e0a0919a`（与对照臂相同） |
| **声明差异（冻结）** | 两臂各恰加 **444,864** 参数（12 键，`global_mixing.block.{n1,qkv,proj,n2,ff}`）；**参数量逐位对齐**、**算力不对齐**（见 §6、§8） |
| 父（迁移 v3-BD 1600，逐 seed pin） | 41 `e8b3188d…` / 42 `2766f601…` / 43 `73d1053b…`（`outputs/r7_s3_rollout_dose_parent_migrated_20261015_attempt02/`） |
| 设备（决策 0047 双卡） | `device_mode=dual`；`gpu_uuids=[GPU-408ad137…, GPU-9d1624af…]`；`neighbour_reserve_bytes=6 GiB`；`estimated_peak_bytes=9.5 GiB`；`headroom_margin_bytes=0.5 GiB` |
| 预算 | planned 86400 s / hard 115200 s / per-seed 43200 s（**soft/hard overrun 均 0**） |
| cohort pin | 472 / 468 / 460 / 444 / 428（逐 lead，两臂三 seed 一致） |
| 外部参考 A | 2400 剂量臂（`record:s3-wide-interior-dose`，pinned，**未重训练**） |
| test | `test.jsonl` 从不读取；`test_read=false` |

## 3. 训练与评分回执

| seed | 臂 | checkpoint SHA256 | 训练 s | 五 lead 评分 s | 通路初始化 `pathway_init_sha256` | 峰值 reserved |
| ---: | --- | --- | ---: | ---: | --- | ---: |
| 41 | global (G) | `14cf25c3203483432f0fd7a9d958e87714336a4bd3319d1edf1a1b3f1b7d6f2f` | 5079.2 | 906.8 | `55a9bd77671d9f192b5df7fbd825b772a028f27a3ac43dd7c0e0ff0fb7402fb1` | 8.955 GiB |
| 41 | local (L) | `99fb51bc3e87ec86056e1972f0f42bb801dff2e8929c6379ca3d977af2c5e5ee` | 4935.6 | 956.9 | `55a9bd77671d9f192b5df7fbd825b772a028f27a3ac43dd7c0e0ff0fb7402fb1` | 8.959 GiB |
| 42 | global (G) | `a879a968538fcc032597e1057cc5a1f9fa21425b6b9d4a5e56c534cecccc7f32` | 5390.6 | 916.1 | `db5efa0232caf2494ace2f6d3ef153f8f89d66a0f1f989ea0863913d568703e0` | 8.955 GiB |
| 42 | local (L) | `f972bb766c077711a61be754314053d3cd9ec2b25c7e024dad77591fcdedc078` | 5066.4 | 933.1 | `db5efa0232caf2494ace2f6d3ef153f8f89d66a0f1f989ea0863913d568703e0` | 8.959 GiB |
| 43 | global (G) | `fa4cd50ce89d3546596ab672c5e7a4048d8e0139ca643c8740821e4a0efced04` | 5065.6 | 913.5 | `2c5e44bc7e00f3c64ad69ef84e916e9a5c6d202b9b4715e0f01664efaacf6793` | 8.955 GiB |
| 43 | local (L) | `3620f13a611f4870c90b0b8d4aef55a4151cbfecb662ea2740191e4cb93df6ec` | 4991.3 | 935.3 | `2c5e44bc7e00f3c64ad69ef84e916e9a5c6d202b9b4715e0f01664efaacf6793` | 8.959 GiB |

- 六个训练都到 `total_updates=2400`；每 seed 每臂每 lead `n_evaluated == n_available_windows` 且等于冻结
  cohort（472/468/460/444/428）；每 seed 每臂的 `supervision` 与冻结 `mask_sha256` 逐字相同（驱动断言）。
- **同 seed 两臂的通路初始化逐位相同**：`pathway_init_sha256` 在两臂相等（41 `55a9bd77…` / 42 `db5efa02…`
  / 43 `2c5e44bc…`）；`parent_state_sha256` 随 seed 不同（41 `7c416ccb…` / 42 `857d8804…` / 43 `61fdac58…`），
  **同 seed 两臂 `expanded_state_sha256` 也逐字相同**（41 `97ac5e3a…` / 42 `84620339…` / 43 `7ba50956…`）
  ——这正是「12 个新增张量在 `torch.manual_seed(seed)` 下逐位相同、两臂只差混合范围」的意思。
- **正控制逐 seed 通过**：`shared_bit_identical=true`、`n_shared_compared=167`；
  `shell_parity.parameter_counts_equal=true`（两臂各 `444864`）；`zeroed_block_is_identity=true` 且
  `zeroed_block_max_abs=0.0`（清零输出投影 **与** FF 输出层 ⇒ 层为恒等 ⇒ 整个 forward 退回 off 路径）；
  `proj_only_is_identity=false` 且 `proj_only_max_abs` 0.183–0.225（**只清投影不等于 off**，实测反证）；
  `arms_are_not_identical=true` 且 `arms_probe_max_abs` 0.312–0.429（两臂在大于一个窗口的网格上 forward 不同）。
- seed 42 的部分墙钟较长（训练 5390.6 s）源于**两臂与邻居共驻**——**墙钟不是科学判据**，GPU-h 记账见 §6。

## 4. 逐阶段回执

| 阶段 | 文件 | exit_code | status | elapsed_seconds |
| --- | --- | ---: | --- | ---: |
| prepare | `prepare_process.json` | 0 | success | 474.082 |
| archive | `archive_process.json` | 0 | success | 474.443 |
| seed 41（两臂） | `seed41_process.json` | 0 | success | 14110.153 |
| seed 42（两臂） | `seed42_process.json` | 0 | success | 14551.025 |
| seed 43（两臂） | `seed43_process.json` | 0 | success | 14153.615 |
| reading | `reading_process.json` | 0 | success | 473.411 |

六阶段全 `success`、`returncode=0`、无信号；每阶段 spawn 前只读 `gpu_gate`（`nvidia-smi read-only`）；
**未对任何非本实验进程发送任何信号**（邻居全程只共享显存，R-054 / 决策 0026）。

## 5. 公平性守卫

- 两臂三 seed 的逐 lead cohort 计数与已登记对照臂**逐字相同**（472/468/460/444/428）。
- **共享初始化（H-B 轮 medium finding 的修复，必须照搬）**：同 seed 两臂的 `pathway_init_sha256` 相等、
  `parent_state_sha256` 与 `expanded_state_sha256` 逐字相同，共享张量 `shared_bit_identical=true`、比较 167 个键
  ⇒ L→G 的差异只能归因到**混合范围**，**不是**归因到协议/容量/随机初始化差异。
- **参数量逐位对齐**：两臂由同构 block 外壳声明同名同形状的 12 个张量，各加**恰好 444,864** 参数
  （`shell_parity.parameter_counts_equal=true`、`new_params=444864`）；因此 L→G **不**是「单纯加容量」的对照。
- **算力不对齐（已声明）**：两臂的注意力长度 68.0625×、SDPA 子成本 57.19140625×、整层 1.9129018382845497×
  （解析值，`reference` 见 receipt 的 `positive_control.flops`）——**这一点是已声明的混杂，见 §8**。
- **正/负控制**：正控制三条（清零 proj+ff ⇒ 恒等退回 off；只清 proj ⇒ **不**等于 off；两臂 forward 不同）
  逐 seed 通过（§3）；负控制＝L 臂本身（同参数量、只改混合范围）。
- reading 阶段按 `boundary_rmse_csv_sha256` 复核已登记 2400 剂量臂的逐 seed 逐 lead 表
  （`scripts/study_r7_s3_global_mixing.py:413-421` 的 `_reference_delta`，drift 即 `raise`）；reading 阶段 exit 0
  ⇒ 15 张表匹配其 pin，A **未重训练**。
- 气候态分母取自 `wide_climatology_reference`（train-only、与对照臂同一分母），`readings.json` 里的
  `protocol_sha256` 与 `protocol.json` 的最终 digest 逐字一致。

## 6. 成本

| 阶段 | whole 秒 | GPU-h | 说明 |
| --- | ---: | ---: | --- |
| prepare（CPU） | 474.082 | 0.1317 | |
| archive（CPU） | 474.443 | 0.1318 | |
| seed 41（两臂 2400 更新 + 五 lead） | 14110.153 | 3.9195 | |
| seed 42（两臂 2400 更新 + 五 lead） | 14551.025 | 4.0420 | 与 41 并发（双卡） |
| seed 43（两臂 2400 更新 + 五 lead） | 14153.615 | 3.9316 | |
| reading（CPU） | 473.411 | 0.1315 | |
| **本轮合计** | 44,236.729 | **12.287980** | 累计 93.331638 → **105.619618** |

- **GPU-h 口径（保守）**：attempt 目录下全部 6 个 `*_process.json` 的 `elapsed_seconds` 之和 ÷ 3600
  = 44,236.729079 s ÷ 3600 = **12.287980 GPU-h**；attempt 目录下**无** attempt 级 wall clock 文件，故直接取
  该和。因 seed 41 与 seed 42 在两张卡上**并发**，逐阶段求和大于真实墙钟——这是**保守（偏大）**口径。
  （上表逐阶段 GPU-h 为 4 位小数显示，四舍五入后相加为 12.2881，与 6 位小数的 12.287980 差 0.0001，属显示舍入。）
- **soft overrun 0、hard overrun 0**（per-seed 43200 s 内，planned 86400 s 内）。
- 网络 **0**（离线，R-028）。磁盘 ≈14 GiB。
- **峰值口径**：协议 `estimated_peak_bytes` 写 9.5 GiB，实测两臂峰值 8.955 GiB（G）与 8.959 GiB（L）
  ——本轮估计**略高**（高估约 0.54 GiB），故余量门是保守的；两臂共享同一 `per_seed_seconds`/peak 估计，
  不破坏 L→G 可比性。

## 7. 已确认与推测

**已确认（读到的产物）**：上表全部 exit_code / elapsed / receipt / 逐 lead 评分 CSV / cohort 计数；
六个 checkpoint SHA256；逐 seed 两臂 `pathway_init_sha256` 相等、`parent_state_sha256`/`expanded_state_sha256`
逐字相同；正控制三条逐 seed 通过；`readings.json` 的逐 lead rmse / Δ / skill（G→L、两臂 vs 剂量臂）；
reading 阶段 exit 0 表明 2400 剂量臂的 15 张 `boundary_rmse.csv` 匹配其 pin。

**推测（未证）**：① **L→G 无差异**是否等价于「感受野**完全**不是瓶颈」——本轮只证明「在全局 N×N 与窗口
两种范围下读数不变」，未分离「混合范围无贡献」与「两种范围在本设置下等价」。② 72 h 残余负 skill 的
**主因仍未定位**：候选有自回归误差累积、目标/训练信号组织、数据覆盖，三者本轮**未分离**。③ 若换**低秩/池化
全局算子**（把注意力 FLOPs 从 O(N²) 降到 O((N/s²)²)，从而**同时压算力**）或让该层有更大学习幅度，
L→G **是否**会有差异——本轮**未测**，不能由本 null 外推。

**未做**：不重训练任何已登记臂（A pinned 复用）、不读 test、不做显著性检验、不报泛化/SOTA/收敛因果、不加 seed。

## 8. limitations

- 单实例、单 ROI、17 通道、2017–2021 train / 2022 val 的开发筛选；两臂三 seed 只是一致性证据，**无显著性**。
- **算力不对齐（本轮最重要的混杂，必须记住）**：两臂**参数量**逐位对齐（各 444,864），但**算力不对齐**——
  全局臂的注意力长度是局部臂的 **68.0625×**、**仅 SDPA 子成本** **57.19140625×**、**整层**（n1+qkv+sdpa+proj+n2+ff）
  **1.9129018382845497×**（FF 占大头，故整层差远小于注意力差）。前两个数字**不是同一个量**。因此**若 G 胜出，
  差异可能来自算力而非「全局性」**——本轮**不得**声称 FLOPs 已对齐，也**不得**把 G 的任何优势无条件归因到
  「全局性」。要同时压 FLOPs 须改用**池化/低秩全局算子**（先在粗网格做注意力再上采样），而不是全分辨率 N×N；
  低秩路线**也无法**逐位对齐参数量（整数 r 下 `Linear(dim,r)+Linear(r,dim)` 取不到 148,224）。
  **（本轮实际读数里 G 并未胜出，故此混杂未兑现为「假阳性优势」；但它限定了这个 null 的可归因范围。）**
- **L→G 的 null 与「额外容量」不可分**：两臂**都**加了同样的 444,864 参数，故「加一层是否有效」本轮**未**与
  「全局性」分离。L→G 无差异只说明**混合范围**不是瓶颈；它**不**说明「加这层容量」对长 lead 无贡献，
  也**不**说明「加这层容量」对长 lead 有贡献——两者本轮都**未分离**。
- 外部参考 A（2400 剂量臂）是 **pinned 复用**，不是同协议内同批重训练；协议内对照 L 是本轮重训练的，
  故 **L→G 是 in-protocol**，而 G/L vs A 是跨协议比较。
- 两臂都**与注册父的权重联合训练**，故每臂价格的是「从注册父出发的一条 context-mixing 路由」，
  不是该层本身的孤立效应。
- `interior_32` 监督使外圈在训练中无约束；只评分 `interior_32`。
- **L→G 的 null 在 ~1e-3…1e-2 K 量级**（大于 H-B 的 float32 噪声 ~1e-6 K）：本设计能分辨约 1e-3 K 的差异，
  而它没有出现；但同一尺度也意味着本实验**无法**区分「两臂混合范围完全相同」与「两臂差异 < 1e-3 K」。
- 三 seed 是一致性证据，不是显著性检验；GPU 运行共驻，seed 42 的部分墙钟因此较长。
- 可复现等级：`protocol_sha256` + `code.zip` SHA + 逐 seed×两臂 receipt + `readings.json` 齐备，
  可按同一协议复现；但 **GPU 训练非逐位可复现**（同机同 seed 也不保证逐位一致），
  故只能复现「同一协议」，不能复现「同一结果」。

## 9. 下一动作

按**决策 0048**（`docs/decisions/0048-capacity-axis-scope-and-cap.md`）与本轮预声明出口：
**关闭 H-C 这一支**——不调宽度/秩/层数、不加 seed、不重跑。**结论如实写清**：L→G 无差异 ⇒
**瓶颈不是感受野**，且该无差异**与「额外容量」不可分**（两臂都加了同样的 444,864 参数）。

H-C 是 H-A、H-B 关闭后的**第三根架构**轴；容量轴（深度＋宽度）已按 0048 **永久关闭**。此后架构方向
（plan 0022 的 H-A/H-B/H-C）已全部关闭；训练信号 / 目标组织方向亦开放（数据按 D3 暂缓、放到最最后）。
下一轮应先写新的 round brief，再 `round-guard` → `freeze` → 逐 stage → 四件套登记。

test 未读、r=0、S4 未启动不变。
