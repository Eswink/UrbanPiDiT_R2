# S3 H-C 全局混合轮的计算量口径勘误（2026-11-02）

**只读独立复核；`scientific_claim: false`。不重写原实验、不改任何冻结判据与终态、不重跑训练。**
本页更正 `docs/R7_S3_GLOBAL_MIXING_RESULT.md`（索引记录 `record:s3-global-mixing-result`）
及其协议/回执里 `declared_compute_gap` 与 `positive_control.flops` 的**口径错误**：
那四个数字是按 **1089 tokens** 算的，而该轮实际训练与评分的模型输入是 **129×129**，
对应 **4225 tokens**。原页的**科学结论（L→G null）不受影响**，受影响的是
「算力差多少」这一条已声明混杂的**幅度**。

原页与索引记录的 digest 不改，反向引用继续有效；引用修正后的计算量须同时引用本页。

## 1. 勘误的事实基础（五项独立实测）

| # | 实测 | 命令/来源 | 结果 |
| ---: | --- | --- | --- |
| 1 | 该轮 store 的空间形状 | `zarr.open(.../r7_s3_wide_instance_v4_20261009_attempt01/store/cache.zarr)["state"].shape` | `(3360, 17, 129, 129)` |
| 2 | 该轮训练集的真实样本形状 | `ZarrLongRolloutDataset(train.jsonl, physical_steps=12)[0]` | `coarse_history (2,17,129,129)`、`atmos_target (17,129,129)`、`physical_targets (12,17,129,129)` |
| 3 | 该轮模型规格 | `protocol.json` 的 `arms.global.model.spec` | `in_channels 17, patch_size 2, window_size 4, dim 192, depth 4, heads 4` |
| 4 | 用该轮**归档 `code.zip`** 里的代码构建模型并前向 | `/tmp/hcinspect`（从 `code.zip` 解包的 `training/r7_experiment.py` + `model/**`） | 129×129 输入 ⇒ `token_hw (65,65)`、**4225 tokens**、`forecast (1,17,129,129)` |
| 5 | 用该轮**自己的** `_analytic_flops()`（从 `code.zip` 抽出，原样 exec）在两个几何上各跑一次 | `scripts/r7_s3_global_mixing_parent.py:128-163`（归档副本） | 默认 ref ⇒ `68.0625 / 57.19140625 / 1.9129018383`；传真实 ref（tokens 4225、windows 289）⇒ `264.0625 / 241.2768301254 / 4.5945744831`；比值 `4.218760228953673` / `2.401887222421166` |

第 4 项同时排除了「数据集在进模型前被裁成 65×65」这一解释：对归档执行路径
（`data/r7_long_rollout_dataset.py`、`data/r7_zarr_dataset.py`、`data/r7_autoregressive_dataset.py`、
`model/coarse_encoder.py`、`model/layers/window_attention.py`、`model/weather_forecaster_r7.py`、
`training/r7_long_rollout_runner.py`、`scripts/study_r7_s3_global_mixing.py`）逐个搜过
`coarse_history[` / `crop_native` / `[...:N:` / `frozen_box`，**命中 0 处**。

**65 这个数字是从哪来的**：`region.frozen_box_points = [65,65]` 与
`interior_selector = boundary_masks(129,129,(32,))['interior_32']`。`129 − 2×32 = 65`，
`65² = 4225` 是**评分格数**；`frozen_box_points` 是**被评分的那块区域**，不是模型输入的边长。
`FLOPS_REFERENCE` 里的 `tokens: 1089, padded_hw: [36,36], windows: 81` 恰好是
**65×65 输入**经 `pad_to_patch`（65→66）再 patch-2（33×33）再 window-pad 到 36×36 的结果
（`33²=1089`、`36/4=9`、`9²=81`）。**即：参考块描述的是评分盒的几何，被当成了模型几何。**

## 2. 用该轮**自己的** FLOP 口径重算

沿用归档 `scripts/r7_s3_global_mixing_parent.py` 的 `flops()`（三种口径分开报，
`norms=2nd2`、`qkv=3nd²·2`、`proj=nd²·2`、`ff=2·(nd·4d·2)`，
global `attn=2(n²d·2)`、local `attn=2(heads·windows·16²·(d/heads)·2)`），
只把 `tokens/windows` 换成实际值：

| 口径 | 已登记（tokens 1089, windows 81） | **实际**（tokens 4225, windows 289） | 被低估 |
| --- | ---: | ---: | ---: |
| attention length ratio | 68.0625 | **264.0625** | ×3.8797 |
| global attention FLOPs | 910,787,328 | **13,709,280,000** | — |
| local attention FLOPs | 15,925,248 | **56,819,712** | — |
| **attention ratio（SDPA 子成本）** | 57.19140625 | **241.2768301254**（= 17850625/73984） | **×4.2188** |
| global layer FLOPs | 1,875,101,184 | **17,450,534,400** | — |
| local layer FLOPs | 980,239,104 | **3,798,074,112** | — |
| **layer ratio（整层含 FF）** | 1.9129018382845497 | **4.5945744831**（= 22722050/4945409） | **×2.4019** |

两种口径**都不对**、且**不是同一个量**这一点原页已写明（`docs/R7_S3_GLOBAL_MIXING_RESULT.md:141-143`），
本页只改它们各自的**幅度**；「三种口径须分别报、不得混称」的纪律不变。

## 3. 这条勘误**不**改变什么

- **L→G 的 null 不变**：原页的配对 delta（五个 lead seed 均值 −0.00019 … −0.00247 K、三 seed 不同号）
  是**读数**，与 FLOPs 口径无关；两臂的参数量逐位对齐（各 444,864）也是**构造性**事实，
  由 `shell_parity.parameter_counts_equal=true` 在六个回执里逐 seed 记录，不受 token 数影响。
- **关闭 H-C 的决定不变**：预声明出口（72h 三 seed 全正 **且** G−L 72h 三 seed 同号为优）两条都未满足；
  这是**读数**判定，不是算力判定。
- **`estimated_peak_bytes` 与实测峰值不变**（协议 9.5 GiB / 实测 8.955–8.959 GiB，均按真实 129×129 跑出）。
- **不重跑、不改 outputs、不改任何 digest**。原 `protocol.json`（`36e93cc8…`）与六份回执原样保留。

## 4. 这条勘误**改变**什么（为什么它不是纯文书工作）

**它改变了「H-C 的 null 能外推到什么范围」这一条 limitation 的强度。**

原页 §8 把算力混杂写成「G 若胜出，优势可能来自算力」——即把混杂当成**对 G 有利**的方向。
按实际差（SDPA ×241 而非 ×57、整层 ×4.59 而非 ×1.91），混杂的**幅度**是原记录的两倍多到四倍多，
方向不变但显著更大。因此：

1. **更强的操纵确实被施加过**：原页 §9 与决策 0056 都记着「页内明写『更强操纵下未测』」，
   并据此把它列为可重估理由。按实测，那一轮的 G 臂**已经是**「在 129×129 上做全注意力」——
   `local` 对照也在 4225 tokens 上按 4×4 窗口铺开。**「更强操纵」这一条动机在已关闭臂上已经部分兑现**，
   只是幅度没被如实记下来。
2. **「感受野不是瓶颈」的 null 仍然成立，但它的算力代价被记小了 2.4–4.2 倍**：
   若将来有人以「算力还不够」为理由重开该支，必须先读本页——已有的 null 是在**更大**的算力差下取得的。
3. **对 R3（全局混合重估）准入的直接影响**：计划 0025 §4.2 给 R3 的理由是
   「吃算力不吃显存、表达力型机制最贴合『数据变大』前提」。本页实测表明：
   **新制度（65×65 原生）的 token 数是 1089，正好等于旧轮参考块里那个被误用的数字**；
   即若在 65×65 上重跑，全局/局部注意力长度比是 68.06（而非旧轮的 264.06）。
   换言之**新实例上的操纵比旧实例弱 3.88 倍**（注意力长度）——这与「数据变大所以该重估」的直觉**相反**。
   R3 的准入论证**不能**建立在「更强操纵」上；它只能建立在**曝光匹配 + 数据跨度**上，
   且必须如实写明：**新实例的感受野/算力操纵比旧轮更弱**。

## 5. 已确认与推测

**已确认（本页实跑）**：§1 五项实测；§2 的精确比值（分数形式给出，可复算）；
归档执行路径中无 65×65 裁剪（8 个文件、0 命中）；`FLOPS_REFERENCE` 的 `(tokens, windows)`
与 65×65 输入几何**逐字相等**。

**推测（未证）**：**为什么**会写成 1089——是「照评分盒几何填了参考块」还是别的路径，
本页**无法**从产物判定（`FLOPS_REFERENCE` 是硬编码常量，`protocol.json` 只记录结果不记录推导）。
原页作者意图不可考，本页只记录「写的值与实际跑的不符」这一可复算事实。

**未做**：未重跑训练、未重评分数、未改 `outputs/` 下任何文件、未改原页或原索引记录、
未读 2023 test、未改任何冻结判据、未宣布任何科学通过。GPU-h：**0**。

## 6. 处置

- 原页 `docs/R7_S3_GLOBAL_MIXING_RESULT.md` 与索引记录 `s3-global-mixing-result` **原样保留**，
  不重写、不改 digest；引用修正后的计算量须同时引用本页（同 `s3-v3-numerical-errata` 先例）。
- 新增索引记录 `record:s3-global-mixing-compute-gap-errata`（`outcome_class: audit`、
  `candidate_state: not-candidate`、`gpu_hours: 0.0`），并在 `s3-global-mixing-result` 上追加
  `correction_record_id` 指向它。
- 新增账本行（0 GPU-h、累计不变），不新增 GPU 支出。
- **不**重开 H-C、**不**改 R3 的队列位置、**不**因此把任何旧读数改叙为可用。
