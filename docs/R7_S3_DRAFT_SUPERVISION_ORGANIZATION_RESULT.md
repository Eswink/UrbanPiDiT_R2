# S3 K 步草稿的监督组织（final-only vs 注册深监督）：72 h skill 三 seed 全负、72 h 配对 Δ 符号不一致；短 lead（6/12 h）三 seed 全改善、24/48/72 h 仅 seed 43 改善；预声明出口未满足，该支关闭

**状态：已完成（2026-10-19，候选臂 × 三 seed 全部跑完并评分）；`scientific_claim: false`；2023 test 未读；r 不消耗。**
本页对应 `docs/goals/main-model-climatology-campaign.md` §8 与 `docs/goals/s3-supervision-organization.md`
（该长文的 `<!-- round-spec: ... -->` 声明本轮 `arm=s3-draft-supervision-organization`）。
它是**开发筛选读数**，不是科学确认，也不是 S4 就绪声明。

**同一因子的登记序号**：这是「K 步草稿在训练目标中的**监督组织**」这一因子的**第一次**登记。
本因子此前没有任何已登记 record。它与 `record:s3-anchor-isolation-result`（反馈表征 `draft − C_valid`）、
`record:s3-dense-step-supervision-result`（12 个物理步的监督覆盖）都是**不同**因子；
本轮只把 K 轴的组织从注册深监督改为 final-only，物理权重逐字保持注册稀疏轮廓。

## 1. 结论摘要

- **单因素**：K 轴草稿的监督组织从注册深监督（`initial + K=4` 个草稿，权重
  `linspace(1,2,5)/sum = [0.133333, 0.166667, 0.2, 0.233333, 0.266667]`，非最终草稿占 **73.333%**）
  改为 **final-only**（只对 `drafts[:, -1]` 计算同一个
  `latitude_weighted_mse(..., mask=interior_32)`，权重 1.0）；两侧权重都归一化为 sum=1。
  initial 与 3 个中间草稿不再有直接目标项。
- 其余逐字不变：宽 129×129 输入、`interior_32` 监督（4225 点，`mask_sha256 4a2a576b…`）、
  2400 updates、K=4、depth=4、dim=192、注册物理权重 `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)`、
  迁移 v3-BD 1600 父 attempt04、seed 41/42/43、评分域、气候态分母、cohort。
- **模型前向完全不变**：仍产生 K+1 个草稿、full-BPTT、`internal_k_detach=false`、
  `physical_step_detach=false`；`model_code_sha256=3fe747b2…` 与 H-D 轮逐字相同，**无需新迁移**。
  `training_code_sha256=ce94cc8b…` 因训练损失实现新增 final-only 模式而变化。
- **控制臂 N（pinned，不重训练）**：`record:s3-anchor-isolation-result` 的 control 臂
  （`outputs/r7_s3_anchor_isolation_20261015_attempt01/`；同父、同 seed/预算、注册深监督；
  锚惰性，`anomaly_feedback=False`、`anchor_output_prior=False`）。
  **外部参照 A（pinned，不重训练）**：`record:s3-wide-interior-dose`
  （`outputs/r7_s3_wide_interior_dose_20261009_attempt02/`）。
- **跨 attempt 复用**：无。本轮只有 attempt01；三 seed 全部在本 attempt 内重新训练
  （`resumed_from_updates=0`、`updates_this_run=2400`），未使用 `reuse-plan` 或任何跨 attempt seed 复用。
  N 与 A 是 pinned 复用的**已登记臂**，不属于 seed 复用。
- **结果（预声明出口逐条）**：
  - (a) 候选 72 h `interior_32` t2m skill 三 seed 全正 → **false**：
    **−0.177705 / −0.282612 / −0.150532**（三 seed 全负；均值 −0.203616）。
  - (b) 同 seed 候选−控制 72 h ΔRMSE 三 seed 全负 → **false**：
    **+0.004698 / +0.007867 / −0.008708 K**（seed 均值 **+0.001286 K**，符号不一致）。
  - (c) 候选 48 h skill 三 seed 全正 → **true**：
    **+0.058027 / +0.021656 / +0.116401**。
  - (d) 候选−控制在 6/12/24 h 的 seed 均值 ΔRMSE ≤0 → **false**：
    **−0.017295 / −0.017043 / +0.003828 K**（24 h 均值为正）。
  - (e) 桥一致性：N−A 的 15 个 seed×lead 全部落在 anchor 轮已报告的 `−4.8e-6…+1.8e-5 K` 内
    （实际范围 `−4.787001e-6…+1.778038e-5 K`）→ **true**，`identity_unresolved=false`。
  - ⇒ `scientific_exit_met=false`、`predeclared_exit_met=false`、`identity_unresolved=false`。
  - **按预声明出口永久关闭「K 草稿监督组织：final-only vs 注册 linspace(1,2,K+1) 深监督」这一支**：
    不扫中间草稿权重、不加 seed、不调 2400 updates、不改 partial/加权 final-only、
    不与 anchor/feedback/architecture/容量组合、不换名重开。
- **怎么读（不要误读为「final-only 全面更差」）**：6 h 与 12 h 的 F−N 在**三 seed 全部为负**
  （候选更优），24/48/72 h 只有 seed 43 为负，seed 41/42 为正；15 个 seed×lead 单元中 9 个为负。
  72 h 的绝对 skill 三 seed 全负，且 72 h 配对差均值略正，因此**不能**把本轮读成候选项支持。
  同时，final-only **必然**移除 4 个非最终直接项**并**把最终项归一化权重从 0.2667 提到 1.0；
  两侧权重都 sum=1、总损失尺度可比，但本轮把「组织」作为单一因子整体检验，
  **不声称分离 support（哪些槽被监督）与 emphasis（最终项权重）**。
- **相对 A（pinned 剂量臂）**：与 F−N 几乎相同（控制 N 与 A 的差在训练噪声级）；
  候选 6/12 h 更优（均值 −0.017295/−0.017043 K），24/48/72 h 略差（+0.003830/+0.003171/+0.001292 K）。
- **不进 S4 冻结包**；test 未读、r=0、S4 未启动。

**表 A：逐 seed 的 `interior_32` t2m RMSE、Δ 与 skill**

| seed | metric | 6 h | 12 h | 24 h | 48 h | 72 h |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 41 | candidate RMSE (K) | 2.273891 | 2.861172 | 2.634132 | 3.254693 | 3.601156 |
| 41 | control RMSE (K) | 2.281941 | 2.868234 | 2.627162 | 3.248984 | 3.596458 |
| 41 | Δ candidate−control (K) | -0.008050 | -0.007062 | +0.006970 | +0.005709 | +0.004698 |
| 41 | candidate skill | 0.582010 | 0.327475 | 0.412786 | 0.058027 | -0.177705 |
| 42 | candidate RMSE (K) | 2.263707 | 2.855683 | 2.652043 | 3.316933 | 3.758125 |
| 42 | control RMSE (K) | 2.291619 | 2.882104 | 2.645820 | 3.306613 | 3.750258 |
| 42 | Δ candidate−control (K) | -0.027912 | -0.026421 | +0.006223 | +0.010321 | +0.007867 |
| 42 | candidate skill | 0.585746 | 0.330052 | 0.404773 | 0.021656 | -0.282612 |
| 43 | candidate RMSE (K) | 2.226254 | 2.808324 | 2.582227 | 3.152235 | 3.559369 |
| 43 | control RMSE (K) | 2.242177 | 2.825972 | 2.583936 | 3.158765 | 3.568077 |
| 43 | Δ candidate−control (K) | -0.015922 | -0.017647 | -0.001709 | -0.006530 | -0.008708 |
| 43 | candidate skill | 0.599340 | 0.352089 | 0.435700 | 0.116401 | -0.150532 |

**表 B：控制 N − 已登记剂量臂 A（桥一致性，K）**

| seed | 6 h | 12 h | 24 h | 48 h | 72 h |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | 1.287682e-07 | 3.890023e-07 | 5.205200e-06 | 5.314766e-06 | 5.413140e-06 |
| 42 | -2.619265e-07 | 3.053607e-08 | -1.234054e-06 | -3.200736e-06 | -4.787001e-06 |
| 43 | 2.378506e-08 | 7.430943e-07 | 3.897610e-06 | 1.162615e-05 | 1.778038e-05 |

**表 C：候选 − 已登记剂量臂 A（K；负值 = 候选更优）**

| seed | 6 h | 12 h | 24 h | 48 h | 72 h |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | -0.008050 | -0.007062 | +0.006975 | +0.005714 | +0.004703 |
| 42 | -0.027912 | -0.026421 | +0.006221 | +0.010318 | +0.007863 |
| 43 | -0.015922 | -0.017647 | -0.001705 | -0.006519 | -0.008691 |

**表 D：全 17 变量 × 五 lead 的 seed 均值物理 RMSE 与 Δ（不跨单位平均；q 用科学计数法）**

**6 h**

| variable | candidate (seed mean) | control (seed mean) | Δ candidate−control |
| --- | ---: | ---: | ---: |
| mslp | 131.5210 | 132.4036 | -0.8825 |
| q500 | 6.754811e-04 | 6.763560e-04 | -8.749282e-07 |
| q850 | 1.170288e-03 | 1.170731e-03 | -4.431675e-07 |
| t2m | 2.2546 | 2.2719 | -0.0173 |
| t500 | 1.0435 | 1.0437 | -0.0002 |
| t850 | 1.6259 | 1.6275 | -0.0016 |
| u10 | 1.2813 | 1.2821 | -0.0008 |
| u250 | 5.2580 | 5.2592 | -0.0013 |
| u500 | 3.5502 | 3.5524 | -0.0023 |
| u850 | 2.3433 | 2.3499 | -0.0066 |
| v10 | 1.4472 | 1.4509 | -0.0036 |
| v250 | 5.4218 | 5.4305 | -0.0087 |
| v500 | 3.5239 | 3.5261 | -0.0023 |
| v850 | 2.4245 | 2.4302 | -0.0057 |
| z250 | 171.0328 | 171.2483 | -0.2154 |
| z500 | 103.5825 | 103.8724 | -0.2899 |
| z850 | 79.3509 | 79.4434 | -0.0925 |

**12 h**

| variable | candidate (seed mean) | control (seed mean) | Δ candidate−control |
| --- | ---: | ---: | ---: |
| mslp | 201.8577 | 202.3004 | -0.4427 |
| q500 | 8.700827e-04 | 8.713042e-04 | -1.221548e-06 |
| q850 | 1.507830e-03 | 1.508407e-03 | -5.765270e-07 |
| t2m | 2.8417 | 2.8588 | -0.0170 |
| t500 | 1.4991 | 1.4989 | +0.0003 |
| t850 | 2.2032 | 2.2064 | -0.0031 |
| u10 | 1.5766 | 1.5782 | -0.0016 |
| u250 | 7.5614 | 7.5635 | -0.0021 |
| u500 | 4.8004 | 4.8020 | -0.0016 |
| u850 | 3.0778 | 3.0849 | -0.0071 |
| v10 | 1.8663 | 1.8715 | -0.0052 |
| v250 | 7.3168 | 7.3323 | -0.0155 |
| v500 | 4.4996 | 4.5031 | -0.0035 |
| v850 | 3.2208 | 3.2269 | -0.0061 |
| z250 | 297.9980 | 298.3256 | -0.3276 |
| z500 | 176.9315 | 177.3352 | -0.4038 |
| z850 | 124.5272 | 124.4208 | +0.1065 |

**24 h**

| variable | candidate (seed mean) | control (seed mean) | Δ candidate−control |
| --- | ---: | ---: | ---: |
| mslp | 295.7752 | 296.2309 | -0.4557 |
| q500 | 1.008139e-03 | 1.008168e-03 | -2.903147e-08 |
| q850 | 1.724044e-03 | 1.723895e-03 | +1.495545e-07 |
| t2m | 2.6228 | 2.6190 | +0.0038 |
| t500 | 2.0449 | 2.0419 | +0.0029 |
| t850 | 2.5417 | 2.5469 | -0.0052 |
| u10 | 1.8128 | 1.8145 | -0.0017 |
| u250 | 9.8055 | 9.8146 | -0.0091 |
| u500 | 5.7536 | 5.7545 | -0.0009 |
| u850 | 3.6376 | 3.6424 | -0.0048 |
| v10 | 2.2143 | 2.2188 | -0.0045 |
| v250 | 9.0386 | 9.0422 | -0.0036 |
| v500 | 5.4010 | 5.3998 | +0.0012 |
| v850 | 4.0103 | 4.0082 | +0.0021 |
| z250 | 494.7187 | 494.8424 | -0.1237 |
| z500 | 286.9341 | 287.1740 | -0.2399 |
| z850 | 194.2216 | 194.3429 | -0.1213 |

**48 h**

| variable | candidate (seed mean) | control (seed mean) | Δ candidate−control |
| --- | ---: | ---: | ---: |
| mslp | 398.1452 | 398.5409 | -0.3957 |
| q500 | 1.141173e-03 | 1.139697e-03 | +1.475728e-06 |
| q850 | 2.016008e-03 | 2.012825e-03 | +3.182198e-06 |
| t2m | 3.2413 | 3.2381 | +0.0032 |
| t500 | 2.6638 | 2.6573 | +0.0065 |
| t850 | 3.3966 | 3.4017 | -0.0051 |
| u10 | 2.0075 | 2.0078 | -0.0004 |
| u250 | 11.7858 | 11.7937 | -0.0079 |
| u500 | 6.4794 | 6.4755 | +0.0039 |
| u850 | 4.0816 | 4.0854 | -0.0038 |
| v10 | 2.5918 | 2.5941 | -0.0023 |
| v250 | 10.6393 | 10.6223 | +0.0170 |
| v500 | 6.2434 | 6.2383 | +0.0051 |
| v850 | 4.6962 | 4.6897 | +0.0066 |
| z250 | 734.0865 | 733.2348 | +0.8517 |
| z500 | 400.9097 | 400.6820 | +0.2277 |
| z850 | 258.4448 | 258.5350 | -0.0902 |

**72 h**

| variable | candidate (seed mean) | control (seed mean) | Δ candidate−control |
| --- | ---: | ---: | ---: |
| mslp | 444.4992 | 445.3432 | -0.8441 |
| q500 | 1.172336e-03 | 1.169905e-03 | +2.430719e-06 |
| q850 | 2.122261e-03 | 2.117183e-03 | +5.077578e-06 |
| t2m | 3.6395 | 3.6383 | +0.0013 |
| t500 | 3.0363 | 3.0320 | +0.0044 |
| t850 | 3.8897 | 3.8924 | -0.0027 |
| u10 | 2.0612 | 2.0608 | +0.0005 |
| u250 | 12.9952 | 12.9831 | +0.0121 |
| u500 | 6.8743 | 6.8620 | +0.0123 |
| u850 | 4.2351 | 4.2355 | -0.0004 |
| v10 | 2.7329 | 2.7317 | +0.0012 |
| v250 | 11.2978 | 11.2830 | +0.0148 |
| v500 | 6.5433 | 6.5390 | +0.0043 |
| v850 | 4.8788 | 4.8695 | +0.0093 |
| z250 | 865.0506 | 863.6766 | +1.3739 |
| z500 | 454.4960 | 454.0113 | +0.4847 |
| z850 | 277.5058 | 277.6708 | -0.1650 |

（逐 seed 的候选/控制/剂量 RMSE、skill 与全部 17 变量逐 seed 值在
`outputs/r7_s3_draft_supervision_organization_20261018_attempt01/readings.json`；
表 D 的 seed 均值只用于本页阅读，判定仍用逐 seed 值。）
单位取自 `boundary_rmse.csv` 的 `unit` 列（t2m/t850/t500 为 K、u/v 为 m s⁻¹、mslp 为 Pa、
z 为 m² s⁻²、q 为 kg kg⁻¹）；不跨单位平均。

## 2. 身份与协议

| 项 | 值 |
| --- | --- |
| attempt 目录 | `outputs/r7_s3_draft_supervision_organization_20261018_attempt01/` |
| 协议 digest（`protocol.json`，最终） | `43a61f3a82d6a550d14e9a0f67a081ac0654946958f09ae8963ff22a3d4a975e` |
| 准备/预备 digest（`preparation_protocol.json`） | `06a15d1c7159ddd9b6e23bc64c0f48a822ba1699cb76b6994ab184a97c98d806`（文件 SHA256 `35c5157d7c3f1a8ddc5c2aa37e109ad7f7d1d323c48959da81d13e16b07ebf31`） |
| `protocol.json` 文件 SHA256 | `97487f4248b452708f4ab20b1eac804ebb045979df0709c965eb510c0fb7528d` |
| 归档 `code.zip` SHA256 | `52972e97aa2740d7229b589dbca0900c0054dde3285067103f34afd76df095a2`（1434 条目） |
| ZIP comment / 代码 commit | `95aa630a020a4b947e9ef5f868cb6ef62def1fb5` |
| `model_code_sha256` | `3fe747b28e7d0ef53febc07739f210f103514a73fd19bfba6643670db65ffeb1`（与 H-D `3fe747b2…` 逐字相同；本轮未改模型） |
| `training_code_sha256` | `ce94cc8b802bec8782d3c392b7c193e10e7df5b0416668a420a7a1d2ef39198f`（与 H-D `8ad77d93…` 不同；本轮改了训练损失实现） |
| `source_sha256` | `6bc9a1a2d93d345ce0dcb39d4f3a26f175ea3bcc15082ac4931a179ca13b7701` |
| train / val data identity | `2360d42b7393cd6f53cd6f48961abc7bcd62948d3c2fa69b2d810d798d137c12` / `f590883e7e1413e68cf566b5ba002b737ac35a49878250d093eda3b4c65de933` |
| 执行文件身份 | 191 个（`execution_files_sha256` 的键数） |
| 监督块（冻结） | `interior_32`；`spatial_mask`、`[129,129]`、4225 点、`mask_sha256=4a2a576b…`（与对照臂相同） |
| 父（迁移 v3-BD 1600，逐 seed pin） | 41 `f9f12e…` / 42 `8d4e05…` / 43 `5af646…`（`outputs/r7_s3_rollout_dose_parent_migrated_20261015_attempt04/`） |
| 设备（决策 0047 双卡） | `device_mode=dual`；`gpu_uuids=[GPU-408ad137…, GPU-9d1624af…]`；`neighbour_reserve_bytes=6 GiB`；`estimated_peak_bytes=9.5 GiB`；margin 0.5 GiB |
| 预算 | planned 43200 s / hard 86400 s / per-seed 21600 s（**soft/hard overrun 均 0**） |
| cohort pin | 472 / 468 / 460 / 444 / 428（逐 lead，三 seed 一致） |
| 外部参考 A | 2400 剂量臂（`record:s3-wide-interior-dose`，pinned，**未重训练**） |
| test | `test.jsonl` 从不读取；`test_read=false` |
| 实验 commit 的 CI | `R7 CPU CI` run `39143818022` completed/success（commit `95aa630a020a4b947e9ef5f868cb6ef62def1fb5`；另 17 条标签 workflow `skipped` 是设计行为、非实验 PASS） |

## 3. 训练与评分回执

| seed | checkpoint SHA256 | training report SHA256 | 训练 s | 五 lead 评分 s 合计 | 峰值 reserved | 父 checkpoint SHA256 |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 41 | `46f246cf7ec92c7c1ef228b6169513b6487ef7adae31df1261f0496df8b5f8be` | `de94ce54dc2bee21cea295a6235bddedbd699652938fb0724ab7a5d9516760b4` | 5878.347 | 1319.802 | 8973713408 B | `f9f12e…` |
| 42 | `10e7e16f6fc9193cf5406cf6d443404e620b293c7f507e7c509bf1e9b9eba11b` | `4abb3b78b9786f39f8c75b5897c0b17c8e1b69ed55eca3fab907b5e9090e1c55` | 6096.998 | 1040.882 | 8973713408 B | `8d4e05…` |
| 43 | `33c585f4107db33302724b7b6caacc24ac2eabf420bb8b339821261f4b6c0187` | `693fa9d028d3be8f4c9ed7d284164753cf264c7b5bf21d76c6abfd1fa34dc0fd` | 4811.156 | 901.103 | 8973713408 B | `5af646…` |

- 三个训练都到 `total_updates=2400`、`resumed_from_updates=0`、`updates_this_run=2400`、`selected_update=2400`；
  每 seed 每 lead `n_evaluated == n_available_windows` 且等于冻结 cohort（472/468/460/444/428）；
  每 seed 的 `supervision` 与冻结 `mask_sha256` 逐字相同（驱动断言）。
- 每个 seed 的 `draft_supervision=final_only`、`objective=final_draft_latitude_area_mse`、
  `internal_deep_supervision=false`、`physical_weights=(1,.5,0,.5,0,0,0,.5,0,0,0,.5)`；
  `internal_k_detach=false`、`physical_step_detach=false`（full-BPTT，模型前向未改）。
- 训练窗口：三 seed 一致 `input_windows=2360`、`usable_windows=2140`、`excluded_sample_ids` 220 个
  （父序列边界样本）；三 seed 使用同一父、同一数据、同一预算、同一评分域。
- 逐 seed 回执：`seed41_receipt.json`、`seed42_receipt.json`、`seed43_receipt.json`；
  读数汇总：`readings.json`。

## 4. 逐 lead 读数

### 4.1 候选 vs 配对控制（表 A）

见 §1 表 A：逐 seed 的候选/控制 `interior_32` t2m RMSE、Δ 与候选 skill 覆盖全部五个 lead。
判定只用逐 seed 值；seed 均值仅用于阅读：
候选 RMSE `2.254618/2.841726/2.622801/3.241287/3.639550 K`，
控制 RMSE `2.271912/2.858770/2.618973/3.238121/3.638264 K`，
Δ 均值 `−0.017295/−0.017043/+0.003828/+0.003166/+0.001286 K`。
候选 skill 均值 `0.589032/0.336539/0.417753/0.065361/−0.203616`；
控制 skill 均值 `0.582700/0.328557/0.419468/0.067267/−0.202684`。

### 4.2 桥一致性（表 B）

控制 N 与已登记剂量臂 A 的差在 15/15 个 seed×lead 上落在
`−4.787001e-6…+1.778038e-5 K`，即 anchor 轮已报告的 `−4.8e-6…+1.8e-5 K` 带内；
`identity_unresolved=false`。控制臂在行为上复现了注册剂量臂。

### 4.3 候选 vs 已登记剂量臂 A（表 C）

见 §1 表 C。候选−A 的 seed 均值
`−0.017295/−0.017043/+0.003830/+0.003171/+0.001292 K`。

### 4.4 全 17 变量（表 D）

见 §1 表 D。所有 17 个变量 × 五个 lead 都按各自单位报告，不跨单位平均；
判定与候选状态不使用表 D 的 seed 均值替代逐 seed 值。

## 5. 逐阶段回执与成本

| stage | status | exit | elapsed s |
| --- | --- | ---: | ---: |
| prepare | success | 0 | 428.882 |
| archive | success | 0 | 423.592 |
| seed41 | success | 0 | 8541.780 |
| seed42 | success | 0 | 8426.108 |
| seed43 | success | 0 | 7101.340 |
| reading | success | 0 | 468.137 |

- **GPU-h 口径** = attempt 下全部 6 个 `*_process.json` 的 `elapsed_seconds` 之和 ÷ 3600 =
  **7.052733205904207 GPU-h**（prepare 0.119134 + archive 0.117665 + seed41 2.372717 +
  seed42 2.340586 + seed43 1.972594 + reading 0.130038）。三 seed 在不同时间窗口依次执行；
  该口径按串行时间相加，不把并发时间重复折算。
- **墙钟**：attempt 内最早产物 `preparation_protocol.json` 的 mtime（2026-10-19T00:25:28）到
  最后产物 `readings.json` 的 mtime（2026-10-19T07:28:39）= **25,391.449 s ≈ 7 h 3 min**。
  这是文件 mtime 导出的墙钟，不是进程内计时器。
- **soft/hard overrun 均 0**；网络 0（只读本地已登记 store，无下载阶段）；磁盘
  **14,436,984,411 B = 13.44 GiB**（attempt 目录；code 输入归档另 10,832,451 B）。
- 六阶段全部 `status=success`、`exit_code=0`（逐阶段回执见 `*_process.json`；回执未记录信号字段）。
- 协议 `estimated_peak_bytes=10,200,547,328 B = 9.5 GiB`，实测峰值
  **8,973,713,408 B = 8.3579 GiB**（本轮**高估**约 1.142 GiB，余量门保守）。
- 三次 seed 的 spawn 前只读 GPU 余量门全部 `passed=true`（`free_bytes` 18,516,803,584 /
  18,516,803,584 / 25,257,050,112 B，`required_bytes` 10,737,418,240 B）。

## 6. 公平性守卫

1. **单一因子**：只改 K 轴草稿的监督组织；物理权重、输入区域、监督域、updates、K/depth/dim、
   父、LR/warmup/FP32/batch/clip、seed、评分域与气候态分母逐字不变。两侧 K 轴权重都归一化为 sum=1：
   控制 `[0.133333,0.166667,0.2,0.233333,0.266667]`（非最终 73.333%），候选 `[0,0,0,0,1]`。
2. **模型未改**：`model_code_sha256=3fe747b2…` 与 H-D 逐字相同；`training_code_sha256=ce94cc8b…` 变化
   来自训练损失实现（`draft_supervision` 关键字与 final-only 分支），不是模型改动；
   父 checkpoint 三 seed 与 H-D 逐字相同（attempt04），无新迁移。
3. **数据/评分不变**：同一宽 store、同一 train/val identity、同一 `interior_32` mask、同一 cohort pin、
   同一气候态分母；2023 test 从未读取。
4. **桥一致性**：控制 N 与已登记剂量臂 A 的 15 个 seed×lead 全部在 `−4.8e-6…+1.8e-5 K`；
   `identity_unresolved=false`。
5. **控制/参照不重训**：N 与 A 都是 pinned 复用；只有候选 F 在本协议内训练。F−N 是同协议同 seed
   配对；F/N vs A 是跨协议 pinned 参照。
6. **无跨 attempt 复用**：attempt01 首次尝试，三 seed 全部 `resumed_from_updates=0` 重新训练；
   未使用任何复用白名单。
7. **实验代码 CI**：实验 commit `95aa630a…` 的 `R7 CPU CI` run `39143818022` completed/success
   （本次登记时经 GitHub API 只读查询确认；标签 workflow `skipped` 是设计行为）。

## 7. 已确认与推测

- **已确认（实跑/实读）**：三 seed 2400 updates 与五 lead 评分完成、cohort 逐 lead 一致、
  checkpoint/父/data digest 齐全；表 A–D 的 RMSE/skill/Δ；桥一致性；预声明五个 flag 的真值
  （a=false、b=false、c=true、d=false、e=true）；成本与峰值；test 未读；无跨 attempt 复用。
- **推测（未由本轮分离）**：
  - 6/12 h 的配对改善与 24/48/72 h 的轻微退化分别由「移除 4 个非最终直接项」还是
    「最终项归一化权重 0.2667→1.0」造成，本轮**没有分离**；
  - 72 h 残余负 skill 的主要限制是优化/超参数、信息还是结构，本轮没有分离；
  - seed 41/42 与 seed 43 在 24/48/72 h 上的符号差异是 seed 噪声还是父/数据月份构成差异，
    三 seed 不足以判定。

## 8. limitations

- 开发筛选，单一宽实例（129×129 输入、`interior_32` 监督、单 ROI、17 通道、2017–2021 train / 2022 val）；
  无显著性、收敛、SOTA 或泛化声明。
- final-only 必然同时移除 4 个非最终直接项**并**把最终项归一化权重从 0.2667 提到 1.0；
  两侧权重都归一化为 sum=1、总损失尺度可比，但本轮把「组织」作为单一因子整体检验，
  **不声称分离 support 与 emphasis**；若后续要机制归因，需要匹配最终权重的对照。
- 控制 N 与外部参照 A 都是 pinned 已登记臂，不在本协议内重训练；F−N 是同协议配对，
  F/N vs A 是跨协议。
- 三 seed 是一致性证据，不是显著性检验；GPU 共驻。
- 候选继续迁移 attempt04；父是在深监督下训练的，故本轮价格的是「从该父出发的路线」，
  不是从零重训。
- `interior_32` 监督在训练中不约束外圈；只在 `interior_32` 评分。
- 气候态是 4 个月季节均值（train/val 只覆盖 Jan/Apr/Jul/Oct 的 16 个 month-hour 桶），
  不是 12 个月年循环。
- val-only；2023 test 在后续预注册读取前保持封存。
- 全 17 变量只按各自单位报告，不跨单位平均；表 D 的 seed 均值不替代逐 seed 判定。
- GPU 训练不逐位可复现；可复现等级为 **protocol-level replayable**
  （`protocol_sha256` + `code.zip` SHA + 逐 seed 回执 + `readings.json` 可重放同一协议），
  不是逐位重现同一结果。
- 索引记录的 `evidence_commit` 绑定当前已存在的实验 commit `95aa630a…`；
  本页与 index 的登记提交由后续 `commit-register` 生成（登记 commit 的 SHA 只能在提交后回填）。
- 未读取 `test.jsonl`、未消耗 r、未启动 S4；不宣布科学通过、不称泛化/SOTA。

## 9. 下一动作（S3）

- K 草稿监督组织支已按预声明**永久关闭**：不扫中间草稿权重、不加 seed、不调 2400 updates、
  不改 partial/加权 final-only、不与 anchor/feedback/architecture/容量组合、不换名重开。
- 下一根独立的**训练信号 / 目标组织**轴按长文 §8 为 **K 步草稿的单调改进约束**
  （要求后一草稿的 loss 不差于前一草稿的 hinge；不增容量、不加数据、不改模型前向）；
  先写新的 round brief、先检索一手资料，再冻结新协议。数据扩充按 D3 继续暂缓。
- test 未读、r=0、S4 未启动；不宣布科学通过、不称泛化/SOTA。
