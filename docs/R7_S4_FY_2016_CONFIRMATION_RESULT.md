# S4：FY candidate 在完整未见年份 2016 上的首次确认（Track 0 / D3）

> **状态：读数已出、判决未宣。** `scientific_claim: false`。
> 协议先冻结（`protocol_sha256 = 0dd100dcb4436a848bcd1b4fa91fdd8709f2a99a03e686b750b765572be2db1c`），
> **r = 1、`alpha_1` = 0.025 已在评分前登记**。实例机制见**决策 0059**；
> 协议人读版见 `docs/R7_S4_FY_2016_CONFIRMATION_PROTOCOL.md`。

## 1. 实例与身份

- **目标年 2016**：完整未见年份（1,464 帧、6 h 步长），四季齐全 ⇒ 满足
  `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md:111` 的覆盖门。
- **病例数**：**1451**（全部 5 lead 共享同一病例集；四季组 DJF 351、MAM 368、JJA 368、SON 364）。
- **帧来源**：多年 store `outputs/r7_s3_fy_multiyear_20261023_attempt01/store/cache.zarr`（只读）。
- **归一化**：FY 2017–2021（mean `e15c3f26…` / std `afc998a3…`），按 sha256 钉住后显式施加。
- **气候态分母**：FY 的 48 桶 train-only（2017–2021），`n_selected_steps = 7,304`，按 reference 声明核对。
- **身份门（照真校验）**：6/6 份 checkpoint 的 `contract["data_identity"]` 等于
  `dataset_identity(FY 该臂 train.jsonl)`，全部通过（treatment `c17494ab…`、control `0469312978…`）。
- **非有限读数**：0 个。

## 2. 主读数（t2m，`full`，物理 K）

| 分组 | 臂 | 6 h | 12 h | 24 h | 48 h | 72 h |
| --- | --- | --- | --- | --- | --- | --- |
| total set | treatment | 2.1347 | 2.6752 | 2.6227 | 3.4201 | 3.8223 |
| total set | control | 2.1524 | 2.6957 | 2.6583 | 3.4970 | 3.9503 |
| DJF | treatment | 2.3282 | 2.7988 | 2.6300 | 3.5635 | 4.0351 |
| DJF | control | 2.3726 | 2.8456 | 2.7021 | 3.6699 | 4.1596 |
| MAM | treatment | 2.2660 | 2.9259 | 3.0254 | 3.8393 | 4.2346 |
| MAM | control | 2.2851 | 2.9463 | 3.0188 | 3.8810 | 4.3509 |
| JJA | treatment | 1.8759 | 2.3752 | 2.1712 | 2.7264 | 2.9547 |
| JJA | control | 1.8565 | 2.3531 | 2.1665 | 2.7665 | 3.0395 |
| SON | treatment | 2.0469 | 2.5724 | 2.5942 | 3.4593 | 3.9477 |
| SON | control | 2.0679 | 2.6047 | 2.6779 | 3.5771 | 4.1285 |

**t2m skill = 1 − mean MSE_model / mean MSE_climatology**：

| 分组 | 臂 | 6 h |  | 12 h |  | 24 h |  | 48 h |  | 72 h |
| --- | --- | --- | --- | --- | --- | --- |
| total set | treatment | 0.7028 | 0.5330 | 0.5508 | 0.2330 | 0.0384 |
| total set | control | 0.6978 | 0.5259 | 0.5385 | 0.1981 | -0.0271 |
| DJF | treatment | 0.7135 | 0.5847 | 0.6316 | 0.3056 | 0.0834 |
| DJF | control | 0.7025 | 0.5707 | 0.6111 | 0.2635 | 0.0259 |
| MAM | treatment | 0.6824 | 0.4715 | 0.4360 | 0.1024 | -0.0831 |
| MAM | control | 0.6771 | 0.4641 | 0.4385 | 0.0828 | -0.1435 |
| JJA | treatment | 0.5783 | 0.3236 | 0.4323 | 0.0893 | -0.0935 |
| JJA | control | 0.5869 | 0.3361 | 0.4347 | 0.0623 | -0.1571 |
| SON | treatment | 0.7684 | 0.6343 | 0.6290 | 0.3462 | 0.1634 |
| SON | control | 0.7636 | 0.6251 | 0.6046 | 0.3008 | 0.0851 |

## 3. 相对同数据 incumbent 的变化（candidate / control − 1，MSE）

| 变量 | 6 h | 12 h | 24 h | 48 h | 72 h |
| --- | --- | --- | --- | --- | --- |
| t2m | -0.01640 | -0.01513 | -0.02665 | -0.04349 | -0.06376 |
| u10 | -0.00335 | -0.00482 | -0.00779 | -0.00858 | -0.01399 |
| v10 | -0.00885 | -0.01488 | -0.01496 | -0.00452 | -0.01468 |
| mslp | -0.02004 | -0.03337 | -0.04734 | -0.05560 | -0.07305 |

## 4. 全部 17 变量的 RMSE（treatment，总集，`full`）

| 变量 | 6 h | 12 h | 24 h | 48 h | 72 h |
| --- | --- | --- | --- | --- | --- |
| t2m | 2.1347 | 2.6752 | 2.6227 | 3.4201 | 3.8223 |
| u10 | 1.2906 | 1.6307 | 1.9033 | 2.0966 | 2.1449 |
| v10 | 1.5000 | 1.9510 | 2.3449 | 2.6769 | 2.7555 |
| mslp | 144.9623 | 237.7164 | 370.2505 | 508.7464 | 561.7549 |
| z850 | 92.3722 | 151.4652 | 240.1106 | 319.3454 | 339.6359 |
| t850 | 1.5951 | 2.2440 | 2.7876 | 3.8235 | 4.2669 |
| q850 | 0.0011 | 0.0015 | 0.0018 | 0.0022 | 0.0024 |
| u850 | 2.3748 | 3.1778 | 3.8603 | 4.3809 | 4.5270 |
| v850 | 2.5698 | 3.4858 | 4.3750 | 5.0861 | 5.2082 |
| z500 | 120.2181 | 208.2864 | 335.4905 | 460.6724 | 502.3620 |
| t500 | 1.1422 | 1.6965 | 2.3558 | 3.0740 | 3.4141 |
| q500 | 0.0006 | 0.0008 | 0.0010 | 0.0011 | 0.0012 |
| u500 | 3.7747 | 5.1957 | 6.4414 | 7.2869 | 7.5834 |
| v500 | 3.7214 | 4.7614 | 5.6662 | 6.4549 | 6.6868 |
| z250 | 193.1625 | 339.8747 | 558.3652 | 821.0537 | 947.8076 |
| u250 | 5.4914 | 7.8324 | 10.1750 | 12.2828 | 13.1969 |
| v250 | 5.8242 | 7.8677 | 9.5503 | 11.0724 | 11.6894 |

`interior` / `edge`（t2m，treatment，总集）：

| 区域 | 6 h | 12 h | 24 h | 48 h | 72 h |
| --- | --- | --- | --- | --- | --- |
| interior | 2.1059 | 2.6364 | 2.6099 | 3.4043 | 3.8034 |
| edge | 2.5415 | 3.2187 | 2.8141 | 3.6571 | 4.1054 |

## 5. D4 同时区间（`alpha = 0.025`、block = 14、resamples = 4,000、seed 20160101）

primary family（t2m，per-case skill 的 seed 均值）按**分组**实现——
`training/r7_simultaneous_stats.simultaneous_paired_interval` 要求同一 family 的所有 cell
共享同一组 case key，而四季组是对 2016 病例的划分，故「总集 + 各年 + 四季组同一 family」不可表达。

| 分组 | 临界值 | 下界 ≤0 的 cell |
| --- | --- | --- |
| total set | 2.626 | 48h|t2m, 72h|t2m |
| DJF | 2.738 | 48h|t2m, 72h|t2m |
| MAM | 2.658 | 24h|t2m, 48h|t2m, 72h|t2m |
| JJA | 2.747 | 48h|t2m, 72h|t2m |
| SON | 2.640 | 48h|t2m, 72h|t2m |

**⚠ estimator 歧义（如实记录，判决因此不宣）**：协议 §4 把 primary 定义在
**分组均值之比** `skill = 1 − mean MSE_model / mean MSE_clim` 上；而冻结的统计工具算的是
**per-case 序列的均值**及其区间。两者是不同的量：
- total set 48 h：分组比值 skill = **0.2330**（>0），而 per-case 均值 skill 的同时下界 = **-0.2670**（≤0）。
- total set 72 h：分组比值 skill = **0.0384**（>0），而 per-case 均值 skill 的同时下界 = **-0.5230**（≤0）。
- SON 48 h：分组比值 skill = **0.3462**（>0），而 per-case 均值 skill 的同时下界 = **-0.4592**（≤0）。
- SON 72 h：分组比值 skill = **0.1634**（>0），而 per-case 均值 skill 的同时下界 = **-0.7681**（≤0）。

⇒ **本条不宣布 primary 门通过或不通过**：判据写的是比值 skill，工具给的是 per-case 均值区间，
二者不一致；要出判决必须先**预注册**用哪一个估计量，并（若选比值）实现比值感知的块 bootstrap。
这不是「放宽判据」，是把一个此前没被点名的估计量歧义留在门前。

**守门（u10/v10/mslp 对同数据 incumbent）**：⚠ **本节于 2026-10-28 更正过（见 §5.1）**。
协议声明的守门覆盖 **6 个分组 × 3 变量 × 5 lead = 90 个 cell**；
原文只报了 total set 这一族的 15 格却写成"全部"，并把计数写错。更正后的实测：

| 口径 | cell 数 | 点估计 ≤0 | 同时区间上界 ≤0 |
| --- | --- | --- | --- |
| total set 单族 | 15 | 15/15 | **5/15**（6h/12h/24h 的 mslp + 6h/12h 的 v10） |
| **协议声明的全部** | **90** | **78/90**（**12 个点估计 >0**） | **25/90** |

**12 个正的点估计**集中在 DJF/JJA/SON 的 u10/v10（例如 DJF 48h|v10 = **+0.02909**、
DJF 72h|u10 = **+0.01598**、JJA 24h|u10 = **+0.00367**）。
⇒ 按协议字面（「每 seed ≤0 且同时区间上界 ≤0」）**守门未达**，且比原文记录的**更差**：
不只是区间上界未过，连点估计都不是全 ≤0。如实记录，不放宽。

**另有一个未覆盖的子句（实质性缺口）**：协议要求「**per seed** ≤ 0」，
但 `readings.json` 只保存了 seed 平均后的 per-case 序列，**没有逐 seed 值** ⇒
该子句**在本读数上无法评估**（不是"通过"，也不是"未通过"，是**未测**）。
原文没有点名这一点，同样是漏报。

## 5.1 更正记录（2026-10-28）

本节记录本项目**自己**在证据页上的一处事实错误，以及两处漏报。更正不改变任何
冻结判据、不重算任何读数 —— `readings.json` 未变（sha256 `dd3a0709…` 仍成立），
只是此前对它的**读法**写错了。

| # | 原文 | 事实 | 性质 |
| --- | --- | --- | --- |
| 1 | "15 个 cell 的点估计全部 ≤0" | 协议声明的守门是 **90** 个 cell；原文只读 total set 一族（15 格）却写成全部 | 口径漏报 |
| 2 | "同时区间上界 ≤0 的只有 **-50/15**" | 计数是乱的；total set 单族实为 **5/15**，全部 90 格为 **25/90** | 笔误 + 口径 |
| 3 | （未提） | **12/90 点估计 >0**（DJF/JJA/SON 的 u10/v10）⇒ "全部 ≤0" 为假 | 结论方向错 |
| 4 | （未提） | 协议要求"**per seed** ≤0"，但读数只存 seed 平均序列 ⇒ 该子句**未测** | 未覆盖子句 |

**更正后的守门结论**：仍然**未达**，且理由比原文更强（不只是区间上界，连点估计都不是全 ≤0）。
**判决仍不宣**，与 §5 的估计量歧义无关 —— 即便估计量问题解决了，守门这一关也过不了。

**这处错误的教益**（写进后续轮的要求）：读数文件同时保存了 6 个分组族，报告时必须
按**协议声明的 cell 全集**统计，不能只引用最显眼的那个族；`readings.json` 应当**同时保存
逐 seed 值**，否则「per seed ≤0」这类子句永远无法评估。R2–R5 的驱动按这两条写。

## 6. 成本与复现

- 墙钟 **2 h 03 m 26 s**（01:36:04 → 03:39:30），单卡，**GPU-h = 2.0572**。
- 复现：`.venv/bin/python scripts/study_r7_s4_fy_2016_confirmation.py --protocol outputs/r7_s4_fy_2016_20261026_attempt01/protocol.json --out <新目录> --device cuda`。
- `readings.json` sha256 `dd3a0709d0aed69cc1b37f7897616d1486c7aba4f4f882e839c54881d0221777`。

## 7. limitations

- the climatology denominator is fitted on 2017-2021 while the target period is 2016: denominator period and test period are different eras (ERA5 has a trend), so t2m skill here is not the same quantity the FY round measured; the u10/v10/mslp comparison is against the same-data incumbent and is not affected
- per decision 0059 the store split-year ownership clause of validate_record is the only validation clause skipped; all other integrity clauses (index bounds, timestamp identity, init_time, lead, cadence) are kept
- the 2016 frames come from the pinned multiyear store, not a purpose-built 2016 store
- scientific_claim: false; this is a first confirmation readout, not a publication-grade claim
- primary 门的**估计量歧义未决**，因此本条**不宣布**通过/不通过。
- 守门按协议字面**未达**（全 6 族 90 格：点估计 78/90 ≤0、区间上界 25/90 ≤0）；见 §5.1 的更正记录。
- 守门的"per seed ≤ 0"子句**未测**（读数只存 seed 平均序列）。
- 这是首次确认读数，不是发布级科学认证；不称泛化/SOTA；不自行宣布最终 goal 完成。
