# #69 第二阶段：双月段（M2）与桶数扩展

**状态：已完成。** 本文件报告在 **M2 双月段**（2016-01-01..2016-02-29，240 时次连续）
上的单一实验：5 个臂 × 3 个 seed 在 800 更新预算下的自由 rollout 评估，
以及第一次在**八个** `(month,hour)` 桶上重算的零参数 climatology 基线。

本阶段要回答的问题是 objective 指定的：「climatology 在 t2m 上占优」是**方法无效**
还是**数据范围太窄**的产物。前置缺陷（跨单位比较）与更正见决策 0010；
本文件只报告修好之后、在更宽数据上得到的结论。

| 项 | 值 |
| --- | --- |
| 起点 / 终点 SHA | `0e09800` → `9a4e742`（三个提交，见 §12） |
| 段范围 | `2016-01-01T00:00` .. `2016-02-29T18:00`（60 天 / 240 时次，连续无缺） |
| 源快照 | Earthmover `ZFKDHBCTBVHVXM3BQFV0`（匿名公开 S3，无新增凭据） |
| 合并源 SHA256 | `496084a9260bacfaf6293a01d89439c1e49d6afa8f09bc1f51d89a1d1f9bda21`（37,734,176 B） |
| 实测网络字节 | **7,252,462,353 B = 6.754 GiB**（4 个成功 part 各自 `/proc/net/dev` delta 之和） |
| 墙钟 | 下载 2500.7 s（41.7 min，**4 个成功 part** 的 `elapsed_seconds` 之和）；另有一次按 30 min 时限失败（1800.4 s，未产出数据）；训练 4338.0 s |
| 新产物 | **3.588 GiB**（用**第二阶段档位** ≤100 GiB；decoded ≤256 GiB） |
| 测试 / 门禁 | 见 §7 |
| 协议 digest | `e068bea4d5c0e64f12d3aee0ec6ca2e352a5adb84c592ae32696c99137af9d02`（3 个 seed 全部相同） |
| `data_identity` | `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07` |
| 实测 GPU 训练 | **1.204 GPU-h**（15 份 `training_report.json` 的 `elapsed_seconds` 求和） |

---

## 1. `(month,hour)` 桶数：4 → 8（实测，非断言）

`fit_training_climatology` 在 M2 store 上的实际输出：

```
kind            = train-only-month-hour-grid-mean-v1
selection       = declared_train_time_ranges
n_selected_steps= 188
bucket counts   = 01-00:31  01-06:31  01-12:31  01-18:31
                  02-00:16  02-06:16  02-12:16  02-18:16      <= 8 个桶
```

对比：D1 / B2 / 1 月重切全部只有 **4** 个桶（`(1,00/06/12/18)`）。

**关键分离**：2 月的桶由 `02-01..02-16` 拟合，而被打分的块是 `02-23..02-29`——
比拟合窗口晚 **6 天**。所以基线不再是「同一天的均值」。
但这仍**不是**跨季节或跨年证据：两个月同属一个冬季、一个区域、一年。

## 2. 主结果：climatology 在 t2m 上**仍然压倒**神经臂

test 块 `[02-23, 03-01)`，物理单位，latitude 加权，每 seed 每变量一格：

| lead | unet | native_window | afno_small | generic | process | climatology（0 参数） |
| --- | --- | --- | --- | --- | --- | --- |
| 6h | 4.899 | 4.617 | 4.612 | 5.012 | 5.169 | **3.857** |
| 12h | 5.677 | 5.546 | 5.631 | 6.647 | 6.875 | **3.908** |
| 24h | **4.328** | 6.569 | 5.690 | 6.601 | 6.291 | 4.010 |
| 48h | **4.858** | 7.447 | 6.176 | 7.649 | 7.287 | 4.321 |
| 72h | **5.229** | 6.952 | 6.220 | 7.839 | 7.580 | 4.546 |

**t2m 逐 (arm, seed, lead) 全部 75 个单元**：climatology 胜 **74**，唯一例外是
unet seed43 在 24h（ratio 0.950）。**换句话说：桶数从 4 增到 8、打分块移出拟合窗口
6 天之后，t2m 上的压倒性优势依然成立，且跨 seed 一致（3/3 seed 同向）。**

全部 17 变量 × 5 时效 × 5 臂 × 3 seed = **1275 个单元**：

| lead | better | worse | of |
| --- | --- | --- | --- |
| 6h | 240 | 15 | 255 |
| 12h | 226 | 29 | 255 |
| 24h | 155 | 100 | 255 |
| 48h | 58 | 197 | 255 |
| 72h | 26 | 229 | 255 |
| **合计** | **705** | **570** | **1275** |

即：**整体上神经臂略占多数（705/1275），但这个多数完全来自 6/12h；t2m 与 48/72h
仍系统性输给气候态。** 每 seed 的计数几乎相同（234/191、234/191、237/188），
说明这不是被某个 seed 拉动的。

**逐变量（全部 5 个时效合计，75 个单元/变量）**：

| 变量 | better | worse | 变量 | better | worse |
| --- | --- | --- | --- | --- | --- |
| t2m | **1** | **74** | u850 | 39 | 36 |
| q850 | 22 | 53 | z500 | 42 | 33 |
| u10 | 27 | 48 | mslp | 45 | 30 |
| v850 | 28 | 47 | v500 | 45 | 30 |
| u500 | 30 | 45 | v250 | 46 | 29 |
| v10 | 34 | 41 | t850 | 51 | 24 |
| z850 | 56 | 19 | z250 | 53 | 22 |
| t500 | 58 | 17 | q500 | 60 | 15 |
| u250 | 68 | 7 | | | |

## 3. 为什么 climatology 在 M2 上变弱了——归因分解（不是猜）

M2 的 6h 气候态 RMSE（3.857 K）比 1 月重切的物理值（2.601 K）**更差**。
两种可能解释：桶多了（基线变弱）或打分块离拟合日更远（两者都变难）。用同一 store 的
数组直接分解三者：

| 基线（M2 test 块，t2m） | RMSE |
| --- | --- |
| 8 桶月-小时气候态（M2 train 全用） | **3.793 K** |
| 仅 1 月的 4 桶气候态，应用到 2 月打分块 | 6.325 K |
| 单一全训练均值场（0 桶） | 6.135 K |

**读数**：4 桶基线（6.325）比 8 桶基线（3.793）**差得多**——所以桶的存在确实重要，
若退化成「只有 1 月桶」，基线会垮掉。8 桶基线比单场（6.135）好 **38%**。

同时，段的难度本身变了：test 块相对训练均值的固有离散度从 1 月的 **3.511 K**
升到 M2 的 **6.135 K**。用无量纲比值看（0 = 完美，1 = 与训练均值一样差）：

| 段 | climatology / spread | persistence RMSE |
| --- | --- | --- |
| 1 月重切（4 桶） | **0.744** | 4.963 K |
| M2（8 桶） | **0.618** | 6.755 K |

**归因**：M2 的气候态**相对**其所在段其实比 1 月**更强**（0.618 < 0.744），
变差的是**绝对** K 值，因为 2 月的天气离散度更大（冬季风暴活动）。
所以「9.09 倍」既不是方法缺陷，也不是「基线很强」——它是**在一个气候态近乎等于
记忆平均场的单月段上，用一个跨单位比值衡量出来的**。修好单位、换到 8 桶之后，
t2m 上仍然是气候态赢，但量级是 **1.2–1.4 倍**，不是 9 倍。

## 4. `9.09 倍` 的解释：成立与否

| 主张 | 判定 |
| --- | --- |
| 「9.09 倍」是可引用的数量级 | **不成立**。它是物理（分子）÷归一化（分母）的产物，实测 1275/1275 单元比值恒等于训练 std（决策 0010）。 |
| climatology 在 t2m 上占优 | **成立，且在扩到 8 桶后依然成立**（t2m 74/75 单元，3/3 seed 同向）。 |
| 该优势是「基线很强」 | **不成立**。它是「climatology 在 t2m 上占优」这一**事实**，不等于基线在预报技巧上强：t2m 的日内与季节方差绝大部分由 (month,hour) 桶本身解释。 |
| 该优势是「单季节不可外推」的产物 | **部分成立**：M2 证明优势在**两个**月、**8** 个桶上仍存在，所以不能归因于「只有 4 个桶」；但两月同属一个冬季，**跨季节仍 UNVERIFIED**。 |
| 方法（神经臂）在 t2m 上无效 | **不能区分**。6/12h 整体多数格是神经臂赢（240/255、226/255），t2m 却系统性输——这是一条**变量特异**的结论，不是「方法全面无效」。 |

## 5. process vs generic：在 M2 上仍未分离

同一 store、同一协议、同一 800 更新、共享初始化锚点（generic 的初始化复制给 process），
经 **#60 比较器**（`training/r7_coreasoning_compare`）在 **test** 上配对：

| lead | improved | worsened | unresolved | seed 一致变量数 |
| --- | --- | --- | --- | --- |
| 6h | 0 | 5 | 12 | 17/17 |
| 12h | 0 | 3 | 14 | 17/17 |
| 24h | 0 | 1 | 16 | 17/17 |
| 48h | 1 | 2 | 14 | 17/17 |
| 72h | 1 | 1 | 15 | 17/17 |
| **合计** | **2** | **12** | **71** | |

**85 个单元里 71 个 unresolved**（跨 seed 翻号），可分辨的 14 个里 12 个方向是
**generic 更好**。这与 B2 的既有结论一致（B2：73 unresolved、仅 6h 跨 seed 同号且
generic 更好）。**在 M2 上 process 结构依然没有可分离的收益。**

## 6. 训练曲线与预算

15 个 arm-seed 全部跑满声明的 **800 更新**端点，**无早停**：

| 臂 | seed41 ratio | seed42 | seed43 | 更新数 |
| --- | --- | --- | --- | --- |
| unet | 0.705 | 0.703 | 0.683 | 800/800/800 |
| native_window | 0.660 | 0.648 | 0.648 | 800/800/800 |
| afno_small | 0.546 | 0.547 | 0.547 | 800/800/800 |
| generic | 0.609 | 0.594 | 0.594 | 800/800/800 |
| process | 0.605 | 0.587 | 0.595 | 800/800/800 |

（ratio = 末个完整 epoch 均值 ÷ 首个完整 epoch 均值；M2 train 有 186 窗口，
每 epoch = 93 更新，故完整 epoch 为 9 个。）

**完整 epoch 均值曲线**（每一格为 3 个 seed 在同一 epoch 上的 `epoch_mean_loss` 均值，
9 个 epoch 全部列出，不做挑选）：

| epoch | unet | native_window | afno_small | generic | process |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.1876 | 0.1841 | 0.1792 | 0.1826 | 0.1826 |
| 1 | 0.1754 | 0.1688 | 0.1598 | 0.1646 | 0.1642 |
| 2 | 0.1644 | 0.1571 | 0.1436 | 0.1511 | 0.1503 |
| 3 | 0.1543 | 0.1482 | 0.1316 | 0.1411 | 0.1405 |
| 4 | 0.1458 | 0.1391 | 0.1203 | 0.1311 | 0.1308 |
| 5 | 0.1389 | 0.1316 | 0.1115 | 0.1228 | 0.1223 |
| 6 | 0.1339 | 0.1245 | 0.1032 | 0.1146 | 0.1139 |
| 7 | 0.1307 | 0.1200 | 0.0980 | 0.1094 | 0.1087 |
| 8 | 0.1301 | 0.1182 | 0.0950 | 0.1072 | 0.1064 |

**逐 seed 首/末完整 epoch 明细**：

| 臂 | seed41 | seed42 | seed43 |
| --- | --- | --- | --- |
| unet | 0.1874 → 0.1320 (0.705) | 0.1877 → 0.1319 (0.703) | 0.1878 → 0.1282 (0.683) |
| native_window | 0.1839 → 0.1213 (0.660) | 0.1843 → 0.1194 (0.648) | 0.1841 → 0.1194 (0.648) |
| afno_small | 0.1787 → 0.0976 (0.546) | 0.1786 → 0.0977 (0.547) | 0.1803 → 0.0987 (0.547) |
| generic | 0.1826 → 0.1113 (0.609) | 0.1822 → 0.1081 (0.594) | 0.1831 → 0.1087 (0.594) |
| process | 0.1826 → 0.1105 (0.605) | 0.1820 → 0.1068 (0.587) | 0.1831 → 0.1089 (0.595) |

**读数**：15 条曲线全部单调下降且 epoch0 的跨臂差 ≤2.5%（同一初始化尺度）；
nepoch 均为 9，无早停，所以没有「哪个臂被掐在下降中途」的混淆。
**但训练损失与 test 上输给气候态并不矛盾**：这是 +6h 单步目标上的损失，
而 test 是 6–72h 自由 rollout；`generic` 与 `process` 的曲线几乎重合
（两者在功能上只差一个投影），与 §5 的 71/85 unresolved 一致。

| 项 | 实测 |
| --- | --- |
| M2 训练 | **1.204 GPU-h**（15 臂-seed 的 `elapsed_seconds` 求和 = 4333.9 s） |
| 下载 | 41.7 min 墙钟（4 个成功 part），6.754 GiB 网络 |
| 新产物 | **3.588 GiB**（第二阶段档位 100 GiB，用掉 3.6%） |
| 第二阶段累计 GPU-h | 1.204（本阶段唯一训练） |

## 7. resume 一致性（在 M2 store 上实跑，非引用 fixture）

本阶段要求 resume 一致性被覆盖。用 `scripts/study_r7_69_resume_check.py` 在 **M2 store**
上跑三条腿：参考跑（端点到 12，每 6 更新验证一次并因此发布一个中间 checkpoint）、
从该**中间 checkpoint** 续跑的腿、以及一条同种子全新跑作为对照。比较 8 个字段
（`model` / `optimizer` / `rng` / `cursor` / `epoch` / `updates` / `model_code_sha256` / `contract`）。

| 设备 | 两条全新跑的位级噪声底 | resume 腿相对参考的最大偏差 | 判定 |
| --- | --- | --- | --- |
| **CPU** | **0.0**（逐位相同） | **0.0**（全部 8 个字段） | **bitwise consistent** |
| GPU | 1.229e-07 | 1.192e-07 | 在平台噪声底之内 |

**必须记录的一次方法论纠正**：这个检查的**第一版是我的测试设计错误**，不是代码缺陷。
第一版让「前半腿」用 `total_updates = 6` 训练、而参考腿用 `total_updates = 12`；
`warmup_cosine_factor` 是 `total_updates` 的函数，所以前者的前 6 步**不是**后者的前缀——
两条腿每一步的学习率都不同，差值 5.2e-4 完全由测试构造产生。
改为从参考跑**自己的**中间 checkpoint 续跑（schedule、验证节奏、样本顺序、contract
签名全部一致）之后，CPU 上偏差归零。**若不设对照腿，这个错误会被误报成「resume 有缺陷」。**

另外，checkpoint 一律经仓库自带的 `load_checkpoint` 读取（`weights_only=True`，
digest 不符即 fail closed），脚本不使用任意对象的反序列化。

## 8. 交付清单与验证

| 判据 | 状态 |
| --- | --- |
| 跨月连续段 ≥60 天 | ✅ 240 时次，合并时重新断言唯一、递增、严格 6 小时连续、端点等于冻结范围 |
| `(month,hour)` 桶 ≥8 | ✅ **实测 8 个**（§1），train 188 步 |
| 在新区段重算 climatology 基线 | ✅ 同一评估器内，`climatology` 臂与 17 变量逐格 |
| 重跑神经臂对照 | ✅ 5 臂 × 3 seed × 5 时效 × 17 变量 |
| held-out 支持 72h rollout | ✅ val 22 窗口、test 26 窗口（6h）；72h 为 **11 / 15**（重切段是 **0 / 7**） |
| 不变量：climatology 用 train-only | ✅ `selection = declared_train_time_ranges`，`n_selected_steps = 188` |
| 协议在 optimizer step 前冻结 | ✅ 每个 seed 先写 `protocol.json`；3 seed digest 全同 `e068bea4…` |
| 协议不得misdescribe数据 | ✅ 新增 fail-closed 守卫：非 B2 store 未提供 `segment_note`/`limitations` 即拒绝运行（回测钉住 B2 归档摘要 `244af00b…` 逐字节可复现） |
| #60 比较器为唯一口径 | ✅ process/generic 与对两控制的判定全部经它 |
| 单位不再跨比 | ✅ `rmse.csv` 与 `climatology_skill.csv` 现在同单位；比较器在这两个文件的 unit 不一致时**拒绝比较** |
| 失败留痕 | ✅ 一次按时限失败写 `part_b_receipt.json` = `failed-no-fallback`，**保留未删**，无合成替代 |
| resume 一致性 | ✅ 在 M2 store 上实跑三条腿；**CPU 逐位一致（8/8 字段偏差 0.0）**，GPU 在自测噪声底内（§7） |

## 9. 局限（如实）

| 局限 | 说明 |
| --- | --- |
| 仍是单一年、单一冬季、单一区域 | 两个月同属 2016 年 1–2 月；**跨季节、跨年、跨区域结论均为 UNVERIFIED** |
| 桶与打分块共享月份 | 8 桶中 4 个是 2 月的，打分块也在 2 月；买到的只是「比拟合窗口晚 6 天」，不是季节外样本 |
| 800 更新、非收敛 | 15/15 到达端点，但端点不是收敛点；不构成 SOTA 或收敛声明 |
| 3 seed 不是显著性检验 | seed 一致性只是一致性，不是显著性 |
| 逐格计数未做多重比较校正 | 1275 个单元按符号计数，未做 FDR；报告的是方向与一致性 |
| 只有一条 val 协议 | val 22 窗口（72h 11 个）可支撑选择，但本阶段未做选型对照 |

## 10. 复现

```bash
# 1) 只读 preflight（不写任何东西）
.venv/bin/python -m data.download.earthmover_spatial_m2 --preflight \
    --report outputs/r7_m2_segment/preflight_report.json
# 2) 分 part 下载（每 part 保持在 30 min 界内）
.venv/bin/python -m data.download.earthmover_spatial_m2 --write --offset 0   --stamps 120 \
    --out outputs/r7_m2_segment/part_a.nc --receipt outputs/r7_m2_segment/part_a_receipt.json
for off in 120 160 200; do
  .venv/bin/python -m data.download.earthmover_spatial_m2 --write --offset $off --stamps 40 \
    --out outputs/r7_m2_segment/part_$off.nc \
    --receipt outputs/r7_m2_segment/part_${off}_receipt.json
done
# 3) 合并（重断言连续性；已存在的输出会被拒绝）
.venv/bin/python -m data.download.earthmover_spatial_m2 --merge \
    --parts outputs/r7_m2_segment/part_a.nc outputs/r7_m2_segment/part_120.nc \
            outputs/r7_m2_segment/part_160.nc outputs/r7_m2_segment/part_200.nc \
    --part-receipts <same order>_receipt.json \
    --out outputs/r7_m2_segment/source.nc --receipt outputs/r7_m2_segment/source_receipt.json
# 4) 转 store（显示 --write 才写）
.venv/bin/python prepare_r7_local.py --source outputs/r7_m2_segment/source.nc \
    --config configs/r7_era5_m2_two_month.yaml --write \
    --store outputs/r7_m2_segment/store/cache.zarr \
    --manifests outputs/r7_m2_segment/store/manifests --max-raw-gib 1
# 5) 训练 5 臂 x 3 seed（协议在每个 seed 的第一步前冻结）
bash scripts/run_m2_multiseed.sh
# 6) 合并 + 冻结判据
.venv/bin/python scripts/study_r7_b2_multiseed.py --mode finalize --phase confirm \
    --manifests outputs/r7_m2_segment/store/manifests --out outputs/r7_m2_multiseed
# 7) 桶数对照（读同一次评估的两个 CSV，单位不一致即拒绝）
.venv/bin/python scripts/study_r7_69_bucket_comparison.py \
    --manifests outputs/r7_m2_segment/store/manifests \
    --train-root outputs/r7_m2_multiseed --out outputs/r7_m2_comparison
# 8) resume 一致性（CPU 上应给出逐位一致；GPU 上与自测噪声底比较）
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/study_r7_69_resume_check.py \
    --out outputs/r7_m2_resume_check_cpu --device cpu
.venv/bin/python scripts/study_r7_69_resume_check.py \
    --out outputs/r7_m2_resume_check --device cuda
```

## 11. 结论

1. **桶数扩展达成**：4 → **8**（实测）。
2. **「climatology 在 t2m 上占优」在 8 桶上依然成立**：t2m 74/75 单元、3/3 seed 同向，
   量级 1.2–1.4×（不是原报告的 9.09×）。
3. **但整体图景是时效依赖的**：6h 240/255、12h 226/255 神经臂占优；24h 约打平；
   48/72h 气候态明显更好。所以准确表述是「**t2m 与长时效上气候态仍然更强，
   短时效与其余变量上神经臂更强**」，不是「方法全面无效」。
4. **9.09 倍不可引用**（跨单位）。更正后的物理比值与逐格计数在本文件与决策 0010。
5. **process 相对 generic 在 M2 上仍不可分离**（71/85 unresolved），
   与 B2 一致；本阶段**不**据此重启动 #66，也**不**改默认策略。
6. **不更新论文包的正面结论**：新增的是负结果的**范围扩展**（8 桶、两月），
   不是新技巧。论文包应引用本文件更正后的数字，删除 9.09×。

## 12. 提交与 CI 绑定

| SHA | 内容 | ci.yml run |
| --- | --- | --- |
| `478d982` | 单位缺陷修正 + 协议不得misdescribe数据的守卫 | 无独立 run（见下） |
| `be5ce21` | M2 双月段、桶数扩展、训练与对照 | `36330894071` **completed / success** |
| `9a4e742` | resume 一致性检查 | `36332301655` **completed / success** |

### 12.1 关于「开 issue + `Closes #N` 关闭」的豁免（环境限制，非跳过）

objective ⑤ 要求「新发现的问题自行开 issue 并在 DONE 后经 `Closes #N` + main ff 关闭」。
**本条在本环境中不可执行，理由为实测**：

| 通道 | 实测结果 |
| --- | --- |
| `POST /repos/Eswink/UrbanPiDiT_R2/issues` | **HTTP 401**（匿名写被拒） |
| `GET /issues/69` | **HTTP 404**（该 issue 从未存在） |
| `gh` CLI | **未安装**；AGENTS.md 亦明令本项目不用 `gh` |
| git / SSH | 可用（本次 ff 推送即经它完成） |

`Closes #N` 只能关闭**已存在**的 issue，而创建 issue 需要的正是那条 401 的写通道。
因此本阶段**无法**开 issue、也无法用 closing keyword 关闭任何东西。**这不是放宽判据**：
- 三个提交的 message 里**没有**任何 `Closes`/`Fixes`/`Resolves` 关键字（已 grep 核实），
  所以不存在「意外关闭」或「假装关闭」；
- `#69` 只作为**标签**出现在 subject 中，指向一个不存在的编号，GitHub 侧无副作用；
- 因此本阶段的成果**不依赖**任何 issue 状态，全部结论以 `docs/R7_69_BUCKET_EXPANSION.md`
  与 `docs/decisions/0010-*` 为准（治理文件已进版本控制）。
- **待用户在 ZCode 之外的终端**：如需 issue 追踪，请手工创建 #69 并挂 `Closes #69`；
  但按上表，**科学结论不因该步骤缺失而改变**。

`478d982` 没有自己的 run 是 GitHub 的 push 语义：它与 `be5ce21` 在同一次 push 中推上，
每次 push 只在**分支尖端**触发一次 workflow。`478d982` 是 `be5ce21` 的祖先，所以
`36330894071` 跑到的树已包含它的全部改动——该 run 覆盖它。

三个提交已快进推送到 `main`（`0e09800..9a4e742`，非 force，线性），因此
`docs/R7_67_SEALED_TEST_REPORT.md` 的 9.09× 更正现在位于**默认分支**上。
17 条实验 workflow 全部 `skipped`（tag-gated 设计，不是失败）。
