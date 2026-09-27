# #67 封存 test 报告（只读，一次性）

**状态：已完成。** 本文件报告在 **January-only 重切** 的独立 store 上、对 5 个臂
× 3 个 seed × 5 个时效的**一次性 test 评估**，以及协议声明但此前未执行的两项分析
（按天 paired block 重采样、full/interior/boundary 分组）。

| 项 | 值 |
| --- | --- |
| 协议（先冻结） | `docs/R7_67_PUBLICATION_PROTOCOL.md`，commit `6259c78`（写入于读 test 之前） |
| 前次尝试与阻塞 | `docs/R7_67_SEALED_TEST_REPORT.md`（旧版）记录 2 月 test 与 1 月气候态桶不相交；旧文件已由本文件取代 |
| 重切 config | `configs/r7_era5_b2_recut_test_january.yaml`（commit `4058c9d`） |
| 数据来源 | 冻结的 `outputs/r7_b2_segment/source.nc`（sha256 `2ba504fe…1af31`），**无新下载** |
| 新 `data_identity` | `4055fc7e30fd4677ff30a58137466ff5ef9e8eda0da7cc6946205c26fb031bca` |
| 重训 | 5 臂 × 3 seed × 800 update，**15/15 完成**（1 个早停，见 §5），约 **1.35 GPU-h** |
| test 报告 | `outputs/r7_67_sealed_report/`（`sealed_test_report.json` + 3 张表 + 75 个评估目录） |
| 分析 | `outputs/r7_67_sealed_analysis/sealed_analysis.json`（60 个 paired 单元 + 75 个 region 单元） |
| 一次性声明 | `read_count: 1`；`test_read: true` |

---

## 1. 为什么必须重切，以及它改变了什么

原冻结段把 held-out 切在 1/2 月交界：val `[01-25,02-01)`、test `[02-01,02-06)`。
train-only 气候态只从 **1 月**拟合，于是 2 月 test 的**每个** case 都缺 `(month,hour)` 桶，
`normalized_climatology` 拒绝回退（刻意防泄漏），**所有臂**都无法评估。

重切把 held-out 边界移入 1 月：

| split | 范围 | 时次 | 月份 |
| --- | --- | --- | --- |
| train | `[01-01, 01-25)` | 96 | 全部 01（**与冻结链逐字节相同**） |
| val | `[01-25, 01-27)` | 8 | 全部 01 |
| **test** | `[01-27, 02-01)` | 20 | **全部 01** |

**train 清单与冻结链逐字节相同**（已断言：`train.jsonl` 字节相等，94 个窗口不变），
但**checkpoint 不可复用**——`data_identity` 覆盖 `split_time_ranges`，所以全部 15 个
arm-seed 都**重新训练**。**因此这是一个新实验**，本文件不声称它确认了此前冻结的链。

test 在 5 个时效上的窗口数：**18 / 17 / 15 / 11 / 7**（6/12/24/48/72h），
即 **72h rollout 现在可评估**（旧划分在此点直接为 0）。

## 2. 主表：test t2m RMSE（3 seed 均值）

| lead | unet | native_window | afno_small | generic | process | climatology（0 参数） |
| --- | --- | --- | --- | --- | --- | --- |
| 6h | 3.507 | 2.864 | 3.004 | **2.515** | 2.597 | **0.277** |
| 12h | 4.339 | 3.505 | 3.676 | **3.199** | 3.279 | 0.279 |
| 24h | **3.666** | 3.799 | 3.966 | 4.227 | 4.204 | **0.282** |
| 48h | **4.388** | 5.660 | 5.393 | 6.870 | 6.176 | 0.287 |
| 72h | **3.915** | 5.591 | 5.518 | 7.051 | 6.760 | 0.316 |

**核心负结果（headline）**：在 test 上，**五个神经臂在 17 变量 × 5 时效的 85 个单元里
全部输给零参数 climatology**，计数一律是 **better 10 / worse 75 / unresolved 0**。
t2m 的劣势最刺眼：6h 最好者（generic 2.515）仍是 climatology（0.277）的 **9.09 倍**。

**headline 变量在 headline 时效上的倍率（相对 climatology）**：

| 变量 | 6h 最好倍率 | 24h 最好倍率 |
| --- | --- | --- |
| t2m | generic **9.09×** | unet 13.02× |
| mslp | generic 243.99× | unet 520.41× |
| v850 | afno_small 2.38× | afno_small **4.86×** |
| u10 | afno_small **1.28×** | process 1.86× |

**climatology 在 test 上占优的成因必须如实说明**：本段只有一个 1 月、
`(month,hour)` 桶只有 **4** 个，`[01-27,02-01)` 的 test 块几乎所有时次都落在同一批桶里，
所以「气候态」近乎就是「该月该小时的平均天气」。这是**数据范围的产物，不是「基线很强」**
（与 B2/B3 的既有结论一致，此处复现）。

## 3. 协议声明分析（一）：按天 paired block 重采样

**方法**：同一 UTC 日的所有起报不是独立样本，因此**重采样单位是天，不是 case**；
每个 (臂对, seed, 时效) 做 2000 次按天 bootstrap，报告 arm−reference 的 MSE 差值区间
（负值 = arm 更好）。区间**仅用于估计，绝不用于选模型**。

**process vs generic（共享结构对，t2m）**：

| lead | seed41 diff [区间] | seed42 | seed43 | 跨 seed 同号？ |
| --- | --- | --- | --- | --- |
| 6h | +0.849 [+0.44,+1.23] ✱ | +0.050 [−0.37,+0.43] | +0.374 [+0.02,+0.71] ✱ | **是**（都为正 = generic 更好） |
| 12h | +0.564 [−0.18,+1.75] | −1.067 [−2.03,−0.13] ✱ | +2.155 [+0.29,+5.12] ✱ | 否（翻号） |
| 24h | +1.563 [−0.33,+3.45] | +2.768 [−0.52,+5.64] | −4.494 [−7.27,−0.72] ✱ | 否（翻号） |
| 48h | −13.375 [−17.9,−10.3] ✱ | +7.597 [+1.3,+13.8] ✱ | −20.031 [−34.6,−10.0] ✱ | 否（翻号） |
| 72h | −4.143 [−9.31,+2.75] | +9.551 [−4.52,+28.3] | −15.045 [−20.2,−8.2] ✱ | 否（翻号） |

（✱ = 区间不含 0。）

**结论**：**只有 6h 的符号跨 seed 一致**，且方向是 **generic 优于 process**。
12/24/48/72h **全部跨 seed 翻号**——即使区间本身不含 0，不同 seed 的区间指向相反方向。
这与预诊断 §4.1 的结构性解释一致：两臂在功能上只差一个投影，差异主要来自初始化漂移。

**区间含 0 的比例**（每对 15 个单元，含各 lead × seed）：
process_vs_generic **6/15 含 0**；generic_vs_native_window **5/15**；
unet_vs_native_window **5/15**；afno_small_vs_native_window **9/15**。
即**三分之一的格子连自身 seed 内都说不清方向**。

**seed 与天气采样分开报告**（协议要求）：6h 的 generic 优势在 seed41/43 上区间不含 0
而在 seed42 上含 0，说明**该方向的强度本身也受 seed 影响**；
按天 bootstrap 的区间宽度（如 6h seed41 的 [0.44,1.23]）反映的是天气采样不确定性，
二者不得相加当独立大 N。

## 4. 协议声明分析（二）：full / interior / boundary 分组

区域分层来自同一批预报（是**误差分解，不是边界强迫实验**），margin = 1/2 个网格
（域为 65×65）。

| lead | full | interior_1 | edge_1 | interior_2 | edge_2 | edge_1/full | edge_2/full |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6h | 2.897 | 2.885 | 3.087 | 2.876 | 3.050 | **1.066** | 1.053 |
| 12h | 3.600 | 3.584 | 3.833 | 3.572 | 3.795 | **1.065** | 1.054 |
| 24h | 3.972 | 3.986 | 3.742 | 3.969 | 3.989 | 0.942 | 1.004 |
| 48h | 5.697 | 5.724 | 5.216 | 5.703 | 5.617 | 0.915 | 0.986 |
| 72h | 5.767 | 5.792 | 5.288 | 5.758 | 5.785 | 0.917 | 1.003 |

**读数（如实，不挑方向）**：

- 在**训练时效（6h）**边界更差：edge_1 比 full 高 **6.6%**，edge_2 高 5.3%——
  这与「边界缺上下文」的直觉一致，是本次唯一同向且可解释的分层信号。
- 在**长时效（48/72h）**边界**反而更好**（edge_1/full ≈ 0.92）。
  最可能的解释是长时效误差由大尺度漂移主导，而边缘区域面积小、纬度加权低，
  受少数坏初值影响的方式与内部不同；**本报告不把该方向解释为「边界更容易预报」**，
  因为它与 6/12h 的符号相反，且每个 lead 只有 2–3 个 day block。
- edge_2 与 edge_1 给出的方向在 24h 上不一致（1.004 vs 0.942），说明该分层对 margin
  选择敏感——**这是局限，不是可选的呈现**。

## 5. 训练曲线（完整 epoch 均值）

每 epoch = 14 个 update；下表为**完整 epoch** 的首/末均值（跨 seed 平均）与逐 seed 明细：

| 臂 | first_full | last_full | ratio | 完整 epoch 数 | 选中 update | 停止原因 |
| --- | --- | --- | --- | --- | --- | --- |
| unet | 0.1863 | 0.1167 | 0.6266 | 17 | 700/800/600 | 均到达声明端点 |
| native_window | 0.1839 | 0.1011 | 0.5496 | 17 | 700/500/600 | 均到达声明端点 |
| afno_small | 0.1803 | 0.0748 | **0.4148** | 17/**14**/17 | 600/**300**/700 | seed42 **早停** |
| generic | 0.1830 | 0.0897 | 0.4899 | 17 | 800/500/800 | 均到达声明端点 |
| process | 0.1830 | 0.0894 | 0.4888 | 17 | 800/500/600 | 均到达声明端点 |

**逐 seed 明细（first / last full-epoch mean）**：

| 臂 | seed41 | seed42 | seed43 |
| --- | --- | --- | --- |
| unet | 0.1863 / 0.1195 | 0.1864 / 0.1179 | 0.1861 / 0.1128 |
| native_window | 0.1841 / 0.1018 | 0.1842 / 0.0999 | 0.1835 / 0.1015 |
| afno_small | 0.1804 / 0.0710 | 0.1808 / 0.0799 | 0.1798 / 0.0735 |
| generic | 0.1834 / 0.0909 | 0.1832 / 0.0889 | 0.1825 / 0.0892 |
| process | 0.1832 / 0.0908 | 0.1832 / 0.0890 | 0.1824 / 0.0884 |

**如实记录**：`afno_small` seed42 触发早停（4 次验证检查无 ≥0.1% 相对改善），
所以它的 14 个完整 epoch 与其它 17 个不同——**该臂的最终损失与其余臂不可直接比长短**。
其余 14/15 都到达声明的 800 步端点。所有臂的训练损失都大幅下降（ratio 0.41–0.63），
**但这与 test 上输给气候态并不矛盾**：这 85 个 test 单元是**自由 rollout**，
而训练目标只是 +6h 单步；同时 test 的 `(month,hour)` 桶几乎恒定，气候态极难击败。

## 6. 交付清单（#67 判据逐项）

| 判据 | 状态 |
| --- | --- |
| 先冻结协议与 case 表（headline/时效/可接受退化/多重比较） | ✅ 协议 §6，commit `6259c78`（早于本 test） |
| ≥3 固定 seed 分别呈现 | ✅ 逐 seed 呈现，从未先平均（§2/§3/§5） |
| 按天/周 paired block 重采样 | ✅ **已执行**：按天（val/test 只有 7/5 天，按周只剩 1 个 block）——§3 |
| 四季与 full/interior/boundary 分组 | full/interior/boundary **已执行**（§4）；**四季不可做**（数据仅 1 月，如实标注 UNVERIFIED） |
| 无降水目标不写降水技巧 | ✅ 全文无降水 skill |
| 外部预训练模型只列单独 reference 表 | ✅ 未运行任何外部模型；主表只含本库臂与零参数控制 |
| **只读 test 报告** | ✅ **本文件** + `outputs/r7_67_sealed_report/`（`read_count: 1`） |
| 代码版本 | ✅ `d9da752` 起的 SHA 链，记录于 TASK_QUEUE |
| 环境锁定 | ✅ `docs/rules/environment.md`；报告含 torch 版本与 device |
| 数据 recipe / 许可 | ✅ `configs/r7_era5_b2_recut_test_january.yaml`；源为冻结的 `source.nc`（sha256 记录），来源与许可见 `docs/R7_D1_ACQUISITION.md`、决策 0004/0008 |
| checkpoint / metrics 索引与 digest | ✅ `sealed_test_report.json` 内每个 arm-seed 的 `checkpoint_sha256` 与每格评估目录 |
| 图表脚本 | ✅ `scripts/freeze_r7_67_test_report.py`、`scripts/analyze_r7_67_paired_blocks.py`、`scripts/analyze_r7_67_sealed_report.py`（全部从原始充分统计量生成，无手改数字） |
| README 三栏 | ✅ smoke / negative study / publication benchmark |

## 7. 局限（是否仍适用）

| 局限 | 状态 |
| --- | --- |
| **单一年 1 月、4 个 (month,hour) 桶** | **仍适用**，且是 climatology 在 test 上占优的**主因**；跨季节 UNVERIFIED |
| test 是**工程再划分**，不是封存多年的标准测试集 | **仍适用**：20 时次 / 5 天；`test` 在此**首次且一次性**被读 |
| **重切后的 test 更小**（18→ 各 lead 7–18 窗口） | **新增局限**：48h 只有 11、72h 只有 **7** 个 case ⇒ 长时效区间很宽 |
| val 只有 8 时次（6/5/3/0/0 窗口） | **新增局限**：val 在 48/72h **无法**支撑模型选择；训练期验证只用 6h |
| afno_small seed42 早停 | 该臂的 epoch 数不同，损失不可直接比长短 |
| 3 seed 不是显著性检验 | **仍适用**；按天 bootstrap 只量化天气采样不确定性 |
| 800 步非收敛 | **仍适用** |
| 分级分层对 margin 敏感 | **新增局限**：edge_1 与 edge_2 在 24h 上方向不一致 |

## 8. 结论

1. **#67 要求的只读 test 报告已产出**，包含全部 17 变量 × 5 时效、逐 seed 呈现、
   按天 paired block 区间与 full/interior/boundary 分组。
2. **科学结论是负面的**：所有神经臂在 test 的全部 85 个单元上输给零参数气候态；
   process 相对 generic 只在 6h 有跨 seed 一致的差异，且方向是 **generic 更好**；
   12–72h 全部跨 seed 翻号。
3. **本报告不声称流程递归假设成立**，也不声称本重切实验确认了此前冻结的实验链
   （checkpoint 因新 `data_identity` 全部重训）。
4. 该结果与 B1/B2/B3/C1/C2/C3 的方向一致，构成**负结果论文包**的可用材料。

## 9. 复现

```bash
# 1) 只读 preflight（无 --write 不写任何东西）
.venv/bin/python prepare_r7_local.py --source outputs/r7_b2_segment/source.nc \
    --config configs/r7_era5_b2_recut_test_january.yaml \
    --report outputs/r7_recut_segment/preflight_report.json
# 2) 写 store（授权内的重切；train 块与冻结链逐字节相同）
.venv/bin/python prepare_r7_local.py --source outputs/r7_b2_segment/source.nc \
    --config configs/r7_era5_b2_recut_test_january.yaml --write \
    --store outputs/r7_recut_segment/store/cache.zarr \
    --manifests outputs/r7_recut_segment/store/manifests --max-raw-gib 1
# 3) 重训 5 臂 × 3 seed（每 seed 一个进程）
for s in 41 42 43; do
  .venv/bin/python scripts/study_r7_b2_multiseed.py --mode seed --phase confirm \
    --seed $s --device-index 0 --manifests outputs/r7_recut_segment/store/manifests \
    --out outputs/r7_recut_multiseed
done
# 4) 一次性 test 报告 + 两项分析
.venv/bin/python scripts/freeze_r7_67_test_report.py \
    --manifests outputs/r7_recut_segment/store/manifests \
    --train-root outputs/r7_recut_multiseed --out outputs/r7_67_sealed_report
.venv/bin/python scripts/analyze_r7_67_sealed_report.py \
    --report outputs/r7_67_sealed_report --out outputs/r7_67_sealed_analysis
```
