# S3 Stage D 白盒化：训练后模型上 P/Z 两态的实际读数（CPU-only，0 GPU-h）

- **日期**：2026-11-07
- **状态**：**已测量**；结论**否证** `model/process_forecast_r7.py` 关于 8 个 anchored 槽位
  「逐一对应 8 个诊断代理」的文档陈述
- **`scientific_claim: false`**；`test_read: false`；`r = 0`；**GPU-h = 0.0**
- **决策**：0069（静态协变量接线，已 `accepted`）、**0094**（C1 前置否决的教训，方法复用）
- **产物**：`outputs/r7_s3_whitebox_trained_20261107_attempt01/probe.json`
  （`format: r7-s3-whitebox-probe-v1`）+ `probe.progress.npy`（33×33 逐 token 进度场）

---

## §0 一句话

在**训练后的真实权重**上（p1600 端点，128,528 updates），8 个 anchored 槽位**不是** 8 个独立信号：
它们两两之间的诊断剖面相关 **0.94–1.00**，8 个里只有 **2 个**的 argmax 落在其文档声称的诊断上。
Z 态逐位置求解进度与城市/植被静态场的相关只有 **|ρ| ≤ 0.43**，且方向是**建成区越高、进度越低**。

---

## §1 这一页补的是什么

`tools/r7_s3_whitebox_probe.py` 原本只做**定义级**检查，并在自己的 `limitations` 里写明
「**no trained checkpoint is loaded here**」。而 Stage D 的 brief 要求的是在**训练后的模型**上
测两件事。本页是那次测量：工具新增 `--checkpoint/--manifest/--store/--cases` 路径，
在 p1600 端点上跑 60 个 train case（`1000, 1500, …, 30500`，步长 500）。

**口径**：65×65 多年实例的 **train manifest**（不读 val、不读 2023 test）；每个 case 一次
`no_grad` 前向，`reasoning_steps=4`；P 态取 `process_predictions`（`[B,K,8]`，对 K 取均值）；
诊断目标由 `compute_process_tensor_diagnostics` 从**同一窗口的分析场**独立算出（8 维）；
Z 态取 `solver_state`（`[1,1089,192]` → 33×33 token 网格）的逐位置特征均值。

---

## §2 P 态：8 个 anchored 槽位 vs 8 个诊断目标

**60 个样本**（不是 5 个——5 个样本时 Spearman 0.9 只是 4/5 一致，无法支撑任何结论；
第一次跑就是 5 个，已作废重跑）。

### §2.1 逐槽位 argmax（文档声称槽位 i ↔ 诊断 i）

| 槽位 | argmax 诊断 | \|ρ\| | 是否等于文档声称 |
| ---: | --- | ---: | :---: |
| 0 | `vertical_wind_shear_850_500_mean` | 0.318 | ✗ |
| 1 | `vertical_wind_shear_850_500_mean` | 0.323 | ✗ |
| 2 | `vertical_wind_shear_850_500_mean` | 0.312 | ✗ |
| 3 | `vertical_wind_shear_850_500_mean` | 0.310 | ✗ |
| 4 | `moisture_advection_850_mean` | 0.253 | **✓** |
| 5 | `vertical_wind_shear_850_500_mean` | 0.311 | ✗ |
| 6 | `vertical_wind_shear_850_500_mean` | 0.330 | ✗ |
| 7 | `vertical_wind_shear_850_500_mean` | 0.349 | **✓**（巧合：槽位 7 的文档目标是 d7，而 d7 恰是 shear） |

**aligned_slots = 2 / 8。**

### §2.2 更关键的一条：8 个槽位几乎是同一个信号

把每个槽位对 8 个诊断的 Pearson 剖面归一化后求两两相关（槽位间「诊断画像」的相似度）：

```
        s0      s1      s2      s3      s4      s5      s6      s7
slot0 +1.000  +0.997  +0.999  +0.998  +0.952  +0.999  +0.999  +0.990
slot1 +0.997  +1.000  +0.997  +0.999  +0.954  +0.997  +0.998  +0.991
slot2 +0.999  +0.997  +1.000  +1.000  +0.943  +1.000  +0.999  +0.986
slot3 +0.998  +0.999  +1.000  +1.000  +0.947  +0.999  +0.999  +0.987
slot4 +0.952  +0.954  +0.943  +0.947  +1.000  +0.939  +0.948  +0.977
slot5 +0.999  +0.997  +1.000  +0.999  +0.939  +1.000  +0.999  +0.984
slot6 +0.999  +0.998  +0.999  +0.999  +0.948  +0.999  +1.000  +0.990
slot7 +0.990  +0.991  +0.986  +0.987  +0.977  +0.984  +0.990  +1.000
```

**0.94–1.00。** 八个槽位在「它们各自跟踪哪个诊断」这件事上几乎完全一致，
即它们**没有**分化成八个不同的过程代理。同时逐槽位的 |ρ| 只有 **0.25–0.35**——
弱相关。两者合起来的意思是：这些槽位**不是**八个诊断的读数，
而是一个弱相关的公共因子被复制了八份。

**这否证了** `model/process_forecast_r7.py:130` 的文档陈述
（"the first anchored_processes slots align to meteorological diagnostic proxy targets"）
在**数值意义上**成立。它可能仍然是一个合理的**设计意图**，但训练后的权重没有实现它。

---

## §3 Z 态：逐位置求解进度 vs 静态场

`solver_state` 在 **33×33 token 网格**上（65 / patch 2 = 32.5 → 33），静态产物在 **65×65 像素网格**上。
两网格**尺寸确实不同**，因此不做重采样、而是把每个 token 与其覆盖的像素块均值比较
（映射由两个 shape 推导，写进 `probe.json` 的 `mapping` 字段；不同网格会在此处**报错**而不是静默广播）。

进度场：`min 0.3792 / max 0.4439 / mean 0.3943`，`nonzero_fraction 1.0`（每个 token 都在动，
但动态范围只有 **0.065**，即进度**高度均匀**）。

| 静态变量 | Pearson | Spearman |
| --- | ---: | ---: |
| `lc_built_frac_2021` | −0.3030 | **−0.4150** |
| `lc_crop_frac_2021` | −0.2910 | −0.3307 |
| `lc_water_frac_2021` | −0.0916 | −0.3124 |
| `lc_tree_frac_2021` | +0.0728 | −0.1026 |
| `lc_bare_frac_2021` | +0.2533 | +0.0493 |
| `built_surface_frac_2020` | −0.3118 | **−0.4170** |
| `building_height_m` | −0.1289 | −0.2640 |
| `pop_density_km2_2020` | −0.2279 | **−0.4289** |

**方向**：建成区比例、建筑高度、人口密度**越高，求解进度越低**（三个负相关一致，ρ 到 −0.43）。
**量级**：全部 |ρ| ≤ 0.43，即静态场最多解释约 18% 的进度方差。

**不能说的**：不能说「求解器在城市上更省力」或「求解器忽略了城市」——0.43 的相关既不足以支持
前者，也不足以支持后者；而且进度的动态范围只有 0.065，本来就没有多少方差可解释。
**能说的**：进度与地表类型**有**可测但**弱**的负向关系，且**没有**出现「地表越复杂、进度越大」
这一直觉方向。

---

## §4 这一页**不**说明什么

- **不是**关于预报技巧的结论：两态读数描述训练后的权重**做了什么**，与 skill 无关；
  `scientific_claim: false`。
- **不是**对 8 槽位设计的判决：可能设计意图就是「一个公共因子的八种读出」；
  本页只否证**文档里的「逐一对应」表述**，不否证该设计本身。
- **不外推到其它 checkpoint**：只测了 p1600 端点（seed 41）。更早的 checkpoint 可能分化不同。
- **不外推到 129×129 wide 实例**：token 几何与 patch 相同但像素网格不同，静态产物也不同。
- **不给阈值判定**：campaign 对这两个关系**没有**规定任何数值门槛，因此本页只报测量、
  不宣布通过或不通过（工具在 `checks` 之外单独报告 `trained_state_measured`，不并入 `all_checks`）。
- **`aligned_slots = 2/8` 这个数不是「2 个槽位坏了」**：因为八个槽位彼此相关 0.94+，
  argmax 的分配本身不稳定；真正的读数是 §2.2 的槽位间相关，不是 argmax 计数。

---

## §5 复现

```
.venv/bin/python tools/r7_s3_whitebox_probe.py \
  --checkpoint outputs/r7_s3_throughput_physics_b_p1600_20261106_attempt01/seed41/training/update_0128528.pt \
  --manifest outputs/r7_s3_fy_multiyear_20261023_attempt01/store/manifests_treatment/train.jsonl \
  --store outputs/r7_s3_fy_multiyear_20261023_attempt01/store/cache.zarr \
  --cases 1000,1500,...,30500 \
  --out outputs/r7_s3_whitebox_trained_20261107_attempt01/probe.json
```

CPU-only，0 GPU-h，约 8 分钟（60 个 case × 一次前向）。检查点经 `load_checkpoint` 的
`model_code_sha256` 校验后才载入；静态场经 `require_static_identity()` 身份门后才读取。

**可复现等级**：`analysis-reproducible`（CPU 前向在固定权重下确定性；工具已入库，
`tests/test_r7_s3_whitebox_probe.py` 13 项覆盖定义级检查）。

---

## §6 未做

- 未测 seed 42/43 或更早 checkpoint（n=1 模型）；
- 未做槽位**消融**（逐个清零看预测变化）——那需要 GPU 或多次前向，且属另一轮设计；
- 未在 129×129 实例上测；
- 未把 §2.2 的槽位间相关做成自动回归测试（当前只在证据页记录）；
- 未修改 `model/process_forecast_r7.py` 的文档陈述——改文档属代码变更，须自己的决策与测试。
