# R7 城市静态协变量接线轮（S3 / 决策 0069）

- **日期**：2026-10-29（UTC 读数见各 receipt；attempt02 的 prepare 起于 `2026-10-29T15:43:34Z`）
- **`scientific_claim`: false**；**`test_read`: false**
- **产物**：`outputs/r7_s3_static_covariates_wiring_20261029_attempt02/`
- **`static_identity`**：`8671ed7ce0c82d1ec9ac834e68e2e3c7bbc30e14fa6b8a07501e070d27708b03`
- **`protocol_sha256`**：`48c09621e2fbac1d5705839d3251dd762172eb43e1f366290255d3fbbeb9bbca`
- **代码提交**：`67ebf14`（实现 `896eabf`，探针修正 `67ebf14`）；`model_code_sha256 =
  a38be16c9a04f5f1704fed6747e3d8da0f8507168464a337a85975de6b8a1a0a`
- **不是**科学轮：本轮证明**通路存在且可审计**，**不**证明静态协变量有用。

## 0. 一句话结论

四个预声明工程出口**全部为真**：`static_identity` 确实进入 `protocol_sha256` 而
`data_identity` 未动（W1）；`model/**` 改动前后的指纹**逐位相同**（W2）；两臂各自在 val 上
跑完五个 lead、`boundary_rmse.csv` 行数逐行相等（W3）；模型读进的静态场与已发布
`.npz` **逐位相等**，独立循环再派生在其声明的容差内（W4）。
**判决：`engineering-positive`。这四句话没有一句是关于天气预报技巧的。**

## 1. 本轮做了什么（以及为什么只做这些）

- **新增开关**：`model/static_conditioning_r7.py` 提供 `StaticConditioning`，
  是一个**时不变**、**零训练参数**的逐 token 加性项；`model/weather_forecaster_r7.py` 与
  `model/process_forecast_r7.py` 各加一个 `static_conditioning: bool = False` 并透传，
  布尔校验并入既有的 `_validate_switches`。
- **身份形态**（决策 0063 第 3 条 / 0069 第 1 条）：`static_identity` 进入
  `protocol_sha256` 的 `static` 段，**不**进入 `data_identity`。两臂的
  `train_data_identity` 实测**相同**（`81307008b572e3ff…`）。
- **单因子**：两臂 spec 的差集实测**只有** `static_conditioning` 一个键；
  两臂参数量实测**相等**（`3,286,037`），`parameter_delta = 0`。
  所以 on/off 差异**不可能**来自额外容量。

## 2. 四个预声明出口（逐条实测，不看读数之前的冻结）

出口在 `docs/goals/s3-urban-static-wiring.md` §4.2 与决策 0069 第 3 条中**先于运行**写定。

| 出口 | 判据 | 实测读数 | 结果 |
| --- | --- | --- | --- |
| **W1** | 改 `static_identity` 一个字符 ⇒ `protocol_sha256` 必变，`data_identity` 不变 | `48c09621e2fb…` → `a52465bea934…`；`data_identity_unchanged = true`；`data_identity_carries_the_static_identity = false` | **true** |
| **W2** | 改 `model/**` 前后同 seed 同固定输入的三项指纹逐位相等 | `differences = []`；参照 `model_code_sha256 = 60de3b2d…`，实测 `a38be16c…`（不同，正是"参照先于改动"的证据） | **true** |
| **W3** | 两臂各跑完五个 lead，`boundary_rmse.csv` 行数逐行相等 | control/treatment 均为 `6h/12h/24h/48h/72h = 51/51/51/51/51` 行；每 lead `n_evaluated = 10` | **true** |
| **W4** | 模型读进的静态场与发布 `.npz` 逐位相等 | `buffer_bitwise_equal = true`；`module_features_bitwise_equal_expansion = true`；独立循环再派生最大绝对差 `5.55e-17`（容差 `1e-12`） | **true** |

`readings.json` 的机器可读判决：`{"w1": true, "w2": true, "w3": true, "w4": true,
"all": true, "verdict": "engineering-positive", "scientific_claim": false}`。

### 2.1 W2 的做法（写清顺序，因为顺序本身就是证据）

参照读数在**任何 `model/**` 编辑之前**落盘：

```
.venv/bin/python scripts/r7_static_wiring_reference.py \
  --out outputs/r7_static_wiring_20261029/pre_change_reference.json
→ model_code_sha256 60de3b2dc5de2871f45c7a563153ef8e1cfa688be38367c89d0c718dfc06df3e
  initial_parameter_digest 511da521b55c8194…
  first_update_loss 0.01961401104927063
  revision: head 5e4adf9f6ad9, model_dirty false
  sha256 4d0dc228699ae374b277f57cf3693cde1233ccdd554557dc68ad88a2a6ca3230
```

同一命令跑第二次 `values` **完全相同**（参照本身可复现）。改动后同命令加 `--compare`
输出 `{"differences": [], "passed": true}`。

**为什么不是"两臂互相比"**：两臂都跑新代码，那只能证明开关自洽，证明不了 off 路径
等于改动之前的行为。所以 W2 判的是**改前 vs 改后**。

### 2.2 attempt01 是失败 attempt（如实记账，不删除）

`attempt01` 的 prepare 跑完（507.58 s），W1/W2 通过、**W4 失败**——失败在**探针**里，不在模型里：
探针用 float64 重建展开式再与模块的 float32 输出做**逐位**比较，实测差
`7.152557373046875e-07`（在 3.6 量级的数值上正好是 1 个 float32 ulp），**这条判据永远不可能通过**。
改用模块自己的 float32 算术重建后**逐位相等**。
`attempt01` 保留 `FAILED_ATTEMPT.json`，**0 GPU-h**（没有跑任何 seed）。
**没有放宽任何判据**：W4 的出口原文（buffer 对 `.npz` 逐位）一字未改，独立循环校核也保留其
声明的 float64 归约阶容差 `1e-12`。

## 3. on/off 读数（**记录，不判读**）

`val` 2022，`max_samples = 10`/lead，t2m，`full` 区域，seed 41：

| lead | control RMSE (K) | treatment RMSE (K) | Δ (treat − ctrl) |
| --- | --- | --- | --- |
| 6 h | 5.737827 | 5.739231 | **+0.001404** |
| 12 h | 6.969414 | 6.971441 | **+0.002027** |
| 24 h | 2.669372 | 2.673001 | **+0.003629** |
| 48 h | 3.051431 | 3.055721 | **+0.004290** |
| 72 h | 3.391581 | 3.394615 | **+0.003034** |

**这些数字不构成任何结论，原因逐条写在 §6。** 最直接的一条：每臂只有 **120 updates**
（约 0.037 epoch）与 **10 个样例**，且**单 seed**。要把 `+0.001…+0.004 K` 读成"静态协变量有害"
是把噪声当成信号。

两臂确实**真的不同**，不是"D 变了但没接线"：
- 6 个进度点的 loss **没有一个逐位相同**，最大差 `9.43e-03`；
- 首步 loss `1.597623348236084`（control）vs `1.5997238159179688`（treatment）；
- 两个 checkpoint 的 sha256 不同。

## 4. 身份链

| 项 | 值 |
| --- | --- |
| `static_identity` | `8671ed7ce0c82d1ec9ac834e68e2e3c7bbc30e14fa6b8a07501e070d27708b03` |
| 发布 `output_sha256` | `1e9473f04bf1fa47ac279680a4ac4b54de933fee6afde934478e73f6e2dc37e5` |
| 发布 recheck | `5.1535351518462713e-08`（抽样格独立再派生，float32 精度内） |
| 数据轮登记 | `record:s3-urban-static-covariates-acquisition`（`engineering-positive`） |
| `protocol_sha256` | `48c09621e2fb…`（含 `static` 段） |
| `data_identity`（两臂相同） | `81307008b572e3ff032d5251e788c582f339b0e3dbbc3fea4226d44c6a335af0` |
| `model_code_sha256` | `a38be16c9a04f5f1704fed6747e3d8da0f8507168464a337a85975de6b8a1a0a` |
| 执行文件清单 | **94** 个具名文件（import 闭包，决策 0067），不是活 glob |

**静态场用到的 8 个变量**（时不变口径，多期层取最新一历元）：

```
lc_built_frac_2021, lc_tree_frac_2021, lc_crop_frac_2021, lc_water_frac_2021,
lc_bare_frac_2021, built_surface_frac_2020, building_height_m, pop_density_km2_2020
```

`built_surface_frac` 与 `pop_density_km2` 的**多期数组（5×65×65）保留在产物里**，
留给将来的时变轮；本轮不用，因为用了就不再是"时不变条件"。

## 5. 成本（实测，不是估计）

| 项 | 值 |
| --- | --- |
| attempt01 prepare（失败 attempt） | 507.58 s，**0 GPU-h** |
| attempt02 prepare | 512.26 s（CPU） |
| control：120 updates 训练 | 309.1 s（实测 2.58 s/update） |
| treatment：120 updates 训练 | 300.6 s（实测 2.51 s/update） |
| seed 阶段总计（含两臂评分） | 3247.23 s |
| 软预算 / 硬上限 | 7200 s / 14400 s；`soft_overrun_seconds = 0.0` |
| 网络 | 0 B（不下载） |
| **GPU-h** | **0.9020**（3247.23 s / 3600，seed 阶段全部墙钟） |

预算是**先测后冻**的：2.44 s/update（20 updates / 48.7 s 探针）、首个评估 lead 冷启 736 s、
预热后 12.6 s。这些数字写进了驱动注释，冻结的 `per_seed_seconds = 3600`。

## 6. 限制（本轮**不能**支撑的任何说法）

1. **`scientific_claim: false`，且本轮没有科学判据。** 四个出口全是工程断言。
2. **单 seed、120 updates、10 样例/lead。** 不是种子一致性证据，不是技巧读数。
   §3 的 Δ 只能当"通路跑通时两端数字确实不同"的旁证。
3. **零参数的代价是"学不到哪个协变量重要"。** 固定投影 + 固定二次展开让
   `parameter_delta = 0`（于是 on/off 差异不是容量差异），但也意味着本实现无法学习
   变量权重或交互。**因此 on 臂的 null 不能读成"静态协变量无信息"。**
4. **本轮改了 `model_code_sha256`（`60de3b2d…` → `a38be16c…`）。**
   `training/r7_experiment.py:124-134` 的 `load_checkpoint` 按设计拒绝外来实现，
   所以**所有已登记的迁移父在本代码下不可加载**。R2–R5 开轮前必须先重跑
   `scripts/migrate_r7_parent_state.py --out <新目录>`（model-only 安装，六个源 pin 与
   `state_digest` 双向核对不变）。**本轮没有做这件迁移**，它是一次独立的身份操作。
   ⇒ 本轮的臂是**新代码基线**，与 R1–R5 那些迁移父臂**不可横向比较**。
5. **评测上限 `max_samples = 10`** 是为了把五个 lead × 两臂装进墙钟；这不是 skill 读数。
6. **静态场是派生统计量，不是观测真值**（`derived_statistic_not_truth` 写在发布 manifest 里）。
   它从 WorldCover/GHSL 原生栅格面积加权聚合到 0.25°，不得当作地表真值使用。
   **LandScan 缺层** ⇒ 人口维度只有 GHSL-POP，没有交叉验证。
7. **GPU 不可逐位复现**：本节所有 GPU 数字来自一次顺序运行，没有重复运行来界定方差。
8. **`execution_files` 的活 glob 缺陷仍在 R1 的老驱动里**（决策 0065/0067）。
   本轮的新驱动用 94 项具名清单，测试含"新增无关 `.py` 不改变清单"的反证。

## 7. 复现命令

```
# W2 参照（必须先有，且必须早于 model/** 编辑）
.venv/bin/python scripts/r7_static_wiring_reference.py --out <ref.json>
.venv/bin/python scripts/r7_static_wiring_reference.py --compare <ref.json>

# 本轮
.venv/bin/python tools/r7_round.py freeze
.venv/bin/python tools/r7_round.py stage prepare
.venv/bin/python tools/r7_round.py stage seed 41
.venv/bin/python -m pytest tests/test_r7_static_covariates_wiring.py -q
```

`attempt01` 不可续跑（执行树不同 ⇒ 不同 `protocol_sha256`）；
`attempt02` 的 `readings.json` 是唯一判决来源。

## 8. 未做

- **没有**做 on/off 的科学比较，**没有**据此推进或关闭任何轴。
- **没有**重跑父迁移（见 §6 第 4 条）——这是 R2–R5 的**前置**，不是本轮的产物。
- **没有**用多期静态场（`built_surface_frac` / `pop_density_km2` 的 5 个历元）做时变条件。
- **没有**引入任何可学习的静态编码器（那会引入容量混杂，需要自己的 ADR 与参数匹配控制）。
- **没有**读 2023 test；`r` 仍为 0。
