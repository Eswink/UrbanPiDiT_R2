# 0113 — 城市静态协变量的三臂消融：placebo 臂、state-dict 级差异、以及「用但未消融」作为文献空白

- **日期**：2026-11-13
- **状态**：proposed
- **范围**：`docs/goals/s3-urban-static-ablation.md`（唯一执行源）、
  `scripts/study_r7_urban_static_ablation.py`（待写驱动）、
  `scripts/r7_urban_static_ablation_protocol.py`（待写协议构建器）、
  `tests/test_r7_urban_static_ablation.py`（待写反证）、
  `tools/reconcile_r7_static_parent.py`（**只复用，不修改**）。
  **本条不修改 `model/**` 任何文件。**
- **依据**：`docs/plans/0032-urban-static-covariate-ablation.md`（本条对应的归档计划）、
  决策 **0063**（静态数据获取）、**0069**（接线，`accepted`）、**0097 D1.d**（父迁移）、
  **0049**（显式 tendency 输入关闭，线性冗余）、
  `record:s3-typed-spatial-result`（4 个物理诊断量作输入，float32 噪声级）、
  `docs/R7_STATIC_COVARIATES_WIRING.md`（wiring 轮的两个坑）、
  `outputs/r7_static_parent_reconcile_20261108_attempt02/receipt.json`（父可加载的实测证据）
- **一手文献**（2026-11-13 由 `arxiv-researcher` 读正文）：`arXiv:2212.12794` §3.3、
  `2211.02556` §III-B、`2202.11214` §2.2、`2304.02948` §2.1、`2301.10343` App. Table 9、
  `2405.13063` App. §8.1、`2406.01465` §3 Table 1、`2409.02120` §5.4

## Context

**① 七篇主流神经天气模型全部使用了静态输入，而没有任何一篇消融过它。**

2026-11-13 的一手核对（逐字引文见计划 0032 §0.3）得到的静态输入集合是：

- GraphCast（`2212.12794` §3.3）：「the binary land-sea mask, the geopotential at the surface,
  the cosine of the latitude, and the sine and cosine of the longitude」
- Pangu-Weather（`2211.02556` §III-B）：「three constant masks (i.e., the topography mask,
  land-sea mask and soil type mask)」
- ClimaX（`2301.10343` App. Table 9）：「Static: Land-sea mask (LSM)」「Static: Orography」
- Aurora（`2405.13063` App. §8.1）：「three static variables … geopotential at the surface …
  land–sea mask … soil-type」
- AIFS（`2406.01465` §3 Table 1）：「Land-sea mask, orography, standard deviation of sub-grid
  orography, slope of sub-scale orography, insolation, latitude/longitude, time of day/day of year」
- FourCastNet（`2202.11214` §2.2）与 FengWu（`2304.02948` §2.1）**没有静态输入**

**消融方面是零命中。** GraphCast 消融 multi-mesh（§7.3.1）、FengWu 消融 replay buffer（§4.4）、
Aurora 消融训练数据多样性（§15）——**没有一篇消融静态输入本身**。
⇒ 五篇使用静态特征的论文全部是「**使用但未消融**」。

**② 城市 / 土地覆被这四类协变量，在上述全部七篇里都是零命中。**

`lc_built_frac`、`building_height_m`、`pop_density_km2`、`lc_*_frac` 从未作为任何神经天气模型的
静态输入。城市尺度的最近邻 `2409.02120` 用的是 LightGBM 而非神经天气模型、单城市、
且也没有静态特征消融。

**③ 但我们自己的 wiring 轮留下了一个会毁掉归因的缺陷。**

`docs/R7_STATIC_COVARIATES_WIRING.md` 逐字：

> 「the static term is a fixed projection over a fixed polynomial expansion, so `parameter_delta` is
> exactly 0; that removes the capacity confound but also means the path cannot learn which covariate
> matters, so **an on-arm null is not evidence the covariates are uninformative**」

代码核对确认：`StaticConditioning` 的 `projection` 是
`torch.randn(16, 192, generator=Generator().manual_seed(0)) / √16`
（`model/static_conditioning_r7.py:178-184`），`register_buffer` 而非 `Parameter`。
⇒ **任何空间变化的 8 场输入都会给出互不相同的逐 token 码。**
而模型**已经有** `position_features()` 的 `sin/cos(lat)`、`sin/cos(lon)` 四个谐波
（`model/spacetime_conditioning_r7.py:217`）。

⇒ **一个两臂 on/off 的阳性结果，无法排除「这只是给模型补了一张更细的位置查找表」。**
这个替代解释必须在设计阶段就被实验排除，而不是等读数出来再辩解。

**④ 上一轮的父封锁问题已经被解掉，本轮可以父锚定。**

wiring 轮的登记逐字写「every registered migrated parent is unloadable under this code」。
今天不再成立：`tools/reconcile_r7_static_parent.py` 在 `0097 D1.d` 下跑过，
`outputs/r7_static_parent_reconcile_20261108_attempt02/receipt.json` 记录
`"strict_load_of_reconciled_passed": true`、`"parent_model_code_sha256": "59340710623c…"`，
而实测 `model_code_digest()` = `59340710623c…`，与该 pin **相等**。

## Decision

### D1 —— 采用三臂：control / treatment / placebo

| 臂 | `field` | `projection` |
| --- | --- | --- |
| control | 不构造（`static_conditioning=False`） | —— |
| treatment | 发布的 8 层真实静态场 | 发布投影 |
| placebo | 同一份 8 层场**逐格空间置换**（固定种子） | **同一份**发布投影 |

**placebo 的定义必须精确到可检验**：对每一层独立做**固定种子的逐格置换**，
使得「排序后的值」与 treatment **逐位相同**（同一个多重集），只有空间位置被打乱。

**必须空间置换、不能打乱通道标签**：打乱通道会改变每个 token 的码的分布；
逐格置换保持每个协变量的边缘分布完全不变，只破坏空间自相关。
这样 treatment 与 placebo 的差别就**只有**「协变量是否与地理位置一致」。

**这只在 state dict 层实现。** 三臂共享同一份代码、同一个 `model_code_sha256`、
同一份父权重、同一个 seed、同一批更新。差异化只在两个 buffer 的取值上。

### D2 —— 不修改 `model/**`，不用开关区分臂

`model_code_digest()`（`training/r7_experiment.py:31`）覆盖 `model/**/*.py` **全部文件**，
包括 `static_conditioning_r7.py`。而 `load_checkpoint`（`training/r7_experiment.py:128`）
逐字要求 `checkpoint['model_code_sha256'] == model_code_digest()`。

⇒ **任何模型代码编辑都会让三 seed 的注册父全部不可加载**，重演 wiring 轮的封锁。

因此：**treatment 与 placebo 的差别通过传给 `fine_tune_long_rollout(parent_weights=...)`
的 state dict 实现**（`_validate_imported_model` 在 `training/r7_autoregressive_runner.py:161`
做 `model.load_state_dict(parent_weights, strict=True)`）。
`tools/reconcile_r7_static_parent.py` 的 `_published_buffers()` 已经展示了如何在模块外部
构造这两个 buffer——本轮复用它，不改它。

### D3 —— `parameter_delta` 三臂恒为 0，这是设计的核心优势

`field` 与 `projection` 都是 `register_buffer`，不在 `model.parameters()` 里。
⇒ 任何 skill 差异**不可能**来自额外容量。必须由协议逐臂实测并断言，不得写成字面量。

### D4 —— (f) 是主判据，且它判的是归因不是成功

预声明出口（全文见计划 0032 §2.3）：

| 判据 | 要求 |
| --- | --- |
| (a) | 72 h t2m skill 三 seed 配对 Δ(treatment − control) **全 ≥ 0** |
| (b) | 24/48 h skill 配对 Δ 三 seed **全 ≥ 0** |
| (c) | 6/12 h skill 配对 Δ **≥ −0.005**（容许短期微降） |
| (d) | control 臂**逐位等价**（由代码路径 + 反证测试支撑，**不由读数自证**） |
| (e) | 预检：8 层全有限；`static_identity` 通过；三臂 `parameter_delta == 0`；**placebo 逐变量多重集与 treatment 逐位相同** |
| (f) | **归因**：Δ(treatment − placebo) 在 72 h **≥ 0.001 K**（treatment 优于 placebo） |

**若 (a) 真而 (f) 假，本轮记为负**，结论写「静态协变量的效应与空间结构无关，等价于位置码」。
这仍是一个可发表的机制负结果，比一个无法归因的正结果有价值。

**阈值 0.001 K 的来源要说清**：它是**观测分辨率的量级**——`boundary_rmse` 以 K 记，
wiring 轮的 on/off Δ 落在 `+0.0014 … +0.0043 K`（120 updates 的噪声级），
而 Stage A 的可分辨效应在 `0.06` 量级（skill Δ）。0.001 K 取在「噪声上界」，
即要求 treatment 至少要比 placebo **好过一个噪声底**，不是要求它好过一个显著量。

### D5 —— 剂量与预算冻结前写定，事后不放宽

- **3 臂 × 3 seed × 4,000 updates**，从 pin 的 p1600 父微调，batch 4，recipe 逐字沿用
- 软预算 **115,200 s** / 硬上限 **230,400 s**（与 Stage A 同口径）
- 不扫任何权重（本轮无自由权重——buffer 是固定的）、不加 seed、不改 4,000 updates

## Consequences

**正面**

- 把「七篇主流模型用了静态特征但零篇消融」这一**可直接引用的文献空白**，做成一次自有测量。
  无论正负，结论都有明确的文献位置可比。
- placebo 臂把「位置查找表」这个致命的替代解释变成**可测量的对照**，而不是答辩时的辩解。
- state-dict 级差异化让三臂共享 `model_code_sha256`，父全部可加载，
  避开了 wiring 轮的父封锁——不需要再做一次父迁移。
- `parameter_delta` 三臂为 0，容量混杂被结构性排除，这比事后统计控制强得多。
- 首次把城市建成区/建筑高度/人口密度/土地覆被放进神经天气模型。四类在全部七篇已核论文里零命中。

**负面（如实列，R-033）**

1. **固定随机投影没有可解释性。** 192 维码由 `randn` 固定矩阵生成，即使 (a)(f) 都真，
   也不能声称「建成区比例是主因」。**代价**：逐变量解释需要可训练读出，而可训练读出会让
   `parameter_delta > 0`，重新引入容量混杂，必须另立 ADR——本轮拿不到这个答案。
2. **placebo 可能比 control 更差。** 逐格置换后某些格点的协变量与地形完全不符，
   模型会学到「这些码无意义」。**代价**：若如此，(a) 的配对会**高估** treatment 的增益。
   缓解：(a) 只要求 treatment ≥ control；(f) 比的是 treatment vs placebo，两臂都有码表，
   噪声代价对称——这正是为什么 (f) 必须是主判据。
3. **静态场是派生统计量，不是观测真值。** 登记记录逐字：「a derived statistic of real native
   observations … it is NOT interpolation and MUST NOT be used as an observational truth field」。
   **代价**：本轮结论的适用域是「这组发布协变量」，**不是**「城市化的真实大气效应」。
   论文里不得写成后者。
4. **LandScan 缺失，人口维度无交叉校验。** ORNL 账号门槛，记 `failed-no-fallback`。
   **代价**：`pop_density_km2` 只有 GHSL 一个来源，误差未量化，无法做双源一致性。
5. **GHSL 是五年间隔跨传感器合成。** `built_surface_frac_2020` 与 `pop_density_km2_2020` 都不是年观测。
   **代价**：不能称其为「2020 年真值」，只能说「该历元产品」。
6. **`lc_*` 五个份额不求和为 1**（未映射份额约 0.23–0.26，WorldCover 无全海洋 3° 瓦片）。
   **代价**：`lc_bare_frac` 是「裸地在已映射类别中的占比」，不是裸地绝对占比。
7. **4,000 updates 的剂量论证借自 Stage A，不是本轮自证。** 若静态协变量的适应期更长，
   (a) 的阴性可能是剂量不足。**代价**：这是本轮唯一无法用自身读数排除的假阴性来源；
   (f) 部分缓解（两臂同剂量），但不能完全排除。
8. **n = 1 模型、单实例、单分辨率、单区域（27–43°N / 107–123°E）。**
   与 C1/C2/R/A 相同的适用域限制，不得外推到全球。
9. **本条不保证有正面结果。** 若 (a)–(c) 任一为假，或 (a) 真而 (f) 假，如实记负并停。
   若本轮也负，则输入侧与损失侧两侧的主要方向都已测过，论文的正面结果只能靠
   **证伪章与机制发现**——这一点必须在动手前就承认，不能等到读数出来再改叙事。

## 与其它决策的关系

- **受约束 0069**：接线形态（时不变、零参数、加性逐 token）一字不改。
  本条只改**臂的构造方式**（state dict 而非开关），不改接线本身。
- **受约束 0063**：静态产物的获取、身份与「派生统计量非真值」的定性一字不改。
  本条不重新下载任何数据。
- **不冲突 0049**：0049 关闭的是**线性冗余**（tendency = `X_t − X_{t−1}`）。
  静态协变量是 `S(x, y)`，**不能由 17 个动态通道在任何变换下导出**。
  这是信息论差别，不是名词差别。
- **不冲突 `record:s3-typed-spatial-result`**：那一轮测的是**同期可导出**的非线性变换
  （散度/涡度/平流/静力稳定度），B→C `|Δrmse| ≤ 3.2e-6 K`。本轮输入与 `X` 无函数关系。
- **不解除任何封锁条款**：C1/C2/Stage P/R/C/A、fine-tuning-only 空间梯度、tendency 输入、
  物理诊断量输入，全部维持关闭。
- **不授权**把 `projection` 改成可训练、不扫权重、不加 seed。

## 恢复第一条命令

```bash
.venv/bin/python tools/r7_round.py state
```
