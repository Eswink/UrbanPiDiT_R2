# 0032 — 城市静态协变量的三臂消融：把「用但未消融」的文献空白变成一次可证伪测量

- **日期**：2026-11-13
- **状态**：**定稿归档**（archived）；**未执行**，见文末 `## 实际结果`
- **分支**：`r7/weather-reasoning`；起点 HEAD `8b5e8c0`
- **`scientific_claim: false`**、`test_read: false`、`r = 0`
- **上游**：`docs/plans/0030-thesis-physics-informed-mainline.md`（物理信息主线）、
  `docs/plans/0031-physics-informed-sequence-c-a-b.md`（阶梯 C→A→B）、
  决策 **0063**（静态数据获取）、**0069**（接线，已 `accepted`）、**0097 D1.d**（父迁移）、
  **0112**（序列准入，Step A 已负）
- **一手文献**：`arXiv:2212.12794`（GraphCast）、`2211.02556`（Pangu-Weather）、
  `2202.11214`（FourCastNet）、`2304.02948`（FengWu）、`2301.10343`（ClimaX）、
  `2405.13063`（Aurora）、`2406.01465`（AIFS）、`2409.02120`（LiDAR 城市形态降尺度）
  —— 全部 2026-11-13 由 `arxiv-researcher` 读正文
- **唯一执行源**：`docs/goals/s3-urban-static-ablation.md`

---

## §0 摘要：为什么这一轮不是第七次换名

### §0.1 六轮物理信息全部失败，且全部在**损失侧**

| 轮次 | 形式 | 结果 | 成本 |
| --- | --- | --- | --- |
| C1 | 无目标约束束 | 真值被罚 10.2×，8/8 方向反 | 0 GPU-h（前置否决） |
| C2 | 导数域目标匹配 | 15/15 (lead,seed) 全负 | 66.99 GPU-h |
| Stage P | 谱几何前置 | P1 判 `skip-q`、P2 三方向全不过 | 0 GPU-h |
| Stage R | 后处理振幅校正 | 85/85 (变量,lead) 全劣化 | 0 GPU-h |
| Stage C | 缺口衰减律 + β_j | α=0.051（减半需 77.5 万倍训练） | 0 GPU-h |
| Stage A | β_j AMSE-analogue 归一化 | (a) 振幅三 seed 一致升 **+0.114…+0.125**，但 skill 9/9 全降 | 21.83 GPU-h |

Stage A 是最干净的诊断：**方法精确地起了它声称的作用（抬振幅），而那件事本身有害**。
它排除了「剂量不足」这个假阴性解释——因为直接作用点已经三 seed 一致地移动了。

**这六轮的共同结构是：它们全都改了「按什么算分」，没有一个改了「模型能看到什么」。**

### §0.2 输入侧还剩什么没测

| 输入侧分支 | 状态 | 依据 |
| --- | --- | --- |
| 显式 tendency 输入 | **关闭** | 决策 0049 —— 线性冗余被预先承认 |
| 4 个物理诊断量作输入 | **已测，无差别** | `record:s3-typed-spatial-result`：B→C Δrmse `+1.2e-7 … +3.2e-6 K`（float32 噪声级） |
| **静态地表协变量** | **接线已通，但从未做过科学规模的消融** | 见 §0.3 |

### §0.3 文献结论（2026-11-13 一手核对，逐字）

**Q1 —— 七篇主流神经天气模型的静态输入集合：城市/土地覆被零命中。**

| 论文 | 静态输入清单（逐字核对） | 含城市/LULC |
| --- | --- | --- |
| GraphCast `2212.12794` §3.3 | 「the binary land-sea mask, the geopotential at the surface, the cosine of the latitude, and the sine and cosine of the longitude」 | **否** |
| Pangu-Weather `2211.02556` §III-B | 「three constant masks (i.e., the topography mask, land-sea mask and soil type mask)」 | **否** |
| FourCastNet `2202.11214` §2.2 | 仅 20 个预报变量，**无静态输入** | **否** |
| FengWu `2304.02948` §2.1 | 仅 189 个 predictand，**无静态输入** | **否** |
| ClimaX `2301.10343` App. Table 9 | 「Static: Land-sea mask (LSM)」「Static: Orography」 | **否** |
| Aurora `2405.13063` App. §8.1 | 「three static variables … geopotential at the surface … land–sea mask … soil-type」 | **否** |
| AIFS `2406.01465` §3 Table 1 | 「Land-sea mask, orography, standard deviation of sub-grid orography, slope of sub-scale orography, insolation, latitude/longitude, time of day/day of year」 | **否** |

**Q2 —— 「有静态 vs 无静态」的消融：零命中。**

GraphCast 消融的是 multi-mesh（§7.3.1）；FengWu 消融的是 replay buffer（§4.4）；
Aurora 消融的是训练数据多样性（§15）。**没有一篇消融过静态输入本身。**

⇒ 五篇使用静态特征的论文全部是「**使用但未消融**」。这是一个可直接引用的文献空白，
而不是我们编的动机。

**Q3 —— 注入方式无论在文比较。** GraphCast（`[x^{t-1}, x^t, f^{t-1}, f^t, f^{t+1}, c_i]` → MLP）、
Pangu（追加到地表变量）、Aurora（「appending them to the collection of surface-level variables」）、
AIFS（同一线性层）——全部是输入层拼接，**无一比较过替代注入方式**。

**Q4 —— 城市尺度最近邻是 `2409.02120`**：用 LiDAR 提取的建筑形态（H、BR、SVF、CCR…）做气温
统计降尺度到 5 m。但它是 **LightGBM 不是神经天气模型**、**单城市（阿姆斯特丹）**、
**无静态特征消融**、以 UrbClim 100 m 输出为 pseudo truth。属弱相关，只能当动机引用。

### §0.4 所以这一轮的增量是什么

**把「七篇主流模型都用了静态特征、没有一篇消融过」这个文献空白，做成一次我们自己的可证伪测量；
并且第一次把「城市建成区 / 建筑高度 / 人口密度 / 土地覆被」这四类在任何神经天气模型里都没用过的
协变量放进去。**

这不是「换了个输入名词」——它的族判别是可检验的（§2.2）。

---

## §1 起点（可核查，2026-11-13 实测）

| 项 | 值 |
| --- | --- |
| HEAD | `8b5e8c0`（工作树干净） |
| `model_code_sha256` | `59340710623ca24c7ee59ebb5711f52193ebfbaa72ed4532b41bdd987f5d9324` |
| 父 pin `MODEL_CODE_SHA256` | `59340710…` —— **与当前 digest 相等** |
| 父 checkpoint | `outputs/r7_s3_position_retention_parent_migrated_20261103_attempt01/seed{41,42,43}/parent_update_0001600.pt` |
| 父 sha256 | `e81e4904…` / `7e2cd4cc…` / `ee33f690…`（三 seed 各自 pin） |
| `static_identity` | `8671ed7ce0c82d1ec9ac834e68e2e3c7bbc30e14fa6b8a07501e070d27708b03` |
| 静态产物 | `outputs/r7_urban_static_covariates_attempt01/published/`（`static_fields.npz` 8 层 65×65 float32） |
| `used_gpu_h` | `593.325957`（**仅会计**，决策 0030/0038） |
| 2023 test | **未评分**，`r = 0` |

**基础设施已就绪，本轮不新建模型代码**：

| 组件 | 位置 | 状态 |
| --- | --- | --- |
| `StaticConditioning` | `model/static_conditioning_r7.py:157` | 时不变、**零训练参数**（field/projection 都是 buffer） |
| `static_features` | `model/static_conditioning_r7.py:128` | 逐 token patch 平均 + 中心化 + 平方展开，`[N, 2K]` |
| `load_static_fields` | `model/static_conditioning_r7.py:103` | 读发布产物并**校验 `static_identity`** |
| 接线点 | `model/weather_forecaster_r7.py:149-152` | `context = context + static_conditioning_pathway(token_hw)` |
| 父 reconcile | `tools/reconcile_r7_static_parent.py:239` | 已实测 `strict_load_of_reconciled_passed: true`（157 个张量） |
| 两臂驱动 | `scripts/study_r7_static_covariates_wiring.py` | 575 行，结构可派生 |

### §1.1 上一轮（wiring）明确留下的两个坑

`docs/R7_STATIC_COVARIATES_WIRING.md` 与本轮起点直接相关，逐条引自该文件的登记记录：

**坑 ①（阻断级，但已被后续轮修掉）**：

> 「this round moves `model_code_sha256` from `60de3b2d…` to `a38be16c…`, and `load_checkpoint`
> refuses a foreign implementation by design, so **every registered migrated parent is unloadable
> under this code**; R2–R5 need a fresh parent migration, which this round did not perform」

**这条今天不再成立**：`tools/reconcile_r7_static_parent.py` 已在 `0097 D1.d` 下跑过两次，
receipt `outputs/r7_static_parent_reconcile_20261108_attempt02/receipt.json` 逐字记录
`"strict_load_of_reconciled_passed": true`、`"parent_model_code_sha256": "59340710…"`（等于当前）。
⇒ **on 臂可以由「父 state + 2 个 buffer」严格加载；本轮不需要再做父迁移。**

**坑 ②（本轮必须正面处理的科学缺陷）**：

> 「the static term is a fixed projection over a fixed polynomial expansion, so `parameter_delta` is
> exactly 0; that removes the capacity confound but also means the path cannot learn which covariate
> matters, so **an on-arm null is not evidence the covariates are uninformative**」

如实承认：`StaticConditioning` 的 `projection` 是 `torch.randn(16, 192, generator=seed(0))/√16`
的**固定随机矩阵**（`model/static_conditioning_r7.py:178-184`，`register_buffer` 非 `Parameter`）。
所以任何空间变化的输入都会给出**互不相同**的逐 token 码。

⇒ **如果 on 臂赢了，第一个替代解释就是「它只是一张位置查找表」**——而模型**已经有**
`position_features()` 的 `sin/cos(lat)`、`sin/cos(lon)` 四个谐波（`model/spacetime_conditioning_r7.py:217`）。
**这个替代解释必须被实验排除，否则本轮结论不可发表。** §2.3 的 placebo 臂就是为此设的。

---

## §2 本轮设计

### §2.1 唯一改动因子，三臂

**三臂共享同一份代码、同一份父权重、同一个 `model_code_sha256`、同一个 seed、同一批更新。**
差别**只在两个 buffer 的取值**：

| 臂 | `field`（8 个协变量） | `projection` | 语义 |
| --- | --- | --- | --- |
| **control** | 不构造（`static_conditioning=False`） | —— | 17 通道基线，逐位等于改动前行为 |
| **treatment** | 发布的 8 层真实静态场 | 发布投影 | 真实城市/土地覆被信息 |
| **placebo** | 同一份 8 层场**逐格空间置换**（固定种子） | **同一份**发布投影 | 同等码空间、同等容量、**空间结构被破坏** |

**为什么 placebo 必须空间置换而不是打乱通道标签**：打乱通道会改变每个 token 的码的分布；
逐格置换保持**每个协变量的边缘分布完全不变**（同一个多重集），只破坏空间自相关。
这样 placebo 与 treatment 的差别**只有一个**：协变量与地理位置/地形是否一致。

**为什么不能只做 on/off（两臂）**：坑 ② 说得很清楚，on 臂的赢无法归因。三臂才能把
「有位token码就有用」与「这些码携带真实地表信息才有用」分开。

**`parameter_delta` 三臂恒为 0**：`field` 与 `projection` 都是 buffer。这是本设计的核心优势——
它把容量混杂彻底移除，使得任何 skill 差异**不可能**来自多出来的参数。

### §2.2 族判别（J4）：与两个已关闭输入侧分支的可检验差别

| 分支 | 关闭依据 | 与本轮的关系 |
| --- | --- | --- |
| 显式 tendency 输入 | 决策 0049:55 逐字「显式 tendency 的**线性冗余**被预先承认」 | tendency 是 17 通道的**线性变换**，不含新信息 |
| 4 个物理诊断量作输入 | `record:s3-typed-spatial-result`：B→C `|Δrmse| ≤ 3.2e-6 K` | 诊断量是 17 通道的**非线性变换**，但是**同期可导出**的 |
| **本轮：静态地表协变量** | 未测 | **时不变且外生**：`lc_built_frac`、`building_height_m`、`pop_density_km2` 等**不能由 17 个动态通道在任何变换下导出**——它们描述的是下垫面，不是大气状态 |

**这是三者唯一真正的信息论差别**：前两个分支的输入都是 `f(X_t, X_{t-1})` 的形式，
本轮输入是 `S(x, y)`，与 `X` 无函数关系。**这不是修辞，是可以从数据直接检验的**：
把 17 通道送进任何回归器都预测不出建成区比例（下垫面属性不由当日大气决定）。
这一点必须写进 protocol 的 `family_argument` 并附上可核查的申请（§4 的 (f) 判据）。

### §2.3 预声明出口（任一为假即永久关闭该支）

| 判据 | 要求 | 为何必须存在 |
| --- | --- | --- |
| **(a)** | 72 h t2m skill 三 seed 配对 Δ（treatment − control）**全 ≥ 0** | 主结论 |
| **(b)** | 24/48 h skill 配对 Δ 三 seed **全 ≥ 0** | 中期一致性 |
| **(c)** | 6/12 h skill 配对 Δ **不劣化**（Δ ≥ −0.005，容许短期微降） | 短期的合理容差，冻结前写定 |
| **(d)** | control 臂**逐位等价**于改动前 | 由「`static_conditioning=False` 时 `static_conditioning_pathway=None`」这条代码路径 + 会失败的反证测试支撑，**不由读数自证**（0109 的教训：`c=True` 是字面量） |
| **(e)** | 预检：8 层静态场在 65×65 全有限；`static_identity` 校验通过；三臂 `parameter_delta == 0`；placebo 的逐变量边缘分布与 treatment **逐位相同**（只置换位置） | 防止 placebo 不是真 placebo |
| **(f)** | **归因判据**：`Δ(treatment − placebo)` 在 72 h **≥ 0.001 K**（或至少：treatment 优于 placebo） | **这是本轮的科学核心**。若 treatment 只与 control 有差、与 placebo 无差，则结论只能是「多一张位置查找表有用」，而模型已有位置编码 ⇒ 按预声明关闭该支 |

**(f) 的方向必须说清**：它是**归因**判据不是**成功**判据。若 (a) 真而 (f) 假，本轮**记为负**并
写明「静态协变量的效应与空间结构无关，等价于位置码」——这仍然是一个可发表的**机制负结果**，
比一个无法归因的正结果有价值。

### §2.4 剂量与预算（冻结前写定）

- **3 臂 × 3 seed × 4,000 updates**，从 pin 的 p1600 父微调（父已可加载，见 §1.1 坑①）
- batch 4，recipe 逐字沿用注册配方
- 实测速率参考：wiring 轮 2.51–2.58 s/update（CPU 上 120 updates）。Stage A 实测
  1.1096 s/update 量级。按 3 臂 × 3 seed × 4,000 = 36,000 updates 估 ≈ **11–26 GPU-h**
- **软预算 115,200 s / 硬上限 230,400 s**（与 Stage A 同口径，决策 0030）

**为什么 4,000 updates 够**：Stage A 用同样预算让直接作用点（细带振幅比）产生三 seed 一致的
移动（+0.114…+0.125）。若本轮 (a) 为阴性，**不能再以「训练太短」解释**——
这是 Stage A 花钱买来的、可以复用的排除。

---

## §3 实施顺序

1. **P0 —— 预检（0 GPU-h，必须先全绿）**
   - `static_identity` 校验、8 层 65×65 有限
   - 三臂 `parameter_delta == 0` 逐臂实测
   - placebo 构造：逐格空间置换，**逐变量多重集与 treatment 逐位相同**（排序后比较）
   - 父加载：三 seed × 三臂全部 `strict=True` 通过
   - **(f) 判定所需的数据不在此处算**（它要等读数），但 placebo 的正确性在这里冻结
2. **P1 —— 冻结协议**（`protocol.json` + `protocol_sha256`，含 `static` 段与 `family_argument`）
3. **P2 —— 归档代码身份**（`code.zip`，带 commit 注释）
4. **P3 —— 3 臂 × 3 seed 训练**（`fine_tune_long_rollout`，父权重从 pin 载入）
5. **P4 —— 五 lead 评分**（6/12/24/48/72 h，val 2022）→ `boundary_rmse.csv`
6. **P5 —— 读数与判定点**：逐条写 (a)–(f) 真或假
7. **P6 —— 证据页 + 四件套登记**

---

## §4 成本表

| 阶段 | 内容 | GPU-h |
| --- | --- | ---: |
| P0 | 预检（CPU） | 0 |
| P3 | 3 臂 × 3 seed × 4,000 updates | ≈11–26（按 20 估） |
| P4 | 五 lead × 3 臂 × 3 seed 评分 | 含在上项 |
| **合计** | | **≈20（软预算 32 h / 硬上限 64 h）** |

**参照**：C2 用 66.99 GPU-h 换来一个否定结论；Stage A 用 21.83 GPU-h 换来一个**可归因的**否定结论
（方法起作用了，那件事有害）。本轮 20 GPU-h 的预算与之同量级，且有 (f) 保证归因。

---

## §5 明确不做

| 不做 | 理由 |
| --- | --- |
| 重开 C1 / C2 / Stage P / Stage R / Stage C / Stage A | 全部按预声明关闭（0084 §6、0094、0109、0111、0112） |
| 重开 fine-tuning-only 的空间梯度支 | 其结论成立，且已被 FastNet §4.2.1 解释 |
| 重开显式 tendency 输入 | 决策 0049，线性冗余已承认 |
| 重开 4 个物理诊断量作输入 | `record:s3-typed-spatial-result` 已测，float32 噪声级 |
| 改 `model/**` 任何文件 | 会改 `model_code_sha256` ⇒ 父再次不可加载（wiring 轮坑①）。**三臂靠 state dict 区分，不靠开关** |
| 把 `projection` 改成可训练 | 会让 `parameter_delta > 0`，重新引入容量混杂 |
| 重建 store / 新增数据 / 改 `data_identity` | 静态场已在 `attempt01/published/`，只读 |
| 用多期数组做时变轮 | 那不再是「时不变条件」，属另一个轴 |
| 扫加权、加 seed、改 4,000 updates | 冻结前定死，事后不放宽 |
| 读 2023 test | `r = 0` |
| 承诺一定有正面结果 | 若 (a)–(c) 任一为假，如实记负并停 |

---

## §6 风险（如实列）

1. **固定随机投影没有可解释性。** 192 维码由 `randn` 固定矩阵生成，无法读出「哪个协变量重要」。
   **代价**：即使 (a)(f) 都真，也不能声称「建成区比例是主因」。缓解：(f) 只判「空间结构是否有用」，
   不判逐个变量；逐变量的解释留给后续轮（需要可训练读出，会引入容量混杂，须另立 ADR）。
2. **placebo 的置换可能破坏不止空间结构。** 逐格置换后，某格的建成区比例可能与地形完全不符，
   模型在训练中会学到「这些码无意义」——这正是 placebo 该有的行为，但它也意味着 placebo 臂的
   损失可能**比 control 更差**（学到了噪声）。**代价**：若如此，(a) 的配对会**高估** treatment 的增益。
   缓解：判据 (a) 只要求 treatment ≥ control；(f) 才是主判据，且 (f) 比的是 treatment vs placebo，
   两者都有一张码表，噪声代价对称。
3. **静态场是派生统计量，不是观测真值。** 记录逐字：「a derived statistic of real native observations
   aggregated to the 0.25 deg cell edges; it is NOT interpolation and MUST NOT be used as an
   observational truth field」。**代价**：本轮结论的适用域是「这组发布协变量」，不是「城市化的真实效应」。
4. **LandScan 缺失，人口维度无交叉校验。** 决策 0063 的失败记录：ORNL 账号门槛，`failed-no-fallback`。
   **代价**：`pop_density_km2` 只有 GHSL 一个来源，其误差未量化。
5. **GHSL 是五年间隔跨传感器合成，不是年观测。** `built_surface_frac_2020` 与 `pop_density_km2_2020`
   均如此。**代价**：不能把它们当作 2020 年的真值，只能说「该历元的产品」。
6. **n = 1 模型、单实例、单分辨率、单区域。** 与 C1/C2/R/A 相同的适用域限制。
7. **`lc_*` 五个份额不求和为 1**（未映射份额约 0.23–0.26）。**代价**：`lc_bare_frac` 不是「裸地占比」，
   是「裸地在已映射类别中的占比」，解读时必须带上这层。
8. **4,000 updates 的剂量论证借自 Stage A。** 若静态协变量的适应期比振幅项更长，(a) 的阴性
   可能是剂量不足。**代价**：这是本轮唯一无法用自身读数排除的假阴性来源；(f) 部分缓解——
   因为 (f) 比的是同剂量的两臂。

---

## §7 workflow 与提示词

执行形态是**一份 goal 长文 + 一个派生 workflow**，两者合在 objective 里
（照 `docs/goals/s3-throughput-and-physics-campaign.md:18` 的形态）。

- `docs/goals/s3-urban-static-ablation.md`，头部 `<!-- round-node: S3 -->` + `<!-- round-spec: {...} -->`
- `.zcode/workflow-drafts/r7-urban-static-ablation.dwf.ts`，从 `r7-campaign-round.dwf.ts` 派生
- **必须用 `CreateWorkflow {path, args}`，不要用 `AmendWorkflow`**——实测 `AmendWorkflow` 丢 `args`
  让 `dryRun` 静默变 true
- 零成本编译校验：`CreateWorkflow` + `args: {maxRounds: 0, dryRun: true}`

---

## 实际结果（Actual Results）

**NOT EXECUTED YET —— 本轮只归档设计、brief 与 workflow 草稿；未训练、未评分、未产出读数。**

- GPU-h = **0**
- 2023 test **未读**，`r = 0`
- 未写 `protocol.json`、未生成 `code.zip`、未产生 `readings.json`
- `docs/goals/s3-urban-static-ablation.md` 的 (a)–(f) 全部**未判定**

**在此之前，任何引用本文件作为「静态协变量有效/无效」的证据都是错的。**
本文件目前的全部内容是：一个设计、一份文献空白的一手核对、以及一个预声明的可证伪判据集。
