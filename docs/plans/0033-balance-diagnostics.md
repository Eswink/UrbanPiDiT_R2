# 0033 — 物理平衡诊断：把「RMSE 好但物理不可信」做成一次大样本测量

- **日期**：2026-11-14
- **状态**：**定稿归档**（archived）；**P0 预检已实跑**，P1–P5 **未执行**，见文末 `## 实际结果`
- **分支**：`r7/weather-reasoning`；起点 HEAD `78cd65a`
- **`scientific_claim: false`**、`test_read: false`、`r = 0`
- **上游**：`docs/plans/0030-thesis-physics-informed-mainline.md`（物理信息主线）、
  `docs/plans/0031-physics-informed-sequence-c-a-b.md`（阶梯 C→A→B）、
  `docs/plans/0032-urban-static-covariate-ablation.md`（三臂消融）、
  决策 **0110**（区域分块分解）、**0112**（序列准入）、**0113**（静态协变量准入）
- **一手文献**（2026-11-14 读正文）：`arXiv:2604.01454`（Pasquini et al. 2026，区域 DL 模型的
  物理平衡评估）、`arXiv:2309.08473`（Bonavita 2023，DL 天气模型的保真度与物理一致性）、
  `arXiv:2501.19374v2`（AMSE / double penalty）、`arXiv:2506.11987`（神经天气模型的误差诊断）
- **唯一执行源**：`docs/goals/s3-balance-diagnostics.md`

---

## §0 摘要：为什么这一轮和前面七轮不是同一个问题

### §0.1 七轮负结果的共同结构

| 轮次 | 改了什么 | 结果 | 成本 |
| --- | --- | --- | --- |
| C1 | 无目标约束束 | 真值被罚 10.2×，8/8 方向反 | 0 GPU-h |
| C2 | 导数域目标匹配 | 15/15 (lead,seed) 全负 | 66.99 GPU-h |
| Stage P | 谱几何前置 | P1 判 `skip-q`、P2 三方向全不过 | 0 GPU-h |
| Stage R | 后处理振幅校正 | 85/85 (变量,lead) 全劣化 | 0 GPU-h |
| Stage C | 缺口衰减律 + β_j | α=0.05111，减半需 775,102× 训练量 | 0 GPU-h |
| Stage A | β_j AMSE-analogue 归一化 | 振幅三 seed 升 +0.114…+0.125，skill 9/9 全降 | 21.83 GPU-h |
| 城市静态 | 三臂输入消融 | (a) 假 ⇒ 按预声明关闭 | 20.46 GPU-h |

**共同结构：七轮全都在改「按什么算分」或「模型能看到什么」，没有一轮问过「模型算出来的场在物理上是否可信」。**

这不是修辞。Stage A 是最清楚的证据：它把细带振幅精确地抬高了 12%，而 skill 因此全面下降。
**一个能精确执行指令却让结果变差的方法，说明「振幅」这个量本身不是瓶颈**——但七轮里没有任何一次
测量去问「被抬高的那部分振幅，是不是物理上本来就不该有的」。

### §0.2 文献给出的答案（逐字，2026-11-14 读正文）

**`arXiv:2309.08473`（Bonavita）** 摘要：

> "Pangu-Weather forecasts, and possibly those of similar ML models, **do not have the fidelity and
> physical consistency of physics-based models** and their advantage in accuracy on traditional
> deterministic metrics of forecast skill can be at least partly attributed to these peculiarities."

**`arXiv:2604.01454`（Pasquini et al. 2026）**——目前唯一一篇系统评估**区域** DL 天气模型物理平衡的工作。
它对比 Bris（Met Norway 拉伸网格，2.5 km）与 MEPS 控制预报，评估四个平衡：

| 平衡 | 公式（逐字抄自正文） | 章节 |
| --- | --- | --- |
| 静力 | `(∂Φ/∂p)_HB = −R T_v / p`；`dΦ_HB = −R·T̄_v/p̄·(p_{i+1}−p_i)`；`δ_HB = (dΦ − dΦ_HB)/dΦ` | §3.1 式 (1)–(3) |
| 地转 | `v_g = (1/f) k̂ × ∇_p Φ`；`Ro = ‖(v·∇_p)v‖/(f‖v‖)`；`ws_ag = ‖v − v_g‖`，限 `Ro < 0.1` | §3.2 式 (4)–(6) |
| 梯度风 | `u_θ²/r + f u_θ = ∂Φ/∂r`，用 `N,M ≤ 5` 的截断径向-方位角多项式拟合 | §3.3 式 (7)–(8) |
| 连续方程 | 垂直积分的散度 | §3.4 |

外加 **§4.6 谱分析**：`std(Φ_f1 − Φ_f13)` 作为细尺度噪声代理，以及 FFT 振幅谱 vs 波长。

它的结论：

> "despite its relatively good performance in terms of RMSE, Bris struggles to capture important
> mesoscale features of the event and that **it significantly disrupts several atmospheric balances**.
> This unrealistic disruption is mainly linked to the fine-scale noise evidenced in its output fields"

以及一条对本仓直接适用的观察（§4.6）：

> "at large wavelengths (30-300 km), the DDM spectra of the horizontal wind components (u and v) are
> **markedly underestimated**. In contrast, the amplitude spectra of the horizontal winds and the
> geopotential are **significantly amplified at smaller wavelengths**"

### §0.3 文献的局限，以及我们的增量

**该文的 n = 1**：一个风暴（Poly，2023-07-05）、一个 lead（+6 h 与少数几个）、一个 443×443 子域。
它的四个平衡是**案例描述**，不是统计量。

**我们能做的**：
- 全年 2022 val：**1447 个初始时刻 × 5 个 lead**
- 两组受控臂：Stage A 的 on/off（振幅被精确抬升 12%）与城市静态的 control/treatment/placebo
- 3 seed × 15 个已训练的 checkpoint，**不需要重新训练**

⇒ 可以把「物理不平衡」从**一个风暴的描述**变成**一个可做统计的分布**，并且第一次能问：

> **被 Stage A 抬高的那 12% 振幅，是否正是物理上不该有的那部分？**

这个问题用文献的方法可以回答，用任何 RMSE 都回答不了。

---

## §1 起点（可核查，2026-11-14 实测）

| 项 | 值 |
| --- | --- |
| HEAD | `78cd65a`（工作树干净） |
| `model_code_sha256` | `59340710623ca24c7ee59ebb5711f52193ebfbaa72ed4532b41bdd987f5d9324` |
| 数据实例 | `outputs/r7_s3_fy_multiyear_20261023_attempt01/store/cache.zarr`，`[34084,17,65,65]` |
| 通道单位 | `z*` 是 **位势 `m**2 s**-2`**（不是米）；`t*` 是 K；`q*` 是 `kg kg**-1`；`mslp` 是 Pa |
| 归一化 | 物理单位保留（`physical_units_retained: True`），归一化在读时施加 ⇒ **档位读出的就是物理单位** |
| val manifest | `outputs/.../store/manifests_treatment/val.jsonl`（451,980 B，2022 全年） |
| 可用 checkpoint | static ablation 9 个 + Stage A 6 个 = **15 个**，3 seed |
| `used_gpu_h` | `613.790428`（**仅会计**） |
| 2023 test | **未评分**，`r = 0` |

### §1.1 一个致命的历史事实：预报场从未落盘

三路调查一致确认：`find outputs -path "*/evaluation/*" -type f ! -name "*.csv" ! -name "*.json"` → **0**。

`training/r7_boundary_metrics.py:99` 的 `sum((-1,-2))` 把空间两轴立即求和坍缩，逐格误差场在那一行之后
永久消失。`training/r7_rollout_metrics.py:75` 的注释写得很直白：

> `# Only compact [L,C] sufficient statistics survive update().`

⇒ **任何空间型指标（平衡、谱、FSS、双惩罚）都无法从已有产物重算，必须重推理。**
但 checkpoint 全在，所以这是 **≈6.5 GPU-h 的推理**，不是重新训练。

---

## §2 P0 预检已执行（2026-11-14，0 GPU-h，真实 store 数据）

这一步在写文档之前就跑完了，因为诊断量必须先证明它在真实场上有意义。

### §2.1 静力平衡偏差 `δ_HB`（850–500 hPa，含虚拟温度）

| frame | median | mean | std |
| ---: | ---: | ---: | ---: |
| 1500 | **+0.02611** | +0.02632 | 0.00353 |
| 5000 | **+0.02144** | +0.02118 | 0.00364 |
| 12000 | **+0.02915** | +0.02972 | 0.00368 |
| 25000 | **+0.02753** | +0.02743 | 0.00472 |

**这是一个系统偏置，不是噪声**：`mean ≈ median`、`std ≈ 0.004` 比 `|median| ≈ 0.026` 小 6 倍。

**偏置来源已查清**：把测得的平均厚度 3994.84 m 代回 hypsometric 关系，反推
`ln(p1/p2) = 0.53283`，而真实的 `ln(850/500) = 0.53063` ⇒ **隐含上层气压 498.90 hPa**，
与 500 hPa 差 0.22%。**这说明 z 与 t 的层定义一致，+2.8% 是真实的大气偏离**（ERA5 分析场含非静力结构，
且 `z` 与 `t` 是从不同模式层插值到气压层的），不是单位错误。若 `z` 被除以 `g`，隐含上层气压会变成
约 780 hPa——明显错误，这正是该检查的牙齿所在。

**干/湿对照**：frame 1500 干空气 +0.02658 vs 含 T_v +0.02611；frame 5000 干 +0.02388 vs 湿 +0.02144。
湿度只贡献约 0.002，主偏置不来自 T_v。

### §2.2 地转平衡（真值分析场本身，frame 1500）

| 层 | `Ro < 0.1` 覆盖率 | `ws_ag` 中位数 | 平均风速 | 比值 |
| --- | ---: | ---: | ---: | ---: |
| 850 hPa | 10.5% | 3.12 m/s | 7.17 m/s | **43.5%** |
| 500 hPa | 15.8% | 4.48 m/s | 27.83 m/s | 16.1% |
| 250 hPa | 6.8% | 8.66 m/s | 57.12 m/s | **15.2%** |

**覆盖率是这一条最重要的读数**：`Ro < 0.1` 只覆盖 6.8%–15.8% 的格点。这意味着地转偏差统计
**只建立在约十分之一的格点上**，必须在证据页里逐层报出，不能当成全域结论。

### §2.3 连续方程残差与谱

- 柱积分散度残差：`median = −1.478e-07 s**-1`，`mean|r| = 3.317e-05 s**-1`，全有限
- `z500` 的 f13 细尺度噪声：**39.97**（位势单位，即约 4.07 m 等效高度）
- 谱：46 个径向 bin，最短波长 **40.2 km**（0.25° 网格的 Nyquist 约 2×0.25°×111 km ≈ 55 km，
  径向 bin 的最小值是 40.2 km，说明对角方向的 bin 也在）

### §2.4 单位反证（避免重演 224× 事故）

`scripts/r7_s3_operator_j1_probe.py:70-90` 记录过一次真实的 224× 单位事故：符号写反让
36.6 m 的残差变成 8193 m，会凭空制造一个「分析场严重违反静力平衡」的结论。
本轮把这条固化成测试 `test_the_unit_trap_inflates_the_deviation` 与
`test_an_era5_like_column_reproduces_its_own_layer_pressures`。

---

## §3 诊断量的定义（全部照文献，不发明）

### D1 — 静力平衡偏差 `δ_HB`

```
T_v = T (1 + 0.61 q)                                  # 式 (1)
dΦ_HB = −R · T̄_v / p̄ · (p_{i+1} − p_i)                # 式 (2)
δ_HB  = (dΦ − dΦ_HB) / dΦ     其中 dΦ = z_upper − z_lower   # 式 (3)
```

**层对**：`850–500`（有 q，用 T_v）与 `500–250`（**store 没有 t250/q250**，只能用干温度。
这是一个必须在证据页写明的近似，不是一个可以忽略的细节）。

**单位纪律**：`z*` 是位势 `m**2 s**-2`，`R·T` 也是 `m**2 s**-2`，所以 `δ_HB` 无量纲且**不得除以 g**。
`geopotential_to_height()` 只用于报告米数，**绝不能出现在偏差计算里**。

### D2 — 地转平衡偏差 `ws_ag`

```
v_g  = (1/f) k̂ × ∇_p Φ                                # 式 (4) ⇒ u_g=−(1/f)∂Φ/∂y, v_g=(1/f)∂Φ/∂x
Ro   = ‖(v·∇_p)v‖ / (f‖v‖)                            # 式 (5)
ws_ag = ‖v − v_g‖     仅在 Ro < 0.1 的格点上            # 式 (6)
```

`f = 2Ω sin φ`。**`f` 在本仓活跃代码里此前零出现**，必须新写。赤道附近 `f → 0`，代码**拒绝**而不是除零
（`test_the_equator_is_refused`）。

**物理验证已做**：在 store 自己的纬向网格上，`u_g` 实测 67.19 m/s vs 解析 67.12 m/s（0.1% 差，
来自 f 随纬度变化）。符号经三个方向确认：NH 向北降压 ⇒ 西风；反号场 ⇒ 东风。

### D3 — 连续方程残差

`D850 + D500 + D250`（垂直积分散度）。复用 `scripts/r7_s3_operator_j1_probe.py:93` 的同一归约，
两个实现不可能不一致。

### D4 — 谱与细尺度噪声（文献 §4.6）

- **噪声代理**：`std(field − box_mean(field, 13))`
- **振幅谱**：径向平均 `sqrt(mean(|F|²))` vs 波长，DC bin 标为 `inf` 波长

**两条必须写进证据页的纪律**：① `band 0` 是**域平均**不是变率（Stage A 已记录把 1 K 偏置误读成谱内容）；
② 最短波长为 40.2 km，接近 0.25° 网格的 Nyquist，**低于该尺度的谱不可信**。

---

## §4 五个预声明出口

| 判据 | 要求 | 为什么是这个 |
| --- | --- | --- |
| **(a)** | 三臂（control/treatment/placebo）的 `δ_HB` 中位数**在三 seed 上一致地互不相同** | 若诊断量对受控扰动无反应，它就没测量任何东西 |
| **(b)** | **背离**：至少一个 (lead, 变量) 上，RMSE 更好的臂平衡偏差更差 | 这是 Bonavita 论点的可证伪表述，也是「新指标有信息」的唯一检验 |
| **(c)** | 三 seed 的平衡偏差**极差 < 臂间差** | 否则信号被 seed 离散淹没——上一轮 72 h 的教训 |
| **(d)** | 真值/分析场上的偏差**显著小于模型场**（阳性对照） | 防止诊断量本身是坏的。0109 的教训：判据必须先能失败 |
| **(e)** | 预检：单位检查通过、`f` 量级正确、`Ro<0.1` 覆盖率逐层报出、65×65 全有限 | 单位事故在本仓真实发生过 |

**若 (a)–(d) 任一为假，如实记负并停。**
若 (b) 为假（平衡更好的臂 RMSE 也更好），那会**否证** Bonavita 论点在区域小模型上的适用性——
这同样是有价值的结论，不得掩盖。

### §4.1 (b) 是主判据，且它的方向要说清

(b) 不是「诊断量有效」，而是「诊断量提供了 RMSE 没有的信息」。两种失败的形态不同：
- (a) 假 ⇒ 诊断量是死的（对任何扰动无反应）
- (b) 假 ⇒ 诊断量活了，但它和 RMSE 说的是同一件事（冗余）

只有 (a)(b) 都真，才可以说「物理平衡是一个独立的、有信息的评价轴」。

---

## §5 实施顺序与成本

| 阶段 | 内容 | 成本 |
| --- | --- | --- |
| **P0** | 预检（已执行） | **0 GPU-h** |
| P1 | `training/r7_balance_diagnostics.py`（已写，25 条测试全过） | 0 GPU-h |
| P2 | field-dump 评估分支（新脚本，不改 `r7_evaluate.py`） | 0 GPU-h |
| P3 | 15 checkpoint 重推理 → 场落盘 | **≈6.5 GPU-h** |
| P4 | 诊断计算 → `diagnostics.json` | 分钟级 CPU |
| P5 | 判定 (a)–(e) | 0 |
| P6 | 证据页 + 四件套登记 | 0 |

**软预算 4 h / 硬上限 8 h**（重推理必须落盘场，之后诊断可反复重跑 0 成本——这是本轮设计的关键，
上一轮的教训是「场丢了就永远丢了」）。

---

## §6 明确不做

| 不做 | 理由 |
| --- | --- |
| 重开 C1/C2/Stage P/R/C/A | 全部按预声明关闭（0084 §6、0094、0109、0111、0112） |
| 重开显式 tendency 输入 / 4 个物理诊断量作输入 | 0049、`record:s3-typed-spatial-result` |
| 重开城市静态协变量 | 0113 的 (a) 为假，已关闭 |
| **训练任何模型** | 本轮纯评估；15 个 checkpoint 已存在 |
| 改 `model/**` | 会改 `model_code_sha256`，让 15 个 checkpoint 全部不可加载 |
| 改 `training/r7_evaluate.py` / `r7_long_rollout_runner.py` | 316 行 / **597/600 行（R-051）** |
| 做梯度风平衡（文献式 7–8） | 需要识别气旋中心并做 `N,M≤5` 径向拟合；区域 65×65 上气旋未必在域内，且我们不是单案例研究。**列为未做的限制**，由地转+连续方程覆盖 |
| 引入 `pywt`/`pyshtools`/`ducc0` | 区域 patch 上 SHT 不良定义（决策 0110） |
| 重建 store / 新增数据 | 只读 |
| 读 2023 test | `r = 0` |
| 承诺正面结果 | 若 (a)–(d) 任一为假，如实记负 |

---

## §7 风险（如实列）

1. **`500–250` 层对只能用干温度。** store 没有 `t250`/`q250`。**代价**：该层对的 `δ_HB` 会混入
   湿度的缺失，必须与 850–500 分开报告，不得合并成「平均静力偏差」。
2. **`δ_HB` 有一个 +2.6% 的系统偏置。** 已查清来自真实大气（隐含上层气压 498.9 vs 500 hPa），
   不是单位错误。**代价**：臂间比较必须用**差值**而不是绝对值，否则 2.6% 的基线会淹没任何模型差异。
3. **地转偏差只覆盖 6.8%–15.8% 的格点。** **代价**：该统计不是全域结论，必须在每个 (lead, 变量) 上
   同时报出覆盖率。
4. **谱的最低可信尺度是 40.2 km。** 0.25° 网格的 Nyquist 约 55 km，径向 bin 到 40.2 km。
   **代价**：文献里「<20 km 谱被放大」那一类读数我们**做不到**，只能说「接近 Nyquist 的 bin」。
5. **15 个 checkpoint 来自两组不同的训练配方。** Stage A 的 on/off 与城市静态的三臂不是同一个父亲。
   **代价**：(a)(c) 的臂间比较只在**同一组内**成立，跨组只能比较「诊断量的量级」而不是 delta。
6. **重推理本身有 GPU 非确定性。** 前几轮记录过默认多线程 GPU 运行逐位不同。
   **代价**：本轮不做逐位复现承诺，只做 config-reproducible；场一旦落盘，诊断部分的重复是确定的。
7. **n = 1 模型、单实例、单分辨率、单区域（27–43°N / 107–123°E）。** 与前面七轮相同的适用域限制。

---

## §8 workflow 与提示词

执行形态是**一份 goal 长文 + 一个派生 workflow**，两者合在 objective 里。

- `docs/goals/s3-balance-diagnostics.md`，头部 `<!-- round-node: S3 -->` + `<!-- round-spec: {...} -->`
- `.zcode/workflow-drafts/r7-balance-diagnostics.dwf.ts`，从 `r7-urban-static-ablation.dwf.ts` 派生
- **必须用 `CreateWorkflow {path, args}`，不要用 `AmendWorkflow`**——实测它丢 `args` 让 `dryRun` 静默变 true
- 零成本编译校验：`CreateWorkflow` + `args: {maxRounds: 0, dryRun: true}`
- **判定读 `readings.json` 时用 `json.load`，不要用正则匹配自报文本**——上一轮实测一条 judge 命令
  返回空 body，正则把所有子句默认成 false，卡片与实际读数相反（见 `da1ed00`）

---

## 实际结果（Actual Results）

**P0 预检已执行；P1 的实现与测试已落盘；P2–P5 未执行。**

### 已发生

- **`training/r7_balance_diagnostics.py`**（约 275 行）：`coriolis_parameter` / `virtual_temperature` /
  `hydrostatic_thickness` / `hydrostatic_deviation` / `geopotential_to_height` /
  `geostrophic_wind` / `rossby_number` / `geostrophic_departure` / `column_divergence_residual` /
  `divergence_from_wind` / `fine_scale_noise` / `radial_amplitude_spectrum`。
  纯 torch，不读文件、不开 store、不吃 contract（由 AST 扫描测试断言）。
- **`tests/test_r7_balance_diagnostics.py`**：**25 条测试全过**，含六条反证加三条真实数据钉子。
- **P0 读数**（0 GPU-h，真实 store）：见 §2。核心三条：`δ_HB` 中位数 **+0.021…+0.029**（系统偏置，
  已查清来源）、地转覆盖率 **6.8%–15.8%**、`z500` f13 噪声 **39.97**。

### 未发生

- **P2 未执行**：没有 field-dump 脚本，没有重推理，**没有任何模型场的平衡读数**
- **P3–P5 未执行**：没有 `diagnostics.json`、没有 `readings.json`、没有 (a)–(e) 判定
- **GPU-h = 0**
- 未写 `protocol.json`、未生成 `code.zip`、未登记

### 未验证

- 模型场的平衡偏差是否与真值场不同（(a)(d) 的全部内容）
- RMSE 与平衡偏差是否会背离（(b)）
- 三 seed 的离散度是否小于臂间差（(c)）
- 地转偏差在 6.8%–15.8% 格点上的统计是否稳定

**在此之前，任何引用本文件作为「模型物理不平衡」或「平衡诊断有效」的证据都是错的。**
本文件目前的全部内容是：一个设计、一份文献的一手核对、一个已在真值场上跑通的诊断模块、
以及 25 条已经通过的测试。**模型场一次都还没看过。**

---

## §9 P2 冒烟实测（2026-11-14，6 个 case，CPU，非正式读数）

`scripts/r7_balance_field_dump.py` 已端到端跑通（1 个 checkpoint × 6 个 case × 5 lead），
`training/r7_balance_diagnostics.py` 在落盘的场上给出下列读数。
**这不是本轮的正式读数**（6/1447 个 case、单 seed、单臂），但它决定了判据设计是否成立。

### §9.1 一个必须先修的缺陷（已修）

第一次落盘把**归一化空间**写成了物理单位：`ZarrRolloutDataset` 在读时施加
`(frames − mean)/std`（`data/r7_evaluation.py:113-114`），所以 `forecast` 与 `rollout_targets`
都是归一化的。结果是 `δ_HB` 中位数 **−51.12**、真值 `std` **7088**——物理上不可能。
修复：落盘前用 `dataset.mean` / `dataset.std` 反归一化，并把两者一并存进 `.npz`。
修复后 `z500` 均值 **54757 / 54826** `m² s⁻²`、`t2m` 均值 **276.2 / 275.8 K`，单位正确。

**这条是本轮第一个真实缺陷，且它正是「单位事故」那一类**——与 224× 事故同型，
只是这次发生在归一化而不是 `/g` 上。

### §9.2 最关键的一张表：模型 vs 真值，逐 lead

| lead | 模型 `ws_ag` (m/s) | 真值 `ws_ag` (m/s) | 模型 `z500` f13 噪声 | 真值 f13 噪声 |
| ---: | ---: | ---: | ---: | ---: |
| 6 h | 4.58 ± 0.22 | **2.46 ± 0.21** | 34.41 ± 1.56 | **29.45 ± 1.53** |
| 12 h | 5.97 ± 0.33 | **2.41 ± 0.23** | 38.79 ± 2.31 | **28.39 ± 2.11** |
| 24 h | 7.48 ± 0.31 | **2.69 ± 0.32** | 44.07 ± 3.48 | **26.68 ± 1.94** |
| 48 h | 8.78 ± 0.59 | **2.77 ± 0.41** | 50.53 ± 3.53 | **24.87 ± 0.94** |
| 72 h | 9.75 ± 0.73 | **2.34 ± 0.34** | 57.32 ± 2.85 | **21.42 ± 2.10** |

**这张表是本轮设计的全部依据，而且它同时回答了 (b) 和 (d)：**

- **真值场的 `ws_ag` 在五个 lead 上是平的**（2.34–2.77 m/s，无趋势），细尺度噪声甚至**下降**
  （29.5 → 21.4）。分析场随 lead 变长不会变得更不平衡——它本来就不是一个随 lead 演化的量。
- **模型场的 `ws_ag` 单调上升 2.1 倍**（4.58 → 9.75），噪声单调上升 1.7 倍（34.4 → 57.3）。
- 模型/真值的比值从 **1.86×** 扩大到 **4.17×**（`ws_ag`）与 1.17× 到 2.38×（噪声）。

⇒ **(d) 阳性对照通过**：诊断量确实能把模型场与真值场分开，且分开的方式是有序的（随 lead 单调）。
⇒ **(b) 的背离在最粗的意义上已经可见**：72 h 的 RMSE 最差（3.66 K），同时平衡偏差也最大——
但注意这一条**不是** (b) 要的「背离」，(b) 要的是**臂间** RMSE 序与平衡序相反。§9.2 只证明了
诊断量有分辨力，还没证明它与 RMSE 不同源。

### §9.3 静力偏差没有 lead 趋势

| lead | 模型 `δ_HB` 中位数 | 真值 `δ_HB` 中位数 | Δ |
| ---: | ---: | ---: | ---: |
| 6 h | +0.02743 | +0.02624 | +0.00119 |
| 12 h | +0.02866 | +0.02645 | +0.00221 |
| 24 h | +0.02826 | +0.02884 | −0.00058 |
| 48 h | +0.02787 | +0.02846 | −0.00059 |
| 72 h | +0.02758 | +0.02808 | −0.00050 |

模型与真值都在 2.7%–2.9%，**差异 ≤ 0.0022 且 72 h 时模型反而略好**。
⇒ 静力平衡（D1）在这个模型上**不是一个有分辨力的诊断量**。这条必须如实写进证据页，
而它恰好加强了地转（D2）与谱（D4）的价值：**四个诊断量不是等价的，其中两个有分辨力、一个没有。**
