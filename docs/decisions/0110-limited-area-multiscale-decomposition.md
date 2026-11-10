# 0110 — 区域网格上的多尺度分解：为什么用 rfft2 径向分带 + DCT-II 交叉校验，而不用球谐或 wavelet

- **日期**：2026-11-11
- **状态**：proposed —— 前置探针 P0 跑完、给出分解合法性表之后才转 accepted
- **范围**：`training/r7_spectral_geometry.py`（新模块）、`training/r7_objective_axes.py`（唯一分发点）、
  `scripts/r7_s3_spectral_geometry_probe.py`。**本条 0 GPU-h、零读数。**
- **依据**：`docs/plans/0030-thesis-physics-informed-mainline.md` §2.3、§3.1；
  `docs/R7_PHYSICS_LOSS_LITERATURE_REVIEW.md` §1.1/§1.2/§4.2（及其 §7 更正）；
  决策 0084 §6（族判别）、0094（C1 前置否决）、0105（守卫必须能失败）、0109（C2 关闭）

## Context

**① 一条必须修的事实误引。** `docs/plans/0029-physics-operator-candidates.md` §4.1 与
`docs/R7_PHYSICS_LOSS_LITERATURE_REVIEW.md` §4.2 都写「本仓是 **global** 65×65，所以球谐理论上适用」。
实测 `docs/DATA_SPEC.md:36`：ROI `27.0–43.0°N / 107.0–123.0°E`、0.25°、`[65, 65]` —— 这是一个
**区域 patch**。球谐变换在此不良定义，且 venv 里 `pyshtools`/`ducc0`/`torch_harmonics` **均未安装**。

**② 一手文献自己否掉了「直接套 rfft2」。** `arXiv:2501.19374v2` §4.2（访问 2026-11-10）逐字：

> This is simple enough for **global** latitude/longitude grids, but it might be **difficult for
> local-area models** without a regular global grid structure. In these cases, we think that the basic
> intuition behind (6) might still apply through other multiscale decompositions such as **wavelet
> lifting** … **Without such a decomposition (e.g. applying the adjustment of (6) globally, without the
> harmonic transform), the model might be able to "cover up" a lack of fine-scale variability by
> over-emphasizing coarser scales.**

所以「不加任何尺度分解、直接在整个区域算一个谱距离」是论文**点名**的反模式：模型可以靠
**过度加权粗尺度**来降低该项，而细尺度依然不足 —— 那正好是本轮要修的病。

**③ 但谱几何不能放弃。** 它是唯一有**解析证明**支撑的物理尺度信息：MSE 的最优解是
`σ_X = ρ < 1`（`E[MSE] = σ_X² + 1 − 2σ_Xρ` 对 `σ_X` 求极小），即**振幅不足是损失函数的最优解性质**，
不是训练不足。这条与模型规模、数据量无关（同文 §1.1）。而我们已实测模型场比真值**平滑 2.9 倍**
（C1 前置 §3），方向一致。

**④ 三维约束**：① 区域网格 ⇒ 无 SHT；② 必须有尺度分解 ⇒ 不能直接用全区域单个谱距离；
③ 不得引入新依赖（改依赖即改 `training_code_sha256` 与执行闭包，且 `training/r7_long_rollout_runner.py`
已 597/600 行，模块必须另开）。

## Decision

**采用「2D `rfft2`（`norm='ortho'`）+ 径向波数分带」作为 AMSE 原则在区域网格上的多尺度分解，
并以 DCT-II（`scipy.fft.dctn`）做独立交叉校验；两种变换必须得出同一结论，否则结论作废。**

具体：

1. **分解**：对每个 `[C,H,W]` 场做 `torch.fft.rfft2(..., norm='ortho')`，按**总波数模长**
   `k = √(kx² + ky²)` 分带（band 边界**冻结**在协议里，由 65×65 的可用波数范围确定）。
2. **逐带量**：`PSD_x(k)`、`PSD_y(k)`、`coh_k = ΣRe(α_x α_y*) / √(PSD_x PSD_y)`。
3. **AMSE-analogue 形式**（对应论文式 (6)，但按带而不是按球谐阶）：
   `Σ_bands (√PSD_x − √PSD_y)² + 2·max(PSD_x, PSD_y)·(1 − coh)`。
4. **交叉校验**：同一测量在 `scipy.fft.dctn(type=2, norm='ortho')` 下重算，报两者一致性。
   DCT 隐含**偶对称延拓**、rfft2 隐含**周期延拓**，两者对边界处理假设不同；
   若两条独立路径给出相反的结论，就说明结论由边界假设驱动，**不得**采信。
5. **强制报出的量**：**接缝能量占比**（周期延拓引入的人工不连续所占的谱能量比例）。
   不设阈值（设阈值等于事后调参），但**必须在每次测量里报出**，并写进 limitations。

## 显式声明的三项近似（不得省略）

| # | 近似 | 后果 | 缓解 |
| --- | --- | --- | --- |
| 1 | **周期接缝**：区域块左右/上下周期延拓引入人工不连续 | 会向谱里注入真实场没有的能量，偏向细尺度 | P0 量化「接缝能量占比」并每次报出；DCT（偶对称延拓）作对照 |
| 2 | **方域径向分带**：band 边界是**选择**，不是物理常数 | 可被质疑为调参 | 边界**冻结**并写进 `protocol.json`；DCT 交叉校验；不扫 band |
| 3 | **不是球谐** | 与论文不是同一个变换 | **命名纪律**：一律称 *AMSE-analogue on a limited-area grid*，**不得**称 AMSE |

## Consequences（含负面后果，R-033）

**正面**

- 让「尺度相对权重」这件事在**本仓实际的数据形状**上可测、可训，而不需要新依赖、不需要 SHT；
- 交叉校验（rfft2 vs DCT）给出一条**机制性**的判别：若两法结论相左，问题在边界假设，
  那本身就是一条关于「区域谱损失可不可信」的结论；
- 与已关闭的三支都不同族（J4）：`pred = target` 时精确 0，最优解是**目标谱**而非常数场；
  它**替换** MSE 的几何而不是**加**一项，所以 C2 的剂量问题（0.29–0.41）不适用。

**负面（如实列）**

1. **这是近似，不是论文的方法。** 所有结论的适用域是「区域 patch + rfft2/DCT 分带」，
   不能外推到全球谱方法，也不能说「复现了 AMSE」。**这条的代价是它削弱了论文的一条宣传口径**：
   我们做的是 *analogue*，不是复现；答辩时不能把 1,250 km → 160 km 那个数字搬来当自己的结果。
2. **接缝是真实污染源。** 若接缝能量占比不可忽略，逐带 `PSD` 与 `coh` 都带偏，
   而 bias 的方向未知（可能夸大也可能低估细尺度缺口）。P0 只能**量化**它，不能消除它。
3. **方域径向分带在 65×65 上分辨率很粗**：可用波数只有约 33 个径向 band（Nyquist 到 DC），
   分带后每带样本少，`coh` 的估计方差大。
4. **DCT 交叉校验不是独立证据的充分条件**：两者都在同一张 65×65 网格上，共享网格本身的
   离散化误差；一致只能排除「边界假设驱动结论」，不能排除「网格分辨率驱动结论」。
5. **一个已知的退化分支**：当模型振幅**超过**目标时，`max(PSD_x, PSD_y)` 的行为与振幅不足时不同
   （论文未讨论这一支的稳定性）。本条的记录里必须**分别报**振幅不足与超过两种情形的样本数。
6. **本条不授权开轮。** Q 轮（Stage Q）的开轮前提是 P0 的分解合法性表 + P1 的缺口趋势 +
   J3/J4 的剂量与族判别实测；本条只解决「用哪个分解」这一个前置问题。**代价**：它把开轮的门槛
   又推后了一步——本轮结束时可能只有探针读数、没有训练读数。

## 否决的方案与理由

| 方案 | 否决理由 |
| --- | --- |
| **wavelet lifting**（论文建议的替代） | `pywt` **未安装**；引入新依赖改 `training_code_sha256` 与执行闭包；且去抽样 DWT **非平移不变**，在 65×65 上会引入相位伪影，与「逐带 PSD/coh」这个不需要相位对齐的测量不兼容 |
| **单独用 rfft2、不分带** | 正是论文 §4.2 点名的反模式：模型可靠过度加权粗尺度「cover up」细尺度不足 |
| **引入 `pyshtools` / `ducc0` / `torch_harmonics` 做 SHT** | 区域网格上 SHT **不良定义**，装了也没用；且三个包都不在 venv |
| **改 `training/r7_long_rollout_runner.py` 接线** | 该文件已 **597/600** 行（R-051），加代码会立即超限；必须另开模块 |
| **用 legacy `physical_consistency.py` 里的 FFT 项** | 打包了已关闭/未授权的 FFT、梯度、日变化、拖曳项（决策 0092 明文） |
| **把该项做成「无目标」的谱约束** | 那就是 C1 那一族（无目标、方向反），已被 0094 实测否决（8/8 例方向反） |

## 与其它决策的关系

- **不冲突**：0094（C1 无目标 ⇒ 否决）与 0109（C2 加一项 ⇒ 关闭）都是关于「**加**一个约束项」；
  本条是「**换**目标函数的几何」，两者的最优点集合不同（J4）。
- **受约束**：0084 §6 要求族判别论证可检验。本条的 J4 论证落在两个可测事实上：
  ① `pred = target` 时 AMSE-analogue 精确为 0（而空间梯度损失在真值处不为 0）；
  ② 常数场的谱只在 DC 带非零，故它**不是**该项的最优解（而它是梯度损失的最优解）。
  这两条必须在 P0 里实测，不能只在文档里写。
- **依赖**：0105 的教训（守卫必须能失败）直接落在 P0 的 J2 反证上。

## 恢复第一条命令

```bash
.venv/bin/python tools/r7_round.py state
```
