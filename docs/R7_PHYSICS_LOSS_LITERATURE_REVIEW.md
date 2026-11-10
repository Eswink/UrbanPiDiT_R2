# 物理信息损失的一手文献检索（2026-11-10）——对本轮候选方向的裁决

- **日期**：2026-11-10
- **状态**：**检索完成，未开轮**；**0 GPU-h**
- **`scientific_claim: false`**；`test_read: false`；`r = 0`
- **通道**：arXiv（`mcp__arxiv__*`），一手论文正文与摘要
- **服务对象**：`docs/plans/0029-physics-operator-candidates.md` 的五个候选方向
- **检索执行**：主链路直接调用 arXiv MCP（委派 `arxiv-researcher` 三次均遇到 provider 错误，
  见 §5 的降级记录）

---

## §0 一句话结论

**检索把 plan 0029 的优先级排序改写了。** 最重要的不是「再加一个物理约束」，而是：

> **标准 MSE 在数学上鼓励模型压低振幅**（最优解 `σ_X = ρ < 1`），
> 这一条有一手论文的**解析证明 + 实测**，而且给出了一个 **parameter-free 的替换损失（AMSE）**，
> 在 GraphCast 上把有效分辨率从 **1,250 km 提升到 160 km**。

这个发现对 plan 0029 的影响是：

- **方向 E（谱保真度）从「离群候选」升为第一优先**，但**形式要改**：
  不是「分波数带加权的 log 能量差」，而是 AMSE 那种**参数零调制**的形式；
- **方向 A（静力平衡）的优先级下降**，因为 PINN-Cast 这类工作用了「未缩放的有限差分」
  还能报告正向结果，说明**量纲不精确的物理项也能"work"**——这反而使 J1 判据的解释力变弱；
- **方向 B/C（整层散度积分、热成风）没有任何直接文献支持**，且 C 已知在低纬失效。

---

## §1 检索到的三条核心证据

### 1.1 【最强】MSE 的平滑是**最优解性质**，不是训练不足

**来源**：Subich, Husain, Separovic, Yang, *Fixing the Double Penalty in Data-Driven Weather
Forecasting Through a Modified Spherical Harmonic Loss Function*, **arXiv:2501.19374v2**
（访问日期 2026-11-10，一手，已读正文 §2.1–§2.4、§4、§5）。

**解析论证（其式 (1)–(2)）**：设目标 `Y ~ N(0,1)`，预测 `X = σ_X(ρY + √(1−ρ²)N(0,1))`，则

```
E[MSE(X,Y)] = σ_X² + 1 − 2σ_Xρ
```

对 `σ_X` 求极小 ⇒ **最优 `σ_X = ρ`**。因 `0 < ρ < 1`（尺度不可完全预测），
⇒ **最优预测的振幅必然小于真值**。这不是优化不充分，是**损失函数的最优解**。

**实测（其 Figure 2）**：1° GraphCast 训练中，总波数 100（波长约 400 km）处，
**振幅比与相干性紧密跟随**（`√(PSD_x/PSD_y) ≈ coh_k`），且自回归步数增加时平滑加剧。

**判决**：这是一条**关于损失函数几何性质**的结论，与模型规模、数据量无关。
**它适用于本仓**：我们的训练目标就是标准 MSE（`result.loss`），所以
**我们的模型同样在数学上被鼓励压低振幅**——这与我们在 off 臂看到的
「24h skill > 12h skill」的非单调、以及 72h skill 只有 +0.23 的形态**方向一致**
（但本条不声称因果，只是相容）。

### 1.2 AMSE：parameter-free 的替换损失

**形式（其式 (5)–(6)）**：把 MSE 按球谐分解写成

```
MSE = Σ_k (√PSD_k(x) − √PSD_k(y))²  +  2√(PSD_k(x)PSD_k(y))(1 − coh_k(x,y))
```

然后**打断第二项里振幅与相干性的耦合**：

```
AMSE = Σ_k (√PSD_k(x) − √PSD_k(y))²  +  2·max(PSD_k(x), PSD_k(y))·(1 − coh_k(x,y))
```

**关键性质（对我们 J4 族判别直接有用）**：
- **零当且仅当 `x = y`**（与 MSE 相同）；
- **在 `x = y` 处与 MSE 有相同的 Taylor 展开**（⇒ 局部行为不变，不是另一族）；
- **parameter-free**：不需要选截止波数、不需要给两项加权；
- 梯度**始终指向** `coh_k → 1` 与 `PSD_k(x) → PSD_k(y)`；
- **不再是度量**（不满足三角不等式）。

**实测**：微调 ¼°/13 层 GraphCast，**有效分辨率 1,250 km → 160 km**，
改善集合离散度、热带气旋强度、地面风速极值（其 §0 摘要与 §4）。

**成本如实记**：其微调用了约 **26.2 GPU-days**（4×A100/节点，1–2 节点）。

### 1.3 PINN-Cast：物理软约束"有效"，但**自己承认量纲不精确**

**来源**：Saleem, Salim, Purcell, *PINN-Cast*, **arXiv:2604.27313v1**（访问日期 2026-11-10，一手，
已读全文）。

**它的物理损失**：

```
L_physics = L_lat-weight + α·L_kinetic + β·L_thermo
L_kinetic = |½(u²+v²)_pred − ½(u²+v²)_true|
L_thermo  = E[(ΔT/Δt + u·ΔT/Δx + v·ΔT/Δy)²]
```

权重：**α = 0.1–0.5，β = 0.8**（其 Table 1）。

**它报告的结论**：消融显示「physics-informed loss」这一项**带来改进**（其 §4.6）。

**但它明写的局限（其 §5 Limitations，逐字要点）**：
> The physics-informed penalties rely on **unscaled finite differences** and simplified
> advection proxies rather than **dimensionally exact** governing equations.
> …「we compute finite differences on the native grid **without explicit metric scaling**;
> the penalty therefore encourages **directional consistency** rather than enforcing a
> dimensionally exact advection residual.」

**这对本仓的直接含义（重要，且是负面的一份）**：
`L_thermo` 与 C1/C2 的失败模式**同源**——都是「在归一化/未换算的网格上算物理量」。
它仍然报告正向，说明：

- **量纲不精确不等于一定失败**；
- ⇒ 我们 C1 的失败**不能**简单归因于"量纲问题"（C1 的真正死因是**方向反**：
  分析场被罚 7.1–10.2×）；
- ⇒ 但**也不能**把 PINN-Cast 的"成功"当成"物理约束有用于天气预报"的普遍证据：
  它是 5.625° 粗分辨率、单变量集合（t2m/t850/u10/v10）、50 epoch、4×V100 约 2 天，
  且其物理项**不读目标**（`L_kinetic` 是 pred 与 true 的差、`L_thermo` 是 pred 自身的平流残差）
  ⇒ **它是 C1 那一族，不是 C2 那一族**。

---

## §2 检索**没有**找到的（如实登记为「未能确认」）

| 想找的 | 结果 |
| --- | --- |
| 有论文**定量报告**「物理约束项对真值的惩罚大于对预测」这类方向性问题 | **未能确认**（无直接命中）。C1 的这一失败模式在本轮检索范围内**没有文献先例** |
| **静力平衡 / 热成风**作为**训练损失**（而非同化强约束或诊断工具）在 NWP 的先例 | **未能确认**。唯一命中是 Kepler 恒星的 `2507.06357`（Kurucz-a1，恒星大气 PINN），与大气预报不同域 |
| 「整层散度垂直积分≈0」作为损失 | **未能确认**，零命中 |
| 平衡关系在再分析场上的**残差量级**定量文献 | **未能确认**，零命中 |
| 谱损失**退化为平滑先验**的反面证据 | **未能确认**。AMSE 论文反而指出：MSE 的平滑可能起"隐式正则化"作用、加速训练（其 §5 猜测），但**未测** |
| 65×65 或相近粗分辨率的**有效波数上限** | **未能确认** |
| 物理项/数据项**权重比的量级判据** | **未能确认**。只有 PINN-Cast 的 α=0.1–0.5、β=0.8 一组具体值，无普适判据 |

---

## §3 对 plan 0029 五个方向的**逐条裁决**

| 方向 | 检索支持 | 裁决 |
| --- | --- | --- |
| **A 静力平衡** | **无 NWP 训练损失先例**（只有恒星大气） | **降级**。可作为"通道间关系"的候选，但没有文献支撑 J1 会过；且其量纲敏感性与 C1 同族风险未解 |
| **B 整层散度积分** | 零命中 | **保持「预期失败」**。它是无目标约束，与 C1 同族 |
| **C 热成风** | 零命中；已知低纬 `f→0` | **降级为最低优先** |
| **D 湿物理违反** | 未检索到直接文献 | **保持"先测违反率"**（0 GPU-h 前提不变） |
| **E 谱保真度** | **强支持**（arXiv:2501.19374，解析 + 实测） | **升为第一优先，且形式改为 AMSE** |

### 3.1 方向 E 的**形式修正**（这是本节最实质的改动）

plan 0029 §3.E 原来提议：

```
L_spec = Σ_k w_k (log E_pred(k) − log E_target(k))²
```

**这个形式有问题**，检索给出了更好的：

1. **`log` 域差**只约束**振幅**，丢掉了**相位/相干性**。AMSE 的关键是把损失分成
   「振幅误差」与「去相干误差」两项，且**只对后者打断耦合**；
2. **`w_k` 是可调参数** ⇒ 引入自由度、需要调参、且会成为事后放宽的入口。
   AMSE **parameter-free**；
3. AMSE **在 `x=y` 处与 MSE 同 Taylor 展开** ⇒ 满足 J4 的「不是另一族」要求，
   且比"log 能量匹配"更容易论证。

⇒ **方向 E 应当直接采用 AMSE（式 (6)），而不是自创 log 谱损失。**

---

## §4 对本仓的三条具体可执行结论

### 4.1 一条**零 GPU-h** 就能验证的强预测

AMSE 论文的式 (2) 给出可测预测：**标准 MSE 训练下，预测的谱振幅比应约等于相干性**。

⇒ 在我们已有的 checkpoint 上（p1600 端点）可以直接测：

```
对每个波数带 k：amplitude_ratio(k) = √(PSD_pred(k)/PSD_target(k))
                coherence(k)       = ΣRe(α_p α_t*) / √(PSD_p PSD_t)
检验：amplitude_ratio(k) ≈ coherence(k) 是否成立
```

**若成立** ⇒ 我们的 off 臂也在经历同样的"振幅不足"，**AMSE 有明确的改进空间**，
且这个测量**不需要任何训练**。这比 plan 0029 §4 原来列的五个探针**更直接**。

### 4.2 AMSE 的关键限制（必须先想清楚，否则会白费）

1. **球谐变换的成本**：AMSE 需要 `SHT`。本仓 `model/r7_baselines.py:117` 用的是
   **平面 `rfft2`**，不是球谐。65×65 的经纬网格上，**平面 FFT 与球谐在极区差异大**
   ——而我们的域是 `full` 全域，含极区。
   AMSE 论文自己在 §4.2 明说：「simple enough for global lat/lon grids, but might be
   **difficult for local-area models**」，并建议 wavelet lifting 作为替代。
   **本仓是 global 65×65**，所以理论上适用，但需要一个球谐变换实现（或近似）。
2. **`max(PSD_x, PSD_y)` 的行为**：当模型振幅**超过**目标时，第二项退化为 `2·PSD_x(1−coh)`，
   梯度行为与振幅不足时不同。论文未讨论这一支的稳定性。
3. **它不再是度量** ⇒ 不能直接用它做验证指标（验证仍用 MSE/RMSE/ACC）。
4. **多变量聚合**：论文用「每变量权重 × 气压层权重 × 每变量每层标准差归一化」。
   本仓 17 通道的归一化已由 `normalization_mean` 提供，需核对是否等价。

### 4.3 与 C2 的关系（不冲突，但会**改** C2 的解读）

C2 测的是「导数域目标匹配」，与 AMSE 无关。但 AMSE 的存在意味着：

- 若 C2 最终为负，**不能**推出「物理/结构信息无用」；
- 只能说「**在 MSE 目标下、以单一派生量为约束**的物理信息无用」；
- 而 AMSE 提供了一个**不同层次**的改进（改目标函数的几何，而不是加约束项）。

⇒ **即使 C2 关闭，方向 E（AMSE）仍然独立成立**。这是 plan 0029 §6 里
「C2 判决前不要开新物理轮」这条约束的**例外**，应当明确写入。

---

## §5 降级记录（如实）

**委派失败**：三次委派 `arxiv-researcher` 子智能体——
（1）物理约束有效性、（2）平衡关系作为损失、（3）谱损失——
**第 1 与第 3 次返回 provider server error，第 2 次被用户中断**。

**降级动作**：改由主链路直接调用 `mcp__arxiv__*` 工具（`search_papers` / `get_abstract` /
`download_paper` / `read_paper` / `search_paper_text`）。**主链路的 `WebSearch`/`WebFetch`
未被使用**（R-049 闸门仍生效）。

**可达性实测**：

```
curl -sS -I -L -m 8 -o /dev/null -w 'arxiv=%{http_code}'  https://arxiv.org/          -> 200
curl -sS -I -L -m 8 -o /dev/null -w 'export=%{http_code}' https://export.arxiv.org/api/query?... -> 200
```

**检索式（记录以便复算）**：

```
1. "physics-informed" AND "weather" AND "loss"                       (cs.LG, physics.ao-ph)
2. "spectral loss" AND (weather OR atmospheric OR "energy spectrum")
3. "hydrostatic" AND ("loss" OR "constraint") AND ("neural" OR "deep learning") AND atmosphere
4. "blurring" OR "spectral bias" OR "double penalty" AND "data-driven weather"
5. "mass conservation" AND ("neural network" OR "machine learning") AND ("weather" OR "atmosphere")
```

第 2、3、5 条命中稀疏或为零；第 4 条命中 **arXiv:2501.19374**（本轮最重要发现）。

---

## §6 引用纪律与边界

- **一手**：上述所有数字与结论均读自 arXiv 论文**正文或摘要原文**，arXiv ID 已给出。
- **访问日期**：2026-11-10（UTC）。
- **未做**：没有用 `WebSearch`/`WebFetch`；没有读二手综述；没有把检索摘要当证据
  （每条结论都回到论文正文段落）。
- **不能声称**：
  - 不能声称「AMSE 在我们的模型上会有效」——**未测**；
  - 不能声称「C1 的失败模式有文献先例」——**未能确认**；
  - 不能把 PINN-Cast 的 α/β 当作普适权重建议——那是它自身分辨率与架构下的具体值。

**恢复第一条命令**：

```bash
.venv/bin/python tools/r7_round.py state
```
