# Stage A — β_j AMSE-analogue 归一化的 on/off 读数与判定

- **日期**：2026-11-13
- **变更类型**：实验
- **触发原因**：执行 `docs/goals/s3-physics-informed-sequence.md` 的 Step A
- **前置 ADR**：决策 **0112**（准入）、**0110**（分带分解）、**0109**（(d) 不得是字面量）
- **结论**：**(a) 真、(b) 假、(c) 假、(e) 真** ⇒ 按预声明**永久关闭 `spectral_geometry` 支**
- **`scientific_claim: false`**、**`test_read: false`**、**`r = 0`**
- **变更文件清单（本页提交）**：`training/r7_spectral_geometry.py`、`scripts/r7_s3_spectral_geometry_a_protocol.py`、`scripts/study_r7_s3_spectral_geometry_a.py`、`tests/test_r7_s3_spectral_geometry_a.py`、`docs/goals/s3-physics-informed-sequence.md`、`docs/decisions/0112-*.md`、`docs/rules/size-thresholds.md`

---

## 1. v1 总结

Step A 问的是一个可失败的问题：**把 FastNet 式 (10) 的逐变量 β_j 归一化接到已有的
`spectral_geometry` 轴上，能不能在抬高细带振幅的同时不退化 t2m skill？**

读数给出一个**罕见地干净的分裂**：

| 判据 | 结果 | 读数 |
| --- | --- | --- |
| **(a)** 细带 `amp_ratio` 相对同 seed off 臂升高 | **真** | 三 seed 全部升高：**+0.125128 / +0.113816 / +0.116573** |
| **(b)** 72 h t2m skill 三 seed 配对 Δ 全 ≥ 0 | **假** | **−0.066368 / −0.057539 / −0.062297** |
| **(c)** 24/48 h skill 配对 Δ 三 seed 全 ≥ 0 | **假** | 24 h：−0.029613 / −0.025704 / −0.032269；48 h：−0.051183 / −0.049916 / −0.046225 |
| **(d)** off 路径逐位等价 | 由代码路径承载 | `training/r7_objective_axes.py:43` 的 `None` 早退 + 两条反证测试（见 §4） |
| **(e)** 预检：β_j 非全相等且 amse 全有限 | **真** | β 跨度 86.75497080940221×；8 case × 12 step 全有限 |

**⇒ `verdict = permanently-close-the-spectral-geometry-branch`。**

**这不是「方法没起作用」，而是「它精确地起了它声称的作用，而那件事本身有害」。**
(a) 是 AMSE 的直接作用点：β_j 归一化把细带振幅比从 0.72 抬到 0.84–0.85（+11.4% 到 +12.5%），
三 seed 一致。而同一改动让 **9/9 个 (lead, seed) 单元的 skill 全部下降**，且 lead 越长降得越多
（24 h ≈ −0.029、48 h ≈ −0.049、72 h ≈ −0.062）。方向一致、剂量单调、三 seed 无例外。

这排除了「剂量不足」这个最常见的假阴性解释：如果 4,000 updates 只是太短，那 (a) 不该三 seed
全部显著升高。**振幅确实被抬起来了，而抬起振幅这件事降低了 skill。**

---

## 2. 证据与可复现性

### 2.1 身份

| 项 | 值 |
| --- | --- |
| 起点 HEAD | `9fb7a19dbb2b4218360a15b1ff99c36385444a8d` |
| `protocol_sha256` | `75897e513d5784dda9f326fdaefc8e212530c4357d1eef7cbb2405fd1b9727e2` |
| `protocol.json` sha256 | `f754d28c04d843ada5004d98c18880a444935a612c19cb040b7c853f2040b821` |
| `code.zip` sha256 | `00f27dbe2983517ab21aa236120636af61b7d467c67afc79a4d9e8ac8b1515a0` |
| `model_code_sha256` | `59340710623ca24c7ee59ebb5711f52193ebfbaa72ed4532b41bdd987f5d9324`（与父 pin 相等） |
| `training_code_sha256` | `e767411883ce47887eb11557f3cd16c4b2ad07ca2e06f74c28f33aaa384e81b0` |
| `readings.json` sha256 | `c53d645faf97b710a8d91d71e6c9ddc84a55becf008bcc56b9af930689ed1e1d` |
| 执行闭包 | 96 个文件（含 `training/r7_spectral_geometry.py` 与驱动） |
| 产物目录 | `outputs/r7_s3_spectral_geometry_a_20261112_attempt02/` |
| CI | `R7 CPU CI` run **43363119418** on `94a3795` — **success**，9/9 步全绿 |

**可复现等级**：`config-reproducible`（协议 + 代码归档 + 三 seed receipt 齐全；GPU 训练**不是**
逐位可复现，见 §7）。**没有**逐位重放证据，因此**不得**声称 bit-identical。

### 2.2 协议里的 β_j 冻结块

`beta_weights.matched_step_c = true`，与 Step C 逐通道最大相对差
**4.71347918118202e-06**，17 个通道，跨度 **86.75497080940221×**。
`beta_normalisation.frozen = true`、`recomputed_during_training = false`。

### 2.3 (e) 预检的实测（冻结前）

8 个 case × 12 step 全部 `all_finite = true`。`amse_total / L_mse` = **[58.01630849980626,
95.21716212447576]**。

**这个剂量在冻结前就被写进了 `dose_decision`**（`scripts/r7_s3_spectral_geometry_a_protocol.py`），
并明确记下它的含义：MSE 只占 **1.04%–1.69%**，所以 on 臂训练的是 FastNet §4.1 的 **MSH 制度本身**，
**不是**「MSE + 一个小修正」。这条在跑之前写定，不能事后从结果反读。

---

## 3. (a) 的读数：振幅被抬起来了，三 seed 一致

判据口径：`amp_ratio = sqrt(Σ PSD_pred / Σ PSD_target)`，在**冻结五 lead**（6/12/24/48/72）上
按功率域聚合，取**细带（band ≥ 3）均值**；case 是 Stage P1 用的同一组八个 train case，
所以两轮的逐带数字可直接对照。

| seed | off 臂 | on 臂 | Δ |
| ---: | ---: | ---: | ---: |
| 41 | 0.725063 | 0.850191 | **+0.125128** |
| 42 | 0.719877 | 0.833694 | **+0.113816** |
| 43 | 0.723594 | 0.840167 | **+0.116573** |

**三个 seed 的 off 臂彼此相差 < 0.005，on 臂彼此相差 < 0.017**，而臂间差是 0.114–0.125 ——
臂间差比 seed 间离散大一个数量级以上。这是 (a) 判真的实质依据，不是「符号碰巧对上」。

---

## 4. (d) 的支撑：不是读数自证

决策 0109 的教训是 C2 的 `c = True` 是**字面量**，`d` 只读两个自写常量，所以**伪造三 seed skill
为 0.5 仍然 `all=True`**。本轮因此**没有**在 `evaluate_exit` 里写任何 (d) 的布尔值，只写了一个
说明字段。实际的承载是两条：

**① 代码路径**：`training/r7_objective_axes.py:41-44`

```python
enabled = [name for name in AXES if autoregression.get(name) is not None]
if not enabled:
    return None
```

`None` 在**构造任何东西之前**返回，所以 off 臂的算术与已登记配方逐位相同。

**② 会失败的测试**（`tests/test_r7_s3_spectral_geometry_a.py`）：

| 测试 | 它抓住什么 |
| --- | --- |
| `test_off_arm_matches_the_registered_arithmetic` | 传入一个**属性访问即抛异常**的对象；若 dispatcher 在 off 轮碰了 `result.forecasts`，测试以该异常失败。伪造读数无法满足它，因为那里没有读数 |
| `test_the_axis_matches_the_runner_contract` | `_update` 执行 `result.loss + constraint["total"]`；axis 若返回裸 tensor 就 `IndexError`。**这个缺陷真实存在过**，见 §6 |

两条测试在登记前实跑：**2 passed in 1.93s**。

---

## 5. (b)(c) 的读数：9/9 单元同向为负

`skill = 1 − (rmse / climatology_t2m_rmse)²`，气候态分母是冻结的 D2 参考（train-only，22 年）。

| lead | seed 41 | seed 42 | seed 43 |
| ---: | ---: | ---: | ---: |
| 24 h（on − off） | **−0.029613** | **−0.025704** | **−0.032269** |
| 48 h（on − off） | **−0.051183** | **−0.049916** | **−0.046225** |
| 72 h（on − off） | **−0.066368** | **−0.057539** | **−0.062297** |

绝对 skill（on / off）：

| seed | 24 h | 48 h | 72 h |
| ---: | ---: | ---: | ---: |
| 41 | +0.5460 / +0.5756 | +0.2594 / +0.3106 | +0.0636 / +0.1300 |
| 42 | +0.5354 / +0.5611 | +0.2613 / +0.3112 | +0.0872 / +0.1447 |
| 43 | +0.5460 / +0.5783 | +0.2791 / +0.3253 | +0.0751 / +0.1374 |

**降幅随 lead 单调增大**（24 h ≈ −0.029 → 48 h ≈ −0.049 → 72 h ≈ −0.062）。
这个梯度本身就是机制证据：β_j 归一化把权重从风场挪向位势/温度，抬高了细带振幅；
而预报越到长 lead 越依赖小尺度结构，**注入的振幅越到长 lead 越与实际信息量脱节**。

---

## 6. 本轮修掉的两个真实缺陷（都不是「可能有问题」）

### 6.1 `spectral_geometry_term` 返回裸 tensor，第一次 update 就崩

`r7_long_rollout_runner._update` 执行 `result.loss + constraint["total"]`，而该轴返回裸
`torch.Tensor` ⇒ `IndexError: too many indices for tensor of dimension 0`。**这条轴从未端到端
跑过**（Stage Q 被判 `predeclared-skip`），所以接口不匹配一直没暴露。

修法（`f333ca0`）：返回 `{"spectral_geometry": …, "beta_weighted": …, "total": …}`，与
`constraint_term`/`matching_term` 同形。反证测试：`test_the_axis_matches_the_runner_contract`。

### 6.2 权重与带索引建在 CPU 上，GPU 第一次 update 崩

实测（attempt01，seed41/42 各约 8,600 s 后）：

```
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, cuda:0 and cpu!
```

`hermitian_weights()` 与 `radial_band_index()` 建在 CPU，`spectrum_x` 在 `cuda:0`。**Stage P1 探针
从不调 `model.to(device)`（全 CPU 跑），所以 29 项测试也全在 CPU 上、看不见它。**

修法（`f333ca0`）：`weights`/`index` 跟随 `prediction.device`；`band_statistics` 显式
`.detach().cpu()`。新增两条**真 CUDA 反证测试**（无 GPU 机器上 skip）：
`test_the_loss_runs_on_cuda_not_just_cpu`、`test_the_axis_entry_point_runs_on_cuda`。

**为什么这个缺陷值得单独写出来**：(e) 预检在 CPU 上跑，它能证明「量级正确」，
**不能**证明「能在 GPU 上跑」。这是本轮最贵的一课。

### 6.3 一条被实测推翻的前提

计划 0031 与决策 0112 代价③都写「`_terms` 修复改了 `training_code_sha256`，A 须重做父迁移」。
**实测不成立**：`training/r7_spectral_geometry.py` 不在 runner 的 17 个 `SOURCE_FILES` 里，
带/不带该修复 `training_code_sha256` **都是** `e7674118…`（`git stash` 前后各跑一次实测）。
`model_code_sha256` 仍是 `59340710…`、与 pin 相等，三个父 checkpoint 的 SHA 与 pin 相等且
`load_checkpoint` 全部通过（`updates=1600`）。**父迁移未做，因为它没有对象。**

### 6.4 登记后在干净树上跑全量测试，暴露两个失败

两个都是真的，且都不是「环境敏感」，其中一个**推翻了本轮自己的提交说明**。

**① 测试拆分把一条断言遗留在错误的函数体里。** `ce03b06` 的提交说明写「no assertion dropped」。
实际上 `assert "mixed" in mixed["regime"]` 被留在了**父文件**、落在
`test_the_exit_can_fail_the_preflight` 里——那个作用域里根本没有 `mixed` 绑定。

```
E       NameError: name 'mixed' is not defined
tests/test_r7_s3_spectral_geometry_a.py:309
```

断言**没有失传**，是**放错了位置**：把它移回被拆出的
`test_the_dose_decision_can_name_the_inert_regime` 就恢复了原本的覆盖。它之所以只算一个提交的
笔误而不是永久缺口，**仅因为**孤儿断言会以 `NameError` 失败，而不是静默跳过。

**② C2 的注册表钉子自 `0440993`（2026-11-11）就是红的，不是本轮引入的。**
`0440993` 把 `spectral_geometry` 注册成第三个轴，而
`tests/test_r7_s3_derivative_matching_c2.py` 把 `AXES` 钉死成二元组
`("physical_constraints", "derivative_matching")`。

修法不是"补上第三个名字"：成员钉测试的是**注册表的长度**，那是注册该轴的那条决策的事，
不是 C2 路由的事；而且下次再注册一个轴它还会红。改成断言测试名本身声称的事——每个轴各自
路由到自己的模块，分支数按 `len(AXES) - 1` 计（最后一个轴走 fallback `return`）。

**这不是放宽，反证在案**：注册第四个轴但不加分支 ⇒ 2 个分支对 3 ⇒ 失败；加了未注册的分支
⇒ 3 对 2 ⇒ 失败。它现在抓的是一个**活**故障：`AXES` 里登记了轴却没有分支，
`auxiliary_term` 会落到 C2 的模块上**静默跑错损失**。

**这两个缺陷都不影响 Step A 的科学读数**，而且**登记时的测试面是好的**：

- ①的断言在拆分**之前**是对的——`git show ce03b06~1:tests/test_r7_s3_spectral_geometry_a.py`
  的第 398 行，它确实在 `test_the_dose_decision_can_name_the_inert_regime` 体内。拆分是
  **纯行数措施**（为把父文件压回 R-021 的 400 行目标），发生在训练、读数与登记**之后**。
  也就是说：**读数所用的测试量尺当时是完好的，是拆分把它弄坏的**，而 `ce03b06` 的提交说明
  「no assertion dropped」没兑现。
- ② 更是本轮之前就红的（`0440993`，2026-11-11），与 Step A 无因果关系。

所以本页 §4 对 (d) 的支撑与 §2.3 的 (e) 读数**都不需要修订**；需要修订的只有一处表述：
§6.4 的两个缺陷是**登记之后**的测试面维护问题，不是读数问题。

**教训**（与 `e16362f` 同源，值得并置）：那次的教训是「NameError 在函数体内，import 与
不执行该函数的测试都看不见」；这次是它的变体——**拆分脚本搬走了函数，却把一条断言留在了
原地**，而孤儿断言恰好以 `NameError` 失败，所以它被抓住了。若那条断言碰到的是一个**上游
作用域里存在的名字**（比如把 `mixed` 换成任何模块级名字），它会静默断言在错误的对象上。
拆分后**逐条核对断言数**是比「目视比对」更可靠的检查，本轮缺的正是这一步。

---

## 7. 成本（实测，不是估计）

| 项 | 时间 | 说明 |
| --- | ---: | --- |
| 训练（3 seed × 2 臂） | 44,310.1 s = **12.308 GPU-h** | off 臂 ≈ 1.17 s/update；**on 臂 ≈ 2.33–2.58 s/update** |
| 五 lead 评分（6 臂） | 13,288.0 s = **3.691 GPU-h** | |
| **attempt02 有效合计** | **15.999 GPU-h** | |
| `prepare` | 680.6 s（CPU） | β_j 重导 + (e) 预检 |
| `archive` | 6.3 s（CPU） | |
| `reading` | 668.5 s | 含振幅读数的 3×2×8×12 rollout |
| **attempt01 浪费** | **4.778 GPU-h** | §6.2 的 device 崩溃，2 worker × ~8,600 s，零读数 |
| **`stage-seeds` 超时截断浪费** | **1.048 GPU-h** | seed43 on 臂跑到 1620/4000 被 workflow 的 9.5 h 调用上限杀掉 |
| **本轮总计** | **21.825 GPU-h** | |

**on 臂比 off 臂慢 2.0–2.2×**：AMSE 项要对 17 通道 × 12 step 各做一次 `rfft2`。
计划按 1.1096 s/update 估 ≈12 GPU-h（按 20 估）；**实测有效 16**，在估计的安全侧，
但**加上两次真实浪费后是 21.8**，略超计划的「按 20 估」。

**软/硬预算**：`per_seed_seconds = 32,400`（9 h），`hard_cap = 64,800`（18 h）。
实测每 seed 5.65–6.4 h + 1.48–1.50 h 评分 ⇒ **无一 seed 触软预算**，`soft_overrun_seconds = 0.0`。

---

## 8. 判定点（逐条，按 `docs/goals/s3-physics-informed-sequence.md` §3.1）

| 判据 | 真/假 | 依据 |
| --- | --- | --- |
| **(a)** 细带 `amp_ratio` 相对同 seed off 臂升高 | **真** | 三 seed Δ = +0.125128 / +0.113816 / +0.116573，全部 > 0 |
| **(b)** 72 h t2m skill 三 seed 配对 Δ 全 ≥ 0 | **假** | −0.066368 / −0.057539 / −0.062297，全部 < 0 |
| **(c)** 24/48 h skill 配对 Δ 三 seed 全 ≥ 0 | **假** | 24 h 三 seed 全负、48 h 三 seed 全负 |
| **(d)** off 路径逐位等价 | **支撑在位** | `r7_objective_axes.py:43` 的 `None` 早退 + 两条反证测试实跑 2 passed |
| **(e)** 预检：β_j 非全相等且 amse 全有限 | **真** | 跨度 86.755×；8×12 全有限；剂量 58.0–95.2 已在冻结前写定 |

**⇒ 按预声明：(b) 与 (c) 为假 ⇒ 永久关闭 `spectral_geometry` 支。**
Step B 不在本 objective 授权内（须新 ADR 0113，且只在 A 失败后）；本次**不开 B**，
留待用户在读过本页后决定。

---

## 9. 这个结果意味着什么（诚实解读，不夸大也不缩小）

**已确认（实测）**：

1. β_j 归一化**确实**把细带振幅抬起来了，三 seed 一致，效应量比 seed 间离散大一个数量级。
2. 同一改动让 t2m skill 在 **9/9 个 (lead, seed) 单元**下降，降幅随 lead 单调增大。
3. 剂量足够让 on 臂训练论文 §4.1 的 MSH 制度本身（MSE 仅占 1.04%–1.69%），
   所以这是对「β_j 加权的 AMSE-analogue 作为训练目标」的直接检验，不是剂量不足。
4. 修复了两个会让训练崩溃的真实缺陷（§6.1、§6.2）。

**是推测，不是结论**：

- **为什么**抬高细带振幅会降低 skill。最自然的解释是：AMSE 的 `max(P_x,P_y)` 形式
  （决策 0110 的三项近似之一）在模型振幅**不足**时推动放大，但真实预报的细带误差里
  有相当部分是不可约的噪声；强行放大把噪声一起放大。**本轮没有测这个机制**。
- 是否「β_j 归一化在更大的模型/更长训练/别的损失形式下也如此」。n = 1 模型、单实例、
  单分辨率、4,000 updates。

**不得**从本页得出的结论：

- 不得说「物理信息无用」。本页测的是**一个**归一化方案在**一个**轴上的 on/off，
  且 (a) 显示它精确地作用在了预期位置。
- 不得说「复现/否证了 FastNet 式 (8)/(10)」。本仓是区域 `rfft2` 分带的 **AMSE-analogue**
  （决策 0110），不是球谐 AMSE；论文的 β_j 是逆球谐 AMSE，本页的 β_j 是它的区域分带类比。
- 不得把 (b)(c) 的降幅当成精确效应量：三 seed 是配对比较，不是显著性检验。

---

## 10. 局限（如实列）

1. **单模型、单实例、单分辨率**：65×65 multi-year 实例、dim-192 depth-4 trunk。129×129 wide
   实例是不同 store，未覆盖。
2. **4,000 updates**：远短于 FastNet 的微调日程（36,525 + 4×7,305）。(a) 的三 seed 一致升高
   排除了「项没起作用」，但**不能**排除「更长的适应期会改变 (b)(c) 的符号」。
3. **β_j 来自单一年（2001）**：年际方差本轮**未测**；论文声称极小，但那是论文的存储，不是本仓的。
4. **on 臂同时改了两件事**：损失形状从「池化 band 和」变成「逐变量和」，因为逐变量权重无法
   施加在已池化的量上。**严格地说，(b)(c) 的负结果归属到「β_j 归一化 + 逐变量分解」这一组合，
   不单独归属到 β_j。** 这是本轮设计的一个真实代价，事前写在计划 §6 风险 3 附近，但未拆开。
5. **振幅读数的 case 是 train manifest 的八例**（`ZarrLongRolloutDataset` 只接受 `train.jsonl`）。
   这是 in-sample；它使两臂同时受益于记忆化，因此**偏向于不发现臂间差异**，不是偏向假阳性。
6. **不是逐位可复现**：GPU 训练无可比位承诺（runner 自己记 `reproducibility:
   config-reproducible`）。本页声称的只有配置级可复现 + 三 seed 一致的方向。

---

## 11. 复现命令

```bash
# 起点与轮次
.venv/bin/python tools/r7_round.py state

# Step C 的门（应打印 verdict = open-a）
.venv/bin/python scripts/r7_s3_gap_decay_and_beta.py --out /tmp/stageA_stepc_gate.json

# (e) 预检 + β_j 重导（CPU，约 11 分钟）
.venv/bin/python -c "from scripts import r7_s3_spectral_geometry_a_protocol as p; import zarr; \
s=zarr.open_group(str(p.curve.STORE), mode='r'); c=p._store_constants(); \
b,_=p._assert_beta_matches_step_c(p._derived_beta(s), c['channels']); print(p.preflight(41,b)['dose_band_observed'])"

# (d) 的载体（两条反证，必须都过）
.venv/bin/python -m pytest -q tests/test_r7_s3_spectral_geometry_a.py::test_off_arm_matches_the_registered_arithmetic \
    tests/test_r7_s3_spectral_geometry_a.py::test_the_axis_matches_the_runner_contract

# 读数与判定（GPU，约 11 分钟）
.venv/bin/python tools/r7_round.py stage reading
.venv/bin/python -c "import json; d=json.load(open('outputs/r7_s3_spectral_geometry_a_20261112_attempt02/readings.json')); print(json.dumps(d['exit_conditions'], indent=2))"
```

---

## 12. 相关记录

- 目标长文（唯一执行源）：`docs/goals/s3-physics-informed-sequence.md`
- 设计与成本：`docs/plans/0031-physics-informed-sequence-c-a-b.md`
- 准入裁决：`docs/decisions/0112-sequence-admission-beta-normalisation.md`
- 上游读数：`docs/R7_S3_SPECTRAL_GEOMETRY_RESULT.md`（Stage P）、决策 **0110**、**0111**
- 教训来源：决策 **0109**（(d) 不得是字面量）
- 证据索引：`docs/R7_EVIDENCE_INDEX.jsonl` 的 `s3-spectral-geometry-a-negative`
