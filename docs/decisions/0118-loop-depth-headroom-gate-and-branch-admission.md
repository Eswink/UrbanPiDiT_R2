# 0118 — 动循环架构之前先量 headroom：G0 三子句决策表与条件式 G1-B 准入

- **日期**：2026-11-25
- **状态**：proposed
- **范围**：`docs/plans/0036-loop-depth-headroom-before-architecture.md` 与
  `docs/goals/loop-depth-headroom-before-architecture.md` 的**准入判定**。本轮 **0 GPU-h**：
  只跑一次只读诊断（`diagnose_r7_gain.py`，val split，不训练、不改 `model/**`）。
  **本 ADR 不实现 G1-B、不冻实验协议、不跑训练。**
- **依据**：决策 **0048**（容量轴封顶已触发）、**0057** §4.4（须论证不是同一族）、
  **0077**（H-C 否决，改动混合强度须新 ADR）、**0030/0038**（方向内执行授权）、
  `docs/R7_65_PREDIAGNOSTIC.md` §3（唯一的既有 K 轨迹读数）、
  `training/r7_gain_oracle.py` 的只读实测接口、`docs/R7_LOOP_TRANSFORMER_LITERATURE_AUDIT.md`

## Context

**① 本仓已经是 looped transformer，不是「非 loop 架构」。** 实测（`ProcessForecastCoReasoner`，
`in_channels=17, dim=192, depth=4, heads=4, window_size=4, patch_size=2, anchored=8, free=8`，
65×65 输入）：

| 组件 | loop 文献里的对应物 | 实测 |
| --- | --- | --- |
| `CoarseEncoder` 的 4 个不共享 `WindowAttentionBlock` | **prelude** | 1,806,144 参数 |
| `GenericRecursiveCell` 被 K 次复用（`model/recursive_weather_r7.py:406`） | **recurrent core（weight-tied）** | 742,272 参数 = 全模型 **26.5%** |
| `correction_head`（`model/coarse_forecast.py`） | **coda** | 88,913 参数 |
| `deep_supervised_forecast_mse` 每步草稿都进 loss（`training/r7_recursive_losses.py:151-157`） | **deep supervision** | 权重 `linspace(1,2,K+1)/sum` |

即已是 Huginn 式 **prelude–recurrent–coda sandwich + deep supervision**。所以「换成最新 loop 架构」
不是一次替换，而是**重开一批已关闭的轴**（容量 0048、H-C 0077、H-A、深监督组织、typed）。

**②「K 4→8 无效」不是关于 loop 的证据。** `record:s3-reasoning-depth` 是 **unresolved**（三 seed
只完成一个、reading 从未运行、单 seed Δ≤0.009 K 与 seed 间波动同量级）。它测的是「同一个 16-token
状态被多跑几遍」，不是「循环能否改写空间表征」。

**③ 实测：循环在成本上已经不便宜。** 单 batch、65×65、RTX 3090、`torch.no_grad` 前向：

| K | 前向 ms | loop 占前向比例 |
| ---: | ---: | ---: |
| 0 | 7.0 | — |
| 3 | 13.6 | **49.7%** |
| 4 | 16.8 | 59.1% |
| 8 | 24.2 | 71.7% |

训练步 K=1→3 的 loop 部分约占总步时 **25.6%**（6.5 ms/step）。⇒ 加深循环不是免费的，
每加一步都要有可测的理由。

**④ 三条结构差异，但**没有数据决定先补哪一条**。**

| 差异 | 实测位置 | 文献地位 |
| --- | --- | --- |
| 循环只更新 16 个 process token；1089-token 的 `context` 全程逐位不变，每步只做一次加性读出 | `model/process_forecast_r7.py:517`、`model/process_step_r7.py:112` | 与 Huginn/Ouro「循环改写全分辨率表征」根本不同 |
| 无 per-iteration 门控输入注入 | `model/recursive_weather_r7.py:402`（`z=latent.expand(B,-1,-1)`） | 有受控消融支持（arXiv:2610.23036） |
| 无 tied-depth 残差缩放 | `model/recursive_weather_r7.py:47` `GenericRecursiveCell`（pre-LN + 裸残差） | 有理论 + 实证（arXiv:2607.13491），但该文是 post-LN |
| loop-index embedding | 仅 `model/local_solver_state_r7.py:126` 的 solver 用正弦步编码 | **Tier 3**：Huginn/Ouro 普遍省略，实测中性或略负 |

**⑤ 既有 K 轨迹证据指向饱和/欠射，而不是过放大。** `docs/R7_65_PREDIAGNOSTIC.md` §3（4 个 val
case、800 步 checkpoint）：K=1 贡献全部收益，K=2 只有 1/3，**K=3、K=4 净效果转负**；相邻修正
cosine 0.864 → 0.939 → **0.982**；修正范数 0.043 → 0.023 单调下降。**若这条读法成立，则
「残差缩放」这条路在机制上就是错的**（它假定更新过放大）。该证据只有 4 个 case，不足以决策。

**⑥ 与 0048（容量轴永久关闭）的关系。** 0048 第 3 条把封顶枚举成四个禁止动作：
不得重试 `depth`、不得再扫 `dim`、不得做宽＋深组合、不得以「再调一点容量」为名复活该支；
其容量口径是**参数量**（`0048:65-68`）。本 ADR 的 G1-B（若被准入）满足：
`state_dict` 键集**逐个相同**、参数计数**逐位相同**、`dim`/`depth` **逐字不变**、父 checkpoint
张量**逐位可装**。但本 ADR **不**用「参数没变」当挡箭牌：真正的族判别是 G1-B 的**主对比 A−B**
——两臂的参数量、算力、键集完全相同，唯一差别是残差增益**是否依赖 visit 序号**。这不是容量对比，
是**信号传播/优化**对比，因此不落在 0048 的枚举内。

**⑦ 与 0077（H-C 否决）的关系。** 两者都不触碰 `model/global_mixing_r7.py` 的 `MIXING_MODES`、
不改变注意力对规模、不换网格、不改感受野。G1-B 也不做低秩全局算子。

**⑧ 七 gate（并行准入）。**

| gate | 本轮（G0） | G1-B（若准入） |
| --- | --- | --- |
| 假设独立 | 过：单轮只读诊断，结论不依赖其他臂 | 过：A−B 可单独证伪 |
| 产物隔离 | 过：`outputs/r7_s3_loop_depth_headroom_20261125/seed{N}` 排他 | 过：新 `outputs/` stem |
| 计算资源 | 过：只读推理，两卡空闲，共驻余量门已核（各 24 GiB free，0 计算进程） | 过：由统一父调度按卡串行 |
| 数据实例 | 过：复用已冻结 `train_data_identity` 的 val manifest | 同 |
| 可变状态 | 过：不写 checkpoint、不改 store | 过：新 attempt |
| 可独立验证 | 过：诊断 JSON 自带 `checkpoint_sha256`、`validation_manifest_sha256`、`forecaster_unchanged` | 过：新 protocol + digest |
| 多重比较 | 过：不读 test，`r=0` | 过：`r=0`，只作开发筛选 |

## Decision

1. **本轮只做 G0，0 GPU-h，不训练，不改 `model/**`。** 用一个已存在的只读入口
   （`diagnose_r7_gain.py` → `training/r7_gain_oracle.py:collect_process_errors`）在三个已注册
   incumbent checkpoint 上测 K=1..8 的逐档归一化 MSE 与 greedy/optimal 深度分布。

2. **G0 的三个子句与阈值在此冻结，读数前不得改动。** 记 `m_k` = `mse_by_step[k-1]`
   （等通道、纬度加权归一化 MSE；`atmos_target` 是 **+6 h** 场）：

   - **(a) 饱和**：对每个 seed，`(m_k − m_{k+1}) / m_k < 0.01` 对所有 `k ∈ {3,4,5,6,7}` 成立
     （即 k≥3 之后没有任何一步带来 ≥1% 的相对改进）。**(a) := 3/3 seeds 成立。**
   - **(b) 病态**：对每个 seed，`(m_4 − m_3) / m_3 > 0.001`（k3→4 相对**上升** >0.1%）。
     **(b) := ≥2/3 seeds 成立。**
   - **(c) 深度可用**：对每个 seed，`(m_4 − m_8) / m_4 ≥ 0.02`（k4→8 相对改进 ≥2%）。
     **(c) := ≥2/3 seeds 成立。**

3. **决策表（冻结；按行序优先匹配，第一行为准）。**

   | 条件 | 判定 | 允许的下一步 |
   | --- | --- | --- |
   | **(b) 真** | `admit-g1b` | **允许**派生 G1-B（残差缩放）的 ADR/brief/测试与训练轮；仍须新 brief + 冻结 |
   | **(b) 伪 且 (a) 真** | `saturated-negative` | **不开** G1-B。结论=「多跑几遍同一个 cell 无用」；下一步只写 C 的新 ADR（循环改写空间表征），零训练 |
   | **(b) 伪 且 (a) 伪 且 (c) 真** | `depth-usable-unresolved` | **不开** G1-B。只许停机/策略方向；**禁止重开 K**（`record:s3-reasoning-depth` 已登记勿重跑） |
   | 其余组合 | `unresolved` | 停，写原因与恢复命令 |

4. **fail-closed 规则（比阈值更硬）。** 任一 seed 缺失、任一 `m_k` 非有限、`max_steps < 8`、
   身份门失败（`model_code_digest()` ≠ `59340710…`）、或 `forecaster_unchanged != true`
   ⇒ 整体判 `unresolved`，**不得**用可用 seed 的子集外推，**不得**用 `null`/缺失值当作允许开 G1-B
   的依据。三 seed 缺一即 `unresolved`。

5. **G1-B 的唯一可站立形态（若被准入；本 ADR 不实现）。**
   - **唯一因子**：每次 visit 的残差分支增益。`GenericRecursiveCell` 增加 `visit_gain` 模式，
     默认 `None` = 现有实现**逐操作相同**（不建 buffer、不增参数、不加 RNG 抽取）。
   - **两臂**：**A `visit_damped`**（`g_k = 1/sqrt(1+k)`，k 为 visit 序号）vs
     **B `uniform_damped`**（`g = 1/sqrt(1+(K−1)/2)`，与 A 同平均阻尼但**不依赖 k**）。
     **C = 已注册 incumbent（pinned，不重训）**。
   - **主对比 A−B**（隔离「visit 对齐」与「整体缩小步长」）；**B−C 只作描述性**并同时声明参数差（0）。
   - **身份证据**：父 checkpoint 张量逐个**逐位相同**，只需重签 `model_code_sha256`
     （`scripts/migrate_r7_parent_state.py`），并把每 seed 的 `state_digest` 与归档父逐位比对。
   - **预声明出口**：①72 h `interior_32` t2m 的 A−B 三 seed 同号为负；②6/12 h 的 A−B seed 均值 ≤0；
     ③桥：A/B 首 update loss 与归档父同 update 逐位相同；④参数/键集/父 `state_digest` 逐位相同。
     任一失败即关闭该支，**不**加 seed、**不**扫增益函数族、**不**调 K。
   - **不得**称它为「loop 架构升级」或「容量提升」；只能称「visit 对齐阻尼的新假设」。

6. **明确不做（逐条有理由）。**
   - **不替换架构**、不加新 loop 算子、不动 `MIXING_MODES` / 感受野 / 网格（0048、0077 保持关闭）。
   - **不做** loop-index embedding、per-loop LoRA、更大 K（Tier 3 或已登记勿重跑）。
   - **不预设**增益函数族的取值（`1/sqrt(1+k)` 是 G1-B 的**待证**定义，不是已验证的最优形式；
     若 G1-B 被准入，该式须在 brief 里作为**唯一因子**冻结，不许扫）。
   - **不读 test**、`r=0`、不改 `campaign-state` 指针、不加账本行（0 GPU-h 无机器核数，
     按主计划 §7 例外并在 §8 写明）。

## Consequences

**变容易的：**

- 「该不该动 loop 架构」从一次昂贵的架构轮变成一次 0 GPU-h 的只读诊断；**并且它能否决 G1-B**
  ——(a) 真且 (b) 伪时，本 ADR 明确禁止推进残差缩放，避免把一次「方向错」的架构改动做成一轮训练。
- 阈值与决策表在读数前写死，读数后无法事后调整；缺 seed / 非有限 / 身份失败一律 fail-closed。
- 与 0048 / 0077 的关系写成了可核对的形式（键集、参数量、`MIXING_MODES`），后来者不必重新论证。

**变难 / 代价（如实列出）：**

1. **G0 的主判据是 +6 h 目标上的机制诊断，不是 72 h 技能结论。** `atmos_target` 是
   `physical_targets[:,0]`，即 +6 h 场。因此 (a)(b)(c) 谈的是「循环在这一时次目标上是否还有
   边际收益」，**不能**外推成「72 h 会变好」。72 h 轨迹若未安全实跑一律记 `null`，
   且 `null` 不参与开 G1-B。
2. **本 ADR 零读数时不是证据，连否决都不是。** 它只冻结了判据与准入规则；G0 的读数在它自己的
   证据页里登记。
3. **G1-B 的增益式来自一个 post-LN 的论文（arXiv:2607.13491），我们是 pre-LN。** 直接迁移是外推，
   必须写进 limitations，且**不得**引用该文数字当我们的预期。若 G0 判 `saturated-negative`，
   这条外推连测的机会都没有——这是刻意的。
4. **G0 的样本上限是 1024（实现硬上限）**，且 val manifest 是时间顺序的：取前 1,024 条只覆盖
   2022-01 至 2022-09，**缺秋冬**。因此本轮读数不含四季均衡，不能称「全年」。
5. **G1-B 的成本是估算不是实测**：按最近一轮同族实测 1.11–1.46 s/update + 3,200 s/5-lead 评分，
   2 臂 × 3 seed × 2,400 updates ≈ **11.1 GPU-h**，双卡三波 ≈5.6 h 墙钟。训练实际速率未测。
6. **G0 需要模型代码身份与归档一致。** 实测 `model_code_digest() = 59340710…` 与 attempt02 协议
   逐位相同，所以本轮可以直跑；一旦 `model/**` 有任何改动，本轮的所有 JSON 都必须用归档
   `code.zip` 重放，不能改比对逻辑。

**备选方案与否决理由：**

- **备选 A：直接开 G1-B（残差缩放）训练轮。** 否决：既有 K 轨迹指向饱和/欠射而非过放大
  （Context ⑤）；在没有 headroom 读数时开轮，等于把「方向可能错」的假设做成 6–12 GPU-h 的实验，
  且事后无法区分「阻尼无效」与「本来就不需要更多步」。
- **备选 B：直接开 C（SMELT 式把循环移进空间栈）。** 否决：C 要改 backbone 结构 ⇒ 参数量与算力
  都对不齐，须自己的 ADR 并论证与 0048 不同族；且 G0 尚未告诉我们瓶颈在深度还是在状态容量。
- **备选 C：只写文档，不跑 G0。** 否决：本轮 G0 是 0 GPU-h 的只读推理，没有理由把唯一能判方向
  的读数留到下一轮再花一次上下文。
- **备选 D：把 G0 的阈值定得更宽松（例如 (b) 用 1% 而非 0.1%）。** 否决：`docs/R7_65_PREDIAGNOSTIC.md`
  §3 的 K=3→4 在 800 步 checkpoint 上是 **+0.38%**（0.12675 → 0.12723）；用 1% 会把一个已观测到的
  病态形态判成正常。

## 未决（留给下一轮）

1. **G0 读数本身**：三 seed 的 `mse_by_step` 与决策表命中行，登记在
   `docs/R7_S3_LOOP_DEPTH_HEADROOM_RESULT.md`。
2. **72 h 逐 lead 轨迹**：本轮不测。若将来要测，须证明 long-rollout 路径不会把 72 h 真值泄漏进
   初始化，并单独冻结。
3. **C 的方案形态**：若 G0 判 `saturated-negative`，C 的 ADR 必须回答「为什么改写空间表征不是
   0048 的容量轴」以及「为什么不是 0077 的感受野轴」——这两条现在都还没有论证。
