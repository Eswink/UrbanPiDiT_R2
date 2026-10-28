# 0066 — S4 primary 估计量的预注册，以及守门未达时判决的可判定性

- **日期**：2026-10-28
- **状态**：accepted
- **范围**：`docs/goals/s3-matched-exposure-reeval.md` §4.1（Track 0）的 **S4 判决口径**。
  只解决两件事：① primary 门的估计量歧义；② 守门未达时**判决是否仍可判定**。
  **不改**任何已冻结判据、**不重算**任何读数、**不改** `protocol.json`（其
  `protocol_sha256 = 0dd100dc…` 保持可读）、**不改** `r = 1` / `alpha_1 = 0.025` 的登记。
- **依据**：`docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md:69-71`（守门定义 + 「若守门不成立，
  可报告 t2m 获益但不达到本终极 package 门」）；`training/r7_climatology_skill.py:120-126`
  （本项目**权威** skill 定义 = 池化 MSE 之比）；`docs/R7_S4_FY_2016_CONFIRMATION_RESULT.md`
  与 §5.1（读数与更正）；`outputs/r7_s4_fy_2016_20261026_attempt01/protocol.json`
  （冻结的 `family.primary` / `family.guard`）；决策 0059（实例机制）。

## Context

**① 歧义是真的，且是我自己在冻结协议里写进去的。** 协议 `family.primary` 把 primary 定义在
`skill = 1 − mean MSE_model / mean MSE_climatology`（**分组均值之比**），而同一条 `family`
的 `interval_series` 又写「per-case skill `1 − mse_model/mse_climatology`」并交给
`simultaneous_paired_interval` 取均值。两者**不是同一个量**，且实测在 48/72 h **符号相反**
（total set：比值 skill +0.2330 / +0.0384，而 per-case 均值的同时下界 −0.2670 / −0.5230）。
证据页因此**没有**宣布 primary 通过或不通过。

**② 但"权威定义"并不模糊 —— 模糊的是我写的 `interval_series`。**
`training/r7_climatology_skill.py:120-126` 是本项目**唯一**的气候态 skill 实现：

```python
mse_forecast   = self.forecast_squared_error / self.initializations
mse_climatology= self.climatology_squared_error / self.initializations
skill[usable]  = 1.0 - mse_forecast[usable] / mse_climatology[usable]
```

即 skill **是池化（分组均值）的 MSE 之比**，**不是** per-case skill 的均值。
R1/FY 两轮驱动的出口判据也用同一个定义（`full_t2m_skill` 由 `boundary_rmse.csv` 的分组
RMSE 反算）。⇒ 协议 `family.primary` 与全仓口径一致；**`interval_series` 那一行才是笔误**。

**③ 守门是合取项，且它的失败不依赖 ①。**
`docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md:69-71` 逐字写着：守门要求
「每 seed 的相对 MSE 变化均 `<=0`；同时区间上界 `<=0`」，并明写
「**若守门不成立，可报告 t2m 获益但不达到本终极 package 门**」。
实测（`readings.json`，sha256 `dd3a0709…` 未变）在**每一种**读法下都不成立：

| 读法 | cell 数 | 同时上界 ≤0 | 点估计 ≤0 |
| --- | --- | --- | --- |
| 协议字面（全部 6 族） | 90 | **25/90** | **78/90**（12 个 >0） |
| 只算 total set 一族 | 15 | **5/15** | 15/15 |
| 逐变量（各 30 格） | u10 30 / v10 30 / mslp 30 | **1/30 / 9/30 / 15/30** | 21/30 / 27/30 / 30/30 |

最宽的读法（某变量只要任意一格过就算过）能"救" u10 与 v10 —— 但那**不是**协议字面
（协议要求「上述各确认组」全部），且 u10 的 5 个 lead 在 total set 上全不过。
⇒ **守门在最忠实于协议文本的读法下未达，这一事实与 ① 无关。**

## Decision

1. **primary 的估计量登记为「分组均值之比」**（`1 − mean MSE_model / mean MSE_climatology`），
   依据 `training/r7_climatology_skill.py:120-126` 的权威实现与 R1/FY 两轮出口的同一口径。
   协议 `family.primary` 本来就这么写；**本次登记的是对 `interval_series` 那一行笔误的裁断**：
   该行不得被读成"primary 换成 per-case 均值"。
2. **同时区间必须与估计量匹配**：primary 若要用同时区间，需要**比值感知**的块 bootstrap
   （在重采样块上重算 `1 − Σmse_model/Σmse_clim`），而不是对 per-case 序列取均值再标准化。
   **本轮不实现它**，理由见 Decision 第 4 条。
3. **守门未达 ⇒ 明写「未达到终极 package 门」，并且这不是"判决未宣"。**
   按 `:71` 的成文后果，即使 t2m primary 全部通过，结论也只能是
   「t2m 获益但未达 package 门」。S4 的 `outcome_class` 保持 `unresolved`、
   `candidate_state` 保持 `not-candidate`，但**理由要改写成"守门未达"**，
   而不是目前的"估计量歧义 ⇒ 无法判定"。
4. **因此本轮不需要为重算 primary 花 GPU。** primary 的比值区间既不能把守门的失败救回来
   （守门是独立合取项），也不会改变"未达 package 门"的结论。
   花一次重算去把 primary 变成"有区间"是**为了漂亮而付费**，不做。
   **若将来守门在某个新实例上通过，那时才需要比值感知 bootstrap**，并把它的实现
   预先登记（而不是事后挑）。
5. **修正"per seed ≤0"子句的**不可评估**状态：`readings.json` 只存 seed 平均序列 ⇒ 该子句
   **未测**。诚实记法：守门在两个子句上都未通过 —— 一个**未达**（同时上界），一个**未测**
   （per seed）。两者都**不是**通过。后续轮（R2–R5）的驱动必须同时保存逐 seed 值。

## Consequences

**变容易的：**

- S4 不再是"悬而未决"，而是**有明确结论的**："t2m 有获益，但守门未达 ⇒ 未到 package 门"。
  这比现状（"估计量歧义导致无法判定"）信息量更大，也更符合冻结合同的成文后果。
- 省掉一次不必要的重算（约 2 GPU-h）与一次比值 bootstrap 的实现工作。
- 把「权威 skill 定义在 `r7_climatology_skill.py`」固定下来，后续轮不必再猜。

**变难 / 代价（如实列出）：**

- **primary 在本轮仍没有同时区间**。Decision 第 4 条是**主动放弃**这个区间，理由是守门已否决
  package 门。代价是：若有人日后只读 primary 表，会看到一个没有区间的 skill 序列，
  必须同时读到守门那段才知道为什么不给区间。缓解：证据页把这两段相邻放置并互相引用。
- **这是"解释"而非"重算"**：我没有重新生成任何数字。若我的读法与别人不同，
  分歧会在文字上而不是在数据上暴露 —— 这是比数字更难自动发现的一类风险。
  缓解：把每种读法的计数都列进表（Decision 第 3 条的 3 行），让读者自己核对。
- **承认了自己写进冻结协议的一处笔误**（`interval_series`）。这削弱了"协议一次写对"的预期；
  但它写的是**实现提示**而非判据，且 `family.primary`（判据）是对的，故不触及判据变更。
- **"per seed"子句未测这一点无法补救**：要补就得重跑评分（每次 seed 的 per-case MSE 都要落盘），
  约 2 GPU-h。本轮不做，因为守门在另一个子句上已经未达 —— 补测不会改变结论。
  代价是这个子句将永久停留在"未测"，如实写进 limitations。
- S4 的结论因此**不能**被读成"FY candidate 在 2016 上被否决"：被否决的是
  **package 门**（含关键变量守门），t2m 那一路的获益是真实的（比值 skill 五 lead 全 >0，
  且逐 lead 优于同数据 incumbent）。

**备选方案与否决理由：**

- **备选 A：把 primary 改成 per-case 均值，与冻结工具对齐。** 否决：那会**改判据**
  （AGENTS.md 禁止事后放宽/更改已冻结判据），且与全仓权威 skill 定义相反；
  更糟的是它会让 primary 与 R1/FY 两轮**不可比**。
- **备选 B：实现比值感知 bootstrap 并重算。** 否决（本轮）：守门是独立合取项且已未达，
  重算 primary 不改变 package 门结论；在结论已知的情况下花 GPU 换一个更漂亮的 primary 表
  属于为美观付费。
- **备选 C：维持"估计量歧义 ⇒ 判决未宣"。** 否决：这与 `:71` 的成文后果冲突 ——
  合同已经写明守门不成立时该怎么办，"未宣"会让一条本可判定的结论停在模糊里，
  反而降低可审计性。
- **备选 D：补跑 per-seed 读数把"未测"补上。** 否决：约 2 GPU-h，且守门在"同时上界"
  这个子句上已未达，补测不改变结论。如实写"未测"优于花钱买到同一个结论。
