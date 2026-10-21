# 0049 显式 tendency 输入单因素轮的授权边界（不重开容量/架构轴）

- **日期**：2026-10-21
- **状态**：accepted
- **依据证据**：`docs/goals/main-model-climatology-campaign.md` §8 的
  2026-10-21「重新激活：显式 tendency 输入作为独立表示因子」块（本轮开始时尚未提交）；
  `docs/decisions/0030-autonomous-execution-and-direction-boundary.md`（方向内实验设计下放）；
  `docs/decisions/0048-capacity-axis-scope-and-cap.md`（容量轴与 H-A…H-D 已关闭）；
  `docs/goals/s3-tendency-input.md` §3（本轮冻结的预声明出口与设计）；
  `docs/R7_S3_GRADIENT_SURGERY_RESULT.md`（上一轮 mixed/negative，支已关闭）；
  `training/r7_experiment.py:model_code_digest` 与 `scripts/migrate_r7_parent_state.py`
  （模型 revision 改变后父迁移的既有唯一路径）

## Context

运行时要求从 `docs/goals/main-model-climatology-campaign.md` §8 恢复下一独立单因素轮。
2026-10-20 的「暂停并等待具名授权」块因此被 2026-10-21 的重新激活块取代，但只限于**显式
tendency 输入**这一表示因子：容量轴（决策 0048）、H-A…H-D 架构轴、PE/scale/typed/ordering/
anchor/rollout-dose/EMA/dense/final-only/spatial-gradient/zero-base decoder/gradient-surgery
各支仍然关闭，数据扩充按 D3 暂缓。§8 块是工作树里的未提交文本；本记录把它变成可审计的
`accepted` 决策，并规定 freeze 的前置条件。

该因子之所以不是已关轴的重命名：它只改 encoder 的**输入表示**，把 `D_t = X_t − X_{t−1}`
作为第三组 17 通道与两个历史帧拼接（34→51 列），新增一个独立的 17→dim patch（bias=False，
零初始化，+13,056 参数）。模型其余层、persistence decoder base、数据、损失、2400 updates、
LR/warmup、K/depth/dim、物理权重、seed、评分域与气候态分母全部逐字不变；不增加容量、不切换
架构、不改反向或数据制度。

## Decision

1. **授权一轮**：允许在 `docs/goals/s3-tendency-input.md` 冻结并执行候选单臂 × seed 41/42/43，
   对照为 pinned `s3-anchor-isolation-result`，外部参照为 pinned `s3-wide-interior-dose`。
2. **单一因素**：`tendency_input` False→True；`D_t = X_t − X_{t−1}`，同一归一化输入空间；
   独立 `Conv2d(17, dim, 2, stride=2, bias=False)` 零初始化；参数增量 13,056 必须作为
   **已声明的容量混杂**报告。不得增加 parameter-matched 控制臂或任何 sweep。
3. **预声明出口**：沿用本轮 brief §3 的 (a)–(e)，不新增阈值。全真→`candidate_state:
   needs-review`，只提请独立复核，不自动进 S4；任一 (a)–(d) 假→**永久关闭**该支（不扫
   tendency 定义/缩放/符号、不加 seed、不改 updates/LR/warmup/K/depth/dim、不与已关轴组合、
   不换名）；(e) 假只记 `identity_unresolved=true`，是工程身份失败而非科学判定。
4. **null 口径**：因为 `D_t` 是两个既有帧的线性函数，支路不增加信息；只有最终 patch 权重
   恰好全零或每个配对 Δ 在记录精度下恰好 0.0 才记 null/inert。小而非零效应按 (a)–(d) 判读。
5. **身份与迁移**：模型代码改变后必须用 committed `scripts/migrate_r7_parent_state.py` 在
   全新排他目录重跑父迁移；不得复用旧 revision checkpoint，不得放宽
   `load_checkpoint` 的 `model_code_sha256` 校验。
6. **前置与预算**：本记录与 §8 块必须在 `freeze` 前已提交；`test.jsonl` 未读、r=0、S4 未启动。
   `per_seed_seconds=21600`、`planned_seconds=43200`（软）、`hard_cap_seconds=86400`（硬）。
7. **不改变既有封闭**：本记录不重开决策 0048 的容量轴或任何架构/训练信号支，也不授权数据扩充。

## Consequences

**正面**：

- 该表示因子获得明确、可引用的授权边界，不必在灰色地带推进，也不会被误读为重开容量/架构轴。
- 出口、null 口径、身份迁移与预算在开跑前写死，事后不能根据读数放宽。
- 显式 tendency 的线性冗余被预先承认：负结果或 null 都有清晰解释，不会被迫“调出”一个效应。

**代价（必须记住）**：

- 消耗一轮 GPU 预算（计划约 6–8 GPU-h）；控制是 pinned 旧代码，比较混合模型 revision 与
  run-to-run 方差，不是同代码重训对照。
- +13,056 参数是未受控的容量混杂；零初始化使 update-0 行为与对照相同，因此任何效应只能来自
  训练轨迹，不能读成“输入信息”的因果效应。
- 只有一个 tendency 定义（相邻 6 h 后向差分）与一个尺度；无论结果如何都不能外推到其他定义。
- 三 seed 仅为一致性证据，不是显著性检验；单实例/ROI/季节与 GPU 共驻限制仍然成立。
