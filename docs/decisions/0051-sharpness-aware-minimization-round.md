# 0051 Sharpness-Aware Minimization 单因素轮的授权边界（只改训练期更新规则）

- **日期**：2026-10-21
- **状态**：accepted
- **依据证据**：`docs/goals/main-model-climatology-campaign.md` §8 的
  2026-10-21「重新激活：SAM 作为独立优化因子」块（本轮开始时尚未提交）；
  `docs/decisions/0030-autonomous-execution-and-direction-boundary.md`（方向内实验设计与优化方法下放）；
  `docs/decisions/0048-capacity-axis-scope-and-cap.md`（容量轴与 H-A…H-D 已关闭）；
  `docs/decisions/0049-explicit-tendency-input-round.md`（上一表示因子轮，支已按预声明关闭）；
  `docs/goals/s3-tendency-input.md`（上一轮冻结的 (a)–(e) 出口，本轮逐字沿用）；
  `docs/R7_S3_TENDENCY_INPUT_RESULT.md`（上一轮 mixed/not-candidate）；
  `training/r7_long_rollout_runner.py:_update`（唯一改动点，off 分支已独立）；
  `training/r7_gradient_surgery.py`、`training/r7_ema.py`（加性开关「默认关闭即不写任何字段」的既有先例）；
  Foret et al., arXiv:2010.01412v3 §3.1（SAM 的 p=2 对偶解与默认 ρ=0.05）；
  SAMformer, arXiv:2402.10198v3 Table 4（最接近的时序先例与 rho* 区间）

## Context

2026-10-20 的「暂停单因素训练/优化搜索」块被 2026-10-21 的 tendency 重新激活块部分取代，
该表示因子轮已按预声明出口（72h skill 三 seed 全负、72h 配对 Δ 非全负）**永久关闭**。§8 的
「下一动作」要求提出一个与已关闭轴不重叠、可独立检验的新因子。上一轮结束后，本方向仍未检验
过的单因素只剩**优化器更新步本身**：不是反向的分组（gradient surgery，已关）、不是输入通路
（tendency 输入，已关）、不是容量/深度/宽度（决策 0048 已关）、不是损失/监督组织（多支已关）。

SAM 正好落在这个缺口里：它不增加参数、不增加输入、不改损失定义，只把注册的单步
`result.loss.backward()` 换成预声明的双 forward/backward 扰动更新。它也因此**不是**任何已关轴的
重命名——没有一条已关轴是「更新点」。

## Decision

1. **授权一轮**：允许在 `docs/goals/s3-sharpness-aware-minimization.md` 冻结并执行候选单臂 ×
   seed 41/42/43，对照为 pinned `s3-anchor-isolation-result`（注册更新规则、final optimizer
   weights、不重训），外部参照为 pinned `s3-wide-interior-dose`。
2. **单一因素**：`sam_rho` 从 `None` 改为预声明的 **0.05**。规则写死为：`e = rho·g/(‖g‖+eps)`，
   `eps = 1e-12`，`‖g‖` 是对全部可训练参数的**一次全局**范数（float64 累加）；`w → w + e`；
   在扰动点重算**同一注册损失**并取其梯度作为更新梯度；用保存张量的 `copy_` 把参数逐位写回
   （不是 `w − e`）；之后照走注册 `clip_grad_norm_(clip=1)` 与 `optimizer.step()`。
   **禁止**扫 rho、改扰动范数/eps/符号、加 parameter-matched 臂或与任何已关轴组合。
3. **模型与父不变**：候选 model spec 必须与 pinned 控制 spec **逐字相等**（不是「加一个开关」），
   本轮不改 `model/*.py`，`model_code_sha256` 保持 `60de3b2d…`，因此复用既有 committed 父迁移
   （receipt `73c019dd…`）；控制 spec 下的扩张只新增两个已发布 anchor buffer，逐 seed expanded
   digest 必须重算并 pin，**不得**沿用上一轮含零 tendency patch 的值。
4. **预声明出口**：沿用 brief §3 的 (a)–(e)，不新增阈值。全真→`candidate_state: needs-review`，
   只提请独立复核，不自动进 S4；任一 (a)–(d) 假→**永久关闭 SAM 支**（不扫 rho、不改扰动定义/
   范数/eps/符号、不加 seed、不改 updates/LR/warmup/K/depth/dim/clip、不做 parameter-matched
   臂、不与已关轴组合、不换名重开）；(e) 假只记 `identity_unresolved=true`，是工程身份失败而非
   科学判定。
5. **null 口径**：只有 SAM 支**每个 update** 的扰动恰为零（梯度恒为 0，更新与注册路径逐位相同）
   或每个配对 Δ 在记录精度下恰好 0.0 才记 null/inert。小而非零效应按 (a)–(d) 判读，不得改标 null。
6. **算力混杂必须分报**：SAM 每 update 两次 forward/backward，FLOPs/GPU-h 约 2×。必须分报参数/
   FLOPs/GPU-h 并声明算力混杂；一手依据 arXiv:2010.01412v3 §3.1 该文自己「给非 SAM 基线两倍
   epoch」来对齐算力，故**同 updates 比较在其自身框架下对基线有利**。本轮**不**跑 4800-update
   算力匹配控制——那会是第二臂与新 recipe，超出本记录授权。
7. **负结果的范围必须写清**：rho=0.05 取自 arXiv:2010.01412v3 §3.1 的默认值；最接近的时序先例
   SAMformer（arXiv:2402.10198v3 Table 4）最优 rho*=0.1–1.0 且用 Adam 而非 AdamW（本 recipe 用
   AdamW + weight_decay 1e-4）。因此负结果只关闭「预声明 rho 下的 SAM」，**不证明** SAM 在其他
   rho 无效；不得据此外推。
8. **前置与预算**：本记录与 §8 块必须在 `freeze` 前已提交；`test.jsonl` 未读、r=0、S4 未启动。
   `per_seed_seconds=21600`、`planned_seconds=43200`（软）、`hard_cap_seconds=86400`（硬），逐字
   沿用上一轮单臂 envelope；per-seed 或 hard cap 截断只算 attempt 失败，不改因子或判据、不加时。
9. **不改变既有封闭**：本记录不重开决策 0048 的容量轴或任何架构/表示/训练信号支，也不授权数据
   扩充（D3 暂缓仍然有效）。

## Consequences

**正面**：

- 该优化因子获得明确、可引用的授权边界；它是「更新点」这一从未检验过的轴，与全部已关轴不重叠。
- 出口、null 口径、算力混杂、负结果范围与预算在开跑前写死，事后不能根据读数放宽。
- 参数个数不变，没有上一轮那样的容量混杂；唯一的代码差异可以用「off 分支是独立逐位注册路径」
  的 rho=0 反证测试与 update-0 未扰动损失逐位相等来界定。

**代价（必须记住）**：

- 消耗一轮 GPU 预算（计划约 15–16 GPU-h，约为上一轮的 2×）；单 seed 时长逼近 21600 s 的
  per-seed 硬界，可能以工程 attempt 失败收场。
- 控制是 pinned 旧代码读数，比较混合 training-code revision、model-code revision 与 run-to-run
  方差，不是同代码重训对照。
- 同 updates 非同算力：本轮的比较对基线有利，但也不能反过来读成「SAM 在同算力下更好/更差」。
- 只检验一个预声明 rho；负结果不外推到其他 rho、其他优化器或其他数据制度。
- 三 seed 仅为一致性证据，不是显著性检验；单实例/ROI/季节与 GPU 共驻限制仍然成立。
