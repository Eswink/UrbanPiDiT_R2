# TRM 逐项差异表（#64 D-4）

**上游**：[SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
**revision**：`c01103738605ba39d1430519b1ee0c62f4c707f8`（提交时间 2026-03-31 20:50:39 −0400
= **2026-04-01 UTC**，与审计 §6 记的日期一致；`git clone --depth 1` 实测 HEAD）
**license**：MIT（`LICENSE`，Copyright (c) 2025 Samsung Electronics Co., Ltd.）
**本仓对应物**：`model/recursive_weather_r7.py:74` `GenericRecursiveWeatherForecaster`
（B1 六基线之一；`docs/R7_B1_BASELINE_AUDIT.md:60` 第 6 项）

**本表结论（先说）**：本仓的实现是 **generic recursive baseline**，**不是** TRM 的逐行复现，
也不是 HRM。下面逐项列出**不同的地方**；一致的只有「参数共享的递归改进 + 深监督」这个
**概念层**。任何报告不得把本仓 `generic` 结果写成「TRM 结果」。

**读取方式**：下表读的是**上游 `main` 分支实测源码**，不是论文正文；
行号指向 `c01103738605` 的 `models/recursive_reasoning/trm.py`（297 行）与
`config/arch/trm.yaml`。凡是本段**没有**核对的上游细节，在「未核对」栏里写明，
不用推测填满。

## 1. 架构与递归语义

| 项目 | 上游 TRM（`c01103738605`） | 本仓 `GenericRecursiveWeatherForecaster` | 差异性质 |
| --- | --- | --- | --- |
| **领域 / 输入输出** | ARC-AGI 类**离散 token 网格谜题**：`inputs` 是 token id，输出 `vocab_size` logits；`puzzle_emb_ndim` 为每个谜题学习一个 embedding，`puzzle_emb_len=16` 作为前缀 token（`trm.py:133`、`:170-174`、`:186-187`） | **连续大气场**：输入 `[B,T=2,C=17,H,W]` 归一化物理场，输出同样的连续场（`_Base.inputs`/`out_channels`） | **根本不同**：离散符号推理 vs 连续场回归；损失是 cross-entropy vs 纬度加权 MSE |
| **两套隐状态** | **两个** latent：`z_H` 与 `z_L`，`H_cycles=3`、`L_cycles=6`（`trm.py:186-187`、`config/arch/trm.yaml`） | **一个** latent：`self.latent`（`recursive_weather_r7.py:93`，`[1, latent_tokens, dim]`，默认 16 token），单层递归无 H/L 分层 | **不同**：本仓没有层级结构，等价于上游 `trm_singlez` 那种单 latent 变体的**形状**，但连那个也不是逐行对应 |
| **答案/草稿的载体** | 答案 `y` 是**离散 token 序列的 embedding**，每步由 `lm_head` 重新解码（`trm.py:220`） | draft 是**连续场 `[B,C,H,W]`**，每步经 `draft_encoder`（patch conv）编码回 token 再进 cell（`recursive_weather_r7.py:118`） | **不同**：本仓多一层「场↔token」往返编码 |
| **cell 内容** | `TinyRecursiveReasoningModel_ACTV1Block`（`:65`，`L_layers=2`）：`rms_norm` + RoPE attention + `SwiGLU`，`hidden_size=512`、`num_heads=8`（`:72,84,96-103`） | `GenericRecursiveCell`（`recursive_weather_r7.py:47`）：LayerNorm + `SDPAttention`（自注意力）+ `CrossBlock`（cross-attention 到 context）+ `FeedForward`，`mlp_ratio=3`，可配 `dim/heads` | **结构不同**：本仓 cell 有显式 cross-attention 到编码 context，上游靠把 x 加到 z 上（`trm.py:211,215`） |
| **条件化方式** | 把 `input_embeddings` **直接加到** `z_L` 上（`trm.py:211,215`；`_input_embeddings` `:162`） | `solver_conditioning`（`recursive_weather_r7.py:13`）：把 pooled summary **加回** context token；`spatial_feedback=True` 时再对齐加 draft token | **不同**：本仓把条件化放在 context 侧，上游放在 latent 侧；本仓的 `spatial_feedback` 无上游对应 |
| **答案更新** | `z_H = L_level(z_H, z_L)`（`trm.py:216`），再由 `lm_head(z_H)` 得到输出（`:220`） | `correction_head(conditioned, …)`：解码 patch 为 tendency 并**加到** `initial` 上（`coarse_forecast.py:49`），即 **persistence-relative 残差** | **不同且重要**：本仓的 draft 是**残差改进**（base_state + tendency），上游是重新解码 |
| **每步 forward 次数** | 每步 1 次 `L_level` 组调用（L_cycles 次），另加一次 `z_H` 更新；ACT 目标 Q 计算在训练时**再跑一次** inner（`trm.py:294`，注释说明这是 no-replay-buffer 的 bootstrap） | 每步 1 次 cell + 1 次 draft 编码 + 1 次 correction head（`recursive_weather_r7.py:117-125`） | **不同**：本仓无二次 forward；上游的「第二次 forward」是 ACT 的 Q 目标，**不是**某些二手描述所说的「每步两次前向」 |

## 2. 梯度与深监督语义（#64 明确点名要对照的 x/y/z）

| 项目 | 上游 TRM | 本仓 | 差异性质 |
| --- | --- | --- | --- |
| **哪些步带梯度** | `H_cycles-1` 轮在 `torch.no_grad()` 下跑（`trm.py:207-213`），**只有最后一轮**（1 次 L_cycles + 1 次 z_H 更新，`:214-217`）带梯度 | **所有步都带梯度**：`for step in range(steps)` 里每步 cell/draft 都在图内（`recursive_weather_r7.py:117-126`），除非显式 `detach_between_steps=True` | **不同**：上游是「先无梯度跑够步数、最后一步才学习」，本仓默认整条链反传 |
| **跨步 state 是否 detach** | **是**（硬编码，不可关）：`new_carry = …InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad`（`trm.py:219`） | 默认**不** detach；`detach_between_steps` 是**可选**开关（默认 `False`，`:80,127-128`），开时在 `step<steps-1` 处 detach `z` 与 `draft` | **不同**：上游 truncation 是设计的一部分，本仓是可选项 |
| **中间步是否被监督** | **否**：只有最终输出进 loss（`losses.py:87` 的 `lm_loss` 用当次 `outputs["logits"]`，即最后一次 inner 的结果）；无「每步都对答案算损失」的结构 | **是**（deep supervision 是本仓的核心）：`deep_supervised_forecast_mse` 对 `out.draft_forecasts` 的**每一步**算纬度加权 MSE，权重 `linspace(1.0, final_weight=2.0, steps)` 归一化（`training/r7_recursive_losses.py:7-27`），被 `R7RecursiveLightningModule._step` 用作训练损失（`training/r7_recursive_lit_module.py:25-31`） | **本仓比上游强**：上游没有对各步 draft 的深监督；本仓的 `final=latitude_weighted_mse(out.forecast,…)` 还会**另外**再算一次最终损失 |
| **halt / ACT** | 有：`q_head`（`CastedLinear(hidden,2)`，`trm.py:131`）预测 halt/continue，Q-learning 目标 + exploration（`halt_max_steps=16`、`halt_exploration_prob=0.1`、`no_ACT_continue=True`，`trm.py:267-295`、`losses.py:87-102`） | **模型本身没有 halt 头**。自适应的停步在**另一个模块**：`training/r7_halting.py` 的 gain controller（`controller_calibration_loss`/`calibrate_controller_step`），对**冻结主干**做后训练校准，不经 `GenericRecursiveWeatherForecaster` 的前向 | **不同**：上游 halt 是模型内生、随主损失联合训练；本仓是外挂控制器、分离校准 |
| **推理深度怎么给** | `halt_max_steps` 上限内由 ACT 决定；**推理时注释明说总是跑满 max steps**（`trm.py:278`） | 由调用方给 `reasoning_steps`（模型默认 `default_reasoning_steps=4`，B1 用 K=3）；`model/r7_rollout.py:30` 记录 `cumulative_reasoning_steps` 并注明该量 "NOT latency/FLOPs" | **不同**：本仓深度是显式超参，无推理期学习式停机 |

## 3. 本仓**有**而上游没有的东西（避免把差异读成缺失）

- **残差式（persistence-relative）预报**：`correction_head` 返回 `base_state + tendency`
  （`coarse_forecast.py:49`），且残差头零初始化（`coarse_forecast.py:37-38`），因此未训练模型 ≈ persistence
  （`docs/R7_B0_LEARNABILITY.md:32`）。上游直接解码答案，无此结构。
- **经纬度加权误差与物理单位**：损失与指标用 `latitude_weighted_mse` / `RolloutRMSEAccumulator`
  的 cos(lat) 面积权重；上游是逐 token 的 cross-entropy。
- **跑满步数的 roll-out 语义**：本仓一个 lead 一次前向（`autoregressive_rollout`），
  递归 K 步是「同一时次的深度」而非时间推进——与上游「改进同一个答案」概念相近，
  但本仓另有 lead/lag 语义（`lead_time_hours` embedding）。
- **可选 spatial solver feedback**（`spatial_solver_feedback`）：无上游对应。

## 4. 命名与结论纪律（沿用审计既有判定，本段复核）

- `model/recursive_weather_r7.py:75` docstring：*"Generic parameter-shared recursive
  baseline without process semantics."* —— **不冒称 TRM**。
- `docs/R7_BUDGET_PARITY.md:3-5` 称其为 *"generic TRM-like recursive"* —— 本表正是
  支撑这句「-like」的逐项依据（**概念相近、实现不同**）。
- 因此报告口径固定为：**generic parameter-shared recursive baseline (TRM-inspired,
  not a TRM reproduction)**。B1 表里的标签用 `generic`，不用 `TRM`。

## 5. 未核对 / 不主张

- **未读论文正文**：上表全部来自 `c01103738605` 的**源码与配置**。论文对
  「2 forward passes per recursion」「no gradient through steps」的**措辞**未核对，
  故本表**不引用**这些二手说法（审计 §6 提示过这类转述可能与代码不一致）。
- **未跑上游代码**：本段只做静态阅读（零 GPU），没有安装其依赖或复现其任何数字。
- **未核对上游其它变体**：`hrm.py`、`trm_hier6.py`、`trm_singlez.py`、`transformers_baseline.py`
  只确认存在与大致形态，未逐项对照。本仓 `generic` 与它们**任一个**都不是逐行对应。
- **未做代码移植**：本表是**差异记录**，不是移植计划。上游 MIT 允许借用，
  但 #64 要求的是「先做差异表」，本段停在差异表；任何借用须先记 revision/license/
  文件与改动表（`docs/R7_B1_BASELINE_AUDIT.md` §6 已立此规）。
