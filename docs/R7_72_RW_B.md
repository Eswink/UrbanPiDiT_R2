# RW-B：局部门控求解状态、锚定提案与来源角色（#72 M2-B）

**状态：真实模型代码与定向测试已完成；本文件不声称任何预报收益。**
本轮**未执行**有界 train/val 对照（D5）——它按决策 0021 需要用户在执行那一刻的授权，
本轮结束前未取得（见 §7）。因此本文件里没有任何 RMSE、任何 skill、任何"更好/更差"。

> **更新（2026-09-29，后续一轮）**：D5 **已执行**。用户在 2026-09-29 按决策 0021 明确授权后，
> 跑完 2 seed × 4 臂 × 400 updates（实测 0.8356 GPU-h），判决与全部读数记在
> **`docs/R7_72_RW_B_PILOT.md`**——那里是本轮结果唯一的落点。一句话：登记主端点
> `RW-B − RW-A` **未获支持**（t2m 12/24h 改善 −0.08 K，但 6/48/72h 恶化 +0.12/+1.07/+1.58 K，
> 模态读数为 worsened）。本文件其余内容**按写作时的状态保留**（代码与测试仍成立），
> §7/§8 的「未做」与「下一项」以该更新块为准。

| 项 | 值 |
| --- | --- |
| 起点 SHA | `e1a915cbc996d9a553be0f717fd24e871a61fc90`（开工 `git rev-parse HEAD`） |
| 设计权威 | `docs/R7_MAIN_MODEL_V2_DESIGN.md` §3/§3.1/§3.2/§8（本文件不另写第二份定义） |
| 新增模块 | `model/local_solver_state_r7.py`（Z 的更新单元、逐位置门控、锚定提案、门控展开）、`model/process_step_r7.py`（唯一一份 step） |
| 接线 | `model/process_forecast_r7.py`、`model/recursive_weather_r7.py`、`model/r7_halting.py`、`training/r7_streaming.py`、`training/r7_halting.py`、`training/r7_gain_oracle.py` |
| 新开关 | `source_role_markers`、`local_solver_state`，**均默认 `False`** |
| 测试 | `tests/test_r7_local_solver_state.py`（27 项）、`tests/test_r7_shared_step_paths.py`（10 项） |
| 实跑 | `pytest -q` → **1456 passed, 3 skipped**（全仓；新增 37 项全部通过） |
| 门禁 | `check_conventions.py` 37 条阻断 **0 违规**；`git show --check` 干净 |
| 实测代价 | 参数 **+314,898 / +10.61%**；前向 FLOPs **×1.2097**（见 §4） |
| `scientific_claim` | `false`（本文件与两份运行产物内均带该字段） |

---

## 1. 本轮做了什么（实现，不是结果）

`docs/R7_MAIN_MODEL_V2_DESIGN.md` §3 冻结的四件事，逐条落地：

1. **Z（每 patch 的工作状态）**：`model/local_solver_state_r7.py` 的 `LocalSolverState`。
   每步读 `C`、`E(Y_k)`、`R_k` 与固定正弦 step embedding，经 pointwise 投影 →
   **3×3 深度可分离卷积**（唯一一处空间混合，O(N)）→ pointwise 门投影 → ConvGRU 式更新。
   没有 N×N attention，没有 4D correlation volume。
2. **锚定 X_t 的提案**：`Y_proposal = X_t + Decoder(Z)`，由 `proposal_head`（`CoarseForecastHead`）
   实现。`X_t` 由 `R7ForecastOutput.base_state` 提供——**在 backbone 里决定、随输出返回**，
   而不是在各调用点各自重算"最后一步 history"，因为第二份定义正是三份实现漂移的起点。
3. **逐位置门控**：`PositionGate` 产生 `[B,N,1]`，`expand_token_gate` 按 token 自己的
   `patch_size × patch_size` 块最近邻复制再套用 decoder 的裁切（与 `CoarseForecastHead` 的
   转置卷积是同一套 patch 布局），最后 `Y_(k+1) = Y_k + g_k·(Y_proposal − Y_k)`。
   门控 bias 初值使 sigmoid ≈ **0.25**，不饱和。
4. **来源角色**：`recurrent_key()` 是 key 装配的唯一位置；开启时给 `[C; E(Y_k)]` 的两半
   各加一个可学向量（`[1,1,D]`，共 2·D 个参数）。关闭时返回的就是原来的
   `torch.cat([context, draft_tokens], dim=1)`。

**一处唯一实现**（#72 明文要求）：三份镜像的递推步收敛为
`model/process_step_r7.py::process_reasoning_step`，由 fixed `forward`、`training/r7_streaming.py`
与 `model/r7_halting.py` 三处调用。`tests/test_r7_shared_step_paths.py` 里有一条**结构性反证**：
`model/r7_halting.py` 与 `training/r7_streaming.py` 中不得再出现 `_process_prediction` /
`process_conditioning`，否则测试失败——防止今后有人把公式抄回去。

## 2. 「关 = 逐位相同」的证据（D2）

两层，都是实跑：

1. **同进程逐位钉住**（`tests/test_r7_switched_path_equivalence.py`，**既有测试，未改动**）：
   23 个 digest 覆盖三族 forward / rollout / 自适应 / streamed BPTT 的**全参数梯度**，
   对照组是 `git archive 93d89aa` 的冻结实现（仓库既有做法：同进程双包名导入、
   import 改写可逆校验）。本轮所有 `model/**` 改动之后仍**通过**。
2. **新增开关的"关"**（`tests/test_r7_local_solver_state.py`）：把 `source_role_markers=False` /
   `local_solver_state=False` **显式写出来**构造的模型，与完全不写这两个参数构造的模型
   `state_dict` 键集合相同、逐张量 `torch.equal`、参数总量相同、前向 `torch.equal`。
   即"关"不是近似，而是同一个模型。

**第三个独立佐证（本轮实测，非计划）**：`scripts/measure_r7_rw_b_cost.py` 在**审计配置**
（17 通道 / 65×65 / dim=192 / batch=2）上测得的 RW-A 前向 FLOPs 是
`13,904,603,520`、前向+反向 `41,596,684,032`，与二轮 `arm_table.csv` 里 C 臂归档的
**同一个数字**逐位相同。这独立说明共享 step 重构没有改变既有路径的**计算量**。

## 3. 边界与它们的反证（D3/D4）

设计契约 §3.1 的三条边界，每条都有断言，且每条断言都带一个"证明它不是空断言"的反证：

| 边界 | 断言 | 反证 |
| --- | --- | --- |
| 门控不得切断提案梯度 | 全部 RW-B 参数的梯度非零且有限 | 把门控 bias 饱和到 −50 后，`proposal_head` 的最大梯度跌到原来的 1e-6 以下——说明"非零"是被量出来的，不是构造上必然 |
| 提案锚定 `X_t`，不是累加在 `Y_k` 上 | 只平移 `X_t`，提案随之平移；tendency 不变 | 关闭 `local_solver_state` 时**不需要** anchor；开启后不传 anchor 直接 `ValueError`，不静默退化成累加 |
| 不得把 `spatial_solver_feedback` 重新包装成新方法 | 四种 RW-B 配置下该开关全为 `False` | C2 的负结果原文记在测试 docstring 里 |

其余定向测试：K=0/1/2/4 前向（K=0 返回初始预报且不产生 Z）、参数总量不随 K 变化、
odd/non-divisible 网格（5×7、7×5、1×3、3×1）前向+反向、`expand_token_gate` 与手工构造对齐、
门控落在开区间 (0,1) 且不同位置可以不同、门控初始均值在 0.25 附近、step encoding 固定且区分深度、
**poison**（把 `atmos_target` / `future_diagnostic_targets` / `process_targets` / `atmos_baseline`
换成极端值后 forward **逐位不变**）、**halting 选择**在 poison 下逐位不变（`active_masks`、
`decision_masks`、`reasoning_steps_per_sample`、`predicted_gains`、`continue_probabilities`）、
checkpoint 往返（经 `save_exclusive`/`load_checkpoint` 真实路径，RW-B 张量全在且前向逐位相同）、
BF16 streamed backward 梯度有限、三路（fixed/streamed/adaptive）在开关开与关时**都**逐位一致、
自适应 active-subset 在"无人停止"时等于固定深度、来源角色的判据（关时对 draft token 重排
`allclose`，开时**不再** allclose）与"角色向量确实进了前向"。

## 4. 代价（实测，非估计）

`scripts/measure_r7_rw_b_cost.py`，审计配置（17 通道 / 65×65 / patch 2 / dim 192 /
`reasoning_steps=3` / batch 2），CPU，FLOPs 约定与二轮声明的一致（`FlopCounterMode` +
`enable_grad`，反向单独测不按 2× 假设，不用参数 hook）。产物：
`outputs/r7_e0_diagnostic/rw_b_cost_batch2.json`。

| 臂 | 参数 | Δ vs RW-A | 前向 FLOPs | 前向+反向 FLOPs |
| --- | --- | --- | --- | --- |
| RW-A（对照） | 2,968,259 | — | 13,904,603,520 | 41,596,684,032 |
| RW-A + 角色标记 | 2,968,643 | **+384**（+0.01%） | 13,904,603,520 | 41,596,684,032 |
| RW-A + RW-B | 3,283,157 | **+314,898**（**+10.61%**） | 16,820,126,592（**×1.2097**） | 54,656,320,512（**×1.3140**） |
| RW-A + 两者 | 3,283,541 | +315,282（+10.62%） | 同上 | 同上 |

- **参数目标 ≤25% 达成**（10.61%），无需缩 hidden，配置可按 §7 冻结。
- 角色标记按本约定**不改变 FLOPs**：它只是逐 token 加法，而本约定不计 elementwise；
  这不等于它零成本，只是"零计数"。
- **未测显存**：设计契约 §7 禁止未经测量就称 VRAM 恒定，本节因此**不做**任何显存陈述。
  持续保存 Z 会增加每轮工作内存这一点在 §6 的 limitations 里明写。

## 5. 与 E0 的关系（为什么门控作用在幅度上）

E0（`docs/R7_E0_DIAGNOSTICS.md`，E-196）的读数：加一条 process 读取通路会在三个 seed 上
**一致放大** solver 的修正幅度（C/B 池化比 1.31），但**不会**让修正与误差更反相关
（`cos(e,d)` 一致升高，方向与"读取把 context 拉偏"的预测相反）。

门控因此被放在**幅度**这一侧：它让模型逐位置地**推辞**一个大提案，而不是去纠正一个方向。
这是设计选择，不是"门控已被证明有用"——后者只能来自 §7 未执行的那个对照。

## 6. limitations

- 本文件**没有任何预报指标**：没有 RMSE、没有 skill、没有 ACC，也没有"效果"陈述。
- 门控是候选的稳定机制，**不是**收敛性、单调改进或物理正确性的保证。
- 未测显存；未测墙钟；FLOPs 是 CPU 上的算子计数，不是 GPU 时间。
- 只在**一个**配置（dim 192 / 17 通道 / 65×65）上测了参数与 FLOPs。
- 三个 seed / 一次运行**一个都没有**：本轮没有跑训练，因此没有一致性可言。
- 来源角色的判据是"重排 draft token 后输出是否变化"，它证明角色**可达**，
  **不证明**角色有用；要判有用必须靠一个真实对照。
- `local_solver_state` 被构造性地绑定在 `positional_process_readout=True` 与
  `use_forecast_feedback=True` 上：换成池化读取或关掉反馈，测到的就不是设计契约冻结的那条方程，
  因此代码**拒绝**这两种组合而不是静默替换。
- generic 模型只接了共享 key 装配与来源角色；**matched Generic 的完整 RW-B 未实现**
  （理由见 §7），因此决策 0023 要求的"匹配 Generic 对照"这一前置**仍未被满足**。

## 7. 未做的事

- **D5（有界真实 train/val 对照）未执行**（**已由后续一轮补齐**，见文首更新块与
  `docs/R7_72_RW_B_PILOT.md`）：按决策 0021 它需要在执行那一刻由用户授权
  （范围 / 预算 / 产物与证据 / 失败与 skip 处理）。本轮结束时未取得该授权，
  因此没有协议、没有 run、没有 GPU 秒数。E0–E5 的 E1 全部未开始。
- 未训练、未租 GPU、未下载数据、未读 `test.jsonl`、未写 `data/raw|interim|processed`、
  未改动任何已归档 checkpoint 或既有 `outputs/r7_71_72_*` 产物。
- 未给 `GenericRecursiveWeatherForecaster` 接 RW-B：设计契约把 `R_k`（过程读）定义为
  `LocalUpdate` 的输入，而 generic 没有过程状态，它的 `R_k` 会是一个**新的定义**——
  本轮不发明新机制，因此如实记为未做（M5 的前置）。
- 未做 M3/M4/M5；未改任何既有开关的默认值；未新增任何阈值或放宽任何已冻结判据。
- **本轮未修订三轮 §13 的那句错话**（见 `docs/R7_E0_DIAGNOSTICS.md` §5）：已冻结的文档是证据，
  更正只记在 E0 记录与 E-197，不回溯改写。

## 8. 下一项的第一个具体动作

按决策 0021 取得 D5 的授权后，用二轮/三轮同一协议族跑三臂 × 2 seed × 400 updates
（旧 mean-Ours / RW-A / RW-B；**每次只改一个结构**，即 B 与 C 只差 `local_solver_state`，
`source_role_markers` 单独成臂或先关）、协议在第一次优化器更新前以 `'x'` 排他写入
`protocol.json` 并记录 digest，主端点跑前冻结，只读 val、test 封存，实测 GPU-h 写进结果。
**先读 `docs/R7_E0_DIAGNOSTICS.md` §5**：t2m 72h 那一格在两轮里都不稳定，
因此**不得**把 72h 的单格差异当作机制证据，48h 那一格才是可复核的那一个。
