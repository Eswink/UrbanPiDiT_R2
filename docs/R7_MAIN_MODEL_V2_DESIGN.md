# R7 主模型 V2 设计契约（P / Z / Y）

- **状态**：设计冻结（供 #72 及后续实现使用）；实验状态见 `docs/plans/0004-r7-main-model-v2.md`
- **冻结时的 HEAD**：`e1d5a7d73aefbc8f1402da484214f14c988e8146`
- **权威性**：本文件是 P/Z/Y 三态、方程、时间语义与信息白名单的**唯一**定义处。
  实现或 issue 文字与本文件冲突时以本文件为准；要改就改本文件（并留痕），
  不要在另一处写第二份定义。
- **不声称**：本文件描述的是**设计**，不是结果。任何「有效/更优」的说法只能来自冻结协议的实测。

## 1. 为什么需要这份文件

#72 是本期主模型创新任务，涉及三个新增概念（P 的读写、Z 的局部状态、Y 的门控提案）、
两条时间轴（内部推理 k / 物理预报 h）和一类新的监督语义（过程诊断的三种时刻）。
缺少单一定义时，不同会话会对「RW-A 的 R 到底是什么」「K 变了要不要重算 Z」产生完全不同
的理解——第二轮与第三轮的实现已经因此各自修正过一次主读者。因此把定义固定在这里。

## 2. 三个状态

| 符号 | 名称 | 形状 | 语义 | 生命周期 |
| --- | --- | --- | --- | --- |
| **C** | context | `[B,N,D]` | 历史 + 已知时空条件编码后的 patch 网格；N=ceil(H/p)·ceil(W/p)（65×65、p=2 → 33×33=1089） | 每个物理转移内**固定**；rollout 的每一步重新计算 |
| **P** | process state | `[B,M,D]`，M=16（8 anchored + 8 free） | 跨区域的**气象过程摘要**；前 `anchored_processes` 个槽位与诊断目标 1:1，其余为非束缚潜变量 | 每个内部步更新一次，跨物理转移重新初始化 |
| **Z** | spatial solver state | `[B,N,D]` | **每 patch 的工作/求解状态**（每个位置保存自己当前的求解进度） | 每个内部步更新一次；**跨物理转移重新初始化** |
| **Y** | forecast draft | `[B,C_out,H,W]` | 当前预报草稿；`Y_0` 是 backbone 的初始预报 | 每个内部步更新一次；rollout 时喂回 history |

**三者都不是自然语言 token。** P 与 Z 是隐式向量计算（latent recursive computation），
不存在「把推理写出来」的中间文本；把 P 叫「过程推理」是语义标签，不是 CoT。

**P 与 Z 的分工**：P 表达「这片区域整体处在什么天气过程里」（跨位置共享的小集合），
Z 表达「这一格当前的求解到了哪一步」（逐位置私有）。P 必须能被不同位置**以不同方式**读取，
Z 必须能**逐位置**保存不同的中间量——这两句就是 #72 的全部内容。

## 3. 更新方程（V2 第一版候选）

```
# 读取（RW-A，已实现）
R_k(i)     = CrossAttention( Q(C_i, E(Y_k)_i, pos_i), K=P_k, V=P_k )      # O(N·M)

# 局部状态更新（RW-B，待实现）
Z_(k+1)    = LocalUpdate( Z_k, C, E(Y_k), R_k, step_embedding )           # O(N)·小 kernel

# 提案与门控混合（RW-B，待实现）
Y_proposal = X_t + Decoder( Z_(k+1) )
Y_(k+1)    = Y_k + g_k * ( Y_proposal − Y_k )        g_k = sigmoid(·) ∈ (0,1)

# 过程状态更新（已实现，V2 只加 role 标记）
P_(k+1)    = Cell( P_k, C ⊕ E(Y_k) ⊕ role )
```

符号表：

- `E(·)`：draft 的 patch 编码器（`DraftTokenEncoder`），输出 `[B,N,D]`。
- `pos_i`：位置 `i` 的**固定**多尺度正弦编码（非可学每位置表，避免引入分辨率上限）。
- `X_t`：当前物理时刻的已知状态，是提案的**锚点**。
- `g_k`：门控，逐位置标量，`[B,N,1]`。
- `⊕`：序列拼接。

### 3.1 三条必须保持的边界（否则不是 RW-B）

1. **提案锚定 `X_t`，不是累加在 `Y_k` 上。** 现行实现（`model/process_forecast_r7.py:272-277`）
   把当前 `draft` 作为 `correction_head` 的 `base_state`，于是 `Y_(k+1) = Y_k + tendency`，
   修正量逐轮累积。V2 的提案是「相对同一 `X_t` 的绝对估计」，由门控决定向它移动多少。
2. **门控不能切断必要梯度。** `g_k` 初始化必须温和（bias 使 sigmoid≈0.1–0.5，而不是饱和到 0）；
   测试必须断言 `Y_proposal` 一侧的梯度非零。`sigmoid` 门控**不是**收敛性/物理正确性保证，
   只是候选的稳定机制——不得在文档里写成「保证单调改进」。
3. **不得重新包装已测过的方案。** `spatial_solver_feedback=True`（把 `draft_tokens` 逐位相加进
   solver context）已在 C2 得到负结果（t2m@48h 三 seed 一致恶化 1.12–1.22 K，28/85 格恶化），
   且它是 0 参数、FLOPs 逐位相同的一次加法。**重开这个开关不是新方法。**

### 3.2 一处已知缺口：拼接的 key 没有「来源角色」

现行 `recurrent_context = torch.cat([context, draft_tokens], dim=1)`
（`model/process_forecast_r7.py:258-260`）把 C 与 E(Y_k) 拼成 `[B,2N,D]` 交给 cell，
**没有任何标记告诉 cell 哪一半是 context、哪一半是 draft**。这正是 #71 第 2 条要求的
「区分 history/context/draft 的 role」，也解释了 #70 的 CPU 探针为什么看到「只反转 draft token
顺序，默认输出 allclose（max 差 1.19e-7）」。

V2 的处置（**待验证的候选，不是既成结论**）：在拼接前给 draft 那一段加一个**可学的 role 向量**
`role_draft ∈ R^D`（context 段加 `role_context` 或保持零）。它 0 参数增量很小、不改变任何现有
开关的默认路径（默认关闭时逐位相同），且可以直接用「反转 draft token 顺序后输出是否变化」
这一个判据检验。**若 E1 显示它无效应，如实记为负结果并保留 role 标记与否的对照。**

## 4. 时间语义：k 与 h 是两个轴

| 轴 | 名称与字段 | 含义 | 是否推进物理时间 |
| --- | --- | --- | --- |
| **k** | `reasoning_steps`（内部推理深度） | 在同一 valid time 上重估草稿：P、Z、Y 各更新一次 | **否** |
| **h** | `rollout_steps` / 物理转移（每个转移固定 +6h） | 把 `Y_(t+6)` 写回 history，求 `Y_(t+12)` | **是，每次 +6h** |

冻结条款：

1. **内部 K 绝不推进时间。** 跨 K 的诊断是**同一 valid time 的重估**，不是「预测未来」。
2. **每个物理转移仍然是 +6h**，不得用 `H=12` 替换单步的 `lead_time_hours` 嵌入。
3. 字段命名必须能让读者分辨两轴；`reasoning_steps` 与 `rollout_steps` **不得混用同一个字段**。
   相关陷阱已有先例：DDP 契约曾把验证深度与更新次数混用（#61）。
4. 已知时空条件的 valid time = `init + 累积 lead`。多步 rollout 中每一步**重新计算**合法时空特征，
   不得让初始日期贯穿 72h。这一条只在 `spacetime_inputs=True` 时成立
   （`model/r7_rollout.py:24-25` 用 `model.spacetime_inputs` 判断），V2 保持不变。
5. **训练不得使用 `no_grad` 的评估 rollout。** M4 的可微展开必须是独立实现或明确声明的截断版本，
   且「内部 K 的 streamed 截断」与「物理步的截断」是两个不同的轴，报告时必须分开。

## 5. 信息白名单（what the model may read）

模型只能看到**预测前已确定**的量。白名单由 `model/r7_halting.py:23-38` 的
`DECLARED_MODEL_INPUTS` / `forecast_inputs` 机械执行：只有列在其中的字段会被转发，
`atmos_target`、`process_targets`、未来的气象 baseline **永远不进前向**。

| 类别 | 字段 | 状态 |
| --- | --- | --- |
| 观测历史 | `coarse_history` | 已有 |
| 预报时效 | `lead_time_hours` | 已有 |
| 地理 | `latitude`、`longitude` | 已有（`spacetime_inputs=True`） |
| 初始时刻相位 | `init_utc_hour`、`init_day_of_year` | 已有 |
| 派生的 valid-time 相位 | init + lead 的年/diurnal 相位 | 已有（由 init 与 lead 决定，无未来观测） |
| 局部太阳时 | — | **尚未实现**；若要加，必须由经度 + UTC 纯推导（`local solar time`），不得读时钟 |
| 来源角色 | — | 见 §3.2 的候选 |

**禁止**：缺失 metadata 时静默填系统时间或伪造地点——必须 `require_*` 抛错（既有做法见
`model/spacetime_conditioning_r7.py:138-167`）。日历/经纬度是确定性上下文，给神经网络这部分
**不是泄漏**（气候态基线本来就按 valid-time 的 month/hour 选桶）。

**泄漏判据（实施时必须做的测试）**：把未来字段（`atmos_target`、`future_diagnostic_targets`、
真实 future diagnostics）放进 batch，**模型的 forward 输出与 halting 选择必须逐位不变**。

## 6. 过程诊断语义：三种时刻，三套名字

现行实现把**同一个输入时刻的 proxy** 喂给每一个内部步
（`training/r7_process_forecast_losses.py:33-50`，按 K 广播）。V2 要求按**时刻来源**把名字拆开，
不得把不同时刻的东西塞进同一个字段：

| 名字 | 来源 | 可用时机 | 角色 |
| --- | --- | --- | --- |
| `input_diagnostics` | 初始时刻的最后历史状态 | 训练与推理**都可得** | 辅助监督（现有语义） |
| `future_diagnostic_targets` | **训练未来的真实标签** | **仅训练** | 监督目标 |
| `draft_diagnostics` | **模型自己的草稿 Y_k**，经固定的 train 反归一化与地理 metric 计算 | 推理时可得 | 自一致性候选特征 |

冻结条款：

1. `D(Y*)` 只由训练未来 label 生成，是**监督**；`D(Y_k)` 由模型自己的草稿算出，是**推理时可得**的
   证据。两者不得共用一个字段名。
2. 跨 K 的 `draft_diagnostics` 是同一 valid time 的重估，**不得错加 lead**。
3. **推理时禁止读取真实 future 诊断或真实 error。** 允许把 `draft_diagnostics` 作为下一轮 read/solver
   的小特征——那是 learned self-consistency，**不是**保证满足完整气象方程。平流符号不得被硬编码成
   温度趋势的因果规则。
4. **尺度修复先于解释**：`eps=1e-6` floor 在 8 个 proxy 上命中 2 个
   （`moisture_advection_850_mean` 原始 std 9.1e-9、`moisture_convergence_850_mean` 1.7e-8；
   归一化后 std 0.0076 / 0.0118）。修复方式是按 train split 物理量纲缩放 + train-only 标准化 +
   **显式 `degenerate` mask**；**禁止**用 1/floor 放大噪声。零方差通道标记为 degenerate 并 mask。
5. 缩放元数据必须**版本化**并与原 std/floor 一起记录；**不原地修改**旧 store 或旧 checkpoint 的
   identity；派生物作为 sidecar 发布，其 identity 纳入新训练 contract。
6. MetPy 只作**离线 oracle**（诊断公式/单位/地理 metric 的一致性对照）；GPU 前向使用可微 torch 张量，
   **不在前向里经 CPU/pint 往返**。

## 7. 复杂度目标

| 路径 | 目标 | 禁止 |
| --- | --- | --- |
| P 的读 | O(N·M)：每个位置一个 query 读全部 M 个 process token | 全局 N×N attention；完整 Perceiver 栈 |
| Z 的更新 | O(N) × 小 kernel（3×3 深度可分离卷积 + pointwise） | 逐像素全对全；RAFT 4D correlation volume；warp |
| 参数增量 | 冻结前实测；目标 ≤ 现有主模型的 **25%**，超出则先缩模块 hidden 再冻结 | 未测量就宣称「轻量」 |
| 显存 | 持续保存 Z 会**增加**每轮工作内存 | 未经测量就称 VRAM 恒定 |

参考量级（HEAD 实测）：process 主模型 2,799,779 参数（pooled）/ 2,968,259（含 RW-A）/ generic 2,799,202；
forward FLOPs 14,162,671,488（含 RW-A）；B3 容量档 18.12M 是 6.47×，属**容量验证**而非默认档。

## 8. checkpoint / 版本化策略

- **模型身份**：`model_code_sha256` 覆盖所有非 legacy 的 `model/**/*.py`
  （`training/r7_experiment.py:31-40`）。V2 的任何字节改动都会改变它 → 提交信息必须带
  `[model-digest-change]`，旧产物只能用其归档 `code.zip` 重放。**不得**修改比对逻辑绕过它。
- **新开关必须默认关闭且逐位等价**：新参数在 `isolated_stream()` 下**最后**构造
  （既有做法见 `model/process_forecast_r7.py:162-165`），使「关」= 前实现逐位相同。
  回归测试 `tests/test_r7_switched_path_equivalence.py` 钉住默认路径。
- **数据身份**：`data_identity` 绑 manifest sha + store attrs + 形状 + 归一化数组。
  M2 段为 `ef8c6691…`；只有 source/case/seed/预算/scale 严格匹配时才能复用旧分数。
- **协议身份**：`protocol.json` 必须在**第一次优化器更新之前**以 `'x'` 排他写入，
  并记录进结果；**已知缺口**——协议 digest **不在 checkpoint 里**，因此 resume 保证只由
  `model_code_sha256` 与 `data_identity` 绑定。V2 的新协议必须显式记录到结果侧。
- **已知 resume 陷阱**：`total_updates` 不在 runner 契约里，而 warmup/cosine 学习率是
  `total_updates` 的函数 → 用较小端点训练的一段**不是**较长端点的前缀。续跑只能从**同一全程端点**
  运行自身的中间 checkpoint 续起。
- **同一步骤三处实现的风险**：`model/process_forecast_r7.py:249-282`、`training/r7_streaming.py`、
  `model/r7_halting.py:123-140` 目前是等价的三份拷贝。V2 的落地要求把它们收敛到**一个共享 step 函数**，
  并由等价性测试断言 fixed / streamed / adaptive 三条路径数值一致（#72 明文要求）。

## 9. 术语表（避免同词异义）

| 词 | 在这里的意思 | 不是 |
| --- | --- | --- |
| positional（位置化） | 每个输出位置形成自己的 query 去读 P | **不是**「有序」；读取仍是集合语义，对 P 的 token 顺序不变 |
| process（过程） | 与气象过程有关的摘要/诊断语义 | 不是自然语言推理步骤 |
| feedback | 把 draft 送进递推 | 不是监督信号 |
| RW-A / RW-B | #72 的两个可分离步骤 | 不是两个模型 |
| matched Generic | 同结构、同已知输入、**无过程语义**的对照臂 | 不是「弱基线」 |
| K | 内部推理深度 `reasoning_steps` | 不是预报步数 |
