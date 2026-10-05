# R7 S2 / #79：有类型的局部诊断证据（三臂设计，执行前冻结）

<!-- round-node: S2 -->

- 撰写日期：2026-10-05；状态：**design-frozen, not yet executed**；`scientific_claim: false`。
- 本文是 `docs/goals/s2-climatology-mechanism-screening.md` §2 D5 的交付物，也是 #79 的开跑前置。
- **性质**：设计文档。在此冻结机制、三臂、判据与不变量；实际执行必须另写 `protocol.json`
  （在第一步之前），本条不产生任何读数。

## 1. 要检验的假说（可证伪）

**现状（已核实于当前工作树）**：`training/r7_process_tensor_diagnostics.py` 的 8 维诊断
（`compute_process_tensor_diagnostics`）目前只服务辅助 loss 与控制器；forecast-facing 路径
（`process_step_r7.process_reasoning_step` 的 C、E(Y_k)、P、可选 Z）**不含任何物理诊断字段**。
C 当前 aux=0，诊断预测头对预报没有贡献。

**假说**：把**有类型**的局部诊断证据（div850、vort850、t850 平流、静力稳定度）从 X_t 与当前
草稿 Y_k 就地提取并路由进对应 typed process slots，能让预报路径吃到过程归纳偏置；该偏置应表现
为 **C 臂优于同容量、同诊断信息的通用融合 B 臂**（B→C 才归因 typed 结构；A→C 只是整体增益）。

**反证条件**：C 相对 B 的注册主格（t2m 6h/12h）三 seed 不同号为负/多数恶化，或 C 的
typed 路由断言测试无法证明"每类证据确实到达其槽位"，即该机制在本预算/实例被证伪并撤回候选。

## 2. 机制（最小新增，复用已有算子）

1. **局部场**：对 `(X_t, Y_k)` 反归一化到物理单位后，用**已有**可微算子
   `spherical_divergence`、`spherical_vorticity`、`horizontal_advection`、
   标量差（`t500` 与 `t850` 的位温差）得到 4 个 `[B,65,65]` 局部场。
   复用 `training/r7_process_tensor_diagnostics.py` 的 FP32/非周期边界/mask 实现，
   **不新写第二套微分**、不在每步转 CPU/Pint。
2. **局部证据 tokens**：4 个场按与 C/E(Y_k) 相同的 patch 网格编码（共享 patch 投影），
   经每类独立的 `nn.Linear(dim→dim)` 类型投影，得到 4 组 typed 证据 tokens。
   **类型投影逐类独立且参数计量**；如某场在 train 上方差低于 float32 相对下限（复用
   `r7_process_scale_sidecar` 的退化规则），该类型被 mask 掉并在协议中记录，**不伪造分量**。
3. **路由**：typed 证据以加和进入对应 anchored process slot 的初始状态（`p_i ← p_i + proj_i(D_i)`），
   随后完全走现有 `_reason` / `process_conditioning` / `SOLVER` 读写链；free slots 不变。
   16 全局 slots 数量不变；新增参数 = 4 个类型投影 + 1 个共享 patch 投影（全部在协议里给出参数量）。
4. **不变量**：`Y_(k+1) = Y_k + LearnedCorrection(...)` 仍是学习修正；诊断只作特征/归纳偏置，
   不硬编码符号规则、不作严格真值；初始/草稿/目标物理时刻字段继续分开；内部 K 始终同一 future
   valid_time（复用 `_batch_times` 的既有契约）。

## 3. 三臂（同容量、同信息、同更新）

| 臂 | 结构 | 归因 |
| --- | --- | --- |
| A `process_v2` | 当前 C（incumbent 配置，aux=0），不含诊断输入 | 基线 |
| B `generic_fusion` | 同 4 个诊断场、同参数量：单一共享投影消费全部场再融合（**无类型约束**） | B→C 的 typed 归因控制 |
| C `typed_routing` | 每类独立投影 → 对应 anchored slots | 候选 |

B 与 C 必须参数/FLOPs 对齐（B 把 4 份投影合并为一个 4×dim 输入的单投影；以测量为准）。
A→C 只看整体；**只有 B→C 支持才允许声称 typed 结构本身有用**。三臂同 seed 初始化共享
（沿用 `anchor_transfer_rule` 与 `verify_arm_pairing`），数据/更新/预算完全一致。

## 4. 首轮设置（冻结）

- **aux=0**：只验证前向结构是否有贡献；future/draft 诊断辅助 loss、PCGrad 等一律不混入首轮。
- 数据：S1 dev store（train 340 / val 34 / test 22），val-only；test 全程封存。
- seeds 41/42/43 预声明；400 更新；K4；其余超参继承 #77/#78 同一套（LR 2e-4、warmup 80、
  batch 2、clip 1.0、验证每 100、patience 4）。诊断归一化尺度与反归一化用 **train-only**
  统计（store 的 `normalization_*` 与 `process_normalization_*`），跨 split 不泄漏。
- 注册主格：t2m，6h/12h，`C − B` 与 `C − A` 两条配对，各自按每 seed 同号规则读；
  C 必须同时对 B 与 A 非劣（supported 要求两条配对都 supported 或 C−A 不劣且 C−B supported）。
- 全 17 变量 × 5 lead 如实报告，不并入判定。

## 5. 验收清单（执行时必须逐条留下可核验证据）

1. **无未来信息**：不含任何 future target 的 batch 即可前向；毒化 future 标签不影响
   forecast/K 选择；改变合法 initial/draft 诊断会经指定路径改变 forecast（反证测试）。
2. **算子与单位**：解析矢量场验证微分/球面单位；常数场、零风、mask、odd grid、BF16 稳定。
3. **路由可核验**：typed slots 确实收到各自证据（逐类探针：改变 D_i 只移动 slot i）；
   关闭路径与原 V2 位级一致；参数/FLOPs 增量与梯度流记录在案。
4. **同一中央 step**：fixed / streamed / active-subset 三条路径贯通同一实现。
5. **身份**：新 buffer/投影参数进 `model_code_sha256`；诊断归一化 sidecar 进数据身份；
   不改旧 checkpoint 校验逻辑。
6. **报告**：全变量全 lead；坏结果保留；attention 可视化仅描述，不作 B/C 归因。

## 6. 成本与预算（预计）

- 三臂 × 3 seed × 400 更新，按 #77/#78 实测 ~200 s/臂 → 约 **0.5 GPU-h**（训练）+ 评估；
  软预算 2.0 GPU-h 内，单轮 hard cap 6.0 GPU-h（承接 S2 长文 §5）。
- 新增参数量预计 < 5% 主干（4×dim² + patch 投影），必须以 `measure_arms` 实测记录，不得估报。

## 7. 与旧尝试的差异（信息增益声明）

- 与 #72/#73 不同：不重做 P/Z/Y 或 future/draft 辅助 loss；本轮把诊断**作为前向输入**，
  且首轮 aux=0，机制与旧负结果（无 aux 时 forecast 梯度等价）**不重复**。
- 与"只加一个 aux 头"不同：typed 路由有逐类到达探针与 B 臂同容量控制；若 C≈B，
  直接得"类型约束无增益"的干净负结论，不再堆诊断种类。
- 回主线动作：无论正负，均把结论登记进 S2 证据页与索引；负则撤回候选、保留 V2 性能路线；
  正（B→C supported）才申请把它带进未见年份确认实例的设计冻结。

## 8. 外部来源（实现前锁定）

- MetPy kinematics 仅作公式/单位 oracle，计算用本库已有 tensor 实现
  （复用对象为 `training/r7_process_tensor_diagnostics.py`，其单位/边界契约已有测试）；
- PhyDNet/Perceiver 仅参考分解与查询思路（issue #79 已登记链接）；实施前如要移植任何源码，
  按 R-050 锁 commit/许可证并留引用账本。**本轮设计不引入任何外部依赖。**
