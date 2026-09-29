# 0004 R7 Main Model V2：Process–Forecast Co-Reasoner 的第二阶段

Status: PLANNED
Base branch: r7/weather-reasoning
Base SHA: e1d5a7d73aefbc8f1402da484214f14c988e8146
Parent: #70
Children: #71 #72 #73 #74 #75
Companion documents: docs/R7_MAIN_MODEL_V2_DESIGN.md、docs/R7_MAIN_MODEL_V2_REFERENCES.md、
docs/goals/main-model-v2-rw-b-round.md、docs/decisions/0023-main-model-first-baseline-freeze.md

> 本计划的 Base SHA 是**只读核对**时的 HEAD（git ref + GitHub 匿名 API，2026-09-29）。
> 计划归档在实现之前，因此不存在属于它的实验产物；`## 实际结果（Actual Results）` 一节
> 因此写 NOT EXECUTED YET，并如实记录本次 Plan Mode 自身做过的事。

## Objective

把主模型（Process–Forecast Co-Reasoner）从「已实现已知时空输入 + 位置化 process 读取」推进到
**空间局部求解状态（RW-B / M2-B）**，并在此之后按序推进与预报相关的过程监督（M3）、
真正的两步可微自回归微调（M4）、最小确认与有条件自适应（M5）。

本轮**不是** baseline 扩容轮，也不是重新规划轮：开发期基线冻结（决策 0023），
主模型是关键路径。

## Why main-model-first now

- 全部 17 条实验 workflow 都是 CPU-only，双 3090 本机可用；继续在 CPU 上比基线的边际信息
  已经很低：`docs/R7_ROADMAP_GATE_STATUS.md` 记录 G1–G4「没有一条被正面支持」，
  B1/B2/B3、M2、S3–S5 都已经是负结果或混合结果并把局限写清了。
- 负结果已经指出瓶颈**不在**基线强度，而在主模型自身的读写结构：
  - C1（`docs/R7_65_C1_PROCESS_SUPERVISION.md`）：aux=0.1 在三个 seed 上全部**恶化** t2m
    训练时效（+0.05…+0.35 K），12/12 无早停，1.08 GPU-h。
  - C2（`docs/R7_65_C2_C3_FEEDBACK_AND_DEPTH.md`）：打开 `solver_on` 恶化 28/85 格；
    t2m@48h 在三个 seed 上一致恶化 1.12–1.22 K。关掉 reasoner feedback 只是弱正向
    （14 改善/8 恶化/63 未决），且它同时省 10.7% FLOPs——是**效率**陈述，不是结构陈述。
  - C3：深度自由缩放不成立；独立训练 K=1 省 27.9% FLOPs，18 改善/14 恶化/53 未决。
  - 预诊断（`docs/R7_65_PREDIAGNOSTIC.md`）：`process_readout` 在 `process_weight=0`
    时**从未被训练过**（‖g‖=0，梯度到不了它）；B2/B3 的 process 臂与 generic 臂的差别
    只剩最后一个投影层，对齐后逐位相同（`torch.equal`）。
- 因此「再加一个 baseline family」不能回答任何当前问题；能回答问题的只有主模型的
  P→空间读写路径本身。

## Current evidence

三块必须分开引用、**不得相加或并置**（各自 protocol digest、model code digest 不同）。

### 1. V2 第一/二/三轮：已知时空输入（#71/M1）与位置化 process 读取（#72 RW-A）**已实现并已判决**

| 轮次 | 文档 | 臂 | 判决（原文口径） | 实测 |
| --- | --- | --- | --- | --- |
| 一 | `docs/R7_71_72_M1_AND_RWA.md` | pooled vs spacetime_rwa | 85 格 42 改善/11 恶化/32 未决；**容量混杂**（+6.02% 参数、+8.26% FLOPs） | 0.1846 GPU-h，2 seed |
| 二 | `docs/R7_71_72_ROUND_TWO_ATTRIBUTION.md` | A/B/C/D 四臂 | **C−D（位置依赖本身）= 8 改善/13 恶化/64 未决 → 位置依赖未获支持** | 0.5253 GPU-h，3 seed |
| 三 | `docs/R7_71_72_ROUND_THREE.md` | A/B/E/P（E=constant，与 B **张量与 FLOPs 逐位相等**） | B−E 40/4/41；s_E=(E−A)/(B−A)= −0.037/0.055/0.126（12/24/48h）；**「容量/偏置不重现收益 ×1，信息主导 ×2」** | 0.5489 GPU-h，3 seed |

第三轮的作者读数：B−A 的 87%–104% 来自「模块被喂了什么」，即**信息**，而不是容量或位置化读取的结构。
第三轮另记 `E − A` 在 t2m 48h −0.247 K、72h −0.291 K（三 seed 同号）——「模块在场、输入无信息」的效应。

三处实现状态（HEAD 现状）：

- `model/spacetime_conditioning_r7.py`：`spacetime_inputs` / `spacetime_field_mode`
  （`fields` / `constant` / `shuffled`），字段 `latitude` / `longitude` / `init_utc_hour` /
  `init_day_of_year`，valid time = init + lead。
- `model/process_readout_r7.py::PositionalProcessReadout`：`positional_process_readout` /
  `pooled_readout_query`；每位置一个 query 对 M 个 process token 做 cross-attention（N×M），
  输出**相加**进 solver context；**无门控、无每位置状态**。
- 缺什么：**没有 Z（每位置工作状态）**、**没有门控**、**过程监督与预报修正仍只有弱耦合**、
  **训练从未做真实自回归**。

### 2. 过程监督与深度（#65 C1/C2/C3、#66）——已判决，不再重跑

C1/C2/C3 结论见上；关键阈值事实：`eps=1e-6` floor 在 8 个 proxy 上**恰好命中 2 个**
（`moisture_advection_850_mean` 原始 std 9.1e-9、`moisture_convergence_850_mean` 1.7e-8，
归一化后 std 仅 0.0076 / 0.0118），因此**现行 aux 监督里有 2/8 个标签不可用**。
#66/S4（`docs/R7_66_GATE_AUDIT.md`）：自适应的前置 gate **不成立**——最优 K 只在 6h
seed-一致，而那里最优 K 恰是最贵的 K=4 且增益 0.00%；oracle 逐样本最优 K 波动 +6.9…+13.0%
不可部署。**默认预期是 M5 的 adaptive 部分不启动。**

### 3. 数据与基线现状（引用时必须带条件）

- M2 段（`docs/R7_69_BUCKET_EXPANSION.md`）：240 个连续 6 小时时次（2016-01-01…02-29），
  17 通道、65×65、0.25° 原生；manifest train 186 / val 22 / test 26；
  `data_identity ef8c6691…`；气候态桶 4→8。
- **test 已经被看过**，因此不能再包装成 pristine；开发只用 train/val。
- M2 多种子（1.204 GPU-h）：t2m 上气候态 74/75 格胜出（三 seed 一致），量级 **1.2–1.4×**
  （「9.09×」已被撤回，见决策 0010）；全表 705 优于/570 劣于（1275 格）；
  process vs generic = 2 改善/12 恶化/71 未决（85 格）。

## Current architecture limitations

`model/process_forecast_r7.py:249-282` 的递推现状（行号为 HEAD 实测）：

1. **P 是被池化的全局集合。** 默认路径 `process.mean(dim=1)` 在
   `model/process_forecast_r7.py:182`，投影成 `[B,D]` 后被
   `model/recursive_weather_r7.py:33` 的 `conditioned = context + summary[:, None, :]`
   **广播到所有位置**——不同位置无法选择不同的 process 信息。
2. **RW-A 的读取是集合语义、加性、无门控、无状态。** 见 `model/process_readout_r7.py:10-17`
   自己写明的三条边界；它**不是**有序的，也没有每位置工作状态。
3. **draft 反馈是参数free 的逐位相加。** `spatial_solver_feedback` 只做
   `conditioned + draft_tokens`（`model/recursive_weather_r7.py:42-45`），0 参数，
   打开/关闭 FLOPs 逐位相同（C2 已测），且已有负结果——**不得把重开这个开关当新方法**。
4. **没有 Z。** 唯一空间分辨的递推量是 draft Y；P 只能通过「键长翻倍的 cross-attention」
   与（可选）那次常量加法间接看到 Y。
5. **时间语义未分轴。** 内部 K 步**不推进物理时间**；`lead_time_hours` 只在
   (`model/weather_forecaster_r7.py:68-70`) 注入一次；真正的自回归只在
   `model/r7_rollout.py` 的**评估**路径里，训练从未见过模型自己的预测。
6. **修正量是累加的，不是锚定的。** `correction_head` 每一步返回 `base_state + tendency`
   且 `base_state` 传的是当前 `draft`（`model/process_forecast_r7.py:272-277`），
   于是 Y 是增量的累积，没有「相对 X_t 的绝对提案 + 门控混合」这一步。
7. **同一递推步骤有三份实现**：`model/process_forecast_r7.py:249-282`、
   `training/r7_streaming.py`、`model/r7_halting.py:123-140`。#72 明文要求「共同复用一个 step
   实现，防公式漂移」——这是一处已知的漂移风险，不是本轮才发现的问题。

## V2 architecture

第一版候选（方程与符号在 `docs/R7_MAIN_MODEL_V2_DESIGN.md` 冻结，那里是权威定义）：

```
C          = Encoder(history, known spatiotemporal context)      # 已有（M1）
R_k(i)     = CrossAttention(Q(C_i, E(Y_k)_i, pos_i), P_k)        # 已有（RW-A 的读）
Z_(k+1)    = LocalUpdate(Z_k, C, E(Y_k), R_k, step_embedding)    # 新增（RW-B）
Y_proposal = X_t + Decoder(Z_(k+1))                              # 锚定 X_t，而非累加
Y_(k+1)    = Y_k + g_k * (Y_proposal - Y_k)                      # 门控混合
```

- **P**：全局/半全局气象过程状态，M≈16 个 token，`[B,M,D]`；它表达跨区域的过程摘要。
- **Z**：**每 patch 的空间工作/求解状态**，`[B,N,D]`，由**局部门控单元**（3×3 深度可分离卷积 +
  pointwise 的 ConvGRU 式更新）在 patch 网格上更新，O(N)。
- **Y**：预报草稿 `[B,C_out,H,W]`。
- 三者都是**隐式向量计算**，不是自然语言 CoT；**不引入** N² 全图 attention，
  **不引入** RAFT 的 4D correlation volume。

计算边界（#72 冻结）：读 O(N·M)，局部更新 O(N)×小 kernel；持续保存 Z 会增加每轮工作内存，
**不得未经测量就称 VRAM 恒定**。

规模纪律：第一轮维持现有 ~2.8M 档（process 2,799,779 / +RW-A 2,968,259 / generic 2,799,202），
**不**默认扩到 15M/18M/30M；只有小规模机制成立后才允许 15–20M 做容量验证
（B3 的 6.47× 容量实验没有证明单纯扩模解决核心问题）。

## Tensor contracts

权威定义在 `docs/R7_MAIN_MODEL_V2_DESIGN.md`。此处只列实现前必须固定的接口：

| 量 | 形状 | 语义 |
| --- | --- | --- |
| `history` | `[B,T,C,H,W]`，T=2 | 已归一化历史，6 小时间隔 |
| `C` / `context_tokens` | `[B,N,D]`，N=ceil(H/p)·ceil(W/p)=33·33 | patch 网格上的编码 |
| `P` | `[B,M,D]`，M=anchored+free | 过程状态；前 anchored 个与诊断目标 1:1 |
| `Z` | `[B,N,D]` | **每位置**工作状态（RW-B 新增） |
| `E(Y_k)` / `draft_tokens` | `[B,N,D]` | draft 的 patch 编码 |
| `R_k` | `[B,N,D]` | 逐位置过程读（N×M） |
| `Y_k` | `[B,C_out,H,W]` | 预报草稿；`Y_0` = backbone 初始预报 |
| `g_k` | `[B,N,1]`（门控标量） | sigmoid，初值温和 |

## M1–M5 phases

阶段顺序**按机制依赖**，不按 issue 编号。

| 阶段 | issue | 状态（HEAD 实测） | 内容 |
| --- | --- | --- | --- |
| M1 已知时空条件 | #71 | **已实现 + 已跑三轮（判决：信息主导）** | `spacetime_inputs` 已贯通 dataset→白名单→forward→streamed→rollout→adaptive |
| M2-A 空间 process 读写（RW-A） | #72 | **已实现 + 已跑三轮（判决：位置依赖未获支持）** | `PositionalProcessReadout`；加性、无门控、无状态 |
| M2-B 局部门控求解状态（RW-B） | #72 | **未实现 —— 本计划的关键路径** | Z、门控、锚定 X_t 的绝对提案 |
| M3 预报相关的过程监督 | #73 | 未实现 | 尺度修复（维度化缩放 + degenerate mask）+ 明确时刻语义（input / future / draft 三类名字分离） |
| M4 真自回归修正训练 | #74 | 未实现 | 从同一 +6h 父 checkpoint 做 2 步可微 rollout，L6+λL12 |
| M5 最小确认 + 有条件自适应 | #75 | 未实现 | ≥3 预声明 seed；旧 Ours / 匹配 Generic V2 / Process V2；adaptive 前置 gate 当前**不成立** |

**两轮各自留下的、已预注册但未执行的「下一项第一个具体动作」必须作为下一轮的第一步**（见 E0）：

- 二轮 §11：用已归档 checkpoint 跑 `diagnose_r7_gain.py`，比较 B 与 C/D 的 correction 幅度与符号，
  判断负向是否来自「读取把 context 拉偏」。
- 三轮 §13：把三轮 `E − A`（t2m 48/72h −0.247/−0.291 K，三 seed 同号）与二轮 `C−B`/`D−B`
  （0.22–0.52 K）并排核对，判断这 0.25–0.29 K 是否跨两次独立运行稳定复现。

## Dependency graph

```
M1 已完成 ──┐
            ├──> M2-A 已完成（判决：位置依赖未获支持）
            │
诊断 E0 ────┴──> M2-B (RW-B) ──┬──> M4 (2-step AR fine-tune)
                               │        │
M3 尺度/标签准备（可并行）──────┴────────┴──> M5 最小确认 ──> （conditional）adaptive
```

- M2-B 与 M3 的**标签/尺度准备**可以并行；M3 的**集成**依赖 M2-B 的 solver 冻结。
- M4 必须等主模型能正确 forward/train 之后；它只依赖一个已学 +6h 的父 checkpoint。
- M5 的 adaptive 只有在固定 K 形成**有用的 accuracy–compute 前沿**时才执行；
  #66/S4 已证当前不成立，因此默认预期是**不启动**，而不是「找办法让它成立」。

## Exact source files expected to change

`model/**` 任一字节变化都会改变 `model_code_sha256`（`training/r7_experiment.py:31-40`），
提交信息必须带 `[model-digest-change]`，旧产物只能用其归档 `code.zip` 重放（Q-009）。

| 文件 | 预期改动 | 阶段 |
| --- | --- | --- |
| `model/local_solver_state_r7.py`（**新增**） | Z 的局部门控更新单元 + 门控头 | M2-B |
| `model/process_forecast_r7.py` | 新开关 `local_solver_state`；Z 的初始化/推进；锚定 X_t 的提案 + 门控混合；在 `isolated_stream()` 下**最后**构造新参数以保持「关=逐位相同」 | M2-B |
| `model/recursive_weather_r7.py` | 允许 generic 挂同款局部状态（匹配 Generic V2 归因对照的前置） | M2-B/M5 |
| `model/process_readout_r7.py` | 只在需要时把读取改成有门控的写入端（RW-B 的写） | M2-B |
| `model/r7_halting.py` | 让 adaptive wrapper 复用共享 step，而不是再抄一份 | M2-B |
| `training/r7_streaming.py` | 同上：streamed 路径复用共享 step | M2-B |
| `training/r7_process_forecast_losses.py` | M3 的三类诊断名字与 mask | M3 |
| `data/preprocess/process_diagnostics.py` + 派生 sidecar | 维度化缩放与版本化元数据（**不改写 store、不重下数据**） | M3 |
| `training/r7_scheduled_runner.py` | 2 步物理展开的训练接线（尽薄） | M4 |
| `scripts/study_r7_*.py`（新增一轮驱动） | 冻结协议 + 逐 seed 运行 | E1/E4 |
| `tests/test_r7_local_solver_state.py`（新增）及既有等价性测试 | 见测试清单 | M2-B |

**一处需要单列的非局部决定**：目前同一递推步骤有三份实现（见「限制 7」）。建议在 RW-B 落地时
抽成**一个共享 step 函数**供 `forward` / streamed / adaptive 三处调用（替代方案：继续三份镜像 +
三处等价性测试）。这是本计划里唯一的非局部重构，故在此显式列出以便评审否决。

## Experimental sequence E1–E5

**禁止**回到 `6 baseline × 5 depths × N seeds × N sizes` 的大矩阵。

| 编号 | 内容 | 臂 | 预算 | 输出 |
| --- | --- | --- | --- | --- |
| **E0** | 执行二轮 §11 与三轮 §13 的预注册诊断 | 无训练，只读已归档 checkpoint + val | 0 GPU-h（CPU/val） | 决定 RW-B 的假设是「读取把 context 拉偏」还是别的 |
| **E1** | RW-B vs RW-A（只改求解结构，**不同时改 loss**） | 旧 mean-Ours / RW-A / RW-B | 2 seed × 400 updates，≤1.0 GPU-h | 逐变量 6/12/24/48/72h + K1/2/4 探针 + 修正幅度/角度 |
| **E2** | 过程监督（固定在已冻结的 V2 solver 上） | 最多三臂：aux off / 修尺度 input aux / 修尺度 future+draft aux | 2 seed 探索，≤1.0 GPU-h | 尺度修复是否让诊断信号正常；不重做大规模权重扫参 |
| **E3** | 2 步可微自回归微调 | 同一 +6h 父 checkpoint：继续 +6h 训练 vs L6+λL12 真实 2 步 rollout | 2 seed，≤1.0 GPU-h | 明确记录**算力不相等** |
| **E4** | 确认 | 旧 Ours / 匹配 Generic V2 / Process V2，≥3 预声明 seed | ≤2.5 GPU-h | 每变量/时效三态计数；同 seed 配对 |
| **E5** | 有条件自适应 | 仅在 E4 出现固定 K 前沿时才讨论 | 另立预算 | 否则如实写「不启动」 |

每次只改一个模型因素；获支持的候选才补第三 seed。E0 未完成前不开 E1。

## Data contract

- 复用已存在、已验证的 M2 段（`outputs/r7_m2_segment/store/cache.zarr`，240 时次、17 通道、
  65×65、`data_identity ef8c6691…`、`BUILD_COMPLETE.json`），**本轮不下载任何数据**。
- 只用 train/val；**test 全程封存**，且不得用于任何方法选择。
- **现有 test 已被开发阶段看过，不能重新包装成 pristine**；真正的 publication test 以后另行冻结
  （届时才值得申请一次明确的数据预算）。
- **申请新数据预算的触发条件**：V2 的**固定**机制出现可重复的 validation 信号。
  不是「先下几年数据再看看」。
- 任何进入训练的真实数据仍须先过 `prepare_r7_local.py` 的只读 preflight 并取得显式 `--write` 授权；
  `data/raw|interim|processed` 一律不写。
- M3 的尺度修复产出**派生 sidecar**（放在 store 同级、`outputs/` 下），其 identity 纳入新训练
  contract；**不原地修改旧 store 或旧 checkpoint 的 identity**。

## GPU/resource budget

两级预算，具体值必须**在执行那一刻重读账本**，不得沿用旧提示词数字：

- **Pilot**：每候选 ≤2 seed、每臂固定最大 updates（建议 400）与墙钟上限；目的是方向筛选。
  每轮自设上限 **≤1.0 GPU-h**（与二/三轮同规格）。
- **Confirmation**：只有 Pilot 支持后才补 ≥3 固定 seed，本轮上限 **≤2.5 GPU-h**。

成本参考（实测，非估计）：M2 段上 `dim=192` 约 **0.31–0.41 s/update/臂**；
400 updates × 4 臂 × 3 seed ≈ 0.55 GPU-h；800 × 5 臂 × 3 seed ≈ 1.2 GPU-h。
第二批次授权 ≤24 GPU-h；第三轮文档起点的剩余为 **21.537 GPU-h**，此后未重记——开工前必须重读。

双 3090 策略：**GPU0 = arm/seed A、GPU1 = arm/seed B 独立训练**；DDP 不作前置
（实测该规模 DDP 慢 ~7%，resume 111/111 权重逐位相同）。
单次实验 ≤30 min，超时须拆分（决策 0010 §3 的第二阶段常量）。

## Evaluation protocol

- 指标用**已修复单位**的 RMSE / climatology MSE skill / ACC；不同物理单位**不得**直接平均；
  正 ACC **不自动**等于正 MSE skill。
- 逐变量、逐时效（6/12/24/48/72h）全部公布，**长时效失败不得隐藏**。
- 案例与 seed 按 #60 比较器显式配对；单位/案例集不一致一律 fail closed。
- **primary 端点必须在运行前冻结**（建议 t2m 的 6/12h，并写明允许的退化），不得事后挑。
- 同 K=4 checkpoint 上取出的 K=1 探针必须标注为「不是独立训练的 K=1」。
- 成本四表必备：参数量 / forward FLOPs / forward+backward FLOPs / 墙钟与 case 数。
- `scientific_claim: false` 与 `limitations` 写进协议、逐 seed 结果、合并结果与回执四处。

## Scientific gates

三个层级必须分开报，不能互相冒充：

1. **ENGINEERING PASS**：代码正确（forward/backward/形状/梯度/开关默认关逐位相同/resume/BF16）。
   它**不**意味着模型更好。
2. **EXPERIMENTAL SUPPORT**：在冻结的小规模 protocol 下主要端点有**重复**改善
   （同一 seed 配对、同号）。
3. **SCIENTIFIC SUPPORT**：≥3 固定 seed、精确配对案例、公平信息预算、公平算力报告、
   代表性验证、冻结的最终 test、不确定度、以及**保留负结果**。

**基线冻结不等于写便宜话**（决策 0023）：开发期不要求基线全胜；但任何「过程结构有用」的声明
必须带**匹配 Generic**（同结构、同已知输入、无过程语义）对照；完整强基线表在发表阶段按冻结配方
重跑一次。

## External Reference Ledger

完整台账（字段：Repository / Paper / Access date / Commit-tag / License / Exact source file /
Concept borrowed / What we implement ourselves / What we deliberately do NOT copy / Why relevant）
在 `docs/R7_MAIN_MODEL_V2_REFERENCES.md`。摘要：

| 来源 | 借鉴什么 | 明确不做什么 |
| --- | --- | --- |
| TRM（MIT，`c0110373…`，已归档） | 共享潜变量递归精化与 draft 的交替更新语义 | 不声称逐行 TRM，不把文本/谜题答案状态当作气象过程态 |
| HRM（Apache-2.0，`ac15626f…`） | 「规划态/工作态」双时间尺度分工的动机 | 不复现其层级架构，也不用它的 halting（我们已测无益） |
| Perceiver IO（Apache-2.0，JAX；**SHA 待固定**） | 输出 query 从少量潜变量读出密集输出 | 不引入整个 Perceiver/JAX 框架 |
| RAFT（BSD-3，`2888e15a…`） | 局部门控更新单元 + 逐步监督的成熟配方 | 不复制光流任务、4D correlation volume、warp |
| MetPy（BSD-3，`07df928b…`） | 诊断公式/单位/地理 metric 的**离线 oracle** | 不在 GPU 前向里经 CPU/pint 往返 |
| WeatherBench-X / WeatherBench 2（Apache-2.0） | 指标定义与约定的对照 | 不迁移整个评估框架；现有已验证实现优先 |
| 新增：arXiv:2608.21677「Read, Write, Relax」 | 「latent-token attention 是空间低通滤波、全局潜态与局部处理必须混合」的机制论证——**与本计划假设最接近的公开论证** | 它是 PDE 仿真，无天气、无递归预报、无过程监督 |
| 新增：Aurora（arXiv:2405.13063） | 逐位置从小潜变量集合读写的工程先例 | 1.3B 基础模型、潜变量是垂直层、无递归推理——不同规模与不同科学问题 |

novelty 判定照实写：**组合**（小规模潜变量递归精化 + 空间寻址的过程态用于区域天气）未见发表，
但每个组件都有强先在性；论文的贡献必须落在「小数据/小规模下逐位置过程条件化 vs 全局广播的
受控对比」。

## Risks

| 风险 | 触发信号 | 处置 |
| --- | --- | --- |
| RW-B 无效 | E1 上无同号改善 | 走「失败出口」：先查 position binding / 梯度 / query-write 路径 / 容量；工程正确才允许执行**一次**预声明 RW-B 变体 |
| 三处 step 实现漂移 | fixed/streamed/adaptive 数值不一致 | 共享 step 函数 + 等价性测试 |
| 门控切断梯度 | 门控饱和到 0，decoder 梯度消失 | 门控 bias 初值温和 + 测试断言梯度非零 |
| 容量混杂 | 参数/FLOPs 差超 25% | 冻结前实测，超出则缩 hidden；容量差如实报告为混杂 |
| 「匹配 Generic」缺失 | 只有 process 臂变强 | M5 前必须补齐同结构同输入的 Generic 对照 |
| eps floor 未修就测 aux | 2/8 标签被 floor 压制 | M3 先修尺度 + degenerate mask，否则不得解释 aux 结论 |
| 3 seed 被当显著性 | 任何「显著」措辞 | 三个 seed 只给一致性，不引入阈值 |
| 模型 digest 变化打断旧产物 | 旧 checkpoint 拒绝加载 | 归档 `code.zip` 重放；提交带 `[model-digest-change]`；CHANGELOG 记录 |
| 预算越界 | 未重读账本 | 执行前必须重读并取得决策 0021 授权 |

## Failure exits

- **M1 negative**：保留正确接口，继续 M2；known context 不作为论文创新点（**已发生**：第三轮读数
  是「信息主导」，因此 M1 的价值是提供信息，不是提供机制）。
- **RW-A negative**：第二轮已发生「位置依赖未获支持」；不把它包装成新创新，RWB 成为下一假设。
- **RW-B negative**：先查 position binding、梯度、query/write 路径、容量；**工程正确时只执行一次**
  预声明变体；仍然 negative 就**停止继续发明**（不做 RW-C/RW-D/RW-E），转而重新检查
  forecast state、training objective、data regime、autoregressive exposure。
- **M3 negative**：过程监督可降级为诊断/可解释性辅助，**不**强行保留为核心训练 loss。
- **M4 negative**：不自动扩到 4/8/12 步。
- **Adaptive negative**：保留 fixed V2，不为了「核心创新必须成功」无限加 controller。

## Rollback strategy

- 每个阶段一个可独立回退的开关，**默认关闭时逐位等价于前实现**（仓库既有做法：
  新模块在 `isolated_stream()` 下最后构造，见 `model/process_forecast_r7.py:162-165`）。
- 候选负面即回退该候选并记录可反驳假设，**不**因此重启数据/基线战役。
- 连续两次定向改动无收益：停止加开关，重新分析预测状态/边界/数据量。
- 计划被证据否定时**改写计划而不是改写判据**；已冻结的阈值与已记录的负结果不动。

## Definition of Done

1. RW-B 的真实模型代码（forward + 训练路径 + adaptive 路径同源），开关默认关闭逐位等价；
2. 完整梯度与接口测试（含 poison、odd grid、K=0/1/2/4、resume、BF16、active-subset 等价）；
3. **至少一轮有界真实 M2 train/val 主模型对照**，协议在第一步优化器更新前冻结；
4. 参数量/FLOPs/显存/墙钟实测四表；
5. 逐变量 × 6/12/24/48/72h 全表（含坏变量），三态计数（改善/恶化/未决）；
6. `docs/R7_*.md` 证据文档 + `docs/R7_EVIDENCE_INDEX.jsonl` 记录 + 回执（若适用）；
7. 明确写清「未做的事」与 limitations；negative 结果原样保留。

**发明一个名字、增加 Issue 或 CI 测试数不算主模型创新成功。**

## 实际结果（Actual Results）

NOT EXECUTED YET —— 本计划在实现之前归档，尚无属于它的实验。E0–E5 全部未执行；
`protocol.json`、GPU hour、逐变量表格都不存在，不得被任何下游文档引用为结果。

本次 Plan Mode **实际做过**的事（与计划正文区分）：

- 只读核对：`git` ref（HEAD `e1d5a7d` / origin/main `dafd22e0`）、GitHub 匿名 API
  （仓库元数据、分支头、#70–#75 全文与评论、最近 5 次提交）、模型与训练代码、
  S3–S5 与 V2 三轮的既有证据文档、规则与技能（`docs/rules/`、`.agents/skills/`）。
- 产出：本计划、`docs/R7_MAIN_MODEL_V2_DESIGN.md`、`docs/R7_MAIN_MODEL_V2_REFERENCES.md`、
  `docs/goals/main-model-v2-rw-b-round.md`、决策 0023、`docs/R7_ISSUE_COMMENTS.md` 的
  #70–#75 评论草稿、`docs/rules/EVIDENCE.md` 的 **E-195**（设计契约 §3.2 的代码事实依据）。
- 验证：`python tools/check_conventions.py`（37 条阻断，0 违规）、
  `python tools/check_goal_brief.py --brief docs/goals/main-model-v2-rw-b-round.md`
  （failures=0, advisories=0, exit 0）、本计划与两个子目录 README 的索引行、逐文件 EOF 空行与
  行尾空白检查（均为 0）、`git status --porcelain`（改动只落在 `docs/**`）。
  精确计数与 CI run id 见本轮提交信息。
- **未做**：未修改 `model/**`、`training/**`、`data/**`、`scripts/**` 的行为；未训练、未租 GPU、
  未下载数据、未读封存 test、未新增 baseline family、未关闭任何 issue、未合并 main、未 force push、
  未创建定时任务、未自动进入 Goal Mode。

- **与计划的差异**：提示词假定 #71 与 #72 RW-A 尚未开始并要求下一轮实现它们；实测二者**已实现并已跑三轮**
  （见「Current evidence」），因此本计划把 M1/RW-A 记为「已完成 + 已判决」，把 RW-B 作为关键路径。
  三处刻意的偏离（阶段重排、goal 文件名改为 kebab-case、issue 评论改为落仓草稿）已在本计划 §0/§1
  与提交信息中写明。
- **遗留**：`docs/rules/OPEN_QUESTIONS.md` 的 Q-009（HEAD 与归档产物的重放脆弱性）、Q-011（未声明依赖）、
  Q-012（历史 `--device cuda` 声明）、Q-013（子会话是否跑 hook）全部仍为 open，本计划不解决它们。
