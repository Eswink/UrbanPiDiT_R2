# 0022 — S3 架构方向下一轮：候选假设与结构测绘（定稿归档）

**状态：定稿归档，不是 round brief，不产生任何科学结论。** 本文件把 2026-10-12 两个窗口留在
`.zcode/plans/`（git-ignored 工作态）里的侦察笔记整理归档，供上下文丢失后恢复。
**正式 brief 必须在下一轮由主链按完整协议 / 身份 / digest 写定**，本文件不替代它。
预算是 0 GPU-h：本文件的全部内容都来自只读测绘与文献检索，**没有任何实验被执行**。

---

## 0 触发条件（本计划何时才适用）

本计划只服务于 **capacity（depth 4→8）轮之后**的那一轮，且**仅当** capacity 轮的预声明出口触发时适用：

- capacity 轮 `docs/goals/s3-capacity-or-architecture.md` §3 预声明出口：**72 h 三 seed 全正 → 提请独立验收**；
- 否则 **关闭"靠加大容量解决 72 h"这一支**，转**架构方向**（本计划即该方向的候选）。

若 capacity 轮的 72 h 三 seed 转正，本计划**搁置**，下一动作改为独立验收，而不是再开架构臂。

## 1 起点与已关闭 / 已登记的支（不重开）

- 起点：`docs/goals/main-model-climatology-campaign.md` §8 最新块 + `docs/R7_S3_CAPACITY_DEPTH_RUN.md`（待登记）。
- **训练配方支已关闭**（不重开）：输入范围（index67 全网格 null）、监督域（index68 稀释已分离）、
  剂量 800→2400（index69，48 h 转正、72 h 仍负）、长 lead 权重 0.5→1.0（index70 近乎无效）、
  推理深度 K 4→8（index71，操作者中止、单 seed 旁证 Δ≤0.009 K）。
- **已登记负结果（不重跑同一因子）**：#77 PE 频带（unresolved）、#78 R-A 变化尺度、#78 R-B 加权损失、
  #79 有类型诊断（**pooled 版本**，B→C 归因负）。
- **暂缓**：数据扩充（§0 objective 的 D3 明确"暂缓、放到最最后"），本计划不涉数据。

## 2 只读测绘得到的结构事实（已由主链逐条复核 file:line）

1. **P_0 与样本无关**：`model/process_forecast_r7.py:434` `process=self.process_queries.expand(B,-1,-1)`
   —— 过程状态初值 `[1,16,dim]` 对每个样本逐位相同，输入信息只经 `C`（context）的 cross-attention 进入。
2. **推理算子权重共享、唯一步数深度 = 1**：`model/recursive_weather_r7.py:127-139` 单个
   `GenericRecursiveCell` 被 K 次复用（`process_forecast_r7.py:250-252, 441`）。
3. **诊断读出只覆盖前 8 个 anchored slot 且不进前向**：`process_forecast_r7.py:254-257, 385-391`
   （`LayerNorm+Linear(dim,1)`，`process_predictions [B,K,8]`），只被 loss / 控制器消费
   （`training/r7_process_forecast_losses.py:171-172`、`model/r7_halting.py:190-195`）。
4. **RW-B 空间混合只有 1 个 3×3 depthwise conv**：`model/local_solver_state_r7.py:107-108, 129`；
   解码器 `model/coarse_forecast.py:79-80` 是 `ConvTranspose2d + Conv2d(3×3)`，**无全局混合**；
   全局信息只能经 `model/coarse_encoder.py:23-24` 的**移位局部窗口**间接传播。
5. **有类型证据把空间场压成每类型 1 个向量**：`model/typed_evidence_r7.py:199`
   `torch.stack([value.mean(dim=1) for value in projected], dim=1)` —— `dim=1` 是 patch 位置轴，
   **局部空间结构在注入前被均值抹掉**；只写 4 个 anchored slot（`:50, 220-233`）。
6. **legacy 位置基在 33 点轴上只有约 16/192 通道随空间变化**：`model/process_readout_r7.py:68-79, 86-88, 113`。
7. **长 rollout 只监督 12 个物理步中的 5 个**：`training/r7_long_rollout.py:81-96` + 权重向量
   `scripts/study_r7_s3_wide_single_factor.py:38`（非零步 0,1,3,7,11 → 6/12/24/48/72 h）。

### 2b 两条"加深"轴：项目**已经**都具备（本轮的关键载荷事实）

- `GenericRecursiveWeatherForecaster`（`model/recursive_weather_r7.py:254`）**同时**具备：
  - **主干 `depth`**（`NativeAtmosForecaster` 内 `WindowAttentionBlock` 层数，`recursive_weather_r7.py:339`）
    —— **不同块堆叠**；本轮 capacity 轮测的就是它（4→8）。
  - **递归 cell `K`**（`default_reasoning_steps`，`recursive_weather_r7.py:336/406`）——`z` 在**权重共享**的
    `GenericRecursiveCell`（`:346`）里迭代 K 次，每步用 `generic_reasoning_step`（`:175`）精修 `draft`。
    这正是文献所说的 **recurrent depth（weight-tied 循环）**。
- 因此：**"循环共享块加深"这条轴项目已经试过**（K 4→8，读数 Δ≤0.009 K、无变化）。
  若本轮 depth 4→8（不同块）也无收益，则两条"加深"轴都无收益 → 指向**不是容量、而是训练信号 / 目标组织**。
- 注意：`z` 初值是学习参数 `latent`（`:344`）扩到 batch，**每步不喂输入**，只喂 draft 反馈；
  若 K 无收益，可疑点是**多步缺乏可用信号**（后几步对 loss 无边际贡献），而非 cell 容量。

## 3 候选假设（每个按 D2 要求写全六项）

> 口径：全部为 val-only 单因素筛选；对照 = 已登记 2400 剂量臂（`record:s3-wide-interior-dose`，pinned 复用）；
> 评分域 `interior_32`；test 从不读取；`scientific_claim:false`；不宣布通过。

### H-A（推荐首选）输入条件化的过程状态初始化

- **实质差异**：把 `P_0` 从"全体样本共享的 `process_queries`"改为**逐样本条件化**
  （例：`P_0 = process_queries.expand(B,·,·) + proj(pool(C))`，或对 `C` 做一次 cross-attention 得到
  per-sample 初值）。**唯一改动因子 = P_0 的条件化**；K、depth、宽输入、监督域、updates、权重、父、
  seed、评分域、气候态分母全部逐字不变。
  - **插入点（已核对 `model/process_forecast_r7.py:434-435`）**：在
    `process=self.process_queries.expand(B,-1,-1)` 之后、`initial_process_state(...)` 之前插入逐样本项，
    例如 `process = process + self.process_init_proj(context.mean(dim=1)).unsqueeze(1)`。
    `process_init_proj`（`Sequential(LayerNorm, Linear(dim,dim))`）必须**最后、在 `isolated_stream()` 下**
    构造（与 `process_reader`/`solver_*`/`typed_evidence` 同规矩），使"off"路径逐位等于现实现；
    再加一条"`Linear` 权重清零"的控制臂做逐位恒等反证。
- **正/负控制**：正控制 = `proj` 权重置零时应**逐位**退回注册臂（恒等反证，仿 depth 扩张的
  `encoder_bit_identical`）；负控制 = `matched_generic` 同图等价路径（同参数量/FLOPs，只改路由）。
  必须报新增参数量与 FLOPs，且**同 updates 非同算力**分开报。
- **预期信息**：若 P_0 与样本无关是"推理起点无输入信息"的瓶颈，则长 lead（尤其 48/72 h）t2m
  Δrmse 应显著为负；若 Δ 在噪声内，则该结构事实**不是** 72 h 瓶颈，据此停。
- **新 protocol/输出**：新驱动 + 新输出目录 + 新 `protocol.json`/`code.zip`；receipt 记
  `p0_conditioning` 声明块与 `p0_zeroed_bit_identical` 断言。
- **停止/饱和出口**：72 h 三 seed 全正 → 提请独立验收；否则**关闭该支**、不调 proj 宽度/不加 seed。
- **回主线动作**：停后回 §3 的 H-B/H-C 或数据方向。

### H-B 空间分辨的有类型证据（与 #79 pooled 版本**不同机制**）

- **实质差异**：把 `typed_evidence_r7.py:199` 的"按位置均值压成 1 向量"改为**保留 patch 位置**
  （每类型保留 `[B,P,dim]` 局部证据 token，注入对应 anchored slot 时用位置对齐的加法/注意力，
  而不是 broadcast 一个全局向量）。**同信息、不同路由/组织** —— 这正是协议允许的"同信息不同路由"对照。
- **正/负控制**：同信息对照 = **A 当前（无类型）** vs **B 同信息无类型（pooled）** vs **C 类型+空间保留**；
  只检 B→C 的归因。控制臂参数量/FLOPs 对齐；类型标签置零反证。
- **预期信息**：若 #79 的负结果来自"空间结构被均值抹掉"而非"类型信息无用"，则空间保留版应出现
  局部/长 lead 收益；否则类型路由本身无效。
- **新 protocol/输出**：新驱动/输出/协议；mask/证据声明块进 contract 与 signature。
- **停止/饱和出口**：同 H-A；B→C 三 seed 同号且 72 h 转正才提请验收，否则停该支。
- **回主线动作**：停后回 H-A/H-C。

### H-C 长 lead 的全局混合路径（成本最高，排最后）

- **实质差异**：给解码器或 solver 增加一条**全局混合**路径（低秩全局算子或单层全局注意力），
  补上结构事实 4 的缺口；局部窗口与 3×3 conv 不变。
- **正/负控制**：与"同参数量的额外局部层"对照（区分"全局性"与"单纯加容量"）；参数量/FLOPs 对齐。
- **预期信息**：长 lead 误差以大尺度为主，若全局混合缺失是瓶颈，则 48/72 h 应改善；若与加局部层
  无差异，则瓶颈不是感受野而是别的。
- **停止/饱和出口**：同 H-A。
- **回主线动作**：停后回数据方向或重估主线。

## 4 排序与推荐

**H-A > H-B > H-C**。理由：H-A 改动最小（一个 proj + 恒等反证）、直接对应一个明确的结构缺陷、
成本最低；H-B 与已登记的 #79 pooled 负结果构成**同信息不同路由**的干净对照，信息量高；
H-C 改动最大、与"容量"假设部分重叠，排最后。

## 5 下一轮开工前必须补的 CPU 工作（不在本文件内做）

1. 从注册驱动派生新驱动，写新输出目录/新 `protocol.json`/新 `code.zip`；**逐行 diff** 证明单因素保真。
2. 为 H-A 写 `p0_zeroed_bit_identical` 反证测试；为 H-B 写 mask/证据声明块测试与类型置零反证。
3. 每 seed 冻结上限、planned/hard、`estimated_peak_bytes` 按实测**重新标定**（不得沿用旧值）。
4. 按 `planner-delegation` 出多步计划并过 `check_planner_plan`；按 `decision-record` 记本轮 ADR。
5. 跑 `tools/r7_round.py round-guard` 与提交前 `commit-guard`。

## 6 文献旁证（访问 2026-10-12）与其局限

一手来源（arXiv；**web-researcher 与 web-researcher-backup 均 provider 不可用**，按 objective 的降级路径
由主链路 `curl` 取得，原因已写明）：

1. **Huginn / recurrent depth** — Geiping et al., "Scaling up Test-Time Compute with Latent Reasoning:
   A Recurrent Depth Approach", arXiv **2502.05171**。迭代**同一个** recurrent block，测试时可 unroll 到任意深度；
   扩到 3.5B 参数，证明靠**循环共享块**而非堆叠不同块来加深。
2. **Universal Transformers** — arXiv **1807.03819**。循环 + 自适应计算。
3. **Deep Equilibrium Models** — arXiv **1909.01377**。深层序列模型看成收敛到不动点，等价于
   **无限深 weight-tied** 前馈网，常数内存。
4. 近期 recurrent-depth 方向（arXiv 检索，2025–2026）：`2604.07822` Loop/Think/Generalize、
   `2605.03999` RD-ViT、`2603.10544` SCORE（用 contractive recurrent depth **替代** layer stacking）。

**对 depth 4→8 的含义（推测）**：文献主流是**循环共享块**来加深，而非堆叠不同块；本项目已试过
循环深度 K 4→8（读数 Δ≤0.009 K、无变化）。若本轮 depth 4→8 仍无收益，则"加深主干"两条路
（不同块 / 循环）都无收益，指向**不是容量、而是训练信号或目标组织**（与 #79 typed diagnostics
进入前向、#78 变化量参数化/尺度 相关）。

**局限（必须照此读）**：主流 AI 天气模型（GraphCast/Pangu/FuXi/Aurora/NeuralGCM/GenCast）评估到 10–15 天，
**未见一手文献记载"确定式单模型在 48–72 h 相对气候态优势迅速衰减"**；但它们用 RMSE/ACC 对比 IFS，
**不是** MSESS vs train-only 气候态，且是**全球基础模型 + 40 年 ERA5**，与本仓"区域 + 少数据 + 小模型"
不可直接类比。**迭代精修用于确定性中期天气未找到一手工作** ⇒ 架构方向属**探索性**，文献不保证收益。
**不得把本节摘要当证据**；引用需带 URL 与访问日期。

## 实际结果

（归档时点 2026-10-12 深夜。）

- **本计划尚未被触发执行**：capacity（depth 4→8）轮**仍未跑 GPU**。三次 attempt 全部止步于
  prepare/archive（`attempt01` prepare 619.8 s + archive 614.6 s；`attempt02` 仅 prepare 593.3 s；
  `attempt03` prepare 630.2 s + archive 490.1 s），`attempt03` 的 `stage seed 41` 曾启动但在 preflight
  被停，**无 seed41 目录、无 readings.json**；**0 GPU-h**，GPU 全程只有邻居作业。
- **中断根因**：run `dwfrun-6c3847ad` 的 `.mjs` 进程从未死，但运行记录被判 `stopped/interrupted`、
  进度事件被丢 29 次（`"Dynamic workflow run progress dropped: parent session is not this app"`）——
  **会话跨了 ZCode app 实例**，完成通知无法投递。该 workflow 已由主链 SIGTERM 停掉。
- **下一动作**：先在**单一窗口**把 capacity 轮跑完并按 §3 预声明出口读（详见
  `docs/goals/s3-capacity-or-architecture.md` §6/§7 与 master §8 最新块）；本计划的 H-A/H-B/H-C
  **只在 capacity 轮出口触发"关闭容量支"之后**才启用。
