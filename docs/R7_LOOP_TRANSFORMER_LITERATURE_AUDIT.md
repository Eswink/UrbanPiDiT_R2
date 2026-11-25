# Loop transformer 文献审计：我们已经是什么、还缺什么、哪些差距值得测

**状态：只读审计，2026-11-25。`scientific_claim: false`、`test_read: false`、`r = 0`、0 GPU-h。**
本页回答一个具体问题：「用最新的 loop transformer 架构替换或结合我们当前的」。它**不产生任何
训练读数**，也不宣称任何架构改动会带来收益。所有本地事实带 `file:line`；所有外部事实带 arXiv id、
访问日期与「核到了哪一层」（摘要 / 章节标题 / 正文段落）。

## 0. 一句话结论

**我们已经是 looped transformer**——prelude–tied-core–coda sandwich 加 deep supervision，
正是 2026 年那批「looped models done right」论文推荐的形态。所以「替换成最新 loop 架构」不是一次
升级，而是**重开已经关闭的轴**；真正待补的是三条结构差异，而**先补哪一条取决于一次尚不存在的
headroom 读数**（本页不预设答案，判据见 `docs/decisions/0118-loop-depth-headroom-gate-and-branch-admission.md`）。

## 1. 我们的循环在代码里真的是什么（实测）

配置：`ProcessForecastCoReasoner`，`in_channels=out_channels=17`、`history_steps=2`、`dim=192`、
`depth=4`、`heads=4`、`window_size=4`、`patch_size=2`、`anchored_processes=8`、`free_processes=8`、
`default_reasoning_steps=4`（`outputs/r7_s3_typed_evidence_in_protocol_20261120_attempt02/protocol.json`
的 `registry[0].model.spec`）。

| 结构角色 | 本仓实现 | 参数量 | 占全模型 |
| --- | --- | ---: | ---: |
| **prelude** | `CoarseEncoder` 的 4 个**不共享** `WindowAttentionBlock`（`model/coarse_encoder.py`） | 1,806,144 | 64.5% |
| **recurrent core** | `GenericRecursiveCell`，被 K 次**复用**（`model/recursive_weather_r7.py:406`） | 742,272 | 26.5% |
| **coda** | `correction_head`（`model/coarse_forecast.py`） | 88,913 | 3.2% |
| 其它（process 侧） | `process_to_context` / `draft_encoder` / `process_queries` / `process_readout` | 142,071 | 5.1% |
| **合计** | | **2,799,779** | 100% |

**deep supervision 确在其位**：`training/r7_recursive_losses.py:151-157` 用
`torch.linspace(1.0, final_weight, steps)`（默认 `final_weight=2.0`）再归一化，对每个草稿都算
纬度加权 MSE；非最终草稿占权重 **73.3%**。

**循环的成本已经不可忽略**（实测，单 batch、65×65、RTX 3090、`torch.no_grad` 前向）：

| K | 前向 ms | loop 占前向 |
| ---: | ---: | ---: |
| 0 | 7.0 | — |
| 3 | 13.6 | **49.7%** |
| 4 | 16.8 | 59.1% |
| 8 | 24.2 | 71.7% |

训练步 K=1→3 的循环部分约占总步时 **25.6%**（6.5 ms/step）。

## 2. 与文献最佳实践的五条结构差异

### 2.1 循环状态只有 16 个 token，空间表征全程冻结（**最重要**）

- 过程态：`process = self.process_queries.expand(B,-1,-1)`（`model/process_forecast_r7.py:517`），
  即 **8 anchored + 8 free = 16 token**。
- 每步的 key/value 由 `reasoning_source_key(...)` 组装（`model/process_step_r7.py:112`）：
  `context` 是 prelude 产出的 33×33 = **1089 个 token，整个循环内逐位不变**；
  `draft_tokens` 每步把**空间场重新编码**回 1089 个 token。
- 每步**写回**的只有 `conditioned = context + summary`（per-position 加性调制），
  1089-token 的表征本身**从不被更新**。

⇒ 这是「16 个 token 的状态，读 2×1089、写回一条加性调制」。Huginn / Ouro 的循环把**全部 N 个
token 当状态**，每个 loop 完整读写整条序列。**我们的循环从未拥有改写空间表征的能力。**

### 2.2 无 per-iteration 门控输入注入

`z = self.latent.expand(B,-1,-1)`（`model/recursive_weather_r7.py:402`）：初值是学习常数、
batch 共享；之后**不再注入输入**，输入信息只经每步恒定的 cross-attention 进入。
已有的 `process_init_conditioning`（H-A）是**只注入一次**、且已按预声明判负关闭
（`record:s3-process-init-result`）。

### 2.3 无 tied-depth 残差缩放

`GenericRecursiveCell.forward`（`model/recursive_weather_r7.py:47`）是 pre-LN + 裸残差
`x = x + attn(LN(x))`，无形如 `α/x + β·branch(x)` 的 tied-depth 缩放。

### 2.4 无 loop-index embedding（仅 solver 有）

`step_index` 只进 `LocalSolverState` 的正弦编码（`model/local_solver_state_r7.py:126`），
tied cell 完全不知道现在是第几轮。

### 2.5 深监督权重方案与文献不同

我们 `linspace(1,2)` 递增、非最终占 73.3%。**但这条已测两轮并永久关闭**
（`record:s3-dense-step-supervision-result`、`record:s3-draft-supervision-organization-result`，
后者 final-only 也失败），**不再触碰**。

## 3. 外部证据分级（一手核对范围如实标注）

访问日期 **2026-11-25**。核到「摘要」的只按摘要写；核到章节标题/正文段落的另标。

| 论文 | arXiv | 核到层级 | 对我们有用的内容 |
| --- | --- | --- | --- |
| SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers | 2609.01343 | 摘要 + §2 引言 + §4.5 标题 | 同时匹配 per-token FLOPs、非嵌入参数、KV cache 三预算下，把**中间一半层循环两次**；省 6.8–18.0% 训练 FLOPs；§4.5 标题即「Two loops beat three or four」，并写「therefore fix loop count at 2」 |
| Towards Looped Models Done Right, Part I | 2610.23036 | 摘要 + §3.3/§3.2 段落 | 在匹配参数/逻辑深度/token 预算下隔离三轴；prelude–recurrent–coda 稳定改善；**input injection 主要提升知识检索与上下文利用**；随机初始化与分离 H/L 状态为 mixed/negative |
| DeepLoop: Depth Scaling for Looped Transformers | 2607.13491 | 摘要 | tied-depth 的一阶扰动界由 visit-alignment 系数 κ_R 控制；aligned 区间指数须从 1/4 提到 1/2，α=(2N)^{1/2}、β=(8N)^{-1/2}；**只在 recurrent depth 启用后**才有增益；架构是 **post-LN DeepNorm** |
| Ouro: Scaling Latent Reasoning via Looped Language Models | 2510.25741 | 摘要 | latent 空间迭代 + 熵正则的深度分配 + 7.7T token；优势来自**知识操作**而非知识容量 |
| Two-Scale Latent Dynamics（旁证） | 2509.23314 | 委派返回的摘要级 | 二阶（加速度）停机规则优于 KL/步长范数 |

**Tier 3（无受控消融，不投入）**：loop-index 正弦编码、per-loop LoRA、spectral norm、
zero-init 输出头。委派检索的结论是这些在 2024–2026 文献中**普遍省略或实测中性**。

**天气/PDE 领域的空白**：委派检索未找到把 looped / recurrent-depth transformer 用于天气或
PDE 代理的工作。**我们是空白区**——既没有可对标的先例，也没有可抄的数字。这既是机会也是风险：
任何收益都不能引用外部先例当预期。

## 4. 已关闭的相邻轴（这决定了哪些「结合方案」现在不能开）

| 轴 | 记录 / 决策 | 状态 | 本轮是否触碰 |
| --- | --- | --- | --- |
| K 4→8（同 cell 多跑几遍） | `record:s3-reasoning-depth` | unresolved（单 seed），**勿重跑** | **不** |
| backbone `depth` 4→8（容量轴） | `record:s3-capacity-depth-result`、决策 **0048** | **negative，整条容量轴永久关闭** | **不** |
| 宽度 192→256 | `record:s3-capacity-width-result` | negative，触发 0048 封顶 | **不** |
| 全局混合 / 感受野（H-C） | 决策 **0077** | **否决（not-admitted）** | **不** |
| 过程态初值条件化（H-A） | `record:s3-process-init-result` | negative | **不** |
| 深监督覆盖 / 草稿监督组织 | 两条 record | 两支**永久关闭** | **不** |
| 有类型证据 | `record:s3-typed-evidence-in-protocol-result` | negative（C−B 十三格显著变差） | **不** |

## 5. 为什么「K 4→8 无效」不能推出「loop 无用」

那是 **unresolved**（三 seed 只完成一个、reading 从未运行），且它测的是「同一个 16-token 状态
被多跑几遍」。而文献的 loop 收益来自两件我们**都没测过**的事：让循环改写全分辨率表征
（§2.1 的缺口），以及在 tied 深度下修正残差缩放（§2.3 的缺口）。把前者或后者当作「再加大 K」
的同义词，就是 `AGENTS.md` 明禁的「换名重开」。

## 6. 本页不回答的问题

1. **哪条差异更重要**——本页没有读数。判据与决策表在决策 0118 里冻结，读数在
   `docs/R7_S3_LOOP_DEPTH_HEADROOM_RESULT.md`。
2. **72 h 行为**——本页及其引用的诊断主判据是 **+6 h** 目标（`atmos_target` 是
   `physical_targets[:,0]`），不能外推成 72 h 技能。
3. **外部数字的可迁移性**——DeepLoop 是 post-LN、SMELT 是 MoE、Ouro 是 7.7T token 的语言模型；
   三者的具体增益**都不是我们的预期值**。
4. **是否值得做**——成本参见决策 0118 的 Consequences；本页不提供推荐。
