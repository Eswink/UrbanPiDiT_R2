# 0028 — 吞吐前置、学习曲线定标、物理约束与有界重开的组合 campaign

- **日期**：2026-11-04
- **状态**：定稿归档（本文件是不可变记录；执行源是 `docs/goals/s3-throughput-and-physics-campaign.md`）
- **起点**：分支 `r7/weather-reasoning`，HEAD `407d20e09152449380ae846bb0ae1d4c316195a6`（与 origin 同步，工作树干净）
- **范围**：S3 节点内的方法探索轴扩张 + 吞吐前置 + 最终配置选定
- **`scientific_claim: false`**；本文件**不含任何科学结论**，只含实测事实、设计约束与成本口径
- **上游**：计划 0025（W1/W2 已落地）、0026（R1–R5 队列）、决策 0048/0057/0061/0077/0078/0083

---

## §0 摘要：为什么必须转向

三条实测事实推翻了「S3 单因素空间已饱和」这一判断（campaign `:2029-2031` 的原文）。

### 0.1 本模型远未受算力限制

| 项 | 实测值 | 来源 |
| --- | --- | --- |
| 参数量 | **3,286,037**（155 张量） | `docs/R7_S3_FY_YEAR_COVERAGE_MATCHED_RESULT.md:401`（并更正早前「约 2,400 万」错了 7.3 倍，`:402-404`） |
| 单次前向 FLOPs | **13,904,603,520**（17ch / 65×65 / dim192 / batch2） | `outputs/r7_e0_diagnostic/rw_b_cost_batch2.json` |
| 单次 fwd+bwd FLOPs | **41,596,684,032**（同上） | 同上 |
| 一个 update 的构成 | **12 次前向 + 1 次全 BPTT**（`detach_physical_steps: false`，`detach_between_steps=False` 强制） | `training/r7_long_rollout.py:89-100`；`training/r7_autoregressive_rollout.py:112-116` |
| 一个 update 的 GPU 时间 | **<0.1 s**（12 × 41.6 GFLOP ≈ 0.5 TFLOP） | 由上两行直接推算 |
| **实测一个 update 的墙钟** | **2.4220 s**（control 均值）/ **2.5965 s**（treatment 均值） | R1 六臂 241,958.7 s / 96,426 updates；R2 独立复现 2.5839–2.6802 |

⇒ **95% 以上的时间不在 GPU 计算里。** 逐项定位（详见 §0.4）。这意味着「几百 epoch」在当前实现下不可付，但**修完可付**。

### 0.2 「mixed」会随算力翻转

| 臂 | 72 h t2m skill | 说明 |
| --- | --- | --- |
| 锚反馈旧臂（129×129，2400 upd） | **+0.000576**（三 seed **2/3 正**） | 按绝对门 2/3 永久关闭该族（`docs/R7_S3_ANCHOR_ISOLATION_RESULT.md:37-40`） |
| R2 锚反馈（65×65 多年，16071 upd） | **+0.264772 / +0.263564 / +0.284917** | 四出口**全真**，15/15 配对改善（`docs/R7_S3_ANCHOR_FEEDBACK_65X65_RESULT.md`） |
| R1 控制臂，updates 2,400 → 16,071 | +0.0258 → **+0.1527**（仅增加算力，数据不变） | `docs/R7_S3_FY_YEAR_COVERAGE_MATCHED_RESULT.md` |

⇒ **已登记的 22 条 negative / 23 条 mixed 里，有一批是在 ~1 epoch 的 regime 下判死的。** 这不等于它们会翻，但等于**它们没有被公平检验过**。

### 0.3 本模型并不「物理信息」

| 项 | 现状 | 位置 |
| --- | --- | --- |
| 空间一阶差分惩罚（唯一真正则） | **关闭**：`gradient_lambda = ZERO_GRADIENT_LAMBDA = 0.0`，且不在任何已登记协议里 | `training/r7_recursive_losses.py:18`；`:162-163` 走不到该分支 |
| 散度 / 涡度 / 平流算子 | **仅作前向输入**，无惩罚项 | `model/r7_process_tensor_diagnostics.py:133,139,145` |
| 8 向量过程诊断损失 | 权重 **0.0**（`"process_weight": 0.`） | `training/r7_long_rollout_runner.py:185` |
| 质量守恒 / 静力平衡 / 连续方程 / 谱损失 / CRPS | **全仓零命中** | 见探查结论 |
| 完整 `physical_consistency.py` | 存在，但**当前代码不 import** | `legacy_v531_full/losses/physical_consistency.py:248,280` |

⇒ 主目标的「物理信息」这一半**从未被实验过**，不是失败，是空白。

### 0.4 吞吐的四个真实瓶颈（逐项实测）

| # | 瓶颈 | 实测 | 位置 |
| --- | --- | --- | --- |
| 1 | 单线程 zarr 逐帧读，无 DataLoader / 无 worker / 无 pin_memory | 每样本 **0.53–0.68 s**；batch 2 合计 **1.1–1.3 s/update** | `data/r7_long_rollout_dataset.py:129-150`；`data/r7_zarr_dataset.py:47-79` |
| 2 | 数据放大 **12.4×** | 每样本读 **47.5 MiB** 只为 **3.84 MiB** 有用数据；202 次 `store.get`；同一 8 个 chunk 被取 6–8 次 | chunk `(16,17,64,64)` vs 逐时刻 `(17,65,65)`；65 的第 65 行列各占独立 chunk |
| 3 | checkpoint 全留 + 每 20 步 fsync+reload | 每臂 **804 个 × 40.97/55.11 MiB = 32.2/43.4 GiB**；`torch.load` 0.70–0.80 s；一轮六臂 **227 GiB** | `training/r7_long_rollout_runner.py:293-297` |
| 4 | 每步全模型 155 张量 `isfinite` 强制 GPU 同步 | 每 update 一次全设备同步 | `training/r7_autoregressive_runner.py:66` |

batch 标度表是决定性佐证（`outputs/r7_s3_batch_probe_20261024_attempt01/batch_probe.json`）：

| batch | s/update | 相对 batch1 | 峰值 reserved | 共驻门 |
| --- | --- | --- | --- | --- |
| 1 | 1.8385 | 1.000× | 2,322 MiB | 过（+13,335） |
| 2 | 2.3505 | 1.278× | 4,528 MiB | 过（+12,903） |
| 4 | 3.3602 | 1.828× | 8,950 MiB | 过（+8,481） |
| 8 | 5.6200 | 3.057× | 18,092 MiB | **不过（−661）** |

batch 1→2 只涨 27.8% 墙钟 ⇒ 存在约 **1.4–1.5 s/update 的固定串行成分**，与 17ch/65×65/dim192 的算术无关。

---

## §1 起点状态（可核查）

- `campaign-state`（`docs/goals/main-model-climatology-campaign.md:459`）：`used_gpu_h 401.377595`、`remaining_gpu_h −381.377595`、`current_round_goal docs/goals/s3-position-retention-reeval.md`、`current_node S3`、`next_node_proposal S4`
- 账本 **79 行**；证据索引 **98 条**；分布 negative 22 / mixed 23 / unresolved 6 / engineering-positive 13 / audit 34；`candidate_state: candidate` 仅 **2** 条（其中 `b2-multiseed-negative` 是 `outcome_class: negative` 的历史错配）
- **§7 合计行（`:389`）仍写 247.356641，与机算 401.377595 不符** —— 既有漂移，本轮由执行者修正（只改这一个字符串，不回写历史行）
- **R4**：协议冻结 `ee79b066…`、`code_commit b3619ff`、seed41 控制臂停在 **1,440/16,071**（1.156–1.172 h）、**无 `readings.json`、无出口判定、四件套未登记**；重开是**重跑**约 77 GPU-h（训练器 `"no failed-attempt resume is supported"`）
- **R2**：四出口全真、77.4371 GPU-h、`candidate_state: needs-review`、**无独立复核记录**、**未进 S4**
- **FY candidate（月份 4→12）**：四出口全真、`candidate_state: candidate`、**仍等用户验收是否进 S4**
- **S4 门**：决策 0066 判未达（90 格中 78/90 点估计 ≤0，仅 25/90 同时区间上界 ≤0）
- 磁盘：`/data` 余 **884 GiB**；两张 3090 均空闲（38 / 17 MiB）

---

## §2 六个 Stage

### Stage A — 吞吐前置（约 2 GPU-h，只测量不做训练）

**为什么先做**：不修，100 epoch 在 22 年清单上要 187 h/seed；修完目标 3–5×。

| 项 | 改什么 | 硬验收 |
| --- | --- | --- |
| **A1** 批量读窗口 | 把 14 个时刻合成一次跨步读；删掉 `targets[0]` 的重复读（同一 index 被读两次）；`normalization()` 每样本只算一次（现在算两次）；按 chunk id 加解码 LRU | `window_sha256` / `usable_windows` / `excluded_sample_ids` **逐字节不变**；固定 seed 前 200 步 loss **逐位相同** |
| **A2** 并行取数 | `DataLoader(num_workers=4, pin_memory=True, persistent_workers=True)` + `non_blocking`；**顺序推导留在主进程**（`randperm(seed+epoch)`），worker 只接 index | 同上逐位相同 |
| **A3** 同步守卫 | `_finite_state` 每步扫 155 张量 → 改为每步只检 loss/grad-norm（已在算），全模型检每 100 步 + 每 checkpoint | **需 ADR**：明确「每步全检降为周期全检」是安全换速率，写明负面后果 |
| **A4** checkpoint 策略 | 每 5000 步 + final + val 最优，预注册 val 选点规则 | **需 ADR**（0068 有「剪 checkpoint 致 92 处审计引用悬空」先例）：只对新 run 生效，登记保留清单 |
| **A5** 实测 | A1–A4 每个优化落地后，用 `scripts/r7_s3_batch_ceiling_probe.py` 重测 batch 4 与 batch 10 两档（A6 的基线见 §2.1） | 与 §2.1 的 8/10/12/14 逐档对比，证明优化确实降低了 s/update 与峰值 |
| **A6** batch 上限扫描 | **已执行完（2026-11-04）**，见 §2.1：实测上限 batch 14，共驻上限 batch 10，16 与 32 OOM。本轮无需重跑；**A1/A2 落地后须重新扫一次**，因为批量读会改变中间张量驻留 | 已交付：六份逐档 JSON + `summary.json`（`outputs/r7_s3_batch_ceiling_20261104_attempt01/`） |

**出口**：batch 4 后 **<1.0 s/update** ⇒ 继续；否则停下报告，**不改判据硬凑**。

**不做**：不重建 store。重建会改 `data_identity`，断掉与全部历史记录的可比性 —— 只改**读法**。

#### §2.1 batch 上限扫描（用户 2026-11-04 提出「试到 32–128」，已**真跑实测**）

**结论先说：本实例的真实上限是 batch 14，batch 16 与 32 实测 OOM；32–128 不可能。**
这不是拟合，是逐档起独立进程真跑出来的（`outputs/r7_s3_batch_ceiling_20261104_attempt01/summary.json`）。

**方法**（口径必须记住，否则数字对不上）：
- 每档一个**全新 Python 进程**。`torch.cuda.max_memory_reserved` 是全进程高水位，同进程连测会让后一档继承前一档的缓存池 ⇒ 只有新解释器才给出该档自己的峰值。
- 真实 store、真实 manifest（treatment 32,131 窗）、真实排除集（11 条，取自冻结协议）、真实 spec（K=4/depth=4/dim=192/12 步全 BPTT）。
- **数据构造在计时区间内**（`dataset[index]` × batch + `default_collate` + H2D），因为实测 40% 以上的时间就在那里；只计 GPU 步会低估约一半。
- 每档跑 2 个 update，单张 3090（24 GiB），邻居空闲。
- 参数 2,819,267（这是**未加 typed-evidence 的路径**；3,286,037 是加了 typed evididence 的臂，两者不可混用）。

**实测结果**：

| batch | 结果 | 峰值 reserved | s/update | samples/s | 相对 batch 2 |
| --- | --- | --- | --- | --- | --- |
| 8 | 跑通 | 12,918 MiB | 5.558 | 1.439 | 1.69× |
| 10 | 跑通 | 15,494 MiB | 6.776 | 1.476 | 1.73× |
| **12** | **跑通** | **19,270 MiB** | 7.937 | 1.512 | 1.78× |
| **14** | **跑通（最大）** | **22,404 MiB** | 9.258 | **1.512** | **1.78×（饱和）** |
| 16 | **OOM** | — | — | — | — |
| 32 | **OOM** | — | — | — | — |

OOM 原文（如实记）：batch 32 在「Tried to allocate 52.00 MiB」时已占 **23.49 GiB / 23.56 GiB**；batch 16 在「Tried to allocate 14.00 MiB」时已占 **23.51 GiB**。⇒ 卡是被真占满后才失败的，不是被守卫提前拒绝。

**每 batch 单位约 1,550–1,615 MiB**（8/10/12/14 四档一致），与 batch 4 的历史实测（8,950 MiB ⇒ 2,238 MiB/unit，那是**另一套探针口径**）不同 —— 两套口径不可混算，本表只认本轮的实测。

**三条由实测得出的结论，替代此前所有拟合**：

1. **上限是 14，不是 7 也不是 10**。此前的线性拟合（2,254.6 MiB/batch）**高估了每单位开销**，因此**低估了上限**；这是拟合被实测推翻的一次记录。
2. **吞吐在 batch 12–14 就饱和**：samples/s 从 batch 8 的 1.439 只升到 1.512（+5%），而峰值从 12.9 GiB 涨到 22.4 GiB（+74%）。**再往上加 batch 只买显存不买速度**。
3. **共驻门槛会把 14 挡在门外**：现行共驻要求 `peak + 512 MiB margin + 6,144 MiB 邻居预留 ≤ 24,087 MiB`，batch 14 需 22,404 + 512 + 6,144 = **29,060 MiB > 24,087**，**过不了**。按共驻能落地的上限是 **batch 10**（15,494 + 512 + 6,144 = 22,150 ≤ 24,087）；batch 12（19,270 + 6,656 = 25,926）已经超。**要用 12–14 必须独占（去掉邻居预留），需具名授权**。

**所以 A6 的最终形态**（已执行，成本实测约 0.6 GPU-h，含两次 OOM 的失败尝试）：

| 档 | 结果 | 可用性 |
| --- | --- | --- |
| 8 | 跑通 | 共驻可用 |
| 10 | 跑通 | **共驻上限** |
| 12 / 14 | 跑通 | 需独占授权 |
| 16 / 32 | OOM | 不可用；128 未测且必然更差 |

**不再测 64 / 128**：16 已 OOM 且在 23.51 GiB 处才失败，按 ~1,600 MiB/unit 外推 128 需约 200 GiB，而这是**已被 16 的实测否证方向**，再花 GPU-h 没有信息量。

**必须写进 ADR 的负面后果**：大 batch 会减少**每个 epoch 的 update 数**（1 epoch = ⌈窗数/batch⌉）。固定 epoch 预算下这等于**降低总更新数**，而 R1 已实测「updates 2,400→16,071 让 72h skill 从 +0.0258 到 +0.1527」。所以大 batch 若同时降低 updates，可能**净损失**。因此 Stage B 的学习曲线必须以 **updates（而非 epoch）** 为主轴登记。

**优先于大 batch 的替代路径**：`activation_checkpointing` 已存在（`model/recursive_weather_r7.py:264,338,382-384`，默认 `False`，走 `torch.utils.checkpoint`），可用约 30% 计算换数倍激活显存，且不改 update 语义、不改 `model_code_sha256`。若要「更大的有效 batch」，先试它。这条要自己的 ADR。

### Stage B — 学习曲线（本 campaign 最关键的单个实验，约 43 GPU-h）

**问法**：72 h skill 随训练量是**饱和**还是继续涨？这决定「几百 epoch」是否值得。

- 单臂、单 seed、treatment 清单（32,131 窗）、**batch 4（固定，不由 A6 的上限驱动）**、基础配方（K=4/depth=4/dim=192/LR 2e-5/warmup 10）
- **为什么 Stage B 不用 batch 10/14**：batch 放大到 10 能把 samples/s 从 1.19 提到 1.476（+24%），但 1 epoch 的 update 数会从 8,033 掉到 3,213。R1 已实测 updates 而非 samples 是 skill 的主因（2,400→16,071 让 72h skill +0.0258→+0.1527），所以在**定标**阶段改变 batch 会污染「训练量→skill」这条曲线本身。batch 只允许在 Stage F 的最终长跑里作为**成本选项**出现，且那时必须同时固定 updates 口径以便与阶段 B 对齐
- 阶梯：**0.25 / 1 / 4 / 16 epoch** = 2,008 / 8,033 / 32,132 / 128,528 updates
- 每点全 17 变量 × 5 lead 评分
- 成本（按 Stage A 后 0.84 s/update 的**估计**，须以实测替换）：0.47 + 1.87 + 7.50 + 30.0 = **39.8 h** + 四点评分 ≈ 3.6 h ⇒ **约 43 GPU-h**；四点独立，双卡墙钟约 **30–32 h**

**预声明判据（冻结后不放宽）**：
- 若 16 epoch 相对 4 epoch 的 72 h skill 一阶差分 **> 0.01** ⇒ 预算未饱和，最终 run 取 16–64 epoch
- 若 **< 0.005** 且低于上一档 ⇒ 已饱和，最终 run 取饱和点
- 若介于两者 ⇒ 只记录，不加点、不外推；**不得**用「再多跑一点就好了」这类不可证伪结论

**注意**：无断点续跑，四个点必须各自独立跑完；四个点各自一份 `readings.json`。

### Stage C — 物理约束损失（2 轮，全新轴，约 64 GPU-h/轮）

当前代码零物理约束，这是主目标「物理信息」的正面缺口。**分两轮**避免多因子。

**C1 — 无目标约束束**（不读取目标，只惩罚预测自身的物理不合法）
- 散度/质量一致性、比湿非负、露点 ≤ T（对齐 legacy 的 `w_wind_div: 0.1`、`w_dew_leq_t: 0.5` 口径，但**在新模块重写，不 import legacy**）
- 唯一因子 = 约束束 ON/OFF；束内成分逐项记录，**若赢必须预注册后续分离实验，不得把束当结论**

**C2 — 导数域监督**（读取目标，监督预测的导数场）
- penalize `|∇·(pred) − ∇·(target)|²` + 涡度同款
- 复用 `model/r7_process_tensor_diagnostics.py:133,139` 的 `spherical_divergence` / `spherical_vorticity`（复用算子，不复用 legacy 损失）

**两轮都必须过计划 0025 §4.4 全套**：新 ADR（四段齐全、Consequences 含负面后果）→ 新 brief 带 (a)–(d) + 桥 (e) → 七 gate → 冻 `protocol.json` → 跑 → 四件套登记。

**ADR 必须论证与已关闭的「空间梯度损失」不是同一族**（`docs/R7_S3_SPATIAL_GRADIENT_LOSS_RESULT.md`）：那条惩罚预测自身的梯度（无目标的平滑先验），C1 是无目标**约束**、C2 是**导数域的目标匹配**。论证不成立则不得开轮。

预算口径：5 年清单 × 25 epoch = 45,575 updates ≈ 10.6 h/arm；2 臂 × 3 seed ⇒ **约 64 GPU-h/轮**。

### Stage D — 静态协变量接线 + 白盒化（约 1 GPU-h + 0）

**D1 接线**（最大未用资产）
- 现状：13 层已下载发布（14,484,496,452 B，`static_identity 8671ed7c…`，attempt04 重派生 `063fb805…`）；`model/static_conditioning_r7.py:79-100` 有身份校验与开关；但 `training/` **零引用**、不在任何已登记协议里、ADR 0069 仍是 `proposed`
- 要做：把 `static_identity` 绑进 `protocol_sha256`（**不是** `data_identity`）；打通 `static_conditioning` 到 `training/` 与协议构建器；修「接线报告称四出口已通过但 ADR 仍 `proposed`」的**文档漂移**
- ⚠️ 这会改 `model_code_sha256` ⇒ 所有既有迁移父失效，**必须重做父迁移**（0075/0083 已有四次先例）

**D2 白盒化探针**（CPU-only，0 GPU-h）
- P 态 8 个 anchored 槽位与 8 个诊断目标 1:1（`model/process_forecast_r7.py`）⇒ 测各槽位与散度/涡度/水汽辐合的相关系数、消融敏感度
- Z 态逐位置求解进度 ⇒ 与地形/海陆/土地覆盖的相关性（静态协变量正好在此接入）
- 产出可解释性证据页，服务「白盒化」这一拓展目标

### Stage E — 有界重开（2–3 轮，**以 Stage B 为门**）

仅当 Stage B 判定「未饱和」才启动 —— 否则会把欠训练误读成方法无效，正是本轮要避免的错误。

| 轮 | 轴 | 依据 | 约束 |
| --- | --- | --- | --- |
| **E1** | R5 训练信号族 | 0057 已授权、从未开轮（`docs/plans/0026:97`） | 只取两个最有信号的成员（dense 12 步、final-only 草稿监督），在 Stage B 定出的预算下跑 |
| **E2** | 容量轴重开（depth 或 width） | 模型仅 3.29M 参数，容量在 ~1 epoch 用绝对门判死，与 R2 同病灶 | **0048 无重开条款 ⇒ 需新 ADR 明确取代 0048 + 用户具名授权**（参照 0061 先例，但**不得**沿用其「仅限该轴」的自我限制） |
| **E3** | H-C 全局混合 | 0077 唯一允许的形态 | **同网格改操纵强度**（低秩 vs 全 N×N），算力与参数量分别声明；**不得称「H-C 的重估」** |

**R4 位置保留不在 E 列**：协议已冻结、停在 1,440/16,071，重开是**重跑**约 77 GPU-h，属**用户决定**（不自动续）。

单轮成本按 Stage C 口径 ≈ 64 GPU-h；三轮上限 ≈ 190 GPU-h。

### Stage F — 最终配置菜单（用户手选）

**batch 在这里才作为选项出现。** Stage B 固定 batch 4（见其理由），Stage F 才允许改 batch，
且改了 batch 必须同时按 **updates** 折算，否则与 Stage B 的曲线不可比。

**实测成本口径（已替换全部估计）**：
- 数字来自 §2.1 的真跑：batch 4 未单测，按其相邻档（batch 8 = 5.56 s/update）与历史 R1 的
  2.42–2.63 s/update @ batch 2 插值，**batch 4 ≈ 3.36 s/update 且数据构造在内**；这是唯一还带
  插值的数字，**A5 落地后必须用实测替换**。
- 其余档位全部是实测：batch 8 = 5.558、batch 10 = 6.776、batch 12 = 7.937、batch 14 = 9.258 s/update。

epoch → updates 换算（**分母随 batch 变**）：
- 5 年清单 7,291 窗：batch 4 ⇒ 1 epoch = **1,823** updates；batch 10 ⇒ 730；batch 14 ⇒ 521
- 22 年清单 32,131 窗：batch 4 ⇒ 1 epoch = **8,033** updates；batch 10 ⇒ 3,214；batch 14 ⇒ 2,295

| # | 清单 | 架构 | 物理 | 静态 | epoch | batch | 共驻可行 | 单 seed | 3 seed（双卡墙钟） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | 5 年 | 基础 2.82M | 关 | 关 | 100 | 4 | 是 | 42.5 h | **64 h** |
| C2 | 22 年 | 基础 | 关 | 关 | 25 | 4 | 是 | 46.9 h | **70 h** |
| C3 | 5 年 | 基础 | C1 束 | 关 | 100 | 4 | 是 | ~43 h | **~65 h** |
| C4 | 22 年 | 基础 | C2 导数域 | 关 | 25 | 4 | 是 | ~48 h | **~72 h** |
| C5 | 5 年 | depth8 或 dim256 | 关 | 关 | 50 | 4 | 是 | ~32 h | **~48 h** |
| C6 | 22 年 | 加宽 | C1 束 | **开** | 25 | 4 | 是 | ~95 h | **~143 h** |
| C7 | 5 年 | 基础 | C1 束 | 开 | 100 | 4 | 是 | ~45 h | **~68 h** |
| C8 | 5 年 | 基础 | C1 束 | 开 | 100 | 10 | 是（共驻上限） | ~34 h（samples/s 更高） | **~52 h** |
| C9 | 22 年 | 加宽 | C1+C2 | 开 | 25 | **14** | **否 ⇒ 需独占授权** | ~76 h | **~115 h** |

**菜单旁必须写明的四条硬约束**：
1. **22 年清单上跑 100 epoch 不可行**：batch 4 时 187 h/seed × 3 = **562 GPU-h** ≈ 11.7 天单卡。epoch 数必须与清单大小配对。
2. **共驻上限是 batch 10**（实测：batch 10 需 15,494+512+6,144 = 22,150 ≤ 24,087 MiB；batch 12 需 25,926 已超）。batch 12–14 只有 C9 这类标注「需独占授权」的项才允许，且授权必须具名。
3. **batch 越大，1 epoch 的 updates 越少**。C8 虽然 samples/s 比 C1 高 24%，但固定 100 epoch 时它的总 updates 只有 C1 的 40%。**若你要的是「训练量」，应固定 updates 而不是固定 epoch** —— 这一条必须在选定配置时明确，否则 C8 与 C1 不可比。
4. **磁盘**：checkpoint 按 A4 稀疏化后每臂 <8 GiB；不稀疏则 100 epoch 会写出 **~1.3 TB**（当前余 884 GiB）。

## §3 预算与停止

| Stage | GPU-h（估） | 门 |
| --- | --- | --- |
| A | ~2 | 无（纯测量） |
| B | ~43 | 76 h 硬上限 |
| C | 64 × 2 | 每轮 144 h 硬上限 |
| D | ~1 | 48 h 硬上限 |
| E | 64 × ≤3 | 每轮 144 h 硬上限 |
| F | 见菜单 | 用户选定后另立 |

**停止条件（三条，与 campaign §0 一致）**：触保留授权边界（付费/租卡/独占/main/方向改变）；达已冻结科学停止条件（某出口判定完成）；确无可安全独立推进的工作。停时必须在 §6 写「为何停 + 恢复第一条命令」。

**不设总 GPU-h 上限**（决策 0030/0038）；`cap`/`remaining` 仅会计。

---

## §4 风险与负面后果（R-033：ADR 必须含负面后果，此处先列）

1. **吞吐优化可能改掉一位**。A1/A2 若让 loss 序列不再逐位相同，本轮全部历史读数的可比性受损。缓解：A1/A2 的验收判据是**逐位相同**，不通过就回退不合并。
2. **A3 削弱逐样本安全性**。把每步全模型 `isfinite` 降为周期检查，理论上会晚 100 步发现 NaN。这是**明知的交易**，必须写进 ADR 的负面后果，并在每 checkpoint 处保留全检。
3. **A4 剪 checkpoint 会断审计链**。0068 已记录「剪除中间 checkpoint ⇒ 92 处历史审计引用悬空」。缓解：只对新 run 生效 + 预注册保留清单。
4. **Stage B 可能证明「饱和」**，从而否定「几百 epoch」这一原始设想。这是**有效结果**，必须如实登记为 negative，不得因不符合预期而改判据或改点。
5. **物理约束可能无效**。legacy 的 `w_wind_div 0.1` 从未在本仓被独立验证；若 C1/C2 四出口不全真，按预声明永久关闭该支，不得扫 λ。
6. **重开容量轴会削弱「预声明出口不可翻案」纪律**。0061 已自我记录过同一代价。E2 的 ADR 必须写明这一点，并把先例范围限定在容量轴。
7. **成本可能翻倍**。Stage A 若达不到 1.0 s/update，菜单里所有数字都要乘以实际倍数，F 的可行性随之变化。

---

## §5 明确不做

- **不重建 65×65 store**（会改 `data_identity`，断掉与全部 98 条已登记记录的可比性）
- **不新建 129×129 多年源**（用户已选 65×65；宽域是后续独立决定）
- 不动 `data/raw|interim|processed`、`data/legacy_v531/`、`model/legacy_v531*`、`legacy_v6/`
- **不读 2023 test**（`r` 保持 0，`test_read: false`）；S4 未启动不变
- 不重开**方向明确的负结果**：SAM / zero-base decoder / plain EMA / 长 lead 权重 / rollout 剂量（0057 §Decision 5 明文）
- 不改任何已冻结判据；不用「换名重开」；不为报告好看删改测试
- **不自宣** R2 通过独立复核、不自宣 FY candidate 进 S4 —— 两者仍是用户/复核者的决定
- 不 paid / rental / exclusive / cron / 守护 / 会话外续跑
- 不对邻居进程发任何信号；GPU 共驻按 R-054 只读核余量

---

## §6 交付物

| # | 交付物 | 证据形态 |
| --- | --- | --- |
| D1 | 方向扩围 ADR | `docs/decisions/0084-<slug>.md`（四段齐全、Consequences 含负面后果）+ README 索引行 |
| D2 | 吞吐 ADR ×2 | A3（同步守卫）、A4（checkpoint 策略）各一份 |
| D3 | Stage A 证据页 | 四档 `seconds_per_update` + 逐位相同证明 + 回退决策 |
| D4 | Stage B 学习曲线 | 四点 `readings.json` + 饱和判定（由驱动算，不手填） |
| D5 | Stage C 两轮 | 各一份 ADR + brief + 冻结协议 + 逐 seed receipt + 结果页 |
| D6 | Stage D 接线与探针 | `static_identity` 绑进 `protocol_sha256` 的协议 + 可解释性证据页 |
| D7 | Stage E 各轮 | 各一份 ADR（E2 须显式取代 0048）+ brief + 结果页 |
| D8 | 四件套登记 | index record / 账本行 / campaign §8 / campaign-state |

---

## §7 执行入口

执行源是 `docs/goals/s3-throughput-and-physics-campaign.md`；可粘贴的 objective 见该文件 §0，
本文件 §8 原样收录同一份文本以备审计。

**两个已知坑（都实测过，必须写进执行者的提示词）**：
1. 桥 (e) 不能要求「零 query 聚合逐位等于 `mean(dim=1)`」—— 数学上不可达（实测差 2.98e-08），会在第一个 seed 直接 `raise` 中止整轮（ADR 0082）
2. `code.zip` 必须带 commit 注释，否则 `stage archive` 拒绝启动（`b3619ff`）

**第三个坑**：跑任何 stage 前先 `tools/r7_round.py state` 确认 `current_round_goal` 指向本轮 brief，
否则会冻结错轮次（R4 期间真实发生过一次：把 R2 协议冻进了 R4 的 attempt 目录）。

---

## §8 可粘贴的 objective（与 goal brief §0 同文）

见 `docs/goals/s3-throughput-and-physics-campaign.md` §0 —— 该处是唯一维护点，本文件不复制第二份，
避免两处漂移（R-034 的单一来源原则）。

---

## 实际结果（Actual Results）

**NOT EXECUTED YET** —— 本计划在实现之前归档，尚无属于它的实验读数。

- Stage A–F 的**训练与科学读数**全部未执行；`protocol.json`、GPU-h、逐变量表格都不存在；
  **不得被任何下游文档引用为结果**。
- **例外（已执行，纯工程测量，无科学含义）**：`A6` batch 上限扫描已于 2026-11-04 真跑完成，
  六份逐档记录在 `outputs/r7_s3_batch_ceiling_20261104_attempt01/`，实测上限 batch 14、
  共驻上限 batch 10、16 与 32 OOM。该测量推翻了本文件早前一版写入的线性拟合（2,254.6 MiB/单位
  ⇒ 预测上限 7 且 12 应 OOM），真值为约 1,600 MiB/单位。**这是本计划内唯一被实测修正过的判断**，
  如实留痕以免下游继续引用那个错值。
- 本计划**不含**任何科学结论：`scientific_claim: false`。
- 本次 Plan Mode **实际做过**的事（与计划正文区分）：
  - 只读探查：`docs/goals/main-model-climatology-campaign.md` 的 `campaign-state` 与 §7/§8、
    `docs/R7_EVIDENCE_INDEX.jsonl` 全 98 条、`docs/decisions/` 0048–0083、
    `docs/plans/0025`/`0026`、`docs/R7_S3_POSITION_RETENTION_HANDOFF_20261103.md`、
    `docs/R7_S3_ANCHOR_FEEDBACK_65X65_RESULT.md`、`docs/R7_S3_FY_YEAR_COVERAGE_MATCHED_RESULT.md`
  - 只读实测：六条已完成臂的 `progress.jsonl` 与 receipt 时间（2.4220/2.5965 s/update）、
    `outputs/r7_s3_batch_probe_20261024_attempt01/batch_probe.json` 四档标度、
    checkpoint 数量与体积（804 × 40.97/55.11 MiB）、zarr chunk 形状 `(16,17,64,64)`、
    `outputs/r7_e0_diagnostic/rw_b_cost_batch2.json` 的 FLOPs、参数量 3,286,037
  - 代码只读核对：`training/r7_long_rollout.py:89-100`（12 步 BPTT）、
    `data/r7_long_rollout_dataset.py:129-150`（逐帧 zarr 读、norm 算两次）、
    `training/r7_autoregressive_runner.py:66`（155 张量每步同步）、
    `training/r7_recursive_losses.py:18`（`gradient_lambda` 默认 0.0）、
    `model/static_conditioning_r7.py:79-100`（开关存在但 `training/` 零引用）
  - 产出：本计划 + `docs/goals/s3-throughput-and-physics-campaign.md`
  - 验证：`check_conventions.py` 37 条阻断 0 失败；`check_goal_brief.py --brief`
    `failures=0 advisories=0`；`git diff --cached --check` 干净
  - **未做**：未修改 `model/**`、`training/**`、`data/**`、`scripts/**` 的行为；
    未训练、未租 GPU、未下载数据、未读封存 test、未关闭 issue、未合并 main、
    未 force push、未创建定时任务、未自动进入 Goal Mode
