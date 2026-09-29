# #72 M2-B 第五轮：RW-B 的减法归因（留一对照与其负控制缺陷）

**状态：本轮执行完毕；判决是「不能归因」，原因在负控制本身，不在两臂读数。**
本文件是这一轮唯一的证据页，**不覆盖** `docs/R7_72_RW_B_PILOT.md` 及其之前各轮：那些文档仍是各自唯一的
证据，`protocol_sha256` 与 `model_code_sha256` 都不同，**数字不可相加、不可并列**。

## 0. 一句话结论

两个子开关的拆分是**工程上成立的**（两条逐位等价都有实跑证据）；0 GPU-h 的机制探针在两 seed 上一致地
把「×1.8 幅度」定位到**门控+锚定提案这条通路**；但本轮预注册的**负控制臂 `RW-B−(a)` 在构造上就是
RW-A 本身**（相同权重下输出逐位相同），因此它的 0.00003–0.00008 K 级别的「worsened」是浮点噪声而
不是混杂——按冻结的分支规则，本轮**不做归因**，并把这条设计缺陷如实记为下一轮的可反驳假设。

| 项 | 值 |
| --- | --- |
| 起点 SHA | `e6085bc8a8ab173f6208ed210bf8484f323a56a4`（开工 `git rev-parse HEAD`） |
| 授权 | 决策 0021 的会话内通道；用户在 `AskUserQuestion` 选择「按上述范围跑」（2 seed × 4 臂 × 400 updates，≤1.0 GPU-h） |
| 数据 | M2 双月段 `outputs/r7_m2_segment/store/manifests`：train 186 / val 22 / test 26（**test 全程未读**），`data_identity ef8c6691…` |
| 协议 digest | `58fc74b7a7aaa513197d85f836684cd55851013b3c7f8519f184649837b357d4`（8 个 run **全同**） |
| `model_code_sha256` | `11090929930da4e1259698699cbbf12b3738cdfb2f609c3af516c24399144476`（本轮新值） |
| 实测 GPU | **0.4128 GPU-h**（训练 1353.5 s + 评估 132.5 s，取自产物 `elapsed_seconds` 求和） |
| 本轮自设上限 | ≤1.0 GPU-h（实测 0.4128，**未超**）；第二批账本 20.701 → **20.288** |
| D1 探针 | **0 GPU-h**（CPU 6.2 s），只读上一轮 32 个 checkpoint 与 val |
| 分支判定 | `stop-confounded-control`（冻结规则照写触发）；诊断：负控制**退化**而非混杂 |
| 归因 | **未做**（按 §3 规则第 1 条停下报告） |

---

## 1. D1：0 GPU-h 机制探针（只读上一轮 32 个 checkpoint）

**驱动**：`scripts/study_r7_rw_b_subtraction_probe.py`（新增）。
**产物**：`outputs/r7_rw_b_subtraction_probe/probe.json`（sha256 `d56cb8b4effd803b5569cfa6…`）。
**命令**：`python scripts/study_r7_rw_b_subtraction_probe.py --code-root . --output outputs/r7_rw_b_subtraction_probe/probe.json --windows 8 --seeds 41,42`。

它只做三件事，全部是对**已归档 checkpoint** 的推理期改造，`model/` 一个字节都不动：

- **`recurrence_removed`（= (b) 关）**：把 checkpoint 自己的 `solver_cell` 换成一个**已注册的恒等
  `nn.Module`**，于是 `Z` 停在 `solver_init`，而锚定提案与逐位置门控**照常施加**。
- **`gate_proposal_removed`（= (a) 关）**：把同一个 checkpoint 的**共享张量**装进同一个类、
  但 `local_solver_state=False` 的构造里，于是该步走 pre-RW-B 的 correction 通路。
- **role 标记（(c)）**：焦点臂自身 `source_role_markers=False`、不带 role 张量，探针**如实记为不适用**
  并指向上一轮已登记的对照（0.0057–0.0282 K、方向不一致），**不重跑、不写成机制声明**。

**守卫**：`--code-root` 必须解析到训练那些 checkpoint 的修订；探针会核对
`running_model_code_sha256 == 归档 model_code_sha256` 才继续，并打印全部 32 个 checkpoint 的 sha256。
读数只用轮内已有的三个量（第 1 步修正幅度、逐步误差—修正余弦、第 3 步恶化比例），
**未新增任何阈值**（`thresholds_added: 0`）。

### 1.1 读数（两 seed；`|d|` 为归一化单位）

| 臂 | seed | 步1 \|d\| | 步1 cos | 步2 cos | 步3 cos | 步3 恶化 |
| --- | --- | --- | --- | --- | --- | --- |
| RW-A | 41 | 0.03587 | −0.233 | −0.149 | −0.078 | 0.412 |
| RW-B | 41 | **0.06649** | −0.137 | **+0.012** | **+0.070** | 0.610 |
| RW-B−(b) 递推移除 | 41 | 0.05324 | **+0.010** | **+0.137** | **+0.230** | 0.809 |
| RW-B−(a) 门控提案移除 | 41 | **0.00704** | −0.004 | +0.017 | +0.040 | 0.647 |
| RW-A | 42 | 0.02969 | −0.203 | −0.094 | −0.007 | 0.500 |
| RW-B | 42 | **0.05428** | −0.109 | **+0.013** | **+0.057** | 0.566 |
| RW-B−(b) 递推移除 | 42 | 0.09012 | **+0.021** | **+0.229** | **+0.343** | 0.890 |
| RW-B−(a) 门控提案移除 | 42 | **0.00700** | −0.017 | +0.007 | +0.034 | 0.654 |

### 1.2 探针能说什么、不能说什么

**能说的（两 seed 同向）**：

1. **×1.8 的幅度来自 (a) 那条通路**：关掉门控+锚定提案后，第 1 步幅度塌到 RW-A 的 **0.20/0.24 倍**
   （0.0070）。这是两 seed 同向的、幅度上一个数量级的读数。
2. **关掉递推后余弦从第 1 步就转正**（+0.010/+0.021；RW-A 与 RW-B 在第 1 步都是负的），
   即**修正一开始就指向误差同侧**；但幅度**不**回到 ×1.8，而是在两 seed 间从 1.48 跳到 3.04——
   幅度读数在 seed 间不稳。
3. 因此「×1.8 与余弦转正由同一件带来」这句话**不成立**：幅度需要 (a) 通路存在，
   余弦转正则出现在**任何**让 Z 停止跨步递推的配置里。

**不能说的（重要）**：`gate_proposal_removed` 这一行把 checkpoint 的共享权重送进 `correction_head`，
而**该臂训练时 `correction_head` 从未收到过梯度**。探针把这件事**测出来并写进产物**：
两个 seed 上 `correction_head_moved_by_training = False`（`tensors_moved_by_training` 137 个张量里
一个 correction_head 张量都没有）。所以这一行是「一个未被训练的修正头 + RW-B 的其余权重」，
**不能**当成「RW-B 去掉 (a) 之后的样子」。这正是本轮 D4 必须**从零重训**留一臂、
而不能拿 D1 这一行当结论的原因。

**归档一致性**：CPU 重跑对上一轮 CUDA 归档读数，`max |Δ|` 为 6.7e-06（seed41）与 7.4e-03（seed42 RW-A）。
后者是**一个窗口的布尔标记翻面**：归档 0.3529 对本次 0.3603（即 1/17 窗口差一格），
数值（`|d|`、余弦）差异仍在 1e-05 量级。这是 CPU/CUDA 归约顺序与边界标记的差别，**不是**用错 checkpoint
（每个 checkpoint 的 sha256 都已在产物里逐一对上）。

---

## 2. D2/D3：两个子开关与定向测试

**代码**（`model/process_forecast_r7.py`、`model/process_step_r7.py`）：

| 开关 | 默认 | 语义（写进 docstring） |
| --- | --- | --- |
| `solver_state_recurrence` | **True** | 关 ⇒ `Z` 不跨步携带：停在 `solver_init`，提案与门控**仍施加**，且**不**把状态传给下一步 |
| `solver_gate_proposal` | **True** | 关 ⇒ 该步走 pre-RW-B 的 correction 通路；递推若仍开则 `Z` **照常前进**但只是闲置计算，**不**悄悄换成别的东西 |

两个开关默认开、且**只在 `local_solver_state=True` 时存在**：把任一子开关关掉而整机制关着，
构造时**直接报错**（沿用既有「静默忽略的开关是配置错误」的风格）。两开关**不新增任何参数**。

**测试**（`tests/test_r7_rw_b_subtraction.py`，15 项）：

- **两条逐位等价（本条判据的核心）**，用冻结修订树 + 双包名导入，逐字节比较、零容差：
  - `local_solver_state=False` ≡ 起点修订 `e6085bc` 的实现；
  - `local_solver_state=True` 且两子开关默认 ⇒ ≡ 起点修订的 RW-B。
  - **反证**：把冻结树里的 `GATE_INITIAL_PROBABILITY` 从 0.25 改成 0.30，等价必须被打破。
- **组合矩阵**：三项输出互异 + **一处按冻结语义必须发生的坍缩**——
  `(递推开, 门控关)` 必须与 `(递推关, 门控关)` **逐位相同**（输出不读 Z），
  而唯一区别是前者**仍返回**状态。这条断言方式本身就是「按声明语义检查」而不是「断言四项都不同」。
- 门控与提案**梯度非零**；递推关 ⇒ 状态返回 `None` 且输出确实改变（带开/关镜像反证）；
  门控关 ⇒ 输出走 pre-RW-B 通道但状态仍前进；三路（forward/streamed/adaptive）等价；
  checkpoint 往返带上开关；BF16 streamed 反向四种组合全部有限。

`pytest -q`：本机 **1499 passed, 3 skipped**（skip 全为 `test_real_data_pipeline` 的三条：
可选真实数据 fixture 不在版本控制内，**不允许**用合成数据兜底）。本轮新增 15 项、
无删除、无弱化；R-009 基线 885/2180 → **900/2210**。

---

## 3. D4：一轮有界留一对照

**驱动**：`scripts/study_r7_72_rw_b_subtraction.py` + 冻结协议模块 `training/r7_rw_b_subtraction_protocol.py`。
**产物**：`outputs/r7_72_rw_b_subtraction/`（`protocol.json`、逐 seed `seed_result.json`、
`merged_result.json`、`paired_comparison.json`、六张表）。

四臂，同一协议族、同一 M2 store、全部从零训练、全部打开时空条件通路：

| 臂 | 开关 | 角色 |
| --- | --- | --- |
| `process_spacetime_rwa` | `positional_process_readout` | **RW-A**：比较的基准 |
| `process_local_solver` | `+ local_solver_state` | **RW-B**：两子开关默认 |
| `process_local_solver_no_gate_proposal` | `+ local_solver_state, solver_gate_proposal=False` | **RW-B−(a)**：登记的**负控制** |
| `process_local_solver_no_recurrence` | `+ local_solver_state, solver_state_recurrence=False` | **RW-B−(b)**：登记的**主问句** |

协议在每 seed **第一次 `optimizer.step()` 之前**以 `'x'` 排他写入、写完回读并重算 digest，
再核对 primary 登记项与脚本常量一致；8 个 run 的 `protocol_sha256` **全部相同**；
`test_read: false` 写在协议与逐 seed 结果两处；四臂**均未早停**，全部选中 update 400。

**臂配对是实测**：四臂同 seed 构造、共享张量逐位相同；RW-A 的 `state_dict` 经 trainer 自己的
transfer 规则载入其余三臂，`applied=131 / ignored=0`，载入后逐张量 `torch.equal`。

**实测代价**（协议冻结前测量，`arm_table.csv` / `memory_table.csv` / `training_table.csv`）：

| 臂 | 参数量 | 前向 FLOPs | 前向+反向 FLOPs | 训练峰值 allocated | 墙钟（2 seed 合计） |
| --- | --- | --- | --- | --- | --- |
| RW-A | 2,968,259 | 13,904,603,520 | 41,596,684,032 | 226.4 MiB | 339.6 s |
| RW-B | 3,283,157（+10.61%） | 16,820,126,592（×1.2097） | 54,656,320,512 | 254.3 MiB | 356.6 s |
| RW-B−(a) | 3,283,157 | 16,817,617,536（×1.2095） | 44,509,698,048 | 272.9 MiB | 342.5 s |
| RW-B−(b) | 3,283,157 | 13,907,112,576（×1.0002） | 35,198,999,040 | 232.2 MiB | 312.4 s |

`RW-B−(b)` 的前向 FLOPs 与 RW-A **几乎相同**（×1.0002），因为 `Z` 仍被算出来但不进输出；
`RW-B−(a)` 也几乎相同（×1.2095 略低于 RW-B 的 ×1.2097）。**四臂不是等容量的**，
凡涉及 RW-A 的对都是容量混杂的。

## 4. D5：预声明单一对比的读数与分支

比较器：既有 #60 比较器（`training/r7_coreasoning_compare`），depth 0，逐 seed 同号三态。
**未新增阈值、未新增端点、未改案例集**。用例数 22/21/19/15/11（6/12/24/48/72h）。

### 4.1 三对读数（t2m，K）

| 时效 | **RW-B−(a) − RW-A**（负控制） | **RW-B−(b) − RW-A**（主问句） | RW-B − RW-A（本轮参照） |
| --- | --- | --- | --- |
| 6h | −0.000006 / −0.000011 → supported | +0.1551 / +0.1453 → **worsened** | +0.1986 / +0.0372 → worsened |
| 12h | +0.000006 / −0.000022 → unresolved | −0.1735 / −0.2197 → supported | −0.0436 / −0.1147 → supported |
| 24h | +0.000046 / +0.000012 → **worsened** | −0.1850 / −0.3167 → supported | −0.0220 / −0.1273 → supported |
| 48h | +0.000057 / +0.000030 → **worsened** | +0.8344 / +0.7363 → **worsened** | +1.0617 / +1.0685 → worsened |
| 72h | +0.000027 / +0.000078 → **worsened** | +1.5182 / +0.8295 → **worsened** | +1.3439 / +1.8171 → worsened |

聚合三态计数（17 变量 × 5 时效 = 85 格）：`RW-B−RW-A` 29/39/17、`RW-B−(a)−RW-A` 41/7/37、
`RW-B−(b)−RW-A` 17/49/19（improved/worsened/unresolved）。

### 4.2 分支判定：`stop-confounded-control`

按 §3 冻结规则第 1 条，负控制 `RW-B−(a)` 自己出现**同号恶化**（24/48/72h）⇒
**「协议或权重有混淆，停下报告，本轮不做归因」**。规则照写触发，本轮的归因**未做**。

### 4.3 但规则的前提不成立：负控制是**退化**的，不是被混淆的

规则假设「负控制恶化」意味着协议或权重出了错。本轮直接测了这件事，结论相反：

1. **负控制在构造上就是 RW-A。** `solver_gate_proposal=False` 时该步走 pre-RW-B 的 correction 通路，
   `Z` 虽然照常递推但**不进输出**。把**同一份权重**分别装进 RW-A 与 RW-B−(a) 两个构造，
   前向、逐步 drafts 与 adaptive 三路**逐位相同**（`max |Δ| = 0.0`），唯一区别是后者**多返回**一个状态。
2. **它连梯度都相同。** 同一份权重、CPU、一步 streamed backward：总损失**逐位相同**，
   131 个共享张量的梯度**全部逐位相同**（最大相对差 `0.000e+00`），
   而 solver 侧（`solver_init`/`solver_cell`/`solver_gate`/`proposal_head`）**收到 0 个非零梯度**——
   它是 RW-A 加上一段**死计算**。
3. 因此训练出的两个 checkpoint 之间那 1.6e-4（相对）的权重差、以及 3e-05…8e-05 K 的「delta」，
   只能是本机**已知的 GPU 非确定性**（同机同 seed 不逐位一致，见 `project-gpu-training-nondeterminism`）；
   作为量级参照，同一比较下 RW-B 与 RW-A 的相对权重差是 **1.74**，两者相差四个数量级。
4. 负控制的三张修正几何读数也与 RW-A **逐位相同**（0.03587/−0.2328/0.199 与 RW-A 完全一致），
   而 RW-B 明显不同。也就是说比较器是在**同一个模型的不同浮点轨迹**之间读符号。

所以正确说法是：**负控制没有失败于「被混淆」，而是失败于「不可区分」**——它无法作为控制臂，
因为它按定义就等于基准臂。**这是本轮设计（我选的这个控制）的缺陷，不是协议或权重的缺陷。**

### 4.4 主问句自己的读数（明确标为**非**登记分支）

`RW-B−(b)` 是**真**臂（递推关、门控与锚定提案仍施加），它的读数是干净的：
48h **+0.785**、72h **+1.174**（两 seed 同号）仍为 worsened ⇒
**若**负控制可用，规则会归因到 **(a) 门控 + 锚定提案**——因为递推被去掉后恶化仍在，
「只剩门控+锚定提案被施加」。这与 D1 探针同向（×1.8 的幅度由 (a) 通路带来）。

但**本轮不把这条写成归因**，理由有两条，都必须照写：
负控制未通过登记门槛；且 `RW-B−(b)` 的 72h 幅度在两 seed 间是 1.52 对 0.83（相差 83%），
而 48h 是 0.83 对 0.74——方向一致、幅度不稳，与上一轮 72h 的 30% 离散同类。

### 4.5 一条顺带但结实的观察

**训练会让门关小。** D1 在**冻结门控**的归档 checkpoint 上做递推移除，第 1 步幅度是 RW-A 的
1.48/3.04 倍；而**从零训练**的 `RW-B−(b)` 第 1 步幅度只有 **0.0015**（RW-A 的 0.04/0.05 倍，
约 23 倍更小），余弦全程为正。同一个「去掉递推」的干预，推理期消融与训练期消融差 35–60 倍：
**门控在「Z 不递推」这个配置里学会几乎关闭**。这是门控作为稳定器起作用的直接证据，
也是 D1 那些推理期读数**不能**直接当成训练期结论的定量说明。

---

## 5. 本轮复现了上一轮的登记读数（稳定性观察，不合并）

同一 M2 store、同一协议族、同 seed、**但 `model_code_sha256` 与 `protocol_sha256` 都不同**
（本轮新增两个子开关与新的协议正文），`RW-B − RW-A` 的 t2m 逐 seed delta 与上一轮**几乎逐位重合**：

| 时效 | 上一轮 seed41/seed42 | 本轮 seed41/seed42 |
| --- | --- | --- |
| 6h | +0.198633 / +0.037213 | +0.198648 / +0.037218 |
| 12h | −0.043599 / −0.114768 | −0.043572 / −0.114740 |
| 24h | −0.022067 / −0.127439 | −0.022000 / −0.127350 |
| 48h | +1.061712 / +1.068386 | +1.061699 / +1.068513 |
| 72h | +1.344045 / +1.816831 | +1.343897 / +1.817077 |

**读法**：这是「上一轮的负结果不是一次性噪声」的可复现性证据，**不是**支持或加重任何机制声明，
也**不得**与上一轮数字相加或并列（两轮 digest 不同）。同时它反过来说明：本轮新增两个子开关
**没有移动默认路径**——与 §2 的两条逐位等价测试一致。

---

## 6. limitations（照写）

`scientific_claim: false` 写在协议、逐 seed 结果、合并结果与配对结果四处。

- **本轮的负控制无效**，这是设计缺陷；因此本轮的**归因判据未通过**，只有 §4.4 标为「非登记分支」的
  一条读数与 D1 的推理期读数可用，两者都**不是**归因。
- one bounded four-arm run at 400 updates, not a convergence or SOTA comparison
- two seeds: sign agreement across two seeds is consistency, not significance, and no
  significance threshold is introduced or relaxed; two seeds are weaker than the three
  the earlier rounds used
- the arms are NOT capacity-matched: RW-A and RW-B each add parameters and FLOPs, so
  every pair here is capacity-confounded and only the measured counts say by how much
- the leave-one-out arms remove a piece **at training time** instead of reusing the
  archived checkpoints, because the gate+proposal-removed row would otherwise route an
  arm through a correction head that arm never trained（本轮把它测了出来）
- the reference arm RW-A is **retrained** in this round rather than reused: the model code
  digest and the protocol differ, so the two rounds' numbers are never pooled
- the correction probe reports error/update geometry on validation windows only, and its
  retrospective damping uses future truth: a diagnostic, never a deployable rule
- **修正探针每 seed 只有 8 个窗口**，per-step 比例是这 8 个窗口上的比例，不是分布估计
- one winter segment of one year in one region: no seasonal, cross-year or cross-region
  conclusion is testable
- the space-time conditioning pathway is on in every arm, so this round says nothing about
  the pathway itself
- validation-split only; the test split stays sealed and was never opened for selection
- the val climatology is the store's 8-bucket (month, hour) train-only mean, not a strong
  seasonal climatology
- **D1 与 D4 是同一条问题的两种干预**（推理期消融 vs 训练期消融），两者数量级不同，
  不得互相替代或混用；§4.5 给出了差多少。

**本轮特有的三条**：

1. 判决是在这个 400-update 预算、这个容量档、这个种子对下的判决；不能外推成
   「局部门控求解状态这个机制不可能有用」，也不能外推成「(a) 一定有害」。
2. 48/72h 的恶化在两轮里都复现（§5），**这是本轮最结实的一条**；但它仍然只是**负结果的可复现性**。
3. §4.3 的「负控制 ≡ RW-A（逐位）」是**构造**结论，不依赖 seed、不依赖训练；
   它是本轮唯一一条不靠统计、不靠两 seed 同号的结论。

---

## 7. 未做的事（写清楚，不用「应该没问题」代替）

- **未做归因**：按 §3 规则第 1 条，负控制触发即停。本轮**没有**为补救而另加臂
  （那会破坏预注册），也**没有**改判据或阈值去让主问句单独成立。
- 未做确认轮（≤2.5 GPU-h，**未动用**）；未跑第三 seed；E2/E3/E4 未做；M3/M4/M5 未做。
- 未读 test（全程封存）、未下载数据、未租 GPU、未合并 main、未 force push、未关闭 #70–#75。
- **未重跑或改写上一轮任何产物**：`outputs/r7_72_rw_b_pilot/` 与其 attempt1 目录只被**只读**打开；
  D1 探针把自己的产物写到 `outputs/r7_rw_b_subtraction_probe/`。
- 未把 role 标记的小幅一致改善写成机制声明（(c) 本轮**不重测**，只引用上一轮已登记的对照）。
- 未把 12/24h 的改善当作加码理由或支持证据。
- 未对 correction_head 未训练这一事实做冻结例外的放宽；只把它记为局限性。

---

## 8. 下一项的第一个具体动作（按停止条件 3）

§5 停止条件 3 的条件**部分**满足：留一对照**无法归因**（负控制失效）。因此：

1. **停止发明新模块**（本轮本来也没加）。
2. 记下本轮给出的**可反驳假设**：*若换一个真正可区分的负控制——例如把 `Z` 换成同形状的
   冻结随机张量、其余不变——则 48/72h 的恶化应当仍然出现，因为 §4.4 与 D1 都指向 (a)。*
   这条假设可被下一轮直接证伪，且**不需要新部件**。
3. 按停止条件 3 的字面要求，转去**重新检查 forecast state / training objective / data regime**：
   具体第一动作是把 §5 的「两轮几乎逐位重合」与 §4.5 的「门学会关闭」放在一起看——
   两者都指向「问题可能不在 solver 结构，而在被修正的对象（forecast state）与被优化的目标」。

---

## 9. 提交与 CI

| 提交 | 内容 | `R7 CPU CI` |
| --- | --- | --- |
| `0b93ca4` | 两个子开关、等价与定向测试、D1 探针、规则文档与基线同步（`[model-digest-change]`） | 见 §10 |
| `fed56f4` | 修 `EARLY_STOPPING_PATIENCE` 漏导入（两 seed 在写协议前即失败，**未耗 GPU**） | 见 §10 |

本轮提交信息**未**带任何实验标签，因此 17 条实验 workflow 按 commit-message 标签门控
**skipped**（设计行为，**不算失败，也不当通过**）。

## 10. 本机 conventions 与测试计数

`python tools/check_conventions.py`：37 条阻断 **0 违规**；`git show --check` 干净。
R-009 基线随本轮从 885/2180 抬到 **900/2210**，同步更新了 `docs/rules/MIGRATION.md` 与
`docs/rules/testing.md`（两处数字由 `tests/test_check_conventions.py` 直接对 checker 常量断言）；
规模报告标记同步为 R-020=44 / R-021=28 / R-023=17（`size-thresholds.md` 的机器核对标记同行更新）。
第一次写的 624 行驱动**越过 R-051 的 600 行硬上限**，按规则拆出
`training/r7_rw_b_subtraction_protocol.py` 后回到上限内，**未新增任何例外**。

| 口径 | 结果 | skip 原因 |
| --- | --- | --- |
| 本机（有 `outputs/` 产物） | **1499 passed, 3 skipped** | 三条 `test_real_data_pipeline`：可选真实数据 fixture 不在版本控制内，且**不允许**用合成数据兜底 |
| 干净检出（`git clone` 后 checkout 本提交） | 见 §11 | 同上，另加因本地产物缺失而跳过的用例 |
