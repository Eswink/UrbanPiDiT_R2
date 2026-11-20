# 0116 — typed evidence 的 in-protocol 三臂重训，与「无截断」的预算设计

- **日期**：2026-11-20
- **状态**：proposed
- **范围**：新建 `scripts/r7_s3_in_protocol_protocol.py`、`scripts/r7_s3_in_protocol_preflight.py`、
  `scripts/r7_s3_in_protocol_verdict.py`、`scripts/r7_s3_in_protocol_assembly.py`、
  `scripts/study_r7_s3_in_protocol.py`、`scripts/r7_s3_joint_criterion.py`、
  `tests/test_r7_s3_in_protocol.py`；**不修改** `model/**`、`training/**`、`data/**`
- **依据**：`docs/plans/0035-typed-evidence-three-arm-in-protocol.md`、
  决策 0030（自主执行）、0045（pinned 复用）、0048（容量轴已关闭）、0062（配对控制必须本轮重跑）、
  0113（静态协变量轮的安慰剂教训）、0115（上一轮的截断与判据选择）
- **执行源**：`docs/goals/s3-typed-evidence-in-protocol.md`

## Context

毕业论文题目是「基于深度学习的物理信息混合天气预测研究与应用」，因此**物理信息必须有真实贡献**。
上一轮（0115）把 #79 的容量匹配三臂从 400 updates 提到 32,132 updates，结果三个 `typed` 臂
全部被每 seed 35 h 的冻结包络截断，`C − B` 依然没有被评分过。本轮要回答的就是这一个问题，
并且必须先把截断这件事从根上解决。

### 开轮前已经定下的四个事实

1. **`C − B` 仍是唯一悬空的问题。** 400 updates 只测了方向（`#79`：t2m 6h/12h 双 lead
   `supported`，−0.0533 K）；2,400 updates 测出 `|Δrmse| ≤ 3.2e-6 K`
   （`s3-typed-spatial-result`）；32,132 updates 的 typed 臂在评分前被截断。
2. **上一轮的 `B − A` 无法归因。** `generic` 比 `incumbent` 多 **148,800 参数（+4.5%）**。
   把它写成「物理信息有效」是把容量增益记在物理账上。
3. **上一轮的 (b) 判据选错了变量。** 判 `t2m` 且要求 `−0.005 K`，实测 `−0.0016…−0.0032 K`。
4. **截断的真正开关不是 `hard_cap_seconds`。** 是 `tools/r7_round_stages.py:292` 的
   `limit = min(deadline, now() + per_seed)`。`tools/` 不在 `training_code_digest()` 集内。

### 段 B 在开轮前给出的机制信息（`exploratory`，0 GPU-h）

在上一轮已存在的 15 份 `boundary_rmse.csv` / `provenance.json` 上装配族并求同时区间：

- 族 P（`t500,t850,u850,v850` × 5 lead）：**10/20 格区间排除零，全正、零负**
- 族 S（其余 13 通道 × 5 lead）：**29/65 格区间排除零，全正、零负**
- 最强在 **z500（72h +0.69%）/ z250（72h +0.51%）**，族 P 内最强 `t500`（48h +0.46%）、
  `v850`（24h +0.45%）
- 每族共用 **1,447** case（五 lead 交集，丢 11 个只在短 lead 出现的 case），1,420 blocks

**它不构成确认**：族划分与 0.001 下限是看着这些数字定的。它解释并纠正了上一轮的判据选择，
但只有本轮新训的读数能检验冻结后的判据。

## Decision

### D1 — 三臂全部在本轮协议内重训，不做 pinned 复用

`C − B` 的价值全在「同协议、同父、同数据、同剂量」。pinned 复用会把 `C − B` 从 in-protocol
降级为跨协议（0045 允许，但见 D6 的代价），并且在 32,132 updates 尺度上仓里**没有**
run-to-run 桥可引用——现有桥都在 2,400 updates。三臂全重训同时白得一座 32k 桥：A 臂同代码、
同数据、同 parent、同 seed 重训，是本轮唯一能给出「同配置两次独立训练差多少」的东西。

### D2 — 判据的族按**机制**划分，不按观测量

- **族 P（主判据，20 格）**：`t500,t850,u850,v850` × 5 lead。这四个通道是
  `TYPED_EVIDENCE_CHANNELS` 声明的、`typed_local_fields()` 实际消费的集合
  （`model/r7_process_tensor_diagnostics.py:158-165`）。它在任何读数之前就已确定。
- **族 S（次要，65 格）**：其余 13 个通道 × 5 lead。
- 分区若按「哪个变量好看」来划，本轮的正面结果就是自证。所以分区必须由代码结构决定。

### D3 — 度量改成量纲无关的 skill 增量

`Δskill_i = (mse_对照,i − mse_处理,i) / D[L,c]`，`D` 是该格的气候态 MSE
（`climatology_skill.csv` 的 `rmse_climatology` 平方，臂无关且固定）。

逐 case 除以一个常数是**单调缩放**：它改不了 delta 的符号，也改不了区间是否排除零。
它做的是让 K、m·s⁻¹、Pa、kg·kg⁻¹、m²·s⁻² 五种单位的 17 个变量进同一个族而不被单位主导。
上一轮的 `0.005 K` 只能对温度有意义，这正是它选了一个对通路最不利的变量的后果。

下限取 **0.001**（气候态 MSE 的千分之一）：真机制能过，可分辨但无意义的符号过不了。

### D4 — 用 max-statistic 同时区间控多重比较，族内一个临界值

`training/r7_simultaneous_stats.simultaneous_paired_interval`：α=0.05，
block_length=28 cases（7 天，验证集 6 小时步），4,000 次重采样。
它要求同族所有格共享同一 case key 集，而五个 lead 的 case 集是
**1458/1457/1455/1451/1447** 严格嵌套。本轮在**交集 1,447** 上建族，并把这个数报出来
（`n_dropped_cases`）——静默用短 lead 的 cohort 去验一个五 lead 的声明，是另一种自证。

### D5 — 「无上限」落成三件结构上的事，而不是一个数字

用户显式授权 GPU 训练时间无硬上限，要求严格避免截断。字面意义的无上限在不改 `training/` 的
前提下**结构上不可达**：`_validate_options` 拒绝非有限 deadline，driver 拒绝 `None`，
`subprocess.run` 拿到的是有限 timeout。改 `training/` 会改 `training_code_sha256`，
作废父 pin。因此本轮改的是三件可验证的事：

1. **包络按 treated 速率推**：1.47 s/upd（高于全部六个实测值）→ 单 seed 需 151,302 s
   （42.03 h）→ 取 2.05× = 310,000 s；硬上限 660,000 s 覆盖两波。
2. **守卫读两个独立数字**：`envelope()` 从 treated 常数重算需求，测试断言
   `PER_SEED_SECONDS ≥ 2.0 × need`。另有一条测试断言「按 incumbent 速率 1.112 算出的包络
   **小于**本轮需求」——那是上一轮的原错误，现在它会失败。
3. **探针早停**：每 seed 先跑三臂各 **400 updates**（约二十分钟），用实测速率外推；
   若剩余包络的 **60%** 装不下，立刻中止并记 `refused-insufficient-envelope`。
   包络再大也仍是预测，探针让预测可被证伪，代价是分钟级而不是十小时。
4. **截断必留收据**：任何非正常结束写 `seed{N}_{arm}_partial_receipt.json`，含已完成
   updates、实测 s/upd、`progress.jsonl` 回指。上一轮 30.2520 GPU-h 的空转是靠人工从
   `progress.jsonl` 反推的，glob receipt 的账本会静默少计。

### D6 — 本轮**不做**安慰剂臂（并承认这是本轮的局限）

安慰剂（同参数、同 FLOPs、物理场逐 cell 置换）是分离「物理内容」与「容量」的唯一干净办法。
0113 的静态协变量轮正是因为缺安慰剂，无法排除「多一张位置查找表」这个替代解释，本轮不重犯
同样的错误，但结论是**本轮仍然不做**，理由是机制查证后的硬约束：

- 场在 `model/typed_evidence_r7.py:270` 的 `field_maps(anchor, batch)` 内部由
  `typed_local_fields()` **现算**；`evidence()` / `token_evidence()` / `inject()` /
  `initial_process_state()` / `forward()` 这条链上没有任何形参接受外部场。
- 唯一能逐臂换值的通道是 state-dict，而 8 个标准化量
  （`field_scale/field_mean/field_std/denorm_mean/denorm_std/field_active`）全是
  `persistent=False`，**不进 checkpoint**（`model/typed_evidence_r7.py:151-164`）。
  0113 之所以能做，是因为它的场是 `persistent=True` 的 buffer（`model/static_conditioning_r7.py:183-184`）。
- 因此安慰剂必须改 `model/` ⇒ `model_code_sha256` 变 ⇒ 本轮即将产出的 6 个 32k checkpoint
  与 3 个父 checkpoint 全部不可加载（`training/r7_experiment.py:128`），而父迁移虽为 0 GPU-h，
  会作废大量既有产物（当前修订下有 36 份 `training_report.json` 与 23 份 `protocol.json`
  引用 `59340710…`）。

**顺序论证**：安慰剂轮只需训 D 一个臂（`B/C/A` 可 pinned），约 42 GPU-h。所以
「先三臂（≈115）→ 若 `C − B` 有信号再补 D（≈42）」与「一次四臂（≈156）」总账相同，
但若 `C − B` 是空的，可以在花那 42 之前停下。**这可能意味着要多跑一轮**，这是本决策
显式接受的代价：本轮 `B − A` 的任何正面读数**只能当描述性工程读数**，必须同时写
`+148,800 参数（+4.5%）`，不得声称归因于物理内容。

### D7 — 族判别：本决策不重开任何已关闭的轴

| 已关闭 | 本轮 |
| --- | --- |
| 0048 容量轴（dim/depth） | **不动结构**；三臂参数量按设计（B≡C，A 严格少 148,800） |
| EMA/SWA（永久关闭） | 不做任何权重平均 |
| 长 lead 权重（关闭） | 不改 loss 权重 |
| C1 物理约束损失 / C2 导数匹配 | **不加任何 loss 项**；物理量作**前向输入证据** |
| Stage R / 低秩带 / 平衡带 / 地转投影 / case 自适应 | 全部不触碰（0115 已实测关闭） |
| 0113 城市静态 | 不碰；但借它的安慰剂教训（D6） |
| 「不改 4,000 updates」纪律 | 32,132 updates 是已注册的 p4000 档位，不是新档位 |

## Consequences

**变难的地方**：

1. **成本约 117 GPU-h / 84 h 墙钟（3.5 天）**。软预算 210,240 s（58.4 h 墙钟）会被超，
   记 overrun；硬包络 660,000 s 是结构上限，只有它截断。账本 `used_gpu_h` 在修掉上一轮的
   双计后为 **735.378628**（cap 20.0 / `remaining_gpu_h` −715.378628），本轮逐行登记。
2. **`B − A` 仍然无法归因**。这是 D6 的直接代价。即使 `B − A` 在两族上都过，可写的也只是
   「物理信息模型 vs 生产配置的工程读数」，**不是**「物理信息有贡献」。真正能归因的只有
   `C − B`，而它是「融合位置」不是「有无物理」。
3. **`C − B` 可能仍是零**。2,400 updates 时它是 `|Δrmse| ≤ 3.2e-6 K`。若 32k 仍是零，
   物理类型结构这条支线应当关闭，而题目里的「物理信息」只能依赖前向通路整体的描述性读数
   与机制性发现。**这一点在动手前就承认。**
4. **段 B 的 exploratory 读数有被误用的风险**。它是在 6 个已完成臂（缺 typed）上算的，
   协议 digest 与代码身份都与本轮不同。证据页只能引它作阈值动机，**不得**当成本轮的读数。
5. **`z500@72h` 护栏有真实的失败风险**：p4000→p1600 已把该指标从 427.35 升到 445.78
   （+4.3%），说明预算本身在恶化它。若 G 为假，结论必须限定，并写明动力学反向。
6. **三 seed 是致性证据，不是显著性检验**；block bootstrap 的区间是对 case 的，不是对 seed 的。
7. **GPU 共驻**：每次 spawn 前只读核对余量，**禁止**对非本实验进程发送信号。

**代价**：本轮把「归因」这一半推给下一轮（若值得做）。换来的是：更便宜的决定性读数先落地、
零父迁移、零 code identity 变更、旧 checkpoint 全部仍可加载、以及仓里第一座 32k run-to-run 桥。

**Trade-off**：选「先三臂后补安慰剂」而不是「一次四臂」，是用「可能需要两轮」换
「若 C−B 为负就不花那 42 GPU-h」。这个交换是显式的：若 `C − B` 为负，安慰剂轮**不应该开**，
因为那时连「物理场有无贡献」都不必再问——通路整体没有增量。

## 已否决的替代方案

| 方案 | 否决理由 |
| --- | --- |
| 一次四臂含安慰剂（≈156 GPU-h） | 必须改 `model/` → 作废本轮 6 个 32k checkpoint 与 3 个父；且为负时那 42 GPU-h 白花 |
| 只重训 typed、A/B pinned（≈42 GPU-h） | `C − B` 降级为跨协议；32k 尺度的 run-to-run 桥不存在，无法量化降级幅度 |
| 沿用 `t2m` + 0.005 K 单一判据 | 已知效应只有门槛的 1/3，而 z500 上 13/15 格 CI 排除零；会重复上一轮的 False |
| 把 `hard_cap_seconds` 调成 30 天 | `tools/r7_round_stages.py:292` 的 `min()` 让 `per_seed_seconds` 才是绑定约束，调它无效 |
| 改 `training/` 让 `deadline=None` 可达 | 改 `training_code_sha256` → 父 pin 失效 → 需父迁移，且违反「不改 training」的本轮约束 |
| 直接把段 B 的 exploratory 读数当结果 | 阈值与分区是看着它定的；那样是自证，不是检验 |
