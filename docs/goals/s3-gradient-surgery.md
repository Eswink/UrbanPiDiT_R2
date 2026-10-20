# S3：short/long gradient surgery（单因素训练改动）——按短/长梯度冲突投影 g_A，检验 72h interior_32 t2m 是否改善

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-gradient-surgery", "driver": "scripts/study_r7_s3_gradient_surgery.py", "output_stem": "r7_s3_gradient_surgery_20261020", "control_record": "s3-anchor-isolation-result", "reference_record": "s3-wide-interior-dose", "seeds": [41, 42, 43], "device_mode": "dual", "test_path": ["tests/test_r7_s3_gradient_surgery.py", "tests/test_r7_s3_gradient_surgery_protocol.py"], "commit_paths": ["training/r7_long_rollout_runner.py", "training/r7_gradient_surgery.py", "scripts/study_r7_s3_gradient_surgery.py", "scripts/r7_s3_gradient_surgery_support.py", "scripts/r7_s3_gradient_surgery_text.py", "tests/test_r7_s3_gradient_surgery.py", "tests/test_r7_s3_gradient_surgery_protocol.py", "docs/goals/s3-gradient-surgery.md", "docs/goals/main-model-climatology-campaign.md", "docs/rules/size-thresholds.md"], "commit_message": "feat(r7): add the short/long gradient-surgery single-factor driver", "register_paths": ["docs/R7_S3_GRADIENT_SURGERY_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md", "docs/goals/s3-gradient-surgery.md"], "register_message": "feat(r7): register the gradient-surgery run as a new evidence record", "code_archive": "outputs/r7_s3_gradient_surgery_code.zip"} -->

状态 **实现中：单因素驱动与 CI-safe 测试已派生，尚未冻结/未跑 GPU、未登记**（2026-10-20）。
唯一主计划 `docs/goals/main-model-climatology-campaign.md`；本轮承接 decoder 负结果
（`record:s3-decoder-base-anchor-result`）之后的训练信号方向：只改反向传播中的梯度手术，
不改模型、数据、父、预算或任何已关轴。

## §0 Objective（可粘贴；单段）

> 在 docs/goals/s3-gradient-surgery.md 接续 S3/index85/157.289621 GPU-h、2023 test 未评分 r0、S4 未启动：在注册的 2400-update、12-step interior_32 long-rollout recipe 上只改一个训练期因子 gradient_surgery off→pcgrad——组 A 为加权非零步 0/1/3（6/12/24h，权重 1.0/0.5/0.5），组 B 为加权非零步 7/11（48/72h，权重 0.5/0.5），两组都由同一 per_step_losses 张量求和，零权重步仍按注册方式推进 history；每个 update 两次 autograd.grad（第一次 retain_graph）遍历全部可训练参数，展平/dot/范数在 float64 中计算，仅当 dot(g_A,g_B)<0 时 g_A←g_A−(dot/||g_B||²)g_B，更新为 g_A+g_B，随后沿用已注册的 clip_grad_norm_(clip=1) 与 optimizer.step，顺序固定 A→B（预声明的 one-sided 特化：只投影 g_A，g_B 不变；不是对称的 two-way PCGrad Algorithm 1）；模型 spec 逐字等于 pinned s3-anchor-isolation-result 控制 spec（无 decoder_base_mode 键，默认 persistence），model_code_sha256 保持 bec3859fc1386b14777300ad3e387546829c8a0fbf52e817e1cb90324b988ac1，复用 decoder 轮父迁移 receipt 002f08439b99fc3658a422e0732183f7a48c5b43d5790c133c53c4eaf004d6c3 并断言 state digest 7c416ccb…/857d8804…/61fdac58…、expanded 00ada5…/baed2f…/4dda64…、anchor init de0e76…；数据、seed 41/42/43、2400 updates、LR 2e-5、warmup 10、K=4、depth=4、dim=192、batch 1、clip 1、deep K-axis supervision、物理权重 (1,.5,0,.5,0,0,0,.5,0,0,0,.5)、评分域与气候态分母全部逐字不变；预声明出口（不新增阈值）：(a) 候选 72h interior_32 t2m skill 在三个 seed 全 >0；(b) 同 seed 候选−pinned 控制 72h ΔRMSE 三 seed 全 <0；(c) 候选 48h skill 三 seed 全 >0；(d) 候选−控制在 6/12/24h 的 seed 均值 ΔRMSE ≤0；(e) 桥：pinned 控制对 pinned s3-wide-interior-dose 的 15 个 seed×lead RMSE 差落在 −4.8e-6…+1.8e-5 K；(a)–(e) 全真→candidate_state needs-review 并提请独立复核，不自动进 S4；任一 (a)–(d) 假→永久关闭 short/long gradient-surgery 支（不扫组定义/系数/顺序、不加 seed、不改 updates/LR/clip、不与已关轴组合、不换名）；(e) 假只写 identity_unresolved=true，是工程身份失败而非科学判定；若三 seed 每步都零冲突零投影，候选即注册梯度（至多 FP 结合次序不同），记 unresolved（更差则 negative）、not-candidate、关支且不得加 seed 或调组/系数去制造冲突；预算 per-seed 21600 s、planned 43200 s（软）、hard 86400 s，软超继续并记 overrun，硬截断即 attempt 失败不算科学通过；test 不读、r=0、S4 不启动、scientific_claim=false。

（objective 字符数：1772；单段。）

## §1 起点与实质差异

- 起点工作分支 `r7/weather-reasoning`；累计 **157.289621 GPU-h**、index **85**（decoder 轮已登记）。
- 起点证据：`docs/R7_S3_DECODER_BASE_ANCHOR_RESULT.md`（negative，decoder-base 支关闭）、
  `docs/R7_S3_ANCHOR_ISOLATION_RESULT.md`（配对控制 N，registered gradient）、
  `record:s3-wide-interior-dose`（pinned 外部参照 A）、
  `docs/R7_S3_TRAIN_GRADIENT_COMPONENTS.md`（四例 weighted_gram：组 A/B 余弦
  +0.202574/+0.138378/+0.598252/+0.471881，组范数 A 1.28–3.70 vs B 5.23–14.61）。
- **为什么选这个因子**：主计划的训练信号方向需要检验“长程失败是否来自短/长梯度方向冲突”。
  PCGrad（arXiv:2001.06782v4 §2.2、§3.1、Algorithm 1）给出可证伪的投影规则；本轮采用预声明的 one-sided 特化（只投影 g_A，g_B 不变）；
  仓库内的四例诊断说明该实例的全局冲突可能稀少，因此零冲突 null 是必须如实报告的
  真实可能结果，而不是要调参绕开的失败。
- **与已关闭支的区别**：本轮不改模型前向（`model_code_sha256` 不变）、不改损失定义/权重/
  掩码/数据/父/预算；只把注册的 `result.loss.backward()` 换成两组 `autograd.grad` 后的
  条件投影。off 路径是独立分支，逐位保持注册反向。
- **单因素边界**：控制是 pinned 旧代码读数（`s3-anchor-isolation-result` 的 control 臂），
  候选是新训练代码三 seed 独立轨迹；两臂共享同一父状态、同一数据、同一 2400 updates 与
  同一评分域，但 `training_code_sha256` 不同（控制 `8ad77d93…`），wall-clock 与
  run-to-run 方差也不同（limitations 明写）。

## §2 交付物

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 本轮冻结设计（本文件 §3） | 本文件 |
| D2 | 训练期梯度手术模块（`off` 默认不进入；`pcgrad` 两组投影） | `training/r7_gradient_surgery.py` |
| D3 | runner 接线（off 路径仍是 `result.loss.backward()`；新契约字段只有 `gradient_surgery`） | `training/r7_long_rollout_runner.py` |
| D4 | 新驱动 + 支持模块 + prose 模块 + 新 CI-safe 测试（合成树/monkeypatch/反证，不读 outputs/） | `scripts/study_r7_s3_gradient_surgery.py`、`scripts/r7_s3_gradient_surgery_support.py`、`scripts/r7_s3_gradient_surgery_text.py`、`tests/test_r7_s3_gradient_surgery.py`、`tests/test_r7_s3_gradient_surgery_protocol.py` |
| D5 | 新协议 / 新输出 / code.zip | `preparation_protocol.json`、`protocol.json`、`code.zip` + `archive_receipt.json` |
| D6 | 候选臂三 seed 训练与评分 | `seed{41,42,43}_receipt.json`、每 seed 五 lead 的 `boundary_rmse.csv`、`gradient_surgery_summary` |
| D7 | 配对读数 | `readings.json`：候选−控制、候选−pinned dose、控制−pinned dose 桥、17 变量表、手术摘要 |
| D8 | 登记与精确 CI | `docs/R7_S3_GRADIENT_SURGERY_RESULT.md`、index record、brief、账本行、§8、campaign-state |

## §3 判据与设计决定（本轮冻结）

科学门与验收形式只来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` 与 D4 口径；本轮
**不新增**任何数值阈值，也不宣布通过。

- **唯一改动因子**：`gradient_surgery` 从 `off`（控制，pinned 注册反向）改为 `pcgrad`（候选）。
  组定义冻结：A = 加权非零步 0/1/3（权重 1.0/0.5/0.5），B = 加权非零步 7/11（0.5/0.5）；
  零权重步 2/4/5/6/8/9/10 不属于任何组，但仍按注册路径推进物理 history。
- **候选反向**：每个 update 两次 `torch.autograd.grad`（第一次 `retain_graph=True`）覆盖
  全部可训练参数；展平/dot/范数在 float64 中累加；全局冲突判定 `dot(g_A,g_B)<0`；
  冲突时 `g_A ← g_A − (dot/||g_B||²) g_B`；写回 `.grad` 为 `g_A+g_B`（cast 回参数 dtype），
  然后沿用已有 `clip_grad_norm_(clip=1)` 与 `optimizer.step`。顺序固定 A→B；
  PCGrad Appendix H 的随机顺序优势写入 limitations，本轮**不**引入随机顺序 RNG；本轮只投影 g_A，g_B 不变，因此不是对称的 two-way Algorithm 1。
- **off 路径**：独立分支，精确调用 `result.loss.backward()`；契约/报告**不加**任何
  `gradient_surgery` 字段；测试断言与注册反向逐位相同。
- **保持不变**：模型 spec 逐字等于 pinned 控制 spec（无 `decoder_base_mode` 键，默认
  persistence）、`model_code_sha256=bec3859f…`、父迁移 receipt `002f0843…` 与逐 seed
  state/expanded/anchor_init digest、2400 updates、LR 2e-5、warmup 10、K=4、depth=4、
  dim=192、batch 1、clip 1、deep supervision、物理权重、seed 41/42/43、评分域、气候态分母。
- **禁止组合**：候选要求 `draft_supervision='deep'`、`spatial_gradient_lambda=0.0`、
  `ema_decay=None`、注册物理权重，否则 `ValueError`（训练入口与驱动双重校验）。
- **控制臂 N（pinned，不重训练）**：`record:s3-anchor-isolation-result` 的 control 臂
  （`outputs/r7_s3_anchor_isolation_20261015_attempt01/`），registered gradient（报告无
  `gradient_surgery` 键）、final optimizer weights；驱动的 prepare 断言其 spec 与本轮
  `arm_spec()` **完全相等**、`training_code_sha256=8ad77d93…`。
- **正/负控制（测试逐条断言，失败即 raise）**：
  1. 组定义精确是 0/1/3 与 7/11 且权重/零权重步冻结；改动零权重步不改变任一组和，
     改动加权步按注册权重改变组和。
  2. 二进（dyadic）线性算例上：冲突时 g_A=(1,0)、g_B=(−1,1) → 投影系数 −0.5、
     `.grad=(−0.5,1.5)` 逐位相等；不冲突时 `.grad=(2,1)`；全局 dot 与逐参数 dot 不同
     （−4/+1 但全局 −3）时仍按全局规则投影。
  3. off 路径在真实 tiny 模型上对同一 batch 与 `loss.backward()` 的 `.grad` **逐位相等**，
     且 metrics 键集合与注册完全一致；pcgrad 路径才出现手术诊断字段。
  4. pcgrad 的 `.grad` 与独立重算的两组向量重组逐位相等；`group_sum_residual` 在
     二进算例上为 0。
  5. CPU smoke：12 步合成树的 `fine_tune_long_rollout(..., gradient_surgery='pcgrad')`
     成功，每行含全部手术诊断；off 的 report/contract 无新键；与闭轴组合、改权重轮廓、
     off 声明手术字段均被拒绝。
- **预声明出口（不新增阈值）**：
  - (a) 候选 72h `interior_32` t2m skill 在 seed 41/42/43 **全部 >0**；
  - (b) 同 seed 候选−控制 72h ΔRMSE **三 seed 全部 <0**（负值 = 候选更优）；
  - (c) 候选 48h skill **三 seed 全部 >0**；
  - (d) 候选−控制在 6/12/24h 的 **seed 均值** ΔRMSE **≤0**；
  - (e) 桥：pinned 控制 N 的 15 个 seed×lead RMSE 对 pinned A 的差落在 `−4.8e-6…+1.8e-5 K`。
  **同时满足 (a)–(e) → 提请独立复核（不自动进 S4）**；任一 (a)–(d) 不满足 → **永久关闭**
  short/long gradient-surgery 支：不扫组定义（含 6h-vs-72h 或任何其它划分）、不扫投影
  系数/顺序、不加 seed、不改 updates/LR/warmup/clip/head reset、不与已关轴组合、不换名重开。
- **Null/inert**：若三 seed 每个 update 都零冲突零投影，候选即注册梯度（至多 FP 结合次序
  不同）；outcome_class 记 unresolved（更差则 negative）、not-candidate、关支，**不**加 seed、
  不调组/系数以制造冲突。四例诊断的正余弦使这是真实可能结果。
- **工程身份门**（只算 attempt 失败，修复仅限已确认工程原因，不改因子或判据）：
  `model_code_sha256 == bec3859f…`（无 `[model-digest-change]`）；父 receipt `002f0843…` 与
  state/expanded/anchor_init digest 匹配；控制/参照 pins 匹配且未重训；候选 spec == 控制
  spec；唯一新契约字段是 `gradient_surgery`；off 路径逐位；数据/source/windows/supervision/
  气候态分母 pins 匹配；GPU assignment/gates 合法；test 2023 未读；r=0；S4 未启动；
  `scientific_claim=false`。候选与控制的 `training_code_sha256` 不同：协议记录两值、
  `equal` 与活动路径（deep/0.0/None/pcgrad），该差异本身不是通过判据。
- **证据页 `docs/R7_S3_GRADIENT_SURGERY_RESULT.md` 必须写**：逐 seed 72h skill 与对控制
  delta、48h skill、6/12/24h seed 均值 delta、桥、全 17 变量 × 5 lead、逐 update 冲突摘要
  （conflict ratio、projection count、cosine、norm ratio、per-parameter negative-dot count、
  projection-relative-norm）、first-update losses、checkpoint/parent/protocol/code 身份、
  实际 GPU-h/wall/peak/disk/network，以及 limitations：四例诊断的正余弦、固定顺序 vs
  Appendix H 随机顺序、跨代码候选-vs-pinned 控制、两组求和的 FP 结合次序、两组并非独立
  task（晚期梯度流经早期预测状态）、三 seed 为一致性证据非显著性、单实例/ROI/季节、
  GPU 共驻与不可逐位复现。
- **报告口径**：五个 lead（6/12/24/48/72h）与全 17 变量的 `interior_32` RMSE 都照报；
  不跨单位平均、不挑赢家。`group_sum_residual` 是 scalar-loss 结合次序的描述性残差
  （两次 grad 不额外做第三次反向），不是 gate。
- **test 策略**：`test.jsonl` 从不读取；`test_read=false`、`scientific_claim=false`。

## §4 预算与停止

- **执行预算（冻结）**：`per_seed_seconds=21600`、`planned_seconds=43200`（软预算）、
  `hard_cap_seconds=86400`（硬上限，为计划的 2 倍）。软超预算继续并记 overrun，硬截止即停
  attempt 并全额保留 `failure.json`；失败 attempt 不在原目录 resume、不缩短重试。
  这是**单臂**候选 envelope；控制与参照均 pinned 不重训。
- 只训练 **候选一个臂 × 三 seed**，由 `stage-seeds` 在双卡上按决策 0047 调度；每阶段 spawn
  前只读 `gpu_gate`，**禁止**对非本实验进程发送任何信号（R-054）。
- 计划成本约 3 ×（2400 updates + 五 lead 评分）≈ 6–8 GPU-h；每个 update 多一次
  `autograd.grad`，实测开销按回执逐阶段记账（GPU-h/overhead 照报）。
- **停止条件**：任一 (a)–(d) 不满足即永久关闭该支；null/inert 同样关支；工程失败 / 身份门 /
  hard cap 只算 attempt 失败，修复仅限已确认工程原因，不改因子或判据；test 未读、r=0、
  S4 不启动。

## §5 明确不做

不重训练 N 或 A、不重跑已登记轮次、不读 test、不扫组定义/投影系数/顺序、不加 seed、不改
2400 updates/LR/warmup/K/dim/depth/监督域/物理权重、不 reset 输出头、不改 change_scale 或
output scale、不引入随机任务顺序 RNG、不与 anchor/feedback/architecture/容量/K 轴组织/
损失加权/EMA/空间梯度组合、不换名重开、不改 `data/raw|interim|processed` 或归档快照、
不合成替代数据、不改科学合同/冻结证据/判据、不 main 合并/发版/关 issue、
不 paid/rental/exclusive/force/mirror/破坏性/cron。

## §6 进度与交接

- **已定**：设计（§3）；登记名 `docs/R7_S3_GRADIENT_SURGERY_RESULT.md` / record
  `s3-gradient-surgery-result`。
- **已完成（2026-10-20，实现轮）**：
  1. `training/r7_gradient_surgery.py`（两组损失、float64 全局投影、逐 update 诊断、
     摘要；`off` 不进入该模块）；`training/r7_long_rollout_runner.py` 增加
     `gradient_surgery` 参数与独立 off 分支，SOURCE_FILES 纳入新模块，
     `training_code_sha256` 改变；`model_code_sha256` 不变。
  2. 新驱动 `scripts/study_r7_s3_gradient_surgery.py`、支持模块
     `scripts/r7_s3_gradient_surgery_support.py`、prose 模块
     `scripts/r7_s3_gradient_surgery_text.py`；新 CI-safe 测试
     `tests/test_r7_s3_gradient_surgery.py` + `_protocol.py`（合成树/monkeypatch/反证，
     不读 `outputs/`）。
  3. 父迁移复用 decoder 轮 receipt `002f0843…`，逐 seed digest 由 `expand_parent_state`
     断言；控制 spec 断言逐字段相等；控制训练代码 pin `8ad77d93…`。
  4. 本长文与 campaign-state 指针已写。
- **未做**：未冻结 protocol、未建 code.zip、未跑 GPU、未登记证据页/index/brief/账本；
  `round-guard` 将在实现提交前实跑并记录。test 未读、r=0、S4 未启动。
- **下一动作（S3）**：实现提交 → `freeze`/`prepare`/`archive` → `stage-seeds` 三 seed →
  `reading` → 按 §3 预声明出口读；全真才提请独立复核，任一科学出口假即永久关支；
  身份失败只修工程原因。
- **恢复第一条命令**：`.venv/bin/python tools/r7_round.py state`。
- 不宣布科学通过、不称泛化/SOTA。
