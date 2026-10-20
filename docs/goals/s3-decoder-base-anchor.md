# S3：zero-base decoder（单因素模型改动）——把持久化 anchor 换成 0，检验它在 residual fine-tune 实例中是否承重

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-decoder-base-anchor", "driver": "scripts/study_r7_s3_decoder_base_anchor.py", "output_stem": "r7_s3_decoder_base_anchor_20261020", "control_record": "s3-anchor-isolation-result", "reference_record": "s3-wide-interior-dose", "seeds": [41, 42, 43], "device_mode": "dual", "test_path": ["tests/test_r7_s3_decoder_base_anchor.py", "tests/test_r7_s3_decoder_base_anchor_protocol.py"], "commit_paths": ["model/weather_forecaster_r7.py", "model/process_forecast_r7.py", "scripts/r7_s3_decoder_base_anchor_parent.py", "scripts/r7_s3_decoder_base_anchor_support.py", "scripts/r7_s3_decoder_base_anchor_text.py", "scripts/study_r7_s3_decoder_base_anchor.py", "tests/test_r7_s3_decoder_base_anchor.py", "tests/test_r7_s3_decoder_base_anchor_protocol.py", "docs/goals/s3-decoder-base-anchor.md", "docs/goals/main-model-climatology-campaign.md"], "commit_message": "feat(r7): add the zero-base decoder single-factor driver [model-digest-change]", "register_paths": ["docs/R7_S3_DECODER_BASE_ANCHOR_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md", "docs/goals/s3-decoder-base-anchor.md"], "register_message": "feat(r7): register the zero-base decoder run as a new evidence record", "code_archive": "outputs/r7_s3_decoder_base_anchor_code.zip"} -->

状态 **完成：negative / not-candidate，支永久关闭**（2026-10-20）。三 seed 跑完，预声明出口
(a)–(d) 全未满足、(e) 通过；zero-base 在全部 15 个 seed×lead 单元上差于 pinned 控制。
唯一主计划 `docs/goals/main-model-climatology-campaign.md`；承接已登记的 EMA 负结果
（`record:s3-ema-weight-averaging-result`）与「残差目标参数化侦察被阻断」条目所指明的、唯一可辩护的
后续：**另立预声明模型改动轮**，只改 decoder 的 base 语义。

本轮直接承接 2026-10-20 §8 侦察结论：当前注册模型已经是 persistence-anchored 残差解码器
（`model/weather_forecaster_r7.py:113-126`、`model/coarse_forecast.py:98-108`、
`model/local_solver_state_r7.py:187-209`），因此“把目标改写成 Δ”不是单因子；真正的残差参数化是
把 decoder anchor 设为 0。本轮只做这一个改动，不组合任何已关轴，也不重开架构/容量/损失加权方向。

## §0 Objective（可粘贴；单段）

> 在 docs/goals/s3-decoder-base-anchor.md 接续 S3/index84/149.623834 GPU-h、2023 test 未评分 r0、S4 未启动：在注册的 persistence-anchored 解码器（forecast = X_t + head(...)，X_t = history[:,-1]）上，仅把 decoder base 置零（decoder_base_mode='zero'，forecaster 的直接 head anchor 与 R7ForecastOutput.base_state 同步置零，local solver 的 proposal_anchor 因而为 0），并保持父权重、近零初始化的输出头、2400 updates、LR 2e-5、warmup 10、K=4、depth=4、dim=192、数据/损失/物理权重/seed/评分域/气候态分母逐字不变；不 reset 最后一层、不改 change_scale、不组合任何已关轴；两臂保留同一惰性气候态锚（anchor_output_prior=False、anomaly_feedback=False，锚值不进入任一输出（前向仍会求值锚表），清零锚表不改变任一臂）。预声明出口（不新增阈值）：(a) 候选 72h interior_32 t2m skill 在 seed 41/42/43 全 >0；(b) 同 seed 候选−控制 72h ΔRMSE 三 seed 全 <0；(c) 候选 48h skill 三 seed 全 >0；(d) 候选−控制在 6/12/24h 的 seed 均值 ΔRMSE ≤0；(e) 桥：pinned 控制对 pinned s3-wide-interior-dose 的 15 个 seed×lead RMSE 差落在 anchor 轮 −4.8e-6…+1.8e-5 K 内。全部 (a)–(e) → 提请独立复核（不自动进 S4）；任一 (a)–(d) 不满足 → 永久关闭「zero-base decoder」支，不扫 anchor 值（含 base=X_0）、不加 seed、不改 2400 updates/LR/head reset/output scale、不与任何已关轴组合、不换名重开；(e) 失败只写 identity_unresolved=true（工程身份失败，非科学判定）。证据页必须写：控制为旧代码 pinned 读数（新代码 persistence 路径只做逐位等价断言、未重训控制），比较混合 base 语义与 run-to-run；zero-base 不是 from-scratch 绝对模型（继承残差父的近零输出头 + 2400 updates），不能外推为「绝对预测不可行」；三 seed 为一致性证据非显著性；单实例/ROI/季节。预算 per_seed 21600 s、planned 43200 s（软）、hard 86400 s；软超继续并记 overrun，硬截断即 attempt 失败、不算科学通过。test 不读、r=0、S4 不启动、scientific_claim=false。

（objective 字符数：1350；单段。）

## §1 起点与实质差异

- 起点工作分支 `r7/weather-reasoning`；累计 **149.623834 GPU-h**、index **84**。
- 起点证据：`docs/R7_S3_EMA_WEIGHT_AVERAGING_RESULT.md`（negative，plain EMA 支关闭）、
  `docs/R7_S3_ANCHOR_ISOLATION_RESULT.md`（H-D，本轮配对控制 N）、
  `record:s3-wide-interior-dose`（pinned 外部参照 A）。
- **为什么选这个因子**：侦察轮已证明注册模型是 persistence-anchored 残差解码器，因此
  “残差目标”不是独立因子；唯一能干净检验“persistence base 是否承重”的改动，是把 base 换成 0，
  让 head 直接输出 tendency、让 local solver 的 proposal 等于自己的 decoder 输出。它只改一个
  语义开关，不改层数、参数量、输出尺度或最后一层。
- **与已关闭支的区别**：EMA 改的是评分权重来源（训练代码，`model_code_sha256` 不变）；本轮回改
  **模型前向**（`model_code_sha256` 变），因此必须按工程身份门重跑父迁移。架构/容量/损失加权/
  K 轴组织/锚反馈/空间梯度/监督组织各支均已关闭，本轮不触碰。
- **单因素边界**：控制是 pinned 旧代码读数，候选是新代码三 seed 独立轨迹；两臂共享同一父状态、
  同一数据、同一 2400 updates 与同一评分域，但 wall-clock 与代码版本不同（limitations 明写）。

## §2 交付物

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 本轮冻结设计（本文件 §3） | 本文件 |
| D2 | 模型开关 `decoder_base_mode`（默认 `persistence` 逐位保持既有路径） | `model/weather_forecaster_r7.py`、`model/process_forecast_r7.py` |
| D3 | 新父迁移 + 父模块（钉 receipt SHA256、current_model_code_sha256、逐 seed state_digest） | `outputs/r7_s3_decoder_base_anchor_parent_migrated_20261020_attempt02/`、`scripts/r7_s3_decoder_base_anchor_parent.py` |
| D4 | 新驱动 + 支持模块 + prose 模块 + 新 CI-safe 测试（合成树/monkeypatch/反证，不读 outputs/） | `scripts/study_r7_s3_decoder_base_anchor.py`、`scripts/r7_s3_decoder_base_anchor_support.py`、`scripts/r7_s3_decoder_base_anchor_text.py`、`tests/test_r7_s3_decoder_base_anchor.py`、`tests/test_r7_s3_decoder_base_anchor_protocol.py` |
| D5 | 新协议 / 新输出 / code.zip | `preparation_protocol.json`、`protocol.json`、`code.zip` + `archive_receipt.json` |
| D6 | 候选臂三 seed 训练与评分 | `seed{41,42,43}_receipt.json`、每 seed 五 lead 的 `boundary_rmse.csv` |
| D7 | 配对读数 | `readings.json`：候选−控制、候选−pinned dose、控制−pinned dose 桥、17 变量表 |
| D8 | 登记与精确 CI | `docs/R7_S3_DECODER_BASE_ANCHOR_RESULT.md`、index record、brief、账本行、§8、campaign-state |

## §3 判据与设计决定（本轮冻结）

科学门与验收形式只来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` 与 D4 口径；本轮**不新增**
任何数值阈值，也不宣布通过。

- **唯一改动因子**：`decoder_base_mode` 从 `persistence`（控制，pinned spec 缺省）改为 `zero`（候选）。
  `NativeAtmosForecaster` 在解析出 `base = history[:,-1]` 后，仅当 mode 为 `zero` 时把 `base`
  置为 `torch.zeros_like(base)`；同一个张量既作为 head 的直接 anchor，也作为
  `R7ForecastOutput.base_state` 返回，因此 `ProcessForecastCoReasoner` 传给
  `process_reasoning_step` 的 `anchor` 与 `proposal_anchor` 都是 0。
  `model/process_forecast_r7.py` 只把该开关传给 backbone；不新增参数、buffer 或 state-dict key。
  `zero + anchor_output_prior=True + 已装锚` 这个会让 head anchor 与 `base_state` 不同步的组合
  在 backbone 构造与 `_install_climatology_anchor` 两处都被 `ValueError` 拒绝（由本轮冻结 spec 的
  `anchor_output_prior=False` 避开），因此“同步置零”是结构约束而非约定。
- **保持不变**：父权重（v3-BD 1600 + 新模型迁移）、近零输出头（`std=1e-3` 初始化不被 reset）、
  `change_scale_mode`、层数/宽度/输出尺度、2400 updates、LR 2e-5、warmup 10、K=4、depth=4、
  dim=192、batch 1、clip 1、数据/监督掩码/损失/物理权重/seed 41/42/43/评分域/气候态分母逐字不变。
- **惰性气候态锚**：两臂都安装同一份已发布 train-only 表（`outputs/r7_s3_anchor_table_20261015_attempt02`），
  但 `anchor_output_prior=False`、`anomaly_feedback=False`：前向仍会求值锚表（`table.index_select`），
  但其值不进入任一输出；清零锚表必须不改变任一臂（测试断言）。
- **控制臂 N（pinned，不重训练）**：`record:s3-anchor-isolation-result` 的 control 臂
  （`outputs/r7_s3_anchor_isolation_20261015_attempt01/`），spec 中**没有** `decoder_base_mode`
  字段，按默认语义 = persistence。驱动在 prepare 时加载其 `protocol.json`，断言
  `candidate_spec − decoder_base_mode == control_spec` 且 `candidate_spec == control_spec + one key`。
  继承的 `narrow_control` 臂块保留在协议里（未被任何 reading 使用）；候选的 `selection` 文案与
  注册控制一致为 `frozen endpoint; no validation selection`。
- **正/负控制（测试逐条断言，失败即 raise）**：
  1. **省略 `decoder_base_mode` 键**（pinned 控制声明）与显式 `persistence` 的 state_dict/前向
     `torch.equal`；且用模块自身部件重建改前 forward（encoder→lead→spacetime→head(base=history[:,-1])）
     与省略键的默认路径逐位相同——默认路径就是改前实现。
  2. zero 模式 `forecast == tendency`（`torch.equal`）、`base_state` 全 0；在可精确表示的
     base/tendency 上 `persistence.forecast − zero.forecast == base`（`torch.equal`）。
  3. local solver：spy `process_step_r7.anchored_proposal`，persistence 收到 `history[:,-1]`，
     zero 收到全 0；两者之差逐位等于 anchor；proposal 公式 `anchor + Decoder(Z)` 未变。
  4. 两模式 state_dict keys/shapes/参数量相同且张量逐位相同；未知 mode 被 `ValueError` 拒绝；
     `zero + 输出先验 + 已装锚` 被 `ValueError` 拒绝。
  5. 清零惰性锚表：persistence 与 zero 两臂的前向都不变（`torch.equal`）。
  6. 新迁移的 `expanded_state_sha256` 与 `anchor_init_sha256` 必须逐 seed 等于 H-D control receipt
     的 `00ada5…/baed2f…/4dda64…` 与 `de0e76…`（`expand_parent_state` 内断言）。
- **诊断（不作 gate）**：update-0 的 zero-base 训练 loss 应高于 persistence（同一父、同一 batch），
  用于确认初始 handicap 机制；若相反，只如实记录，不改变出口。
- **预声明出口（不新增阈值）**：
  - (a) 候选 72h `interior_32` t2m skill 在 seed 41/42/43 **全部 >0**；
  - (b) 同 seed 候选−控制 72h ΔRMSE **三 seed 全部 <0**（负值 = 候选更优）；
  - (c) 候选 48h skill **三 seed 全部 >0**；
  - (d) 候选−控制在 6/12/24h 的 **seed 均值** ΔRMSE **≤0**；
  - (e) 桥：pinned 控制 N 的 15 个 seed×lead RMSE 对 pinned A 的差落在 `−4.8e-6…+1.8e-5 K`。
  **同时满足 (a)–(e) → 提请独立复核（不自动进 S4）**；任一 (a)–(d) 不满足 → **永久关闭**
  「zero-base decoder」支：不扫 anchor 值（含 base=X_0）、不加 seed、不调 2400 updates/LR/
  head reset/output scale、不与任何已关轴组合、不换名重开。
- **工程身份门**（只算 attempt 失败，修复仅限已确认工程原因，不改因子或判据）：
  - 新 `model_code_sha256` 未预声明或未记录；
  - 父迁移 receipt 缺失、`current_model_code_sha256` 与运行时不符，或迁移后 state_digest
    不等于 `7c416ccb…/857d8804…/61fdac58…`；
  - `decoder_base_mode='persistence'` 默认路径非逐位不变；
  - 候选 spec 与 pinned 控制 spec 除 mode 外有任何差异；
  - 控制/参照被重训或误用；test 被读；
  - `expanded_state_sha256`/`anchor_init_sha256` 与 H-D control receipt 不同。
  - 候选与 pinned 控制的 `training_code_sha256` 不同：协议必须记录两值、`equal` 与活动路径
    （`draft_supervision=deep`、`gradient_lambda=0.0`、`ema_decay=None`）；该差异本身是已知的
    跨轮代码版本差异，不伪装成相同，也不是本轮的通过判据。
  父迁移必须用已提交 `scripts/migrate_r7_parent_state.py` 在新代码下重跑到新的排他目录
  （`outputs/r7_s3_decoder_base_anchor_parent_migrated_20261020_attempt02`），记录
  `migration_receipt.json` 的 SHA256、`current_model_code_sha256` 与逐 seed `state_digest`；
  禁止复用 H-D attempt04 的旧 digest checkpoint。
- **证据页 `docs/R7_S3_DECODER_BASE_ANCHOR_RESULT.md` 必须写**：控制为旧代码 pinned 读数
  （新代码 persistence 路径只做逐位等价断言、未重训控制），比较混合 base 语义与 run-to-run；
  zero-base 不是 from-scratch 绝对模型（继承残差父的近零输出头 + 2400 updates），不能外推为
  「绝对预测不可行」；三 seed 为一致性证据非显著性；单实例/ROI/季节限制。
- **报告口径**：五个 lead（6/12/24/48/72h）与**全 17 变量**的 `interior_32` RMSE 都照报；
  不跨单位平均、不挑赢家。
- **test 策略**：`test.jsonl` 从不读取；`test_read=false`、`scientific_claim=false`。

## §4 预算与停止

- **执行预算（冻结）**：`per_seed_seconds=21600`、`planned_seconds=43200`（软预算）、
  `hard_cap_seconds=86400`（硬上限，为计划的 2 倍）。软超预算继续并记 overrun，硬截止即停
  attempt 并全额保留 `failure.json`；失败 attempt 不在原目录 resume、不缩短重试。
  这是**单臂**候选 envelope；pinned 的 H-D 控制是**双臂** envelope（43200/86400/115200 s）。
  该差异由 `budget_profile_note` 写进协议；只有 hard 截断会影响科学读数，2400 updates 的单臂
  实测余量充足。
- 只训练 **候选一个臂 × 三 seed**（N 与 A 均 pinned 复用），三 seed 由 `stage-seeds` 在双卡上按
  决策 0047 调度；每阶段 spawn 前只读 `gpu_gate`，**禁止**对非本实验进程发送任何信号（R-054）。
- 计划成本约 3 ×（2400 updates + 五 lead 评分）≈ 6–8 GPU-h；实际按回执逐阶段记账。
- **停止条件**：任一 (a)–(d) 不满足即永久关闭该支；工程失败 / 身份门 / hard cap 只算 attempt
  失败，修复仅限已确认工程原因，不改因子或判据；test 未读、r=0、S4 不启动。

## §5 明确不做

不重训练 N 或 A、不重跑已登记轮次、不读 test、不扫 anchor 值、不加 seed、不改 2400 updates/
LR/warmup/K/dim/depth/监督域/物理权重、不 reset 输出头、不改 change_scale 或 output scale、
不与 anchor/feedback/architecture/容量/K 轴组织/损失加权组合、不换名重开、不改
`data/raw|interim|processed` 或归档快照、不合成替代数据、不改科学合同/冻结证据/判据、
不 main 合并/发版/关 issue、不 paid/rental/exclusive/force/mirror/破坏性/cron。

## §6 进度与交接

- **已定**：设计（§3）；登记名 `docs/R7_S3_DECODER_BASE_ANCHOR_RESULT.md` / record
  `s3-decoder-base-anchor-result`。
- **已完成（2026-10-20，实现轮）**：
  1. `model/weather_forecaster_r7.py` 新增 `decoder_base_mode`（默认 `persistence`；
     `zero` 时把直接 head anchor 与 `base_state` 一起置零），`model/process_forecast_r7.py`
     仅传递该开关；`persistence` 路径保持改前操作序列，`model_code_sha256` 变为
     `bec3859f…`。
  2. 用已提交 `scripts/migrate_r7_parent_state.py` 在新代码下重跑到
     `outputs/r7_s3_decoder_base_anchor_parent_migrated_20261020_attempt02`（三个 seed 全 success，
     `semantics_equal=false` 是预期的跨代码语义差异，state_digest 与 export 逐 seed 相同）。
     receipt SHA256 `002f0843…`、逐 seed state_digest `7c416ccb…/857d8804…/61fdac58…`、expanded state digest `00ada5…/baed2f…/4dda64…` 与 anchor_init `de0e76…` 均与 H-D control receipt 相同。
  3. 新驱动 `scripts/study_r7_s3_decoder_base_anchor.py`（505 行）、支持模块
     `scripts/r7_s3_decoder_base_anchor_support.py`（292 行）、父模块
     `scripts/r7_s3_decoder_base_anchor_parent.py`（289 行）、prose 模块（60 行）；
     新 CI-safe 测试 `tests/test_r7_s3_decoder_base_anchor.py` + `_protocol.py`（合成树/monkeypatch/反证，
     不读 `outputs/`；拆成两个模块以守住 R-051）。
  4. 本长文与 campaign-state 指针已写。
  5. 已按独立复核处置：执行文件清单加入实际被 import 的
     `scripts/study_r7_s3_anchor_isolation.py` 与 `scripts/r7_s3_anchor_isolation_parent.py`
     （execution_files 196 项，测试断言 membership）；协议记录候选/控制的
     `training_code_sha256`、`equal` 与活动路径；`dose_readings` 补齐 pinned A 的全 17 变量；
     继承的 `narrow_control` 臂块保留；预算差异写入 `budget_profile_note`；
     `zero + 输出先验 + 已装锚` 被拒绝；`expanded_state_sha256`/`anchor_init_sha256` 逐 seed 钉住。
  6. 实证：`round-guard` **passed**（`172 passed`；37 条阻断规则 0 failing）；在**不含
     `outputs/`** 的临时副本里两个测试文件 **52 passed**（CI-safe 反证）；真实 `_identity()`
     （pinned 控制 spec 逐字段比对、父迁移 receipt pin、pinned dose pin、code-version 记录）与
     seed41 的父迁移装载 + 129×129 zero-base forward（`base_state` 全 0、head 输出 == tendency）
     集成检查通过。`model_code_sha256=bec3859f…`、migration receipt `002f0843…`、逐 seed
     state_digest 与 expanded/anchor_init digest 已核对。
- **运行与登记（2026-10-20）**：实验 commit `a6f7a14`（`R7 CPU CI` **39344273382** green；
  前一提交 `df76556` 的 CI 因测试文件 EOF 空行与尺寸标记漂移为红，机械修复后重建 code.zip）。
  `freeze` → `prepare` → `archive` → 三 seed（8260.966/9265.008/8798.317 s，全 success）→ `reading`
  全 success；attempt `outputs/r7_s3_decoder_base_anchor_20261020_attempt01/`。
- **读数（预声明出口）**：三 seed 2400 updates + 五 lead 全跑完（cohort 472/468/460/444/428）。
  (a) 72h `interior_32` t2m skill **−1.366793/−1.234491/−1.236888**（三 seed 全负）→ **false**；
  (b) 候选−控制 72h Δrmse **+1.508636/+1.210100/+1.394940 K**（三 seed 全正，均值 +1.371225）→ **false**；
  (c) 48h skill **−1.148466/−1.045917/−1.062917**（三 seed 全负）→ **false**；
  (d) 6/12/24h seed 均值 Δ **+2.544271/+1.556526/+1.899833 K**（全为正）→ **false**；
  (e) 桥一致性 N−A 在 **−4.787001e-6…+1.778038e-5 K**（15/15）→ **true**。
  ⇒ `scientific_exit_met=false`、`predeclared_exit_met=false`、`identity_unresolved=false`：
  **按预声明永久关闭「zero-base decoder」支**（不扫 anchor 值、不加 seed、不调 updates/LR/head
  reset/output scale、不与任何已关轴组合、不换名重开）。候选在全部 15 个 seed×lead 单元上
  差于控制；72h skill 低至 −1.37，说明该残差父的近零输出头无法在 2400 updates 内承担绝对
  状态尺度。**zero-base 不是 from-scratch 绝对模型**，不能外推为「绝对预测不可行」。
- **成本**：process 合计 27,596.832 s、**7.665786755 GPU-h**（prepare 0.117911 + archive 0.119331 +
  seed41 2.294713 + seed42 2.573613 + seed43 2.443977 + reading 0.116242）；墙钟 18,347.936 s
  （5.0966 h）；soft/hard overrun 均 0；网络 0；磁盘 19.52 GiB；峰值 8,996,782,080 B =
  8.378 GiB（协议估计 9.5 GiB，高估约 1.12 GiB）；累计 **149.623834 → 157.289621 GPU-h**。
- **登记**：证据页 `docs/R7_S3_DECODER_BASE_ANCHOR_RESULT.md`、record
  `s3-decoder-base-anchor-result`（index85）、账本行、§8 与 campaign-state；**不进 S4 冻结包**。
- **无未做/待做项**：`test` 未读、r=0、S4 未启动。
- **下一动作（S3）**：本轮支已永久关闭；回到 `docs/goals/main-model-climatology-campaign.md` §8，
  基于本轮负结果另选独立因子。`test` 未读、r=0、S4 未启动。
- **恢复第一条命令**：`.venv/bin/python tools/r7_round.py state`。
- 不宣布科学通过、不称泛化/SOTA。
