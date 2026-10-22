# S3：Sharpness-Aware Minimization（单因素训练期更新规则）——检验 72 h interior_32 t2m skill 与外生 (a)–(e) 出口

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-sharpness-aware-minimization", "driver": "scripts/study_r7_s3_sharpness_aware_minimization.py", "output_stem": "r7_s3_sharpness_aware_minimization_20261021", "control_record": "s3-anchor-isolation-result", "reference_record": "s3-wide-interior-dose", "seeds": [41, 42, 43], "device_mode": "dual", "test_path": ["tests/test_r7_s3_sharpness_aware_minimization.py", "tests/test_r7_s3_sharpness_aware_minimization_protocol.py"], "code_archive": "outputs/r7_s3_sharpness_aware_minimization_code.zip", "commit_paths": ["training/r7_sam.py", "training/r7_long_rollout_runner.py", "scripts/study_r7_s3_sharpness_aware_minimization.py", "scripts/r7_s3_sharpness_aware_minimization_parent.py", "scripts/r7_s3_sharpness_aware_minimization_support.py", "scripts/r7_s3_sharpness_aware_minimization_text.py", "tests/test_r7_s3_sharpness_aware_minimization.py", "tests/test_r7_s3_sharpness_aware_minimization_protocol.py", "docs/goals/s3-sharpness-aware-minimization.md", "docs/goals/main-model-climatology-campaign.md", "docs/decisions/0051-sharpness-aware-minimization-round.md", "docs/decisions/README.md", "docs/rules/size-thresholds.md"], "commit_message": "feat(r7): add the sharpness-aware-minimization single-factor driver", "register_paths": ["docs/R7_S3_SHARPNESS_AWARE_MINIMIZATION_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md", "docs/goals/s3-sharpness-aware-minimization.md"], "register_message": "feat(r7): register the sharpness-aware-minimization run as a new evidence record"} -->

状态 **进行中：实现轮**（2026-10-21）。唯一主计划
`docs/goals/main-model-climatology-campaign.md`；本轮承接 2026-10-21 §8 重新激活块（就 SAM
这一优化因子）与决策 `0051-sharpness-aware-minimization-round.md`，只改训练期更新规则，
不改模型层、输入表示、spec、父、数据、source、损失定义或任何已关轴。

## §0 Objective（可粘贴；单段）

> 在 docs/goals/s3-sharpness-aware-minimization.md 接续 S3/index87/172.818261 GPU-h、2023 test 未评分 r0、S4 未启动：在冻结的 2400-update wide/interior_32 long-rollout recipe 上只改一个训练期因子 sam_rho——从 None（注册单步 result.loss.backward()）改为预声明的 0.05（SAM 双 forward/backward 扰动更新，arXiv:2010.01412v3 §3.1 的 p=2 对偶解）：先用注册标量损失的全局梯度 g 把全部可训练参数扰动 e = 0.05·g/(‖g‖+1e-12)，在扰动点重算同一注册损失并取其梯度作为更新梯度，把保存的原参数逐位 copy_ 回来，再走注册 clip_grad_norm_(clip=1) 与 optimizer.step()；模型层、输入表示、spec（逐字等于 pinned H-D 控制 spec，无 tendency_input、无 decoder_base_mode）、父、数据/source、损失定义、2400 updates、LR 2e-5、warmup 10、K=4、depth=4、dim=192、batch 1、clip 1、deep K 轴监督、物理权重 (1,.5,0,.5,0,0,0,.5,0,0,0,.5)、seed 41/42/43、评分域 interior_32 与气候态分母全部逐字不变，参数个数不变；因为训练代码改变，training_code_sha256 由 8ad77d93… 变为新 revision，model_code_sha256 仍是 60de3b2d…（本轮不改 model/*.py），可复用既有 committed 父迁移（receipt 73c019dd…），扩张在控制 spec 下只新增已发布的两个 anchor buffer，逐 seed expanded digest 变为 00ada5e2…/baed2ff9…/4dda647f…；对照 record 为 pinned s3-anchor-isolation-result（注册更新规则、final optimizer weights，不重训），外部参照为 pinned s3-wide-interior-dose；预声明出口（不新增阈值，与 docs/goals/s3-tendency-input.md:89-97 逐字同形）：(a) 候选 72h interior_32 t2m skill 三 seed 全 >0；(b) 同 seed 候选−pinned 控制 72h ΔRMSE 三 seed 全 <0；(c) 候选 48h skill 三 seed 全 >0；(d) 候选−控制在 6/12/24h 的 seed 均值 ΔRMSE ≤0；(e) 桥：pinned 控制对 pinned s3-wide-interior-dose 的 15 个 seed×lead RMSE 差落在 −4.8e-6…+1.8e-5 K；全真→candidate_state needs-review 并提请独立复核、不自动进 S4；任一 (a)–(d) 假→永久关闭 SAM 支（不扫 rho、不改扰动定义/范数/eps/符号、不加 seed、不改 updates/LR/warmup/K/depth/dim/clip、不做 parameter-matched 臂、不与已关轴组合、不换名重开）；(e) 假只写 identity_unresolved=true，是工程身份失败而非科学判定；null/inert 仅在 SAM 支每个 update 的扰动恰为零（e 恒 0，更新与注册路径逐位相同）或每个配对 Δ 在记录精度下恰为 0.0 时成立，小而非零的效应按 (a)–(d) 判读、不得改标 null；预算 per-seed 21600 s、planned 43200 s（软）、hard 86400 s，软超继续并记 overrun，硬截断即 attempt 失败不算科学通过；test 不读、r=0、S4 不启动、scientific_claim=false。

（objective 字符数：1808；单段。）

## §1 起点与实质差异

- 起点工作分支 `r7/weather-reasoning`；累计 **172.818261 GPU-h**、index **87**。
- 起点证据：`docs/R7_S3_TENDENCY_INPUT_RESULT.md`（mixed/not-candidate，支关闭）、
  `docs/R7_S3_ANCHOR_ISOLATION_RESULT.md`（配对控制 N，registered gradient、final weights）、
  `record:s3-wide-interior-dose`（pinned 外部参照 A）、
  `docs/goals/main-model-climatology-campaign.md` §8 的 2026-10-21 重新激活块。
- **为什么选这个因子**：主计划的「训练信号/优化」方向。上两轮（梯度手术、显式 tendency 输入）
  都改的是**反向的分组**或**输入通路**；SAM 改的是**更新点本身**——它不增加参数、不增加输入、
  不改损失，只把注册的单步更新换成预声明的双 forward/backward 扰动更新。这是与所有已关闭轴
  都不重叠的一条独立检验。
- **一手检索（线索，不作证据）**：Foret et al., *Sharpness-Aware Minimization for Efficiently
  Improving Generalization*, **arXiv:2010.01412v3** §2（p=2 对偶解：ε̂(w) 即「把梯度重标到范数 ρ」）
  与 §3.1（默认 ρ=0.05，且该文自己**给非 SAM 基线两倍 epoch** 以对齐算力）；
  SAMformer, **arXiv:2402.10198v3** Table 4（最接近的时序先例：最优 ρ*=0.1–1.0，用 Adam 而非
  AdamW）。检索只作线索、不作证据；本轮**不**扫 rho。
- **单因素边界**：控制是 pinned 旧训练代码读数（`s3-anchor-isolation-result` 的 control 臂），
  候选是新训练代码三 seed；两臂共享同一父 state、同一数据、同一 2400 updates 与同一评分域，
  但 training-code revision 与 model-code revision 都不同（limitations 明写）。
- **不是已关轴的重命名**：不改容量（dim/depth）、不改架构开关、不改损失/监督/权重、不改输入
  表示、不改反向分组、不改数据制度；只改优化器更新步。

## §2 交付物

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 本轮冻结设计（本文件 §3） | 本文件、`docs/decisions/0051-sharpness-aware-minimization-round.md` |
| D2 | `sam_rho` 更新规则模块（`require_sam_rho` / `require_sam_axes` / `sam_backward` / `summarize_updates`，off 路径不导入） | `training/r7_sam.py` |
| D3 | 训练器加性开关：`fine_tune_long_rollout(sam_rho=...)`、`_contract`/`_sam_semantics`、`_update` 的独立 SAM 分支 | `training/r7_long_rollout_runner.py` |
| D4 | 控制 spec 下的父扩张 pin（复用既有迁移 receipt；重算 expanded digest）与合成更新规则反证 | `scripts/r7_s3_sharpness_aware_minimization_parent.py` |
| D5 | 新驱动 + 支持模块 + prose 模块（逐字保留预算/权重/seed/父/评分域/气候分母） | `scripts/study_r7_s3_sharpness_aware_minimization.py`、`scripts/r7_s3_sharpness_aware_minimization_support.py`、`scripts/r7_s3_sharpness_aware_minimization_text.py` |
| D6 | 新 CI-safe 测试（合成树/monkeypatch/反证，不读 `outputs/`） | `tests/test_r7_s3_sharpness_aware_minimization.py`、`tests/test_r7_s3_sharpness_aware_minimization_protocol.py` |
| D7 | 新协议 / 新输出 / code.zip | `preparation_protocol.json`、`protocol.json`、`code.zip` + `archive_receipt.json` |
| D8 | 候选臂三 seed 训练与评分、配对读数 | `seed{41,42,43}_receipt.json`、每 seed 五 lead 的 `boundary_rmse.csv`、`readings.json` |
| D9 | 登记与精确 CI | `docs/R7_S3_SHARPNESS_AWARE_MINIMIZATION_RESULT.md`、index record、brief、账本行、§8、campaign-state |

## §3 判据与设计决定（本轮冻结）

科学门与验收形式只来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` 与 D5 口径；本轮
**不新增**任何数值阈值，也不宣布通过。

- **唯一改动因子**：`sam_rho` 从 `None`（控制，注册单步 `result.loss.backward()`）改为
  `0.05`（候选）。
  - 扰动：`e = rho · g / (‖g‖ + eps)`，`eps = 1e-12`，**一次全局标量**重标（`‖g‖` 是对全部
    可训练参数的全局范数，float64 累加）；不做逐参数归一化。
  - 上升方向就是注册标量损失的梯度本身：`w → w + e`。
  - 在扰动点**重算同一注册损失**并取其梯度作为更新梯度（第二次 forward/backward；第一次的
    图已被 `backward()` 释放，两张图不同时存在；第一次的 `.grad` 在第二次 backward 前清空，
    不叠加）。
  - 恢复：保存的原参数用 `copy_` 逐位写回（不是 `w − e`，浮点上不可逆）。
  - 之后**照走注册** `clip_grad_norm_(clip=1, error_if_nonfinite=True)` 与 `optimizer.step()`。
  - 每 update 两次 forward/backward ⇒ FLOPs/GPU-h 约 2×（见 limitations ②）。
- **保持不变**：模型层与 spec（**逐字等于** pinned H-D 控制 spec，无 `tendency_input`、
  无 `decoder_base_mode`）、persistence decoder base、数据/source/split、损失定义、
  2400 updates、LR 2e-5、warmup 10、K=4、depth=4、dim=192、batch 1、clip 1、deep K 轴
  supervision、物理权重、seed 41/42/43、评分域 `interior_32`、气候态分母、五 lead 与 17 变量
  报告口径。**参数个数不变**。
- **父与身份**：本轮不改 `model/*.py`，`model_code_sha256` 保持 `60de3b2d…`，因此既有
  committed 父迁移（`scripts/migrate_r7_parent_state.py` 在上一 revision 下重跑到
  `outputs/r7_s3_tendency_input_parent_migrated_20261021_attempt02`，receipt `73c019dd…`）
  可复用；逐 seed 迁移 state digest 仍是 `7c416ccb…/857d8804…/61fdac58…`。**扩张不同**：
  上一轮 target spec 带 `tendency_input=True`，扩张态多一个零 tendency patch；本轮 target
  spec 是控制 spec，扩张**只**新增两个已发布 anchor buffer，故 expanded digest 重算并 pin 为
  `00ada5e2…/baed2ff9…/4dda647f…`，anchor init 仍 `de0e76…`。
- **控制臂 N（pinned，不重训练）**：`record:s3-anchor-isolation-result` 的 control 臂
  （`outputs/r7_s3_anchor_isolation_20261015_attempt01/`），spec **逐字等于**本轮
  `control_spec()`，`model_code_sha256=3fe747b2…`、`training_code_sha256=8ad77d93…`、
  final optimizer weights。候选 spec 必须与它**逐位相等**（不是「加一个开关」）。
- **正/负控制（测试逐条断言，失败即 raise）**：
  1. `sam_rho=None` 时 `_update` 走**独立**的注册分支，梯度与 `result.loss.backward()` 逐位
     相等，metrics 不含任何 `sam_*` 字段。
  2. `sam_rho=0.0` 时扰动恰为零、恢复逐位、更新梯度与注册路径逐位相等（证明 SAM 分支是注册
     路径的真推广，而不是另一条路径）。
  3. 更新梯度等于**扰动点**的梯度：在独立重建的 `w + e` 上重做 forward/backward 逐位比对；
     并断言它既不是第一次的梯度、也不是「第一次+第二次」的叠加（反证不叠加）。
  4. 参数在 `optimizer.step()` 之前逐位恢复（记录式 optimizer 抓 `step()` 时刻的参数）。
  5. 反号（`w − e`）给出不同的更新梯度；逐参数归一化给出不同的扰动方向。
  6. 注册 clip 仍作用在 SAM 梯度上（`clip=1e-6` 后存储的梯度范数为 1e-6）。
  7. `sam_rho` 与梯度手术不可组合；闭轴（`final_only`/`gradient_lambda>0`/`ema_decay`）被拒；
     声明半径与实际不符被拒；注册路径声明了 `sam_rho` 被拒。
  8. CPU smoke：合成 fixture 的 2-update `fine_tune_long_rollout` 接受 `sam_rho=0.05`，
     report/checkpoint 契约携带 `sam_rho`；`sam_rho=None` 的 report 逐字不含 SAM 字段。
- **预声明出口（不新增阈值）**：
  - (a) 候选 72h `interior_32` t2m skill 在 seed 41/42/43 **全部 >0**；
  - (b) 同 seed 候选−控制 72h ΔRMSE **三 seed 全部 <0**（负值 = 候选更优）；
  - (c) 候选 48h skill **三 seed 全部 >0**；
  - (d) 候选−控制在 6/12/24h 的 **seed 均值** ΔRMSE **≤0**；
  - (e) 桥：pinned 控制 N 的 15 个 seed×lead RMSE 对 pinned A 的差落在 `−4.8e-6…+1.8e-5 K`。
  **同时满足 (a)–(e) → 提请独立复核（不自动进 S4）**；任一 (a)–(d) 不满足 → **永久关闭
  SAM 支**：不扫 rho、不改扰动定义/范数/eps/符号、不加 seed、不改
  updates/LR/warmup/K/depth/dim/clip、不做 parameter-matched 臂、不与已关轴组合、不换名重开。
- **Null/inert**：只有两种精确情形记 null——SAM 支**每个 update** 的扰动恰为零（梯度恰为 0，
  `e ≡ 0`，更新与注册路径逐位相同），或每个配对 Δ 在记录精度下恰好 0.0。小而非零的效应按
  (a)–(d) 判读，**不得**改标 null。
- **工程身份门**（只算 attempt 失败，修复仅限已确认工程原因，不改因子或判据）：
  `model_code_sha256 == 60de3b2d…`；父 migration receipt `73c019dd…` 与逐 seed checkpoint/state/
  expanded/anchor init digest 匹配；控制/参照 pins 匹配且未重训；候选 spec **逐字等于**控制
  spec；数据/source/windows/supervision/气候态分母 pins 匹配；GPU assignment/gates 合法；
  test 2023 未读；r=0；S4 未启动；`scientific_claim=false`。
- **证据页 `docs/R7_S3_SHARPNESS_AWARE_MINIMIZATION_RESULT.md` 必须写**：逐 seed 72h skill 与
  对控制 delta、48h skill、6/12/24h seed 均值 delta、桥、全 17 变量 × 5 lead、逐 seed 的 SAM
  更新摘要（updates、zero_perturbation_updates、perturbation_norm min/max/mean、grad_norm_mean、
  loss_increase min/max/mean、restore_residual_max）与 update-0 未扰动损失、checkpoint/parent/
  protocol/code 身份、实际 GPU-h/wall/peak/disk/network，以及 §3 末列出的全部 limitations
  （不豁免，逐条抄入 `LIMITATIONS`）。
- **报告口径**：五个 lead（6/12/24/48/72h）与全 17 变量的 `interior_32` RMSE 都照报；
  不跨单位平均、不挑赢家；参数/FLOPs/GPU-h 分报。
- **test 策略**：`test.jsonl` 从不读取；`test_read=false`、`scientific_claim=false`。

## §4 预算与停止

- **执行预算（冻结，逐字沿用上一轮单臂 envelope）**：`per_seed_seconds=21600`、
  `planned_seconds=43200`（软预算）、`hard_cap_seconds=86400`（硬上限，为计划的 2 倍）。
  软超预算继续并记 overrun，硬截止即停 attempt 并全额保留 `failure.json`；失败 attempt 不在
  原目录 resume、不缩短重试。这是**单臂**候选 envelope；控制与参照均 pinned 不重训。
- **算力风险的如实记录**：SAM 每 update 两次 forward/backward，按上一轮实测（三 seed 2400
  updates + 五 lead 评分合计 7.805404837 GPU-h）估算本轮约 **2× ≈ 15–16 GPU-h**，单 seed 约
  5–5.5 GPU-h = 18000–20000 s，逼近 21600 s 的 per-seed 硬界。若某 seed 触及 per-seed 或
  hard cap，按预声明只算 **attempt 失败**（工程失败），不改因子、不改判据、不加时。
- 只训练 **候选一个臂 × 三 seed**，由 `stage-seeds` 在双卡上按决策 0047 调度；每阶段 spawn
  前只读 `gpu_gate`，**禁止**对非本实验进程发送任何信号（R-054）。
- **前置**：决策 `0051` 与 §8 重新激活块必须在 freeze 前已提交；`test` 未读、r=0、S4 不启动。
- **停止条件**：任一 (a)–(d) 不满足即永久关闭该支；null/inert 同样关支（按结果类型登记）；
  工程失败 / 身份门 / hard cap 只算 attempt 失败，修复仅限已确认工程原因，不改因子或判据。

## §5 明确不做

不重训练 N 或 A、不重跑已登记轮次、不读 test、**不扫 rho**、不改扰动定义/范数/eps/符号、
不做 parameter-matched 臂、不加 seed、不改 2400 updates/LR/warmup/K/dim/depth/clip/监督域/
物理权重、不 reset 输出头、不改 change_scale 或 output scale、不引入随机顺序 RNG、不与
anchor/feedback/architecture/容量/K 轴组织/损失加权/EMA/空间梯度/梯度手术/tendency 输入/
PE/scale/typed/ordering/rollout-dose 组合、不换名重开、不改 `data/raw|interim|processed`
或归档快照、不合成替代数据、不改科学合同/冻结证据/判据、不 main 合并/发版/关 issue、
不 paid/rental/exclusive/force/mirror/破坏性/cron。

## §6 进度与交接

- **已定**：设计（§3）；登记名 `docs/R7_S3_SHARPNESS_AWARE_MINIMIZATION_RESULT.md` / record
  `s3-sharpness-aware-minimization-result`；output stem
  `r7_s3_sharpness_aware_minimization_20261021`。
- **已完成（2026-10-21，实现轮）**：
  1. `training/r7_sam.py` 新增 SAM 更新规则；`training/r7_long_rollout_runner.py` 加
     `sam_rho` 加性开关（`sam_rho=None` 时注册分支逐字不变，off 路径不导入任何 SAM 语义）。
  2. 控制 spec 下的父扩张 pin：复用既有迁移 receipt `73c019dd…`，重算并 pin 逐 seed
     expanded digest `00ada5e2…/baed2ff9…/4dda647f…`（只多两个 anchor buffer），
     `scripts/r7_s3_sharpness_aware_minimization_parent.py`。
  3. 新驱动/支持/prose 模块与两个 CI-safe 测试文件（合成树/monkeypatch/反证，不读 `outputs/`）。
  4. 本长文、决策 0051、§8 重新激活块、campaign-state 指针、size-thresholds 复算。
  5. 回归：既有 `tests/test_r7_s3_gradient_surgery*.py`、`tests/test_r7_s3_tendency_input*.py`、
     `tests/test_r7_s3_ema_weight_averaging.py` 共 163 passed（off 路径未变）。
- **未做（如实）**：未读 2023 test、未启动 S4、未做 parameter-matched 臂、未跑 4800-update
  算力匹配控制、未扫 rho；未把任何结果称为泛化或 SOTA。
- **已跑完（2026-10-22，结果轮）**：`attempt01` 因父模块缺 `ANCHOR_TABLE_RUN` 再导出全部 seed
  失败（目录保留不删）；修复后重新冻结为 `attempt02`（协议 `b7ae03e1…`、code commit `23f9af3`、
  code.zip `fd572377…`），六阶段全 success、soft/hard overrun 均 0、process 13.195922 GPU-h、
  墙钟 8.0188 h、峰值 9,024,045,056 B、网络 0、磁盘 19.52 GiB。三 seed 各 2400 更新全部施加非零
  扰动（`zero_perturbation_updates` 0/0/0、`restore_residual_max` 0.0）。**读数：15/15 个
  seed×lead 配对 ΔRMSE 全为正（候选更差，+0.0963…+0.2229 K）**；72 h skill 三 seed 全负
  （−0.267285/−0.351110/−0.227842）；48 h skill −0.017313/−0.107794/+0.009830；
  6/12/24 h seed 均值 Δ +0.119104/+0.172023/+0.098197 K；桥 15/15 通过。⇒ 预声明出口
  (a)(b)(c)(d) 全部未满足、(e) 通过 ⇒ **按预声明永久关闭 SAM 支**。已登记证据页
  `docs/R7_S3_SHARPNESS_AWARE_MINIMIZATION_RESULT.md`、index record
  `s3-sharpness-aware-minimization-result`（`negative`/`not-candidate`）、brief、账本行与 §8。
- **下一动作（S3）**：SAM 支已关闭，且 15/15 一致变差 ⇒ 不再在 ρ/eps/seed 上追加尝试。
  下一独立单因素轮必须先做一手检索再提假设；若检索不到与已关轴不重叠的新可证伪因子，
  应把结论落到「当前 recipe 的瓶颈不在这些单因素上」，转向长 lead 误差增长与目标/监督结构的
  更基础假设。
- **恢复第一条命令**：`.venv/bin/python tools/r7_round.py state`。
- 不宣布科学通过、不称泛化/SOTA。
