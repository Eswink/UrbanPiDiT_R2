# S3：EMA 权重平均（单因素）——用 decay=0.999 的滑动平均权重替代第 2400 update 最终权重评分

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-ema-weight-averaging", "driver": "scripts/study_r7_s3_ema_weight_averaging.py", "output_stem": "r7_s3_ema_weight_averaging_20261019", "control_record": "s3-anchor-isolation-result", "reference_record": "s3-wide-interior-dose", "seeds": [41, 42, 43], "device_mode": "dual", "test_path": ["tests/test_r7_s3_ema_weight_averaging.py"], "commit_paths": ["scripts/study_r7_s3_ema_weight_averaging.py", "scripts/r7_s3_ema_weight_averaging_text.py", "tests/test_r7_s3_ema_weight_averaging.py", "training/r7_ema.py", "training/r7_long_rollout_runner.py", "docs/goals/s3-ema-weight-averaging.md", "docs/goals/main-model-climatology-campaign.md", "docs/rules/size-thresholds.md"], "commit_message": "feat(r7): add the EMA weight-averaging single-factor driver", "register_paths": ["docs/R7_S3_EMA_WEIGHT_AVERAGING_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md", "docs/goals/s3-ema-weight-averaging.md"], "register_message": "feat(r7): register the EMA weight-averaging run as a new evidence record", "code_archive": "outputs/r7_s3_ema_weight_averaging_code.zip"} -->

状态 **已跑完并登记（2026-10-20）；预声明出口 (a)–(d) 全未满足，该支永久关闭（negative / not-candidate）**。
证据页 `docs/R7_S3_EMA_WEIGHT_AVERAGING_RESULT.md`、record `s3-ema-weight-averaging-result`。
唯一主计划 `docs/goals/main-model-climatology-campaign.md`；承接训练信号/目标组织方向，接在已关闭的
spatial-gradient/dense-step/draft-organization 三支之后（三支都按预声明关闭）。本轮只检验
「2400 update 端点的残差里是否有一部分是优化噪声、可用同一轨迹的 EMA 权重过滤」这一可证伪假设。

## §0 Objective（可粘贴；单段）

> 在 docs/goals/s3-ema-weight-averaging.md 接续 S3 与已登记的 s3-anchor-isolation-result（配对控制）、s3-wide-interior-dose（外部参照）：在相同当前代码迁移父（attempt04，model_code_sha256 不变、无需新迁移）、seed 41/42/43、2400 updates、K=4、depth=4、dim=192、宽 129×129 输入、interior_32 监督、注册稀疏物理权重 (1,.5,0,.5,0,0,0,.5,0,0,0,.5) 与注册 K 轴深监督 linspace(1,2,5)/sum（final_weight=2）下，唯一改动因子是评分 checkpoint 的权重来源：把第 2400 update 的最终优化器权重换成同一轨迹上 decay=0.999 的 EMA 权重——从父权重初始化、覆盖全部 trainable 参数、每个 optimizer step 后更新、无 bias correction、无 warm-up；前向、损失、数据、AdamW、LR 2e-5、warmup 10、2400 updates、种子、评分域与气候态分母逐字不变。最终优化器权重仍写出但不用于评分。预声明出口（不新增数值阈值）：(a) 候选 72h interior_32 t2m skill 三 seed 全 >0；(b) 同 seed 候选−控制 72h ΔRMSE 三 seed 全 <0；(c) 候选 48h skill 三 seed 全 >0；(d) 候选−控制在 6/12/24h 的 seed 均值 ΔRMSE ≤0；(e) 桥：控制−pinned s3-wide-interior-dose 的 15 个 seed×lead RMSE 差落在 anchor 轮 −4.8e-6…+1.8e-5 K 内。全部 (a)–(e) → 提请独立复核（不自动进 S4）；任一 (a)–(d) 不满足 → 永久关闭 plain EMA decay=0.999 支，不扫 decay、不加 seed、不改 2400 updates、不与其他已关轴组合、不换名重开（BEMA/SWA/checkpoint averaging 也不作为同一支的补救）；(e) 失败只写 identity_unresolved=true，是工程身份失败而非科学判定。test 不读、r=0、S4 不启动、scientific_claim=false。

（objective 字符数：1076；单段。）

## §1 起点与实质差异

- 起点工作分支 `r7/weather-reasoning`；累计 **142.407852 GPU-h**、index **83**。
- 起点证据：`docs/R7_S3_SPATIAL_GRADIENT_LOSS_RESULT.md`（mixed，空间梯度支按预声明关闭）；
  `docs/R7_S3_ANCHOR_ISOLATION_RESULT.md`（H-D mixed，其 control 臂是本轮配对控制）；
  `record:s3-wide-interior-dose`（pinned 外部参照）。
- **为什么选这个因子**：已关闭的三支改的都是**训练信号/目标组织**（空间梯度、K 轴组织、物理步覆盖），
  而注册 2400-update 端点的 72h 误差在三 seed 上稳定为负 skill。一个不同且可独立证伪的解释是：
  端点权重里含有沿轨迹的优化噪声（后期 LR 已 cosine 衰减但仍非零），用同一轨迹的权重平均可以过滤它，
  而不改变任何前向、损失、数据或预算。这是优化/checkpoint 选择方向，不重开已关闭的轴。
- **与已关闭支的区别**：不动损失（`gradient_lambda=0`、`draft_supervision="deep"` 逐字不变）、
  不动模型（`model/*.py` 不动、`model_code_sha256` 不变）、不动 K/dim/depth/监督域/物理权重/
  updates/LR/warmup/data/seeds。唯一差异是评分时读 EMA checkpoint 而不是最终优化器 checkpoint。
- **单因素边界**：候选与控制在评分权重上不同；但两者是**同一父、同 seed、同数据、同 2400-update/
  每臂计划预算的两次独立 GPU 轨迹**（wall-clock envelope 不同：候选 21600/43200/86400 s，双臂控制
  43200/86400/115200 s；控制为 pinned，不在本协议重训练），不是同一轨迹的逐位开关——limitations 明写。

## §2 交付物

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 本轮冻结设计（本文件 §3） | 本文件 |
| D2 | 训练侧 EMA 分支（`ema_decay=None` 默认逐字保持注册报告/合同） | `training/r7_ema.py`、`training/r7_long_rollout_runner.py` |
| D3 | 新驱动 + 新 CI-safe 测试（合成树/monkeypatch/反证，不读 outputs/） | `scripts/study_r7_s3_ema_weight_averaging.py`、`scripts/r7_s3_ema_weight_averaging_text.py`、`tests/test_r7_s3_ema_weight_averaging.py` |
| D4 | 新协议 / 新输出 / code.zip | `preparation_protocol.json`、`protocol.json`、`code.zip` + `archive_receipt.json` |
| D5 | 候选臂三 seed 训练与评分 | `seed{41,42,43}_receipt.json`、每 seed 五 lead 的 `boundary_rmse.csv`；receipt 同时记录 EMA 与最终优化器 checkpoint 的 SHA256 |
| D6 | 配对读数 | `readings.json`：候选−控制、候选−pinned dose、控制−pinned dose 桥、17 变量表 |
| D7 | 登记与精确 CI | `docs/R7_S3_EMA_WEIGHT_AVERAGING_RESULT.md`、index record、brief、账本行、§8、campaign-state |

## §3 判据与设计决定（本轮冻结）

科学门与验收形式只来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` 与 D4 口径；本轮**不新增**
任何数值阈值，也不宣布通过。

- **唯一改动因子**：评分 checkpoint 的权重来源。候选：plain EMA，
  `shadow ← 0.999 * shadow + 0.001 * parameter`，在每个 optimizer step 之后执行；
  `shadow` 初始化为父 checkpoint（模型在 fine-tune 前已 strict-load 父状态）；
  覆盖全部 trainable 参数（runner 已强制所有参数可训练且 FP32）；**无 bias correction**
  （不除以 `1 − 0.999^t`）、**无 warm-up**（第一步即用 0.999）。最终 update-2400 优化器 checkpoint 仍写出。
- **EMA 边界反证（测试）**：`decay=1.0` 永远保持父权重；`decay=0.0` 精确等于最新权重；
  一步后 `shadow = 0.999*parent + 0.001*w1`（不是 `w1`，证明无 bias correction/warm-up）。
- **控制臂 N（pinned，不重训练）**：`record:s3-anchor-isolation-result` 的 control 臂
  （`outputs/r7_s3_anchor_isolation_20261015_attempt01/`），注册深监督、**锚惰性**（控制臂 spec 保留
  `climatology_anchor_spec`/`anchor_output_prior=False`/`anomaly_feedback=False`；H-D 正控制已证明这些键
  未被读取、forward 与无锚模型逐位相同）、`spatial_gradient_lambda` 不存在、评分为 update-2400 最终
  优化器权重。驱动会核对其 receipt/report 顶层不含 EMA 字段（复核已逐层确认实际也无）且 checkpoint
  名为 `update_0002400.pt`。
- **外部参照 A（pinned，不重训练）**：`record:s3-wide-interior-dose`。
- **桥一致性 (e)**：N 的 pinned 5-lead `interior_32` t2m RMSE 对 A 的差必须落在
  `−4.8e-6…+1.8e-5 K`；这是**身份检查**，失败只写 `identity_unresolved=true`，不是科学判据。
- **预声明出口**（不新增阈值，只沿用既有 skill>0 绝对气候态门与配对符号）：
  - (a) 候选 72h `interior_32` t2m skill 在 seed 41/42/43 **全部 >0**；
  - (b) 同 seed 候选−控制 72h ΔRMSE **三 seed 全部 <0**（负值 = 候选更优）；
  - (c) 候选 48h skill **三 seed 全部 >0**；
  - (d) 候选−控制在 6/12/24h 的 **seed 均值** ΔRMSE **≤0**；
  - (e) 桥一致性如上。
  **同时满足 (a)–(e) → 提请独立复核（不自动进 S4）**；任一 (a)–(d) 不满足 → **永久关闭**
  「plain EMA decay=0.999」支：不扫 decay、不加 seed、不调 2400 updates、不与 anchor/feedback/
  architecture/容量/K 轴组织组合、不换名重开；BEMA/SWA/checkpoint averaging 也不作为同一支的补救。
- **工程身份门**（只算 attempt 失败，修复仅限已确认工程原因，不改因子或判据）：EMA checkpoint
  未按冻结常量生成、评分未读 EMA checkpoint、父/模型/数据身份漂移、控制臂被误当 EMA 臂。
- **报告口径**：五个 lead（6/12/24/48/72h）与**全 17 变量**的 `interior_32` RMSE 都照报；
  不跨单位平均、不挑赢家。
- **test 策略**：`test.jsonl` 从不读取；`test_read=false`、`scientific_claim=false`。

## §4 预算与停止

- **执行预算（冻结）**：`per_seed_seconds=21600`、`planned_seconds=43200`（软预算）、
  `hard_cap_seconds=86400`（硬上限，为计划的 2 倍）。软超预算继续并记 overrun，硬截止即停
  attempt 并全额保留 `failure.json`；失败 attempt 不在原目录 resume、不缩短重试。
- 只训练 **候选一个臂 × 三 seed**（N 与 A 均 pinned 复用），三 seed 由 `stage-seeds` 在双卡上按
  决策 0047 调度；每阶段 spawn 前只读 `gpu_gate`，**禁止**对非本实验进程发送任何信号（R-054）。
- 计划成本约 3 ×（2400 updates + 五 lead 评分）≈ 6–8 GPU-h；实际按回执逐阶段记账。
- **停止条件**：任一 (a)–(d) 不满足即永久关闭该支；工程失败 / 身份门 / hard cap 只算 attempt
  失败，修复仅限已确认工程原因，不改因子或判据；test 未读、r=0、S4 不启动。

## §5 明确不做

不重训练 N 或 A、不重跑已登记轮次、不读 test、不扫 decay、不加 seed、不改 2400 updates、不调
K/dim/depth/监督域/物理权重/LR/warmup、不改 K 轴深监督与 final_weight、不与 anchor/feedback/
architecture/容量/K 轴组织组合、不用 BEMA/SWA/checkpoint averaging 补救、不改
`data/raw|interim|processed` 或归档快照、不合成替代数据、不改科学合同/冻结证据/判据、
不 main 合并/发版/关 issue、不 paid/rental/exclusive/force/mirror/破坏性/cron。

## §6 进度与交接

- **已定**：设计（§3）；登记名 `docs/R7_S3_EMA_WEIGHT_AVERAGING_RESULT.md` / record
  `s3-ema-weight-averaging-result`。
- **已完成（2026-10-19，实现轮）**：① `training/r7_ema.py` 新增 plain EMA（无 bias correction、
  无 warm-up），`training/r7_long_rollout_runner.py` 接受 `ema_decay` 并在每个 optimizer step 后更新，
  在最终端点额外发布 `ema_update_0002400.pt`；`ema_decay=None` 时合同/报告逐字保持注册路径；
  ② 新驱动（派生自已登记 spatial-gradient 驱动）与新 CI-safe 测试（合成树/monkeypatch/反证，
  不读 `outputs/`）；③ 本长文与 campaign-state 指针已写。
- **已完成（2026-10-20，运行/登记轮）**：freeze → prepare → archive → seed41/42/43 → reading 全 success；
  三 seed 2400 updates + 五 lead 全跑完（cohort 472/468/460/444/428），attempt01 无 `failure.json`。
  结果：72h skill **−0.266241/−0.321691/−0.171025**（三 seed 全负）→ (a) false；72h 配对 Δ
  **+0.137607/+0.064690/+0.022851 K**（三 seed 全正）→ (b) false；48h skill 非全正 → (c) false；
  6/12/24h seed 均值 −0.029423/−0.026619/+0.018023 K → (d) false；桥 15/15 → (e) true。
  ⇒ 预声明出口未满足，**永久关闭 plain EMA decay=0.999 支**；证据页、index84、账本行与 campaign-state
  已登记。6/12h 三 seed 全改善、24/48/72h 三 seed 全退化；比较口径为 pinned 独立控制，
  候选自己的最终优化器 checkpoint 已写出但未评分（limitation）。
  成本 7.215982213 GPU-h（累计 149.623834），soft/hard overrun 0；实验 CI `39263655966` 绿。
- **未做 / 不再做**：不扫 decay、不加 seed、不调 2400 updates、不与其他已关轴组合、不换名重开；
  BEMA/SWA/checkpoint averaging 不作为补救；test 未读、S4 未启动。
- **下一动作（S3）**：本支关闭；下一轮重新检索一手资料，在仍允许的表示或另一个优化机制方向另写 brief
  并预声明出口；数据扩充按 D3 继续暂缓。
- **恢复第一条命令**：`.venv/bin/python tools/r7_round.py state`。
- test 未读、r=0、S4 未启动；不宣布科学通过、不称泛化/SOTA。
