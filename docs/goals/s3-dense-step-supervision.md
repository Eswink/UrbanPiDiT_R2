# S3：12 个 6h 物理步的**密集直接监督覆盖**（单因素）——把 5 个稀疏 lead 的 support 扩到全部 12 步

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-dense-step-supervision", "driver": "scripts/study_r7_s3_dense_step_supervision.py", "output_stem": "r7_s3_dense_step_supervision_20261018", "control_record": "s3-wide-interior-dose", "paired_control_record": "s3-anchor-isolation-result", "seeds": [41, 42, 43], "device_mode": "dual", "test_path": ["tests/test_r7_s3_dense_step_supervision.py"], "commit_paths": ["scripts/study_r7_s3_dense_step_supervision.py", "tests/test_r7_s3_dense_step_supervision.py", "docs/goals/s3-dense-step-supervision.md", "docs/goals/main-model-climatology-campaign.md"], "commit_message": "feat(r7): add the dense 12-step physical supervision single-factor driver", "register_paths": ["docs/R7_S3_DENSE_STEP_SUPERVISION_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md"], "register_message": "feat(r7): register the dense physical-step supervision run as a new evidence record", "code_archive": "outputs/r7_s3_dense_step_supervision_code.zip"} -->

状态 **CPU 实现与 CI-safe 测试已完成（2026-10-18）；待 commit-round / freeze / GPU**。唯一主计划
`docs/goals/main-model-climatology-campaign.md`；承接 H-D
（`record:s3-anchor-isolation-result`，预声明出口未满足、气候态锚族已关闭），本轮回主线的
**训练信号 / 目标组织**方向，取「直接监督覆盖（coverage）」这一此前从未单因素测过的缺口。

## §0 Objective（单段）

> 在 docs/goals/s3-dense-step-supervision.md 接续 S3、`record:s3-anchor-isolation-result` 之后的训练信号方向：在相同当前代码迁移父（attempt04，`model_code_sha256 3fe747b2…` 不变、无需新迁移）、seed 41/42/43、2400 updates、K=4、宽 129×129 输入、`interior_32` 监督与总物理权重 3.0 下，只把 12 个 6h 物理步的直接监督权重（`physical_weights`）从注册稀疏 5 步轮廓 `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)` 改为 12 步均匀 `(0.25,)*12`，首次把 18/30/36/42/54/60/66h 这 7 个零直接监督过渡步纳入 support；控制臂 N 是 pinned `record:s3-anchor-isolation-result` 的 control 臂（同代码迁移父、同 seed/预算、稀疏权重，不重训练），外部参照 A 是 pinned `record:s3-wide-interior-dose`（不重训练）；预声明出口：同时满足 (a) F 的 72h `interior_32` t2m skill 在 seed 41/42/43 全部 >0、(b) 同 seed F−N 的 72h ΔRMSE 三 seed 全部 <0、(c) F 的 48h skill 三 seed 全部 >0、(d) F−N 在 6/12/24h 的 seed 均值 ΔRMSE ≤0、(e) 桥一致性 N 的 pinned 5-lead RMSE 对注册剂量臂的差落在 anchor 轮已报告的 −4.8e-6…+1.8e-5 K 内 → 提请独立复核/验收；任一 (a)–(d) 不满足 → **永久关闭**「dense physical-step supervision coverage」支（不扫 0.1/0.25 等权重分配、不加 seed、不调 2400 updates、不与 anchor/feedback/architecture/容量组合），按 negative/mixed 如实登记；桥 (e) 失败只算 identity/engineering unresolved，不是科学判定；test 不读、r=0、S4 不启动，不宣布科学通过、不称泛化/SOTA。本 objective 为单段并指向本文件（字符数见 §4 实测）。

## §1 起点与实质差异

- 起点工作分支 `r7/weather-reasoning`；累计 **120.775625 GPU-h**、index **80**（H-D 已登记）。
- 起点证据：`docs/R7_S3_ANCHOR_ISOLATION_RESULT.md`（H-D，mixed：F−N 五 lead 全为配对改善，
  但 72h 绝对 skill 三 seed 未全正，气候态锚族按预声明关闭）；`record:s3-wide-interior-dose`
  （2400 updates / wide / interior_32 / K=4 / dim 192 / seeds 41/42/43，pinned 外部参照）。
- **为什么选这个因子**：注册 recipe 的 12 个 6h 物理步里只有 **5 步**有非零直接目标
  （6/12/24/48/72h，权重 `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)`，sum 3.0）；另外 **7 个过渡步**
  （18/30/36/42/54/60/66h）权重恰为 0，**没有直接目标损失**，只经 full-BPTT 从后续 lead 间接传播。
  文献线索（只作线索，不作证据）：GraphCast arXiv:2212.12794v2 §14.2–14.3 对 12 个自回归步的
  每个 lead 都计算 loss 并对 lead 均匀平均；Stormer arXiv:2312.03876v2 §3.1.2 对 K 步 rollout 取平均；
  FuXi arXiv:2306.12873v3 §3.3.2 沿用 GraphCast 的 2→12 步课程。**未检索到**稀疏 5 步 vs 同总权重
  密集 12 步在同类实例上的系统消融——这正是本轮唯一可证伪的 coverage 缺口。
- **与已关闭的长 lead 加权臂的区别**：`record:s3-wide-long-lead-weight` 只把 48/72h 从 0.5 提到
  1.0、总损失 3.0→4.0，测的是**既有已监督 lead 的 emphasis**；本轮总权重固定 3.0、48/72h 权重
  反而从 0.5 降到 0.25，测的是**coverage**，不是长 lead 加权重开。
- **与 H-D 的区别**：H-D 改推理循环的反馈表征（`anomaly_feedback`），本轮完全不碰模型前向；
  模型、`training/*.py`、父迁移、数据、K、预算、seed、评分域与气候分母逐字不变，只改训练目标的
  权重向量，因此 `model_code_sha256` / `training_code_sha256` 不变、**无需新迁移**。

## §2 交付物

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 本轮冻结设计（本文件 §3） | 本文件 |
| D2 | 新驱动 + 新测试 | `scripts/study_r7_s3_dense_step_supervision.py`、`tests/test_r7_s3_dense_step_supervision.py` |
| D3 | 新协议 / 新输出 / code.zip | `prepared_protocol.json`、`protocol.json`、`code.zip` + `archive_receipt.json` |
| D4 | 候选臂 F 三 seed 训练与评分 | `seed{41,42,43}_receipt.json`、每 seed 五 lead 的 `boundary_rmse.csv` |
| D5 | 配对读数 | `readings.json`：F−N（pinned control）、F−A（pinned dose）、N−A 桥 |
| D6 | 登记与精确 CI | `docs/R7_S3_DENSE_STEP_SUPERVISION_RESULT.md`、index record、brief、账本行、§8、campaign-state |

## §3 判据与设计决定（本轮冻结）

科学门与验收形式只来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` 与 D4 口径；本轮**不新增**
任何数值阈值，也不宣布通过。

- **唯一改动因子**：训练目标中 12 个 6h 物理步的直接监督权重向量。控制向量
  `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)`（sum 3.0，5 个直接监督步，7 个零权重过渡步）逐字来自
  `scripts/study_r7_s3_wide_single_factor.py:38`；候选向量 `(0.25,)*12`（sum 3.0，12 个直接监督步）。
  两个向量都由 `training.r7_long_rollout.validate_physical_weights` 接受（有限非负、非空、至少一个正）。
- **控制臂 N（pinned，不重训练）**：`record:s3-anchor-isolation-result` 的 control 臂
  （`outputs/r7_s3_anchor_isolation_20261015_attempt01/`）。它是同一当前代码迁移父 attempt04、
  同 seed/预算、稀疏权重；其 `anomaly_feedback=False` 且 `anchor_output_prior=False`，锚不被任何
  前向路径读取，故在行为上就是注册的无锚模型（该轮实测 `unanchored_max_abs=0.0`）。
- **外部参照 A（pinned，不重训练）**：`record:s3-wide-interior-dose`
  （`outputs/r7_s3_wide_interior_dose_20261009_attempt02/`）。
- **桥一致性 (e)**：N 的 pinned 5-lead `interior_32` t2m RMSE 对 A 的差必须落在 anchor 轮已报告的
  `−4.8e-6…+1.8e-5 K`；这是**身份检查**，失败只写 `identity_unresolved=true`，不是科学判据。
- **预声明出口**（不新增阈值，只沿用既有 skill>0 绝对气候态门与配对符号）：
  - (a) F 的 72h `interior_32` t2m skill 在 seed 41/42/43 **全部 >0**；
  - (b) 同 seed F−N 的 72h ΔRMSE **三 seed 全部 <0**（负值 = F 更优）；
  - (c) F 的 48h skill **三 seed 全部 >0**（D4 既有绝对门）；
  - (d) F−N 在 6/12/24h 的 **seed 均值** ΔRMSE **≤0**（不允许短 lead 退化换长 lead）；
  - (e) 桥一致性如上。
  **同时满足 (a)–(e) → 提请独立复核/验收**；任一 (a)–(d) 不满足 → **永久关闭**
  「dense physical-step supervision coverage」支：不扫 0.1/0.25 等步权重分配、不加 seed、
  不调 2400 updates、不与 anchor/feedback/architecture/容量组合，按 negative/mixed 如实登记。
- **报告口径**：五个 lead（6/12/24/48/72h）与**全 17 变量**的 `interior_32` RMSE 都照报；48h 单独
  列出但不单独作门；不跨单位平均、不挑赢家。
- **test 策略**：`test.jsonl` 从不读取；`test_read=false`、`scientific_claim=false`。

## §4 预算与停止

- **执行预算（冻结）**：`per_seed_seconds=21600`、`planned_seconds=43200`（软预算）、
  `hard_cap_seconds=86400`（硬上限，为计划的 2 倍）；软超预算继续并记 overrun，硬截止即停 attempt
  并全额保留 `failure.json`；失败 attempt 不在原目录 resume、不缩短重试。
- 只训练 **F 一个臂 × 三 seed**（N 与 A 均 pinned 复用），三 seed 由 `stage-seeds` 在双卡上按
  决策 0047 调度；每阶段 spawn 前只读 `gpu_gate`，**禁止**对非本实验进程发送任何信号（R-054）。
- 计划成本约 3 ×（2400 updates + 五 lead 评分）≈ 6–8 GPU-h；实际按回执逐阶段记账。
- **停止条件**：任一 (a)–(d) 不满足即永久关闭该支；工程失败 / 身份门 / hard cap 只算 attempt
  失败，修复仅限已确认工程原因，不改因子或判据；test 未读、r=0、S4 不启动。

## §5 明确不做

不重训练 N 或 A、不重跑已登记轮次、不读 test、不扫权重分配、不加 seed、不改 2400 updates、不调
K/dim/depth/监督域、不与 anchor/feedback/architecture/容量组合、不改 `data/raw|interim|processed`
或归档快照、不合成替代数据、不改科学合同/冻结证据/判据、不 main 合并/发版/关 issue、
不 paid/rental/exclusive/force/mirror/破坏性/cron。

## §6 进度与交接

- **已定**：设计（§3）；登记名 `docs/R7_S3_DENSE_STEP_SUPERVISION_RESULT.md` / record
  `s3-dense-step-supervision-result`（成功或失败同一命名）。
- **已完成（2026-10-18）**：① 新驱动 `scripts/study_r7_s3_dense_step_supervision.py`（单臂 F，
  复用 attempt04 父 pin 与 anchor-parent 模块，600 行，R-051 通过）；② 新测试
  `tests/test_r7_s3_dense_step_supervision.py`（30 passed；在**无 `outputs/`** 的 clean checkout 中
  同样 30 passed，含权重向量/支持集/总和反证、协议篡改 6 例反证、`decision_flags` 逐项失败反证、
  全 17 变量解析反证、CPU smoke 接受 12 步密集权重）；③ `round-guard` 通过
  （150 passed，commit-guard 37 条阻断规则 0 失败）；④ 真实 `_identity()` 只读 dry-run 通过
  （model/training digest 未变、paired-control protocol 与 index 一致、parent pins 三 seed 齐全）；
  ⑤ 父 checkpoint 在无锚当前模型上 `strict=True` 加载通过（3,286,037 参数）。
- **待做**：`commit-round` → 重建 code zip → `freeze` → `stage prepare/archive/stage-seeds/reading`
  → 四件套登记与精确 CI。
- **父迁移前提**：本轮不改 `model/*.py`，当前 `model_code_sha256=3fe747b2…` 与 attempt04 迁移
  回执一致，**不需要**新迁移；驱动仍逐 seed 复核迁移回执与 checkpoint SHA，不符即 `raise`。
- **CI 安全**：上一轮因测试依赖 `outputs/` 让 CI 变红；本轮测试已在 `git archive HEAD` 的
  无 `outputs/` clean checkout 中实跑 30 passed，且不读 test.jsonl。
- **第一条命令**：`.venv/bin/python tools/r7_round.py round-guard` → 重建 zip →
  `.venv/bin/python tools/r7_round.py freeze` → `stage prepare/archive/stage-seeds/reading`。
- **本轮尚未开跑**：0 GPU-h；无读数、无结论。

## §7 下一动作（S3）

- CPU 实现与测试已完成。下一步：`round-guard`（已通过）→ `commit-round` → 重建 code zip →
  `freeze` → 逐 stage GPU → 四件套登记。
- 预声明出口见 §3；本轮**未产生任何读数**，不宣布通过、不关闭任何支。
