# S3 Stage B 第三点（4 epoch = 32,132 updates）：学习曲线的第三个纵坐标

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-throughput-physics-b-p4000", "driver": "scripts/study_r7_s3_throughput_physics_b_p4000.py", "output_stem": "r7_s3_throughput_physics_b_p4000_20261105", "control_record": null, "seeds": [41], "device_mode": "dual", "test_path": ["tests/test_r7_s3_curve_large_points.py"], "planned_seconds": 122400.0, "hard_cap_seconds": 244800.0, "code_archive": "outputs/r7_s3_throughput_physics_b_p4000_code.zip", "commit_paths": ["scripts/r7_s3_curve_point_protocol.py", "scripts/r7_s3_curve_point_driver.py", "scripts/study_r7_s3_throughput_physics_b_p4000.py", "tests/test_r7_s3_curve_large_points.py", "docs/goals/s3-throughput-physics-b-p4000.md", "docs/decisions/0090-stage-b-point-three-envelope-must-be-derived.md", "docs/decisions/0089-stage-b-point-two-shared-curve-engine.md", "docs/decisions/README.md", "docs/goals/main-model-climatology-campaign.md"], "commit_message": "feat(r7): run Stage B point 3 (4 epoch / 32,132 updates) at batch 4", "register_paths": ["docs/R7_S3_THROUGHPUT_PHYSICS_B_P4000_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md", "docs/goals/s3-throughput-physics-b-p4000.md"], "register_message": "docs(r7): register Stage B point 3 (4 epoch) readings"} -->

本文件是 Stage B 第三点的**唯一执行源**；机器可读的 `round-spec` 在上方。
上游是不可变记录：`docs/goals/s3-throughput-and-physics-campaign.md`（六 Stage 与判据）、
`docs/plans/0028-throughput-physics-campaign.md`（设计与全部实测数字）、
决策 **0084**（方向扩围与判据拆分）、**0086**（同步守卫）、**0087**（checkpoint 策略）、
**0088**（第一点设计）、**0089**（第二点与共享引擎）、**0090**（本点的包络推导与并列实例）。

**`scientific_claim: false`、`test_read: false`、`r = 0`。**

---


## §1 Objective（这一轮问的是什么）<!-- 实测 1194 字符 -->

> 在 `/data/esw/UrbanPiDiT_R2` 的 `r7/weather-reasoning` 上跑 S3 Stage B 的**第三点**，判据与相位以本文件 `docs/goals/s3-throughput-physics-b-p4000.md` 为准、设计依据是 `docs/goals/s3-throughput-and-physics-campaign.md` 与 `docs/plans/0028-throughput-physics-campaign.md`：单臂、单 seed 41、treatment 清单（32,131 窗）、batch 固定 4、注册 recipe（K=4/depth=4/dim=192/LR 2e-5/warmup 10/12 物理步全 BPTT），恰好 **32,132 updates（4 epoch）**，从 `59340710…` 修订下的迁移父出发，跑完在整张 65×65 上对全部 17 个变量 × 5 个 lead（6/12/24/48/72 h）评分。这是定标曲线的**第三个纵坐标**，不是因子比较、不是显著性检验、也不能单独判定饱和——饱和规则是「16 epoch 相对 4 epoch 的 72 h skill 一阶差分」，本点只把那条规则逐字记进协议并令 `saturation_verdict` 为 `null`。预声明出口是工程性的 (a)–(d)：精确 updates/batch、五个 lead 全部在场且 17 变量全有限、t2m 72 h skill 分母为注册 D2 值、无判据被改，任一假即该点作废、从头重跑（**无断点续跑，尤其不从前两点续**）。桥 (e) 逐 seed 断言父身份（`model_code_sha256` + 155 张量 `state_digest`），失败只记 `identity/engineering unresolved`、不进入 (a)–(d)。预算由第二点的**实测** 1.1505 s/update 与 2,697.3 s 评分推导：训练估计 `32,132 × 1.1505 = 36,967.866 s`、`point_seconds_estimate 39,665.166 s ≈ 11.02 GPU-h`，包络 `per_seed = planned = 122,400 s`、`hard = 244,800 s`（按决策 0090 由估计推导，**不继承第二点的 36,000 s**，否则本点会被时长截断）；`scientific_claim: false`、`test_read: false`、`r = 0`，不读 2023 test、不重建 store、不改任何冻结判据、不自宣饱和或未饱和、不自宣 Stage B 或 goal 完成。

## §2 冻结的配方（与前两点逐字相同，只有 updates 不同）

- **实例**：`outputs/r7_s3_fy_multiyear_20261023_attempt01`（65×65 多年，17 通道）
- **数据**：treatment manifest，**32,131 可用窗**、11 条年界排除（`window_sha256 e8f22c30…`）
- **训练**：`32,132 updates × batch 4 = 128,528` 样本 ⇒ **4.000124 epoch**（标签写 4 epoch）
- **K=4、depth=4、dim=192、LR 2e-5、warmup 10、weight_decay 1e-4、clip 1.0、FP32**
- **物理权重** `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)`（12 个 6 h 物理步，全 BPTT）
- **评分**：整张 65×65（`boundary_margins=(1,)`、`full` 4,225 点），五个 lead 6/12/24/48/72 h，
  **全部 17 个变量**（逐字取自 store 的 `channels`）
- **气候态分母**：48 桶 train-only 的 D2 reference；**只有 t2m 有注册分母**，
  其余 16 个变量报原始 RMSE、**不给 skill 数**（不发明分母）
- **父**：`outputs/r7_s3_position_retention_parent_migrated_20261103_attempt01/`，
  当前 `model_code_sha256 59340710…`，155 张量 `state_digest` 逐 seed 断言等于钉值
- **checkpoint**：每 5,000 步 + 端点（决策 0087）；本点跨 6 个边界 ⇒ **7 个**
- **同步守卫**：全模型有限性每 100 步 + 每 checkpoint + 端点（决策 0086）

## §3 交付物清单

| # | 交付物 | 证据形态 |
| --- | --- | --- |
| D1 | 本点冻结协议 | `outputs/r7_s3_throughput_physics_b_p4000_20261105_attempt01/protocol.json` |
| D2 | 代码身份归档 | `outputs/r7_s3_throughput_physics_b_p4000_code.zip`（带 commit 注释） |
| D3 | 逐 seed receipt | `seed41_receipt.json`（含父身份、7 个 checkpoint、loss 摘要、线程数、s/update） |
| D4 | 五点十七变量读数 | `readings.json`（`leads[6\|12\|24\|48\|72][rmse\|skill]`） |
| D5 | 证据页 | `docs/R7_S3_THROUGHPUT_PHYSICS_B_P4000_RESULT.md` |
| D6 | 四件套登记 | index record / 账本行 / campaign §8 / `campaign-state` |

## §4 判据：预声明出口（冻结后不放宽）

**本轮的出口是工程性的，不是科学性的。** 一个纵坐标不能判定饱和，而把「这个点跑完了」
写成科学结论正是本轮要避免的错误。判据与相位的上游是
`docs/goals/s3-throughput-and-physics-campaign.md` §3（饱和判据原文）、
`docs/plans/0028-throughput-physics-campaign.md` §2 Stage B（设计与成本口径）与
`docs/decisions/0090-stage-b-point-three-envelope-must-be-derived.md`（包络推导与 `(b)` 收紧），
三者都是不可变记录；本文件不发明新阈值。

- **(a)** 实跑到**精确** `32,132` updates、batch `4`；
- **(b)** **五个 lead 全部在场**，每一档上**全部 17 个变量**的 `full` 区域 RMSE **有限**，
  五个 cohort 与冻结的 `VAL_COHORTS` 逐字相符（缺一档即假，见决策 0090 Context ④）；
- **(c)** 72 h 的 t2m skill **有限**，且分母是注册的 D2 值（不是自造的）；
- **(d)** 没有任何冻结判据被改动（`protocol_sha256` 自校验、`test_read is False`、17 变量清单未变）。

**(a)–(d) 全真 ⇒ 该点作为曲线的一个纵坐标**可用**，可进入下一点。**
**任一假 ⇒ 该点作废**，如实登记为工程失败并**从头重跑**（无断点续跑，**不从前两点续**）。

**桥 (e)，非科学判定**：本点的父身份**逐 seed 断言**——`load_checkpoint` 的
`model_code_sha256` 检查 + 155 张量 `state_digest == 钉值`。失败记
`identity/engineering unresolved`，**不进入** (a)–(d)，也不得据此判读任何 skill 数。
（本轮没有零 query 聚合那类逐位判据，因此**不涉及**决策 0082 的 float32 界问题。）

**饱和判定不在这里做。** 规则（`Δ = skill_72h(16e) − skill_72h(4e)`；`Δ > 0.01` 未饱和、
`Δ < 0.005` 且低于上一档饱和、其余只记录不外推）**逐字记进 `protocol.json` 的
`saturation_rule`**，`readings.json` 里 `saturation_verdict` 为 `null`。
**任何本轮的输出都不得引作饱和与否的证据。**

## §5 预算与停止

| 项 | 值 | 依据 |
| --- | --- | --- |
| 训练时间估计 | **36,967.866 s** | `32,132 × 1.1505 s/update`（第二点**实测**） |
| 评分余量 | **2,697.3 s** | 第二点五 lead 的**实测**耗时 |
| `point_seconds_estimate` | **39,665.166 s ≈ 11.02 GPU-h** | 上两者之和 |
| `per_seed_seconds` | **122,400 s** | `ceil(3 × estimate / 3600) × 3600` |
| `planned_seconds` | 122,400 s | 与包络相同 |
| `hard_cap_seconds` | 244,800 s | 包络的两倍；**只有硬上限因时长截断** |
| 峰值显存估计 | 9,126 MiB | 第二点**实测** reserved peak |

软预算超了继续并记 `soft_overrun`；硬上限截断 attempt（记 `budget_limited`、全额记账、
**不算通过**）。**不在运行中改硬限。** 无总 GPU-h 上限（决策 0030/0038）。

**停止三条**：触保留授权边界；达已冻结停止条件；确无可安全独立推进的工作。

## §6 明确不做

- 不重建 store、不新建 129×129 源、不动 `data/raw|interim|processed` 与归档快照
- 不读 2023 test（`r = 0`、`test_read: false`）
- 不改本轮任何判据；不事后放宽 (a)–(d)
- **不从前两点的 checkpoint 续跑**（三个点是各自独立的 attempt）
- 不把本点当作 Stage B 的结论；不据此声明饱和或未饱和
- 不重开 SAM / zero-base decoder / plain EMA / 长 lead 权重 / rollout 剂量
- 不 paid / rental / exclusive / cron / 守护 / 会话外续跑；不对邻居进程发任何信号
- 不自宣 Stage B 完成、不自宣科学通过或 goal 完成

## §7 进度与交接

**（执行者在此追加进度块；停时必须写「为何停 + 恢复第一条命令」。）**

- 2026-11-05 建立：状态 `prepared`，未冻结、未开跑，0 GPU-h。

## §8 下一动作（S3）

冻结后依次 `prepare` → `archive` → `seed 41` → `reading`，然后做四件套登记。
跑完后下一动作（S3）是**第四点**（16 epoch = 128,528 updates），仍按本点实测重算预算、另立 brief + ADR。
