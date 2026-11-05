# S3 Stage B 第四点（16 epoch = 128,528 updates）：学习曲线的最后一个纵坐标

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-throughput-physics-b-p1600", "driver": "scripts/study_r7_s3_throughput_physics_b_p1600.py", "output_stem": "r7_s3_throughput_physics_b_p1600_20261106", "control_record": null, "seeds": [41], "device_mode": "dual", "test_path": ["tests/test_r7_s3_curve_large_points.py"], "planned_seconds": 453600.0, "hard_cap_seconds": 907200.0, "code_archive": "outputs/r7_s3_throughput_physics_b_p1600_code.zip", "commit_paths": ["scripts/study_r7_s3_throughput_physics_b_p1600.py", "tests/test_r7_s3_curve_large_points.py", "docs/goals/s3-throughput-physics-b-p1600.md", "docs/decisions/0093-stage-b-point-four-final-ordinate.md", "docs/decisions/README.md", "docs/goals/main-model-climatology-campaign.md"], "commit_message": "feat(r7): run Stage B point 4 (16 epoch / 128,528 updates) at batch 4", "register_paths": ["docs/R7_S3_THROUGHPUT_PHYSICS_B_P1600_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md", "docs/goals/s3-throughput-physics-b-p1600.md"], "register_message": "docs(r7): register Stage B point 4 (16 epoch) readings"} -->

本文件是 Stage B 第四点的**唯一执行源**；机器可读的 `round-spec` 在上方。
上游是不可变记录：`docs/goals/s3-throughput-and-physics-campaign.md`（六 Stage 与判据）、
`docs/plans/0028-throughput-physics-campaign.md`、决策 **0084**、**0086**、**0087**、**0088**、**0089**、
**0090**（包络推导与并列实例）、**0091**（短抽样缺陷）、**0093**（本点设计）。

**`scientific_claim: false`、`test_read: false`、`r = 0`。**

---


## §1 Objective（这一轮问的是什么）<!-- 实测 1281 字符 -->

> 在 `/data/esw/UrbanPiDiT_R2` 的 `r7/weather-reasoning` 上跑 S3 Stage B 的**第四点**，判据与相位以本文件 `docs/goals/s3-throughput-physics-b-p1600.md` 为准、设计依据是 `docs/goals/s3-throughput-and-physics-campaign.md` 与 `docs/plans/0028-throughput-physics-campaign.md`：单臂、单 seed 41、treatment 清单（32,131 窗）、batch 固定 4、注册 recipe（K=4/depth=4/dim=192/LR 2e-5/warmup 10/12 物理步全 BPTT），恰好 **128,528 updates（16 epoch）**，从 `59340710…` 修订下的迁移父出发，跑完在整张 65×65 上对全部 17 个变量 × 5 个 lead（6/12/24/48/72 h）评分。这是定标曲线的**最后一个纵坐标**，也是**饱和判据的另一端**——`Δ = skill_72h(16e) − skill_72h(4e)` 的 4e 端已由第三点给出（`+0.230544`），16e 端只有本点能给；但**判定不在本点内做**，由 `tools/r7_s3_saturation_verdict.py` 在两点齐备后按 §3 规则求值。预声明出口是工程性的 (a)–(d)：精确 updates/batch、五个 lead 全部在场且每档 17 变量全有限、t2m 72 h skill 分母为注册 D2 值、无判据被改，任一假即该点作废、从头重跑（**无断点续跑，尤其不从前三点续**）。桥 (e) 逐 seed 断言父身份（`model_code_sha256` + 155 张量 `state_digest`），失败只记 `identity/engineering unresolved`、不进入 (a)–(d)。**预算双记录**：按第三点**实测** 1.1536 s/update 推得训练 `148,269.901 s` + 评分 `2,667.1 s` = **150,937.001 s ≈ 41.93 GPU-h**；冻结协议里记的是长点引擎钉的第二点速率 1.1505 推出的 `150,569 s ≈ 41.82 GPU-h`，包络 `per_seed = planned = 453,600 s`（126 h）、`hard = 907,200 s`——两个估计**并列记录、各带出处**，不互相冒充，0.27% 的差落在实测运行间方差内且不影响任何判据；`scientific_claim: false`、`test_read: false`、`r = 0`，不读 2023 test、不重建 store、不改任何冻结判据、不自宣饱和或未饱和、不自宣 Stage B 或 goal 完成。

## §2 冻结的配方（与前三点逐字相同，只有 updates 不同）

- **实例**：`outputs/r7_s3_fy_multiyear_20261023_attempt01`（65×65 多年，17 通道）
- **数据**：treatment manifest，**32,131 可用窗**、11 条年界排除（`window_sha256 e8f22c30…`）
- **训练**：`128,528 updates × batch 4 = 514,112` 样本 ⇒ **16.000498 epoch**（标签写 16 epoch）
- **K=4、depth=4、dim=192、LR 2e-5、warmup 10、weight_decay 1e-4、clip 1.0、FP32**
- **物理权重** `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)`（12 个 6 h 物理步，全 BPTT）
- **评分**：整张 65×65（`boundary_margins=(1,)`、`full` 4,225 点），五个 lead，**全部 17 个变量**
- **气候态分母**：48 桶 train-only 的 D2 reference；**只有 t2m 有注册分母**，其余 16 个报原始 RMSE
- **父**：`outputs/r7_s3_position_retention_parent_migrated_20261103_attempt01/`，
  `model_code_sha256 59340710…`，155 张量 `state_digest` 逐 seed 断言等于钉值
- **checkpoint**：每 5,000 步 + 端点（决策 0087）；本点跨 25 个边界 ⇒ **26 个**
- **同步守卫**：全模型有限性每 100 步 + 每 checkpoint + 端点（决策 0086）

## §3 交付物清单

| # | 交付物 | 证据形态 |
| --- | --- | --- |
| D1 | 本点冻结协议 | `outputs/r7_s3_throughput_physics_b_p1600_20261106_attempt01/protocol.json` |
| D2 | 代码身份归档 | `outputs/r7_s3_throughput_physics_b_p1600_code.zip`（带 commit 注释） |
| D3 | 逐 seed receipt | `seed41_receipt.json`（含父身份、26 个 checkpoint、loss 摘要、线程数、s/update） |
| D4 | 五点十七变量读数 | `readings.json` |
| D5 | 证据页 | `docs/R7_S3_THROUGHPUT_PHYSICS_B_P1600_RESULT.md` |
| D6 | 四件套登记 | index record / 账本行 / campaign §8 / `campaign-state` |

## §4 判据：预声明出口（冻结后不放宽）

**本轮的出口是工程性的，不是科学性的。** 判据与相位的上游是
`docs/goals/s3-throughput-and-physics-campaign.md` §3、`docs/plans/0028-throughput-physics-campaign.md` §2
与 `docs/decisions/0093-stage-b-point-four-final-ordinate.md`；本文件不发明新阈值。

- **(a)** 实跑到**精确** `128,528` updates、batch `4`；
- **(b)** **五个 lead 全部在场**，每档上**全部 17 个变量**的 `full` 区域 RMSE **有限**，
  五个 cohort 与冻结的 `VAL_COHORTS` 逐字相符；
- **(c)** 72 h 的 t2m skill **有限**，且分母是注册的 D2 值；
- **(d)** 没有任何冻结判据被改动。

**(a)–(d) 全真 ⇒ 该点作为曲线的最后一个纵坐标**可用**。**
**任一假 ⇒ 该点作废**，如实登记为工程失败并**从头重跑**。

**桥 (e)**：父身份逐 seed 断言（`model_code_sha256` + 155 张量 `state_digest`）。
失败记 `identity/engineering unresolved`，**不进入** (a)–(d)。

**饱和判定不在这里做。** 规则逐字记进 `protocol.json` 的 `saturation_rule`，
`readings.json` 的 `saturation_verdict` 为 `null`。本点跑完后由
`tools/r7_s3_saturation_verdict.py` 用第三点与本点的 readings 求值。
**任何本点的输出都不得被单独引作饱和与否的证据。**

## §5 预算与停止

| 项 | 值 | 依据 |
| --- | --- | --- |
| 训练时间估计（brief 口径） | **148,269.901 s** | `128,528 × 1.1536`（**第三点实测**） |
| 评分余量 | **2,667.1 s** | 第三点五 lead 的**实测**耗时 |
| 合计（brief 口径） | **150,937.001 s ≈ 41.93 GPU-h** | 上两者之和 |
| 冻结协议里的估计 | 150,569 s ≈ 41.82 GPU-h | 长点引擎钉的第二点速率 1.1505（见 ADR 0093） |
| `per_seed_seconds` | 453,600 s（126 h） | `ceil(3 × 150,569 / 3600) × 3600` |
| `planned_seconds` | 453,600 s | 与包络相同 |
| `hard_cap_seconds` | 907,200 s（252 h） | 包络的两倍 |
| 峰值显存估计 | 9,126 MiB | 第二点**实测** reserved peak |

软预算超了继续并记 `soft_overrun`；硬上限截断 attempt（记 `budget_limited`、全额记账、**不算通过**）。

**停止三条**：触保留授权边界；达已冻结停止条件；确无可安全独立推进的工作。

## §6 明确不做

- 不重建 store、不新建 129×129 源、不动 `data/raw|interim|processed` 与归档快照
- 不读 2023 test（`r = 0`、`test_read: false`）
- 不改本轮任何判据；不事后放宽 (a)–(d)
- **不从前三点的 checkpoint 续跑**（四点是各自独立的 attempt）
- **不在本点内计算饱和判定**
- 不重开 SAM / zero-base decoder / plain EMA / 长 lead 权重 / rollout 剂量
- 不 paid / rental / exclusive / cron / 守护 / 会话外续跑；不对邻居进程发任何信号
- 不自宣 Stage B 完成、不自宣科学通过或 goal 完成

## §7 进度与交接

**（执行者在此追加进度块；停时必须写「为何停 + 恢复第一条命令」。）**

- 2026-11-06 建立：状态 `prepared`，未冻结、未开跑，0 GPU-h。

## §8 下一动作（S3）

冻结后依次 `prepare` → `archive` → `seed 41` → `reading`，然后做四件套登记。
**跑完后下一动作（S3）**：用 `tools/r7_s3_saturation_verdict.py --low <第三点 attempt> --high <本点 attempt>`
按 §3 的规则求 `Δ = skill_72h(16e) − skill_72h(4e)` 并登记判定结果——**三种结果都必须原样登记**，
包括「饱和」这一与原始设想相反的结论。
