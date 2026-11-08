# S3 Stage E1b：final-only 草稿监督 vs 注册 deep 形态（16 epoch 制度），三 seed 配对

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-terminal-step-supervision-e1b", "code_archive": "outputs/r7_s3_e1b_terminal_step_supervision_code.zip", "commit_message": "feat(r7): freeze Stage E1b terminal-step draft supervision as a single factor", "commit_paths": ["scripts/study_r7_s3_e1b_terminal_step_supervision.py", "scripts/r7_s3_e1b_terminal_step_supervision_protocol.py", "tests/test_r7_s3_e1b_terminal_step_supervision.py", "docs/goals/s3-throughput-physics-e1b.md", "docs/decisions/0096-stage-e1-training-signal-reopen-under-16-epoch.md"], "control_record": null, "device_mode": "dual", "driver": "scripts/study_r7_s3_e1b_terminal_step_supervision.py", "hard_cap_seconds": 1843200.0, "output_stem": "r7_s3_e1b_terminal_step_supervision_20261108", "planned_seconds": 921600.0, "register_message": "docs(r7): register Stage E1b terminal-step supervision readings", "register_paths": ["docs/R7_S3_E1B_TERMINAL_STEP_SUPERVISION_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md", "docs/goals/s3-throughput-physics-e1b.md"], "seeds": [41, 42, 43], "test_path": ["tests/test_r7_s3_e1b_terminal_step_supervision.py", "tools/check_driver_attributes.py"]} -->

本文件是 Stage E1b 的**唯一执行源**；机器可读的 `round-spec` 在上方。
上游是不可变记录：`docs/goals/s3-throughput-and-physics-campaign.md`（Stage E1）、
决策 **0096**（设计与三种结局的预声明）、**0057**（重开队列的「不是同一族」要求与新制度）。

**`scientific_claim: false`、`test_read: false`、`r = 0`。**

---

## §1 Objective（这一轮问的是什么）<!-- 实测 1287 字符 -->

> 在 `/data/esw/UrbanPiDiT_R2` 的 `r7/weather-reasoning` 上跑 S3 Stage E1b：判据与相位以本文件 `docs/goals/s3-throughput-physics-e1b.md` 为准、设计依据是决策 0096 与 0057。单因子 on/off、三 seed 配对（41/42/43）、treatment 清单（32,131 窗）、batch 固定 4、注册 recipe（K=4/depth=4/dim=192/LR 2e-5/warmup 10/12 物理步全 BPTT），**恰好 128,528 updates = 16.000498 epoch**，从 `59340710…` 修订下的迁移父出发。唯一差别是**草稿监督的组织方式**：off 臂用注册 `deep`（`linspace(1,2,5)/sum`，73.333% 的权重落在非最终草稿上），on 臂用 `final_only`（只监督 `drafts[:, -1]`、权重 1.0）；**12 步物理权重向量两臂共享且逐字钉为注册稀疏向量**（这是与 E1a 的分界，两者不得合并成一轮）。预声明出口 (a)–(d)：72 h t2m skill 相对**同代码身份** off 臂的配对均值不退化、on 臂**回读自训练报告**的 `draft_supervision` 等于 `final_only`、off 臂是注册 `deep` 逐字且物理权重未被移动、无判据被改；**任一假即永久关闭「final-only 草稿监督」这一支**，不调权重、不加 seed；已关闭旧臂读数是 `mixed`，故「再给一点算力」不是可用辩解。桥 (e) 逐 seed 断言父身份（`model_code_sha256` + 155 张量 `state_digest`）。**已实跑通过**（不是推断）：`final_only` 用**真实** `deep_supervised_forecast_mse` 实测非最终四张草稿梯度**精确为 0**、只有最后一张动（0.156995），`deep` 五张全动；`driver.freeze` 端到端自校验通过（`epochs = 16.000498`）、单测 11 项全过。预算按 Stage B 四点**最慢实测** 1.1740 s/update 重算：训练 `150,892 s` + 评分 `2,588.1 s` = `153,480 s ≈ 42.63 GPU-h`/arm，两臂 × 三 seed ⇒ **约 255.8 GPU-h**（**估计**）；包络 `per_seed = planned = 921,600 s`（256 h）、`hard = 1,843,200 s`。`scientific_claim: false`、`test_read: false`、`r = 0`，不读 2023 test、不重建 store、不改任何冻结判据、不自宣科学通过或 goal 完成。

## §2 冻结的配方（与 E1a 同制度，唯一差别是草稿监督组织）

- **实例**：`outputs/r7_s3_fy_multiyear_20261023_attempt01`（65×65 多年，17 通道）
- **数据**：treatment manifest，**32,131 可用窗**、11 条年界排除
- **训练**：`128,528 updates × batch 4` ⇒ **16.000498 epoch**
- **K=4、depth=4、dim=192、LR 2e-5、warmup 10、weight_decay 1e-4、clip 1.0、FP32**
- **草稿监督**：off = `deep`（注册）；on = `final_only`
- **物理权重**：两臂共享 `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)`，协议断言**未被移动**
- **评分**：整张 65×65（`boundary_margins=(1,)`、`full` 4,225 点），五个 lead，全部 17 个变量
- **气候态分母**：48 桶 train-only 的 D2 reference；**只有 t2m 有注册分母**
- **父**：`outputs/r7_s3_position_retention_parent_migrated_20261103_attempt01/`，逐 seed 断言
- **checkpoint**：每 5,000 步 + 端点（决策 0087）
- **同步守卫**：全模型有限性每 100 步 + 每 checkpoint + 端点（决策 0086）

## §3 交付物清单

| # | 交付物 | 证据形态 |
| --- | --- | --- |
| D1 | 本轮冻结协议 | `outputs/r7_s3_e1b_terminal_step_supervision_20261108_attempt*/protocol.json` |
| D2 | 代码身份归档 | `outputs/r7_s3_e1b_terminal_step_supervision_code.zip`（带 commit 注释） |
| D3 | 逐 seed receipt | `seed{41,42,43}_receipt.json`（两臂各一份，含监督模式回读与父身份） |
| D4 | 十七变量 × 五 lead 读数 | `readings.json`（含 `exit_conditions`） |
| D5 | 证据页 | `docs/R7_S3_E1B_TERMINAL_STEP_SUPERVISION_RESULT.md` |
| D6 | 四件套登记 | index record / 账本行 / campaign §8 / `campaign-state` |

## §4 判据：预声明出口（冻结后不放宽）

**判据上游**：`docs/goals/s3-throughput-and-physics-campaign.md` Stage E 与决策 **0096**；
本文件不发明新阈值。

- **(a)** 72 h t2m skill 相对**同代码身份**的 off 臂**不退化**（三 seed 配对差的均值 ≥ 0）；
- **(b)** 单一因子**确实生效**：on 臂**回读自训练报告**的 `draft_supervision` 等于 `final_only`；
- **(c)** off 臂是注册 `deep` 形态**逐字**（同样从训练报告回读），且**物理权重向量未被移动**；
- **(d)** 无判据被改动、`test_read is False`、`scientific_claim: False`、`r = 0`。

**(a)–(d) 全真 ⇒ 提请独立复核。** **任一假 ⇒ 永久关闭「final-only 草稿监督」这一支**，
不调权重、不换项、不加 seed。**特别地**：已关闭的旧臂读数是 `mixed`（不是 null），
所以「再给它一点算力」**不是**可用的辩解——这一点在开局前就写进决策 0096。

**桥 (e)**：父身份逐 seed 断言（`model_code_sha256` + 155 张量 `state_digest`，
由 `_parent_model` 在**每次训练前**实跑）。
失败记 `identity/engineering unresolved`，**不进入** (a)–(d)。

**因子为何是「组织」而不是「剂量」**：`final_only` 把 **4/5 的草稿**从梯度里**精确**移出
（实测每草稿梯度 `[0,0,0,0,0.156995]`），`deep` 则五张全动（`[0.021,0.028,0.031,0.037,0.042]`）。
这不是「权重更小」，而是**被监督的集合不同**，故与 E1a 的分配变更是两个因子。

## §5 预算与停止

| 项 | 值 | 依据 |
| --- | --- | --- |
| 训练时间估计 | **150,892 s** | `128,528 × 1.1740`（Stage B 四点中**最慢**实测） |
| 评分余量 | **2,588.1 s** | 第四点五 lead 的**实测**耗时 |
| 合计 | **153,480 s ≈ 42.63 GPU-h**/arm | 上两者之和 |
| 两臂 × 三 seed | **约 255.8 GPU-h** | **估计**；不是实测 |
| `per_seed_seconds` | 921,600 s（256 h） | `ceil(3 × 两臂估计 / 3600) × 3600` |
| `planned_seconds` | 921,600 s | 与包络相同 |
| `hard_cap_seconds` | 1,843,200 s（512 h） | 包络的两倍 |

软预算超了继续并记 `soft_overrun`；硬上限截断 attempt（记 `budget_limited`、全额记账、**不算通过**）。

**停止三条**：触保留授权边界；达已冻结停止条件；确无可安全独立推进的工作。

## §6 明确不做

- 不重建 store、不新建 129×129 源、不动 `data/raw|interim|processed` 与归档快照
- 不读 2023 test（`r = 0`、`test_read: false`）
- 不改本轮任何判据；不事后放宽 (a)–(d)
- **不与 E1a 合并**（两个因子合一轮会让赢家不可归因，同 0092/0095 的否决理由）
- **不移动物理权重向量**（那会把本轮变成 E1a 的重复）
- 不重开 SAM / zero-base decoder / plain EMA / 长 lead 权重 / rollout 剂量
- 不 paid / rental / exclusive / cron / 守护 / 会话外续跑；不对邻居进程发任何信号
- 不自宣科学通过或 goal 完成

## §7 进度与交接

**（执行者在此追加进度块；停时必须写明「为何停 + 恢复第一条命令」。）**

- 2026-11-08 建立：状态 `code-ready`，**未冻结、未开跑，0 GPU-h**。

  **已实测（不是推断）**：
  - `tools/check_driver_attributes.py` 对驱动与协议 exit 0；
  - `final_only` 的梯度切割用**真实** `deep_supervised_forecast_mse` 实测：
    非最终四张草稿梯度**精确为 0**、只有 `drafts[:,-1]` 动（0.156995），
    而 `deep` 五张全动；反证测试在**禁用该分支**后失败、恢复后通过；
  - `driver.freeze(...)` 在临时目录**端到端实跑**，`protocol_sha256` 自校验通过，
    `recipe.epochs = 16.000498`、`scientific_claim: false`、`test_read: false`、`r: 0`；
  - `tests/test_r7_s3_e1b_terminal_step_supervision.py` **11 项全过**。

  **一处已修的致命缺陷（0 GPU-h 发现）**：驱动原本**没有 `freeze` 入口**，
  被 `load_driver` 的 `DRIVER_REQUIRED_NAMES` 拒绝（`RoundError: ... does not expose ['freeze']`）。
  修法同 E1a 的三行转接，并新增反证测试（移除转接 ⇒ 以同一条生产错误失败）。

  **恢复第一条命令** = `.venv/bin/python tools/r7_round.py state`；
  然后需先把 `campaign-state.current_round_goal` 指向本文件，再
  `freeze` → `prepare` → `archive` → `stage-seeds` → `reading`。
  **本轮不启动**：C2 的 `attempt08` 正在跑（钉 95 文件与 `model_code_sha256`），
  R-054 要求 GPU 串行，且 E1b 的训练前置是 C2 结束（决策 0096 的 `执行前置`）。

  > **2026-11-09 更正**：此前这里写的是「`attempt06` 正在跑」。**`attempt06` 已在决策 0105
  > 中作废**（目标函数跨 lead，23.684603 GPU-h 全额记账）；当时的在跑轮次是
  > **`attempt08`**（决策 0107 重开、修正后配对）。指令本身不变，只是轮次名更新。

## §8 下一动作（S3）

1. 等 `attempt08` 六臂跑完 → `stage reading` → C2 的 (a)–(d) 求值 → 四件套登记。
2. 之后由用户/复核者裁定是否把 `current_round_goal` 切到 E1a 或本文件。
3. E1a 与 E1b **各自一轮**，顺序不可合并；先跑哪个由用户/复核者定。
