# S3：主干容量（深度）单因素——训练配方支关闭后的第一个容量臂

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-capacity-depth", "driver": "scripts/study_r7_s3_capacity_depth.py", "output_stem": "r7_s3_capacity_depth_20261012", "control_record": "s3-wide-interior-dose", "seeds": [41, 42, 43], "test_path": ["tests/test_r7_s3_capacity_depth.py"], "commit_paths": ["scripts/study_r7_s3_capacity_depth.py", "tests/test_r7_s3_capacity_depth.py", "docs/goals/s3-capacity-or-architecture.md"], "commit_message": "feat(r7): implement the S3 capacity-depth (trunk depth 4->8) single-factor arm", "register_paths": ["docs/R7_S3_CAPACITY_DEPTH_RUN.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md"], "register_message": "feat(r7): register the S3 capacity-depth (depth 4->8) run as a new evidence record", "code_archive": "outputs/r7_s3_capacity_depth_code.zip"} -->

状态 **驱动与 CI-safe 测试已实现、已决定恢复运行**（2026-10-12 操作者决定）。唯一主计划
`docs/goals/main-model-climatology-campaign.md`；承接 `docs/goals/s3-reasoning-depth.md` 的预声明出口
（该臂被操作者中止、未完成，见 `docs/R7_S3_REASONING_DEPTH.md`）与主计划 §8 最后一条"下一动作"。

**登记口径（2026-10-12 决定改回后新增）**：本因子此前曾被登记为**未运行**
（`docs/R7_S3_CAPACITY_DEPTH.md`、`record:s3-capacity-depth`、0 GPU-h），依据是上一轮 K4→8 臂唯一
完成的 seed 41「效果基本上没变化」。**那条依据跨因子、不成立**——`K`（每步内部推理次数，参数量不变）
与主干 `depth`（`WindowAttentionBlock` 层数，参数量 3,286,037 → 5,065,493，+54.2%）是两个不同的轴。
故本轮恢复运行，读数登记为**新** record `s3-capacity-depth-run` 与新证据页
`docs/R7_S3_CAPACITY_DEPTH_RUN.md`；原「未运行」record 与证据页**逐字节保留**，只在其上追加
`correction_record_id` 指向新 record。

本轮实现（2026-10-12）：驱动 `scripts/study_r7_s3_capacity_depth.py`、测试
`tests/test_r7_s3_capacity_depth.py`（28 例全过）。**冻结的深度扩张规则**：depth-4 父状态按
identity-gated 规则扩成 depth-8 —— 新增块 `i∈{4..7}` 在**每个参数**上镜像块 `i-4`，并把两个
输出投影（`proj`、`ff.net.3`）清零，使每个新增块恰为恒等；因此 depth-8 主干**从注册的 depth-4
函数出发**。逐 seed 在固定合成批（`(1,2,17,129,129)`、seed 1234）上断言编码器 FP32 输出**逐位
一致**，并把 `parent_state_sha256`/`expanded_state_sha256`/`encoder_output_sha256` 写进 receipt。
`estimated_peak_bytes` 按 K=8 实测 13.55 GiB 重新标定为 15 GiB（margin 4 GiB），**未沿用 2 GiB**。

## §0 Objective（单段，实测 869 字符）

> 在 docs/goals/s3-capacity-or-architecture.md 接续S3/index71/53.3064GPU-h、2023test未评分r0、S4未启动。训练配方支已按预声明关闭：输入范围本身无收益、监督域收回目标盒后收益只到 48h、剂量 800→2400 把 48h推过气候态（三 seed 全正 +0.06/+0.03/+0.11）而 72h 仍三 seed 全负、长 lead 权重 0.5→1.0 几乎无效、推理深度 K 4→8 在唯一完成的 seed 上 Δ≤0.009 K 且 72h 仍负（该臂被操作者中止、未完成）。本轮回主线换一个**与训练配方无关**的可证伪假设：**72h 的残余误差是否由主干容量不足主导**。单因素：在**已登记的最佳臂**（宽 129×129 输入 + `interior_32` 监督 + 2400 更新 + K=4 +注册物理权重 `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)` + 迁移 v3-BD 1600 父 + seed 41/42/43）上，**只把主干深度 `depth` 从 4 提到 8**，宽度/输入/监督/updates/K/权重/父/seed/评分域/气候态分母逐字不变；对照是**已登记的 2400 剂量臂**（index `record:s3-wide-interior-dose`，pinned 复用、不重训练）与已登记窄臂。读法：72h 三 seed 转正 → 提请独立验收；否则**关闭"靠加大容量解决 72h"这一支**，转**架构**（可推理信息组织/递归机制）或**数据**方向，不再加 seed、不重复同一 test、不无限调容量。本轮只报读数，**不宣布科学通过**、不称泛化/SOTA、不自行宣布最终goal完成。不paid/rental/exclusive/main/release/issueclose/force/mirror/破坏性/cron，不改科学合同/冻结证据/用户配置，不读 test。

## §1 起点、证据与实质差异

- 起点工作分支 `r7/weather-reasoning`；累计 **53.3064 GPU-h**、index **71**（含被中止的 K4→8 轮）。
- 起点证据：`docs/R7_S3_REASONING_DEPTH.md`（未完成）、`docs/R7_S3_WIDE_LONG_LEAD_WEIGHT.md`（§8 三轮合并）、
  `record:s3-wide-interior-dose`（48h 三 seed 全正 / 72h 三 seed 全负）。
- 与上一轮的实质差异：**只有 `depth`**（4 → 8）。其余全部逐字不变。
- 为何这是"回主线"：训练配方四个杠杆（输入范围/监督域/剂量/长 lead 权重）加推理深度 K 都已试过且
  72h 未过；容量是**与配方无关**的模型侧因子，直接对应"每步表征能力是否够表达长 lead 所需信息"。

## §2 交付物清单

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 本轮目标与冻结设计决定 | 本文件 §3 |
| D2 | 新协议/新输出/新 code.zip | `preparation_protocol.json`、`prepared_protocol.json`、`code.zip` + `archive_receipt.json` |
| D3 | 三 seed 真实训练与评分 | `seed{41,42,43}_receipt.json`、每 seed 五 lead 的 `boundary_rmse.csv` |
| D4 | 配对读数 | `readings.json`（本臂 vs 2400 剂量臂 vs 窄臂） |
| D5 | 登记与精确 CI | `docs/R7_S3_CAPACITY_DEPTH.md`、index record、brief、账本行、§8、campaign-state |

## §3 判据与设计决定（本轮冻结）

科学门与验收形式只来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` 与主计划 §8；本轮
**不触碰**任何阈值，也**不宣布**通过。

- **臂**：最佳已登记臂 + **`depth` 4→8**（`default_reasoning_steps` 仍 4、`heads` 仍 4、`dim` 仍 192）。
- **对照**：**已登记的 2400 剂量臂**（pinned 复用）为主对照；窄臂为次级参照。两者都不重训练。
- **实现约束**：不改动已登记的驱动与证据；本轮派生新驱动（或按已登记驱动的 `--depth` 变体），
  写进**新输出目录**，`code.zip` 与 `protocol_sha256` 都是新的。
- **预算**：深度翻倍使每步前向约 ×2，per-seed 期限相应放大（先按上一轮实测 2400 步 ≈ 11,210 s × 2
  ≈ 22,400 s 估，取 per-seed **28800 s**、planned **86400 s**、hard **115200 s**；先冻后跑，不在运行中改）。
- **预声明出口**：72h 三 seed 全正 → 提请独立验收；否则关闭"靠加大容量解决 72h"这一支，
  转架构或数据方向，不再加 seed、不重复同一 test、不无限调容量。
- **test 策略**：`test.jsonl` 从不读取；`test_read=false`、`scientific_claim=false`。

## §4 数字预算与停止

- planned 86400 s（软预算）、hard 115200 s（宽松硬上限）、每 seed 28800 s；软超继续并记 overrun，
  hard 截止即停 attempt 并全额留 `failure.json`，不运行中改硬限、不重试至偶然通过。
- 每阶段 spawn 前做**只读** `gpu_gate`（`GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b`），
  按实测标定 `estimated_peak_bytes`（K=8 实测 13.55 GiB，深度翻倍须重新标定，**不得**沿用 2 GiB 旧值）；
  **禁止**对任何非本实验进程发送信号或做冻结/终止自动化。
- 失败 attempt 原样保留、全额计费、不做 in-place resume。
- **不要在跑的时候停 workflow**（停止会 `SIGKILL` 在跑的 `world.run` 子进程，见计划 0021 §1）。

## §5 明确不做

不重训练任何已登记臂（窄/800 内部监督/2400 剂量/长 lead 权重/K4→8）、不重跑更早的已登记轮次、不读 test、
不改 `data/raw|interim|processed`、不改归档快照、不合成替代数据、不删部分输出复活、不改科学合同与
冻结证据与判据、不 main 合并/发版/关 issue、不 paid/rental/exclusive/force/mirror/破坏性/cron。

## §6 进度与交接

- 驱动与测试已实现（2026-10-12），CPU preflight 已过：`worker prepare` 写出 depth=8 candidate spec；
  `_parent_model(seed=41)` 对**真实**迁移父实测 `bit_identical=True`、`added=[4,5,6,7]`、
  `non_block_keys_unchanged=True`、严格 `load_state_dict` 通过（父 sha `fba27e26…`）。
- 确定性快门 `tools/r7_round.py round-guard` **通过**（28 例本轮测试 + 合并跑 148 例 + commit-guard 0）；
  `outputs/r7_s3_capacity_depth_code.zip` 已建（comment `710612d`、SHA256 `de9f6bc4…`、1380 文件）；
  commit `710612d` 已推送，CI run `38205959882` 绿。
- **进度（2026-10-12）**：本轮先被登记为**未运行**（`docs/R7_S3_CAPACITY_DEPTH.md`、0 GPU-h），
  随后操作者**决定改回、恢复运行**（依据：K 与 depth 是不同因子，跨轴证据不成立，见本文件开头）。
- **运行方式**：`code.zip` **不再预建**——`.zcode/workflow-drafts/r7-s3-capacity-depth-run.dwf.ts`
  会在 `freeze` 之前按当时的 HEAD 重建它（`git archive` 把 commit SHA 写进 ZIP comment，
  archive 阶段断言 `comment == freeze 时的 HEAD`），所以任何后续提交都不会让本轮失效。
- **第一条命令**（若手工跑）：`.venv/bin/python tools/r7_round.py round-guard` → 重建 zip →
  `.venv/bin/python tools/r7_round.py freeze` → `stage prepare/archive/seed{41,42,43}/reading`。

## §7 下一动作（S3）

本轮**已恢复运行**（depth 4→8 单因素；读数登记为新 record `s3-capacity-depth-run` 与新证据页
`docs/R7_S3_CAPACITY_DEPTH_RUN.md`，原「未运行」登记逐字节保留）。**下一动作（S3）**：按 §3 的
预声明出口读——72 h 三 seed 全正 → 提请独立验收；否则**关闭"靠加大容量解决 72 h"这一支**，
转架构方向（数据扩充按 §5 暂缓、放到最最后），并在 §8 写清"为何停 + 恢复第一条命令"。
宽输入支与训练配方支已关闭不重开；test 未读、r=0、S4 未启动不变。
