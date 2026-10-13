# S3：主干容量（深度）单因素——训练配方支关闭后的第一个容量臂

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-capacity-depth", "driver": "scripts/study_r7_s3_capacity_depth.py", "output_stem": "r7_s3_capacity_depth_20261012", "control_record": "s3-wide-interior-dose", "seeds": [41, 42, 43], "device_mode": "dual", "test_path": ["tests/test_r7_s3_capacity_depth.py"], "commit_paths": ["scripts/study_r7_s3_capacity_depth.py", "tests/test_r7_s3_capacity_depth.py", "training/r7_device_selection.py", "tests/test_r7_device_selection.py", "docs/goals/s3-capacity-or-architecture.md"], "commit_message": "feat(r7): implement the S3 capacity-depth (trunk depth 4->8) single-factor arm", "register_paths": ["docs/R7_S3_CAPACITY_DEPTH_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md"], "register_message": "feat(r7): register the S3 capacity-depth (depth 4->8) run as a new evidence record", "code_archive": "outputs/r7_s3_capacity_depth_code.zip"} -->

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

**（2026-10-13 追加，不改上文一字）出口的适用范围**：上一条出口所指的"靠加大容量解决 72h"读作
**深度轴**——即本轮作为唯一单因子所测的 `depth` 4→8，而不是整条容量轴。宽度是**独立轴**，在当前
主线上尚未单因素检验；但它被**封顶为最后一个容量臂**：若宽度臂也未能让 72h 三 seed 全正，则整条
容量轴（深度＋宽度）**永久关闭**，此后不得再开任何容量臂（不重试 depth、不再扫 `dim`、不做宽＋深
组合）。理由、**时序披露**（本追加作出时 seed 41 的 72h 读数已可见）与代价见决策 **0048**
（`docs/decisions/0048-capacity-axis-scope-and-cap.md`）。**本追加不改任何阈值、门或判据**。

**（2026-10-13 追加）本轮登记名**：成功版的证据页与 record 用 `docs/R7_S3_CAPACITY_DEPTH_RESULT.md`
与 `s3-capacity-depth-result`；上面 §2/§7 里写的 `docs/R7_S3_CAPACITY_DEPTH_RUN.md` 与
`s3-capacity-depth-run` **已被那次 seed 41 CUDA OOM 的失败登记占用**（索引 index73、`blocked`），
两条既有登记（`s3-capacity-depth`、`s3-capacity-depth-run`）逐字节保留，更正指针追加在 blocked 那条上。

## §4 数字预算与停止

- planned 86400 s（软预算）、hard 115200 s（宽松硬上限）、每 seed 28800 s；软超继续并记 overrun，
  hard 截止即停 attempt 并全额留 `failure.json`，不运行中改硬限、不重试至偶然通过。
- 每阶段 spawn 前做**只读** `gpu_gate`（`GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b`），
  按实测标定 `estimated_peak_bytes`（K=8 实测 13.55 GiB，深度翻倍须重新标定，**不得**沿用 2 GiB 旧值）；
  **禁止**对任何非本实验进程发送信号或做冻结/终止自动化。
  - **2026-10-12 用户授权放宽 + 代理臂纠正**：本机默认共驻，邻居作业在声明 GPU 上占用 ~14 GiB，原
    `15 GiB + 4 GiB = 19 GiB` 门槛使本轮无限等待。两处纠正：① **原估计用错了代理臂**——K 主导
    `reserved` peak（同 K=4 的剂量臂实测 **8562 MiB**，K=8 臂实测 13878 MiB）；本臂是 depth-8/**K=4**，
    depth 翻倍只加深编码器（一次前向 ~0.1 GiB），故足迹 ~8.5 GiB，而非按 K=8 臂估的 14 GiB。
    ② 用户明确「不用算那么准、GPU 不可能全空闲、差不多能跑就行」，故门槛定为 `8.5 GiB 估计 + 0.5 GiB margin
    = 9 GiB`（floor 8 GiB），共驻卡可通过。这**不是**改科学判据，也不放宽「禁止对邻居发信号」；仅 resource
    gate 的精度。据此**新开 attempt**（此前 attempt 的 prepare/archive 保留、不删、不续跑；zip/协议随新 HEAD 重建）。
- **2026-10-12 双卡分配（决策 0047）**：本轮 `device_mode: "dual"`，冻结块另记 `gpu_uuids`
  （声明卡 `GPU-408ad137…` 排第一）与 `neighbour_reserve_bytes = 6 GiB`，门槛变为
  `required = max(估计, 实测峰值) + margin + reserve`。由 `tools/r7_round.py stage-seeds` 驱动：
  **每张卡同一时刻至多一个本轮 worker**，一卡一 seed 并发，seed 的偏好卡是 `gpu_uuids` 上的
  确定性轮转（三个 seed 分两张卡不会全挤在声明卡）。**不改训练配置、不引入 DDP**
  （先例 `scripts/study_r7_b3_scale.py`）；`single` 模式保留旧行为（`gpu_uuids = [声明卡]`，
  逐位不变）。reserve 是**判断不是保证**——6 GiB 取自 seed 41 那次的邻居增长实测（+6.2 GiB），
  邻居若涨得更多 seed 仍会 OOM，如实登记为资源失败，不写成「已解决」。逐张卡仍只读探测、
  全不过即 `headroom-blocked` 有界等待，**仍不对邻居发任何信号**。
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
- **本轮准备提交与 CI（2026-10-12）**：`5ca88d9`（拆 stage/register 提交口径 + 恢复本轮可运行）、
  `d5859a0`（同步 R-021 规模 marker 91→93）已推工作分支；`R7 CPU CI` run `38233411450` 对
  `d5859a0…` **九主步骤全 success**。首次 run `38228709491` 因 R-021 anti-drift marker 漂移而红
  （本轮把 `tools/r7_round_stages.py` 362→416、`tests/test_r7_round_stages.py` 347→428 推过 400 行目标），
  已按本页先例同步 marker，**未改阈值 / 规则级别 / 冻结例外**。
- **gate 校准与登记口径修复（2026-10-12 深夜，另一窗口提交 `9e8124f`/`8ff541b`）**：19 GiB → 16 GiB →
  **9 GiB**（显存主因是 K 不是 depth；本臂 depth-8/**K=4** 的真实足迹约 8.5 GiB，原按 K=8 臂
  13.55 GiB 高估约 2×）；并把 workflow 结尾的登记提交由 `commit-round` 改为 `commit-register`。
  **有效驱动身份由此前进到 `8ff541b`**（index72 的 record 与 `docs/R7_S3_CAPACITY_DEPTH.md`
  按"旧 SHA 不改、修正以追加表达"的惯例**不动字节**）。
- **三次 attempt，全部 0 GPU-h、无 seed、无读数**（原样保留、不删不续）：
  `attempt01` prepare 619.8 s + archive 614.6 s（`code_commit 5386f7c`、zip `615bb2d0…`）；
  `attempt02` 仅 prepare 593.3 s；`attempt03` prepare 630.2 s + archive 490.1 s（`code_commit 8ff541b`、
  zip `a4bf0746…`）。`attempt03` 的 `stage seed 41` 曾启动但在 preflight 被停，
  **无 seed41 目录、无 readings.json**。
- **中断与根因**：run `dwfrun-6c3847ad` 的 `.mjs` 进程从未死（一直在跑），但运行记录被判
  `stopped/interrupted`、进度事件被丢 29 次，日志原文
  `"Dynamic workflow run progress dropped: parent session is not this app"`——**会话跨了 ZCode app 实例**，
  完成通知无法投递，故 goal 永远等不到结果。该 workflow 已被本窗口 SIGTERM 停掉。
- **操作教训**：**goal 与 workflow 必须在同一个窗口/同一个 app 实例内跑**；跨实例会丢完成通知。

## §7 下一动作（S3）

本轮仍**未跑 GPU**（三次 attempt 全部止步于 prepare/archive，0 GPU-h；详见 §6）。**下一动作（S3）**：
在**单一窗口**重开干净一轮——`round-guard` → 重建 `code.zip`（comment 必须 == HEAD）→ `freeze`
（自动创建 `attempt04`）→ `stage prepare/archive/seed{41,42,43}/reading` → 四件套登记
（新 record `s3-capacity-depth-run` + 新证据页 `docs/R7_S3_CAPACITY_DEPTH_RUN.md`，
原「未运行」登记逐字节保留）。随后按 §3 预声明出口读——72 h 三 seed 全正 → 提请独立验收；
否则**关闭"靠加大容量解决 72 h"这一支**，转架构方向（数据扩充按 §5 暂缓、放到最最后）。
宽输入支与训练配方支已关闭不重开；test 未读、r=0、S4 未启动不变。
