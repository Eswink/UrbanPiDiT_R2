# S3：H-B 空间分辨的类型化证据（pooled → 保留 patch 位置）——H-A 关闭后的第二根架构轴

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-typed-spatial", "driver": "scripts/study_r7_s3_typed_spatial.py", "output_stem": "r7_s3_typed_spatial_20261014", "control_record": "s3-wide-interior-dose", "seeds": [41, 42, 43], "device_mode": "dual", "test_path": ["tests/test_r7_s3_typed_spatial.py"], "commit_paths": ["model/typed_evidence_r7.py", "scripts/r7_s3_typed_spatial_parent.py", "scripts/study_r7_s3_typed_spatial.py", "tests/test_r7_s3_typed_spatial.py", "docs/goals/s3-typed-spatial.md"], "commit_message": "feat(r7): implement the H-B spatially-resolved typed-evidence arm with its in-protocol pooled control [model-digest-change]", "register_paths": ["docs/R7_S3_TYPED_SPATIAL_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md"], "register_message": "feat(r7): register the H-B spatially-resolved typed-evidence run as a new evidence record", "code_archive": "outputs/r7_s3_typed_spatial_code.zip"} -->

状态 **设计已定、待实现模型改动与 CPU 前置**（2026-10-14）。唯一主计划
`docs/goals/main-model-climatology-campaign.md`；承接 **H-A 已按预声明出口关闭**
（`record:s3-process-init-result`，72h 三 seed 全负、相对剂量臂长 lead 反而更差）与决策 **0048**
（容量轴永久关闭）。本轮取 `docs/plans/0022` §3 的 **H-B**（§4 排序 H-A > **H-B** > H-C 中的第二位）。

## §0 Objective（单段）

> 在 docs/goals/s3-typed-spatial.md 接续 S3/index77/77.888623 GPU-h、2023 test 未评分 r0、S4 未启动；H-A 支已按预声明出口关闭、容量轴已按决策 0048 永久关闭，本轮回主线的**第二根架构轴**——**只把类型化证据 token 的空间路由从"按位置求均值压成 1 个向量"改为"保留 patch 位置 `[B,P,dim]` 并做位置对齐注入"**（`model/typed_evidence_r7.py` 的 `TYPED_ROUTING` 分支；同信息、不同路由），其余（宽 129×129 输入、`interior_32` 监督、2400 更新、K=4、depth=4、`dim`=192、注册物理权重、seed 41/42/43、评分域、气候态分母）逐字不变；**B/C 两臂在同一协议内同跑**（B = 同信息 pooled、即 #79 现状；C = 同信息空间保留 = 本轮候选），参数量/FLOPs 对齐，只检验 **B→C** 的归因，A（无类型）用已登记 2400 剂量臂 pinned 复用；预声明出口：**72h 三 seed 全正 → 提请独立验收，否则关闭 H-B 这一支**、转 H-C 或训练信号/目标组织；本轮只报读数、**不宣布科学通过**、不称泛化/SOTA、不自行宣布最终 goal 完成；不 paid/rental/exclusive/main/release/issueclose/force/mirror/破坏性/cron，不改科学合同/冻结证据/用户配置，**不读 test**。

## §1 起点与实质差异

- 起点工作分支 `r7/weather-reasoning`；累计 **77.888623 GPU-h**、index **77**。
- 起点证据：`docs/R7_S3_PROCESS_INIT_RESULT.md`（H-A，negative）、`record:s3-wide-interior-dose`（A 对照，pinned）。
- **只读结构事实**（已复核 file:line）：`model/typed_evidence_r7.py:199`
  `return torch.stack([value.mean(dim=1) for value in projected], dim=1)` —— `TYPED_ROUTING` 分支在
  位置维 `P` 上取 `mean(dim=1)`，把每类型证据压成**一个全局向量**再广播注入 anchored slot。
  H-B 的假设：这个均值是 #79 负结果的原因（空间结构被抹掉），而不是"类型信息无用"。
- 与上一轮的**实质差异**：H-A 改的是**过程状态初值 `P_0` 的构造**；本轮改的是**证据 token 的空间路由**。
  两轮都改 `model/*.py`（故都需父重迁移），但因子互不重叠。

## §2 交付物清单

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 本轮目标与冻结设计决定 | 本文件 §3 |
| D2 | 新协议/新输出/新 code.zip | `preparation_protocol.json`、`prepared_protocol.json`、`code.zip` + `archive_receipt.json` |
| D3 | **两臂**三 seed 真实训练与评分 | `seed{41,42,43}_{b,c}_receipt.json`、每 seed 每臂五 lead 的 `boundary_rmse.csv` |
| D4 | 配对读数 | `readings.json`（C vs B 同协议；C/B vs A pinned） |
| D5 | 登记与精确 CI | `docs/R7_S3_TYPED_SPATIAL_RESULT.md`、index record、brief、账本行、§8、campaign-state |

## §3 判据与设计决定（本轮冻结；科学门只引用冻结文档）

科学门与验收形式只来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` 与
`docs/plans/0022-s3-architecture-next-round.md`（H-B 的六项口径）；本轮**不触碰**任何阈值，也**不宣布**通过。

- **候选臂 C**：已登记 2400 剂量臂 + **类型化证据开启且空间保留**（唯一因子）。`depth`=4、`dim`=192、
  K=4、宽 129×129 输入、`interior_32` 监督、2400 更新、注册物理权重、seed 41/42/43 逐字不变。
- **协议内对照臂 B**：同信息、同初始化，**只把位置聚合换回 pooled 均值**（即 #79 现状）。B 与 C 必须在
  同一协议、同一 seed、同一预算、**同一通路初始化**下各跑一遍 —— 只此才可把 **B→C** 的差异归因到
  "位置聚合"，而**不是**归因到协议差异、容量差异或随机初始化差异。
- **参考臂 A**：`record:s3-wide-interior-dose`（pinned 复用、不重训练）为外部参考；`record:s3-rollout-dose` 为次级。
- **共享初始化（关键；#79 先例的阻断式防护）**：两臂的 typed-evidence 通路张量（`conv`/`norm`/四个
  `projections`）必须在**同一 round seed 下逐位相同** —— `expand_parent_state` 在 `torch.manual_seed(seed)`
  下构造目标模型，并把两臂共有键的 `pathway_init_sha256` 写进各自 receipt（两者必须**相等**）。
  #79 轮用 `verify_arm_pairing` + `shared_initial_state` 防的正是这个混杂；若两臂各自随机初始化，
  则 B→C 的差值不可归因到路由。**本条为硬约束，不满足即不得开跑。**
- **参数量（已声明的差异，不是混杂）**：C 的聚合用**一个共享的** `[dim]` query，故两臂除
  `typed_evidence.position_query` 外**逐位相同**；C 恰多 `dim`=192 个参数（+0.13% of 148800）。
  此差异**必须**分报（`declared_difference`）并写进 limitations，**不**声称参数量逐位相等。
- **实现约束（模型改动，须 `[model-digest-change]`）**：在 `model/typed_evidence_r7.py` 新增
  `typed_spatial` 模式：保留 `P` 维（不再 `mean(dim=1)`），用 softmax 注意力（一个共享可学习 query）聚合。
  **pooled 模式必须逐位等于 #79 现实现**（既有 #79 测试即为此反证）；query 为 0 时 spatial 的聚合等于
  pooled 的均值（须实测并写进 receipt）。**注意**：`pooled` 臂**不是**"注册 2400 剂量臂"——后者根本没有
  typed evidence；A 只作外部 pinned 参考。
- **正/负控制**：正控制＝（i）query 为 0 时 spatial 聚合 == pooled 均值；（ii）清零四个 per-type 投影后
  两模式证据**恒为 0**；（iii）类型标签置零后该类型证据恒为 0。**任一不满足即 `raise` 中止本轮**
  （不是只写进 receipt）。负控制＝B 臂本身（同信息、pooled）。
- **预算**：两臂 × 三 seed = 六次训练；per-seed/planned/hard 按 H-A 轮实测（6.8590 GPU-h 六阶段）
  **重新标定**（先冻后跑，不在运行中改）。相对 H-A 驱动的 per-seed 28800→43200 s 与峰值估计
  12198→9728 MiB 是**两臂的后果**，两臂共享同一 per_seed_seconds/peak 估计，不破坏 B→C 可比性。
- **预声明出口**：**B→C 在 72h 三 seed 同号且 C 的 72h 三 seed 全正 → 提请独立验收**；
  否则**关闭 H-B 这一支**，不调注入宽度/不加 seed。若 B→C 无差异 ⇒ 结论是**类型路由本身无效**
  （而非"空间保留无用"），如实写清。
- **test 策略**：`test.jsonl` 从不读取；`test_read=false`、`scientific_claim=false`。

## §4 数字预算与停止

- 每 seed 上限、planned（软预算）、hard（宽松硬上限）在实现时按 H-A 轮实测重新标定后写入驱动常量；
  软超继续并记 overrun，hard 截止即停 attempt 并全额留 `failure.json`。
- 每阶段 spawn 前做**只读** `gpu_gate`；`estimated_peak_bytes` 按 H-A 轮实测峰值 **8.361 GiB** 标定
  （H-A 协议写 11.91 GiB 属高估 3.55 GiB，故本轮用实测值加 margin）。**禁止**对任何非本实验进程发送信号（R-054 / 决策 0026）。
- **双卡分配（决策 0047）**：`device_mode: "dual"`，冻结块记 `gpu_uuids` 与 `neighbour_reserve_bytes = 6 GiB`。
- 失败 attempt 原样保留、全额计费、不做 in-place resume。

## §5 明确不做

不重训练任何已登记臂（A 只 pinned 复用）、不重跑更早的已登记轮次、不读 test、不改 `data/raw|interim|processed`、
不改归档快照、不合成替代数据、不删部分输出复活、不改科学合同与冻结证据与判据、不 main 合并/发版/关 issue、
不 paid/rental/exclusive/force/mirror/破坏性/cron。**不重开容量轴**（决策 0048）、**不重开 H-A 支**（已按预声明出口关闭）。

## §6 进度与交接

- **已定**：设计（§3）、登记名（成功版 `docs/R7_S3_TYPED_SPATIAL_RESULT.md` / record
  `s3-typed-spatial-result`；本因子此前只有 #79 的**不同实例**登记 `s2-79-typed-evidence-screening`，
  不是同协议对照，故无 correction 指针）。
- **父迁移前提（必须照搬，否则重蹈 H-A attempt01）**：本轮同样改 `model/*.py` ⇒ `model_code_digest()` 会变 ⇒
  注册时的迁移父会被 `training/r7_experiment.py:129` 的身份门**按设计**拒绝。**必须**先用**已提交的**
  `scripts/migrate_r7_parent_state.py` 在**当前**代码下重跑迁移，再按 `scripts/r7_s3_process_init_parent.py`
  的**同一模式**建 `scripts/r7_s3_typed_spatial_parent.py`（钉新路径＋新 SHA，并在 `pinned_parent` 内复核
  receipt 的 `current_model_code_sha256`），由驱动用 `install_parent_pins(body)` 覆写冻结协议的 checkpoint pins。
  **不得**修改比对逻辑或放宽任何门。
- **待做（开工前 CPU 工作）**：改 `model/typed_evidence_r7.py`（保留 `P` 维 + 路由开关 + 位置对齐注入）、
  派生 `scripts/study_r7_s3_typed_spatial.py`（**R-051 600 行硬限**：派生自 H-A 驱动的 600 行，
  新增行须等量削减或外移到新模块）、写 `tests/test_r7_s3_typed_spatial.py`（含"类型置零 ⇒ 恒零"反证、
  "off/pooled 路径逐位不变"断言、B/C 参数量与 FLOPs 对齐断言）、逐行 diff 单因素保真、
  独立复核（≤3）、`round-guard` 与 `commit-guard`、提交推工作分支并等 CI 绿（提交信息须带 `[model-digest-change]`）。
- **CI 安全**：测试**不得**依赖 `outputs/`（H-A 曾两次在此处让 CI 红）——收窄 `execution_files()` 到实际存在的文件
  并 stub `sha256_file`；在干净 worktree（无 `outputs/`）里实跑确认。
- **文献旁证**：`.zcode/plans/s3-hb-lit-2026-10-13.md`（**工作态草稿，不进 index**）——结论＝只有间接支持、
  没有直接实证；主流做法保留空间结构，但**没人做过"同信息只改路由"的定量对照**，故方向不可由文献决定。
- **第一条命令**：`.venv/bin/python tools/r7_round.py round-guard` → 重建 zip →
  `.venv/bin/python tools/r7_round.py freeze` → `stage prepare/archive/stage-seeds/reading`。
- **本轮尚未开跑**：0 GPU-h；无读数、无结论。

## §7 下一动作（S3）

- 在 **S3** 节点：先实现模型改动与 `scripts/r7_s3_typed_spatial_parent.py`，再派生驱动与 CI-safe 测试，
  然后 `round-guard` → `freeze` → 逐 stage GPU → 四件套登记。
- 第一条命令：`.venv/bin/python tools/r7_round.py round-guard`。
- 预声明出口见 §3：**B→C 三 seed 同号且 C 的 72h 三 seed 全正 → 提请独立验收；否则关闭 H-B 这一支**。
- 本轮**未产生任何读数**；不宣布通过、不关闭任何支。
