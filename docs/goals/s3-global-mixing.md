# S3：H-C 长 lead 的全局混合路径（全局 vs 同参数量额外局部层）——H-B 关闭后的第三根架构轴

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-global-mixing", "driver": "scripts/study_r7_s3_global_mixing.py", "output_stem": "r7_s3_global_mixing_20261015", "control_record": "s3-wide-interior-dose", "seeds": [41, 42, 43], "device_mode": "dual", "test_path": ["tests/test_r7_s3_global_mixing.py"], "commit_paths": ["model/process_forecast_r7.py", "model/global_mixing_r7.py", "scripts/r7_s3_global_mixing_parent.py", "scripts/study_r7_s3_global_mixing.py", "tests/test_r7_s3_global_mixing.py", "docs/goals/s3-global-mixing.md"], "commit_message": "feat(r7): implement the H-C global-mixing arm with its matched extra-local-layer control [model-digest-change]", "register_paths": ["docs/R7_S3_GLOBAL_MIXING_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md"], "register_message": "feat(r7): register the H-C global-mixing run as a new evidence record", "code_archive": "outputs/r7_s3_global_mixing_code.zip"} -->

状态 **设计已定、待实现模型改动与 CPU 前置**（2026-10-15）。唯一主计划
`docs/goals/main-model-climatology-campaign.md`；承接 **H-A 关闭**（`record:s3-process-init-result`，negative）、
**H-B 关闭**（`record:s3-typed-spatial-result`，B→C null，且已记录为**弱操纵下的 null**）与决策 **0048**
（容量轴永久关闭）。本轮取 `docs/plans/0022` §3 的 **H-C**（§4 排序 H-A > H-B > **H-C** 中的最后一位；
排最后是**成本**理由，不是证据理由——它是唯一直接针对"感受野/大尺度混合"这一结构缺口的候选）。

## §0 Objective（单段）

> 在 docs/goals/s3-global-mixing.md 接续 S3/index78/93.331638 GPU-h、2023 test 未评分 r0、S4 未启动；H-A 与 H-B 两支已按各自预声明出口关闭、容量轴已按决策 0048 永久关闭，本轮回主线的**第三根架构轴**——**只给解码器（或 solver）加一条全局混合路径**（低秩全局算子或单层全局注意力，补上"局部窗口 + 3×3 conv 感受野受限"这一结构缺口；局部窗口与 3×3 conv 本身不变），并**在同一协议内同跑一个参数量/FLOPs 对齐的"额外局部层"对照臂**，以区分「全局性」与「单纯加容量」；其余（宽 129×129 输入、`interior_32` 监督、2400 更新、K=4、depth=4、`dim`=192、注册物理权重、seed 41/42/43、评分域、气候态分母）逐字不变；新路径**最后、在 `isolated_stream()` 下构造**使"off"路径逐位等于现实现，并有"全局路径输出投影清零 ⇒ 逐位退回注册臂"的正控制；预声明出口：**72h 三 seed 全正且全局臂相对局部对照臂 72h 三 seed 同号为优 → 提请独立验收，否则关闭 H-C 这一支**、转训练信号/目标组织或重估主线；本轮只报读数、**不宣布科学通过**、不称泛化/SOTA、不自行宣布最终 goal 完成；不 paid/rental/exclusive/main/release/issueclose/force/mirror/破坏性/cron，不改科学合同/冻结证据/用户配置，**不读 test**。

## §1 起点与实质差异

- 起点工作分支 `r7/weather-reasoning`；累计 **93.331638 GPU-h**、index **78**。
- 起点证据：`docs/R7_S3_TYPED_SPATIAL_RESULT.md`（H-B，B→C null）、`docs/R7_S3_PROCESS_INIT_RESULT.md`（H-A，negative）、`record:s3-wide-interior-dose`（外部对照，pinned）。
- **只读结构事实（`docs/plans/0022` §2，待本轮 CPU 阶段逐条复核 file:line）**：主干用
  `window_size` 局部窗口注意力 + 3×3 卷积，感受野有限；而 48h 已能过气候态门、**72h 从未过**，
  多天误差增长以大尺度为主 —— 这是 H-C 的动机。
- 与上一轮的**实质差异**：H-B 改的是**证据 token 的聚合方式**；本轮改的是**主干/解码器的空间混合范围**。
  两者都改 `model/*.py`（故都需父重迁移），因子互不重叠。

## §2 交付物清单

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 本轮目标与冻结设计决定 | 本文件 §3 |
| D2 | 新协议/新输出/新 code.zip | `preparation_protocol.json`、`prepared_protocol.json`、`code.zip` + `archive_receipt.json` |
| D3 | **两臂**三 seed 真实训练与评分 | `seed{41,42,43}_{global,local}_receipt.json`、每 seed 每臂五 lead 的 `boundary_rmse.csv` |
| D4 | 配对读数 | `readings.json`（全局 vs 局部同协议；两臂 vs 已登记 2400 剂量臂 pinned） |
| D5 | 登记与精确 CI | `docs/R7_S3_GLOBAL_MIXING_RESULT.md`、index record、brief、账本行、§8、campaign-state |

## §3 判据与设计决定（本轮冻结；科学门只引用冻结文档）

科学门与验收形式只来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` 与
`docs/plans/0022-s3-architecture-next-round.md`（H-C 的六项口径）；本轮**不触碰**任何阈值，也**不宣布**通过。

- **候选臂 G（global）**：已登记 2400 剂量臂 + **一条全局混合路径**（唯一因子）。`depth`=4、`dim`=192、
  K=4、宽 129×129 输入、`interior_32` 监督、2400 更新、注册物理权重、seed 41/42/43 逐字不变。
- **协议内对照臂 L（local）**：**同参数量、同 FLOPs** 的**额外局部层**（同图、同位置，只把混合范围限制回
  局部窗口/3×3）。G 与 L 必须在**同一协议、同一 seed、同一预算、同一通路初始化**下各跑一遍 —— 只此才可把
  **L→G** 的差异归因到**混合范围**，而不是"单纯加容量"。**这条正是容量轴已关闭（0048）之后必须证明的事**。
- **参考臂 A**：`record:s3-wide-interior-dose`（pinned 复用、不重训练）为外部参考。
- **参数量/FLOPs 对齐（关键）**：G 与 L 必须逐位对齐（须实测并分报，**不**声称近似相等）；不等则本轮的归因
  不成立，须如实记为混杂。
- **共享初始化（H-B 轮复核抓到的 medium finding，必须照搬其修复）**：两臂的新增参数必须在**同一 round seed**
  下逐位相同（`torch.manual_seed(seed)` 构造），并把两臂共有键的 `pathway_init_sha256` 写进各自 receipt
  （两者必须**相等**），`_seed` 在不等时 `raise`。**本条为硬约束，不满足即不得开跑。**
- **实现约束（模型改动，须 `[model-digest-change]`）**：新路径**最后、在 `isolated_stream()` 下构造**，
  使"off"路径逐位等于现实现；由一个显式 spec 开关选择；插入点（解码器 vs solver、全局注意力 vs 低秩算子）
  在 CPU 阶段以**只读测绘**固定并写进协议，**不在运行中改**。
- **正/负控制**：正控制＝把全局路径的输出投影**清零** ⇒ 逐位退回注册臂（恒等反证，须逐 seed 断言且
  **失败即 `raise`**，不是只写进 receipt）；负控制＝L 臂本身（同参数量、只改混合范围）。
- **预算**：两臂 × 三 seed = 六次训练；per-seed/planned/hard 按 H-B 轮实测（15.443038 GPU-h 六阶段、
  per-seed 4.33–5.77 GPU-h）**重新标定**（先冻后跑，不在运行中改）。
- **预声明出口**：**72h 三 seed 全正 且 G 相对 L 在 72h 三 seed 同号为优 → 提请独立验收**；
  否则**关闭 H-C 这一支**，不调宽度/秩/层数、不加 seed。**若 L→G 无差异 ⇒ 结论是"瓶颈不是感受野"**
  （与"额外容量"不可分），如实写清；**若 G 与 L 都无改善**，则长 lead 瓶颈既非感受野也非容量。
- **test 策略**：`test.jsonl` 从不读取；`test_read=false`、`scientific_claim=false`。

## §4 数字预算与停止

- 每 seed 上限、planned（软预算）、hard（宽松硬上限）在实现时按 H-B 轮实测重新标定后写入驱动常量；
  软超继续并记 overrun，hard 截止即停 attempt 并全额留 `failure.json`。
- 每阶段 spawn 前做**只读** `gpu_gate`；`estimated_peak_bytes` 按 H-B 轮实测峰值（spatial 9.276 GiB /
  pooled 9.041 GiB）标定。**禁止**对任何非本实验进程发送信号（R-054 / 决策 0026）。
- **双卡分配（决策 0047）**：`device_mode: "dual"`，冻结块记 `gpu_uuids` 与 `neighbour_reserve_bytes = 6 GiB`。
- 失败 attempt 原样保留、全额计费、不做 in-place resume。

## §5 明确不做

不重训练任何已登记臂（A 只 pinned 复用）、不重跑更早的已登记轮次、不读 test、不改 `data/raw|interim|processed`、
不改归档快照、不合成替代数据、不删部分输出复活、不改科学合同与冻结证据与判据、不 main 合并/发版/关 issue、
不 paid/rental/exclusive/force/mirror/破坏性/cron。**不重开容量轴**（0048）、**不重开 H-A 支**、**不重开 H-B 支**
（两者均已按预声明出口关闭；H-B 的"弱操纵"限定已写进其证据页 §7b）。

## §6 进度与交接

- **已定**：设计（§3）、登记名（成功版 `docs/R7_S3_GLOBAL_MIXING_RESULT.md` / record
  `s3-global-mixing-result`；本因子此前**无**任何已登记 record，故无 correction 指针）。
- **父迁移前提（必须照搬，否则重蹈 H-A attempt01）**：本轮同样改 `model/*.py` ⇒ `model_code_digest()` 会变 ⇒
  注册时的迁移父会被 `training/r7_experiment.py:129` 的身份门**按设计**拒绝。**必须**先用**已提交的**
  `scripts/migrate_r7_parent_state.py` 在**当前**代码下重跑迁移，再按 `scripts/r7_s3_typed_spatial_parent.py`
  的**同一模式**建 `scripts/r7_s3_global_mixing_parent.py`（钉新路径＋新 SHA、复核 receipt 的
  `current_model_code_sha256`、`install_parent_pins(body)`、`expand_parent_state(..., seed=...)` 共享初始化）。
  **不得**修改比对逻辑或放宽任何门。
- **待做（开工前 CPU 工作）**：只读测绘并固定插入点；实现 `model/global_mixing_r7.py` 与 spec 开关；
  派生 `scripts/study_r7_s3_global_mixing.py`（**R-051 600 行硬限**）；写 `tests/test_r7_s3_global_mixing.py`
  （含"清零输出投影 ⇒ 逐位退回"反证、"off 逐位不变"断言、G/L 参数量与 FLOPs 对齐断言、两臂共享初始化断言）；
  逐行 diff 单因素保真、独立复核（≤3）、`round-guard` 与 `commit-guard`、提交推工作分支并等 CI 绿。
- **CI 安全**：测试**不得**依赖 `outputs/`（H-A 与 H-B 都曾在此处让 CI 红）——收窄 `execution_files()`
  到实际存在的文件并 stub `sha256_file`、`evidence_payload`、`recipe._shared`；在干净 worktree（无 `outputs/`）
  里实跑确认。
- **第一条命令**：`.venv/bin/python tools/r7_round.py round-guard` → 重建 zip →
  `.venv/bin/python tools/r7_round.py freeze` → `stage prepare/archive/stage-seeds/reading`。
- **本轮尚未开跑**：0 GPU-h；无读数、无结论。

## §7 下一动作（S3）

- 在 **S3** 节点：先只读测绘固定插入点，再实现模型改动与 `scripts/r7_s3_global_mixing_parent.py`，
  然后派生驱动与 CI-safe 测试，最后 `round-guard` → `freeze` → 逐 stage GPU → 四件套登记。
- 第一条命令：`.venv/bin/python tools/r7_round.py round-guard`。
- 预声明出口见 §3：**72h 三 seed 全正且 G 相对 L 在 72h 三 seed 同号为优 → 提请独立验收；否则关闭 H-C 这一支**。
- 本轮**未产生任何读数**；不宣布通过、不关闭任何支。
