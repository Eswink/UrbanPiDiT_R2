# S3：H-A 过程状态输入条件化（P_0 逐样本）——容量轴永久关闭后的第一根架构轴

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-process-init-conditioning", "driver": "scripts/study_r7_s3_process_init.py", "output_stem": "r7_s3_process_init_20261014", "control_record": "s3-wide-interior-dose", "seeds": [41, 42, 43], "device_mode": "dual", "test_path": ["tests/test_r7_s3_process_init.py"], "commit_paths": ["model/process_forecast_r7.py", "scripts/study_r7_s3_process_init.py", "tests/test_r7_s3_process_init.py", "docs/goals/s3-process-init-conditioning.md"], "commit_message": "feat(r7): implement the H-A process-state input-conditioning single-factor arm [model-digest-change]", "register_paths": ["docs/R7_S3_PROCESS_INIT_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md"], "register_message": "feat(r7): register the H-A process-state input-conditioning run as a new evidence record", "code_archive": "outputs/r7_s3_process_init_code.zip"} -->

状态 **设计已定、待实现模型改动与 CPU 前置**（2026-10-14）。唯一主计划
`docs/goals/main-model-climatology-campaign.md`；承接决策 **0048**：容量轴（深度 depth 4→8 与宽度
`dim` 192→256）三 seed 72h 全负 ⇒ **永久关闭**；此后只剩**训练信号 / 目标组织 / 数据**（数据按 D3 暂缓）。
本轮取 `docs/plans/0022` §3 的 **H-A**（零规则风险候选）：**输入条件化的过程状态初始化 `P_0`**。

## §0 Objective（单段）

> 在 docs/goals/s3-process-init-conditioning.md 接续 S3/index75/71.0296 GPU-h、2023 test 未评分 r0、S4 未启动；容量轴已按决策 0048 永久关闭，本轮回主线的**架构轴**——**只把过程状态初值 `P_0` 从"全体样本共享的 `process_queries`"改为逐样本条件化**（`P_0 = process_queries.expand(B,·,·) + process_init_proj(pool(context))`），其余（宽 129×129 输入、`interior_32` 监督、2400 更新、K=4、depth=4、`dim`=192、注册物理权重、seed 41/42/43、评分域、气候态分母）逐字不变；新增 `process_init_proj`（`Sequential(LayerNorm, Linear(dim,dim))`）**最后、在 `isolated_stream()` 下构造**，使"off"路径**逐位等于**现实现，并有"`Linear` 权重清零 ⇒ 逐位退回注册臂"的**正控制**与 matched_generic **负控制**；预声明出口：**72h 三 seed 全正 → 提请独立验收，否则关闭 H-A 这一支**、转 H-B/H-C 或训练信号/目标组织；本轮只报读数、**不宣布科学通过**、不称泛化/SOTA、不自行宣布最终 goal 完成；不 paid/rental/exclusive/main/release/issueclose/force/mirror/破坏性/cron，不改科学合同/冻结证据/用户配置，**不读 test**。

## §1 起点与实质差异

- 起点工作分支 `r7/weather-reasoning`；累计 **71.0296 GPU-h**、index **75**。
- 起点证据：`docs/R7_S3_CAPACITY_WIDTH_RESULT.md`（宽度，negative）、`record:s3-wide-interior-dose`（对照臂，pinned）。
- **只读结构事实**（`docs/plans/0022` §2，已复核 file:line）：`model/process_forecast_r7.py:434`
  `process=self.process_queries.expand(B,-1,-1)` —— 过程状态初值对**每个样本逐位相同**，输入信息只经 `C`（context）
  的 cross-attention 进入。H-A 的假设：这是"推理起点无输入信息"的瓶颈。
- 与上一轮的**实质差异**：上两轮只改 **spec**（depth / dim），本轮改 **模型前向**（新增一条逐样本初始化路径）。

## §2 交付物清单

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 本轮目标与冻结设计决定 | 本文件 §3 |
| D2 | 新协议/新输出/新 code.zip | `preparation_protocol.json`、`prepared_protocol.json`、`code.zip` + `archive_receipt.json` |
| D3 | 三 seed 真实训练与评分 | `seed{41,42,43}_receipt.json`、每 seed 五 lead 的 `boundary_rmse.csv` |
| D4 | 配对读数 | `readings.json`（本臂 vs 2400 剂量臂 vs 窄臂） |
| D5 | 登记与精确 CI | `docs/R7_S3_PROCESS_INIT_RESULT.md`、index record、brief、账本行、§8、campaign-state |

## §3 判据与设计决定（本轮冻结；科学门只引用冻结文档）

科学门与验收形式只来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` 与
`docs/plans/0022-s3-architecture-next-round.md`（H-A 的六项口径）；本轮**不触碰**任何阈值，也**不宣布**通过。

- **臂**：已登记 2400 剂量臂 + **`process_init_conditioning=True`**（唯一因子）。`depth`=4、`dim`=192、
  K=4、宽 129×129 输入、`interior_32` 监督、2400 更新、注册物理权重、seed 41/42/43 逐字不变。
- **对照**：`record:s3-wide-interior-dose`（pinned 复用、不重训练）为主对照；`record:s3-rollout-dose` 为次级。
- **实现约束（模型改动，须 `[model-digest-change]`）**：
  - 在 `model/process_forecast_r7.py` 构造器**最后、在 `isolated_stream()` 下**加
    `self.process_init_proj=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim))`（仅当新 spec 开关为真）。
  - 在 `:434` 之后加 `process = process + self.process_init_proj(context.mean(dim=1)).unsqueeze(1)`（仅当开关为真）。
  - 因模块最后构造且在 rewind stream 下，**"off"路径逐位等于现实现**（须逐 seed 断言）。
- **正/负控制**：正控制＝把 `process_init_proj` 的 `Linear` 权重/偏置清零 ⇒ 输出逐位退回注册臂（恒等反证）；
  负控制＝**matched_generic** 同参数量/FLOPs、只改路由（broadcast 而非逐样本），用于区分"信息增益"与"单纯加参数"。
  必须分报新增参数量与 FLOPs，同 updates 非同算力分开报。
- **预算**：新增一个 `Sequential(LayerNorm, Linear(192,192))` ≈ +37k 参数（远小于 depth/width 轮），
  per-seed/planned/hard 按 depth 轮实测**重新标定**（先冻后跑，不在运行中改）。
- **预声明出口**：**72h 三 seed 全正 → 提请独立验收**；否则**关闭 H-A 这一支**，不调 proj 宽度/不加 seed。
- **test 策略**：`test.jsonl` 从不读取；`test_read=false`、`scientific_claim=false`。

## §4 数字预算与停止

- 每 seed 上限、planned（软预算）、hard（宽松硬上限）在实现时按 depth 轮实测重新标定后写入驱动常量；
  软超继续并记 overrun，hard 截止即停 attempt 并全额留 `failure.json`。
- 每阶段 spawn 前做**只读** `gpu_gate`；`estimated_peak_bytes` 按 ≥11.90 GiB 标定
  （width 轮实测 11.9023 GiB，连续两轮低估）。**禁止**对任何非本实验进程发送信号（R-054 / 决策 0026）。
- **双卡分配（决策 0047）**：`device_mode: "dual"`，冻结块记 `gpu_uuids` 与 `neighbour_reserve_bytes = 6 GiB`。
- 失败 attempt 原样保留、全额计费、不做 in-place resume。

## §5 明确不做

不重训练任何已登记臂、不重跑更早的已登记轮次、不读 test、不改 `data/raw|interim|processed`、不改归档快照、
不合成替代数据、不删部分输出复活、不改科学合同与冻结证据与判据、不 main 合并/发版/关 issue、
不 paid/rental/exclusive/force/mirror/破坏性/cron。**不重开容量轴**（决策 0048 已永久关闭）。

## §6 进度与交接

- **已定**：设计（§3）、登记名（成功版 `docs/R7_S3_PROCESS_INIT_RESULT.md` / record
  `s3-process-init-result`；本因子**尚无**已登记 record，故无 correction 指针）。
- **待做（开工前 CPU 工作）**：改 `model/process_forecast_r7.py`（新开关 + 最后构造的 `process_init_proj` +
  `:434` 注入）、派生 `scripts/study_r7_s3_process_init.py`、写 `tests/test_r7_s3_process_init.py`
  （含"清零 ⇒ 逐位退回"反证与"off 逐位不变"断言）、逐行 diff 单因素保真、独立复核（≤3）、
  `round-guard` 与 `commit-guard`、提交推工作分支并等 CI 绿（提交信息须带 `[model-digest-change]`）。
- **第一条命令**（若手工跑）：`.venv/bin/python tools/r7_round.py round-guard` → 重建 zip →
  `.venv/bin/python tools/r7_round.py freeze` → `stage prepare/archive/stage-seeds/reading`。
- **本轮尚未开跑**：0 GPU-h；无读数、无结论。
