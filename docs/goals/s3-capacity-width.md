# S3：主干容量（宽度）单因素——决策 0048 定义的最后一条容量臂

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-capacity-width", "driver": "scripts/study_r7_s3_capacity_width.py", "output_stem": "r7_s3_capacity_width_20261013", "control_record": "s3-wide-interior-dose", "seeds": [41, 42, 43], "device_mode": "dual", "test_path": ["tests/test_r7_s3_capacity_width.py"], "commit_paths": ["scripts/study_r7_s3_capacity_width.py", "tests/test_r7_s3_capacity_width.py", "docs/goals/s3-capacity-width.md"], "commit_message": "feat(r7): implement the S3 capacity-width (trunk dim 192->256) single-factor arm", "register_paths": ["docs/R7_S3_CAPACITY_WIDTH_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md"], "register_message": "feat(r7): register the S3 capacity-width (dim 192->256) run as a new evidence record", "code_archive": "outputs/r7_s3_capacity_width_code.zip"} -->

状态 **设计已定、待实现驱动与 CPU 前置**（2026-10-13）。唯一主计划
`docs/goals/main-model-climatology-campaign.md`；承接决策 **0048**
（`docs/decisions/0048-capacity-axis-scope-and-cap.md`）：深度轴已按 index74 关闭
（`interior_32` t2m 72h skill −0.1512/−0.2545/−0.1432 三 seed 全负），**宽度是最后一个容量臂**；
若宽度臂也未能让 72h 三 seed 全正，则**整条容量轴（深度＋宽度）永久关闭**。

## §0 Objective（单段）

> 在 docs/goals/s3-capacity-width.md 接续 S3/index74/63.0619 GPU-h、2023 test 未评分 r0、S4 未启动。
> depth 轴已关闭（三 seed 72h 全负）。本轮回主线的**最后一根容量轴**：**只把主干 `dim` 192 → 256**，
> `depth`/K/宽 129×129 输入/`interior_32` 监督/2400 更新/注册物理权重/seed 41/42/43/评分域/气候态分母
> 逐字不变。**已知混杂（决策 0048 第 4 条）**：宽度**没有逐位恒等扩张**——`dim` 192→256 时 state_dict
> 155 个张量里 150 个形状变化，注册 v3-BD 1600 父（dim-192/depth-4）装不进去；depth 轮靠
> identity-gated 镜像+输出投影清零做到"起点逐位等于注册函数"，**宽度没有这种构造**，故"宽度"必然与
> "初始化/优化轨迹"混杂。本 brief 与读数**如实标注**该混杂，**不照搬 depth 的"单因素保真"措辞**。
> 预声明出口：**72h 三 seed 全正 → 提请独立验收；否则整条容量轴永久关闭**，此后只剩训练信号/
> 目标组织/数据（数据按 D3 暂缓、放最最后）。本轮只报读数，**不宣布科学通过**、不称泛化/SOTA、
> 不自行宣布最终 goal 完成。不 paid/rental/exclusive/main/release/issueclose/force/mirror/破坏性/cron，
> 不改科学合同/冻结证据/用户配置，**不读 test**。

## §1 起点与实质差异

- 起点工作分支 `r7/weather-reasoning`；累计 **63.0619 GPU-h**、index **74**。
- 起点证据：`docs/R7_S3_CAPACITY_DEPTH_RESULT.md`（depth 4→8，negative）、
  `record:s3-wide-interior-dose`（对照臂，pinned 复用、不重训练）、
  `record:s3-rollout-dose`（窄 800 臂，次级参照）。
- 与上一轮的**实质差异**：上一轮只改 `depth`（4→8）；本轮只改 `dim`（192→256）。
  这是**两根不同的轴**（决策 0048 §Context）。
- 为何是"回主线"：训练配方四杠杆（输入范围/监督域/剂量/长 lead 权重）与推理深度 K 都已试过且 72h 未过；
  depth 轴刚关闭；宽度是容量轴上**唯一未在当前主线上单因素检验**的一根（0048 第 2/5 条）。

## §2 交付物清单

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 本轮目标与冻结设计决定 | 本文件 §3 |
| D2 | 新协议/新输出/新 code.zip | `preparation_protocol.json`、`prepared_protocol.json`、`code.zip` + `archive_receipt.json` |
| D3 | 三 seed 真实训练与评分 | `seed{41,42,43}_receipt.json`、每 seed 五 lead 的 `boundary_rmse.csv` |
| D4 | 配对读数 | `readings.json`（本臂 vs 2400 剂量臂 vs 窄臂） |
| D5 | 登记与精确 CI | `docs/R7_S3_CAPACITY_WIDTH_RESULT.md`、index record、brief、账本行、§8、campaign-state |

## §3 判据与设计决定（本轮冻结；科学门只引用冻结文档）

科学门与验收形式只来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` 与主计划 §8；本轮
**不触碰**任何阈值，也**不宣布**通过。

- **臂**：已登记最佳臂 + **`dim` 192→256**（`depth` 仍 4、`heads` 仍 4、K 仍 4）。
- **对照**：**已登记的 2400 剂量臂**（`record:s3-wide-interior-dose`，pinned 复用）为主对照；
  窄臂 `record:s3-rollout-dose` 为次级参照。两者都不重训练。
- **初始化与混杂（决策 0048 第 4 条，必须显式处理）**：宽度**无逐位恒等扩张**。本轮采用
  **冻结的零填充扩张**——把注册 v3-BD 1600 父（dim-192）的每个参数张量按**重叠切片**拷进
  dim-256 目标张量、其余位置置零（1-D 参数拷前 192、余 64 置零）。该规则**确定性、可复现**，
  但**不是恒等**：LayerNorm 在 256 维上归一化、且被置零的高 64 维并非严格惰性，
  故本臂测的是"**宽度 + 该零填充初始化**"的联合效应，**不是**宽度单因素。
  此混杂**必须**写进 `protocol.json` 的 `notes`、证据页与 `readings.json`。
- **预算**：宽度 256 比 depth-8（5,065,493 参数）更大（**5,790,389，+76%**，决策 0048 第 5 条），
  但 depth 更浅。先按 depth 轮实测（单 seed ≈ 2.5–3.2 GPU-h）**重新标定** per-seed/planned/hard，
  先冻后跑，不在运行中改。
- **预声明出口**：**72h 三 seed 全正 → 提请独立验收**；否则**整条容量轴（深度＋宽度）永久关闭**
  （决策 0048 第 3 条），不再开任何容量臂（不重试 depth、不再扫 dim、不做宽＋深组合）。
- **test 策略**：`test.jsonl` 从不读取；`test_read=false`、`scientific_claim=false`。

## §4 数字预算与停止

- 每 seed 上限、planned（软预算）、hard（宽松硬上限）在实现时按 depth 轮实测**重新标定**后写入
  驱动常量；软超继续并记 overrun，hard 截止即停 attempt 并全额留 `failure.json`，不运行中改硬限、
  不重试至偶然通过。
- 每阶段 spawn 前做**只读** `gpu_gate`（声明卡 `GPU-408ad137…` 排第一），
  `estimated_peak_bytes` 按**实测重新标定**（不得沿用 8.5 GiB 旧值；depth 轮三 seed 实测峰值
  均为 11.03 GiB，真正兜住的是决策 0047 的 6 GiB 邻居 reserve）。
  **禁止**对任何非本实验进程发送信号或做冻结/终止自动化（R-054 / 决策 0026）。
- **双卡分配（决策 0047）**：`device_mode: "dual"`，冻结块记 `gpu_uuids` 与
  `neighbour_reserve_bytes = 6 GiB`；由 `tools/r7_round.py stage-seeds` 驱动，每卡同一时刻至多一个 worker。
- 失败 attempt 原样保留、全额计费、不做 in-place resume。
- **不要在跑的时候停 workflow**（停止会 `SIGKILL` 在跑的 `world.run` 子进程）。

## §5 明确不做

不重训练任何已登记臂（窄/800 内部监督/2400 剂量/长 lead 权重/K4→8/depth4→8）、不重跑更早的已登记轮次、
不读 test、不改 `data/raw|interim|processed`、不改归档快照、不合成替代数据、不删部分输出复活、
不改科学合同与冻结证据与判据、不 main 合并/发版/关 issue、不 paid/rental/exclusive/force/mirror/破坏性/cron。

## §6 进度与交接

- **已定**：设计（§3）、登记名（成功版 `docs/R7_S3_CAPACITY_WIDTH_RESULT.md` / record
  `s3-capacity-width-result`；本因子**尚无**任何已登记 record，故无 correction 指针）。
- **待做（开工前 CPU 工作）**：派生 `scripts/study_r7_s3_capacity_width.py`（从
  `scripts/study_r7_s3_capacity_depth.py` 派生，把 `capacity_spec()` 的 `depth` 改成 `dim`、
  把 `expand_parent_state` 换成冻结零填充扩张、按实测重标定 `estimated_peak_bytes` 与时长）、
  写 `tests/test_r7_s3_capacity_width.py`、逐行 diff 证明单因素保真、独立复核（≤3）、
  `round-guard` 与 `commit-guard`、提交推工作分支并等 CI 绿。
- **第一条命令**（若手工跑）：`.venv/bin/python tools/r7_round.py round-guard` → 重建 zip →
  `.venv/bin/python tools/r7_round.py freeze` → `stage prepare/archive/stage-seeds/reading`。
- **本轮尚未开跑**：0 GPU-h；无读数、无结论。
