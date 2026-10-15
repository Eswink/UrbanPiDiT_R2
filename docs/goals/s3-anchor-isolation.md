# S3：孤立气候态锚 package 的**异常反馈**机制（单因素）——H-C 关闭后转入训练信号/目标组织方向

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-anchor-isolation", "driver": "scripts/study_r7_s3_anchor_isolation.py", "output_stem": "r7_s3_anchor_isolation_20261015", "control_record": "s3-wide-interior-dose", "seeds": [41, 42, 43], "device_mode": "dual", "test_path": ["tests/test_r7_s3_anchor_isolation.py"], "commit_paths": ["model/weather_forecaster_r7.py", "model/process_step_r7.py", "model/process_forecast_r7.py", "scripts/r7_s3_anchor_isolation_parent.py", "scripts/study_r7_s3_anchor_isolation.py", "tests/test_r7_s3_anchor_isolation.py", "docs/goals/s3-anchor-isolation.md", "docs/rules/size-thresholds.md"], "commit_message": "feat(r7): isolate the climatology-anchor anomaly-feedback mechanism as a single factor [model-digest-change]", "register_paths": ["docs/R7_S3_ANCHOR_ISOLATION_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md"], "register_message": "feat(r7): register the H-D anchor-isolation run as a new evidence record", "code_archive": "outputs/r7_s3_anchor_isolation_code.zip"} -->

状态 **设计已定、待实现模型开关、锚表派生与 CPU 前置**（2026-10-15）。唯一主计划
`docs/goals/main-model-climatology-campaign.md`；承接 **H-C 关闭**（`record:s3-global-mixing-result`，negative）、
**H-A/H-B 关闭**与**决策 0048**（容量轴永久关闭）。本轮回主线 §8 指向的**训练信号/目标组织**方向。

## §0 Objective（单段）

> 在 docs/goals/s3-anchor-isolation.md 接续 S3/index79/105.619618 GPU-h、2023 test 未评分 r0、S4 未启动；
> 架构方向（H-A/H-B/H-C）与容量轴均已关闭，本轮回主线的**训练信号/目标组织**方向，取
> `record:s3-climatology-anomaly-anchor` 那一族里**唯一可辩护的缺口**——该 package 自己写明
> *"同时改变输出先验、proposal 先验及 feedback 表示，故…不是对单一机制的因果归因"*——**孤立出其中的
> 「推理循环的反馈表征」这一个机制**：`anomaly_feedback`（把 K 步循环的草稿反馈从绝对场改成
> `draft − C_valid` 的异常场），并**明确不动输出先验与 proposal 先验**（二者由仓库自己的校验规则
> 与「锚存在」耦合，无法再分，故本轮不假装分开）；为做到这一点新增**一个**开关 `anchor_output_prior`
> （默认 False：锚可以「装着但不作为输出先验」），使新增臂与注册臂在**初始化时逐位相同**、只差
> `anomaly_feedback` 一个因子；其余（宽 129×129 输入、`interior_32` 监督、2400 更新、K=4、depth=4、
> `dim`=192、注册物理权重、seed 41/42/43、评分域、气候态分母）逐字不变；协议内同跑**两臂**（N=控制、
> F=候选）＋已登记 2400 剂量臂 A 作 pinned 外部参考；正控制＝**把锚表清零 ⇒ `draft − 0 = draft` ⇒ F 与 N
> 逐位相同**（须逐 seed 断言，失败即 `raise`）；负控制＝N 本身（锚已装、两个消费者都关 ⇒ 初始化须与
> 注册臂逐位相同）；预声明出口：**72h 三 seed 全正 且 F 在 6/12/24h 不差于 N → 提请独立验收；否则
> 永久关闭「气候态锚」这一族**（不调表粒度/不减锚/不加 seed）；本轮只报读数、**不宣布科学通过**、
> 不称泛化/SOTA、不自行宣布最终 goal 完成；不 paid/rental/exclusive/main/release/issueclose/force/mirror/
> 破坏性/cron，不改科学合同/冻结证据/用户配置，**不读 test**。

## §1 起点与实质差异

- 起点工作分支 `r7/weather-reasoning`；累计 **105.619618 GPU-h**、index **79**。
- 起点证据：`docs/R7_S3_GLOBAL_MIXING_RESULT.md`（H-C，negative）、`docs/R7_S3_CLIMATOLOGY_ANOMALY_ANCHOR.md`
  （`record:s3-climatology-anomaly-anchor`，negative **但含正项**）、`record:s3-wide-interior-dose`（外部对照，pinned）。
- **为什么选这个因子（唯一可辩护的缺口，逐条有据）**：
  1. 该 package 的**实测**是 t2m **48/72h 相对两个对照都改善**（48h 13.708 vs 16.062/14.018 K²；
     72h 8.769 vs 15.768/19.464 K²），而 6/12/24h 退化，故预声明的 necessary-support 门判 false。
     **它是本 campaign 至今唯一在长 lead 上有正向线索的方向。**
  2. 但该 package **同时**改了输出先验（`Y0 = C_valid + A0`）、proposal 先验（`anchored_proposal(..., climate)`）
     与 feedback 表示（`E(Y_k − C_valid)`），其 limitations 自己写明 *"no isolated mechanism attribution"*。
     **缺口的本体是「无法归因」，不是「机制本身没用」。**
  3. **输出先验与 proposal 先验在仓库里无法分开**：`model/process_forecast_r7.py:65-66`
     `if anomaly_feedback and climatology_anchor_spec is None: raise`，而**锚一旦存在**，
     `model/weather_forecaster_r7.py:109-113` 的输出头锚与 `model/process_step_r7.py:147` 的 proposal 锚
     **同时**改成 `C_valid`。故「孤立输出先验」在本仓**不是一个因子而是两个**（输出先验＋proposal 先验），
     除非再加一个开关——**本轮不再加第二个开关**去假装分开它们。
  4. **输出先验还有独立的、与本题无关的失败风险**：package 的 t2m 6h MSE 是 **8.201**，
     对两个对照 3.754/3.480（约 **2.2×** 更差），24h 17.953 对 7.174/9.087。这不是边际退化，
     而是把父状态在输出空间的语义整体换掉（父预测全场，改成预测异常后父的权重不再对应）。
     而 D4 验收门要求**各 lead** skill>0 且 u10/v10/mslp 不得更差，故输出先验臂**几乎必然**因
     与本题无关的理由不过门。**因此本轮把输出先验关掉**（`anchor_output_prior=False`），
     只动反馈表征——这才是能干净孤立、且 package 自己最清楚点名的那一个机制。
- 与上一轮的**实质差异**：H-C 改的是主干出口的**空间混合范围**（架构）；本轮改的是**推理循环看到的反馈表征**
  （训练信号/信息组织），两者互不重叠。两轮都改 `model/*.py`（故都需父重迁移）。

## §2 交付物清单

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 本轮目标与冻结设计决定 | 本文件 §3 |
| D2 | **宽 129×129 train-only 气候态锚表**（新派生数据集，身份齐全） | `outputs/r7_s3_anchor_table_<date>/`：spec JSON（`r7-train-climatology-anchor-v1`）＋table `.pt`＋`BUILD_COMPLETE`＋preflight 报告 |
| D3 | 新协议/新输出/新 code.zip | `preparation_protocol.json`、`prepared_protocol.json`、`code.zip` + `archive_receipt.json` |
| D4 | **两臂**三 seed 真实训练与评分 | `seed{41,42,43}_{control,candidate}_receipt.json`、每 seed 每臂五 lead 的 `boundary_rmse.csv` |
| D5 | 配对读数 | `readings.json`（F−N 同协议；两臂 vs 已登记 2400 剂量臂 pinned） |
| D6 | 登记与精确 CI | `docs/R7_S3_ANCHOR_ISOLATION_RESULT.md`、index record、brief、账本行、§8、campaign-state |

## §3 判据与设计决定（本轮冻结；科学门只引用冻结文档）

科学门与验收形式只来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` 与 D4 口径；本轮**不触碰**任何阈值，
也**不宣布**通过。

- **协议内控制臂 N**：锚**已装**（`climatology_anchor_spec=<宽 train-only 表>`）但
  `anchor_output_prior=False`、`anomaly_feedback=False`。**两个消费者都关 ⇒ 锚不被任何前向路径读取**，
  故 N 在**初始化时**必须与注册臂（无锚）**逐位相同**（新增的 `backbone.climatology_anchor.{table,ready}`
  两个 persistent buffer 除外，且它们不参与前向）。这是负控制，也是「装锚本身是惰性的」的断言。
- **候选臂 F**：与 N 同锚、同 `anchor_output_prior=False`，**只把 `anomaly_feedback=True`**：
  `model/process_step_r7.py:104` 的 `feedback_draft = draft - climate`。唯一因子。
- **参考臂 A**：`record:s3-wide-interior-dose`（pinned 复用、不重训练）为外部参考。
- **为什么这是单因素**：N 与 F 由**同一 seed、同一预算、同一父状态、同一锚表**构造，且 N 初始化与注册臂
  逐位相同；两臂**只差一个布尔**（`anomaly_feedback`），该布尔只影响 `process_step_r7.py:104` 一处张量减法。
- **正控制（恒等反证，失败即 `raise`）**：把锚**表清零**（`table.zero_()`，`ready` 仍 True）⇒
  `climate = 0` ⇒ `feedback_draft = draft - 0 = draft` ⇒ **F 的前向必须与 N 逐位相同**。
  须逐 seed 在固定合成批上断言 `torch.equal`，不等即 `raise`（不是只写进 receipt）。
- **负控制**：N 本身（锚装着、两个消费者都关）；断言 N 与注册臂初始化逐位相同。
- **新开关（唯一模型改动）**：`anchor_output_prior: bool = **True**`。**默认 True 保持既有语义逐位不变**
  （装锚即维持 package 行为：输出头锚与 proposal 锚都用 `C_valid`）；为 **False** 时：
  - `model/weather_forecaster_r7.py` 的 `forward`：输出头锚**改用 `base`**（最后一帧历史），但
    `R7ForecastOutput.climatology_anchor` **仍返回 `C_valid`**（供 step 反馈使用）；
  - `model/process_step_r7.py` 的 `anchored_proposal`：**改用 `anchor`**（`base`），不用 `climate`。
  **默认取 True 是被实测纠正过的**：最初取 False，结果 `tests/test_r7_climatology_anchor_paths.py`
  **10 例失败**——它把「装锚即改输出头与 proposal」这一既有语义编码进了断言，默认 False 会**静默改变**
  一个既有配置的含义。取 True 后该文件 475 例全过。
  校验规则放在 `ProcessForecastCoReasoner.__init__`（**不是** backbone）：`climatology_anchor_spec`
  在 backbone 构造时**尚未传入**（锚是构造后单独 `_install_climatology_anchor` 装上去的），
  故 backbone 看不到 spec，放那里会把所有无锚模型误判为非法。
  唯一被拒的组合是 `anchor_output_prior=False` 且**无锚**（该开关将无人读取）。
- **锚表派生（D2，本轮新增前置）**：注册实例是**宽 129×129**，而现有合格锚表是 **2026-10-07 的 65×65**
  实例（早于宽 store 的 2026-10-09 建立），**不能在宽网格上复用**。故须从**既有宽 store 的 train split**
  派生 per-(month, hour) 均值表（17 通道 × 129×129），并写齐
  `source_sha256` / `train_data_identity` / `climatology_mean_identity_sha256` / `table_sha256` /
  `bucket_keys` / `bucket_counts` / `training_years` / `selection` 等 spec 必填字段
  （`table_shape` 为 **4 值** `[buckets, C, H, W]`，已由 `TrainClimatologyAnchor._validate_spec` 实测确认）。
  **D3 合规**：不新增下载、不重建 store，只读既有 store 派生；写入**全新排他** `outputs/` 路径；
  先只读 preflight 留 digest，再显式 `--write`；失败留 `failed-no-fallback`。
- **已实测的开关恒等性（合成锚，非真值；实现阶段已完成）**：
  | 断言 | 实测 |
  | --- | --- |
  | 无锚 + 默认 ⇒ 与改动前逐位相同 | 95 个共有键全 `torch.equal` |
  | N（锚装、两消费者关）初始化 ≡ 无锚模型 | 仅多 `backbone.climatology_anchor.{table,ready}` 两键，其余逐位相同 |
  | N 前向 ≡ 无锚前向（锚未被读取） | `max|d| = 0.0` |
  | F 前向 ≠ N 前向（反馈确被读取） | `max|d| = 4.96e-04` |
  | **表清零 ⇒ F 前向 ≡ N 前向**（正控制） | `max|d| = 0.0` |
  | `anchor_output_prior=True` 前向 ≠ 无锚前向 | `max|d| = 2.72` |
  | 默认 ≡ 显式 True | `torch.equal` 为真 |
  | 无锚 + False 被拒 | `ValueError` |
- **预算**：两臂 × 三 seed = 六次训练；per-seed/planned/hard 按 H-C 轮实测（12.287980 GPU-h 七阶段、
  per-seed 约 4.0–4.4 GPU-h）重新标定（先冻后跑，不在运行中改）。
- **预声明出口**：**72h 三 seed 全正 且 F 在 6/12/24h 不差于 N（seed 均值）→ 提请独立验收**；
  否则**永久关闭「气候态锚」这一族**，不调表粒度、不减锚、不加 seed。**若 F−N 无差异 ⇒ 结论是
  "异常反馈表征无效"**，则 package 的 48/72h 增益只可能来自输出先验（而输出先验已实测摧毁短 lead），
  如实写清。
- **test 策略**：`test.jsonl` 从不读取；`test_read=false`、`scientific_claim=false`。

## §4 数字预算与停止

- 每 seed 上限、planned（软预算）、hard（宽松硬上限）在实现时按 H-C 轮实测重新标定后写入驱动常量；
  软超继续并记 overrun，hard 截止即停 attempt 并全额留 `failure.json`。
- 每阶段 spawn 前做**只读** `gpu_gate`；`estimated_peak_bytes` 按 H-C 轮实测峰值标定
  （H-C 七阶段峰值见其 receipts）。**禁止**对任何非本实验进程发送信号（R-054 / 决策 0026）。
- **双卡分配（决策 0047）**：`device_mode: "dual"`，冻结块记 `gpu_uuids` 与
  `neighbour_reserve_bytes = 6 GiB`。
- 失败 attempt 原样保留、全额计费、不做 in-place resume。

## §5 明确不做

不重训练任何已登记臂（A 只 pinned 复用）、不重跑更早的已登记轮次、不读 test、不改
`data/raw|interim|processed`、不改归档快照、不合成替代数据、不删部分输出复活、不改科学合同与冻结证据与判据、
不 main 合并/发版/关 issue、不 paid/rental/exclusive/force/mirror/破坏性/cron。
**不重开架构方向**（H-A/H-B/H-C 均已关闭）、**不重开容量轴**（0048）、
**不重开任何「给损失加权」的臂**（`record:s3-wide-long-lead-weight` 与 `record:s2-78-rb-loss-screening`
已两次失败，同族重开违反防打转）。

## §6 进度与交接

- **已定**：设计（§3）、登记名（成功版 `docs/R7_S3_ANCHOR_ISOLATION_RESULT.md` / record
  `s3-anchor-isolation-result`；本因子此前**无**任何已登记 record，故无 correction 指针）。
- **待做（开工前 CPU 工作）**：① 派生并合格化宽 129×129 train-only 锚表（D2）；
  ② 实现 `anchor_output_prior` 开关（默认 False，off 路径逐位不变）；
  ③ 建 `scripts/r7_s3_anchor_isolation_parent.py`（钉本轮迁移＋装载锚表＋`expand_parent_state(..., seed=...)`）；
  ④ 派生 `scripts/study_r7_s3_anchor_isolation.py`（**R-051 600 行硬限**，两臂 N/F 同协议）；
  ⑤ 写 `tests/test_r7_s3_anchor_isolation.py`（含「表清零 ⇒ F≡N 逐位」「off 开关逐位不变」
  「N 初始化 ≡ 注册臂」「两臂只差一个布尔」反证）；⑥ 逐行 diff 单因素保真、独立复核（≤3）、
  `round-guard` 与 `commit-guard`、提交推工作分支并等 CI 绿。
- **父迁移前提**：本轮改 `model/*.py` ⇒ `model_code_digest()` 会变 ⇒ 注册时的迁移父会被
  `training/r7_experiment.py` 的身份门**按设计**拒绝。**必须**先用**已提交的**
  `scripts/migrate_r7_parent_state.py` 在**当前**代码下重跑迁移，再按
  `scripts/r7_s3_global_mixing_parent.py` 的**同一模式**建本轮 parent 模块。
  **不得**修改比对逻辑或放宽任何门。**迁移一旦跑定，`model/**/*.py` 不得再改。**
- **CI 安全**：测试**不得**依赖 `outputs/`（H-A/H-B/H-C 都在此处让 CI 红过）——收窄 `execution_files()`
  到实际存在的文件并 stub `sha256_file`、`recipe._shared`；在干净 worktree（无 `outputs/`）里实跑确认。
- **第一条命令**：`.venv/bin/python tools/r7_round.py round-guard` → 重建 zip →
  `.venv/bin/python tools/r7_round.py freeze` → `stage prepare/archive/stage-seeds/reading`。
- **本轮尚未开跑**：0 GPU-h；无读数、无结论。

## §7 下一动作（S3）

- 在 **S3** 节点：先派生宽锚表（D2）并合格化，再实现 `anchor_output_prior` 开关与父模块，
  然后派生驱动与 CI-safe 测试，最后 `round-guard` → `freeze` → 逐 stage GPU → 四件套登记。
- 第一条命令：`.venv/bin/python tools/r7_round.py round-guard`。
- 预声明出口见 §3：**72h 三 seed 全正且 F 在 6/12/24h 不差于 N → 提请独立验收；否则永久关闭气候态锚这一族**。
- 本轮**未产生任何读数**；不宣布通过、不关闭任何支。
