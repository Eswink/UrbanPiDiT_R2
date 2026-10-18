# S3：K 步草稿的**监督组织**（单因素）——注册深监督 vs final-only

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-draft-supervision-organization", "driver": "scripts/study_r7_s3_supervision_organization.py", "output_stem": "r7_s3_draft_supervision_organization_20261018", "control_record": "s3-anchor-isolation-result", "paired_control_record": "s3-anchor-isolation-result", "reference_record": "s3-wide-interior-dose", "seeds": [41, 42, 43], "device_mode": "dual", "test_path": ["tests/test_r7_s3_supervision_organization.py"], "commit_paths": ["scripts/study_r7_s3_supervision_organization.py", "tests/test_r7_s3_supervision_organization.py", "training/r7_recursive_losses.py", "training/r7_autoregressive_rollout.py", "training/r7_long_rollout.py", "training/r7_long_rollout_runner.py", "docs/rules/size-thresholds.md", "docs/goals/s3-supervision-organization.md", "docs/goals/main-model-climatology-campaign.md"], "commit_message": "feat(r7): add the final-draft-only supervision organization single-factor driver", "register_paths": ["docs/R7_S3_DRAFT_SUPERVISION_ORGANIZATION_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md", "docs/goals/s3-supervision-organization.md"], "register_message": "feat(r7): register the draft-supervision organization run as a new evidence record", "code_archive": "outputs/r7_s3_draft_supervision_organization_code.zip"} -->

状态 **已完成（2026-10-19；候选臂 × 三 seed 全部跑完并评分；预声明出口未满足，支关闭）**。唯一主计划
`docs/goals/main-model-climatology-campaign.md`；承接 H-D 与 dense-step 两轮
（`record:s3-anchor-isolation-result`、`record:s3-dense-step-supervision-result`），本轮继续
**训练信号 / 目标组织**方向，取「K 步草稿的监督组织（organization）」这一此前从未单因素测过的缺口。

## §0 Objective（可粘贴；单段）

> 在 docs/goals/s3-supervision-organization.md 接续 S3、record:s3-anchor-isolation-result 与已关闭的 dense-step 轮之后的训练信号方向：在相同当前代码迁移父（attempt04，model_code_sha256 3fe747b2… 不变、无需新迁移）、seed 41/42/43、2400 updates、K=4、宽 129×129 输入、interior_32 监督与注册稀疏物理权重 (1,.5,0,.5,0,0,0,.5,0,0,0,.5) 下，只把 K 轴草稿的监督组织从注册深监督（initial + K=4 个草稿，归一化 linspace(1,2,5)/sum，非最终草稿占 73.333%）改为 final-only（只对 drafts[:, -1] 计算同一个 latitude_weighted_mse(..., mask=interior_32)，权重 1.0），initial 与三个中间草稿不再有直接目标项；模型前向完全不变（仍产生 K+1 个草稿、full-BPTT、无 detach）；控制臂 N 是 pinned record:s3-anchor-isolation-result 的 control 臂（注册深监督、同父/seed/预算，不重训练），外部参照 A 是 pinned record:s3-wide-interior-dose（不重训练）；预声明出口：同时满足 (a) F 的 72h interior_32 t2m skill 在 seed 41/42/43 全部 >0、(b) 同 seed F−N 的 72h ΔRMSE 三 seed 全部 <0（负=候选更优）、(c) F 的 48h skill 三 seed 全部 >0、(d) F−N 在 6/12/24h 的 seed 均值 ΔRMSE ≤0、(e) 桥一致性 N 的 pinned 5-lead RMSE 对注册剂量臂的差落在 anchor 轮已报告的 −4.8e-6…+1.8e-5 K 内 → 提请独立复核/验收；任一 (a)–(d) 不满足 → 永久关闭「K 草稿监督组织：final-only vs 注册 linspace(1,2,K+1) 深监督」支；桥 (e) 失败只写 identity_unresolved=true，不是科学判定；test 不读、r=0、S4 不启动，不宣布科学通过、不称泛化/SOTA。本 objective 为单段并指向本文件。

（objective 字符数：1084；单段。）

## §1 起点与实质差异

- 起点工作分支 `r7/weather-reasoning`；累计 **127.229721 GPU-h**、index **81**（H-D 与 dense-step 均已登记）。
- 起点证据：`docs/R7_S3_ANCHOR_ISOLATION_RESULT.md`（H-D，mixed：F−N 五 lead 全为配对改善，但
  72h 绝对 skill 三 seed 未全正，气候态锚族按预声明关闭）；`docs/R7_S3_DENSE_STEP_SUPERVISION_RESULT.md`
  （dense coverage 支已永久关闭）；`record:s3-wide-interior-dose`（pinned 外部参照）。
- **为什么选这个因子**：注册 recipe 在**每个物理 6h 步**上对 initial + 全部 K=4 个推理草稿计算
  深监督 `latitude_weighted_mse`，权重 `linspace(1,2,5)/sum = [0.133333,0.166667,0.2,0.233333,0.266667]`。
  非最终草稿合计 **73.333%（11/15）**，而真正被推入下一历史并参与评分的 `drafts[:, -1]` 只有
  **26.667%（4/15）**。如果中间草稿的直接目标项稀释了最终草稿目标，那么同一原始损失、同一父/预算/数据下
  final-only 应改善配对读数。文献线索（只作线索，不作证据）：UT arXiv:1807.03819v3 §3.1 只从最终
  decoder state 出目标分布；DEQ arXiv:1909.01377v2 §4.1.2/§7 只对 equilibrium 输出算损失、且明确说
  无显式层就不加 auxiliary loss；HRM arXiv:2506.21734v4 §3 用 detached deep supervision；
  RSM arXiv:2603.15641v1 摘要报告 final-only + detached warm-up 比 deep supervision/长 unroll 训练快
  >20×、误差低约 5×；deep supervision 综述 arXiv:2207.02376v1 摘要/§2 说中间监督通常有益但
  「如何用」未充分研究。**本轮检索未找到**天气模型对 K 草稿监督的系统消融。
- **与 H-D 的区别**：H-D 改推理循环的反馈表征（`anomaly_feedback`），本轮不碰模型前向。
- **与 dense-step 的区别**：dense 轮改的是**物理步** `physical_weights` 的覆盖（5→12 步）；本轮
  物理权重逐字保持注册稀疏轮廓，只改**每个物理步内部** K 轴草稿的组织。
- **与已关闭长 lead 加权的区别**：`record:s3-wide-long-lead-weight` 改 48/72h 物理权重；本轮不动。
- **模型身份**：只改训练损失实现（`training/*.py`），不改 `model/*.py`，故 `model_code_sha256`
  保持 `3fe747b2…`、**无需新迁移**；`training_code_sha256` 会变。

## §2 交付物

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 本轮冻结设计（本文件 §3） | 本文件 |
| D2 | 新驱动 + 新测试 + 默认不变的 final-only 损失模式 | `scripts/study_r7_s3_supervision_organization.py`、`tests/test_r7_s3_supervision_organization.py`、`training/r7_recursive_losses.py`、`training/r7_long_rollout.py`、`training/r7_long_rollout_runner.py` |
| D3 | 新协议 / 新输出 / code.zip | `prepared_protocol.json`、`protocol.json`、`code.zip` + `archive_receipt.json` |
| D4 | 候选臂 F 三 seed 训练与评分 | `seed{41,42,43}_receipt.json`、每 seed 五 lead 的 `boundary_rmse.csv` |
| D5 | 配对读数 | `readings.json`：F−N（pinned control）、F−A（pinned dose）、N−A 桥 |
| D6 | 登记与精确 CI | `docs/R7_S3_DRAFT_SUPERVISION_ORGANIZATION_RESULT.md`、index record、brief、账本行、§8、campaign-state |

## §3 判据与设计决定（本轮冻结）

科学门与验收形式只来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` 与 D4 口径；本轮**不新增**
任何数值阈值，也不宣布通过。

- **唯一改动因子**：K 轴草稿的监督组织。控制 N 使用注册
  `deep_supervised_forecast_mse`：initial + K=4 个草稿，权重 `linspace(1,2,5)/sum`，非最终草稿
  占 73.333%；已核实其 `training_report` 为 `objective=deep_supervised_latitude_area_mse`、
  `internal_deep_supervision=true`、`physical_weights=(1,.5,0,.5,0,0,0,.5,0,0,0,.5)`、K=4。
  候选 F = final-only：只对 `drafts[:, -1]` 计算同一个 `latitude_weighted_mse(..., mask=interior_32)`，
  权重 1.0；initial 与 3 个中间草稿不再有直接目标项。
- **默认路径逐位不变**：`deep_supervised_forecast_mse` 新增 `draft_supervision` 关键字，默认
  `"deep"` 走原算术；测试以 `torch.equal` 对照注册公式，final-only 以 `torch.equal` 对照
  `latitude_weighted_mse(drafts[:, -1], ..., mask=interior_32)`。
- **模型前向完全不变**：仍产生 K+1 个草稿、`drafts[:, -1] == forecast`、full-BPTT、无 detach；
  训练仍要求全部 K+1 草稿存在，不允许 final-only 的 forward fallback。
- **控制臂 N（pinned，不重训练）**：`record:s3-anchor-isolation-result` 的 control 臂
  （`outputs/r7_s3_anchor_isolation_20261015_attempt01/`），同一当前代码迁移父 attempt04、
  同 seed/预算、注册深监督；其 `anomaly_feedback=False` 且 `anchor_output_prior=False`，
  锚不被任何前向路径读取，故在行为上就是注册的无锚模型。
- **外部参照 A（pinned，不重训练）**：`record:s3-wide-interior-dose`
  （`outputs/r7_s3_wide_interior_dose_20261009_attempt02/`）。
- **桥一致性 (e)**：N 的 pinned 5-lead `interior_32` t2m RMSE 对 A 的差必须落在 anchor 轮已报告的
  `−4.8e-6…+1.8e-5 K`；这是**身份检查**，失败只写 `identity_unresolved=true`，不是科学判据。
- **预声明出口**（不新增阈值，只沿用既有 skill>0 绝对气候态门与配对符号）：
  - (a) F 的 72h `interior_32` t2m skill 在 seed 41/42/43 **全部 >0**；
  - (b) 同 seed F−N 的 72h ΔRMSE **三 seed 全部 <0**（负值 = F 更优）；
  - (c) F 的 48h skill **三 seed 全部 >0**（D4 既有绝对门）；
  - (d) F−N 在 6/12/24h 的 **seed 均值** ΔRMSE **≤0**（不允许短 lead 退化换长 lead）；
  - (e) 桥一致性如上。
  **同时满足 (a)–(e) → 提请独立复核/验收**；任一 (a)–(d) 不满足 → **永久关闭**
  「K 草稿监督组织：final-only vs 注册 linspace(1,2,K+1) 深监督」支：不扫中间草稿权重、不加 seed、
  不调 2400 updates、不改 partial/加权 final-only、不与 anchor/feedback/architecture/容量组合、
  不换名重开，按 negative/mixed 如实登记。
- **混杂声明**：final-only 必然同时移除 4 个非最终直接项**并**把最终项归一化权重从 0.2667 提到 1.0；
  两侧权重都归一化为 sum=1，总损失尺度可比，但本轮把「组织」作为单一因子整体检验，**不声称分离
  support 与 emphasis**；若通过，后续仍需匹配最终权重的对照才能做机制归因。
- **工程身份门**（只算 attempt 失败，修复仅限已确认工程原因，不改因子或判据）：默认路径非逐位相同、
  final-only 损失不等于最后草稿 MSE、父/模型/数据身份漂移。
- **报告口径**：五个 lead（6/12/24/48/72h）与**全 17 变量**的 `interior_32` RMSE 都照报；48h 单独
  列出但不单独作门；不跨单位平均、不挑赢家。
- **test 策略**：`test.jsonl` 从不读取；`test_read=false`、`scientific_claim=false`。

## §4 预算与停止

- **执行预算（冻结）**：`per_seed_seconds=21600`、`planned_seconds=43200`（软预算）、
  `hard_cap_seconds=86400`（硬上限，为计划的 2 倍）；软超预算继续并记 overrun，硬截止即停 attempt
  并全额保留 `failure.json`；失败 attempt 不在原目录 resume、不缩短重试。
- 只训练 **F 一个臂 × 三 seed**（N 与 A 均 pinned 复用），三 seed 由 `stage-seeds` 在双卡上按
  决策 0047 调度；每阶段 spawn 前只读 `gpu_gate`，**禁止**对非本实验进程发送任何信号（R-054）。
- 计划成本约 3 ×（2400 updates + 五 lead 评分）≈ 6–8 GPU-h；实际按回执逐阶段记账。
- **停止条件**：任一 (a)–(d) 不满足即永久关闭该支；工程失败 / 身份门 / hard cap 只算 attempt
  失败，修复仅限已确认工程原因，不改因子或判据；test 未读、r=0、S4 不启动。

## §5 明确不做

不重训练 N 或 A、不重跑已登记轮次、不读 test、不扫中间草稿权重、不加 seed、不改 2400 updates、
不调 K/dim/depth/监督域/物理权重、不与 anchor/feedback/architecture/容量组合、不改
`data/raw|interim|processed` 或归档快照、不合成替代数据、不改科学合同/冻结证据/判据、
不 main 合并/发版/关 issue、不 paid/rental/exclusive/force/mirror/破坏性/cron。

## §6 进度与交接

- **已定**：设计（§3）；登记名 `docs/R7_S3_DRAFT_SUPERVISION_ORGANIZATION_RESULT.md` / record
  `s3-draft-supervision-organization-result`（`mixed` / `not-candidate`）。
- **已完成（2026-10-18，实现轮）**：① 训练损失新增 `draft_supervision="final_only"` 模式，
  默认 `"deep"` 算术未动；② 新驱动、新 CI-safe 测试（合成张量/合成树/monkeypatch、含反证）；
  ③ 本长文与 campaign-state 指针已写。
- **已完成（2026-10-19，GPU 轮 attempt01）**：freeze → prepare → archive → seed 41/42/43 → reading
  六阶段全 success、`returncode=0`；三 seed 均 `resumed_from_updates=0`、2400 updates、五 lead 全评
  （cohort 472/468/460/444/428）；无跨 attempt seed 复用。**结果**：预声明出口 (a)(b)(d) 未满足、
  (c)(e) 满足 ⇒ `predeclared_exit_met=false`、`scientific_exit_met=false`、
  `identity_unresolved=false`，按预声明**永久关闭**本支。成本 7.052733205904207 GPU-h，
  soft/hard overrun 0，test 未读、r=0。证据页
  `docs/R7_S3_DRAFT_SUPERVISION_ORGANIZATION_RESULT.md`，index record
  `s3-draft-supervision-organization-result`。
- **下一动作（S3）**：K 草稿监督组织支已按预声明永久关闭（不扫中间草稿权重、不加 seed、不调
  2400 updates、不改 partial/加权 final-only、不与 anchor/feedback/architecture/容量组合、
  不换名重开）；下一根独立的**训练信号 / 目标组织**轴是 **K 步草稿的单调改进约束**
  （要求后一草稿的 loss 不差于前一草稿的 hinge；不增容量、不加数据、不改模型前向）——
  先写新的 round brief、先检索一手资料，再冻结新协议；数据扩充按 D3 继续暂缓。
- **恢复第一条命令**：`.venv/bin/python tools/r7_round.py state`。
- test 未读、r=0、S4 未启动；不宣布科学通过、不称泛化/SOTA。
