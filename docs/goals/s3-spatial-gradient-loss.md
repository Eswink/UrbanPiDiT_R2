# S3：空间梯度损失（单因素）——注册深监督 MSE + 冻结 first-difference 惩罚

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-spatial-gradient-loss", "driver": "scripts/study_r7_s3_spatial_gradient_loss.py", "output_stem": "r7_s3_spatial_gradient_loss_20261019", "control_record": "s3-anchor-isolation-result", "paired_control_record": "s3-anchor-isolation-result", "reference_record": "s3-wide-interior-dose", "seeds": [41, 42, 43], "device_mode": "dual", "test_path": ["tests/test_r7_s3_spatial_gradient_loss.py"], "commit_paths": ["scripts/study_r7_s3_spatial_gradient_loss.py", "tests/test_r7_s3_spatial_gradient_loss.py", "training/r7_recursive_losses.py", "training/r7_autoregressive_rollout.py", "training/r7_long_rollout.py", "training/r7_long_rollout_runner.py", "docs/goals/s3-spatial-gradient-loss.md", "docs/goals/main-model-climatology-campaign.md"], "commit_message": "feat(r7): add the spatial-gradient-loss single-factor driver", "register_paths": ["docs/R7_S3_SPATIAL_GRADIENT_LOSS_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md", "docs/goals/s3-spatial-gradient-loss.md"], "register_message": "feat(r7): register the spatial-gradient-loss run as a new evidence record", "code_archive": "outputs/r7_s3_spatial_gradient_loss_code.zip"} -->

状态 **实现完成、GPU 轮未开始（2026-10-19）**。唯一主计划
`docs/goals/main-model-climatology-campaign.md`；承接训练信号/目标组织方向，接在已关闭的
dense-step 与 draft-organization 两支之后。本轮只检验「注册 MSE 的平滑偏差在 2400 updates 下
是否是约束」这一可证伪假设。

## §0 Objective（可粘贴；单段）

> 在 docs/goals/s3-spatial-gradient-loss.md 接续 S3 与已登记的 s3-anchor-isolation-result、s3-wide-interior-dose 参照：在相同当前代码迁移父（attempt04，model_code_sha256 3fe747b2… 不变、无需新迁移）、seed 41/42/43、2400 updates、K=4、depth=4、dim=192、宽 129×129 输入、interior_32 监督、注册稀疏物理权重 (1,.5,0,.5,0,0,0,.5,0,0,0,.5) 与注册 K 轴深监督 linspace(1,2,5)/sum（final_weight=2）下，唯一改动是在每个物理 6h 步、每个 K+1 草稿的 per-draft 损失上加冻结的 λ_grad=0.1 first-difference 空间梯度惩罚：L = L_mse + 0.1 * L_grad，L_grad 为相邻格点对（两端都在 boundary_masks(129,129,(32,))['interior_32'] 内；4160 个 H 对、4160 个 W 对；H 对权重=对两端 cos(lat) 均值，W 对权重=所在行 cos(lat)，按选中对归一化）上的纬向加权前向差分平方误差均值。λ_grad 是设计系数而非通过/关闭阈值，不扫参；λ_grad=0.0 走独立 Python 分支且与注册损失逐位相同（torch.equal）。模型前向、数据、评分域、气候态分母、seed 与预算全部不变，只改训练损失代码。预声明出口（不新增数值阈值）：(a) 候选 72h interior_32 t2m skill 三 seed 全 >0；(b) 同 seed 候选−控制 72h ΔRMSE 三 seed 全 <0（控制=pinned record s3-anchor-isolation-result 的 control 臂，不重训练）；(c) 候选 48h skill 三 seed 全 >0；(d) 6/12/24h 的 seed 均值 ΔRMSE ≤0；(e) 桥一致性：控制−pinned s3-wide-interior-dose 的 RMSE 落在 anchor 轮报告的 −4.8e-6…+1.8e-5 K 内。全部 (a)–(e) → 提请独立复核；任一 (a)–(d) 不满足 → 永久关闭空间梯度损失支（不扫 λ、不加 seed、不改 updates、不与其他已关轴组合、不换名重开）；(e) 失败只写 identity_unresolved=true，是工程身份失败而非科学判定。test 不读、r=0、S4 不启动、scientific_claim=false。

（objective 字符数：1184；单段。）

## §1 起点与实质差异

- 起点工作分支 `r7/weather-reasoning`；累计 **134.282454 GPU-h**、index **82**（draft-organization 已登记）。
- 起点证据：`docs/R7_S3_DRAFT_SUPERVISION_ORGANIZATION_RESULT.md`（mixed，K 轴组织支按预声明关闭）；
  `docs/R7_S3_ANCHOR_ISOLATION_RESULT.md`（H-D mixed，其 control 臂是本轮配对控制）；
  `record:s3-wide-interior-dose`（pinned 外部参照）。
- **为什么选这个因子**：一手线索（不是证据）指出 MSE 训练会通过 double penalty 抑制不可预测尺度的
  振幅（Subich et al., arXiv:2501.19374v2，摘要与 §2.1/§3.1），而频率原则说明 DNN 先拟合低频
  （Xu et al., arXiv:2201.07395v4，§2.1/§2.2）。FastNet 使用全局谱振幅损失且不报告 48/72h RMSE；
  本轮检索**未找到**区域网格 first-difference/Laplacian 空间梯度损失在干净 48/72h 物理 RMSE 消融
  下的第一手研究（已核全文 2501.19374、2212.12794、2304.02948、2312.03876、2409.05975、2510.09484、
  2609.37038、2607.21080；主机上 arXiv 检索/API 被限流）。因此这是一个可证伪的机制筛选，不是对文献
  结论的复现。
- **与已关闭支的区别**：不是 typed/spatial model-mixing（model/*.py 不动）、不是 H-D 谱/lead 重加权、
  不是 K 草稿组织（`draft_supervision` 仍为 `deep`，K 轴权重逐字不变），也不是物理步覆盖（12 步权重不变）。
- **模型身份**：只改训练损失实现（`training/*.py`），故 `model_code_sha256` 保持 `3fe747b2…`、
  **无需新迁移**；`training_code_sha256` 会变。

## §2 交付物

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 本轮冻结设计（本文件 §3） | 本文件 |
| D2 | 新驱动 + 新测试 + 默认逐位不变的 λ=0 分支 | `scripts/study_r7_s3_spatial_gradient_loss.py`、`tests/test_r7_s3_spatial_gradient_loss.py`、`training/r7_recursive_losses.py`、`training/r7_autoregressive_rollout.py`、`training/r7_long_rollout.py`、`training/r7_long_rollout_runner.py` |
| D3 | 新协议 / 新输出 / code.zip | `preparation_protocol.json`、`protocol.json`、`code.zip` + `archive_receipt.json` |
| D4 | 候选臂三 seed 训练与评分 | `seed{41,42,43}_receipt.json`、每 seed 五 lead 的 `boundary_rmse.csv` |
| D5 | 配对读数 | `readings.json`：候选−控制、候选−pinned dose、控制−pinned dose 桥 |
| D6 | 登记与精确 CI | `docs/R7_S3_SPATIAL_GRADIENT_LOSS_RESULT.md`、index record、brief、账本行、§8、campaign-state |

## §3 判据与设计决定（本轮冻结）

科学门与验收形式只来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` 与 D4 口径；本轮**不新增**
任何数值阈值，也不宣布通过。

- **唯一改动因子**：训练目标。对每个物理 6h 步、每个 K+1 草稿：
  `L = L_mse + λ_grad * L_grad`，`λ_grad=0.1`；`L_mse` 是注册的
  `latitude_weighted_mse(..., mask=interior_32)`；`L_grad` 是相邻格点对上的纬向加权前向差分
  平方误差均值：
  `Σ_H w_h (Δ_h pred − Δ_h target)^2 + Σ_W w_w (Δ_w pred − Δ_w target)^2` 除以
  `C * (Σ_H w_h + Σ_W w_w)`（C=17 个通道；与 `latitude_weighted_mse` 的
  `combined.expand_as(error)` 一致，17 个相同通道的 `L_grad` 与单通道相同）。选中对 = 两端都在
  `boundary_masks(129,129,(32,))['interior_32']` 内（4160 H 对、4160 W 对）；
  `w_h = mean(cos(lat) endpoints)`，`w_w = cos(lat) row`。
- **K 轴不变**：仍用 `linspace(1,2,K+1)/sum` 深监督，`final_weight=2`，非最终草稿质量 73.333%；
  `draft_supervision="deep"`，`gradient_lambda=0.0` 是与注册损失逐位相同的独立分支
  （`torch.equal` 测试；且 λ=0 时惩罚函数不被调用）。
- **物理轴不变**：12 步权重 `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)`，直接监督步 `[0,1,3,7,11]`。
- **模型/数据/预算不变**：宽 129×129 输入、17 通道、K=4、depth=4、dim=192、2400 updates、
  迁移父 attempt04、seeds 41/42/43、LR 2e-5、warmup 10、FP32、batch 1、clip 1。
- **模型 spec 结构**：candidate 使用 H-D 对照 base spec 去掉三个惰性锚键
  （`climatology_anchor_spec`、`anchor_output_prior=False`、`anomaly_feedback=False`）后的同一个
  dict（canonical digest `d69e33…`）；复核者用 H-D 正控制实测
  `unanchored_equals_off=True`、`unanchored_max_abs=0.0`、`bit_identical_outside_anchor=True`
  （本次实现轮未重跑该正控制）。candidate spec 不允许任何锚键（`arm_spec()` 会拒绝）。
- **控制臂 N（pinned，不重训练）**：`record:s3-anchor-isolation-result` 的 control 臂
  （`outputs/r7_s3_anchor_isolation_20261015_attempt01/`），注册深监督、无锚、`spatial_gradient_lambda=0`。
- **外部参照 A（pinned，不重训练）**：`record:s3-wide-interior-dose`。
- **桥一致性 (e)**：N 的 pinned 5-lead `interior_32` t2m RMSE 对 A 的差必须落在
  `−4.8e-6…+1.8e-5 K`；这是**身份检查**，失败只写 `identity_unresolved=true`，不是科学判据。
- **预声明出口**（不新增阈值，只沿用既有 skill>0 绝对气候态门与配对符号）：
  - (a) 候选 72h `interior_32` t2m skill 在 seed 41/42/43 **全部 >0**；
  - (b) 同 seed 候选−控制 72h ΔRMSE **三 seed 全部 <0**（负值 = 候选更优）；
  - (c) 候选 48h skill **三 seed 全部 >0**；
  - (d) 候选−控制在 6/12/24h 的 **seed 均值** ΔRMSE **≤0**；
  - (e) 桥一致性如上。
  **同时满足 (a)–(e) → 提请独立复核/验收**；任一 (a)–(d) 不满足 → **永久关闭**
  「空间梯度损失」支：不扫 λ、不加 seed、不调 updates、不与 anchor/feedback/architecture/容量/K 轴
  组织组合、不换名重开，按 negative/mixed 如实登记。
- **工程身份门**（只算 attempt 失败，修复仅限已确认工程原因，不改因子或判据）：λ=0 路径非逐位相同、
  λ>0 未进入报告/合同、父/模型/数据身份漂移。
- **报告口径**：五个 lead（6/12/24/48/72h）与**全 17 变量**的 `interior_32` RMSE 都照报；48h 单独
  列出但不单独作门；不跨单位平均、不挑赢家。
- **test 策略**：`test.jsonl` 从不读取；`test_read=false`、`scientific_claim=false`。

## §4 预算与停止

- **执行预算（冻结）**：`per_seed_seconds=21600`、`planned_seconds=43200`（软预算）、
  `hard_cap_seconds=86400`（硬上限，为计划的 2 倍）；这是单臂预算（H-D 两臂对照为
  43200/86400/115200 的墙钟预算，2400 updates 的训练预算本身不变）。软超预算继续并记 overrun，硬截止即停 attempt
  并全额保留 `failure.json`；失败 attempt 不在原目录 resume、不缩短重试。
- 只训练 **候选一个臂 × 三 seed**（N 与 A 均 pinned 复用），三 seed 由 `stage-seeds` 在双卡上按
  决策 0047 调度；每阶段 spawn 前只读 `gpu_gate`，**禁止**对非本实验进程发送任何信号（R-054）。
- 计划成本约 3 ×（2400 updates + 五 lead 评分）≈ 6–8 GPU-h；实际按回执逐阶段记账。
- **停止条件**：任一 (a)–(d) 不满足即永久关闭该支；工程失败 / 身份门 / hard cap 只算 attempt
  失败，修复仅限已确认工程原因，不改因子或判据；test 未读、r=0、S4 不启动。

## §5 明确不做

不重训练 N 或 A、不重跑已登记轮次、不读 test、不扫 λ、不加 seed、不改 2400 updates、不调
K/dim/depth/监督域/物理权重、不改 K 轴深监督与 final_weight、不与 anchor/feedback/architecture/
容量/K 轴组织组合、不改 `data/raw|interim|processed` 或归档快照、不合成替代数据、不改科学合同/
冻结证据/判据、不 main 合并/发版/关 issue、不 paid/rental/exclusive/force/mirror/破坏性/cron。

## §6 进度与交接

- **已定**：设计（§3）；登记名 `docs/R7_S3_SPATIAL_GRADIENT_LOSS_RESULT.md` / record
  `s3-spatial-gradient-loss-result`。
- **已完成（2026-10-19，实现轮）**：① 训练损失新增 `gradient_lambda` 与
  `spatial_gradient_mse`，λ=0 走独立分支、逐位保持注册算术；② 新驱动、新 CI-safe 测试
  （合成张量/合成树/monkeypatch、含反证，不读 outputs/）；③ 本长文与 campaign-state 指针已写。
- **未做**：freeze/prepare/archive/GPU 训练/评分/登记均未开始；`round-guard` 之后由 workflow
  继续冻结与执行。
- **下一动作（S3）**：通过 `round-guard` 后 `freeze` → `archive` → `stage-seeds` → `reading`；
  若预声明出口失败则永久关闭本支，下一独立轴按主计划 §8 决定。
- **恢复第一条命令**：`.venv/bin/python tools/r7_round.py state`。
- test 未读、r=0、S4 未启动；不宣布科学通过、不称泛化/SOTA。
