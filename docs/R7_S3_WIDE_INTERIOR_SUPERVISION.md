# S3 宽输入 + 内部盒监督：上一轮的 null 主要来自监督稀释

**状态：已完成（2026-10-09）；`scientific_claim: false`；2023 test 未读；r 不消耗。**
本页对应 `docs/goals/main-model-climatology-campaign.md` §8 与
`docs/goals/s3-wide-interior-supervision.md`。它是**开发筛选读数**，不是科学确认，也不是 S4 就绪声明；
`primary`/`gate` 形式在这里只是**报告**，因为对照是已登记臂的 pinned 复用。

## 1. 结论摘要

- **单因素**：在**宽 129×129 输入**下把监督域从全网格收回 `interior_32`
  （= 冻结中心 65×65 盒 = 已登记窄臂自己的网格）。相对上一轮（宽输入 + 全 129×129 监督），
  唯一改动因子是**监督域**；相对已登记窄臂，唯一改动因子是**输入范围**。
- **读数：宽输入 + 内部监督在 seed 均值上**每个 lead 都优于已登记窄臂。

| lead | delta（宽内部 − 窄，K）seed 均值 | 范围 | 更优 seed | 本臂 `interior_32` t2m skill 均值 | 窄臂同域 skill 均值 |
| ---: | ---: | ---: | :---: | ---: | ---: |
| 6 h | **−0.0348** | −0.0459 … −0.0141 | **3/3** | +0.5944 | +0.5816 |
| 12 h | **−0.0401** | −0.0515 … −0.0259 | **3/3** | +0.3341 | +0.3151 |
| 24 h | **−0.0069** | −0.0204 … +0.0138 | 2/3 | +0.3700 | +0.3668 |
| 48 h | **−0.0528** | −0.1044 … +0.0111 | 2/3 | −0.0868 | −0.1191 |
| 72 h | **−0.0445** | −0.0948 … +0.0152 | 2/3 | −0.3996 | −0.4309 |

- **与上一轮全网格监督臂直接对照**（同一宽 store、同一配方、同一父、同一 seed）：本臂在
  五个 lead 的 seed 均值上**都更好**（上一轮 delta 全为正：+0.0013/+0.0147/+0.0189/+0.0121/+0.0231 K；
  本臂全为负）。skill 也全面更高（48h −0.0868 对 −0.1271、72h −0.3996 对 −0.4485）。
- **判定**：这支持"上一轮的 null **主要来自监督稀释**，而不是"域外上下文无用""。把监督域收回到与
  窄臂相同的盒后，宽输入带来一致的（但小的）正收益。**但它不等于**"冻结盒缺长 lead 所需信息"已被
  证实：收益幅度小（≤0.10 K），48/72h 的**绝对气候态门仍未过**——只有 seed 43 在 48h 首次为正
  （+0.0140），seed 41/42 仍为 −0.1581/−0.1163，72h 三 seed 全负。
- 因此**仍不进 S4 冻结包**：本轮的证据把上一轮的混淆**分离掉了**，并把该臂推到"值得在更强剂量下
  继续"的位置，但绝对门未过，封印配置没有依据。

## 2. 身份与协议

| 项 | 值 |
| --- | --- |
| 协议 SHA256 | `6fc50d049d484182f90ea51bf6e86d9b3cc146324ae183d360a7926c399c5fc7` |
| 归档 `code.zip` SHA256 | `2e60dfd60449eef4af2b7947b6ff37d189a27124328f56eacdcfabebd71b2351` |
| 归档 commit | `7ef43cbdc3a72d4942c018b6817384584b2bc3f3` |
| 执行文件身份 | 183 个（`execution_files_sha256`，逐文件核过归档） |
| 监督块（冻结） | `kind=spatial_mask`、`shape=[129,129]`、`selected_cells=4225`、`mask_sha256=4a2a576bf948b92924720fa28a44c128a61445dd29b4906805930430e0a0919a` |
| `model_code_sha256` | `d3fb58dbd0ed9efbd249fc258cb09c488d543c0a8ad77dac5689ac8f9ba77bab`（未变） |
| `training_code_sha256` | `8ad77d9349bdc4930e6bbba05f0c417814fe3093fe39aa921e816af197494974`（**变**：新增 masked 目标） |
| 源 SHA256 / train identity | `6bc9a1a2…` / `2360d42b…`（与上一轮同） |
| GPU | `GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b`（共驻，只读余量门） |
| test | `test.jsonl` 从不读取；`test_read=false` |

## 3. 目标层改动（本轮唯一代码改动）

- `training/r7_losses.py::latitude_weighted_mse(..., *, mask=None)`：`mask=None` 时**逐位不变**；
  给定 mask 时结果是"mask × cos(lat) 加权、只在被选格点上求平均"，分母是权重的和而非格点数。
- `training/r7_recursive_losses.py`、`training/r7_autoregressive_rollout.py::_draft_loss`、
  `training/r7_long_rollout.py::training_long_rollout(..., supervision_mask=None)` 逐层透传。
- `training/r7_long_rollout_runner.py`：`fine_tune_long_rollout(..., supervision_mask=None)`；
  mask 的**声明块**（kind/shape/selected_cells/mask_sha256/归一化方式）进入绑定 contract 的
  `autoregression.supervision`，因此进入 `signature`，也进入 `training_report.json`。
- **反证**（`tests/test_r7_s3_wide_interior_supervision.py`，8 测试全过）：单格 mask 上手工算出的
  受限加权均值必须被复现（全网格 1/3 对受限 1.0）；全 1 mask 必须**逐位等于**原目标；
  空 mask / 非 2-D mask / 负值 mask 必须报错；驱动对已存在输出与缺失 store 必须拒绝。

## 4. 训练与评分回执

| seed | checkpoint SHA256 | 训练 s | 五 lead 评分 s | reserved peak B | 训练损失 1→800 |
| ---: | --- | ---: | ---: | ---: | --- |
| 41 | `590b2cc70cb441b97c4a062b310434037207bce861fae5cd20950d83e89adb85` | 1605.0 | 885.8 | 8,977,907,712 | 1.6504→1.1299 |
| 42 | `b292865e029f9a7b3efc980f9253f7c38d8dd24f9602f7e449b6bd90dd7d94ba` | 1519.8 | 858.4 | 8,977,907,712 | 1.2072→0.8516 |
| 43 | `8377f63d8ab8093ff9dabed3d9184c791c57f9ca49e40b74fdda96d288ac6ab7` | 1595.0 | 879.7 | 8,977,907,712 | 1.4735→0.6006 |

- 三 seed 都跑到 `total_updates=800`、`selection_split=null`；每 seed 每 lead
  `n_evaluated == n_available_windows` 且等于冻结 cohort（472/468/460/444/428）。
- 每 seed 的 `training_report.json` 的 `supervision` 与冻结的 `mask_sha256` 逐字相同（驱动断言）。

## 5. 公平性守卫

与上一轮相同且全过：`WIDE_VAL_COHORTS` 逐 lead 与窄臂回执核对；宽区 train-only 气候态
`interior_32` t2m RMSE 与**已登记 v3-D2 在 ~1e-10 内相同**（两臂共用分母）；reading 阶段按
`rmse_csv_sha256` 复核窄臂（无漂移，窄臂**未重训练**）；父 checkpoint 与迁移回执逐位一致。

## 6. 成本

| 阶段 | whole 秒 | GPU-h | 说明 |
| --- | ---: | ---: | --- |
| prepare（CPU） | 436.9 | 0.1214 | 协议冻结与身份 |
| archive（CPU） | 434.2 | 0.1206 | 逐 `.py` 核归档 |
| seed 41 | 3765.2 | 1.0459 | |
| seed 42 | 3660.1 | 1.0167 | |
| seed 43 | 3747.5 | 1.0410 | |
| reading（CPU） | 432.1 | 0.1200 | 报告 only |
| **本轮合计（whole 12,475.851618 s）** | — | **3.4655** | soft overrun 3,475.85 s、hard overrun 0 |

- 口径同前：whole ÷ 3600（CPU 阶段也按 1 GPU 当量计入，偏保守）。网络 **0**（离线）；
  产物磁盘 **4.6 GiB**。累计 31.1244 → **34.5899**。
- 六阶段全 `success`、`returncode=0`、`signals=[]`、`reaped=true`；每阶段 spawn 前只读 `gpu_gate`；
  **未对任何非本实验进程发送信号**。

## 7. 已确认与推测

**已确认**：masked 目标在 `mask=None` 时逐位等于原目标（回归测试）；冻结 mask = 4225 点中心盒；
本臂 seed 均值在五个 lead 上都优于窄臂与上一轮全网格臂；每 seed 每 lead 的病例数与 pin 一致；
窄臂 pin 无漂移；六 worker 干净退出。

**推测（未证）**：收益的**来源**（域外上下文 vs 更少的监督噪声 vs 更有效的梯度预算）未被分离；
收益幅度小且 48/72h 绝对值仍为负，是否随剂量放大**未知**。

**未做**：不重训练窄臂、不读 test、不做显著性检验、不报泛化/SOTA/收敛因果。

## 8. limitations

- 单实例、单 ROI、17 通道、2017–2021 train / 2022 val 的开发筛选；三 seed 只是一致性证据，**无显著性**。
- 窄对照是 **pinned 复用**（index `record:s3-rollout-dose`），不是本协议内同批重训练。
- 只监督 `interior_32` 意味着外圈在训练中**无约束**；外圈误差未作科学判读。
- 父状态在窄域、全网格监督下训练，本臂只改监督域与输入范围，**优化景观**差异未分离。
- 绝对气候态门 48/72h **仍未过**（只有 seed 43 的 48h 为正）。
- 对照臂之间**宿主负载不同**（共驻），未做负载配平。

## 9. 下一动作

**仍不进 S4 冻结包。** 本臂是目前最好的开发臂，下一步是**在同一冻结监督域下提高剂量**
（800 → 2400 updates，父仍为迁移的 v3-BD 1600），检验收益是否随剂量放大到 48/72h 绝对门；
剂量是该族唯一已登记有正响应的杠杆（rollout-dose 200→800）。若提高剂量后 48/72h 三 seed 仍不过
绝对门，按防空转纪律**停该支**，转向别的可证伪假设（例如把 lead 条件化/物理权重重新分配），
不再加 seed 或重复同一 test。

test 未读、r=0、S4 未启动不变。
