# S3 宽区域单因素：放宽输入区域**没有**带来长 lead 收益（null 读数）

**状态：已完成（2026-10-09/10）；`scientific_claim: false`；2023 test 未读；r 不消耗。**
本页对应 `docs/goals/main-model-climatology-campaign.md` §8 的 S3 接续与
`docs/goals/s3-wide-single-factor.md`。它是**开发筛选读数**，不是科学确认，也不是 S4 就绪声明。
它**不**宣布任何科学通过：`primary`/`gate` 形式在这里只是**报告**，因为本轮的对照是已登记臂的
pinned 复用、不是同协议内重训练。

## 1. 结论摘要

- 单因素：唯一改动因子是**输入区域**（65×65 → 129×129）。配方逐字沿用注册的 rollout-dose
  （`long_rollout`、12 步物理权重、800 updates、LR 2e-5、warmup 10、FP32、K4、batch 1、clip 1、
  checkpoint_every 20、seed 41/42/43），父状态为同一份迁移的 v3-BD 1600。
- 评分域是冻结中心盒：`boundary_masks(129,129,(32,))['interior_32']` = 4225 点 = 中心 65×65，
  与窄臂的 `full` 域**逐点相同**，因此配对 delta 是同一批病例上的同类量。
- **读数：宽臂在五个 lead 上都没有一致收益，seed 均值全部为正（= 更差）。**

| lead | 宽 − 窄（K，seed 均值） | 范围 | 宽更优的 seed | 宽 `interior_32` t2m skill（均值） | 窄臂同域 skill（均值） |
| ---: | ---: | ---: | :---: | ---: | ---: |
| 6 h | **+0.0013** | −0.0132 … +0.0192 | 2/3 | +0.5812 | +0.5816 |
| 12 h | **+0.0147** | −0.0101 … +0.0317 | 1/3 | +0.3081 | +0.3151 |
| 24 h | **+0.0189** | +0.0012 … +0.0340 | 0/3 | +0.3581 | +0.3668 |
| 48 h | **+0.0121** | −0.0124 … +0.0585 | 2/3 | −0.1271 | −0.1191 |
| 72 h | **+0.0231** | −0.0650 … +0.1428 | 2/3 | −0.4485 | −0.4309 |

- **48/72 h 的绝对气候态门仍未过**，且宽臂比窄臂**略差**（−0.1271 vs −0.1191、−0.4485 vs
  −0.4309）。72 h 的 seed 均值 delta 被 seed 41 的 +0.1428 拉正，seed 42/43 各为 −0.0650/−0.0083。
- **对假设的判定**：这个读数**不支持**"冻结 16° 盒缺长 lead 所需信息"——若域外大尺度上下文是
  长 lead 的瓶颈，扩大输入区域应当带来可辨的收益，而实测是持平到略差。
  它**也不构成**对该假设的反证：本轮是**全 129×129 全监督**，监督面积比窄臂大 3.94×，
  信息增益与监督稀释是**混淆的**。要区分二者必须另做"同监督域、只改输入范围"的对照臂（见 §10）。
- 因此**不进入 S4 冻结包**：本轮的证据只支持"把稀释与信息增益分开"的下一步，不支持封印配置。

## 2. 身份与协议

| 项 | 值 |
| --- | --- |
| 协议 SHA256 | `d016eb7a152efa60cae7bfb42ca877456a5204fe205d921c710bad42fe31d324` |
| 代码归档 `code.zip` SHA256 | `fa5752f60b00817614299b04ea5b0edea5456337b6b0a83df3068f9ad64337ab` |
| 归档 commit（ZIP comment 与 `code_commit.txt`） | `13f3c440557b4e0d2d7ef5bcdb1be1985a8aeb5f` |
| 执行文件身份 | 182 个（`execution_files_sha256`，逐文件 SHA256 核过归档） |
| 宽 store | `outputs/r7_s3_wide_instance_v4_20261009_attempt01/store/cache.zarr` |
| 源 SHA256 | `6bc9a1a2d93d345ce0dcb39d4f3a26f175ea3bcc15082ac4931a179ca13b7701` |
| train / val data identity | `2360d42b…` / `f590883e…` |
| `model_code_sha256` | `d3fb58dbd0ed9efbd249fc258cb09c488d543c0a8ad77dac5689ac8f9ba77bab` |
| `training_code_sha256` | `6e4363d5707ea4935d5449c800e4fecefa99bf553b42a747dd4dad37f8d548db` |
| GPU | `GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b`（共驻，只读余量门） |
| test | `test.jsonl` 从不读取；`test_read=false` |

宽 store 是 v3 的**真 drop-in**：split 与窗口计数（train 2360/val 472）与 v3 相同，3360 stamps，
17 通道，中心 65×65 与 v3 `source.nc` 逐位相同（见 `docs/R7_S3_WIDE_REGION_FULL.md`）。

## 3. 配对比较的公平性守卫（都通过）

- **同一病例集**：`WIDE_VAL_COHORTS = {6:472, 12:468, 24:460, 48:444, 72:428}` 在 freeze 时与
  窄臂回执**逐 lead 核对**，不等即失败；实测每 seed 每 lead 的 `n_evaluated == n_available_windows`
  且等于该 pin（例：48 h 三 seed 均 444/444）。
- **同一气候态分母**：宽区 train-only 气候态（2400 步，2017–2021 拟合）的 `interior_32` t2m RMSE
  与**已登记 v3-D2** 在 ~1e-10 内相同，因此窄臂的 skill 与宽臂的 skill 用同一个分母，可直接比。
- **窄臂 pin 复核**：`reading` 阶段按 index record `s3-rollout-dose` 的 `rmse_csv_sha256` 重新核对
  每个 seed 每个 lead 的 `rmse.csv`，漂移即 `RuntimeError`（本轮全部匹配）。窄臂**不在本协议内
  重训练**。
- **父状态**：三 seed 的 `parent_checkpoint_sha256` 分别为 `fba27e26…`/`5d4b05ed…`/`becc3d78…`，
  与迁移回执逐位一致；`parent_optimizer_imported=false`。

## 4. 训练与评分回执

| seed | checkpoint SHA256 | 训练 s | 五 lead 评分 s | reserved peak B | signature |
| ---: | --- | ---: | ---: | ---: | --- |
| 41 | `0174f1328876d5b62167d7ad738a3b4e6ec7cb9845be22cdb7cafc402bc1e42e` | 1618.1 | 889.9 | 8,973,713,408 | `d7ee8e66…` |
| 42 | `0082b41e23a268e8a3f9df2bee11ce51107024ae4eee47c280586460deabfc3c` | 1616.8 | 940.7 | 8,973,713,408 | `387dbcc5…` |
| 43 | `b1bee658ac12035c056a77c964819ee8b919f95ab9329ca01d4e8e414a39b39f` | 1582.4 | 863.5 | 8,973,713,408 | `53843261…` |

- 三 seed 都跑到 `total_updates = 800`（`resumed_from_updates = 0`，`selection_split = null`，
  `selection_metric = "frozen endpoint; no validation selection"`），训练损失 1.7700→1.0091 /
  1.4168→0.7750 / 1.5550→0.8734。
- 每 seed 五个 lead 的 `boundary_rmse.csv` SHA256 均记入 `seed*_receipt.json`（逐 lead 可核）。
- 逐 lead 评分秒数（seed 41）：6 h 93.4、12 h 117.2、24 h 146.6、48 h 230.6、72 h 302.1。

## 5. 共驻门与进程卫生

- 每个阶段 spawn 前写一份 `*_spawn_gate.json`（`source: "nvidia-smi read-only"`）。例：
  seed 41 门读到 free 25,241,321,472 B / total 25,769,803,776 B、required 4,294,967,296 B、
  `passed=true`。**未对任何非本实验进程发送信号**，未做冻结/终止自动化。
- 六个 worker（prepare/archive/seed41/seed42/seed43/reading）全部 `success`、`returncode=0`、
  `signals=[]`、`reaped=true`。邻居进程（GPU1）全程未被触碰。

## 6. 成本

| 阶段 | whole 秒 | GPU-h | 说明 |
| --- | ---: | ---: | --- |
| prepare（CPU，协议冻结与身份） | 426.6 | 0.1185 | 只读 store/manifest 与 pin |
| archive（CPU，code.zip 复核） | 427.1 | 0.1187 | 逐 `.py` 核归档字节 |
| seed 41（训练 + 五 lead 评分） | 3763.0 | 1.0453 | |
| seed 42（训练 + 五 lead 评分） | 3849.7 | 1.0694 | |
| seed 43（训练 + 五 lead 评分） | 3816.8 | 1.0602 | |
| reading（CPU，配对读数与 pin 复核） | 439.0 | 0.1220 | 报告only，无通过声明 |
| **本轮合计（whole 12,721.940569 s）** | — | **3.5339** | soft overrun 3,721.94 s、hard overrun 0 |

- 口径沿用本方向既有约定：**whole attempt 墙钟 ÷ 3600**（与 probe/screen 两轮一致），
  因此 CPU-only 的 prepare/archive/reading 也按 1 GPU 当量计入，偏保守。
- 网络 **0**（离线 `deny_network()`）；无下载、无数据发布、无 clone。产物磁盘 **4.6 GiB**（三 seed 各 1.5 GiB）。
- 累计：27.5905 → **31.1244** GPU-h（cap20/remaining 仅会计，不是总许可上限）。

## 7. 已确认与推测

**已确认（有逐位/逐文件证据）**：协议与代码归档身份；宽 store 与 v3 的 split/窗口/中心块身份；
宽区气候态分母与 v3-D2 同值；每 seed 每 lead 的病例数 = 登记 cohort；窄臂 `rmse.csv` 未漂移；
六个 worker 干净退出；共驻门记录；上表的配对 delta 与 skill 由 `readings.json` 直接给出。

**推测（未证）**：宽臂长 lead 略差**可能**来自全 129×129 监督的稀释（监督面积 3.94×），
也可能来自父状态是在窄域上训练的、迁到宽域后需要更多更新才适应；**两者都未被本轮分离**。

**未做**：不重训练窄臂、不读 test、不做显著性检验、不报泛化/SOTA/收敛/遗忘因果。

## 8. limitations

- 单实例、单 ROI、17 通道、2017–2021 train / 2022 val 的开发筛选；**无显著性**，三 seed 只是一致性证据。
- 窄对照是**pinned 复用**（index `record:s3-rollout-dose`），不是本协议内同批重训练；其
  `rmse.csv` 已按 SHA pin 复核，但两臂的**环境**（共驻邻居负载）不完全相同。
- 全 129×129 监督使**信息增益与监督稀释混淆**，这是本轮读数的核心歧义。
- 父状态在窄域训练、在宽域复用；模型 spec 与区域无关，但**优化景观不同**。
- 评分只在 `interior_32`；外圈 129×129 的误差未作科学判读。
- 绝对气候态门在 48/72 h 仍未过（两臂都是负 skill）。

## 9. 下一动作

**不做 S4 冻结包。** 下一条独立、可证伪的实验是**分离稀释与信息增益**的单因素对照：

- **臂**：宽 129×129 输入 + **只在 `interior_32` 上监督**（与窄臂同监督域）；对照是**已登记**的
  窄臂（65×65 输入、`full`=65×65 监督）。这样两臂的监督域逐点相同，唯一差别回到输入范围。
- **第一条命令**（先 CPU，写新 protocol/输出/停止出口与定向反证，再申请 GPU）：
  `.venv/bin/python -m pytest -q tests/test_r7_s3_wide_single_factor.py`，
  然后在 `scripts/study_r7_s3_wide_single_factor.py` 的 `_seed` 里加一个 `--supervision interior`
  变体（新输出目录、新协议 digest、新 `code.zip`），复用本轮的 pin/cohort/门与评分路径。

test 未读、r=0、S4 未启动不变。
