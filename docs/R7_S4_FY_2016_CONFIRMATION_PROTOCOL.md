# S4 首次确认协议：FY candidate 在完整未见年份 2016 上的评分（Track 0）

> **状态：协议已冻结、评分尚未运行（2026-10-26）。** `scientific_claim: false`。
> 这是 `docs/goals/s3-matched-exposure-reeval.md` §4.1（Track 0）的 D2 交付物。
> 实例机制按 **决策 0059** 构造（不是 §4.1 的字面机制，原因见该决策）。

## 1. 实例（决策 0059）

- **目标年 2016**：多年 store `outputs/r7_s3_fy_multiyear_20261023_attempt01/store/cache.zarr`
  里 2016 完整在位（**1,464 帧**，`2016-01-01 00:00 → 2016-12-31 18:00`，6 小时步长，单调唯一）。
- **不新建 store**：目标帧直接取自该多年 store；该 store 的 `state` 形状（`[34084,17,65,65]`）、
  `time_ns`、`latitude`/`longitude` 与 attrs 摘要共同钉住帧集合。
  `source.nc` sha256 `1db839554cce30cd14c5730633e166a6e370b3dd8c954fc6572ebdba853b32f5`。
- **可用初值（实测，本仓窗口规则）**：6/12/24/48/72 h = **1,462 / 1,461 / 1,459 / 1,455 / 1,451**。
  （目标长文 §1 记的 1,461/1,460/1,458/1,454/1,450 每个 lead 少 1；以实测为准。）
- **归一化 = FY candidate 的 2017–2021**：从 FY store 读并**显式**施加，不依赖被评 store 的声明。
  `normalization_mean` sha256 `e15c3f2612185bb9179844e4d45d0525879abb68132fc62d5918701b445eea36`、
  `normalization_std` sha256 `afc998a31fdd96af3d48a6bd1eb5af4081de11a0728c924fdfcc41c961fb0783`。
- **气候态分母 = FY 的 48 桶 train-only（2017–2021）**：`reference.json` sha256
  `4c89c70937cd848682fdfefd216fd77e1d648cfb0292c416e87e2d5d3aeb2135`。
  该文件只记 provenance 与逐 lead 的 `rmse_climatology`，**不含 48 桶均值表** ⇒ 驱动从 FY store 的
  声明 train 年份**重拟合**该表，并在使用前逐字段核对 `kind` / `training_years` /
  `n_selected_steps`（实测 7,304）。
- **被评 checkpoint**（FY 轮 `attempt03` 的三份已冻结候选 + 三份同数据 incumbent）：

  | 臂 | seed | checkpoint sha256 | `data_identity` |
  | --- | --- | --- | --- |
  | treatment | 41 | `c5a6c4ec328154db…` | `c17494ab63d54855…` |
  | treatment | 42 | `2a49a07e255926eb…` | `c17494ab63d54855…` |
  | treatment | 43 | `2c3b78f8e1305ca4…` | `c17494ab63d54855…` |
  | control | 41 | `53c0a4997000faaa…` | `04693129781a234e…` |
  | control | 42 | `9ff43fb20293e2a6…` | `04693129781a234e…` |
  | control | 43 | `29d00f46da82ef75…` | `04693129781a234e…` |

- **身份门照真校验**（不绕过）：对每个 checkpoint 断言
  `dataset_identity(FY <arm> train.jsonl) == 协议 data_identity == contract['data_identity']`，
  其中 `train.jsonl` 在 FY 目录里读（那里 `../cache.zarr` 就是 FY store）。
  treatment `train.jsonl` sha256 `b77d59fdcf0fb2b7…`、control `99226495e057240b…`。
- **唯一被跳过的校验子句**：`validate_record` 的「跨 split 年份」归属
  （`data/r7_store.py:179-183`）——多年 store 声明 train 2000–2021 / val 2022 / test 2023，
  不描述 2016 是 held-out；其余完整性子句（索引在界、时间戳逐位一致、`init_time`、lead、等间隔）
  全部保留。

## 2. r 与 alpha 登记（先登记后评分；§4.1 第 4 条）

- **r = 1**：本次是本实例的**首次真实 test 评分**，消耗 r=1，**不可逆**。
- **family-wise alpha_1 = 0.025**（同时 97.5%），与 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md:91` 一致。
- 后续确认按 `alpha_r = 0.05 / (r*(r+1))`。

## 3. D4 组成与判据（不新增阈值、不放宽任何已冻结判据）

- **primary**：t2m、region `full`、物理 K、`skill = 1 − mean MSE_model / mean MSE_climatology`
  （分组均值之比），lead `{6,12,24,48,72}`，分组 = 总集 / 预声明年份 2016 / DJF / MAM / JJA / SON。
  单年份实例下**总集与 2016 是同一集合**；两者都按目标长文 §8 的口径声明。
- **同时区间**：`training/r7_simultaneous_stats.simultaneous_paired_interval`，per-case 序列 =
  seed 平均后的 per-case skill；block 长 **14**（72 h 最长 rollout + 12 h 历史依赖，6 h 步长），
  resamples **4,000**、seed **20160101**，`alpha = 0.025`。
- **family 表达力（实测限制，已写进协议）**：该函数要求同一 family 的**所有 cell 共享同一组
  case key**，而四季组是对 2016 病例的**划分** ⇒ 「总集 + 各年 + 四季组同处一个 family」
  **不可表达**。因此 family 按**分组**实现（每组 5 个 lead），并在读数里逐字记录这一点。
- **关键变量退化守门**：u10/v10/mslp 对**同数据 incumbent**（FY control 臂）比较，同分组同 lead，
  per-case 相对 MSE 变化；要求每 seed ≤0 且同时区间上界 ≤0。
- **不足块**（`insufficient-evidence`）**不是通过**；undefined 格不得删格或换分母。
- **覆盖率门**：`docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md:111` 要求「至少覆盖一个完整未见年份
  和全部四季」；2016 是完整未见年份且四季齐全 ⇒ 该门**在本实例上可满足**（仍以读数为准）。

## 4. limitations（冻结，随读数一起报）

- **气候态分母期（2017–2021）与 test 目标期（2016）不同代**（ERA5 有趋势）⇒ 此处的 t2m skill
  **不是** FY 轮测的同一个量；u10/v10/mslp 那条是「对同数据 incumbent」的比较，不受此影响。
- 2016 帧取自多年 store，不是专门构建的 2016 store；`validate_record` 的年份归属子句是唯一被跳过的子句。
- `scientific_claim: false`；这是首次确认读数，不是发布级科学认证。

## 5. 产物与复现

- 冻结协议：`outputs/r7_s4_fy_2016_20261026_attempt01/protocol.json`，
  `protocol_sha256 = 0dd100dcb4436a848bcd1b4fa91fdd8709f2a99a03e686b750b765572be2db1c`。
- 驱动：`scripts/study_r7_s4_fy_2016_confirmation.py`（只读已冻结产物，只写自己的 `--out`）。
- 复现第一条命令：

  ```bash
  .venv/bin/python scripts/study_r7_s4_fy_2016_confirmation.py \
    --protocol outputs/r7_s4_fy_2016_20261026_attempt01/protocol.json \
    --out outputs/r7_s4_fy_2016_20261026_attempt01/reading01 --device cuda
  ```

## 6. 明确未做

- **评分尚未运行**：本文件冻结的是协议与 r/alpha 登记，**没有任何读数**。
- 没有跑 Track 1 的任何一轮；没有读 2023 test；没有新增下载；没有改 `model/**`。
- 不宣布科学通过、不称泛化/SOTA、不自行宣布最终 goal 完成。
