# G1-B 区域补跑：objective 字面 `interior_32` 上的出口① 读数

**状态：`supplement`（2026-11-26）；`scientific_claim: false`；`test_read: false`；`r = 0`；1.6017 GPU-h。**

本页是一次**补充测量**，不改任何冻结判据、不重训任何臂、不加 seed。
主轮读数见 `docs/R7_S3_VISIT_GAIN_RESULT.md`。

## 1. 为什么要补跑

objective 与决策 0118/0036 把出口①写成 **72 h `interior_32` t2m**，而本轮 `protocol.json`
冻结的是 `full(65×65)`（ADR 0119 的理由：65×65 上 32 格边距只剩 1 格，且宽实例的
`interior_32` 已被记录为 65×65 全网格）。那是一次**对 objective 字面文本的替换**，所以
「出口①为假」这个结论原本依赖一个修订过的口径。

补跑把这个依赖去掉：**用同一批已冻结 checkpoint，在 objective 写明的区域上再评一次 72 h。**
不训练、不加 seed、不扫增益族、不改 K。

## 2. 怎么做的

`scripts/r7_s3_visit_gain_region_supplement.py`：对六个已冻结 checkpoint 各跑一次
`evaluate_local`，`lead_hours=(72,)`、`boundary_margins=(1, 32)`，于是同一张
`boundary_rmse.csv` 里同时有 `full`（4,225 点）、`interior_1`（3,969）、
`interior_32`（**1** 点）与各自的 `edge_*`。

- 检查点身份：每份收据记录 `checkpoint_sha256` 并与主轮收据交叉核对（不一致即拒绝）。
- 数据身份：`val_manifest_sha256`、`train/val_data_identity` 原样继承冻结协议。
- 写入：新目录 `outputs/r7_s3_visit_gain_region_supplement_20261126_attempt01/`，独占创建；
  **不改** `readings.json`（一次写入、不覆盖），**不改** `protocol.json`。
- 六臂各评到 **1,447** 个初始化（与主轮 72 h 相同 case 集），各 934.0–990.4 s。

## 3. 读数：72 h t2m，A−B（负 = A 更好）

| seed | **`interior_32`（objective 字面）** | `full`（主轮冻结） | `interior_1` |
| ---: | ---: | ---: | ---: |
| 41 | **−0.002725** | +0.000323 | +0.000337 |
| 42 | **+0.004107** | −0.001023 | −0.000979 |
| 43 | **+0.000267** | −0.000801 | −0.000820 |
| 符号 | **`− + +`** | `+ − −` | `+ − −` |
| 三 seed 同号为负？ | **否** | 否 | 否 |
| 均值 | **+0.000550** | −0.000500 | −0.000488 |

## 4. 结论

1. **出口① 在 objective 的字面区域上同样为假**（seed42 +0.004107、seed43 +0.000267）。
   ⇒ 区域口径**不承重**：两种读法都关支。
2. **更强的一条**：符号与均值在两个区域间**系统反向**（`full` 上 A 略优、`interior_32` 上 A 略差；
   均值 −0.000500 → **+0.000550**）。所以「72 h 上 A 略优」这个方向**依赖区域选择、不稳健**，
   不应被引用。这把一个已关闭的支关得更实。
3. `interior_32` 在 65×65 上只有 **1 个格点**，该 RMSE 只有一个空间样本、噪声远大于 `full`。
   **两个口径都不支持任何效应声明**；能确定的只有「三 seed 同号为负」这个合取不成立。

## 5. 限制

- 单格点区域的 RMSE 统计意义很弱；本页只用它判**符号**，不用它定效应大小。
- 补跑的是已冻结 checkpoint，不是新臂，也不是对 2,400-update 剂量的复现。
- `r = 0`、`test_read: false`，未读 2023 test。
- 北界与主轮相同：单一 65×65 实例、三 seed 为一致性证据而非显著性检验。

## 6. 复现

```bash
.venv/bin/python scripts/r7_s3_visit_gain_region_supplement.py \
  --seed 41 --arm visit_damped \
  --out outputs/r7_s3_visit_gain_region_supplement_20261126_attempt01 --device cuda:0
```
