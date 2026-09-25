# 0005 发布契约新增「时间段切分」模式（单连续段工程数据的 split）

- **日期**：2026-09-26
- **状态**：accepted

## Context（背景）

R7 的本地数据集发布契约（`data/preprocess/r7_era5_zarr.py` + `contracts.chronological_splits` +
`r7_store.validate_store`）把窗口按**日历年**划入 train/val/test，且要求三个 split
都非空、年份严格递增、`normalization_years` 恰好等于声明的 train 年份集合。
这套契约在 #13、#6 等跨三年真实数据上工作正常。

#63 冻结的 D1 段是**单一年份（2016）内连续 30 天**的工程段。任何年份式声明都无法
让它产生三个非空 split：val/test 年份（2017+）在数据里不存在，builder 会按
「every split needs at least one complete exact-time window」拒绝。也就是说，
冻结的 D1 范围与现行发布契约存在真实的接口间隙——要么放弃把 D1 变成可训练
store（#64 B0 就没有真实训练窗口），要么扩展契约。

## Decision（决定）

在 `build_r7_era5_zarr_from_dataset`（及 `r7_preflight` 的 preflight/写路径）新增
**可选**参数 `split_time_ranges`：以显式的半开 ``[start, stop)`` UTC 时间区间
（而不是日历年）给 train/val/test 指派窗口。约束全部 fail-closed：

1. 键必须恰为 train/val/test；区间两两不重叠；三个 split 按时间严格先后
   （train < val < test），镜像 `chronological_splits` 的年序语义；
2. 所有区间必须落在**声明的 `split_years['train']` 年份以内**——时间段切分只是
   对冻结 train 年份的工程性再划分，绝不触碰声明的 val/test 年份数据；
3. train 区间实际覆盖的年份集合必须**恰好等于**声明的 train 年份集合——保证
   `validate_store` 的 `normalization_years == splits['train']` 校验语义不变；
4. 窗口（history 帧 + init + target）**必须整体落在同一个区间内**——
   「窗口不得跨越 split 边界」的防泄漏守卫以时间形式保留；
5. store 元数据记录 `split_mode='time_ranges'` 与逐区间 ISO 时间戳；默认
   年份模式的行为与元数据**完全不变**（现有 store 与测试不受影响）。

配套决定（D1 特有）：D1 转换声明 `split_years = {train: [2016], val: [2017],
test: [2018]}`。val/test 年份是**读者契约的声明**（让 `validate_store` 的年序与
归一化年份校验继续成立），数据与归一化统计实际只来自 2016 的三个时间段
（train 24 天 / val 3 天 / test 3 天）；`zarr_metadata.json` 同时携带年份声明与
精确时间段，诚实性由元数据自证，并在 limitations 中写明「该 val/test 是 2016
段内的时间段再划分，不是 v2 的 2019 验证年或 2021 test 候选」。

## Consequences（后果）

- **正面**：D1 可以经既有审计发布路径（`fresh_outputs` + `BUILD_COMPLETE.json` +
  train-only 归一化 + 窗口 manifest）变成可训练 store，#64 B0 有真实窗口可用；
  `earthmover_spatial_d1.py` 只需新增 chunk 寻址，不需要另建发布系统。
- **代价 / 风险**：发布契约出现第二种 split 语义。处置：默认值 `None` =
  原年份模式，路径完全不变；时间段模式有独立校验函数与专项测试
  （`tests/test_r7_time_range_splits.py`）；元数据显式记录模式与区间。
  科学数据集（跨年、有真实 val/test 年份）**不应**使用本模式。
- **明确不做**：不放宽 `chronological_splits` 本身；不改动 `validate_store`；
  不为 D1 免除 `BUILD_COMPLETE`/有限性/单位守卫中的任何一项。
