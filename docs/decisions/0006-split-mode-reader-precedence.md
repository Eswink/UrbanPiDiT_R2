# 0006 时间段模式下读者以 `split_time_ranges` 为准，`split_years` 保留为声明

- **日期**：2026-09-26
- **状态**：accepted

## Context（背景）

决策 0005 给发布契约加了可选的 `split_mode='time_ranges'` 与 `split_time_ranges`，让
D1 这样的**单年连续工程段**能产生三个非空的 split。但那次只改了**写入方**：所有
**读取方**仍按 `split_years` 判窗口归属。D1 store 因此陷入自相矛盾的工作状态——

- 它声明 `split_years={'train':[2016],'val':[2017],'test':[2018]}`（年式读者契约需要
  三个递增年份），
- 同时声明 `split_mode='time_ranges'`、train=[01-01,01-25) / val=[01-25,01-28) /
  test=[01-28,02-01)，且**全部 120 个时次都在 2016 年**。

于是年筛在 D1 上退化为「全取」或「全弃」：

| 读取方 | 症状（实测，修前） |
| --- | --- |
| `r7_store.validate_record` | val **0/10**、test **0/10** 抛 `forecast window crosses split years`；train 94/94 通过 |
| `r7_evaluation.ZarrRolloutDataset` | val/test 抛 `no complete held-out rollout windows`（窗口数 0，取不到任何 lead） |
| `r7_evaluation.fit_training_climatology` | 取 **120/120** 时次（应为 96），把 24 个 val/test 时次当训练数据，t2m 基线偏乐观约 +6.9% |

`validate_store` 通过、store 合法：缺陷在读者，不在数据。`scripts/verify_r7_d1_store.py`
也不调 `validate_record`（它自己按 `split_time_ranges` 重算了一遍），所以发布验证全绿
**不能**覆盖这条缝隙。

必须定下的一件事：一个 store 同时给出两份互相矛盾的 split 声明时，**谁说话**。

## Decision（决定）

**在 `split_mode == 'time_ranges'` 的 store 上，窗口归属与 train-only 选取一律以
`split_time_ranges` 为准；`split_years` 在归属判定上不参与。**

1. 归属判定收敛到**单一入口** `data/r7_store.py:split_time_labels(root)`：
   年模式返回 `None`（读者走原年逻辑），时间段模式返回逐时次的
   `'train'/'val'/'test'`/`''` 标签数组。**三处读者**
   （`validate_record`、`ZarrRolloutDataset`、`fit_training_climatology`）全部经它，
   不得各自重算。
2. 区间解析函数上移到 `data/preprocess/contracts.py:parse_split_time_ranges`，
   写入方与读取方**共用同一实现**（`r7_era5_zarr` 改为再导出），杜绝写读语义漂移。
3. **`split_years` 不删**。旧产物与 `validate_store` 仍引用它；它保留两项意义：
   归一化年份校验的对象，以及「时间段只是对声明 train 年份的工程性再划分」这一
   已在写入方强制的不变量（区间边界必须落在声明 train 年份内）。
4. 时间段模式下 `split_years['val']/['test']` 是**占位声明**，不是数据来源——
   这一点写进 `split_time_labels` 的 docstring 与本文，避免后来者按年复算。
5. 声明坏掉时**失败关闭**：`split_mode='time_ranges'` 但 `split_time_ranges`
   缺失或不可解析 → 直接抛错，不静默回退到年模式（静默回退正是本缺陷的形态）。
6. **年模式逐位不变**：`split_time_labels` 返回 `None` 时，三条读者路径的代码与
   修改前等价。已用 HEAD 版模块对三个真实跨年 store 做对照实测（见下）。

## Consequences（后果）

**变容易的：**

- D1 从「val/test 读不出来」变成可读：`validate_record` val/test **0/10 → 10/10**，
  `ZarrRolloutDataset` 在 lead=6/12 上取到 val/test 各 10 个窗口、lead≤48 各 3 个；
  `fit_training_climatology` 由 120 个时次收敛到声明 train 区间的 **96 个**（24/桶）。
- 归属规则只有一处实现，新增读者不会再各自漂移；发布验证改用同一实现，
  同类缝隙被结构性堵住。

**变难 / 代价（如实列出）：**

- 契约里并存两种 split 语义，读代码的人必须知道「先看 `split_mode`」。
  缓解：单一入口 + docstring 写明优先级 + `tests/test_r7_time_range_readers.py` 固化。
- 时间段模式下的 val/test **不是**独立年份：D1 的 val/test 是 2016 段内的时间段
  再划分，不是 v2 的 2019 验证年 / 2021 test 候选。任何引用 D1 val/test 的结论
  都必须写明这一点（B1 报告已照此声明）。
- **实测边界**：本修复只覆盖被声明过的区间语义。D1 的三个范围都落在 2016 内，
  所以「train 窗口的年份集合恰好等于声明 train 年份」仍成立；若将来有 store 用
  时间段跨越声明之外的年份，写入方的 `parse_split_time_ranges` + 年份一致性守卫
  会先拒绝（决策 0005 第 2/3 条），不依赖读者。

**备选方案与否决理由：**

- **删掉矛盾的 `split_years`、只留时间段**：会让 `validate_store` 的
  `normalization_years == splits['train']` 校验与旧产物引用失效，等于用破坏
  兼容换取表面干净。否决。
- **改写 D1 store 的 `split_years` 为 2016/2016/2016**：`chronological_splits`
  要求 train < val < test，直接违反年序契约；且需重建 store（触碰冻结产物）。否决。
- **在读者里各修一处（三份年筛各加分支）**：三处逻辑重复，正是本次能同时漂移的
  原因。否决，改为单一入口。

## 回归证据（年模式逐位不变）

用 `git worktree` 把 HEAD 版 `data/` 模块导出到临时目录，对三个**真实跨年 store**
（`r7_coreasoning`、`r7_coreasoning_v2`、`r7_regional_real`）跑同一探针，逐位比较：

- `validate_record` 各 split 通过数：38/38/38、38/38/38、16/16/16 — 相同
- `fit_training_climatology` 的 `n_selected_steps`、逐桶计数、`training_years`、
  **means 的 fp64 逐通道 SHA256**（`db7046a975a7f0f1`）— 相同
- `ZarrRolloutDataset` 在 lead=(6,)、(6,12)、(6,12,24,48,72) × val/test 的窗口数
  （38/37/27）— 相同

整体 JSON 输出 `IDENTICAL: True` 且两文件 `cmp` 逐位相同
（`outputs/r7_b1_readiness/year_mode_before.json` / `year_mode_after.json`）。
