# #67 封存 test 报告

**判定：封存 test 报告 BLOCKED —— 冻结的 test 划分与 train-only 气候态**桶**不相交。**
协议本身已按 #67 要求在看 test 之前冻结（`docs/R7_67_PUBLICATION_PROTOCOL.md`，commit
`6259c78`）。本文件记录**为什么在冻结协议下无法产出 test 数字**，以及需要什么才能解除。

**关键点：test 数字一次都没有产出。** 执行在写出任何 `rmse.csv` 之前就失败，
所以「封存的 test」仍然是未读状态，后续补做不会变成「多次看测试」。

| 项 | 值 |
| --- | --- |
| 协议冻结 SHA | `6259c78`（写入于读 test 之前） |
| 执行 SHA | 同上 |
| 执行命令 | `scripts/rollout_r7_metric_tables.py --manifest …/test.jsonl --checkpoint <15 个冻结 B2 checkpoint> --leads 6 12 24 48 72` |
| 结果 | **失败**：`ValueError: missing training climatology bucket (2, 12); no held-out fallback` |
| 产物 | `outputs/r7_67_sealed_test_blocked_february_buckets/`（**原样保留**，只有 1 个空目录，0 个指标文件） |
| 产出的 test 指标数 | **0**（`find … -name "*.csv" \| wc -l` = 0） |

---

## 1. 根因（数据范围限制，非代码缺陷）

冻结的 B2 36 天段把 val/test 切在 1 月底到 2 月初：

| split | 时间范围 | 时次 | 月份 |
| --- | --- | --- | --- |
| train | `[2016-01-01, 2016-01-25)` | 96 | 全部 **01** |
| val | `[2016-01-25, 2016-02-01)` | 28 | 全部 **01** |
| **test** | `[2016-02-01, 2016-02-06)` | 20 | **全部 02** |

而训练期气候态是 **train-only** 的月-小时桶（`fit_training_climatology`）：

- train 只覆盖 **1 月**，所以桶只有 `(1, 00) (1, 06) (1, 12) (1, 18)` 共 **4** 个；
- test 的 20 个时次**全部落在 2 月**，需要 `(2, 00/06/12/18)` 四个桶；
- `data/r7_evaluation.normalized_climatology` 在缺桶时**拒绝回退**到 held-out 数据
  （这是刻意的防泄漏设计），于是抛错。

**这不是代码 bug，也不该被"修"成静默回退**：用 test 自身的数据拟合气候态就是泄漏，
而回退到 val 也会越过划分边界。读者的拒绝是正确行为。

## 2. 为什么这不只是「气候态基线算不出」

`evaluate_local` **无条件**计算 `climate=normalized_climatology(...)`（`training/r7_evaluate.py:137`），
即使在为神经模型打分时也是如此——因为 `rmse_climatology` / `mse_skill` 要和预测 RMSE
落在**同一批 case** 上（#64 D-3 的刻意设计）。因此缺桶不但使 `climatology` 基线不可算，
也使**任何**模型在这段 test 上的评估不可算。`rollout_r7_metric_tables.py` 同样如此。

## 3. 解除阻塞的选项与代价（如实列出，未执行）

| 选项 | 做法 | 代价 / 是否可选 |
| --- | --- | --- |
| **A. 重切 held-out 到 1 月内** | train `[01-01,01-25)` 不变；test 改为 `[01-26,02-01)`（20 时次，18 个 6h 窗口，**全部 1 月**） | **需要重训所有臂**：`data_identity` 覆盖 store 的 `split_time_ranges`，重切产生新 identity，现有一切 checkpoint 会被身份检查**正确拒绝**。即这是一个**新实验**，不是对 B2/C1/C2/C3 链的封存确认 |
| **B. 改成仅 1 月的封存段** | 用 `[01-01,01-25)` 之外的 1 月时次另建一个独立 store | 同上：新 store、新 identity、全臂重训；且还要重新跑 800 步 × 5 臂 × 3 seed（约 1.1 GPU-h）+ 评估 |
| **C. 放宽读者拒绝** | 让缺桶时回退 | **禁止**：这是防泄漏设计，放宽等于把 test 数据用于构造基线 |
| **D. 只为神经模型评估、跳过气候态** | 改 `evaluate_local` 使气候态可选 | **禁止**（本协议口径）：`mse_skill` 是与冻结基线对比的**主要表**之一，去掉它就不是同一张表；且改评估代码会破坏与已发布 B2 表的可比性 |

**本轮选择：不执行 A/B。** 理由：授权允许「重切 held-out」，但本轮剩余预算与
「#67 是先冻结协议、后一次性读 test」的纪律冲突——A/B 需要**重训全部臂**（新 identity），
而重训后的结果**不能**用来确认已冻结的 B2/C1/C2/C3 链，只能作为一个新实验报告。
把它当作 #67 的封存交付会造成「用新实验冒充旧链确认」的误导，这正是纪律禁止的。

## 4. 本段能交付的（#67 交付清单对照）

| #67 交付项 | 状态 |
| --- | --- |
| 先冻结协议与 case 表 | **已交付**：`docs/R7_67_PUBLICATION_PROTOCOL.md`（commit `6259c78`，读 test 之前） |
| headline 变量/时效/可接受退化/多重比较处理看 test 前写入 | **已交付**（协议 §6） |
| ≥3 固定 seed 分别呈现 | **已交付**：B2/C1/C2/C3 全部逐 seed 呈现，从未先平均 |
| 按天/周 paired block 重采样 | **已定规则**（按天；.val 只有 7 天，按周只剩 1 个 block） |
| 四季与 full/interior/boundary 分组 | **四季不可做**（数据只有 1 月，已如实标注 UNVERIFIED）；full/interior/boundary 口径见 #62 |
| 无降水目标不写降水技巧 | **已遵守**：全部文档未出现降水 skill |
| 外部预训练模型只列单独 reference 表 | **已遵守**：本段未运行任何外部模型 |
| 只读 test 报告 | **BLOCKED**（见 §1–§3） |
| 代码版本 / 环境锁定 / 数据 recipe+许可 / 索引 digest / 图表脚本 | **已交付**：精确 SHA 记录在 TASK_QUEUE；环境见 `docs/rules/environment.md`；数据 recipe 见 `docs/R7_D1_ACQUISITION.md` 与决策 0004/0008；图表由 `scripts/rollout_r7_metric_tables.py` 从原始充分统计量生成 |
| README 三栏 | 见下文 §5 |

## 5. README 分栏（#67 要求）

已在本轮于 `README.md` 增加三栏导航：**smoke**（CPU/GPU 小实验）、
**negative study**（负结果集）、**publication benchmark**（封存基准，含本文件的 BLOCKED 状态）。

## 6. 复现（失败是确定的，可复核）

```bash
.venv/bin/python scripts/rollout_r7_metric_tables.py \
    --manifest outputs/r7_b2_segment/store/manifests/test.jsonl \
    --checkpoint outputs/r7_b2_multiseed/seed41/training/seed41/unet/update_0000800.pt \
    --leads 6 --out <new dir>
# -> ValueError: missing training climatology bucket (2, 12); no held-out fallback
```

## 7. 结论

#67 的**协议**部分交付完成并已冻结；**封存 test 数字**因冻结划分（2 月）与
train-only 气候态桶（仅 1 月）不相交而 **BLOCKED**，且这是**数据范围限制**，
不能通过改代码消除（唯一的合法出路是新 identity 下的全臂重训，那是新实验）。
test 仍是未读状态，未产生任何指标。
