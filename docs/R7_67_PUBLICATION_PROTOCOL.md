# #67 期刊评估协议与封存测试（S5）

本文件是 #67 要求的**在看 test 之前写入的协议**。它冻结任务定义、主要表、
指标口径、不确定性处理、分组与多重比较处理，以及封存与交付清单。

**状态：协议已写入（冻结）；test 封存报告见配套文件。** 按 #67 与项目纪律：
`test` 在整个 S3/S4 阶段不得用于选型；本协议写入时**尚未读 test**。

---

## 1. 任务定义（冻结，不得事后扩大）

| 项 | 冻结值 |
| --- | --- |
| 区域 | 有限区域，起报时**不做全球上下文**；原生 0.25° 网格 |
| 变量 | store 声明的 **17 个通道**：`t2m, u10, v10, mslp, z850, t850, q850, u850, v850, z500, t500, q500, u500, v500, z250, u250, v250` |
| 输入 | `history = (t−6h, t)`，即 2 个历史时次，间隔 6h |
| 目标 | 未来 +6/12/24/48/72h（free rollout），6h 步长 |
| 数据来源 | ERA5 经 Earthmover，2016-01 的 36 天工程段，`data_identity = 50517d94…54cd6961`（B2 段） |
| 划分 | store 自带的 `split_time_ranges`（决策 0005）：train `[01-01, 01-25)`、val `[01-25, 02-01)`、**test `[02-01, 02-06)`** |
| **不是** | 全球 foundation SOTA；**不是** 1 km 动态真值；**不是**封存多年的标准测试集 |

**test 的性质必须如实标注**：`test` 是 2016 段内的**工程再划分**（18 个窗口，5 天），
**不是** v2 计划中的 2021 封存 test 年份，也**不是**跨季节或跨年测试。
任何「在 test 上表现如何」的陈述都只在这 5 天、这一个 1 月段内成立。

## 2. 主要表（冻结）

同数据、同初始化、同 rollout 口径的**主要表**：

| 族 | 成员 | 参数 |
| --- | --- | --- |
| 零参数控制 | `persistence`、`climatology`（**train-only** 月-小时气候态） | 0 |
| 神经基线 | `unet`、`native_window`（Swin-like）、`afno_small` | ~2.8M（B1 实测 1.483% 离散） |
| 递归族 | `generic`、`process` | ~2.8M |
| 条件项 | `adaptive` | **仅当 #66 门槛通过** |

**外部预训练模型只可列单独 reference 表**：GraphCast/Pangu 等若出现，必须
**单独成表**并披露其全球上下文、训练年份、变量集与初始化差异；
**不得**放进主表，也**不得**声称在同表内被公平超越。本段**没有**运行任何外部模型。

## 3. 口径（冻结，含已声明的定义陷阱）

1. **不做未来信息注入**：不注入未来 ERA5、未来过程标签、未披露的 NWP 边界。
   模型的输入白名单由 `model/r7_halting.forecast_inputs` 强制
   （只转发 `coarse_history` 与 `lead_time_hours`）。
2. **指标从原始充分统计量生成**：每变量 weighted RMSE、pooled ACC、
   相对 persistence 与 train-only climatology 的 MSE skill。
3. **ACC 定义必须随表给出**：本库 ACC 用的气候态与**未再中心化**的定义
   （见 `docs/R7_ROLLOUT_TABLES.md`）。**正 ACC ≠ 正 MSE skill**；
   负 ACC 的含义依赖同基线与同权重前提。
4. **禁止**平均定义不同的 ACC，**禁止**直接平均各 batch 的 RMSE。
5. 每格必须带**例数**；缺项的格子 fail closed，不得静默跳过。

## 4. 不确定性（冻结，写入于看 test 之前）

1. **至少 3 个固定训练 seed 分别呈现**（不是先平均再报）。本项目已声明 41/42/43。
2. **天气不确定性用按天/周的 paired block 重采样**，block 规则**在 validation 阶段选定**：
   本段采用**按天**分块（同一 UTC 日的所有起报为一个 block），因为 val 只有 7 天，
   按周只剩 1 个 block 无法重采样。重采样只用于区间估计，不用于选择模型。
3. **seed 不确定性与天气采样不确定性分开报告**，**不得**把 3 seed × 同一 case
   当成独立大 N。B2 已实测：6h 的 seed 离散 ~3%，72h 达 20–23%（个别格 65%）。
4. 时间相邻的起报**不是独立样本**；网格点**不是**独立重复。

## 5. 分组与覆盖（冻结）

- **四季**：本段**只有 2016 年 1 月**，因此**四季分组不可做**。
  任何跨季节结论一律 **UNVERIFIED**；这是数据范围的硬限制，不是方法选择。
- **起报时刻**：按 UTC 时次分组（val 覆盖 06/12/18/00Z）。
- **full / interior / boundary**：按 #62 已建立的分组口径给出（区域边界与内部）。
- **极端事件**：阈值**只来自 train**；本段可评温度/风。
- **降水**：**本 store 没有降水目标，因此不写任何降水技巧（skill）**。

## 6. Headline 变量、时效、可接受退化与多重比较（冻结）

| 项 | 冻结值 |
| --- | --- |
| **Headline 变量** | `t2m`、`mslp`、`v850`、`u10` |
| **Headline 时效** | `6h`（训练时效）与 `24h`（自由 rollout 中段） |
| **全变量附录** | 全部 **17** 变量 × 5 时效，**必须完整呈现**，不只展示胜出的子集 |
| **可接受退化** | headline 变量上相对**固定参照**的相对退化 **≤ 1%** 视为非劣 |
| **参照** | 默认参照为 `native_window`（B1 审计过的强非递归基线）；零参数控制另列 |
| **多重比较** | 17 变量 × 5 时效 × 若干臂会产生大量格子。处理方式：**不报单一聚合 p 值**；以「每 seed 同号」作为方向性判据（#60 比较器口径），并**同时**给出按天 block 重采样的区间；区间跨 0 或符号不一致一律记 **unresolved**，不得因单项 improved 而假通过 |
| **胜负计数** | 每个 (变量, 时效) 给出 improved / worsened / unresolved 三态计数 |

**写入时机声明**：以上 headline 与阈值在**读 test 之前**写入本文件；
写入后不得增删 required cell，也不得改用「ours 赢」的变量。

## 7. 计算与创新证据（冻结的记录项）

每个模型必须记录：参数量、counted-ops 的定义（本项目用 `FlopCounterMode` over one
forward pass under `enable_grad`，不计 elementwise/normalization）、训练 update 数、
样本数、GPU-hours、**真实单卡 latency/吞吐**、K 分布（如适用）、peak memory。
**full BPTT / retained-truncated / streamed 的语义不同**，必须在方法中写明
（本项目训练用 streamed truncated BPTT；B0/GPU bring-up 已记录其内存优势）。

## 8. 交付与封存清单（冻结）

1. **只读 test 报告**：冻结配置、签名与 checkpoint **之后**执行**一次**，
   结果只读保存并登记访问；
2. 代码版本（精确 SHA）；
3. 环境锁定（Python / torch / CUDA 版本，见 `docs/rules/environment.md`）；
4. 数据获取 recipe 与**许可**；
5. checkpoint / metrics 的 artifact 索引与 digest；
6. **图表生成脚本**（表格与图从原始充分统计量生成，不手工编辑数字）；
7. `README` 分栏：**小 CPU/GPU smoke** / **negative study** / **publication benchmark**。

**后续改方法需新版本并透明说明**，不得声称「继续未见测试」。

## 9. 本段能回答与不能回答的（如实边界）

`test` 只有 5 天、18 个窗口，因此本段能回答的是「在一个 1 月工程段的 5 天 held-out 上，
各臂相对零参数控制与非递归基线如何」。**不能**回答：跨季节、跨年、全球 SOTA、
1 km 真值、降水技巧、以及任何统计显著性的主张。三个 seed 不是显著性检验。

## 10. 复现

封存 test 报告由既有脚本驱动（#8 已交付，不新写第二套口径）：

```bash
# 1) 从冻结 checkpoint 生成 6/12/24/48/72h 的 RMSE/ACC/skill 表（不触碰训练代码，
#    并拒绝混用不同 model digest 或不同 data identity）
.venv/bin/python scripts/rollout_r7_metric_tables.py \
    --manifest outputs/r7_b2_segment/store/manifests/test.jsonl \
    --checkpoint <每个臂的 selected checkpoint> --label <每个臂的名字> \
    --leads 6 12 24 48 72 --out outputs/r7_67_test_tables
```

**执行前置条件（本协议声明，执行时逐条核对）**：

1. 所有臂的 checkpoint 的 `model_code_sha256` 必须等于当前 `model/` 树的 digest；
   不一致即拒绝（脚本已强制，fail closed）。
2. 所有臂必须共享同一个 `data_identity`。
3. **test 只执行一次**；执行后不得再改方法并复用同一份「未见测试」的表述。
4. 报告必须包含全部 17 变量 × 5 时效，以及 improved/worsened/unresolved 三态计数。

**当前状态**：见 `docs/R7_67_SEALED_TEST_REPORT.md`（或该文件不存在时的 TASK_QUEUE 条目）。
