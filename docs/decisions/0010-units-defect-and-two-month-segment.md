# 0010 —— 第二阶段：单位缺陷修正与双月（M2）段

- **日期**：2026-09-27
- **状态**：accepted
- **范围**：`docs/goals/full-auto-campaign.md` 授权的第二阶段（气泡数扩展）
- **代码 SHA 范围**：`0e09800` → 本轮提交
- **依据证据**：`docs/R7_67_SEALED_TEST_REPORT.md`、`docs/R7_69_BUCKET_EXPANSION.md`

## Context（背景）

`docs/goals/full-auto-campaign.md` §5 记下一条已知线索：**climatology 在 t2m 上占优
是数据范围的产物**（1 月只有 4 个 `(month,hour)` 桶），不是「基线很强」。第二阶段的核心
科学任务就是把这条线索变成可判定的实验：取跨月连续段使桶数从 4 增到 ≥8，再看
「climatology 压倒神经臂」是否仍成立。

## Decision（决定）

### 1. 先修单位缺陷，再解释「9.09 倍」

`training/r7_climatology_skill.py` 的 `RolloutClimatologySkillAccumulator` 没有
`training_std` 参数，其两个 RMSE 列因此是**归一化**（无量纲）值；而
`RolloutRMSEAccumulator` 会用训练集 std 还原物理单位。`training/r7_evaluate.py` 给两份
CSV 盖的是同一个 `unit` 字符串，于是 `climatology_skill.csv` 把无量纲数字标成了
`K`/`Pa`/`m s**-1`。

**实测证据**（`outputs/r7_67_sealed_report/evaluation/generic_s42/lead_006h/`）：
同一目录下 `rmse.csv` 的 t2m 6h 是 **2.476919 K**，而 `climatology_skill.csv` 的
`rmse_forecast` 是 **0.263453**；比值 **9.401745** 正好等于该 store 的
`normalization_std[0]`。对全部 **1275/1275** 个 arm-seed-lead-variable 单元验证通过，
0 例外。

因此 `docs/R7_67_SEALED_TEST_REPORT.md` 的两个派生结论**不成立**：

| 原结论 | 修正后 |
| --- | --- |
| 「85 个单元全部输给 climatology，better 10 / worse 75」 | 物理单位下 **better 646 / worse 629**（1275 单元）；与单位无关的 `mse_skill` 符号给出的计数**完全一致** |
| 「t2m 6h 最好臂仍是 climatology 的 **9.09 倍**」 | 物理比值 **0.95**（generic 略优于 climatology）；9.09 是「物理 ÷ 归一化」 |

**为什么必须修而不是绕过。** `mse_skill` 是两个同场 MSE 之比，本来就与单位无关，所以它一直
是对的；错的只有两个 RMSE 列。修法与 `RolloutRMSEAccumulator` 完全对齐（同一个
`training_std`/`units` 契约），并在 `r7_evaluate.py` 加了 fail-closed 断言：两个累加器的
units 不一致就拒绝写文件。

### 1.1 影响范围（实测界定，不扩大也不缩小）

**受影响**：只有 `climatology_skill.csv` 的 `rmse_forecast` / `rmse_climatology` 两列，
以及所有**直接读这两列**的派生表述——即 `docs/R7_67_SEALED_TEST_REPORT.md` 的
「9.09 倍」与「85 单元 better 10 / worse 75」。

**不受影响**（逐个核实，不属于本次更正范围）：

| 消费者 | 依据 |
| --- | --- |
| B2 的胜负计数与 skill gate | 走 #60 比较器，读每个臂（含 `climatology` 臂）的 `rmse.csv`，两侧都是物理单位 |
| `docs/R7_B2_MULTISEED.md` 的 RMSE 表 | 同上；其 climatology 列取自 climatology 臂的 `rmse.csv`（物理），实测 24h = 2.5510 = 0.27134×9.4017 |
| B1/B3 的胜负计数 | 同走比较器 |
| 按天 paired block 分析 | 读 `mse` 差值，与单位无关 |
| `mse_skill` 列本身 | 同场 MSE 之比，与单位无关 |

所以「climatology 在 t2m 上占优」这个**方向性**结论在 B2/B3 上并未被推翻；被推翻的是
**幅度**与 #67 报告里那条「85 单元全负」的计数。

### 2. 用 M2（1–2 月，60 天）段解除单月局限

D1/B2/重切段全是 1 月，`(month,hour)` 桶恒为 4，且打分块就落在拟合桶所平均的那 30 天里。
M2 取 **2016-01-01..2016-02-29**（240 时次，60 天连续）：

| split | 范围 | 时次 | 72h 窗口 | 月份 |
| --- | --- | --- | --- | --- |
| train | `[01-01, 02-17)` | 188 | — | 1 月全月 + 2 月前 16 天，**8 桶** |
| val | `[02-17, 02-23)` | 24 | **11** | 全部 02 |
| test | `[02-23, 03-01)` | 28 | **15** | 全部 02 |

**为什么 val 是 6 天而不是更短**：72h 自由 rollout 需要「1 帧历史 + 12 帧 lead」都在 split 内，
N 时次的块只承载 N−13 个 72h 窗口。重切段的 8 时次 val 承载 **0** 个，所以那里的模型选择
读不到 48/72h；M2 两块都过线（11 / 15）。

**M2 真正买到的东西，以及它没买到的**：2 月的桶由 `02-01..02-16` 拟合，而打分块是
`02-23..02-29`，即基线是在**离拟合日一周之外**被评的，不再是「同一天的均值」。
但这仍**不是**跨季节或跨年证据：两个桶都来自同一个冬季、同一个区域、同一年。
按 `docs/goals/full-auto-campaign.md` §5 的要求，跨季节结论仍标 **UNVERIFIED**。

M2 请求**严格包含** D1 的 120 时次（下标 119）与 B2 的 144 时次（下标 143）作为前缀，
所以早前各段的对照关系不被破坏。

### 3. 预算档位：使用第二阶段常量，不改写第一阶段身份

M2 用 `SECOND_STAGE_NEW_ARTIFACT_BYTES_CAP`（100 GiB）与
`SECOND_STAGE_DECODED_BYTES_CAP`（256 GiB），经 `frozen_protocol(stage="second", ...)`
得到独立摘要 `0685679d…`。第一阶段 16/64 GiB 常量与 `d3161af5…` **未改动**，
D1/B2 receipt 归档的身份保持可复现。

### 4. 冻结协议不得misdescribe自己的数据

`scripts/study_r7_b2_multiseed.py` 的 `protocol_payload` 原先把 B2 段落在代码里写死，
在 M2 上会**宣称错误的数据范围**。现改为：从 store 的 `split_time_ranges` 读取真实范围，
与归档的 B2 范围不符时必须由调用方显式给出 `segment_note`，否则拒绝运行。
B2 的归档摘要 `244af00b…` 经回归测试逐字节钉住（改 `B2_SEGMENT_NOTE` 或
`B2_SPLIT_RANGES` 都会触发）。三个 split 全部比对——重切段沿用了 B2 的 **train** 范围，
只比 train 会让它蒙混过关。

## Consequences（后果）

- `climatology_skill.csv` 的 `rmse_*` 列现在是物理单位；修复前归档的是归一化值，
  乘 store 的 `normalization_std` 即可换算，`mse_skill` 无需换算。
- `docs/R7_67_SEALED_TEST_REPORT.md` 的两条派生结论必须按本文档更正，不能保留原措辞。
- 跨月对照（M2）的新结果记在 `docs/R7_69_BUCKET_EXPANSION.md`。

**代价与放弃的选项**：

| 选项 | 放弃理由 | 代价 / 遗留 |
| --- | --- | --- |
| 直接扩到 2016 全年（1460 时次） | 网络与墙钟约 6×，且本阶段只要求 ≥8 桶；2 个月已把桶数翻倍 | 季节覆盖仍只有冬春；跨年仍 UNVERIFIED |
| 借 ARCO 公共源取多年份 | 会引入**第二个数据身份**，使与 D1/B2 的逐位对照不再可能 | 未采用；保持单一 Earthmover 快照 |
| 保留归一化列、只在文档里更正 | 缺陷会随每次新实验复现；下一个作者仍会跨单位比 | 未采用；改为在代码层 fail-closed |
| 让 M2 沿用 `FIRST_STAGE_*` 上限 | 240 时次的 decoded 估算超出 64 GiB | 未采用；改用独立档位，不动第一阶段摘要 |
| 已归档的 `climatology_skill.csv` 重算 | 重算等于改写历史产物，与「归档不可篡改」冲突 | **保留原样**；改用 `mse_skill` 或乘 std 换算，换算关系记录在本文档 |
