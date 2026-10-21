# 0050 — 全年覆盖单因素轮的授权边界（4 个月 → 12 个月，年份固定 2017–2021）

- **状态**：accepted
- **日期**：2026-10-21
- **范围**：`docs/plans/0024-fy-annual-coverage-round.md` 的 S3 数据覆盖轮

## Context

用户在 2026-10-21 明确指示「用我们最新下载的数据」，并选择「数据方向（全年 store + 全年 vs
4 个月验证）」。这是决策 0038 已授权的方向内的具体一轮：免费区域天气数据的派生产物可以写入
**全新排他** `outputs/` 路径，但本轮的边界仍需逐条写定，否则「数据量大」会变成无法归因的多因子
改动。

D3（数据扩充为最后手段）的前置条件已满足：架构轴 H-A/H-B/H-C、容量轴（决策 0048）、训练信号/
目标组织的若干单因素臂都已按各自预声明出口关闭或未能越过气候态门；本轮是用户点名选择的数据
方向，不是执行者自行重开。已下载的 FY 源覆盖 2000–2022 的 **65×65** 区域，2023 只有四个月块
（`outputs/r7_s2_batch2_20222023/source.nc`）；仓库没有 65→129 的上采样管线，store 契约明确
`spatial_resampling: false`，插值外圈不是观测。

**2026-10-21 追加的时序事实**：授权之后，另一窗口把 tendency-input 轮提交为 `15ba518`
（`model_code_sha256` 由 `3fe747b2…` 变为 `60de3b2d…`），campaign-state 的
`current_round_goal` 随之指向 `docs/goals/s3-tendency-input.md`。本轮因此**不是当前 campaign
轮次**，且本轮的迁移父 attempt04 在 HEAD 上会被 `load_checkpoint` 按设计拒绝。要保持「注册架构
上的纯数据单因素」，本轮必须在最后一个 `3fe747b2…` 修订（H-D 实验提交 `0becf24`）的干净
checkout 上运行，并在那里建一个只含本轮文件的临时提交用于冻结/归档；`fd92268` 已含 H-D
之后的架构轮（`bec3859f…`），不能作为基线。把数据因子叠加到 `60de3b2d…` 会同时改变模型
输入表示与数据覆盖，必须另立决策、不能借用本决策。

## Decision

1. **唯一因子：训练覆盖的月份数，4 → 12；年份固定 2017–2021。** 两臂使用**同一个**全年 store、
   同一套全年归一化、同一模型 spec、同一迁移父（`outputs/r7_s3_rollout_dose_parent_migrated_20261015_attempt04`，
   `model_code_sha256 3fe747b2…`）、同 2400 updates、同 K=4 / depth=4 / dim=192、同注册物理权重、
   同 seed 41/42/43、同 2022 全年 val。**只差训练清单覆盖的月份。**
2. **控制臂 N（4 个月）**：从全年 store 的 train 时间轴取 Jan/Apr/Jul/Oct、每月 1–30 日、
   UTC 00/06/12/18，6h 清单 2,360 条；再按 **72h 完整窗**过滤（`t+6..t+72` 每个目标都在同一
   子集），得到 **2,140** 条（丢弃 220 条跨到 31 日/下个月的窗）。不做这一步，长 rollout 会
   从全年 store 读到其他月份的帧，控制臂就不再是纯 4 个月。**候选臂 F（12 个月）**：
   2017–2021 全部日期，7,302 条（长 rollout 可用 7,291）。两套清单各自独立目录、各自
   `BUILD_COMPLETE`，val/test 清单逐字节相同。
3. **本轮只在 65×65 实例上成立。** `boundary_margins=(1,)`，评分读 `full`（4,225 点）。
   不把 65×65 结果外推为 129×129 宽实例结论；宽实例全年需要新采集，**不在本决策授权内**。
4. **不新增下载、不重建 129×129 源、不写 `data/raw|interim|processed`。** 只读 preflight 后
   `--write` 到全新排他路径；源身份、单位、schema、`BUILD_COMPLETE` 不豁免。
5. **预声明出口（冻结前写死，之后不放宽）**：
   - (a) F 的 72h t2m skill（`full`）在 seed 41/42/43 **全部 > 0**；
   - (b) F−N 的 72h t2m ΔRMSE **三 seed 全部 < 0**；
   - (c) F 的 48h t2m skill **三 seed 全部 > 0**；
   - (d) F−N 在 6/12/24h 的 **seed 均值** ΔRMSE **≤ 0**。
   全真 → 只提请独立复核/验收，不自动进 S4；任一 (a)–(d) 假 → **永久关闭「全年覆盖」这一支**：
   不扫月份组合、不调 updates、不加 seed、不把 2000–2016 塞进同一支、不与已关轴组合。
6. **桥 (e) 不是科学判定**：控制 N 的 5-lead t2m RMSE 与已登记的 65×65 窄 v3 四个月臂
   （`record:s3-v3-budget-dose`，1600 updates）比较，差值不得超过该登记臂自身的 seed 极差或
   其均值的 5%（取较大者）。它只用于发现身份/工程错误；失败记 `identity/engineering unresolved`，
   不进入 (a)–(d)。**归一化、case 集和 updates 都不同，桥不能当作等价性证明。**
7. **双卡（决策 0047）**：`device_mode: "dual"`，一卡一 seed，`neighbour_reserve_bytes = 6 GiB`；
   每阶段 spawn 前只读余量门，**不对任何邻居进程发信号**。失败 attempt 原样保留、全额计费、
   不做 in-place resume。
8. **登记**：`test.jsonl` 从不读取，`test_read=false`、`scientific_claim=false`；结果写证据页 +
   index record + 账本行 + §8 + campaign-state，并核验精确 CI run。

## Consequences

- **代价 1（科学范围）**：结果只在 65×65 上成立；它回答不了「129×129 宽实例全年是否更好」，
  也无法与宽实例的 2400-update 臂做逐字节配对。宽实例全年需要重新采集（约 34–36 h、~1 TB
  网络、~27 GiB store），本轮明确不做。
- **代价 2（归一化混杂）**：两臂共用全年 train 拟合的归一化，因此控制臂 N 与注册的四个月窄臂
  不是同一把尺；桥 (e) 只能发现量级错误，不能证明两者等价。
- **代价 3（预算/算力）**：六次 2400-update 训练 + 每臂 5 lead 评分；固定 updates 意味着
  treatment 只看到约 1/3 个 epoch，control 约 1 个 epoch。即使 F 更好，也只能说明「同更新预算下
  更多月份覆盖更有效」，不能说明「更多数据在任意预算下都更好」。
- **代价 4（单因子封闭性）**：2000–2016 未使用；「更多年份」是后续独立因子，不能与本轮合并解释。
  2023 只有四个月，test 仍封存，不构成完整未见年份的全年确认。
- **收益**：这是目前唯一能在**不新增下载、不改模型代码**的前提下，直接检验「现有模型是否被
  4 个月训练覆盖限制」的一轮；失败会永久关闭这条支，避免在数据方向上反复打转。
- **风险与缓解**：合并 2023 块必须断言坐标/变量/单位一致（已在 `merge_receipt.json` 记录
  `coordinates_equal/variables_equal/units_equal`）；`q850/q500` 的小 std 风险沿用 FY 页记录；
  两臂清单的 `dataset_identity` 不同，禁止混用（各自独立目录 + `BUILD_COMPLETE`）。

## References

- `docs/plans/0024-fy-annual-coverage-round.md`（本轮工作态计划与实际数据准备事实）
- 决策 0038（自主研究与数据扩充授权）、0047（双卡 seed 分配）、0048（容量轴关闭）
- `docs/R7_S3_FY_ACQUISITION.md`（FY 源、65×65 选择、无 2023 全年）
- `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`（科学门形式）
- R-006（先冻协议）、R-028（离线禁网）、R-054（共驻只读）
