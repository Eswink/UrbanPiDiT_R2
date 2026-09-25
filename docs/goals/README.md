# 目标长文

goal 模式（`/goal <objective>`、无头 `--target <objective>`）的**长文目标源**放在本目录。

## 为什么需要这个目录

goal 模式的 objective 有 **4000 字符硬上限**（超出直接报错），而完成判定由**独立 verifier**
在**不能调用工具**的回合里做 —— 它读不到文件，只能核对对话里已出现的证据。

所以长任务提示词的标准写法是：**把细节写成本目录下的一份文件，objective 里只放要点 + 指向它的路径。**

## 命名与写法

- 文件名：`<kebab-slug>.md`（不需要编号 —— 目标是可被取代的活文档，不是决策记录）。
- 开头写清三件事：
  1. **交付物清单**：objective 里会被逐条核对的条目（越可核查，自动续跑越有效）；
  2. **停止条件与预算**：目标未达成会自动续跑，不写预算容易无限扩张 scope；
  3. **判据的证据来源**：verifier 不能调工具，所以判据必须能靠对话里已有的证据核对。

配套要点（已在 `~/.zcode` 的 memory 与 `docs/R7_GPU_BRINGUP_BRIEF.md` 中记录）：

- 目标状态机是 `active / paused / budget_limited / complete`，**不要自己宣布完成**。
- 被取消 / 排队 / skipped 的运行**不算通过**，判据里要写明去查哪类 run。

## 现有条目

| 文件 | 主题 | 备注 |
| --- | --- | --- |
| [`gpu-bringup-3090.md`](gpu-bringup-3090.md) | R7 GPU 工程验收（双 3090） | 含 Stage A/B/C、交付物 D1–D7、数据限制、证据落点 |
| [`open-issue-resolution.md`](open-issue-resolution.md) | 按序解决并关闭全部 open issue（含真实数据） | 依赖顺序、真实 ERA5 源与实测成本、自迭代边界、停止条件；**已完成**（8/8 closed，2026-09-25） |
| [`iteration-campaign.md`](iteration-campaign.md) | 迭代战役 #59–#68：先修验收链再验证过程递归 | S0–S5 依赖顺序、各 issue 验收、预算冻结、诚实纪律 |
| [`d1-acquisition-and-b0.md`](d1-acquisition-and-b0.md) | D1 获取（Earthmover spatial，决策 0004）+ #64 B0 可学习性 | 源已冻结、实测速率、两段范围与非目标、预算/停止条件 |
| `docs/R7_GPU_BRINGUP_BRIEF.md` | （已迁移） | R-034 登记为早于约定的例外；现仅为指向本目录的指针，不再维护第二份定义 |
