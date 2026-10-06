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

- 目标状态机是 `active / paused / budget_limited / complete`，**不要自己宣布最终 goal 完成**。
- 决策 0030：用户只把控总体方向；实验、时长、普通决策与节点推进常设下放，同一 goal 内可按
  冻结前置/出口与每节点对表连续推进，不逐轮等用户。独立审阅核证据，不是新推进许可；
  付费/独占/main 合并/破坏性等保留边界不变，旧长文只作其当轮记录。决策 0038 仅在主模型超
  气候态方向具名扩展免费区域多年度/四季数据下载与自审 preflight 后的新 outputs 路径派生发布。
- 被取消 / 排队 / skipped 的运行**不算通过**，判据里要写明去查哪类 run。

## 怎么写、怎么跑

- 做法见 `.agents/skills/goal-loop/SKILL.md`：长文骨架（§0 objective … §8 进度块）、
  objective 压缩（单段、≤4000、**自带交付物清单**）、以及 harness goal 不可用或不用它时的
  会话手工循环（四态状态机、每轮固定动作、独立复核替代、不得自宣完成）。
- 结构子集可机械检查（只读、**非阻断**）：
  `.venv/bin/python tools/check_goal_brief.py --brief docs/goals/<slug>.md`。
- **早于该约定的长文不回溯改写**（它们是证据）：`gpu-bringup-3090` /
  `open-issue-resolution` / `iteration-campaign` / `full-auto-campaign` /
  `d1-acquisition-and-b0` 会被校检器报 `G-02/G-06/G-07/G-08`，那是报告而非待办。

## 现有条目

| 文件 | 主题 | 备注 |
| --- | --- | --- |
| [`gpu-bringup-3090.md`](gpu-bringup-3090.md) | R7 GPU 工程验收（双 3090） | 含 Stage A/B/C、交付物 D1–D7、数据限制、证据落点 |
| [`open-issue-resolution.md`](open-issue-resolution.md) | 按序解决并关闭全部 open issue（含真实数据） | 依赖顺序、真实 ERA5 源与实测成本、自迭代边界、停止条件；**已完成**（8/8 closed，2026-09-25） |
| [`iteration-campaign.md`](iteration-campaign.md) | 迭代战役 #59–#68：先修验收链再验证过程递归 | S0–S5 依赖顺序、各 issue 验收、预算冻结、诚实纪律 |
| [`full-auto-campaign.md`](full-auto-campaign.md) | 全自动战役：#65 归因→方法验证→#67 封存（含 #64 余项/#66/#68/#59） | **显式授权自主拍板**（预算≤24 GPU-h 第二批、数据≤16 GiB 内扩展、关闭 DONE issue）；预诊断四项；各 issue 完成判据 |
| [`d1-acquisition-and-b0.md`](d1-acquisition-and-b0.md) | D1 获取（Earthmover spatial，决策 0004）+ #64 B0 可学习性 | 源已冻结、实测速率、两段范围与非目标、预算/停止条件 |
| [`m1-and-rw-a-iteration.md`](m1-and-rw-a-iteration.md) | 主模型 V2 第一轮：#71 已知时空输入贯通 + #72 RW-A 位置化 process 读写 | 交付物 D1–D8、planner 委派草稿的 6 处更正、本轮 ≤1.5 GPU-h、停止条件；**已完成**（`2586477`/`17d7722`/`14c7a24`，CI 三次 success） |
| [`v2-round-two-attribution.md`](v2-round-two-attribution.md) | V2 第二轮：四臂 × 三种子拆开「时空输入」与「位置化读写」，并加容量控制臂 | 消除 round one 的三条局限（开关不可分、容量不对齐、两种子）；≤1.0 GPU-h；C−D 未获支持即如实写；**已完成**（实测 0.5253 GPU-h，结论见 `docs/R7_71_72_ROUND_TWO_ATTRIBUTION.md`） |
| [`v2-round-three-m1-attribution.md`](v2-round-three-m1-attribution.md) | V2 第三轮：给时空输入（M1）补容量控制臂（constant / shuffled） | 把 B−A 的 1.0–2.0 K 拆成「信息」与「容量/偏置」；预登记 primary；≤0.9 GPU-h |
| [`main-model-v2-campaign.md`](main-model-v2-campaign.md) | 旧主模型V2路线及收尾（N0–N5，对表/账本/历史科学出口） | 旧V2唯一权威；N5六issue收尾有证据、machine N5/active保留，最终complete不自裁；新方向另见下一行 |
| [`main-model-climatology-campaign.md`](main-model-climatology-campaign.md) | 新长期主模型超气候态方向（S0–S4）：gap、单因素探索、多年度/四季数据、未见确认 | 新方向唯一权威/0038；S3 进行中（batch-2 获取、v2 确认实例已登记）；objective单段≤4000、代理与11技能触发明确、`--campaign`显式对表 |
| [`s2-climatology-mechanism-screening.md`](s2-climatology-mechanism-screening.md) | S2 节点：单因素机制筛选（#77 → #78 R-A → #79 → #78 R-B） | 已完成：四项全部登记（unresolved/worsened/supported-needs-review/worsened），回主线动作明确 |
| [`s3-confirmation-baselines-and-candidate.md`](s3-confirmation-baselines-and-candidate.md) | S3 节点：同数据基线、incumbent 重训、R-C/预算/短rollout筛选 | v3-D2/D3/BD及短rollout已登记；短rollout13/45守门negative，S4未开始，test未评分 |
| [`s3-long-rollout-supervision.md`](s3-long-rollout-supervision.md) | S3 接续：直接12步全BPTT，监督48/72h生成历史 | S3开发screen完整：主格supported、控制gate0/45，长lead气候态仍全负；归档15CSV精确，全额登记，非S4/科学接受 |
| [`main-model-v2-rw-b-round.md`](main-model-v2-rw-b-round.md) | V2 第二阶段：RW-B 局部门控求解状态（#72 M2-B） | 已完成（negative，`docs/R7_72_RW_B_PILOT.md`）；含 goal 校验悬挂的恢复记录指针 |
| [`main-model-v2-rw-b-subtraction.md`](main-model-v2-rw-b-subtraction.md) | RW-B 减法归因（(a) 门控+锚定提案 / (b) Z 递推 / (c) role 标记） | 已完成（`branch=stop-confounded-control`：负控制按构造退化，不能归因） |
| [`main-model-v2-pivot-audit.md`](main-model-v2-pivot-audit.md) | N1 转向审计（四块 0 GPU-h）+ 冻结随机 Z 可证伪臂 | 已完成（两 seed 反号 → `cannot-distinguish`，只提议 N2d） |
| [`n1-cost-supplement-repair.md`](n1-cost-supplement-repair.md) | N1 独立 evaluation 成本补测的失败审阅、v2 重测（逐评估新进程）与 GPU 共驻政策落地 | **prepared（2026-10-01 扩围）**（未执行；D0 落共驻政策、D2 改余量门槛；执行那一刻按决策 0021 取具名授权） |
| [`n2a-m3-process-supervision.md`](n2a-m3-process-supervision.md) | V2 阶段 A（节点 N2a / #73）：过程监督的尺度修复与「三类时刻」语义（三分离字段 + 有界三臂对照） | **budget_limited（2026-10-02）**；工程/尺度已验证，唯一尝试6训练/7评估后deadline失败，D5完整对照缺失；0.497254GPU-h全额记账，不重试、不推进或关闭issue |
| [`n3-m4-autoregressive-rollout.md`](n3-m4-autoregressive-rollout.md) | V2 阶段 B（节点 N3 / #74）：真实 2 步可微自回归（`L6+λL12`）+ matched-Generic V2 接线 | **prepared（2026-10-02）**；warm-start 自既有 +6h 父 checkpoint；≤1.0 GPU-h |
| [`n4-m5-confirmation.md`](n4-m5-confirmation.md) | V2 阶段 C（节点 N4 / #75）：≥3 seed 三臂最小确认闭环 + adaptive 决定 | **prepared（2026-10-02）**；adaptive 默认不启动；≤2.5 GPU-h |
| [`v2-issue-closeout.md`](v2-issue-closeout.md) | V2 收尾轮（节点 N5）：补 #71/#72 完成判据、登记六条判定并关闭 #70–#75 | **已完成（2026-10-04）**：判定六段+六行登记、0037 生效；关闭提交 `f0b7705` 经 CI 37215981072（12/12 success）ff 落 main，匿名复核六 issue closed（16:29Z）；关闭≠科学成功 |
| [`v2-autonomous-completion-and-closeout.md`](v2-autonomous-completion-and-closeout.md) | 0030 新授权下的 N2a 补全 → N3 → N4 → N5/有条件关闭 | paused；N2a已补全但原any-unresolved出口触发，N3/N4/N5前置BLOCKED；最终goal未裁定完成 |
| [`n2a-m3-validation-complement.md`](n2a-m3-validation-complement.md) | N2a独立23项val补测与跨attempt来源/全额成本汇总 | **N2a规定交付完成，科学paused**；尺度/三时刻/反证及23新val/30总覆盖、D6/index18/账本齐，B/C精确CI九步成功；negative/mixed如实验收，原失败/归档与any-unresolved暂停不改；全权授权接续与验收见计划0013 |
| [`v2-l6-confirmation-current.md`](v2-l6-confirmation-current.md) | V2 阶段 C（节点 N4 / #75）：actual C 三臂确认 + M1 补全 + UTC 统计的实证窗口 | **实证完成（2026-10-04）**：C 144/144 独立接受、M1 24/24 verify、precision 独立接受、两侧 UTC 链完成；E-245–E-247 |
| [`v2-remaining-stages-exploration.md`](v2-remaining-stages-exploration.md) | 0032独立autoregressive-exposure主线，接续B/C/D/E与必要主线探索 | **已收尾（2026-10-04）**；M3辅助监督终结且保留原paused出口；六 issue 已按 N5/决策 0037 关闭（负面/未决读数不改） |
| `docs/R7_GPU_BRINGUP_BRIEF.md` | （已迁移） | R-034 登记为早于约定的例外；现仅为指向本目录的指针，不再维护第二份定义 |
