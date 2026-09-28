# 0015 goal-loop 能力：目标撰写与会话手工循环，校检器只读且非阻断

- **日期**：2026-09-28
- **状态**：accepted
- **代码 SHA 范围**：`107b2cd`（起点）→ 本轮提交
- **依据证据**：`.agents/skills/goal-loop/SKILL.md`；`tools/check_goal_brief.py`；
  `tests/test_check_goal_brief.py`（12 个测试函数，含每条规则的反证与技能示例自洽断言）；
  `docs/goals/m1-and-rw-a-iteration.md` §6 与 `docs/goals/v2-round-two-attribution.md` §6（同一流水线两次）；
  `docs/R7_MANUAL_ITERATION.md:8-30`

## Context

goal 模式（`/goal <objective>`）在本仓已经用过四份目标长文，但**只有撰写侧的约定**
（R-034 + `docs/goals/README.md`），没有执行侧的能力；而且 R-034 自述为「人工自觉」，
`docs/rules/README.md` 把它列为未机械化规则之一。三件事在实测中反复出现：

1. **objective 有 4000 字符硬上限**，且完成判定由**不能调用工具**的独立 verifier 做——
   判据必须能在对话里当场核对，读不到文件的判据等于不存在（`docs/rules/artifact-storage.md:61-62`）。
   最近两份长文的 objective 分别压到 1373 与 1620 字符，压缩过程是重复劳动。
2. **harness 的 goal 模式可能不可用或不该用**：客户端没有该命令、objective 超限、
   需要跨会话人工看护，或用户明确不想用它。此时若没有替代流程，容易退化成
   「执行者自己宣布完成」——而本仓的纪律恰恰是**不自行宣布完成**（`docs/goals/README.md:22`）。
3. **同一套流水线已经重复两次**且形状一致（`docs/goals/m1-and-rw-a-iteration.md` §6、
   `docs/goals/v2-round-two-attribution.md` §6：侦察 → `planner-delegation` 委派 → JSON 过契约 →
   内容复核（每次都查出事实错误）→ 长文 + 单段 objective → 提交），满足
   `docs/skills/README.md:9` 的「重复 ≥2 次且步骤顺序固定」门槛。

## Decision

**采用：把「目标」立成一个能力（`goal-loop`），并把它的结构子集机械化为一个只读、非阻断的校检器。**

1. **技能范围**：既覆盖撰写（长文骨架 §0 objective … §8 进度块、objective 单段 ≤4000、
   自带交付物清单），也覆盖执行——包括 harness goal 不可用/不使用时的**会话手工循环**
   （每轮固定动作、四态状态机、预算与停止条件、**不自行宣布完成**、无后台续跑）。
2. **进度状态写在目标文件内**（`§8 进度块`：状态机 + 已完成/未做/下一动作），单一来源，
   随文件提交（R-037）；不另建平行台账。
3. **校检器只读且非阻断**：`tools/check_goal_brief.py` 检查结构子集（`G-01`…`G-08`，
   建议项 `A-01`/`A-02`），**不接入** Stop hook、CI 或阻断规则集；规则号属于它自己的命名空间，
   不是 `docs/rules/` 的 `R-0xx`。R-034 的级别、判据与例外**不变**，只补一句说明。
4. **不回溯改写旧长文**：早于本约定的五份 `docs/goals/*.md` 是证据，校检器对它们的
   `G-02/G-06/G-07/G-08` 命中只报告。
5. **独立复核的边界写进技能**：机械门禁 + 精确 SHA/run id + planner 交叉复核**都不是 verifier**，
   都不能给「完成」定论。

## Consequences

- **好处**：目标撰写从"每次重写"变成按骨架填；4000 字符与单段要求可在提交前机械核对；
  goal 模式不可用时有一条明写的降级路径，且它不会伪装成有独立判定。
- **代价**：多一份非阻断工具、技能与索引行需要维护；校检器**只判结构、不判内容真伪**，
  因此它不能替代 R-034 的人工自觉，也不能替代独立 verifier；手工循环里"是否完成"仍需用户
  或下一轮以新证据重审，**这一点不会因为有了校检器而变简单**。
- **执行约束**：校检器若被接入阻断门禁，必须先修好五份旧长文或给出显式例外清单——
  本轮**刻意不做**，以免把"结构新约定"变成"改写历史证据"。
- **风险**：技能内嵌示例与校检器可能漂移 → 由 `tests/test_check_goal_brief.py` 的
  「示例必须被接受」「强制的规则号必须写在技能里」两条断言钉住（沿用 `planner-delegation` 的手法）。
- **不改**：`.zcode/config.json` 的 hooks、任何科学判据/阈值、`data/**` 与归档。
