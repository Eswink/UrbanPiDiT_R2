# 0003 goal-loop 技能：目标撰写 + harness goal 不可用时的会话手工循环

- **来源**：工作态计划（ZCode 计划模式，2026-09-28）；开工时 `git rev-parse HEAD` = `107b2cd`
- **定稿**：2026-09-28

## 目标

把「目标」立成仓库能力：一个 skill 覆盖 ①把一次迭代包成 `docs/goals/` 长文与
≤4000 字符单段 objective；②harness 的 goal 模式不可用、或用户不想用它时，按会话手工推进
目标的完整循环（四态状态机、每轮固定动作、预算与停止、独立复核替代、不得自宣完成）。
同时把 R-034 现在**纯人工自觉**的结构子集机械化成一个**只读、非阻断**校检器。

立项依据：`docs/skills/README.md:9` 要求「重复 ≥2 次且步骤顺序固定」——撰写流水线已重复两次
（`docs/goals/m1-and-rw-a-iteration.md` §6、`docs/goals/v2-round-two-attribution.md` §6 同一形状）；
手工循环的纪律已有 `docs/R7_MANUAL_ITERATION.md:8-30`（会话驱动、绑定精确 SHA、无后台续跑）。

## 计划要点（三个设计选择经用户确认）

1. 覆盖「撰写 + 手工循环」；
2. 加只读校检器，但**不接入** Stop hook / CI / 阻断规则集；
3. 进度状态写在目标文件内的 `§8 进度块`。

交付物：`.agents/skills/goal-loop/SKILL.md`（7 段固定骨架 + 第 8 段规范示例）、
`tools/check_goal_brief.py`（`G-01`…`G-08` + 建议项 `A-01`/`A-02`）、
`tests/test_check_goal_brief.py`（含每条规则的反证与技能示例自洽断言）、
治理层同步（`docs/skills/README.md`、`AGENTS.md` 路由表、`docs/goals/README.md`、
R-034 执行方式补一句、`docs/rules/CHANGELOG.md`）、决策记录、R-009 基线更新、本归档。

## 实际结果

- **完成情况**：七项交付物全部完成，无未做项。计划里的三条验证全部实跑并通过；
  计划外只多做了一件必须做的事——把第二轮自己占用的决策编号补齐（见下）。

- **与计划的差异**（4 处，均已核对）

  1. **决策编号 0014 → 0015**。计划写 0014；落地时发现第二轮的 goal 运行已经用掉了
     `docs/decisions/0014-round-two-pooled-query-capacity-control.md`，而被它占用时
     **没有同步 `docs/decisions/README.md` 的索引**（索引当时止于 0013）。R-033 的
     「编号不得复用」机械检查因此报了第二条 0014。处置：新决策改名为
     `0015-goal-loop-scope.md`，并**回填 0014 的索引行**（一行、只补索引、不改内容）。
  2. **起点 SHA 与计划时不同**：规划时 HEAD 是 `36bf10b`（我刚提交的目标长文），
     落地时第二轮已完整落地（`d8aff68`…`107b2cd`），因此开工起点记为 `107b2cd`。
     计划里引用的两份目标长文路径仍成立。
  3. **校检器多了一个 `--root`**（计划未列）：`main()` 默认以本检出为根，
     测试要在 `tmp_path` 下校验长文，于是把根变成显式参数（只读，无副作用）。
  4. **技能示例的字符标注**：计划未要求，落地时实测示例 objective = **216 字符**并写进
     示例标题（避免示例里出现一个编造的数字）。

- **验证**（实跑命令与结果）

  | 命令 | 结果 |
  | --- | --- |
  | `tools/check_goal_brief.py --brief docs/goals/m1-and-rw-a-iteration.md --brief docs/goals/v2-round-two-attribution.md` | `failures=0`，退出码 **0**（仅 `A-01` 缺进度块、`A-02` §0 未标长度两条建议） |
  | `tools/check_goal_brief.py --brief docs/goals` | 7 份长文：`failures=15 advisories=9`——15 条全部落在**早于约定的五份**（`gpu-bringup-3090` / `open-issue-resolution` / `iteration-campaign` / `full-auto-campaign` / `d1-acquisition-and-b0`），按设计**只报告、不回溯改写** |
  | `pytest tests/test_check_goal_brief.py -q` | **18 passed**（含 7 条参数化反证、技能示例自洽、规则号防漂移、只读断言） |
  | `pytest tests/test_check_goal_brief.py tests/test_check_planner_plan.py -q` | **65 passed** |
  | `pytest -q`（全量） | **1282 passed / 2 failed / 3 skipped**；两条失败是 `tests/test_check_conventions.py` 对**真实仓库**的断言，读其输出确认唯一成因是既有的 R-044 命中（未跟踪遗留文件），**不是本次改动引入** |
  | `tools/check_conventions.py` | 34 条阻断规则：**1 条失败**，即上述同一条既有 R-044；其余全部 PASS |

- **遗留**

  1. `tests/fixtures/r7_equivalence_recipe.py`（第一轮遗留、未跟踪、hook 拒绝代理删除）
     必须**由用户在 ZCode 之外删除**；在它消失前，本机 `check_conventions` 与全量 pytest
     会各报同一条 R-044（CI 不受影响——它跑提交树）。未绕过、未放宽。
  2. 校检器按设计**不在** CI/Stop hook 里；要升级为阻断规则，前提是先处置五份早于约定的长文
     （修好或给出显式例外清单），本轮刻意不做——见决策 0015 的 `Consequences`。
  3. 手工循环**没有**独立 verifier：技能已写明机械门禁 / 精确 SHA 证据 / planner 交叉复核
     **都不是** verifier，完成裁定仍需用户或下一轮以新证据重审。
