# 0006 R7 V2 的主计划与每轮对表（含 N1 审计轮的目标长文）

- **日期**：2026-09-30
- **状态**：已完成（仓库侧；审计轮本身尚未开工——按"逐轮触发"等用户发起）
- **触发**：用户提出两件事——「下一轮的提示词是什么」，以及「goal 不会自己沿着方向迭代，
  想把长期迭代下放给代理，但每轮不能跑偏，需要 recheck 与主计划」。

## 计划正文（批准后修订）

### 1. 问题

- **下一动作已被冻结文档写死**（`docs/goals/main-model-v2-rw-b-subtraction.md:206-208`：
  停止发明新模块 → 记录可反驳假设 → 转向重查 forecast state / training objective / data regime；
  `docs/plans/0004-r7-main-model-v2.md:323-325` 另列 autoregressive exposure，`:335` 命中
  「连续两次定向改动无收益」），**但没有任何机制保证下一轮真的从它出发**。
- **对表全部靠人工**：`R7_TASK_QUEUE.md` 停更、`full-auto-campaign.md` 状态过期、GPU 账本是手写散文、
  `check_goal_brief.py` 非阻断且不进 CI。
- **harness 不会自续跑**：校验器两种失败形态已定位（决策 0024），且仓库有 2026-09-23 的
  「禁止后台续跑」用户决定（`docs/R7_MANUAL_ITERATION.md:3-6`）。

### 2. 计划的动作

1. 主计划活文档 `docs/goals/main-model-v2-campaign.md`（节点图 N0–N4、每轮对表六条、账本表）。
2. 机器 recheck `tools/check_campaign_state.py` + 自测，接入 CI 新步骤（决策 0025）。
3. 决策 `0025`；`docs/rules/CHANGELOG.md` 与 `ci-and-verification.md` 同步。
4. 下一轮（N1 审计轮 + 一条预声明可证伪臂）的目标长文 `docs/goals/main-model-v2-pivot-audit.md`
   + 可粘贴的**完整** objective。
5. `goal-loop` 技能加「每轮对表」固定动作；本计划按 R-032 归档。

## 实际结果

### 完成情况

全部完成：`docs/goals/main-model-v2-campaign.md`（主计划）、`tools/check_campaign_state.py`（约 330 行）
+ `tests/test_check_campaign_state.py`（13 项，含 7 条反证与"真实仓库必须通过"）、
`docs/goals/main-model-v2-pivot-audit.md`（N1 轮次，带 `<!-- round-node: N1 -->`）、
`docs/decisions/0025-campaign-master-plan-and-per-round-recheck.md` + 索引行、
`docs/rules/CHANGELOG.md` 与 `docs/rules/ci-and-verification.md` 同步、`.agents/skills/goal-loop/SKILL.md`
的前置条件/步骤/检查点三处更新。**本轮不跑实验、不碰 GPU**；审计轮等用户触发。

### 与计划的差异（4 处）

1. **计划外发现并修掉一个既有工具缺陷**：写 N1 长文时 `check_goal_brief.py` 报 `G-02 没有 objective 引用块`
   ——根因是它按"标题含 objective 的第一个节"选取，而 N1 长文的 **H1 里就写着 "training objective"**，
   于是标题节遮蔽了真正的 §0。已修（改选第一个**真的带引用块**的候选节）+ 加反证测试
   `test_title_mentioning_objective_does_not_shadow_the_objective_section`（19 项测试通过）。
2. **工具的契约在实现中收紧了两次**：账本行必须用 `record:<id>` 引用才做数值交叉核对（否则如实列 note）；
   C-06 登记的是**证据页**而不是 goal 长文——因此主计划状态块增列 `previous_round_evidence`。
3. **上一轮（减法轮）早于本机制**：它的 §8 下一动作不可能写出节点号 `N1`。没有静默放过，
   而是在状态块里显式声明 `previous_round_predates_mechanism: true`，由工具输出一条 note——
   机制**前瞻生效**，历史文档不回溯改写。
4. **账本只有 2/6 行有索引支撑**：RW-B pilot 与减法轮有 `record:` 可核，M2 段/第一/二/三轮没有索引记录，
   工具把它们列为「无机器支撑」的 note 而不是当成已核。把索引补成全量账本留给后续轮次。

### 验证

| 检查 | 命令 | 结果 |
| --- | --- | --- |
| 两份新长文结构 | `.venv/bin/python tools/check_goal_brief.py --brief docs/goals/main-model-v2-campaign.md --brief docs/goals/main-model-v2-pivot-audit.md` | `briefs=2 failures=0 advisories=0` |
| 对表 | `python tools/check_campaign_state.py` | `node=N1 failures=0 notes=6`，exit 0 |
| 校检器自测 | `.venv/bin/python -m pytest -q tests/test_check_goal_brief.py` | 19 passed |
| recheck 自测 | `.venv/bin/python -m pytest -q tests/test_check_campaign_state.py` | 13 passed |
| 阻断规则 | `python tools/check_conventions.py` | 37 条 0 违规 |
| 全量测试 | `.venv/bin/python -m pytest -q` | 见收尾提交（含本机与干净检出两个口径） |
| CI | `Check campaign state` 步骤 | 首次随本提交运行（作业由八步变九步） |

### 遗留

- 审计轮（N1）尚未开工：用户选择"逐轮触发"，提示词已给出，等你发起。
- 账本 4/6 行无索引支撑（如实列为 note）；把索引补成全量账本是后续可做的一件小事。
- 干净检出下 `pytest` 的 skip 数会高于本机（既有 `test_r7_72_rw_b_study` 等依赖 M2 store 的用例），
  收尾时按既有做法两个口径都记。
