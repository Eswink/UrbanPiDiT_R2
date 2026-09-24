---
name: issue-lifecycle
description: 当开始、推进、或要关闭一个 issue，或要判断某项工作能否标 DONE 时使用。含证据要求、绑定精确 SHA 与"取消不算通过"的判定规则。
---

# 推进与关闭 issue

## 何时使用

- 要开始一项被跟踪的工作，或推进一个已开着的 issue；
- 要判断某项工作**能否**标 DONE；
- 一个父 issue 下有关卡仍被真实依赖阻塞（缺数据、缺硬件）。

**不适用**：
- 纯粹的工程收尾（那是 `ci-workflow-triage` 与 `result-freeze`）；
- 发布或合并 PR（需单独授权，见主契约硬约束）。

## 前置条件

- 已读 `docs/R7_TASK_QUEUE.md`：它记录了任务状态、剩余研究关卡与它们的阻塞原因。
- 已确认当前分支与 commit：`git rev-parse HEAD`。
- 已确认工作树状态：`git status` 无意外改动。

## 步骤

1. **先查队列，不要重建已完成的结论。**
   `docs/R7_TASK_QUEUE.md` 的 "Prior work is not pending" 段落明确列出哪些已做完
   （源码填充值/预算正确性、四季源与重放、多种子研究、基线与缓存恢复等）。
   **不要**按旧对话摘要重做它们。

2. **状态推进按四态**：`TODO → IN_PROGRESS → VERIFY → DONE`；
   被真实外部依赖阻塞时标 `BLOCKED` 并**写出那个依赖**（不是"暂时没做"）。

3. **VERIFY 阶段绑定精确 SHA。** 验证必须指向确切的代码版本：
   - 记录实际运行的 run id 与 CI run id（`docs/R7_CPU_REFINEMENT_RESULTS.md` 的形态）；
   - 用**确切的 SHA** 关联，不要用"最新提交"这种会漂移的描述。

4. **收集验收证据。** 关闭前必须具备：
   - 改动面（哪些文件、哪些行为）；
   - 实际执行过的验证命令与结果（含 pass/skip 计数）；
   - 未解决的限制（**照实写**，含负面结果）；
   - 若涉及产物：其 digest 或 run id。

5. **判断能否关闭。**
   - 验收测试通过 → 可以关闭该**工程子项**；
   - 但若父 issue 还含真实数据实验、硬件基准或研究比较 → **父 issue 保持开着**。

6. **记录到文档。** 把结论与证据写进对应 `docs/R7_*.md`，并在
   `docs/rules/CHANGELOG.md` 追加规则/能力层面的变更（若有）。
   若这次工作产生了**跨任务的决策**（改了目录结构、数据契约、执行方式），
   另写一份决策记录 —— 见 `.agents/skills/decision-record/SKILL.md`。
   若这次工作是**按计划模式完成的**，把计划归档进 `docs/plans/NNNN-<slug>.md`
   并补「实际结果」一节（R-032）。

## 检查点

- **步骤 1 后**：若你要做的事已在 "not pending" 列表中，**停止**并改用现有结论。
- **步骤 3 后**：若无法绑定到确切 SHA，**不要**进入 VERIFY —— 无法复核的验证等于没验证。
- **步骤 5 前**：确认这次运行不是 `queued` 或被取消的。
  **取消或排队的运行不算通过**（`docs/R7_MANUAL_ITERATION.md:17`）。
- **步骤 5 前**：确认没有为了让门禁变绿而放宽测试/阈值/断言。
  若某项确实做不到，如实记 `BLOCKED` + 原因，而不是降级判据。
- **关闭前**：确认评论里写的是**实际** commit、精确 SHA 的运行链接、pass/skip 计数
  与未解决限制，而不是重复的状态占位符。
- **关闭前**：若用了计划模式，确认计划已归档到 `docs/plans/` 且含「实际结果」
  （`python tools/check_conventions.py --rule R-032` 会检查）。
- **关闭前**：若产生了跨任务决策，确认 `docs/decisions/` 里有对应记录
  （`--rule R-033 --rule R-036`）。

## 常见失败

- **按旧摘要重做已完成的工作**：先读 `docs/R7_TASK_QUEUE.md`。
- **把工程通过当成科学结论**：单元测试增长、wheel 能编译、单批次有改善、
  oracle 挑到有利深度——这些**都不**证明预报技巧
  （`docs/R7_EXPERIMENT_HANDOFF.md` 的 "What does not justify closing a research parent"）。
- **把混合/负面结果藏起来**：本项目明确要求**保留全部负面结果**
  （例：三种子对照显示空间反馈对某些风场分数有帮助但使 T500 变差）。
  负结论是有效结论。
- **为"有进展"而造新模块**：不得为了回避"结果是混合的"而新增投机性模块。
- **声称 SOTA 或把推理步数当加速**：`docs/R7_MANUAL_ITERATION.md` 禁止把
  reasoning-step count 说成 wall-clock speedup，也禁止把想要的 SOTA 当作实测结果。
- **把 BLOCKED 写成"待办"**：BLOCKED 必须指名真实依赖（如"需要真实共址的精细分辨率动态真值"）。

## 完成判据

- 状态与 `docs/R7_TASK_QUEUE.md` 一致，且 BLOCKED 项写明了真实依赖；
- 验证绑定到确切 SHA，评论含 run id 与 pass/skip 计数；
- 未解决限制与负面结果如实记录；
- 父 issue 在仍有真实数据/硬件/研究关卡时保持开启。

## 明确不覆盖

- 不覆盖合并 PR、发版、force push（需显式授权）；
- 不覆盖研究关卡本身的科学判定（只覆盖"证据是否齐备"）；
- 不覆盖 CI 失败的技术排查（见 `ci-workflow-triage`）；
- 不覆盖产物定稿的 digest 核对流程（见 `result-freeze`）。
