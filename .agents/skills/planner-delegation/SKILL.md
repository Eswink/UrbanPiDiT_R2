---
name: planner-delegation
description: 当需要多步方案设计、迭代计划或 TODO 撰写、架构与技术选型、大改动前风险评估时使用；含委派前置、JSON 契约校验与降级路径。不用于单步修改，也不外委科学判据。
---

# 委派 planner 子智能体

## 何时使用

- 需要**多步方案设计**（任务拆解、步骤排序、依赖识别）；
- 一次迭代结束后要撰写**计划文件或 TODO**（`docs/plans/` 定稿、`docs/R7_TASK_QUEUE.md` 行、goal 长文骨架）；
- 需要**架构或技术选型**的对比与推荐；
- **大改动前的风险评估**（改动面清单、回滚点、失败模式）。

**不适用**：
- **简单单步修改**（改一行、跑一个测试）——委派的开销大于收益；
- **已冻结的科学判据**（阈值、判据、DONE 定义）——这些必须在本仓既有纪律下写定并
  实验前冻结，**不外委**（见 `docs/decisions/0012-planner-delegation-scope.md`）；
- 需要读 `outputs/` 产物才能判断的事 —— planner 看不到产物，只能看到代码。

## 前置条件

1. **确认真的需要 planner**：任务是否多步、是否有设计取舍。单步修改直接自己做。
2. **已完成只读侦察**：planner 只读代码（Read/Grep/Glob），**读不到 `outputs/` 下的产物**。
   所以必须由你把现状写进委派提示词：当前 SHA、相关 issue 状态、既有证据、
   硬约束、预算上限、以及**已冻结的判据放在哪个文件**。
3. **确认 planner 可用**：额度耗尽或不可达时按检查点的降级路径处理，不阻塞工作。

## 步骤

1. **自查是否委派。** 单步或已有明确方案 → 自己做，不委派。
2. **委派，并交足上下文。** 提示词里写明：任务范围、必须遵守的硬约束（不得改
   `data/raw|interim|processed` 与归档、不合并/release/force push）、当前状态与 SHA、
   预算、以及"科学判据在哪个文件里"（让它**引用**而不是**定义**）。
3. **接收 JSON。** planner 只回一份 JSON，不写任何文件。把它存成工作态临时文件
   （如 `/tmp/plan.json`）——它是**草稿，不是证据**。
4. **过校验器**（不可跳过）：
   ```bash
   python tools/check_planner_plan.py --plan /tmp/plan.json
   ```
   必须 `"verified": true`、退出码 0。有失败就**退回重委派或自己修**，
   不得绕过校验直接落地。
   - **围栏容忍**：实测 planner 会在自己的提示词明令"不要围栏"的情况下仍输出
     ```` ```json ```` 包裹。校验器**恰好容忍一个包裹围栏**（并在报告里标
     `"fence_stripped": true`），因为这是传输格式而非内容缺陷；但**围栏外有散文、
     两个围栏、或围栏未闭合仍然拒绝**——放宽那几种会真正削弱"回复即对象"的契约。
5. **复核内容（校验器查不出的部分）**：校验器只保证结构与边界，**不保证内容正确**。
   落地前必须自己抽查它引用的事实：`file:line` 是否真的指向所说内容、数字是否与产物
   一致、步骤顺序是否真的可执行。实测一次委派里出现过**行号差一**的引用；
   这类问题只能由你发现。反过来，若它的 `open_questions` 指出你的任务框架有误
   （实测发生过：两个月段的 (月,时) 桶数只有 8 而非 12），**以它的质疑为准去核对**。
6. **转成目标产物**（由你写，不由 planner 写）：
   - 迭代计划 → 按 R-032 整理进 `docs/plans/NNNN-<slug>.md`（编号在归档时分配）；
   - 目标骨架 → `docs/goals/<slug>.md`，objective 另行压缩到 4000 字符内（R-034）；
   - 任务行 → `docs/R7_TASK_QUEUE.md` 的表格行；
   - 跨任务决策 → 另写 `docs/decisions/NNNN-*.md`（见 `decision-record`）。
   **科学判据由你补写并说明来源**，不得直接采用 planner 给的数值。
7. **记录来源。** 在产物里注明该计划源自一次 planner 委派（工作态草稿），
   并在当轮报告中如实说明是否发生降级。

## 检查点

- **步骤 1 后**：若是单步修改，**停止**委派，直接做。
- **步骤 3 后**：`science_criteria_refs` 必须只含**文档指针**（含 `/` 且以 `.md` 结尾）。
  若 planner 给了裸数字或阈值，**拒绝该字段**并自己写。
- **步骤 4 后**：校验器 0 失败是**落地前提**。校验器是契约的唯一机械真相；
  手改 JSON 让它通过而不修内容，等于绕过门禁。若报告标了 `"fence_stripped": true`，
  说明 planner 又加了围栏——**不算违规**，但值得在报告里提一句它的提示词未被完全遵守。
- **步骤 5 后（复核内容）**：至少抽查两处 `file:line` 引用是否真的指向所说内容，
  并核对它引用的数字与产物一致。**校验器查不出行号错误或数字错误**，
  这一步是唯一防线。
- **步骤 6 后**：确认 `files_to_touch` 未含 `data/raw|interim|processed` 或归档路径
  （校验器已拦，但要复核它是否与你的实际改动一致）。
- **planner 不可用时**：**自行规划**，并在报告中写明「planner 不可用，已降级为自规划」。
  这是合法路径，不阻塞；协议与校验器仍然可用（你手写的 JSON 过同一契约）。
  实测确认：**子智能体可用性受会话约束**——工作区定义文件存在但当前会话未注册时，
  委派会直接报 `Agent type not found`，此时按本节降级，不要改成别的 agent 类型顶替。

## 常见失败

- **把 planner 的 JSON 当证据**：它是草稿。证据是运行记录、产物 digest、CI run id
  （见 `result-freeze`）。planner 输出不能作为任何结论的依据。
- **让 planner 定科学判据**：阈值、非劣容差、DONE 定义一旦被外部生成，
  「实验前冻结 + digest 校验」那套防线就失效了。planner 只**引用**判据所在文件。
- **跳过校验器直接落地**：契约的意义在于拦住畸形或越界计划；跳过它等于没有契约。
- **planner 输出与仓库现状脱节**：它看不到 `outputs/` 产物，也不读会话历史。
  现状必须由你在委派提示词里给出；否则它会写出与既有证据矛盾的步骤。
- **把一次性委派当审计用**：planner 是规划工具，不是审查者；它不能替代
  `ci-workflow-triage` 或安全检查。
- **为了"用了新工具"而委派**：单步任务委派只是多绕一圈。
- **以为校验器能保证内容正确**：它只保证结构与边界。实测一次委派里出现**行号差一**
  的引用（值对、位置错），只有人工抽查能发现。反过来它的 `open_questions` 可能比你的
  任务框架更准（实测：它指出两个月段的桶数是 8 而非 12，核对后确认它对）。

## 完成判据

- 计划 JSON 通过 `tools/check_planner_plan.py`（0 失败）；
- 已转成目标产物，且科学判据由本仓纪律写定、来源可追溯；
- 产物中注明来源为 planner 委派（工作态草稿）；
- 若发生降级（planner 不可用 / 已自行规划），在报告中如实写明。

## 明确不覆盖

- 不替代 `decision-record`（决策记录仍需按 R-033 写）、`result-freeze`（结果定稿）、
  `issue-lifecycle`（issue 推进与关闭）；
- **不产生证据**：planner 输出既不是运行记录也不是产物；
- **不做科学判定**：判据、阈值与结论的写定权在本仓纪律，不在 planner；
- 不覆盖机械门禁本身的执行（那是 `check_conventions.py` 与 hooks）；
- planner 不可用时**不阻塞**任何工作。

## 规范示例（契约的可执行样例）

`tools/check_planner_plan.py` 的测试会断言下面这份示例**能通过校验器**、
且其 key 集合恰好等于契约（必需 + 可选）——所以示例与契约不会各说各话。
字段含义：`steps` 必须按依赖顺序排列（`depends_on` 只能引用**更早**的步骤，
因此环与前向引用在结构上无法表达）；`science_criteria_refs` 是**文档指针**；
`files_to_touch` 是仓库相对路径且不得命中只读前缀。

```json
{
  "slug": "example-iteration-plan",
  "objective": "One single-line objective, at most 400 characters, stating what this iteration delivers.",
  "steps": [
    {
      "id": "recon",
      "action": "Read the current state and record the starting SHA.",
      "verify": "git rev-parse HEAD output recorded in the report",
      "depends_on": [],
      "files": []
    },
    {
      "id": "implement",
      "action": "Make the change described by the objective.",
      "verify": "targeted test passes",
      "depends_on": ["recon"],
      "files": ["tools/check_planner_plan.py"]
    }
  ],
  "verification": [
    "python -m pytest -q",
    "python tools/check_conventions.py"
  ],
  "risks": [
    {
      "risk": "the documented contract drifts away from the enforced one",
      "mitigation": "the validator test asserts the embedded example passes"
    }
  ],
  "not_doing": [
    "no scientific criteria are set or changed by this plan"
  ],
  "stop_conditions": [
    "the frozen budget or the declared scope is exceeded"
  ],
  "science_criteria_refs": [
    "docs/R7_B1_BASELINE_AUDIT.md"
  ],
  "files_to_touch": [
    "tools/check_planner_plan.py"
  ],
  "open_questions": [
    "anything the planner could not decide from code alone goes here as a question"
  ],
  "budget_notes": "informational only; the budget ceiling is set by the repository, not by the planner"
}
```
