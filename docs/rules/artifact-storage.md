# 成品存放

范围：代理与人在工作中产生的记录性产物 —— 计划、决策、目标长文、工具运行态。

**核心判据（单一问句）**：一个新克隆是否需要它来**理解、复现或审计**这项工作？
需要 → 提交；不需要 → 忽略。本页所有规则都是这条判据的具体化。

外部惯例来源（供核对，非依据）：Nygard ADR（2011）、MADR、GitHub spec-kit、
GitHub `Global/Agents.gitignore`、DVC（`dvc.yaml`/`dvc.lock`）、`targets`（`_targets/meta/meta`）。
**依据仍然只取本仓库实测**，见各条「依据」栏。

## R-032 计划分两层存放

- **级别**：必须
- **范围**：所有进入计划模式完成的任务
- **陈述**：计划的工作态留在 `.zcode/plans/`（**不提交**，文件名由客户端按会话 ID 生成）；
  任务完成后把定稿整理进 `docs/plans/NNNN-<kebab-slug>.md`（**提交**），并附「实际结果」一节。
- **依据**：E-160（客户端写入路径与命名规则）、E-162（`.zcode/*` 被忽略的实测）
- **现状**：B 类 —— 本约定确立前只有工作态、无归档；已有 1 份归档（`docs/plans/0001-*`）
- **执行方式**：人工按流程（`.agents/skills/issue-lifecycle` 的关闭检查点指向本规则）；
  文件名格式由 `check_conventions.py --rule R-033` 的同族检查顺带覆盖
- **例外**：无
- **引入日期**：2026-09-24
- **复核触发**：当客户端改变计划的存放位置或命名时

### 为什么不直接放行 `.zcode/plans/`

文件名带会话 ID（`plan-sess_800d8127-...md`），可读性差，且会把半成品计划一并带进历史。
两层结构让「工作态」与「定稿态」各自干净；归档时换可读名，并补上计划与实际的差异。

## R-033 决策一个文件一编号

- **级别**：必须
- **范围**：一切影响架构、约定或长期行为的**已接受**决定
- **陈述**：每个已接受的决策必须在 `docs/decisions/NNNN-<kebab-slug>.md` 有一份记录，
  含 `Context`、`Decision`、`Consequences` 三段与合法 `Status`
  （`proposed`/`accepted`/`deprecated`/`superseded`）；`Consequences` **必须含负面后果**。
  编号四位、单调递增、**不复用**（删过的号不回收）；被取代的记录**不删**。
- **依据**：E-161（决策此前散在 `CHANGELOG.md`、`OPEN_QUESTIONS.md`、`R7_PUBLIC_DATA.md`、
  `RESOURCE_BUDGET.md` 四处，无统一形式与取代链）
- **现状**：B 类 —— 目录新建，已有 1 份（`0001-artifact-storage-convention.md`）
- **执行方式**：脚本（`--rule R-033`，阻断）
- **例外**：无
- **引入日期**：2026-09-24
- **复核触发**：当决策数量增长到需要分类（例如按领域分子目录）时

### 与 `OPEN_QUESTIONS.md` 的分工

| 内容 | 放哪 |
| --- | --- |
| 还没定的问题（需人拍板） | `docs/rules/OPEN_QUESTIONS.md`（Q-xxx） |
| 已经定下的决定（含为什么） | `docs/decisions/`（NNNN） |

Q 解决时**写一份决策记录**，并在 Q 标题加指针。只改 Q 不算记过决策。

## R-034 goal 长文放 `docs/goals/`

- **级别**：必须
- **范围**：goal 模式（`/goal`、无头 `--target`）的任务提示词
- **陈述**：长文写 `docs/goals/<kebab-slug>.md`；objective 只放要点 + 指向该文件的路径。
- **依据**：objective 有 4000 字符硬上限，且完成判定由**不能调用工具**的独立 verifier 做
  （见 `docs/R7_GPU_BRINGUP_BRIEF.md` 的记载）—— 读不到文件的判据等于不存在
- **现状**：B 类 —— 目录新建；`docs/R7_GPU_BRINGUP_BRIEF.md` 早于本约定，保留原位
- **执行方式**：人工自觉（objective 内容无法机械判定）
- **例外**：`docs/R7_GPU_BRINGUP_BRIEF.md`（早于约定，不迁移）
- **引入日期**：2026-09-24
- **复核触发**：当客户端取消或提高 objective 长度上限时

## R-035 工具运行态不进版本控制

- **级别**：禁止
- **范围**：`.mimosa/**`、`.zcode/plans/**`、`.zcode/workflow-drafts/**`、`outputs/**`、
  `logs/**`、`*.ckpt`、`/input_*/`（CI 产物暂存）
- **陈述**：这些路径的内容不得提交。判据是 R-035 开头的单一问句，而不是逐个工具的名单 ——
  新引入的工具按其状态性质归入，不为每个工具加一条规则。
- **依据**：E-162（`.gitignore` 现有规则的实测）、E-163（`.zcode/workflows/` 放行后
  `.zcode/plans/` 仍被忽略的实测）
- **现状**：A 类（按设计满足：这些路径在版本控制中为空或仅有 `.gitkeep`）
- **执行方式**：脚本（`--rule R-035`，报告型 —— 列出被忽略但仍存在于工作树的工具态路径）
- **例外**：无。**注意 `data/manifests/` 不在本规则范围**：那里的溯源文件是刻意跟踪的
  （见 `docs/DATA_SPEC.md` 与 R-012 的注释）
- **引入日期**：2026-09-24
- **复核触发**：当某工具态被证明需要跨机器共享时（则改为提交，并说明理由）

## R-036 取代链必须可达

- **级别**：必须
- **范围**：`docs/decisions/*.md`
- **陈述**：标记为 `superseded` 的决策，其 `superseded by NNNN` 指向的目标文件必须存在；
  被指向的新决策应写明它取代了谁（双向可追溯）。
- **依据**：E-161（此前无取代链机制）
- **现状**：A 类 —— 当前无 superseded 记录，故无悬空
- **执行方式**：脚本（`--rule R-036`，阻断）
- **例外**：无
- **引入日期**：2026-09-24
- **复核触发**：当出现第一条 superseded 记录并验证链可达后

## R-037 治理层必须在版本控制内

- **级别**：必须
- **范围**：`AGENTS.md`、`docs/rules/**`、`.agents/skills/**`、`docs/skills/README.md`、
  `tools/**`、`tests/test_check_conventions.py`、`tests/test_agent_hooks.py`、`docs/plans/**`、
  `docs/decisions/**`、`docs/goals/**`
- **陈述**：这些文件必须被 git 跟踪。它们是**可执行**的治理资产：CI 会运行
  `tools/check_conventions.py`，skill 会被代理自动加载，契约会被注入每轮上下文 ——
  未跟踪意味着干净克隆上直接失效。
- **依据**：E-159（实测治理层全部未被跟踪，而 `ci.yml` 已在运行其中之一的脚本）
- **现状**：B 类 —— 本规则确立时全部未跟踪；随本规则的落地一并纳入版本控制
- **执行方式**：脚本（`--rule R-037`，阻断）
- **例外**：无
- **引入日期**：2026-09-24
- **复核触发**：当治理层新增顶层文件或目录时（应同步加入本规则范围与检查清单）

### 为什么这条必须阻断

这不是风格问题。本规则落地前，**干净克隆上 `ci.yml` 的 Check repository conventions
步骤必然失败**（脚本不在仓库里），而本地因为文件都在、感觉一切正常。
这类"只在别人机器上失败"的问题必须由门禁在本地就拦住。
