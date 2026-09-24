# 1. 成品存放约定：计划 / 决策 / 目标 / 工具态

- **日期**：2026-09-24
- **状态**：accepted
- **依据**：`docs/rules/EVIDENCE.md` E-159 – E-163

## Context（背景）

代理与人都往仓库里写东西，但此前没有任何约定说明「什么该提交、什么该忽略、放哪」。实测暴露三个问题：

1. **治理层自身没进版本控制**：`AGENTS.md`、`docs/rules/*`（13 篇）、`.agents/skills/*`（7 个）、
   `tools/check_conventions.py`、`tools/agent_hooks/*`、对应测试全部未被跟踪，
   而 `.github/workflows/ci.yml` 已经在跑 `python tools/check_conventions.py`
   —— 干净克隆上 CI 会直接失败（E-159）。
2. **计划从不落盘进仓库**：客户端把计划写到 `.zcode/plans/plan-<session-id>.md`，
   而 `.zcode/*` 被忽略，会话结束即沉在本地（E-160）。
3. **决策散在四处**：`docs/rules/CHANGELOG.md`（按规则轮次）、`docs/rules/OPEN_QUESTIONS.md`（按 Q 号）、
   `docs/R7_PUBLIC_DATA.md`（自称 decision log）、`docs/RESOURCE_BUDGET.md`（租卡决策）——
   没有「一个决策一份记录」的可检索形式，也没有 `superseded` 链（E-161）。

外部惯例（Nygard ADR、MADR、GitHub spec-kit、GitHub `Global/Agents.gitignore`、DVC、
`targets`、Cline Memory Bank）共同给出一个三层划分：**临时态忽略 / 持久记录提交 / 重负载只提交指针**。
本仓库的 R-003（派生数据不得成为唯一真相）与 R-012 已实现第三层，缺前两层的明文规则。

## Decision（决定）

**我们采用三层存放约定，并把它写成可机械检查的规则。**

1. **计划分两层**：工作态留在 `.zcode/plans/`（忽略，文件名由客户端按会话 ID 生成）；
   定稿归档到 `docs/plans/NNNN-<kebab-slug>.md`（提交），归档时附最终结果与与计划的差异。
   —— 采用 spec-kit 的 flow-forward 模型：已完成的计划视为不可变，需求变化时新建一份，
   旧目录保留供审计与对比。
2. **决策一决策一文件**：`docs/decisions/NNNN-<kebab-slug>.md`，含
   `Status`（`proposed` / `accepted` / `deprecated` / `superseded`）、`Context`、`Decision`、
   `Consequences`（**必须含负面后果**，照 Nygard 原文 "All consequences should be listed here,
   not just the positive ones"）。编号单调递增、不复用；被取代的不删，标 `superseded by NNNN`
   并双向可追溯。
   - **与 `OPEN_QUESTIONS.md` 的分工**：Q 是**待决队列**（还没定的）；decision 是**已决记录**。
     Q 解决时写一份 decision，并在 Q 标题加指针。Q 保留原始分析（沿用现有 `<details>` 惯例）。
3. **目标长文**放 `docs/goals/<kebab-slug>.md`。goal 模式的 objective 有 4000 字符硬上限，
   且完成判定由**不能调工具**的独立 verifier 做，所以 objective 只放要点 + 指向该文件的路径。
4. **工具运行态不进版本控制**，判据是单一问句：**「一个新克隆是否需要它来理解、复现或审计这项工作？」**
   需要 → 提交；不需要 → 忽略。据此 `.mimosa/`、`.zcode/plans/`、`outputs/`、`logs/` 忽略。
5. **治理层必须在版本控制内**（`AGENTS.md`、`docs/rules/`、`.agents/skills/`、`tools/`、其测试）。
   这条直接防止第 1 个问题复发。
6. **不放行 `.zcode/workflows/` 到 `docs/`**：客户端按固定路径 `.zcode/workflows/<name>.dwf.ts`
   发现按名可运行的 agent workflow，放到别处会变成「文件存在但不可用」。
   故在 `.gitignore` 中显式放行该子目录。
7. **不建 `docs/memory/`**：ZCode/Claude 的 memory 按设计存放在
   `~/.zcode/cli/memories/projects/<id>/memory/`，**机器本地、不跨机器共享**；
   而仓库内的 `AGENTS.md`（常驻契约）+ `docs/rules/`（细则）+ `.agents/skills/`（流程）
   已经实现了外部 memory 方案想达到的「小索引常驻 + 细节按需加载」，且分层更干净。
   再建一处会制造第二处真相。

## Consequences（后果）

**变容易的：**

- 新克隆上 CI 能通过：它依赖的文件确实在仓库内。
- 决策可检索、编号稳定、取代关系可追溯；不必再判断「这个决定记在哪」。
- 计划不再是会话私产，可被后来者读到「当时怎么想的、实际结果如何」。
- 新增了 5 条可机械检查的规则（R-032 – R-037，其中 4 条阻断），漂移会被当场抓住。

**变得更难 / 代价（如实列出）：**

- **多了一步人工归档**：计划定稿要从 `.zcode/plans/` 整理成可读名放进 `docs/plans/`。
  这一步没有工具强制，可能被跳过 —— 缓解：`issue-lifecycle` 技能里写明关闭前归档。
- **`docs/decisions/` 有变成装饰的风险**：只建目录不写内容等于没有。
  缓解：本文件就是第一份，目录立刻有真实内容；且 `decision-record` 技能定义了触发条件。
- **`.zcode/workflows/` 放行后，该目录下的任何文件都会进仓库**（包括草稿）。
  客户端另有 `.zcode/workflow-drafts/`（机器所有、保持忽略）用于草稿，两者是不同的路径。
- **R-037 在治理层未提交时会让门禁变红**。这是**刻意**的：它正是要抓的那个漏。
- 新增 3 个顶层目录，`docs/rules/repository-layout.md` 的目录表需同步。

## 备选方案与否决理由

- **放行 `.zcode/plans/` 直接提交**：否决。文件名带会话 ID（`plan-sess_800d8127-...md`），
  可读性差且会累积半成品计划；两层结构让「工作态」与「定稿态」各自干净。
- **建 `docs/workflows/` 而不放行 `.zcode/workflows/`**：否决。客户端按固定路径发现,
  放平级目录会得到「存在但按名调不到」的坏状态。**已实测**：`.zcode/*` +
  `!.zcode/workflows/` 确实能放行子目录（用隔离临时仓库验证 `check-ignore` 与 `git add` 行为一致）。
- **建 `docs/memory/` 作为共享记忆**：否决，理由见 Decision 第 7 条。
- **批量搬迁历史决策到 `docs/decisions/`**：否决。会制造又一处副本且无法与原文同步；
  历史决策由开放问题机制在解决时自然补入。
