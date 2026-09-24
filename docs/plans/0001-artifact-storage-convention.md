# 计划 0001：成品存放约定（plan / decision / goal / memory / workflow）

- **日期**：2026-09-24
- **状态**：已完成
- **来源**：会话 `sess_800d8127-5991-498e-9855-2c16b724787d` 的计划模式（工作态原文件在 `.zcode/plans/`）
- **相关决策**：[`0001-artifact-storage-convention.md`](../decisions/0001-artifact-storage-convention.md)

## 一、先说三个已核实的发现（它们决定方案形状）

**1. 最关键的：整个治理层目前都没进版本控制。** 实测 `git ls-files` 全部为 0：

| 未跟踪的关键资产 | 数量 |
|---|---|
| `AGENTS.md` | 1 |
| `docs/rules/*.md` | 13 |
| `.agents/skills/*/SKILL.md` | 7 |
| `tools/check_conventions.py` + `tools/agent_hooks/*.py` | 4 |
| `tests/test_check_conventions.py` + `tests/test_agent_hooks.py` | 2 |
| `docs/skills/README.md`、`legacy_v6/` | — |

**而 `.github/workflows/ci.yml` 已经在跑 `python tools/check_conventions.py`。** 也就是说：
**干净克隆上 CI 会直接失败**，因为那个脚本根本不在仓库里。这是本次要先补的漏，
否则新建的存放约定同样不会被提交，等于白建。

**2. plan 从未进版本控制。** 客户端把它写在 `.zcode/plans/plan-<session-id>.md`
（命名规则来自客户端源码：`plan-` + 会话 ID 消毒），而 `.zcode/*` 被 gitignore
—— 会话一结束就沉在本地。

**3. memory 按设计就是机器本地的。** 它在 `~/.zcode/cli/memories/projects/<project-id>/memory/`，
**不跨机器共享**。而仓库里的 `AGENTS.md`（常驻契约）+ `docs/rules/`（13 篇细则）
+ `.agents/skills/`（7 个）**已经实现了"小索引常驻 + 细节按需加载"**—— 这正是外部 memory 方案
（Claude auto memory、Cline Memory Bank）想达到的效果，而且分层更干净。
所以**再建 `docs/memory/` 会制造第二处真相**。建议不建，改为把"什么事实该写哪"写成规则。

## 二、外部惯例对照（结论：只补真正的缺口）

| 外部惯例 | 本仓库现状 | 判断 |
|---|---|---|
| ADR（Nygard：一决策一文件，`proposed/accepted/superseded`，编号不复用） | 决策**散在四处**：`CHANGELOG.md`、`OPEN_QUESTIONS.md`、`R7_PUBLIC_DATA.md`（自称 decision log）、`RESOURCE_BUDGET.md` | **真缺口** → 建 `docs/decisions/` |
| Spec/plan（spec-kit `specs/NNN-*/`，flow-forward：旧目录保留供审计） | `docs/R7_RESEARCH_PLAN.md`、`R7_TASK_QUEUE.md`、`.zcode/plans/`（忽略） | **缺持久归档层** → 建 `docs/plans/` |
| Goal 长文（objective 有 4000 字符上限） | `docs/R7_GPU_BRINGUP_BRIEF.md` 已在做并已写明限制 | **缺命名约定** → 建 `docs/goals/` |
| Agent workflow（`.zcode/workflows/<name>.dwf.ts`） | 本机**从未创建过任何 workflow**（数据库表 0 行） | 客户端路径被忽略 → 放行该子目录 |
| Memory bank（Cline 6 文件 / Claude `MEMORY.md` + 主题文件） | 已由 AGENTS.md + docs/rules + skills 覆盖，且更干净 | **不建**，只写边界规则 |
| 临时态 vs 持久态（GitHub `Agents.gitignore`、DVC、targets） | `.gitignore` 注释已表达，但**不是编号规则**、无依据与复核触发 | **提升为规则** |

外部惯例的三层划分：**临时态忽略 / 持久记录提交 / 重负载只提交指针**。
本仓库的 R-003 与 R-012 已实现第三层，缺前两层的明文规则。

## 三、要建的东西

### A. 修复跟踪漏（最高优先，先做）

`git add` 治理层并提交。**不做这一步，后面全部白建。**

### B. 新目录（3 个）

```
docs/plans/       归档后的计划    NNNN-<kebab-slug>.md
docs/decisions/   决策记录（ADR）  NNNN-<kebab-slug>.md
docs/goals/       目标长文        <kebab-slug>.md
```

- **`docs/plans/`**：与 `.zcode/plans/` 是两层——工作态（会话名，忽略）→ 定稿归档
  （可读名，提交）。归档时附最终结果与差异说明。这是 spec-kit 的 flow-forward 模型。
- **`docs/decisions/`**：一个决策一份记录，含 `Status`/`Context`/`Decision`/`Consequences`
  （含**负面后果**）。编号单调递增不复用；被取代的**不删**，标 `superseded by NNNN` 并互相链接。
  - **与 `OPEN_QUESTIONS.md` 的分工**：Q 是**待决队列**；decision 是**已决记录**。
    Q 解决时写一份 ADR，并在 Q 标题加指针。
  - 首批只写**本轮新决策**，不批量搬迁历史决策（避免制造又一处副本）。
- **`docs/goals/`**：goal 模式的长文目标源，objective 只放要点 + 指向该文件。

### C. `.gitignore` 调整（含一个必须实测的 git 语义）

```
.zcode/*
!.zcode/config.json
!.zcode/workflows/      # 新增：agent workflow 定义要进版本控制
```
`.zcode/plans/` **保持忽略**（会话态、文件名带会话 ID）。

**已实测**（用隔离临时仓库验证 `check-ignore` 与 `git add` 行为一致）：
`.zcode/*` + `!.zcode/workflows/` 确实能放行子目录 —— 与上一轮"文件级放行需要 `dir/*`"
的教训不冲突，子目录与文件的重包含规则不同。

### D. 新增规则（`docs/rules/artifact-storage.md`，R-032–R-037）

R-032 计划两层 / R-033 决策一文件一编号 / R-034 goal 长文 / R-035 工具态不进版本控制 /
R-036 superseded 链可达 / R-037 治理层必须在版本控制内。

### E. 扩展执行面（`tools/check_conventions.py`）

R-037（阻断）、R-033（阻断）、R-036（阻断）、R-035（报告），每条配反证测试。

### F. 技能

新增 `.agents/skills/decision-record/SKILL.md`；扩展 `issue-lifecycle` 指向它。

### G. 契约与文档同步

`AGENTS.md` 路由表加行 + "成品存放"小节；`docs/rules/README.md` 计数；
`CHANGELOG.md` 第六遍；`EVIDENCE.md` 追加证据并修正重复 ID。

## 四、验证

1. `git check-ignore -v` 逐条核验 `.gitignore` 改动。
2. `python tools/check_conventions.py` → 0 违规。
3. `pytest tests/test_check_conventions.py -q` → 全绿，含新规则反例测试。
4. 全量 `pytest -q` → 与基线 745 passed / 3 skipped / 0 failed 对比。
5. 编号与引用一致性：决策编号无跳号无重复、`superseded` 链可达、规则 ID 与 E-xxx 无悬空。
6. **干净克隆验证**：`git archive HEAD` 后在副本里跑门禁，确认 CI 依赖的文件在仓库内。

## 五、边界（不做）

- **不建 `docs/memory/`**（会制造第二处真相；memory 按设计机器本地）。
- **不建 `docs/workflows/`**（客户端要求固定路径，平级目录会"存在但不可用"）。
- **不批量搬迁历史决策**（会制造副本且无法与原文同步）。
- **不碰** `data/`、归档目录、`legacy_v531_full/`。
- 不改 `.zcode/plans/` 的客户端命名（那是客户端契约）。

## 六、风险与回退

- **最大风险是 git 的 `!` 语义** —— 已用隔离仓库实测确认可行。
- **`docs/decisions/` 空转风险** —— 缓解：本轮决策本身写第 1 份 ADR。
- **R-037 会立刻让门禁变红**（治理层未提交时）—— 这是**刻意**的，正是要抓的那个漏。

## 实际结果

**完成情况**：全部按计划完成，无未做项。

- ✅ 新目录 3 个（`docs/plans/`、`docs/decisions/`、`docs/goals/` + 各自 README）
- ✅ 首份决策记录 `docs/decisions/0001-artifact-storage-convention.md`（目录立刻有真实内容，未空转）
- ✅ 本文件即首份计划归档
- ✅ 新规则 6 条（R-032 – R-037），写入 `docs/rules/artifact-storage.md`
- ✅ 执行面：4 条新检查（R-032/R-033/R-035/R-036/R-037，其中 4 条阻断）
- ✅ 新技能 `.agents/skills/decision-record/` + 扩展 `issue-lifecycle`
- ✅ 契约与文档同步（AGENTS.md、rules README、repository-layout、CHANGELOG、EVIDENCE）
- ✅ 治理层纳入版本控制（41 个文件）

**与计划的差异**：

1. **`gitignore` 的 `!` 语义风险不存在**。计划把"父目录被排除后能否放行子目录"列为主要风险，
   实测**可行**（E-163）—— 用隔离临时仓库同时验证了 `check-ignore` 与真实 `git add` 行为。
   与第二轮"文件级放行需 `dir/*`"的教训不矛盾：子目录与文件的重包含规则不同。
   因此不需要备用方案（`docs/workflows/`），少走一次弯路。
2. **`R-037` 上线即报出 11 处违规**，比计划的预期更严重（计划说"治理层未跟踪"，
   实测是 5 个文件 + 6 个目录全部未跟踪）。这正是它该抓的漏。
3. **顺带发现两个既有缺陷**（计划未预见，见 E-165、E-166）：
   `EVIDENCE.md` 自身有编号复用（`E-004`、`E-009` 各被用于两个发现）；
   `repository-layout.md` 关于 `.mimosa/` 的陈述过时。两者已修正。
4. 计划里"检查脚本自测 41 个"是旧数字；实际做到 **55 个**（新增 14 个针对新规则的反证）。

**验证**（全部实跑）：

| 项目 | 结果 |
| --- | --- |
| 阻断规则 | 24 条，**0 违规**（原 20 条） |
| 检查脚本自测 | **55 passed**（原 41） |
| hook 自测 | **123 passed**（不变） |
| 全量测试 | **759 passed, 3 skipped, 0 failed**（上一遍基线 745） |
| `.gitignore` 逐条核验 | `config.json` / `.zcode/workflows/` 可提交；`plans/` / `workflow-drafts/` / `.mimosa/` 仍忽略 |
| **干净克隆验证** | 用 `git checkout-index` 导出 index → 在新 git 仓库里跑门禁：**24/24 通过** |

**遗留**：无阻塞项。两项待决仍开着（与本次无关）：
Q-011（把 `icechunk` 等 4 个未声明依赖并入 requirements）、
Q-012（确认是否有历史结论声称在 GPU 上跑过而实际走了失败路径）。
