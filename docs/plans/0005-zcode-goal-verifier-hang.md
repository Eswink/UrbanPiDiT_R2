# 0005 ZCode goal 完成校验挂起：成因定位、恢复与本地超时补丁

- **日期**：2026-09-30
- **状态**：仓库侧已完成；本地客户端补丁已落盘，重启生效与冒烟见「实际结果」
- **触发**：用户报告 goal 模式下 UI 一直停在「第 1 次迭代 · 目标校验中」不推进

## 计划正文（批准后修订）

### 1. 要解决的问题

goal 会话 `sess_95da7dd5-…`（目标 `target_mumqbvg1_…`，长文 `docs/goals/main-model-v2-rw-b-round.md`）
第 1 轮主回合正常结束后，迭代 1 的**完成校验调用**长时间不返回，UI 停在「目标校验中」，该会话新输入被排队。

### 2. 计划的动作

- **A1** 扩展 `docs/R7_ZCODE_GOAL_VERIFIER_ABORTS.md`：把「失败形态二（长时间不返回）」写成与形态一
  并列的一节，含时间线、代码依据、26 次校验调用的按 provider 分布、已排除项、恢复步骤与三处判据。
- **A2** 更新 `.agents/skills/goal-loop/SKILL.md`：两种形态的识别签名与恢复流程；把「开工前复核校验路线」
  写进前置条件；把「回合内先写完结论再结束回合」写成纪律。
- **A3** `docs/rules/EVIDENCE.md` 新增 E-201；新增决策 `0024`；R-034 加指针；`CHANGELOG.md` 一条。
- **A4** 本文件（R-032 归档）。
- **A5** `docs/goals/main-model-v2-rw-b-round.md`：纠正过期横幅，§8 状态改 `paused` 并记录轮末校验结果。
- **C1–C5** 本地客户端超时补丁：备份 → 定位唯一调用点 → 包一层 10 分钟 `Promise.race` →
  `node --check` + diff 复核 → 重启时机与用户确认 → 冒烟（正常路径 + 超时路径）→ 把补丁与重放步骤写回 A1 文档。

### 3. 判据

- 机械：`tools/check_conventions.py` 37 条阻断规则 0 失败；`tools/check_goal_brief.py` 0 失败；`pytest` 全绿；
  CI `R7 CPU CI` 绿。
- 事实：恢复动作有三处互相一致的终态判据；补丁改动在产物中唯一匹配、语法通过、`diff` 只有一处变化。

## 实际结果

### 完成情况

- **A1–A5 全部完成**（文件清单见提交 `git show --stat`）。**C1–C2 完成**：补丁已落盘并通过复核
  （`~/.zcode/backups/zcode.cjs.20260930-before-local-patch` 为备份；sha256 `fad4c35c…` → `bbf12e80…`；
  唯一一处改动，起始字节 12863802，+238 B，其后 1,957,013 B 逐字节不变；`node --check` 通过）。
- **C3 完成（用户要求重启）**：只给两个 `zcode-cli` 运行时（233916 / 974055）发 SIGTERM，
  不动 `zcode-server` 与 SSH 连接；服务 6 秒后拉起新运行时（pid 1123056，01:12:08），
  脚本逐 10 秒记录到 `~/.zcode/backups/restart-runtime-20260930.log`——**bundle 全程仍是补丁版**
  （桌面端重连没有重装运行时），目标仍为 `paused`。补丁**自 01:12:08 起已被真实运行时加载**
  （本机全部会话此后都由该进程提供，主回合/工具/子智能体路径正常）。
- **C4 部分完成**：补丁语义用**同构最小复现**实跑（`~/.zcode/backups/verify-verifier-race.mjs`，
  从 bundle 逐字提取 race 表达式，stub 掉 `jy`/`generateText`）：正常返回 50 ms 原样通过、
  provider 抛错 0 ms 原错误透传、永不返回在快速模式 300 ms 拒绝、**原文 600000 ms 版 600.0 s 拒绝**。
  **无头模式的端到端冒烟走不通**：`--prompt` 默认 provider 是 `api.b.ai`（`glm-5.3-flash`），
  本机连不上（`Cannot connect to API` / `ETIMEDOUT`），`/model` 也无输出——与补丁无关。
  **真实 goal 轮末的冒烟仍待做**，步骤写在缺陷文档 §5（正常路径 + 超时路径）。
- 诊断结论（已确认）：校验调用在客户端**没有超时**，只有 `abortSignal`（主流式回合的 10 分钟 idle
  看门狗不覆盖它）。全量 26 次校验调用里 `new-provider-4`（cline-pass 路由）**0/6 成功**
  （4 次快速 error、2 次悬挂后被用户中止），另两条路由 **20/20 成功**（含 53 万输入 token 的大请求）
  ⇒ 与请求体积无关、与路由相关。

### 与计划的差异

1. **缺陷文档做了重排而不是追加**：原文件通篇假定只有「会话被打断」一种形态，追加会读不通，
   故标题扩为两种形态、原内容整体下移为 §1（原文与表格逐字保留），新形态写入 §2、补丁写入 §4。
2. **「换一条 provider 路由」从「操作性规避」升级为开工检查**——原因是有硬证据（0/6 vs 20/20），
   因此把它写进技能的**前置条件**，而不是只放在规避清单里。
3. **意外发现并顺手改正**：`docs/rules/EVIDENCE.md` 的「统计」块与按行重数不符
   （写 140/21/39 与 197/1；实际 142/14/44 与 199/1），已在加 E-201 时按行改正并注明。
4. **计划里的「超时路径冒烟」需要一次服务重启**，而重启会中断当前会话；实际处置是
   **先落补丁文件**（不重启不影响任何行为），重启与冒烟留给用户选时机，而不是自作主张重启。

### 验证

| 检查 | 命令 | 结果 |
| --- | --- | --- |
| 阻断规则（37 条） | `python tools/check_conventions.py` | `blocking rules=37 failing=0`，report-only hits=0 |
| 决策编号 / 取代链 | `python tools/check_conventions.py --rule R-033 --rule R-036` | 0 失败 |
| goal 长文结构 | `.venv/bin/python tools/check_goal_brief.py --brief docs/goals/main-model-v2-rw-b-round.md` | `briefs=1 failures=0 advisories=0`，exit 0 |
| 全量测试 | `.venv/bin/python -m pytest -q` | `1484 passed, 3 skipped`（3 条 skip 是既有的「干净检出无真实数据 fixture」），154.67 s |
| 空白 | `git diff --check` | 无输出 |
| 台账统计 | 按行重数（`awk` 统计表格末两列） | 201 行；覆盖度 142/15/44、置信度 200/1/0（原 140/21/39、197/1 为漂移值，已改正） |
| 客户端补丁（仓库外） | `node --check`；字节级 diff；唯一性计数 | 语法通过；唯一一处改动（起始字节 12863802，+238 B，其余 1,957,013 B 逐字节不变）；旧式调用 0 次、race 包装 1 次 |

CI：`R7 CPU CI` run **`36601602896`**（commit `26c23b2`）= **success**（八步含 37 条阻断规则、证据索引与
候选 brief、离线编译与空白检查、全量 pytest；17 条实验 workflow 按 commit-message 标签门控 **skipped**，
是设计行为、不算失败也不算通过）。

### 遗留

- **端到端冒烟仍待做**：正常路径与超时路径的步骤写在缺陷文档 §5，都需要在 UI 里建一个临时目标
  （超时路径还要临时改 provider 并还原）。在此之前，补丁的验证强度是「语法 + 唯一性 + 字节级 diff +
  同构语义实跑 + 已被真实运行时加载」，**不是**真实 goal 轮上的端到端复现。
- 客户端补丁是仓库外的本地改动：桌面端更新会覆盖它，届时按缺陷文档 §4 的重放步骤重做；
  上游报告尚未提交（缺陷文档即报告草稿）。
- 与本次无关但同批发现：项目 hooks 自 2026-09-29 起在日志里显示
  `config.project_hooks.pending_trust`（pending workspace trust，blocked），09-28 之后未见项目 hook 运行记录——
  `AGENTS.md` 描述的 7 个 hook 当前可能并未生效，需要用户在工作区信任一次后再复核（不在本计划范围内）。
