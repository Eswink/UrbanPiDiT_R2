# 0043 自迭代连续 goal（一轮输出作为下一轮输入）

- **日期**：2026-10-09
- **状态**：accepted

## Context

用户报告连续任务会话 `sess_1d199296-0818-4484-a384-786216eabbe0`「跑了一会儿，然后自己停了」，
要求给出让它**自己一轮接一轮跑下去**（上一轮的输出作为下一轮的输入）的 goal 提示词，并能使用 ZCode
的 workflow（用户侧触发名 `/workflow`）能力。

只读查 `~/.zcode/cli/db/db.sqlite` 定位到**真实停止原因**（非推测）：该会话在给出轮次总结后，客户端
发起的 goal 完成校验（独立非流式调用，`query_source='target_completion_verification'`、`tools:[]`）
provider 报错（该 `model_usage` 行 `status='error'`、`duration_ms=31157`、0 token），harness **fail-open**：
`session_entry` 最新校验条 `json_extract(data,'$.payload.status')='failed_closed'` 却
`verification.passed=true`、reason `Completion verifier request failed: Model request failed.`，
`session_target.status='complete'`、`active_run_started_at` 为空。**科学门一条未过**（r=0、2023 test
未评分、S4 未启动）。这属 `docs/R7_ZCODE_GOAL_VERIFIER_ABORTS.md` 记录的**形态一**。

- 既有约定（决策 0024）已把「自动校验 best-effort、不依赖它结项」写成规则，`goal-loop` 技能也记了两种
  失败形态；但**没有**规定「一轮结束后如何自己进入下一轮」——实践中仍是每轮停下等人触发。
- 该会话等待的主要时间花在**阻塞式 CI 轮询**上（多次 `sleep 130–500`），不过它确实完成了一个真实轮次。
- 顺带修正 DB 文档漂移：`session_entry` 真实列为 `id/session_id/type/time_created/time_updated/data`，
  状态在 `json_extract(data,'$.payload.status')`；`docs/R7_ZCODE_GOAL_VERIFIER_ABORTS.md` 里用 `payload`
  列的示例 SQL 与库不符（该文档本轮不改，此处留痕）。

## Decision

1. **自迭代契约**：一个已授权 goal 采用「每轮交接四件套 + 立刻续跑」——每轮结束写 ① 长文 `§8` 进度块
   （含下一动作第一条命令）② 账本行 ③ 证据索引 record ④ `campaign-state` 注释；写完**立即**进入下一轮，
   **不等用户触发、不等 goal 校验**。落地于 `docs/goals/main-model-climatology-campaign.md` 新增的
   「连续自我迭代契约」节。
2. **不等 verifier**：发现目标被标 `complete` 而 `§8` 科学门未过时，一律按 `§8`「下一动作」恢复继续，
   不把 harness 的 complete 当科学完成；`goal-loop` 技能补「连续自我迭代（不等 verifier）」一节。
3. **停止条件仅三种**：触保留授权边界 / 达已冻结科学停止条件 / 确无可安全独立推进的工作；停时必须在
   `§8` 写「为何停 + 恢复第一条命令」。
4. **防空转**：同一假设连续无收益就换或停；不重复同一 test；不加 seed/剂量练到赢；CI 用后台轮询，
   不阻塞整轮。
5. **workflow 接入**：objective 要求先加载 bundled 技能 `dynamic-workflows`（`/workflow` 的落地技能；
   未加载时 `CreateWorkflow` / `AmendWorkflow` / `SaveWorkflow` / `EvalWorkflowSnippet` 直接拒绝运行），
   再按决策 0042 的七 gate 决定是否真的并行；可复用脚本存 `.zcode/workflows/<name>.dwf.ts`，一次性用
   inline。技能清单由 12 项增至 **13 项**（`dynamic-workflows` 为 ZCode bundled 技能，不属本仓
   `.agents/skills/`，故不进 `docs/skills/README.md` 能力表）。

## Consequences

- **代价/风险**：① 自迭代放大**多重比较**与「练到赢」风险——靠既有科学门（r/alpha、seed 预声明、
  冻结停止条件）与防空转条款约束，不放松任何判据；② 「不等 verifier」若被误读成可以无停止条件空转，
  会失控——故把停止条件写成封闭三条并要求写「为何停」；③ 后台 CI 轮询下，CI 结果可能晚于下一轮开工，
  须在下一轮开工前回补精确绑定。
- **不改**：科学合同、冻结证据、index/brief、账本历史行、生产模型与训练代码、用户配置与安全 hook。
- **依赖**：本决策**不加机械检查器**（自迭代是流程约定，无稳定的 AST 判据）；`check_goal_brief.py`
  只保证 objective 单段 ≤4000 字符与自指针。
- **引用**：`docs/goals/main-model-climatology-campaign.md`「连续自我迭代契约」节与 `§5` CI 纪律、
  `.agents/skills/goal-loop/SKILL.md`、计划 0019、决策 0042、`docs/R7_ZCODE_GOAL_VERIFIER_ABORTS.md`。
