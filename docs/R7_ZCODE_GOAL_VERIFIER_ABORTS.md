# ZCode goal 会话异常终止：成因、证据与规避

**状态：已定位到具体调用与失败签名，成因在客户端（ZCode runtime），不在本仓库。**
本文件只留痕与给规避，不改动任何实验或规则判据。调查时间 2026-09-28。

## 1. 现象

使用 goal 模式跑长任务时，会话会在「目标显示未完成、需要继续」的时刻**异常终止**，
随后运行时把 goal 状态从存储里恢复（表现为会话被重启、后台进程被杀、上下文被摘要）。
`docs/goals/v2-round-two-attribution.md`（第二轮）与
`docs/goals/v2-round-three-m1-attribution.md`（第三轮）两次长时间 goal 会话都出现过。

## 2. 证据（全部来自本机客户端日志，可复核）

会话日志：`~/.zcode/cli/rollout/model-io-sess_<session-id>.jsonl`（每行一条模型调用记录）。

| 项 | 值 |
| --- | --- |
| 第三轮会话 | `sess_14d38227-43ad-48bd-b971-185b93181b02`，日志 **232 行** |
| **最后一行**就是失败调用 | `"querySource":"target_completion_verification"`，`completedAt 2026-09-28T11:36:16.541Z`（本地 19:36:16），日志文件 mtime 同为 19:36 |
| 失败签名 | `AI_APICallError: Invalid JSON response`（栈：`postToApi` → `doGenerate` → `_retryWithExponentialBackoff` → `generateText`），`durationMs 23731` |
| 该请求的体积 | **1,320,363 字节**（一行 JSON） |
| 该请求带的消息 | `bodyMessagesKind: "full"`、`bodyMessageOffset: 0`、**516 条消息**；同一记录里另有 `messagesKind: "tail"` / `messageOffset: 454`（即 **64 条**窗口）的表示 |
| 同一会话前一条正常调用 | 154,632 字节（**该请求是它的 8.5 倍**） |
| 模型与窗口 | `cline-pass/deepseek-v4.1-flash`（`providerId: new-provider-4`），配置 `contextWindow: 512000` |
| 复现率 | 全会话**只有两个**会话用过 goal 完成校验器：本会话与 `sess_59b983b2-0c3c-447d-bee5-a844756777e4`（第二轮），**2/2 都在校验器调用上死掉**；后者日志里 `Invalid JSON response` 出现多次并以失败行结尾 |
| 另一条同时刻的失败 | 同一会话 `10:03:38Z` 的**标题生成**调用 `TimeoutError`（60.35 s 超时）——这解释了为什么会话标题是 objective 的截断文本 |

客户端 bundle 内存在 `failOpenGoalCompletionVerification`、`failedGoalCompletionVerification`、
`cloneGoalVerificationEntryForFork`、`copyGoalVerificationEntriesForFork`、
`deriveForkedGoalStatusFromCopiedVerifications` 等符号，说明「校验失败 → fail-open → 把 goal
分叉/恢复到新会话」是**设计路径**；异常终止的是**被中断的那次会话本体**。

## 3. 结论（事实与推断分开写）

**事实**：会话的最后一次模型调用是 goal 完成校验器；它携带**整段对话**（516 条消息、1.32 MB），
而不是普通回合使用的 tail 窗口；该调用以 provider 返回**非 JSON 响应**告终，之后会话即终止。

**推断**（未取得 provider 的原始响应正文，故标为推断）：请求体量越过了该路由的实际上限
（512K 配置窗口按 3–4 字节/token 估算约合 330–440K token，加上输出预留与网关侧请求体上限即可能被拒），
上游返回错误页/截断体，SDK 无法解析为 JSON → `AI_APICallError`；SDK 的重试没有改变结果。

**推断的修复位置**（不在本仓库）：完成校验器应当像普通回合一样只发 **tail 窗口**或摘要，
而不是整段对话；日志里同时存在 tail 表示说明客户端**已有**裁剪机制，只是这条路没用上。

## 4. 规避（本仓库内可执行，已生效）

1. **一个 goal 会话只跑一轮**，轮次结束就把结论落进证据文档与目标源的 §9 进度块；
   goal-loop 技能的「四态状态机 + 进度块」正是为此。
2. **让会话保持有界**：单条命令的输出尽量短（日志里 154 KB 是正常回合的量级，1.32 MB 是校验器的量级）；
   长输出写进文件再引用摘要，而不是整段贴回对话。
3. **进度块先行**：`docs/goals/*.md` 的 §9 每次推进都更新，恢复后的新会话据此续跑，
   不需要重新读一遍全部历史（第一次恢复就是这个路径奏效）。
4. **识别签名**：会话在「目标未完成」提示后立刻消失 + `~/.zcode/cli/rollout/model-io-sess_*.jsonl`
   末行是 `AI_APICallError: Invalid JSON response` ⇒ 就是这个已知问题，不是模型或仓库的错误。
5. **不要依赖「自动结项」**：校验器失败时目标状态不会被判定，恢复后仍需人工/显式判定结项。

## 5. 未做的事

- 未修改客户端（`~/.zcode/server/agents/glm/zcode.cjs` 是打包产物，且不属于本仓库的治理范围）。
- 未向上游提交缺陷报告（需要用户决定；本文件即报告草稿的证据部分）。
- 未尝试通过环境变量/设置关闭校验器：`~/.zcode/v2/setting.json` 中未见相关开关。
