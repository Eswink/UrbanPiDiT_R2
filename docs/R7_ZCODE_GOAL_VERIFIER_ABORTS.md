# ZCode goal 完成校验的两种失败形态：会话被中止与调用长时间不返回

**状态：两种形态都定位到具体调用与判据；成因在客户端（ZCode runtime），不在本仓库。**
本文件只留痕、给规避与恢复，不改动任何实验或规则判据。
形态一（异常终止）调查于 2026-09-28；形态二（长时间不返回）2026-09-30 定位。
文件名沿用当时的窄标题——它是 skill 与其它文档的引用锚点，不改名；内容已扩为两种形态。

## 0. 一览

| 形态 | 用户看到的现象 | 本地可核的判据 | 落库终态 |
| --- | --- | --- | --- |
| ① 校验调用被 provider 拒绝（非 JSON 响应） | 会话在「目标未完成、要求继续」时**异常终止**，随后被运行时恢复 | rollout 末行 `AI_APICallError: Invalid JSON response`，`querySource:"target_completion_verification"` | `session_entry.payload.status:"failed_closed"` |
| ② 校验调用长时间不返回 | UI 停在「第 N 次迭代 · 目标校验中」；该会话新输入被排队；**只有用户暂停/停止才结束** | `session_entry` 最新一条长时间停在 `status:"started"`；`model_usage` 行 `status='cancelled'`、`cancelled_by_user=1`、时长 20+ 分钟 | `status:"cancelled"` + `pauseActiveTargetForCancellation` |

两条共同点：都发生在**回合结束之后**的那一次独立模型调用上（非流式、`tools: []`、携带整段对话），
都不是主回合、不是工具、也不是本仓产物。**不要把它们当判据失败**——目标的完成判定与它们无关。

---

## 1. 形态一：校验调用被 provider 判为非 JSON 而中止（2026-09-28 定位）

### 1.1 现象

使用 goal 模式跑长任务时，会话会在「目标显示未完成、需要继续」的时刻**异常终止**，
随后运行时把 goal 状态从存储里恢复（表现为会话被重启、后台进程被杀、上下文被摘要）。
`docs/goals/v2-round-two-attribution.md`（第二轮）与
`docs/goals/v2-round-three-m1-attribution.md`（第三轮）两次长时间 goal 会话都出现过。

### 1.2 证据（全部来自本机客户端日志，可复核）

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

**2026-09-30 更新（来自形态二的取证）**：把 `model_usage` 里全部 26 次校验调用按 provider 分组后
（见 §2.4），同一模型名 `deepseek-v4.1-flash` 在另一条路由（`new-provider`）上 **14/14 成功**，
其中包含 **468,442 / 510,730 / 532,730 输入 token** 的请求。因此「请求体积越限」只能作为
**该路由**（`new-provider-4` = `https://api.cline.bot/api/v1`）的解释，不能当作普遍规律；
形态一的四条 `error` 行（17.1 / 23.8 / 44.1 / 46.9 秒，`input_tokens=0`）就是这一族失败。

### 1.3 结论（事实与推断分开写）

**事实**：会话的最后一次模型调用是 goal 完成校验器；它携带**整段对话**（516 条消息、1.32 MB），
而不是普通回合使用的 tail 窗口；该调用以 provider 返回**非 JSON 响应**告终，之后会话即终止。

**推断**（未取得 provider 的原始响应正文，故标为推断）：请求体量越过了该路由的实际上限
（512K 配置窗口按 3–4 字节/token 估算约合 330–440K token，加上输出预留与网关侧请求体上限即可能被拒），
上游返回错误页/截断体，SDK 无法解析为 JSON → `AI_APICallError`；SDK 的重试没有改变结果。
**该推断只对 `new-provider-4` 这条路由成立**（同族请求在 `new-provider` 上按 §2.4 成功）。

**推断的修复位置**（不在本仓库）：完成校验器应当像普通回合一样只发 **tail 窗口**或摘要，
而不是整段对话；日志里同时存在 tail 表示说明客户端**已有**裁剪机制，只是这条路没用上。

---

## 2. 形态二：校验调用长时间不返回，客户端没有超时（2026-09-30 定位）

### 2.1 现象

目标 `target_mumqbvg1_7e64da2f-2885-486e-ad5c-2901aa0f0442`（会话
`sess_95da7dd5-ed88-473e-bf28-f4c1f15e97b4`，goal 长文 `docs/goals/main-model-v2-rw-b-round.md`）
第 1 轮的主回合在本地 `2026-09-30 00:02:03` 正常完成，随后进入迭代 1 的完成校验。校验调用
**悬挂 1773 秒（29.55 分钟）后由用户暂停中止**：UI 全程停在「第 1 次迭代 · 目标校验中」，
该会话的新输入被排队（`goalVerifying → enqueue`，reasonCode `goalVerifierAcceptsFutureInput`），
没有完成、没有报错、没有任何超时事件。

### 2.2 时间线（本地时间 = UTC+8；日志文件按**本地**日期命名，故 00:02 的条目在 `zcode-2026-09-30.jsonl`）

| 本地时刻 | 事件 | 来源 |
| --- | --- | --- |
| 00:02:03.164 | 校验起始：`session_entry 7d7ce89b-999d-41f0-8c8c-80739fcec8b6`，`status:"started"`、`goalIteration:1`、`verificationId e6c36daf-e5a8-4a`（seq 91161） | db `session_entry` |
| 00:02:03.231 / .233 | `session.event.persistence.started/completed`（`sessionEventType:"model_request"`，seq 91162） | 日志 span `e6c36daf` |
| 00:02:03.497 | `config.project_hooks.pending_trust`（该会话的项目 hooks 处于 pending workspace trust 且 blocked） | 日志 |
| 00:02:03 → 00:31:36 | **该 span 无任何日志行**（无 `model.request.started/completed/failed`，无 timeout 事件） | 日志 |
| 00:31:36.034 | `session_target.time_updated` —— 用户按暂停 | db `session_target` |
| 00:31:36.050 | 合成消息（`role=user`、`synthetic=true`、`metadata.source="goal_state_change"`、`visibility="model-only"`）：`The active session goal is paused. Do not continue pursuing it unless the user resumes or replaces the goal.` | db `message` |
| 00:31:36.174 | `model.request.failed`：`durationMs 1772914`、`status:"cancelled"`、`reason:"cancelled"`、`retryable:false`、`attempt:1`、`maxAttempts:11`、`baseURL:"https://api.cline.bot/api/v1"`、`transport:"http"` | 日志 |
| 00:31:36.198 | `model.retry.delay.resolved`：`canRetry:false`（**不重试**） | 日志 |
| 00:31:36.200 | `model_usage` 行：`status='cancelled'`、`cancelled_by_user=1`、`duration_ms=1772967`、`input_tokens=0`、`output_tokens=0` | db `model_usage` |
| 00:31:36.283 | 校验终态：`session_entry 08141ff0-19ff-4a6f-af99-bedda07d1d21`，`status:"cancelled"`、`verification.passed=false`、reason `Completion verifier request was cancelled.`（seq 91166）；transcript `msg_goal_verify_…_1` 的 `finish:"cancelled"` | db `session_entry` / `part` |
| 00:35:17 | 一对 persistence 事件（`sessionEventType:"session_title_updated"`）—— 与校验无关 | 日志 |

同一晚还有一次同族事件：`2026-09-28 13:36:16 → 14:00:12`（本地）的校验调用以
`cancelled_by_user=1`、`duration_ms 1436595`（23.9 分钟）结束——即用户此前已经手工中止过一次悬挂。

### 2.3 根因：这条调用在客户端**没有任何超时**

以下行号/符号为客户端打包产物 `~/.zcode/server/agents/glm/zcode.cjs`（2026-09-29 17:11 版本）内的定位：

- 完成校验 = 一次**非流式** `generateText`：`generateTargetCompletionVerificationText`（`Jka`，偏移 ≈12863500）
  内 `return await jy(o,()=>e.model.generateText({abortSignal:e.abortSignal,messages:e.messages,options:{maxOutputTokens:…},tools:[]}))`。
- 该路径**只有 `abortSignal`**：executor（≈4022247）把请求与 abort 信号竞速（`H4s` ≈4028635）；
  请求构造（`gWr` ≈3974142）**不传 `timeout`**，而只有传了 `timeout` 才会用 `AbortSignal.timeout()`（`J9r` ≈3821571）。
- **对照**：主流式回合有 10 分钟 stream idle 看门狗（`YV=6e5` → `modelStream.idleTimeoutMs`，
  抛 `MODEL_STREAM_IDLE_TIMEOUT`，`readNextWithStreamIdleTimeout` ≈3728961）——这条保护**不覆盖**非流式的校验调用。
- 重试也不救：`maxAttempts:11` 只对**抛错**生效；本次是用户中止（`retryable:false`），
  而「永不返回」这种情形根本不会进入重试判定。唯一的 start-plan 专用重试
  （`DIo=[1e3,2e3]`，`Vka` 只含 `account:bigmodel-start-plan` / `account:zai-start-plan`）不适用。
- 失败语义（`verifyTargetCompletion` 的 catch，偏移 ≈12862744）：`abortSignal.aborted` ⇒ 写终态
  `status:"cancelled"` 并 `pauseActiveTargetForCancellation`；否则 ⇒ 记一条告警事件
  **`target.completion_verification.failed_open`**（"Goal completion verification failed open"）
  并写终态 **`status:"failed_closed"`**，reason 为 `Completion verifier request failed: <message>`。
  **这两个名字是反的**（事件叫 open、落库叫 closed）；排查时以 `session_entry.payload.status` 为准。

结论：provider 不返回 ⇒ 调用既不完成也不报错 ⇒ `goal.status` 停在 `"verifying"`，
直到用户中止（或进程重启）。**这条路径没有自我恢复能力。**

### 2.4 决定性对照：全部 26 次校验调用按 provider 分组

来源 `~/.zcode/cli/db/db.sqlite` 的 `model_usage` 表，`query_source='target_completion_verification'` 全量：

| provider | 模型 | 行数 | 结果 | 时长 | 输入 token |
| --- | --- | --- | --- | --- | --- |
| `new-provider` | `deepseek-v4.1-flash` | 14 | completed × 14 | 11.6 – 47.4 s | 最大 **532,730** |
| `account:zai-start-plan` | `GLM-5.3-Flash` | 6 | completed × 6（1 次 `server_error` 后重试 1 次仍成功） | 25.7 – 123.8 s | 最大 303,477 |
| `new-provider-4` | `cline-pass/deepseek-v4.1-flash` | 6 | **completed × 0**：`error` × 4（17.1 / 23.8 / 44.1 / 46.9 s，`input_tokens=0`）+ `cancelled` × 2（1436.6 s / 1773.0 s，均 `cancelled_by_user=1`） | — | 0 |

时间范围（本地）：`new-provider` 09-24 20:38 → 09-28 01:18；`account:zai-start-plan` 09-25 16:23 → 09-26 07:53；
`new-provider-4` **09-28 04:35 → 09-30 00:02**。

**读法**：切到 `new-provider-4` 之后，校验调用 **6/6 未成功**（4 次快速失败 + 2 次悬挂到用户中止）；
在此之前的两条路由 **20/20 成功**，且成功的请求里包含 46.8 万–53.3 万输入 token 的大请求。
所以失败**不与请求体积相关**，而与**路由**相关（单机、单账户证据，不是普遍结论）。

### 2.5 已排除（都不是本次卡住的原因）

- **objective 与判据**：`docs/goals/main-model-v2-rw-b-round.md` §0 objective 实测 **990 码点**（上限 4000），
  `tools/check_goal_brief.py` 0 失败（2026-09-30 复跑）；goal 长文已提交（`c035877`），工作区干净。
- **本仓 hooks**：该会话日志显示项目 hooks **处于 pending workspace trust 且 blocked**
  （`config.project_hooks.pending_trust`），即这条 goal 会话根本没有运行本仓 hook ⇒ 与 7 个 hook、
  Stop hook 无关。（`check_conventions.py --quiet` 本身实测 9.7 s < 20 s 超时，且只报告不阻断。）
- **工具/子智能体**：校验的 `tools: []`；若后台任务在跑，续跑会 `defer` 而不是 `verifying`。
- **「完成条件永不满足」**：校验只等一次调用返回，与 objective 措辞无关；续跑循环没有迭代上限。

### 2.6 恢复（2026-09-30 已实证）

1. **在 goal 会话里按停止或暂停**（协议里该校验是可停的：`stopTargetKind:"goalVerifier"`、`canStop:true`；
   `/goal pause` 在 `active` / `verifying` / `notSatisfied` 状态下允许）。
2. **三处判据**（本次全部核验通过）：
   - `session_target.status='paused'` 且 `active_run_started_at` 为空；
   - `session_entry` 中 `type='target_completion_verification'` 的最新一条 `payload.status` ∈
     {`cancelled`, `failed_closed`}，不再是 `started`；
   - `model_usage` 中该 `query_source` 不再有 `status='running'` 的行（本次为 `cancelled` + `cancelled_by_user=1`）。
3. 只读复核示例（不要用写连接）：
   ```bash
   sqlite3 -readonly ~/.zcode/cli/db/db.sqlite \
     "select status,active_run_started_at,time_updated from session_target where target_id='target_...';"
   sqlite3 -readonly ~/.zcode/cli/db/db.sqlite \
     "select id,sequenceNumber,json_extract(payload,'\$.status') s from session_entry \
      where type='target_completion_verification' order by time_created desc limit 3;"
   ```
4. **不要做的事**：不要直接写 sqlite 伪造终态（状态机会不一致）；不要以为 `/goal resume` 能绕过——
   它会再次进入校验，同一条路由上会再次失败/悬挂。
5. **暂停之后目标不会自动续跑**：下一轮要么按目标长文 §8 手工推进（goal-loop 技能的手工循环），
   要么显式 resume（先按 §3 换路由）。

### 2.7 未做与未知

- **本次请求的体积未知**：同会话的 rollout 已被 `modelIoFullRetentionEnabled:false` 的保留策略回收；
  且 `model.request.started` 对这条路径**本来就不打点**（与 09-27/09-28 至少 6 次校验调用做同口径
  比对确认：它们的起始→结束窗口内该文件同样 0 行），因此不能用「没有 started 行」推断「请求没发出」。
- 未取得 provider 侧响应正文 ⇒「网关排队 / 连接悬挂 / 上游限流」三者无法区分。
- 未向上游提交缺陷报告（本文件即报告草稿的证据部分）。

---

## 3. 规避（本仓库可执行）

1. **goal 轮次不要用 `new-provider-4`（cline-pass 路由）作为会话模型**——校验跟随会话的模型选择，
   该路由上 6/6 失败（§2.4）。开工前先看一眼这张表，或按同一 `query_source` 复核最近一条是否 `completed`。
2. **一个 goal 会话只跑一轮**，轮次结束就把结论落进证据文档与目标源的 §8 进度块；
   goal-loop 技能的「四态状态机 + 进度块」正是为此。
3. **让会话保持有界**：单条命令的输出尽量短（日志里 154 KB 是正常回合的量级，1.32 MB 是校验器的量级）；
   长输出写进文件再引用摘要，而不是整段贴回对话。
4. **进度块先行**：`docs/goals/*.md` 的 §8 每次推进都更新，恢复后的新会话据此续跑，
   不需要重新读一遍全部历史（第一次恢复就是这个路径奏效）。
5. **回合内先把结论写完再结束**：校验在回合结束之后才发生，且它可能失败或悬挂；
   不要把「等自动校验」当作收尾动作。
6. **识别签名**：见 §0 的判据列。形态②下不要反复发消息（会被排队），直接按 §2.6 处理。
7. **不要依赖「自动结项」**：校验器失败/取消时目标状态不会被判定，恢复后仍需人工/显式判定结项。

## 4. 本地客户端超时补丁（2026-09-30；仓库外，不进版本控制）

**位置**：`~/.zcode/server/agents/glm/zcode.cjs`（打包产物，客户端更新会被覆盖）。
**备份**：`~/.zcode/backups/zcode.cjs.20260930-before-local-patch`。
**改动**：仅一处，把 `generateTargetCompletionVerificationText`（`Jka`）里的校验调用包成与
10 分钟定时器的 `Promise.race`，使「永不返回」变成一个普通错误，从而走 §2.3 的既有失败路径
（写终态 `failed_closed`、UI 离开「校验中」、目标保持 active）。

| 项 | 值 |
| --- | --- |
| 改前 sha256 | `fad4c35c4c36ec210d8a06d3fa0e77de23c8545e2eb6ff90aea1eb38d1e6275f`（14,820,968 B） |
| 改后 sha256 | `bbf12e80507037730b756e29314512bc65f2fd4c6cc8c5dd8e9ee4707d44653e`（14,821,206 B） |
| 改动范围 | 唯一一处：起始字节 12863802，前后 1,957,013 字节逐字节不变（+238 B） |
| 超时值 | `6e5` ms = 10 分钟（与主流式回合的 idle 看门狗同量级；历史正常校验 11.6–123.8 s） |

**覆盖范围**：只覆盖 goal 完成校验这一条调用；主流式回合的 10 分钟 idle 看门狗不受影响。
**副作用（如实）**：定时器到点时**不会**取消那条仍在途的请求（补丁不持有该请求的 controller），
只是让上层不再等它；该请求在后台自行结束，其迟到结果被 `Promise.race` 忽略。定时器已 `unref()`，
不会拖住进程退出。
**生效状态（2026-09-30 01:12 起）**：用户要求重启；重启取最小的一档——只给两个 `zcode-cli` 运行时
（pid 233916 = 本项目、974055 = 插件工作区）发 SIGTERM，**不动** `zcode-server` 与 SSH 连接。
服务在 6 秒后拉起新运行时（pid 1123056，01:12:08 起）；脚本每 10 秒记一次
`~/.zcode/backups/restart-runtime-20260930.log`，全程显示 bundle 的 sha256 仍是 `bbf12e80…`
（**桌面端重连没有重装运行时、没有冲掉补丁**），目标状态未受影响（仍为 `paused`）。

**已做的验证**：

1. 语法与唯一性：`node --check` 通过；改动前该源码串在产物中只出现 1 次，改动后旧式调用 0 次、
   race 包装 1 次、标记串 1 次；字节级 diff 只有一处（起始字节 12863802，+238 B，其余 1,957,013 B 逐字节不变）。
2. **补丁语义实跑**（`~/.zcode/backups/verify-verifier-race.mjs`：从 bundle **逐字提取** race 表达式，
   放进同构最小上下文、stub 掉 `jy` 与 `generateText`）：

   | 情形 | 结果 |
   | --- | --- |
   | 正常返回（50 ms 后返回 `{text:"ok"}`） | 50 ms 原样通过 —— 不回归 |
   | provider 抛错 | 0 ms **原错误透传** —— 没有被吃掉、也没被换成超时 |
   | 永不返回（快速模式：timer 常量 300 ms） | 300 ms 本地超时拒绝，消息带 `LOCAL PATCH …` |
   | 永不返回（**原文**，600000 ms） | 600.0 s 本地超时拒绝，消息带 `LOCAL PATCH …`（见计划 0005 的实际结果） |
3. 集成层现状：**补丁正被真实运行时加载**——01:12:08 之后本机全部会话（含这次排查会话）都由
   pid 1123056 提供，主回合、工具、子智能体路径均正常。

**未做的验证**：真实 goal 轮末的端到端冒烟（要在 UI 里建一个临时目标跑完一轮，观察校验是否按时
返回、或超时后是否 fail-open）。无头模式在这台机器上走不通：`--prompt` 的默认 provider 是
`api.b.ai`（`glm-5.3-flash`），本机连不上（`AI_APICallError: Cannot connect to API` / `ETIMEDOUT`），
连 `/model` 这种不发请求的命令也 45 秒无输出——那是这条链路自己的问题，与补丁无关。冒烟步骤见 §5。

**客户端更新后的重放**（人工，约 1 分钟）：

1. 备份新产物；用唯一串定位：
   `grep -c 'e\.model\.generateText({abortSignal:e\.abortSignal,messages:e\.messages' <产物>` 期望 `1`；
2. 把 `return await jy(o,()=>e.model.generateText({...}))`（完整串见下方"替换前"）替换为
   `return await Promise.race([jy(o,()=>e.model.generateText({...})),new Promise((_,_zr)=>{let _zt=setTimeout(()=>_zr(new Error("LOCAL PATCH …")),6e5);_zt.unref&&_zt.unref()})])`
   （`{...}` 处保持原样：`abortSignal:e.abortSignal,messages:e.messages,options:{maxOutputTokens:e.model.optionSpecs.maxOutputTokens.max},tools:[]`）；
3. `node --check <产物>`；确认字节差只有一处；
4. 重启服务进程后按计划 0005 的冒烟步骤验证。

## 5. 端到端冒烟步骤（补丁生效后待做）

**正常路径**（验证补丁不回归，并顺带验证目标能正常结项）：

1. 新开一个**临时会话**（别用正在跑实验的工作区），会话模型选**历史上校验能正常结束的路由**
   （如 `new-provider` / `account:zai-start-plan`，见 §2.4）；
2. 发一个**一轮就能完成**的小目标，例如
   `/goal 把 hello 写入 ./smoke.txt 并在回复里给出文件内容。交付物：D1 该文件；判据：回复中出现 hello。不要自行宣布目标完成。`
3. 观察：一轮结束后应进入「目标校验中」，并在 ~30 秒内结束（历史正常量级 11.6–123.8 s）；
   结束后按 §2.6 的判据看终态是 `completed` / `failed_closed` / `cancelled`；
4. 收尾：`/goal clear`（或删掉那个临时会话）。

**超时路径**（验证补丁确实把"无限挂起"变成 fail-open）：

1. 备份 `~/.zcode/v2/provider_config.json`，把某个 provider 的 `baseUrl` 临时指向本机一个
   **只监听、不响应**的端口（例如 `nc -l 127.0.0.1 8799`）；
2. 用那个 provider 当会话模型，重复上面的小目标；
3. 期望：进入「目标校验中」后**最多 10 分钟**离开，`session_entry` 最新校验条为 `failed_closed`、
   reason 里带 `LOCAL PATCH …`，目标保持 `active`（不被暂停）；日志出现
   `target.completion_verification.failed_open`；
4. 验证完**还原** `provider_config.json` 并核对 sha256。

两条都做完，把结果记进 `docs/plans/0005-zcode-goal-verifier-hang.md` 的「实际结果」。

## 6. 未做的事

- 未向上游提交缺陷报告（需要用户决定；本文件即报告草稿的证据部分）。
- 未尝试通过环境变量/设置关闭校验器：`~/.zcode/v2/setting.json` 中未见相关开关。
- 未验证「重启服务进程会把悬挂调用收敛成什么终态」（本次由用户暂停收敛，属已知路径）。
