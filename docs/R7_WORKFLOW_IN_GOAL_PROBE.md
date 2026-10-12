# 只读探针：在 goal 会话里发起 dynamic workflow（实测记录）

**状态：已完成一次实测（2026-10-11）。** 本文件是一次**机制探针**的观测记录，不是科研轮、
不是证据登记、不产生任何科学结论。**全程 0 GPU-h、无下载、未改科学合同/冻结证据/账本/证据索引、
未跑 campaign 轮、未 commit/push。**

- 目标长文：`docs/goals/workflow-in-goal-probe.md`（objective 见其 §0）
- 探针脚本：`.zcode/workflow-drafts/r7-workflow-in-goal-probe.dwf.ts`（git-ignored）
- 观测会话：`sess_e24eacc9-2e3b-4f77-81e1-926265bb8101`（分支 `r7/weather-reasoning`）
- 起点 HEAD（探针内读到）：`6c380ab`；账本 49 行 / 48.2867 GPU-h；证据索引 70 条

---

## 1. 结论（可行 / 不可行 + 边界）

**结论：在 goal 会话里发起 dynamic workflow 这条链路，本次实测端到端可行。**

六环全部走通且各留读数：① 技能加载 → ② 编译 → ③ 相位图 → ④ 子代理调度 →
⑤ 确定性闸门（三个只读校验器实跑）→ ⑥ 卡片发布与返回值。一次运行即完成，**无需
`ResumeWorkflowRun`、无需 `args` 重起**。

**已知边界（本探针观测到或明确未覆盖）：**

1. **轮末 goal 完成校验没有触发**（2026-10-11 全库无任何 `target_completion_verification` 记录）。
   最可能的原因（**推断，未验证**）：回合结束时后台 workflow 仍在运行，goal 续跑被
   `defer` 而非进入 `verifying`（`docs/R7_ZCODE_GOAL_VERIFIER_ABORTS.md` §2.5 已记此行为）。
   因此本探针**没有**产生「校验器读数」——它没跑，不是跑失败。
2. 探针**不覆盖**：GPU 轮编排、freeze/stage、commit/push、campaign 四件套登记（脚本
   `notCovered` 亦如实写明）。这些在本次运行里一次都没有执行。
3. 卡片「可见」是从运行 API（完成通知 + `GetWorkflowRun` 的 artifacts 列表）观测到的，
   **不是**从 UI 亲眼看到渲染结果；UI 层是否可点开未核。
4. 本探针只验证「goal 会话能发起并收尾一次 workflow」，**不**验证无人值守多轮 campaign
   驱动、也不把 `report()`/卡片当科学结论。

---

## 2. run id 与终态

| 项 | 值 |
| --- | --- |
| run id | `dwfrun-8b989f84-1043-408e-bd07-4690677135fc` |
| 标签（name） | `goal 会话内发起 workflow 的只读探针` |
| args | `{ "label": "goal-probe-1" }`（已生效：出现在交付物标题与 conclusion 里） |
| 终态 | **`completed`**（非 skip / cancelled / queued / errored / stopped） |
| 用时 | **1 分 07 秒**（4 个相位全部 done，9 个 step 全部 settled，0 失败） |
| token | 50,743（`状态读者` 16,335 + `独立复核者` 34,408 + 其余） |
| 发起会话 | 本 goal 会话（`owned_by_this_session: true`） |

**判据核对（对齐 `docs/R7_ZCODE_GOAL_VERIFIER_ABORTS.md` 的终态口径）**：终态是 `completed`，
不是被取消/排队/未完成 ⇒ D1 判据满足。

---

## 3. 时序（UTC；本机时钟与运行记录一致，见 §6 的例外）

| 时刻 | 事件 | 来源 |
| --- | --- | --- |
| 09:28:17 | goal target 创建（`target_mv3meosq_047ff681-…`，status `active`） | db `session_target` |
| 09:28:38 | 发起前 Bash `date -u` 读数 | Bash |
| **09:34:31** | **workflow run `created_at`** | run record |
| 09:34:31 → 09:34:48 | 运行期间**不轮询**，改读两份参考文档 | Read |
| 09:35:07 | 一次非阻塞快照 `GetWorkflowRun`（phase 2/4，3 step settled） | run record |
| **09:35:22** | **本回合正常结束（yield）**——此刻 workflow 仍在运行 | Bash `date` |
| **09:35:39** | **workflow 运行结束（completed）** | run record `updated_at` |
| 09:35:39 → 09:35:46 | 完成通知送达 → **新回合开始** | db `session_target.active_run_started_at` |
| 09:35:53 | 完成后 Bash `date -u` 读数 | Bash |

> §6 记录的 `created_at` 与发起前读数之间约 5m53s 的间隔**未能解释**，本文件只留痕不臆断。

---

## 4. 五项观测（objective 第 ④ 步逐条）

### ④.1 goal 轮是否在运行期间正常结束 —— **是**

后台 workflow 于 09:34:31 起运行，本回合于 **09:35:22 正常结束**（产出终稿文本、无异常、
会话未被打断、未被摘要、无 `AI_APICallError`）。此后 workflow 又继续跑了约 17s 才结束。
即：**goal 回合不必等后台 workflow 结束即可正常收尾**。

### ④.2 完成通知是否回来并触发新回合 —— **是**

workflow 于 09:35:39 结束后，完成通知**自动送达**（`<task-notification>`，`status: completed`，
携带完整 `result` JSON、1 条 `report` 项、2 个 artifact），并**触发了一个新回合**
（`session_target.active_run_started_at` = 09:35:46，即通知后约 7s）。
**未使用 `TaskOutput` 阻塞轮询**；唯一一次运行中查询是非阻塞的 `GetWorkflowRun` 快照。

### ④.3 轮末 goal 完成校验是否触发及其结果 —— **未触发（因此无结果读数）**

- 本会话 rollout `~/.zcode/cli/rollout/model-io-sess_e24eacc9-….jsonl`（15 行）里
  `querySource` 分布为 **14 × `main_turn` + 1 × `session_title`，`target_completion_verification` = 0**。
- db 侧同样为空：`session_entry where type='target_completion_verification'` 对本会话 **0 行**；
  `model_usage where query_source='target_completion_verification'` 对本会话 **0 行**。
- 全库最近一条校验记录是 **2026-10-10 19:54:46**（`failed_closed`），**2026-10-11 当天一次都没有**。

**读法**：这不是「校验失败」，而是**校验没发生**。最可能是回合结束时后台任务仍在跑，
goal 续跑走了 `defer`（`docs/R7_ZCODE_GOAL_VERIFIER_ABORTS.md` §2.5：「若后台任务在跑，
续跑会 `defer` 而不是 `verifying`」）——**此归因是推断，未取得客户端内部事件直接佐证**。
因此本探针**没有**「校验器读数」可记；目标的 `complete` 判定也未发生（`session_target.status`
仍为 `active`）。

### ④.4 workflow 发布的卡片是否可见 —— **已发布并随通知送达；UI 渲染未核**

运行发布 **2 张卡**（`GetWorkflowRun` 与完成通知的 artifacts 列表一致）：

| 卡片 id | 类型 | 标题 | 说明 |
| --- | --- | --- | --- |
| `probe-report` | markdown（**primary**，v1，946 B，`text/markdown`） | 只读探针：goal-probe-1 | **交付物**，完成卡与运行面板以其为首 |
| `probe` | metrics（v1，1 item） | 探针各步状态 | 仪表盘：`checksPassed` / `factsMatched` / `factsTotal` |

两张卡都**在完成通知里列出、在运行记录里可查**；按协议它们已作为卡片呈现在用户面前。
**未核**：UI 里是否实际渲染/可点开（本探针只能从运行 API 侧观测）。

### ④.5 是否需要用 `ResumeWorkflowRun` 或 `args` 重起 —— **都不需要**

- 一次 `CreateWorkflow` 即 `completed`，无 provider 错误、无超时、无 `ScriptError`；
  因此**未调用** `ResumeWorkflowRun`、`AmendWorkflow` 或任何重试。
- `args={label:"goal-probe-1"}` 正确传入（出现在交付物标题与 conclusion），**无需**重起补参。

---

## 5. 运行内读数（脚本自跑，非转述）

三个只读校验器由脚本以 `world.run` **实跑**并分支于退出码：

| 校验器 | 命令 | 退出码 |
| --- | --- | --- |
| 仓库约定 | `.venv/bin/python tools/check_conventions.py --quiet` | **0** |
| campaign 状态 | `.venv/bin/python tools/check_campaign_state.py --campaign docs/goals/main-model-climatology-campaign.md --quiet` | **0** |
| 证据索引 | `.venv/bin/python tools/verify_r7_evidence_index.py --index docs/R7_EVIDENCE_INDEX.jsonl --check-brief docs/R7_CANDIDATE_BRIEF.md` | **0** |

- **通过 3/3**（`log`：`三个校验器退出码：0, 0, 0（通过 3/3）`）。
- `tools/r7_round.py state` 退出码 **0**；子代理归纳：HEAD `6c380ab`、账本 49 行、
  索引 70 条、累计 48.2867 GPU-h。
- 独立复核：`独立复核者` 从**原始 JSON** 自数四项（`ledger_rows`/`ledger_total_gpu_h`/
  `index_records`/`current_node`）与摘要比对，**4/4 一致，0 项不一致**。

> 这些读数只说明**探针自身的只读闸门跑通**，不是科学结论，也不代表 campaign 方向正确。

---

## 6. 未验证 / 未做的事（如实）

1. **未取得 goal 完成校验读数**——校验未触发（见 ④.3），「defer」归因是推断，未取得
   客户端内部事件直接佐证。**没有**任何「校验器通过/失败」的读数可报。
2. **卡片 UI 渲染未核**——只从运行 API 侧确认已发布、已随通知送达。
3. **`created_at` 与发起前读数约 5m53s 的间隔未解释**（09:28:38 → 09:34:31）。
   候选解释（**均为推断**）：`CreateWorkflow` 的运行确认窗口耗时，或后台 run 生成延迟；
   本探针未定位。**不臆断成因。**
4. **未覆盖**：GPU 轮编排、freeze/stage、commit/push、campaign 四件套登记（脚本
   `notCovered` 亦写明）。
5. **未做**：未 commit、未 push、未改任何科学合同/冻结证据/账本/证据索引、未跑 campaign 轮、
   未动另一窗口 staged 的 `scripts/study_r7_s3_reasoning_depth.py` 与
   `tests/test_r7_s3_reasoning_depth.py`。
6. **未验证**：无人值守多轮 campaign 驱动；把 `report()`/卡片当科学结论（本探针明确不作）。

---

## 7. 与目标长文的对应

| 交付物 | 证据 |
| --- | --- |
| D1 一次由 goal 会话发起的 workflow 运行 | run id `dwfrun-8b989f84-…`，终态 `completed`（§2） |
| D2 观测记录 `docs/R7_WORKFLOW_IN_GOAL_PROBE.md` | 本文件 |
| D3 可行性结论与边界 | §1（可行 + 四条边界） |
| D4 目标长文 §5 进度块 | `docs/goals/workflow-in-goal-probe.md` §5 |

**本文件不自行宣布目标完成**；最终 goal 状态由用户或运行时独立校验裁定。
