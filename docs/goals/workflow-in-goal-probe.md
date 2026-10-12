# 目标：在 goal 会话里发起 dynamic workflow（只读探针）

本文件是 goal 模式的目标长文源。objective 见 §0；交付物见 §2；判据见 §3；预算与停止见 §4。

**状态：未开始（2026-10-11）。** 这是一次**机制探针**，不是科研轮：目的是验证「goal 会话 →
加载 `dynamic-workflows` → `CreateWorkflow` → 相位图 → 子代理 → 确定性闸门 → 卡片 → 返回值」
这条链路的每一环，并如实记录 goal 循环与后台 workflow 的时序。**全程只读，0 GPU-h。**

## 0. objective（可直接粘贴，单段，实测 ≤4000 字符）

> 在/data/esw/UrbanPiDiT_R2的r7/weather-reasoning做一次**只读机制探针**：验证「在 goal 会话里发起 dynamic workflow」是否可行，并把实测结果写进 docs/goals/workflow-in-goal-probe.md 的进度块与 docs/R7_WORKFLOW_IN_GOAL_PROBE.md。步骤：① 先加载 dynamic-workflows 技能（即 /workflow；未加载时 CreateWorkflow 直接拒跑，这是本探针要确认的第一环）；② 用 CreateWorkflow 以 path=.zcode/workflow-drafts/r7-workflow-in-goal-probe.dwf.ts、args={label:"goal-probe-1"} 发起一次运行，记下 run id；③ 运行期间**不要**用 TaskOutput 阻塞轮询，先去读 docs/R7_ZCODE_GOAL_VERIFIER_ABORTS.md 与 .agents/skills/goal-loop/SKILL.md，等完成通知回来再继续（这一步本身就是被测对象：goal 轮能否在后台运行期间正常结束）；④ 记录五件事：goal 轮是否在运行期间正常结束、完成通知是否回来并触发新回合、轮末 goal 完成校验是否触发及其结果、workflow 发布的卡片是否可见、以及是否需要用 ResumeWorkflowRun 或 args 重起；⑤ 写 docs/R7_WORKFLOW_IN_GOAL_PROBE.md，含 run id、终态、时序、卡片名、校验器读数与未验证项；⑥ 做完即停，不自宣目标完成。**本轮 0 GPU-h、不下载、不改科学合同/冻结证据/账本/证据索引，不跑 campaign 轮，不动另一个窗口 staged 的 scripts/study_r7_s3_reasoning_depth.py 与 tests/test_r7_s3_reasoning_depth.py，不 commit/push。** 禁止：把 workflow 的 report() 或卡片当科学结论；把 goal 被标 complete 当科研完成；反复重启同一个运行或无限重试；为了好看删改测试或断言。预算与停止：整轮墙钟 ≤30 分钟、0 GPU-h；若 workflow 在 15 分钟内未结束、或报 provider 错误、或 CreateWorkflow 拒跑，就如实记录终态与原始错误文案后停下，不要重启。结束汇报：run id 与终态、卡片、时序观测、校验器读数、未做的事与下一步。

## 1. 起点与前置

- 起点：开工时用 `git rev-parse HEAD` 记录；预期为 `0ed73ed` 或其后继（分支 `r7/weather-reasoning`）。
- 探针脚本：`.zcode/workflow-drafts/r7-workflow-in-goal-probe.dwf.ts`（git-ignored，机器所有）。
  它只调用只读命令：`tools/r7_round.py state`、`tools/check_conventions.py --quiet`、
  `tools/check_campaign_state.py --campaign … --quiet`、`tools/verify_r7_evidence_index.py --check-brief …`。
- 参考：`docs/plans/0020-campaign-driver-workflow.md`（方案与成本）、决策 0044（workflow 作执行者与
  六层打断抗性）、决策 0043（自迭代契约）、`.agents/skills/parallel-research-workflow/SKILL.md`
  的「campaign 驱动 workflow」节。

## 2. 交付物清单

| # | 交付物 | 证据形态 |
| --- | --- | --- |
| D1 | 一次由 goal 会话发起的 workflow 运行 | run id + 终态（completed / stopped / errored） |
| D2 | 观测记录 `docs/R7_WORKFLOW_IN_GOAL_PROBE.md` | run id、终态、时序、卡片名、三个校验器退出码、未验证项 |
| D3 | 可行性结论与边界 | 写进 D2 的结论节：可行 / 不可行 + 已知边界（进程退出即 interrupted、校验器看不到 workflow 结果、子代理不跑 hook） |
| D4 | 本文件的进度块 | §5 状态机：已完成 / 未做 / 下一动作 |

## 3. 判据

- D1 有真实 run id 与终态，且**不是** skip/cancelled/queued —— 与
  `docs/R7_ZCODE_GOAL_VERIFIER_ABORTS.md` 记的终态口径一致；被取消或未完成不算通过。
- D2 含三个只读校验器的**实测退出码**（不是转述），并如实写明哪几项没验证。
- D3 的结论与 D2 的观测一致，不把「编译通过」写成「跑得通」。
- 全程 0 GPU-h；未改科学合同、冻结证据、账本、证据索引；未动另一个窗口 staged 的两个文件。
- 判据只引用本文件与上述冻结文档，不新增阈值、不放宽任何已冻结判据。

## 4. 预算与停止

- **预算**：整轮墙钟 ≤30 分钟；**0 GPU-h**；不下载、不发布、不 clone。
- **停止条件（仅三种）**：(a) D1–D4 齐且 D2/D3 写完；(b) workflow 15 分钟内未结束或报 provider
  错误，或 `CreateWorkflow` 拒跑 —— 如实记录终态与原始错误文案后停；(c) 触到保留授权边界。
- 停下必须在 §5 写「为何停 + 恢复第一条命令」。**不重启**同一个运行，不无限重试。
- 本轮不 commit、不 push（除非用户另行要求）。

## 5. 进度块（状态机）

- **状态**：active（探针已跑完一轮；最终 goal 状态不由本文件宣布）。
- **已完成**：
  - D1 ✅ 一次由 goal 会话发起的 workflow 运行——run id `dwfrun-8b989f84-1043-408e-bd07-4690677135fc`，
    终态 **`completed`**，用时 1m07s，args `{label:"goal-probe-1"}` 生效。
  - D2 ✅ 观测记录 `docs/R7_WORKFLOW_IN_GOAL_PROBE.md`（run id / 终态 / 时序 / 卡片名 /
    三个校验器退出码 0,0,0 / 未验证项）。
  - D3 ✅ 可行性结论：**可行**（六环全通；边界见 D2 §1）。
  - D4 ✅ 本进度块。
- **关键读数**：① 技能加载成功、CreateWorkflow 未拒跑；② 一次运行即完成，**无需**
  `ResumeWorkflowRun`/`args` 重起；③ 回合在 workflow 运行期间**正常结束**（09:35:22），
  完成通知**自动送达并触发新回合**（09:35:46）；④ **轮末 goal 完成校验未触发**
  （2026-10-11 全库 0 条 `target_completion_verification`；最可能因后台任务在跑而 `defer`，
  **推断**）；⑤ 卡片 `probe-report`(primary) 与 `probe` 已发布并随通知送达（UI 渲染未核）。
- **未做 / 未验证**：goal 校验读数（未触发故无读数）、卡片 UI 渲染、`created_at` 与发起前
  读数约 5m53s 间隔的成因；GPU 轮编排 / freeze / commit / push 均未执行。
- **下一动作**：无（探针目标已按 §0 六步走完）。若用户要推进 campaign，第一步是
  `.venv/bin/python tools/check_campaign_state.py --campaign docs/goals/main-model-climatology-campaign.md`
  并按主计划当前节点对表；**本轮未做、也不属本探针范围**。
