# 0019 自迭代连续 goal 提示词（含 workflow，从真实 S3 接续）

**状态：文档交付与本地验证中（2026-10-09）；科研未执行、未提交推送。**
起点 `9c45341d259bce47ab36a10bd6038cb7656b2092`，分支 `r7/weather-reasoning`。
唯一主计划与可复制 objective：`docs/goals/main-model-climatology-campaign.md`；本计划不是第二个 master。

## 1. 本轮范围与真实起点

用户要求：分析连续任务会话 `sess_1d199296-0818-4484-a384-786216eabbe0` 为何自己停了，并给出能
**自我迭代**（一轮输出作为下一轮输入）的 goal 提示词，且能用 ZCode 的 workflow；其余要求与上一轮
（计划 0018）一致；归档文件后给出提示词。

本轮只改**治理/技能/文档层**：不新增机械检查器，不动生产模型与训练代码，不下载/发布/clone/训练/评分/
查 GPU，不暂存提交推送，不改冻结证据、科学合同、index/brief、用户配置与安全 hook，不关闭 issue。
本轮新增 **0 GPU-h**。

已核起点：仍 **S3**；索引 **65** 条；累计 **27.5905 GPU-h**（cap20/remaining−7.5905 为会计字段）；
`current_round_goal=docs/goals/s3-wide-region.md`；2023 test 未评分、r=0；S4 未启动；HEAD
`9c45341d259bce47ab36a10bd6038cb7656b2092`，工作树仅有用户自己的两处改动
（`.zcode/agents/web-researcher.md`、`.zcode/config.json`，本轮不碰）。

## 2. 停止原因分析（数据库证据，非推测）

只读查 `~/.zcode/cli/db/db.sqlite`（`sess_1d199296-0818-4484-a384-786216eabbe0`）：

| 位置 | 实测值 |
| --- | --- |
| `session_target.status` | `complete`（`active_run_started_at` 为 NULL） |
| `session_entry`（`target_completion_verification` 最新条） | `json_extract(data,'$.payload.status')='failed_closed'`、`verification.passed=true`、reason `Completion verifier request failed: Model request failed.`、`goalIteration=1` |
| `model_usage`（该 query_source，仅 1 行） | `status='error'`、provider `new-provider`、model `cline-pass/deepseek-v4.1-flash`、`duration_ms=31157`、0 token |

结论：**不是科研完成、也不是用户打断**，而是轮末 goal 完成校验调用 provider 报错、被 harness
**fail-open** 判成通过并把目标标 `complete`。这与 `docs/R7_ZCODE_GOAL_VERIFIER_ABORTS.md` 的**形态一**
一致。**科学状态未变**（r=0、2023 test 未评分、S4 未启动）。次要观察：该会话在阻塞式 CI 轮询上花了
约一小时（多次 `sleep 130–500`），但它确实完成了一个真实轮次（宽区域获取路径 + 1-part 真实 pilot，
登记 index65，已提交推送）。

## 3. 关于 `/workflow`

已核清：**不存在名为 `workflow` 的技能**。`/workflow` 是用户侧触发名，落地契约是 bundled 技能
**`dynamic-workflows`**，且 `CreateWorkflow` / `AmendWorkflow` / `SaveWorkflow` / `EvalWorkflowSnippet`
在该会话加载此技能前会**直接拒绝运行**。该技能还规定「只有显式请求才启动 workflow」——用户本次的
显式要求即为授权。故提示词写：**先加载 `dynamic-workflows`（即 `/workflow` 的落地技能）**，再按并行
准入判断决定是否真的用 workflow。

## 4. 交付物

| # | 交付物 | 位置 |
| --- | --- | --- |
| D1 | 自迭代 objective（单段、实测 4000 字符） | master `§0` |
| D2 | 「连续自我迭代契约」节 | master（`§1` 前） |
| D3 | CI 不阻塞纪律 | master `§5` |
| D4 | 本轮 0 GPU-h 交接块 | master `§8` |
| D5 | 「连续自我迭代（不等 verifier）」节 | `.agents/skills/goal-loop/SKILL.md` |
| D6 | 决策 0043 | `docs/decisions/0043-self-iterating-continuous-goal.md` |
| D7 | 本计划 0019 与三处索引/留痕 | `docs/plans/`、`docs/decisions/README.md`、`docs/plans/README.md`、`docs/rules/CHANGELOG.md` |

新 objective 相对上一轮的净增内容：**自迭代契约**（四件套 + 立刻续跑）、**不等 verifier**（含本次
fail-open 实测）、**停止条件封闭三条**、**防空转**、**CI 后台轮询**、**workflow 接入**
（先加载 `dynamic-workflows`，再受决策 0042 七 gate 约束）。技能清单由 12 项增至 **13 项**
（`dynamic-workflows` 为 ZCode bundled 技能，不属本仓 `.agents/skills/`，故不进 `docs/skills/README.md` 能力表）。

## 5. 验证

- `check_goal_brief.py --brief`（单段 / 自指针 / ≤4000 code points）：`briefs=1 failures=0 advisories=0`。
- `check_campaign_state.py --campaign`：`node=S3 failures=0 notes=6`（notes 均为历史 0 成本/无 index 行）。
- `check_conventions.py` 37 条阻断 + `--paths` 模拟提交；`verify_r7_evidence_index.py`（65 条）；
  `git diff --check`。
- 四个完整 CPU 模块（`test_check_goal_brief` / `test_check_campaign_state` / `test_check_conventions` /
  `test_verify_r7_evidence_index`）：venv `-B`、禁网、隐藏 CUDA、仓外 tmp、无 pytest cache。
- 独立只读审阅：提示词科学门完整性 + 技能 / ADR / 计划一致性；有必修先修后复核。

## 6. 明确不做

不启动实验/下载/发布/clone/GPU；不改生产模型与训练代码、科学合同、冻结证据、index/brief、账本历史行；
不 commit/push；不改用户配置；本轮不创建任何 workflow 脚本文件（只写文档）；不把「不等 verifier」
解释成可以无停止条件地空转。

## 实际结果

- **已落盘**：master `§0` 换成自迭代 objective（单段、实测 **4000 code points**，13/13 技能名、5/5 代理、
  D1–D6 齐全）；master 新增「连续自我迭代契约」节与 `§5` CI 纪律、`§8` 本轮 0 GPU-h 交接块；
  `goal-loop` 技能补「连续自我迭代（不等 verifier）」节（260 行，未超 R-051 的 600 行）；
  新增决策 0043 与本计划 0019；三处索引/留痕同步。
- **停止原因**：DB 三处证据一致指向「轮末 goal 校验 provider 报错 → fail-open → 标 `complete`」，
  科学门一条未过；本轮据此把「不等 verifier、按 `§8` 恢复」写进 objective 与技能。
- **验证**：见 §5；`check_goal_brief` 与 `check_campaign_state` 实跑结果如上；四模块 CPU 测试与
  独立审阅的回执写入本轮交付说明（本地 tmp 回执不是新科学索引 record）。
- **独立只读审阅**（两轮，均为只读、不改文件）：
  - 第 1 轮：科学门 8 条逐条在且未被弱化；自迭代契约与 `campaign-state`/index/账本口径自洽；
    DB 三处终态实测与文档陈述完全一致；无越界。报 **2 条必修**：技能计数（master §4 表与
    `docs/skills/README.md` 未含 `dynamic-workflows`）与 `goal-loop` 行数（261→260），均已修。
  - 第 2 轮复核：抓出上一条修复**自身引入**的重复句与「上表 12 项」计数矛盾，报 **2 条必修**，
    均已修（§4 表后注改为「其中 12 项为本仓能力、第 13 项为 bundled」）；并确认第 7 条 0 GPU-h 例外
    与 §8/§7 一致。**审阅不是科学接受，也不是最终 goal verifier。**
- **未跑**：全量 suite、wheel、远端 CI；既有绿色 CI 不覆盖本轮未提交改动。
- **影响**：仅治理/技能/文档层——1 个技能文件扩节、1 份 ADR、1 份计划、master 四处与 3 处索引；
  公开接口、默认参数、模型 digest、数据与结果、依赖、安全 hook/配置/凭据均无改动。新增文件未跟踪，
  治理层未提交时干净克隆仍不可用。
- **遗留**：自迭代契约尚无**真实连续轮次**的实证（首次执行轮必须留痕验证）；dynamic workflow 通道
  在本仓从未使用，首次使用须留痕验证；最终科学目标未通过，仍 S3，test 未评分、r=0、S4 未进，
  保留授权边界不变。
