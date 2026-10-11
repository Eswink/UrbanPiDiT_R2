# 0020 campaign 驱动 workflow：归档与支持代码

**状态：文档与支持代码交付中（2026-10-11）；不运行 workflow、不开 GPU、不下载。**
起点 `0df159d81e04fada4efaefe01855c6dfdab8d197`，分支 `r7/weather-reasoning`。
唯一主计划与可复制 objective：`docs/goals/main-model-climatology-campaign.md`；本计划不是第二个 master。

## 1. 本轮范围与真实起点

用户问：能不能用 workflow 搞一个**特别大型**的、针对本项目的编排，让目标循环**自迭代**。本轮把方案
**归档进仓库并写出它需要的支持代码**，最后交一份可粘贴的提示词。**本轮不运行 workflow、不开 GPU、
不下载、不改科学合同与冻结证据。**

已核起点（只读）：HEAD `0df159d8`；工作树另有另一个窗口 staged 的
`scripts/study_r7_s3_reasoning_depth.py` 与 `tests/test_r7_s3_reasoning_depth.py`，以及
`docs/rules/CHANGELOG.md`、`docs/rules/size-thresholds.md` 的改动——**本轮不动它们**。
索引 **70** 条；账本 **49** 行、合计 **48.2867 GPU-h**（cap20/remaining−28.2867 为会计字段）；
`current_round_goal=docs/goals/s3-reasoning-depth.md`；2023 test 未评分、r=0；S4 未启动。

**§0 objective 已过期**：它写的「下一轮用宽区域数据做单因素真实 train/val」那一轮已经跑完，而且
「宽输入支作为长 lead 的解」已被判**关闭**（negative）。真实下一动作是**推理深度 K 4→8**。

## 2. 调研发现（决定支持代码形状的四个硬事实）

1. **驱动没有任何 resume**：产物全部 `open("x")` 独占创建，attempt 目录必须全新
   （`fresh_output` → `FileExistsError`，父路径 `mkdir(exist_ok=False)`），`failure.json` 明文
   「no automatic retry or in-place resume」。**被打断的一轮 = 该 attempt 作废、新 attempt 重付 GPU。**
2. **驱动已是子进程编排器**：`--worker prepare|archive|seed|reading`，`seed` **一次一个**（`--seed N`）；
   但 `--worker` 依赖 `preparation_protocol.json`（只有 `freeze()` 写）与 `protocol.json`
   （父路径在 prepare 成功后写）先存在。
3. **`world.run` 的 argv 必须是编译期字面量**（无 shell / 管道 / 变量展开），而驱动的
   `--out`/`--deadline` 是运行时值 ⇒ 编排必须走「**文件即参数**」：脚本只传字面量子命令，动态值落在
   盘上的机器可读块里（`<!-- round-spec: {...} -->`）。
4. **三个能力此前不存在**：打印 campaign-state/账本的 JSON 工具、CI 状态轮询脚本、独立只读 GPU
   余量探针。另：`.ts` 脚本**不受任何 convention 规则约束**；新增 `tools/*.py` 受 ≤600 行 / 函数 ≤200 行。

以及两处治理冲突/欠账：`.zcode/workflows/` 的跟踪状态在决策 0001 与 0042 之间自相矛盾；
2026-10-11 已存在一次**未留痕**的真实 workflow 草稿运行（`dwfrun-0f324485-a59c-468e-866a-34d7bcdbfda3`）。

## 3. 交付物

| # | 交付物 | 位置 |
| --- | --- | --- |
| D1 | workflow 作本轮执行者 + 脚本存放/跟踪裁决 | `docs/decisions/0044-workflow-as-round-executor.md` |
| D2 | 跨 attempt pinned seed 复用 | `docs/decisions/0045-pinned-seed-reuse-across-attempts.md` |
| D3 | 确定性支持逻辑（状态、复用准入、账本算术、校验器、提交闸门、GPU 余量探针） | `tools/r7_round_support.py` |
| D3b | 逐 stage 驱动包装（attempt 目录、skip-if-done、freeze、stage spawn、commit/push 原语） | `tools/r7_round_stages.py` |
| D4 | 薄 CLI（`state`/`verify`/`commit-guard`/`reuse-plan`/`ledger-plan`） | `tools/r7_round.py` |
| D5 | 匿名 CI 状态轮询（`curl` 子进程，无 Python 网络客户端） | `tools/r7_ci_wait.py` |
| D6 | CI-safe 反证测试（90 例） | `tests/test_r7_round.py`、`tests/test_r7_round_reuse.py`、`tests/test_r7_round_stages.py`、`tests/test_r7_ci_wait.py` |
| D7 | campaign 驱动 workflow 草稿（每轮：侦察→实现+独立复核→冻结+提交+等 CI→GPU 轮→读数+登记→交接） | `.zcode/workflow-drafts/r7-campaign-round.dwf.ts`（git-ignored） |
| D8 | 技能与路由：新增「campaign 驱动 workflow」一节 + 硬约束一条 | `.agents/skills/parallel-research-workflow/SKILL.md`、`AGENTS.md` |
| D9 | master §0 换成当前前沿的新提示词；§8 追加本轮块（含首次使用欠账登记） | `docs/goals/main-model-climatology-campaign.md` |
| D10 | 三处索引同步 | `docs/decisions/README.md`、`docs/plans/README.md`、`docs/rules/CHANGELOG.md` |

## 4. 六层打断抗性（对「尽可能降低打断风险」的正面回答）

1. **workflow journal + `ResumeWorkflowRun`**：同 run id 恢复，已完成的 `world.run`/`files`/`git` 调用
   按记录重放，**不再花 token**。
2. **按单元切分**：一轮 ≈ 十二个 `world.run` 单元，而不是一个 5 小时巨型调用；恢复只重做无记录的单元。
3. **盘上 receipt + skip-if-done**：由 helper 实现（驱动本身不会）⇒ 即使 run 记录丢了，新起一次运行
   也能接着做。
4. **每轮落 git**：四件套提交后，任何新会话只读 §8 就能续跑，零 workflow 状态依赖。
5. **跨 attempt pinned seed 复用**（决策 0045）：把最贵的一环从「三个 seed 全重训」降到「只补未完成的」。
6. **有界等待 + 干净停下**：不把「等不到」写成失败；停下必须留「为何停 + 恢复第一条命令」。
   恢复流程：重启后 `ListWorkflowRuns` → 找 `interrupted` → `ResumeWorkflowRun`；run 没了就用
   `args.resumeFromRound` 新起。

## 5. 验证

- 四个校验器：`check_conventions.py --quiet`（37 条阻断 0 失败）、`check_campaign_state.py --campaign`、
  `check_goal_brief.py --brief`、`verify_r7_evidence_index.py --check-brief`；`git diff --check`。
- 新测试模块 `tests/test_r7_round.py`（含逐字段反证：`REUSE_PROTOCOL_FIELDS` 每个字段单独漂移都必须
  拒绝复用；校验器一条红即不通过；提交闸门红即阻止；GPU 余量不足即不通过）与 `tests/test_r7_ci_wait.py`
  （含反证：`cancelled`/`skipped`/`queued`/`partial` 一律不算通过）。
- 既有 `test_check_conventions` / `test_check_campaign_state` / `test_agent_hooks` 回归。
- 全部在 venv `-B`、禁网、隐藏 CUDA、仓外 tmp、无 pytest cache 下运行。
- 独立只读审阅：ADR/技能/计划/提示词一致性 + 科学门完整性。

## 6. 风险

- **支持代码里最贵的一环尚未实跑**：`tools/r7_round*.py` 的 GPU 轮编排（freeze → 逐 stage spawn）
  没有端到端验证过，因为本轮不开 GPU。故本轮**只交付确定性与可测的部分**，并把「首次真实使用必须
  留痕验证」写成硬要求（决策 0044 第 4 条）。
- **复用是新的正确性表面**（决策 0045）：靠白名单全等 + 逐字段反证测试约束。
- **成本**：一次 5 轮运行约 30 个模型会话 + 5 个 GPU 轮（约 25 GPU-h）。故首次真实使用只跑 1 轮。

## 7. 明确不做

不运行 workflow、不开 GPU、不下载、不发布、不 clone；不改生产模型/训练代码、科学合同、冻结判据、
冻结证据、index/brief、账本历史行；**不编辑另一个窗口 staged 的 K=8 驱动与测试**；不推 main、不 force、
不 merge、不关 issue；不建 cron/守护/会话外续跑；不把 `report()` 或 CI 绿当科学结论。

## 实际结果

- **已落盘**：ADR 0044/0045；`tools/r7_round_support.py`（确定性逻辑，全部函数可注入 runner）、
  `tools/r7_round.py`（薄 CLI，按仓库惯例用 `importlib` 载入同目录模块）、
  `tools/r7_round_stages.py`（逐 stage 包装：attempt 目录、skip-if-done、freeze、stage spawn、commit/push）、`tools/r7_ci_wait.py`
  （`curl` 子进程轮询，**未 import 任何 Python 网络客户端**，故 R-016/R-025 不受影响）；
  `tests/test_r7_round.py` + `tests/test_r7_round_reuse.py` + `tests/test_r7_round_stages.py` + `tests/test_r7_ci_wait.py` 共 **90 例全通过**；草稿
  `.zcode/workflow-drafts/r7-campaign-round.dwf.ts`；技能/AGENTS/master/三处索引同步。
- **实测**：`tools/r7_round.py state` 在真仓库输出正确（HEAD `0df159d8`、账本 49 行、合计 48.2867、
  索引 70 条、末三条 record id、§8 末条「下一动作」）；`ledger-plan 4.2867` 得 new_total 52.5734 /
  remaining −32.5734；`verify` 三个校验器全 `exit_code=0`；`commit-guard` 对本轮七个新文件
  `exit_code=0`（37 条阻断 0 失败）。
- **修过的真实缺陷**：① 首版把 `check_conventions.py` 当可执行文件直接 spawn → `PermissionError`，
  改为 `sys.executable` 前缀；② 首版 `REUSE_RECEIPT_FIELDS` 把 `seed` 拿去和 protocol 比（protocol
  没有 `seed` 字段）→ 会导致**所有** seed 被误判为漂移，改为 seed 键与 receipt 自身比对；
  ③ `tests/test_r7_round.py` 一度涨到 625 行触发 R-051 → 拆出 `tests/test_r7_round_stages.py`；
  首版提交后 CI 的 `test_size_report_counts_match_the_checker` 变红（我的新文件把 R-019/R-019b/R-021/R-023
  四个报告计数推高）→ 把长行折行、拆出 `tests/test_r7_round_reuse.py`、把 `run_stage` 参数降到 8 个，
  使本轮新增文件对这些计数贡献为 0（不动 `docs/rules/size-thresholds.md`，那是另一个窗口的工作树改动）；
  ④ 独立只读审阅报 2 条必修：草稿 workflow 的 `world.run` 命令名用了运行时变量（违反 facade 的
  编译期字面量规则）→ 改为逐个 stage 展开的 7 次字面量调用；本计划的测试数与交付物表陈旧 → 已改。
- **未跑**：全量 suite、wheel、远端 CI；**GPU 轮编排未端到端验证**（本轮不开 GPU），故首次真实使用
  必须留痕验证（决策 0044 第 4 条）。
- **影响**：新增治理层工具与测试、2 份 ADR、1 份计划、1 份草稿脚本、技能/AGENTS/master 与 3 处索引；
  公开接口、默认参数、模型 digest、数据与结果、依赖均无改动。
- **遗留**：跨 attempt 复用的实际收益未实测（要等第一次被中断的真实轮）；`.zcode/workflows/` 的固化
  与入库尚未做（按用户选择「先 draft 后固化」）；最终科学目标未通过，仍 S3、test 未评分、r=0、S4 未进。
