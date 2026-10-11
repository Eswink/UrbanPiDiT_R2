# 0044 workflow 运行充当本轮执行者（含脚本存放与跟踪的裁决）

- **日期**：2026-10-11
- **状态**：accepted

## Context

用户要求「用 workflow 搞一个特别大型的、针对本项目的、让目标循环自迭代」的编排。调研后确认了
可行边界与四个硬事实：

- **可表达但有界**：动态 workflow 允许 `while`/`for`/`Promise.all`/`break`，harness **没有节点上限**，
  `world.run` 也没有超时上限，所以「一次运行内循环 N 轮、每轮含 GPU 训练与登记」技术上可表达。
  但它是**归当前进程所有的后台运行**：进程退出即 `interrupted`，只能靠 `ResumeWorkflowRun` 恢复；
  子代理**不能嵌套** workflow；子代理**很可能不跑本仓 hook**（E-186/E-187，Q-013 仍未实测）。
  ⇒ 它是**有界、可恢复的批量编排器**，不是无人值守守护进程；后者违反决策 0030/0038 明文的禁 cron、
  禁会话外续跑。
- **`world.run` 的 argv 必须是编译期字面量**（无 shell / 管道 / 变量展开），而驱动的 `--out`/`--deadline`
  是运行时值 ⇒ 编排必须走「**文件即参数**」：脚本只传字面量子命令，动态值落在盘上的机器可读块里。
- **驱动没有任何 resume**：产物全部 `open("x")` 独占创建，attempt 目录必须全新
  （`fresh_output` → `FileExistsError`），`failure.json` 明文「no automatic retry or in-place resume」。
  ⇒ 被打断的一轮 = 该 attempt 作废、新 attempt 重付 GPU。这是打断风险的真缺口（由决策 0045 有限放宽）。
- **三个能力此前不存在**：打印 campaign-state/账本的 JSON 工具、CI 状态轮询脚本、独立的只读 GPU
  余量探针 CLI。另：`.ts` 脚本**不受任何 convention 规则约束**（R-051/052/035/037/038/047/053 均不覆盖），
  但新增 `tools/*.py` 受 ≤600 行 / 函数 ≤200 行约束。

同时暴露两处治理冲突与欠账：

- **`.zcode/workflows/` 的跟踪状态自相矛盾**：决策 0001 第 6 条明确在 `.gitignore` 放行该子目录
  （客户端按固定路径发现按名可运行的 workflow，放别处会「文件存在但不可用」），而决策 0042 第 3 条
  与本仓 `parallel-research-workflow` 技能都写「workflow 脚本属工具运行态、不进版本控制（R-035）」。
- **首次真实使用未留痕**：2026-10-11 已存在草稿 `.zcode/workflow-drafts/第-6-轮推理深度-K8-驱动实现与独立复核.dwf.ts`
  与一次真实运行产物 `.zcode/workflow-runs/dwfrun-0f324485-a59c-468e-866a-34d7bcdbfda3.mjs`，
  但 master §8、计划与证据索引均无记录；而决策 0042 与该技能都要求「首次真实使用必须留痕并验证」。

## Decision

1. **授权 workflow 运行充当本轮执行者**，即允许一次 workflow 运行完成「实现 → 独立复核 → 冻结 →
   GPU 训练/评分 → 读数 → 四件套登记 → 提交推送」的完整一轮，但必须同时满足：
   - **自跑确定性闸门，不依赖 hook**：提交前由脚本自己执行 `tools/check_conventions.py --quiet`
     （等价于提交闸门），登记后由脚本自己执行四个校验器；任何一步失败即停下，不得提交。
   - **只推工作分支**：`git push` 仅允许目标 `r7/weather-reasoning`；**禁止**推 `main`、force、
     `--mirror`、`git merge`、删除分支、关闭 issue。
   - **每轮一个臂、串行执行**：GPU 仍由该运行的单一父调度串行持有（R-054），每次 spawn 前只读探余量，
     **绝不向非本实验进程发信号**；由此决策 0042 的七 gate 由「串行」自然满足，不得借 workflow 之名
     并行跑 GPU 臂。
   - **有界**：脚本必须自设轮数硬上限；停止条件沿用决策 0043 的封闭三条，停下必须在 §8 写
     「为何停 + 恢复第一条命令」。
   - **不得自宣科学完成**：`report()` 输出与 CI 绿都不是科学结论（决策 0024、`ci-and-verification.md`）；
     最终 goal 的 `complete` 仍由运行时独立校验或用户裁定。
2. **裁决脚本存放与跟踪**（澄清 0001 与 0042 的冲突）：
   - **草稿/一次性脚本** 放 `.zcode/workflow-drafts/`（机器所有、git-ignored，R-035 范围）；
   - **保存命名、按名复用的脚本** 放 `.zcode/workflows/<name>.dwf.ts`，按决策 0001 第 6 条**入库跟踪**
     （客户端按固定路径发现它，未跟踪等于干净克隆上不可用）；
   - 决策 0042 第 3 条「不进版本控制」**只适用于草稿与运行态**（`.zcode/workflow-drafts/`、
     `.zcode/workflow-runs/`），不适用于保存命名的 workflow。
3. **六层打断抗性**（对「尽可能降低打断风险」的正面回答，细则见计划 0020 与
   `parallel-research-workflow` 技能）：
   (a) workflow journal + `ResumeWorkflowRun`（同 run id 恢复，已完成调用按记录重放、不再花 token）；
   (b) 一轮切分为约十二个 `world.run` 单元而非一个巨型调用；
   (c) 盘上 receipt + **skip-if-done**（由 helper 实现，驱动本身不会）；
   (d) 每轮落 git，使任何新会话只读 §8 即可续跑；
   (e) 跨 attempt 的 pinned seed 复用（决策 0045）；
   (f) 有界等待 + 干净停下 + 写明恢复命令。
4. **首次真实使用必须留痕**：本轮补登记 2026-10-11 的那次草稿运行（写入 master §8 与本决策引用），
   并规定此后每次真实使用都要在 §8 留下运行产物路径、轮数与结论。

## Consequences

- **代价与风险**：① 成本随轮数线性增长——每个子代理是一次完整模型会话，一次 5 轮运行约 30 个模型会话
  加 5 个 GPU 轮（约 25 GPU-h），必须由轮数上限与预算硬约束住；② 子代理不跑 hook，安全只能靠
  「脚本自跑闸门 + 只读/排他写约定 + 只推工作分支」三重构造，**不得**把安全寄托在 hook 上；
  ③ workflow 无法自动恢复自己，进程退出后必须由人或外层 goal 循环执行 `ResumeWorkflowRun`；
  ④ 若把无上限循环当「永久自迭代」，会违反防空转条款与决策 0030/0038。
- **不改**：科学合同与冻结判据、冻结证据、index/brief、账本历史行、生产模型与训练代码、
  GPU 共驻与禁信号规则、R-006/R-028、`scientific_claim: false` 与 limitations 要求。
- **依赖**：需要计划 0020 的支持代码（`tools/r7_round.py` + 支持模块、CI 轮询 helper）与
  `.zcode/workflow-drafts/r7-campaign-round.dwf.ts` 草稿；`SaveWorkflow` 固化到 `.zcode/workflows/`
  按本决策第 2 条执行。
- **引用**：决策 0001（脚本位置）、0024（不依赖 verifier）、0026/R-054（GPU 共驻）、0030/0038（授权边界与
  禁 cron）、0042（七 gate 与编排形态）、0043（自迭代契约）、0045（跨 attempt 复用）；
  `.agents/skills/parallel-research-workflow/SKILL.md`、`docs/plans/0020-campaign-driver-workflow.md`、
  `docs/R7_ZCODE_GOAL_VERIFIER_ABORTS.md`。
