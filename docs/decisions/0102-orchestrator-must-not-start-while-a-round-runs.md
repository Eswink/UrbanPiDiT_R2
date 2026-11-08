# 0102 — 编排器**不得**在轮次在跑时启动：`implementer` 会在 `dryRun` 早退**之前**重指 `campaign-state`

- **日期**：2026-11-08
- **状态**：accepted
- **范围**：`.zcode/workflow-drafts/r7-throughput-physics-round.dwf.ts`（S3 吞吐/物理 campaign
  的编排器）以及任何**同一骨架派生**的编排器。**本 ADR 0 GPU-h、零读数、不改任何冻结判据。**
- **依据**：目标长文 `docs/goals/s3-throughput-and-physics-campaign.md`（「跑 stage 前先跑
  `tools/r7_round.py state` 核对 `current_round_goal`，否则冻结错轮次」）；
  决策 **0044**（自跑确定性闸门、只推工作分支、有界）；
  `tools/r7_round.py` 的 `orphans` / `current_goal` 实现（实测）

## Context

**① 触发这次检查的场景。** 目标长文要求「用 workflow 编排」。`attempt06` 正跑到 54–56%，
此时考虑跑一次 `dryRun` 是自然的：它被设计成「只侦察与实现+复核，不冻结、不开 GPU」。

**② 实测的次序（不是读注释，是定位行号）：**

| 行 | 内容 | 何时执行 |
| --- | --- | --- |
| 384 | `scout.ask(SCOUT_ASK + state)` | 每轮先侦察 |
| 391 | `implementer.ask(IMPLEMENT_ASK(plan))` | **在 dryRun 判断之前** |
| 432 | `if (dryRun) { …; break; }` | **之后**才早退 |

而 `IMPLEMENT_ASK` 的 prompt 第 2 条（第 202 行）逐字要求实现者：

> `2) **把 campaign-state.current_round_goal 改成 " + plan.goalPath`

⇒ **`dryRun` 拦不住它**。「dryRun 不动状态」是**注释里的意图**，不是执行次序保证的
性质。这是本仓第 N 次「注释/命名声称的安全性 ≠ 实测行为」。

**③ 为什么这在当前时刻是**有害**的。** `r7_round.py` 的每个子命令都从
`campaign_state()["current_round_goal"]` 解析本轮 spec（`tools/r7_round.py:70-76`）。
当前指针指向 `docs/goals/s3-throughput-physics-c2.md`。若被重指：
- `stage reading` 会解析**另一份** brief 的 `round-spec`；
- 其 `output_stem` 不同 ⇒ `current_attempt_dir()` 定位到**别的** attempt 目录；
- 于是读数要么找不到 receipt，要么（更坏）在错的目录里产出 `readings.json`。

目标长文把这个坑写成「跑 stage 前先核对 `current_round_goal`」——那条是**给执行者**的。
本条指出它在**编排器内部**有一个镜像形态，而**运维纪律挡不住它**，因为重指动作发生在
编排器自己身上。

**④ 现场实测：`orphans` 能回答「有没有轮次在跑」，且现在答案是非空。**

```
running: [{'phase':'seed','seed':41,'running':True,'pids':[1817520,1819196,...]},
          {'phase':'seed','seed':42,'running':True,'pids':[1817521,1819313,...]}]
```

`orphans` 是**只读**的（`tools/r7_round_liveness.py`），不发任何信号，符合 R-054。

## Decision

1. **编排器在 `implementer.ask` 之前加一条硬前置**：先跑
   `.venv/bin/python tools/r7_round.py orphans`；
   - `exit 0` 且 `running` 非空 ⇒ **停**，写明「有轮次在跑」，不动任何文件；
   - 输出无法解析 ⇒ **停**（不宽容）；
   - `orphans` 自身非 0 ⇒ **停**（不宽容）。
2. **停止必须发生在任何写动作之前**，因此前置的位置在 `phase("实现与独立复核")` 之后、
   `implementer.ask` 之前，而不是在 `dryRun` 判断处——放在 `dryRun` 之后对**非** dryRun 的运行
   仍然是错的。
3. **不修 `IMPLEMENT_ASK` 里那句「改指针」**：在**空闲**时那是正确且必要的动作
   （`r7_round.py` 靠它定位本轮 spec）。本条修的是**时序**，不是那条指令本身。
4. **本条的守卫不覆盖**：`Scout`/`Auditor`/`Registrar` 这三个子代理也可能写文件。本条只堵
   已实测到的那一条路径；其余按同一理由应各自核，但**没有**在本条里逐个实测，故不声称覆盖。

## Consequences

**负面后果（代价，必须如实列出）**

1. **我不能声称这个 560+ 行的 draft 通过了类型检查。** 仓内没有 TypeScript 编译器，
   `npx typescript` 离线不可用。我改的 guard 块自身 `(`/`)` 与 `{}` 平衡、位置在第 401 行
   （`implementer` 之前、`dryRun` 之前），但**整文件可编译性**只有它被提交给
   `CreateWorkflow` 时才第一次被真正验证。**这是本条最大的未验证面。**
2. **我改坏过一次这个文件。** 第一次编辑把 `if (plan.blocked) { … }` 的收尾三行截掉，
   使第 388 行语句断裂。自查（大括号失衡 + 断行）后发现并补回。**记下来是因为**
   「改一个 560 行、无编译器的文件」这件事本身就容易留下这种伤口，而它若没被发现，
   会在有人**真要**跑编排器时才爆。
3. **守卫引入了一个新的失败模式**：若 `orphans` 因为与本轮无关的原因返回非 0（例如工具被
   重命名），编排器会**拒绝启动**而不是照常跑。这是**故意**的选择（宁可停也不要在不确定时
   重指指针），但代价是有人得读懂停止原因再手动解除。
4. **`orphans` 只看得见**本仓**的 stage worker**（按 attempt 目录与命令行匹配）。若有别的会话
   在跑同一个 attempt，或有人在别的路径上跑同一 store，本守卫看不见。
5. **本条零读数、零 GPU-h**，不构成任何关于编排器「好用」或 C2 结论的证据。

**正面后果**：把「dryRun 安全」这一**错误直觉**纠正为「dryRun 也会写 `campaign-state`，
所以有时间序依赖」；给出一个**只读、不信号**的前置探测（`orphans`）而不是靠运维记得核对；
把「目标长文里给执行者的那条核对纪律」在编排器内部**机械化**了一次；
并如实登记了「无编译器 ⇒ 未验证整文件」这一面，而不是让「我改好了」含混过去。

**否决**：

- **靠「我会记得先核对 current_round_goal」**。否决：目标长文已把这条写成纪律，
  而它没挡住这次的时序问题——纪律挡不住自动化内部的写动作。
- **把守卫放在 `if (dryRun)` 之后**。否决：那只保护 dryRun，非 dryRun 的运行仍是错的。
- **改 `IMPLEMENT_ASK` 删掉「改指针」那条**。否决：空闲时那是必要动作；
  删了会让 `freeze` 读到上一轮的 spec。
- **在 `attempt06` 在跑时跑一次 dryRun「试试」**。否决：实测 `running` 非空，
  正是禁用条件；试的代价是 `stage reading` 指向错目录。
- **声称编排器已验证可用**。否决：无编译器、且从未跑过一次；本条只修一条实测缺陷。

## 未决

- 谁来为这个 draft 做**第一次真实编译**（`CreateWorkflow` 或离线装编译器）——
  在没有它之前，「编排器可用」只能写成**未验证**。
- `Scout`/`Auditor`/`Registrar` 三条写路径是否也该有各自的前置（本条未测）。
- `orphans` 看不见跨会话/跨路径的同 store 并发（R-054 的可见性边界）。
