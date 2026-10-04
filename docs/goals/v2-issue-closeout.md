# v2-issue-closeout：补 #71/#72 完成判据、登记六条判定并关闭 #70–#75

<!-- round-node: N5 -->

**状态：prepared（2026-10-02）。本文件是待执行的目标长文（收尾轮，非实验轮，0 GPU-h）。执行者先跑
`tools/check_campaign_state.py` 对表，再按 §4 执行；本轮的产物是**证据补齐 + 判定登记 + 关闭提交**，
不是新的科学结论。**

本文件承接 `docs/plans/0009-r7-v2-completion-and-closeout.md` 的阶段 D+E，是主计划
`docs/goals/main-model-v2-campaign.md` 的收尾节点（执行时在 §8 声明为 `N5`）。它只在 A/B/C 三轮都已
登记之后执行。

## §0 Objective（可粘贴；实测 1503 字符）

> 本轮目标：执行 docs/goals/v2-issue-closeout.md 的收尾轮——补齐 #71/#72 的完成判据缺口，登记 #70–#75 六条判定，并把它们关闭。0 GPU-h，全部只读+提交。D 部分（证据补齐，只新增验证、绝不放宽判据）：#71 补 tests/ 直接断言——年末（doy 365→1 连续）、闰年（doy 366 / 365.25 处理）、UTC 跨日、东西经（含 negative longitude）、batch 内不同日期、odd grid+padding、以及「内部 K 保持同一 valid_time」（reasoning_steps>1 时相位/条件不变），并把这些写成对 docs/goals/n2a-m3-process-supervision.md 与 tests/test_r7_spacetime_inputs.py 现有覆盖的补充而非替换；#72 补「局部草稿扰动经位置化反馈到达相应 latent」的**现役机制**测试（现役路径是 positional_process_readout + local_solver_state，而 tests/test_r7_spatial_solver_feedback.py:53 测的是在 RW-B 配置里保持关闭的 spatial_solver_feedback），以及 process_reader 参数的非零梯度断言；两处都写成反证可失败的测试。E 部分（判定与关闭）：把 #70–#75 每条的证据、verdict（DONE/NEGATIVE/BLOCKED）、未做项写进 docs/R7_ISSUE_COMMENTS.md 与 docs/R7_TASK_QUEUE.md；#70 EPIC 按「已交付 V2 源码与最小证据链，科学增益为 negative/mixed」如实写，不冒称科学成功；#75 必须把「科学增益 / 工程完成 / 某试验否定」三类标记分开写，不通过「所有 Issue 已关闭」冒充模型研究已完成；解除主计划 §2（docs/goals/main-model-v2-campaign.md:71）与各 goal 长文重述里的「不关闭 #70–#75」禁令，并新立一条决策记录该授权的范围与负面后果；然后写一个默认分支提交，正文逐行 Closes #70 到 Closes #75，先在工作分支等 CI 全绿，再非 force ff 推送到 main（决策 0003 放行；合并仍禁止且本计划不需要），最后用匿名 API 确认六个 issue 变 closed，并把 docs/goals/open-issue-resolution.md 标 complete、更新主计划 §7 账本与 §8 状态。判据见 docs/goals/v2-issue-closeout.md §3，权威来源 docs/goals/main-model-v2-campaign.md 与 docs/decisions/0003-push-vs-merge-hook-policy.md。禁止：改已冻结科学判据/阈值/端点/案例集、重跑或覆盖 outputs/ 归档、为关闭而删除或弱化测试与负面结果、读封存 test、下载数据、租 GPU、force push、--mirror、删默认分支、建立定时任务或后台续跑。预算 0 GPU-h；停止条件为任一 issue 的证据不足以支撑其 verdict、禁令未按要求解除、或 ff 推送被拒（此时停在分支上并报告）。不要把「全部关闭」当作目标完成。

## §1 现状（2026-10-02 只读核对）

- **六个 issue 全 open**（#70–#75；#70 有 1 条评论，其余 0 条；无 closing keyword）。近期 `docs(r7)`
  提交只写「close the round's bookkeeping」，**不是** issue 关闭。
- **#71 的判据缺口**（完成判据第三组）：`tests/test_r7_spacetime_inputs.py` 只覆盖
  `test_the_phase_is_the_valid_time_and_wraps`（`:166`，跨午夜）等部分；全仓 `docs/` + `tests/` 对
  **年末 / 闰年 / 西经** 无任何直接或间接验证；**「内部 K 保持同一 valid_time」无直接测试**；
  batch 不同日期 / odd grid / padding 只有间接覆盖。
- **#72 的判据缺口**（集成测试清单）：现有最接近「局部草稿扰动→相应 latent」的是
  `tests/test_r7_spatial_solver_feedback.py:53`，但它测的是 `spatial_solver_feedback`——该开关在 RW-B
  配置里**被明确保持关闭**（`tests/test_r7_local_solver_state.py:14-15`），其 off 分支断言的正是
  「到达 = 0」；现役的 `positional_process_readout` + `local_solver_state` 路径**没有**等价断言。
  `process_reader` 参数也**没有**独立的非零梯度断言（write/update 侧有，
  `tests/test_r7_local_solver_state.py:151`）。
- **关闭通道**：GitHub API 写返回 **401**（无 `gh`、无 PAT）；`Closes/Fixes/Resolves #N` 只有落在**默认
  分支 main** 的提交里才自动关闭（`docs/rules/ci-and-verification.md:165`、决策 0002）；main 是本分支
  严格祖先（HEAD..main=0），**非 force ff 推送**由决策 0003 放行；**合并仍被 hook 拒绝**且不需要。
  判定正文按先例（commit `e5be0c0` 关 8 个 issue、`90307e7` 关 6 个）写进提交信息，并在仓内留档。
- **禁令位置**（取消关闭必须逐处处理）：主计划 §2 `docs/goals/main-model-v2-campaign.md:71`；各轮 goal
  长文的重述（`n1-cost-supplement-repair.md:147`、`main-model-v2-pivot-audit.md:113`、
  `main-model-v2-rw-b-subtraction.md:142`、`main-model-v2-rw-b-round.md:124` 等）；`docs/goals/open-issue-resolution.md`。
- **判定先例格式**：`docs/R7_ISSUE_COMMENTS.md` 每 issue 一段
  （`## #<N> — <title>` + 引用块：verdict、artifact SHA、证据页、核验计数、`Not done / limits`、`Closure
  statement`）；`docs/R7_TASK_QUEUE.md:172-180` 的 Remaining research gates 表逐行给
  `DONE/NEGATIVE/BLOCKED … CLOSED on GitHub <时间>`。

## §2 交付物清单

| 编号 | 交付物 | 证据形态 |
| --- | --- | --- |
| D1 | #71 证据补齐（0 GPU-h） | 新测试（年末/闰年/跨日/东西经/batch 多日期/odd grid+padding/内部 K 同 valid_time），含反证；证据补记进 `docs/R7_71_72_M1_AND_RWA.md` 或新短页 |
| D2 | #72 证据补齐（0 GPU-h） | 现役机制的位置化反馈到达测试 + `process_reader` 非零梯度断言，含反证 |
| D3 | 判定登记（0 GPU-h） | `docs/R7_ISSUE_COMMENTS.md` 六段 + `docs/R7_TASK_QUEUE.md` 六行；#70 按 negative/mixed；#75 三类标记分离 |
| D4 | 禁令解除（0 GPU-h） | 主计划 §2 改写 + 各长文重述同步 + 新决策（下一个可用编号，预期 0028，范围=仅 `Closes #70–#75` 的收尾提交；仍禁 force push/`--mirror`/删默认分支）+ `docs/rules/CHANGELOG.md` |
| D5 | 关闭提交与 ff 落 main（0 GPU-h） | 工作分支上逐行 `Closes #70`…`Closes #75` 的默认分支提交；**先工作分支 CI 全绿**，再非 force ff 推送 main；匿名 API 复核六个 closed |
| D6 | 收尾回写（0 GPU-h） | 主计划 §7 账本行 + §8 状态；`docs/goals/open-issue-resolution.md` 标 complete；证据索引/ E 条目按需 |

## §3 判据与预声明读法

- **补齐只增不减**（硬约束）：新增测试是**补充验证**，不改任何已冻结判据、阈值、端点或案例集，不删、
  不跳过、不弱化任何测试或断言；若新增测试暴露**真实缺陷**，如实记录并按 shall-not-hide 处理，不静默修
  以迎合关闭。
- **判定必须有证据指针**：每条 verdict 必须指向已登记的运行记录/证据页与 digest；「无法证明逐位一致」
  的写明可复现等级（`docs/rules/` 与 `docs/goals/README.md` 的诚实纪律）。
- **关闭不等于完成**（issue 原文）：#75 要求「科学增益、工程完成和某个试验否定分别标记，不能通过
  『所有 Issue 已关闭』冒充模型研究已完成」；#70 EPIC 的交付条是「一个**实际改进**的 V2 及其最小证据链」，
  若拿不出实际改进，按 negative/mixed 如实写，**不得**用关闭动作替代科学结论。
- **自动关闭的唯一路径**：`Closes` 关键词必须出现在**默认分支 main** 的提交里（决策 0002/0003、
  `docs/rules/ci-and-verification.md:165`）；非 force ff 推送放行，**合并不放行也不需要**。
- **可复现等级与诚实**：FF 推送前必须已有工作分支上的**green** `R7 CPU CI` 运行；skip/cancelled/queued
  不算通过（`docs/rules/ci-and-verification.md:155-161`）。

## §4 实施顺序（不跳步）

1. **对表**：跑 `tools/check_campaign_state.py`（退出 0）；确认 A/B/C 三轮已登记、账本已更新、main 仍是
   工作分支严格祖先（`git rev-list --count HEAD..origin/main` = 0）。
2. **D1/D2（0 GPU-h）**：补测试与证据页补记；`pytest -q` 全绿（含反证），`check_conventions.py` 37 条 0 违规。
3. **D4（0 GPU-h）**：解除禁令（主计划 §2 + 长文重述 + 新决策 + CHANGELOG），先于任何关闭动作。
4. **D3 + D5**：写六段判定与 `Closes` 提交；推工作分支等 CI 绿；再 **非 force ff** 推送 main；用匿名
   API 复核 `state=closed`（若推送被拒，停在分支并报告，不重试 force）。
5. **D6 回写**：主计划 §7/§8、`open-issue-resolution.md` 标 complete；本长文 §7 记实际结果。

## §5 预算与停止条件

- 预算 **0 GPU-h**；不新增训练/评估/臂/seed；不动数据。
- **停止条件**（满足任一即停并报告，不自行放宽）：任一 issue 的证据不足以支撑其 verdict；禁令未能按
  §2 D4 解除；工作分支 CI 未绿；ff 推送被拒或 main 出现分歧（HEAD..main ≠ 0）。
- 目标状态 `active / paused / budget_limited / complete`；**执行者只可提议，不得自宣完成**。

## §6 明确不做

- 不改已冻结科学判据、阈值、端点或案例集；**不为让关闭好看而删除、跳过或弱化测试与断言**；不覆盖负面
  与归档结果。
- 不重跑、不覆盖、不改写任何归档产物与历史证据页（`outputs/` 一律只读）；不读封存 test、不评估、不下载、
  不租 GPU。
- 不 force push；不 `--mirror`；不删默认分支；**不做合并**（本计划不需要）；不建立定时任务或后台续跑。
- 不把「六个 issue 全部关闭」写成「模型研究已完成」；不宣告 SOTA 或 SCIENTIFIC SUPPORT。

## §7 进度块

- **状态**：执行窗口打开（2026-10-04）。前置全部就绪：actual C（144/144，独立接受 `aa44b8cd…`）、
  actual M1（24/24，verify `c5df8cc6…`）、新包 precision（独立接受 `8503adfe…`）与两侧 UTC 统计链
  已实跑并登记（E-245–E-247，证据页冻结于 `858eddb`）；#71/#72 的判据补强测试早已随工程 616b029
  落地（`tests/test_r7_v2_time_and_feedback.py`），本轮复跑通过，无需新增测试。
- **D1/D2（等价于已完成、本轮复跑）**：时间/地理与现役反馈测试 33 passed（CUDA-hidden）。
- **D3（本轮完成）**：`docs/R7_ISSUE_COMMENTS.md` 终局判定六段 + `docs/R7_TASK_QUEUE.md` 六行；
  #70 按 negative-mixed、#75 三类分写。
- **D4（本轮完成）**：主计划 §2 禁令按 0037 修订；决策 0037 accepted + decisions/README 行 + CHANGELOG 段。
- **D5（已完成）**：关闭提交 `f0b77057fca425c22826687cd6a0a28342ab5cf8`（逐行 `Closes #70`…`Closes #75`）
  经工作分支精确 SHA `R7 CPU CI` run 37215981072 全部 12 必要步骤 success（step 1–9 + post 17/18/19）；
  非 force ff 推送 `dafd22e..f0b7705` 落 main（无合并提交）；匿名 API 复核六 issue 全部 `state=closed`
  （closed_at 2026-10-04T16:29:26–28Z，`state_reason: completed`）。
- **D6（本轮完成）**：主计划 §7/§8、`docs/goals/README.md` 行、本页与 `open-issue-resolution.md` 回写；
  证据索引 26 记录、canonical brief 重渲染、campaign-state 对表 exit 0；queue 六行与 comments
  状态段已写入关闭时间戳与 run/SHA。
- **未做**：最终独立年份/季节测试、可分辨过程语义新协议、adaptive 重评——仅提议。
- **本轮实跑暴露并处置的工程缺陷（如实记录）**：非 CUDA-hidden 全量跑出一个次序相关失败——
  `training/r7_comparison_plan.py:30` 的 `torch.random.fork_rng()` 未传 `devices=[]`，在可见 CUDA 的
  机器上会初始化全部 GPU，触发 `training/r7_v2_driver.py:268`「CPU prepare must not initialize CUDA」
  守卫（守卫正确、未弱化）；该文件不在任何冻结 file map（M1 96 文件、revision06 111 依赖、C 均无），
  已按仓库既有惯用写法改 `devices=[]`（与 `r7_v2_profile.py:54`、`r7_autoregressive_runner.py:153`
  一致），定向前后对（baselines+driver）恢复 12 passed。正交性实测：`stash` 该改动后 M1 verify 仍以
  同一 `source_head` 条款失败（HEAD 已从冻结的 562e526 前进到证据提交 858eddb——按设计；`files`/
  `runtime` 映射与 `model_digest` 均逐项相等），即该修复不触碰任何冻结身份。**仍存在**：
  `training/r7_experiment.py` 的 `rng_state()` 在 CUDA 可见但未使用时也调
  `torch.cuda.get_rng_state_all()` 使 CUDA 初始化——该文件在冻结 map 内，本轮不改，如实记为
  「CUDA-visible 次序敏感、CI（CPU-only）与 CUDA-hidden 基线不受影响」的已知限制；
  重开条件=任何 GPU-visible 全量跑要求时。
- **下一动作（仅提议）**：关闭完成后为最终泛化测试另立预算与 goal 长文；执行者不自动开始。
- **诚实声明**：关闭动作不改写任何负面/未决读数；「六 issue 全部关闭」≠模型研究完成。

## §8 下一动作（仅提议）

关闭完成后**只提议**为「最终独立年份/季节的正式测试」（#75 的后置事项）另立预算与 goal 长文；执行者
不自动开始，也不宣告目标完成。