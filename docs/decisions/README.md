# 决策记录（ADR）

一个决策一份文件。格式与编号规则见 `docs/rules/artifact-storage.md`（R-033、R-036）与
`.agents/skills/decision-record/SKILL.md`。

## 与其它账目的分工

| 想记录什么 | 放哪 |
| --- | --- |
| 还没定的问题（需要人拍板） | `docs/rules/OPEN_QUESTIONS.md`（Q-xxx，待决队列） |
| 已经定下的决定（含为什么） | 本目录（NNNN，已决记录） |
| 规则本身的增删改 | `docs/rules/CHANGELOG.md`（按轮次） |
| 观察到的事实（含证据位置） | `docs/rules/EVIDENCE.md`（E-xxx） |
| 某次任务的计划与最终结果 | `docs/plans/`（NNNN） |

**边界**：Q 解决时**写一份决策记录**，并在 Q 标题加指针；不要只改 Q 就算记过决策了。

## 命名与状态

- 文件名：`NNNN-<kebab-slug>.md`，编号四位、单调递增、**不复用**（删除过的编号也不回收）。
- 状态：`proposed` / `accepted` / `deprecated` / `superseded`。
- 被取代时**不删旧文件**，把状态改成 `superseded by NNNN` 并在新文件里写明取代了谁。
  R-036 会机械检查这条链是否可达（指向的目标必须存在）。

## 索引

| 编号 | 标题 | 状态 | 日期 |
| --- | --- | --- | --- |
| [0001](0001-artifact-storage-convention.md) | 成品存放约定：计划 / 决策 / 目标 / 工具态 | accepted | 2026-09-24 |
| [0002](0002-controlled-main-write-for-issue-closing.md) | 受控 main 写入以在 GitHub 上关闭 8 个 open issue | accepted | 2026-09-25 |
| [0003](0003-push-vs-merge-hook-policy.md) | push 与 merge 的 hook 策略 | accepted | 2026-09-25 |
| [0004](0004-d1-source-selection.md) | D1 数据源选择（Earthmover spatial） | accepted | 2026-09-26 |
| [0005](0005-time-range-split-mode.md) | 发布契约新增「时间段切分」模式 | accepted | 2026-09-26 |
| [0006](0006-split-mode-reader-precedence.md) | 时间段模式下读者以 `split_time_ranges` 为准 | accepted | 2026-09-26 |
