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
| [0007](0007-b1-data-scope-d1-only.md) | D-1 数据范围：B1 只用 D1 工程段 | accepted | 2026-09-27 |
| [0008](0008-b2-segment-extension.md) | B2 数据范围：D1 段扩到 36 天（train 逐位不变） | accepted | 2026-09-27 |
| [0009](0009-campaign-scope-and-narrowing.md) | 全自动战役 S3–S5 的执行决策与范围收窄 | accepted | 2026-09-27 |
| [0010](0010-units-defect-and-two-month-segment.md) | 单位缺陷修正与双月（M2）段 | accepted | 2026-09-27 |
| [0011](0011-issue-tracking-exemption.md) | 会话内无 GitHub 写通道，issue 追踪按环境限制豁免 | accepted | 2026-09-28 |
| [0012](0012-planner-delegation-scope.md) | planner 子智能体的委派边界与 JSON 契约 | accepted | 2026-09-27 |
| [0013](0013-v2-round-one-switched-pathway.md) | V2 第一轮：时空输入与位置化 process 读写，双开关默认关 + 逐位等价证明 | accepted | 2026-09-28 |
| [0014](0014-round-two-pooled-query-capacity-control.md) | V2 第二轮：池化 query 容量控制臂 + 四臂三 seed 归因设计 | accepted | 2026-09-28 |
| [0015](0015-goal-loop-scope.md) | goal-loop 能力：目标撰写与会话手工循环，校检器只读且非阻断 | accepted | 2026-09-28 |
| [0016](0016-round-three-field-modes-and-primary-reader.md) | V2 第三轮：字段模式的臂内替换、batch=1 的确定性语义、primary 读者按冻结文字实现 | accepted | 2026-09-28 |
