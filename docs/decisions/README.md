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
| [0017](0017-local-gate-scope-and-commit-guard.md) | 本地闸门的判定范围（按已跟踪集合）与提交前闸门 | accepted | 2026-09-28 |
| [0018](0018-web-research-route.md) | 外部检索的唯一出口（web-researcher）与引用纪律 | accepted | 2026-09-28 |
| [0019](0019-verification-receipt-pilot.md) | 机器可读验证回执先行试点 | accepted | 2026-09-29 |
| [0020](0020-verification-contract-and-governance-drift.md) | 验证回执身份契约与治理漂移防护 | accepted | 2026-09-29 |
| [0021](0021-experiment-authorization-channel.md) | 实验授权的会话内询问通道 | superseded by 0029 | 2026-09-29 |
| [0022](0022-size-and-naming-hard-caps.md) | 规模硬上限与目录命名：先量后立、例外只许缩小 | accepted | 2026-09-29 |
| [0023](0023-main-model-first-baseline-freeze.md) | 主模型优先与基线冻结：开发期门禁与发表期门禁分开 | accepted | 2026-09-29 |
| [0024](0024-goal-verifier-best-effort-and-standard-recovery.md) | goal 自动完成校验按 best-effort：保留 harness 驱动，收尾不依赖自动结项 | accepted | 2026-09-30 |
| [0025](0025-campaign-master-plan-and-per-round-recheck.md) | campaign 主计划与每轮对表（机器可查部分交给 CI） | accepted | 2026-09-30 |
| [0026](0026-shared-gpu-coresidency-policy.md) | 本机 GPU 默认共驻与只读余量取卡政策 | accepted | 2026-10-01 |
| [0027](0027-n1-cost-acceptance-boundaries.md) | N1 成本审计的共驻验收与历史身份读取例外 | accepted | 2026-10-02 |
| [0028](0028-m3-process-sidecar-and-time-contract.md) | M3 派生尺度 sidecar 与三类时刻训练契约 | accepted | 2026-10-02 |
| [0029](0029-standing-experiment-delegation.md) | 实验常设下放与时长契约（取代 0021 的逐次询问通道） | superseded by 0030 | 2026-10-02 |
| [0030](0030-autonomous-execution-and-direction-boundary.md) | 实验、节点推进与普通决策常设下放；总体方向归用户（继承 0029，取代 0025 逐轮触发） | accepted | 2026-10-02 |
| [0031](0031-m3-cross-attempt-validation-provenance.md) | M3独立23项补测与跨尝试逐文件来源；保留原失败、身份强核和冻结暂停出口 | accepted | 2026-10-02 |
| [0032](0032-independent-autoregressive-exposure-route.md) | 用户前瞻选择：终结M3辅助监督假设，独立自回归暴露主线接续B/C/D/E；旧出口不改 | accepted | 2026-10-03 |
| [0033](0033-explicit-calendar-year-conditioning.md) | 显式已知init_calendar_year修Gregorian年界；旧字段/平均年周期路径逐位兼容 | accepted | 2026-10-03 |
| [0034](0034-explicit-draft-query-feedback.md) | 显式default-off局部draft query补齐设计；旧父保留、完整Generic配对与直接因果反证 | accepted | 2026-10-03 |
| [0035](0035-complete-known-context-contract.md) | 原local solar/history offsets/source位置契约显式补齐，B先封印、新C独立公平包级确认 | accepted | 2026-10-03 |
| [0036](0036-fp32-loss-audit-and-statistics-complement.md) | 实际FP32目标逐运算核验；保留B聚合failed，仅独立零GPU统计补全 | accepted | 2026-10-03 |
| [0037](0037-issue-closeout-window-scope.md) | N5收尾轮：#70–#75具名有条件关闭的范围、判定口径与代价 | accepted | 2026-10-04 |
| [0038](0038-autonomous-research-and-data-expansion.md) | 主模型超气候态方向自主研究；免费区域多年度/四季数据与自审preflight后新路径派生发布授权 | accepted | 2026-10-04 |
| [0039](0039-complete-source-fingerprint-for-prepare.md) | 源指纹必须完整：preflight 不再对大文件降级为 stat-only | accepted | 2026-10-05 |
| [0040](0040-explicit-valid-time-climatology-anchor.md) | default-off train-only valid-time 解码锚与异常反馈；物理状态分离和显式权重迁移 | accepted | 2026-10-07 |
