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
| [0041](0041-saved-readback-role-json-size.md) | 保存读回角色 JSON 单文件上限统一为 64 MiB，免去仅为 2 MiB 限制的格式工程 | accepted | 2026-10-08 |
| [0042](0042-parallel-research-workflow.md) | 并行科研与 workflow 编排能力：并行准入七 gate、编排形态分级、隔离与科学完整性；限治理层 | accepted | 2026-10-08 |
| [0043](0043-self-iterating-continuous-goal.md) | 自迭代连续 goal：每轮交接四件套后立刻续跑、不等 verifier（含 fail-open 实测）、停止条件封闭三条、防空转与 CI 不阻塞；接入 `dynamic-workflows`（/workflow 落地技能） | accepted | 2026-10-09 |
| [0044](0044-workflow-as-round-executor.md) | workflow 运行充当本轮执行者：自跑确定性闸门不依赖 hook、只推工作分支、有界且不自宣完成；并裁决 `.zcode/workflows/` 跟踪冲突 | accepted | 2026-10-11 |
| [0045](0045-pinned-seed-reuse-across-attempts.md) | 跨 attempt 的 pinned seed 复用：白名单全等准入、按 seed 原子复用、逐 seed 如实报告、被打断的 GPU 全额记账 | accepted | 2026-10-11 |
| [0046](0046-arxiv-literature-research-route.md) | 文献检索通道：新增 `arxiv-researcher`（arXiv MCP 只读 9 工具）承担一手论文，网络通道维持 0018 的单一出口；文献通道无机械闸门、靠约定约束 | accepted | 2026-10-12 |
| [0047](0047-dual-card-seed-distribution.md) | 一轮内双卡 seed 分配：一卡一 seed 并发（默认双卡、保留单卡、不引入 DDP）、逐卡只读余量门 + 邻居增长 reserve；界定与 0042 第 4 条的适用范围 | accepted | 2026-10-12 |
| [0048](0048-capacity-axis-scope-and-cap.md) | 容量轴的适用范围与封顶：容量轮出口收窄为**深度轴**、宽度封顶为**最后一个**容量臂（失败即整条容量轴永久关闭）；含时序披露、宽度无逐位恒等扩张、成本口径更正 | accepted | 2026-10-13 |
| [0049](0049-explicit-tendency-input-round.md) | 显式 tendency 输入单因素轮的授权边界：只改 encoder 输入表示、+13,056 参数作为容量混杂、预声明 (a)–(e) 出口与 null 口径、新 revision 父迁移；不重开容量/架构轴 | accepted | 2026-10-21 |
| [0050](0050-fy-annual-coverage-round.md) | 全年覆盖单因素轮：4→12 个月、年份固定 2017–2021、65×65 限制、预声明 (a)–(d) 出口与非科学桥 (e)；不新增下载、不改模型 | accepted | 2026-10-21 |
| [0051](0051-sharpness-aware-minimization-round.md) | SAM 单因素轮：只改训练期更新规则（rho=0.05 全局重标扰动 + 二次 forward/backward + 逐位恢复），模型/spec/父不变；预声明 (a)–(e) 出口、null 口径、2× 算力混杂与负结果范围 | accepted | 2026-10-21 |
| [0052](0052-fy-annual-coverage-rebinding.md) | 全年覆盖轮改绑当前模型修订（`3fe747b2…` → `60de3b2d…`）：两个新增开关经测试断言为惰性、父迁移状态逐 seed 相同、桥 (e) 解释力下降如实声明；不放宽模型修订门 | accepted | 2026-10-22 |
| [0053](0053-fy-annual-coverage-budget-correction.md) | 全年覆盖轮预算更正：按首次真实使用实测把每 seed 9,000 s → 21,600 s（planned 43,200 / hard 86,400），并修 `default_runner` 超时逃逸导致 `stage-seeds` 整轮崩溃的工具缺陷；`attempt02` 保留为失败 attempt | accepted | 2026-10-22 |
| [0054](0054-fy-year-coverage-round.md) | 训练年份覆盖单因素轮：5 → 22 年、月份固定 12 个月、65×65；一手检索确认该轴在神经网络天气预测中无受控消融；预声明 (a)–(d) 出口与非科学桥 (e)；不新增下载、不改模型代码 | accepted | 2026-10-23 |
| [0055](0055-fy-year-coverage-budget-correction.md) | 年份覆盖轮预算更正：按本轮 `attempt01` 的 prepare 实测（与两臂同一条逐记录校验路径，>95 min）把每 seed 21,600 s → 43,200 s（planned 86,400 / hard 172,800）；`attempt01` 无 seed、0 GPU-h，保留为失败 attempt | accepted | 2026-10-23 |
| [0056](0056-closed-axis-reevaluation-admission.md) | 计划 0025 §4.2 的「已关闭轴重估」队列：实测六条已关闭臂本来就在同一 regime 且每臂 1.12 epoch（匹配算力不是差别），而 129×129 多年源不存在（加数据只能换 65×65，同时改网格与监督域）⇒ §4.4 要求的「不是同一族」论证按实测不成立，四条都不开轮；另立新假设须用户决定 | superseded by 0057 | 2026-10-24 |
| [0057](0057-supersede-0056-queue-is-executable.md) | 取代 0056：§4.3 的使能制度是「数据 × 匹配算力」的**合取**，0056 只核了算力一半（旧臂 1.12 epoch 属实，但数据一半未满足）⇒ §4.2 队列按序可执行；每条 ADR 须写明新制度、禁写「旧臂欠训练」动机、如实声明与旧臂不可直接比较 | accepted | 2026-10-25 |
| [0058](0058-anchor-anomaly-feedback-reeval-admission.md) | W3 队列①「气候态锚异常反馈」开轮准入：新制度 = 65×65 多年（2000–2021 train / 2022 val）× `full_grid` × 每臂 ≥1 epoch；「不是同一族」的三条论证（实例/网格/监督域/跨度全变、关闭判据 72h 绝对门在新实例上无锚臂已 3/3 过、暴露 1.12→≥1 epoch）；锚表改 48 桶；预声明 (a)–(d)+恒等桥 (e)；成本表 batch 1/2/4 = 103.8/68.2/50.3 GPU-h，**batch 由用户定** | accepted | 2026-10-25 |
| [0059](0059-s4-2016-instance-store-contract.md) | S4 首次确认的 2016 实例：store 契约无法同时表达「2016 目标帧」与「2017–2021 归一化」⇒ 不改契约/不伪造属性，改用排他派生评估容器 + 驱动内显式身份断言（身份门照真校验）、窗口选择不经 split 归属并逐字点名被跳过的唯一子句 | accepted | 2026-10-26 |
| [0060](0060-matched-exposure-budget-correction.md) | 匹配算力重估队列的预算更正：每 seed 43,200 → 129,600 s（planned 259,200 / hard 518,400，1:2:4），只改时间不动科学门；`per_seed_seconds` 在复用白名单内 ⇒ 旧 attempt 的 seed 不可复用 | accepted | 2026-10-26 |
| [0061](0061-r1-year-coverage-override.md) | R1（年份覆盖 5y→22y）的 override：按决策 0057 成文定义，只改算力**不满足**「不是同一族」（store/网格/监督域/跨度四项全不变）⇒ 用户具名授权推翻 `docs/goals/s3-year-coverage.md` 的「不调 2400 updates」；负面后果＝削弱预声明不可翻案纪律、为按算力重测开先例，先例范围限定在该轴 | accepted | 2026-10-26 |
| [0062](0062-r1-year-coverage-matched-compute-round.md) | R1 的执行设计（不重新授权，授权见 0061）：spec 落在目标长文顶部并由 `current_round_goal` 指向、两个提交路径列表分开；配对控制臂**本轮内重跑**（旧臂归一化/气候态/算力三者都不同，且 0045 白名单使旧 seed 不可复用）；驱动自包含 + 纯数据拆 support 模块；出口 (a)–(d) 逐字沿用 | accepted | 2026-10-26 |
| [0063](0063-urban-static-covariate-acquisition.md) | 城市静态协变量的获取授权、范围与集成形态：0038 只授权**天气**数据、技能 `:99` 排除城市形态 ⇒ 由用户指令新立具名授权；清单只含匿名可下载层（WorldCover/GHSL BUILT-S·BUILT-H·POP/LandScan），注册门源与 km 动态真值均不做；**独立发布产物 + `static_identity` 绑 `protocol_sha256`**（不重建 store、不加 state 通道）；只重排主计划 D3 的城市静态子情形 | accepted | 2026-10-27 |
| [0064](0064-urban-static-covariate-budget-correction.md) | 城市静态协变量轮的预算更正：按首次实测（整轮 9550 s、网络 13.49 GiB、磁盘消耗 27 GiB，三者均未触门）核对名义刻度 ⇒ **planned 21,600 / hard 86,400 维持**，网络/磁盘预算不变，只涉时间不动科学门；已发布 attempt01 的 `protocol.json` 不改（会断身份链），另记「磁盘峰值未进冻结协议」并由证据页如实标注 | accepted | 2026-10-27 |
| [0065](0065-r1-seed43-inventory-drift-replay.md) | R1 seed43 的 `incomplete execution source inventory`：根因是 `execution_files()` 对 `training/model/data` 做**活体 rglob**，另一条研究线的 commit `7065f19` 新增 2 个模块（195→197；195 个冻结条目逐位未变、无删改）⇒ 唯一补跑方式是用归档 `code.zip` 在**冻结绝对路径**下重放（非特权 namespace bind，不改比对逻辑、不重新冻结）；否决放宽判据、重新冻结（0045 白名单 ⇒ 三个 seed 全重训）与临时移走模块；教训：R2–R5 改用显式文件清单 | accepted | 2026-10-28 |
| [0066](0066-s4-primary-estimator-and-guard-verdict.md) | S4 判决口径的预注册：primary 的权威估计量是**分组均值之比**（`training/r7_climatology_skill.py:120-126`）⇒ 协议里 `interval_series` 的 per-case 写法是笔误，不是判据变更；**守门是独立合取项**且按协议字面未达（90 格：上界 25/90、点估计 78/90，12 个 >0）⇒ 依`R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md:71` 明写「t2m 有获益但未达终极 package 门」，不再以「估计量歧义无法判定」为由悬置；主动**不**重算 primary（守门已否决 package 门，为美观付费无意义），并登记「per seed ≤0」子句因只存 seed 平均而**未测** | accepted | 2026-10-28 |
| [0067](0067-execution-files-must-be-explicit.md) | 执行文件清单必须是**具名清单**，不得对 `training/model/data` 做活体 `rglob("*.py")`：R1 的 195 项里 **182 项（93%）来自 glob**，只有 13 项具名 ⇒ 无关研究线 commit `7065f19` 新增 2 个模块就让已跑完两 seed 的轮次在 `validate_protocol` 处整轮失败（seed43 exit 1 @ 20.76 s）；新驱动须按「本轮 import 闭包」生成并落进 `protocol.json` 明文、下次按明文比对；**不回头改已冻结轮次**（R1 继续走 0065 的归档重放），R2–R5 须含「新增 `.py` 不得改变清单」的反证。**proposed**，尚未机械化 | proposed | 2026-10-28 |
| [0068](0068-checkpoint-prune-dangling-audit-references.md) | 空间收缩剪除中间 `update_*.pt`：用户授权 50 个**已登记**历史轮次、排除在跑的当前轮；执行到 **33/50** 时**主动停止**（释放 164G→390G，剪除 5,224 个中间权重，12 个 selected endpoint 实测全在）。停止原因是**执行者的安全检查口径过窄**：放行前只查 `parent_checkpoint`/`selected_checkpoint` 等键名得「0 依赖」，而按**绝对路径存在性**的全量扫描（4,178 个 JSON）发现 **92 处引用指向已删文件**，集中在 `r7_74_autoregressive_20261003_attempt01`（74 处，2.45 GiB）与 `r7_73_process_supervision`（18 处，0.59 GiB），引用方是 `artifact_manifest.json` 与 `r7_m3_validation_complement_acceptance_20261002` 的 4 份审计 JSON。性质：**可追溯性弱化，非可复现性破坏**（`selected_checkpoint` 全保留、评估产物/报告/协议全未动、已登记数字仍可复现；但 6 份审计 JSON 里的中间点 sha256 无法再拿文件复算）。**不**改审计原件（那才是篡改）、不伪造重建（R-008）；剩余 17 轮暂停，恢复条件=先按存在性口径逐轮验证零悬空，并把该检查机械化进 `plan_prune` 的 fail-closed 闸门。**附记**：第二个执行者在 ADR 落盘前（01:06 < 01:45）已断点续跑完剩余 22 轮，顺序与 §4 相反但结果经审计**零新增悬空**（92 处全归属先前会话）；闸门随后落地并由 3 轮补跑 + 12 项测试验证；50 轮目标全部完成，`/data` 剩余 164G→634G（96%→82%），`outputs/` 741G→268G，selected endpoint 0 缺失 | accepted | 2026-10-29 |
