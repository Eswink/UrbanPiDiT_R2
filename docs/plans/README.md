# 计划归档

计划**分两层**存放（见 `docs/rules/artifact-storage.md` R-032）：

| 层 | 位置 | 是否提交 | 命名 |
| --- | --- | --- | --- |
| 工作态 | `.zcode/plans/plan-<session-id>.md` | **忽略** | 客户端按会话 ID 自动生成 |
| 定稿态 | 本目录 | **提交** | `NNNN-<kebab-slug>.md` |

工作态由 ZCode 的计划模式自动写入，文件名带会话 ID、可读性差，且会话结束就沉在本地。
任务完成后把计划**整理**到这里：换可读名字、补上最终结果与「计划 vs 实际」的差异。

## 为什么归档而不是直接提交工作态

采用 spec-kit 的 flow-forward 模型：已完成的计划视为**不可变**记录。
需求变化时新建一份计划，旧的那份保留 —— 它能回答「当时为什么这么想、后来怎么变了」，
这比一份不断被改写的活文档更有审计价值。

## 归档时要写什么

保留计划正文（修订后即可），并追加一节：

```markdown
## 实际结果

- **完成情况**：哪些做了、哪些没做（没做的写原因）
- **与计划的差异**：哪里偏离了、为什么（计划被证据否定也要写明）
- **验证**：实际跑过的命令与结果（含 pass/skip 计数）
- **遗留**：未解决的限制、需要人拍板的问题（如有，指向 OPEN_QUESTIONS）
```

**如实写负面结果**。本项目明确要求保留全部负面结论（见 `docs/R7_TASK_QUEUE.md`）；
一份只写成功项的计划归档没有审计价值。

## 索引

| 编号 | 标题 | 结果 | 日期 |
| --- | --- | --- | --- |
| [0001](0001-artifact-storage-convention.md) | 成品存放约定（plan / decision / goal / workflow） | 已完成（4 处与计划的差异，见文内） | 2026-09-24 |
| [0002](0002-agent-hooks-and-skills.md) | 项目专属 hooks 与 skills（含覆盖率审计） | 已完成（修 24 处绕过；1 项原理性上限如实声明） | 2026-09-24 |
| [0003](0003-goal-loop-skill.md) | goal-loop 技能 + 只读 goal 长文校检器 | 已完成（4 处与计划的差异，含决策编号 0014→0015 冲突的处置） | 2026-09-28 |
| [0004](0004-r7-main-model-v2.md) | R7 主模型 V2：RW-B 局部门控求解状态与 M3–M5 阶段 | **计划中（NOT EXECUTED YET）**：2026-09-29 Plan Mode 归档，E0–E5 未执行 | 2026-09-29 |
| [0005](0005-zcode-goal-verifier-hang.md) | ZCode goal 完成校验挂起：成因定位、恢复与本地超时补丁 | 仓库侧已完成（补丁已落盘，重启生效与冒烟留待用户选时机；4 处与计划的差异见文内） | 2026-09-30 |
| [0006](0006-v2-campaign-master-plan-and-recheck.md) | R7 V2 的主计划与每轮对表（含 N1 审计轮的目标长文） | 已完成（含 1 处计划外的工具缺陷修复与 3 处契约收紧，见文内） | 2026-09-30 |
| [0007](0007-n1-cost-supplement-repair-brief.md) | N1 成本补测失败审阅与修复轮长文（prepared） | 已完成（0 GPU-h 审阅 + 修复轮长文；修复执行待新具名授权） | 2026-10-01 |
| [0008](0008-n1-cost-coresidency-handoff.md) | N1 成本轮的共驻扩围与 goal 交接（prepared） | 已完成（只产出 objective 与承载文档；共驻政策正文与 v2 计量留给执行轮 D0/D2） | 2026-10-01 |
| [0009](0009-r7-v2-completion-and-closeout.md) | R7 主模型 V2 收尾：做完 M3/M4/M5 并关闭 #70–#75（交接给另一窗口执行） | 已归档（**未执行**：只产出本计划、四份 goal 长文与 objective、两处索引；实验/提交/推送/GPU 留待执行窗口） | 2026-10-02 |
| [0010](0010-experiment-delegation-and-time-contract.md) | 实验常设下放与时长契约：新决策 0029 取代 0021（D1–D10 施工图 + 执行记录） | 已执行；治理 c4d45cf/CI 36996404191 九步成功，审阅补正与尾核另行交付；无新实验或节点推进 | 2026-10-02 |
| [0011](0011-autonomous-execution-and-goal-handoff.md) | 实验/普通决策/节点推进常设下放与新窗口 V2 收尾 goal（0030） | 已实施；b961b3b/CI37024597813九步成功，正式objective2027字符；新实验未启动 | 2026-10-02 |
| [0012](0012-r7-m3-evidence-registration-unblock.md) | M3 证据登记的安全闸门分诊与条件化收尾 | 原分诊BLOCKED保留；用户关闭Mimosa后有限D6已登记，A/index18/账本齐，B=da4939e/CI37108664102九步成功；完整CPU2395通过9跳过，非安全/科学通过，N2a暂停不变 | 2026-10-03 |
| [0013](0013-n2a-delivery-acceptance-and-delegation.md) | N2a规定交付验收与全部授权下放接续 | 规定交付验收矩阵与全权授权已落实；本次46CPU/37阻断及18身份核对通过，精确新CI待绑定；科学paused/原failed不改，0新增GPU-h | 2026-10-03 |
| [0014](0014-v2-autoregressive-engineering-and-query-repair.md) | 独立自回归工程与显式草稿query修正 | 工程616b029/精确CI37133487340成功，3107CPU通过9跳过；真实precision8updates通过、B实际运行，未作科学结论 | 2026-10-03 |
| [0015](0015-complete-known-context-contract.md) | 原known-context/source-key验收补齐；B先封印、新C独立公平确认 | 原文缺口与0035/合法最小计划已核；B期间无源码修改，补齐/C尚未执行 | 2026-10-03 |
| [0016](0016-main-model-climatology-campaign.md) | 主模型超气候态新campaign：科学合同、数据扩围、自主研究及子代理/技能路由 | 文档与校检支持交接；研究S0–S4未执行，0GPU-h，无下载/发布/commit/push；实测验证见文内 | 2026-10-04 |
| [0017](0017-climatology-s3-long-term-handoff.md) | 主模型超气候态长期goal：从真实S3接续，自主探索与最终提示词 | 本轮仅四文档交接、0新增GPU-h；既定科学门/证据/索引/账本不改，无研究或提交推送，实际验证见文内 | 2026-10-07 |
| [0018](0018-parallel-research-workflow.md) | 并行科研能力改造与终极 goal 提示词（workflow 编排） | 新增 `parallel-research-workflow` 技能与决策 0042、刷新 §0 objective 至 3964 字符、机械同步 index63 账本；0新增GPU-h，无研究或提交推送 | 2026-10-08 |
| [0019](0019-self-iterating-continuous-goal.md) | 自迭代连续 goal 提示词（含 workflow）：分析会话自停原因（DB 证据）并让目标自己一轮接一轮跑 | 文档与本地验证中；master §0 换自迭代 objective（实测 4000 字符）、新增自迭代契约/§5 CI 纪律/§8 交接块、goal-loop 补节、决策 0043；0 新增 GPU-h，无研究或提交推送 | 2026-10-09 |
| [0020](0020-campaign-driver-workflow.md) | campaign 驱动 workflow：把「大型多轮自迭代编排」归档进仓库并写出支持代码 | 文档与支持代码交付中；ADR 0044/0045、`tools/r7_round*.py` + `tools/r7_ci_wait.py` + 90 例测试（4 个测试模块）、技能/AGENTS/master/索引同步；0 GPU-h，GPU 轮编排未端到端验证，无 workflow 运行 | 2026-10-11 |
| [0021](0021-round-interruption-hardening.md) | 编排加固：打断后的孤儿 worker 护栏 | 已落地；`tools/r7_round_liveness.py`（只读 `/proc` 扫描）+ `run_stage` 拒绝重复 spawn + 余量门读实测峰值 + `orphans` 子命令 + 17 例反证；本地 87 passed / 37 阻断 0 失败 / 尺寸计数未变；R-009 基线同步 2103/5607；未推送未绑定 CI，0 GPU-h | 2026-10-12 |
| [0022](0022-s3-architecture-next-round.md) | S3 架构方向下一轮：候选假设与结构测绘（定稿归档） | 定稿归档；0 GPU-h，只读测绘 + 文献检索，**不是 round brief**、无科学结论 | 2026-10-12 |
| [0023](0023-literature-research-subagent.md) | 文献检索子智能体（`arxiv-researcher`）与它的接线 | 定稿归档；新增一手文献通道（决策 0046），无实验、无科学结论 | 2026-10-13 |
| [0024](0024-fy-annual-coverage-round.md) | S3 全年覆盖单因素轮（4 个月 → 12 个月，2017–2021 / 2022） | 工作态计划；Phase 1–3 的 CPU 数据准备已执行，训练/评分未开始 | 2026-10-21 |
| [0025](0025-training-speed-observability-axis-reeval.md) | 训练速度优化、观测性修复与已关闭轴重估队列 | 已执行（实测 −43.8% GPU-h、−39.9% 墙钟；接受判据逐字节一致） | 2026-10-24 |
| [0026](0026-matched-exposure-reeval-and-s4.md) | 匹配算力重估与 S4 就绪（含训练年份轴重开的程序冲突） | 已归档；**交付形状在 `docs/goals/s3-matched-exposure-reeval.md` §6 更正**：workflow 写进 goal，不另挂说明 | 2026-10-25 |
| [0027](0027-urban-static-covariate-round.md) | 城市静态协变量数据准备轮（L2 城市证据落地） | 文档轮：授权（决策 0063）+ 本计划 + goal 长文 + L2 契约 + 技能扩围 + workflow 草稿；**0 GPU-h、未下载、未执行** | 2026-10-27 |
| [0028](0028-throughput-physics-campaign.md) | 吞吐前置、学习曲线定标、物理约束与有界重开的组合 campaign（六 Stage） | 已归档；只有 A6（batch 上限扫描）真跑完成，**推翻了本计划自己的线性拟合**（实测上限 14、共驻 10，16/32 OOM）；Stage B 四点判 `unsaturated`（Δ=+0.030124）；C1 前置否决 0 GPU-h；C2 三 seed 一致退化、66.993549 GPU-h 永久关闭 | 2026-11-04 |
| [0029](0029-physics-operator-candidates.md) | 物理信息算子的候选清单与开轮判据 J1–J4 | 工作态；**2026-11-11 由 0030 接续**；§4.1 的「global 65×65」前提被实测推翻（网格是区域 patch），已就地加更正块并改指决策 0110 | 2026-11-10 |
| [0030](0030-thesis-physics-informed-mainline.md) | **毕业论文主线**：物理信息混合预报（谱几何 → 平衡约束 → 正面结果），三 Stage P/Q/R | 定稿归档、**未执行**（`NOT EXECUTED YET`）；统一论证把 C1 的 10.2× 方向反、C2 的 15/15 退化、模型比真值平滑 2.9× 归为同一病的三个侧面；Q 轮改为从 p1600 端点微调（≈12 GPU-h，而非从零 242） | 2026-11-11 |
| [0031](0031-physics-informed-sequence-c-a-b.md) | 物理信息阶梯 C→A→B：β_j AMSE-analogue 归一化 + 「预训练期」制度 | **Step C 已执行（0 GPU-h）**：缺口衰减 α=0.05111（减半需 775,102× 训练量≈3.07×10⁷ GPU-h，推翻 `skip-q` 的实质含义而保留其形式）、β_j 归一化空间跨 **86.75×**（z250+z500 占 60.16% vs 无权重池化的 1/17）；**顺带修掉 `_terms` 在 `coh=−1` 返回 `inf` 的真实缺陷**（实测 108/201 命中，字面极限 3210.34）。Step A/B **未执行** | 2026-11-12 |
| [0032](0032-urban-static-covariate-ablation.md) | 城市静态协变量**三臂消融**：control / treatment / placebo（逐格空间置换） | 定稿归档、**未执行**（`NOT EXECUTED YET`，GPU-h 0）。**文献空白是关键依据**：七篇主流神经天气模型（GraphCast/Pangu/FourCastNet/FengWu/ClimaX/Aurora/AIFS）的静态输入只有地形/海陆 mask/土壤类型/经纬度，**城市建成区、建筑高度、人口密度、土地覆被占比零命中**；且这七篇**无一篇消融过静态输入本身**（GraphCast 消融 multi-mesh、FengWu 消融 replay buffer、Aurora 消融数据多样性）⇒ 五篇用静态特征的全部是「用但未消融」。**placebo 臂排除致命替代解释**：`projection` 是固定随机矩阵、模型已有 `position_features` 的 sin/cos 经纬度，故两臂阳性无法排除「多一张位置查找表」；(f) 判据要求 Δ(treatment−placebo) 72h ≥ 0.001 K。**不碰 `model/**`**：三臂只在 state-dict 层差两个 buffer，`model_code_sha256` 保持 `59340710…` 与父 pin 相等（wiring 轮的父封锁已由 `0097 D1.d` 的 reconcile 解掉）。`parameter_delta` 三臂恒 0，容量混杂结构性排除 | 2026-11-13 |
| [0033](0033-balance-diagnostics.md) | **物理平衡诊断**：把「RMSE 好但物理不可信」从案例描述变成大样本统计量 | 定稿归档、**P0 预检 + P4 冒烟已跑，正式轮未执行**。七轮负结果的统一诊断：全部改了「按什么算分」或「模型能看到什么」，没有一轮问「模型算出来的场物理上是否可信」。文献 `arXiv:2604.01454`（唯一系统评估区域 DL 模型物理平衡的工作，四个平衡 + 谱分析）的 **n = 1**（一个风暴一个 lead 一个子域），而我们有 1447 初始时刻 × 5 lead × 3 seed × 2 组受控臂。**P0 冒烟实测（6 case）给出决定性背离**：模型 `ws_ag` 随 lead 单调升 **4.58→9.75 m/s（2.1×）**，真值场**平**（2.34–2.77，无趋势）；模型 `z500` f13 噪声 **34.4→57.3（1.7×）**，真值**下降**（29.5→21.4）；比值 1.86× → **4.17×**。静力 `δ_HB` **无分辨率**（模型与真值都在 2.7–2.9%，差 ≤0.0022）⇒ 四个诊断量不等价，必须分开报。**修掉一个真实单位缺陷**：首次落盘把归一化空间当物理单位，`δ_HB` 报 −51.12、真值 std 7088；反归一化后 `z500` 均值 54757/54826、`t2m` 276.2/275.8 K。0 GPU-h（冒烟用 CPU） | 2026-11-14 |
| [0034](0034-typed-evidence-under-budget.md) | 物理证据进前向 × 训练预算：让 #79 typed evidence 在足够训练下兑现 | 定稿归档、**未执行**（`NOT EXECUTED YET`）。**九轮负结果的共同位置是事后或损失侧**，本轮又做七项只读判决测试全部为负：物理失衡与误差大小的逐 lead 相关最大 **0.2676**、与误差空间图样中位 Spearman 仅 **0.048–0.074** 且打不过 `\|∇z500\|`（剔除后残差 **+0.010**）、谱增益 `g≤−1` **恒等于 box5**（残差 **0.000e+00**）、低秩带退化到平坦常宽带、平衡子空间带 `R²_bal`=**0.110**、case 自适应 oracle `P(all)`=0.005 **低于**基线、地转投影 **20/20 RMSE 全升**（`ws_ag` 改善是构造恒等式）。**第七项反推出机制**：`corr(v_a, v_g 误差)`=**−0.79…−0.96**（残差在补偿错误的 `v_g`），根源是 **z 的 f13 噪声 2.56× 真值**（风场反而 0.44–0.89×）⇒ 要修 z 的高频且必须在前向修。**唯一有正面先例的前向装置**：#79 在参数量 **3,097,571**/FLOPs **16,776,992,640** 逐位相等的三臂下 **6h/12h 双 lead supported**（t2m **−0.0533 K**），但只跑过 **400 updates**。**剂量是主导变量**：剂量阶梯 15/15 单调、`unsaturated`（Δ=+0.030124），而已关闭的物理臂全在 **0.075–0.5 epoch**。**三臂**：A 3,286,037 / B 与 C 各 3,434,837（B≡C ⇒ 只有 `C−B` 可归因类型结构）。**段 B 零训练**：p1600 的 26 个 checkpoint（仅端点曾评分）+ K∈{2,4,6,8} 推理深度扫描。**判据**：(a)(b) 各需 ≥2 lead 三 seed 一致 + 增益 ≥0.005 K；(c) 限制声明范围；(d) 跨剂量方向；(e) z500@72h ≤5%；(f) 预检六条。**未测必须是 `null`**（`da1ed00` 修正）。成本 39–93 GPU-h，16-epoch 档位**条件启动不授权** | 2026-11-16 |
| [0035](0035-typed-evidence-three-arm-in-protocol.md) | typed evidence 的 **in-protocol 三臂重训**：回答此前任何预算都没回答过的 `C − B` | 定稿归档、**未执行**（`NOT EXECUTED YET`，GPU-h 0）。**段 B 已在开轮前给出机制信息**（`scripts/r7_s3_joint_criterion.py --dry-check` 在上一轮 15 份 CSV/provenance 上装配、0 训练、纯 CPU）：族 P（`t500,t850,u850,v850`×5=**20 格**）**10 格区间排除零、全正零负**，族 S（其余 13×5=**65 格**）**29 格排除零、全正零负**；最强不在物理输入通道而在 **z500（72h +0.69%/48h +0.66%）与 z250（72h +0.51%）**，族 P 内最强 `t500`（48h +0.46%）、`v850`（24h +0.45%）。每族共用 **1,447** case（五 lead 交集，丢 11 个只在短 lead 出现的 case），block_length 28、**1,420 个 blocks**。⇒ 这解释并纠正了上一轮 (b) 为何判 False：它只看 t2m。**但阈值与分区是看着这些数字定的，段 B 不构成确认**。**截断机制查清**：真正的绑定约束是 `tools/r7_round_stages.py:292` 的 `min(deadline, now()+per_seed)`，调 `hard_cap_seconds` 无效；本轮包络按 treated 1.47 s/upd 的 **2.05×** 定 310,000 s，并加**每 seed 三臂各 400 updates 的探针早停**（剩余包络 60% 装不下即 `refused-insufficient-envelope`）+ **partial_receipt**（上一轮 30.2520 GPU-h 空转靠人工反推）。判据按机制分族、度量量纲无关（`Δskill`）、max-statistic 同时区间控多重比较。**不做安慰剂臂**（D6：场在 `field_maps` 内现算、8 个标准化量 `persistent=False`，造安慰剂必须改 `model/` 并作废 6 个 32k checkpoint）。**实测验证**：41 条反证全过、`(f)` 六条全绿、dry-check 族 P 20/20 与族 S 65/65 装配成功、37 条阻断规则 0 失败。成本 ≈117 GPU-h / ≈84 h 墙钟 | 2026-11-20 |
| [0036](0036-loop-depth-headroom-before-architecture.md) | 动循环架构之前先量 headroom：G0 三子句决策表 + 条件式 visit-gain 分支 | **G0 已执行（0 GPU-h）**：三 seed × 1024 case × K=1..8 只读诊断，判定器输出 **`admit-g1b`** —— (a) 饱和 **3/3**、(b) k3→4 病态 **3/3**（+0.1375/+0.1102/+0.2318%）、(c) 深度可用 **0/3**（k4→8 反而变差 2.5–3.0%）；三 seed **各自独立**给出最优深度 **K=3**，k1→k3 累计改善 −0.825% 其中 **92.5%** 来自第一步，k3→k8 累计 +2.935%。**主信号是「K=3 之后单调恶化」而非「残差过放大被确证」**——(b) 幅度比门槛只高一点，而 k4→8 的衰退大一个量级。主判据是 **+6 h** 目标（`atmos_target`），72 h 记 `null`；val 覆盖 2022-01…09，缺秋冬。**G1-B 已准入但未派生**：仍需 own ADR + brief + 父迁移 + 冻结；成本估 11.1 GPU-h 未实测。`scientific_claim: false`、`test_read: false`、`r = 0` | 2026-11-25 |
