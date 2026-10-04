# main-model-v2-pivot-audit：N1 转向审计（forecast state / training objective / data regime / autoregressive exposure）

<!-- round-node: N1 -->

**N1 于 2026-09-30 执行；D2 主48/72h均unresolved，已按停止条件暂停，只提议N2d。不是目标完成声明。**

本文件是**主计划 `docs/goals/main-model-v2-campaign.md` 的节点 N1** 的目标长文，不是平行真相：
节点、账本、对表清单都在主计划里；本文件只写「这一轮怎么做」。启动后第一件事是按主计划 §3 对表
（跑 `tools/check_campaign_state.py`），再按 §4 执行。

## §0 Objective（可粘贴；实测 705 字符）

> 本轮目标：执行主计划 docs/goals/main-model-v2-campaign.md 的节点 N1——在不再新增 solver 部件的前提下，对四个嫌疑对象做 0 GPU-h 审计（forecast state / training objective / data regime / autoregressive exposure），并跑一条预声明的可证伪臂（把 Z 换成同形状冻结随机张量、其余不变，与 RW-B 配对）回答「48/72h 的恶化是否还需要一个学到的 Z 才出现」。判据与边界见 docs/goals/main-model-v2-pivot-audit.md，节点与账本见主计划。交付物：D1 四块审计（每块给 file:line 与可复核命令）；D2 可证伪臂（2 seed × 400 updates、协议在第一次优化器更新前冻结、只读 val、test 封存、≤0.45 GPU-h，执行前按决策 0021 取授权并重读账本）；D3 能机械化就落复算脚本；D4 证据页 + E 条目 + 索引记录（outcome_class: audit）；D5 下一节点提议与它的预声明门禁（只提议，不自行宣布完成）。禁止：新增 solver 部件、改已冻结判据或阈值、重跑或改写归档产物、读 test、下载数据、租 GPU、改 main、force push、关闭 #70–#75、自动进入下一节点。预算 ≤0.45 GPU-h（账本结余 20.2888 之内）；停止条件为预算用尽、审计需要新判据、审计判定「不可分辨」、或需要新数据/租 GPU/合并 main。不要自行宣布目标完成。

## §1 现状（带 file:line，2026-09-30 核对）

- 起点：主计划 `docs/goals/main-model-v2-campaign.md` 的 `current_node = N1`，上一节点 N0 已终结为
  negative（`docs/R7_72_RW_B_PILOT.md:84-94`）且复现到 1.8e-4 K（`docs/R7_72_RW_B_SUBTRACTION.md` §5）。
  启动时必须自己 `git rev-parse HEAD` 并重读账本。
- **本轮的由来（冻结文字）**：`docs/goals/main-model-v2-rw-b-subtraction.md:206-208` 的「下一动作」
  要求停止发明新模块、记录可反驳假设、转向重查三块；`docs/plans/0004-r7-main-model-v2.md:323-325`
  在同样出口上多列一条 **autoregressive exposure**，`:335` 另有「连续两次定向改动无收益：停止加开关，
  重新分析预测状态/边界/数据量」——两条都已命中。
- **可反驳假设（上一轮写下的原文，本轮 D2 直接检验它）**：*若换一个真正可区分的负控制——例如把 `Z`
  换成同形状的冻结随机张量、其余不变——则 48/72h 的恶化应当仍然出现，因为 §4.4 与 D1 都指向 (a)。*
  （`docs/R7_72_RW_B_SUBTRACTION.md:301-311`）
- **四个嫌疑对象的既有线索**（审计的输入，全部已登记，不得在审计里另立判据）：
  - **forecast state**：E-195（`docs/rules/EVIDENCE.md:321`）——递推键 `torch.cat([context, draft_tokens])`
    无来源角色标记与 mask，cell 无法区分两半；设计契约 `docs/R7_MAIN_MODEL_V2_DESIGN.md:70-82`（§3.2）
    记同一缺口与候选处置；`correction_head` 在焦点臂**从未收到梯度**（E-202）；「关掉递推后门控学会
    几乎关闭」（E-206，且同一干预在推理期与训练期相差 35–60×）。
  - **training objective**：损失权重轴是内部推理步 K（`training/r7_streaming.py:126-128`，
    `weights = linspace(1.0, final_weight, steps + 1)`），而报告分组轴是物理时效
    （`training/r7_coreasoning_compare.py:240`）；训练 target 只有 **+6h**
    （`outputs/r7_m2_segment/store/manifests/train.jsonl` 实测 186 条 `lead_time_hours` 全为 `[6]`）；
    `PROCESS_WEIGHT = 0.0`（`training/r7_rw_b_subtraction_protocol.py:31`）⇒ 过程监督从未参与训练；
    损失在归一化空间按变量等权（`training/r7_halting.py:16`），报告在物理单位逐变量
    （`training/r7_rollout_metrics.py:69`）。
  - **data regime**：M2 段 train 186 / val 22 / test 26，逐 lead 窗口 22/21/19/15/11
    （`outputs/r7_72_rw_b_subtraction/case_table.csv`）；72h 的 seed spread 20–23%（`docs/R7_B2_MULTISEED.md`）；
    气候态是 8 桶（month × hour）train-only 均值（弱基线）；归档产物的 `climatology_skill.csv`
    **仍是归一化单位**（决策 0010：保留原样，读历史文件要乘 `normalization_std`）。
  - **autoregressive exposure**：内部 K 步是自条件（草稿反馈），物理自由 rollout 从未被训练
    （计划 0004 `:325`；#64 curriculum 的「换训练时效＝零和再分配」是同向的既有证据）。
- 上一轮的两条可用读数（**非归因**，本轮的输入而不是结论）：主问句 `RW-B−(b)` 在 48h +0.785 / 72h +1.174 K
  仍为 worsened；同一「去掉递推」干预在训练期把第 1 步幅度压到 RW-A 的 0.04/0.05 倍
  （`docs/R7_72_RW_B_SUBTRACTION.md` §4.4/§4.5）。

## §2 交付物清单

| 编号 | 交付物 | 证据形态 |
| --- | --- | --- |
| D1 | 四块 0 GPU-h 审计 | 每块一节：事实（file:line）、可复核命令、以及「该嫌疑是否被支持为解释」的明确读法；**不新增阈值** |
| D2 | 一条预声明可证伪臂 | `protocol.json`（第一次 `optimizer.step()` 前冻结，含 digest）+ 逐 seed 结果 + t2m 五时效三态 + 四张成本表 + 实测 GPU-h；控制臂的**构造方式**（冻结参数怎么做）写进协议与证据页 |
| D3 | 复算脚本（能机械化就落） | 例如 data regime 的「Δ 与 seed spread 之比」复算工具；跑过的命令与输出 |
| D4 | 证据页与登记 | `docs/R7_*.md` + `docs/rules/EVIDENCE.md` 新条目 + `docs/R7_EVIDENCE_INDEX.jsonl` 记录（`outcome_class: audit`）+ brief 同步 + CI run id 与 SHA 绑定 |
| D5 | 下一节点提议 | 四方向证据强度排序 + 对 N2a/N2b/N2c/N2d 的**预声明门禁**（写明「什么读数会支持哪个节点」）；执行者只提议 |

## §3 判据与证据来源

- **工程判据**：`python tools/check_conventions.py` 37 条阻断 0 违规；`pytest -q` 无失败（skip 逐条给理由）；
  `git show --check` 与 `git diff --cached --check` 干净；干净检出用 `git clone` 复核；
  主计划对表 `python tools/check_campaign_state.py` 退出码 0。判据来源见 `docs/rules/ci-and-verification.md`。
- **D1 四块的读法（预先写定，避免事后挑解释）**：每块必须给出「**支持 / 不支持 / 不可分辨**」之一，
  并写明它据以判断的既有登记证据（上表 §1 的 file:line）。四块都不允许引入新阈值或新端点；
  若某块需要新判据才能判定，**停下报告**（停止条件 2）。
- **D2 臂的预声明读法**（对 t2m × 6/12/24/48/72h × 2 seed，沿用 #60 比较器的逐 seed 同号三态，
  `depth = 0`，与上一轮同一配对方式）：
  1. 冻结随机 Z 臂的 48h 与 72h **仍然 worsened**（逐 seed 同号）⇒ 排除「学到的 Z」为主因，
     恶化落在「锚定提案 + 门控被施加」这条通路或更外层；
  2. 48h/72h **不再 worsened** ⇒ 学到的 Z 是载体——与 D1 探针的 (a) 倾向冲突，必须在证据页里
     明确记录这一冲突并重审 D1 的相关读数；
  3. 两 seed **反号（unresolved）** ⇒ 读作「在 2 seed × 400 updates 下不可分辨」，按停止条件 3 触发 N2d 提议。
- **D1 内的可复算读数（0 GPU-h，用归档产物）**：data regime 至少算出「主端点 48/72h 的登记差值与
  同臂之间 seed spread 的比值」（分子分母都取自既有文件，**不新增阈值**，只报告该比值与它的含义）；
  training objective 至少量化「K 轴权重与物理时效的关系」与「+6h-only target」的事实；
  forecast state 至少复述 E-195/E-202/E-206 三条与「模型能看到什么」的关系；
  autoregressive exposure 至少写清「内部 K 自条件覆盖了什么、物理自由 rollout 没被训练什么」。
- **运行资格**：`queued` / `cancelled` / `skipped` / `partial` 不算通过；commit 标签只是触发意图，
  不是授权（决策 0021）。
- **科学判据（本轮不要求达到）**：≥3 固定 seed、精确配对、公平信息预算与算力报告、冻结 test、
  不确定度、负结果保留——见 `docs/plans/0004-r7-main-model-v2.md` 的 Scientific gates 一节。

## §4 实施顺序（不跳步）

1. **对表**：读主计划 §8 与上一轮 §8 的下一动作 → `python tools/check_campaign_state.py` 必须退出 0；
   记录起点 SHA 与账本（结余 20.2888 起算，重算后再用）。
2. **D1（0 GPU-h）**：四块审计，先做 forecast state 与 training objective（纯读代码与既有文档），
   再按 D3 落复算脚本做 data regime 的比值，最后写 autoregressive exposure 的「覆盖/未覆盖」清单。
3. **D2（有界实验）**：执行那一刻按决策 0021 取授权（范围/预算/产物与证据/失败与 skip 处理）。
   冻结协议 → 训练 → 评估 → 比较；**`model/` 不新增开关**：冻结随机 Z 的控制臂在 driver 层实现
   （例如把 `solver_*` 参数置为固定随机值并冻结，或让递推输出等价于固定张量），做法必须写进协议与证据页，
   并给出「其余路径与既有 RW-B 逐位相同」的核对证据。
4. **D5 提议**：按 §3 的读数给出四方向排序与 N2 候选的门禁；**不自行开跑下一节点**。
5. **D4 收尾**：证据页、E 条目、索引记录、brief 同步、CI 绑定；回写主计划 §8（节点/账本/下一动作）。

## §5 预算与停止条件

- **D1/D3**：0 GPU-h（只读代码、文档与归档产物；不重训、不覆盖）。
- **D2**：单轮自设上限 **≤0.45 GPU-h**（3 臂 × 2 seed × 400 updates 量级；上一轮同形状 4 臂实测 0.4128）；
  单次实验 ≤30 min；账本结余 20.2888 起算。
- **停止条件**（满足任一即停并向用户报告，不自行扩大范围）：
  1. 预算用尽或账本不足；
  2. 任一审计块需要**新判据**才能判定；
  3. 审计判定为「不可分辨」，或 D2 的读数落进 §3 第 3 种（两 seed 反号）——此时提议 N2d，不追加臂数；
  4. 需要新数据、租 GPU、合并 main 或任何破坏性操作。
- 目标状态 `active / paused / budget_limited / complete`；**执行者只可提议，不得自宣完成**。

## §6 明确不做

- 不新增 solver 部件、不新增 `model/` 开关（D2 的控制在 driver 层）；不改已冻结判据、阈值、端点或案例集。
- 不重跑、不覆盖、不改写 `outputs/` 下的任何归档产物与历史证据页（一律只读）。
- 不读封存 test；不下载数据；不租 GPU；不动 main；不 force push；不合并；不关闭 #70–#75。
- 不自动进入下一节点；不建立定时任务或后台续跑。
- 不把 12/24h 的小改善当作加码理由；不把 role 标记的小幅改善写成机制声明；不把 2 seed 写成显著性。

## §7 进度块

- **状态**：`paused`（D2两主长lead均unresolved，2026-09-30触发停止条件3；current_node仍N1，不自行判定完成）
- **起点 SHA**：`efe7d82d11542047897196e4d278cdb18752d6b2`；分支 `r7/weather-reasoning`。
- **开工对表（主计划 §3 六条）**：① current_node=N1、上一轮下一动作指向转向审计；②账本逐行合计
  3.7112、24−3.7112=20.2888 GPU-h；③上一轮登记与本轮 brief 结构核验通过；④本轮由 N1 派生，
  常设禁止项保持；⑤开工 `git diff -- docs/rules docs/R7_72_RW_B_SUBTRACTION.md docs/R7_72_RW_B_PILOT.md`
  为空，冻结判据无未记录改动；⑥无硬漂移，继续 D1。`check_campaign_state.py`：failures=0、notes=6；
  其中四行历史账本没有索引机器支撑，证据页指针存在，主计划 §7 已解释，并不伪称全量核账。
- **实际验证**：`check_goal_brief.py --brief docs/goals/main-model-v2-pivot-audit.md`：0 失败。
  当前 provider 的历史完成校验查询无记录；harness 校验仍仅作 best-effort，不依赖其结项。
- **已执行的 D1 读数**：只读源代码与归档 CSV；K=3 的四个草稿权重为 1/6、2/9、5/18、1/3，
  都对同一 +6h target；train 186、val 22 的 lead 全为 6。RW-B−RW-A 的 48/72h 均值 delta
  1.065106/1.580487 K，除以 RW-B 两 seed 的 max−min spread 为 1.599049/0.757468。
  这些比值只是描述，不是新阈值或因果判据。
- **D1/D3 产物**：`docs/R7_N1_PIVOT_AUDIT.md` 四节；`outputs/r7_n1_audit/audit.json`
  SHA256 `d895c96dd4936ebb75adbf9bd23bb8799185bd00b1c41482c6d8182c7a2c4a07`；源文件摘要前后未变。
- **工程准备验证**：N1 定向合并 310 passed；runner/rollout 回归 61 passed；无真实产物隔离 git clone
  全量 1795 passed / 14 skipped / 0 failed（158.97s）；14 skip=6 GPU 被显式隐藏 +8 可选真实产物不存在，
  不视为通过。driver 设备映射修正后另实跑 14 passed。
- **D2 授权**：2026-09-30 执行前 `AskUserQuestion` 用户回复「全部授权」；三臂 ×seed41/42 ×400 updates，
  现有 M2 train/val，只读 val、test 封存；一张空闲本地 RTX3090 顺序执行，≤0.45 GPU-h、≤30min；
  失败/partial/unresolved 保留全部证据即停、不自动重跑或进入下一节点。回执
  `outputs/r7_n1_audit/authorization.json`。同次授权工作分支提交与普通 push，只触发工程 CI。
- **D2实际执行**：`outputs/r7_72_frozen_z/`一次尝试status=success；三臂×seed41/42每臂400更新，
  selected400、无early stop；30val评估、510逐seed RMSE cell、255depth0行、三pair各85cell。
  protocol canonical digest `e19ef488be60136364702b1df389e5f58be7ebf3845487289139e10d30231e01`，
  code commit `c4e7e83a5deaaade1caa21fe82faa064e75b3b72`，code.zip SHA256
  `5fd26146af2a7d11016fb769d67390f5daa23a73620de9ae83e2e6cc38a35a0a`。原归档/代码不改、不重跑。
- **当前实耗与账本**：执行前再核余20.2888、campaign failures=0；D1/D3为0GPU-h，D2 GPU区间
  1292.544625543058s＝**0.3590401737619605 GPU-h**（≤0.45），whole1297.8983452636749s（≤1800）。
  ledger记0.3590，已用4.0702/余19.9298；精确加和4.070240173761961/19.92975982623804，
  显示舍入不扩授权。`merged.whole`字段实际是GPU区间，真实whole取attempt，不修改归档。
- **主读法与停止**：冻结Z−RW-A在48h为+0.063121651/−0.193231282 K，72h为+0.325881772/−0.151426589，
  两seed反号、mean=null，branch=cannot-distinguish/stop_required=true；只提议N2d，不追加seed/臂。
  同轮RW-B仍48/72h worsened；冻结−RW-B长时效改善为次对比，不替换主问句，不推导Z必要性。
- **D4/D5证据**：`docs/R7_N1_PIVOT_AUDIT.md`记录D1四块、D2全部角色/成本/digest、限制与逐项覆盖；
  E-207–E-212；索引`record:n1-pivot-audit-frozen-z-unresolved`为audit/needs-review、canonical brief同步；
  evidence_commit `33ee4702b22bfac5db303164886c1a57a3406a59`，evidence SHA256
  `e5448064f1c4612ce30026ababb3b1b5a725048da566f72be4eec7cec92b8106`；该commit文件字节与索引hash一致。
  index.ci_run_id36691526554仅绑定ci_commit efa410b的工程实现，不冒充最终登记提交的CI。
- **登记前验证**：CPU隐藏GPU实跑campaign/goal/index/conventions/D3/protocol六文件 **330passed**（20.51s）；
  E条目机械重数212、覆盖度145/20/47、确认211/推测1；37阻断0违规、brief0失败、campaign0失败/6notes，
  `git diff --cached --check`与证据commit的`git show --check`干净。最终登记四文档快照的干净clone
  全量 **1795passed/14skipped/2warnings**（157.29s）；此后仅证据术语/指针澄清，实验代码仍不变。
  四视图指参数/FLOPs/吞吐/内存（case表是覆盖而非成本）；独立文档审阅确认停止/身份/预算自洽，
  非触发分支因果措辞只按已冻结协议限域、N2b提议出处是执行前c4e7e83证据页§7而非新科学判据；
  这两项及四视图措辞在证据澄清提交33ee470中修正追溯说明，固定长文判据/实验代码未变。
  澄清后登记定向campaign/goal/index/conventions四文件 **166passed**（20.59s），结构/索引/账本重检无失败。
  D1排序目标/暴露/数据/状态（前两项共享结构证据、不是因果排名）；N2d门禁已触发，其他节点不执行。
- **独立工程复核与未齐项**：终态全集合、共同case、protocol/contract/source元数据交叉一致；
  归档#60纯元数据重算paired JSON逐字节相同。validation中途deadline不覆盖、finalizer未强制全集合；
  两seed30次evaluation峰值继承末臂training值255424000B，独立eval峰值缺失，内存视图不能标PASS。
  三项不热改/补测，仅列未来driver使用前工程前置项；source/checkpoint/cache bytes未再次认证。
- **工程CI**：c4e7e83的run36690064571失败，本地重现R-021 marker34/actual35；只修文档标记，
  `efa410b812cada42ccf0eba3d1f47afd54b3aecb`的run36691526554 completed/success、九主步骤绿。
  精确efa410b的干净clone1795passed/14skipped/2warnings（161.04s），skip原因同上；远端test计数未取得。
  一手API访问2026-09-30，run/jobs响应在`outputs/r7_n1_audit/ci/`；最终登记提交的CI尾记录只写本进度，
  不改冻结证据页而产生digest自引用。Mimosa报scanner_enobufs，扫描无结论，不是安全通过。
- **最终登记验证（2026-09-30）**：`6b7875b8da3e047f3bddf35a830deee67064be07`已普通SSH推送至
  `r7/weather-reasoning`，索引/brief/账本/轮次结构均核验；工程CI **36699197294 completed/success**，
  精确head_sha匹配、九主步骤成功。一手来源
  [run API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/36699197294)、
  [jobs API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/36699197294/jobs?per_page=100)，
  访问日期2026-09-30；web-researcher初读为in_progress（未当PASS），curl终态原响应排他归档于
  `outputs/r7_n1_audit/ci/36699197294{,_jobs}.json`，SHA256分别
  `f2a1c17e7d01a94599b9f8c661a1547f49c00b4149eb48478a0ddadc98f1f516` /
  `e90cbc02fd5b8ac919fabb40205363ee4c8f56441b109052467d199624445d45`。
  精确6b7875b的干净clone实跑 **1795passed/14skipped/2warnings**（162.16s），test计数是本地值，
  不是未取得的远端日志计数；skip理由仍为六CUDA隐藏、八可选真实产物缺失，未合成回退。
  七条ledger数值一致（四条历史无索引notes保留）、13条index/brief同步、37阻断0违规、whitespace干净；
  工作树仅原两个未跟踪工具文件，无其它变动，冻结实验代码/历史证据/数据路径未动。
  D1–D5覆盖对表在证据页§8，独立eval峰值缺口未标PASS；本次CI只认证工程，不认证机制/科学/安全。
  本条是终止登记的尾记录，不改证据页/index digest；尾提交自身CI只在最终回复报告，不再次递归写文档。
- **成本补测准备（2026-09-30；当时GPU未执行）**：独立只读验收裁定停止合规，但D2独立eval内存缺口阻断完整验收；
  原实验授权不等于接受缺项。用户回复「允许跑实验，反正就是把它搞得完整。」明确要求补齐证据。
  只准备原六selected400 checkpoint×五val时效的独立计量，0训练更新，拟≤0.09GPU-h/≤600s；
  用原归档code.zip、source/model/data/checkpoint/BUILD_COMPLETE身份，独立新输出，原归档及冻结证据页不变。
  执行前具名范围/预算/归档重放例外/失败停止的问题未收到回答，没有补测授权回执、protocol或GPU运行。
  只读CPU身份核对和fake CUDA守卫测试已做，准备记录见`docs/R7_N1_SUPPLEMENTAL_COST.md`；
  初跑195passed，独立复核守卫修复后定向232passed；都不等于实验通过。新增实耗0，
  账本4.0702/19.9298、原主unresolved与N2d提议保持。
- **一次具名成本补测（2026-09-30）**：用户AskUserQuestion答「授权上述一次补测 (Recommended)」，
  范围原30val/0训练/≤0.09GPU-h/≤600s，新输出、归档成本重放例外、失败即停不自动重试。
  用户手动腾出GPU1后复核空闲/source/checkpoint/BUILD_COMPLETE，使用计量commit1e03f820、原code.zip；
  新protocol canonical `1ce9222321bfe6d799b0f86d7bc0ff4de127d451edaa0e5e8a45ca5a4a3ffc22`在CUDA前冻结。
  首项seed41/RW-A/+6h零baseline、peak39590400/46137344B、22case/17RMSE精确重放；准备lead12h时
  基线非零守卫在evaluator前抛错，failed1/30，无result/四成本表，不自动重试或诊断再跑。
  实耗21.443860329687595s＝0.005956627869357666GPU-h、whole22.372659532353282s，failed照记。
  N1原+失败0.3649968016313182≤0.45；ledger显示4.0762/19.9238，精确4.076196801631319/19.923803198368685。
  新证据`docs/R7_N1_COST_SUPPLEMENT_ATTEMPT.md`、E-213/E-214与独立audit/blocked记录；原证据页/hash保持。
  基线失败的具体allocated/reserved字节数未记录、原因未知；独立只读确认部分有效但完整成本仍BLOCKED。
  单个旧PID清卡授权两次在发信号前失败（API缺失/旧PID消失），没有向其它项目进程发送信号；GPU1由用户手动腾出。
- **失败补测独立复核/登记**：只读复核81输入byte pins、六checkpoint原始字节与118归档源码hash匹配；
  首项RMSE/ACC/skill逐字节一致、provenance只差elapsed，零基线与UUID绑定有效；全集合guard拒绝1/30。
  独立结论FAILED/完整成本BLOCKED，未认证cache张量或持续无竞争；回执`outputs/r7_n1_cost_failed_independent_review.json`。
  新证据commit`33d57d67fc8b247262e807aa74510cb5828792f8`/SHA256
  `4b350357af20ce95bc9c2e31b7411f83108fda4dc4366e681b61e0cbecdac9fc`；单独索引
  `record:n1-evaluation-cost-supplement-failed`为audit/blocked，原记录字节不变，canonical brief共14条同步。
  index.ci_run_id36718455666只指计量准备commit1e03f82，不冒充随后登记CI；自身发布CI只在最终回复报告。
  登记后CPU五文件232passed（20.83s），37阻断0违规、两brief0失败、campaign0失败/6历史notes、whitespace干净；
  没有把工程测试或CI当成本通过，也不因结果failed删记录或不记实耗。隔离登记快照全量CPU
  1861passed/14skipped/2warnings（162.05s）；6CUDA显式隐藏+8可选真实产物缺失，skip不算通过，
  2warning为既有Lightning未挂Trainer日志调用。此后只追加进度记录，代码未改；远端test计数未取得。
- **未做**：其余29项独立eval内存、完整成本验收及目标完成裁定；没有补测重试或扩大诊断。
  原validation/finalizer缺口未热改、训练逐位复现未做；显著性、跨季节区域、matched-Generic、M3/M4/M5/确认轮未做。
  未读test/下载/租GPU/改model或main/force/merge/关闭#70–#75/动原归档。
- **下一动作**：N1仍paused/current_node=N1；当前一次补测授权已用，失败即停。仅提议原因审阅与单独具名
  修复/重试决策，不能把预算余量或这次许可当重跑授权；原unresolved/N2d提议不变，不自宣完成或推进节点。

- **后续适用指针（2026-10-04）**：本页中「不关闭 #70–#75／本轮不关闭 issue／关闭是收尾轮的事」为当轮禁止项，如实保留不改写；自决策 0037（`docs/decisions/0037-issue-closeout-window-scope.md`）起，#70–#75 的有条件关闭按 `docs/goals/v2-issue-closeout.md` 执行（单一 `Closes` 提交、精确 CI、非 force ff 推 main、匿名复核；关闭≠科学成功）。
