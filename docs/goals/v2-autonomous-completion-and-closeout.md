# v2-autonomous-completion-and-closeout：N2a 补全至 N5 的自主执行交接

<!-- round-node: N2a -->

**状态：paused（2026-10-02）。N2a新23项val补测及完整汇总已实跑，原any-unresolved出口触发，N3/N4/N5前置未解除；最终goal未裁定完成。**
它承接主计划末尾的决策 0030 授权扩围，与计划 0009 的科学任务相同，但不沿用旧逐轮许可/预算闸门。
主计划是唯一权威；本文件是总体 goal，首次执行节点为 N2a。进入后续节点时先登记前一节点证据和
next-action，再一致更新主计划/current_round_goal/round-node；可由本文件派生独立新版节点长文，
不回写旧失败/历史目标。不是用一个 N2a 标记冒充后续所有节点均已完成。

## §0 Objective（可粘贴；单段，实测 2027 字符）

> 你在 /data/esw/UrbanPiDiT_R2 的 r7/weather-reasoning 分支继续，先核 git status/HEAD，通读 docs/goals/v2-autonomous-completion-and-closeout.md、docs/plans/0009-r7-v2-completion-and-closeout.md、docs/goals/main-model-v2-campaign.md 与决策0030。用户只把控总体实验方向；该方向内的实验、时长预算、普通工程/方法决策及节点推进已全部常设下放，在当前goal内连续完成N2a补全→N3→N4→N5/关闭，不逐次请示、不逐轮等用户触发；每节点前后对表、核冻结前置/出口、登记证据再推进，独立审阅是质量检查而非新许可，不绕失败停止线。D1补N2a/M3：核六个已完成400-update checkpoint、原code.zip、source/model/sidecar身份及已有7/30评估；优先全新输出/独立协议补缺23项val，不无故重训、不改旧1800秒协议或失败终态；旧checkpoint按归档代码重放，需新时长支持用版本化补测驱动、不改冻结常量、不绕digest。新汇总逐项区分旧训练/旧7项与新补测来源和全额成本，交付全17变量×6/12/24/48/72h、显式seed配对、四成本视图与新登记；不能合法复用时自主预注册新方案，不伪造拼接。D2执行N3：复用既有组件构造同结构/同已知输入/无过程语义的matched-Generic V2，实现可微两步物理rollout；同父checkpoint两臂×两固定seed比较L6与L6+0.5L12，按冻结要求验证预测喂回、future只改loss、梯度回传、时间更新、reasoning_steps/rollout_steps分轴、resume/finite/BF16；不用no_grad评估rollout训练，不扩4/8/12步，算力不等照报，模型改动标model-digest-change并合法处理旧checkpoint身份。D3执行N4：旧Ours/matchedGenericV2/ProcessV2三臂至少三预声明seed，冻结primary/配对/单位/成本口径，全17变量×五时效照报；三seed不称显著性，adaptive默认不启动，冻结有效准确率—成本前沿成立才另立协议。D4执行N5：补#71年末/闰年/UTC跨日/东西经/batch多日期/odd-grid-padding/内部K同valid_time反证，补#72现役位置化反馈到达latent与process_reader非零梯度；逐issue登记#70–#75证据、DONE/NEGATIVE/BLOCKED、负面结论和未做项。按0009/0030具名关闭授权，前置与验收证据齐备后写Closes提交，先工作分支精确SHA主CI九步成功，再非force ff推main并匿名核issue状态；不合并main、不为全部关闭弱化测试，证据不足保持开放/BLOCKED，关闭不等于科学成功。每实验在长文写死范围/产物/失败skip处理、planned_seconds软预算及hard_cap_seconds宽松硬上限（默认约2倍），运行前冻结protocol/digest；整轮墙钟含CPU前置/训练/评估/启动间隔/清理，超软预算继续并记soft_overrun_seconds，硬截断failed/budget_limited全额记账且不算通过。无总GPU-h上限，cap/used/remaining只记账；承载新实验的CI timeout-minutes×60须高于硬上限并留收尾余量。GPU默认共驻，每次启动/spawn只读核UUID/余量，不信号邻居。总体方向改变，或需付费/租卡、新或多年度下载、数据发布--write、独占、main合并、破坏性操作时按保留边界停下取授权；普通排障和推进不要再问用户，force push/绕hook仍禁止。R-006/R-028/R-054/R-009、身份校验、原始数据/归档只读、scientific_claim:false与limitations不豁免；不读封存test、不据曝光test选配置、不事后改冻结阈值/端点/案例集，不建立cron或脱离当前goal后台续跑。逐节点实跑conventions、campaign、当前goal结构、定向及全量pytest、index/brief、空白检查，提交推工作分支核精确SHA九步CI；skip/cancelled/queued/partial/failed不算通过，工程/实验/科学结论分开。收尾逐交付物核实际运行、digest、测试、CI和issue终态，报告九项影响、失败与未做事项，不自行宣布最终goal完成。

## §1 现状与本轮由来（2026-10-02）

- 交接起点 `256ef256c85ff48369681fd43f31cfd9c079eda3`，工作分支 `r7/weather-reasoning`。
  该 SHA 的 CI `36997601485` 九主步骤成功是上一治理轮事实，新窗口开工重新核当前 HEAD。
- 主计划 state：N2a/budget_limited、cap/used/remaining=24.0/4.7018/19.2982、
  budget_mode=accounting-only；本交接不推进节点、不增加成本。主计划 §8 的治理授权扩围是本轮由来。
- M3 工程 D1–D4 已落，六 training 各 400、七评估已完成，但完整覆盖/paired/四成本缺，见
  `docs/goals/n2a-m3-process-supervision.md:168-193` 与 `docs/R7_73_PROCESS_SUPERVISION.md` §6/§7。
  原 GPU 成本 0.49725426027008024h 与 whole1805.1085775829852s/预算未完全兑现不可改。
- 当前只读核对 campaign=0 failures/4 历史 notes；四条无索引支撑行只记 note，不冒称机器核数。
- N3/N4/旧关闭长文 prepared 与计划 0009 的工作清单是输入，不是已经实现/执行的证据。
  其旧 0021、≤30 min、1/2.5/8/24 GPU-h 等授权/预算文字不再是现行闸门；冻结科学配置、种子、
  端点、案例集与失败事实依然有效。不得照旧 goal 调度后再事后改其 protocol。

## §2 交付物清单

| # | 交付物 | 必需证据形态 |
| --- | --- | --- |
| D0 | 新窗口开工对表与节点具体协议 | 实际 HEAD/status；主计划六项 recheck；原产物 digest/冻结判据；每次 planned/hard 两个数字、范围/产物/失败处理、训练/评估前 protocol digest |
| D1 | N2a/M3 完整补测 | 六 checkpoint/训练及旧七评估身份清单；缺23项新 val 记录；全30项覆盖/510 RMSE/原528案例口径、seed配对/四成本；新结果来源 manifest、完整失败/成本历史，不回改旧attempt |
| D2 | N3/M4 实现与对照 | matched-Generic 同结构契约；可微两步 rollout 与反证；两臂×两固定 seed、λ=0.5、同父 warm-start 的完整结果与四表；额外 forward/backward 及算力不等限制 |
| D3 | N4/M5 最小确认 | 旧Ours/matchedGeneric/ProcessV2三臂≥3seed；冻结评估/primary/单位/配对；全17变量×五lead、三态/四表/成本；adaptive“不启动”或有门槛证据的独立协议 |
| D4 | N5 #71/#72 追加反证及六issue裁定 | 时间/地理/内部K测试与现役反馈/reader梯度反证；新证据页；六项 verdict/限制/digest，不靠删测试关闭 |
| D5 | 条件成立后的具名关闭 | 0030限定禁令解除；前置/验收齐；Closes工作分支提交精确SHA绿CI九步；非force ff main；匿名API终态/URL/访问日；不足项保留开放/BLOCKED |
| D6 | 每节点登记与最终交付审计 | 新证据页/E/index/canonical brief、逐行账本、主计划state与上一轮next-action；代码/协议/产物digest和精确CI；逐项prompt-to-artifact审计，工程/实验/科学分别报告 |

### D1 优先级与归档重放

1. 先核旧六 checkpoint 的 opaque hash/contract、所选400更新、三臂/seed/val源和原 code.zip。
   现行代码相同也不能跳身份校验；旧模型重放用它归档的代码。
2. 优先只补缺评估，新输出排他创建；补测驱动归档自己的源码/协议，并显式记录实际被调用的归档
   evaluator 源身份。不在原实验目录里 resume、finalize 或手工填完整 manifest。
3. 新合并对象引用旧训练、旧七评估及新23评估逐文件 digest，分别记录采集时间/代码/成本。
   原 attempt 仍 failed；新结果能接受哪些覆盖、无法恢复哪些峰值/字段逐项说明，不能用旧峰值代填。
4. 若既有 finalizer 无法表达跨尝试来源，新建版本化聚合/补测契约，保持原接受条件和反证不弱化。
   无法合法复用时，先说明具体不兼容/身份/覆盖原因，执行者可自定新的预注册方案，不无理由重训。

## §3 判据与证据来源

本文件**不新增科学阈值，不外委科学判据，不事后放宽门槛**：

- 总体任务/依赖：`docs/plans/0009-r7-v2-completion-and-closeout.md`；机制/评估预声明：
  `docs/plans/0004-r7-main-model-v2.md`、`docs/R7_MAIN_MODEL_V2_DESIGN.md`。
- N2a：旧 M3 长文 §3、`docs/R7_65_PREDIAGNOSTIC.md`、决策 0028 三类时刻/sidecar contract；
  工程已通过不是完整三臂实验判定，原失败页只读。
- N3：`docs/goals/n3-m4-autoregressive-rollout.md` §3；预测喂回、future仅loss、时间轴分离、梯度
  与精确窗口；λ=0.5/两步/配对父 checkpoint 的冻结要求保持。改模型时不能忽略旧 checkpoint digest，
  合法的版本兼容/导入设计须另行记录并验证，不等于放宽加载校验。
- N4：`docs/goals/n4-m5-confirmation.md` §3、决策 0023 的 matched-Generic 前置、#60 显式配对；
  已冻结端点不变，运行前尚未写定的端点/容许退化按既有纪律先写定，不在看完结果后改。
  三seed只给一致性；正ACC不等于正MSE skill，不同物理单位不直接平均。
- N5：`docs/goals/v2-issue-closeout.md` §3 的证据与不弱化要求；新关闭用决策 0030 的具名范围
  和决策 0003 的非force ff机制。旧 closeout 中“预期0028”不是编号许可，新增ADR取实际下一可用号。
- 现行授权/时长：`docs/decisions/0030-autonomous-execution-and-direction-boundary.md`；每节点
  对表：决策 0025 与 `docs/goals/main-model-v2-campaign.md`；诚信/CI：`docs/rules/ci-and-verification.md`。
- 结果声明：ENGINEERING PASS / EXPERIMENTAL SUPPORT / SCIENTIFIC SUPPORT 分开。
  partial/失败不等于 negative/mixed；没有完整对照不能据已测短lead判断三臂 forecast 效果。
  最终独立年份/季节测试是后置事项，不自动扩本路线到新数据/新总体方向。

## §4 实施顺序（新窗口执行，不跳步）

1. **D0**：核 status/HEAD 和授权；通读本长文/0009/主计划/0030；实跑对表，检查旧源/归档身份，
   准备当次具体协议。治理有未提交遗留先独立处理，不混实验实现；无关未跟踪文件不提交。
2. **D1**：以 N2a 为当前节点，新增版本化补测与证据协议、CPU反证/工程CI后运行补测。
   定向原尺度/三时刻测试无需重复实现；已有训练无故不重做。失败即停止该attempt并留痕，另立
   新方案而不恢复旧attempt；完整实验门槛不足不以绿CI推N3验收。
3. **登记 N2a**：新证据页/E/index/账本和当前工作态登记；独立复核实际覆盖与来源，前一节点
   下一动作/主计划state/下一长文 marker同步，重跑campaign再进入N3。审阅不是新许可。
4. **D2/N3**：matched-Generic与可微rollout实现/反证→模型身份兼容验证→工程CI→冻结协议后实验→
   全结果/成本登记与科学读法。无收益如实负结果，不加长unroll“练到赢”。
5. **D3/N4**：必要前置已实际可运行→冻结评估/三臂协议→工程CI→确认实验→adaptive门槛决定。
   不把matched-Generic计划文本当已可构造；不靠已曝光test选λ/增updates。
6. **D4/D5/N5**：前三节点必要证据实际登记后补反证，逐issue验收和未做项，依0030条件化禁令
   同步新执行长文而不改历史页；具名Closes提交先工作分支精确CI九步绿再ff main，匿名核issue。
   main非ff拒绝即停不force，BLOCKED项不能靠关闭计数掩盖。
7. **D6**：每节点及末尾的交付审计，核实际文件/运行/digest/测试/CI/API；回写主计划并持续对表。
   执行者写真实结论与未做项，不自行把整个goal或open-issue历史目标设complete。

## §5 预算与停止条件

- **本交接产出轮**：0 GPU-h；只写治理与目标，不运行任何训练/评估，账本不增加。
- **新窗口实验**：无总 GPU-h 上限，旧24/8/1/2.5等历史额度不是现行闸门。每个具体实验先由
  执行者按旧实测时间/覆盖成本自定数值 `planned_seconds`、`hard_cap_seconds`（默认约2倍），
  写在该节点长文并冻结protocol；尚未定数时只能准备，**不能直接启动实验**。本总体目标不凭空
  分配所有节点时长，也不替代单次具体协议。
- 整轮计时包含 CPU 前置/训练/评估/进程间隔/清理；软超继续、overrun如实记录；仅硬上限因
  时长截断，失败/budget_limited全额记账。需要CI承载才调整其时限，严格高于硬cap留收尾余量。
- 失败停止当前attempt，不静默续跑或重复旧授权范围；在0030范围内的修复/新补测协议可自主
  定义。真实错误、身份校验/共驻余量不足、无法解释的覆盖/账本漂移须先排障，不能改门槛。
- 已冻结“不可分辨/不能归因”停止线仍有效：停止相关探索、完整登记，不靠增seed/算力练到赢；
  新总体方向、付费资源/租卡、新或多年度下载、--write发布、独占、main合并、破坏性操作交用户。
- 无后台守护/cron或会话结束后自建新任务；终止时留可审计进度。最终 complete 由用户或运行时
  独立完成校验裁定，不把工具退出0、审阅无缺项或全部issue关闭当作代理自行裁定权。

## §6 与 planner 草案的差异

治理实施计划源自只读 planner 草稿，经主链收敛为 `/tmp/r7_0030_plan_20261002.json`（verified=true）。
原草稿混淆了 objective 的400（planner）/4000（goal）上限、出现不存在的0029文件名和不准确
“0006门禁”说法，均以实际0030/0009/既有checker纪律纠正；科学阈值没有由planner产出。
本长文与objective由主链按已冻结指针和实际M3失败覆盖编写；草稿不是实验或完成证据。

## §7 明确不做

- 不覆盖旧实验目录、原attempt/协议/checkpoint/失败证据；原始/interim/processed与legacy只读。
- 不读封存test清单/数组作评估或选择、不新增read_count重新封存曝光M2 test、不新数据下载/发布。
- 不事后改科学阈值/端点/案例集、不删或弱化测试、不用partial当negative或CPU/CI代实验。
- 不force/merge main/--mirror/删除默认分支/绕hook；N5之前不写main、不关闭issue。
- 不非本实验信号/腾卡/冻结邻居、不租卡、不建立定时或后台守护；超方向先交用户。
- 不宣告SOTA，不把三seed写显著性，不把六issue关闭或普通节点完成当整个goal已完成。

## §8 进度块

- **状态**：paused；起点`caea510f25cf7eb77bd65638773c561e8e92c25c`，当前N2a补测已完成，原any-unresolved出口触发，不推进N3/N4/N5。
- **已完成**：status/HEAD与四份指定目标/计划/0030已核；campaign0失败/4历史notes，当前goal结构0失败；
  两GPU只读余量充足，无邻居信号。无关未跟踪两文件保持不提交。
- **新节点长文**：`docs/goals/n2a-m3-validation-complement.md`，软1800/硬3600秒，整轮计时、0训练/23缺val；
  主计划current_round_goal指向新页，原M3失败与冻结页/1800协议保持。只读归档审计与planner已分别委派。
- **D0及补测工程**：独立原109文件/24历史checkpoint/六selected400/七val身份审计已完成，成功回执SHA
  `020b0647434db6cf28acd9ac07116672e7ec34ddc7c49dd5dec2506e8f9718bc`；当前真实源只读prepare前置已实跑。
  独立新protocol/归档隔离driver/逐文件来源聚合与反证已落地，最终定向CPU123passed/13.04s、0skip。
  两项写路径缺陷修复后限范围复核resolved；AST基线1210函数/3112断言，只同步规模marker，不放宽阈值。
- **完整CPU工程验证**：2395passed/9skipped/2warnings，233.74s；跳过不算通过。37阻断/campaign/两当前
  goal/index/brief/compile/空白均通过；43旧源码/目标与两无关文件hash保持。精确新工程CI尚待核。
- **实际N2a运行**：工程011ab4c的CI37036869969精确九步success后，仅一次新prepare/run；canonical74d05ab3…09dd5，0训练，缺23val全success。跨来源30eval/510RMSE/528case与三pair各85齐，新GPU0.17899193391850632h、whole700.5363800507039s/overrun0；原109文件/失败成本不变。
- **实际停止出口**：三pair改善/恶化/unresolved为19/45/21、21/36/28、3/44/38，any-unresolved暂停；attempt paused且覆盖完整，不是failed/partial或forecast成功。N3/N4/N5必要出口未解除，不进入后续实验/关闭。
- **最终封印**：独立stdlib67380检查/4.3066s/exit0，276新+109旧文件hash前后不变，全覆盖/三来源/单位/进程/配对/四成本/最终整轮核齐。seal0810665b…97a72/verification576a72b2…2ab79仅accepted-engineering，科学暂停与advancefalse不变。
- **版本化收尾阻塞**：新证据页已保存，本地账本全额新0.1790/总4.8808/余19.1192；但证据commit被Mimosa L3拒绝（只读legacy12high/1low，覆盖不完整），整条提交命令未执行/HEAD011ab4c。未改归档/安全配置或绕过，新audit只旁侧pending草稿，canonical index仍17条，精确登记CI未取得。
- **未做**：N2a正式证据提交/索引及精确登记CI；N3 matchedGeneric/rollout、N4确认/adaptive、N5追加反证/正式验收/Closes/main均未执行。六issue实查open。
- **最新终态验证**：完整CPU2395passed/9skipped/2warnings/223.13s、定向221passed/17.50s；37阻断/两当前goal/index17/compile/空白通过，campaign0fail5notes。原109/新276/开工43pins不变；七文档本地暂存未提交、HEAD仍011ab4c，main未动。逐要求20项审计确认已执行与未做，不把审计本身当完成判据。
- **下一项受阻的具体动作**：安全闸门与归档只读冲突须用户裁定后才能证据commit/正式登记CI；科学any-unresolved出口亦未解除。当前只保存与审计已运行N2a及停止事实，不改变冻结出口、绕安全闸门或自行宣布最终goal完成。

- **登记阻塞诊断接续（2026-10-03，0新增GPU-h）**：获批计划0012已完成本地原事件投影与12处有限静态分诊，工程scanner_enobufs/high拒绝/Stop超时区分；原109/新276seal与43科学pins重核一致，未改安全hook、ignore、归档或外发。官方安装包没有已证实可用的非降级L3处置接口，按硬停点保留BLOCKED，没有重试commit/push、A/B/C登记或新CI。CPU532passed/38.34s/0skip，首次仓内tmp反证24failed/508passed保留，只改仓外tmp后同集合通过，未改测试/阈值。诊断包 `outputs/r7_m3_registration_diagnostic_20261003T024214Z/`，正文 `docs/R7_SECURITY_SCAN_TRIAGE.md`；canonical17、账本与N2a/paused不变。下一具体工程依赖是官方支持纠正及实际门禁放行；若需安全政策范围例外必须另有明确授权和接口，不能将本轮诊断批准当例外。N3/N4/N5、issue关闭与最终goal完成仍未做，issue状态未重新联网查询。

- **用户关闭工具后仅D6恢复（2026-10-03）**：用户先答「我对其进行明确授权」，随后说明「我已经关闭了Mimosa，可以继续进行」。只读核本地插件开关false，执行者未改安全配置且用户差异不提交，不外发支持工单、不升级或新深扫；无官方处置/L3安全通过结论，历史拒绝与覆盖不足保持。最新继续限定为D6，不复活总体goal自动N3/N4/N5推进或关闭。
- **本地正式证据绑定**：补测页A=`1de3a672d859b42efa7a5fac3293c2832df4a6ee`/blob SHA256`17c446253af0e6df838cf60ef02800b7d64c6cce0490b981ddfc5e22789b4a85`已冻结；index17→18单增audit/blocked，canonical brief同步、账本新行只增record。原失败/原109/新276/366归档/43科学pins不变，0新增GPU；本次完整CPU2395passed/9skipped/2warnings230.96s，skip非通过。B提交/精确登记CI仍待绑定，不能以旧工程CI覆盖；N2a/paused、any-unresolved与最终goal未自裁保持。
- **当前未做与边界**：D6精确CI等待实际工程运行，不等待支持工单选择；N3 matchedGeneric/可微rollout、N4确认/adaptive、N5追加反证/正式验收/Closes/main仍未做，不借恢复D6推进。仓外新回执 `/tmp/r7_m3_registration_resume_u8t215bd/`。

- **D6登记完成后的当前边界（2026-10-03）**：正式B=`da4939e613eb1a0b39f6303142ffd7aa6ab8657c`已推工作分支，CI37108664102/job111162098631精确SHA completed/success九主步骤成功，匿名run/jobs API访问2026-10-03及回执SHA256`b9184dc513cddea1d9c8748b09e98b2720194668ead9ed64a396007055e161bb`见补测长文。index18与canonical brief/账本record齐，campaign0fail4历史notes；532定向37.67s/0skip，完整2395passed/9skip/2warnings230.96s为本地实跑，skip非通过/远端计数未取。31项原目标审计只解除D6工程缺失，原failed/成本/weakcoverage及any-unresolved保留；17核查全过，回执SHA256`7ec1a3a03966da8af2f6fcb3b473b2efb61722252ef9f2e9a0a16301ac27134a`。
- **仍未做/下一研究动作**：N3/N4/N5、main/issue关闭、MetPy与新GPU、完整真实17通道CPU端到端、数据下载/发布未做。Mimosa增量Write仍真实拒绝不安全候选，纯只读候选按反馈修正成功，未改设置或取得L3/官方安全裁定；不发支持工单。只追加C活进度/plan尾并独立核C的CI，不回写A/index。工程D6已登记不能解除科学暂停，下一研究动作仅独立审阅negative/mixed/any-unresolved及总体方向，不在本轮推进，也不自宣最终goal完成。

## §9 Objective 副本（与 §0 相同，便于交接）

> 你在 /data/esw/UrbanPiDiT_R2 的 r7/weather-reasoning 分支继续，先核 git status/HEAD，通读 docs/goals/v2-autonomous-completion-and-closeout.md、docs/plans/0009-r7-v2-completion-and-closeout.md、docs/goals/main-model-v2-campaign.md 与决策0030。用户只把控总体实验方向；该方向内的实验、时长预算、普通工程/方法决策及节点推进已全部常设下放，在当前goal内连续完成N2a补全→N3→N4→N5/关闭，不逐次请示、不逐轮等用户触发；每节点前后对表、核冻结前置/出口、登记证据再推进，独立审阅是质量检查而非新许可，不绕失败停止线。D1补N2a/M3：核六个已完成400-update checkpoint、原code.zip、source/model/sidecar身份及已有7/30评估；优先全新输出/独立协议补缺23项val，不无故重训、不改旧1800秒协议或失败终态；旧checkpoint按归档代码重放，需新时长支持用版本化补测驱动、不改冻结常量、不绕digest。新汇总逐项区分旧训练/旧7项与新补测来源和全额成本，交付全17变量×6/12/24/48/72h、显式seed配对、四成本视图与新登记；不能合法复用时自主预注册新方案，不伪造拼接。D2执行N3：复用既有组件构造同结构/同已知输入/无过程语义的matched-Generic V2，实现可微两步物理rollout；同父checkpoint两臂×两固定seed比较L6与L6+0.5L12，按冻结要求验证预测喂回、future只改loss、梯度回传、时间更新、reasoning_steps/rollout_steps分轴、resume/finite/BF16；不用no_grad评估rollout训练，不扩4/8/12步，算力不等照报，模型改动标model-digest-change并合法处理旧checkpoint身份。D3执行N4：旧Ours/matchedGenericV2/ProcessV2三臂至少三预声明seed，冻结primary/配对/单位/成本口径，全17变量×五时效照报；三seed不称显著性，adaptive默认不启动，冻结有效准确率—成本前沿成立才另立协议。D4执行N5：补#71年末/闰年/UTC跨日/东西经/batch多日期/odd-grid-padding/内部K同valid_time反证，补#72现役位置化反馈到达latent与process_reader非零梯度；逐issue登记#70–#75证据、DONE/NEGATIVE/BLOCKED、负面结论和未做项。按0009/0030具名关闭授权，前置与验收证据齐备后写Closes提交，先工作分支精确SHA主CI九步成功，再非force ff推main并匿名核issue状态；不合并main、不为全部关闭弱化测试，证据不足保持开放/BLOCKED，关闭不等于科学成功。每实验在长文写死范围/产物/失败skip处理、planned_seconds软预算及hard_cap_seconds宽松硬上限（默认约2倍），运行前冻结protocol/digest；整轮墙钟含CPU前置/训练/评估/启动间隔/清理，超软预算继续并记soft_overrun_seconds，硬截断failed/budget_limited全额记账且不算通过。无总GPU-h上限，cap/used/remaining只记账；承载新实验的CI timeout-minutes×60须高于硬上限并留收尾余量。GPU默认共驻，每次启动/spawn只读核UUID/余量，不信号邻居。总体方向改变，或需付费/租卡、新或多年度下载、数据发布--write、独占、main合并、破坏性操作时按保留边界停下取授权；普通排障和推进不要再问用户，force push/绕hook仍禁止。R-006/R-028/R-054/R-009、身份校验、原始数据/归档只读、scientific_claim:false与limitations不豁免；不读封存test、不据曝光test选配置、不事后改冻结阈值/端点/案例集，不建立cron或脱离当前goal后台续跑。逐节点实跑conventions、campaign、当前goal结构、定向及全量pytest、index/brief、空白检查，提交推工作分支核精确SHA九步CI；skip/cancelled/queued/partial/failed不算通过，工程/实验/科学结论分开。收尾逐交付物核实际运行、digest、测试、CI和issue终态，报告九项影响、失败与未做事项，不自行宣布最终goal完成。
