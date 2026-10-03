# n2a-m3-validation-complement：保留失败历史的 M3 独立验证补全

<!-- round-node: N2a -->

**交付状态：N2a规定工程/实验/登记已完成；科学状态：paused（any-unresolved出口触发）。**
23项补测及30总覆盖、D6均已齐，适用0030；不是原1800秒attempt的resume，也不将交付完成等同科学成功、N3放行或最终goal complete。
原 `outputs/r7_73_process_supervision/`、失败证据页与原目标保持只读；只有身份审计、工程门禁与新协议冻结后才启动23项补测。

## §0 Objective（单段；实测794字符，≤4000）

> 按 docs/goals/n2a-m3-validation-complement.md 补全主campaign的N2a/M3：D0核起点HEAD、六项对表、六个400-update checkpoint及原code.zip/source/model/sidecar/旧7项val身份；D1用归档代码和全新输出/独立协议补缺23项val，0训练更新，不改原1800秒协议、digest校验或failed终态；D2新来源manifest逐文件区分旧训练/旧7评估/新23评估，报全17变量×6/12/24/48/72h、显式seed配对、单位正确RMSE/skill/ACC与四成本视图，全额保留原失败和补测成本；D3定向与全量pytest、conventions/campaign/当前goal/index/brief/空白实跑，工作分支精确SHA主CI九步成功后运行；D4新证据页、E/index/账本、独立只读审阅及逐要求审计后按冻结出口进入N3。新实验planned_seconds=1800，hard_cap_seconds=3600，整轮含CPU前置/评估/启动间隔/聚合清理；软超继续记overrun，硬截断failed/budget_limited全额记账。按0030自主普通决策/实验/节点，不逐次询问；GPU共驻每spawn核UUID/余量，离线禁网。禁止读封存test、改原数据/归档/旧失败、下载发布/付费独占、main/合并/force/关闭issue、改冻结判据/阈值/端点/案例集、删弱测试或后台续跑；scientific_claim:false与limitations不豁免。身份不符或评估失败停止本attempt并登记，不伪造拼接或拿工程绿代科学接受。不能合法复用时在原方向内另立具名新协议，不无故重训。最终goal由用户/运行时独立裁定，不自行宣布完成。

## §1 现状与由来

- 起点：`caea510f25cf7eb77bd65638773c561e8e92c25c`，分支 `r7/weather-reasoning`；仅 `.zcode/agents/web-researcher-backup.md` 与 `.zcodeignore` 无关未跟踪，保留不提交。
- 由来：总体目标 `docs/goals/v2-autonomous-completion-and-closeout.md` D1，主campaign当前N2a；0030取代旧逐次实验/推进许可。
- 原attempt的六训练各400/selected400，7/30val、119/510 RMSE、131/528 case；完整paired/四表未有。原GPU 0.49725426027008024h、whole1805.1085775829852s及1800秒计时缺口不改。
- 冻结源事实见 `docs/R7_73_PROCESS_SUPERVISION.md` §5–7；完整身份由独立只读审计核实际字节，不从计划或摘要推定。
- 旧M3长文是原失败历史。本轮不回写其停止条款；主计划 `current_round_goal` 指向本新长文，仍为N2a。
- 本轮真实源的只读 `prepare_r7_local.py` 已在GPU前实跑：17通道/240时次/65×65，train186/val22，source SHA与原一致；报告 `/tmp/r7_m3_complement_source_readonly_preflight_20261002.json` SHA256 `ef4ed74883b5b45694f5a7f7de92cbe47ad42e226b9b3c6546c3ae684478f1ac`。没有--write、下载或重建。历史M2独立只读conversion报告/单独授权回执仍待有限定位，不能把本次报告冒称历史报告；原written-local-cache记录与原第二阶段授权另核并如实限定。

## §2 交付物清单

| # | 交付物 | 实际证据形态 |
| --- | --- | --- |
| D0 | 起点与复用资格 | 实际status/HEAD、六条recheck、source/zip/模型/sidecar/六checkpoint及旧7项的逐文件SHA与契约、selected400报告 |
| D1 | 缺23项独立val补测 | 独立冻结protocol/digest、新驱动源码归档、新排他输出；每项归档evaluator身份、新进程、UUID/余量与allocator证据，0训练更新 |
| D2 | 可审计跨attempt汇总 | 来源manifest区分old-training/old-evaluation/supplement；30项、528 case、510 RMSE格、全17变量五lead、同seed配对、RMSE/ACC/物理单位skill及四成本表 |
| D3 | 工程门禁 | 定向及全量pytest pass/skip实数，37阻断/campaign/brief/index/compile/whitespace，精确工程SHA主CI九步success |
| D4 | 登记及出口审计 | 新 `docs/R7_73_VALIDATION_COMPLEMENT.md`、E/index/canonical brief、独立复核、全额账本、精确登记CI及prompt-to-artifact核对 |

## §3 判据与证据来源

- M3科学/工程接受条件只引 `docs/goals/n2a-m3-process-supervision.md` §3、`docs/R7_65_PREDIAGNOSTIC.md`、`docs/R7_MAIN_MODEL_V2_DESIGN.md`、决策 `docs/decisions/0028-m3-process-sidecar-and-time-contract.md`。
- 数据与训练/选择配置全部继承原协议；seed41/42、三臂、K4/400、selected400实际核后引用，val五lead案例22/21/19/15/11。新协议只改变补测编排/时长/来源，不改变case或指标。
- 新调用按原code.zip的实际evaluator执行，不用当前模型绕旧digest；source SHA、BUILD_COMPLETE、训练contract、fixed inverse和sidecar身份强核。
- 结果来源可跨attempt逐文件引用，但不能伪造原finalizer/原attempt成功。未测项/无法恢复峰值明确缺失；旧evaluation不以训练峰值代填。
- 显式seed配对与四成本依据既有#60比较器，是同seed两臂差而非seed41对seed42。M3本身的skill CSV已为物理RMSE；先按原接受器核实际单位，不对已正确M3数字盲目二次缩放。决策0010的旧归一化缺陷只适用于其历史产物，原CSV不改。不同物理单位不直接平均，正ACC不等于MSE skill。
- 原 `training/r7_m3_results.py:196-205` 的 `descriptive_outcome` 与原协议reporting已冻结：任何unresolved cell要求暂停、不推进。新汇总复用此判定，不因0030的普通推进权而取消；若触线，新attempt完整覆盖但paused，N3/N4/N5必要前置仍未解除。
- R-006/R-028/R-054/R-009、决策0030及 `docs/rules/ci-and-verification.md` 保持。工程、完整实验覆盖与科学支持分开；两seed不称显著性。

## §4 实施顺序

1. 六项对表、只读归档身份审计、规划JSON契约及两处事实抽查。
2. 版本化补测驱动/聚合契约及反证；不修改冻结1800常量或原加载身份逻辑。
3. 全工程门禁与工作分支精确SHA九步CI；再冻结新protocol与源身份。
4. 单次23项新进程val，只读原store/source/sidecar/六checkpoint；任一失败停止attempt，保留中间证据。
5. 跨来源30项完整汇总/seed配对/四成本与负面结果；独立只读审阅、登记及出口审计。
6. 完整覆盖与N2a必要出口有证据后同步上一轮next-action/主计划state/新N3长文，重新campaign对表再推进；质量审阅不是许可。

## §5 预算与停止条件

| 项 | 数值与执行语义 |
| --- | --- |
| planned_seconds | 1800，软预算；超出继续并记 `soft_overrun_seconds=max(0,whole-planned)` |
| hard_cap_seconds | 3600，宽松硬上限；整轮从新protocol prepare最早CLI进入起包含准备身份/源码归档与冻结、prepare至run启动间隔、运行身份重核/归档提取、23评估、imports/启动/聚合/清理；用冻结同机boot单调anchor，需预留本实验子进程清理时间 |
| GPU-h | 无总上限；连续实际GPU区间全额记账，不只累计成功评估kernel；原失败0.49725426027008024h另列且合计 |
| 共驻 | 固定一次选定UUID；每spawn只读余量门槛至少2048MiB并覆盖预声明owned峰值安全余量；外部PID只记不发信号 |
| 失败/skip | 身份不符、余量不足、指标/案例不齐或任一评估失败即停本attempt；skip/cancelled/queued/partial不接受，不自动resume旧失败 |
| 运行通道 | 本地离线实验，不新建承载GPU实验CI；CPU主CI只验证工程、不冒充GPU实验，故不批量修改workflow时限 |

1800软/3600硬根据已有每项新进程val计量与缺23项规模留足余量，不是科学阈值。旧1800协议不改。总体方向改变、需要付费/租卡、新或多年度下载、数据发布--write、独占、main合并或破坏性操作时停止相关动作取用户决定。已冻结不可分辨/不能归因等停止线仍有效，不靠加预算练到赢。

## §6 与planner草稿的差异

只读planner返回围栏JSON，主链收敛为 `/tmp/r7_n2a_complement_plan_20261002.json`，实际校验 `verified:true/fence_stripped:false/failures:0`。抽查原worker154–195的评估输出与来源、原results196–211的暂停/1800接受条款后，纠正草稿中不存在的文档/比较器指针、种子互配错误、回改冻结证据页、删除preflight证据、余额作为停止闸门及自行complete等建议。新增专用complement模块不动旧M3，结果进新证据页；实际1800软/3600硬已由主链先写定，原暂停线保持。planner不制定科学判据，草稿不是证据。

## §7 明确不做

- 不训练/新增seed臂/改selected checkpoint，不改原协议、旧失败终态、原输出或历史证据页。
- 不写data/raw/interim/processed与legacy；不读封存test、不以已曝光test选择配置、不下载/发布数据。
- 不改冻结判据、端点、案例集，不删/弱化测试；不把fixture作天气真值。
- 不signal邻居、不独占/付费/租卡，不cron/后台续跑；不main/合并/force/关闭issue。
- 不把覆盖补齐、工程CI或独立审阅当科学成功或最终goal完成。

## §8 进度块

- **状态**：paused，23项补测与完整跨来源汇总已运行，原any-unresolved停止出口触发；补测0.17899193391850632GPU-h已全额记账，最终封印已核；D6证据/index/账本已正式登记，登记B精确CI37108664102九步成功；科学暂停与最终goal未自裁保持。
- **六项开工recheck**：①N2a与N1尾下一动作一致；②账本逐行算术24.0/4.7018/19.2982、四历史无索引notes照保留；③N1已登记证据commit可达；④新目标派生总体goal的N2a/D1并引用冻结文档；⑤ `git diff -- docs/rules` 为空，历史M3科学页/协议未改；⑥campaign实际0fail/4notes，无需修机械漂移，先身份审计不启动实验。
- **验证路线**：只读数据库显示当前provider最近三条goal校验均error/约600秒；这是客户端路线事实，不判本任务失败，也不自判完成。
- **D0实际独立审计**：`/tmp/r7_m3_complement_audit_20261002_final_s6ciyxh4/results/audit_receipt.json` SHA256 `020b0647434db6cf28acd9ac07116672e7ec34ddc7c49dd5dec2506e8f9718bc`，audit_valid/technical_artifact_reuse_eligible=true，无pin mismatch，完整原实验仍不接受。六selected400及24历史checkpoint CPU身份/contract已核、旧七val119格/131case、相关zip/model/source/sidecar/109文件前后不变。首次自写审计过严clock采样断言failed另存、未改冻结标准；failed receipt SHA `5fed3f3365fe19f45039791dabb0db699c8cca549598fb3070b76280b41f659d`保留，纠正按归档driver两次采样时序，0GPU新增。
- **数据资格读法**：本次GPU前只读prepare报告已实跑；旧发布written-local-cache可核，full-auto授权声明在事前commit `ecdae0d5bd0c70799f91d0ddfa02bcd04bf76e6f`已版本化、ADR0010将M2纳入第二阶段。有限范围未找到历史单次原始用户回执/独立转换前报告，不据缺文件推无许可，亦不倒填历史报告；本轮只消费既有完成cache、不新发布--write。
- **具名输出**：`outputs/r7_m3_validation_complement_20261002/`（冻结前核不存在）；旁侧接受回执 `outputs/r7_m3_validation_complement_acceptance_20261002/`保留独立审计/只读preflight/开工pins/逐要求清单，非实验原目录。
- **新聚合反证实跑**：跨来源聚合及旧M3 driver/receipt CPU合集实际 `114 passed in 17.65s`，日志 `/tmp/r7_m3_complement_aggregate_targeted_20261002.log`；全集合拒绝、原失败保留、物理单位不二次缩放、phase snapshot与最终成本引用、any-unresolved暂停有反证。这仅工程子集，尚未代表完整新驱动/全量/CI或GPU结果。
- **版本化收尾阻塞**：新证据页commit/索引登记/精确登记CI被安全闸门阻断；实际文件/成本证据已保存，未伪造commit或以旧CI覆盖。旧失败不回改，不能继续实验。
- **最终驱动与路径复核**：prepare入口同boot单调anchor、归档隔离、写前拒非法receipt与evaluation祖先symlink、完整来源及真实aggregation接口反证已落地；主链最终定向 `123 passed in 13.04s`（0skip），日志 `/tmp/r7_m3_complement_targeted_final_20261002.log`。原两high限范围只读复核均resolved，不当许可。中途编辑fixture签名不齐的33errors保留，已由最终实跑覆盖。
- **基线/规模实测**：1210测试函数/3112断言/133文件，新增加34函数/137断言；只同步checker与文档。新最大文件543行/函数87行，R019=0/R019b310/R02046/R02144/R02231/R02328，仅机器marker同步，无阈值或例外放宽。
- **全量工程实跑**：`CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m pytest -q -p no:cacheprovider` 最终2395passed/9skipped/2warnings，233.74s，日志 `/tmp/r7_m3_complement_full_cpu_20261002.log`。六GPU条件测试/三可选真实fixture跳过，不算通过；两个Lightning无Trainer日志warning照保留。37阻断0违规/campaign0fail4notes、两当前goal0fail、index/brief17records、compile与staged/unstaged空白均exit0。43旧源码/目标及两无关文件hash保持，新具名输出确认尚不存在。
- **精确工程CI**：`011ab4cdc3b4fa9f6671a22962a2cf3227998c66` 的run37036869969/job110937121161 completed/success九步齐，匿名API访问2026-10-02；接受回执SHA256 `12c4e33c1af986b20387779d38134ff6fe9b1989bb666e3e4e28a3abdf81e5cb`。工程绿后仅一次prepare/run，未resume或重训。
- **实际协议/运行**：独立canonical `74d05ab36a5c15451893e20f9cadcaecdac62170717af61a3e7524af57509dd5`；新zip `e44ef233b0455c38279549d22c1d864d604400933cc5a61210d3591df37abb43`（81源），归档原80成员隔离执行。prepare4.993492s，从其入口含全部间隔、CPU前后置与聚合清理；23新eval/23不同PID均success，三基线全零。原六训练/七eval+新23=30eval/510RMSE/528case，255汇总/三pair各85齐。
- **冻结出口实测**：future−off=19改善/45恶化/21unresolved；input−off=21/36/28；future−input=3/44/38。any-unresolved为true，attempt=`paused`/finalized/coveragecomplete，非partial/非budget_limited，advance_next_node=false。同归档比较器metadata精确重放一致，回执SHA256 `9ef4d22ca91fe3b7d34f3ef6795a6017e09e8a83c809282af018999c928729a9`；没有GPU第二次重放。
- **实际全额成本**：新连续GPU644.3709621066228s=0.17899193391850632h，最终整轮700.5363800507039s，soft_overrun=0；聚合前stage694.8902271008119s不冒称最终whole。原失败0.49725426027008024h/whole1805.1085775829852s保持，M3两attempt共0.6762461941885866h/2505.644957633689s。old109及43旧源码/目标hash再次不变。
- **最终封印实跑**：stdlib独立67380项检查/4.3066s，exit0/accepted-engineering；276新文件与109原文件hash/集合前后不变，代码/input/三来源/30项四指标/per-case/配对/13表/所有进程退出/prepare连续时钟核齐。旁侧seal SHA256 `0810665b861a3f2206b53556a6878b179e68f2923e560e11c7ef906af6797a72`、verification SHA256 `576a72b26efd4148ddc9a6ffc59347c8e3bddb0b979968f6f9fcd5167ca2ab79`；stage pending保持原样，final receipt实际hash与成本只在旁侧补齐。工程接受不取消scientific_paused，不当许可或goal判定。
- **报告复核与登记阻塞**：有限独立复核关键指标/配对/旧109hash/成本/精确工程CI/六open一致，纠正011新driver与旧归档evaluator身份混称。新证据页 `docs/R7_73_VALIDATION_COMPLEMENT.md` 已保存未提交；Mimosa L3 PreToolUse对单独证据commit拒绝，报告只读legacy中12high/1low且覆盖不完整，整条提交命令未执行。未改归档/安全hook或绕过；HEAD保持011ab4c，新索引仍未登记/CI未取。旁侧阻塞回执保留，新record只pending草稿，不填伪造evidence_commit。
- **本地全额账本**：0.1790新行、总4.8808/余19.1192，精确总4.880706349145538/余19.11929365085446；新行因commit阻塞暂无index，campaign第五note如实保留。旧四历史行不改，不把显示余额当许可。
- **阻塞终态实跑**：最新完整CPU2395passed/9skipped/2warnings/223.13s，定向221passed/17.50s；37阻断/两当前goal/index17条/compile/两侧空白通过，campaign0fail5notes，新未登记成本行note明确。实际新276/原109/开工43与两无关文件hash再核不变；原科学CSV与安全配置未改。七文档已本地暂存未commit，HEAD/origin工作分支均011ab4c，main仍dafd22e。
- **下一项受阻的具体动作**：安全闸门与只读归档边界需用户裁定后才可冻结证据commit/正式索引及精确登记CI；N3/N4/N5另因原冻结暂停出口BLOCKED，不启动或关闭，最终goal不自判完成。
- **安全阻塞分诊（2026-10-03，0新增GPU-h）**：按获批计划0012只做本地诊断，SQLite只读核三原始工具记录，工程commit/push的scanner_enobufs、后续12high/1low拒绝和Stop的ETIMEDOUT分别保存，不混为完整扫描。12处high的九源码hash/有限输入调用链核齐，low位置仍未知，不执行归档或认定全误报；随包文档未提供已证实的L3归档风险例外或非降级可靠性修复接口，因此按硬停点保持BLOCKED，没有重试commit/push、改hook/ignore/归档或外发。新包 `outputs/r7_m3_registration_diagnostic_20261003T024214Z/`；分诊见 `docs/R7_SECURITY_SCAN_TRIAGE.md`，原109/新276 seal集合与43科学pins重核一致。相关CPU同集合532passed/38.34s/0skip；首次仓内临时fixture导致24failed/508passed，改为仓外临时目录后重跑，未改测试断言。未进入A/B/C登记，不新增完整pytest/CI冒称通过；canonical17/五notes、账本与科学paused不变。当前具体工程依赖是官方支持的规则/可靠性纠正及实际门禁放行；若只能改政策须另有明确授权和正式接口，文字风险接受不等于放行。
- **用户恢复D6（2026-10-03，0新增GPU-h）**：用户先答「我对其进行明确授权」，再说明「我已经关闭了Mimosa，可以继续进行」。只读核用户及仓库本地开关false，执行者未编辑安全配置，用户配置差异/两无关untracked不提交；不再外发支持工单。此次仅正常证据/index/账本/工作分支git/SSH/CI，原官方处置未取得、未跑新L3/深扫/升级，不伪称安全通过或漏洞修复；历史12high1low/覆盖不完整拒绝保留。若仍拒绝就停，不换路。科学paused/advancefalse与N3/N4/N5/main/关闭边界不改。
- **证据A与本地索引真实绑定**：正常提交A=`1de3a672d859b42efa7a5fac3293c2832df4a6ee`仅新补测页，blob SHA256`17c446253af0e6df838cf60ef02800b7d64c6cce0490b981ddfc5e22789b4a85`；正式单增record:m3-validation-complement-complete-paused，audit/blocked/scientific_claim=false，index18与canonical brief同步，旧17条原文及历史pending草稿不变。账本只增加record指针，不新增成本或修旧失败；当前B提交/精确CI待实际执行，不冒称旧工程CI覆盖。
- **恢复登记完整CPU实跑**：2395passed/9skipped/2warnings、230.96s，exit0；六CUDA条件与三可选真实fixture跳过非通过，两Lightning无Trainer警告保留。此前本会话定向420passed/53.87s/0skip与3399只读产物核查通过，不合并为独立科学样本；MetPy、GPU重放与完整真实17通道CPU端到端未执行。恢复前原109/新276/366归档/43科学pins核齐，新独立回执/日志在 `/tmp/r7_m3_registration_resume_u8t215bd/`，不重跑旧seal、不写旧/新运行目录。
- **D6正式登记及精确CI（2026-10-03）**：B=`da4939e613eb1a0b39f6303142ffd7aa6ab8657c`已正常git/SSH推工作分支，run37108664102/job111162098631的head_sha精确匹配、completed/success，九主步骤全部成功；[run API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/37108664102)、[jobs API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/37108664102/jobs?per_page=100)匿名curl访问2026-10-03。run响应SHA256`c1e8892c3fab72cbfb7fef69702193e15c1b52b0e14774b1cb15c1a17b7832be`、jobs响应SHA256`127447175ec123d5a97aacb549a4f8cadce97fb61f9ff7774cbfc00e3d3f782b`、核验回执SHA256`b9184dc513cddea1d9c8748b09e98b2720194668ead9ed64a396007055e161bb`。远端pytest日志/计数未取得，本地2395/9/2不冒称远端计数；旧工程CI仅原作用域，正式登记CI在此绑定而不回写冻结页/index。
- **登记出口与逐要求审计**：532定向passed/37.67s/0skip；37阻断0违规、campaign0fail4历史notes、两goal0失败、index18/brief与两侧空白通过。纯只读18checks/359活跃Python AST与E235统计核齐，原109/新276/366归档/43科学pins、用户配置与两无关文件不变；12文档独立只读复核无必修，不当安全/科学/goal校验。31项原prompt-to-artifact更新只解除D6工程缺失，保留原failed/whole超cap及所有weakcoverage，审计17checks全过，回执SHA256`7ec1a3a03966da8af2f6fcb3b473b2efb61722252ef9f2e9a0a16301ac27134a`。
- **安全与收尾边界**：用户开关false之外，Write增量扫描仍拒绝仓外动态subprocess核验草稿，草稿未写入；按反馈去掉所有子进程后的纯只读版本正常Write成功，不改安全设置或换通道，成功提交不当L3通过。只完成D6，不外发支持工单、不新GPU/MetPy/真实17通道端到端/下载/test解码、不推进N3/N4/N5或main/关闭。C仅活进度与计划实际结果尾，另核自身CI并在外部最终回执交付，不递归改A/index。下一研究动作停在独立审阅已登记negative/mixed/any-unresolved及总体方向裁定；本次不自宣最终goal完成。
- **N2a规定交付验收（2026-10-03，计划0013）**：原目标D1–D6和本页D0–D4逐项已有证据，结构正确+negative/mixed原本允许验收；节点工程/实验/登记已完成，当前科学paused不表示又缺训练/23val/D6。原单次failed/budget_limited/whole超旧cap仍不改，新独立complete-paused不伪造原finalizer成功；index18/audit-blocked保持，三seed/MetPy意向/e2e和更强analytic只按原范围分清未做，不事后造完成硬门。
- **全权授权与自主选择**：用户原话「N2a是否已经已经完成，如果未完成，则计划完成，并且明确所有授权全部下放给你。」完整记入计划0013；作为0030接续确认，N2a完成/收尾所需执行、工程修复、实验安排、预算、验证和普通决策自主承担，不逐项询问。执行者据规定交付已经齐，选择0新增GPU-h的六文档验收收尾，不是缩减用户授权或等待授权；不改冻结事实/判据，不借此推进N3–N5、main/关闭或自行设置最终goalcomplete。
- **暂停界限与版本**：N2a交付完成、原failed、新complete-paused、any-unresolved/advancefalse及账本四历史notes同时保留。收尾C=`20230567eb4272b9fd70a73e0f6e5dc8af34ed63`的CI37109369425/job111164135531精确九步success，已有仓外回执SHA256`2a451777c4d4a2791fd0adfbd9201380427df2b1ec18ea2ec9517d0aa152f1dc`核到；历史A/B/Cpending字段是写入时点，不回写冻结页/index或误造新缺项。本次文档自己的精确CI另核，不能用旧绿覆盖。
- **下一动作（2026-10-03，用户新方向0032）**：本辅助监督假设终结且冻结paused出口保留；不补23val，不改旧advance=false。独立autoregressive-exposure路线进入N3，执行 `docs/goals/v2-remaining-stages-exploration.md` 的matched-Generic、合法父导入与可微两步训练；此为新路线选择，不伪称旧M3放行或成功。
