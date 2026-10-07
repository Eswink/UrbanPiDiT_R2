# S3：等病例曝光的训练顺序干预小试验

<!-- round-node: S3 -->

状态 active / 真实执行中，2026-10-07。唯一主计划
`docs/goals/main-model-climatology-campaign.md`；承接原目标—最终评分诊断的跨病例负响应，
不重做单Jan80更新或精确反算，不进入S4。

## §0 Objective（单段，实测1176字符）

> 在 docs/goals/s3-case-interleaving-pilot.md 接续S3/index61/21.9433GPU-h、test未评分/r0。D1承接docs/R7_S3_OBJECTIVE_FORECAST_DIAGNOSTIC.md的同Jan改善而其余train/四dev退化，冻结不同数据制度假设：原2021四train、各20次曝光、80updates/batch1，blocked每例连续20次对interleaved四例循环20轮，同外锚original0权重/freshAdamW/seed41/原12步deep-K/K4/FP32/同globalLRrecipe；原bulk是randperm不是seasonblock，不能把人工blocked当历史原因。D2只用归档code.zip与新TEMP薄wrapper，完整source/preflight/BUILD_COMPLETE/窗口/单位/新wholecontract/普通loader/外锚case与schedule资格先核，实际CPU真tinyProcess训练/投毒/计数/同初始化/完整contract/错scope/失败反证与独立prelaunch齐后才真实训练。D3两臂完整80更新并原四train+四val开发例17变量×6/12/24/48/72h×full/interior_1/edge_1同train气候态评分，endpoint80固定不选val；原0同cohortincumbent从已保存合法分数复用，不更新旧结果/恢复父optimizer。D4独立savedfact/math/identity/cost审阅、全部正负登记及index/brief/master/工作分支精确SHA必要CI；一seed八病例是诊断不证明泛化/机制或科学接受，unsupported止本实例，不自动加seed/dose/开关/完整val。准备planned3600/hard10800/reserve180；实际整轮planned5400/hard14400/reserve180先冻，soft超继续记，真错或hard停attempt全部留。GPU1默认共驻每启动/spawn核即时UUID、maxknown2491416576B与新peak加2GiB，不signal邻居；无总GPU-h许可上限，cap20/remaining仅会计。科学门只引用docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md。不得改旧数据/产物/归档/冻结门/用户配置/安全或绕身份，不paid/rental/exclusive/main/release/issueclose/破坏性/force/cron或会话外续跑，不自行宣布最终goal完成。

## §1 承接、事实与起点

- 起点branch `r7/weather-reasoning`，HEAD `3533483df9d8cafd9064ddf561335e64bfb3d102`。
  前一证据冻结于 `0befd262f971318cacbcae627d62a16a6e5723b4`、SHA
  `ff02acac4fb87160bbb1eafd82bdbd1465fd4e724b5c5ded1afec0ac55a35a58`；
  index61，累计21.9433GPU-h，2023 test科学评分0/r0/S4未执行。
- 原目标—最终配对诊断真实完整：Jan拟合目标下降58.88%、83/85最终cell改善，但其余三train/四dev
  目标全部上升；val最终241/340cell退化、t2m18/20case-lead退化。所有24train/evalfinal hash同。
  已确认同病例响应与跨病例负响应分离；记忆/干扰/数据制度的因果仍未判定。
- 本轮只研究等病例曝光下的ordering，属于0030/0038方向内普通方法选择。
  人工blocked不是历史bulk训练重现；活跃long runner为randperm(seed+epoch)+cursor，
  `training/r7_long_rollout_runner.py:90`，不能宣“原失败原因已证明是季节块”。
- 原dataset2140完整窗口、220排除；`data/r7_long_rollout_dataset.py:116` 保存manifest到long index映射。
  本轮四原病例不是原全数据训练，不能把4例训练效果推为五年数据制度的结论。
- 旧PE/scale/loss/two-step负面不复活；typed B→C曾supported、相对incumbent未决，
  不将其历史改写为全部negative。科学与机制接受仍分层。

## §2 交付物与证据

| 编号 | 实际交付 | 必需证据 |
| --- | --- | --- |
| D1 | 等病例multiset、不同序列与共同初态/recipe/数据冻结 | 每臂80、每例20、schedule digest、source/archive/newcontract/普通loader外锚 |
| D2 | TEMP实现、完整实际CPU与独立prelaunch | 真tinyProcess训练/未来目标投毒/optimizerstep与计数/完整coherent错body反证、fixedcodehash |
| D3 | 两真实训练臂与固定endpoint80评分 | 逐stepreceipt、标准checkpoint及state、所有17×5×3/train/dev/climatology分数、坏变量/undefined、无test |
| D4 | 独立保存事实与全正负登记 | cases/math/schedule/identity/process/gate/cost首末hash、证据commit/index/brief/ledger及精确CI |

完整val/新数据/全年确认不是本轮交付；未做必须如实记录，不用CPU/CI或issue关闭替代。

## §3 判据与预声明解释

最终科学门只引用 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`。
本轮没有新增科学确认门槛或容差，只有以下前瞻开发期比较与停止文字：

1. 同seed41，比较interleaved对blocked，以及两个新臂对共同update0：逐病例原目标与最终物理分数均报，
   不平均不同单位；端点80固定，不按val选更早端点。
2. 数据制度开发支持的必要描述：四dev pooled t2m/full五lead对blocked及共同0均不退化、至少一lead严格改善，
   同病例集的u10/v10/mslp五lead/full相对共同0及blocked均不退化；undefined不能充通过。
   单seed/小病例即使全同向也只支持本实例接续，不构成正式候选或最终科学接受。
3. 若仅train改善或dev不支持，登记negative/mixed并终结此ordering实例，不默认增加dose、seed或扩大val。
   对人工blocked获益但未优于共同0时，只报告人工顺序对照效果，不说主模型性能提升。
4. 同global learning-rate schedule并不使每个病例获得相同LR时间位置；序列干预包含该耦合。
   必须报告逐病例LR曝光和顺序，不能将效果独占归因于遗忘/梯度cosine/季节因果。

前一既定climatology mean identity
`c633df585b1894704d942e24305047a74ff7af2460cbc24ca2ebe7be1948a205`；
仅fit原train2400 stamps/16 buckets×150、单位/归一化/区域相同。
共同update0分数只能从前诊断原0八病例合法payload复用，hash先核；数据若改变本轮拒绝，不偷换基线。

## §4 顺序与依赖

1. 核index61/前证据/独立终态/账本/精确CI；Explore定位API，planner实际不可用后自规划同契约校检，独立内容审核。
2. 只写全新TEMP薄training wrapper复用归档原_update与评分/数据资格；新contract绑定四case全体、每stepplan和计数。
   原singlecase完整body不能直接冒充新contract，不改旧加载校验或restoreoptimizer/cursor/RNG。
3. 完整CPU合成测试及独立固定包prelaunch，no真实字段/GPU；旧失败保留，新范围资格逐项核。
4. 实际wholeprotocol排他冻结后CPUprepare核source/preflight/BUILD_COMPLETE/model/training/norm/window/三端点普通身份。
   每臂共同original0（不是Jan拟合80）模型权重、freshoptimizer、同seed与原recipe。
5. 统一父调度blocked80、interleaved80及两臂endpoint80×原八病例目标/最终评分、CPU读取/endhash。
   每spawn即时GPU余量，ownedwatchdog只持handle直接child；无val反馈、test或oracle选择。
6. 独立完整savedfact/numerical/identity/cost，result-freeze登记正负、双campaign/goal/index/brief/branch精确CI。

## §5 预算与停止

| 阶段/量 | 冻结数字 |
| --- | --- |
| TEMP准备整轮 | planned3600/hard10800/reserve180秒，clock开始即记录、不得reset |
| 实际完整试验 | planned5400/hard14400/reserve180秒，prepare/两train/score/read/inventory/reap均含 |
| 每臂训练 | 80updates、batch1、四case各20次，共两臂160次曝光 |
| 图与精度 | 12physical/K4、FP32、alltrainable、原deepK/physicalweights/clip1 |
| 推断 | 每臂固定endpoint80、原4train+4dev、同K4/FP32、5lead/3region/17变量 |
| 设备 | GPU1 UUID GPU-9d1624af-9d77-aa7c-0620-b6cb778f4ced，默认共驻 |
| 启动/spawn | max(known2491416576B,新ownedreservedpeak)+2147483648B，即时query，不能用历史free |
| 失败 | identity/finite/runtime/scope/coverage/schedule/gate/hard停止该attempt，全partial/cost保留，不resume/retry |

软超继续记overrun，硬限不运行中扩；不足资源只能冻结范围内有限等待或停止，不干预邻居。
无总GPU-h许可上限；会计cap/remaining不重建授权天花板。正式完整确认仍需最终科学合同与独立裁定。

## §6 自规划与审核来源

planner本会话实际 `Provider authentication failed.`，不改provider/config。
新工作态 `/tmp/r7_s3_case_interleaving_plan_20261007.json` 由主链自规划，
`tools/check_planner_plan.py` verified:true、fence_stripped:false，科学refs只有既定合同。
Explore确认randperm/cursor而非bulk seasonblock，以及原_update每次zero_grad/clip/step；
本轮batch1等曝光ordering不假用四次_update做一次gradientaccum。
独立内容审核完成，约354.74秒、soft300超54.74/hard900未超、reserve90保留；不是prelaunch或科学判定。四项启动前必修已明确：两80项具体序列及digest/块顺序与起相；普通update0完整body/文件/state外锚和freshoptimizer/RNG；无现成public sampler/train-evaluator的薄TEMP调度及newcontract/诊断评分接线；精确登记CI与完整shared0/两臂比较、开发判读不可提前接受。具体sequence为blocked `[0]*20+[1]*20+[2]*20+[3]*20`、interleaved `[0,1,2,3]*20`，对应Jan/Apr/Jul/Oct，manifest indices1941/2059/2177/2295；实例将另冻结实际case-ID列表和digest。
共同update0普通pt SHA `298eccd8b8c4b9648c3214e7d860c2ea8dbd7a5dc620bc15931762c5dd42b6b0`、state `a23444f530ed4ff80285f51e8b4a86829b27488683a8ee6d258dec4425e99d82`、完整parent contract `954cef09cae2a81b6933340dbe8b2e8a4aa987719dc1d14b9bceb6e6e59b6ab6`，不是update80。两臂freshAdamW相同lr2e-5/warmup10/minimumratio0.1/globalupdates1–80/weight_decay1e-4/clip1，不按病例块重置；逐step及每病例LR曝光分列。任何普通修正必须在真实训练前冻结，不事后调整门。

## §7 明确不做

不增加单Jan剂量/seed/loss/K/aux/typed/PE网格、不重做历史80或exactreplay；
不新下载/派生发布、test科学评分、S4或最终goal完成，不把四season30日块当全年。
不改protecteddata/归档/旧store/outputs/注册证据/用户配置/安全或凭据；
不paid/rental/exclusive/main/merge/release/issueclose/force/mirror/破坏性，不signal邻居、cron/daemon或会话外续跑。

## §8 进度与交接

- 前诊断完整并注册mixed/not-candidate/index61/21.9433GPU-h；原60前缀与旧证据字节不变。
- 本轮自规划已结构verified，Explore只读接入核，独立设计审核与TEMP实现实际调度。
  no新的真实训练/GPU/test评分，r0保持。前诊断登记提交3533483精确CI37645924906已completed/success、pytest112876528185/五必要steps及全部实际steps成功；首次908.967591秒检查在post-checkout仍in_progress/null的旧快照拒绝，原failed-incomplete保留，后精确run/jobs200终态全齐，未改CI或检验门。
- independent设计四项必修已写§6，TEMP实现收到具体schedule/recipe/普通newcontract/分列LR曝光与评分作用域，不将旧bulk随机训练误记为blocked。不把审核可准备当prelaunch许可。
- scope提交 `bd97ac022cd95508552a400b1c07004e39edbac6` 已推工作分支，精确主CI37651269934 completed/success、必要job/steps齐；首匿名请求curl28在45.014895秒0bytes超时，失败保留，新查询625.384873秒成功，不称首尝试通过。
- 新整套治理CPU四模块213passed/0failed/0error/0skip，socket禁网/CUDA隐藏/无过滤，whole37.713866秒、planned600/hard1800/reserve90超0、exit0/reaped/signals=[]。receipt `/tmp/r7_s3_case_interleaving_governance_cpu_20261007_attempt01/receipt.json`、JUnit SHA `1ec7dd1ddb4397ab7e4e1ed11276801667a0e728814f43cd59c45d983e5560b6`；不是新wrapper或科学代理。
- producer原准备clock `/tmp/r7_s3_case_interleaving_impl_20261007_attempt01/clock.json` SHA `89c244da003eb2cf26f709f53c904c1ccdd714ffbb1333c1c7b9b990f43bd95c`，始2026-10-07T15:54:09.354346Z、3600/10800/reserve180不reset。第一固定版继承完整19/19、新增完整15/15过；两次精确LR算术断言失败保留，未加容差或删断言；术语只改为人工blocked压力条件后又完整15/15过，四轮共644合成updates，真实updates0。final handoff SHA `fe06c59a32362f373675cd33ccb68e0143b629e5008903c7a9547ed658ea71c8`、manifest `10c59e5e5c0a40c3c0fd7cb9f8aeb092d17668ce909504c7678e8df64fdb888e`、pins `5f9115e6fb5c387ea7368f3e8c25e8817e677e123eccadbe9ae4f0319321061d`、source ZIP `add3ce75e37b59c8f14713c26d57f7c97286bb3800492e74d2a2b9515ec6ce7d`保留；最终交接4017.764秒/soft超417.764/hard超0。排他launchcopy回执SHA `9c0c33ce3eed38eda7605e66564df12a1ca49c9e54d44ca0aa5b9ad9a00e6519`；主链首copy precheck错假设旧19有real_weights_loaded字段而KeyError，失败独立记录，按真实旧schema核对后复制，未回写旧回执。
- 该固定版独立prelaunch **失败且没有ACK**。新完整15/15（161次合成update）与继承19/19均独立实跑过；targeted15为11pass/4fail/0error/0skip，完整49项45pass/4fail。read_rows遗漏native train/eval grad模式、固定6h lead及报告physicalweights一致性拒绝，四污染仍被complete接受；第四项只是回执控制不一致，不证明实际训练目标已改变。failed SHA `90c313cb58308f00dc35a707405f9afe41f9af641a7a773b696ead73e22b6889`、review receipt `a6d5fbebd498bfb3fecc3964ef9d5d686dc452f1b73ea2bcd53ab2b1ac4ec6e7`、targeted source `53064ef0ffb826e7e9e4af98f28156bd43b51e249bcd282476e2b397b199b291`，均在 `/tmp/r7_s3_case_interleaving_prelaunch_independent_20261007_attempt01/`。whole1292.752148秒/soft600超692.752148/hard1800超0/reserve90；owned children已reap，故意watchdog仅signal自己直接child，模拟邻居自然退出。原失败不追认，producer按原准备clock最小修复并另冻新bytes后须新独立资格，真实parent仍未启动。
- 失败prelaunch及其未qualified源码副本已排他归档 `outputs/r7_s3_case_interleaving_prelaunch_failed_20261007_attempt01/failed_prelaunch_evidence.zip` SHA `2fca8b4de1522a83deb4dec0a965110a0fa38143a941e028d05cbb4f9c938420`，1644文件/119096063输入bytes、2.141715秒、0GPU/0网络；仅含合成CPU权重，不改失败状态。
- 外层父监督和保存结果导出器新增完整合成CPU分别5/5及4/4、0skip；不是真实GPU或天气运行。监督原source SHA `0165fd051565f11fac716c75de28f8e9248aef441fcc771d6473224740c4be33`，外层14760/reserve90仅reaping缓冲，不增加内层14400计算deadline；整轮cost按ceil(supervisor秒)/3600向上四位，含prepare/两臂/评分/读取/inventory/reap。独立审阅指出prepare允许精确val manifest metadata；训练子进程train-only数据路径并拒val/test manifest，但共享Zarr cooperativeguard不是字段索引级OS隔离，不冒称绝无任何val元数据读取。
- 新campaign指向前goal时literal“下一实际动作”不被机械门识别，报告C-04失败，按checker原契约改为“下一动作”后双campaign/goal/index61/brief0失败，未改门。用户配置/旧科学/旧注册证据不动。
- 修复版仅driver.read_rows及一个既有test函数变化，training/old11/recipe/病例序列和科学门不变。新manifest `05f71e52741b266fab3206ec14f6c0a4d72059c2544fe9e436ee249b14ba87ad`、pins `922a87d5877ec24c0e2306c19c3ade7713f6878fa62e22891e4d4309f2087ebe`、source ZIP `2f029b0518c33be3e42b144cea69eea094c9ccf7748c6edc17ec09aaec738ba3`，新排他launchcopyattempt02回执SHA `7854afa568b560b2edd302893895f1efe81688b4f110c8b882baa18826d9f5f4`。producer完整15+19实跑过；独立另阶段完整15+19+原targeted15全49pass/0fail/error/skip，四原错误接受全拒，oldtargeted callback/断言不变。ACK `373e62545f4a476746b62a772f87976fa6c88babb926f2a7828b6f20749de25e`、review receipt `9d272c9205c6a20f26b010ab324701e65a6fd64243b61278a90acacf72263a65`，位于 `/tmp/r7_s3_case_interleaving_prelaunch_independent_20261007_attempt02/`；新独立whole622.313355秒、soft600超22.313355/hard1800超0/reserve90，三个suitechild exit0/reaped/signals=[]，原FAILED永不追认。真实准备原clock终态7746.249412秒/soft3600超4146.249412/hard10800超0，回执SHA `117d224606f4be91924a90ddf632bcb7e7bb713464579128c61c980871cc887f`，没有reset。
- 独立保存事实helper首次准备有66项65pass/1fixture alias错误，deepcopy自身fixture后完整66pass/0skip；但最后收尾elapsed1980.061082秒、hard1800超180.061082，故该阶段为budget_limited-finalization，不追认预算PASS。冻结manifest `54ef9a2b99034f00ccd462a95b5848abeb865c87e8ba18122042d1a5a626c0ce`、last clock SHA `e894a125c40e69a191ce87865e4668c34a4d6934d7f6878c756eaf39b13bebe9`，0ownedchildren/signals=[]；另固定同字节资格阶段在做完整66及显式新stagebudget入口，旧self-testclock不能reset。
- producer/repair/帮助程序失败准备已排他归档 `outputs/r7_s3_case_interleaving_preparation_20261007_attempt01/preparation_evidence.zip` SHA `45b8fa4323b96ed3a5b644717bdf225f8160b85ae31970ac2d1ebc1c783ae6e4`，5953文件/519674081输入bytes、9.111919秒、0GPU/网络；未来独立帮助资格不伪含其中。
- 2026-10-07T18:03:42Z已启动一个真实父attempt `outputs/r7_s3_case_interleaving_20261007_attempt01/`，先CPUprepare后两臂80与固定8例评分；修复父监督source SHA `ac34206b4668bd2f6e9b0c25c41a383596af89d36851557b8830293fdfb676ea`，另同完整5合成CPU过，纯pins/path常量差异。startup GPU1 free25280118784B≥4638900224B、disk576757682176B，即时门passed；其后每spawn仍核maxknown/newpeak+2GiB。**本条仅启动/prepare进行中，不是训练或评分成功**。parent soft5400/hard14400/reserve180与外层14760/reserve90不变；最终cost尚未计算，不先写PASS。
- 阶段实际状态：CPUprepare完整668.972305秒、blocked完整80更新616.803438秒，两个worker均exit0/reaped，标准0/80及逐80行已保存；interleaved进行中、评分pair尚0，不把单臂完整写成全试验成功。资格进度提交 `a1c96392f1eaa8beeea2c6d77213febb5f2dc226` 已推工作分支，精确主CI `37664236520` completed/success、pytest112939230305、五必要steps及全部12实际steps均completed/success；inspector909.859271秒/soft900超9.859271/hard2700超0，工程CI不代替真实终态。
- 独立helper同字节另资格阶段完整66/0fail/error/skip与3实际terminalCLI拒绝反证齐，724.334095秒/soft600超124.334095/hard1800超0；结果SHA `972233759b0e12d9ddb7bf0403977a1891292c2fdc67ba16d27567d833c5900b`，位于 `/tmp/r7_s3_case_interleaving_terminal_helper_qualification_20261007_attempt01/`。两个runner import失败（pycache异常类/stdlib symlink范围误判）保留，未修改helper/旧clock/断言；完整66走exact ordering_tests.execute而不是旧耗尽clock的self-testCLI，3负例证显式新stagebudget/whitelist接线，实际端到端terminal仍未做，旧hard失败不追认。
- 实际父完整终态：两臂各80update/四case各20、16新endpoint80×病例train/evalscore全齐、五phaseexit0/reaped/signals=[]；whole2464.990661秒/soft5400/hard14400/reserve180超0，ceil2465秒向上四位0.6848GPU-h。原0分数强hash复用，未重复预测；protocol canonical `6efae97d11db3d4fa23e0d4be49771010804cede9ccf6909a9ed20ad5bbdf877`、result SHA `24a35359b2c8d0fa701655ae2c45c28d7a0cd2c76e470ee194d68d44845ab723`。新16train/evalfinalhash全同，分解residual照报，不宣通用逐位复现。
- 独立实际terminal03已核160行/160更新、16新pair+8共同0、947allowedfiles首末hash、FP32目标/FP64 draft/17×5×3物理/std²/ACCpool/null/geometry/契约/calendar/RNG/scope/五processgate/cost齐。独立necessarysupport=false与producer一致：t2m仅72h对blocked/shared0都严格好，6/12/24/48比blocked差；u10无满足both的lead，v10只6/12/48，mslp只72。单seed八病例不科学/候选；登记negative并终结本ordering实例，不扩dose/seed/fullval。
- actualhelper terminal01错将eval原生FP64小时/INT64年日当train编码而失败；修独立clone后terminal02结果发布重复keyword且success提前置真，虽exit0/receiptcomplete但failure存在/无result，明确FAILED保留。另仅publication修复success后置/exceptfalse及dictmerge，新完整72实跑0fail/error/skip后terminal03真正result存在/无failure/digest齐。原helper54ef/原hard失败/资格runner失败/两actual失败不追认，不改producer/输出/科学。新helper manifest `ff88929bbde22765c0aa3d514ae6ab7a78c7137b5ecb801700e0ffb259116da6`，terminal03 result `0cca2beacc0e88cbb78b6b37f4cc874d23f44e7cdb3c458d52b47cc621c22cfd`、receipt `44a6e844662790b38740c4158a74bc274907137057b541b4b5cc4fab13a405c2`；整个同terminalstage1053.426838秒/soft600超453.426838/hard1800超0/reserve90、三ownedchildrenreaped/signals=[]，0model/weather/GPU/network，仅4新pt字节hash。
- 文档数值首审先核1530unique CSV行/7650保存metric值0mismatch，但收尾hard900耗尽，elapsed1054.037306秒/hard超154.037306，20first/end0，FAILED不qualified，回执SHA `b52649158adc16363778947f0171390e5e5254487cf63e6e415f2624194da5a4`。只将residual明确为16新pair最大及24含shared0最大；另180/600/reserve90限定source-bound supplement实跑0.382062秒、20first/end一致、normalized新页还原仅该两行后旧SHA相同，prior numericstats按旧首hash绑定而不重跑7650，issues=[]，qualified仅该限定范围，回执 `b602d2c79e84961ed6466a7bd20af0266b81472fdbd9f9a0062805ed9c9c3c3e`。不追认首预算失败，不冒称独立重核CI或全科学接受。
- 未做：证据页冻结commit/index62/canonicalbrief/masterledger与最终登记精确CI；实际两臂与独立终态已做，test科学评分/r0/S4/最终接受未变。
- 下一动作（S3）：冻结 `docs/R7_S3_CASE_INTERLEAVING_PILOT.md` 并登记negative/not-candidate/完整0.6848GPU-h，独立文档数字对照后同步index/brief/master及精确CI；本ordering实例终结。另只读探查主模型显式train-only气候态锚/异常场表达的实质不同路径，不把它假称已支持的新候选。
