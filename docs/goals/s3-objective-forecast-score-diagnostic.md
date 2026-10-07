# S3：deep-K 目标与最终物理评分的配对诊断

<!-- round-node: S3 -->

状态 active / 准备中，2026-10-07。唯一主计划为
`docs/goals/main-model-climatology-campaign.md`；本轮接续其最新 §8 的无新增更新诊断，
不重做单病例 80 更新、三端点精确反算或旧 failed replay，不进入 S4。

## §0 Objective（单段，实测1099字符）

> 在 docs/goals/s3-objective-forecast-score-diagnostic.md 接续主模型超气候态 campaign 的 S3/index60/21.4177GPU-h。D1核归档code.zip、完整新80contract、普通loader、既有0/20/80端点及原2021四季train和2022四val开发病例，同train气候态/单位/完整窗口身份；D2以全新protocol和排他outputs、无optimizer/backward/updates，分别记录原train/enable_grad的12物理步all17 normalized deep-K监督及普通eval/no_grad最终17变量×6/12/24/48/72h的full/interior_1/edge_1物理MSE/RMSE/skill/ACC，草稿与最终头分开、future真值只监督不进前向，state不变；D3实跑合成CPU接线/目标投毒/分解与单位/完整contract/案例错配/undefined/失败反证、独立prelaunch后才读真实字段并GPU诊断，独立保存事实核查和全部正负登记；D4据同例、跨train和四开发例的证据自主选择下一实质不同可反驳主模型假设，只有开发支持才扩完整val，不默认增加剂量或复活negative；D5登记code/data/protocol/resultsdigest、复现等级/limitations、全部失败和成本、index/brief/master与工作分支精确SHA必要CI。准备planned3600/hard10800/reserve180；实际诊断整轮planned3600/hard10800/reserve180秒先冻，soft超继续记、真错或hard停该attempt全部留痕。GPU1默认共驻每启动/spawn核即时UUID、maxknown2491416576B/新实测peak加2GiB，父统一调度不signal邻居。0030/0038无总GPU-h许可上限；旧cap20/remaining−1.4177只会计。科学门只引用docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md；八小样本/单seed只开发诊断不证明泛化，2023test未评分、r0不变。不得改旧outputs/归档/数据、冻结门/用户配置/安全，未授权paid/rental/exclusive/main/release/issueclose/破坏性，不cron或会话外续跑，不自行宣布最终goal完成。

## §1 承接与起点

- 起点 HEAD `d892e20d93afc1a4fb54006300cadfe887d5485d`，分支 `r7/weather-reasoning`。
  原工作区有用户配置/研究员定义、长期交接四份 docs 与 `.zcodeignore`，起点字节
  已排他保存 `/tmp/r7_s3_objective_scoring_resume_20261007/resume.json`，不改用户配置。
- 主计划最新交接：`docs/goals/main-model-climatology-campaign.md` §8 最后段；前一证据
  `docs/R7_S3_FIXED_CASE_OBJECTIVE_RESPONSE.md`，index60，累计21.4177GPU-h。
- 旧单 Jan 病例原目标有限响应为58.8800%，全部12 loss下降；不代表最终天气输出改善或泛化。
  既有80更新和独立精确反算均 audit/not-candidate，不重跑、不变为候选。
- 原目标接线：`training/r7_long_rollout.py:64` 调原 `_draft_loss`，initial与全部K草稿受监督；
  `training/r7_autoregressive_rollout.py:112` 要求最后draft等于forecast，只有该final进入下一物理步。
  `model/r7_rollout.py:79` 是独立普通eval/no_grad目标不可见评分路径，不能偷换训练mode。
- 生产代码来自原61e46bd归档code.zip SHA
  `72953e44fa814db1e1da4d7f5c7505bcb878dc2a9765bc9f8875022a822b6ea5`，新诊断行为由独立
  byte-pinned TEMP薄wrapper承担，不用当前checkout修改checkpoint身份。
- 普通新80完整contract signature
  `954cef09cae2a81b6933340dbe8b2e8a4aa987719dc1d14b9bceb6e6e59b6ab6`；
  training/bound_contract.json 文件 SHA `9fcf23198a91bc2538f9d6cb05790f0fda6406d09f1e37ef5a9682de971ab93a`。
  三标准pt的SHA由前证据§4与本轮新protocol逐项绑定，不拿原long200 signature冒充。
- 当前provider最近goal verifier记录为cancelled/user，原错 `Model request was cancelled.`；
  本会话不依赖它推进节点，也不把独立内容审阅当最终verifier。

## §2 交付物与 prompt-to-artifact 核对

| 编号 | 必须实际交付 | 证据形态 / 当前状态 |
| --- | --- | --- |
| D1 | 原归档/数据/完整new80 contract/0-20-80与八原病例、同train气候态身份 | 外锚protocol、ordinary loader与完整metadata资格、first/end hashes；准备中 |
| D2 | 24 endpoint-case原deep-K分解与最终全17×5×3物理评分 | 每对原12 loss/total、五draft×channel normalized误差、train-final与真实eval-final区别、RMSE/skill/ACC充分统计、完整病例；未做 |
| D3 | CPU反证、独立prelaunch、GPU每spawn共驻与owned截止 | 真tinyProcess/poisoning/单位/聚合/全body错配/state/nooptimizer、process/peak/budget；未做 |
| D4 | 正负结果与可证伪接续 | 同Jan/其他train/四dev分报，草稿/最终/跨例干扰问题对应；证据支持才扩val；未做 |
| D5 | 独立保存事实/math/identity/cost审阅、冻结登记及精确CI | 新证据/index/brief/ledger、工作分支精确SHA与CI；未做 |

长期D3数据扩展与D4正式独立年度确认不在本诊断内假装完成。现数据只有各季30日块，
不等于完整未见年份。所有17变量及坏变量必须保留；不同单位不平均。

## §3 判据与证据来源

科学门只有 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`。本轮没有新增科学获益、
候选或显著性阈值，正常负/混合诊断也可登记，不等于最终goal通过。
复用前证据 `docs/R7_S3_FIXED_CASE_OBJECTIVE_RESPONSE.md`、
`docs/R7_S3_SAME_CASE_GAP.md`、`docs/R7_S3_TRAIN_GRADIENT_COMPONENTS.md` 的合法pins。

- endpoint0/20/80全覆盖，原四train2021及四val2022都为Jan/Apr/Jul/Oct14日12Z，
  lower median rank53/107，manifest index分别1941/2059/2177/2295与53/171/289/407。
- 每次诊断的loss与final预测都从严格相同端点/病例出发；normalized训练目标与不同物理单位MSE分开。
  不把earlydraft平均当final，不把eval目标不可见路径换成training路径。
- 保存原loss算术、可分解draft/channel误差及native FP32重构差，重构差异照报；
  新forward不作为已完成精确readback重试，不要求跨不同数学聚合逐位同值。
- 原train-only2400stamps/16bucket各150；复用原估计器与全部mean身份，
  mean_identity `c633df585b1894704d942e24305047a74ff7af2460cbc24ca2ebe7be1948a205`。
  统计不fit val/test。ACC先汇总dot/energy后除，不平均病例ACC，零energy保留undefined/null，不加epsilon。
- CPU反证和独立qualification只限定工程接线，不能替代真实天气诊断或科学确认。
- 原final与eval路径若不同，先登记并查mode/metadata/head/评分，再决定修复；不静默换路径或改旧结果。

## §4 顺序与依赖

1. 复核master/两campaign/goal/index与起点，Explore定位；公开issue正文评论实际只读检索。
2. planner实际服务端认证失败后，主链自规划JSON经 `tools/check_planner_plan.py` 同契约校验，
   两处API事实核对与独立内容审阅。它不是planner成功委派或科学证据。
3. TEMP薄wrapper只改新行为，合成CPU真路径与反证；严格归档导入和完整new80 ordinary loader资格。
   old `_restore` 固定1600/200，不将其直接换入0/20/80；不改oldloader/model/evaluator。
4. exact代码/pins/病例/同train气候态/单位、preparation scope先冻结；完整source/preflight/
   BUILD_COMPLETE/norm/train与val窗口metadata通过后才缓存八例。新val_read=true只属于此新诊断，
   不回写旧fixed-case的val_read=false。
5. 独立prelaunch后统一父调度CPU准备、GPU24配对两路径、CPU读取聚合与endhash。
   每spawn核即时共驻UUID/maxpeak+margin；真实错误或硬截止停止attempt、不自动retry/resume。
6. 独立read-only保存事实核查，result-freeze登记全部正负/cost/index/brief/master，精确分支CI。
   诊断后按证据规划不同主模型假设，不把未完成的计划列为研究交付。

## §5 预算与停止条件

| 项 | 运行前冻结值 |
| --- | --- |
| TEMP准备整轮 | planned3600 / hard10800 / 收尾reserve180秒，起点resume receipt |
| 实际诊断整轮 | planned3600 / hard10800 / 收尾reserve180秒，CPU准备/归档/字段/24两路径/读取/清理全含 |
| 规模 | 单seed41、八原病例、endpoint0/20/80、24配对，不新optimizer/backward/update |
| 物理/internal | 原12步weights[1,.5,0,.5,0,0,0,.5,0,0,0,.5]、K4、FP32、原全图trainpath与evalpath分开 |
| GPU | GPU1 UUID GPU-9d1624af-9d77-aa7c-0620-b6cb778f4ced，默认共驻 |
| 每启动/spawn余量 | max(known2491416576B,新实测reservedpeak)+2GiBmargin，不能取历史最大free替代即时值 |
| 失败 | identity/finite/runtime/path/不完整病例/不足资源/hard停止当前attempt，全额费用和partial保留 |

软超继续并登记，只有已冻结hard因时长截断；新独立协议可设不同数字，不运行中改硬限。
无总GPU-h授权上限，cap/used/remaining只会计。独立审阅与CPU全套如触发另冻数字和收尾余量。
网络实验请求0，新下载0，磁盘增量与墙钟/训练0/评分/审阅/GPU保守口径分列。

## §6 planner降级与内容修订

- 实际planner委派失败：`Provider authentication failed.`，duration142427ms。
  自规划工作态 `/tmp/r7_s3_objective_scoring_resume_20261007/planner_fallback.json` 已verified:true。
  不把主链JSON称委派完成，不修改provider或用户配置来绕失败。
- 主链实读API确认 `_training_inputs` 必须grad/train，final-only eval独立，旧_restore固定端点，
  region独立coslat面积和ACC pooled充分量。独立只读内容审阅完成，原JSON首末hash一致，
  末hash阶段815.332秒（soft600超215.332/hard1800未超，reserve90有余）。确认方案有新增信息价值，
  但要求统一科学判据来源与goal改名引用；原JSON保留不改，新revision SHA
  `00492b642afd817868b141f0d8fa9021e07e6e31e87a7893f100afd79987aab6` 已同步两项并重新verified:true。
  API偏一行已修；metadata_case_plan返回20+4，八例精确过滤，不冒称API直接返回八例；
  24配对每对含两条12步路径，共576次物理transition，预算包含两路径。不把内容审阅当prelaunch或科学接受。

## §7 明确不做

不重复case80/原精确readback/旧failedreplay，不训练、加剂量、改clip/loss/K或选test结果。
不读testmanifest/weather，不消耗r，不宣称泛化/欠拟合/clip因果/全年/机制或scientific_pass。
不改protecteddata/archive/旧outputs/registered evidence/用户配置/安全/凭据，不增依赖。
不paid/rental/exclusive/main/merge/release/issueclose/force/mirror，不signal邻居、不cron/daemon或会话外续跑。

## §8 进度与交接

- 2026-10-07：起点已核，两个campaign0失败（各4历史notes），goal0失败；index60/21.4177GPU-h。
  首次master校检传绝对路径被拒，按CLI改仓库相对路径后通过，未修改机械门。
- 已实际加载goal-loop、decision-record、issue-lifecycle、planner-delegation、pinned-artifact-replay、
  bounded-study-run及web-research；Explore定位与general-purpose TEMP实现/独立内容审阅已调度，
  web-researcher在只读公开issue检索。planner真实认证失败，自规划同契约verified。
- 已只读提取原八病例/全new80body/归档/气候态身份至起点临时inventory；未真实字段诊断。
- 公开issue正文/全部评论已合法curl补齐并实读，9请求全HTTP200，#76–#79仍open；最新20项#79至#60中无#80+，旧#70–#75仍closed。原web通道失败与备选未取全#76保留，不用摘要充原文；来源与259290body bytes/11.347789s及官方文献回核登记 `docs/R7_S3_DEVELOPMENT_SOURCES.md`，0GPU。
- 新包首轮完整合成CPU17/17通过是producer中间结果，最终稳定pins/完整普通loader与独立prelaunch尚待，不提前写qualified。
- 本轮治理CPU四完整模块已新实跑213passed/0failed/0error/0skip，整轮37.970701s，soft600/hard1800/reserve90、超时0，socket拒出网/CUDA隐藏/无过滤，workerexit0/reaped/signals=[]/0GPU。
  回执 `/tmp/r7_s3_governance_cpu_20261007_attempt01/receipt.json`、JUnit SHA
  `e2e9bd77ee4dba1feccd3c446dc8033b600f38f043e69937d62dc597a5eef83c`。
  这是治理模块而非新wrapper、fullsuite/GPU或科学接受；当前活跃生产代码未改，不重复旧4003全量当本次实跑。
- 文档提交 `3c354e642682b3767f4f94771104ea7c7aa937b8` 已非force推工作分支；精确主CI `37629370467` completed/success，pytest job `112819315099` 与必要steps及全部实际steps均success。首次匿名API检查416.865428秒因HTTP403中断，保留failed-incomplete，不称测试失败；后续精确run/jobs两次HTTP200及原字节核对确认成功，回执 `/tmp/r7_s3_objective_prepare_ci_20261007_attempt02/verified_receipt.json`。未main/merge/issueclose。
- 首次独立指标CPU88项55passed/33failed/0skip，33个FALSE_ACCEPT均保留：shape、外锚units/std、bool/count、coverage、ACC域及region补集。原snapshot SHA `0565af9d2f3d9ac5e4b1bcd4491d60a23e2abdc749cac6ab210479a580bf37b8`、terminal SHA `a37266be52c10efaaeb410479f43a48549b0b29aeb25fabf082163529c605503` 不回写。
- 最小修复最终manifest SHA `6ca336511c8eda4d2d04f8bd34806387db1af0cc3dd0546b487ca5f649714254`、new pins SHA `31b7d5d8a4afc2c24cc4728c65391e8c91022e1156914ebba441027544a70959`；producer完整CPU19passed/0failed/0error/0skip，原88断言AST未改、仅适配required expected/coverage与owned路径后实际88passed/0failed/0skip。完整24唯一身份由outer `(update,case_index)` 与原casebody核，pool只证计数；summary保留channels/units/leads/std、backward_calls0有实际tripwire。producer原计时4766.362815秒、soft3600超1166.362815/hard7200未超、reserve180，0GPU/真实字段0/optimizer0；所有修复失败保留，implementation receipt SHA `0feb003317f9e55ec2eb45eb20d368a8a6001be0e7627875c1a2d54e005f7956`。
- 固定包已核字节复制 `/tmp/r7_s3_objective_forecast_launch_20261007_attempt01/`；独立prelaunch另冻600/1800/reserve90、2026-10-07T14:14:00.963122Z起点，正在CPU限定审阅。真实source/BUILD_COMPLETE/metadata与普通三端点加载仍由新protocol CPUprepare实际资格，不用合成测试代替。
- CPU准备证据归档 `outputs/r7_s3_objective_preparation_20261007_attempt01/preparation_evidence.zip` SHA `43183ba86490d06d03dee9e49cf2530fde8705b82c85c4674559e04b96eaeca3`，1248files/127245323输入bytes，整轮2.916154秒、0GPU；保留producer/原失败审阅/治理/CI原响应日志，不是天气证据。
- 原准备时钟不reset；elapsed5867.760794秒、soft超2267.760794/hard剩4932.239206、reserve180。用户配置/原index60/brief/科学合同起点hash仍同；双campaign各0失败4历史notes、goal0失败、37阻断0失败。index首调用少传必需 `--index` 返回usage错误，按原CLI补参数后60条/brief通过，未改门。
- 独立prelaunch限定工程PASS，receipt SHA `600a7abf8b1e98f05a7b19ffc8e4a73e88cf2c6b9360ec5187122261a4999aff`：实际完整CPU19/19、独立read_rows38/38，指标原92为91pass/1审阅TMP接线fail/0skip（原失败保留），原33错误接受均独立实拒绝；原环境断言补测及三gate检查4/4闭环。共153calls=152pass+1保留fail/0skip，不冒称单一all-pass套件。whole1515.378506秒、soft600超915.378506/hard1800剩284.621391、reserve90；全部owned直接child reaped，唯一signal为watchdog自有PID208994。首末FINAL全hash相同，未真实字段/source/checkpoint/test/GPU/network。归档 `outputs/r7_s3_objective_prelaunch_20261007_attempt01/prelaunch_evidence.zip` SHA `fd71dafbddb7fb5ccbc23bec1959f67c0fcba4eebc601d86502028460033e2ba`。
- 原整轮preparation已于2026-10-07T14:42:02.706533Z终态：7178.884820秒、soft超3578.884820/hard10800未超、reserve180，0GPU；时钟未reset。新实际诊断排他 `outputs/r7_s3_objective_forecast_20261007_attempt01/` 已启动，先冻结preparation protocol canonical `fa93763b9dbffa6190db02749e8fbeb311fdf78cc97534e19218ef301977fdc1`（文件 SHA `e92937ace108a45b993f85088fc2e9e8d8d2593e0e58b191e0ba45fcc51a6f99`），planned3600/hard10800/reserve180、24train+24eval paths、optimizer/backward/update0、valtrue/testfalse。CPUprepare真实source/BUILD_COMPLETE/metadata/三端点资格正在执行；尚未称weather诊断完成。
- 实际CPUprepare完成：671.480996秒、exit0/reaped/signals=[]，真实full source540856239B SHA `bc2ff9cfadcce604fc243bb999b3c430d5164d17fcf1de201273716a5db065f8`、preflight/BUILD_COMPLETE/train/val/归档model与training身份、普通0/20/80外锚state/contract均符；prepare receipt SHA `2954fb34745ebb89e319188a802ef064eee1d3442ee28ba56b02898ff88999dd`，state_fields_read=false。final protocol canonical `26a17743b50c159b4b5e1e70703c89c3a9dd0ee56aaf4efebe4e687663d2a59d`、file SHA `ebad8a894dd5d42947d39d43359f1706f977059bec954a9e9b9af1e446cb756a`，CPUprepare后才准备fieldworker。diagnostic spawn即时GPU1 free25280118784B≥known2491416576+2GiB门4638900224B，父调度启动成功；24配对仍执行中，不提前填写天气读数。
- 实际24配对完整终态：三worker671.480996/1205.413004/6.282190秒、exit0/reaped/signals=[]；parent含inventory/reaping1891.377279秒、planned3600/hard10800超0、保守ceil1892s0.5256GPU-h。原Jan目标−58.8800%、final83/85改善；其余3train/4val目标全升，val最终241/340变差，t2m18/20caselead变差；所有24train/evalfinal tensorhash同、maxabs0。不是泛化/容量/clip/干扰因果，不能扩完整val或晋为候选。全17×5×3两路径/病例/FP64-native residual/坏变量齐，结果 SHA `ca8919139c6999d9a7c2532ee91e2e2eae6dd37fb830f424ab5d5a12999e9d9a`。
- 独立terminal原helper合成47/47通过，首schema错误/服务overloaded中断保留；同字节helper新600/1800/reserve90单次真实savedfact核194.56s成功，754allowedfiles首末/原cases/nativeFP32/FP64/physical/pooledACC/counts/gates/process/cost无mismatch。receipt SHA `c60add04a8cd1952b637524fb8b0383b0183dd6a8dfd07c3adec2ec7f83eff0e`，终态证据zip `c0a90f893e0c3184b8ae6c8a92f919e8fed81a044178ed094333ede8c194b281`；未followtrueweather/ckpt/source，不伪称独立重读或科学接受。
- 新证据 `docs/R7_S3_OBJECTIVE_FORECAST_DIAGNOSTIC.md` 冻结于 `0befd262f971318cacbcae627d62a16a6e5723b4`，SHA `ff02acac4fb87160bbb1eafd82bdbd1465fd4e724b5c5ded1afec0ac55a35a58`；已追加index61 `mixed/not-candidate`、canonical brief同步，原60前缀SHA不变，账本21.9433GPU-h/remaining−1.9433仅会计。最终登记 `3533483df9d8cafd9064ddf561335e64bfb3d102` 精确主CI37645924906已completed/success、pytest112876528185必要及全部steps成功；首次旧post-checkout in_progress/null快照导致908.967591秒检查failed-incomplete留存，后新精确run/jobs200终态齐，不改checker/CI。无新optimizer/update/backward/test/S4/r/科学接受。
- 未做：实质下一主模型干预、全年独立确认/科学goal接受；本诊断运行/保存事实已齐，工程绿不代最终goal。
- 下一动作（S3）：按跨病例负响应冻结不同数据制度/信息路由可证伪主模型干预，CPU资格齐后实跑同数据incumbent/climatology控制的train/val小试验；不默认增加剂量/seed/开关，不复活negative，不重复case80/exactreplay。
