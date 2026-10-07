# S3：固定长监督配方的 train 损失分量与梯度方向诊断

<!-- round-node: S3 -->

状态active / preparation，2026-10-07。唯一主计划 `docs/goals/main-model-climatology-campaign.md`，
上一轮同case诊断已登记，当前仍S3，不做S4/test或新方向。

## §0 Objective（单段，实测1207字符）

> 按docs/goals/main-model-climatology-campaign.md的S3接续docs/R7_S3_SAME_CASE_GAP.md，执行docs/goals/s3-train-gradient-components.md的固定权重train-only信号诊断。D1严格固定seed41 long200 endpoint及原12步深监督权重[1,.5,0,.5,0,0,0,.5,0,0,0,.5]，metadata先冻结2021 months1/4/7/10各一个完整lower-median窗口，全部t−6,t,t+6..72h同train，无val/testmanifest或label；D2按原training_long_rollout model.train/FP32/K4完整BPTT计算6/12/24/48/72h原all17归一化深监督loss分量/gradient及weighted total，记录活动掩码、norm/Gram/cosine/总和数值残差、clip1描述性比例和state不变，不optimizer、不改loss/clip或因果推断；D3临时archive-only wrapper/CPU解析及actual tiny Process反证/独立prelaunch齐，原code.zip与wrapperhash/源全文/BUILD_COMPLETE/train归一化/窗口/端点/模型语义先核，fresh outputs先protocol，再首例测peak、其余三例/判读；D4独立算术/身份/费用审阅和代表例另冻重放，证据/index/brief/master与精确CI登记。整轮soft1800/hard3600含准备/归档/梯度/判读/清理，soft超继续全额记、hard或真error停该attempt不resume；0030/0038无总GPUh许可上限，known训练reserved2434793472B加2GiBmargin且每spawn携带max已知/实测peak，GPU1共驻，仅直接拥有Popen可截止清理，禁邻居信号/付费/租卡/独占/main/cron/安全配置变更。旧protecteddata/archive/证据与所有失败只读，原model/evaluator/训练objective不改；train四例稀疏、全变量normalizedloss不同于物理天气skill，梯度冲突/clipfactor不证明欠拟合、泛化、收敛或裁剪伤害。科学判据仅来自docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md与docs/goals/s3-confirmation-baselines-and-candidate.md，不设新赢门、不读2023test、不启动S4、不自裁goal complete。

## §1 承接与已确认事实

- 起点 `46ddd262fdfdfb60319bc455ac428a61c170dade`，工作分支r7/weather-reasoning。
- 上轮冻结证据 `docs/R7_S3_SAME_CASE_GAP.md`，09447bc194562f4dac80868237122084ee5745dc，
  24同case/原气候态/父与长200完整且归档精确再生；index54、方向19.1089GPU-h、test/S4未启动。
- t2m48/72h长200 skill train20为−.779233/−1.002562，val4为−.335360/−1.208763；train每year/month亦负。
  已排除“这些病例只有val失败”的简单解释，不能据此定位优化/表达/数据的因果瓶颈。
- 旧600training preclipnorm均>1，仅为旧记录，varyingcases不判收敛/clip损伤；本轮固定samecase才检查分量方向。
- 执行兼容code.zip为61e46bd78d4d7ce49da7a87d574e40c6c18a1c99，ZIP72953e44fa814db1e1da4d7f5c7505bcb878dc2a9765bc9f8875022a822b6ea5。
  candidate SHA449bc4ce6d9c15178fecb75a954a265ffb43f7ea8cea92575d95387047ee066b；model源码digest3ddab46b1e4c2c7e66449e39ab8247c9c7e642f45023c1c9cc14bca8f74fd217不改。

## §2 交付物

| # | 交付物 | 证据形态 |
| --- | --- | --- |
| D1 | metadata train-only四例/完整12transition和身份先冻 | protocol.json/selection digest/strict endpoint/schema receipts，无val/test reader |
| D2 | 固定长objective分量gradient与total、Gram/cosine/活动/不更新 | FP32张量统计、state前后digest、undefined说明和tmp synthetic反证 |
| D3 | 首例测peak后再spawn其余case、有界owned流程 | code.zip/wrapperhash/process/gate/完整时长/failed保留 |
| D4 | 独立复核/代表例重放、所有费用/限制登记 | artifact hash/E/index/brief/master/工作分支精确CI，非科学PASS |

## §3 判据与解释边界

终极判据仅 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`，原S3screen来源
`docs/goals/s3-confirmation-baselines-and-candidate.md`。本诊断无primary/scientific阈值或候选通过决定。

按 `training/r7_long_rollout.py:63-97` 原目标和顺序加权；
`training/r7_autoregressive_rollout.py:86-108` 要求grad-enabled model.train，不能用eval冒充训练梯度。
K轴draft权重原样保留，futuretarget只监督loss，zero-weight physical步骤仍传递生成历史。

全17变量是原normalized训练目标，不将物理K/Pa/m/s/q差异直接平均成预报技巧；本轮不作t2m新损失重加权。
None梯度对已inactive参数保留显式mask及数学零，不假定全155张量活跃。
零范数/cosine分母标undefined/null，无epsilon制造方向；clip factor仅原实现描述，不能因果称“应调clip”。

## §4 顺序和依赖

1. 完整登记同case证据并核登记精确CI；本次planner草稿不直接当科学或可执行证据。
2. 临时wrapper/core/tests仅work-state，不改production；严格archive-only导入和checkpoint/source/train/norm/BUILD_COMPLETE。
3. CPU解析/合成分量和真实tiny Process阳性/反证及独立prelaunch，wrapper/辅助字节hash先固定。
4. fresh有界CPUmetadata准备，四病例协议发布后首例梯度测peak；按实测门再启动其余固定病例。
5. 全四receipts/精确设置/finite/state invariance与读数核；任一真error停该attempt、保留所有费用。
6. 独立复核及另冻代表病例临时再现，E/index/brief/master和精确CI登记才选择后续不同假说。
   代表病例在原结果未读前固定为case index0（2021-01-14T12:00:00，January lower median），不是按梯度结果挑选。
   重放另冻whole soft1800/hard3600、cleanup90s、prepare→measurement两个owned worker及parent纯scalar判读；
   原全四case先核，case0完整JSON只排除新protocol摘要、elapsed和两项owned peak，其余数值/typed结构要求精确匹配。
   匹配失败保留failed-restoration与成本，不改容差；config-reproducible不升格raw梯度逐位GPU复现。

## §5 预算与停止

| 项 | 具体值/边界 |
| --- | --- |
| whole soft/hard | planned1800 / hard3600s，CPU准备/归档/源身份/梯度/判读/清理全额 |
| cases | 四例2021train months1/4/7/10，首例为同协议feasibility不是删选 |
| GPU | GPU1 UUID GPU-9d1624af-9d77-aa7c-0620-b6cb778f4ced，共驻 |
| 每spawnfree | max2GiB estimate/known2434793472B/本轮实测peak +2GiB margin；至少4370MiB |
| 停止 | identity/finite/执行error或hard截断停该attempt，无自动fallback/detach/OOM重启/in-place resume |

方向cap20/used19.1089/remaining0.8911仅会计，不因账面余数禁止0030范围内有界研究。
首例新增retain-graph分量autograd可能提高峰值，必须测后携带最大值，不能用低推理峰值降门槛。

## §6 planner降级与已核实际接口

planner实读返回草稿，但误用model.eval违反actual trainobjective、以instrumented7.45s推120/180s整轮忽略源/身份耗时、
建议zip-r live snapshot和GPUskip-only薄测试，None梯度停止也与已知inactive参数不符；均不采用。
主链实读 `training/r7_long_rollout.py:63-97` 与 `training/r7_autoregressive_rollout.py:86-149`，
另写工作plan `/tmp/r7_gradient_component_reviewed_plan_20261007.json`，checker verified:true/fence_stripped:false。
预算/样本/模式/验收事实由主链按既定合同冻结，planner不外委科学判据。

## §7 明确不做

不训练更新/改loss或clip/parent臂/val或test reader/新剂量/正式S4；不读test天气场/消耗r；
不动protecteddata/archive/旧failed/证据digest/model/evaluator；不下载/付费/租卡/独占/main/issue关闭/force；
不改config/安全/credentials/dependencies，不cron/守护或会话外自建续跑，不信号邻居。
不从梯度cosine/范数推因果优化结论，不自裁科研goal complete。

## §8 进度与交接

- 状态active/preparation，S3，上一轮同case54index/19.1089账本已登记；登记commit46ddd26精确CI37553027758已completed/success，全部必要steps success。
- 已完成：主链接口侦察、拒绝错误planner字段、校检后的自规划；临时implementation初套48CPU passed（441.63s）、补receipt acceptance后111passed（185.54s），均非天气证据。
- 独立prelaunch第一阶段966s，soft600超366/hard1200未超，确认两个essential反证，历史未通过不回改：
  postprepare hash失败被pop后的protocol摘要KeyError掩盖、未发布failure/cost；全部前后/中间state摘要可协同替换而未锚到strict restored checkpoint。
  审阅两次optional guard在pytest collect前失败（合4.842s）是审阅harness缺陷非项目失败，已弃用，成本含于966s。
- 两项极小修复已落临时bundle：先hash准备receipt再构造新protocol，失败保留旧摘要；strict CPU restore记录state anchor并绑定每例前/后/五个中间摘要。
  最终owner实跑113CPU passed（191.20s，warnings0/skips0）；新独立定向复核181s（soft120超61/hard300未超），
  45targeted passed/20deselected（1.60s，warnings0/skips0）及三项独立scalar核查，指定两blockers/updated acceptance范围无剩余essential。
  定向复核不是全套113重跑，也不回改原966s审阅。不得把工程qualified写成天气/scientific通过。
- 固定临时执行字节：driver633bccbd353a44a02bd7d96e63d1fc235dfd71011f41af83f2020dd3d5239bf8，
  core75c6a3f4708ea74ae6700841a6ec92e2bfdca744f69ca54c7c7e1b8bd89ad351，
  supportdbbb9785ad12e421ca9d07a92c9daad0a1b4eeaeebc8ccfdd7a5b2fe06248460；归档生产代码61e46bd保持不变。
- 实际attempt01已由统一父调度启动：`outputs/r7_s3_train_gradient_components_20261007_attempt01/`，
  原code.zip与临时bundle字节先冻；CPU prepare528.977713s、exit0/reaped、记录signals为空，full source/schema/train/endpoint资格后发布四例final protocol。
  probe972.787152s、exit0/reaped、signals空，case0完整且state anchor不变；其内测量5.068407s不是整worker成本。
  reservedpeak2489319424B高于旧known2434793472B，携带新peak的cases worker已启动，余量门≥4422MiB；尚无四例终态，不预写通过/最终费用。
- 代表例replay临时实现已handoff：11syntheticCPU passed（2.038s）；准备1203s、soft600超603/hard1200超约3s，
  已停止且不是preparation PASS或launch-ready。新独立prelaunch另冻soft600/hard1200，保留原超时事实；不修改原实验预算。
  独立saved-fact审计helper已准备（11syntheticCPU tests），567s、soft300超267/hard600未超，未审实际终态。
- 本轮governance定向93passed（0.58s），节点/goal/index54/37blocking检查通过；prelaunch文档aa9e4ac已推工作分支，精确CI37559461860尚in_progress时读取。
- attempt01四例完整终态：2465.088904s，soft超665.088904/hard超0，保守0.684746918GPU-h（登记舍入0.6847），
  四worker elapsed528.977713/972.787152/956.780425/2.472022s，均exit0/reaped、记录signals空；reserved最大2491416576B。
  final protocol44b634ed478d398754dfa87bebd176cdd060048385f04e2f7f8713a827c4ddc9，result a9c7da6d1fe28c177f952ba61423b436430f6e90198f31edb86bdf654fbac11a。
  terminal assignment f8bb95d93c28494fc63a26c4c0f9eecbf3d662afa1ec4f7cc84b7dbeeab6e0fd已交独立saved-fact审阅，未冒称raw梯度或科学确认。
- 全量本地4009passed/3skipped/6warnings（1257.87s，4012JUnit、0failure/error）；skips为未跟踪optional真数据fixture，不算通过。
  该默认全量测试未隐藏CUDA，实际六项CUDA条件测试执行，不是CPU-only；未留该套专用启动UUID/余量记录，是执行证据缺口，不能称R-054/有界GPU合规。
  JUnit起点至最终log1258.349468s，ceil1259保守计0.3497GPU-h；deviation回执07023eb8d925ec0ba926ec4065a174b6ba9d67215bd986a9f8a4b505bbfbcc17保留，不回写原诊断cost。
  另冻明确CUDA_VISIBLE_DEVICES空的CPU全套soft1500/hard3000：4003passed/9skipped/6warnings（1202.81s），
  whole1206.472975s、soft/hard超0、ownedexit0/reaped、signals空、0GPU-h，不能追认原取卡纪律。
  aa9e4ac0c43f5f1295233d9b7446a34803736ea3精确CI37559461860 completed/success，全部必要steps success；17实验标签门控skip未当实跑。
- 代表例独立prelaunch被provider INTERNAL_ERROR中断，未取得verdict；保留partial审阅，另冻恢复soft300/hard600接续已有工作，不冒称通过。
- 独立terminal saved-fact复核完成：721文件起终pin一致、四例全scalar/设置/身份/state/cost/gate资格齐，326.754s、soft300超26.754/hard600超0。
  audit_report fa5b43c8fb56791e84546c790f3f9b07f3cf240ed3055c09b672feeac82047c4；不冒称rawtensor/OS生命周期/全年科学确认。
- 代表例工具另发现bootstrap失败无cost，历史not-qualified保留；最小修复157s soft180/hard360未超，owner15syntheticpassed。
  新独立定向255.218s（soft120超135.218/hard300超0），15currenttests及4B1反证/25文件saved integration齐，工程qualified；
  原1203s准备hard超3、provider中断及恢复446.991s软超146.991均保留。
  qualified replay driver586c3f4e8b31546c3573b4a2a56df46219fa5b64541e09d822aefa50f1b9392d，
  support431e9ad1f89ebd0b24b9b2f02de8ccb0abc572b4c643f9766cd6569f00b7bf7c，
  必需25files/pins0c3b754e0909fb0a83d6f8adc11ef93ae81269b50d0816dbe422beecaeb1587f，CPUprepare之后才允许Jan字段测量。
- 代表例实际attempt01已失败且停止：prepare归档写入fixture val/test manifest basename被read守卫拒绝，未进入实际天气场/CPU恢复或GPU梯度。
  whole1.834366536s、soft/hard超0、0.000509546GPU-h（四位0.0005）；prepare exit1/reaped、signals空，后续未spawn。
  failure c7b8fc4410cc4e128f2a5ae96c746bc2066b682bf99d90e333b86d954af8825d，旧failedoutputs与包字节保留。
  只修归档复制/实际manifest READ禁边界，原资格没覆盖全ZIP guarded extraction的缺口保留；原数值/science判据不改。
- 最小replay-only extraction adapter保留完整ZIP身份/全部692Python，跳过非Pythonfixture，原guard READ/write禁不变。
  owner16tests通过但399s超hard360约39s，历史not-qualified保留；独立新阶段16tests/真实exactZIP guarded extraction/两次tinyProcess/full12/digest和mutation齐，
  314.312s soft180超134.312/hard360未超，snapshot工程qualified。driver仍586c3f4e...，support新ab1e7609a811a593219c78e1fa27eec6352438f29b0091f55725832f551f238c。
  attempt02必需pins1ecc06c80a5e63e7be44390ea6ae0a918cca7a7d7123bad998aa6f338d30189a，旧attempt01/原pin保留不改。
- attempt02完成CPU身份准备/Jan梯度，却在完整JSON精确匹配失败：161个梯度统计/派生numeric leaves不同；loss/state/RNG/activity相同为主链初读，独立差异审阅待核。
  whole1569.817662s、soft/hard超0、0.436060462GPU-h（四位0.4361）；prepare545.355336/measurement1021.508986s，exit0/reaped、signals空，但terminal failed-restoration非pass。
  failure5b994e3511f2501ad51aac1e44bee48635dfc3680830f3cb8b10ce09772ced2f，recomputed fcfec322597a217b9883db42002ccd4c861e684918c59e45ba6a85b7af47f381；无RESTORATION_ACCEPTED。
  frozen匹配判据不改，不设epsilon/tolerance、不启动attempt03到偶然精确；最高config-reproducible，不能从原loss相同冒称raw梯度逐位恢复。
- 独立negative-replay terminal复核已齐：161numeric leaves（Gram/派生重复值，不是161独立测量）与主diff全同，cosine符号保持但不挽救精确门。
  原/重算loss/state/RNG/activity/FP32clip/maximumresidual精确，配置/math资格齐；浮点autograd非确定性是相容解释非唯一确证rootcause。
  1462文件end-rehash零不符，whole373.965s、soft300超73.965/hard600未超；两个failed和marker缺失均核。
- 本轮描述性/failed证据已冻结commit6c3bb69696b5d4855da81e9b15eb9f5dba3acbe9，
  `docs/R7_S3_TRAIN_GRADIENT_COMPONENTS.md` SHA627b324d60f08f442592a3430113f088bb356d4dccf1b6431cbbadc5f69bbdba，
  finalfactreview143.126s（soft120超23.126/hard300未超）无mustfix；原注册页hash保持。
  index54→58四记录（原诊断/audit，两replaynegative，mixedtests资源缺口/audit）；主账本19.1089+1.4710=20.5799，remaining−0.5799会计字段。
- 登记提交70fe46ea0a78f10f08ac1a6c765336b305648d45精确CI37569443913 completed/success，全部必要steps success；17标签门控skip不当实跑。
  D1–D3原四例描述性证据已登记，D4代表例精确恢复明确negative并独立归因，非成功复现；没有科学PASS或旧failed回改。
- 未做：代表例精确恢复（实际未过）、test/S4/optimizer；不自裁科研goal complete。
- 下一动作：S3内另冻固定病例原目标响应诊断，沿 `docs/goals/s3-fixed-case-objective-response.md`；不改clip/loss、不重复bulk剂量或原replay到偶然精确。
