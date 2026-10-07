# R7 S3：固定 train 病例的原目标响应

状态：真实80更新及三端点独立精确反算完整、限定保存事实/状态语义核验齐，2026-10-07。
本页是issue #76方向内的描述性诊断，不是候选通过或科学确认。
唯一主计划 `docs/goals/main-model-climatology-campaign.md`；本轮目标
`docs/goals/s3-fixed-case-objective-response.md`；科学接受仅来自
`docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`。`scientific_claim: false`。

## 1. 范围与可核查结论

本轮接续已注册的固定端点梯度诊断，而不是重试旧精确恢复或延长bulk训练剂量。
固定已冻结的January2021 train病例：2021-01-14T12:00:00 UTC，107完整窗口中的
lower median rank53，原manifest index1941，历史t−6/t、全部t+6..72属于train。
从seed41 long200的严格模型权重出发，只在同一缓存batch上执行恰80更新。
不读取val/test，不拟合新气候态，不评分物理天气skill，不消耗confirmation r。

**已确认的描述性事实：**

- fresh AdamW按原目标、clip1、K4、all17 normalized、12物理步及完整BPTT完成80更新。
  新80更新cosine长度被显式冻结，不是原200更新schedule无缝resume；旧optimizer/cursor/RNG不导入。
- 单病例原目标在预声明0/20/80端点为1.29634428024292、0.774080753326416、0.5330567359924316；
  endpoint80比endpoint0低58.8800%。全部12个per-step loss在这三个端点均下降。
- 全80行是每次update之前的loss；单独端点20/80是更新之后重算。
  第80行loss0.5335893630981445不等于post80端点0.5330567359924316。
- 80行before-step total没有增加；preclip norm范围0.6833153367042542–15.411456108093262，
  其中42行>1。实际仍使用原clip1，未据此更改clip或推出clip伤害因果。
- 155参数的原病例外锚mask为141 active/14 unused；每次实际backward hook都核相同mask，
  native seed41 RNG在各更新/端点未改变。标准0/20/80 checkpoint完整，fresh optimizer slots为0/141/141。
- 准备、训练、读取三个owned worker均exit0/reaped，保存signals为空；整轮1547.848237s，
  planned3600/hard7200无超时。保守ceil1548秒计0.4300GPU-h，包括CPU准备/读取/归档和收尾。

**登记边界：**训练与反算成本分列0.4300/0.4078GPU-h；登记仅为audit/not-candidate，不能升级科学或候选接受。

**未确认：**泛化、其他病例或变量的物理收益、整体模型容量、欠拟合/收敛、clip因果、
全年独立确认、同时统计区间或超过气候态接受门。本页不新增科学或候选通过阈值。

## 2. 代码、数据与新端点身份

训练产物：`outputs/r7_s3_fixed_case_response_20261007_attempt01/`。
反算产物：`outputs/r7_s3_fixed_case_readback_20261007_attempt01/`。
生产实现来自兼容原code.zip，新增行为由独立byte-pinned TEMP wrapper承担；
不使用当前checkout绕开旧checkpoint的model/training identity。

| 身份 | 精确值 |
| --- | --- |
| 兼容生产commit | `61e46bd78d4d7ce49da7a87d574e40c6c18a1c99` |
| code.zip（692 Python项） | `72953e44fa814db1e1da4d7f5c7505bcb878dc2a9765bc9f8875022a822b6ea5` |
| training TEMP driver | `3073ea784dbd76a44c1f88adb1329c66ce120194235e18120425029467da97f8` |
| training TEMP support | `7e4540b42debde4befdda247edcbc4c18b81af25c0fa862aa5f6d297bf75ba86` |
| model code | `3ddab46b1e4c2c7e66449e39ab8247c9c7e642f45023c1c9cc14bca8f74fd217` |
| training code | `24dde71ba380312f2c201c8f5fd2cabb9c884585bb0ce904fdee768fc5f675b9` |
| 原long200 checkpoint | `449bc4ce6d9c15178fecb75a954a265ffb43f7ea8cea92575d95387047ee066b` |
| 原long200 contract | `944e6741bfefba9fce090c7ed69e74c90db9c63fa0341ba6ffabd10af973def9` |
| full source（540856239 bytes） | `bc2ff9cfadcce604fc243bb999b3c430d5164d17fcf1de201273716a5db065f8` |
| source preflight | `1b914a216bb3a79e044c316749bfad45a2a27bcd5df8bd07cd5cbf9ce4bd58e8` |
| train identity | `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac` |
| full12窗口identity | `dccfeef926286ddeccb7690aacd7bac45c1bb2036fe6ebbe826841b971bb39ec` |
| 原January case JSON | `3e92dd4e1780bf7ed5291c95a172774fb808cb0d6b7ce8638a4ca15673af49ca` |
| preparation canonical | `61df2419ee52015f24980d116307613549c7d77b7201f98c1ab90cf8d5f03816` |
| final canonical protocol | `699ee5e5fbb67369185ea7824d9e5f33a14b3dce3cf9226dac3facffd0a0fa0f` |
| protocol.json bytes | `12b5010ba65d4a7789e88e4d358d00c585b0f42cfe3eb84e610256a97cbc3b42` |
| 新80完整contract | `954cef09cae2a81b6933340dbe8b2e8a4aa987719dc1d14b9bceb6e6e59b6ab6` |
| result.json | `a59614df732c0f090e26798bcc2fa084252a7ab26ab7e606939d0e483840e3b7` |
| attempt.json | `43d18c3e493dd1496ef9de3c18a4e08457a4b634844d3b34bddbb0c833751650` |

source preflight完整SHA256为
`1b914a216bb3a79e044c316749bfad45a2a27bcd5df8bd07cd5cbf9ce4bd58e8`。
完整train发布于v3实例，2017–2021 train、2022 val、2023 test。
原2360记录、2140完整12步窗口、220排除、BUILD_COMPLETE、source/preflight、
train-only归一化、17通道及单位、模型语义/state/全窗口先核后才缓存该病例。
只选一例不豁免完整metadata资格。原protecteddata、旧outputs和归档未改。

## 3. 原目标、新recipe与出版契约

物理权重保持 `[1,.5,0,.5,0,0,0,.5,0,0,0,.5]`，有效lead为6/12/24/48/72h。
K轴仍为initial及全部K drafts的原深监督；零权重物理步仍推进生成历史并保留图。
futuretargets只监督loss，不进入model输入。`training.r7_long_rollout_runner._update`
逐项forward/loss/backward/clip1/step/finite/zero_grad原语未改。

fresh AdamW：seed41、batch1、恰80updates、lr2e-5、warmup10、cosine minimum0.1、
weight_decay1e-4、FP32、无autocast、无detachment、process_weight0。
无earlystop、loss-based checkpoint选择、隐藏更新、续跑或自动加剂量。

新contract不冒充旧long200：包含singlecase与完整原window/数据/源/初始化/构造语义、
new80 recipe、每wrapper/primitive digest、native RNG/schema及原病例参数活动外锚。
CPUprepare模板只排除最终protocol_sha256；finalfreeze注入后train/read校完整body。

原save_exclusive临时link会短暂nlink2，与未改的owned guard冲突。
本轮仅私有出版binding适配为exclusive xb/flush/fsync，仍标准r7-local-v1 payload及普通loader。
**非原子**：失败可能保留partial .pt；close/fsync/strictload未齐不出版endpoint/report/success，
不覆盖或删除partial，不增加native/ctypes旁路，也不放宽guard。

## 4. 全部端点与轨迹读数

全部数值均为原all17 normalized监督目标，不是K单位RMSE或climatology skill。

| endpoint | total | L6 | L12 | L24 | L48 | L72 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.29634428024292 | 0.195212588 | 0.302204847 | 0.416509837 | 0.772458434 | 0.711090088 |
| 20 | 0.774080753326416 | 0.179354221 | 0.265362442 | 0.295052707 | 0.341156662 | 0.287881225 |
| 80 | 0.5330567359924316 | 0.150469795 | 0.196166500 | 0.201986313 | 0.211790204 | 0.155230910 |

endpoint80相对0的五分量降幅为22.9200%、35.0882%、51.5050%、72.5823%、78.1700%；
这说明同一训练病例对有限原recipe优化有响应，不能外推到未见天气场。
前20/末20行before-step total均值1.0095538407564164/0.5404378354549408。
首/末学习率均为原native算术2.0000000000000003e-6；原始80行及全部12loss均保留。

| checkpoint | state `_state_digest` | 文件SHA256 |
| --- | --- | --- |
| 0 | `a23444f530ed4ff80285f51e8b4a86829b27488683a8ee6d258dec4425e99d82` | `298eccd8b8c4b9648c3214e7d860c2ea8dbd7a5dc620bc15931762c5dd42b6b0` |
| 20 | `d5f61d3d4990c2bfe4e2ee965f5da28f3b266db4b3d2c74ad261ee0e9fbd01db` | `c982db31832d498f1ff08d833efc2e222402ab18833115fb699da1371f6eb846` |
| 80 | `c1a6f340cae2644dfe43100ef74e3f298b78b315243bd8531bb3699702434b27` | `98cbd2d51bb5ce7c3b55d544b7b234e91397d06c02d2ed328201dce6ab35c9e9` |

`core.state_identity`的原parent摘要ad13a1af…与`_state_digest`初始a23444f5…是两种算法，
不是state变化；按对应算法/全state inventory分别严格核对，不混比digest。

## 5. 实际执行与独立保存事实核查

GPU1 UUID `GPU-9d1624af-9d77-aa7c-0620-b6cb778f4ced`，默认共驻。
startup及每spawn只读即时free14253293568B，均满足known2491416576+2GiB=4638900224B。
首步reserved2409627648B、整训练reserved2434793472B，下一门仍带更大的known值，不缩小margin。

| worker | whole worker秒 | 终态 |
| --- | ---: | --- |
| CPUprepare | 482.2013135869056 | exit0/reaped/signals=[] |
| train80 | 1050.4287329651415 | exit0/reaped/signals=[] |
| CPUreading | 11.147346183657646 | exit0/reaped/signals=[] |

worker和1543.777393s小于整轮1547.848237s；费用按最后attempt account，
ceil1548/3600=0.4300GPU-h，不把内部训练/梯度时间替代整轮。

独立assignment `/tmp/r7_fixed_case_terminal_assignment_20261007_attempt01.json`
SHA `1218a3a0572ded85165af8172111cfb395a03a21c8fd370f20d6c710de01dc43`覆盖807文件。
标准库独立核全部FP32逐项累加、80步schedule、mask/count、完整模板与recipe、
端点/state framing、saved gate/process/cost及first/end SHA/bytes无mismatch。

首次terminal审阅checkpoint worker在cleanup截止被ownedSIGTERM、rc−15/reaped，无完成回执；
审阅whole1964.930446s超hard1800的164.930446s，标incomplete，不算通过。
另冻结600/1200纯CPU补齐只读checkpoint，13.571614s成功：普通strict loader、
独立构造语义/key/shape/dtype/FP32finite/state/RNG原生seed41、完整CUDA16byte CPU列表、
AdamWflags/slots/step/moments均核；producer validator仅secondary。
completion receipt SHA `b2c0276818f4aa188b4afe82c686c31e87a334c375e463dfe5eacbd4661d5ad2`。

保存事实审阅未follow source/store/weather/旧parent路径；完整源hash和实际CUDA对模板验证由
执行worker完成，不能把独立savedfact审阅说成独立重新获取天气或全运行时追踪。

## 6. 工程资格、失败与预算偏差均保留

| 阶段 | 实际/软/硬秒 | 如实状态 |
| --- | --- | --- |
| 初版training TEMP准备 | 1234 / 600 / 1200 | 旧34pass1fail，hard超34，非PASS |
| 首次修复失联 | 精确elapsed未知，hard截止05:20:53Z | interrupted-unqualified，部分快照保留 |
| 恢复修复 | 1268.007564 / 600 / 1200 | hard超68.007564，额外检查截止取消，partial |
| 独立路线准备 | receipt372.154166 /180 /360 | hard超12.154166，总含开场/报告更长未知 |
| 独立完整training资格 | 1671.521 /900 /1800 | 86过但2外锚coherent反证漏检，negative |
| 两外锚最小修复 | 1230.119210 /600 /1500 | 同pins91过，等待独立 |
| 独立两外锚定向 | 1032.230798 /600 /1500 | 91过与完整coherent反证闭合，限定CPU资格 |
| readback TEMP准备 | 2386.537055 /1200 /2400 | 50过但deep未齐，收尾余13.463<90违规 |
| readback独立初审 | 1966.954969 /900 /2400 | 50过但4scope全链漏检，negative |
| readback scope修复 | 1082.041479 /300 /900 | hard超182.041479，未spawn50/deep，failed |
| readback独立scope闭合 | 630.870804 /600 /1800 | 50过、deep正例、14scope双入口28拒绝，CPU资格 |
| terminal helper准备 | 1662.336293 /900 /1800 | 合成10方法过，真实集成未核 |
| 首次terminal审阅 | 1964.930446 /900 /1800 | JSON齐、checkpoint取消、收尾hard超164.930446 |
| CPUcheckpoint补齐 | 13.571614 /600 /1200 | 三端点语义核成功，旧取消不追认 |
| 反算终态独立审阅 | 948.497180 /300 /900 | savedfact齐、收尾hard超48.497180，NOT_QUALIFIED |
| 反算终态冻结helper补齐 | 5.029937 /300 /900 | 同独立helper原字节及1532pins实际核成功，旧失败不追认 |
| 文档首次事实审阅 | 581.968135 /180 /600 | 收尾余18.032<90，当前逐JSON末hash未齐，issues/not-qualified |

所有skip/cancel/partial/failed与超hard都不算通过；新的限定资格不回写老阶段状态。
原16RNG/AdamW/report/contract缺口，以及后发现的完整coherent native RNG/活动改写，
分别由外部冻结seed41/native状态和原病例参数活动mask闭合；runtime变化失败留partial。
readback四scope反义必须与实际producer schema一致：required科学/heldoutfalse要存在，
已存在optional严格typedFalse；未输出字段保留缺省状态，不能defaultFalse伪造证据。

精确生产archive CI37543875564、预启动文档f00a0de CI37585610390及实际进度28c2458
CI37590994358全部必要steps success；17标签门控实验skipped未算实验PASS。
TEMP最终training独立91项、readback独立50项及deep反证均实跑；生产代码未改，
本轮不重复既有CPU全量4003/9skip为新跑，也不冒称GPU全套回归。
首次文档事实审阅核主要读数并指出第7节过期标题，但581.968135s未满足硬截止前90s冻结，
当前逐JSON末hash未齐，receipt `b60ad6c2063d7296f33e98efe955c9ac4b3621c499f548b0069cc52e44e5936e`
保留issues/not-qualified；修正标题不追认该阶段资格。

## 7. 独立三端点反算（真实完整，限定终态保存事实审阅已补齐）

另冻wholeplanned1800/hard3600、reserve90、endpoint0/20/80。
原归档/全source/preflight/BUILD_COMPLETE/train/window/新80contract外锚先核，
独立调用原training_long_rollout，不复用训练endpoint计算函数、不创建optimizer/backward/step。
原model.train/alltrainableFP32/enable_grad/dropout0保持，同cachedbatch与全部12监督不变。
保留原/重算全向量及typed exact差异，不allclose/epsilon；不一致failed-exact-readback，
不重试到偶然相同。最高configuration-only，不主张全GPU训练逐位可复现。

mandatory父115files外锚pins SHA
`0285b166ecc637ebee02f3852ebbe16823de7ca306cf7f4daf0edf14f1404507`。
readback driver `4df86db1f6ea8c4a7f7d4b6e0e4c42259e3c8d4e64fda80419e231e7b07fe517`，
support `f4408d2ed919998c04f901b92fac3ac800c9de2aa35452f62767ce1bb22f685d`。
真实三端点已完整重算，逐项typed projection的全部12loss、total、state及recipe/RNG/flags均精确一致，
各端点difference_count0，合计0；没有epsilon/allclose、无新增容差或恢复重试。
这只证明这些已保存端点在本配置下可精确重算，不升级为普遍bit-reproducible GPU训练。

| 身份 | 精确值 |
| --- | --- |
| readback final canonical | `6b9be74b035a19f47407eb74158e342ec15e7c4f37bfe0f4271c6eb9920801a6` |
| readback protocol bytes | `b549223f235a038ffb997ce0dd856c54979ce01d52e1fc9fd8489651fe0f9560` |
| readback result | `663e085a2c27514beb9af6eb5740264361307101876333047e372c3077585372` |
| readback attempt | `e1e23874d804bf3a7ce255b7f9633970809f917592eaeecb2832fd7e69218a8e` |
| terminal725+parent807 assignment | `57f4cd1bf1ea57e92e303e8e7f98e8251e0bc2b6872cb4c6ab4006c9bf4759cb` |

prepare/readback/reading worker分别458.0225573834032/988.1744947116822/12.850959781557322s，
均exit0/reaped，recorded signals空。whole1467.4802325293422s，planned1800/hard3600均无超；
ceil1468秒保守记0.4078GPU-h，与原训练0.4300分列，合计0.8378。
reading时free25280118784B，其余门14253293568B；known2491416576+2GiB持续携带。
首次反算终态独立审阅完整savedfact已核，但收尾whole948.497180s超hard900的48.497180s，
标FAIL_STAGE_TIME_BOUNDARY_NOT_QUALIFIED；其receipt SHA
`775607a268588603f9094d4f2cfeb27e6ae333879fb433179faada5a7d194de0`不追认。

独立标准库helper原字节 `01b1e0724bda10a012023cdd64731703ea8469230a90362685f025a1febe183b`
在新排他临时路径、另冻结300/900时钟重新执行，5.029937s、rc0/reaped、soft/hard超0，
725+807文件first/end SHA一致；独立fullprojection、115父pins、692archive、pre/final/prepare、
comparison/reading/result canonical、三worker/gates/peakcarry和费用0.4078全部核。
此新执行结果SHA `8ccd747afa09b70ef800a6bae5ebcf22d6e67cb8589bd869224798c71b5506f8`
与原完整savedfact相同，但不把先前预算失败追认为合规。
completion receipt `/tmp/r7_fixed_case_readback_audit_finish_20261007_attempt01/completion_receipt.json`
SHA `fa8d67c2d52f41224140decbacd68effc46111c3fa52bf1b91afb67ec651fd62`。
限定工程savedfact/精确反算资格不等于科学接受，未新读源/weather或重新forward/训练。

## 8. 限制与科学状态

单个in-sample病例重复拟合存在记忆效应；明显下降不证明未见数据、全年气候态或门控变量收益。
normalized训练损失不是物理天气RMSE，不能跨单位解释成温度改善。
这里的有限响应不重裁旧S3 screening，也不授权重复同剂量直到成功。
旧代表gradientreplay的161numeric差异与failed-restoration不放宽、不开attempt03。

Python离线/import/owned guards不是OS沙箱；记录signals空不是全局无邻居信号证明。
源身份、余量门与冻结证据强于摘要，但没有原始梯度张量或内存操作全链追踪。
只做方向内本地授权研究：未租卡/付费/独占/main/force/issue关闭/cron/凭据或安全配置变更，
未改protecteddata、旧产物、归档或用户配置，未新增依赖。
当前仍S3，S4、2023评分、r消耗、完整未见全年/三seed同时区间与最终goal接受均未完成。
