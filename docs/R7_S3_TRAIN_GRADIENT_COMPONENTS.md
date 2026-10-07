# R7 S3：固定长监督目标的 train 梯度分量诊断

状态：描述性原诊断已独立核验；代表例精确恢复未通过，负面冻结，2026-10-07。
issue #76 的研究诊断，不是候选通过或科学确认。
唯一主计划 `docs/goals/main-model-climatology-campaign.md`；本轮目标
`docs/goals/s3-train-gradient-components.md`；科学接受只来自
`docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`。`scientific_claim: false`。

## 1. 范围与可核查结论

既有同病例证据 `docs/R7_S3_SAME_CASE_GAP.md` 显示：seed41 long200 相对父端点改善，
但所选 in-sample train 与开发 val 的 t2m 48/72h 气候态 skill 仍负。
本轮只检查该固定端点上的原训练目标是否具有可测梯度、各 lead 分量方向及大小；
没有更新权重，没有改 loss、clip、K、输入信息或 checkpoint，没有新增天气评分。

实际四例均为 train2021 的 January/April/July/October，初始化分别为各月14日12:00 UTC。
每例是107个完整12-transition窗口的 lower median，rank53，历史t−6/t及全部t+6..72同train。
病例在天气场读取前冻结，不按 loss 或梯度选例。全部17归一化变量保留原深监督，
不能将本页的 normalized loss 当作物理单位 RMSE 或气候态 skill。

**已确认的描述性事实：**

- 四例均完成五个分量及实际 total 的 autograd，所有参数/缓冲区状态与严格恢复端点的摘要一致，
  前、后及五个中间状态均锚定同一身份；`.grad` 没有累积，optimizer 创建/更新均0。
- 原155个 trainable参数张量中141个 total梯度非零，14个 unused/zero以显式mask保留，没有把None当失败或藏掉。
- 48/72h weighted梯度范数之和占五分量范数之和约76.0%–86.1%。这是欧氏参数空间范数的描述，
  不是可相加的因果贡献率、变量重要性或天气技巧比例。
- 6h–72h分量cosine四例分别为 +0.037387/−0.108328/−0.021354/−0.011985；
  48h–72h为 −0.000759/+0.815056/+0.810262/+0.591809。方向依病例变化，不支持“所有lead均冲突”的概括。
- 四例 total与每个分量的dot均正；因此不能将局部相消直接解释为各分量对total都反向。
- 原clip1实现的描述性比例约0.0649/0.0755/0.1645/0.0890，**没有实际应用clip**。
  preclip范数>1只说明若应用原算术会缩放，不能证明clip造成欠拟合、应调clip或应加训练剂量。

**未确认：**梯度对长期训练收益的因果作用、优化收敛/欠拟合、泛化、各变量物理天气收益、
全年独立确认、统计区间或主模型达到气候态接受门。本页不新增通过阈值。

## 2. 代码、数据、端点与原目标身份

产物：`outputs/r7_s3_train_gradient_components_20261007_attempt01/`。
运行使用原归档生产代码与临时byte-pinned诊断，不使用当前checkout加载旧checkpoint。

| 身份 | 精确值 |
| --- | --- |
| compatible生产代码commit | `61e46bd78d4d7ce49da7a87d574e40c6c18a1c99` |
| code.zip | `72953e44fa814db1e1da4d7f5c7505bcb878dc2a9765bc9f8875022a822b6ea5` |
| 临时driver | `633bccbd353a44a02bd7d96e63d1fc235dfd71011f41af83f2020dd3d5239bf8` |
| 临时core | `75c6a3f4708ea74ae6700841a6ec92e2bfdca744f69ca54c7c7e1b8bd89ad351` |
| 临时support | `dbbb9785ad12e421ca9d07a92c9daad0a1b4eeaeebc8ccfdd7a5b2fe06248460` |
| model_code_sha256 | `3ddab46b1e4c2c7e66449e39ab8247c9c7e642f45023c1c9cc14bca8f74fd217` |
| training_code_sha256 | `24dde71ba380312f2c201c8f5fd2cabb9c884585bb0ce904fdee768fc5f675b9` |
| candidate seed41 long200 | `449bc4ce6d9c15178fecb75a954a265ffb43f7ea8cea92575d95387047ee066b` |
| 原training protocol | `a1631d9b87721c2260a28c5862ef26ae7245fdd5f7c30cd404cb05708a8d10ea` |
| candidate contract | `944e6741bfefba9fce090c7ed69e74c90db9c63fa0341ba6ffabd10af973def9` |
| restored model state | `ad13a1af8a0007450b64dd079afe52a120120db6d67921a282c41c20983eab7e` |
| source | `bc2ff9cfadcce604fc243bb999b3c430d5164d17fcf1de201273716a5db065f8`，540856239 bytes |
| source preflight | `1b914a216bb3a79e044c316749bfad45a2a27bcd5df8bd07cd5cbf9ce4bd58e8` |
| train data identity | `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac` |
| train12-step窗口identity | `dccfeef926286ddeccb7690aacd7bac45c1bb2036fe6ebbe826841b971bb39ec` |
| 四例selection | `649c20bfa508f83a75b5cb905ac1352d3585509b63c8135a75f4d38688a0b9a4` |

严格普通CPU恢复核原r7-local-v1 contract、keys/shapes/dtypes/finiteness、模型语义和train/source/
归一化/BUILD_COMPLETE/窗口后才读取天气场。普通loader反序列化checkpoint中的opaque optimizer字段，
但不创建optimizer、不导入其状态用于运行。仅加载long200，无parent或actual-C臂。

原目标不变：model.train、torch.enable_grad、FP32、K4、12物理步完整internal/physical BPTT；
权重 `[1,.5,0,.5,0,0,0,.5,0,0,0,.5]`；K轴仍为initial及全部K drafts的原1..2线性归一化权重。
零权重物理步仍推进生成历史、保留图，futuretarget只监督loss，不进入model输入。
五分量autograd保留图，None转数学零并保存activity，total来自原逐项FP32累加而非另改目标。
每参数FP64 dot/reduction生成6×6 Gram，FP32梯度总和残差如实保存，不设残差科学阈值。

## 3. 四例完整描述性读数

列顺序均为6/12/24/48/72h。下面每行数值源于完整case JSON，全部12loss、6×6 Gram、
155参数activity、RNG/module flags和状态身份均保留，不只保存本页摘要。

| 月 | 五项原normalized loss | 原total loss |
| --- | --- | --- |
| Jan | 0.195213 / 0.302205 / 0.416510 / 0.772458 / 0.711090 | 1.29634428024292 |
| Apr | 0.135792 / 0.312651 / 0.594487 / 0.698795 / 0.570180 | 1.2238487005233765 |
| Jul | 0.122328 / 0.188955 / 0.264821 / 0.287408 / 0.305153 | 0.6454952955245972 |
| Oct | 0.096278 / 0.182731 / 0.391462 / 0.929460 / 0.613923 | 1.1550655364990234 |

| 月 | weighted分量梯度范数 | total范数FP64 | cancellation ratio | 原clip1比例（未应用） |
| --- | --- | ---: | ---: | ---: |
| Jan | 0.484893 / 0.590576 / 2.173393 / 7.562413 / 12.508724 | 15.41145594365589 | 0.66086863824846 | 0.06488679349422455 |
| Apr | 0.398935 / 0.865539 / 2.780833 / 5.548249 / 7.260323 | 13.240752040249031 | 0.785620448927343 | 0.07552441209554672 |
| Jul | 0.284917 / 0.335877 / 0.899370 / 2.160003 / 3.320194 | 6.078913099940799 | 0.8683714333895012 | 0.1645030826330185 |
| Oct | 0.270592 / 0.433554 / 1.624510 / 5.894737 / 5.396405 | 11.24021256746771 | 0.8252848521038937 | 0.08896627277135849 |

Cancellation ratio是实际total范数除以五weighted分量范数之和，不是统计显著性或新通过门。
cosine零范数分母明确undefined/null，无epsilon制造方向；实际四例所报分量均非零。
Gram对角cosine可能略超1（例如1.0000000000000002），保留原浮点数，不事后clip美化。

| 月 | weighted和与实际total的FP32最大残差 | FP64残差范数 |
| --- | ---: | ---: |
| Jan | 2.995133399963379e-06 | 7.419698673115482e-05 |
| Apr | 2.942979335784912e-06 | 8.755111144317598e-05 |
| Jul | 1.434236764907837e-06 | 2.8514852524597637e-05 |
| Oct | 2.1010637283325195e-06 | 4.621297471611665e-05 |

残差是两条FP32累加/反向顺序的保存观察，不冒称梯度张量逐位相等，也不由残差大小判断收敛。

## 4. 预算、GPU共驻与读域

| 阶段 | 整worker墙钟秒 |
| --- | ---: |
| CPU prepare | 528.9777133092284 |
| 首例probe | 972.7871515965089 |
| 其余三例单worker | 956.780424784869 |
| 纯scalar reading | 2.4720216589048505 |
| worker合计 | 2461.017311349511 |
| whole attempt含间隔/身份/发布 | 2465.08890417777 |

soft1800超665.08890417777s，hard3600超0；whole保守GPU-h
0.6847469178271584，账本四位舍入 **0.6847**。四例内部梯度interval合计17.71275949s，
不是整轮费用；不能据此删掉源hash/metadata/加载/归档/清理成本。
四worker exit0/reaped，保存signals均空。

GPU1 UUID `GPU-9d1624af-9d77-aa7c-0620-b6cb778f4ced`，默认共驻。
known→首例→全worker reserved为2434793472→2489319424→2491416576B，
每spawn保留max(2GiB estimate, known/measured)+2GiB margin，四门required
4582277120/4582277120/4636803072/4638900224B（4370/4370/4422/4424MiB）。
未用低推理峰值替代known训练峰值，没有邻居信号或抢卡代码；保存的门无独立时间戳，
实际gate-to-spawn立即性只能由代码顺序核验，不是运行时追踪证明；也不声称全局无邻居信号已被证明。

四例共48个物理transition、192个internal reasoning step；每例两历史+12target，共56场请求。
这些是源码/receipt静态范围，非OS I/O tracing。没有climatology fit、新训练或val/test manifest/天气评分。
确实解析了旧gap protocol内的val metadata；完整source hash含2023字节仅用于身份，
不能声称“完全未读取任何test年份源字节”。确认r与test标签曝光状态不变。

## 5. 独立审阅、失败与真实验证

初次prelaunch966s（soft600超366/hard1200未超）发现两个essential缺陷：
postprepare摘要被移除后hash失败掩盖原错误且丢失费用；病例state只有内部一致、未锚恢复端点。
原失败保留。最小修复后owner **113passed/191.20s，warnings0/skips0**；
独立定向 **45passed/20deselected，1.60s** 加三项scalar反证，整阶段181s、soft超61/hard超0。
20deselected不是pass，定向不冒称全套113重跑。两次审阅optional guard collect前失败合4.842s含在966s。

独立saved-fact helper `f6c7755dd879915aee0b2aab7f65d947702e3936308aa295c42d35f44d687e2d`，
11synthetic CPU tests；准备567s、soft300超267/hard600未超。新terminal阶段
326.754s、soft300超26.754/hard600超0，实际helper2.425s。所有 **721 assigned files** 初始/终末hash一致，
四例完整原顺序、FP32逐项loss、Gram-derived scalar、activity、state锚、recipe/单位/归一化、
两phase freeze、ZIP/Python inventory、四process/gate与全费用均通过saved-fact资格；
math.fsum分量范数之和无差异。没有source/checkpoint/store外部路径跟读或GPU运行。
此审阅不独立证明raw梯度张量/reduction、模型state重新计算或OS生命周期。

审阅报告 `/tmp/r7_gradient_terminal_audit_20261007_attempt01/audit_report.json`
SHA `fa5b43c8fb56791e84546c790f3f9b07f3cf240ed3055c09b672feeac82047c4`；
stage receipt SHA `fa53ed9d73f06c967671322c5701e2733dd112366a3c2a7047c850fd6dde8ec8`。

governance定向 **93passed/0.58s**，37blocking0失败，goal/campaign/index54结构通过。
默认本地全量 **4009passed/3skipped/6warnings，1257.87s**，4012JUnit、0failure/error。
这套没有隐藏CUDA，实际六项CUDA条件测试执行，**不是CPU-only**；没有该套专用启动UUID/余量记录，
故其取卡/协议证据有缺口，不能称R-054或有界GPU合规。JUnit起点至最终log1258.349468s，
ceil1259s保守另记 **0.3497GPU-h**，不改原诊断费用。deviation receipt SHA
`07023eb8d925ec0ba926ec4065a174b6ba9d67215bd986a9f8a4b505bbfbcc17`。
另冻明确CUDA不可见的CPU全套soft1500/hard3000：**4003passed/9skipped/6warnings，1202.81s**，
整owned运行1206.472975s、soft/hard超0、exit0/reaped、signals空，0GPU-h。
六CUDA缺设备skip与三optional真fixture skip均不算pass、不合成替代；明确CPU验证不追认原取卡纪律。
CPU attempt receipt `3a80cd958f033fa5f10ddf0f17b14295cb172de37f0bcdafd24ab5b99bdbd54c`，
CPU log `32f2926d34b25159b545f0d47fd095ce9ff8a4db67199285e30f26fddf063d70`。

兼容生产commit61e46bd精确CI37543875564success；前轮登记46ddd26精确CI37553027758success；
本轮prelaunch文档commit `aa9e4ac0c43f5f1295233d9b7446a34803736ea3` 精确CI
**37559461860 completed/success，全部必要steps success**。
17实验标签门控workflow skipped是设计行为，不是17次实跑或科学接受。最终证据登记SHA/CI待落。

## 6. 代表病例重放（failed-restoration，独立差异已核）

原结果读取前固定index0 January2021 lower median，不按梯度大小/冲突/通过选例。
另冻soft1800/hard3600/cleanup90，CPUprepare→single measurement及parent纯scalar/hash判读。
完整case JSON只排除新protocol摘要、elapsed与两owned peak，其余所有numeric/typed结构精确匹配，
无rtol/epsilon；失败保留、不得放宽匹配。最高仍config-reproducible，不证明跨设备或raw梯度逐位一致。

临时重放准备11synthetic tests通过，但整阶段1203s超hard1200约3s，保留为未qualified。
独立prelaunch provider中断保留partial成本，恢复后11测试/真实archive-only tinyProcess完整12步阳性/
25-file原receipt接线通过，但确认bootstrap失败记账遗漏，因此**未获启动资格**。
bootstrap极小修复后owner15tests通过，修复157s soft180/hard360未超；独立定向255.218s（soft120超135.218/hard300超0）
核15tests/4B1反证和25-file接线且确认只改main异常边界，工程qualified后实际启动attempt01。
但实际prepare在code.zip的fixture `val.jsonl/test.jsonl` 排他写入时被basename守卫拒绝，
未进入真实字段/模型恢复或GPU梯度；这种归档复制与实际持留集读取冲突在前述资格样本中未覆盖，不能追认实跑成功。

`outputs/r7_s3_train_gradient_replay_20261007_attempt01/` 保留failed-restoration：
whole1.8343665357679129s，soft/hard超0，保守0.0005095462599355314GPU-h，登记四位 **0.0005**。
唯一prepare worker0.966528s、exit1/reaped、记录signals空；后续measurement未spawn，RESTORATION_ACCEPTED不存在。
failure.json `c7b8fc4410cc4e128f2a5ae96c746bc2066b682bf99d90e333b86d954af8825d`，
prepare.log `d5f6d76fdc0c04f474fa5349cc8f02ff99e870a310e17738fd09eff765771880`，
replayfreeze.json `d53b4e1966158b317d73d5ee6e2ac14fdb0aa4b77e51be6473130c668faab77e`。
只修归档复制/实际manifest读禁的工程边界，保留code.zip全字节身份与actual val/test READ拒绝；
另立新路径/预算/qualification才可重执行，不覆写或续跑failed attempt01，不放宽数值或科学判据。

replay-only extraction adapter随后只落全部692个verified Python成员，完整ZIP/hash/comment/inventory仍核，
非Python fixture manifest不落盘；实际manifest READ/write守卫逐字不变，model/training digest不变。
owner16synthetic tests通过，但修复399s超hard360约39s、soft180超219s，保留为未qualified阶段。
独立新阶段16tests（5.882s）及exact-ZIP worker（7.012s）：全Python extraction/原guard/archive-only imports/
两次unmocked tinyProcess full12/精确repeat及ZIP/code mutation拒绝均齐；314.312s、soft180超134.312/hard360超0。
receipt `6bceff7a6f10181a0ba04c63e54ae5dd516122abd7cfb9976c105c1919967599` 只表工程资格。
新support `ab1e7609a811a593219c78e1fa27eec6352438f29b0091f55725832f551f238c`，driver仍586c3f4e...。
新pins `1ecc06c80a5e63e7be44390ea6ae0a918cca7a7d7123bad998aa6f338d30189a`，
新排他 `outputs/r7_s3_train_gradient_replay_20261007_attempt02/` 实际完成两worker，却在parent完整JSON比较失败，
因此 **failed-restoration**，不是精确数值恢复。原匹配规则无容差且保持；未启动attempt03去重复至偶然精确。

whole **1569.81766172871s**，soft1800/hard3600超0；保守 **0.4360604615913083GPU-h**，账本四位 **0.4361**。
CPUprepare545.3553357943892s、measurement1021.508986058645s，均exit0/reaped、记录signals空，
但worker执行成功不替代terminal恢复判据。完整original/recomputed case0均保留，RESTORATION_ACCEPTED不存在。

独立终态审阅确认 **161个numeric leaves不同**，与主链保存的全部diff rows逐项相同。
这是重复表示的Gram/派生统计，不是161项独立测量；6×6 Gram36项全不同。
all12loss/objective、state锚/RNG/activity/qualification/module flags、FP32 clip norm/factor及FP32最大残差精确相同。
所有比较的cosine符号/方向保持，但**不能挽救精确恢复判据**。
差异与FP32 autograd非逐位确定性相容，没有raw梯度/kernel tracing，不建立唯一根因或跨硬件证明。
不因差异小就另加epsilon/rtol或删掉困难字段，不将数值相近重分类为PASS。

| 字段 | 不同leaf数 | 最大绝对差（描述性） | 最大相对差（以原值为分母） |
| --- | ---: | ---: | ---: |
| gram_with_total | 36 | 2.399301971e-5 | 3.180095968e-5 |
| weighted_cosines | 23 | 6.060825331e-8 | 3.180183502e-5 |
| total_norm_fp64 | 1 | 7.784151084e-7 | 5.050886245e-8 |
| cancellation_ratio | 1 | 3.266454085e-8 | 4.942667719e-8 |
| weighted_sum_residual_norm_fp64 | 1 | 1.314087107e-7 | 1.771078806e-3 |

这些没有新增接受阈值。独立原四例重新审阅及Jan/recomputed各自audit_case通过saved-fact资格，
两phase freeze/全部原copied pins/source/checkpoint provenance/fullZIP与692Python一致；无外部source路径跟读。
三timestamped gates carry原最大2491416576B+2GiB=4638900224B，gate至worker start间隔
prepare0.000490198s/measurement0.000347340s；仍不是global/lifetime OS证明。

独立negative-replay审阅whole373.965s、soft300超73.965/hard600超0，1462文件end-rehash零不符。
独立result `/tmp/r7_gradient_replay_attempt02_terminal_audit_20261007/result.json`
`681cb41b69116163be156e59c49cf1f8753a390148a7c2e792782fc46f0c7aea`；
全部rows `ed24cebe419495e02397686eaaa60f19f5fceafc3887bb730fa16312b6050855`；
final receipt `0bd6742ba828a79e690d712d5b2a6ccfde395eb8fd307b38f64c4b252101d3b1`。
旧attempt01也核failed/cost，两个失败均无RESTORATION_ACCEPTED/成功attempt/fullresult，不能据worker exit0标恢复成功。

failure.json `5b994e3511f2501ad51aac1e44bee48635dfc3680830f3cb8b10ce09772ced2f`；
recomputed_case0 `fcfec322597a217b9883db42002ccd4c861e684918c59e45ba6a85b7af47f381`；
replayfreeze `8fec4c744a0ca08146c3479d0278d9a09426e58110321f1fdc5eb55eaafb5787`；
terminal assignment `1bc89d0b3332ebaf6a2fd33c290b502385eade4747c2229472c21bec28c4163b`；
主链descriptive diff `5530320add223605ba8aa24a1c7e029382b9a500112045ee9ce05a71aad7d92b`。
登记原四例时必须连同本失败、限制、全部费用，最高config-reproducible，不声称精确梯度恢复。

## 7. 产物固定索引与解释边界

| 文件或digest | SHA256 |
| --- | --- |
| preparation canonical | `e977ccfc4a63b6f6c7b5291b78b08e53649adee287cf34a45cd515da378fb516` |
| preparation_protocol.json | `a9e38c3e24b2ff55b1721f46486d3ee80ecd0bf00ce47a9a15585a394b6c08b9` |
| final canonical protocol | `44b634ed478d398754dfa87bebd176cdd060048385f04e2f7f8713a827c4ddc9` |
| protocol.json | `69f73009b0429f840575898e26eb0186799949d2168146fcb1bf4e072aae243a` |
| prepare_receipt.json | `b5616ba732970031670caad42f5cb0babe29a8b518d830b47ab87c859356ce3f` |
| reading_receipt.json | `8aae2362bbb168b90edb3aadc136b1c9d669de29876ee74063bbc190cc74a037` |
| result.json | `a9c7da6d1fe28c177f952ba61423b436430f6e90198f31edb86bdf654fbac11a` |
| attempt.json | `7881a3c78bca5a46d9d2c61efc5ddc0f36869686641b2f67d8467e25cb8caffd` |
| case0 January | `3e92dd4e1780bf7ed5291c95a172774fb808cb0d6b7ce8638a4ca15673af49ca` |
| case1 April | `afe18bccf738b6909d3aebc812db824365ab13169cda236326c38ac13f130319` |
| case2 July | `d3ace693f3320395ffb2e3fe9c328a5196913cadb0be4c608d1e49ed44d53bee` |
| case3 October | `1d28d8cc1b7342cb59e10a2d35bc9eb33fcfe25e8ec0644523da2dfb69ee0d29` |
| terminal assignment | `f8bb95d93c28494fc63a26c4c0f9eecbf3d662afa1ec4f7cc84b7dbeeab6e0fd` |

旧模型/训练目标/evaluator/数据/归档/failed/注册证据页未改；没有接口、依赖、凭据、安全配置、
付费/下载/main/issue关闭变化。临时工具只在owned fresh路径写产物，不进入wheel。
原诊断、两个failed replay和混合全量测试另项费用合计 **1.4710GPU-h**；
方向累计19.1089→**20.5799GPU-h**，cap20/remaining−0.5799为会计，不是总GPU-h许可闸门。
纯CPU审阅/测试/准备墙钟及其软硬超时分列，不重复算作GPU实验；不能删除准备hard超时或原测试取卡缺口。
本页冻结描述性及负面事实；后续登记索引/账本/精确CI只追加，不改已注册页hash。
精确恢复未过明确保留，不将本轮诊断自裁为科学goal complete。
