# S3 同病例 train/val 诊断：长 lead 失败亦见于 in-sample train

**同病例24例诊断完整；长监督改善父模型，但t2m48/72h仍输气候态，不能归为仅val泛化失败。**
`scientific_claim: false`。train为in-sample、val为开发，只有seed41、每年/季节块一个metadata代表病例。
本页不是正式候选screen、显著性、因果欠拟合判定或科学PASS。当前仍S3，2023test未评分，S4未启动。

## 1. 固定身份和工程资格

运行根 `outputs/r7_s3_same_case_gap_20261006_attempt01/`；执行代码
`61e46bd78d4d7ce49da7a87d574e40c6c18a1c99`，工作分支r7/weather-reasoning。
精确执行SHA主CI [37543875564](https://github.com/Eswink/UrbanPiDiT_R2/actions/runs/37543875564)
已completed/success，全部必要steps success（公开API访问2026-10-06/07）；标签实验skip不是研究PASS。

| 身份 | SHA256 / 值 |
| --- | --- |
| code.zip | `72953e44fa814db1e1da4d7f5c7505bcb878dc2a9765bc9f8875022a822b6ea5`，9,789,745bytes，ZIP comment为执行SHA |
| canonical protocol | `8b336951185b092ad30b05f4ba2eb7aa6b21828ae8601378b22dba0a1c7ca216` |
| protocol文件 | `65bca5eca7dd041cc5a15eae229bce8464d88bd62845771cbe90d653d3f14b7e` |
| measurements | `81c6ab827817288df87def06118720b5159793c2d5e9ae0f742a4a5842e4dd75` |
| result | `a72b2901a3bfa7ac45454c4615b41481d6fe74cb2ef237a6ed026e0b19babb5c` |
| measurement receipt | `587af288792cbead46d059948ec281afd6dec587cfdcef0292ad3a22a692144d` |
| reading receipt | `70e626f6e2c6636a3b8214bdfbbbba1cd15ecf467e4fc77c71d20161a9678a6e` |
| attempt | `3cfeab81fb9a9da7ed775036c2801d3f03624a5c0d1f7af590d7847831478c22` |
| selection canonical digest | `b6b0e23231271fe71deb07ab6e6f91f5f7bb653acefb5214c5e6755f2c629706` |
| full source / bytes | `bc2ff9cfadcce604fc243bb999b3c430d5164d17fcf1de201273716a5db065f8` / 540,856,239 |
| train identity | `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac` |
| val identity | `0c34a887216b02b41716e7837f5be4d515ad2027e7b4a4728cbef15accd02d5a` |
| model source digest | `3ddab46b1e4c2c7e66449e39ab8247c9c7e642f45023c1c9cc14bca8f74fd217`，模型不改 |
| parent seed41 BD1600 | `637857c5d5cc8b3c9e33448cae4dd1b53aed5f525639d6911f03d7420492bdac` |
| candidate seed41 long200 | `449bc4ce6d9c15178fecb75a954a265ffb43f7ea8cea92575d95387047ee066b` |
| parent / candidate contracts | `8e2d540c01962b652f4542a9b526e3963fa9189cfa4c903e3dd32a094fa4315e` / `944e6741bfefba9fce090c7ed69e74c90db9c63fa0341ba6ffabd10af973def9` |
| climatology mean identity | `c633df585b1894704d942e24305047a74ff7af2460cbc24ca2ebe7be1948a205` |

新core/driver不改heldout-only evaluate_local/ZarrRolloutDataset接口、model、源/派生数据或旧证据。
严格r7-local-v1 code/signature/state-template/形状/dtype/finite/语义/source/data/window核验后只导入模型；
optimizer_updates=0、optimizer_imported=false，无AdamW构造/step。固定FP32/K4、12次target-free物理递推。

实际qualification：122定向CPU反证63.50s；完整CPU suite **4003passed/9skipped/6warnings，1210.87s**。
六CUDA用例在CUDA_VISIBLE_DEVICES空时skip，另三可选真实fixture缺失；skip不作通过。
213治理tests38.05s、installed-wheel1passed24.75s、37阻断conventions与两个campaign/goal/index一致。
完整suite日志 `/tmp/r7_gap_full_pytest_20261006.log` SHA
`61cfae963c31634e007ae7070053b45d313f2ae773e4e9ce612178be7e29bfa6`。

prelaunch独立审阅983s（soft600超383、hard1200未超），无剩余essential blocker。
真实tinyZarr core→reading24例阳性39.116s，最终新header反证6passed/56deselected5.04s；仅工程fixture。
历史qualification失败保留：误用forecaster→实际backbone的接口修复、fixture共享dict→deepcopy，
停止/不完整及并行旧fixture13failed/20passed/84deselected11.91s均不追认PASS。test-only metadata快照
仅加速合成读法，production未改，独立阳性没有该快照；旧断言/规模阈值/科学门未弱化。

## 2. 先metadata选病例，再读取字段

train2017–2021每年months1/4/7/10各一例，共20；val2022四月块各一例，共4。
每组完整12步窗口107个，lower chronological median rank53，init均该年/月14日12:00UTC。
精确历史t−6,t和每一步t+6..72h均属于同split；选取前不读取state字段，不按误差选样。
原train2360manifest→2140完整/220排除；val472manifest→428完整，不能拿排除数当val病例数。
全部record hash、索引、history/target/valid times、日历/units/normalization与ordered plan一起冻结。

五lead6/12/24/48/72h、17变量、full65×65网格；cos(latitude)面积权重，先逐病例物理MSE再equal-case平均，
RMSE取pooled MSE平方根，不平均逐例RMSE。物理误差乘原train-only std；不同变量单位不直接平均。
原climatology只fittrain2017–2021、2400时次、16month/hour桶各150，once fit，不换弱逐lead baseline。
train气候态统计含这些训练目标，in-sample乐观必须保留；val只有开发四病例而非全cohort。

## 3. t2m物理RMSE与skill：改善但长lead仍负

| split | lead | parent RMSE K | candidate RMSE K | climatology RMSE K | candidate MSE skill |
| --- | ---: | ---: | ---: | ---: | ---: |
| train20 | 6h | 1.942122 | 1.901798 | 2.578106 | +0.455839 |
| train20 | 12h | 2.285324 | 2.208156 | 2.439344 | +0.180567 |
| train20 | 24h | 3.206134 | 3.012047 | 2.938226 | −0.050880 |
| train20 | 48h | 5.127242 | 3.993355 | 2.993792 | −0.779233 |
| train20 | 72h | 6.324184 | 4.099891 | 2.897206 | −1.002562 |
| val4 | 6h | 1.965378 | 1.937445 | 2.478372 | +0.388881 |
| val4 | 12h | 2.205122 | 2.041813 | 2.342907 | +0.240510 |
| val4 | 24h | 3.067046 | 2.678474 | 3.572763 | +0.437961 |
| val4 | 48h | 5.654964 | 3.744069 | 3.239997 | −0.335360 |
| val4 | 72h | 8.222445 | 4.411828 | 2.968548 | −1.208763 |

candidate t2m五lead在两个split均低于parent；48h train20/20与val4/4病例改善，72h train19/20、val4/4改善。
这不是17变量所有lead全部改善：candidate低于parent变量格数train **7/8/14/16/17**、val **9/10/12/16/17**。
candidate正climatology skill变量数train **17/17/16/5/3**、val **17/17/15/4/2**。

candidate t2m48/72h pooled train和val均负；train每个year组和month组的这两个lead也均负。
逐病例仍有混合：train正skill例数 **17/11/10/4/2**（分母20），val **4/2/3/2/1**（分母4）。
不能用部分好病例盖住pooled/组失败，也不能把24例充当三seed或真正未见全年确认。

candidate t2m val-minus-train RMSE为 **+0.035647/−0.166343/−0.333574/−0.249287/+0.311937K**。
parent相同gap **+0.023257/−0.080202/−0.139088/+0.527722/+1.898261K**。不同天气病例分布，
gap既非同天气反事实也非置信区间；不能认定模型欠拟合、收敛、clip因果或没有泛化误差。

**已确认**：长lead失败不仅在val出现；直接长监督确实改善这组train/val父相对长lead误差，但还没达到气候态。
**仍是推测**：优化信号、表达能力、长期递推误差和有限区域数据制度可能共同限制；本诊断不能区分。
不据此盲目增加同一剂量，不耗test调到赢。下一个独立诊断优先检查同case长监督损失分量与梯度对齐，
仅train/无optimizer先测信号，不把saved600条clip前norm>1直接当clipping损伤证据。

## 4. 成本与读取范围

| 阶段 | 秒数 |
| --- | ---: |
| prepare | 540.8596328208223 |
| archive | 1.9696422405540943 |
| measurement | 1203.2677189037204 |
| reading | 5.028186560608447 |
| 四worker总计 | 1751.1251805257052 |
| root间隔/开销 | 0.2960169641301036 |
| published整轮 | **1751.4211974898353** |
| soft1800 / hard3600 overrun | **0 / 0** |

保守整轮GPU-h **0.48650588819162094**，账本**0.4865**，不是GPU利用率或独占时长。
measurement内部fit/推理/指标区间100.14822249673307s；准备/元数据/身份/复载等其余时间仍全额计费，
未分别测出其因果归因。publication tail和外层进程exit延迟未独立计时，不能称完整外层亚秒精度计量。

owned reserved peak92,274,688B、allocated59,976,192B；所有spawn仍用更大的known训练峰值2,434,793,472B，
加2GiB margin需要4,582,277,120B=4370MiB，记录free25,280,118,784B。四worker均exit0/reaped、signals空；
owned workers无记录信号，未发现邻居干预；无全生命周期OS tracing或全局无邻居信号证明。
未独占/付费/下载/配置或凭据变更；gate无独立观测timestamp，不将receipt当生命周期追踪。

静态读取范围：两次全文source hash共1,081,712,478bytes（包括test年份源字节的工程身份，不是test天气评分）；
2400climate train+280train history/12targets+28val history/五targets=2708字段请求，576模型transition，6120MSE格。
不读取test.jsonl或test state评分；以上是源码/receipt范围，不是OS级访问记录。

## 5. 独立复核：失败predicate保留，独立算术互补

第一次终端audit `/tmp/r7_s3_same_case_terminal_audit_20261006_attempt01/` **FAILED**，不改时限或5e−14/0 predicate。
`audit_result.json` SHA `339374c4602b5780f85c126dc5ea22e78705a4372014b60e85f58e6874ca10f7`，
elapsed3.3476998833939433s、through-publication3.354258810169995s；owned CPU worker exit1/reaped/no signals。
失败前已核23terminal artifacts、177execution entries、metadata/source/units/normalization/选择及全部24case算术；
aggregate中train48h t500 skill超过冻结相对误差线，后续strictCPU/process阶段未在此attempt完成。

另立descriptive supplement协议（非复活/改predicate）：
`/tmp/r7_s3_same_case_terminal_supplement_20261006_attempt02/`，600soft/1200hard，9stage、383file identities。
独立NumPy axis0实现（不import core的summary/aggregate/gap）对51,340指标numeric leaves+144浮metadata、
19,500typed结构叶精确同stored；包括6120 MSE输入、全部case/组/gap，0nonidentical。
独立math.fsum重算：case0差、aggregates4705差、gap377差，**5082非精确数值相同**，全部residual留存；
不新设容忍或epsilon，也不据此把audit01改为通过。

拒绝格t50048h train20：candidate MSE12.093321804642287两法相同；climate分母stored/NumPy
12.072168439423947 versus fsum12.07216843942395K²。skill −0.0017522423849934692 versus
−0.0017522423849933217，abs1.474514954580286e−16、relative8.415017050200539e−14>旧5e−14。
这是reduction顺序及相减cancel的实证，不是weather值不一致；失败事实保留。
最大relative residual4.198363724658305e−12在val24h v250 paired RMSE delta，abs3.552713678800501e−15m/s；
各变量/量纲分别保留absolute maxima，不跨单位汇总胜负。

supplement严格unmodified CPU loader/model template复载两checkpoint、训练source/code/窗口/语义/state一致，
candidate初态state digest与父一致；未构造/恢复optimizer，opaque optimizer字段仍按普通loader反序列化。
全部原gate/process/成本及source/BUILD_COMPLETE重新核；只读audit不重生成天气或climate场。

| supplement artifact | SHA256 |
| --- | --- |
| supplement_result.json | `2a8959724b56ca953b2b49486514250ec3d401030fef1a17c0148f2b003f5870` |
| supplement_report.json | `9b7d6e0461d359c4fb6252c68c91b238edc2273cc0612b5dcd63c7eab122a3f0` |
| arithmetic_characterization.json | `2c12ec2ed4d5b7ec9e81cae6ceddf4a8a759ff57743130e21664c9e4b554216c` |
| fsum_nonidentical_residuals.json | `320cfe6d44d62d85d868600618e1c98d687a3e7b67493f43048a166f59ea7f03` |
| canonical supplement protocol | `a087cd298e2c107ce8e9175479fda8133b43329a29c965e869fa0d9797405050` |

supplement elapsed8.92634251434356s、through-publication8.934714877977967s；overrun0/0、worker exit0/reaped/no signals。
失败+supplement计12.288973688147962s，不含另外已声明post-run receipt inspection；纯CPU不冒称GPU耗费。
独立复核不是最终goal verifier，不能授权scientific complete。完整天气数值再生属下一节归档重放。

## 6. 归档代码重放与终态

完整重放于 `outputs/r7_s3_same_case_gap_replay_20261006_attempt01/`，三worker exit0/reaped/no signals。
wrapper预先独立审阅无essential blocker；103synthetic CPU反证通过，独立103passed0.83s，
总静态/synthetic审阅292.499s在soft300/hard600内。直接owned helper/devnull O_RDWR实测可spawn/reap，
无跳过此路线。wrapper595行 SHA `62a0d93f5c0a34d9f98fad7f241943de4eadf655fff343f560009a527d47a802`。
原exact code.zip、原protocol不改，另立1800soft/3600hard新deadline，不复用已过期旧截止。

全24病例、两模型+once-fit原climate、6120MSE、全部病例/组/gap/单位/源/端点/气候态mean identities
native-float JSON projection精确相同，numeric_difference_count=0；完整original/recomputed payload均保留。
只排除top-level elapsed_seconds与两个CUDApeak，device_type不排除；不是容忍误差或仅重算摘要。
投影SHA两份均 `936b7d265f22912c21fecda8943dc13149b391667a806910df29e01dc03aa406`。

| replay身份 | SHA256 |
| --- | --- |
| canonical replayfreeze | `e18034c17d15f82cfba8372d5a8379f1ec2b3843f48b5101e2b4bebdc2be5835` |
| replayfreeze文件 | `c34cd73c541f13e48da3f603d97e75f6f5344f47aeffa791a12a4a41355c523d` |
| fullresult.json | `647274a1f22b273f22eb406c6c547554036219436a77762c8e6b2ecfe755fb57` |
| RESTORATION_ACCEPTED.json | `1a8a20d5d03bf8cd154368e8c157d39b5f41567184c2d6b82dbfe8c71b753201` |
| recomputed_measurements.json | `e921acde13c8d483f22af94f6002ac2006430700283fdd6f15cd1e33fcdb5219` |

整轮2844.160050935112s，soft超1044.160050935112s、hard超0；prepare539.843382947147s、
measurement1756.517423166893s、reading545.616955853999s，全部source/metadata/导入/判读成本均计。
保守GPU-h **0.7900444585930867**，账本**0.7900**（直接elapsed/3600，不套用上一replay的ceil规则）。
每spawn已记录UUID/free/gate时点并保持known训练峰值+2GiB，不用小推理peak降低门槛；原产物不改。
原诊断0.4865+replay0.7900共 **1.2765GPU-h**；前账17.8324合计 **19.1089**，cap20/余0.8911仅会计。
独立terminal精确数值恢复gate已核：两payload各6120MSE+6120RMSE及全部病例/组/gap/气候态/身份精确相同，
三worker on-disk receipts、四gate、清理reserve、新deadline和publication before-hard-limit证据均一致。
工作JSON `/tmp/r7_gap_terminal_audit_20261006_z8x9z0oq/terminal.json`，SHA
`e9d2ba47337fcd91259aaede36ae6625a6dbd1b3a368622c36afbf24f92bbb22`；阶段226.115s在300soft/600hard内，
实际verification1.431s含于stage，priorprelaunch292.499s另记，不能把两stage混作同一次预算。
审阅者第一次额外假定replay必须等原deadline过期，非合同条件失败audit.json原样保留；实际合同只需
独立新deadline，事实核正而未改变冻结numeric equality/时长门。source/旧NumPy-fsum审阅显式继承不重复。
同环境exact仍最高config-reproducible，不升级跨设备训练bitwise。

证据页数值定向只读复核52.366s（120soft/300hard），上述audit/supplement数字、t2m表/符号/成本匹配；
原“未信号邻居”改为owned记录与未发现干预、明确不证明全局不存在，“非逐位”改为精确数值比较范围。
此文字精确化不改变任何旧产物或科学判据。

## 7. 限制与未做

单seed、稀疏24病例、train in-sample自参照、val开发，full-only；无ACC/interior/edge完整报告、统计区间、
完整真正未见全年/四季、三seed正式确认。test未评分、r未消耗；不重开历史negative，不改冻结科学合同。
没有训练更新、梯度或clipping干预；长leadin-sample失败不证明单独优化/容量原因。
source/config/chunk身份仍非lifetime tracing；外部actual-C非Pythonprotocol具名live pin不在ZIP，
不能称archive完全自包含。最高config-reproducible，不作跨设备训练bitwise或SOTA承诺。
当前只登记描述性诊断；终极气候态越过门未达到，不自行宣布goal完成。
