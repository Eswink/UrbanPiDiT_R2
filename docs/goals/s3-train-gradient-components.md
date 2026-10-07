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
- 未做：本轮实际四case metadatafreeze/source新核/梯度/GPU/结果/成本/重放；不得预写通过。
- 下一动作：新排他outputs冻结1800soft/3600hard，CPU身份准备后发布四例protocol，首例测peak后单worker完整其余三例并纯scalar判读。
