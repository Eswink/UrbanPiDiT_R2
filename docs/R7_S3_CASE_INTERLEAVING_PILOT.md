# S3：等病例曝光的训练顺序干预小试验

2026-10-07；`scientific_claim: false`、`scientific_pass: false`。
关联 #76，未关闭 issue。科学合同只引用 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`；
本轮冻结开发比较见 `docs/goals/s3-case-interleaving-pilot.md`，不改科学接受门。

## 1. 实际完成与预声明范围

`outputs/r7_s3_case_interleaving_20261007_attempt01/` 已完整执行：
两臂共同 original update0、seed41、fresh AdamW、batch1、各80更新，
原2021 Jan/Apr/Jul/Oct四训练病例每臂各20次曝光。

- blocked：Jan20、Apr20、Jul20、Oct20；schedule SHA
  `383f3665e2145abd6ad4ffd54bd0a8fc2e9c33568f3dbee5f7d59472c4b9f5f2`。
- interleaved：Jan/Apr/Jul/Oct循环20轮；schedule SHA
  `de1d3c5dc69099ac44fb010fb9a472d1f034fe954a56117cef41e05b2539ba33`。
- 同global80 LR2e-5、warmup10、minimum ratio0.1、weight decay1e-4、clip1，
  不在病例块边界重置LR或AdamW。逐病例LR时间位置不同，曝光列表完整保存。
- 原12步full-BPTT、K4、FP32、initial+allK deep supervision和物理权重不变；
  零权重物理步仍执行并传播最终预测。未来目标只监督，不进前向输入。
- 固定endpoint80，两臂各原四train/四2022val开发病例，共16新endpoint-case对；
  每对原train/enable_grad目标和普通eval/no_grad分别执行，无评分期backward或optimizer。
- 评分为17变量 × 6/12/24/48/72h × full/interior_1/edge_1，原train-only climatology。
  共同0的原八例合法分数按强hash复用，无新0预测、无val-oracle端点选择。
- 真实160 optimizer updates、1920训练物理transitions、384评分transitions；
  两臂模型参数均3286037、155张量，每病例原活动外锚141 active/14 unused。
- 2023 test未科学评分，确认r=0、S4未进入；没有正式候选或最终科学接受。

人工blocked是新压力条件，不是历史bulk训练的复现，更不是已证明的最坏顺序。
原bulk runner为randperm(seed+epoch)+cursor；四重复病例不等于2140窗口的完整训练制度。

## 2. 已确认读数与开发解释

预声明的interleaved开发支持为 **false**：t2m/full全部五lead对blocked及shared0均不差，
且至少一个lead对两者都严格改善；u10/v10/mslp的五lead/full也均不差；undefined不能通过。
实际20个开发比较cell只有5个满足对两项对照均不差。它们是相关的描述cell，不是独立天气样本。

### 四val病例 pooled t2m/full

| lead h | shared0 RMSE K | blocked RMSE K | interleaved RMSE K | interleaved MSE变化对0 | interleaved MSE变化对blocked | interleaved skill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 1.937445 | 1.844828 | 1.865397 | −7.2991% | +2.2424% | +0.433487 |
| 12 | 2.041813 | 1.990894 | 2.075466 | +3.3235% | +8.6763% | +0.215268 |
| 24 | 2.678474 | 2.588456 | 3.014403 | +26.6566% | +35.6191% | +0.288141 |
| 48 | 3.744069 | 3.769138 | 4.007738 | +14.5806% | +13.0614% | −0.530064 |
| 72 | 4.411828 | 4.464696 | 3.970911 | −18.9891% | −20.8964% | −0.789338 |

72h对两项对照改善，但48/72h气候态skill仍负。6–48h均不优于blocked，
12/24/48h也不优于共同0，不能只报72h赢家。

关键变量interleaved对共同0的相对MSE变化，lead按6/12/24/48/72h：

- u10：+3.1113%、+6.2150%、+17.2310%、+23.4808%、+67.1851%，全五lead退化；
  相对blocked前四lead改善但72h+17.6657%。
- v10：−5.2099%、−1.4090%、+6.1695%、−12.8621%、+35.0700%；
  相对blocked24h+1.3982%、72h+44.1567%。
- mslp：+14.5012%、+9.8120%、+11.1149%、+7.1887%、−19.4060%；
  相对blocked6/12/24h亦退化，48/72h改善。

这些是同变量同病例物理MSE比较，不平均不同单位；v10/mslp各有获益也不能掩盖u10或t2m退化。

### 每病例原normalized deep-K目标

| 原病例 | shared0 | blocked80 | interleaved80 | interleaved对0 |
| --- | ---: | ---: | ---: | ---: |
| train2021 Jan | 1.2963442802 | 1.0673048496 | 0.8009865880 | −38.2119% |
| train2021 Apr | 1.2238487005 | 0.7753455043 | 0.8190842867 | −33.0731% |
| train2021 Jul | 0.6454952955 | 0.5834330320 | 0.5211768746 | −19.2594% |
| train2021 Oct | 1.1550655365 | 1.0264889002 | 0.6500490904 | −43.7219% |
| val2022 Jan | 0.8380196095 | 1.2634007931 | 1.0581506491 | +26.2680% |
| val2022 Apr | 0.7385417223 | 1.0689493418 | 1.0226271152 | +38.4657% |
| val2022 Jul | 0.9373079538 | 1.0581070185 | 1.0083206892 | +7.5762% |
| val2022 Oct | 0.8843467832 | 0.7918585539 | 0.7172807455 | −18.8915% |

两臂四train目标均下降；interleaved有3/4 train目标优于blocked，四dev目标也都低于blocked，
但其中3/4 dev仍高于共同0。目标改善不等于五lead主目标或关键变量一致改善。
最终full全部17×5的描述cell对0：blocked train241/340改善、99变差，val132/340改善、208变差；
interleaved train302/340改善、38变差，val151/340改善、189变差。
16新pair train-final/eval-final hash完全一致、maxabs0；这只是本次运行事实，不是通用模式等价保证。
16新pair的FP64分解对native FP32最大保存绝对残差7.925557665711125e-8；
若连同8个复用shared0计24项则最大9.053387040047767e-8，不能宣逐位等价。

**已确认：**等曝光可让四train共同下降，但本次循环顺序没有满足预声明开发支持。
本实例登记negative/not-candidate并停止，不默认加seed、剂量、开关或完整val。
**未证明：**遗忘、容量、季节干扰、梯度机制或历史bulk失败原因；序列含LR位置、AdamW记忆与端点近因。

完整1530行（shared0/blocked/interleaved × train/val × 17 × 5 × 3）物理MSE/RMSE/skill/ACC：
`outputs/r7_s3_case_interleaving_analysis_20261007_attempt01/all_17_by_5_by_3.csv`，
SHA `560553478aa2d1502a46fdb820bd216f1fed753c30e4c7d0684a3e7112a6cbad`。
analysis JSON SHA `0242d9da119019dbde9395f52f0d5ae92096f8ed929b71755672d1ed6e1d5787`。
RMSE先pool MSE再开方，ACC先pool dot/energies；全cell保留undefined与坏变量，不跨物理单位平均。

## 3. 源码、数据与完整契约

| 身份 | SHA256 / commit |
| --- | --- |
| 原归档code.zip | `72953e44fa814db1e1da4d7f5c7505bcb878dc2a9765bc9f8875022a822b6ea5` |
| archive commit | `61e46bd78d4d7ce49da7a87d574e40c6c18a1c99` |
| model digest | `3ddab46b1e4c2c7e66449e39ab8247c9c7e642f45023c1c9cc14bca8f74fd217` |
| training digest | `24dde71ba380312f2c201c8f5fd2cabb9c884585bb0ce904fdee768fc5f675b9` |
| 修复薄包manifest | `05f71e52741b266fab3206ec14f6c0a4d72059c2544fe9e436ee249b14ba87ad` |
| 新external pins | `922a87d5877ec24c0e2306c19c3ade7713f6878fa62e22891e4d4309f2087ebe` |
| 薄源码ZIP | `2f029b0518c33be3e42b144cea69eea094c9ccf7748c6edc17ec09aaec738ba3` |
| original0 checkpoint | `298eccd8b8c4b9648c3214e7d860c2ea8dbd7a5dc620bc15931762c5dd42b6b0` |
| original0 state | `a23444f530ed4ff80285f51e8b4a86829b27488683a8ee6d258dec4425e99d82` |
| 原full new80 parent contract | `954cef09cae2a81b6933340dbe8b2e8a4aa987719dc1d14b9bceb6e6e59b6ab6` |
| source540856239B | `bc2ff9cfadcce604fc243bb999b3c430d5164d17fcf1de201273716a5db065f8` |
| preflight | `1b914a216bb3a79e044c316749bfad45a2a27bcd5df8bd07cd5cbf9ce4bd58e8` |
| train identity | `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac` |
| val identity | `0c34a887216b02b41716e7837f5be4d515ad2027e7b4a4728cbef15accd02d5a` |
| climatology mean identity | `c633df585b1894704d942e24305047a74ff7af2460cbc24ca2ebe7be1948a205` |
| blocked complete contract | `8fb342d0b0eadf9bb6c3ff8819ffb1d24820c9179be948b4d04167e4a5717617` |
| interleaved complete contract | `24583f157fd910504aa8e7690ac1de9b7cf0d757b104102556ad05cdf3816958` |

CPUprepare实核原source全字节、preflight、BUILD_COMPLETE、归一化/单位/窗口与普通original0身份。
new contract绑定完整80项case-ID序列、dose、recipe、wrapper、case-specific梯度外锚与
`epoch=0,cursor=updates`，故意不兼容旧singlecase/resume契约，不绕普通loader。
两臂仅导入共同0模型权重，不恢复父optimizer/RNG/cursor；fresh AdamW与seed41 native状态资格齐。
归档692 Python entries运行，不以当前checkout production模块替代。原train统计2400stamps、16桶各150。
prepare允许精确val manifest的metadata读取；训练子进程train-only dataset拒val/test manifest，
但共享Zarr cooperativeguard不是字段索引级OS隔离，范围依赖固定病例/数据调用路径。

## 4. 协议、真实进程与整轮成本

| 文件/身份 | SHA256 |
| --- | --- |
| preparation protocol canonical | `415a8f74657b2bc7b880b7662b248b46d5048e162f58cfc062d7614f348bb8f9` |
| final protocol canonical | `6efae97d11db3d4fa23e0d4be49771010804cede9ccf6909a9ed20ad5bbdf877` |
| final protocol bytes | `6e50de16e7e2aba2446ead7441bf88d84aac0d2fc876b8efc478ab6ab051c5e7` |
| prepare receipt | `228044c745169fe7948961ebf868beaca66672e7ecfafc905e2077919ee8e2ad` |
| result | `24a35359b2c8d0fa701655ae2c45c28d7a0cd2c76e470ee194d68d44845ab723` |
| reading receipt | `d42930e0689e32a3dcea83ccd8f4754a1e1e83fd150e5a774332e362a40df92a` |
| attempt | `08c5193463d68f080bac1b6c4afba03159cae8f4176f48adec92e90f995df401` |
| inventory | `21a8b0a45c604496e2977c9a9f6399f95e7c7772ba7ab11f1fdd6cf561990ae2` |
| parent supervisor receipt | `20dc8563904e2b2bd35e5dcd3aadb4f8a28750b130ad7d29379863fcd95a8d2e` |

实际五owned worker：prepare668.972305s、blocked616.803438s、interleaved586.965237s、
score565.497345s、reading14.407126s，均exit0/reaped/signals=[]。
父PID659509、各worker PID659683/682070/704632/726024/747576；没有故意留下运行子进程。
parent whole2464.990661s，含最终inventory与reap；inner result2461.939966s。
planned5400/hard14400/reserve180先冻结，外层14760/reserve90仅cleanup缓冲，soft/hard超0。
ceil2465秒/3600向上四位 **0.6848GPU-h**，非train-only耗时；新累计应为22.6281GPU-h。

startup及每spawn绑定GPU1 `GPU-9d1624af-9d77-aa7c-0620-b6cb778f4ced`，
known2491416576B与新reservedpeak取max，加2147483648B。
五spawn free为25280118784/25280118784/25280118784/14444134400/25280118784B，均满足门；
score阶段余量变化保留，不删除共驻测量。两train ownedallocated2304645120/reserved2434793472B，
score2270715392/2409627648B；均低于known，后门没有缩小known。
网络请求/新下载/派生发布0，test评分0。没有邻居信号记录；这不是全系统寿命trace证明。

## 5. 工程资格与所有失败保留

第一固定包producer完整19+15通过，独立完整19+15亦通过；但targeted15有四错误接受：
新评分读取器未拒train grad=false、eval grad=true、eval lead12h及physicalweights全1。
原资格为FAILED/noACK；第四项仅报告控制不一致，不证明真实objective已被改动。
原失败49项45pass/4fail，whole1292.752148s、soft600超692.752148/hard1800超0；
failed SHA `90c313cb58308f00dc35a707405f9afe41f9af641a7a773b696ead73e22b6889`。
修复仅read_rows与原完整test一个函数，old11/training/recipe/science不变；
新独立完整15+19+原targeted15 **49/49、0fail/error/skip**，四污染全拒，callback/断言不改。
ACK SHA `373e62545f4a476746b62a772f87976fa6c88babb926f2a7828b6f20749de25e`，
review receipt `9d272c9205c6a20f26b010ab324701e65a6fd64243b61278a90acacf72263a65`，
whole622.313355s/soft600超22.313355/hard1800超0；原失败不追认。

producer原两LR精确浮点断言失败保留，只修算术顺序/错误literal，不加容差；
原clock15:54:09.354346Z，准备whole7746.249412s/soft3600超4146.249412/hard10800超0，不reset。
主链copy precheck错误假设旧19回执字段、独立archive路径定位错误亦保留，未改旧回执/数据。
外层supervisor完整5合成CPU、导出器4、CI parser7、治理四模块213均实跑，0skip；
这些不是科学通过或独立天气证据。

保存事实helper准备首66项有1fixture alias错误，修自身deepcopy后完整66过，但最终1980.061082s
超hard1800的180.061082s，记budget_limited-finalization，不追认。
同字节另资格完整66与3实际CLI拒绝反证齐，724.334095s/hard1800超0；
两个runner import失败保留，仅修pycache异常类型/stdlib部署路径，不改helper/旧clock/断言。
完整66走exact execute，未假称旧已耗clock的self-testCLI通过。

准备/失败/资格归档分别位于outputs下，全部0GPU复制：
- failed prelaunch ZIP SHA `2fca8b4de1522a83deb4dec0a965110a0fa38143a941e028d05cbb4f9c938420`，2.141715s。
- preparation ZIP SHA `45b8fa4323b96ed3a5b644717bdf225f8160b85ae31970ac2d1ebc1c783ae6e4`，9.111919s。
- qualified prelaunch ZIP SHA `e984b56e36122aec85e645026cb9771ca4c2ab18b2e8a33766271adb6f6f303a`，2.143238s。
- helper qualification ZIP SHA `114a31530efbddd272ee527adf7fd07af37e9c5a40042a0dc95e88a78208ca88`，0.017892s。

工作分支资格提交 `a1c96392f1eaa8beeea2c6d77213febb5f2dc226`，精确主CI37664236520
completed/success、五必要steps与全部12实际steps齐；不是本页或最终登记提交CI。

## 6. 独立终态复核与登记

独立实际保存事实终态复核已完成，本页待冻结登记，工程核验不改变negative开发读数。
947个明确白名单文件首末SHA均一致；实际160训练行/160更新、新16pairs与旧8共同0复用、
384新评分calls/16320draftcells、完整17×5×3、FP32原目标累计/FP64分解/物理std²、
区域geometry/pooled ACC/null、完整外锚契约/calendar/RNG/scope及五process/gate/maxpeak/cost全部核齐。
四个新.pt仅字节hash，没有load、模型/forward/天气/source/store读取或联网；runtime真实资格由原prepare/worker承担。
独立重算20个开发cell全defined、support=false，与producer规则一致；wholecost0.6848GPU-h核齐。

已执行的terminal01因helper错误把普通eval的原生calendar dtype当train dtype而失败；
mode-specific修复只改独立helper与fixture，不改producer/输出/判据。
terminal02已核至首末hash，但发布result的重复scientific_claim keyword抛TypeError，
虽CLI误exit0/receiptcomplete，实际failure存在且无result，仍明确FAILED。
另排他publication修复将success后置、异常强false，并完整72项（原66+4 native dtype+2发布反证）
实跑0fail/error/skip；旧误绿回执与失败均保留，没有手工补原失败result。

最终独立helper manifest SHA `ff88929bbde22765c0aa3d514ae6ab7a78c7137b5ecb801700e0ffb259116da6`；
接受的terminal03位于
`/tmp/r7_s3_case_interleaving_terminal_helper_20261007_attempt01/results/publication_repair02/results/terminal03/`：
result SHA `0cca2beacc0e88cbb78b6b37f4cc874d23f44e7cdb3c458d52b47cc621c22cfd`，
receipt `44a6e844662790b38740c4158a74bc274907137057b541b4b5cc4fab13a405c2`，
digest `1698f2ede2d4fff8be6959c175c9d7ccb417ab09e70c4da1ac119b0021e1e8f6`。
整独立stage1053.426838s/soft600超453.426838/hard1800超0/reserve90，clock不reset；
三个actualhelper child全reaped/signals=[]。
总stage final receipt SHA `02a1a733fa203307c8ee423ba8c9221c4627c931073f71d0574d77b132d66b84`，
clocklast `d041af32771834d7773f5173408b9295550c0fb648a772c0254548e28bda201d`，
stagedigest `f7c1d6e81830e7c2f956c3b38287e5b7e96804360724b0e38e9cddc3a6a8d70e`。
这些回执核保存事实，不代替最终goal或科学裁定。独立helper全部阶段/失败、父监督、
分析与资格CI已排他归档 `outputs/r7_s3_case_interleaving_terminal_20261007_attempt01/terminal_evidence.zip`，
SHA `606c0b5493171e2ea3f5d9a35679f02e9375ac433f234e1a139ea7d807150397`；
259文件/85962228输入bytes、1.077344s、0GPU/网络。归档不新增天气结果。

## 7. 影响与研究边界

没有活跃模型/训练接口、旧数据/store/归档/旧证据、依赖、凭据、用户配置或安全变更。
仅TEMP调度/严格读取器、独立核验和新checkpoint/artifacts；新ordering契约故意不兼容旧singlecase/resume。
最高configuration-only，未重复训练到偶然通过，未宣逐位CUDA训练复现。
单seed、四重复train与四已曝光dev、四季30日块不是完整未见年度；没有同时置信区间。
本实例negative，不扩完整val或确认；其他实质不同主模型假设可按新协议独立探索。
不自宣SOTA、机制归因或最终goal完成，不main/merge/release/issue关闭。
