# S3：train-only valid-time 气候态锚与异常反馈 package 筛选

2026-10-08；`scientific_claim: false`、`scientific_pass: false`。
关联 #76，未关闭 issue；科学合同只引用 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`。
**冻结版（登记用）：真实执行、描述性表与保存记录读回工程闭合均已完成并经独立复核；本页作为 index 63 负面证据页冻结。**
本轮前瞻开发比较见 `docs/goals/s3-climatology-anomaly-anchor.md` §3；不改科学接受门。

## 1. 预声明 package 与实际完成

唯一真实输出 `outputs/r7_s3_climate_anchor_20261007_attempt01/` 已完整执行旧模型资格导出、
train-only 统计/迁移、80 更新、16 个新端点—病例配对、CPU 读回五阶段。

- 输出初稿：`Y0 = C_valid + A0(history)`；RW-B 的 absolute proposal 同样从 `C_valid` 解码。
- 草稿反馈：`E(Y_k − C_valid)`。物理 `X_t`、typed physical inputs 与 absolute autoregressive
  history 不改，只有最终 `Y_K` 推动下一物理状态。
- 内部 K4 始终使用同一 C；下一 6h 物理转移才重新按有效时间查询。物理日历不依赖 PE 开关。
- 原 process/backbone、dim192/depth4/heads4/patch2/window4、两帧历史、17通道、65×65、
  K4/full12-step BPTT/FP32、initial+all-K deep supervision与全部原开关保留；aux/process_weight0。
- 物理权重 `[1,0.5,0,0.5,0,0,0,0.5,0,0,0,0.5]`；零权重步仍执行并推进物理历史。
- 单 seed41、batch1、fresh AdamW、lr2e-5/warmup10/minimum ratio0.1/weight_decay1e-4/clip1；
  global80 schedule不重置。原四个2021 Jan/Apr/Jul/Oct训练病例循环20轮，各20次曝光。
- 合法导入 original update0 的 model-only 参数，不恢复旧 optimizer/RNG/cursor。
  候选0及80各评分原四train、四2022val开发病例，共16新pairs；train/enable_grad原目标与
  eval/no_grad最终物理预测分别保存，无评分期backward/optimizer。
- oldshared0与oldinterleaved80分数按强字节pins复用，不重训、不重新预测或评分、不重复计費。
- 17变量 × 6/12/24/48/72h × full/interior_1/edge_1全表及undefined保留；未来真值仅监督不入前向。
- **2023 test未评分，确认r=0，S4未进入，无正式候选、科学接受或最终goal完成。**

同时改变输出先验、proposal先验及feedback表示，故这是性能 **package** 筛选，
不是对单一机制的因果归因，也不是精确 `X_t−C` 补偿后的坐标改名。

## 2. 保存读数与冻结开发门

以下数字可追溯至已固定真实结果和描述性导出；独立终态资格仍待闭合。

预声明必要开发支持要求候选80 pooled四dev的t2m/full全部五lead MSE不高于**两项**旧控制，
至少同一个t2m lead严格低于两者，同时u10/v10/mslp/full五lead全部不高于两者。
容忍0，undefined或非有限失败。必要小开发支持不等于正式候选或科学接受。

实际保存 `necessary_support=false`；20格全defined，7格满足对两项控制均不差且严格改善，13格未满足。

### 四开发病例 pooled t2m/full

| lead h | shared0 MSE K² | interleaved80 MSE K² | candidate80 MSE K² | candidate80 RMSE K | climate MSE K² | candidate80 skill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 3.753694 | 3.479707 | 8.201087 | 2.863754 | 6.142328 | −0.335176 |
| 12 | 4.169001 | 4.307558 | 5.079027 | 2.253670 | 5.489212 | +0.074726 |
| 24 | 7.174221 | 9.086623 | 17.952668 | 4.237059 | 12.764632 | −0.406438 |
| 48 | 14.018049 | 16.061963 | 13.708108 | 3.702446 | 10.497579 | −0.305835 |
| 72 | 19.464225 | 15.768137 | 8.769223 | 2.961287 | 8.812275 | +0.004885 |

48/72h对两项控制改善，但48h仍不如同数据train-only气候态，72h仅微正。
6/12/24h均差于两控制；不能只报长时效赢家或把12/72h小开发正skill当独立年度成功。

### 全20个开发格的对两控制均不差

lead按6/12/24/48/72h，满足标记如下（本轮严格更好标记与不差相同）：

| 变量 | 6h | 12h | 24h | 48h | 72h |
| --- | --- | --- | --- | --- | --- |
| t2m | 否 | 否 | 否 | 是 | 是 |
| u10 | 否 | 否 | 否 | 否 | 是 |
| v10 | 否 | 否 | 否 | 是 | 是 |
| mslp | 否 | 否 | 否 | 是 | 是 |

例如candidate80 u10/6h MSE2.579773，对shared0/interleaved80为1.437660/1.482389；
v10/6h为6.452708，对1.915844/1.816031；mslp/6h为299718.853917 Pa²，对22870.582208/26187.102116。
完整所有守门格数值保存，禁止跨物理单位平均。

### 每病例原 normalized deep-K native FP32总目标

| 病例 | candidate0 | candidate80 | shared0 | interleaved80 |
| --- | ---: | ---: | ---: | ---: |
| train2021 Jan | 1.979001 | 1.545540 | 1.296344 | 0.800987 |
| train2021 Apr | 1.850344 | 1.413505 | 1.223849 | 0.819084 |
| train2021 Jul | 0.982093 | 0.834224 | 0.645495 | 0.521177 |
| train2021 Oct | 1.897598 | 1.555263 | 1.155066 | 0.650049 |
| dev2022 Jan | 1.059554 | 0.942408 | 0.838020 | 1.058151 |
| dev2022 Apr | 1.826293 | 1.628640 | 0.738542 | 1.022627 |
| dev2022 Jul | 1.871785 | 1.905440 | 0.937308 | 1.008321 |
| dev2022 Oct | 1.087425 | 0.944130 | 0.884347 | 0.717281 |

四train目标相对候选0下降，但候选80四train全部高于两项旧控制；四dev有3例相对候选0下降，
并非全lead最终物理评分一致改善。新增C统计信息与改变表示本身改变candidate0预测，不能套旧0分数。
FP32 native总目标、保存的FP64分解与stdlib重构分列，不声称逐位相同。
32个case总目标的FP64-vs-nativeFP32最大差为1.4315635565864682e-7（candidate0/train Oct），
而384个单物理step的最大差9.053387040047767e-8；不能把原导出字段未注明的step最大值冒充total。
stdlib-vs-source FP64总量最大4.440892098500626e-16。原analysis_result/outputpins不回改；
另只读自身outputs补充 `objective_summary_errata01/objective_summary_errata.json`，
SHA `93db0b2c19fac4427ad69656e58f4d81a5eb4a1c4221178bff365f5d96818e53`，明确粒度与坏措辞。
该勘误的body-ready观测104.0623735357076秒，最终回执构造采样104.06480570416898秒
（六位舍入104.064806，`errata_output_pins.json` SHA
`7c21b32e10aa3ee0b8f6769f90af354f902032f836082a2cd09217d22466c760`）；
两者都早于各自最终serialization/write，**不是实测terminal whole-process结束时间**。
先冻soft300/hard900/reserve90，所存两个时间点软硬均未超，154207B自身读，
无原天气/控制/原结果重新评分或门变化；whole结束时间未观测。

独立描述性复核已对这页所引全表、20格门、32个目标、155行活动、五phase及成本算术完成43项
核对；15个明确输入30次FIRST/END实际5,384,466B与raw/stat identity相同，原clock1516.116925秒、
soft超616.116925/hard0/noreset。限定回执SHA
`a6a57535818acb35acdfe5197454a2ba9c42c5030385d871985211c29ec99bb5`，
END `b8ab1885b39f028977b4dda62890bcb65367ec81aed12f67446969eeddebbe06`。
它只核导出与文档的数值一致，**不认证原actual记录source/scope/math/process或科学接受**。
reviewer另加的未源码资格scalar ACC算法产生每path661个末位差异，被明确保留为自身harness范围/
公式错误，未证明导出缺陷、未增epsilon；provider EOF也保留。上述勘误时间措辞按该审阅明确口径修正。

按冻结停止出口，本anchor+anomaly实例停止，不追加seed/dose/开关/完整val。
**未证明：**锚无效的通用结论、遗忘、clip/容量/收敛原因或上述package内部哪项造成短时效退化。

## 3. 全表、归一化与实际活动

描述性导出根 `outputs/r7_s3_climate_anchor_analysis_20261008_attempt01/saved_pooled_export01/`：

- `pooled_eval_metrics.csv`：2040行，SHA `adfeb9bea938fc7718d61969e96bd6593704181a4faf583d5338bc9e4eaf1a35`。
- `pooled_train_mode_final_metrics.csv`：2040行，SHA `05a35a6366cabf8cb739cc25db0ad6e514396f52864a3650962098a7bf7bcbde`。
- `development_gate_all20.json`：SHA `c559afe7e6e55f3fc3d094791ac25d134f7f56262bf40f81e33be4f53ae3636e`。
- `analysis_result.json`：SHA `f3e817ae159fbea79fd3e06f70c9fbc99b81c0c2935fd3eb1630ce0a0367c331`。
- `output_pins.json`：SHA `411d698b1d23af0ecb2ed634cdb9ae8cea148ccca91a1077a0cd385059c69c06`，含全CSV/source/时长/输入首末pins。

四role（candidate0/candidate80/oldshared0/oldinterleaved80）×train/dev×17×5×3全表，
MSE先按原病例denominator pool、RMSE再开方，ACC先pool dot/energies。
MSE/RMSE/skill/forecastACC全defined，climatologyACC全2040格/每path为null并标零预测异常，未过滤。
保存的最终pooled scalar是导出权威；raw pair sufficient-stat检查用source-equivalent ordered加法，
不把Python3.12优化的scalar `sum`误作原NumPy array有序加法。没有增加epsilon或放宽身份。

原155trainable tensors/3,286,037参数保留；新真实80行mask与native endpoint observation统计
141active/14unused，active numel3,159,107/unused126,930，恰80更新/四例各20。
活动来自本次新mode观测，不复制旧控制活动外锚；同参数量/updates不等于同信息或同FLOPs。

Train-only C：原2017–2021训练membership、2400 stamps、16个月/UTC小时桶各150。
physicalFP64mean identity `c633df585b1894704d942e24305047a74ff7af2460cbc24ca2ebe7be1948a205`。
物理FP64 mean减原nativeFP32 mean、除nativeFP32 std后castFP32。
16×17×65×65×4 persistent表为4,596,800B，新增buffer而非trainable参数；
normalized表SHA `a8aeabcaa94a5392de915c55705447416e3675ce17ae47d485561cf9c7ae6f60`。
不拟合heldout，不构造365日插值或缺桶填零。

## 4. 普通身份、checkpoint与生命周期

| 身份 | digest / commit |
| --- | --- |
| experiment/new code commit | `abe5439d3444a89ee569aef3172588dc92ec9ca8` |
| new committed code.zip | `e42e9bf67df954846669491db4c8ff9733b6b28a6ef55f7da15d6d187cd0bab7` |
| new model digest | `d3fb58dbd0ed9efbd249fc258cb09c488d543c0a8ad77dac5689ac8f9ba77bab` |
| ordinary long-rollout training digest | `6e4363d5707ea4935d5449c800e4fecefa99bf553b42a747dd4dad37f8d548db` |
| repaired producer37 manifest | `9100abb886b18c3fe7ab2b54cbaeef074c862fd72ebec67f1307a64cf6c4007d` |
| external pins | `ad05dc5b0d7c95a6dfaf8245c320351712ec9a2aa483fd18415729692d14832c` |
| producer source bundle | `dd519f6c4f72bef27b94f2974c34b61464c4cbb278306ff6eee94f09248e1ecf` |
| independent prelaunch ACK | `53b60fee941ca15e18f94d1b39ac2cca9c121bd14223ce6ac82b9637654540cf` |
| independent source audit | `a3b940ec89b11dca8a433c2574b9f586782d7488302d5e19c37f75dcaa87ed6d` |
| result | `7f83e7c7d2d9144c06f56082cffd6c2018be21fed0c91202b53bc2a042d762c2` |
| attempt | `b5e7625ff6fe937c371a8cd8bcf5a18ac35c4007908ce44e34bd9c09928d9e5f` |
| outer CONTROL receipt | `0e45dac5f6c2f809090baeeb4d2beb35503f6c2ec4a17253b6ff46e89795afd3` |

旧普通loader用旧归档active code执行资格导出，逐key/shape/dtype/bytes model-only mapping，
只加table/ready buffers；不绕旧/new model digest或普通r7-local-v1 checkpoint完整contract。
constructor/forward不fit或file/store I/O；UNREADY拒绝，one-shot verified install及普通顶层
`load_state_dict`递归检查缺失/篡改/non-FP32/非有限/hash/ready。
表公共buffer仍可变，forward不整表rehash；PyTorch whole-model load非事务，错误前其他参数可能已拷贝。
这些限制不因工程测试通过而消失。

Native history_offsets为FP64 `[−6,0]`，train/eval原dataset/rollout保留；calendar各模式dtype分开核，
不能用合成FP32fixture掩盖真实ABI。旧fullOLD archive692中的209inactive legacy只读参考，
OLD active483/NEW active487/source bundle32映射不扩入legacy，不被import或计入SOURCE_FILES身份。

## 5. 五进程、默认共驻与 whole成本

原保存的direct-owned过程：

| phase | 秒 | PID | rc / reap / signals |
| --- | ---: | ---: | --- |
| old_export | 574.298967 | 1770106 | 0 / true / [] |
| new_prepare | 625.477688 | 1790835 | 0 / true / [] |
| train | 655.842428 | 1813046 | 0 / true / [] |
| score | 568.257334 | 1837223 | 0 / true / [] |
| reading | 20.871294 | 1857536 | 0 / true / [] |

父PID1770012亦rc0/reaped/signals=[]。各phaseelapsed含独立进程bootstrap，
training loop138.062802秒不能替代整轮655.842428训练phase或整个attempt费用。

Whole **2468.1777711212635秒**，含CPU旧export、新fit/mapping、train/score/read、真正source-END
bytehash、最终inventory及reap；soft5400/hard14400/reserve180先冻，overrun0。
外层hard14760/reserve90额外360秒仅收尾，不延candidatecompute。
`ceil(whole)=2469`，再向上四位 **0.6859 conservative GPU-h**，不是literalGPU kernel-time。
旧控制新增训练/评分/费用0；网络/新数据/派生发布0。
目前index62/账本22.6281不预改，待独立资格登记后新累计23.3140/remaining−3.3140仅会计。

默认GPU1 UUID `GPU-9d1624af-9d77-aa7c-0620-b6cb778f4ced`，启动和每spawn只读核
free≥max(known2491416576B,new-owned-peak)+2147483648B；启动query至Popen0.012757秒，门10秒。
没有对非本实验PID/进程组/会话发signal；saved direct handles不是OS全系统行为oracle。
源540856239B，完整identity `bc2ff9cfadcce604fc243bb999b3c430d5164d17fcf1de201273716a5db065f8`；
preflight `1b914a216bb3a79e044c316749bfad45a2a27bcd5df8bd07cd5cbf9ce4bd58e8`。
真正postrun源END来自CONTROL `data_source_end_readback.json`流式实际字节观测，不复制旧pins/runtime冒充END。

## 6. 工程验证与原失败保留

完整CPU/wheel4223cases/4214passed/9skipped，六CUDA/三可选真实fixture逐项NONPASS；
独立repaired core306/306，producer33/33，outer28tests/108counterproofs，nativeoffsethelper305，
最终preloadadapter76。这些限定工程资格不代天气/科学成功。

Code精确主CI **37695634597**、job **113046551336**、精确HEADabe5439，全部12实际steps与五必要stepssuccess；
不是本页或index最终登记commit的CI。新治理四完整CPU模块213passed/0skip、37阻断/两个campaign
与goals/indexbrief62/空白检查通过；不代postrun数值资格。

原design/source漂移、feedback-guard先后顺序红反证、B1–B6独立prelaunch BLOCK、outer/helper schema、
FP64 offsets误要求FP32、缺真正DATA END、native preflight父目录错误、postguard lazy offset/encoding导入、
provider失联超过hard未资格、主动取消重复suite、archive预检失败与reviewer自身harness失败全部保留。
只在新工程stage修已确认接口，未回改oldfailed/PASS范围、测试、真实实验、科学门或预算。

后处理描述性export也保留三次工程失败：108103887B diagnosticJSON超64MiB；
Python3.12 scalar sum与原NumPy ordered array+的1.7763568394002505e-15差异；
第三selectedsnapshot预写size拒绝（没有写超大文件）。分别闭合failed原clock，最终新排他小表导出
367.672604秒，source-equivalent ordered检查无epsilon；providerstream恢复不reset。
整个分析read ledger约1.856GB，0GPU/网络/新模型执行，不冒称terminal/source/scope独立资格。

## 7. 独立 postrun读回：尚未完成

原generic all-original-path读回真实准入BLOCK：完整1357产物（OLD692/NEW487复制源码）加inventory和
497旧validated pin原路径，合计1945distinct输入超已冻1500；209inactive复制legacy不是active map成员，
不能删清单或放宽genericprotected-path guard。observer+adapter两遍全量读取还会超2GiB。
原25observer/76adapter/305helper局部测试不追认为该实际method通过，历史contract不改。

新单一retained original snapshot bridge另工程stage冻结，维持global1500/累计2GiB/64MiBJSON。
FIRST流式bytehash同时收集一次原JSON以省重复读；真实actualinventory全字节FIRST/preauthorityEND/
postnumericEND三遍。497外部旧pins按完整原outer真实FIRST/END与此前独立identity packets源码资格归因，
**不声称450旧CPU/33原prepared路径本次又逐字观测**；active source maps与普通digests不变。
非活跃复制归档仅精确member/O_NOFOLLOW/statbytehash capability，不parse/import真实legacy。

独立scope/readback先出版，主链另外读取真实五packet字节并外部命名/pin，只经direct-owned stdin一次
≤64KiB的外部authority-file descriptor释放；不可selfauthor authority或无限轮询。
冻结math helper/纯qualifier functions用保留的原doc/hash snapshot，不重新predict/fit或读weather/pt。
实际dependencybootstrap/NumPy/native字节/源码容器与身份、directschema/ZIP/model额外reads全部须统计，
不把NumPy说成stdlib、藏依赖成本或修改原数学。具体新source/回归/独立review及actualstatplan尚待资格。

首snapshot桥固定source `a6476026efde79895102a4c328ee2411d414a74c5a25be3e29fb4df777d76b5f`
真实独立审阅为 **BLOCK，未有ACK**。实际CPython合成反证：普通absolute unplanned open被拒，
但同一路径可经`Ledger.raw`自动加入path集合及guard允许relative O_NOFOLLOW component opens而读取，
破坏本来明确冻结的controlled-input白名单。只读了reviewer自有49B canary，未读weather/受保护数据。
反证SHA `7d30e943fad7939bcca744a37d979327f6ffe9bd801743dc977a230ffbe060c4`。
独立native ABI窄probe确认47ELF/172容器Python/14原API来源及纯数值反证部分通过，
SHA `ef16212b054a9273729e0f58d94729120a44d4b99a3630fa08f2da631524b15d`，不抵消真实白名单失败。

同固定包完整回归helper305/adapter76通过，observer25有24pass/1error、新25未运行；
test-only wrapper错设ROOT使原test25拒绝exclusive child。failed receipt
`ca6949eff41ba5b2a1a2bb171481f9840da308cfdff3b7e6a0863663f211320a`，源首末同。
更早探索性full suites/loader probes缺整个engineeringstage完整actual-byte meter，工程累计≤2GiB
**未证实**，不得用将来actual预算计划追认；限制回执
`ad0e266f8f7adf0e695212fb58a738f474e424ed0b3a120c12075daf78747eca`。
publication反证还在旧adapter root留下5个空exclusive目录，未写旧source但仍属scope偏离，
全部保留未删，不声称“没有旧root写入”。固定源、这些失败与provider错误保持原样。

独立失败审阅已在880.021054秒闭合，15个held文件raw/stat FIRST/END相同，未有剩余owned子进程；
失败回执SHA `4abedcea6ced9ca9367457d856875498b74dda9622430edd720eef1e95b84d04`，
END `6bb34febcaf474a3cb6a62d8fe78d2eeaa573e925b9b26f5919531cd4fb70a2d`，
HOLD释放 `4eb0767a27d98efc9f15588e13ff56750f3ad3ebfc68980220ddbec6b836b482`；主链实际读取核对这些字节。
原工程stage已truthful failed-source-and-read-accounting-unqualified闭合，
`stage_closed.json` SHA `62971123e1abbbaa2a247222f80aad117c881e243e80ddf3cc3d984a26cdce28`；旧版本未改。

新排他 `whitelist_and_publication_repair01` 在读取/复制前冻结1800软/5400硬/reserve180，
start6209778.539712201，protocol SHA `b5206d6ad842d9bf19a854652ac5bdcc2fc1aa21b1c3b35275ad9b0beee8e409`，
从起点完整meter controlled reads，不追认原工程stage。最小修复严禁reader自动加入unassigned路径，
secureopenat仅精确冻结path/FD capability内放行，补observer test-ABI root、新authority/5packet/7role
最终字节封存与原scope-body检查，不改变原数学/测试body/科学门。
父协调器仅另叉BRIDGE路径绑定，SHA `693144463490564dd69cecadb300cf116b4b4670e3cc0618631d2460609f1316`；
原29测试全部pass/0error/skip，原parent/test保持相同，独立回执
`a6bbf8a297e5505bb9fd973da6dec817feda07d6949d7f8cc9220882d3d606c0`。
这只证明synthetic父接口，不认证新reader或真实保存事实。
新修复whole3795.088284秒、soft超1995.088284/hard0/noreset闭合；106,820,726B实际observed读与
151,122,636B显式conservative allowance分列，合计257,943,362B<2GiB，不称OS全读oracle。
固定生产freeze `38112b72b3dc49969d35d4b345d7622ae4fa0798aec8a816b5eb8881337c8b33` 原six源码首末同，
完整原helper305/adapter76/observer25通过，但首次new29仍有1项fakeFD123 overflow fixture error；
原failed receipt `b4a92f6448994e3435d4fecbbfeb4b805c5035edb688875e1261b8860bb68b76` 保留。
仅另叉真实pipeFD测试夹具，mock overflow及拒绝断言未改，纠正后的29/29pass/0skip。
composition回执 `7df819b3e00338c140480d4ca7c42b3b16e4f0eaabaf77068d5db49d00f4bb22` 明确复用
适用未变source的305/76/25与另一次corrected29，**不是一套fresh435执行**，也不追认旧failed29。
新handoff `8df3436535818c823b77c77d7350f0b2d76d1ce090c8f62005a41f876fe80077`、
SOURCEHOLD02 `acdc85acc9da802d4b7316b61dcdd6765f1f1dbff7d57f92a2b92386a710d81c` 与wholecloseout
`cfdc98aa89b4a24b585943bc26ab9a2adcdfd165037d1b2efa5d1f76547cb663` 已主链实际raw-pin核验。
该固定包独立review再以**BLOCK**闭合：source/template/plan仍指向已失败封存的旧ACK root，
`input_rows`会拒绝新review的合法ACK；不能回写旧failed root。新回执
`88c72b28cd355e9ea574ab76752ae97a3daeaa3b7863e43005de9de5c9b4633f`、
HOLD释放 `90418b5792bc8bb871ab8e3cc67ce6f5dcc3d8a6c64fd6e9ff0373736cc57573` 主链核同；
review427.376827秒/soft-hard超0/noreset，21held字节/stat首末相同，观察9,281,295B与allowance37,072B
分列；未发ACK、未做fulltoy/139或真实读回。只能新排他`ack_path_repair01`另先冻900/2700/reserve90，
只更正ACK/root/协调source literal及完整input_rows正反例，不改生产数学、FD、13-END或科学门。
该narrow repair在560.848443秒闭合，soft900/hard2700/reserve90/超0/noreset；observed17,202,843B+
显式allowance58,770,040B=75,972,883B。新productionfreeze
`f95682e58d4ff537ad543a58be7edd05775f5fbf1a391cbeeb52d19e2462c556`，
handoff `b58e7d26801af59fcc95ac4ae1e80aa63bc94d1e11d343653dd178922c5abc39`、
closeout `4706897be4e4966184f614e190241adec2afeaa06c6de04b044aedaa806e82ae` 主链实际raw核同。
input_rows合成接受新ACK并拒旧ACK/旧bootstrap且零original-content open；postguard/13绑定END与
mutation拒绝通过，corrected29/29pass，原305/76/25仅24immutable源身份重核后明确复用，非fresh435。
MAIN新parent/root与bootstrap仅另叉literal，原29+22测试完整通过/0skip，源首末同；不认证actual记录。
新固定source独立审阅4959.773620秒闭合、soft超3159.773620/hard0/noreset；
observed1,338,040,958B与conservative700,667,209B分列、合计2,038,708,167B<2GiB。
限定ACK `9dfec2378aea54b6fd78ee935abd8bdc5d32641e122705604bb97b336c52ad92`，
review receipt `a88f03f421544f65684163f93f2005eb18d44e0153017417f6a0e24d67d4249f`，
SOURCEHOLD释放 `0298a705ed9a41e1ef787dae5942000bac03e8a8c3321d0e22dc49350818f4bf`。
250source raw/stat及47native首末一致；fulltoy03 direct-owned rc0/reaped、three原snapshot hash+
13future END+未改build_roles/project_roles/verify_documents API实际合成通过；guard10/ACK3/negative118/
authority25/committed-parent18/policy12按applicability分门登记，不冒称fresh139。
原toy01/02与authority/pipe/AST自身harness错误、provider中断全部保留；toyHEAD为合成1*40，
实际abe父context另反证。**archive_check只完成原body/API/旧source-bound测试资格，非新actual调用**，
实际终态仍须不变调用它。前瞻actual1500labels/paths、knownupper2,057,082,979B不放宽。

该ACK后一次真实saved-record attempt另先冻1800/5400/reserve180，start6221024.783868065，
protocol `5ac826ec0c1d6fe8dafab2d5b0c5040cd11fe298c1e1d792cae7286dedbfea42`；
MAIN明确source/ACK/bootstrap preflight实际1,274,344B由request绑定并进累计账。
实际stat准入与FIRST/preEND、完整copiedinventory/source scope/event112、五packets MAIN外部bytepin
及bounded stdin authority握手已执行；但immutable numerical终态**未执行完**：data role pretty6,443,132B
超过冻结2MiB，publication guard拒绝。父直接owned PID2515933 exit1/reaped/signals=[]，whole32.095591秒，
bridge已计1,268,340,990B/失败JSON SHA
`4075a3b795c05daa2a4298fea95dcfddb0ea329bfbf9b7b3877f142a6a19a67e`，原attempt未续跑/重命名PASS。
verify_documents/archive_check/第三original END/全13future END未做，partial scope不算terminal资格。

另有限只读diagnostic同body canonical/字段保留证明compact仍3,205,394B>2MiB，
zlib9压缩215,095B，回执
`86bb3bb81addf364559544dbcb512643d63dac85adb5d08e751d2c0ad0876182`。
新排他lossless_role_storage_repair01先冻1200/3600/reserve120，只修7role的无损JSON wire envelope与
有界解压/字节及canonical双hash/完整类型恢复，raw per-role2MiB/decodedJSON64MiB/
global1500/2GiB和数学API/assertion/科学门不改。损坏/尾随/炸弹/错误length/hash/fields必须拒绝；
全部展开bytes进账，decoded7role aggregate先检查14MiB再出版。MAIN02新路径只防覆盖旧failed读回，
不追认第一attempt。lossless新工程已在1267.304352秒闭合，soft1200超67.304352/hard3600超0/reserve120/noreset；
观测physical+expandedJSON167,819,231B与conservative109,144,074B分列，总276,963,305B。
新sourcefreeze `2d8a8eeeff1e80fb73c59b7fc5541bc7dcbb6e42564a234c2d53c21b866141f3`，
handoff `87785be4c45fa0440a078fd28e354248e959fbee4862d06a647ddfe33b995ae5`、
close `d9af65ca7b763a816e98cbb2e7e28f6734d37ef314a6d0843073cd2e1a215640` 主链actualraw核同。
正确29+codec15共44actualpass/0skip，原305/76/25在24immutable源身份核对后明确复用/notfresh450；
两次未改verify_documents合成调用恢复role body后的canonical report相同，不称真实数值资格。
实际失败data-role metadata3,205,393B compact（无尾LF，与诊断3,205,394B口径分开）被430,637B envelope
精确恢复字段/类型/JSON字节/hash；炸弹展开bytes同样计账。新前瞻1500inputs/paths、knownupper2,057,103,838B
仍待actualstat准入。限定storage新source独立审阅待资格，**没有第二actual reader**，旧失误全保留。
协调source与首失败回执已另排他archive20files：ZIP SHA
`7cd9340f750142899a0f4f006b76f2c75f281c3431ff3b289d38bfbf2190d79a`、0.125044秒、0GPU/网络，
只工程身份与失败保全，不代terminal数值资格。
闭合修复root完整143file/link、26目录及archive metadata已归档171members，ZIP
`c5a27aeadf28361ca19b17d73f75ca2d003382af5bc4af18e387f67e9a94ced2`，3,530,383B、
17,573,494B controlled charge、229.073秒/soft600/hard1800/reserve90/nooverrun，symlink仅link-text不follow。
描述性review完整37own files及全部15exact输入快照另41members ZIP
`50b8316f7c1b777d7ced5ff02e2320527668498481ccfdacfdae8c35ec893084`，4,181,706B，
20,808,339B owned/expanded reads、349.269786秒/soft300超49.269786/hard900超0/reserve90；
唯一此前未复制的output_pins从自身mapping一次serialize仅在精确6051B/SHA匹配后接受为恢复，
没有重读original path。两ZIP及delivery receipts主链实际bytehash核同，不回开旧review clocks。
任何新actualstage须完整独立复核与真实stat准入后重新冻结单clock。

用户在侧聊明确将未来每role上限改为64MiB并保留普通JSON，已记录决策0041；该工程变更不回改
任何旧2MiB源码/协议/failed。lossless独立review因此以superseded/not-qualified闭合：原clock
524.650679秒、soft/hard超0/reset0，十held raw/stat END相同，observed656639B+
allowance16792339B=17448978B；receipt
`7c2c0dca009854a073dc2f2ffc4da4cc8341660a0bd53ce90769add47a219708`、HOLD release
`5f7b13e831c02ac28a5fecb03e8db5ac880af89d85903bf1b7259c0d405a5885`。
没有codec probes、integration/actual DATA读取或ACK，44工程测试不追认为独立资格。
四闭合roots无过滤archive另89members，ZIP
`d2ed74bb6500bb5036ddba0a78424da6bfd4c4b89fa2dce8e91616d087311543`、13,184,379B，
235.548441秒与65,985,245B controlled charge含显式allowance；只额外授权精确17B自有synthetic
opaque.pt按原byte归档/hash、不deserialize，其他checkpoint/真实天气不读。先前archive范围冲突预检失败
保持failed/noZIP；新归档成功不证明实际reader或lossless资格。

首次普通JSON64修复01因暂停/provider中断后原clock已过期，保持interrupted/expired/partial-unqualified：
原start6223570.437012013/hard1800，lateclosure3545.748463秒、hard超1745.748463，
`stage_closed.json` SHA `a4e02d201ed51c19dfabe639d41ddb9bdbaf45fc3f57ba30ef28e1ad2dbcedfc`。
已经实现plain预写检查并做有限反证/两次原helper合成报告一致及24source identity；最终coordbinding/
1500plan/sourcefreeze/当前完整29/独立ACK/实际reader未做。记录130,430,575B与晚保全1,729,138B
分列，晚身份核验不算及时资格；未创建owned Popen，不reset或复活partial01。
MAIN后来补齐新协调parent/bootstrap，只改路径/hash/stage-id literals，原完整29+22合成测试0fail/error/
skip，source raw/stat FIRSTEND同，0.388295秒/600soft/1800hard/90reserve。
新parent SHA `df836fa439f27c728ec4763fb34f4814dbc7cc9dc610e104a9bfa96fe0009775`、bootstrap
SHA `6a197b7bfc5f82dbfaf3cbf22843f7904b80ab410e9d8d154d6e053baaad2132`。
另一排他plain_role_json64_repair02先冻独立protocol/meter后复制partial工作；每role64MiB、all7
两次wire总量76MiB、未来reserve160MiB为新协议分配，不扩大global1500/2GiB。须在任何七role写入
前计齐third-original/native/archive/schema/13futureEND并准入；没有压缩、删字段或新weather执行。
plain_role_json64_repair02代理server error后也没有及时交付；原clock2627.781172秒lateclose、hard1800超
827.781172，stage_closed `3f6ee0484ecd99ec7c6fe5b82577ab6947169d14974a4c2dcd01a902ae333a70`。
最终freeze/HOLD/current29/ACK/actual未做，不追认旧partial。MAIN以新排他repair03接管：协议
`3029be34192a337bfba999905d93b742fe23bb013c663c7c9a820e9272a16689`、start6229372.377148973、
soft1800/hard7200/reserve180，约492.236870秒闭合准备、managed charge84,493,492B，
close `6f66ecb60b220ea1e92475c05b8edbf3a827b5776890c5a941b93c792c56e2e4`。
最终sourcefreeze `f307a3b5806462e1c8599e55dd845d9b3494dcd0556a887adac0cf521c0ce18a`、HOLD
`1c03f03646f347db2f0e6705b10c30920bc28af3a6feaf0c5ed192c6237809e6`、handoff
`cdaad724ec8b8d238a83ee1af5852cce39fdd838bafd7640494e284debfa40d3`。
一次完整current29/0skip与7定向plain预写反证通过；actual失败DATA6,443,132B ordinary prettywire
raw1387及完整字段/types保持相同，单文件/聚合/remaining-END费用超额在任何role syscall前拒绝。
主chain finalparent `a1efdd44b4ea2bc722d9f04f537f6236f2569943c5ea4989464e891c7e44603d`、bootstrap
`96d6235d51f9bc0ca359f209539d26f96ba1f80f840964ded3d3693951957d52`仅ROOT03另叉，原29+22合成
通过/0skip、source raw/stat FIRSTEND同；旧305/76/25明确source-bound复用而非fresh435。
原guard/native/API/math/13-END等函数适用identity核同，只有普通role预写检查/新reserve/run publication
及精确literal绑定变更；前瞻inputplan1500paths/labels、knownupper2,107,423,417B/40,060,231B余量，
仍需新的actualstat准入。独立窄风险review已调度；没有第二actual reader、额外天气运行或科学接受。
f307 plain64独立风险review已真正闭合限定source资格：ACK
`eb7f900e8ca4386d49e646d24ba205de433ca59e8ba06a8ad8b0e16a344bfdbb`，receipt
`4215a84d8b1b36ea1f9b76ae39bec25a40c2cd4d5650dcfd7d1045497e61d199`，release
`a4569396a7f42262ad3533bf83c3858986ef1180e299a8c0a8fad70ca55f82fd`。
原clock934.018155秒/soft900超34.018155/hard5400超0/reserve180/reset0；observed69,233,491B+
allowance320,870,180B=390,103,671B。十五fresh size/cost/type反证、actual DATA wire身份、七synthetic
roles原helper两次报告相同、guarded read/十三future END和mutation拒绝实际通过；68held/reference raw/stat
首末相同。独立精算1500paths/labels/2107423417B，旧305/76/25和native/parent只适用source-bound复用，
没有重复full139/stdintoy/native47或假称actual numerical/archive_check已经做完。

随后一次MAIN02 actual另冻1800/5400/reserve180、protocol
`44d844fed33875af0b019f1efa019cc5b63764fc231cc081c9512da3148500b0`，preflight1,278,425B，
只执行固定bootstrap/parent。stat/FIRST/preEND/fullinventory/scope/五外部pins/一次release及七完整plain
roles出版通过；大小120292/6443132/12625/54689/9415625/190186/7677B，未压缩删字段。
原不变verify_documents首次实际比较却拒绝`exact: saved new reading training_receipt`，childPID2766626
exit1/reaped/signals=[]，whole42.501045秒；failure
`bb3484ccd31a555b6e88ac76449718c4ea157753122683163e1a951af48e1bfd`、charge1,284,487,457B，
MAIN closure `eb1d0a4e4a3bc1f3d294b16e763f0210424181c01e8dbedb4c2d1b94ebc58578`。
verify_documents未完成，archive_check/第三original/13futureEND未做；该attempt failed不续跑或改原比较器。
新的有限read-only receipt-interface forensic已独立调度，仅精确保存JSON/源text定位typed字段差异，不读
天气/model/tensor或新评分。另完整review与failedactual65entries归档ZIP
`94408ab22c47b5f3fae19c655641c2f2721f3dc3a2f6b8e8f4f61a292be3eca4`、20,606,642B、0.359176秒，
charge119,781,086B含16MiB显式allowance，归档不将failed promotion成PASS。

独立forensic因provider stream error中途失败（root另存`stage_closed_failed.json`，observed10,572,337B+
allowance2,116,264B，hard1800超后late closure不追认）；MAIN另冻独立只读阶段protocol
`87a2a843c5b45656f385c840f4a21e1a75fdc672cd59026c04070d7eaae61c2c`（600/3600/reserve120，每JSON64MiB、
聚合256MiB），562.413秒/soft超0/hard超0，observed162,767,377B+2MiB allowance，report
`bcfdb41bd80e205cea65a65af90fb87d42f1e89da7a41243213ff225dc8433dd`、closure
`262c3389c3a61c006cd3d41cde0990ef65b8c3ab8c8bec015c86f368d01ec4bb`。
**根因已证**：原reading_receipt.json（及result.json同字段）的`training_receipt`比不变v1 helper
`training_check`返回多一个键`checkpoint_observations`（子键0/80，每端四字段
body/body_sha256/path/sha256）；其余八键全部canonical相等，递归typed diff只有这一条
`$.checkpoint_observations only_in_saved`。内嵌副本与独立文件`checkpoint_observations_0/80.json`
四字段实测全等：body与文件canonical相同、body_sha256==canonical(body)、sha256==文件真hash
（`790ff7f428dbdf28898000e8f4f984a718562646f02c52ea1c26e2361e8b44c0`/
`5da0255e0bb7595d31303c9215e3e8360897f5541148aed566e7bb3729bbd841`）、path==实际路径。
错因时间线：producer v0（05:30）read_training返回8键（正与v1 helper模型一致）；producer repair v1
（06:26）加入observation_root并把观测嵌入回执（9键）且写独立文件，producer自身test_01/test_b3 sanction
该形态；helper v1（07:24）未跟随该接口修订仍按v0 8键建模（fixture的reading即8键精简，且套件把内嵌对象
原样当training传入故自洽）。故此为**helper回执模型缺口**：非reader投影缺陷（helper按raw文件读
reading_receipt.json且要求整体canonical相等，任何改喂入body都是篡改证据）、非数据缺陷（八键全等、
内嵌观测与独立文件完全互证）。v1不变即无法通过；精确修复=新排他v2模型补全（数值math逐字节不变，把
checkpoint_observations按saved facts独立重建并强校验内嵌副本==独立文件），全套件+新增篡改反证+独立复核，
再新preparation/binding与全新MAIN03 actual。
v2已实现并冻结：新排他root `native_reading_receipt_model_repair01`（继native_offsets_repair01），
protocol `4cb8c75cc17626909163a655cd3fd4da12d07dacd4ea6fbc4be73e06349395dc`；改动仅限声明函数
（climate_facts.training_check+观测重建块、climate_terminal.verify_documents传参、climate_fixture.full_fixture
装配、climate_tests.{execute,scorecheck,traincheck}+recomputed_training、climate_cli.validate_clock/HELPER_FILES、
新增climate_reading_receipt_tests.py），独立AST复算证明272测试方法逐函数恒等、8文件字节不变、数值断言与
tolerance逐字未动。冻结helper_manifest `e6116a875e400e2e6de4121a761cc3d15cc54925888f0de8af549a100b3c46ac`
（refreeze_count=3；前两次312/312运行0fail/2error与1fail/0error分别保留在synthetic_cpu01/02）；
最终synthetic_cpu03 312/312、0fail/error/skip、suite131.95秒、receipt
`d6bb3bed9afcb00de24125b7364b0e6cfdfea116272fa7d6f51b0e147f242166`；7项新反证全PASS（含缺键必拒，
证明非整体放宽）。stage_closed `f25a882616cc616e784fc007d8e9e1af4930e3d40ee9325f78a5f5a6d29177a9`、
handoff `bcfb3d7c54a5025b52eb183ca9f159b9d69652d4f7931284ceddf153ba11b93e`，HOLD保持。
独立窄复核已 **ACK**（qualified-for-new-snapshot-preparation-only）：receipt
`3ee6c389267f3fbfed960c16e927072341d98beeaffa22eb876f6d858c64a162`、ACK
`54777a9027dafb13c1078ee5faf44209719c7115161ca0c2340e343a098161f6`、release
`14a9a1c493b3c9146bceb19d7f11628934dd8b8d55c90cf46d99d3b915557e57`、stage_closed
`1538732d8b64130d634a889429ac26fb74fd5bca7891f13d37183d9b3d5235b9`；591.66秒/noreset，
独立AST复现+9项实测复跑9/9 pass+59路径FIRSTEND同，拒绝标签全库唯一落在新比较点故判别力可信；
限制已列（HELPER_FILES常量经added_files声明、四字段仅observation_root非空时producer写入而v2恒要求=
fail-closed方向）。MAIN主链已raw核五个报告hash与ACK字段（绑e6116a87）。随后P准备链已构建：
BRIDGE_V2=`.../reading_receipt_model_v2_repair01`，freeze
`a584d394f0ab41868d88e2641feb703a3a848ceef795070d57a979afe2a8f625`，容器
`7cf2e480...`、binding `8f6dd9f6...`、新父源 `484a1b83...`、新引导 `c1fd4f7a...`；MAIN复算证明
差异恰为声明的字面量集（桥3文件4行、父2行、引导5行、模板5处纯路径、binding 15字段路径含
runtime_api_expanded_bytes 263181→264023与两成员增量精确相符），容器14成员逐字节与binding/R2冻结互证。
该P链独立复核#2已 **ACK2**（status逐字qualified-for-new-snapshot-bridge-readback-only）：
ACK2 `895b98e5b067da6df184d06adc755a58bb68852c3ae8fdd22b4349bbf7bcd7a4`、receipt
`8aedec2d…`、release `b0b29adc…`、stage_closed `2e8a22ce…`；327秒/无reset、静态全复算+桥三硬断言本地模拟。
MAIN随之执行v2引导：**预检fail-closed**——模板`parent_source_descriptor.sha256`仍是旧父源哈希（上轮只改了
path串），attempt03保留失败件（bootstrap_failure `a127adcb…`、protocol `612acea4…`、12次预检读
1,262,405B、0.025秒、error=request held repairedparent），**未删部分输出复活失败**。修正走新排他V3 fork：
freeze `4d2399f0c6cbd58b403806b94e92086800f82527c10339d4b9ac8323f2a33171`、binding `9c9b0a97…`、
模板 `23ee394e…`（含关键sha值修正+attempt04/ACK3字面量补全stage_id）、新父源 `0a223f66…`、新引导
`9e90bd23…`；MAIN复算差异恰为声明集，容器沿用V2`7cf2e480…`并与R2冻结互证；V3复核#3已 **ACK3**
（`60039a46c88b7ced14a8623a1a8bece6c1dd5b6992f85e680046b548bc5ec74e`，165秒，验证关键修复四方一致、
attempt03字节未动），V3引导成功（attempt04 预检1,273,640B、reader未启动）。随后**attempt04 实际启动即
fail-closed（保留失败）**：桥梁 `failure.json` 记录 `ValueError: GLOBAL <=1500 labels/paths`——取证证明
`snapshot_contract.reserved()` 的 `main/parent-source` 字面量仍继承 v2 父源路径（与 request/binding/
bootstrap 全指的 v3 父源构成同 label 两条不同 path），distinct paths=1501>1500；attempt02 记录恰
1500/1500（`actual_snapshot01/stat_plan.json`），离线校准计数器对 A/B 两案逐一复现 1500/1500 与
1500/1501。parent 侧 `KeyError: 'schema'` 为级联伪影（`line_read` 把子进程失败行当 READY 解析，
parent_v3.py 266→197），已写入 attempt04 `forensic_conclusion.json`。修正走新排他V4 fork：freeze
`e91d286fc9c9b42b458412ca745903dbcdccf862d18dcf4c6b5faa0fa340eb3b`、binding `212227e6…`、模板
`0a5c32f0…`、新父源 `0c2aa823…`（reserved 字面量修正为五方一致）、新引导 `cebf8306…`；差异恰为声明集
（snapshot 3行含修复、inventory 1行、父 ROOT/BRIDGE 2行+2行失败分类守卫、引导5行、模板7处、binding
1字段），容器/H2模块/上限逐字未变（**1500/2GiB/64MiB 未放宽**）；离线计数=1500/1500（within_cap）。
V4复核#4已 **ACK4**（`4817a4a9d2a6d6729d315c9991b788c8046eab1d913c76b41f56a6a47673d11b`，281.9秒，逐项PASS，
含独立复跑三用例计数与父守卫合成反证）。V4引导成功（attempt05 预检1,275,174B、reader未启动）后**attempt05
实跑 fail-closed（保留失败）**：`TypeError: training_check() missing 3 required positional arguments:
'observations', 'observation_hashes', and 'root'`——取证=v2 回执模型补全把 `climate_facts.training_check`
改为 9 参并更新了 `climate_terminal` 调用，但**第三个调用方** `qualifier_facts.build_roles`（line 433，6 参）
不在 v2 编辑面内（位于 qualifier 暂存根），312 套件亦未覆盖该运行路径。修复走新排他 Q 根
`.../qualifier_receipt_call_repair01`（manifest `5f3235a1…`）：仅该调用点 6→9 参（原始 docs 观测体+docs
哈希+root，与另两处调用同构），并附 `qualifier_receipt_call_tests.py` **全量调用签名扫描**——对前身 14 成员
集定向反证**恰好一处**失败（qualifier_facts:433 6参）、对修正集 0 失败（tests_receipt
`pass-directed-counterproof`）；Q 复核#5已 **ACK5**（`cc2511a7…`，preparation-only，159.7秒，含独立复跑扫描
逐字节同）。V5 链构建（单成员容器替换：qualifier_facts `8c1e7ce0…`；首版组装漏写 qualifier 行的 `member`
字段——由 MAIN 构建核对在**任何复核/运行前**发现，缺陷 binding `d8764789…` 归档
`/tmp/r7_v5_pre_review_defect_20261008/`，按声明修复后重建核对全绿，freeze `afb262b4…`、容器
`2231b1ee…`）。V5 复核#6 判 **BLOCK（F-RV6-01，保留）**：三个运行时 ACK 引用被错钉在 ACK5（preparation-only
且缺 `source_freeze_sha256`），按 V4 先例应为**本链自身评审根**的 ACK；两种传参模式经复核员实测均
fail-closed。修复走新排他 V6 fork：三处引用全部改指 RV7（本链前瞻评审根）、attempt07/v6 字面量重算、
容器沿用 v5 未变；freeze `8b7565ed…`、binding `8ecf0a53…`、父源 `dcefdb77…`、引导 `945b784b…`；MAIN 核对
chain-ACK 三方一致、binding 仅父源两叶变化、计数 1500/1500；V6 复核#7运行中。这些都不是实际读回或科学资格。
**终态（2026-10-08）：保存记录读回工程闭合，负面判定不变。** V6 链（freeze
`8b7565ed845c30012222c3af902a8cf4437cac87c815f430769904346052a404`，三处运行时 ACK 引用改指本链自身评审根
RV7）经独立复核#7 **ACK7**（`1969a5128790c7118a399eab2c31d148190da80b7f1f1e125ddaabae700b66f3`，7/7 PASS，
含合约内存重放：RV7 行接受、旧根行拒绝；bootstrap 三硬断言以真实 ACK 重放通过）后，引导创建 attempt07（新时钟
1800/5400/180，预检 1,274,324B），父源实际读回 **exit 0**（whole 40.59019962884486s、无信号无错误）：准入恰
**1500/1500 labels/paths**、总读 **1,946,575,475B**（<2GiB）、`receipt.json`
`dd6204554b3057328545ab84efff09359a464ef5e4f17a7ebd245fdf3a31371f`（`complete-new-snapshot-bridge-engineering-only`）、
`completion.json` `c06cfe68af17b1f9098a38996b41c23dc094230f9f126cddc43d1759b9c3237d`、五个 packet 与七个 role
齐备、**无 failure.json**；`three_original_hash_passes=True`、`source_FIRST_END_equal=True`、
`future_all13_FIRST_END_equal=True`、clock_resets=0、soft_overrun=0.0；独立重算报告：实际保存训练行 80、7 个独立
role、16 新 pair、384 评分调用/16320 draft cells、checkpoint_loads=0、additional_model_executions=0、
audit_fit_calls=0、tensor_deserializations=0、weather_reads=0、network_requests=0；v2 回执模型（含 qualifier 9 参
调用路径）实跑验证通过。读回本身**不是**科学通过：candidate80 开发门 necessary_support=false（t2m 6/12/24h
劣于两控制，仅 48/72h 及部分门格改善），本实例负面判定与既有负面读数不变。证据归档
`outputs/r7_s3_climate_anchor_readback_20261008_attempt01/`（57 文件/22,298,486B，manifest
`4f0148dfebe7c818bf92d4abe6e079bdb457d225cda889a69d637bdd7748830a`，含 attempt07、桥读回、V6/Q/RV5–RV7 复核件
与 attempt03/04/05+被阻 v5 链失败件）。保留失败链：attempt03（预检）、attempt04（GLOBAL 1500 越界）、attempt05
（qualifier 六参调用）、v5 链（RV6 BLOCK F-RV6-01）；上限 1500/2GiB/64MiB 全程未放宽。本页随登记提交冻结为
index 63（negative/not-candidate，成本 0.6859 GPU-h 保守计一次）。

## 8. 影响、可复现等级与出口

活跃实验实现显式default-off锚/异常接口及persistentbuffer；后处理不再改活跃model/training、旧data/store/
legacy/旧outputs、科学合同、用户配置/security/credentials或依赖安装。新modecheckpoint必须匹配新model
身份与完整table/contract，普通load强核不变；default-off接口兼容已工程资格。

最高configuration-only；未重复训练到偶然逐位一致。单seed、四重复train、四已曝光dev及四季30日片段
不等于真正完整未见年度/四季/3seed确认；没有同时置信区间、独立天气样本或单机制因果证据。
negative终结本实例，主研究方向仍S3，不main/merge/release/issue关闭，不自宣SOTA或最终goal完成。

下一具体动作：等待上述v2独立窄复核闭合（ACK或BLOCK），通过后以v2重建preparation/binding，另冻新时钟
执行全新MAIN03 actual（完整1500/global2GiB/64MiB准入、verify_documents(v2数值不变)/archive_check/
第三original/native/13futureEND），MAIN01/MAIN02维持failed不追认。之后冻结本页、append record63、
同步brief/账本/goal并核工作分支精确登记CI；本工作态不预写资格或完成。
另备100行launch wrapper未执行、未qualified，不加入恰1500路径runtime读集，也不因“协调”名义豁免范围。

官方issue正文/评论经web-researcher检索后由MAIN一手匿名HTTP保存（访问2026-10-08）：
[#76](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/issues/76)、
[#77](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/issues/77)、
[#78](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/issues/78)、
[#79](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/issues/79)均open，评论数1/0/1/0，最新更新时间
仍2026-10-04；建议不当实验事实、issue并未关闭。原文和URLs/访问时间保存于
`outputs/r7_s3_climate_anchor_issue_refresh_20261008_attempt01/`，receipt
`33d64ffc04165214eecbd0a730899c8f5037c97bec059d7cb458fc2f8406fc3c`，48,695B响应body、9.536403秒，
soft300/hard900/reserve90超0、0GPU；不是训练数据下载或科学门变化。

闭合plain01/02/03源码、协调与superseded review已另124entries完整归档：
`outputs/r7_s3_climate_anchor_engineering_20261007_attempt01/plain64_source_and_failures01/closed_plain64_source_and_failures.zip`，
SHA `e483540107ce4019a85ce63e2b0814b95d34f5abc9aaed97c80d1dcec84d28c0`，6,467,209B，
0.145519秒/soft600/hard1800/reserve90超0；observed32,271,750B+显式allowance16,777,216B=
49,048,966B，不deserialize自有synthetic opaque.pt、symlink仅linktext。归档身份不代独立source或actual资格。
