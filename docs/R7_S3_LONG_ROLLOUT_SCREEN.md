# S3 12 步长物理监督：开发主格 supported、控制守门清零，仍未超长 lead 气候态

**完整三 seed 开发筛选受支持，原零容忍守门 0/45 正格；不是科学 PASS 或最终 goal 完成。**
`scientific_claim: false`。保存的冻结决定为 `advance-to-S4-freeze`，只表示可准备独立候选冻结包，
不等于 S4 评分已开始或允许忽略新科学合同。**t2m 48h/72h 气候态 skill 仍每 seed 全负**，
完整真正未见全年/四季与同时置信区间未做，2023 test 未评分。当前 campaign 节点仍 S3。

## 1. 冻结身份与完整执行

运行根：`outputs/r7_s3_long_rollout_20261006_attempt01/`。

| 项 | 固定身份或实际值 |
| --- | --- |
| 执行代码 | `66836d29dd257a11b0c946c8af2622b35042a5ce`；工作分支 `r7/weather-reasoning` |
| 执行 SHA 精确 CPU CI | [37500335847](https://github.com/Eswink/UrbanPiDiT_R2/actions/runs/37500335847)，`completed/success`，所有必要步骤 success |
| 可行性登记提交及 CI | `e9db4fbee1460893c432a2bcacdcba3c070f8049` / [37509906302](https://github.com/Eswink/UrbanPiDiT_R2/actions/runs/37509906302)，`completed/success` |
| code.zip SHA256 | `3d8a494587f41e0b33e6f5e1cbab9de8b23ad552433282e4dd03022cf28eabb8`；ZIP comment/code_commit.txt 为执行 SHA |
| canonical screen protocol | `a1631d9b87721c2260a28c5862ef26ae7245fdd5f7c30cd404cb05708a8d10ea` |
| protocol 文件 SHA256 | `c35ea607d31d4675ba9f3a681ca94ffd2880573d369b2c0fd401a84b09ef8463` |
| result 文件 SHA256 | `a6239caa9e65320064d295bea557ad95729e098f1ff7e039033d8eb31d712ca2` |
| attempt 文件 SHA256 | `0f0bef97e9db1ebc137e20bc70c1589f44d03e61c1916646157878693d7f654e` |
| readings 文件 SHA256 | `a58505955d6e5efba85d261f5ed2daf7b944c0fd9bc9efa0fd8e1b72a3643ecd` |
| readings canonical JSON SHA256 | `aa65dac1117c0ba6f109a70cf383a825f95bbeda3918f179cefc4c47da5f09cc` |
| 完整源 SHA256 / bytes | `bc2ff9cfadcce604fc243bb999b3c430d5164d17fcf1de201273716a5db065f8` / 540,856,239 |
| train / val identities | `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac` / `0c34a887216b02b41716e7837f5be4d515ad2027e7b4a4728cbef15accd02d5a` |
| model code digest | `3ddab46b1e4c2c7e66449e39ab8247c9c7e642f45023c1c9cc14bca8f74fd217`，model 源码不改 |
| long training code digest | `24dde71ba380312f2c201c8f5fd2cabb9c884585bb0ce904fdee768fc5f675b9` |
| frozen wall budgets | soft6300 / hard12600 / per-seed3600 s，准备/归档/训练/评估/判读/清理均计整轮 |
| GPU | GPU1 `GPU-9d1624af-9d77-aa7c-0620-b6cb778f4ced`，默认共驻 |

原 probe 及归档单样本重放已独立核验、提交5abc3da并完整登记，见
`docs/R7_S3_LONG_ROLLOUT_FEASIBILITY.md`。本 screen 冻结其 receipt SHA
`3fec7af62af699934567605be426acf7d91a2ffc317e13d9d30f9791ecd6f80b`，每 worker 重核执行/数据/窗口兼容。
准备与归档成功之后三 seed 顺序执行，全部200 endpoint及15组全 cohort val完整，最后有界 reading成功。
六个直接拥有的 worker 均 exit0/reaped、signals空；无邻居信号、自动 retry 或 in-place resume。

冻结时工作树仅用户 `.zcode/config.json` 与 `.zcodeignore` 未提交；运行中只回写 goal 进度，执行代码未改。
没有下载、数据发布、model 源码、依赖或安全/凭据配置变更。主 CI 成功不代表真实天气科学通过。

## 2. 机制、训练与计算分母

- 与已终结短双步配方的实质区别：精确12步目标，直接监督48/72h生成历史；
  `L6 + .5(L12 + L24 + L48 + L72)`，其他物理步仅推进但保持图，无未来真值 forcing。
- 固定 parent：注册 v3-BD 的每 seed l6×1600 endpoint；严格导入 model state，**fresh AdamW**，不恢复父 optimizer。
- 每 seed200更新，LR2e-5、warmup10、weight decay1e-4、batch1、clip1，K4/FP32，物理步及内部步均不 detach。
  endpoint预冻结200，无 validation checkpoint选择。全部十个 checkpoint/200 loss rows保留。
- metadata：2360原始/2140完整/220精确边界排除，窗口 digest
  `dccfeef926286ddeccb7690aacd7bac45c1bb2036fe6ebbe826841b971bb39ec`。
  新窗口集合与物理目标一起变化，不能称纯单一因果对照。
- 155 registered trainable tensors入AdamW；141有活跃 moments/counters，14 alternate/diagnostic参数lazy无state，
  保持父值。独立CPU检查的活跃counter为20/100/200（seed41全部20…200），不是1600+更新。
  不把registered数量冒称逐更新梯度观测。

| seed | candidate endpoint SHA256 | training_report SHA256 |
| --- | --- | --- |
| 41 | `449bc4ce6d9c15178fecb75a954a265ffb43f7ea8cea92575d95387047ee066b` | `ca31479ba79212b8b5b21e77930a73e8ef16b17667ea719f2fee142eddc70026` |
| 42 | `841ed4669456bdbf3bfa923a83ef178da3ba3aaa77c5cd63364fc2cd4d12948d` | `e896205e4c444325cdc28680be3599e91bd764b85f16a579a12ae1abf69733ee` |
| 43 | `1e2f7494325b164af039939dad2e6e54abd0583abcda23efed400308c18e422a` | `25019f9bf2e043dc5cf9052b99789519c961a07cc7deae85516a6422d344d74a` |

可行性探针的单样本 GPU supported-aten forward/backward为393,859,201,536 FLOPs；新增200更新
估算78,771,840,307,200 FLOPs。部署配方还含parent1600 L6；以既有CPU L6计量作分母，
相对D3 400L6的描述性估算10.157565×。未另跑GPU L6同设备分母，后端计量覆盖可能不同；
不宣称精确硬件FLOP比或等算力。FLOPs遗漏elementwise/normalization等，更新数不等于计算公平。

## 3. 原 S3 合取清门：逐 seed 判读，不平均反号

判据逐字沿原冻结 t2m/full 6h/12h、每seed candidate-minus-D3 RMSE负号为supported；
关键变量u10/v10/mslp五lead全部三seed相对MSE<=0.0。科学终极门仍来自
`docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`；本次只开发点估计。

| seed | 6h ΔRMSE vs D3（K） | 12h ΔRMSE vs D3（K） | 控制相对守门正格 |
| --- | ---: | ---: | ---: |
| 41 | −0.432458 | −0.546425 | 0/15 |
| 42 | −0.763971 | −0.736492 | 0/15 |
| 43 | −0.436942 | −0.498234 | 0/15 |

两主格三seed均严格负；控制相对守门**0/45**、容忍0，全部17×5×3=**255格RMSE均低于控制**。
独立直接符号重算同保存primary/gate/decision一致。既有 helper 对abs(delta)<1e-12置0的实现事实未改；
本轮主格不接近该界，literal sign判读无差异，不把helper细节引入科学容忍。

保存决定`advance-to-S4-freeze`，**只是独立冻结包准备资格**，不是已消耗test或完成全年度确认。
参考控制相对的改善不能掩盖绝对气候态失败，也不能声称过程语义归因成立。

parent相对守门仍**16/45正格**，全部在6h/12h：seed41的v10/mslp两lead；seed42/43的u10/v10/mslp两lead。
例如seed43 mslp/12h相对父MSE+5.3612%；近lead父相对t2m在seed41/6h、seed43/6h与12h也退化。
所有parent-relative cells仍在readings/result保留，不替代冻结控制门，不称parent所有变量全面改善。

## 4. 物理单位与气候态：长 lead 仍三 seed 全负

climatology仅fit2017–2021 train，2400时次、16month/hour桶各150；2022val每lead完整472/468/460/444/428。
物理RMSE为K，skill为无量纲MSE skill，不混用归一化损失。以下是全部t2m/full，不按获胜格选择。

| seed | lead | candidate RMSE（K） | control RMSE（K） | parent RMSE（K） | candidate climatology skill |
| --- | ---: | ---: | ---: | ---: | ---: |
| 41 | 6h | 2.258056 | 2.690514 | 2.257673 | +0.587811 |
| 41 | 12h | 2.892400 | 3.438825 | 2.903726 | +0.312714 |
| 41 | 24h | 2.905004 | 3.399823 | 3.076151 | +0.285808 |
| 41 | 48h | 3.988983 | 4.911733 | 5.049674 | −0.414955 |
| 41 | 72h | 4.655095 | 5.855897 | 7.044754 | −0.967931 |
| 42 | 6h | 2.259569 | 3.023539 | 2.262796 | +0.587259 |
| 42 | 12h | 2.890136 | 3.626628 | 2.906694 | +0.313789 |
| 42 | 24h | 2.941607 | 3.663562 | 3.212256 | +0.267697 |
| 42 | 48h | 3.911406 | 5.196928 | 4.980755 | −0.360455 |
| 42 | 72h | 4.437239 | 5.767906 | 6.457677 | −0.788045 |
| 43 | 6h | 2.230434 | 2.667376 | 2.222929 | +0.597834 |
| 43 | 12h | 2.856115 | 3.354350 | 2.838502 | +0.329850 |
| 43 | 24h | 2.818701 | 3.508653 | 2.878543 | +0.327613 |
| 43 | 48h | 3.714207 | 4.823784 | 4.066026 | −0.226734 |
| 43 | 72h | 4.210250 | 5.489308 | 4.895729 | −0.609788 |

全部17变量的正climatology skill seed-cell数（每lead51格）为 **51/51/51/22/6**，
不同于“RMSE优于控制”的 **51/51/51/51/51**。72h每seed仍15/17变量负skill，只有q500/u250为正；
mslp/q850/t2m/t500/t850/u10/u500/u850/v10/v250/v500/v850/z250/z500/z850坏格全部保留。
本轮仅full，未做新interior/edge正式确认。不存在“全17变量超气候态”的结果。

## 5. 全额成本、软超与邻居边界

| worker/区间 | 秒数 |
| --- | ---: |
| prepare | 491.57634775619954 |
| archive（含身份重核） | 493.0238215373829 |
| seed41 owned worker | 2501.028194752522 |
| seed42 owned worker | 2498.7323137465864 |
| seed43 owned worker | 2512.617889424786 |
| reading | 496.9080803049728 |
| 六worker合计 | 8993.88664752245 |
| root开销/间隔 | 0.35127387195825577 |
| **整轮published elapsed** | **8994.237921394408** |
| **soft overrun** | **2694.2379213944077** |
| hard overrun | 0 |

保守整轮GPU-h **2.4983994226095576**（账本**2.4984**），不是utilization-hours或独占计时。
三seed report训练合计1117.8780767703428s，15个evaluation loop合计2495.029747732915s；
其他加载/身份核验/准备/写盘/判读时间全部计费，不猜测其每项归因。审阅确认六worker均在各自截止内reap。

owned CUDA reserved peak每seed均2,434,793,472bytes；第一次gate带probe2409627648+2GiB，
后续gate按新peak+2GiB=**4,582,277,120bytes=4370MiB**，reading亦携带峰值。每次只读UUID/free。
无邻居信号/冻结/终止，无GPU独占、租卡或收费。网络0、无新下载；当前前项14.7962加本轮=**17.2946**，
重放实际费用出口后再加，不预记0。Published计时截止于最后reap后的elapsedcapture，publication tail与
外层进程exit延迟未独立计时，不能称亚秒精度的完整外层进程利用率。

## 6. 独立审阅、工程回归与限制

审阅按seed增量隔离，不重复已有资格。以下工作态JSON均仅写/tmp，旧文件不改：

| scope | 工作路径 | SHA256 | inclusive wall |
| --- | --- | --- | ---: |
| early protocol/archive/gates | `/tmp/r7_long_screen_independent_early_audit_20261006_63fzh_gf/audit.json` | `50eddbfe5a94c1ade8b4cad2b4ad3573d8f6c32b42da67abbcd388d58454a7a3` | 419.384s |
| seed41 training | `/tmp/r7_long_screen_seed41_training_audit_20261006_7qlsf8db/audit.json` | `dda83fe67e0a2e0e5984be87323cfe0c3b307b0857c5beb55d5a362ea77ff491` | 611.304s |
| seed41 validation | `/tmp/r7_long_screen_seed41_validation_audit_20261006_kss61uc8/audit.json` | `12418f0aa4ae8b031c24270e7d31ab41c5ce066ef540f730896e236ef0ad7dfd` | 317.360s |
| seed42 training | `/tmp/r7_long_screen_seed42_training_audit_20261006_5eb_1pmf/audit.json` | `c5599478ba137ebbf89154d5e0b5c1e540dcf464b078a1a6e3d2dcb261e0e603` | 180.601s |
| seed42 validation | `/tmp/r7_long_screen_seed42_validation_audit_20261006_qdpaa3ek/audit.json` | `af64f11f45c7db99f59d86fe3604ec17d3eaf669888ce7f2edb74ea541e43b44` | 215.794s |
| seed43 training | `/tmp/r7_long_screen_seed43_training_audit_20261006_p4i50cc0/audit.json` | `4fd76dd992ff79f0dfbd3ada69d165afe70d1f09fb720713a3781a47e3559953` | 145.730s |
| seed43/三seed reading | `/tmp/r7_long_screen_three_seed_reading_audit_20261006_3gs7mzix/audit.json` | `a4eb34bb77d4bbd3dc0258b356e60aa5b72615b907b2103c43ed7db2df6b557d` | 205.954s，含在下一行 |
| terminal cost/gates | `/tmp/r7_long_screen_terminal_audit_20261006_83s5va8k/audit.json` | `f0f6f14a81db7498ee9aae1a08fc25940a9c782784140d99a558a17fc4074f2f` | 372.199s，含上一行 |

unique inclusive审阅wall **2262.372s**，不双算metric prefix；seed41训练审阅超soft600 **11.304s**，
所有hard1200内。一次audit guard/PyTorch注册碰撞的失败partial JSON
`/tmp/r7_long_screen_seed41_training_audit_20261006_3jds2l63/audit.json`保留，包含在611.304s，不归零。
CPU时间未全部汇总，不把审阅等同GPU使用。

- 新early51/training128+74+73/validation368+369+379/terminal87归组检查，无未解决失败。
  30个checkpoint全hash；seed41全部checkpoint、seed42/43的20/100/200及各parent/D3普通严格CPU复载，
  其余14个中间checkpoint仅hash覆盖，不夸大tensor审阅。600条loss/7200step loss全部核weighted FP32与LR。
- 45evaluation folders、765RMSE rows及20,448case records身份/单位/配对独立重算，最大pooled RMSE差
  7.9581e-13，skill算术差8.8818e-16；三seed主格/控制gate及parent bad格与保存readings/result一致。
- 全126terminal artifacts有hash覆盖；旧metric产物显式继承已核hash并检查stat，不重复全字段审阅。
  source/norm/window资格显式继承已登记probe独立审阅；不伪称本轮重复读取全部source/store。
- 气候态fit metadata与saved统计算术已核，场未在只读审阅再生成；derived天气/process chunks未穷尽hash。
  actual-C非Python配置为另一个具名pin，不在code.zip，不声称独立archive包含外部数据/config。
- snapshot/源码/receipt不是lifetime tracing；scope/network/峰值均不冒称系统沙箱观测。独立审阅不跑GPU/
  forward/backward/collector、天气字段或test，不授权科学goal完成。
- 130定向测试通过，完整3887passed/3skipped/6warnings；optional真实fixture跳过不作科学PASS。
  37阻断conventions、campaign/goal/index/brief均通过；治理74CPU回归另通过。依赖无变化。

## 7. 归档代码临时重放（完整，15CSV与病例精确再现）

新路径 `outputs/r7_s3_long_rollout_replay_20261006_attempt01/`，整轮soft1800/hard3600，
固定执行archive/model/source/protocol/result/seed41checkpoint/readingscanonical pins；wrapper SHA256
`495fdff90c55237131faa350f966da7dc37201b8301fc32a22ad347a85259ff1`，仅从已核code.zip导入。
离线有界CPU setup后直接拥有GPU worker，全部三seed归档collector exact JSON判读；实际只重放seed41
五lead全val，15CSV及每case MSE精确比较，任何差异须归因。

初稿128CPU synthetic/static checks并非运行接受。独立静态审阅484s发现两个执行前blocker：
保护回调拒绝helper必需的stdin/dev/null open，以及GPU worker未再重算真实long窗口metadata。
在首次执行前定向修复，只窄放trusted supervisor DEVNULL路线，worker禁止special write/descendants保持；
archived preflight窗口精确比较放在forecast前。新增36CPU反证，独立两项定向复查191s确认无剩余blocker；
600行/max函数108，不放宽规模/身份/科学门。此包装器准备/审阅无实际weather/GPU运行。

实际replay完整：两个worker均exit0/reaped/no signals；三seedcollector精确canonical摘要
`aa65dac1117c0ba6f109a70cf383a825f95bbeda3918f179cefc4c47da5f09cc`同原readings。
seed41五lead的15个RMSE/skill/ACC CSV SHA256与原值完全一致，全部病例/单位/气候态身份/MSE精确相同，
MSE difference count 0；只重放seed41forecast，不重训或给其余seed伪造重放。

| replay identity | digest |
| --- | --- |
| canonical replay protocol | `e1ca165ceda0c3dc06303bcf51cd1bc95a2f58666d2dc2f3f6b8cddb37daa789` |
| protocol 文件 SHA256 | `acd401412b974dbba656e0b429d2ebd2963a89933a73cf1a0ecdc69b8a388780` |
| replay_result.json SHA256 | `99476ac244872a58aea423e924ad3bdc668983aa72fe911695d10a1ec46accfe` |
| RESTORATION_ACCEPTED.json SHA256 | `734a6746555c15c91f93e2c21565c7393975cf2da57bc0e29d55770c450e642c` |
| replay_report.json SHA256 | `8c4326918129e486bd52159690518355dd459659ae3cefd78c005262ac63c65a` |

整轮published elapsed **1935.4622196508572s**，soft1800超 **135.46221965085715s**，hard3600未触。
本wrapper按ceil整秒保守记账1936/3600=**0.5377777777777778GPU-h**（账本**0.5378**），
不是直接1935.462219/3600，向上舍入差明确保留。prepare1.418029035s、evaluate1932.981566806s；
计时含源/身份/window复核、实际评分、清理及parent判读，不单报评估loop为费用。
本次实际峰值77,594,624bytes为推理worker，不替代训练knownpeak；所有gate仍保守用
2,434,793,472+2GiB=4,582,277,120bytes。原screen文件不改，没有实际失败replay。

本screen2.4984+replay0.5378=**3.0362GPU-h**，从已登记14.7962累加得**17.8324**。
最高config-reproducible；同机精确15CSV不证明跨平台GPU训练逐位稳定，不升级科学声明。

独立replay terminal审阅153检查，无比较失败；直接hash全部15CSV，2272case/38,624物理MSE scalar零差，
五provenance除elapsed外精确同原值，gate/process/driver/source/archive及全额账核一致。
工作JSON `/tmp/r7_long_screen_replay_terminal_audit_20261006_lxqrgdmp/audit.json`，SHA256
`086f4f3db9be5884c3630348e1f8169c6ffee99ac127683e7b2924e5fdda0b39`。
核心比较569.297s在hard600内；可选证据页数值核对收尾后capture603.402s，soft300超303.402s、
**hard超3.402s**，不是全预算合规审阅，不回改时限。所有独立审阅unique累计2865.774506s，
含此前失败partial；核心证据仍逐项可核，预算例外照实保留，不据此写科学PASS。
复核未重执行collector，replay只发布重算摘要非完整payload；三seed exact collector结论依据具名driver/
receipt和原savedreadings独立摘要，不能冒称又独立重跑。未重复source/norm/windows或GPU天气字段读取。

## 8. 已保存训练信号的零GPU诊断（描述性，不判收敛）

只读取三个已核training_report，按固定50更新块统计，不重新forward。工作JSON
`/tmp/r7_training_50update_block_summary_20261006_hb_f5lz8/summary.json`，SHA256
`f1bcde2198ebf23fb4946a06f565bcb7c51cb42db73c37db44cd9de8284cfc8e`；9检查无失败，
实测read/hash/parse/aggregation0.155297s（不含工具传输和最终JSON serialization），读14,932,972bytes。

| seed | weighted loss，块1/2/3/4 | L72，块1/2/3/4 |
| --- | --- | --- |
| 41 | .941297/.916230/.824065/.858404 | .652374/.562722/.498751/.523475 |
| 42 | .947693/.896195/.911270/.858640 | .580594/.537469/.495094/.470942 |
| 43 | .963778/.874859/.875592/.904283 | .650004/.525081/.530845/.545331 |

全部600更新的记录clip前gradient norm都>1；最后块各seed L48/L72与weighted比首块低，
L6/L12稍高，但块3→4只有seed42继续降低、41/43上升。块内训练样本不同，不能据此判定持续下降、
plateau/收敛或clipping导致优化不足；归一化train loss不充当物理val技巧。下一独立假说应先做有界
同case梯度/损失分量诊断或数据制度分析，不把此描述用作“再加算力就会赢”的证明。

## 9. 冻结包就绪度与明确未做

已满足原开发S3主格/守门，支持进一步独立研究，不替代终极门：本实例仍一个ROI、四季30日块、2022val，
**t2m48/72h仍全负、72h15/17变量负**，未完整全年、未独立季节/年度同时区间、未新interior/edge。
未评分2023test，确认r未消耗；不拿弱控制胜出当气候态胜出，不宣称收敛、过程机制因果或SOTA。
已終结短双步/旧failed与全部成本保留，不能拼seed或修改判据。

原冻决定允许准备candidate冻结包，但当前不得把它冒充正式S4确认已就绪；本轮接续仍是S3长lead差距的
独立开发与数据制度分析。普通下一方案自主取舍且必须另冻，不能无止境重复本剂量或看test调到赢。
本页只记录实证/限制，不自行宣布goal complete，未执行项如实留空。
