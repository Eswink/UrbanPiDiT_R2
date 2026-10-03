# V2 剩余路线只读 preflight 与 D2 最小机制探针

**状态：准备与工程证据，非冻结预报结果；`scientific_claim: false`。**
日期 2026-10-03；起点 HEAD `edcc33539178f95646983eca3084f7c16c241692`。
依据 `/data/esw/UrbanPiDiT_R2/docs/decisions/0032-independent-autoregressive-exposure-route.md`，
M3 辅助监督假设结束而旧 failed / paused / any-unresolved / advance=false 保持；B/C 是独立路线。
本页区分已确认和推测，不自定科学判据、不宣布阶段或 goal 完成。并发工作树正在改变，
将产生不同 model digest；**不声明当前 snapshot 稳定，也不把旧 digest 移植为当前代码身份**。

## 1. 核验范围与新增回执

本页作者仅写本页与 `/data/esw/UrbanPiDiT_R2/outputs/r7_v2_remaining_acceptance_20261003/`
内新 `exclusiveidentity_preflight*.json`；其他 outputs、代码、测试、配置、Git 均未修改。
限定核验用 `.venv/bin/python -B`、CPU4、`CUDA_VISIBLE_DEVICES=''`、双 socket 禁网；
四个 selected400 以 `torch.load(map_location='cpu', weights_only=True)` 解析。
source 只 hash bytes，不解码天气；store 只读 metadata/norm/shape，未读 state chunks；
未读封存 test manifest/天气、未直接打开 tests 源、未执行/import 归档代码或上游代码。
收尾 conventions 使用既有自动 AST 扫描；这不等于读取封存天气 test。
读取 owner stdout/receipt 不等于本页作者重跑其测试，也不是 OS/syscall 零读取证明。

| 新回执（同一 acceptance 目录） | 文件 SHA256 | 实际范围 |
| --- | --- | --- |
| `exclusiveidentity_preflight_identity_crosscheck.json` | `d99444f10bc07c9ba8b36fec0e58fc21821ef16311c7da0c62291aaf94b4b002` | 1.2831s；source/data/sidecar、四父、M3 80-member map |
| `exclusiveidentity_preflight_owner_evidence_crosscheck.json` | `39fe64abef2e4d31e6329604c37f2fd291ad5ac1ac48f6ab970e00a665d14259` | 精确 train 窗口独立重算与 D4 两份 stdout/receipt hash |
| `exclusiveidentity_preflight_identity_attempt_failure.json` | `6ec7f648d6377b011d9c5465ebcd9c2fc359bf254b5f6e93390aa5858718981c` | 首次 checker 的真实 exit1，不能算完成核验 |
| `exclusiveidentity_preflight_d2_protocol.json` | `e56f52e8845a293731c399db909aa8770c0ac0302d4e971f56824b0c26e02580` | 排他冻结 120s soft / 240s hard、完整小探针源码与 hash |
| `exclusiveidentity_preflight_d2_result.json` | `642f8c38ea6f80780f584b81e8788db0b7dda7b3ecd316f0863c9ea202d32b60` | 2.7188s、0 GPU-h、0 optimizer updates、无 overrun |

首次 verifier 错把 dataset_identity 的 stores-map key 改为绝对路径，得到错误 train
`8bfec604…` / val `45084764…` 后断言失败。原 `/data/esw/UrbanPiDiT_R2/training/r7_experiment.py`
使用 record 原串 `../cache.zarr`；只修独立 checker 后重核一致，**未改原身份比对、协议或数据**。
这不是已证实的源/父损坏，失败仍单独保留。表中回执 status 不是预报或科学接受标记。

## 2. actual 准备起点：原 source / preflight / data / sidecar 强关联

主链 identity owner 返回 pin 后，本页作者独立重算 bytes/canonical digest，而非引用聊天摘要。
source 为 `/data/esw/UrbanPiDiT_R2/outputs/r7_m2_segment/source.nc`，**37,734,176 B**，
SHA256 `496084a9260bacfaf6293a01d89439c1e49d6afa8f09bc1f51d89a1d1f9bda21`。
source receipt `/data/esw/UrbanPiDiT_R2/outputs/r7_m2_segment/source_receipt.json` SHA256
`8a681e90a91ff5099f010b3c05a55ff7724ef42fbdd38c70a26ae10c8c23af58`；`synthetic_fallback=false`。

- 原 `/data/esw/UrbanPiDiT_R2/outputs/r7_m2_segment/store/manifests/source_preflight.json`
  SHA256 `40dcec9eb8df3fe16dd9aab4c7c050642a521aa000cd1f1548394dd2900d13f8`，
  mode 是 `written-local-cache`，不误称原报告为本轮新只读 preflight。
- 主链新实际只读报告 `/data/esw/UrbanPiDiT_R2/outputs/r7_v2_remaining_acceptance_20261003/source_readonly_preflight.json`
  SHA256 `ef4ed74883b5b45694f5a7f7de92cbe47ad42e226b9b3c6546c3ae684478f1ac`；
  mode=`read-only-preflight`，`scientific_training_certified=false`。本页核其原 JSON/bytes，未重跑源解析。
- 原/新 preflight、receipt、M3 protocol 与实际 source 的 path/hash/bytes 相同；store source attrs 相同。
  shape metadata `[240,17,65,65]`；train **186** / val **22**，无新下载、发布 `--write` 或 cache 重建。
- `/data/esw/UrbanPiDiT_R2/outputs/r7_m2_segment/store/manifests/train.jsonl` SHA256
  `60e56464cbc88a90abfeae4793f97dfea4f6c786deb210db2e5919bf5732a4a2`；重算 train identity
  `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07`。
- `/data/esw/UrbanPiDiT_R2/outputs/r7_m2_segment/store/manifests/val.jsonl` SHA256
  `218512c8a1490dfc72f8c8639f2c48d3d950e0c06f53c92e6777318c015398b7`；重算 val identity
  `6ee286c7eb5c54525e2466719a75a0c58e2459d1639a087d04205992dbf3ab16`。
- identity 规范化绑定 attrs、state shape、latitude/longitude/time_ns、atmos/process mean/std，
  **不 hash 每个大气 chunk**；M3 固定 train inverse 与原有序17通道/mean/std 及父 contract 相同。
- `/data/esw/UrbanPiDiT_R2/outputs/r7_m3_scale_sidecar/scale_metadata.json` 文件 SHA256
  `d7c2837ea0083b5ce98f98cd02da729c1f12147387838dccad420393b7295cec`；重算 sidecar identity
  `4fed1c78e4c4c09a41d95457649925b02d0c8ebf89aa47c2ac8dc6496734912d`，preflight identity
  `189c33e97e24384081adf2c6aba2e293f937e669627fb9b237feaff705fe07f3`。
  train186窗口、history/target并集188帧及时间戳/样本ID/source/preflight关系独立核对。
  原 raw diagnostic hash `a3927b42b1c8daed3f3bfe02f67c503e205d19a6b6b3d16891dd9df576bd2e45`
  仅核签名元数据，**未重新解码/拟合 proxy**。
- store manifests 与 sidecar 的 `BUILD_COMPLETE.json` bytes SHA256 均为
  `a553781a5b68c3e19bb4f8921e819c856c3d7b4062cead535d637dbfe77047b9`，实际 marker 为 complete。

精确两步窗口原件 `/data/esw/UrbanPiDiT_R2/outputs/r7_v2_remaining_acceptance_20261003/exact_training_windows_preflight.json`
文件 SHA256 `ce9278e9fab328567cb881c83da1b8686cf8898e3714953e84754c8cb00de5ab`。
本页用 train manifest 精确 timestamp join 独立得到 **185/186**，window digest
`8f3f3bfd627f52fc40a3faed7505804d71ebaab294752063ae34c135eff3037f`；唯一排除
`era5z_train_2016021612_p006h` 的 t+12 为 `2016-02-17T00:00:00`，已越 train 半开边界。
此排除须在新 B protocol 中预声明并两臂共用，不可训练中默默 skip 或跨 split 补相邻帧。

## 3. 父资格与旧 archive 缺口（已确认）

首选两文件均在 `/data/esw/UrbanPiDiT_R2/outputs/r7_73_process_supervision/seed{41,42}/training/aux_off/update_0000400.pt`。
独立 CPU 解析与原 training_report.selected_checkpoint 核对均 selected400、**RW-A/K4**，
`local_solver_state=false`；不是 RW-B。四个父均131张量/2,968,259元素，schema相同不等于函数等价。

| 父 | checkpoint 文件 SHA256 | 原 contract/signature SHA256（独立重算） |
| --- | --- | --- |
| M3 aux_off seed41 | `b486b41af65cd4266b3312d6b46c65a68328aa8198b6841d162b424b1ef9eea6` | `93cdb77bdb5e671fb6443e89640c7ba1e9a4f293a85d2355d126c7755a6cdbce` |
| M3 aux_off seed42 | `70da2c5fdc62f0fdc33299b41d4367cd6d3e2c9984bc6fba423044aef4ae3471` | `8aa879ff06643ead805afde71efeff171dc17d82362e9446c7c4c7598f4c3901` |
| 旧 RW-A seed41 | `e7a9a33f9b690d07aaa6645b10e4b00e2806f6157bf1039592cf3cb347442dd2` | `dcbf0a25066822e26d79b4c20f638a6e2c6c888c6dcedceb714e2509c2b2e4a3` |
| 旧 RW-A seed42 | `c2b97f1b20427d7b69d698de3e5ee5a21bdf1a47fede61598b0ae90a56a9db74` | `dd1ffba5e098f6102fb522643106fe140b418da893786744a40d807896912e5e` |

旧两文件实际位于 `/data/esw/UrbanPiDiT_R2/outputs/r7_72_rw_b_subtraction/seed{41,42}/training/process_spacetime_rwa/update_0000400.pt`；
实际 **RW-A/K3**，虽目录叫 rw_b_subtraction，不能照旧 N3 计划误称 RW-B。
旧 protocol canonical `58fc74b7a7aaa513197d85f836684cd55851013b3c7f8519f184649837b357d4`。
指定旧 #72 输出树内未找到 `code.zip`，是**有范围的归档缺口**，不声称全磁盘不存在、不用当前源码补造它。

M3 `/data/esw/UrbanPiDiT_R2/outputs/r7_73_process_supervision/protocol.json` canonical
`404cf32b8ee8f6c3ff192d46c1de6765abe4ae3fa72967469af800a774fde15d`。
其 `code.zip` SHA256 `18595abce5acfa9e6f3252342f04eace470a06f48c3eea97c6ba463e3ea02d95`，
**80** 成员的完整 hash-map 与 protocol 逐项一致，map canonical
`e2a4e562a592e6b6f92c7d18a4d0fb236e95ebaf676464fccebe9acbe707f29b`。
从 archive model bytes 独立重算的旧 model digest 与四父相同：
`11090929930da4e1259698699cbbf12b3738cdfb2f609c3af516c24399144476`。
归档 base `d6c98cf1c33eca5885772c473805af3ef0ad62ba` 且冻结时 `working_tree_modified=true`，
故80成员字节优先于“干净commit”等同推测。只读取 zip bytes，未提取执行。
M3 aux_off 仍绑定 sidecar、固定 inverse、原 protocol，不能因三辅助权重均0而丢这些身份。

前瞻复用必须保留旧身份接受，显式逐参数 old/new mapping、新 model/训练/protocol 身份；
不能修改 `/data/esw/UrbanPiDiT_R2/training/r7_experiment.py` 的旧 digest 拒绝逻辑。
新 `/data/esw/UrbanPiDiT_R2/training/r7_parent_import.py` 的静态路径是权重复制而非 optimizer/RNG/schedule resume；
本页未执行该 importer，未声称新旧 forward 逐位等价。主链导入的实际回执应独立登记后才用于 B。

## 4. 预测状态、梯度、暴露与公平归因：可反驳线索

**已确认的实现边界**来自活跃 `/data/esw/UrbanPiDiT_R2/model/process_readout_r7.py` 与
`/data/esw/UrbanPiDiT_R2/model/process_step_r7.py`，不是照搬设计方程作“已实现”证明。
RW-A query 实际为 **LayerNorm(context)+position**，不直接含 `E(Y_k)`。
draft 经 recurrent key 更新 P 后间接改变 reader 的 K/V；forecast correction 又以当前 draft 作 base_state，
这是 solver 的直接 draft 路径。aux_off 父没有 Z/local solver；不能把前者说成严格 draft-dependent query。
仅开启 RW-B 时 solver_cell 才直接读 draft_tokens；默认关的 spatial_solver_feedback 不能代替现役反馈证据。
P（及启用时Z）在物理转移内递推，下一物理步重初始化；跨物理步写回的是 Y/history。

**已确认的训练目标差别**：历史 M3 用单物理+6h、内部K streamed-truncated，仍监督各 draft。
旧 `/data/esw/UrbanPiDiT_R2/model/r7_rollout.py` 的评估路径 `@torch.no_grad()`，不是训练展开。
新 `/data/esw/UrbanPiDiT_R2/training/r7_autoregressive_rollout.py` 静态实现使用普通 grad-enabled forward，
内部 `detach_between_steps=False`，两步间将第一 final draft 写入 history、保持两次 lead embedding=6，
`atmos_target`/`future_target` 仅监督；L6/L12 各沿用内部K归一化 deep supervision，再作 L6+0.5L12。
这与旧 direct +12/+24 curriculum 目标重分配不同；静态读代码不替代 owner 的梯度/poison/resume/精度实跑。

**推测 H-B（待 B 证伪）**：单步训练只见真实历史而自由预报消费自身状态，产生状态分布错配；
可微两步训练可能减少这一错配，也可能牺牲+6h/放大噪声，没有已有预报收益证明。
只加 L12 却 detach 物理 feedback 不再是同一梯度目标；更多K也不等于更多物理暴露。
B 的 L6 控制与两步臂必须同父/seed/窗口/全K梯度路径，否则 full-K 改变与物理暴露混杂。
同updates不是同算力；equal-compute 的400 L6对200两步只是当前goal前瞻控制，实际FLOPs/时长差照报。

**推测 H-C（必要负向对照）**：aux0 Process 与映射 matched-Generic **可能 functionally equivalent**。
新 `/data/esw/UrbanPiDiT_R2/model/recursive_weather_r7.py` 将 process_queries→latent、reasoning_cell→cell、
process_to_context→latent_to_context，复用 reader/solver；无辅助约束时“anchored”名字不自行赋予气象语义。
Process 多出的标量 diagnostic head 不作为 forecast 输入，aux0 时没有该头的 forecast-loss 梯度；
删除它不自动产生不同的天气算法。数值/全forecast梯度映射等价须实测，不能只凭同参数数量或类型名。
因此 C 的旧Ours/matched-Generic/Process三臂是**必要且可能完全负向**的归因交付，
不能预先 claim 过程有效；若两者不可区分，应终结这一语义差异假设，而非无限加模块/seed。

## 5. 官方一手机制参考：直接读原文，不采 webagent 摘要

一手固定 URL：https://github.com/google-deepmind/weathernext/tree/858301cde5de5c728f8172f782dafba1ea07ac2e 。
访问日期 **2026-10-03**；clone 路径
`/data/esw/UrbanPiDiT_R2/outputs/reference_sources/graphcast/858301cde5de5c728f8172f782dafba1ea07ac2e/`。
独立 `git rev-parse HEAD` 等固定 SHA；目录名 graphcast 不改变仓库 URL 身份。
直接阅读 `LICENSE`（Apache-2.0）与 `graphcast/autoregressive.py:223–311`。

- LICENSE SHA256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`。
- autoregressive.py SHA256 `07c51d7602679221f1d04f8aa0883e3a4f6a3abf8ee453af1baf20ecd6c5ce3c`。
- 原 clone receipt `/data/esw/UrbanPiDiT_R2/outputs/r7_v2_remaining_acceptance_20261003/reference_source_receipt.json`
  SHA256 `db19b08c92b94b01a208479bd8ee555f15511c09da80997c89c6fb982812d24c`，上述两文件本页重 hash。
- 原文单步返回 loss/diagnostics/**predictions**；next_frame 合并 predictions 与 forcings，
  `_update_inputs` 写下一 history；`hk.scan` 串联 carry，序列长>1时可用 `hk.remat(one_step_loss)`，
  per-timestep loss 最后取 mean。所读跨步边界未见 detach/stop-gradient，支持完整可微反馈的机制借鉴；
  **未执行上游，未验证其整个训练系统或外部天气效果**。remat 是重新计算，不是截断梯度。
- 借鉴仅 **prediction feedback + full-gradient rollout / 可选重算**；本仓自行用 PyTorch 实现。
  未逐行搬源码、无新增 JAX/Haiku 依赖、不搬安装/训练脚本、forcings数据/大数据/权重、
  上游 lossweights、12-step unroll、noise配方或其算力/收益结论。许可与署名原文已核，来源仍需主链台账登记。

## 6. D2 最小探针：已实跑，非天气实验

协议在任何 tensor 计算前以 `x` 写入，120s软/240s硬；结果引用 protocol 文件 SHA。
完整 **65行**小源码存协议字段并带源码 SHA256
`dd9d19c60cd315e069711320e0453d4c973c08e1c0048b3014dc5adc96b13df5`，没有新增大脚本。
双 socket 连接入口禁网；torch2.11.0+cu128 CPU4，CUDA未初始化；记录2.7188s、0updates、0GPU-h。
该时长采样在结果JSON序列化前；尾部写入/退出未单独采时，未据此宣称精确整轮成本验收。
固定合成标量 Y6=w·x+b、Y12=w·Y6+b，x=2,w=.5,b=.25，target6=1.5,target12=1.75。

| 路径 | Y6 / Y12 / 总loss | ∂L12/∂Y6 | 总loss对(w,b)梯度 | toy saved events bytes / unique storage B |
| --- | --- | --- | --- | --- |
| full | 1.25 / .875 / .4453125 | -.875 | (-2.96875,-1.8125) | 40 / 40 |
| detach反馈 | 相同 | **断开**（None，不伪写为测得0） | (-2.09375,-1.375) | 32 / 32 |
| non-reentrant remat | 相同 | -.875 | 与full相同 | 48 / 48 |

反证：相同 forward/loss **不能证明相同训练目标梯度**；detach切断跨物理步项，remat保留。
toy 重算的记录内存反而较大，**不能 claim 实际模型节省显存**；events会受alias/导数调用影响。
纯解析FP32大小：B2×17×65×65 final field574,600 B、两帧history1,149,200 B、
B2×1089×192 context1,672,704 B、B2×16×192 latent24,576 B；额外 context+final2,247,304 B。
这些只是命名tensor字节，非实际 activation lifetime、RSS、allocated/reserved峰值或训练全图成本。
真实两步 FP32/BF16/full-K 峰值、forward+backward成本、resume/finite仍须专门实跑，不能用解析下界代填。

## 7. 日历缺陷与 D4 已有实跑（不是预报收益）

修前真实 owner stdout `/tmp/r7-d4-pre-fix-esixxnwf/pytest_pre_fix_year_end.log` SHA256
`1a150b3165a878e2971614932ff164a04563940eabf0efd210a455a4aba52a92`：exit1、**2 failed/1.48s**。
它固定 edcc335 的旧 source SHA256 `252698fc28e5a66cf98414ef939788dd92a8851c4661c7cf45c50f4d7bd6a6e0`；
2023/2024年末18UTC+6h，365.25相位不精确归到新年零相位，max差
**0.004300592702981974 / 0.012901459949419653**，D2用纯解析式独立重现这两数。
本页未执行旧源码；首轮未redirect的失败只在工具历史，不伪造“原日志”。

修复 owner 独立stdout `/tmp/r7-d4-final-cpu-r37rlzsb/pytest_direct.log` 为 **33 passed/4.68s**，
SHA256 `b9a22e520212468f8e952ee6f8a4e429a9002c22e1d487dcaf3f53c09ea15d58`；
receipt_direct SHA256 `18fd6f713cce21d01f030556efab1ff2dfb6599b08b551de5a96acaa642c7a7f`。
更早联合stdout `pytest_related.log` 为 **117 passed/1 failed/8.53s**，exit1，SHA256
`31e9d8d09d762c083d5d962675ad08ecc3e609f67fa08d0720e7a0e7106c6656`，receipt SHA256
`4e81d898ad7c6bb9312244f113ceb2c0cba1bdd53667c6e9223625ee241cf8a8`；失败是旧Generic属性假设，
不能把33项成功冒充联合/全量成功。receipt内三owned源前后hash相同，不证明整个并发工作树稳定。
optional `init_calendar_year` 修复由 time owner 实施；这是已知日历接口纠正，**没有预报收益测量**。

## 8. 返回 B/C 的有限动作与 limitations

1. **B**：以两aux_off RW-A/K4优先；完成显式权重导入与新身份回执、185窗口冻结、梯度/未来poison/
   两轴/精度/resume反证后，按独立protocol跑同父两seed的L6与L6+0.5L12及可负担equal-compute。
   先实测 full-K/full-physical 内存/F+B成本；保留+6锚及所有长lead坏变量，不因负面扩unroll。
2. **C**：先验证映射forecast/梯度等价；仍执行0032的三臂三预声明seed必要负向确认，
   报全17变量×6/12/24/48/72h、全域/边界、同checkpoint K1/2/4与四成本；K1非独立训练。
   不可区分也交付，不将aux0的命名差异作过程语义证据；adaptive无独立前瞻有效前沿门则不启动。
3. 科学判据仅指 `/data/esw/UrbanPiDiT_R2/docs/R7_MAIN_MODEL_V2_DESIGN.md` §4/5/8、
   `/data/esw/UrbanPiDiT_R2/docs/decisions/0023-main-model-first-baseline-freeze.md`、
   `/data/esw/UrbanPiDiT_R2/docs/goals/n3-m4-autoregressive-rollout.md` §3、
   `/data/esw/UrbanPiDiT_R2/docs/goals/n4-m5-confirmation.md` §3及旧plan0004/0009；旧父RW-B误称不继承。
   当前路线/预算适用0030/0032和 `/data/esw/UrbanPiDiT_R2/docs/goals/v2-remaining-stages-exploration.md`，
   C primary/容忍/配对/单位/选择须新运行前冻结；本页不挑新阈值或回改旧出口。

限制与未做：这是身份/机制准备而非result-freeze；无新真实训练/预报评估、未做统计/泛化、
未重放归档、未证明导入后的历史forward等价、未读封存test、未做新全量测试/工程CI或issue关闭。
本页收尾实际执行 `tools/check_conventions.py --quiet`：37阻断规则、failing=0、exit0；
`git diff --check` exit0，报告行数实核≤250。只验证当时工程/文本状态，不证明整个并发工作树冻结。
两月冬季/少seed/旧400updates不能给科学支持；本probe仅解析反证，最高为本环境精确标量断言，
不代表GPU逐位可复现。未做完整安全扫描；无安全/凭据/config/Git/依赖/原数据变化。
主链负责 references/E/index 登记与后续实际B/C证据，不将本页当完成全部交付的凭证。
