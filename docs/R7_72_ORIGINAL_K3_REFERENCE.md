# #72 真正原 K3 参照：实际执行与独立元数据补全证据

日期 2026-10-03；mainmodelV2 文档子工作；`scientific_claim: false`。

> **接受范围仅为 `accepted-reference-evidence-metadata-complement-only`。** 真正 #72
> selected400 / RW-A / K3 的十个验证已实际完成，独立来源、完整库存和数值证据已闭合；
> 不是科学天气收益、原历史全源码逐位重放、Gregorian 配对或 goal 最终完成。
> 原 `attempt.json` 保持 `success / finalized:true`，原 `reference_result.json` 保持
> **`complete-not-accepted`**。新的独立接受在外部回执，不倒改原结果，不伪造原 frozen manifest。
> 委派时新工程 CI 为 `in_progress`；冻结前已确认该精确 CI **失败**，详见下方登记补记，**不写 PASS**。

## 1. 实际范围与原 K3 身份

实际产物根为
`/data/esw/UrbanPiDiT_R2/outputs/r7_original_k3_reference_20261003_attempt01`。

- 2 seed（41 / 42）× 5 lead（6 / 12 / 24 / 48 / 72 h）= **10 evaluations**；
  **0 train / 0 optimizer updates**。每 seed 88 case evaluations，总 **176**。
- 真正父权重来自 `/data/esw/UrbanPiDiT_R2/outputs/r7_72_rw_b_subtraction/seed41/training/process_spacetime_rwa/update_0000400.pt`
  与 `/data/esw/UrbanPiDiT_R2/outputs/r7_72_rw_b_subtraction/seed42/training/process_spacetime_rwa/update_0000400.pt`。
  两父的原 reasoning_steps=3、selected_update=400；不是把 M3 K4 权重改名 K3，未替换 checkpoint。
- archived unchanged strict loader 由实际 evaluator 执行，检查 **131 state keys / 2,968,259 entries**。
  独立审计及本页编写者均不解码张量、不独立重数参数；这些检查只绑定冻结 evaluator 与 opaque
  父身份，不能写成独立 tensor recount。
- 验证保留 **17 variables × 5 leads × 2 seeds × 3 regions × 3 forecasts = 1530 metric rows**。
  forecasts 是 original_k3、persistence、train-only climatology；模型 full 区域 **170 cells**。
- 本子工作只创建本页及独立 `/tmp/` 文档回执；不写原产物、其他 docs/source/tests/index/HEAD。
  只读指定元数据/回执/指标 CSV，未 rerun GPU、训练、验证或整套审计，未读天气/test payload，
  未解码 checkpoint，未重复旧 B 的 361 文件 hash。

## 2. 冻结协议、真实来源与 HEAD 分层

原 actual protocol 在执行前冻结：planned **1800** / hard **3600** 秒，cleanup reserve 10 秒，
最早同 boot CLI 入口计时，不在 CUDA phase 或重新加载 run 时重置。

| 身份项 | SHA256 / digest |
| --- | --- |
| actual canonical protocol | `8c42e5031e88c04f4c9514236d31bb4bc89da5b1fb96aa86bb9700ab63d8bdbc` |
| actual protocol 文件字节 | `c0e57aa1842a4c9459c00a7c606797d9caccc0e4223546ad020c29160e4b6fdf` |
| attempt.json | `05ce01a29ea875355cacaee9b97a1bf32ab34626774e18a410fc0e35be69498f` |
| reference_result.json（complete-not-accepted 保留） | `ca2c7f0828b6b90e3a970e44ae14f2bdb970a5565d849ff97eb84a5946554b03` |
| metrics.csv（1530 rows） | `965ce69e30728be58d51bdbb11743acf3fad8b0e5e72987c484f1727a14f4591` |
| source_identity.json | `9874210ef6259de7196dc605219344ecc6f2c12126633670e535702b690bda36` |
| compatible archived sourcebridge code.zip | `18595abce5acfa9e6f3252342f04eace470a06f48c3eea97c6ba463e3ea02d95` |
| sourcebridge source tree | `e2a4e562a592e6b6f92c7d18a4d0fb236e95ebaf676464fccebe9acbe707f29b` |
| archived model code（28 Python 成员） | `11090929930da4e1259698699cbbf12b3738cdfb2f609c3af516c24399144476` |
| 七 companion runtime 文件集合 | `9596dde84055329e1a815ef0244f4678d1329268bdc92822800f3486df1ae76f` |
| companion_code.zip | `f7861557055e2047cf101fd00f477080e1d40f0ac13eb904a37c80e373211a16` |
| seed41 真正父 checkpoint（仅 opaque 身份） | `e7a9a33f9b690d07aaa6645b10e4b00e2806f6157bf1039592cf3cb347442dd2` |
| seed42 真正父 checkpoint（仅 opaque 身份） | `c2b97f1b20427d7b69d698de3e5ee5a21bdf1a47fede61598b0ae90a56a9db74` |
| seed41 原 training_report.json | `acaee1d7eee4e0cae76f920f782bb83483e1c85b34c2e63ce559f50efe63e051` |
| seed42 原 training_report.json | `53ab535c9d32726c941f2ebb13b67f43df224e7332d50601f67aab4460ba154c` |
| 父 canonical protocol（两 seed 同） | `58fc74b7a7aaa513197d85f836684cd55851013b3c7f8519f184649837b357d4` |
| 父 protocol 文件字节 | `0bdab50ccfed2279f5c8fe2d188b6896d1a3af1272c0c518462dfc6003b2e134` |

sourcebridge 归档路径为 `/data/esw/UrbanPiDiT_R2/outputs/r7_73_process_supervision/code.zip`，
**80 archive members / 28 model Python members**，仅作本次已资格核对的兼容源码桥。
#72 没有记录完整历史 code.zip/commit，不能将这个桥称为 #72 原历史完整源码、原历史 score
轨迹逐位复现或伪造 historical commit。归档 model digest 与真正父一致，strict loader 没有放宽。

`source_identity` 明确 `source_base_commit = 616b029dce569b92bd08295737981512180a1ad3`、
`source_commit:null`：base 加精确 companion bytes，不虚构一个历史/新 commit。
实际启动的 before/after HEAD 为 **`1615795d70117af608787ad6de15d2e5640b901c`**，
launcher 记录 unchanged；七 runtime pins 相同，archived runtime 独立隔离，不靠当前 HEAD import。

委派时新的工程 HEAD 为 **`523c819b8d42a3f43163eb53509f76136d8038f4`**，精确 CI
[37155293949](https://github.com/Eswink/UrbanPiDiT_R2/actions/runs/37155293949) 为 **in_progress / pending**。
该状态由协调者提供；本子工作未查询 Git/API 或读 CI 日志，**没有接受该新 CI**。
不能把当前工程 SHA/未完成 CI 追认成 actual K3 的启动身份或已通过凭据。

## 3. 数据与计算语义：不作 Gregorian 配对

| 数据身份项 | SHA256 / identity |
| --- | --- |
| M2 source.nc（只读元数据披露，不读字段） | `496084a9260bacfaf6293a01d89439c1e49d6afa8f09bc1f51d89a1d1f9bda21` |
| source_receipt.json | `8a681e90a91ff5099f010b3c05a55ff7724ef42fbdd38c70a26ae10c8c23af58` |
| source_preflight.json | `40dcec9eb8df3fe16dd9aab4c7c050642a521aa000cd1f1548394dd2900d13f8` |
| BUILD_COMPLETE.json | `a553781a5b68c3e19bb4f8921e819c856c3d7b4062cead535d637dbfe77047b9` |
| train.jsonl | `60e56464cbc88a90abfeae4793f97dfea4f6c786deb210db2e5919bf5732a4a2` |
| val.jsonl | `218512c8a1490dfc72f8c8639f2c48d3d950e0c06f53c92e6777318c015398b7` |
| train/data identity | `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07` |
| val data identity | `6ee286c7eb5c54525e2466719a75a0c58e2459d1639a087d04205992dbf3ab16` |
| store metadata/stat snapshot inventory | `b78a1514017846ca338f99c671813c85bd711ce8a3e5d19aa87468917e23c238` |

数据路径为 `/data/esw/UrbanPiDiT_R2/outputs/r7_m2_segment/source.nc` 与
`/data/esw/UrbanPiDiT_R2/outputs/r7_m2_segment/store/cache.zarr`；原 preflight/BUILD_COMPLETE/
train/val 字节身份已在 actual protocol 及独立 before/after 中核，不重新发布数据或碰封存 test。
store guard 对 metadata/坐标/normalization 的 pins 与 state/process chunk 的 stat 库存核不变，
不是读取 chunk 天气或证明全 weather payload 独立重建。

- 原 rollout 语义保持 **accumulated lead**、固定初始化 day/hour、init_year 仅元数据与
  **365.25 feature denominator**；未改成 B/C 的 Gregorian calendar / 每物理步固定 +6 h 查询语义。
  本结果不是 calendar-matched、architecture-matched 的 B/C 同构配对，不报告跨语义收益 Δ。
- lead 6 / 12 / 24 / 48 / 72 h 每 seed 实际 val cohorts 为 **22 / 21 / 19 / 15 / 11**。
  max_samples=32 不截断这些 cohort；短 lead 未缩成 72 h 的 11 病例。
- 原区域标签 **full / interior_2 / edge_2** 原样保留，margin=2；`interior_2` 不静默改名为 C 的
  `interior`。full 与子区域暴露重叠，不当独立天气事件。
- persistence 与 climatology 在同 seed/lead/case/variable/region、同 area weights 上评分。
  baseline 的 K 为 null，model_depth_applicable:false，training updates/parameters/trainable=0；
  无额外模型 forward 或独立 GPU job，但评分时间包含在已有验证 worker 成本，**不是零 runtime**。
- 独立重构使用归档 normalized D/P/T sufficient statistics × 冻结 train std²；按精确初始化数
  pooled sums 后才开方/除法。RMSE 不是平均 case RMSE，ACC 不是平均 case ACC；MSE skill 为
  `1 - MSE/MSE_climatology`。原容差 relative `1e-9`、scaled absolute
  `1e-12*max(1,abs(a),abs(b))` 不放宽，不作跨物理单位 RMSE 均值或 p-values。

## 4. 原样保留的数值、坏值与完整 full 参照

以下是已独立核元数据参照，不是科学 positive。完整 1530 rows 在 metrics.csv；
full_model_170 与其摘要在独立 audit 中，未只选择 t2m 或有利 lead。

| forecast（各 510 region rows） | 负 MSE skill | 负 pooled ACC | undefined skill | undefined ACC |
| --- | ---: | ---: | ---: | ---: |
| 真正 original_k3 | 262 | 100 | 0 | 0 |
| persistence | 280 | 68 | 0 | 0 |
| climatology | 54 | 0 | 0 | 510 |

模型负 skill 范围 **-5.81668558486345 至 -0.0024973669329437787**；persistence 为
**-3.4223091494296494 至 -0.0027814434222190876**，均保留不删。
climatology 的 54 个微小负 skill 均为 **-2.220446049250313e-16**：保留序列化实际值，
披露 binary64 舍入幅度，不伪称新科学劣化、改成零、剔除或放宽冻结容差。
510 climatology ACC undefined 全保留，不用 0 代替。

| per-case/variable/region 暴露 | rows | 负 MSE skill | 负 anomaly dot | undefined ACC |
| --- | ---: | ---: | ---: | ---: |
| original_k3 | 8976 | 3956 | 1811 | 0 |
| persistence | 8976 | 4594 | 1824 | 0 |
| climatology | 8976 | 2308 | 0 | 8976 |

8976 = 176 × 17 × 3，为 overlapping region/seed/variable 暴露，不是 8976 独立天气事件。
climatology case 级舍入负号也不删除/拿作模型科学 gain。

### t2m / full 的精确单 seed 参照（非跨 calendar 配对）

| seed | lead h | RMSE K | MSE skill | pooled ACC | n initializations |
| --- | ---: | ---: | ---: | ---: | ---: |
| 41 | 6 | 2.733911198574701 | -0.006685396652259312 | 0.5222211555758938 | 22 |
| 42 | 6 | 2.5281422294632243 | 0.1391490425408256 | 0.5879544816738296 | 22 |
| 41 | 12 | 3.707138650134707 | -0.8518204572437016 | 0.22745298994350424 | 21 |
| 42 | 12 | 3.5248593240646273 | -0.6741902064861478 | 0.2935032862958127 | 21 |

### 全 170 full-model RMSE cells：seed41

| variable | 物理单位 | 6 h | 12 h | 24 h | 48 h | 72 h |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| t2m | K | 2.733911198574701 | 3.707138650134707 | 4.411621691314006 | 4.36525486671083 | 4.490057954953439 |
| u10 | m s**-1 | 1.6856860880621105 | 2.1942395831243444 | 2.633258491030036 | 2.6881235622838933 | 2.180212396448863 |
| v10 | m s**-1 | 1.752057655940522 | 2.2802715298904 | 2.6327691740082697 | 3.0882135421900214 | 3.206495274355627 |
| mslp | Pa | 191.84623461138298 | 320.24648386792944 | 528.4775643866992 | 891.302417419755 | 1085.388257541861 |
| z850 | m**2 s**-2 | 120.76921664432136 | 224.36888508975198 | 388.77407622676924 | 574.7264688083199 | 640.6672796185051 |
| t850 | K | 1.6807306528113795 | 2.3231150088500634 | 2.844212587313946 | 3.8357831802962674 | 4.440113905214403 |
| q850 | kg kg**-1 | 0.0006040749529383815 | 0.0008337263375501427 | 0.0010393579037809519 | 0.001748726462376675 | 0.0023063424500775987 |
| u850 | m s**-1 | 2.8674729166177166 | 3.7388300754299824 | 4.2557990884544195 | 4.507086541012526 | 4.764262496093934 |
| v850 | m s**-1 | 3.209280696926816 | 4.844329672564316 | 6.510159418222112 | 8.494037261101617 | 6.5466862685830804 |
| z500 | m**2 s**-2 | 151.27929982683455 | 276.67934877891577 | 434.71332222201323 | 582.725847999781 | 692.4555619489445 |
| t500 | K | 1.7145199037129204 | 2.7381403095869725 | 4.08491438737497 | 5.171326415848209 | 4.115358115034676 |
| q500 | kg kg**-1 | 0.00034836522855385125 | 0.0004459224936632192 | 0.0005782806929627563 | 0.0007925422584790724 | 0.0008187271196329589 |
| u500 | m s**-1 | 5.248410848021846 | 7.305094679113306 | 9.849494118462227 | 11.843232609301543 | 11.671957695657644 |
| v500 | m s**-1 | 5.77287984720112 | 7.273459567704756 | 9.556136378578179 | 11.61716636624568 | 11.252139010304402 |
| z250 | m**2 s**-2 | 271.01417562505566 | 519.4008508013984 | 915.3543489123172 | 1311.714415918078 | 1497.6700245131606 |
| u250 | m s**-1 | 6.55019940444939 | 10.057526059036576 | 13.954752081634801 | 17.606024171859385 | 17.46954393376604 |
| v250 | m s**-1 | 6.330172571981159 | 9.165372658657029 | 13.249824446342583 | 19.08204078998483 | 22.356335083443756 |

### 全 170 full-model RMSE cells：seed42

| variable | 物理单位 | 6 h | 12 h | 24 h | 48 h | 72 h |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| t2m | K | 2.5281422294632243 | 3.5248593240646273 | 4.551936780801498 | 5.024528121772664 | 6.103416466359008 |
| u10 | m s**-1 | 1.6796280223871676 | 2.2458346510900298 | 2.8017517659231865 | 2.8919579617299953 | 2.368970428672503 |
| v10 | m s**-1 | 1.76099775897993 | 2.3736123249178007 | 2.9802775019492342 | 3.4399558493155356 | 3.293126927715661 |
| mslp | Pa | 186.86462731927304 | 323.0663767513897 | 542.9025456480955 | 850.1525019533328 | 1005.3544980815452 |
| z850 | m**2 s**-2 | 117.2459462544653 | 214.89674436546457 | 363.11677920553427 | 506.707012907629 | 569.7315397233886 |
| t850 | K | 1.6186581369696083 | 2.308076131249295 | 3.1565802248418713 | 5.001227040140469 | 5.799529189514883 |
| q850 | kg kg**-1 | 0.000601140889358702 | 0.0008397135712032429 | 0.0011571642051991166 | 0.002022001589970128 | 0.0028031410427865906 |
| u850 | m s**-1 | 2.8960575717640933 | 3.8460430244774773 | 4.440284075534302 | 4.966511860723858 | 5.460702829508949 |
| v850 | m s**-1 | 3.259418068162763 | 5.01272357826541 | 7.119485785621016 | 9.385009797729492 | 7.125785738617169 |
| z500 | m**2 s**-2 | 148.4003324394866 | 272.72329375125486 | 450.6901390730516 | 719.2327551986587 | 810.3910630024299 |
| t500 | K | 1.7405463003976516 | 2.83054665126593 | 4.473252897677044 | 6.234392975710701 | 5.162803496037869 |
| q500 | kg kg**-1 | 0.0003511491958137664 | 0.00046169551607033347 | 0.0006072395259045557 | 0.000803340863184249 | 0.0007593390633444856 |
| u500 | m s**-1 | 5.222255321062432 | 7.31593423342848 | 10.440079676003714 | 13.644530899626718 | 14.34106278480549 |
| v500 | m s**-1 | 5.820885980602581 | 7.460190069971504 | 9.524540465939038 | 12.153298821990344 | 12.067787761320655 |
| z250 | m**2 s**-2 | 264.2115679224383 | 503.0122507711547 | 906.4566420314393 | 1439.4164421971986 | 1391.7278678730138 |
| u250 | m s**-1 | 6.646005386907366 | 10.177937776206242 | 14.354848744854358 | 17.523496035885362 | 16.42317974954836 |
| v250 | m s**-1 | 6.43417688404354 | 9.555732240667176 | 13.455227981596531 | 21.41691854320371 | 25.19736732527839 |

full-model cells 每 lead 34（17 × 2 seed）；6/12/24/48/72 h 负 skill cells 为 **1/6/19/34/30**，
负 pooled ACC 为 **0/0/0/18/16**。所有负值、undefined 和短长 lead 均留存；不从表中挑出新的
科学端点，也不拿 K3/K4 或不同 calendar 的数值差作为严格配对收益。

## 5. 实际成本、计账与资源限制

| 范围 | whole 秒 | soft / hard 秒 | soft overrun 秒 | 新增 GPU-h | 接受范围 |
| --- | ---: | ---: | ---: | ---: | --- |
| actual K3 十验证，0 训练 | 176.6783818155527 | 1800 / 3600 | 0 | 0.04818810004533993 | 实际参照执行记录；只记一次 |
| 独立实际 audit 完整闭合 | 719.8785105217248 | 900 / 1800 | 0 | 0 | reference evidence metadata complement only |

actual 连续 GPU phase **173.47716016322374 秒 = 0.04818810004533993 GPU-h**，从 archived
驱动 startup admission 前时钟跨全部 jobs 到 owned reap；**不是 worker 秒相加**。
全轮 176.6783818155527 秒含 prepare/import/archive/checkpins/startup/gaps/evaluation/aggregation/
cleanup；whole hours `0.04907732828209797` 不是额外 GPU-h。与文档/CPU audit 不重复记账。

独立实际审计核 owned first-job start 至 last reap envelope **173.40569350868464 秒**、10 worker
成功退出及 owned driver reap、11 次只读余量 admission。原 GPU start/end 变量未独立序列化，
故只接受 duration arithmetic、归档源操作与完整 owned timing envelope，不声称独立再现时钟位型。
本机共驻，不干预邻居；实际 UUID `GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b`，
10 fresh worker allocator 起点为 0/0。postflight 最大 peak allocated **43148800 bytes**、
peak reserved **71303168 bytes**，非整卡总显存；共享 GPU 时间未作邻居校正。

176 病例产生 **784 model forward calls / 2352 internal K steps**；region/baseline 不额外调用模型。
调用数不是 FLOPs、延迟或 speedup；没有本子工作新 FLOP/latency profile。

audit_result 内层 elapsed **560.5624908776954 秒** 不代替 final closure 全程
**719.8785105217248 秒**。审计 0 GPU/0新训练/0新评估，无 owned child/background handle，
`owned_unreaped:false`；两预算均无 overrun、无 hard 截断。文档子工作另冻 300 soft / 600 hard，
早入口计时及最终 SHA 回执在 `/tmp/r7_k3_doc_freeze_20261003_a0tnxbfl`，不混入 actual/audit 成本。

## 6. 完整库存与独立闭合：不制造原 manifest

外部 terminal owner map 位于
`/tmp/r7_k3_actual_launch_20261003_lleiqc08/terminal_artifact_filemap.json`，SHA256
**`f4b6c9d4d778d26bab7e6969f95416ae36b8d828e07c01e7888a5e8d66a8df43`**；
同目录 basic `verification_receipt.json` SHA256
**`06263a8374650ef2a088e33c8ef2eb9bceae75dd15f4bc562f4a9fc143011733`**。
它们是 postflight 声明/核对，不单独替代独立接受，不伪造执行前原 frozen whole-owned manifest。

独立实际审计递归核 **136 actual regular files = 49 postflight generated artifact pins +
80 archived extracted code pins + 7 frozen companion pins**，全 before/after 相等；
**无需排除任何原输出文件**（own map 在外部）。无遗漏/多余/不匹配 pin，拒绝全部 symlinks
含 broken；`failed_attempt_seal.json`、`publication_failure`、`publication_failure.json` 均不存在。
原目录没有 frozen whole-owned manifest 这一事实保留，接受来自新的独立 metadata complement。

独立 audit 核 **24 unique opaque inputs** 和 store metadata/stat before/after 不变，checkpoint 只
opaque bytes SHA256、0 deserialization。审计日志曾说 26 explicit inputs；最终 closure 已说明
实际计数来自完整 pin dictionaries 为 24，不把 narration 当真库存。本页不重做这些输入 hash。

独立最终回执目录为 `/tmp/r7_original_k3_actual_audit_wjar1kjk`，本页已实读最终 JSON 并核
字节摘要、closure cross-pins、acceptance scope；所有 1530 CSV 行 exact match、170 full-model
cells、单位与 normalized D/P/T/std²/pooling 已被该 audit 核验，不在本子工作重复数值实验。

| 审计回执/证据文件 | SHA256 |
| --- | --- |
| audit_result.json | `14fb35178f646b4d50ad60b0f50c8ff816dbe1ae7ff597fdc3b674326d15b0a1` |
| audit_closure.json | `7f0e755b24428b8a339750d26f9fbf1ab18ac826f4e2f1fadf1bfa6856610ab0` |
| closure_receipt.json | `226121c83175db96d7598f67e81ce0b6a9d1127384fb1e9d9f4643e5f0b85cd7` |
| registration_receipt.json | `a275b6957f922c5f3f8af3affce32c60f2d31276f36ad128861a2a6d7906eeaa` |
| independent protocol.json（文件字节） | `43aefc885d1e792d04ddeda26d96fe04428dd39d2c390e33492cdcecf0bc8d74` |
| input_qualification.json | `e2cc8f28e6a9c6f1aed319fadea1eb7fd1103a5535615a6cebe7b76ffa15a5fb` |
| full_model_170.json | `59c3e901c16e6aa2c0ed17addadfcfebc63e060ed5b1bc949a5c1e30b9a281c4` |
| before_pins.json | `bbe9208f9164bc098e39e1bdeffa7fddad259e99a125cec2a6ca51cf287016ac` |
| after_pins.json | `a9ca75561997dcb3d91f9ef84d6513a9f7a345cc417632431a02c19979a13376` |
| final_after_pins.json | `c238d8f5c2d994853c1a8bc7b43ba69ec52cefb566c3b8474930d85152226a91` |

audit_result 为 `accepted-original-k3-reference-evidence-only`，final audit_closure/closure_receipt
为 **`accepted-reference-evidence-metadata-complement-only`**，`accepted:true / scientific_claim:false`。
原 candidate `complete-not-accepted`、原 success/finalized:true 均保持；没有写 acceptance marker
到原结果。全包终态以新独立 final closure 为准，不仅以 basic 或 attempt.success 判断。

临时 raw audit 与 owner map 后续由主链复制归档到 outputs 并给出精确 copy map；**本页子工作
尚未执行复制**，不捏造未来目标路径或复制 PASS。当前引用是上述确切 `/tmp/` 路径与字节 pins，
复制后只能核字节同一，不重新解释原结果/变造 manifest。

## 7. 状态、限制、未做与下一项

| 层次 | 实际状态 | 不得等同 |
| --- | --- | --- |
| 原 K3 execution | 10 eval success / finalized:true；176 cases；0 train | 天气技巧接受 |
| 原 reference_result | complete-not-accepted，未改 | 自报接受或原 manifest 完备 |
| 新独立 audit final closure | accepted-reference-evidence-metadata-complement-only | Gregorian 配对、历史逐位重放或 goal 完成 |
| 工程 SHA 523c819b… / CI 37155293949 | 委派时 in_progress；本页未检查接受 | PASS / accepted CI |
| 供独立 C 的原参照 | 完整数值/来源证据可引用；timing 接受限于记录范围 | C 已实跑或跨语义 paired positive |

可复现等级为 **config/source/data/protocol pinned numerical reference**；独立复核为
**identity-bound numerical(metadata); not GPU bitwise**。不是独立天气 oracle、跨设备/GPU
bitwise、完整 #72 历史源码 score replay 或科学显著性。单冬季、单区域、两 seed、小 val、已有
开发数据；封存 test 未开，未作外部泛化、SOTA、收敛或新 scientific criteria。

已确认：指定 actual/audit/owner 元数据实际存在、字节定位 pins、本文数值、独立完整库存/数值
受限接受回执。本页仅核 JSON/CSV/回执，不自称运行这些 GPU/evaluator/audit。未确认的新 CI
保持 pending；131 keys/参数量只来自实际冻结 evaluator 契约，不冒称独立 decode/recount。

未做：新训练/评估、K1/K2/K4 depth probes、GPU/FLOP/latency 重测、precision 新接受、独立 C、
UTC 分组、cross-calendar/architecture paired test、full CPU suite/new CI 验收、历史全 source replay、
test/天气 payload 读取、下载/数据发布、依赖安装、凭据/安全配置、issue 关闭、commit/push/index。
不改已接受页 `/data/esw/UrbanPiDiT_R2/docs/R7_74_STATISTICS_COMPLEMENT.md` 或原冻结 B 页
`/data/esw/UrbanPiDiT_R2/docs/R7_74_AUTOREGRESSIVE_ATTEMPT.md`；B failed、01 FAIL 与 M3 暂停不变。

接口、模型/数据、源码/tests、依赖、安全/凭据均无变化。兼容性风险是把 sourcebridge 当历史原
源码、K3 改称 M3 K4、旧 365.25/accumulated lead 与 Gregorian 作 paired 或元数据接受写成
科学 positive；本文显式分层并保留负/undefined。没有新增决策或放宽冻结标准。

下一项由主链完成 raw evidence copy-map 归档及精确工程 CI 收尾，再按原冻结节点推进 C/验收；
本页仅释放已完成文档子工作，不触发新 GPU、关闭 issue 或停止/裁定整个 goal。
若来源/代码/协议/产物 pins、136-file 库存或单位/充分统计发生变化，须重开独立证据资格，保留
原回执和失败，不倒改原 candidate/manifest/语义，不以加算力或删负值练到 positive。

### 冻结登记前补记（主链，2026-10-03）

- 原文的 CI pending 是文档委派时事实，不是本次冻结的终态。精确工程 `523c819b8d42a3f43163eb53509f76136d8038f4` / CI 37155293949 / attempt 1 / job 111297340351 已确认 **completed/failure**；1–8 success、9 tests failure、17 post skipped、18/19 success。回执 `outputs/r7_v2_remaining_acceptance_20261003/engineering_523c819_ci_network_recovery_20261003T215523Z_6cb51964/verification_receipt.json` SHA256 `a2e054d5903f4f48a55eb046780de803efcebdc5d9e6629c617e2f0def05ab5f` 为 `accepted:false`。官方 jobs/check/HTML 于 2026-10-03 匿名访问；未得到远端 failing test 名称和计数。whole 988.875 秒 / soft overrun 88.875 秒 / hard 1800 未截断。独立 metadata 接受不把此 CI 改成 green。
- 实际 audit 与 owner map 的 raw copy 已完成，map 为 `outputs/r7_v2_remaining_acceptance_20261003/repair_evidence_supplemental04_copy_mapping.json`，SHA256 `502bdfe1ef8012d93918093e67dd041433c7452ced0a46e78b430951c01b9c00`。精确目录来自该 map；复制只保存原 bytes，不新造原 manifest、接受标记或重复 GPU 执行。
- 原文档 owner 的冻结 SHA `3c30fb25460293b7da9308c869e1ac36c90d62e90f41076d3c73dfbdc1610ae4` 及回执保留为补记前 snapshot；正式 evidence index 必须使用本次实际 Git blob 的最终页 SHA 和 containing commit，不倒用旧页 hash。
