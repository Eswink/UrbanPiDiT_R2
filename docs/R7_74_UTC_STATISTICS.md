# R7-74：B-only UTC 分桶统计（metadata-only）

- 登记日期：2026-10-03；`scientific_claim: false`、`actual_pass: false`、`scientific_gate_evaluated: false`。
- **已确认**：正式 corrected CLI 为 `actual-statistics-complete`；独立验收为 `accepted-official-utc-metadata-evidence-only`。
- **未接受**：原 owner receipt 的 whole-package，失败名 `OWNER_REQUIRED_ARTIFACT_FULL_LOG_PIN_MISMATCH`。本页不能将其改写成整包通过。
- 仅展开并分桶 **B 已评分 case 的标量充分统计量**；不是新 forecast、天气真值、训练成功或科学 PASS；不覆盖 M1，也不覆盖 C。
- 原 B 仍 `failed` / `finalized:false`；科学状态仍 `negative_or_mixed`、`selected_mode:l6`。B02 是独立 metadata complement，不复活原 B。

## 1. 证据层与接受边界

| 层 | 状态 / 接受范围 |
| --- | --- |
| 原 B `r7_74_autoregressive_20261003_attempt01` | failed/unfinalized；完整 36 worker receipts，不等于 driver 最终成功 |
| Complement01 | FAIL；`NEW_MANIFEST_UNDECLARED_INVENTORY`，不可作为 accepted 输入 |
| Complement02 | `aggregation-complete`；独立 `accepted-metadata-complement`，非科学接受 |
| 首轮实际 UTC `rdj7bd36` | failed；0 rows / 0 groups，未发布统计表，永久保留 |
| 标签修复 | 独立 `accepted-label-serialization-fix-only`；新 runtime 身份，不回改失败轮 |
| Corrected UTC `6duasm0f` | CLI exit 0；完整实际 20,400 rows / 400 groups |
| Owner 自审整包 | **NOT ACCEPTED**；原 required full-log pin 不满足 |
| 独立 UTC `qualified` | 接受官方 UTC metadata evidence；新独立公式复算，不以 owner 自审为 oracle |

独立 qualified 入口（本页核验 exact bytes）：

- Receipt：`/tmp/r7_utc_b_corrected_independent_20261003_4Qji7Sw2/qualified/verification_receipt.json`
  - SHA256：`594a11816066c28998586343f18ba29d91ac30b3aa512b030d7dea3d1ef3f611`
- Seal：`/tmp/r7_utc_b_corrected_independent_20261003_4Qji7Sw2/qualified/final_seal.json`
  - SHA256：`cecf4b1e999c3744a609630c4308b3fe3ed53a8ad18edb2501c5b2417d3e7aa3`
- 审阅代码归档：`/tmp/r7_utc_b_corrected_independent_20261003_4Qji7Sw2/qualified/code.zip`
  - SHA256：`350ee8cb1cbca52278470188861955e045c4f665b90b2b102b3b652a07e4b0b4`

## 2. 完整覆盖、公式与身份

实际覆盖 17 variables、5 leads（6/12/24/48/72 h）、3 regions（full/interior/edge_2）、seeds 41/42、
3 model arms（continue_l6/rollout_l6_l12/equal_compute_l6）、model K=4；全部 30 evaluation jobs。
每 lead 使用自己的完整 cohort，分别为 22/21/19/15/11 case，不做跨 lead intersection 或 72 h 缩窄。

- 独立审阅检查原 CSV **80,784 行**：model、persistence、climatology 各 26,928 行。
- 零训练 baseline 在重复 arm/K 导出全部语义一致后才去重；最终 30 model entities、20 baseline entities。
- 每个变量/单位/region/lead/arm/seed/K 保持独立；不跨物理单位平均 RMSE/ACC，不用平均 case RMSE/ACC。
- 先将 normalized anomaly `acc_dot/acc_forecast_energy/acc_target_energy` 乘冻结 training std²；MSE/climate MSE 原值已为 physical。
- 按 case 等权汇总五统计量，再计算 `RMSE = sqrt(mean MSE)`、`skill = 1 - mean MSE / mean climate MSE`、
  `ACC = mean dot / (sqrt(mean forecast energy) × sqrt(mean target energy))`。
- 独立实现为新 standalone stdlib / `math.fsum`，不 import producer helpers 或 owner review；数值容差未变：`rel=1e-9 / abs=1e-12`。
- 全部 20,400 row hashes、400 group/case/source/time identity hashes 已核对；同 lead 样本身份跨 arm/seed 完整一致。

### 实际发布计数

| 实体 | 行数 | 负 MSE skill | 负 pooled ACC | Undefined skill | Undefined ACC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Model | 12,240 | 6,166 | 2,552 | 0 | 0 |
| Persistence | 4,080 | 2,352 | 620 | 0 | 0 |
| Climatology | 4,080 | 808（全部近零） | 0 | 0 | 4,080 |
| 总量 | **20,400** | 不作科学合并判定 | 不作科学合并判定 | 0 | 4,080 |

**400 groups、empty groups=0、empty cells=0 均为实测**，不是以预期计数代替执行。
Climatology 的 808 个 signed negative skill 全部 `|skill| < 1e-12`，保留 binary64 原值并单列，
不 clamp、不 filter，也不把它们当成新的科学 bad cases。
这些计数是变量/region 单元、且分别进入 init/valid 两套分桶；region 重叠、axis 重复，**不是独立天气事件数**。

原 producer 保存的 per-case 标签仍核验精确一致：model negative skill 11,255 / negative ACC 5,656；
persistence 13,782 / 5,472；climatology undefined ACC 26,928。
此输入口径包含跨 arm 重复 baseline 导出；保存标签与从舍入统计量重算符号的差异为 4,128 行，未改原 CSV/provenance。

### UTC cohort 小表

每一行给同 lead 原 cohort 的 `[00,06,12,18]` UTC case 数；不是跨 model 重复加总。

| Lead h | 总 case | init UTC `[0,6,12,18]` | valid UTC `[0,6,12,18]` |
| --- | ---: | --- | --- |
| 6 | 22 | `[5,6,6,5]` | `[5,5,6,6]` |
| 12 | 21 | `[5,6,5,5]` | `[5,5,5,6]` |
| 24 | 19 | `[4,5,5,5]` | `[4,5,5,5]` |
| 48 | 15 | `[3,4,4,4]` | `[3,4,4,4]` |
| 72 | 11 | `[2,3,3,3]` | `[2,3,3,3]` |

Naive timestamps 只按冻结契约解释为 UTC；aware timestamps 必须 UTC，且为 exact 6 h slots。
`sample_id/init_time/valid_time/valid_times` 保留在 group identities，逐 cell 链接 case/group hash。

### Null 与 baseline schema

- 内存/JSON undefined metric 为 `null`；CSV 表达为 blank，必须同时保留对应 status，不能填 0。
- 零 climate MSE：`undefined_zero_climatology_mse`；零 anomaly energy：`undefined_zero_anomaly_energy`。
- 空桶仍必须发布：`n_initializations=0`、空 case list、null metrics、`undefined_empty_hour_group`；本轮实际无空桶。
- Baseline `arm=null`、`K=null`、`model_depth_applicable=false`；updates/parameters/trainable_parameters 为 0，不伪装为 model K。
- 原保存指标在原 finite/numeric/status 容差校验后用于诊断标签签名；embedded `bad_reasons` 仍精确比较。
  修复只避免用舍入后的重算符号重新签保存标签，真正 negative/undefined、篡改或丢失标签仍拒绝。

## 3. 精确代码、协议与产物 pins

原 B ZIP 是 **93-member raw source archive**；静态 results/tables exact file bytes 与原 AST function pins 核验。
不得替换为 complement 的 corrected runtime 或当前 live source；不 import 原 model/data 代码。

| 身份 | SHA256 |
| --- | --- |
| 原 B `code.zip` | `d3d10a74c77303849d38e06a44fa2fcb06d2ac8da0c7ddc4646b4218e0cb8bd8` |
| 原 raw `training/r7_v2_results.py` | `0e22a1d3506e762ef3a7a0b38d1e512715b53980f2bf72e759ee10edbec50de3` |
| 原 raw `training/r7_v2_tables.py` | `8e039374596cbb2781ce83de814abd018176339b859867da5305c287f9044f8f` |
| B02 protocol canonical | `076f28dfe88cd6e600d0ed37089f40b8e0562008e5d254facb99f93bb9a449e9` |
| Corrected runtime four-file map | `aa212882688bf75eca273f60de9178f2fdceeaf40562afa996690ff87ff7be59` |
| UTC CPU protocol canonical | `9a82c1357750b41b9a930879543757dfe6ca537228ea862b2f5ba5d6bc57c5e8` |
| UTC CPU protocol exact file bytes | `b1ef1991718c92beb94785d8c8e87eadfd973be0012282f7a20826c654a67ab3` |
| `utc_group_metrics.csv` file bytes | `c51fbd1d7befff2e4042fe4b412b476f68c586da02df5b06d42d1c3cfe16c443` |
| Typed rows canonical digest | `401327f121b4e50cf47b09ac0380598da43c2d371c98b16d85a793d016f24205` |
| Group identities canonical digest | `f67ae82c5f4cb1d0cd81d0460a273ae4cb83542d0a1a076af73e3308b5a0915c` |

四 runtime 文件 pins：

| 文件 | SHA256 |
| --- | --- |
| `training/r7_v2_utc_frozen_stats.py` | `1ddc249b22fcee2636192938e7aa68efdfd4bcdc38949a92d0cc08316645299d` |
| `training/r7_v2_utc_contract.py` | `a7b487e895d8b455c1f02cd149ee6078f10f82898df66055b3af2cb25fe9cd3b` |
| `training/r7_v2_utc_statistics.py` | `064a5e10d9c33c701c2a44b685bfa448d5aed2c696b0dd78e1d8bd1c5228aef7` |
| `scripts/stats_r7_v2_utc.py` | `848fe64b938e79bafe54f529179e16f45e8cb8df2b7973ea7b9a080e12ac9274` |

Canonical protocol digest 与 serialized 文件 SHA **不是同一种身份**；CLI 使用 exact file SHA。
Prepare protocol 的登记值也是 exact file SHA `01a216364585e77e16515bdc5823227463010e98c09e03ad0b625fec0f13a286`，
不是其 canonical digest `8217d02656b72285db51518c0e55a4b866dd279525eb04bcfe0f8b3c3ddd1cba`。

## 4. Inventory、前后检查与路径

独立审阅绑定并 postverify **248 文件**；133 CLI inputs、210 preparation inputs、36 worker receipts + 36 timing sidecars。
输入及冻结 runtime 前后不变；所有 required publication markers 与 symlink 均拒绝，本轮无 marker。
没有声称重核原 B 全部 361 文件的 opaque library；其完整历史 audit 是另一条证据链。

正式 output 恰 10 文件：`run_started.json`、`protocol.json`、`helper_code.zip`、`source_identity.json`、`run.log`、
`utc_group_metrics.csv`、`group_identities.json`、`statistics_summary.json`、`attempt.json`、`pins.json`。
`pins.json` 钉住其余 9 文件，仅排除自身；helper ZIP 对应四 frozen runtime 文件。

- 原 B：`/data/esw/UrbanPiDiT_R2/outputs/r7_74_autoregressive_20261003_attempt01`
- B02：`/data/esw/UrbanPiDiT_R2/outputs/r7_74_stats_complement_20261003_attempt02`
- 新 freeze：`/tmp/r7_utc_b_corrected_freeze_20261003_6duasm0f`
- 新 output：`/tmp/r7_utc_b_corrected_output_20261003_6duasm0f`
- 全部 CSV：`/tmp/r7_utc_b_corrected_output_20261003_6duasm0f/utc_group_metrics.csv`
- 全部 group identities：`/tmp/r7_utc_b_corrected_output_20261003_6duasm0f/group_identities.json`
- Owner post-publication closure：`/tmp/r7_utc_b_corrected_review_20261003_6duasm0f/post_publication_closure.json`
  - SHA256：`68e06fa10d403e2014fc45624ab831f2f1b3eedf69eabad440b11596c4392c46`
- 独立数值复算：`/tmp/r7_utc_b_corrected_independent_20261003_4Qji7Sw2/independent_numerical_replay.json`
  - SHA256：`8bc5752195c9b3b88c07695b65b0d502b7a30a912f2d1338a44050a5d6e1aa58`

持久 raw-copy 路由（复制不升级原接受状态）：

- Map：`/data/esw/UrbanPiDiT_R2/outputs/r7_v2_remaining_acceptance_20261003/repair_evidence_supplemental04_copy_mapping.json`
  - SHA256：`502bdfe1ef8012d93918093e67dd041433c7452ced0a46e78b430951c01b9c00`
- Freeze copy：`/data/esw/UrbanPiDiT_R2/outputs/r7_v2_remaining_acceptance_20261003/utc_b_corrected_freeze_evidence_supplemental04`
- Output copy：`/data/esw/UrbanPiDiT_R2/outputs/r7_v2_remaining_acceptance_20261003/utc_b_corrected_output_evidence_supplemental04`
- Owner review copy：`/data/esw/UrbanPiDiT_R2/outputs/r7_v2_remaining_acceptance_20261003/utc_b_corrected_review_evidence_supplemental04`

Qualified 独立审阅已逐字节复制至 `/data/esw/UrbanPiDiT_R2/outputs/r7_v2_remaining_acceptance_20261003/utc_corrected_independent_final_evidence_supplemental04_addendum/qualified`。精确 map 为 `repair_evidence_utc_independent_addendum_copy_mapping.json`，SHA256 `3047f6c054112211ad166a30f0c0d97e0b54405d78d399c527e8669f911e5dab`；20 文件 / 445,508 bytes、完整收尾 148.091 秒、300 soft / 600 hard、overrun 0。复制不升级旧 owner whole-package，也不重新运行统计。

## 5. 时钟与成本（不能混用快照）

全部预算为 soft / hard；soft overrun 继续并记账，hard 截断不能接受。GPU 增量均为 0。

| 运行 / 出口 | whole seconds | soft / hard | soft overrun | 说明 |
| --- | ---: | --- | ---: | --- |
| 原 B failed | 3,200.912431 | 原冻结契约 | 保留原账本 | 原 GPU-h 0.852605681 全额只计一次 |
| 首轮 UTC CLI 发布 | 374.239732 | 600 / 1200 | 0 | failed，0 行 / 0 组 |
| 首轮 UTC 完整 owner 收尾 | 837.300674 | 600 / 1200 | 237.300674 | 失败不复活，不用 CLI 快照代替全轮成本 |
| Corrected CLI attempt snapshot | 147.126101 | 900 / 1800 | 0 | receipt 序列化前快照 |
| Corrected CLI publication | 147.193569 | 900 / 1800 | 0 | 正式完成发布，不含随后独立审阅 |
| Corrected 初次 owner seal | 712.884945 | 900 / 1800 | 0 | **不是最终全收尾成本** |
| Corrected owner final closure snapshot | 949.853657 | 900 / 1800 | 49.853657 | closure 序列化前快照 |
| Corrected owner complete terminal | **949.854352** | 900 / 1800 | **49.854352** | 全 closure 发布完成 |
| 独立 audit receipt 前快照 | 1,340.697945 | 900 / 1800 | 440.697945 | qualified receipt 字段 |
| 独立 audit final seal 快照 | 1,340.699071 | 900 / 1800 | 440.699072 | seal 序列化前字段 |
| 独立 audit complete terminal | **1,340.701317** | 900 / 1800 | **440.701317** | 主链记录的完成出口；无 hard truncation |

Owner / independent 的预算分别计，不合并成一次训练速度，也不双计原 B GPU 成本。
Prepare anchor 为 `5834115.397342777`，早于 setup/imports/pins/freeze；同 boot 贯穿 prepare gap、CLI、postflight 和 closure。
Literal `.venv/bin/python`；CUDA hidden、threads=4；socket 禁网先于 trusted imports；owned children 回收后才钉 final logs。

## 6. 失败与日志身份：不可修饰

1. 首轮 actual UTC 因保存 climatology skill `0.0` 与 binary64 重算 `-2.22e-16` 的标签签名不一致而失败。
   原数值容差通过，精确标签 gate 拒绝；原 source/失败 output/receipt 未改。修复与新 protocol 是另一轮。
2. 独立审阅先遇 prepare file-SHA vs canonical-SHA 的 harness 约定错误；数值复算已完成，旧失败 receipt 保留。
3. 随后的 owner whole-package 拒绝为 `OWNER_REQUIRED_ARTIFACT_FULL_LOG_PIN_MISMATCH`，不得因公式正确而忽略。
   原 owner `review.log` required full-file pin 为 `1408f8899cb7a98c04b3065753bf2b488b7b10e275f251655c13b696b470c861`（475-byte prefix）；
   completed full log 为 `49afc3bdc0edbb222ccc505b4ace901b9e432294035b7de326f791b1d464def7`（878 bytes）。
   Receipt 后追加发布事件解释了差异，但 **旧整包 required pin 仍不满足**。
4. 新 `68e06...` closure 与 final full-log pins 被独立核验，只接受官方 metadata evidence，不追认旧 owner whole-package。
   历史拒绝 receipt：`/tmp/r7_utc_b_corrected_independent_20261003_4Qji7Sw2/final/verification_receipt.json`，
   SHA256 `091dc31021dd01c7d2a65ab1bd258b06f12c0a0351059b668796dfce8e1af153`。

## 7. 工程状态与 limitations

工程 commit：`523c819b8d42a3f43163eb53509f76136d8038f4`。本次 actual 的 held HEAD provenance 是
`1615795d70117af608787ad6de15d2e5640b901c`，实际身份由四文件 exact pins 与归档代码决定，不能以新 commit 覆盖历史 pin。
主链 full CPU **3467 passed / 9 skipped / 3 warnings** 仅为工程记录；六 CUDA 与三 optional fixture 的 skip 不算通过，三条 warnings 保留，不是科学证据。
新 CI run **37155293949 在本页原登记请求时仍 pending**；冻结前主链已确认 exact `523c819b8d42a3f43163eb53509f76136d8038f4` / attempt 1 / job 111297340351 **completed/failure**：前八步 success，测试 step 9 failure，post step 17 skipped，18/19 success。正式回执 `outputs/r7_v2_remaining_acceptance_20261003/engineering_523c819_ci_network_recovery_20261003T215523Z_6cb51964/verification_receipt.json` SHA256 `a2e054d5903f4f48a55eb046780de803efcebdc5d9e6629c617e2f0def05ab5f`，`accepted:false`；官方 jobs/check/HTML 于 2026-10-03 匿名访问，测试名称和远端计数未取得。网络观察失败与真实 CI 失败分开；本地 full green 不代替该失败，也不覆盖未执行的新包 precision/C。

可复现等级：**identity-bound independent scalar numerical metadata replay**，不是天气/tensor 重建、GPU 逐位复现或跨平台 bitwise。

- 无新 forecast/forward、training/evaluation、GPU、checkpoint/tensor load、weather/test-field reads 或 network；不引入新数据。
- 只审阅已有验证 cohort；两 seeds、冬季片段、重叠 region 与双 UTC axis 只能描述，不做 significance/generalization 推断。
- 本页不覆盖 M1/C，也不升级 rollout hypothesis、原 B 状态或科学 negative_or_mixed/l6 判定。
- 不通过可解释 prefix mismatch 自动修复旧 receipt；历史 FAIL 与成本必须继续可见。
- 当前接受的是 **官方 UTC metadata evidence only**；科学 PASS、最终研究目标完成与工程 full-suite/CI closure 是不同事项。
- 下一项：主链按本页冻结 SHA 非递归登记索引，归档 qualified independent exact copy，另对新 commit 的 CI 完成状态验收；不由本页替代。
