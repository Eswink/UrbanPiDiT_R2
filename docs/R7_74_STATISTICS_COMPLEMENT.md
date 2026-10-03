# #74 零 GPU 统计补全（定稿：仅元数据统计交付接受）

日期 2026-10-03；mainmodelV2 文档子工作；`scientific_claim: false`。

> **接受范围仅为 ACCEPTED-METADATA-COMPLEMENT ONLY。** attempt02 的完整元数据统计交付
> 已由最终独立 audit、registration 与 closure 回执接受，本页据此定稿；不是训练成功或科学 positive。
> 原 B 永久保留 `failed / finalized:false`；统计 attempt01 独立 inventory 资格 **FAIL** 不变，
> 不能用其自报 `finalized:true` 覆盖失败。数值结果仍为 `negative_or_mixed / selected_mode:l6`，
> 不支持对强 update-count 控制的科学增益，不把 #70 或最终研究 goal 判为 positive/DONE。

## 1. 实际范围（actual scope）

本页补充而不修改原冻结页
`/data/esw/UrbanPiDiT_R2/docs/R7_74_AUTOREGRESSIVE_ATTEMPT.md`，依据已接受决策
`/data/esw/UrbanPiDiT_R2/docs/decisions/0036-fp32-loss-audit-and-statistics-complement.md`。
授权仅创建/更新本补页：不改原页、索引、Git、源产物、代码、配置、依赖或决策。
工作上下文由协调者声明为 HEAD `adb2866`；本子工作未执行 Git 身份查询或写操作。

- 原计算目录为 `/data/esw/UrbanPiDiT_R2/outputs/r7_74_autoregressive_20261003_attempt01`。
  六训练、1600 optimizer updates、三十 K4 单 lead 验证和 528 case evaluations 已完成；
  聚合失败仍是真实失败，不把 36 个成功 worker 当作整轮成功。
- 新补全使用排他新目录、新冻结 CPU 协议及独立 helper 归档，消费原 CSV / provenance / loss
  记录及身份元数据；原预测协议与新统计协议同时绑定派生记录。不是在原目录 run/finalize。
- attempt01 与 attempt02 均记录新增训练 0、新增评估 0、GPU 0、`test_read:false`，
  `weather_getitem:false`、`source_write:false`；补全不重复原 GPU 成本。
- 本页编写者只读指定协议、attempt、provenance、paired comparison、aggregate result、表及最终
  审计回执，核对回执字节 SHA256 和 closure 交叉 pins；未读取 checkpoint 张量、天气数组或
  source fields，未运行训练、评估、GPU 或整套独立审计。独立 audit agent 648 的工作不由本页重复。

## 2. 身份与来源（identities）

原运行不是仅凭一个 Git commit 声称干净代码：原 provenance 声明
`working_tree_modified:true`，身份为精确 HEAD 加 content-addressed dirty source closure。
原基线 commit 为 `616b029dce569b92bd08295737981512180a1ad3`；重聚合来源是其冻结归档，不是
用当前 HEAD 替换旧训练/评估代码。以下摘要来自原冻结页及新补全协议/provenance。

| 身份项 | SHA256 / canonical digest |
| --- | --- |
| 原 forecast canonical protocol | `9f76e6e3e0eb0d7e27d1aaaa8d59ff7616241ff50e13c30b26be7f13f5e62b02` |
| 原 protocol 文件字节 | `2d937850fb817d715219050c9418811f8d3dbf56a4d48b3e13fe253afd9557dc` |
| 原 failed attempt 文件 | `1868e0a5904d9916b19845475a203c4eaf7aa84d37ecb8e51563bf993bbb0664` |
| 原 93 成员 code.zip | `d3d10a74c77303849d38e06a44fa2fcb06d2ac8da0c7ddc4646b4218e0cb8bd8` |
| 原 source tree | `5de6b16dccb9e59116116017a9f5286ac0576c8d7db270d0f6f8fe7b9c1cf2c9` |
| 原 model code | `0cc9c16e7a12bf23150fb04e13acb72f548f8cb461ad64df1d45123b387a223e` |
| 原真实 source 身份（仅元数据，不读 source fields） | `496084a9260bacfaf6293a01d89439c1e49d6afa8f09bc1f51d89a1d1f9bda21` |
| 原 data identity | `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07` |
| 原 scale sidecar identity | `4fed1c78e4c4c09a41d95457649925b02d0c8ebf89aa47c2ac8dc6496734912d` |
| 361 原文件 pins 文件 | `e0151b6345453b581e44088e7fdb7eaa29af24a56a0701b033782b2b0317dc84` |
| 361 原文件集合 digest | `ef7f7b268e60752c22915049939b652ad4360ca2505a7108aef2636e8b4f6e13` |
| attempt01 statistics canonical protocol（资格失败） | `ecd8ecdc33151537cd11c50c08036d19dfbaeec712742cf09732ce696fd721af` |
| attempt02 statistics canonical protocol（仅元数据统计接受） | `076f28dfe88cd6e600d0ed37089f40b8e0562008e5d254facb99f93bb9a449e9` |
| 修正前训练 loss validator | `0e22a1d3506e762ef3a7a0b38d1e512715b53980f2bf72e759ee10edbec50de3` |
| 修正后训练 loss validator | `ef8759394a667b513259db26647408cf0ddbbdc775ef6607462800bc69ec29aa` |
| FP32 objective receipt helper | `96643a94cc44860b816e23e85a32c72c628dda7039c08a720eb50132d7e864be` |
| attempt02 active complement helper | `187c52f6037886363afda4ad28df4677ac13a753d7e9fa4327945d5f79dab4e9` |
| attempt02 complement_code.zip（协议声明，未在本页解包） | `97801a2403d2e994f710257fc01354d5aef6ed5448c377e49437fcf9b55aa258` |

361 原文件共 3,089,276,264 bytes；pins 位于
`/data/esw/UrbanPiDiT_R2/outputs/r7_v2_remaining_acceptance_20261003/b_failed_source_pins.json`。
本页披露其身份，不重新读/hash 原 checkpoint 或源天气。原 BUILD_COMPLETE、只读 preflight 与
train/val/source/sidecar 契约仍在原记录中，不绕过身份检查、不重新发布数据。

### 可定位的统计产物

下表文件均位于
`/data/esw/UrbanPiDiT_R2/outputs/r7_74_stats_complement_20261003_attempt02`。
字节摘要是本页只读提取并与最终审计 pins 对表的定位信息，接受依据是下列独立最终回执，
不是 hash 本身。不声称这些文件与 attempt01 逐字节一致：统计协议及其逐行绑定不同。

| 文件 | SHA256 |
| --- | --- |
| `protocol.json`（文件字节，区别于 canonical digest） | `4f64911351f6ab94aecf1f4c12fde1233e504627638728f1c5374fde8b5f5f6d` |
| `attempt.json` | `072a01f385c9b2e82df497cfea0220bf47d422aa70f5a0d22ccefb62c51839b0` |
| `artifact_manifest.json` | `3d4b512a2110626077dbd9eef8c25de22830161736750827871931e81b91abe5` |
| `worker.log`（actual = manifest = attempt pin） | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `provenance.json` | `b83760792c8726c0963b635f2a2d59668344bc761bfec4b65bef482655fa96cc` |
| `aggregate_result.json` | `545d63a42b69d8c42ce345a5b17fbe0ecd650e0f44580c7addf8caf99cf16eb8` |
| `paired_comparison.json` | `84f452d72330f5ad46b142b3dc6a896558e7ed5175209f714913579607918caf` |
| `rmse_table.csv` | `3150a7a5326af291cc5d90db81fcbcc4e8b9b17f7c232e209fa2e9b34fb0a95f` |
| `acc_table.csv` | `b505eba216b9222939da5a6ee9f0f3d66906e0073f1333ee58890d7da09e2870` |
| `aggregate_table.csv` | `c9dbf448500ab6c58c4a1b2385d7794517f1cd585baafe9708adcc52fe8bd5ca` |
| `baseline_table.csv` | `7ab680b0f07d08a0ef66d598dcb353abcea14ae56ca7628536c121b2603a5b61` |
| `case_table.csv` | `6c39c8382e34056fee5055ef77f054a825ca61b1c7d52bfa9c6c08253ea9ca82` |
| `training_table.csv` | `8d90281f0653fc8c7bb03533719a0f8777549a70a9482341f8681ead1621197a` |
| `evaluation_table.csv` | `a0d89a5878b998978a5de7df29fa46aa860be0bea0bd7467dccd3f4ff3f5de79` |
| `parameter_table.csv` | `cb4d6a32d9f03a1c978205274e8412acf90eafee29b98e97fced31c4120c244d` |
| `flops_table.csv` | `cc2b8f0ee4772cf4546eed430282803218d1ef94967d41bc09fcce16f8d91a76` |
| `allocator_table.csv` | `04e5b6fc80252435fe8acb721c9d28cb479ee54395fe163de015486dd0d277c8` |
| `state_table.csv` | `4bdae60e91b0588d3003feb8797ba15e0373fa07def0b7d15e6f24fca68f0039` |

attempt01 的 `/data/esw/UrbanPiDiT_R2/outputs/r7_74_stats_complement_20261003_attempt01/attempt.json`
字节 SHA256 为 `f078384d9cde92a214e825c0662e3b2d4db4e26010010038f360a13516f64d25`。
该失败资格记录及其全部派生结果保留，不覆盖或升级。

### 最终独立接受与闭合证据

回执目录为 `/tmp/r7_actual_metadata02_audit_ymwa6mjo`。本页定稿时已读取四个 JSON，
实核前三项字节摘要与协调者提供值相同，第四项 closure_receipt 交叉绑定前三项；
不重跑审计脚本，也不自行扩大接受范围。

| 回执文件 | SHA256 |
| --- | --- |
| `audit_result.json` | `0f965f2e7fd8dff509637835866737f3961003eb56b434704abf515676bf4e8b` |
| `audit_closure.json` | `d3e493da84432fef9316a2475baa06c569f61511e1c4f587755fd84ce5dea11f` |
| `registration_receipt.json` | `323c60f86946ecdefa169390351d98d34f4c515dc8f282905ba422d83c7217ed` |
| `closure_receipt.json`（完整 audit 证据 pins） | `cf4004e7dc6b0838fd5f39b8c0d611b4f48aae1bf85a81ee4acd9e66e812813b` |

最终回执为 `accepted:true / accepted-metadata-statistical-complement-only`，
`scientific_claim:false / scientific_state:negative_or_mixed / selected_mode:l6`。
`audit_closure.section_receipt_semantics` 明确：复用的 numerical_replay/qualification_result
内部 `accepted:false` 表示计算分节不独立裁整包接受；其 verified 结果由最终 audit/registration/
closure 的受限 `accepted:true` 闭合，不是漏 pin 失败，也不是把分节假改成科学接受。

独立实际审计已核实以下既有资格出口。

- 全部 **361 原文件 + 113 补全文件** before/after pins 不变；原 checkpoint 仅 opaque 字节 hash，
  没有 torch load。原 B failed/finalized:false 和被拒绝的 01 均不改。
- manifest **111 pins + 2 exclusions**，只排除 `artifact_manifest.json` 与 `attempt.json`；
  没有遗漏、虚构 pin、failure markers 或 symlinks（含 broken）。实际 worker.log、manifest 与
  parent-reap 最终 attempt pin 三方一致。
- 原归档 **93** 成员、仅许可修复后的 runtime **94** 成员、helper zip **5** 成员、model Python
  **28** 成员身份已独立核；不 import 当前 active code，model/source digest 与旧归档相同。
- **6 train / 30 eval / 36 receipts** 的父来源、provenance、时钟/owned-reap/headroom 记录及
  原状态已核；没有新训练/评估或 GPU。**1530 metrics / 765 aggregate / 3 × 255 pairs** 的
  物理充分统计、same-case、容差、负/undefined、配对和原候选规则重构与刊出数值相符。
- 所有 **11 CSV** 的精确行库存与逐行 `row_sha256` 已验证，非仅文件存在或行数近似。

| CSV | 精确行数 |
| --- | ---: |
| `rmse_table.csv` | 1530 |
| `acc_table.csv` | 1530 |
| `aggregate_table.csv` | 765 |
| `baseline_table.csv` | 3060 |
| `case_table.csv` | 30 |
| `training_table.csv` | 6 |
| `evaluation_table.csv` | 30 |
| `flops_table.csv` | 6 |
| `parameter_table.csv` | 6 |
| `allocator_table.csv` | 36 |
| `state_table.csv` | 1 |

该接受仅保证失败 B 的完整元数据统计补全与证据闭合；不认证天气真值、不改变原实验失败，
不授权 adaptive/GPU，不提供 rollout 假设支持。精确全程 CPU 审计成本见 4.5，非内层 elapsed。

## 3. 工程失败与具名修复（engineering failures）

### 3.1 原 B 的 FP32 / binary64 运算域错配

原聚合器以 Python binary64 重组已从 FP32 tensor 导出的 `L6 + 0.5*L12`，却使用物理指标
`1e-9 / 1e-12` 容差核训练 loss。首条真实记录为 loss `0.2679187059402466`、
L6 `0.15507206320762634`、L12 `0.22569331526756287`；binary64 组合为
`0.2679187208414078`。按真实训练逐次 FP32 乘法、再 FP32 加法重构
`round32(L6 + round32(0.5*L12))`，两 seed 共 400 条逐位相等；旧 binary64 拒绝 99 / 88 条，
最大偏差分别 `2.9802322387695312e-8` / `1.4901161193847656e-8`。
这是训练损失**元数据核验**缺陷，不是改变目标、重训证据或天气收益。

具名修复为 `exact-FP32-loss-multiply-then-add-receipt-check`。新协议只允许替换归档中的
`training/r7_v2_results.py` 并添加 `training/r7_v2_objective_receipt.py`；有限性、数字类型、
篡改 loss 的拒绝保留。`tables.close`、物理指标容差、科学不等式、seed、端点与病例均不放宽。

协调者提供的独立工程证据为：旧 38 测试绿色仍存在 3 个 guard FAIL，失败记录保留；
修复后 58 + 28 项通过。不能把旧绿色改写成旧验收通过。本页未重跑这些测试，
未获逐项 guard 名称/回执摘要的部分不补造。

### 3.2 attempt01 的 worker.log inventory 漏 pin

attempt01 自报 `aggregation-complete / finalized:true`，实际耗时 `51.71911076363176` 秒、0 GPU；
独立实际 audit 却发现 **113 文件 / 110 pins**。排除项只允许 manifest 自身与 attempt，
**不允许排除 `worker.log`**。即使日志为 0 bytes，仍是实际文件；未 pin 令资格 **FAIL**。
自报 finalized、完整数值库存或候选状态不能消除该失败，不能将 01 称为 accepted。

具名后续修复为 `worker.log inventory pin + parent-reap final hash check`：manifest 直接 pin 日志，
owned worker 被 parent reap 后，再核实际日志、manifest pin 与最终 attempt pin 相等；错误或
结束后的新增字节须被拒绝。协调者报告修复 59 测试 + 3 项独立检查 PASS，证据目录
`/tmp/r7_worker_log_final_review_gxaumy0h`；这是 helper 工程证据，不是 attempt02 实际接受。

attempt02 使用新目录、新冻结协议与新的 helper zip；不原地修补/重试 01。其 attempt 记录
`worker_log_sha256 = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`，
即空字节文件的 SHA256。最终独立实际审计已核 **113 文件 / 111 pins / 2 exclusions**，
无遗漏；actual / manifest / parent-reap 后最终 attempt 日志摘要一致。该受限接受来自最终回执，
不是先前主链 quickcheck 或 helper 单测；01 的资格 FAIL 不变。

## 4. 实际数字与四类成本（actual numbers and four costs）

**本节为独立接受的元数据统计，仍非科学 positive。** 原首稿分别读取 01 与实际 02；
最终独立 02 audit 又核物理充分统计、RMSE/ACC/配对数值与完整库存一致。接受只绑定 02 的
最终回执，不因两次物理数字相同而升级 01；完整负面、undefined 与原失败继续保留。

### 4.1 全库存、计算单位与坏值口径

- 2 seed × 3 arm × 5 lead × 3 region × 17 variable = **1530** 个 seed 级 region metric
  records；跨 seed **765** aggregate rows；三组官方配对各 **255** cells。
- val lead 6 / 12 / 24 / 48 / 72 h 的完整病例分别为 22 / 21 / 19 / 15 / 11，每 arm/seed 相同，
  六组共 528 case evaluations；短 lead 不裁成 72 h 的 11 病例。原 train 为 185 个合法 t+12 窗口。
- `full` 4225 点；`interior` margin 2、3721 点；`edge_2` 504 点。区域有重叠语义，
  不是三组互不相关的天气样本。baseline 是同病例 persistence 和 train-only 月/UTC-hour/网格均值
  climatology（2016 train、188 步），不是 test 拟合。
- 每一变量/区域/lead/seed 单独以原权重的物理 MSE 充分统计计算 RMSE。独立重构从归档 case
  D/P/T normalized sufficient statistics 乘冻结 train std² 转物理量；region pooled sums 先按精确
  初始化数合并/归一，再做物理转换，最后开方/比值，不取天气数组或事后重拟 normalization。
  对病例 MSE 取原规定平均后开方；跨 seed 的 `rmse_seed_mean` 是 RMSE 均值，
  `rmse_pooled_cases` 是病例 MSE 按病例数合并后开方，两者不能混称。
  MSE skill 为 `1 - MSE_model/MSE_climatology`；pooled ACC 用原 anomaly dot / energy
  充分统计合并后计算 `dot / sqrt(forecast_energy*target_energy)`，不是简单平均 case ACC。
- 温度 t2m/t850/t500 的 RMSE 单位 K；u/v 风分量为 `m s**-1`；mslp 为 Pa；
  z250/z500/z850 为 `m**2 s**-2`；q850/q500 为 `kg kg**-1`。MSE 使用对应单位平方，
  skill/ACC 无量纲。不同单位的 RMSE 不跨变量平均，training normalized loss 也不当物理 RMSE。
- 所有 1600 更新 loss 记录、200/400 固定端点、原每 20 更新 checkpoint 与日志保留；
  不根据 val 事后挑更新数。负值和 undefined 均不剔除，不把 undefined 写成 0。

| 诊断计数范围 | 记录/暴露数 | 负 MSE skill | 负 ACC/dot | undefined |
| --- | ---: | ---: | ---: | ---: |
| 模型 case-variable-region 暴露 | 26928 | 11255（worse_than_climatology） | 5656（negative_acc_dot） | 0 |
| 模型 seed 级 region records | 1530 | 784 | 322（pooled ACC） | 0 |
| 模型跨 seed aggregate rows | 765 | 386 | 156（pooled ACC） | 0 |
| 两种 baseline 的 region records | 3060 | 1014 | 204（defined pooled ACC） | 1530 ACC undefined_zero_anomaly_energy |

26928 = 528 × 17 × 3，是反复计入 arm/seed/variable/region 的诊断暴露，不是 26928 个独立天气事件。
baseline 的 1530 个 climatology ACC undefined 与 26928 个 zero-anomaly-energy 暴露原样保留；
zero_climatology_mse 为 0。本表不是显著性样本量或真天气 oracle 认证。

基线表的 **1014** 个序列化负 MSE-skill 行是表层符号计数；最终独立 audit 按冻结容差重构，
标记真正 worse-than-climatology 的 persistence region rows 为 **840**、per-case 为 **13782**，
climatology 两级 genuine negative skill 均为 **0**。精确 climatology skill 的序列化 0 保留；
逆物理单位的 binary64 舍入可能出现可忽略负号，但只按未改的 `1e-9 / 1e-12` 容差核，
不把舍入负号当新天气劣化、不据此删行或放宽阈值。3060 baseline region 引用在三 arm 重复，
唯一 same-seed/kind/lead/region/variable cells 为 1020（每 kind 510）；每 kind 唯一 per-case
rows 为 8976。baseline 重复引用与重叠区域不当独立样本。

### 4.2 primary t2m / full / K4 的精确物理 RMSE（K）

| seed | lead h | 200-update continue_l6 | 200-update rollout_l6_l12 | 400-update equal_compute_l6 |
| --- | ---: | ---: | ---: | ---: |
| 41 | 6 | 2.488010656727011 | 2.399201147893384 | 2.245044396750102 |
| 42 | 6 | 2.1840923554612273 | 2.1829710698297764 | 2.0618285692503577 |
| 41 | 12 | 3.237403300077342 | 2.9822891212183555 | 2.9418834428035905 |
| 42 | 12 | 2.9051251627039134 | 2.8616239770116922 | 2.7306562389830353 |

所有 Δ 都是同 seed、同病例、同物理单位的 **focus minus baseline**；负为 RMSE 降低。

| seed | lead h | rollout − continue（K） | rollout − equal_compute（K） | equal_compute − continue（K） |
| --- | ---: | ---: | ---: | ---: |
| 41 | 6 | -0.08880950883362715 | +0.15415675114328176 | -0.2429662599769089 |
| 42 | 6 | -0.0011212856314508635 | +0.12114250057941867 | -0.12226378621086953 |
| 41 | 12 | -0.2551141788589866 | +0.040405678414765056 | -0.29551985727375163 |
| 42 | 12 | -0.04350118569222117 | +0.1309677380286569 | -0.17446892372087808 |

两 seed 两 primary leads 均优于 200-update L6，但均差于 400-update L6。
不能省略更强 update-count 控制，以较弱对照的收益声称科学 gain。

| arm | lead h | seed RMSE 均值 K | pooled-case RMSE K | pooled MSE skill | pooled ACC | 病例暴露 | worse-than-climatology / negative-dot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| continue_l6 | 6 | 2.336051506094119 | 2.340988727564845 | 0.2618857231570074 | 0.6633621810898819 | 44 | 10 / 1 |
| rollout_l6_l12 | 6 | 2.29108610886158 | 2.293635633635679 | 0.29144459542390233 | 0.6699228525970312 | 44 | 9 / 0 |
| equal_compute_l6 | 6 | 2.15343648300023 | 2.1553841180119155 | 0.3742882397906794 | 0.7108740726549518 | 44 | 6 / 0 |
| continue_l6 | 12 | 3.0712642313906278 | 3.075754569071393 | -0.27474869050956885 | 0.4756277959190359 | 42 | 26 / 5 |
| rollout_l6_l12 | 12 | 2.921956549115024 | 2.922579356351643 | -0.1509430731520427 | 0.4917715327445494 | 42 | 20 / 2 |
| equal_compute_l6 | 12 | 2.836269840893313 | 2.838235515821307 | -0.08547063556454115 | 0.5413087839342119 | 42 | 22 / 2 |

病例暴露 44 / 42 是同 22 / 21 初始化在两个 seed 上的重复，不是额外独立 val 病例。
12 h 三 arm 的负 climatology skill 全保留。

### 4.3 三组官方配对及冻结选择

每组 cell 按 #60 原规则：两 seed Δ 都 <0 才 improved，都 >0 才 worsened，其余 unresolved；
不使用 p-values 或跨单位汇总 score，不以 cell 总数替代 primary gate。

| focus − baseline | cells | improved | worsened | unresolved |
| --- | ---: | ---: | ---: | ---: |
| rollout_l6_l12 − continue_l6 | 255 | 113 | 33 | 109 |
| rollout_l6_l12 − equal_compute_l6 | 255 | 83 | 70 | 102 |
| equal_compute_l6 − continue_l6 | 255 | 77 | 37 | 141 |

原冻结选择规则要求两个 primary leads 的**每个 seed**同时优于 continue_l6，且不差于
400-update equal_compute_l6；否则保留 L6 给独立 C。实际元数据记录
`negative_or_mixed / selected_mode:l6 / no_posthoc_selection:true`，
`hypothesis_action: terminate rollout hypothesis only`，`independent_C_continues:true`。
这是已独立核验的原规则输出，不是科学正面结论；仅终止当前 rollout 假设，
不停止独立 C 或整个研究目标，不以工程/实验记录闭合替代 scientific support。

### 4.4 四类实际成本：参数、实测 FLOPs、墙钟、内存

**参数。** 六训练 arm/seed 的 `parameters = trainable_parameters = 2,968,259`。
两步共享模型参数，不是把参数量翻倍；baseline 为零训练模型，不借用模型 K4 深度标签。

**实测 FLOPs。** 原 CPU `FlopCounterMode / enable_grad` 的实际 forward 与 `loss.backward`
统计，两 seed 一致；这是原测量记录，不是本次补全重新 profile。

| 原物理 objective | 每更新实际 model forward calls / physical steps | forward FLOPs | forward + backward FLOPs |
| --- | ---: | ---: | ---: |
| L6（continue 与 400-update 控制） | 1 / 1 | 7,774,613,952 | 23,265,272,448 |
| 两步 L6 + 0.5 L12 | 2 / 2 | 15,549,227,904 | 46,587,416,832 |

反向包括内部 K 与跨物理步完整图，不用假定倍率。计数不含 unsupported elementwise/normalization
等全部训练开销。`equal_compute_l6` 是原 arm 名和可负担 update-count 控制，**不是严格等 FLOPs、
等墙钟或同 updates = 同 compute** 的证明；实测两步 forward+backward 也不恰为 L6 的两倍。

**训练墙钟。** 原 worker elapsed 包含 worker 开销；training loop 单列。以下来自实际表，
不从更新数估算 latency，也不把较小 loop 时间拿来替代失败整轮计账。

| seed | arm | updates | worker 秒 | training loop 秒 |
| --- | --- | ---: | ---: | ---: |
| 41 | continue_l6 | 200 | 140.76261591911316 | 73.26382527779788 |
| 41 | rollout_l6_l12 | 200 | 144.38420284260064 | 77.57030401565135 |
| 41 | equal_compute_l6 | 400 | 207.06185968965292 | 139.80597799271345 |
| 42 | continue_l6 | 200 | 137.02115666493773 | 71.12003517523408 |
| 42 | rollout_l6_l12 | 200 | 147.1321131810546 | 80.34992816857994 |
| 42 | equal_compute_l6 | 400 | 207.59856270439923 | 140.49549204483628 |

三十验证 worker 总 `2027.87038651295` 秒；每 arm 为两个 seed × 五 lead。
表中范围按 4 位小数展示，原逐 job 精确值在 evaluation_table；isolated 单 batch median 范围
是各 job 独立 forward 记录的范围，不是整个 physical rollout 的速度。

| arm | 验证 jobs | worker 秒合计 | 单 worker 秒范围 | isolated K4 forward median ms 范围 |
| --- | ---: | ---: | ---: | ---: |
| continue_l6 | 10 | 669.9397266404703 | 63.2648–68.6890 | 19.9502–23.7073 |
| rollout_l6_l12 | 10 | 677.0327446805313 | 66.4107–69.2271 | 20.0646–28.6119 |
| equal_compute_l6 | 10 | 680.8979151919484 | 66.7755–70.8312 | 20.0539–29.0046 |

isolated 范围来自 resident batch、3 次 warmup 排除、10 次同步测量；不包含 IO/transfers/metrics。
实际 evaluation loop 包含 IO、rollout、模型、baseline 指标与 CSV，不能与 isolated latency 混称；
共驻测量不构成严格 wall-clock speedup 声明。

**实际 allocator 内存。** 36 fresh CUDA worker 的 pre-init / initialized baseline 均为 0 / 0。
MiB = bytes / 2^20；peak allocated 与 reserved 分开，不能代 GPU 总显存或参数大小。

| 训练 arm（两 seed 相同） | peak allocated bytes | allocated MiB | peak reserved bytes | reserved MiB |
| --- | ---: | ---: | ---: | ---: |
| continue_l6 / equal_compute_l6 | 194268160 | 185.2685546875 | 224395264 | 214 |
| rollout_l6_l12 | 329185280 | 313.935546875 | 358612992 | 342 |

| 验证 lead h（六 arm/seed workers 相同） | peak allocated bytes | peak reserved bytes |
| --- | ---: | ---: |
| 6 | 39882240 | 46137344 |
| 12 | 43040768 | 71303168 |
| 24 / 48 / 72 | 43153408 | 71303168 |

### 4.5 原失败全成本与两次 CPU 补全成本

原 B 连续 GPU 从首 spawn 到末 owned reap 为 `3069.380453112535` 秒，
**0.8526056814201487 GPU-h**；36 成功 worker 合计 `3011.8308975147083` 秒不替代连续计账。
最早 CPU 入口到最终成本快照为 **3200.912431293167 秒**；包含准备、import/profile/archive、
启动/间隔、训练/评价、聚合失败与 owned 清理。receipt 序列化在快照之后，非精确 CLI-return 全时。
原 planned 5400 / hard 10800 秒，soft overrun 0，`owned_unreaped:false`，失败全额记账。

原冻结页记录独立成本审计已接受，范围**仅 wall/GPU 成本**；其回执位于
`/data/esw/UrbanPiDiT_R2/outputs/r7_v2_remaining_acceptance_20261003/b_independent_cost_review/revision02/audit_result.json`，
SHA256 `c00dd04ca68152a9be57f41a433e1234657abd6db6139a132ab9e131d3f900db`。
不能把成本接受等同原聚合或天气结果接受。

| CPU 补全 | whole_elapsed_seconds | soft / hard 秒 | soft overrun 秒 | GPU-h | 独立实际资格 |
| --- | ---: | ---: | ---: | ---: | --- |
| attempt01 | 51.71911076363176 | 600 / 1200 | 0 | 0 | FAIL，worker.log 漏 pin |
| attempt02 | 52.027686852030456 | 600 / 1200 | 0 | 0 | ACCEPTED-METADATA-COMPLEMENT ONLY |

两补全均从最早 CPU prepare/imports 入口计时，含 freeze、prepare-run 间隔、重聚合、输出与 owned
reap；10 秒 cleanup reserve；不是仅统计 worker 内核时间。独立 audit 自身成本另列，不混入补全。

| 独立 CPU audit | whole 秒 | soft / hard 秒 | soft overrun 秒 | hard 截断 | 新增 GPU-h | 状态 |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| 旧 attempt01 audit（协调者提供闭合值） | 1396.540345 | 900 / 1800 | 496.540345 | 否 | 0 | inventory qualification FAIL 保留 |
| 最终 attempt02 audit（audit_closure / closure_receipt） | 442.5259756054729 | 900 / 1800 | 0 | 否 | 0 | 仅完整元数据统计补全接受 |

02 最终审计全程含最早入口、非 time imports、protocol freeze、准备、全部 hash/metadata/numerical
replay、process return/log closure 与 receipt archive；`audit_result` / `registration_receipt` 内层
elapsed `301.0507361162454` 秒不代替 **442.5259756054729** 秒全程成本。无 background/child
handles，`owned_unreaped:false`。原 B GPU-h **只记一次**，新补全与两次 audit 均无新增 GPU。

旧 01 audit 软超后继续并记真实 overrun，未达 hard 1800、没有时长截断；这是工程 inventory
资格失败，不是预算失败或原 B 科学通过。协调者提供旧 `audit_closure.json` SHA256
`83cad7c2e2277e9c7f825b5cbb9daa42dd1b7af52d3fe395d56a9ca40cf026e6`；旧
`qualification.py` 未跑事实保留，不冒称那一轮完成全部 qualification，也不升级其 FAIL。
该旧回执路径/字节本页未另读核，值按协调者已封记录披露；02 的四项最终回执则已实读核对。

campaign 已记累计 **5.780427976370344 GPU-h** 不变，补全不重记原 B GPU-h。
历史 24 GPU-h 会计基数不是当前许可上限，本页不新增 GPU 预算或训练授权。

## 5. 状态分层（state）

| 层次 | 当前已知状态 | 不得推断 |
| --- | --- | --- |
| 原 B workers | 六训练/三十验证真实完成 | 原聚合成功 |
| 原 B attempt | failed / finalized:false；全额成本保留 | 复活为 success |
| 原 B 成本独立核验 | 仅实际 wall/GPU 成本接受 | 完整数值/科学接受 |
| 统计 attempt01 | 自报 finalized:true，但独立 qualification FAIL | accepted / 已冻结 |
| 统计 attempt02 | aggregation-complete；最终独立 ACCEPTED-METADATA-COMPLEMENT ONLY；52.027686852030456 秒、0 GPU | 原 B 训练成功或 scientific positive |
| 最终 02 audit | 442.5259756054729 秒；900 / 1800、无 overrun、0 GPU；最终回执闭合 | 内层 elapsed 代替全程成本 |
| 原科学选择规则核验 | negative_or_mixed；L6；两 primary 均不胜强 update-count 控制 | rollout positive 科学 gain |

M3 原 failed、补测 paused、any-unresolved、advance:false 不变，不重跑其 23 评估。
独立 C 继续原节点路线；这里仅终止当前 rollout hypothesis，不终止整个 goal。
不据本页把 #70 标为 DONE/positive，不自行关闭 issue 或判最终 goal 完成。
真正旧 #72 参照为 RW-A/K3，仍待实际运行；本 B 的 M3 aux_off/RW-A/K4 同 seed 父不是旧 K3 参照，
不能拿旧分数冒称 K4 比较。旧 365.25/累计 lead 与本 B Gregorian/固定 +6 h 语义也不能冒称完全匹配。

## 6. 限制与已确认/未确认的边界（limits）

- 已确认的是指定元数据实际内容、字节定位摘要、各表数值/计数及最终独立受限接受回执；
  本页已核最终回执摘要与闭合交叉 pins。原成本与测试按原冻结页/协调者证据披露，本页写作者
  不是这些运行、测试或整套审计的执行者。
- attempt02 已最终独立接受元数据统计；依据是 actual02 audit/registration/closure，不是 01/02
  数字相同或 finalized:true。独立核验解析记录与充分统计，不是重新测量真实天气场、
  syscall 层禁网证明或 GPU 训练重做；checkpoint 只 hash opaque bytes，没有 tensor load。
- 可复现层级为 **identity-bound numerical(metadata); not GPU bitwise**（identity-bound metadata
  reaggregation）；原普通 GPU 训练最多 code/data/protocol 绑定数值可复现，不能声称 GPU /
  cross-device / training bitwise。400 条 FP32 loss receipt
  运算位一致只涉及损失元数据重构，不外推为模型训练逐位一致。
- 单冬季、单区域、两 seed、小 validation 且数据已参与开发；没有未读 test、跨季节/区域或独立
  prospective 泛化证据，不声称显著性、收敛、SOTA。重复 cell/seed/region 不是独立天气 oracle。
- B 的 query 新开关仍 False；这里不是 0035 的 local-solar/history-offset/source-position 对照。
  对强 400-update 控制的科学增益缺失，不用工程通过、较弱对照改善或部分好 cell 抹掉负面。

## 7. 未做与影响（unperformed）

未重训/重评、未 GPU profile、未读封存 test 或天气/source arrays、未解码 checkpoint、未下载/发布
数据、未做新的统计检验或增加 seed/unroll、未新立科学判据、未放宽任何旧阈值。
未运行旧 #72 K3 真参照、独立 C、UTC 分组或 adaptive 实裁；这些均不因本页变为完成。

本子工作未重跑 58+28 或 59+3 测试、未运行 full suite 或新 CI、未复做整套独立审计。
02 最终接受回执已经取得并只读实核四项字节摘要、接受范围与 closure 交叉 pins；其余实际执行
是指定元数据解析、表计数/数值提取与 SHA256 定位。旧 01 audit 的 qualification.py 未跑事实
与 FAIL 保留；full suite/new CI/UTC/K3/C 均仍未跑，不用 02 元数据接受补成这些步骤的 PASS。

只改本页；未修改任何源产物。接口、数据、模型结果、依赖、安全/凭据均无变动；兼容性风险是
误把补全 accepted 等同原训练成功或科学 positive，故本页显式分层。
无配置、安装、commit/push、索引或原冻结页写入。

## 8. 已满足的冻结出口、重开条件与下一项（reopen conditions）

actual attempt02 的最终独立 audit、registration、closure 证据已封且实核摘要；本页仅据其受限
接受定稿，不另等权限、不替代整套审计。0036 及既有协议要求的 361 原 pins、93/94/5/28 归档
身份、113 文件/111 pins/2 exclusions、11 CSV exact rows/row_sha、6 train/30 eval、same-case
1530/765 与三组 255 cells、原物理容差/充分统计、全负/undefined、日志/parent-reap 最终 hash
及独立 600/1200 秒 CPU 补全成本已由最终回执验证；这不是新增科学标准。

若后续发现来源/协议/产物 digest 漂移、库存遗漏、行身份或物理单位/充分统计不一致、日志闭合
不一致，须按原协议重新打开**统计接受资格**并保留原回执与失败，不能原地倒改来源、01 FAIL
或原 B failed 状态。新来源/语义或另一研究假设须前瞻独立冻结；不事后改旧规则、删负值、
增加算力练到赢或借未运行的 K3 分数补 positive。

已接受的仅是**完整统计交付**，不是原 B 成功或强对照 scientific gain，#70 与研究 goal 不判
positive/DONE。按原规则保留 L6 给独立 C，当前 rollout 假设记录 negative_or_mixed 且仅终止该
假设。下一项在主路线继续既有 UTC 分组、旧 K3 真参照、独立 C 及验收补齐；这些仍未执行，
本页子工作不运行或授权它们。主 goal 不因文档子工作闭合而停止或自行裁定完成。
