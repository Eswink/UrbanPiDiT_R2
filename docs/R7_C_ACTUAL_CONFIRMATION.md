# R7-V2-C：独立 L6 三臂确认（actual C，独立接受）

- 登记日期：2026-10-04；`scientific_claim: false`、`actual_pass: false`、`test_read: false`。
- **已确认**：独立、排他、整轮冻结的 actual C 协议已实跑完成 144/144 job（9 训练 + 135 同 checkpoint K1/K2/K4 评估），
  并由独立审阅接受（`actual_c_review.json` SHA256 `aa44b8cdc89588874c56eeedd13a222511d6377034abae4a38e0efd19c0ad8dc`）。
- **未接受**：本页不是科学增益、显著性或 SOTA；adaptive 门未全过（`gate_met: false`），控制器未训练、`oracle_deployed: false`。
- 唯一判据统计来源：`outputs/r7_v2_comparison_20261004_attempt01/`（2.8G→7.9G 只读产物根）。

## 1. 身份与协议（运行前冻结）

| 项 | 值 |
| --- | --- |
| 输出根 | `outputs/r7_v2_comparison_20261004_attempt01`（排他新输出） |
| protocol 文件 SHA256 | `57e659de40d9f0fcb6e28102e693e236692459f187175e5ab466f827857ca231` |
| 内部 `protocol_sha256` | `ea0efb80e1b11eb7813fae74b4ee04ca67d145d16519eb8aa442af8d8851ad80` |
| base commit | `562e526afc5fdbdc99053a25ab20036b66a2a8bc`（C/M1 共同 HEAD，冻结） |
| code.zip SHA256 | `534274a62a7583e2f74c754af5306ce7036f4653a359e65e729ce300d69ac992` |
| model_code_sha256 | `551261c4a501a6f9727fef2a712b40ab67ec51adaa460298a6b5b59bf8ae5dc1` |
| source_tree_sha256 | `5eb63cff6472c30f29ff71f8faa553daf83d8409cc41d89fba6c16738fc976c8` |
| data_identity | `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07` |
| source_sha256 | `496084a9260bacfaf6293a01d89439c1e49d6afa8f09bc1f51d89a1d1f9bda21` |
| sidecar_identity | `4fed1c78e4c4c09a41d95457649925b02d0c8ebf89aa47c2ac8dc6496734912d` |
| 预算 | planned 10800 s / hard 21600 s（半开整轮：最早 prepare 入口 → 聚合 → 封印） |

组成：三臂 `old_ours` / `process` / `matched_generic` × seed 41/42/43，每臂 fresh scratch anchor、
各 400 次 L6 更新（Ktrain=4、full BPTT、selected=400）；评估为同 checkpoint 的 K1/K2/K4，
17 变量 × lead 6/12/24/48/72h × region full/interior/edge_2，val 病例逐 lead 22/21/19/15/11（完整五 cohort）；
零训练 persistence 与 train-only climatology 基线同表。144 = 9 train + 135 eval。

## 2. 执行终态（对账数字）

| 项 | 值 |
| --- | --- |
| attempt 状态 | `success`，`finalized: true`，`partial: false`，`budget_limited: false`，`owned_unreaped: false` |
| jobs | 144/144 completed（`jobs_planned == jobs_completed`） |
| whole clock | 11684.287771989591 s（soft_overrun 884.2877719895914 s，硬上限未触及） |
| GPU 连续计费 | 首 spawn→末 owned reap 11358.011682933196 s = **3.155003245259221 GPU-h** |
| manifest | `artifact_manifest.json` `stage-sealed`，1333 文件全量 pin（`files_digest` = digest(files_sha256)） |
| 独立审阅 | `review_receipt.json` → `actual_c_review.json` = `aa44b8cd…0ad8dc`；独立审阅 whole 204.25774977356195 s |

## 3. 主要读数（描述，不作显著性）

Primary 端点按运行前冻结的 `docs/R7_V2_CONFIRMATION_PREREGISTRATION.md`：t2m、lead 6/12h、K4、region full、单位 K、退化容忍 0.0，同 seed/lead/K/region 配对。

| seed | lead | old_ours RMSE | process RMSE | matched_generic RMSE | process−old_ours | generic−old_ours |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | 6h | 3.9089 | 2.3474 | 2.3474 | −1.5615 | −1.5615 |
| 41 | 12h | 4.8160 | 2.8351 | 2.8351 | −1.9809 | −1.9809 |
| 42 | 6h | 3.7350 | 2.4802 | 2.4802 | −1.2549 | −1.2549 |
| 42 | 12h | 4.3664 | 3.1132 | 3.1132 | −1.2531 | −1.2531 |
| 43 | 6h | 3.7789 | 2.4486 | 2.4486 | −1.3302 | −1.3302 |
| 43 | 12h | 4.4247 | 2.9646 | 2.9646 | −1.4601 | −1.4601 |

**已确认**：整个 package（新输入/结构体系）相对 old_ours 在 primary 6/6 cell 上严格改善（package_improvement 门 met）。
**已确认**：process 与 matched_generic 的 primary 同细胞差在 1e-5 K 量级且正负不稳（`k4|full|6h|t2m`：
−9.37e−6 / +3.88e−6 / +4.27e−6；12h：−8.35e−6 / +5.91e−9 / +4.15e−6），三态判为 unresolved——
**本实验不能把 package 改善归因于过程语义读写的独立贡献**；按 #75 原文，此时只能声明「整体系改善」。

全表三态（官方 #60 规则，765 cell/对，三 seed 同号要求）：

| 对 | improved | unresolved | worsened |
| --- | ---: | ---: | ---: |
| process − old_ours | 133 | 296 | 336 |
| process − matched_generic | 119 | 515 | 131 |
| matched_generic − old_ours | 133 | 296 | 336 |

(process−old_ours 与 generic−old_ours 的 totals 相同是两臂 primary 数值近同所致，非复制错误；
全表仍有 119/131 的 process−generic 非平 cells。)

## 4. Adaptive 门（冻结四门，逐项判定）

`stage_result.json` 的 `adaptive_gate` = `{evaluated: true, gate_met: false, status: "not-started", controller_training_executed: false, oracle_deployable: false, oracle_deployed: false}`：

- `package_improvement`：**met**（primary 6/6 严格改善）。
- `accuracy_cost_tradeoff`：**not met**——即 K4 相对 K1 的准确率改善与实测延迟的联合门未过。
  seed41/6h 例：K1 RMSE 2.6384、median 0.0328 s/batch → K4 RMSE 2.3474、median 0.0447 s/batch（同 resident-batch 范围）。
- `case_heterogeneity`：**met**（逐病例 oracle 描述存在异质性；oracle 用未来真值、不可部署、未部署）。
- `complete_engineering_evidence`：**met**。

**结论**：`gate_met: false` → 控制器不训练、不另立协议、adaptive 保持 not-started；不是 failed，是门未全过。

## 5. 独立接受链（外层）

| 层 | 产物 | SHA256 |
| --- | --- | --- |
| 实际 C 独立审阅 | `/tmp/r7_v2_c_actual_review_20261004_attempt01/actual_c_review.json` | `aa44b8cdc89588874c56eeedd13a222511d6377034abae4a38e0efd19c0ad8dc` |
| 独立审阅 receipt | 同目录 `review_receipt.json` | `ef9855a99e1ed58c9dce38857f3388707f4d56e7f3febb3886ad6465933b3ad6` |
| closure 报告 | `/tmp/r7_v2_c_closure_review_20261004_attempt01/independent_c_report.json` | `29fd7f393c8cefbbda542c8528c96cbaf061a3650158f46d22096fc2046517d5` |
| closure 封印 | 同目录 `independent_final_inventory.json` | `26592bcb34a547a4034047cb8d3518fabd286e64314bc3fd9e7e52868c27c12a` |
| closure review.log | 同目录 `review.log` | `b5e12b21f1ab01c9b06ff801dbf8777944686165fc7fbcc9da35cac5306c0a40` |

closure 报告为 accepted/actual_execution/independent_review=true、scope=`independently-accepted-actual-complete-C-training-and-evaluation`、
全 1333 文件 inventory 无排除项；closure supervisor 自身 receipt 记录 whole 222.73240518476814 s。

## 6. 限制（如实）

- 三个 seed、一个冬季区域、单次运行；描述性一致性，不是显著性，也不构成泛化证据。
- package 改善不能拆归因到某一 reader/query/过程语义；matched_generic 同结构同输入对照仍 resolved 为「不能区分」。
- 准确率—成本门未过；adaptive 不启动。K 数是推理轮次，不代表延迟优势。
- 指标含负值与 undefined（AC 对 climatology 的 normalize 语义），全部保留未过滤；tiny 负 climatology 符号不作科学退化判定。
- 可复现等级：identity-bound numerical replay，非 GPU 逐位一致。
- 本页只登记证据指针；数值重建属独立审阅职责，本页不重算。

## 7. 下一动作（仅提议）

- 本页不改 N4 goal 状态；M1 单因素补全与 UTC 统计见 `docs/R7_M1_ACTUAL_AND_UTC.md`。
- 若未来要评估过程语义的独立贡献，须另立「同输入信息量可分辨」的实验协议；当前证据不支持追认。
