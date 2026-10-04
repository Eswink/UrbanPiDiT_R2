# R7-V2：新工程包 GPU 精度探针的独立接受（metadata-only）

- 登记日期：2026-10-04；`scientific_claim: false`、`actual_pass: false`（本页只登记工程精度探针及其独立接受）。
- **已确认**：新工程包（base `562e526afc5fdbdc99053a25ab20036b66a2a8bc`）的 FP32/BF16 精度探针实跑 8 次优化更新
  （两精度各 4）、4 次 checkpoint decode，并由独立审阅接受为
  `accepted-actual-new-package-precision-metadata-only`。
- **未接受**：不是预报收益、收敛或 SOTA；GPU forward 未被独立重算（接受文件自己如此限定）。

## 1. 生产者（实际执行）

| 项 | 值 |
| --- | --- |
| 输出根 | `outputs/r7_v2_package_precision_probe_20261004_attempt01/` |
| producer protocol | `edc83fb64df15ef9b1925758a029251c25dbc213b3ebca3e1b9b87b0a9c46f68` |
| attempt 状态 | `success`，`finalized: true`，`budget_limited: false` |
| attempt SHA256 | `f889db3499149b0ee09d6efb63eb2c7f5c1006a183a57e3f67bae205fc111f6a` |
| closeout SHA256 | `62b19de5aecde60e7ea13c813948b6239089ae07bb6b577758d9d814a6cf320f` |
| 实际更新 | FP32 4 + BF16 4 = 8 |
| checkpoint decode | 4 |
| 成本 | whole 273.1497834455222 s；GPU phase 195.99174571223557 s = **0.0544421515867321 GPU-h**；soft overrun 0 |

## 2. 独立接受（meta层）

| 项 | 值 |
| --- | --- |
| 接受文件 | `/tmp/r7_v2_precision_live_review_fix02_20261004_attempt05/acceptance.json` |
| 接受文件 SHA256 | `8503adfe30d66dbb8aaf5ff6d41b8a5c5fe10d5d01c92012a80f940aa3ab6e6b` |
| 格式 | `accepted-actual-new-package-precision-metadata-only`，`accepted/independent_accepted/actual_probed/actual_precision_accepted = true` |
| 接受协议 SHA256 | `eb90c7725ba90084cd7e60e32fc18275b6577e077aa9ceaf1633b4f711587f0b` |
| binding SHA256 | `246dde6ca40bc5742c9416c1e99990b62cab8ae020e3b318a6418fd612bc92c9` |
| model_code_sha256 | `551261c4a501a6f9727fef2a712b40ab67ec51adaa460298a6b5b59bf8ae5dc1` |
| source_tree_sha256 | `3c767f6b2b273c62d52b28d4c26079f4fd9956821a3520919a893723f30dfabd` |
| data / source identity | `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07` / `496084a9260bacfaf6293a01d89439c1e49d6afa8f09bc1f51d89a1d1f9bda21` |

接受范围（已确认，逐字限定）：weights/optimizer/RNG/losses resume 严格相等；L12-only 第一步与
`process_reader` 梯度 finite>0；poison/calendar 断言通过；既有 contract 修复复验。

## 3. 限制（如实）

- 独立接受是 metadata/source/inventory 校验与 CPU 端点对比；**GPU forward 未独立重算**，
  记录的后更新诊断不是更新前 GPU 预报证据。
- 不构成 forecast-skill、收敛、B/C 天气接受或任何科学声明。
- 该 0.0544421515867321 GPU-h 与旧 precision 0.04711594580465721 是不同对象，各记一次。
- CI green 绑定是本地提供的证据，不是从 GitHub 独立拉取。

## 4. 与 C/M1 的关系

本探针是 actual C 执行前的工程精度前置之一（D1 门）的独立接受；它的完成登记不是 C 的必要条件
（C 有自己的冻结协议与独立审阅），但补齐了“新包在 FP32/BF16/resume/poison 层面的精度契约”这一
带证据的完成项。C/M1 的实测见 `docs/R7_C_ACTUAL_CONFIRMATION.md` 与 `docs/R7_M1_ACTUAL_AND_UTC.md`。
