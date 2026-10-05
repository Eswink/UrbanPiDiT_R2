# S3-D4：R-C lead-coverage 候选（two_step、FLOP 匹配）单因素筛选 — 注册负结果

**状态：已完成并登记（2026-10-06）；`scientific_claim: false`；test（2023）未读。**
本页对应 `docs/goals/s3-confirmation-baselines-and-candidate.md` 的 D4 交付物，
记录第一轮 R-C 候选（训练目标 lead 覆盖）的同数据筛选。**判定：worsened，候选停当轮。**

## 1. 结论摘要

- 预注册主格（t2m/full、6h/12h、每 seed 同号、候选更低才 supported）**判定 worsened**：
  三 seed 在 6h 与 12h 一致显示候选 RMSE 更高（6h +0.9202/+1.0247/+0.9263 K；
  12h +1.3219/+1.4587/+1.2365 K）。
- 守门预审（u10/v10/mslp 相对同数据 incumbent 的每 seed 相对 MSE 变化 ≤0）**不通过**：
  34 个失败 cell（6h/12h 全部 3 变量 × 3 seed 全正，24h 起部分为正）。
- **决定**：`candidate-stops-registered-negative`。不进入 D5 S4 冻结包；按 S3 brief §4.4
  回到独立开发侧，下一条假设独立预注册。
- 整轮 3050.1 s（planned 5400 / hard 10800，soft overrun 0.0），首个 spawn→末次 reap
  3001.2 s，保守 GPU-h **0.8472**；与 D3 合计本节点 **1.9460 GPU-h**（节点软预算 2.5 内）。
- 该结果**独立复现**了旧 B 阶段「rollout/l6+l12 目标相对同预算 L6 控制无增益」的负结论——
  但这是新实例、新预算分配（FLOP 匹配 200 vs 400）下的独立一轮，不是旧结论的引用。

## 2. 登记内容（运行前冻结）

| 项 | 值 |
| --- | --- |
| 协议文件 | `outputs/r7_s3_d4_rc_candidate_20261005_attempt01/protocol.json`，文件 SHA256 `f27e600dd49277324535cdb7bc308ad44c888f641297e6cfa06fa3aaaa80498b`，内部 `protocol_sha256 c078f80432416e236a1eb4e560c48e19f262055f82042df912ff2aaf8b6864af`（训练前 `'x'` 写入） |
| 结果 / attempt | `result.json` SHA256 `152f6a3c2717d9924a8abf63620c1ad4d0f955f644a54b953700798537efbcd1`；`attempt.json` SHA256 `bff10e5a8561710bad4a26dcc0db4a8c95c488642a754c0923326941138eefce` |
| 代码 | commit `c46981e139b820d35f14d183723fc7be859b7abc`（protocol 内 `code.commit`，含 dirty 记录） |
| 机械差异（单因素） | 训练目标：control `l6`（400 updates）× vs 候选 `two_step`（l6 + 0.5·l12，200 updates）；其余同 spec、同初始化、同数据、同评估路径 |
| FLOP 匹配依据 | v2 probe 实测 `two_step/l6` forward+backward 比 **2.0018**；候选 200 更新 ≈ 控制 400 更新（相对差 0.09%） |
| 对照 | D3 注册 run 的三 seed incumbent（RMSE CSV 以 SHA256 pin 进本协议，读时逐 lead 复核） |
| GPU | GPU0 `GPU-408ad137-…`（冻结时空闲）；共驻、只读余量门、不 signal 邻居 |
| 判据文本 | 见协议 `decision.primary.text` / `decision.gate_pre_screen.text`（本页 §1 同义复述，以协议为准） |

## 3. 主格读数（t2m/full，RMSE 增量 = 候选 − incumbent，K）

| seed | 6h | 12h | 24h（次要） | 48h（次要） | 72h（次要） |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | **+0.9202** | **+1.3219** | −0.5335 | −0.5984 | −0.4558 |
| 42 | **+1.0247** | **+1.4587** | +0.0949 | +0.2622 | +0.2460 |
| 43 | **+0.9263** | **+1.2365** | +0.1751 | +0.1275 | +0.1125 |

主格判定：6h、12h 均全 seed 同号为正 → **worsened**（不支持）。
次要读数 24–72h 符号跨 seed 不一致（seed41 三格为负、42/43 为正），如实报告、不并入判定。

## 4. 守门预审（u10/v10/mslp，相对 MSE 变化 ≤0.0，5 lead × 3 seed）

- **不通过**：34 个失败 cell。分布：6h 9 个（3 变量×3 seed 全正）、12h 9 个、24h 8 个、
  48h 5 个、72h 3 个。
- 最严重：6h mslp seed41 +0.3024，12h mslp seed41 +0.2636，v10 6h 三 seed +0.080…+0.094。
- 按预注册规则：即使主格 supported，守门不通过也不能进 S4 冻结；本轮两者皆不通过。

## 5. 训练与评估执行

| seed | 训练 s | final loss | l6 | l12 | checkpoint SHA256（前 16） | 5 lead 评估 s（6/12/24/48/72） |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 41 | 93.6 | 0.20264 | 0.1320 | 0.1413 | `32ca5340567ba123` | 74.1/95.9/127.9/213.7/288.6 |
| 42 | 93.5 | 0.18605 | 0.0970 | 0.1782 | `f436af2c4c68441d` | 73.1/98.4/131.3/209.5/289.0 |
| 43 | 91.4 | 0.29339 | 0.1845 | 0.2178 | `36ae1807e0f8a400` | 73.2/96.7/127.9/209.4/286.5 |

- 三 seed 初始化均先经 CPU 逐位断言（与归档 actual C pairing 一致），与 D3 同一机制。
- 候选 objective 监督 `+12h` 步（报告 `l12` 非空断言通过），训练窗口 468、cohort 与 D2/D3 相同
  （472/468/460/444/428），climatology 只 fit 2017。

## 6. 成本

| 项 | 值 |
| --- | --- |
| 整轮墙钟 | 3050.1 s（planned 5400 / hard 10800）；soft overrun 0.0 |
| 首个 spawn→末次 reap | 3001.2 s；保守 GPU-h 0.8472 |
| 网络 / test | 0 请求；test 未读（`test_read: false`） |
| 失败 | 无；三 seed 全部完成。FLOP 探针（CPU）在协议冻结前执行，属整轮计时 |

## 7. confirmed / 推测

**已确认（事实）**

1. 在 v2 实例、同数据同 cohort 下，two_step（200 更新、FLOP 匹配）相对 l6 incumbent（400 更新）
   在 t2m 6h/12h 全面 worse（三 seed 同号，约 +0.92…+1.46 K）。
2. 守门三变量相对 incumbent 的 MSE 在多数 cell 上升（34 个正 cell）。
3. 与旧 B 阶段不同预算下的负结论在本实例上独立重现——**"扩展训练目标到 +12h"这条 R-C 杠杆
   在两个数据实例上均无正增益**。

**推测（待验证）**

- 候选在 6/12h worse 而 24h 起接近或局部更低，可能是两目标加权（λ=0.5）把容量从近端 lead
  抽出、近端退化、远端微改善——但这需要专门实验（如 λ 扫描或分离近端/远端头）才能区分，
  不在本轮范围。
- S0 的第二条事实（400 更新末段 loss 仍在降）尚未被本轮检验：本轮的 update 预算对照是
  FLOP 匹配而非"训练更久"，下一独立假设可用 (l6, 800 更新) 直接检验"是否停早了"。

## 8. limitations

- 单实例（2017 train / 2022 val）、3 seed；描述性一致，不是显著性。
- FLOP 匹配基于单个 probe batch 的实测比；训练全程实际墙钟/FLOPs 未逐 step 计量。
- K4 推理探针；GPU 共驻邻居（10.4 GiB 常驻）影响墙钟，不 signal、不剔除。
- 候选停当轮不产生 S4 候选；负结果按规则登记，不得改名重试。

## 9. 证据指针

- 运行根：`outputs/r7_s3_d4_rc_candidate_20261005_attempt01/`（protocol/result/attempt + 三 seed 训练/评估/配对表）
- 驱动与测试：`scripts/study_r7_s3_d4_rc_candidate.py`、`tests/test_r7_s3_d4_study.py`（commit `c46981e` 前后多次小修）
- 对照：`docs/R7_S3_D3_INCUMBENT.md`（pin 于本协议的 `arms.control.pins`）
- B 阶段历史：`docs/R7_74_AUTOREGRESSIVE_ATTEMPT.md`（旧 dev store 的负结论，独立于本页）
