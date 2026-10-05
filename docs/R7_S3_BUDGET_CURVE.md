# S3-BC：训练更新预算曲线（l6 1600 更新 vs 注册的 400 更新控制）— 主格 supported / 守门未过 17 cell，预算饱和于单年数据

**状态：已完成并登记（2026-10-06）；`scientific_claim: false`；test（2023）未读。**
本页是 S3-UB（800 更新）后的第二剂、也是 v2 实例上的最后一剂预算因子读数；
它回答的问题（预先声明在 protocol 的 `factor.budget_response_text`）：预算响应曲线是否继续改善、
长 lead 守门是稳定还是恶化。

## 1. 结论摘要

- 预注册主格（t2m/full、6h/12h、每 seed 同号、候选更低才 supported）**判定 supported**：
  6h −0.6427/−0.7454/−0.9084 K，12h −0.6195/−0.7667/−0.9339 K；24h 也全负；
  48h 两负一微负、72h 两负一正（seed42 +0.767）。
- 守门预审 **不通过且比 800 更新更差**：17 个正 cell（48h 8 个、72h 9 个；u10 6、v10 6、mslp 5），
  对比 800 更新的 13 个（且 800 时 13 个全在 48/72h，本次同样集中在长 lead）。
- **预算响应读数（预声明、非判据）**：t2m 增益继续改善（6h skill +0.59…+0.60、12h +0.325…+0.338、
  24h +0.209…+0.242，全部 51/51 正），但长 lead 的守门代价同步上升（13→17 cell，且 72h 出现
  第一个 seed 级 t2m 正增量）。**结论：在单年 2017 train 上，更新预算的边际收益是 t2m 越好、
  风场守门越差——预算饱和的证据落在数据而非算力上。** 下一能力投资是数据（batch-3 已在并行获取）。
- 训练 loss 到 1600 仍在降（最后四个 segment 均值约 0.113–0.121，低于 800 端点的 0.124–0.133），
  所以「loss 未收敛」不是停止线；停止线是守门合取规则。
- 按冻结规则 `registered-negative`（primary 与 gate 合取未过）→ 不进 D5/S4 冻结包。
- 整轮 4937.6 s（planned 5400 / hard 10800，soft overrun 0.0），首个 spawn→末次 reap
  4884.4 s，保守 GPU-h **1.3716**；与 D3+D4+UB 合计本方向 **6.5104 GPU-h**。

## 2. 登记内容（运行前冻结）

| 项 | 值 |
| --- | --- |
| 协议文件 | `outputs/r7_s3_budget_curve_20261006_attempt01/protocol.json`，文件 SHA256 `f50f328d211bf6526562922f7cff0228e19ca3e3b6bef6f684f54807a545b83c`，内部 `protocol_sha256 08eaf4768dd4fc06a2bed26080cf1f37ddced51e277799fa58e419561f0fa434`（训练前 `'x'` 写入） |
| 结果 / attempt | `result.json` SHA256 `4f00f2215409191fe44666a271a5feeb902ea859b04cf78115ddf1ee9937a35e`；`attempt.json` SHA256 `b12d7d1df93a0c2494618751dbfbc331f3164d2038269a896364d0a043bc2e51` |
| 代码 | commit `2559c1f` 之上冻结（protocol 内 `code.commit` 含 dirty 记录；驱动 commit `3995267`） |
| 机械差异（单因素） | 训练更新预算：control `l6` 400 updates × vs 候选 `l6` **1600 updates**；同 spec、同初始化、同数据、同评估路径、同 warmup 比例（5% 端点：80/1600） |
| 剂量声明 | 本协议 `factor.dose=2`、`prior_doses=[800 (S3-UB)]`、`budget_response_text` 明写「v2 实例上的**最后**一剂预算」，后续预算问题移到 batch-3 扩年实例 |
| 已知不对称 | 候选训练 FLOPs 与控制比 **4.0**（每 update 前反向同为 31,981,732,992 FLOPs，v2 probe 实测） |
| 对照 | D3 注册 run 的三 seed incumbent（RMSE CSV 以 SHA256 pin 进本协议，读时逐 lead 复核） |
| GPU | GPU0 `GPU-408ad137-…`；共驻、只读余量门、不 signal 邻居 |
| 判据文本 | 见协议 `decision.primary.text` / `decision.gate_pre_screen.text` / `decision.budget_response.text`（本页同义复述，以协议为准） |

## 3. 主格与全 lead 读数（t2m/full，RMSE 增量 = 候选 − incumbent，K）

| seed | 6h | 12h | 24h | 48h | 72h |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | **−0.9084** | **−0.9339** | −0.9161 | −0.9899 | −0.6043 |
| 42 | **−0.7454** | **−0.7667** | −0.5643 | −0.1904 | **+0.7670** |
| 43 | **−0.6427** | **−0.6195** | −0.3257 | −0.7223 | −0.6774 |

t2m 相对同数据气候态 skill（51 cell 计数）：6h 51/51 正（+0.5895…+0.6033）、12h 51/51 正
（+0.3250…+0.3384）、24h 51/51 正（+0.2087…+0.2418）、48h 7/51、72h 2/51。
预算曲线（D3 400 → UB 800 → BC 1600，seed 均值）：6h +0.272/+0.498/+0.595，
12h −0.077/+0.196/+0.333，24h −0.116/+0.120/+0.226——近三 lead 单调改善。

**训练 loss segment 均值（seed41/42/43 最后四段）**：0.1195–0.1150 / 0.1238–0.1166 /
0.1232–0.1102，均低于各自 800 端点段（0.1275 / 0.1321 / 0.1360）。

## 4. 守门预审（u10/v10/mslp，相对 MSE 变化 ≤0.0，5 lead × 3 seed = 45 cell）

- **不通过**：17 个正 cell（48h 8 个、72h 9 个），全部落在长 lead；6h/12h/24h 全数通过。
- 对比 800 更新轮的 13 个（48h 6、72h 7）：**守门代价随预算上升而增加**，
  这是本页把「预算饱和」判给数据侧的定量依据。
- 变量分布：u10 6 个、v10 6 个、mslp 5 个；72h 成为退化的主要集中区。

## 5. 训练与评估执行

| seed | 训练 s | final loss | checkpoint SHA256（前 16） |
| ---: | ---: | ---: | --- |
| 41 | 640.0 | 0.13849 | `d9453acafa1c067a` |
| 42 | 635.0 | 0.19431 | `cb5419aeeddc8319` |
| 43 | 659.1 | 0.20162 | `51b2d58c96ca2d9c` |

- 三 seed 初始化均先经 CPU 逐位断言（与归档 actual C pairing 一致）。
- l6 目标监督断言（`losses[*].l12` 全 None）、`updates_this_run=selected_update=1600` 断言通过；
  训练窗口 468、cohort 与 D2/D3/D4/UB 相同（472/468/460/444/428），climatology 只 fit 2017。

## 6. 成本

| 项 | 值 |
| --- | --- |
| 整轮墙钟 | 4937.6 s（planned 5400 / hard 10800）；soft overrun 0.0 |
| 首个 spawn→末次 reap | 4884.4 s；保守 GPU-h 1.3716 |
| 训练 FLOPs | 候选 3 seed × 1600 × 31,981,732,992 = 153.51 TFLOP（控制同口径 38.38 TFLOP） |
| 网络 / test | 0 请求；test 未读（`test_read: false`） |
| 失败 | 无；三 seed 全部完成。FLOP 探针（CPU）在协议冻结前执行，属整轮计时 |

## 7. confirmed / 推测

**已确认（事实）**

1. 400→800→1600 的预算曲线在 t2m 近三 lead（6/12/24h）上单调改善：6h skill 从 +0.272
   （400 更新 seed 均值）到 +0.595（1600），12h 从 −0.077 到 +0.333，24h 从 −0.116 到 +0.226。
2. 长 lead 的守门代价随预算上升：13 → 17 个正 cell，且 1600 时 72h 出现首个 seed 级
   t2m 正增量（seed42 +0.767）——继续加预算不能同时保住 t2m 长 lead 与风场守门。
3. 训练 loss 到 1600 仍在下降；「停早了」问题在 800 已经部分变现，1600 的额外收益
   集中在近端 lead，远端改善被守门吃掉。

**推测（待验证）**

- 在单年 2017 训练集上同时追求 t2m 全 lead 与风场守门的容量/数据不足；
  batch-3（2018–2021 四季，正在获取）把 train 扩到 5 年后，同样的 1600 预算预期
  应以更多有效样本（约 5×472 窗口）缓解长 lead 过拟合——这是下一轮的主假设。

## 8. limitations

- 单实例（2017 train / 2022 val）、3 seed；描述性一致，不是显著性。
- 候选刻意使用 4× 训练更新与 FLOPs；supported 是对「该预算的配方」而言，不是算力对等比较。
- GPU 共驻；latency 含邻居负载，不作基准。
- 本轮是 v2 单年实例上的**最后一剂**预算；进一步加预算（3200+）不再是本实例的合法问题，
  迁移到 batch-3 扩年实例后才重新有意义。

## 9. 证据指针

- 运行根：`outputs/r7_s3_budget_curve_20261006_attempt01/`
- 驱动与测试：`scripts/study_r7_s3_budget_curve.py`、`tests/test_r7_s3_budget_curve.py`
- 共享判读：`training/r7_verdict_readings.py`
- 对照与前序：`docs/R7_S3_D3_INCUMBENT.md`、`docs/R7_S3_D4_RC_CANDIDATE.md`、`docs/R7_S3_UB_UPDATE_BUDGET.md`
