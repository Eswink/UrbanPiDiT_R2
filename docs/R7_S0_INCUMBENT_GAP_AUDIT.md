# R7 主模型 S0：actual C incumbent 身份复核与 D1 公平差距表

<!-- round-node: S0 -->

- 登记日期：2026-10-05；`scientific_claim: false`、`actual_pass: false`、`test_read: false`。
- 本文是 `docs/goals/main-model-climatology-campaign.md` §2 D1 的交付物，对应节点 S0。
- **性质**：只读身份复核 + 零训练差距算术。不是科学胜出证据，也不把 actual C 的
  package 相对 old_ours 改善追认为超过气候态。

## 1. incumbent 身份（不可变 pins，逐项独立复核）

| 项 | 值 | 复核方式 |
| --- | --- | --- |
| 归档根（排他） | `outputs/r7_v2_comparison_20261004_attempt01` | 目录存在，`stage-sealed` |
| 冻 protocol SHA256 | `ea0efb80e1b11eb7813fae74b4ee04ca67d145d16519eb8aa442af8d8851ad80` | 逐字段重算 canonical digest，一致 |
| artifact manifest `files_digest` | `bb4569f6681e1bdec08e22271cd048acf015be89b61176cc46b7203295feaa1c` | 对 1330 条 `files_sha256` 重算，一致 |
| `model_code_sha256` | `551261c4a501a6f9727fef2a712b40ab67ec51adaa460298a6b5b59bf8ae5dc1` | 用当前工作树 `model/` 重算，一致 |
| 代码提交（含 dirty 记录） | `562e526afc5fdbdc99053a25ab20036b66a2a8bc` | `code_commit.txt` + 归档 `code.zip` |
| `data_identity`（train） | `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07` | 对 `manifests/train.jsonl` 重算，一致 |
| `val_data_identity` | `6ee286c7eb5c54525e2466719a75a0c58e2459d1639a087d04205992dbf3ab16` | 对 `manifests/val.jsonl` 重算，一致 |
| `source_sha256` | `496084a9260bacfaf6293a01d89439c1e49d6afa8f09bc1f51d89a1d1f9bda21` | 对 `outputs/r7_m2_segment/source.nc`（37,734,176 B）重算，一致 |
| sidecar identity | `4fed1c78e4c4c09a41d95457649925b02d0c8ebf89aa47c2ac8dc6496734912d` | 归档 protocol 记录 |
| 九个 endpoint checkpoint | `seed{41,42,43}/{old_ours,process,matched_generic}/update_0000400.pt` | 逐个独立 SHA256，与 worker receipt 全部一致 |

incumbent 选定：**`process` 臂（V2 package）、K4、400 次 L6 更新**。`old_ours` 是前代
（`solver_gate_proposal/solver_state_recurrence=true`，无 conscious 输入面）；
`matched_generic` 是同信息等价控制，**不是**必须击败的竞争对手。三臂都从 fresh scratch
anchor、同 seed 同样本序训练，`mode='l6'`（`lambda12=0`），full BPTT，lr 1e-4。

## 2. 数据与案例身份

- 单 store `outputs/r7_m2_segment/store/cache.zarr`，zarr v3，shape `[240,17,65,65]`，0.25°。
- 17 变量冻结顺序：t2m, u10, v10, mslp, z850, t850, q850, u850, v850, z500, t500, q500,
  u500, v500, z250, u250, v250；单位见归档 protocol `data.units`（K / m s⁻¹ / Pa / m²s⁻² / kg kg⁻¹）。
- 半开 split（time_ranges 模式）：train `[2016-01-01T00, 2016-02-17T00)`（186 输入窗，185 可用）、
  val `[2016-02-17T00, 2016-02-23T00)`、test `[2016-02-23T00, 2016-03-01T00)`（未读）。
- 本次评估只用 val；逐 lead 完整 cohort 病例数 22/21/19/15/11（6/12/24/48/72h），**无 cohort 收窄**。
- 基线与被评模型共用同一初始化病例、变量、网格、mask、物理单位与区域权重。

## 3. 复核方法与自证

工具 `tools/recompute_r7_s0_gap_audit.py`（纯标准库，离线、零 GPU），测试
`tests/test_recompute_r7_s0_gap_audit.py`（14 例：身份链、逐条反证、CLI 排他写）。审计：

1. 重算 protocol/manifest/model-code/source/checkpoint 五类身份；
2. 从 135 个 worker 的 `per_case_metrics.csv` 与 `baseline_per_case_metrics.csv` 独立重算
   **全部 6,885 个 pooled cell**（3 臂 × 3 K × 5 lead × 3 区域 × 17 变量 × 3 seed）；
3. 与归档 pooled 行比较：最大相对偏差 **6.63e-16**（float64 求和次序），远低于 1e-9 界限。

零训练、零网络、零 GPU。运行成本见 §6。

## 4. 主要读数（描述性，不作显著性）

Train-only 气候态 = 该 store train 段逐格 `(month,hour)` 均值（M2 上 8 桶：01/02 月 × 00/06/12/18 UTC），
逐网格、按 valid-time 取桶，缺桶 fail-closed。persistence = 最后合法历史帧复制。

### 4.1 t2m、K4、region full、单位 K（incumbent = process）

`skill = 1 − MSE_model / MSE_climatology`；每 seed 独立

| seed | 6h | 12h | 24h | 48h | 72h |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | **+0.2578** | −0.0831 | −0.2655 | −1.4991 | −1.9876 |
| 42 | **+0.1715** | −0.3060 | −0.7333 | −3.1511 | −5.3649 |
| 43 | **+0.1924** | −0.1843 | −0.2752 | −1.7720 | −2.8657 |
| seed 均值 | **+0.2073** | −0.1911 | −0.4246 | −2.1407 | −3.4061 |

**只有 6h/全三 seed 为正；12h 起全部为负，且随 lead 单调恶化。** 对照 old_ours 同配置
（6h：−1.058/−0.879/−0.923）可见 package 相对前代确有大幅改善，但**相对气候态仍然只在 6h 获胜**。
两件事在此分开陈述：package-vs-old_ours 是性能改善；model-vs-climatology 是科学门，尚未过。

### 4.2 17 变量 × 5 lead（K4、full、process，正号 seed 数 / 3）

| 变量 | 6h | 12h | 24h | 48h | 72h |
| --- | ---: | ---: | ---: | ---: | ---: |
| t2m | 3/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| u10 | 3/3 | 0/3 | 0/3 | 0/3 | 3/3 |
| v10 | 3/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| mslp | 3/3 | 3/3 | 0/3 | 0/3 | 0/3 |
| z850 | 3/3 | 3/3 | 0/3 | 0/3 | 3/3 |
| t850 | 3/3 | 3/3 | 0/3 | 0/3 | 0/3 |
| q850 | 3/3 | 3/3 | 0/3 | 0/3 | 0/3 |
| u850 | 3/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| v850 | 3/3 | 1/3 | 0/3 | 0/3 | 1/3 |
| z500 | 3/3 | 3/3 | 3/3 | 0/3 | 0/3 |
| t500 | 3/3 | 3/3 | 0/3 | 0/3 | 0/3 |
| q500 | 3/3 | 3/3 | 3/3 | 0/3 | 0/3 |
| u500 | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 |
| v500 | 3/3 | 3/3 | 0/3 | 0/3 | 0/3 |
| z250 | 3/3 | 3/3 | 2/3 | 0/3 | 0/3 |
| u250 | 3/3 | 3/3 | 3/3 | 2/3 | 2/3 |
| v250 | 3/3 | 3/3 | 0/3 | 0/3 | 0/3 |

**118/255** 个 seed-cell 为正；137/255 非正；0 个 undefined。6h 全部 17 变量全 seed 为正；
lead 增长后普遍转负。高空 u500/u250 是唯一在 48–72h 仍多数为正的变量组。

区域分解（K4/process）：full 正 118 / 非正 137；interior 正 116 / 非正 139；edge_2 正 121 / 非正 134——
**三区域结论一致，不存在靠区域挑赢家**。

### 4.3 守门变量（K4、full、process，seed 均值 skill vs 气候态）

| 变量 | 6h | 12h | 24h | 48h | 72h |
| --- | ---: | ---: | ---: | ---: | ---: |
| u10 | +0.277 | −0.221 | −0.397 | −0.277 | +0.051 |
| v10 | +0.440 | −0.047 | −0.399 | −0.433 | −0.154 |
| mslp | +0.811 | +0.414 | −0.639 | −1.465 | −0.588 |

按新合同，守门是「相对**同数据 incumbent** 的每 seed 相对 MSE 变化 ≤0」而非「相对气候态」；
本页只记录当前 incumbent 本身的水平，合同守门要在最终确认时对候选与同数据 incumbent 重算。

### 4.4 推理深度 K 的效应（t2m/full/process/seed 均值 skill vs 气候态）

| K | 6h | 12h | 24h | 48h | 72h |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | +0.058 | −0.332 | −0.534 | −2.206 | −3.164 |
| 2 | +0.147 | −0.240 | −0.457 | −2.069 | −3.078 |
| 4 | +0.207 | −0.191 | −0.425 | −2.141 | −3.406 |

K 增大在 6h 单调有益（+0.058→+0.207），12/24h 轻微有益，48/72h 无稳定收益——
**与「K 只是同一 K4 checkpoint 的推理深度探针、不是独立训练模型」一致，不能当算力比较**。

### 4.5 相对 persistence

t2m 在 6h/12h 大幅优于 persistence（+0.70/+0.73 seed 均值），24h 起转负；u10/v10 在多数 lead
优于 persistence。persistence 与 climatology 逐 lead 在所有 seed 上逐位相同（同病例同一参考），
说明参考构造无 seed 泄漏。

## 5. 差距诊断（事实与推测分开）

**已确认（事实）**

1. incumbent 只在 t2m 6h 稳定超过 train-only climatology；12–72h 全面落后，且随 lead 恶化。
2. 该结论在 full/interior/edge_2 三区域一致，不是区域 artifact。
3. 身份链完整可复核：protocol/manifest/model-code/checkpoint/data/source 六类 digest 全部独立重算一致。
4. 评估 6,885 cell 与归档 pooled 行独立重算一致（≤6.63e-16）。
5. 训练报告显示 400 次更新后 t2m loss 仍在下降（seed41/process 每 50-update 块：
   0.1882→0.1820→0.1651→0.1674→0.1539→0.1460→0.1498→0.1522），末段 gradient_norm 仍在 0.2–0.5。

**推测（待验证，不当作结论）**

- incumbent 的训练目标只有 **L6（物理 6h）** 深监督（`mode='l6'`、`lambda12=0`），
  却被 fold out 到 72h 自回归评估。长期 lead 落后气候态可能与「训练目标不含长 lead」有关，
  但这需要单独的可证伪实验（#78 R-C 的延长/配方轮）来区分于「容量不足」或「优化未收敛」。
- M2 气候态只有 8 桶（2 月 × 4 UTC），逐格均值极平滑；这与「越长的 lead 气候态越难以被超越」
  一同出现，但本页不把该相关当因果。
- 旧文档记录的「9.09×」等跨单位数值不进入本页；本表全部为同一物理单位下的同案例算术。

## 6. 执行、成本与失败处理

| 项 | 值 |
| --- | --- |
| 输出根 | `outputs/r7_s0_gap_audit_20261005_attempt01`（排他新路径） |
| protocol SHA256 | `3209fcd2a89f89f4151a3b7a02b6f16e7122c62f0e747865cb866f0bacd3feef` |
| 工具 SHA256 | 见 `audit_report.json` `script.sha256` |
| gap table | `gap_table.csv`，6,885 行，SHA256 `7201859a0cb27e5f4d61d43e6e0cfb910e2825ebb756458ab4be69d50158ffc9` |
| 审计报告 | `audit_report.json` |
| 实测墙钟 | 5.523 s（planned 600 / hard 1200；soft overrun 0.000） |
| GPU-h | 0.0000（无 CUDA 设备打开） |
| 网络 | 0 请求；工具仅导入标准库，禁网由进程内约束 + 无网络依赖保证 |
| 失败 | 无。若失败按冻结策略整轮停留、不原地重试 |

## 7. limitations（如实）

- 单冬季两个月（2016-01-01..2016-03-01）、单区域（27–43N/107–123E，65×65）、单 store；
  不能支撑跨年/四季结论。旧 test 段已曝光，不能重新封存。
- 三 seed 是描述性一致性，不是显著性；时间块 bootstrap 与新合同的同时区间尚未实现在此审计中。
- 「K1/K2/K4 是同一 K4 checkpoint 的推理深度探针」，不是独立训练模型，也不是算力比较。
- 逐格 pooled 独立重算只证明**算术与来源一致**，不重新证明上游模型前向或 GPU 结果。
- matched_generic 与 process 的 1e-5 K 级差异仍是 unresolved；本页不重新裁决过程语义归因。
- t2m 目标未达不代表全部 17 变量无技巧：u500/u250 在 48–72h 仍多数为正，如实保留。

## 8. 对下一节点（S1）的具体含义

1. **主线目标明确**：把 t2m 12–72h 的负 skill 扭正，同时不破坏 6h 的正 skill 与守门变量。
2. 三个候选机制方向（#76 的 P0→#77→#78 次序）在本页证据下按此优先级进入 S1/S2：
   - #78 R-C（训练配方/目标 lead 覆盖）与 #78 R-A（归一化解码）证据支持最强：训练曲线未收敛
     且训练目标不含评估 lead，是当前最大的可解释缺口；
   - #77（位置编码频带）是单因素、低成本、可解析验证的实现；
   - #79（有类型诊断进前向）保持独立设计，首轮 aux=0。
3. **数据需求**：要越过气候态必须同时重建同数据气候态；M2 的 8 桶冬季气候态太弱/太窄，
   需要按 D3 分批扩到全年季节与独立年份，并在新数据上重训气候态与必要的同数据 incumbent。
   旧 8 桶气候态不得与新数据混用。

## 9. 证据指针

- 审计工具：`tools/recompute_r7_s0_gap_audit.py`；测试：`tests/test_recompute_r7_s0_gap_audit.py`（14 passed）
- 运行产物：`outputs/r7_s0_gap_audit_20261005_attempt01/{protocol.json,audit_report.json,gap_table.csv,attempt.json}`
- 冻结归档：`outputs/r7_v2_comparison_20261004_attempt01/`（只读）
- 合同与判据：`docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`、`docs/goals/main-model-climatology-campaign.md`

## 10. 下一动作

进入主计划的 **S1**：按 `real-data-acquisition` 冻结第一批多季节/多年度真实数据的数字范围、
网络/decoded/磁盘预算与软硬秒数，先小 pilot 核真实速率与 schema，再自审 preflight 后向全新
`outputs/` 路径 `--write`；并行完成 #77 位置编码频带的解析与 CPU 阳性反证、#78 R-A/R-C 的
train-only 统计与接线、#79 三臂设计的可证伪接口测试。S2 起按 #76 的 P0→#77→#78 次序做单因素
真实 train/val 小试，性能改善与机制归因分开陈述，不等待逐项授权、不自宣 complete。
