# S3：主模型超气候态确认实例 v3（train 2017–2021 五年扩年 store）

**状态：数据实例已发布并登记（2026-10-06）；`scientific_claim: false`；test（2023）从未评分。**
本页对应 `docs/goals/main-model-climatology-campaign.md` 的 S3 节点与决策 0038/0039。
它记录一次**数据准备**，不产生任何模型超越气候态的结论。

## 1. 结论摘要

- 三个已验证获取批（S1 四季 2017 + batch-3 四季 2018–2021 + batch-2 四季 2022/2023）
  按精确时间戳合并为一个 **3360 时间点**源（0.25°、65×65、17 通道、四个 UTC init 小时），
  由 `prepare_r7_local.py` 在全新排他路径构建为 **train 2017–2021 / val 2022 / test 2023**
  的年度切分 store。combine 重跑探针验证字节确定（同 `bc2ff9cf…`）。
- 该实例是对 S3-UB/S3-BC 已登记预算读数的直接响应：单年 train 上预算饱和
  （1600 更新时 t2m 仍改善但 48/72h 守门 13→17 恶化），预声明读数判定「下一能力投资是
  数据」；v3 把 train 年份扩到 5 年（机械窗口 472→2360，×5.0）。
- 三个 train-only sidecar 全部发布并可由严格加载器读回：change-scale `f76c373d…`、
  process-scale `0e87fe40…`、typed-evidence `6f9bedbf…`，三者绑定同一
  `data_identity 2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac`。
- 成本：GPU 0、付费 0、网络 0（复用已获取批次）；全流程墙钟约 509 s
  （combine 41.1 + 只读 preflight 8.3 + write 75.7 + 三 sidecar 预检/发布
  66.1/63.1 + 51.1/52.7 + 75.3/75.8），远低于冻结预算 planned 1200 s / hard 3600 s。

## 2. 冻结与构建顺序

| 步 | 动作 | 产物与 digest |
| --- | --- | --- |
| 1 | combine（`data/download/multi_year_source.py`） | `source.nc` **540,856,239 字节**、SHA256 `bc2ff9cfadcce604fc243bb999b3c430d5164d17fcf1de201273716a5db065f8`（3360 stamps）；`combine_receipt.json` SHA256 `afa80b64ee99f3035f59b5bff0e011279c6fda199a7ae630075e35440c66cc83` |
| 2 | 冻结构建协议 | `store_build_protocol.json`（文件 SHA256 `efb5fca4d6c8eb6673ed358c0841753103aa3434f4fde98766426761b2377bfc`；canonical body `protocol_sha256 2151c91e79865541a5235854e077cc206e6865346e52ccdeb32e03a965d2461a`） |
| 3 | 只读 preflight | `preflight_store_report.json` SHA256 `9b8739eba896b93622d3e1434be316f1b3930afbec9508c2b98e1ad293312189`：3360×17×65×65、0.25°、train/val/test 窗口 2360/472/472、指纹 scope `full-local-file` |
| 4 | 执行者自审 | `store_build_review.json` SHA256 `47b0a07fa87a27cfd59178c27bb003c770974cad69571adbb908d99dfe99ffef`（14 项机械核对全过，`decision=accept to --write`） |
| 5 | `--write` | `store/cache.zarr`（716 MB）+ `store/manifests/`（`BUILD_COMPLETE.json`、`build_complete=true`、`schema_version=1`）；`write_report.json` SHA256 `1b914a216bb3a79e044c316749bfad45a2a27bcd5df8bd07cd5cbf9ce4bd58e8` |
| 6 | 三 sidecar | change-scale 预检（identity `7c852fb6…`）→发布 `f76c373d…`；process-scale 预检（identity `00f6d738…`）→发布 `0e87fe40…`；typed-evidence 预检（identity `f51e6f4e…`）→发布 `6f9bedbf…` |

构建顺序与 v2 相同：combine → 冻结协议 → 只读 preflight → 执行者自审 → `--write` →
sidecar。协议在 `--write` 之前冻结；`--write` 的 `source_preflight_sha256`
（`1b914a21…`，process-scale 身份链的一部分）与 `write_report.json` 文件 SHA256 一致。

## 3. 身份链（v3）

| 项 | 值 |
| --- | --- |
| 源 | `s3://earthmover-icechunk-era5/icechunkV2`，快照 `ZFKDHBCTBVHVXM3BQFV0`（匿名，CC-BY-4.0） |
| 输入批 1 | `outputs/r7_s1_seasons_2017/source.nc`，SHA256 `3b2c2dad4c8054ff735930a4539ec554ea420d4f98a1bd91bd43ff0f6729d8b5`（480 stamps） |
| 输入批 2 | `outputs/r7_s3_batch3_20182021/source.nc`，SHA256 `97d29bca0633cc55df81a97081733f5d6ad425d4562aac78165ef2f0044daef8`（1920 stamps） |
| 输入批 3 | `outputs/r7_s2_batch2_20222023/source.nc`，SHA256 `44a24ca0096c3802fefc92a5e089c7283812001cfcffd6f88b2e06187607ec62`（960 stamps） |
| 合并源 | `outputs/r7_s3_confirmation_train2017_2021_v3/source.nc`，SHA256 `bc2ff9cfadcce604fc243bb999b3c430d5164d17fcf1de201273716a5db065f8`（540,856,239 字节，3360 stamps，2017-01-01T00 → 2023-10-30T18） |
| combine 回执 | `combine_receipt.json` SHA256 `afa80b64…`（`status: combined-real-source`、`n_sources: 3`、`local_artifact.sha256` 与上一致；合并轴等于三源各自记录时间戳的精确有序并集：唯一、递增、四个 UTC init 小时） |
| store 构建协议 | `store_build_protocol.json`，canonical `protocol_sha256 2151c91e…`（文件 SHA256 `efb5fca4…`） |
| 预检报告 | `preflight_store_report.json` SHA256 `9b8739eb…`：指纹 scope `full-local-file`、`raw_state_GiB 0.899`、`scientific_training_certified: false` |
| 审阅回执 | `store_build_review.json` SHA256 `47b0a07f…` |
| 写入报告 | `write_report.json` SHA256 `1b914a21…`（即 process-scale 侧记录的 `source_preflight_sha256`） |
| store | `outputs/r7_s3_confirmation_train2017_2021_v3/store/cache.zarr` + `store/manifests/`（`train.jsonl` 2360 / `val.jsonl` 472 / `test.jsonl` 472 行） |
| data identity | `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac`（v3 train manifest 派生；三个 sidecar 与两个 sidecar 预检五处一致） |
| change-scale sidecar | `outputs/r7_s3_confirmation_train2017_2021_v3/change_scale/`，identity `f76c373da82073e627cb2bed9300b1e6146aafe9c4a15dec368e06e9d455d595`（2380 train pairs；17 通道全 active；ratio 0.0795–0.6865） |
| process-scale sidecar | `outputs/r7_s3_confirmation_train2017_2021_v3_process_scale/`，identity `0e87fe406764663c3cc59289bbc3c04a8e18adbc4d99a3e9cf59ac52cf8545a4`（2400 历史+目标帧；源身份完整 `full-local-file`） |
| typed-evidence sidecar | `outputs/r7_s3_confirmation_train2017_2021_v3_typed_evidence/`，identity `6f9bedbf79237c3724358377f49ec7a6d571bfd7e3b0a1a69ef4bc231008dbde`（2380 pairs；4 场 `divergence_850`/`vorticity_850`/`temperature_advection_850`/`static_stability_850_500` 全 active） |
| test 读取 | **否**：2023 的测试段只在 `write_report` 的 split 计数里出现，从未打开 `test.jsonl` 或对 2023 评分 |

combine 字节确定性：同输入重跑到 `/tmp` 探针，源 SHA256 同为 `bc2ff9cf…`（墙钟 41.1 s），
即 v3 源可由三输入批随时重放得到同一字节流。

## 4. 实例内容

- 时间轴：2017-01-01T00 → 2023-10-30T18（3360 个 6 小时点；每年四个月份块
  Jan/Apr/Jul/Oct 各 30 天 × 4 UTC 小时）。
- 切分：**年模式**，train `[2017, 2018, 2019, 2020, 2021]` / val `[2022]` / test `[2023]`
  互相不相交，窗口不跨年边界；训练段覆盖 val/test 的每一个 (month,hour) 桶，
  fail-closed 的 train-only 气候态可完整评分。
- 每 lead 窗口（机械，manifest 实测）：6h train 2360 / val 472 / test 472；
  其余 lead 的重叠裁剪数在评估时按既有机制逐一报告（v2 的 val 逐 lead
  472/468/460/444/428 由同一代码路径给出，batch-2 源未变）。
- 基线纪律：climatology（train-only (month,hour) 逐格均值，在本实例 5 年 train 上重新 fit）
  与 persistence 必须在本实例上**重建**，不得沿用 v2/2017 dev store 的数值；
  400 更新 incumbent 控制同样重训（D3-analog），不做数值搬运。

## 5. 成本与预算对表

| 项 | 值 |
| --- | --- |
| GPU-h | 0.0000（v3 全流程无 CUDA 设备打开） |
| 网络 | 0 字节（复用已获取批次，未新增下载） |
| 磁盘 | 新占用约 1.26 GB（源 541 MB + store 716 MB + 三 sidecar < 1 MB）；`/data` 余量 571 GB |
| 墙钟 | combine 41.1 s + 只读 preflight 8.3 s + write 75.7 s + change-scale 预检/发布 66.1/63.1 s + process-scale 51.1/52.7 s + typed-evidence 75.3/75.8 s ≈ 509 s |
| 冻结预算 | planned 1200 s / hard 3600 s / raw cap 3.0 GiB（`raw_state_GiB` 实测 0.899） |
| 失败 | 无；v3 是新增构建，不取代 v2（v2 作为 2017 单年 train 实例保留，UB/BC 读数绑定其上） |

## 6. 与 v2 的关系与不回改声明

- v2（`outputs/r7_s3_confirmation_2017_2022_2023_v2/`）是 train 2017/val 2022/test 2023 的
  单年 train 实例，S3-D2/D3/D4/UB/BC 全部读数绑定其上，**不被 v3 取代或修改**；v3 是
  独立的扩年实例（train 五年的新 data identity `2564eeaf…`），val/test 源相同但重建成新 store。
- 任何跨实例比较必须显式声明实例根与 data identity；v2 与 v3 的相同数值不得互相搬运
  （基线/控制在各自实例内重建）。

## 7. 已确认与推测

**已确认**：合并轴等于三输入各自记录时间戳的精确有序并集（唯一、递增、四个 UTC 小时）；
17 通道单位（K/Pa/m s⁻¹/kg kg⁻¹/m² s⁻²）与 levels 经只读 preflight 复核；0.25° 网格 65×65；
manifest 窗口计数 2360/472/472（6h）；源指纹为完整全文 SHA256 且与 combine 回执一致；
store 与三个 sidecar 均由 BUILD_COMPLETE + data identity `2564eeaf…` 绑定；combine 重放
字节确定；test 未读。

**推测（未证实）**：五倍 train 数据会抬高 train-only 气候态基线（5 年 (month,hour) 均值
比单年稳定，可能压低 6h 的既有正 skill），同时可能改善长 lead 的 pattern skill（ρ），
缓解 48/72h 守门退化。v3 上重建的控制与预算读数将检验这一点，本页不预写结果。

## 8. limitations（如实）

- 单 ROI（27–43N/107–123E）、0.25°、17 通道；季节块是各 30 天样本，不是整季或全年。
- 构建是准备动作，**不产生任何模型结果或科学声明**；`scientific_claim: false` 保留。
- 指纹审计证明的是源字节身份，**不**认证源本身适合科学训练。
- v2 与 v3 两棵实例树同时存在；正式运行必须显式指向各自根，混用会被身份校验拒绝。
- 训练窗口从 472 扩到 2360 只改变数据量，不改变窗口构造机制；「更多数据更好」仍是
  待检验假设，本页不做任何推断性结论。

## 9. 证据指针

- 合并源与回执：`outputs/r7_s3_confirmation_train2017_2021_v3/source.nc`、`combine_receipt.json`
- 构建协议 / 预检 / 自审 / 写入报告：同根下 `store_build_protocol.json`、
  `preflight_store_report.json`、`store_build_review.json`、`write_report.json`
- store：`outputs/r7_s3_confirmation_train2017_2021_v3/store/`（BUILD_COMPLETE + manifests）
- sidecar：`outputs/r7_s3_confirmation_train2017_2021_v3/change_scale/`、
  `outputs/r7_s3_confirmation_train2017_2021_v3_process_scale/`、
  `outputs/r7_s3_confirmation_train2017_2021_v3_typed_evidence/`
- 配置：`configs/r7_s3_confirmation_v3.yaml`（`053570e` 起在版本控制内）
- 索引记录：`record:s3-confirmation-instance-v3`（`docs/R7_EVIDENCE_INDEX.jsonl`）
- 上游：`docs/R7_S3_BATCH3_ACQUISITION.md`（batch-3 来源批次）、
  `docs/R7_S3_CONFIRMATION_INSTANCE.md`（v2 实例与决策 0039）

## 10. 下一步

1. v3 上重建同数据 climatology 与 persistence（D2-analog，CPU 零 GPU），逐 lead 全覆盖。
2. v3 上重训 400 更新 incumbent 控制（D3-analog，三 seed 逐位初始化保真），评 val 全 cohort。
3. 在 v3 上跑预算剂量读数（单因素，1600 或按 5× 数据量重定标），检验「数据扩年提升
   pattern skill」假设；通过 primary+gate 合取才进 D5 S4 冻结包。
4. test（2023）保持封印，直到预注册的未见年份确认轮。
