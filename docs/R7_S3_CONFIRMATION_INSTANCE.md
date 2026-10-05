# S3：主模型超气候态确认实例（2017/2022/2023 三年度 store，v2 修正构建）

**状态：数据实例已发布并登记（2026-10-05）；`scientific_claim: false`；test（2023）从未评分。**
本页对应 `docs/goals/main-model-climatology-campaign.md` 的 S3 节点与决策 0038/0039。
它记录一次**数据准备与身份修正**，不产生任何模型超越气候态的结论。

## 1. 结论摘要

- 两个已验证获取批（S1 四季 2017 + S3 batch-2 四季 2022/2023）按精确时间戳合并为一个
  1440 时间点源（0.25°、65×65、17 通道、四个 UTC init 小时），由 `prepare_r7_local.py`
  在全新排他路径构建为 **train 2017 / val 2022 / test 2023** 的年度切分 store。
- 修正构建 v2（`outputs/r7_s3_confirmation_2017_2022_2023_v2/`）与 v1 的**天气字节逐位相同**
  （9 个 zarr 数组 SHA256 全等；唯一差异是 `source` 属性里的路径），差别只在 preflight 审计：
  v1 的源指纹被旧 64 MiB 上限降级为 `file-stat-only`，v2 携带完整 `full-local-file`
  SHA256 `e0b51616a7c31f29b9832221e74522f8b5cad4b36b72b0d7586d140787e3b42d`。
- 修正依据为**决策 0039**：`source_fingerprint()` 删除 `max_hash_bytes` 与 stat-only 降级分支，
  常规文件总是全文哈希。触发事实是一次真实发布缺陷（见 §2），不是风格偏好。
- v2 上三个 train-only sidecar 全部发布并可由严格加载器读回：change-scale `6fd9e774…`、
  process-scale `b2830014…`、typed-evidence `8760894a…`，三者绑定同一
  `data_identity e01828e9988ba8baed8d29c7e1c2a06553700e6fe3ee6f56f6a22e31bad76a0e`。
- 成本：GPU 0、付费 0、网络 0（复用已获取批次）；v2 重建墙钟约 97 s（combine 24 s +
  preflight 7 s + write 37 s + 三 sidecar 发布 17/17/20 s），远低于冻结预算
  planned 1200 s / hard 3600 s。

## 2. 缺陷与修正（真实失败保留）

v1 store（`outputs/r7_s3_confirmation_2017_2022_2023/`）建成后发布 process-scale sidecar 时，
只读 preflight 在 `data/preprocess/r7_process_scale_sidecar.py:260` `_source_identity` 抛出
`ValueError: source bytes differ from audited full-file fingerprint`：

1. `data/preprocess/r7_preflight.py` 的 `source_fingerprint()` 自 #30 轮起对 >64 MiB 的文件
   只记 `file-stat-only`（bytes/mtime，`sha256: null`）；
2. 合并源 231,865,649 字节（221 MiB）超过该上限，v1 的 `source_preflight.json` 因此只有弱指纹；
3. process-scale sidecar（以及复用同一要求的 M3/V2 身份链
   `training/r7_m3_identity.source_identity`）只接受 `scope == 'full-local-file'` 且非空
   `sha256`，所以**任何源超过 64 MiB 的实例都结构上无法发布该 sidecar**；
4. 该 sidecar 是 S4 incumbent（actual C `process` 臂）训练契约的必需输入
   （`training/r7_process_training_contract.py` 的 `supervision_training_contract`），
   因此这个缺陷直接挡住正式确认；S1 的 77,177,840 字节源同样因此从未有 process-scale sidecar。

据决策 0039 的修法是**完整哈希成为默认且唯一形态**（更强身份，不是放宽）：删除
`max_hash_bytes=64*2**20` 参数与 stat-only 分支，常规文件总是全文 SHA256；目录的
`zarr-root-metadata-only` 行为不变。回归测试
`tests/test_r7_preflight.py::test_fingerprint_is_complete_beyond_the_old_stat_only_bound`
钉住"超过旧阈值必须得到完整指纹且输入不被修改"。读取方的接受判据（只收
`full-local-file`）**未改**——改的是生产方不再产出弱指纹。

v1 的审计是 write-once，**不回改、不编辑、不删除**：修正走全新排他路径 v2，v1 作为
被记录的缺陷构建原样保留（其 change-scale sidecar 与任何消费由 v2 取代）。
combin 步骤经 `/tmp` 探针验证为字节确定（重跑得到同一 `e0b51616…`），所以 v2 的源
与 v1 逐位相同，只换审计。

## 3. 身份链（v2）

| 项 | 值 |
| --- | --- |
| 源 | `s3://earthmover-icechunk-era5/icechunkV2`，快照 `ZFKDHBCTBVHVXM3BQFV0`（匿名，CC-BY-4.0） |
| 输入批 1 | `outputs/r7_s1_seasons_2017/source.nc`，SHA256 `3b2c2dad4c8054ff735930a4539ec554ea420d4f98a1bd91bd43ff0f6729d8b5`（480 stamps） |
| 输入批 2 | `outputs/r7_s2_batch2_20222023/source.nc`，SHA256 `44a24ca0096c3802fefc92a5e089c7283812001cfcffd6f88b2e06187607ec62`（960 stamps） |
| 合并源 | `outputs/r7_s3_confirmation_2017_2022_2023_v2/source.nc`，SHA256 `e0b51616a7c31f29b9832221e74522f8b5cad4b36b72b0d7586d140787e3b42d`（231,865,649 字节） |
| combine 回执 | `combine_receipt.json` SHA256 `400ba6f5e75b2eb2a91046582884756f66254d1e668c62a9fbb2552ee9f01a46` |
| store 构建协议 | `store_build_protocol.json`，canonical `protocol_sha256 f4160989b309e29b5785ca90d9c1d131bc4b0ad21109e84c4b08daadb67499b0`（文件 SHA256 `62a6e18c…`） |
| 预检报告 | `preflight_store_report.json`，SHA256 `f536599b56789afc48e92d820faa8d6f997c60ac941509050196efe61edfe00a`：1440×17×65×65、0.25°、train/val/test 窗口 472/472/472、指纹 scope `full-local-file` |
| 审阅回执 | `store_build_review.json`（执行者自审，14 项机械核对全过，`decision=accept to --write`） |
| 写入报告 | `write_report.json`，SHA256 `744bf7fb65862204c2fff929b65bd3cf0da4ed8f781b2f852901dfca8a1bb0ce`（即 process-scale 侧记录的 `source_preflight_sha256`） |
| store | `outputs/r7_s3_confirmation_2017_2022_2023_v2/store/cache.zarr` + `manifests/`（`BUILD_COMPLETE.json`、`build_complete=true`、`schema_version=1`） |
| data identity | `e01828e9988ba8baed8d29c7e1c2a06553700e6fe3ee6f56f6a22e31bad76a0e`（v2 train manifest 派生） |
| change-scale sidecar | `outputs/r7_s3_confirmation_2017_2022_2023_v2/change_scale/`，identity `6fd9e77456de6378b2c3116a16614541d771c81c2e24871e267c178cce311465`（476 train pairs；ratio 0.0756–0.7279，17 通道全 active） |
| process-scale sidecar | `outputs/r7_s3_confirmation_2017_2022_2023_v2_process_scale/`，identity `b28300145a2d9785099fdce1358b9f142470bddb54683eb092c2e3e531d7cd7e`（480 帧；8 通道全 active；源身份完整） |
| typed-evidence sidecar | `outputs/r7_s3_confirmation_2017_2022_2023_v2_typed_evidence/`，identity `8760894aa4c83ba38761111d044e37518d5b4d1a82860bd443a9b0cb25bc4495`（476 pairs；4 场全 active） |
| test 读取 | **否**：测试段只在 `write_report` 的 split 计数里出现，从未打开 `test.jsonl` 或对 2023 评分 |

v1 与 v2 的数组逐位一致（`state`、`process_diagnostics_raw`、`time_ns`、`latitude`、`longitude`、
`normalization_mean/std`、`process_normalization_mean/std` 九个数组 SHA256 全等），
唯一属性差异是 `source` 路径字符串。

## 4. 实例内容（与 v1 相同）

- 时间轴：2017-01-01T00 → 2023-10-30T18（1440 个 6 小时点；每年四个月份块
  Jan/Apr/Jul/Oct 各 30 天 × 4 UTC 小时）。
- 切分：**年模式**，train `[2017]` / val `[2022]` / test `[2023]` 互相不相交，窗口不跨年边界；
  训练段覆盖 val/test 的每一个 (month,hour) 桶，fail-closed 的 train-only 气候态可完整评分。
- 每 lead 机制窗口数（`store_build_protocol.json` 记录）：6h 472、12h 468、24h 460、48h 444、
  72h 428。
- 基线纪律：climatology（train-only (month,hour) 逐格均值）与 persistence 必须在本实例上
  **重建**，不得沿用 2017 dev store 或 v1 的数值。

## 5. 成本与预算对表

| 项 | 值 |
| --- | --- |
| GPU-h | 0.0000（v2 全流程无 CUDA 设备打开） |
| 网络 | 0 字节（复用已获取批次，未新增下载） |
| 磁盘 | v2 新占用约 560 MB（源 222 MB + store 308 MB + 三 sidecar < 1 MB）；`/data` 余量 594 GB |
| v2 墙钟 | combine 24.0 s + preflight 7.0 s + write 37.5 s + change-scale 发布、process-scale 预检 16.6 s/发布、typed-evidence 预检 20.5 s/发布 ≈ 123 s |
| 冻结预算 | planned 1200 s / hard 3600 s / raw cap 3.0 GiB（`raw_state_GiB` 实测 0.3853） |
| 失败 | 无新增失败；v1 缺陷构建与 v2 修正构建都保留（§2）。process-scale 首次发布尝试因
  destination 规则拒绝在实例树内嵌套（`sidecar must be disjoint from all existing input artifact
  trees`），零写入零残留，改放同层兄弟路径后成功——该拒绝是设计行为，不是失败。 |

## 6. 已确认与推测

**已确认**：合并轴等于两输入各自记录时间戳的精确有序并集（唯一、递增、四个 UTC 小时）；17 通道
单位（K/Pa/m s⁻¹/kg kg⁻¹/m² s⁻²）；0.25° 网格 65×65；三抽头窗口计数 472/468/460/444/428；
v2 源指纹为完整全文 SHA256 且与 combine 回执一致；store 数组与 v1 逐位相同；
三 sidecar 由各自严格加载器读回且绑定同一 data identity；test 未读。

**推测（未证实）**：四季覆盖会削弱"同月气候态"的强度，从而给出比 M2 双月更强的 skill 基线；
val 2022 上的候选比较能否支持进入 S4，按 S2→S3 的预注册门判定，本页不预写结果。

## 7. limitations（如实）

- 三年度单 ROI（27–43N/107–123E）、0.25°、17 通道；季节块是各 30 天样本，不是整季。
- 构建是准备动作，**不产生任何模型结果或科学声明**；`scientific_claim: false` 保留。
- 指纹审计证明的是源字节身份，**不**认证源本身适合科学训练。
- v1 缺陷构建的存在意味着 outputs 里有两棵近似树；任何正式运行必须显式指向 v2 根，
  不得混用 v1 的 sidecar（`data_identity` 不同，混用会被身份校验拒绝）。

## 8. 下一步

1. 在本实例上重建 train-only 气候态与 persistence（同数据公平基线），并做同数据 incumbent
   训练/选择（S3 出口要求）。
2. S4 未见年份确认：冻结协议（t2m/full 6/12/24/48/72h、≥3 seed、每年/四季 skill>0、
   seed 均值同时区间下界>0、u10/v10/mslp 对 incumbent 零退化守门、`alpha_r=0.05/(r(r+1))`）
   后一次进入 2023 test；`training/r7_simultaneous_stats.py` 已就绪。
3. #79 typed-evidence 是否随候选进确认实例，按 S2→S4 主线决定并以新协议预注册。

## 9. 证据指针

- 构建协议 / 审阅回执 / 预检 / 写入报告 / combine 回执：见 §3 表格（均在 v2 根内）
- 三 sidecar：`outputs/r7_s3_confirmation_2017_2022_2023_v2/change_scale/`、
  `outputs/r7_s3_confirmation_2017_2022_2023_v2_process_scale/`、
  `outputs/r7_s3_confirmation_2017_2022_2023_v2_typed_evidence/`
- 修正决策：`docs/decisions/0039-complete-source-fingerprint-for-prepare.md`
- 回归测试：`tests/test_r7_preflight.py`（新增 >64 MiB 完整指纹用例）、
  `tests/test_r7_process_scale_sidecar.py`（既有 96 项含指纹路径）
- 前置批：`docs/R7_S1_FOUR_SEASON_ACQUISITION.md`（v2 使用的两输入批与获取回执）
