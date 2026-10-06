# S3 batch-3：四季 2018–2021 区域 ERA5 获取与合并（16/16 成功、无失败）

**状态：获取/合并已完成并登记（2026-10-06）；`scientific_claim: false`。**
本页对应 `docs/goals/main-model-climatology-campaign.md` 的 D3 分批扩围与决策 0038；
它记录一次**数据获取**，不产生任何模型结论。v3 确认实例将使用本批输出（train 扩至
2017–2021），见 `docs/R7_S3_CONFIRMATION_INSTANCE_V3.md`（该页在 v3 构建后另立）。

本批是对 S3-UB/S3-BC 已登记预算读数的直接响应：单年 train 上预算饱和（1600 更新时
t2m 仍在改善但 48/72h 守门恶化 13→17），预声明读数判定「下一能力投资是数据」——
本批把 train 年份从 1 年扩到 5 年。

## 1. 结论摘要

- 16 个 (year, season) part（2018–2021 的 Jan/Apr/Jul/Oct 各 30 天 × 120 时间点）
  **全部** `downloaded-real-source`，**零失败**；合并源 **1920 时间点**、
  SHA256 `97d29bca0633cc55df81a97081733f5d6ad425d4562aac78165ef2f0044daef8`
  （309,074,574 字节）。
- 实测网络 **57,084,203,564 字节（57.08 GB = 53.16 GiB）**，为冻结计划
  58,000,000,000 的 **98.4%**，远低于 100 GiB 硬上限（53.2%）；16 个 part 合计
  20,709.0 秒（约 5.75 h），低于 planned 21600 s，硬上限 43200 s 未触。
- 全批次串行执行（一次一个 part）；每 part 由 `timeout -k 60 -s TERM 1860` 看门狗
  约束（batch-2 挂死修复的直接继承），16 个 part 没有一个触发看门狗或 deadline。
- 氪金/GPU 零消耗：本节点 0 GPU-h、0 付费资源；获取期间实验侧无 GPU 子进程。

## 2. 冻结与修正

| 项 | 值 |
| --- | --- |
| 获取协议 | `outputs/r7_s3_batch3_20182021/acquisition_protocol.json`，canonical `protocol_sha256 df9cc22b75f608ef4493c8165ffc65ae66d8d5adb7b9aed8c925c92e790772c5`（文件 SHA256 `eaca2e85b1fc98f51aa2278c2bdea2be4967f1a5a5ee01471472947f062895b2`） |
| 源 | `s3://earthmover-icechunk-era5/icechunkV2`，快照 `ZFKDHBCTBVHVXM3BQFV0`（匿名，CC-BY-4.0） |
| 区域/分辨率 | ROI 27–43N/107–123E、0.25°、65×65、17 通道（与 S1/batch-2 同合同） |
| 计划 | 16 part × 120 stamps = 1920 stamps；每 part 一个 (year, season) 块 |
| 预算（冻结） | 每 part deadline 1800 s、planned 网络 58,000,000,000 B、硬上限 100 GiB、planned 21600 s、hard 43200 s、磁盘峰值 8 GiB、余量门 20 GiB |
| 预算依据 | 按 batch-2 注册测量（28,773,423,423 B / 960 stamps ≈ 30.0 MB/stamp）；1920 stamps × 30 MB ≈ 57.6 GB |
| 恢复规则（冻结） | **不实现续传**；失败 part 只以新 artifact 名重试，绝不删除部分文件让其"重跑成功" |
| 看门狗 | `timeout -k 60 -s TERM 1860` 每 part（下载器 deadline 只在 stamp 边界检查，阻塞读不触发；见 batch-2 §5.2） |
| 只读 preflight | `preflight_d1_namespace.json`（SHA256 `ce1d04cd…`）命名空间/单位/levels 审计；`preflight_review.json`（SHA256 `168085e1…`）执行者自审 14 项机械核对全过，`decision=accept to --write` |
| 授权 | 决策 0038：免费合法区域数据，执行者自审只读 preflight 后向全新排他 `outputs/` 路径 `--write` |

## 3. 逐 part 实测

顺序即合并顺序（按年、按季 winter→spring→summer→autumn）。

| # | part | 状态 | 网络（字节） | 墙钟（秒） | 首/末时间戳 | 产物 SHA256 |
| ---: | --- | --- | ---: | ---: | --- | --- |
| 1 | winter_2018 | `downloaded-real-source` | 3,554,716,077 | 1267.3 | 2018-01-01T00 → 2018-01-30T18 | `7e6e13dafc8c30ce052d19556e6735360205da107826d6131c1f54c9b131d6bf` |
| 2 | spring_2018 | `downloaded-real-source` | 3,611,965,938 | 1298.9 | 2018-04-01T00 → 2018-04-30T18 | `5b0fde408bdd0959a59037c0a1a2b905e3a85f6a468a74caaaf96fd63e04f71c` |
| 3 | summer_2018 | `downloaded-real-source` | 3,548,080,435 | 1283.2 | 2018-07-01T00 → 2018-07-30T18 | `da50b13c88712a69cf5f9f2b6a1555759cab51f7b0c8356d08d5718c602a3dd5` |
| 4 | autumn_2018 | `downloaded-real-source` | 3,558,843,655 | 1291.5 | 2018-10-01T00 → 2018-10-30T18 | `487b089b87626c57e61c21ac5625eaad1050bcf51ea5299abde8916bd4b3058f` |
| 5 | winter_2019 | `downloaded-real-source` | 3,569,186,877 | 1249.8 | 2019-01-01T00 → 2019-01-30T18 | `0d15a7d7c5989403cb9e7b68cc63a84d2673fb54f112e843c07b36eb3d7d09e7` |
| 6 | spring_2019 | `downloaded-real-source` | 3,601,697,005 | 1299.2 | 2019-04-01T00 → 2019-04-30T18 | `79e4bbb52d9dd5940e06d722c18d45dbe37cf9f8e16ef223415d5eccbf6d8596` |
| 7 | summer_2019 | `downloaded-real-source` | 3,566,949,951 | 1309.5 | 2019-07-01T00 → 2019-07-30T18 | `0c0c958ddf4e36089ba9cfa5156f9a8513a7cfcd1ade6142ba000345ca295d3f` |
| 8 | autumn_2019 | `downloaded-real-source` | 3,564,459,726 | 1267.9 | 2019-10-01T00 → 2019-10-30T18 | `ede21b19136175fb934ecf878c12de81744b51c2baba70331f621e830f8a4d96` |
| 9 | winter_2020 | `downloaded-real-source` | 3,542,441,051 | 1263.8 | 2020-01-01T00 → 2020-01-30T18 | `8905d2f3f64f7d06203ec0adec34ba88ba997781fdb8494d44684a9d9d2bac33` |
| 10 | spring_2020 | `downloaded-real-source` | 3,599,021,429 | 1237.7 | 2020-04-01T00 → 2020-04-30T18 | `ea3e5a9135965e14b031cfaa99f6c00aaa39fade86b4b2a0b0ee243e0f6ac7eb` |
| 11 | summer_2020 | `downloaded-real-source` | 3,564,407,487 | 1426.9 | 2020-07-01T00 → 2020-07-30T18 | `f45741031f561130dbf0bc51c494cb082e08fa3b2cf3a1169c12c7566bc311b4` |
| 12 | autumn_2020 | `downloaded-real-source` | 3,542,447,026 | 1313.6 | 2020-10-01T00 → 2020-10-30T18 | `1ce48d9483369aec4d7ec4203c76d4a3cab17000f757a1e6e4fb6c5077367553` |
| 13 | winter_2021 | `downloaded-real-source` | 3,548,071,120 | 1295.4 | 2021-01-01T00 → 2021-01-30T18 | `4794a1bd1094e37986cb06afb81b4fa231c8e3b66e1ad145f84e5d493b12f1f9` |
| 14 | spring_2021 | `downloaded-real-source` | 3,589,361,870 | 1286.5 | 2021-04-01T00 → 2021-04-30T18 | `eaf027eeb7add08db829a2fbfa4a3714b719d974dbf2c00d78efe3b3bab9b87f` |
| 15 | summer_2021 | `downloaded-real-source` | 3,567,205,610 | 1344.9 | 2021-07-01T00 → 2021-07-30T18 | `7e3f83dd3f01ab32df4850ca45db7d3c8f6668d2bde17d5abdd97a7209f60dc0` |
| 16 | autumn_2021 | `downloaded-real-source` | 3,555,348,307 | 1273.0 | 2021-10-01T00 → 2021-10-30T18 | `516b0876cd8c48a1919017091767e1e5f6a21fd424a89962cdcf0cb1e861e7ca` |
| **合计（16/16）** | — | — | **57,084,203,564** | **20,709.0** | 2018-01-01T00 → 2021-10-30T18 | — |

每 part 回执都带 `synthetic_fallback: false`、`scientific_claim: false`、
`source_is_real_reanalysis: true`；17 通道 payload 逐个 SHA256 与 min/max 范围记录在
各 part 回执与合并回执中。

合并命令（下载器 `--merge` 模式，要求输入 outputs 全新且不存在）：16 个 part 与
16 个回执同序传入，`--expected-stamps 1920`。合并轴等于 16 个 part 各自记录时间戳的
精确有序并集（唯一、递增、覆盖四个 UTC init 小时；下载器逐 part 复验回执 SHA256 后
才拼接，`refusing to merge unverified input`）。

## 4. 身份链

| 项 | 值 |
| --- | --- |
| 源 | `s3://earthmover-icechunk-era5/icechunkV2`，快照 `ZFKDHBCTBVHVXM3BQFV0`（匿名，CC-BY-4.0） |
| 合并源 | `outputs/r7_s3_batch3_20182021/source.nc`，309,074,574 字节，SHA256 `97d29bca0633cc55df81a97081733f5d6ad425d4562aac78165ef2f0044daef8`（1920 stamps） |
| 时间范围 | 2018-01-01T00 → 2021-10-30T18（1920 stamps） |
| 合并回执 | `outputs/r7_s3_batch3_20182021/merge_receipt.json`，SHA256 `68e6f5e09319df26ffffc6e67c153cb05d85ae29281c7b8b3167abd65e4bf579`（`status: merged-real-source`、`n_parts: 16`、`local_artifact.sha256` 与上一致、`synthetic_fallback: false`） |
| 逐 part 回执 | `outputs/r7_s3_batch3_20182021/parts/part_<season>_<year>_receipt.json`（16 个，全部 `downloaded-real-source`） |
| 消费方 | v3 确认实例（train 2017–2021 / val 2022 / test 2023；与 S1 2017 源、batch-2 2022/2023 源按时间顺序合并）；v3 页面在构建后另立 |
| test 读取 | **否**：2023 的测试段来自 batch-2 源，本批不含 2022/2023；本页全程未打开任何 test 评分 |

## 5. 真实失败与处置

**本批零失败**（16/16 `downloaded-real-source`），无 `failed-no-fallback` 记录、
无重试、无预算修正：冻结的 1800 s 每 part deadline 与 `timeout -k 60 -s TERM 1860`
看门狗在全部 16 个 part 上都没有触发；解码预算 16 GiB 每 part 也没有拒绝任何一次
字段读（各 part 实际解码约 9.5 GB，与 S1/batch-2 的审计读数一致）。

对比 batch-2 的两次保留失败（8 GiB 解码上限拒读、winter_2023 挂死），本批直接继承
了两项修正：解码上限按已审计实际值 16 GiB 冻结、每 part 带看门狗；因此两项失败模式
在本批都没有复现。这是"修正有效"的一致观察，不构成对 batch-2 失败根因的
独立验证（本批没有发生挂死，所以看门狗路径没有被实际触发过——它是预防性约束）。

## 6. 成本与预算对表

| 项 | 值 |
| --- | --- |
| GPU-h | 0.0000（本批为纯下载/合并，无 CUDA 设备打开） |
| 网络 | 57,084,203,564 字节（53.16 GiB）= 冻结计划的 98.4%、硬上限的 53.2% |
| 墙钟 | 16 个 part 合计 20,709.0 s（5.75 h）；下载器会话从 20:53:56 到 02:40:31（ALL_PARTS_DONE） |
| 冻结预算 | planned 21600 s / hard 43200 s / 每 part deadline 1800 s / 磁盘峰值 8 GiB / 余量门 20 GiB |
| 磁盘 | 16 个 part 约 309 MB + 合并源 309 MB；`/data` 余量 571 GB（远高于 20 GiB 门） |
| 失败 | **无**（16/16 成功）；无预算修正、无重试、无删除 |
| 付费/凭据 | 0 付费、0 凭据（匿名公开 S3 读取） |

## 7. 已确认与推测

**已确认**：16/16 part `downloaded-real-source` 且逐 part SHA256 记录在案；合并轴等于
16 个 part 记录时间戳的精确有序并集（唯一、递增、四个 UTC 小时）；合并源 SHA256
`97d29bca…` 与合并回执一致；17 通道单位与 levels 经只读 preflight 审计；每 part
网络实测与冻结依据（30 MB/stamp）一致；test 未读；零失败。

**推测（未证实）**：五倍 train 数据（2017–2021）会让同数据气候态更强（5 年 (month,hour)
均值比单年稳定），从而抬高候选必须超越的基线；同时可能改善长 lead 的 pattern skill
（ρ），缓解 48/72h 守门退化。v3 上的控制重训与预算读数将检验这一点，本页不预写结果。

## 8. limitations（如实）

- 单 ROI（27–43N/107–123E）、0.25°、17 通道；季节块是各 30 天样本，不是整季或全年。
- 本批是**数据获取**：不训练模型、不评分、不产生任何科学声明；`scientific_claim: false`。
- 看门狗未被实际触发过，它的有效性在正常路径上没有证据（预防性约束）。
- 网络读数为每 part 的非 loopback 接口 recv-byte 增量，是主机级计数，含少量非本进程
  流量（方法学同 batch-2）。
- 本批**没有**复现 batch-2 的失败模式，因此不能据此说那些根因已被验证消除。

## 9. 证据指针

- 协议：`outputs/r7_s3_batch3_20182021/acquisition_protocol.json`（`df9cc22b…`）
- preflight 命名空间报告：`outputs/r7_s3_batch3_20182021/preflight_d1_namespace.json`
- preflight 自审回执：`outputs/r7_s3_batch3_20182021/preflight_review.json`
- 下载脚本：`outputs/r7_s3_batch3_20182021/download_parts.sh`（SHA256 `49b44cde978fb0bf9a2fc12ef51345c6a6040f458fa772bee9c30cc06b2308d3`）
- 运行日志：`outputs/r7_s3_batch3_20182021/download_run.log`（含逐 part 回执行与 `ALL_PARTS_DONE 02:40:31`）
- 16 个 part 与回执：`outputs/r7_s3_batch3_20182021/parts/`
- 合并源与回执：`outputs/r7_s3_batch3_20182021/source.nc`、`merge_receipt.json`
- 索引记录：`record:s3-batch3-20182021-acquisition`（`docs/R7_EVIDENCE_INDEX.jsonl`）

## 10. 下一步

1. 合并三个已注册源（S1 2017 480 stamps + 本批 1920 + batch-2 960），经
   `prepare_r7_local.py` 只读 preflight 与执行者自审后向全新路径
   `outputs/r7_s3_confirmation_train2017_2021_v3/` `--write` 建 v3 实例
   （`configs/r7_s3_confirmation_v3.yaml` 已冻结：train 2017–2021 / val 2022 / test 2023）。
2. v3 上重建同数据 climatology 与 persistence（D2-analog）、重训 400 更新 incumbent
   （D3-analog，逐位 seed 保真），再跑预算剂量读数，检验"数据扩年提升 pattern skill"假设。
3. S4 冻结包只在候选通过 primary+gate 合取后才启动；test 保持封印，直到预注册的
   未见年份确认轮。
