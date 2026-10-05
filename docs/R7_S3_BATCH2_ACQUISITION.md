# S3 batch-2：四季 2022/2023 区域 ERA5 获取与合并（含两次保留失败）

**状态：获取/合并已完成并登记（2026-10-05）；`scientific_claim: false`。**
本页对应 `docs/goals/main-model-climatology-campaign.md` 的 D3 分批扩围与决策 0038；
它记录一次**数据获取**，不产生任何模型结论。v2 确认实例使用本批输出，
见 `docs/R7_S3_CONFIRMATION_INSTANCE.md`。

## 1. 结论摘要

- 8 个 (year, season) part（2022 与 2023 的 Jan/Apr/Jul/Oct 各 30 天 × 120 时间点）
  全部 `downloaded-real-source`；合并源 **960 时间点**、
  SHA256 `44a24ca0096c3802fefc92a5e089c7283812001cfcffd6f88b2e06187607ec62`（154,740,969 字节）。
- 实测网络 **28,773,423,423 字节（28.77 GB）**，为冻结计划 28,800,000,000 的 **99.9%**，
  远低于 48 GiB 硬上限（55.8%）；8 个成功 part 合计 9,832.8 秒（约 2.73 h）。
- **两次真实失败全部保留**（`failed-no-fallback`，无合成回退），重试按冻结恢复规则用
  **新 artifact 名**（`_r2`）执行，从未删除部分产物：
  1. `winter_2022`：冻结的每 part 解码上限 8 GiB 过低被拒（`decoded-chunk budget exceeded
     BEFORE field read`，零字节计费、零文件写入）；决策留痕为
     `budget_amendment.json`（8→16 GiB，即 S1 实际跑通的数值），成功后以 `winter_2022_r2` 重试。
  2. `winter_2023`：挂死超过冻结 1800 s 每 part 期限（全线程 futex 等待、15 s 零 CPU-time
     增量、无接收队列进展；下载器的 deadline 只在 stamp 边界检查，阻塞读不触发）。
     执行者向**自己的进程**发 SIGTERM 终止（共驻规则禁止信号非本实验进程；这是本实验），
     自撰 executor 审计回执（下载器失败处理不覆盖 SIGTERM 路径），并把
     `download_parts.sh` 加上 `timeout -k 60 -s TERM 1860` 每 part 看门狗；随后以
     `winter_2023_r2` 成功。
- 氪金/GPU 零消耗：本节点 0 GPU-h、0 付费资源。

## 2. 冻结与修正

| 项 | 值 |
| --- | --- |
| 获取协议 | `outputs/r7_s2_batch2_20222023/acquisition_protocol.json`，`protocol_sha256 7847fa1e9ebe94cf7d5d2fd138a973024844dd1be249563d5d5821d9e3cf88cb` |
| 源 | `s3://earthmover-icechunk-era5/icechunkV2`，快照 `ZFKDHBCTBVHVXM3BQFV0`（匿名，CC-BY-4.0） |
| 区域/分辨率 | ROI 27–43N/107–123E、0.25°、65×65、17 通道（与 S1 dev store 同合同） |
| 计划 | 8 part × 120 stamps = 960 stamps；每 part 一个 (year, season) 块 |
| 预算（冻结） | 每 part deadline 1800 s、planned 网络 28,800,000,000 B、硬上限 48 GiB、planned 10800 s、hard 28800 s、磁盘峰值 8 GiB、余量门 20 GiB |
| 预算修正 | `budget_amendment.json`：每 part 解码上限 8→16 GiB（失败后、任何 part 成功前冻结；理由是 S1 审计显示同规格 part 实际解码约 9.48 GB，8 GiB 上限必然拒绝第一次字段读） |
| 恢复规则（冻结） | **不实现续传**；失败 part 只以新 artifact 名重试，绝不删除部分文件让其"重跑成功" |

## 3. 逐 part 实测

| part | 状态 | 网络（字节） | 墙钟（秒） | 产物 SHA256（前 16） |
| --- | --- | ---: | ---: | --- |
| winter_2022 | `failed-no-fallback`（预算拒读） | 0 incurred | 8.0 | — |
| winter_2022_r2 | `downloaded-real-source` | 3,583,080,527 | 1206.7 | 见 part 回执 |
| spring_2022 | `downloaded-real-source` | 3,631,590,726 | 1270.0 | 见 part 回执 |
| summer_2022 | `downloaded-real-source` | 3,617,398,110 | 1268.8 | 见 part 回执 |
| autumn_2022 | `downloaded-real-source` | 3,584,852,151 | 1247.7 | 见 part 回执 |
| winter_2023 | `failed-no-fallback`（挂死，执行者终止） | 未计费（见回执） | ~2364（约 39.4 min） | — |
| winter_2023_r2 | `downloaded-real-source` | 3,551,496,594 | 1140.3 | 见 part 回执 |
| spring_2023 | `downloaded-real-source` | 3,655,721,062 | 1315.4 | 见 part 回执 |
| summer_2023 | `downloaded-real-source` | 3,582,410,082 | 1209.3 | 见 part 回执 |
| autumn_2023 | `downloaded-real-source` | 3,566,874,171 | 1174.5 | 见 part 回执 |
| **合计（成功 8 part）** | — | **28,773,423,423** | **9,832.8** | — |

合并回执 `merge_receipt.json`（44,441 字节）记录每 part 的逐条 SHA256 与全部 payload；
合并轴等于各 part 记录时间戳的精确有序并集（唯一、递增、覆盖四个 UTC init 小时）。

## 4. 身份链

| 项 | 值 |
| --- | --- |
| 合并源 | `outputs/r7_s2_batch2_20222023/source.nc`，154,740,969 字节，SHA256 `44a24ca0096c3802fefc92a5e089c7283812001cfcffd6f88b2e06187607ec62` |
| 时间范围 | 2022-01-01T00 → 2023-10-30T18（960 stamps） |
| 合并回执 | `outputs/r7_s2_batch2_20222023/merge_receipt.json`（`local_artifact.sha256` 与上表一致；`synthetic_fallback: false`） |
| 消费方 | v2 确认实例（`docs/R7_S3_CONFIRMATION_INSTANCE.md`）：与 S1 2017 源合并为 `e0b51616…` 后建 store |
| test 读取 | **否**：2023 的天气标签只在 v2 store 构建时按 split 分区写入，从未评分 |

## 5. 真实失败与处置（逐条）

1. **winter_2022 预算缺陷（配置错误，非数据/网络问题）**：首跑在第一次字段读之前被
   `decoded-chunk budget exceeded` 拒绝，回执 `status=failed-no-fallback`、零字节计费、
   零文件写入。处置：冻结 `budget_amendment.json`（`decided: before any successful part;
   after the first attempt was refused`），仅改每 part 解码上限 8→16 GiB，其余预算不动；
   以新名 `winter_2022_r2` 重试成功。失败名 `part_winter_2022_receipt.json` 原样保留。
2. **winter_2023 挂死（真实运行时失败）**：39.4 分钟后仍无进展（全线程 futex、零 CPU 增量、
   无网络接收），超过冻结的 1800 s 每 part 期限。根因（已确认的事实）：下载器 deadline 只在
   stamp 边界检查，阻塞中的读不会触发；网络侧逐连接限速（本批实测单流约 0.60 MiB/s、
   12 流并发约 6.36 MiB/s 的隔离读数见 `HANDOFF_transport_amendment.md`，该文件是并行只读
   分析窗口的工作态交接，**不是**本批证据，数字未按登记纪律复测、不进入任何 digest）。
   处置：执行者终止自己的进程（合规；未触碰任何非本实验进程），自撰 executor
   `failed-no-fallback` 回执（含 `receipt_author: executor` 与 `retry_rule`），给
   `download_parts.sh` 加 `timeout -k 60 -s TERM 1860` 看门狗，以 `winter_2023_r2`
   重试成功。失败回执 `part_winter_2023_receipt.json` 原样保留。

两次失败都没有删除任何部分产物；`_r2` 命名规则写在冻结的恢复策略里，不是事后发明的捷径。

## 6. 已确认与推测

**已确认**：8 part 全部真实下载且逐 part 回执齐（`status=downloaded-real-source`、
`synthetic_fallback=false`）；网络总量 28,773,423,423 字节 = 计划 99.9%；合并轴为精确
有序并集；源字节 SHA256 与合并回执一致；17 通道与四 UTC 小时合同与 S1/M2 一致；
两次失败全部有 `failed-no-fallback` 审计且无合成替代。

**推测（未证实）**：`HANDOFF_transport_amendment.md` 的并发提速主张（把
`async.concurrency` 从 1 提到 6–12 可提速 5–10×）来自并行只读窗口的隔离 curl 读数，
未按本批证据纪律复测，**不据此宣称任何已实现收益**；本批实际吞吐为串行单流。
若后续批次要提速，须先冻结新协议并用 pilot 复测后再改下载器。

## 7. limitations（如实）

- 两年度单 ROI（27–43N/107–123E）、0.25°、17 通道；每季节仍是 30 天样本。
- 本批不建 store、不训练、不评分；`scientific_claim: false` 保留。
- winter_2023 的挂死没有拿到堆栈级根因（进程已终止），"单流阻塞读"是**推测**，
  与"逐连接限速"的隔离事实一致，但未做受控复现。
- `HANDOFF_transport_amendment.md` 是跨窗口工作态交接，非证据产物，不参与身份链。

## 8. 证据指针

- 获取协议：`outputs/r7_s2_batch2_20222023/acquisition_protocol.json`
- 预算修正：`outputs/r7_s2_batch2_20222023/budget_amendment.json`
- 逐 part 回执：`outputs/r7_s2_batch2_20222023/parts/part_*_receipt.json`（含两个
  `failed-no-fallback`）
- 合并回执与源：`outputs/r7_s2_batch2_20222023/merge_receipt.json`、`source.nc`
- 下载脚本（含看门狗与 `_r2` 规则）：`outputs/r7_s2_batch2_20222023/download_parts.sh`
- 消费方实例：`docs/R7_S3_CONFIRMATION_INSTANCE.md`
