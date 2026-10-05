# 0039 源指纹必须完整：`source_fingerprint` 不再对大文件降级为 stat-only

- **日期**：2026-10-05
- **状态**：accepted

## Context

`data/preprocess/r7_preflight.py` 的 `source_fingerprint()` 自 D1 期（E-080 之后的 #30 轮）起
对常规文件带一个 64 MiB 的有界哈希上限：≤64 MiB 记 `full-local-file`（完整 SHA256），
更大的文件降级为 `file-stat-only`（只有 bytes/mtime，`sha256: None`）。当时唯一的 store 源
是 37,734,176 字节的 `outputs/r7_m2_segment/source.nc`，上限从未生效。

新的主模型超气候态方向把源推进到多季节/多年度尺寸：S1 四季 2017 源 77,177,840 字节、
S3 确认实例合并源 231,865,649 字节，都超过该上限。后果在一次真实发布中暴露：S3 确认 store
（train 2017 / val 2022 / test 2023）建成后，change-scale sidecar 可发布，但 process-scale
sidecar 的只读 preflight 在 `r7_process_scale_sidecar.py:260` `_source_identity` 失败——
该模块（以及 `training/r7_m3_identity.source_identity` 与复用它的整条 M3/V2 身份链）
要求 `scope == 'full-local-file'` 且 `sha256` 为 64 位十六进制，stat-only 指纹会被拒绝。
即：任何源超过 64 MiB 的真实实例都**结构上不可能**发布 process-scale sidecar 或通过该身份链，
而 S4 确认的 incumbent（actual C 的 `process` 臂）训练契约需要 per-store sidecar，
所以这个缺陷会直接挡住正式确认。S1 的 77 MB 源同样因此从未有 process-scale sidecar。

两个既有文本已经指向正确方向：ADR 0038 第 4 条把"完整 source SHA256 核对后"作为 `--write`
的前置；`.agents/skills/real-data-acquisition/SKILL.md` 的检查点明确写
"`file-stat-only`…**弱于完整 SHA256**；对科学运行，应设法取得完整 hash"。
64 MiB 上限是 D1 期的成本防御，不是身份契约要求的判据；真实源在 132 MiB/s 下 221 MB
只需约 1.7 秒，成本已不成立。

## Decision

1. `source_fingerprint()` 对常规本地文件**总是**做完整哈希，返回
   `{'scope': 'full-local-file', 'sha256': <完整 SHA256>, 'bytes': <大小>}`；
   删除 `max_hash_bytes` 参数与 `file-stat-only` 降级分支。对目录（zarr 根元数据）
   行为不变（`zarr-root-metadata-only`，全文读入根元数据文件）。
2. 完整哈希是**更强**的数据身份，不是放宽：它使多年度/大源实例能够通过既有的
   process-scale sidecar 与 M3/V2 身份校验，而不是绕过它们。任何读取方仍以
   `full-local-file` 为唯一可接受 scope。
3. 已写出的 v1 确认 store 审计（`source_preflight.json` 记 stat-only）**不修改**：
   fresh_outputs/write-once 规则下，修正方式是在**全新排他路径**重建实例（v2），
   v1 与其缺陷读数原样保留。旧 store 的既有审计不回改、不编辑。
4. S1/S3 记录里以别的渠道（merge/combine receipt 的完整哈希）补齐的 source hash 事实
   不被改写；本决策只改变**将来**的 preflight 指纹行为。
5. 回归测试钉住边界：超过旧 64 MiB 阈值的文件必须得到完整 `full-local-file` 指纹
   （`tests/test_r7_preflight.py`），且输入文件不被修改。

## Consequences

**变容易的：** 多季节/多年度真实源现在能在 preflight 审计里携带完整内容哈希，process-scale
sidecar 与 M3/V2 身份链对任何尺寸的实例可用；S1/S3 记录中"指纹弱于完整哈希、需从 receipt
补读"的手工说明不再必要；S4 确认实例的前置依赖（per-store sidecar）可以按原设计构建。

**变难 / 代价（如实列出）：** 每次 preflight 都要读一遍源文件全部字节——大源（数百 MB 到数 GB）
会增加秒级到分钟级 CPU/磁盘读时间与页缓存压力；在超大源（>数十 GB）上若无节制可能成为真实
成本，将来需要在上限策略上另立决策。删除 `file-stat-only` 使旧路径以 stat-only 指纹
建立的行为不可再表达，若有不需内容哈希的场景（例如纯速率 pilot）仍会付全额哈希成本。
本决策不改任何读取方的接受判据（仍要求 full-local-file），因此对既有已发布 store 的
校验结果无影响；但它**不能**追溯修复已写出的 v1 审计——那需要重建实例。

**备选方案与否决理由：** 保留上限并给 `file-stat-only` 加"可选完整哈希"参数不采用——
它让正确路径成为例外、错误路径成为默认，且 process-scale sidecar 只接受完整 scope，
等于要求调用方记得加参数；把 64 MiB 提到某个更大数字不采用——任何固定上限都会再次
被下一批更大的真实源越过，同样的失败会以同样方式复发；在 sidecar 侧放宽为接受 stat-only
不采用——那是弱化数据身份校验，直接违反"绕过 checkpoint/数据身份校验"禁令，也会让
M3/V2 身份链失去内容级防篡改能力。
