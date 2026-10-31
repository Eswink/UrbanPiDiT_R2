# 0073 — GHSL 缺瓦片是瞬时节流而非不可达；缺瓦片被静默记成零 ⇒ 补下载 + 全 role coverage 掩码 + `attempt04`

- **状态**：accepted
- **日期**：2026-10-31
- **范围**：三项授权，按顺序执行：
  1. **实测更正**：`ghsl_built_s_2000/2005/2010` 在 `attempt03` 被记为 `failed-no-fallback`
     是**错误的归类**——那是 `read timeout` 与 `502` 造成的瞬时节流，不是层不可达。
     实时只读复测 45/45 瓦片全部 `HTTP 200`，无一 404。
  2. **修复一个更严重的缺陷**：`aggregate_built_surface` / `aggregate_building_height` /
     `aggregate_population` 三个 role **没有任何 coverage 掩码**。缺瓦片的格被算成
     `0`，与"该处确实没有建成区/没有人口"（如海面、荒漠）**不可区分**。
     `attempt03` 因此发布了带静默零值的 `built_surface_frac_2015` 与 `building_height_m`。
  3. **重新派生** `attempt04`（全新排他 `outputs/` 路径），要求 recheck 通过且
     coverage 掩码完整，使本轮交付一个**自身闸门通过、可用于科学**的已发布产物。
  **不改** `model/**`、**不重建** store、**不动** `data/raw|interim|processed` 与归档快照、
  **不删**任何 attempt 的字节、**不放宽**任何已冻结判据。
- **依据**：决策 0063（数据授权）、0070（编排重跑）、0071（格界归属）、0072（recheck 修复与
  fail-closed）、R-006、R-008、R-016（下载器唯一网络面）、R-051/R-052。

## Context

### ① `attempt03` 的"不可达"是超时，不是不可达（实测）

`attempt03/endpoint_report.json` 里 `ghsl_built_s_2000/2005/2010` 被记进 `dropped`，
`verdict = failed-no-fallback`，9 个瓦片全被标 `reachable_tiles = 0`。但逐条看 probe：

```
http_status = None
error = 'RuntimeError: 下载请求被拒绝或失败：The read operation timed out'
```

**没有一个是 404**。同一批探测里 `ghsl_built_s_2015` 的 9 个瓦片有 7 个 200、1 个 timeout、
1 个 `HTTP 502` —— 同一层内混合结果，是典型的服务端限流/抖动，而不是文件消失。

实时只读复测（`2026-10-31T11:01:41Z`，每个瓦片最多 3 次重试、超时放宽到 60/90/120 s）：

| 层 | 可达 / 总数 | 观测状态码 |
| --- | --- | --- |
| `ghsl_built_s_2000` | 9/9 | 200×9 |
| `ghsl_built_s_2005` | 9/9 | 200×9 |
| `ghsl_built_s_2010` | 9/9 | 200×9 |
| `ghsl_built_s_2015` | 9/9 | 200×9 |
| `ghsl_built_s_2020` | 9/9 | 200×9 |

⇒ **不是丢层，是抖动**。`attempt01`（`04:02:21Z`）与 `attempt02`（`2026-10-30T20:03:19Z`）
都是 9/9，`attempt03` 单次探测恰好撞上抖动，就把三个层判了死刑。

**这是探测策略的缺陷，不是数据源的问题**：`probe()` 默认 `timeout=60.0` 且**不重试**，
一次超时 = 一层丢弃。在 `attempt03` 那次运行里，这个误判还恰好掩盖了 ② 那个真缺陷。

### ② 真缺陷：缺瓦片被静默算成零（实测，比 ① 严重）

`aggregate_land_cover` 有 coverage 意识（`land_cover_coverage_frac_*` 单独发布），但
`aggregate_built_surface` / `aggregate_building_height` / `aggregate_population` 三个函数
**只看 `paths`，不看这些 tile 是否覆盖了整个格**。缺一个瓦片，该区域的贡献就是零，
于是发布出来的 `0` 与"该处确实没有建成区"完全同形。

`attempt03` 实际下载到的瓦片数（对比 catalog 的 9）：

| 层 | 应得 | 实得 | 缺失瓦片 |
| --- | --- | --- | --- |
| `ghsl_built_s_2015` | 9 | 7 | `R5_C29`、`R7_C29` |
| `ghsl_built_h_2018` | 9 | 8 | `R7_C29` |
| `ghsl_built_s_2020` | 9 | 9 | — |
| `ghsl_pop_2020` | 9 | 9 | — |

把缺失瓦片在 ROI 内的格算出来，与 `attempt01`（当时 9/9 齐全）逐块对比：

| 块 | 格数 | `attempt01` 均值 | `attempt03` 均值 | `attempt03` 零值格 |
| --- | --- | --- | --- | --- |
| `s2015` NW（`R5_C29`，rows 0–15 × cols 0–11） | 192 | `0.003024` | `0.000000` | **192/192** |
| `s2015` SW（`R7_C29`，rows 57–64 × cols 0–11） | 96 | `0.003765` | `0.000000` | **96/96** |
| `h2018` SW（`R7_C29`，同一块） | 96 | `3.1006` | `0.0000` | **96/96** |

**精确的全块归零**，而同期 `built_surface_frac_2020`（9/9 齐全）在同一区域与 `attempt01`
一致（`0.004023` vs `0.004023`）。这不是时空变化，是数据丢失被写成了观测值。

⇒ 这违反本轮最核心的那条纪律："聚合派生量不得被当成观测真值"。一个把"没下载到"
渲染成"这里是 0 建成区"的产物，比不发布更糟。

### ③ 因此 `attempt03` 有两处独立的不可用理由

1. `recheck_receipt.json` 记 `passed: false`（决策 0072 已定为 recheck 自身缺陷，修复后
   在同一批瓦片上复算 `max_relative_delta = 5.154e-08`）；
2. 即使 recheck 通过，`built_surface_frac_2015` 与 `building_height_m` 的 288 个格
   （192+96，其中 96 格两字段重叠）是**静默零值**，不可用于任何统计。

第 2 条是决策 0072 未发现的，且**不会被 recheck 抓到**——recheck 抽样的 3 个格里
`(7,64)` 在 NW 块内，但 `lc_built_frac`（WorldCover，48/48 瓦片齐全）通过，
`built_surface_frac` 取的是 `[-1]`（2020，9/9 齐全）。也就是说 **coverage 空洞正好落在
抽样格之外**。这说明 recheck 的抽样设计对"层内不完整"没有覆盖能力。

## Decision

### 第 1 条：探测加重试，超时不再等同于不可达

`urban_static_sources.probe` 的调用点（`verify_endpoints`）改为**有限重试**：
每个瓦片最多 3 次，超时 60 → 90 → 120 s，间隔 2 s/4 s。只有 HTTP **4xx**
（明确表示文件不存在/无权限）才允许判为 `failed-no-fallback`；`5xx`、超时、
DNS/TLS 失败一律标记为 `transient`，并在报告里与"层不存在"分开计数。

**不改变 R-008 的实质**：真的不可达仍要丢层并留理由。改变的是**判定依据**——
一次超时不构成"该层不存在"的证据。

### 第 2 条：所有 role 都必须带 coverage 掩码，不完备即 fail-closed

1. `aggregate_built_surface` / `aggregate_building_height` / `aggregate_population`
   各自返回一个 `coverage` 数组（按 tile bounds 求，复用已有的
   `aggregate.coverage_fraction`），字段发布为
   `built_surface_coverage_frac_<epoch>`、`building_height_coverage_frac`、
   `pop_density_coverage_frac_<epoch>`。
2. **`aggregate` 相位在任何层 coverage < 1.0 时 `SystemExit`**，除非该层在协议里
   被显式登记为 `allows_partial_coverage: true` 并写明理由。默认值是不允许。
3. 这条是本次最重要的守卫：它把"缺瓦片"从**静默零值**变成**拒绝发布**。

### 第 3 条：重新派生 `attempt04`

- 落全新排他路径 `outputs/r7_urban_static_covariates_attempt04/`（`fresh_outputs` 保证不覆盖）。
- 六相位全跑，`recheck` 必须 `passed: true`，GHSL 各层必须 9/9 瓦片。
- 用 workflow 非 dryRun 驱动，忠实性闸门 F1–F4 照旧（决策 0071）。
- 出口 E1–E6 见下节，**科学口径不放宽**。

### 第 4 条：`attempt03` 的处置

不删、不改字节；在证据页标注为**双重不可用**（recheck 未过 + 288 格静默零值），
并在 `docs/R7_URBAN_STATIC_COVARIATES.md` 增补这一节的实测表。

## 预声明出口（E1–E6，跑之前冻结）

- **E1 六相位全绿**：`verify-endpoints`/`freeze`/`download`/`aggregate`/`recheck`/`publish`
  退出码全 0，且 `BUILD_COMPLETE.json` 存在。
- **E2 层完整性**：13 层中的 12 个 GHSL/WorldCover 层**每层瓦片数等于 catalog 数**
  （WorldCover 允许其公开的全海空瓦片缺失，但必须在 `absent_tiles` 里具名）；
  LandScan 仍 `failed-no-fallback` 且 `reason` 非空（决策 0072 第 4 条）。
- **E3 coverage 完备**：每个 GHSL 层的 `*_coverage_frac_*` 在全部 4225 格上等于 1.0；
  任何 < 1.0 必须让 `aggregate` 相位失败，不得进入 `recheck`。
- **E4 recheck 通过**：`max_relative_delta <= 1e-05`、`passed: true`。
- **E5 与 `attempt01` 的差异可解释**：`built_surface_frac_2020`/`pop_density_km2_2020`/
  `building_height_m` 与 `attempt01` 的差应仅来自格心→格界（决策 0071 实测中位
  `0.139%`、最大 `1.225%`），**不得**出现整块归零或数量级差异。
- **E6 身份与登记**：新 `static_identity` 记录、`protocol_sha256` 与下载时一致、
  账本新增记录（新 `record_id`）且 95 条旧记录不重复；`tools/r7_round.py verify` 退出码 0。

**停止条件**：磁盘余量跌破 60 GiB；网络累计超过 40 GiB；墙钟逼近 86400 s；
或任一 GHSL 层在重试后仍 < 9/9 瓦片（此时按 E3 fail-closed，不发布，等抖动过去重试）。

## Consequences

- 探测重试会让 `verify-endpoints` 变慢（9 层 × 9 瓦片 × 最坏 3 次 × 120 s），但这是
  **用时间换正确性**：一次超时丢掉三个真实存在的层，代价远大于多等几分钟。
- coverage 掩码让产物变大（每个 GHSL 层多一个 65×65 float32 数组，可忽略），
  但换来"零值可信"这一性质。
- `attempt04` 若成功，将是本轮**第一个**自身闸门通过、可用于科学的产物。
  在此之前，本轮交付物只有 `scientific_claim: false` 的数据准备读数。
- 预算是软预算 + 宽松硬上限（决策 0030）：名义 `planned 21600 s / hard 86400 s` 不变；
  `attempt01` 实测整轮 9550 s（决策 0064），`attempt03` 的下载量 `13,727,171,047 B`
  与之一致，故无需另立预算更正决策。
- 0 GPU-h，只推工作分支。

## 与既有决策的关系

- **0072**：本 ADR 发现的 coverage 空洞是 0072 未覆盖的，且**recheck 抓不到**（抽样格恰好
  在空洞之外）。0072 的第 4 条（丢层必须写理由）已让 `attempt03` 的三条 `reason: null`
  暴露出来；本 ADR 进一步查明那些"丢层"本身是误判。
- **0071**：格界归属不变；`attempt04` 仍用格界规则。
- **0070**：其实测到编排器可以跑完整轮，本 ADR 复用该路径。
- **R-008**：第 1 条细化了"失败"的判定门槛——超时不等于不存在——但**没有**放宽
  "失败必须留审计记录"的要求。
