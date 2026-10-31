# 0072 — 第三套 recheck 的量纲缺陷修复、publish 对 recheck 不通过 fail-closed、被丢弃层强制具名理由

- **状态**：accepted
- **日期**：2026-10-31
- **范围**：四项互相独立的工程修复，**全部只收紧、不放宽**：
  1. **修复** `data/preprocess/urban_static_recheck.py` 的 `land_cover` 与 `building_height`
     两个分支的**量纲（sin-band）缺陷**，使独立再派生改用**绝对球面重叠面积 m²** 作为权重。
  2. **补一个会抓住该缺陷的单测**——现有测试集里没有任何用例覆盖该模块，缺陷是靠 attempt03
     的 `passed: false` 才被发现的。
  3. 让 `publish()` 对 `recheck_receipt["passed"] is False` **fail-closed**（当前只把结果写进
     manifest，不阻断）。
  4. 让 `freeze()` 对**任何被丢弃的具名层**强制要求**非空 `reason`**（当前 `ghsl_built_s_2000/
     2005/2010` 三条实测 `reason: null`）。
  **不改** `model/**`、**不重建** store、**不动** `data/raw|interim|processed` 与归档快照、
  **不删**任何 attempt 的字节、**不改**任何已发布的科学数值、**不放宽**任何已冻结判据。
- **依据**：决策 0063（数据授权）、0070（编排重跑）、0071（格界归属 + 重放忠实性闸门）、
  R-006（先冻协议）、R-008（失败不合成、必须留 `failed-no-fallback` 审计）、
  R-051/R-052（规模上限）、`docs/rules/testing.md`（fixture 与断言纪律）。

## Context

### ① `land_cover` / `building_height` 的量纲缺陷（实测）

attempt03 的 `recheck_receipt.json` 记 `passed: false`、`max_relative_delta = 913872.595`，
容差 `1e-05`。12 项比较中 `built_surface_frac`（`1.8e-08`–`2.4e-08`）与 `pop_density_km2`
（`8.9e-10`–`6.7e-09`）通过，而 `lc_built_frac`（`7.9e+05`–`9.1e+05`）与
`building_height_m`（`2.7e-05`–`4.3e-04`）失败。

根因是**分子与分母不同量纲**。该模块的 `share = lat_fraction * lon_fraction` 是
**每个原生像素落入目标框的比例**，无量纲；对格子 (7,64) 实测
`sum(all share) = 9,000,000`（= 3000 行 × 3000 列），即"像素个数"单位。但分母写的是

```python
total = (np.sin(np.radians(lat_north)) - np.sin(np.radians(lat_south))) * native_cols
```

即 `sin_band × 3000 = 9.8416`——**无量纲 sin band 乘以列数**，既不是面积也不是像素计数。
`9.0e6 / 9.8416 = 914,634`，与实测 `913,872` 同量级，确认这就是那 5 个数量级。

`building_height` 同源：`value_sum / area_weight` 两处都用 `share`，得到一个**未按面积加权**
的均值，故偏差小得多（`2.7e-05`–`4.3e-04`）但仍是错的——纬度越高原生像素面积越小，
不加权就是系统性偏差。

### ② 但**聚合器是对的**（实测，这一步必须先做，否则会"修坏"正确的东西）

在动手前用**完全独立的暴力实现**（逐瓦片、逐原生像素、绝对球面面积）复算同一格：

| 量 | 值 |
| --- | --- |
| 已发布 `lc_built_frac[7,64]` | `0.28414282` |
| 独立绝对面积复算 | `0.284142837` |
| 相对差 | `5.15e-08` |

⇒ 已发布数值**正确**，缺陷只在 recheck 一侧。这一点很重要：**0046 之前的一版"独立再派生"
之所以给出 `5.15e-08`，是因为它与聚合器共用格心规则**（决策 0071 ④）；本次的 `5.15e-08`
是**第三套实现用不同规则算出的真一致**，两者数值相同而含义完全不同。

### ③ 四个 role 的正确权重**不是同一个**

产物里各层的原生值语义不同，故权重必须分开。用绝对球面重叠面积 `m²` 与重叠比例
分别试算，实测：

| role | 原生值语义 | 正确权重 | 修正后相对差 |
| --- | --- | --- | --- |
| `land_cover` | 类别码（指示量） | **绝对重叠面积 m²** ÷ `cell_area_m2` | `2.8e-08`–`5.2e-08` |
| `building_height` | ANBH，强度量 | **绝对重叠面积 m²** | `4.9e-09`–`3.1e-08` |
| `built_surface` | m²，**已含面积** | **重叠比例**（无量纲） | `1.8e-08`–`3.3e-08` |
| `population` | 人数，**已含面积** | **重叠比例**（无量纲） | `8.4e-10`–`3.0e-09` |

若对 `built_surface`/`population` 也乘绝对面积，实测偏差跳到 `6.5e+03`–`7.5e+03`
（**双计面积**）。故修法是：`land_cover`/`building_height` 改用**绝对面积**，
`built_surface`/`population` **保持比例**不动，且 `land_cover` 的分母统一为
`cell_area_m2`（与其余三个 role 同一个量），不再自造 `sin_band × native_cols`。

### ④ 现有测试完全没有覆盖该模块

`data/preprocess/urban_static_recheck.py` 没有被任何测试 import。缺陷不是被测试抓住的，
是被一次真实运行的 `passed: false` 抓住的——而那次运行之所以失败，是因为我在**另一条路径**
（聚合器）上先踩过同一个坑并写了断言。⇒ 该模块需要一个**会在量纲错时失败**的用例。

### ⑤ 修复后对真实 attempt03 的复算（实测，本节是本文的核心证据）

把修复后的第三套实现直接跑在 `outputs/r7_urban_static_covariates_attempt03` 的真实瓦片上，
对同一组 3 格 × 4 字段共 12 项比较：

| 字段 | 修复前 `relative_delta` | 修复后 |
| --- | --- | --- |
| `lc_built_frac` | `7.9e+05`–`9.1e+05` | `2.8e-08`–`5.2e-08` |
| `building_height_m` | `2.7e-05`–`4.3e-04` | `4.9e-09`–`3.1e-08` |
| `built_surface_frac` | `1.8e-08`–`2.4e-08` | `1.8e-08`–`2.4e-08`（未动） |
| `pop_density_km2` | `8.9e-10`–`6.7e-09` | `8.9e-10`–`6.7e-09`（未动） |

`max_relative_delta` 由 **`913872.595` → `5.154e-08`**（容差 `1e-05`）。

⇒ **聚合器自始至终是对的**；`passed: false` 的唯一原因是 recheck 自己。
这一步必须先做对再宣布修复，否则「修 recheck」可能只是在迁就一个错误的聚合结果。

### ⑥ `publish()` 记录但不阻断
`scripts/r7_urban_static_round.py:488-489` 把 `recheck.passed` 写进 manifest，
`_publish` 无任何条件分支。于是 attempt03 在 `passed: false`、`max_relative_delta = 9.1e+05`
的状态下**照常发布了 `static_identity`**。这与"不可证明即不可声明"的契约相反：
一个明确的**内部一致性检查失败**不应产出可被下游引用的身份。

### ⑦ 被丢弃的具名层没有理由

`verify_endpoints()` 在 `scripts/r7_urban_static_round.py:137-149` 构造 dropped 记录时
**不含 `reason` 键**；`DROPPED_LAYERS` 里那两条手写的才有。attempt03 的
`endpoint_report.json` 实测：

```
dropped: ghsl_built_s_2000 | reason= None
dropped: ghsl_built_s_2005 | reason= None
dropped: ghsl_built_s_2010 | reason= None
```

R-008 要求失败留下 `failed-no-fallback` 审计记录；`reason: null` 的记录无法通过审计——
读者无法判断是"不可达"、"被授权排除"还是"下载失败"。这三层在 attempt01/02 里是可达的
（13 层），attempt03 掉到 10 层却没有任何解释。

## Decision

### 第 1 条：recheck 权重按 role 分派，`land_cover` 分母统一为 `cell_area_m2`

`urban_static_recheck.py` 改为：`land_cover` 与 `building_height` 用**绝对球面重叠面积**
（`R² · Δlon_rad · [sin(over_top) − sin(over_bottom)]`，其中 `Δlon_rad` 按重叠列比例缩放）；
`built_surface` 与 `population` 保持**重叠比例**。`land_cover` 的 `total` 直接取
`cell_area_m2`，删除 `sin_band × native_cols` 这一自造量纲。同时在模块 docstring 里写明
**四个 role 的权重不同及原因**，防止下一次"统一化"把③的表抹掉。

### 第 2 条：补单测，必须能在量纲错时失败

新增测试合成一个**已知解析解**的小瓦片（例如 2 个 0.0625° 原生像素覆盖 1 个 0.25° 目标格，
或直接用真实尺度的小窗），断言：

- `land_cover` 各类别分数之和 ≤ 1（无量纲、是比例，不是像素计数）；
- 全格为单一类别时分数**恰为 1.0**（这是最能抓住 `sin_band × native_cols` 分母的断言，
  因为该分母会给出 ~`9e5`）；
- `building_height` 在南北两格同高时给出同一个均值，而在"北格高值、南格低值"时给出
  **面积加权**结果（与不加权结果可区分）；
- `built_surface`/`population` 在把瓦片值除以 4 后结果也除以 4（证明是线性求和，
  没有被二次面积加权）。

### 第 3 条：`publish()` 对 recheck 不通过 **fail-closed**

`publish()` 在调用 `_publish` 之前检查 `recheck_receipt["passed"]`，为 `False` 时
**抛 `SystemExit`** 并打印失败项与 `max_relative_delta`，不产出 `static_identity`。
已发布的历史 attempt 不改写（它们如实记着 `passed: false`）。

### 第 4 条：被丢弃的具名层强制非空 `reason`

`verify_endpoints()` 为每个 dropped 记录填入 `reason`（不可达 → 明确写出探测结果；
部分瓦片缺失 → 不算 dropped，仍按 `reachable-with-absent-tiles` 留在 layers 里），
并在 `freeze()` 与 `_publish()` 入口做一次断言：任何 `dropped_layers` 条目
`reason` 为空即 `SystemExit`。这是 fail-closed 的守卫，不是文档约定。

### 第 5 条：不改已发布数值，attempt03 标注为不可用于科学

本次修复**只改 recheck 与守卫**，不动聚合器、不重跑下载、不改 attempt03 的任何字节。
attempt03 在证据页标注为：

> 其 `recheck_receipt.json` 记着 `passed: false`（**是 recheck 的缺陷，不是聚合的缺陷**：
> 用修好的第三套实现复算同一批真实瓦片得 `max_relative_delta = 5.154e-08`，见 ⑤），
> 且被丢弃的三层缺 `reason`。按第 3 条的新守卫，这样一个 attempt **不会再被发布**。
> 已发布的 `static_identity` 与字段**不被本文改写**（改写会破坏已登记的身份链），
> 但在重新派生出一个 recheck 通过的 attempt 之前，**不得用于科学声明**。

这一条是本文唯一"不如人意"的地方，必须写清：字段本身经独立复算是对的，
使它不可用的原因是**当时的闸门没有拦住**，而不是数值错了。二者不能互相抵消——
一个当时没通过自身校验、事后因"校验器有病"而被追认的产物，其可信度来自事后复算，
不来自当时的闸门；把两者混为一谈就等于让 `passed: false` 变成可协商的。

## Consequences

- recheck 从"会放过量纲错的橡皮章"变成真闸门：修复后在**真实 attempt03 瓦片**上
  12 项比较全部 ≤ `5.2e-08`（`max_relative_delta` `913872.595 → 5.154e-08`）。
  反证已实测：把修复前的模块放回去跑新测试，**4 个用例失败**（一格全为单类别时
  `fraction` 不是 1.0、两类别之和不是 1.0、`building_height` 不是面积加权、
  跨格界列不是按重叠比例计入）；换回修复后 14 个用例全过。所以这些测试不是装饰。
- `publish` 的 fail-closed 会把 attempt03 这类产物挡在发布之外。**代价**：合法的
  "recheck 因容差边缘失败但聚合正确"的情形也没有出路——这是刻意的，先证明再发布。
- 丢弃层带理由后，attempt03 的 3 层缺口必须有解释；若确认是可达性退化，那是需要
  单独调查的数据问题，不是本次能修的。
- 不改 `model_code_sha256`（不动 `model/**`），故不影响 `load_checkpoint` 与 R2–R5 队列。
- 0 GPU-h，无网络，无付费资源。

## 与既有决策的关系

- **0071**：本 ADR 执行其第 1 条要求的"第三套实现"所**遗留的缺陷修复**。0071 立了第三套
  实现的原则，但该实现本身带量纲错；本文把它修对，并把"单测覆盖"补上。
- **0070**：其出口 E4 是"recheck 最大相对偏差 ≤1e-05"。attempt03 实测 `9.1e+05` ⇒ **E4 未达**。
  本 ADR 不修改 E4，只说明未达的原因是 recheck 自身缺陷而非聚合错误（②已证）。
- **0063**：授权范围不变；仍在 `outputs/` 下、仍不重建 store、仍不写 state 通道。
- **R-008**：第 4 条把"失败留审计"的要求从文档落到代码守卫。
- **R-006**：不重新冻结协议；本次无下载。
