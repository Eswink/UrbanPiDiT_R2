# D1 数据获取测速（2026-09-25/26）

针对 #63 冻结的 D1 范围（2016-01-01 起连续 30 天、6 小时节奏 = **120 个时次**、
17 通道、65×65 ROI 27–43N/107–123E）实测两个候选源的读取速度与网络量。
`read_plan_frozen` 的 `http_bytes` 此前恒为 null，本文补上实测值。

**方法**：真实网络字节取自 `/proc/net/dev` 非环回接口差值；每源独立进程、
先预热 1 个时次再计入稳态；拒绝任何非有限值（顺带验证数据可用）。所有源匿名只读。

## 结论

| 源 / 读取布局 | 稳态每时次 | D1（120 时次） | 判定 |
| --- | --- | --- | --- |
| **Earthmover `spatial` 布局** | **8.78 s / 25.15 MiB** | **≈17.6 min / 2.95 GiB** | **最快最省** |
| ARCO `wb13-6h`（12×12 chunk 但全球场/时次） | 53.77 s / 489.9 MiB | ≈107.5 min / 57.4 GiB | 慢 6 倍、大 19 倍 |
| Earthmover `temporal` 布局（生产 `earthmover_pilot` 用的） | **>300 s/时次（两次超时未完成）** | 不可用 | 对 65×65 灾难性 |

三者都在 64 GiB 第一阶段 decoded 上限内（ARCO 57.4 GiB 逼近），但**时间是硬差别**：
Earthmover spatial 约 18 分钟，ARCO 约 1.8 小时。

## 为什么 temporal 布局不可用（本次最重要的发现）

Earthmover 快照同时提供两个 namespace，chunk 几何相反：

| namespace | chunk | 覆盖语义 |
| --- | --- | --- |
| `{single,pressure}/spatial` | `(1, 721, 1440)` / `(1, 1, 721, 1440)` | 一个时次一整张全球场 |
| `{single,pressure}/temporal` | `(8736, 12, 12)` / `(8736, 1, 12, 12)` | 364 天 × 12×12 空间块 |

- `temporal` 是为**小区域**设计的：chunk 在时间上铺 364 天、空间只 12×12。
  生产 `earthmover_pilot.py` 的 12×12 tile 正好一个 chunk 全覆盖，所以高效。
- **#63 冻结的 ROI 是 65×65**：在 12×12 空间 chunk 下需要 `ceil(65/12)²=36` 个空间
  chunk × 17 数组 = **612 个 chunk/时次**，每个 5.03 MiB → 约 3.1 GiB/时次，
  即使压缩后仍远超任何合理预算。这就是两次探针（各 900 s）连预热都跑不完的原因。
- `spatial` 布局对 65×65 只需 17 个 chunk/时次（全球场读进来、只留 0.4% 像素），
  空间利用率低但绝对量小（25 MiB），因此**总时间反而最短**。

**推论**：若选 Earthmover，必须为 `spatial` namespace 写新的读取路径——
现有生产代码（`earthmover_pilot.py`）硬编码 12×12 tile 且只走 `temporal`，
两者都不覆盖 v2 的 65×65 范围。这是选源时必须计入的工程量。

## 原始回执

- `outputs/speedtest_earthmover_spatial.json`：warmup 25.37 s/96.8 MiB（含 manifest
  预载），稳态 12 时次 105.35 s/301.8 MiB。
- `outputs/speedtest_arco_wb13.json`：warmup 50.59 s/489.6 MiB，稳态 4 时次
  215.08 s/1959.7 MiB。
- Earthmover `temporal`：无回执（两次 900 s 超时，未达稳态测量点）。

## 待确认的取舍（需你决定）

| 选项 | 时间 | 网络 | 工程量 |
| --- | --- | --- | --- |
| **Earthmover spatial** | ≈18 min | 2.95 GiB | 需新增 spatial-namespace 读取路径 |
| **ARCO wb13** | ≈108 min | 57.4 GiB | 现有 ARCO 读取路径可复用/小改 |

两者都在预算内。Earthmover 省 1.5 小时与 54 GiB，代价是一段新的读取代码；
ARCO 慢 6 倍但工程改动最小。**这不是科学取舍，是成本/工程量取舍**——请指定其一，
D1 即可执行（仍受"显式下载授权"约束，#63 的 BLOCKED 状态以此解除）。

## 边界

- 测速本身不是科学结果：`scientific_claim: false`，每份回执均带此标志。
- 探针只读公共匿名源，未下载完整数据集，未写入 `data/raw|interim|processed`。
- 网络量是**实测**而非解码估算（`read_plan_frozen` 的成本模型算的是 decoded；
  实测值通常更小，因为源侧有压缩）。
- 本机网络与其他负载未受控；绝对时间可能有波动，但两个源的相对差距（6 倍）
  远超噪声。temporal 的失败是量级问题，不是抖动。
