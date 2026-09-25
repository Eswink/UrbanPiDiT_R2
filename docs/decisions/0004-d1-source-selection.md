# 0004 D1 数据源选择：Earthmover icechunk `spatial` namespace

- **日期**：2026-09-25
- **状态**：accepted

## Context（背景）

#63 的 D1 段（2016-01-01 起连续 30 天、6 小时节奏 = 120 时次、17 通道、
65×65 ROI 27–43N/107–123E）需要显式下载授权才能执行，此前记为 BLOCKED。
两个候选源在冻结成本表里都 ≤64 GiB 上限，无法据此区分优劣，因此先做了实测
测速（`docs/R7_D1_READ_SPEED.md`，回执在 `outputs/speedtest_*.json`）：

| 源 / 布局 | 每时次 | D1 估计 | 网络量 |
| --- | --- | --- | --- |
| Earthmover `spatial` | 8.78 s / 25.15 MiB | ≈17.6 min | 2.95 GiB |
| ARCO `wb13-6h` | 53.77 s / 489.9 MiB | ≈107.5 min | 57.4 GiB |
| Earthmover `temporal`（生产 `earthmover_pilot.py` 所用） | 两次 900 s 超时未达稳态 | 不可用 | — |

关键几何事实（决定了上表）：Earthmover 快照同时提供两个 namespace，chunk 相反——
`spatial` 是「一时次一张全球场」`(1, 721, 1440)`，`temporal` 是「364 天 × 12×12 空间块」
`(8736, 12, 12)`。v2 的 65×65 ROI 在 `temporal` 下需约 612 chunk/时次（空间利用率
灾难），在 `spatial` 下只需 17 chunk/时次。

**推论**：现有生产读取路径（`earthmover_pilot.py`，硬编码 12×12 tile + `temporal`）
对 v2 的 65×65 范围**不适用**；`arco_regional_bounded.py` 走的是 ARCO，也不适用于
Earthmover。所以选择任一源都需要新代码——差别只是写多少。

## Decision（决定）

**D1 使用 Earthmover icechunk `spatial` namespace**（快照 `ZFKDHBCTBVHVXM3BQFV0`，
只读匿名），并新增一条 `spatial` 读取路径，复用 `arco_regional_bounded.py` 已在
使用的**发布契约与守卫**（budget/deadline/ROI 校验/有限性检查/`BUILD_COMPLETE`/
`failed-no-fallback` receipt/SHA256），而不是另建一套数据系统。

理由：省约 1.5 小时与 54 GiB 网络；且 `spatial` 的「全球场/时次」chunk 与
`arco_regional_bounded` 的既有读取假设一致，新代码可大量复用其校验逻辑。

## Consequences（后果——含代价）

**变容易的：** D1 从「约 1.8 小时 + 57 GiB」降为「约 18 分钟 + 3 GiB」，重试成本低；
Earthmover 快照与 ARCO 是独立身份，v2 的来源多样性更好（两个源互为交叉验证）。

**变难 / 代价（Trade-off，如实列出）：**

- **必须新写并维护第二条读取路径**（`spatial` namespace）。这与 #63 原文
  「优先复用、不重新发明已有 data contract」存在张力——处置是复用发布契约与
  守卫逻辑，只新增 chunk 寻址部分，并在文档中写明两者关系。
- **空间利用率低**：`spatial` 每时次读入 4.15 MiB 全球场 × 17 = 约 25 MiB，
  而 ROI 只需 65×65×17×4 B ≈ 0.29 MiB（约 0.4% 利用率）。绝对量仍很小，
  但这是"选它是因为便宜"的诚实表述——不是因为高效。
- **`temporal` namespace 不再是本项目的可选路径**：若将来 ROI 缩回 12×12，
  应重新评估（那是 `earthmover_pilot.py` 的领域，不要混用）。
- **测速的绝对时间未受控**（本机负载/网络抖动）：约 17.6 min 是估计而非保证；
  6 倍源间差距远超噪声，但单源绝对时间应视为量级而非精确值。
- **D1 仍不产生任何科学结论**：它是工程段（构窗/单位/IO 验证），#63 明文
  「不能拿这一步报告期刊技巧」。
- 本次授权**仅覆盖 D1 的 30 天**；D2（多季节/全年扩展）与 2021 test 封存审计
  仍需另行授权，不在本决定范围内。

**备选方案与否决理由：** 选 ARCO wb13（工程改动最小，但慢 6 倍、网络大 19 倍，
且 #63 的候选设计希望避免单一源依赖）；继续用 Earthmover `temporal`
（对 65×65 实测不可行）；扩大 ROI 到全球（超出冻结范围，否决）。
