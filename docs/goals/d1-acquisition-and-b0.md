# D1 获取与 #64 B0 可学习性（S1 收口 + S2 起步）

本文件是 goal 模式目标的长文源（R-034）。objective 有 4000 字符硬上限，细节在此。
在线权威来源是 GitHub #63/#64 与 #59（EPIC）；冲突时以 GitHub 为准。

**状态：尚未执行。** 前置已就绪：源选择已冻结（决策 0004）、测速已实测
（`docs/R7_D1_READ_SPEED.md`）、D1 授权已由用户给出。

---

## 0. 前置事实（已核实，2026-09-25）

| 项 | 值 |
| --- | --- |
| HEAD = origin/分支 | `38d744d`（main 在其后一提交内，ff 可推） |
| D1 源（决策 0004） | Earthmover icechunk **`spatial`** namespace，快照 `ZFKDHBCTBVHVXM3BQFV0`，只读匿名 |
| 实测速率 | 8.78 s / 25.15 MiB 每 6 小时时次 → D1（120 时次）≈17.6 min / 2.95 GiB |
| D1 范围（#63 冻结） | 2016-01-01 起连续 30 天、6 小时节奏、17 通道、65×65 ROI 27–43N/107–123E |
| 预算 | 新产物 ≤16 GiB、decoded ≤64 GiB（D1 实测约 3 GiB，远低于上限） |
| 生产读取路径 | `earthmover_pilot.py` 走 `temporal` 且硬编码 12×12 tile → **对 65×65 不适用**，须新增 spatial 读取路径 |
| 可复用契约 | `arco_regional_bounded.py` 的 budget/deadline/ROI 校验/有限性/`BUILD_COMPLETE`/`failed-no-fallback`/SHA256 发布契约 |
| 通道顺序来源 | `data/preprocess/r7_era5.DEFAULT_R7_ERA5_CHANNELS`（17 通道，**断言，不重写**） |
| 现有测试 | 954 passed / 3 skipped；34 条阻断规则 0 违规 |

---

## 1. 本 goal 的范围（两件事，顺序不可换）

### A. D1 获取（收口 #63 的获取部分）

1. **新增 spatial 读取路径**：一个模块，按 `spatial` namespace 取 17 通道；
   复用 `arco_regional_bounded` 的发布契约与守卫，**只新增 chunk 寻址**，
   不另建数据系统（#63 原文要求优先复用）。
2. **冻结范围不得改动**：ROI/年份/通道顺序/节奏全部来自 `read_plan_frozen`，
   从模块读取而非在探针里重写；任何偏差即失败。
3. **写新目标路径**（R-001 排他创建、R-004 不得写 `data/raw|interim|processed`）；
   成功后必须有 `BUILD_COMPLETE.json`；失败必须留 `status='failed-no-fallback'`
   且 `synthetic_fallback=False`（R-008）。
4. **前置只读 preflight 报告 → 再 `--write`**（`.agents/skills/real-data-acquisition`）。
5. 记录：源身份（快照 id + 变量/层清单）、时间覆盖、四 UTC 起报时刻、17 通道单位、
   payload SHA256、本地产物 SHA256、实际网络字节（实测，不用 decoded 代替）。
6. **D1 是工程段**：#63 明文「不能拿这一步报告期刊技巧」。产物标
   `scientific_claim: false`。

### B. #64 B0 可学习性探针（S2 起步）

`#64` 的 B0：**固定**真实训练集 1/8/32 个窗口，禁用随机正则，保持物理 loss 与
归一化契约，验证最小 native/window/generic 模型能显著降低这些固定样本的误差。

- 检查：每通道 loss、反归一化、输入历史索引、lead 对应、**梯度是否抵达
  encoder/solver**。
- 初始诊断可限定单步 +6h。
- **若无法拟合**：先查实现/归一化/优化器，**不跑长训**。
- 记录：目标、最大更新数、停止条件。**过拟合只证明可学习性，不证明泛化**。
- 起点：现有 `dim≈32/depth=2` 级别的极小配置（不冲参数），用 D1 的真实窗口。

---

## 2. 明确的非目标（本 goal 不做）

- 不做 D2（多季节/全年扩展）——需另行授权。
- 不做 2021 test 封存审计——需另行授权。
- 不训练 1–5M 强基线（那是 #64 B1，依赖 B0 通过）。
- 不跑 17 条 tag-gated 实验 workflow（用户决定）。
- 不改既有 `earthmover_pilot.py` 的 12×12 语义（那是另一条路径）。
- 不动 `data/raw|interim|processed` 与归档；不做 force push/合并/release。

---

## 3. 预算与停止条件

- 获取：≤30 分钟墙钟、≤3.5 GiB 网络（实测估计 17.6 min / 2.95 GiB，留余量）；
  decoded 上限 64 GiB 不触及。
- B0：CPU 亦可，GPU 若用则 ≤0.5 GPU-hours（上限 4）。
- **停止条件（满足即收尾并如实报告）**：
  (a) D1 产物落盘且通过 `BUILD_COMPLETE` + 离线重放验证，**且** B0 在
      1/8/32 窗口上给出可解释的 loss 下降（或明确的"不可学习"诊断与原因）；
  (b) 预算到界或遇到需用户决策的分叉（超预算、数据异常、需扩量）——
      先报成本与范围选项，不静默扩量；
  (c) B0 显示无法拟合且诊断指向需要新实现决策（记录下来，不无限试）。
- **B1 及以后不是本 goal 的完成条件**：B0 通过只解锁 B1，不等于科研结论。

---

## 4. 纪律（沿用既有契约）

- 数据：`real-data-acquisition` skill 流程；单位不符即停止不得猜换算；
  禁止把插值/合成当真实；失败留审计不静默回退。
- 诚实：负结果与 BLOCKED 是合法终态；不删坏变量、不挑 seed、不用 test 调参、
  不为好看放宽判据；被取消/跳过/排队的运行不算通过。
- 工程：#60 之后的比较器是唯一比较口径；新文件守命名规则（snake_case、禁词、
  `from __future__ import annotations`、显式 `encoding`、pathlib）；治理文件 `git add`。
- GitHub：ff 推送 main 已放行；合并/release 需用户授权；issue 评论 401 → 证据写
  `docs/` + `Closes #N` 经 main ff；hook 是跨行文本匹配器（避免 "git merge"+"main"
  或 "merge-base"+"origin/main" 同现于一条命令）。
- 外部源码借用前记录 revision/license/文件与改动表。

---

## 5. 每轮固定动作

1. 记录起点 `git rev-parse HEAD`；
2. A（D1 获取）优先——它是 B 的前置；
3. 实现→测试→自查→修→再测；
4. 全量 `pytest -q` + `check_conventions.py`（34 条 0 违规）；
5. commit（`type(r7): summary (#N)`，不带实验标签）+ push 分支；
6. ci.yml 自跑、记录 run id、不等待不作门槛；
7. 结果写入 `docs/R7_*.md` 与 `docs/R7_TASK_QUEUE.md`，绑定精确 SHA；
8. 判定 DONE / BLOCKED（写出真实依赖）；
9. 下一轮。

---

## 6. 收尾报告必须包含

本轮 issue 编号与起止 SHA；改动面；实测验证（命令/计数/run id）；
**D1 的源身份与产物 SHA256、实际网络字节、墙钟时间**；B0 的 1/8/32 窗口结果
（含每通道 loss 与梯度到达性）；预算消耗（GiB / GPU-hours）；DONE/BLOCKED 判定依据；
负面结果与未做的事；下一项 issue（预期是 #64 B1）及其第一个具体动作。
不得只说「继续推进」。

---

## 7. 与 #63/#64 原文的对应

| 本 goal 动作 | 出处 |
| --- | --- |
| D1 连续 30 天、不报期刊技巧 | #63 D1 |
| 元数据先行、成本表、实测网络量 | #63「读取成本与数据契约」1/2 |
| `BUILD_COMPLETE`/schema/receipt/checksum | #63 交付 5 |
| 四 UTC 起报、+72h 可用、不跨 split 拼窗 | #63 D1/D2、验收 |
| 1/8/32 窗口、禁随机正则、查梯度到达 | #64 B0 |
| 不冲参数、不跑长训 | #64 B0/B3 |
