# 0047 一轮内的双卡 seed 分配（一卡一 seed）与邻居增长 reserve

- **日期**：2026-10-12
- **状态**：accepted
- **代码 SHA 范围**：`58706dd`（起点）→ 本次提交
- **依据证据**：`docs/decisions/0026-shared-gpu-coresidency-policy.md`（共驻只读门槛、不信号邻居、独占需具名授权）；
  `docs/decisions/0042-parallel-research-workflow.md` 第 4 条（GPU 由统一父调度按共驻余量串行）；
  `docs/decisions/0045-pinned-seed-reuse-across-attempts.md`（跨 attempt 复用、失败全额记账）；
  `docs/rules/gpu-resources.md`（R-054 正文）；
  `scripts/study_r7_b3_scale.py`（既有先例：one independent seed per 3090，no DDP）；
  `outputs/r7_s3_capacity_depth_20261012_attempt03/`（seed 41 的 OOM 现场与邻居增长观测）；
  `training/r7_device_selection.py`；`tools/r7_round_stages.py`；`tools/r7_round.py`；
  `scripts/study_r7_s3_capacity_depth.py`；`tests/test_r7_device_selection.py`；
  `tests/test_r7_round_stages.py`；`tests/test_r7_s3_capacity_depth.py`

## Context

capacity 轮（trunk depth 4→8）此前把每个驱动**硬编码**到一张卡（`wide.GPU_UUID = GPU-408ad137…`，
即 GPU0），并且一次只跑一个 seed。两个后果：

1. **第二张卡全程空转。** 本机是 2×RTX 3090；一张卡的 seed 跑 8h 上限时另一张只做别的无关事。
   用户明确要求「把两张卡都用起来，默认单张，保留单张」。
2. **「被邻居挤爆」发生了，而且门槛看不见它。** `attempt03` 的 seed 41 在 spawn 时只读门槛
   **通过**（free 15.29 GiB ≥ 要求的 9 GiB），却在 32.5 分钟后 CUDA OOM：同一张卡上的邻居
   作业从 ~8.1 GiB **长到 14.32 GiB**（+~6.2 GiB），我们自己的进程稳定在 9.18 GiB，合计
   23.5/23.56 GiB 卡满。**spawn 时刻的一次只读读取，原理上无法约束邻居的未来增长。**

用户同时要求：**不改任何训练配置**（保住与对照臂 `s3-wide-interior-dose` 的严格单因素），
只加「余量」与「双卡」。这排除了「打开 activation checkpointing 把峰值压下去」这类改动。

## Decision

**采用：把「一轮内的 seed」按卡分配——每张卡同一时刻至多一个本轮的 worker，一卡一个 seed、
并发跑；默认双卡，`single` 模式逐位保留旧行为；并在余量门槛上增加一条声明式
`neighbour_reserve_bytes`。不引入 DDP。** 具体：

1. **分配**（`training/r7_device_selection.py`，新增；不动共享模块与 `wide.GPU_UUID`）：
   冻结块记 `device_mode`（`single`/`dual`）、`gpu_uuids`（列表，声明的卡排第一）与
   `neighbour_reserve_bytes`。seed 的**偏好卡**是 `gpu_uuids` 上的一次**确定性轮转**
   （`seed_order`，位置取 round-spec 的 seed 序号），所以三个 seed 分两张卡时不会全挤在 GPU0。
   `single` 时 `gpu_uuids == [wide.GPU_UUID]`，语义与今天一致。
2. **门槛**：`required = max(估计峰值, 实测峰值) + margin + reserve`；逐张卡只读探测，取第一张通过的，
   并**保留每张卡的观测**（哪张、差多少）。全部不过 ⇒ `headroom-blocked`，**有界等待后重问**，
   从不给邻居发信号（R-054）。
3. **并发编排**：`tools/r7_round.py stage-seeds` 用有界父调度跑完剩余 seed，**每张卡至多一个 worker**；
   任一 seed 真失败即停止派生新的（已跑的保留、全额记账）。`stage seed <N>` 保留给单卡/手工。
4. **不自动重试失败 seed**（决策 0026/0045 不变）：余量不足仍全额计费。

### 与决策 0042 第 4 条的边界（本决定最重要的一条）

0042 第 4 条写的是「GPU 仍由统一父调度按共驻余量**串行**」。本决定**不推翻**它，而是明确它的**适用范围**：

- 0042 要防的是**互不依赖的研究线**同时抢卡（多个父、各自以为卡是空的）；
- 本决定约束的是**同一个父调度内部、同一轮的独立 seed**：卡分配由**同一个**父调度做，
  每卡**至多一个** worker，且仍然逐次过只读余量门。并发的来源是「一张卡一个 seed」，
  不是「多开一条研究线」。
- 因此准入七 gate 仍然逐轮适用（假设独立/产物隔离/…）；**并行线**的准入没有被放宽，
  放宽的只是「一轮内 seed 的卡排布」。

## Consequences

**正面**：

- 第二张卡进入本轮，两个 seed 的墙钟重叠，同一轮的**墙钟**约减半（GPU-h 不变）。
- 「邻居增长」这一失败模式从「不可见的意外」变成「冻结的、可审计的声明」：reserve 写进
  `protocol.json` 并参与 digest，事后能查这一轮到底按什么余量要求开的。
- 历史臂逐位不变：`gpu_uuid` 仍在，`gpu_uuids` 缺失时回退旧的单卡语义；
  `training/r7_s3_v3_screen.py` 的 `gpu_gate` / `pin_declared_gpu` 一字未改。

**负面（代价，必须记住）**：

- **reserve 是判断，不是保证。** 6 GiB 取自**一次**实测的邻居增长（+6.2 GiB）。邻居若涨得更多，
  seed 仍会在训练中途 OOM；这类失败如实登记为资源失败，**不得**写成「已解决」。
- **并发改变墙钟，也改变资源争用。** 两个 seed 同时跑会共享 CPU / PCIe / 磁盘，单 seed 墙钟会变长；
  GPU-h 不变。**墙钟不是科学判据**，逐 seed 的墙钟对比只在同一 device_mode 内可比。
- **父调度是新的编排路径。** 「每卡至多一个 worker」由反证测试锁住（同卡第二个 worker 必须被拒），
  不靠注释；但它是单进程内的线程池，**不跨进程**——若有人绕过 `stage-seeds` 直接手工起两个 worker，
  本保证不生效（靠 R-054 的余量门兜底，不靠分配器）。
- **`single` 与 `dual` 的 receipt 不可直接混用**：device_mode 进了协议 digest，两种模式的 attempt
  是不同身份，不能把 `single` 的 receipt 拿来当 `dual` 的已完成 seed。
- 双卡让「本轮驱动身份」再次改变 ⇒ 下一次运行必须是**新 attempt**（attempt05），旧 attempt 原样保留。

### 被否决的替代方案

- **DDP（两卡数据并行跑一个 seed）**：`scripts/study_r7_b3_scale.py` 已经实测过——两张 3090
  无 NVLink，DDP 的通信惩罚从未收回；而且 DDP 会改变有效 batch / 优化轨迹，破坏与对照臂的单因素性。
- **把峰值压下去（activation checkpointing / 改 batch）**：用户明确要求**不改训练配置**；
  这会动到对照臂的同一个因子。
- **给邻居发信号 / 冻结邻居腾卡**：R-054 与决策 0026 明令禁止；本决定只做只读探测与有界等待。
- **继续单卡、只加 reserve**：reserve 能降低 OOM 概率，但第二张卡仍空转，且「一次只跑一个 seed」
  让本轮的墙钟翻倍——用户已明确要双卡。
