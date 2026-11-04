# 0085 — Stage A1/A2 的读法改造：一次跨帧读 + 每 store 一次的元数据 + worker 取数

- **日期**：2026-11-04
- **状态**：accepted
- **范围**：`data/r7_zarr_dataset.py`、`data/r7_long_rollout_dataset.py`、`data/r7_store.py`
  的 `validate_record` 增加可选 `time_ns` 参数、`training/r7_long_rollout_runner.py` 的
  `_run_updates` / `_update` / 新增 `_index_stream` 与 `_BatchLoader`
- **依据**：决策 0084（方向扩围与 A 的判据拆分）、`docs/plans/0028-throughput-physics-campaign.md`
  §2 A1/A2、本轮实测（见 §Context ②）

## Context

**① 计划指出的两个病灶在真实 store 上被复现并量化。** 对
`outputs/r7_s3_fy_multiyear_20261023_attempt01/store/cache.zarr`（`state (34084,17,65,65)`、
chunk `(16,17,64,64)`、`time_ns` 为独立的 `[T]` 数组）逐段计时：

| 项 | 实测 | 说明 |
| --- | ---: | --- |
| 14 个时刻逐帧读 | `0.352 s` | 每帧一次 zarr 往返 + 每次重解所触的 chunk |
| 14 个时刻一次跨步读 | `0.0228 s` | **15.4×**，且 `array_equal` 与逐帧读**逐位相同** |
| `normalization()` | `6.62 ms` | 每样本调一次（原实现每样本两次） |
| `validate_record` 单条 | `15.09 ms` | 其中 9 次 `root['time_ns'][i]` 标量读 |
| `root['time_ns']` 全列 | `1.71 s` | 一次读完整列，而 `validate_store` **已经读过了** |
| 标量 `time_ns[i]` | `3.22 ms` | 每次标量读是**一次完整往返** |

⇒ 建集 32,142 条 × 9 次标量读 × 3.22 ms ≈ **930 s 上界**，实测 **493.81 s**：
`validate_store` 已经把整列读进来（`validate_store` 返回 `raw`），`_discover_long_windows`
也拿到了它，却没有把它传给 `validate_record`，于是每条记录又重新标量读 9 次。

**② 三项改动，全部落在「怎么读」而不是「读什么」。**

- **A1a**：`validate_record(root, record, time_ns=None)` 增加可选整列；传了就按索引取值，
  不传就照旧逐标量读。**没有调用者被强制改动**。
- **A1b**：`ZarrAtmosWindowDataset` 把 `time_ns` 与 `normalization` 缓存在 `_aux` 里，
  每 store 一次；`_discover_long_windows` 把 `validate_store` 已经返回的 `raw` 传给
  `validate_record`。
- **A1c**：新增 `span` / `read_normalized` / `sample_from_frames`，让长 rollout 的
  14 帧窗口**一次读完**。非连续窗口（当前 store 产生不出来，但没有检查禁止）退回逐索引路径，
  **不被拒绝**。
- **A2**：`_index_stream` 在本进程按 `randperm(seed + epoch)` + 游标**预先物化**整轮索引流，
  worker 只消费这些索引；`_BatchLoader` 每次 `batch(indices)` 都断言
  `stream[consumed:consumed+n] == indices`，**偏离即 raise**。

**③ 取数顺序留在主进程，是 A2 可验收的前提。** 用 `DataLoader(sampler=stream)` 会让
批次边界由 sampler 自己决定，而 epoch 边界是本进程的游标算术——两者一旦不同，
「同样的 200 步」就不再是同一个序列。物化索引流让批次边界**由构造重合**，而不是靠推断。

## Decision

**1. 采用上述 A1a/A1b/A1c/A2，判据是逐位的**：`window_sha256`、`usable_windows`、
`excluded_sample_ids` 逐字节不变；6 个抽样窗口的**全部字段**（张量 sha256、字符串、列表）
与 HEAD 代码零差异；60 个 batch 的逐 batch sha256 在 `workers=0` 与 `workers=4` 之间相同。

**2. `_finite_state` 的每步全检改为周期 + 每 checkpoint（A3），另立 0086。**

**3. worker 数取 4。** 实测 batch 4 下 4/8/12 个 worker 分别
`1.1193 / 1.1420 / 1.2604 s/update`：4 是最好，更多只增加争用。这是实测，不是默认值。

**4. 逐位验收必须写明线程数。** 见 0084 负面后果 3：同一配置连跑两次 loss 已不同
（GPU 上 119/200 个 update 有 1 ULP 差），单线程下才逐位相同。因此本 ADR 的验收条件是
「**固定 `torch.set_num_threads(1)` 下逐位相同**」，且线程数要进 receipt。

## Consequences

**负面后果（代价，必须如实列出）**

1. **`_aux` 缓存假设 store 在进程内不变**。它按 store 路径做键，若有人在同进程内改写 store
   字节（本仓无此路径，且 store 是 `mode='r'` 只读打开），缓存会陈旧。`mode='r'` 是实际护栏。
2. **`_index_stream` 物化整轮索引流**：16,071 updates × batch 4 = 64,284 个 int，
   128,528 updates 时 514,112 个，约 4 MB 的 Python 列表。可接受，但不是零成本；它换来的是
   批次边界可断言。
3. **`workers>0` 时每个 worker 进程都持有自己的 store 句柄**。4 个 worker 各开一次
   zarr group，内存与文件句柄都 ×4。实测这些进程的只读打开不写盘。
4. **未量化**：本轮没有测 `persistent_workers=True` 的启动摊销对短 run（<100 updates）的影响；
   短 run 上 worker 启动可能反而更慢。Stage B 的四个点都是长 run，不受影响。
5. **A 的出口未达到**（batch 4 实测 1.0947 > 1.0），见 0084 负面后果 1；本 ADR 不因此改判据，
   也不把 batch 6+ 冒充成出口。

**正面后果（实测）**：每样本 `0.5091 → 0.0570 s`（8.9×）、建集 `493.81 → 187.66 s`（2.6×）、
batch 4 每 update `3.3479 → 1.0947 s`（**3.06×**，HEAD 同口径实测），数据构造降到
**1.2%** 的 update 时间。修完之后**计算成为主项（97%）**，这与计划 0028 §0.4 的
「95% 在 I/O」归因相反，已记入 0084 §Context ③。

**否决**：把 `validate_record` 的整列参数设成必填（会强制所有调用者改，且没有收益）；
用 `DataLoader(shuffle=True)` 让 sampler 决定顺序（epoch 边界不可断言）；
把非连续窗口直接拒绝（会缩窄合法输入集，越过「只改读法」的边界）；
缓存 `sample` 对象本身（会跨 epoch 复用同一批张量，改变训练语义）。
