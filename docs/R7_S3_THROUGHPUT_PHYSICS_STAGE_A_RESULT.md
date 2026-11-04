# S3 Stage A — 吞吐前置实测证据页

- **日期**：2026-11-04
- **状态**：Stage A **已完成测量**；**出口未达到**（batch 4 实测 1.0848 s/update > 1.0）
- **`scientific_claim: false`**；`test_read: false`；本轮无任何科学读数
- **执行源**：`docs/goals/s3-throughput-and-physics-campaign.md`（Round node S3）
- **决策**：0084（方向扩围与判据拆分）、0085（读法与 worker）、0086（同步守卫）、0087（checkpoint）
- **代码**：本轮改 `data/r7_store.py`、`data/r7_zarr_dataset.py`、`data/r7_long_rollout_dataset.py`、
  `training/r7_long_rollout_runner.py`、`scripts/r7_m3_offline.py`、`scripts/r7_s3_batch_ceiling_{probe,worker}.py`

---

## §1 A1/A2/A3 的逐位验收（**通过**）

判据（决策 0085 §Decision 1，等价于目标长文 §3 的 A1/A2 硬验收）：
`window_sha256`、`usable_windows`、`excluded_sample_ids` 逐字节不变；
固定 seed 的 loss 序列逐位相同。下表是对 HEAD `499adbf` 与新代码逐项的实测。

| 项 | HEAD | 新代码 | 结论 |
| --- | --- | --- | --- |
| `window_sha256` | `e8f22c303c98fc9427e629057cf7f1adf111f57a50977f3bff742ec3089e1136` | **同一个值** | 逐字节不变 |
| `usable_windows` | 32131 | 32131 | 相同 |
| `excluded_sample_ids` | 11 条（年界 t12–t72） | 同一 11 条、同顺序 | 相同 |
| 6 个抽样窗口（index 0/1/7/100/5000/30000）全部字段 | — | — | **0 个差异字段** |

抽样对比的字段覆盖 `coarse_history`、`atmos_target`、`physical_targets`、`future_target`、
`lead_time_hours`、`latitude`、`longitude`、`grid_spacing_deg`、`sample_id`、`init_time`、
`init_*` 日历字段、`history_offsets_hours`、`physical_valid_times`、`init_calendar_year`；
张量按 `sha256(raw bytes)` 比较，即**逐字节**而非近似。

batch 序列（60 个 update、seed 41、batch 2）：`workers=0` 与 `workers=4` 的
**逐 batch sha256 完全相同**（`diff` 无输出，`DIFFLINES=0`）。

**逐位相同的边界（实测，必须连同结论一起引用）**：

| 对比 | 结果 |
| --- | --- |
| CPU、`torch.set_num_threads(1)`、`workers=0` vs `workers=4`、40 updates | **逐位相同**（40/40） |
| CPU、`workers=0` 连跑两次（默认线程数） | **逐位相同**（60/60） |
| GPU、`workers=0` 连跑两次（默认线程数） | **119/200 个 update 不同**，首个差异在 update 2，差 1 ULP |
| GPU、`workers=0` vs `workers=4` | **114/200 个 update 不同** |

⇒ 「逐位相同」在一个**固定线程数**下成立；GPU 上的默认多线程归约本身就不逐位可复现，
与本次改动无关（`workers=0` 连跑两次也不同）。线程数必须记进 receipt。
这一条已写进决策 0084 负面后果 3 与 0085 §Decision 4。

---

## §2 吞吐实测

### §2.1 数据侧

| 项 | HEAD | 新代码 | 倍数 |
| --- | ---: | ---: | ---: |
| 建集（32,142 条 manifest 发现） | **493.81 s** | **187.66 s** | **2.6×** |
| 每样本 `dataset[i]` | **0.5091 s** | **0.0570 s** | **8.9×** |

建集的剩余成本主要是 `validate_record` 仍需按记录做一次 `utc_time_index` 与 split 标签查表；
标量 `time_ns` 往返已消除。

### §2.2 每 update（batch 4，单卡 3090，邻居空闲）

| 配置 | s/update | 相对 HEAD |
| --- | ---: | ---: |
| HEAD `499adbf`（逐帧读、无 worker、每步全检） | **3.3479** | 1.00× |
| A1+A2+A3（4 worker） | **1.0848** | **3.09×** |
| A1+A2+A3（8 worker） | 1.1420 | 3.05×（更差） |
| A1+A2+A3（12 worker） | 1.2604 | 2.85×（更差） |
| A1+A2 只有 A1（batch 2，workers=0） | 1.2834 | — |
| A1+A2 只有 A1（batch 4，workers=4，无 A3） | 1.1813 | — |

**worker 数取 4**：4 最好，8 与 12 更差（争用）。

### §2.3 分段计时（batch 4、40 updates、每段 `torch.cuda.synchronize()`）

| 段 | 秒/update | 占比 |
| --- | ---: | ---: |
| 数据构造（worker 取数 + collate） | 0.0137 | 1.2% |
| H2D | 0.0035 | 0.3% |
| **fwd + bwd** | **1.0977** | **97.8%** |
| 优化器（clip + step） | 0.0079 | 0.7% |
| 全模型同步 | 0.0 | 0.0% |
| 合计 | 1.1228 | 100% |

隔离测量：单次 `fwd+bwd`（batch 4、K=4、12 步全 BPTT、3,286,037 参数）= **1.286 s**；
裸前向（K=4）= **0.0475 s**。`_forward_final` 的两处 `isfinite` 不是瓶颈
（裸前向 0.0475 vs 加两次全量 isfinite 0.0471，差值在噪声内，**未改动**）。

---

## §3 A 的出口判定

**判据（目标长文 §3，冻结后不放宽）**：batch 4 后 `seconds_per_update < 1.0`。

**实测：1.0848 s/update（4 worker、batch 4、300 updates、经真实 `_run_updates`）⇒ 未达到。**

按计划「达不到即停下报告实际倍数，**不改判据**」，本页**只报告，不改判据**，也不把
batch 6+ 的读数冒充成 A 的出口（那是 Stage F 的成本选项）。实测 batch 6 为
1.1302 s/update（200 updates），同样 > 1.0。

**差距归因（实测，不是推测）**：A3 落地后，**97.8% 的时间是 fwd+bwd**。
I/O 已从「95%」降到 1.2%。因此：
- **继续优化读法不会把 1.0848 降到 1.0 以下**——能省的上限是 1.2%。
- 要把每 update 再降 8%，只能在计算侧（更小 K、更大 batch 摊薄、activation checkpointing
  换显存、bf16），而每一条都改 update 语义或算力口径，**不属于 Stage A 的授权范围**
  （A 的定义是「只改读法，不重建 store」）。
- 这一条同时**更正了计划 0028 §0.4 的归因**：那份诊断在本实例上已被实测部分否定，
  见决策 0084 §Context ③。

---

## §4 A6 重扫（A1/A2 之后，`outputs/r7_s3_batch_ceiling_20261104_attempt02/`）

每档一个**全新 Python 进程**、真实 store、真实 treatment manifest（32,131 窗）、
11 条冻结排除集、真实 spec（K=4/depth=4/dim=192/12 步全 BPTT）、4 worker、
数据构造在计时区间内、单卡 3090。`scientific_claim: false`。

| batch | 结果 | 峰值 reserved | s/update（3 次均值） | 旧口径 s/update | 说明 |
| --- | --- | ---: | ---: | ---: | --- |
| 4 | 跑通 | **6,270 MiB** | **1.0145** | ~3.36（插值，未实测） | 旧「3.36」是插值，本行是首个实测 |
| 6 | 跑通 | 9,548 MiB | **0.9814** | — | 三档里唯一 <1.0 |
| 8 | 跑通 | 12,910 MiB | 1.0195 | 5.5579（旧实测） | |
| 10 | 跑通 | 15,556 MiB | 1.1710 | 6.7763 | 共驻上限 |
| 12 | 跑通 | 19,270 MiB | 1.1429 | 7.9367 | 需独占 |
| 14 | 未重扫 | — | — | 9.2579 | 旧口径已测；本轮未重跑 |
| 16 / 32 | 未重扫（旧口径 OOM） | — | — | — | 旧口径 23.51/23.49 GiB 处失败 |

**三条与旧口径的差异必须记住**：
1. **峰值几乎不变**（batch 8 旧 12,918 vs 新 12,910 MiB；batch 12 旧 19,270 vs 新 19,270）——
   批量读没有增加中间张量驻留，A6 的显存结论**继续成立**：共驻上限仍是 batch 10。
2. **s/update 大幅下降**（batch 8 5.5579 → 1.0195，5.45×；batch 10 6.7763 → 1.1710，5.79×）。
   旧表里 batch 8→10 的「samples/s 饱和」现象在新口径下**不再出现**（新口径 batch 6 最好）。
3. **batch 4/6 是本轮新增的档**，旧表从 8 起测。

**因此 Stage F 菜单里所有 batch 成本必须按本表重算，不能再用 §2.1 的旧数**。

---

## §5 本轮实际执行的命令（可复现）

```
# 建集与逐样本计时（HEAD 与新代码各一次）
.venv/bin/python /tmp/a1probe/baseline.py            # HEAD 口径
.venv/bin/python /tmp/a1probe/a1check.py             # 新代码
# 逐位对比（HEAD worktree /tmp/headwt vs 工作树）
git worktree add --detach /tmp/headwt HEAD
.venv/bin/python /tmp/a1probe/fingerprint.py /tmp/headwt > head.json
.venv/bin/python /tmp/a1probe/fingerprint.py /data/esw/UrbanPiDiT_R2 > new.json
# 损失序列与 batch 逐位对比
.venv/bin/python /tmp/a1probe/a2check.py --workers {0,4} --device {cpu,cuda:0} --batch {2,4}
.venv/bin/python /tmp/a1probe/hashcmp.py --workers {0,4}   # 60 个 batch 的 sha256
# 分段计时与 A6 重扫
.venv/bin/python /tmp/a1probe/segments.py
.venv/bin/python scripts/r7_s3_batch_ceiling_probe.py --batches 4 6 8 10 12 --updates 3 --workers 4 --device 0 --out outputs/r7_s3_batch_ceiling_20261104_attempt02
# 端到端（真实 _run_updates）
.venv/bin/python /tmp/a1probe/e2e.py --workers 4 --batch 4 --updates 300 --out .../e2e_b4.json
# 闸门
.venv/bin/python tools/check_conventions.py          # blocking rules=37 failing=0
.venv/bin/python -m pytest tests/test_r7_stage_a_read_path.py tests/test_r7_long_rollout.py tests/test_r7_long_rollout_runner.py tests/test_r7_long_rollout_evidence.py -q   # 138 passed
```

**临时量测脚本的位置**：`/tmp/a1probe/`（未提交；它们是本次量测的一次性harness，
不是本仓的治理层代码。复现所需的三个已提交工件是：本页、`tests/test_r7_stage_a_read_path.py`、
`scripts/r7_s3_batch_ceiling_{probe,worker}.py` 的 `--workers` 开关）。

---

## §6 GPU 成本

| 项 | GPU-h |
| --- | ---: |
| Stage A 全部量测（含 A6 重扫 5 档、多次复跑、HEAD 基线、端到端） | **见账本行** |

全部为单卡、邻居空闲、共驻不触发；未对任何邻居进程发信号（R-054）。
**无训练、无科学读数、未读 2023 test**（`r = 0`、`test_read: false`）。

---

## §7 结论（区分已确认与推测）

**已确认（实测）**
- A1/A2 的逐位验收通过：`window_sha256`、`usable_windows`、排除集不变；6 个窗口的全部字段零差异；
  60 个 batch 逐字节相同。
- 数据侧 8.9×/2.6×；每 update 从 3.3479 降到 1.0848 s（3.09×）。
- **A 的出口未达到**（1.0848 > 1.0），且**剩余时间 97.8% 是 fwd+bwd**，继续改读法救不回来。
- A6 的显存结论不变（共驻上限 batch 10），但速度结论**已被新口径取代**。
- 「逐位相同」只在固定线程数下成立；GPU 默认多线程本身就不逐位可复现。

**推测（未验证，不得当结论）**
- 若把 K 从 4 降下来或改用 bf16，每 update 可能进入 <1.0；**本轮未测**，且两者都会改
  update 的算力口径，需要各自的 ADR，不属于 A 的授权。
- `activation_checkpointing` 换显存后能否在共驻下把 batch 抬到 12–14：**未测**。

**未做（如实登记）**
- 未重扫 batch 14/16/32（旧口径已测，本轮未重跑）。
- 未在 batch 8/14 上单独测 A3 的相对贡献（只在 batch 4 上测）。
- 未测 `persistent_workers` 启动摊销对短 run（<100 updates）的影响。
