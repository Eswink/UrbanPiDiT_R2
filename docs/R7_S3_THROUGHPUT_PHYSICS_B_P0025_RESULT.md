# S3 Stage B 第一点（0.25 epoch / 2,008 updates）读数

- **日期**：2026-11-04
- **状态**：**工程出口 (a)–(d) 全真**；该点作为学习曲线的一个**纵坐标**可用
- **`scientific_claim: false`**；`test_read: false`；`r = 0`
- **执行源**：`docs/goals/s3-throughput-physics-b-p0025.md`（`attempt02`）
- **决策**：0084（判据拆分）、0085/0086/0087（Stage A 三项工程改动）、**0088**（本点执行设计）
- **协议**：`a3e06e16c8c03b0132af3066cd7989b7de978c480c4f59605a87f7f4bd94ec15`
- **代码**：`36d4d27`（`code.zip` 注释与 HEAD 一致）；`model_code_sha256 59340710…`

> **本轮不判定饱和。** 饱和规则是「16 epoch 相对 4 epoch 的 72 h skill 一阶差分」，
> 一个点不可计算。`readings.json` 的 `saturation_verdict` 为 `null`，
> 规则逐字记在 `protocol.json` 的 `saturation_rule`。**本页的任何数字都不得引作饱和与否的证据。**

---

## §1 预声明工程出口 (a)–(d)：全真

| 判据 | 要求 | 实测 | 结果 |
| --- | --- | --- | --- |
| (a) | 精确 `2,008` updates、batch `4` | `updates 2008`、`batch_size 4`、`epochs 0.249977` | **真** |
| (b) | 5 lead × 17 变量全有限，cohort 与冻结值相符 | 85 个 RMSE 全部有限；cohort 1,458/1,457/1,455/1,451/1,447 逐字相符 | **真** |
| (c) | 72 h t2m skill 有限，分母为注册 D2 值 | `0.098082`，分母 `3.926891361747067` | **真** |
| (d) | 无判据被改动 | `protocol_sha256` 自校验、`test_read is False`、17 变量清单未变 | **真** |

`readings.json` 的 `exit_conditions`（驱动自算、未手改）：
`{"a": true, "b": true, "c": true, "d": true, "all": true, "verdict": "advance-to-next-curve-point"}`

**桥 (e)，非科学判定**：父身份逐 seed 断言通过——
`expansion.parent_state_sha256 = 7c416ccb4054cf7e…`，等于 `MIGRATED_PARENT_STATE_DIGEST[41]`
（155 张量），`load_checkpoint` 的 `model_code_sha256 59340710…` 检查也通过。
失败会 `raise` 且不进入 (a)–(d)；本点没有触发。

---

## §2 t2m 读数（本点唯一的注册 skill 分母）

| lead | t2m RMSE (K) | 气候态分母 (K) | skill |
| ---: | ---: | ---: | ---: |
| 6 h | 2.286245 | 3.918840657 | **+0.659646** |
| 12 h | 2.858136 | 3.919603711 | **+0.468282** |
| 24 h | 2.610901 | 3.921682295 | **+0.556763** |
| 48 h | 3.332818 | 3.925382069 | **+0.279126** |
| 72 h | 3.729343 | 3.926891362 | **+0.098082** |

`full` 区域、4,225 点、逐 lead cohort 见 §1(b)。

**这一行数字只说明**：在这条配方、这个父、这个数据清单下，**0.25 epoch** 的模型在 72 h 上
比 48 桶 train-only 气候态好 **9.81%**（skill 是 `1 − (RMSE/分母)²` 的口径）。
它**不说明**：训练量再多是否继续改善（那是饱和问题）、是否泛化、是否优于任何已登记臂。

**与已登记臂不可直接对照**：本点的 `training_code_sha256` 因 Stage A 的读法与有限性检查改动
而与所有历史轮不同；父端点是 **1,600** updates 而非 R1/R2 的起点。引用时必须同时读这条。

---

## §3 十七个变量 × 五个 lead 的完整 RMSE

单位按 store 的 `channels` 与 `boundary_rmse.csv` 的 `unit` 列；`q*` 为 kg/kg，`z*` 为 m²/s²，
`*10/*50/*85/*250` 为 m/s。

| 变量 | 6 h | 12 h | 24 h | 48 h | 72 h |
| --- | ---: | ---: | ---: | ---: | ---: |
| t2m | 2.2862 | 2.8581 | 2.6109 | 3.3328 | 3.7293 |
| u10 | 1.3296 | 1.6347 | 1.8724 | 2.0551 | 2.0896 |
| v10 | 1.5208 | 1.9444 | 2.3142 | 2.6759 | 2.7691 |
| mslp | 141.6202 | 225.9942 | 346.9120 | 469.9724 | 507.1666 |
| z850 | 87.5482 | 140.8405 | 222.5163 | 298.3481 | 315.8428 |
| t850 | 1.6503 | 2.2457 | 2.6669 | 3.5581 | 3.9708 |
| q850 | 0.0012 | 0.0015 | 0.0018 | 0.0021 | 0.0022 |
| u850 | 2.4400 | 3.1987 | 3.7929 | 4.2517 | 4.3569 |
| v850 | 2.5368 | 3.3532 | 4.1431 | 4.7770 | 4.9084 |
| z500 | 110.1262 | 186.7643 | 298.6928 | 412.3811 | 457.5393 |
| t500 | 1.1273 | 1.6130 | 2.1835 | 2.7635 | 3.0116 |
| q500 | 0.0007 | 0.0008 | 0.0010 | 0.0011 | 0.0011 |
| u500 | 3.8235 | 5.1712 | 6.2318 | 6.9882 | 7.3465 |
| v500 | 3.7467 | 4.6960 | 5.5534 | 6.3411 | 6.5143 |
| z250 | 175.8900 | 306.0942 | 504.2575 | 737.6227 | 848.7269 |
| u250 | 5.2164 | 7.4768 | 9.8019 | 11.8928 | 12.8524 |
| v250 | 5.4416 | 7.3247 | 9.0614 | 10.6998 | 11.2150 |

**只有 t2m 有 skill 数**，其余 16 个变量报原始 RMSE：D2 reference 只拟合了 t2m 的
48 桶气候态。**不发明分母**；若要 17 变量的 skill，必须先注册它们的气候态，那是另一份工作。

---

## §4 成本（实测，非估计）

| 项 | 实测 |
| --- | ---: |
| 训练（2,008 updates、batch 4） | **2,358.3 s**（≈ **1.174 s/update**） |
| 五个 lead 评分 | **2,633.8 s** |
| **合计** | **1.3867 GPU-h** |
| `planned_seconds` | 43,200 s ⇒ **soft overrun 0** |
| 硬上限 | 86,400 s（未触发） |
| checkpoint | **1 个**（`update_0002008.pt`），决策 0087 生效 |
| 峰值 reserved | **9,372,172,288 B = 8.729 GiB** |
| 线程数 | 28（记入 receipt；逐位复现只在固定线程数下成立，见决策 0084 负面后果 3） |

**训练 s/update 实测 1.174**，与 Stage A 的四次均值 **1.1279** 相差 4.1%，
落在那次测得的运行间方差（约 4.4%）内 ⇒ **Stage A 的吞吐结论被这次真实长跑独立复现**。

**预算估计被实测推翻**：本 ADR 层估计约 12 GPU-h，实测 **1.3867 GPU-h**。
原因是 `planned_seconds` 里的评分余量按最坏情况给，而 5,000 步 checkpoint 策略把这部分省掉了。
**估计值不得写成实测值**——这里如实记下这个 8.7 倍的差。

---

## §5 本轮的三次失败尝试与它们的修复（如实登记）

`attempt01` **作废、不参与任何读数**，产物保留不删：

1. **`deny_network()` 误杀 worker IPC**（`attempt00`）：它替换**所有** `socket.connect`，
   包括 `multiprocessing` 资源分享器用的 `AF_UNIX` 本地套接字 ⇒ 第一个 seed 死于
   `Pin memory thread exited unexpectedly`。修法：只拒绝**非 `AF_UNIX`**，DNS 与
   `create_connection` 仍全拒（R-028 管的是**出网**，不是禁本地 IPC）。
2. **驱动缺 `if __name__ == "__main__"` 守卫**：`stage prepare` 曾 0.06 s「成功」却没有任何产物，
   因为导入模块只定义了 `main` 而没调用。修法：补守卫 + 一条 `ast` 反证测试。
3. **`main()` 返回 worker 结果**：`sys.exit(value)` 把 38 MB 协议体打印进 stage log。
   修法：丢弃返回值 + 一条「`main` 不得有 `return` 值」的反证测试。
4. **checkpoint 策略未生效**（`attempt01`）：协议声明 `checkpoint_every 5000`，
   但驱动没把参数传给 `fine_tune_long_rollout`（默认 20）⇒ 实写 **101 个 / 5.6 GiB**，
   而 0087 规定本点只保留 **1 个**。修法：显式传参 + 源码级反证测试。
5. **读数阶段 `KeyError: 'leads'`**（`attempt01`）：`_climatology_denominator` 读冻结摘要块
   `climatology`（只有桶数与年份），而 per-lead 的 t2m 分母在 prepare 解析出的
   `climatology_reference` 里。修法：改读后者，缺 key 即 `raise`（不发明分母）。

前三项在 `attempt01` 的 seed 之前/之中修掉；后两项在 `attempt01` 之后修掉并要求重跑，
因此 `attempt01` 的读数**不登记**。每一条都配了一条会失败的反证测试（`tests/test_r7_s3_throughput_physics_b.py`，22 项）。

---

## §6 可复现性与未做

**可复现**：协议 `a3e06e16…` 自校验、`code.zip` 注释 == `code_commit`、90 项具名执行闭包逐文件哈希、
父 155 张量 digest、checkpoint 单文件 sha256、五 lead 的 `boundary_rmse.csv`/`skill.csv`/`provenance.json`
各有 sha256。**等级：config-reproducible**（GPU 训练非逐位可复现，见决策 0084 负面后果 3）。

**未做**：
- **未判定饱和**（一个点不可判，规则只在阶梯完整时求值）。
- **未跑另外三个点**（8,033 / 32,132 / 128,528 updates）。
- **未给 16 个非 t2m 变量编 skill 分母**。
- **未做多种子**（n=1 是决策 0088 的明确选择，不是遗漏）。
- 本页不含任何科学结论；`scientific_claim: false`。

---

## §7 CI 状态（一次红、已定位并修好，绿已确认）

**这一次不是「未能确认」，而是实测红过一次。** 如实登记：

| 提交 | `R7 CPU CI` | 结论 |
| --- | --- | --- |
| `6254c94`（登记本点读数） | run **`41824751122`** | **failure**，job `pytest` 第 9 步「Run unit, integration and installed-wheel tests」 |
| `157f531`（修该红） | run **`41837758372`** | **completed / success**，job `pytest` 第 5/6/7/8/9 步全 success |
| `99d0045`（登记本页 §7） | run **`41842156903`** | **completed / success**，`pytest` job 全部步骤 success |

**红的真因**（在干净 worktree 上复现后定位，非猜测）：`tests/test_r7_gap_diagnostic.py` 的
`test_full_tiny_process_same24_cases_once_train_only_climate_no_optimizer_or_test` 断言
`len(reads) == 340 + 20*14 + 4*7`（648）。决策 0085 把每个 train 窗从 14 次标量读改成
**一次 14 帧 span 读**，于是同一个 fixture 的往返次数变成 **388**，而**覆盖的索引完全没变**。
这条断言把**读法**写成了数字，而不是它所代表的不变式——与 Stage A 曾修的
`test_preflight_and_construction_never_read_state_fields` 是同一类缺陷。

**修法**（只改测试，未改任何被测代码）：改断不变式——
`340 + 4*7` 次标量读（气候态步 + 4 个验证窗各 7 帧）、**20** 次连续 14 帧 span、
`len(scalars) + sum(len(windows)) == 648`（索引覆盖与旧断言逐字相同）。
既没有删断言、也没有放宽上限：`R-051` 恰好落在 600 行硬上限（本文件改前 593 行）。

**降级证据（与 CI 同形，非等价）**：干净 worktree `/tmp/cleanwt4`（**无 `outputs/`**、
无本机的 `.venv` 符号链接）上跑**整包** `pytest -q` ⇒ **5,438 passed, 27 skipped, 0 failed**
（与 CI 同一命令；红的那版是 5,434 passed + 4 failed，总数同为 5,438）。

**另有一类只在本地出现的假红**：`tests/test_build_round_code_archive.py` 的 3 项在
`/tmp/cleanwt3` 上失败，原因是该 worktree 里我建的 `.venv` 符号链接让 `git status` 变脏，
而归档器拒绝归档脏树。CI 的裸检出没有这个文件，故与 CI 无关；移走符号链接后 67 项全过。
**这一条不得读成「CI 也红」**，也不得读成代码缺陷。
