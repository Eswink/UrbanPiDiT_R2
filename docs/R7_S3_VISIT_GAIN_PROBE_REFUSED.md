# G1-B visit-gain：探针拒绝开轮，四个出口全部 `null`（不是负结果）

**状态：`probe-refused`（2026-11-26）；`scientific_claim: false`；`test_read: false`；`r = 0`；1.1330 GPU-h。**
本页是 G1-B 的**执行记录**，不是技能声明，**不是**对 visit-gain 假设的否定，
**不**读 2023 test，**不**消耗确认序号。

## 1. 一句话结论

**没有测到 A−B，因此没有任何科学结论。** 两臂的探针判 `refused-insufficient-envelope` 并
**按冻结规则拒绝开训**：实测 2.61–2.67 s/update，外推需要剩余包络的 **0.728 / 0.743**，
超过冻结的 **0.60** 拒绝阈值。**没有放宽包络重试，没有改判据，没有训练任何臂。**
四个预声明出口**全部保持 `null`**，分支**既未关闭也未确认**。

但探针**量错了**：在同一台机器、同一份权重、同一条训练路径上，后 30 步的边际速率是
**1.583 s/upd（off）/ 1.733 s/upd（on）**，而 400 步窗口的总速率是 2.5。
差额是**启动摊销**，不是实现或算力的代价。

## 2. 冻结了什么（读数之前）

| 项 | 值 |
| --- | --- |
| 协议 | `protocol_sha256 1ec96777607c4905cccc5e6bbf96642fa122da40654f376eca336aefa818ad50` |
| 代码 | `code_commit 389bfa02…`、`code_zip_sha256 23ee65cc…` |
| 模型 | `model_code_sha256 a6cf6bbbdb2263e7da4e50e3d15562652d24101c02a1f2c8d55ee14d5036fda0` |
| 训练代码 | `training_code_sha256 e7674118…`（与 in-protocol 轮**逐位相同**，未改 `training/`） |
| 数据 | `train_data_identity 81307008…`、`val_data_identity e5fb694f…` |
| 父 | 三 seed 的 p32,132 incumbent，迁移后 `state_digest` 与归档**逐位相同** |
| 两臂 | A `visit_damped` `g_k=1/sqrt(1+k)`；B `uniform_damped` `g=1/sqrt(1+(K−1)/2)`，K=4 |
| 区域 | `full(65×65)`，`boundary_margins=(1,)`，4,225 点 |
| 判据 | 决策 0119；出口 ①–④ 见 §5 |

## 3. 探针读数（实测，唯一有数字的一节）

| seed | A 臂 s/upd | B 臂 s/upd | treated（取大） | 需要 | 剩余包络 | 占比 | 判定 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 41 | 2.5016 | 2.6077 | **2.6077** | 18,917 s | 25,995.6 s | **0.7277** | 拒绝 |
| 42 | 2.6731 | 2.5103 | **2.6731** | 19,230.9 s | 25,899.3 s | **0.7425** | 拒绝 |

两个 seed 各跑完两臂 × 400 updates 的探针后拒绝；seed 43 **未启动**（`stage-seeds` 在
第一个非 `headroom-blocked` 失败处停止后续 seed）。实耗 **1.1330 GPU-h**
（seed41 0.5628 + seed42 0.5702），**未训练任何臂、未评分任何 lead**。

## 4. 为什么 2.6 s/upd 是探针口径的伪影（实测诊断）

同一台机器、同一份归档权重、同一条 `fine_tune_long_rollout`、同 `batch_size=2`，
60 updates 取后 30 步的**边际**速率（剔除启动摊销）：

| 臂 | 60-update 总速率 | 后 30 步边际速率 |
| --- | ---: | ---: |
| **off（`visit_gain=None`，即归档父）** | 4.804 s/upd | **1.583 s/upd** |
| **on（`visit_damped`）** | 4.942 s/upd | **1.733 s/upd** |

- **实现不是瓶颈**：on 只比 off 慢 **~9%**（两次额外乘法），且与已登记的 in-protocol 轮
  （1.432 / 1.447 / 1.393 s/upd，batch 4）同量级。
- **前向计时独立佐证**：真实 batch 上 batch2 off 60.46 ms / on 69.83 ms、
  batch4 off 63.29 ms / on 62.39 ms —— gain 开关不是 72% 的代价。
- **400 步太短**：单次 60-update 运行的总速率是 4.8 s/upd，边际速率 1.58；
  同一副启动成本摊到 400 步仍是 ~0.9 s/upd 的加项，正好把 1.58 抬到 2.5。

⇒ **探针用了总速率而不是边际速率**，这是估计量的缺陷，不是算力或代码的缺陷。

## 5. 四个预声明出口的状态

| # | 判据 | 值 | 为什么是这个值 |
| --- | --- | --- | --- |
| **①** | 72 h `full(65×65)` t2m 的 A−B 三 seed 同号为负 | **`null`** | 没有任何臂训练、没有评分 ⇒ 未测 |
| **②** | 6 h / 12 h 的 A−B seed 均值 ≤0 | **`null`** | 同上 |
| **③** | 身份见证（off 逐位惰性 + A 在 k=0 逐位相等） | **`true`**（已实测，见 §6） | 在 prepare 阶段**先于训练**测量并落盘 |
| **④** | 每 seed 键集与父 `state_digest` 逐位相同 | **`true`**（已实测，见 §6） | 迁移收据三 seed 全部 `bitwise_identical` |

出口 ③④ 是**工程身份**，它们与训练无关，所以能在拒绝前就成立并被验证。
出口 ①② 是**唯一的科学读数**，**一个数字都没有**。

## 6. 已实测成立的两条身份证据（D1/D2）

**③ 身份见证**（`identity_witness.json`，`torch.set_num_threads(1)` 固定线程）：

| 子句 | 值 |
| --- | --- |
| `bitwise_equal_to_pre_switch_expression` | **true** |
| `bitwise_equal_across_visit_indices` | **true** |
| `key_set_equal` / `tensors_equal` / `parameter_count_equal` | **true / true / true** |
| `no_new_buffers` | **true**（`missing_from_parent=[]`、`extra_in_parent=[]`） |
| `visit_damped_k0.bitwise_equal` | **true**，`gain_at_zero = 1.0` |
| `visit_damped_k0.diverges_at_visit_1` | **true**（证明 A 的改写从第二步开始） |
| `arms_share_post_parent_state` | **true** |

**④ 父迁移**（`outputs/r7_s3_visit_gain_parent_migrated_20261126_attempt01`）：

| seed | 归档 `state_digest` | 装入后 | 相等 | 键集 | 张量数 |
| ---: | --- | --- | --- | --- | ---: |
| 41 | `4d0a3fd2…` | `4d0a3fd2…` | **是** | 同 | 155 |
| 42 | `8c34a6d1…` | `8c34a6d1…` | **是** | 同 | 155 |
| 43 | `d6d55fda…` | `d6d55fda…` | **是** | 同 | 155 |

`all_seeds_bitwise_identical: true`。迁移只是**重签代码身份**（`59340710…` → `a6cf6bbb…`），
权重是同一批字节；不带优化器、cursor 或 RNG。

## 7. 两个缺陷（都只记录，不在本轮修）

1. **探针拒绝不写 partial receipt。** brief 要求任何非正常结束都写，但
   `probe()` 在臂循环**之前**抛异常，而 `_seed()` 只把 `_train_arm` 包在 partial 处理器里
   ⇒ 拒绝路径**静默漏收据**。两份 `seed{N}_probe_partial_receipt.json` 已按各臂自己的
   `progress.jsonl` **事后重建**，并标注 `written_retroactively: true`。
   修它要改 `scripts/` 下已进 `execution_files_sha256` 的文件 ⇒ 须新 attempt + 重冻。
2. **探针估计量选错。** 400 步的总速率把 ~300 s 启动成本摊成 ~0.9 s/upd。
   应取后段**边际**速率，或把探针拉长到启动可忽略。这是 0116「探针早停」形态的
   一个估计量缺陷，不是那个形态本身的错。

## 8. 与既有各轴的关系（本轮未触碰）

未改 `MIXING_MODES`、感受野、网格（**0077** 保持关闭）；`dim`/`depth` **逐字不变**、
参数计数**逐位相同**（两臂与 C 都是 3,286,037）、父张量**逐位可装**（**0048** 保持关闭）；
未重开 K（`record:s3-reasoning-depth` 仍 `unresolved`，K 仍 4）；未做 loop-index embedding、
per-loop LoRA；未改深监督组织；未加物理 loss。**未读 2023 test**。

## 9. 未做 / 下一动作

- **未做**：两臂训练、五 lead 评分、出口 ①② 的判定、72 h 轨迹、四季均衡。
- **分支状态**：**既未关闭也未确认**。拒绝是工程事件，把它读成「阻尼无效」会把一次
  未执行的测量写成负结果 —— brief 明禁。
- **下一动作（若继续这一支）**：先写一个**取代 0119 探针口径**的 ADR（边际速率或更长探针），
  新 attempt + 重冻，再训练。**不得**在已冻结的 attempt01 上放宽包络重试。
- **恢复第一条命令**：`.venv/bin/python tools/r7_round.py state`。

## 10. 成本的如实记账

| 项 | GPU-h |
| --- | ---: |
| seed41 探针（2 臂 × 400 updates） | 0.5628 |
| seed42 探针（2 臂 × 400 updates） | 0.5702 |
| **合计** | **1.1330** |

未训练任何臂、未评分。账本新增一行 1.1330 GPU-h；**不登记科学 index record**
（没有可登记的读数，且 `outcome_class` 若记 `negative`/`unresolved` 都会把
未测说成测过）。本页本身登记为 `outcome_class: audit` 的工程记录。
