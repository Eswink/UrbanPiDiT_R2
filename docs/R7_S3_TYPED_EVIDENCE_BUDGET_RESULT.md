# S3 typed evidence under an adequate budget —— 证据页

- **日期**：2026-11-16 开轮；**2026-11-17 部分收敛**（typed 臂不可能完成，见 §3）
- **状态**：**部分结果。typed 臂在三个 seed 上均被每 seed 的 35 h 冻结包络截断，
  (a)/(c)/(d) 为 `null`（未测）；(b)/(e) 只在 incumbent+generic 齐全的 seed 上可判。**
- **`scientific_claim: false`**、`test_read: false`、**`r = 0`**（2023 test 从未读取）
- **执行源**：`docs/goals/s3-typed-evidence-budget.md`（Round node S3）
- **决策**：**0115**；计划 `docs/plans/0034-typed-evidence-under-budget.md`
- **产物**：`outputs/r7_s3_typed_evidence_budget_20261116_attempt01/`
- **可复现等级**：`config-reproducible`（同 protocol、同父 checkpoint、同 manifest；
  但不做逐位 GPU 承诺，且 `PER_SEED_SECONDS` 缺陷使本轮**不可**按原协议复现出 typed 臂）

---

## §1 判定点（逐条）

**读数状态**：`readings.json` **不存在**，且**按冻结规则不可能被写出**——`readings()` 对每个
臂做裸 `json.loads`（`scripts/study_r7_s3_typed_budget.py:313`），typed 无 receipt 即
`FileNotFoundError`；`worker(phase="reading")` 另有 `complete` 守卫拒绝落盘
（`scripts/study_r7_s3_typed_budget.py:391-395`）。因此本页**不引用 `readings.json`**，
也不假装从它读出了数。

| 判据 | 结果 | 依据 |
| --- | --- | --- |
| **(a)** `C − B` ≥2 lead 同号为负 + 三 seed 一致 + 区间排除零 | **`null`（未测）** | typed 臂 0/3 seed 有 receipt。**未测 ≢ 假**：包络装不下，不是机制被否定 |
| **(b)** `B − A` ≥2 lead 同号为负 | **待 seed43 定案** | 只有 incumbent 两 seed 有 receipt；见 §4 |
| **(c)** 声明范围 | **`null`** | 依赖 (a)；(c) 是范围规则不是检验 |
| **(d)** 4 / 16 epoch 方向一致 | **`null`** | 本轮无 16-epoch 读数。**不得**记 `false` 也不得记 `true` |
| **(e)** z500@72h 相对各自 seed 的 incumbent 恶化 ≤5% | **待 seed43 定案** | 只依赖 inc+gen；见 §4 |
| **(f)** 预检六条 | **`true`** | `f_preflight_passed: true`；incumbent 3,286,037 / generic ≡ typed 3,434,837 |

**为什么不给 (a) 一个 `false`**：判据 (a) 要的是「三 seed 一致 + 区间排除零」。typed 臂没有
任何 seed 的读数，所以它的**符号、均值、区间全部不存在**。把它写成 `false` 会把「预算装不下」
冒充成「类型结构被测出无贡献」，这是 `da1ed00` 那一类（卡片与读数相反）的镜像错误。

---

## §2 段 B：剂量与深度探针（完成，实测）

来源：`outputs/r7_s3_segB_probe_20261116/segment_b_probe.json` 与 `segment_b_depth.json`，
`json.load` 结构化读取。

| 读出字段 | 值 |
| --- | --- |
| `decision.dose_curve_status` | **`"saturated"`** |
| `decision.sixteen_epoch_is_warranted` | **`false`** |
| `decision.depth_candidates` | **`[]`** |
| `decision.cheaper_depth_candidates` | **`[]`** |
| `dose_curve.tail_delta` | **+0.041507**（正值 = 末点更差） |
| `gpu_hours` | **9.4348**（B-1）+ **3.3461**（B-2）= **12.7809**（实测） |

末 5 点 t2m@72h：`3.3350, 3.3324, 3.3239, 3.3538, 3.3765` —— 先降后升，**不是**严格下降。

**深度扫描（t2m，`delta_vs_k4`，负值=优于 K=4）**：

| K | 6h | 12h | 24h | 48h | 72h | 达 ≥0.005 K 的 lead 数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | **−0.01783** | **−0.02327** | −0.00110 | +0.02726 | +0.05594 | **2**（需 ≥3） |
| 4 | 0 | 0 | 0 | 0 | 0 | 参照 |
| 6 | +0.03760 | +0.05681 | +0.05150 | +0.05431 | +0.04385 | 0 |
| 8 | +0.12063 | +0.19116 | +0.23932 | +0.28838 | +0.30092 | 0 |

**判读**：无深度合格 ⇒ `depth_candidates = []`。K=2 只在 6/12 h 更好、48/72 h 反向变差，
按预声明**不记候选**（「更便宜但更差」不得记成候选）。K=6/8 全面更差且随 lead 放大。

**16-epoch 档位：数据不支持，本轮不启动。** 单轨迹 14 点、同一代码身份的末段饱和度
比跨档位表观单调更可信（后者跨两次 `training_code_sha256`）。

---

## §3 成本：physical-evidence 通路的实测前向开销（本轮最硬的工程读数）

这是本轮**唯一确定性成立**的可发表事实，且它是负向的工程事实。

| 臂 | seed41 实测 s/upd | seed42 实测 s/upd | 相对 incumbent |
| --- | ---: | ---: | ---: |
| incumbent | **1.1162** | **1.1150** | — |
| generic | **1.3886** | **1.4266** | **+24.4% / +27.9%** |

来源：incumbent 取 `seed4{1,2}_incumbent_receipt.json` 的 `elapsed_seconds / 32132`；
generic 取 `seed4{1,2}/training/generic/progress.jsonl` 的后半段增量口径。

**参数只涨 4.5%（+148,800），时间涨 24–28%。** 两者的差额说明开销不在参数量的线性项上，
而在四场球面算子与逐 patch 编码的前向计算里。这一条直接推翻了冻结协议的前提：

```
per_seed = 3 × (32132 × 1.1536 + 2682.3) = 119,249 s = 33.12 h   ← 假设三臂同速
PER_SEED_SECONDS = 126,000 s = 35.00 h                            ← 余量 5.6%
```

按实测 treated 速率重算：**37.287 h（seed41）/ 37.964 h（seed42）**，**超出 35 h 包络
+2.287 / +2.964 h**。即使 typed 按 incumbent 速率跑（不可能），缺口仍为 **+0.072 / +0.336 h**
⇒ **typed 臂在三个 seed 上必然被 `_check_deadline` 截断**。

### §3.1 这条缺陷为什么没被现有测试拦住

`tests/test_r7_s3_typed_budget.py:380` 的 `test_the_per_seed_envelope_exceeds_what_one_seed_actually_does`
用 `proto.MEASURED_SECONDS_PER_UPDATE`（=1.1536，incumbent 速率）推导期望值，与冻结同源：

| 口径 | per_arm | ×3 | vs 126,000 s |
| --- | ---: | ---: | ---: |
| 测试/冻结同源（1.1536） | 11.04 h | 33.13 h | **+1.875 h 通过** |
| 实测 generic/seed41（1.3886） | 13.30 h | 39.90 h | **−17,629 s 不足** |
| 实测 generic/seed42（1.4266） | 13.65 h | 40.95 h | **−21,400 s 不足** |

**教训**：断言若从被测对象自己的常数推导期望值，只能防「常数被误改」，不能防「常数与现实不符」。

---

## §4 部分读数工具：为什么需要它，以及它给出什么

`readings.json` 写不出时，(b) 并没有消失——`B − A` **从不读 typed 臂**。新增
`tools/r7_s3_typed_budget_partial_reading.py`（190 行）从**逐臂** receipt 装配 (b)/(e)：

- 复用冻结 `r7_typed_budget_verdict` 的 `_clause_pair` / `clause_e`，**不重述阈值**；
- `(a)/(c)/(d)` 恒 `None` 并附理由 ⇒ 缺失不会被读成负结果；
- 守卫空 seed 集（冻结 `_clause_pair` 在此处 `ZeroDivisionError`，见 §7）；
- 落在 `tools/`，**不进** `execution_files_sha256`（已断言：84 项冻结清单中 0 项在 `tools/`）。

**当前实测输出**（2026-11-17T19:0xZ，仅 incumbent 两 seed）：`seeds_with_both_arms: []` ⇒
(b) 为 `null` 并说明「尚无 seed 同时具备 inc+gen」。这是**正确的未测**，不是失败。

---

## §5 已申报的差异与限制（`limitations`）

1. **B 与 C 容量相等但计算图不同**：generic 在每个 patch 位置融合后池化，typed 按类型池化后
   投影。因此对照量是「融合发生在哪里」，不是「物理是否在场」（`protocol.json` 原文）。
2. **A → B 增加通路固有 +148,800 参数（+4.5%）**。该差异**不可移除**，任何 (b) 的胜出必须
   连同它一起申报，**不得**声称已排除容量解释。
3. **`epochs` 字段少了 batch 因子**（errata，不改动）：`readings()` 与 preflight 都算成
   `updates / 32,131 = 1.000031`，正确是 `updates × 4 / 32,131 = 4.000124`。该字段
   **不被任何判据读取**，但会永久留在协议里，故在此点名——引用「1.000031 epoch」会把本轮
   说成 1/4 剂量。
4. **typed 臂未测**：三个 seed 均因包络不足被截断。**这不是机制无效的证据。**
5. **与 `record:s3-typed-spatial-result` 的关系**：该记录已在 2,400 updates 下把同一对臂测为
   `|Δrmse| ≤ 3.2e-6 K`。本轮只改**剂量**（400/2,400 → 32,132 updates），未改机制。
   即使将来 (a) 为正，**也不得**回溯成「之前那轮错了」；若为负，也只能说
   「在 0.075–4 epoch 全域无效」。
6. **`PER_SEED_SECONDS` 缺陷使本轮不可按原协议复现 typed 臂**：任何忠实重跑都会再次截断。
7. **截断臂不写 receipt**：`segment_a_hours()` 只 glob receipt，故其 GPU-h 会漏记；
   必须从 `progress.jsonl` 末条补算（见 §6）。

---

## §6 成本

| 项 | GPU-h | 性质 |
| --- | ---: | --- |
| 段 B（9.4348 + 3.3461） | **12.7809** | **实测** |
| 段 A incumbent ×2 | **21.358** | **实测**（receipt：训练 + 五 lead 评分） |
| 段 A generic ×2 | 进行中 | 完成后按 receipt 实测登记 |
| 段 A typed ×2 | — | **必被截断**，无 receipt；须从 `progress.jsonl` 补算 |
| 本轮合计 | 待定 | **不得**用估算值写进账本 |

**软预算 96 GPU-h / 硬上限 144 GPU-h。** 按实测速率外推本轮约 **118.9 GPU-h**
（超软约 22.9，在硬上限内）——**这是外推，不是实测，不得写进账本**。其中 typed 臂空转
约 **33.4 GPU-h** 换不到任何子句，**不可回收**（R-054 禁止对进程发信号）。

**墙钟**：`planned 172,800 s / hard 259,200 s`；「seed 串行、两卡两波」口径需 75.25 h > 72 h
⇒ **即使不改代码，总墙钟也超硬上限**。

---

## §7 本轮发现的缺陷（全部 0 GPU-h 留证，未改冻结源）

| # | 缺陷 | 影响 |
| --- | --- | --- |
| 17 | `clause_c` 在 (a)F+(b)T 时输出「通路无增量」，与 (b) 真矛盾 | 文案字段；`verdict` 字符串本身正确。**本轮不改**：文件在冻结清单（`b4bc4f7c…`），改则 reading 阶段必失败 |
| 18 | 每 seed 包络按 incumbent 速率定，装不下三臂 | **typed 臂三个 seed 全部必被截断**（§3） |
| 19 | 装配器把 `source_sha256`/`data_identity` 读成 `null`，校验器静默放行 | 未来接登记会产出空身份记录（实测 `exit 0`） |
| 20 | 截断臂只写 `failed_attempt.json`，无 receipt | `segment_a_hours()` 漏记其 GPU-h |
| 21 | `_clause_pair` 在空 seed 列表上 `ZeroDivisionError` | 部分读数调用方会崩；新工具已加守卫 |

**两条交接指令被拒绝执行，理由是它们会破坏本轮证据**：
① 修缺陷 17 = 改冻结源 ⇒ reading 阶段 `execution source changed`（§5.2d/§5.2x 已各发生一次）；
② 重建 `code.zip` 到 HEAD ⇒ `code_zip_sha256` 与 `protocol.json` 脱钩，`_archive` 会拒收。
该 zip 的 comment 是 `f0da923`、sha256 是 `cc58c88c…`，与冻结值**逐位一致**（已实测），
测试里那条 skip 是**预期行为**。

---

## §8 本轮未做与下一项任务

**未做**：不启动 seed43（其完整包络窗口是 +12.28…+14.20 h，**仅 1.92 h 宽**，会话内无法跨越，
且 `cron`/守护/会话外续跑被明令禁止）；不延长 deadline；不重冻结；不向任何进程发信号；
不产出 `readings.json`；不宣布任何科学结论。

**下一项任务**：在窗口内 `tools/r7_round.py stage seed 43`（**绝不** `stage-seeds`，那会把
41/42 当作未跑过而重跑 21.4 GPU-h），约 24.4 h 后运行
`tools/r7_s3_typed_budget_partial_reading.py --attempt <attempt>` 取 (b)/(e)，再按
brief §5.6 登记（`outcome_class: unresolved`、`candidate_state: not-candidate`）。

**本轮不得**：把 (b) 为真写成「物理类型结构有贡献」（那需要 (a)）；把 (a) 的 `null` 写成负结果。
