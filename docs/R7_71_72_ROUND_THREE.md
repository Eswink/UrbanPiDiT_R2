# #71 第三轮：给时空输入（M1）补容量控制臂——把 B−A 拆成「信息」与「容量/偏置」

**状态：本轮的代码、测试与一轮有界实验均已完成；科学结论按下面的局限读，不得外推。**
本文件**不覆盖**前两轮：`docs/R7_71_72_M1_AND_RWA.md`（第一轮）与
`docs/R7_71_72_ROUND_TWO_ATTRIBUTION.md`（第二轮）仍是各自唯一的证据；三份文档的
`protocol_sha256` 与 `model_code_sha256` 都不同，**数字不可相加、不可并列**。

| 项 | 值 |
| --- | --- |
| 起点 SHA | `ac6a3ef486d56d43e11261b62e7294872d1e5eae`（开工时 `git rev-parse HEAD`） |
| 数据 | M2 双月段 `outputs/r7_m2_segment/store/manifests`：train 186 / val 22 / test 26（**本轮 test 未读**） |
| 协议 digest | `d62db6dbd8cf9e894beee4c2cab72488b81bf83a09f04e65868eba6ed0d667c2`（**12 个 run 全部相同**，逐 run 核对） |
| `model_code_sha256` | `f349adceb5ba03e10fbb45b514bfbc25352c6b81848b255d615127fb36b047e1` |
| 实测 GPU 训练 | **0.5489 GPU-h**（12 份训练记录 `elapsed_seconds` 求和 = 1976.1 s；训练循环墙钟 1979.6 s = 0.5499 GPU-h） |
| 本轮自设上限 | ≤0.9 GPU-h（实测 0.5489，**未超**）；第二批授权 ≤24 GPU-h，已用 1.914 → **2.463**，余 **21.537** |
| 墙钟 | 一个 seed 一进程、同卡顺序：seed41 14.0 min / seed42 14.0 min / seed43 13.8 min（**均 < 30 min**）；整轮 18:18:38→19:01:23 |
| 实验产物 | `outputs/r7_71_72_round_three/`（protocol、逐 seed 结果、合并结果、配对比较、四张表、模式探针） |
| 回归钉住证据 | `tests/test_r7_spacetime_input_modes.py`（11 个冻结 digest，取自 `ac6a3ef`，改动后逐位相同；含敏感度反证） |
| 默认路径等价 | `tests/test_r7_switched_path_equivalence.py`（23 digest + 反证，**实跑**）、`tests/test_r7_input_path_consistency.py`（**实跑**） |
| 决策 | `docs/decisions/0016-round-three-field-modes-and-primary-reader.md` |

---

## 1. 交付物逐条核对

| # | 要求 | 状态 | 证据 |
| --- | --- | --- | --- |
| D1 | 新增 `constant` 与 `shuffled` 两个模式，默认 `fields` | **完成** | `model/spacetime_conditioning_r7.py`（`FIELD_MODES`、`require_field_mode`、`apply_field_mode`）；`SpacetimeConditioning(field_mode=…)` 默认 `fields`，三个模型类默认 `fields`、非法模式报错、`spacetime_inputs=False` 时给非 `fields` 模式报错（不静默忽略）；测试 8 项 |
| D2 | 替换发生在模型内部：dataset/rollout 仍带真实字段，`DECLARED_MODEL_INPUTS` 与输入路径一致性测试**不得改变** | **完成** | 替换点在 `require_spacetime_fields` 之后；`tests/test_r7_input_path_consistency.py` 6 项**原样实跑通过**（未改一字）；新模式测试断言三种模式下缺字段仍 `KeyError`、越界仍 `ValueError` |
| D3 | 默认路径逐位不变：23 digest 等价测试实跑通过（含反证） | **完成** | `pytest tests/test_r7_switched_path_equivalence.py tests/test_r7_input_path_consistency.py -q` → **10 passed**；脚本模式 `bitwise_identical: true`、`digest_count: 23`（比对冻结修订 `93d89aa`）；反证测试 `test_the_comparison_detects_a_changed_implementation` 通过 |
| D4 | 改动**前**冻结 `spacetime_inputs=True` 路径的 forward/反向 digest，改动后断言相同 | **完成（钉住方式已修正一次，见下）** | 冻结值取自 `ac6a3ef`（改动前的工作树；当时的 `git status` 只有本测试文件，以及两处既有未跟踪项 `.zcodeignore` 与 `tests/fixtures/r7_equivalence_recipe.py`，二者都不在配方读取范围内），11 个 digest 见 §11；**CI 里断言的是同进程双实现比对**（`git archive ac6a3ef` 的冻结修订 vs 工作树），因为机器本地的 digest 常量不可跨机复用（详见 §11.1） |
| D5 | 预登记 primary：第一次 `optimizer.step()` 之前写进 `protocol.json` 并冻结 digest | **完成** | `protocol.json` 的 `primary_registration`（变量 t2m、5 个时效、四对、判定文字），在每 seed 第一次 step 前落盘并**回读校验**（变量/时效/四对/非空文字逐项断言）；跑后未修改（digest 12/12 相同） |
| D6 | 12 个 run：`protocol_sha256` 全同、只读 val、test 封存、实测 GPU-h、四臂参数/FLOPs（E 与 B 逐张量相同、算力相同） | **完成** | 12/12 同 digest；`split` 全 `val`、`test_read: false`；0.5489 GPU-h；参数 A 2,799,779 / B=E=P 2,819,267，前向 FLOPs A 12,843,777,408 / B=E=P 12,927,412,608（E 与 B 逐张量相同，见 §5） |
| D7 | 比较器出各对计数，逐 seed 同号才算 improved/worsened，全 17 变量 × 5 时效照报 | **完成** | `paired_comparison.json` 六对（B−A、B−E、E−A、P−A、E−P、P−B），85 格全报；逐变量计数见 §7；`A−A` 自检**未做**（目标里标为可选，比较器对同臂自比不保证接受） |
| D8 | 本文件：primary 判定、全变量汇总、重跑稳定性（A/B 与 round two 对照）、`scientific_claim: false`、limitations、未做的事、下一项 | **完成** | 本文件 |
| D9 | 提交（治理层 `git add`、`[model-digest-change]`）并核对 CI；skipped 的实验 workflow 不算失败 | **完成** | 见 §14 |

---

## 2. 本轮的问题与四臂设计

第二轮量到 `B − A` 在 t2m 上 −1.0…−2.0 K，同时记下一条硬局限：**B 只比 A 多 19,488 个
参数，没有容量控制**，所以「读到初始化的时空信息」与「模块本身能表示的偏置」分不开。
本轮只补这一条，不引入新机制：

| 臂 | 配置（`spacetime_inputs` / `spacetime_field_mode`） | 角色 |
| --- | --- | --- |
| A `process_pooled` | 关 / `fields`（默认） | 参照臂（= 第二轮的 A） |
| B `process_spacetime_only` | 开 / **`fields`** | 真实字段（= 第二轮的 B） |
| E `process_spacetime_constant` | 开 / **`constant`** | 模块在场、参数与算力相同、**输入无信息** |
| P `process_spacetime_shuffled` | 开 / **`shuffled`** | 真值但**错配对**（确定性批内错配） |

于是 `B − E` 是**信息**、`E − A` 是**模块容量/偏置**、`B − A` 是两者之和、`P − A` 是
「真值错配对」。E 与 B 的参数张量逐个相同、实测 FLOPs 相同（§5），所以 `B − E` 是
**输入对比**；A 缺整个模块，凡涉及 A 的对仍按容量混杂读。

**模式语义（模型内部，字段校验之后）**：`constant` 把四个字段换成同形状零张量；
`shuffled` 把逐样本字段（`init_utc_hour`、`init_day_of_year`）沿样本轴 `roll(1, dims=0)`。
纬度/经度按构造是批不变的（一批里网格不一致即报错），批内 roll 对它们无事可做，
也不另造一个空间旋转——那会是另一种操作。**batch = 1 时 `shuffled` 是恒等**：
批内错配没有伙伴，而验证与评估路径（`score_validation`、`evaluate_local`）都是单窗口一次
前向；这一性质写在冻结协议里、由测试断言、并在 limitations 中标注，见 §12.1。

---

## 3. 判据与预登记 primary（跑前冻结）

判据只有两条，都不新增、不放宽：**#60 比较器的逐 seed 同号规则（`depth=0`）**，
以及**协议里跑前冻结的 primary 判定文字**。

**预登记（`protocol.json` 的 `primary_registration`，每 seed 第一次 step 之前落盘并回读）**：

- 变量 **t2m**，时效 **6/12/24/48/72 h**；
- 四对：**B−A**（信息+容量）、**B−E**（信息）、**E−A**（容量/偏置）、**P−A**（容量+错配对）；
- 判定文字（原文，跑前冻结、跑后未改）：

  > Read on the three-seed paired mean delta of t2m, per lead, only where the comparator's
  > per-seed sign rule makes the cell sign-consistent (disagreement is unresolved and is
  > never averaged into a verdict). Write s_E = (E-A)/(B-A), s_P = (P-A)/(B-A) and
  > s_I = (B-E)/(B-A) for that cell. Then: (i) if s_E >= 0.5 the cell reads
  > 'capacity/bias dominated - the module without information reproduces at least half of
  > B-A'; (ii) if s_E <= 0.25 and s_I >= 0.75 the cell reads 'information dominated';
  > (iii) between 0.25 and 0.5 the cell reads 'mixed, attribution unresolved' and no
  > attribution is claimed; (iv) if E-A is opposite in sign to B-A the cell reads
  > 'capacity/bias does not reproduce the gain at this lead'. The headline of the round is
  > the modal reading over the sign-consistent leads with the full per-lead table reported
  > beside it; if the leads disagree, that disagreement is the result and is reported as
  > such. P is reported next to every cell: if s_P >= 0.5 as well, the round states that
  > real-valued but mispaired fields suffice for at least half of B-A. The variable and the
  > leads above are fixed by this text and are never re-picked after the numbers exist; no
  > threshold beyond these fractions is introduced, and no existing threshold is relaxed.

**读者实现的一处缺陷与修正（如实记录）**：第一版读者额外要求 `P−A` 也逐 seed 同号
（冻结文字只把 P−A 列为**并列报告项**，参与判读的是 B−A / B−E / E−A 三对），
于是五个时效全被判成 `unresolved`。读者已修正为逐字实现上述文字；**判据文字未改一字**
（它在冻结的 `protocol.json` 里，digest 12/12 相同），**任何数字未改**。修正后重跑 finalize：
其余 6 个派生文件**逐字节相同**，只有 `primary` 块变化。修正前的输出保存在
`outputs/r7_71_72_round_three/paired_comparison_before_reader_correction.json`（headline：
`no sign-consistent primary cell`），任何人都可以两份对照着看。

---

## 4. 预算与协议冻结

| 项 | 值 |
| --- | --- |
| 协议落盘 | 每 seed 的 `protocol.json` 在训练目录创建**之前**写入（`open("x")`），并在第一次 `optimizer.step()` 之前回读校验 digest 与 primary 登记 |
| 12 个 run 的 digest | 12/12 = `d62db6db…`（逐 run 记录在每份 `seed_result.json` 的 `training[*].protocol_sha256`） |
| 只读 val | 全部 60 次评估（4 臂 × 5 时效 × 3 seed）`split=val`；`test_read: false`；`refused_test_manifest` 在读 manifest 前拒绝 `test.jsonl` |
| 案例集 | 6h 22 / 12h 21 / 24h 19 / 48h 15 / 72h 11（四臂逐时效案例数**相同**，不同即失败；比较器 `case_identity: exact`） |
| 实测 GPU-h | 1976.1 s = **0.5489**（`elapsed_seconds` 逐臂求和）；训练循环墙钟 1979.6 s = 0.5499 |
| 单次实验墙钟 | 14.0 / 14.0 / 13.8 min（`DEADLINE_SECONDS=1800` 在每臂开始前检查） |
| 峰值显存 | 258.0 MiB（四臂、三 seed 相同量级） |
| 提交上限 | 本轮自设 ≤0.9 GPU-h（实测 0.5489）；第二批 24 − 2.463 = **余 21.537 GPU-h** |

**启动记录（如实）**：一次启动，`DEV=1 setsid nohup bash scripts/run_71_72_round_three_multiseed.sh`，
2026-09-28T18:18:38（+08:00）发出，19:01:23 打印 `R7_71_72_ROUND_THREE_DONE`；
三个 seed 顺序执行，无中断、无部分 seed 被丢弃，也没有任何未完成的 run 被当作数字。

---

## 5. 四臂参数 / FLOPs 与「E≡B」的实测断言

`count_parameters` 与 `count_forward_flops`（同一约定：`FlopCounterMode`、`enable_grad`、
无参数 hook；backward 单独测、不假定 2×），在协议冻结**之前**测量并写进协议：

| 臂 | 参数 | 前向 FLOPs | 前向+反向 FLOPs | 模式 |
| --- | --- | --- | --- | --- |
| A `process_pooled` | 2,799,779 | 12,843,777,408 | 38,417,551,104 | 关 |
| B `process_spacetime_only` | 2,819,267 | 12,927,412,608 | 38,665,111,296 | 开 / `fields` |
| E `process_spacetime_constant` | 2,819,267 | 12,927,412,608 | 38,665,111,296 | 开 / `constant` |
| P `process_spacetime_shuffled` | 2,819,267 | 12,927,412,608 | 38,665,111,296 | 开 / `shuffled` |

- **E 与 B 逐张量相同**：同 seed 构造的 `state_dict` 张量名集合与数值**全部逐位相等**
  （`pairwise_shared_tensors["process_spacetime_constant|process_spacetime_only"]
  ["all_tensors_identical"] = true`，三个 seed 都成立），这是 `run_seed` 的前置断言，
  不成立即抛错。
- **E 与 B 算力相同**：参数、前向、前向+反向 FLOPs 三项逐个相等，同样在运行内断言。
- B 与 A 的差仍是 19,488 参数（4 个张量），与前两轮一致；A 与其他三臂的对比仍按
  容量混杂读（limitations）。
- 臂配对同时实测：以 A 为锚，A 的 `state_dict` 按训练器自身的转移规则加载进 B/E/P，
  `applied/ignored` 计数写进产物，加载后逐个张量逐位相等（不成立即失败）。

---

## 6. 训练记录（400 updates，三 seed，全部到 400、无早停）

| seed | 臂 | 选中 update | val MSE（归一化） | 训练秒 | 峰值显存 MiB |
| --- | --- | --- | --- | --- | --- |
| 41 | A `process_pooled` | 400 | 0.217967 | 165.4 | 258.0 |
| 41 | B `process_spacetime_only` | 400 | 0.209510 | 165.6 | 258.0 |
| 41 | E `process_spacetime_constant` | 400 | 0.217576 | 165.7 | 258.0 |
| 41 | P `process_spacetime_shuffled` | 400 | 0.218545 | 166.5 | 258.0 |
| 42 | A | 400 | 0.216359 | 162.3 | 258.0 |
| 42 | B | 400 | 0.208611 | 166.7 | 258.0 |
| 42 | E | 400 | 0.216468 | 165.2 | 258.0 |
| 42 | P | 400 | 0.217033 | 165.9 | 258.0 |
| 43 | A | 400 | 0.216678 | 161.2 | 258.0 |
| 43 | B | 400 | 0.210170 | 163.7 | 258.0 |
| 43 | E | 400 | 0.216460 | 164.4 | 258.0 |
| 43 | P | 400 | 0.216367 | 163.5 | 258.0 |

**校验点上的第一层信号**（归一化 val MSE，越低越好；这不是判据，判据是 §7 的比较器）：
`B − A` = −0.00846 / −0.00775 / −0.00651（三个 seed 同号）；`E − A` = −0.00039 / +0.00011 /
−0.00022（**量级 1e-4，且不同号**）；`P − A` = +0.00058 / +0.00067 / −0.00031（**不同号**）。
即在训练自己选择的检查点上，E 与 A 基本不可区分，而 B 稳定更低。

---

## 7. D7：配对结果（#60 比较器，`depth=0`，逐 seed 同号）

17 个变量 × 5 个时效 = **85 格**；一格只在每个 seed 的 delta 同号时才算 improved/worsened，
否则 unresolved；delta < 0 表示 focus 臂 RMSE 更低。**没有新增任何阈值**。

| 对（focus − baseline） | improved | worsened | unresolved | 同号格数 | 同号格相对量级中位数 | 最大 |
| --- | --- | --- | --- | --- | --- | --- |
| **B − A**（信息+容量） | 36 | 3 | 46 | 39 | 6.5e-2 | 2.9e-1 |
| **B − E**（信息） | **40** | 4 | 41 | 44 | 4.0e-2 | 2.8e-1 |
| **E − A**（容量/偏置） | 14 | **18** | 53 | 32 | 7.4e-3 | 5.5e-2 |
| **P − A**（容量+错配对） | **7** | 10 | **68** | 17 | 8.9e-3 | 4.0e-2 |
| E − P | 15 | 10 | 60 | 25 | 1.1e-2 | 4.7e-2 |
| P − B（错配对 vs 真值） | 3 | **40** | 42 | 43 | 3.7e-2 | 4.0e-1 |

「相对量级」= |三 seed 平均 delta| ÷ 基线臂该格 seed 均值 RMSE（单位无关），
只把量级摆在计数旁边，**不构成新判据**。`beats_baseline_everywhere` 对每一对都是 false。

**逐 lead 计数**（improved/worsened/unresolved）：

| 对 | 6h | 12h | 24h | 48h | 72h |
| --- | --- | --- | --- | --- | --- |
| B − A | 8/0/9 | 7/0/10 | 9/1/7 | 6/0/11 | 6/2/9 |
| B − E | 9/1/7 | 8/2/7 | 9/0/8 | 8/1/8 | 6/0/11 |
| E − A | 5/4/8 | 2/7/8 | 2/5/10 | 3/1/13 | 2/1/14 |
| P − A | 0/4/13 | 0/3/14 | 3/1/13 | 3/0/14 | 1/2/14 |
| E − P | 6/1/10 | 2/1/14 | 1/2/14 | 5/4/8 | 1/2/14 |
| P − B | 0/10/7 | 1/8/8 | 0/9/8 | 0/7/10 | 2/6/9 |

**逐变量计数（每个变量 5 个时效；全部 17 个变量照报）**：

| 变量 | B − A | B − E | E − A | P − A |
| --- | --- | --- | --- | --- |
| t2m | 4/0/1 | **4/0/1** | 2/1/2 | 0/0/5 |
| mslp | 2/0/3 | 2/0/3 | 2/1/2 | 0/0/5 |
| q500 | 0/0/5 | 0/1/4 | 5/0/0 | 1/0/4 |
| q850 | 1/0/4 | 1/0/4 | 0/1/4 | 1/0/4 |
| t500 | 2/0/3 | 2/0/3 | 0/0/5 | 0/0/5 |
| t850 | 3/0/2 | 3/0/2 | 1/1/3 | 0/0/5 |
| u10 | 4/0/1 | 4/0/1 | 2/0/3 | 0/0/5 |
| u250 | 3/0/2 | **5/0/0** | 0/1/4 | 1/1/3 |
| u500 | 3/1/1 | 4/0/1 | 0/4/1 | 1/1/3 |
| u850 | 2/0/3 | 3/1/1 | 1/0/4 | 0/2/3 |
| v10 | 0/0/5 | 0/0/5 | 0/2/3 | 0/3/2 |
| v250 | 1/1/3 | 1/0/4 | 0/0/5 | 0/0/5 |
| v500 | 1/0/4 | 1/1/3 | 1/0/4 | 0/1/4 |
| v850 | 1/0/4 | 1/0/4 | 0/2/3 | 0/0/5 |
| z250 | 3/0/2 | 3/0/2 | 0/2/3 | 1/0/4 |
| z500 | 4/0/1 | **4/0/1** | 0/2/3 | 1/2/2 |
| z850 | 2/1/2 | 2/1/2 | 0/1/4 | 1/0/4 |

**t2m（primary 族）逐 lead 的三 seed 均值 delta 与构成**（负 = focus 更好）：

| lead | B − A | B − E | E − A | P − A | s_E=(E−A)/(B−A) | s_I=(B−E)/(B−A) | 三个 seed 同号 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6h | −1.0454 | −1.0454 | **+0.00000** | +0.0071 | — | — | B−A 是，B−E 是，E−A **否**，P−A 否 |
| 12h | −1.2710 | −1.3183 | **+0.0473** | +0.0019 | **−0.037** | 1.037 | B−A 是，B−E 是，E−A 是，P−A 否 |
| 24h | −1.3735 | −1.2980 | −0.0755 | −0.0437 | **0.055** | 0.945 | B−A 是，B−E 是，E−A 是，P−A 否 |
| 48h | −1.9631 | −1.7163 | −0.2468 | −0.1637 | **0.126** | 0.874 | B−A 是，B−E 是，E−A 是，P−A 否 |
| 72h | −1.0606 | −0.7691 | −0.2915 | −0.1430 | — | — | 四对**全部否** |

（逐 seed delta 明细在 `paired_comparison.json` 的 `cells["<lead>h|t2m"]["seed_deltas"]`，
本文不逐条抄写以免与产物不同步；四臂 seed 均值 RMSE 见下表。）

**头部变量的四臂 seed 均值（val、物理单位、同一评估器、同一案例集）**：

| lead | 变量 | 单位 | A | B | E | P | 8 桶气候态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6h | t2m | K | 3.6839 | **2.6385** | 3.6839 | 3.6910 | 2.7248 |
| 12h | t2m | K | 4.8234 | **3.5524** | 4.8707 | 4.8253 | 2.7242 |
| 24h | t2m | K | 5.6183 | **4.2448** | 5.5428 | 5.5746 | 2.7457 |
| 48h | t2m | K | 6.7335 | **4.7704** | 6.4867 | 6.5698 | 2.4458 |
| 72h | t2m | K | 6.9452 | **5.8846** | 6.6537 | 6.8022 | 2.4089 |
| 6h | mslp | Pa | 219.949 | **194.927** | 218.247 | 220.654 | 484.562 |
| 72h | z500 | m² s⁻² | 915.120 | **650.780** | 902.830 | 847.127 | 503.205 |
| 72h | t850 | K | 6.5829 | **5.8938** | 6.53055 | 6.64586 | 2.8492 |
| 72h | q850 | kg kg⁻¹ | **0.0027343** | 0.0028103 | 0.0027924 | 0.0028624 | 0.0011253 |

（`rmse_seed_mean.json` 有全部 17 变量 × 5 时效 × 4 臂；`rmse_table.csv` 是逐 seed 明细。）

**E − A 的同号格都在小数第三位量级**：32 格同号，相对量级中位数 7.4e-3、最大 5.5e-2
（最大是 72h|u500，worsened，+0.658 m s⁻¹ / 12.0）；t2m 上 E − A = +0.0000 / +0.0473 /
−0.0755 / −0.2468 / −0.2915 K，即**只有 48h/72h 的 t2m 出现 0.25–0.29 K 的模块在场效应**，
方向与第二轮 `C−B`/`D−B` 记到的 0.22–0.52 K 一致，量级小一档。
**B − E 的同号格遍布头部变量**：44 格同号，t2m 五时效中四个同号（−1.045…−1.716 K），
z500 72h −252 m² s⁻²，u250 五时效全同号。

---

## 8. primary 判定（按跑前冻结的判定文字）

按 §3 的判定文字，**在能判读的时效上**：

| lead | s_E = (E−A)/(B−A) | 判定 |
| --- | --- | --- |
| 6h | （E−A 不同号，不可判读） | unresolved：不写归因（B−A 与 B−E 同号且几乎相等，−1.045 vs −1.045） |
| 12h | −0.037 | **capacity/bias does not reproduce the gain**（E−A 与 B−A 反号） |
| 24h | 0.055 | **information dominated**（E−A 只复现 5.5%） |
| 48h | 0.126 | **information dominated**（E−A 只复现 12.6%） |
| 72h | （四对全不同号，不可判读） | unresolved：不写归因 |

**headline（读者按文字给出的原文）**：
`the sign-consistent leads disagree: capacity/bias does not reproduce the gain x1, information dominated x2 - the per-lead table is the result`。

**结论（我自己写的话，不冒充判据）**：在三个可判读的时效上，**容量/偏置复现 B−A 的比例是
−3.7%、5.5%、12.6%**，也就是说 **B−A 的 87%–104% 来自「模块读到了什么」**；
12h 那一格的 E−A 与 B−A 反号（E 比 A 略差 0.047 K），按冻结文字读作「容量/偏置不复现该收益」。
**不存在任何可判读时效支持「该收益主要是容量/偏置能力」**。
两处 unresolved 是**判据本身**的结果（逐 seed 同号不成立），不是可以事后挑选的读法：
6h 的 E−A 三个 seed 是 +0.014/−0.001/−0.013（均值 ~0），72h 的 B−A 本来就不同号
（第二轮同样如此：seed42 的 B−A 在 72h 为正）。

**P（错配对）的并列报告**：P − A 在**五个时效上全部不同号**，均值 +0.007 / +0.002 /
−0.044 / −0.164 / −0.143 K，即**用错配对训练出来的模块与 A 的差别不可与噪声区分**；
而 P − B 在四个时效同号且**worsened**（+1.05 / +1.27 / +1.33 / +1.80 K）。
合并读作：**丢掉字段与样本的对应关系，就把收益丢掉了**（`s_P ≥ 0.5` 的判断在五个时效上
都无法成立，因为 P−A 从不同号）。

---

## 9. 重跑稳定性：A 与 B 对第二轮

第二轮与第三轮的 A、B 臂是同一配置、同一数据、同一协议族、同一 seed，但**协议 digest
与 `model_code_sha256` 都不同**（本轮 `model/**.py` 变更），因此只作稳定性观察，不合并数字。

| 检查 | 结果 |
| --- | --- |
| 逐检查点 val MSE（归一化） | 6 个（seed × {A,B}）全部在 **1e-6 以内**：A 0.217967/0.216359/0.216678 对 0.217967/0.216358/0.216678；B 0.209510/0.208611/0.210170 对 0.209510/0.208611/0.210170 |
| B − A 的 t2m 逐 lead 均值 | −1.0454/−1.2710/−1.3735/−1.9631/−1.0606 对第二轮 −1.0454/−1.2710/−1.3734/−1.9631/−1.0607（差 ≤ 2e-4 K） |
| B − A 的 85 格判定 | **85/85 与第二轮完全相同**（improved/worsened/unresolved 逐格一致） |
| B − A 逐 seed delta 差 | 85×3 个 delta 的最大差 9.6e-2（出现在 72h\|z250，其基线 RMSE ≈ 1.9e3，相对 ~5e-5）；t2m 上 ~2e-5 K |
| 逐 lead 计数 | 8/0/9、7/0/10、9/1/7、6/0/11、6/2/9 —— 与第二轮**逐格相同** |

即：在 GPU 浮点非确定性下（同机同 seed 不保证逐位一致），**A/B 两个臂在本轮被复现到
1e-5 相对量级**，第 2–7 节读到的 B、E、P 差异（1e-2…4e-1 相对量级）都不是这个量级的噪声。

---

## 10. 模式探针（训练后的 checkpoint 上实测）

对每个 seed 的四个训练后 checkpoint，用**两个真实 val 窗口**（同批，batch=2，错配 roll 生效）
跑一次条件模块，记录其贡献与「处处同一个向量」的偏离（`position_/sample_deviation_relative`
= 绝对偏差 ÷ 项范数均值；容差是**数值**容差 1e-6，不是科学判据）：

| 臂 | 声明模式 | 模块在场 | 位置偏离（相对） | 样本偏离（相对） | 与声明一致 |
| --- | --- | --- | --- | --- | --- |
| A `process_pooled` | `fields`（关） | 否 | — | — | 是（无模块） |
| B `process_spacetime_only` | `fields` | 是 | 3.1e-2 / 3.3e-2 / 3.5e-2 | 2.3e-1 / 1.9e-1 / 1.9e-1 | 是 |
| E `process_spacetime_constant` | `constant` | 是 | **0.0 / 0.0 / 0.0** | **0.0 / 0.0 / 0.0** | 是 |
| P `process_spacetime_shuffled` | `shuffled` | 是 | 3.9e-2 / 4.4e-2 / 4.1e-2 | 1.8e-1 / 1.4e-1 / 1.3e-1 | 是 |

（三个数字 = seed 41/42/43；12 个 `consistent_with_declared_mode` 判定**全部为真**，
其中 E 的 3 个 `constant_up_to_float32_rounding` 也都为真、B/P 的 6 个都为假。）

这回答了「臂是不是它自称的那个臂」：训练之后 E 的贡献仍然是**一个向量**（GPU 上偏差恰为
0.0），B/P 的贡献仍随位置与样本变化，A 没有模块。**注意**：`constant` 的「常数」在 CPU 上
只到 float32 舍入（实测最大相对偏差 1.2e-7，约 1 ULP，来自 matmul 对不同行的不同舍入），
机制层面则按精确断言——网络输入逐位相同；判据见 `FLOAT32_CONSTANCY_TOLERANCE` 的注释。

---

## 11. 逐位回归钉住与默认路径等价（D2/D3/D4）

**D4（改动前冻结）**：在改动任何模型代码之前，用当前工作树（`ac6a3ef`；当时 `git status`
只有本测试文件与两处既有未跟踪项 `.zcodeignore`、`tests/fixtures/r7_equivalence_recipe.py`）
跑固定配方，冻结 `spacetime_inputs=True` 路径的
forward 与反向 digest（11 个）；改动后同一配方**逐位相同**：

| digest | 值（截断显示，完整值在测试文件的 `FIELDS_PATH_CAPTURE_DIGESTS` 里） |
| --- | --- |
| `fields.forecast` | `e3117a9e74cd8da1…` |
| `fields.initial_forecast` | `9cdcd55964428689…` |
| `fields.draft_forecasts` | `9d9e0f27013cd613…` |
| `fields.final_correction` | `c1bc5b96ba0b0fc2…` |
| `fields.process_state` | `b87560f2596bcee0…` |
| `fields.context_tokens` | `836c8a45b4177ea4…` |
| `fields.streamed.total` | `d91a4901b196e238…` |
| `fields.streamed.forecast` | `ba9d82711b22a920…` |
| `fields.streamed.process` | `a23dfff43bf07b21…` |
| `fields.streamed.final_forecast` | `9be6491fa8ab44fc…` |
| `fields.gradients`（全部参数与梯度） | `f3eed8aadea93989…` |

敏感度反证（`test_the_frozen_pin_detects_a_changed_path`）：把条件模式换成 `constant` /
`shuffled` 时 `fields.forecast` 与 `fields.gradients` 必须变化，换输入（lead 48h）时
`fields.forecast` 也必须变化，并且**被扰动的冻结修订**（改 `coarse_forecast` 的隐藏宽度）
必须与工作树的 digest 不同——否则「配方什么都没量」也会看起来像通过。

### 11.1 钉住方式的修正：机器本地常量不能跨机断言（如实记录）

第一版的钉住测试把上面 11 个 digest 写成**常量**并断言工作树与之逐位相同。它在本机通过，
但**这个提交的 CI 失败了**（`ci.yml` 的 pytest 步骤，run 36413939126，step "Run unit,
integration and installed-wheel tests"）：同一份代码在另一台机器上算出不同的浮点字节。
本机复现证实了机制，而不是猜的：

| 设置 | 与冻结常量的比对 |
| --- | --- |
| 默认线程 + oneDNN | 11/11 相同（本机捕获时的设置） |
| `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` | `fields.gradients` **不同** |
| `torch.backends.mkldnn.enabled = False` | **8/11 不同** |

即：float32 线性层的结果依赖 BLAS/线程/指令路径，**在 CI 上不成立的不是代码，是我的仪器**。
修正为仓库既有的、与机器无关的方式：**同一进程内**把 `git archive ac6a3ef` 取出的改动前实现
与工作树各跑一次同一配方，按原始 float 字节比对（无容差）；原先那 11 个值保留为
**记录**（`FIELDS_PATH_CAPTURE_DIGESTS`，注明是捕获机的读数、不作断言），
证据文档仍照抄为「改动前的冻结值」。修正后：`OMP_NUM_THREADS=1` 与默认线程下**都是 8 passed**，
即比对比机器无关；反证仍然有效（扰动冻结修订 / 换模式 / 换输入都会让 digest 动）。
同一轮里把 `constant` 模式的两层断言也明确化：网络输入**逐位相同**（精确、与机器无关），
贡献在**数值容差**内为一个向量（相对 1e-5，实测本机 1.2e-7、GPU 0.0；该臂的真实偏离是 9.2e-2，
相差三个数量级，容差不会掩盖死臂）。

**D3（默认路径等价）**：`tests/test_r7_switched_path_equivalence.py` 把当前树与冻结修订
`93d89aa`（改动前的最后一次提交，`git archive` 取出）在同一进程里对跑：
`bitwise_identical: true`、`digest_count: 23`、脚本模式退出码 0；反证测试
（扰动冻结实现必须使 digest 变化）通过。

**D2（输入路径不变）**：`tests/test_r7_input_path_consistency.py` 6 项**未改一字**、
全部通过（含 `test_the_declared_inputs_are_exactly_what_the_model_reads`：
开关打开时读到的字段集合恰好等于 `DECLARED_MODEL_INPUTS`，关闭时严格小于它）。
新增测试另外断言：`constant` / `shuffled` 下缺任一字段仍 `KeyError`、`init_utc_hour=30` 仍
`ValueError`——**控制臂不能借模式换一条输入路径**。

**全仓测试**：`pytest tests/ -q` → **1290 passed, 3 skipped, 2 failed**；两处失败是**同一条既有**
R-044 阻断命中（未跟踪、本会话删不掉的 `tests/fixtures/r7_equivalence_recipe.py`，
见 §13），**不是本轮引入**：本轮新增文件 `tests/test_r7_spacetime_input_modes.py` 命名合规
（8 个测试函数、30 条断言，全部 `test_*`）。

---

## 12. 与冻结设计的两处偏差（如实记录）

**12.1 `shuffled` 在 batch < 2 时是恒等，而不是报错。**
目标长文 §4.3/§6.2 写的是「`batch < 2` 报错」。落地时发现：验证（`score_validation`）与
评估（`evaluate_local` → `rollout_model_input`）都是**单窗口一次前向**（batch = 1），
报错会让 P 臂在第一次验证就崩、根本拿不到数字——而这恰恰是长文 §6.2 想避免的结果
（「写死 batch=2 会让控制臂在评估时静默或直接崩」）。批内置换在单样本批上**必然是恒等**，
不是可以选择的实现细节。因此本轮选择：恒等 + **显式声明**（冻结协议 `field_mode_semantics`
写明、测试 `test_the_shuffled_mode_rolls_the_sample_axis_deterministically` 直接断言
batch=1 时与 `fields` 逐位相同、limitations 写明 P 的训练前向错配而验证/评估前向正确配对）。
**代价**：`P − A` 因此不能读成「错配对在评估时也无害」，只能读成「用错配对训练出来的模块
离 A 有多远」。判据（§3）不涉及这一条。

**12.2 primary 读者按冻结文字修正（见 §3 末段）。** 判据文字与所有数字未改；
修正前后两份 `primary` 输出都在产物里。

**另外一处必须说明的数值事实**：`constant` 臂在 CPU 上的贡献不是**逐位**常数，而是
1 ULP 量级（1.2e-7 相对）的浮点噪声（float32 matmul 对不同行不保证相同舍入）；
本轮 GPU 实测为 0.0。测试与探针因此把「常数」拆成两层断言：网络输入**逐位相同**（精确），
贡献**在数值容差内为一个向量**（1e-6 相对，容差来源写在注释里）。

---

## 13. limitations / scientific_claim / 未做的事

`scientific_claim: false`（协议、逐 seed 结果、合并结果、配对比较四处都带该字段）。

**limitations（协议里冻结的原文，逐条照抄）**：

- one bounded four-arm run at 400 updates, not a convergence or SOTA comparison；
- three seeds: sign agreement across three seeds is consistency, not significance,
  and no significance threshold is introduced or relaxed；
- A 与 B/E/P 在参数与 FLOPs 上按构造不同，凡涉及 A 的对（B−A、E−A、P−A）仍是容量混杂的；
  **只有 B−E 固定了参数、模块与实测 FLOPs，只变「模块被喂了什么」**；
- 控制臂只测容量与字段配对，**不测位置依赖**（第二轮已回答）；
- P 的错配作用于训练前向（batch 2）；验证与发布评估是单窗口一次前向，
  那里批内 roll 没有伙伴、是恒等，因此 P 是「用错配对训练、用正确配对评估」；
- primary 只登记 t2m 的 6/12/24/48/72 h 与三个 seed；其余变量与聚合计数是次要报告，
  不能替代 primary；
- 一个冬季、一年的一个区域：不能做季节/跨年/跨区域结论；
- 只用 val；test 本轮全程封存，**没有为方法选择打开过**；
- val 气候态是 store 的 8 桶（月、时）train-only 均值，不是强季节气候态；
- 第二轮的数字化是不同协议 digest、不同模型 code digest，只作稳定性观察，**从不合并**。

**另外三条本轮特有的**：控制臂新增的 `constant` 语义在 CPU 上只能到 1 ULP（§12）；
primary 读者修正发生在看到数字之后（§12.2）——判据文字与数字未变，但这是本轮
**最需要外部复核**的一步；D4 的钉住仪器第一版把**机器本地**的 digest 当常量断言，
**在 CI 上失败**（run 36413939126），已按仓库既有方式改为同进程双实现比对（§11.1）——
这同样是一次实现缺陷修正，判据与数字未变。

**本机 conventions 状态（不得宣称干净）**：`python tools/check_conventions.py` 仍报
**一条**既有 R-044 阻断命中：未跟踪、本会话删不掉的 `tests/fixtures/r7_equivalence_recipe.py`
（第一轮遗留，本轮未触碰、未提交；`tests/` 下的删除被 `guard_protected_paths` hook 拒绝）。
CI 不受影响（该文件未进版本控制）。本轮**未新增**任何阻断违规；报告型 R-009 通过
（基线 734/1814 是下界，本轮只增不减，基线已随本轮抬到 742/1844，见 §14）。

**未做的事**：

- 未做 C/D 臂（位置化读写，第二轮已回答）、未做交互项第五臂、未做 RW-B、未做 M3/M4/M5；
- 未读 test（全程封存），**未用 test 做任何方法选择**；
- 未新增或放宽任何阈值；未事后更换 primary 变量或时效；
- 未新增数据下载；未合并 main、未 force push、未租 GPU、未写/关任何 issue；
- `A − A` 自检未做（目标标为可选）；
- 未做收敛性/显著性分析（三个 seed 只能给一致性，见 limitations）；
- 未覆盖第一轮/第二轮的产物与文档（两份旧产物目录未改写）。

**下一项的第一个具体动作**：把本轮 `E − A` 记到的那条「模块在场、输入无信息」效应
（t2m 48h −0.247 K、72h −0.291 K，三 seed 同号）与第二轮 `C−B`/`D−B` 的 0.22–0.52 K
放在一起看——具体动作是：读 `outputs/r7_71_72_round_two/paired_comparison.json` 的
`C − B`、`D − B` 两对在 t2m 48/72h 的逐 seed delta，与本轮 `E − A` 的同格逐 seed delta
并排（三项都是「只加/只留模块、无位置读取」的近似对照），判断这 0.25–0.29 K 是否
在两次独立运行里都稳定复现；**若稳定**，下一轮的问题是「这 0.29 K 来自模块的哪一部分」
（例如把 `constant` 模式的 MLP 换成零映射＝真正不给模块任何加性项），**若不稳定**，
下一轮先解决 t2m 长时效上 72h 的 seed42 异号（两轮都出现），再谈机制。

---

## 14. 提交与 CI

- 治理层提交：`model/spacetime_conditioning_r7.py`、`model/weather_forecaster_r7.py`、
  `model/process_forecast_r7.py`、`model/recursive_weather_r7.py`、
  `tests/test_r7_spacetime_input_modes.py`、`scripts/study_r7_71_72_round_three.py`、
  `scripts/run_71_72_round_three_multiseed.sh`、`docs/R7_71_72_ROUND_THREE.md`、
  `docs/decisions/0016-*.md`（+ README 索引）、`docs/rules/CHANGELOG.md`、
  `tools/check_conventions.py`（仅 R-009 基线）
- 提交信息带 **`[model-digest-change]`**（`model/**.py` 字节变化 →
  `model_code_sha256` 从 `8d9262d1…` 变为 `f349adce…`，旧产物只能用其归档 `code.zip` 重放，
  见 Q-009）；`docs/rules/CHANGELOG.md` 已记录该变化与 R-009 基线更新。
- CI：`ci.yml` 在推送到 `r7/weather-reasoning` 时运行（CPU-only）；17 条实验 workflow
  是 commit-message 标签门控，**skipped 是设计行为，不是失败**。判定见 §14 的 run id
  （由提交后的核对补记）。

**提交与 CI 的实际结果**（提交后补记）：

- 工作分支提交：`<见提交记录>`；推送 `origin/r7/weather-reasoning`。
- `ci.yml` run：`<见提交后核对>`。
