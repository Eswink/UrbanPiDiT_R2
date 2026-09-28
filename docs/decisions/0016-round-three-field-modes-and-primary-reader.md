# 0016 V2 第三轮：字段模式的臂内替换、batch=1 的确定性语义、primary 读者按冻结文字实现

- **日期**：2026-09-28
- **状态**：accepted
- **代码 SHA 范围**：`ac6a3ef`（起点，`git rev-parse HEAD`）→ 本轮提交
- **依据证据**：`docs/R7_71_72_ROUND_THREE.md`（本轮证据文档）；
  `docs/goals/v2-round-three-m1-attribution.md`（目标长文 §2 D1–D9、§4 实施顺序、§6 与草案的差异）；
  `model/spacetime_conditioning_r7.py`（`FIELD_MODES` / `apply_field_mode`）；
  `tests/test_r7_spacetime_input_modes.py`（8 个测试函数，含反证）；
  `outputs/r7_71_72_round_three/`（协议 + 12 个 run + 配对比较）；
  `docs/R7_71_72_ROUND_TWO_ATTRIBUTION.md`（上一轮的 B−A 数字）

## Context

第二轮量到 `B − A`（开 `spacetime_inputs`）在 t2m 上 −1.0…−2.0 K，是整场战役唯一的
大效应，但 B 只比 A 多 19,488 个参数且**没有容量控制**：模块本身能表示的偏置与它读到
的信息混在一起。第三轮用两个控制臂把它拆开，落地时出现三个必须写下来的选择——它们都会
影响后续轮次怎么复用这套臂，因此按 R-033 记录。

## Decision

**1. 控制臂的替换发生在模型内部、字段校验之后，字段模式是模块构造参数。**

`FIELD_MODES = ("fields", "constant", "shuffled")`，默认 `fields`；
`apply_field_mode` 在 `require_spacetime_fields` **之后**改写已校验的字段张量：
`constant` 换成同形状零张量（模块在场、算力相同、输入无信息），
`shuffled` 把逐样本字段沿样本轴 `roll(1, dims=0)`（真值、错配对）。
dataset、rollout、`DECLARED_MODEL_INPUTS` 与输入路径一致性测试**不变**：缺字段或字段越界
在三种模式下都照旧抛错，控制臂不能借模式换一条输入路径。
`spacetime_inputs=False` 时给非 `fields` 模式报错，而不是静默忽略。

**2. `shuffled` 在 batch = 1 时是恒等，这是声明而不是报错。**

批内错配需要一个同批伙伴；单样本批没有伙伴，任何批内置换在那里都是恒等。验证与
公开发布评估路径都是**单窗口一次前向**（`score_validation`、`evaluate_local` 均 `unsqueeze(0)`），
因此「batch < 2 报错」会让 P 臂在第一次验证时直接崩、根本无法得分。
选择恒等、并要求在**冻结于跑前的 protocol** 里写明该性质（`field_mode_semantics`）、
由测试直接断言（`test_the_shuffled_mode_rolls_the_sample_axis_deterministically`），
以及在证据文档的 limitations 里写清「P 的错配作用于训练前向（batch 2），其验证/评估前向
携带正确配对」。纬度/经度按构造是**批不变**的（一批样本网格不一致即报错），
批内 roll 对它们无事可做，也不另外发明一个空间旋转。

**3. primary 读者必须逐字实现冻结的判定文字，读者实现缺陷按缺陷修、不按判据改。**

`PRIMARY_DECISION_TEXT` 在第一次 `optimizer.step()` 之前写进 `protocol.json` 并冻结 digest
（12 个 run 全部相同）。判读以 **B−A / B−E / E−A 三对**在各 seed 同号为前提，
**P−A 是并列报告项**，不是前提。第一版读者额外要求 P−A 也同号，导致五个时效全判
「unresolved」——那不是冻结文字要求的读法。修正读者（不改判据、不改任何数字）后重跑
finalize：其余 6 个派生文件逐字节相同，只有 `primary` 块变化；修正前的输出保存在
`outputs/r7_71_72_round_three/paired_comparison_before_reader_correction.json`。

## Consequences

- **好处**：臂的差异只剩「模块被喂了什么」，`B − E` 因此是参数、模块、实测 FLOPs 与
  初始化全部相同的**输入对比**（E 与 B 逐张量相同，前向/反向 FLOPs 相同，run 内实测断言）；
  控制语义写在模型内，任何沿用 `make_model` 的路径（训练、验证、rollout、评估、剖析）
  都自动一致，不存在某一处忘了替换的风险。
- **代价（如实记）**：`constant` 的贡献是「一个向量」而非逐位常数——float32 matmul 对同一
  输入的不同行不保证相同舍入，实测偏差约 1 ULP（1.2e-07 绝对，项范数 ~1）。因此
  「常数」只能断言到**数值容差**（相对 1e-6，`FLOAT32_CONSTANCY_TOLERANCE`），
  机制层面则按精确断言（网络输入逐位相同）。本轮 GPU 实测该偏差为 **0.0**。
- **代价**：P 臂的训练与评估条件不同（见 Decision 2），`P − A` 不应被读成「错配对在
  评估时也无害」，只能读成「用错配对训练出来的模块离 A 有多远」。
- **代价**：读者修正发生在看到数字之后。这是本轮最需要外部复核的一步：判据文字从未改动
  （它在冻结的 `protocol.json` 里），修正只把读者对齐到文字，并且修正前后两份输出都在产物里。
- **不变**：所有既有阈值、比较器判据（#60 逐 seed 同号，`depth=0`）、默认路径逐位行为
  （23 digest 等价 + 字段回归钉住）都不变；`model_code_sha256` 因 `model/**.py` 字节变化而
  变化（`8d9262d1…` → `f349adce…`），提交带 `[model-digest-change]`，旧产物仍须用其归档
  `code.zip` 重放，见 Q-009。
