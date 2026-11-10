# S3 Stage C2（导数域目标匹配 on/off）读数 + 预声明出口 (a)–(d)

- **日期**：2026-11-10（UTC，渲染时刻）
- **状态**：预声明出口 (a)–(d) 逐条求值；`all = False`、`verdict = permanently-close-derivative-matching-branch`
- **`scientific_claim: false`**；`test_read: false`；`r = 0`
- **协议**：`a3ca9fe7b36bb9e65d11f42a7e7f3026ce6390f934b8a45001842af320b4c764`
- **代码**：`2e0aac88aeb9c658d29dd3e791e27d8fce3f207c`；`model_code_sha256 59340710…`、`training_code_sha256 e7674118…`

---

## §1 预声明工程出口 (a)–(d)

| 判据 | 要求 | 实测 | 结果 |
| --- | --- | --- | --- |
| (a) | 72 h t2m skill 配对均值相对同代码 off 臂**不退化** | 逐 seed Δ：41=-0.011634、42=-0.016342、43=-0.013380；均值 `-0.013785` | **假** |
| (b) | on 臂逐步记录两项匹配且**严格为正** | preflight 四行 total 全正：True；三 seed 的 on 臂 `constraint_terms_recorded`：41=True、42=True、43=True | **真** |
| (c) | off 路径与注册算术**逐位等价** | **该行不由读数证明**：驱动里 `c = True` 是常量，所以它恒为真、不携带信息（见 §4 限制条）；真正支撑它的是 dispatcher 的 `None` 早退与其反证测试，不在本页 | **真** |
| (d) | 无判据被改动 | `scientific_claim=False`、`test_read=False` | **真** |

`exit_conditions`（驱动自算、未手改）：`{"a_72h_skill_mean_delta": false, "all": false, "b_matching_bit": true, "c_off_path_unchanged": true, "d_no_criterion_changed": true, "delta_72h": {"41": -0.01163350798502183, "42": -0.016342101176373247, "43": -0.013380310548156138}, "verdict": "permanently-close-derivative-matching-branch"}`

---

## §2 逐 seed 的 72 h t2m skill（同 seed 配对）

| seed | off | on | Δ(on−off) |
| --- | --- | --- | --- |
| 41 | +0.230542 | +0.218909 | -0.011634 |
| 42 | +0.243262 | +0.226920 | -0.016342 |
| 43 | +0.231662 | +0.218282 | -0.013380 |

六个臂的五个 lead 全部 skill（t2m，分母为注册 D2 值）：

| seed | 臂 | 6 h | 12 h | 24 h | 48 h | 72 h |
| --- | --- | --- | --- | --- | --- | --- |
| 41 | off | +0.709851 | +0.570865 | +0.638135 | +0.409806 | +0.230542 |
| 41 | on | +0.706437 | +0.568616 | +0.630686 | +0.399992 | +0.218909 |
| 42 | off | +0.707532 | +0.570677 | +0.641139 | +0.418349 | +0.243262 |
| 42 | on | +0.702135 | +0.566727 | +0.634089 | +0.405215 | +0.226920 |
| 43 | off | +0.712336 | +0.573275 | +0.642319 | +0.417142 | +0.231662 |
| 43 | on | +0.707202 | +0.568450 | +0.634567 | +0.405630 | +0.218282 |

---

## §3 判决与后续

至少一项预声明出口为假 ⇒ 按预声明**永久关闭「导数域目标匹配」这一支**（`permanently-close-derivative-matching-branch`）：不调权重、不换项、不加 seed、不换名重开。

**如实记录**：本页不解释机制，也不主张该方向在别的形式下同样无效——预声明关闭的是**这一支**。

---

## §4 限制（不得省略）

- `scientific_claim: false`、`test_read: false`、`r = 0`；未读 2023 test。
- 本页所有数字**逐项取自 `readings.json`**，未手抄；`readings.json` 由驱动的 `readings()` 计算。
- 单 seed 单次运行，GPU 训练不保证逐位可复现；checkpoint/源码/数据身份是严格校验的。
- **(c) 与 (d) 恒为真，不构成判别**：驱动 `evaluate_exit` 里 `c = True` 是常量，`d` 只读 `scientific_claim`/`test_read` 两个本页自己写的字段。实测把三 seed 的 skill 全部伪造成 0.5，四项仍全真、判决仍是 `advance-to-independent-review`。**真正带信息的是 (a) 与 (b)。** 该文件在 attempt08 的 95 条执行 pin 之内，改它就是让 seed43 拒绝启动，故本轮只记录、不修。
- 72 h skill 的分母是 train-only 的 D2 reference：`sha256 096c384a25a8a7bb020e0bc3eb0759e8de25ba9e96c692dd9a8ac63be6dfd37f`、`n_selected_steps 32144`、`training_years 2000–2021`（22 年，**止于 2021**，不含 2022 val 与 2023 test）、`format r7-s3-fy-year-coverage-d2-reference-v1`。**这个摘要是本页现算的**：C2 的 `readings()` 不记录分母摘要（该字段在 95 条 pin 之内，本轮无法补），见 brief 的已知缺口条。
- 预声明的 (a) 是**保号判据**（均值 Δ ≥ 0），不是效应量判据；一个接近零的正值不代表有意义的改进。
