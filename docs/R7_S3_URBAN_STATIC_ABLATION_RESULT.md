# S3 城市静态协变量的三臂消融：**(a) 为假，按预声明永久关闭该支**

- **日期**：2026-11-14（UTC；训练起于 2026-11-13T17:08:56Z，读数落盘于 2026-11-14T06:38:30Z）
- **变更类型**：实验（三臂 × 三 seed，真实 GPU 轮）
- **触发原因**：执行 `docs/goals/s3-urban-static-ablation.md` 的 P3–P6
- **前置**：计划 `docs/plans/0032-urban-static-covariate-ablation.md`（设计与全部负面后果）、
  决策 **0113**（准入与 placebo 判据）、决策 **0069**（接线，`accepted`）、决策 **0063**（数据），
  教训来源决策 **0109**（(d) 不得是字面量）
- **结论**：**(a) 假、(b) 真、(c) 真、(d) 由代码路径 + 反证测试承载、(e) 真、(f) 真**
  ⇒ `verdict = permanently-close-the-urban-static-input-branch`（读自 `readings.json` 的
  `exit_conditions.verdict`，未改判、未复判）
- **`scientific_claim: false`**、**`test_read: false`**、**`r = 0`**（2023 test 从未打开）
- **GPU-h**：**20.464471**（`stage_ledger.jsonl` 六行 elapsed 之和 ÷ 3600 = 73,672.094 s；
  其中三个 seed 阶段 **20.403652**，CPU 阶段 prepare/archive/reading 合计 218.945 s）

---

## §0 一句话

三臂消融跑完了：**72 h t2m skill 的三 seed 配对 Δ(treatment − control) 是 −0.000743 / −0.003099 / +0.000892，不是三个都 ≥ 0 ⇒ (a) 为假**，按事先冻结的合取规则**永久关闭城市静态协变量这条输入支**，如实记负。

**同一批读数里 (b)(c)(f)(e) 都是真的**：24 h 三 seed 全正 （+0.001704 / +0.006306 / +0.002668）、48 h 三 seed 全正 （+0.005789 / +0.007451 / +0.006158）、6/12 h 全部落在 −0.005 容差内、(f) 的 72 h 归因差 0.045998 / 0.039604 / 0.037011 K 全部 ≥ 0.001 K。
**这些都不改变判定**：判据是合取的，(a) 单独为假即关闭；本轮**没有**重跑、**没有**加 seed、
**没有**放宽任何阈值，也**不**把 (b)(c)(f) 的阳性改写成「部分成功」。

---

## §1 判定点（逐条，判据原文见 brief §3）

| 判据 | 真/假 | 依据（实测数字） |
| --- | --- | --- |
| **(a)** 72 h t2m skill 三 seed 配对 Δ(treatment − control) 全 ≥ 0 | **假** | −0.000743 / −0.003099 / +0.000892；seed41 与 seed42 为负 |
| **(b)** 24/48 h 配对 Δ 三 seed 全 ≥ 0 | **真** | 24 h +0.001704 / +0.006306 / +0.002668；48 h +0.005789 / +0.007451 / +0.006158 |
| **(c)** 6/12 h 配对 Δ ≥ −0.005 | **真** | 6 h +0.003309 / −0.000120 / +0.002502；12 h +0.003253 / +0.005877 / +0.008065 |
| **(d)** control 臂逐位等价 | **由代码路径 + 反证测试承载，不由读数自证** | `model/weather_forecaster_r7.py:115` 的 `pathway = None` 早退 + `tests/test_r7_urban_static_ablation.py:51/65/84` 三条反证测试（实跑 3 passed） |
| **(e)** 预检 | **真** | 6/6 子句实测为真（§4） |
| **(f)** Δ(treatment − placebo) 72 h ≥ 0.001 K | **真** | 0.045998 / 0.039604 / 0.037011 K |

**读数的原始出处**：`outputs/r7_s3_urban_static_ablation_20261113_attempt01/readings.json`，`exit_conditions` 由驱动 `scripts/study_r7_urban_static_ablation.py:250` 的 `evaluate_exit` 自算。
本页**没有**重算这套判定；只做了两件事：① 把 `exit_conditions` 里的每个 Δ 与 `seeds[*][arm][metric]`
逐位对表（15 个 skill Δ + 3 个 (f) Δ，全部 `abs(diff) < 1e-15`），② 把表格数字直接由该文件格式化输出。

---

## §2 身份与可复现性

| 项 | 值 |
| --- | --- |
| `protocol_sha256`（canonical digest，已现算复核） | `cd6958eea98f4df53946dc3c21aa72937f93fb94723599377013eb57692fc020` |
| `protocol.json` 文件 sha256（33,130,530 B） | `7b2b828d5216bccabb7c0d56d948d1d08fd25338c2b92c03a69df446bee1fa9f` |
| `code.zip` sha256（14,302,614 B，1,862 条目，`testzip()` 返回 None） | `c783e1244a5ccb16994202e6a2e6f2880a1dc555dfb893ae158b4f05dd0c5f82` |
| `code.zip` 注释（= `code_commit`，archive 阶段断言它等于协议值） | `c3e736ea3a50ddb40c124231d27da7c67f820cce` |
| `model_code_sha256` | `59340710623ca24c7ee59ebb5711f52193ebfbaa72ed4532b41bdd987f5d9324`（与父 pin `scripts/r7_s3_position_retention_parent.py:24` 相等） |
| `training_code_sha256` | `e767411883ce47887eb11557f3cd16c4b2ad07ca2e06f74c28f33aaa384e81b0` |
| `train_data_identity` | `81307008b572e3ff032d5251e788c582f339b0e3dbbc3fea4226d44c6a335af0` |
| `val_data_identity` | `e5fb694f1b33aad9ba32b4d836c646f184603db544c1278d2cd3081d1aab88ea` |
| `source_sha256`（store） | `1db839554cce30cd14c5730633e166a6e370b3dd8c954fc6572ebdba853b32f5` |
| `readings.json` sha256（24,202 B） | `54e92aa71f6516923054d30156e5b0b818872115e68d105eab79bfbd1d0b11ad` |
| `static_identity` | `8671ed7ce0c82d1ec9ac834e68e2e3c7bbc30e14fa6b8a07501e070d27708b03` |
| 静态场 `output_sha256`（= `preflight.fields_sha256`） | `1e9473f04bf1fa47ac279680a4ac4b54de933fee6afde934478e73f6e2dc37e5` |
| 证据索引 record | `record:s3-urban-static-ablation-negative` |

**可复现等级：`config-reproducible`**（三份 receipt 的 `training_report.reproducibility` 逐字为
`config-reproducible; no failed-attempt resume or bitwise GPU promise`）。协议 + 代码归档 +
三 seed 逐臂 receipt 齐全，**没有**任何逐位重放证据，因此**不得**声称 bit-identical。

**`protocol_sha256` 的复核方式（现算，不是抄写）**：把 `protocol.json` 去掉 `protocol_sha256` 键后
过 `training.r7_experiment.canonical_digest`，得到与记录值逐字符相同的串；`readings.json` 里保存的
`protocol_sha256` 与之相同。三条（`protocol.json` 记录值 / 现算值 / `readings.json` 引用值）一致。

**`evidence_commit` 的口径（如实说明）**：本页与索引 record 在**同一个登记提交**里落盘，
因此写记录时没有任何提交包含本页。索引里的 `evidence_commit` / `experiment_commit` 绑定
`c3e736ea3a50ddb40c124231d27da7c67f820cce`——它是**本轮训练所用的代码提交**（`code.zip` 注释即该 SHA），
也是写记录时的 `HEAD`。同一处理在 `docs/R7_URBAN_STATIC_COVARIATES_REPLAY.md` §11 已有先例
（那条记录绑定登记前的 `HEAD`，而不是包含该页的提交）。**不**把这条写成「evidence_commit 指向本页」。

### §2.1 逐 seed 逐臂 receipt

| seed | 臂 | 父 checkpoint sha256 | 父 endpoint updates | 本臂 checkpoint sha256 | 训练秒 | s/update |
| ---: | --- | --- | ---: | --- | ---: | ---: |
| 41 | control | `e81e4904b65c65daf81d74215eb135e7d306277109c3f3544b979ecfefed3b98` | 1600 | `030cff7cd8fa2cf12d18739fbba4c19363f687dfd40edc2a13261d55cfb2f206` | 4516.853 | 1.1292 |
| 41 | treatment | `e81e4904b65c65daf81d74215eb135e7d306277109c3f3544b979ecfefed3b98` | 1600 | `78a183afd765963be059af953b7083aa80a05eb47cdd92f969a5e79a43378fd0` | 4595.704 | 1.1489 |
| 41 | placebo | `e81e4904b65c65daf81d74215eb135e7d306277109c3f3544b979ecfefed3b98` | 1600 | `b250e978bd135924243326ffdedc5b10d6126a2f8a00c7d614a2dbba0c7da644` | 4566.346 | 1.1416 |
| 42 | control | `7e2cd4cc0d07565e19c660dacdf5f67180ee8e5526f81904dc0b37afcdbe95f3` | 1600 | `5a1c18b00d7cea12deb199c42d4a68c1792658bec51ef7a9a5ba3c77d24d3f2a` | 4632.328 | 1.1581 |
| 42 | treatment | `7e2cd4cc0d07565e19c660dacdf5f67180ee8e5526f81904dc0b37afcdbe95f3` | 1600 | `068d974cbf304c4d16c9686c5cb98e7ec162ec8f6c8b5b7c2b6232d6a4cfebf1` | 4665.125 | 1.1663 |
| 42 | placebo | `7e2cd4cc0d07565e19c660dacdf5f67180ee8e5526f81904dc0b37afcdbe95f3` | 1600 | `141d147ee62e8a783f08ae4f14beb8248449876e0b7b714cabefa860dacd61ee` | 4648.409 | 1.1621 |
| 43 | control | `ee33f690653c625d4b041fb95276dfb239cb9a21dc140e2bb65d26754a855d0c` | 1600 | `d583f458cb0b1b1f1ad30444b6bb1ec84cd4f3ec8cb2e2443140307a1a3d4426` | 4432.543 | 1.1081 |
| 43 | treatment | `ee33f690653c625d4b041fb95276dfb239cb9a21dc140e2bb65d26754a855d0c` | 1600 | `a032cc86a7185929650fac3c2c6928aee37729927abad23d9316645509d524e4` | 4473.646 | 1.1184 |
| 43 | placebo | `ee33f690653c625d4b041fb95276dfb239cb9a21dc140e2bb65d26754a855d0c` | 1600 | `b7400b0ac084ae6f6fe3a7f50f9f317a3d2d67b4023a0822a018fb5d527287a0` | 4471.615 | 1.1179 |

三臂的 `updates=4000`、`batch_size=4`、`reasoning_steps=4` 在九份 receipt 里逐字相同；
`parent_checkpoint_sha256` 在同一 seed 的三臂之间相同（各自分别等于父 pin 的 41/42/43 项）。

receipt 文件本身的 sha256：

- `seed41_receipt.json`（9,334 B）`48360b66635af929896d6dfd3b39dda62896a076e2f0262e1f9a096093cb73a1`
- `seed42_receipt.json`（9,335 B）`44d631dd9822b05c81518e9573880886d82405053b422673d32a726309895375`
- `seed43_receipt.json`（9,331 B）`3ed4e54aca0b4b79f42dcbbbfe96847090ee9e8c089673cb8c9d3e3a504cd6d7`

---

## §3 三臂设计与「只有一个东西不同」的实测

| 臂 | `static_conditioning` | `field` | `projection` | state tensors |
| --- | --- | --- | --- | ---: |
| control | False（规格里**根本没有这个键**） | 不构造 | 不构造 | 155 |
| treatment | True | 发布的 8 层真实静态场 | 发布投影 | 157 |
| placebo | True | 同一 8 层场**逐格空间置换** | **同一份**发布投影 | 157 |

三臂参数量实测均为 **3,286,037**（155 个 parameter tensor），`parameter_delta_zero_per_arm = true`；两个 buffer 不在 `parameters()` 里（`static_buffers_are_not_parameters = true`）⇒ 容量混杂被结构性排除。
treatment 与 placebo 的 state dict 只差一个键：`backbone.static_conditioning_pathway.field`（`treatment_vs_placebo_differ_only_in_field = true`）。

三 seed 的父状态 digest（`preflight.seeds[*].parent_state_digest`）与 control 臂的初始状态 digest 逐个相等：

- seed 41：父 `7c416ccb4054cf7e077788a43b02917e6a2dcf2503058152a7f0ab4b0ab70707`，control 初始 `7c416ccb4054cf7e077788a43b02917e6a2dcf2503058152a7f0ab4b0ab70707`（相等：true）
- seed 42：父 `857d880466f76500ed74c534e757732ae9eff4c2b3f0bbc0fa74b895fcae3623`，control 初始 `857d880466f76500ed74c534e757732ae9eff4c2b3f0bbc0fa74b895fcae3623`（相等：true）
- seed 43：父 `61fdac58b7bfacf828e1eae4968817b1cbd46616442c6d16ac0fbbb63e294e1b`，control 初始 `61fdac58b7bfacf828e1eae4968817b1cbd46616442c6d16ac0fbbb63e294e1b`（相等：true）

---

## §4 (e) 预检：6/6 子句实测为真

`preflight.clauses` 逐条（`readings.json` 与 `protocol.json` 的 `preflight` 块**逐字段相等**，已对表）：

| 子句 | 实测 |
| --- | --- |
| `eight_layers_finite_on_65x65` | `true` |
| `static_identity_verified` | `true` |
| `parameter_delta_measured_zero_all_arms` | `true` |
| `placebo_multiset_bitwise_identical` | `true` |
| `strict_load_all_seeds_all_arms` | `true` |
| `treatment_placebo_differ_only_in_field` | `true` |

`e_preflight_passed = true`；8 层 × 65×65 全有限；`static_identity` 校验通过（`True`）。

注意 (e) 的一个**已知边界**：预检在 CPU 上算 placebo 对比与参数计数，它证明「量级与结构对」，
**不**证明「能在 GPU 上跑」——上一轮 Stage A 正是在这条上栽过（CPU 全绿、GPU 第一次 update 就崩）。
本轮没有出现该类失败（三个 seed 阶段全部 `exit_code 0`；每个阶段内在同一张卡上顺序跑完三臂），
但这是运行结果，不是预检的功劳。

---

## §5 逐 lead 读数（2022 val，17 变量中的 t2m，full 65×65）

### §5.1 绝对 skill（`1 − (rmse / climatology_rmse)²`）

| 臂 | seed | 6 h | 12 h | 24 h | 48 h | 72 h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 41 | 0.665328 | 0.483553 | 0.575584 | 0.310554 | 0.129976 |
| control | 42 | 0.646337 | 0.461454 | 0.561072 | 0.311217 | 0.144711 |
| control | 43 | 0.672202 | 0.491887 | 0.578259 | 0.325339 | 0.137418 |
| treatment | 41 | 0.668638 | 0.486806 | 0.577288 | 0.316343 | 0.129233 |
| treatment | 42 | 0.646217 | 0.467331 | 0.567377 | 0.318669 | 0.141613 |
| treatment | 43 | 0.674704 | 0.499952 | 0.580927 | 0.331497 | 0.138310 |
| placebo | 41 | 0.661870 | 0.476528 | 0.570698 | 0.298007 | 0.107234 |
| placebo | 42 | 0.641024 | 0.452613 | 0.555877 | 0.298633 | 0.122823 |
| placebo | 43 | 0.670127 | 0.487871 | 0.576338 | 0.317979 | 0.120723 |

### §5.2 绝对 RMSE（K）

| 臂 | seed | 6 h | 12 h | 24 h | 48 h | 72 h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 41 | 2.267082 | 2.816794 | 2.554867 | 3.259357 | 3.662812 |
| control | 42 | 2.330518 | 2.876427 | 2.598180 | 3.257789 | 3.631661 |
| control | 43 | 2.243679 | 2.793974 | 2.546803 | 3.224219 | 3.647111 |
| treatment | 41 | 2.255845 | 2.807908 | 2.549733 | 3.245646 | 3.664375 |
| treatment | 42 | 2.330914 | 2.860690 | 2.579450 | 3.240120 | 3.638234 |
| treatment | 43 | 2.235101 | 2.771712 | 2.538736 | 3.209472 | 3.645226 |
| placebo | 41 | 2.278764 | 2.835886 | 2.569532 | 3.288883 | 3.710373 |
| placebo | 42 | 2.347959 | 2.899944 | 2.613510 | 3.287415 | 3.677837 |
| placebo | 43 | 2.250768 | 2.804992 | 2.552597 | 3.241759 | 3.682237 |

### §5.3 配对 Δ(treatment − control)，skill

| lead | seed 41 | seed 42 | seed 43 | 三 seed 全 ≥ 0（(b)(c) 口径） |
| ---: | ---: | ---: | ---: | --- |
| 6 h | +0.003309 | −0.000120 | +0.002502 | `true`（要求 ≥ −0.005） |
| 12 h | +0.003253 | +0.005877 | +0.008065 | `true`（要求 ≥ −0.005） |
| 24 h | +0.001704 | +0.006306 | +0.002668 | `true`（要求 ≥ 0） |
| 48 h | +0.005789 | +0.007451 | +0.006158 | `true`（要求 ≥ 0） |
| 72 h | −0.000743 | −0.003099 | +0.000892 | `false`（要求 ≥ 0） |

72 h 的配对差均值 **−0.000983**。control 臂自身 72 h skill 的三 seed 极差是 **0.014736**，所以 72 h 的 |Δ| ≤ 0.0031 落在 seed 间离散之内。
**这条观察不改变判定**：判据是「三 seed 全 ≥ 0」，不是「均值 ≥ 0」或「不显著为负」；
按冻结口径 (a) 为假，本轮**没有**用这个观察去重开轮或放宽出口。

### §5.4 (f) 的归因差：Δ(placebo − treatment) 的 72 h RMSE，单位 K

| seed | placebo 72 h RMSE | treatment 72 h RMSE | Δ(placebo − treatment) |
| ---: | ---: | ---: | ---: |
| 41 | 3.710373490 | 3.664375411 | **+0.045998079** |
| 42 | 3.677837074 | 3.638233501 | **+0.039603573** |
| 43 | 3.682236505 | 3.645225810 | **+0.037010694** |

阈值 0.001 K 由 brief §3.1 冻结前写定（wiring 轮 on/off Δ 落在 +0.0014…+0.0043 K，
那是 120 updates 的噪声上界）。三 seed 分别为阈值的 **46.00× / 39.60× / 37.01×**。

---

## §6 (d) 的支撑：代码路径 + 会失败的反证测试，**不是**读数自证

决策 0109 的教训是 C2 的 `c = True` 是**字面量**、`d` 只读两个自写常量，所以伪造读数仍然 `all=True`。
本轮因此**没有**在 `evaluate_exit` 里写任何 (d) 的布尔值，只留了一个说明字段：

`exit_conditions.d_control_bitwise_not_evaluated_here`（原文照录）：

> carried by the static_conditioning=False -> pathway=None code path plus tests/test_r7_urban_static_ablation.py's counter-proof; decision 0109 forbids a literal verdict in the driver

**① 代码路径**（`model/weather_forecaster_r7.py`）：

```python
# :115  构造：先置 None，只有 static_conditioning 为真才建
self.static_conditioning_pathway=None
if self.static_conditioning:
    field,variables=load_static_fields()
    self.static_conditioning_pathway=StaticConditioning(...)

# :149  前向：身份检查，None 时**整段跳过**，不做任何算术
if self.static_conditioning_pathway is not None:
    context=context+self.static_conditioning_pathway(token_hw=token_hw).to(tokens.dtype)
```

而且 `scripts/r7_urban_static_ablation_protocol.py:163-181` 的 `arm_spec` 对 control 臂
**连 `static_conditioning` 键都不加**（`False` 也不写）：它先断言归档规格里没有这个键，
否则直接抛错——所以 control 臂得到的字典与已注册配方逐字相同，不是「加了个 False」。

**② 会失败的反证测试**（`tests/test_r7_urban_static_ablation.py`）：

| 测试 | 它抓住什么 |
| --- | --- |
| `:51 test_control_arm_constructs_no_static_pathway` | 断言 `model.static_conditioning is False`、`backbone.static_conditioning_pathway is None`、state dict 里**没有任何** `static_conditioning_pathway` 键 |
| `:65 test_the_control_spec_carries_no_static_key_at_all` | 断言 control 规格里**没有**该键，而 treatment/placebo 的是 `True` |
| `:84 test_the_control_forward_never_touches_the_pathway` | 造一个「一访问就抛异常」的哨兵并断言它真会炸；若 trunk 变成无条件读取，control 臂会在这里抛错而不是静默通过 |

**实跑（本次登记会话内）**：

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_r7_urban_static_ablation.py::test_control_arm_constructs_no_static_pathway \
  tests/test_r7_urban_static_ablation.py::test_the_control_spec_carries_no_static_key_at_all \
  tests/test_r7_urban_static_ablation.py::test_the_control_forward_never_touches_the_pathway
```

输出：`...  [100%]` / `3 passed in 1.56s`。整模块另跑一次：`32 passed in 3.70s`。

**这条支撑能做到什么、不能做到什么（如实写）**：它证明 control 臂**不构造**、**不读取**、state dict 里
**没有**那两个 buffer，从而那部分是「改动前的算术路径」。它**不**是本轮的一次前向逐位比对——
本轮**没有**跑「同一权重、同一 batch 下 control 与改动前代码的前向输出 sha256 相等」这类位级见证。
因此严格的说法是：**结构上等价 + 该路径未被触碰**，而不是「实测逐位相等」。

---

## §7 (f) 为真，但它不能说什么

**(f) 是归因判据，不是成功判据。** 本轮 (f) 为真，可是要把它读准，必须同时看两件事：

**① placebo 与 treatment 差两件事，不只一件。** 逐格置换保住了每个协变量的原始多重集
（8/8 变量排序后逐位相同），但 `static_features` 先做 2×2 patch 平均再中心化，所以置换把每个
token 的特征朝全局均值收缩：

| 量 | treatment | placebo | 比值 |
| --- | ---: | ---: | ---: |
| 原始多重集（排序后逐位） | — | — | **8/8 相同** |
| per-token 特征标准差 | 1× | — | **0.6566×** |
| 192 维投影码标准差 | 1× | — | **0.6920×** |
| 逐 token 码 L2 均值 | 4.8332 | 3.2066 | 0.6635× |
| 空间自相关（四邻域 Moran 型均值） | **0.7435** | **−0.000117** | — |

⇒ placebo 同时**没有空间结构**且**token 幅度更低**。所以 (f) 真只能支持：
「**相干的、全幅度的静态项胜过被打散且衰减的版本**」，**不能**单独说「效应来自空间结构」。
这句话在跑之前就写进了 `proto.LIMITATIONS`、`frozen_constants()["decision"]` 与
`preflight.placebo_contrast.confound_statement`，并由 `test_the_placebo_contrast_reports_its_own_attenuation`
钉住（若未来把它改成「干净」，该测试转红）。备选构造（2×2 / 4×4 块置换）也测了并记为
`rejected_constructions`：token 幅度保持更好（0.7843 / 0.7885）但**残留空间结构**（Moran 0.3028 / 0.3812）。
看到这些数字后**没有**改构造——那会变成按结果挑 placebo。

**② 更关键：72 h 上 placebo 比 control 还差。** 把 placebo 臂与 control 臂相比（同 seed 配对）：

| seed | control 72 h skill | treatment 72 h skill | placebo 72 h skill | control − placebo | treatment − placebo |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | 0.129976 | 0.129233 | 0.107234 | **+0.022741** | **+0.021998** |
| 42 | 0.144711 | 0.141613 | 0.122823 | **+0.021888** | **+0.018789** |
| 43 | 0.137418 | 0.138310 | 0.120723 | **+0.016695** | **+0.017587** |

treatment 与 control 在 72 h 上实质打平（Δ 见 §5.3，两负一正，均值 −0.000983），而 placebo 三 seed
一致地比 control 差 **−0.0167 到 −0.0227 skill**。**因此 (f) 的通过主要来自「placebo 被弄坏了」，
不是来自「treatment 变好了」**——它证明「把静态场打散并衰减有害」，不证明「真实静态场有用」。
这一条在判定上不影响 (a) 的结论（(a) 已经独立地为假），但它是本页最重要的诚实读法：
**若有人只看 (f) 就宣布「三臂消融通过」，那是把归因判据当成功判据**，而 brief §3.1 与
`proto.decision.text` 都逐字禁止这种读法。

---

## §8 成本（实测，不是估计）

| 阶段 | elapsed | GPU-h | 设备 | soft overrun | hard overrun |
| --- | ---: | ---: | --- | ---: | ---: |
| prepare | 207.075 s | 0.057521 | `CPU（device=null）` | 0.0 | 0.0 |
| archive | 6.043 s | 0.001679 | `CPU（device=null）` | 0.0 | 0.0 |
| seed 41 | 24,442.909 s | 6.789697 | `GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b` | 0.0 | 0.0 |
| seed 42 | 24,893.958 s | 6.914988 | `GPU-9d1624af-9d77-aa7c-0620-b6cb778f4ced` | 0.0 | 0.0 |
| seed 43 | 24,116.282 s | 6.698967 | `GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b` | 0.0 | 0.0 |
| reading | 5.827 s | 0.001619 | `CPU（device=null）` | 0.0 | 0.0 |
| **合计** | **73,672.094 s** | **20.464471** | | **0.0** | **0.0** |

- **GPU 阶段（三个 seed，含训练 + 五档评分 + worker 开销）**：73,453.149 s = **20.403652 GPU-h**
- **CPU 阶段（prepare/archive/reading）**：218.945 s（prepare 207.075 s, archive 6.043 s, reading 5.827 s）
- **总 elapsed（六行之和，账本行取这个数）**：73,672.094 s = **20.464471 GPU-h**
- **逐 seed receipt 里的训练 + 评分**：
  - seed 41：训练 13,678.902 s（三臂）+ 评分 7,763.746 s（三臂 × 五档）= 21,442.648 s
  - seed 42：训练 13,945.862 s（三臂）+ 评分 7,949.694 s（三臂 × 五档）= 21,895.556 s
  - seed 43：训练 13,377.805 s（三臂）+ 评分 7,763.478 s（三臂 × 五档）= 21,141.282 s
- **receipt 之外的 seed 阶段开销**：seed41 3,000.261 s、seed42 2,998.401 s、seed43 2,975.000 s，合计 **8,973.663 s**（占 seed 阶段 12.2%）。
  这部分是 worker 启动、评估集构造与落盘等，**本轮未进一步归因**——如实记为「已测量但未归因」，
  不写成「训练时间」。
- **墙钟**：`stage_ledger` 首行 `2026-11-13T17:08:56+00:00` → 末行 `2026-11-14T06:38:30+00:00` = **48,574 s = 13.4928 h**
  （**两卡并发**：seed41 在 `GPU-408a…` 与 seed42 在 `GPU-9d16…` 于 `17:09:04` 同时启动；
  seed41 于 `23:56:27` 结束后，seed43 在 `23:56:27.7` 被拉到 `GPU-408a…` 上启动 ⇒ 同一时刻最多 2 个 worker。
  墙钟 13.49 h 明显小于 20.40 GPU-h 就是并发的结果。R-054 未对任何邻居进程发信号。）
- **磁盘**：`outputs/r7_s3_urban_static_ablation_20261113_attempt01/` **1,663,247,240 B**（1.549 GiB）+ `outputs/r7_s3_urban_static_ablation_code.zip` **14,302,614 B** = **1,677,549,854 B**
- **网络**：**0**（训练 worker 入口调用 `scripts/r7_m3_offline.deny_network()`，见
  `scripts/study_r7_urban_static_ablation.py:314`；本轮未下载、未重下任何数据）
- **overrun**：六行 **soft 0.0 / hard 0.0**；per-seed 软预算 64,800 s，实测单 seed 阶段 24,116–24,894 s（占软预算 38.4%）

**估计 vs 实测（如实分开）**：协议冻结的 per-seed 估计是 **21,079.5 s**，其依据是**另一轮**的
1.1096 s/update（p4000）+ 单臂五档评分 2,588.1 s × 3。实测：

- s/update：1.1081–1.1663，**估计 1.1096 落在实测区间内**（训练侧估计准确）
- 单臂五档评分（以 control 臂为样本）：seed41 231.873+302.684+410.997+674.095+959.590 = **2,579.24 s**、seed42 229.745+308.324+415.746+698.501+988.810 = **2,641.13 s**、seed43 229.660+315.232+403.711+689.261+965.526 = **2,603.39 s**，**与借用的 2,588.1 s 同量级**（−0.3% … +2.0%）
- per-seed 阶段实测 24,116.3–24,894.0 s，
  比 21,079.5 s 的估计**高 14.4%–18.1%**，差值来自上面那笔未归因开销
- 总预算：planned 115,200 s / hard 230,400 s，实测 73,672.1 s（占 planned 64.0%）

---

## §9 结论（按 brief §3.1 与冻结的 `decision` 块）

**(a) 为假 ⇒ 按预声明永久关闭城市静态协变量这条输入支，如实记负。**

`protocol.decision.close` 逐字：

> any of (a)(b)(c) false -> permanently close the urban static input branch

**这不是「(a) 真而 (f) 假」那条分支**，本轮 (f) 为真；但两者在结论上同样指向关闭，只是理由不同：

- 本轮关闭的理由是 **72 h 主判据未达成**：三 seed 里有两 seed 的 Δ 为负。
- 若未来有人想用「24/48 h 三 seed 全正」重开这条支，必须先解释为什么 72 h 的符号可以不算——
  那将是对已冻结判据的事后修改，本轮**不做**，并把它记为一条明确禁止。

**brief §0 动手前写下的那句话，照录（一字不改）**：

> 不承诺本轮一定有正面结果：若 (a)(b)(c) 任一为假或 (a) 真而 (f) 假，如实记负并停；那时输入侧与损失侧两侧主要方向都已测过，论文的正面结果只能靠证伪章与机制发现，这一点动手前就承认，不得等读数出来再改叙事。

**本轮就是这一条兑现**：(a) 为假，输入侧（城市静态协变量）与损失侧（C1/C2/P/R/C/A 六轮）
的主要方向**都已测过**。截至本页，本方向**没有**任何一个通过预声明出口的科学正面结果；
论文的正面结果只能靠**证伪章与机制发现**。这句话在读数出来之前写在 brief 里，现在不改写、不软化。

**明确了不做的三件事（本轮也没做）**：不重跑、不加 seed、不改 4,000 updates、不换 placebo 构造、
不放宽 (a) 的「三 seed 全 ≥ 0」、不把 (b)(c)(f) 的阳性拼成「部分成功」。

---

## §10 局限（如实列）

1. scientific_claim: false - this round reports paired arm differences on one model, one instance, one resolution (65x65) and one region (27-43N / 107-123E); it is not a claim about global weather forecasting and not a claim about the real atmospheric effect of urbanisation
2. the placebo's per-cell permutation preserves each covariate's raw multiset exactly but the pathway reads 2x2 patch means, so the placebo's token-level amplitude is measured lower than the treatment's (see preflight.placebo_contrast); exit (f) cannot separate 'spatial structure' from 'code amplitude' on its own
3. the static field is a derived statistic of real native observations aggregated to 0.25 deg cell edges, NOT interpolation and not an observational truth field (decision 0063); the result's domain of applicability is this published covariate set
4. pop_density_km2 has a single source: LandScan is account-gated and was recorded failed-no-fallback, so no cross-source consistency check exists for the population layer
5. GHSL built_surface and population are five-yearly cross-sensor composites, not annual observations; they are 'that epoch's product', not 2020 truth
6. the five lc_* shares do not sum to 1 (unmapped share about 0.23-0.26), so lc_bare_frac is bare land's share among mapped classes, not its absolute share
7. the projection is a fixed random matrix, so no per-covariate attribution is available: even (a) and (f) both true would not license 'built fraction is the driver'
8. 4,000 updates is borrowed from Stage A's dose argument (it moved its own direct target three seeds consistently at this budget); a negative here therefore cannot be attributed to a dose too small to act, but a longer adaptation period is not excluded
9. three seeds are a paired comparison against same-code arms, not a significance claim; no multiplicity correction is applied because the five exits are predeclared and conjunctive
10. the 2023 test split is never read: r = 0, and every number here is 2022 validation
11. **本轮的一次前向位级见证缺失**：(d) 只有结构证据（§6），
  没有「control 臂与改动前代码在同一权重、同一 batch 下前向输出逐位相同」的测量。
12. **(f) 的归因力被 placebo 的双重差异削弱**（§7）：
  置换同时降低 token 幅度（0.6566× 特征、0.6920× 码），因此 (f) 真不等于「空间结构是原因」。
13. **72 h 的 (a) 阴性幅度小于 seed 间离散**（三 seed skill 极差 0.014736，|Δ| ≤ 0.0031），所以这是一个「未能证明有增益」的阴性，**不是**一个「已证明有害」的阴性；
  按冻结的合取规则它仍然关闭该支，因为规则问的是符号，不是效应量。
14. **placebo 比 control 更差这件事本身未做机制归因**（§7②），
  本轮只记录符号与幅度，不解释为什么一个被打散且衰减的 8 通道项会稳定地伤害 72 h skill。

---

## §11 复现命令

```bash
# 轮次状态（current_round_goal 必须指向本轮 brief）
.venv/bin/python tools/r7_round.py state

# 读数与判定（不重跑训练，只读已有产物）
cd outputs/r7_s3_urban_static_ablation_20261113_attempt01
.venv/bin/python -c "import json; d=json.load(open('readings.json')); print(json.dumps(d['exit_conditions'], indent=2))"

# 协议 digest 现算复核（应与记录的 cd6958ee… 逐字符相同）
.venv/bin/python -c "import json; from training.r7_experiment import canonical_digest as c; b=json.load(open('protocol.json')); r=b.pop('protocol_sha256'); print(r); print(c(b)); print(r==c(b))"

# (d) 的三条反证测试
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_r7_urban_static_ablation.py
```

**重跑训练**（不推荐、也不在本次授权内的「再试一次」，仅列出入口）：
`.venv/bin/python tools/r7_round.py stage-seeds` 会因既有 receipt 逐 seed 跳过；真要重跑须新 `attempt`
与新协议，本轮**不做**。

---

## §12 相关记录

- 目标长文（唯一执行源）：`docs/goals/s3-urban-static-ablation.md`
- 设计与全部负面后果：`docs/plans/0032-urban-static-covariate-ablation.md`
- 准入裁决：`docs/decisions/0113-urban-static-ablation-and-placebo.md`
- 接线决策：`docs/decisions/0069-static-covariate-wiring.md`；数据决策 **0063**
- 教训来源：决策 **0109**（(d) 不得是字面量）
- 数据页：`docs/R7_URBAN_STATIC_COVARIATES_ACQUISITION.md`（获取与发布，0 GPU-h）
- 证据索引：`docs/R7_EVIDENCE_INDEX.jsonl` 的 `s3-urban-static-ablation-negative`
- 上一轮同类阴性（损失侧）：`docs/R7_S3_SPECTRAL_GEOMETRY_A_RESULT.md`

---

**`scientific_claim: false`**。本页不声称任何科学通过、不声称 SOTA、不声称「城市静态协变量有效/无效」的
一般结论；它只报告一个模型、一个实例、一个分辨率、一个区域、一个已发布协变量集上、
三 seed 配对的三臂差，以及这些差按**事先冻结**的合取判据给出的判定。
