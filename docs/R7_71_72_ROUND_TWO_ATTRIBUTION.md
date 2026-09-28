# #71 / #72 第二轮：四臂 × 三种子——把「时空输入」与「位置依赖」与「容量」拆开

**状态：本轮的代码、测试与一轮有界实验均已完成；科学结论按下面的局限读，不得外推。**
本文件**不覆盖**第一轮那份（`docs/R7_71_72_M1_AND_RWA.md` 仍是第一轮的唯一证据）；
两份文档的 `protocol_sha256` 与 `model_code_sha256` 都不同，**数字不可相加、不可并列**。

| 项 | 值 |
| --- | --- |
| 起点 SHA | `36bf10b`（开工时 `git rev-parse HEAD` = `36bf10bd97bbfe181b9e3023c349ec0e7438c475`） |
| 数据 | M2 双月段 `outputs/r7_m2_segment/store/manifests`：train 186 / val 22 / test 26（**本轮 test 未读**） |
| 协议 digest | `5367fc597c4f9251d9009817b7fd1a832695a7a86c1e2cd2c44b9004737462a7`（**12 个 run 全部相同**，逐 run 核对） |
| `model_code_sha256` | `8d9262d1abb537f11fc0b74b98ec30e450eadf4e7bd7300ad32a957de5e38e44` |
| 实测 GPU 训练 | **0.5253 GPU-h**（12 份 `training_report.json` 的 `elapsed_seconds` 求和 = 1891.0 s） |
| 本轮预算 | ≤1.0 GPU-h（自设）/ 第二批授权 ≤24 GPU-h（已用 1.389 → **1.914**） |
| 墙钟 | 三个 seed 顺序跑，每 seed 一个进程；单进程 ≈9.6–14 min（远低于 30 min 上限） |
| 实验产物 | `outputs/r7_71_72_round_two/`（协议、逐 seed 结果、合并结果、配对比较、四张表、位置探针） |
| 逐位等价证据 | `tests/test_r7_switched_path_equivalence.py`（23 digest + 反证，**实跑**） |
| 池化钉住证据 | `tests/test_r7_process_readout_pooled_query.py`（7 项）+ 训练后位置探针 |
| 决策 | `docs/decisions/0014-round-two-pooled-query-capacity-control.md` |

---

## 1. 交付物逐条核对

| # | 要求 | 状态 | 证据 |
| --- | --- | --- | --- |
| D1 | 新增「池化 query」构造开关（默认关、bool 校验与既有开关同风格），默认路径不产生任何新参数 | **完成** | `model/process_readout_r7.py`、`model/process_forecast_r7.py`；参数张量集合与总量在开启前后**逐一相同**（2,968,259） |
| D2 | `tests/test_r7_switched_path_equivalence.py` 实跑逐位相同（23 digest） | **完成** | `pytest -q` 4 passed；脚本模式 `bitwise_identical: true`、`digest_count: 23` |
| D3 | 四臂臂配对**实测**：跨四臂共享张量逐位相同、锚臂 `state_dict` 应用到其余三臂、`applied/ignored` 写进产物 | **完成** | 每个 seed 的 6 个臂对全部 `shared_tensors_identical: true`；锚臂转移 115 applied / 0 ignored 且加载后逐张量相等；训练报告计数与实测一致（不一致即失败） |
| D4 | 池化被钉住：池化时各输出位置逐位相同、非池化时有位置差异；**训练后**探针 D 的 spread ≈ 0、C 显著非 0 | **完成** | 测试 7 项（含 CUDA 实形状 1088/1088 行逐位相同）；训练后探针：C `spread/max` = 0.68 / 0.84 / 0.75，D **恰好 0.0**（三个 seed） |
| D5 | 一轮有界实验：新输出目录、协议在每 seed 第一步前冻结、12 个 run 的 `protocol_sha256` 相同、只读 val、test 封存 | **完成** | `outputs/r7_71_72_round_two/`；协议先于 `training/` 落盘；12/12 同 digest；`split` 全为 `val`、`test_read: false` |
| D6 | #60 比较器、`depth=0`，至少 B−A、C−B、C−D、D−B 四对，逐 seed 同号 | **完成** | `paired_comparison.json`（6 对全覆盖，四对必需的已标记 `required: true`）；结果见 §7 |
| D7 | 本文件：起点 SHA、协议 digest、实测 GPU-h、四臂参数/FLOPs、逐变量每时效 RMSE 与胜负、C−D 判定、重跑稳定性、`scientific_claim: false`、limitations、未做的事、下一项 | **完成** | 本文件 |
| D8 | 提交（治理层进版本控制）并核对 CI | **完成** | 见 §14 |

---

## 2. 本轮的问题与四臂设计

第一轮把两个开关一起打开，并记下三条硬局限；本轮逐条消掉，**不引入新机制**：

| 第一轮的局限 | 本轮怎么消 |
| --- | --- |
| 两个开关一起开，分别贡献 unresolved | 增加 B 臂（只开 `spacetime_inputs`）→ B−A 与 C−B 可分别读 |
| 两臂参数/算力不对齐，赢面不能归因于机制 | 增加 D 臂（同一 readout 模块、**同一参数、同一算力**，只把 query 按位置池化）→ C−D 是唯一容量对齐的一对 |
| 只有 2 个 seed | 三个 seed 41/42/43（**一致性，不是显著性**；本轮不引入显著性阈值） |

| 臂 | 开关（`spacetime_inputs` / `positional_process_readout` / `pooled_readout_query`） | 角色 |
| --- | --- | --- |
| A `process_pooled` | 关 / 关 / 关 | 第一轮的参照臂 |
| B `process_spacetime_only` | 开 / 关 / 关 | 单独量 #71 |
| C `process_spacetime_rwa` | 开 / 开 / 关 | 在 #71 之上量 #72 |
| D `process_rwa_capacity_control` | 开 / 开 / **开** | 位置消融的容量控制 |

**D 臂的构造**：`query_norm(context)` 先在**输出位置维**取均值，广播回每个位置，**且不加**位置编码；
attention 仍是 N 个 query × M 个 process token。因此在**给定 process token** 时读取与位置无关，
而模块、参数、键、前向算力与 C 完全相同（§3）。

---

## 3. D1 / D4：池化 query 与它的钉住

**D1 开关**：`PositionalProcessReadout(..., pooled_readout_query: bool = False)` 与
`ProcessForecastCoReasoner(..., pooled_readout_query: bool = False)`；默认 `False`；非布尔值报错
（`type(value) is not bool`，与既有开关同风格）；**开启但未开启 `positional_process_readout` 直接报错**
（一个被静默忽略的开关是配置错误，不是控制组）。

**默认路径不产生任何新参数**（实测）：开启与关闭的 `state_dict` 键集合相同、逐张量相等、
参数总量相同（2,968,259）。**C 与 D 的参数与算力完全一致**：

| 臂 | 参数 | 前向 FLOPs | 前向+反向 FLOPs | 共享张量 | 新增张量 |
| --- | --- | --- | --- | --- | --- |
| A `process_pooled` | 2,799,779 | 12.8438e9 | 38.4176e9 | 115 | — |
| B `process_spacetime_only` | 2,819,267（+19,488 / +0.70%） | 12.9274e9（+0.65%） | 38.6651e9（+0.64%） | 与 A 共享 115 | +4 |
| C `process_spacetime_rwa` | 2,968,259（+168,480 / +6.02%） | 13.9046e9（+8.26%） | 41.5967e9（+8.27%） | 与 A 共享 115 | +16 |
| D `process_rwa_capacity_control` | **2,968,259（与 C 相同）** | **13.9046e9（与 C 相同）** | **41.5967e9（与 C 相同）** | 与 C 共享 **131** | 0 |

（FLOPs 约定见协议 `flop_convention`：`FlopCounterMode` + `enable_grad`，不使用参数 hook。
A 与 C 的数值与第一轮**逐位相同**，说明测量与设备无关。）

**与目标源 §4 第 8 条预期的差异（如实记录）**：目标源预期「池化把 q 投影从 N 个位置降到 1 个位置，
FLOPs 会略低」。**实测没有降低**：池化发生在 `query_norm(context)` 之后、attention 之前，
均值被**广播回 N 个位置**再进入 attention 自己的 q 投影，因此投影仍是 N 次。
这使 C−D 成为**参数与算力双重对齐**的对照（比预期更强），而不是「算力略低」的对照；
若要省算力需先池化再投影（线性层下 `q(mean) == mean(q)`，只差浮点求和顺序），
但那会让 C−D 同时差着算力，故不采用（决策 0014 记录了否决理由）。

**D4 钉住（测试，实跑）**：`tests/test_r7_process_readout_pooled_query.py` 7 项 ——
池化时 1089 个输出位置**逐位相同**（含单 token 扰动的响应逐位相同）、非池化时位置间有差异、
池化在**位置编码之前**（同一 12 token 换网格 3×4 → 2×6 输出逐位不变，非池化时改变）、
只接受 `bool`、缺 `positional_process_readout` 即报错、开启不新增参数、经模型自身
`process_conditioning` 的同一断言。CUDA 实形状（dim 192 / 33×33 token / 16 process token）复测：
池化 1088/1088 行逐位相同、响应 spread **恰好 0.0**；非池化 spread/max = 0.59。

**训练后位置探针**（每个 seed 用各自选中的 checkpoint，真实 val 窗口）：

| seed | A | B | C | D |
| --- | --- | --- | --- | --- |
| 41 | rank 2，spread 0 | rank 2，spread 0 | rank 3，spread **6.80e-7**（spread/max **0.676**） | rank 3，spread **0.0**（0.0） |
| 42 | rank 2，spread 0 | rank 2，spread 0 | rank 3，spread **8.87e-7**（**0.844**） | rank 3，spread **0.0**（0.0） |
| 43 | rank 2，spread 0 | rank 2，spread 0 | rank 3，spread **7.27e-7**（**0.746**） | rank 3，spread **0.0**（0.0） |

即：**训练之后** D 的读取仍然逐位不随位置变化（spread 恰好 0），C 的读取显著随位置变化
（响应差异占其量级的 68–84%）。A/B 没有逐位置读取（rank 2），spread 按构造为 0。

---

## 4. D2：关闭时与改动之前逐位相同（实跑）

`tests/test_r7_switched_path_equivalence.py`（对照组是 `93d89aa` 的冻结实现）：

```
.venv/bin/python -m pytest tests/test_r7_switched_path_equivalence.py -q   ->  4 passed
.venv/bin/python tests/test_r7_switched_path_equivalence.py                ->  digest_count 23
                                                                              bitwise_identical true
```

本轮新增开关后重跑，23 个 digest（三族 forward / rollout / 自适应 / streamed BPTT 的
**全参数梯度**）与**开工前那次重跑**的输出逐位相同（`same_as_recon: true`），
也就是说「默认关」不是近似默认，而是同一实现。反证测试（扰动冻结副本的一个常量必须使 digest 不同）
同批通过。

---

## 5. D3：四臂配对是实测，不是假定

每个 seed 在训练前都做两项实测，并把结果写进产物（`protocol.json → arm_pairing` 的规则说明、
`seed_result.json → arm_pairing` 的数字、`training/*/training_report.json → shared_initial_state` 的计数）：

1. **六个臂对逐一比较同名张量**：三个 seed 全部 `all_shared_pairs_identical: true`。
   张量数：A 115、B 119（+4，`spacetime.*`）、C 131（+16，`spacetime.*` + `process_reader.*`）、
   D **131（与 C 完全相同）**；A∩B∩C∩D 的 115 个张量逐位相同。
2. **锚臂 `state_dict` 真的加载进其余三臂**（用训练器自己的转移规则，`load_state_dict(..., strict=False)`），
   加载后**逐张量比对**：三臂均 `applied 115 / ignored 0`、`post_load_all_applied_bitwise_equal: true`。
3. **运行时一致性**：每个 arm 的训练报告里的 `applied/ignored` 计数必须等于上面预测的计数，
   否则运行**报错退出**（`run_seed` 里的显式检查）。12 个 run 的产物计数见 §6 训练表。

---

## 6. D5：一轮有界实验

**协议**：每个 seed 一份 `protocol.json`，写盘后**回读校验**，且发生在该 seed 的任何
`optimizer.step()` 之前（时序证据：`protocol.json` 的 mtime 早于 `training/` 目录）。
12 个 run 的训练记录各自携带 `protocol_sha256`，合并时要求 **12 个值全同**：

| 项 | 值 |
| --- | --- |
| `protocol_sha256` | `5367fc59…462a7`（**12/12 相同**，`merged_result.json → run_protocol_sha256`） |
| 种子 | 41 / 42 / 43（开工前声明，全部报告） |
| 更新预算 | 400 updates/臂，**12 个 run 全部跑满 400、无早停** |
| 共享控制 | AdamW lr 2e-4 / wd 1e-4、warmup 80 → cosine 到 0.1×、clip 1.0、bs 2、验证每 100 步、patience 4、val 6h、按 val 归一化 MSE 选 checkpoint |
| 划分 | 只读 `train.jsonl` 与 `val.jsonl`；`test_read: false`；`refused_test_manifest` 对 `test.jsonl` 直接报错 |
| 案例数（每 lead，四臂一致） | 6h 22、12h 21、24h 19、48h 15、72h 11（val 可用窗口上限） |
| `scientific_claim` | `false`（协议与结果都带） |

**训练表**（`elapsed_seconds` 求和 = 1891.0 s = **0.5253 GPU-h**）：

| seed | 臂 | 选中 update | 秒/update | 墙钟 s | 选中 val MSE（归一化，6h） | 峰值显存 | applied/ignored |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 41 | A | 400 | 0.3770 | 150.8 | 0.217967 | 258 MiB | 0/0 |
| 41 | B | 400 | 0.3917 | 156.7 | 0.209510 | 258 MiB | 115/0 |
| 41 | C | 400 | 0.4344 | 173.8 | 0.208738 | 260 MiB | 115/0 |
| 41 | D | 400 | 0.4004 | 160.2 | 0.208716 | 260 MiB | 115/0 |
| 42 | A | 400 | 0.3771 | 150.9 | 0.216358 | 258 MiB | 0/0 |
| 42 | B | 400 | 0.3791 | 151.7 | 0.208611 | 258 MiB | 115/0 |
| 42 | C | 400 | 0.3918 | 156.7 | 0.209765 | 260 MiB | 115/0 |
| 42 | D | 400 | 0.3898 | 155.9 | 0.209729 | 260 MiB | 115/0 |
| 43 | A | 400 | 0.3899 | 156.0 | 0.216678 | 258 MiB | 0/0 |
| 43 | B | 400 | 0.3952 | 158.1 | 0.210170 | 258 MiB | 115/0 |
| 43 | C | 400 | 0.4049 | 162.0 | 0.210761 | 260 MiB | 115/0 |
| 43 | D | 400 | 0.3960 | 158.4 | 0.210734 | 260 MiB | 115/0 |

**关于一次被中断的启动（如实记录）**：本轮第一次启动（12:15，GPUs 0/1 各一个 seed 并发）
在约 12:17 被**会话恢复**杀掉，两个 run 未完成。其产物目录已删除，**本文件不使用它的任何数字**。
第二次启动（12:20 起，`setsid nohup` 脱离会话，单卡 GPU 1 顺序跑三个 seed）产出本文件引用的全部产物。
第二次改为单卡顺序的原因：一枚与本项目无关的作业在 12:16 占用了 GPU 0。
两次启动的差别只在设备与并发方式，协议、种子、数据、预算均未变。

---

## 7. D6：配对结果（#60 比较器，`depth=0`，逐 seed 同号）

17 个变量 × 5 个时效 = **85 格**。一格只有在**每个 seed 的 delta 同号**时才算
improved/worsened，否则 unresolved；delta < 0 表示 focus 臂 RMSE 更低。
**没有新增任何阈值**，也没有放宽比较器。（`case_identity: exact`——四臂同一案例集，
案例集 digest 与初始化数逐 seed 相同，比较器 fail-closed 通过。）

| 对（focus − baseline） | improved | worsened | unresolved | 同号格数 | 同号格相对量级中位数 | 最大 |
| --- | --- | --- | --- | --- | --- | --- |
| **B − A**（只加时空输入） | **36** | **3** | 46 | 39 | 6.5e-2 | 2.9e-1 |
| C − A（两个都加） | 26 | 3 | 56 | 29 | 6.4e-2 | 3.3e-1 |
| **C − B**（在时空输入之上加位置化读取） | 8 | **16** | 61 | 24 | 1.9e-2 | 1.2e-1 |
| D − A（容量控制 vs 参照） | 26 | 3 | 56 | 29 | 6.4e-2 | 3.3e-1 |
| **D − B**（在时空输入之上加池化读取） | 7 | **17** | 61 | 24 | 1.8e-2 | 1.2e-1 |
| **C − D**（位置依赖本身） | 8 | **13** | **64** | 21 | **4.1e-4** | **2.3e-3** |

「相对量级」= |三个 seed 的平均 delta| ÷ 基线臂该格 seed 均值 RMSE（单位无关）；
它只是把量级摆在计数旁边（比较器的判据里没有量级），**不构成新判据**。

**逐 lead 计数**：

| 对 | 6h | 12h | 24h | 48h | 72h |
| --- | --- | --- | --- | --- | --- |
| B − A | 8/0/9 | 7/0/10 | 9/1/7 | 6/0/11 | 6/2/9 |
| C − B | 1/3/13 | 2/3/12 | 2/4/11 | 1/4/12 | 2/2/13 |
| D − B | 1/3/13 | 2/3/12 | 1/4/12 | 1/5/11 | 2/2/13 |
| C − D | 2/5/10 | 1/4/12 | 1/3/13 | 1/0/16 | 3/1/13 |

（improved/worsened/unresolved。`beats_baseline_everywhere` 对**每一对**都是 false——
不存在「全面更好」的臂。）

**逐变量（C − D，每变量 5 个时效）**：t2m 4 好 0 差 1 unresolved（唯一明显偏向 C 的变量族）；
q850 1 好；u250 / v250 / v10 各 1 好 1 差；mslp 0 好 **2 差**；u500 0 好 **3 差**；
z500 0 好 2 差；q500 / t500 / u10 / v500 / v850 各 0 好 1 差；t850 / u850 / z250 / z850 全 unresolved。

**C − D 的同号格全部很小**（决定性的一点）：

| 格 | 判定 | 单位 | 平均 delta | D 的 RMSE | 相对量级 |
| --- | --- | --- | --- | --- | --- |
| 72h\|u250 | improved | m s⁻¹ | −0.0416 | 17.783 | 2.3e-3 |
| 72h\|t500 | worsened | K | +0.00943 | 4.5658 | 2.1e-3 |
| 24h\|t2m | improved | K | −0.00610 | 4.2529 | 1.4e-3 |
| 6h\|z500 | worsened | m² s⁻² | +0.131 | 151.43 | 8.7e-4 |
| 12h\|t2m | improved | K | −0.00316 | 3.5457 | 8.9e-4 |
| （其余 16 格更小） | | | | | ≤ 8.9e-4 |
| 12h\|q500 | worsened | kg kg⁻¹ | +2.7e-8 | 4.536e-4 | 5.9e-5 |
| 6h\|q850 | improved | kg kg⁻¹ | −3.3e-8 | 6.014e-4 | 5.5e-5 |

**对照**：B − A 的同号格量级是 0.5%–29%（例如 6h t2m 从 3.684 降到 2.639 K，−28%），
即**时空输入的效果比 C−D 的任何一格大两个数量级以上**。

**头部变量的四臂 seed 均值（val，物理单位，同一评估器、同一案例集）**：

| lead | variable | 单位 | A | B | C | D | 8 桶气候态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6h | t2m | K | 3.6839 | **2.6386** | 2.6124 | 2.6129 | 2.7248 |
| 12h | t2m | K | 4.8234 | **3.5524** | 3.5425 | 3.5457 | 2.7242 |
| 24h | t2m | K | 5.6183 | **4.2449** | 4.2468 | 4.2529 | 2.7457 |
| 48h | t2m | K | 6.7335 | **4.7705** | **4.5450** | 4.5469 | 2.4458 |
| 72h | t2m | K | 6.9453 | 5.8846 | **5.3626** | 5.3685 | 2.4089 |
| 6h | mslp | Pa | 219.95 | 194.93 | 192.68 | **192.69** | 484.56 |
| 72h | mslp | Pa | 1099.45 | **1136.69** | 1112.31 | 1115.69 | 502.12 |
| 24h | z500 | m² s⁻² | 446.60 | **422.30** | 434.84 | 434.66 | 524.48 |
| 72h | z500 | m² s⁻² | 915.11 | **650.78** | 727.89 | 727.42 | 503.20 |

（`rmse_seed_mean.json` 有全部 17 变量 × 5 时效 × 4 臂；`rmse_table.csv` 是逐 seed 的明细。）

### 7.1 C − D 的判定：**位置依赖本身未获支持**

按本轮冻结的判据（逐 seed 同号 + #60 比较器，`depth=0`）：

- **13 格 worsened / 8 格 improved / 64 格 unresolved**——**反向占优**，不是「不显著的好」；
- 即便只看同号的 21 格，其相对量级中位数 **0.04%**、最大 **0.23%**，与 B−A 的 0.5%–29% 不在一个量级；
- 唯一系统性偏向位置化臂的是 t2m（5 个时效中 4 个 improved，量级 0.02%–0.14%）。

**结论（按目标源 §3 的预设写法）**：**位置依赖本身未获支持。**
在参数量、模块、键与实测算力都对齐的 C−D 对照上，把「读取是否要看位置」这件事改成
**不看位置**（D）并没有让 val 变差；D 在更多格上略好，且两者差值都很小。
本轮**没有**换变量、换网格、换时效去凑一个赢面，也**没有**放宽任何判据。

**同时必须连着读的两条**：①C 与 D 都比 B **更差**（C−B 8/16、D−B 7/17），
即「加一条把 process 状态读给 solver 的通路」在时空输入已经打开的前提下是**负向**的，
无论它是否看位置；②A 参与的所有配对仍然混着容量（+19,488 与 +168,480 参数），
所以 B−A / C−A / D−A 的赢面**不能**归因到机制，只能说明「打开时空输入的这一族更贵且更好」。

---

## 8. 重跑稳定性

第一轮与本轮在**共有 seed（41/42）**上各有 A 与 C 两臂，协议常量相同（400 updates、
同 lr/warmup/cosine/clip/bs/验证节奏/选择规则），代码除新增的默认关开关外相同
（`model_code_sha256` 因此不同）。**同一 seed 两次独立运行的选中 val MSE**：

| seed | 臂 | 第一轮 | 第二轮 | 相对差 |
| --- | --- | --- | --- | --- |
| 41 | A | 0.21796682883392682 | 0.2179673354734074 | 2.3e-6 |
| 41 | C | 0.20873777907003055 | 0.20873750136657196 | 1.3e-6 |
| 42 | A | 0.21635833247141403 | 0.21635829318653454 | 1.8e-7 |
| 42 | C | 0.20976575667207892 | 0.20976530557329004 | 2.1e-6 |

- **不是逐位一致**（checkpoint `sha256` 也不同），与项目已记录的 GPU 浮点非确定性一致；
  差异在 1e-7–2e-6 相对量级。
- **排序稳定**：两轮在 seed 41 与 seed 42 上都是 **C < A**（C 的 val MSE 更低）。
- **墙钟不可比**：第一轮两个进程并行占用两张卡，第二轮单卡顺序跑；
  A 的秒/update 从 0.4120/0.4059 降到 0.3770/0.3771（−8.5%），这是并发方式的差别，
  不是模型变快。
- **没有 seed 43 可对照**（第一轮只有 41/42）；上述只是两个共有 seed 的观察，不是显著性。

---

## 9. 局限（如实）

1. 一轮 400 updates、3 seed 的有界运行，**不是**收敛或 SOTA 对照；12 个 run 全部停在
   update 400（仍在下降），端点不是收敛点。
2. **三个 seed 给的是符号一致性，不是显著性**；本轮**不新增**任何显著性阈值或检验。
3. **只有 C−D 是容量对齐的**；A 与 B/C/D 之间差着参数与算力（见 §3），
   B−A / C−A / D−A 的胜负**不能**归因到机制。
4. C−D 只消掉**读取处**的位置依赖：位置信息仍可经 context 本身进入模型（A 也有同样通道），
   所以它测的是「读取是否要看位置」，不是「模型是否用到位置」。
5. 时空输入在 C 与 D 上都开着，两个机制未在本轮重新分离；交互项（第五臂）明确不做。
6. 单一冬季（2016-01/02）、单一年、单一区域的两月段；跨季节/跨年/跨区域全部 UNVERIFIED。
7. 只读 val；test 仍封存，且**从未**用于任何方法选择（包括本轮）。
8. val 气候态是 store 的 8 桶 `(month,hour)` 训练均值，不是强季节气候态；
   6h t2m 低于气候态这一点**不能**用来推翻 M2 在 test 上的「气候态压倒 t2m」结论
   （不同划分、不同预算、本轮 test 未读）。
9. C−D 的同号格量级（0.005%–0.23%）**大于**同 seed 重跑的非确定性（1e-7–2e-6），
   即在三轮 seed 上可复现；但这仍然是「小」，不足以支持任何收益声明。
10. 池化控制臂**不省算力**（仍投影 N 次）；「先池化再投影」的等价实现未做（决策 0014 记录理由）。
11. 位置化读取仍是**集合语义**（对 process token 顺序不变），"positional" ≠ "ordered"；
    位置基是固定正弦，不是可学的每位置表。
12. 本轮第一次启动被会话恢复中断（§6），其产物已删除且未计入任何数字；第二次启动的顺序/并发
    方式与第一次不同，但协议、种子、数据、预算未变。

---

## 10. 未做的事

- **未读 test**（`test_read: false` 写入协议与结果；`evaluate_local` 的 `split` 断言全为 `val`）。
- **未做**第五臂（`spacetime_inputs=False + readout=True` 的交互项）、未做 RW-B（门控 + 局部状态）、
  未做 M3/M4/M5、未扩基线家族、未新增任何数据下载。
- **未用 test 做方法选择**、未放宽任何已冻结判据、**未新增阈值**（包括对 C−D 的量级没有引入门槛）。
- **未复用**第一轮的 checkpoint（`model_code_sha256` 与 `protocol_sha256` 都不同，按设计拒绝）。
- **未改动** `data/raw|interim|processed`、任何归档快照、任何入口点位置。
- **未合并 main、未 force push、未租 GPU、未写 issue**。
- **一处待用户清理的本机遗留**：见 §12。

---

## 11. 下一项的第一个具体动作

**给「加一条把 process 状态读给 solver 的通路」这件事本身找一个可判定的方向**：
本轮 C−B 与 D−B 都是负向（8/16 与 7/17），说明在当前 400-update 预算下，
这条读取通路在时空输入已开时拖累了 val。下一个具体动作是**先做诊断而不是加臂**：
用已选的 12 个 checkpoint 跑 `diagnose_r7_gain.py`（仓库既有的修正诊断入口），
在 val 上比较 B 与 C/D 的 **correction 幅度与符号**，判断负向是否来自「读取把 context 拉偏」
（若 correction 幅度显著放大且与误差反相关，则下一个假设是 readout 需要归一化/门控）。
该诊断必须仍只用 val、只读已归档 checkpoint，不新增数据、不新增阈值；
若诊断本身需要新判据，先停止并向用户报告。

---

## 12. 本机遗留文件（需要用户处置，与第一轮相同）

`tests/fixtures/r7_equivalence_recipe.py`（第一轮遗留、**未跟踪、未提交**）仍在本机，
使 `python tools/check_conventions.py` 报 **R-044 一条阻断命中**。本轮**未触碰**该文件
（`guard_protected_paths` 拒绝一切 `tests/` 下的删除/移动）。
**CI 不受影响**（跑提交树，该文件未跟踪）；但本机因此**不能**宣称「conventions 干净」。
处置方式同第一轮 §12（用户在 ZCode 之外的终端 `rm` 一次即可）。

---

## 13. 复现

```bash
# 1) 逐位等价（23 digest）+ 反证 + 池化钉住
.venv/bin/python tests/test_r7_switched_path_equivalence.py
.venv/bin/python -m pytest tests/test_r7_switched_path_equivalence.py \
    tests/test_r7_process_readout_pooled_query.py tests/test_r7_process_readout_positional.py -q

# 2) 一轮有界对照（单卡顺序；协议在每个 seed 的第一步前冻结）
DEV=1 bash scripts/run_71_72_round_two_multiseed.sh
# 3) 合并 + 四对（六对）配对 + 四张表 + 训练后位置探针
.venv/bin/python scripts/study_r7_71_72_round_two.py --mode finalize \
    --manifests outputs/r7_m2_segment/store/manifests --out outputs/r7_71_72_round_two
```

## 14. 提交与 CI

见本节末表（`git log`/CI run 在提交后补记）。核对方式（匿名只读 API，AGENTS.md 记载本项目
API 写 401、读可用）：

```bash
curl -s "https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs?head_sha=<full sha>" \
  | python -c "import json,sys;[print(r['id'],r['name'],r['status'],r['conclusion']) for r in json.load(sys.stdin)['workflow_runs']]"
```

**17 条实验 workflow 在普通 push 上显示 skipped 是设计行为**（commit-message 标签门控），
**不算失败**；本轮的实验证据来自本机单张 RTX 3090，不走 workflow。

| SHA | 内容 | `ci.yml` run |
| --- | --- | --- |
| _（提交后补记）_ | | |
