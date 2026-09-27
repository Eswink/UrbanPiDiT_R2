# #71 / #72 第一轮：时空输入贯通（M1）与位置化 process 读写（RW-A）

**状态：本轮的代码、测试与一轮有界实验均已完成；科学结论按下面的局限读，不得外推。**
`model/` 的字节改动使 `model_code_sha256` 变化，因此本轮**不复用**任何旧 checkpoint，
旧分数与本轮数字**不可相加**。

| 项 | 值 |
| --- | --- |
| 起点 SHA | `93d89aa`（开工时 `git rev-parse HEAD`） |
| 数据 | M2 双月段 `outputs/r7_m2_segment/store/manifests`：train 186 窗口 / val 22 / test 26（**本轮 test 未读**） |
| 协议 digest | `376492b2d580b63fdc812a1566dafbf94dbf2275a27eb629fec05d03aa7c0d71`（两个 seed 相同） |
| `model_code_sha256` | `746fa1660a30bdfa8a567e14e463b8945192af97e4f2d8d2129602d1c82dbbe6` |
| `data_identity` | 见 `outputs/r7_71_72_m1_rwa/seed*/protocol.json` 的 `data.data_identity` |
| 实测 GPU 训练 | **0.1846 GPU-h**（4 份 `training_report.json` 的 `elapsed_seconds` 求和 = 664.5 s） |
| 本轮预算 | ≤1.5 GPU-h（自设）/ 第二批授权 ≤24 GPU-h（已用 1.204 → 1.389） |
| 实验产物 | `outputs/r7_71_72_m1_rwa/`（协议、逐 seed 结果、配对比较、参数/FLOPs/RMSE/案例表、位置探针） |
| 逐位等价证据 | `tests/test_r7_switched_path_equivalence.py`（23 个 digest + 反证），起点 SHA 冻结实现 |
| 决策 | `docs/decisions/0013-v2-round-one-switched-pathway.md` |

---

## 1. 交付物逐条核对

| # | 要求 | 状态 | 证据 |
| --- | --- | --- | --- |
| D1 | init 时刻 / 年内·日内相位 / 地理坐标进入 forward；相位只由 init+lead 计算；缺字段显式失败 | **完成** | `model/spacetime_conditioning_r7.py`；`model/weather_forecaster_r7.py`；`data/r7_store.init_time_fields`；测试 `tests/test_r7_spacetime_inputs.py`（12 passed） |
| D2 | 白名单与固定/截断/流式/自适应/rollout 各路径读到同一组字段，并有测试钉住 | **完成** | `model/r7_halting.DECLARED_MODEL_INPUTS`；`tests/test_r7_input_path_consistency.py`（6 passed，7 条路径互锁） |
| D3 | RW-A：每输出位置一个 query 读 process token（N×M），替换 `process.mean` 广播；仍是加法式 | **完成** | `model/process_readout_r7.py`；`process_conditioning`；`solver_conditioning` 的 `[B,N,D]` 分支；`tests/test_r7_process_readout_positional.py`（10 passed） |
| D4 | 新通路默认关；关闭时与重构前**逐位相同**；不改 digest 比对逻辑、不复用旧 checkpoint | **完成** | `tests/test_r7_switched_path_equivalence.py`（23 digest 逐位相等 + 扰动反证 + 默认关断言）；`training/r7_experiment.py` 未改 |
| D5 | 因果探针：改未读字段→不变；改相位/lead→改变；单 token 扰动各位置响应不均匀 | **完成** | 初始化态探针见 `test_r7_spacetime_inputs.py` / `test_r7_process_readout_positional.py`；**训练后**探针见 `outputs/r7_71_72_m1_rwa/merged_result.json → positional_probe` |
| D6 | 一轮有界对照：同修订两臂、固定种子与 updates、协议第一步前冻结、只读 val、报告 GPU-h/参数/FLOPs、负面结果照写 | **完成** | `scripts/study_r7_71_72_spacetime_rwa.py` + `scripts/run_71_72_multiseed.sh`；产物见上表；结果见 §6 |
| D7 | 证据文档（本文件），含 `scientific_claim: false`、limitations、未做的事、下一项 | **完成** | 本文件 §6–§8 |
| D8 | 提交并核对 CI；skipped 的实验 workflow 是标签门控、不算失败 | **完成** | 见 §11 |

---

## 2. M1：已知时空条件如何进入主模型

**新增读取的字段恰为四个**（`model/r7_halting.DECLARED_MODEL_INPUTS` 单点声明）：

```
coarse_history, lead_time_hours, latitude, longitude, init_utc_hour, init_day_of_year
```

- **init 时刻的来源**：store 自带的 `time_ns`（窗口最后一个历史时次），由
  `data/r7_store.init_time_fields` 用**整数运算**分解为 `init_utc_hour` / `init_day_of_year`
  （`init_year` 同时导出供审计）。非整点时间戳**直接拒绝**，不做静默取整。
  两个生产者（训练读取器与 rollout 读取器）调用**同一个**函数，故不会各自定义"初始时刻"。
- **相位**：`annual = 2π·(doy-1 + (init_hour+lead)/24)/365.25`、`diurnal = 2π·(init_hour+lead)/24`，
  即**两个谐波项都用有效时间 `init + lead`**（因此 24h 的 lead 会把年相位推进一天，
  跨午夜/跨年由正弦自身周期性处理）。模块**不 import 任何时钟源**（AST 反证：
  `test_the_conditioning_module_reads_no_clock`）。
- **地理**：`latitude`/`longitude` 先按 patch 单元取平均（与编码器同样的 padding），
  再取 `sin/cos` 得到**每 token** 的位置特征；与相位特征拼接后过一个小 MLP（8→dim/2→dim），
  加到 token 流上。**没有**把经纬度压成域均值标量——反证测试把纬度轴**反转**
  （均值不变）并要求输出改变。
- **缺字段即失败**：`require_spacetime_fields` 对四个字段同时校验存在性、形状、有限性与范围；
  没有默认分支，也没有时钟兜底。反证测试逐个删除字段并要求报错（关掉开关时同一 batch 被接受，
  以此证明这是通路的性质而非 batch 的性质）。

## 3. RW-A：位置化 process 读写

- 旧路径（`model/process_forecast_r7.py:216` 的那一行）把 `process.mean(dim=1)` 投影后
  **广播**给所有位置；RW-A 让**每个输出位置**用自己的 query（该位置自己的 context token
  经 LayerNorm，加上固定的二维多尺度正弦位置编码）对 M 个 process token 做 cross-attention，
  得到 `[B,N,D]` 摘要；`solver_conditioning` 按秩分支：`[B,D]` 走原来的广播、`[B,N,D]` **直接相加**。
- **仍是加法式**：无门控、无逐位置状态、process 递推本身未改（`_reason` 原样保留）。
  测试用 `torch.equal(solver_conditioning(context, summary), context + summary)` 钉死加法语义。
- **N×M 代价**：注意力输入形状被测试直接捕获（`[B,N,D]` 查询 × `[B,M,D]` 键值），
  N=1089（33×33 token 网格）≠ M=16 时才通过。
- **集合语义（必须如实说）**：query 不含 key，因此 permute process token 后输出在浮点求和
  噪声内不变——RW-A 让摘要**依赖位置**，**不**引入 token 顺序。测试把这条性质钉住，
  防止后人把它读成"ordered"。
- **训练后探针**（`positional_probe`，两 seed）：
  | 臂 | 摘要秩 | 位置数 | 单 token 扰动响应 max | 跨位置 spread | spread / max |
  | --- | --- | --- | --- | --- | --- |
  | process_pooled | 2 | 1（广播） | 0.0 | **0.0（构造上恒为 0）** | 0.0 |
  | process_spacetime_rwa | 3 | 1089 | 8.76e-07 / 1.17e-06 | 5.92e-07 / 8.53e-07 | **0.676 / 0.731** |

  即：**训练之后**该读取仍是位置化的（响应在位置间变化约七成幅度），而对照臂的方差恒为 0。

## 4. D4：关闭时与改动前逐位相同——怎么证、证到了什么

`tests/test_r7_switched_path_equivalence.py` 用 `git archive 93d89aa model training`
把**改动前的实现**读入**同一个进程**（包名 `pre_change_model`/`pre_change_training`），
唯一改动是把 `from model.…` 重写为冻结包名，且该重写**可逆校验**
（`assert_import_rewrite_is_the_only_edit`：反向重写必须逐字节还原归档文本）。

同一份 recipe 对两侧各跑一遍，比较 **23 个 digest**：三个模型族的 forward
（forecast / tendency / draft_forecasts / final_correction / state / context_tokens）、
自由 rollout（含累积推理步数）、自适应推理（`force_full_depth`）、一次
streamed truncated BPTT（总损失、三项分损失、每步误差、**全部参数梯度**）。
digest 取形状+ dtype + **原始小端浮点字节**，任何 1 ULP 差异都会失败；不使用容差。

- 结果：`bitwise_identical: true`（23/23），实测 3.0 s。
- **反证**：把冻结副本的 `coarse_forecast.py` 中 `LeadTimeEmbedding` 的隐层宽度 32→64
  后，比较必须失败，且 `native.forecast` 与 `process.forecast` 都必须出现在差异清单里
  （vacuity 防护）。
- **反自我比较**：两侧都记录自己实际 import 的 `model`/`training` 路径，测试断言冻结侧在
  临时目录下、live 侧在仓库下；若某次运行误用了同一棵树，测试报"比较对象是自己"而不是通过。
- 未改 `model_code_digest` / `load_checkpoint` 的比对逻辑；本轮**未加载任何旧 checkpoint**
  （新代码下 `model_code_sha256` 已变，旧 checkpoint 会被拒绝——这是设计，不是绕过）。

## 5. D2：七条路径的输入集合互锁

`tests/test_r7_input_path_consistency.py` 用 forward 钩子记录**模型（及其 backbone）实际收到
的键集合**，逐条比对（不是与手写清单比，而是与 `forecast_inputs` 的输出比，避免平行清单漂移）：

| 路径 | 调用点 | 实测集合 |
| --- | --- | --- |
| 固定（native 训练） | `training/r7_local_runner.update_group` | 6 个声明字段 |
| 截断/流式（process） | `training/r7_streaming.backward_streamed_truncated` | 同上 |
| 验证 rollout | `training/r7_scheduled_runner.score_validation` | 同上 |
| 自适应推理 | `AdaptiveProcessForecaster.initial_state` | 同上 |
| 自由 rollout | `model/r7_rollout.autoregressive_rollout` | 同上 |
| 发布评估 | `training/r7_evaluate.evaluate_local`（含从 checkpoint 重建模型） | 同上 |
| 推理 profiler | `training/r7_inference_profile.profile_forward` | 同上 |

另外两处**钻穿**测试：

- `test_the_whitelist_is_the_declared_set_and_carries_no_target`：白名单 == `DECLARED_MODEL_INPUTS`
  且与 9 个泄漏类键（`atmos_target`/`atmos_baseline`/`rollout_targets`/`process_targets`/
  `sample_id`/`init_time`/`valid_times`/`init_year`/`grid_spacing_deg`）**无交集**；
  缺 `coarse_history` 仍抛 `KeyError`（不静默）。
- `test_the_declared_inputs_are_exactly_what_the_model_reads`：用一个**记录访问**的 dict 包住
  白名单结果，实测（开关开）读取集合 == 声明集合，**缺失访问集合 == {atmos_baseline}**
  （唯一被刻意剥离的可选覆盖字段）。开关关时读取集合退化为
  `{coarse_history, lead_time_hours}` 且是声明集合的真子集。任何"声明了却没人读"或
  "读了却被白名单剥掉"的情况都会在这里显形。

## 6. D6：一轮有界对照实验

**协议**（第一次 `optimizer.step()` 之前写盘并回读校验，digest `376492b2…`，两个 seed 相同）：
两臂同 revision、同数据、同种子集合、同 400 updates、同 lr/warmup/cosine/clip/bs=2/验证节奏
（每 100 步，patience 4，val 6h）、同 checkpoint 选择规则；`test_read: false`；
`scoring: val only`。

| 臂 | 开关 | 参数 | 前向 FLOPs | 前向+反向 FLOPs |
| --- | --- | --- | --- | --- |
| `process_pooled` | 全关 | 2,799,779 | 12.8438e9 | 38.4176e9 |
| `process_spacetime_rwa` | 全开 | 2,968,259（**+168,480 / +6.02%**） | 13.9046e9（**+8.26%**） | 41.5967e9（**+8.27%**） |

**臂配对（实测，不是假定）**：115 个共享张量**逐位相同**，新增 16 个张量；训练时
`shared_initial_state` applied=115 / ignored=0。即两臂**只差这条通路**；
但参数与算力**不对齐**（见上表），故本轮不能把差异归因于机制而非容量。

**训练**（4 个 run 全部跑满 400 updates，无早停）：

| seed | 臂 | 选中 update | 秒/update | 墙钟 s | 选中 val MSE（归一化，6h） | 峰值显存 |
| --- | --- | --- | --- | --- | --- | --- |
| 41 | pooled | 400 | 0.4120 | 164.8 | 0.217967 | 258 MiB |
| 41 | spacetime_rwa | 400 | 0.4225 | 169.0 | **0.208738** | 260 MiB |
| 42 | pooled | 400 | 0.4059 | 162.4 | 0.216358 | 258 MiB |
| 42 | spacetime_rwa | 400 | 0.4178 | 167.1 | **0.209766** | 260 MiB |

读数：新臂在**两个 seed** 上 val（6h）MSE 都更低（−4.2% / −3.0%），**同向**；
墙钟仅 +2.5%（+8.3% 的前向 FLOPs 被数据读取与验证打分稀释）。

**配对结果**（#60 比较器，val，17 变量 × 5 时效 = 85 格，逐 seed 配对；
一格只有在**每个 seed 的 delta 同号**时才算 improved/worsened，否则 unresolved）：

| lead | improved | worsened | unresolved |
| --- | --- | --- | --- |
| 6h | 9 | 3 | 5 |
| 12h | 6 | 1 | 10 |
| 24h | 9 | 1 | 7 |
| 48h | 10 | 2 | 5 |
| 72h | 8 | 4 | 5 |
| **合计** | **42** | **11** | **32** |

（块级计数：improved 64 / worsened 21 / unresolved 32 of 85；53 格符号一致，其中 42 好 11 差。
`beats_baseline_everywhere: false`——不是"全面更好"。）

**逐变量（每变量 5 个时效）**：q850 / t850 / u10 5/5 全好且无 unresolved；
t2m 4/5 好（72h unresolved）；u850 4/5 好；v850 4/5 好 1 差；mslp 2/5 好、3 unresolved；
z500 与 t500 全部 unresolved（5/5）；v250/v500/z850/v10 各 0–1 好、2 差。

**头部变量种子均值（val，物理单位，同一评估器同一案例集）**：

| lead | variable | unit | pooled | spacetime_rwa | climatology(8 桶) |
| --- | --- | --- | --- | --- | --- |
| 6h | t2m | K | 3.7024 | **2.6310** | 2.7248 |
| 6h | mslp | Pa | 219.38 | **189.36** | 484.56 |
| 6h | v850 | m s⁻¹ | 3.2664 | 3.2343 | 5.3144 |
| 12h | t2m | K | 4.8153 | **3.6160** | 2.7242 |
| 12h | mslp | Pa | 337.71 | 321.66 | 488.85 |
| 24h | t2m | K | 5.6507 | **4.4818** | 2.7457 |
| 48h | t2m | K | 6.5855 | **4.6948** | 2.4458 |
| 72h | t2m | K | 6.5649 | 5.2966 | 2.4089 |

**这条数字必须连着限制读**：①是 **val**（M2 的"climatology 压倒 t2m"结论是在 **test** 上；
本轮的 6h t2m 之所以低于气候态（2.631 < 2.725）**不能**用来推翻它——不同划分、不同更新数、
本轮 test 未读）；②两臂参数/算力不对齐，赢面不能归因到机制；③两个 seed 只给一致性，不是显著性；
④唯一的"改进"判据是 #60 比较器的同号配对，**没有新增任何阈值**。

**位置探针（训练后）**见 §3 表：开臂 spread/max ≈ 0.68–0.73，关臂恒为 0。

## 7. 局限（如实）

1. 一轮 400 updates、2 seed 的有界运行，**不是**收敛或 SOTA 对照；"选中 update = 400"
   意味着两个 seed 都还在下降，端点不是收敛点。
2. **两个开关一起开**：时空输入与 RW-A 的**分别贡献 unresolved**（本轮设计无法分离）。
3. 两臂**参数/算力不对齐**（+6.0% 参数、+8.3% 前向 FLOPs），因此"更好"不能等同于"机制有效"。
4. 单一冬季、单一年、单一区域的两月段；**跨季节/跨年/跨区域全部 UNVERIFIED**。
5. 只读 val；test 仍封存。绝对数字不可与 M2 的 test 数字并列（不同划分、不同预算）。
6. val 气候态是 store 的 8 桶 `(month,hour)` 训练均值，不是强季节气候态。
7. RW-A 的读取是**集合语义**（对 token 顺序不变），"positional" ≠ "ordered"。
8. 位置基是**固定**正弦编码，不是可学的每位置表（换取任意网格可用）。
9. 等价性测试依赖仓库历史：没有 `git archive` 可用的历史时会**失败而不是跳过**。
10. `init_year` 随样本携带但**不被读取**（年相位是周期量），该事实由探针钉住，
    但它确实是一个"样本里有、模型不读"的字段——与 `grid_spacing_deg`/`sample_id` 同类，
    是审计用的样本元数据。

## 8. 未做的事

- **未读 test**（本轮设计中 test 全程封存，`test_read: false` 写入协议与结果）。
- **未做** RW-B 的门控与局部状态、未做 M3/M4/M5、未做新的数据下载、未扩基线家族。
- **未用 test 做方法选择**、未放宽任何已冻结判据、未新增阈值。
- **未复用旧 checkpoint**（新代码下其 digest 不匹配，按设计拒绝）。
- **未提交**：截至写文档时改动仍在工作区；提交与 CI 见 §11。
- **一处未解决的本地卫生问题（需要用户处置）**：本轮早期在
  `tests/fixtures/r7_equivalence_recipe.py` 放了一份"食谱文件"，随后发现 R-044
  （`tests/**` 下只允许 `test_*.py`）会阻断它，于是把食谱并入了
  `tests/test_r7_switched_path_equivalence.py`。**该遗留文件无法由本人删除**：
  `tools/agent_hooks/guard_protected_paths.py` 拒绝一切 `tests/` 下的删除/移动
  （实测 deny："BLOCKED (AGENTS.md 硬约束): removal of test files"），而 AGENTS.md
  记载的逃生口（改 `.zcode/config.json`）在会话中途不生效。
  文件**未跟踪、未提交**，因此**不影响 CI**；但它使本机 `check_conventions` 报
  R-044 一条阻断命中、工作区不"干净"。处置方式见 §12。

## 9. 下一项的第一个具体动作

**把两个机制拆成可分离的三臂实验**：在同一 store、同一 400-update 协议下增加第三臂
`process_spacetime_only`（只开 `spacetime_inputs`），与现有 `process_pooled` /
`process_spacetime_rwa` 一起跑 2 seed，用同一 #60 比较器做**两两配对**，
从而把"时空输入"与"位置化读写"的贡献分开；协议里的 `limitations` 相应去掉第 2 条。
（预算：第三臂 2 seed ≈ 0.06 GPU-h，仍在 ≤24 GPU-h 授权内。）

## 10. 复现

```bash
# 1) 冻结实现的逐位等价（23 digest）+ 反证；也可直接当脚本读两侧 digest
.venv/bin/python tests/test_r7_switched_path_equivalence.py
.venv/bin/python -m pytest tests/test_r7_switched_path_equivalence.py tests/test_r7_spacetime_inputs.py \
    tests/test_r7_input_path_consistency.py tests/test_r7_process_readout_positional.py -q

# 2) 一轮有界对照（一卡一进程；协议在每个 seed 的第一步前冻结）
bash scripts/run_71_72_multiseed.sh
# 3) 合并 + 配对 + 表 + 训练后位置探针
.venv/bin/python scripts/study_r7_71_72_spacetime_rwa.py --mode finalize \
    --manifests outputs/r7_m2_segment/store/manifests --out outputs/r7_71_72_m1_rwa
```

## 11. 提交与 CI

| SHA | 内容 | `ci.yml` run（R7 CPU CI / pytest） |
| --- | --- | --- |
| `2586477` | #71/#72 第一轮实现（model/data/training）+ 4 个测试文件 + 两个既有输入集合断言改写 + 实验脚本 + 本文件 + 决策 0013 + R-009 基线 | `36347857577` **completed / success** |
| 见下条提交 | 本节补齐 + `.gitignore` 回填 `.zcode/agents/` | 该提交自身的 run（文档/配置改动） |

核对方式（匿名只读 API，AGENTS.md 记载本项目 API 写 401、读可用）：

```bash
curl -s "https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs?head_sha=<full sha>" \
  | python -c "import json,sys;[print(r['id'],r['name'],r['status'],r['conclusion']) for r in json.load(sys.stdin)['workflow_runs']]"
```

**17 条实验 workflow 本次全部 skip**，且是**设计行为**：它们的触发是 `push` 到本分支，
但 job 级 `if: contains(github.event.head_commit.message, '<tag>')` 未命中——本轮提交的
message 里**没有**任何标签（`[cpu-study]`、`[seasonal-study]`、`[real-smoke]` 等）。
逐 run 核对过 job 级结论（`.../actions/runs/<id>/jobs` → `completed / skipped`），
不是失败，也不是排队。本轮的实验证据来自**本机 2×RTX 3090**，不走 workflow。

**CI 与本机的差异（如实）**：CI 是 Python 3.11 + CPU-only torch；本机是 3.12 + cu128。
逐位等价测试**不依赖**任何预存 digest（它在同一进程里跑冻结实现与当前实现），
所以在 CI 机器上同样是有效比较——run `36347857577` 的 success 就是这条在**另一台机器**上的
独立复现（含 `git archive` 路径与 `fetch-depth: 0`）。

## 12. 遗留文件的处置（需要用户）

`tests/fixtures/r7_equivalence_recipe.py`（未跟踪、非测试、已被合并进
`tests/test_r7_switched_path_equivalence.py`）需要删除。可选三种方式：

1. 在 **ZCode 之外的终端** `rm tests/fixtures/r7_equivalence_recipe.py`（推荐，一次即可）；
2. 从 `.zcode/config.json` 删掉 `guard_protected_paths` 的 PreToolUse 条目后**重启会话**，
   再由 agent 删除；
3. 明确授权 agent 用非 Bash 路径（`os.remove`）删除该文件——即接受一次被 hook 拒绝的
   删除，理由是非测试文件、未提交、且被 R-044 阻断。

在删除之前，本机 `python tools/check_conventions.py` 会报 **R-044 一条阻断命中**
（R-044 按文件名扫描文件系统，不区分该文件是否被 git 跟踪）；**CI 不受影响**，
因为它跑的是提交树。
