# #65 预诊断：过程—预报递归的定向诊断（S3-a）

**状态：预诊断完成（四项齐备），归因已判定。** 本文件只记录**只读诊断**结果，
不包含训练与选型。`test` 未读；未改任何判据；未训练任何模型。

| 项 | 值 |
| --- | --- |
| 起点 SHA | `f292b1a`（预诊断代码）；诊断运行于 `f292b1a` |
| 诊断产物 | `outputs/r7_65_prediagnostic/prediagnostic.json`（未进版本控制，R-035） |
| 诊断设备 | **CPU**，`gpu_used: false`，`training_executed: false` |
| 诊断耗时 | 8.9 s（4 个 validation case） |
| 只读对象 | `outputs/r7_b2_segment/store/cache.zarr` + 冻结 checkpoint |
| checkpoint | `seed41/…/process/update_0000700.pt`（B2 harness 自身按 validation 选出的那个），
sha256 `1d39da75…0627e7`，诊断前后各校验一次，未变 |
| data identity | `50517d945e…54cd6961`（= B2 train manifest 的 identity；val manifest 指向同一 store，已断言） |
| 复现 | `.venv/bin/python scripts/diagnose_r7_process_reasoning.py --store … --val-manifest … --checkpoint … --out …` |

诊断代码：`training/r7_process_diagnostic.py`（复用 #53 的 `correction_terms`，
不另写第二套几何）。测试：`tests/test_r7_process_prediagnostic.py`（18 例，含反证）。

---

## 1. Proxy 数值尺度审计（#65 第 3 条）

逐 proxy 记录真实 std、store 里烘进去的 std、floor 是否生效、标准化后范围：

| proxy | raw std | store std | floor 生效 | 标准化后 std | 标准化后 abs max | 可用？ |
| --- | --- | --- | --- | --- | --- | --- |
| mslp_gradient_strength | 2.809e-04 | 2.731e-04 | 否 | 1.0000 | 2.70 | 是 |
| divergence_850_rms | 6.419e-06 | 6.858e-06 | 否 | 1.0000 | 3.40 | 是 |
| vorticity_850_rms | 8.360e-06 | 8.042e-06 | 否 | 1.0000 | 3.24 | 是 |
| temperature_advection_850_mean | 3.599e-05 | 3.832e-05 | 否 | 1.0000 | 2.54 | 是 |
| **moisture_advection_850_mean** | **7.588e-09** | **1.000e-06** | **是** | **0.0076** | 0.024 | **否** |
| **moisture_convergence_850_mean** | **1.182e-08** | **1.000e-06** | **是** | **0.0118** | 0.040 | **否** |
| static_stability_850_500_mean | 2.706e+00 | 2.852e+00 | 否 | 1.0000 | 2.27 | 是 |
| vertical_wind_shear_850_500_mean | 3.895e+00 | 4.442e+00 | 否 | 1.0000 | 2.71 | 是 |

**结论（与已知线索一致，并修正了它的推论方向）**：`eps=1e-6` floor 在 8 个 proxy 里
**只对两个水汽通道生效**，把 7.6e-9 / 1.2e-8 的真实尺度按 1e-6 去除，标准化后 std 只剩
**0.0076 / 0.0118**（即信号被压低 ~132× 与 ~85×），全部样本都落在「近乎零」区间
（`fraction_below_usable_scale = 1.000`）。这**确实**使这两个 proxy 作为**监督标签**失效。

**但推论方向必须修正**：#65 原文怀疑这会让「process state 的水汽通道近乎常量」——
按代码路径**不成立**。`model/r7_halting.forecast_inputs` 的转发白名单只有
`coarse_history` 与 `lead_time_hours`；`process_targets` 根本**不进模型前向**
（`measurement_input_contract()` 已断言：`process_targets_is_forwarded: false`）。
所以这个缺陷能**削弱辅助损失**，不能污染前向状态、不能让任何输入通道变成常量。
把它写成「structure 的信息输入残缺」是**过度归因**，本文件不这样写。

（另：7.588e-09 与 9.1e-9、1.182e-08 与 1.7e-8 的差异来自本次统计的数组范围不同——
本次用 B2 36 天段全部 144 时次，早先记录用的是 D1 的 120 时次。两者同为「被 floor 压掉」的结论。）

---

## 2. 辅助梯度 vs 主梯度（#65 第 2 条）

train 模式，4 个真实 val case，按参数分组记录（共享 trunk / `process_queries` / `process_readout`）：

| 辅助权重 | 组 | ‖g_forecast‖ | ‖g_process‖ | cos(g_fc, g_proc) | 定义？ |
| --- | --- | --- | --- | --- | --- |
| **0.0**（B2/B3 实际值） | shared | 8.72e-01 | **0.0** | — | 否 |
| | process_queries | 3.44e-02 | **0.0** | — | 否 |
| | process_readout | **0.0** | **0.0** | — | 否 |
| 0.01 | shared | 8.72e-01 | 5.44e-02 | −0.053 | 是 |
| | process_queries | 3.44e-02 | 1.81e-02 | −0.073 | 是 |
| | process_readout | 0.0 | 3.71e-02 | — | 否（主梯度为 0） |
| 0.1 | shared | 8.72e-01 | 5.44e-01 | −0.053 | 是 |
| | process_queries | 3.44e-02 | 1.81e-01 | −0.073 | 是 |
| | process_readout | 0.0 | 3.71e-01 | — | 否（主梯度为 0） |

两条结构性事实（**代码路径决定，不是数值巧合**）：

1. **forecast loss 对 `process_readout` 的梯度恒为 0**。读出头的输出只被辅助损失打分，
   不在预报路径上。所以在 B2/B3 用的 `process_weight = 0.0` 下，这个头**完全收不到梯度**——
   不是「训练不足」，是**从未被训练**（`test_the_process_readout_is_off_the_forecast_path` 锁住）。
2. **权重 0 时 process 梯度恰好为 0，夹角无从定义**。诊断把它记为 `undefined` 而不是
   伪造一个 0 或 90°，也**不**把它当作因果证据（#65 明文要求）。

权重不为 0 时夹角**轻度为负**（shared −0.053、queries −0.073，4 个 case 里 2 正 2 负，
单 case 可达 −0.225）。按 #65 的要求，这**只作诊断记录，不作因果证据**：一次前向的
几何量不能推出「辅助任务损害了预报」。原始逐 case 数值在产物 JSON 里。

---

## 3. K0–K4 轨迹（#65 第 1 条）

同一 model/data 签名，4 个真实 val case，复用 #53 的 `correction_terms`
（误用/零向量/overshoot 的定义与原实现完全一致；`algebra_residual < 1e-9` 每次成立）：

| K | mse_before | mse_after | Δ | 修正范数 | err·upd cosine |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.13019 | 0.12752 | **−0.00267** | 0.04265 | −0.0621 |
| 2 | 0.12752 | 0.12675 | **−0.00077** | 0.02927 | −0.0128 |
| 3 | 0.12675 | 0.12723 | **+0.00048** | 0.02484 | −0.0041 |
| 4 | 0.12723 | 0.12880 | **+0.00156** | 0.02344 | +0.0274 |

相邻修正 cosine（衡量「多轮是否在重复同一个修正」）：

| 相邻对 | cos |
| --- | --- |
| K1→2 vs K0→1 | +0.8638 |
| K2→3 vs K1→2 | +0.9387 |
| K3→4 vs K2→3 | **+0.9815** |

K0→K4 整体：0.13019 → 0.12880（**−1.07%**）。

**读法**：净效果是**递减的**——K=1 贡献全部收益，K=2 只有其 1/3，K=3、K=4 开始**变差**。
同时相邻修正的 cosine 单调**趋于 1**（0.864 → 0.939 → 0.982），修正范数单调**下降**
（0.0427 → 0.0234）。即：多轮在做**几乎相同的、越来越小的修正**，且第 3、4 轮已经在
往错方向走。这与 #53 早先在 24 个 case 上的观察同向，**但不是同一批 case**
（本次 4 个，全部来自 2016-01-25/26 的 val 块），且逐 case 差异明显
（4 个 case 里 1 个 K4 仍在改善，其余 3 个 K3/K4 变差）。

**内存**：trace 只保留标量（每 K 的 per-channel 均值与逐样本标量），不保留完整计算图；
`recursion_trajectory` 在 `no_grad` 下运行并恢复原有 train/eval 状态。
streamed backward 的内存优势**未**被这次 trace 触及（它读的是 eval 前向的 `draft_forecasts`）。

---

## 4. 归因结论（#65 第 4 条）

#65 要求就两个负结果给出「最可能的原因」，并区分支持/反对证据。

### 4.1 决定性发现：B2/B3 的 `process` 臂不是过程结构的检验

这条不是三项候选之一，而是**先于三项**的事实，必须在归因里先排除：

- B2 与 B3 的 `process` 臂都以 `process_weight = 0.0` 运行（协议文件与每份
  `training_report.json` 的 `contract.process_weight` 均如此；`process_weight` 在
  `study_r7_b2_multiseed.py:76`、`study_r7_b3_scale.py:65` 就写成 `0.0`）。
- 在该权重下，辅助损失恒为 0 ⇒ `process_readout` 收不到任何梯度（第 2 节第 1 条）。
- 共享初始化按**参数名**复制，而两臂的参数名不同前缀：anchor 的
  `cell.*` / `latent` / `latent_to_context.*` 在 process 里叫
  `reasoning_cell.*` / `process_queries` / `process_to_context.*`，**72 个 applied、
  39 个 ignored**（B2 doc §6 已如实记录 ignored=39）。
- `tests/test_r7_process_arm_equivalence.py` 证明：两臂**唯一**实质差异是最后那个
  `latent_to_context.1` / `process_to_context.1` 投影（`weight`/`bias` 两个张量；
  `cell.*` 与 `latent` 虽被 skip，但两臂从同一 seed 流的同一点建出，**逐位相同**）。
  **把那一个投影强制对齐后，generic 与 process 给出逐位相同的 forecast、
  draft_forecasts、initial_forecast 与递归状态**（`torch.equal` 全真，4 个 seed 都成立）。

所以 B2 的 `process − generic` 在 85 格里 73 格 unresolved、B3 的 `process − generic`
跨 seed 翻号，**最可能的原因是这次比较没有把「过程结构」当作自变量**：
两臂在功能上是一个模型的两个初始化漂移，差异量级落在 seed 噪声内。实测支持：

| 证据 | 值 |
| --- | --- |
| 训练后 common 参数的相对差（median，3 seed） | 2.85% / 2.81% / 2.84% |
| process−generic 的 arm 间隔 / generic 自身 seed 离散 | 中位 **0.244**，均值 0.416 |
| arm 间隔 < seed 离散的格子 | **80 / 85（94%）** |
| 符号一致（3 seed 同号）的格子 | 12/85；**73/85 unresolved** |

**反对证据（如实记录）**：`process_to_context` 那个投影**确实**在训练前就不同，
所以两臂不是 idempotent 的同一模型，理论上可以有真实差异；B2 里也确有 12 格符号一致。
但 2.85% 的 median 漂移与「1/4 个 seed 离散」的量级说明：**即使存在真实效应，
本次 800 步 / 3 seed / 单季节的设计也无法把它与初始化噪声分开**。

### 4.2 三项候选的裁决

| 候选 | 裁决 | 支持证据 | 反对证据 |
| --- | --- | --- | --- |
| **A. proxy 数值失效** | **部分是**：两个水汽 proxy 作为**监督标签**确实失效 | floor 生效，标准化后 std 0.0076 / 0.0118，100% 样本低于可用尺度 | ①proxy **不是模型输入**，不能污染前向状态；②`process_weight=0` 时该损失**根本没参与**训练，所以它对 B2/B3 负结果**没有贡献**。它只对**未来**「打开辅助权重」的实验构成真风险 |
| **B. 算力未对齐** | **是（对 B3）；对 B2 不适用** | B3 的 800 步未随容量从 2.8M→18M 调整（B3 doc 自陈）；四族 forward FLOPs 差 1.46×，B2 的 5 臂也不等算力 | B2 五臂同 800 步同 batch，且**都**输给零参数 climatology ⇒ 单纯加算力不能解释「输给 0 参数」。B3 有 6/6 臂单调衰减、无早停，所以「18M 不稳定」这个借口不成立，但「容量 vs 步数」**至今未决** |
| **C. 数据太窄** | **是（对 climatology 的对比结论）** | 单一年 1 月、(month,hour) 桶仅 **4** 个 ⇒ clim 几乎就是「当月该小时均值」，在 t2m 占优**是本数据范围的产物**；B2 自陈种子噪声从 6h ~3% 涨到 72h 20–23%（个别格 65%） | 数据窄解释**不了** generic vs process 的 73/85 unresolved（同数据同预算同 seed 的对照）；它解释的是「为什么零参数基线这么难打」 |

### 4.3 一句话归因

> **B2/B3 的「process 不如 generic」不是关于过程结构的证据**：那次比较的自变量被
> `process_weight=0.0` + 名字不匹配的共享初始化抵消掉了，两臂功能上是同一模型的两个
> 初始化漂移（对齐一个投影即逐位相同），arm 间隔在 94% 的格子里小于 seed 离散。
> 真正的、可执行的结论是：**过程监督从未被打开过**，而这正好是 #65 C1 要测的东西。
> 水汽 proxy 的 floor 缺陷是**真实且未修**的，但它在 `weight=0` 下对已发生的负结果
> **没有贡献**；它约束的是 C1 里 `aux weight>0` 的臂——那两个通道的标签在修复前不可用。

**证据不足处如实标注**：无法区分「对齐算力后 process 是否更好」，
因为**该实验尚未做过**（B2 没打开辅助权重，B3 没对齐算力）。本文件不编造该结论。

---

## 5. 对 #65 后续（C1/C2/C3）的直接约束

1. C1 必须**真的打开辅助权重**（0 / 0.01 / 0.1），并**同时报告** readout 是否收到梯度——
   否则又是一次「没测到自变量」的运行。
2. 因为两个水汽 proxy 的标签在 floor 下不可用，C1 的 `weight>0` 臂必须说明：
   辅助损失里有 2/8 个通道是**近乎常量**的目标。要么承认这一点，要么先把归一化修好
   （修 store = 新 store、新 identity，成本另计）。**本轮不修归一化**（见决策 D-1）。
3. 四张表（参数量 / forward+bwd FLOPs / wall time / 例数）必须与精度表同时给出——
   这次 `process_fb_off` 的 forward FLOPs 就比 `process_fb_on` 低 11%。
4. C3 必须区分「从 K=4 checkpoint 评 K=1」与「独立训练 K=1」，两者不是同一个问题。

---

## 6. 局限（是否仍适用）

| 局限 | 是否仍适用 |
| --- | --- |
| 单一年 1 月、4 个 (month,hour) 桶 | **仍适用**；climatology 占优依旧不可外推 |
| val/test 是 2016 段内工程再划分 | **仍适用**；本文件全程 `test_read: false` |
| 800 步非收敛 | **仍适用**；本文的 K 轨迹是 checkpoint 的轨迹，不是收敛后的轨迹 |
| 3 seed 不是显著性检验 | **仍适用**；本文用「arm 间隔 vs seed 离散」作描述性比较 |
| 跨季节结论 UNVERIFIED | **仍适用** |

**本文新增的局限**：K 轨迹只用了 **4 个 val case**（#65 明确禁止反复优化那 8/24 个
已看过的 case，故只取前 4 个、只做一次性记录，未据此调任何参数）；梯度夹角是
**单批**几何量，只作诊断。
