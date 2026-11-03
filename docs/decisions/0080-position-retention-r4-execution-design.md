# 0080 — R4（位置保留 H-B）执行设计：两臂**都从零起点**（路线②），并先重建 65×65 的 typed-evidence 侧车；`scale` 作为唯一新增常数字段

- **日期**：2026-11-03
- **状态**：accepted
- **范围**：`docs/decisions/0078-position-retention-reeval-admission.md`（准入）与
  `docs/decisions/0079-position-query-scale-is-not-a-learning-amplitude-knob.md`（问法收窄）
  的执行设计；本轮**尚未开跑**，本 ADR 只定「怎么跑」
- **依据**：0078/0079、`model/typed_evidence_r7.py:128-150`、`data/preprocess/r7_typed_evidence_scale.py`、
  `scripts/r7_s3_typed_spatial_parent.py:78-120`、`outputs/r7_s3_fy_multiyear_20261023_attempt01/` 的只读实测

## Context

**① 0079 留下的那个选择，两个候选不是同一道题。** 0078 §Context ⑥ 实测：typed-evidence 通路在主线
（R1/R2 的父与两臂）上**不存在**——三处 `typed_evidence_mode` 全为 `None`，当前迁移父 155 键、
**0** 个 `typed_evidence.*`。因此 R4 必须为 9 个张量定初始化。0079 §Decision 4 指出两条路线
给出**不同的起点**：

| 路线 | 起点 `‖position_query‖` | 本轮实际问的是 |
| --- | --- | --- |
| ① 迁移 H-B 归档的 9 张量 | **0.007393 / 0.012458 / 0.008398**（H-B 训练后） | 「从旧轮的训练终点**续跑**，位置加权能否改变读数」 |
| ② `isolated_stream()` 下新初始化 | **恰为 0.0**（实测四个种子全为 0） | 「从 pooled 起点**重跑**，位置加权能否改变读数」 |

**② 实测（本 ADR 新增）：路线①会把**另一个 store** 的标准化统计带进来——但**只带 9 个张量，不带 buffers**。**
`TypedEvidenceRouter` 的五个标准化/统计 buffer 全部 `persistent=False`
（`model/typed_evidence_r7.py:138-150`：`denorm_mean`、`denorm_std`、`field_scale`、`field_mean`、
`field_std`、`field_active`），实测 H-B 的 `update_0002400.pt` 里 `typed_evidence.*` **恰是 9 个键**
（`conv.weight/bias`、`norm.weight/bias`、`position_query`、`projections.{0..3}.weight`），
**没有任何 `denorm`/`field` 键**。⇒ 迁移这 9 个张量在**结构上不会**把旧 store 的统计搬过来；
统计随 spec 的 payload 走，与 checkpoint 无关。

**③ 但 payload 本身**是 store 特定的，而 H-B 的 payload 属于 129×129 宽实例。**
`field_maps` 用 `denorm_*` 把模型输入反归一化回物理单位，再用 `field_*` 标准化
（`model/typed_evidence_r7.py:165-175`）。实测两个 store 的 `normalization_*` 差异
（H-B 宽实例 vs FY 65×65）：

| 通道 | 相对差（mean） | 相对差（std） |
| --- | ---: | ---: |
| t2m | 3.36e-3 | 2.51e-2 |
| u10 | 2.93e-1 | 7.68e-2 |
| v10 | 2.18e+0 | 4.01e-3 |
| q850 | 1.10e-1 | 6.81e-2 |
| u850 | 1.92e-1 | 3.39e-2 |
| v250 | 4.47e+0 | 1.53e-1 |
| **最大** | **4.47e+0** | **1.53e-1** |

把 H-B 的 payload 用在 65×65 上（在真实帧 1000 上做「先用自己的向量归一化、再用 payload 的
`denorm_*` 反归一化」的错配解码），物理量误差为：

| 通道 | 真实均值 | 错配解码均值 | 相对误差 |
| --- | ---: | ---: | ---: |
| t2m | 289.312 | 290.355 | 3.61e-3 |
| u10 | 0.158733 | 0.136798 | **1.38e-1** |
| v10 | −1.48964 | −1.25592 | **1.57e-1** |
| u850 | −1.59319 | −1.11208 | **3.02e-1** |
| v850 | −2.42548 | −1.66077 | **3.15e-1** |
| **v500** | **−0.0354971** | **+0.598519** | **符号翻转** |
| z500 | 56559.6 | 56789 | 4.06e-3 |

（对照：payload 与 store 一致时往返误差 `7.1e-15`，即 float64 精度。）
⇒ **路线①的「迁移」若连 payload 一起沿用，会喂给标准化器一份别的 store 的统计**，
风速类场偏差 14–31%，`v500` 的符号都会翻。这不是风格问题，是**会改变被计算的四个诊断场**的问题。

**④ 实测：65×65 的 typed-evidence 侧车**不存在**，但生成它的工具存在且是只读先行的，
且本 ADR 已**实跑过一次只读 preflight**。**
`outputs/r7_s1_seasons_2017/typed_evidence/` 是**s1 四季 store**的侧车
（`data_identity 894b8d1b…`，`pair_count 344`），不是 FY 65×65 的。
`data/preprocess/r7_typed_evidence_scale.py` 提供 `typed_evidence_preflight(store, manifest)`
（只读、返回 `preflight_identity`）与 `publish_typed_evidence_scale(..., preflight_identity)`
（显式写入、`_destination` 拒绝已存在路径、`fresh_outputs` 自带 `BUILD_COMPLETE`）。

本 ADR 已在 FY 65×65 的 train 清单上**实跑只读 preflight**（零写入、零 GPU）：

| 项 | 实测 |
| --- | --- |
| `preflight_identity` | `6e6e224a2714de3f8d01873e4461392a9cfc990be7fd949377c8a8723fc7c6e3` |
| `data_identity` | `81307008b572e3ff032d5251e788c582f339b0e3dbbc3fea4226d44c6a335af0`（**等于** FY 65×65 train 清单的身份，即 §Decision 2(d) 的断言已先验证一次） |
| `pair_count` / `count` | 32,143 / 271,608,350 |
| `degenerate` / `active_mask` | 四场**全非退化** / 四场**全启用** |
| `runtime_scale` | `[1.0, 1.0, 1.0, 1.0]` |
| `field_mean` | `2.6578375294968702e-06, −1.724736622419589e-05, −0.0008499384953675593, −290.00488305603335` |
| `field_std` | `0.00017022464475439984, 0.00024940630354299573, 0.006233944735638324, 59.11283675089141` |

**与 H-B 的 `field_*` 相比，这四场统计量的差异同样大**（这正是「不得沿用 H-B payload」的第二个、
比 §③ 更直接的理由——`field_*` 是**直接标准化**这四个场的系数）：

| field | H-B mean | FY65 mean | 相对差 | H-B std | FY65 std | 相对差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `divergence_850` | 9.2258e-07 | 2.6578e-06 | **1.881e+0** | 1.6149e-04 | 1.7022e-04 | 5.41e-2 |
| `vorticity_850` | −1.3071e-05 | −1.7247e-05 | 3.195e-01 | 2.3749e-04 | 2.4941e-04 | 5.02e-2 |
| `temp_advection_850` | −7.5863e-04 | −8.4994e-04 | 1.204e-01 | 5.7556e-03 | 6.2339e-03 | 8.31e-2 |
| `static_stability_850_500` | −345.613 | −290.005 | 1.609e-01 | 63.973 | 59.113 | 7.60e-2 |

⇒ 若沿用 H-B 的 `field_*`，标准化的就是 `(x − hb_mean)/hb_std` 而不是 `(x − fy_mean)/fy_std`，
**发散场的均值差 1.88 倍**。这条与 §③ 的 `denorm_*` 错配**互相独立**，任一条都足以否掉「沿用 payload」。
⇒ 路线②的完整前置＝**在 FY 65×65 的 train 清单上跑 preflight → 审阅 → 显式发布到全新排他目录**。
preflight 已跑过一次（上表），但**发布仍需显式授权**，本 ADR 不代替那个授权。
这与 `real-data-acquisition` 技能要求的「只读 preflight 后显式 `--write`」是同一纪律。

**⑤ `scale` 必须落到 `model/**`（已由 0078 ⑦ 实测确认），且必须与两臂共享。**
`_build_typed_evidence` 对 `typed_evidence` 的字段集做**精确相等**断言
（`model/process_forecast_r7.py:152-157`），故新字段无法只经 spec 偷偷传入——要么加字段、
要么加参数，两条都改 `model/**`。按 0079 §Decision 3，`scale` 是**前向敏感度旋钮**，
不承担修正优化器病态之责；两臂**都带同一个 `scale`**，唯一因子仍是聚合方式。

**⑥ 影响面实测（0078 ⑧ 已记，此处只落执行含义）**：`model_code_sha256` 一旦移动，
`load_checkpoint`（`training/r7_experiment.py:128-129`）会拒绝所有钉在旧修订上的父。
R2 的 `a38be16c…` 闭包含 `model/typed_evidence_r7.py`，**而 R2 的 seed+reading 已于 2026-11-03 全部结束**
⇒ 该改动现在可以落地。两个已归档的 wiring 轮不必重跑（读数已登记、`code.zip` 可重放），
但其父钉若要重冻就走 0075 的重绑路径。

## Decision

1. **采用路线②：两臂都从零起点、新初始化。** 理由是可核查的三条：
   - 0079 已把本轮的问法收窄为「给定前向敏感度下，位置加权是否改变长 lead 读数」，
     而**零起点正好是该问题的自然起点**：两臂在 step 0 逐位相同，差异只能来自 `scale` 之后的聚合；
   - 路线①的起点是 H-B 训练了 2400 步的随机游走产物（`|q|` 均值 0.694× 随机游走期望），
     把它带进来会让本轮变成「续跑一个已知弱操纵的终点」，与 0079 的诊断直接冲突；
   - 路线①**没有**结构上的额外信息优势：9 个张量只是 148,992 参数的初始化值，
     而两臂共享同一初始化的可断言性（`pathway_init_sha256` 相等）在路线②里更强。
2. **但 payload 必须为 65×65 **重新生成**，不得沿用 H-B 的那份。** 步骤（顺序固定、前一步不完成不进下一步）：
   - (a) 在 FY 65×65 的 train 清单上跑 `typed_evidence_preflight`（**只读**，零写入）；
   - (b) 审阅其 `preflight_identity`、`data_identity`、`degenerate`/`active_mask`、四场统计量；
   - (c) 取得显式授权后 `publish_typed_evidence_scale(...)` 到**全新排他**路径
     （候选 `outputs/r7_s3_typed_evidence_fy65_20261103_attempt01/`），留 `BUILD_COMPLETE`；
   - (d) 协议里钉 `typed_evidence_identity` 与 `data_identity`，并断言其
     `data_identity` **等于** FY 65×65 train 清单的 `81307008…`（不等即 `raise`）。
3. **`scale` 的承载方式：新增一个 `typed_evidence.position_scale` 数值字段**（**不是**可学参数、
   **不是** `nn.Parameter`），默认 `1.0` 以保持旧路径逐位不变；`scale != 1.0` 时才改变打分。
   实现位置：`TypedEvidenceRouter._position_attention` 的
   `scores = (value @ q) * self.position_scale / sqrt(dim)`。
   **两臂共用同一个冻结值**，写进 `protocol.json` 并由 brief 预声明。
4. **桥 (e) 按路线②重写（逐条可断言）**：
   - `|position_query| == 0.0` 在两臂的初始 state 上都成立（零起点）；
   - 两臂 `pathway_init_sha256` 相等（共享同一初始化）；
   - `scale` 逐字相同（两臂同值）；
   - `zero_query_is_pooled`：把 query 清零后 C 的聚合**逐位等于** B 的均值。
   失败只记 `identity/engineering unresolved`，**不进入** (a)–(d)。
5. **`scale` 的定值走 0079 §Decision 2 的实测映射**：以真实 val 批次量出的归一化熵落在
   `[0.90, 0.99]` 为准（`P=1089` 时对应 `score_std ∈ [0.374, 1.186]`）；熵 `< 0.5` 视为
   退化成 one-hot、该轮读数**不得**用于判定原假设。**跑前定值、不扫、不看读数后调。**
6. **父钉按 0075 重绑**：`model/**` 改动落地后复算 `model_code_sha256`，重跑已提交的迁移脚本，
   逐 seed 断言 155 张量的 `state_digest` 不变。**不放宽任何身份门。**
7. **预声明出口 (a)–(d) 沿用 0078 §Decision 3 的原文**，不改：唯一因子仍是聚合方式，
   5 lead × 3 seed 的判定形式不变；只把「`scale` 修正学习幅度」的叙述换成 0079 的
   「给定前向敏感度」。

## Consequences

**变容易的：**
- 路线②让桥 (e) 变成**四条可逐位断言的恒等式**（零起点、共享初始化、同 `scale`、清零即 pooled），
  比路线①的「迁移张量的 SHA 等于归档」更容易独立复核，也更贴合 0079 收窄后的问法。
- 把「payload 必须按 store 重算」这条从隐含变成显式步骤，等于提前挡住一个会静默污染四个诊断场的坑。

**变难 / 代价（如实列出）：**
- **本轮读数与 H-B 的读数不可直接对照**：H-B 从「带 typed 的注册父」出发、本轮从零起点，
  两者不是同一实验。任何「比 H-B 更好/更差」的说法都是错的。这条必须进 brief。
- **多了一整个前置阶段**：65×65 侧车的 preflight + 审阅 + 发布，是 0078 起草时**没有**计入的成本
  （当时只算了 6 臂 ≈ 76.4 GPU-h 的训练与评分）。该阶段是 CPU、零 GPU，但要走
  `real-data-acquisition` 的只读审阅 + 显式写入纪律，**不能一键跑完**。
- **`scale` 仍是人为常数**：0079 已把它的身份从「学习幅度修正」降为「前向敏感度」，
  故本轮**不能**回答「优化器自然会学到多大 query」。这是 0079 已接受的代价，此处只承接。
- **`model/**` 改动会移动 `model_code_sha256`**：两个已归档 wiring 轮的父钉若要重冻需走 0075，
  属维护成本而非读数变更。
- **不产生「位置保留是否必要」的一般结论**：只在选定的 `scale` 与零起点下回答。

**备选方案与否决理由：**
- **备选 A：路线①（迁移 H-B 的 9 张量）。** 否决：起点是 H-B 2400 步随机游走的终点，
  与 0079 的诊断冲突；且它对「本轮问的是从 pooled 起点重跑」没有帮助——反而把两个不同的实验
  混成一个。**注意**：否决的是**迁移张量**，不是否决「重算 payload」——后者在两条路线下都必须做。
- **备选 B：沿用 H-B 的 payload（不重算侧车）。** 否决：实测错配解码在风速类场偏 14–31%、
  `v500` 符号翻转（本 ADR §Context ③）。这会改变被计算的四个诊断场，属静默污染。
- **备选 C：把 `scale` 做成可学参数。** 否决：与可学 query 完全冗余（只以乘积进入打分），
  引入不可辨识自由度，且破坏「唯一因子是聚合方式」。
- **备选 D：不改 `model/**`，把 `scale` 硬编码进打分。** 否决：那就是把实验常数写死进模型代码，
  协议无法钉它、复核者无法从产物复算，违反 R-006 的协议先行与可追溯要求。
