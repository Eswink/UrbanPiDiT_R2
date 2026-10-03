# R7 主模型 V2 外部参考台账（External Reference Ledger）

- **访问日期**：原条目为2026-09-29；2026-10-03一手源码补核见§12、§13。
- **取数通道**：委派 `web-researcher` 子智能体（R-049）；GitHub 数值取自 `api.github.com` 的
  HEAD 端点，论文取自 `arxiv.org`。**检索摘要不作为证据**：本文件区分「已核对」与「未核对」，
  未固定的 commit 一律写 `TO BE PINNED BEFORE IMPLEMENTATION`，**不伪造 SHA**。
- **用途**：为 #72（空间过程读写 / 局部求解状态）与 #73（过程监督）的机制选择提供来源与边界。
  本文件**不**声称任何外部结果可搬到本项目的天气任务上。

## 0. 判定摘要

1. 「小规模潜变量递归精化 + 空间寻址的过程态用于区域天气预报」这一**组合**未见发表；
   但每个组件都有强先在性（递归精化来自 TRM/HRM，逐位置读取来自 Perceiver IO，
   局部迭代更新来自 RAFT 一脉）。
2. 与我们的架构假设**最接近的公开论证**是 2026-08-21 的「Read, Write, Relax」
   （arXiv:2608.21677）：它明确主张全局潜态与局部处理**必须混合**，并以机制语言指出
   「latent-token attention 起空间低通滤波的作用」。它属于 PDE 仿真，不是天气。
3. **「全局池化广播是瓶颈」这一命题，在天气/空间预测领域没有找到以该命题为主结果的受控负结果
   论文。** 因此它目前的证据强度是「机制性主张 + 架构设计惯例」，不是文献结论。
   这也意味着本项目的对照（RW-A/RW-B vs 池化广播）有机会成为该命题在气象域的**首个受控证据**。

## 1. TinyRecursiveModels (TRM)

| 字段 | 内容 |
| --- | --- |
| Repository | https://github.com/SamsungSAILMontreal/TinyRecursiveModels |
| Paper | 《Less is More: Recursive Reasoning with Tiny Networks》 https://arxiv.org/abs/2510.04871 |
| Access date | 2026-09-29 |
| Commit/tag | `c01103738605ba39d1430519b1ee0c62f4c707f8`（仓库 HEAD，2026-04-01；已核对） |
| License | MIT |
| Exact source file | `models/recursive_reasoning/trm.py`（另有 `trm_hier6.py`、`trm_singlez.py` 变体） |
| Concept borrowed | 共享网络的潜变量递归精化：以「潜变量 z + 当前答案 y」交替更新，外层 K 次改进；深度监督与展开训练的技巧 |
| What we implement ourselves | 我们用 P（过程摘要，`[B,M,D]`）+ Z（逐位置求解状态，`[B,N,D]`）+ Y（预报场）三分工，并把 P 的读取做成**位置化**（TRM 无空间寻址）；气象诊断的语义与时刻定义完全自定 |
| What we deliberately do NOT copy | 不声称「原版天气 TRM」；不搬其谜题答案状态与训练数据；不把它的超参当作本任务的配方；**它的 y 是棋盘级离散答案，没有空间寻址读取**——它本身就是我们怀疑的瓶颈形态，因此只作机制参照，不作结构模板 |
| Why relevant to UrbanPiDiT | R7 的递推家族（`GenericRecursiveCell`、draft 反馈、deep supervision）与之同源；V2 要说明我们的 P/Z 分工是**任务针对性设计**，而非 TRM 的直接套用 |

仓库已 **archived（只读）**，作者声明停止维护。这意味着只在**设计阶段**引用，不引入依赖。

## 2. HRM（Hierarchical Reasoning Model）

| 字段 | 内容 |
| --- | --- |
| Repository | https://github.com/sapientinc/HRM |
| Paper | https://arxiv.org/abs/2506.21734 |
| Access date | 2026-09-29 |
| Commit/tag | `ac15626f8db096a63c775b84c9dc868776a6feda`（HEAD，2026-03-31；已核对） |
| License | Apache-2.0 |
| Exact source file | `models/`（README 未点名具体文件——**未核对**） |
| Concept borrowed | 「慢规划态 / 快工作态」双时间尺度分工的**动机**，对应我们把 P（跨区域过程）与 Z（逐位置求解）分开 |
| What we implement ourselves | 两个状态都是空间化的气象量；halting 部分我们不采用（我们在该规模已实测自适应无益，见 `docs/R7_66_GATE_AUDIT.md`） |
| What we deliberately do NOT copy | 不复现其层级架构、1D 序列设定、27M 规模与训练配方；不引入 ACT 式 halting |
| Why relevant to UrbanPiDiT | 为「为什么需要两个状态而不是一个」提供外部先例，避免把 P/Z 分工写成无出处的自创 |

## 3. Perceiver / Perceiver IO

| 字段 | 内容 |
| --- | --- |
| Repository | https://github.com/google-deepmind/deepmind-research/tree/master/perceiver |
| Paper | Perceiver https://arxiv.org/abs/2103.03206 ；Perceiver IO https://arxiv.org/abs/2107.14795 |
| Access date | 2026-09-29 |
| Commit/tag | `9176a9f23ced8e3d6024718d757be8d67cfb6927`（2026-10-03固定clone、HEAD及原文补核，详见§13；原API失败保留） |
| License | Apache-2.0（仓库 LICENSE，已核对）；README 另注明数据与参数为 CC-BY-4.0 |
| Exact source file | `perceiver/perceiver.py`（输出 query 读取潜变量）、`perceiver/position_encoding.py`（Fourier 位置编码）、`perceiver/io_processors.py`（位置与输入/输出 query 绑定） |
| Concept borrowed | **输出 query 从少量潜变量读出密集输出**——这就是 RW-A 已实现的东西（`model/process_readout_r7.py`），也是 RW-B 读取端的形状依据；位置编码用固定基而不是可学每位置表 |
| What we implement ourselves | 用 PyTorch 写小规模 cross-attention（已有 `SDPAttention`），按仓库命名与规模上限实现；位置基固定、非持久 buffer，不进 state_dict |
| What we deliberately do NOT copy | **不引入整个 Perceiver 栈、JAX/Haiku 框架或它的前向交错结构**；不复制其 latent array 规模与层数 |
| Why relevant to UrbanPiDiT | 这是「逐位置从全局潜态读」的**原生出处**；novelty 主张必须承认机制先在性，落在「气象过程态 + 小数据受控对比」上 |

## 4. RAFT

| 字段 | 内容 |
| --- | --- |
| Repository | https://github.com/princeton-vl/RAFT |
| Paper | https://arxiv.org/abs/2003.12039 （ECCV 2020, Teed & Deng） |
| Access date | 2026-09-29 |
| Commit/tag | `2888e15a51fa41140771d3f498ed8023cff098d1`（master HEAD，2025-08-24；已核对） |
| License | BSD-3-Clause |
| Exact source file | `core/update.py`、`core/raft.py`、`train.py`（2026-10-03实际固定clone并读对应原文段落，§13记录hash与边界；取代原二手确认） |
| Concept borrowed | ① 局部门控递归更新单元（ConvGRU 式）的成熟配方；② 对多次精化结果逐步监督的加权序列损失 |
| What we implement ourselves | Z 的更新单元用可微 torch 张量、在 patch 网格上做 3×3 深度可分离卷积 + pointwise；门控是**逐位置标量**并断言梯度非零；序列监督复用我们已有的 deep supervision 权重（`training/r7_recursive_losses.py`） |
| What we deliberately do NOT copy | **不复制光流任务、4D correlation volume、warp 操作、mask/阈值策略**；不复制其迭代次数与学习率配方 |
| Why relevant to UrbanPiDiT | RW-B 的「局部 + 门控」正好是 RAFT 的 update block 思路；同时它是「迭代精化」范式的代表作，必须在 novelty 讨论里被点到 |

## 5. MetPy

| 字段 | 内容 |
| --- | --- |
| Repository | https://github.com/Unidata/MetPy |
| Paper | 无（官方文档 https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.html ） |
| Access date | 2026-09-29 |
| Commit/tag | `07df928b0d47fce73696116d6988d0722bfaa57e`（HEAD，2026-08-24；已核对）；PyPI 最新 1.7.1（2025-08-29） |
| License | BSD-3-Clause；遵循 semver，1.x 内向后兼容 |
| Exact source file | `src/metpy/calc/kinematics.py`（涡度、散度、平流等；另有热力学与探空指数模块） |
| Concept borrowed | 诊断公式的**单位与地理 metric 约定**（`dx/dy` 网格参数、纬度缩放），以及可作为过程监督目标的诊断清单 |
| What we implement ourselves | 前向里用**可微 torch 张量**实现同一诊断，并与 MetPy 做一致性测试（离线小数组 oracle）；#73 的 `draft_diagnostics` 必须能在推理时由模型自己的草稿算出 |
| What we deliberately do NOT copy | **不在 GPU 前向里经 CPU/pint 往返**；不引入 MetPy 作为运行期依赖；不因为 MetPy 有某个诊断就自动加进监督（诊断数上限见 #73） |
| Why relevant to UrbanPiDiT | 现有 8 个 proxy 已有实现（`data/preprocess/process_diagnostics.py`）且被 eps floor 压制 2 个；MetPy 是修复尺度与新增诊断时的**外部校验依据** |

**可行性未核对**：65×65、17 通道（含 13 个压层的子集）的配置下，哪些 MetPy 诊断可算
（例如 `cape_cin` 需要多压层探空），以及 `dx/dy` 常量是否满足其运动学函数的假设——
这两点需要读本仓数据后再判断，本文件不预设结论。

## 6. WeatherBench-X

| 字段 | 内容 |
| --- | --- |
| Repository | https://github.com/google-research/weatherbenchX |
| Paper | 无独立论文（WB2 评估代码的继任框架） |
| Access date | 2026-09-29 |
| Commit/tag | `c47c45fcc686209ba870ca3b1439fa970fe9eed0`（HEAD，2026-09-29 当日提交；已核对） |
| License | Apache-2.0（README 注明 not an officially supported Google product） |
| Exact source file | 未点名（框架层：data loaders / interpolations / metrics / aggregation 四类可插拔组件） |
| Concept borrowed | 指标与聚合的**定义口径**（跨变量如何聚合、插值如何声明）作为我们实现的对照 |
| What we implement ourselves | 继续用仓库已验证的实现（`training/r7_rollout_metrics.py`、`r7_acc.py`、`r7_climatology_skill.py`），它们的语义已被测试与决策 0010 的单位修正钉住 |
| What we deliberately do NOT copy | **不迁移整个评估框架**（#75 明文：现有已验证评价实现优先）；不引入 Apache Beam 依赖 |
| Why relevant to UrbanPiDiT | 我们的面积加权 RMSE、ACC、气候态 skill 的口径需要一个外部参照，避免自定义指标无法与文献对齐 |

**未核对**：README 层面未见「自定义/区域数据 loader」的现成示例，能否直接喂 65×65 单区域
需读其 docs；本文件不作结论。

## 7. WeatherBench 2

| 字段 | 内容 |
| --- | --- |
| Repository | https://github.com/google-research/weatherbench2 |
| Paper | 无（数据与评估框架） |
| Access date | 2026-09-29 |
| Commit/tag | `d2c6a1553a0c532332d4c2d3be285508c514bfc2`（HEAD，2026-09-29 当日提交；已核对） |
| License | Apache-2.0（© 2023 Google LLC） |
| Exact source file | 未点名（`weatherbench2/metrics.py` 的逐项清单**未抓取**） |
| Concept borrowed | 评估约定：ERA5 的 6 小时降采样习惯、基线表结构、按变量报告而不是跨变量平均 |
| What we implement ourselves | 全部指标实现；我们的数据是**非全球**的 65×65 区域、0.25° 原生，沿用我们自己的面积权重与 split 契约 |
| What we deliberately do NOT copy | 不使用其数据桶（全球网格设定不匹配我们的区域）；不做一阶守恒重网格 |
| Why relevant to UrbanPiDiT | 「跨变量不做单一聚合分数」这一惯例在仓库既有文档里已采用，WB2 是它的外部出处；README 自身也建议新工作转向 WeatherBench-X |

**未核对**：数据指南未讨论区域子域/自定义数据集；指标逐项清单未取到。

## 8. 新增：Read, Write, Relax（最接近的公开论证）

| 字段 | 内容 |
| --- | --- |
| Paper | 《Read, Write, Relax: Why Neural PDE Surrogates Need Both Global and Local Processing》 https://arxiv.org/abs/2608.21677 （2026-08-21） |
| Repository | 未定位（本文件不声称有官方代码） |
| Access date | 2026-09-29 |
| Commit/tag | N/A（无仓库） |
| License | N/A |
| Concept borrowed | **机制论证**：交替「潜变量注意力（全局）+ 消息传递/局部松弛（局部）」，并指出 latent-token attention 起**空间低通滤波**作用、纯局部传递缺乏全局可达性；两者互补（类比 multigrid 的谱段分工） |
| What we implement ourselves | 我们把它当作 RW-A/RW-B 分工的**外部论据**，而不是实现来源；判据仍是本仓自己的受控对照 |
| What we deliberately do NOT copy | 它是 PDE 网格仿真、工业求解器场景；**无天气、无递归预报、无过程监督**——不搬其数据集、指标与结论幅度 |
| Why relevant to UrbanPiDiT | 它为我们「池化广播不足、需要逐位置读取 + 局部门控更新」的假设提供了目前**最直接的机制性支持**；同时也是最近邻的 novelty 边界，必须在论文里被点到 |

**证据强度声明**：该论文的消融细节在正文，本次**未逐条核验**；引用时不得把它当作
「全局广播已被证伪」的受控负结果。

## 9. 新增：Aurora

| 字段 | 内容 |
| --- | --- |
| Paper | 《Aurora: A Foundation Model for the Earth System》 https://arxiv.org/abs/2405.13063 （v3 2024-11-21；Nature 2025） |
| Repository | 官方代码仓库**未定位**（本文件不声称存在） |
| Access date | 2026-09-29 |
| Commit/tag | N/A |
| License | 未知（无已核对的仓库） |
| Concept borrowed | 「逐位置 ↔ 小潜变量集合」的读写已被工程化验证的先例：其 3D Perceiver 编/解码器用少量潜变量（编码器 L=3 个潜压层）对逐位置信息做 cross-attention |
| What we implement ourselves | 我们的潜变量是**气象过程态**（不是垂直层），且模型是递归精化的、小规模、小数据 |
| What we deliberately do NOT copy | 1.3B 基础模型、多源预训练、垂直层潜变量、无递归推理循环、无自适应步数——**规模与科学问题都不同** |
| Why relevant to UrbanPiDiT | novelty 讨论必须承认「逐位置读潜态」的组件先在性，把主张收窄到「小数据/小规模下的受控机制对比」 |

## 10. 其他被考虑并保留的条目（仅作背景，不影响架构决定）

| 来源 | 一句话 | 为什么只作背景 |
| --- | --- | --- |
| neural-lam（MIT；paper https://arxiv.org/abs/2309.17370 ） | 区域有限域图神经网络气象预报，含数据管道与基线 | SHA 未固定；无递归潜变量机制 |
| ArchesWeather https://arxiv.org/html/2405.14527v1 | 少量 GPU-day 训练的 1.5° 低成本全球模型 | 纯架构裁剪，无递归/过程态 |
| Online model error correction with the IFS https://arxiv.org/html/2403.03702v1 | 对 NWP 做在线递归误差修正 | 修正对象是 NWP，不是独立学习的预报器；无潜过程态 |

**明确排除**：扩散类（masked latent diffusion / GenCast 等，机制不同）、纯统计后处理
（Rasp & Lerch 一脉）。

## 11. 检索到的空白（对我们有意义）

- **早退 / 自适应计算 × 天气**：未找到把 early-exit 或 adaptive computation 用于天气预报模型的一手论文。
  这与我们「该规模下 halting 无益」（#66/S4）的实测相容，也说明该交叉点近乎空白。
- **DEQ（deep equilibrium）× 天气**：未找到。
- **test-time compute × 天气**：未找到。检索曾把 arXiv:2510.14232 归为天气 test-time compute，
  **直接核实后确认它是 IOI 竞赛编程论文，与天气无关**——该线索已纠正并弃用。
  （记这条是因为它正说明「检索摘要不能当证据」。）

## 12. GraphCast：自回归训练的前瞻机制参考（2026-10-03）

本条经web-researcher定位后，主链**实际git clone、固定commit并读一手文件**，摘要不作证据。
引用只针对原版tag的机制，不把上游大规模结果或12步训练搬成本项目收益。

| 字段 | 内容 |
| --- | --- |
| Repository / source URL | https://github.com/google-deepmind/weathernext/tree/858301cde5de5c728f8172f782dafba1ea07ac2e （以此已clone地址为准，不依赖旧仓重定向推测） |
| Access date / tag / commit | 2026-10-03 / v0.1 / `858301cde5de5c728f8172f782dafba1ea07ac2e` |
| License | Apache-2.0，clone中的LICENSE SHA256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`；不下载权重或其它非代码材料 |
| Exact source | `graphcast/autoregressive.py`，SHA256 `07c51d7602679221f1d04f8aa0883e3a4f6a3abf8ee453af1baf20ecd6c5ce3c`；辅助graphcast.py `6aeb9cfa2490b0e2dbfa70f255594914ee1765d5e20c3e66ca912b80bc66e953`、losses.py `d4d4d3b0852ac140c13ca4786f3f4b5d90f9e5cc596db098ee7c7004295920a9` |
| Isolated local source | `outputs/reference_sources/graphcast/858301cde5de5c728f8172f782dafba1ea07ac2e/`；读取参考，不import或运行上游安装/训练 |
| Concept borrowed | autoregressive.py:261–286单步loss接各时刻target，`next_frame = xarray.merge([predictions, forcings])`后`_update_inputs`拼预测历史；:288–309可选hk.remat后hk.scan，平均per-time loss。读取文件未见detach/stop_gradient；重算不是物理梯度截断 |
| Actual implementation / port | 本仓自行写PyTorch精确两步history反馈与full-BPTT；保留L6+.5L12已预声明，**没有逐行移植**JAX源码、forcing或loss权重。通过解析梯度/poison/time/resume反证接受实现而非上游论文替代 |
| Deliberately not copied | 无JAX/Haiku依赖、GraphCast架构/全球mesh/训练数据/权重/上游脚本；不照抄12步课程、长unroll与大规模超参，不做外部baseline竞赛 |
| Relevance / boundary | 支持“预测历史继续作输入、标签只监督、物理展开可回传”的实施先例；不是本区域模型的forecast改善证据 |

取回回执 `outputs/r7_v2_remaining_acceptance_20261003/reference_source_receipt.json`固定文件hash与
未执行上游事实。检索员对Science正文403/上游新main结构等线索不纳入本条已确认结论；本条
只依据已固定原版源码，未核上游性能、发表版全文或当前main等价实现。

## 13. Perceiver / RAFT：直接query与迭代反馈的一手补核（2026-10-03）

检索员定位后，主链实际SSH clone到隔离目录、detached checkout、核HEAD/clean status、读原文及许可。
两轮HTTPS的GnuTLS -110失败原日志保留，未改安全或git配置；后续SSH成功不改写先前失败。
完整取回回执为 `outputs/r7_v2_remaining_acceptance_20261003/reference_query_source_receipt.json`。

### Perceiver

- URL：https://github.com/google-deepmind/deepmind-research/tree/9176a9f23ced8e3d6024718d757be8d67cfb6927/perceiver
  ；访问日2026-10-03；commit `9176a9f23ced8e3d6024718d757be8d67cfb6927`。
- 隔离源码：`outputs/reference_sources/perceiver/9176a9f23ced8e3d6024718d757be8d67cfb6927/`。
  code许可Apache-2.0，LICENSE SHA256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`；
  README:96–98另声明included data/parameters为CC-BY-4.0，不读取这些材料作实验输入。
- `perceiver/perceiver.py` SHA256 `b68017bf5fd20f0de77b2068a2fee2548eb0b6f24c938bf21fcf35f56aa7b6ef`。
  实际读:521–622：BasicDecoder的default-off `concat_preprocessed_input`将已处理输入与位置通道
  concat后形成query；:777–815：FlowDecoder的query直接返回inputs。
- **实际借鉴**仅为“query可携带局部当前内容且仍读小latent”的机制先例。本仓0034独立选择
  `LN(C_i + E(Y_k)_i) + pos_i`，不是上游concat公式，也不是“同形移植”；沿用已有draft encoder，
  Generic/Process同开关、无新增参数。固定P/C/pos的局部因果与梯度反证而非文献摘要接受该实现。
- **未搬**JAX/Haiku、latent规模、上游架构/数据/权重/训练脚本/成绩；没有upstream import或安装执行。
  位置编码/io_processors只固定文件hash，不把固定hash冒充逐行审计。

### RAFT

- URL：https://github.com/princeton-vl/RAFT/tree/2888e15a51fa41140771d3f498ed8023cff098d1
  ；访问日2026-10-03；commit `2888e15a51fa41140771d3f498ed8023cff098d1`。
- 隔离源码：`outputs/reference_sources/raft/2888e15a51fa41140771d3f498ed8023cff098d1/`；BSD-3-Clause，
  LICENSE SHA256 `399af7e243d625e8dea2b3f81fec676ea19bacf7d9ce8cfd4ce15530d8a66173`。
- 实际读 `core/raft.py`:121–142（SHA256 `e7280b82d0e224eff760ae8de5e8ea68923a7928c1e34920378b6f3a039a8633`）：
  每次先`coords1.detach()`，当前flow进入相关查询及update，随后加delta；
  `core/update.py`:89–136（SHA256 `7302f91ffc24c85cf739a25feb17d9d1e537f7344a2ddc331daf8ab9db6c3f74`）
  以flow/corr构motion features并进局部GRU；`train.py`:50–72
  （SHA256 `2e14d06c1a8507c2930ef6f566fbd6fb57c7a2a45e110ccf09a2cfbbf177625b`）以gamma加权中间预测。
- **实际借鉴**限局部迭代状态/门控和多draft监督的机制，本仓既有patch solver与归一化deep supervision
  自行实现；不移植光流correlation/warp/mask/gamma/lr/权重或上游脚本。特别是上游显式detach
  **不支持**本仓物理full-BPTT决定，不把其任务结果当本区域天气技巧。

本条为来源/机制核证，不是天气实验进展或SOTA依据。0034的default-off兼容与direct query判据另有
实际合成CPU证据；B保留旧母体query flag off，C若用修复则必须在独立协议里明确package contrast。

## 14. 维护规则

- 每条在**实施引用之前**必须把 `Commit/tag` 从 `TO BE PINNED BEFORE IMPLEMENTATION` 换成实测 SHA，
  并核对 License 是否允许我们的使用方式。
- 本文档的条目只保留**会改变架构决定**的材料；「文献很多」不是保留理由。
- 新增条目必须写清「我们自己实现什么」与「刻意不搬什么」两栏，否则不算完成登记。
