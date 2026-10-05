# R7 S2 / #79：有类型的局部诊断证据（三臂筛选，注册读数）

<!-- round-node: S2 -->

- 撰写日期：2026-10-05；状态：**executed, registered**；`scientific_claim: false`。
- 设计前置：`docs/R7_S2_79_TYPED_DIAGNOSTICS_DESIGN.md`（三臂、aux=0、判据与不变量，开跑前冻结）。
- 产物：`outputs/r7_79_typed_evidence_pilot/`（种子独立协议 + `paired_comparison.json`）、
  `outputs/r7_s1_seasons_2017/typed_evidence/`（train-only 标准化 sidecar，含 BUILD_COMPLETE）。

## 1. 注册读数（冻结决定文本逐字执行）

| 项 | 值 |
| --- | --- |
| 归因配对（primary） | `typed_routing − generic_fusion`，t2m，6h/12h |
| 判定 | **6h 与 12h 均 supported**（三 seed 同号为负） |
| seed 均值增量 | 6h **−0.05330 K**；12h **−0.05330 K**（逐 seed −0.0518/−0.0546/−0.0535） |
| 整体配对（secondary） | `typed_routing − process_v2`：6h/12h 均 **unresolved**（符号不一致），无 worsened |
| headline | `supported: the typed pair is supported on both leads and the overall pair does not worsen` |
| 全格报告 | 配对 1：13 improved / 8 worsened / 64 unresolved；配对 2：11/11/63；未并入判定 |

**能说的与不能说的**：能说的是——在**同参数量（3,097,571）、同 FLOPs（forward 16,776,992,640）、
同输入信息、同更新**的条件下，类型化路由 B→C 的三 seed 一致优于无类型融合；这正是设计里唯一
允许归因给**类型结构**的对照。不能说的是——C 相对 incumbent(A) 的两条主格都 unresolved，
所以本读数**不构成**"超过现状"或"超过气候态"的任何主张：三臂 t2m skill 均为负（≈−0.11，见 §4），
距 train-only climatology 仍有实质差距。

## 2. 机制与接线核验（执行时逐条留下）

1. **证据确实进入前向**（逐 seed，训练前）：off 臂首步状态与 `process_queries` 位级相同
   （差 0.0）；开启臂证据使首步状态移动 3.2–3.7（L∞）；扰动任一类型只移动其对应槽位
   （4×4 到达矩阵严格对角）；三臂 forecast 互不相同。三 seed 全部 `within_tolerance: true`。
2. **容量匹配**：B 与 C 参数量/前后向 FLOPs 逐位相等（实测记录在协议 arms 表）；A 少 148,800 参数
   （4 个类型投影 + 共享 patch 投影），harness 断言 A 的参数严格少于 B/C，否则拒绝。
3. **初始化可归因**：三臂同 seed 构建、trunk 位级相同（`shared_tensors_identical: true`），
   路径模块在 `isolated_stream` 下最后构建，开/关不移动既有权重流。
4. **关闭即旧实现**：`typing_evidence_mode=None` 与不传参的模型位级一致（CPU 反证测试
   `test_off_path_is_bitwise_the_previous_model`），首步状态返回同一 tensor。
5. **无未来信息**：铜化 `atmos_target`/`process_targets` 不改变 forecast（测试
   `test_future_labels_are_never_read_and_poisoning_them_is_a_no_op`）；证据只读 `X_t` 与模型自产
   `Y_0`。
6. **三路同一中央 step**：fixed / streamed / adaptive 三条路径都经
   `initial_process_state` 起步，streamed 与 fixed 首步输出位级一致（参数化测试）。
7. **身份**：诊断算子移入 `model/r7_process_tensor_diagnostics.py`（`model_code_sha256` 覆盖
   前向数值依赖，消除"模型前向调用 training/ 下算子"的身份洞）；sidecar 身份
   `6b46f450…`、数据身份 `894b8d1b…` 写进每个 seed 协议。

## 3. 与旧尝试的实质差异（信息增益）

- 与 #72/#73 不同：不重做 P/Z/Y 或 future/draft 辅助 loss；诊断**作为前向输入**，aux=0。
- 与"只加一个辅助头"不同：有逐类到达探针（可证伪"类型槽位真的收到各自证据"）与
  B 臂同容量控制；若 C≈B 就是"类型约束无增益"的干净负结论，本**不是**该情况。
- 与 #77/#78 的差异：那两轮改读取/解码的数值表达，本轮改**前向吃什么信息**；
  两者的主格都 unresolved/worsened，本轮是 S2 第一个 B→C 归因 supported 的机制。

## 4. 全变量全 lead 如实报告（描述性，不进判定）

- B→C 分 lead：6h 6 improved/0 worsened/11 unresolved；12h 6/1/10；24h 1/0/16；48h 0/2/15；
  72h 0/5/12。**短 lead 改善、长 lead 恶化的分界**与 #78 恰好相反（#78 是短 lead 恶化、
  长 lead 改善），两者都只作描述、不并入任何判定，也不得据此事后改注册主格。
- t2m 绝对量级（seed 均值，K）：6h A≈3.32 / B≈3.36 / C≈3.32；12h A≈3.78 / B≈3.86 / C≈3.80；
  climatology ≈3.18（6h）/3.19（12h）。即**三臂都未超过气候态**。

## 5. 验收清单完成情况

| # | 要求 | 状态 |
| --- | --- | --- |
| 1 | 无未来信息；合法输入改变会经指定路径改变 forecast | 完成（毒化反证 + 移动探针） |
| 2 | 算子/单位解析核验；常数/零风/mask/odd grid/BF16 | 完成（既有 26 项算子测试在移动后全绿，覆盖解析梯度/球面单位、2×2 与 2×4/4×2 小网格、零风常数场有限零梯度、常数偏移抵消、FP32/BF16 一致性、FP64 gradcheck、非法输入拒绝；本文件新增 fixture 复算 4 场统计） |
| 3 | 逐类到达探针；关闭路径与原 V2 位级一致；参数/FLOPs/梯度流 | 完成（4×4 对角矩阵、off 位级、梯度非零、B/C 相等） |
| 4 | fixed/streamed/active-subset 同一中央 step | 完成（streamed↔fixed 位级一致；adaptive 经同一 `initial_process_state`） |
| 5 | buffer/参数进 `model_code_sha256`；sidecar 进数据身份 | 完成（算子入 `model/`；sidecar 身份 `6b46f450…`） |
| 6 | 全变量全 lead；坏结果保留 | 完成（见 §4 与 `rmse_table.csv`） |

## 6. 成本

- 训练 0.498 GPU-h（1793.0 s）+ 评估/收尾：整轮 0.7129 GPU-h（2566.5 s），
  **soft overrun 0.0 s**（planned 9000 s / hard cap 18000 s 均未触）。
- S2 累计 **1.7386 / 6.0 GPU-h**（#77 0.6498 + #78 0.3759 + #79 0.7129）。

## 7. 已确认 / 推测 / 未做

**已确认**：B→C 类型结构归因 supported（三 seed、双 lead、同容量同信息同预算）；接线全部可核验；
off 路径位级不回归；身份链完整。

**推测（未证实）**：短 lead 改善/长 lead 恶化是否稳定、是否随训练量或数据量改变——单 dev store、
400 更新、单区域，不足以下结论。

**未做**：未读 test（`test_read: false`）；未做 R-B（差值尺度 loss）；未改任何冻结判据；
未因本轮正结果宣布任何"超过气候态/SOTA"；未触发 #79 的 GitHub 关闭（保留授权未继承）。

## 8. 回主线动作

按设计 §7 的承诺：正结果（B→C supported）**允许**把"typed evidence"带进下一确认实例的
设计冻结讨论；是否携带由 S3→S4 的数据扩围与主线决定，且必须带**新的独立协议**与预注册
（含每 seed/每年/四季的 skill 门与 u10/v10/mslp 非劣界），本读数不能替代该确认。
无论是否携带，本轮的负向侧事实（相对 incumbent 与 climatology 均无优势）一并进确认实例的
limitations。
