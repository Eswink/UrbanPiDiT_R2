# 0069 — 静态协变量接入模型：`static_identity` 绑进 `protocol_sha256`，并预声明本轮出口

- **日期**：2026-10-29
- **状态**：proposed
- **范围**：`docs/goals/s3-urban-static-covariates.md` 之后的**模型侧接线轮**。
  **不改**已发布的静态数据产物、**不重建** store、**不**把静态量做成 state 通道、
  **不改**任何已冻结判据、**不动** `data/raw|interim|processed`。
- **依据**：决策 0063 第 3 条（"将来模型接线时把 `static_identity` 绑进 `protocol_sha256`
  （不是 `data_identity`）"）；决策 0063 第 3 条同时明确"本 ADR 不授权接线"⇒ 接线需**自己的** ADR；
  `AGENTS.md`「批量实验**必须**在训练开始前把协议写入 `protocol.json`」；
  决策 0067（执行文件清单必须具名）。

## Context

**① 数据已经就绪，而且身份是可核查的（实测）。**

```
outputs/r7_urban_static_covariates_attempt01/published/manifest/BUILD_COMPLETE.json
  {"schema_version": 1, "build_complete": true}
outputs/r7_urban_static_covariates_attempt01/published/manifest/manifest.json
  schema: r7-urban-static-manifest-v1
  static_identity: 8671ed7ce0c82d1ec9ac834e68e2e3c7bbc30e14fa6b8a07501e070d27708b03
  output_sha256:   1e9473f04bf1fa47ac279680a4ac4b54de933fee6afde934478e73f6e2dc37e5
  protocol_sha256: 8cc3c673aa98d69fc9baaa8dd10d539233c738118ceeaeef63851e40a8e54904
  source_sha256:   0115dc3c809e0e6019096a3933cc3cc730c84f7a65326f81598b846212b30d42
  recheck_max_relative_delta: 5.15e-08   （抽样格独立重算 vs 聚合中间量）
  grid: 65×65, 0.25°, lat 降序 / lon 升序；roi 与 R7 冻结 ROI 逐字相同
```

已登记 `record:s3-urban-static-covariates-acquisition`（`engineering-positive`、
`not-candidate`、0 GPU-h）。**数据轮不需要重跑。**

**② 为什么不能把静态量做成第 18 个 state 通道。** 决策 0063 已定：写进 store 的 `attrs`
会改 `data_identity`，把本轮与 R1–R5 的已登记队列**脱钩**（每个历史 receipt 的
`train_data_identity` 都会对不上）；不写进 `attrs` 则 `validate_store` 与 `dataset_identity`
都不管它，两份不同的静态场会得到**同一个** `data_identity`——那是**审计洞**。
⇒ 正确形态是**独立产物 + 身份绑进 `protocol_sha256`**。

**③ 本轮只做"通路"，不做"机制"。** 城市感知的算法设计（如何利用建成区/人口/土地覆盖）
是**再下一轮**的事。本轮只证明：静态场能被模型**按身份**读进来并参与前向，
且关掉开关时**逐位退回**现有行为。这与本项目对每个新通路的既有做法一致
（参考 tendency 轮：`tendency_input` off 路径不构造张量、清零正控制逐位退回）。

## Decision

1. **身份绑定形态：`static_identity` 进入 `protocol_sha256`，不进入 `data_identity`。**
   驱动的 `frozen_constants()` 增加一个 `static` 段：
   `{"static_identity": <sha256>, "manifest_sha256": <sha256>, "fields": [...],
     "grid": {...}, "aggregate_spec": {...}}`。
   该段参与 `canonical_digest` ⇒ 两份不同静态场必然产生两个不同的 `protocol_sha256`。
   **`data_identity` 保持不变**（它描述 store + 清单，静态场不改变它）。
   反证要求：把 `static_identity` 改一个字符 ⇒ `protocol_sha256` **必须**变化
   （否则身份绑定是假的）。

2. **开关与逐位退回。** 模型新增 `static_conditioning` 开关，默认 **off**：
   - **off**：不构造静态张量、不读 `.npz`，前向与本轮之前**逐位相同**。这条**不能**用
     "同一 seed 下 off 臂 vs 新代码另一臂"来证——两臂都跑新代码，那只能证明开关自洽。
     可操作的形态是**改动前后**比较：在**动手改 `model/**` 之前**先落一份参照读数
     （固定 seed 的初始参数摘要 + 一个固定 batch 的 loss），改完后再算一遍，两者必须相等。
     动手顺序就是证据的一部分——这一点必须在执行记录里写明先后，不能事后补。
     跨修订的重放（`tools/r7_archived_replay.py`）仍可用于更早的历史产物，但本轮的 W2
     被判据固定为"改前/改后"，因为它不依赖任何归档是否完整。
   - **on**：只把静态场作为**时不变**条件（非 state 通道、非预测目标的监督）。
   - 参数增量必须**显式申报**；若为 0，说明实现是"调制"而非"额外层"，也要写明。

3. **预声明出口（本轮必须先冻结，再看读数）。** 这是一个**工程/机制**轮，
   不是"静态协变量有没有用"的科学轮。出口写成**可证伪的工程断言**：

   | 出口 | 判据 | 性质 |
   | --- | --- | --- |
   | (W1) | `static_identity` 改动 1 字符 ⇒ `protocol_sha256` 必变 | 身份绑定为真 |
   | (W2) | 参照读数在**改动之前的 `model/` 字节**（`model_code_sha256 60de3b2d…`，= R1 的父固定值）下产生并落盘；改动后在同一 seed、同一固定输入上必须重算出**同一组**摘要：全部参数摘要相等 + 一个 update 后的 loss 与参数摘要相等 | 无副作用 |
   | (W3) | `on` 臂从 seed 初始化跑一份**有界** updates，在 val 上产出 5 个 lead 的 `boundary_rmse.csv`，行数与 off 臂逐行相同 | 通路可用 |
   | (W4) | 抽样 65×65 格上的静态场与已发布产物**逐格相同**（容差 0，float32 位相等） | 读进来的确实是发布的那份 |

   **W3 的一处前置更正（写于任何运行之前）**：初稿把 W3 写成「`n_initializations` 与父一致」，
   那隐含「本轮从迁移父训练」。实际上**任何 `model/**` 字节变化都会改变 `model_code_sha256`**
   （`training/r7_experiment.py:31-39` 对 `model/*.py` 做哈希），而
   `training/r7_experiment.py:124-134` 的 `load_checkpoint` 按设计拒绝外来实现 ⇒
   旧迁移父（`current_model_code_sha256 = 60de3b2d…`）在本轮代码下**不可加载**。
   要加载它就必须重跑 `scripts/migrate_r7_parent_state.py`——那是一次**独立的身份操作**，
   本 ADR 不授权（它有自己的前置：源 checkpoint 的 SHA256 六组 pin）。因此 W3 改为
   「从 seed 初始化、有界 updates、两臂逐行等长」。
   **代价如实写**：本轮读数**不能**与 R1–R5 那些迁移父臂横向比较；它是新代码基线，
   不是那些轮的延续。这一点必须进证据页的 limitations。

   **本轮不设科学判据**：不得宣称"静态协变量改善天气"，也不得据此关闭任何轴。
   任何"on vs off 谁更好"的读数，本轮**只作为下一轮预注册的输入**，不作结论。

4. **`on` 臂必须带算力匹配说明。** 若 `on` 引入参数或额外计算，必须申报
   `parameter_delta` 与每 update 的相对代价；未做 parameter-matched 控制的，
   必须像 tendency 轮那样**在 limitations 里写明**，不得把容量效应说成机制效应。

5. **执行文件清单按决策 0067 具名。** 新驱动用 `tools/r7_explicit_inventory.py`
   的 import 闭包生成清单，写进 `protocol.json` 明文；测试必须含
   "往 `training/`/`data/` 新增一个无关 `.py` 不改变清单"的反证。
   **`scripts/read_r1_seed_rows.py` 这类落在 `scripts/` 的文件**说明：若清单还是活 glob，
   任何新脚本都会打崩在跑的轮次——这正是 0067 要消灭的。

6. **单臂还是两臂，先声明再跑。** 若本轮只验证通路（W1–W4），**单臂 × 1 seed** 足够，
   并在协议里写明"这是工程验证，不是三 seed 一致性证据"；若要做 on/off 比较，
   则必须两臂 × 三 seed，且预注册比较量。**不得**用单 seed 的 on/off 差异下科学结论。

## Consequences

**变容易的：**

- 静态场与模型之间有了**可机械检查的身份链**：`protocol_sha256` 覆盖 `static_identity`，
  所以"训练时用的是哪份静态场"不再靠人记。
- off 路径逐位退回，保证新通路**不可能**悄悄改变已登记的 R1–R5 结论。

**变难 / 代价（如实列出）：**

- **本轮不产出科学结论。** 它买的是"通路存在且可审计"，不是"静态量有用"。
  按本项目已有先例（H-A/H-B/H-C 等），通路轮的价值**完全**取决于下一轮是否预注册了好的比较。
- **静态场是派生统计量，不是观测真值。** 它由 WorldCover/GHSL 面积加权聚合到 0.25°，
  `derived_statistic_not_truth` 已写进已发布 manifest。接线后**不得**在任何报告里把
  它说成"真实地表"。`recheck_max_relative_delta = 5.15e-08` 说明聚合可复现，
  **不**说明聚合语义在物理上正确。
- **LandScan 缺层。** 人口维度只有 GHSL-POP，没有 LandScan 交叉验证；这一点在
  已登记记录里就是 `dropped_layers` 的一部分，接线时**不得**补造。
- **参数增量是最可能被误读的地方。** 若 `on` 加了参数，on/off 差异就混合了"静态信息"
  与"额外容量"两个原因。缓解：Decision 4 强制申报，且本轮不下科学结论。

**备选方案与否决理由：**

- **备选 A：把静态量写进 store，作为第 18 个通道。** 否决：改 `data_identity` ⇒
  与 R1–R5 已登记队列脱钩；且决策 0063 已明确否决。
- **备选 B：不绑身份，只按文件路径读静态场。** 否决：两份不同静态场会得到同一个
  `protocol_sha256`，那是决策 0063 点名的审计洞。
- **备选 C：本轮直接做 on/off 科学比较。** 否决：通路还没验证逐位退回（W2 未过）就做比较，
  一旦读出差异将无法区分"静态信息"与"实现副作用"。先通路、再比较。
- **备选 D：重跑数据轮以顺带接线。** 否决：已发布产物带 `BUILD_COMPLETE` 与
  `static_identity`，重跑只会换身份、不增加证据。

## 未决（留给执行轮）

- 静态场的**变量子集**与**归一化**（按 `full` 区域统计还是按训练格统计）——必须先冻结再跑。
- on 路径的具体注入位置（encoder 前的额外通道 / FiLM 式调制 / 输出偏置）——
  三者在参数增量与语义上不同，必须选定一个并写明**为何**选它。
- 是否需要 parameter-matched 的 off 控制臂（取决于选定的注入方式是否加参数）。
