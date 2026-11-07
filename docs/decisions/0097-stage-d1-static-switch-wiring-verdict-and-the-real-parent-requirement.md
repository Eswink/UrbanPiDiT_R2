# 0097 — Stage D1 接线判定：开关**已**贯通到一般训练路径，但目标长文设想的「父迁移」前提不成立，真正缺的是**缓冲区对账**

- **日期**：2026-11-08
- **状态**：proposed
- **执行前置**：C2 在跑（`attempt04`，协议 `ac61e3f…`），其 `execution_files_sha256` 钉了 **94** 个文件
  （含 17 个 `training_code_digest` 源文件中的全部）。本 ADR 只做**只读判定**与**形态定义**，
  **不写任何 `model/**` 或 `training/**` 代码**，不冻协议、不跑 GPU
- **范围**：`docs/goals/s3-throughput-and-physics-campaign.md` Stage D 的接线部分（D1）；
  白盒化部分（D2）已于 2026-11-07 完成并登记（`docs/R7_S3_WHITEBOX_TRAINED_STATE_RESULT.md`）
- **依据**：决策 **0069**（静态协变量接线，已 `accepted`）、**0075**/**0083**（父钉重绑的两次先例）、
  目标长文 Stage D 原文（「把 `static_identity` 绑进 `protocol_sha256` 而非 `data_identity`，
  打通 `static_conditioning` 到 training 与协议构建器，修『接线报告称四出口已通过但 ADR 0069
  仍 proposed』的漂移；**这会改 `model_code_sha256` 故必须重做父迁移**，照 0075 与 0083 先例」）

## Context

**① 先说结论性事实：D1 的三件事里，两件已经完成，第三件的**前提是错的**。**

审计方（完成校验器）称「Stage D1 接线未做」，依据是 `training/*.py` 里对
`static_conditioning` / `static_identity` **零引用**。这个观察是**对的**，
但它的**推论是错的**：`training/` 不需要引用这个键，因为开关是**经由 spec 字典**
流通的，而 spec 由 `training/r7_experiment.make_model` 原样 `**options` 展开。
「零引用」在这里是**正确的设计形态**，不是缺失。

**② 实测：开关确实贯通一般训练路径（不是只在该轮自己的脚本里）。**

下面的判定全部在**当前仓库字节**上实跑，用的是**注册 recipe 的 spec**（从在跑的
`attempt04/protocol.json` 之外的同源构型取得），不是手搓的 spec：

| 断言 | 实测 | 意义 |
| --- | --- | --- |
| 注册 spec 是否自带该键 | `'static_conditioning' in spec` → **False** | 默认 off，符合 0069 第 2 条 |
| `make_model('process', spec)` 是否吃这个键 | **是**（不抛 `TypeError`，`__init__` 显式收它） | 一般工厂路径已贯通 |
| 参数增量 | off/on 均 **3,286,037**，delta **0** | 符合 0069 第 2 条「若为 0 要写明是调制」 |
| `state_dict` 键集 | 键集**不同**：on 臂多 **2** 个 | 见 ③ |
| 开关关闭时是否构造张量 | `static_conditioning_pathway` 为 **None** | off 路径不读 `.npz`、不加项 |
| 开关打开时是否真的加项 | 项形状 `(1, 1089, 192)`、范数 **182.7212**、含非零 | 不是空转 |
| 加载的训练参数里有没有静态量 | `[n for n,_ in named_parameters() if 'static' in n]` → **`[]`** | 「信息差而非容量差」 |
| 两个缓冲区是否确定 | 两次独立构造逐位**相同**（`torch.equal` True） | 见 ④，这是对账可行的根据 |
| 缓冲区是不是发布的那份 | `field` 与 `load_static_fields()` 的栈**逐位相同**（`torch.equal` True），8 变量 | W4 在**构造层**成立 |

`make_model` 是 `training/r7_long_rollout_runner` 做模板校验时**同一个**工厂
（`_validate_imported_model` 里 `template = make_model(contract["kind"], spec)`），
所以「驱动传进来的模型」与「模板」在**同一 spec** 下键集与形状一致——
这正是 0069 W3 那次真的跑起来（两臂各 120 updates、产出五个 lead 的
`boundary_rmse.csv`）得以成立的原因。

**③ 真正的缺口是缓冲区键的不对称，而且它在两个方向上都存在（实测）。**

on 臂比 off 臂多的两个键是 **buffer**（不是参数），且都是 `persistent=True`：

```
backbone.static_conditioning_pathway.field       is_buffer True  persistent True  float32  (8, 65, 65)
backbone.static_conditioning_pathway.projection  is_buffer True  persistent True  float32  (16, 192)
```

于是**严格加载在两个方向上都失败**，实测：

| 方向 | `strict=True` | `strict=False` |
| --- | --- | --- |
| off 父 → on 臂 | **RuntimeError**（`Missing key(s)`，缺 2 个） | **静默丢弃 2 个键**（`missing 0 / unexpected 2`） |
| on 检查点 → off 臂 | **RuntimeError**（多了 2 个 `unexpected`） | 静默丢弃 |

**这条为什么重要**：`scripts/migrate_r7_parent_state.py:105` 用的是
**`strict=True`**，`training/r7_autoregressive_runner._validate_imported_model`
喂 `parent_weights` 时也是 **`strict=True`**。因此**注册的那个迁移父
（`outputs/r7_s3_position_retention_parent_migrated_20261103_attempt01/`）
不能直接用于任何 `on` 臂**——实测该父的 `state_dict` 里静态键为 `[]`
（它是 off 臂产物，构造时开关关着，两个缓冲区从未被构造，自然没存）。
`strict=False` 是**可以**绕过的，但那正是本条要禁止的东西：它会**静默**跳过
两个缓冲区，而这两个缓冲区**恰好携带全部静态信息**——跳过之后 on 臂
会在**没有静态场**的情况下点亮开关，或者更糟，用 `StaticConditioning.__init__`
自己重建的缓冲区（数值上相同，但**没有任何东西证明**它等于发布产物）。
这是一个「看起来跑通了」的静默错误，与 0069 第 2 条要防的是同一类。

**④ 因此目标长文那句「这会改 `model_code_sha256` 故必须重做父迁移」在本情形下不成立。**`model_code_digest()` 对 `model/**.py` 的**字节**做哈希
（`training/r7_experiment.py:31-39`，`rglob('*.py')`）。而
`static_conditioning` 的模型侧代码**早就存在**——`896eabf`（2026-10-29）
就已落盘，早于注册父的迁移（2026-11-03）。**把开关从 `False` 改成 `True`
不改变 `model/` 的任何一个字节**，因此 `model_code_sha256` **不动**，
0075/0083 那种「改 `model/` ⇒ 重跑迁移并重绑四处常量」的仪式**不适用**。

反过来，上一轮（0069 的 W3）之所以「从 seed 初始化」而不是从迁移父出发，
当时写下的理由是 `model_code_sha256` 不匹配——那条理由**现在已过期**
（0080/0083 已把父重绑到 `59340710…`，与当前字节相符）。**真正仍然成立的
理由是 ③ 的缓冲区缺失**，而它**不是**一次代码身份迁移，是一次**状态字典对账**。

**⑤ 有一件事本 ADR 必须记下来：campaign 里那条「必须重做父迁移」的措辞，
在写它的时候就是对未来的一个错误预测。** 它不是「现在被推翻了」，
而是**从写下的那天起就不成立**，因为开关先于父迁移存在。据实登记，
不改目标长文（它是不可变记录），由本 ADR 承担更正。

**⑥ 可复现等级：本条**不是**全仓可复现的，据实标注。** ② 的十项实测用的是
**注册 recipe 的 spec**，而该 spec 的来源是
`r7_s3_curve_point_large_protocol.model_spec()` → `dose.recipe._shared().archived_process_spec()`，
它**读 `outputs/r7_v2_comparison_20261004_attempt01/protocol.json`**。实测：在
`126c08e` 的**干净工作树**里跑同样的脚本，第一步就
`FileNotFoundError: outputs/r7_v2_comparison_20261004_attempt01/protocol.json`
——因为 `outputs/` 不入版本控制（R-035 的既有判定）。
所以本条的复现等级是 **`analysis-reproducible`，且需既有产物树**，
**不是** `clean-clone-reproducible`。这不影响判定本身（三处核心事实——
`model/weather_forecaster_r7.py` 的 `static_conditioning_pathway` 构造与读取、
两个 persistent buffer 的存在、`migrate_r7_parent_state.py:105` 的 `strict=True`——
都可以在干净克隆里由**源码**直接读到），但「十项数值」需要
`outputs/r7_v2_comparison_20261004_attempt01/` 在场。**下一个 ADR 若要建守卫，
应当把 spec 从源码可复算的常量补齐**，而不是依赖一份归档。

## Decision

1. **D1.a 与 D1.b 判定为已完成，不再重复做。**
   - **D1.a**（`static_identity` 进 `protocol_sha256`、不进 `data_identity`）：0069 的 W1
     已**实测**——`static_identity` 改 1 字符 ⇒ `protocol_sha256 48c09621… → a52465be…`，
     两臂 `data_identity` 同为 `81307008…`。本 ADR 不重跑。
   - **D1.b**（把开关打通到 `training/` 与协议构建器）：按 ② 的十项实测**成立**。
     `training/` 零文本引用是**设计**（spec 字典流通），不是缺失；判据是
     「`make_model` 吃这个键 + 开关两侧行为可测」，两者都已实测。
   - **D1.c**（漂移：报告称四出口已通过而 0069 仍 `proposed`）：已于 2026-11-05 修完，
     0069 状态 `accepted`，文末「状态更正」逐条列了凭证。

2. **D1.d（新，替代被证伪的那一条）：`on` 臂从注册父出发需要的是一次**缓冲区对账**，
   不是一次代码身份迁移。** 形态固定为：

   - 读注册父的 `state_dict`（**不含**两个静态键）；
   - 用 `load_static_fields()` 从**已发布产物**重建 `StaticConditioning`，
     取它的 `field` / `projection` 两个缓冲区；
   - 断言这两个缓冲区与 `load_static_fields()` 的栈**逐位相等**
     （即「重建出来的确实是发布的那份」，不是「碰巧数值相近」）；
   - 用 `build_state_dict` 意义上**显式**拼出 on 臂的完整字典后 `strict=True` 加载，
     **禁止** `strict=False`；
   - 落盘一份 receipt，记录父 `state_digest`、两个缓冲区的 SHA256、
     `static_identity`、以及「对账后 `strict=True` 通过」这一事实。

   **这不是训练、不产生读数、不改任何判据、0 GPU-h。** 它唯一的产出是
   「`on` 臂可以严格地从注册父出发」这一条可复算的工程事实。

3. **本 ADR 不授权执行 D1.d。** 理由同 0095/0096：C2 的 `reading` 会
   `training_code_digest()` 校验 `training/**` 的 17 个源文件、并逐文件校验 94 项闭包。
   对账脚本若落在 `scripts/`（新文件）则**不会**改 `training_code_digest()`
   （它哈希的是一份**固定元组** `SOURCE_FILES`，不是 glob），也不会进 94 项闭包；
   但本 ADR 仍然选择**等 C2 结束再写**，因为「从在跑轮次的闭包之外动手」这类
   判断已经在 2026-11-08 的第三次 CI 红里错过一次（先改被钉文件、后验证），
   **不重犯**。

4. **一处必须写明的口径不对称**（据实记录，不声称等价）：`model_code_digest()`
   对 `model/**.py` 用 `rglob`，所以**新增**一个 `model/` 下的文件会让它变；
   而 `training_code_digest()` 只哈希 17 个**具名**文件，所以新增 `training/` 下的
   文件**不会**让它变。这条差异来自两个函数的不同实现，**不是**设计意图；
   本 ADR 只如实记录，不在 C2 运行期间利用它。

5. **不得据本 ADR 宣称静态协变量有效或无效。** 0069 第 3 条「本轮不设科学判据」
   完全不变；③④ 讨论的是**能不能加载**，不是**加载了有没有用**。

## Consequences

**负面后果（代价，必须如实列出）**

1. **D1 仍然不是一个已完成的 Stage。** 本 ADR 把「接线未做」这个错误指控纠正为
   「两件已完成、一件前提被证伪、缺的是一件新的小工程」，但**缺的那件确实还没做**。
   在 D1.d 执行并登记之前，Stage D 的接线部分只能报**部分完成**。
2. **目标长文的措辞与事实不符，且不修。** 「必须重做父迁移」这句话会继续留在
   不可变记录里，后来者只读目标长文会得到错误印象。缓解只有本 ADR 的反向指针。
3. **缓冲区对账把一个真实的陷阱变成了必须显式处理的步骤。** 在此之前，
   `strict=False` 会静默吞掉两个键；本 ADR 只是**命名**了这个陷阱并禁止它，
   并没有机制在**任何**未来代码里拦住 `strict=False`（没有自动守卫）。
4. **判定依赖一份具体构型的 spec。** ② 的十项实测用的是 Process/dim-192/depth-4
   的注册 recipe；换架构（`native`/`generic`）或换维度时
   `weather_forecaster_r7` 的路径是否同样成立**没有测**。
5. **`parameter_delta = 0` 这条读数容易被误读。** 它意味着 on/off 的差异
   纯是**信息差**；但也意味着任何 on/off skill 差异**不可能**被解释为「多了参数」——
   反过来说，**也没有**一个 parameter-matched 控制臂能用来分离「信息」与
   「调制方式」。这条限制在下一轮做比较时仍然在。
6. **本 ADR 为 0 GPU-h、零读数。** 它不应被计入任何 Stage 的算力，
   也不构成 D 段约 1 GPU-h 预算的消耗。
7. **十项数值不是干净克隆可复现的**（见 Context ⑥）：spec 取自归档的
   `outputs/r7_v2_comparison_20261004_attempt01/protocol.json`，而 `outputs/` 不入版本控制。
   干净克隆只能复现**定性**部分（源码可读的三个事实），复现十项数值需要既有产物树。

**正面后果**：把一条**会导致静默错误**的实现细节（两个 persistent buffer 在
`strict=False` 下被丢弃）在**动手之前**测出来并写下来；纠正了一条从写下之日起
就不成立的目标长文预测；让 D1 的剩余工作量从「重跑一次父迁移 + 重绑四处常量」
（0075/0083 量级，且会改 `model_code_sha256`、使**所有**已冻结身份失效）
缩小为「一次确定性的缓冲区对账」（0 GPU-h、不改任何字节）。

**否决**：

- **直接把 `training/` 加一句 `if spec.get("static_conditioning")` 之类的引用。**
  否决：那是为了满足一个**文本 grep 指标**而改生产代码，而开关路径**已经**通了（②）；
  且它正好会改 `training_code_digest()`，在 C2 跑完前让 C2 身份失效。
- **用 `strict=False` 加载父权重。** 否决：会**静默**丢弃两个缓冲区（实测
  `missing 0 / unexpected 2`），而它们携带全部静态信息——静默错误，见 ③。
- **照 0075/0083 重跑父迁移。** 否决：`model/` 字节没变（④），重跑只会换身份、
  不增加任何证据，并把 `model_code_sha256` 从 `59340710…` 推走，
  使已登记的各轮一致性与在跑的 C2 **同时**失效。
- **把 `static_conditioning` 写进 store 当第 18 通道。** 否决：0063/0069 已定
  （改 `data_identity` ⇒ 与 R1–R5 已登记队列脱钩）。
- **宣称「training/ 零引用 ⇒ 未接线」并据此重做一遍。** 否决：② 的十项实测
  直接否证该推论；重做会产生一个**新的** `model_code_sha256` 与一批新读数，
  却回答不了任何未被回答的问题。

## 未决（留给执行轮）

- D1.d 的对账脚本放 `scripts/` 还是 `training/`：**倾向 `scripts/`**
  （新文件不进 `training_code_digest()` 的 17 项固定元组，也不改 `model_code_digest()`），
  但要在下一个 ADR 里写明它与**既有** `training/r7_parent_import.py` 的职责边界。
  **那条工具的 docstring 已经写下了同一条原则**——「a fresh current-code Process model
  must account for every parent tensor **without adding or dropping a state key**」——
  所以 D1.d 不是新发明，而是这条既有原则在一个**新增两个 persistent buffer** 的情形下的
  第一次应用；下一个 ADR 应当**复用**该原则的措辞，而不是另造一套。
- 是否需要一个**自动守卫**，让 `strict=False` 在本仓的父加载路径上直接报错
  （本 ADR 只记录陷阱，未建守卫）。
- `on` 臂若从对账后的父出发，`initialization.description` 该怎么写才既不是
  「seeded initialization」（那是 0069 W3 的形态）也不是「verified parent」
  （那是 0075/0083 的形态）——需要下一个 ADR 定。
