# 0099 — Stage D1.d 执行：缓冲对账已完成，并更正 0097 对 `strict=False` 后果的一处夸大

- **日期**：2026-11-08
- **状态**：accepted
- **范围**：执行决策 **0097** 的 D1.d（把注册父补齐成 `on` 臂可严格加载的状态字典）；
  只新增 `tools/reconcile_r7_static_parent.py` + 其测试，**不改 `model/**`、不改 `training/**`**、
  不改任何冻结判据
- **依据**：决策 **0097**（D1 判定与 D1.d 形态）、**0069**（静态协变量接线的四个出口）、
  `docs/goals/s3-throughput-physics-c2.md` §8；目标长文 Stage D「打通 `static_conditioning`
  到 training 与协议构建器」

## Context

**① 为什么现在能做而 `attempt04` 在跑时不能。** 0097 第 3 条明确「本 ADR 不授权执行 D1.d」，
理由是 C2 的协议钉住了 94 项闭包。`attempt04` 已作废（决策 0098）、`attempt05` 在 0 GPU-h
作废，当前在跑的是 `attempt06`（协议 `029cbaab…`、**95** 项闭包）。核对后确认：
新工具落在 `scripts/` 之外的 `tools/` 下，**不在** 95 项闭包里，也**不改变**
`model_code_digest()`（只哈希 `model/**.py`）与 `training_code_digest()`（只哈希 17 个具名
`SOURCE_FILES`）。实测两者在写入工具前后**未变**（`59340710…` / `e7674118…`）。
因此这次执行对 `attempt06` 的身份**零影响**——执行前逐条核过，不是事后声称。

**② 执行结果（实测，不是设计）。** `tools/reconcile_r7_static_parent.py` 在**注册父**
（`parent_update_0001600.pt`，155 张量，`parent_static_keys == []`）上跑通，产物：

```
reconciled_tensor_count            157          (= 155 + 2 buffers)
reconciled_state_digest            4e56d823dec89b79e625f149a266a7a71229ccd36a3996811be168cef6cbe4ee
strict_load_of_reconciled_passed   true
bare_parent_into_on_arm_failed     true
strict_false_would_leave_missing   [field, projection]
static.field_equals_published_stack true
static.static_identity             8671ed7c…（与 0069 登记值逐字相同）
gpu_hours                          0.0
reconciled_state_sha256            29af995b5b27c4118bdad75cac2cb1cbeec5b20ff2e3fd2f3247f6ad69a477a9
```

即：`on` 臂**可以**从注册父严格出发，前提是把两个缓冲区按**已发布产物**补上并**断言逐位相等**。

**③ 一处必须更正的夸大（本条最重要的内容）。** 0097 写的是 `strict=False` 会**静默丢弃**
两个缓冲区。实测**不是**那样：

| 断言 | 实测 |
| --- | --- |
| `strict=False` 加载裸父后 `field` 是否等于已发布值 | **是**（`torch.equal` True） |
| 同上，`projection` | **是** |
| 三次独立构造的 `projection` 校验和 | **三次相同**（`−1.35186005`） |

根因：`StaticConditioning` 用 `load_static_fields()` 建 `field`、
用 `torch.Generator().manual_seed(0)` 建 `projection`，所以**任何一个新构造的臂本来就带着
正确的数值**。`strict=False` 造成的**不是位错，而是未经验证**。

**因此真正的风险降级为**：`strict=False` 会让一份**未经检查**的缓冲区进入训练，
而它**恰好携带全部静态信息**；若未来 `StaticConditioning` 的构造方式改变（换种子、换变量集、
换 patch 聚合），未经验证的路径会**静默**用上新数值而没有任何断言。**这仍然不可接受**，
但**不能**再说成「丢弃了数据」。0097 的措辞按本条更正，**不改 0097 正文**（它是留痕），
由本条承担更正。

**④ 这个更正不是文字游戏——它改变了工具该断言什么。** 若危险是「位错」，工具只需补值；
既然是「未验证」，工具就必须**逐位断言** `field` 等于已发布栈、并在两边方向上都做
`strict=True` 尝试、把失败**记录**下来。实现就是这么做的（见 ②）。

## Decision

1. **D1.d 执行完毕，形态照 0097 第 2 条落地**，只新增 `tools/reconcile_r7_static_parent.py`
   （248 行）与 `tests/test_reconcile_r7_static_parent.py`（129 行）。
   **不改** `model/**`、`training/**`、任何协议、任何判据。

2. **工具的六条硬性行为（每条都有反证测试）：**
   - 只从**已发布产物**重建两个缓冲区，并**断言** `field` 与 `load_static_fields()` 的栈
     **逐位相等**（不等即 `RuntimeError`，不写入）；
   - 断言 `on` 臂比 `off` 臂多出的键**恰好**是那两个缓冲区（`model/` 若改名，此处失败而不是静默对账空集）；
   - 断言父的键集**恰好**等于 `off` 臂键集（不多不少）；
   - 对**补齐后**的字典 `strict=True` 加载必须成功；对**裸父** `strict=True` 必须**失败**
     （若它成功了，说明 0097 描述的两缓冲区缺口不存在，工具必须报错而不是伪造一份 receipt）；
   - 记录 `strict=False` 会留下哪些 missing，并**实测**这些键的值是否仍等于已发布值
     （结果写进 receipt 的 `strict_false_buffer_values`）；
   - 每次尝试写**排他新目录**，已存在即 `FileExistsError`（不覆盖历史 receipt）。

3. **reproducibility 等级如实标注**：`analysis-reproducible`，**需既有产物树**。
   父 checkpoint 与静态产物都在 `outputs/` 下、不入版本控制，故**干净克隆跑不了**；
   测试对该情形用**具名 skipif/skip**写明原因（`no synthetic fallback`），
   **不**造合成父。这与 0097 Context ⑥ 记录的同一限制一致。

4. **对 `attempt06` 零影响，且这一点有实测支撑**：写入工具前后
   `model_code_digest()`、`training_code_digest()` 均未变，工具不在 95 项闭包内。
   **`attempt06` 不需要重跑、不需要重冻。**

5. **不据此宣称静态协变量有效或无效。** 0069 第 3 条「本轮不设科学判据」完全不变。
   `scientific_claim: false`、`test_read: false`、`r = 0`、`gpu_hours: 0.0`。

## Consequences

**负面后果（代价，必须如实列出）**

1. **0097 的一处措辞被实测推翻，而 0097 正文不改。** 后来者只读 0097 会得到
   「`strict=False` 会丢弃数据」这一**过强**的印象。缓解只有本条的反向指针。
   这是**第二次**出现「前一条 ADR 的经验判断需要后续实测更正」（前一次是 0092 → 0094），
   说明**在 ADR 里写未实测的机制性断言本身有风险**，应尽量写成待测项。
2. **工具只覆盖「补齐缓冲区」这一种缺口。** 若将来 `on` 臂多出的是**参数**而不是缓冲区，
   或静态产物增加变量使 `field` 形状变化，本工具的对账语义都要重新论证；
   它没有通用性，也没有被设计成通用。
3. **仍然没有自动守卫拦住 `strict=False`。** 工具**自己做**严格加载并记录，但仓内
   其它父加载路径仍可用 `strict=False`，没有机制禁止。这条与 0097 的未决项同源，**未解决**。
4. **干净克隆不能复现**（见 Decision 3）。父与静态产物都在 `outputs/`；
   测试 coverage 在有产物树时才完整（7 项，缺产物时部分 skip）。
5. **`reconciled_state.pt` 是一份 157 张量的新产物，会占磁盘**，且它与父的
   `parent_state_digest` 不同（多了两个缓冲区）。任何后续轮次引用它时必须说明
   「父 + 已发布静态缓冲区」，不能只说「从注册父出发」。
6. **`gpu_hours: 0.0` 是构造性的，不是测量的。** 全部工作是 CPU 上的 `load_state_dict`
   与一次 `numpy().tobytes()`，没有 GPU 参与；不要把它当成「实测 0」与其它轮次的实测值并列比较。

**正面后果**：目标长文 Stage D 的「打通 `static_conditioning`」现在有了**可执行的最后一步**——
`on` 臂能从注册父严格出发，且「缓冲区就是已发布产物」是**断言**而不是假设；
D1.d 从 0097 的「待办」变为**已执行并留 receipt**；一处过强的风险描述被实测更正为准确描述。

**否决**：

- **用 `strict=False` 直接加载父就开始训练。** 否决：实测数值虽正确，但**没有任何断言**
  证明它就是已发布产物；一旦 `StaticConditioning` 的构造改变，这条路径会静默用错数据。
- **照 0075/0083 重跑父迁移。** 否决：`model/` 字节未变（0097 ④），重跑只换身份、
  不增加证据，还会把 `model_code_sha256` 推走使在跑的 `attempt06` 失效。
- **把这条更正留到以后再说。** 否决：`attempt06` 之后若有人按 0097 的字面理解去排查
  「数据是否被丢弃」，会白花时间；且夸大的风险描述可能被用来支持一个不必要的重开。
- **给工具加通用 `strict=False` 拦截。** 否决（**本轮**）：那要改 `training/`，
  在 `attempt06` 解析 `validate_protocol` 的窗口内不可做；已列为未决项。

## 未决（留给下一轮）

- 是否值得在 `training/` 的父加载路径上把 `strict=False` 变成硬错误（需在 `attempt06`
  结束、其 `training_code_digest` 不再被校验之后）。
- 若将来 `on` 臂引入**参数**增量（而非缓冲区），D1.d 的对账语义要重写还是复用。
- `reconciled_state.pt` 若要被下一轮的 `parent_weights` 直接使用，需要一个
  `initialization.description` 的准确措辞（0097 未决项仍未定）。
