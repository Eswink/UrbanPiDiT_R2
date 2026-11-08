# 0108 — 前置守卫只查"能解析"，等于没查"是不是完整读数"

- **日期**：2026-11-09
- **状态**：accepted
- **范围**：`tools/check_round_receipts.py::_check_one`、`tests/test_check_round_receipts.py`。
  **本条 0 GPU-h、零读数、不改任何冻结判据。**
- **依据**：决策 0103（「通过了形状检查」≠「指向了真东西」）、
  决策 0105（「按构造通过」的前置守卫）、本条的实测矩阵

## Context

**① 背景：目标长文指定 `check_round_receipts.py --require-reading` 作为"确认非半截文件"的命令。**
这个工具存在的理由是：`write_exclusive()` 先 `open("x")` 再 `json.dump(allow_nan=False)`，
非有限值会在**写入过程中**抛错，留下一个**半截文件**；而半截的 `readings.json` **就是** reading
阶段的 receipt，于是 wrapper 会把它当成"已完成"并跳过，整轮以"没有任何判决"结束。

**② 我发现的问题：那个守卫只证明了"能解析成 JSON"，没有证明"含有判决"。** 用七种输入实测：

| 输入 | 修复前 | 修复后 |
| --- | --- | --- |
| 文件缺失 | exit 1 `missing` | exit 1 `missing` |
| 截断的 JSON | exit 1 `truncated or invalid JSON` | 同 |
| `{"hello":"world"}` | **exit 0 通过** | exit 1 `missing keys [...]` |
| `{"scientific_claim": false}` | **exit 0 通过** | exit 1 `missing keys [test_read, protocol_sha256, exit_conditions, seeds]` |
| 有四个键但**没有** `exit_conditions` | — | exit 1 `missing keys ['exit_conditions']` |
| `exit_conditions: null` | — | exit 1 `exit_conditions is not an object` |
| 完整形状 | exit 0 | exit 0 |

⇒ 修复前，**一个恰好闭合了括号的半截文件会被放行**——正是这个守卫要防的那件事。
判决要从 `exit_conditions` 读，而旧实现从不检查它是否存在。

**③ 为什么既有测试没抓到：测试与代码犯了同一个错。** 原有的
`test_a_complete_document_passes_and_missing_seeds_fail` 把 `{"scientific_claim": false}`
当成"完整文档"的样例。**测试断言的是"这个文件应该通过"，所以它把缺陷固化了**——
修完代码后这一项立刻失败（`returncode 1 != 0`），这正是它一直以来在保护错误的假设。

## Decision

1. **`_check_one` 增加可选的 `required_keys`**，只对 `reading` 阶段传
   `("scientific_claim", "test_read", "protocol_sha256", "exit_conditions", "seeds")`，
   并要求 `exit_conditions` 是 **object**。
2. **这两个 protocol receipt 不加 key 要求**：它们由驱动自己校验，在这里重复会造出第二处真相。
3. **不改判决逻辑。** 守卫**不重算**任何出口；它只拒绝"不可能含有判决"的文档。
4. **修正那个把缺陷固化的测试 fixture**，并新增三项：不完整被拒、完整通过、
   `exit_conditions: null` 被拒。

## Consequences

**变容易的：** `--require-reading` 现在真的能区分"半截"与"完整"。七种输入的矩阵全部符合预期，
且**两个方向都测**（不完整的必须失败、完整的必须通过），所以"永远失败"的实现也不会通过测试。

**变难 / 代价（如实列出）：**

1. **我要求的是"键存在"，不是"键里的值有意义"。** 一个 `exit_conditions: {}`（空对象）**会通过**。
   真正的强度是"出口被求值过且结论可读"，那需要跨到 `evaluate_exit` 的语义，
   而本工具**故意不重算**任何判决。**这个缺口没有关掉。**
2. **`seeds: {}` 也会通过。** 三 seed 齐全这件事由 `--require-seeds 41 42 43` 单独表达，
   不是这个键的职责；但如果有人只跑 `--require-reading`，空 `seeds` 不会被拦。
3. **这条缺陷存在了一段时间，而它是"守卫"本身。** 更值得注意的是**它的测试与它共谋**：
   测试用同一个弱假设当期望值，所以测试绿了、守卫虚设了。这与 0105 的
   `derivative_matching_terms(target, target, …)` 是同一种病：**自证式检查**。
4. **本条零读数、零 GPU-h**，不构成任何科学结论。它只保证 attempt08 的读数一落地就能被
   正确判定"是不是半截"。

**否决：**

- **只在 `_check_one` 里检查 `exit_conditions` 是否存在、不加其余四键。** 否决：
  `{"exit_conditions": {}}` 单独存在就能过，等于把门槛降到一个键。
- **让这个工具调用 `evaluate_exit` 重算判决。** 否决：那会把"只读守卫"变成第二处判决来源，
  与 0103 的教训相反（结论必须只有一处）。
- **保留旧测试 fixture、只加新测试。** 否决：旧 fixture 断言的是错误命题，
  留着它会让下一次修改再次被迫迁就那个假设。

## 未决

- `exit_conditions: {}` 与 `seeds: {}` 仍能通过（负面后果 1、2）。是否要求 `exit_conditions`
  必须含 `all`/`verdict`/`delta_72h` 三个键，属于可以再加的一层；本条没有加，
  因为那开始接近"校验判决内容"，需要先想清楚边界。
- 这个守卫与 `r7_round_stages.STAGE_RECEIPTS` 的耦合只靠字符串 `"readings.json"`，
  没有共享常量；若哪天 receipt 改名，两处会静默漂移。
