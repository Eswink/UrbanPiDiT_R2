---
name: decision-record
description: 当要做或记录一个影响架构、约定或长期行为的决定时使用。含 ADR 模板、状态生命周期、编号规则与与 OPEN_QUESTIONS 的分工。
---

# 记录一个决策

## 何时使用

- 你做了一个会影响**后续工作方式**的决定（改目录结构、改数据契约、引入/移除工具、
  改变某条规则的执行方式）；
- 你解决了 `docs/rules/OPEN_QUESTIONS.md` 里的一个 Q；
- 你**否决**了某个方案，且这个否决理由值得后人知道（避免有人反复重提）。

**不适用**：
- 只是执行既定规则（没有新决定）；
- 还没定、需要人拍板的问题 → 那是 `OPEN_QUESTIONS.md`，不是本流程；
- 规则本身的增删改 → 记 `docs/rules/CHANGELOG.md`；
- 观察到的事实 → 记 `docs/rules/EVIDENCE.md`。

## 前置条件

- 确认这个决定**确实已经做出**（不是还在权衡）。没定的走 `OPEN_QUESTIONS.md`。
- 确认它不是已有决策的重复。先读 `docs/decisions/README.md` 的索引与
  `docs/rules/OPEN_QUESTIONS.md`，避免制造第二处真相。

## 步骤

1. **查现有索引。** 读 `docs/decisions/README.md`，确认没有同一个决定已被记录。
   ```bash
   ls docs/decisions/
   ```

2. **取下一个编号。** 编号是**四位、单调递增、不复用**（删过的号也不回收）。
   取当前最大值 +1；如果是第一个，用 `0001`。

3. **写文件** `docs/decisions/NNNN-<kebab-slug>.md`，四段齐全：

   ```markdown
   # NNNN <标题>

   - **日期**：YYYY-MM-DD
   - **状态**：proposed | accepted | deprecated | superseded by NNNN

   ## Context

   迫使这个决定的是什么？（价值中立地陈述事实与约束，不夹带结论。）

   ## Decision

   决定了什么？用主动语态、完整句子写「我们采用 X」。

   ## Consequences

   **变容易的：** …
   **变难 / 代价（如实列出）：** …
   **备选方案与否决理由：** …
   ```

4. **写负面后果。** 这是硬要求（R-033 会机械检查）。照 Nygard 原文：
   "All consequences should be listed here, not just the positive ones."
   代价通常包括：多了一步人工动作、某处可能漂移、引入了新的失败模式。

5. **更新索引。** 在 `docs/decisions/README.md` 的表格加一行（编号 / 标题 / 状态 / 日期）。

6. **如果是解决一个 Q**：写一份指针到该决策记录，Q 本身**保留**原始分析
   （沿用现有 `<details><summary>原始分析</summary>` 惯例，不删除推理过程）。

7. **如果是取代一个旧决策**：旧文件**不删**，把状态改为 `superseded by NNNN`；
   新文件里写明它取代了谁。双向都要能走通。

## 检查点

- **步骤 2 后**：`python tools/check_conventions.py --rule R-033` 应仍通过
  （它会查编号唯一、无跳号、四段齐全、状态合法、含负面后果）。
- **步骤 4 后**：自查「如果有人只读 Consequences，能否知道这个决定的成本？」
  答不上来就是漏了负面后果。
- **步骤 7 后**：`--rule R-036` 必须通过 —— 指向的编号文件必须真实存在。
- **收尾前**：`python tools/check_conventions.py` 全部阻断规则 0 违规。

## 常见失败

- **只改 Q 不写决策记录**：Q 是待决队列；决定一旦做出，就得有一份可检索的决策记录。
  只在 Q 里改字会让决定淹没在问题列表里。
- **编号复用或跳号**：删掉过一个决策就顺手把号让给下一个 —— 不允许。
  号是稳定引用（其它文档会指向它），复用会指向错误的东西。
- **只写好处**：R-033 会拦住；但更重要的是，只写好处的决策记录在后人需要
  重新评估时会失去全部价值。
- **把决策写成任务清单**：那是 `docs/plans/`。决策是"我们为什么选 A 不选 B"。
- **写得太长**：Nygard 建议一到两页、当作"写给未来的开发者的一段对话"。
  超出篇幅通常意味着把背景资料该放别处的东西塞了进来。

## 完成判据

- `docs/decisions/NNNN-<kebab-slug>.md` 存在，四段齐全、状态合法、含负面后果；
- `docs/decisions/README.md` 索引已更新；
- `python tools/check_conventions.py --rule R-033 --rule R-036` 通过；
- 若取代了旧决策，链双向可达；若解决了 Q，Q 有指针且保留原始分析。

## 明确不覆盖

- 不覆盖**还没定**的问题（走 `OPEN_QUESTIONS.md` 与人的决策）；
- 不覆盖规则条文的增删改（走 `docs/rules/CHANGELOG.md` 与对应规则文件）；
- 不覆盖科学结论与方法学取舍（那些走 `docs/R7_*.md` 与 `result-freeze`）；
- 不负责判断某个决定是否"够重要"—— 判据是"它是否改变后续工作方式"，不是规模。
