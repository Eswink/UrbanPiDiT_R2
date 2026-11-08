# 0106 — 修正 C2 的 on 臂目标函数：改为 lead 对齐的 `physical_targets[:, -1]`，并给前置守卫一条会失败的反证

- **日期**：2026-11-08
- **状态**：accepted
- **范围**：`training/r7_derivative_matching.py::matching_term`、
  `scripts/r7_s3_derivative_matching_c2_protocol.py::_preflight_cases` 与 `preflight` 的守卫、
  `tests/test_r7_s3_derivative_matching_c2.py`（+3 项判别性测试）、
  `tools/reproduce_c2_cross_lead_defect.py`。**本条 0 GPU-h、纯 CPU、零读数。**
- **依据**：决策 0105（缺陷发现与作废）、决策 0095（C2 设计与族判别）、
  决策 0104（同一 `atmos_target` 语义错误在 C1 的发现）、决策 0084 §6

## Context

**① 0105 定性了缺陷，本条修它。** 0105 记录了：`matching_term` 把 `forecasts[:, -1]`（第 12 步 = +72 h）
与 `moved["atmos_target"]` 配对，而 `atmos_target` 是 `physical_targets[:, 0]`（+6 h）。后果是
**完美 +72 h 预测被罚 0.4957–0.6074、梯度范数 0.011982**，而 MSE 在同一点为 0——目标函数的最优解
不包含真值，0095 的族判别按实现而非按物理失效。

**② 修复的可行性先核实过，不是猜的。** `r7_derivative_matching.py` **不在** 17 个
`training_code_sha256` 源文件里，也不在 `model/**.py` 里 ⇒ `load_checkpoint` 的
`model_code_sha256` 校验**不受影响**，因此**不需要重做父迁移**（0075/0083 那类工作）。
代价只是新 attempt 要重新 freeze（它确实在 95 项 `execution_files_sha256` 里）。

**③ 修法。** `matching_term` 改读 `moved["physical_targets"][:, -1]`，并在两处**拒绝**而非兜底：

```python
targets = moved.get("physical_targets")
if targets is None:
    raise ValueError("the c2 axis requires physical_targets in the batch; atmos_target is "
                     "the +6 h field and is not lead-matched to forecasts[:, -1]")
if targets.shape[1] != result.forecasts.shape[1]:
    raise ValueError(... the last forecast has no lead-matched target ...)
target = targets[:, -1]
```

**④ 前置守卫也一起改，因为原守卫按构造通过。** `_preflight_cases` 原本用
`derivative_matching_terms(target, target, …)` 算 `zero_at_target`——**target 与它自己比**，恒零、
与部署的配对无关。现在：

- `target` 改为 `physical_targets[:, -1]`（与预测同 lead）；
- 新增 `nonzero_for_cross_lead_field`：把 **+6 h 的 `atmos_target`** 与 +72 h 的 target 比，
  **必须 > 0**。这才是判别力所在——一个恒零的检查换成另一种写法仍然是恒零；
- `preflight()` 增加一条 `raise`：若该反例不 > 0，**拒绝冻结**并说明守卫已失去判别力。

**⑤ 实测（修复后，用**部署入口** `matching_term` 本身，不是我的手算重建）：**

| case | 完美 +72h 预测经 `matching_term` 的项值 | 梯度范数 | 跨 lead 反例（须 >0） |
| ---: | ---: | ---: | ---: |
| 1000 | **0.0000000000** | **0.0000000** | 0.495737 |
| 9000 | **0.0000000000** | **0.0000000** | 0.591632 |
| 17000 | **0.0000000000** | **0.0000000** | 0.607382 |
| 25000 | **0.0000000000** | **0.0000000** | 0.492830 |

修复前同一测量是 0.4957–0.6074 与梯度 0.011982（0105 记录）。复算工具
`tools/reproduce_c2_cross_lead_defect.py` 现在同时报**数据事实**
（`atmos_target_is_the_6h_field`、`forecast_last_step_is_72h`）与**部署行为**
（`deployed_term_is_zero_at_a_perfect_forecast`），因此它既能复现旧缺陷、也能验证新修复。

**⑥ 反证（关键，因为上一轮的测试全是假阴性）。** 新增的三项测试在**旧代码**上全部失败、在**新代码**
上全部通过：

```
旧（git stash 掉修复）: 3 failed, 21 deselected
  test_matching_term_uses_the_last_physical_target_not_atmos_target
  test_matching_term_refuses_a_batch_without_physical_targets
  test_matching_term_refuses_a_target_with_the_wrong_number_of_steps
新（恢复修复）        : 24 passed
```

第一项测试的构造是判别性的：batch 里 `atmos_target` 故意设成 99.0、`physical_targets[:, -1]` 设成
另一个场，预测等于后者 ⇒ **lead 对齐则精确零，读错字段则非零**。原有 21 项测试在**缺陷代码上也全过**，
这是本轮最值得记的事：**它们测的是算子的性质，没有一项测"运算符读到哪个字段"。**

## Decision

1. **`matching_term` 读 `physical_targets[:, -1]`**，并在缺失或步数不符时 `raise`。
2. **前置守卫加一条会失败的反证** `nonzero_for_cross_lead_field > 0`，并让 `preflight()` 在它不成立时
   **拒绝产出 protocol**。旧的 `zero_at_target` 保留（它对"pred == target 精确零"仍有意义），
   但**它不再是唯一依据**。
3. **新增三项判别性测试**（见 ⑥），并把"旧代码上必须失败"写进本条作为反证记录。
4. **不改 `atmos_target` 的名字。** 它才是根因（0104 与 0105 各咬一次），但改名会动数据契约与大量
   冻结哈希；记入「未决」，不在修正缺陷的 ADR 里顺手做。
5. **本条不重跑 C2。** 重跑要新 freeze（新的 `execution_files_sha256`、新的 attempt 目录、约 44 h），
   属于下一轮的动作；本条只保证"若重跑，目标函数是对的"。

## Consequences

**变容易的：** C2 的 on 臂目标函数现在在完美预测处**精确为零、梯度为零**，0095 声称的族性质
（最优解包含真值）第一次**实际成立**而不是只在 docstring 里成立。守卫从"按构造通过"变成
"有一个能失败的判别器"，复算工具同时报数据事实与部署行为，所以旧缺陷与新修复都能被同一条命令验证。

**变难 / 代价（如实列出）：**

1. **修复没有经过任何 GPU 运行验证。** 它只经过 CPU 上的四项 preflight case 与 24 项测试。
   真实训练里 `moved["physical_targets"]` 是否总在 batch 中、形状是否恒等于预测的 N，
   要等下一次 `stage-seeds` 才第一次在 GPU 上被检验。**我无法声称"修好了就能跑通"。**
2. **新守卫的反例阈值是" > 0"，不是从数据导出的量。** 我用的是四个实测值 0.4928–0.6074，
   但没有声明"低到多少就算失去判别力"。若将来某个数据集的 +6/+72 场恰好近似相同，
   这个反例会变小而仍通过。**这是一个我没有关掉的缺口。**
3. **`zero_at_target` 仍然存在且仍然恒零**（它比的是 target 与自己）。我选择保留它是因为它表达
   "同一场与自己比为零"这条算子性质；代价是**它看起来仍像一个前置检验**，未来可能又有人只读它。
   本条在代码注释里写明"它不是唯一依据"，但**没有删掉它**。
4. **作废成本不会因为修复而追回。** attempt07 的 4.968110 GPU-h 已按 0105 全额记账、零读数。
   本轮三次作废合计 **35.208213 GPU-h**，全部是这个方向的前置核查不够硬换来的。
5. **本条零读数、零科学结论。** 它**不**证明导数域匹配有效或无效；C2 至今**未被测过**。

**否决：**

- **只在 `matching_term` 里兜底（`physical_targets` 缺失时回退 `atmos_target`）。** 否决：
  那会把"字段缺失"变成一个**静默的跨 lead 配对**——正是本缺陷的形状。宁可 `raise`。
- **把 `zero_at_target` 改成 `terms(truth, truth)` 就算修好。** 否决：仍然恒零、仍然无判别力；
  必须加**跨 lead 反例**。
- **顺手重命名 `atmos_target` → `atmos_target_6h`。** 否决：牵连数据契约与大量冻结哈希，
  应单独一次决策；记未决。
- **修好后立即开 attempt08。** 否决：本轮不自行再花约 44 h；重跑需要新的 freeze 与独立记录，
  且"修好"与"值得重开 C2"是两件事（本条只做到前者）。
- **声称 C2 现在可以跑了。** 否决：修复只在 CPU 与单测上验过，GPU 路径未验（负面后果 1）。

## 未决

- `atmos_target` 是否重命名为带 lead 的名字，以及会牵连多少冻结哈希。
- 新守卫的反例阈值是否应设成数据导出的下界（如"低于分析场自身 |div| 的 10%"），而不是裸 `> 0`。
- C2 是否值得用修正后的目标函数重跑：本条只证明**旧实现无意义**，不证明修好后 C2 会赢；
  重开须新 ADR 用**部署配对**的 `zero_at_true` 作族判别证据，并重新 freeze。
