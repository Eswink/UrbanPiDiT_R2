# 0052 — 全年覆盖轮改绑当前模型修订（`3fe747b2…` → `60de3b2d…`）

- **状态**：accepted
- **日期**：2026-10-22
- **范围**：`docs/goals/s3-fy-annual-coverage.md` 的 S3 数据覆盖轮（决策 0050 的修订，不新开轮）

## Context

决策 0050 把全年覆盖轮冻在 `model_code_sha256 3fe747b2…`（H-D 实验提交 `0becf24`）上，
并要求它在那个修订的干净 checkout 里跑。理由是当时 `60de3b2d…` 会同时改**模型输入表示**
（显式 tendency 通道）与数据覆盖，破坏单因子。

那条理由现在已经不成立，而约束本身变成了纯粹的工程障碍：

1. **两个新增开关都是惰性的。** `3fe747b2…` 之后 `model/` 只多了 `tendency_input`（默认
   `False`）与 `decoder_base_mode`（默认 `'persistence'`）两个开关。两者取默认值时，前向与
   损失与 `3fe747b2…` 逐位相同——本轮新增的
   `tests/test_r7_s3_fy_annual_coverage.py::test_the_two_post_hd_switches_are_inert_at_their_registered_defaults`
   直接构造两种 spec（不带键 vs 显式写默认值）并断言 `training_long_rollout` 的 `loss` 与
   `per_step_losses` 逐位相等，把「惰性」从假设变成断言。
2. **本轮的因子本来就不经过模型代码。** 0050 第 1 条已经写明：两臂用同一个 store、同一套全年
   归一化、同一模型 spec、同一迁移父、同 2400 updates、同 seed，**只差训练清单覆盖的月份**。
   模型修订对两臂是对称的，因此在任一个修订上跑，因子都仍然是数据覆盖。
3. **父迁移已经存在，且状态完全相同。** tendency 轮在 `60de3b2d…` 下产出
   `outputs/r7_s3_tendency_input_parent_migrated_20261021_attempt02`（receipt `73c019dd…`）。
   它与 0050 冻结的 `outputs/r7_s3_rollout_dose_parent_migrated_20261015_attempt04` 读取**同一批**
   源 checkpoint（v3-BD 1600 端点 `637857c5…/8f7369fb…/08cc7db9…`），逐 seed 的 tensor
   `state_digest` 也**完全相同**（`7c416ccb…/857d8804…/61fdac58…`）。只有内嵌的代码身份不同，
   因此两个迁移文件的自身 SHA256 不同（`fb1e4d65…/d97ea802…/8aa0fac6…` vs
   `f9f12e06…/8d4e05ed…/5af64655…`），而张量内容没有变化。
4. **干净 checkout 的代价与风险更高。** 那要求一个分离的 worktree、一条只含本轮文件的临时提交、
   与主分支不同步的 CI，而且冻结/归档必须与工作分支分离；同时本轮的数据产物
   （`outputs/r7_s3_fy_annual_20261021_attempt01/`）与 D2 气候态
   （`outputs/r7_s3_fy_annual_d2_20261021_attempt01/reference.json`）都在主工作树里。改绑当前
   修订可以用同一条 `round-guard → freeze → stage` 通道，少一层可能出错的手工步骤。

**这不是「放宽判据」**：`load_checkpoint` 的模型修订门没有被绕过，也没有被削弱。
`scripts/r7_s3_fy_annual_parent.py` 仍然要求 receipt 的 `current_model_code_sha256` 等于
`model_code_digest()`、迁移文件 SHA256 等于 pin 值，并且新增的
`state_equality_evidence()` 会在 freeze 时读**两份 receipt**，逐 seed 断言源 checkpoint SHA 与
`state_digest` 同时相等、且 attempt04 侧仍等于 0050 冻结的 pin。任一不等即报错。

## Decision

1. **本轮改绑 `model_code_sha256 60de3b2d…`**（当前 HEAD 修订）。`scripts/r7_s3_fy_annual_parent.py`
   的 `MODEL_CODE_SHA256`、迁移父路径与三 seed pin 全部改为上面第 3 条列出的当前修订值。
2. **因子不变**：训练清单覆盖的月份数 4 → 12，年份固定 2017–2021。两臂共用同一 store、同一全年
   归一化、同一模型 spec、同一迁移父状态、同 2400 updates、同 K=4/depth=4/dim=192、同注册物理
   权重、同 seed 41/42/43、同 2022 全年 val、同 48 桶 train-only 气候态。**不改 `model/**/*.py`。**
3. **预声明出口逐条沿用 0050 第 5 条**（(a) F 72h skill 三 seed 全 >0；(b) F−N 72h ΔRMSE 三 seed
   全 <0；(c) F 48h skill 三 seed 全 >0；(d) F−N 6/12/24h seed 均值 ΔRMSE ≤0），桥 (e) 仍为
   **非科学判定**。任一 (a)–(d) 假 ⇒ 永久关闭「全年覆盖」支：不扫月份组合、不调 updates、
   不加 seed、不把 2000–2016 塞进同一支、不与已关轴组合。
4. **65×65 限制不变**：`boundary_margins=(1,)`、评分读 `full`（4,225 点），不外推为 129×129
   宽实例结论；不新增下载、不建 129×129 全年源、不碰 `data/raw|interim|processed`。
5. **桥 (e) 的对照仍是 `record:s3-v3-budget-dose`**（`3fe747b2…` 下跑出的窄 v3 四个月臂）。
   它现在同时混有**模型修订**与数据/归一化差异，所以桥的解释力比 0050 时**更弱**，只用于发现
   身份/工程错误，失败记 `identity/engineering unresolved`，不进入 (a)–(d)。
6. **登记**：`test_read=false`、`scientific_claim=false`；结果写证据页 + index record + 账本行 +
   §8 + campaign-state，并核验精确 CI run。父迁移的 `state_equality_evidence()` 进协议，
   供独立复核。

## Consequences

- **收益**：本轮不再需要一个分离的干净 checkout 与临时提交，走与其它轮相同的冻结/归档通道；
  因子仍是纯数据覆盖，且「新开关惰性」由测试断言而不是由人记忆保证。
- **代价 1（桥更弱）**：如第 5 条，桥 (e) 现在同时混有模型修订差异。这是本轮**明确声明的**
  解释力下降，不隐藏。
- **代价 2（与更早读数的可比性）**：本轮的候选臂 F 与已登记的 `3fe747b2…` 臂不可逐位配对；
  但本轮的科学比较是 **F 与 N 在同一进程/同一修订/同一父下的配对**，不受该差异影响。
- **风险与缓解**：若 `model/` 在未来再次变化，`pinned_parent` 与 `state_equality_evidence` 会
  立刻失败并要求重新评估，而不是静默沿用旧 pin。

## References

- 决策 0050（本决策修订的原始边界）、0038（数据扩充授权）、0047（双卡 seed 分配）
- `docs/goals/s3-fy-annual-coverage.md`（本轮 brief）
- `scripts/r7_s3_fy_annual_parent.py`（改绑后的 pin 与 `state_equality_evidence()`）
- `tests/test_r7_s3_fy_annual_coverage.py`（惰性开关断言与 pin 断言）
- `outputs/r7_s3_tendency_input_parent_migrated_20261021_attempt02/migration_receipt.json`（`73c019dd…`）
- `outputs/r7_s3_rollout_dose_parent_migrated_20261015_attempt04/migration_receipt.json`
- R-006（先冻协议）、R-028（离线禁网）、R-054（共驻只读）
