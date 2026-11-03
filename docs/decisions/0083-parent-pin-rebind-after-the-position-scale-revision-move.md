# 0083 — 模型修订再次移动（`59340710…`）后的父钉重绑：tendency / SAM 两个模块改钉 0080 的迁移

- **日期**：2026-11-03
- **状态**：accepted
- **范围**：`scripts/r7_s3_tendency_input_parent.py` 的四个父钉常量（`MIGRATED_PARENT_RUN`、
  `CURRENT_MODEL_CODE_SHA256`、`MIGRATION_RECEIPT_SHA256`、`MIGRATED_PARENT_SHA256`），
  以及 `tests/test_r7_s3_tendency_input_protocol.py` 与
  `tests/test_r7_s3_sharpness_aware_minimization_protocol.py` 里各钉 `MIGRATION_RECEIPT_SHA256` 的一行字面量。
  `scripts/r7_s3_sharpness_aware_minimization_parent.py` **零改动**（它按名继承 tendency 模块）。
- **依据**：决策 0075（同类重绑的先例与其理由）、决策 0052（改绑当前模型修订的更早先例）、
  决策 0080（`model/` 新增 `typed_evidence.position_scale` 使修订移动）、
  `outputs/r7_s3_position_retention_parent_migrated_20261103_attempt01/migration_receipt.json`
  （`current_model_code_sha256 59340710…`）

## Context

**① 红的是哪两条守卫，为什么。** 我在 R4 里按 0080 §Decision 3 给
`model/typed_evidence_r7.py` 加了 `position_scale` 字段（默认 `1.0`、实测逐位不变、零新参数），
`model_code_digest()` 由 `a38be16c…` 移到 **`59340710…`**。tendency 与 SAM 两个父模块各有一条
「钉住当前修订」的守卫，于是 `test_the_model_digest_is_the_current_revision`（两处）在 HEAD 上转红。
**这是 CI 抓到的**（提交 `de63679` 与 `ffd24ef` 的 `R7 CPU CI` 均在第 9 步失败），
我在本地跑的分片测试集没有覆盖到它——这一点如实记下。

**② 这两条守卫的意图就是「修订移动了就来重绑」（0075 §Context ②）。** 它们不保护任何读数：
两个轮次的协议都已归档、读数都已登记。它们保护的是「该模块钉住的迁移确实产自当前修订」这一前提。

**③ 迁移已经在当前修订下跑过一次，且状态逐位相同。** R4 自己的父迁移
`outputs/r7_s3_position_retention_parent_migrated_20261103_attempt01/` 正是用提交的
`scripts/migrate_r7_parent_state.py` 在 `59340710…` 下重跑得到的。实测（本 ADR 前）：

| 检验 | 结果 |
| --- | --- |
| receipt 的 `current_model_code_sha256` | `59340710…`（等于当前 `model_code_digest()`） |
| receipt 的 `archived_model_code_sha256` | `3ddab46b…`（与 0075 时的归档修订相同） |
| `export_manifest_sha256` 与 0075 那次迁移是否相同 | **相同**（`7b076469…`） |
| 三 seed `state_digest` 与 0075 那次是否相同 | **逐位相同**（`7c416ccb…` / `857d8804…` / `61fdac58…`） |
| 三 seed `source_checkpoint_sha256` 与 0075 那次是否相同 | **逐位相同** |
| `n_tensors` | 155（三 seed 一致） |
| `semantics_equal` | `false`（模块树又多了文件），与 0075 当年同类记录一致、已接受 |

**④ 状态级的一切钉值都不变。** tendency 模块的 `MIGRATED_PARENT_STATE_DIGEST` 三值仍是
`7c416ccb…` / `857d8804…` / `61fdac58…`——**我没有动它们，它们仍然通过**。这正是本次重绑成立的证据：
容器身份（revision、文件 SHA256）变了，被保护的状态没变。

## Decision

1. **重绑 tendency 模块**：`MIGRATED_PARENT_RUN` →
   `outputs/r7_s3_position_retention_parent_migrated_20261103_attempt01`、
   `CURRENT_MODEL_CODE_SHA256` → `59340710…`、`MIGRATION_RECEIPT_SHA256` → `393766ee…`、
   `MIGRATED_PARENT_SHA256` → 该迁移的三 seed 文件 SHA256
   （41 `e81e4904…`、42 `7e2cd4cc…`、43 `ee33f690…`）。
2. **SAM 模块零改动**：它按名继承上述常量（`CURRENT_MODEL_CODE_SHA256 = migrated.CURRENT_MODEL_CODE_SHA256`
   等），所以只改 tendency 一处即两处生效。
3. **两条测试各改一行字面量**（旧 receipt SHA256 `3e97d78a…` → 新 `393766ee…`）。
   **其余断言一字不动**：`MIGRATED_PARENT_STATE_DIGEST`、`EXPANDED_STATE_SHA256`、
   `ANCHOR_INIT_SHA256`、`TENDENCY_PARAMETERS`、`TENDENCY_KEY`、SAM 的
   `EXPANDED_STATE_SHA256`/`SAM_RHO`/`SAM_EPS` 全部保持原值。
4. **跨轮复用同一份迁移目录**（0075 §Decision 4 的既有做法）：tendency/SAM 复用 R4 的迁移目录。
   两份独立的状态等价见证仍然成立：R4 模块的 `state_equality_evidence()`（比 R2 迁移）与
   tendency 模块的 `MIGRATED_PARENT_STATE_DIGEST`（比 0075 钉值）。
5. **不改**任何协议、阈值、读数、出口；**不改** `r7_s3_fy_annual_parent`
   （它钉 `60de3b2d…` 是 R1 归档协议的回放钉，重放走归档 `code.zip`，不是漂移）。

## Consequences

**变容易的：**
- CI 的两条守卫重新变绿，而且**没有**通过放宽任何断言达成——改的只有「哪个迁移目录/哪个修订」
  这层容器身份，状态摘要一个字节没动。
- 又一次证明「迁移是修订的属性、不是轮次的属性」：同一份 155 张量状态在第三个修订下仍然逐位相同。

**变难 / 代价（如实列出）：**
- **这是同一模式的第四次执行**（0052 → 0058/0074 → 0075 → 本条）。每 `model/` 改动一次，
  就得再跑一次迁移、再改四处常量与两行测试。这是 R-006 协议冻结与 `load_checkpoint` 强校验的
  **真实代价**，不是可以顺手绕掉的噪音。
- **本地分片测试集漏掉了它**：我提交 `de63679` 前只跑了与本轮直接相关的六个测试文件，
  没跑全会让这两条守卫暴露。**这是本次的实际教训**，已在 §1 写明；后续提交前应跑全量或至少
  跑 `tests/test_r7_s3_*_parent`/`*_protocol` 这一族。
- **本 ADR 零 GPU、零新读数**：它只重绑常量，不得被引作任何实验证据。

**备选方案与否决理由：**
- **备选 A：把两条守卫改成「钉住某个历史修订即可」。** 否决：那正是 0075 §Context ② 说的
  「守卫的意图是修订移动了就重绑」；放宽它等于让 `CURRENT_MODEL_CODE_SHA256` 不再是当前修订，
  名字就骗人了。
- **备选 B：让 tendency/SAM 继续钉 `a38be16c…`，只在测试里 skip。** 否决：skip 不算通过，
  且会掩盖「迁移未在当前修订下重跑」这一真实前提。
- **备选 C：为 tendency/SAM 单独跑一份迁移。** 否决：0075 §Decision 4 已确立跨轮复用同一份
  迁移目录的做法；另跑一份会产生两个内容相同、身份不同的目录，只增加审计面而不增加证据。
