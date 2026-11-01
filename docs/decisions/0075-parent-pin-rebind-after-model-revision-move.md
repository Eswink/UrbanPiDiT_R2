# 0075 — 模型修订移动后的父钉重绑：tendency / SAM 两个模块改钉当前修订下的同态迁移

- **日期**：2026-11-01
- **状态**：accepted
- **范围**：`scripts/r7_s3_tendency_input_parent.py`（及其继承者
  `scripts/r7_s3_sharpness_aware_minimization_parent.py`）的父钉常量；
  `tests/test_r7_s3_tendency_input_protocol.py` 与
  `tests/test_r7_s3_sharpness_aware_minimization_protocol.py` 里钉 `MIGRATION_RECEIPT_SHA256` 的两行
- **依据**：决策 0052（改绑当前模型修订的先例）、决策 0074（本轮在同一修订下重跑的迁移）、
  `outputs/r7_s3_anchor_feedback_65x65_parent_migrated_20261101_attempt01/migration_receipt.json`
  （`current_model_code_sha256 a38be16c…`，155 张量逐 seed `state_digest` 与 FY 钉逐位相同）

## Context

**① 两条漂移守卫在 HEAD 上是红的（先于本会话的改动）。** 提交 `896eabf`（2026-10-29）把城市静态
协变量接进主干（默认关），`model_code_sha256` 由 R1 冻结时的 `60de3b2d…` 移到 **`a38be16c…`**。
tendency 与 SAM 两个父模块各有一条「钉住当前修订」的守卫
（`assert parent.CURRENT_MODEL_CODE_SHA256 == model_code_digest()`），于是自该提交起，
`tests/test_r7_s3_tendency_input_protocol.py::test_the_model_digest_is_the_current_revision` 与
`tests/test_r7_s3_sharpness_aware_minimization_protocol.py::test_the_model_digest_is_the_current_revision_and_the_expansion_is_new`
在分支 HEAD（`2d12e98`）上失败——**这是别的窗口暂停时留下的松散端，不是本会话引入的**
（本会话未改 `model/**`、`training/**`、`data/**`）。

**② 这两条守卫的设计意图就是「修订移动了就来重绑」。** 它们不保护任何读数：两个轮次的协议都已
归档、读数都已登记；它们保护的是「该模块钉住的迁移确实产自当前修订」这一前提。

**③ 迁移是修订的属性，不是轮次的属性。** `scripts/migrate_r7_parent_state.py` 在当前修订下重跑，
产出 `outputs/r7_s3_anchor_feedback_65x65_parent_migrated_20261101_attempt01/`：
**155 个张量逐 seed 的 `state_digest` 与旧迁移逐位相同**
（41 `7c416ccb…`、42 `857d8804…`、43 `61fdac58…`），只有容器内嵌的代码身份与文件 SHA256 不同；
`semantics_equal: false`（模块树多了文件）与 tendency 迁移当年的同类记录一致、已接受。

**④ 状态级的一切钉值都不变。** 实测（用当前修订的迁移 + 真实 archived spec）：
tendency 的 `expand_parent_state` 复算出的 `expanded_state_sha256` 与 `anchor_init_sha256`
**逐位等于** 模块里钉住的三 seed 常量（41 `7105700b…` / `de0e76be…`），tendency patch 仍逐位为零，
`parent_state_sha256` 等于钉住的 `7c416ccb…`。SAM 模块完全继承 tendency 模块的父钉
（`MIGRATED_PARENT_RUN = migrated.MIGRATED_PARENT_RUN` 等），其自身只另钉控制 spec 下的
`EXPANDED_STATE_SHA256`，不受容器身份影响。

## Decision

1. **重绑**：`r7_s3_tendency_input_parent.py` 的
   `MIGRATED_PARENT_RUN` → `outputs/r7_s3_anchor_feedback_65x65_parent_migrated_20261101_attempt01`、
   `CURRENT_MODEL_CODE_SHA256` → `a38be16c…`、`MIGRATION_RECEIPT_SHA256` → `3e97d78a…`、
   `MIGRATED_PARENT_SHA256` → 该迁移的三 seed 文件 SHA256（41 `7cc3a497…`、42 `dde3ae09…`、
   43 `bd968e5d…`）。SAM 模块**零改动**（它按名继承这些常量）。
2. **两条测试各改一行字面量**（旧 receipt SHA256 → 新 receipt SHA256）。**其余断言一字不动**：
   `MIGRATED_PARENT_STATE_DIGEST`、`EXPANDED_STATE_SHA256`、`ANCHOR_INIT_SHA256`、
   `TENDENCY_PARAMETERS`、`TENDENCY_KEY`、SAM 的 `EXPANDED_STATE_SHA256`/`SAM_RHO`/`SAM_EPS`
   全部保持原值——这些才是真正保护状态的断言，而它们**逐位不变**正是本次重绑成立的证据。
3. **先例与范围**：这是决策 0052「改绑当前模型修订」的第三次执行（前两次：0052 的 FY 轮、
   0058/0074 链路里 R2 自己的迁移）。范围**只**限这两个父模块与其两行测试字面量；
   **不**改任何协议、阈值、读数、出口，也**不**改 `r7_s3_fy_annual_parent`
   （它钉住 `60de3b2d…` 是 R1 归档协议的**回放**钉，重放走归档 `code.zip`，不是漂移）。
4. **跨轮复用同一份迁移目录**：`r7_s3_fy_annual_parent` 复用 tendency 迁移目录是既有先例；
   本次 tendency/SAM 复用 R2 的迁移目录同理。`state_equality_evidence()`（R2 模块内）与
   `MIGRATED_PARENT_STATE_DIGEST`（tendency 模块内）是两份独立的状态等价见证。

## Consequences

**变容易的：**
- 分支 CI 从红回绿（两条守卫按设计意图被满足），Track 1 R2 的登记可以照例记「实验 commit 的
  R7 CPU CI completed/success」。
- 「修订移动 → 父钉重绑」这条维护路径又走过一次并有案可查；状态级钉值不变这一点被实测复核。

**变难 / 代价（如实列出）：**
- **两个已关闭轮次的父模块指向了一个以 R2 命名的目录**：跨轮复用是先例，但命名会让人误以为
  tendency/SAM 依赖 R2。若将来有人重放 tendency/SAM 的归档 `code.zip`，走的是归档里的模块与
  归档的迁移路径，与本模块无关；本模块只服务「在当前修订下重新冻结」这一情形。
- **测试字面量的改动必须被读成「重绑」而不是「放宽」**：改的是容器身份的钉值，状态级的钉值
  一个都没动。若有人只看 diff 不看本 ADR，可能误判为弱化断言——所以本 ADR 必须与该改动同一提交。
- `semantics_equal: false` 继续存在（模块树多了文件）：它与 tendency 迁移当年的记录同类，
  是「模块树语义 digest 对文件集合敏感」的已知现象，不是前向变化的证据；前向等价由
  `896eabf` 自带的 off 路径对照实测与本轮的恒等反证共同支撑。
- **本 ADR 不改任何科学判据**，也不为任何已关闭轮次的读数提供新的解释。

**备选方案与否决理由：**
- **备选 A：留着红 CI，只在 R2 的登记里记「红是先前的」。** 否决：登记要求「实验 commit 的 CI
  completed/success」；带着已知的红去登记会把「先前的红」和「本轮的红」混在一起，违背
  「取消/红不算通过」的判定纪律。
- **备选 B：把两条守卫测试改成比较「模块常量 == 模块常量」。** 否决：那是把「修订移动就来重绑」
  的触发器拆掉，是真正的弱化。
- **备选 C：为 tendency/SAM 各自再跑一份迁移、各用一个目录。** 否决：同样的状态、同样的事件，
  三份迁移目录只会增加混淆；跨轮复用一份迁移目录是既有先例。
