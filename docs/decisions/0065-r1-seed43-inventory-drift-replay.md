# 0065 — R1 seed43 的执行文件清单漂移：按归档 `code.zip` 在冻结路径下重放，不改比对逻辑

- **日期**：2026-10-28
- **状态**：accepted
- **范围**：`docs/goals/s3-matched-exposure-reeval.md` §4.2 队列第 1 轮（R1）**唯一一个未完成 seed 的
  补跑方式**。**不**改科学门、**不**改判据、**不**改任何已冻结产物、**不**重新冻结 R1 的协议。
- **依据**：`AGENTS.md`「**禁止**绕过 checkpoint / 数据身份校验 …… 旧产物要用它**归档的 `code.zip`**
  重放，不能修改比对逻辑」；决策 0045（跨 attempt 的 pinned seed 复用及其白名单）；
  `.agents/skills/pinned-artifact-replay/SKILL.md:24-25`（"旧产物要用它当时的代码……不要用当前
  HEAD"）；决策 0062（R1 的执行设计）。

## Context

**① 实测事实：seed41/seed42 成功，seed43 在 20.76 s 内 exit 1。**

```
$ .venv/bin/python tools/r7_round.py state
 "next_action": ...（见 §8）
$ cat outputs/r7_s3_fy_year_coverage_matched_20261026_attempt01/seed43.log
RuntimeError: incomplete execution source inventory
  File ".../scripts/study_r7_s3_fy_year_coverage_matched.py", line 279, in validate_protocol
```

三个 seed 的 `stage_ledger.jsonl` 记录：seed41 `success` 89,519.6 s；seed42 `success` 88,362.4 s；
seed43 `failed` 20.76 s（`recorded_at` 2026-10-28T04:09:42+00:00）。

**② 根因是**清单口径**，不是源码漂移。** `validate_protocol`
（`scripts/study_r7_s3_fy_year_coverage_matched.py:275-280`）断言
`set(body["execution_files_sha256"]) == set(execution_files())`；而 `execution_files()`
经 `wide.execution_files()` 落到 `scripts/study_r7_s3_v3_rollout_ft_worker.py:31-37`，
它对 `training/`、`model/`、`data/` 做**活体 `rglob("*.py")`**：

```python
for directory in ("training", "model", "data"):
    names.extend(str(path.relative_to(ROOT)) for path in (ROOT / directory).rglob("*.py") ...)
```

冻结时是 **195** 个文件，现在是 **197**。差集**只有新增、没有删改**（实测）：

```
$ 对比 195 个冻结条目逐一 sha256
frozen entries: 195 / missing: [] / content-changed: 0
$ 集合差
ONLY NOW: ['data/download/urban_static_sources.py', 'data/preprocess/urban_static_aggregate.py']
ONLY FROZEN: []
```

这两个文件由**另一条互不相关的研究线**（城市静态协变量轮，决策 0063/0064）在 commit
`7065f19`（2026-10-27 14:37:32 +0800）引入；`grep` 实测**没有任何冻结清单内的文件 import 它们**，
只有那条线自己的 `scripts/r7_urban_static_round.py` 与 `tests/test_r7_urban_static_round.py` 引用。

**③ 时间线解释了为什么只有 seed43 撞上。** seed41 于 2026-10-26T02:45Z 通过校验、
seed42 于 2026-10-27T03:36Z 通过校验 —— 两者都**早于** `7065f19` 落地（2026-10-27T06:37Z）；
seed43 于 2026-10-28T04:09Z 启动，看到的是 197。⇒ **本轮没有任何科学内容变化，只是一个
过宽的清单口径在中途被外部提交撞到。**

**④ 这不是可以"放宽判据"解决的失败。** 两条错路都要排除：

- **改 `validate_protocol`**（例如把集合相等改成子集）：直接违反 AGENTS.md
  「不能修改比对逻辑」，且会把一个真实的完整性检查变成永久的漏洞。
- **重新冻结 attempt02**：`protocol_sha256` 与 `code_zip_sha256` **都在决策 0045 的
  `REUSE_PROTOCOL_FIELDS` 白名单里** ⇒ 一旦重新冻结，seed41 与 seed42 的成果**按规则不可复用**，
  三个 seed 全部重训（实测单 seed 约 88,000–90,000 s ⇒ 约 **74 GPU-h**），而这一轮的科学问题
  一个字都没变。这是纯粹的时间浪费，不是更严格。

**⑤ 归档 `code.zip` 是完备的，且能逐位复现冻结树。** 实测：

```
frozen entries 195 / missing from zip 1 / hash mismatches 0
```
唯一缺的那 1 个是 `outputs/r7_v2_comparison_20261004_attempt01/protocol.json` ——
归档器按设计不含 `outputs/`（`zip py NOT in frozen` 里 407 个未进清单的 `.py` 也同源），
而它在运行时由真实 `outputs/` 提供。

## Decision

1. **唯一的补跑方式是"用归档 `code.zip` 在冻结的绝对路径下重放 `seed 43`"。**
   不改 `protocol.json`、不改 `code.zip`、不改 `validate_protocol`、不改任何 195 个冻结文件。
   实测该重放在 `validate_protocol` 处**通过**，并复现 `prepare` 记录的同一 digest
   `15a92e3230c5`（`protocol_sha256` 前 12 位）。
2. **"在冻结的绝对路径下"是硬要求，不是实现细节。** `validate_protocol:290-300` 把
   `source.nc` 的**绝对路径字符串**记进协议并比对，因此把归档解到别的根目录会以
   `the merged multi-year source identity changed` 失败（实测）。实现用**非特权
   user+mount namespace** 把归档树 bind 到仓库真实路径，再把真实的 `outputs/` 与 `.venv/`
   bind 回去（数据只读、venv 只读），**不移动、不删除、不修改任何真实文件**。
3. **重放只补 seed43，不重跑 seed41/42。** 两者的 receipt 存在且与本轮协议同源（同一
   `protocol.json`、同一 `code.zip`），按 0045 属"同 attempt 内已完成"，无需再做复用判定；
   本次补跑也不写 `reused_seeds`（那是跨 attempt 复用才需要的记录）。
4. **如实登记这条"重放"事实。** R1 的证据页与 index record 必须写明：seed43 是在归档树重放下
   完成的、原因是外部提交造成的清单口径漂移、以及 195 个冻结文件逐位未变（给出上面的实测行）。
   **不得**把这条写成"无异常完成"。
5. **给后续轮的设计教训（本轮不改 R1）。** `execution_files()` 对三个包目录做**活体 `rglob`**
   是一个**过宽的身份表面**：任何无关研究线在该目录下新增模块都会让已冻结的轮次在校验处崩掉。
   R2–R5 的驱动应改为**显式文件清单**（或把 glob 收窄到本轮实际 import 的闭包），
   在各自的 ADR 里写明；**R1 的驱动保持原样**（它的协议必须保持可读）。
6. **重放机制本身必须可复现，不能只活在一次会话的 `/tmp` 里。** 本条落地为
   `tools/r7_archived_replay.py`（子命令 `prepare` / `verify` / `run`）：
   `prepare` 从归档 `code.zip` 重建重放树，`verify` 在 namespace 内跑驱动自己的
   `validate_protocol`，`run` 把任意命令放进该 overlay。三个内置保护都**实测**过：
   ① 覆盖顺序由 `layout_faults()` 断言（真实的 `outputs/` 与 `.venv/` 必须在归档覆盖根
   **之前**被捕获、之后再 bind 回来；顺序颠倒会 bind 到归档里那个空目录而静默丢掉全部结果）；
   ② `prepare` 拒绝覆盖**有存活进程挂载中**的重放树（`tree_in_use()` 读
   `/proc/*/mountinfo`，实测对 seed43 的树返回 `[3557323, 3557331, 3557334]` 并 exit 3）；
   ③ `run` 在 overlay 布局不合规时拒绝执行（exit 2）。反证在
   `tests/test_r7_archived_replay.py`（9 项，含「把捕获移到 overlay 之后必须报错」与
   「被占用时必须拒绝」）。端到端实测：从 `code.zip` 全新解出的树返回
   `execution_files(): 195` 与 `VALIDATE_PROTOCOL: PASS 15a92e3230c5`。

## Consequences

**变容易的：**

- 只花 **1 个 seed（约 24.5 GPU-h）** 而不是 3 个（约 74 GPU-h），且**不产生新的协议 digest**
  ⇒ seed41/42 的成果继续有效，这一轮仍是同一份冻结协议下的三个 seed。
- 归档 `code.zip` 的完备性被实测证明（194/195 逐位相同 + 集合相等 + `validate_protocol` 通过），
  这同时是对 `pinned-artifact-replay` 纪律的一次真实检验。

**变难 / 代价（如实列出）：**

- **引入了一个新的执行机制：namespace bind-mount 重放。** 它是这一段唯一"不寻常"的东西；
  若有人不看本 ADR 而只看日志，会误以为绕过了身份校验 —— 事实相反：它把树**恢复**到冻结状态，
  比重新冻结更忠实。缓解：证据页逐字引用上文的实测行。
- **重放窗口是有时限的。** 冻结协议的 `deadline_perf_counter` 是机器 uptime 计数器，
  实测剩余 **86.3 h**（`1815221.49 − 1504504.14`）；seed43 的 per-seed 上限 129,600 s（36 h），
  实测两个已完成 seed 用了约 88,000–90,000 s ⇒ 够用，但**不是一个可以无限期推迟的窗口**。
  机器重启会使该计数器归零，届时重放与整轮都需重新裁定 —— 这一点写进证据页的 limitations。
- **清单口径的缺陷只被"绕过"，没有被修掉。** 本轮不改 R1 的驱动（理由见 Decision 第 5 条），
  所以**任何未来的 commit 若再往 `training/ model/ data/` 加 `.py`，这次重放也会再次失败**。
  这是刻意的取舍（改驱动会让协议不可读），代价是本轮的补跑必须在那之前完成。
- **重放不改善任何科学判据。** 它只是让 seed43 得到与 seed41/42 相同的对待；(a)–(d) 的结论
  完全取决于三个 seed 的读数，与本次修复无关。若 (d) 仍假，买到的仍是一个更强的负结果
  —— 这正是决策 0061 已写明的代价。
- **一次三 seed 的一致性证据，无显著性**；GPU 共驻，墙钟不是科学判据。

**备选方案与否决理由：**

- **备选 A：把 `validate_protocol` 的集合相等放宽为子集。** 否决：AGENTS.md 明文禁止修改比对逻辑；
  放宽后任何"少了文件"的情况也会静默通过。
- **备选 B：重新冻结 attempt02 并重跑三个 seed。** 否决：`protocol_sha256`/`code_zip_sha256`
  在 0045 白名单内 ⇒ seed41/42 不可复用 ⇒ 多花约 50 GPU-h 换取一个**更弱**的可比性
  （新 digest 会让本轮三个 seed 分属两份协议）。
- **备选 C：临时移走那两条无关研究线新增的两个模块，跑完再放回。** 否决：它们是已提交的、
  属于另一轮的有效产物（R-002 之外的"不删除已有产物"纪律），移走会破坏那条线的测试；
  而且"跑的时候改工作树"比"在只读快照里跑"更难审计。
- **备选 D：把归档解到 `/tmp` 下、用新根目录重放。** 实测否决：`validate_protocol` 比对
  `source.nc` 的绝对路径，会以 `the merged multi-year source identity changed` 失败。
- **备选 E：本轮顺手修 `execution_files()` 的 glob 口径。** 否决（对 R1 而言）：驱动文件在
  195 个冻结条目里，改它必然改变 `execution_files_sha256` ⇒ 等于重写已冻结协议。
  该教训改写进 R2–R5 的设计要求（Decision 第 5 条）。
