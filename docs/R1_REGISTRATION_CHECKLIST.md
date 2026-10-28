# R1 登记清单（receipt 落盘后照做，逐步可核查）

**适用前提**：`outputs/r7_s3_fy_year_coverage_matched_20261026_attempt01/seed43_treatment_receipt.json`
与 `seed43_control_receipt.json` 都已落盘，且 `seed43_process.json` 显示 exit 0。

本文件是**机械步骤表**，不是判决来源。判决**只能**来自 `readings.json` 的 `exit_conditions`；
不得手算、不得把跨 attempt 的数字混在一起。任何一步的实测输出与预期不符，就停在那一步并在
`docs/goals/s3-matched-exposure-reeval.md` §8 写下"为何停 + 恢复第一条命令"，**不得**跳过。

---

## 0. 先确认 receipt 真的落盘（不做这一步就往下走是最容易犯的错）

```bash
cd /data/esw/UrbanPiDiT_R2
ls -la outputs/r7_s3_fy_year_coverage_matched_20261026_attempt01/seed43_*_receipt.json
.venv/bin/python tools/r7_round.py stage-plan seed 43
```

预期：两份 receipt 存在；`stage-plan` 返回 `"already_done": true`、`"already_running": false`。
**若 `already_running: true`**：worker 还在跑（含 control 未完成），**停**，不要 spawn 任何东西。
**若 receipt 缺失且无 worker 存活**：见本文件末尾「失败恢复」。

## 1. 在决策 0065 的重放内跑 `reading`

真实工作树里 `execution_files()` 是 **197**，冻结清单是 **195**，直接跑必然
`RuntimeError: incomplete execution source inventory`。**必须**走重放：

```bash
.venv/bin/python tools/r7_archived_replay.py run -- \
    .venv/bin/python tools/r7_round.py stage reading
```

预期：退出 0，生成 `outputs/.../readings.json`。
首次使用前若重放树不在，先：

```bash
.venv/bin/python tools/r7_archived_replay.py prepare \
    --code-zip outputs/r7_s3_fy_year_coverage_matched_20261026_attempt01/code.zip
.venv/bin/python tools/r7_archived_replay.py verify \
    --out outputs/r7_s3_fy_year_coverage_matched_20261026_attempt01
# 预期：execution_files(): 195 / VALIDATE_PROTOCOL: PASS 15a92e3230c5
```

**余量闸门已实测会放行**（free **18.0 GiB** ≥ required **5.7 GiB**，实测
`free_bytes=19326304256`、`required_bytes=6119489536`），`nvidia-smi` 在 namespace 内可读。

## 2. 只读 `readings.json` 的判决（唯一判决来源）

```bash
.venv/bin/python - <<'PY'
import json
r = json.load(open('outputs/r7_s3_fy_year_coverage_matched_20261026_attempt01/readings.json'))
print('seeds present:', sorted(r['seeds']))
print(json.dumps(r['exit_conditions'], indent=1))
print(json.dumps(r['bridge_e'], indent=1))
PY
```

**先确认 `sorted(r['seeds']) == ['41','42','43']`**，再看 `exit_conditions`。
缺 seed 时 `evaluate_exit` 会因缺件返回全 false —— **那不是判决**，不得据此关闭分支。

判据（冻结，不得事后放宽）：

| 出口 | 判据 | 任一 seed 不满足即整体 false |
| --- | --- | --- |
| (a) | 候选 F 的 72 h t2m skill > 0 | |
| (b) | F−N 的 72 h ΔRMSE < 0 | |
| (c) | F 的 48 h t2m skill > 0 | |
| (d) | F−N 在 6/12/24 h 的 **seed 均值** ΔRMSE ≤ 0 | 逐 lead 分别判 |

(bridge_e) 是**工程核对**，不是科学判决；不通过要如实记录但**不改变** (a)–(d) 的结论。

**任一 (a)–(d) 为 false ⇒ 按预声明永久关闭「训练年份覆盖」这一轴**（这是第二次也是最终一次关闭，
因为算力此时已匹配）。

## 3. 填证据页

```bash
.venv/bin/python scripts/read_r1_seed_rows.py --seed 43 --arm treatment
```

把输出逐行替换 `docs/R7_S3_FY_YEAR_COVERAGE_MATCHED_RESULT.md` §6 中 seed 43 那一列
（当前是 `待 seed43`）。**不要手抄别的来源**。同时补 §2 表格的 seed43 列与 §7 成本表第 6 行
（实测耗时取自 `stage_ledger.jsonl`，不是估计）。

核对一致性：`readings.json` 的 `full_t2m_rmse` 必须与 `--arm treatment` 输出的 t2m 行逐位相同。

## 4. 四件套登记（顺序不可换）

### 4a. index record（追加一行到 `docs/R7_EVIDENCE_INDEX.jsonl`）

必填字段与约束（`tools/verify_r7_evidence_index.py:71-140`）：
`format="urbanpidit-r7-evidence-record-v1"`、`record_id`（kebab-case）、`evidence_path`、
`evidence_commit`（40 位小写 SHA）、`reason`、`evidence_sha256`（须与证据页实际字节相同）、
`experiment_commit`、`protocol_sha256`、`data_identity`、`source_sha256`、
`outcome_class ∈ {audit, engineering-positive, mixed, negative, unresolved}`、
`candidate_state ∈ {blocked, candidate, needs-review, not-candidate}`、`priority ∈ [0,100]`、
`scientific_claim: false`、非空 `limitations` 列表、`metrics` 为标量 JSON 对象。

⚠ `evidence_sha256` 必须在**证据页文字最终定稿之后**计算，否则登记提交会自相矛盾。

**⚠ `data_identity` 在本轮是个需要判断的字段（先查清，别照抄 FY 记录）。**
校验器要求 `candidate_state == "candidate"` 时 `data_identity` **非空**，而本轮
`protocol.json` 的**顶层** `data_identity` 是 **`None`**（实测），因为本轮是**两臂单因子**轮次：
两臂各有自己的清单身份，没有单一"数据集身份"。实测值：

```
control   train_data_identity = 71bb33e01a9d44024be9895c8e7728f712e0cad61ee026a111688b0c0416ab44
treatment train_data_identity = 81307008b572e3ff032d5251e788c582f339b0e3dbbc3fea4226d44c6a335af0
```

已登记的两条 candidate 记录（`s3-fy-annual-coverage-result`、`b2-multiseed-negative`）
都是**单臂**轮次，它们的 `data_identity` 就等于那一臂的身份，因此**不能**直接照抄该约定。
本轮按"记录的是**候选臂**（treatment）的数据身份"写，并把**两臂**身份都放进 `metrics`
（`train_data_identity_control` / `train_data_identity_treatment`）以便审计；
`reason` 里必须写明"本轮是两臂单因子，`data_identity` 记的是候选臂，
控制臂身份见 metrics"。**这需要在登记时确认校验器与 `C-03` 的数值核对都通过**。

**单因子断言已实测通过**（`require_single_factor` 对冻结的两臂，未做任何剔除）：
两臂**只**在 `('train_data_identity', 'train_manifest', 'train_windows')` 上不同，
其余 18 个受控项共享 ⇒ 这是"唯一改变的是训练清单"的直接证据，可写进 `reason`。

### 4b. 账本行（**必须 4a 之后**，顺序不可换）

```bash
.venv/bin/python tools/r7_round.py ledger-plan <新的 GPU-h>
```

写入 `docs/goals/main-model-climatology-campaign.md` §8 的账本表。累计与 `used_gpu_h` 必须同时
更新，否则 `check_campaign_state.py` 的 C-02 会失败。

**⚠ 两条硬约束，已在 master plan 的临时副本上用真实校验器预演过（实测，非推测）**：

1. **index record 必须先于账本行存在。** 预演时账本行引用了一个尚未入索引的
   `record:<id>`，`check_campaign_state.py` 立刻报
   `FAIL C-03 ledger row '...' cites unknown record 'r1-dryrun-probe'`。
   所以顺序是 **4a → 4b**，反过来会红。
2. **记录的 `metrics.gpu_hours` 必须等于账本行的 GPU-h**（容差 **1e-4**，
   `check_campaign_state.py:127-133` 的 `record_gpu_hours` 只读
   `metrics.gpu_hours` 或顶层 `gpu_hours`）。**账本行的数字因此必须写进记录本身**，
   而不能只写在 `reason` 文字里。

**C-02 算术在预演中自动成立**：新累计 = 上一累计 + 新 GPU-h、`used_gpu_h` 同步、
`remaining_gpu_h = cap − used`，三者一致即可，无需手工校验其他字段。
（预演只报了上面那一条 C-03 失败，其余 8 条 note 都是既有行的历史豁免。）

**账本行的 GPU-h 只能用 `stage_ledger.jsonl` 的实测秒数换算，不得把 estimate 写成实测。**
seed43 未完成前，整轮成本**不可**填入账本行。

已实测的**当前**成本（截至 seed42，不含 seed43）：

| 项 | 秒 | 备注 |
| --- | --- | --- |
| prepare | 632.0159 | |
| archive | 5.8183 | |
| seed41 | 89,519.5642 | |
| seed42 | 88,362.3835 | |
| seed43 失败尝试 | 20.7579 | **必须保留**，不得从表里删 |
| 小计 | 178,540.5399 | = **49.594594 GPU-h** |

seed43 完成后**必须**用 `stage_ledger.jsonl` 的实测秒数替换估计值，不得把 estimate 写成实测。

### 4c. campaign §8 进度块 + 4d. campaign-state

`current_round_goal` 仍指向本轮直到登记完成；登记后按任务 4 切到
`docs/goals/s3-urban-static-covariates.md`。
`used_gpu_h` / `remaining_gpu_h` 与本轮账本行同步。

**⚠ 4d 有一个必做的前置，已在临时 master 副本上实测（`FAIL C-04`）**：
`check_campaign_state.py` 要求**上一轮** brief 的最后那个「下一动作」里**出现**
master plan 的 `current_node`。把 `previous_round_goal` 指向本轮的 brief、并把
`current_round_goal` 切到静态协变量轮之后，若本轮 brief 的尾部只写
「等 receipt → 清单第 0–6 步」，校验器会报：

```
FAIL C-04 s3-matched-exposure-reeval.md's next action does not name the master plan's current node S3
```

该措辞**已于 2026-10-28 修正**（`docs/goals/s3-matched-exposure-reeval.md` 末尾补上 `S3`），
重测后不再触发。**登记时若改动了那一段尾部文字，必须重新确认它仍含 `current_node`。**

同时注册器还会报 `FAIL C-06 previous round evidence ... is not registered`——这条是**预期的**，
在 4a 把记录写进索引后自动消失（不要在 4a 之前试图"修"它）。

## 5. 收尾验证（四项都要绿）

```bash
.venv/bin/python tools/r7_round.py round-guard
.venv/bin/python tools/r7_round.py verify
```

预期：`round-guard` → `passed: true`；`verify` → 三校验器 exit_code 均 0，
证据索引 `PASS ... N+1 records`。

## 6. 提交与推送

```bash
.venv/bin/python tools/r7_round.py commit-register
.venv/bin/python tools/r7_round.py push-round
```

注意：`commit-register` 会暂存 `register_paths` 全部 5 条，**每条都必须存在**。

---

## 失败恢复（只有在 receipt 缺失或 worker 已死时才用）

先确认没有任何存活 worker（`stage-plan` 的 `already_running` 必须是 `false`）：

```bash
.venv/bin/python tools/r7_round.py stage-plan seed 43
.venv/bin/python tools/r7_round.py orphans
```

**只有在 `already_running: false` 且 `orphans` 无 seed43 时**才补跑，且**必须**在重放内：

```bash
.venv/bin/python tools/r7_archived_replay.py run -- \
    .venv/bin/python tools/r7_round.py stage seed 43
```

**禁止**对任何非本实验进程发信号（决策 0026/R-054）；**禁止**为了让它跑完而改冻结判据。
