# R4（位置保留重估）交接分析报告 —— 暂停点、已完成资产、未完成项与重开条件

- **日期**：2026-11-03
- **分支**：`r7/weather-reasoning`，HEAD `b3619ffa7c9db83e9519f2671ddb02d4d0c8a268`（与 origin 同步，工作树干净）
- **状态**：**用户要求全部暂停**。R4 的 CPU 前置（协议冻结、prepare、archive）**已完成**；
  训练在 **seed 41 控制臂跑到 1,440/16,071 updates 时被人工终止**；**无读数、无科学结论、四件套未登记**。
- **本报告不含任何科学判定**。所有数字都是工程/运行事实，`scientific_claim: false`。

---

## 1. 一句话结论

R4 的**设计与全部 CPU 前置已经就绪并可复算**（五份 ADR、65×65 侧车、`position_scale` 字段、
父钉重绑、冻结协议 `ee79b066…`、归档 `b3619ff`），但**训练被主动停在 8.96% 处**，
因此本轮**没有任何可用于判定 (a)–(d) 的读数**。重开的成本是**重跑 6 条臂约 75–77 GPU-h**，
不是「接着跑」——训练器明确**不支持断点续跑**。

---

## 2. 暂停时到底发生了什么（可核查）

| 项 | 实测 |
| --- | --- |
| 终止对象 | worker pid 2513946 + 其 wrapper 2513941（`kill -TERM`，只针对本实验进程） |
| 终止时进度 | `update_*.pt` 共 72 个 ⇒ **1,440 / 16,071 updates**（8.96%） |
| 终止时 `elapsed_seconds` | **4,220.6 s = 1.172 h** |
| 已烧 GPU-h | **1.156–1.172 GPU-h**（单卡共驻，GPU0 峰值 5,226 MiB） |
| 终止后 `r7_round.py orphans` | `{"running": []}` |
| 终止后 `nvidia-smi` | GPU0 38 MiB / GPU1 17 MiB（均已释放） |
| 残留进程 | `pgrep` 无 `study_r7_s3_position_retention*` 或 `r7_round.py stage*` |
| `stage_ledger.jsonl` | 仅 `prepare success` / `archive success` 两行；**无 seed 行**（因为 seed 未完成） |

**这段损失不可回收**：`training/r7_long_rollout_runner.py` 的 `LIMITATIONS` 明写
`"A new optimizer imports verified model weights only; no failed-attempt resume is supported."`，
`report` 里也固定 `"resume_permitted": False`。所以**没有「从 update 1440 接着跑」这条路**。

**但已产出的部分训练痕迹仍在**（`seed41/training/control/`，4.1 GB）：
`fine_tune_started.json`（含完整冻结 contract，`signature 549dcb11…`）、73 行 `progress.jsonl`、
72 个 `update_*.pt`。这些**不是读数**，只可作为「contract 确实按冻结值构造」的旁证：
其中 `typed_evidence_mode=typed_routing`、`position_scale=8000.0`、
`typed_evidence_identity=a61d22aa…`、父 `parent_checkpoint_sha256=e81e4904…`（seed41 钉值）。

训练迹线本身（**仅供工程观察，不得引用为技能**）：loss 首行 1.1091 → 末行 0.9313，
首 20 行均值 0.9312、末 20 行均值 0.8824，中位 0.8803；`epoch` 恒为 0（正确，
一 epoch = 16,071 updates，此处才 9%）。

---

## 3. 已完成且可复用的资产（这是重开的价值所在）

### 3.1 决策记录（全部 `accepted`，已入 `docs/decisions/README.md` 索引）

| ADR | 作用 | 关键实测 |
| --- | --- | --- |
| **0079** | 否证「`position_scale` 是学习幅度旋钮」 | `position_query` 初始为**精确 0**；1e6× 梯度缩放给出同一轨迹（`max‖Δq‖=1.26e-06`，相对 2.8e-6）——AdamW 约掉整体缩放；H-B 训练后 `\|q\|` 落在随机游走期望的 55–92% |
| **0080** | R4 执行设计：**路线②**（两臂都从零起点新初始化）+ **必须为 65×65 重算侧车** | H-B 的 payload 是 store 特定的：错配解码风速偏 14–31%、`v500` **符号翻转**；匹配时往返 7.1e-15 |
| **0081** | 冻结 `position_scale = 8000.0` | 0078 的 `[1e3,2e3]` 在真实 val 帧上给出熵 **0.9977–0.9992**（与均匀不可区分）⇒ 不可用；8000 在 8/24 帧、3 个 init seed、全部实测 `\|q\|` 上都在 `[0.90,0.99]` 内（worst 0.98890/0.92422）；并记录一个方法论坑：不固定种子构造 router 会让两次运行结论不同 |
| **0082** | **改写了桥 (e)④**：原文「零 query 聚合**逐位等于** `value.mean(dim=1)`」**数学上不可达** | `_position_attention` 先乘 `1/P` 再求和，`mean(dim=1)` 先求和再除 `P`；实测最坏差 **2.98e-08**。驱动原写作 `if difference != 0.0: raise` 且每 seed 训练前都跑 ⇒ **会让第一个 seed 立即中止整轮**。改为「权重逐位等于 `1/P`（精确）」+「聚合差 ≤ `2·P·eps·max\|v\|`（实测占用 2.4e-05）」 |
| **0083** | `model/` 改动后的父钉重绑（tendency / SAM） | 修订 `a38be16c…` → **`59340710…`**；三 seed 155 张量 `state_digest` **逐位相同**（`7c416ccb…`/`857d8804…`/`61fdac58…`），`MIGRATED_PARENT_STATE_DIGEST` 一个字节没动 |

> 0082 与 0083 是**在冻结前**发现并修掉的（本轮当时还没冻结、还没开跑），不是事后放宽判据。

### 3.2 数据与模型侧资产

| 资产 | 路径 | 身份 |
| --- | --- | --- |
| 65×65 typed-evidence 侧车 | `outputs/r7_s3_typed_evidence_fy65_20261103_attempt01/` | `typed_evidence_identity a61d22aaa45b376970ff50479517aa816af5a56e83e9dd4258f36b9df2d81a81`、`data_identity 81307008…`（= FY train 清单身份）、`BUILD_COMPLETE` |
| `position_scale` 常数探针 | `outputs/r7_s3_position_scale_probe_20261103_attempt01/scale_probe.json` | SHA256 `e34ccedce26d41772afcb51cfce4f0d3cf0b2c6e2762bb35dcea0c0442f5df68` |
| R4 父迁移 | `outputs/r7_s3_position_retention_parent_migrated_20261103_attempt01/` | receipt `393766ee…`、修订 `59340710…`、三 seed 文件 SHA `e81e4904…`/`7e2cd4cc…`/`ee33f690…` |
| `model/` 改动 | `model/typed_evidence_r7.py`、`model/process_forecast_r7.py` | 新增 `typed_evidence.position_scale`（默认 `1.0`，**实测逐位不变**、**零新参数**）；`model_code_sha256 = 59340710…` |

### 3.3 代码与协议（已提交并 CI 绿）

| 项 | 值 |
| --- | --- |
| 驱动 | `scripts/study_r7_s3_position_retention_65x65.py`（473 行） |
| 协议层 | `scripts/r7_s3_position_retention_protocol.py`（311 行） |
| support | `scripts/r7_s3_position_retention_support.py`（147 行） |
| 父钉 | `scripts/r7_s3_position_retention_parent.py`（129 行） |
| 测试 | `tests/test_r7_s3_position_retention_65x65.py`（548）+ `..._router.py`（166） |
| 归档构建器 | `tools/build_round_code_archive.py`（117）+ 其测试（93） |
| brief | `docs/goals/s3-position-retention-reeval.md`（objective 1,733 字符，`check_goal_brief` 0 失败） |
| 本地全量测试 | **5,408 passed / 27 skipped**（在干净 worktree `cfae620` 上跑，20m13s） |
| CI | `R7 CPU CI` 在 `cfae620` **success**（commit `b3619ff` 只改了工具与指针，未再单独观察 CI） |

**冻结的三份协议身份**（`outputs/r7_s3_position_retention_65x65_20261103_attempt02/`）：

```
preparation_protocol_sha256  bc00c3b442be2a79237e80b6280dc50a195eeff19e29403b26910893db2f7d1b
protocol.json (prepared)     ee79b066c6738b45a1b3e5ca6716fc009c84b3e36b7e5307a7f5c44b674c8274
code_commit                  b3619ffa7c9db83e9519f2671ddb02d4d0c8a268
code_zip_sha256              9f914abda6ade2c01ff2d103448c4f6631b4ff0537bc0beb6e1879b1641efa8d
model_code_sha256            59340710623ca24c7ee59ebb5711f52193ebfbaa72ed4532b41bdd987f5d9324
training_code_sha256         96a68d6eb29767f2b54df75e381688875d98dc8ebfdfc88f63b2fe39355975f2
execution_files_sha256       92 项（decision 0067 的具名 import 闭包）
archive_receipt.json         c5994995acfbab69a306e8e75e4ebc37aca81a6954df505cf531c94124189469
```

**已冻结的单因子结构**（`prepared_protocol.json` 实测）：
两臂 `updates=16071 batch=2 steps=4 lr=2e-05 warmup=10`，`position_scale=8000.0`，
typed-evidence payload **两臂逐字相同**，两个 spec **只在 `typed_evidence_mode` 上不同**
（`typed_routing` vs `typed_spatial`）。评分域 `full`（4,225 点），`test_read=false`、`scientific_claim=false`。

---

## 4. 未完成项（重开窗口必须做的全部事情）

按冻结顺序，**一个都没做**：

1. **seed 41**：控制臂 8.96% 后终止；候选臂**从未启动**。
2. **seeds 42 / 43**：从未启动。
3. **`stage reading`**：未运行 ⇒ **无 `readings.json`**、无出口 (a)–(d)、无桥 (e) 判定。
4. **四件套登记**：无结果页、无 `docs/R7_EVIDENCE_INDEX.jsonl` 记录、无账本行、`campaign-state` 未更新。
5. `docs/R7_CANDIDATE_BRIEF.md` 未重算（本轮没有新记录，所以仍与索引一致）。

**另外两处需要新窗口知道的历史遗留**：

- `outputs/r7_s3_position_retention_65x65_20261103_attempt01/`：**作废**。它在 `cfae620` 下冻结，
  但那个 revision 的 `code.zip` 少了 zip 注释（见 §5），archive 阶段拒绝；`code_zip_sha256` 与现行归档不符。
  **它只有 prep/prepared/protocol/日志，零运行产物，已弃用不删**（审计留痕）。
  重开时用 `attempt02` 或更新的 attemptNN。
- 曾经误把 R2 协议冻结进 `outputs/r7_s3_anchor_feedback_65x65_20261101_attempt02/`
  （`campaign-state.current_round_goal` 当时仍指 R2）。**该目录只有一份 prep protocol、零运行产物，已删除**；
  R2 真正登记的 `attempt01` **未被触碰**。指针已改为 R4 brief 并通过 `check_campaign_state`（failures=0）。

---

## 5. 这次踩到的三个真实缺陷（对新窗口最有价值的部分）

这三个都不是「配置没调好」，而是**会让运行失败或让判据失效**的硬问题。新窗口若重写提示词，应把它们作为**已解决的先验**写进去，避免重犯。

### 5.1 桥 (e)④ 的判据数学上不可满足（ADR 0082）

- **症状**：驱动在**每个 seed 训练前**断言零 query 聚合与 `value.mean(dim=1)` 的差为 **0**，否则 `raise`。
- **根因**：两条路径是不同的 float32 程序（乘 `1/P` 再和 vs 先和再除 `P`），实测最坏 **2.98e-08**。
- **后果**：**第一个 seed 立刻中止整轮**，0 产出。
- **处置**：拆成「权重逐位等于 `1/P`（精确，实测 `softmax(zeros)` 与 `full(1/P)` 位相同）」+
  「聚合差 ≤ `2·P·eps·max|v|`（实测占用 2.4e-05）」，两者逐 seed 断言；`zero_query_max_abs`
  从判据降为**记录值**，receipt 同时存 `bound` 与 `ratio`。
- **教训**：**把「相等」写进判据前，先测它到底能不能相等**。这条花费极小（一次 CPU 测量），
  但如果不测，代价是一轮 75 GPU-h 白跑。

### 5.2 code.zip 缺 commit 注释 → archive 阶段拒绝启动

- **症状**：`stage archive` 报 `RuntimeError: archive commit differs from freeze`。
- **根因**：每轮的 `_archive` 把 **zip 注释**与冻结的 `code_commit` 比对；R2 的 zip 有注释
  （`c48c57f…`），我新写的构建器写的是**空注释**。
- **处置**：`tools/build_round_code_archive.py` 写入 `target.comment = resolved`，并加一条测试钉住；
  重建归档后 archive 通过。
- **教训**：**「复现了文件清单」不等于「复现了格式」**。我当时只比了 namelist，漏了 comment。

### 5.3 `campaign-state.current_round_goal` 指错 → 冻结进错误的轮次

- **症状**：`r7_round.py freeze` 把 **R2 的**协议冻结进 `..._anchor_feedback..._attempt02/`。
- **根因**：`current_round_goal` 仍指 `s3-anchor-anomaly-feedback-reeval.md`（R2），
  而所有 stage 子命令都从它解析当前轮——**argv 不参与**。
- **处置**：改指 R4 brief、`previous_round_goal` 接住 R2；删掉那个只有一份 prep protocol 的目录；重新 freeze。
- **教训**：**跑任何 stage 之前先 `r7_round.py state` 看 `current_round_goal`**，
  否则你会以为自己冻结的是 R4。

### 5.4 附带：CI 抓到我本地没跑全量的两个红灯

- R-021 的 `measured` 标记过期（我新增两个 >400 行文件：548 与 473）⇒ 已用 checker 复算后更新为 135；
  顺手把 9 参数的测试辅助函数改成 3 个 dict 参数，R-023 保持 47。
- `model/` 移动后 **tendency/SAM 两条「钉当前修订」守卫**转红 ⇒ 按 0075 先例做 0083 重绑。
- **教训**：**提交前跑全量，或至少跑 `tests/test_r7_s3_*_protocol.py` 这一族**。
  我当时只跑了本轮相关文件，两条守卫是在 CI 里才暴露的。

---

## 6. 重开需要知道的操作事实

| 项 | 值 |
| --- | --- |
| GPU | 2× RTX 3090，各 24,576 MiB；**默认共驻**，每卡本实验峰值实测 5,226 MiB |
| 冻结显存门槛 | `estimated_peak_bytes 5,596,409,024`（5,337 MiB）+ margin 512 MiB + 邻居预留 6 GiB = **11,993 MiB/卡** |
| 实测速度 | **2.90 s/update**（batch 2）⇒ 单臂 16,071 updates ≈ **12.9 h** |
| 单 seed 预算 | `per_seed_seconds 129,600`（36 h）= 两臂各 ~13 h，**够用**（实测 worker deadline 留 34.8 h） |
| 轮次预算 | `planned 259,200 s`（72 h）/ `hard 518,400 s`（144 h）；**6 臂约 77 GPU-h ⇒ 预计软超约 5 h**，按冻结规则继续跑并记 `soft_overrun`，只有硬上限截断 |
| 对照：R2 实测 | 训练 70.4065 + 评分 4.7009 + reading = **77.4371 GPU-h**，墙钟 ~51.1 h（双卡共驻） |
| `stage-seeds` 行为 | 在冻结双卡列表上**轮转**（42 首选 GPU1、43 首选 GPU0），`max_workers=len(uuids)`，**每卡最多一个 worker、不共享卡**；已有 `seed{N}_receipt.json` 的 seed 会跳过；headroom-blocked 只读等待，**不向邻居进程发信号** |
| 训练**不可续跑** | 见 §2；重跑必须从父 checkpoint 重新开始 |
| 磁盘 | `attempt02/seed41` 单臂 4.1 GB；6 臂 + 评估预计数十 GB，重开前先看 `df -h` |

**重开的最小命令序列**（在 `outputs/` 已存在 attempt02 的情况下）：

```bash
cd /data/esw/UrbanPiDiT_R2
git rev-parse HEAD                      # 应为 b3619ff 或其后代
.venv/bin/python tools/r7_round.py state              # 先确认 current_round_goal 指 R4 brief
.venv/bin/python tools/r7_round.py stage seed 41      # 或 stage-seeds 一次跑完剩下三个
.venv/bin/python tools/r7_round.py stage reading
```

> 注意：`attempt02` 的 `protocol.json` 绑的是 `code_commit b3619ff` 与
> `code_zip_sha256 9f914abd…`。**若新窗口的 HEAD 不同，`validate_protocol` 会因
> `execution_files_sha256` 不符而拒绝**——那时应**重新 freeze 一个 attemptNN**，
> 而不是改 attempt02 的协议。

---

## 7. 如果新窗口要重写提示词，建议显式写进去的三件事

1. **顺序约束**：`round-guard` → 提交 → **等 CI 绿** → 重建 code zip（**带 commit 注释**）→ `freeze` →
   `stage prepare/archive` → seeds → `stage reading` → 四件套登记。**每一步的产物 digest 都要落盘可查**。
2. **判据纪律**：出口 (a)–(d) 与桥 (e) 已在 `docs/goals/s3-position-retention-reeval.md` §3 冻结；
   **冻结后不放宽、不扫 `scale`、不加 seed、不换聚合族、不换名重开**；任一假即**永久关闭该支**。
   桥 (e) 已按 0082 修正为「精确权重 + 有界聚合」，**不要**再写成 `zero_query_max_abs == 0`。
3. **两个反直觉但已实测的事实**：`position_scale` 是**前向敏感度**不是学习幅度旋钮（0079）；
   两臂**参数不匹配**（`typed_spatial` 多 192 参数 = +0.13%），这是已声明差异，**不得**声称参数对齐。

---

## 8. 明确未做的事

- 未跑任何 seed 到完成；未产出 `readings.json`；未判定 (a)–(d)；未评估桥 (e)。
- 未修改任何已冻结阈值、出口或 ADR 的结论。
- 未触碰 `data/raw|interim|processed`、未新增下载、未重建 store、未合并 main、未 force push、未关闭 issue。
- 未对邻居进程发任何信号；终止只针对本实验的 pid 2513946/2513941。
- 未把 `scientific_claim` 置真；本报告**不是**科学结论。
