# 0070 — 城市静态协变量轮的 workflow 编排重跑：独立 attempt02、预声明出口、指针临时回归

- **状态**：accepted
- **日期**：2026-10-30
- **范围**：只授权**一次**、有界的重跑：用 `CreateWorkflow` 以 `dryRun: false` 运行
  `.zcode/workflow-drafts/r7-urban-static-round.dwf.ts`，产物落**全新排他**
  `outputs/r7_urban_static_covariates_attempt02/`。
  **不改** `attempt01` 的任何字节（它是已登记轮次的证据）、**不改**任何已冻结判据、
  **不动** `data/raw|interim|processed` 与归档快照、**不改** `model/**`、**不重建** store。
- **依据**：`docs/goals/s3-urban-static-covariates.md` 目标明写「用 workflow 编排」；
  决策 0063（数据获取授权）、0064（时间刻度，只改时间不动科学门）、0044（workflow 是有界
  可恢复编排器）、0045（复用白名单）、0067（执行文件必须具名）。

## Context

**① 交付物已经真实存在，但编排路径从未被执行（实测）。**

六相位的数据产物在 `outputs/r7_urban_static_covariates_attempt01/` 全部落地并经核验：
`published/manifest/BUILD_COMPLETE.json` = `{"schema_version": 1, "build_complete": true}`、
`static_identity = 8671ed7c…`、36 个数组形状 (65,65)、`protocol_sha256 = 8cc3c673…`、
独立再派生 12 项最大相对偏差 `5.15e-08`。

但那条路径是**执行者逐子命令直跑**的：

```
.venv/bin/python scripts/r7_urban_static_round.py {verify-endpoints,freeze,download,aggregate,recheck,publish}
```

`CreateWorkflow` 只用默认 `dryRun=true` 编译过两次（`dwfrun-1f67679b-dd53-43a2-9fa8-14b468dbfb1c`、
`dwfrun-87bbd520-01a9-4ada-894f-122db5230455`）。**`dryRun: false` 从未运行**，
所以「本轮由一个有界、可恢复的编排器驱动」这一工程属性**没有证据**。

**② 此刻按字面补跑会回退已登记的轮次（实测）。**

当前 `campaign-state`（`docs/goals/main-model-climatology-campaign.md`）：
`current_round_goal = docs/goals/s3-urban-static-wiring.md`、
`previous_round_goal = docs/goals/s3-urban-static-covariates.md`。接线轮（决策 0069）已登记。

**③ 指针回归会触发一条可机械判定的约束失败（实测，非推测）。**

`tools/check_campaign_state.py:216-266` 的 C-04 要求：若 `previous_round_goal` 存在，
其「下一动作」末尾必须**含 `current_node`**。把 `current_round_goal` 改回本长文后，
`previous_round_goal` 就必须是接线轮长文 `docs/goals/s3-urban-static-wiring.md`，
而该文件「下一动作」尾部（304 字符，实测）**不含 `S3`**：

```
下一动作（可执行的第一条命令）**：
  `.venv/bin/python tools/r7_round.py state`
  —— 核对指针；随后是 **R2–R5 的父迁移解阻塞**（见主计划 §8 2026-10-29 块）
```

⇒ 只改指针会让 `check_campaign_state.py` 报 `C-04 ... next action does not name the
master plan's current node S3`。因此本 ADR 必须同时授权**对 previous 长文的下一动作做
最小补写**（只补节点名，不改其结论、不改任何判据）。

## Decision

1. **指针临时回归，范围只到本轮结束。** 把 `current_round_goal` 指向
   `docs/goals/s3-urban-static-covariates.md`，`previous_round_goal` 指向接线轮长文，
   `previous_round_evidence` 指向 `docs/R7_STATIC_COVARIATES_WIRING.md`（其账本行已登记，
   `evidence_sha256 = d0171038…`）。
2. **接线轮长文的「下一动作」补一个节点名 `（节点 S3）`**，使 C-04 在回归期间可满足。
   这是**指针记账的适配**，不是对接线轮结论的改写；该轮的所有读数与出口一字不动。
3. **重跑必须落全新排他目录。** 驱动的 `new_attempt()` 自动取 `attempt{max+1}` ⇒
   `attempt02`。`attempt01` 的字节不得改动；重跑产物与 attempt01 的差异**本身就是读数**
   （可复现性等级的证据）。
4. **预声明出口**（先冻结后运行，运行中不放宽）：
   - **E1 六相位全绿**：`verify-endpoints`/`freeze`/`download`/`aggregate`/`recheck`/`publish`
     六个 stage 退出码全 0，且 `published/manifest/BUILD_COMPLETE.json` 的
     `build_complete` 为 `true`。
   - **E2 层可达性不劣化**：可达层数 ≥ 13（attempt01 实测值）；LandScan 仍须
     `failed-no-fallback`，地形仍不在具名清单内且**不得**被合成顶替（R-008）。
   - **E3 身份可复现**：`static_identity` 与 attempt01 的 `8671ed7c…` 的关系必须**如实记录**。
     若因 `published_utc` 等时间戳进入 `static_identity` 而不同，则记录差异逐项，并声明
     「输入与聚合规范相同、仅时间戳不同」，不得声称逐位相同。
   - **E4 再派生核对**：`recheck` 的最大相对偏差 ≤ `1e-05`（attempt01 冻结的容差）。
   - **E5 预算不超**：网络 ≤ 40 GiB、磁盘余量 ≥ 60 GiB、墙钟 ≤ `hard_cap_seconds`
     86,400 s；超过 `planned_seconds` 21,600 s 记 soft overrun 但继续（决策 0064 第 2 条）。
   - **E6 登记不破坏既有**：`docs/R7_EVIDENCE_INDEX.jsonl` 的既有 94 条 `record_id` **不得重复**
     （重复会让 `verify` 直接失败）；新增记录必须是**新 `record_id`**。
5. **停止条件**（任一触发即停，如实记，不重试到偶然过）：某 stage 非 0 且非重试可解；
   磁盘余量跌破 60 GiB；墙钟逼近 hard cap；`check_campaign_state.py` 或 `commit-guard` 失败
   且非本 ADR 覆盖的原因；发现需要改动 `data/raw|interim|processed`、归档或 `model/**`。
6. **本轮 0 GPU-h**；只推 `r7/weather-reasoning`（禁 main/force/merge/关 issue）。

## Consequences

**变容易的：** 「本轮由可恢复编排器驱动」获得实测证据；`dryRun: false` 分支
（分支断言 → commit-guard → 提交 → 六 stage → verify → 登记提交 → 推送）首次被执行。

**变难 / 代价（如实列出）：**
- **重下 14.5 GiB**、多占约 20 GiB 磁盘。`/data` 余量实测 570 GiB，够用但非无限。
- **`static_identity` 可能不同**（若含 `published_utc`）。这会引入「同名身份两个值」的
  解释成本；E3 要求如实记录差异，**不得**为了好看而声称一致。
- **指针回归是临时状态**：期间若另一条研究线（如 R2–R5 父迁移）要开轮，会解析到本长文。
  故本 ADR 要求重跑**有界、尽快结束**，结束后把指针恢复为接线轮。
- **接线轮长文被补一个节点名**：改变了已登记轮的文档字节。这是 C-04 的机械要求所致，
  已在此明写，且不改其任何读数/出口/结论。
- **attempt02 不产生新的科学信息**：它验证的是编排路径与可复现性，不是天气真值提升。
  `scientific_claim: false` 不变。

## 与 0063/0064/0069 的关系

- **不取代** 0063（数据获取授权）与 0064（时间刻度）：本 ADR 沿用其范围与预算。
- **与 0069 无冲突**：0069 是模型侧接线（已登记）；本 ADR 只补数据轮的编排证据，
  **不碰** `model/**`，也不重跑父迁移。
