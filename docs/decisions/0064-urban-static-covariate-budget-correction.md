# 0064 — 城市静态协变量轮的预算更正：按首次实测核对墙钟刻度（planned 21,600 / hard 86,400 维持）

- **状态**：accepted
- **日期**：2026-10-27
- **范围**：`docs/goals/s3-urban-static-covariates.md` 的网络 / 磁盘 / 墙钟预算。
  沿用 0053/0055/0060 的先例：**只改时间**，不动任何科学门；**不在运行中改硬限**。

## Context

本轮冻结的 `protocol.json`（`outputs/r7_urban_static_covariates_attempt01/protocol.json`，
digest `8cc3c673…`）里的预算刻度是**名义值**：

| 字段 | 冻结值 |
| --- | --- |
| `planned_seconds` | 21,600（6 h） |
| `hard_cap_seconds` | 86,400（24 h） |
| `network_bytes_cap` | 40 GiB |
| `disk_headroom_bytes` | 60 GiB |
| `single_tile_bytes_cap` | 512 MiB |

首次实测（attempt01，2026-10-27，手工逐子命令执行）：

- 整轮 `wall_clock_seconds = 9550`（2 h 39 min）**<** `planned_seconds` 21,600 ⇒ **无 soft overrun**。
- 逐阶段（产物 mtime，UTC）：端点闸门 `04:02:21` → `download_receipt` `05:13:48`（≈70 min）
  → `aggregate_receipt` `06:21:26`（≈68 min）→ `recheck_receipt` `06:31:03`（≈10 min）
  → `published/manifest` `06:31:18`（秒级）。
- 网络：`network_bytes = 14,484,496,452 B`（13.49 GiB）< 40 GiB 上限（用掉 35%）。
- 磁盘：`native/` 20 GiB + `published/` 0.7 MB；`/data` 余量 269 → 242 GiB（消耗 27 GiB），
  最低余量 242 GiB > 60 GiB 门。

**这不是「放宽判据」**：ROI、格界、65×65 / 0.25°、`area_weighted_coarsen`、逐层 tile 清单、
`static_identity` 口径、六相位判据全部不动；本决策只涉及**时间**。

**机制点**：已发布的 attempt01 的 `protocol.json` 及其 digest **不改**——它被 `protocol_sha256`
与 `static_identity` 覆盖，改它会打断身份链。本决策只对**后续运行**（含 workflow 编排的那次）生效。

## Decision

1. **墙钟刻度维持** `PLANNED_SECONDS = 21,600 s`、`HARD_CAP_SECONDS = 86,400 s`。
   实测未触发任何门（9550 s 是 planned 的 44%、hard 的 11%），**没有给出改刻度的依据**；
   名义值相对实测的余量（2.26× / 9.0×）正好留给 workflow 编排的 agent 开销
   （dryRun 实测的「实现 + 复核」段不计入 stage 的 6 h 上限）。
2. **实测基准入档**（本决策 Context）：后续轮次与 workflow 编排运行以 `9550 s` 为对照；
   若某次实测超过 `planned_seconds`，按同一纪律**另立**一条更正决策改刻度。
3. **网络 / 磁盘预算不变**（本决策只改时间）：`network_bytes_cap = 40 GiB`、
   `disk_headroom_bytes = 60 GiB`、`single_tile_bytes_cap = 512 MiB`。
4. **不动科学门**：六相位判据、(a)–(d) 式出口、ROI / 网格 / 聚合规则、`static_identity` 口径全部不变。

## Consequences

**变容易的：** 名义刻度已由首次实测背书（2.26× 余量），workflow 编排运行不必先改预算就能开跑；
「实测 9550 s」成为后续任何超时的对照锚点。

**变难 / 代价（如实列出）：**
- **维持刻度 = 放弃一次收紧机会**：若后续运行长期只用到 6 h 的一小部分，名义值会继续显得偏松；
  这是刻意的——在**没有**超时证据时收紧冻结预算，本身就是对「先冻结」纪律的破坏。
- **实测基准来自手工执行**，不含 workflow 编排的 agent 阶段；编排运行的真实墙钟**尚未实测**，
  因此本决策的对照值是**下界**。若编排运行超 `planned_seconds`，必须另立更正决策，不得就地改硬限。
- **`planned_seconds` / `hard_cap_seconds` 在 `tools/r7_round_support.py` 的复用白名单内**（决策 0045）：
  本决策**未**改这两个值，故 attempt01 的产物与任何以它为准的复用**不受影响**；
  若将来改了，旧 attempt 不可复用。
- **磁盘峰值未进冻结协议**：phase ② 要求的「磁盘峰值与余量门」里，冻结字段只有余量门
  （`disk_headroom_bytes`），峰值只有实测值（本决策 Context 与证据页 §8）。本轮**不**追改已发布的
  `protocol.json`（会断身份链），改由证据页如实标注，并在下一轮把 `estimated_peak_bytes` 写进协议。

**备选方案与否决理由：**
- **按 9550 s 收紧到 planned 14,400 s**：实测虽是 9550 s，但**编排运行的 agent 开销未实测**，
  收紧可能把编排运行挤成 `budget_limited`；且没有超时证据支撑，否决。
- **把 hard 抬到 172,800 s**：没有任何一次实测触门，抬高只会削弱硬上限的保护作用，否决。
- **在运行中改硬限**：被 0055/0060 的纪律明确禁止，否决。
- **改已发布 attempt01 的 `protocol.json` 以补 `estimated_peak_bytes`**：会改变 `protocol_sha256`
  与 `static_identity`、打断身份链；改为在证据页如实标注 + 下一轮协议补字段，否决。
