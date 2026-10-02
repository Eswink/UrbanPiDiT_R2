# 0010 — 实验常设下放与时长契约：新决策 0029 取代 0021

**状态：已批准（2026-10-02）；本轮 D1–D10 治理施工已落盘，最终提交/CI 正在绑定。**
以下 §0–§7 是原执行前计划归档（R-032），保留当时事实；「实际结果（计划归档会话）」记的是原会话。
本轮执行证据与差异见文末「执行记录（2026-10-02）」，不要把旧「未执行」描述当作当前进度。

## §0 授权依据（用户 2026-10-02 于计划模式内以选项确认）

用户三项决定（原话为选项选择，此处如实转写）：

| # | 决定 | 含义 |
| --- | --- | --- |
| 1 | **实验类动作全下放** | 本地 GPU 运行、训练/评估、触发实验 workflow、以及**每次实验的时长与预算分配**，全部由执行者自主决定，不再逐次 `AskUserQuestion` |
| 2 | **不设总上限** | 不再保留「campaign 账本 ≤24 GPU-h」式的总封顶；账本从「闸门」降为「记账」，只在需要付费资源时停下 |
| 3 | **软预算 + 宽松硬上限** | 计划时长为软预算：超出只记录 overrun、不中止；保留一个宽松硬上限防跑飞；同步允许提高 CI `timeout-minutes` |

**仍然逐次授权的项（下放范围之外，用户未下放）**：GPU 租赁与付费资源；多年度 / 新数据下载与数据发布 `--write`；
`main` 合并、force push、破坏性数据操作。

**仍然不可豁免的硬约束**：R-006（先冻结 protocol.json）、R-028（离线实验禁网）、R-054（共驻、不对非本实验进程发信号）、
R-009（不弱化判据、skip/取消/排队不算通过）、R-002/R-004/R-031（原始数据与归档只读）、R-008（禁止合成回退）、
身份校验（`model_code_sha256`、source SHA256、`BUILD_COMPLETE`）、`scientific_claim: false` 与 limitations 如实记录。

**方向把控仍归用户**：节点推进（`current_node`）与目标完成判定只能由用户或独立复核裁定，执行者仍只可提议。

## §1 起点事实（2026-10-02 只读核对）

| 项 | 值 | 来源 |
| --- | --- | --- |
| 工作分支 / HEAD | `r7/weather-reasoning` / `6e25c7e` | `git rev-parse HEAD` |
| 分支上新增提交（相对本会话开始时的 `ed1a034`） | `055ee9f`、`6e25c7e`（M3 失败登记与审计，已提交） | `git log --oneline ed1a034..HEAD` |
| 工作区（**待新窗口先处理的遗留**） | 2 modified（`docs/goals/README.md` 3 行、`docs/plans/README.md` 1 行）+ 本计划文件与 4 份未跟踪长文（`n3-m4-autoregressive-rollout.md`、`n4-m5-confirmation.md`、`v2-issue-closeout.md`、`0009-r7-v2-completion-and-closeout.md`） | `git status --short` |
| 当前决策最大编号 | **0028**（`0028-m3-process-sidecar-and-time-contract.md`）→ 新决策用 **0029** | `ls docs/decisions/` |
| 当前计划最大编号 | **0009** → 本文件为 **0010** | `ls docs/plans/` |
| Campaign 状态块 | `current_node=N2a`、`status=budget_limited`、`cap_gpu_h=24.0`、`used_gpu_h=4.7018`、`remaining_gpu_h=19.2982` | `docs/goals/main-model-v2-campaign.md:144` |
| M3 终态 | failed/budget_limited/partial，6 train 各 400、7/30 eval；第八项 seed41/input_aux/+24h 父 deadline timeout；整轮 1805.1s 超 30 min 文字 5.1s，如实记为「预算门未完全兑现」 | `docs/goals/main-model-v2-campaign.md` §8 尾 |
| 结算余额 | 精确 4.701714415227032 / 余 19.29828558477297 | 同上 |

关键静态事实（后续改动会碰到）：
- `tools/check_conventions.py` 的 **R-033** 要求决策三段齐全（`## Context` / `## Decision` / `## Consequences`）、
  状态行 `- **状态**：`、Consequences 含「变难/代价」；**R-036** 要求 `superseded by NNNN` 的 NNNN 存在。
- `tools/check_campaign_state.py` 的 **C-02** 用 `cap_gpu_h`/`used_gpu_h`/`remaining_gpu_h` 做算术对表（容差 1e-3），
  C-01 只要求「必需键存在」——**新增键不会转红，删键会转红**。
- 冻结协议常量被测试钉住：`tests/test_r7_frozen_z_protocol.py:122` 断言 `global_deadline_seconds == 1800.0`；
  `tests/test_r7_m3_driver.py:79,91` 钉住 `max_round_seconds == 1800`——**本轮不许动它们**。
- 承载实验的 18 条 workflow 的 `timeout-minutes` 为 10–30 分钟；超时后 GitHub 取消该 job（终态 `cancelled`，
  连 `always()` 收尾都可能跑不到）。

## §2 目标（一句话）

把「实验要逐次问用户」改成**常设下放 + 记录在案**：新写决策 0029 取代 0021，同步 AGENTS.md 硬约束、
两份规则文件、两份 SKILL、campaign 主计划与 CHANGELOG；不动任何检查器、规则编号、冻结常量与 workflow 数值。

## §3 执行清单（D1–D10，按序；每项给出精确锚点与目标文本要点）

### D1 新建决策 `docs/decisions/0029-standing-experiment-delegation.md`

按 ADR 模板（`.agents/skills/decision-record/SKILL.md` §模板）写，四段齐全：

- 头部：`- **日期**：2026-10-02`、`- **状态**：accepted`。
- `## Context`：复述 0021 把通道固定为「逐次问」后、用户 2026-10-02 决定把实验与时长整体下放；
  说明本轮要取代 0021 的**哪一条**（逐次询问），保留**哪一条**（授权不豁免 R-006/R-028 与诚实性判据）；
  如实写出 0021 的备选方案「代理自行决定有界实验曾被否决」以及用户现在推翻了该否决理由。
- `## Decision`，五条：
  1. **常设下放**：实验类动作（本地 GPU 运行、训练/评估、触发实验 workflow）与**每次实验的时长与预算分配**，
     由执行者在 R-006/R-028/R-054 与项目边界内自主决定，不再逐次 `AskUserQuestion`；
     「已决定预算端点写下来之后再开始」的纪律不变（`bounded-study-run` §前置条件）。
  2. **保留逐次授权项**：GPU 租赁与付费资源、多年度/新数据下载与 `--write`、`main` 合并、force push、
     破坏性数据操作，仍按原边界逐次取得；本决策不改变这些边界。
  3. **时长契约**：每次实验在协议中声明 `planned_seconds`（软）与 `hard_cap_seconds`（宽松，默认约 2× 计划，
     在该轮长文写死）；超软预算**不中止**，运行记录记 `soft_overrun_seconds`；仅硬上限截断，终态记
     `budget_limited`/`failed`、全额记账、不算科学通过。承载实验的 workflow `timeout-minutes` **必须高于**该实验硬上限
     （先内部截断、留证据）；无新增实验时**不批量改**现有数值。
  4. **账本降为记账**：campaign 账本不再设总上限；`cap_gpu_h`/`used_gpu_h`/`remaining_gpu_h` 保留为会计字段
     （C-02 算术不破坏），state 块可加 `budget_mode` 键；「预算用尽」不再单独作为停止条件，
     其余停止条件（需新判据 / 不可分辨 / 需付费资源或新数据或合并 main）不变。
  5. **不可豁免清单**：R-006、R-028、R-054、R-009、身份校验、`scientific_claim: false` 与 limitations 照旧；
     `skip`/`cancelled`/`queued`/`partial`/`failed` 不算通过；节点推进与目标完成判定仍只可由用户/独立复核裁定。
     另明写：**不回溯**——冻结协议常量、历史「≤30 min」文字、M3 的 `budget_limited` 终态均不改；本决策只对之后的实验生效。
- `## Consequences`：**变容易的**（不再有询问往返与授权欠账；长实验可一次跑到自然终点）；
  **变难 / 代价（如实列出）**（无总上限只剩单次硬上限约束，GPU-h 消耗可能高于原 24 上限节奏；执行者需要自己
  在每轮把 `planned/hard` 写进协议，写不清就没有可比对的软预算；用户不再逐次看到实验内容，审计只能靠长文与账本）；
  **备选方案与否决理由**（保留逐次问——用户明确否决；完全去掉硬上限——卡死会无限占资源，不采用；
  改冻结常量实现新语义——会破坏归档重放，不采用）。

### D2 `docs/decisions/0021-experiment-authorization-channel.md`

- 状态行改为 `- **状态**：superseded by 0029`（**不删正文、不删文件**；R-036 会核 0029 存在）。
- 可在文件末尾加一行指针：`> 2026-10-02：本决策的「逐次询问」通道由决策 0029 取代（实验与时长已常设下放）；见 docs/plans/0010-*.md`。

### D3 `docs/decisions/README.md`

- 0021 行状态列：`accepted` → `superseded by 0029`。
- 追加：`| [0029](0029-standing-experiment-delegation.md) | 实验常设下放与时长契约（取代 0021 的逐次询问通道） | accepted | 2026-10-02 |`。

### D4 `AGENTS.md` 硬约束段（当前 :18–:21）

- `:18` 的禁列改写为：`- **禁止**在无显式授权时执行：GPU 租赁与付费资源、多年度或新数据下载、\`main\` 合并、force push、破坏性数据操作。（本地实验的规模与时长已按决策 0029 常设下放，见下一行。）`
- `:19–:21` 的三行**替换**为：
  ```
  - 实验类授权按**决策 0029**：本地 GPU 运行、训练/评估、触发实验 workflow 与**每次实验的时长与预算**
    由执行者自主决定，不再逐次询问；协议仍须先冻结（R-006）、离线实验仍须禁网（R-028）、GPU 仍须共驻（R-054）。
    时长契约＝**软预算 + 宽松硬上限**：超软预算继续跑并记 overrun，仅硬上限截断；截断终态不算科学通过。
  ```
- `:22` 共驻行不动（把其中「决策 0026」保留），`:23` 诚实性行不动。

### D5 `docs/rules/ci-and-verification.md`

- R-029（:51–:61）：复核触发补一句——「承载有界实验的 workflow 的 `timeout-minutes` 必须**高于**该实验的
  `hard_cap_seconds`（决策 0029）；提高数值而不是删除时限的纪律不变」。
- R-030（:63–:78）：陈述补一句——「决策 0029 起，模块内截止检查的语义是**宽松硬上限**：低于计划时的软预算不中止，
  只在硬上限截断并记 `soft_overrun_seconds`；冻结协议常量（如 `global_deadline_seconds=1800.0`）不回溯修改」。
  例外清单与 Q-010 指针不动。
- 触发授权段（:155–:157）：改为引用决策 0029（「标签只表达触发意图、不构成授权；`skip`/`cancelled`/`queued` 不算通过」
  保留；删去「执行前用 `AskUserQuestion` 提问」）。
- 新增小节（建议放在 :157 之后）：**「实验时长契约（决策 0029）」**——`planned_seconds` 软 / `hard_cap_seconds` 硬 /
  `soft_overrun_seconds` 记录 / 超时后「继续等待、记录在案」的语义 / 不回溯冻结常量。

### D6 `docs/rules/gpu-resources.md`

- `:3` 与 `:28` 的「实验授权仍按决策 0021」→「按决策 0026/0029（本机共驻不变；实验与时长已常设下放）」；其余不动。

### D7 `.agents/skills/bounded-study-run/SKILL.md`

- 「## 授权」段（:18–:22）：改为常设下放表述 + 保留项清单 + 不豁免清单（照 D1 第 1/2/5 条压缩）。
- 前置条件（:28）「已决定预算端点……写下来之后再开始」：补 `planned_seconds` 与 `hard_cap_seconds` 两个端点。
- 步骤 3（:45–47）：改为「循环内检查**硬上限**；超过软预算不中止，记 `soft_overrun_seconds`；只允许在硬上限截断」。
- 「常见失败」(:79)「超时才发现预算错」：改为「软预算写成硬上限：计划没写软/硬两个数字，事后无法区分 overrun 与截断」。

### D8 `.agents/skills/goal-loop/SKILL.md`

- 前置条件 2（:30–33）：把「第一阶段 4 GPU-h、单次 ≤30 min / 第二阶段 ≤24 GPU-h」等旧上限表述改为
  「按决策 0029：无总上限，逐轮自设 `planned`/`hard` 并写进长文；旧数字仅作为历史引用保留在各自文件里」。
- 「常见失败」中「预算自设超过既有授权」（:105）：改为「把历史上限或已被取代的旧额度当成现行闸门」。
- 其余（§步骤、完成判据）不动。

### D9 `docs/goals/main-model-v2-campaign.md`

- §3 判据来源（:91–:94）：`docs/decisions/0021-...md` 指针改为 `0029-...md`（或并引 0021 注明已取代）。
- §5（:106–:112）：批次授权行改写——「无总上限（决策 0029）；账本为记账；单次实验以 `planned`/`hard` 两段控制」；
  停止条件删去「预算用尽或账本不足」，保留其余三条。
- §7（:122–:140）：标题与说明改为「账本（记事，非闸门）」；表内数值与证据指针**不动**；说明段「把索引补成全量账本」照留。
- §8 state 块（:144）：`cap_gpu_h/used_gpu_h/remaining_gpu_h` **保留原值**，追加 `"budget_mode": "accounting-only"`（C-01 只查必需键存在，新增键安全）。
- §8 追加一条本轮进度尾：记录本计划产出、决策 0029 的落点与「执行与新规则下的实验调度留待新窗口」；
  **不改** `current_node=N2a`、`status=budget_limited` 与上一轮结论。

### D10 `docs/rules/CHANGELOG.md`（顶部追加，照 2026-09-29 决策 0021 条目的范式）

标题：`## 2026-10-02 — 实验常设下放与时长契约（决策 0029，取代 0021）`。正文写明：
范围（新决策、AGENTS 两行、ci-and-verification 两处 + 新小节、gpu-resources 指针、两份 SKILL、campaign §3/§5/§7/§8、本条目）；
**没有改什么**（任何检查器与规则编号、workflow 的 `timeout-minutes` 数值、冻结协议常量、历史证据页与归档、
R-006/R-028/R-054/R-009 的判据本身）；验证命令与结果（见 §4）。

### 明确不做（写进新决策或 CHANGELOG 的「没有改」清单）

- 不改 `tools/check_conventions.py`、`tools/check_campaign_state.py`、`tools/check_goal_brief.py`
  （无新的可机械判定项；C-02 算术与 R-033/R-036 语义原样复用）。
- 不新增 `R-0xx` 编号、不删任何规则、不改 R-006/R-028/R-054/R-009 的判据文本。
- 不改任何 `.github/workflows/*.yml` 的数值；不重跑实验、不补 M3 的 23 项缺失评估（那是后续实验调度，按新规则另轮自主做）。
- 不回溯改 `training/r7_m3_protocol.py`、`training/r7_frozen_z_protocol.py` 等冻结常量与钉住它们的测试。
- 不重写历史证据页、已归档 `outputs/`、`docs/R7_*.md`（含 0021 当年的落痕）。

## §4 验证（每条都要实跑并记录结果；写入 CHANGELOG 与收尾报告）

1. `.venv/bin/python tools/check_conventions.py` —— 37 条阻断 0 违规；
2. `.venv/bin/python tools/check_conventions.py --rule R-033 --rule R-036` —— 新 ADR 三段/状态/代价标记齐全、supersede 链可达；
3. `.venv/bin/python tools/check_campaign_state.py` —— 退出 0（新增 `budget_mode` 键后 C-01..C-06 不变红）；
4. `.venv/bin/python tools/check_goal_brief.py --brief docs/goals --quiet` —— 目录级会**报告** 15 处失败，
   全部来自 5 份早于约定的历史长文（`d1-acquisition-and-b0` / `full-auto-campaign` / `gpu-bringup-3090` /
   `iteration-campaign` / `open-issue-resolution`；`docs/goals/README.md:29-34` 已登记为「报告而非待办」，
   不回溯改写）。对现行四份长文逐一运行必须 **0 失败**：
   `--brief docs/goals/n2a-m3-process-supervision.md --brief docs/goals/n3-m4-autoregressive-rollout.md`
   `--brief docs/goals/n4-m5-confirmation.md --brief docs/goals/v2-issue-closeout.md`（2026-10-02 实测 exit=0）。
5. `.venv/bin/python -m pytest tests/test_check_conventions.py tests/test_check_campaign_state.py `
   `tests/test_check_goal_brief.py -q` —— 全绿；
6. `git show --check` 与空白检查（`git diff --check`）；
7. 提交（`docs(r7): ...`，**不带**实验标签）→ 推分支 → 等 `ci.yml` 九步全绿（含 `Check campaign state`），
   如实记录 run id；失败/取消不算通过。

## §5 风险与不做（如实列出）

- **「大批量训练」的解读**：本计划按用户选项把「本地资源内的训练规模与时长」整体下放，
  即 AGENTS.md :18 的「大批量训练」从「无授权禁列」移出，但**要求租卡即停**。若用户希望保留一个
  更保守的口径（例如单轮超过某个 GPU-h 也要先问），在 D1 的第 3 条加一个数字即可，其余不变。
- **无总上限的代价**：只剩单次硬上限约束。账本继续逐行记账（保留证据指针与 C-03 比对），以便事后审计；
  若某轮超过用户心理额度，用户可以随时在长文里收紧（但**不得**回溯改写已发生的记账）。
- **超时语义的边界**：软预算超时「继续等待、记录在案」是对**实验自身**的；CI 侧仍然有 workflow 上限，
  所以长实验要么把 `timeout-minutes` 提到高于硬上限，要么拆批。M3 的教训（`whole 1805.1s` 超 30 min 文字 5.1s、
  截止只约束 GPU 阶段留下 CPU 前置时间缺口）在 D5 的新小节里写明：**硬上限按整轮墙钟计**，避免同一缺口复现。
- **Q-010**（`r7_cpu_study`/`r7_baseline_study` 缺内部截止）保持未决，不在本计划内顺手改。

## §6 交接注意事项（新窗口开工必读，否则 `check_campaign_state.py` 或提交闸门会红）

0. **先处理工作区遗留**：`git status` 里 2 modified（`docs/goals/README.md`、`docs/plans/README.md`）+
  本计划文件 + 4 份未跟踪 goal 长文，加上新窗口自己产生的改动，应**先**做成一条独立的 `docs(r7)` 提交并绑 CI
  再开始 D1，避免与规则改动混在一起（`git diff --cached` 复核）。
1. **R-037**：`docs/decisions/`、`docs/plans/`、`docs/rules/`、`.agents/skills/` 都在治理层清单里，改完必须 `git add`。
2. **R-033/R-036**：D1 的三段标题与状态行、D2 的 `superseded by 0029` 都要逐字对上检查器的正则
   （`- **状态**：`；Consequences 里必须出现「变难」或「代价」）。
3. **C-02**：D9 的 state 块**只加键、不改值**；`cap_gpu_h/used_gpu_h/remaining_gpu_h` 三个必需键一个都不能少。
4. **提交闸门**（`guard_conventions_before_commit`）会在提交时跑一遍阻断规则；规则文件改动若让某条转红会被
   当场拒绝——这也是 §4 第 1、2 条要先跑的原因。
5. 实验 workflow 是 commit-message 标签门控：普通 push 上它们 skipped 是设计行为；本计划的提交**不带**标签，
   不触发任何实验。推分支后只需等 `ci.yml`。

## §7 新窗口提示词（可直接粘贴）

```text
你在 /data/esw/UrbanPiDiT_R2（工作分支 r7/weather-reasoning）继续。本轮任务：落地「实验常设下放与时长契约」治理变更（新决策 0029 取代 0021）。完整执行计划在 docs/plans/0010-experiment-delegation-and-time-contract.md —— 先通读它（含 §2 起点事实、§3 的 D1–D10、§4 验证、§6 交接注意事项），再按序执行。

用户已定的三项决定（不要重新询问、不要再让用户逐项拍板）：
1) 实验类动作全下放：本地 GPU 运行、训练/评估、触发实验 workflow、每次实验的时长与预算分配，均由执行者自主决定，不再逐次 AskUserQuestion。
2) 不设总 GPU-h 上限：campaign 账本从「闸门」降为「记账」，保留 cap/used/remaining 会计字段与逐行证据指针。
3) 时长契约＝软预算 + 宽松硬上限：超软预算继续跑并记录 soft_overrun_seconds，仅硬上限截断；截断终态记 budget_limited/failed、全额记账、不算科学通过。

仍须逐次取得用户授权：GPU 租赁与付费资源、多年度/新数据下载、main 合并、force push、破坏性数据操作。
节点推进（current_node）与目标完成判定仍只可由用户/独立复核裁定，执行者只可提议。

纪律：不豁免 R-006（先冻结 protocol）/R-028（禁网）/R-054（共驻）/R-009（不弱化判据；skip/cancelled/queued 不算通过）；不回溯改冻结协议常量（r7_m3_protocol.py、r7_frozen_z_protocol.py 等）与历史证据页；不改任何 workflow 的 timeout-minutes 数值；负面结果照写；不自行宣布目标完成。

按计划 §4 验证，全部实跑并记录：python tools/check_conventions.py；--rule R-033 --rule R-036；tools/check_campaign_state.py（退出 0）；python tools/check_goal_brief.py --brief docs/goals --quiet；pytest tests/test_check_conventions.py tests/test_check_campaign_state.py tests/test_check_goal_brief.py -q；git diff --check。然后按 AGENTS.md 提交纪律提交（docs(r7): ...，不带实验标签），推分支，等 ci.yml 九步全绿并记录 run id。

开工第一步：git status 核对工作区遗留（计划 §2 已列：2 modified + 5 个未跟踪文件，含本计划），先把它们做成一条独立 docs(r7) 提交并绑 CI，再开始 D1。

收尾：按 AGENTS.md「完成任务时必须报告」九项报告；并写明此后 campaign 实验（N2a 补全 → N3 → N4 → N5 → 关闭轮）按 docs/plans/0009 与主计划在新规则下自主调度，除保留项外不再逐次询问用户。
```

## 实际结果（计划归档会话，保留原记录）

- **完成情况**：**未执行**（按用户指示）。本次会话只产出：本计划文件、`docs/plans/README.md` 索引行，
  以及本文件 §7 的交接提示词。D1–D10 的规则改动、提交、推送、CI、GPU 均未触碰。
- **与计划的差异**：计划起草时 HEAD 为 `ed1a034`，落盘时已前进到 `6e25c7e`（M3 失败登记两个提交由并行工作
  先行提交）；§1 已如实更新为落盘时的准确起点。第一版方案曾设想在本次会话直接执行全部改动（并在计划模式里
  被否决一次），用户随后明确「只给计划文件与提示词，我在新窗口继续」，故执行部分整体后移。
- **验证**：本文件落盘后跑只读检查——`python tools/check_conventions.py`（37 条阻断 0 违规，
  含 R-032 对新计划的「实际结果」检查）；`python tools/check_goal_brief.py` 对现行四份长文逐一检查
  0 失败，目录级 `--brief docs/goals` 报 15 处失败且全部来自 5 份早于约定的历史长文（README 已登记为
  报告而非待办）；`git diff --check` 空白干净（新文件经 `--no-index` 复核无空白错误）。
  未提交、未推送、未跑 CI（本会话无写远端动作）。
- **遗留**：D1–D10 全部待执行；工作区遗留（本文件 + 索引行 + 4 份 goal 长文）尚未提交；
  执行窗口按 §6 第 0 条先处理提交，再按 §3 逐项落地。

## 执行记录（2026-10-02）

### 起点、授权与顺序

- 用户本轮明确要求执行本计划，沿用 §0 的三项已定决定，不重新询问。
  实际起点 `6e25c7eef57dd622e68ce3df4af06e7d6c1f261f`、分支 `r7/weather-reasoning` 与计划一致。
- §6 第 0 条先落实：只暂存审阅过的两个索引、计划 0009/0010、n3/n4/closeout 三份未跟踪长文，
  共 7 文件独立提交 `5058610744377e6729f12e49b7066b544a9e56aa` 并推工作分支。
  CI `36993386517` 对精确 SHA completed/success、九主步骤均 success 后，才开始 D1。
  `.zcode/agents/web-researcher-backup.md` 与 `.zcodeignore` 是无关遗留，未改、未暂存。
- 本轮只实施治理、0 新增 GPU-h，不训练/评估、不触发实验 workflow、不改节点或关闭 issue。
  goal 自动完成校验最近一条只读记录为 error（600002ms）；按决策 0024 best-effort，不依赖它结项。
  独立只读内容审阅与逐条实证审计用于交叉核对，不替代用户/独立目标裁定。

### Prompt-to-artifact 交付对表

| 要求 | 当前产物与证据 | 判定边界 |
| --- | --- | --- |
| D1 | `docs/decisions/0029-standing-experiment-delegation.md`，2026-10-02/accepted，Context/Decision/Consequences，五项决定与代价/备选 | 常设下放、保留项、整轮软/硬时长、无总上限、不可豁免/不回溯齐全；ADR 检查 exit 0 |
| D2 | `docs/decisions/0021-experiment-authorization-channel.md` 状态 `superseded by 0029`、追加反向指针 | 原正文逐字保留，只改状态与追加指针 |
| D3 | `docs/decisions/README.md` 的 0021/0029 两行 | 0021 取代状态与 0029 accepted/日期对应 |
| D4 | `AGENTS.md` 硬约束两条 | 本地实验规模/时长下放；共驻与诚实性原文未改 |
| D5 | `docs/rules/ci-and-verification.md` 的 R-029/R-030、触发授权、新时长小节 | planned/hard/overrun、整轮墙钟与严格 CI 外层余量；例外/Q-010 不改 |
| D6 | `docs/rules/gpu-resources.md` 首段与 N1 参数授权定位 | 0026/0029 指针同步；R-054 本体与 N1 数值不改 |
| D7 | `.agents/skills/bounded-study-run/SKILL.md` 授权/前置/执行/常见失败 | 写死软/硬端点、只因硬上限时长截断；新数据/write 保留授权 |
| D8 | `.agents/skills/goal-loop/SKILL.md` 前置条件 2 与旧额度误用失败项 | 无总上限、逐轮 planned/hard；其它步骤与完成判据保留 |
| D9 | `docs/goals/main-model-v2-campaign.md` §3/§5/§7/§8，另同步 §4 现行指针 | 仅新增 budget_mode；原 state/账本/历史进度与 N2a/budget_limited 未改 |
| D10 | `docs/rules/CHANGELOG.md` 顶部决策 0029 条目 | 范围、没有改、实跑命令与结果如实登记；提交 CI 待补绑定 |
| §6 顺序 | 遗留提交 `5058610` + CI `36993386517` 九步 | 在 D1 前已绑定成功，不混提交 |
| §4 第 1–6 项 | 下表实跑日志；771 冻结文件/9 保留长文/11 账本行/原 state 的只读 hash/结构核验 | 已实跑；goal 目录历史 exit 1 不冒称全绿 |
| §4 第 7 项 | 治理提交、工作分支推送与精确 SHA 的 CI 待绑定 | 不用遗留 CI 代替，最终记录另附 |

### 实际验证（2026-10-02）

| 实跑命令 | 结果 |
| --- | --- |
| `.venv/bin/python tools/check_conventions.py` | 37 条阻断、0 违规，exit 0 |
| `.venv/bin/python tools/check_conventions.py --rule R-033 --rule R-036` | ADR 结构/状态/代价与取代链 0 违规，exit 0 |
| `.venv/bin/python tools/check_campaign_state.py` | node=N2a、0 failures/4 历史 C-03 notes，exit 0 |
| `.venv/bin/python tools/check_goal_brief.py --brief docs/goals --quiet` | 17 briefs、15 failures、exit 1；JSON 核实全部来自 §4 第 4 项所列五份历史长文，非待办 |
| 现行 n2a/n3/n4/closeout 四份逐一 `check_goal_brief.py --brief ...`，以及四个 `--brief` 合并检查 | 全部 0 failures/0 advisories，exit 0 |
| `.venv/bin/python -m pytest tests/test_check_conventions.py tests/test_check_campaign_state.py tests/test_check_goal_brief.py -q` | 152 passed in 22.53s，0 skip，exit 0 |
| `git show --check`、`git diff --check` | 均 exit 0 |
| `tools/verify_r7_evidence_index.py --index docs/R7_EVIDENCE_INDEX.jsonl --root . --check-brief docs/R7_CANDIDATE_BRIEF.md` | 17 records，exit 0；原索引/简报未改 |
| 只读基线 hash/结构审计 | 771 冻结范围路径、9 份保留长文、两无关文件均不变；11 账本行原样，原 state 全字段原样，仅加 budget_mode；历史 campaign 尾、0021 正文与旧 CHANGELOG 均保留 |

工作态日志在 `/tmp/r7_0029_validation_20261002/`；基线在 `/tmp/r7_0029_baseline.json`。
这些不提交，长期可复核路径与实测结果登记在本节/CHANGELOG，检查器与测试源码不变，可在新克隆重跑。
遗留 CI 直接来源：
[run API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/36993386517)、
[jobs API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/36993386517/jobs?per_page=100)，
匿名 curl 访问 2026-10-02；精确 SHA/九步逐项摘要和响应 SHA256 存
`/tmp/r7_0029_handoff_ci_20261002/verification.json`。未取远端 pytest 日志计数，不把本地计数称远端。

### 差异、未做与影响

- **与计划的差异**：在指定文件内作三处必要一致性补正：D7「完整训练/多年度数据另授权」拆开；
  D9 §4 现行授权指针换为 0029；§3 算术说明明确 24 仅为会计基数。只读审阅指出前两处若留旧文会
  抵消新规则；没有扩大 D1–D10 的工程改动范围。计划/索引追加执行结果，而不是覆盖旧归档记录。
- **没有改**：检查器与规则编号、workflow 数值、冻结协议/测试、R-006/R-028/R-054/R-009 判据、
  历史证据/outputs/数据/模型/依赖/凭据/安全配置；账本值、当前节点与原科学结论。没有放宽身份校验，
  没补 M3 的 23 项缺评估，没有新训练、实验重放、取数/发布、租卡、main 写入或 issue 关闭。
- **安全边界**：Mimosa commit/push hook 未得完整结论（scanner_enobufs），未运行完整安全审计，
  不声明项目安全；临时验证脚本泛化命令封装被写前扫描拒绝后改为固定命令实跑，未绕过扫描或改配置。
- **兼容性**：治理授权从 0021 转为 0029；历史 runner/冻结常量并未自动实现新时长语义。未来新实验须
  在该轮长文和协议写死 planned/hard，并核 CI 外层余量，不能按旧 ≤30 min 或总账本数字重新设授权闸门。
  旧 prepared n3/n4 长文和计划 0009 原文保留为历史/准备依据，开工前需明确 0029 的适用。
- **下一项（留待新窗口）**：按计划 0009 与主计划准备 N2a 补全 → N3 → N4 → N5 → 关闭轮，
  除保留项外不再逐次询问实验/时长授权；仍按冻结判据、失败记录、节点与完成裁定边界执行。
  本轮不运行这些实验，不自行宣布当前 goal 完成。
