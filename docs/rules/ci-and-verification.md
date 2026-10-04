# CI 与验证

范围：`.github/workflows/**`、门禁脚本、失败处理纪律。

## 现状（实测，18 个 workflow）

| 类别 | workflow | 说明 |
| --- | --- | --- |
| 主测试门禁 | `ci.yml` | push 到 `r7/weather-reasoning` 或 PR 到 `main` 时触发；`compileall` + `git show --check` + `pytest -q`；30 分钟超时 |
| 离线实验（10 个） | baseline-study, common-case, continuous-control, correction-audit, cpu-study, extended-control, restored-diagnostic, seasonal-study, spatial-solver, pressure-replay | 由提交信息标签触发，下载已归档产物，禁用 socket，跑实验，归档代码身份 |
| 真实数据获取（7 个） | continuous-pilot, earthmover-probe, pressure-pilot, public-data, real-smoke, seasonal-pilot, surface-pilot | 允许联网，受字节/时间预算约束 |

触发机制：除 `ci.yml` 外，每个 workflow 由提交信息里的方括号标签触发
（如 `[cpu-study]`、`[public-data]`）。全仓 134 个提交中 133 个带 issue 引用，
34 个带这类标签。

## R-027 commit 信息遵循 Conventional Commits 并引用 issue

- **级别**：默认
- **范围**：全部分支
- **陈述**：提交信息形如 `type(r7): 摘要 (#NN) [可选标签]`。
- **依据**：E-094（最近 134 个提交中 133 个匹配 `^[a-z]+(\([^)]*\))?:\s`；
  类型分布 feat 28 / test 8 / fix 6 / data 6 / ci 5 / docs 4 / experiment 2 / build 1；
  scope 全部为 `r7`；唯一不合规者是仓库早期的 `feat: Implement R2LightningModule...`）
- **现状**：B 类 —— 1 处历史例外；抽样最近 30 条 **0 处**不合规
- **执行方式**：脚本（`--rule R-027`，报告型；抽样 30 条提交）
- **例外**：仓库首个提交 `feat: Implement R2LightningModule and R2Loss for multi-target training`
  （早于本约定确立）
- **引入日期**：2026-09-24
- **复核触发**：当引入 squash 合并或改用语义化发版工具时

## R-028 离线实验必须真正禁网

- **级别**：必须
- **范围**：消费已归档产物的 workflow 与实验脚本
- **陈述**：不打算联网的实验必须显式禁用出网（monkeypatch `socket.socket.connect` 与
  `socket.create_connection` 使其抛错），而不是"希望它不联网"。
- **依据**：E-077（10 个 workflow 内联安装该拒绝逻辑，如 `r7-cpu-study.yml:37-39`）、
  E-095（`training/r7_baseline_study.py:85-91` 在子进程中同样禁网，并在 `:96-98` 检查返回码）
- **现状**：A 类 —— 所有"下载产物 + 跑有界实验"的 workflow 均已实现
- **执行方式**：**脚本（`--rule R-028`，阻断）**——检查所有消费归档 artifact 的 `r7-*.yml` workflow，确认其实验/诊断步骤同时含两个 socket 赋值
- **例外**：无
- **引入日期**：2026-09-24
- **复核触发**：当把该片段抽成共享 `deny_network()` 辅助时（应改为检查 import 而非文本赋值）

### 已知局限（如实记录）

该防护是**进程内属性替换**，能挡住 `urllib`/`requests`/`s3fs` 等常规路径，但挡不住
绕过 Python socket 层的调用。它是"防意外"而非"防对抗"。不要声称它是沙箱。

## R-029 workflow 必须设超时

- **级别**：必须
- **范围**：`.github/workflows/*.yml`
- **陈述**：每个 workflow 的 job 必须设置 `timeout-minutes`。
- **依据**：E-096（18 个 workflow 全部设置了 `timeout-minutes`，取值 10–30 分钟）
- **现状**：A 类 —— 18/18 合规
- **执行方式**：**脚本（`--rule R-029`，阻断）**
- **例外**：无
- **引入日期**：2026-09-24
- **复核触发**：当实验规模增大需要更长时限时（应提高数值，而不是删除时限）。承载有界实验的
  workflow 的 `timeout-minutes` 必须高于该实验的 `hard_cap_seconds`（换算为秒比较，决策 0030（继承 0029）），
  先内部截断并留出证据收尾时间；提高数值而不是删除时限的纪律不变。

## R-030 长任务必须有内部截止时间

- **级别**：默认
- **范围**：`training/r7_*study.py`、`training/r7_*control.py`
- **陈述**：批量实验在执行循环内检查墙钟截止时间并抛错，避免耗尽 CI 时限后才失败。
  决策 0030（继承 0029） 起，后续实验的模块内截止检查语义是**宽松硬上限**：计划时的软预算到达或超出均不中止，
  只在硬上限因时长截断，并记录 `soft_overrun_seconds`；冻结协议常量
  （如 `global_deadline_seconds=1800.0`）不回溯修改。
- **依据**：E-097（`r7_continuous_control.py:92-93`、`r7_extended_control.py:103-104`、
  `r7_spatial_study.py:103-104` 均在循环内检查 1080 秒并 `raise RuntimeError`）、
  E-131（`r7_seasonal_study.py` 原本缺此检查，已于第二遍补上，签名为 keyword-only
  `deadline_seconds=1080`）
- **现状**：C 类（报告）—— 4 个模块合规，2 个例外
- **执行方式**：脚本（`--rule R-030`，报告型）
- **例外**（精确路径，均为 R-030 约定确立之前的模块，见 OPEN_QUESTIONS Q-010）：
  - `training/r7_baseline_study.py`
  - `training/r7_cpu_study.py`
- **引入日期**：2026-09-24
- **复核触发**：当这两个模块被重跑或修改时，应补上检查并清空例外

### 本文件的强制机制（第二遍新增）

`tools/check_conventions.py` 已接入 CI：`.github/workflows/ci.yml` 的
`Check repository conventions` 步骤运行 **37 条阻断规则**（不含 R-009、R-027、R-030 等报告型，另有 7 条规则未机械化）。

**2026-09-30 追加**：同一作业新增 `Check campaign state` 步骤，运行
`python tools/check_campaign_state.py --quiet`（决策 0025）。它把主计划
（`docs/goals/main-model-v2-campaign.md`）的当前节点、GPU 账本算术、上一轮证据的索引登记与
轮次长文的结构对表；规则号 `C-01`…`C-06` 属该工具自己的命名空间，**不是** `R-0xx`；
硬漂移（节点不一致、账本对不上、证据未登记、长文结构失败）退出码 1。
它**不判断科学方向**——通过只表示「与主计划不矛盾」。该步骤使作业从八步变**九步**；
历史证据页里的「八步」是它们当时的实测，不回溯改写。

**2026-10-04 接续（0038/计划0016）**：同一个 `Check campaign state` 步骤另调用
`python tools/check_campaign_state.py --campaign docs/goals/main-model-climatology-campaign.md --quiet`。
默认旧master检查保留，新路径同样强核C-01–C-06；不宽免空账本/不可达证据、不新增实验workflow、
标签、依赖或时限。未来CI核实际精确SHA的全部必要步骤，不照抄历史九/十二步数。工具绿只说明
机械一致，不是超气候态门；本轮未commit/push，不能声称已有远端CI覆盖这些改动。

`ci.yml:35-40` 原有步骤（第二遍已修正范围）：

1. `python -m compileall -q data model training scripts train_r7_local.py evaluate_r7_local.py`
   —— 第二遍补上 `scripts/`。
2. `git show --check --format= HEAD` —— 第二遍从 `git diff --check` 改为检查最近提交，
   因为在干净 checkout 上原命令的 diff 为空、实际不产生门禁作用（E-099、Q-006）。
3. `pytest -q` —— 真正的门禁。

**没有**任何静态分析门禁：全仓无 ruff / flake8 / mypy / black / pylint 配置，
也没有 `.pre-commit-config.yaml`（E-080）。这是本次引导识别出的最大工程缺口，见 `MIGRATION.md`。

### 机器可读验证回执试点

`r7-cpu-study` 在 `always()` 收尾步骤生成并只读校验 `verification_receipt.json`。回执绑定
`run_id`、workflow、commit/protocol/source/data identity、必需产物和逐文件 SHA256，并保留
`scientific_claim: false`、`limitations` 与失败原因。仅 `status: success` 且 `--require-success` 校验通过
才是工程上的成功；`partial`、`failed`、`cancelled`、`queued`、`skipped` 均不算通过，也不产生科学结论。
该试点不改变 workflow 标签、预算、禁网边界或现有结果冻结/人工科学判定。

**首次真实运行（2026-09-29）**：`R7 real offline multiseed CPU study` run `36549248954`（commit `b967fc1`）
的回执步骤经 `--require-success` 严格校验通过；主门禁在修复简报末尾空行后（commit `bf96ea8`，
run `36549905342`）全绿。两条证据见 `EVIDENCE.md` 第六遍（E-189、E-190）。

CI 主门禁还会调用 `tools/verify_r7_evidence_index.py --check-brief`，确保提交的
`docs/R7_CANDIDATE_BRIEF.md` 与 `docs/R7_EVIDENCE_INDEX.jsonl` 的规范渲染逐字一致；该步骤只读，
不会自动生成或覆盖简报。

## Workflow 触发契约与 GitHub 通道（2026-09-25 追加）

背景：用户实测发现"GitHub 上很多 workflow 都是 skipped"，并明确本项目**只用 `git`/SSH
操作 GitHub，不用 `gh`，也不要假定 PAT 存在**。这两点共同决定了 workflow 的触发与
issue 的关闭方式。

### 17 条实验 workflow 是 commit-message 标签门控

每条 `r7-*.yml` 的 push 触发限定在 `r7/weather-reasoning`，且 job 上有
`if: contains(github.event.head_commit.message, '[<tag>]')`。因此**普通 push 上它们显示
skipped 是设计行为，不是失败**——它们是有意做成"按需运行"的实验入口：

| 标签 | workflow |
| --- | --- |
| `[baseline-study]` | r7-baseline-study |
| `[common-case]` | r7-common-case |
| `[continuous-control]` | r7-continuous-control |
| `[continuous-pilot]` | r7-continuous-pilot |
| `[correction-audit]` | r7-correction-audit |
| `[cpu-study]` | r7-cpu-study |
| `[era5-temporal-probe]` | r7-earthmover-probe |
| `[extended-control]` | r7-extended-control |
| `[pressure-pilot]` | r7-pressure-pilot |
| `[pressure-replay]` | r7-pressure-replay |
| `[public-data]` | r7-public-data |
| `[real-smoke]` | r7-real-smoke |
| `[restored-diagnostic]` | r7-restored-diagnostic |
| `[seasonal-pilot]` | r7-seasonal-pilot |
| `[seasonal-study]` | r7-seasonal-study |
| `[spatial-solver]` | r7-spatial-solver |
| `[surface-pilot]` | r7-surface-pilot |

需要运行某条实验时，把它的标签写进 commit message 再推到工作分支；一条 commit 可带多个
标签。不要为了"消除 skip"而给它们加 `paths:` 过滤——那会改变既有触发契约且无证据支持。

**（2026-10-02 更新）实验与时长预算按决策 0030（继承 0029） 常设下放**：本地 GPU 运行、训练/评估、触发实验
workflow 与每次时长/预算由执行者自主决定；先记录范围、软/硬端点、产物与证据、失败与 skip 处理并
冻结协议，仍受 R-006/R-028/R-054 约束。付费资源、新数据下载与数据发布 `--write` 等保留项仍须
逐次显式授权。总体研究方向归用户；普通决策和节点推进也按 0030 下放，可在当前 goal 内按前置/
出口与每节点对表继续，独立审阅不构成新推进许可。标签只表达触发意图，**不构成授权**；
`skip` / `cancelled` / `queued` 不算通过。最终 goal 完成不得由执行者自行裁定。

### 实验时长契约（决策 0030（继承 0029））

- `planned_seconds` 是计划时的软预算；`hard_cap_seconds` 是宽松硬上限，默认约为计划的 2 倍，
  具体两个数字在该轮长文写死并于训练前写入冻结的 `protocol.json`。只有声明了两个端点，才能审计
  软预算 overrun 与硬上限截断的区别。
- 软/硬时长都按**整轮墙钟**计，包括 CPU 前置、训练、评估、进程启动/间隔与清理。M3 历史整轮
  1805.1086s 超过「≤30 min」文字 5.1086s 的缺口，是截止只覆盖 GPU 阶段所致；事实和原终态保留，
  后续实验不得复用该计时缺口。GPU-h 另按实际资源消耗全额记账，不把墙钟与 GPU-h 混为一谈。
- 超软预算后**继续等待、记录在案**，不因软预算中止；运行记录写
  `soft_overrun_seconds = max(0, whole_elapsed_seconds - planned_seconds)`。只有硬上限因时长截断；
  真实错误、身份/资源门禁或冻结停止条件仍会提前停止，不以「继续等待」掩盖失败。
- 硬上限截断后记 `budget_limited`/`failed`、保留失败与中间证据、全额记账，不算科学通过。
  承载实验的 workflow 必须满足 `timeout-minutes * 60 > hard_cap_seconds`，且预留证据收尾余量，
  先由实验自身截断而非等 GitHub 取消 job；必要时按新实验提高 CI 时限或拆批，不能删除时限。
- 不回溯改冻结常量（包括 `global_deadline_seconds=1800.0`、`max_round_seconds=1800`）、
  历史「≤30 min」文字、M3 的 `budget_limited` 终态或归档产物。本轮无新实验，不批量改任何
  workflow 的 `timeout-minutes` 数值，也不声称历史 runner 已自动实现新契约。

**用户决定（2026-09-25）：issue 关闭工作期间不主动触发实验 workflow**（运行成本过高，
本轮无新实验）。skip 状态被接受为设计行为；验收证据以早前已核验的绿色 run 为准，
不得因跳过 CI 而声称新的实验结论。

### issue 的自动关闭只有一条 git 路径

closing keywords（`Closes/Fixes/Resolves #N`）只在**默认分支 main** 的提交里生效；推到
工作分支无效。本仓库无 `gh`、无 token（API 写实测 401），因此：

1. 在工作分支上做收尾提交，message 含 `Closes #N`（可多条）；
2. main 是工作分支的严格祖先（2026-09-25 实测：领先 165、落后 0），所以把该提交
   **fast-forward 推到 main 即可，任何形式的 force 都被禁止**；
3. **hook 自 2026-09-25 起放行非 force 推送到 main**（决策 0003）：`guard_destructive_git`
   已移除 `pushing to main` 规则，agent 可直接执行该 ff 推送，无需逃生口。
   **合并仍被拒绝**（`git merge` 涉及 main、`gh pr merge`）：需用户明确授权后由用户
   在 ZCode 之外终端执行，或会话启动前移除 hook 条目（会话边界实测仍成立）。
   **禁止**用 refspec 拼写绕过匹配正则；删除默认分支与 `--mirror` 亦被拒绝；
4. 两个副作用须写明：main 到达分支顶端后 **PR #12 会显示为 merged**；且 main 的 push
   不触发 `ci.yml`（其 push 触发只限工作分支），所以验收证据是分支上的绿色 push run，
   而不是 main 上的任何 run。

**依据**：用户 2026-09-25 的明确指示（用 git 而非 gh、要求真正关闭 8 个 open issue、
要求处置"大量 skipped workflow"）；实测记录见 `docs/goals/open-issue-resolution.md`。
