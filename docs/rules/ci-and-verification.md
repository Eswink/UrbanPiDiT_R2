# CI 与验证

范围：`.github/workflows/**`、门禁脚本、失败处理纪律。

## 现状（实测，18 个 workflow）

| 类别 | workflow | 说明 |
| --- | --- | --- |
| 主测试门禁 | `ci.yml` | push 到 `r7/weather-reasoning` 或 PR 到 `main` 时触发；`compileall` + `git diff --check` + `pytest -q`；30 分钟超时 |
| 离线实验（11 个） | baseline-study, common-case, continuous-control, correction-audit, cpu-study, extended-control, restored-diagnostic, seasonal-study, spatial-solver, pressure-replay, pressure-pilot | 由提交信息标签触发，下载已归档产物，禁用 socket，跑实验，归档代码身份 |
| 真实数据获取（7 个） | continuous-pilot, earthmover-probe, public-data, real-smoke, seasonal-pilot, surface-pilot | 允许联网，受字节/时间预算约束 |

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
- **执行方式**：**脚本（`--rule R-028`，阻断）**——检查含 `download-artifact` 且匹配实验模式的
  workflow 是否同时含两个 socket 赋值
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
- **复核触发**：当实验规模增大需要更长时限时（应提高数值，而不是删除时限）

## R-030 长任务必须有内部截止时间

- **级别**：默认
- **范围**：`training/r7_*study.py`、`training/r7_*control.py`
- **陈述**：批量实验在执行循环内检查墙钟截止时间并抛错，避免耗尽 CI 时限后才失败。
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
`Check repository conventions` 步骤运行 20 条阻断规则（不含 R-027/R-030 等报告型）。

`ci.yml:35-40` 原有步骤（第二遍已修正范围）：

1. `python -m compileall -q data model training scripts train_r7_local.py evaluate_r7_local.py`
   —— 第二遍补上 `scripts/`。
2. `git show --check --format= HEAD` —— 第二遍从 `git diff --check` 改为检查最近提交，
   因为在干净 checkout 上原命令的 diff 为空、实际不产生门禁作用（E-099、Q-006）。
3. `pytest -q` —— 真正的门禁。

**没有**任何静态分析门禁：全仓无 ruff / flake8 / mypy / black / pylint 配置，
也没有 `.pre-commit-config.yaml`（E-080）。这是本次引导识别出的最大工程缺口，见 `MIGRATION.md`。

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

**用户决定（2026-09-25）：issue 关闭工作期间不主动触发实验 workflow**（运行成本过高，
本轮无新实验）。skip 状态被接受为设计行为；验收证据以早前已核验的绿色 run 为准，
不得因跳过 CI 而声称新的实验结论。

### issue 的自动关闭只有一条 git 路径

closing keywords（`Closes/Fixes/Resolves #N`）只在**默认分支 main** 的提交里生效；推到
工作分支无效。本仓库无 `gh`、无 token（API 写实测 401），因此：

1. 在工作分支上做收尾提交，message 含 `Closes #N`（可多条）；
2. main 是工作分支的严格祖先（2026-09-25 实测：领先 165、落后 0），所以把该提交
   **fast-forward 推到 main 即可，任何形式的 force 都被禁止**；
3. 这次对 main 的写入受 `guard_destructive_git` 拦截。需要执行时按 AGENTS.md 记载的
   逃生口：临时移除 `.zcode/config.json` 里该 hook 条目 → 完成后**立即恢复**，并把移除/
   恢复与授权来源记入一份决策记录（R-033）。**禁止**用 refspec 拼写绕过匹配正则；
4. 两个副作用须写明：main 到达分支顶端后 **PR #12 会显示为 merged**；且 main 的 push
   不触发 `ci.yml`（其 push 触发只限工作分支），所以验收证据是分支上的绿色 push run，
   而不是 main 上的任何 run。

**依据**：用户 2026-09-25 的明确指示（用 git 而非 gh、要求真正关闭 8 个 open issue、
要求处置"大量 skipped workflow"）；实测记录见 `docs/goals/open-issue-resolution.md`。
