# UrbanPiDiT-R² 工程契约

本文件是每轮会话都会注入的常驻契约：只放不可变边界、硬约束与路由表。
细则在 `docs/rules/`，多步骤流程在 `.agents/skills/`（索引见 `docs/skills/README.md`），
检查脚本在 `tools/check_conventions.py`，自动拦截在 `tools/agent_hooks/`。
规则变更留痕在 `docs/rules/CHANGELOG.md`。

工作分支 `r7/weather-reasoning`；PR #12 已于 2026-09-25 merged（此后到 main 的更新
走非 force ff 推送，见下方「GitHub 通道」）。当前处于**科研进行中**状态：
不能把工程通过当作科学结论，不能把合成 fixture 当作天气真值。

## 硬约束

- **禁止**改动、移动或重命名 `data/raw/`、`data/interim/`、`data/processed/` 下的内容；原始数据只读。
- **禁止**把归档快照（`legacy_v531_full/`、`data/legacy_v531/`、`model/legacy_v531/`、`legacy_v6/`）
  当作活跃代码；只读参考，不得被 import、不得写入。
- **禁止**合成或伪造真实数据来替代失败的下载；失败必须留下 `failed-no-fallback` 审计记录。
- **禁止**在无显式授权时执行：GPU 租赁与付费资源、多年度或新数据下载、`main` 合并、force push、破坏性数据操作。（本地实验的规模与时长已按决策 0029 常设下放，见下一行。）
- 实验类授权按**决策 0029**：本地 GPU 运行、训练/评估、触发实验 workflow 与**每次实验的时长与预算**
  由执行者自主决定，不再逐次询问；协议仍须先冻结（R-006）、离线实验仍须禁网（R-028）、GPU 仍须共驻（R-054）。
  时长契约＝**软预算 + 宽松硬上限**：超软预算继续跑并记 overrun，仅硬上限截断；截断终态不算科学通过。
- 本机 GPU **默认共驻**；启动前与每次子进程 spawn 前只读核对余量门槛，**禁止**对非本实验进程发送任何信号或实现冻结/终止自动化；独占须先取具名授权（决策 0026、`docs/rules/gpu-resources.md`）。
- **禁止**未实跑就写 PASS；skip 不算通过；被取消或排队的运行不算通过。
- **禁止**为了让报告好看而删除、跳过或弱化测试与断言，也**禁止**事后放宽已冻结的判据。
- **禁止**绕过 checkpoint / 数据身份校验（`model_code_sha256`、source SHA256、`BUILD_COMPLETE`）；
  旧产物要用它**归档的 `code.zip`** 重放，不能修改比对逻辑。
- 任何进入训练或评估的真实数据，**必须**先通过 `prepare_r7_local.py` 的只读 preflight 报告，并取得显式 `--write` 授权。
- 批量实验**必须**在训练开始前把协议写入 `protocol.json`，并在结果中记录其 digest。
- 实验产物**必须**带 `scientific_claim: false`（或等价显式标志），并如实记录 `limitations`。
- 结论**必须**可追溯到运行记录与产物 digest；无法证明逐位一致时，**必须**写明可复现等级。
- 新增代码受**规模与命名硬上限**约束（决策 0022）：单文件 ≤600 行（R-051）、函数体 ≤200 行（R-052）、
  代码目录 snake_case 且 skill 目录必须含 `SKILL.md`（R-053）。超限的既有文件以精确路径**冻结**在
  清单里、**只许缩小**（拆到上限以下时测试会强制移出该条）；放宽阈值或新增例外必须另立决策记录。
- 报告**必须**区分「已确认」与「推测」；未做的事写未做，禁止用「应该没问题」代替。
- 外部信息**必须**经 `web-researcher` 子智能体取得（主链路的 `WebSearch`/`WebFetch` 被
  PreToolUse 闸门拒绝，R-049）；它不可用时降级为主链路 `curl` 并写明原因；引用必须带
  URL 与访问日期，**禁止**把检索摘要当证据（R-050）。

## 入口点

| 入口 | 是否进 wheel | 运行方式 |
| --- | --- | --- |
| `urbanpidit-r7-{train,evaluate,prepare,calibrate}` | 是（console script） | 安装后直接调用 |
| `train.py` | 否 | 仓库根：`python train.py --config <cfg>` |
| `diagnose_r7_gain.py` / `profile_r7_inference.py` / `tune_r7_halting.py` | 是（py-modules） | 仓库根：`python <name>.py --...`（测试按裸文件名 + `cwd=repo` 调用） |
| `scripts/train_r7{,_process,_recursive}.py` | 否 | `python scripts/<name>.py --config <cfg>`（已含 repo-root 路径修正） |

改动入口位置会破坏按路径调用它们的测试与脚本，见 `docs/rules/OPEN_QUESTIONS.md` Q-002。

## 环境

- 本项目使用仓库内的 `.venv`（Python 3.12，`torch` CUDA 构建）。进入
  `/data/esw/UrbanPiDiT_R2` 或其子目录时由 `~/.bashrc` 的
  `project venv auto-activation` 块自动激活；离开自动退出。手动激活的其它环境会被尊重。
- 在项目内该块会**临时退出 conda base**（服务器上 base 默认自动激活，会与 venv 一起
  显示成 `(.venv) (base)`）；离开项目时自动恢复 base。**其它目录的 conda 行为完全不变**，
  `conda` 命令在项目内仍可手动使用。不想隐藏 base 就把 `__PROJECT_HIDE_CONDA_BASE` 设为 0。
- **优先用 `.venv/bin/python`**，不要用 conda base 或系统 python。
- 三份 `requirements*.txt` 的并集**不等于**全部依赖：`icechunk`、`pcodec`、`numcodecs`、`pip`
  未在 requirements 中声明。完整清单与复现命令见 `docs/rules/environment.md`（缺口见 Q-011）。
- CI 是 CPU-only；本机有 2×RTX 3090，因此本地能跑而 CI 不能跑 GPU 路径（反之亦然）。

## GitHub 通道与 issue 关闭（2026-09-25 起生效）

- **本项目只用 `git`/SSH 操作 GitHub，不用 `gh`，也不要假定 PAT 存在。** 实测：
  API 读匿名可用（public 仓库，限额 60 次/小时）；API 写（评论/关闭 issue）**401**；
  CI 日志下载 403；`git push` 走 SSH（身份 `sqy941013`）可用。
- **自动关闭 issue 只有唯一一条 git 路径**：`Closes/Fixes/Resolves #N` 出现在**默认分支 main**
  的提交里才会生效——推到工作分支无效。main 是本分支的严格祖先（ff 推送，禁 force）。
  **非 force 推送到 main 已被 `guard_destructive_git` 放行**（2026-09-25 用户授权，
  决策 0003）：GitHub 自身拒绝非 ff 推送、全部 force 变体与删除默认分支/--mirror 被
  hook 拒绝，因此 main 只会线性前进到工作分支上已过 CI 的提交。
  **合并仍被 hook 拒绝**（本地 `git merge` 涉及 main、`gh pr merge`）：合并需用户明确
  授权后**由用户在 ZCode 之外的终端执行**，或在会话启动**前**移除 hook 条目——
  **实测边界（2026-09-25，决策 0002 附录）**：hook 配置在**会话启动时**读取，会话中途
  移除条目不影响当前会话（两次 deny 实证）；脚本级策略修改则每次调用即时生效。
  **禁止**用 refspec 拼写绕过正则。
- **17 条实验 workflow 是 commit-message 标签门控**（如 `[cpu-study]`、`[seasonal-study]`、
  `[real-smoke]`；全集见 `.github/workflows/r7-*.yml`）。普通 push 上它们显示 skipped 是
  **设计行为，不是失败**；需要运行某条就在 commit message 里带上它的标签。
  触发契约与判定纪律见 `docs/rules/ci-and-verification.md`。

## 按任务加载

| 当你要做… | 先读 |
| --- | --- |
| 引入新的真实数据 / 重建派生数据集 | `.agents/skills/real-data-acquisition/SKILL.md` |
| 跑一个有界的 CPU 实验并按证据归档 | `.agents/skills/bounded-study-run/SKILL.md` |
| 复现或重放一个历史产物 | `.agents/skills/pinned-artifact-replay/SKILL.md` |
| 把结果写进报告 / 声明可引用 | `.agents/skills/result-freeze/SKILL.md` |
| 换机器 / 缺依赖 / CUDA 不可用 | `.agents/skills/environment-rebuild/SKILL.md` |
| 排查 CI 失败或判断"算不算通过" | `.agents/skills/ci-workflow-triage/SKILL.md` |
| 开始/推进/关闭 issue | `.agents/skills/issue-lifecycle/SKILL.md` |
| 记一个决定（架构 / 约定 / 长期行为） | `.agents/skills/decision-record/SKILL.md` |
| 多步方案设计 / 迭代计划 / 大改动前风险评估 | `.agents/skills/planner-delegation/SKILL.md` |
| 写/推进一个 goal 目标（或 goal 模式不可用、不想用它） | `.agents/skills/goal-loop/SKILL.md` |
| 需要本仓以外的公开事实（库用法 / 版本 / 论文 / 报错） | `.agents/skills/web-research/SKILL.md` |
| 成品该放哪（计划 / 决策 / 目标 / 工具态） | `docs/rules/artifact-storage.md` |
| 新文件/模块/函数该叫什么 | `docs/rules/naming.md` |
| 全部能力清单与触发条件 | `docs/skills/README.md` |
| 改 CI / 加 workflow | `docs/rules/ci-and-verification.md` |
| 改数据契约 / 发布器 | `docs/rules/data-and-artifacts.md` |
| 写测试、加 fixture | `docs/rules/testing.md` |
| 需要全部规则清单与分类 | `docs/rules/README.md` |
| 阈值从哪来 / 迁移计划 | `docs/rules/MIGRATION.md` |
| 需要人拍板的问题 | `docs/rules/OPEN_QUESTIONS.md` |

## 规则与执行

- 规则正文见 `docs/rules/`，每条含级别、范围、依据（E-xxx）、现状分类、执行方式、例外。
- 机械可判定的规则由 `python tools/check_conventions.py` 检查（只读，标准库，纯 AST 判定）。
  - 默认只跑**阻断规则**（A/B 类，当前 37 条）：失败即代表违反契约。
  - `--report` 追加**C 类目标态与趋势**规则的报告，仅供参考，不阻断。
  - 自测（含"故意违规必须报错"的反证与结构性防回归）在 `tests/test_check_conventions.py`。
- 接入状态：已在 CI 中生效——`.github/workflows/ci.yml` 的 **Check repository conventions** 步骤。
  报告型规则仍不阻断。规则实现缺陷与未机械化项见 `docs/rules/MIGRATION.md`。

## 自动拦截（hooks）

`.zcode/config.json` 注册了 7 个 hook，脚本在 `tools/agent_hooks/`（纯标准库，全部 **fail-open**：
脚本自身出错时放行并打印原因，不会卡死会话）。自测在 `tests/test_agent_hooks.py`（含反证）。

| 时机 | 脚本 | 行为 |
| --- | --- | --- |
| PreToolUse | `guard_protected_paths.py` | **拒绝**对 `data/raw|interim|processed`、归档快照的写/删/移动（R-002/R-004/R-031），以及 **`tests/` 下已跟踪内容或名字像测试的文件的删除**（决策 0017）。读操作与正常命令一律放行 |
| PreToolUse | `guard_destructive_git.py` | **拒绝** force-push（含裸 `+ref`）、`reset --hard`、`clean -f`、`branch -D`、合并 main（`git merge` / `gh pr merge`）、删除默认分支、`--mirror`（决策 0003）。非 force 推送（含到 main 的 ff）放行；只读 git 一律放行 |
| PreToolUse | `guard_conventions_before_commit.py` | **拒绝**会让 CI 变红的 `git commit`：判 `git diff --cached` ∪ 同一命令行 `git add/rm/mv` 的操作数 ∪（`-a` 时）已修改的已跟踪文件（决策 0017）。合规提交与只读 git 一律放行 |
| PreToolUse | `guard_web_research_route.py` | **拒绝**主链路的 `WebSearch`/`WebFetch`（R-049，决策 0018）：拒绝文案给出替代路径（委派 `web-researcher`；不可用时 `curl` 降级）。`sess_subagent_` 前缀放行（防自锁）；Bash 不受影响 |
| PostToolUse | `check_model_digest_impact.py` | **提示**（不阻断）：改了 `model/**.py` 会改变 `model_code_sha256`，需标记 `[model-digest-change]` |
| PostToolUse | `note_external_fetch.py` | **提示**（不阻断）：`curl`/`wget` 出现外部 URL 时提醒按 R-050 记录 URL / 访问日期 / 失败原因 |
| Stop | `tools/check_conventions.py --quiet` | 收尾时跑 37 条阻断规则，有违规则要求先处理 |

边界与开关：

- **项目自己的下载器不受影响**（这是刻意设计的）：`python scripts/try_real_downloads.py`、
  `python -m data.download.arco_era5`、`python scripts/prepare_real_smoke.py` 等按脚本默认值写
  `data/raw/`，一律放行。被拦的只有**绕过脚本、直接在 shell 里改写/删除**这些目录的操作。
  若某条 deny 拦住了正当工作，删掉 `.zcode/config.json` 里对应条目即可。
- **测试只禁删除，且只禁"提交会包含的那些"**：新增、编辑、重构、运行测试都不受限；
  `rm`/`git rm`/`find -delete` 只在目标是**已跟踪内容**或**名字像测试**（`test_*.py`/`conftest.py`）
  时被拦（R-009 只统计总数且仅报告，看不见整文件删除）。**未跟踪且非测试名**的草稿可以清理——
  三轮交付里那个删不掉的遗留文件正是被旧的全禁规则锁住的（决策 0017）。
- **阻断判定按已跟踪集合计**：命中对象"自身及其下都无已跟踪内容"时**照常打印但不计失败**，
  于是「本地阻断失败」⟺「提交树上有违规」（`git add` 之后立刻恢复阻断）；git 回答不出来时不宽容。
- 保护清单从 `tools/check_conventions.py` 的 `ARCHIVAL_PREFIXES` **推导**，不另造平行清单；
  `tests/test_agent_hooks.py`（196 个用例，含"审计发现的绕过"反证）断言两者不会漂移。
- Stop hook 会**重复** CI 已有的检查。这是刻意的（本地收尾前就知道结果），不是替代 CI。
- **提交闸门是新增的"会拒绝"项**：它拦的是"这条提交会让 CI 变红"，被判路径与失败规则都会打印。
  配置在**会话启动时读取**，新增条目要**重启会话**才生效（脚本级修改即时生效）。
- **外部检索闸门只管主链路**：工具事件的 hook 载荷没有子智能体身份字段（E-186），且当前构建下
  子会话很可能不跑 hook（E-187，**推测**）。`sess_subagent_` 前缀放行是**防自锁**；
  `curl`/`wget` 永远只提示、不阻断——降级路径必须畅通（决策 0018）。
- 这些 hook 只在本仓库、经由 ZCode 生效；直接命令行操作不受影响。

## 成品存放

判据（R-035）：**一个新克隆是否需要它来理解、复现或审计这项工作？** 需要就提交，否则忽略。

| 产物 | 放哪 | 提交 |
| --- | --- | --- |
| 计划（工作态 / 定稿） | `.zcode/plans/` → `docs/plans/NNNN-<slug>.md` | 工作态否 / 定稿是 |
| 计划者委派返回的 JSON（工作态草稿） | 临时路径（如 `/tmp/`）；定稿仍按上一行整理 | 否 |
| 决策记录 | `docs/decisions/NNNN-<slug>.md` | 是 |
| 目标长文（goal 模式） | `docs/goals/<slug>.md` | 是 |
| 规则细则 / 证据 / 待决问题 | `docs/rules/` | 是 |
| 能力（SOP） | `.agents/skills/<name>/SKILL.md` | 是 |
| 工具运行态 | `.mimosa/`、`.zcode/plans/`、`outputs/`、`logs/` | 否 |

**治理层必须在版本控制内**（R-037）：CI 会跑其中的脚本、skill 会被自动加载 ——
未跟踪意味着干净克隆上直接失效。细则见 `docs/rules/artifact-storage.md`。

## 完成任务时必须报告

改动 / 实际执行过的验证 / 接口变化 / 数据与结果影响 / 安全与凭据变化 /
兼容性风险 / 依赖影响 / 未做的事 / 下一项任务。
