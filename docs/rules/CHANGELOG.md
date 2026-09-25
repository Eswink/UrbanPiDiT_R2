# 规则与能力变更史

每次引导或规则修订追加一条。不静默改写历史；被取代的规则标为 superseded 并保留引用。

## 2026-09-25 — GitHub 通道与 issue 关闭方式（用户授权的例外）

**范围**：`.github/workflows/*.yml` 的触发语义、对 `main` 的受控写入、issue 的自动关闭。
**来源**：用户 2026-09-25 明确指示——本项目**只用 `git`/SSH，不用 `gh`**；要求真正关闭
8 个 open issue；要求处置"大量 skipped workflow"的观感问题。
**落点**：`AGENTS.md` 新增「GitHub 通道与 issue 关闭」小节；
`docs/rules/ci-and-verification.md` 新增「Workflow 触发契约与 GitHub 通道」。

**记录的三件事**：

1. **skipped 是设计行为**：17 条实验 workflow 由 commit-message 标签门控
   （`[cpu-study]` 等全集见 `ci-and-verification.md`），普通 push 上 skip 不是失败。
2. **issue 自动关闭的唯一 git 路径**：closing keywords 必须落在**默认分支 main** 的提交里；
   实测 main 是工作分支的严格祖先（领先 165、落后 0），故 fast-forward 推送可行、
   force 仍被禁止。
3. **hook 例外的受控使用**：该 main 写入会被 `guard_destructive_git` 拦截；按 AGENTS.md
   既有的逃生口，临时移除 `.zcode/config.json` 里该 hook 条目、完成后立即恢复，
   全程记入决策记录。**禁止**用 refspec 拼写绕过匹配正则。

**代价（如实记录）**：main 到达分支顶端会使 PR #12 显示为 merged；main 的 push 不触发
`ci.yml`，验收证据仍以工作分支上的绿色 push run 为准。此条不改变 R-027/R-029/R-030
等任何既有规则的判据。

## 2026-09-25（补充）— hook 逃生口的会话边界实测

**范围**：`tools/agent_hooks/` 的生效时机、对 `main` 的受控写入执行方式。
**来源**：执行决策 0002 时的实测——`.zcode/config.json` 中 `guard_destructive_git`
条目在会话中途移除并验证生效后，`git push origin main` 与只读 dry-run 探测**仍被拒**
（两次 deny 后停止）。**发现**：ZCode 的 hook 定义在**会话启动时**加载，会话中途修改
config 不影响当前会话；用户级 `~/.zcode/config.json` 不存在，项目 config 是唯一来源。
**落点**：`AGENTS.md`「GitHub 通道」小节与 `docs/goals/open-issue-resolution.md` §4.3
补入会话边界；完整证据与可行路径（会话启动前移除条目 / 人在 ZCode 外终端执行 /
PAT）在 `docs/decisions/0002-*.md` 附录。
**代价（如实记录）**：本轮 8 个 issue 的 GitHub 关闭**未完成**（仍 open）；「会话内
移除→推送→恢复」顺序作废，逃生口仅当「条目移除先于会话启动」时可用。

## 2026-09-25（第二）— hook 策略：放行非 force 推送、合并仍需授权（用户指示）

**范围**：`tools/agent_hooks/guard_destructive_git.py` 的规则集、`tests/test_agent_hooks.py`、
`AGENTS.md`、`docs/rules/ci-and-verification.md`、`docs/rules/README.md`、
`docs/decisions/0002-*.md`（superseded）与新建 `0003-*.md`、
`docs/goals/open-issue-resolution.md`、`docs/R7_MANUAL_ITERATION.md`。
**来源**：用户 2026-09-25 明确指示——"需要允许推送，但是如果是合并需要用户明确告知
授权方可进行操作"；背景是决策 0002 的逃生口在会话内不可用（上一条目），goal 无法
自行完成关闭推送。

**变更**：

1. **放行**：非 force 推送到 main（`origin main`、`<分支>:main`、`HEAD:main` 等全部
   refspec 形态）——移除 `pushing to main` 规则。安全依据：GitHub 自身拒绝非 ff 推送、
   全部 force 变体仍被拒，main 只会线性前进到工作分支上已过 CI 的提交。
2. **保留拒绝**：合并——本地 `git merge` 涉及 main（本轮加固：`origin/main` 形态此前
   从未被拦截，一并纳入）、`gh pr merge`。授权路径：用户在 ZCode 之外终端执行，或
   会话启动前移除 hook 条目（会话边界实测仍成立）。
3. **新增拒绝**（放行 main 推送后必须堵的洞）：裸 `+main` force 推送
   （`_FORCE_PUSH_PLUS` 目的地改为可选）、删除默认分支（`:main` / `--delete main`，
   此前从未被任何规则拦截）、`--mirror` 推送（会强推重写 main）。
4. **测试**：`test_agent_hooks.py` 132 → 142 用例（push-to-main 迁入放行组并补三种
   形态；新增 force/删除/mirror 拒绝用例与 `origin/main` 合并拒绝）。全量
   885 passed, 3 skipped, 0 failed；34 条阻断规则 0 违规。

**代价（如实记录）**：①"合并需授权"约束的是 agent 执行的 merge **操作**，merge
commit 仍可经放行的 push 路径到达 main（ff 语义不区分）；②main 的 push 不触发
`ci.yml`，验收证据口径不变；③自主写 main 成为常态策略，防线是 ff 语义 + force 全拒
+ 决策 0003 记录的授权边界。此条不改变 R-027/R-029/R-030 等任何既有规则的判据。

## 2026-09-24（第七遍）— 命名约束

**范围**：为文件/模块/包/类/函数/常量/测试/文档/配置建立命名规则。
**验证**：34 条阻断规则 0 违规；检查脚本自测 83 个全绿；
全量测试 **787 passed, 3 skipped, 0 failed**（对比上一遍基线 759，新增 28 个命名相关用例）。

### 方法：先测量再立规

本项目纪律要求「每条规则必须能指向 E-xxx」。所以命名规则不是照搬惯例写出来的 ——
先对活跃区 **204 个 `.py`** 做了 AST 全量测量（归档目录与 `.venv` 除外，0 个解析失败），
再按实测结果决定哪些能立规。测量结论见 E-169 – E-181。

### 新增规则（11 条，`docs/rules/naming.md`）

| 规则 | 陈述要点 | 实测现状 |
| --- | --- | --- |
| R-038 | 模块/包文件名 snake_case | **204/204** |
| R-039 | 类名 PascalCase（允许前导 `_`） | **89/89** |
| R-040 | 函数/方法名 snake_case | **936/936** |
| R-041 | 模块级常量 UPPER_SNAKE（`__all__` 豁免） | **172/172** |
| R-042 | **标识符**纯 ASCII；注释不受限 | **0 命中** |
| R-043 | 代码路径禁用词与杂物桶名（`audit/` 豁免） | **0 命中** |
| R-044 | 测试命名（范围限 `tests/`） | **315/315** |
| R-045 | `r7` 主题标记出现在定界位置 | **129/129** |
| R-046 | 配置 snake_case.yaml / workflow kebab-case.yml | **10/10、18/18** |
| R-047 | 文档按目录分层命名 | **四层各自 100%** |
| R-048 | 参数名缩写（报告型） | **0 命中** |

阻断规则 24 → **34**；报告规则 9 → **10**；规则条目 36 → **47**。

### 一处假设被证据否定（如实记录）

立规前的假设是：**「`model/` 层规范做法是 `r7_` 前缀，其余写法不规范」**。
**实测否定了这个假设** —— `model/` 内部本身就混用三种方案：

| 方案 | 数量 | 例子 |
| --- | --- | --- |
| `r7_` 前缀 | 3 | `model/r7_baselines.py`、`r7_halting.py`、`r7_rollout.py` |
| `_r7` 后缀 | 3 | `model/process_forecast_r7.py`、`recursive_weather_r7.py`、`weather_forecaster_r7.py` |
| 无标记 | 14 | `model/process_reasoner.py`、`urban_pidit_r2.py` 等 |

所以「统一为前缀」这条规则**会被现有 5 个文件否决**。改证据支持的措辞为
「`r7` 必须出现在**定界位置**」（前缀或后缀均可），129/129 成立（E-176）。

**并且本轮不重命名 `model/` 下任何文件**：改 `model/` 字节会使 `model_code_sha256` 变化，
令 Q-009 记录的 3 条重放 workflow 无法加载归档 checkpoint —— 为"风格统一"付这个代价不值得。

### 三条规则在实现过程中被自己的门禁纠正

1. **R-047 初版写错**（E-180）：初版断言「`docs/rules/` 子目录一律 kebab-case」，
   门禁当场报出 4 例 —— `CHANGELOG.md`、`EVIDENCE.md`、`MIGRATION.md`、`OPEN_QUESTIONS.md`。
   核对后确认**是规则写错而非文件不合规**：该目录的台账类文件刻意沿用顶层 UPPER_SNAKE 惯例。
   规则改为分四层判定后 100% 成立。
2. **R-048 初版判据噪音过高**（E-181）：判据是"名字含 3 个以上连续辅音即报告"，
   实测命中 **113 处**，包含 `mlp_ratio`、`model_cfg`、`training_std` 这类完全清晰的名字。
   **一份噪音率这么高的报告会被直接忽略，比没有报告更糟** —— 该判据被撤回，
   收窄为"单个无信息 token"，实测 0 命中。并用测试
   `test_r048_stays_quiet_on_clear_names` 锁住边界，防止判据再被放宽。
3. **R-041 与 R-045 的判定实现有缺陷**（由新增的反证测试抓到）：
   - R-041 原条件写成 `name.isupper() and not fullmatch(...)`，而 `MyConst.isupper()` 为 False，
     导致混大小写的常量名漏判；且初版误用 `ast.walk` 遍历全部节点，
     把函数内的局部变量（`drafts = []`）当成模块级常量 —— 已改为只遍历 `tree.body`。
   - R-045 原正则 `r7[_a-z0-9]*` 把 `r7x_helper` 也算作"前缀"，
     已有字母紧跟时不算定界位置 —— 已改为 `r7(?:_[a-z0-9]+)*`。

### 范围与例外的实测依据（三处必须写清，否则会误报）

- **R-044 必须限定 `tests/`**：`tests/` 之外有 6 个 PyTorch-Lightning 协议方法以 `test` 开头
  （`test_step` ×4、`test_dataloader` ×2），规则若写成"以 test 开头的函数"会全部误报（E-175）。
- **R-042 只能查标识符**：43/204 个文件含中文注释/docstring，这是项目刻意的写法；
  若用整行正则会全线误报（E-173）。
- **R-043 必须豁免 `audit/**`**：9 个 `final_*` 是冻结的证据文件名，改名会断引用（E-174）。

### 明确不做

- 不重命名 `model/` 下任何文件（digest 代价，见上）。
- 不重命名 `audit/` 下的 9 个 `final_*`（冻结证据）。
- 不把 `docs/rules/*.md` 改成 UPPER_SNAKE（它们是自洽的 kebab regime，10/10 合规）。
- 不为 `v6`/`legacy_` 家族立规 —— 活跃区 0 个成员，立规会约束空集。

## 2026-09-24（第六遍）— 成品存放约定 + 修复治理层未跟踪

**范围**：建立 plan / decision / goal / 工具态的存放约定；修复治理层未被版本控制的漏。
**验证**：24 条阻断规则 0 违规；检查脚本自测 55 个全绿、hook 自测 123 个全绿；
全量测试 **759 passed, 3 skipped, 0 failed**（对比上一遍基线 745 passed，新增 14 个测试）。

### 发现并修复的最重要问题：治理层全部未被跟踪（E-159）

实测 `git ls-files` 为空：`AGENTS.md`、`docs/rules/*`（13）、`.agents/skills/*`（7）、
`tools/check_conventions.py`、`tools/agent_hooks/*`、对应测试 —— 全部未跟踪。

**而 `.github/workflows/ci.yml` 已经在跑 `python tools/check_conventions.py`。**
也就是说：**干净克隆上 CI 必然失败**，本地却因为文件都在而一切正常。
这类"只在别人机器上失败"的问题，正是门禁该拦的。

**处置**：新增 R-037（阻断）机械检查治理层是否被跟踪；并把治理层纳入版本控制。
R-037 上线时**立即报出 11 处违规**，按设计抓到了这个漏。

### 新增规则（6 条，`docs/rules/artifact-storage.md`）

核心判据是一句问话：**一个新克隆是否需要它来理解、复现或审计这项工作？**
需要 → 提交；不需要 → 忽略。

| 规则 | 陈述要点 | 分类 |
| --- | --- | --- |
| R-032 | 计划分两层：工作态 `.zcode/plans/`（忽略）→ 定稿 `docs/plans/`（提交，含「实际结果」） | B |
| R-033 | 决策一文件一编号：`NNNN-<slug>.md`，含 Context/Decision/Consequences，**必须写负面后果**，编号不复用 | B |
| R-034 | goal 长文放 `docs/goals/`，objective 只放要点与路径 | 约定 |
| R-035 | 工具运行态不进版本控制（判据同上，不逐个工具列名单） | A |
| R-036 | `superseded` 决策的取代目标必须存在（防悬空） | A |
| R-037 | 治理层必须在版本控制内 | B |

阻断规则 20 → **24**；报告规则 7 → **9**；规则条目 30 → **36**。

### 新增目录与文件

- `docs/plans/`（+ README 说明两层结构与归档要求）、首份归档 `0001-artifact-storage-convention.md`
- `docs/decisions/`（+ README 含索引与与 OPEN_QUESTIONS 的分工）、首份记录 `0001-artifact-storage-convention.md`
- `docs/goals/`（+ README 说明 goal 模式的 4000 字符上限与写法要求）
- `.agents/skills/decision-record/`（首个 ADR 技能：何时写、模板、状态生命周期、检查点）

### 外部惯例对照（结论：只补真缺口，不重复建）

| 外部惯例 | 本仓库已有 | 判断 |
| --- | --- | --- |
| ADR（Nygard / MADR：一决策一文件、编号不复用、superseded 不删） | 决策散在四处、无统一形式 | **真缺口** → 建 `docs/decisions/` |
| spec-kit / spec-persistence（flow-forward：旧目录保留供审计） | 只有会话态计划，无归档层 | **缺归档层** → 建 `docs/plans/` |
| Memory bank（Cline 六文件 / Claude `MEMORY.md` + 主题文件） | AGENTS.md + docs/rules + skills 已实现同一目标且分层更干净 | **不建**（会制造第二处真相） |
| GitHub `Global/Agents.gitignore`、DVC、`targets` | `.gitignore` 注释已表达三层划分，但非编号规则 | **提升为规则**（R-035） |

**未建 `docs/memory/` 的理由**：ZCode/Claude 的 memory 按设计存放在
`~/.zcode/cli/memories/projects/<id>/memory/`，**机器本地、不跨机器共享**（E-162）；
仓库内的 AGENTS.md（常驻契约）+ docs/rules（细则）+ .agents/skills（流程）
已经实现了外部 memory 方案想要的「小索引常驻 + 细节按需加载」。

### `.gitignore` 调整

新增 `!.zcode/workflows/`（放行 agent workflow 定义；客户端按固定路径发现，
放平级目录会变成"文件存在但按名调不到"）。`.zcode/plans/` 保持忽略。

**已实测**（E-163）：用隔离临时仓库同时验证 `check-ignore` 与真实 `git add` 行为 ——
`dir/*` 排除内容后**可以**对子目录用 `!` 放行。这与第二轮"文件级放行需 `dir/*`"的教训
不冲突：子目录与文件的重包含规则不同。

### 顺带修正的两个既有缺陷（如实记录）

1. `docs/rules/EVIDENCE.md` **自身有编号复用**：`E-004` 与 `E-009` 各被用于两个不同发现
   （E-165）。已改为唯一编号（新条目 E-167、E-168 —— 初次沿用 E-144/E-145 时与第二遍冲突，
   被 `EVIDENCE.md` 的编号唯一性自检当场发现，随即改正），并确认无其它文件引用其复用含义。
2. `docs/rules/repository-layout.md` 曾称 `.mimosa/` 未进 `.gitignore`，实测已被忽略（E-166）——
   该陈述过时，随本轮目录表更新一并修正。

### 边界（本轮明确不做）

- **不建 `docs/memory/`**（理由见上）。
- **不建 `docs/workflows/`**（客户端要固定路径）。
- **不批量搬迁历史决策**到 `docs/decisions/`（会制造副本且无法与原文同步）；
  历史决策由 `OPEN_QUESTIONS.md` 在解决时自然补入。

## 2026-09-24（第五遍）— hook 覆盖率审计与绕过修复

**范围**：对第四遍的 hooks 做对抗性审计（含独立子代理复核），修复实测出的绕过与假阳性。
**验证**：hook 自测 **123 个全绿**（原 86）；全量测试 **745 passed, 3 skipped, 0 failed**；
约定门禁 20 条阻断规则 0 违规。

### 先确认的事：项目自己的下载器**没有**被 hook 拦住

审计的第一问是"`data/raw` 保护会不会打断下载脚本"。实测结论：**不会**。
`scripts/try_real_downloads.py`、`scripts/prepare_real_smoke.py`、
`python -m data.download.arco_era5`、`python -m data.download.worldcover_cog` —— 全部放行。

原因是 hook **只检查命令字符串里的操作数与重定向目标**：`python scripts/x.py` 这个形态里
没有任何 `data/raw` 字面量，所以既不误拦、也确实拦不住"脚本内部写 data/raw"。
这个边界是刻意的：脚本按自己的默认值写 `data/raw` 是**设计行为**（R-004 约束的是
"就地改写/删除既有内容"，不是"禁止这些目录存在新文件"）。

顺带修正了 deny 文案：原文案说"改用新目标路径"，与项目自己的下载器默认值直接矛盾。
现在明确写出"**不拦**这些下载命令"，并只对绕过脚本的 shell 改写/删除给出替代做法。

### 修复的绕过（24 处，全部实测过）

审计用独立子代理逐条复现，以下均**实测为未拦**，修复后逐一验证：

| 类别 | 具体形态 |
| --- | --- |
| 路径归一化 | `data/./raw/x`、`data//raw/x`、`data/RAW/x`（大小写）、`data/ra*`（glob 部分拼写） |
| 相对路径逃逸 | `cd data/raw && rm source.csv`（cd 进保护目录后用相对路径） |
| 重定向变体 | `2>file`、`1>file`、`>|file`（fd 前缀与 noclobber 形式） |
| 外部程序输出 | `curl -o`、`wget -O`、`sort -o`、`dd of=` |
| 就地编辑 | `sed --in-place`（只认 `-i`）、`perl -i` |
| 递归删除 | `find ... -delete`、`find ... -exec rm` |
| git 变体 | `git -C <repo> rm`（全局选项吞掉了子命令）、`git worktree add` |
| 复制写入 | `cp x legacy_v6/f.py`（目标是文件，非目录——原先的 `endswith` 判断漏掉） |
| 解释器内联 | `python -c "open('data/raw/x','w')"`（需同时含写原语与保护路径才拦） |

### 同时修正的假阳性

`echo "see > data/raw/x"`、`grep -rn 'foo > data/raw/x' docs/` 这类**只是在引号里提到**
重定向的只读命令，原先会被误拦。现在先屏蔽引号内文本再解析重定向。

### 新增：`tests/` 删除保护

**背景**：R-009 统计测试函数与断言总数，但它是**聚合且仅报告**的——删掉一个测试文件，
只要总数还没跌破基线就不会被看见；而基线是手工维护的，很容易顺势往下调。

**处置**：`guard_protected_paths.py` 新增一条**只管删除**的检查——
`rm`/`rmdir`/`unlink`/`shred`/`git rm`/`find -delete` 对 `tests/` 下文件的操作被拒。
**新增、编辑、重构、运行测试完全不受限**（已用 5 个放行用例断言）。
这是把 `DELETION_ONLY_PREFIXES` 与 `PROTECTED_PREFIXES` 分开的原因：前者只判删除，
后者对所有写操作生效。

### 本轮修正的自身缺陷（如实记录）

1. `_DEST_ONLY_VERBS` 分支要求目标路径 `endswith` 保护前缀，导致 `cp ... legacy_v6/f.py`
   （目标是文件）漏判。改为只检查**最后一个操作数**（cp/rsync 的语义就是"目标是最后一个参数"）。
2. `_mentions_protected` 预检过窄，glob 形态（`data/ra*`）在预检就被短路，
   导致 `_path_is_protected` 里已实现的 glob 判断永远执行不到。
3. 重构时把 `sed` 分支的条件顺序写错，一度让 `cp` 与 `sed --in-place` 同时回归——
   由既有的参数化测试当场抓到。

### 仍未覆盖的（如实声明，不夸大）

**`python script.py` 内部写 `data/raw` 拦不住**，这是文本级 hook 的原理性上限：
hook 看到的是命令字符串，不会执行脚本。要覆盖它需要进程级监控或文件系统层只读挂载，
两者都超出当前机制范围。同理，`bash -c "..."` 间接、变量展开（`rm -rf $DIR`）、
符号链接逃逸也无法在文本层可靠判定。

**结论**：这些 hook 是"防手滑与防绕过"的护栏，**不是沙箱**。真正的强制执行仍是
CI 的 `check_conventions.py` 与代码自身的身份校验。

## 2026-09-24（第四遍）— 项目专属 hooks 与 skills

**范围**：新增自动拦截 hooks、修正 skill 发现路径、新增 3 个 skill、调整 `.gitignore`。
**验证**：hook 自测 **86 个全绿**（含反证）；全量测试 **708 passed, 3 skipped, 0 failed**
（对比建立虚拟环境后的基线 622 passed，新增 86 个 hook 测试）；
约定门禁 20 条阻断规则 0 违规。

### 发现并修正的问题：4 个 skill 此前从未生效

第一遍把 skill 写在 `docs/skills/`，但 ZCode 的 skill 发现路径是
`~/.zcode/skills`、`~/.agents/skills`、`<repo>/.zcode/skills`、`<repo>/.agents/skills`
—— `docs/skills/` **不在其中**。那 4 个 skill 因此只是文档，模型不会按任务自动加载。

**处置**：移到 `<repo>/.agents/skills/`（未被 gitignore，且是真正的发现路径）。
迁移前后用 sha256 逐文件核验内容一致（4/4 MATCH）。`docs/skills/README.md` 保留为索引。

放 `.zcode/` 不行：上一遍把 `.zcode/` 加进了 `.gitignore`，skill 放那里永远不会被提交。

### `.gitignore` 调整（含一个 git 语义陷阱）

`.zcode/` → `.zcode/*` + `!.zcode/config.json`。

**原因**：`.zcode/` 是**目录级**排除，git 规则下无法用 `!` 重新包含其子文件。
必须写成 `.zcode/*`（只排除内容）才能放行单个文件——这与本文件已有的
`data/raw/*` + `!data/raw/.gitkeep` 是同一手法。

**已逐条核验**：`.zcode/config.json` → 会提交；`.zcode/plans/`、`.zcode/skills/` → 仍忽略。

### 新增执行面：3 个 hook + 1 个 Stop 门禁

配置在项目级 `.zcode/config.json`（`hooks.enabled: true`）。脚本在 `tools/agent_hooks/`，
**纯标准库、全部 fail-open**（自身出错时放行并打印原因，避免门禁坏了卡死会话）。

| Hook | 事件 | 行为 | 依据 |
| --- | --- | --- | --- |
| `guard_protected_paths.py` | PreToolUse | deny 对 `data/raw｜interim｜processed` 与归档快照的写/删/移动 | R-002, R-004, R-031 |
| `guard_destructive_git.py` | PreToolUse | deny force-push / `reset --hard` / `clean -f` / `branch -D` / 对 `main` 的写 | AGENTS.md 硬约束, `R7_MANUAL_ITERATION.md:15` |
| `check_model_digest_impact.py` | PostToolUse | 提示（不阻断）`model/**.py` 改动会使 `model_code_sha256` 变化 | E-146, E-147, Q-009 |
| `tools/check_conventions.py` | Stop | 收尾跑 20 条阻断规则 | 既有规则集 |

设计要点：

- 保护清单从 `check_conventions.ARCHIVAL_PREFIXES` **推导**，`tests/test_agent_hooks.py`
  断言两者不漂移（避免两处清单各自演化）。
- **只拦写类操作**；`Read`、`grep`、`git log/diff/status` 等一律放行——只读 git 是本项目的
  主要证据来源（R-027 甚至抽样 `git log`）。
- 每个 deny 都**给出替代做法**，不只是一句禁止。
- Stop hook 刻意与 CI **重复**：本地收尾前即知结果，不替代 CI。

### 新增能力（3 个）

| skill | 触发条件 | 重复证据 |
| --- | --- | --- |
| `environment-rebuild` | 换机器/容器、`ModuleNotFoundError`、CUDA 不可用 | 2026-09-24 实测重建；4 个未声明依赖 + torch≥2.8 不兼容（E-157, Q-011, Q-012） |
| `ci-workflow-triage` | CI 失败、运行 pending/取消、判断"算不算通过" | 18 条 workflow；`R7_MANUAL_ITERATION.md:17-21`；`R7_TASK_QUEUE.md` |
| `issue-lifecycle` | 开始/推进/关闭 issue，判断能否标 DONE | `R7_TASK_QUEUE.md:59` + 6 个真实 issue（#53–#58） |

**未建的**（证据不足，标准与第一遍一致）：控制器标定→策略选择链、成对 block-bootstrap、
checkpoint 恢复协议、边界/ACC 指标流水线——均只重复过工具、未重复过协议。

### 本轮修正的自身缺陷

hook 自测第一轮抓到 2 个真实判定缺陷（均为实现错误，非规则放宽）：

1. `_REDIRECT` 正则写错（`(?:^|[^0-9\s])>` 要求前导字符，而 `> path` 的 `>` 位置不匹配），
   导致 `echo x > data/raw/f` 这类重定向漏判 → 改为 `(?<![0-9])>>?`。
2. `sed -i` 分支逻辑写反（先置 `is_git_write = False` 又落到 `elif verb == "sed": continue`），
   导致 `sed -i` 原地改写归档漏判 → 改为在 `sed` 分支内先判 `-i` 再检查目标路径。

## 2026-09-24（第三遍）— 建立项目虚拟环境并修复 GPU 路径缺陷

**范围**：创建 `.venv`、安装全部依赖、配置自动激活、修复 `--device cuda`。
**结果**：全量测试 **622 passed, 3 skipped, 0 failed**（含新增 3 个 GPU 回归测试）。

### 环境

- 新建 `.venv`（Python 3.12.3），`torch==2.11.0+cu128`，实测 CUDA 12.8 可用、2×RTX 3090。
- 自动激活：`~/.bashrc` 新增 `project venv auto-activation` 块（基于 `PROMPT_COMMAND`），
  进入受管项目目录自动 `source .venv/bin/activate`，离开自动 `deactivate`；
  手动激活的其它 venv/conda 环境不被覆盖；重复 `source .bashrc` 不会累积钩子。
  原 `.bashrc` 已备份为 `~/.bashrc.bak-20260924-113147`。
- 依赖缺口：`icechunk`、`pcodec`、`numcodecs`、`pip` 在代码/测试中被使用，
  但三份 requirements 与 pyproject 均未声明（CI 是 workflow 级临时 pin）。
  已记录于新增的 `docs/rules/environment.md`，并开 Q-011。

### 修复的代码缺陷（1 处，真实且长期存在）

`training/r7_experiment.py` 的 `select_device` 把裸 `torch.device('cuda')` 传给
`torch.cuda.set_device()`，而 **torch≥2.8 要求整数索引或带索引的 device**，否则抛
`ValueError`。后果：**所有 `--device cuda` 入口在真有 GPU 的机器上都会失败**。

- **为何长期未发现**：唯一相关测试 `test_cuda_unavailable_never_falls_back` 用 monkeypatch
  把 CUDA 设为不可用，在到达 `set_device` 前就抛错，**CUDA 可用路径从未被覆盖**；CI 又是 CPU-only。
- **已核验非本轮引入**：conda base 的 torch 2.8 行为完全相同，说明该缺陷早于虚拟环境搭建。
- **修法**：无索引时回退到 `torch.cuda.current_device()`，保留 `CUDA_VISIBLE_DEVICES` 语义。
  **未绕过任何校验、未改变模型语义、`model_code_digest()` 不变**。
- **新增回归测试 3 个**（无 GPU 时自动 skip）：裸 `cuda` 在真实设备上可用、
  `cuda:0` 仍可用、bf16 不支持时正确报错。另加 1 个非 GPU 的非法设备类型测试。
- 已开 Q-012：需确认是否有历史结论声称在 GPU 上跑过而实际走了失败路径。

### 账目同步

- `R-009` 基线由 280/560 更新为 **283/563**（新增 3 个测试函数与断言）。
- 新增 `docs/rules/environment.md`（环境与依赖缺口的记录性文档，非规则）。
- `AGENTS.md` 新增「环境」小节（优先用 `.venv/bin/python`、requirements 不完备、CI 与本地差异）。

## 2026-09-24（第二遍）— 解决全部待决问题 + 关闭机械化欠账

**范围**：修复规则实现缺陷、落地 Q-001–Q-008 决定、把欠账规则机械化、接入 CI、清理长行。
**验证**：20 条阻断规则 0 违规；41 个自测全绿（含 24 个反证 + 6 个结构性防回归）；
全量可运行测试 **492 passed**（对比基线 451），**0 回归**（两侧同为 29 failed / 3 skipped / 18 errors，
全部为缺失可选依赖 `xarray`/`h5netcdf`/`pytorch_lightning`，非本次改动引起）。

### 修正的规则实现缺陷（5 处，均为判定实现错误，非规则放宽）

| # | 缺陷 | 影响 | 修正 |
| --- | --- | --- | --- |
| 1 | `r_002_archival_readonly` **定义了两次**（AST 版 + 文本版） | Python 取后定义 ⇒ R-002 实际跑的是被 CHANGELOG 声称已废弃的文本匹配版，字符串字面量可骗过它 | 删除后定义与占位符，统一为 AST 版 |
| 2 | `r_018_os_path` **定义了两次**（逐行版 + AST 版） | 同上，取后定义的 AST 版；前定义是死代码 | 删除前定义 |
| 3 | R-002 只认 `legacy_v531` 字面量 | 新增 `legacy_v6/` 归档后会出现漏判 | 引入 `ARCHIVE_PATH_TOKENS`，新归档只改一处 |
| 4 | R-024/R-008 依赖文本匹配 | 会被字符串字面量骗过（新增测试文件本身曾触发 R-018 假阳性） | 全部改为 AST 判定 |
| 5 | 6 条已实现规则不在任何执行分组 | 写了但永不运行（写等于没写） | 并入分组；并新增自测断言"无未分组规则" |

新增结构性防回归测试：`test_every_check_function_is_defined_exactly_once`（用 AST 而非 grep 检测重复定义）、
`test_every_rule_target_is_defined_once_and_callable`、`test_every_implemented_rule_has_an_execution_group`、
`test_blocking_and_report_groups_are_disjoint`、`test_exception_lists_use_exact_paths`。

### 新机械化的规则（5 条）

| 规则 | 检查内容 | 分类 |
| --- | --- | --- |
| R-009 | 测试函数/断言数不得低于记录基线（280/560） | 报告 |
| R-027 | 最近 30 条提交遵循 Conventional Commits | 报告 |
| R-028 | 消费归档产物的 workflow 必须真正禁用 socket | **阻断** |
| R-029 | 每个 workflow job 必须设 `timeout-minutes` | **阻断** |
| R-030 | 有界实验必须有内部墙钟截止 | 报告（2 处例外见下） |

阻断规则 18 → **20** 条；未机械化规则 8 → **3** 条（仅剩 R-003、R-011、R-026，均为语义判断）。

### 待决问题的决定（Q-001 – Q-008 全部关闭）

| 问题 | 决定 | 落地 |
| --- | --- | --- |
| Q-001 | 归档到根 `legacy_v6/` | 新建目录 + README；`pyproject.toml` exclude、`ARCHIVAL_PREFIXES`、`AGENTS.md` 同步；833 行内容逐行比对一致（仅补结尾换行） |
| Q-002 | 3 保留 + 3 移入 `scripts/` | `train_r7{,_process,_recursive}.py` 移入 `scripts/` 并补 repo-root 路径修正；3 个诊断入口留在根目录（测试按裸文件名调用）并补进 `py-modules` |
| Q-003 | 启用 `--strict-markers` | `pytest.ini` 注册 `gpu`/`network`/`slow` 并加严格模式（当前仅用内置 marker，安全） |
| Q-004 | 补 `deterministic` 字段 | `r7_evaluate.py` provenance 增加 `deterministic` 与 `determinism_scope`；不引入假 seed |
| Q-005 | `.mimosa/` 进 `.gitignore` | 已加入，并同时加入 `.zcode/` |
| Q-006 | 改造而非删除空白门禁 | `git diff --check` → `git show --check --format= HEAD`（在干净 checkout 上真正生效） |
| Q-007 | README 加主线指针 | 顶部加 4 行指向 `AGENTS.md` 与 `docs/r7_TASK_QUEUE.md` |
| Q-008 | 三份依赖清单补注释 | 各加头部说明（适用场景/是否被 CI 安装/与 pyproject 的对应）；不改依赖内容 |

### 代码整理

- **长行**：18 处 >200 字符全部处理（13 个文件）。**每处都用 AST 结构比对证明语义一致**
  （`ast.dump` 不含行号，哈希相等即证明等价），全部 MATCH。R-019 命中 18 → **0**。
- **归档**：`training/legacy_physical_consistency.py`（833 行、无人导入）移入 `legacy_v6/`。R-021 命中 1 → **0**。
- **测试文件**：`tests/test_check_conventions.py` 曾达 417 行（自己触发 R-021），改用 `parametrize` 重构至 320 行。

### model_code_sha256 变化（重要）

`model/` 下 5 个文件的字节变化使 `model_code_digest()` 从
`20196c647386eb3f092d51b7ebf1fbe4c687971dc522bf63a9013b7214eb7994`
变为
`d9fb07f2d38d41d681b45c0b4d539edb8f0f9620f44ba0c426c40b43d672f665`。

**这是预期后果，不是缺陷**。`load_checkpoint`（`training/r7_experiment.py:124`）会因此拒绝加载
在旧实现下产生的 checkpoint。受影响的重放路径有 3 条 workflow：`r7-restored-diagnostic.yml`、
`r7-correction-audit.yml`、`r7-extended-control.yml`。

处置方式遵循项目既有政策（`docs/R7_CPU_REFINEMENT_RESULTS.md`："Model-source digest legitimately
changed. Use archived code.zip for older checkpoints; do not bypass integrity checks."）：
**未修改任何校验逻辑、未放宽 digest 比较**，改为在这 3 篇对应文档中记录该约束与新 pin，
要求旧产物用其归档的 `code.zip` 重放。见 `docs/rules/OPEN_QUESTIONS.md` Q-009。

### 执行面接入状态变更

`tools/check_conventions.py` **已接入 CI**：`.github/workflows/ci.yml` 新增
`Check repository conventions` 步骤（阻断规则，约 1 秒）。`compileall` 范围补上 `scripts/`。

## 2026-09-24 — 首次引导（阶段 0–9）

**侦察范围**：全仓只读侦察。活跃区 205 个 `.py`（20,494 行）、归档区 202 个 `.py`（46,890 行）、
67 个测试文件（262 个测试函数）、44 个 `docs/*.md`、18 个 CI workflow。
未运行测试套件、未安装依赖、未下载数据、未做任何 git 写操作。

**新增规则（20 条）**

| 规则 | 陈述要点 | 分类 | 执行方式 |
| --- | --- | --- | --- |
| R-001 | 新数据输出排他创建 | A | 脚本 |
| R-002 | 归档快照只读 | A | 脚本 |
| R-003 | 派生数据必须可重建 | A | 人工自觉 |
| R-004 | 原始/派生数据目录禁止原地写入 | A | 脚本 |
| R-005 | 本地数据集必须经 BUILD_COMPLETE 发布契约 | A | 脚本 |
| R-006 | 批量实验先冻结协议再训练 | B（3 例外） | 脚本 |
| R-007 | 实验产物必须带非科学声明 | B | 脚本 |
| R-008 | 下载失败必须留审计，禁止合成回退 | A | 脚本 |
| R-009 | 不得弱化判据以过门禁 | A | 人工自觉 |
| R-010 | 跑实验的 workflow 必须归档代码身份 | B（3 例外） | 脚本 |
| R-011 | 随机性显式且可记录 | A | 人工自觉 + 测试 |
| R-012 | 原始数据与产物不得进版本控制 | A | 脚本 |
| R-013 | 文本 IO 必须显式 encoding | A | 脚本 |
| R-014 | 禁止硬编码宿主绝对路径 | A | 脚本 |
| R-015 | 禁止裸 except | A | 脚本 |
| R-016 | 出网客户端只允许在下载层 | A | 脚本 |
| R-017 | 库与脚本启用 postponed annotations | B（6 例外） | 脚本 |
| R-018 | 路径操作使用 pathlib | B（4 例外） | 脚本 |
| R-019 / R-019b | 行长目标 120 / 硬上限 200 | C | 脚本（报告） |
| R-020 | 函数体 ≤100 | C | 脚本（报告） |
| R-021 | 单文件 ≤400 | C | 脚本（报告） |
| R-022 | 嵌套 ≤5 | C | 脚本（报告） |
| R-023 | 参数 ≤8 | C | 脚本（报告） |
| R-024 | 测试只写 tmp_path | A | 脚本 |
| R-025 | 测试禁止真实网络客户端 | A | 脚本 |
| R-026 | 每个判据必须有反证 | A | 人工自觉 |
| R-027 | commit 信息规范 + issue 引用 | B（1 历史例外） | 人工自觉 |
| R-028 | 离线实验必须真正禁网 | A | 人工自觉 |
| R-029 | workflow 必须设超时 | A | 人工自觉 |
| R-030 | 长任务必须有内部截止时间 | B（1 例外） | 人工自觉 |
| R-031 | 活跃代码不得 import 归档快照 | A | 脚本 |

说明：分类统计以 `docs/rules/README.md` 的汇总表为准（20 条进入 README 表，另 10 条以约定形式记录在
各自文件里）。**没有**任何规则被标为 superseded——这是首次引导。

**新增能力（4 个）**

- `docs/skills/bounded-study-run/SKILL.md`（11 个 workflow + 8 个模块同一形状）
- `docs/skills/pinned-artifact-replay/SKILL.md`（3 个 replay 模块 + 3 个 workflow）
- `docs/skills/real-data-acquisition/SKILL.md`（preflight 被每个 pilot 复用）
- `docs/skills/result-freeze/SKILL.md`（3 个迭代的记录形态）

**新增执行面**

- `tools/check_conventions.py`：只读、标准库、24 条规则实现（12 阻断 + 6 报告 + 6 其他）
- `tests/test_check_conventions.py`：27 个测试，其中 19 个是"故意违规必须报错"的反证

**新增账目面**

- `docs/rules/EVIDENCE.md`（143 条台账）
- `docs/rules/MIGRATION.md`
- `docs/rules/OPEN_QUESTIONS.md`（8 个待决问题）
- `docs/rules/CHANGELOG.md`（本文件）

**反向校验结果**（阶段 6）

- **18 条**阻断规则在真实仓库上 **0 违规**（3 处 R-006 与 3 处 R-010 例外被显式标记为 tolerated）；
- 6 条 C 类规则共命中 206 处，**未**接入阻断；
- 自测 29 个用例，其中 20 个是"故意违规必须报错"的反证，另含范围边界用例；
- 校验过程中修正了 5 个判定缺陷（详见下方"更正"）。

**更正记录（如实记录自己的判断错误）**

1. 初版 R-006 判定用"首个匹配行"比较，把辅助函数里的调用误判为训练调用，报出 5 处假违规。
   修正为按顶层函数作用域比较。
2. 随后发现 `ast.unparse` 整段函数时会把函数定义行也匹配为"写入点"，
   导致"先训练后写协议"的违规被漏判（反证测试捕捉到）。改为只检查 Call 节点。
3. R-002 初版判定方向错误：它去扫描归档目录**内部**的写操作，而归档快照本身是旧项目源码，
   含写操作是正常的。修正为"活跃代码不得写入归档路径"。
4. R-018/R-024/R-008 初版用文本匹配，会被字符串字面量骗过（本次新增的测试文件本身
   就触发过 R-018 假阳性）。全部改为 AST 判定。
5. R-008 初版范围过宽（把"模块里有 try"当作违规），报出 8 处假违规；收窄为
   "静默吞掉失败（既不重新抛出、也不留审计痕迹）"，并显式排除 `ImportError` 依赖兼容垫片。
   新增 2 个边界用例锁定该范围。

以上 5 处均为**判定实现缺陷**，不是规则放宽。修正后规则范围与阈值未变。

**执行分组修正**：首次自检发现 R-002/R-004/R-005/R-008/R-024/R-025 虽已实现却不在任何
执行分组中——这意味着它们永远不会被运行（写了等于没写）。已全部并入阻断集，
阻断规则由 12 条增至 18 条，并实测全部通过。

**未做的事**（详见交付报告）

- 未运行 `pytest`（除新增的自测文件）；未安装任何依赖；未联网；未做 git 写操作；
- 未修改任何既有项目源码、测试、数据或配置；
- 未在 CI 中接入 `tools/check_conventions.py`（执行方式如实写为"人工运行"）。
