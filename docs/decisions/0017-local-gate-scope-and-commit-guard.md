# 0017 本地闸门的判定范围（按已跟踪集合）与提交前闸门

- **日期**：2026-09-28
- **状态**：accepted
- **代码 SHA 范围**：`6fec347`（起点）→ 本次提交
- **依据证据**：`docs/R7_ZCODE_GOAL_VERIFIER_ABORTS.md`（调查的另一半）；`tools/check_conventions.py`；
  `tools/agent_hooks/guard_protected_paths.py`；`tools/agent_hooks/guard_conventions_before_commit.py`；
  `tests/test_check_conventions.py`（含 tmp git 仓库反证）；`tests/test_agent_hooks.py`；
  `docs/rules/CHANGELOG.md`

## Context

三轮交付（round one/two/three）都记录了同一条本机卫生问题：`tests/fixtures/r7_equivalence_recipe.py`
未跟踪、未提交、R-044 阻断命中，而 `guard_protected_paths` 拒绝一切 `tests/` 下的删除，于是它删不掉、
红灯一直在。用户在一次会话里看到「`2 failed` + `blocking rules=34 failing=1`，但提交照样成功」，
问这是 hooks 坏了还是规则不对。查证结果（全部实测）：

1. **`.git/hooks/` 里 0 个活动 hook** —— git 层根本没有闸门。
2. ZCode 的四个 hook 分工里，唯一跑惯例的是 **Stop**，它在**回合结束**时执行、且是 fail-open；
   `PreToolUse` 只拦「保护路径」与「破坏性 git」，**提交不在其中**。
3. 那条唯一的阻断命中打在一个**未跟踪**文件上；把 HEAD 取成干净工作树实测
   `blocking rules=34 failing=0`、`test_check_conventions.py 83 passed` —— **CI 是绿的，本地红是假的**。
4. 假红是**结构性**的：规则扫**工作目录**（含未跟踪垃圾），CI 扫**提交树**。而常态红灯会被无视，
   这正是用户察觉到的后果。

## Decision

**采用：阻断判定按「git 能看到的内容」计；删除保护按「提交能不能包含它」计；并在 `git commit` 前
加一道判「即将提交的内容」的闸门。** 具体：

1. **阻断判定作用域 = 已跟踪集合**（`tools/check_conventions.py`）：命中对象的路径**自身及其下**
   都没有已跟踪内容时，该命中标 `tolerated` —— **照常打印**（detail 追加 `[untracked: not part of
   any commit]`），但**不计失败**。包在**每一条规则**外面（`RULES` 注册处统一包装，避免手维护子集
   清单漂移）。git 不可用/失败 → `None` → **不做任何宽容**（tmp 树里的单元反证因此保持严格）。
   暂存让文件进 index → 命中立即恢复阻断（「staging 即试金石」）。
2. **`--paths` 语义 = 按「提交内容」判定**（同一条命令行里 `git add X && git commit X` 时 X 还没进
   index，本地 index 说明不了 CI 会看到什么）。`--paths` 同时把筛选与「视为已提交」两件事做完。
3. **删除保护精确化**（`guard_protected_paths.py`）：`tests/` 下的删除只有命中的是**已跟踪内容**
   或**名字像测试**（`test_*.py`/`conftest.py`）时才拒绝；**未跟踪且非测试名**的清理放行。
   判已跟踪内容由 `git ls-files -- <path>` 回答；**任何回答不出来的情况都按「已跟踪」处理**（fail-closed），
   目标不存在也拒绝（"认不出这个文件"不是放行的理由）。
4. **提交前闸门**（新 `guard_conventions_before_commit.py`，PreToolUse / Bash）：判
   `git diff --cached` ∪ 同一命令行里 `git add/rm/mv` 的操作数 ∪（`-a` 时）已修改的已跟踪文件；
   目录/整树形式退回判整棵树（超集是诚实的答案）；不是普通路径的操作数（含空白、shell 元字符、
   前导 `-`）**绝不**被当作路径传给检查器，同样退回判整棵树。拒绝时打印失败的规则、被判路径、
   以及"这就是 CI 会红的那条"。
5. **一切 fail-open**：内部错误放行并说明（沿用既有 hook 约定）——这道门后面还有 CI。

## Consequences

**变容易的：**

- **本地红灯恢复信息量**：「本地阻断失败」⇔「提交树上有违规」，两者由构造相等；
  再也不会因为一个提交永远看不见的未跟踪文件而常态红灯。
- **`tests/` 下的垃圾可以自清理**：会话内即可删掉自己的草稿（实测已删除那个遗留三个轮次的文件，
  R-044 命中归零），而"删测试"的保护对**已跟踪**测试与任何 `test_*.py` 仍然生效。
- **提交前就能知道 CI 会不会红**：`git add X && git commit` 这种本项目最常用的写法也在判定内
  （实测：暂存违规文件 → 拒绝；同一条命令里 add 的违规文件 → 拒绝；合规文件 → 放行）。

**变难 / 代价（如实列出）：**

- **未跟踪文件上的代码质量违规不再本地硬停**（仍打印为 tolerated）。补回它的是「暂存即阻断」与
  第 4 条闸门：真正的代价窗口只剩「写了违规文件但既不暂存也不提交」。
- 提交闸门每次多跑一遍阻断规则（本机 ~1–3 s），且 **hook 配置在会话启动时读取**（决策 0002 附录）：
  **本次新增的条目要重启会话才生效**，脚本级修改则即时生效。
- 删除保护依赖 git：在非仓库环境里 `git ls-files` 回答不出来 → 一律拒绝（保守），代价是
  非仓库目录下连垃圾也删不掉。
- 提交闸门是**尽力而为**（别名、`git -c` 等异形写法不保证识别），且 fail-open；它不是 CI 的替代。

**备选方案与否决理由：**

- *只删掉那个遗留文件*：治不了病根——下一个未跟踪垃圾会再造一次假红，且 `tests/` 删除保护仍会
  把它锁死。故删除与两条规则修订同时做。
- *把 R-044 的作用域改成「只扫已跟踪文件」*：能消掉这一条假红，但同类假红会出现在其它扫文件系统的
  规则上；统一按已跟踪集合判定才是"本地 ≡ CI"的完整表述。
- *加 `.git/hooks/pre-commit`*：不进版本控制 → 按 R-037 不算治理层，且是隐藏的本地行为；改走
  `PreToolUse` 守卫（治理层内、可测、可审）。
- *让 Stop hook 阻断回合结束*：会让会话卡在已知的未跟踪垃圾上（本决策第 1 条解决之后才安全），
  本轮不做；Stop 仍只报告。
- *把提交闸门写成"simple 版"（只判当前工作树）*：实测会漏掉「同一条命令行里刚 add 的新文件」，
  而那正是本项目最常用的提交写法，等于没加。
