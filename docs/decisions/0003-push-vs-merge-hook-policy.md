# 0003 hook 策略：放行非 force 推送到 main，合并仍需用户明确授权

- **日期**：2026-09-25
- **状态**：accepted
- **取代**：0002（其执行机制被实测证伪；0002 已标 superseded by 0003）

## Context（背景）

8 个 open issue（#1、#5、#6、#7、#8、#9、#13、#20）的判定与证据已全部落盘，收尾提交
`e5be0c0`（含全部 `Closes #N`）在分支顶端 `1a1e5fe`，只差一次 fast-forward 推到 main
（origin/main `92a8c4d0` 是严格祖先，领先 167、落后 0）。但关闭 push 被
`guard_destructive_git` 以 "pushing to main" 拒绝四次：决策 0002 的逃生口（会话内
临时移除 `.zcode/config.json` 条目）被实测证伪——hook 定义在**会话启动时**读取，
会话中途改 config 不影响当前会话（0002 Addendum）。

用户 2026-09-25 明确指示修改规则与 hooks：**"需要允许推送，但是如果是合并需要用户
明确告知授权方可进行操作。"**

侦察同时发现两个既有漏洞：一旦放行 main 推送，它们会从"潜伏"变成"致命"——
(a) 裸 `+main`（无 refspec 冒号）的 force 推送不被 `_FORCE_PUSH_PLUS` 匹配；
(b) `git push origin :main` / `--delete main`（删除默认分支）与 `--mirror`
（强推重写全部 ref）从未被任何规则拦截。

## Decision（决定）

修改 `tools/agent_hooks/guard_destructive_git.py` 的规则集（脚本每次工具调用重新
读取，**改脚本即时生效**，不受 hook 配置会话边界的限制）：

1. **放行**：非 force 推送到 main（`origin main`、`<分支>:main`、`HEAD:main` 等全部
   refspec 形态）——移除 `pushing to main` 规则。
2. **保留拒绝**：合并——`git merge` 涉及 main（加固：`origin/main` 形态一并纳入，
   此前只拦裸 `main`）、`gh pr merge`。
3. **新增拒绝**：裸 `+main` force（`_FORCE_PUSH_PLUS` 目的地可选化）、删除默认分支
   （空 source `:main`/`:master` 与 `--delete main`）、`--mirror` 推送。

**合并的授权路径**：hook 无法核对对话内授权，因此合并由用户在 ZCode 之外的终端
执行；或在会话启动**前**移除本 hook 条目（决策 0002 附录记录的会话边界仍成立）。

`.zcode/config.json` 不变（守卫保持注册；改的是脚本不是注册）。

## Consequences（后果——含代价）

**变容易的：** issue 关闭推送可由 agent 自主完成，不再依赖人在 ZCode 外执行单条
命令；GitHub 的 issue 状态与仓库内判定一致，可直接审计。

**变难 / 代价（Trade-off，如实列出）：**

- **"合并需授权"约束的是 merge 操作本身，不是 main 的历史形态**：merge commit
  仍可经放行的 push 路径到达 main（ff 语义不区分普通提交与合并提交）。策略边界
  从"main 的历史"收缩为"agent 执行的 git 操作"。
- **自主写 main 成为常态策略**，不再单次授权。防线收敛为三层：GitHub 拒绝非 ff、
  hook 拒绝全部 force 变体与删除默认分支/--mirror、工作分支每 push 必过 `ci.yml`
  （main 只会收到已过 CI 的提交）。若分支保护在 GitHub 侧启用，push 会被拒并如实
  记录 BLOCKED。
- **main 的 push 不触发 `ci.yml`**（其 push 触发只限工作分支）：main 顶端没有自己的
  绿色 run，验收证据口径仍是工作分支同一 SHA 上的 run，不得谎称"main 上 CI 绿色"。
- **guard 的文本匹配本质未变**：跨行正则会把散文里的 "git merge" 与后文 " main"
  拼成一次命中（本轮改 AGENTS.md 时实测发生一次）；已知的误报类，靠拆分命令规避。
- 测试面变化：`test_agent_hooks.py` 132 → 142 用例（push-to-main 迁入放行组、
  新增 force/删除/mirror 拒绝用例、`origin/main` 合并拒绝）；全量
  885 passed, 3 skipped, 0 failed。

**备选方案与否决理由：** 维持 0002 机制并要求人执行单条 push（用户已明确改为策略级
放行）；用 PAT / `gh auth login`（用户声明只用 git/SSH）；在 hook 里读取"授权标记
文件"（agent 可自行创建该文件，等于自毁门禁，否决）。
