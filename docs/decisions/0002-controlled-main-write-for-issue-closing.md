# 0002 受控 main 写入以在 GitHub 上关闭 8 个 open issue

- **日期**：2026-09-25
- **状态**：superseded by 0003 — 本决策的执行机制（会话内临时移除 hook 条目的
  「逃生口」）被实测证伪（hook 配置在会话启动时读取，见文末 Addendum），且用户
  已在 0003 把策略改为 hook 级放行非 force 推送；本决策的 Context 与 Addendum
  保留为历史记录。

## Context

8 个 open issue（#1、#5、#6、#7、#8、#9、#13、#20）的判定与证据已全部落盘
（`docs/R7_ISSUE_COMMENTS.md` 与各 `docs/R7_*.md`，绑定精确 SHA），但 GitHub 上它们
仍处于 open 状态。本机没有 `gh`、没有 PAT：API 写（评论 / 关闭 issue）实测四次均
返回 401，因此「关闭 issue」无法走 API。唯一生效的 git 路径是 closing keywords
（`Closes/Fixes/Resolves #N`）出现在**默认分支 main** 的提交里；推到工作分支
`r7/weather-reasoning` 无效。实测 main（`92a8c4d0cfef7908e2b358e0ba2a6b4b3f45eebc`）
是该分支的严格祖先（落后 0、领先 165+），fast-forward 推送即可、禁 force。

对 main 的写入会被 `.zcode/config.json` 注册的 PreToolUse hook
`guard_destructive_git`（`tools/agent_hooks/guard_destructive_git.py`）拦截——它是
文本匹配器，命令文本里出现 `git push origin main` 即触发。AGENTS.md 记载了逃生口：
临时移除对应 hook 条目 → 完成后立即恢复。用户于 2026-09-25 明确授权了**这一次**受控
main 写入（并同轮决定：不触发 17 条实验 workflow，不等任何 CI run）。

## Decision

我们采用：把携带全部 `Closes #N` 的收尾提交（本提交）以 **fast-forward** 推到 main，
使 8 个 issue 被 GitHub 自动关闭、Draft PR #12 随 head 分支到达 main 顶端而显示为
merged。写入按 AGENTS.md 逃生口受控执行：临时移除 `.zcode/config.json` 中
`guard_destructive_git` 的 PreToolUse 条目 → 执行 `git push origin main` 这**一次**
授权写入 → **立即恢复**该条目（文件受版本控制，恢复后以 `git diff` 核对逐字节一致）。
全程禁 force、禁以 refspec 拼写（如 `HEAD:main`）绕过正则；若 GitHub 拒绝（分支保护），
如实记录错误并标 BLOCKED，不重试轰炸。判定所依据的实验证据是此前各 SHA 上已核验的
绿色 run（#5/#8 绑定 SHA 上的 cancelled run 由紧随的 docs-only 提交上的绿色 run
覆盖：`90e7d14` run 36090656848、`598857d` run 36099444288）；本轮不触发任何实验
workflow，收尾提交只改文档。

## Consequences

**变容易的：** GitHub 的 issue 状态与仓库内落盘判定一致，可从 GitHub 直接审计；
后续引用issue 结论时有了稳定的 closed 状态与 commit 交叉链接。

**变难 / 代价（如实列出）：**
- 执行窗口内 hook 保护对 main 写入是**关闭的**——若会话在移除与恢复之间被打断，
  main 处于无守卫状态；缓解：窗口仅一条命令，恢复后用 `git diff` 核对配置逐字节一致。
- main 被推到分支顶端后，**今后任何对 main 的写入仍需重新授权**；本次授权是单次的，
  不构成先例。
- main 的 push 不触发 `ci.yml`（其 push 触发仅限工作分支），因此 main 顶端没有
  自己的绿色 run；其内容的 CI 证据是工作分支同一 SHA 上已核验的绿色 run
  （`1449f26` 后继收尾提交的 run），须按此口径引用，不得谎称「main 上 CI 绿色」。
- PR #12 以 fast-forward 方式显示为 merged：**不存在 merge commit**，GitHub 上的
  merged 标记来自 head 提交可达 base 这一事实；回溯审计要看提交链而非合并节点。
- 17 条实验 workflow 在收尾 push 上按标签门控设计保持 skipped——这是设计行为，
  不是失败；需要实验证据时须另行按标签运行。
- 若分支保护拒绝推送，issue 保持 open，本轮以 BLOCKED 收尾并记录错误原文。

**备选方案与否决理由：** `gh auth login` 或提供 PAT（需要人侧动作，当前不可得，
且用户明确声明本项目只用 git/SSH）；直接调 API 关 issue（实测 401，写路径不存在）；
用 refspec 拼写绕过 hook 正则（属于本项目明令禁止的「审计发现的绕过」类，否决）；
继续把判定留在本地草稿不发布（无法满足「GitHub 状态以页面为准」的停止条件）。

## Addendum（2026-09-25 执行记录：写入被本地守卫阻断，issue 仍 open）

按本决定执行受控写入，**未成功**，如实记录：

1. 前置全部就绪：收尾提交 `e5be0c0a9dc43c9ccddadda58ca7499903c1a4d9`（含全部
   `Closes #N`）已推到工作分支；ff 关系实测成立（origin/main
   `92a8c4d0cfef7908e2b358e0ba2a6b4b3f45eebc` 为 HEAD 严格祖先，领先 167、落后 0）。
2. 按逃生口移除 `.zcode/config.json` 中 `guard_destructive_git` 的 PreToolUse 条目，
   并实测验证移除生效（JSON 可解析、PreToolUse 条目数 = 1、文件内
   `guard_destructive_git` 出现次数 = 0）。随后 `git push origin main` 被
   `guard_destructive_git` 以 `pushing to main` 拒绝；再用**只读**的
   `git push --dry-run origin main` 探测一次，同样被拒。两次拒绝后即停止（不轰炸重试）。
3. **环境发现（本轮的实质产出）**：ZCode 的 hook 定义在**会话启动时**加载；
   会话中途修改 `.zcode/config.json` 不影响当前会话。因此 AGENTS.md 记载的逃生口
   只在「条目移除**先于**会话启动」时可用；会话内移除→推送→恢复的顺序无效。
   另实测不存在用户级 `~/.zcode/config.json`，项目 config 是唯一 hook 来源。
4. 已采取的恢复动作：条目**立即恢复**并与 HEAD 逐字节核对（`git status` 干净、
   `git diff HEAD` 为空、PreToolUse 两条目齐全）；任务队列与评论稿中所有
   「已在 GitHub 关闭」的表述已改回「verdict 终局、GitHub 关闭待 main 写入」。
   **8 个 issue 在 GitHub 上仍为 open。**
5. 可行路径（按优先级）：（a）用户在自己终端（ZCode 之外，hook 不作用于直接
   命令行）执行唯一一条 `git push origin main`（ff、禁 force），issue 即自动关闭、
   PR #12 显示 merged；（b）在**先移除条目**之后启动的新会话里执行 push 并立即
   恢复条目；（c）用户提供 PAT / `gh auth login`（用户侧动作）。
6. Consequences 补充的代价：本轮停止条件下「GitHub 上 8 个 issue closed」
   **未达成**；收尾提交已推送但不含 GitHub 状态变更；「skipped workflow 按设计
   接受」与「不等待 CI」的处置不受影响。
