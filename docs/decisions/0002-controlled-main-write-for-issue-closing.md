# 0002 受控 main 写入以在 GitHub 上关闭 8 个 open issue

- **日期**：2026-09-25
- **状态**：accepted

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
