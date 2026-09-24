# 计划 0002：项目专属 hooks 与 skills（含覆盖率审计）

- **日期**：2026-09-24
- **状态**：已完成
- **来源**：会话 `sess_800d8127-5991-498e-9855-2c16b724787d` 的两轮计划（第四遍与第五遍）
- **相关决策**：暂无（本计划未产生独立决策，其产出的规则见 `docs/rules/ci-and-verification.md`
  与 `docs/rules/testing.md`；后续的存放约定见 [决策 0001](../decisions/0001-artifact-storage-convention.md)）

## 背景

项目此前**没有任何能力目录**，也没有任何自动拦截机制（唯一的安全插件只做提示/审查，
不拦路径）。同时确认：4 个已写的 skill 放在 `docs/skills/` —— 那不是 ZCode 的 skill 发现路径，
所以它们只是文档、模型不会自动加载。

## 要建的东西（第 1 轮）

**Hook 脚本**（`tools/agent_hooks/`，纯标准库、全部 fail-open）：

| 脚本 | 事件 | 行为 |
| --- | --- | --- |
| `guard_protected_paths.py` | PreToolUse | 拒绝 `data/raw|interim|processed` 与归档快照的写/删/移动 |
| `guard_destructive_git.py` | PreToolUse | 拒绝 force-push、`reset --hard`、`clean -f`、`branch -D`、对 `main` 的写 |
| `check_model_digest_impact.py` | PostToolUse | 提示（不阻断）`model/**.py` 改动会改变 `model_code_sha256` |

配置写在 `.zcode/config.json`（`hooks.enabled: true`），Stop 事件跑约定门禁。

**Skill 迁移与新**：把 4 个从 `docs/skills/` 迁到 `.agents/skills/`（**真正的发现路径**），
新增 `environment-rebuild`、`ci-workflow-triage`、`issue-lifecycle`。

**第 2 轮（对抗性审计）**：由独立子代理逐条复现绕过，修 24 处，并新增 `tests/` 删除保护。

## 实际结果

**完成情况**：两轮均完成。

- ✅ 3 个 hook 脚本 + 1 个 Stop 门禁，配置在 `.zcode/config.json`
- ✅ 4 个 skill 迁移（迁移前后 sha256 逐文件核验一致）+ 3 个新增 = 7 个
- ✅ `.gitignore` 从 `.zcode/` 改为 `.zcode/*` + `!.zcode/config.json`（让配置可提交）
- ✅ hook 自测 123 个（含 24 个"审计发现的绕过"反证 + 5 个测试删除 + 9 个下载放行）
- ✅ 修复 24 处实测绕过（路径归一化、fd 重定向、外部程序输出、就地编辑、git 变体等）
- ✅ 新增 `tests/` 删除保护（R-009 是聚合且仅报告，看不见整文件删除）

**与计划的差异**：

1. **"移动入口会破坏测试"这一约束是计划阶段就发现的**（3 个诊断入口被测试按裸文件名调用），
   因此改为分组处理：3 个保留根目录、3 个移入 `scripts/`。计划未因此返工。
2. **审计发现的绕过远多于预期**：计划只预期修几个边界，实测 24 处，
   其中 `python -c` 内联写与 `cd` 进保护目录后用相对路径是原理性的（文本级 hook 的上限）。
3. **`_DEST_ONLY_VERBS` 的 `endswith` 判断有缺陷**：`cp x legacy_v6/f.py`（目标是文件）
   漏判；改为只检查最后一个操作数。
4. **`_mentions_protected` 预检过窄**：glob 形态（`data/ra*`）在预检就被短路，
   导致已实现的 glob 判断永远执行不到。
5. **重构时 `sed` 分支条件顺序写错**，一度让 `cp` 与 `sed --in-place` 同时回归 ——
   被既有的参数化测试当场抓到。

**验证**（全部实跑）：hook 自测 123 全绿；全量测试 745 passed / 3 skipped / 0 failed；
20 条阻断规则 0 违规；`git status` 确认 `.venv` 未被跟踪。

**遗留**：

- **文本级 hook 拦不住 `python script.py` 内部写 `data/raw`** —— 原理上限，
  已在 CHANGELOG 与 `AGENTS.md` 中如实声明"是护栏，不是沙箱"。
- **5 条硬约束没有任何机械执行**（未实跑就写 PASS、报告区分已确认与推测、
  GPU/大批量授权等），其中 3 条无法机械化、2 条部分可机械化（见交付报告）。
- **其中 1 条已在后续解决**：治理层未进版本控制 —— 这是第 2 轮审计**未发现**、
  由第 3 轮（存放约定）发现的问题，见 [计划 0001](0001-artifact-storage-convention.md)。
