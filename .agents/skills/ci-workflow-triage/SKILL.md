---
name: ci-workflow-triage
description: 当 CI 失败、运行 pending/被取消、或要判断"这次算不算通过"时使用。含 18 条 workflow 的分类、run-id pin 核对、禁网验证与失败归因顺序。
---

# 排查 CI 运行

## 何时使用

- 某个 workflow 失败、卡在 queued、或被 concurrency 取消；
- 要判断一次运行能否算"通过"（**取消或排队中的运行不算通过**）；
- 改了数据/产物后要确认 workflow 的 pin 是否还对应。

**不适用**：
- 本地测试失败（那是普通调试，不是 CI 排查）；
- 要**加**新 workflow（那要先读 `docs/rules/ci-and-verification.md`）。

## 前置条件

- 知道是哪个 workflow、哪个 commit SHA、哪次 run id。
- 已确认改动面：`git status` 与 `git log -1 --format=%H`。

## 步骤

1. **先分类 workflow。** 本项目 18 条 workflow 分三类，排查路径完全不同：

   | 类别 | 数量 | 触发方式 | 是否该联网 |
   | --- | --- | --- | --- |
   | 主测试门禁 `ci.yml` | 1 | push 到 `r7/weather-reasoning` 或 PR 到 `main` | 否 |
   | 离线实验（study/control/replay/audit） | 11 | **提交信息里的方括号标签**（如 `[cpu-study]`） | **否，必须禁网** |
   | 真实数据获取（pilot/probe） | 7 | 同类方括号标签 | 是（受字节预算约束） |

   **第一件事**：确认你等的是哪一类。实验 workflow **不会**因为 `ci.yml` 绿了就自动跑。

2. **核对触发标签。** 离线/获取类 workflow 只在提交信息含对应标签时启动
   （如 `docs/R7_TASK_QUEUE.md` 记录的 `[continuous-control]`）。
   标签名与文件名**不完全一致**，例如 `r7-earthmover-probe.yml` 用 `[era5-temporal-probe]`。
   查实际标签：读该 workflow 的 `if: contains(github.event.head_commit.message, '...')`。

3. **区分"失败"与"没跑"。** 依次排除：
   - run 状态是 `queued` → **不是失败**，也不能算通过；
   - 被 concurrency 组取消（`cancel-in-progress: true` 的 workflow）→ **取消不算通过**；
   - 标签没写对 → run 根本不存在；
   - 真的是 `failure` → 进步骤 4。

4. **定位失败步骤。** 实验 workflow 的步骤顺序固定：
   install → download-artifact（带固定 `run-id`）→ 禁网 + 跑实验 → `git archive` 归档 → upload。
   逐段看：安装失败属环境；download 失败多半是 **run-id 与 artifact 名不匹配**；
   实验段失败才是真实验失败。

5. **核对 run-id pin（最常见的真实故障）。** 实验 workflow 硬编码
   `run-id:` 与 artifact 名。**改了源数据或产出新 artifact 后必须同步更新 pin**，
   否则会在旧产物上跑出结论。见 `docs/rules/CHANGELOG.md` 的 `model_code_sha256` 变化记录。

6. **确认禁网生效。** 离线 workflow 必须在 run 步骤内 monkeypatch
   `socket.socket.connect` 与 `socket.create_connection`。这是 R-028（**阻断规则**），
   由 `tools/check_conventions.py` 直接检查：
   ```bash
   .venv/bin/python tools/check_conventions.py --rule R-028
   ```
   注意该防护是进程内属性替换，是"防意外"而非沙箱。

7. **代码身份归档。** 跑实验的 workflow 必须归档 `code.zip` 与 `code_commit.txt`（R-010）。
   若失败的是**旧的**归档产物，检查：
   ```bash
   .venv/bin/python tools/check_conventions.py --rule R-010
   ```
   旧产物要用**它自己的 `code.zip`** 重放，不要用当前 HEAD 去加载旧 checkpoint
   （`load_checkpoint` 会因 `model_code_sha256` 不匹配而失败，见 Q-009）。

## 检查点

- **步骤 1 后**：确认目标 workflow 的**实际触发标签**，不要假设它跟 `ci.yml` 一起跑。
- **步骤 3 后**：若状态是 queued/取消，**停止**并如实记录"未通过"。
  不得把排队或取消写成通过（`docs/R7_MANUAL_ITERATION.md:17`）。
- **步骤 5 后**：改过源数据就必须确认 pin 与 artifact 名一致；不一致时先修 pin，
  不要在旧产物上继续。
- **步骤 7 后**：若诊断出 digest 不匹配，**不得**放宽或绕过比对逻辑。
  正确做法是用归档代码重放，并把处置写进文档。

## 常见失败

- **等一个不会跑的 workflow**：实验类需要提交信息标签，`ci.yml` 绿不代表它跑了。
- **把取消当成失败去"修"**：取消通常是同分支新推送触发的 concurrency 替换，
  等新的那次跑完即可。
- **在不该联网的 workflow 里加取数**：离线 workflow 的禁网是设计要求（R-028），
  要取数就属于"真实数据获取"类，应新开 pilot workflow 并遵守字节预算。
- **改了 `model/` 后旧实验失败就怪环境**：那是 **digest 变化**的必然结果。
  先看 `docs/rules/OPEN_QUESTIONS.md` Q-009，用归档 `code.zip` 重放。
- **忘记录 run id**：`docs/R7_CPU_REFINEMENT_RESULTS.md` 的形态是
  每个结论都带 `actual<run-id>` 与 `CI<run-id>`；缺了就无法复核。

## 完成判据

- 明确了 workflow 类别与实际触发标签；
- 区分了"失败 / 排队 / 取消 / 没跑"，没有把非失败当失败，也没有把非通过当通过；
- 若是真失败，定位到了具体步骤并说明了归因（环境 / pin / 实验 / 归档）；
- 结论记录了**确切的 run id 与 commit SHA**，并如实写明未解决的部分。

## 明确不覆盖

- 不覆盖加新 workflow 或改 CI 配置（先读 `docs/rules/ci-and-verification.md`，
  且改 `ci.yml` 会影响每次 push 的反馈）；
- 不覆盖 GPU 相关的 CI（CI 是 CPU-only，GPU 验收属 #20，仍需硬件）；
- 不覆盖"这次结果是否有科学意义"（那是 `result-freeze` 的职责）；
- 不覆盖 workflow 之外的手工运行（那按本地流程排查）。
