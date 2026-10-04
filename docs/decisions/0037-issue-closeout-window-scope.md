# 0037 N5 收尾轮：#70–#75 具名有条件关闭的范围与代价

- **日期**：2026-10-04
- **状态**：accepted

## Context

计划 0009 与决策 0030 §5 已写明：#70–#75 的关闭授权以 N5 收尾轮为限，条件是逐 issue 证据
支持相应 verdict、工作分支精确 SHA 的 `R7 CPU CI` 必要步骤 success、非 force ff 推送 main
与匿名复核；合并仍禁止且不需要。到 2026-10-04，actual C（144/144 job，独立接受
`aa44b8cd…`）、actual M1（24/24 job，verify `c5df8cc6…`）、新包 precision（独立接受
`8503adfe…`）与两侧 UTC 统计链均已实跑并登记；#71/#72 的判据补强测试已由 `616b029`
随工程落地（`tests/test_r7_v2_time_and_feedback.py`），本轮只复跑与登记。

各历史长文（`n1-cost-supplement-repair`、`main-model-v2-pivot-audit`、`main-model-v2-rw-b-round`、
`main-model-v2-rw-b-subtraction`、`n4-m5-confirmation` 等）中的「不关闭 #70–#75」是当轮禁止项。
按 0030 已声明的体例，旧长文是当轮事实、不回溯改写；自本决策起，后续执行以本决策与
`docs/goals/v2-issue-closeout.md` 为准。主计划 §2 的「main/issue 关闭仅限 N5…禁止提前关闭」
正是本窗口要满足的条件句，本轮把它从「待满足」推进为「按本决策执行」。

## Decision

1. **范围**：仅 #70–#75 六个 issue，且仅经 N5 收尾轮的单一关闭提交（正文逐行 `Closes #70`…
   `Closes #75`）生效。不扩展到其它 issue；不为关闭新建或合并任何 issue。
2. **机制**：关闭提交先推工作分支并等精确 SHA 的 `R7 CPU CI` 12 必要步骤 success；再非 force
   ff 推送 main（决策 0003 放行）；推送后以匿名 API 逐条复核 `state=closed`。若任一条件不满足，
   停在分支上报告；不 force、不合并、不改用其它写入路径。
3. **判定口径（与证据页一致，不因关闭放宽）**：#70 判「工程交付 DONE／科学增益
   negative-mixed」；#71 判「工程 DONE／单因素 unresolved、不 DONE-positive」；#72 判
   「结构验收 DONE／实验 negative-mixed（不能归因）」；#73 判「规定交付 DONE／科学 paused」；
   #74 判「NEGATIVE（rollout 假设终结）」；#75 按「科学增益未确立／工程完成／某试验否定」
   三类分开写。关闭不等于研究完成或科学成功。
4. **保留禁止**：force push、`--mirror`、删默认分支、本地/远端合并、其它 issue 关闭、事后放宽
   已冻结判据/阈值/端点/案例集、删弱测试与断言、改写归档或负面结果、读封存 test、无授权下载/
   租卡/发布——全部照旧；R-006/R-028/R-054 与身份校验不豁免。
5. **回写**：关闭复核结果（时间戳、CI run id、提交 SHA、推送输出）回写进 `docs/R7_TASK_QUEUE.md`
   六行、`docs/R7_ISSUE_COMMENTS.md` 状态段与主计划 §8；`scientific_claim: false` 不因关闭改变。

## Consequences

**变容易的：** 六个 issue 的最终判定与证据在一个收尾提交里闭环，GitHub 状态与仓内记录一致；
后续研究（可分辨过程语义的新协议、最终独立年份/季节测试、adaptive 重评）有明确的重开条件可引用。

**变难 / 代价（如实列出）：** 关闭会给外部读者「已结题」的观感，但其中多数判定是
negative/mixed/unresolved——文档必须始终申明关闭≠科学成功，核心单因素贡献仍不可归因；
adaptive 以「门未过、不启动」结题；已被否定的机制（RW-B 门控、两步 rollout、naive feedback）
保留负面、不重开；`main` 前进不改变任何历史 sealed 产物的身份边界（各自 code.zip/protocol 承担），
不因 main 推进获得任何追认。六个 issue 关闭后，新问题需要新的 issue 与新预算，不得复用本窗口。

**备选方案与否决理由：** 继续挂起不关闭——证据已足以支撑逐条 verdict，继续挂起反而使记录失真，
且 0030 已把该决定权交回执行者；API/`gh` 直写——无凭据（401）；合并——不需要且被 hook 拒绝；
把关闭写成科学成功或 DONE-positive——与 #70/#71 的明文条款冲突，不采用。
