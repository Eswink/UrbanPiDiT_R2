# 0011 — 自主执行授权扩围与主模型 V2 新窗口 goal 交接

**状态：治理与交接已实施（2026-10-02），提交 b961b3b/CI37024597813 九步成功；本轮不执行实验。最终记录与术语补正另行提交。**
用户授权原话：「推进权和实验权，普通决策权全部下放给你，注意，我只需要负责总体实验的方向！」
本计划按批准范围落地0030与新goal，不把计划或prepared长文当已完成实验。

## §1 起点与依据

- HEAD `256ef256c85ff48369681fd43f31cfd9c079eda3`，分支 `r7/weather-reasoning`；跟踪工作树干净，
  无关 `.zcode/agents/web-researcher-backup.md`、`.zcodeignore` 未跟踪且不纳入本任务。
- 最新 ADR0029/计划0010，故本轮新编号0030/0011。campaign=N2a/budget_limited，
  cap/used/remaining=24.0/4.7018/19.2982、budget_mode=accounting-only；0失败/4历史notes。
- 前轮实验/时长授权已下放，但0029:48–49仍不下放节点，0025:35–36仍逐轮用户触发，
  主campaign §4与goal-loop步骤仍每轮结束即停，与本次新授权相反。
- M3六训练/七评估但缺23项与完整paired/四成本，历史失败/0.497254GPU-h与1805.1086s事实保留。
  判据只有已有设计、计划0004/0009及各节点§3；本治理轮不外委或新增科学阈值。

## §2 交付物与改动面

| # | 内容 | 文件与验证 |
| --- | --- | --- |
| D1 | 0030新ADR，继承0029时长/保留边界，普通决策与节点推进常设下放 | `docs/decisions/0030-autonomous-execution-and-direction-boundary.md`；0029 superseded正文保留，0025只追加部分取代指针，ADR索引；R-033/R-036 |
| D2 | 更新现行推进分工与条件化关闭路线，不提前改state/账本 | AGENTS、主campaign、两份SKILL、CI细则、CHANGELOG；无逐轮用户许可，前置/失败/诚信门保留 |
| D3 | 新goal长文与可复制单段objective | `docs/goals/v2-autonomous-completion-and-closeout.md`、goal索引；objective≤4000/自指/完整交付物；G-01..G-08 |
| D4 | 计划归档与新窗口起步交接 | 本文件与plans索引；原旧计划/失败长文不回写，R-032 |
| D5 | 实跑门禁、独立内容审阅与提交/CI | conventions/campaign/goal/治理pytest/index/空白，精确SHA九主步骤CI；不触发实验 |

## §3 实施顺序

1. **只读侦察与planner校验**：核编号、HEAD、现行冲突，归档不变性基线；planner原草稿只作工作态，
   主链纠正不存在的0029路径、400/4000上限、漏AGENTS/goal文件和错误“0006门禁”，形成
   `/tmp/r7_0030_plan_20261002.json`，`check_planner_plan.py` verified=true、0失败。
2. **D1**：新0030 accepted；0029改状态并反向指针；0025保留accepted及正文，只声明第5条与
   第4条普通节点许可解释被新授权取代。继承全部时长、会计、硬边界，不扩大至付费/发布/合并。
3. **D2**：用户总体方向、执行者普通选择和节点推进分工落AGENTS/campaign/技能；会话内连续
   推进但不cron/后台守护；独立审阅不作新许可，最终complete不由执行者自判。主计划补N5、
   区分N4确认与adaptive条件，关闭禁令只在0009/0030具名证据/精确CI/非force ff条件下解除。
4. **D3/D4**：新长文prepared、首次标N2a；本轮不改current_round_goal、state或旧next-action，
   下一窗口执行时按实际节点证据同步。单段objective与长文副本相同，保留已有科学任务。
5. **D5**：实跑检查和独立覆盖审阅，治理文件git add、docs(r7)提交推分支，等精确SHA九主步骤CI。
   新实验、main写入、issue关闭均留待新窗口；不把历史绿CI冒称覆盖新增文字。

## §4 验证与停止条件

实跑：`.venv/bin/python tools/check_conventions.py`；`--rule R-033 --rule R-036`；
`tools/check_campaign_state.py`；`tools/check_goal_brief.py --brief docs/goals/v2-autonomous-completion-and-closeout.md`；
现行四goal+新goal；三组治理pytest；index/candidate brief；`git diff --check`、`git show --check`。
目录级旧goal15处失败照报告，不回写五份旧页；新增goal不能增加失败。核本轮仅治理文档改动，
原代码/测试/workflow时限/冻结常量/历史证据/计划/账本/state不变。验证失败修实际问题，不改判据。
本轮0 GPU-h，若任务触及实验执行/保留边界则不是本交接实施范围；总体方向改变交用户决定。

## §5 新窗口提示词与授权边界

正式单段objective唯一源为新goal **§0**（文末同文副本）；可复制形态为 `/goal ` 加该段。
科学任务仍按0009：N2a补缺评估/完整来源与成本→N3两步rollout与matchedGeneric→N4三臂确认→
N5时间/地理/现役机制反证、六issue裁定与有条件关闭。运行每次先写死planned/hard数值，
无总GPU-h闸门；旧失败页/1800秒协议不改。实验、时长、普通排障/协议/节点不再询问用户；
总体方向与保留资源/数据/破坏性边界交用户。目标与整个研究完成不由执行者自行宣布。

## 实际结果

- **已落盘**：D1–D4治理与交接。0030完整继承0029，新增普通决策/节点连续推进授权；原state、
  账本、旧失败/协议/证据均保留。新goal prepared，objective实测2027字符、单段、自指文件。
- **与planner草稿/计划的差异**：按§3第1条收敛，未原样采用错误路径/400字符goal要求/不存在
  门禁名；0030作为具名关闭禁令解除记录，不额外向用户恢复逐节点许可。未回写旧N3/N4/closeout
  长文，新的总体目标和主计划声明当前适用范围。
- **实跑验证**：planner JSON verified=true/0失败；conventions37阻断0违规、R-033/R-036取代链0违规；
  campaign N2a/0失败/4历史notes；新goal+现行四goal共5份0失败/0advisories；三组治理测试
  **152 passed in 22.22s**；index/brief17records、空白与git show --check均exit0。目录级18briefs/
  15历史失败/exit1未增加失败，不冒称目录全绿。原991个跟踪文件/两无关文件hash不变，state全部
  字段/11账本行/历史进度尾保留；2027字符objective两份副本精确一致，主计划objective379字符。
  新文件入跟踪后复跑 **152 passed in 22.19s**、37阻断/ADR/对表/空白exit0。独立只读8文件审阅
  实质PASS，确认普通推进不被独立审阅收回、失败/身份/关闭边界未绕，未作科学或goal完成裁定。
  唯一非阻断措辞残留（goal-loop“独立复核”）统一为“用户/运行时独立完成校验”，不改变推进许可。
  精确提交CI绑定见下节，不引用旧CI替代。日志工作态 `/tmp/r7_0030_validation_20261002/`，不提交。
- **未做**：N2a评估补全、N3/N4代码及实验、N5补反证/六裁定/关闭均未做；无GPU、下载/发布、
  main写入、模型digest、接口/依赖/凭据/安全配置变化，不宣称科学/安全或最终goal完成。
- **下一项**：新窗口直接粘新goal §0，首先实测HEAD/对表/身份，准备并冻结N2a新补测协议后自主执行。

### 精确提交与交付审计

- 治理/交接14文件提交 `b961b3b3084a47ccee8a3bdf2b22b46887055a0e` 已推工作分支，
  [CI37024597813](https://github.com/Eswink/UrbanPiDiT_R2/actions/runs/37024597813) head_sha精确匹配，
  completed/success、九主步骤全部success（含campaign、index与全unit/integration/wheel测试）；
  未获取远端pytest日志数字，本地152不冒称远端。无实验标签/Closes关键字，不触发实验或关闭。
- 一手来源匿名curl访问2026-10-02：
  [run API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/37024597813)、
  [jobs API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/37024597813/jobs?per_page=100)。
  run响应SHA256 `f8fa4bc1bd229c60c844a0808edf26afd63d45cb141cf4ee0be48d272312e1f6`、
  jobs响应SHA256 `47c7afefe8b9161de2e2c1d5d2ef351c4ce42a0b36fd3ff600d48cd901735f7b`；
  `/tmp/r7_0030_governance_ci_20261002/verification.json` SHA256
  `e36d82cd48d130831f180fd1a0fac316e603c678c1f308a9b9f3a4d9e473a14c`。
- 主链逐要求/负面范围/实际日志/精确CI交叉核对1129项、0缺项（其中991项为原跟踪文件不变性）；
  工作态 `/tmp/r7_0030_deliverable_audit_20261002.json` SHA256
  `f726748164f80a95ab1aaee05a9f30536244e43bc0d79ebfee43059661485044`。
  审计不代替科学门禁或runtime完成校验，不能只因manifest齐或测试绿裁定未来实验。
- 本CI尾核、术语补正与实际进度另作后续小提交并核其自己的精确CI，最终结果在收尾报告给出，
  不递归追加提交。Mimosa commit hook报scanner_enobufs，未得完整安全结论，不宣称项目安全。
  旧历史/失败/state/账本/数据/归档/检查器/测试/工作流数值仍不改。
- 正式新窗口 `/goal` 的内容为新长文§0的2027字符段，§9副本相同；本轮仅交付该prepared目标，
  N2a补测至N5关闭全部未启动，执行者未自行修改任何最终goal状态为complete。
