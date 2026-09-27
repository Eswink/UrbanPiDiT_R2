# 0012 planner 子智能体的委派边界与 JSON 契约

- **日期**：2026-09-27
- **状态**：accepted
- **依据证据**：`tools/check_planner_plan.py` 与其 36 个测试；`.agents/skills/planner-delegation/SKILL.md`

## Context

用户在 ZCode 中新增了一个 user 级子智能体 `planner`（模型 claude-opus-5-5，思维档"最高"），
工具集限定为 **Read / Grep / Glob / WebFetch / WebSearch**——**只读、不写任何文件**，
唯一交付物是一份 JSON 计划。用户的意图是：**迭代过程中需要撰写计划文件或 TODO 时，
先委派 planner 产出 JSON，再由主 agent 接收并落地**；planner 不可用（额度耗尽等）时
降级为主 agent 自行规划。

实测确认：本仓**此前没有任何委派/子智能体协议**——`subagent`、`subagent_type`、`planner`、
`委派`、`子智能体` 在活跃区零命中；唯一近似是 2026-09-24 一次性对抗性审计的叙事，
不是可复用流程。因此这是一项**新能力**。

关键约束：**planner 本身不在版本控制内**（ZCode user 级配置，仓库外），所以仓库能固定的
只有**协议与 JSON 契约**，固定不了它的系统提示词。

## Decision

采用委派协议，并把它写成可执行的两件套：

1. **能力**：`.agents/skills/planner-delegation/SKILL.md`（7 段结构，含内嵌规范示例 JSON）。
2. **契约的机械真相**：`tools/check_planner_plan.py`——只读、fail-closed 的校验器，
   退出码 0/1/2。planner 的 JSON 必须过它才能落地（`"verified": true`）。

**授权边界（明确划定）**：planner **仅做工程规划**——任务拆解、步骤排序与依赖、
改动面清单、风险、验收动作、以及需人拍板的问题（`open_questions`）。
**科学判据不外委**：阈值、非劣容差、判据与 DONE 定义仍由本仓纪律写定并实验前冻结。
这条边界由校验器**机械编码**而非仅靠文字约定：`science_criteria_refs` 的每一项必须是
**文档指针**（含 `/` 且以 `.md` 结尾），出现裸数字或阈值即拒绝。

`files_to_touch` 另受机械限制：不得命中 `data/raw|interim|processed` 或归档快照前缀；
该前缀清单**从 `tools/check_conventions.py` 的 `ARCHIVAL_PREFIXES` 推导**，
并有防漂移测试断言两者相等——不另造平行清单。

## Consequences

**变容易的：**
- 多步方案设计与迭代计划有了固定的产出形态，且形态可被机器检查；
- 契约的"文档—实现"一致性由自洽测试保证（skill 内嵌示例必须通过校验器）;
- 委派失败有明确降级路径，不阻塞任何工作。

**变难 / 代价（Trade-off，如实列出）：**
- **planner 的系统提示词不在版本控制内，行为可能漂移**；本仓唯一的护栏是 JSON 校验器。
  它能拦住畸形结构与越界字段，**拦不住内容质量**（步骤本身是否合理仍需主 agent 判断）。
- 多了一步校验动作：委派链路比自规划长，单步任务用它是净损失（skill 已写明不适用）。
- **planner 读不到 `outputs/` 产物**（工具集无 Bash，无法计算 digest 或读运行记录），
  因此可能与既有证据脱节；现状必须由主 agent 写进委派提示词。
- 存在**误用风险**：把 planner 输出当作证据或审查结论。skill 的「常见失败」与
  「明确不覆盖」两段明确写了它不产生证据、不替代审计。
- 额度耗尽即不可用，须降级自规划；这会在报告中记为一次降级，而非静默发生。

**备选方案与否决理由：**
- *只写文档、不做机械校验*：契约会与实际使用脱节，且无法满足 R-026「每个判据必须有反证」；
- *让 planner 参与科学判据*：一旦阈值由外部生成，「实验前冻结 + digest 校验」
  （B2 曾抓到作者自己的事后改动）这套防线即失效；
- *引入 `jsonschema` 依赖*：全仓零使用，手写谓词已够，新增依赖需单独授权；
- *新增 hook 强制委派*：委派是流程选择而非安全边界，加 hook 只会增加误报面。

**边界**：不改 `.zcode/config.json`（不动 hooks）；`tools/check_planner_plan.py` 已加入
R-037 的 `GOVERNANCE_ASSETS`（它是被执行的治理资产）。
