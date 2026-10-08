# 能力清单（索引）

本目录只是一个**索引**。真正的 skill 定义在仓库根的 `.agents/skills/` —— 那是
ZCode 的 skill 发现路径，模型会按 `description` 里的触发条件自动加载。

> 为什么不在 `docs/` 里：`docs/skills/` 不是发现路径，放在那里只是能被人读到的文档，
> agent 不会自动加载。规则正文（一句话约束）在 `docs/rules/`，这里只放"怎么做"。

判定标准：一个流程要成为能力，必须由证据表明它**已重复发生 ≥2 次**且步骤顺序固定。
一次性操作不立 SOP。

| 能力 | 触发条件 | 用途 | 证据 |
| --- | --- | --- | --- |
| [`bounded-study-run`](../../.agents/skills/bounded-study-run/SKILL.md) | 要跑一个受预算约束的**CPU**实验并留下可核查证据 | 冻结协议 → 训练 → 评估 → 归档代码身份；非通用GPU训练SOP | 11 个 workflow 同一形状；8 个模块同一模式（E-055, E-065） |
| [`pinned-artifact-replay`](../../.agents/skills/pinned-artifact-replay/SKILL.md) | 要用已归档的产物重跑或核对历史结果 | 校验源 hash 与 receipt → 重建 → 离线评估 → 写接受标记 | 3 个 replay 模块 + 3 个 workflow（E-129, E-130） |
| [`real-data-acquisition`](../../.agents/skills/real-data-acquisition/SKILL.md) | 要引入真实源/下载器或重建派生数据 | 核具名范围/分批预算→只读preflight审阅→显式新路径发布；0038方向内自审已授权，其余人审 | preflight 被每个 pilot 复用（E-051, E-052）；0038前瞻scope不豁免校验 |
| [`result-freeze`](../../.agents/skills/result-freeze/SKILL.md) | 某个结果要被引用、写进报告或归档 | 复现性核查 → digest 核对 → 登记 | 3 个迭代的 docs 记录都含 run id 与 digest（E-066, E-067） |
| [`environment-rebuild`](../../.agents/skills/environment-rebuild/SKILL.md) | 换机器/容器、`ModuleNotFoundError`、CUDA 不可用 | 虚拟环境搭建顺序 + 未声明依赖清单 | 2026-09-24 实测重建；4 个未声明依赖 + torch≥2.8 不兼容（E-157, Q-011, Q-012） |
| [`ci-workflow-triage`](../../.agents/skills/ci-workflow-triage/SKILL.md) | CI 失败、运行 pending/取消、要判断"算不算通过" | 18 条 workflow 分类 → 触发标签 → run-id pin → 禁网验证 | 18 条 workflow；`R7_MANUAL_ITERATION.md:17-21`；`R7_TASK_QUEUE.md` |
| [`issue-lifecycle`](../../.agents/skills/issue-lifecycle/SKILL.md) | 开始/推进/关闭 issue，或判断能否标 DONE | 四态推进 + 绑定精确 SHA + 证据要求 | `R7_TASK_QUEUE.md:59` + 6 个真实 issue（#53–#58） |
| [`decision-record`](../../.agents/skills/decision-record/SKILL.md) | 要做或记录一个影响架构/约定/长期行为的决定 | ADR 模板 + 状态生命周期 + 编号规则 + 与 OPEN_QUESTIONS 的分工 | R-033/R-036 的机械检查 + `docs/decisions/` 12 份记录 |
| [`planner-delegation`](../../.agents/skills/planner-delegation/SKILL.md) | 多步方案设计、迭代计划/TODO 撰写、架构选型、大改动前风险评估 | 委派只读 planner → JSON 过契约校验 → 由 agent 转成目标产物；含降级路径 | `tools/check_planner_plan.py` 36 个测试（含自洽与防漂移）；决策 0012 |
| [`goal-loop`](../../.agents/skills/goal-loop/SKILL.md) | 要写/推进一个 goal 目标；harness 的 goal 模式不可用或用户不要它而要按会话手工推进 | 长文骨架 + objective 压缩（≤4000、单段、自带交付物清单）→ 过校检器 → 选驱动（harness / 手工循环）→ 独立复核替代 | 两份目标长文同一形状（`docs/goals/m1-and-rw-a-iteration.md` §6、`v2-round-two-attribution.md` §6）；会话驱动纪律 `R7_MANUAL_ITERATION.md:8-30`；决策 0015 |
| [`web-research`](../../.agents/skills/web-research/SKILL.md) | 需要本仓以外的公开事实（库用法、版本变更、API 参考、论文、报错信息） | 委派 web-researcher → 回收并回一手来源核对 → 留痕（URL + 访问日期）→ 不可用时降级 curl | 可用性纪律与「摘要≠证据」先例（E-184）；可达性实测（E-188）；R-049/R-050（决策 0018） |
| [`parallel-research-workflow`](../../.agents/skills/parallel-research-workflow/SKILL.md) | 要在一条 campaign 里同时推进多个互不依赖的研究线、把资料/CPU 准备与实验重叠，或用 workflow 脚本编排多个子代理 | 并行准入七 gate → 编排形态选择（串行/普通扇出/dynamic workflow）→ 隔离与安全 → 科学完整性 → 每臂独立登记 | **前瞻立 SOP**（见下表后准入例外）：并行已被计划 ≥3 次（`docs/plans/0004-r7-main-model-v2.md:183`、`docs/goals/s2-climatology-mechanism-screening.md:54`、`docs/goals/n4-m5-confirmation.md:75`），父调度原则 `docs/plans/0016-main-model-climatology-campaign.md:78`、`docs/plans/0017-climatology-s3-long-term-handoff.md:106`；决策 0042 |

**准入例外（决策 0042）**：`parallel-research-workflow` 属**前瞻立 SOP**。项目准入要求能力「已重复发生 ≥2 次
**且步骤顺序固定**」（`:9`），而本能力目前只有「并行已被**计划** ≥3 次」的先例，**尚无 ≥2 次实际并行执行**。
依据是用户明确要求项目支持 workflow 与并行科研设计，故按决策 0042 前瞻准入，并强制**首次真实并行轮必须留痕
验证**（写进 goal §8 / 计划）；若实践显示步骤不固定或并行不可行，应修订或撤下本行与对应技能，不得让它长期
停留在未验证状态。

## 未立为 SOP 的候选（仅观察）

以下流程有重复迹象但证据不足，**不为它们建 skill**：

- **控制器标定 → 策略选择 → 测试**：有工具，但 `R7_TASK_QUEUE.md:51` 记录
  "严格策略回退到全深度"，没有成功的真实重复，只在合成 fixture 上跑过。
- **成对 block-bootstrap 比较**：只有一次真实比较（#54 vs #56），工具重复而协议不重复。
- **checkpoint 恢复协议**：文档完整（`docs/R7_LOCAL_RUNNER.md`），但只有一个使用示例。
- **边界/ACC 指标流水线**：实现且有测试，但只在合成数据上跑过。

## 相关的执行面

- `tools/check_conventions.py` — 37 条阻断规则，已接入 CI 与 Stop hook。
- `tools/check_planner_plan.py` — planner 委派计划的 JSON 契约校验器（只读；保护前缀
  与 `check_conventions.ARCHIVAL_PREFIXES` 同源，有防漂移测试），见 `planner-delegation`。
- `tools/check_goal_brief.py` — goal 长文的结构校检器（只读、**非阻断**：未接入 Stop hook
  与 CI；规则号 `G-01`…`G-08` 与建议项 `A-01`/`A-02` 属于它自己的命名空间，不是
  `docs/rules/` 的 `R-0xx`），见 `goal-loop`。
- `tools/agent_hooks/` — 六个 PreToolUse/PostToolUse hook（保护路径、破坏性 git、提交前惯例、
  外部检索路由、model digest 提醒、外部抓取留痕），另有 Stop 收尾检查，
  配置在 `.zcode/config.json`，见 `AGENTS.md`。
