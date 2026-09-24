# 能力清单（索引）

本目录只是一个**索引**。真正的 skill 定义在仓库根的 `.agents/skills/` —— 那是
ZCode 的 skill 发现路径，模型会按 `description` 里的触发条件自动加载。

> 为什么不在 `docs/` 里：`docs/skills/` 不是发现路径，放在那里只是能被人读到的文档，
> agent 不会自动加载。规则正文（一句话约束）在 `docs/rules/`，这里只放"怎么做"。

判定标准：一个流程要成为能力，必须由证据表明它**已重复发生 ≥2 次**且步骤顺序固定。
一次性操作不立 SOP。

| 能力 | 触发条件 | 用途 | 证据 |
| --- | --- | --- | --- |
| [`bounded-study-run`](../../.agents/skills/bounded-study-run/SKILL.md) | 要跑一个受预算约束的实验并留下可核查证据 | 冻结协议 → 训练 → 评估 → 归档代码身份 | 11 个 workflow 同一形状；8 个模块同一模式（E-055, E-065） |
| [`pinned-artifact-replay`](../../.agents/skills/pinned-artifact-replay/SKILL.md) | 要用已归档的产物重跑或核对历史结果 | 校验源 hash 与 receipt → 重建 → 离线评估 → 写接受标记 | 3 个 replay 模块 + 3 个 workflow（E-129, E-130） |
| [`real-data-acquisition`](../../.agents/skills/real-data-acquisition/SKILL.md) | 要引入新的真实数据或重建派生数据集 | 只读 preflight 报告 → 人审 → 显式 `--write` 授权 → 发布契约 | preflight 被每个 pilot 复用（E-051, E-052） |
| [`result-freeze`](../../.agents/skills/result-freeze/SKILL.md) | 某个结果要被引用、写进报告或归档 | 复现性核查 → digest 核对 → 登记 | 3 个迭代的 docs 记录都含 run id 与 digest（E-066, E-067） |
| [`environment-rebuild`](../../.agents/skills/environment-rebuild/SKILL.md) | 换机器/容器、`ModuleNotFoundError`、CUDA 不可用 | 虚拟环境搭建顺序 + 未声明依赖清单 | 2026-09-24 实测重建；4 个未声明依赖 + torch≥2.8 不兼容（E-157, Q-011, Q-012） |
| [`ci-workflow-triage`](../../.agents/skills/ci-workflow-triage/SKILL.md) | CI 失败、运行 pending/取消、要判断"算不算通过" | 18 条 workflow 分类 → 触发标签 → run-id pin → 禁网验证 | 18 条 workflow；`R7_MANUAL_ITERATION.md:17-21`；`R7_TASK_QUEUE.md` |
| [`issue-lifecycle`](../../.agents/skills/issue-lifecycle/SKILL.md) | 开始/推进/关闭 issue，或判断能否标 DONE | 四态推进 + 绑定精确 SHA + 证据要求 | `R7_TASK_QUEUE.md:59` + 6 个真实 issue（#53–#58） |

## 未立为 SOP 的候选（仅观察）

以下流程有重复迹象但证据不足，**不为它们建 skill**：

- **控制器标定 → 策略选择 → 测试**：有工具，但 `R7_TASK_QUEUE.md:51` 记录
  "严格策略回退到全深度"，没有成功的真实重复，只在合成 fixture 上跑过。
- **成对 block-bootstrap 比较**：只有一次真实比较（#54 vs #56），工具重复而协议不重复。
- **checkpoint 恢复协议**：文档完整（`docs/R7_LOCAL_RUNNER.md`），但只有一个使用示例。
- **边界/ACC 指标流水线**：实现且有测试，但只在合成数据上跑过。

## 相关的执行面

- `tools/check_conventions.py` — 20 条阻断规则，已接入 CI 与 Stop hook。
- `tools/agent_hooks/` — 三个 PreToolUse/PostToolUse hook（保护路径、破坏性 git、
  model digest 提醒），配置在 `.zcode/config.json`，见 `AGENTS.md`。
