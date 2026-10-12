# 0046 文献检索通道（arxiv-researcher）与它同网络通道的边界

- **日期**：2026-10-12
- **状态**：accepted
- **代码 SHA 范围**：`370dca0`（起点）→ 本次提交
- **依据证据**：`docs/decisions/0018-web-research-route.md`（外部检索单一出口的既有决定）；
  `docs/rules/external-sources.md`（R-049/R-050 正文）；`docs/rules/EVIDENCE.md`（E-182 – E-188）；
  `.zcode/agents/arxiv-researcher.md`；`.zcode/agents/web-researcher.md`；
  `tools/agent_hooks/guard_web_research_route.py`；`tests/test_agent_hooks.py`；
  `docs/plans/0023-literature-research-subagent.md`；用户级 MCP 配置
  `/home/gpu2/.zcode/cli/config.json` 的 `mcp.servers.arxiv`

## Context

用户在本机（**用户级**，不是仓库级）装了一个专门检索 arXiv 的 MCP：

- server 名 `arxiv`，命令 `uvx --from arxiv-mcp-server[pdf]@0.8.2 arxiv-mcp-server
  --storage-path /home/gpu2/.arxiv-mcp-server/papers`，`env.MAX_RESULTS=10`；
- 实测 `arxiv-mcp-server@0.8.2` 暴露 **19 个工具**（源码 `arxiv_mcp_server/server.py` 的
  `list_tools()`），本机日志另有独立佐证：`mcp.server.connected … "mcpServerName":"arxiv",
  "toolCount":19`。

这让「一手科研文献」第一次有了**比通用网页搜索更合适**的通道。既有决定 0018 立的是
**网络检索的唯一出口**：`web-researcher` 持有 `WebSearch`/`WebFetch`，主链路被 PreToolUse 闸门
（R-049）拒绝，不可用时降级主链路 `curl`。该决定的三条约束仍然成立且未被本决定触碰：

1. 闸门只匹配 `WebSearch`/`WebFetch` 两个工具名；
2. 闸门只能管主链路（工具事件载荷没有子智能体身份字段，E-186；子会话很可能不跑 hook，E-187，**推测**）；
3. `tests/test_agent_hooks.py` 断言 `web-researcher` 的工具集**恰好**等于 `{WebSearch, WebFetch}`，
   且 `planner` 不得持有这两个工具（「单一出口」的结构化）。

现状缺口：本仓最近的架构方向判断（`docs/plans/0022-s3-architecture-next-round.md` §6）恰恰
**依赖文献旁证**，而当时的取证方式是「`web-researcher` 与 backup 都不可用 ⇒ 主链路 `curl` 打
arXiv API」——一条**降级路径**被当成了常规路径，检索结果也没有「读的是摘要还是正文」这一层区分。

## Decision

**采用：在既有的网络通道之外，新增一条与之互不重叠的「文献通道」——`arxiv-researcher`
子智能体，只持有 arXiv MCP 的只读工具；网络通道维持 0018 的单一出口不变。** 具体：

1. **新子智能体**：`.zcode/agents/arxiv-researcher.md`，工具集是 arXiv MCP 的 **9 个只读工具**
   （`search_papers`、`get_abstract`、`download_paper`、`read_paper`、`get_paper_outline`、
   `read_paper_section`、`search_paper_text`、`citation_graph`、`list_papers`）。
   **不含** `reindex`/`semantic_search`（会写嵌入）、`watch_topic`/`check_alerts`/`list_watches`/
   `unwatch_topic`（订阅态），**更不含** `WebSearch`/`WebFetch`。
2. **两条通道的边界（互不重叠）**：一手科研文献（论文的摘要/正文/公式/实验设置/引用关系）
   → `arxiv-researcher`；通用网络事实（库用法、版本变更、官方文档、API 参考、报错信息）
   → `web-researcher` → `web-researcher-backup` → 主链路 `curl` 降级。两条通道的**引用纪律同一条**：
   R-050（URL + 访问日期，摘要不等于证据，一手/二手分开）。
3. **接线**：`.agents/skills/web-research/SKILL.md` 把委派阶梯写成两层；
   `AGENTS.md` 的「外部信息」硬约束与「按任务加载」表各补一条；
   `docs/skills/README.md` 的 `web-research` 行与执行面补记。
4. **不改规则正文**：R-049/R-050 的文本**不变**，闸门 `guard_web_research_route.py` 的
   matcher **不变**。文献通道靠**约定 + 技能**约束，不新增 hook。

## Consequences

**变容易的：**

- 架构/机制类判断有了**一手**取证路径：能读正文而不是只读摘要，能按章节精读、能全文检索定位
  公式与超参，能看引用图判断一条研究线是否饱和；取证成本比「委派网页搜索 + 人工回一手」低。
- 两条通道职责分明：文献问题不会再去打通用搜索，通用问题也不会去查 arXiv。
- 0018 的结构化断言**不受影响**——它只钉 `web-researcher` 与 `planner`，新增 agent 不改变这两个集合。

**变难 / 代价（如实列出）：**

- **文献通道没有闸门**：R-049 的 hook 只匹配 `WebSearch`/`WebFetch`，**不覆盖 MCP 工具**
  （工具名是 `mcp__arxiv__*`）。因此主链路**可以**直接调用 `mcp__arxiv__*` 而不被拦——这条通道
  的纪律**只靠约定与技能**，不是机械强制。这是本决定最主要的代价，如实记录。
- **新增一处「真相」的风险**：现在有两份检索纪律（`web-researcher` 与 `arxiv-researcher`），
  将来改引用纪律要同时改两处，否则会漂移。缓解：R-050 的正文只有一处，两份 agent 定义都指向它。
- **MCP 是用户级配置、不在仓库里**：干净克隆上按名委派 `arxiv-researcher` 会成功（agent 文件被
  跟踪），但**它能不能真正检索取决于该机器是否装了 `arxiv` MCP**。仓库侧无法保证这一点——
  这也是它必须带显式降级回报（「arxiv MCP 不可用：<原因>」）的原因。
- **生效时机未验证**：`.zcode/config.json` 的 hook 配置是会话启动时读取（决策 0002 附录实测边界），
  **agent 定义是否同理我没有证据**。因此「新增 agent 后当前会话即可委派」是**未验证**的；
  变通办法（重启会话，或主链路直接调用 `mcp__arxiv__*`）在计划文档里标为未验证。
- **两个子智能体的模型不同**：`web-researcher` 用 `anthropic/claude-sonnet-5`，
  `arxiv-researcher` 也用同一档，但它们对同一问题的结论可能不同——两条通道的结论**不互相引用**，
  各自回一手核对。

**备选方案与否决理由：**

- **把 arXiv MCP 的工具并进 `web-researcher`**：否决。`tests/test_agent_hooks.py` 把
  `web-researcher` 的工具集钉成**恰好** `{WebSearch, WebFetch}`（这是 0018「单一出口」的结构化），
  并进即破坏该断言，也破坏「单一出口」的语义。
- **给新 agent 同时配 `WebSearch`/`WebFetch`**：否决。同上，且会让外部检索出现第二个网络出口。
- **把 arXiv MCP 写进仓库级 `.zcode/config.json`**：否决。MCP 是用户级安装，仓库里重复声明会制造
  第二处真相；仓库只记录**如何使用**它（本决定 + agent 定义 + 技能）。
- **不建 agent，只在技能里写「用 `curl` 打 arXiv API」**：否决。那正是本决定要修掉的现状——
  把降级路径当常规路径，且拿不到正文/章节/引用图。

## 复核触发

- 若出现「两个通道结论互相引用、或文献通道的结论被当作证据直接写进仓库」≥1 次 → 在技能里加
  强制回一手核对的步骤，并考虑给 MCP 通道补一条 PostToolUse 留痕提醒（仿 `note_external_fetch.py`）；
- 若客户端开始给主链路之外的 MCP 工具也上闸门 → 把文献通道也纳入机械约束；
- 若 `arxiv` MCP 在多数机器上不可用（成为常态）→ 重估本通道是否仍值得作为一等路径；
- 若 agent 定义被证实需要重启会话才生效 → 在技能与计划文档里把它写成**已知前提**而不是变通。
