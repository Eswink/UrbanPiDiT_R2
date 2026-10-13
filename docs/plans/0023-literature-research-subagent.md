# 0023 文献检索子智能体（arxiv-researcher）与它的接线

**状态：定稿归档。** 本文件记录「新增一条一手文献检索通道」的设计与落地结果。
决定记录见 `docs/decisions/0046-arxiv-literature-research-route.md`；能力定义在
`.zcode/agents/arxiv-researcher.md`。**不产生任何科学结论，不涉及任何实验。**

---

## 1 为什么做

用户为本机（用户级）装了专门检索 arXiv 的 MCP。而本仓最近的架构方向判断
（`docs/plans/0022-s3-architecture-next-round.md` §6）**依赖文献旁证**，当时却是
「`web-researcher` 与 backup 都 provider 不可用 ⇒ 主链路 `curl` 打 arXiv API」——
**降级路径被当成了常规路径**，而且拿不到「读的是摘要还是正文」这一层。

因此需要：一条**一等**的文献通道，且它不能破坏决定 0018 立下的「网络检索单一出口」。

## 2 设计

### 2.1 两条互不重叠的通道

| 通道 | 承担 | 出口 |
| --- | --- | --- |
| 文献 | 一手论文的摘要/正文/公式/实验设置/引用关系 | `arxiv-researcher`（arXiv MCP，只读 9 工具） |
| 网络 | 库用法、版本变更、官方文档、API 参考、报错信息 | `web-researcher` → `web-researcher-backup` → 主链路 `curl` 降级 |

引用纪律**只有一条**：R-050（URL + 访问日期；摘要不等于证据；一手/二手分开）。

### 2.2 工具面（为什么是这 9 个）

`arxiv-mcp-server@0.8.2` 暴露 19 个工具，本决定只取**只读且切题**的 9 个：

- 取：`search_papers`、`get_abstract`、`download_paper`、`read_paper`、`get_paper_outline`、
  `read_paper_section`、`search_paper_text`、`citation_graph`、`list_papers`
- 弃：`reindex`、`semantic_search`（会写嵌入）、`watch_topic`、`check_alerts`、`list_watches`、
  `unwatch_topic`（订阅态，与「一次检索一件事」的委派形态不符）
- **绝不取**：`WebSearch`、`WebFetch`（决策 0018 单一出口；`tests/test_agent_hooks.py` 把
  `web-researcher` 的工具集钉成恰好 `{WebSearch, WebFetch}`）

### 2.3 为什么必须带显式降级

MCP 是**用户级**配置、不在仓库里。干净克隆上按名委派 agent 会成功（文件被跟踪），但**能不能真
检索取决于该机器装没装 `arxiv` MCP**。所以 agent 契约要求：不可用时如实回报
「arxiv MCP 不可用：<原因>」，**不得用记忆顶替**，由调用方按 `web-research` 阶梯降级。

## 3 接线（四处）

1. `.zcode/agents/arxiv-researcher.md` —— 新子智能体（能力本体）。
2. `.agents/skills/web-research/SKILL.md` —— 委派阶梯写成两层（先文献、后网络）。
3. `AGENTS.md` —— 「外部信息」硬约束段与「按任务加载」表各补一条（常驻契约，每轮注入）。
4. `docs/skills/README.md` —— `web-research` 行与「相关的执行面」补记。

## 4 与既有约束的关系（逐条核对）

- **R-049 闸门**：matcher 仍是 `WebSearch|WebFetch`，**未改**。因此它**不覆盖** `mcp__arxiv__*`——
  文献通道**没有机械闸门**，只靠约定。这是本决定最主要的代价，已在 ADR 的 Consequences 里如实记录。
- **`tests/test_agent_hooks.py`**：两条断言只涉及 `web-researcher` 与 `planner`，新增 agent 不改变
  这两个集合，故不触发漂移。已实跑验证（见下）。
- **R-037**：`.zcode/agents/` 是治理目录，新文件必须被 git 跟踪——已提交。
- **R-050**：正文未改；agent 契约直接引用它。

## 5 不干扰正在跑的轮次（设计时的安全论证）

- 每个 seed/reading worker 调用的 `validate_protocol` 只重算 `execution_files()`，范围是
  `scripts/*.py` 与 `training/`、`model/`、`data/` 下的 `.py`——**不含 `docs/`、`.zcode/agents/`、
  `AGENTS.md`**。
- archive 阶段的 ZIP comment 只与**冻结时写进 `protocol.json` 的 `code_commit`** 比对
  （`scripts/study_r7_s3_wide_interior_supervision.py::_archive`），**从不与实时 HEAD 比对**。
- 故本次全部改动（文档 + 新 agent）**不可能**让在跑的 seed 失败。落地时轮次处于 `stage seed 41`，
  登记阶段尚在数小时之后，不存在与登记员抢同一文件的窗口。

## 实际结果

- **已创建** `.zcode/agents/arxiv-researcher.md`：frontmatter 与既有三个 agent 同格式
  （`name/description/color/model/thoughtLevel/tools/injectAgentsMd`），
  `model: anthropic/claude-sonnet-5`、`thoughtLevel: high`、`tools` 为 2.2 列的 9 个
  `mcp__arxiv__*`。**未声明** `WebSearch`/`WebFetch`。
- **已接线** 第 3 节四处。
- **已记录** `docs/decisions/0046-arxiv-literature-research-route.md`（accepted）。
- **已实跑验证**：`tests/test_agent_hooks.py`（含「闸门与 agent 定义不得漂移」的两条断言）、
  `tests/test_check_conventions.py`、`check_conventions.py --quiet`（R-037 要求 `.zcode/agents/`
  被跟踪）、`check_goal_brief` / `check_campaign_state` / `verify_r7_evidence_index --check-brief`、
  `r7_round verify`、`round-guard`、`commit-guard`。**结果见本次提交的 CI run（`R7 CPU CI` 九主步骤）。**
- **未验证 / 如实标注**：
  1. **新 agent 是否免重启生效未验证**。hook 配置按 `AGENTS.md` 是「会话启动时读取」，
     agent 定义是否同理**我没有证据**。两条变通**都标为未验证**：(a) 重启该会话；
     (b) 主链路直接调用 `mcp__arxiv__*`（R-049 闸门只匹配 `WebSearch`/`WebFetch`，不会拦它）。
  2. ~~**没有做过一次真实的端到端委派**~~ **已于 2026-10-13 完成首次真实端到端委派**：在
     `r7/weather-reasoning` 会话里按名委派 `arxiv-researcher`（检索「保函数网络扩张」），
     它实际调用 `mcp__arxiv__search_papers` → `mcp__arxiv__get_abstract` **两次、全部成功无报错**，
     命中 arXiv:2410.11038（*Towards a More Complete Theory of Function Preserving Transforms*,
     2024）并按契约回传摘要概括。故「按名委派可成功、能按契约回报」**已由实跑确认**。
     这同时说明：**本会话内该 agent 定义生效**（但「新增 agent 是否免重启生效」仍无证据，
     见第 1 条）。
  3. **工具名来自源码与日志两处独立核对**（`server.py` 的 `list_tools()` + 本机
     `mcp.server.connected` 的 `toolCount:19`）。截至 2026-10-13 **已实调 2 个**
     （`search_papers`、`get_abstract`，均成功）；其余 7 个
     （`download_paper`/`read_paper`/`get_paper_outline`/`read_paper_section`/`search_paper_text`/
     `citation_graph`/`list_papers`）**仍未逐个实调**；首次使用其中任一时以运行时报错为准并回填本文件。
- **未做**：不改 R-049/R-050 正文；不改闸门 matcher；不把 MCP 写进仓库级配置；
  不动正在跑的轮次的任何 execution file；不启停任何进程；不占 GPU。
