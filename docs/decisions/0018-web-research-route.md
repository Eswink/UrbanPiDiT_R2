# 0018 外部检索的唯一出口（web-researcher）与引用纪律

- **日期**：2026-09-28
- **状态**：accepted
- **代码 SHA 范围**：`73c61c9`（起点）→ 本次提交
- **依据证据**：`docs/rules/external-sources.md`（R-049/R-050 正文）；`docs/rules/EVIDENCE.md`
  第五遍（E-182 – E-188）；`docs/rules/OPEN_QUESTIONS.md` Q-013；`.zcode/agents/web-researcher.md`；
  `.zcode/agents/planner.md`；`tools/agent_hooks/guard_web_research_route.py`；
  `tools/agent_hooks/note_external_fetch.py`；`tests/test_agent_hooks.py`；`.zcode/config.json`

## Context

用户为本仓新增了 `web-researcher` 子智能体（工具集仅 `WebSearch`/`WebFetch`），要求
「需要网页搜索时，务必调用这个工具；它破产了就回主链路，主链路优先用 `curl`（本机在国内，
部分页面直连不可达）」。取证发现三件与直觉不符的事实（全部实测或源码核对）：

1. **当时不存在任何门禁**：`tools/`、`tests/`、`.github/`、`scripts/`、`training/`、`model/`
   内 `WebSearch`/`WebFetch` **0 命中**（E-182）——「务必」没有任何机制支撑。
2. **「唯一出口」当时不成立**：`web-researcher.md` 甚至**未被 git 跟踪**（干净克隆上按名
   调不到），而 `planner.md` 的工具集里也有这两个工具（E-183）。
3. **闸门只能管主链路**：工具事件的 hook 载荷**没有子智能体身份字段**（E-186，源码核对）；
   当前构建下子会话很可能根本不创建 hook runner（E-187，**推测**，未实测——安全探针需要一次
   会被拒绝的写操作，代价不可接受，已登记 Q-013）。

另有两条真实的引用纪律先例：`cn.bing.com` 的 WebFetch 搜索摘要曾对 HRCLDAS/SMBFD 返回
**完全无关**结果，仓库当时的处置是「搜索摘要不得作为证据，每条可达性以实测端点为准」
（E-184）；`docs/*.md` 现有 26 行 URL，其中 14 行外部引用**没有访问日期**（E-185）。

## Decision

**采用：外部信息只有一条出口——`web-researcher` 子智能体；主链路的 `WebSearch`/`WebFetch`
由 PreToolUse 闸门拒绝；它不可用时降级为 Bash `curl`（永不阻断）；引用必须可核查。** 具体：

1. **规则**：新增 `docs/rules/external-sources.md` 承载 R-049（路由）与 R-050（引用可核查，
   A 类，均如实登记为 checker 不可判定）。
2. **闸门**：`tools/agent_hooks/guard_web_research_route.py`，matcher `WebSearch|WebFetch`，
   主链路 → 退出码 2 并在拒绝文案里给出替代路径（委派命令、`curl` 降级、技能指针）；
   `session_id` 前缀 `sess_subagent_` **放行**（防自锁，不是规则的例外）。
3. **留痕**：`tools/agent_hooks/note_external_fetch.py`（PostToolUse/Bash）在 `curl`/`wget`
   出现外部 URL 时发一条 `systemMessage` 提醒按 R-050 记录；**只提示，不阻断**——降级路径
   必须畅通。
4. **唯一出口的结构化**：`planner.md` 移除 `WebSearch`/`WebFetch`（并在其边界里写明需要外部
   事实时进 `open_questions`）；`web-researcher.md` 纳入版本控制。`tests/test_agent_hooks.py`
   断言闸门的工具集与 agent 定义**不得漂移**，且 `.zcode/agents/` 下没有第二个带网络工具的
   agent。
5. **能力**：`.agents/skills/web-research/SKILL.md`（委派模板 → 回收与一手核对 → 留痕 →
   破产降级，含本机可达性实测表）。

## Consequences

- **代价**：主链路多一次委派往返；对抗性场景（登录/反爬页面）能力下降——`WebFetch` 比 `curl`
  更擅长把页面转成文本，但它在主链路被封，只能靠子智能体完成。
- **收益**：外部信息只有一个可审计的出口，一次委派就带上「URL / 一手优先 / 冲突 / 未能确认」
  的结构；主链路的随手抓取不再可能悄悄进入结论。
- **已知局限（如实记录）**：闸门只在本仓库、经 ZCode 生效（直接命令行不受约束），且**不是
  沙箱**；它管不到子会话（E-186/E-187）；引用纪律不机械判定，历史 14 行无日期引用**不回填**。
- **生效时机**：`.zcode/config.json` 的 hook 配置在**会话启动时**读取——新闸门下一次会话
  才生效（决策 0002 附录的实测边界）；本会话内主链路 `WebFetch` 仍可调用。
- **回退**：单次 revert；或从 `.zcode/config.json` 删除对应条目（下次会话生效）。

## 复核触发

- 客户端开始给子会话跑 hook、或工具事件载荷加入子智能体身份字段 → 改为按身份放行，
  并做一次 Q-013 的零副作用探针；
- `web-researcher` 长期不可用（成为常态）→ 重估强制路由是否仍值得；
- 引用纪律被违反 ≥2 次 → 立仓库内引用台账（URL / 访问日期 / 抓取方式 / sha256）。
