# 0018 并行科研能力改造与终极 goal 提示词（从真实 S3 接续）

**状态：文档交付与本地验证中（2026-10-08）；科研未执行、未提交推送。**
起点 `99e94a32f4638edaabfef64cc9425061ba21d673`，分支 `r7/weather-reasoning`。
唯一主计划与可复制 objective：`docs/goals/main-model-climatology-campaign.md`；本计划不是第二个 master。

## 1. 本轮范围与真实起点

用户要求：给出对应「沿可思考主模型持续迭代直到超过 train-only climatology」的终极 goal 提示词，
并**适当改造项目，使其能使用 workflow、做并行科研设计，能并行才并行、并行前先判断可行性**；
计划中要显式写出何时用子代理、何时用技能并写出技能名；实验决策权与时长全部下放；负结果要自主排障。

本轮只改**治理/技能/文档层**：不新增机械检查器，不动生产模型与训练代码，不下载/发布/clone/训练/评分/
查 GPU，不暂存提交推送，不改冻结证据、科学合同、index/brief、用户配置，不关闭 issue。本轮新增 0 GPU-h。

已核起点：

- 仍 **S3**；索引 63 条；HEAD `99e94a32f4638edaabfef64cc9425061ba21d673`，无未推送提交。
- **确认并已修正的漂移**：账本最后一行是 ordering 轮（0.6848 → 22.6281），**缺 index63 行**，
  而 §8 正文与索引文件早已登记 index63；`campaign-state` 亦停在 22.6281/−2.6281。
- 索引末条真实 id 为 **`s3-climatology-anomaly-anchor`**（negative / not-candidate，0.6859）；
  master §8 早期正文写作 `s3-climatology-anchor` 属该窗口简写，账本行按真实 id 引用。
- 模型身份已变更：`abe5439`（`[model-digest-change]`）新增 train-only climatology anchor，
  新 model digest `d3fb58db…`；旧 checkpoint 保持旧身份，经 pins + model-only 迁移复用。
- 最近两轮（ordering、anchor）均 negative；test 未评分、r=0、S4 未进。
- 记忆记录 2026-10-08 曾有一版 goal+workflow 组合方案被用户叫停回滚、未给失败理由；本次由用户明确要求重启。

## 2. 授权与边界

沿决策 0030/0038/0039 常设下放（方法/架构/数据/预算/排障/节点推进），无总 GPU-h 上限，逐轮冻结
`planned_seconds`/`hard_cap_seconds`。保留边界不变：付费/租卡、GPU 独占、方向改变、main 合并/发版、
破坏性操作需具名许可；新 issue 关闭/main 写入不继承旧六 issue 许可。

本轮新增决策 **0042**：把并行科研与 workflow 编排立为能力，但**范围限治理/技能/文档层**，
不加机械检查器（`docs/skills/README.md:9` 要求重复 ≥2 次才立 SOP）。

## 3. 交付物

### D1 新技能 `.agents/skills/parallel-research-workflow/SKILL.md`

含：何时使用/不适用；**并行准入七 gate**（假设独立、产物隔离、计算资源、数据实例、可变状态、
可独立验证、多重比较纪律），任一不满足即串行并写理由；**编排形态分级**（串行 → 普通 Agent 扇出 →
dynamic workflow 脚本，仅在确有控制流时）；**安全隔离**（子代理可能不跑 hook，默认只读、写只写各自
排他 `outputs/`；GPU 仍由统一父调度按共驻余量串行）；**科学完整性**（每臂独立 protocol/digest/证据/
index record；并行≠共享预算；test 封存与 r/alpha 不放宽）；**降级**（workflow/planner 不可用退回串行扇出）。

### D2 登记与路由

`docs/skills/README.md` 能力表加一行（用 `docs/plans/0004-r7-main-model-v2.md:183`、
`docs/goals/s2-climatology-mechanism-screening.md:54`、`docs/goals/n4-m5-confirmation.md:75`
三次真实并行先例满足 ≥2 次准入）；`AGENTS.md` 任务路由表加一行 + 硬约束加一条并行准入；
`docs/rules/CHANGELOG.md` 顶部留痕。

### D3 决策 0042

`docs/decisions/0042-parallel-research-workflow.md`：Context/Decision/Consequences，如实记录
「曾被回滚后经用户明确要求重启」、dynamic workflow 在本仓从未使用（E-163/E-164）、子代理可能绕过 hook、
并行放大多重比较与资源争用、账本易漏行（现有校检器发现不了缺行）。

### D4 主计划 §0 objective 刷新

从 S3 接续（index63 / 23.3140 / 模型身份已变更 / anchor 已 negative 终结），写入并行准入与编排纪律，
技能清单增至 12 项，保留全部既定科学门；落盘单段实测 **3964 code points**（≤4000）。

### D5 账本与状态机械同步

账本补 index63 行（0.6859），合计 22.6281 → **23.3140**；`campaign-state` `used 23.314 / remaining −3.314`。
只做机械同步，不改科学判据与冻结证据。

## 4. 验证

- `check_goal_brief.py`：master 0 失败 0 建议；objective 单段 3964 code points。
- `check_campaign_state.py`：默认旧 master 与显式新 master 均 0 失败（4 条既有 notes）。
- `check_conventions.py --quiet`：37 条阻断 0 失败；`git diff --check` 干净。
- 四个完整 CPU 模块（goal_brief / campaign_state / conventions / verify_r7_evidence_index）：
  见「实际结果」；venv `-B`、禁网、隐藏 CUDA、仓外 tmp、无 cache。
- 独立只读审阅：技能内容 + objective 科学门完整性。
- 不跑全量 suite、wheel 或远端 CI。

## 5. 风险

- objective 压缩漏科学门 → 逐项机械核对 + 独立审阅。
- 新技能违反 `docs/skills/README.md:9` → 用三次真实先例论证，并在技能内写明「首次真实并行轮必须留痕验证」。
- 账本同步写错算术或引用错 id → 用 C-02/C-03 实核（本轮 C-03 已抓到一次 id 误引用并修正）。
- 子代理绕过 hook → 技能内写死只读/隔离写，不依赖 hook，由 Stop hook 与 CI 兜底。

## 6. 明确不做

不改生产模型/训练代码；不加机械检查器；不启动实验/下载/发布/clone/GPU；不 commit/push；
不改冻结证据、科学合同、index/brief、账本历史行、用户配置；不关闭 issue；不新建第二个 master。

## 实际结果

- **完成情况**：D1–D5 全部落实（新技能、决策 0042、索引/路由/CHANGELOG 登记、§0 objective 刷新、
  账本与状态机械同步）。科研/下载/发布/clone/GPU 未执行；本轮新增 0 GPU-h；未提交推送。
- **与计划的差异**：objective 落盘 3964 code points（计划只要求 ≤4000）；账本行按索引真实 id
  `s3-climatology-anomaly-anchor` 引用，而 master §8 早期正文的简写 `s3-climatology-anchor` 保持原样
  （历史进度文本不静默改写），差异已在新的 §8 交接块与本计划中写明。
- **验证**：
  - `check_goal_brief.py`：master **0 失败 0 建议**；objective 落盘单段 **3964 code points**（≤4000）。
  - `check_campaign_state.py`：默认旧 master 与显式新 master 均 **0 失败 / 4 条既有 notes**；
    账本 0.6859 累加至 23.3140、`remaining −3.3140` 与状态块一致。
  - `check_conventions.py --quiet`：**37 条阻断 0 失败**；对 8 个交付路径 `--paths` 模拟提交树亦 0 失败
    （未暂存）；`git diff --check` 干净。
  - `verify_r7_evidence_index.py`：**63 records** + canonical brief 通过；index/brief 与 11 个只读身份文件
    hash/bytes 未变。
  - 四个完整 CPU 模块（`test_check_goal_brief`/`test_check_campaign_state`/`test_check_conventions`/
    `test_verify_r7_evidence_index`）三轮实录：**attempt01 213 tests / 1 failed / 0 skip**
    （失败为 `test_decision_index_matches_decision_files`：新增 ADR 0042 未登记进 `docs/decisions/README.md`）；
    补登记后 **attempt02 213 passed / 0 failed / 0 skip**（39.615555 秒）；在独立审阅修复后的**最终内容**上
    **attempt03 213 passed / 0 failed / 0 skipped**（39.231717 秒，planned600/hard1200、overrun 0）。
    项目 venv `-B`、`-p no:cacheprovider`、插件 autoload 关闭、测试进程 socket 拒绝出网、CUDA 隐藏、
    临时产物仓外。**attempt01 失败与日志原样保留，不追认；attempt02 的绿发生在修复前版本，仅作过程记录。**
  - 回执：attempt01 `/tmp/r7_parallel_skill_20261008_7v7e464i/cpu_verification_cjykigu9/verification_receipt.json`；
    attempt02 `…/cpu_verification_attempt02_vwvp3n8v/verification_receipt.json`；
    attempt03 `…/cpu_verification_attempt03_final_l3c1qz_6/verification_receipt.json`（各含 JUnit/log digest、
    模块、时间/预算/费用与 limitations）。本地 tmp 回执不是新科学索引 record。
  - 未跑全量 suite、wheel 或远端 CI；既有绿色 CI 不覆盖本轮未提交改动。
- **独立审阅**：独立只读审阅提出 3 条必修，均已在提交前修复：
  （a）本计划的验证结果曾指向不存在的「下方补记」→ 已改为内联实测数字；
  （b）准入 gate 计数不一致（摘要写 4 条、技能/ADR 写 7 条）→ 三处摘要统一为「七 gate」并列出全部七项；
  （c）`docs/skills/README.md:9` 的「重复 ≥2 次」准入未被证据支持 → 已在技能、ADR 与索引行显式标注
  **前瞻立 SOP** 例外及「首次真实并行轮必须留痕验证」的强制要求。
  另修正技能/ADR 中 `AGENTS.md:153-155` 的行号漂移（本轮插入两行后应为 `:155-156`）。
  独立审阅不是科学接受，也不是最终 goal verifier。
- **影响**：仅治理/技能/文档层变化——新增 1 个技能目录、1 份 ADR、1 份计划与 5 处索引/路由/留痕；
  公开接口、默认参数、模型 digest、数据与结果、依赖、安全 hook/配置/凭据均无改动。新增文件未跟踪，
  治理层未提交时干净克隆仍不可用。
- **遗留**：并行能力尚无真实并行轮次的实证；dynamic workflow 通道在本仓从未使用，首次使用须留痕验证；
  最终科学目标未通过，仍 S3，test 未评分、r=0、S4 未进，保留授权边界不变。
