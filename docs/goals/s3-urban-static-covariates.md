# 目标：S3 城市静态协变量获取与发布（L2 城市证据落地）

<!-- round-node: S3 -->

本文件是 goal 模式的目标长文源。objective 见 §0；交付物见 §2；判据见 §3；
实施顺序（含 workflow 编排）见 §4；预算与停止见 §5；明确不做见 §6；进度块见 §7。

**状态：prepared（2026-10-27 建立，未启动）。** 授权见
`docs/decisions/0063-urban-static-covariate-acquisition.md`；计划见
`docs/plans/0027-urban-static-covariate-round.md`（计划是不可变记录，本长文是它的执行源）。
`scientific_claim: false`、`test_read: false`。本轮 0 GPU-h、不下载、不改 `campaign-state`；
**启动时**由执行窗口把 `campaign-state.current_round_goal` 指向本文件。

---

## 0. objective（可直接粘贴，单段，实测 2197 字符）

> 在/data/esw/UrbanPiDiT_R2的r7/weather-reasoning执行目标长文docs/goals/s3-urban-static-covariates.md（先完整读它，判据与相位以它为准；授权见决策0063、计划0027）。任务：把具名城市静态协变量（ESA WorldCover 10m v100/v200、GHSL GHS-BUILT-S/BUILT-H/GHS-POP、LandScan可选）取回并聚合到R7冻结ROI（27.0–43.0°N/107.0–123.0°E，格界26.875–43.125N/106.875–123.125E）的65×65/0.25°网格，发布为独立产物（自己的outputs排他目录+static_identity的sha256+manifest+BUILD_COMPLETE.json）。严格按相位：①只读端点核对（HEAD/range GET，记HTTP码/content-length/许可页URL/访问日期）是第一道硬闸门，不可达即留failed-no-fallback丢该层、禁止合成顶替（R-008），并优先核GHSL分辨率（检索在100m与1km间冲突）与Copernicus DEM的AWS镜像是否匿名；②先冻protocol.json（R-006）：格界、目标网格、逐层source/version/license/预期字节/聚合规则/输出布局、磁盘峰值与余量门、网络字节上限、软硬墙钟；③下载器只放data/download/**（R-016唯一网络面）并走http_public.open_public/http_pinned（SSRF+重定向校验+timeout），禁止照抄worldcover_cog.py（它直连rasterio绕过守卫、多tile直接raise、audit记blocked_or_failed）；④聚合到0.25°的reproject/面积加权代码仓内没有、必须从零写，且必须逐字声明这是对真实观测求的派生统计量、写清CRS/格界/spatial_resampling语义，不得与「插值当真值」混为一谈；⑤抽样格从原始瓦片独立重算并精确比对（不复用聚合中间量）；⑥用contracts.fresh_outputs独占发布+BUILD_COMPLETE，写证据页并做四件套登记。用workflow编排：先加载dynamic-workflows技能（未加载时CreateWorkflow拒跑），从.zcode/workflow-drafts/r7-campaign-round.dwf.ts派生.zcode/workflow-drafts/r7-urban-static-round.dwf.ts（草稿已备：语法与dryRun控制流已用node类型剥离+桩facade实测跑通，但未按workflow门面做类型检查、非dryRun路径未跑，用前仍须CreateWorkflow编译一次；其命令字面量假定驱动是scripts/r7_urban_static_round.py且子命令为verify-endpoints/freeze/download/aggregate/recheck/publish，若改了路径或子命令名两边须一起改），maxRounds硬上限自设、stage超时按本轮实测抬到装得下整轮、world.run命令名保持编译期字面量且运行值落盘（文件即参数）、自跑tools/r7_round.py commit-guard（提交前）与verify（登记后）不依赖hook兜底、只推工作分支r7/weather-reasoning（禁main/force/merge/关issue）；它是有界、可恢复的批量编排器而不是守护进程，被打断按ResumeWorkflowRun续；已知边界（实测）：后台workflow仍在跑时轮末goal完成校验可能不触发（是defer不是失败、也没有校验器读数）、卡片可见性只从运行API观测、探针未覆盖GPU轮/freeze/commit/登记。预算与停止：本轮0 GPU-h；网络/磁盘/墙钟三项先冻结，名义planned21600s/hard86400s，按首次实测另立更正决策重定（只改时间、不动科学门、不在运行中改硬限）；准入上本轮的CPU/网络与R1的GPU不重叠、产物排他⇒允许并行，但用户指定「等R1跑得差不多再启动」，启动那一刻才把campaign-state.current_round_goal指向本长文。禁止：读test、合成或伪造数据顶替失败层、动data/raw|interim|processed与归档快照、改model/**或重建store或加state通道（接线是下一轮）、付费/租卡/独占、cron/守护/会话外续跑、事后放宽判据、把聚合派生量当观测真值。GPU默认共驻，只读核余量，不对邻居进程发任何信号。只在三种情况停：触保留授权边界、达已冻结科学停止条件、确无可安全独立推进的工作；停时必须在§7写「为何停+恢复第一条命令」。每步给可核查证据（命令+输出+digest），估计值不得写成实测值；结束按AGENTS.md汇报改动/验证/接口/数据影响/兼容性/依赖/未做/下一项。不要自行宣布目标完成。

---

## 1. 现状与前置

- **节点**：S3（主模型超气候态 campaign）。本轮**不改**任何科学门、**不改** `model/**`、
  **不重建** store —— 它是数据准备轮，产出的是静态协变量，不是模型读数。
- **为什么现在做**：用户要求"帮我去拉取相关的数据（尽量和我们现在的数据相匹配）"，
  并明确**动态 1 km 真值延后**、**LLM 式推理延后**。授权与范围见决策 0063。
- **ROI 必须用 R7 冻结的**：`data/download/read_plan_frozen.py:34-36` 的
  27.0–43.0°N / 107.0–123.0°E，65×65，0.25°（格心 lat 43.0→27.0 降序、lon 107.0→123.0 升序
  ⇒ 格界 26.875–43.125N / 106.875–123.125E）。`docs/DATA_SPEC.md:3-7` 的京津冀盒是**旧口径**，
  不用。
- **不是 issue #9**：`docs/R7_URBAN_EXTENSION_BLOCKED.md` 的 BLOCKED 针对 **km 级动态真值**；
  静态协变量不在那份审计的结论范围内。它的纪律（禁止升采样 ERA5 当真值、禁止伪造、失败留痕）
  继续适用。
- **集成形态已定**（决策 0063 第 3 条）：独立发布产物 + `static_identity` 绑 `protocol_sha256`；
  **不**把静态量做成第 18 个 state 通道（那会使 `data_identity` 变、与 R1–R5 队列脱钩），
  **不**写进 store 的 `attrs`，**不**做"放进去但不登记"（那是审计洞）。
- **仓内无重采样能力**：`data/preprocess/grid.py:1` 声明 no interpolation；全仓无
  `reproject`/`coarsen`/`WarpedVRT` ⇒ 聚合代码从零写。
- **唯一静态下载器是骨架**：`data/download/worldcover_cog.py`（109 行）——
  `audit/real_data_download_status.json` 记 `blocked_or_failed`，多 tile ROI 直接 `raise`，
  且用 `rasterio.open(url)` 直连、绕过 `http_public` 守卫。**不复用它的网络路径。**
- **workflow 链路已实测可行**：`docs/goals/workflow-in-goal-probe.md` +
  `docs/R7_WORKFLOW_IN_GOAL_PROBE.md`（2026-10-11，run id
  `dwfrun-8b989f84-1043-408e-bd07-4690677135fc`、终态 `completed`、六环全通）。
- **外部检索（摘要，非证据，R-050）**：确认"无需账号"的有 ESA WorldCover、GHSL
  （GHS-BUILT-S / GHS-BUILT-H / GHS-POP）、LandScan（2026 起 CC BY 4.0）；显示"需注册"的有
  Copernicus DEM 官方门户、ALOS AW3D30、SRTM、MODIS MCD12Q1、GPWv4、ERA5-Land、CLDAS、CMFD。
  检索在 GHSL 分辨率（100 m vs 1 km）与 Copernicus DEM 的 AWS 镜像上有**冲突/未确认**，
  列为第一道闸门的核对项。

## 2. 交付物清单

| # | 交付物 | 证据形态 |
| --- | --- | --- |
| D1 | 只读端点核对报告（逐源） | HTTP 码 / `content-length` / 许可页 URL / 访问日期；不可达层的 `failed-no-fallback` 记录 |
| D2 | 冻结的获取协议 | `protocol.json` 的 `protocol_sha256` + 逐层 source/version/license/字节/聚合规则/布局 + 预算 |
| D3 | 具名层的原生分辨率瓦片缓存 | 排他 `outputs/` 路径 + 逐文件 sha256（保留供延后的 1 km 轮） |
| D4 | 聚合到 0.25°/65×65 的静态协变量 | 变量名/单位/布局按 `docs/DATA_SPEC.md` §L2 契约；`static_identity` 的 sha256 |
| D5 | 发布契约 | 目标目录含 `BUILD_COMPLETE.json`；manifest 记 `source_sha256` / 聚合规范 / `limitations` |
| D6 | 独立再派生核对 | 抽样格从原始瓦片重算的逐格比对读数（不复用聚合中间量） |
| D7 | 证据页 | `docs/R7_*.md`：逐层可达性、许可、字节、聚合口径、缺口与限制 |
| D8 | 四件套登记 | 主计划 §8 进度块（含下一动作第一条命令）+ §7 账本行 + `docs/R7_EVIDENCE_INDEX.jsonl` record + `campaign-state` |
| D9 | workflow 运行产物 | run id + 终态 + 相位记录；被打断时写明 `ResumeWorkflowRun` 的恢复命令 |
| D10 | 本长文的 §7 进度块 | 状态机：已完成 / 未做 / 下一动作 |

## 3. 判据与证据来源

- **授权与范围**只引用 `docs/decisions/0063-urban-static-covariate-acquisition.md`；
  **不扩大清单**，不放宽任何已冻结判据。
- **ROI / 网格 / 聚合契约**引用 `docs/DATA_SPEC.md` §L2 与
  `data/download/read_plan_frozen.py:34-36`。
- **数据契约与发布门**引用 `data/preprocess/contracts.py`（`fresh_outputs`）、
  `docs/rules/data-and-artifacts.md`、`.agents/skills/real-data-acquisition/SKILL.md`。
- **禁止伪造**引用 R-008（`failed-no-fallback`、`synthetic_fallback=False`）与
  `docs/R7_URBAN_EXTENSION_BLOCKED.md`（禁止把升采样当真值）。
- **workflow 的边界**引用 `docs/R7_WORKFLOW_IN_GOAL_PROBE.md` §1、决策 0044、计划 0020。
- **机械门**：`python tools/check_conventions.py`（37 条阻断）、
  `python tools/check_goal_brief.py --brief docs/goals/s3-urban-static-covariates.md`、
  `python tools/check_campaign_state.py`、`python tools/verify_r7_evidence_index.py`。
- **不算通过的**：skip / cancelled / queued；未过端点核对的层**不算已获取**；
  `report()` 与 CI 绿都不是科学结论。

## 4. 实施顺序（依赖，不跳步）

### 4.0 workflow 编排（本长文的执行载体）

按决策 0044 与 `.agents/skills/parallel-research-workflow/SKILL.md` 的「campaign 驱动 workflow」节：

- **先加载 `dynamic-workflows` 技能**（未加载时 `CreateWorkflow` 拒跑）；
- 从 `.zcode/workflow-drafts/r7-campaign-round.dwf.ts` 派生
  `.zcode/workflow-drafts/r7-urban-static-round.dwf.ts`（**草稿已备**：语法与 dryRun 控制流已用
  `node --experimental-strip-types` + 桩 facade 实测跑通，但**未按 workflow 门面做类型检查**、
  **非 dryRun 路径未跑** ⇒ 用前仍须 `CreateWorkflow` 编译一次，报错就按报错改；它的命令字面量
  假定驱动是 `scripts/r7_urban_static_round.py` 且子命令为 verify-endpoints / freeze / download /
  aggregate / recheck / publish，**若实现阶段改了路径或子命令名，两边必须一起改**；git-ignored）；
- **`maxRounds` 自设硬上限**；**stage 超时按本轮实测抬到装得下整轮**（本轮的瓶颈是网络而非
  GPU，9.5 h 的模板值需按实测重定）；
- `world.run` 的**命令名保持编译期字面量**，运行值落盘（文件即参数）；
- **自跑确定性闸门**：提交前 `tools/r7_round.py commit-guard <paths>`、登记后 `verify`；
  **不依赖 hook 兜底**（子会话很可能不跑本仓 hook，E-187 推测）；
- **只推 `r7/weather-reasoning`**；禁 main / force / merge / 关 issue；
- **有界、可恢复，不是守护进程**：进程退出即 `interrupted`，按 `ResumeWorkflowRun` 续；
  状态在 §7 与账本里，不在 workflow 里；
- **四条已知边界**（`docs/R7_WORKFLOW_IN_GOAL_PROBE.md` §1）：① 后台 workflow 仍在跑时
  轮末 goal 完成校验可能不触发（是 `defer`，**不是失败、也不等于通过**）；② 卡片"可见"只从
  运行 API 观测；③ 探针**未覆盖** GPU 轮编排 / freeze / commit / 四件套登记 —— 所以本长文必须
  自跑闸门；④ 探针不验证无人值守多轮驱动。

### 4.1 相位（0 GPU-h，网络 + CPU）

1. **只读端点核对（第一道硬闸门）**。逐源做 HEAD / range GET，记 HTTP 码、`content-length`、
   许可页 URL、访问日期。**不可达 ⇒ 留 `failed-no-fallback`、丢该层、不合成**（R-008）。
   优先核两点：GHSL 的确切分辨率、Copernicus DEM 的 AWS Open Data 镜像是否匿名。
2. **冻结获取协议**（R-006）。`protocol.json` 写定：ROI 精确格界、目标网格、逐层
   source/version/license/预期字节/聚合规则/输出布局、磁盘峰值与余量门、网络字节上限、
   软/硬墙钟。
3. **下载**。新下载器放 `data/download/**`（R-016 唯一网络面），走
   `http_public.open_public` / `http_pinned`（SSRF + 重定向校验 + timeout）。
   **不复用 `worldcover_cog.py` 的直连路径。**
4. **聚合到 0.25°**。新写 reproject + 面积加权聚合（原生栅格 → metric CRS → 精确格界统计）。
   必须逐字声明这是对真实观测求的**派生统计量**，写清 CRS / 格界 / `spatial_resampling` 语义。
5. **独立再派生核对**。抽样格从原始瓦片重算，精确比对（不复用聚合中间量）。
6. **发布与登记**。`contracts.fresh_outputs`（独占输出 + `BUILD_COMPLETE.json`）→ 证据页 →
   四件套登记。

## 5. 预算与停止条件

- **预算**：**0 GPU-h**；网络 / 磁盘 / 墙钟三项先冻结。名义 **planned 21,600 s（6 h）/
  hard 86,400 s（24 h）**，按首次实测另立更正决策重定（决策 0055 先例：只改时间、不动科学门、
  **不在运行中改硬限**）。
- **准入**：本轮的 CPU/网络与 R1 的 GPU 资源不重叠、产物排他 ⇒ 七 gate 可过、**允许并行**；
  但用户指定"等 R1（`outputs/r7_s3_fy_year_coverage_matched_20261026_attempt01/`）跑得差不多
  再启动"，该时序由用户定。
- **时长口径**：软预算超了继续并记 `soft_overrun_seconds`；**只有硬上限因时长截断**，
  记 `budget_limited` / `failed` 并全额记账。
- **停止条件（仅三种，决策 0043）**：触保留授权边界（付费 / 租卡 / 独占 / 方向改变 /
  main 合并 / 破坏性 / 无法安全克服的外部硬依赖）；达已冻结的科学停止条件；
  确无可安全独立推进的工作。**停下必须在 §7 写「为何停 + 恢复第一条命令」。**

## 6. 明确不做

- 不下载具名清单之外的层；**不碰**需注册/账号的源（Copernicus DEM 官方门户、ALOS AW3D30、
  SRTM、MODIS MCD12Q1、GPWv4、ERA5-Land、CLDAS、CMFD）。
- 不做 km 级动态真值（issue #9 继续 BLOCKED）；不用任何升采样/插值场冒充真值。
- 不动 `data/raw|interim|processed` 与归档快照；不合成或伪造数据顶替失败的下载（R-008）。
- **不改 `model/**`、不重建 store、不加 state 通道**（接线是下一轮，需要自己的 ADR 与预声明出口）。
- 不读 test；不 main 合并 / force / 关 issue / 删已有产物。
- GPU 默认共驻、只读核余量、**不对邻居进程发任何信号**；不 paid / rental / exclusive；
  不 cron / 守护 / 会话外续跑。
- 不自行宣布科学通过，不把聚合派生量当观测真值，不自行宣布最终 goal 完成。

## 7. 进度块（状态机）

- **状态**：**数据相位已执行并登记**（2026-10-27；`scientific_claim: false`、0 GPU-h）。
  最终 goal 状态不由本文件宣布。**唯一未做的一步是 `campaign-state` 指针**（见下）。
- **已完成（本轮实测）**：
  - **相位 1（第一道硬闸门）**：13 个具名层逐 tile `HEAD`（经 `http_public`），
    `endpoint_report.json`；LandScan 需注册 ⇒ `failed-no-fallback` 丢层；
    GHSL 分辨率冲突解为"3 arcsec 与 30 arcsec 都发布"；Copernicus DEM 的 AWS 镜像实测匿名
    （不在具名清单，不取）。
  - **相位 2**：`protocol.json` 在下载前冻结；digest 见证据页 §2。
  - **相位 3**：195 文件 / 14,484,496,452 B（与冻结预期**逐字节相等**），
    逐文件 sha256；网络面只在 `data/download/urban_static_sources.py`。
  - **相位 4**：从零写的面积加权聚合（WGS84 等积球、逐原生行权重、按格界），
    `spatial_resampling = area_weighted_coarsen`；36 个数组的 `static_fields.npz`。
  - **相位 5**：抽样格独立再派生，最大相对偏差 **5.15e-08**（float32 存储精度内）。
  - **相位 6**：`contracts.fresh_outputs` 独占发布 + `BUILD_COMPLETE.json`；
    `static_identity = 8671ed7c…`。
  - **证据页**：`docs/R7_URBAN_STATIC_COVARIATES.md`。
- **未做**：
  - **`campaign-state.current_round_goal` 未改**：R1 的 workflow 仍在跑，而
    `tools/r7_round.py` 的每个子命令都从 `campaign-state.current_round_goal` 解析当前轮；
    此刻改指针会让 R1 的 seed 42/43 `stage` 与登记 `round-guard`/`ledger-plan` 解析到本长文
    （无 round-spec 块）而失败。**推迟到 R1 的 workflow 退出之后**，命令见证据页 §9。
  - 模型侧接线（把 `static_identity` 绑进 `protocol_sha256`）是**下一轮**，需自己的 ADR 与
    预声明出口；本轮**不改** `model/**`、**不重建** store、**不加** state 通道。
  - **非 dryRun 的编排未跑**：`.zcode/workflow-drafts/r7-urban-static-round.dwf.ts` 已用
    `CreateWorkflow` 编译并跑过一次**有界 dryRun**（run
    `dwfrun-1f67679b-dd53-43a2-9fa8-14b468dbfb1c`，2026-10-27T09:55:11Z 起，completed，926431 tokens；
    止于「实现与独立复核」，未提交/未下载/未发布）。其登记步会改 `campaign-state`
    （同上会打断 R1），因此本轮由执行者直接跑驱动的六个子命令；**`dryRun: false` 从未运行**。
- **编排的非 dryRun 部件已用 workflow 测试台核对（2026-10-27）**：加载 `dynamic-workflows` 技能后，
  用 `EvalWorkflowSnippet`（同一编译器/沙箱/`world.run` 路径，**不落盘、不产生 run**）对**真实仓库状态**
  核了三件事，全部通过：①草稿的六个 stage 字面量（`runStage("…")`）与驱动的 `COMMANDS` 键
  **逐个相等**（`stageMismatch: []`）；②草稿 `IMPLEMENT_PATHS`/`REGISTER_PATHS` 声明的**每个路径都存在**
  （`missingPaths: []`——`git add` 一个不存在的路径会让整次提交失败，这是编排的非 dryRun 提交步最容易踩的坑）；
  ③本长文带 `<!-- round-node: S3 -->`（`goalBriefHasRoundNode: true`）。
  这**不等于**「非 dryRun 路径已跑」——提交/stage/登记三步仍未执行，且必须等 R1 退出。
- **当前草稿的字节已用 `CreateWorkflow` 编译通过（2026-10-27）**：`dwfrun-1f67679b` 那次 dryRun
  跑在**三处草稿修复之前**——实测该 run 结束于 `2026-10-27T09:58:45Z`，而草稿文件最后修改于
  `2026-10-27T13:42:10Z`（`ls --time-style=full-iso`），所以它**不能**证明当前字节可编译。
  加载 `dynamic-workflows` 技能后对**当前字节**重编译一次：**编译干净**，run
  `dwfrun-87bbd520-01a9-4ada-894f-122db5230455`（参数用声明默认值 `dryRun=true` / `maxRounds=1`，
  completed）。该 run 的侦察员在它自己的 ask 内独立复核了本节的停止判定（`blocked=true`，理由与
  §7 一致），并实测确认：发布目录 `outputs/r7_urban_static_covariates_attempt01/`、
  `BUILD_COMPLETE.json` 存在、实现提交 `7065f19`、登记提交 `58e3a18`、证据页 14,824 B，
  以及 R1 仍在跑（PID 777767 `tools/r7_round.py stage seed 42` 与其子进程 777788）。
  该 run **未提交、未下载、未聚合、未发布**，跑完工作树仍干净（`git status --porcelain` 空，
  HEAD 仍是 `1d2eb09`）。**消掉的是「用前仍须编译一次」这条风险；`dryRun: false` 仍未运行。**
- **已修的隐患（2026-10-27 实测发现）**：本长文原先**没有** `<!-- round-node: S3 -->` 标记。
  `check_campaign_state.py` 的 C-05 要求**当前轮** brief 带该标记且节点与 master 的
  `current_node` 一致，而该脚本在 CI 中运行（`.github/workflows/ci.yml`）——因此**只改指针**
  会让 CI 变红（实测：把指针指向本长文后报
  `FAIL C-05 docs/goals/s3-urban-static-covariates.md carries no <!-- round-node: X --> marker`）。
  已在标题下补 `<!-- round-node: S3 -->`；补后用一份**临时** master 副本重测，
  `campaign_state node=S3 failures=0`（原为 1），live 指针与 R1 的 round-spec 均未受影响。
- **预算更正已入档（2026-10-27）**：决策 `docs/decisions/0064-urban-static-covariate-budget-correction.md`
  ——按首次实测（整轮 `wall_clock_seconds = 9550`、网络 13.49 GiB、磁盘消耗 27 GiB，三者**均未触门**）
  核对名义刻度 ⇒ `planned 21,600` / `hard 86,400` 与网络/磁盘预算**维持不变**；只涉时间、不动科学门。
- **如实记的缺口**：phase ② 要求的「磁盘峰值与余量门」里，冻结的 `protocol.json` **只有余量门**
  `disk_headroom_bytes`，**峰值只有实测值**（`native/` 20 GiB + `published/` 0.7 MB，`/data` 269→242 GiB）。
  已发布的 `protocol.json` **不追改**（会改 `protocol_sha256` 与 `static_identity`、断身份链），
  改由证据页 §2 如实标注，`estimated_peak_bytes` 记在下一轮协议里（见决策 0064）。
- **R1 退出后指针的初值（已核实）**：R1 的登记指令第 3 条要求把
  `campaign-state.current_round_goal` 写回 **R1 自己的长文**
  （`.zcode/workflow-drafts/r7-matched-exposure-round.dwf.ts:218`：`把 campaign-state.current_round_goal 改成 " + plan.goalPath`，
  R1 的 `plan.goalPath` = `docs/goals/s3-matched-exposure-reeval.md`）。所以 R1 退出后指针会停在
  **R1 的长文**上，下面第 1 步是一次**真实改动**，`before = docs/goals/s3-matched-exposure-reeval.md`。
- **已修的草稿缺陷（2026-10-27 实测发现）**：草稿的 `REGISTER_PATHS` 原本只有
  `docs/R7_EVIDENCE_INDEX.jsonl`、`docs/R7_CANDIDATE_BRIEF.md`、`GOAL_PATH` 三项，**漏了证据页与 master 长文**
  ——而 `REGISTER_ASK` 明确要求登记员写证据页、追加 master §7 账本行与 §8 进度块、并更新
  `<!-- campaign-state: ... -->`。登记步的 `git add ...REGISTER_PATHS` 因此会**静默漏掉这两份材料**，
  连 `campaign-state` 的指针改动都进不了提交。已补 master 长文进静态清单，并在 `git add` 的 args 里
  追加**运行期**的 `plan.evidencePath`（证据页路径由侦察员在运行期给出，故不能写进静态清单——
  写死一个可能不存在的路径会让 `git add` 整条失败）。参照 R1 的 `register_paths` 形状
  （证据页 + 索引 + brief + master + 本轮长文）。
- **已修的第二个草稿缺陷（2026-10-27）**：实现提交把「`git commit` 非 0」一律当致命错。但**本轮已经实现并提交过**
  ——重跑时 `git add IMPLEMENT_PATHS` 什么都不会暂存，`git commit` 于是以 **"nothing to commit"（exit 1）** 退出，
  草稿会在**任何 stage 之前**就 `break` 并报「实现提交失败」。已在 `git add` 之后加一次
  `git diff --cached --quiet`（exit 0 = 无暂存内容、exit 1 = 有）来分流：无内容时按「已在历史里」记成功，
  有内容才真提交。登记提交**不**做同样处理——那一步之前登记员必然写了证据页与 master，
  真有内容可提交；若登记员什么都没写，`commit` 非 0 正是应有的「登记没做成」信号。
  （`git` 已在草稿的命令集里，故 `git diff` 不新增命令面。）
- **更正（2026-10-27 实测，撤掉一条我先前的误记）**：先前记「重跑时 `publish` 会因
  `fresh_outputs` 独占而失败」——**实测不成立，已撤回**。`verify_endpoints()` 走 `new_attempt()`，
  编号是 `max(existing)+1`；现有只有 `attempt01`，故重跑会新建 **`attempt02`**（实测
  `next attempt would be: attempt02`）。`freeze`/`download`/`aggregate`/`recheck`/`publish` 各自用
  `latest_attempt(<marker>)` 解析，`attempt02` 一旦落盘对应 marker 就排在 `attempt01` 之后 ⇒
  **整轮都在 `attempt02` 里跑**，`attempt02/published/` 是全新的，`fresh_outputs` **不会**拒绝。
  **已发布的 `attempt01` 因此不被触碰**（其 `protocol.json`、`static_identity`、manifest 全部原样）。
  重跑的真实代价是**一次全量重下载（≈14.5 GiB）＋第二份产物与第二条登记**，不是破坏。
  （先前那条错在把「`published/` 独占」当成跨 attempt 的全局约束——它其实在 attempt 目录内。）
- **已修的第三个草稿缺陷（2026-10-27）**：目标要求「自跑 `tools/r7_round.py commit-guard`（提交前）
  ……不依赖 hook 兜底」，但草稿原本只跑 `check_conventions.py` + `pytest`，**没有那个字面调用**。
  两者的实质重叠（`commit_guard(paths)` 就是按路径跑阻断规则，草稿跑的是全仓 `check_conventions`），
  但要求写的是这条命令。已在实现提交前加一次
  `tools/r7_round.py commit-guard <IMPLEMENT_PATHS...>`，非 0 即停、不提交。
  **登记提交前不加**：那一步提交的是证据页与 master 等文档，`commit-guard` 的阻断规则面向代码；
  且登记后已有 `tools/r7_round.py verify` 作为收尾校验。
- **下一动作（可执行的第一条命令）**：R1 的 workflow 退出后，按序执行——
  1. 把 master 的 `<!-- campaign-state: ... -->` 的 `current_round_goal` 改成
     **`docs/goals/s3-urban-static-wiring.md`**，把本文件写进 `previous_round_goal`、
     把 `docs/R7_URBAN_STATIC_COVARIATES.md` 写进 `previous_round_evidence`
     （下一步是节点 **S3** 内的模型侧接线轮，ADR 为决策 0069）；
  2. `.venv/bin/python tools/r7_round.py state`（核对指向）；
  3. `.venv/bin/python tools/check_campaign_state.py --quiet`（应 `failures=0`）；
  4. 进入节点 **S3** 的接线轮：按 `docs/goals/s3-urban-static-wiring.md` 与
     `docs/decisions/0069-static-covariate-wiring.md` 执行。

  **本轮的编排（`dryRun: false` 跑 `.zcode/workflow-drafts/r7-urban-static-round.dwf.ts`）
  已不是下一步**：数据相位的六个子命令已由执行者直接跑完并发布（`attempt01`、
  `BUILD_COMPLETE.json` 在），重跑会新建 `attempt02` 并重下 14.5 GiB，**不增加证据**。
  该草稿保留为历史执行载体，其三个缺陷与编译核对记录见上。

### 2026-10-30 — 登记完成 + 编排路线如实记（已提交并推送）

**登记后校验器读数（实跑，不是估计）**：`.venv/bin/python tools/r7_round.py verify`
→ `passed: true`，三条子校验退出码全 0：`check_conventions.py --quiet`
（`blocking rules=37 failing=0 report-only hits=0`）、`check_campaign_state.py`
（`campaign_state node=S3 failures=0 notes=8`）、`verify_r7_evidence_index.py
--check-brief`（`PASS evidence index and brief: 94 records`）。

**指针已按序推进两次（实测，`git log -S` 核对，不是推测）**：
- R1 的 workflow `dwfrun-442a17eb-ba11-4af9-9b56-6460f32a1e0b` 于
  `2026-10-28T04:09:42Z` settled `completed`，PID 777767/777788 已不存在；其末个
  seed 失败（`seed43` exit 1）是 R1 自己的科学读数（登记为 mixed/not-candidate），
  与本轮无关。
- 提交 `5e4adf9` 把 `current_round_goal` 指向本长文；`896eabf` 起再推进到
  `docs/goals/s3-urban-static-wiring.md`（接线轮，决策 0069）。
- 当前 `d80dfd3` 指针：`current_round_goal = docs/goals/s3-urban-static-wiring.md`、
  `previous_round_goal = docs/goals/s3-urban-static-covariates.md`、
  `previous_round_evidence = docs/R7_URBAN_STATIC_COVARIATES.md`。
  **不得改回本长文**——那会回退一个已登记轮次。

**编排路线：`dryRun: false` 未执行（如实记，不补跑）**：
- 已做：加载 `dynamic-workflows` 后，`CreateWorkflow` 对草稿**当前字节**编译两次通过
  （`dwfrun-1f67679b…`、`dwfrun-87bbd520…`），两次参数都是默认 `dryRun=true`。
- 未做：**`dryRun: false` 的运行从未发生**。本文件要求的六个相位
  （`verify-endpoints`/`freeze`/`download`/`aggregate`/`recheck`/`publish`）是
  **直接调用** `scripts/r7_urban_static_round.py` 逐个子命令跑完的；产物、身份、
  digest、独立重算、`BUILD_COMPLETE` 与四件套登记全部为真（见证据页 §1–§6），
  但**不是**经由 workflow 的 phase 编排路径。
- 不按字面补跑的理由：指针已推进到接线轮，非 dryRun 草稿会重跑整条采集轮
  （新建 `attempt02`、重下 14.5 GiB），那是对**已登记轮次**的重复推进，与 campaign
  的「不重复已登记轮次」冲突；且草稿的 `commit-guard`/`verify` 依赖的当前指针不是本长文。
  因此该缺口**保留为未做**，不以回退指针来补。
- 对产物的影响：无。受影响的是「本轮是否由一个可恢复的编排器驱动」这一工程属性。
- 若日后要补：需自己的 ADR 与预声明出口，在**指针重新指向本长文**的独立 attempt 里跑
  `dryRun: false`，并接受它重跑六相位（产物落在新的排他 `outputs/` 路径）。

**证据页更新**：`docs/R7_URBAN_STATIC_COVARIATES.md` §9 改写为指针的实际完成经过，
新增 §10 记上述替代路线；页面新 digest
`2b111997fdae3d4dd0db9c28bc2aa1b32a93294c319112406248c365849b62e3`
（17,763 B），账本行 `s3-urban-static-covariates-acquisition` 的 `evidence_sha256` /
`evidence_commit` / `evidence_revisions` 已同步，候选简报由索引自带渲染器重生成。

**提交**：`7732dad`（证据页与账本更正）、`d80dfd3`（evidence_commit 指向其修订），
已推送 `30dc58a..d80dfd3` 到 `r7/weather-reasoning`。

### 2026-10-30 — 决策 0070：`dryRun: false` 编排重跑已启动（attempt02）

**为什么有这一轮**：上一节记的缺口是「`dryRun: false` 从未运行」。本轮补它，
数据交付物本身不因此改变（`attempt01` 字节不动）。

**启动前做了什么（每一步都有实测证据）**：
1. 立 ADR `docs/decisions/0070-urban-static-replay-under-workflow.md`（`accepted`）并加进
   `docs/decisions/README.md` 索引；预声明出口 **E1–E6**（六相位全绿 / 层可达数 ≥13 且
   LandScan 仍 `failed-no-fallback` / `static_identity` 差异**如实记录不得声称逐位相同** /
   recheck 相对偏差 ≤1e-05 / 网络 ≤40 GiB 且磁盘余量 ≥60 GiB 且墙钟 ≤86,400 s /
   既有 94 条 `record_id` 不重复）。
2. **发现并修掉一个会让指针回归直接失败的机械约束**：`tools/check_campaign_state.py:216-266`
   的 C-04 要求 `previous_round_goal` 的「下一动作」末尾含 `current_node`。接线轮长文
   `docs/goals/s3-urban-static-wiring.md` 的「下一动作」尾部（304 字符，实测）**不含 `S3`**，
   所以只改指针必失败。已给该行补 `（节点 S3；`——**只补节点名**，不改其任何读数/出口/结论。
3. 指针临时回归：`current_round_goal = 本长文`、`previous_round_goal = 接线轮长文`、
   `previous_round_evidence = docs/R7_STATIC_COVARIATES_WIRING.md`；
   实测 `check_campaign_state.py --quiet` → `failures=0`。
4. 提交 `6b1bb0c`（ADR + 索引 + 指针 + 接线轮补节点名），`commit-guard` `passed: true`、
   `check_conventions.py` 37 条阻断规则 `failing=0`。
5. **改掉草稿三处会在重跑中破坏已登记状态的缺陷**（都在 `.zcode/workflow-drafts/r7-urban-static-round.dwf.ts`）：
   - 侦察员提示词原本默认「首次获取」，会把本轮判成 `blocked` 直接跳过全部 stage；
     改为写明这是 0070 的重跑、`blocked` 不是正确结论。
   - 实现者提示词原本要求**重写** `scripts/r7_urban_static_round.py` 与测试——那会让重跑
     不再是同一实现的复现。改为**只核对不改写**，并要求证据（`file:line`）。
   - 登记员提示词原本会改写 `docs/R7_URBAN_STATIC_COVARIATES.md`（其 sha256 被账本行 pin 住）
     并可能写出重复 `record_id`。改为写**新证据页** `docs/R7_URBAN_STATIC_COVARIATES_REPLAY.md`
     与**新 `record_id`**，并明确**本轮结束必须把指针恢复**到接线轮。
   - `IMPLEMENT_PATHS` 同步收窄为「本轮真正新增的 ADR 与索引」，不再 `git add` 已登记的
     驱动/测试/计划。

**启动（实测）**：`CreateWorkflow` 编译干净，run
`dwfrun-34a44d57-a5b5-4176-a95d-b870f1f7bcfe`，参数 `dryRun: false` / `maxRounds: 1` /
`resumeFromRound: 0`，`2026-10-30T02:55:11Z` 起跑，`owned_by_this_session: true`。
阶段图 5 个（侦察 → 实现与独立复核 → 提交并断言分支 → 数据轮 stage → 读数与登记）。

**这一轮要补的证据**：非 dryRun 分支的首次执行——分支断言 → `commit-guard` → 实现提交 →
六 stage（`verify-endpoints`/`freeze`/`download`/`aggregate`/`recheck`/`publish`）→
`tools/r7_round.py verify` → 登记提交 → 推送 `r7/weather-reasoning`。

**已知风险（如实记）**：
- 将重下约 14.5 GiB、多占约 20 GiB；启动前实测 `/data` 余量约 570 GiB。
- `static_identity` 可能因 `published_utc` 之类的时间戳而**不同**于 attempt01 的 `8671ed7c…`；
  E3 要求如实记差异来源，**不得**声称逐位相同。
- 指针处于临时回归态，期间其他研究线若开轮会解析到本长文；故本轮有界、跑完即恢复指针。
- 该草稿的非 dryRun 路径此前**从未执行**，缺陷可能只有在实跑中暴露；若某阶段非 0，
  按停止条件如实记，不重试到偶然过。
