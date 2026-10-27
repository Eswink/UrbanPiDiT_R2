# 目标：S3 城市静态协变量获取与发布（L2 城市证据落地）

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
  - **workflow 未运行**：`.zcode/workflow-drafts/r7-urban-static-round.dwf.ts` 的登记步会改
    `campaign-state`（同上会打断 R1），因此本轮由执行者直接跑驱动的六个子命令，
    **未** `CreateWorkflow`；草稿的编译仍未做。
- **下一动作（可执行的第一条命令）**：R1 的 workflow 退出后，
  把 master 的 `<!-- campaign-state: ... -->` 的 `current_round_goal` 改成
  `docs/goals/s3-urban-static-covariates.md`；然后 `.venv/bin/python tools/r7_round.py state`。
