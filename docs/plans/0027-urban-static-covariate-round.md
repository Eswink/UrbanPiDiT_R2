# 0027 — 城市静态协变量数据准备轮（L2 城市证据落地）

> 归档自本窗口的计划模式工作态（`plan-sess_c89a8013-…md`）。计划是**不可变记录**：
> 需求变化时新建一份，不改这一份。执行轮完成后再回填文末的「实际结果」节。

## 1. 本轮要解决的问题

用户要求"帮我去拉取相关的数据（尽量和我们现在的数据相匹配）"，并明确**动态 1 km 真值延后**、
**LLM 式推理延后**。仓里的现状是：

- `docs/DATA_SPEC.md:21-23` 的 **L2 城市证据**只有一行层名罗列（WorldCover 10m、Copernicus DEM 30m、
  LCZ 100m、CNBH 10m、GlobalBuildingAtlas、人口、ECOSTRESS LST），**没有许可证、没有来源 URL、
  没有数据契约、没有授权状态**；且它的 ROI（`docs/DATA_SPEC.md:3-7`，京津冀 113–120°E/38–42°N）
  是 **V6 时代的旧盒**，与当前 R7 实例不符。
- `docs/R7_URBAN_EXTENSION_BLOCKED.md` 把城市扩展判为 **BLOCKED**，但那份审计的对象是
  **km 级动态真值**（issue #9），不是静态协变量。
- 决策 0038 只授权**天气**数据；`real-data-acquisition` 的 `SKILL.md:99` 明写不覆盖城市形态数据获取。
- 仓内**没有任何**栅格重采样/聚合能力（`data/preprocess/grid.py:1` 声明 no interpolation）。
- 唯一的静态栅格下载器 `data/download/worldcover_cog.py` 是**从未跑通的骨架**：
  `audit/real_data_download_status.json` 记它 `blocked_or_failed`，多 tile ROI 直接 `raise`，
  且它用 `rasterio.open(url)` **直连**、绕过了 `http_public` 的 SSRF/pinned 守卫。

**本轮的产出**：一份授权（决策 0063）、一份计划（本文件）、一份 goal 长文（执行轮的 brief，
workflow 内联）、L2 的具体契约、技能扩围，以及一个可直接跑的 workflow 草稿与终版提示词。

## 2. 已核实的事实（决定计划形状的那些）

| 事实 | 证据 |
| --- | --- |
| R7 冻结 ROI = 27.0–43.0°N / 107.0–123.0°E，65×65，0.25° | `data/download/read_plan_frozen.py:34-36` |
| 格心 43.0→27.0（降序）/ 107.0→123.0（升序）⇒ 格界 26.875–43.125N / 106.875–123.125E | 同上 + `native_grid_spacing_deg=0.25` |
| 加 state 通道 ⇒ 新实例（`attrs` 全量 + `state.shape` + 归一化数组都进哈希） | `training/r7_experiment.py:89-103` |
| 改 `model/**` ⇒ `model_code_sha256` 变 ⇒ 旧 checkpoint 拒载 | `training/r7_experiment.py:31-40`、`:128` |
| 仓内无 reproject/coarsen/面积加权聚合 | `data/preprocess/grid.py:1`；全仓 grep 无命中 |
| 网络客户端只许在 `data/download/**`（R-016） | `tools/check_conventions.py` 的 `NETWORK_ALLOWED_PREFIXES` |
| `data/download/earthmover_spatial_d1.py` 已**恰好 600 行**（R-051 硬顶） | `wc -l` |
| 独占输出 + `BUILD_COMPLETE.json` 的通用装饰器 | `data/preprocess/contracts.py:11-51` |
| 检索确认"无需账号"：WorldCover / GHSL(BUILT-S、BUILT-H、POP) / LandScan | 委派 `web-researcher`（访问 2026-10-27）；**摘要非证据（R-050）** |
| 检索显示"需注册"：Copernicus DEM 官方门户 / ALOS / SRTM / MODIS / GPWv4 / ERA5-Land / CLDAS / CMFD | 同上 |

## 3. 授权（决策 0063）

具名授权：免费、合法、**匿名可直接下载**、许可允许再分发的**城市静态协变量**，覆盖 R7 冻结 ROI，
聚合到 0.25°/65×65。具名层清单（可缩小不可扩大）：

| 层 | 源 | 分辨率 | 许可 | 派生量 |
| --- | --- | --- | --- | --- |
| 土地覆盖 | ESA WorldCover v100(2020) / v200(2021) | 10 m | CC-BY-4.0 | 逐格类别**面积占比** |
| 建成面 | GHSL GHS-BUILT-S（多历元 1975–2030） | ~100 m | 免费 | 逐格建成面占比（多历元） |
| 建筑高度 | GHSL GHS-BUILT-H（2018） | 100 m | 免费 | 逐格平均建筑净高 |
| 人口 | GHSL GHS-POP（多历元 1975–2030） | 100 m | 免费 | 逐格人口密度（多历元） |
| 人口（备） | LandScan Global | ~1 km | CC BY 4.0 | 逐格人口（交叉核对） |

**明确不做**：km 级动态真值（#9 继续 BLOCKED）、任何需注册/账号的源、付费源、全球全集、
清单外的层。注册门源被排除的**直接后果**：地形（DEM）大概率拿不到，2 m 温度的静态地形协变量
缺失 —— 如实声明，不用别的层顶替。

## 4. 集成形态

**独立发布产物，不重建 store、不加 state 通道。** 静态协变量落在同一 65×65/0.25° 网格，
发布到自己的 `outputs/` 排他目录，带自己的 `static_identity`（源指纹 + 聚合规范 + 输出字节的
sha256）、manifest 与 `BUILD_COMPLETE.json`。将来接线时 `static_identity` 绑进 `protocol_sha256`
（**不是** `data_identity`），使"两份静态场不同 ⇒ 协议身份不同"可机械检查。

**模型侧接线不在本轮**：时不变条件通路（仿 `model/spacetime_conditioning_r7.py:217-243` 的
`position_features`）与城市感知/物理机制的算法设计留给下一轮，需要自己的 ADR 与预声明出口。

## 5. 执行轮的相位（写进 goal 长文 §4）

1. **只读端点核对（第一道硬闸门）**：逐源 HEAD/range GET，记 HTTP 码、`content-length`、
   许可页 URL、访问日期。不可达 ⇒ 留 `failed-no-fallback`、丢该层、**不合成**（R-008）。
   **优先核两点**：GHSL 的确切分辨率（检索在 100 m 与 1 km 间冲突）、Copernicus DEM 的
   AWS Open Data 镜像是否匿名（若匿名可作可选层另行提出）。
2. **冻结获取协议** `protocol.json`：ROI 精确格界、目标网格、逐层 source/version/license/
   预期字节/聚合规则/输出布局、磁盘峰值与余量门、网络字节上限、软/硬墙钟。
3. **下载**：新下载器放 `data/download/**`（R-016 唯一网络面），走 `http_public.open_public` /
   `http_pinned`（SSRF + 重定向校验 + timeout）。**不照抄 `worldcover_cog.py`**。
4. **聚合到 0.25°**：新写面积加权聚合（原生栅格 → metric CRS → 精确格界统计）。
5. **独立再派生核对**：抽样格从原始瓦片重算，精确比对（不复用聚合中间量）。
6. **发布与登记**：`contracts.fresh_outputs`（独占 + `BUILD_COMPLETE.json`）→ 证据页 → 四件套。

## 6. 风险

- **可达性未实测**：检索只给候选与许可线索；端点核对是第一闸门。
- **0.25° 可能过粗**：城市建成区在 0.25° 格内是"占比"，不含城市内结构。**保留原生分辨率瓦片**，
  供延后的 1 km 轮用。
- **多历元不是逐年观测**：GHSL 五年间隔 + 跨代传感器合成，若用作时间变化量必须逐字声明。
- **聚合是派生量**：契约与证据页必须写清统计口径、CRS、格界、`spatial_resampling` 语义；
  不得与"插值当真值"混为一谈。
- **接线轮必须重训**：`model/**` 改动 ⇒ `model_code_sha256` 变；本轮不动模型。

## 7. 预算

- **GPU-h：0**（纯网络 + CPU）。
- 网络 / 磁盘 / 墙钟三项冻结；名义 **planned 21,600 s（6 h）/ hard 86,400 s（24 h）**，
  执行时按首次实测重定（决策 0055 先例：只改时间、不动科学门、不在运行中改硬限）。
- **准入**：本轮的 CPU/网络与 R1 的 GPU 资源不重叠、产物排他 ⇒ 七 gate 可过、**允许并行**；
  但用户指定"等 R1 跑得差不多再启动"，该时序由用户定，写进 goal 长文。

## 8. 本窗口的交付物（文档轮）

| # | 文件 |
| --- | --- |
| D1 | `docs/decisions/0063-urban-static-covariate-acquisition.md` |
| D2 | 本文件 |
| D3 | `docs/goals/s3-urban-static-covariates.md`（goal 长文，workflow 内联） |
| D4 | `docs/DATA_SPEC.md` §L2（具体契约） |
| D5 | `.agents/skills/real-data-acquisition/SKILL.md`（具名范围内解除 `:99` + 静态栅格节） |
| D6 | 三处索引行（decisions / plans / goals 的 README） |
| D7 | `.zcode/workflow-drafts/r7-urban-static-round.dwf.ts`（草稿，git-ignored） |
| D8 | 终版提示词 = D3 的 §0 objective |

## 9. 明确不做

不下载任何数据、不跑 GPU、不重建 store、不改 `model/**`、不读 test、不动
`data/raw|interim|processed` 与归档快照、不改 `campaign-state`（R1 正在跑）、
不合并 main / 不 force / 不关 issue、不 `SaveWorkflow`、不新开 `.py` 文件。

## 实际结果

> 本窗口（准备轮）只写文档。执行轮的读数在它自己的证据页里登记，届时回填本节。

- **完成情况**：待执行轮结束后回填。
- **与计划的差异**：待回填。
- **验证**：待回填。
- **遗留**：待回填。
