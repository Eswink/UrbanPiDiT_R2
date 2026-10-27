# 0063 — 城市静态协变量的获取授权、范围与集成形态

- **日期**：2026-10-27
- **状态**：accepted
- **范围**：主模型超气候态方向的**城市静态协变量**获取与发布（下载器 / 聚合 / 契约 / 发布）。
  **不改** S3 节点任何科学门、**不改** `model/**`、**不重建** store、**不新增** state 通道。
- **依据**：用户指令（"帮我去拉取相关的数据（尽量和我们现在的数据相匹配）"，本轮唯一授权来源）；
  决策 0038（**只**授权免费合法的区域**天气**数据）；
  `docs/R7_URBAN_EXTENSION_BLOCKED.md`（BLOCKED 的是 **km 级动态真值**/issue #9，不覆盖静态协变量）；
  `docs/DATA_SPEC.md:21-23`（L2 计划清单）；
  `data/download/read_plan_frozen.py:34-36`（R7 冻结 ROI）；
  `.agents/skills/real-data-acquisition/SKILL.md:99`（原文排除城市形态数据获取）；
  外部检索（委派 `web-researcher`，访问日期 **2026-10-27**）。

## Context

**① 授权缺口是真实的，不是形式问题。** 决策 0038 `:27-30` 具名授权的对象是"免费合法的**区域天气
数据**"；`real-data-acquisition` 的 `SKILL.md:99` 明写"**不覆盖**城市形态（WorldCover/建筑/DEM）
数据的获取"；主计划 §0 的 D3 还写着"**本轮起暂缓数据扩充**（收益判断偏小）：不新增下载、不重建
store"。⇒ 城市静态协变量的获取**不在任何既有授权内**，必须由本 ADR 具名授权并说明它与 D3 的关系。

**② 用户指令就是授权来源。** 用户要求"帮我去拉取相关的数据（尽量和我们现在的数据相匹配）"，
并明确"动态天气主 target 约 1 km 这种可以后续再来做"。本 ADR 把这句具名化为可执行范围，
不扩大它。

**③ 这不是 issue #9。** `docs/R7_URBAN_EXTENSION_BLOCKED.md` 的 **BLOCKED** 结论针对的是
"km 级**动态**分析真值"（HRCLDAS / SMBFD / HRRR / WeatherBench2 逐一核过，无匿名可达者），
它的三条约束之一是"**动态**（时间演变的分析场）"。城市**静态协变量**不满足那条，因此**不在**
那份审计的结论范围内。该页的纪律（禁止把升采样 ERA5 当真值、禁止伪造、失败留痕）继续适用。

**④ ROI 必须先纠正。** `docs/DATA_SPEC.md:3-7` 的 ROI 是**旧的京津冀盒**（113–120°E / 38–42°N）
加"北京 1 km 广域缓存"，那是 V6 时代的口径，**与当前 R7 实例不符**。当前实例是
`data/download/read_plan_frozen.py:34-36` 冻结的 **27.0–43.0°N / 107.0–123.0°E，65×65，0.25°**。
"尽量和现在的数据相匹配"⇒ 锚到 **R7 冻结 ROI**，不是 DATA_SPEC 那个旧盒。

**⑤ 集成形态有硬程序约束（实测）。**
- `dataset_identity`（`training/r7_experiment.py:89-103`）绑 **manifest 字节 + store 的 `attrs` 全量
  + `state.shape` + 固定数组**（`latitude`/`longitude`/`time_ns`/`normalization_mean`/`normalization_std`）
  ⇒ **把静态量做成第 18 个 state 通道必然是"新实例"、必须重建 store**，会与正在跑的 R1–R5
  匹配算力队列脱钩。
- 做成 store 里的独立 `static` 数组：写进 `attrs` 就改 `data_identity`；**不写进 `attrs` 就
  `validate_store` 不管、`dataset_identity` 也不遍历 ⇒ 两份不同静态场会有同一个 `data_identity`**，
  这是审计洞，与本仓的身份纪律相悖。
- 任何 `model/**` 改动会改 `model_code_sha256`（`training/r7_experiment.py:31-40`），
  使 `load_checkpoint` 拒绝旧模型（`:128`）⇒ **接线轮必须重训**。

**⑥ 仓内没有重采样能力。** `data/preprocess/grid.py:1` 首行即声明"no sorting, interpolation or data
mutation"；全仓无 `reproject`/`coarsen`/`WarpedVRT`；所有下载器 receipt 都写 `interpolation: none`。
把 10–100 m 栅格聚合到 0.25° 的代码**必须从零写**。

**⑦ 外部检索的结论（摘要，非证据，R-050）。** 检索确认"**无需账号**可直接下载"的有：ESA
WorldCover（10 m，CC-BY-4.0）、GHSL 的 GHS-BUILT-S / GHS-BUILT-H / GHS-POP（100 m 级，JRC/AWS
Open Data）、LandScan（~1 km，2026 起 CC BY 4.0）。检索显示**需要注册/账号**的有：Copernicus DEM
GLO-30（Copernicus Data Space）、ALOS AW3D30（JAXA）、SRTM / MODIS MCD12Q1 / GPWv4（NASA
Earthdata）、ERA5-Land（CDS）、CLDAS（CMA）。检索另在 GHSL 分辨率上有 100 m 与 1 km 的**冲突**、
在 Copernicus DEM 的 AWS 镜像是否匿名上**未确认**——这两点列为执行轮第一道闸门的核对项。

## Decision

1. **授权范围（具名）**：免费、合法、**匿名可直接下载**、许可允许再分发的**城市静态协变量**，
   覆盖 **R7 冻结 ROI**（27.0–43.0°N / 107.0–123.0°E），聚合到 **0.25° / 65×65**。具名层清单：

   | 层 | 源 | 分辨率 | 许可 |
   | --- | --- | --- | --- |
   | 土地覆盖 | ESA WorldCover v100(2020)/v200(2021) | 10 m | CC-BY-4.0 |
   | 建成面 | GHSL GHS-BUILT-S（多历元 1975–2030） | ~100 m | 免费 |
   | 建筑高度 | GHSL GHS-BUILT-H（2018） | 100 m | 免费 |
   | 人口 | GHSL GHS-POP（多历元 1975–2030） | 100 m | 免费 |
   | 人口（备，交叉核对） | LandScan Global | ~1 km | CC BY 4.0（2026 起） |

   清单**可缩小不可扩大**：执行轮实测某源不可达就如实丢层、留 `failed-no-fallback`（R-008），
   **不得**用别的层顶替、不得合成。

2. **明确不含**：① km 级**动态**真值（issue #9 继续 BLOCKED）；② 需注册/账号的源
   （Copernicus DEM 官方门户、ALOS AW3D30、SRTM、MODIS MCD12Q1、GPWv4、ERA5-Land、CLDAS、
   CMFD）；③ 付费源；④ 全球全集；⑤ 具名清单之外的层（LCZ、CNBH、ECOSTRESS LST、
   GlobalBuildingAtlas、建筑 footprint 均不做）。

3. **集成形态：独立发布产物，不改 store、不加 state 通道。** 静态协变量发布为**自己的**
   `outputs/` 排他目录，带自己的 `static_identity`（sha256，覆盖源指纹 + 聚合规范 + 输出字节）、
   manifest 与 `BUILD_COMPLETE.json`，落在**同一 65×65 / 0.25° 网格**上。将来模型接线时把
   `static_identity` 绑进 **`protocol_sha256`**（不是 `data_identity`），从而"两份静态场不同 ⇒
   协议身份不同"可机械检查。**模型侧接线（时不变条件通路）与城市感知/物理机制的算法设计是
   下一轮的事，本 ADR 不授权接线，也不改 `model/**`。**

4. **与主计划 D3 的关系**：本 ADR 只**重排具名的城市静态子情形**——它不再等"架构/训练信号/优化
   全无收益"那个最末手段条件。**天气数据扩充仍然暂缓**，D3 对天气数据的措辞不变。

5. **技能与契约同步扩围**：`real-data-acquisition` 的 `SKILL.md:99` 排除条款在**本 ADR 的具名范围内**
   解除，并新增"静态栅格获取"一节（端点核对 → 许可 → 聚合语义 → 契约）；`docs/DATA_SPEC.md` 的
   L2 由"计划"升为"契约"（源/URL/版本/许可/分辨率/ROI/聚合规则/变量名与单位/输出布局）。

6. **时序**：用户指定"等 R1（`outputs/r7_s3_fy_year_coverage_matched_20261026_attempt01/`）跑得
   差不多再启动本轮"。本轮（准备轮）**只写文档**：0 GPU-h、不下载、不改 `campaign-state`。
   启动时由执行窗口把 `campaign-state.current_round_goal` 指向新的 goal 长文。

## Consequences

**变容易的：**

- 城市协变量**独立演进**：它的获取、许可、聚合语义与 R7 的天气实例解耦，R1–R5 队列的
  `data_identity` 与账本**完全不受影响**。
- GHSL 的**多历元**（2000/2005/2010/2015/2020）在 campaign 窗口 2000–2022 内给出**时间变化的
  城市强度**，而不只是单张静态图 —— 这使"城市感知"有机会做成**算法创新**而非"多喂一张静态图"。
- 契约与技能补齐后，下一轮的接线设计有可引用的、有许可与身份的静态场。

**变难 / 代价（如实列出）：**

- **磁盘与网络成本是真实的**：WorldCover 10 m 覆盖 16°×16° 需约 36 个 3°×3° 瓦片/年（两个年份），
  GHSL 多历元按 100 m 计 —— 量级在**数十 GB**。执行轮必须冻结逐层预期字节与磁盘余量门。
- **聚合是派生统计量**：面积加权占比/均值**不是**"插值当真值"（它是对真实观测求统计量），
  但**极易被误读**成"造了一个 0.25° 的城市真值"。契约与证据页必须逐字写清统计口径、CRS、
  格界与 `spatial_resampling` 语义，且**不得**把聚合结果当观测真值使用。
- **重采样代码从零写**（见 Context ⑥）：新增 reproject/面积加权聚合是真实工程量，且是新代码、
  新测试、新 R-016 网络面。
- **0.25° 可能过粗**：建成区在 0.25° 格内是"占比"，不含城市内结构。若后续要 1 km 城市结构，
  必须**保留原生分辨率瓦片**（本轮保留）。**本轮不保证有科学收益。**
- **地形协变量可能拿不到**：DEM 的匿名路径未确认（检索显示官方门户需注册）⇒ 2 m 温度的
  静态地形协变量可能缺失，这是**如实声明**的缺口，不是可以悄悄补上的。
- **接线轮必须重训**：`model/**` 改动 ⇒ `model_code_sha256` 变 ⇒ 旧 checkpoint 不可加载。
  本轮不动模型，但代价记在这里。
- **检索摘要不是证据**：可达性、GHSL 分辨率、Copernicus DEM 镜像三点**未实测**，必须由执行轮
  第一道闸门确认。

**备选方案与否决理由：**

- **备选 A：把静态量做成第 18 个 state 通道（新实例）。** 否决：`data_identity` 必变
  （`state.shape` + `channels` + 归一化数组长度），与正在跑的 R1–R5 队列脱钩；且时不变量放进
  逐窗口读的 `state` 是结构性浪费。
- **备选 B：静态数组写进 store 的 `attrs`。** 否决：`attributes=dict(root.attrs)` 全量进哈希
  ⇒ 改 `data_identity`，等于换实例。
- **备选 C：静态数组放 store 但不登记。** 否决：`validate_store` 不检查额外数组、
  `dataset_identity` 不遍历它们 ⇒ 两份不同静态场共享同一 `data_identity`，是**审计洞**。
- **备选 D：用需注册的源（Copernicus DEM / ALOS / SRTM / MODIS）把地形补上。** 否决：与"不建账号"
  冲突；地形本身也不是 urban 量。若执行轮实测 Copernicus DEM 的 **AWS Open Data 镜像匿名可达**，
  可作为可选层另行提出，不在本 ADR 授权内。
- **备选 E：本轮顺手把模型接线也做了。** 否决：用户明确"准备数据"；接线会改 `model_code_sha256`
  并需要重训，是独立的一轮，且其算法设计（城市感知模块 + 物理机制）需要自己的 ADR 与预声明出口。
- **备选 F：不立新 ADR，直接沿用决策 0038。** 否决：0038 的授权对象是**天气**数据，把城市形态
  数据算进去是**越权解释**；仓规要求新的具名授权有新记录。
