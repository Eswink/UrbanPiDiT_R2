# 0024 — S3 全年覆盖单因素轮（4 个月 → 12 个月，2017–2021 / 2022）

**状态：工作态计划（2026-10-21 用户授权写入），尚未执行。** 本文件是**计划**，不是 round brief，
不产生任何科学结论；正式 brief、冻结协议与预声明出口必须在开工时按 R-006 重新写定。
**授权来源**：用户本轮明确指示「用我们最新下载的数据」，并选择「数据方向（全年 store + 全年 vs
4 个月验证）」；不再新增下载。数据扩充的 D3 前置（架构/训练信号/优化全无收益）当前已满足。

---

## 0 先决约束：这一轮只能在 65×65 实例上做

这是本计划最重要的一条事实，必须先写清楚：

- 已登记的容量/剂量/H-D 臂是 **129×129** 宽实例；FY 下载（`outputs/r7_s3_fy_2000_2022/source.nc`）
  是 **65×65**。仓库**没有** 65→129 上采样管线，store 契约明确 `spatial_resampling: false`
  （`data/preprocess/r7_era5_zarr.py:207-208,223`），插值出来的外圈不是真实观测。
- 因此本轮是 **65×65 实例上的数据覆盖单因素轮**：它回答「同一年份范围内，把 4 个月扩到 12 个月
  是否提高 skill」，**不能**直接替代或外推到 129×129 宽实例。
- 若要在宽实例上做全年，必须重新采集 129×129 全年源（FY 页估算约 34–36 h、~1 TB 网络、
  ~27 GiB store），**不在本授权范围内**；本轮不做、也不以 65×65 结果冒充宽实例结论。
- 65×65 上的评分区域用**整张 65×65 网格**（等价于宽实例的冻结 `interior_32` 目标箱）；
  `interior_32`（margin 32）在 65×65 上只剩 1×1，不能在协议里沿用这个名字而不说明。

## 1 授权与边界

- **允许**：用已下载的 FY 源派生一个**全新排他** `outputs/` 路径的全年 store；只读 preflight 后
  显式 `--write`；派生新驱动/测试；两臂 × 三 seed 训练；四件套登记。
- **不允许**：新增下载（包括 129×129 全年、2023 全年）；写 `data/raw|interim|processed`；
  重开容量轴（0048）、H-A…H-D、PE/scale/typed/ordering/anchor/rollout-dose/EMA/dense/
  final-only/spatial-gradient/zero-base decoder/gradient-surgery；与已关轴组合；改科学门。
- **与 tendency 轮的关系**：另一窗口的 tendency-input 轮是另一个方向，本计划**不合并**它的
  未提交改动；数据轮必须在干净、已提交的代码 revision 上开工（见 §6 Phase 0）。

## 2 因子与对照

**唯一因子：训练覆盖（4 个月 → 12 个月），年份固定 2017–2021。**

- 控制臂 N：同一全年 store，但训练清单只取注册的 4 个月选择（Jan/Apr/Jul/Oct、每月 1–30 日、
  UTC 00/06/12/18），即注册 4 个月实例的同一时间覆盖。
- 候选臂 F：同一全年 store 的**全部 12 个月**、2017–2021 全部日期。
- 两臂同代码 revision、同迁移父（attempt04，`model_code_sha256 3fe747b2…`）、同 seed 41/42/43、
  同 2400 updates、K=4、depth=4、dim=192、注册物理权重、同一 48 桶 train-only 气候态分母、
  同一 val 2022 全年清单；**只差训练清单覆盖的月份**。
- 不在本轮加入 2000–2016：那会同时改变「年数」，破坏单因素。更多年份是**后续独立因子**。

## 3 数据构建

### 3.1 合并 2023 封存块（只补 test split）

FY 源覆盖 2000-01-01–2022-12-31，**没有 2023**；store builder 要求 test split 非空
（`data/preprocess/r7_era5_zarr.py:166-167`）。已有 2023 四个月块：
`outputs/r7_s2_batch2_20222023/source.nc`（65×65、2022-01-01–2023-10-30、Jan/Apr/Jul/Oct、
960 帧）。做法：

1. 只从该文件取 **2023** 的 472 帧（丢掉重复的 2022 四个月）；
2. 与 FY 源按时间拼接，断言 latitude/longitude/level/data_vars/units/dtype 逐位一致
   （优先用 `data/download/multi_year_source.combine_sources` 的兼容性断言）；
3. 写全新 `outputs/r7_s3_fy_annual_20261021_attempt01/source.nc`，记录新 SHA256；
4. test split 只用于 sealed 清单，**永不读取**。

### 3.2 Store

- 配置：`split_years {train: 2017–2021, val: 2022, test: 2023}`，`history_steps=2`、
  `history_interval_hours=6`、`lead_time_hours=6`、`sample_stride_hours=6`、`time_chunk=16`、
  `compute_process_targets=true`；17 通道顺序与注册实例一致。
- 帧数约 9,236（2017–2021 全年 7,304 + 2022 全年 1,460 + 2023 四个月 472）；
  raw state ≈ 2.47 GiB，压缩 store ≈ 1.9 GiB；CPU 构建预计数分钟。
- 先 `prepare_r7_local.py --report`（只读，写 JSON 报告），人工核对 shape/windows/units/grid；
  再 `--write` 到**全新排他**路径，`--max-raw-gib` 高于预检值；`BUILD_COMPLETE` 必须出现。
- 归一化在**全年 2017–2021 train** 上拟合；两臂共用同一 store 归一化。这一点与注册 4 个月臂
  不同，必须写进 limitations；两臂之间仍是同一把尺。

### 3.3 两套清单（同一 store）

- `control_manifests/train.jsonl`：从全年 store 的时间轴按注册 4 个月选择重算窗口
  （Jan/Apr/Jul/Oct、1–30 日、00/06/12/18），预计约 2,360 条；
- `treatment_manifests/train.jsonl`：2017–2021 全部 12 个月窗口，预计约 7,304 条；
- 两套 `val.jsonl` **逐字节相同**：2022 全年 1,460 个窗口；
- 两套 `test.jsonl`：2023 封存，同一内容，不读；
- 每套清单独立目录，避免 `training/r7_evaluate.py:121-123` 的 data identity 检查把两臂混用。

### 3.4 48 桶 train-only 气候态

- 用 `data/r7_evaluation.py::fit_training_climatology` 在**全年 train（2017–2021）**上拟合
  48 个 `(month, hour)` 桶（每桶约 141–155 帧，`n_selected_steps=7,304`）；
- 两臂共用这一个分母；**不得**复用 4 个月的 16 桶 `591ede…` / `da2f28…`；
- 按 D2 方式冻结为一个新的排他 `outputs/` 产物，记录 digest 与 `training_years`；
- 缺桶必须报错，不允许 held-out fallback（`data/r7_evaluation.py:138-141`）。

## 4 预声明出口（冻结前写死）

科学门沿用既有形式，不新增阈值：

- (a) 候选 F 的 72h t2m skill（整张 65×65 网格）在 seed 41/42/43 **全部 >0**；
- (b) 同 seed 的 F−N 72h ΔRMSE **三 seed 全部 <0**（负值 = F 更优）；
- (c) 候选 F 的 48h skill **三 seed 全部 >0**；
- (d) F−N 在 6/12/24h 的 **seed 均值** ΔRMSE **≤0**；
- (e) 桥：控制 N 的 5-lead RMSE 对注册的 **65×65 narrow v3 4 个月臂**的差落在其已登记噪声带内
  （具体带在 brief §3 中引用窄臂读数；若桥失败只记 identity/engineering unresolved，
  不是科学判定）。

全真 → 只提请独立复核/验收，不自动进 S4；任一 (a)–(d) 假 → **永久关闭「全年覆盖」支**：
不扫月份组合、不调 2400 updates、不加 seed、不把 2000–2016 塞进同一支、不与已关轴组合。
`test` 未读、r=0、S4 不启动；不宣布 SOTA/complete。

## 5 预算（先冻后跑）

| 项 | 预计 |
| --- | --- |
| 合并源 + store 构建（CPU，0 GPU） | 数分钟；raw ≈2.47 GiB，store ≈1.9 GiB |
| 48 桶气候态（CPU，0 GPU） | 数分钟 |
| 两臂 × 三 seed 训练 | 6 次；按 65×65 窄实例 receipts 重新标定 |
| 全年 val 评分 | 1,460 窗口/lead（约 4 个月实例的 3.09×） |
| 暂定 per-seed / planned / hard | 7,200 s / 21,600 s / 43,200 s（**开工前按窄臂实测重标**） |

两 seed 并发（决策 0047），每卡同时至多一个 worker；spawn 前只读余量门；不对邻居发信号。
失败 attempt 原样保留、全额计费、不 in-place resume。

## 6 实施阶段

**Phase 0（先决）**：解决工作树占用。另一窗口的 tendency-input 轮仍有未提交改动
（`model/*.py`、`scripts/r7_s3_tendency_input_*`、`docs/decisions/0049-*` 等）。必须先让该轮
停止/提交，数据轮从**干净、已提交**的 revision（当前 `5c821b2`）开工；**不合并、不覆盖**它的改动。

**Phase 1**：合并 FY + 2023 块，核坐标/变量/单位/时间连续性，写新 source + SHA256。

**Phase 2**：`prepare_r7_local.py --report` 只读预检 → 人工核对 → `--write` 新 store →
`BUILD_COMPLETE` → `validate_store` / `dataset_identity`。

**Phase 3**：生成两套清单 + 48 桶气候态；核对控制清单与注册 4 个月覆盖一致
（月份/日/时次/年数），treatment 清单为全年。

**Phase 4**：写正式 round brief（含 round-spec、预声明出口、65×65 限制、预算）与新决策记录
（编号按当时已提交的最大决策号 +1；当前工作树的 0049 属于 tendency 轮，若该轮不落地再用 0049）；
派生驱动/父模块/CI-safe 测试；独立复核；`round-guard`。

**Phase 5**：提交实现 → 推工作分支 → `R7 CPU CI` 绿。

**Phase 6**：`freeze` → `stage prepare/archive/stage-seeds/reading`；两臂 × 三 seed。

**Phase 7**：读数 → 证据页 + index record + 账本行 + §8 + campaign-state → `verify` → 提交/推送/CI。

## 7 风险与限制（必须如实登记）

1. **65×65 ≠ 129×129**：本轮结果只属于 65×65 实例；宽实例全年需要新采集，未授权。
2. **年份固定 2017–2021**：2000–2016 未使用；「更多年份」是后续独立因子，不能与本轮合并解释。
3. **2023 只有四个月**：test 仍封存；它不构成「完整未见年份」的全年确认。
4. **归一化是全年拟合**：控制臂与注册 4 个月臂的归一化不同，桥 (e) 只能近似；两臂之间无此问题。
5. **控制清单是 1–30 日的 4 个月子集**：与注册 4 个月实例的日覆盖一致，但索引来自全年 store。
6. **合并 2023 块**：必须断言坐标/变量/单位一致；FY 页已记录 `q850/q500` 的小 std 风险。
7. **并发墙钟**：两臂/三 seed 并发会改变墙钟，GPU-h 不变；墙钟不是科学判据。

## 8 明确不做

不新增下载；不建 129×129 全年源；不碰 `data/raw|interim|processed`；不改模型代码（因此无需
父重迁移）；不复用 16 桶气候态/锚表；不重开已关轴；不读 test；不 main/force/merge；
不删 `outputs/`；不合并另一窗口的 tendency 改动。

## 实际结果

**工作态计划，尚未执行（2026-10-21）。** 没有 store、没有训练、没有读数；本文件不产生科学结论。
实际结果（两臂 5-lead skill、F−N 配对 Δ、GPU-h、失败/关闭判定）在本轮跑完并登记后补写。
