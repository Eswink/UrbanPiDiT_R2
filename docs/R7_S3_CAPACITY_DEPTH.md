# S3 主干容量 depth 4→8 单因素臂：本轮**未运行**（操作者决定跳过，0 GPU-h）

**状态：未运行（2026-10-12 操作者决定）；`scientific_claim: false`；2023 test 未读；r 不消耗。**
本页**不是**该臂的读数：没有 attempt 目录、没有 seed、没有读数文件、**0 GPU-h**。
登记它的目的是**不留空白轮**，并如实记下"准备就绪但决定不跑"这一操作者判断——
而不是把它写成负结果、更不是写成通过。

## 1. 决定与依据

- **决定**：不运行 capacity 轮（主干 `depth` 4→8）的 GPU 部分，直接登记。
- **操作者给出的依据**：读 seed 41 的读数，**效果基本上没变化**。
- 该 seed 41 属于**上一轮 K4→8 臂**（证据页 `docs/R7_S3_REASONING_DEPTH.md`，索引
  `record:s3-reasoning-depth`），是本轮之前**唯一完整跑完**的 seed。它相对**已登记 2400 剂量臂**
  （K=4，pinned 复用、未重训练）的 `interior_32` t2m 读数为：

| lead | K8 rmse | K4 rmse（已登记） | Δrmse | Δskill |
| ---: | ---: | ---: | ---: | ---: |
| 6 h | 2.289701 | 2.281941 | **+0.00776** | −0.00287 |
| 12 h | 2.877077 | 2.868233 | **+0.00884** | −0.00417 |
| 24 h | 2.618599 | 2.627157 | −0.00856 | +0.00380 |
| 48 h | 3.245894 | 3.248979 | −0.00309 | +0.00178 |
| 72 h | 3.588657 | 3.596453 | −0.00780 | +0.00509 |

（|Δskill| ≤ 0.00509；**72 h skill 仍为负 −0.16954**。数字从产物 `boundary_rmse.csv` 逐位复核。）

## 2. 这条依据能支持什么、不能支持什么

- **已确认**：上表的 Δrmse/Δskill；该臂只完成 1 个 seed；本轮**没有任何产物**。
- **不能支持**：把它当作**本轮主干深度**的读数。`K`（每次预报每步的内部推理次数）与主干 `depth`
  （`WindowAttentionBlock` 层数）是**两个不同的因子**；本轮**从未运行**，**本轮的假设未被检验**。
- 因此本页登记的是**操作者按成本/收益决定跳过**这一决定，**不产生**任何关于 depth 4→8 的科学结论，
  也**不**据此关闭"加大容量解决 72 h"这一支（该支的关闭条件在本轮 brief §3 里是"72 h 三 seed 全正
  → 提请独立验收；否则关闭"，而本轮从未执行，条件既未触发也未满足）。

## 3. 已准备但未运行的东西（保留为可复用资产）

| 项 | 值 |
| --- | --- |
| 驱动 | `scripts/study_r7_s3_capacity_depth.py`（commit `710612d`） |
| CI-safe 测试 | `tests/test_r7_s3_capacity_depth.py`（28 例；`tools/r7_round.py round-guard` 通过） |
| 归档 code.zip | `outputs/r7_s3_capacity_depth_code.zip`，SHA256 `de9f6bc4e14d9eb774b99a4ef62d77bbe6b6335f814eeaf286788a05f112390e`，1380 文件，comment `710612d` |
| attempt / 输出目录 | **不存在**（`outputs/r7_s3_capacity_depth_20261012_attempt01/` 未创建） |
| 冻结的深度扩张规则 | identity-gated：新增块 `i∈{4..7}` 在**每个参数**上镜像块 `i-4`，并把两个输出投影（`proj`、`ff.net.3`）清零 ⇒ 每个新增块恰为恒等；CPU preflight 对真实迁移父（sha `fba27e26…`）得 `bit_identical=True`、`added=[4,5,6,7]`、严格 `load_state_dict` OK |
| 若将来运行的预算 | per-seed 28800 s、planned 86400 s、hard 115200 s；`ESTIMATED_PEAK_BYTES = 15 GiB`（16,106,127,360 B，> K=8 实测 14,552,137,728 B = 13.55 GiB），驱动强制 `MIN_ESTIMATED_PEAK_BYTES = 14 GiB` |

## 4. 成本

**0 GPU-h**：未启动任何 GPU 进程（两块 GPU 全程空闲）。网络 0；无 attempt 目录；soft/hard overrun 均 0。
唯一已发生的成本是 CPU 侧的实现与测试（不计 GPU 账）。

## 5. limitations

- **未运行**：没有 attempt、没有 seed、没有读数；本轮 brief §3 的预声明出口**既未触发也未满足**，
  因为该轮从未执行。
- **跳过依据是旁证、且跨因子**：依据是 K4→8 臂的**单 seed** 读数，而 `K` 与主干 `depth` 不是同一因子；
  这不足以判定 depth 4→8 无效，只记录操作者基于成本/收益的决定。
- **不是科学判定**：本页不产生 primary/gate 结论，不改变 r、不读 test、不宣布任何通过或失败。
- **可复现等级**：无结果产物可复现；只有驱动/测试/code.zip 的身份（`710612d` + `de9f6bc4…`）可核查。

## 6. 下一动作（S3）

不执行本轮的 GPU 部分。**下一动作（S3）**：由操作者与主链路**重新制定方向**（架构 / 数据），
并据此开新的 round brief；本轮 brief `docs/goals/s3-capacity-or-architecture.md` 保留为
"已准备、未运行"的记录，**不重开**。宽输入支与训练配方支此前已按预声明关闭，同样不重开。
test 未读、r=0、S4 未启动不变。
