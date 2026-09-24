# 按序解决并关闭开放 issue（含真实数据验证）

本文件是 goal 模式目标的长文源（R-034）。objective 有 4000 字符硬上限，细节在此。

**状态：尚未执行。** 本文件是任务定义，不是结果。

---

## 0. 首要规则：先读 GitHub 实时状态

1. 用 GitHub REST 核对 HEAD、Draft PR #12、全部 open issues、最近 Actions；
2. 逐条读 issue 正文与评论（**Acceptance 是唯一关闭依据**）；
3. 读 `docs/R7_TASK_QUEUE.md`：其中 "Prior work is not pending" 列出的工作**不要重做**。

与 GitHub 冲突时**以 GitHub 为准**，并把差异写进报告。

---

## 1. 依赖顺序（按此推进，不要跳）

```
#13 数据管线 ──┐
#20 显存/资源 ─┤
               ├─> #5 generic 递归基线 ─> #6 process co-reasoning ─> #7 adaptive ─> #8 rollout
               └─> #9 城市扩展（stretch，很可能保持 BLOCKED）
                                              #1 / PR #12（收尾，需人决策）
```

理由：#13 产出真实数据，#5–#8 都要用它；#7 依赖 #5/#6 稳定；#8 依赖 #5–#7；#9 明确写了 "stretch goal only after R7.1–R7.5 are stable"。

---

## 2. 真实数据：已实测可用（推翻早期"无数据"结论）

### 2.1 可用源与实测成本

匿名可读，无需注册、无需密钥：

| 源 | 变量覆盖 | 原生节奏 | 压力层数 | 实测成本 |
| --- | --- | --- | --- | --- |
| `gs://gcp-public-data-arco-era5/ar/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2` | **R7 所需 8 个全有**（含 `geopotential`） | 6h | 13 | **~2.5 s / 时次（全 11 通道）** |
| `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3` | 同样全有 | 1h | 37 | ~21 s / 时次（慢 8 倍，chunk 大） |

**优先用 wb13 那条**（6 小时节奏与 R7 cadence 一致，chunk 小 3 倍）。

访问方式（实测通过）：

```python
import xarray as xr
ds = xr.open_zarr("gs://gcp-public-data-arco-era5/ar/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2",
                  chunks=None, storage_options={"token": "anon"})
```

`R7 变量 → ARCO 名` 映射（`data/preprocess/r7_era5.py:27` 的 `DEFAULT_R7_ERA5_CHANNELS` 同源）：

```
t2m  -> 2m_temperature        (K)
u10  -> 10m_u_component_of_wind   v10 -> 10m_v_component_of_wind   (m s**-1)
mslp -> mean_sea_level_pressure    (Pa)
z*   -> geopotential @ level       (m**2 s**-2)
t*   -> temperature @ level        (K)
q*   -> specific_humidity @ level  (kg kg**-1)
u*/v*-> {u,v}_component_of_wind @ level (m s**-1)
```

单位已逐一核对，与 `canonical_unit()` 期望一致；纬度 0.25° 间距符合 preflight 要求。

### 2.2 两个关键实测事实

1. **ROI 越大不越贵**：ARCO chunk 是 `(1, 13, 721, 1440)`（整张全球场）。实测 12×12 用 32.4 s、
   161×161 用 28.1 s（3 个时次）——**几乎相同**。所以 #13 要的"更大区域上下文"几乎不增加成本，
   不要因为怕慢而缩 ROI。
2. **区域数据是真实观测场**：实测 2018-06 东亚 12×12 的 t2m 289.7–313.9 K、t850 286.5–298.9 K、
   t500 255.6–264.8 K，量级符合六月物理。**这不是插值、不是合成。**

### 2.3 数据纪律（硬约束）

- **必须**走 `.agents/skills/real-data-acquisition/SKILL.md`：只读 preflight → 人审 →
  显式 `--write` + `--max-raw-gib` → 目标须含 `BUILD_COMPLETE.json`。
- **禁止**把插值 ERA5 称为更高分辨率 truth；**禁止**合成数据冒充天气真值；
  合成只能用于工程 smoke 且必须标注 `scientific_claim: false`。
- **禁止**写入 `data/raw|interim|processed`（R-004）；新数据写到**新的**目标路径（R-001 排他创建）。
- 下载失败**必须**留 `status='failed-no-fallback'`、`synthetic_fallback=False` 记录（R-008），
  不得静默回退到合成。
- 出网客户端只允许出现在 `data/download/**`（R-016）。
- 单位不符**停止**，不得猜测换算。

### 2.4 搜索

本机 **Google / DuckDuckGo 不可达**；`WebSearch` 工具在当前模型下不可用。
可用：`WebFetch` 抓 `https://cn.bing.com/search?q=...`（实测能返回结果列表）、
`https://api.crossref.org/works?query=`、`https://api.openalex.org/works?search=`、
GBIF/GCS 桶列表、`https://weatherbench2.storage.googleapis.com/`。
**搜索得到的每个数字都必须回到一手来源核对**，不得把搜索摘要当证据。

---

## 3. 各 issue 的关闭判据（Acceptance 为准）

| Issue | 关闭所需 | 现状 |
| --- | --- | --- |
| **#20** | 真实 shape 下 K=1/2/4/8 的 GPU peak allocated/reserved 与墙钟时间；**multi-seed 优化/技巧对比**；并入最终训练入口 | 工程部分已在 `50954c94` 完成（`docs/R7_GPU_BRINGUP.md`）；**multi-seed 对比仍缺** |
| **#13** | ①converter 确定且无泄漏 ②单测用合成 fixture 故 CI 不需要网 ③本地有真实子集时跑可选 smoke ④provenance/normalization 与 manifest 同写 | 真实数据现已可取（§2），需按 ①②③④ 补齐并跑通 |
| **#5** | 与 #6 在**可比参数/FLOP 预算**下对比的 generic 递归基线 | 实现已在；缺公平预算下的正式对比 |
| **#6** | 与 #5 公平预算下**优于 generic**；交付 ablation（no process state / no forecast feedback / no recursion） | 三种子证据是**混合的**（spatial feedback 改善部分风场但 T500 变差）。**不得**改写成正面结论 |
| **#7** | adaptive 逼近 fixed-Kmax 精度且平均推理深度**显著更低** | 现有严格策略回退到 full depth；须在冻结的新验证协议下重评 |
| **#8** | 评估脚本产出 paper-ready 指标表，**不改训练代码**；6–72h；东扩 | 需冻结更广、时间上分离的 case，保持 test 隔离 |
| **#9** | 需**真实共址的高分辨率动态真值**（HRCLDAS/SMBFD/区域 NWP）；插值 ERA5 只能做 baseline | 我未找到开放免注册的该类数据；**若确认找不到，保持 BLOCKED 并写明依赖**，不要用插值冒充 |
| **#1 / #12** | 假设被证明 + PR 就绪 | PR 仍 Draft；**合并 main / 发 release 需人显式授权**，不得自动做 |

### 关闭纪律

- `TODO → IN_PROGRESS → VERIFY → DONE`；被真实依赖阻塞标 `BLOCKED` **并写出那个依赖**。
- **工程通过 ≠ 科研假设成立**。父 issue 在还有真实数据/硬件/科研关卡时保持开启。
- **负结论是有效结论**：公平对比做完且完整报告（含全部负面结果）后，"假设未被支持"是
  一个**已答复**的状态，可以据此收尾并如实说明；但**不得**为关闭而伪造正面结果、
  删坏变量、只挑最好 seed、改 test set 或先看 test 再调阈值。
- 被取消 / 排队 / skipped 的运行**不算通过**。

---

## 4. 写入 issue 的现实（必须先验证）

**已实测：本机没有 `gh`、没有 token，`POST /issues/<n>/comments` 返回 401。**
`~/.netrc` 只有 wandb 条目；`git push` 走 SSH（身份 `sqy941013`）可用。

因此"关闭 issue"这一步**当前无法自动完成**。处置顺序：

1. **先再确认一次**写路径（别假设，重新实测）；
2. 若仍不可写 → 证据写入仓库 `docs/`，并产出**可直接粘贴的 issue 评论与关闭说明**
   （含精确 SHA、run/job id、pass/skip 计数、未解决限制），**明确标注「尚未发布」**；
3. **禁止**声称已发布、**禁止**伪造 issue 链接或状态。

> 如需真正自动关 issue，需用户提供 token（PAT，`issues:write`）或先 `gh auth login`。
> 这一项是**用户侧动作**，不要反复重试耗尽预算。

---

## 5. 自迭代边界（用户已授权连续推进）

**可在无逐次确认下自主做**：
- 在 `r7/weather-reasoning` 上 commit + push（该分支是工作分支）；
- 跑本机有界的 CPU/GPU 实验（本机 2×RTX 3090 免费可用，**不需要租卡**）；
- 写测试、文档、脚本；修 bug；跑全量测试与门禁；
- 获取 §2 描述的**有界**真实 ERA5 子集。

**仍必须停下或不得做**（项目硬约束，不因"自迭代授权"而解除）：
- 合并 `main`、创建 release、force push、`reset --hard`、`branch -D`；
- **付费 GPU 租赁**、大规模长训练、多年度全量下载、破坏性数据操作；
- 改动 / 移动 / 删除 `data/raw|interim|processed` 与归档快照（`legacy_*`）；
- 放宽或跳过测试与判据；把 CPU 结果说成 GPU 验证；把工程通过说成科研结论。

**预算**：单次迭代 ≤ ~90 分钟；不跑通宵。每轮结束都要留下可核查证据并推送。

---

## 6. 每轮的固定动作

1. `git rev-parse HEAD` 记录起点；
2. 选当前顺序下**最早未完成**的一项（不要并行开多条主线）；
3. 实现 → 测试 → 自查 → 修 → 再测；
4. 跑全量 `pytest -q` 与 `python tools/check_conventions.py`（34 条阻断规则须 0 违规）；
5. commit（遵循 `type(r7): summary (#N)`）+ push；
6. **等 push 触发的 CI**（本分支 PR 触发被 `ci.yml` 刻意 skip，看 push run），
   记录 run id / job id / 结论；
7. 把结果写入 `docs/R7_*.md` 与 `docs/R7_TASK_QUEUE.md`，绑定精确 SHA；
8. 判定该 issue：DONE（Acceptance 满足）或 BLOCKED（写出依赖）；
9. 回到步骤 2。

**CI 在等的时候不要空转**：转做下一项的源码审查、测试或文档。

---

## 7. 每轮必须报告

```
### 本轮 issue
编号 / 起点 SHA / 终点 SHA

### 做了什么
改动面（哪些文件、哪些行为）

### 验证（实测）
命令 + 结果（含 pass/skip/fail 计数）、run id / job id

### 真实数据
用了哪个源、ROI、时次范围、SHA256、是否通过 preflight / BUILD_COMPLETE

### 关闭判定
DONE（列出满足的 Acceptance 条目）或 BLOCKED（写出真实依赖）

### 未做的事与负面结果
照实写；不得用「应该没问题」代替

### 下一项
明确写下一个 issue 编号与它的第一个具体动作
```

---

## 8. 停止条件

全部满足即结束本轮 goal：

- 每个 open issue 要么 **DONE 且有证据**，要么 **BLOCKED 且写出了真实依赖**；
- 没有未推送的本地改动；没有进行中的实验；
- 最新 SHA 有一条**未取消、未 skip 的绿色 push CI**；
- issue 评论（若写路径恢复）已发布；若仍不可写，草稿已在 `docs/` 且标注未发布。

**不得**为了达成"全部关闭"而降低判据。若某 issue 的科研关卡未被证据支持，
如实记录并保持开启（或按 §3 的"负结论"路径收尾），这**是**正确结果。

---

## 9. 本机已核实事实（2026-09-24；执行时仍须复核）

| 项 | 值 |
| --- | --- |
| HEAD | `c97f3c58f02dd92c140ad80b0159d4edf086bd98`（= origin，工作树干净） |
| PR #12 | open，**Draft** |
| open issues | #1, #5, #6, #7, #8, #9, #13, #20 |
| 本地测试 | 804 passed / 3 skipped / 0 failed（GPU 在场）；`CUDA_VISIBLE_DEVICES=""` 时 798 / 9 / 0 |
| 阻断规则 | **34 条，0 违规** |
| 报告规则 | 11 条（`--report`） |
| GPU | 2×RTX 3090 24 GiB，driver 580.173.02，NVLink 全 inActive，拓扑 `SYS` |
| 栈 | torch 2.11.0+cu128 / CUDA 12.8 / Python 3.12.3 / bf16 OK |
| 数据依赖 | `zarr 3.4.0`、`gcsfs 2026.8.1`、`xarray 2026.7.0`、`icechunk 2.2.2` **均已安装** |
| issue 写入 | **401**（无 token、无 gh） |
| 搜索 | Google/DuckDuckGo 不可达；cn.bing.com（经 WebFetch）、crossref、openalex 可用 |

**注意命名规则**（新文件会被阻断级门禁拦）：`snake_case.py`；禁词含
`final/old/new/tmp/temp/copy/backup/draft/deprecated/misc`；杂物桶目录名（`utils/common/core/...`）禁用；
`r7` 须在定界位置。新文件写好后要 `git add`（R-037 治理层必须在版本控制内）。

**新增测试**：R-009 记录基线为 320 函数 / 613 断言（实测 329 / 631）。新增测试不受限，
但**有意重构**测试时须同步更新 `tools/check_conventions.py` 的基线并在
`docs/rules/CHANGELOG.md` 说明。
