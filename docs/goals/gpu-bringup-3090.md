# R7 GPU 工程验收（双 3090 bring-up）

本文件是 goal 模式目标的长文源（R-034）。objective 有 **4000 字符硬上限**，所以约束细节在此，
objective 只放要点 + 指向本文件的路径。

**状态：尚未执行。** 本文件是任务定义，不是结果。执行证据另记（见「证据落点」）。

---

## 0. 首要规则：先读 GitHub，禁止依赖本文件里的旧状态

开始任何工作前，用 GitHub REST API / git 实时核对：

1. `r7/weather-reasoning` 当前 HEAD；
2. Draft PR #12 状态；
3. 全部 open issues（重点 #20、#13、#5、#6、#7、#8、#1）的最新正文与评论；
4. 最近 Actions / CI 状态（注意 run 的 `event` 与 `conclusion`）。

**若 GitHub 当前状态与本文件冲突，以 GitHub 为准**，并把差异记进最终报告。

### 本机 GitHub 访问的现实（已实测，2026-09-24）

| 能力 | 状态 | 证据 |
| --- | --- | --- |
| 读 issue / PR / Actions | **可用**（未认证 REST） | 仓库 public；`curl -s https://api.github.com/repos/Eswink/UrbanPiDiT_R2/issues/20` 返回 200 |
| 读限额 | 60 次/小时 | `GET /rate_limit` |
| 写 issue 评论 | **不可用** | `POST .../issues/20/comments` → **401** |
| `gh` CLI | **不存在** | `command -v gh` 为空 |
| GitHub token | **不存在** | 环境无 `GITHUB_*`；`~/.netrc` 只有 wandb |
| `git push` | **可用** | SSH 身份 `sqy941013`，`ssh -T git@github.com` 认证成功 |

**因此**：「把结果写回 issue」这一步当前**做不到**。降级写法 —— 证据写进仓库 `docs/`，
并在最终报告里给出**可直接粘贴的 issue 评论草稿**，明确标注「尚未发布」。
**不得**声称已发布、不得伪造 issue 链接。

---

## 1. 硬件（实测真值，执行时仍须复核）

```
GPU0/GPU1 = NVIDIA GeForce RTX 3090，各 24576 MiB（torch 报 23.56 GiB），cc 8.6
驱动 580.173.02 | torch 2.11.0+cu128 | CUDA 12.8 | cuda_available=True | device_count=2 | bf16=True
topo: GPU0↔GPU1 = SYS（跨 NUMA PCIe，GPU0 在 NUMA 0 / GPU1 在 NUMA 1）
NVLink: 全部 inActive（nvlink -s 明确报 "all links are inActive"）
```

**不要**假设 NVLink、显存容量或版本；**不要把两张卡的显存相加**说成「单模型可用显存」。
默认先验证单卡；双卡优先 DDP；除非确有必要，不要一开始做 FSDP / tensor parallel / model parallel。

---

## 2. 工作方式：issue 驱动 + 持续推进

- 四态 `TODO → IN_PROGRESS → VERIFY → DONE`；被真实外部依赖阻塞时标 `BLOCKED` 并**写出那个依赖**。
- 有任务在等 CI / 长训练时：标 `WAITING`，**立即切换到其他未阻塞任务**，不要轮询空等。
- 每次实现后：Implement → Test → Inspect → Fix → Test → CI → issue 更新。
- **只有满足 acceptance criteria 才关闭 issue**；父科研 issue 与工程子 issue 必须分开 ——
  **「代码能跑」不等于「科研假设成立」**。不得因为一次实验好看就关闭 #5/#6/#7/#8。

---

## 3. 第一优先级：Issue #20 GPU 显存验收

先读 #20 最新内容与评论。要点（2026-09-24 实读）：

- #20 正文明确 **"No GPU rental is authorized by this issue."**
- 关联工程子项 **#21 已关闭**（streamed truncated backward；见 `docs/R7_STREAMED_TRAINING.md`）。
  它交付的是**逻辑 saved-tensor 字节**（K=1/2/4/8 为 701368 bytes），**不是 GPU 显存测量**。
- #20 仍列的剩余验收：真实 shape 配置下 K=1/2/4/8 的**实际 GPU peak allocated/reserved 与墙钟时间**、
  multi-seed 优化/技巧比较、并入最终训练入口。

若仓库还没有专门的 GPU bring-up 子 issue，新建并关联 #20（标题例：
`[R7-GPU] Dual-3090 bring-up and recursive-memory validation`）。
**注意 issue 写入当前不可用**（见 §0）——若无法创建，就在报告里给出草稿并标注未发布。

必须测量 `GenericRecursiveWeatherForecaster` 与 `ProcessForecastCoReasoner`：

- K = 1, 2, 4, 8；
- 至少比较 A) full BPTT、B) streamed/truncated recursive training、C) activation checkpointing on/off。

每条记录：`torch.cuda.max_memory_allocated()`、`max_memory_reserved()`、forward / backward /
optimizer-step / total-step 时间、batch size、tensor shape、参数量、K、dtype、
activation checkpointing、training mode、GPU 型号、PyTorch / CUDA 版本。

计时前必须 `torch.cuda.synchronize()`；显存测量前必须 `torch.cuda.reset_peak_memory_stats()`。
**不得用 nvidia-smi 瞬时值替代 PyTorch peak memory**（nvidia-smi 只作辅助证据）。

---

## 4. 验收顺序

### Stage A — 单卡 smoke
`CUDA_VISIBLE_DEVICES=0`，小模型 + 真实/已缓存 fixture。目标：forward / backward / optimizer
update 均为有限值、checkpoint save/load、BF16 可用性、K=1/2/4/8 显存曲线。**先不追求精度。**

### Stage B — 候选规模单卡
逐步增到 R7 计划目标约 15–30M 参数。**不要直接跳到最大配置**；每次只改一个主要因素
（dim / depth / batch / K / checkpointing），找到单卡安全配置。
显存留安全余量 —— 不要训练到接近 100% 才宣布可运行。

### Stage C — 双卡 DDP
只有单卡路径稳定后再做，优先 `torchrun --nproc_per_node=2`（`.venv/bin/torchrun` 已存在）。验证：

- 两卡均实际参与；
- DDP loss 与单卡参考行为一致；
- sampler 正确、无 dataset duplication；
- global batch 定义清楚；
- checkpoint 只保存一次、resume 正确；
- 验证样本无重复统计。

记录 `global_batch = per_gpu_batch × world_size × gradient_accumulation`。
**不要把 DDP 的吞吐提升称为单模型显存提升。**

> 现状：仓库中**没有任何** DDP / `torchrun` / `init_process_group` 代码（已核对）。
> 双卡阶段要先写 launcher 与采样器校验。

---

## 5. 数据：优先复用已核实的真实 ERA5

先读 #13 最新状态。仓库已有真实公开 ERA5 CPU 管线与连续数据实验，**不要重造数据系统**。

目标口径：Earthmover / ERA5、native 0.25°、11 channels、12×12 区域、continuous-250d、
2018 train / 2019 val / 2020 test、1000 six-hour timestamps/year、998 one-step windows/split。
用途：GPU bring-up、显存测试、速度测试、小规模 convergence 测试。
**这仍不是最终东亚论文 benchmark。**

### 本机真实数据现状（关键限制）

| 项 | 实测 |
| --- | --- |
| `data/raw`、`data/interim`、`data/processed` | **只有 `.gitkeep`**（原始数据只读且不在本机） |
| continuous-250d 多变量 ERA5 缓存 | 在 GitHub Actions artifact，由 `data/download/*_replay.py` 的 pin 校验；**不在本机磁盘**，取回需网络与授权 |
| 本机唯一真实数据 | `tests/fixtures/r7_arco_t2m.json`：9×5×7、单变量 t2m、自带 `scientific_training_ready: false` |

**因此**：显存 / 延迟基线可以用合成 shape + 该 fixture 跑通，但报告里**必须写明
「这不是多变量真实 ERA5 基线」**，也**不是更高分辨率真值**。
要真实多变量数据，先按 `.agents/skills/pinned-artifact-replay/SKILL.md` 取回并校验 hash。

禁止：把插值 ERA5 称为更高分辨率 truth；随机时间 split；用 test set 调阈值；
把 CPU pilot 当作 SOTA 结果。

---

## 6. 第二阶段：正式比较 #5 与 #6

当 #20 工程验收基本完成后进入。必须公平比较 Generic Recursive vs Process–Forecast Co-Reasoning。

控制：相同数据、相同训练窗口、相同随机种子、相近参数量、相近更新次数、明确 FLOPs / runtime 差异、
相同 evaluation cases。K = 1/2/4，视 GPU 情况考虑 6/8；**至少 3 seeds**，**不要只汇报最优 seed**。

同时保留对照 —— Process 侧：`anchored + free`、no process auxiliary loss、no forecast feedback、
no recursion；Generic 侧：same recursive depth、same spatial solver feedback option。

> `spatial_solver_feedback` 目前是 **opt-in、默认 False**（#57 结论）。
> **不得**因为 CPU 小试部分变量改善就把它改成默认 True。

---

## 7. Process State 的科研判据（#6）

核心不是「Process 模型能运行」，而是证明 Process-aware Co-Reasoning 在**公平预算**下
优于 Generic Recursive baseline。重点观察 T2M、U10、V10、MSLP、T850/T500、Q850、U/V 850/500，
以及 RMSE / ACC / rollout stability。

**若某些变量变差，必须保留结果。** 禁止：修改 test set、删除坏变量、只选最好 seed、
无限增加训练直到赢、调 validation 后再看 test 再继续调。

> 已有三种子结论是**混合的**：spatial feedback 改善部分风场分数但使 T500 变差；
> process 未一致优于 generic（`docs/R7_TASK_QUEUE.md`、`docs/R7_CPU_REFINEMENT_RESULTS.md`）。
> 本轮不得把这条改写成正面结论。

---

## 8. Adaptive reasoning（#7）

不要首先重新设计 halting controller。已有 CPU 证据显示 validation-selected policy 多次退回 full-depth。
GPU 阶段先回答：**「固定递归模型本身是否足够稳定？」** 只有 fixed K（K1/K2/K4）稳定后，
才继续 adaptive halting。

Adaptive 必须比较 fixed Kmax vs adaptive，并同时报告 RMSE/ACC、avg reasoning steps、
**实际墙钟延迟**、GPU peak memory、samples/sec。
**不得**用「平均 step 数减少 30%」直接声称「推理时间减少 30%」——必须实测。

---

## 9. Rollout（#8）

正式 GPU 小模型稳定后，运行 +6h / +12h / +24h / +48h / +72h。
必须是真正 autoregressive free rollout：模型输出 → 作为下一时刻输入。
**不能偷偷用未来 ERA5 替换历史状态。** 报告 per-variable RMSE、ACC、
lead-time curve、recursive-depth curve。后续再扩 East Asia。

---

## 10. 双卡 3090 阶段禁止过早做的事

暂时不要：全球 0.25° 训练；1B 参数模型；diffusion backbone；MoE；FSDP（除非 DDP 明确不够）；
adaptive-resolution urban extension #9；1 km 动态预测；RL reasoning；natural-language CoT；
test-time 搜索未来真值；大规模超参 sweep。

GPU 的第一价值是：**验证真实训练显存 + 让已有模型充分训练 + 获得可靠的多 seed 对照**，
而不是立刻扩大 scope。

---

## 11. 本轮执行目标（约 30–60 分钟工程 bring-up，不要跑通宵）

1. 获取 GitHub issues 最新状态；
2. 创建 / 更新 GPU bring-up issue（写入不可用时给草稿）；
3. `nvidia-smi` / CUDA / PyTorch 环境记录；
4. 单卡真实数据 smoke（本机可用的真实数据见 §5）；
5. K=1/2/4/8 显存曲线；
6. full vs streamed training 显存比较；
7. activation checkpointing on/off；
8. BF16 是否稳定；
9. checkpoint save/resume；
10. 两卡 DDP 10–50 step smoke；
11. CI / tests；
12. 结果写回 issue（或给未发布草稿）。

**OOM 时**：不要只降低参数然后结束。记录失败配置、OOM 前可得峰值、batch、K、dim/depth、
checkpointing、dtype，然后**有序**降低配置。

### 现成工具（真实存在，勿重写）

| 工具 | 说明 |
| --- | --- |
| `train_r7_local.py --config <cfg> --synthetic|--manifest ... --out <dir> --updates N --device cuda --bf16` | 已含 `synchronize` / `reset_peak_memory_stats` / `peak_allocated_bytes` / `peak_reserved_bytes` |
| `scripts/profile_r7_local.py --config <cfg> --out <dir> --device cuda --steps 1 2 4 8 --bf16` | K sweep 的 shape profiler（内部走合成） |
| `scripts/smoke_r7_streaming.py --steps N --updates N --device cuda --bf16` | 输出 `cuda_peak_allocated_bytes` / `cuda_peak_reserved_bytes` |
| `profile_r7_inference.py --checkpoint ... --manifest ... --device cuda --precision bf16` | #38 交付，含每 forward 的 CUDA peak 计数器 |
| `training/r7_local_runner.py::update_group` / `run_local_updates` | 单卡有界 runner，含同步与峰值统计 |
| `training/r7_streaming.py::backward_streamed_truncated` / `train_streamed_update` | `detach_between_steps=True` 语义；**未实现分布式** |

模型开关：`GenericRecursiveWeatherForecaster` / `ProcessForecastCoReasoner` 的
`activation_checkpointing`、`detach_between_steps`、`spatial_solver_feedback`（示例见 `configs/r7_*_smoke.yaml`）。

### 新写的 GPU 测试必须自行 skip

`pytest.ini` 已注册 `gpu` / `network` / `slow` 三个 marker 且带 `--strict-markers`，
但**当前没有任何 `@pytest.mark.gpu` 测试**；CI 跑裸 `pytest -q`（不排除 marker）。
因此新增 GPU 测试必须**在无 CUDA 时自行 skip**（用 `pytest.mark.skipif(not torch.cuda.is_available(), ...)`），
否则 CPU CI 会失败。

---

## 12. 结束时必须输出

```
### GPU environment
GPU0: / GPU1: / VRAM: / CUDA: / PyTorch: / BF16:

### Single-GPU memory table
| model | params | K | batch | mode | checkpoint | allocated | reserved | step time |

### Dual-GPU smoke
DDP: / global batch: / throughput: / checkpoint resume: / status:

### GitHub
Issue worked: / commit: / workflow: / tests: / open blockers:

### Next action
```

`Next action` 必须明确选 **A 扩模型 / B 长训练 / C 修显存 / D DDP 优化 / E 数据扩展** 之一。
**不得只说「继续训练」。**

---

## 13. 安全与真实性约束（对应硬约束与规则）

严禁伪造：GPU 显存、GPU 型号、训练耗时、实验结果、checkpoint、CI 结果、SOTA 结果。
任何数据必须来自真实运行。**CPU 测试不能声称是 GPU 验证。工程通过不能声称科研假设通过。**
没有证据时明确写 **NOT YET VERIFIED**。

- 被取消 / 排队 / **skipped** 的运行**不算通过**。
- 不得为让门禁变绿而放宽 / 跳过测试与阈值（R-009：**不得弱化判据以过门禁**）。
- 实验产物必须带 `scientific_claim: false` 并如实记 `limitations`（R-007）。
- 有界实验必须先把协议写入 `protocol.json` 再训练，并记其 digest（R-006）。
- 结论必须绑定**精确 SHA**；无法逐位一致时写明可复现等级。

---

## 14. 证据落点（按新成品存放约定）

| 内容 | 放哪 | 依据 |
| --- | --- | --- |
| 本目标的长文 | `docs/goals/gpu-bringup-3090.md`（原 `docs/R7_GPU_BRINGUP_BRIEF.md` 早于约定，保留原位） | R-034 |
| 工作期计划 | `.zcode/plans/`（忽略，不提交） | R-032 |
| 计划定稿 | `docs/plans/NNNN-<slug>.md`（提交，含「## 实际结果」） | R-032 |
| 影响架构 / 长期行为的决定 | `docs/decisions/NNNN-<slug>.md`（含 `## Context` / `## Decision` / `## Consequences`，**必须写负面后果**） | R-033 |
| 实验结论与证据 | `docs/R7_*.md` 与 `docs/R7_TASK_QUEUE.md` | — |
| 运行态（指标 JSON、日志、checkpoint） | `outputs/`、`logs/`（**不提交**） | R-035 |

治理层必须在版本控制内（R-037）—— `docs/goals/**` 属于治理层，新文件要 `git add`。

---

## 15. 本机已核实事实速查（2026-09-24；执行时仍须复核）

| 项 | 实测值 |
| --- | --- |
| HEAD | `50954c94eb0aa15fa61cb19440543b40c6c38326`（= `origin/r7/weather-reasoning`；本地无未提交改动） |
| main | `92a8c4d0cfef7908e2b358e0ba2a6b4b3f45eebc`（**不得改动**） |
| PR #12 | open，`draft: true`，head = `r7/weather-reasoning` |
| 本地全量测试 | **796 passed, 3 skipped, 0 failed**（105.68s；改动本文件后复跑为 108.05s，计数不变） |
| 阻断规则 | **34 条，0 违规**（`python tools/check_conventions.py`） |
| 报告规则 | 11 条，`--report` 下 192 命中（C 类，不阻断） |
| CI（push，HEAD） | run `35976102003` / job `107556844837` = **success** |
| CI（PR 触发） | run `35976106651` = **skipped**（`ci.yml` 刻意对 `r7/weather-reasoning` 的 PR 触发跳过；**不能**作通过依据） |
| 自测 | `tests/test_check_conventions.py` 83 passed；`tests/test_agent_hooks.py` 132 passed |
| hooks | 4 条（PreToolUse ×2、PostToolUse ×1、Stop ×1） |
| workflow 总数 | 18（`ci.yml` + 17 条实验） |
| 磁盘 | `/data` 约 762G 可用 |

**已实测通过的 GPU 路径**（本轮验证，可作起点）：

- `train_r7_local.py ... --device cuda --bf16` → `hardware: NVIDIA GeForce RTX 3090`，
  `device_type: cuda`，`bf16: True`，2 updates 时 peak allocated ≈ 18.6 MiB / reserved ≈ 24.0 MiB；
  resume 到 4 updates 成功（`--resume` + 端点递增）。
- `select_device('cuda')` / `'cuda:0'` / `'cuda:1'` / `bf16=True` 全部返回正常
  —— 即 **Q-012 的 `--device cuda` 缺陷已修复且在本机可用**。
- `scripts/smoke_r7_streaming.py --device cuda --bf16` 输出
  `cuda_peak_allocated_bytes` / `cuda_peak_reserved_bytes`。

---

## 16. 已知的文档漂移（发现于本轮，未擅自修）

`AGENTS.md` 有两处写「**当前 20 条**阻断规则」（第 78、94 行），而实测 checker 跑 **34 条**。
这是第六 / 第七遍规则扩充后未同步的计数。本轮的 bring-up 不受影响（规则本身在跑 34 条），
但若要修，应作为一次独立的小改动（`AGENTS.md` 属治理层，改动后需复跑 `check_conventions.py`）。
