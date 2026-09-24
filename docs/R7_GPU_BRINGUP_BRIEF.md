# R7 GPU bring-up 简报（goal 模式长文源）

本文件是「goal 模式目标」的长文补充。goal 模式的 objective 有 **4000 字符上限**，
因此约束细节放在这里，由 objective 指向本文件。两份合起来等价于完整提示词。

状态：**尚未执行**。本文件只是任务定义，不是结果；执行证据另记。

---

## 0. 首要规则：先读 GitHub，禁止依赖本文件里的旧状态

开始任何工作前，必须通过 GitHub REST API / git 实时核对：

1. `r7/weather-reasoning` 当前 HEAD；
2. Draft PR #12 状态；
3. 全部 open issues；
4. 重点读 #20、#13、#5、#6、#7、#8、#1 的最新正文与评论；
5. `docs/R7_RESEARCH_PLAN.md`、`docs/R7_TASK_QUEUE.md`、`docs/R7_CPU_REFINEMENT_RESULTS.md`；
6. 最近 GitHub Actions / CI 状态。

**若 GitHub 当前状态与本文件冲突，以 GitHub 为准**，并记录差异。

> 本机无 `gh` CLI、无 GitHub token（未认证写接口返回 401）。因此"把结果写回 issue"
> 需要先解决凭据问题；在解决之前，证据写进仓库 `docs/`，并把可直接粘贴的 issue 评论
> 草稿放进最终报告，标注**尚未发布**。不得声称已发布。

---

## 1. 硬件

本机有两张 RTX 3090。不得假设显存容量、CUDA/驱动版本或 NVLink 状态，第一步必须实测：

```bash
nvidia-smi
nvidia-smi topo -m
.venv/bin/python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, p.total_memory / 1024**3)
print("bf16:", torch.cuda.is_bf16_supported())
PY
```

把真实硬件信息记录到对应 GitHub issue（或在无法写入时记入仓库文档）。

**不要把两张显卡的显存相加**描述成"单模型可用显存"。默认先验证单卡；
双卡阶段优先 DDP 数据并行；除非确有必要，不要一开始就做 FSDP / tensor parallel / model parallel。

---

## 2. 工作方式：issue 驱动、持续推进

四态：`TODO → IN_PROGRESS → VERIFY → DONE`。被真实外部依赖阻塞时标 `BLOCKED` 并**写出那个依赖**。

若某任务在等 CI / workflow / 长训练：标 `WAITING`，**立即切换到其他未阻塞 issue**，
不要不停轮询等待。

每次实现后：Implement → Test → Inspect → Fix → Test → GitHub CI → issue 更新。
**只有满足 acceptance criteria 才关闭 issue。**

父科研 issue 与工程子 issue 必须分开：**"代码能跑"不等于"科研假设成立"**。
不得因为一次实验好看就关闭 #5/#6/#7/#8。

---

## 3. 第一优先级：Issue #20 GPU 显存验收

先读 #20 最新内容。若仓库还没有专门的 GPU bring-up 子 issue，
新建 child issue（例如 `[R7-GPU] Dual-3090 bring-up and recursive-memory validation`）并关联 #20。

> 注意：#20 的正文明确写着 **"No GPU rental is authorized by this issue"**，
> 且既有关联子项 #21 已关闭（streamed truncated backward，见 `docs/R7_STREAMED_TRAINING.md`）。
> 新一轮要做的，是它列出的剩余验收：真实 shape 配置下 K=1/2/4/8 的
> **实际 GPU peak allocated/reserved 与墙钟时间**。

目标不是马上长时间训练，而是先建立可靠 GPU 基线。必须测量
`GenericRecursiveWeatherForecaster` 与 `ProcessForecastCoReasoner`：

- 推理深度 K = 1, 2, 4, 8；
- 至少比较 A) full BPTT、B) streamed/truncated recursive training、C) activation checkpointing on/off。

每条记录：`max_memory_allocated()`、`max_memory_reserved()`、forward 时间、backward 时间、
optimizer-step 时间、总 step 时间、batch size、tensor shape、参数量、K、dtype、
activation checkpointing、training mode、GPU 型号、PyTorch/CUDA 版本。

计时前必须 `torch.cuda.synchronize()`；显存测量前必须 `torch.cuda.reset_peak_memory_stats()`。
**不得用 nvidia-smi 瞬时值替代 PyTorch peak memory**（nvidia-smi 可作辅助证据）。

---

## 4. GPU 验收顺序

### Stage A — 单卡 smoke
`CUDA_VISIBLE_DEVICES=0`，小模型 + 真实/已缓存 fixture。目标：forward/backward/optimizer
update 均为有限值、checkpoint save/load、BF16 可用性、K=1/2/4/8 显存曲线。先不追求精度。

### Stage B — 候选规模单卡
逐步增到 R7 计划目标约 15–30M 参数。**不要直接跳到最大配置**；每次只改一个主要因素
（dim / depth / batch / K / checkpointing），找到单卡安全配置。
显存要留安全余量，不要训练到接近 100% 才宣布可运行。

### Stage C — 双卡 DDP
只有单卡路径稳定后再做，优先 `torchrun --nproc_per_node=2 ...`。验证：
两卡均实际参与、DDP loss 与单卡参考行为一致、sampler 正确、global batch 定义清楚、
checkpoint 只保存一次、resume 正确、无 dataset duplication、无验证样本重复统计。

记录 `global_batch = per_gpu_batch × world_size × gradient_accumulation`。
**不要把 DDP 的吞吐提升称为单模型显存提升。**

> 现状：仓库中**没有任何** DDP / `torchrun` / `init_process_group` 代码（已核对）。
> 双卡阶段需要先写 launcher 与采样器校验。

---

## 5. 数据：优先复用已核实的真实 ERA5

先读 #13 最新状态。仓库已有真实公开 ERA5 CPU 管线与连续数据实验，**不要重造数据系统**。

优先复用：Earthmover / ERA5、native 0.25°、11 channels、12×12 区域、
continuous-250d profile、2018 train / 2019 val / 2020 test、
1000 six-hour timestamps/year、998 one-step windows/split。

> **重要限制**：本机 `data/raw|interim|processed` 目前只有 `.gitkeep`。
> continuous-250d 多变量缓存在 GitHub Actions artifact 中（由
> `data/download/*_replay.py` 的 pin 校验，如 `PINNED_CONTINUOUS_SHA256`），
> **不在本机磁盘上**，取回需要网络与授权。本机唯一的真实数据是
> `tests/fixtures/r7_arco_t2m.json`（9×5×7 单变量 t2m，`scientific_training_ready=false`）。
> 因此：显存/延迟基线可以用合成 shape，但必须写明**这不是多变量真实 ERA5 基线**；
> 要真实多变量数据，先按 `pinned-artifact-replay` 取回并校验 hash。

用途：GPU bring-up、显存测试、训练速度测试、小规模 convergence 测试。
注意：**这仍不是最终东亚论文 benchmark。**

禁止：把插值 ERA5 称为更高分辨率 truth；随机时间 split；用 test set 调阈值；
把 CPU pilot 当作 SOTA 结果。

---

## 6. 第二阶段：正式比较 #5 与 #6

当 #20 工程验收基本完成后进入。必须公平比较 Generic Recursive vs Process–Forecast Co-Reasoning。

控制条件：相同数据、相同训练窗口、相同随机种子、相近参数量、相近更新次数、
明确的 FLOPs / runtime 差异、相同 evaluation cases。推理深度 K = 1/2/4，
视 GPU 情况考虑 6/8；**至少 3 seeds**，**不要只汇报最优 seed**。

同时保留对照：Process 侧 `anchored + free`、no process auxiliary loss、no forecast feedback、
no recursion；Generic 侧 same recursive depth、same spatial solver feedback option。

> 仓库已有 `spatial_solver_feedback`，目前是 **opt-in、默认 False**（#57 结论）。
> **不得**因为 CPU 小试部分变量改善就把它改成默认 True。

---

## 7. Process State 的科研判据（#6）

核心不是"Process 模型能运行"，而是证明 Process-aware Co-Reasoning 在**公平预算**下
优于 Generic Recursive baseline。重点观察 T2M、U10、V10、MSLP、T850/T500、Q850、
U/V 850/500，以及 RMSE / ACC / rollout stability。

**若某些变量变差，必须保留结果。** 禁止：修改 test set、删除坏变量、只选最好 seed、
无限增加训练直到赢、调 validation 后再看 test 再继续调。

---

## 8. Adaptive reasoning（#7）

不要首先重新设计 halting controller。已有 CPU 证据显示 validation-selected policy
多次退回 full-depth（见 `docs/R7_POLICY_SELECTION.md`、`docs/R7_ADAPTIVE_HALTING.md`）。

GPU 阶段先回答："**固定递归模型本身是否足够稳定？**"只有 fixed K（K1/K2/K4）稳定后，
才继续 adaptive halting。

Adaptive 必须比较 fixed Kmax vs adaptive，并同时报告 RMSE/ACC、avg reasoning steps、
**实际墙钟延迟**、GPU peak memory、samples/sec。
**不得**用"平均 step 数减少 30%"直接声称"推理时间减少 30%"——必须实测。

---

## 9. Rollout（#8）

正式 GPU 小模型稳定后，运行 +6h / +12h / +24h / +48h / +72h。
必须是真正 autoregressive free rollout：模型输出 → 作为下一时刻输入。
**不能偷偷用未来 ERA5 替换历史状态。** 报告 per-variable RMSE、ACC、
lead-time curve、recursive-depth curve。后续再扩 East Asia。

---

## 10. 双卡 3090 阶段禁止过早做的事

暂时不要：全球 0.25° 训练；1B 参数模型；diffusion backbone；MoE；
FSDP（除非 DDP 明确不够）；adaptive-resolution urban extension #9；1 km 动态预测；
RL reasoning；natural-language CoT；test-time 搜索未来真值；大规模超参 sweep。

GPU 的第一价值是：验证真实训练显存 + 让已有模型充分训练 + 获得可靠的多 seed 对照，
**而不是立刻扩大 scope**。

---

## 11. 第一轮执行目标（30–60 分钟工程 bring-up）

第一轮不要跑通宵。完成：

1. 获取 GitHub issues 最新状态；
2. 创建/更新 GPU bring-up issue；
3. nvidia-smi / CUDA / PyTorch 环境记录；
4. 单卡真实数据 smoke；
5. K=1/2/4/8 显存曲线；
6. full vs streamed training 显存比较；
7. activation checkpointing on/off；
8. BF16 是否稳定；
9. checkpoint save/resume；
10. 两卡 DDP 10–50 step smoke；
11. CI / tests；
12. 结果写回 issue（无法写入时记入 `docs/` 并给草稿）。

若出现 OOM：不要只降低参数然后结束。记录失败配置、OOM 前 peak（如可得）、batch、K、
dim/depth、checkpointing、dtype，然后**有序**降低配置。

---

## 12. 第一轮结束时必须输出的报告

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

`Next action` 必须明确指出是 A 扩模型 / B 长训练 / C 修显存 / D DDP 优化 / E 数据扩展。
**不得只说"继续训练"。**

---

## 13. 安全与真实性约束

严禁伪造：GPU 显存、GPU 型号、训练耗时、实验结果、checkpoint、CI 结果、SOTA 结果。
任何数据必须来自真实运行。**CPU 测试不能声称是 GPU 验证。工程通过不能声称科研假设通过。**
没有证据时明确写：**NOT YET VERIFIED**。

被取消、排队或 skipped 的 CI 运行**不算通过**（`docs/R7_MANUAL_ITERATION.md`）。
不得为让门禁变绿而放宽/跳过测试与阈值。

---

## 14. 持续推进规则

如果 GPU benchmark 正在跑，不要一直等。转去做：issue #13 数据检查、
#5/#6 公平参数预算检查、config cleanup、DDP launcher、checkpoint contract、
profiling utilities、evaluation scripts、documentation。然后再回来验收运行结果。
**除非全部任务均被阻塞，否则持续推进。**

---

## 附：本机已核实事实（2026-09-24，执行时仍须复核）

| 项 | 实测值 |
| --- | --- |
| GPU | 2 × NVIDIA GeForce RTX 3090，各 24576 MiB（torch 报 23.56 GiB），cc 8.6 |
| 驱动 | 580.173.02 |
| 拓扑 | GPU0↔GPU1 = `SYS`（跨 NUMA PCIe，**无 NVLink**，nvlink 计数为 0） |
| torch | 2.11.0+cu128 |
| CUDA | 12.8，`is_available()`=True，device_count=2 |
| BF16 | `is_bf16_supported()`=True |
| HEAD | `c36e3f7f04d8b2bce29d207d1b1bf7ed5ee39e05`（= `origin/r7/weather-reasoning`） |
| 本地测试 | `745 passed, 3 skipped, 0 failed`（102.36s） |
| CI（push，HEAD） | run `35899051997` / job `107310215192` = **success** |
| CI（PR 触发） | run `35899060901` = **skipped**；本分支 PR 触发被 `ci.yml` 条件刻意跳过，**不能作通过依据** |
| 磁盘 | `/data` 762G 可用 |

现成工具（真实存在，勿重写）：

- `train_r7_local.py --device cuda --bf16`（`training/r7_local_runner.py` 已含
  `torch.cuda.synchronize`、`reset_peak_memory_stats` 与 `peak_allocated_bytes` /
  `peak_reserved_bytes` 记录）；
- `scripts/profile_r7_local.py --steps 1 2 4 8`（streamed-training shape profiler，内部走合成）；
- `profile_r7_inference.py`（#38，含每 forward 的 CUDA peak 计数器与同步，`--device cuda --precision bf16`）；
- `scripts/smoke_r7_streaming.py --device cuda --bf16`；
- `training/r7_streaming.py::train_streamed_update`（`detach_between_steps=True` 语义，FP32/BF16，未实现分布式）；
- 模型开关：`GenericRecursiveWeatherForecaster` / `ProcessForecastCoReasoner`
  的 `activation_checkpointing`、`detach_between_steps`（`configs/r7_*_smoke.yaml` 有示例）。

工程注意：`pytest.ini` 已注册 `gpu` / `network` / `slow` 三个 marker 且带 `--strict-markers`，
但**当前没有任何 `@pytest.mark.gpu` 测试**；CI 跑裸 `pytest -q`（不排除 marker），
因此新增 GPU 测试必须**在无 CUDA 时自行 skip**，否则会在 CPU CI 上失败。
