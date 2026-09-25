# #64 B0 可学习性探针（S2 起步）：真实 D1 窗口，2026-09-26

Status: **B0 通过（按判据「可解释 loss 下降」，非科学结论）**。四个极小模型族在
固定 1/8/32 个真实 D1 训练窗口上均显著降低固定样本误差；梯度抵达 encoder 与
solver；历史索引 / lead 对应 / 反归一化逐样本核对无违例。`scientific_claim: false`——
过拟合只证明可学习性，不证明泛化；本页不与任何基线比较排名。

- 协议冻结：`outputs/r7_b0_learnability_run2/protocol.json`，
  digest `c008b59c8174b1fd…`（任何 optimizer step 之前落盘）。
- 数据：D1 store（`docs/R7_D1_ACQUISITION.md`）train.jsonl 的**前 N 个窗口**
  （N ∈ {1, 8, 32}，文件序，不挑样本）；val/test 窗口从未读取。
- 运行器：审计过的 `training/r7_local_runner.run_local_updates`（AdamW lr 2e-4、
  clip 1.0、batch 2、seed 41 唯一预声明、固定 200 更新端点、`error_if_nonfinite`）。
- 随机正则：dropout=0.0；AdamW weight_decay=1e-4 是运行器默认，已在协议声明。

## 结果（12/12 全部可解释下降）

训练 loss（首更新 → 第 200 更新）与固定窗口平均归一化 RMSE（训练前 → 训练后）：

| arm | 1 窗口 | 8 窗口 | 32 窗口 |
| --- | --- | --- | --- |
| native_window | 0.222→0.027 / RMSE ×2.79↓ | 0.205→0.101 / ×1.37↓ | 0.265→0.166 / ×1.16↓ |
| unet | 0.222→0.079 / ×1.66↓ | 0.204→0.104 / ×1.28↓ | 0.265→0.178 / ×1.10↓ |
| afno_small | 0.222→0.049 / ×2.10↓ | 0.205→0.118 / ×1.23↓ | 0.265→0.179 / ×1.10↓ |
| generic (K=3) | 0.222→0.025 / ×2.89↓ | 0.204→0.090 / ×1.47↓ | 0.265→0.151 / ×1.21↓ |

过拟合随窗口数增加而减弱（1 窗口 ≈2.8×，32 窗口 ≈1.1–1.2×）——这正是
「只证明可学习性」的预期形态，不是泛化证据。

## 每通道（32 窗口，物理单位 RMSE，训练前→后）

初始误差 ≈ persistence（模型是 persistence 底座 + 小初始化 tendency 头，见
`_small_head`），因此「训练后 < 初始」等价于「固定训练样本上低于 persistence」：

| 通道 | persistence≈初始 | native_window 后 | generic 后 |
| --- | --- | --- | --- |
| t2m (K) | 4.37 | 3.93 | 3.68 |
| u10 (m/s) | 1.36 | 1.18 | 1.16 |
| v10 (m/s) | 1.37 | 1.28 | 1.24 |
| mslp (Pa) | 218.7 | 190.3 | 173.4 |
| z850 (m²/s²) | 103.5 | 88.8 | 80.6 |
| t850 (K) | 1.56 | 1.42 | 1.37 |
| q850 (kg/kg) | 5.27e-4 | 4.66e-4 | 4.67e-4 |
| z500 | 156.4 | 122.4 | 111.8 |
| t500 (K) | 1.27 | 1.13 | 1.09 |
| z250 | 245.3 | 197.7 | 174.1 |
| …（17/17 通道全部下降，完整数值见 `b0_result.json`） | | | |

**17/17 通道**在两个代表臂上全部下降；四臂 × 三个窗口规模无一例外。

## 必查项（#64 B0 清单）

- **梯度到达 encoder/solver**：12/12 全过。逐子树范数记录在
  `b0_result.json`（如 generic@32w：encoder 4.3e-3、cell 1.6e-3、
  correction_head 2.6e-1，全部非零）。encoder 范数偏小是残差结构的正常表现，
  非断流——若断流，head 的梯度也会非零而 encoder 为零，可区分。
- **反归一化**：窗口逐样本「denormalize(样本) ≈ store 原始态」全过（rtol 1e-4）；
  物理 RMSE 按 store 的 train-only mean/std 计算。
- **输入历史索引**：history_times == [init−6h, init]，且 store 在记录的
  `history_indices` 处的帧与样本内容一致。
- **lead 对应**：target_time − init_time == +6h（单步）；样本 `lead_time_hours==6`。
- 反证：任何一处违例都会使探针在训练前中止（不训练）。

## 预算与身份

- GPU：2×RTX 3090 之一，总训练墙钟 562.4 s = **0.156 GPU-hours**（预算 ≤0.5），
  峰值显存 324 MiB；模型代码 digest `d9fb07f2…`。
- 产物：`outputs/r7_b0_learnability_run2/`（protocol.json、b0_result.json、
  12 组 checkpoint + 逐更新 loss 曲线）。
- 首次运行因脚本 bug（dict 推导缺 `.items()`）在梯度探针处崩溃，修复后换新
  目录重跑；失败运行 `outputs/r7_b0_learnability/` 原样保留，未删除未粉饰。

## 判定与边界

- **B0 判据满足**：四族极小模型在固定真实窗口上误差显著、可解释地下降，
  梯度路径与数据契约核查全过 → 解锁 #64 B1。
- **不是**：泛化、skill、与基线的比较排名、参数结论。B1（1–5M 强基线）需另行
  按 #64 协议执行；若 B1 用 D1 之外的数据，超出本 goal 授权范围。
