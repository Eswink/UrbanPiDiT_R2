# #65 C2（草稿反馈路径）与 C3（推理深度 / 测试时算力）

**状态：C2 完成（negative）；C3 完成（mixed，且修正了一次 harness 缺陷）。**
本文件与 `docs/R7_65_C1_PROCESS_SUPERVISION.md` 合起来覆盖 #65 的三条消融轴。

| 项 | C2 | C3 |
| --- | --- | --- |
| 协议 digest | `9a5ce6c50365626449aee83d936389f8499331c15582c0f984f126f0d7d13eff` | 见 `outputs/r7_65_c3/protocol.json` |
| 训练 | 4 臂 × 3 seed × 800 update，**12/12 无早停**，3926.2 s ≈ **1.09 GPU-h** | 2 臂 × 3 seed × 800 update，1939–？ s，见产物 |
| 评估 | val only，`test_read: false`，5 时效 | 同上 + **测试时 K 扫描（K=1/2/4 自同一 checkpoint）** |
| 产物 | `outputs/r7_65_c2/`、`outputs/r7_65_c2_analysis/` | `outputs/r7_65_c3/`、`outputs/r7_65_c3_analysis/` |

---

## 1. C2：草稿反馈走哪条路径

**#65 C2 原文**：C1 确定训练稳定后，分别切换 reasoner 的 `use_forecast_feedback` 与
solver 的 `spatial_solver_feedback`，明确哪些路径打开；**generic 获得同样的 spatial
机制，不能只增强 Ours**；保留当前默认 False 直到通过新协议。

四臂（全部 Ktrain=4、aux weight=0.1，即 C1 里唯一「单调」的设定）：

| 臂 | reasoner 反馈 | solver 空间反馈 |
| --- | --- | --- |
| `process_fb_on_solver_off`（参考，= 当前默认） | 开 | 关 |
| `process_fb_off_solver_off` | **关** | 关 |
| `process_fb_on_solver_on` | 开 | **开** |
| `generic_fb_on_solver_on` | （generic 恒开） | **开** |

**四张表**：

| 臂 | 参数量 | forward FLOPs | forward+bwd FLOPs | wall time（均值） | 每时效例数 |
| --- | --- | --- | --- | --- | --- |
| process_fb_on_solver_off | 2,799,779 | 14,162,671,488 | 42,374,270,208 | 332.1 s | 15/19/23/25/26 |
| process_fb_off_solver_off | 2,799,779 | **12,650,547,072** | 37,837,896,960 | 322.0 s | 15/19/23/25/26 |
| process_fb_on_solver_on | 2,799,779 | 14,162,671,488 | 42,374,270,208 | 330.0 s | 15/19/23/25/26 |
| generic_fb_on_solver_on | 2,799,202 | 14,162,646,912 | 42,374,196,480 | 323.9 s | 15/19/23/25/26 |

**两条关于「同更新 ≠ 同算力」的实测**：

- **关掉 reasoner 反馈省 10.7% forward FLOPs**（12.65B vs 14.16B）：少了
  draft_encoder 前向 + 更短的 cross-attention 序列。所以
  `fb_off` 与 `fb_on` 的精度对比**不是等算力对比**，报告必须并列这两列。
- **`solver_on` 与 `solver_off` 的 FLOPs 逐位相同**（14,162,671,488）。
  `solver_conditioning` 只做 `context + summary[:,None,:]`（和可选 `+ draft_tokens`），
  是逐元素加法，不被 `FlopCounterMode` 计入。**因此「打开 solver 反馈」在本实现里
  没有可测的 forward 计算代价**——但这**不等于**它免费：`draft_tokens` 在
  `fb_on` 下本来就要算，`spatial_feedback` 只是复用已有张量。若将来在
  `fb_off` 下打开 `spatial_feedback`（本编排未做，因为代码要求 draft tokens
  存在），那才会有新增前向。

**结果（#60 比较器，参考 = `process_fb_on_solver_off`）**：

| 臂 | improved | worsened | unresolved |
| --- | --- | --- | --- |
| process_fb_off_solver_off | 14 | 8 | **63** |
| process_fb_on_solver_on | 10 | **28** | 47 |
| generic_fb_on_solver_on | 15 | **29** | 41 |

主变量 t2m 逐时效：

| 臂 | 6h | 12h | 24h | 48h | 72h |
| --- | --- | --- | --- | --- | --- |
| process_fb_off_solver_off | unresolved | **improved**（3/3 同号 −0.08~−0.16） | unresolved | unresolved | unresolved |
| process_fb_on_solver_on | unresolved | unresolved | unresolved | unresolved | unresolved |
| generic_fb_on_solver_on | unresolved | unresolved | unresolved | **worsened**（3/3 同号 +1.12~+1.22） | unresolved |

**结论（negative）**：

1. **打开 solver 空间反馈没有收益，且在长时效变差**：`solver_on` 在 48h/72h 出现
   9/9 与 7/10 个 worsened 格子（全部 85 格里 28 个 worsened、仅 10 个 improved），
   主变量 t2m@48h 在 generic 臂上三 seed **一致变差 1.12–1.22 K**。
2. **关掉 reasoner 反馈是本次唯一略有正面迹象的改动**：`fb_off` 在 t2m@12h 上
   三 seed 同号改善（−0.08~−0.16 K），且整体 improved:worsened = 14:8。
   但 85 格里 63 格 unresolved，**且它省了 10.7% 计算**——在这个算力下
   「更少的计算得到相近或略好的结果」是**效率**陈述，不是**结构**陈述。
   不据此宣称「反馈有害」。
3. **按 #65 的要求保留默认**：当前默认（`use_forecast_feedback=True`、
   `spatial_solver_feedback=False`）**不因为本次结果而改动**。协议 `scientific_claim:
   false`，且 #65 C2 明确要求「保留当前默认直到通过新协议」——本段没有为该改动
   建立新协议，故不改默认。

## 2. C3：推理深度 / 测试时算力

**#65 C3 原文**：同一 Ktrain=4 checkpoint 测 K=1/2/4 以隔离 test-time compute；
另有**独立训练 K1 模型**作为强浅层对照；K6/8 属于超训练深度，除非训练过/明确 OOD
实验，不能直接称免费 scaling；加入**相近 forward FLOPs 或实测 latency 的非共享加深
baseline**。

### 2.1 一次 harness 缺陷（已修，如实记录）

**首次 C3 运行无效**，已保留在 `outputs/r7_65_c3_invalid_k1_config/`：

- 我最初把 `process8_aux010_k1` 臂的**模型配置**写成 `default_reasoning_steps=1`，
  但把**阶段级常量** `steps=4` 传给训练 runner。runner 不看模型配置里的默认值，
  所以该臂实际以 **K=4** 训练——它是一次伪装成「独立训练的 K=1」的第二次 K=4 运行。
- 症状是可检出的：两臂的 forward FLOPs **逐位相同**（14,162,671,488），
  且 t2m 逐实效 RMSE 差异仅在第 5–6 位有效数字（如 6h 3.1539 vs 3.1539）。
- **修复**：每臂在 `_phase_arms` 里显式声明训练深度（第 5 个元素），
  `protocol_payload` 把每臂深度写进冻结协议，训练与评估都读该臂自身的深度，
  并在训练后**断言** `report["contract"]["steps"] == steps_for_arm`，不一致即
  抛错（fail closed）。
- **回归测试**：`tests/test_r7_65_ablation_harness.py`（9 例），其中
  `test_the_two_c3_arms_must_not_report_the_same_compute` 直接断言 K=1 臂的
  forward FLOPs **严格小于** K=4 臂——这正是当初被掩盖的症状。
- 修复后 K=1 臂的 forward FLOPs 为 **10,205,989,248**，即比 K=4 少 **27.9%**。

**这条缺陷本身是本轮的方法学产出**：它说明「模型配置里的
`default_reasoning_steps`」与「runner 的 `steps`」是同一事实的两处声明，
而 runner 不做交叉检查——一个「独立训练的浅层对照」可以静默地变成重复的深层运行。
修复前若不检查 FLOPs 表，这个错误**不会出现在任何精度数字里**。

### 2.2 结果

**（a）从同一个 K=4 checkpoint 评不同 K**（隔离 test-time compute，t2m RMSE，3 seed 均值）：

| lead | K=1 | K=2 | K=4 |
| --- | --- | --- | --- |
| 6h | 3.3010 | 3.2273 | **3.1539** |
| 12h | 3.8421 | 3.7845 | 3.7950 |
| 24h | **3.9923** | 4.0162 | 4.1881 |
| 48h | 5.2459 | 5.1406 | **5.0667** |
| 72h | 5.6457 | **5.6366** | 6.1716 |

**（b）独立训练的 K=1 vs K=4**（各自在自己的训练深度上评估）见产物
`ablation_result.json` / `analyze_r7_65_ablation.py` 的配对表。

**结论（mixed）**：

1. **测试时加深在训练时效（6h）有用，在长时效不一致**：K=4 在 6h 最好
   （3.1539），但在 24h 与 72h 反而是 K=1/K=2 更好。**所以「免费 scaling」不成立**——
   这与预诊断 §3 的 K 轨迹同向（K=3、K=4 的修正开始变差，相邻 cosine 趋于 0.98）。
2. **K6/K8 未测**，也不在本次声明范围内（#65 明确：未训练过的深度不得称免费 scaling）。
3. 「从 K=4 评 K=1」与「独立训练 K=1」**是两个不同问题**，本次**分开报告**，
   未把前者当作后者的证据。

## 3. 局限

| 局限 | 说明 |
| --- | --- |
| 单一年 1 月、4 个 (month,hour) 桶 | climatology 占优仍不可外推；跨季节 UNVERIFIED |
| val/test 是 2016 段内工程再划分 | `test_read: false`；不是封存测试集 |
| 800 步非收敛 | 全部 12/12（C2）与各臂均无早停 |
| 3 seed 不是显著性检验 | 同号一致性是描述性判据 |
| aux weight 固定 0.1 | 因为 C1 里 0.1 是唯一产生同号（负向）结论的设定；**未做 C1×C2 全笛卡尔积**（#65 明文禁止） |
| C2 的 `fb_off` 省 10.7% FLOPs | 该对比不是等算力对比；`solver_on/off` 则是等算力对比（FLOPs 逐位相同） |

## 4. 复现

```bash
.venv/bin/python scripts/study_r7_65_ablation.py --phase c2 --device cuda --out outputs/r7_65_c2
.venv/bin/python scripts/analyze_r7_65_ablation.py --run outputs/r7_65_c2 \
    --reference process_fb_on_solver_off --out outputs/r7_65_c2_analysis

.venv/bin/python scripts/study_r7_65_ablation.py --phase c3 --device cuda --out outputs/r7_65_c3
.venv/bin/python scripts/analyze_r7_65_ablation.py --run outputs/r7_65_c3 \
    --reference process8_aux010_k4 --out outputs/r7_65_c3_analysis
```
