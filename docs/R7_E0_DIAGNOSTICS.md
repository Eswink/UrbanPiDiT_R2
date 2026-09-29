# E0：两轮预注册但未执行的诊断（执行记录）

**状态：两个诊断均已执行完毕；结论见 §4/§5，其中 §5 推翻了一条已归档的陈述。**
本轮 **0 GPU-h**、只读 val、只用已归档 checkpoint；未训练、未改任何 checkpoint 或数据。

| 项 | 值 |
| --- | --- |
| 起点 SHA | `git rev-parse HEAD` = `e1a915cbc996d9a553be0f717fd24e871a61fc90` |
| 预注册来源 | 二轮 §11（`docs/R7_71_72_ROUND_TWO_ATTRIBUTION.md:329-340`）、三轮 §13（`docs/R7_71_72_ROUND_THREE.md:463-472`） |
| 输入 checkpoint | `outputs/r7_71_72_round_two/seed{41,42,43}/training/*/update_0000400.pt`（12 个，逐文件 sha256 见产物） |
| 回放代码 | commit `d8aff68`（`git archive` 到 `/tmp/r7_e0/code_r2`）；实测 `model_code_digest` = `8d9262d1abb537f11fc0b74b98ec30e450eadf4e7bd7300ad32a957de5e38e44`，与该批 checkpoint 的 `model_code_sha256` **相同** |
| 数据 | M2 段 val `outputs/r7_m2_segment/store/manifests/val.jsonl`（22 窗口，sha256 `218512c8a1490dfc…`）；`test.jsonl` 全程未读 |
| 产物 | `outputs/r7_e0_diagnostic/`（12 × correction + 12 × oracle + 合并 `e0_correction_replay.json` + `e0_paired_stability.json` + `rw_b_cost_batch2.json`） |
| 驱动 | `scripts/study_r7_e0_correction_replay.py`（sha256 `5fcd312aec528742…`）、`scripts/study_r7_e0_paired_stability.py`（sha256 `32fe5ec7ecb893aa…`）；两个 sha 都与产物里记录的 `driver_sha256` **逐位相符** |
| 实测墙钟 | E0-1 **111.1 s**（CPU，12 checkpoint × 22 窗口 × (K=3 修正 + K=4 oracle)）；E0-2 < 1 s（纯 JSON 读） |
| `scientific_claim` | `false`（两份合并产物内均带该字段） |

**一处流程事实（如实记）**：E0-1 第一次跑完后，两个驱动的 `main()` 分别有 137 / 119 行，
越过 R-020（函数体 ≤100 行的 C 类目标态）——而本机跑测试时它们还是未跟踪文件，
`judge_as_committed` 不计入，**CI 上才暴露**。处置是把两个驱动**重组**（拆出
`import_code_root` / `measure_arms` / `check_archived_claim` 等，不改任何数值语句），
然后**用重组后的驱动把两个诊断各重跑一次**：E0-1 的 12 臂 × 2 诊断、E0-2 的全部数字
与第一次**逐字段相同**（只排除 `elapsed_seconds` 与驱动自身 hash）。
产物目录里现在是**重组后那次**的输出，因此记录的 `driver_sha256` 与仓库里的驱动一致、可直接复核；
R-020 命中数回到文档记载的 39，未改任何阈值或标记。

---

## 1. 预注册动作与它字面提到的工具

二轮 §11 的原文要求：用 12 个 checkpoint 跑 `diagnose_r7_gain.py`，在 val 上比较 **B 与 C/D 的
correction 幅度与符号**。三轮 §13 的原文要求：把二轮 `C−B`/`D−B` 与三轮 `E−A` 在 t2m 48/72h 的
逐 seed delta 并排，判断 0.25–0.29 K 是否跨两次独立运行稳定。

**一处必须如实记录的偏差**：`diagnose_r7_gain.py` 算的是**误差随深度的轨迹与 oracle 深度**
（`training/r7_gain_oracle.py`），**不输出** correction 幅度与符号。二轮 §11 把工具名与要测的量
对错了。仓库里真正算 correction 幅度/符号的是 `training/r7_correction_diagnostic.py`
（`update_energy` = 修正能量、`error_update_cosine` = 修正与误差的夹角余弦）。因此**两个都跑了**：
字面点名的工具跑了（§4），真正回答问题的工具也跑了（§3）。这不是新增判据，是把同一个预注册
问题用正确的仪器测出来。

## 2. 仪器与读法

| 量 | 定义（`training/r7_correction_diagnostic.py`） | 有利于预报的方向 |
| --- | --- | --- |
| `update_energy` | 第 3 轮修正 `d = Y_3 − Y_2` 的纬度加权均方（训练归一化单位） | 无所谓大小；"幅度放大"是待检验的现象 |
| `error_update_cosine` | `cos(e, d)`，`e = Y_2 − target` | **负**（修正朝着误差反方向走） |
| `wrong_direction` | `e·d ≥ 0` 的案例占比 | 越低越好 |
| `worsening` | `MSE(Y_3) > MSE(Y_2)` 的占比 | 越低越好 |
| `previous_update_cosine` | 相邻两轮修正的余弦 | 接近 1 表示"反复加同一风格 delta" |

三 seed 只给**一致性**，不引入任何显著性阈值；`update_energy` 是归一化平方量，
跨变量求平均只作索引，**不当开尔文读**。

## 3. E0-1（二轮 §11）：读取是否把 solver context 拉偏

四臂 × 三 seed，第 3 轮修正，22 个 val 窗口（均值；括号为 seed 极差的一半）：

| 臂 | `update_energy`（全变量） | `cos(e,d)`（全变量） | t2m `update_energy` | t2m `cos(e,d)` | t2m `wrong_direction` | t2m `previous_update_cosine` |
| --- | --- | --- | --- | --- | --- | --- |
| A `process_pooled` | 0.000631 (±0.000055) | −0.0844 | 0.001943 | **+0.2192** | 0.682 | 0.986 |
| B `process_spacetime_only` | 0.000567 (±0.000106) | −0.0902 | 0.001558 | +0.0505 | 0.561 | 0.863 |
| C `process_spacetime_rwa` | 0.000744 (±0.000025) | −0.0721 | 0.001997 | +0.0698 | 0.576 | 0.989 |
| D `process_rwa_capacity_control` | 0.000742 (±0.000024) | −0.0723 | 0.001998 | +0.0703 | 0.591 | 0.988 |

逐 seed 配对（focus − baseline；`same_sign` 由驱动机械判定）：

| 对 | Δ`update_energy`（全变量，逐 seed） | 同号 | Δ`cos(e,d)`（全变量，逐 seed） | 同号 |
| --- | --- | --- | --- | --- |
| **C − B** | +0.000178 / +0.000046 / +0.000307 | **是（正）** | +0.0092 / +0.0097 / +0.0355 | **是（正）** |
| **D − B** | +0.000175 / +0.000048 / +0.000304 | **是（正）** | +0.0092 / +0.0096 / +0.0349 | **是（正）** |
| B − A（附） | −0.000011 / +0.000009 / −0.000191 | 否 | −0.0178 / +0.0071 / −0.0070 | 否 |

**判读（按二轮 §11 预注册的那句话逐条对）**：

1. **「correction 幅度显著放大」→ 获支持（限定在全变量索引上）。**
   C 的修正能量在三个 seed 上**一致大于** B，比值 1.07 / 1.32 / 1.66（池化 C/B = 1.31）。
   **但 t2m 单变量上不同号**（seed42 是 −0.000190），所以这句话只在跨变量索引上成立。
2. **「与误差反相关」→ 未获支持，且方向相反。**
   预注册的读法是「读取让修正与误差反相关」。实测 `cos(e,d)` 在三个 seed 上**一致升高**
   （+0.0092/+0.0097/+0.0355）。原本的余弦≈0（近正交），读取后仍然≈0；没有任何一格转向反相关。
   `wrong_direction` 在 t2m 上也没有一致改善（−0.091/+0.045/+0.091）。
3. **幅度放大不是位置依赖带来的。** C 与 D 的数字逐 seed 几乎相同（`update_energy` 相对差
   <0.1%，`cos` 差 <0.001），而 D 的 query 按位置池化、读取**与位置无关**（第二轮 §3 的钉住探针）。
   即：放大效应来自「有这条读写通路」，不是来自「读是位置化的」。

**对 RW-B 的直接含义**：二轮 §11 提的下一假设是「readout 需要归一化/门控」，其**幅度**前提成立
（读取确实把修正整体推大），**方向**前提不成立（修正没有变得与误差更反相关）。门控要处理的是
幅度，不是方向。

## 4. E0-1 附：字面点名的 `diagnose_r7_gain.py`

`step_cost=0`、K=4、22 窗口；`mse_by_step` 是 K=1..4 的纬度加权归一化 MSE 均值：

| 臂 | K=1 | K=2 | K=3 | K=4 | K=1→K=4 |
| --- | --- | --- | --- | --- | --- |
| A `process_pooled` | 0.22069 | 0.21808 | 0.21700 | 0.21697 | −0.00372 |
| B `process_spacetime_only` | 0.21380 | 0.21084 | 0.20943 | **0.20907** | −0.00472 |
| C `process_spacetime_rwa` | 0.21388 | 0.21089 | 0.20975 | 0.21001 | −0.00387 |
| D `process_rwa_capacity_control` | 0.21387 | 0.21087 | 0.20973 | 0.20998 | −0.00389 |

**这个仪器在本设置下是退化的**：12 个 run 的 `mean_objective_regret` 全部为 `0.0`、
`missed_delayed_benefit_count` 全部为 `0`。`step_cost=0` 时 oracle 的最优解恒为「跑满 K」，
所以它**不能**回答任何停止/自适应问题。本轮**不引入**非零 `step_cost`（那会是一个新参数，
属于 §5 停止条件 2 的「新判据」）。可读出的只有一条：C/D 的轨迹在 K=3 取极小后 K=4 回升，
而 B 到 K=4 仍在降——与「读取放大修正、后续修正互相重叠」相容，但**训练协议用的是 K=3**，
K=4 在分布外，故这条只作观察，不作结论。

## 5. E0-2（三轮 §13）：0.25–0.29 K 是否跨两次独立运行稳定

输入：`outputs/r7_71_72_round_two/paired_comparison.json`
（sha256 `bc23f36ccf026453…`）与 `outputs/r7_71_72_round_three/paired_comparison.json`
（sha256 `f2863b9f210e8dba…`），两者都由 `e0_paired_stability.json` 记录并核对。
逐 seed delta（K，focus − baseline）：

| 时效 | 二轮 C − B | 二轮 D − B | 三轮 E − A |
| --- | --- | --- | --- |
| 48h | −0.1061 / −0.3305 / −0.2396 → **均值 −0.2254**，三 seed 同号 | −0.1062 / −0.3220 / −0.2424 → **−0.2235**，同号 | −0.3125 / −0.0198 / −0.4082 → **−0.2468**，同号 |
| 72h | −0.4823 / −0.7678 / −0.3161 → **−0.5221**，同号 | −0.4822 / −0.7505 / −0.3158 → **−0.5162**，同号 | −0.1629 / **+0.2378** / −0.9492 → 均值 −0.2915，**seed42 反号** |

**判定**：

1. **48h 那一段是稳定的。** 三次独立配对（C−B、D−B、E−A）三个 seed 全部同号，
   均值落在 −0.2235…−0.2468 K，彼此相差 11% 以内。预注册问的「0.25 K 量级」在 48h 上复现。
2. **72h 那一段不稳定，而且是预注册文字自己写错的那一格。**
   三轮 §13 与 `docs/plans/0004-r7-main-model-v2.md` 都写「t2m 48h −0.247 K、72h −0.291 K，
   **三 seed 同号**」。两个**均值**都精确复现（−0.2468 / −0.2915，脚本逐格核对 `mean_reproduced: true`）；
   但 **72h 的「三 seed 同号」为假**：seed42 = +0.2378，与 seed41/43 反号。
   −0.2915 是**反号三值的算术平均**，不是一个同号效应。
   三轮文档**自己的** §7 表（`docs/R7_71_72_ROUND_THREE.md:228`）已经把 72h 标成
   「四对**全部否**」，与本轮实测一致——§13 的措辞与 §7 的表互相矛盾，§13 那一句是错的。
3. 因此按三轮 §13 预注册的分支，**「若不稳定」那一支被触发**：72h 的 seed42 异号在两轮里都出现
   （二轮 B−A 72h seed42 = +0.8525；三轮 B−A 72h 同值；三轮 E−A 72h = +0.2378）。

**可比性限制（必须一起读）**：三对不是同一个操作。E−A 加的是时空条件模块并喂零
（+0.70% 参数），D−B 加的是池化 query 的 RW-A 读（+6.02%），C−B 加的是位置化 query 的同一个读
（+6.02%）。三者都能读成「模块在场、无位置信息」，但**不是同一个模块**。

## 6. limitations

- 两次运行各只有**一个** 2016 年冬季段、各三个 seed；这是**稳定性观察**，不是汇总估计。
- 两轮的 `protocol_sha256` 与 `model_code_sha256` 都不同，所以只比较 delta，**从不合并数字**。
- E0-1 是**重新跑前向**得到的新测量，不是对任何已公布数字的重放；被重放的是 checkpoint 与代码修订。
- correction 几何只取**第 3 轮**（协议训练深度就是 K=3）；它不是整条 K 轨迹。
- `update_energy` 是训练归一化平方量；跨变量平均只是索引。
- oracle 诊断在本设置下退化（§4），没有给出停止证据。
- 本节的两个诊断**都不读 test**，也**没有新增任何阈值**。

## 7. 未做的事

- 未训练、未租 GPU、未下载数据、未写 `data/raw|interim|processed`、未改任何 checkpoint。
- 未新增判据或阈值；未引入非零 `step_cost`；未跑第三 seed 之外的 seed。
- **未修**三轮 §13 与计划里的那句错话（已冻结的文档是证据，按 §7 的纪律只记录不回溯改写）；
  更正落在本条记录与 `docs/rules/EVIDENCE.md` 的 E-197。
- 未把 E0 的结论外推到 72h 以外的时效或 t2m 以外的变量。

## 8. 复现

```bash
# E0-1：用 d8aff68 的代码回放 12 个二轮 checkpoint（CPU，约 2 分钟）
git archive --format=tar d8aff68 | (mkdir -p /tmp/r7_e0/code_r2 && tar -x -C /tmp/r7_e0/code_r2)
.venv/bin/python scripts/study_r7_e0_correction_replay.py \
  --code-root /tmp/r7_e0/code_r2 \
  --manifest outputs/r7_m2_segment/store/manifests/val.jsonl \
  --output outputs/r7_e0_diagnostic --label r7-e0-correction-replay-round-two-v1 \
  --arm process_pooled=41=outputs/r7_71_72_round_two/seed41/training/process_pooled/update_0000400.pt \
  ... --pair process_spacetime_rwa:process_spacetime_only \
  --pair process_rwa_capacity_control:process_spacetime_only --max-samples 22

# E0-2：离线并排两个已归档的比较器产物（<1 秒）
.venv/bin/python scripts/study_r7_e0_paired_stability.py \
  --round-two outputs/r7_71_72_round_two/paired_comparison.json \
  --round-three outputs/r7_71_72_round_three/paired_comparison.json \
  --r2-pair "C-B=process_spacetime_rwa:process_spacetime_only" \
  --r2-pair "D-B=process_rwa_capacity_control:process_spacetime_only" \
  --r3-pair "E-A=process_spacetime_constant:process_pooled" \
  --out outputs/r7_e0_diagnostic/e0_paired_stability.json
```
