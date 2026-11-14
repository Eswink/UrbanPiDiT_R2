# 勘误：城市静态三臂消融的「判定卡」与 `readings.json` 相反（记录本身不变）

- **日期**：2026-11-14
- **状态**：**独立只读勘误**（`outcome_class: audit`）；**不改动**任何已冻结判据、读数或结论
- **被勘误的对象**：`dwfrun-38da4731` 运行时发布的「三臂消融判定」卡 / 完成摘要
- **权威记录**：`docs/R7_S3_URBAN_STATIC_ABLATION_RESULT.md`（`sha256 9a0bb8a9…`，未改动）
  与其引用的 `outputs/r7_s3_urban_static_ablation_20261113_attempt01/readings.json`
- **`scientific_claim: false`**、`test_read: false`、`r = 0`、`gpu_hours: 0`（只读核对）

---

## §1 矛盾是什么

`dwfrun-38da4731` 的判定卡与完成摘要逐字写：

> (a)=false (b)=false (c)=false (d 支撑)=true (e)=false (f 归因)=false

而 `readings.json` 的 `exit_conditions` 实测是：

| 判据 | 判定卡说 | **实测（权威）** |
| --- | --- | --- |
| (a) 72 h 配对 Δ 三 seed 全 ≥ 0 | false | **false**（一致） |
| (b) 24/48 h 全 ≥ 0 | false | **true** |
| (c) 6/12 h ≥ −0.005 | false | **true** |
| (e) 预检 6/6 | false | **true** |
| (f) Δ(treatment−placebo) 72 h ≥ 0.001 K | false | **true** |

**b / c / e / f 四项相反。** `verdict` 两处都是
`permanently-close-the-urban-static-input-branch`——因为 (a) 为假已足以关闭该支，
所以**结论没有被这次错误改变**，但卡上四条判据的真值是错的。

## §2 原因（实测定位）

workflow 的判定步骤用 `world.run` 跑一段内联 python，把 `exit_conditions` 打成 JSON。
该步骤的报告项是：

```json
{"probe": "exit-conditions", "exitCode": 0, "body": ""}
```

**`body` 为空**。随后脚本用 `/"a": true/`、`/"b": true/` 一类正则去匹配这串文本；
空串令所有正则都不匹配，于是每个判据都落到 `false`。而该步骤**退出码仍是 0**，
所以脚本没有走「判定读数无法解析」那条停线，而是把「没读到」当成了「读到 false」。

把同一条内联命令在会话里重跑，输出正常（2026-11-14 实测，退出码 0）：

```bash
.venv/bin/python -c '<the same inline judge script>' r7_s3_urban_static_ablation_20261113
# -> {"a": false, "attempt": "outputs/r7_s3_urban_static_ablation_20261113_attempt01",
#     "b": true, "c": true, "e": true, "f": true,
#     "verdict": "permanently-close-the-urban-static-input-branch"}
```

⇒ 空 `body` 是**该次运行内 `world.run` 的输出捕获问题**，不是判定逻辑的问题。

**未定位的部分（如实写）**：「为什么那次捕获为空」**仍未查明**。候选（内联脚本过长、
`output: {tail: 16384}` 与内联 argv 的交互、该步骤的 cwd）都是**推测**，本次没有复现，
因此**不得**写成已诊断清楚。

## §3 为什么登记产物不受污染

证据页、`R7_EVIDENCE_INDEX.jsonl` 的 `s3-urban-static-ablation-negative`、主账本行、
`campaign-state` 的 `used_gpu_h` **全部由 `readings.json` 直接算出**，与实测值一致；
判定卡是 workflow 运行时生成的展示层产物，**没有写进任何登记文件**。

所以本次勘误**不需要**改任何读数、判据、账本或结论。

## §4 一条可复现的重新判定

任何人可据此核对，而不必信任任何摘要：

```bash
.venv/bin/python - <<'PY'
import json
d = json.load(open("outputs/r7_s3_urban_static_ablation_20261113_attempt01/readings.json"))
e = d["exit_conditions"]
print("a", e["a_72h_skill_nonnegative"], e["a_per_seed_delta_treatment_minus_control"])
print("b", e["b_mid_skill_nonnegative"], e["b_per_seed_delta_24h"], e["b_per_seed_delta_48h"])
print("c", e["c_short_skill_within_tolerance"], e["c_per_seed_delta_6h"], e["c_per_seed_delta_12h"])
print("e", e["e_preflight_passed"], "f", e["f_attribution_treatment_beats_placebo"],
      e["f_per_seed_rmse_improvement_kelvin"])
print("verdict", e["verdict"])
PY
```

2026-11-14 实跑输出：

```
a False {'41': -0.0007429155556081257, '42': -0.0030985839248500735, '43': 0.0008917275044889594}
b True {'41': 0.0017037428561246237, '42': 0.006305534694526882, '43': 0.0026677710785605324} {'41': 0.005788561435134287, '42': 0.00745146162198751, '43': 0.006157606692528517}
c True {'41': 0.0033094355472518266, '42': -0.00012043681295248465, '43': 0.0025017323232352506} {'41': 0.0032533803269734207, '42': 0.005876863483042949, '43': 0.008064919019596828}
e True f True {'41': 0.04599807864248673, '42': 0.03960357255385727, '43': 0.037010694232657126}
verdict permanently-close-the-urban-static-input-branch
```

## §5 结论不变，且不能因为 (b)(c)(e)(f) 为真而改读

(a) 为假 ⇒ 按预声明**永久关闭**城市静态协变量输入支。**判据是合取**：
(b)(c)(e)(f) 为真**不构成**部分成功。而且 (f) 的归因力本就有限——placebo 与 treatment
差两件事（无空间结构**且** token 幅度 0.6566×／0.6920×），这一点在主记录 §7 已如实写明。
**本勘误只修复展示层，不改结论、不放宽任何出口。**

## §6 对后续 workflow 的硬要求

1. 判定步骤必须**先断言输出体非空**再解析；空输出一律走停线，不得落到默认值。
2. 判据真值必须从 `readings.json` **结构化读取**（`json.load` 后逐键取值），
   不得用正则去匹配自己刚打印出来的字符串。
3. 摘要里的每一项都要能被一条可复现命令重算（本条 §4 即是范式）。

编排层缺陷的登记另见 `docs/goals/s3-urban-static-ablation.md` §8.6。

## §7 影响面（如实）

- **科学结论**：无变化。(a) 为假这一点在判定卡与权威记录里一致。
- **登记产物**：无变化，未被污染。
- **读者风险**：只看过判定卡、没读证据页的人会以为 (b)(c)(e)(f) 也是假。
  本页即为该风险的补丁，并已写入证据索引。
- **GPU 成本**：0（只读核对，未重跑任何 stage）。
