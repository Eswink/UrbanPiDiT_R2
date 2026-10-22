# S3：全年覆盖单因素轮（训练月份 4 → 12，年份固定 2017–2021，65×65）——结果

> **状态：candidate。** **候选臂 F（12 个月）越过预声明出口**，提请独立复核/验收。
>
> 唯一因子：训练清单覆盖的**月份数**（控制 N = Jan/Apr/Jul/Oct，候选 F = 12 个月），年份固定
> 2017–2021。两臂共用同一全天 store、同一全年归一化、同一模型 spec、同一迁移父状态、同 2400
> updates、同 K=4/depth=4/dim=192、同注册物理权重、同 seed、同 2022 全年 val、同 48 桶
> train-only 气候态分母。**不改任何模型代码**；本轮的候选臂与对照臂在同一进程、同一修订、
> 同一父状态下配对训练。
>
> `scientific_claim: false`；`test_read: false`；确认消耗 r=0；S4 未启动。
>
> 协议（prepare 后 `protocol.json`，即本记录登记的 `protocol_sha256`）`ee532b3c3194e5ffd78cc4727af856a983ea4dcd160db962ebdcb589e617ef26`；
> code commit `16ce94e57db3441607ca1db39cbc884121dbc00d`；`code.zip` sha256 `f980713d4c3ce86646ed01975db952377a9d93eee304df7c84e0cd0ebe4a4ea1`。
>
> **65×65 限制**：FY 源是原生 65×65，仓库没有 65→129 上采样契约，故本轮结论**只在 65×65 上
> 成立**，不得读成宽实例结论。**桥 (e) 不是科学判定**：控制臂与已登记窄 v3 四个月臂
> （`record:s3-v3-budget-dose`，1600 updates）的归一化、case 集与更新预算都不同，且现在
> 还多一层模型修订差异（决策 0052），只能用于发现身份/工程错误。

## 1. 结论摘要

**预声明出口（冻结后不放宽，决策 0050 第 5 条 / 0052 第 3 条）：**

| 条件 | 内容 | 结果 |
| --- | --- | --- |
| (a) | 候选 F 的 72 h t2m skill 三 seed 全 >0 | **true**（0.012121 / 0.032076 / 0.064427） |
| (b) | 同 seed F−N 的 72 h ΔRMSE 三 seed 全 <0 | **true**（-0.060269 / -0.065575 / -0.085518） |
| (c) | 候选 F 的 48 h t2m skill 三 seed 全 >0 | **true**（0.213089 / 0.215091 / 0.263743） |
| (d) | 6/12/24 h 的 seed 均值 ΔRMSE ≤0 | **true**（-0.015227 / -0.012738 / -0.016615） |
| (e) | 控制 N 与已登记窄 v3 臂的桥（**非科学**） | **false** |

⇒ `all = true`、`verdict = advance-to-independent-review`。

**下一步**：全真 ⇒ 只提请独立复核/验收，**不**自动进 S4，也不宣布科学通过。

## 2. 三 seed 主读数（65×65 整张网格，`full`，4225 点）

t2m RMSE（K）与 skill = 1 − MSE_model/MSE_climatology（分母为 48 桶 train-only 气候态）：

| seed | 臂 | 6 h | 12 h | 24 h | 48 h | 72 h |
| --- | --- | --- | --- | --- | --- | --- |
| 41 | control | 2.300536 | 2.888807 | 2.700879 | 3.490223 | 3.933658 |
| 41 | treatment | 2.274618 | 2.854629 | 2.663867 | 3.454905 | 3.873388 |
| 42 | control | 2.343398 | 2.918988 | 2.713109 | 3.466412 | 3.899643 |
| 42 | treatment | 2.330258 | 2.917406 | 2.715391 | 3.450508 | 3.834069 |
| 43 | control | 2.260776 | 2.833279 | 2.634289 | 3.390152 | 3.854968 |
| 43 | treatment | 2.254154 | 2.830826 | 2.619172 | 3.341859 | 3.769450 |

skill：

| seed | 臂 | 6 h | 12 h | 24 h | 48 h | 72 h |
| --- | --- | --- | --- | --- | --- | --- |
| 41 | control | 0.649863 | 0.448154 | 0.518101 | 0.196918 | -0.018861 |
| 41 | treatment | 0.657708 | 0.461135 | 0.531218 | 0.213089 | 0.012121 |
| 42 | control | 0.636694 | 0.436563 | 0.513727 | 0.207838 | -0.001317 |
| 42 | treatment | 0.640757 | 0.437174 | 0.512908 | 0.215091 | 0.032076 |
| 43 | control | 0.661861 | 0.469166 | 0.541570 | 0.242309 | 0.021495 |
| 43 | treatment | 0.663839 | 0.470084 | 0.546817 | 0.263743 | 0.064427 |

同 seed F − N 的 ΔRMSE（K；**负 = 候选更优**）：

| seed | 6 h | 12 h | 24 h | 48 h | 72 h |
| --- | --- | --- | --- | --- | --- |
| 41 | -0.025918 | -0.034179 | -0.037012 | -0.035318 | -0.060269 |
| 42 | -0.013140 | -0.001582 | 0.002282 | -0.015904 | -0.065575 |
| 43 | -0.006621 | -0.002453 | -0.015117 | -0.048293 | -0.085518 |
| seed 均值 | -0.015227 | -0.012738 | -0.016615 | -0.033172 | -0.070454 |

## 3. 单因子保真（两臂唯一允许的差异）

| 项 | 值 |
| --- | --- |
| 控制 N train manifest | `/data/esw/UrbanPiDiT_R2/outputs/r7_s3_fy_annual_20261021_attempt01/store/manifests_control_h72/train.jsonl` |
| 候选 F train manifest | `/data/esw/UrbanPiDiT_R2/outputs/r7_s3_fy_annual_20261021_attempt01/store/manifests_treatment/train.jsonl` |
| N train data identity | `04693129781a234e662daaf2b379c3e4a53c50ca8ba016a73c4478ef12b7e3c1` |
| F train data identity | `c17494ab63d54855697567f1195a613ca61e32cfcc3db22d709a7ff00dc17422` |
| N 完整 72 h 窗 | 2140（清单 2140 条，排除 0） |
| F 完整 72 h 窗 | 7291（清单 7302 条，排除 11） |
| 模型 spec digest（两臂相同） | `d69e33cba50693e31067d76336253ce633fbf9c53fd8b07899638301e5245796` |
| 迁移父 | `/data/esw/UrbanPiDiT_R2/outputs/r7_s3_tendency_input_parent_migrated_20261021_attempt02/seed41/parent_update_0001600.pt` |
| 父 checkpoint sha256（seed41） | `fb1e4d6522d32db1cc078a7590dd66c41f479da49ff58f2c6c525f8dfceca387` |

两臂的模型 spec、父状态、updates/LR/warmup/clip、物理权重、seed、val 清单与气候态分母逐字相同；
`require_single_factor()` 断言差异集合恰为
`{train_manifest, train_data_identity, train_windows}`。**无新增参数**，故无参数量混杂。

## 4. 父迁移与模型修订（决策 0052）

本轮改绑到当前修订，父迁移复用 tendency 轮在 `60de3b2d…` 下产出的那一份。
`state_equality_evidence()` 在 freeze 时读两份 receipt 逐 seed 断言：源 checkpoint 相同、
tensor `state_digest` 相同。证据（协议内 `model_only_migration.state_equality_evidence`）：

```
{
 "current_receipt": "/data/esw/UrbanPiDiT_R2/outputs/r7_s3_tendency_input_parent_migrated_20261021_attempt02/migration_receipt.json",
 "current_receipt_sha256": "73c019ddb5d1c6d2bc551bcd60cd321ca29d968dc956455b17e0262074c21112",
 "attempt04_receipt": "/data/esw/UrbanPiDiT_R2/outputs/r7_s3_rollout_dose_parent_migrated_20261015_attempt04/migration_receipt.json",
 "attempt04_receipt_sha256": "8b7e91e111985627d8e270d656af32070c1df3f7b0b2a1898774879383f078dc",
 "equal_state_per_seed": {
  "41": {
   "source_checkpoint_sha256": "637857c5d5cc8b3c9e33448cae4dd1b53aed5f525639d6911f03d7420492bdac",
   "state_digest": "7c416ccb4054cf7e077788a43b02917e6a2dcf2503058152a7f0ab4b0ab70707",
   "current_migrated_sha256": "fb1e4d6522d32db1cc078a7590dd66c41f479da49ff58f2c6c525f8dfceca387",
   "attempt04_migrated_sha256": "f9f12e063c30cb0b471def01b67d7333d5aae95ecfba1ff0feb824a2a6f9ac45"
  },
  "42": {
   "source_checkpoint_sha256": "8f7369fbbb3983b9e838d93a2e35d608871b01b580686a13581181950849007e",
   "state_digest": "857d880466f76500ed74c534e757732ae9eff4c2b3f0bbc0fa74b895fcae3623",
   "current_migrated_sha256": "d97ea802a24e62d1457854aeb12cf29b83f029fdca4f9903f19117f3585f3d69",
   "attempt04_migrated_sha256": "8d4e05ed5baa285113db353eede34e883812092085b8f86716fbee93fb13770f"
  },
  "43": {
   "source_checkpoint_sha256": "08cc7db9629fca86f3634c0b102a206d292f09a5370a93d4a89d9477c6b50252",
   "state_digest": "61fdac58b7bfacf828e1eae4968817b1cbd46616442c6d16ac0fbbb63e294e1b",
   "current_migrated_sha256": "8aa0fac68609e0502b4c5292d0b6067e59153f74090055e28d08d2f3ac964af5",
   "attempt04_migrated_sha256": "5af64655402cb16f0148b692ef7a84eb8934230cf43ed6ac9306501a7756cafa"
  }
 },
 "claim": "the current-revision migration carries the same v3-BD 1600 parent state that decision 0050 froze; only the embedded code identity differs, so the migrated files' own SHA256 values differ while the tensor digests do not",
 "scientific_claim": false
}
```

## 5. 桥 (e)（**非科学判定**）

| lead | 控制 N 的 seed 均值 | 已登记窄 v3 臂均值 | 差 | 带宽 | 通过 |
| --- | --- | --- | --- | --- | --- |
| 6 h | 2.301570 | 2.247799 | 0.053771 | 0.112390 | true |
| 12 h | 2.880358 | 2.882974 | -0.002616 | 0.144149 | true |
| 24 h | 2.682759 | 3.055650 | -0.372891 | 0.333713 | false |
| 48 h | 3.448929 | 4.698818 | -1.249890 | 0.983647 | false |
| 72 h | 3.896090 | 6.132720 | -2.236630 | 2.149025 | false |

整桥 `passed = false`。normalization, val cases and update budget differ; a failure is identity/engineering unresolved, not a scientific verdict

**怎么读这个失败（必须照此）**：桥在 24/48/72 h 超出带宽，方向是**本轮的 4 个月控制臂比已登记窄 v3
臂更好**（−0.373 / −1.250 / −2.237 K）。这不是科学判定，理由已在冻结时写明并与实际一致：

1. **更新预算不同**：本轮 2,400 updates，登记臂 1,600 updates——差距在长 lead 上最大，正好是本桥
   超差最大的地方；
2. **归一化不同**：本轮两臂共用**全年** train 拟合的归一化，登记臂用四个月归一化；
3. **val case 集不同**：登记臂在 2022 **四个月** val 上评分，本轮在 2022 **全年** val 上评分，
   两者不是同一批病例；
4. **模型修订不同**（决策 0052）：登记臂是 `3fe747b2…` 下的读数。

因此按预声明记 `identity/engineering unresolved`，**不**进入 (a)–(d)。

**身份另行单独核对通过**（这是桥本该发现的那类错误）：控制臂 train data identity `04693129…`、
候选臂 `c17494ab…`、val identity `94a0e509…`，与决策 0050/brief 冻结值逐字相同；控制清单
2,140 条全部可用、0 排除，候选清单 7,302 条中 7,291 可用、11 条年末排除；每臂每 lead 的 val
cohort 与 48 桶气候态由 `_check_provenance` 逐条断言通过。

## 6. 成本与预算

| 项 | 值 |
| --- | --- |
| planned_seconds | 43200.0 |
| per_seed_seconds（软预算） | 21600.0 |
| hard_cap_seconds | 86400.0 |
| 每 seed 进程时长 | 41: 17888.1s / 42: 18042.7s / 43: 17501.8s |
| 每 seed 软超 | 41: 0.0s / 42: 0.0s / 43: 0.0s |
| 每 seed 硬超 | 41: 0.0s / 42: 0.0s / 43: 0.0s |
| process 合计 | 55246.814 s = 15.346337 GPU-h |
| 墙钟 | 38372.695 s = 10.6591 h |
| 磁盘 | 29994117247 B = 27.93 GiB |
| 峰值保留显存（六次训练最大值） | 2436890624 B |
| 网络字节 | 0（R-028 离线；worker 首行 `deny_network()`） |

## 7. 限制（逐字来自协议）

- development screening on the 65x65 instance only (2017-2021 train / 2022 val, 17 channels); the 129x129 wide instance is a different store and is not covered
- the FY source has no 2023: the test split is four months and stays sealed
- both arms share the full-year 2017-2021 normalization, so the four-month control is not byte-comparable with the registered four-month narrow arm
- the budget is fixed at 2400 updates: the treatment sees about one third of an epoch while the control sees about one, so a positive result prices data diversity at this budget, not data volume at every budget
- the external bridge to s3-v3-budget-dose differs in normalization, val cases and update budget; it is an identity/engineering check, never a scientific verdict
- three seeds are consistency evidence, not a significance test
- GPU runs are co-resident; latency/memory observations include neighbor load
- validation split only; the test split stays sealed until a later preregistered read

## 7b. CI 与提交（如实披露）

本轮共 **5 个 commit**，其中前两个被 CI 判红并已修复，最后一个（本记录所在的树）为绿：

| commit | 内容 | CI |
| --- | --- | --- |
| `3312000` | 实现：改绑当前修订（决策 0052）+ 惰性开关断言 + 父迁移复用 | `R7 CPU CI` 39735871063 success |
| `27cfc39` | 修 `manifest_path()`：控制臂误读未过滤清单 | `R7 CPU CI` 39746734336 success |
| `16ce94e` | 预算更正（决策 0053）+ `default_runner` 超时结构化 | `R7 CPU CI` 39775347018 success |
| `fdc7873` | 登记本轮证据（本页 + index record + brief + 账本 + §8） | `R7 CPU CI` 39870014546 **failure** |
| `9c8fbce` | 去掉被截断的 JSON 块（`git show --check` 报本页 120 行行尾空白） | `R7 CPU CI` 39871493608 success |

**失败原因如实记录**：`fdc7873` 在 CI 的 `Compile active modules and check whitespace` 步失败——
本页原先把父迁移的 `state_equality_evidence` JSON **按 1800 字符截断**，切断了一个字符串导致该行
以空格结尾，`git show --check` 判行尾空白。`9c8fbce` 改为输出**完整** JSON 块并同步刷新证据 digest、
index record 与 canonical brief；`git show --check` 本地复跑 exit 0。

**两次失败 attempt 与冻结协议的对应**：`attempt01` 协议 `e2750d58…`（archive 被
`validate_protocol` 拦下）、`attempt02` 协议 `23235448…`（每 seed 预算过低被调度器杀掉）、
本轮读数所用 `attempt03` 协议 `ee532b3c…`。三次 attempt 全部原样保留、全额计费。

## 8. 未做的事与下一步

- 未读 `test.jsonl`（2023 只有四个月块，仍封存），`test_read: false`，确认消耗 r=0；
- 未新增下载、未建 129×129 全年源、未碰 `data/raw|interim|processed`；
- 未与任何已关轴（容量、H-A…H-D、dense、final-only、空间梯度、EMA、zero-base decoder、
  gradient surgery、PE/scale/typed/ordering/anchor/rollout-dose、显式 tendency 输入、SAM）组合；
- 未启动 S4；未宣布科学通过、未称泛化/SOTA、未自宣最终 goal 完成。

<!-- record: s3-fy-annual-coverage-result -->
