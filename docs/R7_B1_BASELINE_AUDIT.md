# #64 B1 强基线名实核对与预注册判据草案，2026-09-26

Status: **DONE（核对与预注册段）**。六个基线的实现逐一核对完毕：四个神经族
（U-Net / native window / AFNO-small / generic recursion）机制**名副其实**，两个非神经
基线（persistence / train-only climatology）**实现存在但不完整**——climatology 在 D1
上有一个真实的**训练/验证泄漏**（详见 §5），且它当前只作为 ACC 的异常基准存在，
**不能**当作独立预报基线使用。`scientific_claim: false`——本页是代码与静态核查结论，
**没有启动任何 B1 训练比较**。

- 起点 SHA `e8834d5`；模型代码 digest `d9fb07f2…`（与 B0 一致，本轮未改模型）。
- GPU 消耗：**0.0 GPU-hours**（全部核查在 CPU 上完成，符合本段预算）。
- 机械证据：`outputs/r7_b1_baseline_audit/probe_evidence.json`
  （SHA256 `0bbf49572bd7edb61b19b959252b8397f98f027ff8461075f5b724cc84c2d5ec`）。
  `outputs/` 被 `.gitignore:31` 忽略，故该文件**本身不进版本控制**——它可由本页 §9
  的命令重放，且 §3/§7.2 的全部数字在本页正文中已逐条写出，不依赖该文件才能读。

## 1. 先决问题：模型工厂入口（B0 是怎么构造这四族的）

`training/r7_experiment.py:13-28` 的 `make_model(kind, config)` 是**唯一**模型工厂
（`scripts/plan_r7_comparison.py:8`、`training/r7_local_runner.py:93`、
`scripts/study_r7_b0_learnability.py:122` 都经它构造；`scripts/bench_r7_gpu_memory.py:51`
与 `scripts/bench_r7_ddp_smoke.py:111` 有各自的局部 helper，但只服务显存/DDP 基准，
不是第三套工厂）。它的契约是**三个字段**，不是两个：

| 参数 | 取值 | 含义 |
| --- | --- | --- |
| `kind` | `native` / `generic` / `process` | 模型**族** |
| `config['architecture']` | `window`（默认）/ `unet` / `convlstm` / `afno_small` | 仅 `native` 族内的**架构** |

映射表在 `r7_experiment.py:21`（native）与 `:25`（recursive）：

```python
native={'window':NativeAtmosForecaster,'unet':UNetForecaster,
        'convlstm':ConvLSTMSmallForecaster,'afno_small':AFNOSmallForecaster}
recursive={'generic':GenericRecursiveWeatherForecaster,'process':ProcessForecastCoReasoner}
```

**因此 `make_model('unet', …)` / `make_model('afno_small', …)` 报
`ValueError: unsupported model kind/architecture` 是正确行为，不是缺陷**：
`unet`/`afno_small` 是 `architecture`，必须写成 `make_model('native', {...,'architecture':'unet'})`。
`process` 族额外要求 `architecture=='window'`（`r7_experiment.py:26`）。

B0 的实际调用方式（`scripts/study_r7_b0_learnability.py:40-51` 的 `ARMS` 表 +
`:126-128`）正是三元组 `(name, kind, architecture, model_config, encoder, solver)`；
`{"architecture": architecture, **model_config}` 合并后交给 `make_model`。
**正确入口已确认，无需新建工厂，也未新建。**

## 2. 六个基线的名实核对表

参数差距以 B1 的 1–5M 档为参考；实测候选一律落在 **2.79–2.83M**（见 §3）。
「名副其实」列只回答**实现是否与名称相符**，不回答性能。

| # | 名称 | 实现位置 | 关键机制 | 名副其实？ | 与 1–5M 目标 |
| --- | --- | --- | --- | --- | --- |
| 1 | persistence | `training/r7_evaluate.py:19-22`（`class Persistence`） | `forecast = coarse_history[:,-1].clone()`，无参数、无训练 | **是**（见 §5.1） | 不适用（非神经，按 #64 原文不硬凑参数） |
| 2 | train-only seasonal/hour climatology | `data/r7_evaluation.py:77-101`（`fit_training_climatology`）+ `:103-110`（`normalized_climatology`）；技能计量在 `training/r7_climatology_skill.py:31-93` | 逐 (month, hour) 的格点均值，均值用**在线递推**（`:95`）；`kind='train-only-month-hour-grid-mean-v1'`；缺桶时 **fail-closed**（`:106-108`「no held-out fallback」） | **部分**：机制属实，但在 D1 上有**泄漏**（§5.2），且当前**不是**独立预报基线（§5.3） | 不适用（非神经） |
| 3 | U-Net | `model/r7_baselines.py:54-79`（`UNetForecaster`），卷积块 `:44-46` | 两级 down/up（`down1/down2` stride 2，`up1/up2` ConvTranspose2d）+ skip concat；`padding_mode='replicate'` + `GroupNorm`；残差小初始化头 `:49-51` | **是**：两阶段编解码+skip，明确标注为 compact regional adaptation（`docs/R7_BASELINES.md:12`），非原论文复现 | 候选 `dim=66` → **2,789,903**（−0.36% vs 2.8M） |
| 4 | native window / Swin-like | `model/weather_forecaster_r7.py:18-49`（`NativeAtmosForecaster`）→ `model/coarse_encoder.py:9-46` → `model/layers/window_attention.py:75-147`；边界 padding `model/layers/patch_grid.py:7-16` | 局部窗口注意力 `O(HW·w²)`；**交替 shifted window**（`coarse_encoder.py:24`，`shift=bool(i%2)`）；**显式有限域注意力 mask** `_shift_attention_mask`（`:26-72`），默认 `periodic_width=False`；ceil-patch 用 `replicate` 而非周期填充 | **是**：有限地理边界 mask **保留**，未照搬图像周期 wrap（见 §4.1 的反证实验） | 候选 `dim=192, depth=6` → **2,803,601**（+0.13%） |
| 5 | AFNO-small | `model/r7_baselines.py:143-161`（`AFNOSmallForecaster`）；`AFNOBlock` `:130-140`；`AFNOMixer` `:101-127` | 真 **Fourier token mixer**：`torch.fft.rfft2`（`:117`）→ **块对角复数 MLP**（`einsum` 两权 `:121,123`）→ **soft threshold** `F.softshrink(lambd=shrinkage)`（`:124`）→ `irfft2`；`shrinkage` 默认 0.01（`:103`）；FFT 全程 FP32（`:116`） | **是**：有 Fourier token mixer + soft threshold，**不是**把普通 FFT 层冒充 AFNO（见 §4.2 的反证实验） | 候选 `dim=248, depth=8, blocks=4` → **2,831,433**（+1.12%） |
| 6 | generic recursion | `model/recursive_weather_r7.py:74-129`（`GenericRecursiveWeatherForecaster`）；cell `:47-59`；条件化 `:13-31`；深监督 loss `training/r7_recursive_losses.py:7-` | 参数**共享**递归 cell（self-attn + cross-attn 到 context + FF），`latent` 参数沿 K 步复用（`:114,121`）；每步产出 draft 并**深监督**（`draft_forecasts` `:129`）；`detach_between_steps` 支持截断 BPTT（`:127-128`） | **是**（作为 generic recursive baseline）；**但**「TRM-like」是**类**，非原论文逐行复现——逐项差异表**尚无**（§6） | 候选 `dim=192, depth=4` → **2,799,202**（−0.03%） |

四族神经基线的参数**极差仅 41,530（1.49% max/min）**，在 ±5% 预注册带内且远优于它。

### 2.1 第五个 native 架构：convlstm（不在 #64 的六项内）

`ConvlSTMForecaster`（`model/r7_baselines.py:82-98`）已实现并通过同路径可构造，
但 #64 B1 的清单只列了六项，**convlstm 不在其中**。本段把它记录为「存在但非 B1 必需」，
不擅自改清单；若 B1 要带上它，候选 `dim=240` → 2,773,457（−0.95%），已实测可用。

## 3. 1–5M 档候选配置（实测，非估算）

计数方式：`make_model` 实例化后 `sum(p.numel() for p in model.parameters())`，17 通道、
`history_steps=2`。命令与完整输出在 §7。

| 基线 | 候选 config（除 `in_channels/out_channels/history_steps`） | 参数量 | vs 2.8M |
| --- | --- | --- | --- |
| U-Net | `architecture='unet', dim=66` | 2,789,903 | −0.36% |
| native window | `architecture='window', dim=192, depth=6, heads=4, window_size=4, patch_size=2, dropout=0.0` | 2,803,601 | +0.13% |
| AFNO-small | `architecture='afno_small', dim=248, patch_size=2, depth=8, blocks=4` | 2,831,433 | +1.12% |
| generic recursion | `kind='generic', architecture='window', dim=192, depth=4, heads=4, window_size=4, patch_size=2, dropout=0.0, latent_tokens=16, default_reasoning_steps=3` | 2,799,202 | −0.03% |

四个候选都已在**真实 D1 训练窗口**上跑通 forward+backward（批 2，65×65，全部参数收到
非零梯度），并在**奇数 65×65** 网格上产出 `(1,17,65,65)` 无形状失配——U-Net 的 stride-2
下采样与 AFNO 的 patch 解码都正确处理了 65 这个奇数尺寸。

## 4. 机制反证实验（不是只看名字）

### 4.1 Swin-like 有限边界：扰动单点不越过边界

对 `WindowAttentionBlock(dim=32, heads=4, ws=4, shift=True, periodic_width=False)` 在
8×8 token 网格上扰动**单个**角点，观察输出变化范围（`model/layers/window_attention.py:26-72`
的 mask 应阻止跨边界注意力）：

| 扰动位置 | 受影响 token | 其它三角是否被改动 |
| --- | --- | --- |
| (0,0) 左上 | 4/64（左上 2×2 窗口） | TR=False BL=False BR=False |
| (7,7) 右下 | 4/64（右下 2×2 窗口） | TL=False TR=False BL=False |

**结论**：影响**严格局限**在扰动点所在窗口内，未出现跨上下/左右边界的传播——有限域
mask 生效，**没有**照搬图像的周期 wrap。对照：`tests/test_window_attention.py::
test_periodic_width_keeps_longitude_wrap_but_not_latitude_wrap` 固化的是
`periodic_width=True`（经度真周期）的**相反**语义，两者都在测试内。

真实 65×65 网格：`patch_size=2` 时 ceil-pad 到 66（最后一列/行 `replicate` 复制，
`patch_grid.py:13`；`periodic_width=False` → 不是 `circular`），token 网格 33×33，
解码后 `crop_native_grid`（`:19-23`）裁回 65。窗口 4 不整除 33，故 shifted 分支的
mask 被真实触发（`window_attention.py:129` 的 `Hp!=H or Wp!=W`）。

### 4.2 AFNO：soft threshold 与 Fourier 混合是真的

对 `AFNOMixer(dim=8, blocks=2)`：

| 实验 | 结果 | 说明 |
| --- | --- | --- |
| `shrinkage=1e6` + 权重清零 | 输出 `absmax = 0.000e+00` | soft threshold 确实把频谱压到 0 |
| `shrinkage=1e6` + 随机权重 | 输出 `absmax = 0.000e+00` | 同上，与权重无关，证明阈值在谱域作用 |
| `shrinkage=0.01`（默认） | 非平凡输出（`absmax≈1.2e-2`） | 不是恒等或空操作 |

参数形状 `w1/w2=(2,4,8,8)`、`b1/b2=(2,4,8)`，即**块对角复数权重**（4 块 × 8 维），
配合 `torch.fft.rfft2` 保留**全部** `H×(W/2+1)` 频段（8×8 → 40 bins）。
`docs/R7_BASELINES.md:14-17` 已明确写明「保留全部频段，不复制 FourCastNet 特定
mode/arch/recipe」，并提示「FFT 隐含周期谱混合，区域边界测试须考虑」——**这条边界保留
在文档里，未被我抹掉**。

## 5. 两个非神经基线的实现状态核对

### 5.1 persistence：实现属实且可作独立基线

`class Persistence`（`training/r7_evaluate.py:19-22`）取 `coarse_history[:,-1]` 作为
预报，无参数、无训练、无 checkpoint。经
`evaluate_r7_local.py --manifest … --persistence`（`:13` 的互斥组）与
`scripts/rollout_r7_metric_tables.py:92-108` 的 `_evaluate_persistence` 两条路径都可用；
后者强制 `report['split']=='test'`（`:100-101`）并**断言与神经模型同 case 集**
（`:156-163` 比对 channel 顺序与 case 数）。**判定：名副其实，可直接进 B1。**

注意一个真实的读法陷阱（既有文档已记录，我复核后确认）：`docs/R7_ROLLOUT_TABLES.md:86-90`
指出跨 lead 比较 persistence 会被「不同起报时刻」混淆，故 persistence 只可作**同 lead、
同 case** 的对照。B1 必须按 (variable, lead) 成对报，不做跨 lead 排名。

### 5.2 climatology 在 D1 上有真实的训练/验证泄漏（本轮新发现）

`fit_training_climatology`（`data/r7_evaluation.py:77-101`）用**年**筛训练样本：

```python
train=set(root.attrs['split_years']['train'])                    # :83
chosen=[i for i in range(start,stop) if times[i].year in train]  # :87
```

而 D1 store 的全部 120 个时次**都在 2016 年**（`split_years={'train':[2016],...}`，
决策 0005 的 `split_mode='time_ranges'`：train=1–24 日 / val=25–27 日 / test=28–30 日，
`outputs/r7_d1_earthmover/store/cache.zarr` 的 `split_time_ranges` 属性）。于是年筛
**退化为「取全部时次」**：

| 量 | 实测 |
| --- | --- |
| 被计入的时次 | **120/120**（含 val 25–27 日、test 28–30 日） |
| 逐 (month,hour) 桶计数 | `(1,0)=(1,6)=(1,12)=(1,18)=30` |
| 正确 train-only 应为 | 24/桶（96 个时次） |

后果是 climatology **看见了自己的验证日**，把它当基线的误差**被压低**。逐通道实测
（val hour-6，纬向加权，物理单位）：

| 通道 | 单位 | 泄漏版 RMSE | 真 train-only RMSE | 抬高了基线多少 |
| --- | --- | --- | --- | --- |
| t2m | K | 22.3954 | 23.9511 | **+6.9%** |
| u10 | m s⁻¹ | 18.1824 | 17.9296 | −1.4% |
| mslp | Pa | 4179.95 | 4299.55 | **+2.9%** |
| v850 | m s⁻¹ | 38.2001 | 40.4853 | **+6.0%** |

**范围界定**：这不是全仓缺陷。跨年 store 不受影响——`outputs/r7_regional_real`（16/48 步）
与 `outputs/r7_coreasoning_v2`（40/120 步）的年筛取到的是真正的 train 年份子集。**泄漏只在
`split_mode='time_ranges'` 的单年段上发生**，即恰好是 B1 要用的 D1。

同一根因还击穿另外两条读路径（都是「按年判 split」与时间段模式不兼容）：

| 位置 | 症状（D1） |
| --- | --- |
| `data/r7_store.py:87-89` `validate_record` | val 10/10、test 10/10 抛 `ValueError: forecast window crosses split years`；train 94/94 通过 |
| `data/r7_evaluation.py:35,39,42` `ZarrRolloutDataset` | val/test 抛 `ValueError: no complete held-out rollout windows at requested horizons` |
| 连带 | `evaluate_local`（`training/r7_evaluate.py:50` 逐 record 调 `validate_record`）在 val 上直接失败 |

`validate_store`（`data/r7_store.py:31-62`）**通过**——store 本身合法。缺陷在**读者**，
不在数据。`scripts/verify_r7_d1_store.py` 的离线重放**没有**调用 `validate_record`
（grep 计数 0），它用的是自己基于 `split_time_ranges` 的时间段判定（`:93-136`），
所以发布验证全绿并**不能**覆盖这条缝隙。

**最小修正方案（不擅自实施，待授权）**：把「按年判 split」改为「按 store 声明的
split 语义判」——当 `root.attrs.get('split_mode')=='time_ranges'` 时用
`split_time_ranges` 判定窗口归属与 train-only 选取，否则保持现年逻辑不变。涉及
`r7_store.validate_record`、`r7_evaluation.fit_training_climatology`、
`r7_evaluation.ZarrRolloutDataset` 三处，**不动** `validate_store`、**不动**
`chronological_splits`，与决策 0005 的「默认年模式行为完全不变」约束一致。
`tests/test_r7_time_range_splits.py` 目前只断言 `validate_store` 与窗口计数
（`:23` 导入，全文无 `validate_record`），**该缺口未被测试覆盖**——修正需同时补测试。

### 5.3 climatology 当前不是独立的预报基线

`RolloutClimatologySkillAccumulator`（`training/r7_climatology_skill.py:31-93`）累计的是
**预报**与**气候态**各自的纬向加权 MSE，报告 `rmse_forecast` / `rmse_climatology` /
`mse_skill = 1 - MSE_f/MSE_c`（`:89-92`）。但它的**唯一**消费者是
`tests/test_r7_climatology_skill.py`——grep 全仓（排除 `legacy`）无任何 `scripts/` 或
`training/` 入口调用它，`scripts/build_r7_gpu_audit_pack.py` 也没有引用。
在生产表里，climatology 只作 **ACC 的异常基准**注入：
`training/r7_evaluate.py:108` 拟合、`:132` 反归一化、`:134` 传给 `acc.update`；而
`scripts/rollout_r7_metric_tables.py` 的 `RMSE_COLUMNS`（`:35`）**不含** climatology 行。

**判定**：机制与计量都在，但**没有一条生产路径输出 climatology 作为预报基线**。
#64 B1 要用它，需要一条把 `rmse_climatology` / `mse_skill` 写进**同一 case 表**的入口
（写盘前须先修 §5.2 的泄漏，否则基线数字偏乐观）。这是**待授权的接口补齐**，本段不写。

## 6. 外部来源的 revision / license（借用前记录）

**本段没有引入、复制或 vendor 任何外部源码**——grep `Copyright|SPDX|Licensed under`
在 `model/ data/ training/ scripts/` 下 **0 命中**，全部实现为本仓自研。下表是**若**
B1/B2 要借鉴时的现状记录，非已发生的借用：

| 项目 | HEAD revision | license | 与本仓的关系 |
| --- | --- | --- | --- |
| [NVlabs/AFNO-transformer](https://github.com/NVlabs/AFNO-transformer) | `941e924ffdb3`（2022-05-02） | **CC-BY-NC-SA-4.0（非商业）** | 只**核对概念**（Fourier token mixer / soft threshold）。**NC 许可意味着不得复制其代码进本仓** |
| [NVlabs/FourCastNet](https://github.com/NVlabs/FourCastNet) | `93360c1720a9`（2023-01-13） | BSD-3-Clause | 只借鉴归一化思路；小型区域重训须标 **FourCastNet-style / AFNO-small**（`model/r7_baselines.py:144` 已写「not the FourCastNet architecture/recipe」） |
| [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer) | `f82860bfb522`（2024-07-15） | MIT | 窗口注意力参考；本仓**保留有限地理边界 mask**，未照搬周期 wrap（§4.1 已证） |
| [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) | `c01103738605`（2026-04-01） | MIT | 概念参考（shared recursive + 深监督）；**逐项差异表尚不存在**（见下） |

**TRM 差异表的缺口（如实记录）**：#64 要求「先做与现有实现的逐项差异表，不称当前网络是
原论文逐行复现」。现状是 `docs/R7_BUDGET_PARITY.md:3-5` 只称其为
"generic TRM-like recursive"、`model/recursive_weather_r7.py:75` 的 docstring 写
"Generic parameter-shared recursive baseline without process semantics"——**命名上已避免
冒称复现**，但 #64 明文要求的**逐项差异表**（x/y/z 与深监督语义对照）**尚未产出**。
这是 B1/B2 前的**待办**，本段不擅自补写（需要读上游代码，超出「静态核查本仓」范围）。

## 7. B1 预注册判据草案（**草案，尚未生效**）

> 本节是 B1 训练**开跑前**要冻结进 `protocol.json` 的判据。按 #64 原文与
> `docs/R7_B0_LEARNABILITY.md` 的形状撰写。**它尚未被任何 run 采用**；正式采用前
> 需先修 §5.2 的泄漏并取得数据范围授权（§8 决策点 D-1）。

### 7.1 参数量约束（±5% 为**候选**）

- 预注册带：四族神经基线各候选落在 **±5%** 的**同一名义容量**内。本段取名义 2.8M，
  实测 2,789,903–2,831,433（极差 1.49%），**远严于** ±5%。
- 参数量**必须实测**（实例化后 `numel` 求和），不得用纸面公式；每个 arm 的
  `parameters` 写进协议并在结果中回显。
- **非神经基线（persistence / climatology）不纳入参数带**——#64 原文明确
  「非神经基线不用硬凑参数」。它们以 **0 可训练参数**记录，不得为凑数加参数。
- §2.1 的 convlstm 不在 #64 六项内，默认不进 B1；若要进须先修 #64 清单。

### 7.2 FLOPs 记录口径（沿用本仓既有约定，不另立）

**必须沿用 `training/r7_budget_audit.py` 的既有约定**（该模块 docstring `:10-21` 与
`count_forward_flops` 的 `:47-52` 就是本仓的 FLOP 计量标准），不新增第二套：

- 计数器：`torch.utils.flop_counter.FlopCounterMode`（`:58`），统计 linear/conv/matmul 与
  经 aten 分派的注意力 matmul；**逐元素与归一化不计**（`:13-15`）。
- **必须在 `torch.enable_grad()` 下计数**（`r7_budget_audit.py:128`）：计数器注册的 hook
  在这些模型上于 `torch.no_grad()` 下会失败（`:18-19`、`:50-51`）。
- **不要用参数 hook**：本仓注意力走 `F.scaled_dot_product_attention`
  （`model/layers/sdpa.py:17`、`model/layers/window_attention.py:135`），**SDPA 无参数**，
  参数 hook 会漏掉全部注意力 matmul 且不报错——静默低估。
- **forward 与 backward 都要记录实测值，并声明报的是哪个量。** 既有文档
  `docs/R7_BUDGET_PARITY.md:39` 写「Backward is not counted. It is roughly 2x forward …
  so only forward is reported」。**「only forward」是这句话的关键**：一个只报 forward 的
  数字会让训练成本看起来约为实际的三分之一。本段实测：

| 基线 | forward FLOPs | forward+backward FLOPs | `fwd+bwd`/`fwd`（总比） |
| --- | --- | --- | --- |
| U-Net | 8,893,199,172 | 26,338,284,984 | 2.962 |
| native window | 12,985,192,320 | 38,841,832,704 | 2.991 |
| AFNO-small | 11,005,050,096 | 32,868,230,624 | 2.987 |
| generic (K=3) | 12,843,758,976 | 38,417,532,672 | 2.991 |

**结论**：`fwd+bwd ≈ 2.96–2.99 × fwd`，四族一致。既有「roughly 2x」作为**增量**
backward 的启发式在本批上仍近似成立（实测增量比 1.96–1.99，最大偏差 −1.9%），
**但它不是** `fwd+bwd`/`fwd` 的比值——二者相差近 1.5 倍，混用会把预算算错约 50%。
B1 的协议里必须写清报的是 `fwd` 还是 `fwd+bwd`，并直接给实测值而不是让人去乘。
- 批次与形状必须随 FLOP 数一起记录（本段：批 2 × `[2,2,17,65,65]`，真实 D1 窗口 0–1）。
  单条 FLOP 数脱离形状不可比。

### 7.3 wall time 与例数的记录口径

- **wall time 与例数分开记，各自带范围声明**。沿用
  `training/r7_local_runner.py:134` 的既有字段：`elapsed_seconds`、
  `updates_this_run`、逐更新 `losses[].samples`（`:128`），并保留其 `:138` 的
  `note='wall time includes batch reads; CUDA peaks cover this bounded run, not a
  forecast-skill benchmark'` 范围声明。
- 每条运行记录四元组：**(arm, updates, samples_seen, elapsed_seconds)**，外加
  `peak_allocated_bytes`。**例数**定义为**前向见到的样本数**（含 accumulation 内所有
  微批），不是 update 数 × batch_size 的想当然——`losses[].samples` 由 `:128` 实际累加。
- wall time **必须声明是否含数据读取**（本仓含，`:138`）。GPU 与 CPU 不可混记；
  本段落盘的是 **CPU 前向墙钟**，两轮 5 次测量的中位数约：U-Net 32–33 ms、
  native window 42–48 ms、AFNO-small 77–79 ms、generic(K=3) 52 ms（65×65、批 2）。
  **该数在共享 CPU 上逐轮波动数 ms**，只有数量级与族间比例可读，**不得**当作 3090
  训练速度，也不得用于跨机比较。
- 计时前的 `torch.cuda.synchronize()` 必须保留（`:104,131`），否则异步执行会把墙钟
  记成提交耗时；`reset_peak_memory_stats`（`:105`）同理须在计时前。

### 7.4 「同更新 ≠ 同算力」（显式声明，必须进协议）

**同样 200 次 optimizer update 不代表同样算力，B1 不得据此声称公平。** 三条实测理由：

1. 本段四族在**同一批、同一更新数**下 forward FLOPs 相差 **1.46×**
   （8.89e9 U-Net → 1.30e10 window），forward+backward 相差 1.47×。**参数带对齐不蕴含
   算力对齐。**
2. generic recursion 的算力随 **K** 线性增长（每次 cell 调用 ≈ 一次 backbone 量级），
   且 K 是**推理深度**参数；比较时必须声明 K，并记录
   `cumulative_reasoning_steps`（`model/r7_rollout.py:30` 已注明该量
   "NOT latency/FLOPs"——**不得**拿它当算力证据）。
3. 既有文档已经踩过这个坑并留了记录：`training/r7_extended_control.py:138` 写
   "Same optimizer updates do not match FLOPs; no adaptive speedup is measured"。

**因此协议中必须同时给出**：参数量表、forward/bwd FLOPs 表、wall time 表、例数，
并对每条比较显式写明「本比较对齐的是 X，未对齐的是 Y」。只对齐参数或只对齐更新数的
比较，结论句里必须写明该限制。

### 7.5 数据与流程约束（沿用 #64 原文，不新增）

- 相同连续区域训练/验证数据、seed、抽样顺序、归一化、目标变量与已知时空特征。
- 每个 arm 的训练/验证每通道曲线、固定 checkpoint、样本身份、loss 定义、seed 维度全部保留。
- 失败与 NaN/OOM **完整保留**（B0 已示范：失败目录 `outputs/r7_b0_learnability/` 原样留着）。
- 预算是 #64 建议的每 arm ≤30 min、首批总 ≤4 GPU-hours；**据实调整前先报告实测 step 时间**。
- 所有产物 `scientific_claim: false`；「强基线可用」与「ours 胜出」是**不同状态**。

## 8. 需要用户决策的分叉（本段不擅自推进）

| 编号 | 分叉 | 选项与成本 | 建议 |
| --- | --- | --- | --- |
| **D-1** | **数据范围**：B1 是否只用 D1（2016-01，94/10/10 窗口，全部 January）？ | (a) 仅 D1：零新增网络/授权成本，但**只有 1 月**，climatology 只剩 4 个 (month,hour) 桶，季节技巧无法测；(b) 扩到 D2 多季节/全年：需 #63 的 D2 授权与新下载预算，成本另估 | 若目标是「小规模强基线可用」而非季节技巧，(a) 够；但需在协议里写明「仅 1 月、季度外推无效」。**选择权在用户。** |
| **D-2** | **§5.2 泄漏的修法**：是否授权按 §5.2 最小方案修三处读者（含补测试）？ | 修：改 `r7_store.validate_record` + `r7_evaluation` 两函数，加时间段分支 + 测试，估计**零 GPU**、低风险；(b) 不修：B1 的 climatology 数字**已知偏乐观**（t2m +6.9%），必须在报告里带此限制 | 建议修——否则 B1 的气候态对照不可信。但改动触及数据契约读者，需明确授权。 |
| **D-3** | **climatology 独立基线入口**：是否授权补一条把它写进同 case 表的路径（§5.3）？ | 补：新入口 + 测试，零 GPU；(b) 不补：B1 只能把 climatology 当 ACC 异常基准，**不能**算 MSE skill | 建议补；顺序应为 D-2 先修、D-3 后补，否则写盘的是泄漏版数字。 |
| **D-4** | **TRM 逐项差异表**（§6）：是否授权本段之外补做？ | 需读上游 `c01103738605` 代码（MIT，可读），产出差异表；零 GPU | 建议在 B1 训练前补，属 #64 明文要求。 |

以上四项**均未实施**。本段在 D-1..D-4 未决时**不启动** B1 正式训练，符合 #64
「先有预注册判据」的顺序要求。

## 9. 实际执行的验证与计数

下表命令均在本段实跑。可复现的探针片段（参数量、FLOPs、墙钟、边界与阈值反证）见 §9.1。

| 命令 | 结果 |
| --- | --- |
| `python tools/check_conventions.py` | `blocking rules=34 failing=0`，report-only hits=0 |
| `python -m pytest --collect-only -q` | `980 tests collected`（起点 977 passed/3 skipped 的集合未变；本段未加/删测试） |
| 参数量实测（§3，`make_model` + `numel`） | 4 个候选：2,789,903 / 2,803,601 / 2,831,433 / 2,799,202 |
| FLOPs 实测（§7.2，`enable_grad` + `FlopCounterMode`） | 4 族 forward 与 forward+backward 各 1 次，见 §7.2 表；`fwd+bwd`/`fwd` = 2.96–2.99 |
| 既有「roughly 2x」启发式的核对 | 增量 backward/forward 实测 1.96–1.99，作为**增量**估计成立（偏差 −1.9%）；但它不等于 `fwd+bwd`/`fwd`（≈2.97），混用会错约 50% |
| 真实 D1 forward+backward（批 2，65×65） | 4/4 通过，参数全部收到非零梯度 |
| 奇数 65×65 形状 | U-Net / AFNO 均输出 `(1,17,65,65)` |
| 边界反证（§4.1） | 扰动单点仅影响 4/64 token，无跨边界传播 |
| soft threshold 反证（§4.2） | `shrinkage=1e6` → 输出恒为 0（两种权重） |
| D1 读路径（§5.2） | `validate_record`：train 94/94 过，val 0/10、test 0/10 失败；`ZarrRolloutDataset` val/test 失败 |
| climatology 泄漏量化（§5.2） | 计入 120/120 时次（应 96）；t2m 基线误差被压低 6.9% |
| 跨年 store 对照 | `r7_regional_real` 16/48、`r7_coreasoning_v2` 40/120——年筛正确，无泄漏 |
| 外部源码 vendor 检查 | `Copyright|SPDX|Licensed under` 在活跃区 **0 命中** |
| GPU 消耗 | **0.0 GPU-hours**（`probe_evidence.json` 的 `gpu_hours_used: 0.0`） |

**未做的事**：未启动 B1 正式训练比较；未训练 15–20M（B3）；未做 B2 多种子；未做 #65 消融；
未跑 17 条 tag-gated workflow；未合并/release/force push；未改 `data/raw|interim|processed`
与任何归档；**未修 §5.2 的缺陷**（待 D-2 授权）；未新建第三套模型工厂。

### 9.1 可重放的探针片段

以下片段即本段实际运行的代码（`.venv/bin/python`，仓库根，全部 CPU）。它们不写任何数据文件，
除最后一段外都会打印到 stdout。

参数量与 forward+backward（§3）：

```python
from training.r7_experiment import make_model
from model.r7_halting import forecast_inputs
from data.r7_zarr_dataset import ZarrAtmosWindowDataset
import torch

C = 17
ds = ZarrAtmosWindowDataset("outputs/r7_d1_earthmover/store/manifests/train.jsonl")
b = torch.utils.data.default_collate([ds[0], ds[1]])
CASES = {
    "unet": ("native", {"architecture": "unet", "dim": 66}, False),
    "native_window": ("native", {"architecture": "window", "dim": 192, "depth": 6,
                                 "heads": 4, "window_size": 4, "patch_size": 2, "dropout": 0.0}, False),
    "afno_small": ("native", {"architecture": "afno_small", "dim": 248, "patch_size": 2,
                              "depth": 8, "blocks": 4}, False),
    "generic": ("generic", {"architecture": "window", "dim": 192, "depth": 4, "heads": 4,
                            "window_size": 4, "patch_size": 2, "dropout": 0.0,
                            "latent_tokens": 16, "default_reasoning_steps": 3}, True),
}
for name, (kind, cfg, recursive) in CASES.items():
    m = make_model(kind, {"in_channels": C, "out_channels": C, "history_steps": 2, **cfg})
    print(name, sum(p.numel() for p in m.parameters()))
```

FLOPs（§7.2；**必须** `enable_grad`，否则 hook 失败）：

```python
from torch.utils.flop_counter import FlopCounterMode
m.train(); m.zero_grad(); inp = forecast_inputs(b)
run = (lambda: m(inp, reasoning_steps=3)) if recursive else (lambda: m(inp))
with torch.enable_grad():
    with FlopCounterMode(display=False) as c:
        run()
    fwd = int(c.get_total_flops())
with torch.enable_grad():
    with FlopCounterMode(display=False) as c2:
        out = run(); out.forecast.square().mean().backward()
    print("fwd", fwd, "fwd+bwd", int(c2.get_total_flops()))
```

Swin-like 有限边界反证（§4.1）——扰动单点，看影响是否越过窗口边界：

```python
from model.layers.window_attention import WindowAttentionBlock
blk = WindowAttentionBlock(32, 4, 4, 4.0, 0.0, shift=True, periodic_width=False).eval()
x = torch.randn(1, 64, 32)
y = blk(x, (8, 8))
xp = x.clone(); xp[0, 0] += 10.0            # 扰动左上角 token
print((blk(xp, (8, 8)) - y).abs().sum(-1)[0].view(8, 8).gt(1e-6).int())
# 期望：只有左上 2x2 为 1，其余为 0 -> 未跨有限边界
```

AFNO soft threshold 反证（§4.2）——阈值极大时谱被压平：

```python
from model.r7_baselines import AFNOMixer
mx = AFNOMixer(8, blocks=2, shrinkage=1e6).eval()
print(mx(torch.randn(1, 8, 8, 8)).abs().max())   # 期望 0.0
```

D1 读路径与 climatology 泄漏（§5.2）：

```python
from data.r7_store import validate_store, validate_record
from data.r7_evaluation import fit_training_climatology, ZarrRolloutDataset
import json, zarr
from pathlib import Path

root = zarr.open_group("outputs/r7_d1_earthmover/store/cache.zarr", mode="r")
validate_store(root)                                    # 通过
man = Path("outputs/r7_d1_earthmover/store/manifests")

def passes(record):
    try:
        validate_record(root, record)
    except ValueError:
        return False
    return True

for split in ("train", "val", "test"):
    recs = [json.loads(l) for l in (man / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if l]
    print(split, sum(1 for r in recs if passes(r)), "/", len(recs))
print(sorted(fit_training_climatology("outputs/r7_d1_earthmover/store/cache.zarr")["counts"].items()))
# 期望：train 94/94，val 0/10，test 0/10；桶计数 30 而非 24 -> 年筛退化为「全取」
```

`ZarrRolloutDataset("outputs/r7_d1_earthmover/store/cache.zarr", split="val",
lead_hours=(6,), history_steps=2, step_hours=6)` 在本 store 上抛
`ValueError: no complete held-out rollout windows at requested horizons`。

## 10. 判定与下一项

- **本段判据满足**：六个基线的名实核对表 + 1–5M 候选配置 + B1 预注册判据草案已落盘；
  四族神经基线机制经反证实验证实名副其实；两个非神经基线的实现状态如实记录（一个可用、
  一个不完整且带泄漏）；分叉已列出选项与成本。
- **不是**：任何基线间的技巧比较、任何 B1 结果、任何参数/算力结论。
- **下一项**：取决于 §8 的授权。若 D-1 选 (a) 且 D-2/D-3 授权，则下一项是
  #64 B1 正式训练；**第一个具体动作**是修 `r7_store.validate_record` 的 split 判定
  （加 `split_mode=='time_ranges'` 分支）并补一个「时间段 store 的 val 窗口能通过
  `validate_record`」的测试——因为在 D1 上**当前连 val 集都读不出来**，B1 无从开跑。
