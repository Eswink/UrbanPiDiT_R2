# S3：训练期 sharpness-aware minimization（单因素优化更新规则）——结果

> **状态：negative / not-candidate。预声明出口 (a)、(b)、(c)、(d) 全部未满足，(e) 通过，按预声明永久关闭「训练期 SAM 更新规则」这一支。**
> SAM 扰动**真实发生**且不是 null/inert：三 seed 各 2400 次更新**全部**非零扰动（`zero_perturbation_updates` = 0/0/0），扰动范数 min/max 在 0.049999997…0.050000003 之间，参数逐位还原（`restore_residual_max` = 0.0），扰动点 loss 平均升高 0.0475…1.8326。但物理读数是**一致的负结果**：候选在 5 个 lead × 3 seed 的 15 个配对 ΔRMSE 上**全部更差**（+0.0963…+0.2229 K），72 h skill 三 seed 全为负（-0.267285 / -0.351110 / -0.227842）。
>
> `scientific_claim: false`；`test_read: false`；确认消耗 r=0；S4 未启动。
>
> 协议（prepare 后 `protocol.json`，即本记录登记的 `protocol_sha256`）`b7ae03e16afde8c36c16a549cc24897cc04b4e52fb0bce56c85ee5d93f3f2680`；freeze 阶段 `preparation_protocol.json` 的 digest 为 `d6607425aba710cf62a414aeef1789ebe6053494dd07b98866afece901469255`；`code.zip` sha256 `fd5723776e3bc11b3f3f8486453d56e285c68f3ef30a9c2af49a73908606bb30`（comment `23f9af32cc1c2f3b2427d1c379ed9805b9cc044b`）；
> 实验代码 commit `23f9af32cc1c2f3b2427d1c379ed9805b9cc044b`（`R7 CPU CI` **39666233779** completed/success）。
>
> **比较口径的重要限制**：候选是新训练代码（`training_code_sha256 a03f9ba6…`，带 SAM 因子）的三 seed 独立轨迹，控制是 pinned 旧代码 `s3-anchor-isolation-result` 的 control 臂读数（未重训，`model_code_sha256 3fe747b2…`、`training_code_sha256 8ad77d93…`）。候选与控制在**第一个 update 的 loss 逐位相同**（1.6504188776016235 / 1.2071821689605713 / 1.4734519720077515，差 0.0），说明同一父、同一 batch、同一前向与损失定义；但之后训练轨迹不同，且配对差混合模型代码版本与 run-to-run 差异。控制−dose 桥 15/15 在 −4.787001e-6…+1.778038e-5 K 内。

## 1. 结论摘要

本轮只检验一个因子：**优化器的更新步**。控制走注册路径——一次 `result.loss.backward()`，然后注册的 `clip_grad_norm_(clip)` 与 `optimizer.step()`；候选把同一步换成**预声明**的 SAM 双 forward/backward 扰动更新：

```
g     = grad of the registered scalar loss at w      (one global norm over every
                                                      trainable parameter)
e     = rho * g / (||g|| + eps)                      (one scalar rescaling)
w     -> w + e
g_sam = grad of the *same* registered loss at w + e
w     -> the saved original parameters, copied back bit for bit
then the registered clip_grad_norm_(clip) and optimizer.step() act on g_sam
```

`rho` 取论文默认 **0.05**（Foret et al., arXiv:2010.01412v3, §3.1 的 p=2 对偶解与默认半径），`eps = 1e-12`，**不扫 ρ**，不做 parameter-matched 臂，不与任何已关轴组合。模型层/输入表示、数据、损失、2400 updates、LR 2e-5、warmup 10、clip 1、K=4、depth=4、dim=192、deep K-axis supervision、物理权重 (1,.5,0,.5,0,0,0,.5,0,0,0,.5)、seed 41/42/43、评分域 `interior_32`（4225 格）与气候态分母全部逐字不变。**模型文件本轮零改动**，`model_code_sha256` 仍为 `60de3b2dc5de2871f45c7a563153ef8e1cfa688be38367c89d0c718dfc06df3e`。

**预声明出口（不新增数值阈值）：**

| 条件 | 内容 | 结果 |
| --- | --- | --- |
| (a) | 候选 72 h `interior_32` t2m skill 三 seed 全 >0 | **false**（-0.267285 / -0.351110 / -0.227842） |
| (b) | 同 seed 候选−控制 72 h ΔRMSE 三 seed 全 <0 | **false**（0.139145 / 0.106915 / 0.108934） |
| (c) | 候选 48 h skill 三 seed 全 >0 | **false**（-0.017313 / -0.107794 / 0.009830） |
| (d) | 6/12/24 h 的 seed 均值 ΔRMSE ≤0 | **false**（0.119104 / 0.172023 / 0.098197） |
| (e) | 控制−pinned dose 桥在 −4.8e-6…+1.8e-5 K | **true**（15/15 单元，-4.787001e-06…1.778038e-05 K） |

⇒ `scientific_exit_met=false`、`predeclared_exit_met=false`、`identity_unresolved=false`。**按预声明永久关闭「训练期 SAM 更新规则」支**：不扫 ρ、不改扰动范数/eps/符号、不加 seed、不改 updates/LR/warmup/K/depth/dim、不做 parameter-matched 臂、不与已关轴组合、不换名重开。

**怎么读（必须照此）**：这不是 null/inert。三 seed 的 2400 次更新**全部**施加了非零扰动，参数逐位还原，扰动点的注册 loss 平均升高 0.157（min 0.0475，max 1.8326；这是**归一化 loss 单位**，不是 K），即上升步确实走到了 loss 更高的邻域；第二个反向也确实在 w+e 处求梯度。但由此得到的权重在**每个** lead、**每个** seed 上都比控制差，且 72 h 绝对 skill 三 seed 全负。所以结论是：在这个冻结的 2400-update 中程天气 recipe 上，SAM 不是改进因子；它既没有把 72 h 的负 skill 拉正，也没有在短 lead 上换来任何配对收益。**不得**把它读成“需要调 ρ/换 eps/加 seed 就能赢”——预声明禁止扫 ρ，且这里 15/15 一致变差，不是噪声方向不一致。

**成本**：process 合计 47505.319 s、**13.195922 GPU-h**（prepare 393.269500 + archive 385.225169 + seed41 18699.107992 + seed42 16284.402920 + seed43 11321.884432 + reading 421.428780）；墙钟 28867.520 s（8.0188 h；seed41/42 并发，seed43 随后接管先空出的卡）；磁盘 20960151541 B（19.52 GiB）；峰值保留显存 9024045056 B。无软/硬超时（`soft_overrun_seconds` = `hard_overrun_seconds` = 0）。

**计算混杂（必须与结果一起读）**：SAM 每次更新跑**两次** forward/backward，所以 FLOPs 与 GPU-h 约为注册臂的两倍；本轮 process 13.1959 GPU-h 对上一轮 tendency 轮的 7.8054 GPU-h，比值 1.691。**没有**跑 compute-matched 的 4800-update 控制臂，因此“SAM 更差”与“SAM 用了更多计算仍更差”这两句话在本轮是同一件事的两种说法，报告不得只留其一。

## 2. 三 seed 主读数

`interior_32`（4225 格）t2m RMSE（K）与 skill = 1 − MSE_model/MSE_climatology：

| seed | 6 h | 12 h | 24 h | 48 h | 72 h |
| --- | --- | --- | --- | --- | --- |
| 41 RMSE | 2.431772 | 3.069271 | 2.687853 | 3.382348 | 3.735603 |
| 42 RMSE | 2.402764 | 3.052331 | 2.766995 | 3.529559 | 3.857172 |
| 43 RMSE | 2.338512 | 2.970777 | 2.696661 | 3.336921 | 3.677011 |
| 41 skill | 0.521951 | 0.226088 | 0.388590 | -0.017313 | -0.267285 |
| 42 skill | 0.533288 | 0.234608 | 0.352055 | -0.107794 | -0.351110 |
| 43 skill | 0.557915 | 0.274962 | 0.384577 | 0.009830 | -0.227842 |

同 seed 候选 − pinned 控制（ΔRMSE，K；**正 = 候选更差**）：

| seed | 6 h | 12 h | 24 h | 48 h | 72 h |
| --- | --- | --- | --- | --- | --- |
| 41 | 0.149831 | 0.201037 | 0.060691 | 0.133363 | 0.139145 |
| 42 | 0.111145 | 0.170227 | 0.121175 | 0.222947 | 0.106915 |
| 43 | 0.096335 | 0.144805 | 0.112725 | 0.178156 | 0.108934 |
| seed 均值 | 0.119104 | 0.172023 | 0.098197 | 0.178155 | 0.118331 |

15/15 配对 Δ 全为正，seed 均值在 5 个 lead 上也全为正，方向完全一致。

## 3. SAM 更新的诊断（证明扰动真实且还原逐位）

| seed | updates | zero-perturbation | perturbation norm min / mean / max | grad norm mean / max | loss increase min / mean / max | restore residual max |
| --- | --- | --- | --- | --- | --- | --- |
| 41 | 2400 | 0 | 0.049999997 / 0.050000000 / 0.050000003 | 2.7264 / 43.2045 | 0.0475 / 0.1569 / 1.8326 | 0.0 |
| 42 | 2400 | 0 | 0.049999997 / 0.050000000 / 0.050000003 | 2.6750 / 42.5126 | 0.0525 / 0.1583 / 1.4933 | 0.0 |
| 43 | 2400 | 0 | 0.049999997 / 0.050000000 / 0.050000003 | 2.7635 / 65.4398 | 0.0531 / 0.1619 / 1.4436 | 0.0 |

扰动范数由**一个**全局标量缩放决定：`||e|| = rho * ||g|| / (||g|| + eps)`，`eps = 1e-12` 只保证零梯度时有限。三 seed 的实测均值都在 0.05 的 1e-9 内，`restore_residual_max` 三 seed 全为 0.0（用保存副本 `copy_` 还原，而不是 `w - e`，后者在浮点下不可逆）。

**独立复核（本轮主链路自查，非本仓测试）**：对闭式二次损失 `L = 0.5·c·w²`，SAM 在 `w0>0` 处的扰动点梯度必须等于 `g + c·rho`。实测 registered grad 2.1、perturbation norm 0.049999999999976、perturbed-point grad 2.2499999999999285（预测 2.25）、restore residual 0.0，与闭式预测一致——更新规则不是空操作。

## 4. 全部 17 变量 × 5 lead（interior_32，物理单位；不做跨单位平均、不挑变量）

候选 / 控制（同 seed 的 pinned 控制读数）。单位：t2m/t850/t500 K，u10/v10/u850/v850/u500/v500/u250/v250 m/s，mslp/z850/z500/z250 m²/s²，q850/q500 kg/kg。

### seed 41

| 变量 | 6 h | 12 h | 24 h | 48 h | 72 h |
| --- | --- | --- | --- | --- | --- |
| t2m | 2.43177 / 2.28194 | 3.06927 / 2.86823 | 2.68785 / 2.62716 | 3.38235 / 3.24898 | 3.7356 / 3.59646 |
| u10 | 1.33085 / 1.28416 | 1.64992 / 1.58036 | 1.86784 / 1.81189 | 2.0396 / 1.99917 | 2.07948 / 2.04703 |
| v10 | 1.51511 / 1.46106 | 1.96334 / 1.8884 | 2.31693 / 2.23286 | 2.68343 / 2.5961 | 2.75464 / 2.73674 |
| mslp | 146.113 / 133.688 | 224.939 / 203.14 | 325.204 / 298.569 | 424.112 / 401.635 | 458.393 / 445.191 |
| z850 | 84.9142 / 79.4557 | 133.874 / 124.345 | 206.638 / 195.44 | 266.334 / 259.936 | 280.413 / 277.922 |
| t850 | 1.69198 / 1.63833 | 2.33855 / 2.22734 | 2.73491 / 2.58441 | 3.6665 / 3.45355 | 4.06887 / 3.91909 |
| q850 | 0.00118047 / 0.00117246 | 0.00153317 / 0.00151464 | 0.00177594 / 0.00174195 | 0.00208842 / 0.00203441 | 0.00220806 / 0.00211671 |
| u850 | 2.46727 / 2.35046 | 3.22856 / 3.08639 | 3.73923 / 3.65528 | 4.11317 / 4.09559 | 4.25066 / 4.24787 |
| v850 | 2.58111 / 2.44308 | 3.47043 / 3.26067 | 4.24831 / 4.03858 | 4.82531 / 4.6844 | 4.90272 / 4.86707 |
| z500 | 114.26 / 103.19 | 191.775 / 175.694 | 305.841 / 283.951 | 421.692 / 394.099 | 477.7 / 442.876 |
| t500 | 1.07791 / 1.0393 | 1.57196 / 1.48935 | 2.18552 / 2.03282 | 2.86783 / 2.6516 | 3.23272 / 3.04225 |
| q500 | 0.000702808 / 0.00067364 | 0.000916906 / 0.000866525 | 0.00104765 / 0.00100375 | 0.00115242 / 0.00113821 | 0.00117595 / 0.00116605 |
| u500 | 3.63888 / 3.58085 | 4.94089 / 4.85869 | 5.98772 / 5.82837 | 6.77339 / 6.5279 | 7.06054 / 6.87836 |
| v500 | 3.80166 / 3.55089 | 4.88596 / 4.53244 | 5.71391 / 5.4239 | 6.28339 / 6.21117 | 6.50551 / 6.49337 |
| z250 | 187.904 / 172.606 | 321.646 / 298.9 | 524.205 / 494.257 | 779.02 / 732.2 | 904.296 / 860.916 |
| u250 | 5.35376 / 5.27835 | 7.72567 / 7.60813 | 10.1504 / 9.8907 | 12.3382 / 11.7756 | 13.4275 / 12.9107 |
| v250 | 5.63647 / 5.45809 | 7.71586 / 7.38165 | 9.48857 / 9.11763 | 10.7154 / 10.6369 | 11.1434 / 11.2332 |

### seed 42

| 变量 | 6 h | 12 h | 24 h | 48 h | 72 h |
| --- | --- | --- | --- | --- | --- |
| t2m | 2.40276 / 2.29162 | 3.05233 / 2.8821 | 2.76699 / 2.64582 | 3.52956 / 3.30661 | 3.85717 / 3.75026 |
| u10 | 1.32738 / 1.28466 | 1.64931 / 1.57586 | 1.88004 / 1.81054 | 2.05069 / 2.00636 | 2.08277 / 2.04937 |
| v10 | 1.51513 / 1.45488 | 1.95365 / 1.8726 | 2.32427 / 2.22093 | 2.68448 / 2.59601 | 2.75226 / 2.73127 |
| mslp | 143.22 / 132.393 | 223.97 / 202.111 | 327.247 / 296.515 | 427.295 / 400.628 | 460.834 / 446.596 |
| z850 | 84.0507 / 79.5594 | 132.718 / 124.401 | 204.631 / 194.268 | 263.298 / 258.105 | 277.742 / 275.903 |
| t850 | 1.68204 / 1.61639 | 2.33076 / 2.18883 | 2.78138 / 2.53833 | 3.72739 / 3.40446 | 4.10092 / 3.93783 |
| q850 | 0.00117718 / 0.00116896 | 0.0015229 / 0.001503 | 0.00175342 / 0.00170953 | 0.00208946 / 0.00200272 | 0.00224266 / 0.00211565 |
| u850 | 2.46173 / 2.34767 | 3.22602 / 3.087 | 3.74746 / 3.64266 | 4.12187 / 4.08894 | 4.24546 / 4.22959 |
| v850 | 2.55089 / 2.41532 | 3.42793 / 3.20333 | 4.25227 / 4.00959 | 4.85049 / 4.67719 | 4.91766 / 4.82959 |
| z500 | 113.563 / 105.596 | 191.503 / 178.69 | 304.145 / 286.018 | 417.915 / 400.782 | 472.693 / 458.575 |
| t500 | 1.09613 / 1.05312 | 1.59519 / 1.51168 | 2.19729 / 2.05267 | 2.84991 / 2.69289 | 3.19504 / 3.08972 |
| q500 | 0.000697517 / 0.000673146 | 0.000906849 / 0.000866013 | 0.00104029 / 0.00100372 | 0.00115104 / 0.00114016 | 0.0011765 / 0.00117745 |
| u500 | 3.62083 / 3.52244 | 4.90601 / 4.76027 | 5.9469 / 5.71599 | 6.77674 / 6.45581 | 7.12679 / 6.87338 |
| v500 | 3.72399 / 3.47095 | 4.78307 / 4.43812 | 5.63629 / 5.35834 | 6.27877 / 6.22392 | 6.49566 / 6.54532 |
| z250 | 192.242 / 174.212 | 327.721 / 301.96 | 523.062 / 496.599 | 769.738 / 738.872 | 891.986 / 869.493 |
| u250 | 5.34925 / 5.26145 | 7.72099 / 7.55888 | 10.1702 / 9.81636 | 12.4334 / 11.8611 | 13.6432 / 13.0702 |
| v250 | 5.62993 / 5.41819 | 7.68116 / 7.29824 | 9.43728 / 8.98488 | 10.6918 / 10.5726 | 11.1179 / 11.2919 |

### seed 43

| 变量 | 6 h | 12 h | 24 h | 48 h | 72 h |
| --- | --- | --- | --- | --- | --- |
| t2m | 2.33851 / 2.24218 | 2.97078 / 2.82597 | 2.69666 / 2.58394 | 3.33692 / 3.15877 | 3.67701 / 3.56808 |
| u10 | 1.32425 / 1.2776 | 1.64812 / 1.5783 | 1.87279 / 1.82107 | 2.04688 / 2.0179 | 2.08624 / 2.08591 |
| v10 | 1.49605 / 1.43662 | 1.93434 / 1.85353 | 2.28588 / 2.20259 | 2.65618 / 2.59029 | 2.74256 / 2.72721 |
| mslp | 141.259 / 131.13 | 221.606 / 201.65 | 319.512 / 293.609 | 414.803 / 393.36 | 452.882 / 444.243 |
| z850 | 84.4016 / 79.315 | 133.588 / 124.516 | 204.575 / 193.321 | 262.803 / 257.564 | 279.292 / 279.187 |
| t850 | 1.68073 / 1.62784 | 2.30418 / 2.20288 | 2.67222 / 2.51785 | 3.56952 / 3.34698 | 3.98768 / 3.8202 |
| q850 | 0.00118075 / 0.00117078 | 0.00153247 / 0.00150758 | 0.00177321 / 0.0017202 | 0.00208681 / 0.00200136 | 0.00220147 / 0.00211919 |
| u850 | 2.47471 / 2.35168 | 3.24066 / 3.0814 | 3.74206 / 3.62935 | 4.11331 / 4.07158 | 4.25085 / 4.22903 |
| v850 | 2.5626 / 2.43218 | 3.42497 / 3.21662 | 4.20478 / 3.97651 | 4.81892 / 4.70741 | 4.89946 / 4.91179 |
| z500 | 113.72 / 102.831 | 192.566 / 177.621 | 306.849 / 291.553 | 421.323 / 407.165 | 473.269 / 460.583 |
| t500 | 1.08066 / 1.03872 | 1.58102 / 1.49554 | 2.18613 / 2.04035 | 2.82375 / 2.62739 | 3.15184 / 2.9639 |
| q500 | 0.000709047 / 0.000682282 | 0.000925081 / 0.000881375 | 0.0010526 / 0.00101704 | 0.00115522 / 0.00114072 | 0.00117512 / 0.00116621 |
| u500 | 3.63552 / 3.55399 | 4.90817 / 4.78702 | 5.9168 / 5.7191 | 6.68013 / 6.44273 | 6.98941 / 6.83437 |
| v500 | 3.80634 / 3.55661 | 4.87117 / 4.53877 | 5.65965 / 5.41711 | 6.25163 / 6.27971 | 6.4795 / 6.57843 |
| z250 | 183.343 / 166.927 | 318.142 / 294.117 | 519.33 / 493.671 | 768.429 / 728.633 | 885.226 / 860.62 |
| u250 | 5.35263 / 5.23787 | 7.72818 / 7.52344 | 10.1231 / 9.73671 | 12.2732 / 11.7444 | 13.4118 / 12.9685 |
| v250 | 5.61703 / 5.41517 | 7.68826 / 7.31694 | 9.46448 / 9.02395 | 10.7039 / 10.6575 | 11.1587 / 11.324 |

## 5. 身份与桥

- `protocol_sha256` = `b7ae03e16afde8c36c16a549cc24897cc04b4e52fb0bce56c85ee5d93f3f2680`（冻结时写入 `preparation_protocol.json`，stage 逐字比对）
- `code_commit` = `23f9af32cc1c2f3b2427d1c379ed9805b9cc044b`；`code_zip_sha256` = `fd5723776e3bc11b3f3f8486453d56e285c68f3ef30a9c2af49a73908606bb30`；ZIP comment = 上述 commit（archive 阶段断言）
- `train_data_identity` = `2360d42b7393cd6f53cd6f48961abc7bcd62948d3c2fa69b2d810d798d137c12`（与 pinned 控制、dose 参考同值）
- `source_sha256` = `6bc9a1a2d93d345ce0dcb39d4f3a26f175ea3bcc15082ac4931a179ca13b7701`
- `model_code_sha256` = `60de3b2dc5de2871f45c7a563153ef8e1cfa688be38367c89d0c718dfc06df3e`（本轮未改模型文件；控制记录为 `3fe747b2…`，其差异是模型修订标识，已用 update-0 逐位 loss 相等界定）
- `training_code_sha256` = `a03f9ba653cdf2051155d8d0dd7f98487326f0ead3c617af6756dfd820331e1d`（承载本轮 SAM 因子）
- 父迁移：`parent_checkpoint_sha256` = `fb1e4d6522d32db1cc078a7590dd66c41f479da49ff58f2c6c525f8dfceca387`；`parent_state_sha256` = `7c416ccb4054cf7e077788a43b02917e6a2dcf2503058152a7f0ab4b0ab70707`；`anchor_init_sha256` = `de0e76beb548881b282fbd8c6f194055286ec7d0dd1d7ba88dad476fc2aa1112`；`anchor_table_sha256` = `da2f28c9b3de0e175dafd6ee02b656cbdbeb9898d7fb94243796a45370ebd2d0`；`tendency_input = false`（本轮不重开 tendency 支）
- 控制−dose 桥（15 单元）在 −4.787001e-6…+1.778038e-5 K 内，与上一轮同量级，说明 pinned 控制在当前评分链上仍复现注册 dose 读数

## 6. 与上一轮的比较（同一分支、不同因子）

| 轮 | 因子 | 72 h skill（41/42/43） | 配对 Δ 方向 | 结论 |
| --- | --- | --- | --- | --- |
| tendency-input | 显式 6 h tendency 输入 | −0.174997 / −0.276587 / −0.155372 | 混合 | mixed / 关闭 |
| SAM（本轮） | 训练期更新规则 | -0.267285 / -0.351110 / -0.227842 | 15/15 候选更差 | negative / 关闭 |

两轮的 72 h 绝对 skill 都是负的，但机制不同：tendency 轮是“短 lead 略正、72 h 仍负”，本轮是**一致变差**。共同点说明问题不在这些单因素上，而在更基础的优化/表示/长 lead 误差增长结构上。

## 7. 限制（`limitations`，逐字来自协议）

- development screening on one wide instance (129x129 input, interior_32 supervision, one ROI, 17 channels, 2017-2021 train / 2022 val); no significance, convergence, SOTA or generalization claim
- the candidate is a new-training-code three-seed trajectory while the paired control is a pinned old-training-code reading (s3-anchor-isolation-result, training_code 8ad77d93..., not retrained), so the paired difference mixes the training-code revision with the SAM factor; the only mitigation evidence is the counter-proof that the off branch is an independent, bitwise-registered path (sam_rho=None leaves the registered loss.backward() branch untouched, and sam_rho=0 reproduces the registered gradient exactly) plus the requirement that the candidate's update-0 unperturbed loss equals the control's bit for bit (1.6504188776 / 1.2071821690 / 1.4734519720)
- same updates is not the same compute: the SAM step runs two forward/backward passes per update, so FLOPs and GPU-hours are roughly doubled. Parameters, FLOPs and GPU-hours are reported separately and the compute confound is declared. The primary source itself (arXiv:2010.01412v3 section 3.1) gives its non-SAM baseline twice the epochs to align compute, so an equal-updates comparison is favourable to the baseline under that paper's own framework; this round does NOT run a 4800-update compute-matched control, which would be a second arm and a new recipe
- rho=0.05 is taken from arXiv:2010.01412v3 section 3.1's default, but the closest temporal precedent, SAMformer (arXiv:2402.10198v3 Table 4), reports an optimal rho*=0.1-1.0 and uses Adam rather than AdamW (this recipe uses AdamW with weight_decay 1e-4). A negative result therefore closes only 'SAM at the predeclared rho'; it does not show that SAM is ineffective at other radii
- single instance, single ROI and one season; three seeds are consistency evidence, not a significance test; GPU runs are co-resident and not bitwise reproducible; validation split only and the test split stays sealed
- the candidate model revision (60de3b2d...) is newer than the pinned control's (3fe747b2...): model/*.py gained the tendency_input attribute in the previous round even though this round's spec leaves that switch off. This round changes no model file, and the declared model-code difference is bounded by the update-0 bitwise loss equality above, but it is an additional declared difference between the two arms rather than something controlled away

## 8. 成本与预算

| 项 | 值 |
| --- | --- |
| planned_seconds | 43200.0 |
| per_seed_seconds（软预算） | 21600.0 |
| hard_cap_seconds | 86400.0 |
| 最大单 seed 进程时长 | 18699.107992 s |
| soft_overrun_seconds | 0.0 |
| hard_overrun_seconds | 0.0 |
| process 合计 | 47505.319 s = 13.195922 GPU-h |
| 墙钟 | 28867.520 s = 8.0188 h |
| 磁盘 | 20960151541 B = 19.52 GiB |
| 峰值保留显存 | 9024045056 B |
| 网络字节 | 0（R-028 离线；worker 首行 `deny_network()`） |

## 9. CI 与提交（如实披露）

本轮共有 **4 个 commit**，其中前两个被 CI 判红并已修复，最终冻结所用 commit 的 CI 为绿：

| commit | 内容 | CI |
| --- | --- | --- |
| `1b3bc4d` | 实现（驱动/测试/父/SAM 模块/目标长文/决策） | `R7 CPU CI` 39644781144 **failure** |
| `7df8042` | 修正 `size-thresholds.md` 的提交集合 R-021 计数（121→124） | `R7 CPU CI` 39653959344 success |
| `86f6689` | 父模块补 `ANCHOR_TABLE_RUN` 再导出 + 2 条反漂移测试 | `R7 CPU CI` 39661905622 **failure** |
| `23f9af3` | 扰动范数断言改为对闭式 `rho*||g||/(||g||+eps)` 断言 | `R7 CPU CI` 39666233779 **success** |

**第一次 GPU 尝试（`attempt01`）全部 seed 失败**：`AttributeError: module 'scripts.r7_s3_sharpness_aware_minimization_parent' has no attribute 'ANCHOR_TABLE_RUN'`（`_arm_contract` 在 GPU 启动后才读到该名）。`attempt01` 目录**保留不删**，作为失败尝试记录；修复后按新 code commit 重新冻结为 `attempt02`（`protocol_sha256 b7ae03e16afde8c36c16a549cc24897cc04b4e52fb0bce56c85ee5d93f3f2680`），本轮读数全部来自 `attempt02`。修复只动了一个再导出与两条测试，未改任何因子、阈值或冻结常量。

## 10. 未做的事与下一步

- 未读 `test.jsonl`（2023 test 仍封存），`test_read: false`，确认消耗 r=0；
- 未做 ρ 扫描、未做 parameter-matched 控制臂、未做 compute-matched 4800-update 控制臂；
- 未与任何已关轴（容量、H-A…H-D、dense 12 步、final-only、空间梯度、EMA、zero-base decoder、short/long gradient surgery、PE/scale/typed/ordering/anchor/rollout-dose、显式 tendency 输入）组合；
- 未启动 S4；数据扩充仍按决策 0038/0041 暂缓。

**下一步（§8）**：训练期优化更新规则（SAM）已按预声明永久关闭，且本轮 15/15 一致变差，不再在 ρ/eps/seed 上追加尝试。下一独立单因素轮须先做一手检索再提假设；若检索不到新的、与已关轴不重叠的可证伪因子，则应把结论落到“当前 recipe 的瓶颈不在这些单因素上”，转入对长 lead 误差增长与目标/监督结构的更基础假设，而不是继续在已关轴上换参数。

<!-- record: s3-sharpness-aware-minimization-result -->
