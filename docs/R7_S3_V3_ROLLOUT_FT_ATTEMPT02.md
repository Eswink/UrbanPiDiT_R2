# S3 v3 rollout 微调 attempt02：完整三 seed，主格 supported，守门 13/45 未过

**终态：完整执行 / registered-negative；不是 S4 候选，不是科学确认。**
`scientific_claim: false`。本次从各自注册 v3-BD 1600 更新 parent 重执行一次短 rollout 配方，
没有续跑 attempt01，也没有拼接旧 seed。2023 test 未评分，确认序号尚未消耗。

## 1. 冻结身份与工程执行

运行根：`outputs/r7_s3_v3_rollout_ft_20261006_attempt02/`。

| 项 | 固定身份或实际值 |
| --- | --- |
| 执行代码 | `8466c2def5df04276cc70db530d667ba9b674d44`，工作分支 `r7/weather-reasoning` |
| 工作树 | 冻结时仅用户 `.zcode/config.json` 修改与 `.zcodeignore` 未跟踪；本轮未修改或提交 |
| 精确主 CI | [37468877656](https://github.com/Eswink/UrbanPiDiT_R2/actions/runs/37468877656)，上述 SHA `completed/success` |
| code.zip SHA256 | `443da3521393092e95da61b88bbe9765de8ba9c8767f9eee544070b6a07b718d`；ZIP commit comment 与 `code_commit.txt` 均为执行 SHA |
| 内部 protocol digest | `f02fec53283154aed8c123435f3f415ca31b653dd86df35e6152a5922581ae37` |
| protocol 文件 SHA256 | `9bc1a91973dc398666d2b9a25563bdb3335df96d065aa2e40723f8aa519bddef` |
| result 文件 SHA256 | `0a39aceab31f8ecb0b6dc7ef8c83f7997242e885b156aa276963c4cb65ddbfd5` |
| attempt 文件 SHA256 | `a531f78a9973de1c1f0722c14e9d6fb2b6788f1fbdd2b059d0f03bd17b4a5fa0` |
| readings 文件 SHA256 | `834c5520edb87e2f73c451f78252c8b3de8d1912a859b0667673fcabf90e9959` |
| 源 SHA256 / 字节 | `bc2ff9cfadcce604fc243bb999b3c430d5164d17fcf1de201273716a5db065f8` / 540,856,239 |
| train identity | `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac` |
| val identity | `0c34a887216b02b41716e7837f5be4d515ad2027e7b4a4728cbef15accd02d5a` |
| GPU | GPU1 `GPU-9d1624af-9d77-aa7c-0620-b6cb778f4ced`，默认共驻；spawn 前只读余量核验 |
| 时长 | 整轮 planned 5400 / hard 10800 / per-seed 3600 秒，均在训练前冻结 |

独立审阅核对 Git tree 和 171 个执行源文件身份；actual-C 配置来自**另一个具名固定协议/归档**，
不假称它包含在本次 code.zip 内。源全文 hash、BUILD_COMPLETE、train/val manifest、归一化身份均匹配。

工程修复只改变执行隔离和证据消费：直接拥有的 Popen 墙钟看门狗覆盖 CPU 准备、每 seed 训练/评估、
最终判读。五个 worker 均 exit0、reaped，signals 为空；没有对邻居发送信号。
原来合作式检查的阻塞根因仍未知，不能据这次完成声称换卡修复了 CUDA/I/O 或邻居问题。

## 2. 配方、父 endpoint 与计算分母

- parent：注册 v3-BD 的每 seed `l6 × 1600` endpoint；严格 CPU checkpoint/model/contract 身份核验后
  仅导入 model state。每 seed 新 AdamW，不导入 parent optimizer，不原地 resume。
- candidate：`two_step × 200`，`L6 + 0.5 L12`，LR `2e-5`，warmup 10，weight decay `1e-4`，
  batch1，clip1，FP32（BF16=false）；每个物理步 K4 深监督，物理步与内部步均不 detach。
- endpoint：预冻结 update200，无 val checkpoint 选择。各 seed 有 200 条 L6/L12 loss 与十个 checkpoint。
- 训练窗口：2360 原始 train 窗口，2340 完整双步，20 个预声明精确时间边界排除；window digest
  `77a9945d2bf2952d589cfe8e522095bc3010c40e9680a303029b024b8736065c`。
- 主比较：candidate 相对注册 v3-D3 400 更新控制；parent-relative 另报，不混为同一效应。

FLOP probe 在 grad-enabled CPU 上计 supported aten 操作：L6 forward+backward
31,981,732,992；two-step 64,020,337,920。新增 200 更新为 12,804,067,584,000 FLOPs。
**部署配方**必须算 parent 1600 L6 + 新增 200 two-step，相对 D3 的 400 L6 比值为
**5.000889131555414**，不是 1× 或只报价新增 200 更新。elementwise/normalization 等未计入，
不是完整硬件 FLOP 或等算力实验。

| seed | candidate endpoint SHA256 | training_report SHA256 |
| --- | --- | --- |
| 41 | `390f038e6de209ac0e4ce978ae3c522c176daf9048c6f1eed5128c31106dfbf1` | `e21d1d50c4817a72466ed5b5d7f01bd41e4687f274fc4ec445497913a27ac622` |
| 42 | `9662db0f876f0de145a9ba92ada5c707db381252e50f1ccd92a2ce2f9889eb9a` | `b031d719b3c85b0a2145b576bc49defc5c795fab88055a33e3c5d88d73b65d15` |
| 43 | `37fc7aefa2975ad7f0584dad0b897a9966aaeae148d5d8aceff0744514a70919` | `067203fbbcd7e1246459709cedc28d531dcee779f9b5d38956a186868efcd2a3` |

独立审阅发现 155 个 registered trainable tensor 共 3,286,037 参数均入 AdamW；14 个非激活的
alternate/diagnostic tensor 无 lazy optimizer state 且保持 parent 值，141 个激活 tensor 的
moment/update counter 均匹配。这不是“全部参数均获得梯度”的证明，不把 registered 数量当激活数量。

## 3. 冻结判读：主格 supported，但合取失败

主格只认 t2m/full 6h 与 12h 的每 seed candidate-minus-control RMSE 同号，不平均反号。
守门为 u10/v10/mslp、全部五 lead、全部三 seed 的相对 MSE 变化 `<=0.0`。
判据来自原 recipe 的冻结文本与 `docs/goals/s3-confirmation-baselines-and-candidate.md`；未修改。

| seed | 6h ΔRMSE vs D3（K） | 12h ΔRMSE vs D3（K） | 守门正格 |
| --- | ---: | ---: | ---: |
| 41 | −0.442620 | −0.561129 | 3 |
| 42 | −0.788822 | −0.771484 | 5 |
| 43 | −0.451298 | −0.529191 | 5 |

两主格三 seed 均 supported；相对 D3，t2m 五 lead 的 15 个 seed-cell 全部降低。
但守门仍有 **13/45 个正格**（48h 5、72h 8），按零容忍合取必须 `registered-negative`。
最重反例为 seed42 的 v10/72h 相对 MSE **+72.4497%**，不能用 t2m 改善掩盖。

| seed | 保留的守门反例（相对 D3 MSE 增幅） |
| --- | --- |
| 41 | 48h v10 +8.9217%；72h u10 +7.2997%、v10 +18.5840% |
| 42 | 48h v10 +30.9776%、mslp +3.7939%；72h u10 +9.1255%、v10 +72.4497%、mslp +40.4997% |
| 43 | 48h v10 +23.4210%、mslp +1.5709%；72h u10 +1.9925%、v10 +58.2233%、mslp +54.7454% |

parent 的守门17格减至13格，只能说**改善未清门**。相对 parent 本次仍有3个正守门格：
seed41 mslp/6h、12h；seed42 u10/72h。t2m 相对 parent 在 seed41/42 长 lead 改善明显，
seed43 的48h/72h反而分别 +0.017668/+0.024324 K；不能称所有 seed 的长 lead 均改善。
本剂量短双步微调终结为负结果，不再重试相同配方或挑 seed。

## 4. 同数据、同单位气候态差距与全变量读数

同 train-only 2017–2021 气候态，逐 lead 全 cohort 472/468/460/444/428；归一化指标不冒充 K。

| lead | t2m skill seed41 | seed42 | seed43 | 同变量 seed 均值（仅描述） |
| --- | ---: | ---: | ---: | ---: |
| 6h | +0.5915 | +0.5963 | +0.6030 | +0.5969 |
| 12h | +0.3197 | +0.3303 | +0.3443 | +0.3314 |
| 24h | +0.2705 | +0.2452 | +0.3058 | +0.2738 |
| 48h | −0.4949 | −0.6944 | −0.4829 | −0.5574 |
| 72h | −1.1016 | −1.7119 | −1.1983 | −1.3373 |

48h/72h t2m 相对气候态仍三 seed 全负。全部17变量正气候态 skill seed-cell 数
**51/51/51/5/0**，而“优于 D3 的 RMSE 格数”为 **51/51/51/32/18**；两者分开报告。
全 17×5×3 的物理 RMSE、climatology skill、ACC 与 per-case 读数分别在每 seed 的
`evaluation/candidate/lead_{006,012,024,048,072}h/`；`result.json.paired_cells` 和
`parent_relative_cells` 保存全部配对格，不删坏变量、不平均不同物理单位。
本轮仅 full-region 预筛，未重新做 interior/edge，不宣称满足正式确认全部报告要求。

## 5. 全额成本与失败保持

| 项 | 实际值 |
| --- | ---: |
| 整轮 elapsed_seconds_total | 4222.618676601909 s |
| 五 worker 区间合计 | 4222.409845595248 s |
| root 开销与间隔 | 0.20883100666105747 s |
| CPU preparation worker | 200.16058273799717 s |
| 三 seed 训练 report 合计 | 295.6816603280604 s |
| 三 seed 15 组评估 loop 合计 | 2337.532753434032 s |
| 最终 CPU reading worker | 7.082830752246082 s |
| 本次保守 GPU-h | **1.1729**（未舍入 1.1729496323894193） |
| soft / hard overrun | 0 / 0 s |
| 前项累计 + 本次 | 12.5554 + 1.1729 = **13.7283**（重放费用另列后再相加） |

训练/评估的局部计时不覆盖全部加载/准备，所以不把它们的和冒充整轮。
GPU-h 为整轮墙钟等价保守计费，不是 utilization-hours，不表示独占。
各 worker owned CUDA reserved peak 为 478,150,656 bytes；不能把此值当整卡使用量。
本轮网络0、付费0、新下载0；无模型源码、源数据、store 或依赖变更。

原 attempt01 仍 failed/partial；2.6316 GPU-h（9473.920853 秒）完整计入旧账本。
protocol/result/failure 三 hash 与原登记一致，原 recipe 与 `dd2139e8…` 字节相同。
两个 rollout attempts 合计未舍入 **3.804594313789728** GPU-h，不把 failed attempt 的费用扣掉。

## 6. 独立只读审阅与工程验证

- 第一轮 seed41 审阅950 checks，工作 JSON
  `/tmp/r7_rft_attempt02_seed41_audit_20261006T141300Z_3d1683b4.json`，SHA256
  `6fc593b2b65241cd53846366ee0d560f8575d6f9a22cdd40ab47d216ab3c416b`。
  83.9 CPU-s 主计算加3.0 CPU-s修正；初步 CPU 未汇总。墙钟约17.2分钟，超600秒目标，照实登记。
- 最终审阅2514 checks，工作 JSON
  `/tmp/r7_rft_attempt02_final_audit_20261006T143025Z_3147e0b4.json`，SHA256
  `318557c61f4546fe910a3aca129a845127013c43931513586579c5e46c273a0b`。
  全765 metric cells独立重算，最大绝对 RMSE 差6.8212e-13、skill差4.8850e-15；
  433条身份记录（421本轮重算、12显式继承）。墙钟368.2秒；主计算约12.0 CPU-s，初步 CPU 未汇总。
- 审阅没有训练/评估/GPU/网络/信号，没有读 test manifest。气候态分母由保存的 CSV 核公式，
  没有在此只读审阅中重新生成气候态场；derived weather/process chunks未逐块穷尽重哈希。
- 99 个定向反证及规模 marker 防漂移测试通过；37 条阻断 conventions 0失败。
  精确执行提交全量回归 **3758 passed、3 skipped、6 warnings，997.19s**；三个 skip 是缺少
  optional real-data fixtures，不当真实天气验证通过。CPU CI 不代表 GPU 科学门通过。
- 122个原运行文件的辅助库存 `/tmp/r7_attempt02_artifact_inventory_20261006.json`，SHA256
  `363c99004faa3f0a1a3d4992dba94c6f71ddec0b71ac06d274ed33ee27e58587`，总1,173,429,828字节。
  这是工作态 inventory，原运行根不补写或覆盖。

## 7. 归档代码临时重放

临时 replay 与原科学 attempt 分开，原输出不动。重放时使用精确执行 code.zip，先核固定 ZIP SHA
再从同一已验证内存字节安全解包；setup/评分均由父直接拥有的 bounded worker 受整轮
planned1800/hard3600约束，spawn前核GPU1 UUID/余量。只重放seed41五lead全val，另用归档collector
核全部三seed保存的读数；不重训、不读test。

两次**评分前失败**均保留并全额记账：

| replay 路径后缀 | 失败原因 | 整轮秒数 | 保守 GPU-h（未舍入） | failure SHA256 |
| --- | --- | ---: | ---: | --- |
| `replay_20261006_attempt01` | Python整数lead键与保存JSON字符串键直接对象比较不相等 | 8.888092823326588 | 0.0024689146731462744 | `017e31b33d8c5d02ab04fc2013ea4a0cc3ba274fa863c21f65ca6638df7d25b3` |
| `replay_20261006_attempt02` | 直接canonical排序：整数6/12与字符串12/6顺序不同 | 8.88523946981877 | 0.0024681220749496586 | `4f3b090e82b0e27620268729050f78d8ea13e1fbd24827978a651508d28ed0f0` |

完整前缀为 `outputs/r7_s3_v3_rollout_ft_attempt02_`。两次均没有生成forecast/lead结果，
没有`RESTORATION_ACCEPTED`，不能算重放成功。失败writer未直接带protocol digest，
通过同排他目录的protocol与两process回执关联，不回补失败字段。

独立复核在clean cwd=/tmp、仅归档仓库模块加venv依赖下确认：归档collector/recipe/screen/data
均从解包目录加载，全部数值、判定和文本无变化。按原保存格式做JSON roundtrip后再canonical排序，
100,747字节精确匹配digest
`3b6b9af8a0cb343685713ec817bf22def9675461e3e0b2a12bc8363aa8a6c7b7`。
一个ULP的RMSE修改或decision修改均仍被拒绝；没有数值容差、比较逻辑绕过或原collector改动。
复核两个成功clean CPU检查合计9.64CPU-s/9.66执行wall-s，另一个import-only依赖过滤诊断失败如实保留说明。

第三个全新replay协议`a0a9b50088223f5917d6681fccc22bc99d87982aca7830d8991d4ddf87f6933e`
在 `outputs/r7_s3_v3_rollout_ft_attempt02_replay_20261006_attempt03/` 完成：

- setup/evaluate两个worker均exit0/reaped，无signals；整轮**1375.7328200042248秒**，
  planned1800/hard3600，overrun均0；本次保守**0.3821480055567291 GPU-h**。
- seed41五lead、全17变量的rmse/skill/ACC共15个CSV SHA256与原输出**完全一致**，
  全部逐case MSE数值最大相对差0.0，case集合及checkpoint/manifest/单位/气候态身份完全一致。
- 全三seed归档collector JSON表示digest精确一致；没有训练权重重放或新科学确认。
- `RESTORATION_ACCEPTED.json` SHA256
  `2c35fc9a8c123f26237cd7006156b8ce6949a567322b9c14fbeb78cbcd174789`；
  `replay_result.json` SHA256
  `34c2876275cd8aadd191732f9e6964eacbbf00bb261f1f3cad886286659cf7db`；
  单独冻结wrapper SHA256
  `4ca6e85b63d77aea216dc2aef1fabc271f1d47e6ab00960328e37a13bbdfa6be`。
- 两个failed重放加本次success合计 **0.38708504230482504 GPU-h**（账本四舍五入**0.3871**），
  不扣失败费用。前项13.7283加全部重放后方向累计**14.1154 GPU-h**。

重放为已确认的同配置数值再现，声明级别仍 **config-reproducible**；同机15个CSV逐字节相同
不证明跨软件/硬件逐位稳定，也不把训练非确定性改称bit-reproducible。原attempt01/02不被重放覆盖。

## 8. 限制与接续

- 本实例为一个区域、每年四个30日季节块，**不是完整未见年份**，不满足终极全年确认。
- 只有2022 val点估计，没有同时置信区间或跨年/四季正式确认；2023尚未评分，r仍未消耗。
- 微调前 parent 不可称已收敛，训练损失最后单样本/四段均值不证明 convergence。
- 最高声明 config-reproducible；同机评估数字重现不等于 GPU training bit-reproducible。
- 无接口、安全/凭据、依赖变化；既有父 checkpoint/evaluation 路径保持。短 rollout 科学假说在本剂量
  未清门，后续必须另立真实不同机制、新协议与新路径，不能把CI、partial或issue关闭当获胜。
- 下一独立问题是将监督**直接覆盖48/72h生成历史**，而不是只重复200更新+12h小剂量。
  先新 train-only多目标精确时间metadata preflight、CPU full-BPTT/泄漏反证及有界FP32可行性；
  可行后才新冻结三seed训练。原failed/negative、本轮门及父身份都不改，不自裁goal complete。
