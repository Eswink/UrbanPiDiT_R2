# #74 同父可微两步：实际计算完成，原聚合attempt失败

日期2026-10-03，N3独立路线0030/0032，`scientific_claim:false`。本页记录原attempt，不记录尚未接受
的派生天气比较；M3原failed/补测paused/any-unresolved/advancefalse全部保留，不重跑其23评估。

## 1. 冻结与真实执行

- 工程/实验commit：`616b029dce569b92bd08295737981512180a1ad3`；精确主CI
  [37133487340](https://github.com/Eswink/UrbanPiDiT_R2/actions/runs/37133487340)九principal及三post/complete
  步骤全部completed/success。匿名Jobs API访问2026-10-03；不是认证下载的pytest日志。
- 真实目录 `outputs/r7_74_autoregressive_20261003_attempt01/`；CLI
  `scripts/study_r7_v2_remaining.py --phase all --stage B`，本仓.venv，固定共驻GPU UUID
  `GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b`。未新下载/发布、未读封存test、未干预邻居。
- canonical协议 `9f76e6e3e0eb0d7e27d1aaaa8d59ff7616241ff50e13c30b26be7f13f5e62b02`；
  planned5400/hard10800秒，同boot从最早CPU CLI入口计时。协议文件SHA256
  `2d937850fb817d715219050c9418811f8d3dbf56a4d48b3e13fe253afd9557dc`。
- 93成员归档zip SHA256 `d3d10a74c77303849d38e06a44fa2fcb06d2ac8da0c7ddc4646b4218e0cb8bd8`，
  source-tree `5de6b16dccb9e59116116017a9f5286ac0576c8d7db270d0f6f8fe7b9c1cf2c9`，model
  `0cc9c16e7a12bf23150fb04e13acb72f548f8cb461ad64df1d45123b387a223e`。
  frozen Git/status字节有强pin；运行期间所有active runtime/source及HEAD保持冻结。
- 既有M2 train/val、source496084a9…f9bda21/data identity ef8c6691…6ccc07、scale sidecar4fed1c78…34912d
  与BUILD_COMPLETE/只读preflight强核。185个精确t+12可用train窗口，唯一split边界排除先声明；
  每lead完整val cohort22/21/19/15/11，不把短lead缩为72h病例。

## 2. 已完成计算，不冒称整轮成功

seed41/42分别从其合法M3 aux_off/RW-A/K4父权重导入：200 continue_l6、200 rollout_l6_l12、
400 equal_compute_l6，共六训练与1600 optimizer updates。均新optimizer，同seed/同185窗口、K4与
initial/all5draft归一化深监督；internal K full BPTT，两步physical graph不断开。
第二history保留第一prediction，不用future truth作输入，每次物理transition固定+6h、Gregorian
日期更新；query新开关保持False，不能把B冒称0035新local-solar/history-offset/source-position对照。

| seed | arm | updates | worker秒 | allocated MiB | reserved MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| 41 | continue_l6 | 200 | 140.7626 | 185.2686 | 214 |
| 41 | rollout_l6_l12 | 200 | 144.3842 | 313.9355 | 342 |
| 41 | equal_compute_l6 | 400 | 207.0619 | 185.2686 | 214 |
| 42 | continue_l6 | 200 | 137.0212 | 185.2686 | 214 |
| 42 | rollout_l6_l12 | 200 | 147.1321 | 313.9355 | 342 |
| 42 | equal_compute_l6 | 400 | 207.5986 | 185.2686 | 214 |

三十单lead/K4验证worker全部success，528 case evaluations、17变量、full/interior/edge_2；每job
同病例persistence与train-only climatology、负/undefined指标均在原CSV/provenance中保留。
36个fresh CUDA worker的pre-init/initialized baseline均0/0；成功worker不等于聚合通过。
原端点200/400及每20更新检查点、全部学习曲线/日志保持，未据验证挑更新数。

原CPU实际FlopCounterMode/enable_grad前向与反向测量两seed一致：L6 forward7,774,613,952，
forward+backward23,265,272,448；两步forward15,549,227,904，forward+backward46,587,416,832。
400-update L6只是可负担update-count计算控制，不假定严格同FLOPs/墙钟；不计支持范围外的elementwise/
normalization，训练/评估worker秒也不等同isolated model latency。

## 3. 真实失败原因与直接诊断

CLI exit1。聚合 `training/r7_v2_results.py` 的training objective检查使用Python binary64计算
`l6 + .5*l12`，再按物理指标1e-9/1e-12比较已从FP32 tensor导出的loss。

首失效记录：loss0.2679187059402466、L6 0.15507206320762634、L12 0.22569331526756287；
binary64组合0.2679187208414078，而真实逐运算FP32组合恰为记录的loss。两seed各200两步记录中，
旧double校验拒绝99/88条；`round32(L6 + round32(.5*L12))`全部400条严格相等、0差异。
最大double偏差2.9802322387695312e-8/1.4901161193847656e-8。这是元数据校验运算域缺陷，
不是更改训练目标、放宽天气指标容差或声称天气收益。

主链metadata-only诊断 `outputs/r7_v2_remaining_acceptance_20261003/b_fp32_loss_cause.json` SHA256
`f07a6712fbdca5906c0dd08fa15d4f7d3de6bcb6fec923eac40d07c48bb2f631`；独立诊断另有新冻结CPU协议。
0036只允许前瞻修正实际FP32组合审计、独立600soft/1200hard零GPU统计补全。原attempt不可复活，
不调用原输出上的finalize、不改原failed/协议/checkpoint/CSV、不重复六训练或三十评估。

## 4. 原失败全成本与强pin

- 原attempt statusfailed/finalizedfalse、budget_limitedfalse、outcome=null；36/36 jobs_completed、
  partialfalse表示worker库存无缺，不表示聚合完成。stage_result/paired/merged/artifact_manifest未产生。
- 连续GPU首spawn5816657.565493299至末owned reap5819726.945946411，3069.380453112535秒
  = **0.8526056814201487 GPU-h**。成功worker秒合计3011.8308975147083，不拿此较小数代连续计账。
- 最早CPU5816552.941254083至最终成本快照5819753.853685376，全轮 **3200.912431293167秒**，
  包含准备、import/profile/archive、启动/间隔、训练/评价、聚合失败及owned清理。
  soft_overrun_seconds0，未达5400/10800秒；最终receipt序列化发生在快照之后，不声称精确CLI-return全时。
- owned_unreapedfalse，每worker已退出清理；仅owned handles，没有私有allocator清理或CPU fallback。
- attempt文件SHA256 `1868e0a5904d9916b19845475a203c4eaf7aa84d37ecb8e51563bf993bbb0664`；
  execution_attempt文件SHA256 `da98745af2d0453b87e5d6aa071c5cdd64efdce062b052266805e49dde6501c3`。
- 361原文件/3,089,276,264 bytes已只读封印，pins文件
  `outputs/r7_v2_remaining_acceptance_20261003/b_failed_source_pins.json` SHA256
  `e0151b6345453b581e44088e7fdb7eaa29af24a56a0701b033782b2b0317dc84`。
  独立只读成本核验已完成，接受范围仅真实wall/GPU成本，不接受原聚合或实验结果。审计回执
  `outputs/r7_v2_remaining_acceptance_20261003/b_independent_cost_review/revision02/audit_result.json`
  SHA256 `c00dd04ca68152a9be57f41a433e1234657abd6db6139a132ab9e131d3f900db`；36不同PID、全部
  allocator0/0、natural-exit/reap与逐spawn UUID/余量核齐，361原文件opaque hash无差异。
  两首audit parser把不同clock调用误当同采样的拒绝记录原样保留；revision02按真正cost anchor
  核严格等式，timing末reap与execution取样差0.325微秒明确记载，不放宽原预算或科学判据。
  成本审计5.3582秒、361文件核对8.1985秒，0GPU，无天气/模型张量解码；完整数值统计接受仍待。

精确campaign累计4.880706349145538+precision0.04711594580465721+B0.8526056814201487
= **5.780427976370344 GPU-h**；显示沿历史逐行舍入5.7805/会计余18.2195，24是历史基数不是许可上限。
该失败全额记账，独立统计补全不重复原GPU-h。

## 5. 未做、限制与下一实际动作

尚未接受B完整聚合、候选选择、UTC分组或统计补全，不把原CSV的部分有利数字先作结论；
未做独立C、adaptive实裁、六issue关闭/main推进。真正原#72参照是RW-A/K3，不是本轮M3 K4父。
专用compatible archived source bridge已核28model members与旧严格loader，但原K3参照尚未实跑；
其旧365.25/累计lead输入语义与B Gregorian/固定+6不同，不能假称calendar完全匹配或旧分数逐位重放。

真实FP32/BF16小探针的resume/梯度证据另见R7_V2_PRECISION_ACCEPTANCE，不替代本轮天气比较。
两seed、单冬季/区域、小validation和已开发数据不支持显著性、泛化、收敛或SOTA；普通GPU训练
最多code/data/protocol绑定数值可复现，非逐位训练。独立核验只核记录与身份，不是syscall/GPU再跑。

下一实际动作是修活跃FP32审计并直接反证，另冻零GPU统计补全核完整库存/单位/充分统计/官方配对，
再按运行前B规则选择C物理objective。0035原验收补齐继续，不为负面增加seed/unroll或算力练到赢。
无新数据写入/发布、依赖安装、凭据、安全配置、收费资源或邻居进程操作；最终goal不自行裁完成。
