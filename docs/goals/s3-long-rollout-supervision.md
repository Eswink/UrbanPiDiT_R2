# S3：直接长物理监督，独立于已终结的短双步微调

<!-- round-node: S3 -->

状态：active / screen-supported-audited，2026-10-06。唯一主计划仍为
`docs/goals/main-model-climatology-campaign.md`；本文件是其S3新的独立开发轮，不是S4或新方向。

## §0 Objective（单段，实测1123字符）

> 沿 docs/goals/main-model-climatology-campaign.md 的S3接续已登记的短rollout微调负结果，执行 docs/goals/s3-long-rollout-supervision.md 的独立机制：直接覆盖48/72h生成历史而非只监督+12h。D1新train-only12步精确时间metadata preflight与多目标适配器、全BPTT和目标隔离CPU阳性/反证，不改旧数据/model或父配方；D2在新路径先冻soft1800/hard3600秒FP32单train样本12物理步forward/backward可行性与FLOPs/ownmemory；D3可行且独立工程审阅后另冻soft6300/hard12600/perseed3600秒三seed41/42/43一次筛选，parent固定v3-BD1600endpoint、freshAdamW、LR2e-5/warmup10/200更新/K4/FP32，物理loss L6+.5(L12+L24+L48+L72)、全部生成历史不detach，不用未来真值forcing；D4同数据2022val全cohort472/468/460/444/428评分全部17变量、气候态只fit2017–2021，按既有t2m6/12h三seed同号主格与u10/v10/mslp五lead零退化合取逐字判读，不平均反号，不读2023test；D5独立审阅、归档代码临时重放、E/index/brief/campaign账本和精确SHA CI齐，完整negative/failed都保留并全额计费。0030/0038授权范围内自主推进，无总GPU-h上限；soft超继续记账，hard或真实错误停当前attempt，不自动retry或原地resume；默认GPU1共驻、每spawn只读核UUID/余量，外部截止只管父直接拥有的worker，禁止signal邻居/cron/守护/付费/独占/main合并/用户安全配置变更。严格model/source/data/BUILD_COMPLETE/checkpoint/协议身份不绕，输出排他，配方预算先冻结；本轮可证伪假说仅直接长lead监督能否减少误差与守门反例，不宣称收敛、机制因果或科学确认。科学门只来自docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md与docs/goals/s3-confirmation-baselines-and-candidate.md；未执行不写PASS，skip/partial不算通过，不自行宣布goal complete。

## §1 承接与事实

- 起点：注册提交 `109e0e106f161a2b56f95d3b8f91aaa8c4b9e66d`；精确主CI
  37487719159 `completed/success`，17实验workflow标签未触发skipped，不能当科学评分。
- `docs/R7_S3_V3_ROLLOUT_FT_ATTEMPT02.md`：短two_step200完整三seed主格supported、守门13/45未过，
  t2m48/72h气候态skill全负。已停止该剂量，不重开同配方，不拼接failed attempt01 seed。
- 当前累计14.1154GPU-h，包括attempt01失败与attempt02及全部重放。cap16/remaining1.8846只是会计标尺。
- 新机制直接对72h全生成历史反传，在6/12/24/48/72h计算预声明监督，其他物理步仅推进状态。
  新训练窗口比双步更少，metadata-only实跑420.026880s已确认2360原始/2140完整/220排除，
  window digest `dccfeef926286ddeccb7690aacd7bac45c1bb2036fe6ebbe826841b971bb39ec`；
  `state_fields_read:false/test_read:false`，工作摘要 `/tmp/r7_long_rollout_metadata_preflight_20261006.json`。
  仍须在实际attempt有界准备worker重算并冻结，不拿该工作摘要代替protocol。
- 底层模型仍actual-C process；model源码和父checkpoint不改，新training源码单独digest绑定。

## §2 交付物清单

| 编号 | 交付物 | 证据形态 |
| --- | --- | --- |
| D1 | metadata-only精确train窗口、12步目标/全BPTT与泄漏反证 | preflight摘要/digest、tmp_path测试实跑；非真实天气fixture声明 |
| D2 | 单train样本FP32可行性、FLOPs与显存 | fresh probe protocol/source/codearchive/worker回执，峰值/梯度/损失，全部实际成本 |
| D3 | 可行后单次三seed冻结训练与val评分 | fresh screen protocol/200endpoint/report/15组完整val/provenance |
| D4 | 原判读合取、全部坏变量与独立审计/重放 | 完整765cell身份审阅、输出digest、可复现等级及负面 |
| D5 | 证据索引/brief、主计划账本、精确CI、交接 | 证据commit与登记commit分开、全额失败/overrun、未做和下一动作 |

## §3 判据与来源

- 科学权威仅 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`，S3判读来自
  `docs/goals/s3-confirmation-baselines-and-candidate.md` §3及原短recipe冻结文本。
  长监督沿用原主格6/12h三seed同号、关键变量五lead每seed相对MSE<=0.0合取，未放宽。
- 不是三seed均值改善就通过；非有限/缺case/错单位/身份/缺endpoint均拒绝，完整negative如实登记。
- 本轮只full-region开发点估计；正式S4的全lead正气候态skill、全年度/四季同时区间及边界报告不省。
  数据四季30日块仍不等于完整未见年。test评分r未消耗。

## §4 实施顺序与依赖

1. 已登记前轮终态；新dataset/objective模块与CPU反证，原父文件/模型不编辑。
2. 严格新freshoptimizer runner、独立身份消费和父own-Popen执行器；独立代码审阅及全量测试。
3. commit/精确archive，CPU准备在有界worker中核source/data/windows/全代码并冻结protocol；
   probe受整轮截止，归档复制也有界不自spawn孙进程；每worker前只读GPU余量。
4. 仅probe身份/梯度/显存成功且新spawn门满足时启动新screen，不能通过换精度/减物理步放宽本协议。
5. screen完整或失败都独立核费用/身份；科学negative结束本剂量，返回独立机制而非同配方重试。

## §5 预算与停止条件

| execution | planned soft | hard | per-seed | GPU |
| --- | ---: | ---: | ---: | --- |
| FP32 probe，单train样本12步forward/backward，无optimizer update | 1800s | 3600s | 不适用 | GPU1共驻 |
| 可行后screen，三seed200updates与全部val | 6300s | 12600s | 3600s | GPU1共驻 |

UUID `GPU-9d1624af-9d77-aa7c-0620-b6cb778f4ced`；spawn余量>=max估计2048MiB、已测ownedreservedpeak
加2048MiBmargin，不看总卡容量代替实际free。无租卡/独占。probe/screen分别冻结、分别计费。
软超继续并记录，仅硬deadline/真实错误停当前attempt，全额成本包括CPU/启动间隔/归档/判读/清理。
本轮前瞻会计cap20.0（现used14.1154/remaining5.8846）非许可门；screen最坏硬3.5h、probe硬1h，
不改旧cap历史事实或任何已冻结协议。若不可行，记录失败和独立分析，不偷偷换loss/step/precision。

物理权重固定 `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)`，每物理步内部K draft正常深监督。
参数/数据/更新保持parent接续规则，新物理target及窗口变更一起报告，不伪称等FLOPs或纯单一因果。

## §6 planner与主链修订

委派planner提供只读代码设计草稿；主链拒绝错误旧路径、BF16变更、总卡22GiB门、过早GPU、
新60秒轮询看门狗、写未编号docs/plans JSON和不准确授权边界。采用新独立模块/既有bounded_process，
科学阈值未外委。工作态审定JSON `/tmp/r7_long_rollout_reviewed_plan_20261006.json` 经
`tools/check_planner_plan.py` verified:true、fence_stripped:false；两处代码引用已实核。
Explore只读定位fixture、exacttime/gradient/whitelist；隔离implementation仅newdataset/objective/test三文件。

## §7 明确不做

不读test、不开S4、不新增下载/数据重建、不改data/raw|interim|processed/归档/旧输出/model/旧recipe；
不弱化/删除测试、不改判据、不平均反号、不无限retry；不新依赖、安全配置、凭据、GPU独占/付费、
main/issue关闭/force/mirror、常驻/cron；不自裁goal complete。

## §8 进度与交接

- 状态：active，screen-supported-audited；S3未变、S4未开始。
- 前轮negative/精确重放/全部费用已登记，来源 `docs/R7_S3_V3_ROLLOUT_FT_ATTEMPT02.md`。
- 新dataset/objective84、runner22、证据消费23个CPU反证加规模anti-drift，共**130 passed/172.22s**；
  metadata-only已核2140/220/window digest。独立prelaunch审阅的三项保障缺口已修复并定向复查无剩余
  essential blocker（不代runtime/科学接受）。执行提交
  `66836d29dd257a11b0c946c8af2622b35042a5ce`已推工作分支，精确主CI
  [37500335847](https://github.com/Eswink/UrbanPiDiT_R2/actions/runs/37500335847) completed/success，
  conventions/campaign/index及全部测试步骤均success；17实验workflow标签未触发skipped不作实验通过。
- 全量回归已实跑：**3887 passed、3 skipped、6 warnings，1169.72s**；三个optional真实fixture跳过
  不作天气验证通过。37阻断conventions、48条index/canonical brief、显式campaign和goal结构均通过；
  四条历史campaign notes照实保留。
- 已冻结并启动fresh单样本FP32 probe：`outputs/r7_s3_long_rollout_probe_20261006_attempt01/`，
  code.zip SHA256 `3d8a494587f41e0b33e6f5e1cbab9de8b23ad552433282e4dd03022cf28eabb8`；
  planned1800/hard3600不变。三个worker均exit0/reaped/no signals，整轮1605.013031s、overrun0，
  单样本loss1.2537332773/gradient40.788814545有限，FLOPs393859201536，reservedpeak2409627648bytes；
  下一spawn门至少4346MiB。仅可行性，无optimizer更新或val/test评分。独立身份/成本/作用域已复核，
  audit soft600超268.864359s而hard1200未触，审阅限制与scalar-conversion警告均保留；
  冻结证据 `docs/R7_S3_LONG_ROLLOUT_FEASIBILITY.md`已提交5abc3da，index两record已登记。
  归档同样本重放完整846.060163s、五测量字段精确同原值，两个workerreaped/no signals；
  独立terminal审阅28项通过（85.756s），所有重放作用域/再现等级限制保留。原probe0.4458+replay0.2350
  全額计费，方向累计14.7962，index现50条；没有optimizer更新或val/test评分。
- 已启动另冻三seed一次screen：`outputs/r7_s3_long_rollout_20261006_attempt01/`，同一执行archive66836d2，
  compatible feasibility自动重核、known peak carried、每spawn余量至少4346MiB；
  soft6300/hard12600/perseed3600、200更新/原physicalweights/K4/FP32/freshAdamW不变。
  三seed均完整200endpoint/五lead全部val，控制相对主格6/12h均supported、守门0/45，
  全255格RMSE低于控制；parent相对守门16/45正格，t2m48/72h气候态skill仍三seed全负。
  独立全部765RMSErow/20,448case审阅、strictcheckpoint/template/optimizer/loss与费用核验齐；
  保存advance-to-S4-freeze只表示独立冻结包准备资格，不是科学通过或已读test。
  整轮8994.237921s、soft超2694.237921s/hard超0，全部6workerexit0/reaped/no signals，2.4984GPU-h
  已本轮index登记加入；seed41归档重放1935.462220s/ceil0.5378GPU-h、soft超135.462220/hard0，
  15CSV/2272case精确同原值，三seedcollector摘要一致。方向累计17.8324，index52条，证据d1dcd46。
  独立replayterminal153项数值无差，optional文档收尾hard超3.402s不是全预算合规，限制全额保留。
  登记提交e9db4fb的精确CI
  [37509906302](https://github.com/Eswink/UrbanPiDiT_R2/actions/runs/37509906302)现已completed/success，
  所有必要约定/campaign/index及测试步骤均success（不是科学接受）。screen protocol已冻结
  `a1631d9b87721c2260a28c5862ef26ae7245fdd5f7c30cd404cb05708a8d10ea`。
- 归档seed41五lead重放与全三seedcollector exact比对已完成；独立静态初稿两项blocker首次执行前修复，
  128+36CPU synthetic检查及独立定向复查齐，wrapper仅/tmp，原screen不改。
  冻结证据 `docs/R7_S3_LONG_ROLLOUT_SCREEN.md`，replay soft1800/hard3600、spawn≥4370MiB。
- 600条training norm全部clip前>1，四50更新块末vs首改善但41/43块3→4反弹，样本不同不判收敛；
  零GPU摘要已登记证据§8，不能据此盲目加剂量。
- 下一具体动作：核本轮登记精确SHA CI，再在S3设计/冻结同case train/val差距诊断或数据制度分析，
  不碰test、不无限重复本剂量、不把advance-to-S4-freeze当科学接受、不自行complete。
  原failed/negative保持，不做optimizer更新于probe，不读val/test评分，不自行complete。
