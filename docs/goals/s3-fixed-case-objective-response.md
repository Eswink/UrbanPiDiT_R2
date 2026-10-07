# S3：单个固定 train 病例的原目标响应

<!-- round-node: S3 -->

状态active / preparation，2026-10-07。仍属唯一主计划
`docs/goals/main-model-climatology-campaign.md` 的S3，不进入S4或独立test确认。
本轮只前置设计；尚未运行80更新或产生新checkpoint，不预写效果。

## §0 Objective（单段，实测1087字符）

> 接续docs/R7_S3_TRAIN_GRADIENT_COMPONENTS.md，在docs/goals/s3-fixed-case-objective-response.md预声明S3单病例原目标响应探针。D1固定已登记train2021January lower-median case0、seed41 long200 parent与原all17 normalized deep-K/12物理步权重/K4/FP32/fullBPTT，完整source/trainmanifest/BUILD_COMPLETE/norm/window/model/training/parentstate及archive-only兼容代码先核；D2新fresh AdamW按原lr2e-5/warmup10/cosine最低0.1/weight_decay1e-4/clip1执行恰80update，只同一已缓存trainbatch，无val/testreader、无新天气skill，记录每update12loss/total/preclipnorm/lr并预声明state/loss端点0/20/80；D3临时CPU实际tinyProcess/完整step/targetpoisoning/身份反证与独立prelaunch齐、新排他outputs冻结protocol/代码及recipe后启动owned有界worker，GPU1默认共驻每spawn核UUID和max已知2491416576B/实测peak加2GiBmargin；D4严格新contract/checkpoint/全部trace/digest/费用与独立savedfact审阅、另冻端点反算后登记，不要求rawGPU逐位相等、不放宽旧gradientreplay失败。whole soft3600/hard7200含CPU资格/归档/训练/读取/清理，soft超继续记，真error或hard停该attempt全额留痕、不resume/自动加剂量；0030/0038无总GPUh许可上限。单病例记忆响应不是泛化、全年气候态超越、模型容量/欠拟合/收敛或clip伤害因果证明，无新primary/scientific阈值，科学接受只引用docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md。旧protecteddata/归档/失败/注册证据只读，无paid/rental/exclusive/main/cron/用户配置/安全凭据变更，不信号邻居，不自裁科研goal complete。

## §1 承接、起点与信息增益

- 起点登记SHA `70fe46ea0a78f10f08ac1a6c765336b305648d45`，工作分支r7/weather-reasoning。
  精确CI37569443913 completed/success、全部必要steps success；工程通过非科学验收。
- 前轮冻结证据commit `6c3bb69696b5d4855da81e9b15eb9f5dba3acbe9`，
  `docs/R7_S3_TRAIN_GRADIENT_COMPONENTS.md` SHA
  `627b324d60f08f442592a3430113f088bb356d4dccf1b6431cbbadc5f69bbdba`。
- 前轮四train例141active/14unused、state不变；weighted48/72norm占76.0–86.1%，cosine依病例变。
  这没有证明clip伤害或需增加bulk训练。Jan原total loss1.29634428024292仅是已记录参考，不设匹配赢门。
- 代表gradientreplay attempt02精确恢复negative：161numeric leaves不同而loss/state/RNG/masks相同，
  旧比较规则与failed不回改，不开attempt03。新探针不是旧恢复重试。
- 新信息问题：固定相同原监督病例，有限80个原recipe更新能否改变训练目标及各lead loss形状？
  同病例重复拟合不同于扩bulk200/400/800剂量；响应即使明显，也不说明独立天气受益。
- index58、方向20.5799GPU-h；cap20/remaining−0.5799是会计，不是许可上限。

## §2 交付物

| # | 交付物 | 证据 |
| --- | --- | --- |
| D1 | 精确单病例、兼容归档parent与新recipe冻结 | source/preflight/train/norm/窗口/checkpoint/code/state receipts、protocol digest |
| D2 | 80原目标update与0/20/80端点 | 全80行12loss/total/preclipnorm/lr、端点state/标准r7-local-v1/checkpoint SHA |
| D3 | CPU工程资格和有界GPU执行 | actual tinyProcess及负反证、独立review、ownedprocess/headroom/deadline/完整费用 |
| D4 | 独立trace/contract资格及端点重算 | savedfact/audit/digest、另冻代表endpoint readback、evidence/index/brief/master/精确CI |

## §3 判据与解释边界

科学判据只有 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`；原S3screen引用
`docs/goals/s3-confirmation-baselines-and-candidate.md`，本探针没有科学/候选/改善阈值。

- 恰80updates、不earlystop或按loss选择checkpoint；state/loss endpoint预声明0/20/80。
  每行记录的是update前同batch监督loss；endpoint在该次step之后另算，不能将第80行误说为post80端点。
- 目标、K、全physical/internal BPTT、clip1、优化器类型/weight decay保持原配方。
  80与原200的cosine schedule长度不同，明确绑定新total_updates80，不冒称200更新后原schedule无缝resume。
- freshoptimizer，只导入严格验证的parent模型权重；不会导入旧optimizer或伪造parent200 contract用于新80端点。
- endpoint没有泛化意义，可能只记忆一个天气病例；不能由有限下降/平坦宣称整体容量、欠拟合或收敛。
- 本轮不做optional physicalMSE，避免事后扩读域和混物理单位；全部17 normalized训练目标保留。

## §4 实施顺序与真实API

1. 核已登记前轮/index/ledger与本goal，再按新冻结样本/recipe设计TEMP代码，不改生产model/evaluator。
2. 固定原candidate `449bc4ce6d9c15178fecb75a954a265ffb43f7ea8cea92575d95387047ee066b`，
   原code.zip `72953e44fa814db1e1da4d7f5c7505bcb878dc2a9765bc9f8875022a822b6ea5`/
   commit61e46bd78d4d7ce49da7a87d574e40c6c18a1c99和新wrapperhash；严格普通loader/构造语义核后恢复CPU。
3. **完整**manifest/window/source/preflight/norm/BUILD_COMPLETE资格不因仅一例免除。
   仍构造原ZarrLongRolloutDataset metadata，再按冻结manifest_index映射选唯一batch并缓存；不建绕过identity的简化reader。
4. TEMP wrapper复用原 `training_long_rollout` 与
   `training/r7_long_rollout_runner.py:71-86` 的update原语，保留逐项loss/backward/clip1/step/finite/zero_grad。
   `fine_tune_long_rollout` 本身固定重建完整dataset（同文件116-150），不能假称给它传single dataset已支持。
5. 新bound contract明示single-case selection、dataset_length1、原完整train窗口身份、parent初始化与新recipe80，
   signature由新canonical contract计算；标准 `_publish_checkpoint`/`load_checkpoint`核身份。
   checkpoint0/20/80与新signature严格一致，不把新state伪装long200。必要wrapper行为/source digest单独冻结。
6. CPU actual tinyProcess至少两个真实optimizer更新、目标不进model/全12/clip路径/状态改与参数有限，
   source/code/pins/timestamp/partialfailure/互斥输出/禁止heldout读等反证实跑；全套验证明确CUDA不可见，skip不冒PASS。
7. 新排他protocol后CPU资格→首步峰值/完整80同batchworker→纯trace/端点资格读取；统一父每spawn核门。
   本轮设置soft3600/hard7200，任何新profile预算调整必须在后续独立协议冻结前，不能运行中缩短/放宽。
8. 独立review按新contract和80完整trace/每step/nohiddenupdates核出口，全部负面/成本登记才选后续不同假说。
9. 训练成功后另冻三端点0/20/80只读readback：wholeplanned1800/hard3600，source/归档/完整新contract外锚/原病例资格、同cached batch/原training_long_rollout独立重算。
   原模型train/alltrainableFP32/enable_grad/dropout0保持，不改requires_grad/module flags或恢复optimizer；无backward/step/heldout。
   比较total和全12原生loss/state等精确字段；有差异即保留failed-exact-readback完整数值，不加allclose/epsilon/重复到偶然相同，最高configuration-only。
   每spawn核UUID与maxknown/实际训练或readbackreserved加2GiB，不能用多次最大free或缩小margin替代即时门。
   此规划经实际planner返回与主链两处API核对修订，工作态 `/tmp/r7_fixed_case_readback_coordinator_plan_20261007.json`已校结构；TEMP准备另1200/2400，不冒已执行。

## §5 预算与停止

| 项 | 值 |
| --- | --- |
| whole soft/hard | planned3600 / hard7200s，CPUmetadata/source/code/checkpoint/启动/80update/端点/判读/清理全含 |
| case | train2021January case0，init2021-01-14T12:00:00、原lower medianrank53/107 |
| recipe | 80updates、batch1、seed41、lr2e-5、warmup10、cosine minimum0.1、weight_decay1e-4、clip1 |
| endpoint | state/loss0/20/80；无按训练loss选型/earlystop |
| GPU | UUIDGPU-9d1624af-9d77-aa7c-0620-b6cb778f4ced，共驻 |
| headroom | max2GiB estimate/known2491416576B/实测reserved +2GiBmargin，至少4424MiB，每spawn核 |
| stop | identity/finite/runtimeerror或hard截断，保留partial/全部费用；不resume/自动增加updates/弱化loss |

物理weights仍 `[1,.5,0,.5,0,0,0,.5,0,0,0,.5]`；GPU峰值可能因optimizer state升高，
测首步后必须携带实际最大值。软预算超继续记，账面remaining负不单独终止授权内研究。

## §6 planner草稿与主链修订

planner只读返回JSON，工作态摘要 `/tmp/r7_fixed_case_planner_draft_20261007.json` 经checker
verified:true/fence_stripped:false；主链已实读long runner116-173、update71-86及schedule52-73。
结构校验不是内容正确性或科学证据，下列建议拒绝：

- “不需要完整manifest discovery”违反完整资格，仍做完整窗口/身份再取唯一case。
- 直接改/猴补fullrunner dataset、继承旧contract到80端点均不安全，薄wrapper复用update原语并新contract。
- 可选事后物理MSE和把all17 MSE标单一K单位不采用，本轮只原normalizedloss。
- `pytest -k not slow` 不采用，明确CPU全套不弱化/过滤测试；CUDA验证只能另冻门/预算后执行。
- 根据dry-run慢而运行中改soft/hard不采用；原budget预先冻结，真error/hard保持失败。
- planner没有权限给“邻居冲突”信号或控制权，共驻仅UUID/余量门，不干预外部PID。
- 后续readback planner草稿错误receipt字段、只凭signature、改requires_grad/dropout、把BatchNorm train说成推理、取三次最大显存/allclose及序列化误差猜测均拒绝。
  完整new80外锚、原module flags/alltrainableFP32、即时余量门和原生精确比较替代；其JSON工作态摘要结构verified但不当作证据。

## §7 明确不做

不读val/testmanifest/weather、不扩评分或r、不做climatology fit/S4/全年统计。
不改loss/clip/K/fullBPTT/model/evaluator或变相重复bulk剂量；不把单病例下降称科学改善。
不动protecteddata/archive/旧outputs/failed/已注册evidence/hash；不下载/付费/租卡/独占/main/issue关闭/force。
不改用户config/安全/凭据/依赖，不cron/守护或会话外续跑，不signal邻居。

## §8 进度与交接

- 状态active/preparation，S3；上一轮梯度证据/negative replay/资源缺口已登记index58、方向20.5799GPU-h。
- 本轮已做：planner实际只读委派、结构校检与主链API抽查，拒绝上述危险/不适用建议，goal前瞻冻结。
  TEMP实现已隔离委派，只许静态代码/固定ZIP的合成CPU接线，preparation soft600/hard1200；不得自行实际数据/GPU或仓库修改。
- 文档4e4b8880a73fe97fb08adf6309487f8e208e49de精确CI37571159799 completed/success，全部必要steps success。
- TEMP初步CPU实现handoff，原runner `_update` 未改；actual tinyProcess合成80update/newcontract/标准0/20/80已执行，非天气证据。
  早期collection失败与28passed/7failed（schedule literal及synthetic缺字段）保留；最后全套34passed/1failed/0skip（48.89s）是guardedAdamW lazyimport错误。
  后独立fresh子进程原guard/all692Python/exactarchive+两真实update/checkpoint0普通load成功，显式torch._dynamo预加载补入wrapper，**最终编辑包未全套重跑**。
  preparation1234s、soft600超634/hard1200超34，保留hard-budget breach/not-PASS，editing停止，未launch-ready。
- 临时snapshot：driverb4ed5e4cdcd06aa2f4ccebdd79fafcf6528a258fefc78a9e8d5a5b75af9b01d6，
  support464f9f674b7ec926e8778058cb93f4e0c3b388d330d5f97c3573301ea8b61099，
  tests5c1c5c472d900726c28ebe1a049e868cab4ca13bb8e346946d1c5726b3bf9c0e。
  checkpoint IO薄adapter使用xb/flush/fsync不放宽原guard，保持标准payload/普通strict loader，但非旧temp/link原子发布，partial可能保留；
  injected-save/fsync failure、完整RNG/部分bootstrap/refreeze资格未齐，不能冒称安全atomic或完整prelaunch。
- 独立静态checkpoint设计审阅291.265s（soft180超111.265/hard360未超）已指出瞬时nlink2守卫冲突、ordinaryloader不校完整state/RNG。
  新独立prelaunch另冻soft600/hard1200执行合成CPU完整/失败留痕/state/RNG/driver边界，不修改owner或实际输入。
- 独立prelaunch初读/实跑确认essential false-accept：checkpoint RNG缺失/坏形状可接受；AdamW漏active slot/伪inactive slot/负二阶矩/step错误及recipe flags未核；
  report反义scope/failed_training sentinel/extra update81仍被complete接受；coherent重签source为零而保持protocol/wrapper也通过，full新contract须锚冻结期待body。
  当前包not-qualified，实际source/GPU尚未启动。重跑35suite仍34pass1fail（45.73/45.80s），child编译cache未隔离是测试自身缺陷，不能伪称全套通过。
  独立自定义guarded实际tiny80完整/0-20-80普通load/注入save-fsync-load失败留partial且无endpoint/report/原binding恢复已核；
  xb非原子但私有owned路径失败不得发布成功，不新增ctypes/audit旁路或放宽nlinkguard。
  新修复另冻soft600/hard1200处理上述反证，保留旧1234s超hard34和原suite失败，修复中只许TEMP/合成CPU。
- 05:00:53Z起、hard截止05:20:53Z的修复任务在会话恢复后不可联系，未收到完整测试或稳定handoff。
  05:21:27Z排他保留 `/tmp/r7_fixed_case_repair_interrupted_snapshot_20261007/` 三文件快照与interruption_receipt；
  driver/test仍为初版，support部分修复SHA `a8638cee505ace5be7c40260fd2f151498c0d26efe50293474bf8bfb386acf4c`。
  该阶段标 `interrupted-unqualified`；精确停时/elapsed/soft及hard超量未知，不编造PASS或费用时长。
  新独立TEMP修复另冻planned600/hard1200，从新执行者实际接单时钟计，保留旧阶段，不reset旧deadline。
  只闭完整contract/RNG/实测梯度活动与AdamW/report/trace及child cache，原guard和非原子partial留痕不改。
- 新TEMP修复05:22:26.749104Z起、hard05:42:26.749104Z，stable receipt于05:43:34.756668Z冻结；
  whole1268.007564s、soft超668.007564/hard超68.007564，标partial-hard-budget-overrun-not-qualified，不reset或追认。
  `/tmp/r7_fixed_case_repair_newstage_20261007_c7a9ztro/stable_handoff_receipt.json`
  SHA `74d12e81ad533545b2d8d7fa75190a7fff038892a5630ff1c78a3b84dcce2ff8`。
  稳定driver `1d0378cb70cba7319981f818bcf636c8a9e8f8c98ec2a83252847e413c3afc8a`(373行/max92)，
  support `2062f8b380ef93e994cbff4f9e26b576f3239680267fb24d95815956320e6cf0`(516/max94)，
  tests `ea1c5368958403b3ff0d9ebedcb411da207476b09303de863f320e408dad2588`(457/max52)，
  新negative tests `a9104aadbd8b2e616da0a525bb68b2303e17057a4ba4160e6950b6d1590d42a0`(310/max47)。
  实际35全过、完整35+51共86过/无skip；最后仅watchdog childcache编辑后最终35全过，最终完整86仍待独立实跑。
  原guard tinyProcess80与fresh CPUreader已做，实测81active/4unused非生产141/14；额外guardedfailure进程被deadline终止并reaped，未完成不算通过。
  `train_cached/read_training`新增必需expected_contract；完整template/finalbody、RNG/AdamW、逐行实测活动与独立累计及精确目录出口均已接入，尚待独立资格。
- 独立复核准备stage soft180/hard360，receipt冻结372.154166s、hard超12.154166；未含此前开场与最终report，精确总时长未知。
  `/tmp/r7_prelaunch_recovery_prep_20261007_nso0in66/preparation_receipt.json`仅路线/合成mask helper，未判资格，保留预算违规。
  稳定新包另冻独立qualification planned900/hard1800（含tests/读取/endhash/报告），无实际source/GPU/outputs访问。
- 文档恢复提交 `39a7aba7a0a037dd0fb5e5e3fa7823d8d3123040` 精确主CI37576366667 completed/success、全部必要steps success；17标签门控实验skipped不是实验PASS。
  治理79 tests通过、索引58条及brief同步，旧gradient/same-case证据SHA与前54行index前缀不变；工程检查非科学接受。
- 未做：实际source/heldout字段/GPU/80真实updates/新真实checkpoint/结果/费用/endpoint重算；科研目标未完成。
- 新独立qualification实际完整86passed/0skip，但独立coherent合法nativeCPU RNG替换及端点声明/checkpointbytehash一致改写仍被reader complete接受；
  `/tmp/r7_fixed_case_independent_qualification_20261007_y7_b0pq9/independent_owned/essential_blocker_rng.json`
  有有效正例且非缺receipt假拒绝。该stable包有essential blocker，真实launch继续禁止；独立复核尚在原900/1800内收尾。
  独立native hooks又确认：真实unused参数在全部80行mask/count、optimizer及端点一致改写为active后仍被接受（81→82），也是完整同权活动证据缺外锚。
  qualification冻结receipt SHA `23d0805dd8cd60fc938b84d8ef097804b7ba74b801f6d34fa0f39c39fadb23a9`，result SHA `ad0251b88c7ae3b2c03d7ef9519b9d417ee469074b44b03508eb3dfb4c76840e`，
  whole1671.521s、soft超771.521/hard超0，原900/1800不reset；全部86/旧16/guarded80及freshreader、保存失败6例/unsafe路径/driver失败费用出口实跑。
  后续仅修2项：把独立seed41 native RNG状态外锚到CPUprepare/finalcontract，逐更新核不变；把已冻结原case0逐参数total_unused作为预声明活动mask外锚，逐行实测核。
  tiny CPU从独立初始同目标autograd推导活动，不能硬码生产数量；真实mask变化必须failed保留，不放宽。不能抹空CUDA、只增加同权自洽hash或声称OS敌对producer保证。
  新最小2anchor修复另冻planned600/hard1500，实际接单开始含全部tests/末hash/receipt，硬截止前留90s；旧partial/negative不追认。
- 最小2anchor新stage06:18:46.601222Z起、hard06:43:46.601222Z；06:39:16.720434Z冻结whole1230.119210s、soft600超630.119210/hard1500超0，硬余269.880790s。
  receipt `/tmp/r7_fixed_case_two_anchor_repair_20261007_ujfp0mdb/stable_handoff_receipt.json` SHA `c06491e29e669996932376e9d759fadced09b4b9bc23fa482fb0a9e1b90dfca8`；旧partial/negative未reset。
  稳定driver `3073ea784dbd76a44c1f88adb1329c66ce120194235e18120425029467da97f8`(375/max93)，support `7e4540b42debde4befdda247edcbc4c18b81af25c0fa862aa5f6d297bf75ba86`(564/max97)，
  tests `bbe4df7cf7d1f92e33770ead0a536880fd26e96d778d240faffe6ecdee0caf42`(466/max52)，negative `d80d6f068ba0a9e2d0ec273c7f60e3d713c7b89bf3ff854ea0a442b8dbda4ccc`(403/max47)。
  最终同pins完整91passed（原86+新5）、0fail/error/skip、无过滤，process182.724s exit0/reaped；原guard tiny80/freshreader动态81/4齐。
  `build_contract`新增必需activity_anchor；native41与originalcase活动入fulltemplate外锚，runtime变更failed留partial；实际CUDA/数据链尚未核。
  新稳定包已委派独立定向requalification planned600/hard1500，完整91及两完整coherent反证/外锚接线必核，不把owner suite当独立科学裁定。
- 双外锚进度文档提交 `14b94b203303099fb92420f39ad831a026ac00fc` 主CI37581732547 completed/success、全部必要steps success，17标签实验skip未算PASS。
  用户.zcode/config.json与.zcodeignore hash未变；另有外部修改的.zcode/agents/web-researcher.md未纳入本任务提交。
- 新定向独立资格已冻结：`/tmp/r7_fixed_case_targeted_requalification_20261007_p9cqpkcw/targeted_qualification_receipt.json`
  SHA `a96525b108b7921e1479108d5ce81bc8ffed49b8bb2cd7a5a058552f9b0fc0c4`，result SHA `5a4c92fe77a2725e33840cdf3cd0fbb933b137cb939c074a9ba9e83cece9d377`。
  最终full91/0skip/errors/failures、190.072s进程exit0/reaped，两完整coherent及同权body改写拒绝、真实RNG/mask变化partial留痕，24endhash未变。
  whole1032.230798s、soft600超432.230798/hard1500超0、硬余467.769202；限定四newpins合成CPU工程qualified，不含实际数据/CUDA/OS敌对保证或科学资格。
  主链`/tmp/r7_fixed_case_prelaunch_qualified_20261007.json`组合前完整审阅+新定向闭合，旧negative/partial不追认；runtime真实身份和nativeCUDA对模板仍必核。
- 独立端点readback TEMP准备实际50passed，但完整deep-parent外锚happy-path尚未资格，不能实launch。
  `/tmp/r7_fixed_case_readback_stage_20261007_4ys7o7nv/stage_receipt.json` SHA `f514aed61e9ca0980ba3075932959348fdba4045dc9d54d9d18d0c06fa8237bf`；
  whole2386.537055s在1200/2400硬内，soft超1186.537055，但收尾仅硬余13.462945s，违反预留90s纪律，保留偏差/旧四失败。
- 下一动作：新排他attempt01依原3600/7200先冻两相protocol并实核数据/CUDA，再恰80更新；readback在自身完整资格前仍禁真实运行。
