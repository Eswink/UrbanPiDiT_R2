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
- 未做：实际source/heldout字段/GPU/80真实updates/新真实checkpoint/结果/费用/endpoint重算；科研目标未完成。
- 下一动作：闭合独立prelaunch真实反证/最终全套资格；任何essential未修不启动训练。
