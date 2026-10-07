# S3：train-only valid-time 气候态锚与异常反馈小试验

<!-- round-node: S3 -->

状态 active / 准备，2026-10-07。唯一主计划 `docs/goals/main-model-climatology-campaign.md`；继承等曝光ordering实例已登记 negative 的下一动作，不重跑或扩展该实例，不进入S4，不自行宣布最终goal完成。

## §0 Objective（单段，实测1457字符）

> 在 docs/goals/s3-climatology-anomaly-anchor.md 接续S3/index62/22.6281GPU-h、test评分0/r0。D1冻结不同主模型表达假设：原process/backbone权重与开关不变，显式train-only valid-time输出锚Y0=Cvalid+A0(history)、RW-Bproposal亦锚Cvalid、feedback E(Yk−Cvalid)，物理Xt/typed输入及最终YK推动absolutehistory不变；这是package筛选非单机制归因，不复活ordering/旧scale。D2 default-off参数/state/RNG/逐操作兼容，模型外一次拟合原2400train/16月小时桶，immutable persistent normalized表与source/train/grid/units/norm/counts/digest完整绑定，缺桶/非有限/身份不符拒绝，无forward I/O/refit/真值/heldoutfill/atmos_baseline偷塞；固定/streamed/adaptive/calibration/oracle及PE-off物理日历贯通，内部K固定C。D3完整CPU回归与独立反证后，用旧归档code.zip原普通loader资格导出model-only权重，逐key/shape/dtype/byte迁移、新code/完整contract/普通checkpoint；不绕旧digest、restoreoptimizer/RNG/cursor。原shared0+interleaved80合法pins复用不重训；候选0另评分。D4新排他protocol/outputs、seed41freshAdamW/global80recipe/clip1/原12步deepK/K4/FP32/interleaved原四train各20曝光，固定endpoint0/80各原四train+四dev/all17×五lead×三区域、原train气候态，必要开发支持先冻、negative止实例不自动加seed/dose/fullval。D5独立savedfacts/math/identity/process/gate/wholecost首末hash，全部失败/正负冻证据/index/brief/master/branch精确CI，科学门仅docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md，不宣泛化/因果/候选或最终接受。准备soft3600/hard10800/reserve180、完整CPUsoft1800/hard5400/reserve180、实际GPUwholesoft5400/hard14400/reserve180先冻，soft超继续记、hard/真错停attempt留partial与全部成本。GPU1默认共驻启动/spawn即时UUID/余量max(known2491416576B,newpeak)+2GiB，不signal邻居，无总GPU-h上限；旧data/store/archive/outputs/冻结科学门/用户配置安全凭据不改，不paid/rental/exclusive/main/release/issueclose/force/mirror/破坏性/cron/会话外续跑，不自行宣布最终goal完成。

## §1 起点、证据与实质差异

- 起点工作分支 `r7/weather-reasoning`，HEAD `2af4e825f70c650536622d3d5726b7895e484289`；其精确主CI `37676532474` completed/success，唯一pytest job `112981363638` 与全部12实际步骤success。工程CI不改变负结论。
- 原ordering证据 `docs/R7_S3_CASE_INTERLEAVING_PILOT.md` 冻结于 `9dde48de1666de88035e2c677f20c70ede33ed48`，SHA `7043fa608de5a74d1ecb2b10c52475419e5f937278460248a8dd6c65d85a63bb`。index62/累计22.6281，cap20/remaining−2.6281仅会计；test评分0/r0/S4未变。
- 四train可同时有限拟合，但 interleaved pooled开发必要支持false；该实例已停止。没有证明遗忘、clip或容量是原因，不重训blocked/interleaved或重复旧Jan80/replay。
- 现有decode从Xt出发，`model/weather_forecaster_r7.py:95`；RW-B两proposal亦Xt，`model/process_step_r7.py:131`。新C是合法train-fit、已知valid-time可query的channel/grid字段，不是旧d_c/s_c固定尺度，也不是精确Xt−C补偿后的坐标改名。新的先验、feedback与proposal同构成性能package，不独占机制归因。
- 原全部开关包括local_solver/known_context/spacetime/draft_query/roles/source_position/readout；本轮保留。改变typed/aux/loss/clip/LR/PE/solver/K或选择赢家均非本实例。
- 既有climate由 `data/r7_evaluation.py:91` 按train membership拟合，`:136` 以原norm转normalizedFP32。2400 stamps/16桶各150，physicalFP64 mean identity `c633df585b1894704d942e24305047a74ff7af2460cbc24ca2ebe7be1948a205`；不等于全年覆盖。

## §2 交付物清单

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 范围、模型表示和独立停止出口写定 | 决策0040、本goal/工作态JSON/独立内容审阅、原控制与新候选身份 |
| D2 | default-off active模型与全部入口接线、immutable表 | 新source digest、解析/投毒/错calendar/缺桶/checkpoint反证、完整CPU JUnit/独立固定包资格 |
| D3 | 归档资格与显式model-only迁移/新contract | 旧普通loader/外锚、每key/shape/dtype/byte receipts、新buffer identity与普通load roundtrip；控制score复用pins |
| D4 | 真实80更新与候选0/80固定原八例评分 | 80逐步receipts/四case各20、16新pairs/trainobjective及eval物理17×5×3全表/undefined、无test |
| D5 | 全事实独立审阅、冻结登记与整轮成本 | first/end/source/数学/scope/process/gates/cost receipts、证据commit/index/brief/ledger/精确CI |

完整年度/完整val/3seed正式确认不是本轮交付。没有结果不写有效；CPU/CI不能替代天气支持。

## §3 判据与前瞻开发读法

最终科学门仅引用 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`，本轮无科学确认或容差变化。有限开发支持采用原ordering必要支持的同严格结构，但比较对象适配新候选并在任何真实评分前写定：

1. 候选80 pooled四dev的t2m/full五lead MSE均不高于**原interleaved80控制**及**原shared0**，至少同一个t2m lead严格低于两者；u10/v10/mslp/full五lead对两者均不高于。所有cell必须定义、有限；undefined不能支持。必要支持不等于正式候选或科学接受。
2. 候选0/80逐病例原目标、最终物理分数、原控制0/80和气候态完整报；不按val选择endpoint、不跨变量单位平均、不用train拟合代开发门。候选0须重新预测，不能套旧0分数。
3. unsupported登记negative/mixed并终结此anchor+anomaly实例，不自动追加seed/dose/开关/完整val。若有必要支持只允许另立新协议讨论扩开发；不默认开启S4。
4. 原控制0/80已被固定资格并同数据/病例/recipe；字节pins+新default-off语义资格后复用，0新增控制训练/评分/费用。共同初始化指原训练参数逐字相同，不指新candidate0 forecast相同；C是新增train统计信息，储存/query/算子成本分列，不把同参数量/updates当同FLOPs。
5. table只fittrain，源身份/单位/norm/calendar/grid/桶/finite完整绑定；缺桶与错误不能以警告、补零或heldoutfit继续。spec-only constructor分配persistent placeholder且UNREADY，forward先拒绝；只有一次显式verified install或普通JSON factory后顶层strict load_state_dict递归核表/hash/shape/FP32/ready可恢复，不接受子模块专属load override绕顶层或缺失/篡改表。全部新pt采用新code身份，普通loader不改。
6. 本候选完整model配置必须绑定 `anomaly_feedback=true`、spec存在且forecast feedback开启；anchor-only `false` 只保留显式接口兼容，不是本轮额外臂；true但缺锚或禁用feedback直接拒绝。

## §4 顺序与依赖

1. 核ordering精确CI/旧页与index62/两个campaign；修订planner草稿并结构校检，独立源码绑定内容复核后冻结。
2. 实现explicit default-off table/backbone/ProcessStepInput，C与Xt分离；固定/streamed/adaptive及校准/oracle贯通，物理calendar independent of PE。
3. 冻结CPU协议后新完整测试、全仓CPU与wheel、独立反证；skip如实nonpass，不删弱测试、不增规模例外。源码变更后归档工作分支新code.zip/commit与digest、精确CI。
4. 新TEMP wrapper复用已qualified旧数据/指标/recipe模块；旧loader在归档子进程读取pt，导出纯model参数与receipts。新实例完整mapping、buffer规格、新contract/普通pt资格；不把r7_parent_import旧专用迁移当通用接口。
5. 实际protocol在source/data/tableprepare之前排他冻结，CPUprepare完整sourceSHA/preflight/BUILD_COMPLETE/window/norm/case资格，train-only气候态复核原mean identity后安装。GPU统一父调度：train candidate80、score candidate0/80、read/inventory/cleanup。控制原分数仅强hashreuse。
6. 独立完整保存事实/数学/身份/模式/calendar/scope/process/gate/peak/cost首末核、全部失败留存；新证据freeze/index/brief/master/branch精确CI齐后新主线动作。

## §5 数字预算与停止

| 阶段 | 预声明数字 |
| --- | --- |
| 源码/包装准备whole | planned3600 / hard10800 / reserve180秒；开始即留固定clock，不能reset |
| 完整CPU回归whole | planned1800 / hard5400 / reserve180秒；无过滤，ownedchild与JUnit/源首末/cleanup计入 |
| 独立prelaunch审阅 | planned1800 / hard5400 / reserve180秒，独立固定source包，失败保留 |
| 实际candidate whole | planned5400 / hard14400 / reserve180秒，prepare/fit/migration/train/score/read/inventory/reap全含 |
| 训练 | candidate80/batch1/seed41/原四case各20/interleaved，global80/freshAdamW/FP32/full12/K4/clip1 |
| 评分 | candidate0与80各原4train+4dev；17变量×6/12/24/48/72h×full/interior_1/edge_1，trainobjective/eval分别 |
| GPU | GPU1 UUID `GPU-9d1624af-9d77-aa7c-0620-b6cb778f4ced`；启动与每spawn即时free≥max(2491416576B,newreservedpeak)+2147483648B |
| 余量不足 | 该wholedeadline内最多1800秒有限等待，每30秒只读query，不降门/换独占/抢邻居；届时停相关attempt |

soft超继续并记overrun；hard−reserve不再开新工作、ownedchild截止/reap与收尾在hard内。真错/identity/finite/scope/coverage/mapping/prelaunch失败停attempt，全部partial/source/成本留，不resume/retry到通过。准备工程失败可另独立修复、冻结新bytes及资格，不追认旧stage，也不能运行中增hard。无总GPU-h许可上限。

## §6 planner草稿与事实复核

实际planner返回有围栏JSON，但0工具读取、仅占位file:line，426字符objective超过400，原结构verified:false；还臆造`arms/interleaved`路径、全年日桶、constructor训练fit、mismatch只warning/skip、只定向tests且未明确新普通身份，均不采纳。原回复保留 `/tmp/r7_s3_climatology_anchor_planner_raw_20261007.json`；主链实质自规划 `/tmp/r7_s3_climatology_anchor_plan_20261007.json` 走相同契约verified:true/fencefalse。不是“planner成功设计”，也不修改provider/用户配置。

主链核两处源码：backbone实际用history最后帧作base而不是不存在decode方法；既有fit按month/hour而非365日表；实际控制目录为 `outputs/r7_s3_case_interleaving_20261007_attempt01/interleaved/`。Explore另定位所有shared-step/adaptive/calibration/oracle/streamed入口；修改清单以真实接口为准。科学判据未外委。

旧全仓CPU harness可作模板但hard/reserve只记录不用于终态、未绑定完整工作树且只管directpytest；新独立runner须固定新预算/实际源/完整JUnit，并如实限定对子进程的控制，不冒称OS沙箱。旧结果不回改。

## §7 明确不做

不重跑Jan80/目标—最终诊断/ordering或旧精确replay；不改旧判据/失败/科学合同、protecteddata/旧store/outputs/归档；不下载/发布新数据、填缺桶、源码只读镜像import；不把futuretruth或val统计送forward；不偷塞baseline/绕digest；不改旧loss/clip/K/PE/aux/typed开关、不做seed/dose网格；不付费/租卡/独占/main/merge/release/issueclose/force/mirror/破坏性，不signal邻居、用户安全/配置/凭据不动；不cron/daemon或会话外续跑；不S4/test评分/确认r或最终goal完成。

## §8 进度与交接

- 起点2af4e825精确CI37676532474已完整核；ordering实例negative已终结，原科学/旧证据/index62/brief/三个用户配置hash未变。CI metadata和parser另排他归档 `outputs/r7_s3_case_interleaving_registration_ci_20261007_attempt01/ci_evidence.zip` SHA `4c85dcc3ad5f6f50414beda7424275e2156ee4f1abead7c20de4edaaf7d3ea3f`，41文件/4653722输入bytes/0.112468秒/0GPU/0网络。
- 本方案原planner草稿拒绝，自规划结构校检通过；具体入口Explore只读完成、两处源码/旧controlmetadata主链核实；尚未运行模型/训练或新candidate评分，新增GPU-h0，账本22.6281/r0不变。
- 首设计独立审阅为 **NOT_QUALIFIED_MANDATORY_ISSUES**，receipt `/tmp/r7_s3_climate_anchor_design_review_20261007_attempt01/receipt.json` SHA `0cdba1a14f1593f97389f288405cce34e03bd0f597b4e7b15fc29505e23c59e0`；655.846578秒/soft300超355.846578/hard900超0/reserve90，五assigneddocs首末60261bytes逐字同。M1自规划来源措辞、M2calibration/oracle清单漏项、M3UNREADY/install/普通顶层递归restore已最小补文档/JSON；旧审阅仍不qualified。主链过早并行可逆implementation导致8源码excerpt漂移，不能称稳定source审阅；实质启动必须等固定source新prelaunch，失败不normalize或追认。
- 最小设计补充另固定180/600/reserve90，仅三文档M1/M2/M3资格闭合，status `QUALIFIED_LIMITED_DESIGN_CLOSURES`、mandatory0，receipt `/tmp/r7_s3_climate_anchor_design_supplement_20261007_attempt01/receipt.json` SHA `f8887280f7a802a24d290f9b619b5d395f36c8b3c812c271700e39e802d2ddff`；111.929543秒/soft-hard超0、三assigneddocs首末32005bytes相同。仅文档closure，不替代固定source/CPU/prelaunch/科学，也不追认原review；之后只按minor建议修ADR工厂call加实际kind参数，不改判据。
- 活跃table/core和新CPU全套runner/TEMP实验包已分别独立分工实现中；无新weather/GPU/真实candidate评分。资料 `docs/R7_S3_CLIMATOLOGY_ANCHOR_SOURCES.md` 官方pinned源码HTTP200核实时间差分残差，否定首检索错误“GraphCast仅absolute”摘要；不以外部资料称本方案有效。
- table接口已稳定278行/max57、SHA `2e12d46dedf2cb7c4155d5292b0668ca05ab5f0d2cc8aba11a49a64dc6a900b5`；新模块完整CPU144passed/0fail/error/skip，finalreceipt SHA `bd3f4855b3bf7f0d383ced4c5c9f37236fa220d2eb4dfdda55b0e834c1cde3ad`。首bootstrap socket类继承失败及独立fixture误判leap Feb29归March保留，修自身bootstrap/期望后完整再跑，未改日历算法/门。新core固定路径43passed/0fail/error/skip，8.492190秒、source首末同；之后8test名字仅snake_case、bodyAST相同，finaltest SHA `d54b34f1af62f3ecd65f1580164f45fa9b9b833e769872eb7212d7fb7e2233a8`，改名finalbytes由下一完整CPU实跑，不假称已原样执行。新model digest `9ec902d8bec395cde5b5632cfbc37d3626b79d53b848612f96f7e8e8bcf87cdb`，原普通loader与旧digest未改。source≤600/function≤200，constructor197，AST测试1910/5113新增43/121及C类marker同步，不放宽阈值/例外。
- 新fullCPU runner完整合成49passed/0fail/error/skip，16.793078秒，source SHA `85e30af4fa84e1905a786de01d1bea7513015325d7f126e03c413322bdafc484`；路径 `/tmp/r7_s3_climate_full_cpu_20261007.py`，receipt SHA `9ed4099e36972ae43f6ae6a7a65c50165e310d01c6a2cbf76f8dcf03b950ae13`。真实全仓运行输出拟 `/tmp/r7_s3_climate_full_cpu_20261007_attempt01`，soft1800/hard5400/reserve180，协议冻结代码/dirty source/完整JUnit/wheel与首末、每skip列nonpass；不冒称OS隔离或所有grandchild生命周期。
- 首完整CPU/wheel已实跑4190passed/9skipped/0fail/error，whole1194.952492秒、soft1800/hard5400/reserve180、overrun0、所有被测source首末同；attempt SHA `a1c9e38cb554e78b75be931f8ea4f8ca03a9ce4c904cc7f40663fd8f12e2b405`。六CUDA/三可选真实fixture skip逐项passed:false，`all_paths_pass:false`，不以工程完成覆盖独立发现错误。
- 首固定source独立核 **NOT_QUALIFIED_MANDATORY_ISSUES**，receipt SHA `16284e6a0ed1ad329a74d9e1c82dfee5bd8ac7466ae1104b298e9c1ff85bc116`，491source首末同、完整新模块187passed/独立29passed/关闭feedback4failed，0error/skip、全owned直接childreaped，wholeverification1025.831秒/1800/5400/reserve180/overrun0。继承HEAD缺失local_solver per-call False guard；本候选anomalyTrue已有拒绝，但advertised anchor-only接口也须正确。原失败不追认。
- 两原source hold结束后仅Process forward/sharedstep补generic相同local_solver反馈拒绝，覆盖K0/K1与四solver分支、表on/off；24新增反证未修前24failed/0error/skip，修后连两原模块211passed/0fail/error/skip。首TEMP收集受/tmp旧包污染与第一修复guard错误优先级5failed均保留，仅修源码guard顺序，不改任何已有测试/断言。新model digest `d3fb58dbd0ed9efbd249fc258cb09c488d543c0a8ad77dac5689ac8f9ba77bab`，普通training digest `6e4363d5707ea4935d5449c800e4fecefa99bf553b42a747dd4dad37f8d548db`未变；AST1912/5116/185Python测试文件，C类markers不变、硬限不变。完整CPU与独立审阅各另固定1800/5400/reserve180新attempt；准备原start6178766.532052265/hard10800/reserve180未reset。
- TEMP包已固定 `/tmp/r7_s3_climate_anchor_impl_20261007_attempt01/`：manifest SHA `377fff2dae173b23e941ad49f8f45ef71ac5f095fd735e33e8fae8514e54d9e5`，pins SHA `ad05dc5b0d7c95a6dfaf8245c320351712ec9a2aa483fd18415729692d14832c`，source ZIP SHA `ae0a67805bb44dfe63a5a0940a933560b087950c4786bb51e8fa22c9ae70f19f`；最新完整24tests0fail/error/skip、whole104.498391秒、tinyProcess80/all155/16syntheticpairs/真实ownedparent TERM childreap反证/outsider自然退出，receipt SHA `45923894ddea080de25ba97d944f7430767e9fddd6558455a777612ab9115f18`。旧20/22/24attempt及partial反证留存；只用predatesguard的synthetic dirty archive，不能冒称production新归档资格/真实fit迁移。launch supervisor与独立terminal savedfacts helper仅TEMP准备，未actual launch。
- 修复后完整CPU/wheel真实attempt02完成：4223cases/4214passed/9skipped/0fail/error，无过滤/deselection；六CUDA/三可选真实fixture逐项nonpass，all_paths_pass:false，wheelpassed，owned子进程reaped、源码/config/status首末相同。receipt `/tmp/r7_s3_climate_full_cpu_20261007_attempt02/attempt.json` SHA `94ddaf30ddf44f578f6f484c99f650b13de6aba5875cd06b7196c92bfc55dc54`，protocol canonical `a2e93fbe916b6257ac41cc0c12cb033dc044467fe343d7ee7ca27e98137a5b9a`；whole1213.055803秒、soft1800/hard5400/reserve180/overrun0。
- 修复后固定source独立attempt02 **QUALIFIED_ENGINEERING_ONLY**，receipt SHA `2d77301884206f530c7cc4e6a7437c02adcb0d55e757526823d51d40404966d8`；211完整三模块一次、旧29独立与原4失败反证、新62反证共306/306passed/0fail/error/skip。492source+41config首末同、7owned直接worker全reaped无signals，whole1174.106421秒/1800/5400/reserve180/overrun0。自身日志open与禁forwardI/O冲突、专属ROOT复用碰撞、补充协议函数数错的三个failed/unaccepted阶段留存，分别修自身harness+全同cases新排他实跑，不回改旧失败/测试。table公共buffer可变/coherent spec不证external truth/wholeload非事务/禁网非OS沙箱等限制未消除。
- 原failed审阅/红反证/首CPU及修复后资格分别新排他归档 `outputs/r7_s3_climate_anchor_engineering_20261007_attempt01/`：`preserved_failures_and_original_cpu.zip` SHA `ce641c710030b354f2a58c9353a93eabd9d9147406c84c7d895ce634ec75bdb3`（509files/10819120inputbytes/0.532172秒）与 `repaired_main_qualification/repaired_main_qualification.zip` SHA `0c823c6dd7f0d787e72c39cf06731af300e2258e22b3c1322396ec658ad9ac50`（521files/10859408inputbytes/0.546367秒）。均0GPU/0网络，不改任何oldreceipt/原数据/用户配置。
- TEMP实验包首prelaunch独立反证 **BLOCK_NOT_QUALIFIED**，receipt `/tmp/r7_s3_climate_anchor_prelaunch_20261007_attempt01/receipt.json` SHA `13bae165fb2db8d745d925630b839cb551ed108371b411f5632f052b6b0c12b7`，whole1150.411527秒/1800/5400/reserve180/overrun0；固定35source/80ZIP成员及487activePython首末同、全owned childreaped。复制完整24syntheticpassed/0fail/error/skip与独立58cases/52passed/6failed均保留；112paths/1344queries/5376内部K的toy消费覆盖不抵消实际守卫失败。B1 worker绝对路径含`..`可能写越界（只调用guard，未outsidewrite）；B2每spawn gate缺rawquery/时戳/查询至Popen年龄门；B3CPU普通checkpoint验证丢弃native optimizer事实；B4完整源/preflight/BUILD_COMPLETE事实未存；B5standalone parent未绑定fullCPU/TERM资格。旧包和审阅保持未qualified，独立hold结束后新排他B1–B5最小工程修复在同原准备clock内重新冻结全部bytes/反证/资格，不改candidate/loss/门或假字段。
- 未做：生产新commit/code.zip精确CI、修复实验包prelaunch、真实权重迁移/table资格、新candidate0/80及独立终态登记；无获益证据，新增GPU-h0/test评分0/r0，账本22.6281不变。
- 下一动作（S3）：归档修复后主模型工作分支精确code/CI，独立闭合新排他实验包B1–B4与完整prelaunch；显式迁移前置齐后冻结实际protocol运行唯一candidate筛选，保持已写定开发停止线。
