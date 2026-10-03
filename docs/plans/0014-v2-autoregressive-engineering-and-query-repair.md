# 0014 V2 独立自回归工程与显式草稿 query 修正

日期2026-10-03；起点 `edcc33539178f95646983eca3084f7c16c241692`。
状态：**工程实施中，真实B/C未执行**。持续工作态见 `docs/goals/v2-remaining-stages-exploration.md`；
此计划记录实现窗口与规划纠正，不是预报结果或最终goal完成声明。

## 来源、授权与判据

主步骤来自planner委派，主链纠正草稿后保存 `/tmp/r7_v2_remaining_plan_20261003.json`；
query修正版 `/tmp/r7_draft_query_feedback_plan_20261003.json` 实跑机械合同verified=true，
fence_stripped=false。规划工具只提供拆解，不选择科学阈值，不把JSON或green CI当实验进展。
权限0030/0032允许本方向普通工程、假设/预算与节点自主执行；保留新数据/发布/付费/独占/main合并
及破坏性边界，旧M3/any-unresolved/paused/advance=false均不变。

科学与接口来源：`docs/R7_MAIN_MODEL_V2_DESIGN.md`、ADR0023、计划0004的Evaluation protocol/
Scientific gates；B沿原L6深监督作真实L6+0.5L12物理展开；C的前瞻schema与primary/tolerance在
`docs/R7_V2_CONFIRMATION_PREREGISTRATION.md` 写定，并须在独立protocol运行前冻结。

## 实施顺序

1. 对表新N3主计划、指定材料、真实source/既有cache与四个父资格；aux_off实际RW-A/K4优先。
2. 实现完整matched Generic，只省过程诊断head577参数，所有reader/solver/gate/proposal映射保留。
   默认关闭保持旧Generic逐位；未来truth继续在白名单外。
3. 原严格归档身份接受后显式权重复制到当前模型，新optimizer/RNG/updates；不修改旧loader拒绝。
   exactint64 t+12、same-train-split185窗口及唯一边界排除预先声明并两臂共用。
4. 实现独立grad-enabled两步训练，第二history为第一finalforecast，两次lead6；内部K不推进时间。
   每个physical step保留initial/allK normalized deep supervision、final_weight2，组合L6+.5L12。
5. 修活跃Gregorian接口以显式init_calendar_year表年界，保留未给字段与旧init_year不读路径。
6. 核D4现役read→Z隔离与梯度；独立审查确认严格direct draft→query缺口后按ADR0034新增default-off
   开关，Q=LN(C+E(Y))+pos，无新参数；Process/Generic共享接线、固定P/C/pos直接反证。
7. 对新驱动/worker/度量/结果做实际跨模块CPUfixture独立审查，补每轮必须报的persistence与climate，
   保留所有17变量/5lead/3region/病例与真实pooledACC。CPU测真正objective forward+backward countedops。
8. 源稳定后完整CPU、规范/节点/goal/index/brief/编译/空白与精确工作分支CI；再独立真实FP32/BF16
   工程探针900soft/1800hard，freshCUDA0/0，checkpoint-body resume范围明确。
9. B独立冻结5400soft/10800hard：seed41/42，各200L6/200two-step/400equal-compute L6，6train+30val。
   源/父/数据/window/sidecar/protocol/outputs强核，whole从最早准备至cleanup，GPU从首spawn到末reap。
10. B全指标/成本/失败与协议选择冻结登记后，独立C的3arm×3seed×400updates，9train+135val
    K1/2/4；actual固定准确率—成本前沿按新predeclaredgate判adaptive，未满足记not-started。
11. 每节点登记、独立审查与精确CI后自主推进；逐issue充分证据支持verdict才Closes、非force ff
    main与匿名终态核验。不能科学失败写positive、cannot-distinguish增加seed练到赢或自判goal complete。

## 与planner草稿的差异

- 首稿误把reader/solver当可省诊断部件、误父为RW-B、父路径不存在、把protocol入git及余额作权限门。
  主链依实际源码/产物纠正：完整结构、RW-A/K4、独立outputs排他协议、无总GPU-h门。
- query草稿含不存在的 `model/positional_process_readout.py`、错误LN(C)+LN(draft)公式、Generic实际
  传None、allclose称逐位、假想state键/strict加载拒绝及不存在ruff/mypy CI步骤。全部未采用。
  实际readout路径是 `model/process_readout_r7.py`，开关两模型都有效，Q=LN(C+draft)+pos，默认用
  torch.equal。原loader根据源digest仍拒旧checkpoint；新无参数开关的架构拒绝靠显式contract，不伪
  造state shape不兼容。主CI以实际ci.yml九步骤核，不虚构检查。
- 独立审查发现累计K单lead列表/整数接口不一致、无aux child sidecar override非法、实际heads/dropout
  配置未全核、roles+noFeedback不合法以及persistence缺项；修活跃实现与直接反证，不放宽判据。
- 单文件超过600时按职责拆support/helper/newtest；不新增例外，不通过压测试空白掩超限。

## 验证与停止

完整pytest不删case/断言；CUDA条件与真实fixture跳过如实列，不算通过。actual worker错误/身份拒绝/
余量不足或硬截断停止该attempt，全额保留；修复另立协议/输出不复活。软预算超继续记overrun。
B阴性只终结对应hypothesis，独立C/必要工程与收尾继续。新数据等保留权限只停止相关动作。

## 实际结果

- **已做**：路线0032/N3同步；source只读preflight240×17×65×65；精确185/186窗口、唯一边界排除；
  0033日历修正、matched Generic、显式parent importer、两步数据/loss/runner、evaluation与薄驱动/结果
  初版；GraphCast一手固定clone/来源登记；D2标量0GPU解析探针真实完成。当前不是B/C预报进展。
- **实际验证**：首次全量2614passed/10failed/9skipped/2warnings，273.61s，不冒称通过。
  主链新输入/Generic/时间定向169passed/16.61s；D4direct33passed；显式导入专向73passed；runner相关
  91passed；evaluation相关102passed；driver60passed；precision CPU/fake31passed。并发审查提出的
  实际缺项仍在合法修复，上述数字各自范围独立，不拼成最终fullsuite。
- **父默认兼容**：独立/tmp工程协议e64acba3…、两aux_off父×65网格/奇数9×13四固定合成case，
  archive旧严格load与当前显式import final/all5drafts全部torch.equal，131key state hash同；主链重核
  109产物hash无差。receipt文件SHA `56acdc7ec1d7aade7a734d039150a24858210393bc2f389f0444641a72c5a794`，
  actual `/tmp/r7_parent_equivalence_6vuw79gx/`。首轮venvsite路径过滤失败完整保留，不回编成功。
  此等价只覆盖无calendar的同机FP32合成工程case，不表示天气技巧、calendar-enabled旧等价或GPU逐位。
- **当前机械对表**：37阻断未见已跟踪违规；campaign0fail/4历史notes、goal0fail/index18records通过，
  compile/whitespace通过。未跟踪文件stage后还须全核，不能把tolerated当提交可过。
- **最终源工程冻结**：query相关234passed，before/after四default case输出/grad/weights/RNG digest同；
  语义/role/query有限独立复核通过，新增query测试仅合规改名、字节不变。driver77passed/8.61s；
  evaluator104相关、precision33 CPU/fake、frontier125；results最终81passed/380.20s，含实际格式C gate
  接入、baseline及code_commit/status身份反证。各范围独立，不拼作全量测试。final模型digest
  `0cc9c16e7a12bf23150fb04e13acb72f548f8cb461ad64df1d45123b387a223e`。
- **最终父身份桥接**：新120soft/240hard协议 `d530320a…78d13`，旧严格loader/当前显式导入四合成
  case final/all5drafts仍逐位相同，109产物pins主链再核；13.803s/0GPU/0overrun。回执SHA
  `4a95a56b2cde9a5ccf703fcc33153636ec2b53b9bc79a0c62939b57faaa3cd23`，原路径/tmp保留，原字节
  映射留存 `outputs/r7_v2_remaining_acceptance_20261003/engineering_parent_final_bridge/`。
- **规则/来源**：最终1408函数/3794断言/146文件，新增198/682，无删除弱化；37阻断/两goal/
  campaign/index18/compile通过。暂存侧实际发现driver新EOF空行已仅去末空行，定向再核，不改语义。
  Perceiver/RAFT实际SSH固定源码/许可/原文/hash补登记，HTTPS失败不改成功；无上游执行。
- **主链最终接口复跑**：runner语义/实际resume及driver/硬截止共160passed/47.81s；EOF空行修正后再验，
  不是合并此前并发source漂移的失败为通过。输出 `/tmp/r7_v2_runner_driver_stable_20261003.log`。
- **最终完整CPU**：3107passed/9skipped/2warnings，767.02s，实际exit0；六CUDA条件与三未跟踪
  optional real fixture跳过不算通过，两Lightning无Trainer警告保留。fullsuite期间仅driver测试末EOF
  空行删去，bytes证明original=current+newline；最终语义/driver160项复跑通过。实际全量日志及380源
  hash快照留 `outputs/r7_v2_remaining_acceptance_20261003/full_cpu_stable_receipt.json`，不冒称逐位GPU。
- **未做**：精确新CI、真实precision GPU、B/C训练评估、结果科学裁定、新ledger/index登记、
  六issue关闭/main推进。本计划不声称这些已完成。
- **影响**：新增known optional calendar和default-off query/matched结构接口改变active model digest；
  历史父只显式接受再导入，旧加载规则保持；无新依赖/天气下载/数据发布/凭据或用户安全配置改动。
