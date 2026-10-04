# 0016 主模型超气候态研究：授权、目标长文与校检支持交接

**状态：用户批准 / 本轮仅文档与工程支持（2026-10-04）；研究未执行。**
起点 `73097d49fd174d8e5f2bbb465dad64037bd486c9`，工作分支 `r7/weather-reasoning`。
用户的 `.zcode/config.json` 差异和两个无关未跟踪项不修改、不提交。

## 1. 本轮范围与总体方向

用户确认上一轮完成，目标转为沿 UrbanPiDiT 可思考/递归主模型持续迭代，公平稳定超过 train-only
climatology。明确实验、时长、普通决策、节点推进自主下放，并允许更多真实数据。本次审批另明确
免费区域多年度/多季节数据、执行者自审preflight后向全新outputs路径--write的具名范围。

本轮只写研究文档/授权/技能路由和必要campaign校检支持，不执行下载、store构建、训练、天气评估、
源码clone或GPU查询，不commit/push。下一窗口用新goal实际执行，不在此轮把“文档交付”说成研究结果。

## 2. 真实起点与新问题

- 旧六issue #70–#75闭环：关闭提交f0b7705、精确CI37215981072及ff/API终态见旧master§8；不能
  自行设旧goal complete，也不重做B/C实现。actual C三seed144jobs已齐，package改善不能归因过程语义。
- `docs/R7_C_ACTUAL_CONFIRMATION.md` 记录primary对old_ours六格改善，完整全表有坏变量；它不是
  model-vs-climatology的胜出证据。旧M2单冬季/区域双月与已曝光test不支持独立跨年四季确认。
- 新open #76–#79于2026-10-04匿名读取，API作者Eswink；GPT6PRO来源按用户说明，不作为事实权威。

| 参考任务 | 第一优先动作 | 不混淆的控制 |
| --- | --- | --- |
| #76主模型性能EPIC | P0真实incumbent开关与公平gap表 | 同图Generic等价控制不是必须赢benchmark；性能≠机制归因 |
| #77位置频带 | 实际激活/解析阳性→单因素小真实试验 | 不并改loss/K/aux/solver，不无限枚举频带 |
| #78变化量尺度 | 保持loss重参数化解码→独立loss权重→等额延训 | train-only连续6h，误差单位s_c/d_c，区分同更新/算力 |
| #79类型诊断前向 | aux0且同信息无类型/有类型路由 | B→C才归因类型，诊断真实可推理、未来target只监督 |

来源URL与访问日期、comments精确单位要求见科学合同§7；不在本轮另找论文或下载源码。

## 3. 授权与不豁免项

决策0038具名扩围0030的本方向数据/发布边界；普通方法/时长/下一实验/节点自主，不逐批问人。
下载先pilot，数字范围、网络/decoded/磁盘预算、soft/hard及恢复策略先冻结；自审preflight必须有回执，
完整hash/schema/单位/finite/fresh_outputs/BUILD_COMPLETE身份不能省。只写全新排他outputs，不动旧受保护data。

无总GPU-h上限，soft超继续记overrun，hard停止该attempt全额留失败；下一独立协议可增时长，不能
运行中改旧hard。通知≠等待批准。GPU默认共驻余量门、不signal邻居；准备联网，实验离线。
仍无付费/租卡/独占/方向改变/main合并/破坏性授权；#76–#79关闭/main写入不继承0037旧六许可。
最终goal complete用户/运行时独立判，不以计划/测试/CI/issue数或单次val赢格代替科学越过。

## 4. 研究路线（执行窗口S0–S4）

1. **S0 gap**：核actual C代码/配置/数据/checkpoint，对共同cases/units/mask重新审计模型、climatology、
   persistence；新科学实例完整再启动，旧结果身份不兼容不填进表。
2. **S1准备**：真实源pilot/网络schema与PE/scale CPU、typed诊断设计、论文源码只读可并行，下载慢不
   停独立工作，不让实验阶段联网；preflight+新的发布契约和train/val/未见test隔离齐。
3. **S2小真实探索**：每轮可反驳机制、新信息/差异、阳性负控制、配对和停止/饱和出口先冻结。
   连续定向无收益转向信号/梯度/优化/数据制度，不无限开关或重复直到赢；负面停该hypothesis。
4. **S3扩训练**：按val证据晋级，不把更多数据/算力称机制成功；改data重训climatology及必要incumbent，
   同信息/参数/更新/样本/算力分开核，冻结候选K/配置/checkpoint与最后确认统计。
5. **S4独立确认**：真正未见年份及四季、3+seed、时间块同时区间/守门；失败记录曝光与r，返回独立
   开发不复封；过门则独立封印/登记并提请最终验收，不自宣SOTA或complete。

## 5. 主链写定的科学合同

`docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` 是独立新合同，不沿用旧C对old_ours门。
primary t2m五lead/full、每seed/确认年/四季正MSE skill、时间块同时下界>0；u10/v10/mslp同数据
incumbent零退化守门。全17变量×五lead/三区域照报；分母0/undefined不得过滤过门。

同时family包含各primary/守门/确认组；r次确认alpha=0.05/(r*(r+1))，首轮0.025，总<=0.05。
具体block长度/重采样/seed/cases在看test之前冻结，依据train/val；不能将格点/重叠窗/seed当独立
天气样本。test首次评分消耗r，partial也留曝光，不因失利重置确认序号。合同可能严格，不能弱化
来“完成”。无保证可达标，单冬季或package增益不代跨年/四季证据。

## 6. 子代理与技能明确路由

| 类型 | 触发和产出 |
| --- | --- |
| Explore | 未知路径/调用链/证据定位，只读有界，主链不重复相同委派 |
| planner | 多步依赖/架构/风险，JSON合同校验，科学门槛不外委 |
| web-researcher | 新机制/官方文献源码/API/错误事实，一手URL/日期/revision/许可 |
| web-researcher-backup | 主检索不可用同任务接续，仍失败curl留因 |
| general-purpose | 隔离定向实现或独立只读候选/最终证据复核，文件/运行不冲突 |

GPU由统一父调度余量核并发；资料/CPU可并行，不反复扫仓。独立复核是质量门，不是重新许可。

| 技能名 | 触发时实际加载 |
| --- | --- |
| goal-loop | goal/节点/上下文交接与harness降级 |
| planner-delegation | 多步方案/风险及planner JSON核验 |
| decision-record | 授权/数据/长期约定ADR |
| issue-lifecycle | issue推进与真实验收、精确SHA |
| real-data-acquisition | 新源/下载/派生发布，0038范围内自审preflight |
| environment-rebuild | 环境/缺依赖/CUDA异常 |
| web-research | 外部一手事实与引用 |
| bounded-study-run | 有界CPU探针/实验，非通用GPU训练SOP |
| pinned-artifact-replay | 旧产物归档code.zip/pins重放 |
| result-freeze | 引用/归档/digest与复现等级登记 |
| ci-workflow-triage | CI失败/pending/cancelled与精确SHA所有必要步 |

公开参考源码留outputs/reference_sources/project/commit隔离，固定license/hash及引用台账；不import镜像或
执行上游安装训练。技能只列名字不实际加载不算完成路由。无额外通用GPUskill或重复平台。

## 7. 本轮改动清单与验证

新建本计划、决策0038、新goal、科学合同。同步AGENTS、ADR/plans/goals/skills索引与rules/CHANGELOG、
data-and-artifacts授权接续、real-data-acquisition与goal-loop前瞻scope。
旧master/closeout只追加真实交接、不删证据/旧next-action和未做事实、不自裁状态complete。

工具复用check_campaign_state新增--campaign仓库内goal相对路径，默认仍旧master，C01-C06不放松，
unsafe/外部路径拒绝；tests正反证与ci现有campaign步增加第二调用。新账本0成本真实文档启动行有
本计划证据指针、无伪实验record，仍按原C02算术；新旧ledger分账不重复9.9139历史成本。
新增测试后按实际AST同步check_conventions的1586/4333基线及testing/MIGRATION；size-thresholds
只更新R-021实测64→65，不改阈值、断言或冻结例外，CHANGELOG留本轮增量与原失败。

验证实际命令在结果中登记；新/旧goal、campaign、conventions、ADR引用、index/brief/空白及工具CPU
测试、旧default兼容和unsafe/节点/账本/证据失败反证。新分支CI只有未来commit/push后核，本轮不伪称。
不改dependency、安全配置、模型digest、训练/下载API；新增支持CLI仅--campaign默认兼容。

## 8. 与 planner 草案的差异及风险

方案来源为只读planner草稿，主链修订JSON经validate_plan返回verified=true，草稿非证据。
原稿把长文4000限制、旧C primary、空账本note宽免、虚构决策CHANGELOG、旧mastercomplete、缺CI/skills
路由等混用，均已拒绝；只有objective<=4000，新科学合同主链写定，旧门/历史不动。

- 风险：自主累积成本和下载阻塞。缓解：数字分批范围/pilot/余量、失败receipt、独立工作并行，
  不能删旧数据或租卡跨边界。
- 风险：直到胜形成test泄漏/选择偏差。缓解：确认r/alpha、实例冻结、真未见数据、时间块同时区间，
  失败不复封、不随机换primary来找赢家；无收益重新分析，而非无限增seed。
- 风险：模型符号/同图等价误归因。缓解：actual配置审计、同信息结构差异、typed B-C控制与直接前向反证。
- 风险：新master弱化旧机械门或工具碰到用户配置。缓解：default兼容C01-C06、unsafe反证、隔离agent
  文件归属；不触.zcode/config.json、不扩例外。

## 实际结果

- **完成情况**：四份新文档、授权/规则/技能/索引接续、旧路线真实交接及最小校检支持已完成。
  `--campaign` 默认兼容并拒绝unsafe路径，旧C01–C06不放宽，CI在原step保留旧调用并增加新master。
  新研究S0–S4 **未执行**；没有下载、构建store、参考clone、训练/天气评估或GPU使用。
  HEAD仍73097d49fd174d8e5f2bbb465dad64037bd486c9；没有暂存、commit/push，新增治理文件仍未跟踪。
- **与计划的差异**：确认alpha明确为可核序号与spending，强化“至少95%同时区间/不重复test练到赢”，
  不改旧判据。独立审阅要求把合同已有验收门内嵌objective，修后实测单段3974 code points；
  新测试有意净增12函数/34断言，按规则同步1586/4333基线及实际R-021 marker64→65，无例外放宽。
- **实际CPU验证**：六模块为test_check_campaign_state、test_check_goal_brief、test_check_planner_plan、
  test_check_conventions、test_verify_r7_evidence_index、test_gpu_policy_metadata。用项目venv的
  `python -B`/`pytest -q -p no:cacheprovider`，禁用插件自动加载、测试进程socket出网拒绝、
  CUDA_VISIBLE_DEVICES为空，所有临时产物在仓外/tmp；planned_seconds=600/hard_cap_seconds=1200。
  首轮 **261passed/1failed/0error/0skip**、整轮28.979858秒，唯一失败为旧规模marker64与实际65漂移；
  原日志不删，修实际观测后同集合 **262passed/0failed/0error/0skip**、整轮33.041215秒，overrun0。
  工程agent修改前campaign模块13passed、修改后60passed；主链上述集合包含这60例，不重复计总数。
- **回执与失败保留**：首轮 `/tmp/r7_climatology_governance_20261004_085g8tt4/verification_receipt.json`，
  SHA256 `61ea2aa5145f529ebe0b362d1f073b77f7150038eb319a09867b93febd17552c`；
  修后 `/tmp/r7_climatology_governance_20261004_fixed_hu7rhnxp/verification_receipt.json`，
  SHA256 `f316f5446cb27b62c00559a04b8c18cebe2bfd71c3c16bf31dc740a85b913ede`。
  两回执含JUnit/log digest、命令模块、软硬/成本及limitations；本地临时回执不是新科学索引record。
- **结构与身份验证**：37阻断0失败；新/旧campaign分别0fail/1note与0fail/4notes，真实未索引文档0成本
  与旧历史notes保留；三个goal0fail0建议；四新文档引用/空白/编号/ADR及显式新路径规则通过。
  evidence index与canonical brief核26record，逐字未改HEAD版；git diff --check通过。
  最终对24个交付路径使用--paths模拟提交树，37阻断0失败，未暂存且排除用户config；不以未跟踪宽免冒称通过。
  两次只读复算脚本误用ALL_RULES及iter_py的tuple接口报AttributeError，按实际RULES/解包接口重跑成功，
  没有修改检查实现或断言来迎合脚本。AST全tests实测1586函数/4333assert/160文件；
  工具376行/最大函数体28行、测试408行/最大25行，
  600/200硬限及13文件/4函数旧冻结清单不变，checker仍1925行。
- **独立复核**：文档初审1必修（objective遗漏合同既定门）已修，独立复读无必修；校检器/测试/CI
  diff与AST独立只读审阅无必修，不把静态审阅称独立实跑60pass、科学接受或最终goal verifier。
- **接口与风险**：仅新增校检CLI --campaign和可选API参数campaign_doc，默认不变；无模型/下载/训练API、
  数据/结果、依赖、凭据、安全hook或用户配置改动。新许可是0038范围内前瞻授权，不允许改原始/旧数据。
  四新治理文件未提交时干净克隆不可用；本地工程通过不冒称remote CI或科学胜出。
- **未做与交接**：未跑全量suite、wheel或远端CI，未关闭新issue/main写入；S0真实gap、source/数据实例、
  逐次soft/hard数字、候选/统计实现及未见确认待执行窗口自主完成。下一项第一具体动作是S0核actual C
  incumbent/source/checkpoint pins，冻结同case的train/val gap审计协议及软硬秒数；最终验收与保留权限不变。
