# 0013 — N2a 交付验收与全权委托接续

**日期：2026-10-03。状态：N2a验收矩阵与全权委托已落实，本次定向/机械检查通过；精确新CI待绑定，科学paused保持。**

## §1 问题与结论

用户问：「N2a是否已经已经完成，如果未完成，则计划完成，并且明确所有授权全部下放给你。」
本计划经用户批准，只做六文档验收留痕、授权接续和精确工作分支CI，不重做已经交付的工作。

**N2a规定的工程与实验交付已经完成；当前paused是科学停止出口，不是缺训练、缺23项评估或缺D6。**
原目标明确「结果可negative/mixed，结构正确即按#73验收」（`docs/goals/n2a-m3-process-supervision.md:15`）；
§2/§3要求尺度、三类时刻、反证和一轮有界真实train/val探索，不要求辅助臂必须取得正增益。
原单次attempt确实failed/budget_limited，不能把它改成成功；ADR0031另立协议合法补23项并逐文件
区分来源后，节点所需完整覆盖/配对/成本已齐。工程交付完成与原失败、科学暂停同时成立。

冻结依据：`docs/goals/n2a-m3-process-supervision.md`、`docs/R7_MAIN_MODEL_V2_DESIGN.md` §6、
`docs/R7_65_PREDIAGNOSTIC.md`及决策0028/0030/0031。只引用判据，不新增完成阈值。
`docs/decisions/0031-m3-cross-attempt-validation-provenance.md:16`明确any-unresolved暂停、不推进；
此出口不能由“交付已齐”或授权下放抹掉。N3/N4/N5、正式issue关闭与整个V2 goal仍未完成。

## §2 授权接续与本轮选择

完整保留用户原话「所有授权全部下放给你」。本请求N2a完成/收尾所需的执行、工程修复、实验安排、
预算、验证和普通决策由执行者自主承担，不逐项询问。作为决策0030的接续确认留痕，不重建逐轮
许可，不另造一个要求用户逐条回答的授权清单。

执行者根据实际完成情况选择0新增GPU-h、只做验收记录；不是把用户的全权授权缩成“只允许写文档”，
而是已经不存在必须补跑的冻结验收缺项。本次批准计划仅六文档，不行使下载、付费/main/关闭/外发
等与本请求无关的动作权限。授权不是科学豁免：不改历史事实/科学阈值、不伪造通过、不绕hook；
不直接把运行时最终goal状态设complete，仍保留用户/运行时独立完成裁定。

## §3 N2a验收矩阵（既有证据，不冒称本轮重跑）

| 原交付物 | 已完成的节点交付 | 可核证据与限制 |
| --- | --- | --- |
| D1 尺度sidecar | train-only有量纲raw RMS缩放、无量纲mean/std、逐通道相对floor/degenerate mask、版本metadata/identity入contract | `data/preprocess/r7_process_scale_sidecar.py`；`tests/test_r7_process_scale_sidecar.py`；`outputs/r7_m3_scale_sidecar/scale_metadata.json` SHA256 `d7c2837ea0083b5ce98f98cd02da729c1f12147387838dccad420393b7295cec`，identity `4fed1c78e4c4c09a41d95457649925b02d0c8ebf89aa47c2ac8dc6496734912d`；train186窗口/188帧8通道active，floor不作除数，旧store不改 |
| D2 八proxy尺度审计 | 八通道旧/新尺度、mask及统计全部报告；两水汽旧normalized std约0.009915493/0.014293376修后约1 | `training/r7_process_diagnostic.py:proxy_scale_sidecar_report`；`outputs/r7_m3_acceptance/actual_train_scale_audit.json` SHA256 `6588b7bc9b36b2b487a2baeb17bc01f3910af7ad853a3ba5c63dcff4a1ee34a0`；helper无直接专门单测/旧CLI默认旧report保持披露，只比较尺度、不作forecast因果归因 |
| D3 三类时刻 | input/future/draft分名和时间契约、固定train inverse/地理metric、内部K同valid-time、新旧loss互斥、strict shape/name/mask | `training/r7_process_forecast_losses.py`、`r7_process_supervision.py`、`r7_process_tensor_diagnostics.py`、`r7_process_training_contract.py`；moments/streaming/contract/inverse-binding反证；适用于实际M3 scheduled/streamed/whitelist/adaptive路径，不泛化所有裸forward/旧Lightning |
| D4 定向反证 | analytic散度/涡度/平流/纬度metric、NaN/shape/name/degenerate拒绝、poison只改loss、无target推理、敏感差分FP32与梯度证据 | `tests/test_r7_process_supervision_moments.py`、`test_r7_process_tensor_diagnostics.py`、`test_r7_process_scale_sidecar.py`及paths/streaming测试；本会话既有420passed/0skip/53.87s；meridional独立analytic、水汽raw极小量还原强度及BF16读法限制不隐瞒 |
| D5 节点完整有界对照 | 六原训练各400updates/K4/seed41–42，旧7val+独立23val覆盖30eval、510RMSE、528case、三pair各85/255汇总，17变量五lead及四成本齐 | 原失败页与新 `docs/R7_73_VALIDATION_COMPLEMENT.md` §3–6；ADR0031逐文件三来源；paired SHA256 `5e845badb28086a32ef016167b3efc48114fe303780fd3d13898b2fa035c2d68`；原attempt失败/原finalizer未接受不改，新attempt完整但paused，不是将原D5单次attempt判成功 |
| D6 证据与登记 | A冻结新证据页，index18单增audit/blocked记录、canonical brief/账本/活进度齐；B/C精确主CI九步success | A=`1de3a672d859b42efa7a5fac3293c2832df4a6ee`、B=`da4939e613eb1a0b39f6303142ffd7aa6ab8657c`、C=`20230567eb4272b9fd70a73e0f6e5dc8af34ed63`；CI37108664102/37109369425，见计划0012和补测长文§8；本轮不重复追加index或回写冻结页 |

补测目标自身D0–D4也已齐：D0复用/数据身份与preflight资格、D1新协议和23缺项、D2全来源/配对/
四成本、D3工程测试与精确执行CI、D4登记/独立只读复核/逐要求审计。接受范围是节点交付，不能
伪造原attempt36个receipt成功或说GPU未来逐位可复现。

新补测页A的实际blob SHA256为 `17c446253af0e6df838cf60ef02800b7d64c6cce0490b981ddfc5e22789b4a85`；
补测protocol canonical `74d05ab36a5c15451893e20f9cadcaecdac62170717af61a3e7524af57509dd5`；
最终seal SHA256 `0810665b861a3f2206b53556a6878b179e68f2923e560e11c7ef906af6797a72`。
既有完整CPU2395passed/9skipped/2warnings/230.96s、登记定向532passed/0skip/37.67s及C尾46passed/
0skip/0.23s仅引用实际历史记录，本轮不冒称重跑；skip非通过，远端pytest日志计数未取得。

## §4 科学出口、费用与未做项

三pair的改善/恶化/unresolved分别19/45/21、21/36/28、3/44/38；两辅助臂对off的T2M两seed五lead均
恶化，是有界描述性结果，不是普遍因果或显著性。any-unresolved冻结出口确实触发，scientific
paused/advance_next_node=false保持；普通授权下放不允许追加seed/预算“练到赢”。

原attemptGPU0.49725426027008024h/whole1805.1085775829852s超过原1800文字上限仍failed，新独立
补测GPU0.17899193391850632h/whole700.5363800507039s complete-paused；M3总0.6762461941885866h。
账本显示used4.8808/remaining19.1192，四历史无索引notes保留。新本轮0GPU-h，不加成本行。

MetPy设计台账的一致性对照意向尚未实做，本次不声称MetPy通过；原N2a验收只限定它离线oracle用途，
不把该意向事后升为新的完成硬门。额外真实17通道CPU端到端/更强独立analytic属于验证补强，三seed/
matched-Generic属于正式贡献评价或后续节点，不反向硬塞N2a两seed探索。各弱覆盖原文保留；
没有科学改善、泛化、全OS/test/禁网或邻居信号无发生的完整证明。

## §5 实施与验证

1. 核实际HEAD/status、原109/新276/366归档/43科学pins、两用户无关文件和本地配置，仓外排他回执。
2. 校验修正plan JSON后落本页；只在campaign自然语言/N2a说明、补测与总体goal进度及两索引留痕。
3. 实跑conventions/campaign/两goal/indexbrief/空白及campaign/goal/index三组CPU定向；复用已有证据。
4. 独立只读复核新验收及授权/暂停界限；核六文档以外旧跟踪文件逐字不变、实验文件集合/hash不变。
5. 六文档正常提交SSH推工作分支，核最终精确SHA主CI九步success；最终回执在仓外，不递归记录自身提交。

新增定稿计划为0013，JSON工作态仓外不提交。规划来自两只读Explore与planner草稿；主链纠正不存在的
ADR0031路径、遗漏新计划文件/C精确CI、错误五notes及多余全量本地重跑建议。实质判断只依据本仓
冻结验收，planner不定科学阈值。没有修改0030/0031、AGENTS、hooks或阈值，科学state/账本不变。

范围：仅本页、plans索引、campaign、补测goal、总体goal§8和goals索引六文档。两冻结证据页/旧goal/
indexbrief/协议/所有实验归档/数据/模型代码/依赖/凭据/用户配置及无关文件不改。正式git门若拒绝即
停止不换路；身份不符或机械失败先核具体原因，真需范围或判据改变不以文档伪造修复。

## 实际结果

- **开工与计划验证已执行**：起点 `20230567eb4272b9fd70a73e0f6e5dc8af34ed63`；修正版plan校验
  `verified=true/fence_stripped=false/failures=[]`，campaign0fail4历史notes。仓外工作态
  `/tmp/r7_n2a_delivery_acceptance_20261003_enmx8cmz/`；起点9核查全过，1016跟踪文件、原109/新276集合
  和rawhash、366归档/43科学pins/用户配置/两无关文件与已核基线一致，起点回执SHA256
  `d7548e32a8698a698b1fb01e04ba2b0b6c6618cf2ff24ba52cc8a8b1da5eab63`。
- **实际落实与验证**：验收矩阵/全权委托与交付完成、scientific paused分开说明，六文档范围已核；
  本次三组CPU定向46passed/0skip/0.23s，37阻断0违规、campaign0fail4历史notes、两goal0失败、
  index18/brief一致、staged/unstaged空白通过。18项范围/引用/identity核对全过，原109/新276集合及
  rawhash、366归档/43科学pins、全部非六文档跟踪字节、用户配置/两无关文件、账本和机器state精确
  不变。核验回执SHA256 `0581dd426a223a54bde0c7b11820691b31668ad8fda5bfd937accd20ebf6638f`；
  定向日志SHA256 `992eb14eaa84884996a1e97ac41bc95a09fe09625f10f9e4b030d59a6430bfd6`。
  本次未重跑完整本地pytest、GPU或数据评估；新提交自身主CI将执行完整回归，精确CI仍待实际结果，
  不以历史B/C的绿覆盖本次文字。
- **有限独立文档复核**：六文档及原验收/0031两对照只读复核无必修；交付完成、科学暂停、原失败和
  全权授权界限一致，MetPy/e2e/三seed未事后升硬门，历史B/C的精确CI不冒称本次文字已验证。
  复核仅8次Read，没有shell/测试/网络/归档执行或产物digest核验，不是科学/安全/最终goal裁定。
  本条写入时新六文档提交/自身CI尚待执行，精确结果在仓外最终回执交付，不递归修改自身提交。
- **终态界限**：N2a规定交付完成不当科学成功，原failed/新complete-paused及费用不改；N3–N5、main/
  issue关闭和整个V2 goal完成未做或自裁。下一研究动作仍为在冻结停止出口下审阅已登记negative/mixed/
  unresolved与总体方向，本计划只收尾N2a交付，不推进。
