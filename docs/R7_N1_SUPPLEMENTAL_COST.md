# N1：独立 evaluation 成本补测（准备记录）

**状态：计量实现与CPU工程测试已准备；具名范围/预算授权未收到，GPU补测未执行。原N1仍paused，完整验收仍阻塞。**

## 1. 来由与授权边界

原证据 `docs/R7_N1_PIVOT_AUDIT.md` §5.4确认：30次eval峰值继承末training峰值，reserved未记录。
独立只读验收复核裁定：停止合规，但D2成本缺口阻断原完整验收；停止不豁免成本交付。

2026-09-30用户自由文本「明确进行授权」后，真正的AskUserQuestion问是否接受不完整停止交付，
用户回答：**「允许跑实验，反正就是把它搞得完整。」** 这明确选择补齐证据，不是接受缺项。
但回复没有具名预算与失败规则，不能据此伪造决策0021的完整授权回执。

执行前已实际用AskUserQuestion列出：`scripts/measure_r7_n1_eval_cost.py`、30个原val评估、六已有checkpoint、
0训练更新、一张空闲RTX3090、≤0.09GPU-h/324s、whole≤600s、新输出
`outputs/r7_n1_eval_cost_supplement/`、原RMSE/case严格一致、失败partial即停不重试、旧归档不写。
**该问题未收到回答**；所以没有`authorization.json`，没有新protocol/result/attempt实验产物，未运行GPU。
同轮只读nvidia-smi显示两卡均被其它进程占用，不终止或干扰它们。

补测的授权例外只拟覆盖这30次归档evaluation的成本重放；不会解除test封存、重训/新臂/下载/租GPU/
main/force/关#70–#75/节点推进等禁止项。最终执行必须取得具名授权，不能把此准备记录当授权。

## 2. 拟冻结的最小协议（尚未执行）

- 六个原selected400 checkpoint：RW-A/RW-B/冻结Z × seed41/42；lead6/12/24/48/72原val窗口
  22/21/19/15/11，训练更新为0。没有新模型选择、探针或科学判据。
- 用原`code.zip`，SHA256 `5fd26146af2a7d11016fb769d67390f5daa23a73620de9ae83e2e6cc38a35a0a`，
  source/model/dataset/checkpoint/BUILD_COMPLETE严格校验；安全提取118个active Python，不import legacy。
- 原protocol `e19ef488be60136364702b1df389e5f58be7ebf3845487289139e10d30231e01`；
  model digest `11090929930da4e1259698699cbbf12b3738cdfb2f609c3af516c24399144476`。
  原证据页hash `e5448064f1c4612ce30026ababb3b1b5a725048da566f72be4eec7cec92b8106`保持不改。
- 当前原实耗0.3590401737619605，原0.45cap余0.0909598262380395；拟补测≤0.09，合计仍≤0.45。
  账本仍已用4.0702/余19.9298；本轮新增GPU实耗0，未添加不存在的补测账本行。
- 每seed/arm一个fresh进程，每lead调用前gc.collect、显式device同步、empty_cache、同步、
  allocated/reserved均0方可reset；call结束同步并读取显式device的两peak。非零baseline拒绝。
- 测量scope为整个归档evaluate_local调用（含load/IO/metrics），不是独立forward latency；
  参数/FLOPs/training吞吐从原归档引用，evaluation内存才是新测量。
- 主进程从首worker前至末worker退出保守计费，包含import/启动/间隔；绝对deadline监督，
  超时只kill自身child并wait、不干扰他人。设备查询也设≤5s且不超过剩余deadline。
- 冻结wrapper ZIP字节与hash同次读取，child只运行提取的wrapper；前后所有输入pin核对，
  各行精确绑定protocol/device/checkpoint/output。全集合不齐不得success。
- RMSE CSV值、case/provenance/MSE严格相同；不引入planner建议的新浮点容差或经验显存阈值。
  原48/72h primary仍unresolved，不从补测改科学解释。
- 新输出排他产生`protocol.json`、`result.json`、30cost_measurement、attempt、四成本CSV、
  measurement_code.zip/code_commit；旧归档完全只读。失败部分全部保留、不得自动重试。

## 3. 实现与实际工程验证

实现文件：`training/r7_n1_cost_replay.py`、`scripts/measure_r7_n1_eval_cost.py`、
`scripts/measure_r7_n1_eval_worker.py`；CPU反证 `tests/test_r7_n1_eval_cost.py`。
没有改原模型/训练runner/evaluator或旧driver，因此旧model code identity不变。

实际已做：

- 只读preflight核30任务、六checkpoint bytes hash、source/receipt/preflight/train/val/BUILD_COMPLETE身份；
  安全提取118个归档active Python到临时目录，尚未执行归档评估器/模型或CUDA。
- 最初40项CPU测试通过。独立源码审阅发现四项guard缺口：查询timeout、实时wrapper身份、
  输入pins不全、逐行身份未绑定；已向前修复并增加反证，不改科学阈值。
- 最新成本测试48实例；连同conventions/campaign/index回归 **195 passed**（21.32s），
  全部CPU/fake CUDA，训练/真实评估均未执行；它们不等于成本测量完成。
- planner返回草案，主链撤掉新RMSE容差/5或10GB阈值、无意义SOP文件和自动验收声明；
  工作态`/tmp/r7_n1_cost_plan.json`经`check_planner_plan.py` verified=true，不当证据。
- Mimosa写前拒绝两类动态命令候选。改用固定启动命令/stdin JSON、参数校验、shell=False；
  没有关闭扫描或修改凭据/安全配置，也没有据此声明项目安全。

四新文件均≤600行、函数≤200；未新增例外、未弱化测试。R-009基线从992/2525更新为1021/2569，
新增29测试函数/44断言，无删除；119测试文件/165parametrize为AST观测，不是CI计数。
暂存后规模实测R-019=0/R-019b=183/R-020=45/R-021=36/R-022=23/R-023=23。

基线和文档同步后实跑（GPU显式隐藏，`PYTHONDONTWRITEBYTECODE=1`）：

```bash
.venv/bin/python -m pytest -q tests/test_r7_n1_eval_cost.py \
  tests/test_check_conventions.py tests/test_check_campaign_state.py \
  tests/test_check_goal_brief.py tests/test_verify_r7_evidence_index.py
```

**214 passed（19.82s）**；37阻断规则0违规、两goal brief均0失败、campaign0失败/6条历史notes，
暂存与工作树whitespace检查通过。冻结§0/§2–3/§5–6与起点efe7d82逐字节比较未变；
原证据页、code.zip/protocol/merged/attempt的SHA256仍匹配，原模型/evaluator/runner/index/brief无改动。
`outputs/r7_n1_eval_cost_supplement/`不存在，未伪造任何补测产物。这些仅是工程准备验证，不是GPU计量或成本验收。

暂存代码的无本地产物隔离clone全量CPU实跑 **1843 passed / 14 skipped / 2 warnings（188.25s）**，
比原1795多48个成本守卫实例；6项skip因CUDA显式隐藏，8项因可选真实产物在干净clone中不存在，
没有合成回退，skip不算通过。两个warning是既有Lightning测试未挂Trainer时调用self.log，未隐藏。
日志`/tmp/r7_n1_cost_preparation_full_tests.log`；这是后续复核修复前的代码快照，不替代修复后的复跑。
此计数是本地值，不冒充远端CI日志。

后续独立CPU复核发现四项残余守卫缺口（不是实测GPU故障）：重算digest可替换val路径、
CUDA与nvidia-smi序号对应未证明、末次evaluation后无占卡检查、收齐30行但不核输出文件。
已修：原protocol主体及val路径/source身份使用前绑定，所有source pins与val split在每次调用前重验；
父进程将选中UUID冻结为子进程唯一`CUDA_VISIBLE_DEVICES`，子用`cuda:0`且核实际torch device UUID；
调用后/worker尾/父进程尾复查占卡；30行均绑定并重验provenance/RMSE/ACC/skill/measurement字节摘要与精确重放。
这些占卡检查只能发现观测时仍存在的竞争，不能证明两个观测之间绝无短暂其它负载；保留计量局限。
补入对应反证，初跑1failed/62passed因fake checkpoint缺字段，修正fixture后63passed，再加入val split反证。
原模型/归档/科学判据及授权边界未变。

修复后定向五文件同一CPU命令实跑 **232 passed（20.50s）**，成本反证66实例；
日志`/tmp/r7_n1_cost_final_repaired_tests.log`。再做只读原归档身份核对，30任务及六checkpoint/source bytes
仍匹配，原protocol绑定的val路径通过；未执行模型/evaluator/CUDA或解码test。原先214/1843计数仅对应修复前快照。

修复后暂存代码的隔离clone全量 **1861 passed / 14 skipped / 2 warnings（161.04s）**；
日志`/tmp/r7_n1_cost_repaired_full_tests.log`。跳过与warning理由同上；比原1795多66成本实例。
独立复核只重查四项发现并实跑66fake测试，确认四项均已处理，没有发现该范围内残余可操作缺陷；
不是全项目安全复核或真实GPU认证，也不豁免短暂占卡与预算/数值重放的未实测风险。
此后只追加准备文档，计量代码保持该验证快照；发布提交与自身CI在最终回复报告，不递归改原证据/索引。

## 4. 接口、兼容性、依赖与安全影响

新增CLI `scripts/measure_r7_n1_eval_cost.py` 接收`--authorization`（必需）、`--archive`、`--out`、
`--device`；没有改变既有训练/评估接口、wheel入口或默认行为。没有新增第三方依赖：父进程与身份核对使用标准库，
worker使用项目`.venv`及归档评估器所需的既有torch/数据依赖，并强制原torch版本。

计量scope、进程隔离及allocator清理与原运行不同，补测峰值不能冒充原运行当时的独立峰值。
本机真实GPU行为、全套归档数值严格重放与324s预算是否可满足尚未实测；CPU守卫通过不覆盖这些风险。
补测遇到基线非零、占卡、数值差异、身份不符或超时必须保留失败并停止，不自动放宽或重试。

没有修改数据、原归档、凭据、安全配置、CI定义或科学结果。仅有本地socket禁网与固定命令/路径身份守卫；
它们是防意外措施，不是沙箱或全项目安全认证。没有终止任何其它GPU进程。

## 5. 未做与下一项

**未做**：具名执行授权、GPU补测、30条独立peak数据、成本验收、补测账本及补测结果索引。
原成本缺口仍为真实未齐项，不能标PASS。没有下载/test/训练/改归档/节点推进/issue关闭。

下一项只是在执行时取得§1具名范围/预算授权，并确认一张设备空闲；未获授权前不得运行。
即使补测成功，也只让独立复核裁定成本证据是否补齐，不由执行者自宣N1目标完成。
