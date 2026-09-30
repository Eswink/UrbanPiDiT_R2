# N1：转向审计与冻结随机 Z 可证伪臂

**状态：D1 零 GPU 审计与 D2 工程准备中；D2 已获决策 0021 授权、未执行。不是完成声明。**

## 0. 起点、范围与证据资格

- 起点 SHA：`efe7d82d11542047897196e4d278cdb18752d6b2`，分支 `r7/weather-reasoning`。
- 主计划：`docs/goals/main-model-v2-campaign.md`，当前节点仍为 N1；本轮长文
  `docs/goals/main-model-v2-pivot-audit.md`。开工对表 failures=0、notes=6。
- 账本：已用 3.7112、余 20.2888 GPU-h；本轮 D2 上限 0.45 GPU-h。当前 GPU 实耗为 0。
- `scientific_claim: false`。以下区分代码/算术事实与解释候选；“支持”只表示既有登记证据支持该嫌疑
  作为解释候选，不表示完成因果归因。没有新增阈值、端点或显著性规则。
- 旧产物只读。所有归档数字来自减法轮自己的 CSV/JSON，不与其它协议的数字合并。
  没有读取 test manifest、test 状态场或旧 checkpoint；没有重跑旧模型。

## 1. D1：forecast state（预测状态）

**读法：支持作为解释候选，证据弱于训练目标与暴露失配；尚未归因。**

### 已确认

1. 本轮焦点 RW-B 没开来源角色标记：`training/r7_rw_b_subtraction_protocol.py:38-56`；
   `model/recursive_weather_r7.py:15-44` 在 role 关闭时直接拼接 C 与 E(Y_k)，没有来源 role 或 mask。
   当前实现已经把三条镜像合成单一 step，但没有因此修复焦点臂所看到的信息语义。
   `model/process_step_r7.py:84-98` 显式读 context、草稿编码和过程读数。
2. 模型输入白名单：`model/r7_halting.py:23-38`。targets/baselines 不进入正常前向。
   `model/process_step_r7.py:121-146` 的 Z、门控和提案在同一物理时刻的 K 轴内更新，提案锚定 X_t，
   不是直接读未来真值。
3. E-195（`docs/rules/EVIDENCE.md:321`）确认角色缺口，但不确认它造成顺序探针不敏感。
   E-202（同页 `:360`）确认焦点臂的 `correction_head` 未训练；把 RW-B 的权重改走它不能当干净消融。
   E-206（`:364`）确认同一递推移除在推理期与训练期幅度差 35–60 倍，训练会压小门控。
4. 上一轮 no-recurrence 的 Z 仍由可学习的 `solver_init` 产生（`model/process_step_r7.py:121-134`），
   因而它没有回答本轮“完全不需要学到的 Z 吗”的问题。

### 解释与局限

“历史上下文、草稿和过程读数的语义不清，以及被修正对象与训练通路的适配”有直接代码/干预线索。
但 role 对照幅度小且方向不稳，未训练修正头不能作反事实；不能把 E-195 写成已知主因，不能把门控关闭
直接外推成任何 solver 的普遍缺陷。尺度/floor 属 #73 的过程监督问题；当前 process_weight=0，
没有证据让 M3 成为本轮最强方向。

可复核命令（只读，无模型执行）：

```bash
.venv/bin/python -c 'import inspect; from model.recursive_weather_r7 import recurrent_key,declared_source_roles; from model.process_step_r7 import process_reasoning_step; print(inspect.getsource(recurrent_key)); print(inspect.getsource(declared_source_roles)); print(inspect.getsource(process_reasoning_step))'
```

## 2. D1：training objective（训练目标）

**读法：支持，四方向中最强的直接代码/协议证据。**

### 已确认与量化

- `training/r7_streaming.py:123-148` 对 K+1 个草稿使用 `linspace(1,final_weight,K+1)` 后归一化，
  每一个 MSE 都对同一个 `atmos_target`。K=3、final_weight=2 时权重为
  **1/6、2/9、5/18、1/3**；不是对 6/12/24/48h 加权。
- 已归档训练清单 `outputs/r7_m2_segment/store/manifests/train.jsonl` 有 **186** 行，lead 全为 **6**；
  val manifest 有 **22** 行，lead 同样全为 6。训练 target 没有 48/72h，K=3 不会制造这些 target。
- `training/r7_rw_b_subtraction_protocol.py:31-32` 的 process_weight=0。当前 aux 标签不能解释这一轮
  的优化方向，更不能把修 floor 当成已获支持的修复。
- `training/r7_halting.py:14-35` 在归一化空间逐通道等权；`training/r7_rollout_metrics.py:68-76`
  先乘 train std，再逐变量报告物理 RMSE。通道等权不等于物理变量同量纲，也不等于长时效目标。
- 原 runner 的验证模型选择也是 +6h 归一化 MSE（起点
  `training/r7_scheduled_runner.py:77-135,294-321`），不是长时效选择。改造只接通干预/截止时间，
  不改变这条冻结选择规则。
- 既有 #64 curriculum（`docs/R7_64_CURRICULUM.md:24-35,84-100`）改变训练目标后出现短长时效权衡；
  这是审计线索，不是本轮重测或因果证明。#65 的总损失与任务收益差异见
  `docs/R7_65_C1_PROCESS_SUPERVISION.md:109-117`。

### 解释与局限

优化对象与报告对象失配是可直接证明的事实，足以支持优先审查目标对齐；但它没有单独证明 RW-B 比 RW-A
差的原因，因为两臂都使用同一损失。禁止把重新分配 +6h 预算说成免费的目标改善，也不在这里改权重/判据。

可复核命令：

```bash
.venv/bin/python tools/recompute_r7_n1_audit.py
.venv/bin/python -c 'import inspect; from training.r7_streaming import backward_streamed_truncated; from training.r7_halting import per_sample_latitude_mse; print(inspect.getsource(backward_streamed_truncated)); print(inspect.getsource(per_sample_latitude_mse))'
```

## 3. D1：data regime（数据与评估支撑）

**读法：支持作为幅度与外推不确定性的解释；不支持把两轮的 48h 同号恶化直接抹成“一次噪声”。**

### 已确认与复算

M2 只有一个冬季、一个区域、一个年份；train 186 / val 22（test 数字只引用既有文档，不打开）。
逐 lead 的 val 窗口为 **22/21/19/15/11**。训练气候态为 month×hour 的 **8 桶**。
窗口高度重叠，11 个 72h 初始化不是 11 个独立天气事件。

以下取减法轮 `rmse_table.csv` 的 t2m 物理 RMSE；spread 为同一臂两 seed 的 **max−min**，
不是置信区间，不是估计噪声阈值。比较器已经判定这两对长时效同号；没有用比值另立门槛。

| 对比 | lead | 平均 Δ (K) | baseline spread (K) | focus spread (K) | Δ / baseline spread | Δ / focus spread |
| --- | --- | --- | --- | --- | --- | --- |
| RW-B − RW-A | 48h | 1.065106 | 0.659273 | 0.666087 | 1.615576 | 1.599049 |
| RW-B − RW-A | 72h | 1.580487 | 1.613359 | 2.086538 | 0.979625 | 0.757468 |
| no-recurrence − RW-A | 48h | 0.785359 | 0.659273 | 0.561109 | 1.191249 | 1.399653 |
| no-recurrence − RW-A | 72h | 1.173881 | 1.613359 | 0.924683 | 0.727601 | 1.269496 |

48h 配对方向与两轮重复读数不能因 seed spread 被否定；72h 幅度与同臂 seed range 同量级，
限制的是归因强度与泛化，不自动推翻冻结的逐 seed 同号判读。只据这两个 seed 不能估计显著性或裁定
“不可分辨”的新数值门槛；若 D2 的已冻结比较器给出长时效 unresolved，才按本轮停止条款处理。

**单位注意**：决策 `docs/decisions/0010-units-defect-and-two-month-segment.md:17-60,103-116`
要求保留修复前的归档 skill CSV。不能把所有年代的 `climatology_skill.csv` 一律当物理单位；本轮复算
只取已归档的 `rmse_table.csv.rmse`，不读其 climatology 列来造比值，也不改写任何 skill 文件。

可复核命令：

```bash
.venv/bin/python tools/recompute_r7_n1_audit.py --archive outputs/r7_72_rw_b_subtraction --manifests outputs/r7_m2_segment/store/manifests
```

## 4. D1：autoregressive exposure（自回归暴露）

**读法：支持，与训练目标并列的结构证据；尚不能单独归因。**

### 覆盖与未覆盖

- **训练已覆盖**：真实历史输入下，同一 +6h valid time 的 K=3 次草稿反馈；每步重新编码 Y_k。
  `training/r7_streaming.py:135-160` 在内部步间 detach P/Y/Z，属于内部 K 的截断 BPTT。
- **训练未覆盖**：把上一物理转移的预测写回 history 后，再对 +12…72h 真值反传；尤其没有对
  第 8/12 个自由转移的累积误差优化。把同一 target 重复 K 次不等于覆盖这些轨迹。
- **评估做了什么**：`model/r7_rollout.py:72-88` 是 `no_grad`、target-free 自由 rollout；
  `:114-139` 逐 +6h 把所有动态通道的预测写回 history，48/72h 分别为 **8/12 次物理转移**。
  spacetime 模式收到累积 lead，训练只见 lead=6，评估还见 12…72。
- **状态寿命**：`model/process_forecast_r7.py:299-326` 每次新的模型 forward 把 Z 重置；内部 K 的
  carry 不自动跨物理转移。增加 K 不能当作训练自由 rollout 的替代。
- 既有设计契约 `docs/R7_MAIN_MODEL_V2_DESIGN.md:83-100` 明确区分 k/h 并禁止拿评估
  `no_grad` rollout 作 M4 训练。计划 0004 的失败出口 `:323-325` 正是本次审计的依据。

### 解释与局限

这支持暴露失配可能放大长时效误差；不会证明二步可微训练必然改善，也不能因此越过 N2 自动执行 M4。

可复核命令：

```bash
.venv/bin/python -c 'import inspect; from model.r7_rollout import autoregressive_rollout; from training.r7_streaming import backward_streamed_truncated; print(inspect.getsource(autoregressive_rollout)); print(inspect.getsource(backward_streamed_truncated))'
```

## 5. D2：预声明可证伪臂（尚未执行）

新增输出预定为 `outputs/r7_72_frozen_z/`，不使用历史默认目录。三臂 RW-A/RW-B/冻结随机 Z，
seed 41/42，每臂 400 updates；参考臂在本轮从零训练，不重用旧 checkpoint。

冻结 Z 由 driver 层 intervention 安装；原 `solver_cell` 模块与权重保留，cell 前向照算但输出被
同形状的固定随机张量替换。只冻结 `solver_init`/`solver_cell`，门控与提案继续训练。
随机张量为独立 CPU Generator 的 FP32 `[1,N,D]`，run seed+1,000,003，std 沿用 0.02；
同 seed 的所有输入、内部步与物理转移复用同一张量。不是新 solver、不是新模型开关。
这些构造细节进入协议与 checkpoint contract；严格恢复前安装并校验 buffer digest。

- 协议第一次优化器更新前排他写入并回读 digest；普通臂路径保持原语义。
- 主读法仍是冻结 Z − RW-A 的 48/72h 是否同号恶化；RW-B − RW-A 为同轮参照，
  冻结 Z − RW-B 是载体对比。三对不混用，不从载体差值反向挑主判据。
- 判据只来自目标长文 §3；两 seed 不显著，反号不平均成判决。
- 四成本视图、逐 seed/时效表、完整 val case 对齐及实耗需实际运行后补，当前一律**未做**。
- 决策 0021：执行前 `AskUserQuestion` 用户回复「全部授权」，范围为上述三臂、两 seed、400 updates、
  0.45 GPU-h /30min、现有 train/val；失败/partial/unresolved 保留证据即停，不自动重跑。
  回执 `outputs/r7_n1_audit/authorization.json`。另授权工作分支提交与普通 push，仅工程 CI。
  目前没有 `protocol.json` 实验产物，没有 GPU 训练/评估数字。

## 6. D3/D4 与工程证据

复算脚本和工程测试仅检查代码、身份、算术、干预构造与截止时间，不认证机制有效。
planner 连接中断，降级自规划；`/tmp/r7_n1_plan.json` 经 `check_planner_plan.py` verified=true，
它是工程草稿，不是证据。

- `outputs/r7_n1_audit/audit.json`：SHA256
  `d895c96dd4936ebb75adbf9bd23bb8799185bd00b1c41482c6d8182c7a2c4a07`（9667 bytes）；
  stdout 与排他保存文件 digest 一致；脚本 SHA256
  `0acb81c18e5ee6944ad10e4034177ed2732ced8ace00186b0e36e6a88c690dba`。
- 复算 cross_checks 覆盖 40 个 t2m CSV 单元、20 个 paired table 行、10 个 cell/primary 均值，
  train/val manifest 与 case table 对上。test/cache 数组/旧 checkpoint 不读取。
- CPU 初始化成本预备记录 `outputs/r7_n1_audit/cpu_preparation.json`：SHA256
  `5b01bc6525a7e08cedd205087ff07682af6573fee1d2dc32816d2c5f5b93a04a`，0 optimizer updates、0 GPU-h。
  RW-A/RW-B/随机 Z total 参数 2968259/3283157/3283157，trainable 2968259/3283157/3057749；
  forward FLOPs 13904603520/16820126592/16820126592，forward+bwd
  41596684032/54656320512/38112013056。原 cell 前向仍算，冻结减少的是反向与可学习容量，不伪称等算力。
- N1 定向合并 **310 passed**；runner/rollout 回归 **61 passed**；driver 最新单独 **14 passed**。
  K=0 backbone/context/草稿逐位一致，共享 step 的 process/prediction 逐位一致；真实 CPU optimizer
  两步后 init/cell 张量逐位未动，而门控与提案确实学习；strict load 恢复固定 Z 且篡改拒绝。
- 无真实产物的隔离 `git clone` 加本轮树，全量 `pytest -q -rs`：**1795 passed /14 skipped /0 failed**，
  158.97s。8 skip 是 ARCO/M2/regional/D1/real fixture 不在克隆，6 skip 是显式隐藏 GPU；
  2 warnings 为 Lightning `self.log` 在未挂 trainer 的单元测试中发出，不是失败。
- 初次指错 `tests/test_r7_arm_harness.py` 使 pytest exit4、没有测试运行，改用实际的 pilot harness
  `tests/test_r7_72_rw_b_study.py` 后通过；`check_conventions.py --json` 不支持，改用其正式报告格式。
  子任务首轮两个工程 fixture 断言误报已修，保留反证测试，最终定向全绿；没有 GPU 失败尝试。
- 37 条阻断 0 违规；R-052 scheduled runner 拆至 200 行并从冻结清单移出，阈值未改。
  `model_code_sha256` 仍为 `11090929930da4e1259698699cbbf12b3738cdfb2f609c3af516c24399144476`。
- 当前没有本轮 CI run，授权的提交/push 尚待实际执行；不借用旧绿 run 证明当前树。

## 7. D5：下一节点提议的预声明门禁（不是执行授权）

D1 的排序为 **training objective / autoregressive exposure / data regime / forecast state**。
前两项共享同一个 +6h-only 证据，不当作两份独立统计证据。排序按事实的直接性与当前问题相关性，
不使用新数值打分。

- **N2b 候选**：D2 冻结 Z 在 48/72h 仍然 worsened、且同轮 RW-B 的恶化仍在，支持先讨论目标对齐。
  进入前需冻结具体目标与预算，不能牺牲 +6h 更新后称同预算公平；保留既有端点、比较器与 test 封存。
- **N2a 候选**：只有审计把 M3 的 scale/eps 与三类诊断时刻列为最强支持时才可提议开跑；当前 process_weight=0，
  这条前置门禁没有成立。随机 Z 消除恶化只会要求重审 forecast state 载体，不自动证明 M3 是答案。
- **N2c 候选**：数据支撑成为阻碍时，只先修评估/单位/案例追溯，不改判据；扩大数据需新授权。
- **N2d 候选**：已冻结的 D2 长时效比较 unresolved，或审计需要新判据才能区分，则提议停止。
  不追加 seed/臂、不调整阈值，不因本页排序自动启动其它节点。

最终提议必须依据实际 D2 读数或授权阻塞填写；当前 N1 仍 active，没有宣布 D5 最终结论。

## 8. 未做与边界

尚未跑 D2、尚未登记本轮 E/索引/CI、尚未完成工程总门禁与覆盖审计。
未读 test、未下载数据、未租 GPU、未改 `model/` 或 main、未 force push、未关闭 #70–#75、
未重跑/修改归档产物、未进入下一节点。模型接口未新增 solver 开关；训练/评估接入会新增可选干预与
截止时间参数，默认 None 的兼容性需实跑测试确认。依赖不新增，安全/凭据配置不改。
