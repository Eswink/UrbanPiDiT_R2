# N1：转向审计与冻结随机 Z 可证伪臂

**状态：D2 已执行；48/72h 主对比均为两 seed 反号，读作「不可分辨」并已停止。只提议 N2d，不宣布目标完成。**

## 0. 起点、范围与证据资格

- 起点 SHA：`efe7d82d11542047897196e4d278cdb18752d6b2`，分支 `r7/weather-reasoning`。
- 主计划：`docs/goals/main-model-v2-campaign.md`，当前节点仍为 N1；本轮长文
  `docs/goals/main-model-v2-pivot-audit.md`。开工对表 failures=0、notes=6。
- 执行前账本：已用 3.7112、余 20.2888 GPU-h；本轮上限 0.45 GPU-h。D1/D3 为 0 GPU-h，
  D2 实耗 **0.3590401737619605 GPU-h**；按账本四位小数记 0.3590，合计已用 **4.0702**、余 **19.9298**。
- 实验代码身份 `c4e7e83a5deaaade1caa21fe82faa064e75b3b72`；工程标记修复与绿 CI 的身份
  `efa410b812cada42ccf0eba3d1f47afd54b3aecb`。后者只改两份规则文档，没有改运行代码。
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

## 5. D2：实际执行与冻结读法

### 5.1 授权、构造与完整性

决策 0021 的执行前 `AskUserQuestion` 用户回复「全部授权」：三臂、seed41/42、每臂400 updates，
现有 M2 train/val，一张空闲本地 RTX3090 顺序执行，≤0.45 GPU-h /30min；失败/partial/unresolved
保留证据即停，不自动重跑。授权回执 `outputs/r7_n1_audit/authorization.json`；另授权工作分支普通
提交/push，仅工程 CI，不含实验标签。执行前重读账本余20.2888，campaign failures=0。

实际命令（**已执行；此处留证，不授权再执行**）：

```bash
.venv/bin/python -B scripts/study_r7_72_frozen_z.py --authorization outputs/r7_n1_audit/authorization.json --device cuda:1 --out outputs/r7_72_frozen_z
```

运行标识为新目录 `outputs/r7_72_frozen_z/`，只有一次尝试；参考 RW-A/RW-B 也从零训练，不重用旧
checkpoint。根/两 seed 协议第一次更新前排他写入并回读；六个run使用同一 canonical protocol digest。
六个训练都400更新、selected_update=400、无early stop；30个val评估完整，case数按6/12/24/48/72h
为22/21/19/15/11，同lead六个运行的case身份相同。17变量共510个逐seed RMSE cell、255个depth0
聚合行，三对各85 cell。这里只读已产出的JSON/CSV核对完整性，不追加模型诊断。

冻结 Z 在driver层安装，原 `solver_cell` 模块/参数对象保留，cell前向照算、输出被固定张量替换；
仅冻结 `solver_init`/`solver_cell`，门控/提案/其余路径仍可学习。独立CPU Generator、run seed+1,000,003、
FP32 normal×0.02；实际shape `[1,1089,192]`，token grid33×33，同seed跨样本/K步/物理转移复用。
协议/checkpoint contract含spec、state_dict含persistent buffer；严格恢复前安装、恢复后校验。
seed41/42的tensor digest分别为 `8004d6c58657cf92aab680e1a48531a73c45a131ce18c1842b5580813ab24216` /
`cfca945e92205d1c0ea653f34fbaaef0caebbc45ce54e7014b769668df676326`。
同seed RW-B与冻结臂的151个原参数张量在安装前相同；两臂从RW-A各转入131张量、ignored=0。
不是新solver，不是修改模型配置开关；冻结降低可学习容量与反向成本，**不声称完全匹配算力**。

### 5.2 t2m 五时效与三个不同角色

Δ为focus−baseline的物理RMSE（K），负为改善；`supported`只对应#60的`improved`读数，不是科学声明。
主问句只读冻结Z−RW-A；另外两对不能替换它。表中逐seed数保留到9位，完整精度见JSON。

| 角色/对比 | lead | seed41 Δ (K) | seed42 Δ (K) | 冻结三态 |
| --- | --- | --- | --- | --- |
| **主问句：冻结Z−RW-A** | 6h | +0.546526798 | +0.601990858 | worsened |
| 同上 | 12h | +0.133628208 | +0.138607318 | worsened |
| 同上 | 24h | −0.825852974 | −1.091755205 | supported |
| 同上 | **48h** | **+0.063121651** | **−0.193231282** | **unresolved** |
| 同上 | **72h** | **+0.325881772** | **−0.151426589** | **unresolved** |
| 同轮参照：RW-B−RW-A | 6h | +0.198608800 | +0.037215869 | worsened |
| 同上 | 12h | −0.043632181 | −0.114769003 | supported |
| 同上 | 24h | −0.022084151 | −0.127470186 | supported |
| 同上 | 48h | +1.061743504 | +1.068480788 | worsened |
| 同上 | 72h | +1.344262252 | +1.817033862 | worsened |
| 单独载体对比：冻结Z−RW-B | 6h | +0.347917998 | +0.564774989 | worsened |
| 同上 | 12h | +0.177260389 | +0.253376321 | worsened |
| 同上 | 24h | −0.803768824 | −0.964285019 | supported |
| 同上 | 48h | −0.998621852 | −1.261712071 | supported |
| 同上 | 72h | −1.018380480 | −1.968460451 | supported |

**分支为 `cannot-distinguish`，`stop_required:true`，提议 `N2d`。** 反号的主48/72h均值为null，
不把反号平均成判决。冻结臂比RW-B长时效好，不能据此说「学到的Z必要」或「已经排除它」。
本轮参照仍复现RW-B长时效受损，但不池化历史轮次。全85格计数（improved/worsened/unresolved）
为：冻结−RW-A **20/53/12**，RW-B−RW-A **29/39/17**，冻结−RW-B **8/54/23**；不跨变量/单位平均。

上一轮的可反驳预测是「换随机Z仍恶化」（长文§1引文）。seed41在两长lead仍正，seed42均负，
因此**没有达到两seed同号支持该预测**，也没有达到反向载体候选分支。D1的(a)倾向不能被继续当作
因果定论；E-202的未训练修正头、E-206的训练期/推理期不等价仍限制该倾向。本轮记录这种张力，
但不拿次对比的改善擅自补全归因，更不新增seed、阈值或臂。

### 5.3 四成本视图与实耗

| 臂 | total参数 | trainable参数 | forward FLOPs | forward+bwd FLOPs |
| --- | --- | --- | --- | --- |
| RW-A | 2,968,259 | 2,968,259 | 13,904,603,520 | 41,596,684,032 |
| RW-B | 3,283,157 | 3,283,157 | 16,820,126,592 | 54,656,320,512 |
| 冻结Z | 3,283,157 | 3,057,749 | 16,820,126,592 | 38,112,013,056 |

FLOPs是CPU、enable_grad下按协议约定测量，registration前冻结；不是GPU实测吞吐。原cell前向仍算，
冻结减少反向与可学习容量。`arm_table.csv.gpu_seconds_total`是训练elapsed求和，不是整轮计费时间。

| seed | 臂 | 训练elapsed(s) | s/update | selected/update数 | training allocated/reserved峰值(MiB) |
| --- | --- | --- | --- | --- | --- |
| 41 | RW-A | 168.28 | 0.420691 | 400/400 | 226.4/260.0 |
| 41 | RW-B | 179.90 | 0.449740 | 400/400 | 254.3/284.0 |
| 41 | 冻结Z | 167.99 | 0.419967 | 400/400 | 243.6/284.0 |
| 42 | RW-A | 170.46 | 0.426162 | 400/400 | 289.2/342.0 |
| 42 | RW-B | 169.94 | 0.424841 | 400/400 | 254.3/342.0 |
| 42 | 冻结Z | 158.29 | 0.395715 | 400/400 | 243.6/342.0 |

training峰值保留实际allocator作用域，reserved并非清空缓存后的隔离显存。四成本**视图**是
参数、FLOPs、训练吞吐、内存：前两项在`arm_table.csv`，吞吐在`training_table.csv`，内存在
`memory_table.csv`；`case_table.csv`是案例覆盖而非第四张独立成本表。成品登记前校正此处术语，
不是更改测量或判据。**evaluation独立峰值缺失，内存视图不得称验收齐全**（见§5.4）。

`attempt.json.status=success`、failure_reason=null：GPU区间 **1292.544625543058 s**，计费
**0.3590401737619605 GPU-h**（≤0.45）；整轮墙钟 **1297.8983452636749 s**（≤1800）。GPU钟从
select_device前起，到最终GPU同步止，包含setup/训练/validation/评估/阶段间时间；CPU收尾不重复计费。
训练subtotal1017.1016330691054 s、评估subtotal102.16334865335375 s，不用其和替代整段实耗。
`merged_result.budget.whole_round_elapsed_seconds`字段实际填GPU区间（1292.5446），不是整轮墙钟；
真实whole值取`attempt`，保持原JSON不改。账本行四位舍入与索引保留精确实耗，差小于既有1e-4容差。

### 5.4 独立工程审阅发现与证据边界

独立只读审阅（源码+JSON/CSV，未运行模型）发现三项，不因工程CI绿而掩盖：

1. **截止时间检查不覆盖validation内部case**：`training/r7_scheduled_runner.py:108-128,367-377`。
   进入score前检查，score内部未接deadline；validation中途过期会继续到外层检查。已有测试只覆盖
   进入前过期。此次实际GPU/whole时长均低于cap，但**不能声称代码对所有路径提供硬时限保证**。
2. **evaluation峰值继承training峰值且未指定device**：`scripts/study_r7_72_frozen_z.py:219-243`。
   每次evaluate前未reset峰值，两seed的30次eval均255424000 bytes，等于末臂training峰值；
   不能解释为该eval的独立峰值。保留表行，不补造/清理数字；此部分成本证据不满足独立测量要求。
3. **finalizer未拒绝同步缺失secondary cell**：`training/r7_arm_harness.py:374-397` 与
   `scripts/study_r7_72_frozen_z.py:289-313`。#60检查双方seed一致与pair存在，不知道声明的完整全集合。
   同步只剩seed41/单个6h-u10仍可生成block。此次终态的精确6训练/30评估/510cell已只读核齐，
   **不表示future guard已修复**，不能用primary reader的缺项unresolved测试代替finalizer拒绝测试。

本轮触发不可分辨即停止，实验代码与归档不热改、不重跑。三项留作任何将来使用driver之前的工程
前置事项，不是下一实验授权。另有上述merged字段作用域误名；不回写冻结产物。

## 6. D3/D4：复算、身份与工程证据

### 6.1 可复核结果与产物摘要

D3标准库脚本 `tools/recompute_r7_n1_audit.py` 的SHA256为
`0acb81c18e5ee6944ad10e4034177ed2732ced8ace00186b0e36e6a88c690dba`，只读六个具名CSV/JSON/manifest，
核40个t2m CSV单元、20个paired table行、10个cell/primary均值；不读cache数组/test/checkpoint。
再次实跑stdout hash与`audit.json`完全相同；六输入digest在`input_pins.json`中前后一致。

D2只把归档`code.zip`的四个纯元数据模块提取到临时目录，用原#60重算JSON/CSV的table/pairs/primary，
`paired_comparison.json` **逐字节相同**，255行/3×85cell，0 GPU-h，未执行模型或打开checkpoint。
第一次序列化误加末尾换行而assert失败，三个内容字段已经相同；按归档writer的无末尾换行格式再次
核验，digest完全匹配。Mimosa拒绝工作态脚本的动态exec，改用hash固定ZIP中的四个文件常规导入；
没有放宽扫描或身份校验。回执 `outputs/r7_n1_audit/metadata_replay.json`。
**这只是比较器/算术可重算，不是GPU训练逐位复现**；本轮训练最高记config-reproducible，未做重训。
独立metadata核对交叉确认source/receipt/preflight/BUILD_COMPLETE与checkpoint引用，**未重新hash source.nc、
checkpoint bytes或cache数组**；它不是再次认证这些tensor。汇总CSV与local CSV转写逐字一致；
两个独立RMSE累计路径有100/510 cell末位1–2 ULP差，最大相对2.753497910377443e-16，未改读数/判据。

可复核最终全集合/预算的标准库命令（只读已有JSON/CSV，不执行模型、不读test）：

```bash
.venv/bin/python -B - <<'PY'
import csv, hashlib, itertools, json
from pathlib import Path
out = Path('outputs/r7_72_frozen_z')
load = lambda name: json.loads((out / name).read_text(encoding='utf-8'))
protocol, merged, comparison, attempt = map(load, ('protocol.json', 'merged_result.json', 'paired_comparison.json', 'attempt.json'))
seeds, leads = (41, 42), (6, 12, 24, 48, 72)
arms = ('process_spacetime_rwa', 'process_local_solver', 'process_local_solver_frozen_z')
variables = 't2m u10 v10 mslp z850 t850 q850 u850 v850 z500 t500 q500 u500 v500 z250 u250 v250'.split()
with (out / 'rmse_table.csv').open(encoding='utf-8', newline='') as stream:
    rows = list(csv.DictReader(stream))
keys = [(int(r['seed']), r['arm'], int(r['lead_hours']), r['variable']) for r in rows]
assert len(keys) == len(set(keys)) == 510
assert set(keys) == set(itertools.product(seeds, arms, leads, variables))
assert len(merged['evaluation']) == 30 and len(comparison['table']) == 255
for seed, arm in itertools.product(seeds, arms):
    report = load(f'seed{seed}/training/{arm}/training_report.json')
    assert report['updates_this_run'] == report['selected_update'] == 400
    assert not report['early_stopped'] and report['selection_split'] == 'val'
assert all(len(pair['cells']) == 85 for pair in comparison['pairs'].values())
assert comparison['primary']['branch'] == 'cannot-distinguish'
assert comparison['primary']['stop_required'] is True
assert attempt['status'] == 'success' and attempt['gpu_phase_elapsed_seconds'] <= 1620
assert attempt['whole_round_elapsed_seconds'] <= 1800
assert attempt['gpu_hours_charged'] == attempt['gpu_phase_elapsed_seconds'] / 3600
assert merged['budget']['whole_round_elapsed_seconds'] == attempt['gpu_phase_elapsed_seconds']
print({'runs': 6, 'evaluation': 30, 'rmse_cells': len(rows), 'gpu_h': attempt['gpu_hours_charged'], 'whole_s': attempt['whole_round_elapsed_seconds'], 'independent_eval_peak': 'MISSING'})
PY
```

| 产物（`outputs/r7_72_frozen_z/`，除另注） | SHA256 |
| --- | --- |
| protocol canonical digest | `e19ef488be60136364702b1df389e5f58be7ebf3845487289139e10d30231e01` |
| protocol.json文件字节 | `03147e681c1f8624cab826e38c28ece45042f84d0221ac2865281501f0167066` |
| code.zip | `5fd26146af2a7d11016fb769d67390f5daa23a73620de9ae83e2e6cc38a35a0a` |
| merged_result.json | `818b05512f51698c71c4ffc711e2936607d6a4579970ca3c305ff1a5184dbe86` |
| paired_comparison.json | `d7a345c17923cd49d1b03f2cbea767c95579e5d18a043965bf29fb697c222a64` |
| arm_table.csv | `d5bc1280722158040030da4d68939901e2da2f9ea36fc7d30e82848acf127f52` |
| training_table.csv | `67586b58725c656569beb6e359bfa85dc117731dacb25b096c8ee1bca3df4c01` |
| memory_table.csv | `df9dd5885cd6b165a2a8474f58f5bd1a2352a426a80fcabf4a096374108fe119` |
| case_table.csv | `b1a336eb2a48affc62a69fa0d5cb0cc4f65ea45e4ebacd2682bb4cbd064d64e6` |
| rmse_table.csv | `ffd69dace3f7782a27e9fc2d4adbfd510a4d8f9c9ab28c7d90a820853cb26aa2` |
| attempt.json | `a4d8da66f8d25f68c89b5ccbf680f498b000fb82e6884e0099b6b9514074f226` |
| `outputs/r7_n1_audit/audit.json`（9667 bytes） | `d895c96dd4936ebb75adbf9bd23bb8799185bd00b1c41482c6d8182c7a2c4a07` |
| `outputs/r7_n1_audit/cpu_preparation.json`（0更新、0GPU-h） | `5b01bc6525a7e08cedd205087ff07682af6573fee1d2dc32816d2c5f5b93a04a` |
| `outputs/r7_n1_audit/authorization.json` | `a229560b217448a9c6d6b4860ee61c0781a02923830f9b2f38dab08905c5b05c` |
| `outputs/r7_n1_audit/input_pins.json` | `e866d202005b4fd94bfbf9737b01966aa9aeaa3d717c05da0589eb935e3f78ec` |
| `outputs/r7_n1_audit/metadata_replay.json` | `f48928d557cacd451a166aa5428d727add87331b0b062226bdda9934335d95d8` |

代码commit见§0；working-tree digest `653daea91a8e7a54541c3e100753fa93cc2f6c23b0e4cf48fcb2d810561cc60b`。
ZIP的953项与该commit的`git show <sha>:<path>`字节核齐；`working_tree_modified:true`是原有两个
未跟踪工具文件（`.zcode/agents/web-researcher-backup.md`、`.zcodeignore`），不在ZIP/提交中。
数据identity `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07`；以下来源分别核对，
不把source与dataset身份冒充成同一个hash：

| 身份 | SHA256 |
| --- | --- |
| source bytes（仅身份hash，不解码test） | `496084a9260bacfaf6293a01d89439c1e49d6afa8f09bc1f51d89a1d1f9bda21` |
| receipt | `8a681e90a91ff5099f010b3c05a55ff7724ef42fbdd38c70a26ae10c8c23af58` |
| local preflight | `40dcec9eb8df3fe16dd9aab4c7c050642a521aa000cd1f1548394dd2900d13f8` |
| BUILD_COMPLETE | `a553781a5b68c3e19bb4f8921e819c856c3d7b4062cead535d637dbfe77047b9` |
| train manifest | `60e56464cbc88a90abfeae4793f97dfea4f6c786deb210db2e5919bf5732a4a2` |
| val manifest | `218512c8a1490dfc72f8c8639f2c48d3d950e0c06f53c92e6777318c015398b7` |

### 6.2 工程验证与CI身份

- N1定向合并 **310 passed**；runner/rollout回归 **61 passed**；driver最新 **14 passed**；
  marker+driver **134 passed**。K=0未干预路径与共享step逐位等价；CPU两次optimizer更新后solver
  init/cell逐位未动而门控/提案学习；strict load round-trip和篡改拒绝有反证。它们不证明机制有效，
  也不覆盖§5.4所列缺口。
- 初版隔离clone全量1795 passed /14 skipped /0 failed，158.97s；最终代码修复commit `efa410b`
  的精确干净clone实跑 **1795 passed /14 skipped /0 failed /2 warnings**，161.04s。
  六skip因显式隐藏GPU，八skip因ARCO/M2/regional/D1/可选真实fixture不在clone；不把skip算通过。
  warnings是既有Lightning未挂trainer的`self.log`提示；installed-wheel测试包括在全量内。
- 初次commit `c4e7e83` 的CI **36690064571 failure**，pytest阶段失败：最后设备映射改动使driver
  399→402行，R-021 marker还写34而actual35。本地反漂移测试重现，只向前修数字为35；不改阈值/测试。
  修复commit `efa410b812cada42ccf0eba3d1f47afd54b3aecb` 的CI **36691526554 completed/success**，
  九个主步骤全绿。run/jobs一手API已保存，**远端test计数未取得**；1795是本地clone数字。
- 37阻断0违规；scheduled runner拆至200行并移出R-052冻结清单，600/200阈值不变。
  `model_code_sha256`保持 `11090929930da4e1259698699cbbf12b3738cdfb2f609c3af516c24399144476`。
- planner连接中断，按技能降级自规划；`/tmp/r7_n1_plan.json`经checker verified=true，不当证据。
  指错不存在的测试路径曾导致pytest exit4/未运行，改用真实文件后通过；不支持的`--json`报告选项
  曾exit2，改用正式格式。早期两处工程fixture误报保留反证修复，没有GPU失败尝试。
- Mimosa提交/push扫描报`scanner_enobufs`并按既有fail-open继续：**扫描无结论，不是安全通过**。
  未运行另一次深度扫描、未修改扫描/凭据配置；没有可用凭据写入源码、产物或测试。

一手CI来源（经web-researcher取得并用curl保存核对，访问日期 **2026-09-30**）：
[run36691526554](https://github.com/Eswink/UrbanPiDiT_R2/actions/runs/36691526554)、
[run API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/36691526554)、
[jobs API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/36691526554/jobs?per_page=100)。
最终登记提交与其CI绑定写入本轮长文进度，**不再为CI尾记录改本证据页并产生digest递归**。

## 7. D5：下一节点提议与预声明门禁（不是执行授权）

D1排序仍为 **training objective / autoregressive exposure / data regime / forecast state**；前两项
共享+6h-only的结构证据，不当两份独立统计结果，不是因果主次排名，也不是已获授权的修复清单。

**本轮只提议N2d停止**：冻结Z−RW-A的48/72h均unresolved，已命中长文§3第3条/§5停止条件3。
N1状态paused、current_node保持N1待用户/独立复核；不自动把主计划移到N2d，不追加种子或臂。
下列门禁沿用实验代码提交`c4e7e83`中本页§7的执行前提议，主计划§2仍是节点通道的权威；
它们不是给N2b另加一个科学判据，不能拿排序越过当前停止条件。长文§3未触发的两条因果措辞按
**运行前已冻结协议**限域为「本预算下必要性排除/载体候选」，不是普遍主因证明；本次只读第三分支，
不回改长文判据，也不据未触发分支宣称机制：

- **N2a**：N1需把M3 scale/eps与三类时刻列为最强支持；当前process_weight=0且未列最强，门禁未成立。
- **N2b**：预提议要求随机Z在48/72h仍worsened且RW-B同轮恶化仍在；后者成立，前者不成立，不能开跑。
  任何未来目标对齐方案先冻结具体目标/预算，不能把牺牲+6h更新说成公平，仍用既有端点/比较器。
- **N2c**：若数据支撑限制成为主问题，只先修评估/单位/案例追溯，不改判据；扩大数据须新授权。
  当前反号并不授权下载或改val/test；不自动执行。
- **N2d**：任一主长lead unresolved或需要新判据即提议停止；本轮已触发。先保留并审阅这些读数与
  §5.4工程缺口，若以后重开，须用户/独立复核推进节点、重新recheck、具名预算/协议和决策0021授权。
  既有停止不自动获得「追加seed就能辨别」的豁免。

## 8. 提示词到产物的覆盖审计与未做

| 要求 | 实际证据 | 判定边界 |
| --- | --- | --- |
| D1四块、每块file:line/命令/支持读法 | §1–4；D3的audit.json | 0GPU-h；只支持解释候选，不作因果归因 |
| D2构造、2seed×400、先冻协议、val/test纪律 | §5.1、6run/30provenance、同digest | 运行success；主问句unresolved，非机制通过 |
| D2五时效/全部变量/案例 | §5.2、510RMSE/255table/3×85cells、case_table | 精确全集合核齐；future completeness guard缺口保留 |
| D2四成本视图/预算 | §5.3：arm_table的参数/FLOPs、training_table的吞吐、memory_table的内存；attempt.json | 实耗未超cap；独立eval峰值未完成，时限guard不完整；case表非成本 |
| D3机械复算 | tools/recompute_r7_n1_audit.py及stdout digest；归档元数据比较器重算 | 无模型/重训；比较器字节可重现，不宣称训练bit-reproducible |
| D4证据/E/索引/brief/CI绑定 | 本页；E-207–E-212；record:n1-pivot-audit-frozen-z-unresolved；本轮长文进度 | outcome_class:audit；needs-review；证据commit在后续索引精确绑定 |
| D5排序与下一节点门禁 | §7 | 只提议N2d；不推进节点或自宣完成 |

**未做**：GPU重训/再次模型评估/逐位训练复现；§5.4三项工程修复及eval独立峰值补测；显著性/季节或
区域泛化；matched-Generic；M3/M4/M5/确认轮；独立verifier的目标完成判定。成本视图的缺口不能标PASS。
未读test manifest/场或评分；source bytes仅作身份hash。未下载/租GPU、未改`model/`/main、未force/merge、
未关闭#70–#75、未重跑/改写归档、未进入下一节点、未设定时器或后台续跑。

**接口/兼容性**：runner新增可选intervention/deadline，evaluator新增deadline并按contract恢复intervention，
默认None的ordinary native/generic/process契约与权重回归通过；成本writer仅协议声明时新增trainable列，
旧schema保持。全模型dtype转换改变固定FP32 buffer会明确拒绝，BF16 autocast局部cast覆盖；依赖不新增。
安全/凭据配置未改，扫描无结论如实保留。后续第一个可做的动作是审阅本页的停止记录，不是开下一实验。
