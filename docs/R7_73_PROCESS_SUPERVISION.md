# M3 / #73：train-only 尺度 sidecar 与三类时刻过程监督

**工程门禁通过，但一次 GPU 尝试为 failed / budget_limited：6/6训练完成、仅7/30评估完成。
第八项评估按连续截止终止；完整三臂对照、paired comparison 与终态四成本表缺失，D5不通过。**
独立只读核验接受已保存的部分审计事实（incomplete-audit-valid），**不接受完整实验**。
N2a停在budget_limited，不重试，不以CPU/CI或protocol存在代替实验，不宣告goal完成、不推进或关闭issue。

## 1. 本轮范围与执行资格

用户本轮明确触发 `docs/goals/n2a-m3-process-supervision.md` 的 M3/N2a：P-A/P-B/P-C 加一次
三臂 × 两个预固定 seed × 400 updates、Ktrain=4。仅改监督因素，使用 #71 后的 RW-A 主干、
已有 M2 source/store/features 与训练 schedule；val 五时效全17变量保留坏变量。上限≤1.0 GPU-h，
主计划单次实验≤30分钟使连续 GPU 执行区间的实际截止更紧，为1800秒。任一评估失败即停并全额计费，
不重试、不放宽判据、端点或案例。无新数据、下载、GPU租赁、main/force/merge/issue关闭授权。

书面具名授权 `r7-73-process-supervision-3arm-2seed-400updates-k4` 已冻结于
`outputs/r7_m3_authorization.json`，SHA256
`a61364394c97484fdd7bc773c02099930b3096395b4dc96b753ba63cc6e0ba83`。它来自真实用户本轮指示，
不是自动通知、余额或代理同意。只覆盖这一尝试，不覆盖补跑或后续节点。

工程版本 `d6c98cf1c33eca5885772c473805af3ef0ad62ba`；`R7 CPU CI` run **36973907623** 的
head SHA精确匹配、completed/success，九主步骤全绿。匿名一手
[run API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/36973907623) 与
[jobs API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/36973907623/jobs?per_page=100)
访问日期 **2026-10-02**；响应与核验回执在 `outputs/r7_m3_acceptance/engineering_ci_*`。
回执 SHA256 `9c121bbecf638ecccc7747a6d2b29effceec1715e762487cb7e85272e5ad6a85`。
这只验证工程提交，不冒称覆盖后续结果/证据登记或证明天气预报改善。实验workflow标签未触发。

## 2. P-A：实际 train 尺度修复与旧产物边界

新构建器 `data/preprocess/r7_process_scale_sidecar.py` 的默认 CLI 是只读 preflight；排他发布必须
给 `--write` 和已核对的 preflight identity。新 sidecar 位于 `outputs/r7_m3_scale_sidecar/`，
按原 fresh-output/BUILD_COMPLETE 发布契约；旧 store 与 checkpoint 不原地修改。

- 实际 train **186** 窗口，history/target帧并集 **188** 帧；只在此并集拟合。
- 有量纲 reference 是每条 proxy 的 train raw RMS，单位明载；除以该 reference 后在无量纲值上
  拟合 train mean/population std，拟合FP64、运行FP32。
- 相对下限为 FP32 epsilon × dimensionless train abs-max；true constant 或 std≤相对下限
  标 degenerate/显式 mask。floor只用于判退化，**不作除数**；masked输出严格0。全部退化、
  active FP32 underflow/overflow、名称/形状/NaN等显式拒绝。
- metadata记录物理单位/原std/floor/缩放后std/有效mask、train样本/帧/时间/原诊断hash、
  source与原data身份；sidecar identity加入新训练contract。

实际 train 的尺度比较（不是 forecast 归因）：

| proxy | 单位 | train raw std | 旧 normalized std | 修后 normalized std |
| --- | --- | ---: | ---: | ---: |
| mslp_gradient_strength | Pa m-1 | 3.330914e-4 | 1.000000 | 1.000000 |
| divergence_850_rms | s-1 | 7.163465e-6 | 1.000000 | 1.000000 |
| vorticity_850_rms | s-1 | 9.601384e-6 | 1.000000 | 1.000000 |
| temperature_advection_850_mean | K s-1 | 4.369278e-5 | 1.000000 | 1.000000 |
| moisture_advection_850_mean | kg kg-1 s-1 | 9.915493e-9 | 0.009915493 | 1.000000 |
| moisture_convergence_850_mean | kg kg-1 s-1 | 1.429338e-8 | 0.014293376 | 1.000000 |
| static_stability_850_500_mean | K | 3.177044 | 1.000000 | 1.000000 |
| vertical_wind_shear_850_500_mean | m s-1 | 4.470436 | 1.000000 | 1.000000 |

8条实际train通道均active；退化/遮罩行为由有反证的CPU合成数学fixture验证，不把fixture当天气真值。
`training/r7_process_diagnostic.py` 的只读审计实际执行，回执
`outputs/r7_m3_acceptance/actual_train_scale_audit.json` SHA256
`6588b7bc9b36b2b487a2baeb17bc01f3910af7ad853a3ba5c63dcff4a1ee34a0`。
旧prediagnostic的数字来自其当时样本范围，本表是本轮实际train并集，不覆盖/回改历史观察。

关键身份：

| 对象 | SHA256 / identity |
| --- | --- |
| sidecar identity | `4fed1c78e4c4c09a41d95457649925b02d0c8ebf89aa47c2ac8dc6496734912d` |
| scale_metadata.json | `d7c2837ea0083b5ce98f98cd02da729c1f12147387838dccad420393b7295cec` |
| sidecar BUILD_COMPLETE.json | `a553781a5b68c3e19bb4f8921e819c856c3d7b4062cead535d637dbfe77047b9` |
| scale preflight identity | `189c33e97e24384081adf2c6aba2e293f937e669627fb9b237feaff705fe07f3` |
| source bytes | `496084a9260bacfaf6293a01d89439c1e49d6afa8f09bc1f51d89a1d1f9bda21` |
| base train data identity | `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07` |
| unchanged model code digest | `11090929930da4e1259698699cbbf12b3738cdfb2f609c3af516c24399144476` |

## 3. P-B：来源、时刻与新训练契约

`training/r7_process_forecast_losses.py` 使用三个分离损失字段，
`training/r7_process_supervision.py` 保存同名tensor与显式UTC ns时刻：

- `input_diagnostics` 是最后历史状态的诊断，时刻等于init，训练/推理可得。
- `future_diagnostic_targets` 是实际train未来标签的诊断，仅作训练监督且detach；明确train split、
  target timestamp=valid time。推理上下文不访问future标签或标签时间，不读真实error。
- `draft_diagnostics` 来自模型own-draft `Y_k`，经固定train atmospheric inverse与球面metric
  得到，保留可微路径。所有内部K对应同一个valid time，**K不加lead**。

legacy `process_weight` 保留默认0.1；新input/future/draft开关分别默认0。同时开legacy与新任务拒绝；
aux启用时要求精确名称、顺序、宽度与active mask，没有 `min(width)` 截断。新任务使用统一helper贯穿
scheduled runner → update_group → streamed truncated backward，全loss与逐K/K累加的梯度有定向核验。
不新增模型参数/solver部件，不改model code digest。

新contract同时绑定sidecar/protocol/三权重与实际train固定FP32 mean/std/有序17通道snapshot及digest。
外层data identity、实际train manifest/store、context均强核；评估新checkpoint必须显式提供sidecar，
再核真实held-out store/root固定inverse。legacy无新contract时签名结构保持。详见决策0028。

## 4. P-C：实际CPU验证与修复记录

- 最终全量CPU：**2272 passed / 9 skipped / 2 warnings，216.37s**。
- 精确工程SHA干净clone：**2267 passed / 14 skipped / 2 warnings，211.99s**。
- 七个尺度/过程/契约测试集合：227 passed/33.96s；最终运行器与receipt反证：52 passed/5.93s；
  治理/guard反证348 passed/23.56s。各集合相互重叠，**不累加成独立计数**。
- 37阻断、600/200硬上限、campaign/goal/index/compile/whitespace门通过；campaign四历史无索引账本
  notes保留，不冒称已机器核数。新增9测试文件，AST基线1176函数/2975断言（增111/284，无删除）。

覆盖analytic涡度/散度/平流、纬度metric、非均匀/下降轴与边界、NumPy离线oracle；常场FP32差分先消去
大offset，零范数梯度有限。FP32/BF16路径敏感算术在FP32，比较的是同一量化输入，不声称BF16无量化损失。
poison future只改loss而不改实际tiny-model forward/halting；no-target推理用会因未来键访问而报错的mapping
反证；shape/NaN/name/order/time/zero variance/全退化等拒绝，streamed与full-truncated全参数梯度核对。
MetPy未安装/未执行；NumPy与analytic检查不是MetPy验证的冒名替代。

如实保留失败：最初完整CPU仅规模marker漂移，随后新增baseline的MIGRATION引用未同步；均修文档而
不弱化检查。fresh CUDA显式init补入后CPU fake漏mock初次失败，补对应fake接口后通过。纳入git跟踪后
R-016发现training直接socket import，禁网guard移至script层，不添加例外。独立工程审阅指出fixed inverse
身份缺口、fresh allocator初始化和cleanup/receipt异常覆盖，逐项修复并有反证；次要stderr OSError也不能
替换原异常。skip不计通过；两个Lightning日志warning不作成功或科学依据。

## 5. 冻结的三臂与实际CPU成本/梯度

| 臂 | input aux | future aux | draft aux | legacy |
| --- | ---: | ---: | ---: | ---: |
| aux_off | 0 | 0 | 0 | 0 |
| input_aux | 0.1 | 0 | 0 | 0 |
| future_draft_aux | 0 | 0.05 | 0.05 | 0 |

总aux预算0.1沿用已有legacy默认与#65权重选择，不扫lambda；未来与draft均分的方案在训练前冻结。
所有臂同config、同seed完整初始化逐字节配对，无容量增减。seed41/42、batch2、lr2e-4、AdamW
weight_decay1e-4、clip1、warmup80/minratio0.1、每100updates验证、patience4/minimprovement0.001、
400updates、K4、FP32；checkpoint按既有+6h normalized val MSE选最优，ties保留早期，不假定选中400。
评估只读val 6/12/24/48/72h，案例22/21/19/15/11，每项最多64，不读封存test。

真实train CPU prepare已实跑、无optimizer step/CUDA初始化；三臂参数都是 **2,968,259**。
FlopCounterMode在enable_grad下计完整forward+各臂实际新loss，再计实际backward，不用参数hook或2×估计：

| 臂 | forward FLOPs（batch2/K4） | forward+backward FLOPs |
| --- | ---: | ---: |
| aux_off | 15,549,227,904 | 46,530,544,896 |
| input_aux | 15,549,227,904 | 46,530,594,048 |
| future_draft_aux | 15,549,227,904 | 46,530,594,048 |

这是full graph profile，**不是streamed训练图的完整成本**；elementwise/normalization/proxy差分等算术
未被counter计数，相同forward FLOPs不意味着三臂实际运行成本相等。吞吐/allocator只能取实际GPU终态。
CPU首个真实train case的各component ownership/norm全记在cpu_profile：aux_off readout无forecast梯度；
input任务到shared/query/readout；future任务到shared/query/readout，draft任务到shared/forecast drafts，
而不训练readout。这是一例梯度描述，不是总体或因果证据。

新输出 `outputs/r7_73_process_supervision/`；canonical protocol digest
`404cf32b8ee8f6c3ff192d46c1de6765abe4ae3fa72967469af800a774fde15d`；code.zip SHA256
`18595abce5acfa9e6f3252342f04eace470a06f48c3eea97c6ba463e3ea02d95`，80相关活跃源/config；
source-tree digest `e2a4e562a592e6b6f92c7d18a4d0fb236e95ebaf676464fccebe9acbe707f29b`。
不做全git/legacy/test清单字节展开。protocol先于任何训练；唯一GPU1 UUID
`GPU-9d1624af-9d77-aa7c-0620-b6cb778f4ced`固定为子进程唯一可见cuda:0。

实际prepare/run命令各一次：

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B scripts/study_r7_73_process_supervision.py \
  --mode prepare --authorization outputs/r7_m3_authorization.json \
  --gpu-uuid GPU-9d1624af-9d77-aa7c-0620-b6cb778f4ced \
  --sidecar outputs/r7_m3_scale_sidecar --output outputs/r7_73_process_supervision

.venv/bin/python -B scripts/study_r7_73_process_supervision.py \
  --mode run --authorization outputs/r7_m3_authorization.json \
  --output outputs/r7_73_process_supervision
```

## 6. GPU终态、完整覆盖缺口与实际成本

`attempt.json` 为 **failed / budget_limited:true / partial:true / finalized:false**。
全部六次training各400updates、四次val检查100/200/300/400，六个selected checkpoint实际均为400。
只完成seed41 aux_off的五lead，以及seed41 input_aux的6/12h，共 **7/30** 评估、**119/510** RMSE格、
**131/528** 次case评估。第八项 seed41/input_aux/+24h 在父deadline等待超时，父仅终止自己持有的
Popen并确认reap；该child日志为空且无result/provenance/metric，不伪造其内部失败原因或峰值。
之后没有任何spawn、retry或finalize。其余23项缺失如实列在独立receipt，不删除已完成或差的变量。

原始7份RMSE/ACC/climatology skill/provenance保持。所有已测17变量都保留在
`outputs/r7_m3_acceptance/partial_all_variable_rmse.csv`（119行，SHA256
`73bf6be237edee7407d154bfdfb7a3c1004e547dabbf07be98f600c80dde42a7`），仅部分运行的只读抄表，
不是完整merged结果。没有two-seed配对覆盖或future_draft_aux评估，**不能给出positive/negative/mixed
三臂forecast判定**；不从seed41两短lead挑数字替代冻结全表。

### 6.1 实测训练吞吐与allocator（六个完成training）

| seed | 臂 | 秒/update | 训练区间秒 | allocated峰值B | reserved峰值B |
| --- | --- | ---: | ---: | ---: | ---: |
| 41 | aux_off | 0.5056528684 | 202.2611473 | 237433856 | 272629760 |
| 41 | input_aux | 0.5877121795 | 235.0848718 | 237446656 | 272629760 |
| 41 | future_draft_aux | 0.7225905556 | 289.0362222 | 237447680 | 272629760 |
| 42 | aux_off | 0.4924545910 | 196.9818364 | 237433856 | 272629760 |
| 42 | input_aux | 0.5542177603 | 221.6871041 | 237446656 | 272629760 |
| 42 | future_draft_aux | 0.7148820485 | 285.9528194 | 237447680 | 272629760 |

7个完成eval各自全新进程，pre-init/初始化后baseline均allocated/reserved=0，allocated峰值
39,590,400–42,861,568B、reserved46,137,344–71,303,168B。只覆盖这7项，不把training峰值代填eval。
参数/FLOPs已实测且6训练吞吐完整，但独立eval内存只有7/30；完整终态四成本表**没有生成**。

14次spawn的UUID/余量前置观察均通过，门槛2390MiB（max历史估计342MiB/本轮owned peak260MiB
+2048MiB）；选定GPU1各次free24107MiB且未见外部compute PID。本机GPU0有邻居但不干预，
政策默认共驻不等于实测GPU1在邻居负载下性能。信号仅见最后自有超时child；无non-owned信号。

### 6.2 全额成本、30分钟边界与账本

- 连续GPU执行计费 **1790.1153369722888s = 0.49725426027008024 GPU-h**，包含14次启动、
  imports、身份核查、所有间隔、已完成/失败eval与自有child清理；未按kernel时间打折。
- `whole_elapsed_seconds` **1805.1085775829852s**，比单次实验30分钟的文字上限多
  **5.1085775829852s**。驱动1800s仅约束GPU执行阶段，执行前CPU身份校验另耗时；**整轮上限
  没有完全兑现**，不声称整轮≤1800或完整预算验收通过，也不事后改protocol/cap。
- 本轮GPU计费未超过1.0h授权，但一次范围已消耗且评估失败即停。剩余数值不是retry许可。
- campaign旧精确已用4.204460154956952，加本轮失败为 **4.701714415227032**，精确余
  **19.29828558477297 GPU-h**；账本逐行显示舍入新行0.4973、累计 **4.7018**、余 **19.2982**。
  显示比精确值保守约8.56e-5h；沿用既有逐行舍入算术，不修写旧历史行。

| 终态对象 | SHA256 |
| --- | --- |
| attempt.json | `525545bc6900f05c0d11f27c2b12bd5db9af39fa3740d52220e7e156009c60ef` |
| execution_attempt.json | `1c5f86f0ec527b628bbdd21f709d2d17e34cee886aec71d47ade3514d69d50d1` |

失败记录、全部109运行文件保持，不改写运行目录。`merged_result.json`、`paired_comparison.json`、
两个seed_result、完整cost/RMSE/case tables与artifact_manifest均缺失；不手工补一个冒充finalizer成功。

## 7. 独立核验、登记与限制

独立标准库checker已在真实终态实跑，7.54s，验证source/sidecar/authorization/codezip/80相关commit
字节、初始化/CPU证据、14spawn精确前缀/ownedPID/余量/连续计费/firstfail/no-later-launch，完成6train
报告/selected checkpoint opaque hash/签名与7val的17变量/时刻/单位/provenance/ACC/skill。
运行目录109文件前后hash/stat/文件集完全不变，最初9pins不变。不import实验finalizer，不读checkpoint
张量或Zarr chunk，不调用网络/CUDA/训练/模型评估。

结论明确区分：**audit_valid:true / incomplete-audit-valid，full_experiment_accepted:false，
D5.full_coverage:false，budget_within_caps:false**。这接受已保存的失败审计事实，不把部分覆盖当PASS。
回执 `outputs/r7_m3_acceptance/independent_m3_73_receipt.json` SHA256
`73afca69d42de3ae8d837b19916f1fc9070cc61bd111782a604682aa00adfe45`，hash sidecar SHA256
`31f9b4cd0e9e94175e946cf9a7ad288c6448aa6ab9bac7b373ead91056ecdbc1`；四checker源已另排他保存在
同acceptance目录（receipt内记录逐文件SHA）。这是只读工程审计，不是OS/syscall全程证明、统计重测
或天气科学接受。没有GPU再训练/重放授权。训练最高config-reproducible，不声称GPU逐位一致；
初始化byte配对不是训练byte可复现。

限制：单冬季区域段、两seed/400updates不证明显著性、收敛、跨季节/年份；过程proxy不等于完整物理真值；
没有matched-Generic，不能作过程结构归因声明；MetPy未执行；禁网guard是防意外不是沙箱；
共驻余量观测不是显存预留，邻居可能影响墙钟且不作校正；旧归档不重跑；未做新的完整安全扫描，
历史Mimosa scanner_enobufs没有安全结论。无模型digest/依赖/凭据/安全配置变更。

本轮D1–D4工程与实测尺度已取得，D5完整对照与四成本交付仍缺；D6只登记本次失败和真实覆盖。
N2a保持 **budget_limited**；下一项仅为审阅此证据及缺口，没有新授权时不补跑评估、重训、改变上限
或进入N3。不关闭#70–#75，不自行宣布目标完成。
