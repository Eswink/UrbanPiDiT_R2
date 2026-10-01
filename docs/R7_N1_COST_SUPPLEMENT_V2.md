# N1 v2 独立 evaluation 成本补测：具名一次执行与成本验收

**实跑状态：success，30/30项、逐项零 allocator 基线与原 val 精确重放，四成本表及 cost_views 齐。
这是成本审计补齐，不是新的机制结论。N1 保持 paused/current_node=N1，原48/72h cannot-distinguish、
N2d仅提议不变；不自行宣布目标完成。**

## 1. 具名授权与执行边界

工程准备与当时未答授权的事实保留在 `docs/R7_N1_COST_V2_PREPARATION.md`，不回改该历史证据。
用户随后明确回复 **「明确授权，刚刚没有看到」**，回答的是执行那一刻 AskUserQuestion 的具名范围
`n1-evaluation-cost-supplement-v2-with-residue-probe`：P1≤60s、仅P1零残渣才P2≤120s、原六selected400
checkpoint×五原val时效（6/12/24/48/72h）共30次全新子进程，0训练更新、test不读；本机默认共驻
（启动/每spawn前只读余量≥2048 MiB）、GPU全计费≤900s/0.25h、整轮≤1200s，失败保留即停全额计费不重试。
不信号邻居、不新增训练/臂/seed/端点/容差，不下载/租GPU/main/force/关闭#70–#75/推进节点。

回执 `outputs/r7_n1_cost_v2_authorization.json` SHA256
`0770b4f80ff1b95b0223d10cf80b498c524ffc626e1ece91f10ae3199db35edf`。
回执如实注明是用户对 AskUserQuestion 的随后明确回答，不把先前未回答当许可，不复用v1授权。

执行起点 `b714458ed2ed1e29282d024ae5b72bf2c4839215`；八份计量/探针/测试源码逐字节对上准备提交
`b227020ca3d4932a765a6cd7a79e26fc659cdeaf`（CI36855190840九主步骤成功、精确clone1988passed/14skipped）。
执行前对表0失败/4notes（四旧账本行无索引支撑如实保留），账本4.0762/余19.9238；30任务、六checkpoint、
source/manifest/receipt/preflight/BUILD_COMPLETE只读核齐，新输出不存在且排他创建。

实际命令（只执行一次）：

```bash
env -u CUDA_VISIBLE_DEVICES PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  scripts/measure_r7_n1_eval_cost.py \
  --authorization outputs/r7_n1_cost_v2_authorization.json \
  --device cuda:1 --out outputs/r7_n1_eval_cost_supplement_v2
```

父进程、探针、worker均禁Python socket出网，原归档代码精确提取/哈希后才import；不是当前HEAD模型。
物理GPU1 UUID `GPU-9d1624af-9d77-aa7c-0620-b6cb778f4ced`固定为子进程唯一可见卡（显式cuda:0），实际UUID重验。
启动free24107MiB；调用前后最小free23713MiB，全部≥2048。**启动/30个spawn/调用前后均未观测到外部compute PID**，
本轮没有发任何邻居信号或要求腾卡。采用共驻政策不等于实测了有邻居负载时的性能，瞬时竞争仍不能排除。

## 2. D1 原因探针：预声明三态①，P2不触发

P1为纯torch 2.11.0+cu128，单次1024² FP32 matmul，删除全部张量引用后gc/sync/empty_cache，记录allocated/
reserved字节及完整memory_snapshot；私有API只用于P1诊断，不进入30项成本路径。

| 阶段 | allocated B | reserved B | snapshot |
| --- | ---: | ---: | --- |
| 全新进程初始 | 0 | 0 | 0段 |
| matmul删除、gc/sync/empty_cache后 | 8,519,680 | 20,971,520 | 1段；active_allocated8,519,680 / inactive12,451,840 B |
| `_cuda_clearCublasWorkspaces()`与再次清理后 | 0 | 0 | 0段 |

因此按冻结§3.1读为 **torch-releasable-workspace-family（torch进程级可释放工作区族）**，
`attribution_confirmed=true`、`requires_p2=false`；**条件P2未执行**，不是漏项或skip。
探针内记录0.9619096238166094s；父计费起点至P1子退出4.174674681s，包含import/pins/setup，≤60s。
`probe_residue.json` SHA256 `b94aa9383e453635bce5a67edaa38696e816a630ef6dda23f8a09b7474f134b9`；
snapshot中段/块字节及digest保留，无snapshot error、无guard_refusal。

**已确认**：纯torch能复现清理后allocator残渣且指定释放接口可清零，与v1第二项前非零基线形态一致。
**未确认**：v1当时残渣的具体字节/持有对象没有记录，不能把本次8,519,680B回填为历史值；也不能定位到
一个具体cuBLAS/cuBLASLt对象或断言项目不存在其它持有。v1原失败记录保持原样，不为追溯再跑P2或v1。

## 3. D4 三十次独立 evaluation 与四成本视图

30项全集＝2seed×3arm×5lead；**30个不同PID与launch ID、30父issued/claimed/exit记录**，每child正好一项，
正常退出0。每spawn及调用前后观察绑定物理UUID/阈值/顺序，source/checkpoint与实际归档import身份重验。
每项 `baseline={allocated_bytes:0,reserved_bytes:0}`，`0<peak_allocated≤peak_reserved`；显式sync后读峰值。

原逐lead案例数22/21/19/15/11，六checkpoint合计528次案例评估；17变量×30＝510RMSE格。
30份 `rmse.csv` **文件SHA256逐份等于原归档**；provenance的checkpoint/data/manifest、channels/units、
lead/step、选项、init/valid_times/逐case MSE/干预规格精确相同。时间/新输出路径等非身份字段不冒称整个
provenance字节相同。未新增容差、未读test或训练更新，原paired_comparison仍cannot-distinguish。

### 3.1 新测 independent whole-evaluate 峰值

每臂10行（两seed×五lead）；以下为范围而非跨cell平均：

| 臂 | allocated峰值范围 B | reserved峰值范围 B | 十次整调用elapsed合计 s |
| --- | ---: | ---: | ---: |
| RW-A / process_spacetime_rwa | 39,590,400–43,148,800 | 46,137,344–71,303,168 | 94.43320592865348 |
| RW-B / process_local_solver | 42,014,208–45,923,840 | 52,428,800–77,594,624 | 95.08049425389618 |
| 冻结Z / process_local_solver_frozen_z | 42,014,208–45,240,832 | 52,428,800–77,594,624 | 100.78523198422045 |

总整调用290.2989321667701s，不等于GPU计费区间：计费额外含P1、import/pins/load/setup、inter-child gaps、
清理和查询。这里是整 evaluate_local（含模型装载/IO/指标）本进程allocator峰值，不是独立forward显存/
延迟、设备总显存，亦不是原运行当时的历史峰值。30进程独立基线由构造保证，不在评估时调用私有clear。

### 3.2 参数/FLOPs/训练吞吐引用原归档，不重新训练

| 臂 | 参数总数 / trainable | forward FLOPs | forward+backward FLOPs |
| --- | ---: | ---: | ---: |
| RW-A | 2,968,259 / 2,968,259 | 13,904,603,520 | 41,596,684,032 |
| RW-B | 3,283,157 / 3,283,157 | 16,820,126,592 | 54,656,320,512 |
| 冻结Z | 3,283,157 / 3,057,749 | 16,820,126,592 | 38,112,013,056 |

六个原training吞吐行各400updates，seconds/update为seed41的0.42069127134047446/0.4497401515999809/
0.41996739951893686与seed42的0.4261623802990653/0.4248410267708823/0.3957153623574413（依表顺序）。
这些是历史原值，不是本次GPU新测；training memory六行也保留原值，原错误eval峰值未被复制。

| 终态表 | 数据行数 | SHA256 |
| --- | ---: | --- |
| parameter_table.csv | 3 | `c4593b827fee558386d321bf79d141946cee6fbc94f0888eaa2930e1ee72503b` |
| flops_table.csv | 3 | `8a0f2a00e2d498a756367cef2a635a36780ebe4f182f526470d47b766c27c050` |
| training_throughput_table.csv | 6 | `fdd4582305cd041b02ef78c63df8b3dcdebfb5a8da11f1f51ea0261eaa52017c` |
| memory_table.csv | 36（6原train+30新eval） | `f723830c5d1ee21727fd0572527a26a652cf8577da2a12901e87198ed5bbedb3` |

`cost_views.json` SHA256 `42606c5782d6f25d90e1686aefbf3f52d74fbf75423655d81c44ee935fbb98db` 绑定全部四表摘要与
非科学/limitations。历史原页/成本表不改，v2新增可引用的独立eval成本证据。

## 4. 预算、账本与停止

attempt/result均success，failure_reason=null，owned_child_exit_unconfirmed=[]；30summary、无失败、无自动重试。

- **全额GPU计费**：461.74807197228074s＝**0.12826335332563354 GPU-h**，≤900s/0.25h。
- **真实整轮墙钟**：463.4213050529361s，≤1200s；P1含启动计费≤60s，P2因条件不触发。
- campaign精确：4.076196801631319＋0.12826335332563354＝**4.204460154956952**，余
  **19.79553984504305 GPU-h**。账本新行记0.1283，显示已用**4.2045**/余**19.7955**，舍入不扩授权。
- 原N1＋v1失败＋本v2累计**0.49326015495695175 GPU-h**。这超过**原N1科学实验0.45h**的历史数值；
  本v2是用户另授≤0.25h的独立成本审计范围，不回改原0.45h授权/记录，也不冒称三段合计≤0.45。
  新成本全部记campaign24h账本，余额仍不构成任何新实验/重试许可。

D5只保留N2d停止提议；成本齐不改变原48/72h cannot-distinguish、不自动推进节点，实验一次范围已用完。

## 5. 身份摘要与可复现等级

产物目录 `outputs/r7_n1_eval_cost_supplement_v2/` 排他创建。关键摘要：

| 对象 | SHA256 / 身份 |
| --- | --- |
| v2 protocol canonical | `877d0cabef976b2896ab3ffd817e78ba6f92ac03e218c6920c1c0ea32a9a510f` |
| protocol.json文件 | `ebac68964738af15f7c6cdab7909c2fdc89c830b87da965fb9fc20718ddb804e` |
| attempt.json | `e172da59a39ceb9b456e2e910b0e616e67876b8a1e51df34e2d530e1f1cae9e0` |
| result.json | `7b758e2d428f9a256e8b60fb97535c479b9114b9098aa20d0d3aa34b5b076b0c` |
| measurement_code.zip（八source文件） | `b14b3252f1b440c942ce220ae1e6707c9c9213cfdbc20a82423e294f4c2cdae0` |
| 原code.zip | `5fd26146af2a7d11016fb769d67390f5daa23a73620de9ae83e2e6cc38a35a0a` |
| 原model code digest | `11090929930da4e1259698699cbbf12b3738cdfb2f609c3af516c24399144476` |
| data identity | `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07` |
| source（37,734,176 B） | `496084a9260bacfaf6293a01d89439c1e49d6afa8f09bc1f51d89a1d1f9bda21` |

可复现等级：本轮30val的RMSE文件逐字节一致、固定provenance/case/MSE语义精确重放；**不**声称未来GPU
allocator/墙钟逐位一致，原training仍config-reproducible。一次授权内没有再重跑本v2目录，登记只读重验。

v1页、review、measurement_code.zip三个hash仍为4b350357…/9540549b…/b7f20e7b…，均未改写；原归档input pins
前后全部重验不变。计量修复不改模型/评测器/数据或科学比较器，只引用其归档源。

## 6. 核验与局限

主链只读 `verify_registration`、`collect_rows`、`verify_pins`、`verify_probe_report` 实跑通过，并逐表/字节/案例/
预算机械重数，不能只凭attempt success接受。独立只读复核结论与登记CI的精确SHA/run id写入本轮目标进度，
不为尾CI递归改本页digest。准备工程CI36855190840和登记前359passed/精确clone1988passed/14skipped是已确认
工程证据，不代替本次GPU产物或科学判定。一手[准备run API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/36855190840)、
[jobs API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/36855190840/jobs?per_page=100)，访问2026-10-01。

本次GPU执行后没有源码/依赖/接口新改动；沿用已通过的v2接口。未读取held-out test、未新增训练/seed/臂/端点、
未下载/租GPU、未信号任何非实验进程、未改main/force/关闭issue/推进节点。无凭据/安全配置变更。
Mimosa提交/push扫描曾scanner_enobufs无结论，未另做完整项目安全审计，不宣称安全通过。

限制：峰值是whole-call本进程allocator而非总卡/forward；共驻余量查询不预留显存，且本次无邻居负载观测；
私有释放API版本依赖且只归因工作区族，v1历史残渣不能逐字节追溯；原两seed冬季段不支持显著性、跨季节或
机制声明；成本补齐不等于matched-Generic/主模型机制门槛成立。Python禁网防意外不等于沙箱。

下一项仅为用户/独立复核审阅本次审计与既有N2d提议；没有新的GPU、训练、确认轮或下一节点授权。
