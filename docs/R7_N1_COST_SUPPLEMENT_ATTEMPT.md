# N1 独立 evaluation 成本补测：一次失败尝试

**状态：failed，完成1/30项；独立内存证据仍不齐，完整成本验收继续阻塞。已按授权停止，不自动重试，不宣告N1目标完成。**

## 1. 具名授权与实际范围

2026-09-30用户输入「继续授权」后，执行前通过`AskUserQuestion`确认一次具名范围，回答
**「授权上述一次补测 (Recommended)」**。范围为`scripts/measure_r7_n1_eval_cost.py`：
原六selected400 checkpoint（RW-A/RW-B/冻结Z，seed41/42）×6/12/24/48/72h原val，共30项；
**0训练更新**，一张空闲本地RTX3090，GPU计费区间≤324s（0.09GPU-h），整轮≤600s。
参数/FLOPs/训练吞吐只引用原归档，独立allocated/reserved峰值才是新测量。

授权例外只覆盖这30次原验证评估的成本重放；原`code.zip`、checkpoint、数据和原证据页只读。
原RMSE/case必须精确一致；身份不符、基线非零、占卡、超时、失败/partial均保留证据与实耗即停，
**不自动重试、不放宽判据**。不读test、不下载、不重训或加臂/seed、不租GPU、不改main/force/关issue/推进节点。
回执`outputs/r7_n1_cost_authorization.json` SHA256
`e222fd17802aac5b80814632a387a752cfcf8fd93bc08441b1cbcad2ca8bb548`。

执行前只读核验原30任务、六checkpoint/source/manifest/receipt/preflight/BUILD_COMPLETE身份；
主计划对表failures=0/notes=6、原账本4.0702/19.9298。用户随后说已手动终止占卡任务并要求继续。
实测GPU1 UUID `GPU-9d1624af-9d77-aa7c-0620-b6cb778f4ced` 无compute进程、18MiB/0%，才启动。

另有单个GPU0 worker PID2823176的SIGTERM授权，**未发生任何信号发送**：首次因项目Python缺
`os.pidfd_open`在发信号前失败；改用libc后原PID已不存在，GPU0被新PID2844662占用。
没有把旧PID许可扩展到替代进程，没有终止共同父进程/其它任务或重置GPU。授权及两次no-signal记录在
`outputs/r7_n1_cost_gpu_release_authorization.json`、`r7_n1_cost_gpu_release_result.json`、
`r7_n1_cost_gpu_release_libc_result.json`。GPU1由用户手动腾出，不宣称是代理清卡成功。

## 2. 代码、冻结协议与运行命令

计量代码commit `1e03f82067e429185a4dc41b6d76da0020d6dece`；准备阶段工程CI36718455666成功，
精确clone1861passed/14skipped/2warnings（164.49s），定向232passed/66成本反证；这些不认证GPU计量。
CI一手[run API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/36718455666)、
[jobs API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/36718455666/jobs?per_page=100)，
访问2026-09-30；原响应与验证记录在`outputs/r7_n1_cost_ci_readback.3iPX7n/`，远端test计数未取得。

新目录`outputs/r7_n1_eval_cost_supplement/`排他创建；protocol在任何CUDA/评估前写入并重读digest。
实际命令：

```bash
env -u CUDA_VISIBLE_DEVICES PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  scripts/measure_r7_n1_eval_cost.py \
  --authorization outputs/r7_n1_cost_authorization.json \
  --device cuda:1 --out outputs/r7_n1_eval_cost_supplement
```

父进程将物理UUID固定为子进程唯一可见卡，worker显式用`cuda:0`并核实际UUID；执行归档评估器，
不是当前HEAD的模型。每seed/arm拟一个fresh worker、五lead逐项gc/sync/empty_cache/零基线/reset，
调用后显式同步读取peak，并严格复核case/逐caseMSE/17变量RMSE。整调用成本含load/IO/metrics，
不是独立forward latency或单纯模型工作显存；父进程计费含import/启动/间隔/清理。

| 身份 | SHA256 / 值 |
| --- | --- |
| 新protocol canonical digest | `1ce9222321bfe6d799b0f86d7bc0ff4de127d451edaa0e5e8a45ca5a4a3ffc22` |
| 新protocol文件bytes | `12ece0ba8fcaae4b5fc560dc6d133a5a8f88ddbf3fba063152c6272c8252e716` |
| measurement_code.zip | `b7f20e7b8adda117c394deb0751fcd1683b536404d5cfa18e121d61c39b742c0` |
| attempt.json | `662dd5ab2862bd4b0a1cfc585366bfe4b29a0e57edffd6f7f03715f405a35479` |
| 原protocol canonical digest | `e19ef488be60136364702b1df389e5f58be7ebf3845487289139e10d30231e01` |
| 原code.zip | `5fd26146af2a7d11016fb769d67390f5daa23a73620de9ae83e2e6cc38a35a0a` |
| 原model code | `11090929930da4e1259698699cbbf12b3738cdfb2f609c3af516c24399144476` |
| data identity | `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07` |
| source | `496084a9260bacfaf6293a01d89439c1e49d6afa8f09bc1f51d89a1d1f9bda21` |

## 3. 已确认的结果与失败

首项`seed41/process_spacetime_rwa/lead_006h/`完整留下provenance、RMSE、ACC、skill与
`cost_measurement.json`。22val case、17变量RMSE和逐caseMSE与原归档精确一致；起点allocated/reserved均0。
该项whole-call allocated peak **39,590,400 B**、reserved peak **46,137,344 B**，elapsed **11.9927125191316s**。
只覆盖一个cell，不能代替其余29项，也不能冒充原运行当时的独立峰值。

计量行SHA256 `b654d17d812d3353e887b392505248f9bbdf5d8796c397b2dccb51ab85dcf270`；
checkpoint SHA256 `ed6a83f12fa9197f3f202f66bafd627f12ea11666ddfa315fa27782d99a08d75`。
该行内包含provenance/RMSE/ACC/skill的独立文件摘要，登记时标准库只读重验全部匹配。

准备第二个lead12h时，`reset_measurement`在gc/同步/empty_cache之后检测allocated或reserved
至少一个非零，抛出`RuntimeError: independent evaluation requires zero allocated/reserved baseline`。
**12h evaluator尚未调用**；worker退出1，父进程将attempt记failed并停止。
日志`worker_seed41_process_spacetime_rwa.log`与`/tmp/r7_n1_eval_cost_execution.log`保留。
没有worker success汇总、`result.json`、四成本CSV或`cost_views.json`；未生成或伪造终态齐套结果。

**未能确认**：失败基线的具体allocated/reserved字节数没有被原guard记录；何种对象/allocator状态导致
非零也未取证。不能把内存泄漏、hook循环、外部竞争或硬件故障说成已确认原因；没有为诊断再跑CUDA。
第一项实测通过不证明同worker连续五lead能独立清零，原CPU fakes未覆盖这条真实运行路径。

## 4. 实耗与账本

attempt原记录：GPU保守计费区间 **21.443860329687595s = 0.005956627869357666 GPU-h**；
整轮 **22.372659532353282s**，均未超cap。failed照记全额，不能因只完成一项而不计费。

- 原N1 `0.3590401737619605` +失败补测 `0.005956627869357666` = **0.3649968016313182 GPU-h**，
  原0.45cap剩 **0.08500319836868184 GPU-h**；这是算术余量，不是自动重试许可。
- campaign精确累计 **4.076196801631319**，精确余 **19.923803198368685 GPU-h**；
  ledger失败行四位小数`0.0060`，显示累计 **4.0762** /余 **19.9238**，舍入不扩大授权。
- 原证据页`docs/R7_N1_PIVOT_AUDIT.md`与原协议/归档/科学读法不改；原48/72h仍unresolved，
  `cannot-distinguish`、N2d仅提议、current_node=N1/paused。

## 5. 验收、影响与下一项

独立只读复核及最终工程登记验证另记轮次进度；本页只记录固定的一次失败尝试。
**D2独立evaluation内存仍缺29/30项，四张终态成本表未产出；完整成本验收仍阻塞。**
没有目标完成声明，没有自动重试或推进节点。两seed不作显著性，补测不产生新的机制结论。

接口、依赖、模型与原数据未改；仅新增授权/失败产物和文档登记。安全配置/凭据未改；
先前Mimosa提交/push扫描`scanner_enobufs`无结论，不是安全认证。occupancy观测不能排除瞬时竞争，
整调用peak不能与单独forward峰值混称；RMSE精确重放仅覆盖已完成一项，不证明全30项可重放。

下一项是失败的工程原因审阅与单独具名的修复/重试决策；当前一次执行授权已用，不再运行。
任何重试均先重读账本、冻结新协议与新输出、重新取得预算授权，不覆盖本次failed目录或改变零基线要求。
