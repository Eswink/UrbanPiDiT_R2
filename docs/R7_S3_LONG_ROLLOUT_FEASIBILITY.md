# S3 长物理监督：FP32 12 步单样本可行性与成本

**已确认：原探针完成，损失/梯度有限且可执行；这不是天气技巧、三 seed 筛选或科学通过。**
`scientific_claim: false`。归档代码临时重放已完成，单样本测量字段精确相同；不把重放当科学确认。
旧短双步 registered-negative 与 failed attempt 均保持，2023 test 未评分，S4 未开始。

## 1. 原运行的冻结身份

运行根：`outputs/r7_s3_long_rollout_probe_20261006_attempt01/`。

| 项 | 固定身份或实测值 |
| --- | --- |
| 执行代码 | `66836d29dd257a11b0c946c8af2622b35042a5ce`；工作分支 `r7/weather-reasoning` |
| 精确 CPU CI | [37500335847](https://github.com/Eswink/UrbanPiDiT_R2/actions/runs/37500335847)，上述 SHA `completed/success`，必要检查与测试步骤均 success |
| code.zip SHA256 | `3d8a494587f41e0b33e6f5e1cbab9de8b23ad552433282e4dd03022cf28eabb8`；ZIP comment 与 code_commit.txt 一致 |
| canonical protocol digest | `78b96197cf0021f36ea326e351b3ac5e7762d753b95da00595751f53d6f27ee5` |
| protocol 文件 SHA256 | `eff10118380f7df83c9eef8b7d9e9369a2c452ad80408eb8d5fa7a47cae9b127` |
| result 文件 SHA256 | `f577fee90462196a134a17872209f672c1a3561393632e59a8a1029e605caaba` |
| attempt 文件 SHA256 | `0d854eb7af190ec685a1b92ef6506c540a0039b00bf30e9703ec9fc5cad53368` |
| feasibility_receipt SHA256 | `3fec7af62af699934567605be426acf7d91a2ffc317e13d9d30f9791ecd6f80b` |
| 完整源 SHA256 / 字节 | `bc2ff9cfadcce604fc243bb999b3c430d5164d17fcf1de201273716a5db065f8` / 540,856,239 |
| train identity | `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac` |
| val identity | `0c34a887216b02b41716e7837f5be4d515ad2027e7b4a4728cbef15accd02d5a` |
| model code digest | `3ddab46b1e4c2c7e66449e39ab8247c9c7e642f45023c1c9cc14bca8f74fd217`（未改 model 源码） |
| long training code digest | `24dde71ba380312f2c201c8f5fd2cabb9c884585bb0ce904fdee768fc5f675b9` |
| 12 步窗口 digest | `dccfeef926286ddeccb7690aacd7bac45c1bb2036fe6ebbe826841b971bb39ec`；2360 原始、2140 完整、220 排除 |
| GPU1 | `GPU-9d1624af-9d77-aa7c-0620-b6cb778f4ced`，默认共驻 |
| 冻结时长 | 整轮 planned 1800 / hard 3600 秒，包括准备、归档、加载、GPU 区间和清理 |

用户 `.zcode/config.json` 修改与 `.zcodeignore` 未跟踪保持，不修改/提交。运行中只更新 goal 的工作态进度，
不改变执行源码。没有新数据下载、store 重建、模型源码或依赖变更。17 条实验 CI 标签未触发的 skipped
不计实验通过。CPU CI 不代表 GPU 科学接受。

## 2. 单样本机制与实际读数

- probe kind 的实际作用域：seed41 注册 v3-BD1600 父 checkpoint 严格导入 model state；仅 train 首个完整
  窗口 `era5z_train_2017010106_p006h`，FP32、K4、12 个物理步，所有生成历史保留全 BPTT 图。
- loss 固定 `L6 + .5(L12 + L24 + L48 + L72)`；其他物理步权重 0，但生成历史不 detach，仍参与后续梯度。
- **没有 optimizer 更新、checkpoint 训练发布、validation 或 test 评分。** protocol 内含未来 screen 的
  三 seed/200 更新/评分判读文本，不能据此声称 probe 实际执行了它们；kind、dispatch 和 receipt 限定上述 scope。
- 原单步/双步 recipe、runner、model 与已登记父 endpoint 均不改。

| 测量 | 实际值 |
| --- | ---: |
| 加权 loss | 1.2537332773208618 |
| clip 前 gradient norm | 40.788814544677734 |
| grad-enabled FP32 forward + backward FLOPs | 393,859,201,536 |
| 自有 CUDA allocated peak | 2,274,339,328 bytes |
| 自有 CUDA reserved peak | 2,409,627,648 bytes = 2.244140625 GiB |
| 计量区间 | 7.454262489452958 s |

12 个 step loss 顺序为
`0.0993191674 / 0.1749944687 / 0.2615442276 / 0.3403165936 / 0.4290478826 / 0.4673823118 /
0.4931091070 / 0.5438693762 / 0.6352182627 / 0.7372297049 / 0.9456760883 / 1.2496478558`。
它们是**同一个 train 样本的归一化训练损失**，不是物理 RMSE、训练收敛或 validation 预报技巧。
早步梯度连通和目标隔离由 CPU 阳性/断图反证覆盖；GPU 探针测量的是完整 loss 的一次 backward。

FLOPs 为 PyTorch 支持的 aten 操作计量，elementwise/normalization 等未计，不是完整硬件 FLOP。
按此单样本形状计量估算新增 200 更新为 **78,771,840,307,200 FLOPs**；部署配方还应加 parent1600 L6。
以既有 CPU L6 31,981,732,992 supported-aten FLOPs 作分母，未来部署计算比的**描述性估算**为
**10.157565033053729**；本次 numerator 为 GPU 实测，未另跑同设备 L6，后端支持计数可能不同，
不冒称精确硬件等算力比。不能把 200 更新叫等算力；这里只估算未来 recipe，并未执行 200 次更新。

三个 spawn 都实核 free >= 4 GiB（冻结估计 2 GiB + margin 2 GiB）。探针出口给出更大的 known peak；
**以后每 spawn 的准确门槛至少 4,557,111,296 bytes = 4346 MiB**，取 max(估计, known peak) 再加 margin，
不能沿用原 4 GiB 或用整卡容量代替 free。没有对邻居发送信号。

## 3. 全额成本与时长

| 项 | 秒数 |
| --- | ---: |
| prepare owned worker | 394.4425724456087 |
| archive owned worker（含身份重核，不只是复制） | 401.6335758501664 |
| probe owned worker（含身份重核/目标加载） | 808.7099605770782 |
| worker 合计 | 1604.7861088728532 |
| root 开销/间隔 | 0.22692217584699392 |
| **整轮实际** | **1605.0130310487002** |
| soft / hard overrun | 0 / 0 |

全部三个 worker exit0、reaped、signals 空；外部墙钟截止只针对直接持有 Popen 句柄的 worker。
整轮保守 GPU-h **0.4458369530690834**（账本 **0.4458**），不是 7.454 秒 GPU 区间，不是 utilization-hours。
相加前一累计 14.1154 得 **14.5612**；重放费用待实际出口另计，不能提前记 0。

## 4. 独立审阅与工程验证

独立只读审阅工作 JSON：`/tmp/r7_probe_independent_audit_20261006_ut8duqym/audit.json`，SHA256
`7ebabda16f6bbf238bc059ead624fdf4b8b7eb44504238cc7b5bfd2c1aa9af85`。25 个归组检查实核：

- 176/176 live 执行库存匹配，175/175 Python 源与 archive 匹配；源全文 hash、train/val 数据身份、
  BUILD_COMPLETE/归一化及 12 步窗口独立重算；parent41 普通严格 CPU checkpoint/state 导入成功。
- canonical protocol、attempt/result/receipt 一致；三个独立 process 文件与 result 内副本一致，
  时间区间不重叠，实际时长/软硬 overrun/首尾开销核对一致。
- loss/gradient/FLOPs/owned peak 正且有限；按保存 step loss 重算 weighted loss 差
  `3.725290298461914e-08`，为 FP32 加法口径，**不是科学门容忍放宽**。
- probe dispatch 无 optimizer/evaluation；审阅没有运行 GPU/forward/replay，没有读源天气字段或 test。

审阅 wall **868.864359s**，自设 soft600 超 **268.864359s**、hard1200 未触；CPU 时间未完整汇总，
如实登记，不伪装成零成本或主探针计时。审阅脚本曾错误要求 probe.log 空而中止 JSON 生成，
检查发现仅 scalar-conversion UserWarning；修正这个无关审阅假设后完成，不改 probe 或冻结门。
`float(result.loss)` 警告是日志事实，不是 loss/gradient 失败；本轮不运行中改源码。

实际工程验证为 130 个定向测试通过（172.22s），全量 **3887 passed / 3 skipped / 6 warnings**
（1169.72s），37 条阻断约定、index/brief、campaign 与 goal 结构通过；3 个 optional 真实 fixture
跳过不作天气验证通过。本次 docs 交接的 campaign/index 工具回归另实跑 **74 passed / 0.40s**。
独立审阅只验证身份/作用域/成本，不是科学 verifier。

## 5. 归档代码临时重放（完成，单样本精确再现）

单独新目录 `outputs/r7_s3_long_rollout_probe_replay_20261006_attempt01/`，soft1800/hard3600，
每 spawn free 至少4346MiB。冻结 wrapper SHA256
`7e71f0409e015963c8a48e532b28fb796c8c9511d761f3fb310bcb4fe012c2c0`。
先核本页具名 ZIP SHA，安全解包并从原 code.zip 导入；两 worker 都重新核 source/train/val norm 身份，
读取同一 train 样本/parent，不做 optimizer 更新或 val/test 评分。比较 loss、全部 12 step loss、
gradient norm、supported FLOPs 和 sample ID；差异须归因，不能弱化比较或覆盖原输出。

独立静态审阅初稿发现 measurement worker 没有再次绑定 data/norm/source 身份：在首次运行前修复，
setup 绑定 protocol、parent helper 源码 hash、失败成本覆盖和 memory/timing 有限性亦补齐。
初稿审阅 JSON `/tmp/r7_probe_replay_static_review_20261006_kvthrtex/review.json`，SHA256
`d8432843b7a5dc7244196cc620add3b911d800b1a67937820bd9e445f4ca2c11`；278.046383s，soft300/hard600 内。
修复后独立定向复查99.481s无剩余 essential blocker（非 runtime/科学接受），12个CPU synthetic身份/路径/
一ULP摘要反证实跑通过，未访问天气/GPU。

实际重放完成：sample ID、loss、全部12 step loss、gradient norm 和 forward/backward FLOPs **五字段精确相同**。
自有 reserved peak 仍2,409,627,648bytes；GPU计量区间7.025372850s，只是局部，不代整轮费用。
两个worker均exit0/reaped/no signals；prepare419.121014831s、reproduce426.856469466s，worker合计
845.977484297s、root0.082678801s，整轮 **846.0601630983874s**，planned1800/hard3600，overrun0。
保守 GPU-h **0.2350167119717743**（账本 **0.2350**）；没有失败重放或评分/optimizer更新。

| 重放身份 | digest |
| --- | --- |
| canonical replay protocol | `8f7e2f6ce24dc58b5f06f351bb5d6393c18dd47035910db87ff1c0b8f5a6c3a8` |
| protocol 文件 SHA256 | `81e06ccb03c4dc54dd4110ebd5ef4476ae2b1586626c2537525beaf4bee8565e` |
| RESTORATION_ACCEPTED.json SHA256 | `3b3680057e75cbceda14f359124961faf4b9f83e0b6ee6befed86a85331cd152` |
| replay_result.json SHA256 | `19e90960a3db375bd83e8fc9f1b7174723e2143eb743a911d487afc553b25ff9` |
| setup_receipt.json SHA256 | `e144995c0e2a7fd5e6fab9c09426bf89deabb2ba605bd7ce3dca40487e6642ca` |

原probe与本次replay未舍入合计0.6808536650408577GPU-h，逐行四位登记0.4458+0.2350=**0.6808**，
累计 **14.7962**。两次执行输出各自排他，原probe未补写/覆盖。最高声明仍config-reproducible，
单样本观察到数值精确不证明跨平台训练逐位再现。

独立 terminal 复核28个检查核五字段、protocol/helper/driver/setup/原receipt pins、两gate余量及成本均一致。
工作 JSON `/tmp/r7_probe_replay_terminal_audit_20261006_rlgmnsmd/audit.json`，SHA256
`324e73bcc08293cfcc7aad110a4f8c3203f6d4069878bc89b597aaa85b95af30`，85.756s，soft120/hard300内。
没有重复 source/window/checkpoint/全库存审阅或执行GPU；allocated peak未另在replay发布，不能推填。
scope/network/peak结论仍依实际receipt与审阅源码，而非独立runtime tracing。

## 6. 限制与接续

- 单 train 样本可行性不证明完整 200 更新、三 seed 或长期显存峰值；下一轮仍每 spawn 核余量及已测峰值。
- archive 不含一个历史 actual-C 非 Python protocol 文件；该文件作为**具名独立外部配置**仍由 live pin
  核验。本次不宣称 archive 单独包含全部数据/外部配置。元数据/源身份核验未穷尽重哈希全部 derived chunks。
- 日志/源码/snapshot 不等于全生命周期 I/O tracing，scope 结论依据冻结 dispatch、metadata 及 receipts；
  未声称建立系统级网络沙箱或邻居信号拦截。
- 最高 config-reproducible；未做训练再现，不声明 bit-reproducible、收敛、因果或天气技巧。
- 未进入 S4，未评分 test，四季 30 日块不是完整未见全年，正式科学协议仍全部保留。
- 普通方法探索自主推进；完成重放与证据登记且 compatible feasibility/headroom 通过后，仅另冻一次
  三 seed long screen，按原主格/守门合取判读，不重复短双步或混入原失败 seed，不自行裁定 goal complete。
