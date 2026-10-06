# S3 v3 rollout 微调 attempt01：硬截止超射、两 seed 完整、第三 seed 未完成

**终态：failed / partial；没有三 seed 判定，不是 registered-negative 科学读数，
不进入 S4。`scientific_claim: false`；test（2023）未评分。**
运行根为 `outputs/r7_s3_v3_rollout_ft_20261006_attempt01/`。
原 protocol、partial result、failure、checkpoint 和驱动均保留原字节；不原地续跑或补写成功。

## 1. 冻结身份与实际失败

| 项 | 实际记录 |
| --- | --- |
| 运行代码 | `dd2139e8acaee758bc928709f5e5db0f8ec3db00`；dirty 仅用户 `.zcode/config.json` 与 `.zcodeignore`，不修改、不提交 |
| 主 CI | [37437569014](https://github.com/Eswink/UrbanPiDiT_R2/actions/runs/37437569014)，上述精确 SHA `completed/success`；不证明本 GPU attempt 成功 |
| 内部 protocol digest | `8e8fc195b5a3e58420a6d4101d5f00a035c3a4eee753b043fcebb90273d0381d` |
| protocol 文件 SHA256 | `af5dc6045885af1f1eb3dcc64349e69154f8b061a7dc3aa24ff0d33c307bb0fa` |
| partial result 文件 SHA256 | `387e1369672b7dcef87b8c404ef5bfdab68a24c055dbc9a39a572315cc0fb4dc` |
| failure 文件 SHA256 | `3c8725c749ba6a271b143c9c1399a69f9e5039fd7a0d56e7253f10141b7d985e` |
| 配方 | v3-BD 1600 更新 endpoint 为 parent；每 seed 新 optimizer，two_step × 200，LR 2e-5，warmup 10，lambda12=0.5，K4 |
| 数据 | train 2017–2021 / val 2022；train identity `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac`；source `bc2ff9cfadcce604fc243bb999b3c430d5164d17fcf1de201273716a5db065f8` |
| 冻结时长 | planned 4200 s / hard 7200 s / per-seed 3600 s；GPU0 共驻 |
| 实际终态 | `RuntimeError: scheduled runner deadline exceeded`；`seeds_completed=[41,42]`；没有 `attempt.json` 或 `decision` |

原始 `failure.json.status` 为 **failed**，没有把它回写成新的状态名。
初次文献转述及其范围更正见 `docs/R7_S3_ROLLOUT_TRAINING_SOURCES.md`，仅用于方法假说。

## 2. 已完成与未完成

- seed41、42 各完成 200 更新及 6/12/24/48/72h 全 cohort val 评估。
  训练耗时分别 99.8236 / 96.8782 s；五 lead 评估合计分别 809.8189 / 875.2225 s。
- seed43 没有 training_report 或评估记录；保留 update20/40/60 checkpoint，不能把
  最新落盘 update60 当完整 200 更新。最后一次未落盘更新数不能从目录猜测。
- failure traceback 来自 `training/r7_autoregressive_runner.py` 的 `_update`：
  forward/backward/clip 返回后才调用 deadline 检查。这是**合作式**截止，不是外部墙钟看门狗。
- 文件时间的观测：seed43 update60 落盘于本机显示时间 17:35:18，failed_attempt 于 19:17:01。
  这说明长间隔存在，不证明是哪一个 CUDA 内核、I/O 或邻居进程造成了阻塞。
- 两完成 seed 只可作 diagnostic-only 的部分记录；不求两 seed 均值代替三个预声明 seed，
  不与未来 attempt 的 seed 拼接成“完整运行”。

## 3. 实际成本全额登记

| 项 | 实际值 |
| --- | --- |
| 整轮 failure.elapsed_seconds | **9473.920853041112 s** |
| soft overrun | **5273.920853041112 s**（9473.920853 − 4200） |
| hard overshoot | **2273.920853041112 s**（9473.920853 − 7200） |
| 保守 GPU-h | **2.6316**（整轮墙钟 / 3600，含准备、失败与所有间隔） |
| 本方向累计 | **12.5554**（前项 9.9238 + 本失败项 2.6316） |
| 收费 / 新数据下载 / test 评分 | 0；无本轮下载或付费；test 没有评分 |

这是执行硬截止没有及时生效的失败，不把超射包装为“软预算允许”。实际成本不能截断到
7200 秒，也不能仅计前两个成功 seed。决策0030/0038 无总 GPU-h 许可上限；12.0 的旧
会计字段被超出是账本事实，不是免除成本或终止全部独立主线的依据。原协议数值仍不可修改。

## 4. 排障范围与接续

已确认的工程缺陷是：截止检查不能打断尚未返回的训练操作。具体阻塞根因 **未确认**，
没有事后 profiler，因此不归因给 GPU 共驻或声称换卡已修好根因。

下一次只允许一次新路径重执行同一科学配方，且需先通过 CPU 故障注入：
父进程看护自己创建的直接 Popen；超过冻结 deadline 仅向该 PID TERM/KILL 并 reap；
CPU preparation 与 GPU 每 seed 均记入整轮墙钟；失败立即停整轮、不自动重试。
GPU1 的 UUID、独立 soft5400/hard10800/seed3600 将在新 protocol 冻结，旧 attempt 不动。

## 5. 独立只读审阅（不提升失败终态）

独立审阅输出工作 JSON `/tmp/r7_rft_audit_a9iayid4.json`，SHA256
`2bd54b27bc8add55264ad5d4888de11b9ce171f2591e8cea93f461cf95b39263`：
三 parent endpoint、23 个 child checkpoint、两完整 endpoint/report、三个协议 digest、
source/build receipt/manifest/data/model/training/初始化 state chain 均匹配；
40 组评估 CSV 的 pooled RMSE 与 provenance 独立核算最大相对误差 1.134e-15。

两完成 seed（41/42）的**诊断**读数：相对 D3 的 t2m 6h 增量 −0.442619/−0.788822 K、
12h −0.561129/−0.771484 K；相对 parent 的 6h −0.009778/−0.028078 K、
12h −0.026029/−0.051550 K。已完成两 seed 相对 D3 的守门仍有 **8 个正 cell**（3+5），
其 parent 原来对应两 seed 为12个；相对 parent 有3个正 cell。
这些完整 endpoint 已显出守门反例，但没有第三 seed，因此 **不能形成预注册三 seed verdict**。
不把 partial 记录改成成功，不对后续重执行挑选这两个 seed 的最好 checkpoint。

## 6. 可复现等级与限制

- 本页冻结的是失败终态与身份，不认证数值、收敛或天气技巧。没有完整三 seed gate。
- GPU 训练非逐位确定；本配方最高为 config-reproducible，不声称子权重 bit-reproducible。
- 源、parent、model/data/contract 身份仍须严格验证；新执行不能通过换 digest 比较逻辑绕过。
- 只有一个 ROI、四季各30日块，而非全年覆盖；2023 尚未评分也不等于完整未见全年确认就绪。
- 失败证据和部分产物保留，不合成替代、不删旧文件、不恢复该 attempt。
