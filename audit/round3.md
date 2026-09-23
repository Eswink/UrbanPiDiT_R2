# 自审 Round 3 — 最终闭环审查

**门槛：90/100**  
**评分：94/100 — PASS**

| 项目 | 分值 | 得分 | 证据 |
|---|---:|---:|---|
| 技术路线一致性 | 30 | 29 | Process State → recurrent Reasoner → per-sample Router → selected Urban Expert → supervised Verifier 完整闭环 |
| 可运行性与测试 | 30 | 29 | 10/10 tests；compileall；smoke forward；Lightning fit→ckpt→test；正式尺寸 backward 均通过 |
| 4090D 资源约束 | 20 | 18 | 13.39M；128²→1024 tokens；8² window；sparse local graph；checkpointing；B=1+accumulation=16 |
| 数据/工程可维护性 | 20 | 18 | 顶层 data/model；manifest contract；preprocess 目录；旧版完整隔离；4090/5090/V100 config |

## 实测结果

- V6 4090D config 参数量：**13.39M**；
- BF16 参数本体约 **25.5 MiB**；AdamW 两个 FP32 state 约 **102.1 MiB**；
- 正式尺寸 CPU 前向：`[1,7,128,128]` 正常；
- 正式尺寸 CPU 前向+反向：成功，Urban stem 获得梯度；
- 当前沙箱该检查峰值 RSS 约 **770 MB**（CPU RSS，不能等同 GPU VRAM）；
- 主路径扫描：未发现 `torch.cdist` 或 `nn.MultiheadAttention`；
- Lightning smoke：`fit → best checkpoint restore → test` 成功；
- 10 个 V6 单元/集成测试全部通过。

## 未满分原因 / 边界

1. 当前执行环境无 CUDA，**尚未实机测得 RTX 4090D/5090 的 peak VRAM 与 step/s**；必须在真实 GPU 上运行 `scripts/benchmark_gpu.py`。
2. 当前真实 SMBFD/HRCLDAS 数据尚未落盘，因此正式天气精度、Router utility 与 process semantics 尚未验证；合成数据只证明工程链路。
3. Residual Diffusion 已提供可运行 scaffold，但按技术路线默认关闭，避免在 reasoning 主线验证前增加算力与变量。

结论：V6 工程达到“可进入真实数据接入与单卡实验”标准，不代表已达到科研性能结论。
