# 计算资源预算

## 目标

V6 核心配置必须以 RTX 4090D 24GB 为第一约束。RTX 5090 32GB 用于最终训练/更大 batch；V100 32GB 只在实测成本更优时采用。

## 当前 V6 4090D 配置

- 约 13.4M 参数；
- Urban tile: 128×128；
- patch=4 -> 32×32=1024 urban tokens；
- window=8 -> 64 tokens/window；
- 24 process states；
- 8 urban blocks；
- activation checkpointing=true；
- physical batch=1，gradient accumulation=16；
- BF16 mixed precision。

13.4M 是有意的保守 MVP，而不是目标模型上限。首先验证 reasoning 与 adaptive compute；后续只有在消融证明容量不足时再增加宽度/深度。

## 租卡决策

不要按峰值 TFLOPS 决策。使用：

```bash
python scripts/benchmark_gpu.py --config configs/r2_v6_5090.yaml --gpu 5090
python scripts/benchmark_gpu.py --config configs/r2_v6_v100.yaml --gpu v100
```

比较 `¥ / 1000 steps`。5090 会员价按用户给定价格约 2.641 元/小时，V100 约 1.786 元/小时。
