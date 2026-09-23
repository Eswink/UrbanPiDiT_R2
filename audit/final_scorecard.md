# Final Scorecard

**最终评分：94/100 — ACCEPT**

通过条件：≥90。Round 1、Round 2 均因实际缺陷被打回并修改；Round 3 通过。

## 必须在真实 4090D 上执行的验收命令

```bash
python scripts/benchmark_gpu.py --config configs/r2_v6_4090d.yaml --gpu local --warmup 20 --steps 100
```

若 `peak_allocated > 21 GB`，优先调整顺序：

1. 保持 `batch_size=1`；
2. 降低 `urban_dim 256→224/192`；
3. `urban_depth 8→6`；
4. 缩小 tile 或保持 128² 但增加 patch；
5. 不应重新引入 dense attention/graph。

## 租卡成本验收

```bash
python scripts/benchmark_gpu.py --config configs/r2_v6_5090.yaml --gpu 5090
python scripts/benchmark_gpu.py --config configs/r2_v6_v100.yaml --gpu v100
```

按实际 `¥/1000 steps` 而非峰值 TFLOPS 选卡。
