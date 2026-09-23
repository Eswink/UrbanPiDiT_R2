# Real Data Audit Round 3 — End-to-End Model & Runtime

## Score: 96/100 — PASS

### Executed paths

1. Real fixture -> preprocessing -> NPZ -> JSONL manifest.
2. Manifest sample -> `validate_sample` single/batched.
3. V6 forward with `force_zoom=True`.
4. `R2Loss` -> backward -> finite nonzero gradient.
5. Hard STOP path: threshold forces no Urban Expert; residual is exactly zero and forecast equals baseline.
6. Hard ZOOM path: Urban Expert is executed and forecast remains finite.
7. Adaptive routing path: finite output and bounded reasoning steps.
8. Lightning `fit` on two real-fixture train batches.
9. Validation -> best checkpoint.
10. Restore best checkpoint -> test batch.

### Runtime evidence

- Smoke tensor contract:
  - coarse `[1,4,4,8,8]`
  - urban history `[1,4,1,32,32]`
  - static `[1,4,32,32]`
  - forecast `[1,1,32,32]`
- Tiny real-data smoke model: ~1.1M trainable parameters.
- Final Lightning test loss is finite.
- Checkpoint restore succeeded.
- Portable smoke checkpoint SHA256: `77cd1a34466534dda896484301406452703a74498ba16cf11859b2926cca4d90`.

### Remaining boundary

No NVIDIA GPU exists in the current sandbox. This round certifies CPU E2E correctness, not 4090D CUDA memory/performance. GPU certification remains delegated to `scripts/benchmark_gpu.py` on the user's machine/rental node.
