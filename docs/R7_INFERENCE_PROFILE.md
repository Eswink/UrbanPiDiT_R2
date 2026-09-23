# Isolated checkpoint inference profiling (#38)

```bash
python profile_r7_inference.py --checkpoint outputs/process/update_0000010.pt --controller outputs/controller/controller.pt --manifest data/manifests/r7/test.jsonl --out outputs/fixed_profile.json --device cuda --precision bf16 --force-full-depth --warmup 3 --repetitions 20
python profile_r7_inference.py --checkpoint outputs/process/update_0000010.pt --controller outputs/controller/controller.pt --policy-selection outputs/policy/selection.json --manifest data/manifests/r7/test.jsonl --out outputs/adaptive_profile.json --device cuda --precision bf16 --warmup 3 --repetitions 20
```

Run each variant in a fresh process on the same hardware, resident input batch,
precision and explicit repeat budget. The first manifest batch is selected and
its tensor hash recorded. This is one-batch latency, not distribution-wide weather
skill or a complete deployment benchmark. Repeat with representative weather
cases and record external GPU load/clock/power settings before paper claims.

Warm-up forwards are excluded. Every timed CUDA forward has synchronization
before and after; CPU uses perf_counter. Timed regions include model forward,
Python routing/control and synchronization, but exclude data IO, host/device
transfer, output validation and metric computation. Actual per-sample reasoning
counts accompany all timings; no speedup is inferred just from fewer K steps.

CUDA peak counters are reset for each measured forward and read before output
validation. Peak allocated/reserved memory includes model/input and the warmed
allocator state; baseline allocated/reserved memory is also reported. CPU memory
fields are null, never fabricated GPU measurements. Requested unavailable CUDA
or unsupported BF16 fails rather than silently switching device/precision.

Model weights/buffers and resident inputs are checked unchanged outside timed
regions. Nonfinite predictions fail. Reports bind parent/controller/selection,
manifest and training identities, hardware/software, dtype and TF32 flags.
Output is exclusive and no checkpoints are modified. Native and fixed recurrent
models work without a controller; frozen selected policies cannot be overridden.

CPU/CLI fixture tests certify timing boundaries and program behavior only.
Actual 4090D inference and training measurements remain under #7/#20. This entry
point performs no rental, download, deployment or training automatically.
