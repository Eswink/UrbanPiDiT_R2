# Bounded local R7 training and profiling

Engineering #25, parent #20. No network downloads, scheduled execution, implicit
GPU rental or silent CUDA-to-CPU fallback. Existing Lightning paths are unchanged.

Run the explicit synthetic engineering smoke:

```bash
python train_r7_local.py --config configs/r7_local_smoke.yaml --synthetic --out outputs/local_smoke --updates 2
python train_r7_local.py --config configs/r7_local_smoke.yaml --synthetic --out outputs/local_smoke --updates 4 --resume outputs/local_smoke/update_0000002.pt
```

A real prepared version-1 dataset uses `--manifest path/to/train.jsonl` instead
of `--synthetic`. Supply a config with exactly matching input/output channels,
process count, history length and normalization conventions. When process_weight
is positive, missing process labels fail instead of silently disabling supervision.
No full real dataset or real-data model checkpoint is bundled.

`--updates` is an absolute optimizer-update endpoint, not epochs. Each accumulation
group has one update; uneven groups are weighted by sample count. CPU batches
are moved one at a time, avoiding resident GPU copies of an entire accumulation
group. Recursive models use #21's tested truncated streamed backward; native uses
ordinary backward. BF16 is opt-in; scaled FP16 and multi-GPU are not implemented.

Checkpoints use state_dicts plus safe primitive/RNG data and weights_only=True
loading. Config/data/optimizer/precision/torch-version identity must match before
resume. Dataset identity binds manifests, channel/coordinate/time metadata and
normalization arrays, NOT every raw meteorological chunk; source checksums and
immutable-cache discipline remain required. Resume exactness is tested on the
same CPU/software stack with dropout; not promised across hardware/releases.

Checkpoint publication does not replace existing paths. An interrupted run may
leave its own logs/partial output; it does not remove user datasets. A metrics-file
failure after checkpoint publication leaves the checkpoint available for recovery.

Optional shape benchmark on the user's own GPU:

```bash
python scripts/profile_r7_local.py --config configs/r7_local_smoke.yaml --out outputs/profile --device cuda --bf16 --steps 1 2 4 8 --updates 3
```

Use a separate config with realistic H/W, channels and model dimensions for #20
acceptance. Each K runs in a fresh process. Reports include synchronized wall time
(including reads), allocated/reserved CUDA peaks and hardware/software identity.
These are training shape measurements on synthetic inputs, not forecast latency,
real-data skill or a claim that a 24GB production run already passed. Failed runs
are recorded with nonzero return codes, never converted into passing results.
