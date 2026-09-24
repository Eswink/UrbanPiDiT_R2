# R7 dual-3090 GPU bring-up

Engineering acceptance for the CPU → GPU transition. This document records
measured GPU memory/latency baselines, the single-GPU checkpoint contract, the
first dual-GPU DDP smoke in this repository, and the OOM boundary.

**This is an engineering record, not a scientific result.** Every artifact here
carries `scientific_claim: false`. No forecast skill, convergence or SOTA claim
is made or implied. `spatial_solver_feedback` stays opt-in and default `False`.

Measured on this machine during the bring-up session; the instruments and this
record are committed at `45d5c93cd8a6360c4a15f4acd4d2c426d7915b32`, whose
push-triggered CI run passed (§9).

---

## 1. Environment (D1, measured)

| Item | Measured value |
| --- | --- |
| GPU ×2 | `NVIDIA GeForce RTX 3090`, 24576 MiB each (PyTorch reports 23.56 GiB) |
| Compute capability | 8.6 (sm_82, 82 SMs) |
| Driver | `580.173.02` |
| CUDA (torch build) | 12.8 |
| PyTorch | `2.11.0+cu128` |
| BF16 | `torch.cuda.is_bf16_supported() == True` |
| Python | 3.12.3 (repo `.venv`) |
| GPU0↔GPU1 topology | **`SYS`** — PCIe plus SMP interconnect between NUMA nodes |
| NUMA | GPU0 → node 0 (`0-13,28-41`), GPU1 → node 1 (`14-27,42-55`) |
| NVLink | **all links inActive** (`nvidia-smi nvlink -s`) |

Verification commands:

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap,pci.bus_id --format=csv
nvidia-smi nvlink -s
nvidia-smi topo -m
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_bf16_supported())"
```

**Two constraints that follow, and both held in practice:**

1. **The two cards' memory is not additive.** No single model can use 48 GiB.
   Every memory number below is per-card. DDP holds **two full replicas**, so it
   reduces per-step time, never single-model memory.
2. **No NVLink.** Collective traffic crosses NUMA/PCIe. DDP is the right first
   choice; FSDP/tensor-parallel are not justified yet.

## 2. Method

Instrument: `scripts/bench_r7_gpu_memory.py` (new). Per the R7 rules it freezes
its protocol (`protocol.json`) before the first measured step and reports its
digest; the digest is recorded with each table below.

- `torch.cuda.synchronize()` before **and** after every measured phase;
- `torch.cuda.reset_peak_memory_stats()` immediately before the measured window
  (after warmup), then `max_memory_allocated` / `max_memory_reserved`;
- `nvidia-smi` was used **only** for identity/topology, never as a memory source;
- full-BPTT loss is the project's own `deep_supervised_forecast_mse` /
  `process_forecast_coreasoning_loss` — no new objective was invented;
- streamed rows call the existing `backward_streamed_truncated`;
- `dtype` is BF16 autocast with FP32 master weights.

**Data limitation (B2).** `data/raw|interim|processed` contain only `.gitkeep`.
The 11-channel continuous-250d ERA5 cache lives in a GitHub Actions artifact
(pinned by `data/download/continuous_pilot_replay.py`) and is **not on this
disk**. Sections 2–4 therefore use the **synthetic shape fixture** at the
11-channel / 12×12 data contract. **These are not multivariate real-ERA5
baselines and not higher-resolution truth.** Section 5 uses the one real dataset
present on this machine.

## 3. Single-GPU baseline table (D2)

Both recursive models, K ∈ {1,2,4,8} × {full BPTT, streamed truncated} ×
activation checkpointing {off, on} = 32 cells per scale. Batch 4, 11 channels,
12×12, FP32 master weights with BF16 autocast. **Every cell below was measured in
a fresh process** (2 warmup + 3 measured steps at scale A, 1 + 2 at scale B),
because `max_memory_reserved` otherwise carries the allocator high-water mark from
earlier cells — see the artefact section under Scale B.

### Scale A — `dim=128, depth=4`, 1,247,894 params (generic) / 1,246,727 (process)

Source: `outputs/gpu_sweep_isolated_a/cells/<cell>/sweep.json`, 32/32 cells `ok`.

| model | K | mode | ckpt | allocated MiB | reserved MiB | step s |
| --- | --- | --- | --- | --- | --- | --- |
| generic | 1 | full BPTT | off | 41.14 | 46.00 | 0.0260 |
| generic | 1 | full BPTT | on | 42.14 | 48.00 | 0.0485 |
| generic | 1 | streamed | off | 41.16 | 46.00 | 0.0397 |
| generic | 1 | streamed | on | 42.16 | 48.00 | 0.0587 |
| generic | 2 | full BPTT | off | 41.14 | 46.00 | 0.0343 |
| generic | 2 | full BPTT | on | 42.14 | 48.00 | 0.0565 |
| generic | 2 | streamed | off | 41.16 | 46.00 | 0.0620 |
| generic | 2 | streamed | on | 42.16 | 48.00 | 0.0737 |
| generic | 4 | full BPTT | off | 43.25 | 48.00 | 0.0445 |
| generic | 4 | full BPTT | on | 42.14 | 48.00 | 0.0801 |
| generic | 4 | streamed | off | 41.16 | 46.00 | 0.0935 |
| generic | 4 | streamed | on | 42.16 | 48.00 | 0.1196 |
| generic | 8 | full BPTT | off | 48.43 | 54.00 | 0.0725 |
| generic | 8 | full BPTT | on | 42.14 | 48.00 | 0.1245 |
| generic | 8 | streamed | off | 41.16 | 46.00 | 0.1974 |
| generic | 8 | streamed | on | 42.16 | 48.00 | 0.1983 |
| process | 1 | full BPTT | off | 41.13 | 46.00 | 0.0268 |
| process | 1 | full BPTT | on | 42.13 | 48.00 | 0.0514 |
| process | 1 | streamed | off | 41.15 | 46.00 | 0.0396 |
| process | 1 | streamed | on | 42.15 | 46.00 | 0.0567 |
| process | 2 | full BPTT | off | 41.13 | 46.00 | 0.0358 |
| process | 2 | full BPTT | on | 42.13 | 48.00 | 0.0540 |
| process | 2 | streamed | off | 41.15 | 46.00 | 0.0517 |
| process | 2 | streamed | on | 42.15 | 46.00 | 0.0731 |
| process | 4 | full BPTT | off | 42.39 | 48.00 | 0.0515 |
| process | 4 | full BPTT | on | 42.13 | 48.00 | 0.1013 |
| process | 4 | streamed | off | 41.15 | 46.00 | 0.0962 |
| process | 4 | streamed | on | 42.15 | 46.00 | 0.1256 |
| process | 8 | full BPTT | off | 46.58 | 52.00 | 0.0759 |
| process | 8 | full BPTT | on | 42.13 | 48.00 | 0.1235 |
| process | 8 | streamed | off | 41.15 | 46.00 | 0.1138 |
| process | 8 | streamed | on | 42.15 | 48.00 | 0.2210 |

The earlier in-process run is retained at `outputs/gpu_sweep_full/sweep.json`
(digest `c10440479a8d8e8b3bc6c508478bd1f5fbd3e3a0ca6e58ed779aa2f5a146ad59`);
there the streamed `reserved` column grew 48 → 54 MiB purely from carryover,
while the isolated values stay at 46 MiB. The affected figure is called out in §10.

### Scale B — `dim=384, depth=8`, ~18.07M params, measured one cell per process

`18,066,838` params (generic) and `18,064,919` (process) — inside the 15–30M
target. Source: `outputs/gpu_sweep_isolated/cells/<cell>/sweep.json`, 32/32
cells `ok`, one process per cell.

**One process per cell is required, not a preference.** `torch.cuda.max_memory_reserved`
is the caching allocator's high-water mark and is *retained across cases inside a
process*, so a sequential in-process sweep inflates `reserved` for later rows.
That artefact is quantified below, and it changed a conclusion.

| model | K | mode | ckpt | allocated MiB | reserved MiB | step s |
| --- | --- | --- | --- | --- | --- | --- |
| generic | 1 | full BPTT | off | 366.31 | 394.00 | 0.0439 |
| generic | 1 | full BPTT | on | 372.75 | 382.00 | 0.0738 |
| generic | 1 | streamed | off | 367.77 | 380.00 | 0.0638 |
| generic | 1 | streamed | on | 371.65 | 384.00 | 0.0869 |
| generic | 2 | full BPTT | off | 367.75 | 402.00 | 0.0479 |
| generic | 2 | full BPTT | on | 372.68 | 384.00 | 0.0754 |
| generic | 2 | streamed | off | 367.77 | 380.00 | 0.0733 |
| generic | 2 | streamed | on | 371.65 | 384.00 | 0.1002 |
| generic | 4 | full BPTT | off | 367.75 | 412.00 | 0.0619 |
| generic | 4 | full BPTT | on | 372.68 | 382.00 | 0.1072 |
| generic | 4 | streamed | off | 367.77 | 380.00 | 0.1258 |
| generic | 4 | streamed | on | 371.65 | 384.00 | 0.1405 |
| generic | 8 | full BPTT | off | 367.75 | 426.00 | 0.0850 |
| generic | 8 | full BPTT | on | 372.68 | 386.00 | 0.1443 |
| generic | 8 | streamed | off | 367.77 | 380.00 | 0.2157 |
| generic | 8 | streamed | on | 371.65 | 384.00 | 0.2365 |
| process | 1 | full BPTT | off | 366.27 | 396.00 | 0.0463 |
| process | 1 | full BPTT | on | 372.71 | 382.00 | 0.0745 |
| process | 1 | streamed | off | 367.74 | 380.00 | 0.0711 |
| process | 1 | streamed | on | 371.61 | 382.00 | 0.0854 |
| process | 2 | full BPTT | off | 367.71 | 404.00 | 0.0498 |
| process | 2 | full BPTT | on | 372.65 | 382.00 | 0.0901 |
| process | 2 | streamed | off | 367.74 | 380.00 | 0.0749 |
| process | 2 | streamed | on | 371.61 | 382.00 | 0.1025 |
| process | 4 | full BPTT | off | 367.71 | 410.00 | 0.0673 |
| process | 4 | full BPTT | on | 372.65 | 382.00 | 0.1028 |
| process | 4 | streamed | off | 367.74 | 380.00 | 0.1094 |
| process | 4 | streamed | on | 371.61 | 382.00 | 0.1273 |
| process | 8 | full BPTT | off | 367.71 | 422.00 | 0.0941 |
| process | 8 | full BPTT | on | 372.65 | 386.00 | 0.1556 |
| process | 8 | streamed | off | 367.74 | 380.00 | 0.2126 |
| process | 8 | streamed | on | 371.61 | 382.00 | 0.2116 |

### Measurement artefact found and corrected (kept as negative result)

An earlier revision of this table came from one process running all 32 cells
(`outputs/gpu_sweep_stageb/sweep.json`, digest
`3664318d35e78ced2ff868e423336859ff28cf8458b26a3cdb29aef7809aac63`). Cell by cell:

| cell | in-process reserved | isolated reserved | inflation |
| --- | --- | --- | --- |
| generic K=1 full BPTT off | 394.00 MiB | 394.00 MiB | 0 |
| generic K=8 full BPTT off | 428.00 MiB | 426.00 MiB | +2 MiB |
| **process K=1 full BPTT off** | **428.00 MiB** | **396.00 MiB** | **+32 MiB** |
| **process K=1 streamed on** | **428.00 MiB** | **382.00 MiB** | **+46 MiB** |

In-process, `process K=1` read 428 MiB — *identical to `process K=8`* — because
the earlier cells' high-water mark was never released. The isolated value
(396 MiB) is the real one. `allocated` is less affected (≤ ~4 MiB) but not
immune. Both sweeps are retained so the discrepancy is auditable.

### What the table shows

- **Streamed truncated training is flat in K for both metrics.** Isolated, every
  streamed row holds `allocated` at ~367.7 MiB and `reserved` at exactly
  **380.00 MiB across K=1…8**, for both models. Full BPTT grows on both metrics:
  `reserved` 394 → 426 MiB (generic), 396 → 422 MiB (process). This is the first
  *measured GPU* confirmation of what `docs/R7_STREAMED_TRAINING.md` previously
  supported only by logical saved-tensor accounting — #21's 701368 bytes at
  K=1/2/4/8 was bookkeeping, not VRAM.
- **Full BPTT's growth appears far more in `reserved` than in `allocated`**
  (+32 MiB vs +1.4 MiB at scale B) because retained activations fragment the
  allocator. A report quoting only `max_memory_allocated` would understate the
  real difference by more than 20×.
- **Activation checkpointing bounds the peak and costs time.** At generic K=8
  full BPTT it cuts `reserved` 426 → 386 MiB while step time rises
  0.0850 → 0.1443 s (+70 %). At scale A the same trade was ~+21 %, so the
  penalty is configuration-dependent, not a constant.
- **No memory crossover at these scales.** 18.07M params needs ~368 MiB of a
  23.56 GiB card. Single-GPU memory is *not* the binding constraint for the
  planned model size — this redirects the next decision (see §7).
- **Step time grows with K in every row.** Nothing here supports the claim that
  fewer reasoning steps is a wall-clock latency win.

Step times differ by up to ~40 % between the two runs for identical cells
(generic K=8 full BPTT off: 0.1203 s in-process vs 0.0850 s isolated) because
only 1–2 steps are measured after warmup. **Treat absolute step times as
indicative and compare within one run**; the K trend holds in both.

### Tensor shapes and batch

Per the 11-channel / 12×12 data contract, each measured step uses:

| Tensor | Shape | Notes |
| --- | --- | --- |
| `coarse_history` | `[4, 2, 11, 12, 12]` | batch 4, 2 history steps, 11 channels |
| `atmos_target` | `[4, 11, 12, 12]` | forecast target, used only in the loss |
| `process_targets` | `[4, 4]` | process rows only; 4 anchored proxies |
| model config | `dim=384, depth=8, patch=2, heads=4, window=8` | `latent_tokens=16` (generic) / `anchored+free = 4+4` (process) |
| dtype | FP32 master weights, BF16 autocast | `torch.cuda.is_bf16_supported() == True` |

### Phase-time breakdown (scale B, seconds)

Every row reports `batch_size`, `grid`, `in_channels`, `dim`, `depth`, `patch_size`,
`dtype`, `params` and both peaks in the per-cell JSON. The full phase split
follows (means over the measured steps after warmup, isolated processes):

| model | K | mode | ckpt | forward | backward | optim step | step total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| generic | 1 | full BPTT | off | 0.0170 | 0.0223 | 0.0040 | 0.0439 |
| generic | 1 | full BPTT | on | 0.0277 | 0.0413 | 0.0042 | 0.0738 |
| generic | 1 | streamed | off | — | — | — | 0.0638 |
| generic | 1 | streamed | on | — | — | — | 0.0869 |
| generic | 2 | full BPTT | off | 0.0200 | 0.0226 | 0.0043 | 0.0479 |
| generic | 4 | full BPTT | off | 0.0261 | 0.0312 | 0.0040 | 0.0619 |
| generic | 8 | full BPTT | off | 0.0338 | 0.0464 | 0.0042 | 0.0850 |
| generic | 8 | full BPTT | on | 0.0479 | 0.0914 | 0.0043 | 0.1443 |
| generic | 8 | streamed | off | — | — | — | 0.2157 |
| generic | 8 | streamed | on | — | — | — | 0.2365 |
| process | 1 | full BPTT | off | 0.0183 | 0.0208 | 0.0067 | 0.0463 |
| process | 8 | full BPTT | off | 0.0370 | 0.0521 | 0.0042 | 0.0941 |
| process | 8 | full BPTT | on | 0.0589 | 0.0905 | 0.0047 | 0.1556 |
| process | 8 | streamed | off | — | — | — | 0.2126 |
| process | 8 | streamed | on | — | — | — | 0.2116 |

(The subset above spans the K range; all 32 rows per scale are in the JSON.)

**The streamed path has no separable forward/backward phases**, and reports
`null` for them rather than an artefactual split: `backward_streamed_truncated`
interleaves loss-backward with the recurrence, so any per-phase timing would
measure the instrumentation rather than the work. Those rows give the total step
time only — still a real wall-clock measurement with `synchronize()` on both edges.

Phase timings are means over **2 measured steps after 1 warmup** at scale B, so
individual cells carry real step-to-step noise; the K trend (each phase grows
with K) is consistent, but treat single-cell phase ratios as indicative. The
scale-A run used 3 measured steps after 2 warmups.

## 4. Checkpoint save → load → resume on GPU (D3)

`training/r7_local_runner.py` contract, exercised end-to-end on `cuda:0` with
`--bf16`:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train_r7_local.py \
  --config configs/r7_gpu_bringup.yaml --synthetic \
  --out outputs/gpu_d3/stage1 --updates 2 --device cuda --bf16

CUDA_VISIBLE_DEVICES=0 .venv/bin/python train_r7_local.py \
  --config configs/r7_gpu_bringup.yaml --synthetic \
  --out outputs/gpu_d3/stage2 --updates 4 \
  --resume outputs/gpu_d3/stage1/update_0000002.pt --device cuda --bf16
```

Result — resumed **5→10 equivalent** run (2→4 updates) against an uninterrupted
4-update run:

| Check | Result |
| --- | --- |
| signature equal | `True` |
| endpoint `updates` | 4 == 4 |
| `epoch`/`cursor` | (0, 8) == (0, 8) |
| model weights | **bitwise identical** (`torch.equal` on every tensor) |
| optimizer state (`exp_avg`) | `torch.equal` `True` |
| loss sequence steps 3–4 | identical to the reference tail |
| peak allocated | 43058176 B both |
| device / dtype | `cuda`, `bf16 True`, `NVIDIA GeForce RTX 3090` |

Reproducibility grade: **bitwise on this machine/software stack**, matching the
runner's documented scope (no cross-platform bitwise promise).

## 5. Real (non-synthetic) data path on GPU (D2 evidence)

The only real dataset on this disk is `tests/fixtures/r7_arco_t2m.json`
(9×5×7, single-variable `t2m`, `scientific_training_ready: false`). It was
verified by hash and pushed through the real pipeline on GPU:

```bash
# fixture SHA256 verified == 5f859932025f2fa889928991225f0310bfa75853e606cf7819f1b684151ffef1
# prepare_local(...) -> windows {'train': 1, 'val': 1, 'test': 1}
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train_r7_local.py \
  --config configs/r7_gpu_real_fixture.yaml \
  --manifest outputs/gpu_real_fixture/manifest/train.jsonl \
  --out outputs/gpu_real_fixture/train --updates 3 --device cuda --bf16
# evaluate_local(..., device_name='cuda') -> n_evaluated 1, scientific_claim False
```

Outcome: 3 GPU updates completed; `test` split evaluated on GPU, `t2m` RMSE
5.589989 K at +6 h over **1 initialization**. `configs/r7_gpu_real_fixture.yaml`
is single-channel by construction and is deliberately **not** the 11-channel R7
data contract.

**This number is a plumbing check, not skill.** One initialization, one
variable, no multi-variable ERA5 — it must not be quoted as a baseline result.

## 6. Dual-GPU DDP smoke (D4)

This repository had **no** `torchrun` / `init_process_group` /
`DistributedDataParallel` code before this work (verified by grep). The new
`scripts/bench_r7_ddp_smoke.py` runs under
`torchrun --nproc_per_node=2 --standalone`.

```bash
.venv/bin/torchrun --nproc_per_node=2 --standalone scripts/bench_r7_ddp_smoke.py \
  --mode ddp --out outputs/gpu_d4/full --steps 10 --per-gpu-batch 2 \
  --world-size 2 --accumulation 1 --reasoning-steps 4 --bf16

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/bench_r7_ddp_smoke.py \
  --mode reference --out outputs/gpu_d4/full --steps 10 --per-gpu-batch 2 \
  --world-size 2 --accumulation 1 --reasoning-steps 4 --bf16

.venv/bin/python scripts/bench_r7_ddp_smoke.py --mode compare --out outputs/gpu_d4/full \
  --steps 10 --per-gpu-batch 2 --world-size 2 --accumulation 1 --reasoning-steps 4
```

| Requirement | Evidence | Result |
| --- | --- | --- |
| both cards actually participate | `rank_devices: ['cuda:0','cuda:1']`, both `NVIDIA GeForce RTX 3090` | PASS |
| loss matches single-GPU reference | max abs delta **3.11e-06** over 10 updates | PASS |
| sampler has no duplication | pooled 32 indices, `no_duplicates` + `covers_dataset` True | PASS |
| checkpoint written exactly once | `checkpoint_file_count: 1`, `checkpoint_written_by_ranks: [true, false]` | PASS |
| resume correct | resumed 5→10 == uninterrupted 10: weights **bitwise identical**, steps 6–10 losses and consumed indices identical | PASS |
| validation set not double-counted | pooled val ids unique, `val_covers_dataset` True, 12 == 12 expected | PASS |
| global batch defined | `global_batch = 2 per_gpu × 2 world_size × 1 accumulation = 4` | PASS |

The reference is not a re-derivation: it **replays the exact sample indices each
rank recorded**, so agreement is per-sample identity of the trained data, not two
curves that happen to look alike.

### Negative result — DDP is slower here, and does not add per-model memory

| | DDP (2 GPUs) | single-GPU global batch |
| --- | --- | --- |
| wall clock, 10 updates | 1.5181 s | 1.4204 s |
| peak allocated per card | 45.86 MiB | 43.18 MiB |

DDP is ~7 % **slower** at this scale, and each rank holds a full replica so
per-card memory is slightly *higher*. The `SYS`/PCIe non-NVLink topology plus
tiny per-step work makes collective overhead dominate. **Do not report this
throughput as a single-model memory improvement.** A DDP win is only plausible
once per-step compute is large enough to hide the all-reduce — which §3 suggests
requires a much larger batch or model, not a config change.

## 7. OOM boundary and ordered degradation (D5)

Deliberately oversized configuration: K=8, batch 48, grid 128×128, `dim=1024`,
`depth=12`, 178.2M params, BF16, checkpointing off. Protocol digest
`ef35fe8deb0aefc845367faaf2f1808aa690da7368456471d45a537f3e6d560e`.

| model | mode | ckpt | status | peak alloc | peak resv |
| --- | --- | --- | --- | --- | --- |
| generic | full_bptt | off | **oom** | 21.90 GiB | 22.61 GiB |
| generic | full_bptt | on | **oom** | 21.59 GiB | 22.61 GiB |
| generic | streamed | off | **oom** | 21.90 GiB | 22.61 GiB |
| generic | streamed | on | **ok** | 19.25 GiB | 22.61 GiB |
| process | full_bptt | off | **oom** | 21.91 GiB | 22.24 GiB |
| process | full_bptt | on | **oom** | 21.98 GiB | 22.99 GiB |
| process | streamed | off | **oom** | 21.91 GiB | 22.99 GiB |
| process | streamed | on | **ok** | 19.19 GiB | 22.99 GiB |

Failure detail is captured per row, e.g. `Tried to allocate 1.50 GiB. GPU 0 has
a total capacity of 23.56 GiB of which 1.35 GiB is free`, with the peak reached
**before** the failure retained for every row.

Degradation was run **one factor at a time** from that baseline
(`scripts/degrade_r7_gpu_sweep.sh`), not by changing several knobs at once:

| Variant (single change) | `full_bptt off` | `full_bptt on` | `streamed off` | `streamed on` |
| --- | --- | --- | --- | --- |
| grid 128×128 → 64×64 | oom 18.80 GiB | ok 13.40 GiB | oom 18.81 GiB | **ok 5.51 GiB** |
| batch 48 → 24 | oom 22.11 GiB | oom 21.54 GiB | oom 23.06 GiB | **ok 10.04 GiB** |
| K 8 → 4 | oom 21.90 GiB | oom 21.59 GiB | oom 21.90 GiB | **ok 19.24 GiB** |
| dim 1024/12 → 512/8 (32.1M) | oom 22.53 GiB | oom 22.73 GiB | oom 22.53 GiB | **ok 8.79 GiB** |

**The reproducible finding:** in **every** variant, including the ones that
reduced grid, batch, depth *and* K, the only survivor is
**streamed truncated + activation checkpointing on**. Reducing a single
dimensional knob was never sufficient on its own; the combination of the two
memory techniques is what fits. Peaks in the 21–23 GiB rows are the measured
pre-OOM maxima, not estimates.

## 8. Verification summary

| Gate | Command | Result |
| --- | --- | --- |
| Full local suite (GPU present) | `.venv/bin/python -m pytest -q` | **804 passed, 3 skipped, 0 failed** (110.93 s) |
| CI-equivalent suite (`CUDA_VISIBLE_DEVICES=""`) | `.venv/bin/python -m pytest -q` | **798 passed, 9 skipped, 0 failed** |
| New GPU tests | `pytest tests/test_r7_gpu_bringup.py -q` | 8 passed with CUDA; 4 of them self-skip without it |
| Blocking rules | `python tools/check_conventions.py` | **34 rules, 0 violations** |
| CI (exact SHA) | see §9 | push runs, all success, none skipped |

The nine CPU-side skips are: 4 new GPU cases, 2 pre-existing CUDA cases in
`tests/test_r7_local_runner.py`, and 3 pre-existing untracked-fixture cases in
`tests/test_real_data_pipeline.py`. **No test was skipped or weakened to obtain
a pass**; the GPU cases skip because that container has no CUDA device, which is
the documented contract for `pytest.mark.gpu`.

`tests/test_r7_gpu_bringup.py` asserts, on real measurements: streamed peak stays
flat in K; the full-BPTT-vs-streamed growth ratio ordering; GPU checkpoint resume
bitwise identity; two-rank sampler partition without duplication. It also guards
the isolation mechanism statically, since an in-process sweep silently
contaminates `reserved`. The GPU
cases use `pytest.mark.skipif(not torch.cuda.is_available(), ...)` so the
CPU-only CI container skips them instead of failing.

## 9. CI binding

Every measurement in §1–§7 was produced by the instruments at
`45d5c93cd8a6360c4a15f4acd4d2c426d7915b32`.

| Item | Value |
| --- | --- |
| Commit carrying instruments + this record | `45d5c93cd8a6360c4a15f4acd4d2c426d7915b32` |
| Push run / job | `35986500299` / `107590304929` — **success**, 10 steps, 0 failures |
| Doc-only follow-up commit | `375d1d2d` — push run `35987165980` / job `107592445423` — **success** |
| Further verification commits | `a3568f2` (run `35987583870` / job `107593767696`), `e4d38d9` (run `35988134364` / job `107595525958`), `b8578e6` (run `35989405974` / job `107599630314`) — all **success** |
| Isolated-measurement correction commit | `9016fa7` — push run `35991675294` / job `107606992215` — **success**, 10/10 steps |
| Final tip at time of writing | `0a19b83` — push run `35992075643` / job `107608283359` — **success**, all steps `success` |
| Job steps (all runs) | conventions, compile+whitespace, unit/integration/installed-wheel all `success` |
| PR runs at those SHAs | `35986505853`, `35987172000` — **skipped** by `ci.yml` design; **not used as evidence** |
| Baseline commit (previous) | `50954c94eb0aa15fa61cb19440543b40c6c38326`, push run `35976102003` / job `107556844837` — success |

The evidence-binding SHA for every measurement is `45d5c93c`. Later commits are
instruments/docs and each carries its own push-triggered run, so this is a
snapshot of verified SHAs rather than of the moving branch tip; the commit that
adds or edits these lines necessarily carries a later run of its own.

Job-log download is not available to an unauthenticated caller
(`GET /actions/jobs/.../logs` → **403 "Must have admin rights to Repository"**),
so the per-test counts above are reproduced locally in the CI-equivalent
configuration (`CUDA_VISIBLE_DEVICES=""`, 797 passed / 9 skipped / 0 failed)
rather than quoted from the job log. The CI job's own step conclusions were
read from the public API and all report `success`.

## 10. Limitations and negative results (kept on purpose)

1. Sections 3–4 and 6 use **synthetic shape data**, not multivariate real ERA5.
2. No real 11-channel ERA5 on this disk (`data/*` hold only `.gitkeep`); the
   continuous cache is in a CI artifact and needs network + authorization.
3. §5's real-fixture RMSE is **1 initialization**, single variable — not skill.
4. **DDP was slower** here and does not increase single-model memory headroom.
5. **Measurement artefact, and a corrected conclusion.** Peak `reserved` grows
   with K *when a whole sweep runs in one process*, because
   `max_memory_reserved` carries the allocator high-water mark between cells.
   Measured per process, streamed `reserved` is flat (46.00 MiB scale A,
   380.00 MiB scale B) across K=1…8. The earlier "reserved rises with K
   everywhere" reading was the artefact. Both datasets are retained.
6. Activation checkpointing bounds the peak but costs **21 % (scale A) to 70 %
   (scale B)** more step time — configuration-dependent, not a constant.
7. The sweep covers K ∈ {1,2,4,8} at two scales and one batch size; no
   multi-seed, convergence or timing-stability claim. Absolute step times vary
   up to ~40 % between runs, so compare within one run.
8. Fewer reasoning steps was **not** measured as a latency win anywhere here;
   step time rises with K in every row.
9. Issue comments could not be published (no `gh`, no token, POST → 401). The
   drafts in §11 are **not published**. The same restriction blocked creating
   the dedicated GPU child issue the brief suggested.

## 11. Issue-comment drafts — NOT PUBLISHED

Posting is blocked on this machine (`command -v gh` empty; no GitHub token;
`POST /issues/20/comments` → 401). The text below is a draft for manual posting.

> **[R7-GPU] Dual-3090 bring-up: measured baselines (draft, unpublished)**
>
> At `45d5c93c` (CI push run `35986500299` / job `107590304929`, success):
> single-GPU sweep over both recursive models × K=1/2/4/8 × {full BPTT,
> streamed truncated} × checkpointing on/off, 32/32 cells OK at
> 1.25M and 18.07M params, **one process per cell**. Streamed peak is **flat
> across K=1…8 on both metrics** — allocated ~41.2 MiB / reserved 46.00 MiB at
> 1.25M params, and 367.7 MiB / 380.00 MiB at 18.07M — while full BPTT grows
> (reserved 394 → 426 MiB at 18.07M). Note: measuring the whole sweep in one
> process inflates `reserved` via allocator carryover (process K=1 read 428 MiB
> instead of 396 MiB), so report per-process numbers. Checkpointing bounds the
> peak but costs 21–70 % more step time depending on config.
> Single-GPU memory is **not** the binding constraint at the planned model
> scale (~368 MiB at 18.07M params). GPU checkpoint resume is bitwise
> identical. First DDP smoke in-repo: 2 ranks, loss matches the single-GPU
> reference to 3.1e-06, sampler covers the dataset exactly once, one checkpoint,
> resume bitwise identical — but DDP is **7 % slower** at this scale on the
> SYS/PCIe (non-NVLink) link. OOM boundary at 178.2M params; only
> streamed+checkpointing survived every ordered degradation variant. Real
> multivariate ERA5 is **not** on this disk; memory numbers use synthetic
> shapes. `scientific_claim: false` throughout.

## 12. Artifact locations (not committed, per R-035)

| Content | Path |
| --- | --- |
| Single-GPU sweeps (isolated, authoritative) | `outputs/gpu_sweep_isolated/`, `outputs/gpu_sweep_isolated_a/` |
| Single-GPU sweeps (in-process, retained for the artefact comparison) | `outputs/gpu_sweep_full/`, `outputs/gpu_sweep_stageb/` |
| Checkpoint resume | `outputs/gpu_d3/` |
| DDP smoke + resume + reference | `outputs/gpu_d4/` |
| OOM boundary + degradation | `outputs/gpu_oom_boundary/`, `outputs/gpu_degrade/` |
| Real-fixture GPU run | `outputs/gpu_real_fixture/` |
| Logs | `logs/gpu_*.log`, `logs/degrade_*.log`, `logs/isolated*.log` |

## 13. Reproduce

**If you re-measure, use one process per cell** — otherwise the `reserved`
column silently inherits earlier cells' allocator high-water mark:

```bash
# environment
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
nvidia-smi nvlink -s; nvidia-smi topo -m

# single-GPU sweep, isolated processes (§3; authoritative tables)
bash scripts/sweep_r7_gpu_isolated.sh outputs/gpu_sweep_isolated 384 8   # scale B
bash scripts/sweep_r7_gpu_isolated.sh outputs/gpu_sweep_isolated_a 128 4 # scale A

# single-GPU sweep in one process (retained only for the artefact comparison)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/bench_r7_gpu_memory.py \
  --out outputs/gpu_sweep_full --device cuda --steps 1 2 4 8 --batch 4 \
  --dim 128 --depth 4 --channels 11 --grid 12 12 --bf16

# dual-GPU DDP smoke + reference + compare (§6)
bash scripts/degrade_r7_gpu_sweep.sh   # ordered single-factor degradation (§7)

# tests and gates
.venv/bin/python -m pytest -q
.venv/bin/python tools/check_conventions.py
```

## 14. What this does and does not authorize

Established: the GPU paths run, memory and latency are measured rather than
assumed, checkpoints resume exactly, and two-card DDP is correct.

**Not** established, and not claimed: that recursion improves forecast skill
(#5/#6), that adaptive halting helps (#7), that rollouts are stable (#8), or
that any parent research issue is satisfied. `#20`'s engineering items are
covered; its **multi-seed optimization comparison is not**, and the parent
research issues stay open. **Engineering passing is not a scientific result.**
