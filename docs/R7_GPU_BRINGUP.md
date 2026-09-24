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
12×12, FP32 master weights with BF16 autocast, 2 warmup + 3 measured steps
(scale A) / 1 + 2 (scale B).

### Scale A — `dim=128, depth=4`, 1,247,894 params (generic)

Protocol digest `c10440479a8d8e8b3bc6c508478bd1f5fbd3e3a0ca6e58ed779aa2f5a146ad59`, 32/32 cells `ok`, 20.4 s.

| model | K | mode | ckpt | peak alloc MiB | peak resv MiB | step s |
| --- | --- | --- | --- | --- | --- | --- |
| generic | 1 | full_bptt | off | 41.14 | 46.00 | 0.0296 |
| generic | 1 | full_bptt | on | 42.14 | 48.00 | 0.0517 |
| generic | 1 | streamed | off | 42.16 | 48.00 | 0.0530 |
| generic | 2 | full_bptt | off | 42.14 | 48.00 | 0.0521 |
| generic | 2 | streamed | off | 42.16 | 48.00 | 0.0757 |
| generic | 4 | full_bptt | off | 44.25 | 50.00 | 0.0716 |
| generic | 4 | full_bptt | on | 42.14 | 50.00 | 0.0931 |
| generic | 4 | streamed | off | 42.16 | 50.00 | 0.1083 |
| generic | 8 | full_bptt | off | 49.43 | 54.00 | 0.1197 |
| generic | 8 | full_bptt | on | 42.14 | 54.00 | 0.1446 |
| generic | 8 | streamed | off | 42.16 | 54.00 | 0.2233 |
| generic | 8 | streamed | on | 42.16 | 54.00 | 0.2590 |

Process rows at the same scale (1,247,255 params) are in
`outputs/gpu_sweep_full/sweep.json`; they track generic within ~1 MiB, e.g.
`process K=8 full_bptt off` = 47.58 MiB vs generic 49.43 MiB, and
`process K=8 streamed off` = 42.15 MiB.

### Scale B — `dim=384, depth=8`, 18,070,000 params (generic, inside the 15–30M target)

Protocol digest `3664318d35e78ced2ff868e423336859ff28cf8458b26a3cdb29aef7809aac63`, 32/32 cells `ok`, 19.3 s.

| model | K | mode | ckpt | peak alloc MiB | peak resv MiB | step s |
| --- | --- | --- | --- | --- | --- | --- |
| generic | 1 | full_bptt | off | 366.31 | 394.00 | 0.0457 |
| generic | 1 | full_bptt | on | 372.75 | 394.00 | 0.0748 |
| generic | 1 | streamed | off | 371.65 | 394.00 | 0.0675 |
| generic | 2 | full_bptt | off | 371.62 | 404.00 | 0.0618 |
| generic | 2 | streamed | off | 371.65 | 404.00 | 0.0902 |
| generic | 4 | full_bptt | off | 371.62 | 412.00 | 0.0739 |
| generic | 4 | streamed | off | 371.65 | 412.00 | 0.1298 |
| generic | 8 | full_bptt | off | 371.62 | 428.00 | 0.1203 |
| generic | 8 | full_bptt | on | 372.75 | 428.00 | 0.1560 |
| generic | 8 | streamed | off | 371.65 | 428.00 | 0.2382 |
| generic | 8 | streamed | on | 371.65 | 428.00 | 0.2762 |

### What the table shows

- **Streamed truncated training is flat in K; full BPTT grows.** At scale A the
  streamed peak is 42.16 MiB for every K from 1 to 8, while full BPTT climbs
  41.14 → 49.43 MiB. This is the first *measured GPU* confirmation of the claim
  `docs/R7_STREAMED_TRAINING.md` previously supported only by logical
  saved-tensor accounting (#21 delivered 701368 bytes at K=1/2/4/8 — bookkeeping,
  not VRAM).
- **Peak `reserved` still rises with K** (46 → 54 MiB at scale A). The allocator
  caches per-round blocks even though live allocation is flat. Reporting only
  `allocated` would understate real VRAM pressure, so both are given.
- **Activation checkpointing buys memory and costs time.** At scale A, `K=8
  full_bptt` drops 49.43 → 42.14 MiB allocated while step time rises
  0.1197 → 0.1446 s (+21 %). Recompute is not free.
- **No memory crossover at these scales.** 18.07M params needs only ~372 MiB of
  a 23.56 GiB card. Single-GPU memory is *not* the binding constraint for the
  planned model size — this redirects the next decision (see §7).

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
| Full local suite (GPU present) | `.venv/bin/python -m pytest -q` | **803 passed, 3 skipped, 0 failed** (122.25 s) |
| CI-equivalent suite (`CUDA_VISIBLE_DEVICES=""`) | `.venv/bin/python -m pytest -q` | **797 passed, 9 skipped, 0 failed** (105.57 s) |
| New GPU tests | `pytest tests/test_r7_gpu_bringup.py -q` | 7 passed with CUDA; 3 passed + 4 self-skipped without it |
| Blocking rules | `python tools/check_conventions.py` | **34 rules, 0 violations** |
| CI (exact SHA) | see §9 | push run, success, not skipped |

The nine CPU-side skips are: 4 new GPU cases, 2 pre-existing CUDA cases in
`tests/test_r7_local_runner.py`, and 3 pre-existing untracked-fixture cases in
`tests/test_real_data_pipeline.py`. **No test was skipped or weakened to obtain
a pass**; the GPU cases skip because that container has no CUDA device, which is
the documented contract for `pytest.mark.gpu`.

`tests/test_r7_gpu_bringup.py` asserts, on real measurements: streamed peak stays
flat in K; the full-BPTT-vs-streamed growth ratio ordering; GPU checkpoint resume
bitwise identity; two-rank sampler partition without duplication. The GPU cases
use `pytest.mark.skipif(not torch.cuda.is_available(), ...)` so the CPU-only CI
container skips them instead of failing.

## 9. CI binding

Every measurement in §1–§7 was produced by the instruments at
`45d5c93cd8a6360c4a15f4acd4d2c426d7915b32`.

| Item | Value |
| --- | --- |
| Commit carrying instruments + this record | `45d5c93cd8a6360c4a15f4acd4d2c426d7915b32` |
| Push run / job | `35986500299` / `107590304929` — **success**, 10 steps, 0 failures |
| Doc-only follow-up commit | `375d1d2d` — push run `35987165980` / job `107592445423` — **success** |
| Latest verification commit | `a3568f2` — push run `35987583870` / job `107593767696` — **success**, 10/10 steps |
| Latest verification commit | `e4d38d9` — push run `35988134364` / job `107595525958` — **success**, 10/10 steps |
| Job steps (all runs) | conventions, compile+whitespace, unit/integration/installed-wheel all `success` |

The evidence-binding SHA for every measurement below is `45d5c93c`. Later
doc-only commits each carry their own push-triggered run, so this table is a
snapshot of verified SHAs rather than of the moving branch tip.
| PR runs at those SHAs | `35986505853`, `35987172000` — **skipped** by `ci.yml` design; **not used as evidence** |
| Baseline commit (previous) | `50954c94eb0aa15fa61cb19440543b40c6c38326`, push run `35976102003` / job `107556844837` — success |

This table names only already-published commits; the commit that adds these
lines necessarily carries its own later run.

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
5. Peak `reserved` grows with K even where `allocated` is flat.
6. Activation checkpointing costs ~21 % step time at K=8.
7. The sweep covers K ∈ {1,2,4,8} at two scales and one batch size; no
   multi-seed, convergence or timing-stability claim.
8. Fewer reasoning steps was **not** measured as a latency win anywhere here;
   step time rises with K in every row.
9. Issue comments could not be published (no `gh`, no token, POST → 401). The
   drafts in §11 are **not published**.

## 11. Issue-comment drafts — NOT PUBLISHED

Posting is blocked on this machine (`command -v gh` empty; no GitHub token;
`POST /issues/20/comments` → 401). The text below is a draft for manual posting.

> **[R7-GPU] Dual-3090 bring-up: measured baselines (draft, unpublished)**
>
> At `45d5c93c` (CI push run `35986500299` / job `107590304929`, success):
> single-GPU sweep over both recursive models × K=1/2/4/8 × {full BPTT,
> streamed truncated} × checkpointing on/off, 32/32 cells OK at
> 1.25M and 18.07M params. Streamed peak allocated is **flat at 42.16 MiB for
> K=1…8** while full BPTT grows 41.14 → 49.43 MiB; peak *reserved* still grows
> 46 → 54 MiB. Checkpointing trades ~21 % step time for ~7 MiB at K=8.
> Single-GPU memory is **not** the binding constraint at the planned model
> scale (~372 MiB at 18.07M params). GPU checkpoint resume is bitwise
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
| Single-GPU sweeps | `outputs/gpu_sweep_full/`, `outputs/gpu_sweep_stageb/` |
| Checkpoint resume | `outputs/gpu_d3/` |
| DDP smoke + resume + reference | `outputs/gpu_d4/` |
| OOM boundary + degradation | `outputs/gpu_oom_boundary/`, `outputs/gpu_degrade/` |
| Real-fixture GPU run | `outputs/gpu_real_fixture/` |
| Logs | `logs/gpu_*.log`, `logs/degrade_*.log` |

## 13. Reproduce

```bash
# environment
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
nvidia-smi nvlink -s; nvidia-smi topo -m

# single-GPU sweep (Stage A / Stage B)
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
