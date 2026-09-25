# R7 #61 — DDP smoke: verified reasoning depth, resume contract, padded samplers

Status: **DONE for the CPU + local 2×RTX 3090 acceptance; real-region throughput
measurement stays BLOCKED on data** (dependency recorded below).
`scientific_claim: false` throughout; synthetic shape fixture, no weather truth.

## The bug (issue #61)

`scripts/bench_r7_ddp_smoke.py` called
`batch_loss(ddp_model, batch, args.steps, args.bf16)` in the validation pass —
`args.steps` is the optimizer-update count, so a "10 updates / K4" smoke
validated K=10. Resume checked only format/world_size/model-code; seed,
per-GPU batch, accumulation, reasoning depth and data identity changes were
accepted. Lengths 31/33 (non-divisible) had never run, `no_sync` was never
exercised, and the script did not say which training semantics it verified.

## What changed (A→D; E measured)

- **A — depth independence**: validation depth comes only from
  `--reasoning-steps` (`run_validation`); a forward hook on the recursion cell
  records the depth actually executed per validation batch and per training
  microbatch, and the run refuses to report a K it did not observe. AST +
  dynamic fake-model CPU tests pin this (`tests/test_r7_ddp_smoke_contract.py`).
  The smoke payload/report keep their original structure (v2 checkpoint format
  adds contract fields; v1 checkpoints are refused with the reason).
- **A — resume contract** (`validate_resume_contract`, pure function):
  rejects changed seed / per_gpu_batch / accumulation / reasoning_steps /
  training_mode / dataset_signature / world_size / model code, checkpoints
  missing any contract field, self-inconsistent signatures, and non-advancing
  endpoints — each with a named-field message. Same-contract resume is
  verified against an uninterrupted run (results below).
- **B/C — real shapes**: `--length/--val-length` expose non-divisible
  datasets. `check_sampler_partition` now reports `DistributedSampler`
  padding explicitly (padded indices and counts) instead of claiming "no
  duplicates"; per-sample weighting is verified by the single-GPU reference
  replaying the exact consumed indices; partial accumulation groups are
  recorded per update.
- **C/D — no_sync**: complete accumulation groups defer the all-reduce to the
  final microbatch (`no_sync_for` rule, forward included in the context);
  incomplete groups synchronize on their final microbatch. Gradient semantics
  are verified by loss/weight identity against the reference.
- **D — mode labels**: `full_bptt` / `retained_truncated`
  (detach-between-steps) / `streamed_truncated` are labeled separately and
  never claimed equivalent; `streamed_truncated` is **refused** in every mode
  of this script (would otherwise silently fall back to full BPTT). The report
  states what is verified (generic arm, synthetic, these modes) and what is
  not (Process arm, real manifests, streamed+DDP).
- **E — throughput**: same global batch 4, 30 updates, K4, bf16:
  single GPU 2.84 s vs dual 2.65 s — no DDP advantage demonstrated at smoke
  scale (toy model, and GPU0 carried a co-tenant job during the run). Standing
  decision per the issue: on SYS topology the two cards run independent seeds
  per arm; DDP is not used for speed here.

## Local 2×RTX 3090 verification (all artifacts in `outputs/r7_ddp_smoke_61/`)

| Run | Contract | Key results |
| --- | --- | --- |
| `main/` | full_bptt, steps=10, K=4, length 31, accum 3, bf16 | observed eval K `[4,4,4,4,4,4]` (never 10); training K per microbatch 4; loss max Δ vs reference 1.7e-06; weight max Δ 4.0e-04 (bf16, 111 keys); padding 31→32 (1 padded index 19); partial groups `[2]×6`; global batches `[12,12,8,…]`; rank0-only checkpoint; val 12 ids unique |
| `retained/` | retained_truncated, steps=6, K=2, length 33, accum 1 | loss max Δ 1.4e-06; observed K=2 eval+train; padding 33→34 (padded index 6); labels recorded |
| `resume_neg/` | resume with `--reasoning-steps 5` after a K=4 run | **rejected by both ranks**: `resume contract change rejected: reasoning_steps is 4 … but 5 …` |
| `resume_a`/`resume_b` | direct 20 updates vs 10 + same-contract resume→20 | **111/111 weight hashes identical (max Δ 0.0)**, optimizer state equal, loss curves paired by update id: max Δ 0.0 over updates 11–20 |
| `perf_single`/`perf_dual` | global batch 4 × 30 updates, K4, bf16 | single 2.84 s vs dual 2.65 s; no DDP throughput case at this scale |

GPU budget consumed: ≈0.1 GPU-hours (all runs short by design).

## CPU verification

- `tests/test_r7_ddp_smoke_contract.py`: 30 tests — AST proof that no
  `batch_loss` call passes `args.steps` as a depth and validation binds
  `reasoning_steps` explicitly; dynamic fake-model observation of K=4 with
  update count 10 as decoy; refusal of unobserved depths; all resume-contract
  negative cases + happy path; sampler padding for 31/32/33; mode-label
  distinctness; streamed refusal; `no_sync_for` truth table; dataset
  signature binding; CPU end-to-end microbatch step (per-sample weighting,
  observed cell calls = 2×K, gradients flow, weights move).
- Full suite: **939 passed / 3 skipped**; conventions **34 blocking rules,
  0 violations**.

## BLOCKED (named dependency)

- **Real-region DDP throughput (E)**: the only real regional store on this
  disk is the 1-window `outputs/gpu_real_fixture/`; a throughput measurement
  needs the #63 D1 continuous 30-day segment. Until then the synthetic
  same-global-batch probe above is the only measurement, and the
  two-cards-independent-seeds decision stands.
- Streamed+DDP and Process-arm DDP: refused/unclaimed by design until a
  dedicated verification exists (labeled in every report).
