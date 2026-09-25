# R7 process co-reasoning: fair-budget comparison (#6)

Issue #6's research gate is that process-aware co-reasoning must **beat the generic
recursive baseline under a comparable budget**. This page reports the fair-budget
comparison on real regional ERA5. The conclusion is negative and, more usefully,
it is now quantified rather than asserted.

`scientific_claim: false`. Test split not read; three seeds; no significance test.

## Why the question was still open

The prior evidence was known to be mixed, and #5 had only just established that
the arms are budget-comparable in principle (+0.030 % parameters,
+0.0001–0.0003 % forward FLOPs). What was missing was a run that (a) held the
budget fixed, (b) held the seed fixed per comparison, and (c) reported **every**
variable instead of a headline — because with 17 channels across K, Pa and m/s, a
single aggregate number would be meaningless.

## Protocol (frozen before any optimizer step)

`protocol.json` was written before the cache was built or any arm trained
(R-006). Digest `253c5a442934fa82f6d7bd2bbdfa8c1d0ca668175faad5bbfbcf700bf6bec296`.

| Item | Value |
| --- | --- |
| Arms | `generic` (16 latent tokens), `process_no_feedback` (8 anchored + 8 free, feedback off), `process_feedback` (same, feedback on) |
| Seeds | 41, 42, 43 — all retained |
| Optimizer updates | 200, identical for every arm and seed |
| Batch / lr / clip | 2 / 2e-4 / 1.0; training depth K=3; process weight 0.1 |
| Ablation depths | K=0 (no recursion), K=1, K=3 |
| Evaluation | validation split only, +6 h lead, 8 capped initializations, all 17 channels |
| Parameters | 92,450 generic vs 92,547 process — a 97-parameter (0.10 %) process readout |
| Data | `outputs/r7_coreasoning_data/source.nc`, SHA256 `db7c02191e520a81873ab90b2a35348a43d556fb65a82da0cd60cce03040…` |

The extraction is one contiguous ten-day block from January 1 in each of
2018/2019/2020 at native 0.25°, 65x65, all nine source variables with the full
13-level axis, acquired anonymously under a 96 GiB touched-chunk cap. It yields
**38 one-step windows per split** after the audited converter, with
`BUILD_COMPLETE.json` and train-only normalization. The file is 91,182,540 bytes
and its SHA256 was re-derived from the bytes and matched against the receipt:
`db7c02191e520a81873ab90b2a35348a43d556fb65a82da0cd60cce03041a2f4`. It took
2,597.3 s to acquire (120 timestamps).

## Result: the gate is not met

**Both process arms fail.** For `process_feedback` at K=3, 10 of 17 variables
have a lower mean RMSE than generic and 7 have a higher one — and **15 of those 17
deltas change sign between seeds**, which means the difference is smaller than
seed-to-seed variation and the win/loss split is not evidence of anything.

Restricting to deltas whose sign is consistent across all three seeds:

| depth | arm | established improved | established worsened | unresolved |
| --- | --- | --- | --- | --- |
| K=0 (no recursion) | process_feedback | 1 (`z250`) | 6 (`mslp`,`t2m`,`t850`,`u500`,`v250`,`z850`) | 10 |
| K=0 | process_no_feedback | 1 (`z250`) | 5 (`mslp`,`t2m`,`u500`,`v250`,`v500`) | 11 |
| K=1 | process_feedback | 3 (`u10`,`u250`,`v850`) | 5 (`mslp`,`t2m`,`t850`,`v250`,`z850`) | 9 |
| K=1 | process_no_feedback | 1 (`v10`) | 5 (`mslp`,`t2m`,`t850`,`v250`,`z850`) | 11 |
| K=3 | process_feedback | 1 (`u10`) | 1 (`t2m`) | **15** |
| K=3 | process_no_feedback | 2 (`t500`,`v10`) | 2 (`t2m`,`v250`) | 13 |

`gate_met: false` for every arm and depth. Three facts are worth stating plainly:

1. **The clearest established effect is a loss.** `t2m` worsens at K=0, K=1 and
   K=3 for both process arms, with all three seeds agreeing each time. The
   magnitude at K=3 is +0.23 K (`no_feedback`) and +0.30 K (`feedback`).
2. **Forecast feedback does not rescue it.** At K=3 the feedback arm has *fewer*
   established improvements (1) than the no-feedback arm (2), and adds a second
   worsening not present in the no-feedback arm (`v250`). On this evidence
   feedback is not the missing ingredient.
3. **Most differences are unresolved.** At K=3, 15/17 deltas for the feedback arm
   and 13/17 for no-feedback flip sign between seeds. The honest reading is that
   the process arms are, for most variables, indistinguishable from generic at
   three seeds — not that they are better.

A single-variable note where this reconciles with earlier work: `t500` is an
established *improvement* for `process_no_feedback` at K=3 (−0.023 K, all seeds),
while `t2m` is an established loss. That is consistent with the previously
recorded mixed picture — some variables move one way, others the other — but the
earlier framing that spatial feedback specifically hurts T500 does not reproduce
here: this run shows T500 improving under the no-feedback arm and unresolved under
the feedback arm at K=3.

## Tests

| Suite | Result |
| --- | --- |
| `tests/test_r7_coreasoning_compare.py` | 12 passed (arm-config parity, protocol freeze and its declared limits, per-variable reporting with units, seed sign-stability gating, depth/lead pairing, refusal of a non-real receipt, refusal of a `synthetic_fallback` receipt) |
| Full `pytest -q` with `CUDA_VISIBLE_DEVICES=""` | 850 passed, 9 skipped, 0 failed |
| `python tools/check_conventions.py` | 34 blocking rules, **0 violations** |

The sign-stability gate is enforced in the module, not just in this prose: a
comparison block reports `seed_paired[].direction` as `unresolved` whenever the
per-seed deltas disagree, and `gate_met` is true only if every sign-consistent
variable improved. Two tests pin exactly the situation this run hit — an arm that
looks better on average while every per-seed delta flips sign must be reported as
unresolved.

## Honest limitations

- **Three seeds is a noise floor check, not a significance test.** No p-values
  were computed; the sign-consistency rule is deliberately conservative and is not
  a substitute for a proper paired test.
- **Small data and a short endpoint.** Ten January days per year; 38 windows per
  split; 200 optimizer updates at `dim=32, depth=2`. This is a bounded CPU
  comparison, not a converged training run.
- **Validation cases were inspected in earlier iterations**, so this is not a
  pristine held-out benchmark.
- **One lead time** (+6 h). Longer leads are #8's scope.
- **The adaptive arm is not part of this comparison**; it has its own record in
  [R7_ADAPTIVE_HALTING.md](R7_ADAPTIVE_HALTING.md).
- Engineering passes and budget parity are **not** evidence for the scientific
  hypothesis, and here the hypothesis is not supported: the process arms do not
  beat the generic baseline under a comparable budget.

## Reproduce

```bash
.venv/bin/python training/r7_coreasoning_compare.py \
  --source outputs/r7_coreasoning_data/source.nc \
  --receipt outputs/r7_coreasoning_data/source_receipt.json \
  --out outputs/r7_coreasoning_v2 --updates 200 --max-samples 8
```

Output directories must be new (`prepare_local` and the result writer both refuse
existing paths).
