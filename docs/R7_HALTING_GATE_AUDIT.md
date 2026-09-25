# R7 adaptive halting: is the gate satisfiable? (#7)

Issue #7's research gate is: *"adaptive model should approach fixed-Kmax accuracy
with meaningfully lower average reasoning depth."* That sentence carries an
assumption — that fixed-Kmax is the accuracy worth approaching. Before re-tuning
any controller, this page tests the assumption against a **measured**
error-versus-depth curve on real ERA5.

The audit is negative, and it explains why the earlier policy search kept
returning full depth.

`scientific_claim: false`. Test split not read. This audits existing bounded CPU
results; it trains nothing and re-tunes nothing.

## What the audit checks

Both checks run from one existing multi-depth evaluation — the #6 fair-budget run,
which evaluated K=0/1/3 for all three arms at three seeds on real regional ERA5.
No new training was performed.

1. **Is the reference the ceiling?** Compare the equal-channel normalized
   objective at each depth, per arm and per seed. This objective is the one the
   training loss minimizes (equal-channel MSE on normalized fields), so it — not
   any single physical variable — is the right yardstick for "accuracy".
2. **Is the per-variable tolerance satisfiable?** The established policy search
   (`training/r7_policy_selection.py`) requires *every* variable and horizon to
   meet a relative-RMSE tolerance separately, with no aggregation. The audit
   reports, per tolerance, whether any shallower depth qualifies and which
   variable blocks it.

## Result 1: deeper is not better

Equal-channel normalized RMSE, mean over three seeds:

| arm | K=0 | K=1 | K=3 | K=3 vs K=0 |
| --- | --- | --- | --- | --- |
| `generic` | 0.31079 | 0.30991 | 0.32117 | **+3.3 %** |
| `process_feedback` | 0.31519 | 0.31484 | 0.31976 | **+1.5 %** |
| `process_no_feedback` | 0.31594 | 0.31589 | 0.32166 | **+1.8 %** |

Counted per seed, `process_no_feedback` is worse at K=3 than K=0 in **all three
seeds**; the other two arms are worse in two of three. So the fixed-Kmax reference
is *not* the best available accuracy on this data — it is worse than the shallowest
measured depth.

This is consistent with the per-variable picture, where the depth trend is
genuinely split: at K=3 versus K=0, `process_feedback` has 5 variables
consistently improving with depth (`mslp`, `t2m`, `t500`, `v250`, `z850`), 5
consistently worsening (`q500`, `q850`, `u500`, `v10`, `v500`), and 7 unresolved.
Deeper reasoning is not uniformly a correction; it trades some variables for
others.

## Result 2: no shallower depth passes the per-variable tolerance

Against the K=3 reference, for `process_feedback`:

| tolerance | cadence | variables within tolerance | any depth feasible | blocking variable | blocking ratio |
| --- | --- | --- | --- | --- | --- |
| 0 % | K=0 | 9/17 | **no** | `z250` | 1.190 |
| 0 % | K=1 | 9/17 | **no** | `z250` | 1.142 |
| 1 % | K=0 | 11/17 | **no** | `z250` | 1.190 |
| 1 % | K=1 | 12/17 | **no** | `z250` | 1.142 |
| 5 % | K=0 | 15/17 | **no** | `z250` | 1.190 |
| 5 % | K=1 | 16/17 | **no** | `z250` | 1.142 |
| 10 % | K=0 | 16/17 | **no** | `z250` | 1.190 |
| 10 % | K=1 | 16/17 | **no** | `z250` | 1.142 |

This is the mechanism behind the earlier observation that the strict policy
"fell back to full depth". It is not that the tolerance was set too tight in
isolation: `z250` sits 14–19 % above the K=3 reference for every shallower depth,
so **no** tolerance below 19 % admits a shallower depth, and even at 10 % a single
variable vetoes the reduction.

## Combined verdict

`gate_satisfiable: false`, for two independent reasons, either of which is
sufficient:

- the fixed-Kmax reference is **not** the accuracy ceiling on this data, so
  "approach fixed-Kmax" would mean approaching a worse objective; and
- no shallower depth passes the per-variable tolerance at any tested tolerance.

## An important caveat about the blocker

`z250` is the tolerance blocker, but its own numbers are seed-unstable: at
`process_feedback` K=0 the per-seed RMSEs are 257.7, 235.7, 188.2, and the K=3
minus K=0 deltas are −48.0, +6.6, −67.3 — **not** the same sign across seeds. So
`z250` is not a robustly-established loss; it is a variable with large
seed-to-seed spread that happens to exceed the tolerance. The honest statement is
that the per-variable tolerance rule is not satisfiable on this data, and the
specific blocking variable is not itself a reliable finding. Both facts are
recorded because either alone would mislead.

## What this does and does not establish

**Established.** On a bounded CPU comparison over real regional ERA5 with three
seeds, deeper reasoning does not improve the normalized objective, and the
existing strict per-variable tolerance cannot admit any depth reduction against a
K=3 reference. The earlier "the policy selected full depth" outcome is explained.

**Not established.** That adaptive halting is impossible in general. A controller
that *reduces* depth wherever depth is harmful could still help — but that is a
different, per-variable objective from the one #7 states, and it would need the
gate rewritten before it could be tested. Claiming a win by quietly relaxing the
gate or by dropping the blocking variable is exactly what must not happen here.

## Tests

| Suite | Result |
| --- | --- |
| `tests/test_r7_halting_gate_audit.py` | 9 passed (per-seed depth-benefit counting, mixed-benefit non-flagging, equal-channel normalization, tolerance feasibility mirroring the policy search, blocker identification, unsatisfiable-gate verdict, minimum-depth requirement, exclusive self-describing output) |
| Full `pytest -q` with `CUDA_VISIBLE_DEVICES=""` | 859 passed, 9 skipped, 0 failed |
| `python tools/check_conventions.py` | 34 blocking rules, **0 violations** |

## Honest limitations

- **Bounded CPU comparison as the substrate**: 200 updates, `dim=32, depth=2`,
  ten January days per year, 38 windows per split, one lead time (+6 h).
- **Three seeds**: a noise-floor check, not a significance test.
- **Validation split only**; the test split was not read.
- **The depths measured are 0/1/3**, so a benefit confined to intermediate or
  larger K cannot be excluded.
- **The audit inherits the substrate's `scientific_claim: false`.** No controller
  was trained, no threshold was selected, and the gate was not relaxed.

## Reproduce

```bash
.venv/bin/python training/r7_halting_gate_audit.py \
  --result outputs/r7_coreasoning_v2/coreasoning_result.json \
  --store outputs/r7_coreasoning_v2/dataset/cache.zarr \
  --out outputs/r7_halting_audit/report.json
```
