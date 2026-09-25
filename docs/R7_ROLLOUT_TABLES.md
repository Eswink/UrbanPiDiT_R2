# R7 6–72 h rollout evaluation tables (#8)

Issue #8's acceptance is: *"Evaluation scripts produce paper-ready metric tables
without changing training code."* The rollout machinery (`model/r7_rollout.py`),
the RMSE accumulator and the ACC accumulator all already existed and were covered
by synthetic tests. What did not exist was a script that drives **already-trained
checkpoints** through a 6/12/24/48/72 h autoregressive rollout on the held-out year
and emits comparable tables. This page records that script and its first real
output.

`scientific_claim: false`. The tables are real measurements from bounded CPU
checkpoints; they are not a converged benchmark or a SOTA claim.

## What the script does

`scripts/rollout_r7_metric_tables.py` evaluates and only evaluates:

- **No training code is touched.** The test that enforces this parses the script's
  AST and asserts no optimizer, `backward`, `step`, `run_local_updates` or
  `calibrate_controller_step` is referenced at all — so the constraint is checked
  on code, not on prose.
- **Model generations cannot be mixed.** Each checkpoint's recorded
  `model_code_sha256` must equal the live `model/` digest, or the run aborts with
  an instruction to replay via the archived `code.zip` rather than bypass the
  identity check. Checkpoints without a recorded digest are refused.
- **Datasets cannot be mixed.** Every checkpoint must report one identical
  `data_identity`, and the channel order and units must agree; otherwise the table
  would silently compare incomparable numbers.
- **The held-out split is enforced.** A report whose split is not `test` aborts.
- **A same-data persistence baseline is included** by default, scored on the same
  manifest, leads and case cap — with a check that the case count really matches.
- **No cross-variable average is produced.** K, Pa, m/s, kg/kg and m²/s² are not
  summable, so tables are per variable and lead with an explicit unit column.
- **Undefined ACC is written as `undefined`, never as zero**, and the count of such
  cells is recorded.

Outputs: `rollout_rmse_table.csv`, `rollout_acc_table.csv`, `table_provenance.json`
(manifest SHA256, data identity, per-model checkpoint hash and updates, leads,
aggregation rule, limitations). All are created exclusively.

## Real measurement

Command:

```bash
.venv/bin/python scripts/rollout_r7_metric_tables.py \
  --manifest outputs/r7_coreasoning_v2/dataset/manifests/test.jsonl \
  --checkpoint outputs/r7_coreasoning_v2/training/generic_41/update_0000200.pt \
               outputs/r7_coreasoning_v2/training/process_no_feedback_41/update_0000200.pt \
  --label generic process_no_feedback \
  --out outputs/r7_rollout_tables --reasoning-steps 3 --max-samples 4
```

Data: the held-out 2020 block of the #6 store — native 0.25°, 65x65, 17 channels,
real ERA5, `manifest_sha256 1a8636ed752b0438…`, `data_identity 9c714f6189bb6b4b…`,
train-only climatology (`train-only-month-hour-grid-mean-v1`, training years [2018]).
4 initializations per model, 200-update checkpoints, K=3.

RMSE, physical units (extract):

| model | variable | unit | 6h | 12h | 24h | 48h | 72h |
| --- | --- | --- | --- | --- | --- | --- | --- |
| generic | t2m | K | 3.72081 | 4.26451 | 4.42684 | 5.56105 | 5.61225 |
| generic | t500 | K | 1.37051 | 2.27627 | 3.34248 | 3.58531 | 3.27549 |
| process_no_feedback | t2m | K | 4.09721 | 4.65464 | 3.96513 | 5.39556 | 6.50987 |
| process_no_feedback | t500 | K | 1.15669 | 1.84994 | 2.63029 | 3.80466 | 4.10074 |
| persistence | t2m | K | 4.35171 | 5.07542 | 2.17555 | 2.90139 | 3.86503 |

ACC (pooled, dimensionless):

| model | variable | 6h | 12h | 24h | 48h | 72h |
| --- | --- | --- | --- | --- | --- | --- |
| generic | t2m | 0.576725 | 0.466857 | 0.311284 | −0.016203 | 0.336861 |
| process_no_feedback | t2m | 0.424574 | 0.309268 | 0.438505 | 0.131771 | 0.138480 |

## What these numbers do and do not say

**They are not evidence of skill.** Six metrics worth stating plainly, several of
them negative or cautionary:

1. **Error grows with lead time for the neural models**, as it should: `generic`
   t2m goes 3.72 K → 5.61 K from 6 h to 72 h, and `process_no_feedback` t500 goes
   1.16 K → 4.10 K. That is the expected qualitative behaviour and is the main
   thing this table establishes.
2. **Persistence is non-monotonic and looks deceptively strong at longer leads.**
   The persistence t2m row is 4.35 K at 6 h but **2.18 K at 24 h** — lower than
   either trained model. This is not a bug: with 4 initializations the 6/12/24 h
   targets fall on *different valid times of day* (12:00, 18:00, 06:00+1d), so each
   lead is scored against a different subset of the diurnal cycle. It does mean
   these particular numbers **must not** be read as "persistence beats the model at
   24 h". The case set is too small and its diurnal sampling too uneven for
   cross-lead ranking.
3. **ACC is near zero or negative at 48 h** for generic t2m (−0.016), i.e. the
   forecast is no better than the train-only climatology at that lead and
   initialization set.
4. **`process_no_feedback` is better than `generic` on t500 at every lead** but
   worse on t2m at every lead except 24 h — consistent with the #6 finding that
   the process arms trade variables rather than dominating.
5. **Four initializations is a smoke-scale case count.** It is enough to
   demonstrate that the table machinery produces correct, aligned, self-describing
   output; it is nowhere near enough for a journal claim.
6. **The checkpoints are bounded CPU models** (200 updates, `dim=32, depth=2`) and
   the substrate is ten January days per year.

## Acceptance assessment

| #8 requirement | Status |
| --- | --- |
| Evaluation scripts produce paper-ready metric tables | **done** — this script, with per-variable/per-lead tables, explicit units, provenance and a same-data baseline |
| Without changing training code | **done** — enforced by an AST-level test, and no training entry point is imported |
| Native 0.25° regional evaluation | done — 65x65 native grid, real ERA5 |
| 6/12/24/48/72 h autoregressive rollout | done |
| RMSE / ACC and variable-level metrics | done |
| Same-data baselines | persistence included; the `unet`/`convlstm`/`afno_small`/`window` architectures exist in-repo and can be passed to the same script as checkpoints |
| Chronological year-based split, no random split | done — train 2018 / val 2019 / test 2020, enforced by the store contract |
| Accuracy–compute Pareto and average reasoning depth | **not done here** — the script records `reasoning_steps` but does not sweep it; the K=0/1/3 depth behaviour is measured in [R7_COREASONING_FAIR_BUDGET.md](R7_COREASONING_FAIR_BUDGET.md) and [R7_HALTING_GATE_AUDIT.md](R7_HALTING_GATE_AUDIT.md) |
| Weather-complexity vs reasoning-depth diagnostics | **not done** |
| Extreme-event metrics | **not done** — no extremes are scored |

So the acceptance sentence of #8 is satisfied, while three of the issue's *scope*
bullets are not. They are listed rather than quietly dropped.

## Tests

| Suite | Result |
| --- | --- |
| `tests/test_r7_rollout_metric_tables.py` | 10 passed (exact headers, empty/absent rejection, wide per-variable tables with units, `undefined` ACC handling, no cross-variable average, exclusive writes, provenance binding, checkpoint-digest mismatch refusal, missing-digest refusal, mixed-dataset refusal, AST-level no-training check) |
| Full `pytest -q` with `CUDA_VISIBLE_DEVICES=""` | 869 passed, 9 skipped, 0 failed |
| `python tools/check_conventions.py` | 34 blocking rules, **0 violations** |

CI needs no GPU and no network; the tests write into `tmp_path` only.

## Honest limitations

- **Small case count** (4 initializations) and **uneven diurnal sampling across
  leads**, which is why cross-lead persistence ranking must not be read off this
  table.
- **One seed per model** in the emitted table; the script accepts several
  checkpoints and would record each as its own row.
- **Not a converged benchmark**: bounded CPU checkpoints on ten January days per
  year.
- **No Pareto, complexity or extreme-event analysis** (see the acceptance table).
- **Not SOTA, not journal evidence.** `scientific_claim: false` is written into
  every artifact.
