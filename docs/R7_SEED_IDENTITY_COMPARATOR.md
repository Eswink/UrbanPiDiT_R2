# R7 #60 — seed-identity-keyed multi-seed comparator (S0-A)

Status: **DONE (engineering/acceptance chain; no new science claim)**.
Verification binds to commit recorded in `docs/R7_TASK_QUEUE.md` and
`docs/R7_ISSUE_COMMENTS.md`. `scientific_claim: false` throughout.

## What was wrong (issue #60)

`training/r7_coreasoning_compare.py` dropped seed ids in `summarize` and zipped
per-seed values by list position in `compare`: reordering the input records
turned `direction=improved` into `unresolved` with identical means; missing
seeds were silently truncated; the join key lacked `unit`; missing reference
rows were skipped; equal `n_initializations` was the only (implicit) case-set
argument; `gate_met` was computed from descriptive sign-consistency alone, so a
single improved variable could "pass" while most variables were unresolved.

## What changed

- Join is now by the full `arm/depth/variable/unit/lead/seed` key. Per-seed
  values are kept under explicit seed ids (`seed_rmse`), deltas are paired by
  seed id (`seed_deltas` entries carry `seed`), and all aggregation iterates
  sorted keys, so **any permutation of the input records produces identical
  output** (tested bit-exactly, including on the real historical artifact).
- Fail closed on: duplicate seed for a key, seed-set mismatch, unit conflict
  for one (variable, lead) across arms, case-set mismatch (counts differ, or
  **exact initialization lists differ despite equal counts** via
  `provenance.json` digests), non-finite/negative RMSE, duplicate table
  entries, malformed rows, missing baseline rows, and records that disagree on
  dataset identity / model code / update budget.
- Two statistics are named separately and never mixed:
  `rmse_seed_mean` (mean over seeds of each seed's cross-case pooled RMSE) and
  `rmse_pooled_cases` (pooled over every case of every seed, from per-case MSE).
  Each seed's `rmse.csv` value is cross-checked against
  `sqrt(mean(per-case MSE))` from `provenance.json` (rel 1e-9); a mismatch is
  data corruption and fails closed.
- `sign_consistent`/win-loss counts stay descriptive. `gate_met` was **removed**
  from comparison blocks; `evaluate_gate(comparison, criteria)` reads a
  pre-frozen criteria dict (required variables/units/leads, required direction,
  unresolved allowance) and fails on absent/blocked/worsened/unresolved-beyond-
  allowance required variables. `run_comparison` writes
  `research_gate: "not_evaluated; run evaluate_gate with pre-frozen criteria"`.
- `reaggregate_historical(result, out)`: re-reads the original per-evaluation
  artifacts, rebuilds the table with the new comparator, writes a **new**
  write-once audit JSON with old-vs-new diff; originals untouched; unreadable
  artifacts become blocked records — no seed/case ids are invented. CLI:
  `python training/r7_coreasoning_compare.py --reaggregate <result.json> --out <audit.json>`.
- `run_comparison` records now embed `{dataset_identity, model_code_sha256,
  optimizer_updates}` per record; result format bumped to
  `r7-coreasoning-fair-budget-result-v2`. No forecast weights touched, no
  retraining, no new data.

## Verification (local, CPU-only)

- `pytest tests/test_r7_coreasoning_compare.py -q` → **26 passed**
  (was 12; 14 new tests cover the #60 acceptance list: positional-vs-seed
  pairing repro, permutation invariance, missing/duplicate seed, unit
  conflict, equal-counts-different-inits, NaN/Inf/negative, duplicate entries,
  missing baseline row, identity mismatch, gate criteria semantics incl. the
  "one improved cannot carry many unresolved" case, tampered-rmse-vs-MSE
  cross-check, re-aggregation audit + blocked-without-fabrication).
- Full suite: **909 passed / 3 skipped** (3 = unchanged optional real-data
  fixtures; baseline was 895/3 before this round).
- `python tools/check_conventions.py` → **34 blocking rules, 0 violations**.
- Shuffle invariance re-verified on the real historical artifact
  (27 records, 153 table rows): `summarize(shuffled) == summarize(original)`.

## Historical re-aggregation audit (originals untouched)

Source: `outputs/r7_coreasoning/coreasoning_result.json`
(sha256 `630cfa8d4c60458afd7a39dad9665e559e65ad79e57f0d9cf802e193bcbf43bb`,
the #6 fair-budget artifact; 27 records = 3 arms × 3 seeds × 3 depths).

Audit output (untracked, in `outputs/`):
`coreasoning_reaggregated_audit.json`
(sha256 `2a796e2ecd51328d8f80c3977f625b6a55336094c80932aeba079afc633e38c1`).

- Records re-aggregated: 27/27; **blocked: 0** (every `rmse.csv` +
  `provenance.json` was readable, so nothing needed fabricating).
- All 153 seed means reproduce the original per-order report **exactly**
  (max |mean_difference| = 0.0): the historical records were written in
  ascending seed order, so the old positional zip happened to coincide with
  seed identity. This is now *verified* rather than assumed.
- 0/34 comparison directions changed at the deepest ablation (K3).
- Pooled-vs-per-case-MSE cross-check passed for all 153 keys × 3 seeds.
- Descriptive outcome at K3 (unchanged, and **not** a gate pass):
  `process_feedback` vs `generic`: 10 improved / 7 worsened by mean, only
  2 sign-consistent (1 improved, 1 worsened), 15 unresolved,
  `beats_baseline_everywhere: false`, case identity exact. Under the new gate
  function this arm cannot claim the #6 gate; any future gate claim must
  freeze criteria *before* the experiment.

## Limits

- The audit cannot retroactively prove what the *original* ordering did for
  artifacts that no longer exist; for this artifact it does (blocked=0).
- Case-set identity relies on `provenance.json` being present; evaluations
  without it stay `count-only` and never claim exact case identity.
- `rmse_pooled_cases` requires per-case MSE; absent provenance leaves it None.
- No GPU run, no new training, no data acquisition in this item (CPU-only fix).
