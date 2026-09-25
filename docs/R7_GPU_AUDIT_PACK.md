# R7 #62 — GPU audit pack, table re-derivation, and result-text corrections

Status: **DONE** for every deliverable the issue lists; one previously missing
measurement (`retained_truncated` memory cells) was added under the frozen
budget. `scientific_claim: false` throughout. No forecast weights were
retrained and no historical value was altered.

## 1. Doc corrections (numbers untouched, correction logs appended)

- `docs/R7_ROLLOUT_TABLES.md`: the "better at every lead" claim contradicted
  its own table (t500 48/72 h and t2m 48 h). Rewritten per-lead; correction
  log appended.
- `docs/R7_COREASONING_FAIR_BUDGET.md`: the `v250` worsening was attributed to
  the wrong arm (the table puts it in `process_no_feedback`, not
  `process_feedback`); a `spatial_solver_feedback` finding was conflated with
  this run's `use_forecast_feedback` arms — two different switches, so the
  "does not reproduce" sentence is gone; "established" downgraded to
  descriptive sign stability; the old `gate_met` prose updated for the #60
  gate refactor. Correction log appended.
- `docs/R7_GPU_BRINGUP.md`: "8.6 (sm_82, 82 SMs)" separated into compute
  capability 8.6 = architecture `sm_86` vs the 82-SM hardware count; timing
  sections now say explicitly they are short microbenchmarks (1 warmup +
  2–3 measured steps) that must not be extrapolated to long-training
  throughput; correction log with the new `retained_truncated` cells.
- `docs/R7_TASK_QUEUE.md`: stale "#8 close pending the authorized main write"
  replaced (the issue closed with the 2026-09-25 main fast-forward); #62
  note added.

## 2. ACC self-correction (not a bug) and the explicit climatology baseline

`training/r7_acc.py` computes the pooled **uncentered** anomaly correlation
against one frozen climatology; on the same cases and weights this satisfies a
one-sided identity with MSE skill:

- `ACC < 0 ⟹ MSE skill < 0` (the forecast is really worse than that
  climatology on those cases),
- `ACC > 0` implies **nothing** about the skill sign (a constructed
  counterexample is pinned in the tests: ACC ≈ +0.4 with skill < 0).

New module `training/r7_climatology_skill.py`:
`RolloutClimatologySkillAccumulator` accumulates latitude-weighted MSE of the
forecast **and of the climatology itself**, reporting `rmse_forecast`,
`rmse_climatology` (the explicit baseline) and `mse_skill` per lead/variable;
`verify_acc_skill_consistency` checks the identity. Tests
(`tests/test_r7_climatology_skill.py`, 5 cases) verify the accumulator against
a direct computation, the implication across 40 randomized trials (negative-ACC
cases constructed via an anti-correlated forecast anomaly), the
positive-ACC/negative-skill counterexample, NaN (undefined) skill at zero
climatology energy, and the field validations. Premises recorded in the module
docstring: one frozen climatology, equal per-initialization weighting,
same-cases-only statements, no averaging with ACC or across variables.

## 3. The audit pack and pack-only table re-derivation

- `scripts/build_r7_gpu_audit_pack.py` collects the GPU artifacts that live
  only in the untracked `outputs/` tree into a write-once pack:
  per-cell `sweep.json` + logs, protocols, multi-seed study cells, the D3/D4
  checkpoint/verification records and the failure/skip cells. Every copied
  file is SHA256-hashed in `MANIFEST.json` (its own hash in a
  `MANIFEST.sha256` sidecar); binaries are excluded and listed; content that
  looks like a credential is refused; `git_commit` and `model_code_sha256`
  bind the code identity; the protocol block carries the environment facts
  (GPU, torch/cuda, dtype, warmup/measure counts).
- `scripts/rebuild_r7_gpu_tables.py` reads **only the pack**: it re-aggregates
  the per-cell sufficient statistics into memory/timing tables and the
  multi-seed cell index, and validates each row's metadata contract
  (dtype/shape/mode/K/batch) — cells with different contracts form separate
  rows instead of being averaged, and non-`ok` cells are flagged as problems,
  never silently dropped into a mean.
- Real run: pack = `outputs/r7_gpu_audit_pack/` (562 files, 8 binaries
  excluded, 0 credential hits, 30 failed/skipped cells listed);
  tables = `outputs/r7_gpu_tables_rebuilt/` (104 rows, 25 multi-seed rows,
  30 metadata problems — every one a genuine `status=oom/skip` probe cell).
  Tests: `tests/test_r7_gpu_audit_pack.py` (4 cases: pack build + exclusion +
  hash verification + write-once, pack-only rebuild + problem flagging,
  manifest-less rejection, contract-conflict isolation).

## 4. The missing `retained_truncated` memory measurement (added)

`scripts/bench_r7_gpu_memory.py` gained the `retained_truncated` mode
(detach between reasoning steps; routed through the full forward/backward/
optimizer split, labeled separately). 12 fresh-process cells measured
2026-09-26 (same scale-B contract, bf16, seed 7, 1 warmup + 2 measured steps):
allocated memory is flat in K (generic/process ≈ 366–373 MiB at K=1/4/8) while
step time grows (generic 56.7→92.6 ms, ckpt off) — filling the pure-memory
comparison: full BPTT grows with K, both truncated modes stay flat, three
distinct semantics with three separate labels. Raw cells in
`outputs/gpu_sweep_retained/`; included in the audit pack and re-derived in
`outputs/r7_gpu_tables_rebuilt/`.

## 5. BLOCKED / not done

- The 17 tag-gated historical experiments were **not** re-run (the issue
  explicitly does not require it); all evidence comes from existing local
  artifacts.
- The pack is a local directory under `outputs/` (untracked by design).
  Publishing it for download needs either the GitHub connector or a user
  upload — an explicit user decision, not silently made here.
- The GPU-memory artifacts record single-seed short microbenchmarks; the
  multi-seed study (`r7_multiseed_real_v3`) remains the 3-seed evidence and
  its `cells.jsonl` is re-derived as descriptive seed coverage, not as
  significance.
