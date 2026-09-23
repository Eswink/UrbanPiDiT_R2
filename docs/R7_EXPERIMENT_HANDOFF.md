# R7 handoff: from verified engineering to a real bounded pilot

This is not a release or a promise of SOTA. Work remains on
`r7/weather-reasoning`, Draft PR #12. No timer or unattended paid job is enabled.

## Evidence already available

The code supports native/generic/process forecasting, streamed truncated
optimization, gain-controller calibration, true active-subset inference,
free-running rollout, physical per-variable RMSE, training-only climatology/ACC,
paired comparisons, and the new evaluation controls #36-#39. The exact code
checkpoint `6ba682021d502523138e735d9798c1418c4abffe` passed 314 tests, with
3 explicitly skipped old local fixtures. Compilation and wheel checks passed.

#35 pins a genuine ERA5 t2m fixture: 315 scalar values over 9 times and a 5x7 ROI,
with source/artifact/data checksums. Its offline end-to-end regression is useful
for reproducibility, but it cannot establish multivariate dynamics, seasonal
coverage, 72-hour weather skill or process-aware model superiority.

## Inputs needed before the next scientific run

Provide the location and access method of an existing ERA5 NetCDF/Zarr cache,
its time span and cadence, region/grid, variable and pressure-level inventory,
units and provenance. Do not send passwords or private keys in an issue/chat.
A path on the user's computer is not automatically mounted in this environment;
use an explicitly authorized computer connection or run the commands locally
and return the reports. A small metadata inventory is enough to design the pilot.

The user's 4090D must be available through an authorized execution path or local
returned logs. The current assistant container is CPU-only. CPU CI cannot measure
GPU memory, consumer-GPU attention backend behavior or real GPU throughput.

When no suitable cache exists, approve a bounded acquisition plan before action:
source/region/variables/time span, storage destination, maximum download volume,
compute time and monetary cap (zero rental is a valid constraint). No multi-year
download or rental has been launched by this engineering iteration.

## Resume sequence

1. Audit metadata/units, immutable cache identity and chronological splits using
   existing data tools. Freeze the pilot dataset and channel order before training.
   Match model input/output channels and anchored-process labels to the actual data.
2. On the user's own GPU, profile realistic shapes with K=1/2/4/8 in fresh
   processes. Stop on OOM/nonfinite failures; preserve failure logs, do not silently
   reduce resolution/channels and report it as the same experiment.
3. Run a bounded native/generic/process pilot under explicitly equal optimizer
   updates/data/precision budgets. Preserve configs, seeds, all checkpoints and
   failure records. A successful pilot is not a final journal comparison.
4. Freeze a process checkpoint. Fit the controller on training samples only.
   Select thresholds on validation with a predeclared tolerance/grid/case cap.
   Keep test data untouched until selection is frozen.
5. Evaluate the frozen candidates on identical held-out 6/12/24/48/72h cases;
   collect per-variable RMSE/ACC, paired uncertainty and full+boundary scores.
   Measure model-forward latency separately; do not interpret total evaluation
   runtime or reasoning-count ratios as GPU speedup.
6. Run retrospective delayed-gain diagnostics on validation. They intentionally
   use labels and are not a deployable baseline. Decide whether a look-ahead or
   consistency redesign is justified only after examining actual failures.

## Existing executable entry points

- Training/checkpoint/resume and shape profiling: [R7_LOCAL_RUNNER.md](R7_LOCAL_RUNNER.md).
- Local timestamp-exact held-out scoring: [R7_LOCAL_EVALUATION.md](R7_LOCAL_EVALUATION.md).
- Frozen-forecaster controller fitting: [R7_CALIBRATION_RUNNER.md](R7_CALIBRATION_RUNNER.md).
- Validation policy search: [R7_POLICY_SELECTION.md](R7_POLICY_SELECTION.md).
- Same-forecast edge/interior scoring: [R7_BOUNDARY_SCORES.md](R7_BOUNDARY_SCORES.md).
- Isolated inference timing: [R7_INFERENCE_PROFILE.md](R7_INFERENCE_PROFILE.md).
- Non-deployable delayed-gain oracle: [R7_DELAYED_GAIN.md](R7_DELAYED_GAIN.md).
- Paired result comparison: [R7_PAIRED_COMPARISON.md](R7_PAIRED_COMPARISON.md).

For a quick engineering-only local GPU check (not production acceptance):

```bash
python scripts/profile_r7_local.py --config configs/r7_local_smoke.yaml --out outputs/profile_smoke --device cuda --bf16 --steps 1 2 4 8 --updates 3
```

That bundled smoke config is deliberately small. Actual #20 acceptance requires
a separately frozen realistic configuration, not just the smoke command passing.

## What does not justify closing a research parent

Unit-test growth, compiling a wheel, tiny real fixtures, a single batch improving,
or an oracle picking a favorable depth do not prove forecasting skill. The
remaining blockers and exact conditions are in [R7_TASK_QUEUE.md](R7_TASK_QUEUE.md).
Do not manufacture new engineering scope to obscure these external dependencies.
When a real data/hardware dependency resolves, the next conversation resumes the
highest-priority blocked task rather than asking again whether to continue.
