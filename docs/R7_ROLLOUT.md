# R7 free-running rollout and RMSE primitives

Engineering child #22 of evaluation parent #8. These utilities enable evaluation;
they do not contain or certify real-weather benchmark results.

## Forecast trajectory

```python
from model.r7_rollout import autoregressive_rollout

model.eval()
trajectory = autoregressive_rollout(
    model, initial_batch,
    lead_hours=(6, 12, 24, 48, 72),
    step_hours=6, history_interval_hours=6,
    inference_kwargs={"reasoning_steps": 4},  # generic/process variants
)
```

Native baseline: omit inference_kwargs. Adaptive wrapper: use, for example,
`{"max_steps": 4}` with a genuinely calibrated controller, or
`{"max_steps": 4, "force_full_depth": True}` for the fixed-depth equivalence control.
An untrained controller is not silently enabled.

Every model call receives ONLY the evolving coarse_history and the single-step
lead_time_hours=6. A +72h trajectory uses twelve +6h transitions, not a +72h
condition at every transition. Next history is formed from previous history and
the model's own output. Future weather targets, process labels and external
baselines are not forwarded. No reanalysis-derived future boundary forcing.

This initial implementation requires equal history and transition cadences and
all input dynamic channels to be predicted. It rejects partial states rather
than filling missing future fields from observations. The caller must select
properly normalized, complete dynamic channels; accumulated precipitation needs
its own interval semantics before entering this recurrent state.

Outputs are requested-horizon snapshots [B,L,C,H,W], explicit horizon values,
number of model calls and cumulative per-sample reasoning steps. Copies protect
earlier snapshots against models which reuse output buffers. Reasoning counts
are NOT elapsed-time, FLOP or speedup measurements. Returned forecast storage
scales with the number of requested horizons.

## Metrics

```python
from training.r7_rollout_metrics import RolloutRMSEAccumulator

metric = RolloutRMSEAccumulator(
    trajectory.lead_hours, channel_names,
    training_std=frozen_training_std, units=channel_units,
)
metric.update(trajectory.forecasts, aligned_targets, latitude)
# Repeat update for each batch; do not average batch RMSE values.
metric.write_csv("outputs/evaluation/rmse.csv")
```

Each initialization has equal weight. Within one field the metric averages
longitude and uses cos(latitude) area weights. It accumulates squared error over
initializations before taking a square root, separately for each lead/channel.
The difference of identically normalized forecast and target is multiplied by
frozen training standard deviation to recover physical units; the shared mean
cancels. Without standard deviations the CSV explicitly says `normalized`.
Physical units are required when restoring normalization. There is no overall
average of temperature, pressure and wind errors in incompatible units.

Nonfinite data are rejected rather than silently removed. CSV writes refuse to
overwrite an existing file by default. A production evaluation must additionally
record checkpoint/data hashes, channel ordering, timestamp alignment, held-out
years and normalization provenance; this primitive cannot verify those from
plain tensors. It does not silently mask missing observations.

## Remaining parent acceptance

#8 still requires real ERA5 rollout window construction, timestamp alignment,
frozen train-only climatology and ACC, boundary-context sensitivity, comparisons
against strong baselines, extreme-event metrics and measured latency/memory.
The CI integration tests run all four R7 variants to +72h on tiny synthetic
inputs only. This does not establish stable or accurate real-weather skill.
