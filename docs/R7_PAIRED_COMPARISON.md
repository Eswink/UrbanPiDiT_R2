# Paired result comparison (#32)

```bash
python scripts/compare_r7_results.py --a outputs/model_a/provenance.json --b outputs/model_b/provenance.json --out outputs/comparison.json --block-days 7 --replicates 1000
```

Only compare outputs with the same manifest, channel order, units, horizons and
exact initialization/valid-time set. Different row ordering is harmless; missing
cases are an error, not silently intersected. This prevents a model from being
scored only on an easier subset. Inputs/results are hashed and output is exclusive.

For each variable/horizon, the point estimate is sqrt(mean(MSE_A)) minus
sqrt(mean(MSE_B)); no averaging of per-case RMSE and no mixing physical units.
Bootstrap resampling uses the same sampled calendar blocks for both models and
keeps cases within each block together. Blocks use UTC-naive initialization dates
and a fixed calendar origin. Report the configured block width, block count and
seed. The percentile interval is not an automatic hypothesis test or SOTA verdict.

Temporal dependence can extend beyond one block. Choose block width based on
validation data/physical timescales, investigate sensitivity, and separately
address few-block reliability, multiple testing and model-selection bias. Do not
select block width or discard cases because it creates a favorable test interval.
