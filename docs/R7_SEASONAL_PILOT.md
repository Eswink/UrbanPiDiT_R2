# Four-season public CPU pilot (#49)

The source snapshot, tile, eleven channels, exact0.25-degree grid and decoded
192MiB cap are unchanged. A named four-season profile selects Jan1-8, Apr1-8,
Jul1-8 and Sep1-8 in2018/2019/2020, every6h:384timestamps,128peryear.
These are excerpts, not three complete years or complete seasons. September
represents autumn; its dates remain in the same source time chunks as the
January pilot. October would touch an extra chunk. This cost decision was
recorded before retrieving weather fields or viewing model performance.

The guarded extractor supports january (unchanged default) and four-season only.
Coordinates, units, pressure selection, missing sentinels and the total read
plan are checked before field access. Decoded accounting is not HTTP bytes/RAM.

After preprocessing, each year has120single-step windows. Timestamp-only
selection creates additional val_balanced/test_balanced manifests alongside
unmodified originals: six complete72h cases per sampled month. This avoids the
ordinary first-N evaluation cap accidentally showing January alone. There is
no interpolation, gap bridging, future forcing or error-driven case selection.
The small demonstration trains a process model for20CPU updates and evaluates
persistence/ProcessK1/ProcessK3 on the same24cases, with month-level RMSE retained.
No skill, convergence or significance claim is attached to this demonstration.

```bash
python scripts/real_r7_seasonal_pilot.py --out outputs/new_seasonal_pilot
```

This explicit command DOES access the anonymous pinned public source. Use a new
output directory. The corresponding workflow is capped at20minutes. No GPU,
paid subscription or scheduled automation is used.
