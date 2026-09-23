# Continuous temporal sample under the existing source cap (#55)

The previous four-season sample retained only384timestamps from time chunks
spanning8736hours. The new named profile `continuous-250d` retains1000six-hour
times starting Jan1 per year2018/2019/2020 (250days;3000times total).
The source tile, variables, snapshot and192MiB decoded-read cap do not change.
Before weather values are read, the exact touched source time-chunk set must
equal the old four-season plan for EACH variable/level.

End timestamps are2018-09-07T18,2019-09-07T18 and2020-09-06T18 because2020is leap.
The old January/four-season modes, their source pins and payloads remain unchanged.
This is NOT three full years, complete seasons, a global mirror or0.1-degree truth.
Spatial selection stays12x12 at the provider0.25-degree grid without interpolation.

The cropped float32 state is19,008,000bytes. A32MiB source-file publication cap,
8MiB perchunk/pervariable output guard and0.05GiB local raw-state estimate apply.
Decoded chunk accounting is not a network-byte, peak-RAM or GPU measurement.

The integration workflow expects exactly998one-step windows per split,
computed from1000time points with two histories and one target. No year gaps are
bridged. All eight input-state process proxies use physical fields and train-only
normalization. Only20Process CPU updates and24balanced VALIDATION cases are run.
No controller fitting, threshold selection or test evaluation occurs.

    python scripts/real_r7_continuous_pilot.py --out outputs/new_continuous_pilot

A reusable NetCDF and complete receipts/code/environment/checkpoint/report are
retained as artifacts. Larger-data accuracy comparisons require separately
predeclared experiments; this integration cannot certify superiority or SOTA.
