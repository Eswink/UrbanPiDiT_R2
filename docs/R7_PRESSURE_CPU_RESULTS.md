# Actual public ERA5 pressure / CPU pilot (#44)

## Verified acquisition and executed experiment

Implementation: `4cee1f853b3f3b4e016f3d2a4ebce88024a01af9`.
Actual workflow: https://github.com/Eswink/UrbanPiDiT_R2/actions/runs/35875707823 .
Artifact `10757057891` contains the real subset, receipts, checkpoints, train/val
metadata, complete per-variable/horizon RMSE/ACC and policy-selection evidence.
The artifact was downloaded; its ZIP digest and all 11 field payload digests were
independently checked against the source receipt. This is reanalysis, not direct
station observations, and no spatial interpolation was used in the extraction.

- Archive: 3,026,772 bytes, SHA256 `bca9e915630ebb6f73df33f5a0d0c569915a1b5d44c174d73ca49a0615d8ff66`.
- Raw NetCDF: 629,696 bytes, SHA256 `13fb72807d3f6988b0fcf2b6b2f130f890207242916018556fb85553686b9a12`.
- Native provider grid: 12x12, 39.25–42 N / 114–116.75 E, spacing 0.25 degrees.
- Jan 1 00UTC–Jan 8 18UTC in each of 2018/2019/2020, sampled every six hours: 96 times total.
- Channels: t2m/u10/v10/mslp/t850/t500/q850/u850/u500/v850/v500.
- Train/val/test years: 2018/2019/2020, 30 one-step windows each; train-only normalization.
- All eight physical-input process diagnostic proxies generated; normalized training labels [30,8], finite.
- Native/generic/process models: 20 CPU optimizer updates each, one seed42, batch2,
  dim32/depth2; recurrent K3, 16 latent tokens or 8 anchored+8 free tokens.
- Controller: 8 train-only updates; validation search four initializations; test
  six identical initializations with free-running 6/12/24/72h forecasts.
- Workflow CPU only (two threads); no paid GPU or GPU execution.

The logical decoded-chunk accounting was 180,142,968 bytes under a 192 MiB cap.
This is NOT instrumented HTTP transfer or measured peak RAM. Source extraction
reported 11.946 s; the complete acquisition + preprocessing + CPU experiment
reported 22.883 s, excluding dependency installation/workflow startup. These
values are neither isolated model latency nor portable runtime guarantees.

## Selected +6h RMSE (full tables are retained, no unfavorable variables discarded)

| Model | T2m K | U10 m/s | V10 m/s | MSLP Pa | T500 K |
|---|---:|---:|---:|---:|---:|
| Persistence | 6.762938 | 0.887299 | 0.997581 | 279.449449 | 0.895535 |
| Native | 6.724022 | 0.884622 | 0.991296 | 277.133520 | 0.886622 |
| Generic K3 | 6.614137 | 0.877075 | 0.975451 | 270.209694 | 0.879062 |
| Process K1 | 6.704476 | 0.882520 | 0.982957 | 277.310017 | 0.897233 |
| Process K3 | 6.652258 | 0.878922 | 0.968829 | 276.156664 | 0.899293 |

The adaptive selector chose the safe **force_full_depth=True, K=3** fallback;
adaptive forecasts equal Process K3 here. No compute saving has been established.
Process K3 does not uniformly beat generic recursion or K1: T500 and humidity
show counterexamples. The data are a tiny January-only tile; six correlated cases,
one seed and 20 updates cannot support convergence, significance, regional skill
or SOTA claims. The scientific parent issues remain open.

## Reuse without another source download (#46)

After downloading/unzipping the experiment archive, use its original source pair:

```bash
pip install -r requirements.txt -r requirements-r7-data.txt
python scripts/real_r7_pressure_pilot.py --out outputs/my_offline_pilot \
  --source /path/to/unzipped/source/era5_pressure_pilot.nc \
  --receipt /path/to/unzipped/source/receipt.json
```

No Icechunk/pcodec is required in this mode. It pins the exact successful raw
file hash, verifies every variable payload plus time/space/units/level metadata,
then rebuilds the cache and executes the same bounded CPU protocol. Output must
be a new directory. Original checkpoints are archived evidence; their recorded
manifest/data identities are path-sensitive, so this command rebuilds and trains
locally rather than silently loading relocated checkpoints against another cache.
An exact file hash binds this artifact; self-written receipts do not authenticate
arbitrary weather. New source snapshots need a separate acquisition audit.
