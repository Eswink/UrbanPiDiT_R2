# Controlled seasonal CPU experiment (#50)

This protocol was posted before analyzing seasonal model outcomes. It reuses
#49's actual source without cloud extraction. The exact raw NetCDF pin is
3b5d7df8973a46d522508a3b483a52394a07c0f9d2db6e07894adb1bcb41e5bb.
The original January verifier/pin remains separate; receipts cannot substitute
arbitrary self-authored weather for either verified artifact.

Forecast settings remain the #47 protocol: seeds41/42/43, native/generic/process/
process-no-aux, dim32/depth2, same batch2/LR2e-4/sample order, K3, endpoints20/200
with exact optimizer resume. Validation at both endpoints, test only at fixed200.
One-step training windows are120 per year; held-out evaluation uses24fixedcases,
six per Jan/Apr/Jul/Sep. These are eight-day excerpts, not complete seasons/years.
Same-checkpoint ProcessK1 is an inference-depth ablation, not independently trained.

Each final process checkpoint is frozen for120train-only controller updates with
batch2, hidden64, Kmax3, LR1e-3 and the corresponding seed. Validation-only search:
gain threshold0, probability0.3/0.5/0.7, 1% tolerance separately per variable and
horizon, plus full-depth fallback. Selected policy is frozen before its test run.
Test failures and full-depth fallbacks are valid outcomes, never triggers to retune.

All39forecast reports, complete seed summaries and per-month views are retained.
Monthly statistics reuse the SAME trajectories rather than retraining/selecting
models per month. Controller reports preserve actual reasoning counts and every
held-out error. Count savings do not imply wall-clock savings or matched FLOPs.
Seed SD describes initialization variation, not significance over independent events.

```bash
python scripts/study_r7_seasonal_cpu.py --source <era5_four_season.nc> --receipt <receipt.json> --out outputs/new_seasonal_study
```

Output must be new. No Icechunk or source networking is required. The GitHub CPU
workflow retrieves the prior artifact first, then prohibits outbound sockets in
the experiment stage. No GPU, paid rental, scheduled task or main merge is used.
The20minute workflow cap bounds the job, not a convergence guarantee. A larger
paper benchmark still requires a separate dataset/protocol and untouched test.
