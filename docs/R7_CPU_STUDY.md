# Bounded real-data CPU learning-budget study (#47)

Protocol is predeclared in the issue and written with a SHA256 before training.
The exact January #44 source file is reused; no source-cloud reads are required.
This follow-up tests optimization budget and initialization sensitivity, not SOTA.

- seeds: 41,42,43; dim32/depth2; same sample order per seed, batch2, LR2e-4.
- native, generic K3/16 latent tokens, process K3/8 anchored+8 free with auxiliary
  weight0.1, and identical process with auxiliary weight0.0.
- endpoints20 and200 on one resumed optimizer trajectory. Both endpoints get
  validation reports; only the predeclared final200 endpoint gets test reports.
- ProcessK1 is an inference-depth ablation of the SAME final trained process
  checkpoint, not a separately trained K1 model.
- lead6/12/24/72 hours, six identical chronological initializations in each split.
- report every model/seed/variable/horizon. CSV summaries average seed-level RMSE
  per variable/horizon and give sample SD across seeds. They are NOT confidence
  intervals over independent weather events. Missing or unpaired runs fail closed.
- persistence is deterministic and reported separately, not replicated as three
  fake training seeds. Same updates do not imply same FLOPs or runtime.
- test2020 has already been inspected in the earlier smoke: this remains exploratory;
  the eventual paper requires a separately frozen protocol and untouched evaluation.

```bash
python scripts/study_r7_cpu.py --source <original.nc> --receipt <receipt.json> --out outputs/cpu_study
```

Output must be new. Maximum200 updates per model/seed and at most3seeds are
validated by StudyPlan. Workflow runtime is capped at20minutes and execution is
CPU-only. All failed or unfavorable results are preserved, never replaced with
synthetic targets or favorable trials. GPU acceptance remains deferred.
