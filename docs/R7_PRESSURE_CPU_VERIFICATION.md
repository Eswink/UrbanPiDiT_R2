# Verification receipt: public real pressure CPU experiment

This receipt supplements R7_PRESSURE_CPU_RESULTS.md; numeric forecast tables there
come from the original actual run, not generated target outcomes.

## Original live acquisition and CPU execution

- Commit: 4cee1f853b3f3b4e016f3d2a4ebce88024a01af9.
- Run: https://github.com/Eswink/UrbanPiDiT_R2/actions/runs/35875707823 — success.
- Artifact10757057891, ZIP SHA256 bca9e915630ebb6f73df33f5a0d0c569915a1b5d44c174d73ca49a0615d8ff66.
- Independent downloaded-file verification checked that ZIP hash, original NetCDF
  hash, every one of the eleven float32 field hashes, shapes, finite values and
  geographic coordinate arrays matched the acquisition receipt.
- NetCDF SHA256: 13fb72807d3f6988b0fcf2b6b2f130f890207242916018556fb85553686b9a12.

## Test failure and correction

Initial full CI failed one deadline test because started=0 does not imply an
expired budget on a freshly booted CI runner. a5fe3e41ad2c07e0acccac0c2f59e68c59b4eec2
uses an explicit clock and asserts no source reads occur after expiry. This is
a test-fixture correction; no extraction checks or measured results were relaxed.
Full CI35876544913 passed afterward, with original acquisition/training code unchanged.

## Offline replay and final code verification

- Commit: 4ba62b1c603bcf8c5dda888dce4afe3bf8e24efe.
- Replay: https://github.com/Eswink/UrbanPiDiT_R2/actions/runs/35877069975 — success.
- It downloaded the existing GitHub artifact before the execution stage, then
  prohibited outbound socket connections while verifying original bytes, rebuilding
  Zarr, recomputing process targets, training/calibrating/selecting and evaluating.
- No Icechunk client or source-cloud access was needed during the replay stage.
- Replay artifact10759475690 downloaded and independently ZIP-hash verified:
  e13b6f9de0bcb1648c516663a4abba80a6481bc87689734fc464e54f46dd7e34.
- Actual replay report: CPU only, source SHA unchanged, 11.901s for the execution
  chain excluding installation and artifact retrieval. Full-depth K3 policy repeated.
- Small floating-point RMSE differences across CPU runs were observed (maximum
  absolute entry difference about 2.79e-4 across stored physical metrics); do not
  claim bitwise-identical cross-hardware model training or compare that mixed-unit
  maximum as a forecast-quality metric.
- Full code CI: https://github.com/Eswink/UrbanPiDiT_R2/actions/runs/35877070004
  job107235638164: **359 passed, 3 skipped, 2 warnings in 76.08s**.
- Three skips require untracked old Beijing/UCI local fixtures; the new tests ran.
  Two warnings are existing Lightning unit-test logging-without-Trainer warnings.

Neither test count nor genuine source identity proves SOTA, convergence, general
forecast skill, causal reasoning, GPU speedup or 24GB feasibility. Those remain
separate research/hardware gates. The completed outcome is reusable true pressure
and surface data plus an exercised CPU training/evaluation chain.
