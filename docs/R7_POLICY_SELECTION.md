# Validation-only halting policy selection (#36)

This extends #31's train-only controller fitting. It does not retrain or mutate
forecast weights and does not perform any automatic cloud job, rental or download.

```bash
python tune_r7_halting.py --checkpoint outputs/process/update_0000010.pt --controller outputs/controller/controller.pt --manifest data/manifests/r7/val.jsonl --out outputs/policy --gain-thresholds 0 0.001 --probability-thresholds 0.3 0.5 0.7 --leads 6 12 24 48 72 --max-samples 32 --relative-rmse-tolerance 0.01
python evaluate_r7_local.py --checkpoint outputs/process/update_0000010.pt --controller outputs/controller/controller.pt --manifest data/manifests/r7/test.jsonl --out outputs/selected_test --policy-selection outputs/policy/selection.json --leads 6 12 24 48 72 --max-samples 32
```

Choose the threshold grid, horizons, sample cap and tolerance before inspecting
validation/test results. The 32-case cap is for engineering, not paper acceptance.
The search refuses training or test manifests. It runs one fixed-Kmax reference
and every explicit candidate (maximum 64) on identical validation initializations,
using real free-running trajectories rather than selecting saved full-K drafts.

The criterion is explicit: every variable/horizon RMSE must be at most
reference_RMSE * (1 + tolerance). Among feasible candidates, minimize mean
cumulative reasoning steps at the largest horizon. Ties use a deterministic
policy digest; reference fixed depth is always included as fallback. Zero
reference error requires zero candidate error. There is no mixed-unit scalar
accuracy score and no conversion of step counts into wall-clock speedup.

The selected artifact binds the parent checkpoint, controller checkpoint,
training data identity, validation manifest, candidate report content hashes,
channels, units, forecast horizons and transition cadence. Test evaluation
loads this frozen artifact; different identities/metric definitions and manual
depth/threshold overrides fail. Evaluation records effective policy thresholds
and the selection file hash. Files/directories are never overwritten.

Artifact digests establish integrity, not trusted authorship or proof that the
raw weather data were never changed. Source provenance and immutable data remain
required. A validation winner is not a held-out skill result. The greedy gain
controller still needs look-ahead and consistency controls, and real multi-seed
ERA5 plus synchronized hardware measurements under parent #7. No SOTA claim.

Engineering tests cover paired case identity, adverse single-variable behavior,
pooled MSE, deterministic ties, zero-error fallback, frozen identity checks and
a train -> controller -> validation selection -> test CLI fixture. No real
multi-year calibration run is performed by adding this code.
