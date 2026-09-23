# Timestamp-aligned local evaluation (#26)

```bash
python evaluate_r7_local.py --manifest data/manifests/r7/test.jsonl --checkpoint outputs/run/update_0000010.pt --out outputs/eval --max-samples 32
python evaluate_r7_local.py --manifest data/manifests/r7/test.jsonl --persistence --out outputs/persistence --max-samples 32
```

These commands read existing completed local Zarr caches only. They do not
fetch ERA5 or start an experiment automatically. Both output directories must
be new. Default leads are 6/12/24/48/72 h. Every intermediate transition time,
not only requested lead times, must be present and in the same held-out split.
All dynamic channels are forecast; no target, process label or future boundary
is passed to the model. The initialization set is the manifest's complete-window
subset, chronologically ordered; a cap is explicitly recorded, not called a
full test-set evaluation.

Climatology is a **train-only month/hour grid-cell mean**, fitted from raw fields
in training years only. It is a documented initial baseline, NOT the exact
WeatherBench2 climatology. Missing train buckets raise an error instead of using
held-out data or a silently substituted climatology. Pooled ACC accumulates
latitude-weighted anomaly dot products and energies across initializations.
Zero-energy cases are reported as undefined, not assigned artificial scores.

Outputs: per-variable/per-horizon rmse.csv, acc.csv, and provenance.json with
exact initialization/valid times, checkpoint/manifest hashes, frozen training
identity, normalization units and climatology bucket counts. Physical RMSE
requires known units; --normalized explicitly requests normalized metrics.
No aggregate is formed across incompatible variable units.

Checkpoint identities must match the training manifest and normalization in the
same manifest directory. Identity checks are bounded metadata fingerprints, not
proof that raw scientific observations have not been altered; archive immutability
and source checksums remain required. Synthetic tests validate this pipeline,
not forecast skill. Parent #8 still requires real data, baseline comparisons,
boundary sensitivity and statistical confidence intervals.
