# Reusing the original seasonal checkpoints without retraining (#52)

The #50 artifact stores original raw source/receipts, checkpoints and manifests,
but excludes the derived cache. Ordinary rebuilding changes the source-path
provenance string and therefore changes the data identity. The new restoration
path instead preserves the ORIGINAL string as historical provenance, never as a
path to open. It reads only the explicitly pinned local NetCDF, re-runs physical
and process data preparation with the original settings, then demands:

- exact generated metadata and all five manifest byte sequences;
- complete recomputed dataset_identity equal to the archived checkpoint contract;
- unchanged model implementation digest and unchanged checkpoint/controller pins.

No identity is overridden and no checkpoint is modified. Numeric differences
that change normalized data cause a failed restoration, not a relaxed check.
Failed output remains inspectable; only success emits RESTORATION_ACCEPTED.json.
This supports the audited four-season source/layout only, not arbitrary caches.

```bash
python scripts/diagnose_r7_restored_pilot.py --artifact-root <unpacked_50_artifact> --out outputs/new_diagnostic
```

The command restores once, evaluates the three ORIGINAL process checkpoints on
validation only, compares to original forecasts at rtol1e-5/atol1e-12, and runs
the existing retrospective K1..3 gain oracle on24balanced cases. It lists the
original policy candidate constraint failures. There is no training, threshold
retuning, source networking or new test selection. The oracle explicitly uses
true future labels retrospectively and is NOT a deployable halting mechanism.
Its one-step equal-channel normalized MSE is not identical to the every-variable,
every-horizon feasibility constraints used for validation policy selection.

The workflow has a ten-minute CPU bound and preserves raw diagnosis and original
identity evidence. A diagnosis does not establish a new model's forecast skill,
causal reasoning, statistical significance, GPU speedup or a completed thesis.

## Model-source digest requirement

This workflow loads checkpoints produced by an earlier commit, so it must run
under the model implementation those checkpoints recorded. `load_checkpoint`
(`training/r7_experiment.py`) compares the checkpoint's `model_code_sha256`
against the digest of every non-legacy `.py` under `model/` and fails closed on
mismatch. Reformatting any `model/` file changes that digest.

Current pinned digest after the Q-001/long-line clean-up: `d9fb07f2d38d41d681b45c0b4d539edb8f0f9620f44ba0c426c40b43d672f665`
(previously `20196c64...`). To replay an older artifact, run it with the
`code.zip` archived beside that artifact — never by bypassing the identity check.
See `docs/rules/CHANGELOG.md` (2026-09-24, second pass) and
`docs/rules/OPEN_QUESTIONS.md` Q-009.
