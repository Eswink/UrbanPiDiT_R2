# Real Data Audit Round 4 — Portability & Packaging

## Initial score: 82/100 — FAIL

### Failure

The first generated manifests stored absolute `/mnt/data/...` NPZ paths. They worked inside the build sandbox but would break after extraction on another machine. This is a delivery-blocking defect.

## Rework

- Manifest `path` is now relative to the manifest directory, e.g. `../../processed/real_smoke/...npz`.
- Provenance uses project-relative source paths when the source is inside the repository.
- Tests were updated to resolve artifacts with the same semantics as `ManifestNPZDataset`.
- The whole project was copied to a new directory. Processed smoke data/manifests were deleted and regenerated there.
- In the new directory, real-data prepare, model smoke, 15-test pytest suite, Lightning fit, checkpoint restore and test all passed.

## Final score: 98/100 — PASS

### Portable checkpoint evidence

`77cd1a34466534dda896484301406452703a74498ba16cf11859b2926cca4d90`

The checkpoint itself is intentionally excluded from the final archive because the tiny smoke weights have no scientific value; the audit log and SHA256 remain.
