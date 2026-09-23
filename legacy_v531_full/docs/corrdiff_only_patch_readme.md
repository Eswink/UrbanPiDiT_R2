# UrbanPiDiT V5.3.1 CorrDiff-only PhysicsNeMo Loader Adapter Patch

This patch contains only the CorrDiff-related additions needed to adapt NVIDIA PhysicsNeMo's CorrDiff example to the UrbanPiDiT loader and fair-baseline runner.

Included:

- `baselines/external_models/corrdiff_adapter.py`
- `baselines/external_models/urban_loader_adapter.py`
- CorrDiff source snapshot from `physicsnemo-main/examples/weather/corrdiff/`
- `baselines/external_sources/PHYSICSNEMO_CORRDIFF_SOURCE_MANIFEST.json`
- CorrDiff configs:
  - `configs/baselines_corrdiff_urban.yaml`
  - `configs/baselines_corrdiff_urban_smoke.yaml`
- CorrDiff protocol doc and CorrDiff tests
- minimal shared baseline runner/metrics files required for CorrDiff training and evaluation

Not included:

- FourCastNet adapter/source/config
- GraphCast adapter/source/config
- GenCast adapter/source/config
- full PhysicsNeMo repository

Recommended wording in the paper:

> We include a NVIDIA PhysicsNeMo CorrDiff-style residual corrective diffusion baseline adapted to the same UrbanPiDiT loader, lead-time setting, static-information protocol, and metric suite. The adapter follows the regression-then-generative-correction design and is trained from scratch on our urban forecasting data.

Do not claim this is an official reproduction of NVIDIA CorrDiff results unless you also use the official PhysicsNeMo data schema, dependencies, checkpoints, and scoring protocol.

Run:

```bash
python -m baselines.train_external_baselines \
  --config configs/baselines_corrdiff_urban.yaml \
  --out_dir outputs/baselines/corrdiff
```

Smoke run:

```bash
python -m baselines.train_external_baselines \
  --config configs/baselines_corrdiff_urban_smoke.yaml \
  --out_dir outputs/baselines/corrdiff_smoke
```

Resume:

```bash
python -m baselines.train_external_baselines \
  --config configs/baselines_corrdiff_urban.yaml \
  --out_dir outputs/baselines/corrdiff \
  --resume
```