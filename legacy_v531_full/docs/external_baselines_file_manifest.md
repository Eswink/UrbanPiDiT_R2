# External baseline adaptation file manifest

## Added source archives/directories

```text
baselines/external_sources/FourCastNet-master/
baselines/external_sources/graphcast-main/
```

## Added runnable UrbanPiDiT-loader adapters

```text
baselines/external_models/__init__.py
baselines/external_models/urban_loader_adapter.py
baselines/external_models/fourcastnet_adapter.py
baselines/external_models/graphcast_adapter.py
baselines/external_models/gencast_adapter.py
```

## Updated baseline registry and training entrypoint

```text
baselines/__init__.py
baselines/forecast_models.py
baselines/train_external_baselines.py
```

## Added configs

```text
configs/baselines_external_weather_suite.yaml
configs/baselines_fourcastnet_urban.yaml
configs/baselines_graphcast_gencast_urban.yaml
```

## Added documentation and tests

```text
docs/external_baseline_adaptation.md
docs/external_baselines_file_manifest.md
tests/test_external_baseline_adapters.py
```

## Smoke test

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q tests/test_external_baseline_adapters.py
```

Expected result:

```text
4 passed
```
