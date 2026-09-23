# External Weather Baseline Adaptation for UrbanPiDiT-V5.3.1

This directory adds three loader-compatible strong weather baselines to the UrbanPiDiT benchmark protocol:

1. **FourCastNet**: AFNO/spectral-mixing style baseline adapted from the uploaded `FourCastNet-master` source tree.
2. **GraphCast**: grid graph-message-passing baseline adapted from the uploaded `graphcast-main` source tree.
3. **GenCast**: probabilistic residual-denoising baseline adapted from the `graphcast-main` source tree, which contains GenCast and GraphCast code.

The official source trees are preserved under:

```text
baselines/external_sources/FourCastNet-master
baselines/external_sources/graphcast-main
```

The runnable UrbanPiDiT-loader adapters are implemented under:

```text
baselines/external_models/
  urban_loader_adapter.py
  fourcastnet_adapter.py
  graphcast_adapter.py
  gencast_adapter.py
```

## Why adapters are needed

The official FourCastNet repository expects ERA5-style HDF5/multifile preprocessing and global weather grids. The official GraphCast and GenCast code is JAX/xarray-based and expects the GraphCast data schema, graph construction, pressure-level variables, and pretrained checkpoint format.

UrbanPiDiT uses a compact urban near-surface loader that returns dictionaries such as:

```text
x_ctx:       [B, C*K + S, H, W]
y:           [B, L, C, H, W]
x0:          [B, C, H, W]
static_raw:  [B, S, H, W]
static_cont: [B, S_cont, H, W]
static_cat:  [B, S_cat, H, W]
norm:        mean/std per dynamic variable
lead_times:  [L]
```

Therefore, the adapter layer converts UrbanPiDiT batches into baseline-ready tensors while preserving the same normalization, static channels, history length, lead times, and train/val/test split protocol.

## Fairness protocol

The external baselines support the same static-information policies as existing baselines:

- `same_static`: model receives the same urban morphology variables as UrbanPiDiT.
- `dynamic_only`: model receives only historical meteorological variables.
- `static_zero`: static fields are zeroed.
- `static_shuffle`: static fields are spatially shuffled.

This enables fair comparison of both model family and morphology-information access.

## Important claim boundary

These adapters are **loader-compatible fair-comparison baselines**, not pretrained official global weather models. They preserve the architectural ideas needed for fair local experiments:

- FourCastNet-style spectral/AFNO mixing;
- GraphCast-style grid graph message passing;
- GenCast-style residual denoising/probabilistic refinement.

Do not claim that these runs reproduce official GraphCast/GenCast/FourCastNet global ERA5 benchmark scores unless the official data schema, dependencies, checkpoints, and evaluation protocol are used.

## How to train and evaluate

Run the full three-baseline suite:

```bash
python -m baselines.train_external_baselines \
  --config configs/baselines_external_weather_suite.yaml \
  --out_dir outputs/baselines/external_weather
```

Run FourCastNet only:

```bash
python -m baselines.train_external_baselines \
  --config configs/baselines_fourcastnet_urban.yaml \
  --out_dir outputs/baselines/fourcastnet
```

Run GraphCast + GenCast:

```bash
python -m baselines.train_external_baselines \
  --config configs/baselines_graphcast_gencast_urban.yaml \
  --out_dir outputs/baselines/graphcast_gencast
```

Each run saves:

```text
outputs/baselines/.../<alias>/best.pt
outputs/baselines/.../<alias>/result.json
outputs/baselines/.../external_baselines_results.json
```

## Config files

```text
configs/baselines_external_weather_suite.yaml
configs/baselines_fourcastnet_urban.yaml
configs/baselines_graphcast_gencast_urban.yaml
```

## Tests

```bash
pytest -q tests/test_external_baseline_adapters.py
```

The test uses dummy UrbanPiDiT-loader-style batches and verifies that all three adapters produce `[B, L, C, H, W]` forecasts and can be built from config.

## V5.3.1 external-baseline training control and metrics update

The external baseline trainer now supports validation-loss early stopping through the
`external_baseline_training.early_stopping` block.  The default monitor is
`val_mse`, with `mode: min`, configurable `patience`, `min_delta`, and
`restore_best`.  The trainer writes the stopping state to each model's
`result.json` under `early_stopping`, including `early_stopped`, `stopped_epoch`,
`best_epoch`, and the monitored value.

The external baseline evaluator now reports the following metrics for every lead
time, both globally and per dynamic variable:

- `RMSE` and `RMSE_per_var`
- `MAE` and `MAE_per_var`
- `Bias` and `Bias_per_var`
- `CRPS` and `CRPS_per_var`
- `ACC` and `ACC_per_var` when climatology is available in the batch

For deterministic baselines such as the loader-compatible FourCastNet and
GraphCast adapters, the reported CRPS is the deterministic single-member CRPS,
which is equivalent to MAE.  This preserves a consistent result schema with
probabilistic/adaptive baselines such as GenCast-style variants; if an adapter
later supplies an ensemble tensor, the same metric code uses the standard ensemble
CRPS approximation.

## V5.3.1 external-baseline checkpoint resume and TensorBoard logging update

The external baseline trainer now supports resumable checkpoints and resumable
TensorBoard logging.

### Checkpointing and resume

Each baseline alias writes two checkpoints:

```text
outputs/baselines/.../<alias>/best.pt
outputs/baselines/.../<alias>/last.pt
```

`best.pt` stores the best monitored validation state. `last.pt` stores the full
training state needed for interruption recovery:

- model parameters;
- optimizer state;
- completed epoch and next epoch;
- global optimization step;
- training history;
- best validation monitor value and best epoch;
- early-stopping bad-epoch counter;
- Python/NumPy/PyTorch RNG state.

Resume can be enabled in YAML:

```yaml
external_baseline_training:
  checkpoint:
    enabled: true
    save_last: true
    save_every: 1
    resume: true
    resume_path: null
    strict: true
    load_optimizer: true
```

or from the command line:

```bash
python -m baselines.train_external_baselines \
  --config configs/baselines_external_weather_suite.yaml \
  --out_dir outputs/baselines/external_weather \
  --resume
```

With `--resume`, the trainer resolves each baseline checkpoint as:

```text
<out_dir>/<alias>/last.pt
```

You can also provide a checkpoint file or an output directory:

```bash
python -m baselines.train_external_baselines \
  --config configs/baselines_fourcastnet_urban.yaml \
  --out_dir outputs/baselines/fourcastnet \
  --resume_path outputs/baselines/fourcastnet/fourcastnet_same_static/last.pt
```

### TensorBoard and resumed logs

TensorBoard logging is controlled by:

```yaml
external_baseline_training:
  tensorboard:
    enabled: true
    log_dir: outputs/baselines/tensorboard
    resume: true
    flush_secs: 30
    log_every_n_steps: 20
    write_jsonl_fallback: true
```

When resuming, the trainer restores the checkpoint `global_step` and opens the
same log directory for the alias. If native TensorBoard is installed, the writer
uses `purge_step=<restored_global_step>` so duplicated stale events after the
resume point are hidden by TensorBoard and new scalar records continue from the
restored step. This gives true breakpoint-continuation logging rather than a new
independent run.

If the `tensorboard` Python package is not installed, training still runs and the
trainer writes a resumable `scalars.jsonl` fallback file under the same log
directory. Install TensorBoard to create native event files:

```bash
pip install tensorboard
```

Then launch:

```bash
tensorboard --logdir outputs/baselines/tensorboard
```

The trainer logs:

- step-level training MSE;
- epoch-level training/validation MSE;
- monitor value, best monitor, bad-epoch count;
- final validation/test RMSE, MAE, Bias, CRPS, and ACC for each lead time;
- per-variable validation/test RMSE, MAE, Bias, CRPS, and ACC.
