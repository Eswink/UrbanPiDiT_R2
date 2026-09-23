# Controlled baseline execution (#29)

The shared local runner supports native model_config.architecture values:
`window` (default), `unet`, `convlstm`, `afno_small`. Generic/process recursive
models retain the window backbone. All consume the same normalized weather
history and lead time, predict persistence-relative tendencies on the native
grid, and use the same checkpoint/evaluation interfaces. Future labels are not
conditioning inputs.

These are compact **regional adaptations**, not exact reproductions of the
original papers or published pretrained systems:
- U-Net: two down/up stages, finite boundaries and regression residual head.
- ConvLSTM: one convolutional recurrent layer, no peepholes, all dynamic channels.
- AFNO-inspired: block-diagonal complex two-layer spectral MLP, soft shrinkage,
  FP32 FFT and a patch decoder. It retains all frequency bins rather than copying
  a FourCastNet-specific mode/architecture/training recipe. Its FFT implies
  periodic spectral mixing, which must be considered in regional boundary tests.
- Window/deeper-window controls are Swin-style finite-window forecasters, not
  claims of an official Swin or global weather-model reproduction.

Primary methodological references:
https://arxiv.org/abs/1505.04597
https://arxiv.org/abs/1506.04214
https://arxiv.org/abs/2111.13587

```bash
python scripts/plan_r7_comparison.py --out outputs/comparison_plan --channels 17 --dim 128 --seeds 42 43 44
```

This writes nine architecture/ablation configurations per seed and measures
actual parameter counts by instantiation only. It launches **no training**.
Use the same immutable ERA5 train/val/test cache, variable order, normalization,
initialization set, tuning budget and stopping criterion for every comparison.
Validation chooses hyperparameters; test data do not tune K/halting thresholds.
Include persistence, fixed K and calibrated adaptive controls. Report actual
training/inference time and memory; parameter count alone is not compute parity.
No SOTA ranking or positive result is implied by providing runnable baselines.
