# R7.4: next-step gain controller

Engineering child: #19. Scientific acceptance remains in #7.

## What is implemented

`model/r7_halting.py` wraps an existing `ProcessForecastCoReasoner`. The frozen
forecaster and its fixed-depth API are not modified. After step k, a small MLP
consumes the predicted process readout, per-channel draft mean/RMS, correction
RMS, and log(1+k). It predicts a signed next-step error reduction and a CONTINUE
logit. These features contain no target, observed error, or process label.

`training/r7_halting.py` generates training labels from frozen-teacher runs:

```
E_k = per-sample cosine-latitude-weighted MSE(Y_k, target)
G_k = E_k - E_(k+1)
continue_k = 1[G_k > gain_threshold]
```

Input fields must already have training-only, per-channel normalization.
Loss is Huber regression of signed gain plus binary cross-entropy of the
CONTINUE decision. All error reductions and regression/BCE inputs use FP32.
For K drafts Y1..YK, there are K-1 supervised decisions: YK has no observed
successor and is not assigned an invented STOP label.

Teacher execution is streamed under `no_grad()` in eval mode. Only compact
features/errors are retained; full-resolution draft histories and teacher
backward graphs are not retained. Original per-module train/eval flags are
restored. The optimizer helper accepts exactly the controller's trainable
parameters, clips/checks gradients, and records only successful optimizer
updates. A loss computation alone does not mark the controller as trained.

## Usage

First train/load a normal R7.3 forecaster. Then, on **training data only**:

```python
import torch
from model.r7_halting import AdaptiveProcessForecaster
from training.r7_halting import calibrate_controller_step

# core is an already trained ProcessForecastCoReasoner.
adapter = AdaptiveProcessForecaster(core, gain_threshold=0.0)
optimizer = torch.optim.AdamW(adapter.controller.parameters(), lr=1e-3)
for batch in train_loader:
    losses = calibrate_controller_step(adapter, optimizer, batch, max_steps=4)

adapter.eval()
# inputs contains initialization-time history/lead only; no target is needed.
result = adapter(inputs, max_steps=4, min_steps=1)
torch.save(adapter.state_dict(), 'r7_adaptive_state.pt')
```

Reconstruct the same architecture and load the state dict to restore forecaster,
controller, thresholds, and update count. The update count is an engineering
safety guard, **not a claim of held-out calibration or scientific readiness**.
Keep checkpoint source SHA, model config, variable order, normalization manifest,
training split and threshold-selection protocol alongside the checkpoint.
Changing forecaster weights requires recalibration; this version does not yet
cryptographically bind a controller to a particular forecaster checkpoint.

## Genuine active-subset execution

Only active rows enter the recurrent cell, draft encoder, process readout and
correction head. Stopped rows keep their last forecast/process readout and never
reactivate. Kmax bounds every sample; min_steps sets the earliest decision.
There is no global-batch-mean stopping condition. Invalid controller outputs
fall back to more computation up to Kmax and remain visible in diagnostics.

`active_masks[B,S]` denotes actual sample-step execution.
`decision_masks[B,S]` distinguishes a real controller query from unqueried zeros
in `predicted_gains` and `continue_probabilities`. `reasoning_steps_per_sample`
counts only refinement steps, not encoder/controller costs. The encoder runs
once. No full-resolution draft history is returned by adaptive inference.

Inference whitelists only `coarse_history` and optional `lead_time_hours` for
this version of the backbone. In particular, it does not forward
`atmos_target`, `process_targets`, or an externally supplied `atmos_baseline`.
Call `.eval()` before inference. An untrained controller is rejected by default.
`force_full_depth=True` is the fixed-depth equivalence/debug path;
`allow_untrained=True` is for explicit engineering tests only.

## Validation commands

```bash
pytest -q tests/test_r7_halting_targets.py tests/test_r7_adaptive_inference.py
python scripts/smoke_r7_halting.py --steps 3 --updates 2
pytest -q
```

The smoke deliberately uses small synthetic CPU tensors. It tests an optimizer
update, unchanged forecaster weights and target-free inference; it does not
validate meteorological skill, 0.25-degree data quality, GPU memory or speedup.

## Research gates still open

- Compare adaptive vs fixed K=1/2/4/8 on frozen held-out ERA5 periods; report
  per-variable error, extremes, multiple seeds and confidence intervals.
- Measure actual latency, throughput and peak allocated/reserved VRAM. Fewer
  sample-steps are not automatically proportional wall-clock savings because
  encoder cost, indexing, synchronization and batch size also matter.
- Calibrate thresholds on validation data only; lock the test set. Compare
  learned policy with confidence-only and fixed-depth policies.
- One-step gain is greedy: an unhelpful next step can precede a useful later
  correction. Test look-ahead targets and an oracle exit reference before
  claiming optimal computation allocation.
- This controller is not a physical consistency verifier. Input-time diagnostic
  anchors are not future-time physical truth. Do not enforce assertions such
  as warm advection always implying net warming while omitting other terms.
- Compute-penalty warm-up/joint policy optimization remain separate work in #7.
- Existing end-to-end recursive training still retains graph-carrying draft
  histories even with detach_between_steps; bounded-memory training is #20.
