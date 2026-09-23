# R7 streamed truncated training

Engineering child #21 of memory investigation #20. Opt-in utility; existing Lightning/full-BPTT and retained-trace baselines are unchanged.

## Semantics

`backward_streamed_truncated` implements the existing `detach_between_steps=True` recurrence. The initial forecast Y0 still receives gradients through the first correction. Between later rounds the recurrent state and forecast are detached. Forecast loss weights are normalized linear weights from 1 to `final_weight`; process MSE is averaged over the K process predictions. All loss reductions are FP32. Process target dimensions must match exactly rather than silently clipping channels.

Instead of retaining all recurrent graphs, the implementation keeps detached interface leaves for initial forecast and shared context. Per-round losses immediately backpropagate into the recurrent parameters and these leaves. After the final round, a single multi-root backward sends the accumulated leaf adjoints through the original backbone graph. This keeps encoder gradients while releasing each recurrent graph before the next round. No `retain_graph=True`, no per-round optimizer step and no full-BPTT claim.

The memory structure is one backbone graph plus one recurrent graph, current fields and O(K) detached scalar errors. It does not promise constant total CUDA allocation: inputs, parameters, optimizer state, allocator reserves, kernels and checkpointing matter too. The computation still grows with K.

## Usage

```python
from training.r7_streaming import train_streamed_update

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
logs = train_streamed_update(
    model, optimizer, [microbatch_a, microbatch_b],
    reasoning_steps=4, process_weight=0.1,
    amp_dtype=torch.bfloat16,
)
```

Call outside autocast; the helper wraps forward operations and runs backward outside those contexts. Model parameters remain FP32. FP32 and BF16 are supported on CPU/CUDA; scaled FP16, distributed execution and Lightning automatic-optimization integration are not implemented. The caller owns the dataloader, epochs, scheduler and checkpoints.

Each explicit microbatch group performs exactly one optimizer update after all backwards. Contributions use batch-size/total-samples weighting, so an uneven last microbatch is not overrepresented. A pre-step loss/gradient failure clears partial gradients and does not call optimizer.step. A failure inside an arbitrary optimizer's own step is not transactionally rolled back. Only the final detached forecast and scalar losses are returned. Avoid retaining GPU input microbatch groups larger than the intended accumulation budget.

Inference inputs are whitelisted: `coarse_history` and optional `lead_time_hours`. `atmos_target` and optional `process_targets` enter losses only; an external future baseline never enters the backbone in this utility.

## Verification and measurement

Regression tests compare the loss, final forecast and every parameter gradient against the retained-trace truncated reference at K=0/1/2/4 with dropout=0. They also cover feedback ablation, checkpointing, BF16, encoder gradients, optimizer ownership, uneven microbatch accumulation and failure cleanup.

`SavedTensorMeter` uses PyTorch saved-tensor hooks to track peak live **logical saved-slot bytes**. Aliases and repeatedly saved weights are counted separately; unsaved activations, optimizer state and CUDA workspaces are excluded. These are graph-lifetime regression numbers, not VRAM or timing measurements. Tests report K=1/2/4/8 in the CI log.

```bash
python scripts/smoke_r7_streaming.py --steps 4 --updates 2
# Run only on the user's available CUDA device; never silently falls back:
python scripts/smoke_r7_streaming.py --steps 4 --updates 2 --device cuda --bf16
```

The smoke uses tiny synthetic tensors and labels its output accordingly. Its optional CUDA counters do not constitute a production-size 4090D acceptance test. #20 remains open until realistic data shape, optimizer, K, peak allocated/reserved memory and runtime are measured on actual hardware. No SOTA or weather-skill claim is implied.

Primary API references:
- https://docs.pytorch.org/docs/stable/generated/torch.autograd.backward.html
- https://docs.pytorch.org/docs/stable/autograd.html#torch.autograd.graph.saved_tensors_hooks
