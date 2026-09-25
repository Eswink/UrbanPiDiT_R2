# R7 generic recursive baseline: parameter/FLOP budget parity (#5)

Issue #5's acceptance criterion is one sentence: the generic TRM-like recursive
baseline "must share comparable parameter/FLOP budget with process-aware R7.3".
The model, K=1/2/4/6/8 support, deep supervision and the truncated-training option
already existed. What did **not** exist anywhere in the repository was **any FLOP
accounting** — so half of that acceptance criterion had never actually been
checked. This page closes that gap with a measured audit.

`scientific_claim: false`. A budget match says nothing about forecast skill; it
only makes a skill comparison meaningful rather than confounded by capacity.

## What was missing, precisely

A repository-wide search for `flop` (case-insensitive, excluding the read-only
archive) returned **nothing**:

```bash
grep -rn "flop" --include="*.py" --include="*.md" . | grep -v .venv | grep -v legacy   # no hits
```

Parameter counts did exist (`training/r7_comparison_plan.py` reports them per
variant), and informal counts looked close — but "close" had never been
quantified, and FLOPs had never been measured at all. The audit therefore reports
both, with the counting convention stated, against real published ERA5 windows.

## Counting convention

A FLOP number without its convention is not comparable to anything, so it is
recorded in the report itself:

- `torch.utils.flop_counter.FlopCounterMode` over **one forward pass**. It counts
  linear, convolution and matmul work, including the attention matmuls dispatched
  through the aten ops it recognises.
- **Elementwise and normalization ops are not counted.** The attention here uses
  `F.scaled_dot_product_attention` (`model/layers/sdpa.py:17`,
  `model/layers/window_attention.py:135`), which has no parameters and whose
  softmax/rescale work is therefore outside the count.
- **Backward is not counted.** It is roughly 2x forward for these graphs, but that
  ratio is architecture-dependent, so it is neither measured nor assumed.
- **Gradients must be enabled.** `FlopCounterMode`'s hooks raise
  `AssertionError: Expected gradient function to be set` under `torch.no_grad()`
  on these models — a real constraint that the module documents and the tests pin.

## Result: parity holds at every depth

Measured at `dim=128, depth=4`, batch 2, reading
`outputs/r7_regional_real/manifests/val.jsonl` (real 17-channel ERA5, 8 windows),
tolerance 5 % declared **before** measurement.

| metric | generic | process | relative difference | within 5 % |
| --- | --- | --- | --- | --- |
| parameters | 1,264,034 | 1,264,419 | **+0.030 %** | yes |
| FLOPs, K=1 | 5,006,418,176 | 5,006,422,272 | **+0.0001 %** | yes |
| FLOPs, K=2 | 5,662,100,736 | 5,662,108,928 | **+0.0001 %** | yes |
| FLOPs, K=4 | 6,973,465,856 | 6,973,482,240 | **+0.0002 %** | yes |
| FLOPs, K=6 | 8,284,830,976 | 8,284,855,552 | **+0.0003 %** | yes |
| FLOPs, K=8 | 9,596,196,096 | 9,596,228,864 | **+0.0003 %** | yes |

Both criteria of #5's acceptance are therefore met, and the FLOP half is now
measured rather than merely assumed.

**The cost structure is what makes the match exact.** At `dim=128, depth=4` the two
models share their heavy components verbatim:

| component | generic | process |
| --- | --- | --- |
| backbone | 862,225 | 862,225 |
| recursive cell (`cell` / `reasoning_cell`) | 331,008 | 331,008 |
| correction head | 42,897 | 42,897 |
| draft encoder | 9,088 | 9,088 |
| state→context projection | 16,768 (`latent_to_context`) | 16,768 (`process_to_context`) |
| state bank | 2,048 (`latent`, 16 tokens) | 2,048 (`process_queries`, 8 anchored + 8 free) |
| process readout | — | 385 (`process_readout`) |
| **total** | **1,264,034** | **1,264,419** |

The 385-parameter difference is *entirely* the process readout. Since both models
carry a 16-token recursive state, the per-step matrix work is the same size, which
is why the FLOP ratio stays at 1.0000 for every K even as absolute cost grows
about 1.9x from K=1 to K=8.

**FLOP parity does not imply the models are interchangeable.** The process model
additionally consumes an 8-wide anchored process target and produces process
predictions; that supervision is a different objective on the same compute budget,
not more compute. This is exactly the property #5 needs so that #6's comparison is
not confounded.

## Tests

| Suite | Result |
| --- | --- |
| `tests/test_r7_budget_audit.py` | 8 passed (parity arithmetic and its edges, nonpositive-baseline rejection, tolerance declared before use, parameter counting against the framework's own sum, FLOP growth with K, real-store parity, exclusive self-describing output, stated convention) |
| Full `pytest -q` (GPU present) | **844 passed, 3 skipped, 0 failed** |
| Full `pytest -q` with `CUDA_VISIBLE_DEVICES=""` | 838 passed, 9 skipped, 0 failed |
| `python tools/check_conventions.py` | 34 blocking rules, **0 violations** |

The tests that need the real store skip cleanly in a clean checkout, and no
synthetic stand-in is substituted. CI needs neither GPU nor network.

## Reproduce

```bash
.venv/bin/python training/r7_budget_audit.py \
  --out outputs/r7_budget_audit/report.json --device cpu
```

The report is written exclusively (a second run to the same path is refused) and
carries its own `report_digest`. Re-running to a fresh path reproduced identical
parameter counts, identical FLOP counts and an identical digest.

## Honest limitations

- **Forward-only.** Backward cost is not counted and not assumed to be equal
  across variants.
- **Elementwise/normalization work excluded**, so these are not total-op counts and
  should not be compared against published totals computed differently.
- **One window batch** from the validation split; this is not a training-budget or
  wall-clock measurement — for those, see
  [R7_GPU_MULTISEED.md](R7_GPU_MULTISEED.md).
- **Parameter counts exclude optimizer state**, which is where AdamW doubles the
  memory footprint.
- **A budget match is not skill.** Whether process state helps is #6's question,
  and its current three-seed evidence is mixed.
- No test split was touched, and no threshold was selected from these numbers.
