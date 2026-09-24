# R7 GPU multi-seed optimization/trick comparison (#20)

This closes the last open item of #20: the bring-up sweep measured each
configuration once, at a single seed, so a small delta could not be told apart
from machine variation. Here every configuration is replicated across three
seeds and each trick is compared against its baseline **within the same seed**.

`scientific_claim: false` throughout. This measures memory and step time on real
multivariate ERA5 data; it is not a forecast-skill or convergence result.

## Why the earlier sweeps could not answer this

`docs/R7_GPU_BRINGUP.md` §10 records two facts that together made the earlier
single-seed numbers insufficient: absolute step times vary by up to ~40 % between
runs on this machine, and the sweep carried no multi-seed comparison at all. A
one-seed difference of a few percent is therefore within noise, and reporting it
as a trick's effect would be an artefact.

Three design choices follow, all enforced rather than described:

1. **Pairing inside a seed.** Deltas are computed as (trick arm − baseline) for
   the same seed, so seed-to-seed drift cancels instead of inflating the effect.
2. **One process per cell.** `max_memory_reserved` is the caching allocator's
   high-water mark and carries across cells inside one interpreter — the
   artefact already corrected for the single-GPU sweep. The shell driver
   `scripts/sweep_r7_multiseed.sh` runs one interpreter per cell; the Python
   module contains no process spawning at all.
3. **Real data.** Cells read the published regional ERA5 R7 store (17 channels,
   65x65, `BUILD_COMPLETE.json`) through the production reader, replacing the
   synthetic 12x12 shapes the earlier sweeps used.

## Frozen protocol

Written to `protocol.json` before any cell is measured; `cells.jsonl` carries its
digest as its header line, and re-deriving that digest from the stored protocol
reproduces the header.

A protocol is only frozen if it describes what was actually measured. The first
driver version passed `--bf16` to the per-cell invocation but not to `--plan`, so
`protocol.json` recorded `dtype: fp32` while the cells really ran bf16 — a
silently wrong frozen record. The driver now keeps one shared `MEASURE_ARGS`
array that both the plan step and every cell expand, and
`tests/test_r7_multiseed_comparison.py` asserts that sharing instead of trusting
it. The digest below is from the corrected run.

| Item | Value |
| --- | --- |
| Protocol digest | `6c84be201134faa3a0895ebb9c1bfb1cf67f4161dd283fd613ba2a33fc70e7f7` |
| Seeds | 41, 42, 43 |
| Models | `generic`, `process` (forecast feedback off, to isolate the trick) |
| K (reasoning steps) | 1, 4, 8 |
| Training modes | `full_bptt`, `streamed_truncated` |
| Activation checkpointing | off, on |
| Cells | 2 x 3 x 2 x 2 x 3 = **72** |
| Data | `outputs/r7_regional_real/manifests/val.jsonl` — real 17-channel ERA5, 8 windows |
| Batch / dim / depth | 4 / 384 / 8, bf16 autocast with fp32 master weights |
| Noise test | dataset is fixed at 8 windows, and the realized batch is 4 of those 8; this is a memory/latency instrument, not a data-scale experiment |

The store's process diagnostics are 8 wide, so the `process` model adopts
`anchored_processes=8` from the data rather than a synthetic default.

## Result: all 72 cells, no failures

```
complete=true  measured=72  planned=72  failed=[]  missing=[]  elapsed=266.4 s
```

Peak allocated / reserved memory, mean over the three seeds (MiB):

| model | K | mode | ckpt | alloc | reserved | step (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| generic | 1 | full_bptt | off | 934 | 974 | 46.6 |
| generic | 1 | full_bptt | on | 425 | 504 | 70.5 |
| generic | 1 | streamed | off | 938 | 1002 | 53.8 |
| generic | 1 | streamed | on | 440 | 576 | 87.4 |
| generic | 4 | full_bptt | off | 1164 | 1222 | 64.4 |
| generic | 4 | full_bptt | on | 528 | 584 | 106.7 |
| generic | 4 | streamed | off | 957 | 1008 | 96.6 |
| generic | 4 | streamed | on | 444 | 576 | 130.2 |
| generic | 8 | full_bptt | off | 1464 | 1510 | 97.7 |
| generic | 8 | full_bptt | on | 723 | 794 | 150.1 |
| generic | 8 | streamed | off | 957 | 1008 | 213.4 |
| generic | 8 | streamed | on | 444 | 576 | 212.5 |
| process | 1 | full_bptt | off | 900 | 948 | 46.2 |
| process | 1 | full_bptt | on | 422 | 532 | 71.1 |
| process | 1 | streamed | off | 900 | 1058 | 56.5 |
| process | 1 | streamed | on | 440 | 552 | 90.9 |
| process | 4 | full_bptt | off | 1015 | 1076 | 72.1 |
| process | 4 | full_bptt | on | 438 | 526 | 102.0 |
| process | 4 | streamed | off | 923 | 1062 | 117.7 |
| process | 4 | streamed | on | 443 | 564 | 138.5 |
| process | 8 | full_bptt | off | 1174 | 1240 | 99.8 |
| process | 8 | full_bptt | on | 540 | 598 | 152.0 |
| process | 8 | streamed | off | 923 | 1062 | 212.5 |
| process | 8 | streamed | on | 443 | 564 | 235.4 |

### Streamed truncated vs full BPTT (per-seed paired)

| model | K | d_alloc (MiB) | sd | d_step (ms) | sd | same sign in all seeds |
| --- | --- | --- | --- | --- | --- | --- |
| generic | 1 | +10 | 7 | +12.0 | 7.2 | yes |
| generic | 4 | −145 | 67 | +27.8 | 11.4 | yes |
| generic | 8 | −393 | 125 | +89.1 | 31.7 | yes |
| process | 1 | +9 | 10 | +15.1 | 6.9 | **no** |
| process | 4 | −44 | 53 | +41.1 | 8.0 | **no** |
| process | 8 | −174 | 85 | +98.0 | 18.5 | yes |

**Reading.** Streamed truncated training is now measured on real multivariate ERA5
to be **flat in K while full BPTT grows**: for `generic`, full BPTT rises
934 → 1164 → 1464 MiB over K=1/4/8 (a 530 MiB rise) while streamed stays
938 → 957 → 957 MiB (a 19 MiB rise), and the paired saving at K=8 is −393 MiB
with every seed agreeing. This independently reproduces the synthetic-shape
bring-up finding on real data.

**Two honest negatives.** Streamed training is **slower at every K** — the saving
in memory is paid for in step time (+12.0 to +98.0 ms; all seeds agreeing for
`generic`). And at K=1 and K=4 for `process`, the memory delta **does not have
the same sign across seeds**, so at those settings the effect is smaller than
seed-to-seed dispersion and this run does not establish it. Those rows are
reported as unresolved rather than as wins.

### Activation checkpointing on vs off (per-seed paired)

| model | K | d_alloc (MiB) | sd | d_step (ms) | sd | same sign in all seeds |
| --- | --- | --- | --- | --- | --- | --- |
| generic | 1 | −503 | 7 | +28.8 | 9.7 | yes |
| generic | 4 | −574 | 67 | +38.0 | 11.0 | yes |
| generic | 8 | −627 | 125 | +25.8 | 32.5 | yes |
| process | 1 | −469 | 10 | +29.6 | 6.8 | yes |
| process | 4 | −528 | 53 | +25.3 | 8.0 | yes |
| process | 8 | −557 | 85 | +37.6 | 18.5 | yes |

Checkpointing is the strongest memory lever measured here: −469 to −627 MiB, a
consistent **51.8 % to 54.5 % cut** of the baseline peak, for +25 to +38 ms per
step. Every row agrees in sign on both memory and step time.

### Cross-run reproducibility: memory is exact, timing is not

The sweep was run twice end to end (the first run used the buggy driver that
mislabelled `dtype`, so both exist and are comparable on the numbers). Comparing
all 72 per-cell values between the two runs:

| metric | max abs difference | mean abs difference | identical cells |
| --- | --- | --- | --- |
| peak allocated | **0.0 MiB** | 0.0 MiB | **72 / 72** |
| peak reserved | **0.0 MiB** | 0.0 MiB | **72 / 72** |
| step time | 41.0 ms | 8.2 ms | 35 / 72 within 5 ms, 52 / 72 within 10 ms |

Memory is deterministic here: every one of the 72 cells reproduced
bit-identically. Step time is not — this directly corroborates the ~40 %
run-to-run variation recorded in `docs/R7_GPU_BRINGUP.md` §10, and it is the
reason a single-seed timing delta should not have been trusted. It also explains
which of the two summaries to believe: the memory columns are stable enough to
pair across seeds, the timing columns are reported with their dispersion and not
as a headline.

One concrete consequence: between the two runs, `d_step` for `process` K=4 moved
from +29.3 ms to +41.1 ms, and the `process` K=8 checkpointing timing delta moved
from mixed-sign to same-sign. Memory deltas did not move at all.

## Tests

| Suite | Result |
| --- | --- |
| `tests/test_r7_multiseed_comparison.py` | 15 passed (offline: grid, tags, pairing, dispersion, incomplete/duplicate/OOM handling, protocol-before-measure, driver contract) |
| Full `pytest -q` (GPU present) | **834 passed, 3 skipped, 0 failed** |
| Full `pytest -q` with `CUDA_VISIBLE_DEVICES=""` | 828 passed, 9 skipped, 0 failed |
| `python tools/check_conventions.py` | 34 blocking rules, **0 violations** |

CI needs no GPU and no network: the tests import the module by path and exercise
the aggregation math with synthetic rows. The script itself refuses to run a
CUDA cell when CUDA is absent rather than silently falling back to CPU.

An incomplete sweep cannot publish a comparison table: `--aggregate` refuses to
emit `summary` unless all planned cells are present and none failed, and it
records `missing_cells` / `failed_cells` so an interrupted run is visible rather
than averaged over.

## Honest limitations

- **Three seeds is a dispersion check, not a significance test.** No p-values are
  computed and none should be inferred. Where signs disagree across seeds, the
  finding is explicitly unresolved.
- **Not a skill or convergence result.** Packed batches, K up to 8, a fixed short
  measured window. Nothing here says recursion improves forecasts.
- **One GPU and one data split** (validation only); no test split was touched.
- **Absolute step times vary between runs** on this machine, which is exactly why
  the deltas are paired and why the single-seed reading was retired.
- **The batch is a fixed noise-test instrument**: 4 of 8 available validation
  windows. Memory numbers are valid for that setting only.
- Analysis artifacts under `outputs/` are not committed (R-012); the code,
  protocol digest and these tables are.

## Reproduce

```bash
# freeze the protocol, then measure one cell per process, then aggregate
bash scripts/sweep_r7_multiseed.sh outputs/r7_multiseed_real

# or step by step
.venv/bin/python scripts/compare_r7_gpu_multiseed.py --plan --out outputs/r7_multiseed_real
.venv/bin/python scripts/compare_r7_gpu_multiseed.py \
  --cell '{"kind":"generic","k":8,"training_mode":"streamed_truncated","activation_checkpointing":false,"seed":41}' \
  --cell-out outputs/r7_multiseed_real/cells/generic_K8_streamed_truncated_ck0_s41 \
  --data real --device cuda --batch 4 --dim 384 --depth 8 --warmup 1 --measure 2 --bf16
.venv/bin/python scripts/compare_r7_gpu_multiseed.py --aggregate --out outputs/r7_multiseed_real
```
