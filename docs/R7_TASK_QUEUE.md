# R7 conversation-driven task queue

Snapshot: implementation of #23, based on 78089c5216011d5a455f6692dd16599e6779f4bf.
No scheduled automation. No automatic merge/release, paid GPU use or large data downloads.

| Task | State | Next action / resume condition |
|---|---|---|
| #23 strict geographic axes | VERIFY | Run exact-SHA full CI; inspect new grid cases and both builder callers |
| #20 streamed training | TODO | Integrate the already-tested #21 optimizer path with a local-data CLI, checkpoint/resume and measured GPU profiling |
| #8 rollout evaluation | TODO | Connect #22 primitives to timestamp-exact Zarr windows and train-only climatology/ACC |
| #13/#16 production data | TODO | Audit non-overwrite behavior and metadata/schema integrity before local real-data acceptance |
| #14 process diagnostics | VERIFY | Check existing diagnostic tests and data-to-model integration; distinguish proxy engineering from weather validity |
| #5/#6 same-data comparisons | BLOCKED (research) | Need actual multi-year ERA5 cache, trained baselines and same-budget multi-seed experiments |
| #7 adaptive scientific gate | BLOCKED (research) | Need a trained forecaster and held-out calibration/evaluation; engineering #19 is complete |
| #9 fine-resolution extension | BLOCKED | Needs genuine high-resolution dynamic targets and completion of the core benchmark |

TODO -> IN_PROGRESS -> VERIFY -> DONE is the default path. A waiting workflow
is recorded with commit/run/resume condition and does not block independent
review, implementation or tests. Do not claim cancelled/in-progress CI passed.
New directly relevant defects become child issues; do not add unrelated features.

#23 longitude policy: ascending or descending, all coordinates in either
[-180,180] or [0,360), strictly monotone with span <360. No wrapped seam or
repeated meridian. Convert dateline representation together with the data;
the guard never silently sorts, interpolates or relabels fields.
