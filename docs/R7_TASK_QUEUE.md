# R7 conversation-driven task queue

No scheduled automation. No automatic merge/release, paid GPU use or large data downloads.

| Task | State | Next action / resume condition |
|---|---|---|
| #23 strict geographic axes | DONE | 6fc7f8c; CI 35852177938: 142 passed / 3 local-fixture skips |
| #24 safe data publication | VERIFY | Exact-SHA full CI: overwrite, failure, ns/us, centered moments and strict reader tests |
| #20 streamed training | TODO | Integrate #21 optimizer path with local-data CLI, checkpoint/resume and actual GPU profiling command |
| #8 rollout evaluation | TODO | Connect #22 to timestamp-exact Zarr windows and train-only climatology/ACC |
| #13/#16 production data | IN_PROGRESS | #24 hardening, followed by real local-data acceptance |
| #14 process diagnostics | VERIFY | Existing diagnostic tests pass; real input-unit audit still needed |
| #5/#6 same-data comparisons | BLOCKED (research) | Multi-year ERA5 cache, trained baselines and controlled multi-seed experiments |
| #7 adaptive scientific gate | BLOCKED (research) | Trained forecaster, held-out calibration/evaluation and actual hardware measurements |
| #9 fine-resolution extension | BLOCKED | Genuine high-resolution dynamic targets and completion of the core benchmark |

A queued/running workflow is WAITING inside VERIFY, not DONE. Record SHA/run and
resume conditions in the issue; switch to independent implementation/review/tests.
New directly relevant defects become child issues; no unrelated scope expansion.
