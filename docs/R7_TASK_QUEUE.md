# R7 conversation-driven task queue

Updated after verified issue #13 regional acquisition at `f37ceba`, the #20
multi-seed comparison at `7dfbea6`, the #5 budget-parity audit at `bf57fd4`, the #6 fair-budget comparison at `1bc1eef`, and the #7 halting-gate audit at `0f5df17`.
No scheduler, main merge/release, force push, paid GPU or uncontrolled data mirror.

## Accepted work in this iteration

| Issue | State | Verification |
| --- | --- | --- |
| #53 variable-wise error/update geometry | DONE | actual35891600411; CI35891600361 |
| #54 fixed200->800 original-checkpoint control | DONE | actual35892145587; CI35892145656 |
| #55 same-budget continuous250day source | DONE | actual35893397412; test-only repairCI35894113212 |
| #56 pinned offline continuous-data800update control | DONE | actual35895446235; CI35895446093 |
| #57 optional spatial solver plus400update matched ablation | DONE (engineering/experiment) | actual35896740916; CI35896740912; defaultFalse retained |
| #58 explicit case selection/common-case retrospective audit | DONE | actual35898091953; CI35898091928 |

Latest full code CI: **578passed,3old-fixture-skipped,2existingLightningwarnings**
in49.25s, job107306975262, plus compile/whitespace/installed-wheel checks.
No new tests were skipped. No new implementation is waiting for verification.

Detailed evidence, limits and mixed results:
[R7_CPU_REFINEMENT_RESULTS.md](R7_CPU_REFINEMENT_RESULTS.md).
Per-archive hashes: [R7_CPU_ITERATION_ARTIFACTS.json](R7_CPU_ITERATION_ARTIFACTS.json).

## GPU bring-up on 2×3090 (engineering, not science)

Measured at `50954c94eb0aa15fa61cb19440543b40c6c38326`. Full record:
[R7_GPU_BRINGUP.md](R7_GPU_BRINGUP.md). Every artifact `scientific_claim: false`.

| Item | Result |
| --- | --- |
| Single-GPU sweep | 2 models × K=1/2/4/8 × {full BPTT, streamed} × ckpt on/off, 32/32 OK at 1.25M and 18.07M params, one process per cell |
| Streamed vs full BPTT | streamed **flat on both metrics** across K=1…8 (46.00 MiB reserved at 1.25M; 380.00 MiB at 18.07M); full BPTT grows (394 → 426 MiB reserved at 18.07M) |
| Measurement artefact | whole-sweep-in-one-process inflates `reserved` via allocator carryover (process K=1: 428 vs true 396 MiB); per-process measurement required |
| Checkpointing cost | 21 % (1.25M) to 70 % (18.07M) more step time to bound the peak |
| GPU checkpoint resume | bitwise identical weights, optimizer state and loss sequence |
| DDP smoke (first in repo) | 2 ranks, loss vs single-GPU reference max delta 3.1e-06, sampler covers dataset once, 1 checkpoint, resume bitwise identical |
| DDP negative result | **7 % slower** than single-GPU at this scale on SYS/PCIe (no NVLink); does not add per-model memory |
| OOM boundary | 178.2M params / batch 48 / 128×128 / K=8; only streamed+checkpointing survived in every degradation variant |
| Real 11-channel ERA5 | **not on this disk** (`data/*` = `.gitkeep` only) — memory numbers use synthetic shapes |

Engineering child items for #20 are covered. The **multi-seed optimization
comparison is not done**, and #5/#6/#7/#8 stay open — engineering passing is
not a scientific result.

## Prior work is not pending

Issues47–52 had already completed: source fill-value/budget correctness,
four-season source/replay, multiseed study, compact strong baselines, original
checkpoint/cache restoration and delayed-benefit diagnostics. Do not recreate
them from old chat summaries. Earlier forecast, controller, streamed backward,
strict data contracts, chronological normalization, rollout, ACC, profiling,
policy selection and boundary scoring remain implemented.

## Available real input

Eleven physical ERA5 channels on one12x12 native0.25-degree tile.
3000six-hour timestamps:250days each in2018/2019/2020, not3full years.
998one-step windows per chronological split, eight real input-process proxies.
NetCDF19,052,672bytes, SHA256
`0609fa38c1d88b82b985a15f93dd502c7eb7031f5d2b7452bbe971bf936dd9c1`.
Decoded source read charge180,142,968bytes remains under192MiB; this is not HTTP
traffic/RAM. Local replay needs no new cloud-source access.

## Remaining research gates

| Parent/task | State | Evidence/next useful action |
| --- | --- | --- |
| #13 representative data | IN_PROGRESS | Bounded regional acquisition completed at `f37ceba`: 3 years x 4 seasons, 65x65 East-Asia, 48 exact 6-hourly timestamps, 9 variables, source SHA256 `d3fa1fba6da46ed59a535ce27f7501813afc2c93cb8b745c90e349454a40960a`. Converted through the audited publication path to a 17-channel store (8 windows per split, `BUILD_COMPLETE.json`, train-only statistics). Coverage is four 24-hour blocks per year, NOT continuous full-year; see [R7_REGIONAL_ACQUISITION.md](R7_REGIONAL_ACQUISITION.md). |
| #5/#6 recurrence/process benefit | #5 acceptance DONE at `bf57fd4`; #6 fair-budget comparison DONE (negative) at `1bc1eef` | #5's parameter/FLOP parity is measured: +0.030% parameters and +0.0001–0.0003% forward FLOPs across K=1/2/4/6/8 on real ERA5 — [R7_BUDGET_PARITY.md](R7_BUDGET_PARITY.md). #6's fair-budget comparison is done and the gate is NOT met: at K=3, 15 of 17 deltas flip sign between seeds, the only established effects are t2m worsening (+0.30 K, all depths and seeds) and t500 improving under the no-feedback arm, and forecast feedback does not rescue it — [R7_COREASONING_FAIR_BUDGET.md](R7_COREASONING_FAIR_BUDGET.md). The earlier 'feedback hurts T500' framing did not reproduce. Keep defaults and all negatives. |
| #7 adaptive benefit | audit DONE (negative) at `0f5df17` | Gate audited and found not satisfiable as stated: K=3 raises the equal-channel normalized objective 1.5–3.3% over K=0 (worse in all seeds for process_no_feedback), so fixed-Kmax is not the accuracy ceiling; and no shallower depth passes the per-variable tolerance at any tolerance ≤10% because `z250` stays 14–19% above the reference. This explains the earlier full-depth fallback. Controller thresholds were NOT changed and no savings are claimed — see [R7_HALTING_GATE_AUDIT.md](R7_HALTING_GATE_AUDIT.md). |
| #8 journal evaluation | IN_PROGRESS | Freeze broader, temporally spaced cases and meaningful uncertainty controls; preserve test separation. Cross-profile comparison now requires explicit cases/common-case audit. |
| #20 memory/resource acceptance | engineering DONE on local 2×3090; multi-seed comparison DONE at `7dfbea6`, science open | Multi-seed paired comparison on real 17-channel ERA5: 72/72 cells, 3 seeds, both tricks, at [R7_GPU_MULTISEED.md](R7_GPU_MULTISEED.md). Streamed stays flat in K (−393 MiB at K=8 vs full BPTT, all seeds agreeing) but is slower at every K; checkpointing cuts 51.8–54.5 % of peak for +25–38 ms. Memory reproduced bit-identically in all 72 cells across two runs; step time did not. No rental used and no fabricated measurement. |
| #9 finer-resolution expert | BLOCKED, scientific labels/core gates | Need real co-located finer-resolution dynamic targets; never interpolated truth. |
| #1 / PR12 scientific release | IN_PROGRESS | Hypotheses unproven; PR remains Draft. Bounded CPU engineering success is not final SOTA. |

## Execution discipline

TODO -> IN_PROGRESS -> VERIFY -> DONE; BLOCKED names a real dependency.
During each independent CI/CPU workflow, advance another issue and return to
actual results. Close only after acceptance. Artifact/publication review is useful
work, but don't add speculative modules merely to avoid saying an experiment
has mixed results. No background continuation or re-enabled timer.

The planned controls53–58 have completed, including repairs and provenance
audits. Remaining scientific questions are not all external blockers; any new
study needs its own fixed hypothesis, bounded budget and case selection before
execution, rather than extending the completed endpoints until a desired win.
