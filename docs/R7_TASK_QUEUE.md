# R7 conversation-driven task queue

Updated after verified issues53–58 at code commit
`a4d5eab9c0969169ae385e649dd2a3bbb1c56f63`.
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
| #13 representative data | IN_PROGRESS | Continuous temporal support exists; larger regional context and full representative coverage still require a bounded protocol. No blanket data-unavailable claim. |
| #5/#6 recurrence/process benefit | IN_PROGRESS | Three-seed controls show mixed outcomes. Spatial feedback helps some wind scores but worsens T500; process does not uniformly beat generic. Keep defaults and all negatives. |
| #7 adaptive benefit | IN_PROGRESS | Existing strict policies fell back to full depth. This iteration does not change controller thresholds or certify savings. Reassess only against an explicitly frozen new validation protocol. |
| #8 journal evaluation | IN_PROGRESS | Freeze broader, temporally spaced cases and meaningful uncertainty controls; preserve test separation. Cross-profile comparison now requires explicit cases/common-case audit. |
| #20 4090D resource acceptance | BLOCKED, hardware explicitly deferred | Existing profiler available when hardware returns. No rental or fabricated CUDA measurement. |
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
