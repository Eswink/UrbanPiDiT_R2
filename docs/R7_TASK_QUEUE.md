# R7 conversation-driven task queue

Updated after verified issue #13 regional acquisition at `f37ceba`, the #20
multi-seed comparison at `7dfbea6`, the #5 budget-parity audit at `bf57fd4`, the #6 fair-budget comparison at `1bc1eef`, the #7 halting-gate audit at `0f5df17`, the #8 rollout tables at `b7f45ab`, the #9 access audit at `6dc388d`, and the #1 cumulative gate status at `ffe501e`;
on 2026-09-25 the eight verdicts were finalized, the closing commit carrying
`Closes #N` for all eight was pushed on `r7/weather-reasoning` (`e5be0c0`), and
after the hook policy change (decision 0003) the fast-forward to `main`
**landed**: `92a8c4d..2d7da05` at 2026-09-25T10:11Z — **all eight issues are
CLOSED on GitHub** (#13/#20/#5/#6/#7/#8/#9/#1, closed 10:11:00–10:11:03Z) and
**PR #12 shows merged** (fast-forward, no merge commit). Merges remain gated
by `guard_destructive_git` (user authorization required); non-force pushes to
main are allowed by decision 0003. No force, no release, no paid GPU, no
uncontrolled data mirror. Experimental workflows stay tag-gated and skipped
by design.

## Accepted work in this iteration

Corrections recorded under #62 (2026-09-26): the #8 close sentence in the
historical table below was stale (the issue closed with the 2026-09-25 main
fast-forward) and now reads CLOSED; wording errors in
R7_ROLLOUT_TABLES.md, R7_COREASONING_FAIR_BUDGET.md and R7_GPU_BRINGUP.md
are fixed in place with per-file correction logs; no measured value changed.


| Issue | State | Verification |
| --- | --- | --- |
| #53 variable-wise error/update geometry | DONE | actual35891600411; CI35891600361 |
| #54 fixed200->800 original-checkpoint control | DONE | actual35892145587; CI35892145656 |
| #55 same-budget continuous250day source | DONE | actual35893397412; test-only repairCI35894113212 |
| #56 pinned offline continuous-data800update control | DONE | actual35895446235; CI35895446093 |
| #57 optional spatial solver plus400update matched ablation | DONE (engineering/experiment) | actual35896740916; CI35896740912; defaultFalse retained |
| #58 explicit case selection/common-case retrospective audit | DONE | actual35898091953; CI35898091928 |
| #60 seed-identity comparator, fail-closed gates, historical re-aggregation (S0-A) | DONE (acceptance chain) | local 909 passed/3 skipped; conventions 34/0; audit 27/27 records, max mean diff 0.0, 0/34 directions changed; see R7_SEED_IDENTITY_COMPARATOR.md |
| #61 DDP verified reasoning depth, resume contract, padded samplers (S0-B) | DONE (CPU+2×3090; real-region throughput BLOCKED on #63 D1) | local 939 passed/3 skipped; CUDA: eval K observed 4 with steps=10, resume 111/111 weight hashes identical, streamed+DDP refused; see R7_DDP_K_CONTRACT.md |
| #62 GPU audit pack, doc corrections, climatology skill baseline (S0-C) | DONE (publish pack = user decision) | local tests incl. 9 new (5 ACC skill + 4 audit pack); real pack 562 files/0 credentials; tables rebuilt from pack only (104 rows, 30 oom/skip flagged); retained_truncated memory cells added; see R7_GPU_AUDIT_PACK.md |

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
| #13 representative data | DONE verdict final; CLOSED on GitHub 2026-09-25T10:11Z (ff 92a8c4d..2d7da05, decision 0003) | Bounded regional acquisition completed at `f37ceba`: 3 years x 4 seasons, 65x65 East-Asia, 48 exact 6-hourly timestamps, 9 variables, source SHA256 `d3fa1fba6da46ed59a535ce27f7501813afc2c93cb8b745c90e349454a40960a`. Converted through the audited publication path to a 17-channel store (8 windows per split, `BUILD_COMPLETE.json`, train-only statistics). Coverage is four 24-hour blocks per year, NOT continuous full-year; see [R7_REGIONAL_ACQUISITION.md](R7_REGIONAL_ACQUISITION.md). |
| #5/#6 recurrence/process benefit | #5 DONE, #6 DONE (negative) verdicts final; CLOSED on GitHub 2026-09-25T10:11Z (ff 92a8c4d..2d7da05, decision 0003) | #5's parameter/FLOP parity is measured: +0.030% parameters and +0.0001–0.0003% forward FLOPs across K=1/2/4/6/8 on real ERA5 — [R7_BUDGET_PARITY.md](R7_BUDGET_PARITY.md). #6's fair-budget comparison is done and the gate is NOT met: at K=3, 15 of 17 deltas flip sign between seeds, the only established effects are t2m worsening (+0.30 K, all depths and seeds) and t500 improving under the no-feedback arm, and forecast feedback does not rescue it — [R7_COREASONING_FAIR_BUDGET.md](R7_COREASONING_FAIR_BUDGET.md). The earlier 'feedback hurts T500' framing did not reproduce. Keep defaults and all negatives. |
| #7 adaptive benefit | DONE (negative audit) verdict final; CLOSED on GitHub 2026-09-25T10:11Z (ff 92a8c4d..2d7da05, decision 0003) | Gate audited and found not satisfiable as stated: K=3 raises the equal-channel normalized objective 1.5–3.3% over K=0 (worse in all seeds for process_no_feedback), so fixed-Kmax is not the accuracy ceiling; and no shallower depth passes the per-variable tolerance at any tolerance ≤10% because `z250` stays 14–19% above the reference. This explains the earlier full-depth fallback. Controller thresholds were NOT changed and no savings are claimed — see [R7_HALTING_GATE_AUDIT.md](R7_HALTING_GATE_AUDIT.md). |
| #8 journal evaluation | acceptance DONE verdict final; CLOSED on GitHub via the 2026-09-25 main fast-forward; Pareto/complexity/extremes NOT done | `scripts/rollout_r7_metric_tables.py` produces paper-ready 6/12/24/48/72 h per-variable RMSE and ACC tables from frozen checkpoints without touching training code (AST-enforced), refusing to mix model generations or datasets and including a same-data persistence baseline — see [R7_ROLLOUT_TABLES.md](R7_ROLLOUT_TABLES.md). Explicitly NOT done: accuracy–compute Pareto, weather-complexity vs depth diagnostics, and extreme-event metrics. Test separation preserved (train2018/val2019/test2020). |
| #20 memory/resource acceptance | DONE verdict final; CLOSED on GitHub 2026-09-25T10:11Z (ff 92a8c4d..2d7da05, decision 0003) | Multi-seed paired comparison on real 17-channel ERA5: 72/72 cells, 3 seeds, both tricks, at [R7_GPU_MULTISEED.md](R7_GPU_MULTISEED.md). Streamed stays flat in K (−393 MiB at K=8 vs full BPTT, all seeds agreeing) but is slower at every K; checkpointing cuts 51.8–54.5 % of peak for +25–38 ms. Memory reproduced bit-identically in all 72 cells across two runs; step time did not. No rental used and no fabricated measurement. |
| #9 finer-resolution expert | BLOCKED verdict final with reopen condition recorded; CLOSED on GitHub 2026-09-25T10:11Z (ff 92a8c4d..2d7da05, decision 0003) | Access audited against live endpoints: HRRR is open but CONUS+Alaska only (not co-located with 107–123 °E); HRCLDAS exposes no key-free endpoint and the CMA portal is a registration route; SMBFD has no open download; WeatherBench2/ARCO contain no km-scale East-Asia product; NOAA PSL has only US km-scale products. **Reopen when** a key-free co-located high-resolution dynamic truth source becomes available: a CMA data-service account with HRCLDAS/CLDAS rights, an open km-scale East-Asia store, or a domain pivot to CONUS (open 3 km HRRR). No interpolation or synthetic truth produced — see [R7_URBAN_EXTENSION_BLOCKED.md](R7_URBAN_EXTENSION_BLOCKED.md). |
| #1 / PR12 scientific release | negative cumulative gate status final; CLOSED on GitHub 2026-09-25T10:11Z (ff 92a8c4d..2d7da05, decision 0003) | Cumulative G1–G4 status in [R7_ROADMAP_GATE_STATUS.md](R7_ROADMAP_GATE_STATUS.md): **no gate is positively supported** — G1/G2 not supported, G3 not satisfiable as stated, G4 shows the model losing to train-only climatology at 48 h. The five things the sequence DID establish are listed there. Verdict finalized as an answered negative-result set per the 2026-09-25 authorization; PR #12 will show as merged once the authorized fast-forward carries the branch tip to main (no merge commit, ff only); a release still requires explicit human authorization. |

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
