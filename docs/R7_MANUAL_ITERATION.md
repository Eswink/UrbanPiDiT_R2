# R7 conversation-driven issue workflow

User decision (2026-09-23): stop scheduled UrbanPiDiT iteration. Continue only
within user-triggered conversations. The R7 automation has been disabled; do not
recreate or enable it without a new explicit request. Unrelated project tasks
are outside this decision.

## Productive work while CI runs

Read the current branch ref, relevant issue, source and existing CI before acting.
Keep a small queue of independent engineering tasks. After submitting a coherent
atomic patch for task A, work on task B's source review, implementation, tests or
documentation while A's workflow runs. Do not repeatedly poll or idle waiting.
Do not make task B depend on unverified behavior in A. Avoid editing the same
files concurrently. Recheck the remote ref before publishing; no force push.

When returning to A, bind verification to the exact SHA. If A fails, fix the
actual failure without weakening scientific/tests requirements. A cancelled or
queued run is not a pass. A later passing combined commit can validate a patch
only after confirming it contains that patch unchanged or its explicit repair.
Avoid repeatedly publishing half-finished file-by-file updates just to occupy CI.

## Closure and reporting

Close an engineering child only after its acceptance tests pass. Keep its parent
open when real-data experiments, hardware benchmarks or research comparisons
remain. Comments should contain actual commits, exact-SHA workflow links,
pass/skip counts and unresolved limitations, not repeated status placeholders.
A conversation may finish with pending CI honestly recorded; no implied
background work after the turn ends.

Work stays on r7/weather-reasoning; PR #12 was merged on 2026-09-25 (see the
push-vs-merge addendum below). No automatic main merge or
release, paid GPU rental, large data download/training, or destructive data
operation without explicit authorization. Never present synthetic fixtures as
weather truth, desired SOTA outcomes as measured results, or reasoning-step
counts as wall-clock speedups. Preserve users' changes and existing assignees.

## Addendum (2026-09-25, decision 0003): push vs merge

Fast-forward pushes of the working branch to `main` are authorized and are how
closing keywords take effect (`guard_destructive_git` now allows non-force
pushes; it still denies force variants, default-branch deletion, `--mirror`,
and merges involving `main`). Merges and releases still require explicit user
authorization and are executed by the user outside ZCode. This addendum does
not weaken any sentence above: no force push, no synthetic truth, no
reasoning-step-count speedups, cancelled runs are not passes.
