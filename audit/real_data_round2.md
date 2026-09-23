# Real Data Audit Round 2 — Data Contract, Leakage & Preprocessing

## Pre-review finding: 86/100 — FAIL

The first implementation split already-windowed samples chronologically. Adjacent sliding windows on different splits could still share raw hours, creating subtle temporal leakage.

## Rework

- Changed policy to **split raw hourly rows first, build windows inside each split second**.
- Current raw source-row ranges are disjoint:
  - train `0–50`
  - val `51–61`
  - test `62–72`
- Added dedicated tests for:
  - dynamic/static checksums;
  - split source-row and timestamp non-overlap;
  - exact first-row normalization;
  - finite tensors and contract shapes;
  - nontrivial real district mask/edge;
  - provenance flags;
  - fixture byte-identical copy;
  - network failure never silently synthesizes data;
  - deterministic WorldCover tile/URL construction.

## Final score: 97/100 — PASS

### Evidence

`pytest -q` after rework/final rerun: **15 tests passed**.

The fixture is still intentionally not scientific training data; that is a declared scope constraint, not a hidden defect.
