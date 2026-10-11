"""Stage liveness: is a worker for this stage still running? (interruption hardening)

A stage is proved finished by its receipt and nothing else, but "no receipt" does not
mean "no work in flight". When an orchestrator is stopped mid-stage its worker can
outlive it, so resuming the round would spawn a second worker onto the same attempt
directory and the two would fight over the GPU and the output tree.

Everything here is **read-only**: it reads ``/proc/<pid>/cmdline`` and never signals a
process. Split from ``tools/r7_round_support.py`` to keep that module inside its
400-line target (R-021) and to keep ``run_stage`` at its 8-parameter budget (R-023).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

#: The proc filesystem the scan reads. A module attribute rather than a parameter,
#: because ``run_stage`` already carries the R-023 budget of 8 parameters; tests point
#: it at a synthetic tree.
PROC_ROOT = Path("/proc")

WORKER_FLAG = "--worker"
OUTPUT_FLAG = "--out"
SEED_FLAG = "--seed"


def _worker_argv_matches(
    tokens: Sequence[str],
    driver_rel: str,
    phase: str,
    output: Path,
    seed: int | None,
) -> bool:
    """Whether a command line is exactly the worker this stage would spawn."""
    if not tokens or not any(token.endswith(driver_rel) for token in tokens):
        return False

    def value_after(flag: str) -> str | None:
        if flag not in tokens:
            return None
        index = tokens.index(flag)
        return tokens[index + 1] if index + 1 < len(tokens) else None

    if value_after(WORKER_FLAG) != phase:
        return False
    out = value_after(OUTPUT_FLAG)
    if out is None or Path(out).resolve() != Path(output).resolve():
        return False
    return value_after(SEED_FLAG) == (None if seed is None else str(seed))


def running_worker_pids(
    driver_rel: str,
    phase: str,
    output: Path,
    seed: int | None = None,
    *,
    proc_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Live processes already running this exact stage — a read-only ``/proc`` scan.

    The match is the full argv shape the stage wrapper builds, so another seed, another
    attempt directory or another driver never matches. Nothing is signalled: a process
    that cannot be read is skipped, not touched.
    """
    base = Path(proc_root) if proc_root is not None else PROC_ROOT
    if not base.is_dir():
        return []
    found = []
    for entry in sorted(base.iterdir(), key=lambda path: path.name):
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        tokens = [token for token in raw.decode("utf-8", "replace").split("\0") if token]
        if _worker_argv_matches(tokens, driver_rel, phase, output, seed):
            found.append({"pid": int(entry.name), "cmdline": tokens})
    return found


def stage_liveness(
    spec: dict[str, Any],
    attempt: Path,
    *,
    proc_root: Path | None = None,
) -> dict[str, Any]:
    """Read-only: which of this round's stages still have a live worker.

    The operator's view after an interruption, and the answer to "may I re-run this
    stage, or is the orphan still training?".
    """
    attempt = Path(attempt)
    stages: list[dict[str, Any]] = [
        {"phase": phase, "seed": None} for phase in ("prepare", "archive", "reading")
    ]
    stages += [{"phase": "seed", "seed": seed} for seed in spec.get("seeds", ())]
    rows = []
    for stage in stages:
        live = running_worker_pids(
            spec["driver"], stage["phase"], attempt, stage["seed"], proc_root=proc_root
        )
        rows.append({**stage, "running": bool(live), "pids": [item["pid"] for item in live]})
    return {
        "attempt_dir": str(attempt),
        "stages": rows,
        "running": [row for row in rows if row["running"]],
    }


def measured_peak_bytes(output: Path) -> int:
    """The largest measured CUDA reserved peak among this attempt's finished seeds.

    The co-residency gate feeds this in beside the protocol's own ``estimated_peak_bytes``,
    which is only a freeze-time guess: a finished seed's receipt carries what the run
    actually reserved. Using the larger of the two is what keeps the gate from admitting
    a second worker onto a card the first one has already filled.
    """
    peaks = []
    for path in Path(output).glob("seed*_receipt.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get("owned_cuda_reserved_peak_bytes")
        except (OSError, ValueError):
            continue
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            peaks.append(value)
    return max(peaks, default=0)
