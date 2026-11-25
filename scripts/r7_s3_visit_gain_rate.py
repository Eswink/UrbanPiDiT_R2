"""How the round measures its own speed (decision 0120).

Kept apart from the identity witness because it answers a different question: the witness asks
whether the gain switch is numerically inert, this asks how long an arm takes and under what host
conditions that answer is trustworthy. ``marginal_rate`` is the decision input; ``quiet_machine``
records the contention the probe ran under, because the attempt01 probe reported the host's
busyness rather than the round's speed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def marginal_rate(progress: Path, *, quarters: int = 4) -> dict:
    """The rate over the last 1/``quarters`` of a probe run, alongside the cumulative rate.

    Decision 0120 replaces the cumulative rate as the decision input. The reason is *not* that
    startup dominates a 400-update window -- measured on an idle machine, the four quarters of a
    400-update run were 1.075/1.098/1.110/1.109 s/update, i.e. flat from the start. It is that a
    cumulative rate folds any early disturbance into the answer permanently, while the marginal
    rate reports what the loader is actually sustaining now. Both are recorded so a disagreement
    between them is visible rather than silent.
    """
    rows = []
    progress = Path(progress)
    if progress.is_file():
        for line in progress.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    usable = [row for row in rows if isinstance(row.get("update"), int)
              and isinstance(row.get("elapsed_seconds"), (int, float))]
    if len(usable) < 2:
        return {"marginal_seconds_per_update": None, "cumulative_seconds_per_update": None,
                "quarter_size": 0, "reason": "fewer than two usable progress rows"}
    last = usable[-1]
    quarter = max(1, len(usable) // int(quarters))
    first = usable[-1 - quarter]
    delta_updates = last["update"] - first["update"]
    marginal = ((last["elapsed_seconds"] - first["elapsed_seconds"]) / delta_updates
                if delta_updates > 0 else None)
    cumulative = last["elapsed_seconds"] / last["update"] if last["update"] else None
    return {
        "marginal_seconds_per_update": (round(marginal, 4) if marginal is not None else None),
        "cumulative_seconds_per_update": (round(cumulative, 4) if cumulative is not None else None),
        "quarter_size": quarter,
        "quarters": int(quarters),
        "last_update": int(last["update"]),
        "marginal_over_cumulative": (round(marginal / cumulative, 4)
                                     if marginal is not None and cumulative else None),
    }


def quiet_machine() -> dict:
    """A read-only snapshot of the contention the probe is about to be measured under.

    Decision 0120 requires the probe to run with no other CPU or GPU load: the trainer's loader
    uses four workers, so a concurrent suite starves it and the measurement reports the host's
    busyness instead of the round's speed. This snapshot is recorded, not enforced -- an automatic
    threshold would itself decide whether the round may run, which is a criterion decision this
    round has no evidence for.
    """
    load = None
    try:
        load = [round(value, 3) for value in os.getloadavg()]
    except OSError:
        load = None
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count()
    return {"loadavg_1_5_15": load, "cpu_count": cpu_count,
            "load_per_cpu": (round(load[0] / cpu_count, 3)
                             if load and cpu_count else None)}


