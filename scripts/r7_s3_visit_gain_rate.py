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
import time
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




def probe(output, body, seed, deadline, *, hooks) -> dict:
    """Measure both arms' rates at 400 updates, then refuse or accept from what was measured.

    ``hooks`` carries one callable per dependency the driver owns (``run_arm``,
    ``envelope_check``, ``write_exclusive``, ``sha256_path``) and one mapping of the frozen
    values this function must not re-derive (``settings``). Passing them as a single mapping
    keeps the signature inside R-023's eight-parameter target and, more usefully, makes the
    dependency list visible in one place.

    The decision input is the **marginal** rate over the last quarter of the probe (decision
    0120); the cumulative rate is recorded beside it so a disagreement is visible.
    """
    import torch

    run_arm = hooks["run_arm"]
    envelope_check = hooks["envelope_check"]
    write_exclusive = hooks["write_exclusive"]
    sha256_path = hooks["sha256_path"]
    settings = hooks["settings"]
    arms = settings["arms"]
    probe_updates = settings["probe_updates"]
    refusal_fraction = settings["refusal_fraction"]
    per_seed_seconds = settings["per_seed_seconds"]

    quiet = quiet_machine()
    rates = {}
    for arm in arms:
        ran = run_arm(output, body, seed, arm, probe_updates, deadline, "probe")
        folder = Path(output) / f"seed{seed}/probe/{arm}"
        measured = marginal_rate(folder / "progress.jsonl")
        decision_rate = measured["marginal_seconds_per_update"]
        if decision_rate is None:
            decision_rate = float(ran["report"]["elapsed_seconds"]) / probe_updates
        rates[arm] = {
            "visit_gain": ran["spec"]["visit_gain"],
            "elapsed_seconds": float(ran["report"]["elapsed_seconds"]),
            "updates": probe_updates,
            "decision_seconds_per_update": round(float(decision_rate), 4),
            "total_seconds_per_update": round(
                float(ran["report"]["elapsed_seconds"]) / probe_updates, 4),
            **measured,
            "checkpoint_sha256": sha256_path(ran["checkpoint"]),
            "parameters": sum(p.numel() for p in ran["model"].parameters()),
        }
        print(f"[seed {seed}] probe {arm}: {decision_rate:.4f} s/update marginal, "
              f"{rates[arm]['total_seconds_per_update']:.4f} cumulative", flush=True)
    treated = max(rates[arm]["decision_seconds_per_update"] for arm in arms)
    remaining = deadline - time.perf_counter()
    decision = envelope_check(probe_seconds_per_update=treated, updates=int(body["updates"]),
                              evaluation_seconds=settings["evaluation_seconds"],
                              remaining_seconds=remaining, arms_left=len(arms))
    record = {
        "scientific_claim": False, "test_read": False, "r": 0,
        "seed": int(seed), "probe_updates": probe_updates, "arms": rates,
        "estimator": "marginal over the last quarter of the probe; cumulative also recorded",
        "decision_seconds_per_update": round(treated, 4),
        "treated_seconds_per_update": round(treated, 4),
        "frozen_envelope_seconds": per_seed_seconds, "decision": decision,
        "quiet_machine": quiet,
        "owned_cuda_reserved_peak_bytes": int(torch.cuda.max_memory_reserved()),
        "limitations": ["a rate probe, not a reading: 400 updates of two arms, scored on nothing",
                        "the rate is only interpretable if no other CPU load ran during the "
                        "probe; quiet_machine records the loadavg at entry but does not enforce it",
                        "uniform_damped's rate is not measured separately from visit_damped's; the "
                        "frozen envelope carries 18 percent of headroom for the difference"],
    }
    write_exclusive(Path(output) / f"seed{seed}_probe.json", record)
    if not decision["permitted"]:
        raise RuntimeError(
            f"seed {seed}: the probe measured {treated:.4f} s/update (marginal), so the remaining "
            f"arms need {decision['need_seconds']} s of the {decision['remaining_seconds']} s left "
            f"({decision['fraction_of_remaining']} of it, over the {refusal_fraction} refusal "
            f"fraction). Refusing to start a run that cannot finish.")
    return record
