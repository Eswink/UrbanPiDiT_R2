"""Shared registered-verdict readings for the S3 screening drivers.

Both the D4 lead-coverage screen and the update-budget screen decide with the
same rule text translated into the same two functions: a per-seed sign verdict
over declared primary cells (disagreement is unresolved and never averaged) and
a fail-closed gate pre-screen over declared variables (any positive
relative-MSE-change cell fails, tolerance 0.0). Keeping one implementation
avoids two copies drifting apart while both rounds claim to execute their own
frozen decision text.

Pure functions over already-read reading tables: no model, no data file, no
network. The caller supplies the frozen seeds/leads/variables so each round
binds its own protocol constants.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


def per_seed_sign_verdict(all_cells: Mapping[str, Mapping[str, Mapping[str, float]]],
                          *, seeds: Sequence[int], primary_leads: Sequence[int],
                          primary_variable: str) -> dict[str, Any]:
    """Per-seed sign agreement over the declared primary cells per lead.

    A lead reads ``supported`` when every declared seed agrees in sign and the
    candidate delta is negative; ``worsened`` when every seed agrees and the
    delta is positive; ``unresolved`` otherwise. Disagreement is never
    averaged. The overall verdict is supported only when every primary lead is
    supported; any worsened lead makes it worsened.
    """
    leads: dict[int, str] = {}
    for lead in primary_leads:
        signs = {}
        for seed in seeds:
            delta = all_cells[str(seed)][f"{lead}|{primary_variable}"]["rmse_delta"]
            signs[seed] = 0 if abs(delta) < 1e-12 else (1 if delta > 0 else -1)
        distinct = set(signs.values())
        if distinct == {-1}:
            leads[lead] = "supported"
        elif distinct == {1}:
            leads[lead] = "worsened"
        else:
            leads[lead] = "unresolved"
    if all(leads[lead] == "supported" for lead in primary_leads):
        overall = "supported"
    elif "worsened" in leads.values():
        overall = "worsened"
    elif all(value == "unresolved" for value in leads.values()):
        overall = "unresolved"
    else:
        overall = "unsupported"
    return {"per_lead": leads, "overall": overall,
            "supported_rule": ("every primary lead supported (all seeds same sign, "
                               "candidate lower)"),
            "no_averaging": "sign disagreement is unresolved and never averaged into a verdict"}


def gate_relative_mse_verdict(all_cells: Mapping[str, Mapping[str, Mapping[str, float]]],
                              *, seeds: Sequence[int], evaluation_leads: Sequence[int],
                              gate_variables: Iterable[str], tolerance: float) -> dict[str, Any]:
    """Fail-closed gate pre-screen: every relative-MSE-change cell must be <= tolerance.

    Zero tolerance is the project's frozen default; any positive cell is a
    failure and the candidate then does not advance even when the primary
    reads supported.
    """
    failures = []
    for seed in seeds:
        for lead in evaluation_leads:
            for variable in gate_variables:
                value = all_cells[str(seed)][f"{lead}|{variable}"]["relative_mse_change"]
                if value > tolerance:
                    failures.append({"seed": seed, "lead_hours": lead, "variable": variable,
                                     "relative_mse_change": value})
    variables = list(gate_variables)
    return {"passed": not failures, "tolerance": tolerance,
            "failures": failures,
            "rule": (f"{'/'.join(variables)} relative MSE change <= {tolerance} at every "
                     "lead/seed/full")}


def advance_decision(primary: Mapping[str, Any], gate: Mapping[str, Any]) -> str:
    """The registered go/no-go: both the primary and the gate pre-screen must pass."""
    return ("advance-to-S4-freeze" if (primary["overall"] == "supported" and gate["passed"])
            else "registered-negative")
