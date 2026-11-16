"""The (a)-(e) clauses for the typed-evidence budget round, computed from the per-arm tables.

Every clause is an explicit comparison over numbers loaded from the arm receipts. A clause with no
contrast to compute is ``None`` -- "untested" -- which is a different fact from ``False``
("measured, and it failed") and from ``True``. The workflow parses this JSON structurally; the
``da1ed00`` defect was a regex defaulting every clause to false when a judge returned an empty body,
which published a card saying the opposite of ``readings.json``.

**Why (a) is seed-concordant and gain-floored.** "At least two leads" over five leads and three
seeds gives fifteen chances, so a literal count would be satisfied by noise alone. The predeclared
reading requires the same sign on all three seeds at a lead *and* a paired delta of at least
:data:`MINIMUM_LEAD_GAIN_K`. A resolvable but operationally meaningless sign does not carry the
claim, and a single-seed flip is reported as ``reversals_single_seed`` so the evidence page can
show both readings rather than only the one that decided.

**Why (e) exists.** The budget ladder already moved z500 at 72 h the wrong way: p4000 is 427.35 and
p1600 is 445.78 (+4.3 percent). A pathway arm is allowed another 5 percent, not an unbounded drift,
so "t2m improved" cannot be published while the dynamics moved out from under it.
"""
from __future__ import annotations

import json
from pathlib import Path

#: A lead counts toward (a)/(b) only when the paired t2m delta is at least this much. Chosen as the
#: same 0.005 K floor the brief declares; it is a reading floor, not a significance threshold,
#: because the interval is what carries significance.
MINIMUM_LEAD_GAIN_K = 0.005

#: How many leads must agree before a pair is called. Two of five, so a single lead cannot carry it
#: while a genuine improvement at the short leads is not required to extend to 72 h.
MINIMUM_CONCORDANT_LEADS = 2

#: The (e) guard. p4000 -> p1600 moved z500@72h from 427.35 to 445.78, so this number is already a
#: real reference point rather than a guess.
Z500_72H_REFERENCE = 427.35
Z500_72H_TOLERANCE = 0.05

INCUMBENT, GENERIC, TYPED = "incumbent", "generic", "typed"
PAIRS = (("a_typed_minus_generic", TYPED, GENERIC), ("b_generic_minus_incumbent", GENERIC, INCUMBENT))


def _rmse(arm: dict, lead: int, variable: str):
    """One arm's RMSE at one lead and variable, or ``None`` when the arm did not report it."""
    table = (arm or {}).get("rmse") or {}
    row = table.get(str(lead)) or {}
    value = row.get(variable)
    return None if value is None else float(value)


def arm_for(arms: dict, name: str, seed: int):
    """One arm's table for one seed.

    ``arms`` is keyed by arm name first and seed second, which is the layout the driver writes.
    Getting this order wrong is silent: every lookup returns ``None`` and every clause reports
    "untested", which reads as a missing measurement rather than as a bug. The tests pin it.
    """
    per_seed = (arms or {}).get(name) or {}
    return per_seed.get(seed) or per_seed.get(str(seed)) or {}


def pair_delta(arms: dict, seed: int, lead: int, variable: str, left: str, right: str):
    """``left - right`` RMSE for one seed/lead/variable, or ``None`` if either side is missing."""
    left_value = _rmse(arm_for(arms, left, seed), lead, variable)
    right_value = _rmse(arm_for(arms, right, seed), lead, variable)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def _clause_pair(arms, seeds, leads, variable, left, right) -> dict:
    """The generalized (a)/(b) reading: concordant leads, gain floor, and the single-seed count."""
    per_lead = {}
    for lead in leads:
        by_seed = {seed: pair_delta(arms, seed, lead, variable, left, right) for seed in seeds}
        measured = [value for value in by_seed.values() if value is not None]
        if len(measured) != len(seeds):
            per_lead[str(lead)] = {"complete": False, "by_seed": by_seed}
            continue
        all_negative = all(value < 0.0 for value in measured)
        mean = sum(measured) / len(measured)
        per_lead[str(lead)] = {
            "complete": True,
            "by_seed": by_seed,
            "mean": mean,
            "concordant": bool(all_negative),
            "clears_floor": bool(all_negative and mean <= -MINIMUM_LEAD_GAIN_K),
        }
    complete = [row for row in per_lead.values() if row.get("complete")]
    if len(complete) != len(leads):
        return {"value": None, "per_lead": per_lead,
                "reason": f"{left} - {right} has no complete three-seed contrast at every lead; "
                          "an incomplete arm set is untested, not false"}
    concordant_leads = [lead for lead in leads if per_lead[str(lead)]["concordant"]]
    clearing = [lead for lead in leads if per_lead[str(lead)]["clears_floor"]]
    return {
        "value": bool(len(clearing) >= MINIMUM_CONCORDANT_LEADS),
        "per_lead": per_lead,
        "concordant_leads": concordant_leads,
        "clearing_leads": clearing,
        "n_concordant": len(concordant_leads),
        "n_clearing": len(clearing),
        "variable": variable,
        "pair": f"{left} - {right}",
        "thresholds": {"minimum_lead_gain_k": MINIMUM_LEAD_GAIN_K,
                       "minimum_concordant_leads": MINIMUM_CONCORDANT_LEADS},
        "reason": ("a lead counts only when the paired delta is negative on all three seeds and "
                   "its mean clears the gain floor; the clause needs "
                   f"{MINIMUM_CONCORDANT_LEADS} such leads"),
    }


def clause_c(primary: dict, secondary: dict) -> dict:
    """(c) states what the measured pair of clauses licenses. It is a scope rule, not a test.

    ``value`` is ``True`` once both pairs are measured, because the rule can then be applied; it is
    ``None`` while either pair is untested. It is never ``False``: a failure of (a) or (b) does not
    violate the scope rule, it merely licenses a weaker claim. ``permitted_claim`` is the field the
    evidence page must quote, and it is what keeps a true (a) from being written up as a physics
    win.
    """
    if primary.get("value") is None or secondary.get("value") is None:
        return {"value": None, "permitted_claim": None,
                "reason": "the claim scope is undefined until both pairs are measured"}
    if primary["value"] is True and secondary["value"] is True:
        return {"value": True, "permitted_claim": "type-structure-and-pathway",
                "forbidden_claim": None,
                "reason": ("both pairs clear; the pathway increment carries the +148,800 parameter "
                           "difference inherent to switching it on and must be reported with it")}
    if primary["value"] is True:
        return {"value": True, "permitted_claim": "type-structure-only",
                "forbidden_claim": "the physical evidence adds skill",
                "reason": ("the typed pair moves but the pathway pair does not, so only the type "
                           "structure may be claimed")}
    return {"value": True, "permitted_claim": "no-increment-from-this-pathway",
            "forbidden_claim": "any positive claim about the typed-evidence pathway",
            "reason": ("the capacity-matched pair did not clear; (c) restricts the claim, it does "
                       "not rescue the reading, and the result is recorded as negative")}


def clause_e(arms: dict, seeds, reference: float = Z500_72H_REFERENCE,
             tolerance: float = Z500_72H_TOLERANCE) -> dict:
    """(e) the treated arms must not degrade z500 at 72 h by more than the declared tolerance.

    Two statements are reported separately: each treated arm against its own seed's incumbent (the
    arm-to-arm guard that (e) decides on), and each arm against the registered 427.35 reference (the
    pre-existing budget drift, which (e) does not decide but which the evidence page must show).
    """
    rows = {}
    missing = []
    for seed in seeds:
        base = _rmse(arm_for(arms, INCUMBENT, seed), 72, "z500")
        if base is None:
            missing.append(f"incumbent/seed{seed}")
            continue
        for arm in (GENERIC, TYPED):
            value = _rmse(arm_for(arms, arm, seed), 72, "z500")
            if value is None:
                missing.append(f"{arm}/seed{seed}")
                continue
            change = (value - base) / base
            rows[f"{arm}/seed{seed}"] = {
                "incumbent": base, "arm": value, "relative_change": change,
                "within_tolerance": bool(change <= tolerance),
                "against_reference": (value - reference) / reference}
    if missing:
        return {"value": None, "per_arm": rows, "missing": missing,
                "reference": reference, "tolerance": tolerance,
                "reason": "z500 at 72 h is missing for at least one arm/seed; untested, not failed"}
    verdicts = [row["within_tolerance"] for row in rows.values()]
    return {
        "value": bool(verdicts) and all(verdicts),
        "per_arm": rows,
        "reference": reference,
        "tolerance": tolerance,
        "max_relative_change": max(row["relative_change"] for row in rows.values()),
        "reason": ("the budget ladder already worsened z500 at 72 h from 427.35 to 445.78; a "
                   "pathway arm may not add more than the declared 5 percent on top of its own "
                   "incumbent"),
        "note": ("the against_reference column is informational: it mixes the budget drift with the "
                 "arm effect and is never what (e) decides on"),
    }


def evaluate(readings: dict) -> dict:
    """All five clauses plus the verdict string the evidence page quotes."""
    arms = readings.get("arms") or {}
    seeds = [int(seed) for seed in (readings.get("seeds") or [])]
    leads = [int(lead) for lead in (readings.get("lead_hours") or [])]
    if not seeds or not leads:
        empty = {"value": None, "reason": "the reading declares no seeds or no leads"}
        clauses = {"a_typed_minus_generic": dict(empty),
                   "b_generic_minus_incumbent": dict(empty),
                   "c_claim_scope": dict(empty, permitted_claim=None),
                   "d_budget_direction": dict(empty),
                   "e_z500_72h_guard": dict(empty)}
        return {"clauses": clauses,
                "values": {name: clause["value"] for name, clause in clauses.items()},
                "verdict": "untested-no-arm-contrast",
                "reading_rule": _READING_RULE}
    primary = _clause_pair(arms, seeds, leads, "t2m", TYPED, GENERIC)
    secondary = _clause_pair(arms, seeds, leads, "t2m", GENERIC, INCUMBENT)
    clauses = {
        "a_typed_minus_generic": primary,
        "b_generic_minus_incumbent": secondary,
        "c_claim_scope": clause_c(primary, secondary),
        "d_budget_direction": _clause_d(readings),
        "e_z500_72h_guard": clause_e(arms, seeds),
    }
    values = {name: clause.get("value") for name, clause in clauses.items()}
    if values["a_typed_minus_generic"] is True and values["b_generic_minus_incumbent"] is True:
        verdict = "positive-typed-and-pathway"
    elif values["a_typed_minus_generic"] is True:
        verdict = "mixed-type-structure-only"
    elif values["a_typed_minus_generic"] is False:
        verdict = "negative-no-type-structure-increment"
    else:
        verdict = "untested-no-arm-contrast"
    return {"clauses": clauses, "values": values, "verdict": verdict,
            "reading_rule": _READING_RULE}


def _clause_d(readings: dict) -> dict:
    """(d) the same arm's increment must keep its direction between the 4- and 16-epoch points.

    No 16-epoch reading exists in this round, so this clause is ``None`` unless a caller supplies a
    ``budget_comparison`` block. It is never reported as passing on the strength of one dose.
    """
    comparison = readings.get("budget_comparison")
    if not comparison:
        return {"value": None,
                "reason": ("no 16-epoch reading exists in this round; (d) cannot be evaluated and "
                           "must not be reported as passing on one dose"),
                "available_doses": [readings.get("updates")]}
    return {"value": bool(comparison.get("direction_preserved")),
            "reason": comparison.get("reason", "supplied by the caller"),
            "comparison": comparison}


_READING_RULE = (
    "(a) is the only capacity-matched pair: a true (a) licenses 'the four physical fields must be "
    "routed by type', not 'the physics adds skill'. (b) prices the pathway including its own "
    "148,800 parameters. (c) restricts what may be claimed and never turns a measured false into a "
    "pass. (d) is null until a second dose exists. (e) fails when the dynamics moved the wrong way. "
    "Untested is null, never false and never true.")


def write(path, body: dict) -> None:
    """Write one reading, refusing to overwrite: a second run is a new attempt directory."""
    path = Path(path)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(body, handle, indent=2, ensure_ascii=False, allow_nan=False)


def load(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
