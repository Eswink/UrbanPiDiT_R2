"""The family verdict for the in-protocol round: a mechanism-partitioned simultaneous test.

Three design decisions carry this module.

**The family is partitioned by mechanism, not by observation.** ``t500,t850,u850,v850`` are the
channels :func:`typed_local_fields` consumes; the model declares that set. Every other channel forms
its own family. The partition therefore cannot be redrawn once a reading exists, which is what stops
this module from grading whichever variable looks best. The previous round graded ``t2m`` against a
0.005 K absolute floor and reported ``False`` while z500 moved 13 of 15 seed-by-lead cells with an
interval excluding zero -- the criterion did not measure the wrong thing, it measured one variable
under a threshold that only has meaning for temperature.

**The metric is a skill delta, so 17 variables are comparable.** For case i, lead L and channel c::

    delta_i = (mse_control_i - mse_treatment_i) / D[L, c]

where ``D`` is that cell's aggregate climatology MSE, read from ``climatology_skill.csv``. ``D`` is
arm-independent and fixed, so dividing every case by it is a monotone rescaling: it cannot change a
sign or the sign of an interval, and it makes K, m/s, Pa, kg/kg and m**2 s**-2 all dimensionless
fractions of an error the model is trying to beat. The sign convention is fixed: **positive means the
treatment is better**.

**A family with no complete contrast is ``None``, never ``False``.** ``False`` means measured and
failed; ``None`` means untested. Collapsing the two is the ``da1ed00`` defect that published a card
saying the opposite of ``readings.json``.

The uncertainty comes from ``training/r7_simultaneous_stats.simultaneous_paired_interval``, whose
max-statistic construction gives one critical value covering the whole family. That function requires
every cell in a family to share an identical case-key set, which the validation split does *not*
satisfy across leads: each lead evaluates a slightly different cohort (1,458 down to 1,447) because
the rollout needs a full horizon. The families are therefore built on the intersection of the case
keys shared by all five leads, and every family reports how many cases that removed.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

#: The floor, dimensionless: a tenth of one percent of climatology MSE. Resolvable-but-trivial signs
#: must not carry a family; see the protocol's ``thresholds`` block for the frozen value.
MINIMUM_CELL_GAIN_SKILL = 0.001
MINIMUM_CELLS_PER_FAMILY = 1
MINIMUM_LEADS_PER_FAMILY = 2
MINIMUM_FAMILY_CASES = 400

INCUMBENT, GENERIC, TYPED = "incumbent", "generic", "typed"
PAIRS = (("a_typed_minus_generic", TYPED, GENERIC),
         ("b_generic_minus_incumbent", GENERIC, INCUMBENT))


def _protocol():
    """Late import: the protocol owns the family labels and the frozen thresholds."""
    from scripts import r7_s3_in_protocol_protocol as proto
    return proto


def arm_for(arms: dict, name: str, seed: int):
    """One arm's table for one seed.

    ``arms`` is keyed by arm name first and seed second, which is the layout the driver writes.
    Getting this order wrong is silent: every lookup returns ``None`` and every clause reports
    "untested", which reads as a missing measurement rather than as a bug. The tests pin the order.
    """
    per_seed = (arms or {}).get(name) or {}
    return per_seed.get(seed) or per_seed.get(str(seed)) or {}


def _lead_dir(arm: dict, lead: int):
    """The evaluation directory for one lead, from the arm row the driver writes."""
    dirs = (arm or {}).get("evaluation_dirs") or {}
    return dirs.get(str(lead)) or dirs.get(lead)


def load_provenance(directory) -> dict | None:
    """One lead's ``provenance.json``, or ``None`` when the directory did not survive."""
    if not directory:
        return None
    path = Path(directory) / "provenance.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def per_case_mse(provenance: dict, *, channels: list[str], lead_hours: int) -> dict:
    """``{init_time: {channel: mse}}`` for one lead, refusing shapes that cannot be paired.

    ``provenance["initializations"][i]["mse"]`` is ``[lead][channel]``; the file is written with a
    single lead per call, so that axis has length one. Reading the wrong axis would silently compare
    t2m against u10, so the shape is asserted rather than assumed.
    """
    rows = {}
    for row in provenance.get("initializations") or []:
        mse = row.get("mse")
        if not isinstance(mse, list) or len(mse) != 1:
            raise ValueError(f"lead {lead_hours}: expected one lead per provenance row, got "
                             f"{None if not isinstance(mse, list) else len(mse)}")
        values = mse[0]
        if not isinstance(values, list) or len(values) != len(channels):
            raise ValueError(f"lead {lead_hours}: mse row has "
                             f"{None if not isinstance(values, list) else len(values)} channels, "
                             f"the family declares {len(channels)}")
        rows[str(row["init_time"])] = {name: float(value) for name, value in zip(channels, values)}
    return rows


def climatology_denominators(directory, *, channels: list[str], lead_hours: int) -> dict:
    """``{channel: climatology mse}`` from the lead's own ``climatology_skill.csv``.

    ``rmse_climatology`` is squared rather than re-derived from ``mse_skill`` so the denominator does
    not depend on the forecast it is meant to normalise. The table is per lead and reflects exactly
    the cases that lead scored, which is why it cannot be lifted to the intersection: it is a scale,
    not a paired quantity, and the interval's width scales with it uniformly.
    """
    path = Path(directory) / "climatology_skill.csv"
    if not path.is_file():
        raise FileNotFoundError(f"lead {lead_hours}: no climatology_skill.csv beside the provenance")
    found = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row["variable"]) not in channels:
                continue
            value = float(row["rmse_climatology"])
            found[str(row["variable"])] = value * value
    missing = [name for name in channels if name not in found]
    if missing:
        raise RuntimeError(f"lead {lead_hours}: climatology omits {missing}")
    for name, value in found.items():
        if not value > 0.0:
            raise RuntimeError(f"lead {lead_hours}: nonpositive climatology mse for {name}")
    return found


def _shared_keys(*tables) -> list[str]:
    """The case keys present in every supplied mapping, sorted for a stable resample order."""
    if not tables:
        return []
    keys = set(tables[0])
    for table in tables[1:]:
        keys &= set(table)
    return sorted(keys)


def _cell_series(arms, seeds, leads, channel, left, right, *, channel_order):
    """Per-case skill deltas for one (lead, channel) cell, seed-averaged, plus what is missing.

    The three seeds are averaged **per case**, so the cell still carries one observation per
    initialization: concatenating the seeds instead would count the same case three times and shrink
    every interval by roughly sqrt(3). The per-seed means are returned separately because the
    three-seed sign rule is applied to them, not to the average.
    """
    per_lead = {}
    missing = []
    for lead in leads:
        rows_for_lead = {}
        for seed in seeds:
            left_prov = load_provenance(_lead_dir(arm_for(arms, left, seed), lead))
            right_prov = load_provenance(_lead_dir(arm_for(arms, right, seed), lead))
            if left_prov is None or right_prov is None:
                missing.append(f"{lead}h/seed{seed}")
                continue
            left_rows = per_case_mse(left_prov, channels=channel_order, lead_hours=lead)
            right_rows = per_case_mse(right_prov, channels=channel_order, lead_hours=lead)
            shared = _shared_keys(left_rows, right_rows)
            if not shared:
                missing.append(f"{lead}h/seed{seed}: no shared initialization")
                continue
            rows_for_lead[seed] = (left_rows, right_rows, shared)
        if not rows_for_lead:
            continue
        keys = _shared_keys(*[set(shared) for _, _, shared in rows_for_lead.values()])
        if not keys:
            missing.append(f"{lead}h: the seeds share no initialization")
            continue
        denominator = _denominator_for(arm_for(arms, left, seeds[0]), lead, channel)
        if denominator is None:
            missing.append(f"{lead}h: no climatology denominator for {channel}")
            continue
        per_case = {}
        for seed, (left_rows, right_rows, _) in rows_for_lead.items():
            # Positive means the left arm has the lower MSE, i.e. it is the better arm.
            per_case[seed] = [(right_rows[key][channel] - left_rows[key][channel]) / denominator
                              for key in keys]
        means = {seed: sum(values) / len(values) for seed, values in per_case.items()}
        averaged = [sum(per_case[seed][position] for seed in per_case) / len(per_case)
                    for position in range(len(keys))]
        per_lead[str(lead)] = {"keys": keys, "deltas": averaged, "by_seed": means}
    return per_lead, missing


def _denominator_for(arm: dict, lead: int, channel: str):
    """The climatology MSE for one lead and channel, from the arm's cached table."""
    table = (arm or {}).get("climatology_mse") or {}
    row = table.get(str(lead)) or {}
    value = row.get(channel)
    return None if value is None else float(value)


def family_cells(arms, seeds, leads, channels, left, right, *, channel_order) -> dict:
    """Every cell of one family, keyed ``"<lead>h|<channel>"``, plus what could not be built."""
    cells = {}
    missing = []
    for channel in channels:
        by_lead, absent = _cell_series(arms, seeds, leads, channel, left, right,
                                       channel_order=channel_order)
        missing.extend(absent)
        for lead_key, row in by_lead.items():
            cells[f"{lead_key}h|{channel}"] = row
    return {"cells": cells, "missing": sorted(set(missing))}


def build_family(cells: dict, *, block_length: int, resamples: int, seed: int, alpha: float,
                 minimum_cases: int = MINIMUM_FAMILY_CASES) -> dict:
    """One simultaneous interval over a family, on the case set all of its cells share.

    The instrument demands one identical key set per family, so the cells are intersected first. The
    number of dropped cases is reported: a family silently built on the short-lead cohort only would
    verify a different claim from the one the protocol froze.
    """
    from training.r7_simultaneous_stats import simultaneous_paired_interval

    if not cells:
        return {"status": "insufficient-evidence", "reason": "the family has no cells"}
    union = sorted({key for row in cells.values() for key in row["keys"]})
    shared = _shared_keys(*[set(row["keys"]) for row in cells.values()])
    if len(shared) < minimum_cases:
        return {"status": "insufficient-evidence",
                "reason": f"the family's shared case set is {len(shared)}, below the declared "
                          f"minimum of {minimum_cases}",
                "n_shared_cases": len(shared), "n_union_cases": len(union)}
    aligned = {}
    for cell, row in cells.items():
        lookup = dict(zip(row["keys"], row["deltas"]))
        aligned[cell] = (shared, [lookup[key] for key in shared])
    outcome = simultaneous_paired_interval(aligned, block_length=block_length,
                                           resamples=resamples, seed=seed, alpha=alpha)
    outcome["n_union_cases"] = len(union)
    outcome["n_dropped_cases"] = len(union) - len(shared)
    outcome["case_set_rule"] = ("each lead scores a slightly different cohort because the rollout "
                               "needs a full horizon; the family is built on the intersection")
    return outcome


def clause_family(arms: dict, seeds, leads, channels, left: str, right: str, *,
                  block_length: int, resamples: int, resample_seed: int, alpha: float,
                  channel_order, floor: float = MINIMUM_CELL_GAIN_SKILL,
                  minimum_cells: int = MINIMUM_CELLS_PER_FAMILY,
                  minimum_leads: int = MINIMUM_LEADS_PER_FAMILY) -> dict:
    """The generalised (a)/(b): simultaneous intervals, a gain floor, and a three-seed sign rule.

    Each conjunct is reported separately. A cell carries the family only when its simultaneous
    interval excludes zero in the improving direction, its difference clears the dimensionless floor,
    and all three seeds agree in sign; the family needs such cells spanning at least two leads. The
    interval carries the significance, the floor keeps a trivial sign from carrying the claim, and
    the sign rule keeps one seed's flip from passing as a mechanism.
    """
    collected = family_cells(arms, seeds, leads, channels, left, right,
                             channel_order=channel_order)
    cells = collected["cells"]
    pair = f"{left} - {right}"
    common = {"pair": pair, "sign_convention": "positive means the left arm has lower MSE",
              "family_channels": list(channels), "missing": collected["missing"],
              "floor": floor, "minimum_cells": minimum_cells, "minimum_leads": minimum_leads}
    if not cells:
        return {**common, "value": None, "n_clearing": 0, "n_concordant": 0,
                "reason": ("no cell of this family has a complete provenance pair; an arm that was "
                           "never scored is untested, not false")}
    complete = {cell: row for cell, row in cells.items() if len(row["by_seed"]) == len(seeds)}
    incomplete = sorted(set(cells) - set(complete))
    if not complete:
        return {**common, "value": None, "n_clearing": 0, "n_concordant": 0,
                "incomplete_cells": incomplete,
                "reason": (f"no cell has all {len(seeds)} seeds; untested rather than failed")}
    outcome = build_family(complete, block_length=block_length, resamples=resamples,
                           seed=resample_seed, alpha=alpha)
    if outcome.get("status") != "ok":
        return {**common, "value": None, "family": outcome, "cells": {},
                "incomplete_cells": incomplete, "n_clearing": 0, "n_concordant": 0,
                "reason": ("the family could not be given a simultaneous interval; untested rather "
                           "than failed")}
    report = {}
    for cell, row in outcome["cells"].items():
        means = complete[cell]["by_seed"]
        report[cell] = {
            "difference": row["difference"],
            "simultaneous_low": row["simultaneous_low"],
            "simultaneous_high": row["simultaneous_high"],
            "excludes_zero": row["excludes_zero"],
            "by_seed": means,
            "concordant": bool(all(value > 0.0 for value in means.values())),
            "clears_floor": bool(row["difference"] is not None and row["difference"] >= floor),
            "improving": bool(row["difference"] is not None and row["difference"] > 0.0),
        }
    clearing = sorted(cell for cell, row in report.items()
                      if row["excludes_zero"] is True and row["clears_floor"] and row["concordant"])
    clearing_leads = sorted({int(cell.split("|")[0].rstrip("h")) for cell in clearing})
    value = bool(len(clearing) >= minimum_cells and len(clearing_leads) >= minimum_leads)
    return {
        **common, "value": value, "cells": report, "incomplete_cells": incomplete,
        "n_cells": len(report), "n_clearing": len(clearing), "clearing_cells": clearing,
        "clearing_leads": clearing_leads, "n_concordant": len(clearing),
        "family": {key: outcome.get(key) for key in
                   ("status", "critical_value", "n_cases", "block_length", "n_blocks",
                    "n_union_cases", "n_dropped_cases", "resamples", "resample_seed", "alpha")},
        "reason": (f"a cell carries the family only when its simultaneous interval excludes zero in "
                   f"the improving direction, its difference clears the {floor} floor and all three "
                   f"seeds agree in sign; the family needs {minimum_cells} such cell(s) spanning "
                   f"{minimum_leads} lead(s)"),
    }


def worry_fraction(arms, seeds, leads, channels, left: str, right: str, *,
                  channel_order) -> dict:
    """How often ``left`` scores worse than ``right`` across every declared cell.

    A single lead can improve while the arm loses broadly elsewhere; the guard needs that count, and
    it must come from the same per-case rows rather than from a second pass over the CSVs.
    """
    collected = family_cells(arms, seeds, leads, channels, left, right,
                             channel_order=channel_order)
    worse = better = 0
    per_channel = {}
    for cell, row in collected["cells"].items():
        for seed, value in row["by_seed"].items():
            wore = value < 0.0
            worse += int(wore)
            better += int(not wore)
            entry = per_channel.setdefault(cell.split("|")[1], {"worse": 0, "better": 0})
            entry["worse" if wore else "better"] += 1
    total = worse + better
    return {"n_worse": worse, "n_better": better,
            "worse_fraction": (worse / total) if total else None,
            "per_channel": per_channel,
            "missing": collected["missing"]}


def clause_guard(arms: dict, seeds, leads, channels, *, tolerance: float,
                 max_worse_fraction: float, channel_order) -> dict:
    """The dynamics guard: z500 at 72 h may not worsen much, and no arm may lose broadly.

    The ladder already moved z500@72h the wrong way between p4000 and p1600, so this clause can fail
    for a reason unrelated to the pathway; when it does, the evidence page must say that rather than
    attributing the failure to the mechanism.
    """
    z500 = {}
    missing = []
    for seed in seeds:
        base_dir = _lead_dir(arm_for(arms, INCUMBENT, seed), 72)
        base_prov = load_provenance(base_dir)
        if base_prov is None:
            missing.append(f"incumbent/seed{seed}")
            continue
        base_rows = per_case_mse(base_prov, channels=channel_order, lead_hours=72)
        denominator = climatology_denominators(base_dir, channels=["z500"], lead_hours=72)["z500"]
        for arm in (GENERIC, TYPED):
            directory = _lead_dir(arm_for(arms, arm, seed), 72)
            arm_prov = load_provenance(directory)
            if arm_prov is None:
                missing.append(f"{arm}/seed{seed}")
                continue
            arm_rows = per_case_mse(arm_prov, channels=channel_order, lead_hours=72)
            shared = _shared_keys(base_rows, arm_rows)
            if not shared:
                missing.append(f"{arm}/seed{seed}: no shared initialization")
                continue
            base = sum(base_rows[key]["z500"] for key in shared) / len(shared)
            value = sum(arm_rows[key]["z500"] for key in shared) / len(shared)
            if base <= 0.0:
                missing.append(f"{arm}/seed{seed}: nonpositive incumbent z500 mse")
                continue
            change = (value - base) / base
            z500[f"{arm}/seed{seed}"] = {
                "incumbent_mse": base, "arm_mse": value, "relative_change": change,
                "within_tolerance": bool(change <= tolerance), "n_cases": len(shared),
                "skill_delta": (base - value) / denominator}
    broad = {}
    for arm in (GENERIC, TYPED):
        broad[arm] = worry_fraction(arms, seeds, leads, channels, arm, INCUMBENT,
                                    channel_order=channel_order)
    fractions = [entry["worse_fraction"] for entry in broad.values()
                 if entry["worse_fraction"] is not None]
    if missing or len(z500) != 2 * len(seeds):
        return {"value": None, "z500_72h": z500, "missing": missing, "broad": broad,
                "tolerance": tolerance, "max_worse_fraction": max_worse_fraction,
                "reason": "z500 at 72 h is missing for at least one arm/seed; untested, not failed"}
    z500_ok = all(row["within_tolerance"] for row in z500.values())
    broad_ok = bool(fractions) and all(value <= max_worse_fraction for value in fractions)
    return {
        "value": bool(z500_ok and broad_ok),
        "z500_72h": z500,
        "broad": broad,
        "z500_within_tolerance": bool(z500_ok),
        "broad_within_fraction": bool(broad_ok),
        "tolerance": tolerance,
        "max_worse_fraction": max_worse_fraction,
        "max_relative_change": max(row["relative_change"] for row in z500.values()),
        "reason": (f"no treated arm may worsen z500 at 72 h by more than {tolerance:.0%} against its "
                   f"own seed's incumbent, and no arm may score worse than the incumbent on more "
                   f"than {max_worse_fraction:.0%} of the declared cells"),
    }


def clause_c(primary: dict, secondary: dict) -> dict:
    """(c) states what the measured pair of families licenses. It is a scope rule, not a test.

    ``value`` is ``True`` once both families are measured, because the rule can then be applied; it
    is ``None`` while either is untested and never ``False``, since a failure of (a) or (b) does not
    violate the scope rule -- it licenses a weaker claim. ``permitted_claim`` is the field the
    evidence page must quote.
    """
    if primary.get("value") is None or secondary.get("value") is None:
        return {"value": None, "permitted_claim": None,
                "reason": "the claim scope is undefined until both families are measured"}
    if primary["value"] is True and secondary["value"] is True:
        return {"value": True, "permitted_claim": "type-structure-and-pathway",
                "forbidden_claim": None,
                "reason": ("both families clear; the pathway family carries the +148,800 parameter "
                           "difference inherent to switching it on and must be reported with it, so "
                           "it remains descriptive rather than attributable")}
    if primary["value"] is True:
        return {"value": True, "permitted_claim": "type-structure-only",
                "forbidden_claim": "the physical evidence adds skill",
                "reason": ("the capacity-matched family moves but the pathway family does not, so "
                           "only the type structure may be claimed")}
    return {"value": True, "permitted_claim": "no-increment-from-this-pathway",
            "forbidden_claim": "any positive claim about the typed-evidence pathway",
            "reason": ("the capacity-matched family did not clear; (c) restricts the claim, it does "
                       "not rescue the reading, and the result is recorded as negative")}


def clause_d(readings: dict) -> dict:
    """(d) the same arm's increment must keep its direction between two doses.

    No second dose exists in this round and the 16-epoch rung is not authorized, so this clause is
    ``None`` unless a caller supplies a ``budget_comparison`` block. It is never reported as passing
    on the strength of one dose.
    """
    comparison = readings.get("budget_comparison")
    if not comparison:
        return {"value": None,
                "reason": ("no second dose exists in this round and the 16-epoch rung is not "
                           "authorized; (d) cannot be evaluated and must not be reported as "
                           "passing on one dose"),
                "available_doses": [readings.get("updates")]}
    return {"value": bool(comparison.get("direction_preserved")),
            "reason": comparison.get("reason", "supplied by the caller"),
            "comparison": comparison}


_READING_RULE = (
    "Families are partitioned by mechanism: the four channels the pathway consumes form the primary "
    "family and every other channel forms its own, so no partition can be chosen after a reading "
    "exists. (a) is the capacity-matched pair C - B; a true (a) licenses 'the four physical fields "
    "must be routed by type', not 'the physics adds skill'. (b) prices the pathway including its own "
    "148,800 parameters, so it stays descriptive. (c) restricts what may be claimed and never turns "
    "a measured false into a pass. Untested is null, never false and never true.")


def evaluate(readings: dict) -> dict:
    """Both families for both pairs, plus the guard and the verdict string."""
    proto = _protocol()
    arms = readings.get("arms") or {}
    seeds = [int(seed) for seed in (readings.get("seeds") or [])]
    leads = [int(lead) for lead in (readings.get("lead_hours") or [])]
    families = readings.get("families") or proto.families()
    thresholds = readings.get("thresholds") or {}
    order = readings.get("channel_order") or proto.store_channels()
    floor = float(thresholds.get("minimum_cell_gain_skill", MINIMUM_CELL_GAIN_SKILL))
    alpha = float(thresholds.get("alpha_family", proto.ALPHA_FAMILY))
    block_length = int(thresholds.get("block_length_cases", proto.BLOCK_LENGTH_CASES))
    resamples = int(thresholds.get("resamples", proto.RESAMPLES))
    resample_seed = int(thresholds.get("resample_seed", proto.RESAMPLE_SEED))
    if not seeds or not leads or not arms:
        empty = {"value": None, "reason": "the reading declares no seeds, no leads or no arms"}
        clauses = {"a_typed_minus_generic": dict(empty),
                   "b_generic_minus_incumbent": dict(empty),
                   "c_claim_scope": dict(empty, permitted_claim=None),
                   "d_budget_direction": dict(empty),
                   "g_dynamics_guard": dict(empty)}
        return {"clauses": clauses,
                "values": {name: clause["value"] for name, clause in clauses.items()},
                "verdict": "untested-no-arm-contrast", "reading_rule": _READING_RULE}

    def per_family(left: str, right: str) -> dict:
        return {label: clause_family(arms, seeds, leads, list(body["channels"]), left, right,
                                     block_length=block_length, resamples=resamples,
                                     resample_seed=resample_seed, alpha=alpha,
                                     channel_order=order, floor=floor)
                for label, body in families.items()}

    primary_label = proto.FAMILY_PRIMARY_LABEL
    typed_vs_generic = per_family(TYPED, GENERIC)
    generic_vs_incumbent = per_family(GENERIC, INCUMBENT)
    primary = typed_vs_generic[primary_label]
    secondary = generic_vs_incumbent[primary_label]
    guard = clause_guard(arms, seeds, leads, order,
                         tolerance=float(thresholds.get("z500_72h_tolerance",
                                                        proto.Z500_72H_TOLERANCE)),
                         max_worse_fraction=float(thresholds.get("max_worse_cell_fraction",
                                                                proto.MAX_WORSE_CELL_FRACTION)),
                         channel_order=order)
    clauses = {
        "a_typed_minus_generic": primary,
        "b_generic_minus_incumbent": secondary,
        "c_claim_scope": clause_c(primary, secondary),
        "d_budget_direction": clause_d(readings),
        "g_dynamics_guard": guard,
        "by_family": {"typed_minus_generic": typed_vs_generic,
                      "generic_minus_incumbent": generic_vs_incumbent},
    }
    values = {name: clause.get("value") for name, clause in clauses.items() if name != "by_family"}
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


def write(path, body: dict) -> None:
    """Write one reading, refusing to overwrite: a second run is a new attempt directory."""
    path = Path(path)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(body, handle, indent=2, ensure_ascii=False, allow_nan=False)


def load(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
