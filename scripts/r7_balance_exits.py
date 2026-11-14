"""The (a)-(e) clauses, computed from the per-arm tables and nothing else.

Every clause is an explicit comparison over numbers loaded from the dumps. A clause with no
contrast to compute is ``None`` -- "untested" -- which is a different fact from ``False``
("measured, and it failed") and from ``True``. The workflow parses this JSON structurally; the
``da1ed00`` defect was a regex defaulting every clause to false when a judge returned an empty
body, which published a card saying the opposite of ``readings.json``.

**Two clauses could pass by luck at this sample size, so both are defined against a fluke.**

(b) says "at least one (lead, variable) where the better-RMSE arm is worse balanced". With 5
leads x several variables x arm pairs x 3 seeds there are hundreds of comparisons, so a literal
"at least one" would be satisfied by noise alone and would measure nothing. The predeclared
reading used here is **seed-consistent**: the reversal must hold at the same (family, lead,
variable) on all three seeds, which is the same stability standard (c) already imposes. The
looser single-comparison count is still reported as ``reversals_any_seed`` so the evidence page
can show both readings rather than only the one that decided.

(a) says the three arms' ``delta_HB`` medians are "consistently mutually distinct". Float noise
makes any two numbers unequal, so distinctness needs a predeclared floor
(:data:`ARM_DISTINCT_TOLERANCE`). It is deliberately *lenient* -- low enough that a diagnostic
with any real response clears it -- because the round's claim requires (a) **and** (b): making
(a) hard to pass would let a false (a) be blamed on the threshold, while making it easy leaves
the claim resting on (b), which is where it belongs.

**Family separation.** Stage A's ``on``/``off`` and the urban-static
``control``/``treatment``/``placebo`` descend from different recipes, so every arm-to-arm gap
below is computed inside one family and never across two. Cross-family rows report levels only.
"""
from __future__ import annotations

from itertools import combinations
import statistics

from scripts.r7_balance_verdict import model_key, truth_key

#: Two ``delta_HB`` medians count as distinct when their gap exceeds this. Dimensionless, on a
#: quantity of order 0.027.
#:
#: **This is a numerical-resolution floor, not a significance threshold.** The dumps are float32,
#: so at a magnitude of 0.027 the representable step is about 3e-9 and a median over thousands of
#: cells is more stable than any single value; 1e-6 is roughly 300x that step, i.e. comfortably
#: above the artefact's own quantisation. It is deliberately *not* set at a level that would
#: decide whether an arm difference matters scientifically -- that question belongs to (c), which
#: asks whether the seed spread is smaller than the arm gap, and to (b), which asks whether the
#: ordering disagrees with RMSE. Setting (a)'s bar at a scientific effect size would duplicate (c)
#: inside (a) and could report "the diagnostic is dead" for a diagnostic that merely moved little.
ARM_DISTINCT_TOLERANCE = 1.0e-6

#: The ``delta_HB`` pair (a) is stated on: 850-500, the pair with humidity and therefore the one
#: whose hydrostatic relation is complete. The 500-250 pair substitutes t500 for the absent t250,
#: so it is reported alongside but never merged with this one and never used to decide (a).
PRIMARY_HYDROSTATIC = "hydrostatic:850-500"

#: Which balance metric speaks to which RMSE variable, frozen before any reading exists. A
#: geopotential error at a level bears on both the thickness across that level and the geostrophic
#: wind at it; a wind error at a level bears on the geostrophic departure at that level. Variables
#: with no defensible partner (t2m, mslp, humidity, 10 m winds) are absent rather than paired with
#: something convenient.
VARIABLE_TO_BALANCE = (
    ("z850", ("hydrostatic:850-500", "geostrophic:850")),
    ("z500", ("hydrostatic:850-500", "hydrostatic:500-250", "geostrophic:500")),
    ("z250", ("hydrostatic:500-250", "geostrophic:250")),
    ("u850", ("geostrophic:850",)), ("v850", ("geostrophic:850",)),
    ("u500", ("geostrophic:500",)), ("v500", ("geostrophic:500",)),
    ("u250", ("geostrophic:250",)), ("v250", ("geostrophic:250",)),
)


#: Two values closer than this are a tie, not an ordering. Used where a clause needs a strict
#: direction (which arm is better balanced); a tie means that pair carries no evidence either way
#: and is skipped rather than broken arbitrarily towards "reversed", which would let an equality
#: manufacture the round's main clause.
TIE_TOLERANCE = 1.0e-12


def _balance(table: dict, lead: str, metric: str, *, side: str = "forecast"):
    """One arm's balance median for a (lead, metric) on one side, or None when absent.

    ``side`` selects the forecast or the analysis field, and the key is built by the producer's
    own functions (`r7_balance_verdict.model_key` / ``truth_key``) so the two modules cannot
    disagree about the spelling -- the defect the end-to-end smoke caught.
    """
    key = truth_key(metric) if side == "target" else model_key(metric)
    entry = (table.get("balance") or {}).get(str(lead), {}).get(key)
    if entry is None:
        return None
    # `case_metrics` emits bare floats for one case; `summarize` wraps them as {"median": ...}
    # across cases. Both are produced by this codebase, so both are readable here.
    return float(entry if isinstance(entry, (int, float)) else entry["median"])


def _rmse(table: dict, lead: str, variable: str):
    return (table.get("rmse") or {}).get(str(lead), {}).get(variable)


def family_members(tables: dict, family: str) -> dict:
    """``{arm: {seed: table}}`` for one family, so within-family pairing is explicit."""
    members: dict[str, dict[int, dict]] = {}
    for table in tables.values():
        if table["family"] != family:
            continue
        members.setdefault(table["arm"], {})[int(table["seed"])] = table
    return members


def _benchmark_family_complete(members: dict, spec: dict) -> bool:
    """True when every declared arm has every declared seed.

    A clause that decided on a partial dump would publish a verdict the frozen design never
    authorised -- the smoke found exactly that: with one seed of five arms present, (a) came back
    ``False`` ("the diagnostic is dead") when the honest answer was "not yet measured". This gate
    is what makes a partial round report ``None`` instead.
    """
    declared_seeds = set(spec.get("seeds") or ())
    return all(arm in members and declared_seeds <= set(members[arm]) for arm in spec["arms"])


def clause_a(tables: dict, families: dict, leads) -> dict:
    """Three arms' ``delta_HB`` medians pairwise distinct, in one order, on every seed."""
    rows, verdicts = [], []
    for family, spec in families.items():
        if len(spec["arms"]) < 3:
            continue
        members = family_members(tables, family)
        if not _benchmark_family_complete(members, spec):
            rows.append({"family": family, "lead": max(leads), "metric": PRIMARY_HYDROSTATIC,
                         "skipped": "arm/seed set incomplete; (a) is untested, not false"})
            continue
        per_seed_ok, orderings = [], []
        for seed in sorted({seed for arm in members.values() for seed in arm}):
            values = {}
            for arm in spec["arms"]:
                table = members.get(arm, {}).get(seed)
                if table is None:
                    continue
                value = _balance(table, str(max(leads)), PRIMARY_HYDROSTATIC)
                if value is not None:
                    values[arm] = value
            if len(values) < 3:
                continue
            gaps = {f"{a}|{b}": values[a] - values[b] for a, b in combinations(sorted(values), 2)}
            ordering = tuple(arm for arm, _ in sorted(values.items(), key=lambda item: -item[1]))
            orderings.append({"seed": seed, "values": values, "gaps": gaps, "ordering": ordering})
            per_seed_ok.append(all(abs(gap) > ARM_DISTINCT_TOLERANCE for gap in gaps.values()))
        # "Consistently distinct" also requires the same order on every seed: three arms whose
        # medians swap places between seeds are not ordered, they are unresolved.
        same_order = len({row["ordering"] for row in orderings}) == 1 and bool(orderings)
        trials = [statistics.fmean(row["values"].values()) for row in orderings] if orderings else []
        if not orderings:
            # Not enough seeds produced all three arms. That is *untested*, not failed: this
            # family contributed no comparison, so it neither passes nor vetoes the clause.
            rows.append({"family": family, "lead": max(leads), "metric": PRIMARY_HYDROSTATIC,
                         "per_seed": orderings, "same_ordering_on_every_seed": None,
                         "distinct_on_every_seed": None, "tolerance": ARM_DISTINCT_TOLERANCE,
                         "seed_values": trials, "skipped": "no seed carried all three arms"})
            continue
        verdicts.append(bool(per_seed_ok) and all(per_seed_ok) and same_order)
        rows.append({"family": family, "lead": max(leads), "metric": PRIMARY_HYDROSTATIC,
                     "per_seed": orderings, "same_ordering_on_every_seed": same_order,
                     "distinct_on_every_seed": bool(per_seed_ok) and all(per_seed_ok),
                     "tolerance": ARM_DISTINCT_TOLERANCE,
                     "seed_values": trials})
    if not verdicts:
        return {"value": None, "reason": "no family in this round has three arms with a full "
                                         "seed set", "rows": rows}
    return {"value": bool(all(verdicts)), "rows": rows,
            "reason": "pairwise gaps exceed the tolerance in the same order on every seed"}


def clause_b(tables: dict, families: dict, leads) -> dict:
    """At least one (family, lead, variable) where the better-RMSE arm is worse balanced.

    The reversal must hold on **all three seeds** at that (family, lead, variable): a single
    flipped comparison among hundreds is noise, and (c) already holds the round to a seed-stability
    standard. ``reversals_any_seed`` reports the looser reading without letting it decide.
    """
    rows, skipped_families = [], []
    for family, spec in families.items():
        members = family_members(tables, family)
        if not _benchmark_family_complete(members, spec):
            # A partial dump must not decide the round's main clause. The smoke showed a
            # one-seed partial dump returning (b)=true; the frozen criterion is stated on three
            # seeds, so anything less is untested.
            skipped_families.append(family)
            continue
        arms = [arm for arm in spec["arms"] if arm in members]
        if len(arms) < 2:
            continue
        for lead in leads:
            for variable, metrics in VARIABLE_TO_BALANCE:
                for metric in metrics:
                    for arm_a, arm_b in combinations(arms, 2):
                        per_seed = []
                        for seed in sorted(members[arm_a]):
                            table_a = members[arm_a].get(seed)
                            table_b = members[arm_b].get(seed)
                            if table_a is None or table_b is None:
                                continue
                            rmse_a, rmse_b = _rmse(table_a, lead, variable), _rmse(table_b, lead, variable)
                            balance_a = _balance(table_a, lead, metric)
                            balance_b = _balance(table_b, lead, metric)
                            if None in (rmse_a, rmse_b, balance_a, balance_b):
                                continue
                            # A tie in either quantity is not a direction. Without this guard the
                            # `else` branch below would silently name the second arm "better" on
                            # an exact equality, and two identical arms would generate reversals
                            # out of nothing.
                            if abs(rmse_a - rmse_b) <= TIE_TOLERANCE:
                                continue
                            if abs(abs(balance_a) - abs(balance_b)) <= TIE_TOLERANCE:
                                continue
                            better_rmse = arm_a if rmse_a < rmse_b else arm_b
                            worse_balance = arm_a if abs(balance_a) > abs(balance_b) else arm_b
                            per_seed.append({"seed": seed, "better_rmse_arm": better_rmse,
                                             "worse_balance_arm": worse_balance,
                                             "reversed": better_rmse == worse_balance,
                                             "rmse": {arm_a: rmse_a, arm_b: rmse_b},
                                             "balance": {arm_a: balance_a, arm_b: balance_b}})
                        if not per_seed:
                            continue
                        rows.append({
                            "family": family, "lead": lead, "variable": variable,
                            "metric": metric, "arms": [arm_a, arm_b],
                            "reversed_on_all_seeds": all(row["reversed"] for row in per_seed),
                            "reversed_on_any_seed": any(row["reversed"] for row in per_seed),
                            "per_seed": per_seed})
    if not rows:
        return {"value": None, "reason": "no comparable arm pair produced both an RMSE and a "
                                         "balance number", "rows": [],
                "skipped_families": skipped_families}
    consistent = [row for row in rows if row["reversed_on_all_seeds"]]
    any_seed = [row for row in rows if row["reversed_on_any_seed"]]
    return {
        "value": bool(consistent),
        "rows": consistent[:40],
        "n_comparisons": len(rows),
        "n_reversed_all_seeds": len(consistent),
        "n_reversed_any_seed": len(any_seed),
        "reversals_any_seed": [{"family": row["family"], "lead": row["lead"],
                                "variable": row["variable"], "metric": row["metric"],
                                "arms": row["arms"]} for row in any_seed[:40]],
        "reason": ("a reversal is counted only when the same arm pair at the same (lead, "
                   "variable) reverses on every seed; the any-seed reading is reported but does "
                   "not decide"),
        "skipped_families": skipped_families,
    }


def clause_c(tables: dict, families: dict, leads) -> dict:
    """Each arm's three-seed range is smaller than the gap between arms, within one family."""
    rows, verdicts = [], []
    for family, spec in families.items():
        members = family_members(tables, family)
        if not _benchmark_family_complete(members, spec):
            rows.append({"family": family, "lead": max(leads), "metric": PRIMARY_HYDROSTATIC,
                         "skipped": "arm/seed set incomplete; (c) is untested, not false"})
            continue
        seed_ranges, medians = {}, {}
        for arm in spec["arms"]:
            values = [_balance(table, str(max(leads)), PRIMARY_HYDROSTATIC)
                      for table in members.get(arm, {}).values()]
            values = [value for value in values if value is not None]
            if len(values) < 2:
                continue
            seed_ranges[arm] = max(values) - min(values)
            medians[arm] = statistics.fmean(values)
        if len(medians) < 2:
            continue
        widest = max(seed_ranges.values())
        gaps = [abs(medians[a] - medians[b]) for a, b in combinations(sorted(medians), 2)]
        smallest_gap = min(gaps)
        verdicts.append(bool(widest < smallest_gap))
        rows.append({"family": family, "lead": max(leads), "metric": PRIMARY_HYDROSTATIC,
                     "seed_ranges": seed_ranges, "widest_seed_range": widest,
                     "arm_medians": medians, "smallest_arm_gap": smallest_gap})
    if not verdicts:
        return {"value": None, "reason": "no family had two arms with three seeds each",
                "rows": rows}
    return {"value": bool(all(verdicts)), "rows": rows,
            "reason": "the widest three-seed range must be smaller than the smallest arm gap"}


def clause_d(tables: dict, leads) -> dict:
    """Positive control: every arm's model field is more unbalanced than the analysis field.

    If this were false the diagnostic would be unable to separate a forecast from its own truth,
    which would make any arm-to-arm ranking meaningless (the 0109 lesson: a criterion that cannot
    fail is not a criterion).
    """
    rows = []
    for arm_id, table in sorted(tables.items()):
        for lead in leads:
            model = _balance(table, lead, "geostrophic:500")
            truth = _balance(table, lead, "geostrophic:500", side="target")
            noise_model = _balance(table, lead, "spectral:fine_scale_noise")
            noise_truth = _balance(table, lead, "spectral:fine_scale_noise", side="target")
            rows.append({
                "arm": arm_id, "lead": lead,
                "model_geostrophic": model, "truth_geostrophic": truth,
                "model_more_unbalanced": None if None in (model, truth) else bool(model > truth),
                "model_noise": noise_model, "truth_noise": noise_truth,
                "model_noisier": None if None in (noise_model, noise_truth)
                else bool(noise_model > noise_truth),
            })
    usable = [row for row in rows if row["model_more_unbalanced"] is not None]
    noisier = [row for row in rows if row["model_noisier"] is not None]
    if not usable:
        return {"value": None, "reason": "no arm produced both a model and a truth statistic",
                "rows": []}
    return {
        "value": bool(all(row["model_more_unbalanced"] for row in usable)
                      and all(row["model_noisier"] for row in noisier)),
        "rows": rows,
        "n_rows": len(usable),
        "n_model_more_unbalanced": sum(1 for row in usable if row["model_more_unbalanced"]),
        "n_noisier": sum(1 for row in noisier if row["model_noisier"]),
        "reason": ("every arm at every lead must show a larger geostrophic departure and a "
                   "larger fine-scale noise in the forecast than in its own analysis target"),
    }


def evaluate(tables: dict, families: dict, leads, preflight_passed) -> dict:
    """All five clauses plus the verdict string the evidence page quotes."""
    leads = tuple(int(lead) for lead in leads)
    clauses = {
        "a_arms_differ": clause_a(tables, families, leads),
        "b_rmse_and_balance_disagree": clause_b(tables, families, leads),
        "c_seed_spread_below_arm_gap": clause_c(tables, families, leads),
        "d_truth_is_less_unbalanced": clause_d(tables, leads),
        "e_preflight": {"value": bool(preflight_passed),
                        "reason": "the (e) gate ran before any field existed"},
    }
    values = {name: clause["value"] for name, clause in clauses.items()}
    if values["a_arms_differ"] is True and values["b_rmse_and_balance_disagree"] is True:
        verdict = "balance-is-an-independent-axis"
    elif values["a_arms_differ"] is False:
        verdict = "negative-diagnostic-is-dead"
    elif values["a_arms_differ"] is True and values["b_rmse_and_balance_disagree"] is False:
        verdict = "negative-diagnostic-is-redundant-with-rmse"
    elif any(value is None for key, value in values.items() if key != "e_preflight"):
        verdict = "untested-no-arm-contrast"
    else:
        verdict = "negative-other-clause-failed"
    return {
        "clauses": clauses,
        "values": values,
        "verdict": verdict,
        "reading_rule": (
            "(a) false: the diagnostic is dead and measured nothing. (b) false: it lives but says "
            "the same thing as RMSE, so it is not an independent axis. Only (a) and (b) both true "
            "licenses 'physical balance is an independent evaluation axis'. Untested is null, "
            "never false and never true."),
        "family_rule": ("every arm-to-arm gap is computed inside one family only; cross-family "
                        "rows are levels, never deltas (decision 0114 D5)"),
    }
