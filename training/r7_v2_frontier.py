"""Pure mechanical implementation of confirmation preregistration section 3.

This is not a controller and never reads data, files, or GPU state. Inputs are
finalize's PER-SEED physical metric records, canonical physical per-case rows,
and the flat measured-latency adapter below, not cross-seed aggregate tables or
raw CSV/normalized ACC statistics. The upstream inventory verifier owns complete
17-variable/bad-case/code/input/allocator/receipt acceptance and the ten-repeat
isolated-forward verification; its literal True attestation is mandatory.
"""
from __future__ import annotations

import math
from statistics import fmean

from .r7_v2_tables import STATISTICS, case_key, close, digest, integer, number, pooled_statistics

ADAPTIVE_GATE_SCHEMA = {
    "format": "r7-v2-adaptive-gate-v1", "candidate": "process", "baseline": "old_ours",
    "region": "full", "variable": "t2m", "unit": "K", "seeds": [41, 42, 43],
    "lead_hours": [6, 12], "kernels": [1, 2, 4],
}
SEEDS = (41, 42, 43)
PRIMARY_LEADS = (6, 12)
KERNELS = (1, 2, 4)
ARMS = ("old_ours", "process", "matched_generic")
LEADS = (6, 12, 24, 48, 72)
REGIONS = ("full", "interior", "edge_2")
PRIMARY_COUNTS = {6: 22, 12: 21}
ISOLATED_SCOPE = "resident-batch model K forward plus synchronization; excludes IO/transfers/metrics/validation"
IDENTITY_FIELDS = frozenset(("seed", "arm", "K", "region", "lead_hours", "variable", "unit"))
METRIC_FIELDS = frozenset((*STATISTICS, "rmse", "rmse_climatology", "mse_skill", "pooled_acc",
                           "skill_status", "acc_status", "n_initializations"))
PIN_FIELDS = frozenset(("protocol_sha256", "model_code_sha256", "source_tree_sha256", "code_zip_sha256",
                       "source_sha256", "data_identity", "sidecar_identity", "windows_sha256",
                       "provenance_sha256", "worker_receipt_sha256", "row_sha256"))
RECORD_FIELDS = IDENTITY_FIELDS | METRIC_FIELDS | {"cases", "case_set_sha256", "bad_case_counts"}
CASE_FIELDS = IDENTITY_FIELDS | METRIC_FIELDS | {"init_time", "valid_times"}
CASE_OPTIONAL = PIN_FIELDS | {"sample_id"}
LATENCY_FIELDS = frozenset(("seed", "arm", "K", "lead_hours", "median_seconds_per_batch", "scope",
                           "gpu_latency_measured", "provenance_sha256"))
CONDITION_NAMES = ("package_improvement", "accuracy_cost_tradeoff", "case_heterogeneity",
                   "complete_engineering_evidence")
LIMITATIONS = (
    "scientific_claim:false; winter single-region validation development signal, not scientific support",
    "three seeds provide descriptive consistency only, not significance or adequate independent samples",
    "package improvement cannot establish process attribution; matched Generic remains a separate control",
    "future-ground-truth error oracle is descriptive only, never a model/controller/deployment input",
    "eligible authorizes a separate frozen train-only-fit/val-only-calibration protocol, not automatic training",
    "complete inventory, identities, bad cases and allocated/reserved evidence are verified upstream",
    "numerical rebuild uses existing metric-integrity tolerance; scientific inequalities have zero tolerance",
    "no cross-lead or cross-unit average; no mean of per-case ACC; bitwise GPU replay is not asserted",
)


def _schema(row, required, optional, label):
    if not isinstance(row, dict) or not required <= row.keys() or row.keys() - required - optional:
        raise ValueError(f"{label}: missing or unknown schema fields")


def _sha256(value, label):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{label}: exact lowercase SHA256 required")
    return value


def _frozen_cases(cases, lead):
    if not isinstance(cases, list) or len(cases) != PRIMARY_COUNTS[lead]:
        raise ValueError("complete frozen primary 22/21 per-lead cases required")
    keys = []
    for item in cases:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("exact initialization/valid-time pairs required")
        keys.append(case_key(item[0], item[1], lead))
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate frozen initialization/valid-time pair")
    return set(keys)


def _protocol_scope(protocol):
    if not isinstance(protocol, dict) or protocol.get("stage") != "C":
        raise ValueError("only independently frozen C can evaluate this gate")
    reporting = protocol["reporting"]
    gate = reporting["adaptive_gate"]
    if not isinstance(gate, dict) or digest(gate) != digest(ADAPTIVE_GATE_SCHEMA):
        raise ValueError("missing/unknown adaptive gate schema; no replacement criteria")
    primary = reporting["primary"]
    if (primary["variable"] != "t2m" or primary["unit"] != "K"
            or primary["lead_hours"] != [6, 12]):
        raise ValueError("primary must be frozen t2m/K/6h/12h")
    tolerance = reporting.get("tolerances", {})
    if "K" not in tolerance or number(tolerance["K"]) != 0:
        raise ValueError("explicit frozen zero primary degradation tolerance required")
    if "degradation_tolerance" in primary and number(primary["degradation_tolerance"]) != 0:
        raise ValueError("primary degradation tolerance is not zero")
    data = protocol["data"]
    channels, units = data["channels"], data["units"]
    if (not isinstance(channels, list) or not isinstance(units, list) or len(channels) != 17
            or len(set(channels)) != 17 or len(units) != 17 or "t2m" not in channels
            or any(not isinstance(unit, str) or not unit or unit in ("normalized", "unknown") for unit in units)):
        raise ValueError("all 17 declared variables and physical units required")
    unit_map = dict(zip(channels, units))
    if unit_map["t2m"] != "K":
        raise ValueError("data t2m physical unit must be K")
    cohorts = {lead: _frozen_cases(data["evaluation_cases"][str(lead)]["cases"], lead)
               for lead in PRIMARY_LEADS}
    return unit_map, cohorts


def _identity(row, units, *, latency=False):
    for name in ("seed", "K", "lead_hours"):
        if type(row[name]) is not int:
            raise ValueError("canonical seed/K/lead integer identity required")
    if (row["seed"] not in SEEDS or row["arm"] not in ARMS or row["K"] not in KERNELS
            or row["lead_hours"] not in LEADS):
        raise ValueError("unknown seed/arm/K/lead identity")
    if not latency and (row["region"] not in REGIONS or row["variable"] not in units
                        or row["unit"] != units[row["variable"]]):
        raise ValueError("unknown region/variable or wrong physical unit")
    if latency:
        focus = row["arm"] == "process" and row["lead_hours"] in PRIMARY_LEADS
    else:
        focus = (row["region"] == "full" and row["variable"] == "t2m"
                 and row["lead_hours"] in PRIMARY_LEADS
                 and (row["arm"] == "process" or (row["arm"] == "old_ours" and row["K"] == 4)))
    return (row["seed"], row["lead_hours"], row["arm"], row["K"]) if focus else None


def _same_metrics(row, calculated):
    for name in (*STATISTICS, "rmse", "rmse_climatology", "mse_skill", "pooled_acc"):
        expected = calculated[name]
        if expected is None:
            if row[name] not in (None, ""):
                raise ValueError(f"undefined {name} cannot be imputed")
        else:
            close(row[name], expected, name)
    for name in ("skill_status", "acc_status"):
        if row[name] != calculated[name]:
            raise ValueError(f"physical canonical status mismatch: {name}")


def _records(rows, units, cohorts):
    if not isinstance(rows, (list, tuple)):
        raise ValueError("explicit per-seed metric record list required, not cross-seed table")
    records = {}
    for row in rows:
        _schema(row, RECORD_FIELDS, PIN_FIELDS, "per-seed aggregate record")
        key = _identity(row, units)
        if key is None:
            continue  # Known nonprimary endpoints belong to the complete upstream inventory.
        if key in records:
            raise ValueError("duplicate primary per-seed aggregate record")
        lead = key[1]
        if (integer(row["n_initializations"], minimum=1) != len(cohorts[lead])
                or _frozen_cases(row["cases"], lead) != cohorts[lead]):
            raise ValueError("primary aggregate exact cohort/count mismatch")
        expected_digest = digest(sorted(protocol_pair for protocol_pair in row["cases"]))
        if row["case_set_sha256"] != expected_digest:
            raise ValueError("primary aggregate case_set_sha256 mismatch")
        for field in PIN_FIELDS & row.keys():
            _sha256(row[field], field)
        records[key] = row
    expected = {(seed, lead, arm, k) for seed in SEEDS for lead in PRIMARY_LEADS
                for arm, ks in (("old_ours", (4,)), ("process", KERNELS)) for k in ks}
    if set(records) != expected:
        raise ValueError("missing primary per-seed/lead old K4 or process K1/2/4 record")
    return records


def _cases(rows, units, cohorts, records):
    if not isinstance(rows, (list, tuple)):
        raise ValueError("explicit canonical physical per-case row list required")
    grouped = {key: {} for key in records}
    for row in rows:
        _schema(row, CASE_FIELDS, CASE_OPTIONAL, "canonical physical per-case row")
        key = _identity(row, units)
        if key is None:
            continue
        case = case_key(row["init_time"], row["valid_times"], key[1])
        if case not in cohorts[key[1]] or case in grouped[key]:
            raise ValueError("changed/duplicate primary case; no silent intersection")
        if integer(row["n_initializations"], minimum=1) != 1:
            raise ValueError("one canonical initialization per per-case row required")
        _same_metrics(row, pooled_statistics([row]))
        for field in PIN_FIELDS & row.keys():
            _sha256(row[field], field)
            if field != "row_sha256" and field in records[key] and row[field] != records[key][field]:
                raise ValueError("per-case and primary aggregate identity pins differ")
        grouped[key][case] = row
    for key, bucket in grouped.items():
        if set(bucket) != cohorts[key[1]]:
            raise ValueError("missing primary case; exact paired cohort required")
        calculated = pooled_statistics([bucket[case] for case in sorted(bucket)])
        _same_metrics(records[key], calculated)
        if records[key]["bad_case_counts"] != calculated["bad_case_counts"]:
            raise ValueError("aggregate bad-case counts differ from unfiltered primary cases")
    for seed in SEEDS:
        for lead in PRIMARY_LEADS:
            buckets = [grouped[(seed, lead, arm, k)] for arm, ks in
                       (("old_ours", (4,)), ("process", KERNELS)) for k in ks]
            for case in sorted(cohorts[lead]):
                ids = [bucket[case].get("sample_id") for bucket in buckets]
                if any(value is not None for value in ids) and (any(value is None for value in ids) or len(set(ids)) != 1):
                    raise ValueError("same-case sample_id differs across arms/K")
    return grouped


def _latencies(rows, units, records):
    if not isinstance(rows, (list, tuple)):
        raise ValueError("explicit flat actual isolated latency row list required")
    measured = {}
    for row in rows:
        _schema(row, LATENCY_FIELDS, frozenset(), "flat measured latency row")
        key = _identity(row, units, latency=True)
        if key is None:
            continue
        if key in measured:
            raise ValueError("duplicate primary latency record")
        value = number(row["median_seconds_per_batch"], nonnegative=True)
        if value <= 0 or row["gpu_latency_measured"] is not True or row["scope"] != ISOLATED_SCOPE:
            raise ValueError("positive actual isolated-forward median latency with exact scope required")
        _sha256(row["provenance_sha256"], "latency provenance_sha256")
        record = records[key]
        if "provenance_sha256" in record and record["provenance_sha256"] != row["provenance_sha256"]:
            raise ValueError("latency and primary aggregate provenance differ")
        measured[key] = {**row, "median_seconds_per_batch": value}
    expected = {(seed, lead, "process", k) for seed in SEEDS for lead in PRIMARY_LEADS for k in KERNELS}
    if set(measured) != expected:
        raise ValueError("complete actual K1/2/4 latency for every primary seed/lead required")
    return measured


def _oracle(buckets):
    counts = {str(k): 0 for k in KERNELS}
    cases = []
    for key in sorted(buckets[1]):
        mse = {str(k): number(buckets[k][key]["mse"], nonnegative=True) for k in KERNELS}
        minimum = min(mse.values())
        winners = [k for k in KERNELS if mse[str(k)] == minimum]
        if len(winners) == 1:
            counts[str(winners[0])] += 1
        cases.append({"init_time": key[0], "valid_times": list(key[1]), "fixed_K_mse": mse,
                      "minimum_mse": minimum, "optimal_K": winners,
                      "unique_optimal_K": winners[0] if len(winners) == 1 else None})
    pooled_mse = fmean(row["minimum_mse"] for row in cases)
    return {"mse": pooled_mse, "rmse": math.sqrt(pooled_mse), "cases": cases,
            "unique_optimal_K_counts": counts, "unique_optimal_depths": sum(count > 0 for count in counts.values()),
            "tie_case_count": sum(row["unique_optimal_K"] is None for row in cases),
            "uses_future_ground_truth": True, "oracle_deployable": False, "oracle_deployed": False}


def _evidence(records, cases, latencies):
    evidence = []
    for seed in SEEDS:
        for lead in PRIMARY_LEADS:
            old = records[(seed, lead, "old_ours", 4)]
            fixed = {k: records[(seed, lead, "process", k)] for k in KERNELS}
            metrics = {str(k): {name: number(row[name], nonnegative=True) for name in ("mse", "rmse")}
                       for k, row in fixed.items()}
            timing = {str(k): latencies[(seed, lead, "process", k)] for k in KERNELS}
            oracle = _oracle({k: cases[(seed, lead, "process", k)] for k in KERNELS})
            best = min(metric["rmse"] for metric in metrics.values())
            package = metrics["4"]["rmse"] < number(old["rmse"], nonnegative=True)
            accuracy = metrics["4"]["rmse"] < metrics["1"]["rmse"]
            cost = timing["1"]["median_seconds_per_batch"] < timing["4"]["median_seconds_per_batch"]
            heterogeneity = oracle["rmse"] < best and oracle["unique_optimal_depths"] >= 2
            evidence.append({"seed": seed, "lead_hours": lead, "region": "full", "variable": "t2m", "unit": "K",
                             "n_initializations": old["n_initializations"], "case_set_sha256": old["case_set_sha256"],
                             "old_ours_K4_rmse": number(old["rmse"], nonnegative=True), "fixed_K": metrics,
                             "actual_latency": timing, "oracle": oracle, "best_fixed_rmse": best,
                             "aggregate_evidence": [dict(old), *(dict(fixed[k]) for k in KERNELS)],
                             "package_improvement": package, "accuracy_improvement_K4_vs_K1": accuracy,
                             "latency_improvement_K1_vs_K4": cost, "accuracy_cost_tradeoff": accuracy and cost,
                             "case_heterogeneity": heterogeneity})
    return evidence


def evaluate_adaptive_gate(protocol, aggregate_rows, per_case_rows, latency_rows, *, inventory_verified):
    """Return a JSON-ready, fail-closed gate result without executing any training.

    ``aggregate_rows`` is finalize.records, NOT aggregate_metrics' cross-seed table.
    ``per_case_rows`` is the canonical PHYSICAL metric schema (already std-squared
    converted), decorated with seed/arm/K and optional sample_id/identity pins.
    ``latency_rows`` has EXACT LATENCY_FIELDS, adapted from verified evaluation
    receipts. Ten actual repetitions/median/mean/actual-K and complete inventory
    verification are upstream responsibilities, never inferred from step counts.
    ``protocol.reporting.adaptive_gate`` must exactly equal ADAPTIVE_GATE_SCHEMA.
    """
    conditions = {name: {"evaluated": False, "met": None} for name in CONDITION_NAMES}
    result = {"format": ADAPTIVE_GATE_SCHEMA["format"], "evaluated": False, "status": "refused",
              "conditions": conditions, "seed_lead_evidence": [], "gate_met": None, "start_training": False,
              "oracle_deployable": False, "oracle_deployed": False, "scientific_claim": False,
              "controller_training_executed": False, "test_read": False, "limitations": list(LIMITATIONS)}
    try:
        units, cohorts = _protocol_scope(protocol)
        conditions["complete_engineering_evidence"] = {"evaluated": True, "met": inventory_verified is True}
        if inventory_verified is not True:
            raise ValueError("literal inventory_verified=True required; incomplete engineering is refused")
        records = _records(aggregate_rows, units, cohorts)
        cases = _cases(per_case_rows, units, cohorts, records)
        latencies = _latencies(latency_rows, units, records)
        evidence = _evidence(records, cases, latencies)
        for name in CONDITION_NAMES[:3]:
            conditions[name] = {"evaluated": True, "met": all(row[name] for row in evidence)}
        eligible = all(condition["met"] is True for condition in conditions.values())
        result.update(evaluated=True, status="eligible" if eligible else "not-started", gate_met=eligible,
                      seed_lead_evidence=evidence, start_training=eligible)
        result["reason"] = ("all four predeclared conditions hold; separate controller protocol still required" if eligible
                            else "one or more predeclared conditions fail; no training started")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        result["reason"] = f"missing or inconsistent frozen evidence: {exc}"
    return result
