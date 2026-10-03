"""Pure same-case physical metrics and official #60 seed-keyed comparisons.

Inputs are explicit normalized receipt dictionaries, never an intersection of
available rows. Undefined skill/ACC and bad cases stay in the complete table.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean

REGIONS = ("full", "interior", "edge_2")
OUTCOMES = ("improved", "worsened", "unresolved")
STATISTICS = ("mse", "climatology_mse", "acc_dot", "acc_forecast_energy", "acc_target_energy")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def number(value, *, nonnegative=False):
    if isinstance(value, bool):
        raise ValueError("boolean is not a measurement")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("finite measurement required") from exc
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError("finite nonnegative measurement required" if nonnegative else "finite measurement required")
    return result


def integer(value, *, minimum=0):
    measured = number(value)
    if measured != int(measured) or measured < minimum:
        raise ValueError("exact integer count required")
    return int(measured)


def close(actual, expected, name):
    if not math.isclose(number(actual), expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"same-case metric mismatch: {name}")


def case_key(init_time, valid_times, lead):
    if not isinstance(init_time, str) or not isinstance(valid_times, list) or len(valid_times) != 1:
        raise ValueError("one exact initialization/valid-time pair per lead required")
    try:
        init, valid = datetime.fromisoformat(init_time), datetime.fromisoformat(valid_times[0])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid initialization/valid time") from exc
    if valid - init != timedelta(hours=lead):
        raise ValueError("valid time does not equal initialization plus lead")
    return (init_time, tuple(valid_times))


def pooled_statistics(rows):
    """Pool sufficient statistics before sqrt/division; never average case ACC."""
    if not rows:
        raise ValueError("no cases to pool")
    values = {key: [number(row[key], nonnegative=key != "acc_dot") for row in rows]
              for key in STATISTICS}
    for row in rows:
        mse_value, climate = number(row["mse"], nonnegative=True), number(row["climatology_mse"], nonnegative=True)
        p, t, d = (number(row[key], nonnegative=key != "acc_dot")
                   for key in ("acc_forecast_energy", "acc_target_energy", "acc_dot"))
        scale = max(p, t, abs(d), mse_value, climate, 1e-12)
        if abs(climate - t) > 1e-9 * scale or abs(mse_value - (p + t - 2 * d)) > 1e-9 * scale:
            raise ValueError("physical MSE/ACC sufficient-statistic identity mismatch")
    mse, climatology = fmean(values["mse"]), fmean(values["climatology_mse"])
    dot = fmean(values["acc_dot"])
    forecast, target = fmean(values["acc_forecast_energy"]), fmean(values["acc_target_energy"])
    denominator = math.sqrt(forecast) * math.sqrt(target)
    if abs(dot) > denominator + 1e-9 * max(denominator, 1e-12):
        raise ValueError("anomaly dot product exceeds its Cauchy bound")
    skill = None if climatology == 0 else 1 - mse / climatology
    acc = None if denominator == 0 else max(-1., min(1., dot / denominator))
    return {"mse": mse, "rmse": math.sqrt(mse), "climatology_mse": climatology,
            "rmse_climatology": math.sqrt(climatology), "mse_skill": skill,
            "skill_status": "undefined_zero_climatology_mse" if skill is None else "defined",
            "pooled_acc": acc,
            "acc_status": "undefined_zero_anomaly_energy" if acc is None else "defined",
            "acc_dot": dot, "acc_forecast_energy": forecast, "acc_target_energy": target,
            "n_initializations": len(rows),
            "bad_case_counts": {
                "worse_than_climatology": sum(m > c for m, c in zip(values["mse"], values["climatology_mse"])),
                "zero_climatology_mse": sum(c == 0 for c in values["climatology_mse"]),
                "negative_acc_dot": sum(d < 0 for d in values["acc_dot"]),
                "zero_anomaly_energy": sum(p == 0 or t == 0 for p, t in
                                           zip(values["acc_forecast_energy"], values["acc_target_energy"]))}}


def validate_metrics(region_rows, case_rows, *, cases, channels, units, lead, regions=REGIONS):
    """Validate canonical rows against a full frozen cohort and recompute every cell.

    Canonical case rows contain init_time/valid_times plus the five STATISTICS;
    region rows carry rmse/rmse_climatology/mse_skill/pooled_acc and statuses.
    This pure boundary intentionally knows nothing about worker file schemas.
    """
    if (len(channels) != 17 or len(set(channels)) != 17 or len(units) != 17
            or any(not unit or unit in ("normalized", "unknown") for unit in units)):
        raise ValueError("all 17 channels and explicit physical units required")
    expected_cases = {case_key(init, valid, lead) for init, valid in cases}
    if not expected_cases or len(expected_cases) != len(cases):
        raise ValueError("nonempty unique frozen case set required")
    expected_cells = {(region, variable) for region in regions for variable in channels}
    grouped = {key: {} for key in expected_cells}
    for row in case_rows:
        cell = (row["region"], row["variable"])
        if cell not in grouped or number(row["lead_hours"]) != lead:
            raise ValueError("undeclared region/variable/lead")
        if row["unit"] != units[channels.index(row["variable"])]:
            raise ValueError("physical unit mismatch")
        key = case_key(row["init_time"], row["valid_times"], lead)
        if key not in expected_cases or key in grouped[cell]:
            raise ValueError("changed or duplicate case; no silent intersection")
        # Validate EACH case, not just an aggregate where corrupt signs can cancel.
        pooled_statistics([row])
        grouped[cell][key] = row
    if any(set(bucket) != expected_cases for bucket in grouped.values()):
        raise ValueError("all exact cases in every variable/region required")
    reported = {}
    for row in region_rows:
        cell = (row["region"], row["variable"])
        if cell not in expected_cells or cell in reported or number(row["lead_hours"]) != lead:
            raise ValueError("duplicate/undeclared regional metric cell")
        if row["unit"] != units[channels.index(row["variable"])]:
            raise ValueError("regional metric physical unit mismatch")
        if integer(row["n_initializations"], minimum=1) != len(cases):
            raise ValueError("regional metric case count mismatch")
        reported[cell] = row
    if set(reported) != expected_cells:
        raise ValueError("complete 17-variable regional table required")
    verified = []
    for region in regions:
        for variable in channels:
            cell = (region, variable)
            calculated = pooled_statistics(list(grouped[cell].values()))
            row = reported[cell]
            for key in (*STATISTICS, "rmse", "rmse_climatology", "mse_skill", "pooled_acc"):
                expected = calculated[key]
                if expected is None:
                    if row[key] not in (None, ""):
                        raise ValueError(f"undefined {key} must remain explicit, not filtered/imputed")
                else:
                    close(row[key], expected, key)
            for key in ("skill_status", "acc_status"):
                if row[key] != calculated[key]:
                    raise ValueError(f"undefined/defined status mismatch: {key}")
            verified.append({"region": region, "variable": variable,
                             "unit": units[channels.index(variable)], "lead_hours": lead,
                             **calculated, "case_set_sha256": digest(sorted(cases)),
                             "cases": sorted(cases)})
    return verified


def aggregate_metrics(records, *, seeds, arms, leads, kernels, regions=REGIONS):
    """Keep per-seed RMSE mean distinct from RMSE pooled over every case/seed."""
    variables = {row["variable"] for row in records}
    if len(variables) != 17 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("all 17 variables and a unique explicit seed set required")
    expected = {(arm, kernel, region, lead, variable)
                for arm in arms for kernel in kernels for region in regions
                for lead in leads for variable in variables}
    grouped = {}
    for row in records:
        key = tuple(row[k] for k in ("arm", "K", "region", "lead_hours", "variable"))
        if key not in expected or row["seed"] not in seeds:
            raise ValueError("undeclared aggregate identity")
        bucket = grouped.setdefault(key, {})
        if row["seed"] in bucket:
            raise ValueError("duplicate seed metric")
        bucket[row["seed"]] = row
    if set(grouped) != expected or any(set(bucket) != set(seeds) for bucket in grouped.values()):
        raise ValueError("all exact seeds, arms, K, regions and leads required")
    table = []
    for (arm, kernel, region, lead, variable), bucket in sorted(grouped.items()):
        ordered = [bucket[seed] for seed in seeds]
        if any(row["unit"] != ordered[0]["unit"] or row["cases"] != ordered[0]["cases"]
               or row["case_set_sha256"] != ordered[0]["case_set_sha256"] for row in ordered):
            raise ValueError("seed units or exact case sets differ")
        n = sum(row["n_initializations"] for row in ordered)
        pooled_input = {key: sum(row[key] * row["n_initializations"] for row in ordered) / n
                        for key in STATISTICS}
        pooled = pooled_statistics([pooled_input])
        bad = {key: sum(row["bad_case_counts"][key] for row in ordered)
               for key in ordered[0]["bad_case_counts"]}
        table.append({"arm": arm, "K": kernel, "depth": kernel, "region": region,
                      "lead_hours": lead, "variable": variable, "unit": ordered[0]["unit"],
                      "seeds": list(seeds), "seed_rmse": {str(seed): bucket[seed]["rmse"] for seed in seeds},
                      "rmse_seed_mean": fmean(row["rmse"] for row in ordered),
                      "rmse_pooled_cases": pooled["rmse"],
                      "mse_skill_pooled_cases": pooled["mse_skill"], "skill_status": pooled["skill_status"],
                      "acc_pooled_cases": pooled["pooled_acc"], "acc_status": pooled["acc_status"],
                      "n_initializations_all_seeds": n, "bad_case_counts": bad,
                      "per_seed_n_initializations": {str(seed): bucket[seed]["n_initializations"] for seed in seeds},
                      "case_set_digests": {str(seed): bucket[seed]["case_set_sha256"] for seed in seeds},
                      "case_identity": "exact"})
    return table


def paired_comparisons(table, *, pairs, kernels, regions=REGIONS):
    """Use the unchanged official #60 compare() for EACH region and K."""
    from .r7_coreasoning_compare import compare

    result = {}
    for focus, baseline in pairs:
        cells = {}
        for region in regions:
            for kernel in kernels:
                subset = [row for row in table if row["region"] == region and row["K"] == kernel
                          and row["arm"] in (focus, baseline)]
                blocks = compare(subset, baseline=baseline, depth=kernel)
                matches = [block for block in blocks if block["arm"] == focus]
                if len(matches) != 1 or matches[0]["case_identity"] != "exact":
                    raise ValueError("official comparator did not return exact declared pair")
                for entry in matches[0]["seed_paired"]:
                    key = f"k{kernel}|{region}|{int(entry['lead_hours'])}h|{entry['variable']}"
                    cells[key] = {"K": kernel, "region": region, "lead_hours": int(entry["lead_hours"]),
                                  "variable": entry["variable"], "unit": entry["unit"],
                                  "outcome": entry["direction"], "sign_consistent": entry["sign_consistent"],
                                  "seed_deltas": {str(item["seed"]): item["delta"] for item in entry["seed_deltas"]}}
        result[f"{focus} - {baseline}"] = {
            "focus_arm": focus, "baseline_arm": baseline, "cells": cells,
            "totals": {outcome: sum(cell["outcome"] == outcome for cell in cells.values()) for outcome in OUTCOMES},
            "rule": "official #60; same seed; all deltas <0 improved, all >0 worsened, otherwise unresolved",
            "difference_direction": "focus minus baseline; no p-values or cross-unit aggregate"}
    return result


def candidate_selection(protocol, pairs):
    """Only the frozen B full-domain K4 t2m 6/12 rule; never posthoc candidates."""
    reporting = protocol["reporting"]
    if "primary" not in reporting or "candidate_selection" not in reporting:
        return {"evaluated": False, "status": "refused", "selected_mode": None,
                "reason": "predeclared primary/candidate selection absent", "scientific_claim": False}
    primary, selection = reporting["primary"], reporting["candidate_selection"]
    if (primary["variable"] != "t2m" or primary["lead_hours"] != [6, 12]
            or primary["unit"] != "K" or primary["degradation_tolerance"] != 0):
        return {"evaluated": False, "status": "refused", "selected_mode": None,
                "reason": "unsupported frozen primary rule; no replacement threshold", "scientific_claim": False}
    focus, baseline, equal = (selection[key] for key in ("focus", "baseline", "equal_compute"))
    checks = []
    for lead in primary["lead_hours"]:
        key = f"k4|full|{lead}h|{primary['variable']}"
        vs_baseline = pairs[f"{focus} - {baseline}"]["cells"][key]
        vs_equal = pairs[f"{focus} - {equal}"]["cells"][key]
        if vs_baseline["unit"] != primary["unit"] or vs_equal["unit"] != primary["unit"]:
            raise ValueError("primary physical unit differs from frozen rule")
        if set(vs_baseline["seed_deltas"]) != set(vs_equal["seed_deltas"]):
            raise ValueError("candidate primary seed sets differ")
        checks.append({"lead_hours": lead, "vs_continue": vs_baseline["seed_deltas"],
                       "vs_equal_compute": vs_equal["seed_deltas"],
                       "strictly_improved_all_seeds": all(d < 0 for d in vs_baseline["seed_deltas"].values()),
                       "no_degradation_all_seeds": all(d <= 0 for d in vs_equal["seed_deltas"].values())})
    supported = all(check["strictly_improved_all_seeds"] and check["no_degradation_all_seeds"] for check in checks)
    return {"evaluated": True, "status": "supported" if supported else "negative_or_mixed",
            "selected_mode": "two_step" if supported else "l6", "primary_checks": checks,
            "scientific_claim": False, "rule": selection["rule"],
            "hypothesis_action": "retain candidate for independent C" if supported else "terminate rollout hypothesis only",
            "independent_C_continues": True, "no_posthoc_selection": True}


def publish_tables(output, protocol, training, evaluations, records, table, pairs, outcome):
    """All rows carry frozen identities; cost snapshots are not whole-attempt finals."""
    from .r7_v2_protocol import job_key, write_path

    identity = {"stage": protocol["stage"], "protocol_sha256": protocol["protocol_sha256"],
                "model_code_sha256": protocol["code"]["model_code_sha256"],
                "source_tree_sha256": protocol["code"]["source_tree_sha256"],
                "code_zip_sha256": protocol["code"]["code_zip_sha256"],
                "base_commit": protocol["code"]["base_commit"],
                "working_tree_modified": protocol["code"]["working_tree_modified"],
                "code_commit_sha256": protocol["code"]["code_commit_sha256"],
                "code_status_sha256": protocol["code"]["code_status_sha256"],
                "source_sha256": protocol["sources"]["source_sha256"],
                "data_identity": protocol["data"]["data_identity"],
                "sidecar_identity": protocol["sidecar"]["identity"], "windows_sha256": digest(protocol["windows"]),
                "scientific_claim": False, "test_read": False}
    files = []
    def emit(name, rows):
        decorated = [{**identity, **row, "row_sha256": digest({**identity, **row})} for row in rows]
        write_csv(write_path(Path(output) / name, output), decorated)
        files.append(name)
    parameters, flops = [], []
    for seed, per_arm in protocol["cpu_profile"]["measurements"].items():
        for arm, measurement in per_arm.items():
            pin = {"seed": int(seed), "arm": arm, "cpu_profile_sha256": digest(protocol["cpu_profile"]),
                   "measurement_sha256": digest(measurement), "provenance": "frozen actual CPU objective/backward measurement"}
            parameters.append({**pin, **{key: measurement[key] for key in ("parameters", "trainable_parameters")}})
            flops.append({**pin, **{key: measurement[key] for key in
                                   ("forward_flops", "forward_backward_flops", "actual_model_forward_calls", "objective",
                                    "physical_steps", "reasoning_steps", "forward_scope", "backward_scope")},
                          "backward_flops": measurement["forward_backward_flops"] - measurement["forward_flops"]})
    emit("parameter_table.csv", parameters)
    emit("flops_table.csv", flops)
    jobs = [*training.values(), *evaluations.values()]
    def job_pin(entry):
        return {"job": entry["job"], "worker_receipt_sha256": entry["receipt_sha256"],
                "seed": entry["job"]["seed"], "arm": entry["job"]["arm"], "K": entry["job"]["reasoning_steps"],
                "lead_hours": entry["job"]["lead"], "provenance": "workers/" + job_key(entry["job"]) + ".json"}
    emit("training_table.csv", [{**job_pin(entry), "updates_run": entry["updates_run"],
                                "elapsed_seconds": entry["elapsed_seconds"], "training_loop_seconds": entry["report"]["elapsed_seconds"],
                                "seconds_per_update_training_scope": entry["report"]["elapsed_seconds"] / entry["updates_run"],
                                "contract_sha256": digest(entry["contract"]), "checkpoint_sha256": entry["checkpoint_sha256"],
                                "training_report_sha256": entry["training_report_sha256"]} for entry in training.values()])
    emit("evaluation_table.csv", [{**job_pin(entry), "elapsed_seconds": entry["elapsed_seconds"],
                                  "evaluation_loop_seconds": entry["evaluation_provenance"]["loop_elapsed_seconds"],
                                  "helper_whole_seconds": entry["evaluation_provenance"]["whole_elapsed_seconds"],
                                  "isolated_forward_median_seconds": entry["evaluation_provenance"]["isolated_forward"]["median_seconds_per_batch"],
                                  "isolated_scope": entry["evaluation_provenance"]["isolated_forward"]["scope"],
                                  "artifact_sha256": entry["artifact_sha256"]} for entry in evaluations.values()])
    emit("allocator_table.csv", [{**job_pin(entry), "phase": entry["job"]["phase"], "baseline": entry["baseline"],
                                 "peak_allocated_bytes": entry["peak_allocated_bytes"],
                                 "peak_reserved_bytes": entry["peak_reserved_bytes"]} for entry in jobs])
    emit("case_table.csv", [{**job_pin(entry), "n_evaluated": entry["n_evaluated"],
                            "n_available_windows": entry["n_available_windows"],
                            "case_set_sha256": digest(sorted(protocol["data"]["evaluation_cases"][str(entry["job"]["lead"])]["cases"])),
                            "cases": protocol["data"]["evaluation_cases"][str(entry["job"]["lead"])]["cases"],
                            "provenance_sha256": entry["artifact_sha256"]["provenance.json"]} for entry in evaluations.values()])
    metric_rows = []
    for row in records:
        entry = evaluations[(row["seed"], row["arm"], row["lead_hours"], row["K"])]
        metric_rows.append({**job_pin(entry), **{key: value for key, value in row.items() if key != "cases"},
                            "provenance_sha256": entry["artifact_sha256"]["provenance.json"]})
    baseline_rows, semantic_pins = [], {}
    for entry in evaluations.values():
        for row in entry["baseline_metrics"]:
            identity_key = (row["seed"], row["baseline_kind"], row["lead_hours"], row["region"], row["variable"])
            if identity_key in semantic_pins and semantic_pins[identity_key] != row["semantic_sha256"]:
                raise ValueError("same-case zero-trained baseline changed across arm/K worker references")
            semantic_pins[identity_key] = row["semantic_sha256"]
            baseline_rows.append({**job_pin(entry), **row, "baseline_provenance_sha256": entry["artifact_sha256"]["provenance.json"]})
    emit("baseline_table.csv", baseline_rows)
    emit("rmse_table.csv", metric_rows)
    emit("acc_table.csv", [{key: value for key, value in row.items() if key not in
                           ("mse", "rmse", "climatology_mse", "rmse_climatology", "mse_skill")} for row in metric_rows])
    metric_inputs_digest = digest(metric_rows)
    emit("aggregate_table.csv", [{**row, "provenance": "complete same-case per-seed tables", "inputs_sha256": metric_inputs_digest}
                                 for row in table])
    emit("state_table.csv", [{"status": outcome["status"], "finalized": False, "coverage_complete": True,
                             "whole_cost_status": "pending-driver-final-seal", "gpu_cost_status": "pending-driver-final-seal",
                             "whole_cost_reference": protocol["whole_round_cost_reference"],
                             "selection": outcome["candidate_selection"], "adaptive_gate": outcome["adaptive_gate"],
                             "provenance": "stage_result.json"}])
    return files


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def write_csv(path, rows):
    if not rows:
        raise ValueError("cannot publish an empty required table")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True, allow_nan=False)
                         if isinstance(value, (dict, list, tuple)) else value
                         for key, value in row.items()} for row in rows)
