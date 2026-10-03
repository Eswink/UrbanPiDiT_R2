"""Exclusive independent metadata-complement tables; never rewrites a forecast protocol."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .r7_v2_stats_sources import digest, path, write_json


def emit_csv(output, name, rows, identity, *, check=lambda: None):
    rows = [{**identity, **row} for row in rows]
    if not rows:
        raise ValueError("required complement table cannot be empty")
    for row in rows:
        check()
        row["row_sha256"] = digest(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path(Path(output) / name).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            check()
            writer.writerow({key: json.dumps(value, sort_keys=True, allow_nan=False)
                             if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})
    check()
    return name


def publish(output, complement, original, training, evaluations, records, aggregate, pairs, selection, source_attempt, *, check=lambda: None):
    """Every row has BOTH forecast and complement identity; old failed costs stay old."""
    identity = {"source_kind": "r7-v2-stats-complement", "scientific_claim": False, "test_read": False,
                "source_protocol_sha256": original["protocol_sha256"], "complement_protocol_sha256": complement["protocol_sha256"],
                "source_attempt_sha256": complement["source_attempt_sha256"], "source_attempt_failed": True,
                "source_model_code_sha256": original["code"]["model_code_sha256"],
                "source_tree_sha256": original["code"]["source_tree_sha256"],
                "source_code_zip_sha256": original["code"]["code_zip_sha256"],
                "source_data_identity": original["data"]["data_identity"],
                "source_sha256": original["sources"]["source_sha256"],
                "corrected_validator_sha256": complement["correction"]["corrected_validator_sha256"],
                "objective_helper_sha256": complement["correction"]["helper_sha256"]}
    published = []
    def emit(name, rows):
        published.append(emit_csv(output, name, rows, identity, check=check))
    def job_pin(entry):
        job = entry["job"]
        return {"job": job, "seed": job["seed"], "arm": job["arm"], "K": job["reasoning_steps"],
                "lead_hours": job["lead"], "source_worker_receipt_sha256": entry["receipt_sha256"]}
    params, flops = [], []
    for seed, per_arm in original["cpu_profile"]["measurements"].items():
        for arm, measured in per_arm.items():
            pin = {"seed": int(seed), "arm": arm, "cpu_profile_sha256": digest(original["cpu_profile"]), "measurement_sha256": digest(measured)}
            params.append({**pin, **{key: measured[key] for key in ("parameters", "trainable_parameters")}})
            flops.append({**pin, **{key: measured[key] for key in ("forward_flops", "forward_backward_flops", "actual_model_forward_calls",
                                                                 "physical_steps", "objective", "reasoning_steps", "forward_scope", "backward_scope")}})
    emit("parameter_table.csv", params)
    emit("flops_table.csv", flops)
    emit("training_table.csv", [{**job_pin(entry), "updates_run": entry["updates_run"], "checkpoint_sha256": entry["checkpoint_sha256"],
                                 "training_report_sha256": entry["training_report_sha256"], "contract_sha256": digest(entry["contract"]),
                                 "elapsed_seconds": entry["elapsed_seconds"], "training_loop_seconds": entry["report"]["elapsed_seconds"]}
                                for entry in training.values()])
    emit("evaluation_table.csv", [{**job_pin(entry), "elapsed_seconds": entry["elapsed_seconds"],
                                   "loop_elapsed_seconds": entry["evaluation_provenance"]["loop_elapsed_seconds"],
                                   "helper_whole_seconds": entry["evaluation_provenance"]["whole_elapsed_seconds"],
                                   "isolated_forward": entry["evaluation_provenance"]["isolated_forward"], "artifact_sha256": entry["artifact_sha256"]}
                                  for entry in evaluations.values()])
    emit("allocator_table.csv", [{**job_pin(entry), "baseline": entry["baseline"], "pre_init_baseline": entry["pre_init_baseline"],
                                  "initialized_baseline": entry["initialized_baseline"], "peak_allocated_bytes": entry["peak_allocated_bytes"],
                                  "peak_reserved_bytes": entry["peak_reserved_bytes"]} for entry in [*training.values(), *evaluations.values()]])
    emit("case_table.csv", [{**job_pin(entry), "n_evaluated": entry["n_evaluated"], "n_available_windows": entry["n_available_windows"],
                             "cases": original["data"]["evaluation_cases"][str(entry["job"]["lead"])]["cases"],
                             "provenance_sha256": entry["artifact_sha256"]["provenance.json"]} for entry in evaluations.values()])
    metric_rows = []
    for row in records:
        entry = evaluations[(row["seed"], row["arm"], row["lead_hours"], row["K"])]
        metric_rows.append({**row, "worker_receipt_sha256": entry["receipt_sha256"], "provenance_sha256": entry["artifact_sha256"]["provenance.json"]})
    emit("rmse_table.csv", metric_rows)
    emit("acc_table.csv", [{key: value for key, value in row.items() if key not in ("rmse", "mse", "climatology_mse", "mse_skill")} for row in metric_rows])
    emit("aggregate_table.csv", aggregate)
    emit("baseline_table.csv", [{**job_pin(entry), **row, "provenance_sha256": entry["artifact_sha256"]["provenance.json"]}
                                for entry in evaluations.values() for row in entry["baseline_metrics"]])
    emit("state_table.csv", [{"status": "aggregation-complete", "coverage_complete": True, "selection": selection,
                              "training_performed": False, "evaluation_performed": False, "gpu_used": False,
                              "source_status": source_attempt["status"], "source_failed_gpu_hours": source_attempt["gpu_hours_charged"],
                              "source_failed_whole_seconds": source_attempt["whole_elapsed_seconds"]}])
    result = {**identity, "status": "aggregation-complete", "coverage_complete": True, "training_performed": False,
              "evaluation_performed": False, "gpu_used": False, "scientific_state": selection["status"], "candidate_selection": selection,
              "source_cost_reference": str(Path(complement["source_attempt"]) / "attempt.json"),
              "source_failed_cost": {key: source_attempt[key] for key in ("status", "gpu_hours_charged", "whole_elapsed_seconds", "soft_overrun_seconds")},
              "limitations": complement["limitations"], "training": {f"{s}/{a}": entry for (s, a), entry in training.items()},
              "evaluation": {f"{s}/{a}/{h}/k{k}": entry for (s, a, h, k), entry in evaluations.items()}, "metrics": records, "aggregate": aggregate}
    write_json(Path(output) / "aggregate_result.json", result, check=check)
    write_json(Path(output) / "paired_comparison.json", {**identity, "pairs": pairs, "candidate_selection": selection,
                                                         "limitations": complement["limitations"]}, check=check)
    return [*published, "aggregate_result.json", "paired_comparison.json"]
