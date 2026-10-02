"""Fail-closed complete M3 coverage and #60 descriptive seed-paired reporting."""
from __future__ import annotations

import csv
import math
from pathlib import Path

from .r7_arm_harness import (
    comparator_blocks, merge_seed_results, pair_cells, sha256_file, write_study_tables,
)
from .r7_coreasoning_compare import read_case_identity, read_rmse_rows
from .r7_m3_protocol import (
    ARM_NAMES, LEADS, LIMITATIONS, PAIRS, RESULT_FORMAT, ROUND_SECONDS, SEEDS, UPDATES,
    expected_training_contract, job_key, planned_jobs, read_json, write_json,
)
from .r7_m3_worker import train_output_dir, worker_result_path


def _exclusive_csv(path, fields, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _verify_training(entry, job, protocol):
    expected_folder = train_output_dir(protocol["output"], job["seed"], job["arm"])
    checkpoint = Path(entry["checkpoint"])
    selected = entry["selected_update"]
    if (entry["updates_run"] != UPDATES or entry["early_stopped"]
            or isinstance(selected, bool) or selected not in (100, 200, 300, 400)
            or checkpoint.resolve() != (expected_folder / f"update_{selected:07d}.pt").resolve()
            or checkpoint.is_symlink() or sha256_file(checkpoint) != entry["checkpoint_sha256"]):
        raise ValueError("fresh selected validation checkpoint / exact 400 updates required")
    report_path = expected_folder / "training_report.json"
    if (Path(entry["training_report"]).resolve() != report_path.resolve()
            or sha256_file(report_path) != entry["training_report_sha256"]):
        raise ValueError("training report bytes/path mismatch")
    report = read_json(report_path)
    if (report["updates_this_run"] != UPDATES or report["early_stopped"]
            or report["selected_update"] != selected or report["selection_split"] != "val"
            or report["contract"]["process_supervision"] != expected_training_contract(protocol, job["arm"])
            or report["contract"]["data_identity"] != protocol["data"]["data_identity"]
            or [item["update"] for item in report["validations"]] != [100, 200, 300, 400]
            or [item["update"] for item in report["losses"]] != list(range(1, UPDATES + 1))):
        raise ValueError("training must cover all updates/validation checks under the new pinned contract")
    expected_hash = protocol["cpu_profile"]["pairing"][str(job["seed"])]["full_initial_state_sha256"][job["arm"]]
    if entry["initial_state_sha256"] != expected_hash:
        raise ValueError("actual initial state hash differs from frozen pairing")
    transfer = entry["shared_initial_state"]
    if not transfer["provided"] or transfer["ignored_count"] or transfer["applied_count"] < 1:
        raise ValueError("full same-architecture state transfer was not recorded")


def _verify_metric_rows(entry, protocol):
    channels, units = protocol["data"]["channels"], protocol["data"]["units"]
    expected = {(name, unit, float(entry["lead_hours"])) for name, unit in zip(channels, units)}
    rows = read_rmse_rows(entry["evaluation_dir"])
    if {(row["variable"], row["unit"], row["lead_hours"]) for row in rows} != expected or len(rows) != 17:
        raise ValueError("all 17 physical RMSE channels and only this frozen lead required")
    if any(row["n_initializations"] != entry["n_evaluated"] for row in rows):
        raise ValueError("RMSE count differs from exact cases")
    for filename in ("acc.csv", "climatology_skill.csv"):
        values = _read_csv(Path(entry["evaluation_dir"]) / filename)
        if (len(values) != 17 or {row["variable"] for row in values} != set(channels)
                or any(float(row["lead_hours"]) != entry["lead_hours"]
                       or int(row["n_initializations"]) != entry["n_evaluated"] for row in values)):
            raise ValueError("ACC/climatology skill must cover the exact 17-channel case set")
        for row in values:
            if filename == "acc.csv":
                if row["status"] == "defined":
                    if not math.isfinite(float(row["pooled_acc"])):
                        raise ValueError("nonfinite defined ACC")
                elif row["status"] != "undefined_zero_anomaly_energy" or row["pooled_acc"] != "":
                    raise ValueError("undefined ACC must retain its explicit status, not skip")
            else:
                index = channels.index(row["variable"])
                if row["unit"] != units[index]:
                    raise ValueError("climatology skill physical unit mismatch")
                for key in ("rmse_forecast", "rmse_climatology"):
                    if not math.isfinite(float(row[key])) or float(row[key]) < 0:
                        raise ValueError("invalid physical climatology/forecast RMSE")
                if row["mse_skill"] and not math.isfinite(float(row["mse_skill"])):
                    raise ValueError("nonfinite climatology skill")
    return len(rows)


def _verify_evaluation(entry, job, protocol, training):
    expected_dir = Path(protocol["output"]) / f"seed{job['seed']}" / "evaluation" / job["arm"] / f"lead_{job['lead']:03d}h"
    if (Path(entry["evaluation_dir"]).resolve() != expected_dir.resolve()
            or entry["arm"] != job["arm"] or entry["lead_hours"] != job["lead"]
            or entry["split"] != "val" or entry["test_read"] is not False
            or entry["channels"] != protocol["data"]["channels"] or entry["units"] != protocol["data"]["units"]
            or entry["baseline"] != {"allocated_bytes": 0, "reserved_bytes": 0}
            or entry["checkpoint_sha256"] != training["checkpoint_sha256"]
            or entry["checkpoint"] != training["checkpoint"]):
        raise ValueError("evaluation job/case/physical metadata or fresh zero baseline mismatch")
    for filename in ("rmse.csv", "acc.csv", "climatology_skill.csv", "provenance.json"):
        if sha256_file(expected_dir / filename) != entry["artifact_sha256"][filename]:
            raise ValueError("evaluation artifact bytes changed")
    cases = read_case_identity(expected_dir)
    declared = protocol["data"]["evaluation_cases"][str(job["lead"])]
    if (cases is None or cases["cases"] != declared["cases"] or cases["split"] != "val"
            or cases["evaluation_manifest_sha256"] != protocol["sources"]["val_manifest_sha256"]
            or entry["n_available_windows"] != declared["n_available"]
            or cases["n_evaluated"] != len(declared["cases"])
            or entry["n_evaluated"] != len(declared["cases"])):
        raise ValueError("exact frozen initialization/valid-time identities must be paired at every lead")
    provenance = read_json(expected_dir / "provenance.json")
    if (provenance["checkpoint_sha256"] != training["checkpoint_sha256"]
            or provenance["training_identity"] != protocol["data"]["data_identity"]
            or provenance["process_scale_sidecar_identity"] != protocol["sidecar"]["identity"]
            or provenance["training_protocol_sha256"] != protocol["protocol_sha256"]):
        raise ValueError("actual evaluation sidecar/checkpoint/dataset/protocol pins mismatch")
    return _verify_metric_rows(entry, protocol)


def validate_full_set(output, protocol):
    """Read only the EXACT 36 planned receipts; never a partial glob merge."""
    training, evaluations = {}, {}
    cells = 0
    for job in planned_jobs():
        entry = read_json(worker_result_path(output, job))
        if (entry.get("status") != "success" or entry.get("scientific_claim") is not False
                or not entry.get("limitations") or entry.get("test_read") is not False
                or entry.get("job") != job or entry.get("protocol_sha256") != protocol["protocol_sha256"]
                or entry.get("model_code_sha256") != protocol["code"]["model_code_sha256"]
                or entry.get("data_identity") != protocol["data"]["data_identity"]
                or entry.get("process_supervision") != expected_training_contract(protocol, job["arm"])):
            raise ValueError("missing/failed/skipped/wrong-seed arm receipt or changed protocol identity")
        for key in ("elapsed_seconds", "peak_allocated_bytes", "peak_reserved_bytes"):
            if isinstance(entry[key], bool) or not isinstance(entry[key], (float, int)) or not math.isfinite(entry[key]) or entry[key] < 0:
                raise ValueError("actual finite training/evaluation elapsed and allocator peaks required")
        identity = (job["seed"], job["arm"])
        if job["phase"] == "train":
            _verify_training(entry, job, protocol)
            training[identity] = entry
        else:
            cells += _verify_evaluation(entry, job, protocol, training[identity])
            evaluations[(*identity, job["lead"])] = entry
    if len(training) != 6 or len(evaluations) != 30 or cells != 510:
        raise ValueError("M3 requires 6 training/30 eval/510 RMSE cells, no partial aggregate")
    return training, evaluations


def _seed_results(output, protocol, training, evaluations):
    for seed in SEEDS:
        per_arm = {arm: training[(seed, arm)] for arm in ARM_NAMES}
        evaluation = {f"{arm}@{lead}h": evaluations[(seed, arm, lead)] for arm in ARM_NAMES for lead in LEADS}
        result = {"format": RESULT_FORMAT, "scientific_claim": False, "limitations": LIMITATIONS,
                  "test_read": False, "seed": seed, "seeds": list(SEEDS), "device": "cuda:0",
                  "gpu": protocol["gpu"]["uuid"], "torch_version": per_arm[ARM_NAMES[0]]["torch_version"],
                  "model_code_sha256": protocol["code"]["model_code_sha256"], "protocol": protocol,
                  "protocol_sha256": protocol["protocol_sha256"],
                  "arm_pairing": protocol["cpu_profile"]["pairing"][str(seed)],
                  "flop_measurements": protocol["cpu_profile"]["measurements"],
                  "training": per_arm, "evaluation": evaluation, "probes": {},
                  "case_counts_by_lead": {str(lead): len(protocol["data"]["evaluation_cases"][str(lead)]["cases"]) for lead in LEADS},
                  "budget": {"training_seconds_total": sum(entry["elapsed_seconds"] for entry in per_arm.values())}}
        write_json(Path(output) / f"seed{seed}" / "seed_result.json", result)


def _supplementary_tables(output, merged):
    """Keep shared harness tables unchanged; separate cost views and reserved eval peaks."""
    arms = merged["protocol"]["arms"]
    _exclusive_csv(Path(output) / "parameter_table.csv", ["arm", "parameters", "trainable_parameters"],
                   [{key: arm[key] for key in ("parameters", "trainable_parameters")} | {"arm": arm["name"]} for arm in arms])
    _exclusive_csv(Path(output) / "flops_table.csv", ["arm", "forward_flops", "forward_backward_flops"],
                   [{key: arm[key] for key in ("forward_flops", "forward_backward_flops")} | {"arm": arm["name"]} for arm in arms])
    memory = []
    throughput = []
    for seed, entries in merged["training"].items():
        for arm, entry in entries.items():
            throughput.append({"seed": seed, "arm": arm, "updates": entry["updates_run"],
                               "seconds_per_update": entry["seconds_per_update"], "elapsed_seconds": entry["elapsed_seconds"]})
            memory.append({"seed": seed, "arm": arm, "phase": "train", "lead_hours": "",
                           "baseline_allocated_bytes": "", "baseline_reserved_bytes": "",
                           "peak_allocated_bytes": entry["peak_allocated_bytes"], "peak_reserved_bytes": entry["peak_reserved_bytes"]})
    acc_rows = []
    for entry in merged["evaluation"].values():
        memory.append({"seed": entry["seed"], "arm": entry["arm"], "phase": "evaluate", "lead_hours": entry["lead_hours"],
                       "baseline_allocated_bytes": entry["baseline"]["allocated_bytes"],
                       "baseline_reserved_bytes": entry["baseline"]["reserved_bytes"],
                       "peak_allocated_bytes": entry["peak_allocated_bytes"], "peak_reserved_bytes": entry["peak_reserved_bytes"]})
        for row in _read_csv(entry["acc_csv"]):
            acc_rows.append({"seed": entry["seed"], "arm": entry["arm"], **row})
    _exclusive_csv(Path(output) / "training_throughput_table.csv", list(throughput[0]), throughput)
    _exclusive_csv(Path(output) / "allocator_table.csv", list(memory[0]), memory)
    _exclusive_csv(Path(output) / "acc_table.csv", list(acc_rows[0]), acc_rows)


def descriptive_outcome(pairs):
    unresolved = {name: [key for key, cell in pair["cells"].items() if cell["outcome"] == "unresolved"]
                  for name, pair in pairs.items()}
    paused = any(unresolved.values())
    return {"scientific_claim": False, "limitations": LIMITATIONS, "paused": paused,
            "status": "paused" if paused else "descriptive-complete",
            "advance_next_node": False, "unresolved_cells": unresolved,
            "rule": "all endpoints reported; unresolved pauses; improved/worsened/mixed retained without significance",
            "pair_totals": {name: pair["totals"] for name, pair in pairs.items()},
            "scientific_gate_evaluated": False}


def finalize(output, protocol, execution):
    elapsed = execution["gpu_phase_elapsed_seconds"]
    if (execution["status"] != "success" or execution["jobs_completed"] != planned_jobs()
            or not math.isfinite(elapsed) or elapsed < 0 or elapsed > ROUND_SECONDS):
        raise ValueError("partial or over-budget execution cannot finalize")
    training, evaluations = validate_full_set(output, protocol)
    _seed_results(output, protocol, training, evaluations)
    merged = merge_seed_results(output, seeds=SEEDS, arms=ARM_NAMES, fmt=RESULT_FORMAT)
    merged.update(limitations=LIMITATIONS, cpu_gradient_ownership=protocol["cpu_profile"]["gradient_ownership"])
    merged["budget"].update(gpu_phase_elapsed_seconds=elapsed, gpu_hours_charged=elapsed / 3600.0,
                           cap_round_seconds=ROUND_SECONDS, continuous_clock=True,
                           billing_scope=protocol["gpu"]["billing"])
    identity = {"dataset_identity": protocol["data"]["data_identity"],
                "model_code_sha256": protocol["code"]["model_code_sha256"],
                "sidecar_identity": protocol["sidecar"]["identity"],
                "declared_update_budget": UPDATES, "evaluation_split": "val"}
    table, blocks = comparator_blocks(merged, pairs=PAIRS, depth=0, identity=identity)
    pairs = pair_cells(blocks, leads=LEADS)
    expected_cells = {f"{lead}h|{channel}" for lead in LEADS for channel in protocol["data"]["channels"]}
    if len(table) != 255 or set(pairs) != {f"{first} - {second}" for first, second in PAIRS}:
        raise ValueError("complete 255 aggregate rows and all declared pairs required")
    for pair in pairs.values():
        if set(pair["cells"]) != expected_cells or len(pair["cells"]) != 85:
            raise ValueError("all 85 paired cells for each declared comparison required")
        if any(set(cell["seed_deltas"]) != {str(seed) for seed in SEEDS} for cell in pair["cells"].values()):
            raise ValueError("all exact seed deltas required in each paired cell")
    outcome = descriptive_outcome(pairs)
    comparison = {"format": "r7-m3-process-supervision-comparison-v1", "scientific_claim": False,
                  "limitations": LIMITATIONS, "test_read": False, "protocol_sha256": protocol["protocol_sha256"],
                  "identity": identity, "pairs": pairs, "table": table, "outcome": outcome}
    write_json(Path(output) / "merged_result.json", merged)
    write_json(Path(output) / "paired_comparison.json", comparison)
    write_study_tables(merged, output)
    _supplementary_tables(output, merged)
    files = ["merged_result.json", "paired_comparison.json", "protocol.json", "cpu_profile.json",
             "code.zip", "code_commit.txt", "code_status.txt", "execution_attempt.json",
             "arm_table.csv", "training_table.csv", "memory_table.csv", "rmse_table.csv", "case_table.csv",
             "parameter_table.csv", "flops_table.csv", "training_throughput_table.csv", "allocator_table.csv", "acc_table.csv"]
    artifact_pins = {name: sha256_file(Path(output) / name) for name in files}
    for job in planned_jobs():
        for suffix in (".json", ".timing.json", ".log"):
            name = "workers/" + job_key(job) + suffix
            artifact_pins[name] = sha256_file(Path(output) / name)
    write_json(Path(output) / "artifact_manifest.json", {"scientific_claim": False, "limitations": LIMITATIONS,
                                                         "protocol_sha256": protocol["protocol_sha256"],
                                                         "files_sha256": artifact_pins, "outcome": outcome})
    return outcome
