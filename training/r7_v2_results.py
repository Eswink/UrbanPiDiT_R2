"""Fail-closed independent B/C receipts; stage seal is NOT the driver's final cost seal."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median

from .r7_v2_protocol import (
    B_SEEDS, C_SEEDS, LEADS, REGIONS, child_contract, digest, evaluation_dir, job_key,
    local_path, read_json, safe_output, sha256_file, train_output_dir, validate_protocol,
    worker_result_path, write_json, write_path,
)
from .r7_v2_inventory import EXCLUDED_RECURSIVE_OR_FUTURE, pin_inventory, verify_inventory
from .r7_v2_tables import (
    STATISTICS, aggregate_metrics, case_key, close, integer, number, paired_comparisons,
    candidate_selection, publish_tables, validate_metrics,
)


def _csv(path):
    with local_path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _list(value):
    result = json.loads(value) if isinstance(value, str) else value
    if not isinstance(result, list):
        raise ValueError("explicit list required in metric artifact")
    return result


def _canonical_metric(row, data, *, pooled=False):
    variable = row["variable"]
    index = data["channels"].index(variable)
    std = number(data["normalization_std"][index], nonnegative=True)
    if std <= 0 or row["unit"] != data["units"][index]:
        raise ValueError("physical metric needs pinned positive training std and exact unit")
    if row["acc_statistic_units"] != "normalized_anomaly_squared":
        raise ValueError("unknown ACC statistic scale; no guessed physical conversion")
    n = integer(row["n_initializations"], minimum=1)
    factor = std ** 2 / n if pooled else std ** 2
    if not pooled and n != 1:
        raise ValueError("each initialization must have n_initializations=1")
    close(row["rmse"], number(row["mse"], nonnegative=True) ** .5, "rmse/MSE")
    close(row["rmse_climatology"], number(row["climatology_mse"], nonnegative=True) ** .5, "climatology RMSE/MSE")
    close(row["mse_climatology"], number(row["climatology_mse"], nonnegative=True), "climatology alias")
    canonical = {key: row[key] for key in ("region", "variable", "unit", "lead_hours", "rmse", "rmse_climatology", "mse_skill")}
    canonical["lead_hours"] = integer(row["lead_hours"], minimum=1)
    canonical.update(mse=number(row["mse"], nonnegative=True),
                     climatology_mse=number(row["climatology_mse"], nonnegative=True),
                     pooled_acc=row["acc"], n_initializations=n, acc_status=row["acc_status"],
                     skill_status="undefined_zero_climatology_mse" if row["skill_status"] ==
                     "undefined_zero_climatology_energy" else row["skill_status"])
    canonical.update({key: number(row[key], nonnegative=key != "acc_dot") * factor for key in STATISTICS[2:]})
    if not pooled:
        canonical.update(init_time=row["init_time"], valid_times=_list(row["valid_times"]))
    return canonical


def _bad_reasons(row):
    return [name for name, bad in (
        ("negative_mse_skill", row["mse_skill"] not in (None, "") and number(row["mse_skill"]) < 0),
        ("negative_acc", row["acc"] not in (None, "") and number(row["acc"]) < 0),
        ("undefined_mse_skill", row["mse_skill"] in (None, "")),
        ("undefined_acc", row["acc"] in (None, ""))) if bad]


def _row_key(row, *, per_case=False):
    cell = (row["region"], row["variable"], integer(row["lead_hours"], minimum=1))
    return (row["init_time"], *cell) if per_case else cell


def _same_metric(actual, expected, data, *, pooled=False):
    first, second = _canonical_metric(actual, data, pooled=pooled), _canonical_metric(expected, data, pooled=pooled)
    for key in first:
        if key in (*STATISTICS, "rmse", "rmse_climatology", "mse_skill", "pooled_acc") and second[key] is not None:
            close(first[key], number(second[key]), f"CSV/provenance {key}")
        elif first[key] != second[key] and not (first[key] == "" and second[key] is None):
            raise ValueError(f"CSV/provenance metadata mismatch: {key}")
    for key in ("margin_cells", "n_grid_points", "full_area_fraction"):
        close(actual[key], number(expected[key]), f"region geometry {key}")
    if _list(actual["bad_reasons"]) != _bad_reasons(actual) or _list(expected["bad_reasons"]) != _bad_reasons(expected):
        raise ValueError("bad cases/variables must be retained and explicitly labeled")


def verify_metric_artifacts(provenance, region_rows, case_rows, protocol, job):
    """Adapt ONLY the actual evaluate_v2 schema and verify every redundant artifact."""
    data, lead, kernel = protocol["data"], job["lead"], job["reasoning_steps"]
    declared = data["evaluation_cases"][str(lead)]
    expected_cases = declared["cases"]
    if (provenance["scientific_claim"] is not False or not provenance["limitations"]
            or provenance["test_read"] is not False or provenance["split"] != "val"
            or provenance["lead_hours"] != [lead] or provenance["step_hours"] != 6
            or provenance["channels"] != data["channels"] or provenance["units"] != data["units"]
            or provenance["n_evaluated"] != len(expected_cases)
            or provenance["n_available_windows"] != declared["n_available"]
            or provenance["reasoning_steps"] != kernel or provenance["checkpoint_training_steps"] != 4
            or provenance["independently_trained_k1"] is not False
            or provenance["inference_options"] != {"reasoning_steps": kernel}
            or provenance["training_identity"] != data["data_identity"]
            or provenance["evaluation_manifest_sha256"] != protocol["sources"]["val_manifest_sha256"]):
        raise ValueError("evaluation provenance physical/case/K/data identity mismatch")
    embedded = []
    inits = provenance["initializations"]
    if sorted([[item["init_time"], item["valid_times"]] for item in inits]) != sorted(expected_cases):
        raise ValueError("exact per-lead cohort mismatch; no 72h narrowing or count-only acceptance")
    for item in inits:
        case_key(item["init_time"], item["valid_times"], lead)
        work = item["cumulative_reasoning_steps"]
        if (not isinstance(work, list) or len(work) != 1 or type(work[0]) is not int
                or work[0] != kernel * lead // 6):
            raise ValueError("actual single-lead cumulative rollout kernel count mismatch")
        if len(item["region_metrics"]) != 51:
            raise ValueError("every case must carry all 17 variables in all three regions")
        embedded.extend({**row, "init_time": item["init_time"], "valid_times": item["valid_times"],
                         "sample_id": item["sample_id"], "reasoning_steps": kernel}
                        for row in item["region_metrics"])
    expected_by_key = {_row_key(row, per_case=True): row for row in embedded}
    if len(expected_by_key) != len(embedded) or len(case_rows) != len(embedded):
        raise ValueError("duplicate or incomplete per-case CSV/embedded metrics")
    seen = set()
    for row in case_rows:
        key = _row_key(row, per_case=True)
        if key not in expected_by_key or key in seen:
            raise ValueError("per-case CSV initialization/region/variable mismatch")
        seen.add(key)
        expected = expected_by_key[key]
        if (row["sample_id"] != expected["sample_id"] or _list(row["valid_times"]) != expected["valid_times"]
                or row["valid_time"] != expected["valid_times"][0]
                or integer(row["reasoning_steps"], minimum=1) != kernel):
            raise ValueError("per-case CSV seed-independent sample/time/K identity mismatch")
        _same_metric(row, expected, data)
    declared_regions = {_row_key(row): row for row in provenance["region_metrics"]}
    if len(declared_regions) != 51 or len(region_rows) != 51:
        raise ValueError("complete regional metrics CSV/provenance required")
    seen = set()
    for row in region_rows:
        key = _row_key(row)
        if key not in declared_regions or key in seen:
            raise ValueError("duplicate/missing regional CSV metric")
        seen.add(key)
        _same_metric(row, declared_regions[key], data, pooled=True)
    if provenance["bad_variable_metrics"] != [row for row in provenance["region_metrics"] if _bad_reasons(row)]:
        raise ValueError("bad variable registry was filtered or changed")
    expected_bad = [{**row, "valid_time": row["valid_times"][0]} for row in embedded if _bad_reasons(row)]
    if len(provenance["bad_case_metrics"]) != len(expected_bad):
        raise ValueError("bad case registry count differs")
    bad_by_key = {_row_key(row, per_case=True): row for row in provenance["bad_case_metrics"]}
    if set(bad_by_key) != {_row_key(row, per_case=True) for row in expected_bad}:
        raise ValueError("bad case registry has missing or unexpected case")
    for row in expected_bad:
        _same_metric(bad_by_key[_row_key(row, per_case=True)], row, data)
    verified = validate_metrics([_canonical_metric(row, data, pooled=True) for row in region_rows],
                                [_canonical_metric(row, data) for row in case_rows],
                                cases=expected_cases, channels=data["channels"], units=data["units"],
                                lead=lead, regions=REGIONS)
    for row in verified:
        row.update(seed=job["seed"], arm=job["arm"], K=kernel)
    return verified


def verify_baseline_artifacts(provenance, region_rows, case_rows, protocol, job):
    """Mandatory zero-trained references, exact same model cases, never pseudo K models."""
    from .r7_v2_worker import verify_baseline_provenance

    verify_baseline_provenance(protocol, job, provenance)
    data, lead = protocol["data"], job["lead"]
    combined = provenance["baseline_region_metrics"]
    embedded = [{**row, "init_time": item["init_time"], "valid_times": item["valid_times"],
                 "sample_id": item["sample_id"]} for item in provenance["baseline_initializations"]
                for row in item["region_metrics"]]
    def key(row, per_case=False):
        return (row["baseline_kind"], *_row_key(row, per_case=per_case))
    expected_rows = {key(row): row for row in combined}
    expected_cases = {key(row, True): row for row in embedded}
    if (len(expected_rows) != len(combined) or len(expected_cases) != len(embedded)
            or len(region_rows) != len(combined) or len(case_rows) != len(embedded)):
        raise ValueError("complete unique two-baseline CSV/provenance rows required")
    for actual_rows, expected, per_case in ((region_rows, expected_rows, False), (case_rows, expected_cases, True)):
        seen = set()
        for row in actual_rows:
            identity = key(row, per_case)
            if identity not in expected or identity in seen:
                raise ValueError("baseline CSV missing/duplicate/undeclared identity")
            seen.add(identity)
            if any(integer(row[field]) != 0 for field in ("zero_train_updates", "parameters", "trainable_parameters")):
                raise ValueError("persistence/climatology must remain zero-trained, zero-parameter references")
            if per_case and (row["sample_id"] != expected[identity]["sample_id"]
                             or _list(row["valid_times"]) != expected[identity]["valid_times"]
                             or row["valid_time"] != expected[identity]["valid_times"][0]
                             or row["reasoning_steps"] not in (None, "")):
                raise ValueError("baseline exact sample/time or zero-training depth identity mismatch")
            _same_metric(row, expected[identity], data, pooled=not per_case)
    records = []
    for kind in ("persistence", "climatology"):
        verified = validate_metrics([_canonical_metric(row, data, pooled=True) for row in region_rows if row["baseline_kind"] == kind],
                                    [_canonical_metric(row, data) for row in case_rows if row["baseline_kind"] == kind],
                                    cases=data["evaluation_cases"][str(lead)]["cases"], channels=data["channels"],
                                    units=data["units"], lead=lead, regions=REGIONS)
        for row in verified:
            if kind == "climatology":
                close(row["mse"], row["climatology_mse"], "zero-trained climatology forecast")
                if row["acc_forecast_energy"] != 0 or row["acc_dot"] != 0 or row["pooled_acc"] is not None:
                    raise ValueError("climatology forecast anomaly energy must be zero/undefined ACC")
            row.update(baseline_kind=kind, zero_train_updates=0, parameters=0, trainable_parameters=0)
            row["semantic_sha256"] = digest(row)
            row.update(seed=job["seed"], reference_arm=job["arm"], reference_K=job["reasoning_steps"],
                       model_depth_applicable=False)
            records.append(row)
    return records


def _hash(path, expected, output):
    path = write_path(path, output)
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"artifact bytes/path mismatch: {path}")
    return path


def _profile(protocol):
    profile = protocol["cpu_profile"]
    seeds = B_SEEDS if protocol["stage"] == "B" else C_SEEDS
    arms = protocol["arm_configs"]
    if (profile["device"] != "cpu" or profile["scientific_claim"] is not False
            or not profile["limitations"] or profile["test_read"] is not False
            or profile["no_optimizer_step"] is not True or profile["full_internal_k_bptt"] is not True
            or profile["full_physical_step_bptt"] is not True
            or set(profile["measurements"]) != {str(seed) for seed in seeds}
            or set(profile["pairing"]) != {str(seed) for seed in seeds}):
        raise ValueError("complete frozen actual CPU profile/initialization required")
    for seed in seeds:
        if set(profile["measurements"][str(seed)]) != set(arms) or set(profile["pairing"][str(seed)]) != set(arms):
            raise ValueError("complete CPU measurements for every seed/arm required")
        for arm, config in arms.items():
            measured = profile["measurements"][str(seed)][arm]
            calls = 2 if config["mode"] == "two_step" else 1
            for key in ("parameters", "trainable_parameters", "forward_flops", "forward_backward_flops"):
                integer(measured[key], minimum=1)
            if (measured["forward_backward_flops"] < measured["forward_flops"]
                    or measured["actual_model_forward_calls"] != calls or measured["physical_steps"] != calls
                    or measured["objective"] != config["mode"] or measured["reasoning_steps"] != 4
                    or number(measured["gradient_norm"], nonnegative=True) <= 0
                    or integer(measured["nonzero_gradient_tensors"], minimum=1) > measured["gradient_tensors"]
                    or not measured["forward_scope"] or not measured["backward_scope"]):
                raise ValueError("actual objective one/two forwards and measured real backward required")
        pairing = list(profile["pairing"][str(seed)].values())
        key = "full_initial_state_sha256" if protocol["stage"] == "B" else "anchor_state_sha256"
        if len({item[key] for item in pairing}) != 1 or any(not item[key] for item in pairing):
            raise ValueError("exact same-seed B imported weights / C mapped anchor required")


def _verify_common(entry, protocol, job):
    expected = {"protocol_sha256": protocol["protocol_sha256"],
                "model_code_sha256": protocol["code"]["model_code_sha256"],
                "source_tree_sha256": protocol["code"]["source_tree_sha256"],
                "code_zip_sha256": protocol["code"]["code_zip_sha256"],
                "data_identity": protocol["data"]["data_identity"], "source_sha256": protocol["sources"]["source_sha256"],
                "sidecar_identity": protocol["sidecar"]["identity"], "windows_sha256": digest(protocol["windows"]),
                "windows": protocol["windows"], "sources": protocol["sources"], "sidecar": protocol["sidecar"],
                "arm_config": protocol["arm_configs"][job["arm"]], "shared_controls": protocol["shared_controls"],
                "cpu_profile": protocol["cpu_profile"]["measurements"][str(job["seed"])][job["arm"]],
                "parent_provenance": protocol["parents"].get(str(job["seed"]))}
    receipt_job = entry["job"]
    if any(type(receipt_job[key]) is not int for key in ("seed", "reasoning_steps")):
        raise ValueError("exact integer seed/K receipt identity required")
    if receipt_job["lead"] is not None and type(receipt_job["lead"]) is not int:
        raise ValueError("exact integer lead receipt identity required")
    if (entry["status"] != "success" or receipt_job != job or entry["scientific_claim"] is not False
            or not entry["limitations"] or entry["test_read"] is not False
            or any(entry[key] != value for key, value in expected.items())):
        raise ValueError("success/exact job/protocol/model/source/data/parent receipt required; no skip or partial")
    for key in ("elapsed_seconds", "peak_allocated_bytes", "peak_reserved_bytes"):
        number(entry[key], nonnegative=True)
    for key in ("baseline", "pre_init_baseline", "initialized_baseline"):
        if entry[key] != {"allocated_bytes": 0, "reserved_bytes": 0}:
            raise ValueError("fresh measured pre-init and initialized zero allocator baseline required")
    if entry["peak_reserved_bytes"] < entry["peak_allocated_bytes"]:
        raise ValueError("allocator reserved peak smaller than allocated peak")


def _verify_training(entry, protocol, job, output):
    config, controls = protocol["arm_configs"][job["arm"]], protocol["shared_controls"]
    updates = config["updates"]
    directory = train_output_dir(output, job["seed"], job["arm"])
    if (entry["updates_run"] != updates or entry["selected_update"] != updates
            or entry["parent_optimizer_imported"] is not False or entry["resume"] is not False
            or local_path(entry["checkpoint"]) != directory / f"update_{updates:07d}.pt"
            or local_path(entry["training_report"]) != directory / "training_report.json"):
        raise ValueError("new own exact final endpoint checkpoint/report required; no resume or validation selection")
    _hash(entry["checkpoint"], entry["checkpoint_sha256"], output)
    report = read_json(_hash(entry["training_report"], entry["training_report_sha256"], output))
    contract = entry["contract"]
    pairing = protocol["cpu_profile"]["pairing"][str(job["seed"])][job["arm"]]
    initialization = entry["initialization"]
    base = child_contract(protocol, job, pairing["model_spec"])
    base.update(initialization=initialization, parent_provenance=initialization if protocol["stage"] == "B" else None)
    if (any(contract.get(key) != value for key, value in base.items() if key != "autoregression")
            or any(contract["autoregression"].get(key) != value for key, value in base["autoregression"].items())
            or digest(initialization) != pairing["initialization_report_sha256"]
            or entry["initial_state_sha256"] != pairing["full_initial_state_sha256"]
            or contract["model"].get("detach_between_steps") is not False
            or entry["signature"] != digest(contract) or report != entry["report"]
            or report["contract"] != contract or report["signature"] != entry["signature"]
            or report["updates_this_run"] != updates or report["total_updates"] != updates
            or report["selected_update"] != updates or report["resumed_from_updates"] != 0
            or report["selection_split"] is not None or report["selected_checkpoint"] != entry["checkpoint"]
            or report["windows"] != protocol["windows"] or report["scientific_claim"] is not False
            or report["test_read"] is not False or not report["limitations"]
            or report["physical_step_detach"] is not False or report["internal_k_detach"] is not False
            or report["parent_optimizer_imported"] is not False
            or [row["update"] for row in report["losses"]] != list(range(1, updates + 1))):
        raise ValueError("full exact new training recipe/initialization/contract/report required")
    options = {"seed": job["seed"], "steps": 4, "total_updates": updates, "mode": config["mode"],
               "lr": controls["lr"], "warmup_updates": controls["warmup"], "weight_decay": controls["weight_decay"],
               "batch_size": controls["batch_size"], "clip": controls["clip"], "bf16": controls["bf16"],
               "checkpoint_every": controls["checkpoint_every"], "output_dir": str(directory), "device_type": "cuda"}
    if any(contract.get(key) != value for key, value in options.items()):
        raise ValueError("training actual options differ from frozen controls")
    for loss in report["losses"]:
        number(loss["loss"], nonnegative=True); number(loss["l6"], nonnegative=True)
        number(loss["gradient_norm"], nonnegative=True)
        from .r7_v2_objective_receipt import verify_training_objective
        verify_training_objective(loss, mode=config["mode"], controls=controls, contract=contract)
    number(report["elapsed_seconds"], nonnegative=True)
    return report


def _verify_evaluation(entry, protocol, job, training, output):
    directory = evaluation_dir(output, job)
    if (local_path(entry["evaluation_dir"]) != directory or entry["split"] != "val"
            or entry["lead_hours"] != job["lead"] or entry["reasoning_steps"] != job["reasoning_steps"]
            or entry["channels"] != protocol["data"]["channels"] or entry["units"] != protocol["data"]["units"]
            or entry["checkpoint"] != training["checkpoint"] or entry["checkpoint_sha256"] != training["checkpoint_sha256"]):
        raise ValueError("evaluation seed/arm/K/checkpoint/physical metadata mismatch")
    names = {"region_metrics.csv": "region_metrics_csv", "per_case_metrics.csv": "per_case_metrics_csv", "provenance.json": "provenance",
             "baseline_region_metrics.csv": "baseline_region_metrics_csv", "baseline_per_case_metrics.csv": "baseline_per_case_metrics_csv"}
    if set(entry["artifact_sha256"]) != set(names):
        raise ValueError("exact five model/mandatory baseline evaluation artifact hashes required")
    for name, field in names.items():
        if local_path(entry[field]) != directory / name:
            raise ValueError("evaluation artifact path differs from own frozen job directory")
        _hash(entry[field], entry["artifact_sha256"][name], output)
    provenance = read_json(entry["provenance"])
    if (provenance != entry["evaluation_provenance"] or provenance["checkpoint"] != training["checkpoint"]
            or provenance["checkpoint_sha256"] != training["checkpoint_sha256"]
            or provenance["checkpoint_contract_sha256"] != training["signature"]
            or provenance["model_code_sha256"] != protocol["code"]["model_code_sha256"]
            or provenance["model_kind"] != training["contract"]["kind"]
            or provenance["model_config"] != training["contract"]["model"]
            or provenance["process_scale_sidecar_identity"] is not None
            or provenance["n_evaluated"] != entry["n_evaluated"]
            or provenance["n_available_windows"] != entry["n_available_windows"]):
        raise ValueError("actual provenance/checkpoint/current no-aux contract mismatch")
    isolated = provenance["isolated_forward"]
    timings = [number(value, nonnegative=True) for value in isolated["seconds_per_batch"]]
    if (isolated["gpu_latency_measured"] is not True or isolated["reasoning_steps"] != job["reasoning_steps"]
            or isolated["repetitions"] != 10 or len(timings) != 10 or min(timings) <= 0
            or isolated["actual_reasoning_steps_per_sample"] != [[job["reasoning_steps"]]] * 10):
        raise ValueError("positive measured isolated GPU latency and actual K required")
    close(isolated["median_seconds_per_batch"], median(timings), "isolated latency median")
    close(isolated["mean_seconds_per_batch"], sum(timings) / 10, "isolated latency mean")
    number(provenance["loop_elapsed_seconds"], nonnegative=True)
    number(provenance["whole_elapsed_seconds"], nonnegative=True)
    entry["baseline_metrics"] = verify_baseline_artifacts(provenance, _csv(entry["baseline_region_metrics_csv"]),
                                                          _csv(entry["baseline_per_case_metrics_csv"]), protocol, job)
    return verify_metric_artifacts(provenance, _csv(entry["region_metrics_csv"]), _csv(entry["per_case_metrics_csv"]), protocol, job)


def validate_full_set(output, protocol):
    """Read exactly protocol.jobs; every stage's full seed/arm/lead/K set is mandatory."""
    output = safe_output(output)
    validate_protocol(protocol)
    if str(output) != protocol["output"] or read_json(output / "protocol.json") != protocol:
        raise ValueError("protocol file/output changed")
    if (read_json(output / "cpu_profile.json") != protocol["cpu_profile"]
            or digest(protocol["code"]["files"]) != protocol["code"]["source_tree_sha256"]):
        raise ValueError("CPU profile/source archive identity changed")
    _hash(output / "code.zip", protocol["code"]["code_zip_sha256"], output)
    code = protocol["code"]
    for name, key in (("code_commit.txt", "code_commit_sha256"), ("code_status.txt", "code_status_sha256")):
        _hash(output / name, code[key], output)
    if ((output / "code_commit.txt").read_text(encoding="utf-8").strip() != code["base_commit"]
            or bool((output / "code_status.txt").read_text(encoding="utf-8").strip()) != code["working_tree_modified"]):
        raise ValueError("frozen Git identity bytes/declared HEAD or dirty status mismatch")
    _profile(protocol)
    counts = {6: 22, 12: 21, 24: 19, 48: 15, 72: 11}
    if any(len(protocol["data"]["evaluation_cases"][str(lead)]["cases"]) != counts[lead] for lead in LEADS):
        raise ValueError("complete per-lead 22/21/19/15/11 cohorts required")
    training, evaluations, records, baseline_semantics = {}, {}, [], {}
    for job in protocol["jobs"]:
        path = worker_result_path(output, job)
        entry = read_json(path)
        _verify_common(entry, protocol, job)
        entry["receipt_sha256"] = sha256_file(path)
        key = (job["seed"], job["arm"])
        if job["phase"] == "train":
            _verify_training(entry, protocol, job, output)
            training[key] = entry
        else:
            records.extend(_verify_evaluation(entry, protocol, job, training[key], output))
            for row in entry["baseline_metrics"]:
                baseline_key = (row["seed"], row["baseline_kind"], row["lead_hours"], row["region"], row["variable"])
                if baseline_key in baseline_semantics and baseline_semantics[baseline_key] != row["semantic_sha256"]:
                    raise ValueError("same-case zero-training baseline mismatch across arm/K receipts")
                baseline_semantics[baseline_key] = row["semantic_sha256"]
            evaluations[(*key, job["lead"], job["reasoning_steps"])] = entry
    expected = (6, 30) if protocol["stage"] == "B" else (9, 135)
    if (len(training), len(evaluations)) != expected or len(records) != expected[1] * 51:
        raise ValueError("full B6train/30eval or C9train/135eval required; partial is not negative")
    return training, evaluations, records


def require_publication_intact(output):
    output = safe_output(output)
    marker = output / "publication_failure.json"
    if marker.exists() or marker.is_symlink():
        raise ValueError("authoritative publication failure; acceptance refused")
    return output


def verify_execution(output, protocol, execution):
    """Continuous first-spawn→last-reap cost is never a sum of successful workers."""
    output = require_publication_intact(output)
    jobs = protocol["jobs"]
    if (execution["status"] != "results-complete" or execution["finalized"] is not False
            or execution["scientific_claim"] is not False or not execution["limitations"]
            or execution["test_read"] is not False or execution["protocol_sha256"] != protocol["protocol_sha256"]
            or execution["stage"] != protocol["stage"] or execution["jobs_planned"] != jobs
            or execution["jobs_completed"] != jobs or len(execution["jobs_results"]) != len(jobs)
            or [row["job"] for row in execution["jobs_results"]] != jobs
            or [row["job"] for row in execution["headroom_checks"]] != jobs
            or execution["partial"] is not False or execution["budget_limited"] is not False
            or execution["owned_unreaped"] is not False or execution["continuous_clock"] is not True
            or execution["failed_job_key"] is not None or execution["failure_reason"] is not None
            or execution["billing_scope"] != protocol["gpu"]["billing"]
            or execution["started_perf_counter"] != protocol["round_started_perf_counter"]
            or execution["monotonic_boot_id"] != protocol["monotonic_boot_id"]):
        raise ValueError("full owned-reaped results-complete stage required, not failed/skipped/partial or final whole attempt")
    first = number(execution["first_gpu_spawn_started_perf_counter"], nonnegative=True)
    last = number(execution["last_owned_gpu_reap_perf_counter"], nonnegative=True)
    started = number(execution["started_perf_counter"], nonnegative=True)
    ended = number(execution["ended_perf_counter"], nonnegative=True)
    if not started <= first <= last <= ended < started + protocol["hard_cap_seconds"]:
        raise ValueError("invalid or hard-limited continuous whole/GPU clock")
    close(execution["gpu_phase_elapsed_seconds"], last - first, "continuous GPU clock")
    close(execution["gpu_hours_charged"], (last - first) / 3600, "continuous GPU hours")
    close(execution["whole_elapsed_seconds"], ended - started, "pre-aggregation whole snapshot")
    previous_reap, observed = first, 0.
    paths = []
    for index, (job, result, headroom) in enumerate(zip(jobs, execution["jobs_results"], execution["headroom_checks"])):
        receipt_path = worker_result_path(output, job)
        if local_path(result["result"]) != receipt_path:
            raise ValueError("execution result path/job differs from frozen inventory")
        receipt = read_json(receipt_path)
        close(result["elapsed_seconds"], number(receipt["elapsed_seconds"]), "execution worker elapsed")
        close(result["peak_reserved_bytes"], number(receipt["peak_reserved_bytes"]), "execution owned peak")
        snapshot = headroom["snapshot"]
        required = max(protocol["gpu"]["estimated_peak_mib"], math.ceil(observed)) + protocol["gpu"]["headroom_margin_mib"]
        if (snapshot["uuid"] != protocol["gpu"]["uuid"] or snapshot["read_only"] is not True
                or snapshot["required_free_mib"] != required or snapshot["free_mib"] < required
                or snapshot["observed_owned_peak_mib"] != observed):
            raise ValueError("before-every-spawn shared GPU read-only headroom evidence required")
        timing_path = Path(output) / "workers" / (job_key(job) + ".timing.json")
        timing = read_json(timing_path)
        spawned, reaped = number(timing["spawned_perf_counter"]), number(timing["last_owned_reap_perf_counter"])
        if (timing["job"] != job or timing["status"] != "success" or timing["returncode"] != 0
                or timing["protocol_sha256"] != protocol["protocol_sha256"] or timing["headroom"] != snapshot
                or timing["test_read"] is not False or timing["scientific_claim"] is not False
                or not timing["limitations"] or timing["cleanup"] != "already-exited"
                or not previous_reap <= spawned <= reaped <= last
                or (index == 0 and spawned != first)
                or number(timing["ended_perf_counter"]) < reaped):
            raise ValueError("complete exact owned-worker success/timing/reap evidence required")
        previous_reap = reaped
        observed = max(observed, receipt["peak_reserved_bytes"] / 2 ** 20)
        paths.extend([receipt_path, timing_path, local_path(Path(output) / "workers" / (job_key(job) + ".log"))])
    require_publication_intact(output)
    return paths


def descriptive_outcome(protocol, pairs):
    selection = candidate_selection(protocol, pairs) if protocol["stage"] == "B" else {
        "evaluated": False, "status": "not-applicable", "selected_mode": protocol["configuration"]["mode"],
        "reason": "independent C configuration; not B posthoc arm selection", "scientific_claim": False}
    adaptive = {"evaluated": False, "status": "refused", "gate_met": None,
                "reason": "no implemented predeclared fixed-frontier/sample-oracle gate schema",
                "start_training": False, "oracle_deployable": False}
    if protocol["stage"] == "B":
        adaptive.update(status="not-applicable", reason="B cannot authorize adaptive; C is independent")
    return {"status": "stage-sealed", "scientific_claim": False,
            "limitations": [*protocol["limitations"],
                            "three C seeds give descriptive consistency only, not a significance test",
                            "stage tables/snapshot precede aggregation completion; whole/GPU cost accepted only via independent driver attempt.json",
                            "undefined and negative per-case/variable metrics retained without filtering"],
            "test_read": False, "coverage_complete": True, "finalized": False, "paused": False,
            "stage": protocol["stage"], "scientific_state": selection["status"] if protocol["stage"] == "B" else "none/refused",
            "candidate_selection": selection, "adaptive_gate": adaptive,
            "pair_totals": {name: pair["totals"] for name, pair in pairs.items()},
            "whole_cost_status": "pending-driver-final-seal", "gpu_cost_status": "pending-driver-final-seal",
            "whole_round_cost_reference": protocol["whole_round_cost_reference"],
            "scientific_gate_evaluated": False, "reproducibility_level": "identity-bound numerical replay, not bitwise GPU",
            "advance_next_node": False, "independent_C_continues": protocol["stage"] == "B"}


def finalize(output, protocol, execution):
    """Seal COMPLETE stage tables; the driver later seals whole-attempt costs."""
    output = safe_output(output)
    validate_protocol(protocol)
    timing_paths = verify_execution(output, protocol, execution)
    training, evaluations, records = validate_full_set(output, protocol)
    seeds, kernels = (B_SEEDS, (4,)) if protocol["stage"] == "B" else (C_SEEDS, (1, 2, 4))
    arms = tuple(protocol["arm_configs"])
    pairs = (("rollout_l6_l12", "continue_l6"), ("rollout_l6_l12", "equal_compute_l6"),
             ("equal_compute_l6", "continue_l6")) if protocol["stage"] == "B" else (
                 ("process", "old_ours"), ("process", "matched_generic"), ("matched_generic", "old_ours"))
    for row in records:
        receipt = evaluations[(row["seed"], row["arm"], row["lead_hours"], row["K"])]
        row.update(protocol_sha256=protocol["protocol_sha256"],
                   model_code_sha256=protocol["code"]["model_code_sha256"],
                   data_identity=protocol["data"]["data_identity"],
                   provenance_sha256=receipt["artifact_sha256"]["provenance.json"],
                   worker_receipt_sha256=receipt["receipt_sha256"])
        row["row_sha256"] = digest(row)
    table = aggregate_metrics(records, seeds=seeds, arms=arms, leads=LEADS, kernels=kernels)
    records_digest = digest(records)
    for row in table:
        row.update(protocol_sha256=protocol["protocol_sha256"],
                   model_code_sha256=protocol["code"]["model_code_sha256"],
                   data_identity=protocol["data"]["data_identity"], inputs_sha256=records_digest)
        row["row_sha256"] = digest(row)
    compared = paired_comparisons(table, pairs=pairs, kernels=kernels)
    table_digest = digest(table)
    for pair in compared.values():
        for cell in pair["cells"].values():
            cell.update(protocol_sha256=protocol["protocol_sha256"], inputs_sha256=table_digest)
            cell["cell_sha256"] = digest(cell)
    expected_cells = 17 * 5 * 3 * len(kernels)
    if len(table) != len(arms) * expected_cells or any(len(pair["cells"]) != expected_cells for pair in compared.values()):
        raise ValueError("every declared variable/lead/region/K/pair required")
    outcome = descriptive_outcome(protocol, compared)
    if protocol["stage"] == "C":
        from .r7_v2_frontier import evaluate_adaptive_gate

        per_cases, latencies = [], []
        for entry in evaluations.values():
            job = entry["job"]
            for raw in _csv(entry["per_case_metrics_csv"]):
                per_cases.append({**_canonical_metric(raw, protocol["data"]), "seed": job["seed"],
                                  "arm": job["arm"], "K": job["reasoning_steps"], "sample_id": raw["sample_id"]})
            isolated = entry["evaluation_provenance"]["isolated_forward"]
            latencies.append({"seed": job["seed"], "arm": job["arm"], "K": job["reasoning_steps"],
                              "lead_hours": job["lead"], "median_seconds_per_batch": isolated["median_seconds_per_batch"],
                              "scope": isolated["scope"], "gpu_latency_measured": isolated["gpu_latency_measured"],
                              "provenance_sha256": entry["artifact_sha256"]["provenance.json"]})
        outcome["adaptive_gate"] = evaluate_adaptive_gate(protocol, records, per_cases, latencies, inventory_verified=True)
        outcome["scientific_state"] = outcome["adaptive_gate"]["status"]
    snapshot = {**execution, "scope": "immutable pre-aggregation execution snapshot; not a final whole-cost seal"}
    write_json(output / "execution_stage_snapshot.json", snapshot, output=output)
    identity = {key: protocol["code"][key] for key in ("model_code_sha256", "source_tree_sha256", "code_zip_sha256")}
    identity.update({key: protocol["code"][key] for key in
                     ("base_commit", "working_tree_modified", "code_commit_sha256", "code_status_sha256")})
    identity.update(data_identity=protocol["data"]["data_identity"], source_sha256=protocol["sources"]["source_sha256"],
                    sidecar_identity=protocol["sidecar"]["identity"], windows_sha256=digest(protocol["windows"]))
    comparison = {"scientific_claim": False, "limitations": outcome["limitations"], "test_read": False,
                  "protocol_sha256": protocol["protocol_sha256"], "identity": identity, "pairs": compared,
                  "outcome": outcome, "input_receipt_sha256": {job_key(entry["job"]): entry["receipt_sha256"]
                                                             for entry in evaluations.values()}}
    write_json(output / "paired_comparison.json", comparison, output=output)
    result = {"format": "r7-v2-stage-result-v1", **outcome, "protocol_sha256": protocol["protocol_sha256"],
              "identity": identity, "protocol": protocol, "execution_stage_snapshot_sha256": digest(snapshot),
              "training": {f"{seed}/{arm}": entry for (seed, arm), entry in training.items()},
              "evaluation": {f"{seed}/{arm}/{lead}/k{kernel}": entry for (seed, arm, lead, kernel), entry in evaluations.items()},
              "metrics": records, "aggregate": table}
    write_json(output / "stage_result.json", result, output=output)
    table_files = publish_tables(output, protocol, training, evaluations, records, table, compared, outcome)
    paths = [Path(output) / name for name in ["protocol.json", "cpu_profile.json", "code.zip", "code_commit.txt", "code_status.txt", "environment.json",
                                             "prepare_started.json", "prepare_attempt.json", "run_started.json",
                                             "stage_result.json", "paired_comparison.json", "execution_stage_snapshot.json", *table_files]]
    paths.extend(timing_paths)
    for entry in training.values():
        paths.extend([local_path(entry["checkpoint"]), local_path(entry["training_report"])])
    for entry in evaluations.values():
        paths.extend(Path(entry["evaluation_dir"]) / name for name in entry["artifact_sha256"])
    pins = pin_inventory(output, paths)
    manifest = {"scientific_claim": False, "limitations": outcome["limitations"], "test_read": False,
                "status": "stage-sealed", "protocol_sha256": protocol["protocol_sha256"], "identity": identity,
                "files_sha256": pins, "files_digest": digest(pins),
                "excluded_recursive_or_future": list(EXCLUDED_RECURSIVE_OR_FUTURE),
                "final_whole_cost_status": "pending-driver-final-seal", "final_whole_cost_reference": protocol["whole_round_cost_reference"]}
    write_json(output / "artifact_manifest.json", manifest, output=output)
    verify_inventory(output, pins)
    require_publication_intact(output)
    return outcome
