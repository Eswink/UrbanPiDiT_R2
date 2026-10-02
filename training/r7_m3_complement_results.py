"""Fail-closed M3 cross-attempt reporting; never resume or accept the failed round.

``finalize_complement`` consumes an in-memory evaluations-complete snapshot.
The driver publishes final execution/attempt receipts AFTER aggregation, including
its actual final CPU/cleanup clock. Future receipts are references, not measured
costs or fabricated hashes in this module's publication-time manifest.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import stat
import zipfile

from .r7_arm_harness import comparator_blocks, merge_seed_results, pair_cells, sha256_file, write_study_tables
from .r7_coreasoning_compare import read_case_identity, read_rmse_rows
from .r7_m3_identity import ROOT, source_identity, verify_code
from .r7_m3_protocol import (
    ARM_NAMES, LEADS, LIMITATIONS, PAIRS, RESULT_FORMAT, SEEDS, UPDATES, digest,
    expected_training_contract, job_key, planned_jobs, read_json, verify_protocol, write_json,
)
from .r7_m3_results import (
    _exclusive_csv, _read_csv, _verify_evaluation, _verify_metric_rows, _verify_training,
    descriptive_outcome,
)
from .r7_m3_worker import worker_result_path

EXECUTION_REQUIRED_KEYS = (
    "status", "evaluation_coverage_complete", "finalized", "jobs_completed",
    "whole_elapsed_seconds", "gpu_phase_elapsed_seconds", "gpu_hours_charged",
    "soft_overrun_seconds", "budget_limited", "partial", "continuous_clock",
    "billing_scope", "execution_whole_scope", "whole_round_cost_reference",
    "scientific_claim", "limitations", "test_read", "started_perf_counter",
    "hard_deadline_perf_counter", "ended_perf_counter",
    "first_gpu_spawn_started_perf_counter", "last_owned_gpu_reap_perf_counter",
)
EXECUTION_WHOLE_SCOPE = "run entry through pre-aggregation snapshot; aggregation/cleanup included only by final attempt.json"
EVALUATION_FILES = {"rmse_csv": "rmse.csv", "acc_csv": "acc.csv",
                    "skill_csv": "climatology_skill.csv", "provenance": "provenance.json"}
HELPER_FILES = (
    "training/r7_m3_protocol.py", "training/r7_m3_results.py", "training/r7_m3_worker.py",
    "training/r7_arm_harness.py", "training/r7_coreasoning_compare.py",
    "training/r7_m3_identity.py", "training/r7_rw_b_subtraction_protocol.py",
)
COMPLEMENT_LIMITATIONS = [*LIMITATIONS,
    "cross-attempt validation coverage does not make the original failed attempt successful",
    "CPU counts/gradient evidence are copied original measurements, not complement measurements",
    "new whole elapsed is a pre-aggregation snapshot; final aggregation/cleanup cost awaits attempt.json",
    "publication-time artifact pins exclude future driver receipts; final round requires a post-run seal",
]


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"finite nonnegative actual {label} required")
    return value


def _path(value):
    path = Path(value)
    if not path.is_absolute() or path != path.resolve() or path.name == "test.jsonl":
        raise ValueError("canonical absolute nonsymlink non-test artifact path required")
    return path


def _pin(path, expected):
    path = _path(path)
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"artifact SHA256 changed or missing: {path}")
    return {"path": str(path), "sha256": expected}


def _inventory(root):
    # Inventory only: receipt acceptance below uses the exact frozen jobs, never a glob.
    result = {}
    for path in sorted(root.rglob("*")):
        _path(path)
        if path.is_file():
            result[path.relative_to(root).as_posix()] = sha256_file(path)
    return result


def _original_file_names():
    """The historical 109-file set, not merely any inventory with 109 entries."""
    names = {"attempt.json", "code.zip", "code_commit.txt", "code_status.txt", "cpu_profile.json",
             "environment.json", "execution_attempt.json", "prepare_attempt.json", "protocol.json", "run_started.json"}
    for job in planned_jobs()[:13]:
        names.update("workers/" + job_key(job) + suffix for suffix in (".json", ".timing.json", ".log"))
        if job["phase"] == "train":
            folder = f"seed{job['seed']}/training/{job['arm']}"
            names.add(folder + "/training_report.json")
            names.update(folder + f"/update_{update:07d}.pt" for update in (100, 200, 300, 400))
        else:
            folder = f"seed{job['seed']}/evaluation/{job['arm']}/lead_{job['lead']:03d}h"
            names.update(folder + "/" + name for name in EVALUATION_FILES.values())
    names.update("workers/" + job_key(planned_jobs()[13]) + suffix for suffix in (".timing.json", ".log"))
    return names


def _archive(protocol, required):
    code = protocol["code"]
    files = code["files"]
    if not set(required) <= set(files) or digest(files) != code["source_tree_sha256"]:
        raise ValueError("archived helper/source identity incomplete or changed")
    _pin(Path(protocol["output"]) / "code.zip", code["code_zip_sha256"])
    with zipfile.ZipFile(Path(protocol["output"]) / "code.zip") as archive:
        if len(archive.namelist()) != len(files) or set(archive.namelist()) != set(files):
            raise ValueError("archive members must exactly match pinned source files")
        for member in archive.infolist():
            name = Path(member.filename)
            if (name.is_absolute() or ".." in name.parts or member.is_dir()
                    or stat.S_ISLNK(member.external_attr >> 16)
                    or hashlib.sha256(archive.read(member)).hexdigest() != files[member.filename]):
                raise ValueError("archive member bytes/path changed")
    verify_code(protocol)  # Actual original/new protocol, never a relocated scientific protocol.
    if "source_root" in code and _path(code["source_root"]) != ROOT:
        raise ValueError("new aggregation source root must be the actual repository")
    for filename in ("code_commit.txt", "code_status.txt"):
        key = filename.removesuffix(".txt") + "_sha256"
        if key in code:
            _pin(Path(protocol["output"]) / filename, code[key])


def _protocols(output, new_protocol):
    output, original_root = _path(output), _path(new_protocol["original_output"])
    body = {key: value for key, value in new_protocol.items() if key != "protocol_sha256"}
    if (new_protocol["protocol_sha256"] != digest(body)
            or new_protocol.get("format") != "r7-m3-validation-complement-protocol-v1"
            or new_protocol.get("scientific_claim") is not False or not new_protocol.get("limitations")
            or new_protocol.get("test_read") is not False or new_protocol["output"] != str(output)
            or output == original_root or original_root in output.parents or output in original_root.parents
            or new_protocol["planned_seconds"] != 1800 or new_protocol["hard_cap_seconds"] != 3600
            or new_protocol["cleanup_reserve_seconds"] != 10):
        raise ValueError("frozen complement identity, separate output and soft/hard budgets required")
    if read_json(output / "protocol.json") != new_protocol:
        raise ValueError("actual complement protocol differs from frozen argument")
    pins = new_protocol["original_file_sha256"]
    if len(pins) != 109 or set(pins) != _original_file_names() or _inventory(original_root) != pins:
        raise ValueError("all 109 immutable original files must match, including failed history")
    original = verify_protocol(original_root / "protocol.json")
    if (original["output"] != str(original_root)
            or original["protocol_sha256"] != new_protocol["original_protocol_sha256"]
            or original["code"]["code_zip_sha256"] != new_protocol["original_code_zip_sha256"]
            or new_protocol["jobs"] != planned_jobs()[13:]
            or new_protocol["gpu"]["policy"] != "shared"
            or any(new_protocol["gpu"][key] != original["gpu"][key]
                   for key in ("uuid", "estimated_peak_mib", "headroom_margin_mib"))
            or new_protocol["code"]["model_code_sha256"] != original["code"]["model_code_sha256"]):
        raise ValueError("inherited science/GPU/model pins or exact 23 missing evaluation jobs changed")
    _archive(original, HELPER_FILES)
    _archive(new_protocol, ("training/r7_m3_complement_results.py",))
    if read_json(original_root / "cpu_profile.json") != original["cpu_profile"]:
        raise ValueError("original CPU evidence differs from frozen protocol")
    return output, original_root, original


def _execution(execution, output, protocol):
    if not set(EXECUTION_REQUIRED_KEYS) <= set(execution):
        raise ValueError("incomplete evaluations-complete execution snapshot")
    whole = _number(execution["whole_elapsed_seconds"], "pre-aggregation whole seconds")
    gpu = _number(execution["gpu_phase_elapsed_seconds"], "continuous GPU seconds")
    hours = _number(execution["gpu_hours_charged"], "GPU hours")
    overrun = _number(execution["soft_overrun_seconds"], "soft overrun seconds")
    started, deadline, ended, first, last = [_number(execution[key], key) for key in (
        "started_perf_counter", "hard_deadline_perf_counter", "ended_perf_counter",
        "first_gpu_spawn_started_perf_counter", "last_owned_gpu_reap_perf_counter")]
    if (not started <= first <= last <= ended <= deadline
            or deadline != started + protocol["hard_cap_seconds"]
            or gpu != last - first or whole != ended - started):
        raise ValueError("actual continuous GPU/whole clocks must match their complete timestamp intervals")
    if (execution["status"] != "evaluations-complete" or execution["evaluation_coverage_complete"] is not True
            or execution["finalized"] is not False or execution["jobs_completed"] != protocol["jobs"]
            or execution["budget_limited"] is not False or execution["partial"] is not False
            or execution["continuous_clock"] is not True or not execution["billing_scope"]
            or execution["scientific_claim"] is not False or not execution["limitations"]
            or execution["test_read"] is not False or not 0 <= gpu <= whole <= protocol["hard_cap_seconds"]
            or not math.isclose(hours, gpu / 3600, rel_tol=1e-12, abs_tol=1e-12)
            or overrun != max(0, whole - protocol["planned_seconds"])
            or execution["execution_whole_scope"] != EXECUTION_WHOLE_SCOPE
            or execution["whole_round_cost_reference"] != str(output / "attempt.json")):
        raise ValueError("partial/failed/nonfinite/over-hard-bound execution cannot aggregate")


def _source_inputs(original):
    # Hash only opaque source bytes and train/val metadata; no arrays or test manifest.
    sources = source_identity(original["manifests"])
    if sources != original["sources"]:
        raise ValueError("actual source/train/val/BUILD_COMPLETE pins changed")
    sidecar = original["sidecar"]
    metadata = _pin(sidecar["path"], sidecar["metadata_sha256"])
    marker = _pin(Path(sidecar["path"]).parent / "BUILD_COMPLETE.json", sidecar["publication_marker_sha256"])
    if (read_json(metadata["path"])["sidecar_identity"] != sidecar["identity"]
            or read_json(marker["path"]) != {"schema_version": 1, "build_complete": True}):
        raise ValueError("actual original sidecar identity/publication changed")
    return {"sources": sources, "data_identity": original["data"]["data_identity"],
            "sidecar": sidecar, "sidecar_files": [metadata, marker]}


def _original_attempt(root, original):
    attempt, execution = read_json(root / "attempt.json"), read_json(root / "execution_attempt.json")
    for entry in (attempt, execution):
        if (entry["status"] != "failed" or entry["finalized"] is not False
                or entry["scientific_claim"] is not False or not entry["limitations"]
                or entry["test_read"] is not False or entry["jobs_completed"] != planned_jobs()[:13]):
            raise ValueError("original attempt must remain failed with its exact 13 accepted jobs")
        seconds = _number(entry["gpu_phase_elapsed_seconds"], "original full GPU seconds")
        if not math.isclose(_number(entry["gpu_hours_charged"], "original GPU hours"), seconds / 3600,
                            rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("original charged hours do not match actual continuous seconds")
    if (attempt["gpu_phase_elapsed_seconds"] != execution["gpu_phase_elapsed_seconds"]
            or _number(attempt["whole_elapsed_seconds"], "original full whole seconds") < seconds
            or execution["jobs_planned"] != original["jobs"] or not execution["billing_scope"]):
        raise ValueError("original full attempt/continuous execution costs disagree")
    return {"attempt": attempt, "execution": execution,
            "attempt_receipt": _pin(root / "attempt.json", sha256_file(root / "attempt.json")),
            "execution_receipt": _pin(root / "execution_attempt.json", sha256_file(root / "execution_attempt.json")),
            "failed_worker_timing": read_json(root / "workers" / (job_key(planned_jobs()[13]) + ".timing.json"))}


def _receipt(entry, job, original):
    if (entry.get("status") != "success" or entry.get("scientific_claim") is not False
            or not entry.get("limitations") or entry.get("test_read") is not False or entry.get("job") != job
            or entry.get("protocol_sha256") != original["protocol_sha256"]
            or entry.get("model_code_sha256") != original["code"]["model_code_sha256"]
            or entry.get("data_identity") != original["data"]["data_identity"]
            or entry.get("process_supervision") != expected_training_contract(original, job["arm"])
            or ("seed" in entry and entry["seed"] != job["seed"])):
        raise ValueError("missing/failed/wrong-seed receipt or original two-level training identity changed")
    for key in ("elapsed_seconds", "peak_allocated_bytes", "peak_reserved_bytes"):
        _number(entry[key], key)
    if entry["peak_reserved_bytes"] < entry["peak_allocated_bytes"]:
        raise ValueError("allocator reserved peak cannot be below allocated peak")


def _evaluation(entry, job, original, training, root):
    directory = root / f"seed{job['seed']}" / "evaluation" / job["arm"] / f"lead_{job['lead']:03d}h"
    if (_path(entry["evaluation_dir"]) != directory or entry["arm"] != job["arm"]
            or entry["lead_hours"] != job["lead"] or entry["split"] != "val"
            or entry["channels"] != original["data"]["channels"] or entry["units"] != original["data"]["units"]
            or entry["checkpoint"] != training["checkpoint"] or entry["checkpoint_sha256"] != training["checkpoint_sha256"]
            or set(entry["artifact_sha256"]) != set(EVALUATION_FILES.values())):
        raise ValueError("evaluation must use its own exact output path and original selected checkpoint")
    for key in ("baseline", "pre_init_baseline", "initialized_baseline"):
        baseline = entry[key]
        if set(baseline) != {"allocated_bytes", "reserved_bytes"} or any(
                _number(value, "allocator baseline") != 0 for value in baseline.values()):
            raise ValueError("three actual fresh zero allocator baselines required")
    for key, filename in EVALUATION_FILES.items():
        if _path(entry[key]) != directory / filename:
            raise ValueError("CSV/provenance pointer escapes the actual evaluation path")
        _pin(directory / filename, entry["artifact_sha256"][filename])
    declared = original["data"]["evaluation_cases"][str(job["lead"])]
    cases = read_case_identity(directory)
    if (cases is None or cases["cases"] != declared["cases"] or cases["split"] != "val"
            or cases["channels"] != original["data"]["channels"] or cases["units"] != original["data"]["units"]
            or cases["lead_hours"] != [job["lead"]]
            or cases["evaluation_manifest_sha256"] != original["sources"]["val_manifest_sha256"]
            or cases["n_evaluated"] != len(declared["cases"]) or entry["n_evaluated"] != len(declared["cases"])
            or entry["n_available_windows"] != declared["n_available"]):
        raise ValueError("exact frozen cases/channels/units/lead/manifest required")
    provenance = read_json(directory / "provenance.json")
    if (provenance["checkpoint_sha256"] != training["checkpoint_sha256"]
            or provenance["training_identity"] != original["data"]["data_identity"]
            or provenance["process_scale_sidecar_identity"] != original["sidecar"]["identity"]
            or provenance["training_protocol_sha256"] != original["protocol_sha256"]
            or provenance["scientific_claim"] is not False
            or provenance["source_declaration"] != original["sources"]["source_path"]
            or provenance["inference_options"] != {"reasoning_steps": 4}
            or provenance["step_hours"] != 6 or provenance["n_available_windows"] != declared["n_available"]):
        raise ValueError("actual evaluation original model/checkpoint/sidecar/training-protocol provenance changed")
    return _verify_metric_rows(entry, original)


def _job_source(root, entry, job, origin, original, new_protocol):
    path = worker_result_path(root, job)
    timing_path = root / "workers" / (job_key(job) + ".timing.json")
    timing = read_json(timing_path)
    if timing["job"] != job or timing["status"] != "success" or timing["returncode"] != 0:
        raise ValueError("successful exact-job process timing required")
    keys = ("started_perf_counter", "spawned_perf_counter", "reaped_perf_counter",
            "last_owned_reap_perf_counter", "ended_perf_counter")
    stamps = [_number(timing[key], key) for key in keys]
    if stamps != sorted(stamps) or stamps[-1] - stamps[0] < entry["elapsed_seconds"]:
        raise ValueError("actual process timestamps cannot omit the measured worker interval")
    artifacts = ({name: {"path": str(Path(entry["evaluation_dir"]) / name), "sha256": pin}
                  for name, pin in entry["artifact_sha256"].items()} if job["phase"] == "evaluate" else
                 {"training_report.json": {"path": entry["training_report"], "sha256": entry["training_report_sha256"]}})
    log_path = root / "workers" / (job_key(job) + ".log")
    return {"job": job, "origin": origin, "kind": "training" if job["phase"] == "train" else "evaluation",
            "receipt": _pin(path, sha256_file(path)), "timing": _pin(timing_path, sha256_file(timing_path)),
            "log": _pin(log_path, sha256_file(log_path)), "timestamps": {key: timing[key] for key in keys},
            "checkpoint": {"path": entry["checkpoint"], "sha256": entry["checkpoint_sha256"]},
            "artifacts": artifacts, "training_protocol_sha256": original["protocol_sha256"],
            "aggregation_protocol_sha256": new_protocol["protocol_sha256"],
            "evaluator_code": entry.get("archived_evaluator_code", original["code"]),
            "cost": {"worker_elapsed_seconds": entry["elapsed_seconds"], "process_elapsed_seconds": stamps[-1] - stamps[0],
                     "billing": "descriptive worker/process subtotal, not the full continuous attempt charge"}}


def validate_complement_set(output, new_protocol, original):
    """Accept exactly six original trains, seven original evals and 23 supplements."""
    training, evaluations, sources = {}, {}, []
    original_root = _path(original["output"])
    cells = 0
    for index, job in enumerate(planned_jobs()):
        root, origin = (original_root, "original") if index < 13 else (output, "complement")
        entry = read_json(worker_result_path(root, job))
        _receipt(entry, job, original)
        identity = (job["seed"], job["arm"])
        if job["phase"] == "train":
            _verify_training(entry, job, original)
            if entry["selected_update"] != 400:
                raise ValueError("actual original selected400 checkpoint required")
            _number(entry["seconds_per_update"], "training seconds per update")
            training[identity] = entry
        else:
            if origin == "original":
                _verify_evaluation(entry, job, original, training[identity])
            else:
                training_path = worker_result_path(original_root, {"phase": "train", "seed": job["seed"], "arm": job["arm"], "lead": None})
                expected_code = {key: original["code"][key] for key in ("code_zip_sha256", "source_tree_sha256", "model_code_sha256")}
                if (entry.get("complement_protocol_sha256") != new_protocol["protocol_sha256"]
                        or entry.get("original_training_receipt_sha256") != sha256_file(training_path)
                        or entry.get("archived_evaluator_code") != expected_code):
                    raise ValueError("supplement protocol/archive/original training receipt pin changed")
            cells += _evaluation(entry, job, original, training[identity], root)
            evaluations[(*identity, job["lead"])] = entry
        sources.append(_job_source(root, entry, job, origin, original, new_protocol))
    if len(training) != 6 or len(evaluations) != 30 or cells != 510 or sum(e["n_evaluated"] for e in evaluations.values()) != 528:
        raise ValueError("exact 6 train/30 evaluation/510 RMSE/528 initialization coverage required")
    return training, evaluations, sources


def _seed_results(output, original, protocol, training, evaluations):
    for seed in SEEDS:
        per_arm = {arm: training[(seed, arm)] for arm in ARM_NAMES}
        evaluation = {f"{arm}@{lead}h": evaluations[(seed, arm, lead)] for arm in ARM_NAMES for lead in LEADS}
        result = {"format": RESULT_FORMAT, "scientific_claim": False, "limitations": COMPLEMENT_LIMITATIONS,
                  "test_read": False, "seed": seed, "seeds": list(SEEDS), "device": "cuda:0",
                  "gpu": original["gpu"]["uuid"], "torch_version": per_arm[ARM_NAMES[0]]["torch_version"],
                  "model_code_sha256": original["code"]["model_code_sha256"], "protocol": original,
                  "protocol_sha256": original["protocol_sha256"], "aggregation_protocol": protocol,
                  "aggregation_protocol_sha256": protocol["protocol_sha256"], "original_finalizer_accepted": False,
                  "arm_pairing": original["cpu_profile"]["pairing"][str(seed)],
                  "flop_measurements": original["cpu_profile"]["measurements"],
                  "training": per_arm, "evaluation": evaluation, "probes": {},
                  "case_counts_by_lead": {str(lead): len(original["data"]["evaluation_cases"][str(lead)]["cases"]) for lead in LEADS},
                  "budget": {"training_seconds_total": sum(entry["elapsed_seconds"] for entry in per_arm.values())}}
        write_json(output / f"seed{seed}" / "seed_result.json", result)


def _costs(history, execution, output, sources):
    original, old_execution = history["attempt"], history["execution"]
    gpu = execution["gpu_phase_elapsed_seconds"]
    return {"original": {"status": "failed", "full_attempt": original, "execution": old_execution,
                         "gpu_phase_elapsed_seconds": original["gpu_phase_elapsed_seconds"],
                         "gpu_hours_charged": original["gpu_hours_charged"],
                         "whole_elapsed_seconds": original["whole_elapsed_seconds"],
                         "includes_failed_worker_and_gaps": True, "billing_scope": old_execution["billing_scope"]},
            "complement": {"status": "evaluations-complete", "execution_snapshot": execution,
                           "gpu_phase_elapsed_seconds": gpu, "gpu_hours_charged": execution["gpu_hours_charged"],
                           "whole_elapsed_seconds_snapshot": execution["whole_elapsed_seconds"],
                           "soft_overrun_seconds_snapshot": execution["soft_overrun_seconds"],
                           "continuous_clock": True, "billing_scope": execution["billing_scope"],
                           "finalization_pending": True,
                           "whole_round_cost_reference": {"path": str(output / "attempt.json"), "sha256": None,
                                                          "available_after_finalization": True}},
            "total": {"gpu_phase_elapsed_seconds": original["gpu_phase_elapsed_seconds"] + gpu,
                      "gpu_hours_charged": original["gpu_hours_charged"] + execution["gpu_hours_charged"],
                      "whole_elapsed_seconds": None, "whole_cost_status": "pending final complement attempt receipt"},
            "accepted_worker_subtotals_not_charges": {
                origin: sum(item["cost"]["worker_elapsed_seconds"] for item in sources if item["origin"] == origin)
                for origin in ("original", "complement")}}


def _tables(output, merged, sources, cpu_pin, costs):
    for filename, keys in (("parameter_table.csv", ("parameters", "trainable_parameters")),
                           ("flops_table.csv", ("forward_flops", "forward_backward_flops"))):
        rows = [{"arm": arm["name"], **{key: arm[key] for key in keys}, "origin": "original-cpu-profile",
                 "measurement": "copied evidence, not remeasured", "source_profile_sha256": cpu_pin["sha256"]}
                for arm in merged["protocol"]["arms"]]
        _exclusive_csv(output / filename, list(rows[0]), rows)
    allocator, throughput, acc, times, initializations = [], [], [], [], []
    for item in sources:
        job = item["job"]
        seed, arm = job["seed"], job["arm"]
        entry = (merged["training"][str(seed)][arm] if job["phase"] == "train" else
                 merged["evaluation"][f"{seed}/{arm}@{job['lead']}h"])
        common = {"seed": seed, "arm": arm, "phase": job["phase"], "lead_hours": job["lead"], "origin": item["origin"]}
        allocator.append({**common, "baseline_allocated_bytes": entry.get("baseline", {}).get("allocated_bytes", ""),
                          "baseline_reserved_bytes": entry.get("baseline", {}).get("reserved_bytes", ""),
                          "peak_allocated_bytes": entry["peak_allocated_bytes"], "peak_reserved_bytes": entry["peak_reserved_bytes"]})
        times.append({**common, **item["cost"], **item["timestamps"]})
        if job["phase"] == "train":
            throughput.append({**common, "updates": entry["updates_run"], "seconds_per_update": entry["seconds_per_update"],
                               "elapsed_seconds": entry["elapsed_seconds"]})
        else:
            acc.extend({"seed": seed, "arm": arm, "origin": item["origin"], **row} for row in _read_csv(entry["acc_csv"]))
            cases = read_case_identity(entry["evaluation_dir"])["cases"]
            initializations.extend({**common, "init_time": init, "valid_times": json.dumps(valid), "split": "val"}
                                   for init, valid in cases)
    for filename, rows in (("allocator_table.csv", allocator), ("training_throughput_table.csv", throughput),
                           ("acc_table.csv", acc), ("phase_time_table.csv", times),
                           ("evaluation_time_table.csv", [row for row in times if row["phase"] == "evaluate"]),
                           ("initialization_table.csv", initializations)):
        _exclusive_csv(output / filename, list(rows[0]), rows)
    write_json(output / "cost_views.json", {"scientific_claim": False, "limitations": COMPLEMENT_LIMITATIONS,
               "test_read": False, "parameters_and_flops": {"source": cpu_pin, "remeasured": False},
               "timing": costs, "allocator": {"rows": len(allocator), "training_baselines": "not recorded; left blank",
                                              "fresh_evaluation_baselines_verified": True}})


def finalize_complement(output, new_protocol, execution):
    """Return the ORIGINAL descriptive outcome; failed guards raise, unresolved pauses.

    No original output/protocol is modified. The unchanged merge helper reads new
    seed files carrying the actual inherited scientific protocol; aggregation
    provenance is an independent field, not a fake relocated original protocol.
    """
    _execution(execution, _path(output), new_protocol)
    output, original_root, original = _protocols(output, new_protocol)
    inputs = _source_inputs(original)
    history = _original_attempt(original_root, original)
    training, evaluations, sources = validate_complement_set(output, new_protocol, original)
    _seed_results(output, original, new_protocol, training, evaluations)
    merged = merge_seed_results(output, seeds=SEEDS, arms=ARM_NAMES, fmt=RESULT_FORMAT)
    costs = _costs(history, execution, output, sources)
    cpu_pin = _pin(original_root / "cpu_profile.json", new_protocol["original_file_sha256"]["cpu_profile.json"])
    copied = output / "cpu_profile.json"
    if copied.exists():
        _pin(copied, cpu_pin["sha256"])
    else:
        with copied.open("xb") as handle:
            handle.write((original_root / "cpu_profile.json").read_bytes())
    merged.update(limitations=COMPLEMENT_LIMITATIONS, aggregation_protocol=new_protocol,
                  aggregation_protocol_sha256=new_protocol["protocol_sha256"], original_finalizer_accepted=False,
                  cpu_gradient_ownership=original["cpu_profile"]["gradient_ownership"],
                  cpu_profile_provenance={**cpu_pin, "remeasured": False}, cost_views=costs)
    merged["budget"].update(original_gpu_phase_elapsed_seconds=costs["original"]["gpu_phase_elapsed_seconds"],
                           complement_gpu_phase_elapsed_seconds=costs["complement"]["gpu_phase_elapsed_seconds"],
                           gpu_phase_elapsed_seconds=costs["total"]["gpu_phase_elapsed_seconds"],
                           gpu_hours_charged=costs["total"]["gpu_hours_charged"], continuous_clock_per_attempt=True,
                           training_and_evaluation_seconds_scope="accepted worker subtotals, not billed continuous costs")
    identity = {"dataset_identity": original["data"]["data_identity"],
                "model_code_sha256": original["code"]["model_code_sha256"], "sidecar_identity": original["sidecar"]["identity"],
                "declared_update_budget": UPDATES, "evaluation_split": "val"}
    table, blocks = comparator_blocks(merged, pairs=PAIRS, depth=0, identity=identity)
    pairs = pair_cells(blocks, leads=LEADS)
    cells = {f"{lead}h|{channel}" for lead in LEADS for channel in original["data"]["channels"]}
    if (len(table) != 255 or set(pairs) != {f"{first} - {second}" for first, second in PAIRS}
            or any(row["seeds"] != list(SEEDS) for row in table)
            or any(set(pair["cells"]) != cells or len(pair["cells"]) != 85
                   or any(set(cell["seed_deltas"]) != {str(seed) for seed in SEEDS} for cell in pair["cells"].values())
                   for pair in pairs.values())):
        raise ValueError("complete 255 aggregate rows/85 paired cells/exact seeds41,42 required")
    outcome = descriptive_outcome(pairs)
    write_json(output / "merged_result.json", merged)
    write_json(output / "paired_comparison.json", {"format": "r7-m3-validation-complement-comparison-v1",
               "scientific_claim": False, "limitations": COMPLEMENT_LIMITATIONS, "test_read": False,
               "protocol_sha256": original["protocol_sha256"], "aggregation_protocol_sha256": new_protocol["protocol_sha256"],
               "identity": identity, "pairs": pairs, "table": table, "outcome": outcome})
    write_study_tables(merged, output)
    _tables(output, merged, sources, cpu_pin, costs)
    if _inventory(original_root) != new_protocol["original_file_sha256"]:
        raise ValueError("original bytes changed during aggregation")
    write_json(output / "source_manifest.json", {"format": "r7-m3-validation-complement-sources-v1",
               "scientific_claim": False, "limitations": COMPLEMENT_LIMITATIONS, "test_read": False,
               "original_output": str(original_root), "original_protocol": original,
               "original_file_sha256": new_protocol["original_file_sha256"], "original_history": history,
               "aggregation_protocol": new_protocol, "inputs": inputs, "jobs": sources,
               "coverage": {"training": 6, "original_evaluations": 7, "complement_evaluations": 23,
                            "evaluations": 30, "rmse_cells": 510, "initializations": 528}, "costs": costs,
               "cpu_profile": {**cpu_pin, "copied_sha256": sha256_file(copied), "remeasured": False}})
    pins = _inventory(output)
    if "artifact_manifest.json" in pins:
        raise FileExistsError("complement artifacts already finalized")
    write_json(output / "artifact_manifest.json", {"scientific_claim": False, "limitations": COMPLEMENT_LIMITATIONS,
               "test_read": False, "protocol_sha256": original["protocol_sha256"],
               "aggregation_protocol_sha256": new_protocol["protocol_sha256"], "files_sha256": pins, "outcome": outcome,
               "scope": "all actual files at aggregation publication, excluding this self-referential manifest",
               "pending_driver_artifacts": [str(output / name) for name in ("execution_attempt.json", "attempt.json")],
               "final_round_sealed": False})
    return outcome
