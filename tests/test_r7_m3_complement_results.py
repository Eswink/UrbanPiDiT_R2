"""CPU-only schema fixtures: synthetic engineering artifacts, never weather truth."""
from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
import zipfile

import pytest

from training import r7_m3_complement_results as results
from training import r7_m3_identity as identity
from training import r7_m3_protocol as protocol
from training.r7_arm_harness import sha256_file
from training.r7_m3_worker import worker_result_path


MODEL_HASH = "c" * 64
CASE_COUNTS = dict(zip(protocol.LEADS, (22, 21, 19, 15, 11)))


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=True), encoding="utf-8")


def _csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _freeze(value):
    value["protocol_sha256"] = protocol.digest({key: item for key, item in value.items() if key != "protocol_sha256"})
    _json(Path(value["output"]) / "protocol.json", value)


def _profile():
    measure = {"parameters": 100, "trainable_parameters": 100, "forward_flops": 200, "forward_backward_flops": 400}
    groups = {key: {"norm": 0.0} for key in ("shared", "query", "readout", "forecast_history", "forecast_drafts")}
    ownership = {"scientific_claim": False, "limitations": ["engineering fixture"], "sample_id": ["fixture"],
                 "components": {key: {"loss": 0.0, "groups": deepcopy(groups)}
                                for key in ("forecast", "input", "future", "draft", "total")}}
    return {"scientific_claim": False, "limitations": ["not a measurement"], "device": "cpu", "reasoning_steps": 4,
            "probe_batch_size": 2, "gradient_probe_seed": 41, "flop_convention": "elementwise arithmetic uncounted",
            "measurements": {arm: dict(measure) for arm in protocol.ARM_NAMES},
            "gradient_ownership": {arm: deepcopy(ownership) for arm in protocol.ARM_NAMES},
            "pairing": {str(seed): {"all_shared_pairs_identical": True, "same_tensor_set": True,
                                   "full_initial_state_sha256": {arm: "d" * 64 for arm in protocol.ARM_NAMES}}
                        for seed in protocol.SEEDS}}


def _archive(root, source_root, names):
    pins = {name: sha256_file(source_root / name) for name in names}
    with zipfile.ZipFile(root / "code.zip", "w") as archive:
        for name in names:
            archive.writestr(name, (source_root / name).read_bytes())
    (root / "code_commit.txt").write_text("fixture commit\n", encoding="utf-8")
    (root / "code_status.txt").write_text("fixture status\n", encoding="utf-8")
    return {"files": pins, "source_tree_sha256": protocol.digest(pins), "code_zip_sha256": sha256_file(root / "code.zip"),
            "model_code_sha256": MODEL_HASH, "source_root": str(source_root), "base_commit": "fixture",
            "code_commit_sha256": sha256_file(root / "code_commit.txt"),
            "code_status_sha256": sha256_file(root / "code_status.txt"), "working_tree_modified": True}


def _inputs(tmp_path):
    manifests = tmp_path / "inputs/store/manifests"
    manifests.mkdir(parents=True)
    source = manifests.parent.parent / "source.bin"
    source.write_bytes(b"tiny opaque source identity fixture, not weather data")
    _json(manifests / "source_preflight.json", {"schema_version": 1, "source_path": str(source),
          "fingerprint": {"scope": "full-local-file", "sha256": sha256_file(source), "bytes": source.stat().st_size}})
    _json(manifests.parent.parent / "source_receipt.json", {"synthetic_fallback": False,
          "local_artifact": {"sha256": sha256_file(source)}})
    _json(manifests / "BUILD_COMPLETE.json", {"schema_version": 1, "build_complete": True})
    for split in ("train", "val"):
        (manifests / f"{split}.jsonl").write_text("opaque engineering identity fixture", encoding="utf-8")
    sidecar = tmp_path / "sidecar/scale_metadata.json"
    _json(sidecar, {"sidecar_identity": "f" * 64, "schema": "r7-process-scale-v1", "fit_split": "train"})
    _json(sidecar.parent / "BUILD_COMPLETE.json", {"schema_version": 1, "build_complete": True})
    pins = {"path": str(sidecar), "identity": "f" * 64, "metadata_sha256": sha256_file(sidecar),
            "publication_marker_sha256": sha256_file(sidecar.parent / "BUILD_COMPLETE.json")}
    return manifests, pins


def _receipt(original, job):
    return {"status": "success", "job": deepcopy(job), "scientific_claim": False, "limitations": ["tmp engineering fixture"],
            "test_read": False, "protocol_sha256": original["protocol_sha256"], "model_code_sha256": MODEL_HASH,
            "data_identity": original["data"]["data_identity"],
            "process_supervision": protocol.expected_training_contract(original, job["arm"]),
            "elapsed_seconds": 1.0, "peak_allocated_bytes": 1000, "peak_reserved_bytes": 2000}


def _training(original, job):
    entry = _receipt(original, job)
    folder = Path(original["output"]) / f"seed{job['seed']}" / "training" / job["arm"]
    folder.mkdir(parents=True)
    for update in (100, 200, 300, 400):
        (folder / f"update_{update:07d}.pt").write_bytes(b"opaque fake checkpoint; never loaded")
    checkpoint = folder / "update_0000400.pt"
    report = {"updates_this_run": 400, "selected_update": 400, "early_stopped": False, "selection_split": "val",
              "contract": {"process_supervision": entry["process_supervision"], "data_identity": entry["data_identity"]},
              "validations": [{"update": update} for update in (100, 200, 300, 400)],
              "losses": [{"update": update} for update in range(1, 401)]}
    _json(folder / "training_report.json", report)
    entry.update(checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
                 training_report=str(folder / "training_report.json"), training_report_sha256=sha256_file(folder / "training_report.json"),
                 updates_run=400, selected_update=400, early_stopped=False, seconds_per_update=.0025,
                 initial_state_sha256="d" * 64, torch_version="fixture-cpu", shared_initial_state={
                     "provided": True, "applied_count": 1, "ignored_count": 0})
    return entry


def _evaluation(original, job, root, training, error):
    entry = _receipt(original, job)
    lead = job["lead"]
    folder = root / f"seed{job['seed']}" / "evaluation" / job["arm"] / f"lead_{lead:03d}h"
    folder.mkdir(parents=True)
    channels, units = original["data"]["channels"], original["data"]["units"]
    count = CASE_COUNTS[lead]
    common = [{"lead_hours": lead, "variable": name, "n_initializations": count} for name in channels]
    _csv(folder / "rmse.csv", [{**row, "unit": unit, "rmse": error} for row, unit in zip(common, units)])
    _csv(folder / "acc.csv", [{**row, "pooled_acc": .5, "status": "defined"} for row in common])
    _csv(folder / "climatology_skill.csv", [{**row, "unit": unit, "rmse_forecast": error,
          "rmse_climatology": 4., "mse_skill": 1 - error ** 2 / 16} for row, unit in zip(common, units)])
    declared = original["data"]["evaluation_cases"][str(lead)]
    provenance = {"scientific_claim": False, "initializations": [{"init_time": init, "valid_times": valid,
                  "mse": [[error ** 2] * 17]} for init, valid in declared["cases"]],
                  "channels": channels, "units": units, "lead_hours": [lead], "split": "val", "n_evaluated": count,
                  "n_available_windows": count, "step_hours": 6, "inference_options": {"reasoning_steps": 4},
                  "evaluation_manifest_sha256": original["sources"]["val_manifest_sha256"],
                  "source_declaration": original["sources"]["source_path"], "training_identity": original["data"]["data_identity"],
                  "checkpoint_sha256": training["checkpoint_sha256"], "process_scale_sidecar_identity": original["sidecar"]["identity"],
                  "training_protocol_sha256": original["protocol_sha256"]}
    _json(folder / "provenance.json", provenance)
    entry.update(arm=job["arm"], lead_hours=lead, split="val", channels=channels, units=units, n_evaluated=count,
                 n_available_windows=count, evaluation_dir=str(folder),
                 **{key: str(folder / filename) for key, filename in results.EVALUATION_FILES.items()},
                 artifact_sha256={name: sha256_file(folder / name) for name in results.EVALUATION_FILES.values()},
                 baseline={"allocated_bytes": 0, "reserved_bytes": 0}, pre_init_baseline={"allocated_bytes": 0, "reserved_bytes": 0},
                 initialized_baseline={"allocated_bytes": 0, "reserved_bytes": 0},
                 checkpoint=training["checkpoint"], checkpoint_sha256=training["checkpoint_sha256"])
    return entry


def _worker(root, entry, index):
    job = entry["job"]
    path = worker_result_path(root, job)
    _json(path, entry)
    _json(path.with_suffix(".timing.json"), {"job": job, "status": "success", "returncode": 0,
          "started_perf_counter": index * 5., "spawned_perf_counter": index * 5. + .25,
          "reaped_perf_counter": index * 5. + 2., "last_owned_reap_perf_counter": index * 5. + 2.1,
          "ended_perf_counter": index * 5. + 2.2})
    path.with_suffix(".log").write_text("fixture process log", encoding="utf-8")


@pytest.fixture
def complete(tmp_path, monkeypatch):
    old, out, source_root = (tmp_path / name for name in ("original", "complement", "source"))
    old.mkdir()
    out.mkdir()
    repo = identity.ROOT
    names = [*results.HELPER_FILES, "training/r7_m3_complement_results.py"]
    for name in names:
        path = source_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((repo / name).read_bytes())
    monkeypatch.setattr(identity, "ROOT", source_root)
    monkeypatch.setattr(results, "ROOT", source_root)
    import training.r7_experiment as experiment
    monkeypatch.setattr(experiment, "model_code_digest", lambda: MODEL_HASH)
    manifests, sidecar = _inputs(tmp_path)
    data = {"channels": [f"channel{i}" for i in range(17)], "units": ["K", "Pa", *["m s**-1"] * 15],
            "data_identity": "a" * 64, "val_data_identity": "b" * 64,
            "normalization_mean": [0.] * 17, "normalization_std": [2.] * 17,
            "evaluation_cases": {str(lead): {"n_available": count,
                 "cases": [[f"fixture-case-{i:03d}", [f"fixture-valid-{i:03d}+{lead}h"]] for i in range(count)]}
                 for lead, count in CASE_COUNTS.items()}}
    original = protocol.build_protocol(manifests=manifests, output=old, dataset=data,
                sources=identity.source_identity(manifests), sidecar=sidecar,
                code=_archive(old, source_root, results.HELPER_FILES), profile=_profile(),
                authorization={}, authorization_sha256="1" * 64, gpu_uuid="GPU-fixture", estimated_peak_mib=342)
    _freeze(original)
    _json(old / "cpu_profile.json", original["cpu_profile"])
    training = {}
    for index, job in enumerate(protocol.planned_jobs()[:13]):
        if job["phase"] == "train":
            entry = _training(original, job)
            training[(job["seed"], job["arm"])] = entry
        else:
            entry = _evaluation(original, job, old, training[(job["seed"], job["arm"])],
                                {"aux_off": 2., "input_aux": 1., "future_draft_aux": 3.}[job["arm"]])
        _worker(old, entry, index)
    failed = protocol.planned_jobs()[13]
    _json(old / "workers" / (protocol.job_key(failed) + ".timing.json"), {
          "job": failed, "status": "failed", "failure_reason": "fixture timeout", "budget_limited": True,
          "started_perf_counter": 100., "ended_perf_counter": 105., "cleanup": "terminated-owned-worker"})
    (old / "workers" / (protocol.job_key(failed) + ".log")).write_text("failed worker history", encoding="utf-8")
    charged = 1790.1153369722888
    history = {"status": "failed", "scientific_claim": False, "limitations": ["fixture old failure"], "test_read": False,
               "finalized": False, "jobs_completed": protocol.planned_jobs()[:13], "budget_limited": True, "partial": True,
               "gpu_phase_elapsed_seconds": charged, "gpu_hours_charged": charged / 3600,
               "cap_round_seconds": 1800., "cap_gpu_seconds": 3600., "failure_reason": "fixture hard timeout"}
    _json(old / "attempt.json", {**history, "whole_elapsed_seconds": 1805.1085775829852, "outcome": None})
    _json(old / "execution_attempt.json", {**history, "jobs_planned": protocol.planned_jobs(),
          "billing_scope": "full continuous original GPU charge including failed worker and gaps"})
    for name in ("environment.json", "prepare_attempt.json", "run_started.json"):
        _json(old / name, {"scientific_claim": False, "fixture": True})
    new = {"format": "r7-m3-validation-complement-protocol-v1", "scientific_claim": False,
           "limitations": ["fixture, not a scientific run"], "test_read": False, "output": str(out), "original_output": str(old),
           "original_protocol_sha256": original["protocol_sha256"], "original_code_zip_sha256": original["code"]["code_zip_sha256"],
           "original_file_sha256": results._inventory(old), "planned_seconds": 1800., "hard_cap_seconds": 3600.,
           "cleanup_reserve_seconds": 10., "jobs": protocol.planned_jobs()[13:], "gpu": original["gpu"],
           "code": _archive(out, source_root, names)}
    _freeze(new)
    for index, job in enumerate(new["jobs"]):
        entry = _evaluation(original, job, out, training[(job["seed"], job["arm"])],
                            {"aux_off": 20., "input_aux": 1., "future_draft_aux": 30.}[job["arm"]] if job["seed"] == 42
                            else {"aux_off": 2., "input_aux": 1., "future_draft_aux": 3.}[job["arm"]])
        train_job = {"phase": "train", "seed": job["seed"], "arm": job["arm"], "lead": None}
        entry.update(complement_protocol_sha256=new["protocol_sha256"],
                     original_training_receipt_sha256=sha256_file(worker_result_path(old, train_job)),
                     archived_evaluator_code={key: original["code"][key] for key in
                                              ("code_zip_sha256", "source_tree_sha256", "model_code_sha256")})
        _worker(out, entry, index)
    execution = {"status": "evaluations-complete", "evaluation_coverage_complete": True, "finalized": False,
                 "jobs_completed": new["jobs"], "whole_elapsed_seconds": 1900., "gpu_phase_elapsed_seconds": 1850.,
                 "gpu_hours_charged": 1850 / 3600, "soft_overrun_seconds": 100., "budget_limited": False, "partial": False,
                 "continuous_clock": True, "billing_scope": "full continuous complement GPU charge",
                 "execution_whole_scope": results.EXECUTION_WHOLE_SCOPE, "whole_round_cost_reference": str(out / "attempt.json"),
                 "scientific_claim": False, "limitations": ["fixture snapshot"], "test_read": False,
                 "started_perf_counter": 10000., "hard_deadline_perf_counter": 13600., "ended_perf_counter": 11900.,
                 "first_gpu_spawn_started_perf_counter": 10020., "last_owned_gpu_reap_perf_counter": 11870.}
    return {"out": out, "old": old, "original": original, "protocol": new, "execution": execution, "source_root": source_root}


def _finish(complete):
    return results.finalize_complement(complete["out"], complete["protocol"], complete["execution"])


def test_complete_cross_attempt_coverage_seed_pairing_costs_and_pins(complete):
    before = results._inventory(complete["old"])
    assert len(before) == 109
    outcome = _finish(complete)
    assert outcome["status"] == "descriptive-complete" and not outcome["paused"] and not outcome["advance_next_node"]
    out = complete["out"]
    merged = protocol.read_json(out / "merged_result.json")
    comparison = protocol.read_json(out / "paired_comparison.json")
    sources = protocol.read_json(out / "source_manifest.json")
    assert merged["protocol"] == complete["original"] and merged["protocol"]["output"] == str(complete["old"])
    assert merged["aggregation_protocol"] == complete["protocol"] and merged["original_finalizer_accepted"] is False
    assert len(comparison["table"]) == 255
    assert all(row["seeds"] == [41, 42] and set(row["seed_rmse"]) == {"41", "42"} for row in comparison["table"])
    assert all(len(pair["cells"]) == 85 for pair in comparison["pairs"].values())
    cell = comparison["pairs"]["input_aux - aux_off"]["cells"]["6h|channel0"]
    assert cell["seed_deltas"] == {"41": -1., "42": -19.}  # focus-minus-baseline, NOT seed41-minus-seed42.
    assert comparison["pairs"]["future_draft_aux - aux_off"]["totals"]["worsened"] == 85
    assert len(results._read_csv(out / "rmse_table.csv")) == len(results._read_csv(out / "acc_table.csv")) == 510
    assert len(results._read_csv(out / "initialization_table.csv")) == 528
    assert len(results._read_csv(out / "case_table.csv")) == len(results._read_csv(out / "evaluation_time_table.csv")) == 30
    assert len(results._read_csv(out / "allocator_table.csv")) == len(results._read_csv(out / "phase_time_table.csv")) == 36
    assert len(results._read_csv(out / "training_throughput_table.csv")) == len(results._read_csv(out / "training_table.csv")) == 6
    assert len(results._read_csv(out / "arm_table.csv")) == len(results._read_csv(out / "flops_table.csv")) == 3
    skill = results._read_csv(out / "rmse_table.csv")[0]
    assert float(skill["rmse"]) == 2. and float(skill["rmse_climatology"]) == 4.  # No second normalization conversion.
    assert sources["coverage"]["initializations"] == 528 and len(sources["jobs"]) == 36
    assert sum(row["origin"] == "original" for row in sources["jobs"]) == 13
    assert sum(row["origin"] == "complement" for row in sources["jobs"]) == 23
    costs = sources["costs"]
    assert costs["original"]["status"] == "failed" and costs["original"]["includes_failed_worker_and_gaps"]
    assert costs["original"]["gpu_phase_elapsed_seconds"] == 1790.1153369722888
    assert costs["original"]["gpu_hours_charged"] == 0.49725426027008024
    assert costs["original"]["whole_elapsed_seconds"] == 1805.1085775829852
    assert sources["original_history"]["failed_worker_timing"]["status"] == "failed"
    assert costs["accepted_worker_subtotals_not_charges"]["original"] == 13. < costs["original"]["gpu_phase_elapsed_seconds"]
    assert costs["total"]["gpu_phase_elapsed_seconds"] == 1790.1153369722888 + 1850.
    assert costs["total"]["whole_elapsed_seconds"] is None and costs["complement"]["soft_overrun_seconds_snapshot"] == 100.
    assert costs["complement"]["whole_round_cost_reference"]["sha256"] is None
    assert not (out / "attempt.json").exists() and not (out / "execution_attempt.json").exists()
    assert merged["budget"]["gpu_phase_elapsed_seconds"] > complete["original"]["gpu"]["max_round_seconds"]
    assert results._inventory(complete["old"]) == before
    assert protocol.read_json(complete["old"] / "attempt.json")["status"] == "failed"
    assert sha256_file(out / "cpu_profile.json") == sha256_file(complete["old"] / "cpu_profile.json")
    assert sources["cpu_profile"]["remeasured"] is False
    manifest = protocol.read_json(out / "artifact_manifest.json")
    assert manifest["final_round_sealed"] is False and manifest["outcome"] == outcome
    assert set(manifest["files_sha256"]) == set(results._inventory(out)) - {"artifact_manifest.json"}
    assert all(sha256_file(out / name) == pin for name, pin in manifest["files_sha256"].items())
    with pytest.raises(FileExistsError):
        _finish(complete)
    assert results._inventory(complete["old"]) == before


@pytest.mark.parametrize("change", ["subtotal", "gpustamp", "wholestamp", "deadline", "missingstamp"])
def test_actual_continuous_cost_intervals_cannot_be_substituted_or_lost(complete, change):
    execution = complete["execution"]
    if change == "subtotal":
        execution["gpu_phase_elapsed_seconds"] = 23.  # Success worker subtotal omits startup/gaps/cleanup.
        execution["gpu_hours_charged"] = 23. / 3600
    elif change == "missingstamp":
        execution.pop("last_owned_gpu_reap_perf_counter")
    else:
        execution[{"gpustamp": "first_gpu_spawn_started_perf_counter", "wholestamp": "ended_perf_counter",
                   "deadline": "hard_deadline_perf_counter"}[change]] += 1
    with pytest.raises(ValueError, match="timestamp|snapshot"):
        _finish(complete)
    assert not (complete["out"] / "merged_result.json").exists()


def test_exact_original_names_reject_rehashed_same_size_replacement(complete):
    old = complete["old"]
    historical = old / "seed41/training/aux_off/update_0000100.pt"
    content = historical.read_bytes()
    historical.unlink()
    (old / "unrelated-artifact.pt").write_bytes(content)
    new = complete["protocol"]
    new["original_file_sha256"] = results._inventory(old)
    assert len(new["original_file_sha256"]) == 109
    _freeze(new)
    with pytest.raises(ValueError, match="109 immutable"):
        _finish(complete)
    assert not (complete["out"] / "merged_result.json").exists()


@pytest.mark.parametrize("change", ["missing", "wrongseed", "skip", "sha", "path", "csvpath", "checkpoint", "trainingpin",
    "archivepin", "newprotocol", "originalprotocol", "model", "data", "contract", "baseline", "baselinebool", "prebaseline",
    "nonfinitepeak", "nonfiniteelapsed", "cases", "caseunits", "skillunits", "skillnan", "rmsepartial", "accpartial",
    "rmsepooled", "sidecar", "trainingprotocol", "source", "timing"])
def test_receipt_and_artifact_guards_never_publish_partial_merge(complete, change):
    out = complete["out"]
    job = complete["protocol"]["jobs"][-1]
    path = worker_result_path(out, job)
    entry = protocol.read_json(path)
    if change == "missing":
        path.unlink()
    else:
        direct = {"skip": ("status", "skipped"), "wrongseed": ("seed", 43), "sha": ("checkpoint_sha256", "0" * 64),
                  "path": ("evaluation_dir", str(Path(entry["evaluation_dir"]).parent)),
                  "csvpath": ("acc_csv", str(complete["old"] / "unrelated.csv")), "checkpoint": ("checkpoint", entry["checkpoint"] + ".other"),
                  "trainingpin": ("original_training_receipt_sha256", "0" * 64), "archivepin": ("archived_evaluator_code", {}),
                  "newprotocol": ("complement_protocol_sha256", "0" * 64), "originalprotocol": ("protocol_sha256", "0" * 64),
                  "model": ("model_code_sha256", "0" * 64), "data": ("data_identity", "0" * 64), "contract": ("process_supervision", {}),
                  "nonfinitepeak": ("peak_reserved_bytes", float("inf")), "nonfiniteelapsed": ("elapsed_seconds", float("nan"))}
        if change in direct:
            key, value = direct[change]
            entry[key] = value
        elif change in ("baseline", "baselinebool", "prebaseline"):
            entry["pre_init_baseline" if change == "prebaseline" else "baseline"]["allocated_bytes"] = False if change == "baselinebool" else 1
        elif change == "timing":
            timing_path = path.with_suffix(".timing.json")
            timing = protocol.read_json(timing_path)
            timing["status"] = "failed"
            _json(timing_path, timing)
        else:
            name = ("provenance.json" if change in ("cases", "caseunits", "sidecar", "trainingprotocol", "source") else
                    "climatology_skill.csv" if change.startswith("skill") else "acc.csv" if change == "accpartial" else "rmse.csv")
            artifact = Path(entry["evaluation_dir"]) / name
            if name == "provenance.json":
                value = protocol.read_json(artifact)
                if change == "cases":
                    value["initializations"][0]["valid_times"][0] = "wrong-valid-time-with-same-count"
                elif change == "caseunits":
                    value["units"][0] = "normalized"
                else:
                    value[{"sidecar": "process_scale_sidecar_identity", "trainingprotocol": "training_protocol_sha256",
                           "source": "source_declaration"}[change]] = "wrong-identity"
                _json(artifact, value)
            else:
                rows = results._read_csv(artifact)
                if change in ("rmsepartial", "accpartial"):
                    rows.pop()
                else:
                    rows[0][{"skillunits": "unit", "skillnan": "rmse_forecast", "rmsepooled": "rmse"}[change]] = (
                        "normalized" if change == "skillunits" else "nan" if change == "skillnan" else "900")
                _csv(artifact, rows)
            entry["artifact_sha256"][name] = sha256_file(artifact)  # Rehashing cannot bypass schema/case checks.
        _json(path, entry)
    with pytest.raises((ValueError, FileNotFoundError)):
        _finish(complete)
    assert not (out / "merged_result.json").exists()
    assert not (out / "artifact_manifest.json").exists()


@pytest.mark.parametrize("change", ["missingpin", "oldfile", "olddigest", "archivebytes", "helperdrift", "soft", "hard",
                                    "jobset", "oldsuccess", "costlost", "checkpoint400", "report", "sidecar", "source"])
def test_protocol_original_training_and_history_guards(complete, change):
    new, old, original = complete["protocol"], complete["old"], complete["original"]
    if change == "missingpin":
        new["original_file_sha256"].pop("code_status.txt")
        _freeze(new)
    elif change in ("oldfile", "archivebytes"):
        with (old / ("code_status.txt" if change == "oldfile" else "code.zip")).open("ab") as handle:
            handle.write(b"changed immutable source")
    elif change == "olddigest":
        original["protocol_sha256"] = "0" * 64
        _json(old / "protocol.json", original)
    elif change == "helperdrift":
        with (complete["source_root"] / "training/r7_m3_results.py").open("ab") as handle:
            handle.write(b"\n# drift must not change the frozen verifier\n")
    elif change in ("soft", "hard", "jobset"):
        if change == "jobset":
            new["jobs"].pop()
        else:
            new["planned_seconds" if change == "soft" else "hard_cap_seconds"] = 9000
        _freeze(new)
    elif change in ("oldsuccess", "costlost"):
        path = old / "attempt.json"
        value = protocol.read_json(path)
        if change == "oldsuccess":
            value["status"] = "success"
        else:
            value["gpu_phase_elapsed_seconds"] = None
        _json(path, value)
        new["original_file_sha256"] = results._inventory(old)
        _freeze(new)
    elif change in ("checkpoint400", "report"):
        path = worker_result_path(old, protocol.planned_jobs()[0])
        entry = protocol.read_json(path)
        if change == "checkpoint400":
            entry["selected_update"] = 300
            entry["checkpoint"] = str(Path(entry["checkpoint"]).with_name("update_0000300.pt"))
            entry["checkpoint_sha256"] = sha256_file(entry["checkpoint"])
        else:
            report = protocol.read_json(entry["training_report"])
            report["losses"].pop()
            _json(Path(entry["training_report"]), report)
            entry["training_report_sha256"] = sha256_file(entry["training_report"])
        _json(path, entry)
        new["original_file_sha256"] = results._inventory(old)
        _freeze(new)
    else:
        Path(original["sidecar"]["path"] if change == "sidecar" else original["sources"]["source_path"]).write_bytes(b"changed source")
    with pytest.raises((ValueError, FileNotFoundError)):
        _finish(complete)
    assert not (complete["out"] / "merged_result.json").exists()


@pytest.mark.parametrize("change", ["jobs", "status", "coverage", "finalized", "partial", "budget", "nan", "bool",
                                    "hard", "gpu", "hours", "overrun", "continuous", "missing", "scope", "futurepath"])
def test_execution_guards_run_before_any_original_artifact_read(complete, monkeypatch, change):
    execution = complete["execution"]
    changes = {"status": ("status", "success"), "coverage": ("evaluation_coverage_complete", False),
               "finalized": ("finalized", True), "partial": ("partial", True), "budget": ("budget_limited", True),
               "nan": ("whole_elapsed_seconds", float("nan")), "bool": ("gpu_phase_elapsed_seconds", False),
               "hard": ("whole_elapsed_seconds", 3600.1), "gpu": ("gpu_phase_elapsed_seconds", 1901.),
               "hours": ("gpu_hours_charged", 0.), "overrun": ("soft_overrun_seconds", 0.),
               "continuous": ("continuous_clock", False), "scope": ("execution_whole_scope", "full round including unmeasured finalization"),
               "futurepath": ("whole_round_cost_reference", str(complete["old"] / "attempt.json"))}
    if change == "jobs":
        execution["jobs_completed"] = execution["jobs_completed"][:-1]
    elif change == "missing":
        execution.pop("billing_scope")
    else:
        key, value = changes[change]
        execution[key] = value
    monkeypatch.setattr(results, "_protocols", lambda *args: pytest.fail("invalid execution must stop before artifact reads"))
    with pytest.raises(ValueError):
        _finish(complete)
    assert not (complete["out"] / "merged_result.json").exists()


def test_any_unresolved_keeps_original_frozen_pause_line(complete):
    job = {"phase": "evaluate", "seed": 42, "arm": "input_aux", "lead": 72}
    path = worker_result_path(complete["out"], job)
    entry = protocol.read_json(path)
    folder = Path(entry["evaluation_dir"])
    rows = results._read_csv(folder / "rmse.csv")
    rows[0]["rmse"] = 21.  # Opposite sign to seed41; every other cell still fully covered.
    _csv(folder / "rmse.csv", rows)
    rows = results._read_csv(folder / "climatology_skill.csv")
    rows[0].update(rmse_forecast=21., mse_skill=1 - 21. ** 2 / 16)
    _csv(folder / "climatology_skill.csv", rows)
    provenance = protocol.read_json(folder / "provenance.json")
    for case in provenance["initializations"]:
        case["mse"][0][0] = 21. ** 2
    _json(folder / "provenance.json", provenance)
    entry["artifact_sha256"] = {name: sha256_file(folder / name) for name in results.EVALUATION_FILES.values()}
    _json(path, entry)
    outcome = _finish(complete)
    assert outcome["paused"] and outcome["status"] == "paused" and outcome["advance_next_node"] is False
    assert outcome["scientific_claim"] is False and outcome["scientific_gate_evaluated"] is False
    assert outcome["unresolved_cells"]["input_aux - aux_off"] == ["72h|channel0"]
    comparison = protocol.read_json(complete["out"] / "paired_comparison.json")
    assert outcome == results.descriptive_outcome(comparison["pairs"])
    assert comparison["pairs"]["input_aux - aux_off"]["totals"] == {"improved": 84, "worsened": 0, "unresolved": 1}
    assert protocol.read_json(complete["old"] / "attempt.json")["status"] == "failed"
