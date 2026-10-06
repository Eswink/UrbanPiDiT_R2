"""Separately frozen train/validation same-case diagnosis, never test scoring."""
from __future__ import annotations

import argparse
from io import BytesIO
import json
import math
from pathlib import Path
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import study_r7_s3_v3_rollout_ft as recipe
from scripts import study_r7_s3_v3_rollout_ft_worker as inventory
from scripts.study_r7_s3_v3_rollout_isolated import fresh_output
from training.r7_experiment import canonical_digest, dataset_identity, model_code_digest
from training.r7_arm_harness import sha256_file
from training.r7_study_process import bounded_process, worker_environment

GPU_UUID = inventory.GPU_UUID
PLANNED_SECONDS, HARD_SECONDS = 1800., 3600.
KNOWN_RESERVED_PEAK = 2434793472
SCREEN = ROOT / "outputs/r7_s3_long_rollout_20261006_attempt01"
SCREEN_RESULT_SHA256 = "a6239caa9e65320064d295bea557ad95729e098f1ff7e039033d8eb31d712ca2"
CANDIDATE_SHA256 = "449bc4ce6d9c15178fecb75a954a265ffb43f7ea8cea92575d95387047ee066b"
VAL_DATA_IDENTITY = "0c34a887216b02b41716e7837f5be4d515ad2027e7b4a4728cbef15accd02d5a"
SOURCE_BYTES = 540856239
DEFAULT_OUTPUT = ROOT / "outputs/r7_s3_same_case_gap_20261006_attempt01"
LEADS = (6, 12, 24, 48, 72)
LIMITATIONS = [
    "One seed and24 deterministic representative cases, not complete weather evaluation or significance.",
    "Training cases and training-fit climatology are in-sample and optimistically self-referential.",
    "Validation is development only; no2023test scoring, independent confirmation or causal underfitting claim.",
    "Physical per-variable errors only; different physical units are never averaged.",
    "No optimizer update or training replay; configuration reproducibility only.",
]


def write_exclusive(path, body):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(body, handle, indent=2, ensure_ascii=False, allow_nan=False)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def execution_files():
    return sorted(set(inventory.execution_files()) | {"scripts/diagnose_r7_s3_same_case_gap.py"})


def checkpoint_pins():
    if sha256_file(SCREEN / "result.json") != SCREEN_RESULT_SHA256:
        raise RuntimeError("registered screen result changed")
    result = read_json(SCREEN / "result.json")
    candidate = result["seeds"]["41"]
    if candidate["checkpoint_sha256"] != CANDIDATE_SHA256:
        raise RuntimeError("registered selected endpoint pin differs")
    parent, parent_sha = recipe.pinned_parent(41)
    return {"parent": {"path": str(parent), "sha256": parent_sha, "updates": 1600, "mode": "l6"},
            "candidate": {"path": candidate["checkpoint"], "sha256": CANDIDATE_SHA256,
                          "updates": 200, "mode": "long_rollout"}}


def frozen_controls():
    return {"selection": "chronological lower median complete initialization per(year,month1/4/7/10)",
        "baseline": "unchanged train-only-month-hour-grid-mean-v1,2400 steps/16buckets of150",
        "expected_climatology": {"n_selected_steps": 2400, "bucket_counts": {
            f"{month:02d}-{hour:02d}": 150 for month in (1, 4, 7, 10) for hour in (0, 6, 12, 18)}},
        "expected_contracts": {
            "parent": {"kind": "process", "seed": 41, "mode": "l6", "total_updates": 1600,
                       "steps": 4, "bf16": False},
            "candidate": {"kind": "process", "seed": 41, "mode": "long_rollout", "total_updates": 200,
                "steps": 4, "bf16": False, "lr": 2e-5, "warmup_updates": 10,
                "weight_decay": 1e-4, "batch_size": 1, "clip": 1.,
                "autoregression": {"physical_steps": 12,
                    "physical_weights": [1., .5, 0., .5, 0., 0., 0., .5, 0., 0., 0., .5],
                    "detach_physical_steps": False, "detach_reasoning_steps": False}}}}


def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    body = {"format": "r7-s3-same-case-gap-protocol-v1", "scientific_claim": False, "test_read": False,
            "scope": "20train and4val deterministic same-case diagnostics; no optimizer/test/selection",
            "instance": str(recipe.INSTANCE), "train_manifest": str(recipe.TRAIN_MANIFEST),
            "val_manifest": str(recipe.VAL_MANIFEST), "seed": 41, "reasoning_steps": 4,
            "lead_hours": list(LEADS), "physical_steps": 12, "bf16": False,
            "train_years": list(range(2017, 2022)), "val_years": [2022], "months": [1, 4, 7, 10],
            **frozen_controls(),
            "interpretation": "Descriptive split/case errors; signs cannot prove optimization or generalization causality.",
            "planned_seconds": PLANNED_SECONDS, "hard_cap_seconds": HARD_SECONDS,
            "started_perf_counter": started, "deadline_perf_counter": deadline,
            "gpu_uuid": GPU_UUID, "estimated_peak_bytes": 2**31, "margin_bytes": 2**31,
            "known_owned_reserved_peak_bytes": KNOWN_RESERVED_PEAK,
            "execution_mode": "direct owned non-spawning bounded workers, first error stops, no resume",
            "source_sha256": recipe._shared().V3_SOURCE_SHA256, "source_bytes": SOURCE_BYTES,
            "registered_screen_result_sha256": SCREEN_RESULT_SHA256,
            "code_archive_input": str(code_archive), "code_commit": code_commit,
            "code_zip_sha256": code_sha256, "limitations": list(LIMITATIONS)}
    body["protocol_sha256"] = canonical_digest(body)
    write_exclusive(output / "preparation_protocol.json", body)


def prepare(output):
    from training.r7_gap_diagnostic import metadata_case_plan
    body = read_json(output / "preparation_protocol.json")
    if body["protocol_sha256"] != canonical_digest({k: v for k, v in body.items() if k != "protocol_sha256"}):
        raise RuntimeError("preparation protocol differs")
    source = recipe.INSTANCE / "source.nc"
    if source.stat().st_size != body["source_bytes"] or sha256_file(source) != body["source_sha256"]:
        raise RuntimeError("full source bytes/SHA differs")
    body.update({"train_data_identity": dataset_identity(recipe.TRAIN_MANIFEST)[0],
                 "val_data_identity": dataset_identity(recipe.VAL_MANIFEST)[0],
                 "model_code_sha256": model_code_digest(), "checkpoints": checkpoint_pins(),
                 "execution_files_sha256": {name: sha256_file(ROOT / name) for name in execution_files()},
                 "case_plan": metadata_case_plan(recipe.TRAIN_MANIFEST, recipe.VAL_MANIFEST)})
    if (body["train_data_identity"] != recipe._shared().V3_DATA_IDENTITY
            or body["val_data_identity"] != VAL_DATA_IDENTITY):
        raise RuntimeError("training/validation identity is not the registered v3 instance")
    for entry in body["checkpoints"].values():
        if sha256_file(entry["path"]) != entry["sha256"]:
            raise RuntimeError("selected checkpoint changed")
    body.pop("protocol_sha256")
    body["protocol_sha256"] = canonical_digest(body)
    write_exclusive(output / "prepared_protocol.json", body)


def validate_protocol(output, deadline):
    body = read_json(output / "protocol.json")
    if body["protocol_sha256"] != canonical_digest({k: v for k, v in body.items() if k != "protocol_sha256"}):
        raise RuntimeError("frozen protocol digest changed")
    expected = {"scientific_claim": False, "test_read": False, "seed": 41, "reasoning_steps": 4,
                "lead_hours": list(LEADS), "physical_steps": 12, "bf16": False,
                "planned_seconds": PLANNED_SECONDS, "hard_cap_seconds": HARD_SECONDS,
                "train_years": list(range(2017, 2022)), "val_years": [2022], "months": [1, 4, 7, 10],
                "gpu_uuid": GPU_UUID, "estimated_peak_bytes": 2**31, "margin_bytes": 2**31,
                "known_owned_reserved_peak_bytes": KNOWN_RESERVED_PEAK,
                "train_manifest": str(recipe.TRAIN_MANIFEST), "val_manifest": str(recipe.VAL_MANIFEST),
                "source_sha256": recipe._shared().V3_SOURCE_SHA256, "source_bytes": SOURCE_BYTES,
                "train_data_identity": recipe._shared().V3_DATA_IDENTITY, "val_data_identity": VAL_DATA_IDENTITY,
                "registered_screen_result_sha256": SCREEN_RESULT_SHA256, "limitations": list(LIMITATIONS),
                **frozen_controls()}
    for key, value in expected.items():
        if body[key] != value or isinstance(value, bool) and type(body[key]) is not bool:
            raise RuntimeError("frozen scope/budget differs: " + key)
    if (not isinstance(deadline, (float, int)) or isinstance(deadline, bool) or not math.isfinite(deadline)
            or deadline > body["deadline_perf_counter"] or deadline <= time.perf_counter()
            or abs(body["deadline_perf_counter"] - body["started_perf_counter"] - HARD_SECONDS) > 1e-6):
        raise RuntimeError("worker deadline differs from frozen round")
    if set(body["execution_files_sha256"]) != set(execution_files()):
        raise RuntimeError("execution inventory changed")
    for name, expected_sha in body["execution_files_sha256"].items():
        if sha256_file(ROOT / name) != expected_sha:
            raise RuntimeError("execution source changed: " + name)
    if body["checkpoints"] != checkpoint_pins():
        raise RuntimeError("selected endpoint declarations changed")
    if body["model_code_sha256"] != model_code_digest():
        raise RuntimeError("model identity changed")
    return body


def archive(output, body):
    import hashlib
    source = Path(body["code_archive_input"])
    if any(p.is_symlink() for p in (source, *source.parents)):
        raise ValueError("nonlinked exact archive required")
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != body["code_zip_sha256"]:
        raise RuntimeError("archive SHA changed")
    with zipfile.ZipFile(BytesIO(payload)) as zipped:
        if zipped.comment.decode() != body["code_commit"]:
            raise RuntimeError("archive commit differs")
        for name, expected in body["execution_files_sha256"].items():
            if name.endswith(".py") and hashlib.sha256(zipped.read(name)).hexdigest() != expected:
                raise RuntimeError("archive execution source differs: " + name)
    with (output / "code.zip").open("xb") as handle:
        handle.write(payload)
    with (output / "code_commit.txt").open("x", encoding="utf-8") as handle:
        handle.write(body["code_commit"] + "\n")
    write_exclusive(output / "archive_receipt.json", {"scientific_claim": False, "test_read": False,
        "code_commit": body["code_commit"], "code_zip_sha256": sha256_file(output / "code.zip")})


def measurement(output, body, deadline):
    import torch
    from training.r7_gap_diagnostic import run_gap_diagnostic
    recipe._shared().pin_declared_gpu(GPU_UUID)
    torch.set_num_threads(4)
    pins = body["checkpoints"]
    result = run_gap_diagnostic(body["train_manifest"], body["val_manifest"],
        parent_checkpoint=pins["parent"]["path"], candidate_checkpoint=pins["candidate"]["path"],
        parent_sha256=pins["parent"]["sha256"], candidate_sha256=pins["candidate"]["sha256"],
        expected_case_plan=body["case_plan"], expected_train_identity=body["train_data_identity"],
        expected_val_identity=body["val_data_identity"], expected_source_sha256=body["source_sha256"],
        protocol_sha256=body["protocol_sha256"], device_name="cuda:0", deadline=deadline, protocol=body)
    write_exclusive(output / "measurements.json", result)
    peak = int(torch.cuda.max_memory_reserved(0))
    write_exclusive(output / "measurement_receipt.json", {"scientific_claim": False, "test_read": False,
        "protocol_sha256": body["protocol_sha256"], "measurements_sha256": sha256_file(output / "measurements.json"),
        "owned_cuda_reserved_peak_bytes": peak, "limitations": list(LIMITATIONS)})


def reading(output, body):
    receipt = read_json(output / "measurement_receipt.json")
    if (receipt["protocol_sha256"] != body["protocol_sha256"]
            or sha256_file(output / "measurements.json") != receipt["measurements_sha256"]):
        raise RuntimeError("measurement identity differs")
    result = read_json(output / "measurements.json")
    if result.get("scientific_claim") is not False or result.get("test_read") is not False:
        raise RuntimeError("diagnostic claims/test flags differ")
    if not result.get("limitations"):
        raise RuntimeError("diagnostic limitations required")
    if result["protocol_sha256"] != body["protocol_sha256"]:
        raise RuntimeError("measurement protocol differs")
    if len(result.get("cases", [])) != 24:
        raise RuntimeError("all24 predeclared cases required")
    for case in result["cases"]:
        if set(case["metrics"]) != {"parent", "candidate", "climatology"}:
            raise RuntimeError("three paired case metric arms required")
        for metric in case["metrics"].values():
            for key in ("mse", "rmse"):
                values = metric[key]
                if (len(values) != 5 or any(len(row) != 17 for row in values)
                        or any(isinstance(v, bool) or not isinstance(v, (float, int))
                               or not math.isfinite(v) or v < 0 for row in values for v in row)):
                    raise RuntimeError("finite physical5x17case metrics required")
    validate_reading(result, body)
    write_exclusive(output / "reading_receipt.json", {"scientific_claim": False, "test_read": False,
        "protocol_sha256": body["protocol_sha256"], "measurements_sha256": receipt["measurements_sha256"],
        "selection_sha256": body["case_plan"]["selection_sha256"],
        "status": "recorded-diagnostic-only", "limitations": list(LIMITATIONS)})


def validate_reading(result, body):
    from training.r7_gap_diagnostic import aggregate_cases, metric_summary, _gap
    plan = body["case_plan"]
    expected = {"case_plan": plan, "selection_sha256": plan["selection_sha256"], "n_evaluated": 24,
        "train_data_identity": body["train_data_identity"], "val_data_identity": body["val_data_identity"],
        "source_sha256": body["source_sha256"], "model_code_sha256": body["model_code_sha256"],
        "channels": plan["channels"], "units": plan["units"], "lead_hours": list(LEADS),
        "training_normalization": {"mean": plan["normalization_mean"], "std": plan["normalization_std"]},
        "optimizer_imported": False, "optimizer_updates": 0, "diagnostic_training_mode": False,
        "bf16": False, "reasoning_steps": 4, "physical_transitions_per_model_case": 12}
    for key, value in expected.items():
        if key not in result or canonical_digest(result[key]) != canonical_digest(value):
            raise RuntimeError("measurement frozen identity/scope differs: " + key)
    source_identity = result["source_identity"]
    source_path = (Path(plan["train_manifest"]).parent / plan["source_declaration"]).resolve()
    if (source_identity["sha256"] != body["source_sha256"] or source_identity["scope"] != "full-local-file"
            or Path(source_identity["path"]).resolve() != source_path
            or type(source_identity["bytes"]) is not int or source_identity["bytes"] != body["source_bytes"]):
        raise RuntimeError("measurement full source identity differs")
    for role, pin in body["checkpoints"].items():
        observed = result["checkpoints"][role]
        for key in ("sha256", "updates", "mode"):
            if canonical_digest(observed[key]) != canonical_digest(pin[key]):
                raise RuntimeError("measurement endpoint identity differs: " + role + "." + key)
        if (observed["optimizer_imported"] is not False or type(observed["seed"]) is not int
                or observed["seed"] != 41 or observed["model_code_sha256"] != body["model_code_sha256"]):
            raise RuntimeError("measurement endpoint optimizer/seed/model code differs")
    climate = result["climatology"]
    climate_expected = {**plan["climatology_metadata"], **body["expected_climatology"],
        "kind": "train-only-month-hour-grid-mean-v1", "channels": plan["channels"]}
    for key, value in climate_expected.items():
        if canonical_digest(climate.get(key)) != canonical_digest(value):
            raise RuntimeError("measurement unchanged climatology differs: " + key)
    if (set(climate["means"]) != set(climate["bucket_counts"])
            or climate["mean_identity_sha256"] != canonical_digest(climate["means"])):
        raise RuntimeError("measurement climatology mean identities differ")
    for mean in climate["means"].values():
        if (mean["shape"] != plan["shape"][1:] or mean["dtype"] != "<f8"
                or not isinstance(mean["sha256"], str) or len(mean["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in mean["sha256"])):
            raise RuntimeError("measurement original FP64 climatology shape/digest differs")
    for case, frozen in zip(result["cases"], plan["cases"], strict=True):
        if canonical_digest({key: case.get(key) for key in frozen}) != canonical_digest(frozen):
            raise RuntimeError("measurement ordered case metadata differs")
        recomputed = metric_summary({role: metric["mse"] for role, metric in case["metrics"].items()})
        for key, value in recomputed.items():
            if canonical_digest(case.get(key)) != canonical_digest(value):
                raise RuntimeError("measurement case metric arithmetic differs: " + key)
    aggregates = aggregate_cases(result["cases"])
    if canonical_digest(result.get("aggregates")) != canonical_digest(aggregates):
        raise RuntimeError("measurement equal-case aggregation differs")
    if canonical_digest(result.get("train_val_gap")) != canonical_digest(_gap(aggregates)):
        raise RuntimeError("measurement train/validation gap arithmetic differs")


def worker(output, phase, deadline):
    from scripts.r7_m3_offline import deny_network
    deny_network()
    if phase == "prepare":
        return prepare(output)
    body = validate_protocol(output, deadline)
    if phase == "archive":
        return archive(output, body)
    if phase == "measurement":
        return measurement(output, body, deadline)
    return reading(output, body)


def run_attempt(output, *, code_archive, code_commit, code_sha256):
    from scripts.r7_m3_offline import deny_network
    deny_network()
    output = fresh_output(output)
    started = time.perf_counter()
    deadline = started + HARD_SECONDS
    output.mkdir(parents=True, exist_ok=False)
    processes, body, peak = [], None, KNOWN_RESERVED_PEAK
    try:
        if (not code_archive or not isinstance(code_commit, str) or len(code_commit) != 40
                or not isinstance(code_sha256, str) or len(code_sha256) != 64):
            raise ValueError("prepared exact archive/commit/hash required")
        freeze(output, started=started, deadline=deadline, code_archive=Path(code_archive).absolute(),
               code_commit=code_commit, code_sha256=code_sha256)
        for phase in ("prepare", "archive", "measurement", "reading"):
            gate = recipe._shared().gpu_gate(GPU_UUID, estimated_peak_bytes=2**31,
                                             owned_reserved_peak_bytes=peak, margin_bytes=2**31)
            write_exclusive(output / f"{phase}_spawn_gate.json", gate)
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", phase,
                       "--out", str(output), "--deadline", str(deadline)]
            process = bounded_process(command, cwd=ROOT, log_path=output / f"{phase}.log",
                                      deadline=deadline, env=worker_environment(GPU_UUID))
            processes.append({"phase": phase, **process})
            write_exclusive(output / f"{phase}_process.json", processes[-1])
            if process["status"] != "success":
                raise RuntimeError(phase + " failed: " + process["status"])
            if phase == "prepare":
                body = read_json(output / "prepared_protocol.json")
                write_exclusive(output / "protocol.json", body)
            elif phase == "measurement":
                peak = max(peak, read_json(output / "measurement_receipt.json")["owned_cuda_reserved_peak_bytes"])
            print(json.dumps({"phase_done": phase, "elapsed_seconds": time.perf_counter() - started}), flush=True)
        elapsed = time.perf_counter() - started
        if elapsed >= HARD_SECONDS:
            raise RuntimeError("hard deadline passed before terminal publication")
        result = {"scientific_claim": False, "test_read": False, "protocol_sha256": body["protocol_sha256"],
            "elapsed_seconds_total": elapsed, "soft_overrun_seconds": max(0., elapsed - PLANNED_SECONDS),
            "hard_overrun_seconds": max(0., elapsed - HARD_SECONDS), "processes": processes,
            "whole_round_conservative_gpu_hours": elapsed / 3600., "decision": "diagnostic-only",
            "measurement_receipt": read_json(output / "measurement_receipt.json"), "limitations": list(LIMITATIONS)}
        write_exclusive(output / "result.json", result)
        write_exclusive(output / "attempt.json", {"status": "complete", "scientific_claim": False, "test_read": False,
            "protocol_sha256": body["protocol_sha256"], "decision": "diagnostic-only", "elapsed_seconds_total": elapsed})
        return result
    except BaseException as exc:
        elapsed = time.perf_counter() - started
        write_exclusive(output / "failure.json", {"status": "failed", "scientific_claim": False, "test_read": False,
            "protocol_sha256": None if body is None else body["protocol_sha256"], "elapsed_seconds_total": elapsed,
            "soft_overrun_seconds": max(0., elapsed - PLANNED_SECONDS), "hard_overrun_seconds": max(0., elapsed - HARD_SECONDS),
            "whole_round_conservative_gpu_hours": elapsed / 3600., "processes": processes,
            "failure_reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
            "limitations": ["Failed attempt retained and fully charged; no retry or in-place resume", *LIMITATIONS]})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--code-archive", type=Path)
    parser.add_argument("--code-commit")
    parser.add_argument("--code-sha256")
    parser.add_argument("--worker", choices=("prepare", "archive", "measurement", "reading"))
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    if args.worker:
        return worker(args.out, args.worker, args.deadline)
    return run_attempt(args.out, code_archive=args.code_archive, code_commit=args.code_commit,
                       code_sha256=args.code_sha256)


if __name__ == "__main__":
    main()
