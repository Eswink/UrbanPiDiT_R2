"""Identity and measurement guards for the N1 evaluation-cost supplement."""
from __future__ import annotations

import csv
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import time
import zipfile

ARMS = ("process_spacetime_rwa", "process_local_solver", "process_local_solver_frozen_z")
SEEDS = (41, 42)
LEADS = (6, 12, 24, 48, 72)
COUNTS = {6: 22, 12: 21, 24: 19, 48: 15, 72: 11}
VARIABLES = "t2m u10 v10 mslp z850 t850 q850 u850 v850 z500 t500 q500 u500 v500 z250 u250 v250".split()
ORIGINAL_PROTOCOL_SHA = "e19ef488be60136364702b1df389e5f58be7ebf3845487289139e10d30231e01"
ORIGINAL_ZIP_SHA = "5fd26146af2a7d11016fb769d67390f5daa23a73620de9ae83e2e6cc38a35a0a"
SUPPLEMENT_CAP_SECONDS = 324.0


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)


def digest(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def plain_path(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("symlink input/output is not allowed")
    return path.resolve()


def pinned_file(path, expected, root=None):
    path = plain_path(path)
    if root is not None and not path.is_relative_to(plain_path(root)):
        raise ValueError("artifact escapes its archive root")
    if sha256(path) != expected:
        raise ValueError(f"artifact digest mismatch: {path}")
    return {"path": str(path), "sha256": expected}


def verify_authorization(value):
    if (value.get("status") != "authorized" or value.get("channel") != "AskUserQuestion"
            or not isinstance(value.get("user_response"), str) or not value["user_response"].strip()
            or value.get("scope") != "n1-evaluation-cost-supplement"
            or value.get("gpu_seconds_cap") != SUPPLEMENT_CAP_SECONDS
            or value.get("training_updates") != 0 or value.get("evaluations") != 30
            or value.get("test_read") is not False or value.get("automatic_retry") is not False):
        raise ValueError("named decision-0021 cost-supplement authorization required")


def verify_registration(path):
    value = read_json(path)
    body = {key: item for key, item in value.items() if key != "protocol_sha256"}
    if (digest(body) != value.get("protocol_sha256")
            or value.get("format") != "r7-n1-evaluation-cost-supplement-v1"
            or value.get("gpu_seconds_cap") != SUPPLEMENT_CAP_SECONDS
            or value.get("training_updates") != 0 or value.get("test_read") is not False
            or value.get("original_protocol_sha256") != ORIGINAL_PROTOCOL_SHA):
        raise ValueError("supplement registration mismatch")
    verify_authorization(value["authorization"])
    expected = set(itertools.product(SEEDS, ARMS, LEADS))
    keys = [(t["seed"], t["arm"], t["lead"]) for t in value["tasks"]]
    if len(keys) != 30 or set(keys) != expected or len(set(keys)) != len(keys):
        raise ValueError("supplement requires exactly the original 30 evaluations")
    verify_validation_scope(value)
    return value


def verify_validation_scope(value):
    original = value["original_protocol"]
    body = {key: item for key, item in original.items() if key != "protocol_sha256"}
    if original.get("protocol_sha256") != ORIGINAL_PROTOCOL_SHA or digest(body) != ORIGINAL_PROTOCOL_SHA:
        raise ValueError("original validation protocol mismatch")
    data = original["data"]
    manifest = plain_path(value["val_manifest"])
    manifests = plain_path(data["manifests_dir"])
    if (manifest != plain_path(data["val_manifest"]) or manifest != manifests / "val.jsonl"
            or value["data_identity"] != data["data_identity"]
            or value["source_identity"] != data["source_identity"]):
        raise ValueError("validation-only source scope mismatch; test stays sealed")
    return original


def verify_task_inputs(task, protocol):
    original = verify_validation_scope(protocol)
    verify_source(original)
    records = [json.loads(line) for line in Path(protocol["val_manifest"]).read_text(encoding="utf-8").splitlines()]
    if len(records) != COUNTS[6] or any(record.get("split") != "val" for record in records):
        raise ValueError("validation-only manifest required before evaluation")
    pinned_file(task["checkpoint"]["path"], task["checkpoint"]["sha256"])


def verify_source(protocol):
    data = protocol["data"]
    manifests = plain_path(data["manifests_dir"])
    pins = data["source_identity"]
    files = {}
    for name, key in (("train.jsonl", "train_manifest_sha256"),
                      ("val.jsonl", "val_manifest_sha256"),
                      ("source_preflight.json", "preflight_report_sha256"),
                      ("BUILD_COMPLETE.json", "build_complete_sha256")):
        files[key] = pinned_file(manifests / name, pins[key])
    receipt_path = manifests.parent.parent / "source_receipt.json"
    files["receipt"] = pinned_file(receipt_path, pins["source_receipt_sha256"])
    preflight, receipt = read_json(manifests / "source_preflight.json"), read_json(receipt_path)
    if (read_json(manifests / "BUILD_COMPLETE.json") != {"schema_version": 1, "build_complete": True}
            or preflight["fingerprint"]["sha256"] != pins["source_sha256"]
            or receipt["local_artifact"]["sha256"] != pins["source_sha256"]
            or receipt.get("synthetic_fallback") is not False):
        raise ValueError("source publication/receipt identity mismatch")
    files["source"] = pinned_file(preflight["source_path"], pins["source_sha256"])
    if Path(files["source"]["path"]).stat().st_size != pins["source_bytes"]:
        raise ValueError("source size differs")
    return files


def inspect_archive(archive):
    archive = plain_path(archive)
    protocol = read_json(archive / "protocol.json")
    body = {k: v for k, v in protocol.items() if k != "protocol_sha256"}
    if digest(body) != protocol.get("protocol_sha256") or protocol.get("protocol_sha256") != ORIGINAL_PROTOCOL_SHA:
        raise ValueError("original frozen protocol mismatch")
    zip_pin = pinned_file(archive / "code.zip", ORIGINAL_ZIP_SHA, archive)
    merged = read_json(archive / "merged_result.json")
    if merged["protocol"] != protocol or merged["protocol_sha256"] != ORIGINAL_PROTOCOL_SHA:
        raise ValueError("original merged protocol mismatch")
    tasks, common, inputs = [], {}, [zip_pin]
    for seed in SEEDS:
        seed_protocol = archive / f"seed{seed}/protocol.json"
        seed_result = archive / f"seed{seed}/seed_result.json"
        if read_json(seed_protocol) != protocol:
            raise ValueError("per-seed protocol differs")
        inputs.extend({"path": str(p), "sha256": sha256(p)} for p in (seed_protocol, seed_result))
        for arm in ARMS:
            entry = merged["training"][str(seed)][arm]
            checkpoint = pinned_file(entry["checkpoint"], entry["checkpoint_sha256"], archive)
            if entry["updates_run"] != 400 or entry["selected_update"] != 400 or entry["early_stopped"]:
                raise ValueError("original training endpoint differs")
            inputs.append(checkpoint)
            for lead in LEADS:
                item = merged["evaluation"][f"{seed}/{arm}@{lead}h"]
                directory = plain_path(item["evaluation_dir"])
                if not directory.is_relative_to(archive):
                    raise ValueError("evaluation path escapes archive")
                provenance = read_json(directory / "provenance.json")
                cases = [[r["init_time"], r["valid_times"]] for r in provenance["initializations"]]
                if (provenance["split"] != "val" or item["test_read"] is not False
                        or provenance["n_evaluated"] != COUNTS[lead]
                        or provenance["n_evaluated"] != len(cases)
                        or provenance["channels"] != VARIABLES
                        or provenance["lead_hours"] != [lead]
                        or provenance["training_identity"] != protocol["data"]["data_identity"]
                        or provenance["checkpoint_sha256"] != checkpoint["sha256"]
                        or cases != common.setdefault(lead, cases)):
                    raise ValueError("original validation/case/checkpoint identity differs")
                rmse_hash = sha256(directory / "rmse.csv")
                provenance_hash = sha256(directory / "provenance.json")
                inputs.extend(({"path": str(directory / "rmse.csv"), "sha256": rmse_hash},
                               {"path": str(directory / "provenance.json"), "sha256": provenance_hash}))
                tasks.append({"seed": seed, "arm": arm, "lead": lead, "n_cases": COUNTS[lead],
                              "checkpoint": checkpoint, "original_dir": str(directory),
                              "rmse_sha256": rmse_hash, "provenance_sha256": provenance_hash})
    expected = {f"{s}/{a}@{h}h" for s, a, h in itertools.product(SEEDS, ARMS, LEADS)}
    if set(merged["evaluation"]) != expected:
        raise ValueError("original evaluation set is incomplete or has extras")
    for name in ("protocol.json", "merged_result.json", "attempt.json", "memory_table.csv"):
        inputs.append({"path": str(archive / name), "sha256": sha256(archive / name)})
    return {"original_protocol": protocol, "tasks": tasks, "input_pins": inputs,
            "source_pins": verify_source(protocol)}


def extract_code(archive_path, target):
    entries, names = [], set()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if (path.is_absolute() or ".." in path.parts or "\\" in info.filename
                    or info.filename in names or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError("unsafe/duplicate ZIP member")
            names.add(info.filename)
            if (not path.parts or path.parts[0] not in ("data", "model", "training")
                    or path.suffix != ".py" or any("legacy" in part for part in path.parts)):
                continue
            if info.file_size > 1024 * 1024:
                raise ValueError("archived source member exceeds its bounded extraction")
            destination = Path(target).joinpath(*path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            raw = archive.read(info)
            with destination.open("xb") as stream:
                stream.write(raw)
            entries.append({"path": path.as_posix(), "sha256": hashlib.sha256(raw).hexdigest()})
    required = {"training/r7_evaluate.py", "training/r7_experiment.py", "model/__init__.py",
                "training/r7_frozen_z_intervention.py", "data/r7_evaluation.py"}
    if not required.issubset({e["path"] for e in entries}):
        raise ValueError("archived evaluator dependencies missing")
    return entries


def gpu_exclusive(device, deadline=None, *, gpu_uuid=None):
    if not isinstance(device, str) or not device.startswith("cuda:") or not device[5:].isdigit():
        raise ValueError("explicit single CUDA device required")
    index = int(device[5:])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    identities = visible.split(",") if visible else []
    if identities and index >= len(identities):
        raise ValueError("logical CUDA device outside visible mapping")
    physical = gpu_uuid if gpu_uuid is not None else identities[index] if identities else str(index)
    timeout = min(5.0, deadline - time.perf_counter()) if deadline is not None else 5.0
    if timeout <= 0:
        raise RuntimeError("device-query deadline exhausted")
    devices = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                             shell=False, check=True, capture_output=True, text=True, timeout=timeout).stdout
    matches = [fields[1] for row in devices.splitlines()
               if len(fields := [f.strip() for f in row.split(",")]) == 2 and physical in fields]
    if len(matches) != 1:
        raise ValueError("physical device mapping is ambiguous or missing")
    uuid = matches[0]
    timeout = min(5.0, deadline - time.perf_counter()) if deadline is not None else 5.0
    if timeout <= 0:
        raise RuntimeError("device-query deadline exhausted")
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                               "--format=csv,noheader"], shell=False, check=True,
                              capture_output=True, text=True, timeout=timeout).stdout
    for row in processes.splitlines():
        fields = [field.strip() for field in row.split(",")]
        if len(fields) == 2 and fields[0] == uuid and fields[1] != str(os.getpid()):
            raise RuntimeError("cost measurement device is occupied; do not interfere")
    return {"logical_device": device, "physical_selector": physical, "gpu_uuid": uuid}


def verify_cuda_device(cuda, device, gpu_uuid):
    actual = str(cuda.get_device_properties(device).uuid)
    if actual.lower().removeprefix("gpu-") != gpu_uuid.lower().removeprefix("gpu-"):
        raise RuntimeError("selected CUDA device UUID differs from exclusive physical GPU")
    return gpu_uuid


def reset_measurement(cuda, device):
    cuda.set_device(device)
    gc.collect()
    cuda.synchronize(device)
    cuda.empty_cache()
    cuda.synchronize(device)
    allocated, reserved = cuda.memory_allocated(device), cuda.memory_reserved(device)
    if allocated != 0 or reserved != 0:
        raise RuntimeError("independent evaluation requires zero allocated/reserved baseline")
    cuda.reset_peak_memory_stats(device)
    return {"allocated_bytes": allocated, "reserved_bytes": reserved}


def read_peaks(cuda, device):
    cuda.synchronize(device)
    allocated, reserved = cuda.max_memory_allocated(device), cuda.max_memory_reserved(device)
    if not isinstance(allocated, int) or not isinstance(reserved, int) or not 0 < allocated <= reserved:
        raise RuntimeError("invalid allocated/reserved evaluation peaks")
    return {"peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved}


def evaluation_artifacts(output):
    directory = plain_path(output)
    names = ("provenance.json", "rmse.csv", "acc.csv", "climatology_skill.csv")
    return {name: sha256(plain_path(directory / name)) for name in names}


def compare_evaluation(task, output, report):
    output = plain_path(output)
    if read_json(output / "provenance.json") != report:
        raise ValueError("replay persisted provenance differs from returned report")
    original = Path(task["original_dir"])
    pinned_file(original / "provenance.json", task["provenance_sha256"])
    pinned_file(original / "rmse.csv", task["rmse_sha256"])
    prior = read_json(original / "provenance.json")
    for key in ("training_identity", "checkpoint_sha256", "evaluation_manifest_sha256",
                "channels", "units", "split", "lead_hours", "step_hours", "n_available_windows",
                "n_evaluated", "inference_options", "initializations", "intervention"):
        if report.get(key) != prior.get(key):
            raise ValueError(f"replay provenance differs: {key}")
    with (original / "rmse.csv").open(encoding="utf-8", newline="") as stream:
        before = list(csv.DictReader(stream))
    with (Path(output) / "rmse.csv").open(encoding="utf-8", newline="") as stream:
        after = list(csv.DictReader(stream))
    if len(before) != 17 or before != after:
        raise ValueError("replay RMSE differs; no new tolerance or changed scientific reading")
    return {"rmse_rows": 17, "case_count": report["n_evaluated"], "exact_numeric_replay": True}


def check_deadline(deadline):
    if time.perf_counter() >= deadline:
        raise RuntimeError("supplement deadline exhausted")


def validate_measurements(rows, protocol=None, output=None):
    expected = set(itertools.product(SEEDS, ARMS, LEADS))
    keys = [(r["seed"], r["arm"], r["lead"]) for r in rows]
    if len(keys) != 30 or len(set(keys)) != 30 or set(keys) != expected:
        raise ValueError("exact 30 cost rows required; partial is not acceptance")
    task_map = {} if protocol is None else {
        (t["seed"], t["arm"], t["lead"]): t for t in protocol["tasks"]}
    for row in rows:
        if protocol is not None:
            task = task_map[(row["seed"], row["arm"], row["lead"])]
            directory = Path(output) / f"seed{row['seed']}" / row["arm"] / f"lead_{row['lead']:03d}h"
            if (row["device"] != protocol["device"]
                    or row["execution_device"] != protocol["execution_device"]
                    or row["protocol_sha256"] != protocol["protocol_sha256"]
                    or row["original_checkpoint_sha256"] != task["checkpoint"]["sha256"]
                    or row["gpu_uuid"] != protocol["physical_gpu_uuid"]
                    or plain_path(row["evaluation_dir"]) != plain_path(directory)):
                raise ValueError("cost row does not bind its frozen task/device/protocol")
            if (evaluation_artifacts(directory) != row["evaluation_artifacts_sha256"]
                    or read_json(directory / "cost_measurement.json") != row
                    or compare_evaluation(task, directory, read_json(directory / "provenance.json")) != row["replay"]):
                raise ValueError("cost row replay artifacts differ or are incomplete")
        if (row["n_cases"] != COUNTS[row["lead"]] or not row["replay"]["exact_numeric_replay"]
                or row["baseline"] != {"allocated_bytes": 0, "reserved_bytes": 0}
                or not 0 < row["peak_allocated_bytes"] <= row["peak_reserved_bytes"]
                or not math.isfinite(row["elapsed_seconds"]) or row["elapsed_seconds"] <= 0):
            raise ValueError("cost row measurement/replay contract differs")


def verify_pins(pins):
    for pin in pins:
        pinned_file(pin["path"], pin["sha256"])
