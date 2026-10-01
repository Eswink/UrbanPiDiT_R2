"""Supervise the separately authorized N1 evaluation-memory cost supplement."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.r7_n1_cost_replay import (
    ARMS, LEADS, MIN_FREE_MIB, ORIGINAL_PROTOCOL_SHA, ORIGINAL_ZIP_SHA, SEEDS,
    SUPPLEMENT_CAP_SECONDS, WHOLE_CAP_SECONDS,
    check_deadline, digest, extract_code, gpu_headroom, inspect_archive, plain_path,
    select_shared_gpu,
    read_json, sha256, validate_measurements, verify_authorization, verify_pins,
    verify_registration, verify_probe_report, write_json,
)

LIMITATIONS = [
    "Cost supplement only: zero optimizer updates, no new arm/seed/lead or selection.",
    "Original code.zip/checkpoints and exactly original val cases; test stays sealed.",
    "Each evaluation begins at zero allocated/reserved bytes with explicit-device reset.",
    "Whole-call memory includes model loading, IO and metrics, not isolated forward memory.",
    "Fresh processes and allocator cleanup change cost scope; original records stay unchanged.",
    "Exact RMSE/case replay required; no new tolerance or scientific criterion.",
    "Original primary stays cannot-distinguish; no new campaign node is authorized.",
    "Default co-residency: free-memory queries are not reservations; neighbors may affect wall time.",
    "External PIDs are recorded only; no neighbor signal, suspension or termination is authorized.",
    "Probe diagnostics use a private release API only in P1, never in accepted cost evaluations.",
]


def deny_network():
    def refused(*args, **kwargs):
        raise RuntimeError("cost supplement is offline")
    socket.socket.connect = refused
    socket.create_connection = refused


def reserve_output(output, archive, manifests):
    output, archive, manifests = map(plain_path, (output, archive, manifests))
    frozen_failure = plain_path(ROOT / "outputs/r7_n1_eval_cost_supplement")
    for source in (archive, manifests, manifests.parent, manifests.parent.parent, frozen_failure):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("supplement output overlaps archived/data inputs")
    output.relative_to(ROOT / "outputs")
    output.mkdir(parents=True, exist_ok=False)
    return output


def archive_wrapper(output):
    names = ("training/r7_n1_cost_replay.py", "scripts/measure_r7_n1_eval_cost.py",
             "scripts/measure_r7_n1_eval_worker.py", "tests/test_r7_n1_eval_cost.py",
             "scripts/probe_r7_n1_allocator.py", "tests/test_r7_n1_allocator_probe.py",
             "tests/test_r7_n1_cost_coresidency.py", "tests/test_gpu_policy_metadata.py")
    blobs = {}
    with zipfile.ZipFile(output / "measurement_code.zip", "x", zipfile.ZIP_DEFLATED) as bundle:
        for name in names:
            raw = (ROOT / name).read_bytes()
            bundle.writestr(name, raw)
            blobs[name] = hashlib.sha256(raw).hexdigest()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            capture_output=True, text=True, shell=False).stdout.strip()
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT, check=True,
                            capture_output=True, text=True, shell=False).stdout
    for name, text in (("code_commit.txt", commit + "\n"), ("code_status.txt", status)):
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write(text)
    return {"measurement_commit": commit, "measurement_source_sha256": blobs,
            "measurement_zip_sha256": sha256(output / "measurement_code.zip")}


def registration(inventory, authorization, device, extracted, code_identity):
    original = inventory["original_protocol"]
    seed_result = read_json(Path(inventory["tasks"][0]["original_dir"]).parents[2] / "seed_result.json")
    body = {
        "format": "r7-n1-evaluation-cost-supplement-v2", "scientific_claim": False,
        "test_read": False, "training_updates": 0, "gpu_seconds_cap": SUPPLEMENT_CAP_SECONDS,
        "whole_wall_seconds_cap": WHOLE_CAP_SECONDS, "authorization": authorization,
        "device_policy": "shared-headroom", "min_free_mib": MIN_FREE_MIB, "evaluations_per_child": 1,
        "probes": {"P1_seconds_cap": 60, "conditional_P2_seconds_cap": 120},
        "original_protocol_sha256": ORIGINAL_PROTOCOL_SHA, "original_code_zip_sha256": ORIGINAL_ZIP_SHA,
        "model_code_sha256": seed_result["model_code_sha256"],
        "original_torch_version": seed_result["torch_version"],
        "original_protocol": original, "val_manifest": original["data"]["val_manifest"],
        "data_identity": original["data"]["data_identity"],
        "source_identity": original["data"]["source_identity"],
        "device": device, "execution_device": "cuda:0", "tasks": inventory["tasks"],
        "input_pins": inventory["input_pins"] + list(inventory["source_pins"].values()),
        "extracted_code": extracted, "code_identity": code_identity,
        "measurement": "one fresh child per seed/arm/lead, 30 children; zero baseline and explicit-device whole-call peaks",
        "budget_scope": ("conservative first-child launch to last-child exit interval; "
                         "imports/setup/cleanup/inter-child gaps included; parent watchdog stops owned child"),
        "replay_rule": "exact RMSE CSV and provenance identities/cases/MSEs; frozen science unchanged",
        "failure_policy": "stop on error, timeout, insufficient headroom, unattributed probe or identity mismatch; retain and charge partial, no retry",
        "limitations": LIMITATIONS,
    }
    return dict(body, protocol_sha256=digest(body))


def wait_owned_child(process, request, deadline, log):
    try:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise RuntimeError("cost deadline exhausted before worker")
        process.communicate(input=json.dumps(request, allow_nan=False), timeout=remaining)
        if process.returncode:
            raise RuntimeError(f"cost worker failed with exit {process.returncode}; see {log}")
    except BaseException:
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("owned child exit unconfirmed after bounded reap; stop and retain evidence") from error
        raise


def run_owned_child(output, code_root, seed, arm, lead, *, deadline, environment):
    if seed not in SEEDS or arm not in ARMS or lead not in LEADS:
        raise ValueError("undeclared cost worker")
    output, code_root = plain_path(output), plain_path(code_root)
    check_deadline(deadline)
    request = {"protocol": str(output / "protocol.json"), "code_root": str(code_root),
               "output": str(output), "seed": seed, "arm": arm, "lead": lead, "deadline": deadline}
    log = output / f"worker_seed{seed}_{arm}_lead{lead:03d}.log"
    launch_id = hashlib.sha256(os.urandom(16)).hexdigest()
    environment = dict(environment, N1_COST_LAUNCH_ID=launch_id)
    write_json(output / f"launch_worker_seed{seed}_{arm}_lead{lead:03d}.json", {
        "kind": f"worker_seed{seed}_{arm}_lead{lead:03d}", "launch_id": launch_id,
        "parent_pid": os.getpid(), "deadline": deadline,
        "protocol_sha256": read_json(output / "protocol.json")["protocol_sha256"]})
    with log.open("x", encoding="utf-8") as stream:
        launched = time.perf_counter()
        process = subprocess.Popen(
            ["./.venv/bin/python", "-B", "measurement/scripts/measure_r7_n1_eval_worker.py"],
            stdin=subprocess.PIPE, stdout=stream, stderr=subprocess.STDOUT,
            env=environment, cwd=code_root, shell=False, text=True)
        try:
            wait_owned_child(process, request, deadline, log)
        finally:
            write_json(output / f"child_seed{seed}_{arm}_lead{lead:03d}.json", {
                "worker_pid": process.pid, "launch_id": launch_id, "seed": seed, "arm": arm, "lead": lead,
                "launched_perf_counter": launched, "finished_perf_counter": time.perf_counter(),
                "returncode": process.poll(), "protocol_sha256": read_json(output / "protocol.json")["protocol_sha256"]})


def run_probe(output, code_root, mode, *, deadline, environment):
    if mode not in ("P1", "P2"):
        raise ValueError("undeclared residue probe")
    cap = 60.0 if mode == "P1" else 120.0
    probe_deadline = min(deadline, time.perf_counter() + cap)
    gpu_headroom("cuda:0", probe_deadline, gpu_uuid=environment["CUDA_VISIBLE_DEVICES"],
                 refusal_path=output / "guard_refusal.json")
    check_deadline(probe_deadline)
    request = {"protocol": str(output / "protocol.json"), "code_root": str(code_root),
               "output": str(output), "deadline": probe_deadline, "mode": mode}
    log = output / f"probe_{mode}.log"
    launch_id = hashlib.sha256(os.urandom(16)).hexdigest()
    environment = dict(environment, N1_COST_LAUNCH_ID=launch_id)
    write_json(output / f"launch_probe_{mode}.json", {"kind": f"probe_{mode}", "launch_id": launch_id,
        "parent_pid": os.getpid(), "deadline": probe_deadline,
        "protocol_sha256": read_json(output / "protocol.json")["protocol_sha256"]})
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(
            ["./.venv/bin/python", "-B", "measurement/scripts/probe_r7_n1_allocator.py"],
            stdin=subprocess.PIPE, stdout=stream, stderr=subprocess.STDOUT,
            env=environment, cwd=code_root, shell=False, text=True)
        try:
            wait_owned_child(process, request, probe_deadline, log)
        finally:
            write_json(output / f"child_probe_{mode}.json", {"worker_pid": process.pid, "mode": mode,
                       "returncode": process.poll(), "finished_perf_counter": time.perf_counter()})
    return read_json(output / ("probe_residue.json" if mode == "P1" else "probe_project_path.json"))


def freeze_wrapper(output, code_root, code_identity):
    target = code_root / "measurement"
    with zipfile.ZipFile(output / "measurement_code.zip") as bundle:
        for name, expected in code_identity["measurement_source_sha256"].items():
            raw = bundle.read(name)
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("measurement wrapper archive differs")
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(raw)
    (code_root / ".venv").symlink_to(ROOT / ".venv", target_is_directory=True)
    return {"root": str(target), "input_repo": str(ROOT)}


def collect_rows(output, protocol):
    rows = []
    for seed, arm, lead in itertools.product(SEEDS, ARMS, LEADS):
        result = read_json(output / f"worker_seed{seed}_{arm}_lead{lead:03d}.json")
        if (result["status"] != "success" or result["protocol_sha256"] != protocol["protocol_sha256"]
                or len(result["rows"]) != 1
                or tuple(result["rows"][0][key] for key in ("seed", "arm", "lead")) != (seed, arm, lead)):
            raise ValueError("worker single-task status/protocol mismatch")
        row = result["rows"][0]
        receipt = read_json(output / f"child_seed{seed}_{arm}_lead{lead:03d}.json")
        spawn = read_json(output / f"spawn_seed{seed}_{arm}_lead{lead:03d}.json")
        kind = f"worker_seed{seed}_{arm}_lead{lead:03d}"
        issued, claimed = read_json(output / f"launch_{kind}.json"), read_json(output / f"claimed_{kind}.json")
        if (claimed != dict(issued, worker_pid=row.get("worker_pid"))
                or issued.get("launch_id") != row.get("launch_id")
                or issued.get("protocol_sha256") != protocol["protocol_sha256"]):
            raise ValueError("worker single-use parent claim evidence differs")
        if (receipt.get("worker_pid") != row.get("worker_pid") or type(row.get("worker_pid")) is not int
                or row["worker_pid"] <= 0 or receipt.get("returncode") != 0
                or not isinstance(row.get("launch_id"), str) or len(row["launch_id"]) != 64
                or receipt.get("launch_id") != row["launch_id"]
                or tuple(receipt.get(key) for key in ("seed", "arm", "lead")) != (seed, arm, lead)
                or receipt.get("protocol_sha256") != protocol["protocol_sha256"]
                or not 0 <= receipt["launched_perf_counter"] < receipt["finished_perf_counter"]
                or spawn.get("gpu_uuid") != protocol["physical_gpu_uuid"] or spawn.get("free_mib", -1) < MIN_FREE_MIB):
            raise ValueError("worker launch/exit/headroom evidence differs")
        rows.extend(result["rows"])
    if len({row["launch_id"] for row in rows}) != 30:
        raise ValueError("exactly thirty distinct process launch identities required")
    validate_measurements(rows, protocol, output)
    return rows


def write_table(path, fields, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_cost_tables(output, rows, inventory):
    original = inventory["original_protocol"]
    archive = Path(inventory["tasks"][0]["original_dir"]).parents[3]
    merged = read_json(archive / "merged_result.json")
    parameters, flops, throughput, memory = [], [], [], []
    for arm in original["arms"]:
        parameters.append({k: arm[k] for k in ("name", "parameters", "trainable_parameters")})
        flops.append({k: arm[k] for k in ("name", "forward_flops", "forward_backward_flops")})
    for seed, arm in itertools.product(SEEDS, ARMS):
        entry = merged["training"][str(seed)][arm]
        throughput.append({"seed": seed, "arm": arm, "updates_run": entry["updates_run"],
                           "elapsed_seconds": entry["elapsed_seconds"],
                           "seconds_per_update": entry["seconds_per_update"]})
        memory.append({"seed": seed, "arm": arm, "phase": "original_training", "lead": "",
                       "n_cases": "", "device": "cuda:1", "baseline_allocated_bytes": "",
                       "baseline_reserved_bytes": "", "peak_allocated_bytes": entry["peak_allocated_bytes"],
                       "peak_reserved_bytes": entry["peak_reserved_bytes"], "elapsed_seconds": entry["elapsed_seconds"]})
    for row in rows:
        memory.append({"seed": row["seed"], "arm": row["arm"], "phase": "supplement_evaluation",
                       "lead": row["lead"], "n_cases": row["n_cases"], "device": row["device"],
                       "baseline_allocated_bytes": row["baseline"]["allocated_bytes"],
                       "baseline_reserved_bytes": row["baseline"]["reserved_bytes"],
                       "peak_allocated_bytes": row["peak_allocated_bytes"],
                       "peak_reserved_bytes": row["peak_reserved_bytes"], "elapsed_seconds": row["elapsed_seconds"]})
    for name, entries in (("parameter_table.csv", parameters), ("flops_table.csv", flops),
                          ("training_throughput_table.csv", throughput), ("memory_table.csv", memory)):
        write_table(output / name, tuple(entries[0]), entries)
    write_json(output / "cost_views.json", {"scientific_claim": False,
        "scope": "original params/FLOPs/training costs referenced; evaluation memory independently measured here",
        "table_sha256": {name: sha256(output / name) for name in
                         ("parameter_table.csv", "flops_table.csv", "training_throughput_table.csv", "memory_table.csv")},
        "limitations": LIMITATIONS})


def run(archive, output, authorization_path, device):
    whole_started = time.perf_counter()
    authorization = read_json(authorization_path)
    verify_authorization(authorization)
    inventory = inspect_archive(archive)
    output = reserve_output(output, archive, Path(inventory["original_protocol"]["data"]["manifests_dir"]))
    status, failure, charged, started = "failed", None, 0.0, None
    try:
        write_json(output / "artifact_inventory.json", inventory)
        code = archive_wrapper(output)
        with TemporaryDirectory(prefix="r7_n1_eval_archive_") as temporary:
            code_root = Path(temporary)
            extracted = extract_code(Path(archive) / "code.zip", code_root)
            frozen_wrapper = freeze_wrapper(output, code_root, code)
            selected = select_shared_gpu(device, whole_started + WHOLE_CAP_SECONDS - 4,
                                         output / "guard_refusal.json")
            selected_device = selected["physical_device"]
            started = time.perf_counter()
            deadline = min(whole_started + WHOLE_CAP_SECONDS - 4, started + SUPPLEMENT_CAP_SECONDS - 4)
            protocol = registration(inventory, authorization, selected_device, extracted, code)
            protocol["output_dir"] = str(output)
            protocol["gpu_phase_started_perf_counter"] = started
            protocol["gpu_phase_deadline_perf_counter"] = deadline
            protocol["whole_started_perf_counter"] = whole_started
            protocol["whole_deadline_perf_counter"] = whole_started + WHOLE_CAP_SECONDS - 4
            protocol["requested_device"] = device
            protocol["physical_gpu_uuid"] = selected["gpu_uuid"]
            protocol["startup_device_query"] = selected
            protocol["wrapper_location"] = frozen_wrapper
            protocol["protocol_sha256"] = digest({k: v for k, v in protocol.items() if k != "protocol_sha256"})
            write_json(output / "protocol.json", protocol)
            verify_registration(output / "protocol.json")
            gpu_headroom(selected_device, whole_started + WHOLE_CAP_SECONDS, gpu_uuid=protocol["physical_gpu_uuid"],
                         refusal_path=output / "guard_refusal.json")
            environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4",
                               CUDA_VISIBLE_DEVICES=protocol["physical_gpu_uuid"])
            environment.pop("PYTHONPATH", None)
            probe = run_probe(output, code_root, "P1", deadline=deadline, environment=environment)
            verify_probe_report(probe, protocol, "P1", output)
            if probe["requires_p2"]:
                probe = run_probe(output, code_root, "P2", deadline=deadline, environment=environment)
                verify_probe_report(probe, protocol, "P2", output)
            if probe["attribution_confirmed"] is not True:
                raise RuntimeError("residue probe cannot attribute the original failure; stop without evaluations")
            for seed, arm, lead in itertools.product(SEEDS, ARMS, LEADS):
                check_deadline(deadline)
                observed = gpu_headroom(selected_device, deadline, gpu_uuid=protocol["physical_gpu_uuid"],
                                        refusal_path=output / "guard_refusal.json")
                write_json(output / f"spawn_seed{seed}_{arm}_lead{lead:03d}.json", observed)
                run_owned_child(output, code_root, seed, arm, lead, deadline=deadline, environment=environment)
            gpu_headroom(selected_device, deadline, gpu_uuid=protocol["physical_gpu_uuid"],
                         refusal_path=output / "guard_refusal.json")
            charged = time.perf_counter() - started
            started = None
            if charged > SUPPLEMENT_CAP_SECONDS:
                raise RuntimeError("measured supplement GPU interval exceeded cap")
            rows = collect_rows(output, protocol)
            verify_pins(protocol["input_pins"])
            verify_pins([{"path": str(ROOT / name), "sha256": expected} for name, expected in
                         code["measurement_source_sha256"].items()])
            write_cost_tables(output, rows, inventory)
            check_deadline(whole_started + WHOLE_CAP_SECONDS)
            write_json(output / "result.json", {"status": "success", "scientific_claim": False,
                "test_read": False, "protocol_sha256": protocol["protocol_sha256"], "rows": rows,
                "gpu_hours_charged": charged / 3600,
                "probe_sha256": {name: sha256(output / name) for name in
                                 ("probe_residue.json", "probe_project_path.json") if (output / name).is_file()},
                "cost_views_sha256": sha256(output / "cost_views.json"), "limitations": LIMITATIONS})
            status = "success"
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        if started is not None:
            charged = time.perf_counter() - started
        unconfirmed = [path.name for path in output.glob("child_*.json") if read_json(path)["returncode"] is None]
        charged_seconds = max(charged, SUPPLEMENT_CAP_SECONDS) if unconfirmed else charged
        write_json(output / "attempt.json", {"status": status, "failure_reason": failure,
            "scientific_claim": False, "test_read": False, "gpu_phase_seconds": charged,
            "gpu_hours_charged": charged_seconds / 3600, "cap_gpu_seconds": SUPPLEMENT_CAP_SECONDS,
            "owned_child_exit_unconfirmed": unconfirmed,
            "charge_scope": "full authorization cap; cleanup incomplete, actual upper bound unverified" if unconfirmed else "full observed GPU interval",
            "whole_wall_seconds": time.perf_counter() - whole_started,
            "cap_whole_wall_seconds": WHOLE_CAP_SECONDS, "device_policy": "shared-headroom",
            "training_updates": 0, "completed_evaluations": len(list(output.glob("worker_seed*_lead*.json"))),
            "limitations": LIMITATIONS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "outputs/r7_72_frozen_z")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/r7_n1_eval_cost_supplement_v2")
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    deny_network()
    run(args.archive, args.out, args.authorization, args.device)


if __name__ == "__main__":
    main()
