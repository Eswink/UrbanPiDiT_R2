"""Offline, bounded N1 allocator diagnostics; never accepted evaluation-cost rows."""
from __future__ import annotations

import gc
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace

MAX_PROBE_SECONDS = 60.0
MAX_P2_SECONDS = 120.0
PROBE_ARM = "process_spacetime_rwa"
REQUIRED_WRAPPER_FILES = ("scripts/probe_r7_n1_allocator.py", "training/r7_n1_cost_replay.py")


def deny_network():
    def refused(*args, **kwargs):
        raise RuntimeError("N1 allocator probe is offline")
    socket.socket.connect = refused
    socket.create_connection = refused


def load_support(path):
    spec = importlib.util.spec_from_file_location("n1_cost_support", path)
    if spec is None or spec.loader is None:
        raise ValueError("frozen allocator support cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plain_path(value):
    path = Path(value).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("symlink probe input/output is not allowed")
    return path.resolve()


def _check_probe_deadline(deadline, mode="P1"):
    if (type(deadline) not in (int, float) or not math.isfinite(deadline)):
        raise ValueError("probe deadline must be a finite perf_counter number")
    if mode not in ("P1", "P2"):
        raise ValueError("allocator probe mode must be P1 or P2")
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise RuntimeError("probe deadline exhausted")
    cap = MAX_P2_SECONDS if mode == "P2" else MAX_PROBE_SECONDS
    if remaining > cap:
        raise ValueError(f"{mode} probe child deadline exceeds {cap:g} seconds")


def parse_request(value):
    if not isinstance(value, dict) or set(value) != {"protocol", "code_root", "output", "deadline", "mode"}:
        raise ValueError("allocator probe request fields differ")
    if value["mode"] not in ("P1", "P2"):
        raise ValueError("allocator probe mode must be P1 or P2")
    _check_probe_deadline(value["deadline"], value["mode"])
    if any(not isinstance(value[name], str) or not value[name] for name in ("protocol", "code_root", "output")):
        raise ValueError("probe paths must be nonempty strings")
    output, protocol, code_root = (_plain_path(value[name]) for name in ("output", "protocol", "code_root"))
    if protocol != output / "protocol.json":
        raise ValueError("probe protocol must equal output/protocol.json")
    metadata = json.loads(protocol.read_text(encoding="utf-8"))
    input_root = _plain_path(metadata["wrapper_location"]["input_repo"])
    if output == input_root / "outputs" or not output.is_relative_to(input_root / "outputs"):
        raise ValueError("probe output outside input repository outputs")
    wrapper_root = Path(__file__).resolve().parents[1]
    return SimpleNamespace(protocol=protocol, code_root=code_root, output=output,
                           deadline=value["deadline"], mode=value["mode"],
                           support=wrapper_root / "training/r7_n1_cost_replay.py")


def probe_tasks(protocol):
    tasks = protocol["tasks"][:2]
    if [(t["seed"], t["arm"], t["lead"]) for t in tasks] != [(41, PROBE_ARM, 6), (41, PROBE_ARM, 12)]:
        raise ValueError("P2 requires the first two original seed41 spacetime evaluations")
    return tasks


def verify_runtime(args, support):
    support.check_deadline(args.deadline)
    protocol = support.verify_registration(args.protocol)
    code_root = support.plain_path(args.code_root)
    wrapper_root = support.plain_path(protocol["wrapper_location"]["root"])
    input_root = support.plain_path(protocol["wrapper_location"]["input_repo"])
    if (wrapper_root != code_root / "measurement"
            or Path(__file__).resolve() != wrapper_root / REQUIRED_WRAPPER_FILES[0]
            or support.plain_path(args.support) != wrapper_root / REQUIRED_WRAPPER_FILES[1]
            or support.plain_path(support.__file__) != wrapper_root / REQUIRED_WRAPPER_FILES[1]
            or support.plain_path(args.output) == input_root / "outputs"
            or not support.plain_path(args.output).is_relative_to(input_root / "outputs")
            or support.plain_path(args.protocol) != support.plain_path(args.output) / "protocol.json"):
        raise ValueError("probe must execute its frozen measurement wrapper/output scope")
    hashes = protocol["code_identity"]["measurement_source_sha256"]
    if not set(REQUIRED_WRAPPER_FILES).issubset(hashes):
        raise ValueError("probe/support frozen wrapper hashes missing")
    for name, expected in hashes.items():
        support.pinned_file(wrapper_root / name, expected, wrapper_root)
    required = {"training/r7_evaluate.py", "training/r7_experiment.py", "model/__init__.py"}
    if not required.issubset({entry["path"] for entry in protocol["extracted_code"]}):
        raise ValueError("archived evaluator/model identities missing")
    for entry in protocol["extracted_code"]:
        support.pinned_file(code_root / entry["path"], entry["sha256"], code_root)
    support.verify_pins(protocol["input_pins"])
    for task in probe_tasks(protocol):
        support.verify_task_inputs(task, protocol)
    uuid = protocol["physical_gpu_uuid"]
    if (not isinstance(uuid, str) or not uuid.startswith("GPU-") or "," in uuid
            or os.environ.get("CUDA_VISIBLE_DEVICES") != uuid or protocol["execution_device"] != "cuda:0"):
        raise ValueError("probe requires one physical UUID mapped to cuda:0")
    support.check_deadline(args.deadline)
    return protocol


def verify_import_origins(protocol, code_root, support):
    pins = {support.plain_path(code_root / entry["path"]): entry["sha256"]
            for entry in protocol["extracted_code"]}
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] not in ("training", "model", "data"):
            continue
        origin = getattr(module, "__file__", None)
        if origin is None or support.plain_path(origin) not in pins:
            raise ValueError(f"probe imported non-archived project module: {name}")
        support.pinned_file(origin, pins[support.plain_path(origin)], code_root)


def load_archived_evaluator(protocol, code_root, support):
    verify_import_origins(protocol, code_root, support)
    sys.path.insert(0, str(code_root))
    importlib.invalidate_caches()
    evaluator = importlib.import_module("training.r7_evaluate")
    experiment = importlib.import_module("training.r7_experiment")
    if experiment.model_code_digest() != protocol["model_code_sha256"]:
        raise ValueError("probe model digest differs from archived model")
    verify_import_origins(protocol, code_root, support)
    return evaluator.evaluate_local


def _validate_allocator(record):
    if (not isinstance(record, dict) or type(record.get("allocated_bytes")) is not int
            or type(record.get("reserved_bytes")) is not int
            or not 0 <= record["allocated_bytes"] <= record["reserved_bytes"]):
        raise ValueError("invalid exact allocator bytes record")
    snapshot = record.get("memory_snapshot")
    required = {"sha256", "segments", "total_bytes", "block_state_bytes", "raw_segments"}
    if (not isinstance(snapshot, dict) or "error" in snapshot or not required.issubset(snapshot)
            or not isinstance(snapshot["sha256"], str) or len(snapshot["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in snapshot["sha256"])
            or type(snapshot["segments"]) is not int or snapshot["segments"] < 0
            or type(snapshot["total_bytes"]) is not int or snapshot["total_bytes"] < 0
            or not isinstance(snapshot["block_state_bytes"], dict)
            or not isinstance(snapshot["raw_segments"], list)
            or snapshot["segments"] != len(snapshot["raw_segments"])):
        raise ValueError("memory_snapshot unavailable/invalid; unattributed, stop")
    return record


def _has_residue(record):
    _validate_allocator(record)
    return record["allocated_bytes"] != 0 or record["reserved_bytes"] != 0


def cleanup_record(record, key, cuda, device, support, deadline):
    support.check_deadline(deadline)
    gc.collect()
    cuda.synchronize(device)
    cuda.empty_cache()
    cuda.synchronize(device)
    record[key] = support.allocator_record(cuda, device)
    _validate_allocator(record[key])
    support.check_deadline(deadline)
    return record[key]


def _initialize(record, torch, support, protocol, output, deadline):
    cuda, device = torch.cuda, torch.device(protocol["execution_device"])
    support.check_deadline(deadline)
    support.gpu_headroom(protocol["execution_device"], deadline, gpu_uuid=protocol["physical_gpu_uuid"],
                         refusal_path=output / "guard_refusal.json")
    cuda.init()
    cuda.set_device(device)
    support.verify_cuda_device(cuda, device, protocol["physical_gpu_uuid"])
    try:
        support.reset_measurement(cuda, device, refusal_path=output / "guard_refusal.json")
    finally:
        # Preserve the initial bytes even if the shared zero-baseline guard refuses.
        record["initial"] = support.allocator_record(cuda, device)
        _validate_allocator(record["initial"])
    support.check_deadline(deadline)
    if _has_residue(record["initial"]):
        raise RuntimeError("probe requires a fresh zero allocated/reserved initial baseline")
    return cuda, device


def _base_record(mode, torch, protocol, deadline):
    _check_probe_deadline(deadline, mode)
    if str(torch.__version__) != protocol["original_torch_version"]:
        raise ValueError("probe torch environment differs from original exact version")
    return {"format": "r7-n1-allocator-probe-v1", "mode": mode, "status": "failed",
            "classification": "unattributed", "attribution_confirmed": False,
            "scientific_claim": False, "training_updates": 0, "test_read": False,
            "accepted_cost_rows": False, "torch_version": str(torch.__version__),
            "gpu_uuid": protocol["physical_gpu_uuid"], "execution_device": protocol["execution_device"],
            "protocol_sha256": protocol["protocol_sha256"], "deadline_perf_counter": deadline,
            "cap_seconds": MAX_P2_SECONDS if mode == "P2" else MAX_PROBE_SECONDS,
            "started_perf_counter": time.perf_counter(), "initial": None, "failure_reason": None,
            "limitations": list(protocol["limitations"]) + [
                "Allocator-only diagnostic; not a scientific claim or an accepted-cost measurement.",
                "A workspace-family observation cannot identify a specific library or tensor holder.",
                "Private release API is version-specific and never used in accepted evaluations.",
                "Parent watchdog owns the absolute deadline; forced termination may leave partial evidence.",
            ]}


def _finish_record(path, record, support):
    record["finished_perf_counter"] = time.perf_counter()
    record["elapsed_seconds"] = record["finished_perf_counter"] - record["started_perf_counter"]
    support.write_json(path, record)


def probe_p1(torch, support, protocol, output, deadline):
    output = Path(output)
    path = output / "probe_residue.json"
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    record = _base_record("P1", torch, protocol, deadline)
    record.update(before_clear=None, after_clear=None, requires_p2=False,
                  private_api={"name": "torch._C._cuda_clearCublasWorkspaces", "available": False, "called": False},
                  matmul={"shape": [1024, 1024], "dtype": "float32", "repetitions": 1})
    record["limitations"].append("P1 imports no project evaluator/model and performs only one pure-torch matmul.")
    try:
        cuda, device = _initialize(record, torch, support, protocol, output, deadline)
        support.check_deadline(deadline)
        left = right = product = None
        started = time.perf_counter()
        try:
            left = torch.ones((1024, 1024), dtype=torch.float32, device=device)
            right = torch.ones((1024, 1024), dtype=torch.float32, device=device)
            product = torch.matmul(left, right)
        finally:
            del left, right, product
            record["matmul_call_seconds"] = time.perf_counter() - started
            cleanup_record(record, "before_clear", cuda, device, support, deadline)
        clear = getattr(getattr(torch, "_C", None), "_cuda_clearCublasWorkspaces", None)
        record["private_api"]["available"] = callable(clear)
        try:
            if callable(clear):
                support.check_deadline(deadline)
                record["private_api"]["called"] = True
                clear()
            elif _has_residue(record["before_clear"]):
                raise RuntimeError("private workspace API unavailable with residue; unattributed, stop")
            else:
                record["limitations"].append("Private workspace API unavailable; no release was performed.")
        finally:
            cleanup_record(record, "after_clear", cuda, device, support, deadline)
        before, after = _has_residue(record["before_clear"]), _has_residue(record["after_clear"])
        if before:
            record["classification"] = "torch-process-residue" if after else "torch-releasable-workspace-family"
            record["attribution_confirmed"] = True
        elif after:
            raise RuntimeError("residue appeared only after clear; unattributed, stop")
        else:
            record["classification"] = "needs-project-probe"
            record["requires_p2"] = True
        support.check_deadline(deadline)
        record["status"] = "success"
    except BaseException as error:
        record.update(status="failed", attribution_confirmed=False, requires_p2=False,
                      classification="unattributed", failure_reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        _finish_record(path, record, support)
    return record


def verify_p1_gate(output, protocol, support):
    path = _plain_path(Path(output) / "probe_residue.json")
    prior = json.loads(path.read_text(encoding="utf-8"))
    if (prior.get("format") != "r7-n1-allocator-probe-v1" or prior.get("mode") != "P1"
            or prior.get("status") != "success" or prior.get("classification") != "needs-project-probe"
            or prior.get("requires_p2") is not True or prior.get("attribution_confirmed") is not False
            or prior.get("scientific_claim") is not False or prior.get("test_read") is not False
            or type(prior.get("training_updates")) is not int or prior["training_updates"] != 0
            or prior.get("protocol_sha256") != protocol["protocol_sha256"]
            or prior.get("torch_version") != protocol["original_torch_version"]
            or prior.get("gpu_uuid") != protocol["physical_gpu_uuid"]
            or prior.get("execution_device") != protocol["execution_device"]
            or type(prior.get("elapsed_seconds")) not in (int, float)
            or not math.isfinite(prior["elapsed_seconds"]) or not 0 <= prior["elapsed_seconds"] <= MAX_PROBE_SECONDS):
        raise ValueError("P2 requires a successful, matching zero-residue pure-torch P1 proof")
    if any(_has_residue(prior.get(key)) for key in ("initial", "before_clear", "after_clear")):
        raise ValueError("P2 forbidden when pure-torch P1 has residue")
    return support.sha256(path)


def probe_p2(torch, support, protocol, output, deadline, evaluate=None, *, load_evaluate=None, verify_origins=None):
    output = Path(output)
    path = output / "probe_project_path.json"
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    p1_sha = verify_p1_gate(output, protocol, support)
    tasks = probe_tasks(protocol)
    record = _base_record("P2", torch, protocol, deadline)
    record.update(p1_probe_sha256=p1_sha, evaluations=[])
    record["limitations"].append("P2 reuses one process and frozen validation evaluations, including training climatology; zero optimizer updates, no held-out test reads.")
    try:
        cuda, device = _initialize(record, torch, support, protocol, output, deadline)
        if evaluate is None:
            if load_evaluate is None:
                raise ValueError("P2 requires the archived evaluator loader")
            evaluate = load_evaluate()
        if verify_origins is not None:
            verify_origins()
        for task in tasks:
            support.check_deadline(deadline)
            support.verify_task_inputs(task, protocol)
            support.gpu_headroom(protocol["execution_device"], deadline, gpu_uuid=protocol["physical_gpu_uuid"],
                                 refusal_path=output / "guard_refusal.json")
            directory = _plain_path(output / "probe_p2" / "seed41" / PROBE_ARM / f"lead_{task['lead']:03d}h")
            if directory.exists() or directory.is_symlink():
                raise FileExistsError(directory)
            observation = {"seed": task["seed"], "arm": task["arm"], "lead": task["lead"],
                           "evaluation_dir": str(directory), "original_checkpoint_sha256": task["checkpoint"]["sha256"],
                           "after_evaluation": None, "replay": None}
            record["evaluations"].append(observation)
            started = time.perf_counter()
            try:
                report = evaluate(protocol["val_manifest"], output_dir=directory,
                                  checkpoint=task["checkpoint"]["path"], lead_hours=(task["lead"],),
                                  max_samples=32, device_name=protocol["execution_device"],
                                  reasoning_steps=3, deadline=deadline)
            finally:
                observation["evaluation_call_seconds"] = time.perf_counter() - started
                cleanup_record(observation, "after_evaluation", cuda, device, support, deadline)
            observation["replay"] = support.compare_evaluation(task, directory, report)
            if observation["replay"].get("exact_numeric_replay") is not True:
                raise ValueError("P2 requires exact original evaluation replay")
            if verify_origins is not None:
                verify_origins()
            support.check_deadline(deadline)
        if not any(_has_residue(item["after_evaluation"]) for item in record["evaluations"]):
            record["classification"] = "unexplained-original-failure"
            raise RuntimeError("P2 has zero residue; unexplained-original-failure, parent must stop")
        record.update(classification="project-path-residue", attribution_confirmed=True)
        support.check_deadline(deadline)
        record["status"] = "success"
    except BaseException as error:
        record.update(status="failed", attribution_confirmed=False, failure_reason=f"{type(error).__name__}: {error}")
        if record["classification"] != "unexplained-original-failure":
            record["classification"] = "unattributed"
        raise
    finally:
        _finish_record(path, record, support)
    return record


def run(args):
    _check_probe_deadline(args.deadline, args.mode)
    if args.mode not in ("P1", "P2"):
        raise ValueError("allocator probe mode must be P1 or P2")
    support = load_support(args.support)
    protocol = verify_runtime(args, support)
    if args.mode == "P2":
        verify_p1_gate(args.output, protocol, support)
    support.claim_parent_launch(protocol, args.output, "probe_" + args.mode, args.deadline)
    import torch
    if str(torch.__version__) != protocol["original_torch_version"]:
        raise ValueError("probe torch environment differs from original exact version")
    torch.set_num_threads(4)
    if args.mode == "P1":
        return probe_p1(torch, support, protocol, args.output, args.deadline)
    return probe_p2(torch, support, protocol, args.output, args.deadline,
                    load_evaluate=lambda: load_archived_evaluator(protocol, args.code_root, support),
                    verify_origins=lambda: verify_import_origins(protocol, args.code_root, support))


def main():
    deny_network()
    run(parse_request(json.load(sys.stdin)))


if __name__ == "__main__":
    main()
