"""Run archived N1 evaluation code with independent, explicit-device CUDA peaks."""
from __future__ import annotations

import importlib.util
import json
import math
import os
from types import SimpleNamespace
from pathlib import Path
import socket
import sys
import time


def deny_network():
    def refused(*args, **kwargs):
        raise RuntimeError("N1 cost replay is offline")
    socket.socket.connect = refused
    socket.create_connection = refused


def load_support(path):
    spec = importlib.util.spec_from_file_location("n1_cost_support", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def failed_call_evidence(directory, device, protocol, torch, support, deadline, error):
    record = {"scientific_claim": False, "failure_reason": f"{type(error).__name__}: {error}"}
    try:
        record["device_after"] = support.gpu_headroom(device, deadline, gpu_uuid=protocol["physical_gpu_uuid"],
                                                     refusal_path=directory.parent / "guard_refusal.json")
    except Exception as query_error:
        record["query_error"] = f"{type(query_error).__name__}: {query_error}"
    if time.perf_counter() < deadline:
        record["allocator"] = support.allocator_record(torch.cuda, torch.device(device))
    else:
        record["allocator_unavailable"] = "deadline exhausted; no further CUDA diagnostic"
    support.write_json(directory.parent / f"{directory.name}_failed_call.json", record)


def measure_task(task, protocol, directory, evaluate, torch, support, deadline):
    support.check_deadline(deadline)
    support.verify_task_inputs(task, protocol)
    execution_device = protocol["execution_device"]
    refusal_path = directory.parent / "guard_refusal.json"
    device_before = support.gpu_headroom(execution_device, deadline, gpu_uuid=protocol["physical_gpu_uuid"],
                                         refusal_path=refusal_path)
    device = torch.device(execution_device)
    support.verify_cuda_device(torch.cuda, device, protocol["physical_gpu_uuid"])
    baseline = support.reset_measurement(torch.cuda, device, refusal_path=refusal_path)
    directory.parent.mkdir(parents=True, exist_ok=True)
    support.write_json(directory.parent / f"{directory.name}_device_before.json", device_before)
    started = time.perf_counter()
    try:
        report = evaluate(
            protocol["val_manifest"], output_dir=directory,
            checkpoint=task["checkpoint"]["path"], lead_hours=(task["lead"],),
            max_samples=32, device_name=execution_device, reasoning_steps=3, deadline=deadline)
        peaks = support.read_peaks(torch.cuda, device)
        elapsed = time.perf_counter() - started
        device_after = support.gpu_headroom(execution_device, deadline, gpu_uuid=protocol["physical_gpu_uuid"],
                                            refusal_path=refusal_path)
    except BaseException as error:
        try:
            failed_call_evidence(directory, execution_device, protocol, torch, support, deadline, error)
        except Exception as diagnostic_error:
            print(f"failure diagnostic unavailable: {type(diagnostic_error).__name__}: {diagnostic_error}", flush=True)
        raise
    support.write_json(directory.parent / f"{directory.name}_device_after.json", device_after)
    replay = support.compare_evaluation(task, directory, report)
    support.check_deadline(deadline)
    row = {"seed": task["seed"], "arm": task["arm"], "lead": task["lead"],
           "n_cases": report["n_evaluated"], "device": protocol["device"], "execution_device": execution_device,
           "baseline": baseline, **peaks, "elapsed_seconds": elapsed, "replay": replay,
           "original_checkpoint_sha256": task["checkpoint"]["sha256"],
           "evaluation_dir": str(directory), "gpu_uuid": protocol["physical_gpu_uuid"],
           "evaluation_artifacts_sha256": support.evaluation_artifacts(directory),
           "protocol_sha256": protocol["protocol_sha256"], "worker_pid": os.getpid(),
           "launch_id": os.environ["N1_COST_LAUNCH_ID"],
           "device_before": device_before, "device_after": device_after,
           "measurement_scope": "whole archived evaluate_local call, including model loading, IO and metrics; not isolated model latency"}
    support.write_json(directory / "cost_measurement.json", row)
    return row


def run(args):
    support = load_support(args.support)
    protocol = support.verify_registration(args.protocol)
    support.verify_pins(protocol["input_pins"])
    code_root = support.plain_path(args.code_root)
    wrapper_root = support.plain_path(protocol["wrapper_location"]["root"])
    if not Path(__file__).resolve().is_relative_to(wrapper_root):
        raise ValueError("worker must execute frozen measurement wrapper")
    for name, expected in protocol["code_identity"]["measurement_source_sha256"].items():
        support.pinned_file(wrapper_root / name, expected, wrapper_root)
    for entry in protocol["extracted_code"]:
        support.pinned_file(code_root / entry["path"], entry["sha256"], code_root)
    support.claim_parent_launch(protocol, args.output,
                               f"worker_seed{args.seed}_{args.arm}_lead{args.lead:03d}", args.deadline)
    sys.path.insert(0, str(code_root))
    import torch
    from training.r7_evaluate import evaluate_local
    from training.r7_experiment import model_code_digest
    import training.r7_evaluate as evaluator
    if (not Path(evaluator.__file__).resolve().is_relative_to(code_root)
            or model_code_digest() != protocol["model_code_sha256"]):
        raise ValueError("worker must use the exact archived evaluator/model")
    if str(torch.__version__) != protocol["original_torch_version"]:
        raise ValueError("cost supplement torch environment differs from original")
    torch.set_num_threads(4)
    tasks = [task for task in protocol["tasks"]
             if (task["seed"], task["arm"], task["lead"]) == (args.seed, args.arm, args.lead)]
    if len(tasks) != 1:
        raise ValueError("worker requires exactly one original evaluation")
    task = tasks[0]
    directory = args.output / f"seed{args.seed}" / args.arm / f"lead_{args.lead:03d}h"
    rows = [measure_task(task, protocol, directory, evaluate_local, torch, support, args.deadline)]
    print(f"measured seed={args.seed} arm={args.arm} lead={args.lead}", flush=True)
    torch.cuda.synchronize(torch.device(protocol["execution_device"]))
    support.write_json(args.output / f"worker_seed{args.seed}_{args.arm}_lead{args.lead:03d}.json", {
        "status": "success", "protocol_sha256": protocol["protocol_sha256"],
        "scientific_claim": False, "test_read": False, "rows": rows,
        "limitations": protocol["limitations"],
    })


def parse_request(value):
    if set(value) != {"protocol", "code_root", "output", "seed", "arm", "lead", "deadline"}:
        raise ValueError("cost worker request fields differ")
    wrapper_root = Path(__file__).resolve().parents[1]
    output = Path(value["output"]).resolve()
    protocol = Path(value["protocol"]).resolve()
    metadata = json.loads(protocol.read_text(encoding="utf-8"))
    input_root = Path(metadata["wrapper_location"]["input_repo"]).resolve()
    if (not output.is_relative_to(input_root / "outputs") or protocol != output / "protocol.json"
            or type(value["seed"]) is not int or value["seed"] not in (41, 42)
            or value["arm"] not in ("process_spacetime_rwa", "process_local_solver", "process_local_solver_frozen_z")
            or type(value["lead"]) is not int or value["lead"] not in (6, 12, 24, 48, 72)
            or isinstance(value["deadline"], bool) or not math.isfinite(value["deadline"])):
        raise ValueError("cost worker request outside its named scope")
    return SimpleNamespace(protocol=protocol, code_root=Path(value["code_root"]),
                           output=output, seed=value["seed"], arm=value["arm"], lead=value["lead"],
                           deadline=value["deadline"], support=wrapper_root / "training/r7_n1_cost_replay.py")


def main():
    deny_network()
    run(parse_request(json.load(sys.stdin)))


if __name__ == "__main__":
    main()
