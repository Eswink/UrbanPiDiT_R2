"""Stdlib-only bootstrap for archive-isolated CPU identity checks and single-lead CUDA evaluation."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import time


def deny_network():
    def refused(*args, **kwargs):
        raise RuntimeError("M3 complement is offline: outbound connections forbidden")
    socket.socket.connect = refused
    socket.create_connection = refused


def check_deadline(deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError("M3 complement owned worker deadline exhausted")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256_file(path):
    path = Path(path).absolute()
    if path.name == "test.jsonl" or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("sealed test/symlink paths forbidden")
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    path = Path(path).absolute()
    if path.name == "test.jsonl" or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("sealed test/symlink artifacts forbidden")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("symlink result paths forbidden")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def preserve_error(original, additional):
    if original is None:
        raise additional
    original.add_note(f"complement worker publication failed: {type(additional).__name__}: {additional}")
    try:
        print(original.__notes__[-1], file=sys.stderr, flush=True)
    except BaseException as exc:
        original.add_note(f"secondary diagnostic failed: {type(exc).__name__}: {exc}")


def isolate_archive(root, source_root):
    """No current local packages, PYTHONPATH or script directory; keep venv dependencies."""
    root, source_root = Path(root).resolve(), Path(source_root).resolve()
    if root.is_symlink() or Path.cwd().resolve() != root:
        raise ValueError("worker cwd must be the independently extracted archived_code root")
    local = ("training", "model", "data", "scripts")
    if any(name.split(".")[0] in local for name in sys.modules):
        raise ValueError("local packages were imported before archive isolation")
    entries = []
    environment_root = Path(sys.prefix).resolve()
    for entry in sys.path:
        if not entry:
            continue
        path = Path(entry).resolve()
        in_environment = environment_root != source_root and path.is_relative_to(environment_root)
        if path == root or (path.is_relative_to(source_root) and not in_environment):
            continue
        if any((path / name).exists() for name in local) and not in_environment:
            continue
        entries.append(str(path))
    sys.path[:] = [str(root), *entries]
    importlib.invalidate_caches()
    return root


def assert_archive_origins(root):
    root = Path(root).resolve()
    if not sys.path or Path(sys.path[0]).resolve() != root or Path.cwd().resolve() != root:
        raise ValueError("archived cwd/sys.path isolation lost")
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] not in ("training", "model", "data", "scripts"):
            continue
        file = getattr(module, "__file__", None)
        paths = list(getattr(module, "__path__", ()))
        if file is not None:
            paths.append(file)
        if not paths or any(not Path(path).resolve().is_relative_to(root) for path in paths):
            raise ValueError(f"local import escaped pinned archived root: {name}")


def read_complement(protocol_path):
    protocol = read_json(protocol_path)
    body = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    output = Path(protocol["output"])
    old = Path(protocol["original_output"])
    if (protocol.get("protocol_sha256") != digest(body)
            or protocol.get("format") != "r7-m3-validation-complement-protocol-v1"
            or protocol.get("training_updates") != 0 or protocol.get("test_read") is not False
            or protocol.get("scientific_claim") is not False or not protocol.get("limitations")
            or not output.is_absolute() or not old.is_absolute()
            or output.resolve() != Path(protocol_path).resolve().parent
            or old.resolve().is_relative_to(output.resolve()) or output.resolve().is_relative_to(old.resolve())
            or protocol.get("planned_seconds") != 1800 or protocol.get("hard_cap_seconds") != 3600
            or protocol.get("cleanup_reserve_seconds") != 10):
        raise ValueError("independent complement protocol/output/budget invalid")
    return protocol


def verify_archived_sources(protocol, root, *, deadline):
    """Only pinned extracted members; never accept partial extraction or current source fallback."""
    old = Path(protocol["original_output"])
    original = read_json(old / "protocol.json")
    if (sha256_file(old / "protocol.json") != protocol["original_file_sha256"]["protocol.json"]
            or original["protocol_sha256"] != protocol["original_protocol_sha256"]
            or original["code"]["code_zip_sha256"] != protocol["original_code_zip_sha256"]
            or sha256_file(old / "code.zip") != protocol["original_code_zip_sha256"]
            or digest(original["code"]["files"]) != original["code"]["source_tree_sha256"]):
        raise ValueError("original frozen archive/protocol identity changed")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("extracted archive symlink forbidden")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(original["code"]["files"]) or len(actual) != 80:
        raise ValueError("exact 80-member extracted archive required; no partial repair")
    for name, expected in original["code"]["files"].items():
        check_deadline(deadline)
        member = Path(name)
        if member.is_absolute() or ".." in member.parts or "\\" in name:
            raise ValueError("invalid archived source member")
        if sha256_file(root / member) != expected:
            raise ValueError(f"archived source pin changed: {name}")
    check_deadline(deadline)
    return original


def import_archived(root):
    from training import r7_m3_protocol, r7_m3_identity, r7_m3_results, r7_m3_worker, r7_experiment
    from training.r7_evaluate import evaluate_local
    import torch
    assert_archive_origins(root)
    return {"protocol": r7_m3_protocol, "identity": r7_m3_identity, "results": r7_m3_results,
            "worker": r7_m3_worker, "experiment": r7_experiment,
            "evaluate_local": evaluate_local, "torch": torch}


def training_receipt(protocol, original, modules, seed, arm, *, deadline):
    check_deadline(deadline)
    old = Path(protocol["original_output"])
    job = {"phase": "train", "seed": seed, "arm": arm, "lead": None}
    key = modules["protocol"].job_key(job)
    path = old / "workers" / (key + ".json")
    entry = modules["protocol"].read_json(path)
    if sha256_file(path) != protocol["original_file_sha256"]["workers/" + key + ".json"]:
        raise ValueError("original training receipt bytes changed")
    contract = modules["protocol"].expected_training_contract(original, arm)
    if (entry.get("status") != "success" or entry.get("job") != job
            or entry.get("protocol_sha256") != original["protocol_sha256"]
            or entry.get("model_code_sha256") != original["code"]["model_code_sha256"]
            or entry.get("data_identity") != original["data"]["data_identity"]
            or entry.get("process_supervision") != contract or entry.get("updates_run") != 400
            or entry.get("selected_update") != 400 or entry.get("early_stopped") is not False
            or entry.get("scientific_claim") is not False or entry.get("test_read") is not False
            or not entry.get("limitations")):
        raise ValueError("original selected400 training qualification failed")
    modules["results"]._verify_training(entry, job, original)
    report = modules["protocol"].read_json(entry["training_report"])
    saved = modules["experiment"].load_checkpoint(entry["checkpoint"], expected=report["signature"])
    expected = {"kind": "process", "model": modules["protocol"].model_config(17),
                "data_identity": original["data"]["data_identity"], "process_supervision": contract,
                "seed": seed, "steps": 4, "batch_size": 2, "bf16": False, "process_weight": 0.0}
    if (saved["updates"] != 400 or saved["contract"] != report["contract"]
            or any(saved["contract"].get(key) != value for key, value in expected.items())
            or sha256_file(entry["checkpoint"]) != entry["checkpoint_sha256"]):
        raise ValueError("archived checkpoint signature/model/fixed inverse/training contract mismatch")
    check_deadline(deadline)
    return entry, sha256_file(path)


def archived_preflight(protocol, root, deadline, job=None):
    modules = import_archived(root)
    modules["torch"].set_num_threads(4)
    original = modules["protocol"].verify_protocol(Path(protocol["original_output"]) / "protocol.json")
    if original["protocol_sha256"] != protocol["original_protocol_sha256"]:
        raise ValueError("archived verifier original protocol digest differs")
    modules["identity"].verify_code(original)
    check_deadline(deadline)
    modules["identity"].verify_input_pins(original)
    check_deadline(deadline)
    old = Path(protocol["original_output"])
    jobs = modules["protocol"].planned_jobs()
    if protocol["jobs"] != jobs[13:]:
        raise ValueError("only the exact missing23 archived validation jobs are permitted")
    for filename in ("attempt.json", "execution_attempt.json"):
        history = modules["protocol"].read_json(old / filename)
        if history["status"] != "failed" or history["jobs_completed"] != jobs[:13]:
            raise ValueError("original failed terminal history must stay unchanged")
    if job is not None:
        if job not in jobs[13:]:
            raise ValueError("undeclared complement job")
        assert_archive_origins(root)
        return original, modules
    training = {}
    for job in jobs[:6]:
        entry, _ = training_receipt(protocol, original, modules, job["seed"], job["arm"], deadline=deadline)
        training[(job["seed"], job["arm"])] = entry
    for job in jobs[6:13]:
        entry = modules["protocol"].read_json(old / "workers" / (modules["protocol"].job_key(job) + ".json"))
        if (entry.get("job") != job or entry.get("status") != "success"
                or entry.get("protocol_sha256") != original["protocol_sha256"]
                or entry.get("model_code_sha256") != original["code"]["model_code_sha256"]
                or entry.get("process_supervision") != modules["protocol"].expected_training_contract(original, job["arm"])):
            raise ValueError("old successful validation receipt identity mismatch")
        modules["results"]._verify_evaluation(entry, job, original, training[(job["seed"], job["arm"])])
        check_deadline(deadline)
    assert_archive_origins(root)
    return original, modules


def evaluate_complement(protocol, original, modules, job, deadline):
    """Direct archived evaluator only; never call an old worker that writes original_output."""
    check_deadline(deadline)
    torch = modules["torch"]
    training, receipt_hash = training_receipt(protocol, original, modules, job["seed"], job["arm"], deadline=deadline)
    baseline = modules["worker"].zero_allocator_baseline(torch.device("cuda:0"))
    check_deadline(deadline)
    directory = Path(protocol["output"]) / f"seed{job['seed']}" / "evaluation" / job["arm"] / f"lead_{job['lead']:03d}h"
    if (any(part.is_symlink() for part in (directory, *directory.parents))
            or directory.absolute() != directory.resolve()
            or not directory.resolve().is_relative_to(Path(protocol["output"]).resolve())
            or directory.resolve().is_relative_to(Path(protocol["original_output"]).resolve())):
        raise ValueError("canonical new evaluation directory without ancestor symlinks required")
    if directory.exists():
        raise FileExistsError("exclusive complement evaluation directory already exists")
    report = modules["evaluate_local"](
        Path(original["manifests"]) / "val.jsonl", output_dir=directory, checkpoint=training["checkpoint"],
        lead_hours=(job["lead"],), max_samples=modules["protocol"].EVALUATION_MAX_SAMPLES,
        device_name="cuda:0", normalized=False, reasoning_steps=modules["protocol"].REASONING_STEPS,
        deadline=deadline, process_scale_sidecar=original["sidecar"]["path"])
    torch.cuda.synchronize(torch.device("cuda:0"))
    check_deadline(deadline)
    if (report["split"] != "val" or report["channels"] != original["data"]["channels"]
            or report["units"] != original["data"]["units"]
            or report["training_identity"] != original["data"]["data_identity"]):
        raise ValueError("archived evaluation must preserve physical17-channel validation identity")
    files = {name: sha256_file(directory / name) for name in
             ("rmse.csv", "acc.csv", "climatology_skill.csv", "provenance.json")}
    entry = {"protocol_sha256": original["protocol_sha256"],
             "complement_protocol_sha256": protocol["protocol_sha256"],
             "original_training_receipt_sha256": receipt_hash,
             "archived_evaluator_code": {key: original["code"][key] for key in
                                         ("code_zip_sha256", "source_tree_sha256", "model_code_sha256")},
             "model_code_sha256": original["code"]["model_code_sha256"],
             "data_identity": original["data"]["data_identity"],
             "process_supervision": modules["protocol"].expected_training_contract(original, job["arm"]),
             "seed": job["seed"], "arm": job["arm"], "lead_hours": job["lead"], "split": "val", "test_read": False,
             "n_evaluated": report["n_evaluated"], "n_available_windows": report["n_available_windows"],
             "channels": report["channels"], "units": report["units"], "elapsed_seconds": report["elapsed_seconds"],
             "baseline": baseline, "pre_init_baseline": dict(baseline), "initialized_baseline": dict(baseline),
             "memory_scope": "archived helper: measured pre-init zero, init bound CUDA without tensors/clearing, initialized zero then reset before allocations",
             "peak_allocated_bytes": torch.cuda.max_memory_allocated(torch.device("cuda:0")),
             "peak_reserved_bytes": torch.cuda.max_memory_reserved(torch.device("cuda:0")),
             "evaluation_dir": str(directory), "rmse_csv": str(directory / "rmse.csv"),
             "skill_csv": str(directory / "climatology_skill.csv"), "acc_csv": str(directory / "acc.csv"),
             "provenance": str(directory / "provenance.json"), "artifact_sha256": files,
             "checkpoint": training["checkpoint"], "checkpoint_sha256": training["checkpoint_sha256"]}
    # Read exact archived case identities directly; never alter the old training protocol.
    from training.r7_coreasoning_compare import read_case_identity
    cases = read_case_identity(directory)
    declared = original["data"]["evaluation_cases"][str(job["lead"])]
    if (cases is None or cases["cases"] != declared["cases"] or cases["split"] != "val"
            or cases["evaluation_manifest_sha256"] != original["sources"]["val_manifest_sha256"]
            or entry["n_available_windows"] != declared["n_available"]
            or entry["n_evaluated"] != len(declared["cases"])
            or report["checkpoint_sha256"] != training["checkpoint_sha256"]
            or report["process_scale_sidecar_identity"] != original["sidecar"]["identity"]
            or report["training_protocol_sha256"] != original["protocol_sha256"]):
        raise ValueError("exact archived validation cases/checkpoint/sidecar/protocol required")
    modules["results"]._verify_metric_rows(entry, original)
    assert_archive_origins(Path(protocol["output"]) / "archived_code")
    check_deadline(deadline)
    return entry


def run_worker(protocol_path, *, mode, deadline, job=None, receipt=None):
    deny_network()  # Must precede every data/model/evaluator import, including CPU verification.
    payload = {"status": "failed", "scientific_claim": False,
               "limitations": ["archive-isolated complement; zero training; original failed attempt unchanged"],
               "test_read": False, "job": job, "pid": os.getpid(), "platform": platform.platform()}
    original_error = None
    candidate = Path(receipt).absolute() if receipt is not None else None
    result_path = None
    try:
        check_deadline(deadline)
        protocol = read_complement(protocol_path)
        payload.update(limitations=protocol["limitations"], complement_protocol_sha256=protocol["protocol_sha256"],
                       protocol_sha256=protocol["original_protocol_sha256"])
        root = Path(protocol["output"]) / "archived_code"
        if mode == "evaluate":
            if job not in protocol["jobs"] or os.environ.get("CUDA_VISIBLE_DEVICES") != protocol["gpu"]["uuid"]:
                raise ValueError("exact evaluation job and bound CUDA UUID required")
            key = f"evaluate_seed{job['seed']}_{job['arm']}_lead{job['lead']:03d}h"
            candidate = Path(protocol["output"]) / "workers" / (key + ".json")
        elif mode != "verify" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("CPU identity verifier must hide CUDA; no device fallback")
        if (candidate is None or not candidate.resolve().is_relative_to(Path(protocol["output"]).resolve())
                or any(part.is_symlink() for part in (candidate, *candidate.parents))):
            raise ValueError("new exclusive output receipt required")
        result_path = candidate  # Assign only after acceptance: never publish to a rejected old/outside path.
        isolate_archive(root, protocol["code"]["source_root"])
        verify_archived_sources(protocol, root, deadline=deadline)
        original, modules = archived_preflight(protocol, root, deadline, job=job)
        if mode == "evaluate":
            payload.update(evaluate_complement(protocol, original, modules, job, deadline))
        else:
            payload.update(qualification="six original selected400 checkpoints and seven validation results",
                           archived_evaluator_code={key: original["code"][key] for key in
                                                    ("code_zip_sha256", "source_tree_sha256", "model_code_sha256")})
        check_deadline(deadline)
        payload["status"] = "success"
    except BaseException as exc:
        original_error = exc
        payload["failure_reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if result_path is not None:
            try:
                write_json(result_path, payload)
            except BaseException as exc:
                preserve_error(original_error, exc)
    return payload


def main(argv=None):
    deny_network()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--mode", choices=("verify", "evaluate"), required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--seed", type=int, choices=(41, 42))
    parser.add_argument("--arm", choices=("aux_off", "input_aux", "future_draft_aux"))
    parser.add_argument("--lead", type=int, choices=(6, 12, 24, 48, 72))
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    if not math.isfinite(args.deadline):
        parser.error("a finite driver-supplied overall deadline is required")
    if args.mode == "evaluate" and (None in (args.seed, args.arm, args.lead) or args.receipt is not None):
        parser.error("evaluation requires an exact seed/arm/lead and its automatic new receipt path")
    if args.mode == "verify" and (args.receipt is None or any(v is not None for v in (args.seed, args.arm, args.lead))):
        parser.error("CPU verification requires a new receipt and no evaluation job")
    job = {"phase": "evaluate", "seed": args.seed, "arm": args.arm, "lead": args.lead} if args.mode == "evaluate" else None
    run_worker(args.protocol.resolve(), mode=args.mode, deadline=args.deadline, job=job, receipt=args.receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
