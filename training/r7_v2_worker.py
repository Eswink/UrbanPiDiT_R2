"""One fresh owned v2 worker; offline bootstrap precedes every project import."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import platform
import time


def deny_network():
    """Only the existing stdlib-only offline bootstrap precedes other project imports."""
    from scripts.r7_m3_offline import deny_network as shared_denial
    shared_denial()


def check_deadline(deadline):
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
        raise ValueError("finite absolute worker deadline required")
    if time.perf_counter() >= deadline:
        raise TimeoutError("v2 worker whole-attempt deadline exhausted")


def zero_allocator_baseline(device, *, torch_module=None):
    if torch_module is None:
        import torch as torch_module
    cuda = torch_module.cuda
    baseline = {"allocated_bytes": cuda.memory_allocated(device), "reserved_bytes": cuda.memory_reserved(device)}
    if baseline != {"allocated_bytes": 0, "reserved_bytes": 0}:
        raise ValueError("fresh train/eval worker requires measured pre-init zero CUDA allocator")
    cuda.init()
    cuda.set_device(device)
    initialized = {"allocated_bytes": cuda.memory_allocated(device), "reserved_bytes": cuda.memory_reserved(device)}
    if initialized != baseline:
        raise ValueError("initialized fresh worker allocator must remain zero before reset")
    cuda.reset_peak_memory_stats(device)
    return baseline


def _verify_pins(protocol, deadline):
    from .r7_v2_identity import verify_code, verify_input_pins
    check = lambda: check_deadline(deadline)
    verify_code(protocol, check=check)
    verify_input_pins(protocol, check=check)
    check()


def _common(protocol, job):
    from .r7_v2_protocol import digest
    return {"protocol_sha256": protocol["protocol_sha256"],
            "model_code_sha256": protocol["code"]["model_code_sha256"],
            "source_tree_sha256": protocol["code"]["source_tree_sha256"],
            "code_zip_sha256": protocol["code"]["code_zip_sha256"],
            "data_identity": protocol["data"]["data_identity"], "source_sha256": protocol["sources"]["source_sha256"],
            "sidecar_identity": protocol["sidecar"]["identity"], "windows_sha256": digest(protocol["windows"]),
            "windows": protocol["windows"], "sources": protocol["sources"], "sidecar": protocol["sidecar"],
            "arm_config": protocol["arm_configs"][job["arm"]], "shared_controls": protocol["shared_controls"],
            "cpu_profile": protocol["cpu_profile"]["measurements"][str(job["seed"])][job["arm"]],
            "parent_provenance": protocol["parents"].get(str(job["seed"])),
            "scientific_claim": False, "limitations": list(protocol["limitations"]), "test_read": False}


def train_worker(protocol, job, deadline):
    from .r7_autoregressive_runner import fine_tune
    from .r7_experiment import load_checkpoint
    from .r7_v2_profile import model_for_arm, state_hash
    from .r7_v2_protocol import child_contract, digest, sha256_file, train_output_dir, write_path
    check_deadline(deadline)
    model, initialization, spec = model_for_arm(protocol["stage"], protocol["parents"],
                                               protocol["configuration"], job["seed"], job["arm"])
    pairing = protocol["cpu_profile"]["pairing"][str(job["seed"])][job["arm"]]
    if (state_hash(model.state_dict()) != pairing["full_initial_state_sha256"]
            or digest(initialization) != pairing["initialization_report_sha256"] or spec != pairing["model_spec"]):
        raise ValueError("worker initial mapped/imported state differs from frozen CPU pairing")
    contract = child_contract(protocol, job, spec)
    contract["initialization"] = initialization
    contract["parent_provenance"] = initialization if protocol["stage"] == "B" else None
    config, controls = protocol["arm_configs"][job["arm"]], protocol["shared_controls"]
    directory = train_output_dir(protocol["output"], job["seed"], job["arm"])
    if directory.exists():
        raise FileExistsError("fresh training arm output only; no resume/retry")
    check_deadline(deadline)
    checkpoint, report = fine_tune(
        Path(protocol["manifests"]) / "train.jsonl", directory, model=model, contract=contract,
        parent_weights=None, steps=controls["steps"], updates=config["updates"], seed=job["seed"], mode=config["mode"],
        lr=controls["lr"], warmup=controls["warmup"], weight_decay=controls["weight_decay"], bf16=controls["bf16"],
        deadline=deadline, device_name="cuda:0", resume=None, checkpoint_every=controls["checkpoint_every"],
        batch_size=controls["batch_size"], clip=controls["clip"], lambda12=controls["lambda12"])
    checkpoint = write_path(checkpoint, protocol["output"])
    if checkpoint.parent != directory:
        raise ValueError("checkpoint must be this new arm's own output")
    check_deadline(deadline)
    saved = load_checkpoint(checkpoint)
    expected = {"kind": config["kind"], "model": spec, "data_identity": protocol["data"]["data_identity"],
                "source_sha256": protocol["sources"]["source_sha256"], "protocol_sha256": protocol["protocol_sha256"]}
    if (report["updates_this_run"] != config["updates"] or report["selected_update"] != config["updates"]
            or report["resumed_from_updates"] != 0 or saved["updates"] != config["updates"]
            or report["contract"] != saved["contract"] or report["windows"] != protocol["windows"]
            or saved["model_code_sha256"] != protocol["code"]["model_code_sha256"]
            or any(saved["contract"].get(key) != value for key, value in expected.items())
            or saved["contract"].get("process_supervision") is not None
            or saved["contract"]["autoregression"]["excluded_sample_ids"] != protocol["windows"]["excluded_sample_ids"]):
        raise ValueError("complete new-child endpoint/identity/contract required, not old parent supervision or resume")
    report_path = write_path(directory / "training_report.json", protocol["output"])
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "training_report": str(report_path), "training_report_sha256": sha256_file(report_path),
            "contract": saved["contract"], "report": report, "signature": saved["signature"],
            "updates_run": report["updates_this_run"], "selected_update": report["selected_update"],
            "initial_state_sha256": pairing["full_initial_state_sha256"],
            "initialization": initialization, "parent_optimizer_imported": False, "resume": False}


def verify_baseline_provenance(protocol, job, provenance):
    """Require both zero-trained references on every exact model case/region/channel."""
    from .r7_v2_protocol import BASELINE_REPORTING, REGIONS
    definition = provenance.get("baseline_definition", {})
    climate_kind = provenance.get("climatology", {}).get("kind")
    if (protocol.get("baseline_reporting") != BASELINE_REPORTING
            or definition.get("persistence") != {"kind": "known-last-history-frame", "training_updates": 0}
            or climate_kind != "train-only-month-hour-grid-mean-v1"
            or definition.get("climatology") != {"kind": "train-only-month-hour", "training_updates": 0}
            or definition.get("case_pairing") != "same exact requested-lead initializations and full/interior/edge_2"):
        raise ValueError("mandatory zero-trained persistence/climatology provenance required")
    expected = {(kind, region, channel, job["lead"]) for kind in ("persistence", "climatology")
                for region in REGIONS for channel in protocol["data"]["channels"]}
    units = dict(zip(protocol["data"]["channels"], protocol["data"]["units"]))
    def verify_rows(rows, count):
        keys = {(row.get("baseline_kind"), row.get("region"), row.get("variable"), row.get("lead_hours")) for row in rows}
        if (len(rows) != len(expected) or keys != expected
                or any(row.get(key) != 0 for row in rows for key in ("zero_train_updates", "parameters", "trainable_parameters"))
                or any(row.get("n_initializations") != count or row.get("unit") != units.get(row.get("variable")) for row in rows)):
            raise ValueError("complete paired zero-trained baseline region/channel rows required; no dropped cases")
    inits = provenance.get("baseline_initializations", [])
    model_inits = provenance["initializations"]
    paired = lambda items: [(item.get("sample_id"), item.get("init_time"), item.get("valid_times")) for item in items]
    if paired(inits) != paired(model_inits):
        raise ValueError("baseline cases must exactly match model sample/time identities")
    verify_rows(provenance.get("baseline_region_metrics", []), len(model_inits))
    for item in inits:
        verify_rows(item.get("region_metrics", []), 1)


def evaluate_worker(protocol, job, deadline):
    from .r7_v2_evaluation import evaluate_v2
    from .r7_v2_protocol import evaluation_dir, read_json, sha256_file, worker_result_path, write_path
    train_job = {"phase": "train", "seed": job["seed"], "arm": job["arm"], "lead": None, "reasoning_steps": 4}
    training = read_json(worker_result_path(protocol["output"], train_job))
    checkpoint = write_path(training["checkpoint"], protocol["output"])
    if (training.get("status") != "success" or training.get("job") != train_job
            or training.get("protocol_sha256") != protocol["protocol_sha256"]
            or sha256_file(checkpoint) != training["checkpoint_sha256"]):
        raise ValueError("matching complete new training receipt and unchanged selected checkpoint required")
    directory = evaluation_dir(protocol["output"], job)
    if directory.exists():
        raise FileExistsError("fresh single-lead/K evaluation output only")
    check_deadline(deadline)
    provenance = evaluate_v2(
        Path(protocol["manifests"]) / "val.jsonl", checkpoint, directory, lead_hours=(job["lead"],),
        reasoning_steps=job["reasoning_steps"], max_samples=32, device_name="cuda:0", deadline=deadline,
        process_scale_sidecar=None)  # New child has no inherited parent process_supervision contract.
    declared = protocol["data"]["evaluation_cases"][str(job["lead"])]
    cases = [[item["init_time"], item["valid_times"]] for item in provenance["initializations"]]
    if (provenance["split"] != "val" or provenance["test_read"] is not False
            or provenance["scientific_claim"] is not False or provenance["channels"] != protocol["data"]["channels"]
            or provenance["units"] != protocol["data"]["units"] or provenance["training_identity"] != protocol["data"]["data_identity"]
            or provenance["n_evaluated"] != len(declared["cases"]) or provenance["n_available_windows"] != declared["n_available"]
            or sorted(cases) != sorted(declared["cases"]) or provenance["lead_hours"] != [job["lead"]]
            or provenance["reasoning_steps"] != job["reasoning_steps"]):
        raise ValueError("physical 17-variable full per-lead cohort/K provenance differs from frozen cases")
    verify_baseline_provenance(protocol, job, provenance)
    files = {name: sha256_file(write_path(directory / name, protocol["output"]))
             for name in ("region_metrics.csv", "per_case_metrics.csv", "baseline_region_metrics.csv",
                          "baseline_per_case_metrics.csv", "provenance.json")}
    return {"evaluation_dir": str(directory), "region_metrics_csv": str(directory / "region_metrics.csv"),
            "per_case_metrics_csv": str(directory / "per_case_metrics.csv"),
            "baseline_region_metrics_csv": str(directory / "baseline_region_metrics.csv"),
            "baseline_per_case_metrics_csv": str(directory / "baseline_per_case_metrics.csv"),
            "provenance": str(directory / "provenance.json"), "artifact_sha256": files,
            "channels": provenance["channels"], "units": provenance["units"], "n_evaluated": provenance["n_evaluated"],
            "n_available_windows": provenance["n_available_windows"], "split": "val", "lead_hours": job["lead"],
            "reasoning_steps": job["reasoning_steps"], "checkpoint": str(checkpoint),
            "checkpoint_sha256": training["checkpoint_sha256"], "evaluation_provenance": provenance}


def run_worker(protocol_path, job, deadline):
    deny_network()
    from scripts.r7_m3_offline import deny_network as shared_denial
    shared_denial()
    from .r7_v2_protocol import (preserve_error, safe_output, verify_protocol, worker_result_path, write_json)
    output = safe_output(Path(protocol_path).parent)
    protocol = verify_protocol(protocol_path)
    result_path = worker_result_path(output, job)
    if (job not in protocol["jobs"] or os.environ.get("CUDA_VISIBLE_DEVICES") != protocol["gpu"]["uuid"]
            or not (output / "run_started.json").is_file() or result_path.exists()):
        raise ValueError("fresh claimed exact worker job and fixed visible GPU UUID required")
    payload = {"status": "failed", "job": job, "pid": os.getpid(), "platform": platform.platform(),
               "scientific_claim": False, "limitations": list(protocol["limitations"]), "test_read": False,
               "protocol_sha256": protocol["protocol_sha256"], "deadline_perf_counter": deadline,
               "monotonic_boot_id": protocol["monotonic_boot_id"], "budget_limited": False}
    error, started = None, time.perf_counter()
    try:
        check_deadline(deadline)
        from .r7_v2_protocol import monotonic_boot_id
        if protocol["monotonic_boot_id"] != monotonic_boot_id() or deadline != (protocol["round_started_perf_counter"] + protocol["hard_cap_seconds"] - protocol["cleanup_reserve_seconds"]):
            raise ValueError("worker must use unchanged same-boot whole-round hard deadline")
        import torch
        torch.set_num_threads(4)
        baseline = zero_allocator_baseline(torch.device("cuda:0"))
        payload.update(baseline=baseline, pre_init_baseline=dict(baseline), initialized_baseline=dict(baseline),
                       memory_scope="fresh pre-init and initialized zeros, reset before training/evaluation project imports; never empty_cache")
        _verify_pins(protocol, deadline)
        payload.update(_common(protocol, job))
        details = train_worker(protocol, job, deadline) if job["phase"] == "train" else evaluate_worker(protocol, job, deadline)
        torch.cuda.synchronize(torch.device("cuda:0"))
        _verify_pins(protocol, deadline)
        check_deadline(deadline)
        payload.update(details, status="success", elapsed_seconds=time.perf_counter() - started,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(torch.device("cuda:0")),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(torch.device("cuda:0")), torch_version=str(torch.__version__))
    except BaseException as exc:
        error = exc
        payload.update(status="failed", failure_reason=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.perf_counter() - started,
                       budget_limited=isinstance(exc, TimeoutError) and time.perf_counter() >= deadline)
        raise
    finally:
        try:
            write_json(result_path, payload, output=output)
        except BaseException as additional:
            preserve_error(error, additional, "v2 worker result publication failed")
    return payload


def main(argv=None):
    deny_network()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--phase", choices=("train", "evaluate"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--lead", type=int)
    parser.add_argument("--reasoning-steps", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args(argv)
    if (args.phase == "train") != (args.lead is None) or (args.phase == "train" and args.reasoning_steps != 4):
        parser.error("K4 training has no lead; evaluation has exactly one lead and explicit K")
    run_worker(args.protocol, {"phase": args.phase, "seed": args.seed, "arm": args.arm,
                               "lead": args.lead, "reasoning_steps": args.reasoning_steps}, args.deadline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
