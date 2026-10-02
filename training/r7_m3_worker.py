"""One owned fresh CUDA worker per M3 training arm or validation lead."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.r7_m3_offline import deny_network
from .r7_arm_harness import sha256_file
from .r7_m3_identity import local_path, verify_code, verify_input_pins
from .r7_m3_profile import seed_cpu, state_hash
from .r7_m3_protocol import (
    ARM_NAMES, BATCH_SIZE, CLIP, EARLY_STOPPING_PATIENCE, EVALUATION_MAX_SAMPLES,
    LEADS, LIMITATIONS, LR, MINIMUM_IMPROVEMENT, MINIMUM_LR_RATIO,
    REASONING_STEPS, SEEDS, UPDATES, VALIDATION_EVERY, VALIDATION_LEADS,
    WARMUP_UPDATES, expected_training_contract, job_key, model_config, read_json, supervision_contract,
    preserve_artifact_error, verify_protocol, write_json,
)


class M3Dataset(Dataset):
    """Train-only adapter: exact integer timestamps verified against manifest/store."""
    def __init__(self, manifest, *, reader=None):
        from data.r7_zarr_dataset import ZarrAtmosWindowDataset
        self.manifest = local_path(manifest, name="train.jsonl")
        self.reader = reader if reader is not None else ZarrAtmosWindowDataset(self.manifest)
        if self.reader.manifest.resolve() != self.manifest:
            raise ValueError("M3 reader manifest mismatch")
        if not self.reader.records or any(r.get("split") != "train" for r in self.reader.records):
            raise ValueError("M3 wrapper may only expose train samples")

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index):
        from data.r7_store import validate_record
        record = self.reader.records[index]
        root = self.reader._store(record)
        indices = validate_record(root, record)
        stamps = np.asarray([root["time_ns"][i] for i in indices])
        if stamps.dtype.kind != "i" or stamps.shape != (len(indices),):
            raise ValueError("stored timestamps must be real integer nanoseconds")
        stamps = stamps.astype(np.int64, copy=False)
        sample = dict(self.reader[index])
        sample.update({"history_time_ns": torch.from_numpy(stamps[:-1].copy()),
                       "init_time_ns": torch.tensor(int(stamps[-2]), dtype=torch.int64),
                       "valid_time_ns": torch.tensor(int(stamps[-1]), dtype=torch.int64),
                       "target_time_ns": torch.tensor(int(stamps[-1]), dtype=torch.int64),
                       "split": "train"})
        return sample


def check_deadline(deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError("M3 owned worker deadline exhausted")


def make_context(metadata, root):
    from data.r7_store import normalization
    from .r7_process_supervision import ProcessDiagnosticContext
    mean, std = normalization(root, count=17)
    return ProcessDiagnosticContext(metadata, mean, std, list(root.attrs["channels"]))


def worker_result_path(output, job):
    return Path(output) / "workers" / (job_key(job) + ".json")


def train_output_dir(output, seed, arm):
    return Path(output) / f"seed{seed}" / "training" / arm


def train_worker(protocol, job, data, reader, root, metadata, deadline):
    from data.r7_evaluation import ZarrRolloutDataset
    from .r7_experiment import load_checkpoint, make_model
    from .r7_scheduled_runner import run_scheduled_updates
    check_deadline(deadline)
    dataset = M3Dataset(Path(protocol["manifests"]) / "train.jsonl", reader=reader)
    context = make_context(metadata, root)
    validation = ZarrRolloutDataset(data["store"], split="val", lead_hours=VALIDATION_LEADS,
                                   history_steps=2, step_hours=6)
    seed, arm = job["seed"], job["arm"]
    seed_cpu(seed)
    anchor = make_model("process", model_config(17)).cpu()
    shared = {name: value.detach().clone() for name, value in anchor.state_dict().items()}
    initial_hash = state_hash(shared)
    if initial_hash != protocol["cpu_profile"]["pairing"][str(seed)]["full_initial_state_sha256"][arm]:
        raise ValueError("CPU initial state differs from frozen bitwise pairing")
    del anchor
    contract = supervision_contract(protocol, arm)
    from .r7_process_training_contract import supervision_training_contract
    derived, _ = supervision_training_contract(
        context, contract, kind="process", process_weight=0.0, data_identity=data["data_identity"])
    if derived != expected_training_contract(protocol, arm):
        raise ValueError("derived fixed train inverse differs from frozen protocol")
    check_deadline(deadline)
    checkpoint, report = run_scheduled_updates(
        dataset, kind="process", model_config=model_config(17), data_identity=data["data_identity"],
        output_dir=train_output_dir(protocol["output"], seed, arm), total_updates=UPDATES,
        batch_size=BATCH_SIZE, steps=REASONING_STEPS, seed=seed, lr=LR, clip=CLIP,
        process_weight=0.0, warmup_updates=WARMUP_UPDATES, minimum_lr_ratio=MINIMUM_LR_RATIO,
        validation_every=VALIDATION_EVERY, early_stopping_patience=EARLY_STOPPING_PATIENCE,
        minimum_improvement=MINIMUM_IMPROVEMENT, validation_lead_hours=VALIDATION_LEADS,
        device_name="cuda:0", bf16=False, validation_dataset=validation,
        shared_initial_state=shared, deadline=deadline,
        process_supervision_context=context, process_supervision_contract=contract,
    )
    check_deadline(deadline)
    saved = load_checkpoint(checkpoint)
    if (report["updates_this_run"] != UPDATES or report["early_stopped"]
            or saved["updates"] != report["selected_update"]
            or saved["contract"]["process_supervision"] != derived
            or saved["contract"]["data_identity"] != data["data_identity"]):
        raise ValueError("partial training, selected checkpoint or new supervision contract mismatch")
    return {"protocol_sha256": protocol["protocol_sha256"],
            "model_code_sha256": saved["model_code_sha256"], "data_identity": data["data_identity"],
            "process_supervision": derived, "initial_state_sha256": initial_hash,
            "updates_run": report["updates_this_run"], "selected_update": report["selected_update"],
            "early_stopped": report["early_stopped"], "stopped_reason": report["stopped_reason"],
            "elapsed_seconds": report["elapsed_seconds"], "seconds_per_update": report["seconds_per_update"],
            "peak_allocated_bytes": report["peak_allocated_bytes"],
            "peak_reserved_bytes": report["peak_reserved_bytes"],
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "shared_initial_state": report["shared_initial_state"],
            "validation_checks": report["validations"],
            "training_report": str(Path(checkpoint).parent / "training_report.json"),
            "training_report_sha256": sha256_file(Path(checkpoint).parent / "training_report.json"),
            "hardware": report["hardware"], "torch_version": str(torch.__version__)}


def zero_allocator_baseline(device):
    """Fresh process, before select_device/model.to; zero is measured, not assumed."""
    baseline = {"allocated_bytes": torch.cuda.memory_allocated(device),
                "reserved_bytes": torch.cuda.memory_reserved(device)}
    if baseline != {"allocated_bytes": 0, "reserved_bytes": 0}:
        raise ValueError("evaluation requires a measured zero allocator baseline in a fresh process")
    # The C++ peak-reset API is not guaranteed to lazy-init a fresh CUDA context.
    # Initialize only the bound visible device; allocate no tensors and clear nothing.
    torch.cuda.init()
    torch.cuda.set_device(device)
    initialized = {"allocated_bytes": torch.cuda.memory_allocated(device),
                   "reserved_bytes": torch.cuda.memory_reserved(device)}
    if initialized != baseline:
        raise ValueError("evaluation initialized allocator must still have a zero baseline")
    torch.cuda.reset_peak_memory_stats(device)
    return baseline


def evaluate_worker(protocol, job, deadline):
    from .r7_evaluate import evaluate_local
    training_job = {"phase": "train", "seed": job["seed"], "arm": job["arm"], "lead": None}
    training = read_json(worker_result_path(protocol["output"], training_job))
    if training["status"] != "success" or training["protocol_sha256"] != protocol["protocol_sha256"]:
        raise ValueError("successful matching fresh training result required")
    if sha256_file(training["checkpoint"]) != training["checkpoint_sha256"]:
        raise ValueError("selected checkpoint bytes changed")
    baseline = zero_allocator_baseline(torch.device("cuda:0"))
    check_deadline(deadline)
    directory = Path(protocol["output"]) / f"seed{job['seed']}" / "evaluation" / job["arm"] / f"lead_{job['lead']:03d}h"
    report = evaluate_local(
        Path(protocol["manifests"]) / "val.jsonl", output_dir=directory,
        checkpoint=training["checkpoint"], lead_hours=(job["lead"],),
        max_samples=EVALUATION_MAX_SAMPLES, device_name="cuda:0", normalized=False,
        reasoning_steps=REASONING_STEPS, deadline=deadline,
        process_scale_sidecar=protocol["sidecar"]["path"],
    )
    torch.cuda.synchronize(torch.device("cuda:0"))
    check_deadline(deadline)
    if (report["split"] != "val" or report["channels"] != protocol["data"]["channels"]
            or report["units"] != protocol["data"]["units"]
            or report["training_identity"] != protocol["data"]["data_identity"]):
        raise ValueError("M3 evaluation must be unchanged physical 17-channel validation scoring")
    files = {name: sha256_file(directory / name) for name in
             ("rmse.csv", "acc.csv", "climatology_skill.csv", "provenance.json")}
    return {"protocol_sha256": protocol["protocol_sha256"],
            "model_code_sha256": protocol["code"]["model_code_sha256"],
            "data_identity": protocol["data"]["data_identity"],
            "process_supervision": expected_training_contract(protocol, job["arm"]),
            "arm": job["arm"], "lead_hours": job["lead"], "split": "val", "test_read": False,
            "n_evaluated": report["n_evaluated"], "n_available_windows": report["n_available_windows"],
            "channels": report["channels"], "units": report["units"],
            "elapsed_seconds": report["elapsed_seconds"], "baseline": baseline,
            "pre_init_baseline": dict(baseline), "initialized_baseline": dict(baseline),
            "memory_scope": "fresh pre-init zero allocator, select bound device without tensors/clearing, initialized zero check then reset before GPU allocations",
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(torch.device("cuda:0")),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(torch.device("cuda:0")),
            "evaluation_dir": str(directory), "rmse_csv": str(directory / "rmse.csv"),
            "skill_csv": str(directory / "climatology_skill.csv"), "acc_csv": str(directory / "acc.csv"),
            "provenance": str(directory / "provenance.json"), "artifact_sha256": files,
            "checkpoint": training["checkpoint"], "checkpoint_sha256": training["checkpoint_sha256"]}


def run_worker(protocol_path, job, deadline):
    deny_network()
    torch.set_num_threads(4)
    output = Path(protocol_path).parent
    result_path = worker_result_path(output, job)
    payload = {"status": "failed", "scientific_claim": False, "limitations": LIMITATIONS,
               "test_read": False, "job": job, "pid": os.getpid(), "platform": platform.platform()}
    original_error = None
    try:
        check_deadline(deadline)
        protocol = verify_protocol(protocol_path)
        if (str(output.resolve()) != protocol["output"] or job not in protocol["jobs"]
                or os.environ.get("CUDA_VISIBLE_DEVICES") != protocol["gpu"]["uuid"]):
            raise ValueError("worker job/output or CUDA UUID differs from frozen protocol")
        verify_code(protocol)
        data, reader, root, metadata = verify_input_pins(protocol)
        check_deadline(deadline)
        details = (train_worker(protocol, job, data, reader, root, metadata, deadline)
                   if job["phase"] == "train" else evaluate_worker(protocol, job, deadline))
        payload.update(details, status="success")
    except BaseException as exc:
        original_error = exc
        payload["failure_reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            write_json(result_path, payload)
        except BaseException as exc:
            preserve_artifact_error(original_error, exc, label="M3 worker receipt publication failed")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--phase", choices=("train", "evaluate"), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--arm", choices=ARM_NAMES, required=True)
    parser.add_argument("--lead", type=int, choices=LEADS)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    args = parser.parse_args(argv)
    if (args.phase == "train") != (args.lead is None):
        parser.error("training has no lead; each evaluation has one explicit lead")
    run_worker(args.protocol.resolve(), {"phase": args.phase, "seed": args.seed,
                                       "arm": args.arm, "lead": args.lead}, args.deadline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
