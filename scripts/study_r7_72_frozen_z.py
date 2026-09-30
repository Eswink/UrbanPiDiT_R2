"""N1: one preregistered frozen-random-Z falsification round, validation only.

The output is new and exclusive. This driver never reuses an archived checkpoint.
A single sequential CUDA device bounds the entire run, including setup and scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.r7_arm_harness import (
    arm_config, comparator_blocks, count_flops, merge_seed_results, pair_cells,
    sha256_file, verify_arm_pairing, write_study_tables,
)
from training.r7_frozen_z_intervention import install_frozen_z, make_spec, validate_frozen_z
from training.r7_frozen_z_protocol import (
    ARMS, ARM_NAMES, BATCH_SIZE, CLIP, COMPARATOR_DEPTH, EARLY_STOPPING_PATIENCE,
    EVALUATION_LEADS, EVALUATION_MAX_SAMPLES, FROZEN_Z_ARM, LR, MINIMUM_IMPROVEMENT,
    MINIMUM_LR_RATIO, PAIRS, PROCESS_WEIGHT, REASONING_STEPS, RW_A_ARM, SEEDS,
    SWITCH_KEYS, UPDATES, VALIDATION_EVERY, VALIDATION_LEADS, WARMUP_UPDATES,
    primary_reading, protocol_payload, refused_test_manifest, verified_registration,
)

GPU_CAP_HOURS = 0.45
RESULT_FORMAT = "r7-n1-frozen-z-result-v1"


def write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)


def deny_network():
    def refused(*args, **kwargs):
        raise RuntimeError("N1 is offline: outbound connections are forbidden")
    socket.socket.connect = refused
    socket.create_connection = refused


def check_deadline(deadline):
    if time.perf_counter() >= deadline:
        raise RuntimeError("N1 whole-round GPU budget deadline exhausted")


def verify_gpu_exclusive(device):
    index = torch.device(device).index
    if index is None:
        raise ValueError("N1 requires an explicit cuda device index")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    identities = visible.split(",") if visible else []
    physical = identities[index] if identities and index < len(identities) else str(index)
    uuid = subprocess.check_output(
        ["nvidia-smi", f"--id={physical}", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True).strip()
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True)
    own_pid = str(os.getpid())
    for row in processes.splitlines():
        fields = [value.strip() for value in row.split(",")]
        if len(fields) == 2 and fields[0] == uuid and fields[1] != own_pid:
            raise RuntimeError(f"N1 GPU {index} is occupied by another process; do not interfere")


def source_identity(manifests):
    preflight_path = manifests / "source_preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    receipt_path = manifests.parent.parent / "source_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    marker_path = manifests / "BUILD_COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker != {"schema_version": 1, "build_complete": True}:
        raise ValueError("N1 requires the unchanged BUILD_COMPLETE publication marker")
    fingerprint = preflight["fingerprint"]
    source = Path(preflight["source_path"])
    source_hash = sha256_file(source)
    if (source_hash != fingerprint["sha256"]
            or source_hash != receipt["local_artifact"]["sha256"]
            or source.stat().st_size != fingerprint["bytes"]
            or receipt.get("synthetic_fallback") is not False):
        raise ValueError("source bytes, local preflight and real-source receipt disagree")
    return {
        "source_sha256": source_hash, "source_bytes": source.stat().st_size,
        "source_receipt_sha256": sha256_file(receipt_path),
        "preflight_report_sha256": sha256_file(preflight_path),
        "build_complete_sha256": sha256_file(marker_path),
        "train_manifest_sha256": sha256_file(manifests / "train.jsonl"),
        "val_manifest_sha256": sha256_file(manifests / "val.jsonl"),
        "scope": "source bytes hashed only for identity; no test fields decoded or scored",
    }


def archive_code(output):
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    added = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=ROOT
    ).decode().split("\0")
    allowed = ("training/r7_frozen_z", "scripts/study_r7_72_frozen_z",
               "tests/test_r7_frozen_z", "tests/test_r7_n1_driver.py", "tools/recompute_r7_n1_audit.py",
               "tests/test_r7_n1_audit.py", "docs/R7_N1_PIVOT_AUDIT.md")
    paths = sorted({name for name in tracked + added if name and
                    (name in tracked or name.startswith(allowed))})
    digest = hashlib.sha256()
    with zipfile.ZipFile(output / "code.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in paths:
            path = ROOT / name
            if not path.is_file() or path.is_symlink():
                continue
            content = path.read_bytes()
            digest.update(name.encode() + b"\0" + content)
            archive.writestr(name, content)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT).decode()
    with (output / "code_commit.txt").open("x", encoding="utf-8") as stream:
        stream.write(commit + "\n")
    with (output / "code_status.txt").open("x", encoding="utf-8") as stream:
        stream.write(status)
    return {"base_commit": commit, "working_tree_sha256": digest.hexdigest(),
            "code_zip_sha256": sha256_file(output / "code.zip"),
            "working_tree_modified": bool(status.strip())}


def measured_arms(probe, channels, specs):
    from training.r7_experiment import make_model, seed_everything
    measured = {}
    for name, kind, config in ARMS:
        seed_everything(SEEDS[0])
        model = make_model(kind, arm_config(kind, channels, config))
        if name == FROZEN_Z_ARM:
            install_frozen_z(model, specs[str(SEEDS[0])])
        forward, backward = count_flops(model, probe, reasoning_steps=REASONING_STEPS)
        measured[name] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "forward_flops": forward, "forward_backward_flops": backward,
        }
    return measured


def pairing_for_seed(channels, seed, spec):
    from training.r7_experiment import make_model, seed_everything
    pairing = verify_arm_pairing(ARMS, baseline=RW_A_ARM, channels=channels, seed=seed)
    if not pairing["all_shared_pairs_identical"]:
        raise RuntimeError("N1 shared initialization is not bitwise identical")
    seed_everything(seed)
    model = make_model("process", arm_config("process", channels, ARMS[-1][2]))
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    rng = torch.get_rng_state().clone()
    install_frozen_z(model, spec)
    unchanged = all(torch.equal(before[name], p) for name, p in model.named_parameters())
    rng_equal = torch.equal(rng, torch.get_rng_state())
    if not unchanged or not rng_equal:
        raise RuntimeError("installing frozen Z changed another weight or RNG stream")
    pairing["frozen_z_installation"] = {
        "all_original_parameters_bitwise_unchanged": unchanged,
        "global_cpu_rng_unchanged": rng_equal,
        "spec_sha256": validate_frozen_z(model, spec),
        "tensor_sha256": spec["tensor_sha256"],
        "same_shape": spec["shape"],
    }
    for name, entry in pairing["anchor_transfer"].items():
        if entry["ignored_count"] or not entry["post_load_all_applied_bitwise_equal"]:
            raise RuntimeError(f"reference transfer failed for {name}")
    return pairing


def train_arm(dataset, validation, output, protocol_path, *, name, kind, config,
              channels, identity, seed, spec, shared, device, deadline):
    from training.r7_experiment import load_checkpoint
    from training.r7_scheduled_runner import run_scheduled_updates
    check_deadline(deadline)
    verify_gpu_exclusive(device)
    protocol = verified_registration(protocol_path)
    checkpoint, report = run_scheduled_updates(
        dataset, kind=kind, model_config=arm_config(kind, channels, config),
        data_identity=identity, output_dir=output / "training" / name,
        total_updates=UPDATES, batch_size=BATCH_SIZE, steps=REASONING_STEPS,
        seed=seed, lr=LR, clip=CLIP, process_weight=PROCESS_WEIGHT,
        warmup_updates=WARMUP_UPDATES, minimum_lr_ratio=MINIMUM_LR_RATIO,
        validation_every=VALIDATION_EVERY, early_stopping_patience=EARLY_STOPPING_PATIENCE,
        minimum_improvement=MINIMUM_IMPROVEMENT, validation_lead_hours=VALIDATION_LEADS,
        device_name=device, validation_dataset=validation, shared_initial_state=shared,
        intervention=spec if name == FROZEN_Z_ARM else None, deadline=deadline,
    )
    saved = load_checkpoint(checkpoint)
    if report["updates_this_run"] != UPDATES or saved["updates"] != report["selected_update"]:
        raise RuntimeError("partial training or selected checkpoint mismatch")
    return {
        "protocol_sha256": protocol["protocol_sha256"],
        "switches": {key: config.get(key, default) for key, default in SWITCH_KEYS},
        "updates_run": report["updates_this_run"], "selected_update": report["selected_update"],
        "early_stopped": report["early_stopped"], "stopped_reason": report["stopped_reason"],
        "elapsed_seconds": report["elapsed_seconds"],
        "seconds_per_update": report["seconds_per_update"],
        "peak_allocated_bytes": report["peak_allocated_bytes"],
        "peak_reserved_bytes": report["peak_reserved_bytes"],
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "shared_initial_state": report["shared_initial_state"],
        "validation_checks": report["validations"],
    }


def evaluate_arm(manifests, output, entry, *, name, lead, device, deadline):
    from training.r7_evaluate import evaluate_local
    check_deadline(deadline)
    verify_gpu_exclusive(device)
    val = manifests / "val.jsonl"
    refused_test_manifest(val)
    directory = output / "evaluation" / name / f"lead_{lead:03d}h"
    report = evaluate_local(
        val, output_dir=directory, checkpoint=entry["checkpoint"], lead_hours=(lead,),
        max_samples=EVALUATION_MAX_SAMPLES, device_name=device,
        reasoning_steps=REASONING_STEPS, deadline=deadline,
    )
    if report["split"] != "val":
        raise RuntimeError("N1 scored a non-validation split")
    return {
        "arm": name, "lead_hours": lead, "split": "val", "test_read": False,
        "n_evaluated": report["n_evaluated"],
        "n_available_windows": report["n_available_windows"],
        "channels": report["channels"], "units": report["units"],
        "elapsed_seconds": report["elapsed_seconds"],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "evaluation_dir": str(directory), "rmse_csv": str(directory / "rmse.csv"),
        "skill_csv": str(directory / "climatology_skill.csv"),
        "intervention": report.get("intervention"),
    }


def run_seed(dataset, validation, output, protocol, *, manifests, identity, channels,
             measured, seed, spec, device, deadline):
    from training.r7_experiment import make_model, model_code_digest, seed_everything
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", protocol)
    verified_registration(output / "protocol.json")
    pairing = pairing_for_seed(channels, seed, spec)
    seed_everything(seed)
    anchor = make_model("process", arm_config("process", channels, ARMS[0][2]))
    shared = {name: value.clone() for name, value in anchor.state_dict().items()}
    result = {
        "format": RESULT_FORMAT, "scientific_claim": False, "test_read": False,
        "limitations": protocol["limitations"], "seed": seed, "seeds": list(SEEDS),
        "device": device, "gpu": torch.cuda.get_device_name(),
        "torch_version": str(torch.__version__), "platform": platform.platform(),
        "model_code_sha256": model_code_digest(), "protocol": protocol,
        "protocol_sha256": protocol["protocol_sha256"], "arm_pairing": pairing,
        "flop_measurements": measured, "training": {}, "evaluation": {},
        "case_counts_by_lead": {}, "probes": {}, "budget": {},
    }
    training_started = time.perf_counter()
    for name, kind, config in ARMS:
        result["training"][name] = train_arm(
            dataset, validation, output, output / "protocol.json", name=name, kind=kind,
            config=config, channels=channels, identity=identity, seed=seed, spec=spec,
            shared=None if name == RW_A_ARM else shared, device=device, deadline=deadline,
        )
        print(json.dumps({"trained": name, "seed": seed}), flush=True)
    result["budget"]["training_seconds_total"] = time.perf_counter() - training_started
    for name in ARM_NAMES:
        for lead in EVALUATION_LEADS:
            result["evaluation"][f"{name}@{lead}h"] = evaluate_arm(
                manifests, output, result["training"][name], name=name, lead=lead,
                device=device, deadline=deadline,
            )
    for lead in EVALUATION_LEADS:
        counts = {result["evaluation"][f"{name}@{lead}h"]["n_evaluated"] for name in ARM_NAMES}
        if len(counts) != 1:
            raise RuntimeError("N1 case counts are not paired")
        result["case_counts_by_lead"][str(lead)] = counts.pop()
    write_json(output / "seed_result.json", result)


def finalize(output, elapsed):
    merged = merge_seed_results(output, seeds=SEEDS, arms=ARM_NAMES, fmt=RESULT_FORMAT)
    merged["limitations"] = merged["protocol"]["limitations"]
    merged["budget"]["whole_round_elapsed_seconds"] = elapsed
    merged["budget"]["gpu_hours_charged"] = elapsed / 3600.0
    if elapsed > GPU_CAP_HOURS * 3600:
        raise RuntimeError("whole-round measured GPU time exceeds its frozen cap")
    identity = {
        "dataset_identity": merged["protocol"]["data"]["data_identity"],
        "model_code_sha256": merged["model_code_sha256"],
        "declared_update_budget": UPDATES, "evaluation_split": "val",
    }
    table, blocks = comparator_blocks(merged, pairs=PAIRS, depth=COMPARATOR_DEPTH,
                                      identity=identity)
    pairs = pair_cells(blocks, leads=EVALUATION_LEADS)
    write_json(output / "merged_result.json", merged)
    comparison = {
        "format": "r7-n1-frozen-z-comparison-v1", "scientific_claim": False,
        "limitations": merged["limitations"], "test_read": False,
        "protocol_sha256": merged["protocol_sha256"], "identity": identity,
        "pairs": pairs, "primary": primary_reading(pairs), "table": table,
    }
    write_json(output / "paired_comparison.json", comparison)
    write_study_tables(merged, output)
    print(json.dumps({"status": "success", "gpu_hours_charged": elapsed / 3600,
                      "primary": comparison["primary"]}, ensure_ascii=False), flush=True)


def run(manifests, output, authorization_path, device):
    from data.r7_evaluation import ZarrRolloutDataset
    from data.r7_zarr_dataset import ZarrAtmosWindowDataset
    from torch.utils.data import default_collate
    from training.r7_experiment import dataset_identity, select_device
    whole_started = time.perf_counter()
    whole_deadline = whole_started + 1800.0
    torch.set_num_threads(4)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if authorization.get("status") != "authorized" or not authorization.get("user_response"):
        raise ValueError("N1 requires an actual recorded user authorization")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    refused_test_manifest(manifests / "train.jsonl")
    refused_test_manifest(manifests / "val.jsonl")
    source = source_identity(manifests)
    dataset = ZarrAtmosWindowDataset(manifests / "train.jsonl")
    identity, _ = dataset_identity(manifests / "train.jsonl")
    probe = default_collate([dataset[0], dataset[1]])
    channels, h, w = probe["coarse_history"].shape[-3:]
    patch, dim = ARMS[-1][2]["patch_size"], ARMS[-1][2]["dim"]
    token_hw = ((h + patch - 1) // patch, (w + patch - 1) // patch)
    specs = {str(seed): make_spec(seed, token_hw, dim, patch) for seed in SEEDS}
    check_deadline(whole_deadline)
    measured = measured_arms(probe, channels, specs)
    check_deadline(whole_deadline)
    output.mkdir(parents=True, exist_ok=False)
    code_identity = archive_code(output)
    protocol = protocol_payload(manifests, identity, channels, measured,
                                intervention_specs=specs, authorization=authorization,
                                code_identity=code_identity, source_identity=source)
    write_json(output / "protocol.json", protocol)
    verified_registration(output / "protocol.json")
    write_json(output / "environment.json", {"python": sys.version, "torch": str(torch.__version__),
                                             "platform": platform.platform()})
    validation = ZarrRolloutDataset(manifests.parent / "cache.zarr", split="val",
                                   lead_hours=VALIDATION_LEADS, history_steps=2, step_hours=6)
    check_deadline(whole_deadline)
    started = time.perf_counter()
    deadline = min(whole_deadline, started + GPU_CAP_HOURS * 3600 - 10.0)
    status, failure_reason, gpu_elapsed = "failed", None, None
    try:
        verify_gpu_exclusive(device)
        select_device(device)
        for seed in SEEDS:
            check_deadline(deadline)
            run_seed(dataset, validation, output / f"seed{seed}", protocol,
                     manifests=manifests, identity=identity, channels=channels,
                     measured=measured, seed=seed, spec=specs[str(seed)], device=device,
                     deadline=deadline)
        torch.cuda.synchronize()
        gpu_elapsed = time.perf_counter() - started
        check_deadline(whole_deadline)
        finalize(output, gpu_elapsed)
        check_deadline(whole_deadline)
        status = "success"
    except Exception as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        elapsed = (time.perf_counter() - started) if gpu_elapsed is None else gpu_elapsed
        write_json(output / "attempt.json", {
            "status": status, "scientific_claim": False, "test_read": False,
            "limitations": protocol["limitations"], "gpu_phase_elapsed_seconds": elapsed,
            "whole_round_elapsed_seconds": time.perf_counter() - whole_started,
            "gpu_hours_charged": elapsed / 3600.0, "cap_gpu_hours": GPU_CAP_HOURS,
            "failure_reason": failure_reason,
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, default=Path("outputs/r7_m2_segment/store/manifests"))
    parser.add_argument("--out", type=Path, default=Path("outputs/r7_72_frozen_z"))
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    if not args.device.startswith("cuda"):
        parser.error("this registered GPU round has no CPU experiment fallback")
    deny_network()
    run(args.manifests.resolve(), args.out.resolve(), args.authorization.resolve(), args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
