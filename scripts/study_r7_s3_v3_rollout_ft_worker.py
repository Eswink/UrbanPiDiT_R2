"""Non-spawning worker for the isolated S3 v3 rollout continuation."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import study_r7_s3_v3_rollout_ft as recipe

GPU_UUID = "GPU-9d1624af-9d77-aa7c-0620-b6cb778f4ced"
PLANNED_SECONDS = 5400.0
HARD_CAP_SECONDS = 10800.0
PER_SEED_SECONDS = 3600.0
D3_RESULT_SHA256 = "7a3f2eacd5901a45eb6aee503cc40067b3d856b8de2cdfef5e8e5771a8188b64"
EXECUTION_ENTRY_FILES = (
    "scripts/study_r7_s3_v3_rollout_ft.py",
    "scripts/study_r7_s3_v3_rollout_ft_worker.py",
    "scripts/study_r7_s3_v3_rollout_isolated.py",
    "scripts/r7_m3_offline.py",
)


def execution_files():
    names = list(EXECUTION_ENTRY_FILES)
    for directory in ("training", "model", "data"):
        names.extend(str(path.relative_to(ROOT)) for path in (ROOT / directory).rglob("*.py")
                     if "legacy_v531" not in path.parts and "legacy_v531_full" not in path.parts
                     and "legacy_v6" not in path.parts)
    names.append(str((recipe._shared().ARCHIVED / "protocol.json").relative_to(ROOT)))
    return sorted(set(names))


def write_exclusive(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)


def prepare(output):
    from training.r7_arm_harness import sha256_file
    shared = recipe._shared()
    parity = recipe.flop_probe(recipe.TRAIN_MANIFEST)
    body = recipe.protocol_payload(parity)
    if sha256_file(recipe.D3_RESULT) != D3_RESULT_SHA256:
        raise RuntimeError("the registered D3 control result changed")
    source = recipe.INSTANCE / "source.nc"
    if sha256_file(source) != shared.V3_SOURCE_SHA256:
        raise RuntimeError("the full local source SHA256 changed")
    parent_result = json.loads((recipe.PARENT_RUN / "result.json").read_text())
    body["arms"]["parent"]["pins"]["evaluations"] = {
        seed: data["evaluations"] for seed, data in parent_result["seeds"].items()}
    body["arms"]["parent"]["pins"]["parent_result_path"] = str(recipe.PARENT_RUN / "result.json")
    body["arms"]["control"]["pins"]["d3_result_path"] = str(recipe.D3_RESULT)
    from training.r7_experiment import model_code_digest
    from training.r7_autoregressive_runner import _training_code_digest
    body["model_code_sha256"] = model_code_digest()
    body["training_code_sha256"] = _training_code_digest()
    body["seeds"] = list(recipe.SEEDS)
    body["evaluation_leads_hours"] = list(recipe.EVALUATION_LEADS)
    body["format"] = "r7-s3-v3-rollout-isolated-protocol-v1"
    body["stage"] = "S3-V3-RFT-attempt02"
    body["gpu"]["uuid"] = GPU_UUID
    body["gpu"]["gate"] = "read-only UUID/free-memory gate immediately before each own worker spawn"
    body["execution"] = {
        "files_sha256": {name: sha256_file(ROOT / name) for name in execution_files()},
        "mode": "direct own Popen, no worker descendants; parent watchdog independent of CUDA calls",
        "original_failed_attempt": "outputs/r7_s3_v3_rollout_ft_20261006_attempt01",
        "single_reexecution": True, "no_in_place_resume": True,
        "cleanup_grace_seconds_per_signal": 5.0,
        "completed_seeds_must_equal": list(recipe.SEEDS),
    }
    body["budgets"] = {
        "planned_seconds_round": PLANNED_SECONDS, "hard_cap_seconds_round": HARD_CAP_SECONDS,
        "deadline_seconds_per_seed": PER_SEED_SECONDS,
        "whole_round_scope": "parent start, CPU preparation, protocol freeze, each spawn/train/eval, reading and cleanup",
        "campaign_cap_note": "prior failed attempt costs 2.6316 GPU-h in full; 0030 has no total GPU-h permission cap",
        "hard_cap_enforcement": "parent wait timeout; only direct owned PID receives TERM/KILL, actual cleanup charged",
    }
    body["limitations"] = list(body["limitations"]) + [
        "attempt01 was incomplete after a cooperative deadline overshot; no scientific verdict is inherited",
        "same recipe reexecuted once on a prospectively declared co-resident GPU1, not bitwise training reproducibility",
    ]
    write_exclusive(output / "prepared_protocol.json", body)


def frozen_protocol(output):
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest
    body = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    digest = body["protocol_sha256"]
    if digest != canonical_digest({key: value for key, value in body.items() if key != "protocol_sha256"}):
        raise RuntimeError("frozen protocol digest mismatch")
    inventory = body["execution"]["files_sha256"]
    if sorted(inventory) != execution_files():
        raise RuntimeError("frozen execution code inventory is incomplete or different")
    for name, expected in inventory.items():
        if sha256_file(ROOT / name) != expected:
            raise RuntimeError(f"execution code drift: {name}")
    if body["gpu"]["uuid"] != GPU_UUID:
        raise RuntimeError("worker GPU differs from its declared UUID")
    if body["arms"]["control"]["pins"]["d3_result_sha256"] != D3_RESULT_SHA256:
        raise RuntimeError("control result pin differs from the registered run")
    if sha256_file(recipe.D3_RESULT) != D3_RESULT_SHA256:
        raise RuntimeError("control result changed after freeze")
    candidate = body["arms"]["candidate"]
    spec, _, _ = recipe._shared().archived_process_spec()
    expected = {"mode": recipe.FT_MODE, "updates": recipe.FT_UPDATES, "lambda12": recipe.FT_LAMBDA12,
                "lr": recipe.FT_LR, "warmup": recipe.FT_WARMUP, "steps": recipe.STEPS,
                "weight_decay": recipe.WEIGHT_DECAY, "batch_size": recipe.BATCH_SIZE,
                "clip": recipe.CLIP, "checkpoint_every": recipe.CHECKPOINT_EVERY, "bf16": recipe.BF16,
                "selection": "frozen endpoint; no validation selection",
                "model": {"kind": "process", "spec": spec, "spec_canonical_digest": canonical_digest(spec)}}
    if any(candidate[key] != value for key, value in expected.items()):
        raise RuntimeError("the frozen recipe changed")
    decision = body["decision"]
    if (decision["primary"]["text"] != recipe.PRIMARY_DECISION_TEXT
            or decision["gate_pre_screen"]["text"] != recipe.GATE_DECISION_TEXT
            or decision["gate_pre_screen"]["relative_mse_change_max"] != recipe.GATE_TOLERANCE
            or decision["primary"]["leads_hours"] != list(recipe.PRIMARY_LEADS)):
        raise RuntimeError("frozen decision differs from the original scientific recipe")
    budgets = body["budgets"]
    if (budgets["hard_cap_seconds_round"] != HARD_CAP_SECONDS
            or budgets["planned_seconds_round"] != PLANNED_SECONDS
            or budgets["deadline_seconds_per_seed"] != PER_SEED_SECONDS):
        raise RuntimeError("frozen execution budgets differ from the declared attempt")
    validate_data_inventory(body)
    return body


def validate_data_inventory(body):
    from training.r7_experiment import dataset_identity, model_code_digest
    from training.r7_autoregressive_runner import _training_code_digest
    if body["seeds"] != list(recipe.SEEDS) or body["evaluation_leads_hours"] != list(recipe.EVALUATION_LEADS):
        raise RuntimeError("frozen seed/lead inventory changed")
    for phase, manifest in (("train", recipe.TRAIN_MANIFEST), ("val", recipe.VAL_MANIFEST)):
        if body[f"{phase}_manifest"] != str(manifest.resolve()):
            raise RuntimeError("manifest path differs from its freeze")
        identity, _ = dataset_identity(manifest)
        if identity != body[f"{phase}_data_identity"]:
            raise RuntimeError("data/normalization identity differs from its freeze")
    if (model_code_digest() != body["model_code_sha256"]
            or _training_code_digest() != body["training_code_sha256"]):
        raise RuntimeError("model/training implementation differs from freeze")


def validate_deadline(body, deadline):
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
        raise ValueError("deadline must be finite")
    timing = body["timing"]
    hard = body["budgets"]["hard_cap_seconds_round"]
    if abs(timing["deadline_perf_counter"] - timing["started_perf_counter"] - hard) > 1e-6:
        raise RuntimeError("frozen round deadline and budget disagree")
    if deadline > timing["deadline_perf_counter"] or deadline <= time.perf_counter():
        raise RuntimeError("worker deadline outside frozen round")
    if deadline - time.perf_counter() > PER_SEED_SECONDS:
        raise RuntimeError("seed deadline exceeds the frozen seed cap")


def run_seed(seed, output, deadline):
    import torch
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import load_checkpoint
    shared = recipe._shared()
    protocol = frozen_protocol(output)
    validate_deadline(protocol, deadline)
    shared.pin_declared_gpu(GPU_UUID)
    torch.set_num_threads(4)
    parent_path, parent_sha = recipe.pinned_parent(seed)
    checkpoint = load_checkpoint(parent_path)
    contract = checkpoint["contract"]
    expected = {"seed": seed, "mode": "l6", "total_updates": 1600,
                "data_identity": shared.V3_DATA_IDENTITY, "source_sha256": shared.V3_SOURCE_SHA256}
    if checkpoint["updates"] != 1600 or any(contract[key] != value for key, value in expected.items()):
        raise RuntimeError("parent model/data/seed/endpoint identity mismatch")
    if protocol["arms"]["parent"]["pins"]["checkpoints"][str(seed)]["sha256"] != parent_sha:
        raise RuntimeError("parent checkpoint pin changed")
    del checkpoint
    receipt = recipe.run_seed(seed, output, deadline=deadline, device_name="cuda:0")
    for entry in receipt["evaluations"].values():
        folder = Path(entry["dir"])
        entry["provenance_sha256"] = sha256_file(folder / "provenance.json")
        entry["skill_csv_sha256"] = sha256_file(folder / "climatology_skill.csv")
    receipt["owned_cuda_reserved_peak_bytes"] = int(torch.cuda.max_memory_reserved(0))
    write_exclusive(output / f"seed{seed}_receipt.json", receipt)


def read_results(output):
    from training.r7_rollout_evidence import collect_readings
    body = frozen_protocol(output)
    readings = collect_readings(output, body)
    write_exclusive(output / "readings.json", readings)


def main(argv=None):
    from scripts.r7_m3_offline import deny_network
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "seed", "reading"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=recipe.SEEDS)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args(argv)
    deny_network()
    started = time.perf_counter()
    try:
        if args.phase == "prepare":
            prepare(args.out)
        elif args.phase == "reading":
            read_results(args.out)
        else:
            if args.seed is None:
                raise ValueError("seed phase requires a declared seed")
            run_seed(args.seed, args.out, args.deadline)
        print(json.dumps({"phase": args.phase, "seed": args.seed,
                          "elapsed_seconds": time.perf_counter() - started}), flush=True)
    except BaseException as exc:
        path = args.out / (f"{args.phase}_failure.json" if args.seed is None else f"seed{args.seed}_failure.json")
        write_exclusive(path, {"status": "failed", "scientific_claim": False,
                               "elapsed_seconds": time.perf_counter() - started,
                               "failure_reason": f"{type(exc).__name__}: {exc}",
                               "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
