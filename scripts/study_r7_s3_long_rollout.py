"""Separately frozen FP32 feasibility and three-seed long physical supervision."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import study_r7_s3_v3_rollout_ft as recipe
from scripts import study_r7_s3_v3_rollout_ft_worker as short_worker
from scripts.study_r7_s3_v3_rollout_isolated import fresh_output
from training.r7_study_process import bounded_process, worker_environment

GPU_UUID = short_worker.GPU_UUID
PHYSICAL_WEIGHTS = (1., .5, 0., .5, 0., 0., 0., .5, 0., 0., 0., .5)
UPDATES, LR, WARMUP = 200, 2e-5, 10
PLANNED_SECONDS, HARD_CAP_SECONDS, PER_SEED_SECONDS = 6300., 12600., 3600.
PROBE_PLANNED_SECONDS, PROBE_HARD_SECONDS = 1800., 3600.
DEFAULT_PROBE = ROOT / "outputs/r7_s3_long_rollout_probe_20261006_attempt01"
DEFAULT_SCREEN = ROOT / "outputs/r7_s3_long_rollout_20261006_attempt01"
ENTRY_FILES = (*short_worker.EXECUTION_ENTRY_FILES, "scripts/study_r7_s3_long_rollout.py")


def write_exclusive(path, body):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(body, handle, indent=2, ensure_ascii=False, allow_nan=False)


def execution_files():
    return sorted(set(short_worker.execution_files()) | set(ENTRY_FILES))


def _identity():
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import dataset_identity, model_code_digest
    from training.r7_long_rollout_runner import training_code_digest
    from data.r7_long_rollout_dataset import preflight_long_rollout_windows
    shared = recipe._shared()
    if sha256_file(recipe.INSTANCE / "source.nc") != shared.V3_SOURCE_SHA256:
        raise RuntimeError("full source SHA changed")
    if sha256_file(recipe.D3_RESULT) != short_worker.D3_RESULT_SHA256:
        raise RuntimeError("registered control changed")
    parent = recipe.parent_pins()
    parent_result = json.loads((recipe.PARENT_RUN / "result.json").read_text(encoding="utf-8"))
    parent["evaluations"] = {seed: data["evaluations"] for seed, data in parent_result["seeds"].items()}
    parent["parent_result_path"] = str(recipe.PARENT_RUN / "result.json")
    control = recipe.control_pins()
    control["d3_result_path"] = str(recipe.D3_RESULT)
    windows = preflight_long_rollout_windows(recipe.TRAIN_MANIFEST, physical_steps=len(PHYSICAL_WEIGHTS))
    spec, _, _ = shared.archived_process_spec()
    from training.r7_experiment import canonical_digest
    return {"instance": str(recipe.INSTANCE), "train_manifest": str(recipe.TRAIN_MANIFEST),
            "val_manifest": str(recipe.VAL_MANIFEST), "source_sha256": shared.V3_SOURCE_SHA256,
            "train_data_identity": dataset_identity(recipe.TRAIN_MANIFEST)[0],
            "val_data_identity": dataset_identity(recipe.VAL_MANIFEST)[0], "train_windows": windows,
            "model_code_sha256": model_code_digest(), "training_code_sha256": training_code_digest(),
            "execution_files_sha256": {name: sha256_file(ROOT / name) for name in execution_files()},
            "arms": {"candidate": {"mode": "long_rollout", "updates": UPDATES, "lr": LR,
                      "warmup": WARMUP, "physical_weights": list(PHYSICAL_WEIGHTS), "steps": 4,
                      "weight_decay": 1e-4, "batch_size": 1, "clip": 1., "checkpoint_every": 20,
                      "bf16": False, "selection": "frozen endpoint; no validation selection",
                      "model": {"kind": "process", "spec": spec, "spec_canonical_digest": canonical_digest(spec)}},
                     "parent": {"updates": 1600, "pins": parent},
                     "control": {"updates": 400, "pins": control}}}


def freeze(output, *, kind, started, deadline, code_archive, code_commit, code_sha256, feasibility=None):
    from training.r7_experiment import canonical_digest
    body = {"format": "r7-s3-long-rollout-protocol-v1", "kind": kind, "scientific_claim": False,
            "test_read": False, "seeds": list(recipe.SEEDS),
            "evaluation_leads_hours": list(recipe.EVALUATION_LEADS), "gpu_uuid": GPU_UUID,
            "estimated_peak_bytes": 2**31, "headroom_margin_bytes": 2**31,
            "execution_mode": "direct owned bounded workers, no descendants; failure stops this attempt",
            "started_perf_counter": started, "deadline_perf_counter": deadline,
            "planned_seconds": PROBE_PLANNED_SECONDS if kind == "probe" else PLANNED_SECONDS,
            "hard_cap_seconds": PROBE_HARD_SECONDS if kind == "probe" else HARD_CAP_SECONDS,
            "per_seed_seconds": PER_SEED_SECONDS, "identity": "metadata prepared in bounded CPU worker below",
            "code_archive_input": str(code_archive), "code_commit": code_commit,
            "code_zip_sha256": code_sha256,
            "hypothesis": "Direct full-BPTT losses at 24/48/72h reduce generated-history error not supervised by +12h alone",
            "difference": "New exact 12-step train windows and target losses; unchanged parent, FP32, K4, LR and 200 updates",
            "decision": {"primary": {"variable": "t2m", "region": "full", "leads_hours": [6, 12],
                                       "text": recipe.PRIMARY_DECISION_TEXT},
                         "gate_pre_screen": {"variables": ["u10", "v10", "mslp"],
                             "leads_hours": list(recipe.EVALUATION_LEADS), "relative_mse_change_max": 0.0,
                             "text": recipe.GATE_DECISION_TEXT}},
            "limitations": ["Development full-region screening, not independent scientific confirmation",
                            "Four30-day season blocks per year are not complete-year coverage",
                            "Training dose and physical losses are not compute-matched to400-update control",
                            "No test access, no scientific/bitwise training claim; failures fully charged"]}
    if feasibility is not None:
        body["feasibility"] = feasibility
    body["protocol_sha256"] = canonical_digest(body)
    write_exclusive(output / "preparation_protocol.json", body)
    return body


def _prepared_protocol(output):
    from training.r7_experiment import canonical_digest
    preliminary = json.loads((output / "preparation_protocol.json").read_text(encoding="utf-8"))
    if preliminary["protocol_sha256"] != canonical_digest({k: v for k, v in preliminary.items()
                                                           if k != "protocol_sha256"}):
        raise RuntimeError("preparation protocol digest mismatch")
    body = {**preliminary, **_identity()}
    if body["kind"] == "screen":
        _validate_feasibility(body)
    body.pop("protocol_sha256")
    body["protocol_sha256"] = canonical_digest(body)
    write_exclusive(output / "prepared_protocol.json", body)


def _validate_feasibility(body):
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest
    pin = body["feasibility"]
    root = Path(pin["root"])
    receipt = json.loads((root / "feasibility_receipt.json").read_text(encoding="utf-8"))
    original = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    attempt = json.loads((root / "attempt.json").read_text(encoding="utf-8"))
    if (sha256_file(root / "feasibility_receipt.json") != pin["receipt_sha256"]
            or attempt["status"] != "complete" or receipt["status"] != "success"
            or original["kind"] != "probe" or receipt["protocol_sha256"] != original["protocol_sha256"]
            or original["protocol_sha256"] != canonical_digest({k: v for k, v in original.items()
                                                               if k != "protocol_sha256"})
            or original["protocol_sha256"] != pin["protocol_sha256"]):
        raise RuntimeError("feasibility identity/status differs from its pin")
    for key in ("source_sha256", "train_data_identity", "val_data_identity", "model_code_sha256",
                "training_code_sha256", "execution_files_sha256", "train_windows", "arms", "gpu_uuid"):
        if original[key] != body[key]:
            raise RuntimeError("feasibility recipe/input differs: " + key)
    if receipt["owned_cuda_reserved_peak_bytes"] != pin["owned_cuda_reserved_peak_bytes"]:
        raise RuntimeError("feasibility owned memory peak changed")
    for key in ("loss", "gradient_norm", "forward_backward_flops", "owned_cuda_reserved_peak_bytes"):
        if not isinstance(receipt[key], (float, int)) or not math.isfinite(receipt[key]) or receipt[key] <= 0:
            raise RuntimeError("invalid feasibility measurement: " + key)


def validate_protocol(output):
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest, dataset_identity, model_code_digest
    from training.r7_long_rollout_runner import training_code_digest
    from data.r7_long_rollout_dataset import preflight_long_rollout_windows
    body = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    if body["protocol_sha256"] != canonical_digest({k: v for k, v in body.items() if k != "protocol_sha256"}):
        raise RuntimeError("protocol digest mismatch")
    if set(body["execution_files_sha256"]) != set(execution_files()):
        raise RuntimeError("incomplete execution source inventory")
    for name, expected in body["execution_files_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise RuntimeError("execution source changed: " + name)
    for split, manifest in (("train", recipe.TRAIN_MANIFEST), ("val", recipe.VAL_MANIFEST)):
        if body[split + "_manifest"] != str(manifest) or dataset_identity(manifest)[0] != body[split + "_data_identity"]:
            raise RuntimeError("dataset identity changed")
    if model_code_digest() != body["model_code_sha256"] or training_code_digest() != body["training_code_sha256"]:
        raise RuntimeError("model/training source identity changed")
    if body["train_windows"] != preflight_long_rollout_windows(recipe.TRAIN_MANIFEST, physical_steps=12):
        raise RuntimeError("train-window metadata changed")
    candidate = body["arms"]["candidate"]
    spec, _, _ = recipe._shared().archived_process_spec()
    expected = {"mode": "long_rollout", "updates": UPDATES, "lr": LR, "warmup": WARMUP,
                "physical_weights": list(PHYSICAL_WEIGHTS), "steps": 4, "weight_decay": 1e-4,
                "batch_size": 1, "clip": 1., "checkpoint_every": 20, "bf16": False,
                "selection": "frozen endpoint; no validation selection",
                "model": {"kind": "process", "spec": spec, "spec_canonical_digest": canonical_digest(spec)}}
    if candidate != expected or body["gpu_uuid"] != GPU_UUID:
        raise RuntimeError("declared GPU/candidate differs from frozen recipe")
    if body["kind"] not in ("probe", "screen"):
        raise RuntimeError("unknown execution kind")
    planned, hard = ((PROBE_PLANNED_SECONDS, PROBE_HARD_SECONDS) if body["kind"] == "probe"
                     else (PLANNED_SECONDS, HARD_CAP_SECONDS))
    if (body["planned_seconds"] != planned or body["hard_cap_seconds"] != hard
            or body["per_seed_seconds"] != PER_SEED_SECONDS
            or abs(body["deadline_perf_counter"] - body["started_perf_counter"] - hard) > 1e-6):
        raise RuntimeError("execution budget differs from frozen deadline")
    if body["train_data_identity"] != recipe._shared().V3_DATA_IDENTITY:
        raise RuntimeError("train identity not the registered v3 instance")
    primary, gate = body["decision"]["primary"], body["decision"]["gate_pre_screen"]
    if (primary["variable"] != "t2m" or primary["region"] != "full" or primary["leads_hours"] != [6, 12]
            or gate["variables"] != ["u10", "v10", "mslp"]
            or gate["leads_hours"] != list(recipe.EVALUATION_LEADS)
            or gate["relative_mse_change_max"] != 0.0):
        raise RuntimeError("scientific reading fields changed")
    if body["kind"] == "screen":
        _validate_feasibility(body)
    if body["seeds"] != list(recipe.SEEDS) or body["evaluation_leads_hours"] != list(recipe.EVALUATION_LEADS):
        raise RuntimeError("seed/lead inventory differs")
    if (body["scientific_claim"] is not False or body["test_read"] is not False
            or body["decision"]["primary"]["text"] != recipe.PRIMARY_DECISION_TEXT
            or body["decision"]["gate_pre_screen"]["text"] != recipe.GATE_DECISION_TEXT):
        raise RuntimeError("frozen claims/readings changed")
    return body


def _parent_model(body, seed):
    from training.r7_experiment import load_checkpoint, make_model
    from training.r7_arm_harness import sha256_file
    path, expected = recipe.pinned_parent(seed)
    if sha256_file(path) != expected or body["arms"]["parent"]["pins"]["checkpoints"][str(seed)]["sha256"] != expected:
        raise RuntimeError("parent checkpoint pin changed")
    saved = load_checkpoint(path)
    contract = saved["contract"]
    if (saved["updates"] != 1600 or contract["seed"] != seed or contract["mode"] != "l6"
            or contract["data_identity"] != body["train_data_identity"]
            or contract["source_sha256"] != body["source_sha256"]):
        raise RuntimeError("parent seed/data/source/endpoint changed")
    spec = body["arms"]["candidate"]["model"]["spec"]
    model = make_model("process", spec)
    model.load_state_dict(saved["model"], strict=True)
    return model, saved["model"], path, expected


def _probe(output, body):
    import torch
    from torch.utils.data import default_collate
    from torch.utils.flop_counter import FlopCounterMode
    from data.r7_long_rollout_dataset import ZarrLongRolloutDataset
    from training.r7_long_rollout import training_long_rollout
    torch.set_num_threads(4)
    dataset = ZarrLongRolloutDataset(recipe.TRAIN_MANIFEST, physical_steps=12,
        expected_exclusions=body["train_windows"]["excluded_sample_ids"],
        expected_window_sha256=body["train_windows"]["window_sha256"])
    sample = default_collate([dataset[0]])
    model, _, _, _ = _parent_model(body, 41)
    model.train().to("cuda:0")
    batch = {name: value.to("cuda:0") if torch.is_tensor(value) else value for name, value in sample.items()}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.enable_grad(), FlopCounterMode(display=False) as counter:
        result = training_long_rollout(model, batch, 4, physical_weights=PHYSICAL_WEIGHTS)
        result.loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    torch.cuda.synchronize()
    if not math.isfinite(float(result.loss)) or not math.isfinite(float(gradient)):
        raise RuntimeError("nonfinite feasibility gradient/loss")
    receipt = {"status": "success", "scientific_claim": False, "test_read": False,
               "protocol_sha256": body["protocol_sha256"], "sample_id": sample["sample_id"][0],
               "forward_backward_flops": int(counter.get_total_flops()),
               "owned_cuda_allocated_peak_bytes": int(torch.cuda.max_memory_allocated()),
               "owned_cuda_reserved_peak_bytes": int(torch.cuda.max_memory_reserved()),
               "loss": float(result.loss.detach()), "per_step_losses": result.per_step_losses.detach().cpu().tolist(),
               "gradient_norm": float(gradient), "elapsed_seconds": time.perf_counter() - started,
               "flop_convention": "supported aten grad-enabled FP32 operations; omitted elementwise/normalization",
               "connection_probe_note": "CPU counterproofs establish early-step gradient connectivity; this probe measures full loss backward",
               "limitations": ["One train sample forward/backward feasibility only, no optimizer update or val/test scoring"]}
    write_exclusive(output / "feasibility_receipt.json", receipt)


def _seed(output, body, seed, deadline):
    import torch
    from training.r7_long_rollout_runner import fine_tune_long_rollout
    from training.r7_evaluate import evaluate_local
    from training.r7_arm_harness import sha256_file
    model, state, parent, parent_sha = _parent_model(body, seed)
    spec = body["arms"]["candidate"]["model"]["spec"]
    contract = {"kind": "process", "model": spec, "data_identity": body["train_data_identity"],
                "source_sha256": body["source_sha256"], "protocol_sha256": body["protocol_sha256"],
                "training_code_sha256": body["training_code_sha256"],
                "initialization": {"parent_checkpoint": str(parent), "parent_checkpoint_sha256": parent_sha,
                                   "parent_endpoint_updates": 1600},
                "autoregression": {"physical_steps": 12, "physical_weights": list(PHYSICAL_WEIGHTS),
                    "window_sha256": body["train_windows"]["window_sha256"],
                    "excluded_sample_ids": body["train_windows"]["excluded_sample_ids"]}}
    folder = output / f"seed{seed}/training/candidate"
    checkpoint, report = fine_tune_long_rollout(recipe.TRAIN_MANIFEST, folder, model=model, contract=contract,
        parent_weights=state, physical_weights=PHYSICAL_WEIGHTS, updates=UPDATES, lr=LR, warmup=WARMUP,
        seed=seed, device_name="cuda:0", deadline=deadline)
    receipt = {"seed": seed, "training_dir": str(folder), "checkpoint": str(checkpoint),
               "checkpoint_sha256": sha256_file(checkpoint), "training_report_sha256": sha256_file(folder / "training_report.json"),
               "parent_checkpoint_sha256": parent_sha, "elapsed_seconds": report["elapsed_seconds"], "evaluations": {}}
    for lead in recipe.EVALUATION_LEADS:
        destination = output / f"seed{seed}/evaluation/candidate/lead_{lead:03d}h"
        provenance = evaluate_local(recipe.VAL_MANIFEST, output_dir=destination, checkpoint=checkpoint,
            lead_hours=(lead,), max_samples=10**9, device_name="cuda:0", reasoning_steps=4, deadline=deadline)
        if provenance["n_evaluated"] != recipe._shared().V3_VAL_COHORTS[str(lead)]:
            raise RuntimeError("evaluation did not complete its declared cohort")
        receipt["evaluations"][str(lead)] = {"dir": str(destination), "n_evaluated": provenance["n_evaluated"],
            "elapsed_seconds": provenance["elapsed_seconds"], "rmse_csv_sha256": sha256_file(destination / "rmse.csv"),
            "skill_csv_sha256": sha256_file(destination / "climatology_skill.csv"),
            "provenance_sha256": sha256_file(destination / "provenance.json")}
    receipt["owned_cuda_reserved_peak_bytes"] = int(torch.cuda.max_memory_reserved())
    write_exclusive(output / f"seed{seed}_receipt.json", receipt)


def worker(output, phase, seed, deadline):
    from scripts.r7_m3_offline import deny_network
    deny_network()
    if phase == "prepare":
        return _prepared_protocol(output)
    if phase == "archive":
        return _archive(output, validate_protocol(output))
    body = validate_protocol(output)
    if phase == "probe" and body["kind"] != "probe" or phase in ("seed", "reading") and body["kind"] != "screen":
        raise RuntimeError("worker phase differs from frozen execution kind")
    if deadline is None or not math.isfinite(deadline) or deadline > body["deadline_perf_counter"] or deadline <= time.perf_counter():
        raise RuntimeError("worker deadline outside frozen round")
    recipe._shared().pin_declared_gpu(GPU_UUID)
    if phase == "probe":
        _probe(output, body)
    elif phase == "seed":
        if seed not in recipe.SEEDS or deadline - time.perf_counter() > PER_SEED_SECONDS:
            raise RuntimeError("seed/deadline differs from freeze")
        _seed(output, body, seed, deadline)
    else:
        from training.r7_long_rollout_evidence import collect_long_readings
        write_exclusive(output / "readings.json", collect_long_readings(output, body))


def _archive(output, body):
    import hashlib
    from io import BytesIO
    import zipfile
    from training.r7_arm_harness import sha256_file
    archive = Path(body["code_archive_input"])
    if any(p.is_symlink() for p in (archive, *archive.parents)):
        raise ValueError("linked archive input forbidden")
    payload = archive.read_bytes()
    if hashlib.sha256(payload).hexdigest() != body["code_zip_sha256"]:
        raise RuntimeError("prepared code archive SHA mismatch")
    with zipfile.ZipFile(BytesIO(payload)) as zipped:
        if zipped.comment.decode() != body["code_commit"]:
            raise RuntimeError("archive commit differs from freeze")
        for name, expected in body["execution_files_sha256"].items():
            if name.endswith(".py") and hashlib.sha256(zipped.read(name)).hexdigest() != expected:
                raise RuntimeError("archive omits or changes execution source")
    with (output / "code_commit.txt").open("x", encoding="utf-8") as handle:
        handle.write(body["code_commit"] + "\n")
    with (output / "code.zip").open("xb") as handle:
        handle.write(payload)
    write_exclusive(output / "archive_receipt.json", {"scientific_claim": False,
                    "code_commit": body["code_commit"], "code_zip_sha256": sha256_file(output / "code.zip")})


def run_attempt(output, *, code_archive, code_commit, code_sha256, probe=False, feasibility_root=DEFAULT_PROBE):
    from scripts.r7_m3_offline import deny_network
    from training.r7_arm_harness import sha256_file
    deny_network()
    output = fresh_output(output)
    started = time.perf_counter()
    hard, planned = (PROBE_HARD_SECONDS, PROBE_PLANNED_SECONDS) if probe else (HARD_CAP_SECONDS, PLANNED_SECONDS)
    deadline = started + hard
    output.mkdir(parents=True, exist_ok=False)
    processes, receipts, feasibility = [], {}, None
    try:
        if not probe:
            source = Path(feasibility_root).resolve()
            if source == output or any(p.is_symlink() for p in (Path(feasibility_root), *Path(feasibility_root).parents)):
                raise ValueError("separate nonlinked feasibility evidence required")
            prior = json.loads((source / "attempt.json").read_text(encoding="utf-8"))
            fit = json.loads((source / "feasibility_receipt.json").read_text(encoding="utf-8"))
            if prior["status"] != "complete" or fit["status"] != "success":
                raise RuntimeError("feasibility did not complete successfully")
            feasibility = {"root": str(source), "receipt_sha256": sha256_file(source / "feasibility_receipt.json"),
                           "protocol_sha256": prior["protocol_sha256"],
                           "owned_cuda_reserved_peak_bytes": fit["owned_cuda_reserved_peak_bytes"]}
        if (not code_archive or not isinstance(code_commit, str) or len(code_commit) != 40
                or not isinstance(code_sha256, str) or len(code_sha256) != 64):
            raise ValueError("prepared exact code archive/commit/SHA required before execution")
        freeze(output, kind="probe" if probe else "screen", started=started, deadline=deadline,
               code_archive=Path(code_archive).absolute(), code_commit=code_commit,
               code_sha256=code_sha256, feasibility=feasibility)
        initial = [("prepare", None), ("archive", None)]
        phases = initial + ([("probe", None)] if probe else
                            [("seed", seed) for seed in recipe.SEEDS] + [("reading", None)])
        peak, body = 0 if feasibility is None else feasibility["owned_cuda_reserved_peak_bytes"], None
        for phase, seed in phases:
            gate = recipe._shared().gpu_gate(GPU_UUID, estimated_peak_bytes=2**31,
                                            owned_reserved_peak_bytes=peak, margin_bytes=2**31)
            tag = phase if seed is None else f"seed{seed}"
            write_exclusive(output / f"{tag}_spawn_gate.json", gate)
            limit = min(deadline, time.perf_counter() + PER_SEED_SECONDS) if phase == "seed" else deadline
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", phase,
                       "--out", str(output), "--deadline", str(limit)]
            if seed is not None:
                command += ["--seed", str(seed)]
            process = bounded_process(command, cwd=ROOT, log_path=output / f"{tag}.log", deadline=limit,
                                      env=worker_environment(GPU_UUID))
            processes.append({"phase": phase, "seed": seed, **process})
            write_exclusive(output / f"{tag}_process.json", processes[-1])
            if process["status"] != "success":
                raise RuntimeError(f"{tag} failed: {process['status']}")
            if phase == "prepare":
                body = json.loads((output / "prepared_protocol.json").read_text(encoding="utf-8"))
                write_exclusive(output / "protocol.json", body)
            elif phase == "seed":
                receipt = json.loads((output / f"seed{seed}_receipt.json").read_text(encoding="utf-8"))
                receipts[str(seed)] = receipt
                peak = max(peak, receipt["owned_cuda_reserved_peak_bytes"])
            print(json.dumps({"phase_done": phase, "seed": seed, "elapsed_seconds": time.perf_counter() - started}), flush=True)
        elapsed = time.perf_counter() - started
        if elapsed >= hard:
            raise RuntimeError("round hard deadline passed before publication")
        result = {"scientific_claim": False, "test_read": False, "protocol_sha256": body["protocol_sha256"],
                  "train_data_identity": body["train_data_identity"], "val_data_identity": body["val_data_identity"],
                  "elapsed_seconds_total": elapsed, "soft_overrun_seconds": max(0., elapsed - planned),
                  "hard_overrun_seconds": max(0., elapsed - hard), "processes": processes,
                  "seeds": receipts, "limitations": body["limitations"]}
        if not probe:
            result.update(json.loads((output / "readings.json").read_text(encoding="utf-8")))
        write_exclusive(output / "result.json", result)
        write_exclusive(output / "attempt.json", {"status": "complete", "scientific_claim": False,
            "test_read": False, "protocol_sha256": body["protocol_sha256"], "elapsed_seconds_total": elapsed,
            "decision": result.get("decision", "feasibility-only")})
        return result
    except BaseException as exc:
        elapsed = time.perf_counter() - started
        write_exclusive(output / "failure.json", {"status": "failed", "scientific_claim": False, "test_read": False,
            "elapsed_seconds_total": elapsed, "soft_overrun_seconds": max(0., elapsed - planned),
            "hard_overrun_seconds": max(0., elapsed - hard), "processes": processes, "seeds_completed": list(receipts),
            "failure_reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
            "limitations": ["Failed attempt preserved and charged in full; no automatic retry or in-place resume"]})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("prepare", "archive", "probe", "seed", "reading"))
    parser.add_argument("--code-archive", type=Path)
    parser.add_argument("--code-commit")
    parser.add_argument("--code-sha256")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--feasibility-root", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    if args.worker:
        return worker(args.out, args.worker, args.seed, args.deadline)
    return run_attempt(args.out or (DEFAULT_PROBE if args.probe else DEFAULT_SCREEN),
                       code_archive=args.code_archive, code_commit=args.code_commit, code_sha256=args.code_sha256,
                       probe=args.probe, feasibility_root=args.feasibility_root)


if __name__ == "__main__":
    main()
