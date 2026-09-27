"""Run the #65 pre-diagnostic on the real B2 segment store, CPU only.

Reads the store's own ``process_diagnostics_raw`` / ``process_normalization_std``
and a frozen B2 ``process`` checkpoint. Writes one JSON report. No training, no
optimizer step, no GPU, and no write to the store or the checkpoint; the
checkpoint digest is verified before and after so the read-only claim is
checked rather than asserted.

    python scripts/diagnose_r7_process_reasoning.py \
        --store outputs/r7_b2_segment/store/cache.zarr \
        --val-manifest outputs/r7_b2_segment/store/manifests/val.jsonl \
        --checkpoint outputs/r7_b2_multiseed/seed41/training/seed41/process/update_0000700.pt \
        --out outputs/r7_65_prediagnostic
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# One representative process arm from the frozen B2 run: seed 41, the
# validation-selected checkpoint the harness itself published. Nothing here
# searches over checkpoints for a favourable one.
DIAGNOSTIC_SEED = 41
MODEL_CONFIG = {
    "in_channels": 17, "out_channels": 17, "history_steps": 2,
    "architecture": "window", "dim": 192, "depth": 4, "heads": 4,
    "window_size": 4, "patch_size": 2, "dropout": 0.0,
    "anchored_processes": 8, "free_processes": 8,
    "use_forecast_feedback": True, "default_reasoning_steps": 3,
}
# The auxiliary weights C1 will compare. 0.0 is what B2/B3 actually ran; the
# non-zero values exist so the gradient diagnostic can describe what an engaged
# auxiliary loss does, not to choose a winner inside this module.
AUXILIARY_WEIGHTS = (0.0, 0.01, 0.1)
TRAJECTORY_STEPS = 4
MAX_CASES = 4


def _sha256_file(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="#65 process pre-diagnostic (CPU, read-only).")
    parser.add_argument("--store", required=True, help="R7 store directory")
    parser.add_argument("--val-manifest", required=True, help="val manifest jsonl")
    parser.add_argument("--checkpoint", required=True, help="frozen process checkpoint")
    parser.add_argument("--out", required=True, help="new output directory")
    parser.add_argument("--cases", type=int, default=MAX_CASES,
                        help="validation cases used (kept small on purpose)")
    args = parser.parse_args()

    if args.cases < 1 or args.cases > 24:
        raise ValueError("cases must be in [1, 24]; the diagnostic is bounded")
    output = Path(args.out)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    started = time.monotonic()

    import numpy as np
    import zarr
    from torch.utils.data import default_collate

    from data.r7_zarr_dataset import ZarrAtmosWindowDataset
    from training.r7_experiment import dataset_identity, load_checkpoint, make_model
    from training.r7_experiment import model_code_digest
    from training.r7_process_diagnostic import (
        gradient_conflict_record, measurement_input_contract, parameter_groups,
        proxy_scale_report, recursion_trajectory, summarise_proxy_effect,
    )

    checkpoint = Path(args.checkpoint)
    checkpoint_before = _sha256_file(checkpoint)

    # ---- 1. proxy numeric scale (store arrays only; no model involved) -------
    root = zarr.open_group(args.store, mode="r")
    raw = np.asarray(root["process_diagnostics_raw"][:], dtype=np.float64)
    stored_std = np.asarray(root["process_normalization_std"][:], dtype=np.float64)
    names = list(root.attrs["process_diagnostic_names"])
    report = proxy_scale_report(raw, stored_std, names)
    report["verdicts"] = summarise_proxy_effect(report, names)
    report["input_contract"] = measurement_input_contract()

    # ---- 2/3. gradient conflict and K trajectory on real val cases ----------
    # ``data_identity`` binds a checkpoint to the manifest it trained on, so it
    # is checked against the *train* manifest. The val manifest is a separate
    # file by construction; what must hold is that both describe the same
    # store, which is checked below by comparing store paths.
    val_manifest = Path(args.val_manifest)
    train_manifest = val_manifest.parent / "train.jsonl"
    if not train_manifest.is_file():
        raise FileNotFoundError(train_manifest)
    identity, _ = dataset_identity(train_manifest)
    dataset = ZarrAtmosWindowDataset(val_manifest)
    if {record["split"] for record in dataset.records} != {"val"}:
        raise ValueError("this diagnostic is validation-only")
    store_paths = {str((val_manifest.parent / record["store_path"]).resolve())
                   if not Path(record["store_path"]).is_absolute()
                   else str(Path(record["store_path"]).resolve())
                   for record in dataset.records}
    if store_paths != {str(Path(args.store).resolve())}:
        raise ValueError("val manifest does not point at the declared store")
    saved = load_checkpoint(checkpoint)
    if saved["contract"]["kind"] != "process":
        raise ValueError("a process checkpoint is required for this diagnostic")
    if saved["contract"]["data_identity"] != identity:
        raise ValueError("checkpoint was trained on a different data identity")
    model = make_model("process", saved["contract"]["model"])
    model.load_state_dict(saved["model"], strict=True)
    groups = parameter_groups(model)
    cases = min(args.cases, len(dataset))

    gradients = {}
    trajectories = []
    model.train()
    for index in range(cases):
        batch = default_collate([dataset[index]])
        per_weight = {}
        for weight in AUXILIARY_WEIGHTS:
            record = gradient_conflict_record(
                model, batch, process_weight=weight, steps=3, groups=groups)
            per_weight[str(weight)] = record
        gradients[dataset.records[index]["sample_id"]] = per_weight

    for index in range(cases):
        batch = default_collate([dataset[index]])
        trajectories.append({
            "sample_id": dataset.records[index]["sample_id"],
            "init_time": dataset.records[index]["init_time"],
            "trajectory": recursion_trajectory(model, batch, max_steps=TRAJECTORY_STEPS),
        })

    if _sha256_file(checkpoint) != checkpoint_before:
        raise RuntimeError("the diagnostic modified the original checkpoint")
    payload = {
        "format": "r7-65-prediagnostic-v1",
        "scientific_claim": False,
        "gpu_used": False,
        "training_executed": False,
        "test_evaluated": False,
        "forecast_modified": False,
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "platform": platform.platform(),
        "model_code_sha256": model_code_digest(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_before,
        "checkpoint_selection": ("validation-selected checkpoint published by the frozen "
                                 "B2 run for seed %d; no checkpoint search was performed"
                                 % DIAGNOSTIC_SEED),
        "data": {"store": str(args.store), "val_manifest": str(args.val_manifest),
                 "data_identity": identity, "cases": cases,
                 "case_ids": [record["sample_id"] for record in dataset.records[:cases]]},
        "proxy_scale": report,
        "auxiliary_weights": list(AUXILIARY_WEIGHTS),
        "gradient_conflict": gradients,
        "recursion_trajectory": trajectories,
        "elapsed_seconds": time.monotonic() - started,
        "limitations": [
            "one checkpoint (seed 41 process) and one validation split; not a seed sweep",
            "the gradient cosine is a one-batch geometric description, never causal evidence",
            "the K0..K4 trace is retrospective: it uses future labels to score updates",
            "no test split is read; no training or optimizer step is performed",
        ],
    }
    with (output / "prediagnostic.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    summary = {
        "complete": True,
        "cases": cases,
        "floor_active": report["floor_active_channels"],
        "unusable": report["unusable_channels"],
        "elapsed_seconds": payload["elapsed_seconds"],
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
