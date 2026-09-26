"""Throughput probe for the B2 update budget (D1 train range == B2 train range).

The B2 store's train split is byte-identical to D1's (decision 0008), so D1's
train manifest is the right fixture for measuring per-arm step cost before the
B2 protocol freezes its update budget. This probe runs a short, discarded
training run per arm and extrapolates; nothing here is reported as a result.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from data.r7_zarr_dataset import ZarrAtmosWindowDataset  # noqa: E402
from training.r7_experiment import dataset_identity  # noqa: E402
from training.r7_scheduled_runner import run_scheduled_updates  # noqa: E402

PROBE_UPDATES = 20
ARMS = {
    "unet": ("native", {"architecture": "unet", "dim": 66}),
    "native_window": ("native", {"architecture": "window", "dim": 192, "depth": 6, "heads": 4,
                                 "window_size": 4, "patch_size": 2, "dropout": 0.0}),
    "afno_small": ("native", {"architecture": "afno_small", "dim": 248, "patch_size": 2,
                              "depth": 8, "blocks": 4}),
    "generic": ("generic", {"architecture": "window", "dim": 192, "depth": 4, "heads": 4,
                            "window_size": 4, "patch_size": 2, "dropout": 0.0,
                            "latent_tokens": 16, "default_reasoning_steps": 3}),
    "process": ("process", {"architecture": "window", "dim": 192, "depth": 4, "heads": 4,
                            "window_size": 4, "patch_size": 2, "dropout": 0.0,
                            "anchored_processes": 8, "free_processes": 8,
                            "use_forecast_feedback": True, "default_reasoning_steps": 3}),
}


def main():
    manifests = REPO / "outputs/r7_d1_earthmover/store/manifests"
    dataset = ZarrAtmosWindowDataset(manifests / "train.jsonl")
    identity, _ = dataset_identity(manifests / "train.jsonl")
    channels = int(dataset[0]["coarse_history"].shape[1])
    print(json.dumps({"train_windows": len(dataset), "channels": channels}), flush=True)

    tmp = Path(tempfile.mkdtemp())
    measured = {}
    for name, (kind, config) in ARMS.items():
        model_config = {"in_channels": channels, "out_channels": channels,
                        "history_steps": 2, **config}
        started = time.perf_counter()
        _, report = run_scheduled_updates(
            dataset, kind=kind, model_config=model_config, data_identity=identity,
            output_dir=tmp / name, total_updates=PROBE_UPDATES, batch_size=2, steps=3,
            seed=41, lr=2e-4, clip=1.0, process_weight=0.0, warmup_updates=2,
            minimum_lr_ratio=0.1, validation_every=0, early_stopping_patience=None,
            device_name="cuda")
        elapsed = time.perf_counter() - started
        measured[name] = {
            "seconds_per_update": elapsed / PROBE_UPDATES,
            "peak_allocated_mib": (report["peak_allocated_bytes"] or 0) / 2**20,
            "peak_reserved_mib": (report["peak_reserved_bytes"] or 0) / 2**20,
        }
        print(json.dumps({name: measured[name]}), flush=True)

    per_seed_minutes = {name: value["seconds_per_update"] * 800 / 60
                        for name, value in measured.items()}
    print(json.dumps({
        "estimate_minutes_for_800_updates": {k: round(v, 1)
                                             for k, v in per_seed_minutes.items()},
        "one_seed_five_arms_minutes": round(sum(per_seed_minutes.values()), 1),
        "three_seeds_sequential_gpu_hours": round(3 * sum(per_seed_minutes.values()) / 60, 2),
        "three_seeds_two_gpus_wall_minutes_est": round(1.5 * sum(per_seed_minutes.values()), 1),
    }, indent=1))


if __name__ == "__main__":
    main()
