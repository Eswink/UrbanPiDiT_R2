"""B0 learnability probe on the real D1 store (#64): fixed 1/8/32 train windows.

Question: can the minimal model families learn at all on the real frozen D1
segment? Each arm trains on the FIRST 1/8/32 windows of the D1 train manifest
(fixed order, no cherry-picking), with random regularization disabled, for a
fixed 200-update endpoint. Reported per arm and window count:

- the training loss curve (from the audited runner);
- per-channel MSE on those same fixed windows before vs after training, in
  normalized and denormalized physical units, plus a persistence reference;
- whether gradients reach the encoder and solver subtrees;
- input-history index, lead-correspondence and denormalization checks against
  the store.

This proves learnability only - never generalization; the val/test windows of
the D1 engineering store are never read. `scientific_claim: false`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WINDOW_COUNTS = (1, 8, 32)
UPDATES = 200
SEED = 41
BATCH_SIZE = 2
LR = 2e-4
CLIP = 1.0
REASONING_STEPS = 3
# (arm name, runner kind, runner architecture, model config, encoder/solver subtrees)
ARMS = (
    ("native_window", "native", "window",
     {"dim": 32, "depth": 2, "heads": 4, "window_size": 4, "patch_size": 2, "dropout": 0.0},
     ("encoder",), ("head", "lead_time")),
    ("unet", "native", "unet", {"dim": 32}, ("enc1", "enc2"), ("head", "dec1", "dec2")),
    ("afno_small", "native", "afno_small",
     {"dim": 32, "patch_size": 2, "depth": 2}, ("stem",), ("layers", "head")),
    ("generic", "generic", "window",
     {"dim": 32, "depth": 2, "heads": 4, "window_size": 4, "patch_size": 2, "dropout": 0.0,
      "latent_tokens": 16, "default_reasoning_steps": REASONING_STEPS},
     ("backbone.encoder",), ("cell", "draft_encoder", "correction_head")),
)


class WindowSubset(torch.utils.data.Dataset):
    """The FIRST N records of a manifest, in file order - no reordering."""

    def __init__(self, base, count):
        if count < 1 or count > len(base):
            raise ValueError(f"window count {count} outside 1..{len(base)}")
        self.base, self.count = base, count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return self.base[index]


def _channel_metrics(model, dataset, mean, std, batch_size=8):
    """Per-channel normalized MSE and physical RMSE over the fixed windows."""
    from torch.utils.data import DataLoader

    from model.r7_halting import forecast_inputs
    from training.r7_halting import per_sample_latitude_mse

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    per_channel_sq, per_channel_phys, cases = [], [], 0
    total_mse = []
    with torch.no_grad():
        for batch in loader:
            output = model(forecast_inputs(batch))
            forecast = output.forecast if hasattr(output, "forecast") else output
            target = batch["atmos_target"]
            error = (forecast - target).square()
            per_channel_sq.append(error.mean((0, 2, 3)).cpu())
            total_mse.append(per_sample_latitude_mse(
                forecast, target, batch.get("latitude")).mean().item())
            physical = (forecast * std[None, :, None, None] + mean[None, :, None, None]
                        - (target * std[None, :, None, None] + mean[None, :, None, None]))
            per_channel_phys.append(physical.square().mean((0, 2, 3)).cpu())
            cases += len(forecast)
    normalized = torch.stack(per_channel_sq).mean(0).sqrt()
    physical = torch.stack(per_channel_phys).mean(0).sqrt()
    return {
        "cases": cases,
        "normalized_rmse_per_channel": [round(float(v), 6) for v in normalized],
        "physical_rmse_per_channel": [float(v) for v in physical],
        "latitude_weighted_mse_mean": float(np.mean(total_mse)),
    }


def _persistence_reference(dataset, mean, std, channels):
    """Last observed frame as forecast; the naive floor for the same windows."""
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    per_channel = []
    with torch.no_grad():
        for batch in loader:
            base = batch["coarse_history"][:, -1, :channels]
            target = batch["atmos_target"]
            physical = (base * std[None, :, None, None] + mean[None, :, None, None]
                        - (target * std[None, :, None, None] + mean[None, :, None, None]))
            per_channel.append(physical.square().mean((0, 2, 3)).cpu())
    return [float(v) for v in torch.stack(per_channel).mean(0).sqrt()]


def _gradient_reachability(kind, architecture, model_config, channels, dataset, device):
    """One backward on a fixed batch; grads must reach encoder AND solver."""
    from model.r7_halting import forecast_inputs
    from training.r7_experiment import make_model, seed_everything
    from training.r7_halting import per_sample_latitude_mse

    seed_everything(SEED)
    model = make_model(kind, {"in_channels": channels, "out_channels": channels,
                              "history_steps": 2, "architecture": architecture,
                              **model_config}).to(device)
    model.train()
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
             for key, value in torch.utils.data.default_collate(
                 [dataset[index] for index in range(min(2, len(dataset)))]).items()}
    if kind == "native":
        prediction = model(forecast_inputs(batch)).forecast
        loss = per_sample_latitude_mse(prediction, batch["atmos_target"],
                                       batch.get("latitude")).mean()
    else:
        output = model(forecast_inputs(batch), reasoning_steps=REASONING_STEPS)
        loss = per_sample_latitude_mse(output.forecast, batch["atmos_target"],
                                       batch.get("latitude")).mean()
    loss.backward()
    return model, float(loss.detach())


def _subtree_grad_norms(model, encoder_names, solver_names):
    norms = {}
    for role, names in (("encoder", encoder_names), ("solver", solver_names)):
        entries = {}
        for name in names:
            module = dict(model.named_modules())[name]
            total = 0.
            found = False
            for parameter in module.parameters():
                if parameter.grad is not None:
                    found = True
                    total += float(parameter.grad.norm() ** 2)
            entries[name] = {"grad_norm": total ** 0.5, "received_gradient": found and total > 0}
        norms[role] = entries
    return norms


def _window_contract_checks(dataset):
    """History indices, lead correspondence and denormalization against the store."""
    import zarr

    records = dataset.base.records[:len(dataset)]
    store_path = (dataset.base.manifest.parent / records[0]["store_path"]).resolve()
    root = zarr.open_group(str(store_path), mode="r")
    mean = np.asarray(root["normalization_mean"][:], dtype=np.float32)
    std = np.asarray(root["normalization_std"][:], dtype=np.float32)
    problems = []
    for position, record in enumerate(records):
        init = np.datetime64(record["init_time"].replace("+00:00", ""))
        target = np.datetime64(record["target_time"].replace("+00:00", ""))
        history = [np.datetime64(value.replace("+00:00", "")) for value in record["history_times"]]
        if target - init != np.timedelta64(6, "h"):
            problems.append(f"{record['sample_id']}: lead != +6h")
        if history[-1] != init or history[0] != init - np.timedelta64(6, "h"):
            problems.append(f"{record['sample_id']}: history != [init-6h, init]")
        indices = record["history_indices"] + [record["target_index"]]
        frames = np.stack([np.asarray(root["state"][index], dtype=np.float32) for index in indices])
        sample = dataset[position]
        restored = sample["coarse_history"].numpy() * std[None, :, None, None] + mean[None, :, None, None]
        if not np.allclose(restored, frames[:-1], rtol=1e-4, atol=1e-4):
            problems.append(f"{record['sample_id']}: history frames do not round-trip the store")
        restored_target = sample["atmos_target"].numpy() * std[:, None, None] + mean[:, None, None]
        if not np.allclose(restored_target, frames[-1], rtol=1e-4, atol=1e-4):
            problems.append(f"{record['sample_id']}: target does not round-trip the store")
        if sample["lead_time_hours"].item() != 6:
            problems.append(f"{record['sample_id']}: sample lead != 6h")
    return {"checked_windows": len(records), "problems": problems,
            "store_path": str(store_path)}


def run_probe(manifests_dir, output_dir, *, updates=UPDATES, device_name="cuda"):
    from training.r7_experiment import (
        dataset_identity, load_checkpoint, make_model, model_code_digest,
        seed_everything, select_device,
    )
    from training.r7_local_runner import run_local_updates
    from data.r7_zarr_dataset import ZarrAtmosWindowDataset

    manifests_dir, output_dir = Path(manifests_dir), Path(output_dir)
    train_manifest = manifests_dir / "train.jsonl"
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing existing output: {output_dir}")
    dataset = ZarrAtmosWindowDataset(train_manifest)
    identity, _ = dataset_identity(train_manifest)
    channels = int(dataset[0]["coarse_history"].shape[1])
    device = select_device(device_name)

    protocol = {
        "format": "r7-b0-learnability-v1",
        "frozen_before_any_step": True,
        "objective": ("minimal model families must significantly reduce error on the "
                      "fixed 1/8/32-window real D1 training samples (single step +6h); "
                      "learnability only, never generalization"),
        "data_identity": str(identity),
        "train_manifest": str(train_manifest),
        "window_subset_rule": "the first N records of train.jsonl in file order",
        "window_counts": list(WINDOW_COUNTS),
        "arms": [{"name": name, "kind": kind, "architecture": architecture,
                  "model_config": {"in_channels": channels, "out_channels": channels,
                                   "history_steps": 2, "architecture": architecture, **config},
                  "encoder_subtrees": list(encoder), "solver_subtrees": list(solver)}
                 for name, kind, architecture, config, encoder, solver in ARMS],
        "seed": SEED,
        "optimizer_updates": updates,
        "max_updates_before_rethink": updates,
        "stop_condition": ("fixed update endpoint; no early stopping, no extension; a "
                           "nonfinite loss aborts via error_if_nonfinite and is recorded"),
        "if_not_fitting": "investigate implementation/normalization/optimizer first; no long training",
        "batch_size": BATCH_SIZE, "lr": LR, "clip": CLIP,
        "reasoning_steps": REASONING_STEPS,
        "process_weight_generic": 0.0,
        "random_regularization": "dropout 0.0 in every config; AdamW weight_decay=1e-4 is "
                                 "the runner default, declared here",
        "evaluation": "per-channel normalized MSE and physical RMSE on the SAME fixed "
                      "windows, before vs after training; persistence reference; "
                      "gradient reachability of encoder/solver; history/lead checks",
        "test_evaluated": False,
        "val_test_read": False,
        "scientific_claim": False,
        "limitations": ["D1 is one 30-day engineering segment (January 2016)",
                        "overfitting on fixed windows is the intended evidence",
                        "single pre-declared seed; no significance test"],
    }
    from training.r7_experiment import canonical_digest
    protocol["protocol_sha256"] = canonical_digest(
        {key: value for key, value in protocol.items() if key != "protocol_sha256"})
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "protocol.json").open("x", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=False, allow_nan=False)

    digest = model_code_digest()
    import zarr as zarr_module
    store = (dataset.manifest.parent / dataset.records[0]["store_path"]).resolve()
    store_root = zarr_module.open_group(str(store), mode="r")
    mean = np.asarray(store_root["normalization_mean"][:], dtype=np.float32)
    std = np.asarray(store_root["normalization_std"][:], dtype=np.float32)
    results = {"format": "r7-b0-learnability-result-v1", "scientific_claim": False,
               "gpu_used": device.type == "cuda", "test_evaluated": False,
               "model_code_sha256": digest, "protocol": protocol, "arms": {}}
    for name, kind, architecture, config, encoder_names, solver_names in ARMS:
        arm_config = {"in_channels": channels, "out_channels": channels,
                      "history_steps": 2, "architecture": architecture, **config}
        for count in WINDOW_COUNTS:
            subset = WindowSubset(dataset, count)
            checks = _window_contract_checks(subset)
            if checks["problems"]:
                raise ValueError(f"window contract violated: {checks['problems'][:3]}")
            # the exact initial model the runner will build (same seed/construction)
            seed_everything(SEED)
            initial = make_model(kind, arm_config)
            initial_metrics = _channel_metrics(initial, subset, mean, std)
            persistence = _persistence_reference(subset, mean, std, channels)
            del initial
            checkpoint, training = run_local_updates(
                subset, kind=kind, model_config=arm_config, data_identity=identity,
                output_dir=output_dir / "training" / f"{name}_w{count}",
                total_updates=updates, batch_size=BATCH_SIZE, steps=REASONING_STEPS,
                seed=SEED, lr=LR, clip=CLIP,
                process_weight=0.0 if kind == "generic" else 0.1,
                device_name=device_name)
            saved = load_checkpoint(checkpoint)
            if saved["updates"] != updates:
                raise ValueError("fixed optimizer endpoint not reached")
            trained = make_model(kind, arm_config)
            trained.load_state_dict(saved["model"], strict=True)
            trained_metrics = _channel_metrics(trained, subset, mean, std)
            grad_model, grad_loss = _gradient_reachability(
                kind, architecture, config, channels, subset, device)
            grad_norms = _subtree_grad_norms(grad_model, encoder_names, solver_names)
            received = all(entry["received_gradient"]
                           for role in grad_norms.values() for entry in role.values())
            initial_curve = training["losses"][0]["loss"]
            final_curve = training["losses"][-1]["loss"]
            results["arms"][f"{name}_w{count}"] = {
                "arm": name, "windows": count,
                "training_loss_first_update": initial_curve,
                "training_loss_final_update": final_curve,
                "loss_curve_ratio": final_curve / initial_curve if initial_curve else None,
                "initial_evaluation": initial_metrics,
                "trained_evaluation": trained_metrics,
                "persistence_physical_rmse_per_channel": persistence,
                "gradient_reachability": grad_norms,
                "gradient_probe_loss": grad_loss,
                "gradients_reach_encoder_and_solver": received,
                "window_contract_checks": {"checked_windows": checks["checked_windows"],
                                           "problems": checks["problems"]},
                "checkpoint": str(checkpoint),
                "elapsed_seconds": training["elapsed_seconds"],
                "peak_allocated_bytes": training["peak_allocated_bytes"],
            }
            print(json.dumps({"arm": name, "windows": count,
                              "loss_first": initial_curve, "loss_final": final_curve,
                              "gradients_ok": received}), flush=True)
    with (output_dir / "b0_result.json").open("x", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return results


def main():
    parser = argparse.ArgumentParser(description="B0 learnability probe on the real D1 store (#64).")
    parser.add_argument("--manifests", required=True, help="D1 store manifest directory")
    parser.add_argument("--out", required=True, help="new output directory")
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--device", default="cuda", help="cuda (default) or cpu")
    args = parser.parse_args()
    results = run_probe(args.manifests, args.out, updates=args.updates, device_name=args.device)
    summary = {key: {"loss_ratio": value["loss_curve_ratio"],
                     "gradients_ok": value["gradients_reach_encoder_and_solver"]}
               for key, value in results["arms"].items()}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
