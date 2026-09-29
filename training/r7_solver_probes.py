"""Validation-only probes on a trained solver checkpoint (#72 M2-B).

Two questions a bounded round asks about a checkpoint that a single RMSE number
cannot answer. First, what does the *trained* solver do when it is stopped
earlier or allowed one step more: the depth probe re-runs one checkpoint at
K=1/2/4, which measures sensitivity to the reasoning budget and is explicitly
**not** an independently trained model at those depths. Second, what does the
correction trajectory look like: the correction probe reports per-step update
energy (magnitude), the error/update cosine (angle), and how often a step
worsens the error, moves it the wrong way or overshoots.

Everything here reads the validation split of the same store the training used,
through the same reader, with the model in eval mode. The retrospective damping
reported by the correction geometry is an *oracle* quantity that uses future
truth; it is a diagnostic and never a deployable rule.
"""
from __future__ import annotations

from pathlib import Path

import torch


def depth_probe(manifests_dir, output_dir, results, *, arms, steps_list, leads,
                device_name, max_samples, trained_steps):
    """Re-run one trained checkpoint at several reasoning depths, val only."""
    from training.r7_evaluate import evaluate_local

    manifests_dir, output_dir = Path(manifests_dir), Path(output_dir)
    probe = {}
    for arm in arms:
        checkpoint = (output_dir / "training" / arm
                      / f"update_{results['training'][arm]['selected_update']:07d}.pt")
        for steps in steps_list:
            run_dir = output_dir / "probes" / "depth" / arm / f"k{steps}"
            report = evaluate_local(manifests_dir / "val.jsonl", output_dir=run_dir,
                                    checkpoint=checkpoint, lead_hours=leads,
                                    max_samples=max_samples, device_name=device_name,
                                    reasoning_steps=steps)
            if report["split"] != "val":
                raise ValueError(f"{arm} depth probe was scored on {report['split']}")
            probe[f"{arm}@{steps}"] = {
                "arm": arm, "reasoning_steps": steps, "trained_steps": trained_steps,
                "lead_hours": list(leads), "n_evaluated": report["n_evaluated"],
                "elapsed_seconds": report["elapsed_seconds"],
                "rmse_csv": str(run_dir / "rmse.csv"),
                "skill_csv": str(run_dir / "climatology_skill.csv"),
                "note": ("probe on a checkpoint trained at a different depth; not an "
                         "independently trained model at this depth")}
    return probe


def correction_probe(manifests_dir, results, *, arms, device_name, windows, max_steps):
    """Per-step error/update geometry on validation windows, per arm.

    ``error_update_cosine`` is cos(e, d) with ``e`` the current error and ``d``
    the step's correction: negative means the correction points against the
    error, which is the direction that reduces it. ``retrospective_damping`` is
    the oracle factor that would have minimised the step's MSE and it uses
    future truth, so it is reported as a diagnostic only.
    """
    from data.r7_evaluation import ZarrRolloutDataset
    from torch.utils.data import default_collate
    from training.r7_correction_diagnostic import collect_correction_terms
    from training.r7_experiment import load_checkpoint, make_model, select_device

    manifests_dir = Path(manifests_dir)
    device = select_device(device_name)
    dataset = ZarrRolloutDataset(manifests_dir.parent / "cache.zarr", split="val",
                                 lead_hours=(6,), history_steps=2, step_hours=6)
    batch = default_collate([dataset[index]
                             for index in range(min(windows, len(dataset)))])
    # The rollout reader hands back one target stack per declared lead; the
    # geometry below wants a single [B,C,H,W] field, so the (single) lead is taken
    # out of its axis. An unexpected depth is an error, not something to reshape.
    targets = batch["rollout_targets"]
    if targets.ndim != 5 or targets.shape[1] != 1:
        raise ValueError(f"the correction probe expects one lead, got {tuple(targets.shape)}")
    batch = dict(batch, atmos_target=targets[:, 0])
    batch = {key: value.to(device) if torch.is_tensor(value) else value
             for key, value in batch.items()}
    probe = {}
    for arm in arms:
        checkpoint = (Path(results["training"][arm]["checkpoint"]))
        saved = load_checkpoint(checkpoint)
        model = make_model(saved["contract"]["kind"], saved["contract"]["model"])
        model.load_state_dict(saved["model"], strict=True)
        model = model.to(device).eval()
        terms = collect_correction_terms(model, batch, max_steps=max_steps)
        entry = {"arm": arm, "reasoning_steps": max_steps,
                 "windows": int(batch["atmos_target"].shape[0]),
                 "checkpoint_sha256": results["training"][arm]["checkpoint_sha256"],
                 "steps": []}
        for index, term in enumerate(terms):
            energy = float(term["update_energy"].mean())
            defined = bool(term["error_cosine_defined"].any())
            entry["steps"].append({
                "step": index + 1,
                "update_energy_mean": energy,
                "update_norm_mean": float(energy ** 0.5),
                "mse_before_mean": float(term["mse_before"].mean()),
                "mse_after_mean": float(term["mse_after"].mean()),
                "error_update_cosine_mean": (
                    float(term["error_update_cosine"][term["error_cosine_defined"]].mean())
                    if defined else None),
                "cosine_defined_fraction": float(term["error_cosine_defined"].float().mean()),
                "worsening_fraction": float(term["worsening"].float().mean()),
                "wrong_direction_fraction": float(term["wrong_direction"].float().mean()),
                "overshoot_fraction": float(term["overshoot"].float().mean()),
                "retrospective_damping_mean": float(
                    term["retrospective_damping"].mean()),
                "algebra_residual_max": float(term["algebra_residual"].abs().max())})
        probe[arm] = entry
        del model
    return probe


def write_depth_probe_table(merged, output_dir):
    """The depth-probe rows flattened to one row per (seed, arm, K, lead, variable)."""
    import csv

    path = Path(output_dir) / "depth_probe_table.csv"
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "arm", "reasoning_steps", "trained_steps", "lead_hours",
                         "variable", "rmse"])
        for seed, per_arm in sorted(merged["probes"].items(), key=lambda item: int(item[0])):
            for _, entry in sorted(per_arm.get("depth", {}).items()):
                with Path(entry["rmse_csv"]).open(encoding="utf-8", newline="") as rows:
                    for row in csv.DictReader(rows):
                        writer.writerow([seed, entry["arm"], entry["reasoning_steps"],
                                         entry["trained_steps"], row["lead_hours"],
                                         row["variable"], row["rmse"]])


def solver_probes(manifests_dir, output_dir, results, *, arms, steps_list, leads,
                  device_name, max_samples, trained_steps, windows):
    """Both probes, keeping the depth and correction sections apart."""
    return {"depth": depth_probe(manifests_dir, output_dir, results, arms=arms,
                                 steps_list=steps_list, leads=leads,
                                 device_name=device_name, max_samples=max_samples,
                                 trained_steps=trained_steps),
            "correction": correction_probe(manifests_dir, results, arms=arms,
                                           device_name=device_name, windows=windows,
                                           max_steps=trained_steps)}
