"""Reproduce the cross-lead defect in Stage C2's on-arm objective (decision 0105).

``training/r7_derivative_matching.py::matching_term`` pairs ``result.forecasts[:, -1]`` (step 12,
+72 h) with ``moved["atmos_target"]``. That key is ``physical_targets[:, 0]`` -- the **+6 h**
field -- so the deployed objective penalises a forecast for being correct at +72 h.

This script measures four things on the frozen protocol's own preflight cases:

1. the lead structure (``forecasts`` has 12 steps; ``atmos_target`` equals ``physical_targets[:,0]``);
2. the term at a *perfect* +72 h forecast, with the frozen scales;
3. the same field against its own +72 h truth (exactly zero, which is what the pairing should give);
4. the gradient of the term at that perfect forecast (nonzero = the optimiser is pushed away).

It reads the store and the Stage B checkpoint; it writes nothing and takes no GPU.

    python tools/reproduce_c2_cross_lead_defect.py --out /tmp/c2_defect.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The Stage B fourth point, the parent C2's preflight measured against.
PARENT_CHECKPOINT = ("outputs/r7_s3_throughput_physics_b_p1600_20261106_attempt01/seed41/"
                     "training/update_0128528.pt")

#: C2's own preflight cases, read from the protocol when present.
DEFAULT_CASES = (1000, 9000, 17000, 25000)


def _context(protocol_path: Path):
    """Store constants, dataset and the frozen derivative scales."""
    import numpy as np
    import torch
    import zarr

    from data.r7_long_rollout_dataset import ZarrLongRolloutDataset
    from scripts import r7_s3_curve_point_large_protocol as curve

    body = json.loads(protocol_path.read_text(encoding="utf-8"))
    scales = body["derivative_matching"]["scales"]
    dataset = ZarrLongRolloutDataset(
        curve.TRAIN_MANIFEST, physical_steps=len(curve.PHYSICAL_WEIGHTS),
        expected_exclusions=body["train_windows"]["excluded_sample_ids"],
        expected_window_sha256=body["train_windows"]["window_sha256"])
    store = zarr.open_group(str(curve.STORE), mode="r")
    channels = list(store.attrs["channels"])
    mean = torch.as_tensor(np.asarray(store["normalization_mean"][:]), dtype=torch.float32)
    std = torch.as_tensor(np.asarray(store["normalization_std"][:]), dtype=torch.float32)
    latitude = np.asarray(store["latitude"][:])
    longitude = np.asarray(store["longitude"][:])
    cases = tuple(body.get("preflight", {}).get("cases") or DEFAULT_CASES)
    return body, dataset, channels, mean, std, latitude, longitude, scales, cases


def _denormalise(field, mean, std):
    """[B,C,H,W] normalized -> physical units, per channel."""
    shape = (1, mean.numel(), 1, 1)
    return field * std.view(shape) + mean.view(shape)


def measure(protocol_path: Path) -> dict:
    """The four measurements, on the protocol's own preflight cases."""
    import torch
    from torch.utils.data import default_collate

    from training.r7_derivative_matching import derivative_matching_terms
    from training.r7_long_rollout import training_long_rollout
    from training.r7_experiment import load_checkpoint, make_model
    from scripts import r7_s3_curve_point_large_protocol as curve

    (body, dataset, channels, mean, std, latitude, longitude,
     scales, cases) = _context(protocol_path)

    saved = load_checkpoint(ROOT / PARENT_CHECKPOINT)
    with torch.random.fork_rng(devices=[]):
        model = make_model("process", curve.model_spec())
    model.load_state_dict(saved["model"], strict=True)
    model.train()   # training_long_rollout refuses eval mode; dropout is 0.0, no BatchNorm

    rows = []
    for case in cases:
        batch = default_collate([dataset[case]])
        with torch.enable_grad():
            rollout = training_long_rollout(
                model, batch, curve.REASONING_STEPS,
                physical_weights=curve.PHYSICAL_WEIGHTS, supervision_mask=None)
        truth72 = _denormalise(batch["physical_targets"][:, -1], mean, std).detach()
        analysis6 = _denormalise(batch["atmos_target"], mean, std).detach()
        # (1) the lead structure: is atmos_target really the first step, not the last?
        is_first = bool(torch.equal(batch["physical_targets"][:, 0], batch["atmos_target"]))
        # (2) the deployed pairing, evaluated at a PERFECT +72 h forecast.
        probe = truth72.clone().requires_grad_(True)
        deployed = derivative_matching_terms(probe, analysis6, channels, latitude, longitude,
                                             scales=scales)
        gradient = float(torch.autograd.grad(deployed["total"], probe)[0].norm())
        # (3) the lead-matched pairing, which should be exactly zero at truth.
        matched = derivative_matching_terms(truth72, truth72, channels, latitude, longitude,
                                            scales=scales)
        # (4) the guard the preflight actually ran: target against itself.
        selfcheck = derivative_matching_terms(analysis6, analysis6, channels, latitude, longitude,
                                              scales=scales)
        rows.append({
            "case": case,
            "atmos_target_is_physical_targets_step_0": is_first,
            "forecast_steps": int(rollout.forecasts.shape[1]),
            "forecast_last_step_lead_hours": int(rollout.forecasts.shape[1] * 6),
            "atmos_target_lead_hours": 6,
            "term_at_perfect_72h_vs_analysis_6h": float(deployed["total"].detach()),
            "term_at_perfect_72h_vs_own_72h_truth": float(matched["total"]),
            "gradient_norm_at_perfect_forecast": gradient,
            "preflight_guard_term_target_vs_target": float(selfcheck["total"]),
        })
    return {
        "protocol_sha256": body["protocol_sha256"],
        "parent_checkpoint": PARENT_CHECKPOINT,
        "parent_checkpoint_updates": int(saved["updates"]),
        "frozen_scales": scales,
        "rows": rows,
        "deployed_pairing_is_cross_lead": all(r["atmos_target_is_physical_targets_step_0"]
                                             and r["forecast_last_step_lead_hours"] != 6
                                             for r in rows),
        "perfect_forecast_is_penalised": all(r["term_at_perfect_72h_vs_analysis_6h"] > 0.0
                                             for r in rows),
        "lead_matched_is_zero_at_truth": all(
            r["term_at_perfect_72h_vs_own_72h_truth"] == 0.0 for r in rows),
        "preflight_guard_is_vacuous": all(
            r["preflight_guard_term_target_vs_target"] == 0.0 for r in rows),
        "scientific_claim": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "outputs/r7_s3_derivative_matching_c2_20261108_attempt07/protocol.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = measure(args.protocol)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("deployed_pairing_is_cross_lead", "perfect_forecast_is_penalised",
                       "lead_matched_is_zero_at_truth", "preflight_guard_is_vacuous")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
