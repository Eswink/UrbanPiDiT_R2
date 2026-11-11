"""Zero-GPU feasibility probe for Stage R: is a coherence-derived amplitude correction well posed?

Stage R's pre-condition (plan 0030 section 3.3) is that Stage P1 showed ``amp_ratio ~= coh``.
P1 measured ``mean|amp - coh| = 0.037-0.048`` on the frozen five leads, which is compatibility
rather than equality, so before spending 5-15 GPU-h on the full correction this probe asks the
three questions that decide whether the correction can work at all. All three are CPU-only and
read the **2022 validation** split; the 2023 test split is never opened.

1. **Is the per-band amplitude deficit stable across windows?** A correction is a property of
   the model, so the deficit it inverts must not swing wildly from window to window. Measured
   as the spread of ``amp_ratio`` across validation windows, per band.
2. **Does the deficit have the sign the correction assumes?** If ``amp_ratio > 1`` in a band
   (the model overshoots), dividing by it amplifies error rather than fixing it. P1 already saw
   overshoot concentrated in band 0, so the per-band sign is counted rather than assumed.
3. **How big is the correction, and does it stay bounded?** The multiplier is
   ``1 / amp_ratio`` per band. If that multiplier falls outside a plausible range the correction
   is extrapolating, and an unbounded multiplier would make the "does not degrade" exit a
   matter of luck rather than of method.

The probe writes no protocol and opens no round; it reports what the correction would do so the
R decision is made on numbers instead of on the plan's expectation.

    .venv/bin/python scripts/r7_s3_amplitude_correction_feasibility.py --out /tmp/r_feas.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The p1600 endpoint whose spectra Stage R would correct.
CHECKPOINT = ("outputs/r7_s3_throughput_physics_b_p1600_20261106_attempt01/"
              "seed41/training/update_0128528.pt")

#: The registered D2 denominator whose `n_selected_steps` the correction must not change.
D2_REFERENCE = ROOT / "outputs/r7_s3_fy_year_coverage_d2_20261023_attempt01/reference.json"

#: How many validation windows to read. The full split is 1,458; a feasibility probe needs
#: enough to see the spread and not enough to spend an hour on zarr reads.
DEFAULT_WINDOWS = 12

REFUSED_WORKERS = ("prepare", "archive", "seed", "seeding", "reading")


def _val_dataset(curve, limit):
    """The **validation** dataset through the audited evaluation entry point.

    ``ZarrLongRolloutDataset`` refuses any manifest whose name is not ``train.jsonl`` -- a
    deliberate guard, since that reader is the training path. Reading validation therefore goes
    through ``EvaluationPreparation``, which validates the held-out manifest against the store
    and asserts a single ``val`` (or ``test``) split. Using it here keeps this probe on the same
    reader the scored evaluation uses, so a spectral number and a skill number cannot come from
    two different definitions of a validation window.

    ``VAL_MANIFEST`` only: the 2023 test split is never opened (``r = 0``).
    """
    from training.r7_evaluate import EvaluationPreparation

    prep = EvaluationPreparation(curve.VAL_MANIFEST, step_hours=6)
    if prep.splits != {"val"}:
        raise ValueError(f"expected the validation split only, got {prep.splits}")
    dataset = prep.dataset((72,))
    return dataset, min(limit, len(dataset))


def measure(windows: int) -> dict:
    """Per-band amplitude ratio across validation windows, plus the implied multiplier."""
    import numpy as np
    import torch

    from scripts import r7_s3_curve_point_large_protocol as curve
    from training import r7_spectral_geometry as geometry
    from training.r7_experiment import load_checkpoint, make_model
    from model.r7_rollout import autoregressive_rollout, rollout_model_input

    checkpoint = ROOT / CHECKPOINT
    if not checkpoint.exists():
        raise FileNotFoundError(f"the p1600 endpoint is missing: {checkpoint}")
    saved = load_checkpoint(checkpoint)
    if int(saved["updates"]) != 128528:
        raise ValueError(f"the endpoint is at {saved['updates']} updates, not 128,528")
    with torch.random.fork_rng(devices=[]):
        model = make_model("process", curve.model_spec())
    model.load_state_dict(saved["model"], strict=True)
    # This is the **evaluation** path and ``autoregressive_rollout`` requires ``model.eval()``
    # -- the opposite of ``training_long_rollout``, which refuses an eval-mode model. Using the
    # evaluation rollout matters here beyond the guard: Stage R is a post-hoc correction of
    # *deployed* forecasts, so the spectra must be the eval-mode ones. The frozen spec's
    # dropout is 0.0 and the trunk has no BatchNorm (asserted below), so the two modes agree
    # numerically -- but the reading is taken the way the metric is taken.
    spec = curve.model_spec()
    if float(spec.get("dropout", -1.0)) != 0.0:
        raise ValueError(f"frozen spec dropout is {spec.get('dropout')!r}, not 0.0")
    normalising = [f"{name}: {type(module).__name__}"
                   for name, module in model.named_modules()
                   if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    if normalising:
        raise ValueError(f"the trunk has batch-dependent normalisation: {normalising}")
    model.eval()
    dataset, count = _val_dataset(curve, windows)
    edges = geometry.band_edges(65, 65)
    per_window = []
    for index in range(count):
        sample = dataset[index]
        initial = rollout_model_input(sample, lead_hours=float(sample["lead_time_hours"]),
                                      device=torch.device("cpu"))
        trajectory = autoregressive_rollout(model, initial, lead_hours=list(dataset.lead_hours),
                                            step_hours=6, history_interval_hours=6,
                                            inference_kwargs={"reasoning_steps": curve.REASONING_STEPS})
        forecast = trajectory.forecasts[0, -1].detach()
        target = sample["rollout_targets"][-1]
        rows = geometry.band_spectra(forecast, target, edges)["bands"]
        per_window.append({"index": index, "bands": rows})
    return {"windows": count, "per_window": per_window,
            "band_edges": edges,
            "model_code_sha256": saved.get("model_code_sha256")}


def summarise(report: dict) -> dict:
    """The three feasibility questions, answered per band from the window set."""
    import numpy as np

    bands = len(report["per_window"][0]["bands"])
    rows = []
    for band in range(bands):
        ratios = [window["bands"][band]["amp_ratio"] for window in report["per_window"]]
        finite = [value for value in ratios if value == value]
        if not finite:
            rows.append({"band": band, "windows": 0})
            continue
        mean = float(np.mean(finite))
        rows.append({
            "band": band,
            "windows": len(finite),
            "amp_ratio_mean": mean,
            "amp_ratio_std": float(np.std(finite)) if len(finite) > 1 else 0.0,
            "amp_ratio_min": float(min(finite)),
            "amp_ratio_max": float(max(finite)),
            "overshoot_windows": int(sum(1 for value in finite if value > 1.0)),
            # The multiplier the correction would apply: 1 / amp_ratio.
            "multiplier": (1.0 / mean) if mean > 0 else None,
            "multiplier_bounded": bool(mean > 0.0 and 0.2 <= 1.0 / mean <= 5.0),
        })
    total_windows = sum(row["windows"] for row in rows)
    overshoot = sum(row["overshoot_windows"] for row in rows)
    return {
        "per_band": rows,
        "windows_total_band_samples": total_windows,
        "overshoot_band_samples": overshoot,
        "bands_with_stable_ratio": int(sum(1 for row in rows
                                          if row.get("amp_ratio_std", 1e9) < 0.15)),
        "bands_with_bounded_multiplier": int(sum(1 for row in rows
                                                if row.get("multiplier_bounded"))),
        "band_count": bands,
    }


def run(windows: int) -> dict:
    report = measure(windows)
    return {"checkpoint_updates": 128528, **report, "feasibility": summarise(report)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=pathlib.Path, required=False)
    parser.add_argument("--windows", type=int, default=DEFAULT_WINDOWS)
    parser.add_argument("--worker", default=None,
                        help="refused: Stage R's feasibility probe writes no protocol")
    args = parser.parse_args(argv)
    if args.worker is not None:
        if args.worker in REFUSED_WORKERS:
            raise SystemExit(
                f"Stage R feasibility refuses --worker {args.worker!r}: a feasibility probe "
                "writes no protocol.json and no archive (decision 0110 / 0094 precedent).")
        raise SystemExit(f"unknown worker {args.worker!r}")

    reference = json.loads(D2_REFERENCE.read_text(encoding="utf-8"))
    record = {"scientific_claim": False, "test_read": False, "r": 0,
              "probe": "stage-r amplitude correction feasibility",
              "d2_reference": {"n_selected_steps": reference["n_selected_steps"],
                               "path": str(D2_REFERENCE.relative_to(ROOT))},
              "split_read": "2022 validation only; the 2023 test manifest was not opened"}
    failed = False
    try:
        record.update(run(args.windows))
    except Exception as exc:  # noqa: BLE001 - record, never raise, so the verdict is written
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback_tail"] = traceback.format_exc().strip().splitlines()[-8:]
        failed = True
    text = json.dumps(record, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
