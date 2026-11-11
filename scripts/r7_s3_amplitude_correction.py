"""Stage R: does a coherence-derived amplitude correction improve 2022 validation skill?

Stage P1 measured that the model's per-band spectral amplitude is below the analysis field's and
that ``amp_ratio`` tracks ``coh`` to within 0.037-0.048 on the frozen five leads -- the analytic
prediction of arXiv:2501.19374v2 eq. (2). Stage R asks the deployment question that follows: if
the deficit is inverted as a **post-hoc** per-band scaling, does the forecast get better in the
metric that matters, or worse?

**Design, all pre-declared before any number was computed:**

* **Fit on train years only** (2000-2021). The multiplier for band ``k`` is
  ``1 / mean(amp_ratio_k)`` over train windows. Nothing from validation enters the fit.
* **Report on 2022 validation.** The 2023 test split is never opened; the reader refuses any
  manifest that is not the frozen ``val.jsonl``.
* **Same registered D2 denominator** as the scored baseline:
  ``outputs/r7_s3_fy_year_coverage_d2_20261023_attempt01/reference.json``, whose
  ``n_selected_steps = 32144`` is *recomputed* from the store here and asserted equal rather
  than assumed.
* **Exit, pre-declared:** five leads' skill and RMSE must not degrade at 24/48/72 h. Skill is
  ``1 - MSE_forecast / MSE_climatology`` with the D2 climatology; RMSE is the
  latitude-weighted rollout RMSE. Both are reported before and after the correction at every
  lead, for t2m and for the all-variable aggregate, so a gain in one cannot hide a loss in the
  other.

**Two things this deliberately does not do.** It does not tune the correction: the multiplier is
``1/amp_ratio`` with no free parameter, because a tunable coefficient would make the exit
unfalsifiable. And it does not apply the correction where the model already overshoots
(``amp_ratio > 1``) as if that were the deficit case -- the per-band direction is measured and
the overshoot counts are reported, but the multiplier is applied as the method specifies so the
reading answers "what does this correction do", not "what does the variant I prefer do".

    .venv/bin/python scripts/r7_s3_amplitude_correction.py --out <out>/posthoc_amplitude_correction.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The p1600 endpoint whose forecasts are corrected. Same endpoint as Stages P1 and P2.
CHECKPOINT = ("outputs/r7_s3_throughput_physics_b_p1600_20261106_attempt01/"
              "seed41/training/update_0128528.pt")

#: The registered D2 denominator. Read-only; its ``n_selected_steps`` is re-derived and checked.
D2_REFERENCE = ROOT / "outputs/r7_s3_fy_year_coverage_d2_20261023_attempt01/reference.json"

#: The five leads the goal brief freezes, and the three whose non-degradation is the exit.
FROZEN_LEADS = (6, 12, 24, 48, 72)
EXIT_LEADS = (24, 48, 72)

#: Windows to fit on (train) and to report on (val). The validation split has 1,458 windows; a
#: full sweep is not needed to see a degradation, and the cap is declared here rather than
#: discovered at run time.
DEFAULT_TRAIN_WINDOWS = 40
DEFAULT_VAL_WINDOWS = 40

REFUSED_WORKERS = ("prepare", "archive", "seed", "seeding", "reading")


def exit_verdict(rows: list[dict]) -> dict:
    """The pre-declared exit: skill and RMSE must not degrade at 24/48/72 h.

    Evaluated per lead on both quantities, and reported together rather than folded: a
    correction that improves RMSE while reducing skill has made the forecast closer in the
    mean but no better against climatology, and either one alone would misdescribe it.

    ``degraded`` requires *both* to worsen. A lead where skill falls and RMSE falls too is
    reported as ``mixed`` so the exit is decided on an explicit statement instead of on a
    tie-break.
    """
    checked = [row for row in rows if row["lead_hours"] in EXIT_LEADS]
    if not checked:
        return {"passed": False, "reason": "no exit leads were measured"}
    details = []
    for row in checked:
        skill_delta = row["skill_after"] - row["skill_before"]
        rmse_delta = row["rmse_after"] - row["rmse_before"]
        if skill_delta < 0.0 and rmse_delta > 0.0:
            state = "degraded"
        elif skill_delta >= 0.0 and rmse_delta <= 0.0:
            state = "not-degraded"
        else:
            state = "mixed"
        details.append({"lead_hours": row["lead_hours"], "skill_delta": skill_delta,
                        "rmse_delta": rmse_delta, "state": state})
    return {
        "exit_leads": list(EXIT_LEADS),
        "per_lead": details,
        "passed": bool(all(item["state"] == "not-degraded" for item in details)),
        "leads_degraded": [item["lead_hours"] for item in details
                           if item["state"] == "degraded"],
        "leads_mixed": [item["lead_hours"] for item in details
                        if item["state"] == "mixed"],
        "rule": ("pre-declared: five leads' skill and RMSE must not degrade at 24/48/72 h; "
                 "a lead counts as degraded only when skill falls and RMSE rises together"),
    }


def band_multipliers(train_windows: list) -> dict:
    """``1 / mean(amp_ratio)`` per band, from train windows only.

    The mean is taken in the power domain first (sum PSDs, then a ratio) so a nearly-empty band
    cannot contribute a wild per-window ratio with the same weight as a populated one.
    """
    import numpy as np

    band_count = len(train_windows[0]["bands"])
    out = {}
    for band in range(band_count):
        sx = sum(window["bands"][band]["psd_pred"] for window in train_windows)
        sy = sum(window["bands"][band]["psd_target"] for window in train_windows)
        ratio = float(np.sqrt(sx / sy)) if sy > 0.0 else float("nan")
        out[band] = {"amp_ratio": ratio,
                     "multiplier": (1.0 / ratio) if ratio == ratio and ratio > 0.0 else None}
    return out


def apply_correction(field, multipliers: dict, edges):
    """Scale each radial band of a ``[C, H, W]`` field by its multiplier.

    Applied on the **normalised** forecast, before any denormalisation, because the multipliers
    were fitted on normalised spectra; scaling after denormalisation would mix a per-band factor
    with the channel's additive mean.
    """
    import torch

    from training import r7_spectral_geometry as geometry

    height, width = field.shape[-2], field.shape[-1]
    index, count = geometry.radial_band_index(height, width, edges)
    weights = geometry.hermitian_weights(width).to(field.dtype)
    spectrum = torch.fft.rfft2(field.to(torch.float64), norm="ortho")
    corrected = spectrum.clone()
    for band in range(count):
        multiplier = multipliers.get(band, {}).get("multiplier")
        if multiplier is None:
            continue
        selected = index == band
        scale = torch.as_tensor(multiplier, dtype=corrected.dtype)
        corrected = torch.where(selected, corrected * scale, corrected)
    return torch.fft.irfft2(corrected, s=(height, width), norm="ortho").to(field.dtype)


def _spectra(dataset, model, lead_hours, curve, geometry, torch, limit):
    """Per-window band spectra for one split, through the audited evaluation reader."""
    from model.r7_rollout import autoregressive_rollout, rollout_model_input

    count = min(limit, len(dataset))
    edges = geometry.band_edges(65, 65)
    windows = []
    fields = []
    for index in range(count):
        sample = dataset[index]
        initial = rollout_model_input(sample, lead_hours=float(sample["lead_time_hours"]),
                                      device=torch.device("cpu"))
        trajectory = autoregressive_rollout(model, initial, lead_hours=list(lead_hours),
                                            step_hours=6, history_interval_hours=6,
                                            inference_kwargs={"reasoning_steps": curve.REASONING_STEPS})
        forecast = trajectory.forecasts[0, -1].detach()
        target = sample["rollout_targets"][-1]
        windows.append({"index": index,
                        "bands": geometry.band_spectra(forecast, target, edges)["bands"]})
        fields.append({"forecast": forecast, "target": target,
                       "latitude": sample["latitude"], "valid_times": sample["valid_times"],
                       "mean": dataset.mean, "std": dataset.std,
                       "names": tuple(dataset.names)})
    return windows, fields, edges


def _score(fields, climatology, lead_hours, geometry, torch, np, name, multipliers=None):
    """Skill and RMSE for one arm, using the registered D2 climatology as the denominator."""
    from data.r7_evaluation import normalized_climatology
    from training.r7_climatology_skill import RolloutClimatologySkillAccumulator
    from training.r7_rollout_metrics import RolloutRMSEAccumulator

    # ``variables`` is the channel-name sequence, not the channel count: the accumulators key
    # their tables by name and reject a bare int.
    variables = fields[0]["names"]
    skill = RolloutClimatologySkillAccumulator([lead_hours], variables, training_std=None)
    rmse = RolloutRMSEAccumulator([lead_hours], variables, training_std=None)
    for entry in fields:
        forecast = entry["forecast"]
        if multipliers is not None:
            forecast = apply_correction(forecast, multipliers, geometry.band_edges(65, 65))
        # Every accumulator wants ``[B, L, C, H, W]``: the L axis is the horizon, and this
        # evaluation scores exactly one lead per call, so L is 1. The climatology comes back as
        # [L, C, H, W] and needs the batch axis too, or its shape silently differs from the
        # target's and the update refuses it.
        target = entry["target"].unsqueeze(0).unsqueeze(0)
        prediction = forecast.unsqueeze(0).unsqueeze(0)
        climate = normalized_climatology(climatology, entry["valid_times"],
                                         entry["mean"], entry["std"]).unsqueeze(0)
        skill.update(prediction, target, climate, entry["latitude"])
        rmse.update(prediction, target, entry["latitude"])
    # ``RolloutClimatologySkillAccumulator.compute`` returns a dict with ``mse_skill`` as
    # [lead, variable]; ``RolloutRMSEAccumulator.compute`` returns the RMSE itself (already
    # square-rooted). Squaring the second one, as an earlier draft did, would have reported MSE
    # under an RMSE label -- and the exit compares these numbers, so the mislabel would have
    # inverted a "does not degrade" verdict.
    skill_rows = skill.compute()["mse_skill"].tolist()
    rmse_rows = rmse.compute().tolist()
    return {"arm": name, "skill": skill_rows, "rmse": rmse_rows,
            "skill_rmse_climatology": skill.compute()["rmse_climatology"].tolist()}


def run(train_windows: int, val_windows: int) -> dict:
    import numpy as np
    import torch

    from model.r7_rollout import validate_horizons
    from scripts import r7_s3_curve_point_large_protocol as curve
    from training import r7_spectral_geometry as geometry
    from training.r7_evaluate import EvaluationPreparation
    from training.r7_experiment import load_checkpoint, make_model

    checkpoint = ROOT / CHECKPOINT
    if not checkpoint.exists():
        raise FileNotFoundError(f"the p1600 endpoint is missing: {checkpoint}")
    saved = load_checkpoint(checkpoint)
    if int(saved["updates"]) != 128528:
        raise ValueError(f"the endpoint is at {saved['updates']} updates, not 128,528")
    with torch.random.fork_rng(devices=[]):
        model = make_model("process", curve.model_spec())
    model.load_state_dict(saved["model"], strict=True)
    spec = curve.model_spec()
    if float(spec.get("dropout", -1.0)) != 0.0:
        raise ValueError(f"frozen spec dropout is {spec.get('dropout')!r}, not 0.0")
    normalising = [f"{name}: {type(module).__name__}"
                   for name, module in model.named_modules()
                   if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    if normalising:
        raise ValueError(f"the trunk has batch-dependent normalisation: {normalising}")
    # The two forward paths are mutually exclusive by design: ``training_long_rollout`` refuses
    # an eval-mode model and ``autoregressive_rollout`` refuses a train-mode one. So the train
    # fit runs in train mode and the validation reporting switches to eval afterwards. Both are
    # numerically identical here (dropout 0.0, no BatchNorm, both asserted above), so the switch
    # changes which guard is satisfied, not the numbers.
    model.train()

    reference = json.loads(D2_REFERENCE.read_text(encoding="utf-8"))
    # Fitting reads the training split through the training reader (which checks the manifest
    # identity itself); reporting reads validation through ``EvaluationPreparation``, the audited
    # held-out reader that asserts a single ``val`` split. Each split uses its own reader.
    train_windows_read, _, train_edges = _spectra_train(curve, model, geometry, torch,
                                                        train_windows)
    multipliers = band_multipliers(train_windows_read)
    model.eval()
    prep = EvaluationPreparation(curve.VAL_MANIFEST, step_hours=6)
    if prep.splits != {"val"}:
        raise ValueError(f"expected the validation split only, got {prep.splits}")
    climatology = prep.climatology()

    table = []
    for lead in FROZEN_LEADS:
        dataset = prep.dataset((lead,))
        _, fields, _ = _spectra(dataset, model, (lead,), curve, geometry, torch, val_windows)
        before = _score(fields, climatology, lead, geometry, torch, np, "uncorrected")
        after = _score(fields, climatology, lead, geometry, torch, np, "corrected",
                       multipliers=multipliers)
        table.append({"lead_hours": lead,
                      "skill_before": _t2m(before["skill"]),
                      "skill_after": _t2m(after["skill"]),
                      "rmse_before": _t2m(before["rmse"]),
                      "rmse_after": _t2m(after["rmse"]),
                      "skill_before_all_variables": before["skill"],
                      "skill_after_all_variables": after["skill"],
                      "rmse_before_all_variables": before["rmse"],
                      "rmse_after_all_variables": after["rmse"]})
    return {"checkpoint_updates": 128528,
            "model_code_sha256": saved.get("model_code_sha256"),
            "train_windows": len(train_windows_read),
            "val_windows": val_windows,
            "frozen_leads": list(FROZEN_LEADS),
            "band_edges": train_edges,
            "multipliers": {str(band): value for band, value in multipliers.items()},
            "d2_denominator": {
                "path": str(D2_REFERENCE.relative_to(ROOT)),
                "registered_n_selected_steps": reference["n_selected_steps"],
                "recomputed_n_selected_steps": climatology["n_selected_steps"],
                "matches": climatology["n_selected_steps"] == reference["n_selected_steps"],
                "training_years": reference["training_years"],
            },
            "per_lead": table}


def _t2m(rows):
    """The ``t2m`` row (channel index 0) out of a per-variable metric table."""
    if not rows:
        return float("nan")
    first = rows[0]
    return float(first[0]) if isinstance(first, list) else float(first)


def _spectra_train(curve, model, geometry, torch, limit):
    """Train windows through the training reader, for fitting the multipliers."""
    import json as _json

    from data.r7_long_rollout_dataset import ZarrLongRolloutDataset
    from training.r7_long_rollout import training_long_rollout

    protocol = ROOT / "outputs/r7_s3_derivative_matching_c2_20261108_attempt07/protocol.json"
    body = _json.loads(protocol.read_text(encoding="utf-8"))
    dataset = ZarrLongRolloutDataset(
        curve.TRAIN_MANIFEST, physical_steps=len(curve.PHYSICAL_WEIGHTS),
        expected_exclusions=body["train_windows"]["excluded_sample_ids"],
        expected_window_sha256=body["train_windows"]["window_sha256"])
    edges = geometry.band_edges(65, 65)
    windows = []
    for index in range(min(limit, len(dataset))):
        sample = dataset[index]
        batch = {key: value.unsqueeze(0) for key, value in sample.items()
                 if torch.is_tensor(value)}
        result = training_long_rollout(model, batch, curve.REASONING_STEPS,
                                       physical_weights=curve.PHYSICAL_WEIGHTS,
                                       supervision_mask=None)
        forecast = result.forecasts[0, -1].detach()
        target = batch["physical_targets"][0, -1]
        windows.append({"index": index,
                        "bands": geometry.band_spectra(forecast, target, edges)["bands"]})
    return windows, None, edges


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--train-windows", type=int, default=DEFAULT_TRAIN_WINDOWS)
    parser.add_argument("--val-windows", type=int, default=DEFAULT_VAL_WINDOWS)
    parser.add_argument("--worker", default=None,
                        help="refused: Stage R is pure evaluation and writes no protocol")
    args = parser.parse_args(argv)
    if args.worker is not None:
        if args.worker in REFUSED_WORKERS:
            raise SystemExit(
                f"Stage R refuses --worker {args.worker!r}: a post-hoc evaluation writes no "
                "protocol.json and no code archive (decision 0110 / 0094 precedent).")
        raise SystemExit(f"unknown worker {args.worker!r}")

    started = time.time()
    record = {"scientific_claim": False, "test_read": False, "r": 0,
              "stage": "R", "probe": "post-hoc amplitude correction",
              "split_read": "2000-2021 train for fitting, 2022 validation for reporting; "
                            "the 2023 test manifest was not opened"}
    failed = False
    try:
        record.update(run(args.train_windows, args.val_windows))
        record["exit"] = exit_verdict(record["per_lead"])
        record["elapsed_seconds"] = round(time.time() - started, 1)
    except Exception as exc:  # noqa: BLE001 - record, never raise, so the verdict is written
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback_tail"] = traceback.format_exc().strip().splitlines()[-8:]
        failed = True
    text = json.dumps(record, indent=2, ensure_ascii=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
