"""`dose = penalty / L_mse` for the J3 gate, computed on the registered denominator.

Stage P's J1 probe reports ``dose: null`` because it never built the loss the training path
uses. This script fills that gap without inventing anything:

* ``L_mse`` is ``training.r7_losses.latitude_weighted_mse`` -- the exact function
  ``deep_supervised_forecast_mse`` reduces over, on the **normalised** forecast/target pair the
  model is trained on.
* The constraint terms are made dimensionless by the **frozen** scales already declared in
  ``training/r7_physical_constraints.py`` (decision 0092/0094): ``DIVERGENCE_SCALE = 1e-5``
  and ``HUMIDITY_SCALE = 1e-3``. No new constant is introduced here.

**One direction cannot be dosed this way and that is reported, not patched.** Hydrostatic
balance has no declared scale: ``penalty`` is a mean square in metres and nothing in the
repository says what a "one scale unit" residual is. Dividing it by ``L_mse`` would mix m^2
with a normalised MSE, and inventing a height scale after seeing the number is exactly the
fitted-parameter move J3 exists to prevent. Its dose is therefore ``null`` with the reason
recorded, which means it can never pass J3 -- the honest outcome, not a workaround.

Read-only: writes only the path given by ``--out``.

    .venv/bin/python scripts/r7_s3_j1_dose_probe.py --out /tmp/dose.json
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

#: The same eight train cases the J1 gate used, so a dose can be held against a direction.
CASES = (1000, 5000, 9000, 13000, 17000, 21000, 25000, 29000)

#: The p1600 endpoint the J1 probe measured.
CHECKPOINT = ("outputs/r7_s3_throughput_physics_b_p1600_20261106_attempt01/"
              "seed41/training/update_0128528.pt")

#: J3's frozen band (plan 0029 section 2). Read, never redefined.
DOSE_BAND = (1e-3, 1e-1)

REFUSED_WORKERS = ("prepare", "archive", "seed", "seeding", "reading")


def dose_verdict(penalty: float, l_mse: float, *, band=DOSE_BAND) -> dict:
    """``dose = penalty / L_mse`` plus the frozen-band test, as one comparable row."""
    if not (l_mse == l_mse) or l_mse <= 0.0:
        return {"penalty": penalty, "l_mse": l_mse, "dose": None,
                "in_band": False, "reason": "L_mse is not a positive finite number"}
    dose = penalty / l_mse
    low, high = band
    return {"penalty": penalty, "l_mse": l_mse, "dose": dose,
            "in_band": bool(low <= dose <= high),
            "band": [low, high],
            "reason": "inside the frozen J3 band" if low <= dose <= high
            else "outside the frozen J3 band"}


def _dimensionless(analysis, prediction, latitude, longitude) -> dict:
    """The two candidates that have a declared scale, made dimensionless per decision 0094.

    ``divergence`` uses ``DIVERGENCE_SCALE`` **without** the dead-band: J3 asks whether the
    term's magnitude is a plausible fraction of the loss, and the dead-band is a modelling
    choice that would make an exactly-zero term read as "doseable". ``humidity`` is the
    non-negative violation at ``HUMIDITY_SCALE``.
    """
    import torch
    from model.r7_process_tensor_diagnostics import spherical_divergence
    from training.r7_physical_constraints import DIVERGENCE_SCALE, HUMIDITY_SCALE

    rows = {}
    for label, fields in (("analysis", analysis), ("model", prediction)):
        divergence = spherical_divergence(fields["u850"], fields["v850"], latitude, longitude)
        dimensionless_divergence = (divergence / DIVERGENCE_SCALE) ** 2
        negative = (fields["q850"] / HUMIDITY_SCALE).clamp_max(0.0) ** 2
        rows[label] = {"divergence": float(dimensionless_divergence.mean()),
                       "humidity": float(negative.mean())}
    return rows


def _assert_physical_units(fields: dict, normalised_forecast, batch) -> None:
    """Refuse a field whose magnitudes contradict the store's declared units.

    The failure this catches is a broadcast, not bad data: if the ``[C, H, W]`` block is
    sliced to ``[H, W]`` before the per-channel constants are applied, every entry becomes a
    17-channel blend dominated by ``mslp`` (~101635) and ``z250`` (~104149). The symptom is a
    ``t2m`` mean far above 340 K and a ``u850`` range far outside wind speed, both of which
    silently inflate every squared constraint.
    """
    t2m_mean = float(fields["t2m"].mean())
    if not 200.0 < t2m_mean < 340.0:
        raise ValueError(f"t2m averages {t2m_mean:.1f}, which is not Kelvin: the fields are "
                         "blended across channels or the normalisation direction is wrong")
    u_mean = float(fields["u850"].abs().mean())
    if not 0.0 < u_mean < 100.0:
        raise ValueError(f"|u850| averages {u_mean:.1f} m/s, which is not a wind speed")

def _load(curve, torch, np):
    """Model, store constants, dataset and the p1600 endpoint, all identity-checked."""
    import json as _json

    import zarr

    from data.r7_long_rollout_dataset import ZarrLongRolloutDataset
    from training.r7_experiment import load_checkpoint, make_model

    checkpoint = ROOT / CHECKPOINT
    if not checkpoint.exists():
        raise FileNotFoundError(f"the p1600 endpoint is missing: {checkpoint}")
    store = zarr.open_group(str(curve.STORE), mode="r")
    names = list(store.attrs["channels"])
    latitude = torch.as_tensor(np.asarray(store["latitude"][:]), dtype=torch.float64)
    longitude = torch.as_tensor(np.asarray(store["longitude"][:]), dtype=torch.float64)
    mean = torch.as_tensor(np.asarray(store["normalization_mean"][:]), dtype=torch.float64)
    std = torch.as_tensor(np.asarray(store["normalization_std"][:]), dtype=torch.float64)
    saved = load_checkpoint(checkpoint)
    if int(saved["updates"]) != 128528:
        raise ValueError(f"the endpoint is at {saved['updates']} updates, not 128,528")
    with torch.random.fork_rng(devices=[]):
        model = make_model("process", curve.model_spec())
    model.load_state_dict(saved["model"], strict=True)
    if float(curve.model_spec().get("dropout", -1.0)) != 0.0:
        raise ValueError("train-mode forward would differ from the deployed model")
    model.train()
    protocol = ROOT / "outputs/r7_s3_derivative_matching_c2_20261108_attempt07/protocol.json"
    body = _json.loads(protocol.read_text(encoding="utf-8"))
    dataset = ZarrLongRolloutDataset(
        curve.TRAIN_MANIFEST, physical_steps=len(curve.PHYSICAL_WEIGHTS),
        expected_exclusions=body["train_windows"]["excluded_sample_ids"],
        expected_window_sha256=body["train_windows"]["window_sha256"])
    return model, dataset, names, (mean, std), (latitude, longitude), saved


def run(cases) -> dict:
    """Per case: the registered L_mse and the dose of every dosed candidate."""
    import numpy as np
    import torch

    from scripts import r7_s3_curve_point_large_protocol as curve
    from training.r7_long_rollout import training_long_rollout
    from training.r7_losses import latitude_weighted_mse

    model, dataset, names, (mean, std), (latitude, longitude), saved = _load(curve, torch, np)
    shape = (len(names), 1, 1)

    def physical(fields):
        return fields.to(torch.float64) * std.reshape(shape) + mean.reshape(shape)

    rows = []
    for case in cases:
        sample = dataset[case]
        batch = {key: value.unsqueeze(0) for key, value in sample.items()
                 if torch.is_tensor(value)}
        result = training_long_rollout(model, batch, curve.REASONING_STEPS,
                                       physical_weights=curve.PHYSICAL_WEIGHTS,
                                       supervision_mask=None)
        forecast = result.forecasts[0, -1].detach()          # normalised, [C,H,W]
        target = batch["physical_targets"][0, -1]            # normalised, [C,H,W]
        # The registered denominator: the same function the training loss reduces over, on the
        # same normalised pair. Not a re-derivation and not a physical-unit variant.
        l_mse = float(latitude_weighted_mse(forecast, target, latitude))
        # ``physical`` applies a ``(C, 1, 1)`` constant vector, so it must receive the whole
        # ``[C, H, W]`` block: a bare ``[H, W]`` slice broadcasts into ``[17, H, W]`` and each
        # "channel" becomes a 17-channel blend. That bug inflated the divergence penalty by
        # 5.4e4 here -- the same mistake the J1 probe made and ``_assert_physical_units``
        # caught there, so it is asserted here too rather than trusted.
        analysis_block = physical(batch["physical_targets"][0, -1])   # [C, H, W]
        model_block = physical(result.forecasts[0, -1].detach())      # [C, H, W]
        analysis_fields = {name: analysis_block[index] for index, name in enumerate(names)}
        model_fields = {name: model_block[index] for index, name in enumerate(names)}
        _assert_physical_units(analysis_fields, result.forecasts[0, -1].detach(), batch)
        terms = _dimensionless(analysis_fields, model_fields, latitude, longitude)
        row = {"case": case, "l_mse_normalised": l_mse}
        for direction in ("divergence", "humidity"):
            row[direction] = {
                "analysis": dose_verdict(terms["analysis"][direction], l_mse),
                "model": dose_verdict(terms["model"][direction], l_mse),
            }
        # Hydrostatic has no declared scale: reported with its reason, never dosed.
        row["hydrostatic"] = {
            "dose": None,
            "reason": ("no scale constant is declared for the hydrostatic residual, so its "
                       "mean square is in m^2 and dividing by a normalised MSE would mix "
                       "units; inventing a height scale after seeing the number is the "
                       "fitted-parameter move J3 exists to prevent"),
        }
        rows.append(row)
    return {"checkpoint_updates": 128528,
            "model_code_sha256": saved.get("model_code_sha256"),
            "cases": list(cases), "dose_band": list(DOSE_BAND), "rows": rows}


def summarise(report: dict) -> dict:
    """Whether each dosed candidate lands inside J3's band, on the analysis side."""
    import numpy as np

    out = {}
    for direction in ("divergence", "humidity"):
        doses = [row[direction]["analysis"]["dose"] for row in report["rows"]
                 if row[direction]["analysis"]["dose"] is not None]
        in_band = [row[direction]["analysis"]["in_band"] for row in report["rows"]
                   if row[direction]["analysis"]["dose"] is not None]
        out[direction] = {
            "dose_mean": float(np.mean(doses)) if doses else None,
            "dose_range": [float(min(doses)), float(max(doses))] if doses else None,
            "cases_dosed": len(doses),
            "cases_in_band": int(sum(in_band)),
            "passes_j3_all_cases": bool(in_band) and all(in_band),
        }
    out["hydrostatic"] = {"dose": None, "passes_j3": False,
                          "reason": "no declared scale; cannot be dosed without fitting one"}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=pathlib.Path, required=False)
    parser.add_argument("--cases", type=int, nargs="+", default=list(CASES))
    parser.add_argument("--worker", default=None,
                        help="refused: Stage P has no prepare/archive/seed stage")
    args = parser.parse_args(argv)
    if args.worker is not None:
        if args.worker in REFUSED_WORKERS:
            raise SystemExit(
                f"Stage P refuses --worker {args.worker!r}: a pre-flight probe writes no "
                "protocol.json and no code archive (decision 0110 / decision 0094 precedent).")
        raise SystemExit(f"unknown worker {args.worker!r}")

    record = {"scientific_claim": False, "test_read": False, "r": 0,
              "probe": "stage-p2 j3 dose", "stage": "p2-dose"}
    failed = False
    try:
        report = run(tuple(args.cases))
        record.update(report)
        record["j3"] = summarise(report)
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
