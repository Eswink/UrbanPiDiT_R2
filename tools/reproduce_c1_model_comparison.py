"""C1 section 2, model half: does the constraint bundle charge the analysis field more than the model?

Decision 0094 permanently closed Stage C1 on a measured claim: the divergence term charges the
analysis field 10.2x what it charges the model, 8/8 cases. ``tools/reproduce_c1_preflight.py`` made
the analysis half recomputable straight from the store. This script does the model half, which needs
a forward pass, on CPU so it cannot disturb a live GPU round.

It reproduces C1's own recipe (its result page §1): one rollout per case, take the last 6 h step,
denormalise to physical units, then run the *same* ``physical_constraint_terms`` on
(a) the prediction and (b) that case's ``physical_targets[:, -1]`` -- the same field the rollout was
scored against and the same one the page calls the analysis field.

Read-only with respect to the repository: writes only the path given by --out.

    python tools/reproduce_c1_model_comparison.py --cases 1000 --out /tmp/c1_one.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The eight window indices C1's result page lists in its per-case table.
C1_CASES = (1000, 5000, 9000, 13000, 17000, 21000, 25000, 29000)

#: The endpoint C1 measured: Stage B's fourth point, asserted by identity in ``_load``.
C1_CHECKPOINT = ("outputs/r7_s3_throughput_physics_b_p1600_20261106_attempt01/seed41/"
                 "training/update_0128528.pt")

#: What the page reports, for side-by-side comparison; never used to compute anything.
PAGE_CLAIM = {"model_total": 0.0829, "analysis_total": 0.8470, "ratio": 10.2,
              "model_divergence": 0.8286, "analysis_divergence": 8.4703}


def _load(protocol_path: Path):
    """Model, dataset and channel order, all identity-checked."""
    import torch

    from data.r7_long_rollout_dataset import ZarrLongRolloutDataset
    from scripts import r7_s3_curve_point_large_protocol as curve
    from training.r7_experiment import load_checkpoint, make_model

    body = json.loads(protocol_path.read_text(encoding="utf-8"))
    saved = load_checkpoint(ROOT / C1_CHECKPOINT)
    if saved["updates"] != 128528:
        raise SystemExit(f"checkpoint is at {saved['updates']} updates, not C1's 128,528")
    with torch.random.fork_rng(devices=[]):
        model = make_model("process", curve.model_spec())
    model.load_state_dict(saved["model"], strict=True)
    # ``training_long_rollout`` refuses an eval-mode model ("differentiable training requires
    # grad-enabled model.train()"), so this runs in train mode with grad on. That is not a
    # different forward pass here: the frozen spec's ``dropout`` is 0.0 and no module in this
    # trunk uses BatchNorm, so train and eval modes are numerically identical; the probe detaches
    # the forecast immediately and the guard below records the two facts the argument rests on.
    spec = curve.model_spec()
    if float(spec.get("dropout", -1.0)) != 0.0:
        raise SystemExit(f"frozen spec dropout is {spec.get('dropout')!r}, not 0.0: train-mode "
                         f"forward would not match the eval-mode one C1 measured")
    # Dropout is only half the train/eval difference. A normalisation layer whose statistics depend
    # on the batch would make this probe measure something other than C1's forward pass, so refuse
    # rather than assume: this walks the real module tree instead of trusting the argument.
    normalising = [f"{name}: {type(module).__name__}" for name, module in model.named_modules()
                   if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    if normalising:
        raise SystemExit("the trunk has batch-dependent normalisation, so train and eval modes "
                         f"are not equivalent and this probe would not reproduce C1: {normalising}")
    model.train()
    dataset = ZarrLongRolloutDataset(
        curve.TRAIN_MANIFEST, physical_steps=len(curve.PHYSICAL_WEIGHTS),
        expected_exclusions=body["train_windows"]["excluded_sample_ids"],
        expected_window_sha256=body["train_windows"]["window_sha256"])
    return model, dataset, curve, saved["model_code_sha256"]


def _denormalise(field, mean, std):
    """[B,C,H,W] normalized -> physical units, per channel."""
    import torch

    shape = (1, mean.numel(), 1, 1)
    return field * std.view(shape) + mean.view(shape)


def _measure_case(model, dataset, curve, case, stats) -> dict:
    """One rollout for one window, then the same constraint bundle on three target pairings."""
    import numpy as np
    import torch
    from torch.utils.data import default_collate

    from training.r7_long_rollout import training_long_rollout
    from training.r7_physical_constraints import physical_constraint_terms

    channels, mean, std, latitude, longitude = stats
    started = time.time()
    batch = default_collate([dataset[case]])
    # No ``no_grad`` here: the guard inside ``training_long_rollout`` rejects it. The forecast is
    # detached below, so no graph survives the measurement.
    result = training_long_rollout(model, batch, curve.REASONING_STEPS,
                                   physical_weights=curve.PHYSICAL_WEIGHTS,
                                   supervision_mask=None)
    prediction = _denormalise(result.forecasts[:, -1].detach(), mean, std)
    # Three "truth" fields. C1's page compared a +72 h forecast against the +6 h analysis field,
    # and the size of the claim depends on which target it is compared with, so this measures all
    # of them rather than picking one and arguing:
    #   * ``atmos_target``     -- the +6 h field named in ADR 0094's text. This is the pairing that
    #                             reproduces the page (10.2x), and it mixes two lead times.
    #   * ``72h``              -- the +72 h field this rollout's last step is actually scored
    #                             against, i.e. the lead-matched partner of the prediction.
    #   * ``lead_matched``     -- each forecast step against its OWN target, then averaged, so
    #                             every comparison is at one lead time.
    target_short = _denormalise(batch["atmos_target"], mean, std)
    target_long = _denormalise(batch["physical_targets"][:, -1], mean, std)
    # Identical function, identical names, identical coordinates: only the field differs.
    model_terms = physical_constraint_terms(prediction, channels, latitude, longitude)
    short_terms = physical_constraint_terms(target_short, channels, latitude, longitude)
    long_terms = physical_constraint_terms(target_long, channels, latitude, longitude)

    matched = _lead_matched(result.forecasts, batch["physical_targets"], stats, curve)
    row = {"case": case, "seconds": round(time.time() - started, 1), **matched}
    for name in ("divergence", "humidity", "dew_point", "total"):
        row[f"model_{name}"] = float(model_terms[name])
        row[f"analysis_atmos_target_{name}"] = float(short_terms[name])
        row[f"analysis_72h_{name}"] = float(long_terms[name])
    for label, terms in (("atmos_target", short_terms), ("72h", long_terms)):
        row[f"ratio_{label}"] = (float(terms["total"]) / row["model_total"]
                                 if row["model_total"] else None)
    return row


def _lead_matched(forecasts, physical, stats, curve) -> dict:
    """Every forecast step against its own target, then averaged over the N axis.

    ``physical_constraint_terms`` reads [..., C, H, W], so the N axis is stepped explicitly.
    ``forecasts`` is [B,N,C,H,W] and ``physical`` matches it step for step.
    """
    import numpy as np

    from training.r7_physical_constraints import physical_constraint_terms

    channels, mean, std, latitude, longitude = stats
    forecasts = forecasts.detach()
    model_total, truth_total, model_div, truth_div = [], [], [], []
    for step in range(forecasts.shape[1]):
        m = physical_constraint_terms(
            _denormalise(forecasts[:, step], mean, std), channels, latitude, longitude)
        t = physical_constraint_terms(
            _denormalise(physical[:, step], mean, std), channels, latitude, longitude)
        model_total.append(float(m["total"]))
        truth_total.append(float(t["total"]))
        # The headline claim is about the divergence term specifically, so match that too.
        model_div.append(float(m["divergence"]))
        truth_div.append(float(t["divergence"]))
    model_mean, truth_mean = float(np.mean(model_total)), float(np.mean(truth_total))
    model_div_mean, truth_div_mean = float(np.mean(model_div)), float(np.mean(truth_div))
    return {
        "lead_matched_model_total": model_mean,
        "lead_matched_analysis_total": truth_mean,
        "lead_matched_ratio": (truth_mean / model_mean) if model_mean else None,
        "lead_matched_model_divergence": model_div_mean,
        "lead_matched_analysis_divergence": truth_div_mean,
        "lead_matched_divergence_ratio": (truth_div_mean / model_div_mean)
                                         if model_div_mean else None,
    }


def _summarise(rows, cases, model_code) -> dict:
    """The report: every pairing's means and its agreement count, keyed so nothing is dropped."""
    import numpy as np

    report = {
        "checkpoint_updates": 128528,
        "model_code_sha256": model_code,
        # C1's own page reports an eval-mode rollout. This probe runs train mode because
        # ``training_long_rollout`` refuses eval mode; the equivalence rests on the frozen spec's
        # dropout being 0.0 and on the trunk having no BatchNorm (both checked, see ``_load``).
        "forward_mode": "train (dropout=0.0, no BatchNorm => identical to eval)",
        "cases": list(cases),
        "rows": rows,
        "mean_model_total": float(np.mean([r["model_total"] for r in rows])),
        "mean_model_divergence": float(np.mean([r["model_divergence"] for r in rows])),
        "page_claim": PAGE_CLAIM,
    }
    for label in ("atmos_target", "72h", "lead_matched"):
        # ``_lead_matched`` names its fields ``lead_matched_*`` while the other two pairings are
        # named ``*_atmos_target``/``*_72h``, so the two key shapes differ for exactly one label.
        # Building either name mechanically raises KeyError on that label -- the bug a refactor
        # introduced and ``test_summarise_consumes_the_row_shape`` now pins.
        matched = label == "lead_matched"
        ratio_key = "lead_matched_ratio" if matched else f"ratio_{label}"
        total_key = "lead_matched_analysis_total" if matched else f"analysis_{label}_total"
        ratios = [r[ratio_key] for r in rows if r[ratio_key] is not None]
        report[f"mean_analysis_{label}_total"] = float(np.mean([r[total_key] for r in rows]))
        report[f"mean_{label}_ratio"] = float(np.mean(ratios)) if ratios else None
        report[f"{label}_ratio_range"] = ([float(min(ratios)), float(max(ratios))]
                                          if ratios else None)
        report[f"all_cases_worse_{label}"] = all(r > 1.0 for r in ratios) if ratios else None
    report["lead_matched_model_total"] = float(
        np.mean([r["lead_matched_model_total"] for r in rows]))
    report["mean_lead_matched_model_divergence"] = float(
        np.mean([r["lead_matched_model_divergence"] for r in rows]))
    report["mean_lead_matched_analysis_divergence"] = float(
        np.mean([r["lead_matched_analysis_divergence"] for r in rows]))
    div_ratios = [r["lead_matched_divergence_ratio"] for r in rows
                  if r["lead_matched_divergence_ratio"] is not None]
    report["mean_lead_matched_divergence_ratio"] = float(np.mean(div_ratios)) if div_ratios else None
    report["all_cases_worse_lead_matched_divergence"] = (
        all(r > 1.0 for r in div_ratios) if div_ratios else None)
    return report


def run(protocol_path: Path, cases, out_path: Path) -> dict:
    """One rollout per case; the same constraint bundle on prediction and on three targets."""
    import numpy as np
    import torch
    import zarr

    model, dataset, curve, model_code = _load(protocol_path)
    store = zarr.open_group(str(curve.STORE), mode="r")
    stats = (list(store.attrs["channels"]),
             torch.as_tensor(np.asarray(store["normalization_mean"][:]), dtype=torch.float32),
             torch.as_tensor(np.asarray(store["normalization_std"][:]), dtype=torch.float32),
             np.asarray(store["latitude"][:]),
             np.asarray(store["longitude"][:]))
    rows = [_measure_case(model, dataset, curve, case, stats) for case in cases]
    report = _summarise(rows, cases, model_code)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "outputs/r7_s3_derivative_matching_c2_20261108_attempt07/protocol.json")
    parser.add_argument("--cases", type=int, nargs="*", default=list(C1_CASES))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.protocol, tuple(args.cases), args.out)
    print(json.dumps({k: report[k] for k in
                      ("mean_model_total", "mean_analysis_atmos_target_total",
                       "mean_atmos_target_ratio", "mean_72h_ratio",
                       "mean_lead_matched_ratio",
                       "all_cases_worse_atmos_target", "all_cases_worse_72h",
                       "all_cases_worse_lead_matched")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
