"""Retrospective error/update geometry; never used to modify a forecast."""
from __future__ import annotations
import json
from pathlib import Path
import torch


@torch.no_grad()
def correction_terms(before, after, target, latitude=None, previous_update=None):
    """Per-case, per-channel weighted terms for e'=e+d, in normalized units.

    Returned oracle damping uses future truth and is NON-DEPLOYABLE. Undefined
    cosines (zero vector) are encoded as zero with explicit defined masks.
    """
    inputs = (before, after, target)
    if any(not isinstance(x, torch.Tensor) or x.ndim != 4 or
           not x.is_floating_point() or min(x.shape) < 1 for x in inputs):
        raise ValueError("nonempty floating [B,C,H,W] tensors required")
    if any(x.shape != before.shape or x.device != before.device for x in inputs):
        raise ValueError("field shapes/devices must match exactly")
    if any(not torch.isfinite(x).all() for x in inputs):
        raise ValueError("nonfinite input field")
    b, _, h, _ = before.shape
    e = (before.double() - target.double()).cpu()
    d = (after.double() - before.double()).cpu()
    if latitude is None:
        weight = torch.ones(b, 1, h, 1, dtype=torch.float64)
    else:
        lat = torch.as_tensor(latitude, dtype=torch.float64).cpu()
        if lat.ndim == 1 and len(lat) == h:
            lat = lat[None].expand(b, -1)
        if lat.shape != (b, h) or not torch.isfinite(lat).all() or (lat.abs() > 90).any():
            raise ValueError("latitude must be finite [H] or [B,H] in [-90,90]")
        weight = torch.cos(torch.deg2rad(lat))[:, None, :, None]
        if (weight.mean((2, 3)) <= 1e-12).any():
            raise ValueError("zero supported latitude area")
        weight = weight / weight.mean((2, 3), keepdim=True)

    def mean(x):
        return (x * weight).mean((-2, -1))

    mse0, mse1, energy, dot = mean(e*e), mean((e+d)**2), mean(d*d), mean(e*d)
    if any(not torch.isfinite(x).all() for x in (mse0, mse1, energy, dot)):
        raise ValueError("diagnostic arithmetic overflow")
    alpha = torch.where(energy > 0, -dot / energy, torch.zeros_like(energy)).clamp(0, 1)
    denom = mse0.sqrt() * energy.sqrt()
    cosine = torch.where(denom > 0, dot / denom, torch.zeros_like(dot)).clamp(-1, 1)
    oracle_mse = mean((e + alpha[:, :, None, None]*d)**2)
    result = dict(mse_before=mse0, mse_after=mse1, mse_change=mse1-mse0,
                  cross_term=2*dot, update_energy=energy,
                  algebra_residual=(mse1-mse0)-(2*dot+energy),
                  error_update_cosine=cosine, error_cosine_defined=denom > 0,
                  retrospective_damping=alpha, retrospective_damped_mse=oracle_mse,
                  worsening=mse1 > mse0, wrong_direction=(dot >= 0) & (energy > 0),
                  overshoot=(dot < 0) & (mse1 > mse0), zero_update=energy == 0)
    if previous_update is not None:
        if (not isinstance(previous_update, torch.Tensor) or
            previous_update.shape != before.shape or previous_update.device != before.device or
            not previous_update.is_floating_point() or not torch.isfinite(previous_update).all()):
            raise ValueError("finite previous update with matching shape/device required")
        previous = previous_update.double().cpu()
        prior_energy, cross = mean(previous**2), mean(previous*d)
        if not torch.isfinite(prior_energy).all() or not torch.isfinite(cross).all():
            raise ValueError("previous-update arithmetic overflow")
        norm = prior_energy.sqrt() * energy.sqrt()
        result["previous_update_cosine"] = torch.where(norm > 0, cross/norm, torch.zeros_like(cross)).clamp(-1, 1)
        result["previous_cosine_defined"] = norm > 0
    return result


@torch.no_grad()
def collect_correction_terms(model, batch, *, max_steps=3):
    from model.r7_halting import forecast_inputs, positive_int
    positive_int(max_steps, "max_steps")
    if max_steps > 8:
        raise ValueError("bounded correction diagnostic requires K<=8")
    if any(m.training for m in model.modules()):
        raise ValueError("correction diagnostic requires eval mode")
    # Future labels and externally supplied baselines are NEVER passed to model.
    out = model(forecast_inputs(batch), reasoning_steps=max_steps)
    drafts = out.draft_forecasts
    if drafts.ndim != 5 or drafts.shape[1] != max_steps+1:
        raise ValueError("full K0..K trajectory required")
    results, previous = [], None
    for k in range(max_steps):
        results.append(correction_terms(drafts[:, k], drafts[:, k+1],
            batch["atmos_target"], batch.get("latitude"), previous))
        previous = drafts[:, k+1] - drafts[:, k]
    return results


def run_correction_diagnostic(manifest, *, checkpoint, output, max_steps=3, max_samples=24):
    from torch.utils.data import default_collate
    from data.r7_zarr_dataset import ZarrAtmosWindowDataset
    from training.r7_experiment import dataset_identity, load_checkpoint, make_model
    from training.r7_calibration_runner import file_sha256, state_digest
    from model.r7_halting import positive_int
    positive_int(max_steps, "max_steps"); positive_int(max_samples, "max_samples")
    if max_steps > 8 or max_samples > 128:
        raise ValueError("bounded diagnostic requires K<=8 and cases<=128")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    ds = ZarrAtmosWindowDataset(manifest)
    if {r["split"] for r in ds.records} != {"val"}:
        raise ValueError("correction diagnostic is validation-only")
    identity, _ = dataset_identity(Path(manifest).parent/"train.jsonl")
    saved = load_checkpoint(checkpoint)
    if saved["contract"]["kind"] != "process" or saved["contract"]["data_identity"] != identity:
        raise ValueError("matching process checkpoint/training identity required")
    model = make_model("process", saved["contract"]["model"]).eval()
    model.load_state_dict(saved["model"], strict=True)
    digest_before, checkpoint_before = state_digest(model), file_sha256(checkpoint)
    channels = list(ds._store(ds.records[0]).attrs["channels"])
    rounds = [[] for _ in range(max_steps)]
    ids = []
    for i in range(min(len(ds), max_samples)):
        values = collect_correction_terms(model, default_collate([ds[i]]), max_steps=max_steps)
        for bucket, terms in zip(rounds, values):
            bucket.append({k:v.tolist()[0] for k,v in terms.items()})
        ids.append(ds.records[i]["init_time"])
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate diagnostic cases")
    summary = []
    for k, bucket in enumerate(rounds, 1):
        for c, name in enumerate(channels):
            row = dict(round=k, variable=name, n_cases=len(ids))
            for metric in bucket[0]:
                vals = torch.tensor([x[metric][c] for x in bucket], dtype=torch.float64)
                row[metric+"_mean"] = float(vals.mean())
            summary.append(row)
    if state_digest(model) != digest_before or file_sha256(checkpoint) != checkpoint_before:
        raise RuntimeError("diagnostic mutated original model/checkpoint")
    result = dict(format="r7-correction-diagnostic-v1", scientific_claim=False, deployable=False,
        future_labels_used_retrospectively=True, forecast_modified=False, training_executed=False,
        device="cpu", checkpoint_sha256=checkpoint_before, training_identity=identity,
        validation_manifest_sha256=file_sha256(manifest), initializations=ids, channels=channels,
        max_steps=max_steps, cases_by_round=rounds, summary=summary,
        metric_units="per-variable training-normalized squared error; not mixed physical units",
        limitations=["oracle damping uses truth and is never applied to the model",
                     "wrong-direction and overshoot are algebraic descriptions, not causal explanations",
                     "zero-vector cosines are zero-coded; inspect defined-mask rates",
                     "one-step validation trajectory; not independent weather events or 72h skill"])
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    return result
