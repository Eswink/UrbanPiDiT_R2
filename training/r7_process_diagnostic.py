"""#65 pre-diagnostic: targeted, read-only inspection of the process-forecast
recursion before any new module or ablation is added.

Three questions, one module, CPU-only, no training and no new GPU work:

1. **Proxy numeric scale.** The store normalizations floor the per-channel
   standard deviation at ``eps=1e-6``. For the two moisture proxies whose real
   spread is ~1e-8 that floor divides by 1e-6, so the normalized label the model
   is trained against collapses toward zero. This module reports the raw std,
   the floored std, whether the floor is active, the normalized range and the
   fraction of the label that survives as recoverable signal. It answers
   whether the process state's moisture channels are effectively constant
   *inputs* to the model, not just awkward labels.

2. **Auxiliary vs main gradient.** For a train-mode forward pass this records
   the gradient norm the forecast loss and the process loss each put on the
   shared parameters (backbone / recurrent cell / draft encoder / correction
   head) and on the process readout, plus the cosine between those two
   gradients. A negative cosine is *recorded as a diagnostic and never used as
   causal evidence* (#65 states this explicitly): two gradients pulling apart
   at one batch does not establish that the auxiliary task damages the
   forecast, and with a zero auxiliary weight the process gradient is exactly
   zero, so its cosine with the forecast gradient is undefined rather than
   informative.

3. **K0..K4 trajectory.** Reuses the #53 update geometry
   (``training.r7_correction_diagnostic.correction_terms``) so that error
   change, correction norm and adjacent-correction cosine come from one
   implementation rather than a second opinion. The trace stores scalars and
   per-channel summaries only; it never retains a full graph, so the streamed
   backward's memory advantage is not traded away for the trace.

Everything here is read-only with respect to the store and to any checkpoint:
the module refuses to write, never calls an optimizer step, and records the
model/checkpoint digests it observed so a later reader can tell which artifact
produced the numbers.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch

DEFAULT_EPS = 1e-6
# A label whose normalized spread is below this contributes so little to a
# unit-scale MSE that it is effectively a constant input for the optimizer.
UNUSABLE_NORMALIZED_STD = 0.05


def normalized_proxy_scale(values, eps: float = DEFAULT_EPS) -> dict:
    """Numeric-scale record for one proxy channel.

    ``values`` is a 1-D+ float array-like; everything is reduced over all
    elements. The returned dict is JSON-safe apart from numpy scalar types,
    which callers convert with ``float()``.
    """
    import numpy as np

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("finite nonempty proxy values required")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    raw_mean = float(array.mean())
    raw_std = 0.0 if np.ptp(array) == 0.0 else float(array.std())
    floored_std = max(raw_std, float(eps))
    floor_active = raw_std < float(eps)
    degenerate = raw_std == 0.0
    normalized = np.zeros_like(array) if degenerate else (array - raw_mean) / floored_std
    normalized_std = float(normalized.std())
    return {
        "count": int(array.size),
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "degenerate": degenerate,
        "active_mask": not degenerate,
        "raw_abs_max": float(np.abs(array).max()),
        "eps": float(eps),
        "normalized_std_used": floored_std,
        "floor_active": floor_active,
        "std_to_eps_ratio": raw_std / float(eps),
        "normalized_std": normalized_std,
        "normalized_min": float(normalized.min()),
        "normalized_max": float(normalized.max()),
        "normalized_abs_p99": float(np.percentile(np.abs(normalized), 99)),
        # Fraction of samples whose normalized magnitude is below the
        # usable-scale threshold: these carry almost no gradient signal.
        "fraction_below_usable_scale": float(
            (np.abs(normalized) < UNUSABLE_NORMALIZED_STD).mean()),
        "amplification_lost_factor": (1.0 / normalized_std) if normalized_std > 0 else None,
        "usable_label": not floor_active and normalized_std >= UNUSABLE_NORMALIZED_STD,
    }


def proxy_scale_report(raw_diagnostics, stored_std, names, eps: float = DEFAULT_EPS) -> dict:
    """Audit every stored proxy channel against the std actually baked in.

    ``raw_diagnostics`` is ``[T, P]`` raw proxy values; ``stored_std`` is the
    ``process_normalization_std`` array the store ships. The store's own std is
    compared with the recomputed one so a store that baked in a different floor
    is visible rather than assumed identical.
    """
    import numpy as np

    raw = np.asarray(raw_diagnostics, dtype=np.float64)
    std = np.asarray(stored_std, dtype=np.float64).reshape(-1)
    if raw.ndim != 2:
        raise ValueError("raw diagnostics must be [T, P]")
    if raw.shape[1] != len(names) or std.shape != (len(names),):
        raise ValueError("proxy names, raw columns and stored std must agree in count")
    if len(set(names)) != len(names) or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("unique proxy names and finite positive stored std required")
    channels = {}
    for index, name in enumerate(names):
        record = normalized_proxy_scale(raw[:, index], eps=eps)
        record["stored_std"] = float(std[index])
        record["stored_matches_recomputed"] = bool(
            np.isclose(record["normalized_std_used"], float(std[index]),
                       rtol=1e-5, atol=0.0))
        channels[name] = record
    unusable = sorted(name for name, record in channels.items()
                      if not record["usable_label"])
    floored = sorted(name for name, record in channels.items()
                     if record["floor_active"])
    return {
        "format": "r7-process-proxy-scale-v1",
        "scientific_claim": False,
        "eps": float(eps),
        "usable_threshold": UNUSABLE_NORMALIZED_STD,
        "channels": channels,
        "floor_active_channels": floored,
        "unusable_channels": unusable,
        "note": ("read-only audit; nothing here re-normalizes the store, and a "
                 "different normalization would be a separately versioned store"),
    }


def proxy_scale_sidecar_report(raw_diagnostics, stored_mean, stored_std, names, metadata):
    """Compare the frozen store's scaling with the separately published sidecar."""
    import numpy as np
    from data.preprocess.r7_process_scale_sidecar import normalize_process_diagnostics

    raw = np.asarray(raw_diagnostics, dtype=np.float64)
    mean = np.asarray(stored_mean, dtype=np.float64)
    std = np.asarray(stored_std, dtype=np.float64)
    report = proxy_scale_report(raw, std, names)
    if mean.shape != std.shape or not np.isfinite(mean).all():
        raise ValueError("stored proxy mean must be finite and match the channel width")
    normalized = normalize_process_diagnostics(raw, metadata, names=names)
    old = (raw - mean) / std
    for index, name in enumerate(names):
        entry = report["channels"][name]
        entry.update(
            old_stored_normalized_std=float(old[:, index].std()),
            physical_unit=metadata["units"][index],
            physical_unit_scale=metadata["physical_unit_scale"][index],
            dimensionless_std=metadata["dimensionless_std"][index],
            relative_floor=metadata["relative_floor"][index],
            scaled_normalized_std=float(normalized[:, index].std()),
            degenerate=metadata["degenerate"][index],
            active_mask=metadata["active_mask"][index],
        )
    report.update(format="r7-process-proxy-scale-sidecar-v1",
                  sidecar_identity=metadata.get("sidecar_identity"),
                  fit_split="train",
                  note="scale comparison only; forecast changes are not attributed solely to scaling")
    return report


def _grad_vector(named_parameters, names):
    """Flatten the selected parameters' gradients, treating absent as zero."""
    pieces = []
    for name in names:
        grad = named_parameters[name].grad
        pieces.append(torch.zeros_like(named_parameters[name]) if grad is None else grad)
    if not pieces:
        return None
    return torch.cat([piece.reshape(-1).to(torch.float64) for piece in pieces])


def gradient_conflict_record(model, batch, *, process_weight, steps, groups):
    """Gradient norm/angle of the forecast and process losses on named groups.

    ``groups`` maps a label to the parameter names that belong to it, so the
    shared trunk and the process readout are never blended into one number. A
    group whose parameters carry no gradient from either loss is reported with
    ``defined: False`` rather than a fabricated zero cosine.
    """
    from model.r7_halting import forecast_inputs
    from training.r7_recursive_losses import deep_supervised_forecast_mse

    if process_weight < 0 or not math.isfinite(process_weight):
        raise ValueError("process_weight must be finite and nonnegative")
    if not model.training:
        raise ValueError("gradient conflict is a train-mode diagnostic")
    target = batch["atmos_target"]
    latitude = batch.get("latitude")
    named = dict(model.named_parameters())
    if any(not group for group in groups.values()):
        raise ValueError("every parameter group must be nonempty")
    missing = [name for group in groups.values() for name in group
               if name not in named]
    if missing:
        raise ValueError(f"unknown parameter names: {sorted(set(missing))[:3]}")

    model.zero_grad(set_to_none=True)
    out = model(forecast_inputs(batch), reasoning_steps=steps)
    forecast_loss = deep_supervised_forecast_mse(out.draft_forecasts, target, latitude)
    process_loss = target.new_zeros(())
    if process_weight > 0:
        if "process_targets" not in batch:
            raise ValueError("positive process weight requires process_targets")
        process_target = batch["process_targets"].to(
            device=out.process_predictions.device,
            dtype=out.process_predictions.dtype)
        if process_target.shape != (out.process_predictions.shape[0],
                                    out.process_predictions.shape[-1]):
            raise ValueError("process targets must exactly match the readout width")
        if not torch.isfinite(process_target).all():
            raise ValueError("process targets must be finite")
        process_loss = torch.nn.functional.mse_loss(
            out.process_predictions,
            process_target.unsqueeze(1).expand_as(out.process_predictions))

    # Forecast-only gradients.
    model.zero_grad(set_to_none=True)
    forecast_loss.backward(retain_graph=True)
    forecast_grads = {label: _grad_vector(named, group) for label, group in groups.items()}
    # Process-only gradients.
    model.zero_grad(set_to_none=True)
    if process_weight > 0:
        (process_weight * process_loss).backward()
    process_grads = {label: _grad_vector(named, group) for label, group in groups.items()}
    model.zero_grad(set_to_none=True)

    records = {}
    for label in groups:
        forecast_grad, process_grad = forecast_grads[label], process_grads[label]
        forecast_norm = float(forecast_grad.norm()) if forecast_grad is not None else 0.0
        process_norm = float(process_grad.norm()) if process_grad is not None else 0.0
        defined = forecast_norm > 0 and process_norm > 0
        cosine = None
        if defined:
            cosine = float(torch.nn.functional.cosine_similarity(
                forecast_grad.unsqueeze(0), process_grad.unsqueeze(0), dim=1).item())
        records[label] = {
            "forecast_grad_norm": forecast_norm,
            "process_grad_norm": process_norm,
            "process_grad_share": (process_norm / (forecast_norm + process_norm)
                                   if (forecast_norm + process_norm) > 0 else None),
            "cosine_forecast_process": cosine,
            "cosine_defined": bool(defined),
            "parameters": len(groups[label]),
        }
    return {
        "format": "r7-process-gradient-conflict-v1",
        "scientific_claim": False,
        "causal_evidence": False,
        "process_weight": float(process_weight),
        "steps": int(steps),
        "forecast_loss": float(forecast_loss.detach()),
        "process_loss": float(process_loss.detach()),
        "groups": records,
        "note": ("cosine_forecast_process is a one-batch geometric description "
                 "and is never causal evidence (#65); at process_weight=0 the "
                 "process gradient is exactly zero and the cosine is undefined"),
    }


def recursion_trajectory(model, batch, *, max_steps=4):
    """K0..Kmax error/update geometry, reusing the #53 implementation.

    Returns one record per step with the per-channel means of the #53 terms,
    the aggregate latitude-weighted error at every K, and the adjacent-update
    cosines between consecutive corrections. Only scalars are retained.
    """
    from model.r7_halting import forecast_inputs
    from training.r7_correction_diagnostic import correction_terms

    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model(forecast_inputs(batch), reasoning_steps=max_steps)
    finally:
        model.train(was_training)
    drafts = out.draft_forecasts
    if drafts.ndim != 5 or drafts.shape[1] != max_steps + 1:
        raise ValueError("full K0..K trajectory required")
    target = batch["atmos_target"]
    latitude = batch.get("latitude")
    per_step = []
    previous = None
    for k in range(max_steps):
        terms = correction_terms(drafts[:, k], drafts[:, k + 1], target,
                                 latitude, previous)
        previous = drafts[:, k + 1] - drafts[:, k]
        entry = {
            "k": k + 1,
            "mse_before": float(terms["mse_before"].mean()),
            "mse_after": float(terms["mse_after"].mean()),
            "mse_change": float(terms["mse_change"].mean()),
            "update_energy": float(terms["update_energy"].mean()),
            "update_norm": float(terms["update_energy"].mean()) ** 0.5,
            "error_update_cosine": float(terms["error_update_cosine"].mean()),
            "worsening_fraction": float(terms["worsening"].double().mean()),
            "wrong_direction_fraction": float(terms["wrong_direction"].double().mean()),
            "zero_update_fraction": float(terms["zero_update"].double().mean()),
            "algebra_residual_max_abs": float(terms["algebra_residual"].abs().max()),
        }
        if "previous_update_cosine" in terms:
            entry["previous_update_cosine"] = float(
                terms["previous_update_cosine"].mean())
            entry["previous_cosine_defined_fraction"] = float(
                terms["previous_cosine_defined"].double().mean())
        per_step.append(entry)
    return {
        "format": "r7-process-recursion-trajectory-v1",
        "scientific_claim": False,
        "future_labels_used_retrospectively": True,
        "forecast_modified": False,
        "training_executed": False,
        "max_steps": int(max_steps),
        "per_step": per_step,
        "note": ("K is internal reasoning depth and never advances physical time; "
                 "the #53 oracle damping is not applied anywhere here"),
    }


def parameter_groups(model) -> dict:
    """Split a process model's parameters into shared trunk and process-only.

    The split exists because #65 requires the readout head and the shared
    parameters to be reported separately: a norm averaged over both hides
    which of them the auxiliary loss actually reaches.
    """
    shared, readout, queries = [], [], []
    for name, _ in model.named_parameters():
        if name.startswith("process_readout."):
            readout.append(name)
        elif name == "process_queries":
            queries.append(name)
        else:
            shared.append(name)
    return {"shared": shared, "process_readout": readout, "process_queries": queries}


def summarise_proxy_effect(proxy_report, names: Sequence[str]) -> dict:
    """One sentence-level verdict per proxy, for the report's headline table."""
    verdicts = {}
    for name in names:
        record = proxy_report["channels"][name]
        if record.get("degenerate"):
            verdicts[name] = "degenerate: true zero variance; explicitly masked"
        elif not record["usable_label"]:
            verdicts[name] = (
                "unusable: the 1e-6 floor divides by %.0fx the real spread and "
                "the normalized label's std is %.4f"
                % (record["amplification_lost_factor"], record["normalized_std"]))
        else:
            verdicts[name] = "usable: normalized std %.4f" % record["normalized_std"]
    return verdicts


def measurement_input_contract() -> dict:
    """Whether the proxies reach the model as *inputs* at all.

    The distinction decides what a proxy-scale defect can and cannot explain.
    A proxy that only ever appears as a *supervision target* cannot corrupt the
    forward state; it can only weaken (or fail to shape) the auxiliary loss.
    This is a property of the code path, so it is stated from the whitelist the
    model is actually called with rather than from intent.
    """
    from model.r7_halting import forecast_inputs

    probe = {
        "coarse_history": torch.zeros(1, 2, 1, 2, 2),
        "atmos_target": torch.zeros(1, 1, 2, 2),
        "process_targets": torch.zeros(1, 8),
        "latitude": torch.zeros(1, 2),
        "lead_time_hours": torch.zeros(1),
    }
    forwarded = sorted(forecast_inputs(probe))
    return {
        "format": "r7-process-input-contract-v1",
        "forwarded_keys": forwarded,
        "process_targets_is_forwarded": "process_targets" in forwarded,
        "atmos_target_is_forwarded": "atmos_target" in forwarded,
        "conclusion": ("process proxies are supervision targets only; they are not "
                       "forwarded into the model, so a proxy scale defect cannot "
                       "make the process state's moisture channels constant as an "
                       "input - it can only weaken the auxiliary objective"),
    }
