"""Bounded training to a fixed schedule with validation-only model selection.

``r7_local_runner.run_local_updates`` intentionally stops at a fixed update
endpoint and does no validation. #64 B2 requires more than that: the learning
rate schedule, warmup, maximum update count, validation frequency, checkpoint
selection rule and early-stopping rule must all be fixed **before** training,
and early stopping may read the validation split only.

This module adds exactly those, reusing the audited pieces rather than
re-deriving them:

- the optimizer step is ``r7_local_runner.update_group``, so the objective,
  the streamed truncated BPTT path and the gradient clipping are the same
  functions B1 trained with;
- checkpoints keep the ``r7-local-v1`` format (``save_exclusive`` plus the
  contract/signature/rng fields), so ``r7_experiment.load_checkpoint`` and
  ``training.r7_evaluate.evaluate_local`` accept them unchanged;
- validation scoring uses the same ``ZarrRolloutDataset`` +
  ``RolloutRMSEAccumulator`` pair as the published evaluation, on the
  validation split only, with no sampling.

What this module deliberately does **not** do: touch the test split, choose
hyperparameters from a metric, or extend training past the declared update
endpoint. ``score_validation`` reads nothing but the declared validation
manifest, and the validation frequency/cap come from the caller's frozen
protocol.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from .r7_experiment import (
    canonical_digest,
    load_checkpoint,
    make_model,
    rng_state,
    restore_rng,
    save_exclusive,
    seed_everything,
    select_device,
)
from .r7_local_runner import _integer, update_group


def warmup_cosine_factor(update, *, total_updates, warmup_updates, minimum_ratio):
    """Linear warmup then cosine decay, as an explicit function of the update.

    ``update`` is 1-based: update 1 runs at ``1/warmup_updates`` of the peak
    rate, the first ``warmup_updates`` updates reach the peak, and the rate
    decays cosinely from the peak to ``minimum_ratio`` of the peak by
    ``total_updates``.
    """
    for value, name, minimum in ((total_updates, "total_updates", 1),
                                 (warmup_updates, "warmup_updates", 0),
                                 (update, "update", 1)):
        _integer(value, name, minimum)
    if warmup_updates > total_updates:
        raise ValueError("warmup_updates cannot exceed total_updates")
    if not math.isfinite(minimum_ratio) or not 0 < minimum_ratio <= 1:
        raise ValueError("minimum_ratio must be in (0, 1]")
    if update <= warmup_updates:
        return update / warmup_updates
    progress = (update - warmup_updates) / max(1, total_updates - warmup_updates)
    progress = min(1.0, progress)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


@torch.no_grad()
def score_validation(model, dataset, *, kind, device, steps, lead_hours=(6,), step_hours=6):
    """Mean latitude-weighted MSE over the declared validation rollout windows.

    ``dataset`` must be a *held-out* rollout dataset (``ZarrRolloutDataset`` over
    the validation split), never the training dataset: #64 B2 fixes validation
    frequency and early stopping on validation only, which is meaningless if the
    scored data is what the model is fitting. Horizons are the declared integer
    leads in hours.

    Returns ``{"mean_mse": float, "n_cases": int, "per_lead_mse": {...}}`` in the
    model's normalized space. The value ranks checkpoints and stops training; it
    must never be reported as a physical-unit score - that is ``evaluate_local``.
    """
    from model.r7_rollout import autoregressive_rollout, rollout_model_input

    # Validate before converting: int(6.5) silently becomes 6, which would score
    # a different horizon than the one the protocol declared.
    if not lead_hours:
        raise ValueError("validation lead hours must be positive integers")
    horizons = []
    for lead in lead_hours:
        if isinstance(lead, bool) or not isinstance(lead, (int, np.integer)) or int(lead) < 1:
            raise ValueError("validation lead hours must be positive integers")
        horizons.append(int(lead))
    horizons = tuple(horizons)
    inference = {} if kind == "native" else {"reasoning_steps": steps}
    was_training = model.training
    model.eval()
    totals = {float(lead): 0.0 for lead in horizons}
    counts = {float(lead): 0 for lead in horizons}
    for index in range(len(dataset)):
        sample = dataset[index]
        if "rollout_targets" not in sample:
            raise ValueError("validation scoring requires a held-out rollout dataset "
                             "(samples carrying rollout_targets), not the training set")
        history = sample["coarse_history"].unsqueeze(0).to(device)
        batch = rollout_model_input(sample, lead_hours=float(step_hours), device=device)
        trajectory = autoregressive_rollout(
            model, batch, lead_hours=horizons, step_hours=int(step_hours),
            history_interval_hours=int(step_hours), inference_kwargs=inference)
        latitude = sample["latitude"].float().to(device)
        if latitude.ndim == 1:
            latitude = latitude.unsqueeze(0).expand(history.shape[0], -1)
        weight = torch.cos(torch.deg2rad(latitude)).clamp_min(0)
        area = weight.sum(1, keepdim=True).clamp_min(1e-12)
        targets = sample["rollout_targets"].float().to(device)
        for position, lead in enumerate(trajectory.lead_hours):
            error = trajectory.forecasts[0, position].float() - targets[position]
            per_case = (error.square().mean((0, 2)) * weight[0]).sum() / area[0, 0]
            totals[float(lead)] += float(per_case)
            counts[float(lead)] += 1
    if was_training:
        model.train()
    if not sum(counts.values()):
        raise ValueError("validation scoring saw no cases")
    per_lead = {lead: totals[lead] / max(1, counts[lead]) for lead in totals}
    return {"mean_mse": sum(per_lead.values()) / len(per_lead), "n_cases": len(dataset),
            "per_lead_mse": {str(lead): value for lead, value in per_lead.items()},
            "space": "normalized forecast MSE, latitude-weighted, no denormalization",
            "split": "val"}


def _check_deadline(deadline):
    if time.perf_counter() >= deadline:
        raise RuntimeError("scheduled runner deadline exceeded")


def _validate_schedule(*, total_updates, batch_size, steps, seed, validation_every,
                       lr, clip, minimum_lr_ratio, early_stopping_patience,
                       minimum_improvement, validation_dataset):
    for value, name, minimum in ((total_updates, "total_updates", 1),
                                 (batch_size, "batch_size", 1),
                                 (steps, "steps", 0), (seed, "seed", 0),
                                 (validation_every, "validation_every", 0)):
        _integer(value, name, minimum)
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(clip) or clip <= 0:
        raise ValueError("positive finite lr and clip required")
    if not math.isfinite(minimum_lr_ratio) or not 0 < minimum_lr_ratio <= 1:
        raise ValueError("minimum_lr_ratio must be in (0, 1]")
    if early_stopping_patience is not None:
        _integer(early_stopping_patience, "early_stopping_patience", 1)
    if not math.isfinite(minimum_improvement) or minimum_improvement < 0:
        raise ValueError("minimum_improvement must be nonnegative and finite")
    if validation_every and validation_dataset is None:
        raise ValueError("validation_every > 0 requires an explicit held-out "
                         "validation_dataset; the training dataset is not validation")
    if validation_every == 0 and early_stopping_patience is not None:
        raise ValueError("early stopping without validation checks is not a rule")


def _scheduled_contract(*, kind, model_config, data_identity, batch_size, steps, seed, lr,
                        process_weight, clip, bf16, device, dataset, warmup_updates,
                        minimum_lr_ratio, validation_every, early_stopping_patience,
                        minimum_improvement, validation_lead_hours, step_hours,
                        validation_dataset, intervention):
    contract = {
        "kind": kind, "model": dict(model_config), "data_identity": str(data_identity),
        "batch_size": batch_size, "steps": steps, "seed": seed, "lr": lr,
        "process_weight": process_weight, "clip": clip, "bf16": bool(bf16),
        "device_type": device.type, "torch_version": str(torch.__version__),
        "dataset_length": len(dataset),
        "optimization": "streamed-truncated" if kind != "native" else "native",
        "schedule": "linear-warmup-then-cosine",
        "warmup_updates": warmup_updates, "minimum_lr_ratio": minimum_lr_ratio,
        "validation_every": validation_every,
        "early_stopping_patience": early_stopping_patience,
        "minimum_improvement": minimum_improvement,
        "validation_lead_hours": [int(lead) for lead in validation_lead_hours],
        "step_hours": int(step_hours),
        "validation_split": "val" if validation_dataset is not None else None,
    }
    if intervention is not None:
        contract["intervention"] = deepcopy(intervention)
    return contract


def _initialize_scheduled_model(*, kind, model_config, device, seed, lr,
                                shared_initial_state, intervention, saved, dataset):
    seed_everything(seed)
    model = make_model(kind, model_config).to(device).train()
    applied, ignored = [], []
    if shared_initial_state is not None:
        target = model.state_dict()
        transfer = {}
        for name, tensor in shared_initial_state.items():
            if name in target and tuple(target[name].shape) == tuple(tensor.shape):
                transfer[name] = tensor
                applied.append(name)
            else:
                ignored.append(name)
        if not applied:
            raise ValueError("shared_initial_state matched no parameter of the target model")
        model.load_state_dict(transfer, strict=False)
    if intervention is not None:
        from .r7_frozen_z_intervention import install_frozen_z
        install_frozen_z(model, intervention)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=lr, weight_decay=1e-4)
    updates, epoch, cursor = 0, 0, 0
    if saved:
        model.load_state_dict(saved["model"], strict=True)
        if intervention is not None:
            from .r7_frozen_z_intervention import validate_frozen_z
            validate_frozen_z(model, intervention)
        optimizer.load_state_dict(saved["optimizer"])
        updates, epoch, cursor = saved["updates"], saved["epoch"], saved["cursor"]
        if not (0 <= cursor <= len(dataset)):
            raise ValueError("checkpoint cursor outside dataset")
        restore_rng(saved["rng"])
    return model, optimizer, updates, epoch, cursor, applied, ignored


def _publish_checkpoint(folder, update_number, *, model, optimizer, signature, contract,
                        epoch, cursor, intervention):
    # Validate before publication as well, so corrupted state cannot be archived
    # as a selected checkpoint. The post-save check guards the publication path.
    if intervention is not None:
        from .r7_frozen_z_intervention import validate_frozen_z
        validate_frozen_z(model, intervention)
    target = folder / f"update_{update_number:07d}.pt"
    save_exclusive(target, {
        "format": "r7-local-v1", "signature": signature, "contract": contract,
        "model": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "optimizer": optimizer.state_dict(), "updates": update_number,
        "epoch": epoch, "cursor": cursor, "rng": rng_state()})
    if intervention is not None:
        validate_frozen_z(model, intervention)
    return target


def _scheduled_output_paths(output_dir, saved, total_updates):
    folder = Path(output_dir)
    metrics_path = folder / "training_report.json"
    if (folder / "selected.pt").exists() or metrics_path.exists():
        raise FileExistsError("selected checkpoint or training report already published")
    if saved and saved["updates"] >= total_updates:
        raise ValueError("resume endpoint must be greater than saved updates")
    folder.mkdir(parents=True, exist_ok=bool(saved))
    return folder, metrics_path


def run_scheduled_updates(
    dataset,
    *,
    kind,
    model_config,
    data_identity,
    output_dir,
    total_updates,
    batch_size,
    steps,
    seed,
    lr,
    clip,
    process_weight=0.0,
    warmup_updates=50,
    minimum_lr_ratio=0.1,
    validation_every=100,
    early_stopping_patience=None,
    minimum_improvement=0.001,
    validation_dataset=None,
    validation_lead_hours=(6,),
    step_hours=6,
    device_name="cuda",
    bf16=False,
    shared_initial_state=None,
    resume=None,
    intervention=None,
    deadline=None,
    process_supervision_context=None,
    process_supervision_contract=None,
):
    """Train to a pre-declared schedule, selecting on validation only.

    The schedule is fixed by the caller's frozen protocol; nothing here adapts
    it to an observed metric except the declared early-stopping rule, which
    reads the validation split and stops *shorter*, never longer.

    ``shared_initial_state`` (optional) is a ``state_dict`` subset to load into
    the freshly constructed model before the first step, for the #64 B2
    requirement that arms sharing a structure start from the same common
    weights. Its applied/ignored parameter names are recorded in the report so
    the difference is an artifact, not a claim. ``intervention`` is an optional
    JSON frozen-Z specification; ``deadline`` is an absolute perf-counter limit.
    Neither option changes the ordinary arm's contract when left at None.
    """
    _validate_schedule(total_updates=total_updates, batch_size=batch_size, steps=steps,
                       seed=seed, validation_every=validation_every, lr=lr, clip=clip,
                       minimum_lr_ratio=minimum_lr_ratio,
                       early_stopping_patience=early_stopping_patience,
                       minimum_improvement=minimum_improvement,
                       validation_dataset=validation_dataset)
    device = select_device(device_name, bf16)
    contract = _scheduled_contract(
        kind=kind, model_config=model_config, data_identity=data_identity, batch_size=batch_size,
        steps=steps, seed=seed, lr=lr, process_weight=process_weight, clip=clip, bf16=bf16,
        device=device, dataset=dataset, warmup_updates=warmup_updates,
        minimum_lr_ratio=minimum_lr_ratio, validation_every=validation_every,
        early_stopping_patience=early_stopping_patience, minimum_improvement=minimum_improvement,
        validation_lead_hours=validation_lead_hours, step_hours=step_hours,
        validation_dataset=validation_dataset, intervention=intervention)
    from .r7_process_training_contract import supervision_training_contract
    supervision, supervision_kwargs = supervision_training_contract(
        process_supervision_context, process_supervision_contract,
        kind=kind, process_weight=process_weight, data_identity=str(data_identity))
    if supervision is not None:
        contract["process_supervision"] = supervision
    intervention = contract.get("intervention")
    signature = canonical_digest(contract)
    saved = load_checkpoint(resume, expected=signature) if resume else None
    folder, metrics_path = _scheduled_output_paths(output_dir, saved, total_updates)

    model, optimizer, updates, epoch, cursor, applied, ignored = _initialize_scheduled_model(
        kind=kind, model_config=model_config, device=device, seed=seed, lr=lr,
        shared_initial_state=shared_initial_state, intervention=intervention,
        saved=saved, dataset=dataset)

    losses, validations = [], []
    best_mse, best_update, stale = None, 0, 0
    selected_path = None
    stopped_reason = "reached the declared update endpoint"
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    def _publish(update_number):
        if deadline is not None:
            _check_deadline(deadline)
        return _publish_checkpoint(
            folder, update_number, model=model, optimizer=optimizer, signature=signature,
            contract=contract, epoch=epoch, cursor=cursor, intervention=intervention)

    while updates < total_updates:
        if deadline is not None:
            _check_deadline(deadline)
        if cursor == len(dataset):
            epoch += 1
            cursor = 0
        order = torch.randperm(len(dataset),
                               generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        indices = order[cursor:min(cursor + batch_size, len(dataset))]
        batches = [default_collate([dataset[i] for i in indices])]
        cursor += len(indices)
        rate = lr * warmup_cosine_factor(updates + 1, total_updates=total_updates,
                                         warmup_updates=warmup_updates,
                                         minimum_ratio=minimum_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = rate
        if deadline is not None:
            _check_deadline(deadline)
        loss = update_group(model, optimizer, batches, kind=kind, device=device, steps=steps,
                            bf16=bf16, process_weight=process_weight, clip=clip,
                            **({"process_supervision_kwargs": supervision_kwargs}
                               if supervision is not None else {}))
        updates += 1
        losses.append({"update": updates, "epoch": epoch, "loss": loss, "lr": rate,
                       "samples": sum(len(batch["coarse_history"]) for batch in batches)})
        del batches

        if validation_every and updates % validation_every == 0:
            if deadline is not None:
                _check_deadline(deadline)
            # Validation must not consume the training RNG stream, or the same
            # seed would produce a different trajectory depending on how often
            # we happened to look. Snapshot and restore around the scoring pass.
            state = rng_state()
            score = score_validation(model, validation_dataset, kind=kind, device=device,
                                     steps=steps, step_hours=int(step_hours),
                                     lead_hours=validation_lead_hours)
            restore_rng(state)
            # The first check always establishes the baseline best (there is no
            # earlier score to compare against), so "stale" only counts checks
            # after a best exists.
            improved = (best_mse is None
                        or score["mean_mse"] < best_mse * (1.0 - minimum_improvement))
            if improved:
                best_mse, best_update, stale = score["mean_mse"], updates, 0
                selected_path = _publish(updates)
            else:
                stale += 1
            validations.append({"update": updates, "mean_mse": score["mean_mse"],
                                "n_cases": score["n_cases"],
                                "per_lead_mse": score["per_lead_mse"],
                                "is_best": bool(best_update == updates),
                                "stale_validations": stale})
            if (early_stopping_patience is not None
                    and stale >= early_stopping_patience):
                stopped_reason = (f"early stop: {stale} validation checks without a "
                                  f">={minimum_improvement:.1%} relative improvement")
                break

    if not validation_every:
        # No validation ever ran: the endpoint checkpoint is the selected one,
        # because the protocol declared that rule rather than a selection metric.
        best_update = updates
        selected_path = _publish(updates)
        stopped_reason = "no validation configured; endpoint checkpoint selected"
    elif selected_path is None:
        raise RuntimeError("validation ran but no checkpoint was selected; refusing to "
                           "silently substitute the endpoint")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    report = {
        "scientific_claim": False,
        "data_identity": str(data_identity),
        "contract": contract, "signature": signature,
        "updates_this_run": len(losses), "total_updates": total_updates,
        "elapsed_seconds": elapsed,
        "seconds_per_update": elapsed / max(1, len(losses)),
        "losses": losses, "validations": validations,
        "selected_update": best_update, "selected_validation_mse": best_mse,
        "validation_checks_configured": bool(validation_every),
        "selected_checkpoint": str(selected_path),
        "stopped_reason": stopped_reason,
        "selection_split": "val",
        "selection_metric": ("mean latitude-weighted normalized MSE over the declared "
                             "validation rollout windows; lower is better; ties keep the "
                             "earlier checkpoint"),
        "early_stopped": bool(validation_every) and updates < total_updates,
        "shared_initial_state": {
            "provided": shared_initial_state is not None,
            "applied_parameters": sorted(applied), "ignored_parameters": sorted(ignored),
            "applied_count": len(applied), "ignored_count": len(ignored),
            "note": ("shared/common parameters were copied before the first step; "
                     "unlisted parameters keep the seeded initialization"),
        },
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_allocated_bytes": (torch.cuda.max_memory_allocated(device)
                                 if device.type == "cuda" else None),
        "peak_reserved_bytes": (torch.cuda.max_memory_reserved(device)
                                if device.type == "cuda" else None),
        "note": ("wall time includes batch reads and validation scoring; CUDA peaks cover "
                 "this bounded run, not a forecast-skill benchmark"),
    }
    if deadline is not None:
        _check_deadline(deadline)
    with metrics_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return selected_path, report
