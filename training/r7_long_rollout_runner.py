"""Frozen-endpoint, fresh-optimizer full-BPTT training over exact train windows."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.utils.data import default_collate

from data.r7_long_rollout_dataset import ZarrLongRolloutDataset
from .r7_long_rollout import training_long_rollout, validate_physical_weights
from .r7_recursive_losses import (
    DRAFT_SUPERVISION_DEEP, DRAFT_SUPERVISION_FINAL_ONLY, DRAFT_SUPERVISION_MODES,
    ZERO_GRADIENT_LAMBDA, require_gradient_lambda,
)
from .r7_autoregressive_runner import (
    SOURCE_FILES as AUTOREGRESSIVE_SOURCES, _bind, _finite_state, _output_path,
    _paths, _validate_imported_model, _validate_options, _write_json,
)
from .r7_ema import ParameterEMA, require_ema_decay
from .r7_experiment import canonical_digest, dataset_identity, load_checkpoint, model_code_digest
from .r7_experiment import rng_state, save_exclusive, seed_everything, select_device
from .r7_scheduled_runner import _check_deadline, _publish_checkpoint, warmup_cosine_factor

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = tuple(AUTOREGRESSIVE_SOURCES) + (
    "data/r7_long_rollout_dataset.py", "training/r7_long_rollout.py",
    "training/r7_long_rollout_runner.py", "training/r7_ema.py",
)
LIMITATIONS = [
    "Normalized draft-supervised training MSE is not held-out physical forecast skill.",
    "Full BPTT uses predicted history, never future observations or forcing.",
    "A new optimizer imports verified model weights only; no failed-attempt resume is supported.",
    "Full source qualification, frozen root protocol, offline workers and external deadlines belong to the caller.",
    "GPU training is not bitwise reproducible; checkpoint/source/model identities remain strict.",
]


def training_code_digest():
    return canonical_digest({name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                             for name in SOURCE_FILES})


def _supervision_block(supervision_mask):
    """Declarative, digestible description of the spatial supervision region."""
    if supervision_mask is None:
        return {"kind": "full_grid",
                "description": "every cell of the target grid is supervised"}
    tensor = torch.as_tensor(supervision_mask)
    if tensor.dim() != 2:
        raise ValueError("supervision_mask must be a 2-D [H,W] mask")
    if not torch.isfinite(tensor).all() or bool((tensor < 0).any()):
        raise ValueError("supervision_mask must be finite and nonnegative")
    selected = int((tensor > 0).sum())
    if selected == 0:
        raise ValueError("supervision_mask selects no cell")
    payload = json.dumps([[int(value > 0) for value in row] for row in tensor.tolist()],
                         separators=(",", ":"))
    return {"kind": "spatial_mask", "shape": [int(tensor.shape[0]), int(tensor.shape[1])],
            "selected_cells": selected,
            "mask_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "normalization": "mask-and-latitude weighted mean over the selected cells"}


def _gradient_semantics(gradient_lambda):
    """Extra contract/report fields for the spatial-gradient penalty.

    ``0.0`` adds nothing, so the registered contract and report stay word for word
    unchanged. A positive value records the frozen design coefficient and formula.
    """
    if gradient_lambda == ZERO_GRADIENT_LAMBDA:
        return {}
    return {"spatial_gradient_lambda": gradient_lambda,
            "spatial_gradient_penalty": "latitude_weighted_first_difference_mse",
            "spatial_gradient_pairs": "both endpoints inside the supervision mask; interior_32",
            "spatial_gradient_application": ("per draft: L_mse + lambda * L_grad, then the "
                                             "registered K-axis weights")}


def _draft_semantics(draft_supervision, gradient_lambda=ZERO_GRADIENT_LAMBDA):
    """The frozen contract/report description of the K-axis organization.

    ``deep`` repeats the registered fields verbatim. ``final_only`` records the
    actual objective instead of pretending the deep form is still in force. The
    spatial-gradient fields are appended only when the penalty is active.
    """
    if draft_supervision == DRAFT_SUPERVISION_DEEP:
        semantics = {"draft_supervision": DRAFT_SUPERVISION_DEEP,
                     "internal_deep_supervision": True,
                     "objective": "deep_supervised_latitude_area_mse", "final_weight": 2.,
                     "k_weights": "linspace(1,2,K+1) normalized; initial plus every K draft"}
    elif draft_supervision == DRAFT_SUPERVISION_FINAL_ONLY:
        semantics = {"draft_supervision": DRAFT_SUPERVISION_FINAL_ONLY,
                     "internal_deep_supervision": False,
                     "objective": "final_draft_latitude_area_mse", "final_weight": 1.,
                     "k_weights": ("final draft only (weight 1.0); initial and intermediate "
                                   "drafts carry no direct target term")}
    else:
        raise ValueError(f"draft_supervision must be one of {DRAFT_SUPERVISION_MODES}")
    semantics.update(_gradient_semantics(gradient_lambda))
    return semantics


def _ema_semantics(ema_decay):
    """Extra contract/report fields for the EMA checkpoint.

    ``None`` adds nothing, so the registered contract and report stay word for word
    unchanged. A value records the frozen rule: plain decay from the parent weights,
    every optimizer step, no bias correction and no warm-up.
    """
    value = require_ema_decay(ema_decay)
    if value is None:
        return {}
    return {"ema_decay": value,
            "ema_initialization": "parent_checkpoint",
            "ema_scope": "all trainable parameters",
            "ema_update": "after every optimizer step",
            "ema_bias_correction": False, "ema_warmup_updates": 0}


def _contract(dataset, supplied, *, output, steps, updates, seed, lr, warmup,
              weight_decay, device, batch_size, clip, checkpoint_every, physical_weights,
              supervision_mask=None, draft_supervision=DRAFT_SUPERVISION_DEEP,
              gradient_lambda=ZERO_GRADIENT_LAMBDA, ema_decay=None):
    contract = deepcopy(supplied)
    gradient_lambda = require_gradient_lambda(gradient_lambda)
    identity, _ = dataset_identity(dataset.manifest)
    if identity != contract["data_identity"]:
        raise ValueError("actual training data identity differs from the frozen contract")
    weights = list(validate_physical_weights(physical_weights))
    _bind(contract, {"model_code_sha256": model_code_digest(),
                     "scientific_claim": False, "test_read": False, "limitations": list(LIMITATIONS),
                     "training_code_sha256": training_code_digest(),
                     "steps": steps, "total_updates": updates, "seed": seed, "mode": "long_rollout",
                     "lr": lr, "warmup_updates": warmup, "weight_decay": weight_decay,
                     "bf16": False, "device_type": device.type, "torch_version": str(torch.__version__),
                     "batch_size": batch_size, "clip": clip, "dataset_length": len(dataset),
                     "checkpoint_every": checkpoint_every, "optimization": "full-bptt",
                     "schedule": "linear-warmup-then-cosine", "minimum_lr_ratio": .1,
                     "process_weight": 0., "output_dir": str(Path(output).resolve())})
    autoregression = contract.setdefault("autoregression", {})
    _bind(autoregression, {"mode": "long_rollout", "physical_steps": len(weights),
                          "physical_weights": weights, "step_hours": 6, "fixed_transition_lead_hours": 6,
                          "detach_physical_steps": False, "detach_reasoning_steps": False,
                          "loss_space": "normalized",
                          "physical_loss": "sum(weight_i * L_6i), no physical-axis normalization",
                          "window_sha256": dataset.summary["window_sha256"], "windows": dataset.summary,
                          "excluded_sample_ids": dataset.summary["excluded_sample_ids"],
                          "supervision": _supervision_block(supervision_mask),
                          **_draft_semantics(draft_supervision, gradient_lambda),
                          **_ema_semantics(ema_decay)})
    return contract


def _update(model, optimizer, batch, contract, device, deadline, supervision_mask=None):
    optimizer.zero_grad(set_to_none=True)
    moved = {name: value.to(device) if torch.is_tensor(value) else value for name, value in batch.items()}
    result = training_long_rollout(model, moved, contract["steps"],
                                  physical_weights=contract["autoregression"]["physical_weights"],
                                  supervision_mask=supervision_mask,
                                  draft_supervision=contract["autoregression"].get(
                                      "draft_supervision", DRAFT_SUPERVISION_DEEP),
                                  gradient_lambda=contract["autoregression"].get(
                                      "spatial_gradient_lambda", ZERO_GRADIENT_LAMBDA))
    result.loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), contract["clip"], error_if_nonfinite=True)
    if deadline is not None:
        _check_deadline(deadline)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    _finite_state(model)
    return {"loss": float(result.loss.detach()), "l6": float(result.l6.detach()),
            "l12": None if result.l12 is None else float(result.l12.detach()),
            "per_step_losses": result.per_step_losses.detach().cpu().tolist(),
            "gradient_norm": float(norm)}


def _publish_ema_checkpoint(folder, update_number, *, model, ema, signature, contract, epoch, cursor):
    """Save the EMA weights as a second checkpoint with the same frozen signature."""
    target = folder / f"ema_update_{update_number:07d}.pt"
    save_exclusive(target, {
        "format": "r7-local-v1", "signature": signature, "contract": contract,
        "model": ema.state_dict(model), "optimizer": {}, "updates": update_number,
        "epoch": epoch, "cursor": cursor, "rng": rng_state(),
        "checkpoint_kind": "ema", "ema_updates": ema.updates, "ema_decay": ema.decay})
    load_checkpoint(target, expected=signature)
    return target


def _run_updates(dataset, folder, model, optimizer, contract, signature, deadline, runtime=None):
    supervision_mask, ema = runtime if runtime is not None else (None, None)
    epoch, cursor, losses, checkpoint, ema_checkpoint = 0, 0, [], None, None
    device = next(model.parameters()).device
    for update in range(1, contract["total_updates"] + 1):
        if deadline is not None:
            _check_deadline(deadline)
        if cursor == len(dataset):
            epoch, cursor = epoch + 1, 0
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(contract["seed"] + epoch))
        indices = order[cursor:min(cursor + contract["batch_size"], len(dataset))].tolist()
        batch = default_collate([dataset[index] for index in indices])
        rate = contract["lr"] * warmup_cosine_factor(update, total_updates=contract["total_updates"],
                                                   warmup_updates=contract["warmup_updates"], minimum_ratio=.1)
        optimizer.param_groups[0]["lr"] = rate
        metrics = _update(model, optimizer, batch, contract, device, deadline, supervision_mask)
        if ema is not None:
            ema.update()
        cursor += len(indices)
        losses.append({"update": update, "epoch": epoch, "lr": rate, "samples": len(indices), **metrics})
        if deadline is not None:
            _check_deadline(deadline)
        if update % contract["checkpoint_every"] == 0 or update == contract["total_updates"]:
            checkpoint = _publish_checkpoint(folder, update, model=model, optimizer=optimizer,
                                             signature=signature, contract=contract,
                                             epoch=epoch, cursor=cursor, intervention=None)
            load_checkpoint(checkpoint, expected=signature)
        if ema is not None and update == contract["total_updates"]:
            ema_checkpoint = _publish_ema_checkpoint(folder, update, model=model, ema=ema,
                                                     signature=signature, contract=contract,
                                                     epoch=epoch, cursor=cursor)
    return checkpoint, ema_checkpoint, losses


def fine_tune_long_rollout(manifest, output, *, model, contract, physical_weights, parent_weights=None,
                          steps=4, updates=200, seed=0, lr=2e-5, warmup=10, weight_decay=1e-4,
                          deadline=None, checkpoint_every=20, device_name="cpu", batch_size=1, clip=1.,
                          supervision_mask=None, draft_supervision=DRAFT_SUPERVISION_DEEP,
                          gradient_lambda=ZERO_GRADIENT_LAMBDA, ema_decay=None):
    weights = validate_physical_weights(physical_weights)
    if draft_supervision not in DRAFT_SUPERVISION_MODES:
        raise ValueError(f"draft_supervision must be one of {DRAFT_SUPERVISION_MODES}")
    gradient_lambda = require_gradient_lambda(gradient_lambda)
    ema_decay = require_ema_decay(ema_decay)
    _validate_options(steps=steps, updates=updates, seed=seed, mode="two_step", lr=lr, warmup=warmup,
                      weight_decay=weight_decay, bf16=False, batch_size=batch_size, clip=clip,
                      lambda12=0., checkpoint_every=checkpoint_every, deadline=deadline)
    if not isinstance(contract, dict) or not isinstance(contract.get("autoregression"), dict):
        raise ValueError("a frozen contract with explicit autoregression metadata is required")
    supplied = deepcopy(contract)
    output = _output_path(output, supplied)
    _validate_imported_model(model, supplied, parent_weights)
    declared = supplied["autoregression"]
    if (declared.get("physical_steps") != len(weights)
            or declared.get("physical_weights") != list(weights)
            or not isinstance(declared.get("window_sha256"), str)):
        raise ValueError("physical steps/weights/window digest must be predeclared exactly")
    if declared.get("draft_supervision", DRAFT_SUPERVISION_DEEP) != draft_supervision:
        raise ValueError("the declared draft supervision organization differs from the actual run")
    if require_gradient_lambda(declared.get("spatial_gradient_lambda", ZERO_GRADIENT_LAMBDA)) != gradient_lambda:
        raise ValueError("the declared spatial gradient lambda differs from the actual run")
    if require_ema_decay(declared.get("ema_decay")) != ema_decay:
        raise ValueError("the declared EMA decay differs from the actual run")
    dataset = ZarrLongRolloutDataset(manifest, physical_steps=len(weights),
        expected_exclusions=declared.get("excluded_sample_ids", ()),
        expected_window_sha256=declared["window_sha256"])
    device = select_device(device_name)
    bound = _contract(dataset, supplied, output=output, steps=steps, updates=updates, seed=seed,
                      lr=lr, warmup=warmup, weight_decay=weight_decay, device=device,
                      batch_size=batch_size, clip=clip, checkpoint_every=checkpoint_every,
                      physical_weights=weights, supervision_mask=supervision_mask,
                      draft_supervision=draft_supervision, gradient_lambda=gradient_lambda,
                      ema_decay=ema_decay)
    signature = canonical_digest(bound)
    if deadline is not None:
        _check_deadline(deadline)
    folder, _ = _paths(output, None, signature, bound)
    started = time.perf_counter()
    try:
        seed_everything(seed)
        model.to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        ema = ParameterEMA(model, ema_decay) if ema_decay is not None else None
        checkpoint, ema_checkpoint, losses = _run_updates(
            dataset, folder, model, optimizer, bound, signature, deadline,
            (supervision_mask, ema))
        if ema is not None and ema_checkpoint is None:
            raise RuntimeError("the EMA checkpoint was not published at the frozen endpoint")
        if deadline is not None:
            _check_deadline(deadline)
        semantics = _draft_semantics(draft_supervision, gradient_lambda)
        report = {"scientific_claim": False, "limitations": LIMITATIONS, "test_read": False,
                  "contract": bound, "signature": signature, "protocol_sha256": bound["protocol_sha256"],
                  "model_code_sha256": bound["model_code_sha256"], "source_sha256": bound["source_sha256"],
                  "data_identity": bound["data_identity"], "optimization": "full-bptt",
                  "internal_k_detach": False, "physical_step_detach": False,
                  "objective": semantics["objective"],
                  "internal_deep_supervision": semantics["internal_deep_supervision"],
                  "draft_supervision": semantics["draft_supervision"],
                  "supervision": bound["autoregression"]["supervision"],
                  "selected_checkpoint": str(checkpoint), "selected_update": updates, "total_updates": updates,
                  "selection_split": None, "selection_metric": "frozen endpoint; no validation selection",
                  "resumed_from_updates": 0, "updates_this_run": len(losses), "losses": losses,
                  "elapsed_seconds": time.perf_counter() - started, "initialization": bound["initialization"],
                  "parent_optimizer_imported": False, "bf16": False, "device_type": device.type,
                  "windows": dataset.summary,
                  "reproducibility": "config-reproducible; no failed-attempt resume or bitwise GPU promise"}
        report.update(_gradient_semantics(gradient_lambda))
        report.update(_ema_semantics(ema_decay))
        if ema is not None:
            report["ema_checkpoint"] = str(ema_checkpoint)
            report["ema_updates_applied"] = ema.updates
            report["final_optimizer_checkpoint"] = str(checkpoint)
            report["selected_checkpoint"] = str(ema_checkpoint)
            report["selected_checkpoint_kind"] = "ema"
        _write_json(folder / "training_report.json", report)
        return checkpoint, report
    except BaseException as exc:
        model.zero_grad(set_to_none=True)
        _write_json(folder / "failed_attempt.json", {"status": "failed", "scientific_claim": False,
                    "limitations": LIMITATIONS, "signature": signature, "test_read": False,
                    "failure_reason": f"{type(exc).__name__}: {exc}", "resume_permitted": False})
        raise
