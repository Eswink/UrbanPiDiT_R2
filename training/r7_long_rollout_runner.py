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
from .r7_autoregressive_runner import (
    SOURCE_FILES as AUTOREGRESSIVE_SOURCES, _bind, _finite_state, _output_path,
    _paths, _validate_imported_model, _validate_options, _write_json,
)
from .r7_experiment import canonical_digest, dataset_identity, load_checkpoint, model_code_digest
from .r7_experiment import seed_everything, select_device
from .r7_scheduled_runner import _check_deadline, _publish_checkpoint, warmup_cosine_factor

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = tuple(AUTOREGRESSIVE_SOURCES) + (
    "data/r7_long_rollout_dataset.py", "training/r7_long_rollout.py",
    "training/r7_long_rollout_runner.py",
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


def _contract(dataset, supplied, *, output, steps, updates, seed, lr, warmup,
              weight_decay, device, batch_size, clip, checkpoint_every, physical_weights):
    contract = deepcopy(supplied)
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
                          "internal_deep_supervision": True, "objective": "deep_supervised_latitude_area_mse",
                          "loss_space": "normalized", "final_weight": 2.,
                          "k_weights": "linspace(1,2,K+1) normalized; initial plus every K draft",
                          "physical_loss": "sum(weight_i * L_6i), no physical-axis normalization",
                          "window_sha256": dataset.summary["window_sha256"], "windows": dataset.summary,
                          "excluded_sample_ids": dataset.summary["excluded_sample_ids"]})
    return contract


def _update(model, optimizer, batch, contract, device, deadline):
    optimizer.zero_grad(set_to_none=True)
    moved = {name: value.to(device) if torch.is_tensor(value) else value for name, value in batch.items()}
    result = training_long_rollout(model, moved, contract["steps"],
                                  physical_weights=contract["autoregression"]["physical_weights"])
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


def _run_updates(dataset, folder, model, optimizer, contract, signature, deadline):
    epoch, cursor, losses, checkpoint = 0, 0, [], None
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
        metrics = _update(model, optimizer, batch, contract, device, deadline)
        cursor += len(indices)
        losses.append({"update": update, "epoch": epoch, "lr": rate, "samples": len(indices), **metrics})
        if deadline is not None:
            _check_deadline(deadline)
        if update % contract["checkpoint_every"] == 0 or update == contract["total_updates"]:
            checkpoint = _publish_checkpoint(folder, update, model=model, optimizer=optimizer,
                                             signature=signature, contract=contract,
                                             epoch=epoch, cursor=cursor, intervention=None)
            load_checkpoint(checkpoint, expected=signature)
    return checkpoint, losses


def fine_tune_long_rollout(manifest, output, *, model, contract, physical_weights, parent_weights=None,
                          steps=4, updates=200, seed=0, lr=2e-5, warmup=10, weight_decay=1e-4,
                          deadline=None, checkpoint_every=20, device_name="cpu", batch_size=1, clip=1.):
    weights = validate_physical_weights(physical_weights)
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
    dataset = ZarrLongRolloutDataset(manifest, physical_steps=len(weights),
        expected_exclusions=declared.get("excluded_sample_ids", ()),
        expected_window_sha256=declared["window_sha256"])
    device = select_device(device_name)
    bound = _contract(dataset, supplied, output=output, steps=steps, updates=updates, seed=seed,
                      lr=lr, warmup=warmup, weight_decay=weight_decay, device=device,
                      batch_size=batch_size, clip=clip, checkpoint_every=checkpoint_every,
                      physical_weights=weights)
    signature = canonical_digest(bound)
    if deadline is not None:
        _check_deadline(deadline)
    folder, _ = _paths(output, None, signature, bound)
    started = time.perf_counter()
    try:
        seed_everything(seed)
        model.to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        checkpoint, losses = _run_updates(dataset, folder, model, optimizer, bound, signature, deadline)
        if deadline is not None:
            _check_deadline(deadline)
        report = {"scientific_claim": False, "limitations": LIMITATIONS, "test_read": False,
                  "contract": bound, "signature": signature, "protocol_sha256": bound["protocol_sha256"],
                  "model_code_sha256": bound["model_code_sha256"], "source_sha256": bound["source_sha256"],
                  "data_identity": bound["data_identity"], "optimization": "full-bptt",
                  "internal_k_detach": False, "physical_step_detach": False,
                  "objective": "deep_supervised_latitude_area_mse", "internal_deep_supervision": True,
                  "selected_checkpoint": str(checkpoint), "selected_update": updates, "total_updates": updates,
                  "selection_split": None, "selection_metric": "frozen endpoint; no validation selection",
                  "resumed_from_updates": 0, "updates_this_run": len(losses), "losses": losses,
                  "elapsed_seconds": time.perf_counter() - started, "initialization": bound["initialization"],
                  "parent_optimizer_imported": False, "bf16": False, "device_type": device.type,
                  "windows": dataset.summary,
                  "reproducibility": "config-reproducible; no failed-attempt resume or bitwise GPU promise"}
        _write_json(folder / "training_report.json", report)
        return checkpoint, report
    except BaseException as exc:
        model.zero_grad(set_to_none=True)
        _write_json(folder / "failed_attempt.json", {"status": "failed", "scientific_claim": False,
                    "limitations": LIMITATIONS, "signature": signature, "test_read": False,
                    "failure_reason": f"{type(exc).__name__}: {exc}", "resume_permitted": False})
        raise
