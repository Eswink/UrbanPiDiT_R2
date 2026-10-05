"""Thin full-BPTT fine-tuning on frozen exact two-step train cases.

The caller owns protocol freeze, verified parent/source import and orchestration.
This module owns only train updates and r7-local-v1 publication, reusing the
scheduled runner's schedule/checkpoint utilities, never its streamed backward.
Resume may continue an interrupted *own intermediate* checkpoint toward the
same frozen endpoint. A recorded failed attempt or completed run cannot revive.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import time

import torch
from torch.utils.data import default_collate

from data.r7_autoregressive_dataset import ZarrAutoregressiveDataset
from .r7_autoregressive_rollout import training_one_step, training_two_step
from .r7_experiment import (canonical_digest, dataset_identity, load_checkpoint,
                           make_model, model_code_digest, restore_rng,
                           seed_everything, select_device)
from .r7_local_runner import _integer
from .r7_scheduled_runner import _check_deadline, _publish_checkpoint, warmup_cosine_factor

ROOT = Path(__file__).resolve().parents[1]
LIMITATIONS = [
    "Training loss is normalized deep-supervised area MSE, not held-out forecast skill.",
    "Full BPTT over internal K and physical steps differs from a truncated parent training path.",
    "Both arms retain initial/all-K-draft deep supervision; neither adds a diagnostic/process objective.",
    "Resume is tested on same-software CPU only; no cross-platform bitwise promise.",
    "Parent/source qualification, offline execution and protocol archive are caller responsibilities.",
]
SOURCE_FILES = (
    "training/r7_autoregressive_runner.py", "training/r7_autoregressive_rollout.py",
    "data/r7_autoregressive_dataset.py", "data/r7_store.py", "data/r7_zarr_dataset.py",
    "training/r7_halting.py", "training/r7_experiment.py", "training/r7_scheduled_runner.py",
    "training/r7_local_runner.py", "training/r7_recursive_losses.py", "training/r7_losses.py",
)


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a SHA256 digest")
    return value


def _state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode() + b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _training_code_digest():
    return canonical_digest({name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                             for name in SOURCE_FILES})


def _finite_state(model):
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()
           if value.is_floating_point()):
        raise ValueError("nonfinite model state; checkpoint publication forbidden")


def _validate_options(*, steps, updates, seed, mode, lr, warmup, weight_decay,
                      bf16, batch_size, clip, lambda12, checkpoint_every, deadline):
    for value, name, minimum in ((steps, "steps", 1), (updates, "updates", 1), (seed, "seed", 0),
                                 (warmup, "warmup", 0), (batch_size, "batch_size", 1),
                                 (checkpoint_every, "checkpoint_every", 1)):
        _integer(value, name, minimum)
    if mode not in ("l6", "two_step") or type(bf16) is not bool:
        raise ValueError("mode must be l6/two_step and bf16 must be boolean")
    for value, name, positive in ((lr, "lr", True), (clip, "clip", True),
                                  (weight_decay, "weight_decay", False), (lambda12, "lambda12", False)):
        if (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
                or value < 0 or (positive and value == 0)):
            raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    if warmup > updates:
        raise ValueError("warmup cannot exceed the frozen update endpoint")
    if deadline is not None:
        if isinstance(deadline, bool) or not isinstance(deadline, (float, int)) or not math.isfinite(deadline):
            raise ValueError("deadline must be a finite absolute perf_counter limit")
        _check_deadline(deadline)


def _bind(target, values):
    for name, value in values.items():
        if name in target and target[name] != value:
            raise ValueError(f"frozen contract field differs from actual fine_tune: {name}")
        target[name] = value


def _semantic_value(value):
    """Typed constructor metadata, not parameter/buffer values or training mode."""
    if value is None or type(value) in (bool, int, float, str):
        return {"type": type(value).__name__, "value": value}
    if isinstance(value, (tuple, list)):
        return {"type": type(value).__name__, "value": [_semantic_value(item) for item in value]}
    raise ValueError(f"unsupported public model configuration value: {type(value).__name__}")


def _module_semantics(model):
    """Read actual controls at every module, including unspecified spec defaults.

    Attention heads/windows, functional attention dropout and nn.Dropout.p do
    not enter state_dict. Nor do periodic padding, checkpointing, default lead,
    readout/solver switches, convolution stride or LayerNorm eps. Compare the
    real module tree against make_model(spec), never an asserted config label.
    Registered trained tensors remain subject to the existing strict state load.
    """
    result = {}
    for path, module in model.named_modules():
        fields = {name: _semantic_value(value) for name, value in vars(module).items()
                  if not name.startswith("_") and name != "training" and not torch.is_tensor(value)}
        result[path] = {"type": type(module).__module__ + "." + type(module).__qualname__, "fields": fields}
    return result


def _check_model_semantics(model, template):
    actual, expected = _module_semantics(model), _module_semantics(template)
    if list(actual) != list(expected):
        raise ValueError("imported model module topology/order differs from the new contract")
    for path, reference in expected.items():
        if actual[path]["type"] != reference["type"]:
            raise ValueError(f"imported model module type differs: {path or '<root>'}")
        fields, target = actual[path]["fields"], reference["fields"]
        for name in sorted(set(fields) | set(target)):
            if fields.get(name) != target.get(name):
                raise ValueError(f"imported model configuration differs: {path + '.' if path else ''}{name}")
    return canonical_digest(expected)


def _validate_imported_model(model, contract, parent_weights):
    if contract.get("kind") not in ("process", "generic") or not isinstance(contract.get("model"), dict):
        raise ValueError("fine_tune requires a declared Generic/Process model contract")
    spec = contract["model"]
    if spec.get("detach_between_steps", False) is not False:
        raise ValueError("new model contract must declare full BPTT, detach_between_steps=False")
    spec["detach_between_steps"] = False
    if not isinstance(contract.get("initialization"), dict) or not contract["initialization"]:
        raise ValueError("explicit verified parent or seeded/shared-weights initialization report required")
    for name in ("data_identity", "source_sha256", "protocol_sha256"):
        _digest(contract.get(name), name)
    # Prove the saved state can be restored by evaluate_local's unmodified
    # make_model path. Constructing this shape template does not move the RNG.
    with torch.random.fork_rng(devices=[]):
        template = make_model(contract["kind"], spec)
    if type(model) is not type(template):
        raise ValueError("imported model type differs from the declared evaluation model")
    expected, actual = template.state_dict(), model.state_dict()
    if set(expected) != set(actual) or any(expected[key].shape != actual[key].shape for key in expected):
        raise ValueError("imported model keys/shapes differ from the new contract")
    _bind(contract, {"model_semantics_sha256": _check_model_semantics(model, template)})
    if parent_weights is not None:
        if (not isinstance(parent_weights, dict) or not parent_weights
                or any(not torch.is_tensor(value) for value in parent_weights.values())):
            raise ValueError("parent_weights must be only a verified model state_dict, never an optimizer/checkpoint")
        model.load_state_dict(parent_weights, strict=True)
    if any(not parameter.requires_grad or parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("fine_tune requires every forecasting parameter trainable and FP32 (BF16 uses autocast)")
    if contract.get("process_supervision") is not None or contract.get("intervention") is not None:
        raise ValueError("new atmospheric-only fine_tune must not inherit parent process supervision/intervention")
    _finite_state(model)
    initial_sha = _state_digest(model.state_dict())
    _bind(contract, {"initial_weights_sha256": initial_sha})


def _contract(dataset, supplied, *, output, steps, updates, seed, mode, lr, warmup,
              weight_decay, bf16, device, batch_size, clip, lambda12, checkpoint_every):
    contract = deepcopy(supplied)
    actual_identity, _ = dataset_identity(dataset.manifest)
    if actual_identity != contract["data_identity"]:
        raise ValueError("actual training data/normalization identity differs from the frozen contract")
    _bind(contract, {"model_code_sha256": model_code_digest(),
                     "training_code_sha256": _training_code_digest(),
                     "steps": steps, "total_updates": updates, "seed": seed, "mode": mode,
                     "lr": lr, "warmup_updates": warmup, "weight_decay": weight_decay,
                     "bf16": bf16, "device_type": device.type, "torch_version": str(torch.__version__),
                     "batch_size": batch_size, "clip": clip, "dataset_length": len(dataset),
                     "checkpoint_every": checkpoint_every, "optimization": "full-bptt",
                     "schedule": "linear-warmup-then-cosine", "minimum_lr_ratio": .1,
                     "process_weight": 0., "output_dir": str(Path(output).resolve())})
    autoregression = contract.setdefault("autoregression", {})
    _bind(autoregression, {"mode": mode, "physical_steps": 2 if mode == "two_step" else 1,
                          "step_hours": 6, "fixed_transition_lead_hours": 6,
                          "lambda12": lambda12 if mode == "two_step" else 0.,
                          "detach_physical_steps": False, "detach_reasoning_steps": False,
                          "internal_deep_supervision": True,
                          "objective": "deep_supervised_latitude_area_mse", "loss_space": "normalized",
                          "final_weight": 2.,
                          "k_weights": "linspace(1,2,K+1) normalized by their sum; initial plus every K draft",
                          "physical_loss": "L6 + lambda12 * L12" if mode == "two_step" else "L6",
                          "window_sha256": dataset.summary["window_sha256"], "windows": dataset.summary})
    autoregression["excluded_sample_ids"] = dataset.summary["excluded_sample_ids"]
    return contract


def _write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _output_path(output, contract):
    """No linked/protected destination or nesting inside an unrelated old run."""
    requested = Path(output).absolute()
    if ("://" in str(output) or ".." in requested.parts
            or any(path.is_symlink() for path in (requested, *requested.parents))):
        raise ValueError("local output without symlink ancestors or traversal required")
    folder, parts = requested.resolve(), tuple(part.lower() for part in requested.parts)
    if (any("legacy" in part for part in parts)
            or any(parts[index:index + 2] in (("data", "raw"), ("data", "interim"), ("data", "processed"))
                   for index in range(len(parts) - 1))):
        raise ValueError("protected data/legacy output path forbidden")
    if folder.is_relative_to(ROOT) and not folder.is_relative_to(ROOT / "outputs"):
        raise ValueError("repository training writes must remain under outputs")
    for parent in folder.parents:
        if any((parent / name).exists() for name in ("training_report.json", "failed_attempt.json", "attempt.json")):
            raise ValueError("cannot nest a new arm inside a completed/failed original output")
        if (parent / "fine_tune_started.json").exists():
            raise ValueError("cannot nest a new arm inside another fine_tune attempt")
    if "outputs" in parts and folder.parent.name != "outputs":
        ancestor = next((parent for parent in folder.parents if parent.parent.name == "outputs"), None)
        if (ancestor is None or not re.fullmatch(
                r"r7_(?:74_autoregressive|v2_comparison|s3_[a-z0-9_]+)_\d{8}_attempt\d{2,}", ancestor.name)
                or not (ancestor / "protocol.json").is_file()):
            raise ValueError("nested original outputs require a new frozen v2 attempt root")
        if (ancestor / "protocol.json").is_symlink():
            raise ValueError("linked output ancestor protocol forbidden")
        protocol = json.loads((ancestor / "protocol.json").read_text(encoding="utf-8"))
        if protocol.get("protocol_sha256") != contract.get("protocol_sha256"):
            raise ValueError("output ancestor protocol differs from the new training contract")
    return folder


def _paths(output, resume, signature, contract):
    folder = _output_path(output, contract)
    if resume is None:
        folder.mkdir(parents=True, exist_ok=False)
        _write_json(folder / "fine_tune_started.json", {"format": "r7-autoregressive-attempt-v1",
                    "signature": signature, "contract": contract, "scientific_claim": False,
                    "limitations": LIMITATIONS})
        return folder, None
    if (folder / "failed_attempt.json").exists() or (folder / "training_report.json").exists():
        raise FileExistsError("failed/completed attempt cannot be revived")
    path = Path(resume).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)) or path.resolve().parent != folder:
        raise ValueError("resume requires this endpoint's own intermediate checkpoint in the same output")
    if any((folder / name).is_symlink() for name in
           ("fine_tune_started.json", "training_report.json", "failed_attempt.json")):
        raise ValueError("linked attempt metadata forbidden")
    started = json.loads((folder / "fine_tune_started.json").read_text(encoding="utf-8"))
    if (started.get("format") != "r7-autoregressive-attempt-v1"
            or started.get("signature") != signature or started.get("contract") != contract):
        raise ValueError("resume attempt/new contract differs; the endpoint cannot be extended")
    saved = load_checkpoint(path, expected=signature)
    _integer(saved.get("updates"), "saved updates")
    if saved["updates"] >= contract["total_updates"]:
        raise ValueError("only an intermediate checkpoint at the same declared endpoint may resume")
    if path.name != f"update_{saved['updates']:07d}.pt":
        raise ValueError("checkpoint filename/update mismatch")
    latest = max(folder.glob("update_*.pt"), key=lambda item: item.name, default=None)
    if latest is None or path.resolve() != latest.resolve():
        raise ValueError("only the latest own intermediate checkpoint may resume; no attempt fork")
    return folder, saved


def _restore_optimizer(model, optimizer, saved, dataset_length, batch_size, contract):
    if saved is None:
        return 0, 0, 0
    updates = saved["updates"]
    per_epoch = math.ceil(dataset_length / batch_size)
    epoch = (updates - 1) // per_epoch
    cursor = min(((updates - 1) % per_epoch + 1) * batch_size, dataset_length)
    if saved.get("epoch") != epoch or saved.get("cursor") != cursor:
        raise ValueError("checkpoint epoch/cursor differs from the frozen deterministic sampling schedule")
    payload = saved["optimizer"]
    groups = payload.get("param_groups", [])
    lr = contract["lr"] * warmup_cosine_factor(updates, total_updates=contract["total_updates"],
                                               warmup_updates=contract["warmup_updates"], minimum_ratio=.1)
    if (len(groups) != 1 or groups[0].get("weight_decay") != contract["weight_decay"]
            or groups[0].get("lr") != lr or tuple(groups[0].get("betas", ())) != (.9, .999)
            or groups[0].get("eps") != 1e-8 or groups[0].get("amsgrad") is not False):
        raise ValueError("checkpoint optimizer differs from the frozen fresh AdamW")
    for state in payload["state"].values():
        if (set(state) != {"step", "exp_avg", "exp_avg_sq"}
                or not all(torch.is_tensor(value) and torch.isfinite(value).all() for value in state.values())
                or float(state["step"]) != updates):
            raise ValueError("checkpoint optimizer state/update is nonfinite or inconsistent")
    model.load_state_dict(saved["model"], strict=True)
    _finite_state(model)
    optimizer.load_state_dict(payload)
    for parameter, state in optimizer.state.items():
        if state["exp_avg"].shape != parameter.shape or state["exp_avg_sq"].shape != parameter.shape:
            raise ValueError("checkpoint optimizer tensor shape mismatch")
    restore_rng(saved["rng"])
    return updates, epoch, cursor


def _update(model, optimizer, batch, *, mode, steps, bf16, device, clip, lambda12, deadline):
    optimizer.zero_grad(set_to_none=True)
    moved = {name: value.to(device) if torch.is_tensor(value) else value for name, value in batch.items()}
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
        result = (training_two_step(model, moved, steps, lambda12) if mode == "two_step"
                  else training_one_step(model, moved, steps))
    result.loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    if deadline is not None:
        _check_deadline(deadline)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    _finite_state(model)
    return {"loss": float(result.loss.detach()), "l6": float(result.l6.detach()),
            "l12": None if result.l12 is None else float(result.l12.detach()), "gradient_norm": float(norm)}


def _run_updates(dataset, folder, model, optimizer, saved, contract, signature, deadline):
    updates, epoch, cursor = _restore_optimizer(
        model, optimizer, saved, len(dataset), contract["batch_size"], contract)
    resumed_updates = updates
    losses, checkpoint = [], None
    while updates < contract["total_updates"]:
        if deadline is not None:
            _check_deadline(deadline)
        if cursor == len(dataset):
            epoch, cursor = epoch + 1, 0
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(contract["seed"] + epoch))
        indices = order[cursor:min(cursor + contract["batch_size"], len(dataset))].tolist()
        batch = default_collate([dataset[index] for index in indices])
        rate = contract["lr"] * warmup_cosine_factor(updates + 1, total_updates=contract["total_updates"],
                                                     warmup_updates=contract["warmup_updates"], minimum_ratio=.1)
        optimizer.param_groups[0]["lr"] = rate
        metrics = _update(model, optimizer, batch, mode=contract["mode"], steps=contract["steps"],
                          bf16=contract["bf16"], device=next(model.parameters()).device,
                          clip=contract["clip"], lambda12=contract["autoregression"]["lambda12"], deadline=deadline)
        updates, cursor = updates + 1, cursor + len(indices)
        losses.append({"update": updates, "epoch": epoch, "lr": rate, "samples": len(indices), **metrics})
        if deadline is not None:
            _check_deadline(deadline)
        if updates % contract["checkpoint_every"] == 0 or updates == contract["total_updates"]:
            checkpoint = _publish_checkpoint(folder, updates, model=model, optimizer=optimizer,
                                             signature=signature, contract=contract,
                                             epoch=epoch, cursor=cursor, intervention=None)
            # Verify the ordinary evaluator checkpoint contract, not a private format.
            loaded = load_checkpoint(checkpoint, expected=signature)
            if loaded["model_code_sha256"] != contract["model_code_sha256"]:
                raise ValueError("model implementation changed during training")
    return checkpoint, losses, resumed_updates


def fine_tune(manifest, output, *, model, contract, parent_weights=None, steps=4, updates=200,
              seed=0, mode="two_step", lr=1e-4, warmup=20, weight_decay=1e-4, bf16=False,
              deadline=None, resume=None, checkpoint_every=20, device_name="cpu", batch_size=1,
              clip=1., lambda12=.5):
    """Fine-tune an already qualified Generic/Process model; return (Path, report).

    ``contract`` requires kind/model/data_identity/source_sha256/protocol_sha256
    and a nonempty ``initialization`` report (verified parent import or explicit
    seeded/shared initialization). ``autoregression.excluded_sample_ids`` is the
    predeclared metadata-only boundary exclusion list. ``parent_weights`` is an
    optional strict model state_dict, NEVER the parent's optimizer. Resume loads
    this run's optimizer/RNG only, with same output, endpoint and full signature.
    ``deadline`` is an absolute perf_counter hard limit supplied by orchestration.
    """
    _validate_options(steps=steps, updates=updates, seed=seed, mode=mode, lr=lr, warmup=warmup,
                      weight_decay=weight_decay, bf16=bf16, batch_size=batch_size, clip=clip,
                      lambda12=lambda12, checkpoint_every=checkpoint_every, deadline=deadline)
    if not isinstance(contract, dict) or not isinstance(contract.get("autoregression", {}), dict):
        raise ValueError("a frozen JSON contract with an autoregression mapping is required")
    supplied = deepcopy(contract)
    # Reject unsafe destinations before reading any training fields or importing weights.
    output = _output_path(output, supplied)
    _validate_imported_model(model, supplied, parent_weights)
    dataset = ZarrAutoregressiveDataset(manifest, expected_exclusions=supplied.get(
        "autoregression", {}).get("excluded_sample_ids", ()))
    device = select_device(device_name, bf16)
    bound = _contract(dataset, supplied, output=output, steps=steps, updates=updates, seed=seed,
                      mode=mode, lr=lr, warmup=warmup, weight_decay=weight_decay, bf16=bf16,
                      device=device, batch_size=batch_size, clip=clip, lambda12=lambda12,
                      checkpoint_every=checkpoint_every)
    signature = canonical_digest(bound)
    if deadline is not None:
        _check_deadline(deadline)
    folder, saved = _paths(output, resume, signature, bound)
    started = time.perf_counter()
    try:
        seed_everything(seed)
        model.to(device).train()
        optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                      lr=lr, weight_decay=weight_decay)
        checkpoint, losses, resumed_updates = _run_updates(
            dataset, folder, model, optimizer, saved, bound, signature, deadline)
        if deadline is not None:
            _check_deadline(deadline)
        report = {"scientific_claim": False, "limitations": LIMITATIONS, "test_read": False,
                  "contract": bound, "signature": signature, "protocol_sha256": bound["protocol_sha256"],
                  "model_code_sha256": bound["model_code_sha256"], "source_sha256": bound["source_sha256"],
                  "data_identity": bound["data_identity"], "optimization": "full-bptt",
                  "internal_k_detach": False, "physical_step_detach": False,
                  "objective": "deep_supervised_latitude_area_mse", "internal_deep_supervision": True,
                  "loss_axes": "physical L6 + lambda12*L12; each loss normalizes linspace(1,2,K+1) over initial/all K drafts",
                  "selected_checkpoint": str(checkpoint), "selected_update": updates, "total_updates": updates,
                  "selection_split": None, "selection_metric": "frozen endpoint; no validation selection",
                  "resumed_from_updates": resumed_updates, "updates_this_run": len(losses),
                  "losses": losses, "elapsed_seconds": time.perf_counter() - started,
                  "initialization": bound["initialization"], "parent_optimizer_imported": False,
                  "reproducibility": "same-endpoint RNG/optimizer continuation; tested same-software CPU",
                  "bf16": bf16, "device_type": device.type, "windows": dataset.summary}
        _write_json(folder / "training_report.json", report)
        return checkpoint, report
    except BaseException as exc:
        model.zero_grad(set_to_none=True)
        try:
            _write_json(folder / "failed_attempt.json", {"status": "failed", "scientific_claim": False,
                        "limitations": LIMITATIONS, "signature": signature, "test_read": False,
                        "failure_reason": f"{type(exc).__name__}: {exc}", "resume_permitted": False})
        except BaseException as additional:
            exc.add_note(f"failure receipt publication failed: {additional}")
        raise
