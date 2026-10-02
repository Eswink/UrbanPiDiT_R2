"""Actual CPU full-forward/new-loss cost and first-train-case gradient ownership."""
from __future__ import annotations

import hashlib
import math
import random

import numpy as np
import torch

from .r7_m3_protocol import ARM_NAMES, LIMITATIONS, REASONING_STEPS, SEEDS, WEIGHTS, model_config


def seed_cpu(seed):
    """CPU preparation must not initialize CUDA or consume any GPU budget."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)


def state_hash(state):
    hasher = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        hasher.update(name.encode() + b"\0" + str(value.dtype).encode() + b"\0")
        hasher.update(str(tuple(value.shape)).encode() + b"\0")
        hasher.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def parameter_groups(model):
    groups = {name: [] for name in ("shared", "query", "readout")}
    for name, parameter in model.named_parameters():
        owner = "query" if name == "process_queries" else (
            "readout" if name.startswith("process_readout.") else "shared")
        groups[owner].append((name, parameter))
    if any(not values for values in groups.values()):
        raise ValueError("CPU gradient profile requires nonempty shared/query/readout groups")
    return groups


def _loss(model, batch, context, weights, loss_fn):
    from model.r7_halting import forecast_inputs
    output = model(forecast_inputs(batch), reasoning_steps=REASONING_STEPS)
    loss = loss_fn(batch, output, process_weight=0.0,
                   process_supervision_context=context, **weights)
    if not bool(torch.isfinite(loss.total)) or not loss.total.requires_grad:
        raise ValueError("CPU full-forward/new-loss profile requires a finite differentiable total")
    return output, loss


def measure_cost(model, batch, context, weights, *, loss_fn=None):
    from torch.utils.flop_counter import FlopCounterMode
    if loss_fn is None:
        from .r7_process_forecast_losses import process_forecast_coreasoning_loss
        loss_fn = process_forecast_coreasoning_loss
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("cost preparation is CPU-only")
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.enable_grad(), FlopCounterMode(display=False) as counter:
        output, loss = _loss(model, batch, context, weights, loss_fn)
        forward = int(counter.get_total_flops())
    del output, loss
    model.zero_grad(set_to_none=True)
    with torch.enable_grad(), FlopCounterMode(display=False) as counter:
        _, loss = _loss(model, batch, context, weights, loss_fn)
        loss.total.backward()
        backward = int(counter.get_total_flops())
    model.zero_grad(set_to_none=True)
    if forward <= 0 or backward < forward:
        raise ValueError("actual full-forward plus new loss FLOP count unavailable")
    return {"parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "forward_flops": forward, "forward_backward_flops": backward,
            "forward_scope": "full forward plus this arm's actual new losses under torch.enable_grad",
            "backward_scope": "measured actual total.backward, full-forward graph; not estimated 2x"}


def _gradient_record(names, values):
    present = [(name, grad) for name, grad in zip(names, values) if grad is not None]
    if any(not bool(torch.isfinite(grad).all()) for _, grad in present):
        raise ValueError("nonfinite CPU gradient probe")
    squares = [float(grad.detach().double().square().sum()) for _, grad in present]
    norm = math.sqrt(sum(squares))
    return {"norm": norm, "grad_tensors": len(present), "missing_grads": len(names) - len(present),
            "nonzero_tensors": sum(bool((grad != 0).any()) for _, grad in present),
            "gradient_parameter_names": [name for name, _ in present], "parameter_names": names}


def gradient_profile(model, batch, context, weights, *, loss_fn=None):
    from model.r7_halting import forecast_inputs
    if loss_fn is None:
        from .r7_process_forecast_losses import process_forecast_coreasoning_loss
        loss_fn = process_forecast_coreasoning_loss
    case = dict(batch)
    case["coarse_history"] = batch["coarse_history"].detach().clone().requires_grad_(True)
    model.train()
    model.zero_grad(set_to_none=True)
    groups = parameter_groups(model)
    named = [(name, parameter) for values in groups.values() for name, parameter in values]
    with torch.enable_grad():
        output = model(forecast_inputs(case), reasoning_steps=REASONING_STEPS)
        tensors = [p for _, p in named] + [case["coarse_history"], output.draft_forecasts]
        breakdown = loss_fn(case, output, process_weight=0.0,
                            process_supervision_context=context, **weights)
        components = {
            "forecast": breakdown.forecast,
            "input": weights["input_diagnostic_weight"] * breakdown.input_diagnostics,
            "future": weights["future_diagnostic_weight"] * breakdown.future_diagnostic_targets,
            "draft": weights["draft_diagnostic_weight"] * breakdown.draft_diagnostics,
            "total": breakdown.total,
        }
        records = {}
        for component, loss in components.items():
            gradients = (torch.autograd.grad(loss, tensors, retain_graph=True, allow_unused=True)
                         if loss.requires_grad else [None] * len(tensors))
            by_name = {name: grad for (name, _), grad in zip(named, gradients)}
            record = {label: _gradient_record([name for name, _ in values],
                                              [by_name[name] for name, _ in values])
                      for label, values in groups.items()}
            record["forecast_history"] = _gradient_record(["coarse_history"], [gradients[-2]])
            record["forecast_drafts"] = _gradient_record(["draft_forecasts"], [gradients[-1]])
            records[component] = {"loss": float(loss.detach()), "groups": record}
    model.zero_grad(set_to_none=True)
    return {"scientific_claim": False, "limitations": LIMITATIONS,
            "scope": "first actual train case, CPU full forward, component weights frozen; no optimizer step",
            "sample_id": batch["sample_id"], "components": records,
            "readout_group": "process_readout only; positional process_reader remains shared solver pathway",
            "causal_evidence": False}


def profile_arms(probe, first_case, context):
    from .r7_experiment import make_model
    channels = probe["coarse_history"].shape[2]
    measured, pairing, gradients = {}, {}, {}
    for seed in SEEDS:
        hashes = {}
        for arm in ARM_NAMES:
            seed_cpu(seed)
            model = make_model("process", model_config(channels)).cpu()
            hashes[arm] = state_hash(model.state_dict())
            if seed == SEEDS[0]:
                measured[arm] = measure_cost(model, probe, context, WEIGHTS[arm])
                gradients[arm] = gradient_profile(model, first_case, context, WEIGHTS[arm])
            del model
        identical = len(set(hashes.values())) == 1
        if not identical:
            raise ValueError("same-seed M3 full initial state is not bitwise paired")
        pairing[str(seed)] = {"all_shared_pairs_identical": True, "same_tensor_set": True,
                              "full_initial_state_sha256": hashes, "seed": seed,
                              "rule": "all arms use exactly the same RW_A config and CPU RNG stream"}
    return {"scientific_claim": False, "limitations": LIMITATIONS, "device": "cpu",
            "measurements": measured, "pairing": pairing, "gradient_ownership": gradients,
            "probe_batch_size": len(probe["coarse_history"]), "gradient_probe_seed": SEEDS[0],
            "reasoning_steps": REASONING_STEPS,
            "flop_convention": (
                "FlopCounterMode counts supported aten linear/conv/matmul/attention including losses/backward; "
                "elementwise future/draft diagnostic arithmetic and normalization are uncounted")}
