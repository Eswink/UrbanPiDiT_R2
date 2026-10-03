"""CPU-only exact initialization pairing and measured full-objective backward cost."""
from __future__ import annotations

import hashlib
import math
import random
import time

from .r7_v2_protocol import B_SEEDS, C_SEEDS, CONTROLS, LIMITATIONS, arm_configs, digest


def seed_cpu(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)


def state_hash(state):
    import torch
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        value.update(name.encode() + b"\0" + str(tensor.dtype).encode() + b"\0")
        value.update(str(tuple(tensor.shape)).encode() + b"\0")
        value.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


def import_parent_model(parent, seed):
    from .r7_parent_import import import_parent
    seed_cpu(seed)
    model, report = import_parent(
        parent["checkpoint"], original_protocol=parent["original_protocol"], codezip=parent["codezip"],
        sidecar=parent["sidecar"], trainmanifest=parent["trainmanifest"],
        target_kind=parent["target_kind"], model_spec=parent["model_spec"])
    if digest(report) != parent["import_report_sha256"] or report != parent["import_report"]:
        raise ValueError("current weight-only parent import differs from frozen acceptance")
    return model.cpu(), report, parent["model_spec"]


def seeded_mapped_model(configuration, seed, arm):
    """No guessed mapping: all target tensors must be explicitly supplied by same anchor."""
    from .r7_experiment import make_model
    from .r7_parent_import import tensor_sha256
    import torch
    initial = configuration["initialization"]
    anchor_spec, target_spec = initial["anchor"], configuration["model_specs"][arm]
    seed_cpu(seed)
    anchor = make_model(anchor_spec["kind"], anchor_spec["model"]).cpu()
    source = anchor.state_dict()
    anchor_hash = state_hash(source)
    with torch.random.fork_rng(devices=[]):
        model = make_model(target_spec["kind"], target_spec["model"]).cpu()
    target, mapping = model.state_dict(), initial["mapping"][arm]
    if set(mapping) != set(target):
        raise ValueError("C explicit same-anchor mapping must cover every target tensor")
    copied, records = {}, []
    for name, source_key in sorted(mapping.items()):
        if (not isinstance(source_key, str) or source_key not in source
                or source[source_key].shape != target[name].shape or source[source_key].dtype != target[name].dtype):
            raise ValueError(f"C explicit target/anchor tensor shape/dtype mapping mismatch: {name}")
        tensor = source[source_key]
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError("C anchor tensor must be finite")
        copied[name] = tensor.detach().clone()
        records.append({"target_key": name, "source_key": source_key, "tensor_sha256": tensor_sha256(tensor)})
    model.load_state_dict(copied, strict=True)
    report = {"format": "r7-v2-seeded-mapped-initialization-v1", "seed": seed,
              "scientific_claim": False, "limitations": list(LIMITATIONS),
              "operation": "seeded scratch, explicit same-anchor tensor mapping; no historical score or optimizer",
              "anchor": anchor_spec, "anchor_state_sha256": anchor_hash,
              "target": target_spec, "target_state_sha256": state_hash(model.state_dict()),
              "mapping": records, "unused_anchor_keys": sorted(set(source) - set(mapping.values())),
              "uninitialized_target_keys": [], "optimizer_reset": True, "resume": False}
    del source, copied, anchor
    return model, report, target_spec["model"]


def model_for_arm(stage, parents, configuration, seed, arm):
    if stage == "B":
        return import_parent_model(parents[str(seed)], seed)
    return seeded_mapped_model(configuration, seed, arm)


def objective(model, batch, mode, *, steps=4, lambda12=0.5):
    from .r7_autoregressive_rollout import training_one_step, training_two_step
    return (training_two_step(model, batch, reasoning_steps=steps, lambda12=lambda12)
            if mode == "two_step" else training_one_step(model, batch, reasoning_steps=steps))


def measure_cost(model, batch, mode, *, check=lambda: None):
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("v2 preparation/profile must be CPU only")
    model.train()
    initial_hash = state_hash(model.state_dict())
    calls = []
    hook = model.register_forward_hook(lambda module, args, result: calls.append(1))
    started = time.perf_counter()
    try:
        check()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad(), FlopCounterMode(display=False) as counter:
            output = objective(model, batch, mode)
            forward = int(counter.get_total_flops())
        forward_calls = len(calls)
        if not bool(torch.isfinite(output.loss)) or not output.loss.requires_grad:
            raise ValueError("finite differentiable actual objective required")
        del output
        check()
        model.zero_grad(set_to_none=True)
        calls.clear()
        with torch.enable_grad(), FlopCounterMode(display=False) as counter:
            output = objective(model, batch, mode)
            output.loss.backward()
            forward_backward = int(counter.get_total_flops())
        expected = 2 if mode == "two_step" else 1
        if forward <= 0 or forward_backward < forward or forward_calls != expected or len(calls) != expected:
            raise ValueError("measured actual objective must execute exactly one/two forwards and its real backward")
        gradients = [(name, parameter.grad) for name, parameter in model.named_parameters()
                     if parameter.grad is not None]
        if not gradients or any(not bool(torch.isfinite(grad).all()) for _, grad in gradients):
            raise ValueError("actual CPU backward gradients missing or nonfinite")
        grad_norm = math.sqrt(sum(float(grad.detach().double().square().sum()) for _, grad in gradients))
        if state_hash(model.state_dict()) != initial_hash:
            raise ValueError("CPU objective profiling must not update initialization")
        check()
        return {"parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "forward_flops": forward, "forward_backward_flops": forward_backward,
                "actual_model_forward_calls": forward_calls, "objective": mode,
                "physical_steps": expected, "reasoning_steps": CONTROLS["steps"],
                "loss": float(output.loss.detach()), "l6": float(output.l6.detach()),
                "l12": None if output.l12 is None else float(output.l12.detach()),
                "gradient_norm": grad_norm, "gradient_tensors": len(gradients),
                "nonzero_gradient_tensors": sum(bool((grad != 0).any()) for _, grad in gradients),
                "elapsed_cpu_seconds": time.perf_counter() - started,
                "forward_scope": "full ordinary model forward(s) plus this arm's actual objective under enable_grad",
                "backward_scope": "actual loss.backward with full internal-K/physical graph; never assumed multiplier"}
    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)


def profile_arms(manifests, windows, parents, *, stage="B", configuration=None, check=lambda: None):
    from data.r7_autoregressive_dataset import ZarrAutoregressiveDataset
    from torch.utils.data import default_collate
    dataset = ZarrAutoregressiveDataset(manifests / "train.jsonl", expected_exclusions=windows["excluded_sample_ids"])
    if dataset.summary != windows:
        raise ValueError("profile dataset exact windows differ from preflight")
    probe = default_collate([dataset[0]])
    pairing, measurements = {}, {}
    for seed in B_SEEDS if stage == "B" else C_SEEDS:
        pairing[str(seed)], measurements[str(seed)] = {}, {}
        for arm, config in arm_configs(stage, configuration).items():
            check()
            model, initialization, spec = model_for_arm(stage, parents, configuration, seed, arm)
            pairing[str(seed)][arm] = {"full_initial_state_sha256": state_hash(model.state_dict()),
                                      "initialization_report_sha256": digest(initialization),
                                      "model_spec": dict(spec), "seed": seed,
                                      "anchor_state_sha256": initialization.get("anchor_state_sha256")}
            measurements[str(seed)][arm] = measure_cost(model, probe, config["mode"], check=check)
            del model
        if stage == "B" and len({record["full_initial_state_sha256"] for record in pairing[str(seed)].values()}) != 1:
            raise ValueError("all B arms of one seed must start from identical imported parent weights")
        if stage == "C" and len({record["anchor_state_sha256"] for record in pairing[str(seed)].values()}) != 1:
            raise ValueError("all C arms of one seed must use the same explicitly mapped seeded anchor")
    return {"scientific_claim": False, "limitations": list(LIMITATIONS), "test_read": False,
            "device": "cpu", "probe_batch_size": 1, "probe_sample_ids": list(probe["sample_id"]),
            "window_sha256": windows["window_sha256"], "pairing": pairing, "measurements": measurements,
            "flop_convention": "supported aten operations including real backward; elementwise/normalization omitted",
            "no_optimizer_step": True, "full_internal_k_bptt": True, "full_physical_step_bptt": True}
