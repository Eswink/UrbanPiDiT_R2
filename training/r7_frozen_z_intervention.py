"""Driver-only frozen random-Z intervention on the existing RW-B solver cell.

This is an engineering control, not a scientific result. CPU FP32 independent
normal draws at SOLVER_INITIAL_SCALE (0.02) are a random-distribution implementation
choice, NOT a scientific threshold. One persistent [1,N,D] buffer is reused across
samples, internal steps and physical transitions; the original cell still runs,
and only its output is discarded. No solver module/parameter is replaced.

``token_hw`` means the PATCH grid: (ceil(H/patch_size), ceil(W/patch_size)). The
caller must explicitly declare this intervention and carry the JSON spec in its
checkpoint contract. Install before strict load, validate after load/before use,
and construct the optimizer from trainable parameters after installation. The
existing r7_experiment checkpoint/code-identity checks remain mandatory/unchanged.
The canonical buffer stays FP32; autocast/output casting is supported, but a
whole-model dtype conversion after installation changes its bytes and is rejected.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import weakref

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from model.local_solver_state_r7 import SOLVER_INITIAL_SCALE, LocalSolverState
from model.process_forecast_r7 import ProcessForecastCoReasoner

FROZEN_Z_FORMAT = "r7-frozen-z-v1"
FROZEN_Z_SEED_OFFSET = 1_000_003
FROZEN_Z_BUFFER_NAME = "_frozen_z_value"
SPEC_KEYS = frozenset(("format", "seed", "random_seed", "shape", "token_hw",
                       "patch_size", "std", "tensor_sha256"))
__all__ = ["make_spec", "install_frozen_z", "validate_frozen_z"]


def _positive_int(value: int, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _token_grid(token_hw) -> tuple[int, int]:
    if type(token_hw) not in (tuple, list) or len(token_hw) != 2:
        raise ValueError("token_hw must contain exactly two positive integers")
    return tuple(_positive_int(value, "token_hw") for value in token_hw)


def _tensor_digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _spec_digest(spec: dict) -> str:
    return hashlib.sha256(json.dumps(
        spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _spec_and_value(seed, token_hw, dim, patch_size) -> tuple[dict, torch.Tensor]:
    if type(seed) is not int or not 0 <= seed <= 2**64 - 1 - FROZEN_Z_SEED_OFFSET:
        raise ValueError("seed must be a nonnegative integer with seed+offset < 2**64")
    rows, columns = _token_grid(token_hw)
    dim = _positive_int(dim, "dim")
    patch_size = _positive_int(patch_size, "patch_size")
    random_seed = seed + FROZEN_Z_SEED_OFFSET
    generator = torch.Generator(device="cpu").manual_seed(random_seed)
    value = torch.randn(1, rows * columns, dim, generator=generator,
                        device="cpu", dtype=torch.float32) * SOLVER_INITIAL_SCALE
    spec = {"format": FROZEN_Z_FORMAT, "seed": seed, "random_seed": random_seed,
            "shape": list(value.shape), "token_hw": [rows, columns],
            "patch_size": patch_size, "std": SOLVER_INITIAL_SCALE,
            "tensor_sha256": _tensor_digest(value)}
    return spec, value


def make_spec(seed: int, token_hw: tuple[int, int], dim: int, patch_size: int) -> dict:
    """Return a JSON-only deterministic reconstruction spec without global RNG use.

    Exact bytes/digest must reconstruct on the executing torch implementation;
    no cross-version bitwise guarantee is assumed or silently waived.
    """
    return _spec_and_value(seed, token_hw, dim, patch_size)[0]


def _reconstruct_spec(spec: dict) -> tuple[dict, torch.Tensor]:
    if type(spec) is not dict or set(spec) != SPEC_KEYS:
        raise ValueError("frozen Z spec has missing/extra fields or is not a JSON object")
    if spec["format"] != FROZEN_Z_FORMAT:
        raise ValueError("unsupported frozen Z spec format")
    if type(spec["token_hw"]) is not list:
        raise ValueError("frozen Z spec token_hw must be a JSON list")
    shape = spec["shape"]
    if type(shape) is not list or len(shape) != 3 or any(type(x) is not int for x in shape):
        raise ValueError("frozen Z spec shape must be a three-integer JSON list")
    if type(spec["random_seed"]) is not int or type(spec["std"]) is not float:
        raise ValueError("frozen Z spec random_seed/std have invalid JSON types")
    expected, value = _spec_and_value(
        spec["seed"], spec["token_hw"], shape[-1], spec["patch_size"])
    if spec != expected:
        raise ValueError("frozen Z spec deterministic reconstruction/digest mismatch")
    return expected, value


def _require_rw_b(model: nn.Module, spec: dict) -> None:
    if not isinstance(model, ProcessForecastCoReasoner):
        raise ValueError("frozen Z requires the existing RW-B process forecaster")
    for name in ("local_solver_state", "solver_state_recurrence", "solver_gate_proposal"):
        if getattr(model, name, None) is not True:
            raise ValueError(f"frozen Z requires RW-B {name}=True")
    if not isinstance(model.solver_init, nn.Parameter) or not isinstance(
            model.solver_cell, LocalSolverState):
        raise ValueError("frozen Z requires the original RW-B solver_init/solver_cell")
    if model.dim != spec["shape"][-1] or model.solver_cell.dim != model.dim \
            or tuple(model.solver_init.shape) != (1, 1, model.dim):
        raise ValueError("frozen Z dim/solver_init shape differs from RW-B")
    if model.patch_size != spec["patch_size"]:
        raise ValueError("frozen Z patch_size differs from RW-B")
    if model.solver_init.device.type == "meta":
        raise ValueError("frozen Z requires materialized, non-meta RW-B parameters")


def _check_value(value, spec: dict, label: str) -> None:
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided \
            or value.device.type == "meta":
        raise ValueError(f"{label} must be a materialized strided tensor")
    if list(value.shape) != spec["shape"] or value.dtype != torch.float32:
        raise ValueError(f"{label} shape/dtype differs from the canonical CPU FP32 spec")
    if value.requires_grad:
        raise ValueError(f"{label} must be frozen (requires_grad=False)")
    if _tensor_digest(value) != spec["tensor_sha256"]:
        raise ValueError(f"{label} tensor bytes digest mismatch")


@dataclass
class _FrozenZControl:
    spec: dict
    spec_sha256: str
    parameters: dict[str, nn.Parameter]
    requires_grad: dict[str, bool]
    solver_cell: LocalSolverState
    forward_handle: RemovableHandle | None = None
    load_handle: RemovableHandle | None = None


def _installed_control(model: nn.Module) -> _FrozenZControl:
    control = getattr(model, "_frozen_z_control", None)
    if not isinstance(control, _FrozenZControl):
        raise ValueError("frozen Z intervention is not installed")
    return control


def _check_installed(model: nn.Module, control: _FrozenZControl) -> None:
    _require_rw_b(model, control.spec)
    if _spec_digest(control.spec) != control.spec_sha256:
        raise ValueError("installed frozen Z spec digest mismatch")
    if model.solver_cell is not control.solver_cell:
        raise ValueError("original solver_cell module identity changed")
    parameters = dict(model.named_parameters(remove_duplicate=False))
    if parameters.keys() != control.parameters.keys() or any(
            parameters[name] is not original for name, original in control.parameters.items()):
        raise ValueError("original RW-B parameter identity/set changed")
    if any(parameter.requires_grad != control.requires_grad[name]
           for name, parameter in parameters.items()):
        raise ValueError("RW-B parameter freeze/requires_grad flags changed")
    if FROZEN_Z_BUFFER_NAME not in model._buffers \
            or FROZEN_Z_BUFFER_NAME in model._non_persistent_buffers_set:
        raise ValueError("frozen Z persistent buffer missing or nonpersistent")
    _check_value(model._buffers[FROZEN_Z_BUFFER_NAME], control.spec, "frozen Z buffer")
    if control.forward_handle is None \
            or control.forward_handle.id not in model.solver_cell._forward_hooks:
        raise ValueError("frozen Z forward hook missing")
    if control.load_handle is None \
            or control.load_handle.id not in model._load_state_dict_pre_hooks:
        raise ValueError("frozen Z checkpoint buffer guard missing")


def validate_frozen_z(model: nn.Module, spec: dict) -> str:
    """Check reconstruction, buffer bytes, freezing and parameter identity.

    Return the canonical JSON spec SHA256 for the caller's contract/evidence.
    Checks are also applied by the output hook (without resampling) and by this
    module's state_dict load pre-hook; r7_experiment is not modified or bypassed.
    """
    expected, _ = _reconstruct_spec(spec)
    control = _installed_control(model)
    digest = _spec_digest(expected)
    if digest != control.spec_sha256:
        raise ValueError("received frozen Z spec digest differs from installed intervention")
    _check_installed(model, control)
    return digest


def install_frozen_z(model: nn.Module, spec: dict) -> RemovableHandle:
    """Set up the intervention before strict checkpoint load/optimizer creation.

    Only solver_init and solver_cell parameters are frozen, with stale gradients
    cleared. All original parameter objects/values, modules and other grad flags
    remain unchanged. The returned handle removes ONLY output replacement (useful
    for the engineering counterproof); a removed hook makes validation fail and
    does not permit a second installation. Setup performs no global RNG operation
    and no implicit CUDA discovery/allocation: generation is explicitly on CPU,
    and the buffer is then placed on the caller-selected solver_init device.
    """
    if hasattr(model, "_frozen_z_control") or hasattr(model, FROZEN_Z_BUFFER_NAME):
        raise ValueError("frozen Z already installed or reserved buffer/control name occupied")
    expected, value = _reconstruct_spec(spec)
    _require_rw_b(model, expected)
    parameters = dict(model.named_parameters(remove_duplicate=False))
    frozen = {id(model.solver_init), *(id(p) for p in model.solver_cell.parameters())}
    flags = {name: False if id(p) in frozen else p.requires_grad
             for name, p in parameters.items()}
    control = _FrozenZControl(expected, _spec_digest(expected), parameters, flags,
                              model.solver_cell)
    owner = weakref.ref(model)

    def replace_output(module, args, kwargs, output):
        parent = owner()
        _check_installed(parent, control)
        grid = _token_grid(kwargs.get("token_hw"))
        if list(grid) != control.spec["token_hw"]:
            raise ValueError("frozen Z token grid mismatch")
        state = args[0] if args else kwargs.get("z")
        if not isinstance(output, torch.Tensor) or output.ndim != 3 \
                or not isinstance(state, torch.Tensor) or output.shape != state.shape \
                or list(output.shape[1:]) != control.spec["shape"][1:]:
            raise ValueError("frozen Z solver output shape mismatch")
        # The cell has already run. Return the same stored values, not its output;
        # casting is local and never mutates/resamples the canonical buffer.
        return parent._frozen_z_value.to(device=output.device, dtype=output.dtype).expand(
            output.shape[0], -1, -1)

    def check_checkpoint_buffer(module, state_dict, prefix, local_metadata, strict,
                                missing_keys, unexpected_keys, error_msgs):
        parent = owner()
        _check_installed(parent, control)
        if local_metadata.get("assign_to_params_buffers", False):
            raise ValueError("frozen Z forbids assign=True (changes parameter identity)")
        key = prefix + FROZEN_Z_BUFFER_NAME
        if key not in state_dict:
            raise RuntimeError("checkpoint missing frozen Z buffer")
        _check_value(state_dict[key], control.spec, "checkpoint frozen Z buffer")

    model.register_buffer(FROZEN_Z_BUFFER_NAME, value.to(device=model.solver_init.device),
                          persistent=True)
    for parameter in parameters.values():
        if id(parameter) in frozen:
            parameter.requires_grad_(False)
            parameter.grad = None
    model._frozen_z_control = control
    control.forward_handle = model.solver_cell.register_forward_hook(
        replace_output, with_kwargs=True)
    control.load_handle = model.register_load_state_dict_pre_hook(check_checkpoint_buffer)
    _check_installed(model, control)
    return control.forward_handle
