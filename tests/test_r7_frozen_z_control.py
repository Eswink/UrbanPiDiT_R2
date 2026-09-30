"""CPU engineering fixtures only: no real data, old checkpoints or science claims.

All disk artifacts live in tmp_path. The original RW-B modules execute unchanged;
these tests check the driver intervention and its rejection counterproofs.
"""
from __future__ import annotations

import copy
import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from model.process_forecast_r7 import ProcessForecastCoReasoner
from model.process_step_r7 import ProcessStepInput, process_reasoning_step
from model.r7_halting import AdaptiveProcessForecaster, forecast_inputs
from model.r7_rollout import autoregressive_rollout
from training.r7_experiment import canonical_digest, load_checkpoint, save_exclusive
from training.r7_frozen_z_intervention import (
    FROZEN_Z_BUFFER_NAME, FROZEN_Z_FORMAT, FROZEN_Z_SEED_OFFSET,
    install_frozen_z, make_spec, validate_frozen_z,
)
from training.r7_streaming import backward_streamed_truncated

SMALL = {"in_channels": 2, "out_channels": 2, "history_steps": 2, "dim": 8,
         "depth": 1, "heads": 2, "window_size": 4, "patch_size": 2, "dropout": 0.0,
         "anchored_processes": 2, "free_processes": 1, "default_reasoning_steps": 3,
         "use_forecast_feedback": True, "spacetime_inputs": True,
         "source_role_markers": True, "positional_process_readout": True,
         "local_solver_state": True}
HW = (5, 3)
TOKEN_HW = (3, 2)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def forbid_cuda(*args, **kwargs):
        raise AssertionError("CPU fixture must not seed or allocate CUDA")

    threads = torch.get_num_threads()
    cpu_rng = torch.get_rng_state()
    initialized = torch.cuda.is_initialized()
    for name in ("_lazy_init", "manual_seed", "manual_seed_all", "get_rng_state_all"):
        monkeypatch.setattr(torch.cuda, name, forbid_cuda)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # CPU AdamW health check
    torch.set_num_threads(1)
    yield
    assert torch.cuda.is_initialized() == initialized
    torch.set_rng_state(cpu_rng)
    torch.set_num_threads(threads)


def _model(seed=23, **options):
    # Model initialization may use the CPU default generator; rewind the fixture
    # stream and never call torch.manual_seed (which also seeds CUDA).
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.default_generator.manual_seed(seed)
        return ProcessForecastCoReasoner(**dict(SMALL, **options))


def _batch(seed=31, size=2, hw=HW):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows, columns = hw
    return {"coarse_history": torch.randn(size, 2, 2, rows, columns, generator=generator),
            "atmos_target": torch.randn(size, 2, rows, columns, generator=generator),
            "process_targets": torch.randn(size, 2, generator=generator),
            "lead_time_hours": torch.full((size,), 6.0),
            "latitude": torch.linspace(-30.0, 30.0, rows),
            "longitude": torch.linspace(0.0, 40.0, columns),
            "init_utc_hour": torch.full((size,), 3.0),
            "init_day_of_year": torch.full((size,), 40.0)}


def _spec(model, seed=41, hw=HW):
    token_hw = tuple((size + model.patch_size - 1) // model.patch_size for size in hw)
    return make_spec(seed, token_hw, model.dim, model.patch_size)


def _cell_inputs(seed=61, size=2, token_hw=TOKEN_HW):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (size, token_hw[0] * token_hw[1], SMALL["dim"])
    tensors = [torch.randn(shape, generator=generator) for _ in range(4)]
    return tensors[0], {"context": tensors[1], "draft_tokens": tensors[2],
                        "read": tensors[3], "step_index": 0, "token_hw": token_hw}


def _record_outputs(model):
    records = []
    handle = model.solver_cell.register_forward_hook(
        lambda module, args, output: records.append(output.detach().clone()))
    return records, handle


def _assert_frozen_records(model, records):
    assert records
    for value in records:
        assert torch.equal(value, model._frozen_z_value.expand(value.shape[0], -1, -1))


def test_spec_json_private_rng_and_exact_reconstruction(tmp_path, monkeypatch):
    model = _model()
    def forbid_discovery():
        raise AssertionError("frozen Z setup must not query CUDA")
    monkeypatch.setattr(torch.cuda, "is_available", forbid_discovery)
    rng, python_rng, numpy_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
    spec = _spec(model)
    path = tmp_path / "frozen_z_spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    restored = json.loads(path.read_text(encoding="utf-8"))
    expected = torch.randn(1, 6, 8, dtype=torch.float32, device="cpu",
                           generator=torch.Generator(device="cpu").manual_seed(
                               41 + FROZEN_Z_SEED_OFFSET)) * 0.02
    assert spec == restored == _spec(model)
    assert spec["format"] == FROZEN_Z_FORMAT
    assert spec["shape"] == [1, 6, 8] and spec["token_hw"] == [3, 2]
    assert spec["seed"] == 41 and spec["random_seed"] == 41 + FROZEN_Z_SEED_OFFSET
    assert spec["patch_size"] == 2 and spec["std"] == 0.02
    install_frozen_z(model, restored)
    assert torch.equal(model._frozen_z_value, expected)
    assert validate_frozen_z(model, restored) == canonical_digest(spec)
    assert torch.equal(torch.get_rng_state(), rng)
    assert random.getstate() == python_rng
    after = np.random.get_state()
    assert numpy_rng[0] == after[0] and np.array_equal(numpy_rng[1], after[1])
    assert numpy_rng[2:] == after[2:]


def test_seed_ignores_model_input_and_default_dtype_rng():
    first, second, third = _model(11), _model(99), _model(77)
    spec = _spec(first)
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        assert _spec(second) == spec
        install_frozen_z(first, spec)
        install_frozen_z(second, spec)
    finally:
        torch.set_default_dtype(dtype)
    install_frozen_z(third, _spec(third, seed=42))
    assert torch.equal(first._frozen_z_value, second._frozen_z_value)
    assert not torch.equal(first._frozen_z_value, third._frozen_z_value)
    assert first._frozen_z_value.dtype == torch.float32


def test_install_preserves_all_weights_parameters_modules_other_grad_flags_rng():
    model = _model()
    model.process_queries.requires_grad_(False)
    next(model.proposal_head.parameters()).requires_grad_(False)
    parameters = dict(model.named_parameters())
    flags = {name: value.requires_grad for name, value in parameters.items()}
    modules = dict(model.named_modules())
    state = {name: value.clone() for name, value in model.state_dict().items()}
    frozen_ids = {id(model.solver_init), *(id(p) for p in model.solver_cell.parameters())}
    for parameter in parameters.values():
        parameter.grad = torch.ones_like(parameter)
    gate_grad = next(model.solver_gate.parameters()).grad
    spec, rng = _spec(model), torch.get_rng_state()
    install_frozen_z(model, spec)
    assert torch.equal(rng, torch.get_rng_state())
    assert dict(model.named_modules()) == modules
    assert set(model.state_dict()) == set(state) | {FROZEN_Z_BUFFER_NAME}
    for name, original in parameters.items():
        assert dict(model.named_parameters())[name] is original
        assert torch.equal(model.state_dict()[name], state[name])
        assert original.requires_grad == (False if id(original) in frozen_ids else flags[name])
        if id(original) in frozen_ids:
            assert original.grad is None
    assert next(model.solver_gate.parameters()).grad is gate_grad
    assert validate_frozen_z(model, spec) == canonical_digest(spec)


def test_original_cell_runs_but_inputs_step_and_batch_never_change_z():
    model = _model()
    raw, raw_handle = _record_outputs(model)  # before intervention: observe real cell output
    install_frozen_z(model, _spec(model))
    buffer = model._frozen_z_value
    with torch.no_grad():
        for seed, size, step in ((61, 2, 0), (62, 2, 0), (61, 2, 7), (63, 1, 2), (64, 3, 9)):
            state, inputs = _cell_inputs(seed, size)
            output = model.solver_cell(state, **dict(inputs, step_index=step))
            assert output.shape == state.shape
            assert torch.equal(output, buffer.expand(size, -1, -1))
            assert output.data_ptr() == buffer.data_ptr()
            assert not output.requires_grad
    assert len(raw) == 5
    assert not torch.equal(raw[0], raw[1]) and not torch.equal(raw[0], raw[2])
    raw_handle.remove()


def test_changed_history_targets_and_physical_rollout_reuse_one_buffer():
    model = _model().eval()
    spec = _spec(model)
    install_frozen_z(model, spec)
    records, handle = _record_outputs(model)
    data = _batch()
    poisoned = dict(data, atmos_target=torch.full_like(data["atmos_target"], 1e5),
                    process_targets=torch.full_like(data["process_targets"], -1e5))
    with torch.no_grad():
        first = model(data, reasoning_steps=2)
        target_changed = model(poisoned, reasoning_steps=2)
        input_changed = model(_batch(32), reasoning_steps=4)
    assert torch.equal(first.forecast, target_changed.forecast)
    assert torch.equal(first.solver_state, input_changed.solver_state)
    histories = []
    history_handle = model.register_forward_pre_hook(
        lambda module, args: histories.append(args[0]["coarse_history"].clone()))
    rollout = autoregressive_rollout(model, data, lead_hours=(6, 12, 18),
                                     inference_kwargs={"reasoning_steps": 2})
    assert rollout.model_calls == 3 and len(histories) == 3
    assert not torch.equal(histories[0], histories[1])
    assert not torch.equal(histories[1], histories[2])
    assert len(records) == 14
    _assert_frozen_records(model, records)
    assert validate_frozen_z(model, spec) == canonical_digest(spec)
    handle.remove(), history_handle.remove()


def test_forward_streamed_truncated_and_force_full_adaptive_match_bitwise():
    model, data = _model().train(), _batch()
    spec = _spec(model)
    install_frozen_z(model, spec)
    records, handle = _record_outputs(model)
    with torch.no_grad():
        fixed = model(forecast_inputs(data), reasoning_steps=3).forecast
    streamed = backward_streamed_truncated(model, data, reasoning_steps=3, process_weight=0.0)
    with torch.random.fork_rng(devices=[]):
        adapter = AdaptiveProcessForecaster(model).eval()
    adaptive = adapter(data, max_steps=3, min_steps=3, force_full_depth=True)
    assert torch.equal(streamed.final_forecast, fixed)
    assert torch.equal(adaptive.forecast, fixed)
    assert (adaptive.reasoning_steps_per_sample == 3).all() and adaptive.active_masks.all()
    assert len(records) == 9
    _assert_frozen_records(model, records)
    assert model.solver_init.grad is None
    assert all(p.grad is None for p in model.solver_cell.parameters())
    assert any(p.grad is not None for p in model.proposal_head.parameters())
    assert any(p.grad is not None for p in model.solver_gate.parameters())
    assert validate_frozen_z(model, spec) == canonical_digest(spec)
    handle.remove()


def test_local_loss_gradients_only_proposal_gate_optimizer_preserves_frozen_values():
    model, data = _model().train(), _batch()
    spec = _spec(model)
    install_frozen_z(model, spec)
    frozen = {name: p.detach().clone() for name, p in model.named_parameters()
              if name == "solver_init" or name.startswith("solver_cell.")}
    buffer = model._frozen_z_value.clone()
    z, inputs = _cell_inputs()
    step = process_reasoning_step(model, ProcessStepInput(
        z[:, :3], inputs["context"], data["coarse_history"][:, -1]), TOKEN_HW,
        anchor=data["coarse_history"][:, -1])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=0.005, weight_decay=0.1)
    (step.draft - data["atmos_target"]).square().mean().backward()
    trained = {}
    for name, parameter in model.named_parameters():
        if name.startswith(("proposal_head.", "solver_gate.")):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            trained[name] = parameter.detach().clone()
        else:
            assert parameter.grad is None, name
    assert sum(p.grad.abs().sum() for p in model.proposal_head.parameters()) > 0
    assert sum(p.grad.abs().sum() for p in model.solver_gate.parameters()) > 0
    optimizer.step()
    assert any(not torch.equal(p, trained[name]) for name, p in model.named_parameters()
               if name in trained)
    for name, value in frozen.items():
        assert torch.equal(dict(model.named_parameters())[name], value)
    assert torch.equal(model._frozen_z_value, buffer)
    assert not model._frozen_z_value.requires_grad and model._frozen_z_value.grad is None
    assert validate_frozen_z(model, spec) == canonical_digest(spec)


def test_full_forward_preserves_non_solver_learning_and_cuts_cell_init_gradient():
    model, data = _model(), _batch()
    install_frozen_z(model, _spec(model))
    output = model(forecast_inputs(data), reasoning_steps=2)
    (output.forecast.square().mean() + output.process_predictions.square().mean()).backward()
    assert model.solver_init.grad is None and not model.solver_init.requires_grad
    assert all(p.grad is None and not p.requires_grad for p in model.solver_cell.parameters())
    for module in (model.proposal_head, model.solver_gate, model.backbone, model.reasoning_cell):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_k_zero_keeps_entire_backbone_and_initial_draft_bitwise_identical():
    original, controlled, data = _model().eval(), _model().eval(), _batch()
    install_frozen_z(controlled, _spec(controlled))
    with torch.no_grad():
        first, second = original(data, reasoning_steps=0), controlled(data, reasoning_steps=0)
    for name in ("forecast", "initial_forecast", "draft_forecasts", "process_predictions",
                 "final_correction", "process_state", "context_tokens"):
        assert torch.equal(getattr(first, name), getattr(second, name)), name
    assert first.token_hw == second.token_hw == TOKEN_HW
    assert first.solver_state is second.solver_state is None


def test_shared_step_keeps_process_and_prediction_bitwise_identical():
    original, controlled, data = _model().eval(), _model().eval(), _batch()
    install_frozen_z(controlled, _spec(controlled))
    with torch.no_grad():
        base = original.backbone(forecast_inputs(data))
        tensors = ProcessStepInput(original.process_queries.expand(2, -1, -1),
                                   base.context_tokens, base.forecast)
        kwargs = {"solver_state": original.solver_init.expand(2, 6, -1),
                  "step_index": 2, "anchor": base.base_state}
        first = process_reasoning_step(original, tensors, TOKEN_HW, **kwargs)
        second = process_reasoning_step(controlled, tensors, TOKEN_HW, **kwargs)
    assert torch.equal(first.process, second.process)
    assert torch.equal(first.prediction, second.prediction)
    assert not torch.equal(first.solver_state, second.solver_state)
    assert torch.equal(second.solver_state, controlled._frozen_z_value.expand(2, -1, -1))
    assert not torch.equal(first.draft, second.draft)


def test_removing_hook_restores_original_input_dependence_and_invalidates_control():
    model = _model().eval()
    first, first_args = _cell_inputs(61)
    second, second_args = _cell_inputs(62)
    with torch.no_grad():
        baseline = model.solver_cell(first, **first_args)
        other = model.solver_cell(second, **second_args)
    assert not torch.equal(baseline, other)
    spec = _spec(model)
    handle = install_frozen_z(model, spec)
    with torch.no_grad():
        assert torch.equal(model.solver_cell(first, **first_args),
                           model.solver_cell(second, **second_args))
        handle.remove()
        assert torch.equal(model.solver_cell(first, **first_args), baseline)
        assert torch.equal(model.solver_cell(second, **second_args), other)
    with pytest.raises(ValueError, match="forward hook missing"):
        validate_frozen_z(model, spec)
    with pytest.raises(ValueError, match="already installed"):
        install_frozen_z(model, spec)


def test_strict_checkpoint_round_trip_registers_buffer_before_load(tmp_path):
    model = _model().eval()
    spec = _spec(model)
    install_frozen_z(model, spec)
    contract = {"kind": "process", "intervention": {"name": "frozen_random_z", "spec": spec},
                "scientific_claim": False, "limitations": ["CPU engineering fixture only"]}
    path = tmp_path / "fixture_checkpoint.pt"
    save_exclusive(path, {"format": "r7-local-v1", "contract": contract,
                          "signature": canonical_digest(contract), "model": model.state_dict()})
    checkpoint = load_checkpoint(path, expected=canonical_digest(contract))
    restored = _model(77).eval()
    with pytest.raises(RuntimeError, match="Unexpected key"):
        restored.load_state_dict(checkpoint["model"], strict=True)
    install_frozen_z(restored, checkpoint["contract"]["intervention"]["spec"])
    parameters = dict(restored.named_parameters())
    result = restored.load_state_dict(checkpoint["model"], strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert validate_frozen_z(restored, spec) == canonical_digest(spec)
    for name, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value), name
    for name, value in parameters.items():
        assert dict(restored.named_parameters())[name] is value
    with torch.no_grad():
        assert torch.equal(model(_batch()).forecast, restored(_batch()).forecast)


@pytest.mark.parametrize("field,value", [
    ("format", "unknown"), ("seed", -1), ("seed", True), ("seed", 2**64),
    ("random_seed", 0), ("random_seed", True), ("shape", [1, 7, 8]),
    ("shape", [2, 6, 8]), ("shape", [1, 6, 0]), ("shape", [1, 6, True]),
    ("shape", "1,6,8"), ("token_hw", [0, 2]), ("token_hw", [3]),
    ("token_hw", (3, 2)), ("std", 0.03), ("std", 0),
    ("tensor_sha256", "0" * 64), ("patch_size", 0),
])
def test_tampered_spec_is_rejected_on_install_and_validation(field, value):
    model = _model()
    spec = _spec(model)
    damaged = dict(spec, **{field: value})
    with pytest.raises(ValueError):
        install_frozen_z(model, damaged)
    assert not hasattr(model, FROZEN_Z_BUFFER_NAME)
    assert model.solver_init.requires_grad
    install_frozen_z(model, spec)
    with pytest.raises(ValueError):
        validate_frozen_z(model, damaged)


@pytest.mark.parametrize("damage", ["not_object", "extra_key", "missing_key"])
def test_spec_schema_rejection_counterproof(damage):
    model = _model()
    spec = _spec(model)
    damaged = [] if damage == "not_object" else dict(spec)
    if damage == "extra_key":
        damaged["unused"] = 1
    elif damage == "missing_key":
        damaged.pop("tensor_sha256")
    with pytest.raises(ValueError, match="missing/extra fields"):
        install_frozen_z(model, damaged)


@pytest.mark.parametrize("options", [
    {"seed": -1}, {"seed": True}, {"seed": 2**64}, {"dim": 0}, {"dim": True},
    {"patch_size": 0}, {"patch_size": 1.5}, {"token_hw": [0, 2]},
    {"token_hw": [3]}, {"token_hw": "3,2"},
])
def test_make_spec_invalid_geometry_seed_counterproof(options):
    with pytest.raises(ValueError):
        make_spec(**dict({"seed": 41, "token_hw": TOKEN_HW, "dim": 8, "patch_size": 2},
                         **options))


@pytest.mark.parametrize("change", ["seed", "grid", "patch"])
def test_valid_but_different_spec_identity_is_rejected(change):
    model = _model()
    spec = _spec(model)
    install_frozen_z(model, spec)
    other = make_spec(42 if change == "seed" else 41,
                      (2, 3) if change == "grid" else TOKEN_HW,
                      8, 3 if change == "patch" else 2)
    with pytest.raises(ValueError, match="received frozen Z spec digest"):
        validate_frozen_z(model, other)


@pytest.mark.parametrize("field", ["local_solver_state", "solver_state_recurrence",
                                    "solver_gate_proposal"])
def test_wrong_rw_b_switch_install_guard(field):
    model = _model(**{field: False})
    with pytest.raises(ValueError, match=field):
        install_frozen_z(model, _spec(model))
    assert not hasattr(model, FROZEN_Z_BUFFER_NAME)


@pytest.mark.parametrize("damage", ["model_type", "init_type", "cell_type", "dim",
                                    "init_shape", "patch", "meta"])
def test_wrong_solver_structure_install_guard(damage):
    model = _model()
    spec = _spec(model)
    if damage == "model_type":
        model = nn.Identity()
    elif damage == "init_type":
        del model._parameters["solver_init"]
        model.solver_init = torch.zeros(1, 1, 8)
    elif damage == "cell_type":
        model.solver_cell = nn.Identity()
    elif damage == "dim":
        model.solver_cell.dim = 10
    elif damage == "init_shape":
        model.solver_init = nn.Parameter(torch.zeros(1, 2, 8))
    elif damage == "patch":
        model.patch_size = 3
    else:
        model.to("meta")
    with pytest.raises(ValueError):
        install_frozen_z(model, spec)


@pytest.mark.parametrize("name", [FROZEN_Z_BUFFER_NAME, "_frozen_z_control"])
def test_reserved_name_and_duplicate_install_guard(name):
    model = _model()
    setattr(model, name, None)
    with pytest.raises(ValueError, match="already installed"):
        install_frozen_z(model, _spec(model))


@pytest.mark.parametrize("grid", [(2, 3), (2, 2)])
def test_runtime_grid_mismatch_guard_including_same_n(grid):
    model = _model()
    install_frozen_z(model, _spec(model))
    state, inputs = _cell_inputs(token_hw=grid)
    with pytest.raises(ValueError, match="token grid mismatch"):
        model.solver_cell(state, **inputs)


@pytest.mark.parametrize("damage", ["bytes", "dtype", "shape", "grad", "missing",
                                    "nonpersistent", "non_tensor", "sparse", "meta"])
def test_buffer_integrity_freeze_persistence_guard(damage):
    model = _model()
    spec = _spec(model)
    install_frozen_z(model, spec)
    if damage == "bytes":
        model._frozen_z_value.add_(1)
    elif damage == "dtype":
        model._frozen_z_value = model._frozen_z_value.double()
    elif damage == "shape":
        model._frozen_z_value = model._frozen_z_value[:, :-1]
    elif damage == "grad":
        model._frozen_z_value.requires_grad_(True)
    elif damage == "missing":
        del model._buffers[FROZEN_Z_BUFFER_NAME]
    elif damage == "nonpersistent":
        model._non_persistent_buffers_set.add(FROZEN_Z_BUFFER_NAME)
    else:
        model._buffers[FROZEN_Z_BUFFER_NAME] = {
            "non_tensor": None, "sparse": model._frozen_z_value.to_sparse(),
            "meta": model._frozen_z_value.to("meta")}[damage]
    with pytest.raises(ValueError):
        validate_frozen_z(model, spec)


@pytest.mark.parametrize("damage", ["parameter", "parameter_set", "cell_module", "init_grad",
                                    "cell_grad", "proposal_grad", "gate_grad", "switch",
                                    "stored_spec", "load_hook"])
def test_parameter_identity_grad_flags_and_installed_contract_guards(damage):
    model = _model()
    spec = _spec(model)
    install_frozen_z(model, spec)
    if damage == "parameter":
        model.process_queries = nn.Parameter(model.process_queries.detach().clone())
    elif damage == "parameter_set":
        model.extra = nn.Parameter(torch.zeros(1))
    elif damage == "cell_module":
        model.solver_cell = copy.deepcopy(model.solver_cell)
    elif damage == "init_grad":
        model.solver_init.requires_grad_(True)
    elif damage == "cell_grad":
        next(model.solver_cell.parameters()).requires_grad_(True)
    elif damage == "proposal_grad":
        next(model.proposal_head.parameters()).requires_grad_(False)
    elif damage == "gate_grad":
        next(model.solver_gate.parameters()).requires_grad_(False)
    elif damage == "switch":
        model.solver_state_recurrence = 1
    elif damage == "stored_spec":
        model._frozen_z_control.spec["seed"] += 1
    else:
        model._frozen_z_control.load_handle.remove()
    with pytest.raises(ValueError):
        validate_frozen_z(model, spec)


@pytest.mark.parametrize("damage", ["bytes", "dtype", "shape", "grad", "missing", "assign"])
def test_checkpoint_buffer_tamper_rejected_before_any_parameter_copy(damage):
    model = _model()
    spec = _spec(model)
    install_frozen_z(model, spec)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    before = {name: value.clone() for name, value in state.items()}
    state["solver_init"].add_(10)  # ensure a rejected load cannot silently mutate parameters
    if damage == "bytes":
        state[FROZEN_Z_BUFFER_NAME].add_(1)
    elif damage == "dtype":
        state[FROZEN_Z_BUFFER_NAME] = state[FROZEN_Z_BUFFER_NAME].double()
    elif damage == "shape":
        state[FROZEN_Z_BUFFER_NAME] = state[FROZEN_Z_BUFFER_NAME][:, :-1]
    elif damage == "grad":
        state[FROZEN_Z_BUFFER_NAME].requires_grad_(True)
    elif damage == "missing":
        state.pop(FROZEN_Z_BUFFER_NAME)
    with pytest.raises((ValueError, RuntimeError)):
        model.load_state_dict(state, strict=True, assign=damage == "assign")
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), name
    assert validate_frozen_z(model, spec) == canonical_digest(spec)


def test_nested_state_dict_prefix_strict_guard_and_uninstalled_validation():
    first, second = _model(), _model(77)
    spec = _spec(first)
    with pytest.raises(ValueError, match="not installed"):
        validate_frozen_z(first, spec)
    install_frozen_z(first, spec), install_frozen_z(second, spec)
    source, restored = nn.Module(), nn.Module()
    source.add_module("forecaster", first), restored.add_module("forecaster", second)
    state = source.state_dict()
    restored.load_state_dict(state, strict=True)
    assert validate_frozen_z(second, spec) == canonical_digest(spec)
    state["forecaster." + FROZEN_Z_BUFFER_NAME] = first._frozen_z_value + 1
    with pytest.raises(ValueError, match="checkpoint frozen Z buffer.*digest"):
        restored.load_state_dict(state, strict=True)


@pytest.mark.parametrize("damage", ["shape", "type"])
def test_output_shape_guard_counterproof(damage):
    model = _model()
    install_frozen_z(model, _spec(model))
    model.solver_cell.register_forward_hook(
        lambda module, args, output: output[:, :-1] if damage == "shape" else "bad",
        prepend=True)
    state, inputs = _cell_inputs()
    with pytest.raises(ValueError, match="output shape mismatch"):
        model.solver_cell(state, **inputs)


def test_bf16_output_cast_never_changes_canonical_buffer():
    model = _model()
    spec = _spec(model)
    install_frozen_z(model, spec)
    state, inputs = _cell_inputs()
    inputs = {key: value.to(torch.bfloat16) if isinstance(value, torch.Tensor) else value
              for key, value in inputs.items()}
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model.solver_cell(state.bfloat16(), **inputs)
    assert output.dtype == torch.bfloat16
    assert torch.equal(output, model._frozen_z_value.bfloat16().expand(2, -1, -1))
    assert model._frozen_z_value.dtype == torch.float32
    assert validate_frozen_z(model, spec) == canonical_digest(spec)
