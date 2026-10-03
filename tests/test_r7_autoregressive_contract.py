"""CPU-only actual configuration guards; equal tensors do not prove semantics."""
from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from test_r7_m4_autoregressive_rollout import cpu_threads, tiny_spec
from training.r7_autoregressive_runner import _validate_imported_model
from training.r7_experiment import make_model


def contract_for(kind, spec):
    return {"kind": kind, "model": dict(spec), "data_identity": "a" * 64,
            "source_sha256": "b" * 64, "protocol_sha256": "c" * 64,
            "initialization": {"format": "explicit-cpu-contract-fixture"}}


def same_tensors(reference, actual):
    actual.load_state_dict(reference.state_dict(), strict=True)
    assert set(reference.state_dict()) == set(actual.state_dict())
    assert all(torch.equal(value, actual.state_dict()[name]) for name, value in reference.state_dict().items())


@pytest.mark.parametrize("kind", ["process", "generic"])
@pytest.mark.parametrize("key,changed", [("dropout", .2), ("heads", 1), ("window_size", 4),
                                         ("periodic_width", True), ("default_lead_hours", 12.),
                                         ("activation_checkpointing", True)])
def test_actual_hidden_constructor_semantics_rejected_with_identical_state(kind, key, changed):
    declared = tiny_spec(kind)
    reference = make_model(kind, declared)
    actual = make_model(kind, dict(declared, **{key: changed}))
    same_tensors(reference, actual)
    state_before = {name: tensor.clone() for name, tensor in actual.state_dict().items()}
    with pytest.raises(ValueError, match="model configuration differs"):
        _validate_imported_model(actual, contract_for(kind, declared), reference.state_dict())
    assert all(torch.equal(value, actual.state_dict()[name]) for name, value in state_before.items())


@pytest.mark.parametrize("kind", ["process", "generic"])
@pytest.mark.parametrize("path,field,value", [
    ("backbone.encoder.blocks.0.ff.net.2", "p", .2),
    ("backbone.encoder.blocks.0", "heads", 1),
    ("backbone.encoder.blocks.0", "hd", 8),
    ("backbone.encoder.blocks.0", "ws", 4),
    ("backbone.encoder.blocks.0", "shift", True),
    ("backbone.encoder.blocks.0", "periodic_width", True),
    ("backbone.encoder", "use_checkpointing", True),
    ("backbone.spacetime", "field_mode", "constant"),
    ("backbone.spacetime", "default_lead_hours", 12.),
    ("process_reader.attention", "heads", 1),
    ("process_reader.attention", "dropout", .2),
    ("process_reader", "pooled_readout_query", True),
    ("backbone.head.decode.0", "stride", (1, 1)),
])
def test_nested_metadata_cannot_hide_behind_matching_parent_constructor(kind, path, field, value):
    spec = dict(tiny_spec(kind), positional_process_readout=True)
    reference, actual = make_model(kind, spec), make_model(kind, spec)
    same_tensors(reference, actual)
    setattr(actual.get_submodule(path), field, value)
    with pytest.raises(ValueError, match="model configuration differs"):
        _validate_imported_model(actual, contract_for(kind, spec), None)


@pytest.mark.parametrize("kind", ["process", "generic"])
def test_unspecified_defaults_checked_and_same_contract_modes_accepted(kind):
    declared = tiny_spec(kind)
    reference, actual = make_model(kind, declared), make_model(kind, declared)
    same_tensors(reference, actual)
    actual.use_forecast_feedback = False  # omitted spec key, but active forward control
    with pytest.raises(ValueError, match="use_forecast_feedback"):
        _validate_imported_model(actual, contract_for(kind, declared), None)
    for overrides in ({}, {"dropout": .2, "heads": 1, "window_size": 4, "periodic_width": True,
                           "activation_checkpointing": True, "default_lead_hours": 12.},
                      {"spacetime_field_mode": "constant", "use_forecast_feedback": False},
                      {"local_solver_state": True, "positional_process_readout": True,
                       "solver_state_recurrence": True, "solver_gate_proposal": True,
                       "source_role_markers": True, "pooled_readout_query": True}):
        spec = dict(declared, **overrides)
        model = make_model(kind, spec).eval()  # runtime mode is not constructor semantics
        before = torch.get_rng_state().clone()
        contract = contract_for(kind, spec)
        _validate_imported_model(model, contract, None)
        assert torch.equal(before, torch.get_rng_state())
        assert len(contract["model_semantics_sha256"]) == 64
        assert len(contract["initial_weights_sha256"]) == 64


@pytest.mark.parametrize("kind", ["process", "generic"])
def test_module_type_topology_and_semantic_digest_tamper_rejected(kind):
    spec = tiny_spec(kind)
    for fault in ("type", "topology", "order", "digest"):
        model = make_model(kind, spec)
        contract = contract_for(kind, spec)
        if fault == "type":
            model.backbone.encoder.blocks[0].ff.net[2] = nn.Identity()
        elif fault == "topology":
            model.backbone.encoder.blocks[0].ff.net.append(nn.Identity())
        elif fault == "order":
            modules = model.backbone.encoder.blocks[0].ff.net._modules
            modules["1"] = modules.pop("1")
        else:
            contract["model_semantics_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="module topology/order differs|module type differs|model_semantics_sha256"):
            _validate_imported_model(model, deepcopy(contract), None)
