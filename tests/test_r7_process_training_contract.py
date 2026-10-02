"""Sidecar/fixed-inverse/signature gates for M3, synthetic records only."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from training.r7_process_training_contract import (
    supervision_training_contract, verify_evaluation_sidecar,
)

DATA_IDENTITY = "d" * 64


def supplied_contract():
    return {"sidecar_path": "unused-sidecar", "sidecar_identity": "a" * 64,
            "protocol_sha256": "b" * 64, "input_diagnostic_weight": .1,
            "future_diagnostic_weight": 0., "draft_diagnostic_weight": 0.}


def mock_published(monkeypatch):
    import data.preprocess.r7_process_scale_sidecar as scale
    import training.r7_process_training_contract as contract
    metadata = {"sidecar_identity": "a" * 64, "fit_split": "train", "store": "fixed-store",
                "data_identity": DATA_IDENTITY}
    inverse = {"channel_names": [f"channel{i}" for i in range(17)],
               "normalization_mean": [0.] * 17, "normalization_std": [1.] * 17,
               "data_identity": DATA_IDENTITY}
    monkeypatch.setattr(scale, "load_process_scale_sidecar", lambda path: deepcopy(metadata))
    def actual(meta, identity):
        if identity != DATA_IDENTITY or meta["data_identity"] != identity:
            raise ValueError("sidecar data identity mismatch")
        return deepcopy(inverse)
    monkeypatch.setattr(contract, "_actual_train_inverse", actual)
    monkeypatch.setattr(contract, "_root_inverse", lambda root, identity: deepcopy(inverse))
    return SimpleNamespace(metadata=deepcopy(metadata), channel_names=inverse["channel_names"].copy(),
                           normalization_mean=torch.zeros(17), normalization_std=torch.ones(17))


def bind(context, supplied=None, **kwargs):
    return supervision_training_contract(context, supplied or supplied_contract(),
        kind="process", process_weight=0, data_identity=DATA_IDENTITY, **kwargs)


def test_legacy_path_keeps_the_original_contract():
    assert supervision_training_contract(None, None, kind="native", process_weight=.1) == (None, {})
    assert verify_evaluation_sidecar({}, None) is None


def test_sidecar_protocol_and_fixed_inverse_digests_are_bound(monkeypatch):
    context = mock_published(monkeypatch)
    bound, kwargs = bind(context)
    assert bound["sidecar_identity"] == "a" * 64
    assert bound["protocol_sha256"] == "b" * 64
    assert bound["fixed_train_context"]["data_identity"] == DATA_IDENTITY
    assert len(bound["fixed_train_context_sha256"]) == 64
    assert kwargs["process_supervision_context"] is context
    assert kwargs["input_diagnostic_weight"] == .1


@pytest.mark.parametrize("field,value", [("sidecar_identity", "c" * 64),
    ("protocol_sha256", "not-a-digest"), ("input_diagnostic_weight", -1.),
    ("future_diagnostic_weight", float("nan")), ("draft_diagnostic_weight", True)])
def test_invalid_identity_and_weight_are_refused(monkeypatch, field, value):
    context = mock_published(monkeypatch)
    with pytest.raises(ValueError):
        bind(context, dict(supplied_contract(), **{field: value}))


def test_context_cannot_silently_differ_from_sidecar(monkeypatch):
    context = mock_published(monkeypatch)
    context.metadata["fit_split"] = "val"
    with pytest.raises(ValueError, match="context differs"):
        bind(context)


@pytest.mark.parametrize("field", ["normalization_mean", "normalization_std", "channel_names"])
def test_fixed_inverse_and_channel_order_cannot_keep_the_same_signature(monkeypatch, field):
    context = mock_published(monkeypatch)
    if field == "channel_names":
        context.channel_names = list(reversed(context.channel_names))
    else:
        getattr(context, field)[0] += 1.
    with pytest.raises(ValueError, match="normalization/channel order"):
        bind(context)


def test_sidecar_must_match_actual_runner_data_identity(monkeypatch):
    context = mock_published(monkeypatch)
    with pytest.raises(ValueError, match="data identity"):
        supervision_training_contract(context, supplied_contract(), kind="process", process_weight=0,
                                      data_identity="e" * 64)


@pytest.mark.parametrize("kind,weight", [("native", 0.), ("process", .1)])
def test_new_and_legacy_tasks_cannot_be_mixed(monkeypatch, kind, weight):
    context = mock_published(monkeypatch)
    with pytest.raises(ValueError):
        supervision_training_contract(context, supplied_contract(), kind=kind, process_weight=weight)


def test_evaluation_requires_the_declared_sidecar_and_fixed_store(monkeypatch):
    context = mock_published(monkeypatch)
    contract = {"process_supervision": bind(context)[0], "data_identity": DATA_IDENTITY}
    with pytest.raises(ValueError, match="explicit scale sidecar"):
        verify_evaluation_sidecar(contract, None)
    with pytest.raises(ValueError, match="explicit held-out"):
        verify_evaluation_sidecar(contract, "unused-sidecar")
    options = {"evaluation_store": "fixed-store", "evaluation_root": object()}
    assert verify_evaluation_sidecar(contract, "unused-sidecar", **options) == "a" * 64
    with pytest.raises(ValueError, match="held-out store"):
        verify_evaluation_sidecar(contract, "unused-sidecar", evaluation_store="other-store",
                                  evaluation_root=object())
    contract["process_supervision"]["sidecar_identity"] = "c" * 64
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_evaluation_sidecar(contract, "unused-sidecar", **options)


@pytest.mark.parametrize("field", ["fixed_train_context", "fixed_train_context_sha256"])
def test_evaluation_refuses_drift_in_checkpoint_inverse(monkeypatch, field):
    context = mock_published(monkeypatch)
    contract = {"process_supervision": bind(context)[0], "data_identity": DATA_IDENTITY}
    contract["process_supervision"][field] = {} if field == "fixed_train_context" else "f" * 64
    with pytest.raises(ValueError, match="fixed train context"):
        verify_evaluation_sidecar(contract, "unused-sidecar", evaluation_store="fixed-store",
                                  evaluation_root=object())


def test_legacy_checkpoint_does_not_accept_new_sidecar_override():
    with pytest.raises(ValueError, match="legacy checkpoint"):
        verify_evaluation_sidecar({}, "unused-sidecar")


def test_scale_audit_marks_true_zero_variance_and_avoids_none_formatting():
    import numpy as np
    from training.r7_process_diagnostic import proxy_scale_report, summarise_proxy_effect
    result = proxy_scale_report(np.full((128, 1), 4.2), [1e-6], ("constant",))
    assert result["channels"]["constant"]["raw_std"] == 0.
    assert result["channels"]["constant"]["degenerate"] is True
    assert result["channels"]["constant"]["active_mask"] is False
    assert result["channels"]["constant"]["normalized_std"] == 0.
    assert "masked" in summarise_proxy_effect(result, ("constant",))["constant"]


@pytest.mark.parametrize("std,names", [([0.], ("a",)), ([float("nan")], ("a",)),
                                      ([1., 1.], ("a", "a"))])
def test_scale_audit_refuses_bad_metadata(std, names):
    import numpy as np
    from training.r7_process_diagnostic import proxy_scale_report
    with pytest.raises(ValueError):
        proxy_scale_report(np.ones((4, len(names))), std, names)
