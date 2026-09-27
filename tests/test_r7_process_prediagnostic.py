"""Tests for the #65 process pre-diagnostic: proxy scale, gradient conflict,
and the K0..K trajectory. Counterproofs are included so a broken audit cannot
pass by always reporting "usable"."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.r7_process_diagnostic import (  # noqa: E402
    DEFAULT_EPS, gradient_conflict_record, measurement_input_contract,
    normalized_proxy_scale, parameter_groups, proxy_scale_report,
    recursion_trajectory, summarise_proxy_effect,
)


def test_scale_audit_flags_a_floored_moisture_like_channel():
    generator = np.random.default_rng(7)
    # Real spread 1e-8, i.e. two orders below the 1e-6 floor.
    values = generator.normal(0.0, 1e-8, size=4096)
    record = normalized_proxy_scale(values)
    assert record["floor_active"] is True
    assert record["usable_label"] is False
    assert record["raw_std"] < DEFAULT_EPS
    assert record["normalized_std"] < 0.05
    assert record["amplification_lost_factor"] > 10


def test_scale_audit_accepts_a_normally_scaled_channel():
    generator = np.random.default_rng(11)
    values = generator.normal(0.0, 3.0, size=4096)
    record = normalized_proxy_scale(values)
    assert record["floor_active"] is False
    assert record["usable_label"] is True
    assert record["normalized_std"] == pytest.approx(1.0, abs=0.05)


def test_counterproof_a_constant_channel_is_not_reported_usable():
    """A channel with no spread at all must not read as a usable label."""
    record = normalized_proxy_scale(np.full(128, 4.2))
    assert record["raw_std"] < 1e-12  # float64 accumulation noise, not spread
    assert record["floor_active"] is True
    assert record["usable_label"] is False


def test_counterproof_audit_does_not_hide_a_floor_by_recentering():
    """Dividing by the floored std is what shrinks the label; recentring alone
    must not make the record look healthy."""
    values = np.linspace(-1e-8, 1e-8, 512)
    healthy = normalized_proxy_scale(values, eps=1e-12)
    floored = normalized_proxy_scale(values, eps=DEFAULT_EPS)
    assert healthy["usable_label"] is True
    assert floored["usable_label"] is False
    assert floored["normalized_std"] < healthy["normalized_std"] / 100


def test_report_compares_store_std_with_recomputed_std():
    generator = np.random.default_rng(3)
    raw = generator.normal(0.0, 2.0, size=(64, 3))
    names = ("a", "b", "c")
    # "a"/"b" carry the std the store would have written; "c" carries a floor.
    stored = np.array([raw[:, 0].std(), raw[:, 1].std(), 1e-6])
    report = proxy_scale_report(raw, stored, names)
    assert report["channels"]["a"]["stored_matches_recomputed"] is True
    assert report["channels"]["b"]["stored_matches_recomputed"] is True
    assert report["channels"]["c"]["stored_matches_recomputed"] is False
    assert {record["floor_active"] for record in report["channels"].values()} == {False}
    assert set(report["channels"]) == set(names)


def test_report_flags_a_store_whose_baked_std_disagrees():
    """A store normalized by a different floor must be visible, not assumed."""
    generator = np.random.default_rng(4)
    raw = generator.normal(0.0, 1e-8, size=(64, 2))
    report = proxy_scale_report(raw, np.array([1e-6, 1e-9]), ("floored", "rescaled"))
    assert report["channels"]["floored"]["stored_matches_recomputed"] is True
    assert report["channels"]["rescaled"]["stored_matches_recomputed"] is False


def test_report_rejects_mismatched_channel_counts():
    with pytest.raises(ValueError):
        proxy_scale_report(np.zeros((8, 3)), np.ones(2), ("a", "b", "c"))


def test_input_contract_separates_supervision_from_inputs():
    contract = measurement_input_contract()
    assert "coarse_history" in contract["forwarded_keys"]
    assert contract["process_targets_is_forwarded"] is False
    assert contract["atmos_target_is_forwarded"] is False
    assert "supervision targets only" in contract["conclusion"]


def _tiny_process_model():
    from training.r7_experiment import make_model
    return make_model("process", {
        "in_channels": 3, "out_channels": 3, "history_steps": 2,
        "architecture": "window", "dim": 16, "depth": 1, "heads": 2,
        "window_size": 2, "patch_size": 2, "dropout": 0.0,
        "anchored_processes": 3, "free_processes": 1,
        "use_forecast_feedback": True, "default_reasoning_steps": 2,
    })


def _tiny_batch():
    generator = torch.Generator().manual_seed(5)
    return {
        "coarse_history": torch.randn(2, 2, 3, 4, 4, generator=generator),
        "atmos_target": torch.randn(2, 3, 4, 4, generator=generator),
        "process_targets": torch.randn(2, 3, generator=generator),
        "latitude": torch.tensor([[30.0, 31.0, 32.0, 33.0],
                                  [30.0, 31.0, 32.0, 33.0]]),
    }


def test_gradient_conflict_zero_weight_leaves_process_gradient_undefined():
    model = _tiny_process_model().train()
    groups = parameter_groups(model)
    record = gradient_conflict_record(model, _tiny_batch(), process_weight=0.0,
                                      steps=2, groups=groups)
    assert record["process_loss"] == 0.0
    for label in ("shared", "process_readout"):
        entry = record["groups"][label]
        assert entry["process_grad_norm"] == 0.0
        assert entry["cosine_defined"] is False
        assert entry["cosine_forecast_process"] is None
    assert record["causal_evidence"] is False
    assert record["groups"]["shared"]["forecast_grad_norm"] > 0


def test_gradient_conflict_positive_weight_reaches_the_readout():
    model = _tiny_process_model().train()
    groups = parameter_groups(model)
    record = gradient_conflict_record(model, _tiny_batch(), process_weight=0.1,
                                      steps=2, groups=groups)
    assert record["process_loss"] > 0
    readout = record["groups"]["process_readout"]
    assert readout["process_grad_norm"] > 0
    assert readout["parameters"] == len(groups["process_readout"])


def test_forecast_loss_never_reaches_the_process_readout():
    """The readout is off the forecast path: its output is scored only by the
    auxiliary loss. This is why an auxiliary weight of zero leaves the head
    untrained rather than merely under-trained."""
    model = _tiny_process_model().train()
    groups = parameter_groups(model)
    record = gradient_conflict_record(model, _tiny_batch(), process_weight=0.1,
                                      steps=2, groups=groups)
    readout = record["groups"]["process_readout"]
    assert readout["forecast_grad_norm"] == 0.0
    assert readout["cosine_defined"] is False
    assert readout["cosine_forecast_process"] is None
    # The shared trunk and the queries are reached by both losses.
    assert record["groups"]["shared"]["forecast_grad_norm"] > 0
    assert record["groups"]["shared"]["process_grad_norm"] > 0
    assert record["groups"]["process_queries"]["process_grad_norm"] > 0


def test_gradient_conflict_leaves_the_model_gradients_clean():
    model = _tiny_process_model().train()
    groups = parameter_groups(model)
    gradient_conflict_record(model, _tiny_batch(), process_weight=0.01, steps=2,
                             groups=groups)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_gradient_conflict_rejects_unknown_parameter_group():
    model = _tiny_process_model().train()
    with pytest.raises(ValueError):
        gradient_conflict_record(model, _tiny_batch(), process_weight=0.1, steps=2,
                                 groups={"shared": ["nope.weight"]})


def test_gradient_conflict_requires_train_mode():
    model = _tiny_process_model().eval()
    with pytest.raises(ValueError):
        gradient_conflict_record(model, _tiny_batch(), process_weight=0.1, steps=2,
                                 groups=parameter_groups(model))


def test_parameter_groups_are_disjoint_and_complete():
    model = _tiny_process_model()
    groups = parameter_groups(model)
    all_names = [name for _, name in
                 ((n, n) for n in dict(model.named_parameters()))]
    covered = groups["shared"] + groups["process_readout"] + groups["process_queries"]
    assert sorted(covered) == sorted(all_names)
    assert len(covered) == len(set(covered))
    assert groups["process_readout"]


def test_recursion_trajectory_reports_every_step_and_keeps_eval_mode():
    model = _tiny_process_model().train()
    record = recursion_trajectory(model, _tiny_batch(), max_steps=3)
    assert record["max_steps"] == 3
    assert [entry["k"] for entry in record["per_step"]] == [1, 2, 3]
    assert record["future_labels_used_retrospectively"] is True
    assert record["forecast_modified"] is False
    assert model.training is True
    for entry in record["per_step"]:
        assert entry["algebra_residual_max_abs"] < 1e-9
        assert entry["update_norm"] >= 0
        assert 0.0 <= entry["worsening_fraction"] <= 1.0
    assert record["per_step"][1]["previous_cosine_defined_fraction"] == 1.0


def test_recursion_trajectory_step_one_has_no_previous_cosine():
    model = _tiny_process_model()
    record = recursion_trajectory(model, _tiny_batch(), max_steps=2)
    assert "previous_update_cosine" not in record["per_step"][0]
    assert "previous_update_cosine" in record["per_step"][1]


def test_summary_names_the_floor_as_the_cause():
    raw = np.random.default_rng(2).normal(0.0, 1e-8, size=(32, 1))
    report = proxy_scale_report(raw, np.array([1e-6]), ("moisture_advection",))
    verdict = summarise_proxy_effect(report, ("moisture_advection",))
    assert "unusable" in verdict["moisture_advection"]
    assert "floor" in verdict["moisture_advection"]
