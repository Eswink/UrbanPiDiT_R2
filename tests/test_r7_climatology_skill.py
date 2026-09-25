"""Tests for the explicit climatology skill baseline and the ACC identity (#62)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
ACC = ROOT / "training" / "r7_acc.py"
SKILL = ROOT / "training" / "r7_climatology_skill.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _modules():
    return _load("r7_acc_under_test", ACC), _load("r7_climatology_skill_under_test", SKILL)


def _fields(b=3, l=2, c=2, h=6, w=5, seed=0):
    generator = torch.Generator().manual_seed(seed)
    prediction = torch.randn(b, l, c, h, w, generator=generator)
    target = torch.randn(b, l, c, h, w, generator=generator)
    climatology = torch.randn(b, l, c, h, w, generator=generator)
    latitude = torch.linspace(-60, 60, h)
    return prediction, target, climatology, latitude


def test_skill_accumulator_matches_a_direct_computation():
    acc_module, skill_module = _modules()
    prediction, target, climatology, latitude = _fields()
    skill = skill_module.RolloutClimatologySkillAccumulator([6, 12], ["t2m", "z500"])
    skill.update(prediction, target, climatology, latitude)
    result = skill.compute()
    weight = torch.cos(torch.deg2rad(latitude.double()))
    weight = weight / weight.sum()

    def pooled_mse(field):
        error = field.double()[:, :, :, :, :] - target.double()
        return (error.square().mean(-1) * weight[None, None, None, :]).sum(-1).mean(0)

    mse_forecast = pooled_mse(prediction)
    mse_climatology = pooled_mse(climatology)
    assert torch.allclose(result["rmse_forecast"], mse_forecast.sqrt(), rtol=1e-12)
    assert torch.allclose(result["rmse_climatology"], mse_climatology.sqrt(), rtol=1e-12)
    assert torch.allclose(result["mse_skill"], 1.0 - mse_forecast / mse_climatology,
                          rtol=1e-12)


def test_negative_acc_implies_negative_skill_across_random_trials():
    """The #62 identity: ACC < 0 forces MSE skill < 0; ACC > 0 implies nothing.

    Note the shared-climatology property this exercises: with independent
    random fields the uncentered anomaly correlation is positively biased
    because both anomalies subtract the *same* climatology, so negative-ACC
    cases are constructed with an anti-correlated forecast anomaly.
    """
    acc_module, skill_module = _modules()
    seen_negative_acc = 0
    for seed in range(40):
        prediction, target, climatology, latitude = _fields(seed=seed)
        if seed % 2 == 0:
            # Anti-correlated forecast anomaly: drives ACC below zero.
            prediction = climatology - 0.6 * (target - climatology) \
                + 0.3 * (prediction - climatology)
        else:
            # Weakly correlated forecast with large independent noise.
            prediction = climatology + 0.2 * (target - climatology) \
                + 2.0 * (prediction - climatology)
        acc = acc_module.RolloutACCAccumulator([6, 12], ["t2m", "z500"])
        acc.update(prediction, target, climatology, latitude)
        skill = skill_module.RolloutClimatologySkillAccumulator([6, 12], ["t2m", "z500"])
        skill.update(prediction, target, climatology, latitude)
        report = skill_module.verify_acc_skill_consistency(
            acc.compute(), skill.compute()["mse_skill"])
        assert report["consistent"], report
        acc_value = acc.compute()
        seen_negative_acc += int((torch.isfinite(acc_value)
                                  & (acc_value < 0)).sum().item())
    assert seen_negative_acc > 0, "constructed trials should include negative-ACC cases"


def test_positive_acc_can_coexist_with_negative_skill():
    """The one-sidedness must be documented by a constructed counterexample."""
    acc_module, skill_module = _modules()
    generator = torch.Generator().manual_seed(7)
    b, l, c, h, w = 2, 1, 1, 4, 4
    target_anomaly = torch.randn(b, l, c, h, w, generator=generator)
    orthogonal = torch.randn(b, l, c, h, w, generator=generator)
    # Make the second component exactly orthogonal to the truth anomaly, then
    # take 0.2 * truth + a large orthogonal part: ACC stays positive while the
    # forecast error energy is dominated by the orthogonal part.
    flat_t = target_anomaly.reshape(b, l, c, -1)
    flat_u = orthogonal.reshape(b, l, c, -1)
    projection = (flat_u * flat_t).sum(-1, keepdim=True) / flat_t.square().sum(
        -1, keepdim=True)
    flat_u = flat_u - projection * flat_t
    orthogonal = flat_u.reshape_as(orthogonal)
    prediction_anomaly = 0.2 * target_anomaly + 3.0 * orthogonal
    climatology = torch.zeros(b, l, c, h, w)
    target = climatology + target_anomaly
    prediction = climatology + prediction_anomaly
    latitude = torch.linspace(-30, 30, h)
    acc = acc_module.RolloutACCAccumulator([6], ["t2m"])
    acc.update(prediction, target, climatology, latitude)
    skill = skill_module.RolloutClimatologySkillAccumulator([6], ["t2m"])
    skill.update(prediction, target, climatology, latitude)
    acc_value = float(acc.compute()[0, 0])
    skill_value = float(skill.compute()["mse_skill"][0, 0])
    assert acc_value > 0
    assert skill_value < 0, "this construction must produce a negative skill"
    report = skill_module.verify_acc_skill_consistency(acc.compute(),
                                                       skill.compute()["mse_skill"])
    assert report["consistent"]
    assert report["positive_acc_negative_skill"] == 1


def test_zero_energy_climatology_leaves_skill_undefined():
    acc_module, skill_module = _modules()
    prediction, target, _, latitude = _fields()
    climatology = target.clone()  # perfect climatology: zero error energy
    skill = skill_module.RolloutClimatologySkillAccumulator([6, 12], ["t2m", "z500"])
    skill.update(prediction, target, climatology, latitude)
    result = skill.compute()
    assert torch.isnan(result["mse_skill"]).all()
    assert torch.allclose(result["rmse_climatology"],
                          torch.zeros_like(result["rmse_climatology"]))


def test_rejects_mismatched_fields_and_empty_updates():
    acc_module, skill_module = _modules()
    prediction, target, climatology, latitude = _fields()
    skill = skill_module.RolloutClimatologySkillAccumulator([6, 12], ["t2m", "z500"])
    with pytest.raises(RuntimeError, match="no initialization samples"):
        skill.compute()
    with pytest.raises(ValueError, match="equal nonempty"):
        skill.update(prediction, target, climatology[:, :, :1], latitude)
    with pytest.raises(ValueError, match="invalid latitude"):
        skill.update(prediction, target, climatology, torch.full((4,), 120.0))
    nan_field = prediction.clone()
    nan_field[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        skill.update(nan_field, target, climatology, latitude)
