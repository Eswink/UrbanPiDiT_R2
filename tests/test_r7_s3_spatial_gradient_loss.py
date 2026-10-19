"""CI-safe checks for the spatial-gradient-loss driver and its loss branch.
S3 adds the frozen ``L = L_mse + 0.1 * L_grad`` penalty on every draft of every
physical step; ``gradient_lambda=0.0`` must stay bit-identical to the registered
formula. Everything runs on synthetic tensors/specs/fixtures: nothing reads
``outputs/`` or ``test.jsonl``, and every frozen claim has a counterproof.
"""
from __future__ import annotations
import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import pytest
import torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

driver = _load("study_r7_s3_spatial_gradient_loss")
wide = driver.wide
interior = driver.interior
VARIABLES = ("mslp", "q500", "q850", "t2m", "t500", "t850", "u10", "u250", "u500", "u850",
             "v10", "v250", "v500", "v850", "z250", "z500", "z850")

import training.r7_recursive_losses as recursive_losses
from training.r7_losses import latitude_weighted_mse
from training.r7_recursive_losses import (
    deep_supervised_forecast_mse, require_gradient_lambda, spatial_gradient_mse,
)
def _synthetic(dtype=torch.float64):
    generator = torch.Generator().manual_seed(19)
    drafts = torch.randn(2, 5, 2, 4, 5, generator=generator, dtype=dtype)
    target = torch.randn(2, 2, 4, 5, generator=generator, dtype=dtype)
    latitude = torch.linspace(60., 36., 4, dtype=dtype)
    mask = torch.zeros(4, 5)
    mask[1:3, 1:4] = 1.
    return drafts, target, latitude, mask
def _registered_deep(drafts, target, latitude, mask):
    weights = torch.linspace(1., 2., drafts.shape[1], dtype=drafts.dtype)
    weights = weights / weights.sum()
    return sum(weight * latitude_weighted_mse(drafts[:, index], target, latitude, mask=mask)
               for index, weight in enumerate(weights))
def _registered_deep_with_gradient(drafts, target, latitude, mask, coefficient):
    weights = torch.linspace(1., 2., drafts.shape[1], dtype=drafts.dtype)
    weights = weights / weights.sum()
    return sum(weight * (latitude_weighted_mse(drafts[:, index], target, latitude, mask=mask)
                         + coefficient * spatial_gradient_mse(drafts[:, index], target, latitude,
                                                               mask=mask))
               for index, weight in enumerate(weights))
def test_zero_lambda_is_bit_identical_to_the_registered_deep_objective():
    drafts, target, latitude, mask = _synthetic()
    default = deep_supervised_forecast_mse(drafts, target, latitude, final_weight=2., mask=mask)
    explicit = deep_supervised_forecast_mse(drafts, target, latitude, final_weight=2., mask=mask,
                                            gradient_lambda=0.0)
    assert torch.equal(default, _registered_deep(drafts, target, latitude, mask))
    assert torch.equal(default, explicit)

def test_zero_lambda_never_evaluates_the_spatial_penalty(monkeypatch):
    drafts, target, latitude, mask = _synthetic()

    def forbidden(*args, **kwargs):
        raise AssertionError("the spatial-gradient branch ran at lambda=0")

    monkeypatch.setattr(recursive_losses, "spatial_gradient_mse", forbidden)
    zero = deep_supervised_forecast_mse(drafts, target, latitude, mask=mask, gradient_lambda=0.0)
    assert torch.equal(zero, _registered_deep(drafts, target, latitude, mask))
    with pytest.raises(AssertionError, match="lambda=0"):
        deep_supervised_forecast_mse(drafts, target, latitude, mask=mask, gradient_lambda=0.1)

def test_positive_lambda_adds_the_penalty_to_each_draft_before_the_k_weights():
    drafts, target, latitude, mask = _synthetic()
    actual = deep_supervised_forecast_mse(drafts, target, latitude, final_weight=2., mask=mask,
                                          gradient_lambda=0.1)
    expected = _registered_deep_with_gradient(drafts, target, latitude, mask, 0.1)
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, _registered_deep(drafts, target, latitude, mask))

def _brute_gradient(prediction, target, latitude, mask):
    """Literal pair loop; the denominator sums pair weights, not channels."""
    batch, channels, height, width = prediction.shape
    numerator, denominator = prediction.new_zeros(()), prediction.new_zeros(())
    for index in range(batch):
        cosine = torch.cos(torch.deg2rad(latitude[index]))
        for row in range(height - 1):
            for column in range(width):
                if mask[row, column] and mask[row + 1, column]:
                    weight = 0.5 * (cosine[row] + cosine[row + 1])
                    denominator = denominator + weight
                    for channel in range(channels):
                        pred_delta = prediction[index, channel, row + 1, column] - prediction[index, channel, row, column]
                        target_delta = target[index, channel, row + 1, column] - target[index, channel, row, column]
                        delta = pred_delta - target_delta
                        numerator = numerator + weight * delta.square()
        for row in range(height):
            for column in range(width - 1):
                if mask[row, column] and mask[row, column + 1]:
                    weight = cosine[row]
                    denominator = denominator + weight
                    for channel in range(channels):
                        pred_delta = prediction[index, channel, row, column + 1] - prediction[index, channel, row, column]
                        target_delta = target[index, channel, row, column + 1] - target[index, channel, row, column]
                        delta = pred_delta - target_delta
                        numerator = numerator + weight * delta.square()
    return numerator / (denominator * channels)

def test_spatial_gradient_matches_a_literal_pair_loop():
    generator = torch.Generator().manual_seed(23)
    prediction = torch.randn(2, 3, 4, 5, generator=generator, dtype=torch.float64)
    target = torch.randn(2, 3, 4, 5, generator=generator, dtype=torch.float64)
    latitude = torch.stack((torch.linspace(60., 36., 4), torch.linspace(50., 26., 4))).double()
    mask = torch.zeros(4, 5, dtype=torch.bool)
    mask[1:3, 1:4] = True
    actual = spatial_gradient_mse(prediction, target, latitude, mask=mask)
    torch.testing.assert_close(actual, _brute_gradient(prediction, target, latitude, mask),
                               rtol=1e-13, atol=1e-13)
    every = torch.ones(4, 5, dtype=torch.bool)
    torch.testing.assert_close(spatial_gradient_mse(prediction, target, latitude, mask=None),
                               _brute_gradient(prediction, target, latitude, every),
                               rtol=1e-13, atol=1e-13)

def test_spatial_gradient_pair_weights_use_the_pair_mean_and_the_row():
    prediction = torch.zeros(1, 1, 3, 3, dtype=torch.float64)
    target = torch.zeros(1, 1, 3, 3, dtype=torch.float64)
    target[0, 0, 1, 0] = 1.
    latitude = torch.tensor([0., 60., 80.], dtype=torch.float64)
    mask = torch.zeros(3, 3, dtype=torch.bool)
    mask[:, :2] = True
    cosine = torch.cos(torch.deg2rad(latitude))
    # Selected: four H pairs (two rows x two columns) and three W pairs (one per row).
    # The target's single 1 at (row 1, col 0) changes both H rows 0-1 and 1-2 at
    # column 0 and the W pair in row 1; column 1 differences stay zero.
    horizontal = 0.5 * (cosine[:-1] + cosine[1:])
    expected = (horizontal.sum() + cosine[1]) / (2 * horizontal.sum() + cosine.sum())
    actual = spatial_gradient_mse(prediction, target, latitude, mask=mask)
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)

def test_spatial_gradient_only_reads_selected_pairs():
    generator = torch.Generator().manual_seed(29)
    prediction = torch.randn(1, 1, 4, 5, generator=generator, dtype=torch.float64)
    target = torch.randn(1, 1, 4, 5, generator=generator, dtype=torch.float64)
    latitude = torch.linspace(60., 36., 4, dtype=torch.float64)
    mask = torch.zeros(4, 5, dtype=torch.bool)
    mask[1:3, 1:4] = True
    baseline = spatial_gradient_mse(prediction, target, latitude, mask=mask)
    outside = prediction.clone()
    outside[0, 0, 0, 0] += 5.
    outside[0, 0, 3, 4] -= 7.
    assert torch.equal(spatial_gradient_mse(outside, target, latitude, mask=mask), baseline)
    inside = prediction.clone()
    inside[0, 0, 1, 1] += 5.
    assert not torch.equal(spatial_gradient_mse(inside, target, latitude, mask=mask), baseline)

def test_spatial_gradient_is_a_channel_mean_and_zero_for_a_perfect_forecast():
    generator = torch.Generator().manual_seed(31)
    prediction = torch.randn(1, 1, 4, 5, generator=generator, dtype=torch.float64)
    target = torch.randn(1, 1, 4, 5, generator=generator, dtype=torch.float64)
    latitude = torch.linspace(60., 36., 4, dtype=torch.float64)
    one = spatial_gradient_mse(prediction, target, latitude)
    many = spatial_gradient_mse(prediction.repeat(1, 17, 1, 1), target.repeat(1, 17, 1, 1), latitude)
    assert torch.equal(one, many) and spatial_gradient_mse(prediction, prediction.clone(), latitude) == 0.

@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), True, "0.1", None])
def test_gradient_lambda_validation_rejects_invalid_coefficients(bad):
    with pytest.raises(ValueError, match="gradient_lambda"):
        require_gradient_lambda(bad)
    drafts, target, latitude, mask = _synthetic()
    with pytest.raises(ValueError, match="gradient_lambda"):
        deep_supervised_forecast_mse(drafts, target, latitude, mask=mask, gradient_lambda=bad)

def test_gradient_reaches_every_draft_under_deep_supervision():
    drafts, target, latitude, mask = _synthetic()
    drafts.requires_grad_()
    loss = deep_supervised_forecast_mse(drafts, target, latitude, mask=mask, gradient_lambda=0.1)
    gradient, = torch.autograd.grad(loss, drafts)
    assert all(gradient[:, index].abs().sum() > 0 for index in range(drafts.shape[1]))

def test_spatial_gradient_requires_a_matching_mask_and_finite_inputs():
    prediction, target = torch.zeros(1, 1, 3, 4), torch.zeros(1, 1, 3, 4)
    broken = target.clone(); broken[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="mask"):
        spatial_gradient_mse(prediction, target, mask=torch.ones(2, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match="no adjacent pair"):
        spatial_gradient_mse(prediction, target, mask=torch.zeros(3, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match="finite"):
        spatial_gradient_mse(prediction, broken)

def test_draft_loss_propagates_the_penalty():
    from training.r7_autoregressive_rollout import _draft_loss
    drafts, target, latitude, mask = _synthetic()
    forecast = drafts[:, -1].detach().clone().requires_grad_()
    actual = _draft_loss(forecast, drafts, target, latitude, mask, gradient_lambda=0.1)
    expected = deep_supervised_forecast_mse(drafts.float(), target.float(), latitude, final_weight=2.,
                                            mask=mask, gradient_lambda=0.1)
    assert torch.equal(actual, expected)

def test_training_long_rollout_applies_the_penalty_on_every_physical_step():
    from test_r7_m4_autoregressive_rollout import AnalyticForecaster, batch
    from training.r7_long_rollout import training_long_rollout
    sample = batch()
    ramp = torch.linspace(-1000., 1000., 12).reshape(1, 1, 3, 4)
    sample["atmos_target"], sample["future_target"] = ramp.clone(), ramp.clone()
    targets = [sample["atmos_target"]] + [sample["future_target"].clone() for _ in range(3)]
    sample["physical_targets"] = torch.stack(targets, dim=1)
    weights = (1., 0., 0., 0.)
    model = AnalyticForecaster()
    result = training_long_rollout(model, sample, 4, physical_weights=weights, gradient_lambda=0.1)
    expected = torch.stack([
        deep_supervised_forecast_mse(trace.draft_forecasts.float(), targets[index].float(),
                                     sample["latitude"], final_weight=2., gradient_lambda=0.1)
        for index, trace in enumerate(model.traces)])
    torch.testing.assert_close(result.per_step_losses, expected, rtol=0, atol=0)
    torch.testing.assert_close(result.loss, sum(weight * loss for weight, loss in
                                                zip(weights, expected)), rtol=0, atol=0)
    baseline = training_long_rollout(AnalyticForecaster(), sample, 4, physical_weights=weights)
    assert not torch.equal(result.loss, baseline.loss)
    assert model.kwargs == [(4, False)] * 4

def test_runner_semantics_are_unchanged_at_lambda_zero():
    from training.r7_long_rollout_runner import _draft_semantics, _gradient_semantics
    assert _gradient_semantics(0.0) == {}
    registered = {"draft_supervision": "deep", "internal_deep_supervision": True,
                  "objective": "deep_supervised_latitude_area_mse", "final_weight": 2.,
                  "k_weights": "linspace(1,2,K+1) normalized; initial plus every K draft"}
    assert _draft_semantics("deep") == registered
    graded = _draft_semantics("deep", 0.1)
    assert graded["objective"] == "deep_supervised_latitude_area_mse"
    assert graded["internal_deep_supervision"] is True
    assert graded["spatial_gradient_lambda"] == 0.1
    assert graded["spatial_gradient_penalty"] == "latitude_weighted_first_difference_mse"

def test_physical_axis_and_k_axis_stay_at_the_registered_contours():
    assert driver.REGISTERED_PHYSICAL_WEIGHTS == (1., .5, 0., .5, 0., 0., 0., .5, 0., 0., 0., .5)
    assert driver.N_PHYSICAL_STEPS == 12 and sum(driver.REGISTERED_PHYSICAL_WEIGHTS) == 3.0
    assert driver.physical_weight_block()["direct_steps"] == [0, 1, 3, 7, 11]
    assert driver.STEPS == driver.REASONING_STEPS == 4
    assert driver.REGISTERED_DEEP_WEIGHTS == pytest.approx(
        (2 / 15, 1 / 6, 1 / 5, 7 / 30, 4 / 15), abs=1e-12)
    block = driver.draft_organization_block()
    assert block["control_draft_supervision"] == block["candidate_draft_supervision"] == "deep"
    assert block["nonfinal_mass"] == pytest.approx(11. / 15., abs=1e-12)
    assert block["final_weight"] == pytest.approx(4. / 15., abs=1e-12)

def test_spatial_gradient_block_is_the_frozen_design():
    block = driver.spatial_gradient_block()
    assert block["candidate_lambda"] == 0.1 and block["control_lambda"] == 0.0
    assert (block["h_pairs"], block["w_pairs"]) == (4160, 4160)
    assert "interior_32" in block["pair_selection"] and "mean cos" in block["weights"]
    assert block["sweep"].startswith("none")

def test_changed_coefficients_or_pair_counts_are_refused(monkeypatch):
    for name, value in (("CANDIDATE_GRADIENT_LAMBDA", 0.2), ("CONTROL_GRADIENT_LAMBDA", 0.1),
                        ("H_PAIRS", 4159), ("W_PAIRS", 4200)):
        monkeypatch.setattr(driver, name, value)
        with pytest.raises(RuntimeError):
            driver.spatial_gradient_block()
        monkeypatch.undo()

def test_changed_physical_or_k_axis_vectors_are_refused(monkeypatch):
    monkeypatch.setattr(driver, "REGISTERED_PHYSICAL_WEIGHTS", (0.5,) * 12)
    with pytest.raises(RuntimeError):
        driver.physical_weight_block()
    monkeypatch.undo()
    monkeypatch.setattr(driver, "REGISTERED_DEEP_WEIGHTS", (0.2,) * 5)
    with pytest.raises(RuntimeError):
        driver.draft_organization_block()

def _archived_spec(dim=driver.DIM, depth=driver.DEPTH, **extra):
    return dict(architecture="window", dim=dim, depth=depth, in_channels=17, out_channels=17,
                history_steps=2, heads=2, window_size=2, patch_size=2, dropout=0.,
                spacetime_inputs=True, default_reasoning_steps=4, **extra)

class _Shared:
    # Only a path string converted to a relative name; nothing reads it.
    ARCHIVED = ROOT / "synthetic_archive"

    def __init__(self, spec):
        self._spec = spec

    def archived_process_spec(self):
        return (self._spec, None, None)

@pytest.fixture(autouse=True)
def _stub_shared(monkeypatch):
    monkeypatch.setattr(driver.recipe, "_shared", lambda: _Shared(_archived_spec()))

def test_arm_spec_is_the_archived_spec_untouched():
    assert driver.arm_spec() == _archived_spec()
    assert not any(key in driver.arm_spec() for key in (
        "climatology_anchor_spec", "anomaly_feedback", "anchor_output_prior",
        "process_init_conditioning", "typed_evidence_mode", "global_mixing_mode"))

def test_arm_spec_refuses_a_switched_or_wrong_shaped_spec(monkeypatch):
    for bad in (_archived_spec(anchor_output_prior=False, climatology_anchor_spec={"x": 1}),
                _archived_spec(global_mixing_mode="attention"), _archived_spec(dim=128),
                _archived_spec(depth=8)):
        monkeypatch.setattr(driver.recipe, "_shared", lambda bad=bad: _Shared(bad))
        with pytest.raises(RuntimeError):
            driver.arm_spec()

def test_protocol_constants_carry_only_the_spatial_gradient_change():
    body = driver.protocol_constants()
    assert body["format"] == driver.FORMAT
    assert body["recipe"]["updates"] == 2400
    assert body["recipe"]["steps"] == 4
    assert body["evaluation"]["reasoning_steps"] == 4
    assert body["recipe"]["physical_weights"] == list(driver.REGISTERED_PHYSICAL_WEIGHTS)
    assert body["candidate_gradient_lambda"] == 0.1 and body["control_gradient_lambda"] == 0.0
    assert body["spatial_gradient_penalty"] == driver.spatial_gradient_block()
    assert body["candidate_draft_supervision"] == body["control_draft_supervision"] == "deep"
    assert body["draft_supervision_organization"] == driver.draft_organization_block()
    assert body["physical_weight_vector"] == driver.physical_weight_block()
    assert body["seeds"] == [41, 42, 43]
    assert body["budgets"] == {"planned_seconds": driver.PLANNED_SECONDS,
                               "hard_cap_seconds": driver.HARD_CAP_SECONDS,
                               "per_seed_seconds": driver.PER_SEED_SECONDS}
    assert body["scientific_claim"] is False and body["test_read"] is False
    assert "0.1" in body["difference"] and "K=4" in body["difference"]

def test_the_identity_pins_the_gradient_arm_and_both_references(monkeypatch):
    from training.r7_experiment import canonical_digest
    monkeypatch.setattr(interior, "_identity", lambda: {"arms": {}, "train_windows": {"x": 1},
                                                       "train_data_identity": "train-id", "source_sha256": "s" * 64})
    monkeypatch.setattr(driver, "previous_arm_pins", lambda: {"record_id": "dose"})
    monkeypatch.setattr(driver, "paired_control_pins", lambda *args, **kwargs: {"record_id": "anchor"})
    monkeypatch.setattr(driver.anchor_parent, "install_parent_pins",
                        lambda body: body["arms"]["parent"].update(pins={"checkpoints": {}}))
    monkeypatch.setattr("training.r7_arm_harness.sha256_file", lambda path: "0" * 64)
    body = driver._identity()
    candidate = body["arms"][driver.CANDIDATE]
    assert candidate["physical_weights"] == list(driver.REGISTERED_PHYSICAL_WEIGHTS)
    assert candidate["physical_steps"] == 12 and candidate["steps"] == 4
    assert candidate["draft_supervision"] == "deep"
    assert candidate["spatial_gradient_lambda"] == 0.1
    assert body["arms"]["previous_dose_arm"] == {"record_id": "dose"}
    assert body["arms"]["paired_control_arm"] == {"record_id": "anchor"}
    assert body["control_gradient_lambda"] == 0.0
    assert candidate["model"]["spec_canonical_digest"] == canonical_digest(driver.arm_spec())
    assert candidate["model"]["anchor_keys_absent"] == ["climatology_anchor_spec", "anchor_output_prior", "anomaly_feedback"]

def _write_boundary(path, variables=VARIABLES, *, duplicate=None, nonfinite=None):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("region", "margin_cells", "n_grid_points", "full_area_fraction",
                         "lead_hours", "variable", "unit", "rmse", "n_initializations"))
        for variable in variables:
            rmse = float("nan") if variable == nonfinite else 1.0
            writer.writerow(("interior_32", 32, 4225, 1.0, 6, variable, "u", rmse, 428))
        if duplicate is not None:
            writer.writerow(("interior_32", 32, 4225, 1.0, 6, duplicate, "u", 2.0, 428))
        writer.writerow(("full", 0, 16641, 1.0, 6, "t2m", "K", 99.0, 428))

def test_all_variable_rmse_returns_exactly_the_17_interior_values(tmp_path):
    path = tmp_path / "boundary_rmse.csv"
    _write_boundary(path)
    values = driver._all_variable_rmse(path)
    assert sorted(values) == sorted(VARIABLES) and set(values.values()) == {1.0}

def test_all_variable_rmse_refuses_duplicates_missing_variables_and_nonfinite(tmp_path):
    for kwargs in ({"duplicate": "t2m"}, {"variables": VARIABLES[:-1]}, {"nonfinite": "t2m"}):
        path = tmp_path / "boundary_rmse.csv"
        _write_boundary(path, **kwargs)
        with pytest.raises(RuntimeError):
            driver._all_variable_rmse(path)

def _seed_block(skill72, delta72, skill48, delta_short, bridge):
    return {"candidate_skill": {"6": 0.5, "12": 0.4, "24": 0.3, "48": skill48, "72": skill72},
            "delta_candidate_minus_control": {"6": delta_short, "12": delta_short,
                                              "24": delta_short, "48": -0.1, "72": delta72},
            "control_minus_previous_dose": {"6": bridge, "12": bridge, "24": bridge,
                                            "48": bridge, "72": bridge}}

def _seed_map(**overrides):
    blocks = {}
    for index, seed in enumerate(driver.SEEDS):
        values = {"skill72": 0.01, "delta72": -0.01, "skill48": 0.02,
                  "delta_short": -0.001, "bridge": 0.0}
        values.update({key: value[index] if isinstance(value, (list, tuple)) else value
                       for key, value in overrides.items()})
        blocks[str(seed)] = _seed_block(**values)
    return blocks

def test_decision_flags_all_true_when_every_predeclared_condition_holds():
    flags = driver.decision_flags(_seed_map())
    assert flags["scientific_exit_met"] is True and flags["predeclared_exit_met"] is True and flags["identity_unresolved"] is False

@pytest.mark.parametrize("override", [
    {"skill72": (0.01, -0.02, 0.03)}, {"delta72": (-0.01, 0.02, -0.01)},
    {"skill48": (-0.01, 0.02, 0.01)}, {"delta_short": (0.001, 0.001, 0.001)}])
def test_each_scientific_condition_can_fail_alone(override):
    flags = driver.decision_flags(_seed_map(**override))
    assert flags["scientific_exit_met"] is False and flags["predeclared_exit_met"] is False

def test_short_lead_uses_the_seed_mean_not_each_seed():
    flags = driver.decision_flags(_seed_map(delta_short=(-0.002, 0.001, 0.0005)))
    assert flags["d_short_lead_seed_mean_not_worse"] is True

def test_out_of_band_bridge_is_identity_unresolved_not_a_science_failure():
    flags = driver.decision_flags(_seed_map(bridge=2e-5))
    assert flags["scientific_exit_met"] is True and flags["predeclared_exit_met"] is False
    assert flags["e_bridge_consistent"] is False and flags["identity_unresolved"] is True

def _protocol_body():
    from training.r7_experiment import canonical_digest
    spec = driver.arm_spec()
    body = {
        "execution_files_sha256": {"x.py": "0" * 64},
        "train_manifest": str(wide.TRAIN_MANIFEST), "train_data_identity": "train-id",
        "val_manifest": str(wide.VAL_MANIFEST), "val_data_identity": "val-id",
        "model_code_sha256": "m" * 64, "training_code_sha256": "t" * 64, "source_sha256": "s" * 64,
        "train_windows": {"stub": 1}, "supervision": {"stub": 1}, "format": driver.FORMAT,
        "recipe": {"updates": driver.UPDATES, "steps": driver.STEPS,
                   "physical_weights": list(driver.REGISTERED_PHYSICAL_WEIGHTS)},
        "evaluation": {"reasoning_steps": driver.REASONING_STEPS},
        "candidate_draft_supervision": "deep", "control_draft_supervision": "deep",
        "candidate_gradient_lambda": 0.1, "control_gradient_lambda": 0.0,
        "spatial_gradient_penalty": driver.spatial_gradient_block(),
        "draft_supervision_organization": driver.draft_organization_block(),
        "physical_weight_vector": driver.physical_weight_block(),
        "arms": {driver.CANDIDATE: {
                     "physical_weights": list(driver.REGISTERED_PHYSICAL_WEIGHTS), "physical_steps": 12,
                     "draft_supervision": "deep", "spatial_gradient_lambda": 0.1,
                     "updates": driver.UPDATES, "steps": driver.STEPS,
                     "model": {"kind": "process", "spec": spec,
                               "spec_canonical_digest": canonical_digest(spec)}},
                 "previous_dose_arm": {"record_id": driver.PREVIOUS_DOSE_ARM_RECORD,
                                       "updates": driver.UPDATES, "seeds": list(driver.SEEDS)},
                 "paired_control_arm": {"record_id": driver.PAIRED_CONTROL_RECORD,
                                        "updates": driver.UPDATES, "data_identity": "train-id",
                                        "source_sha256": "s" * 64,
                                        "physical_weights": list(driver.REGISTERED_PHYSICAL_WEIGHTS),
                                        "draft_supervision": "deep", "gradient_lambda": 0.0,
                                        "seeds": list(driver.SEEDS), "protocol_sha256": "p" * 64},
                 "parent": {"pins": {"checkpoints": {
                     str(seed): {"sha256": driver.anchor_parent.MIGRATED_PARENT_SHA256[seed]}
                     for seed in driver.SEEDS}}}},
        "planned_seconds": driver.PLANNED_SECONDS, "hard_cap_seconds": driver.HARD_CAP_SECONDS,
        "per_seed_seconds": driver.PER_SEED_SECONDS,
        "estimated_peak_bytes": driver.ESTIMATED_PEAK_BYTES,
        "headroom_margin_bytes": driver.HEADROOM_MARGIN_BYTES,
        "scientific_claim": False, "test_read": False,
        "test_manifest_never_read": str(wide.TEST_MANIFEST),
    }
    body["protocol_sha256"] = canonical_digest(body)
    return body

def _patch_validate(monkeypatch):
    import data.r7_long_rollout_dataset as dataset
    import training.r7_arm_harness as harness
    import training.r7_experiment as experiment
    import training.r7_long_rollout_runner as runner
    monkeypatch.setattr(experiment, "dataset_identity",
                        lambda manifest: ("train-id", "x") if str(manifest) == str(wide.TRAIN_MANIFEST)
                        else ("val-id", "x"))
    monkeypatch.setattr(experiment, "model_code_digest", lambda: "m" * 64)
    monkeypatch.setattr(runner, "training_code_digest", lambda: "t" * 64)
    monkeypatch.setattr(dataset, "preflight_long_rollout_windows", lambda *a, **k: {"stub": 1})
    monkeypatch.setattr(wide, "source_sha256", lambda: "s" * 64)
    monkeypatch.setattr(interior, "supervision_block", lambda: {"stub": 1})
    monkeypatch.setattr(driver.devices, "assignment_is_valid", lambda *a, **k: True)
    monkeypatch.setattr(driver, "_index_record", lambda record_id: {"protocol_sha256": "p" * 64})
    monkeypatch.setattr(driver, "execution_files", lambda: ["x.py"])
    monkeypatch.setattr(harness, "sha256_file", lambda path: "0" * 64)

def _write(tmp_path, body):
    (tmp_path / "protocol.json").write_text(json.dumps(body), encoding="utf-8")
    return tmp_path

def test_validate_protocol_accepts_the_frozen_gradient_arm(tmp_path, monkeypatch):
    _patch_validate(monkeypatch)
    body = driver.validate_protocol(_write(tmp_path, _protocol_body()))
    assert body["candidate_gradient_lambda"] == 0.1

def _replace(body, path, value):
    node = body
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value

@pytest.mark.parametrize("path,value", [
    (("recipe", "physical_weights"), [0.3] * 12),
    (("candidate_gradient_lambda",), 0.2),
    (("control_gradient_lambda",), 0.1),
    (("candidate_draft_supervision",), "final_only"),
    (("arms", driver.CANDIDATE, "spatial_gradient_lambda"), 0.0),
    (("arms", driver.CANDIDATE, "draft_supervision"), "final_only"),
    (("arms", driver.CANDIDATE, "physical_steps"), 11),
    (("arms", driver.CANDIDATE, "model", "spec", "depth"), 8),
    (("arms", "paired_control_arm", "record_id"), "s3-wide-interior-dose"),
    (("arms", "paired_control_arm", "gradient_lambda"), 0.1),
    (("arms", "paired_control_arm", "protocol_sha256"), "q" * 64),
])
def test_validate_protocol_rejects_a_tampered_factor(tmp_path, monkeypatch, path, value):
    _patch_validate(monkeypatch)
    body = _protocol_body()
    _replace(body, path, value)
    from training.r7_experiment import canonical_digest
    body.pop("protocol_sha256")
    body["protocol_sha256"] = canonical_digest(body)
    with pytest.raises(RuntimeError):
        driver.validate_protocol(_write(tmp_path, body))

def test_validate_protocol_rejects_tampered_blocks_and_budget(tmp_path, monkeypatch):
    _patch_validate(monkeypatch)
    for mutate in ("gradient_block", "organization", "budget", "digest"):
        body = _protocol_body()
        if mutate == "gradient_block":
            body["spatial_gradient_penalty"] = dict(body["spatial_gradient_penalty"], h_pairs=4159)
        elif mutate == "organization":
            body["draft_supervision_organization"] = dict(
                body["draft_supervision_organization"], nonfinal_mass=0.5)
        elif mutate == "budget":
            body["per_seed_seconds"] += 1.0
        else:
            body["protocol_sha256"] = "0" * 64
        if mutate != "digest":
            from training.r7_experiment import canonical_digest
            body.pop("protocol_sha256")
            body["protocol_sha256"] = canonical_digest(body)
        with pytest.raises(RuntimeError):
            driver.validate_protocol(_write(tmp_path, body))

def test_round_spec_matches_the_driver_and_splits_the_two_commits():
    text = (ROOT / "docs" / "goals" / "s3-spatial-gradient-loss.md").read_text(encoding="utf-8")
    marker = "<!-- round-spec: "
    payload = json.loads(text[text.index(marker) + len(marker):text.index(" -->", text.index(marker))])
    assert payload["arm"] == "s3-spatial-gradient-loss"
    assert payload["driver"] == "scripts/study_r7_s3_spatial_gradient_loss.py"
    assert payload["output_stem"] == "r7_s3_spatial_gradient_loss_20261019"
    assert payload["control_record"] == driver.PAIRED_CONTROL_RECORD
    assert payload["seeds"] == list(driver.SEEDS)
    assert payload["test_path"] == ["tests/test_r7_s3_spatial_gradient_loss.py"]
    assert payload["code_archive"].endswith(".zip")
    assert "docs/R7_S3_SPATIAL_GRADIENT_LOSS_RESULT.md" in payload["register_paths"]
    assert "docs/R7_S3_SPATIAL_GRADIENT_LOSS_RESULT.md" not in payload["commit_paths"]
    for name in ("scripts/study_r7_s3_spatial_gradient_loss.py",
                 "tests/test_r7_s3_spatial_gradient_loss.py",
                 "training/r7_recursive_losses.py", "training/r7_autoregressive_rollout.py",
                 "training/r7_long_rollout.py", "training/r7_long_rollout_runner.py",
                 "docs/goals/s3-spatial-gradient-loss.md"):
        assert name in payload["commit_paths"]
    assert "docs/goals/main-model-climatology-campaign.md" in payload["register_paths"]
    assert payload["commit_message"] and payload["register_message"]

def test_parent_pins_are_the_h_d_attempt04_ones():
    assert driver.anchor_parent.MIGRATED_PARENT_RUN.name.endswith("attempt04")
    assert driver.PAIRED_CONTROL_RUN.name == "r7_s3_anchor_isolation_20261015_attempt01"
    assert sorted(driver.anchor_parent.MIGRATED_PARENT_SHA256) == [41, 42, 43]
    assert all(len(value) == 64 for value in driver.anchor_parent.MIGRATED_PARENT_SHA256.values())

def test_fine_tune_accepts_the_gradient_penalty_on_cpu_and_refuses_a_mismatch(tmp_path):
    import pandas as pd
    from test_r7_m4_autoregressive_rollout import fixture_manifest, tiny_spec
    from data.r7_long_rollout_dataset import preflight_long_rollout_windows
    from training.r7_experiment import canonical_digest, dataset_identity, make_model
    from training.r7_long_rollout_runner import fine_tune_long_rollout
    torch.set_num_threads(1)
    paths, _ = fixture_manifest(tmp_path / "fixture",
                                times=pd.date_range("2016-01-01", periods=24, freq="6h"))
    summary = preflight_long_rollout_windows(paths["train"], physical_steps=12)
    assert summary["usable_windows"] > 0
    spec = tiny_spec("process")
    torch.manual_seed(23)
    model = make_model("process", spec)
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    contract = {"kind": "process", "model": spec,
                "data_identity": dataset_identity(paths["train"])[0],
                "source_sha256": canonical_digest("synthetic-source"),
                "protocol_sha256": canonical_digest("synthetic-protocol"),
                "initialization": {"format": "explicit-seeded-fixture", "seed": 23},
                "autoregression": {"physical_steps": 12,
                                   "physical_weights": list(driver.REGISTERED_PHYSICAL_WEIGHTS),
                                   "draft_supervision": "deep",
                                   "spatial_gradient_lambda": 0.1,
                                   "window_sha256": summary["window_sha256"],
                                   "excluded_sample_ids": summary["excluded_sample_ids"]}}
    mask = torch.ones(3, 4, dtype=torch.bool)
    _, report = fine_tune_long_rollout(paths["train"], tmp_path / "run", model=model, contract=contract,
        parent_weights=weights, physical_weights=driver.REGISTERED_PHYSICAL_WEIGHTS, steps=4, updates=1,
        seed=7, warmup=1, checkpoint_every=1, supervision_mask=mask, gradient_lambda=0.1)
    assert report["objective"] == "deep_supervised_latitude_area_mse"
    assert report["internal_deep_supervision"] is True
    assert report["draft_supervision"] == "deep"
    assert report["spatial_gradient_lambda"] == 0.1
    assert report["contract"]["autoregression"]["spatial_gradient_lambda"] == 0.1
    assert report["contract"]["autoregression"]["physical_weights"] == list(driver.REGISTERED_PHYSICAL_WEIGHTS)
    assert len(report["losses"]) == 1 and len(report["losses"][0]["per_step_losses"]) == 12
    mismatched = copy.deepcopy(contract)
    mismatched["autoregression"]["spatial_gradient_lambda"] = 0.0
    with pytest.raises(ValueError, match="spatial gradient"):
        fine_tune_long_rollout(paths["train"], tmp_path / "refused", model=model, contract=mismatched,
            parent_weights=weights, physical_weights=driver.REGISTERED_PHYSICAL_WEIGHTS, steps=4,
            updates=1, seed=7, warmup=1, checkpoint_every=1, supervision_mask=mask,
            gradient_lambda=0.1)
    assert not (tmp_path / "refused").exists()
