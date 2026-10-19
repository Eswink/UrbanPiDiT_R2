"""CI-safe checks for the EMA weight-averaging driver and the EMA training branch.

The frozen rule is a plain ``decay=0.999`` parameter average taken over every
optimizer step, initialized from the parent, with no bias correction and no
warm-up. These tests run on synthetic tensors and fixtures only: nothing reads
``outputs/`` or ``test.jsonl``. Each frozen claim has a counterproof, including
"the scored checkpoint really is the EMA checkpoint" and "the registered
``ema_decay=None`` path is unchanged".
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


driver = _load("study_r7_s3_ema_weight_averaging")
wide = driver.wide
interior = driver.interior
import training.r7_ema as ema
import training.r7_long_rollout_runner as runner
from training.r7_experiment import load_checkpoint

VARIABLES = ("mslp", "q500", "q850", "t2m", "t500", "t850", "u10", "u250", "u500", "u850",
             "v10", "v250", "v500", "v850", "z250", "z500", "z850")


class _Tiny(torch.nn.Module):
    def __init__(self, frozen=False):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
        self.b = torch.nn.Parameter(torch.tensor([-1.], dtype=torch.float64))
        if frozen:
            self.c = torch.nn.Parameter(torch.tensor([5.], dtype=torch.float64), requires_grad=False)

    def fill(self, a, b):
        with torch.no_grad():
            self.a.fill_(a)
            self.b.fill_(b)


def test_plain_ema_matches_the_manual_recurrence_without_bias_correction():
    model = _Tiny()
    average = ema.ParameterEMA(model, 0.999)
    model.fill(10., 5.)
    average.update()
    assert average.updates == 1
    # No warm-up and no bias correction: the first step is 0.999*parent + 0.001*w1.
    torch.testing.assert_close(average.shadow["a"], torch.tensor([0.999 * 2. + 0.001 * 10.], dtype=torch.float64))
    torch.testing.assert_close(average.shadow["b"], torch.tensor([0.999 * -1. + 0.001 * 5.], dtype=torch.float64))
    model.fill(-3., 7.)
    average.update()
    torch.testing.assert_close(average.shadow["a"], torch.tensor([0.999 * (0.999 * 2. + 0.001 * 10.) + 0.001 * -3.], dtype=torch.float64))
    torch.testing.assert_close(average.shadow["b"], torch.tensor([0.999 * (0.999 * -1. + 0.001 * 5.) + 0.001 * 7.], dtype=torch.float64))


def test_decay_boundaries_are_parent_and_latest_weights():
    model = _Tiny()
    frozen = ema.ParameterEMA(model, 1.0)
    model.fill(10., 5.)
    frozen.update()
    torch.testing.assert_close(frozen.shadow["a"], torch.tensor([2.], dtype=torch.float64))
    following = ema.ParameterEMA(model, 0.0)
    following.update()
    assert torch.equal(following.shadow["a"], model.a.detach())


def test_ema_covers_every_trainable_parameter_only():
    model = _Tiny(frozen=True)
    average = ema.ParameterEMA(model, 0.999)
    assert set(average._parameters) == {"a", "b"}
    model.fill(10., 5.)
    model.c.fill_(9.)
    average.update()
    state = average.state_dict(model)
    assert set(state) == set(model.state_dict())
    torch.testing.assert_close(state["a"], torch.tensor([0.999 * 2. + 0.001 * 10.], dtype=torch.float64))
    assert torch.equal(state["c"], torch.tensor([9.], dtype=torch.float64))


@pytest.mark.parametrize("bad", [True, -0.1, 1.1, float("nan"), float("inf"), "0.999", {}])
def test_invalid_ema_decay_is_refused(bad):
    with pytest.raises(ValueError, match="ema_decay"):
        ema.require_ema_decay(bad)
    with pytest.raises(ValueError, match="ema_decay"):
        ema.ParameterEMA(_Tiny(), bad)


def _options(tmp_path, updates=1):
    from test_r7_m4_autoregressive_rollout import runner_options
    from data.r7_long_rollout_dataset import preflight_long_rollout_windows
    paths, options = runner_options(tmp_path, kind="process")
    options.pop("bf16")
    summary = preflight_long_rollout_windows(paths["train"], physical_steps=3)
    options["contract"]["autoregression"] = {
        "physical_steps": 3, "physical_weights": [1., .5, .5],
        "window_sha256": summary["window_sha256"],
        "excluded_sample_ids": summary["excluded_sample_ids"]}
    options["physical_weights"] = (1., .5, .5)
    options.update(updates=updates, checkpoint_every=1)
    return paths, options


@pytest.fixture(autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_none_ema_path_keeps_the_registered_report_shape(tmp_path, monkeypatch):
    paths, options = _options(tmp_path)
    checkpoint, report = runner.fine_tune_long_rollout(paths["train"], tmp_path / "run", **options)
    assert not any(key.startswith("ema_") for key in report)
    assert "selected_checkpoint_kind" not in report and "final_optimizer_checkpoint" not in report
    assert report["selected_checkpoint"] == str(checkpoint)
    assert Path(checkpoint).name == "update_0000001.pt"
    assert "ema_decay" not in report["contract"]["autoregression"]


def test_cpu_smoke_accepts_ema_and_publishes_the_ema_checkpoint(tmp_path):
    paths, options = _options(tmp_path, updates=2)
    options["contract"]["autoregression"]["ema_decay"] = 0.999
    final, report = runner.fine_tune_long_rollout(paths["train"], tmp_path / "run", ema_decay=0.999, **options)
    assert report["ema_decay"] == 0.999 and report["ema_bias_correction"] is False
    assert report["ema_warmup_updates"] == 0 and report["ema_updates_applied"] == 2
    assert report["selected_checkpoint_kind"] == "ema"
    assert report["selected_checkpoint"] == report["ema_checkpoint"]
    assert Path(report["ema_checkpoint"]).name == "ema_update_0000002.pt"
    assert Path(report["final_optimizer_checkpoint"]).name == "update_0000002.pt"
    saved = load_checkpoint(report["ema_checkpoint"], expected=report["signature"])
    assert saved["checkpoint_kind"] == "ema" and saved["ema_updates"] == 2
    assert any(not torch.equal(saved["model"][key], load_checkpoint(final)["model"][key])
               for key in saved["model"] if saved["model"][key].is_floating_point())


def test_ema_updates_after_every_optimizer_step(tmp_path, monkeypatch):
    paths, options = _options(tmp_path, updates=3)
    options["contract"]["autoregression"]["ema_decay"] = 0.999
    calls = []

    class Spy(ema.ParameterEMA):
        def update(self):
            calls.append(self.updates)
            return super().update()

    monkeypatch.setattr(runner, "ParameterEMA", Spy)
    runner.fine_tune_long_rollout(paths["train"], tmp_path / "run", ema_decay=0.999, **options)
    assert calls == [0, 1, 2]


@pytest.mark.parametrize("declared,actual", [(0.5, 0.999), (None, 0.999)])
def test_declared_ema_mismatch_is_refused(tmp_path, declared, actual):
    paths, options = _options(tmp_path)
    if declared is not None:
        options["contract"]["autoregression"]["ema_decay"] = declared
    with pytest.raises(ValueError, match="EMA"):
        runner.fine_tune_long_rollout(paths["train"], tmp_path / "refused", ema_decay=actual, **options)
    assert not (tmp_path / "refused" / "training_report.json").exists()


def test_frozen_constants_are_the_registered_recipe():
    assert driver.UPDATES == 2400 and driver.STEPS == driver.REASONING_STEPS == 4
    assert (driver.DIM, driver.DEPTH) == (192, 4) and driver.SEEDS == (41, 42, 43)
    assert driver.REGISTERED_PHYSICAL_WEIGHTS == (1., .5, 0., .5, 0., 0., 0., .5, 0., 0., 0., .5)
    assert driver.N_PHYSICAL_STEPS == 12 and sum(driver.REGISTERED_PHYSICAL_WEIGHTS) == 3.0
    assert driver.physical_weight_block()["direct_steps"] == [0, 1, 3, 7, 11]
    assert driver.REGISTERED_DEEP_WEIGHTS == pytest.approx(
        (2 / 15, 1 / 6, 1 / 5, 7 / 30, 4 / 15), abs=1e-12)
    assert driver.draft_organization_block()["nonfinal_mass"] == pytest.approx(11. / 15., abs=1e-12)
    assert driver.ema_block()["decay"] == 0.999
    assert driver.ema_block()["bias_correction"] is False and driver.ema_block()["warmup_updates"] == 0
    assert (driver.BRIDGE_MIN, driver.BRIDGE_MAX) == (-4.8e-6, 1.8e-5)
    assert (driver.PLANNED_SECONDS, driver.HARD_CAP_SECONDS) == (43200., 86400.)


def test_changed_frozen_constants_are_refused(monkeypatch):
    for name, value in (("EMA_DECAY", 0.99), ("EMA_BIAS_CORRECTION", True),
                        ("EMA_WARMUP_UPDATES", 10), ("EMA_INITIALIZATION", "zeros")):
        monkeypatch.setattr(driver, name, value)
        with pytest.raises(RuntimeError):
            driver.ema_block()
        monkeypatch.undo()
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


def test_protocol_constants_carry_only_the_ema_change():
    body = driver.protocol_constants()
    assert body["format"] == driver.FORMAT
    assert body["recipe"]["updates"] == 2400 and body["recipe"]["steps"] == 4
    assert body["evaluation"]["reasoning_steps"] == 4
    assert body["recipe"]["physical_weights"] == list(driver.REGISTERED_PHYSICAL_WEIGHTS)
    assert body["ema_weight_averaging"] == driver.ema_block()
    assert body["candidate_draft_supervision"] == body["control_draft_supervision"] == "deep"
    assert body["draft_supervision_organization"] == driver.draft_organization_block()
    assert body["physical_weight_vector"] == driver.physical_weight_block()
    assert body["seeds"] == [41, 42, 43]
    assert body["budgets"] == {"planned_seconds": driver.PLANNED_SECONDS,
                               "hard_cap_seconds": driver.HARD_CAP_SECONDS,
                               "per_seed_seconds": driver.PER_SEED_SECONDS}
    assert body["scientific_claim"] is False and body["test_read"] is False
    assert "0.999" in body["difference"] and "K=4" in body["difference"]


def test_the_identity_pins_the_ema_arm_and_both_references(monkeypatch):
    from training.r7_experiment import canonical_digest
    monkeypatch.setattr(interior, "_identity", lambda: {"arms": {}, "train_windows": {"x": 1},
                                                       "train_data_identity": "train-id", "source_sha256": "s" * 64})
    monkeypatch.setattr(driver, "previous_arm_pins", lambda: {"record_id": "dose"})
    monkeypatch.setattr(driver, "paired_control_pins", lambda *args, **kwargs: {"record_id": "anchor"})
    monkeypatch.setattr(driver.anchor_parent, "install_parent_pins", lambda body: body["arms"]["parent"].update(pins={}))
    monkeypatch.setattr("training.r7_arm_harness.sha256_file", lambda path: "0" * 64)
    body = driver._identity()
    candidate = body["arms"][driver.CANDIDATE]
    assert candidate["ema_decay"] == 0.999 and candidate["ema_bias_correction"] is False
    assert candidate["ema_warmup_updates"] == 0 and candidate["scored_checkpoint_kind"] == "ema"
    assert candidate["spatial_gradient_lambda"] == 0.0 and candidate["draft_supervision"] == "deep"
    assert body["arms"]["previous_dose_arm"] == {"record_id": "dose"}
    assert body["arms"]["paired_control_arm"] == {"record_id": "anchor"}
    assert candidate["model"]["spec_canonical_digest"] == canonical_digest(driver.arm_spec())
    assert candidate["model"]["anchor_keys_absent"] == ["climatology_anchor_spec", "anchor_output_prior", "anomaly_feedback"]


def test_candidate_receipt_must_be_the_ema_checkpoint(tmp_path):
    receipt = {"seed": 41, "physical_weights": list(driver.REGISTERED_PHYSICAL_WEIGHTS),
               "draft_supervision": "deep", "checkpoint": str(tmp_path / "update_0002400.pt"),
               "selected_checkpoint_kind": "final_optimizer_weights", "evaluations": {}}
    (tmp_path / "seed41_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="EMA"):
        driver._candidate_readings(tmp_path, {"wide_climatology_reference": {"leads": {}}}, 41)


def test_train_arm_contract_carries_ema_and_no_gradient_penalty():
    body = {"train_data_identity": "train-id", "source_sha256": "s" * 64,
            "protocol_sha256": "p" * 64, "training_code_sha256": "t" * 64,
            "train_windows": {"window_sha256": "w" * 64, "excluded_sample_ids": []},
            "supervision": {"runner_block": {"stub": 1}}}
    contract = driver._arm_contract(body, driver.arm_spec(), "parent.pt", "f" * 64)
    autoregression = contract["autoregression"]
    assert autoregression["ema_decay"] == 0.999 and autoregression["ema_bias_correction"] is False
    assert autoregression["ema_warmup_updates"] == 0
    assert "spatial_gradient_lambda" not in autoregression


def test_train_arm_scores_the_ema_checkpoint_not_the_final_one(tmp_path, monkeypatch):
    ema_path = tmp_path / "seed41/training/candidate/ema_update_0002400.pt"
    ema_path.parent.mkdir(parents=True)
    ema_path.write_bytes(b"synthetic-ema-checkpoint")
    final_path = ema_path.parent / "update_0002400.pt"
    final_path.write_bytes(b"synthetic-final-checkpoint")
    report = {"updates_this_run": 2400, "data_identity": "train-id", "supervision": {"stub": 1},
              "objective": "deep_supervised_latitude_area_mse", "internal_deep_supervision": True,
              "ema_decay": 0.999, "ema_initialization": "parent_checkpoint",
              "ema_bias_correction": False, "ema_warmup_updates": 0, "ema_updates_applied": 2400,
              "selected_checkpoint_kind": "ema", "selected_checkpoint": str(ema_path),
              "ema_checkpoint": str(ema_path), "final_optimizer_checkpoint": str(final_path),
              "elapsed_seconds": 0., "contract": {"steps": 4, "autoregression": {
                  "physical_weights": list(driver.REGISTERED_PHYSICAL_WEIGHTS),
                  "draft_supervision": "deep"}}}
    scored = []

    def fake_training(*args, **kwargs):
        assert kwargs["ema_decay"] == 0.999
        return final_path, report

    def fake_evaluate(*args, **kwargs):
        scored.append(kwargs["checkpoint"])
        return {"n_evaluated": 1, "n_available_windows": 1, "elapsed_seconds": 0.,
                "checkpoint_sha256": "0" * 64}

    monkeypatch.setattr(runner, "fine_tune_long_rollout", fake_training)
    import training.r7_evaluate as evaluate
    monkeypatch.setattr(evaluate, "evaluate_local", fake_evaluate)
    monkeypatch.setattr(wide, "_check_provenance", lambda *args, **kwargs: None)
    monkeypatch.setattr("training.r7_arm_harness.sha256_file", lambda path: "0" * 64)
    monkeypatch.setattr(driver, "_parent_model",
                        lambda body, seed: (None, {}, tmp_path / "parent.pt", "f" * 64))
    body = {"train_data_identity": "train-id", "source_sha256": "s" * 64,
            "protocol_sha256": "p" * 64, "training_code_sha256": "t" * 64,
            "train_windows": {"window_sha256": "w" * 64, "excluded_sample_ids": []},
            "supervision": {"runner_block": {"stub": 1}},
            "arms": {driver.CANDIDATE: {"model": {"spec": driver.arm_spec()}}}}
    receipt = driver._train_arm(tmp_path, body, 41, None, None)
    assert scored == [str(ema_path)] * len(wide.EVALUATION_LEADS)
    assert receipt["checkpoint"] == str(ema_path) and receipt["ema_checkpoint"] == str(ema_path)
    assert receipt["final_optimizer_checkpoint"] == str(final_path)

    def wrong_evaluate(*args, **kwargs):
        return {"n_evaluated": 1, "n_available_windows": 1, "elapsed_seconds": 0.,
                "checkpoint_sha256": "1" * 64}

    monkeypatch.setattr(evaluate, "evaluate_local", wrong_evaluate)
    with pytest.raises(RuntimeError, match="did not score the frozen EMA checkpoint"):
        driver._train_arm(tmp_path, body, 41, None, None)


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
    assert sorted(driver._all_variable_rmse(path)) == sorted(VARIABLES)
    assert set(driver._all_variable_rmse(path).values()) == {1.0}


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
    assert flags["scientific_exit_met"] is True and flags["predeclared_exit_met"] is True
    assert flags["identity_unresolved"] is False


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
        "ema_weight_averaging": driver.ema_block(),
        "draft_supervision_organization": driver.draft_organization_block(),
        "physical_weight_vector": driver.physical_weight_block(),
        "arms": {driver.CANDIDATE: {
                     "physical_weights": list(driver.REGISTERED_PHYSICAL_WEIGHTS), "physical_steps": 12,
                     "draft_supervision": "deep", "ema_decay": 0.999,
                     "ema_initialization": "parent_checkpoint", "ema_bias_correction": False,
                     "ema_warmup_updates": 0, "scored_checkpoint_kind": "ema",
                     "updates": driver.UPDATES, "steps": driver.STEPS,
                     "model": {"kind": "process", "spec": spec,
                               "spec_canonical_digest": canonical_digest(spec)}},
                 "previous_dose_arm": {"record_id": driver.PREVIOUS_DOSE_ARM_RECORD,
                                       "updates": driver.UPDATES, "seeds": list(driver.SEEDS)},
                 "paired_control_arm": {"record_id": driver.PAIRED_CONTROL_RECORD,
                                        "updates": driver.UPDATES, "data_identity": "train-id",
                                        "source_sha256": "s" * 64,
                                        "physical_weights": list(driver.REGISTERED_PHYSICAL_WEIGHTS),
                                        "draft_supervision": "deep", "ema_decay": None,
                                        "scored_checkpoint_kind": "final_optimizer_weights",
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
    import training.r7_long_rollout_runner as rollout
    monkeypatch.setattr(experiment, "dataset_identity",
                        lambda manifest: ("train-id", "x") if str(manifest) == str(wide.TRAIN_MANIFEST)
                        else ("val-id", "x"))
    monkeypatch.setattr(experiment, "model_code_digest", lambda: "m" * 64)
    monkeypatch.setattr(rollout, "training_code_digest", lambda: "t" * 64)
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


def test_validate_protocol_accepts_the_frozen_ema_arm(tmp_path, monkeypatch):
    _patch_validate(monkeypatch)
    body = driver.validate_protocol(_write(tmp_path, _protocol_body()))
    assert body["ema_weight_averaging"]["decay"] == 0.999


def _replace(body, path, value):
    node = body
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


@pytest.mark.parametrize("path,value", [
    (("recipe", "physical_weights"), [0.3] * 12),
    (("recipe", "updates"), 2000),
    (("ema_weight_averaging", "decay"), 0.99),
    (("arms", driver.CANDIDATE, "ema_decay"), 0.99),
    (("arms", driver.CANDIDATE, "ema_bias_correction"), True),
    (("arms", driver.CANDIDATE, "scored_checkpoint_kind"), "final_optimizer_weights"),
    (("arms", driver.CANDIDATE, "draft_supervision"), "final_only"),
    (("arms", driver.CANDIDATE, "physical_steps"), 11),
    (("arms", driver.CANDIDATE, "model", "spec", "depth"), 8),
    (("arms", "paired_control_arm", "record_id"), "s3-wide-interior-dose"),
    (("arms", "paired_control_arm", "ema_decay"), 0.999),
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
    for mutate in ("ema_block", "organization", "budget", "digest"):
        body = _protocol_body()
        if mutate == "ema_block":
            body["ema_weight_averaging"] = dict(body["ema_weight_averaging"], warmup_updates=10)
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
    text = (ROOT / "docs" / "goals" / "s3-ema-weight-averaging.md").read_text(encoding="utf-8")
    marker = "<!-- round-spec: "
    payload = json.loads(text[text.index(marker) + len(marker):text.index(" -->", text.index(marker))])
    assert payload["arm"] == "s3-ema-weight-averaging"
    assert payload["driver"] == "scripts/study_r7_s3_ema_weight_averaging.py"
    assert payload["output_stem"] == "r7_s3_ema_weight_averaging_20261019"
    assert payload["control_record"] == driver.PAIRED_CONTROL_RECORD
    assert payload["seeds"] == list(driver.SEEDS)
    assert payload["test_path"] == ["tests/test_r7_s3_ema_weight_averaging.py"]
    assert payload["code_archive"].endswith(".zip")
    assert "docs/R7_S3_EMA_WEIGHT_AVERAGING_RESULT.md" in payload["register_paths"]
    assert "docs/R7_S3_EMA_WEIGHT_AVERAGING_RESULT.md" not in payload["commit_paths"]
    for name in ("scripts/study_r7_s3_ema_weight_averaging.py",
                 "scripts/r7_s3_ema_weight_averaging_text.py",
                 "tests/test_r7_s3_ema_weight_averaging.py",
                 "training/r7_ema.py", "training/r7_long_rollout_runner.py",
                 "docs/goals/s3-ema-weight-averaging.md"):
        assert name in payload["commit_paths"]
    assert "docs/goals/main-model-climatology-campaign.md" in payload["register_paths"]
    assert payload["commit_message"] and payload["register_message"]


def test_control_and_reference_constants_are_the_registered_ones():
    assert driver.PAIRED_CONTROL_RECORD == "s3-anchor-isolation-result"
    assert driver.PAIRED_CONTROL_RUN.name == "r7_s3_anchor_isolation_20261015_attempt01"
    assert driver.PREVIOUS_DOSE_ARM_RECORD == "s3-wide-interior-dose"
    assert driver.anchor_parent.MIGRATED_PARENT_RUN.name.endswith("attempt04")
    assert sorted(driver.anchor_parent.MIGRATED_PARENT_SHA256) == [41, 42, 43]
