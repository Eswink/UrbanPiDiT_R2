"""CI-safe checks for the dense physical-step supervision driver.

S3 changes exactly one thing: the direct-supervision weight vector over the 12 6 h
physical steps goes from the registered sparse contour ``(1,.5,0,.5,0,0,0,.5,0,0,0,.5)``
to ``(0.25,)*12``. Both sum to 3.0; the candidate additionally supervises the seven
transition steps 18/30/36/42/54/60/66 h that the registered contour gives zero weight.

Every check runs against tiny synthetic specs, a synthetic 17-variable CSV and mocked
pinned artifacts. Nothing here reads ``outputs/`` or ``test.jsonl``, so the file runs on
a clean checkout. Each frozen claim has a counterproof: the wrong support, a changed
total, a tampered protocol, a flipped flag and an out-of-band bridge must all fail.
"""
from __future__ import annotations

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


driver = _load("study_r7_s3_dense_step_supervision")
wide = driver.wide
interior = driver.interior
VARIABLES = ("mslp", "q500", "q850", "t2m", "t500", "t850", "u10", "u250", "u500", "u850",
             "v10", "v250", "v500", "v850", "z250", "z500", "z850")


def _archived_spec(dim=driver.DIM, depth=driver.DEPTH, **extra):
    spec = dict(architecture="window", dim=dim, depth=depth, in_channels=17, out_channels=17,
                history_steps=2, heads=2, window_size=2, patch_size=2, dropout=0.,
                spacetime_inputs=True, default_reasoning_steps=4, **extra)
    return spec


class _Shared:
    ARCHIVED = ROOT / "outputs" / "r7_stub_archive"

    def __init__(self, spec):
        self._spec = spec

    def archived_process_spec(self):
        return (self._spec, None, None)


@pytest.fixture(autouse=True)
def _stub_shared(monkeypatch):
    monkeypatch.setattr(driver.recipe, "_shared", lambda: _Shared(_archived_spec()))


# --- the frozen weight vector ----------------------------------------------------------------

def test_the_two_vectors_are_the_frozen_contours():
    assert driver.REGISTERED_PHYSICAL_WEIGHTS == (1., .5, 0., .5, 0., 0., 0., .5, 0., 0., 0., .5)
    assert driver.DENSE_PHYSICAL_WEIGHTS == (0.25,) * 12
    assert driver.N_PHYSICAL_STEPS == 12
    assert sum(driver.REGISTERED_PHYSICAL_WEIGHTS) == 3.0
    assert sum(driver.DENSE_PHYSICAL_WEIGHTS) == 3.0


def test_physical_weight_block_names_the_seven_new_steps():
    block = driver.physical_weight_block()
    assert block["control_direct_steps"] == [0, 1, 3, 7, 11]
    assert block["candidate_direct_steps"] == list(range(12))
    assert block["newly_supervised_steps"] == [2, 4, 5, 6, 8, 9, 10]
    assert block["newly_supervised_lead_hours"] == [18, 30, 36, 42, 54, 60, 66]
    assert block["control_sum"] == block["candidate_sum"] == 3.0


def test_a_changed_control_support_is_refused(monkeypatch):
    monkeypatch.setattr(driver, "REGISTERED_PHYSICAL_WEIGHTS", (0.5,) * 6)
    with pytest.raises(RuntimeError):
        driver.physical_weight_block()


def test_a_changed_dense_support_with_the_same_sum_is_refused(monkeypatch):
    changed = (0.0, 0.5) + (0.25,) * 10
    assert len(changed) == 12 and abs(sum(changed) - 3.0) < 1e-12
    monkeypatch.setattr(driver, "DENSE_PHYSICAL_WEIGHTS", changed)
    with pytest.raises(RuntimeError):
        driver.physical_weight_block()


def test_a_changed_total_weight_is_refused(monkeypatch):
    monkeypatch.setattr(driver, "DENSE_PHYSICAL_WEIGHTS", (0.5,) * 12)
    with pytest.raises(RuntimeError):
        driver.physical_weight_block()


# --- the arm spec and the frozen constants ---------------------------------------------------

def test_arm_spec_is_the_archived_spec_untouched(monkeypatch):
    spec = _archived_spec()
    monkeypatch.setattr(driver.recipe, "_shared", lambda: _Shared(spec))
    assert driver.arm_spec() == spec
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


def test_protocol_constants_carry_only_the_weight_change():
    body = driver.protocol_constants()
    assert body["format"] == driver.FORMAT
    assert body["recipe"]["updates"] == 2400
    assert body["recipe"]["steps"] == 4
    assert body["evaluation"]["reasoning_steps"] == 4
    assert body["recipe"]["physical_weights"] == list(driver.DENSE_PHYSICAL_WEIGHTS)
    assert body["control_physical_weights"] == list(driver.REGISTERED_PHYSICAL_WEIGHTS)
    assert body["physical_step_supervision"] == driver.physical_weight_block()
    assert body["seeds"] == [41, 42, 43]
    assert body["budgets"] == {"planned_seconds": driver.PLANNED_SECONDS,
                               "hard_cap_seconds": driver.HARD_CAP_SECONDS,
                               "per_seed_seconds": driver.PER_SEED_SECONDS}
    assert body["scientific_claim"] is False and body["test_read"] is False
    assert "coverage" in body["difference"] and "K=4" in body["difference"]


def test_the_identity_pins_the_dense_arm_and_both_references(monkeypatch):
    from training.r7_experiment import canonical_digest
    monkeypatch.setattr(interior, "_identity", lambda: {"arms": {}, "train_windows": {"x": 1}})
    monkeypatch.setattr(driver, "previous_arm_pins", lambda: {"record_id": "dose"})
    monkeypatch.setattr(driver, "paired_control_pins", lambda: {"record_id": "anchor"})
    monkeypatch.setattr(driver.anchor_parent, "install_parent_pins",
                        lambda body: body["arms"]["parent"].update(pins={"checkpoints": {}}))
    monkeypatch.setattr("training.r7_arm_harness.sha256_file", lambda path: "0" * 64)
    body = driver._identity()
    candidate = body["arms"][driver.CANDIDATE]
    assert candidate["physical_weights"] == list(driver.DENSE_PHYSICAL_WEIGHTS)
    assert candidate["physical_steps"] == 12 and candidate["steps"] == 4
    assert body["arms"]["previous_dose_arm"] == {"record_id": "dose"}
    assert body["arms"]["paired_control_arm"] == {"record_id": "anchor"}
    assert body["control_physical_weights"] == list(driver.REGISTERED_PHYSICAL_WEIGHTS)
    assert candidate["model"]["spec_canonical_digest"] == canonical_digest(driver.arm_spec())


# --- the 17-variable report ------------------------------------------------------------------

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


# --- the predeclared exit --------------------------------------------------------------------

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
    assert flags["scientific_exit_met"] is True
    assert flags["e_bridge_consistent"] is False and flags["identity_unresolved"] is True
    assert flags["predeclared_exit_met"] is False


# --- the frozen protocol ---------------------------------------------------------------------

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
                   "physical_weights": list(driver.DENSE_PHYSICAL_WEIGHTS)},
        "evaluation": {"reasoning_steps": driver.REASONING_STEPS},
        "control_physical_weights": list(driver.REGISTERED_PHYSICAL_WEIGHTS),
        "physical_step_supervision": driver.physical_weight_block(),
        "arms": {driver.CANDIDATE: {
                     "physical_weights": list(driver.DENSE_PHYSICAL_WEIGHTS), "physical_steps": 12,
                     "updates": driver.UPDATES, "steps": driver.STEPS,
                     "model": {"kind": "process", "spec": spec,
                               "spec_canonical_digest": canonical_digest(spec)}},
                 "previous_dose_arm": {"record_id": driver.PREVIOUS_DOSE_ARM_RECORD,
                                       "updates": driver.UPDATES, "seeds": list(driver.SEEDS)},
                 "paired_control_arm": {"record_id": driver.PAIRED_CONTROL_RECORD,
                                        "updates": driver.UPDATES,
                                        "physical_weights": list(driver.REGISTERED_PHYSICAL_WEIGHTS),
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


def test_validate_protocol_accepts_the_frozen_dense_arm(tmp_path, monkeypatch):
    _patch_validate(monkeypatch)
    body = driver.validate_protocol(_write(tmp_path, _protocol_body()))
    assert body["recipe"]["physical_weights"] == list(driver.DENSE_PHYSICAL_WEIGHTS)


def _replace(body, path, value):
    node = body
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


@pytest.mark.parametrize("path,value", [
    (("recipe", "physical_weights"), [0.5] * 12),
    (("control_physical_weights", 0), 2.0),
    (("arms", driver.CANDIDATE, "physical_steps"), 11),
    (("arms", driver.CANDIDATE, "model", "spec", "depth"), 8),
    (("arms", "paired_control_arm", "record_id"), "s3-wide-interior-dose"),
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


def test_validate_protocol_rejects_a_changed_budget(tmp_path, monkeypatch):
    _patch_validate(monkeypatch)
    body = _protocol_body()
    body["per_seed_seconds"] += 1.0
    from training.r7_experiment import canonical_digest
    body.pop("protocol_sha256")
    body["protocol_sha256"] = canonical_digest(body)
    with pytest.raises(RuntimeError):
        driver.validate_protocol(_write(tmp_path, body))


def test_validate_protocol_rejects_a_tampered_digest(tmp_path, monkeypatch):
    _patch_validate(monkeypatch)
    body = _protocol_body()
    body["protocol_sha256"] = "0" * 64
    with pytest.raises(RuntimeError):
        driver.validate_protocol(_write(tmp_path, body))


# --- the round spec and the parent pins ------------------------------------------------------

def test_round_spec_matches_the_driver_and_splits_the_two_commits():
    text = (ROOT / "docs" / "goals" / "s3-dense-step-supervision.md").read_text(encoding="utf-8")
    marker = "<!-- round-spec: "
    payload = json.loads(text[text.index(marker) + len(marker):text.index(" -->", text.index(marker))])
    assert payload["arm"] == "s3-dense-step-supervision"
    assert payload["driver"] == "scripts/study_r7_s3_dense_step_supervision.py"
    assert payload["output_stem"] == "r7_s3_dense_step_supervision_20261018"
    assert payload["control_record"] == driver.PREVIOUS_DOSE_ARM_RECORD
    assert payload["seeds"] == list(driver.SEEDS)
    assert payload["test_path"] == ["tests/test_r7_s3_dense_step_supervision.py"]
    assert payload["code_archive"].endswith(".zip")
    assert "docs/R7_S3_DENSE_STEP_SUPERVISION_RESULT.md" in payload["register_paths"]
    assert "docs/R7_S3_DENSE_STEP_SUPERVISION_RESULT.md" not in payload["commit_paths"]
    assert "scripts/study_r7_s3_dense_step_supervision.py" in payload["commit_paths"]
    assert "docs/goals/main-model-climatology-campaign.md" in payload["register_paths"]


def test_parent_pins_are_the_h_d_attempt04_ones():
    assert driver.anchor_parent.MIGRATED_PARENT_RUN.name.endswith("attempt04")
    assert driver.PAIRED_CONTROL_RUN.name == "r7_s3_anchor_isolation_20261015_attempt01"
    assert sorted(driver.anchor_parent.MIGRATED_PARENT_SHA256) == [41, 42, 43]
    assert all(len(value) == 64 for value in driver.anchor_parent.MIGRATED_PARENT_SHA256.values())


# --- the training path accepts the new configuration on CPU ----------------------------------

def test_fine_tune_accepts_the_dense_twelve_step_vector_on_cpu(tmp_path):
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
                                   "physical_weights": list(driver.DENSE_PHYSICAL_WEIGHTS),
                                   "window_sha256": summary["window_sha256"],
                                   "excluded_sample_ids": summary["excluded_sample_ids"]}}
    _, report = fine_tune_long_rollout(paths["train"], tmp_path / "run", model=model, contract=contract,
        parent_weights=weights, physical_weights=driver.DENSE_PHYSICAL_WEIGHTS, steps=4, updates=1,
        seed=7, warmup=1, checkpoint_every=1)
    assert report["contract"]["autoregression"]["physical_weights"] == list(driver.DENSE_PHYSICAL_WEIGHTS)
    assert report["contract"]["autoregression"]["physical_steps"] == 12
    assert len(report["losses"]) == 1 and len(report["losses"][0]["per_step_losses"]) == 12
