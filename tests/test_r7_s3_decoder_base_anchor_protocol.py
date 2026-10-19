"""CI-safe protocol and training-path checks for the S3 decoder-base-anchor round.

Split from ``tests/test_r7_s3_decoder_base_anchor.py`` to stay inside R-051. These run
on synthetic fixtures only: nothing reads ``outputs/`` or ``test.jsonl``.
"""
from __future__ import annotations

import copy
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


driver = _load("study_r7_s3_decoder_base_anchor")
parent = driver.parent
report = driver.report
wide = driver.wide
interior = driver.interior
import training.r7_long_rollout_runner as runner
from training.r7_experiment import canonical_digest, load_checkpoint, make_model

VARIABLES = ("mslp", "q500", "q850", "t2m", "t500", "t850", "u10", "u250", "u500", "u850",
             "v10", "v250", "v500", "v850", "z250", "z500", "z850")


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


def _tiny_anchor():
    from scripts.r7_s3_anchor_isolation_parent import _tiny_anchor as build
    return build()


@pytest.fixture
def anchor_stub(monkeypatch):
    anchor_spec, _ = _tiny_anchor()
    spec = _archived_spec()
    monkeypatch.setattr(driver.recipe, "_shared", lambda: _Shared(spec))
    monkeypatch.setattr(driver.anchor_arm, "anchor_payload",
                        lambda: {"spec": anchor_spec, "pins": {}})
    return anchor_spec


@pytest.fixture(autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


# --------------------------------------------------------------------------- protocol validation


def _write_boundary(path, variables=VARIABLES, *, duplicate=None, nonfinite=None):
    import csv
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
    assert sorted(report.all_variable_rmse(path)) == sorted(VARIABLES)
    assert set(report.all_variable_rmse(path).values()) == {1.0}


def test_all_variable_rmse_refuses_duplicates_missing_variables_and_nonfinite(tmp_path):
    for kwargs in ({"duplicate": "t2m"}, {"variables": VARIABLES[:-1]}, {"nonfinite": "t2m"}):
        path = tmp_path / "boundary_rmse.csv"
        _write_boundary(path, **kwargs)
        with pytest.raises(RuntimeError):
            report.all_variable_rmse(path)


def test_dose_readings_report_all_17_variables(tmp_path, monkeypatch):
    leads, climatology = {}, {}
    for lead in wide.EVALUATION_LEADS:
        folder = tmp_path / f"lead{lead}"
        folder.mkdir()
        _write_boundary(folder / "boundary_rmse.csv")
        leads[str(lead)] = {"dir": str(folder), "boundary_rmse_csv_sha256": "0" * 64}
        climatology[str(lead)] = {"interior_32_t2m_rmse": 2.0}
    body = {"arms": {"previous_dose_arm": {"seeds_evaluations": {"41": {"leads": leads}}}},
            "wide_climatology_reference": {"leads": climatology}}
    monkeypatch.setattr("training.r7_arm_harness.sha256_file", lambda path: "0" * 64)
    arm = report.dose_readings(body, 41)
    assert set(arm["all_variable_interior_rmse"]) == {str(lead) for lead in wide.EVALUATION_LEADS}
    assert set(arm["all_variable_interior_rmse"]["6"]) == set(VARIABLES)
    assert arm["interior_t2m_rmse"]["6"] == 1.0


def _protocol_body(anchor_stub):
    spec = copy.deepcopy(driver.arm_spec())
    body = {
        "execution_files_sha256": {"x.py": "0" * 64},
        "train_manifest": str(wide.TRAIN_MANIFEST), "train_data_identity": "train-id",
        "val_manifest": str(wide.VAL_MANIFEST), "val_data_identity": "val-id",
        "model_code_sha256": "m" * 64, "training_code_sha256": "t" * 64,
        "source_sha256": "s" * 64, "train_windows": {"stub": 1}, "supervision": {"stub": 1},
        "format": driver.FORMAT,
        "recipe": {"updates": driver.UPDATES, "steps": driver.STEPS,
                   "physical_weights": list(driver.REGISTERED_PHYSICAL_WEIGHTS)},
        "evaluation": {"reasoning_steps": driver.REASONING_STEPS},
        "candidate_draft_supervision": "deep", "control_draft_supervision": "deep",
        "decoder_base": driver.decoder_base_block(),
        "draft_supervision_organization": driver.draft_organization_block(),
        "physical_weight_vector": driver.physical_weight_block(),
        "budget_profile_note": driver.BUDGET_PROFILE_NOTE,
        "code_version_comparison": {
            "candidate_training_code_sha256": "t" * 64,
            "control_training_code_sha256": "c" * 64,
            "equal": False},
        "arms": {driver.CANDIDATE: {
                     "physical_weights": list(driver.REGISTERED_PHYSICAL_WEIGHTS),
                     "physical_steps": 12, "draft_supervision": "deep",
                     "decoder_base_mode": "zero", "spatial_gradient_lambda": 0.0,
                     "updates": driver.UPDATES, "steps": driver.STEPS,
                     "model": {"kind": "process", "spec": spec,
                               "spec_canonical_digest": canonical_digest(spec),
                               "decoder_base_mode": "zero"}},
                 "previous_dose_arm": {"record_id": driver.PREVIOUS_DOSE_ARM_RECORD,
                                       "updates": driver.UPDATES, "seeds": list(driver.SEEDS)},
                 "paired_control_arm": {"record_id": driver.PAIRED_CONTROL_RECORD,
                                        "updates": driver.UPDATES, "data_identity": "train-id",
                                        "source_sha256": "s" * 64,
                                        "physical_weights": list(driver.REGISTERED_PHYSICAL_WEIGHTS),
                                        "draft_supervision": "deep",
                                        "decoder_base_mode": "persistence",
                                        "training_code_sha256": "c" * 64,
                                        "seeds": list(driver.SEEDS), "protocol_sha256": "p" * 64},
                 "parent": {"pins": {"checkpoints": {
                     str(seed): {"sha256": parent.MIGRATED_PARENT_SHA256[seed]}
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


def test_validate_protocol_accepts_the_frozen_zero_base_arm(tmp_path, monkeypatch, anchor_stub):
    _patch_validate(monkeypatch)
    body = driver.validate_protocol(_write(tmp_path, _protocol_body(anchor_stub)))
    assert body["decoder_base"]["candidate_mode"] == "zero"


def _replace(body, path, value):
    node = body
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


@pytest.mark.parametrize("path,value", [
    (("recipe", "physical_weights"), [0.3] * 12),
    (("recipe", "updates"), 2000),
    (("decoder_base", "candidate_mode"), "persistence"),
    (("arms", driver.CANDIDATE, "decoder_base_mode"), "persistence"),
    (("arms", driver.CANDIDATE, "spatial_gradient_lambda"), 0.5),
    (("arms", driver.CANDIDATE, "draft_supervision"), "final_only"),
    (("arms", driver.CANDIDATE, "physical_steps"), 11),
    (("arms", driver.CANDIDATE, "model", "spec", "decoder_base_mode"), "persistence"),
    (("arms", driver.CANDIDATE, "model", "spec", "depth"), 8),
    (("arms", "paired_control_arm", "record_id"), "s3-wide-interior-dose"),
    (("arms", "paired_control_arm", "decoder_base_mode"), "zero"),
    (("arms", "paired_control_arm", "protocol_sha256"), "q" * 64),
])
def test_validate_protocol_rejects_a_tampered_factor(tmp_path, monkeypatch, anchor_stub, path, value):
    _patch_validate(monkeypatch)
    body = _protocol_body(anchor_stub)
    _replace(body, path, value)
    body.pop("protocol_sha256")
    body["protocol_sha256"] = canonical_digest(body)
    with pytest.raises(RuntimeError):
        driver.validate_protocol(_write(tmp_path, body))


def test_validate_protocol_rejects_tampered_blocks_budget_and_digest(tmp_path, monkeypatch,
                                                                     anchor_stub):
    _patch_validate(monkeypatch)
    for mutate in ("decoder_block", "organization", "budget", "budget_note", "code",
                   "claim", "digest"):
        body = _protocol_body(anchor_stub)
        if mutate == "decoder_block":
            body["decoder_base"] = dict(body["decoder_base"], candidate_mode="persistence")
        elif mutate == "organization":
            body["draft_supervision_organization"] = dict(
                body["draft_supervision_organization"], nonfinal_mass=0.5)
        elif mutate == "budget":
            body["per_seed_seconds"] += 1.0
        elif mutate == "budget_note":
            body["budget_profile_note"] = "a different envelope"
        elif mutate == "code":
            body["code_version_comparison"] = dict(
                body["code_version_comparison"], control_training_code_sha256="d" * 64)
        elif mutate == "claim":
            body["scientific_claim"] = True
        else:
            body["protocol_sha256"] = "0" * 64
        if mutate != "digest":
            body.pop("protocol_sha256")
            body["protocol_sha256"] = canonical_digest(body)
        with pytest.raises(RuntimeError):
            driver.validate_protocol(_write(tmp_path, body))


# --------------------------------------------------------------------------- CI-safe training smoke


def _options(tmp_path, mode="zero", updates=1):
    from test_r7_m4_autoregressive_rollout import runner_options
    from data.r7_long_rollout_dataset import preflight_long_rollout_windows
    paths, options = runner_options(tmp_path, kind="process")
    options.pop("bf16")
    summary = preflight_long_rollout_windows(paths["train"], physical_steps=3)
    options["contract"]["autoregression"] = {
        "physical_steps": 3, "physical_weights": [1., .5, .5],
        "window_sha256": summary["window_sha256"],
        "excluded_sample_ids": summary["excluded_sample_ids"],
        "decoder_base_mode": mode}
    options["physical_weights"] = (1., .5, .5)
    spec = dict(options["contract"]["model"], decoder_base_mode=mode)
    options["contract"]["model"] = spec
    torch.manual_seed(23)
    options["model"] = make_model("process", spec)
    options.update(updates=updates, checkpoint_every=1)
    return paths, options


def test_cpu_smoke_accepts_the_zero_base_training_path(tmp_path):
    paths, options = _options(tmp_path)
    checkpoint, report = runner.fine_tune_long_rollout(paths["train"], tmp_path / "run", **options)
    assert report["contract"]["model"]["decoder_base_mode"] == "zero"
    assert report["contract"]["autoregression"]["decoder_base_mode"] == "zero"
    assert report["updates_this_run"] == 1 and report["objective"] == "deep_supervised_latitude_area_mse"
    assert not any(key.startswith("ema_") for key in report)
    assert Path(checkpoint).name == "update_0000001.pt"
    saved = load_checkpoint(checkpoint)
    assert saved["contract"]["model"]["decoder_base_mode"] == "zero"
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in report["losses"])


