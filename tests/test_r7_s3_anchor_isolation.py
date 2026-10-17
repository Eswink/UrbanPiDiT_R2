"""CI-safe checks for the H-D anchor-isolation driver and its parent module.

H-D isolates one mechanism of the train-only climatology-anchor package: the draft feedback the
K-step reasoning recurrence sees, absolute versus ``draft - C_valid``. The output prior and the
proposal prior are coupled to the anchor's mere presence by the model's own validation rule, so this
round holds both off with ``anchor_output_prior=False`` and varies exactly one boolean.

The fidelity-critical parts are that the control arm is bit-identical to the unanchored registered
model outside the two anchor buffers, that it stays bit-identical in its forward when the anchor is
unread, that the candidate's forward actually changes, that zeroing the table collapses the candidate
onto the control bit for bit, that the expansion adds exactly the two anchor keys and installs the
published bytes, and that the frozen protocol pins both arms and the re-migrated parent.

Every check runs against tiny synthetic specs and a synthetic anchor artifact, so this file is
checked on a checkout with no ``outputs/``. Each check has a counterproof.
"""
from __future__ import annotations

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
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = _load("study_r7_s3_anchor_isolation")
anchor_parent = driver.anchor_parent
wide = driver.wide
interior = driver.interior

CHANNELS, HEIGHT, WIDTH = 17, 4, 5


def _synthetic_spec():
    """A tiny synthetic anchor spec on a 4x5 grid; never weather statistics."""
    keys = [[month, hour] for month in (1, 4, 7, 10) for hour in (0, 6, 12, 18)]
    table = (torch.randn(len(keys), CHANNELS, HEIGHT, WIDTH) * 0.1).contiguous()
    spec = {"format": "r7-train-climatology-anchor-v1",
            "kind": "train-only-month-hour-grid-mean-v1",
            "channels": [f"c{index}" for index in range(CHANNELS)],
            "units": ["u"] * CHANNELS, "source_sha256": "1" * 64,
            "train_data_identity": "2" * 64, "climatology_mean_identity_sha256": "3" * 64,
            "normalization_mean": [0.0] * CHANNELS, "normalization_std": [1.0] * CHANNELS,
            "latitude": [float(50 - 2 * index) for index in range(HEIGHT)],
            "longitude": [float(90 + 2 * index) for index in range(WIDTH)],
            "bucket_keys": keys, "bucket_counts": [2] * len(keys),
            "training_years": [2017, 2018], "n_selected_steps": 2 * len(keys),
            "selection": "declared_train_years", "table_sha256": anchor_parent.table_digest(table),
            "table_shape": list(table.shape)}
    return spec, table


def _payload(spec=None, table=None):
    if spec is None or table is None:
        generated_spec, generated_table = _synthetic_spec()
        spec = generated_spec if spec is None else spec
        table = generated_table if table is None else table
    return {"spec": spec, "table": table,
            "pins": {"run": "synthetic", "table_sha256": anchor_parent.table_digest(table),
                     "spec_sha256": "4" * 64, "table_file_sha256": "5" * 64,
                     "n_buckets": len(spec["bucket_keys"]),
                     "n_selected_steps": spec["n_selected_steps"],
                     "bucket_counts": list(spec["bucket_counts"]),
                     "source_sha256": spec["source_sha256"],
                     "train_data_identity": spec["train_data_identity"],
                     "build_receipt_sha256": "6" * 64}}


def _tiny_spec(dim=8, *, anchor_spec=None, feedback=None, prior=False):
    spec = dict(architecture="window", in_channels=CHANNELS, out_channels=CHANNELS,
                history_steps=2, dim=dim, depth=4, heads=2, window_size=2, patch_size=2,
                dropout=0., spacetime_inputs=True, default_reasoning_steps=4,
                detach_between_steps=False, anchored_processes=4, free_processes=1)
    if anchor_spec is not None:
        spec["climatology_anchor_spec"] = anchor_spec
        spec["anchor_output_prior"] = prior
        spec["anomaly_feedback"] = bool(feedback)
    return spec


def _batch(spec, size=1):
    channels = len(spec["channels"])
    height, width = len(spec["latitude"]), len(spec["longitude"])
    generator = torch.Generator().manual_seed(11)
    return {"coarse_history": torch.randn(size, 2, channels, height, width, generator=generator),
            "atmos_target": torch.randn(size, channels, height, width, generator=generator),
            "future_target": torch.randn(size, channels, height, width, generator=generator),
            "latitude": torch.tensor(spec["latitude"]),
            "longitude": torch.tensor(spec["longitude"]),
            "lead_time_hours": torch.full((size,), 6.), "init_utc_hour": torch.full((size,), 18.),
            "init_day_of_year": torch.full((size,), 366.),
            "init_calendar_year": torch.full((size,), 2016.),
            "history_offsets_hours": torch.tensor([[-6., 0.]]).expand(size, -1).clone(),
            "init_year": torch.full((size,), 1234.), "process_targets": torch.full((size, 1), 9999.),
            "untrusted_metadata": "not a model input"}


@pytest.fixture(autouse=True)
def _stub_outputs(monkeypatch):
    """``arm_spec`` and the parent module read registered artifacts from ``outputs/``."""
    payload = _payload()

    class _Shared:
        ARCHIVED = ROOT / "outputs" / "r7_stub_archive"

        def archived_process_spec(self):
            return (_tiny_spec(dim=driver.DIM), None, None)

    monkeypatch.setattr(driver, "anchor_payload", lambda: payload)
    monkeypatch.setattr(driver.recipe, "_shared", lambda: _Shared())
    return payload


def _model(spec, *, anchor=None, seed=7):
    from training.r7_experiment import make_model
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = make_model("process", spec)
    if anchor is not None:
        model.backbone.climatology_anchor.install(anchor)
    return model.eval()


# --- the switch -------------------------------------------------------------------------------

def test_the_default_preserves_the_package_semantics():
    """An anchor-installed model without the new switch must decode C_valid, as before."""
    spec, table = _synthetic_spec()
    default_spec = _tiny_spec(anchor_spec=spec, feedback=False)
    default_spec.pop("anchor_output_prior")
    plain = _model(default_spec, anchor=table)
    explicit = _model(_tiny_spec(anchor_spec=spec, feedback=False, prior=True), anchor=table)
    sample = _batch(spec)
    with torch.no_grad():
        assert torch.equal(plain(sample, reasoning_steps=4).forecast,
                           explicit(sample, reasoning_steps=4).forecast)


def test_the_switch_can_only_be_suppressed_with_an_anchor():
    from training.r7_experiment import make_model
    with pytest.raises(ValueError):
        make_model("process", dict(_tiny_spec(), anchor_output_prior=False))


def test_the_switch_must_be_boolean():
    from training.r7_experiment import make_model
    with pytest.raises(ValueError):
        make_model("process", dict(_tiny_spec(), anchor_output_prior="no"))


# --- the identity that makes the two arms one factor ------------------------------------------

def test_the_control_adds_exactly_the_anchor_buffers():
    spec, table = _synthetic_spec()
    off = _model(_tiny_spec()).state_dict()
    control = _model(_tiny_spec(anchor_spec=spec, feedback=False), anchor=table).state_dict()
    assert sorted(set(control) - set(off)) == sorted(anchor_parent.ANCHOR_KEYS)
    assert not set(off) - set(control)
    shared = [key for key in off if key in control]
    assert shared and all(torch.equal(off[key], control[key]) for key in shared)


def test_the_control_forward_equals_the_unanchored_forward():
    """With both consumers off the anchor is read by nothing, so the control IS the registered model."""
    spec, table = _synthetic_spec()
    sample = _batch(spec)
    off = _model(_tiny_spec())
    control = _model(_tiny_spec(anchor_spec=spec, feedback=False), anchor=table)
    with torch.no_grad():
        assert torch.equal(off(sample, reasoning_steps=4).forecast,
                           control(sample, reasoning_steps=4).forecast)


def test_the_candidate_forward_differs():
    spec, table = _synthetic_spec()
    sample = _batch(spec)
    control = _model(_tiny_spec(anchor_spec=spec, feedback=False), anchor=table)
    candidate = _model(_tiny_spec(anchor_spec=spec, feedback=True), anchor=table)
    with torch.no_grad():
        difference = (control(sample, reasoning_steps=4).forecast
                      - candidate(sample, reasoning_steps=4).forecast).abs().max()
    assert float(difference) > 0.0


def test_a_zeroed_table_collapses_the_candidate_onto_the_control():
    """draft - 0 = draft, so the positive control is exact."""
    spec, table = _synthetic_spec()
    zero = torch.zeros_like(table)
    zero_spec = dict(spec, table_sha256=anchor_parent.table_digest(zero))
    sample = _batch(spec)
    control = _model(_tiny_spec(anchor_spec=zero_spec, feedback=False), anchor=zero)
    candidate = _model(_tiny_spec(anchor_spec=zero_spec, feedback=True), anchor=zero)
    with torch.no_grad():
        assert torch.equal(control(sample, reasoning_steps=4).forecast,
                           candidate(sample, reasoning_steps=4).forecast)


def test_a_nonzero_table_does_not_collapse():
    """Counterproof: the collapse above is caused by the zeroed table, not by the switch alone."""
    spec, table = _synthetic_spec()
    sample = _batch(spec)
    control = _model(_tiny_spec(anchor_spec=spec, feedback=False), anchor=table)
    candidate = _model(_tiny_spec(anchor_spec=spec, feedback=True), anchor=table)
    with torch.no_grad():
        assert not torch.equal(control(sample, reasoning_steps=4).forecast,
                               candidate(sample, reasoning_steps=4).forecast)


def test_the_output_prior_is_a_real_change():
    spec, table = _synthetic_spec()
    sample = _batch(spec)
    off = _model(_tiny_spec())
    prior = _model(_tiny_spec(anchor_spec=spec, feedback=False, prior=True), anchor=table)
    with torch.no_grad():
        assert not torch.equal(off(sample, reasoning_steps=4).forecast,
                               prior(sample, reasoning_steps=4).forecast)


# --- the parent module ------------------------------------------------------------------------

def test_anchor_payload_verifies_a_published_artifact(tmp_path, monkeypatch):
    spec, table = _synthetic_spec()
    (tmp_path / "anchor_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    torch.save(table, tmp_path / "anchor_table.pt")
    (tmp_path / "build_receipt.json").write_text(json.dumps(
        {"spec_sha256": __import__("hashlib").sha256(
            json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()},
        ), encoding="utf-8")
    (tmp_path / "BUILD_COMPLETE.json").write_text(json.dumps(
        {"schema_version": 1, "build_complete": True,
         "table_sha256": spec["table_sha256"],
         "spec_sha256": __import__("hashlib").sha256(
             json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}),
        encoding="utf-8")
    monkeypatch.setattr(anchor_parent, "ANCHOR_TABLE_RUN", tmp_path)
    payload = anchor_parent.anchor_payload()
    assert payload["spec"]["table_shape"] == list(table.shape)
    assert payload["pins"]["n_buckets"] == len(spec["bucket_keys"])


def test_anchor_payload_refuses_an_incomplete_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(anchor_parent, "ANCHOR_TABLE_RUN", tmp_path)
    (tmp_path / "BUILD_COMPLETE.json").write_text(json.dumps({"build_complete": False}),
                                                  encoding="utf-8")
    with pytest.raises(RuntimeError):
        anchor_parent.anchor_payload()


def test_anchor_payload_refuses_drifted_bytes(tmp_path, monkeypatch):
    spec, table = _synthetic_spec()
    (tmp_path / "anchor_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    torch.save(table + 1.0, tmp_path / "anchor_table.pt")
    (tmp_path / "build_receipt.json").write_text(json.dumps({"spec_sha256": "0" * 64}),
                                                 encoding="utf-8")
    (tmp_path / "BUILD_COMPLETE.json").write_text(json.dumps(
        {"schema_version": 1, "build_complete": True, "table_sha256": spec["table_sha256"],
         "spec_sha256": "0" * 64}), encoding="utf-8")
    monkeypatch.setattr(anchor_parent, "ANCHOR_TABLE_RUN", tmp_path)
    with pytest.raises(RuntimeError):
        anchor_parent.anchor_payload()


def test_expand_adds_exactly_the_anchor_keys(_stub_outputs):
    payload = _stub_outputs
    spec = _tiny_spec(dim=8, anchor_spec=payload["spec"], feedback=True)
    parent = _model(_tiny_spec(dim=8)).state_dict()
    model, state, receipt = anchor_parent.expand_parent_state(parent, spec, payload, seed=41)
    assert sorted(receipt["new_keys"]) == sorted(anchor_parent.ANCHOR_KEYS)
    assert receipt["parent_keys"] == len(parent)
    assert receipt["anchor_table_sha256"] == payload["pins"]["table_sha256"]
    assert receipt["anomaly_feedback"] is True
    for key, value in parent.items():
        assert torch.equal(state[key], value)
    model.load_state_dict(state, strict=True)
    assert torch.equal(model.backbone.climatology_anchor.table, payload["table"])
    assert bool(model.backbone.climatology_anchor.ready)


def test_expand_refuses_a_state_that_is_not_the_registered_parent(_stub_outputs):
    payload = _stub_outputs
    spec = _tiny_spec(dim=8, anchor_spec=payload["spec"], feedback=False)
    wrong = _model(_tiny_spec(dim=8)).state_dict()
    del wrong["process_queries"]
    with pytest.raises(RuntimeError):
        anchor_parent.expand_parent_state(wrong, spec, payload, seed=41)


def test_both_arms_share_one_anchor_table(_stub_outputs):
    payload = _stub_outputs
    parent = _model(_tiny_spec(dim=8)).state_dict()
    _, _, control = anchor_parent.expand_parent_state(
        parent, _tiny_spec(dim=8, anchor_spec=payload["spec"], feedback=False), payload, seed=41)
    _, _, candidate = anchor_parent.expand_parent_state(
        parent, _tiny_spec(dim=8, anchor_spec=payload["spec"], feedback=True), payload, seed=41)
    assert control["anchor_init_sha256"] == candidate["anchor_init_sha256"]
    assert control["anchor_table_sha256"] == candidate["anchor_table_sha256"]


def test_positive_control_reports_the_expected_facts():
    control = anchor_parent.positive_control(anchor_parent._tiny_spec())
    assert control["state_identity"]["bit_identical_outside_anchor"] is True
    assert control["state_identity"]["new_keys_match_anchor_keys"] is True
    assert control["unanchored_equals_off"] is True
    assert control["unanchored_max_abs"] == 0.0
    assert control["feedback_is_read"] is True
    assert control["zeroed_table_is_identity"] is True
    assert control["zeroed_table_delta_max_abs"] == 0.0
    assert control["output_prior_delta_max_abs"] > 0.0


# --- the driver's frozen fields ---------------------------------------------------------------

def test_arm_spec_binds_the_anchor_and_the_one_boolean():
    control = driver.arm_spec(False)
    candidate = driver.arm_spec(True)
    assert control["anomaly_feedback"] is False and candidate["anomaly_feedback"] is True
    for spec in (control, candidate):
        assert spec["anchor_output_prior"] is False
        assert spec["climatology_anchor_spec"] == driver.anchor_payload()["spec"]
        assert spec["dim"] == driver.DIM and spec["depth"] == driver.DEPTH
    assert {k: v for k, v in control.items() if k != "anomaly_feedback"} == \
           {k: v for k, v in candidate.items() if k != "anomaly_feedback"}


def test_arm_spec_refuses_a_spec_that_already_carries_a_switch(monkeypatch):
    monkeypatch.setattr(driver.recipe, "_shared",
                        lambda: type("S", (), {"archived_process_spec": lambda self: (
                            _tiny_spec(driver.DIM, anchor_spec=_synthetic_spec()[0]), None, None)})())
    with pytest.raises(RuntimeError):
        driver.arm_spec(False)


def test_arm_name_round_trips_and_refuses_an_unknown_value():
    assert driver.arm_name(False) == driver.CONTROL
    assert driver.arm_name(True) == driver.CANDIDATE
    with pytest.raises(ValueError):
        driver.arm_name("maybe")


def test_anchor_block_declares_the_mechanism_and_its_controls():
    control = driver.anchor_block(False)
    candidate = driver.anchor_block(True)
    assert control["anomaly_feedback"] is False and candidate["anomaly_feedback"] is True
    assert control["feedback"] == "draft"
    assert candidate["feedback"] == "draft - C_valid"
    assert control["new_keys"] == list(anchor_parent.ANCHOR_KEYS)
    assert control["new_params"] == 0 and candidate["new_params"] == 0
    assert candidate["anchor_output_prior"] is False
    assert candidate["identity_control"] == control["identity_control"]
    assert candidate["anchor_table"] == control["anchor_table"]


def test_protocol_constants_carry_the_factor_and_keep_dim_depth_k4():
    body = driver.protocol_constants()
    assert body["format"] == driver.FORMAT
    assert body["recipe"]["updates"] == 2400
    assert body["recipe"]["steps"] == 4
    assert body["evaluation"]["reasoning_steps"] == 4
    assert body["recipe"]["physical_weights"] == list(wide.PHYSICAL_WEIGHTS)
    assert body["budgets"] == {"planned_seconds": driver.PLANNED_SECONDS,
                               "hard_cap_seconds": driver.HARD_CAP_SECONDS,
                               "per_seed_seconds": driver.PER_SEED_SECONDS}
    assert body["seeds"] == [41, 42, 43]
    assert body["scientific_claim"] is False and body["test_read"] is False
    assert "anomaly" in body["difference"] and "dim 192" in body["difference"]
    assert "K=4" in body["difference"]


def test_round_spec_matches_the_driver_and_device_mode():
    text = (ROOT / "docs" / "goals" / "s3-anchor-isolation.md").read_text(encoding="utf-8")
    marker = "<!-- round-spec: "
    payload = json.loads(text[text.index(marker) + len(marker):text.index(" -->", text.index(marker))])
    assert payload["arm"] == "s3-anchor-isolation"
    assert payload["driver"] == "scripts/study_r7_s3_anchor_isolation.py"
    assert payload["control_record"] == driver.PREVIOUS_DOSE_ARM_RECORD
    assert payload["seeds"] == list(wide.SEEDS)
    assert payload["device_mode"] == driver.DEVICE_MODE
    assert payload["test_path"] == ["tests/test_r7_s3_anchor_isolation.py"]
    assert set(payload["commit_paths"]) >= {
        "model/weather_forecaster_r7.py", "model/process_step_r7.py",
        "scripts/r7_s3_anchor_table_build.py", "scripts/r7_s3_anchor_isolation_parent.py",
        "scripts/study_r7_s3_anchor_isolation.py", "tests/test_r7_s3_anchor_isolation.py"}


# --- the frozen protocol ----------------------------------------------------------------------

def _arms():
    from training.r7_experiment import canonical_digest
    records = {}
    for name, feedback in driver.ARMS:
        spec = driver.arm_spec(feedback)
        records[name] = {"updates": driver.UPDATES, "anomaly_feedback": bool(feedback),
                         "model": {"kind": "process", "spec": spec,
                                   "spec_canonical_digest": canonical_digest(spec)}}
    assert {name for name, _ in driver.ARMS} == {driver.CONTROL, driver.CANDIDATE}
    return records


def _protocol_body():
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest
    body = {
        "execution_files_sha256": {name: sha256_file(ROOT / name) for name in driver.execution_files()},
        "train_manifest": str(wide.TRAIN_MANIFEST), "train_data_identity": "train-id",
        "val_manifest": str(wide.VAL_MANIFEST), "val_data_identity": "val-id",
        "model_code_sha256": "m" * 64, "training_code_sha256": "t" * 64, "source_sha256": "s" * 64,
        "train_windows": {"stub": 1}, "supervision": {"stub": 1}, "format": driver.FORMAT,
        "recipe": {"updates": driver.UPDATES, "steps": driver.STEPS},
        "evaluation": {"reasoning_steps": driver.REASONING_STEPS},
        "climatology_anchor": {"arms": {name: driver.anchor_block(feedback)
                                        for name, feedback in driver.ARMS}},
        "arms": {**_arms(),
                 "previous_dose_arm": {"record_id": driver.PREVIOUS_DOSE_ARM_RECORD,
                                       "updates": driver.UPDATES, "seeds": list(wide.SEEDS)}},
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
    real_files = [name for name in driver.execution_files() if (ROOT / name).is_file()]
    monkeypatch.setattr(driver, "execution_files", lambda: real_files)
    monkeypatch.setattr(harness, "sha256_file", lambda path: "0" * 64)


def _write_protocol(tmp_path, body):
    (tmp_path / "protocol.json").write_text(json.dumps(body), encoding="utf-8")
    return tmp_path


def test_validate_protocol_accepts_the_frozen_arms(tmp_path, monkeypatch):
    _patch_validate(monkeypatch)
    body = driver.validate_protocol(_write_protocol(tmp_path, _protocol_body()))
    assert body["format"] == driver.FORMAT
    assert sorted(body["climatology_anchor"]["arms"]) == [driver.CANDIDATE, driver.CONTROL]


def test_validate_protocol_rejects_a_swapped_feedback_flag(tmp_path, monkeypatch):
    _patch_validate(monkeypatch)
    body = _protocol_body()
    body["arms"][driver.CANDIDATE]["anomaly_feedback"] = False
    from training.r7_experiment import canonical_digest
    body.pop("protocol_sha256")
    body["protocol_sha256"] = canonical_digest(body)
    with pytest.raises(RuntimeError):
        driver.validate_protocol(_write_protocol(tmp_path, body))


def test_validate_protocol_rejects_a_re_enabled_output_prior(tmp_path, monkeypatch):
    _patch_validate(monkeypatch)
    body = _protocol_body()
    body["arms"][driver.CANDIDATE]["model"]["spec"]["anchor_output_prior"] = True
    from training.r7_experiment import canonical_digest
    body["arms"][driver.CANDIDATE]["model"]["spec_canonical_digest"] = canonical_digest(
        body["arms"][driver.CANDIDATE]["model"]["spec"])
    body.pop("protocol_sha256")
    body["protocol_sha256"] = canonical_digest(body)
    with pytest.raises(RuntimeError):
        driver.validate_protocol(_write_protocol(tmp_path, body))


def test_validate_protocol_rejects_a_changed_anchor_block(tmp_path, monkeypatch):
    _patch_validate(monkeypatch)
    body = _protocol_body()
    body["climatology_anchor"]["arms"][driver.CANDIDATE]["new_params"] = 444864
    from training.r7_experiment import canonical_digest
    body.pop("protocol_sha256")
    body["protocol_sha256"] = canonical_digest(body)
    with pytest.raises(RuntimeError):
        driver.validate_protocol(_write_protocol(tmp_path, body))


def test_validate_protocol_rejects_a_changed_budget(tmp_path, monkeypatch):
    _patch_validate(monkeypatch)
    body = _protocol_body()
    body["per_seed_seconds"] = driver.PER_SEED_SECONDS + 1.0
    from training.r7_experiment import canonical_digest
    body.pop("protocol_sha256")
    body["protocol_sha256"] = canonical_digest(body)
    with pytest.raises(RuntimeError):
        driver.validate_protocol(_write_protocol(tmp_path, body))


def test_validate_protocol_rejects_a_tampered_digest(tmp_path, monkeypatch):
    _patch_validate(monkeypatch)
    body = _protocol_body()
    body["protocol_sha256"] = "0" * 64
    with pytest.raises(RuntimeError):
        driver.validate_protocol(_write_protocol(tmp_path, body))


# --- the parent pins --------------------------------------------------------------------------

class _FakeReceipt:
    def __init__(self, text):
        self._text = text

    def read_text(self, encoding="utf-8"):
        return self._text


def test_pinned_parent_refuses_a_stale_revision(monkeypatch):
    monkeypatch.setattr(anchor_parent, "MIGRATION_RECEIPT",
                        _FakeReceipt(json.dumps({"current_model_code_sha256": "0" * 64})))
    with pytest.raises(RuntimeError):
        anchor_parent.pinned_parent(41)


def test_pinned_parent_pins_this_revisions_migration():
    assert anchor_parent.MIGRATED_PARENT_RUN.name == (
        "r7_s3_rollout_dose_parent_migrated_20261015_attempt03")
    for seed in (41, 42, 43):
        assert len(anchor_parent.MIGRATED_PARENT_SHA256[seed]) == 64


def test_pinned_parent_refuses_a_foreign_seed():
    with pytest.raises(ValueError):
        anchor_parent.pinned_parent(999)


def test_install_parent_pins_writes_all_seeds():
    body = {"arms": {"parent": {"pins": {"checkpoints": {"41": {"path": "stale", "sha256": "stale"}}}}}}
    anchor_parent.install_parent_pins(body)
    checkpoints = body["arms"]["parent"]["pins"]["checkpoints"]
    assert sorted(checkpoints) == ["41", "42", "43"]
    for seed, entry in checkpoints.items():
        assert entry["sha256"] == anchor_parent.MIGRATED_PARENT_SHA256[int(seed)]
        assert entry["path"].endswith(f"parent_update_{anchor_parent.ENDPOINT:07d}.pt")
