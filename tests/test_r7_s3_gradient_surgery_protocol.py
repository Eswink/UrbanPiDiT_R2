"""CI-safe protocol and round-spec checks for the S3 gradient-surgery round.

Split from ``tests/test_r7_s3_gradient_surgery.py`` to stay inside R-051. These run
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


driver = _load("study_r7_s3_gradient_surgery")
parent = driver.parent
report = driver.report
wide = driver.wide
interior = driver.interior
import training.r7_gradient_surgery as surgery
import training.r7_long_rollout_runner as runner
from training.r7_experiment import canonical_digest


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


# --------------------------------------------------------------------------- frozen constants


def test_frozen_constants_are_the_registered_recipe():
    assert driver.UPDATES == 2400 and driver.STEPS == driver.REASONING_STEPS == 4
    assert (driver.DIM, driver.DEPTH) == (192, 4) and driver.SEEDS == (41, 42, 43)
    assert driver.REGISTERED_PHYSICAL_WEIGHTS == (1., .5, 0., .5, 0., 0., 0., .5, 0., 0., 0., .5)
    assert driver.N_PHYSICAL_STEPS == 12 and sum(driver.REGISTERED_PHYSICAL_WEIGHTS) == 3.0
    assert driver.physical_weight_block()["direct_steps"] == [0, 1, 3, 7, 11]
    assert driver.REGISTERED_DEEP_WEIGHTS == pytest.approx(
        (2 / 15, 1 / 6, 1 / 5, 7 / 30, 4 / 15), abs=1e-12)
    assert driver.draft_organization_block()["nonfinal_mass"] == pytest.approx(11. / 15., abs=1e-12)
    assert driver.CANDIDATE_GRADIENT_SURGERY == "pcgrad"
    assert driver.CONTROL_GRADIENT_SURGERY == "off"
    assert driver.CONTROL_TRAINING_CODE_SHA256 == (
        "8ad77d9349bdc4930e6bbba05f0c417814fe3093fe39aa921e816af197494974")
    assert (driver.BRIDGE_MIN, driver.BRIDGE_MAX) == (-4.8e-6, 1.8e-5)
    assert (driver.PLANNED_SECONDS, driver.HARD_CAP_SECONDS) == (43200., 86400.)
    assert driver.PER_SEED_SECONDS == 21600.


def test_the_gradient_surgery_block_is_frozen(anchor_stub):
    block = driver.gradient_surgery_block()
    assert block["candidate_mode"] == "pcgrad" and block["control_mode"] == "off"
    assert block["groups"]["A"] == {"steps": [0, 1, 3], "leads_hours": [6, 12, 24],
                                    "weights": [1.0, 0.5, 0.5]}
    assert block["groups"]["B"] == {"steps": [7, 11], "leads_hours": [48, 72],
                                    "weights": [0.5, 0.5]}
    assert block["zero_weight_steps"] == [2, 4, 5, 6, 8, 9, 10]
    assert "two autograd.grad calls" in block["backward"]
    assert block["order"].startswith("fixed A -> B")
    assert "none" in block["sweep"]


def test_changed_frozen_constants_are_refused(monkeypatch):
    for name, value in (("CANDIDATE_GRADIENT_SURGERY", "off"),
                        ("CONTROL_GRADIENT_SURGERY", "pcgrad"),
                        ("CANDIDATE_GRADIENT_SURGERY", "random")):
        monkeypatch.setattr(driver, name, value)
        with pytest.raises(RuntimeError):
            driver.gradient_surgery_block()
        monkeypatch.undo()
    monkeypatch.setattr(driver, "REGISTERED_PHYSICAL_WEIGHTS", (0.5,) * 12)
    with pytest.raises(RuntimeError):
        driver.physical_weight_block()
    with pytest.raises(RuntimeError):
        driver.gradient_surgery_block()
    monkeypatch.undo()
    monkeypatch.setattr(driver, "REGISTERED_DEEP_WEIGHTS", (0.2,) * 5)
    with pytest.raises(RuntimeError):
        driver.draft_organization_block()


def test_arm_spec_is_exactly_the_pinned_control_spec(anchor_stub):
    spec = driver.arm_spec()
    assert spec == dict(_archived_spec(), climatology_anchor_spec=anchor_stub,
                        anchor_output_prior=False, anomaly_feedback=False)
    assert "decoder_base_mode" not in spec
    assert (spec["dim"], spec["depth"]) == (192, 4)


def test_arm_spec_refuses_a_switched_or_wrong_shaped_control_spec(anchor_stub, monkeypatch):
    good = dict(_archived_spec(), climatology_anchor_spec=anchor_stub,
                anchor_output_prior=False, anomaly_feedback=False)
    bad_specs = (dict(good, decoder_base_mode="zero"), dict(good, dim=128),
                 dict(good, depth=8), dict(good, anomaly_feedback=True),
                 dict(good, anchor_output_prior=True))
    for bad in bad_specs:
        monkeypatch.setattr(driver.anchor_arm, "arm_spec", lambda *_a, _bad=bad, **_k: dict(_bad))
        with pytest.raises(RuntimeError):
            driver.arm_spec()


def test_protocol_constants_carry_only_the_gradient_surgery_change(anchor_stub):
    body = driver.protocol_constants()
    assert body["format"] == driver.FORMAT
    assert body["recipe"]["updates"] == 2400 and body["recipe"]["steps"] == 4
    assert body["evaluation"]["reasoning_steps"] == 4
    assert body["recipe"]["physical_weights"] == list(driver.REGISTERED_PHYSICAL_WEIGHTS)
    assert body["gradient_surgery"] == driver.gradient_surgery_block()
    assert body["candidate_draft_supervision"] == body["control_draft_supervision"] == "deep"
    assert body["draft_supervision_organization"] == driver.draft_organization_block()
    assert body["physical_weight_vector"] == driver.physical_weight_block()
    assert body["seeds"] == [41, 42, 43]
    assert body["budgets"] == {"planned_seconds": driver.PLANNED_SECONDS,
                               "hard_cap_seconds": driver.HARD_CAP_SECONDS,
                               "per_seed_seconds": driver.PER_SEED_SECONDS}
    assert body["scientific_claim"] is False and body["test_read"] is False
    assert "pcgrad" in body["difference"] and "K=4" in body["difference"]


def test_the_identity_pins_the_candidate_and_both_references(anchor_stub, monkeypatch):
    monkeypatch.setattr(interior, "_identity", lambda: {
        "arms": {"narrow_control": {"record_id": "narrow"}}, "train_windows": {"x": 1},
        "train_data_identity": "train-id", "source_sha256": "s" * 64,
        "training_code_sha256": "t" * 64})
    monkeypatch.setattr(driver, "previous_arm_pins", lambda: {"record_id": "dose"})
    monkeypatch.setattr(driver, "paired_control_pins",
                        lambda *a, **k: {"record_id": "anchor",
                                         "training_code_sha256": driver.CONTROL_TRAINING_CODE_SHA256})
    monkeypatch.setattr(parent, "install_parent_pins",
                        lambda body: body["arms"]["parent"].update(pins={"ok": 1}))
    monkeypatch.setattr("training.r7_arm_harness.sha256_file", lambda path: "0" * 64)
    body = driver._identity()
    candidate = body["arms"][driver.CANDIDATE]
    assert candidate["gradient_surgery"] == "pcgrad"
    assert candidate["spatial_gradient_lambda"] == 0.0 and candidate["draft_supervision"] == "deep"
    assert candidate["ema_decay"] is None and "decoder_base_mode" not in candidate["model"]["spec"]
    assert candidate["model"]["anchor_keys"] == list(parent.ANCHOR_KEYS)
    assert candidate["selection"] == "frozen endpoint; no validation selection"
    assert body["arms"]["previous_dose_arm"] == {"record_id": "dose"}
    assert body["arms"]["paired_control_arm"]["record_id"] == "anchor"
    assert body["arms"]["narrow_control"] == {"record_id": "narrow"}
    assert candidate["model"]["spec_canonical_digest"] == canonical_digest(driver.arm_spec())
    assert body["gradient_surgery"] == driver.gradient_surgery_block()
    assert body["arms"]["parent"]["pins"] == {"ok": 1}
    assert body["code_version_comparison"]["candidate_training_code_sha256"] == "t" * 64
    assert body["code_version_comparison"]["control_training_code_sha256"] == \
        driver.CONTROL_TRAINING_CODE_SHA256
    assert body["code_version_comparison"]["equal"] is False
    assert body["code_version_comparison"]["active_path"]["gradient_surgery"] == "pcgrad"


# --------------------------------------------------------------------------- protocol validation


def _protocol_body(anchor_stub):
    spec = copy.deepcopy(driver.arm_spec())
    body = {
        "execution_files_sha256": {"x.py": "0" * 64},
        "train_manifest": str(wide.TRAIN_MANIFEST), "train_data_identity": "train-id",
        "val_manifest": str(wide.VAL_MANIFEST), "val_data_identity": "val-id",
        "model_code_sha256": parent.CURRENT_MODEL_CODE_SHA256,
        "training_code_sha256": "t" * 64,
        "source_sha256": "s" * 64, "train_windows": {"stub": 1}, "supervision": {"stub": 1},
        "format": driver.FORMAT,
        "recipe": {"updates": driver.UPDATES, "steps": driver.STEPS,
                   "physical_weights": list(driver.REGISTERED_PHYSICAL_WEIGHTS)},
        "evaluation": {"reasoning_steps": driver.REASONING_STEPS},
        "candidate_draft_supervision": "deep", "control_draft_supervision": "deep",
        "gradient_surgery": driver.gradient_surgery_block(),
        "draft_supervision_organization": driver.draft_organization_block(),
        "physical_weight_vector": driver.physical_weight_block(),
        "code_version_comparison": {
            "candidate_training_code_sha256": "t" * 64,
            "control_training_code_sha256": driver.CONTROL_TRAINING_CODE_SHA256,
            "equal": False,
            "active_path": {"draft_supervision": "deep", "gradient_lambda": 0.0,
                            "ema_decay": None, "gradient_surgery": "pcgrad"}},
        "arms": {driver.CANDIDATE: {
                     "physical_weights": list(driver.REGISTERED_PHYSICAL_WEIGHTS),
                     "physical_steps": 12, "draft_supervision": "deep",
                     "gradient_surgery": "pcgrad", "spatial_gradient_lambda": 0.0,
                     "ema_decay": None, "updates": driver.UPDATES, "steps": driver.STEPS,
                     "model": {"kind": "process", "spec": spec,
                               "spec_canonical_digest": canonical_digest(spec)}},
                 "previous_dose_arm": {"record_id": driver.PREVIOUS_DOSE_ARM_RECORD,
                                       "updates": driver.UPDATES, "seeds": list(driver.SEEDS)},
                 "paired_control_arm": {"record_id": driver.PAIRED_CONTROL_RECORD,
                                        "updates": driver.UPDATES, "data_identity": "train-id",
                                        "source_sha256": "s" * 64,
                                        "physical_weights": list(driver.REGISTERED_PHYSICAL_WEIGHTS),
                                        "draft_supervision": "deep",
                                        "gradient_surgery": "off",
                                        "training_code_sha256": driver.CONTROL_TRAINING_CODE_SHA256,
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
    monkeypatch.setattr(experiment, "model_code_digest", lambda: parent.CURRENT_MODEL_CODE_SHA256)
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


def test_validate_protocol_accepts_the_frozen_gradient_surgery_arm(tmp_path, monkeypatch,
                                                                   anchor_stub):
    _patch_validate(monkeypatch)
    body = driver.validate_protocol(_write(tmp_path, _protocol_body(anchor_stub)))
    assert body["gradient_surgery"]["candidate_mode"] == "pcgrad"


def _replace(body, path, value):
    node = body
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


@pytest.mark.parametrize("path,value", [
    (("recipe", "physical_weights"), [0.3] * 12),
    (("recipe", "updates"), 2000),
    (("gradient_surgery", "candidate_mode"), "off"),
    (("arms", driver.CANDIDATE, "gradient_surgery"), "off"),
    (("arms", driver.CANDIDATE, "ema_decay"), 0.999),
    (("arms", driver.CANDIDATE, "spatial_gradient_lambda"), 0.5),
    (("arms", driver.CANDIDATE, "draft_supervision"), "final_only"),
    (("arms", driver.CANDIDATE, "physical_steps"), 11),
    (("arms", driver.CANDIDATE, "model", "spec", "decoder_base_mode"), "persistence"),
    (("arms", driver.CANDIDATE, "model", "spec", "depth"), 8),
    (("model_code_sha256",), "m" * 64),
    (("arms", "paired_control_arm", "record_id"), "s3-wide-interior-dose"),
    (("arms", "paired_control_arm", "gradient_surgery"), "pcgrad"),
    (("arms", "paired_control_arm", "training_code_sha256"), "c" * 64),
    (("arms", "paired_control_arm", "protocol_sha256"), "q" * 64),
])
def test_validate_protocol_rejects_a_tampered_factor(tmp_path, monkeypatch, anchor_stub,
                                                     path, value):
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
    for mutate in ("surgery_block", "organization", "budget", "code", "format", "claim", "digest"):
        body = _protocol_body(anchor_stub)
        if mutate == "surgery_block":
            body["gradient_surgery"] = dict(body["gradient_surgery"], candidate_mode="off")
        elif mutate == "organization":
            body["draft_supervision_organization"] = dict(
                body["draft_supervision_organization"], nonfinal_mass=0.5)
        elif mutate == "budget":
            body["per_seed_seconds"] += 1.0
        elif mutate == "code":
            body["code_version_comparison"] = dict(
                body["code_version_comparison"],
                active_path=dict(body["code_version_comparison"]["active_path"],
                                 gradient_surgery="off"))
        elif mutate == "format":
            body["format"] = "r7-s3-decoder-base-anchor-protocol-v1"
        elif mutate == "claim":
            body["scientific_claim"] = True
        else:
            body["protocol_sha256"] = "0" * 64
        if mutate != "digest":
            body.pop("protocol_sha256")
            body["protocol_sha256"] = canonical_digest(body)
        with pytest.raises(RuntimeError):
            driver.validate_protocol(_write(tmp_path, body))


# --------------------------------------------------------------------------- readings


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


def _surgery_seed(projection_count):
    summary = {key: 0 for key in ("updates", "conflict_updates", "projection_count")}
    summary.update(updates=driver.UPDATES, projection_count=projection_count,
                   conflict_updates=projection_count)
    return {"candidate": {"gradient_surgery_summary": summary}}


def test_zero_projection_is_reported_as_null_or_inert():
    seeds = {str(seed): _surgery_seed(0) for seed in driver.SEEDS}
    inert, counts = report.null_inert(seeds)
    assert inert is True and counts == [0, 0, 0]
    seeds[str(driver.SEEDS[1])] = _surgery_seed(1)
    inert, counts = report.null_inert(seeds)
    assert inert is False and counts == [0, 1, 0]
    # A conflict without a projection would still not be inert; the rule checks both.
    conflict_only = {str(seed): _surgery_seed(0) for seed in driver.SEEDS}
    conflict_only[str(driver.SEEDS[0])]["candidate"]["gradient_surgery_summary"][
        "conflict_updates"] = 1
    assert report.null_inert(conflict_only)[0] is False


# --------------------------------------------------------------------------- round spec


def test_parent_pins_are_the_reused_decoder_round_migration():
    assert parent.MIGRATION_RECEIPT_SHA256 == \
        "002f08439b99fc3658a422e0732183f7a48c5b43d5790c133c53c4eaf004d6c3"
    assert parent.CURRENT_MODEL_CODE_SHA256 == \
        "bec3859fc1386b14777300ad3e387546829c8a0fbf52e817e1cb90324b988ac1"
    assert parent.MIGRATED_PARENT_STATE_DIGEST == {
        41: "7c416ccb4054cf7e077788a43b02917e6a2dcf2503058152a7f0ab4b0ab70707",
        42: "857d880466f76500ed74c534e757732ae9eff4c2b3f0bbc0fa74b895fcae3623",
        43: "61fdac58b7bfacf828e1eae4968817b1cbd46616442c6d16ac0fbbb63e294e1b"}
    assert parent.EXPANDED_STATE_SHA256 == {
        41: "00ada5e25c4e7ef16acfb24768408f06e5e6ab091d4a91bdc7505c99229de612",
        42: "baed2ff9ca898af7c6590f5ba269411d6bd660bd9601b5bcee7ae2c2a2b660a1",
        43: "4dda647f86ffee6befcab749e04153e7dbf09c6034e62d03a8f7d00bf066cce7"}
    assert parent.ANCHOR_INIT_SHA256 == \
        "de0e76beb548881b282fbd8c6f194055286ec7d0dd1d7ba88dad476fc2aa1112"


def test_execution_files_cover_the_imported_training_and_control_scripts():
    files = driver.execution_files()
    for name in ("training/r7_long_rollout_runner.py", "training/r7_gradient_surgery.py",
                 "scripts/r7_s3_gradient_surgery_support.py",
                 "scripts/r7_s3_gradient_surgery_text.py",
                 "scripts/r7_s3_decoder_base_anchor_parent.py",
                 "scripts/study_r7_s3_anchor_isolation.py",
                 "scripts/r7_s3_anchor_isolation_parent.py"):
        assert name in files, name


def test_round_spec_matches_the_driver_and_splits_the_two_commits():
    text = (ROOT / "docs" / "goals" / "s3-gradient-surgery.md").read_text(encoding="utf-8")
    marker = "<!-- round-spec: "
    payload = json.loads(text[text.index(marker) + len(marker):text.index(" -->", text.index(marker))])
    assert payload["arm"] == "s3-gradient-surgery"
    assert payload["driver"] == "scripts/study_r7_s3_gradient_surgery.py"
    assert payload["output_stem"] == "r7_s3_gradient_surgery_20261020"
    assert payload["control_record"] == driver.PAIRED_CONTROL_RECORD
    assert payload["reference_record"] == driver.PREVIOUS_DOSE_ARM_RECORD
    assert payload["device_mode"] == driver.DEVICE_MODE
    assert payload["seeds"] == list(driver.SEEDS)
    assert payload["test_path"] == ["tests/test_r7_s3_gradient_surgery.py",
                                    "tests/test_r7_s3_gradient_surgery_protocol.py"]
    assert payload["code_archive"] == "outputs/r7_s3_gradient_surgery_code.zip"
    assert "docs/R7_S3_GRADIENT_SURGERY_RESULT.md" in payload["register_paths"]
    assert "docs/R7_S3_GRADIENT_SURGERY_RESULT.md" not in payload["commit_paths"]
    for name in ("training/r7_long_rollout_runner.py", "training/r7_gradient_surgery.py",
                 "scripts/study_r7_s3_gradient_surgery.py",
                 "scripts/r7_s3_gradient_surgery_support.py",
                 "scripts/r7_s3_gradient_surgery_text.py",
                 "tests/test_r7_s3_gradient_surgery.py",
                 "tests/test_r7_s3_gradient_surgery_protocol.py",
                 "docs/goals/s3-gradient-surgery.md",
                 "docs/goals/main-model-climatology-campaign.md"):
        assert name in payload["commit_paths"], name
    for name in ("docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md",
                 "docs/goals/main-model-climatology-campaign.md",
                 "docs/goals/s3-gradient-surgery.md"):
        assert name in payload["register_paths"], name
    assert payload["commit_message"] and payload["register_message"]
    assert "planned_seconds" in text and "hard_cap_seconds" in text
    assert "43200" in text and "86400" in text


def test_control_and_reference_constants_are_the_registered_ones():
    assert driver.PAIRED_CONTROL_RECORD == "s3-anchor-isolation-result"
    assert driver.PAIRED_CONTROL_RUN.name == "r7_s3_anchor_isolation_20261015_attempt01"
    assert driver.PREVIOUS_DOSE_ARM_RECORD == "s3-wide-interior-dose"
    assert surgery.GRADIENT_SURGERY_MODES == ("off", "pcgrad")
