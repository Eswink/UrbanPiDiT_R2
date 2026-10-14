"""CI-safe checks for the H-B spatially-resolved typed-evidence driver and its parent module.

H-B keeps the #79 typed-evidence pathway's per-type projections and its information and changes one
thing: how the ``P`` patch positions are collapsed. The pooled arm averages them uniformly; the
spatial arm aggregates them with a softmax attention over one shared learned query. The fidelity
critical parts are that the two modes differ by exactly ``typed_evidence.position_query`` and by
nothing else, that at a zero query the spatial aggregation is the pooled one, that zeroing the
per-type projections zeroes the evidence in both modes, that the expansion adds exactly the pathway
keys and no others, and that the frozen protocol pins both arms and the re-migrated parent.

Every check runs against tiny synthetic specs and a synthetic payload, so this file is checked on a
checkout with no ``outputs/``. Each check has a counterproof.
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

VARIABLES = ("t2m", "u10", "v10", "mslp", "tcc", "tcwv", "sp", "z500", "t500", "u500",
             "v500", "q500", "z850", "t850", "u850", "v850", "r850")
PAYLOAD = {"channels": list(VARIABLES),
           "denorm_mean": [0.0] * len(VARIABLES), "denorm_std": [1.0] * len(VARIABLES),
           "field_scale": [1.0, 1.0, 1.0, 1.0], "field_mean": [0.0, 0.0, 0.0, 0.0],
           "field_std": [1.0, 1.0, 1.0, 1.0], "field_active": [True, True, True, True]}


def _load(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


spatial = _load("study_r7_s3_typed_spatial")
typed_parent = spatial.typed_parent
wide = spatial.wide


def _tiny_spec(dim=8, mode=None):
    spec = dict(architecture="window", in_channels=17, out_channels=17, history_steps=2, dim=dim,
                depth=4, heads=2, window_size=2, patch_size=2, dropout=0.,
                spacetime_inputs=True, default_reasoning_steps=4, detach_between_steps=False,
                anchored_processes=4, free_processes=1)
    if mode is not None:
        spec["typed_evidence_mode"] = mode
        spec["typed_evidence"] = dict(PAYLOAD)
    return spec


def _router(mode, dim=8, seed=0):
    from model.typed_evidence_r7 import TypedEvidenceRouter
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return TypedEvidenceRouter(dim, 2, mode=mode, **PAYLOAD)


# --- the routing modes -------------------------------------------------------------------------

def test_the_spatial_mode_is_declared_and_per_type():
    from model.typed_evidence_r7 import (
        EVIDENCE_MODES, GENERIC_FUSION, PER_TYPE_MODES, TYPED_ROUTING, TYPED_SPATIAL,
    )
    assert TYPED_SPATIAL == "typed_spatial"
    assert set(EVIDENCE_MODES) == {TYPED_ROUTING, TYPED_SPATIAL, GENERIC_FUSION}
    assert set(PER_TYPE_MODES) == {TYPED_ROUTING, TYPED_SPATIAL}
    assert GENERIC_FUSION not in PER_TYPE_MODES


def test_an_unknown_mode_is_refused():
    from model.typed_evidence_r7 import TypedEvidenceRouter
    with pytest.raises(ValueError):
        TypedEvidenceRouter(8, 2, mode="typed_made_up", **PAYLOAD)


# --- the two typed arms differ by exactly the query --------------------------------------------

def test_the_spatial_arm_adds_exactly_the_position_query():
    from training.r7_experiment import make_model
    from model.typed_evidence_r7 import TYPED_ROUTING, TYPED_SPATIAL
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        pooled = make_model("process", _tiny_spec(mode=TYPED_ROUTING)).state_dict()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        spatial_state = make_model("process", _tiny_spec(mode=TYPED_SPATIAL)).state_dict()
    assert sorted(set(spatial_state) - set(pooled)) == ["typed_evidence.position_query"]
    assert not set(pooled) - set(spatial_state)
    shared = [key for key in pooled if key in spatial_state]
    assert shared and all(torch.equal(pooled[key], spatial_state[key]) for key in shared)


def test_the_query_is_the_only_extra_parameter():
    from training.r7_experiment import make_model
    from model.typed_evidence_r7 import TYPED_ROUTING, TYPED_SPATIAL
    pooled = make_model("process", _tiny_spec(mode=TYPED_ROUTING))
    spatial_model = make_model("process", _tiny_spec(mode=TYPED_SPATIAL))
    difference = (sum(p.numel() for p in spatial_model.parameters())
                  - sum(p.numel() for p in pooled.parameters()))
    assert difference == int(spatial_model.typed_evidence.position_query.numel()) == 8


def test_typed_evidence_keys_match_the_model_side_by_side():
    from training.r7_experiment import make_model
    from model.typed_evidence_r7 import TYPED_ROUTING, TYPED_SPATIAL
    bare = make_model("process", _tiny_spec()).state_dict()
    for mode in (TYPED_ROUTING, TYPED_SPATIAL):
        added = sorted(set(make_model("process", _tiny_spec(mode=mode)).state_dict()) - set(bare))
        assert added == sorted(typed_parent.typed_evidence_keys(mode))


# --- the aggregation identity ------------------------------------------------------------------

def test_a_zero_query_is_the_pooled_aggregation():
    from model.typed_evidence_r7 import TYPED_ROUTING, TYPED_SPATIAL
    pooled, spatial_router = _router(TYPED_ROUTING), _router(TYPED_SPATIAL)
    assert int(torch.count_nonzero(spatial_router.position_query)) == 0
    fields = torch.randn(2, 4, 8, 8)
    with torch.no_grad():
        reference = pooled.token_evidence(fields)
        assert float((reference - spatial_router.token_evidence(fields)).abs().max()) <= 1e-6


def test_a_nonzero_query_moves_the_aggregation():
    from model.typed_evidence_r7 import TYPED_ROUTING, TYPED_SPATIAL
    pooled, spatial_router = _router(TYPED_ROUTING), _router(TYPED_SPATIAL)
    fields = torch.randn(2, 4, 8, 8)
    with torch.no_grad():
        reference = pooled.token_evidence(fields)
        spatial_router.position_query.normal_()
        moved = spatial_router.token_evidence(fields)
    assert float((reference - moved).abs().max()) > 1e-3


def test_zeroing_the_projections_zeroes_the_evidence_in_both_modes():
    from model.typed_evidence_r7 import TYPED_ROUTING, TYPED_SPATIAL
    fields = torch.randn(2, 4, 8, 8)
    for mode in (TYPED_ROUTING, TYPED_SPATIAL):
        router = _router(mode)
        with torch.no_grad():
            if mode == TYPED_SPATIAL:
                router.position_query.normal_()
            for projection in router.projections:
                projection.weight.zero_()
            assert int(torch.count_nonzero(router.token_evidence(fields))) == 0


def test_a_masked_field_stays_zero_in_the_spatial_mode():
    from model.typed_evidence_r7 import TYPED_SPATIAL
    router = _router(TYPED_SPATIAL)
    with torch.no_grad():
        router.position_query.normal_()
        router.field_active[0] = False
        evidence = router.token_evidence(torch.randn(2, 4, 8, 8))
    assert int(torch.count_nonzero(evidence[:, 0])) == 0
    assert int(torch.count_nonzero(evidence[:, 1:])) > 0


# --- the expansion adds exactly the pathway keys ------------------------------------------------

def test_expand_refuses_a_state_that_is_not_the_registered_parent():
    from training.r7_experiment import make_model
    from model.typed_evidence_r7 import TYPED_SPATIAL
    spec = _tiny_spec(mode=TYPED_SPATIAL)
    wrong = make_model("process", _tiny_spec(mode=None)).state_dict()
    del wrong["process_queries"]
    with pytest.raises(RuntimeError):
        typed_parent.expand_parent_state(wrong, spec, PAYLOAD, TYPED_SPATIAL)


def test_expand_installs_the_parent_and_keeps_the_pathway_random():
    from training.r7_experiment import make_model
    from model.typed_evidence_r7 import TYPED_SPATIAL
    spec = _tiny_spec(mode=TYPED_SPATIAL)
    parent = make_model("process", _tiny_spec()).state_dict()
    model, state, receipt = typed_parent.expand_parent_state(parent, spec, PAYLOAD, TYPED_SPATIAL)
    assert sorted(receipt["new_keys"]) == sorted(typed_parent.typed_evidence_keys(TYPED_SPATIAL))
    assert receipt["parent_keys"] == len(parent)
    for key, value in parent.items():
        assert torch.equal(state[key], value)
    model.load_state_dict(state, strict=True)


# --- the driver's frozen fields -----------------------------------------------------------------

def test_arm_spec_sets_the_mode_and_the_payload():
    from model.typed_evidence_r7 import TYPED_SPATIAL
    spec = spatial.arm_spec(TYPED_SPATIAL, PAYLOAD)
    assert spec["typed_evidence_mode"] == TYPED_SPATIAL
    assert spec["typed_evidence"] == PAYLOAD
    assert spec["typed_evidence"] is not PAYLOAD


def test_arm_spec_refuses_a_spec_that_already_carries_a_switch(monkeypatch):
    from model.typed_evidence_r7 import TYPED_SPATIAL
    monkeypatch.setattr(spatial.recipe, "_shared",
                        lambda: type("S", (), {"archived_process_spec": lambda self: (
                            _tiny_spec(mode=TYPED_SPATIAL), None, None)})())
    with pytest.raises(RuntimeError):
        spatial.arm_spec(TYPED_SPATIAL, PAYLOAD)


def test_arm_name_round_trips_and_refuses_an_unknown_mode():
    from model.typed_evidence_r7 import TYPED_ROUTING, TYPED_SPATIAL
    assert spatial.arm_name(TYPED_ROUTING) == spatial.POOLED
    assert spatial.arm_name(TYPED_SPATIAL) == spatial.SPATIAL
    with pytest.raises(ValueError):
        spatial.arm_name("typed_made_up")


def test_typed_block_declares_the_mode_and_its_keys():
    from model.typed_evidence_r7 import TYPED_ROUTING, TYPED_SPATIAL
    pooled = spatial.typed_block(TYPED_ROUTING, PAYLOAD)
    spatial_block = spatial.typed_block(TYPED_SPATIAL, PAYLOAD)
    assert pooled["mode"] == TYPED_ROUTING and spatial_block["mode"] == TYPED_SPATIAL
    assert spatial_block["new_keys"] == list(typed_parent.typed_evidence_keys(TYPED_SPATIAL))
    assert "position_query" not in " ".join(pooled["new_keys"])
    assert spatial_block["positive_control"] == pooled["positive_control"]
    assert spatial_block["field_active"] == PAYLOAD["field_active"]


# --- the frozen protocol ------------------------------------------------------------------------

def _arms():
    from training.r7_experiment import canonical_digest
    from model.typed_evidence_r7 import TYPED_ROUTING, TYPED_SPATIAL
    records = {}
    for name, mode in spatial.ARMS:
        spec = spatial.arm_spec(mode, PAYLOAD)
        records[name] = {"updates": spatial.UPDATES, "mode": mode,
                         "model": {"kind": "process", "spec": spec,
                                   "spec_canonical_digest": canonical_digest(spec)}}
    assert {name for name, _ in spatial.ARMS} == {spatial.POOLED, spatial.SPATIAL}
    assert TYPED_ROUTING in dict(spatial.ARMS).values()
    assert TYPED_SPATIAL in dict(spatial.ARMS).values()
    return records


def _protocol_body():
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest
    body = {
        "execution_files_sha256": {name: sha256_file(ROOT / name) for name in spatial.execution_files()},
        "train_manifest": str(wide.TRAIN_MANIFEST), "train_data_identity": "train-id",
        "val_manifest": str(wide.VAL_MANIFEST), "val_data_identity": "val-id",
        "model_code_sha256": "m" * 64, "training_code_sha256": "t" * 64, "source_sha256": "s" * 64,
        "train_windows": {"stub": 1}, "supervision": {"stub": 1}, "format": spatial.FORMAT,
        "recipe": {"updates": spatial.UPDATES, "steps": spatial.STEPS},
        "evaluation": {"reasoning_steps": spatial.REASONING_STEPS},
        "typed_evidence": {"sidecar_store": "store", "channels": list(PAYLOAD["channels"]),
                           "field_active": PAYLOAD["field_active"],
                           "arms": {name: spatial.typed_block(mode, PAYLOAD)
                                    for name, mode in spatial.ARMS}},
        "arms": {**_arms(),
                 "previous_dose_arm": {"record_id": spatial.PREVIOUS_DOSE_ARM_RECORD,
                                       "updates": spatial.UPDATES, "seeds": list(wide.SEEDS)}},
        "planned_seconds": spatial.PLANNED_SECONDS, "hard_cap_seconds": spatial.HARD_CAP_SECONDS,
        "per_seed_seconds": spatial.PER_SEED_SECONDS,
        "estimated_peak_bytes": spatial.ESTIMATED_PEAK_BYTES,
        "headroom_margin_bytes": spatial.HEADROOM_MARGIN_BYTES,
        "scientific_claim": False, "test_read": False,
        "test_manifest_never_read": str(wide.TEST_MANIFEST),
    }
    body["protocol_sha256"] = canonical_digest(body)
    return body


def _patch_validate(monkeypatch):
    """Neutralise the store/GPU/sidecar reads so validate_protocol runs on a bare checkout."""
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
    monkeypatch.setattr(spatial.interior, "supervision_block", lambda: {"stub": 1})
    monkeypatch.setattr(spatial.devices, "assignment_is_valid", lambda *a, **k: True)
    monkeypatch.setattr(spatial, "evidence_payload", lambda: ({"store": "store"}, dict(PAYLOAD)))
    real_files = [name for name in spatial.execution_files() if (ROOT / name).is_file()]
    monkeypatch.setattr(spatial, "execution_files", lambda: real_files)
    monkeypatch.setattr(harness, "sha256_file", lambda path: "0" * 64)


def _write_protocol(tmp_path, body):
    (tmp_path / "protocol.json").write_text(json.dumps(body), encoding="utf-8")
    return tmp_path


def test_validate_protocol_accepts_the_frozen_arms(tmp_path, monkeypatch):
    _patch_validate(monkeypatch)
    body = spatial.validate_protocol(_write_protocol(tmp_path, _protocol_body()))
    assert body["format"] == spatial.FORMAT
    assert sorted(body["typed_evidence"]["arms"]) == [spatial.POOLED, spatial.SPATIAL]


def test_validate_protocol_rejects_a_swapped_arm_mode(tmp_path, monkeypatch):
    _patch_validate(monkeypatch)
    body = _protocol_body()
    body["arms"][spatial.SPATIAL]["mode"] = spatial.typed_parent.typed_evidence_keys.__name__
    from training.r7_experiment import canonical_digest
    body.pop("protocol_sha256")
    body["protocol_sha256"] = canonical_digest(body)
    with pytest.raises(RuntimeError):
        spatial.validate_protocol(_write_protocol(tmp_path, body))


def test_validate_protocol_rejects_a_changed_typed_evidence_block(tmp_path, monkeypatch):
    _patch_validate(monkeypatch)
    body = _protocol_body()
    body["typed_evidence"]["arms"][spatial.SPATIAL]["new_keys"] = []
    from training.r7_experiment import canonical_digest
    body.pop("protocol_sha256")
    body["protocol_sha256"] = canonical_digest(body)
    with pytest.raises(RuntimeError):
        spatial.validate_protocol(_write_protocol(tmp_path, body))


def test_validate_protocol_rejects_a_changed_budget(tmp_path, monkeypatch):
    _patch_validate(monkeypatch)
    body = _protocol_body()
    body["per_seed_seconds"] = spatial.PER_SEED_SECONDS + 1.0
    from training.r7_experiment import canonical_digest
    body.pop("protocol_sha256")
    body["protocol_sha256"] = canonical_digest(body)
    with pytest.raises(RuntimeError):
        spatial.validate_protocol(_write_protocol(tmp_path, body))


def test_validate_protocol_rejects_a_tampered_digest(tmp_path, monkeypatch):
    _patch_validate(monkeypatch)
    body = _protocol_body()
    body["protocol_sha256"] = "0" * 64
    with pytest.raises(RuntimeError):
        spatial.validate_protocol(_write_protocol(tmp_path, body))


# --- the parent module --------------------------------------------------------------------------

def test_pinned_parent_refuses_a_stale_revision(monkeypatch):
    import json as _json
    monkeypatch.setattr(typed_parent, "MIGRATION_RECEIPT",
                        _FakeReceipt(_json.dumps({"current_model_code_sha256": "0" * 64})))
    with pytest.raises(RuntimeError):
        typed_parent.pinned_parent(41)


def test_pinned_parent_pins_this_revisions_migration():
    assert typed_parent.MIGRATED_PARENT_RUN.name == (
        "r7_s3_rollout_dose_parent_migrated_20261014_attempt02")
    for seed in (41, 42, 43):
        assert len(typed_parent.MIGRATED_PARENT_SHA256[seed]) == 64


def test_pinned_parent_refuses_a_foreign_seed():
    with pytest.raises(ValueError):
        typed_parent.pinned_parent(999)


class _FakeReceipt:
    def __init__(self, text):
        self._text = text

    def read_text(self, encoding="utf-8"):
        return self._text
