"""Pure-JSON N1 D2 protocol/reader tests; no GPU, network or weather-data reads."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from training import r7_frozen_z_protocol as protocol
from training import r7_rw_b_subtraction_protocol as previous

ROOT = Path(__file__).resolve().parents[1]


def _spec(seed):
    # Contract fixture, NOT a real control tensor or a claim about actual data.
    return {"format": "r7-frozen-z-v1", "seed": seed,
            "random_seed": seed + 1_000_003, "shape": [1, 15, 192],
            "token_hw": [3, 5], "patch_size": 2, "std": 0.02,
            "tensor_sha256": hashlib.sha256(f"unit-test-only-{seed}".encode()).hexdigest()}


def _payload(tmp_path, **overrides):
    inputs = {"intervention_specs": {str(seed): _spec(seed) for seed in protocol.SEEDS},
              "authorization": {"record": "unit-test fixture only; not human permission"},
              "code_identity": {"code_zip_sha256": "unit-test archive identity",
                                "tree": {"module": "unit-test source identity"}}}
    inputs.update(overrides)
    measured = {name: {"parameters": 1000 if name != protocol.RW_A_ARM else 900,
                       "trainable_parameters": 800 if name == protocol.FROZEN_Z_ARM
                                               else (900 if name == protocol.RW_A_ARM else 1000),
                       "forward_flops": 100, "forward_backward_flops": 300}
                for name in protocol.ARM_NAMES}
    return protocol.protocol_payload(tmp_path / "nonexistent-manifests", "fixture-source-sha",
                                     17, measured, **inputs)


def _write_protocol(tmp_path, payload, *, rehash=False):
    if rehash:
        payload["protocol_sha256"] = protocol.canonical_digest(
            {key: value for key, value in payload.items() if key != "protocol_sha256"})
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return path


def _replace_path(payload, path, value):
    node = payload
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


def _cell(deltas=(1.0, 2.0)):
    negative = all(value < 0 for value in deltas)
    positive = all(value > 0 for value in deltas)
    return {"unit": "K", "sign_consistent": negative or positive,
            "outcome": "improved" if negative else "worsened" if positive else "unresolved",
            "seed_deltas": {str(seed): delta for seed, delta in zip(protocol.SEEDS, deltas)}}


def _pairs(primary=(1.0, 2.0), reference=(-1.0, -2.0), carrier=(1.0, -2.0)):
    return {f"{focus} - {baseline}": {
        "focus_arm": focus, "baseline_arm": baseline,
        "cells": {f"{lead}h|t2m": _cell(deltas) for lead in protocol.PRIMARY_LEADS}}
        for (focus, baseline), deltas in zip(protocol.PAIRS, (primary, reference, carrier))}


def _primary_cell(pairs, lead=48):
    focus, baseline = protocol.PAIRS[0]
    return pairs[f"{focus} - {baseline}"]["cells"][f"{lead}h|t2m"]


def test_reuses_existing_controls_not_previous_arms_or_branch():
    constants = ("SEEDS", "UPDATES", "BATCH_SIZE", "LR", "WEIGHT_DECAY", "CLIP",
                 "WARMUP_UPDATES", "MINIMUM_LR_RATIO", "VALIDATION_EVERY",
                 "EARLY_STOPPING_PATIENCE", "MINIMUM_IMPROVEMENT", "VALIDATION_LEADS",
                 "EVALUATION_LEADS", "REASONING_STEPS", "PROCESS_WEIGHT",
                 "EVALUATION_MAX_SAMPLES", "COMPARATOR_DEPTH", "DEADLINE_SECONDS")
    assert all(getattr(protocol, name) == getattr(previous, name) for name in constants)
    assert len(protocol.ARMS) == 3
    assert protocol.ARM_NAMES == (protocol.RW_A_ARM, protocol.RW_B_ARM, protocol.FROZEN_Z_ARM)
    assert protocol.ARMS[0][2] == previous.RW_A_CONFIG
    assert protocol.ARMS[1][2] == protocol.ARMS[2][2] == previous.RW_B_CONFIG
    assert all("frozen_z" not in key for _, _, config in protocol.ARMS for key in config)
    assert protocol.PAIRS == ((protocol.FROZEN_Z_ARM, protocol.RW_A_ARM),
                              (protocol.RW_B_ARM, protocol.RW_A_ARM),
                              (protocol.FROZEN_Z_ARM, protocol.RW_B_ARM))
    assert "negative control" not in protocol.PAIR_ROLES
    assert protocol.PRIMARY_DECISION_TEXT != previous.PRIMARY_DECISION_TEXT


def test_protocol_json_freeze_and_read_only_verification(tmp_path):
    payload = _payload(tmp_path)
    path = _write_protocol(tmp_path, payload)
    before, stat = path.read_bytes(), path.stat()
    assert protocol.verified_registration(path) == payload
    assert path.read_bytes() == before and path.stat().st_mtime_ns == stat.st_mtime_ns
    assert list(tmp_path.iterdir()) == [path]  # Payload/verifier never creates manifests/store.
    assert payload["protocol_sha256"] == protocol.canonical_digest(
        {key: value for key, value in payload.items() if key != "protocol_sha256"})
    assert payload["scientific_claim"] is False and payload["test_read"] is False
    assert payload["limitations"] and payload["frozen_before_any_step"] is True
    assert payload["data"]["train_read"] is True and payload["data"]["val_read"] is True
    assert payload["data"]["test_read"] is False
    assert payload["data"]["data_identity"] == "fixture-source-sha"
    assert payload["data"]["source_identity"] is None
    assert "not independently verified" in payload["data"]["source_identity_status"]
    assert payload["intervention_specs"] == {str(seed): _spec(seed) for seed in protocol.SEEDS}
    assert payload["shared_controls"]["final_weight"] == 2.0
    assert payload["shared_controls"]["weight_decay"] == 1e-4
    assert payload["cost_reporting"]["measurement_device"] == "cpu"
    assert payload["arms"][1]["parameters"] == payload["arms"][2]["parameters"]
    assert payload["arms"][2]["trainable_parameters"] < payload["arms"][1]["trainable_parameters"]
    assert payload["budget"]["gpu_hours_cap"] == .45
    assert payload["budget"]["gpu_seconds_cap"] == 1620.0
    assert payload["budget"]["global_deadline_seconds"] == 1800.0
    assert payload["budget"]["max_devices"] == 1
    assert payload["budget"]["deadline_reset_per_seed"] is False
    assert payload["budget"]["exclusivity"]
    assert payload["historical_pooling"] is False
    assert [pair["role"] for pair in payload["comparisons"]] == list(protocol.PAIR_ROLES)
    assert payload["primary_registration"]["pairs"] == payload["comparisons"]
    assert all(pair["reading"] == f"{pair['focus']} - {pair['baseline']}"
               for pair in payload["comparisons"])


def test_source_identity_is_separate_opaque_driver_metadata(tmp_path):
    source = {"source_sha256": "source-bytes", "source_receipt_sha256": "receipt-bytes",
              "preflight_report_sha256": "preflight-bytes", "build_complete": True}
    payload = _payload(tmp_path, source_identity=source)
    assert payload["data"]["source_identity"] == source
    assert payload["data"]["source_identity"] != payload["data"]["data_identity"]
    assert payload["data"]["source_identity_status"] == "driver-supplied; not independently verified"
    assert protocol.verified_registration(_write_protocol(tmp_path, payload)) == payload
    source["source_sha256"] = "changed"
    assert payload["data"]["source_identity"]["source_sha256"] == "source-bytes"


def test_digest_tamper_fails_before_semantic_verification(tmp_path):
    payload = _payload(tmp_path)
    payload["seeds"] = [41]
    with pytest.raises(RuntimeError, match="digest"):
        protocol.verified_registration(_write_protocol(tmp_path, payload))


@pytest.mark.parametrize("field,value", [
    (("seeds",), [41]), (("seeds",), [41, 42, 43]),
    (("shared_controls", "max_updates"), 401),
    (("shared_controls", "weight_decay"), 0),
    (("shared_controls", "final_weight"), 1.0),
    (("shared_controls", "lr_schedule", "warmup_updates"), 81),
    (("shared_controls", "evaluation_max_samples"), 65),
    (("arms", 0, "model_config", "depth"), 5),
    (("arms", 2, "model_config", "local_solver_state"), False),
    (("arms", 2, "kind"), "generic"),
    (("comparisons", 0, "baseline"), protocol.RW_B_ARM),
    (("comparisons", 2, "role"), "primary question"),
    (("primary_registration", "decision_text"), "nonempty but rewritten decision"),
    (("primary_registration", "branch_rule", "cannot-distinguish", "reading"), "continue"),
    (("budget", "gpu_hours_cap"), .46), (("budget", "gpu_seconds_cap"), 1621),
    (("budget", "global_deadline_seconds"), 1801), (("budget", "max_devices"), 2),
    (("budget", "deadline_reset_per_seed"), True), (("budget", "exclusivity"), "shared"),
    (("intervention", "requires_grad"), True),
    (("intervention", "freeze_parameters"), []),
    (("intervention", "cell_policy"), "use learned solver output"),
    (("data", "test_read"), True), (("data", "train_read"), False),
    (("data", "val_manifest"), "test.jsonl"), (("test_read",), True),
    (("limitations",), []), (("frozen_before_any_step",), False),
    (("scientific_claim",), True), (("version",), 2),
])
def test_rehashed_frozen_field_mutation_still_rejected(tmp_path, field, value):
    payload = _payload(tmp_path)
    _replace_path(payload, field, value)
    with pytest.raises(RuntimeError, match="registration"):
        protocol.verified_registration(_write_protocol(tmp_path, payload, rehash=True))


@pytest.mark.parametrize("mutation", ["missing-seed", "random-seed", "std", "shape", "grid",
                                     "thaw", "sha", "trainable", "total", "missing-arm"])
def test_rehashed_noncanonical_spec_or_unfrozen_arm_rejected(tmp_path, mutation):
    payload = _payload(tmp_path)
    if mutation == "missing-seed":
        del payload["intervention_specs"]["42"]
    elif mutation == "random-seed":
        payload["intervention_specs"]["41"]["random_seed"] += 1
    elif mutation == "std":
        payload["intervention_specs"]["41"]["std"] = .03
    elif mutation == "shape":
        payload["intervention_specs"]["41"]["shape"][2] = 193
    elif mutation == "grid":
        payload["intervention_specs"]["42"].update(token_hw=[5, 3])
    elif mutation == "thaw":
        payload["intervention_specs"]["41"]["freeze"] = False
    elif mutation == "sha":
        payload["intervention_specs"]["41"]["tensor_sha256"] = "not-sha"
    elif mutation == "trainable":
        payload["arms"][2]["trainable_parameters"] = payload["arms"][1]["trainable_parameters"]
    elif mutation == "total":
        payload["arms"][2]["parameters"] += 1
    else:
        payload["arms"].pop()
    with pytest.raises(RuntimeError):
        protocol.verified_registration(_write_protocol(tmp_path, payload, rehash=True))


@pytest.mark.parametrize("authorization", [None, {}, [], "", "   ", False])
def test_empty_authorization_rejected_not_fabricated(tmp_path, authorization):
    with pytest.raises(ValueError, match="authorization"):
        _payload(tmp_path, authorization=authorization)
    payload = _payload(tmp_path)
    payload["authorization"] = authorization
    with pytest.raises(RuntimeError, match="authorization"):
        protocol.verified_registration(_write_protocol(tmp_path, payload, rehash=True))


def test_records_are_detached_opaque_and_do_not_authenticate_humans(tmp_path):
    auth = {"record": "caller-supplied, unverified"}
    code = {"arbitrary_archive_format": ["zip", {"sha256": "opaque"}]}
    specs = {str(seed): _spec(seed) for seed in protocol.SEEDS}
    payload = _payload(tmp_path, authorization=auth, code_identity=code, intervention_specs=specs)
    expected = copy.deepcopy(payload)
    auth["record"] = "changed"
    code["arbitrary_archive_format"].clear()
    specs["41"]["token_hw"].clear()
    assert payload == expected
    assert "not proof of human authorization" in payload["authorization_policy"]
    assert payload["intervention"]["verification_status"].startswith("required driver checks")


@pytest.mark.parametrize("deltas,outcome,branch", [
    ((1.0, 2.0), "worsened", "exclude-learned-z-necessity"),
    ((-1.0, -2.0), "supported", "learned-z-carrier-candidate"),
    ((-100.0, 1.0), "unresolved", "cannot-distinguish"),
    ((0.0, 0.0), "unresolved", "cannot-distinguish"),
    ((0.0, 1.0), "unresolved", "cannot-distinguish"),
])
def test_primary_positive_negative_flipped_and_zero_deltas(deltas, outcome, branch):
    result = protocol.primary_reading(_pairs(primary=deltas))
    assert result["branch"] == branch
    assert result["stop_required"] is (outcome == "unresolved")
    assert result["next_node_proposal"] == ("N2d" if outcome == "unresolved" else None)
    assert list(result["primary_question"]["per_lead"]) == ["6", "12", "24", "48", "72"]
    for cell in result["primary_question"]["per_lead"].values():
        assert cell["outcome"] == outcome
        assert cell["seed_deltas"] == dict(zip(("41", "42"), deltas))
        assert cell["sign_consistent"] is (outcome != "unresolved")
        assert cell["delta_seed_mean"] == (sum(deltas) / 2 if outcome != "unresolved" else None)
    assert result["branch_reading"] == protocol.BRANCH_RULE[branch]["reading"]
    assert result["limitations"] and result["scientific_claim"] is False
    assert result["test_read"] is False and result["historical_pooling"] is False
    json.dumps(result, allow_nan=False)


def test_long_lead_mixed_is_not_a_carrier_verdict():
    pairs = _pairs()
    pairs[f"{protocol.FROZEN_Z_ARM} - {protocol.RW_A_ARM}"]["cells"]["72h|t2m"] = _cell((-1., -2.))
    result = protocol.primary_reading(pairs)
    assert result["branch"] == "mixed-long-leads"
    assert result["primary_question"]["long_lead_outcomes"] == {"48": "worsened", "72": "supported"}
    assert "mixed" in result["branch_reading"] and "不归为支持carrier" in result["branch_reading"]
    assert result["next_node_proposal"] is None


def test_three_pairs_report_separately_not_reselected_or_pooled():
    pairs = _pairs(primary=(1., 2.), reference=(-1., -2.), carrier=(1., -2.))
    pairs["historical - baseline"] = {"cells": {"48h|t2m": _cell((-99., -99.))}}
    result = protocol.primary_reading(pairs)
    assert result["branch"] == "exclude-learned-z-necessity"
    for key, role, outcome in zip(("primary_question", "round_reference", "paired_carrier_contrast"),
                                  protocol.PAIR_ROLES, ("worsened", "supported", "unresolved")):
        view = result[key]
        assert view["role"] == role and view["reading"] == view["pair"]
        assert view["per_lead"]["48"]["outcome"] == outcome
        assert view["limitations"] and view["test_read"] is False
    assert len(result["reporting_order"]) == 3


@pytest.mark.parametrize("missing", ["cell", "seed", "pair"])
def test_missing_long_lead_cell_or_seed_unresolved_never_smaller_seed_mean(missing):
    pairs = _pairs()
    key = f"{protocol.FROZEN_Z_ARM} - {protocol.RW_A_ARM}"
    if missing == "cell":
        del pairs[key]["cells"]["48h|t2m"]
    elif missing == "seed":
        del _primary_cell(pairs)["seed_deltas"]["42"]
    else:
        del pairs[key]
    result = protocol.primary_reading(pairs)
    cell = result["primary_question"]["per_lead"]["48"]
    assert cell["outcome"] == "unresolved" and cell["delta_seed_mean"] is None
    assert result["branch"] == "cannot-distinguish" and result["stop_required"] is True
    assert result["next_node_proposal"] == "N2d"


def test_reader_uses_comparator_flags_and_does_not_promote_an_unresolved_cell():
    pairs = _pairs()
    _primary_cell(pairs).update(sign_consistent=False, outcome="unresolved")
    result = protocol.primary_reading(pairs)
    assert result["primary_question"]["per_lead"]["48"]["delta_seed_mean"] is None
    assert result["branch"] == "cannot-distinguish"


def test_zero_even_with_incorrect_consistency_label_is_unresolved():
    pairs = _pairs(primary=(0., 0.))
    for cell in pairs[f"{protocol.FROZEN_Z_ARM} - {protocol.RW_A_ARM}"]["cells"].values():
        cell.update(sign_consistent=True, outcome="worsened")
    result = protocol.primary_reading(pairs)
    assert result["branch"] == "cannot-distinguish"
    assert all(cell["delta_seed_mean"] is None
               for cell in result["primary_question"]["per_lead"].values())


@pytest.mark.parametrize("field,value", [
    ("seed_deltas", {"41": 1., "42": float("nan")}),
    ("seed_deltas", {"41": float("inf"), "42": 2.}),
    ("seed_deltas", {"41": True, "42": 2.}),
    ("seed_deltas", {"41": "1.0", "42": 2.}),
    ("seed_deltas", {"41": 1., "42": 2., "43": 3.}),
    ("seed_deltas", {41: 1., 42: 2.}),
    ("unit", "normalized"), ("outcome", "improved"), ("sign_consistent", "true"),
])
def test_malformed_nonfinite_or_contradictory_primary_cells_rejected(field, value):
    pairs = _pairs()
    _primary_cell(pairs)[field] = value
    with pytest.raises(ValueError):
        protocol.primary_reading(pairs)


def test_short_lead_unresolved_is_preserved_not_a_new_long_lead_gate():
    pairs = _pairs()
    _primary_cell(pairs, 6).update(seed_deltas={"41": 1., "42": -1.},
                                 sign_consistent=False, outcome="unresolved")
    result = protocol.primary_reading(pairs)
    assert result["primary_question"]["per_lead"]["6"]["outcome"] == "unresolved"
    assert result["branch"] == "exclude-learned-z-necessity"


def test_test_manifest_guard_has_a_counterexample():
    protocol.refused_test_manifest(Path("val.jsonl"))
    with pytest.raises(ValueError, match="sealed"):
        protocol.refused_test_manifest(Path("anywhere/test.jsonl"))


def test_standalone_import_and_reader_require_no_torch_numpy_or_network_clients():
    code = """
import sys
class RefuseHeavyImports:
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split('.')[0] in {'torch', 'numpy', 'requests', 'fsspec'}
                or fullname in {'urllib.request', 'http.client'}):
            raise AssertionError('forbidden dependency: ' + fullname)
sys.meta_path.insert(0, RefuseHeavyImports())
from training.r7_frozen_z_protocol import primary_reading
result = primary_reading({})
assert result['branch'] == 'cannot-distinguish'
assert result['test_read'] is False and result['limitations']
"""
    completed = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                                    "CUDA_VISIBLE_DEVICES": ""},
                               text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
