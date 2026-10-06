"""CPU synthetic structural evidence only; no real weather/checkpoint reads.

The fake-byte checkpoints come from the old fixture. Only this collector's
checkpoint loader and training-code digest are replaced; actual tiny CPU model
states and semantics exercise strict structural restoration unchanged. State
faults corrupt the loader's returned structure, not the disclosed fake bytes.
Ordinary serialized checkpoint loading is covered by the runner tests.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta
import json
from pathlib import Path

import pytest
import torch

import test_r7_m4_autoregressive_rollout as autoregressive
import test_r7_rollout_evidence as old
from training import r7_long_rollout_evidence as evidence
from training.r7_autoregressive_runner import _module_semantics
from training.r7_experiment import make_model

WEIGHTS = [1., .5, 0., .5, 0., 0., 0., .5, 0., 0., 0., .5]


def _windows(tmp_path):
    pins = []
    for index in range(3):
        initial = datetime(2017, 1, 1, 6) + timedelta(hours=6 * index)
        pins.append({"sample_id": f"synthetic-train-{index}", "manifest_index": index,
            "store_path": str(tmp_path / "synthetic-not-a-weather-store"),
            "init_time": initial.isoformat(), "history_indices": [index, index + 1],
            "history_times": [(initial - timedelta(hours=6)).isoformat(), initial.isoformat()],
            "target_indices": list(range(index + 2, index + 14)),
            "target_times": [(initial + timedelta(hours=6 * step)).isoformat() for step in range(1, 13)]})
    exclusions = [{"sample_id": "synthetic-boundary", "reason": "missing_exact_t72"}]
    identity = {"physical_steps": 12, "step_hours": 6, "windows": pins, "exclusions": exclusions}
    return {**identity, "input_windows": 4, "usable_windows": 3,
        "excluded_sample_ids": ["synthetic-boundary"], "window_sha256": evidence.canonical_digest(identity),
        "selection": "exact t-6,t,t+6..t+6N, every frame owned by train; no index adjacency",
        "test_read": False, "state_fields_read": False}


def _report(bundle, seed=41):
    _, receipt = old.test_candidate(bundle["output"], seed)
    return json.loads((Path(receipt["training_dir"]) / "training_report.json").read_text())


def _store_report(bundle, report, seed=41, *, sync_checkpoint=True, nonfinite=False):
    receipt_path, receipt = old.test_candidate(bundle["output"], seed)
    report["signature"] = evidence.canonical_digest(report["contract"])
    path = Path(receipt["training_dir"]) / "training_report.json"
    if nonfinite:
        path.write_text(json.dumps(report, sort_keys=True, allow_nan=True), encoding="utf-8")
    else:
        old.test_write_json(path, report)
    receipt["training_report_sha256"] = old.test_sha(path)
    old.test_write_json(receipt_path, receipt)
    if sync_checkpoint:
        saved = bundle["checkpoints"].setdefault(receipt["checkpoint"], {
            "format": "r7-local-v1", "updates": 200, "model": copy.deepcopy(bundle["model_state"])})
        saved.update({"contract": copy.deepcopy(report["contract"]), "signature": report["signature"],
                      "model_code_sha256": report["model_code_sha256"]})


def _long_report(bundle, receipt, seed):
    protocol = bundle["protocol"]
    report = old.test_training_report(protocol, receipt, seed)
    contract = report["contract"]
    contract.update({"mode": "long_rollout", "optimization": "full-bptt", "test_read": False,
        "device_type": "cpu", "process_weight": 0., "model_semantics_sha256": bundle["model_semantics_sha256"],
        "limitations": ["Synthetic structural fixture, not weather or scientific evidence"]})
    windows = copy.deepcopy(protocol["train_windows"])
    contract["autoregression"] = {"mode": "long_rollout", "physical_steps": 12,
        "physical_weights": list(WEIGHTS), "step_hours": 6, "fixed_transition_lead_hours": 6,
        "detach_physical_steps": False, "detach_reasoning_steps": False, "internal_deep_supervision": True,
        "objective": "deep_supervised_latitude_area_mse", "loss_space": "normalized", "final_weight": 2.,
        "k_weights": "linspace(1,2,K+1) normalized; initial plus every K draft",
        "physical_loss": "sum(weight_i * L_6i), no physical-axis normalization",
        "window_sha256": windows["window_sha256"], "windows": windows,
        "excluded_sample_ids": windows["excluded_sample_ids"]}
    per_step = [.1 * step for step in range(1, 13)]
    total = sum(weight * loss for weight, loss in zip(WEIGHTS, per_step))
    report.update({"windows": windows, "optimization": "full-bptt", "internal_k_detach": False,
        "physical_step_detach": False, "internal_deep_supervision": True, "bf16": False,
        "objective": "deep_supervised_latitude_area_mse", "device_type": "cpu",
        "limitations": ["Synthetic structural fixture, not weather or scientific evidence"],
        "losses": [{"update": update, "loss": total, "l6": per_step[0], "l12": per_step[1],
                    "per_step_losses": list(per_step)} for update in range(1, 201)]})
    return report


def _freeze(bundle):
    output, protocol = bundle["output"], bundle["protocol"]
    protocol["protocol_sha256"] = evidence.canonical_digest(
        {key: value for key, value in protocol.items() if key != "protocol_sha256"})
    old.test_write_json(output / "protocol.json", protocol)
    for seed in evidence.recipe.SEEDS:
        path, receipt = old.test_candidate(output, seed)
        receipt["imported_initial_weights_sha256"] = evidence._state_digest(bundle["parents"][seed]["model"])
        old.test_write_json(path, receipt)
        _store_report(bundle, _long_report(bundle, receipt, seed), seed)


def _collect(bundle):
    return evidence.collect_long_readings(bundle["output"], bundle["protocol"])


def _accepted(bundle):
    try:
        _collect(bundle)
    except (ValueError, KeyError):
        return False
    return True


def _evaluation(bundle):
    path, receipt = old.test_candidate(bundle["output"])
    record = receipt["evaluations"]["6"]
    folder = Path(record["dir"])
    return path, receipt, record, folder, json.loads((folder / "provenance.json").read_text())


def _store_evaluation(path, receipt, record, folder, provenance):
    old.test_write_json(folder / "provenance.json", provenance)
    old.test_refresh_record(record)
    old.test_write_json(path, receipt)


@pytest.fixture(autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def _bundle(tmp_path, monkeypatch):
    output, protocol = old.test_evidence_bundle.__wrapped__(tmp_path)
    protocol.update({"format": "r7-s3-long-rollout-protocol-v1", "kind": "screen",
        "synthetic_fixture": True, "limitations": ["Declared fake-byte, nonweather fixture"],
        "seeds": list(evidence.recipe.SEEDS), "evaluation_leads_hours": list(evidence.recipe.EVALUATION_LEADS),
        "train_windows": _windows(tmp_path)})
    candidate = protocol["arms"]["candidate"]
    candidate.pop("lambda12")
    spec = autoregressive.tiny_spec("process")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(41)
        template = make_model("process", spec)
    model_state = {name: value.detach().clone() for name, value in template.state_dict().items()}
    semantics_sha256 = evidence.canonical_digest(_module_semantics(template))
    candidate.update({"mode": "long_rollout", "physical_weights": list(WEIGHTS),
        "model": {"kind": "process", "spec": spec, "spec_canonical_digest": evidence.canonical_digest(spec)}})
    protocol["decision"]["primary"]["text"] = evidence.recipe.PRIMARY_DECISION_TEXT
    protocol["decision"]["gate_pre_screen"]["text"] = evidence.recipe.GATE_DECISION_TEXT
    parents = {}
    for seed in evidence.recipe.SEEDS:
        contract = {"kind": "process", "seed": seed, "mode": "l6",
            "data_identity": protocol["train_data_identity"], "source_sha256": protocol["source_sha256"],
            "model_code_sha256": protocol["model_code_sha256"], "model": copy.deepcopy(candidate["model"]["spec"])}
        contract.update({"scientific_claim": False, "test_read": False,
            "limitations": ["Synthetic parent state only"], "model_semantics_sha256": semantics_sha256})
        parents[seed] = {"format": "r7-local-v1", "updates": 1600, "contract": contract,
            "signature": evidence.canonical_digest(contract), "model_code_sha256": protocol["model_code_sha256"],
            "model": copy.deepcopy(model_state)}
    bundle = {"output": output, "protocol": protocol, "parents": parents, "checkpoints": {}, "loads": [],
              "model_state": model_state, "model_semantics_sha256": semantics_sha256}
    _freeze(bundle)
    parent_seeds = {pin["path"]: int(seed) for seed, pin in protocol["arms"]["parent"]["pins"]["checkpoints"].items()}

    def _load_checkpoint(path, *, expected=None):
        path = str(Path(path))
        bundle["loads"].append((path, expected))
        if path in parent_seeds:
            assert expected is None
            return copy.deepcopy(parents[parent_seeds[path]])
        assert path in bundle["checkpoints"], "only declared synthetic checkpoints may be loaded"
        report = json.loads((Path(path).parent / "training_report.json").read_text())
        assert expected == report["signature"] == evidence.canonical_digest(report["contract"])
        return copy.deepcopy(bundle["checkpoints"][path])

    monkeypatch.setattr(evidence, "load_checkpoint", _load_checkpoint)
    monkeypatch.setattr(evidence, "training_code_digest", lambda: "d" * 64)
    return bundle


def test_collects_exact_paired_inventory_without_writes(_bundle):
    output, protocol = _bundle["output"], _bundle["protocol"]
    before = {path: old.test_sha(path) for path in output.parent.rglob("*") if path.is_file()}
    result = _collect(_bundle)
    assert set(result) == {"paired_cells", "parent_relative_cells", "primary_verdict", "gate_pre_screen",
                           "parent_relative_gate", "decision"}
    expected_cells = {f"{lead}|{name}" for lead in evidence.recipe.EVALUATION_LEADS for name in old.CHANNELS}
    assert set(result["paired_cells"]) == set(result["parent_relative_cells"]) == {"41", "42", "43"}
    for seed in evidence.recipe.SEEDS:
        cells, parents = result["paired_cells"][str(seed)], result["parent_relative_cells"][str(seed)]
        assert len(cells) == len(parents) == 17 * 5
        assert set(cells) == set(parents) == expected_cells
        assert all(cell["candidate_rmse"] == 1. and cell["control_rmse"] == 2.
                   and cell["relative_mse_change"] == -.75 for cell in cells.values())
        assert all(cell["candidate_rmse"] == 1. and cell["control_rmse"] == 1.5 for cell in parents.values())
        report = _report(_bundle, seed)
        assert report["contract"]["initial_weights_sha256"] == evidence._state_digest(_bundle["parents"][seed]["model"])
        assert report["contract"]["autoregression"]["physical_weights"] == WEIGHTS
        assert report["contract"]["model_semantics_sha256"] == _bundle["model_semantics_sha256"]
        assert report["limitations"] and report["contract"]["limitations"]
        assert report["windows"] == protocol["train_windows"] and len(report["losses"]) == 200
    assert all(value.device.type == "cpu" and torch.isfinite(value).all()
               and (not value.is_floating_point() or value.dtype == torch.float32)
               for value in _bundle["model_state"].values())
    assert len(_bundle["loads"]) == 6 and sum(expected is not None for _, expected in _bundle["loads"]) == 3
    assert result["primary_verdict"]["overall"] == "supported"
    assert result["gate_pre_screen"]["passed"] is True and result["parent_relative_gate"]["passed"] is True
    assert result["decision"] == "advance-to-S4-freeze"
    assert {path: old.test_sha(path) for path in output.parent.rglob("*") if path.is_file()} == before


def test_refuses_damaged_protocol_digest(_bundle):
    _bundle["protocol"]["protocol_sha256"] = "0" * 64
    old.test_write_json(_bundle["output"] / "protocol.json", _bundle["protocol"])
    with pytest.raises(ValueError, match="protocol digest"):
        _collect(_bundle)


def test_refuses_missing_seed_and_lead(_bundle):
    path, receipt = old.test_candidate(_bundle["output"])
    original = path.read_bytes()
    path.unlink()
    with pytest.raises(ValueError, match="seed receipts"):
        _collect(_bundle)
    path.write_bytes(original)
    receipt["evaluations"].pop("72")
    old.test_write_json(path, receipt)
    with pytest.raises(ValueError, match="candidate leads"):
        _collect(_bundle)


def test_refuses_control_result_sha_drift(_bundle):
    pins = _bundle["protocol"]["arms"]["control"]["pins"]
    path = Path(pins["d3_result_path"])
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA256"):
        _collect(_bundle)


def test_refuses_actual_parent_state_import_mismatch(_bundle):
    state = _bundle["parents"][41]["model"]
    original = evidence._state_digest(state)
    tensor = next(value for value in state.values() if value.is_floating_point() and value.numel())
    tensor.reshape(-1)[0] += 1.
    assert evidence._state_digest(state) != original
    with pytest.raises(ValueError, match="imported parent state"):
        _collect(_bundle)


def test_refuses_checkpoint_report_contract_mismatch(_bundle):
    _, receipt = old.test_candidate(_bundle["output"])
    # Corrupt the loader's returned structure, not the report or its signature.
    _bundle["checkpoints"][receipt["checkpoint"]]["contract"]["seed"] = 42
    with pytest.raises(ValueError, match="checkpoint/report contract"):
        _collect(_bundle)


def test_refuses_invalid_candidate_state_structure_dtype_and_values(_bundle):
    _, receipt = old.test_candidate(_bundle["output"])
    saved = _bundle["checkpoints"][receipt["checkpoint"]]
    original = copy.deepcopy(saved["model"])
    key = next(name for name, value in original.items() if value.is_floating_point() and value.numel())
    for fault in ("empty", "missing_key", "extra_key", "wrong_shape", "bf16", "float64", "non_tensor", "nan", "inf"):
        state = copy.deepcopy(original)
        if fault == "empty":
            state.clear()
        elif fault == "missing_key":
            state.pop(key)
        elif fault == "extra_key":
            state["undeclared_parameter"] = torch.ones(1, device="cpu")
        elif fault == "wrong_shape":
            state[key] = torch.ones((state[key].numel() + 1,), dtype=state[key].dtype, device="cpu")
        elif fault in ("bf16", "float64"):
            state[key] = state[key].to(torch.bfloat16 if fault == "bf16" else torch.float64)
        elif fault == "non_tensor":
            state[key] = state[key].tolist()
        else:
            state[key].reshape(-1)[0] = float(fault)
        saved["model"] = state
        with pytest.raises(ValueError, match="candidate model state"):
            _collect(_bundle)
    saved["model"] = original


def test_refuses_signed_model_semantics_mismatch(_bundle):
    report = _report(_bundle)
    report["contract"]["model_semantics_sha256"] = "0" * 64
    _store_report(_bundle, report)
    with pytest.raises(ValueError, match="model semantics"):
        _collect(_bundle)


def test_refuses_missing_empty_or_invalid_training_limitations(_bundle):
    original = _report(_bundle)
    for target in ("report", "contract"):
        for fault in ("missing", [], [" "], [False], "not a list"):
            report = copy.deepcopy(original)
            value = report if target == "report" else report["contract"]
            if fault == "missing":
                value.pop("limitations")
            else:
                value["limitations"] = fault
            _store_report(_bundle, report)
            with pytest.raises(ValueError, match="nonempty limitations"):
                _collect(_bundle)
    _store_report(_bundle, original)


def test_refuses_non200_checkpoint_and_report_endpoint(_bundle):
    _, receipt = old.test_candidate(_bundle["output"])
    saved = _bundle["checkpoints"][receipt["checkpoint"]]
    saved["updates"] = 199
    with pytest.raises(ValueError, match="checkpoint endpoint"):
        _collect(_bundle)
    saved["updates"] = 200
    original = _report(_bundle)
    for field in ("selected_update", "updates_this_run", "total_updates"):
        report = copy.deepcopy(original)
        report[field] = 199
        _store_report(_bundle, report)
        with pytest.raises(ValueError, match=field):
            _collect(_bundle)


def test_refuses_changed_physical_weights_and_windows(_bundle):
    original = _report(_bundle)
    for field in ("physical_weights", "window_sha256", "windows", "report_windows"):
        report = copy.deepcopy(original)
        autoregression = report["contract"]["autoregression"]
        if field == "physical_weights":
            autoregression[field][3] = 0.
        elif field == "window_sha256":
            autoregression[field] = "0" * 64
        elif field == "windows":
            autoregression[field]["usable_windows"] -= 1
        else:
            report["windows"]["usable_windows"] -= 1
        _store_report(_bundle, report)
        with pytest.raises(ValueError, match="physical_weights|window|windows"):
            _collect(_bundle)


def test_refuses_training_source_model_and_spec_mismatch(_bundle):
    original = _report(_bundle)
    changes = [("source_sha256", "0" * 64), ("model_code_sha256", "0" * 64),
               ("training_code_sha256", "0" * 64), ("kind", "generic"),
               ("model", {**original["contract"]["model"], "in_channels": 18})]
    for field, value in changes:
        report = copy.deepcopy(original)
        report["contract"][field] = value
        if field == "kind":
            report["contract"]["model"] = autoregressive.tiny_spec("generic")
        _store_report(_bundle, report)
        with pytest.raises(ValueError, match="contract|candidate|model semantics"):
            _collect(_bundle)
    _store_report(_bundle, original)
    _bundle["protocol"]["arms"]["candidate"]["model"]["spec_canonical_digest"] = "0" * 64
    _freeze(_bundle)
    with pytest.raises(ValueError, match="spec digest"):
        _collect(_bundle)


def test_refuses_signed_contract_claim_and_test_read_flags(_bundle):
    original, accepted = _report(_bundle), []
    for field in ("scientific_claim", "test_read"):
        report = copy.deepcopy(original)
        report["contract"][field] = True
        _store_report(_bundle, report)
        if _accepted(_bundle):
            accepted.append(field)
    _store_report(_bundle, original)
    assert accepted == [], f"collector accepted forbidden signed contract flags: {accepted}"


def test_refuses_changed_report_full_bptt_flags(_bundle):
    original, accepted = _report(_bundle), []
    for field, value in (("optimization", "truncated-bptt"), ("internal_k_detach", True),
                         ("physical_step_detach", True), ("internal_deep_supervision", False),
                         ("scientific_claim", True), ("test_read", True)):
        report = copy.deepcopy(original)
        report[field] = value
        _store_report(_bundle, report)
        if _accepted(_bundle):
            accepted.append(field)
    _store_report(_bundle, original)
    assert accepted == [], f"collector accepted altered report flags: {accepted}"


def test_refuses_detach_or_transition_contract_flags(_bundle):
    original = _report(_bundle)
    for field, value in (("physical_steps", 11), ("detach_physical_steps", True),
                         ("detach_reasoning_steps", True), ("internal_deep_supervision", False),
                         ("fixed_transition_lead_hours", 12), ("mode", "two_step")):
        report = copy.deepcopy(original)
        report["contract"]["autoregression"][field] = value
        _store_report(_bundle, report)
        with pytest.raises(ValueError, match=f"autoregression {field}"):
            _collect(_bundle)


def test_refuses_missing_losses_and_incorrect_per_step_axis(_bundle):
    original = _report(_bundle)
    for fault in ("missing_losses", "missing_loss", "missing_per_step", "missing_update", "short_axis", "long_axis"):
        report = copy.deepcopy(original)
        row = report["losses"][0]
        if fault == "missing_losses":
            report.pop("losses")
        elif fault == "missing_loss":
            row.pop("loss")
        elif fault == "missing_per_step":
            row.pop("per_step_losses")
        elif fault == "missing_update":
            report["losses"].pop()
        elif fault == "short_axis":
            row["per_step_losses"].pop()
        else:
            row["per_step_losses"].append(.1)
        _store_report(_bundle, report)
        with pytest.raises((ValueError, KeyError), match="loss|physical"):
            _collect(_bundle)


def test_refuses_inconsistent_weighted_total_and_nan(_bundle):
    original = _report(_bundle)
    report = copy.deepcopy(original)
    report["losses"][0]["loss"] += .1
    _store_report(_bundle, report)
    with pytest.raises(ValueError, match="weighted objective"):
        _collect(_bundle)
    report = copy.deepcopy(original)
    report["losses"][0]["per_step_losses"][2] = float("nan")  # Even a zero-weight step must be finite.
    _store_report(_bundle, report, nonfinite=True)
    with pytest.raises(ValueError, match="nonfinite"):
        _collect(_bundle)


def test_refuses_unit_and_paired_case_mismatches(_bundle):
    path, receipt, record, folder, original = _evaluation(_bundle)
    csv_paths = [folder / name for name in ("rmse.csv", "climatology_skill.csv")]
    csv_originals = [p.read_bytes() for p in csv_paths]
    provenance = copy.deepcopy(original)
    provenance["units"][0] = "degC"
    for p in csv_paths:
        p.write_text(p.read_text().replace(",K,", ",degC,", 1), encoding="utf-8")
    _store_evaluation(path, receipt, record, folder, provenance)
    with pytest.raises(ValueError, match="paired.*metadata"):
        _collect(_bundle)
    for p, content in zip(csv_paths, csv_originals):
        p.write_bytes(content)
    provenance = copy.deepcopy(original)
    case = provenance["initializations"][0]
    case["init_time"] = (datetime.fromisoformat(case["init_time"]) + timedelta(minutes=1)).isoformat()
    case["valid_times"][0] = (datetime.fromisoformat(case["valid_times"][0]) + timedelta(minutes=1)).isoformat()
    _store_evaluation(path, receipt, record, folder, provenance)
    with pytest.raises(ValueError, match="paired.*metadata"):
        _collect(_bundle)


def test_refuses_incomplete_or_nonchronological_case_evidence(_bundle):
    path, receipt, record, folder, original = _evaluation(_bundle)
    for fault in ("missing_case", "chronology", "valid_time", "case_mse"):
        provenance = copy.deepcopy(original)
        cases = provenance["initializations"]
        if fault == "missing_case":
            cases.pop()
        elif fault == "chronology":
            cases[0], cases[1] = cases[1], cases[0]
        elif fault == "valid_time":
            cases[0]["valid_times"] = [cases[0]["init_time"]]
        else:
            cases[0]["mse"][0][0] = 4.
        _store_evaluation(path, receipt, record, folder, provenance)
        with pytest.raises(ValueError, match="cohort|chronological|timestamp|case MSE"):
            _collect(_bundle)


def test_refuses_wrong_candidate_paths(_bundle):
    path, original = old.test_candidate(_bundle["output"])
    for field in ("training_dir", "checkpoint", "evaluation"):
        receipt = copy.deepcopy(original)
        if field == "evaluation":
            receipt["evaluations"]["6"]["dir"] = str(_bundle["output"] / "seed42/evaluation/candidate/lead_006h")
        else:
            receipt[field] = receipt[field].replace("seed41", "seed42")
        old.test_write_json(path, receipt)
        with pytest.raises(ValueError, match="exact evidence path"):
            _collect(_bundle)


def test_refuses_candidate_symlinks(_bundle):
    output = _bundle["output"]
    linked = output.parent / "linked-attempt"
    linked.symlink_to(output, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        evidence.collect_long_readings(linked, _bundle["protocol"])
    _, receipt = old.test_candidate(output)
    path = Path(receipt["training_dir"]) / "training_report.json"
    saved = path.with_name("saved_report.json")
    path.rename(saved)
    path.symlink_to(saved)
    with pytest.raises(ValueError, match="symlink"):
        _collect(_bundle)


def test_refuses_csv_hash_drift_and_rehashed_schema_or_cohort(_bundle):
    path, receipt, record, folder, provenance = _evaluation(_bundle)
    csv_path = folder / "rmse.csv"
    original = csv_path.read_text()
    csv_path.write_text(original + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        _collect(_bundle)
    for text, match in ((original.replace("rmse,", "wrong_rmse,", 1), "CSV schema"),
                        (original.replace(",472", ",471", 1), "CSV cohort count")):
        csv_path.write_text(text, encoding="utf-8")
        _store_evaluation(path, receipt, record, folder, provenance)
        with pytest.raises(ValueError, match=match):
            _collect(_bundle)


def test_refuses_refrozen_primary_and_gate_lead_flags(_bundle):
    protocol = _bundle["protocol"]
    original, accepted = copy.deepcopy(protocol["decision"]), []
    for section, field, value in (("primary", "leads_hours", [12]), ("primary", "variable", "u10"),
                                  ("primary", "region", "boundary"), ("gate_pre_screen", "leads_hours", [6, 12]),
                                  ("gate_pre_screen", "variables", ["u10", "v10"])):
        protocol["decision"] = copy.deepcopy(original)
        protocol["decision"][section][field] = value
        _freeze(_bundle)  # Re-sign every report/receipt so this tests the decision guard, not stale hashes.
        if _accepted(_bundle):
            accepted.append(f"{section}.{field}")
    protocol["decision"] = original
    _freeze(_bundle)
    assert accepted == [], f"collector accepted changed frozen decision declarations: {accepted}"
