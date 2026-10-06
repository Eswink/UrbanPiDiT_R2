"""Temporary synthetic evidence, never weather data or a scientific result."""
import copy
import csv
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import socket
import subprocess

import pytest

from training import r7_rollout_evidence as evidence

CHANNELS = "t2m u10 v10 mslp z850 t850 q850 u850 v850 z500 t500 q500 u500 v500 z250 u250 v250".split()
UNITS = ["K", "m s**-1", "m s**-1", "Pa", "m**2 s**-2", "K", "kg kg**-1", "m s**-1",
         "m s**-1", "m**2 s**-2", "K", "kg kg**-1", "m s**-1", "m s**-1", "m**2 s**-2",
         "m s**-1", "m s**-1"]


@pytest.fixture(autouse=True)
def test_forbid_network_children(monkeypatch):
    def test_forbidden(*args, **kwargs):
        raise AssertionError("evidence must not open network or spawn children")
    monkeypatch.setattr(socket.socket, "connect", test_forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", test_forbidden)
    monkeypatch.setattr(subprocess, "Popen", test_forbidden)


def test_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")


def test_refresh_record(record):
    for filename, key in (("rmse.csv", "rmse_csv_sha256"), ("climatology_skill.csv", "skill_csv_sha256"),
                          ("provenance.json", "provenance_sha256")):
        record[key] = test_sha(Path(record["dir"]) / filename)


def test_make_evaluation(folder, protocol, checkpoint_sha, lead, value):
    folder.mkdir(parents=True)
    count = evidence.screen.V3_VAL_COHORTS[str(lead)]
    cases = []
    for index in range(count):
        initial = datetime(2022, 1, 1, 6) + timedelta(hours=6 * index)
        cases.append({"init_time": initial.isoformat(), "valid_times": [(initial + timedelta(hours=lead)).isoformat()],
                      "mse": [[value * value] * 17], "cumulative_reasoning_steps": [4 * lead // 6]})
    provenance = {"scientific_claim": False, "checkpoint_sha256": checkpoint_sha,
        "training_identity": protocol["train_data_identity"],
        "evaluation_manifest_sha256": test_sha(Path(protocol["val_manifest"])),
        "source_declaration": str(Path(protocol["instance"]) / "source.nc"),
        "channels": CHANNELS, "units": UNITS, "split": "val", "lead_hours": [lead], "step_hours": 6,
        "n_evaluated": count, "n_available_windows": count, "elapsed_seconds": 1.0,
        "inference_options": {"reasoning_steps": 4}, "initializations": cases,
        "selection": "first N in chronological order; explicit cap",
        "climatology": {"kind": "train-only-month-hour-grid-mean-v1", "training_years": [2017, 2018, 2019, 2020, 2021],
                        "baseline_table": "climatology_skill.csv", "selection": "declared_train_years"}}
    test_write_json(folder / "provenance.json", provenance)
    with (folder / "rmse.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lead_hours", "variable", "rmse", "unit", "n_initializations"])
        writer.writerows((lead, name, value, unit, count) for name, unit in zip(CHANNELS, UNITS))
    with (folder / "climatology_skill.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lead_hours", "variable", "rmse_forecast", "rmse_climatology", "mse_skill", "unit", "n_initializations"])
        writer.writerows((lead, name, value, 3.0, 1 - (value / 3.0) ** 2, unit, count)
                         for name, unit in zip(CHANNELS, UNITS))
    record = {"dir": str(folder), "n_evaluated": count, "elapsed_seconds": 1.0}
    test_refresh_record(record)
    return record


def test_training_report(protocol, receipt, seed):
    windows = copy.deepcopy(evidence.screen.V3_TRAIN_WINDOWS)
    parent = protocol["arms"]["parent"]["pins"]["checkpoints"][str(seed)]
    initialization = {"parent_checkpoint": parent["path"], "parent_checkpoint_sha256": parent["sha256"],
                      "parent_endpoint_updates": evidence.recipe.PARENT_ENDPOINT_UPDATES}
    contract = {"kind": "process", "model": protocol["arms"]["candidate"]["model"]["spec"],
        "data_identity": protocol["train_data_identity"], "protocol_sha256": protocol["protocol_sha256"],
        "source_sha256": protocol["source_sha256"], "model_code_sha256": protocol["model_code_sha256"],
        "training_code_sha256": protocol["training_code_sha256"], "model_semantics_sha256": "a" * 64,
        "initial_weights_sha256": receipt["imported_initial_weights_sha256"], "scientific_claim": False,
        "seed": seed, "output_dir": receipt["training_dir"], "initialization": initialization,
        "autoregression": {"windows": windows, "mode": "two_step", "lambda12": 0.5, "physical_steps": 2,
            "step_hours": 6, "detach_physical_steps": False, "detach_reasoning_steps": False,
            "internal_deep_supervision": True}}
    for key, value in evidence._options().items():
        if key != "lambda12":
            contract[{"updates": "total_updates", "warmup": "warmup_updates"}.get(key, key)] = value
    return {"scientific_claim": False, "test_read": False, "contract": contract,
        "signature": evidence._digest(contract), "data_identity": protocol["train_data_identity"],
        "protocol_sha256": protocol["protocol_sha256"], "source_sha256": protocol["source_sha256"],
        "model_code_sha256": protocol["model_code_sha256"], "selected_checkpoint": receipt["checkpoint"],
        "selected_update": 200, "updates_this_run": 200, "total_updates": 200, "resumed_from_updates": 0,
        "selection_split": None, "parent_optimizer_imported": False, "initialization": initialization,
        "windows": windows, "losses": [{"update": i, "loss": 0.2, "l6": 0.1, "l12": 0.2} for i in range(1, 201)]}


def test_freeze(output, protocol):
    protocol["protocol_sha256"] = evidence._digest({k: v for k, v in protocol.items() if k != "protocol_sha256"})
    test_write_json(output / "protocol.json", protocol)
    for seed in evidence.recipe.SEEDS:
        path = output / f"seed{seed}_receipt.json"
        receipt = json.loads(path.read_text())
        report_path = Path(receipt["training_dir"]) / "training_report.json"
        test_write_json(report_path, test_training_report(protocol, receipt, seed))
        receipt["training_report_sha256"] = test_sha(report_path)
        test_write_json(path, receipt)


@pytest.fixture
def test_evidence_bundle(tmp_path):
    output = tmp_path / "isolated_attempt02"
    output.mkdir()
    manifest = tmp_path / "instance/store/manifests/val.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("fixture val metadata\n")
    spec = {"in_channels": 17, "out_channels": 17, "detach_between_steps": False}
    protocol = {"scientific_claim": False, "test_read": False, "instance": str(tmp_path / "instance"),
        "train_manifest": str(manifest.with_name("train.jsonl")), "val_manifest": str(manifest),
        "source_sha256": evidence.screen.V3_SOURCE_SHA256, "train_data_identity": evidence.screen.V3_DATA_IDENTITY,
        "val_data_identity": "b" * 64, "model_code_sha256": "c" * 64, "training_code_sha256": "d" * 64,
        "arms": {"candidate": {**evidence._options(), "model": {"kind": "process", "spec": spec,
                                                            "spec_canonical_digest": evidence._digest(spec)}}},
        "decision": {"primary": {"variable": "t2m", "region": "full", "leads_hours": [6, 12]},
                     "gate_pre_screen": {"variables": ["u10", "v10", "mslp"], "leads_hours": [6, 12, 24, 48, 72],
                                         "relative_mse_change_max": 0.0}}}
    for name, prefix, arm, updates, value in (("control", "d3", "process", 400, 2.0),
                                             ("parent", "parent", "candidate", 1600, 1.5)):
        root = tmp_path / name
        result = {"scientific_claim": False, "test_read": False, "seeds": {},
                  "train_data_identity": protocol["train_data_identity"], "val_data_identity": protocol["val_data_identity"],
                  "gate_pre_screen": {"failures": list(range(17))}}
        pins = {f"{name}_updates": updates, "seeds": {}} if name == "control" else {
            "parent_updates": updates, "parent_gate_failures_vs_control": 17, "checkpoints": {}, "evaluations": {}}
        for seed in evidence.recipe.SEEDS:
            checkpoint = root / f"seed{seed}/training/{arm}/update_{updates:07d}.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(f"fixture {name} checkpoint seed{seed}".encode())
            receipt = {"seed": seed, "checkpoint": str(checkpoint), "checkpoint_sha256": test_sha(checkpoint),
                       "initial_state_sha256": "e" * 64, "evaluations": {}}
            for lead in evidence.recipe.EVALUATION_LEADS:
                receipt["evaluations"][str(lead)] = test_make_evaluation(
                    root / f"seed{seed}/evaluation/{arm}/lead_{lead:03d}h", protocol, receipt["checkpoint_sha256"], lead, value)
            result["seeds"][str(seed)] = receipt
            if name == "control":
                pins["seeds"][str(seed)] = {"initial_state_sha256": receipt["initial_state_sha256"], "leads": {
                    lead: {"evaluation_dir": r["dir"], "rmse_csv_sha256": r["rmse_csv_sha256"]}
                    for lead, r in receipt["evaluations"].items()}}
            else:
                pins["checkpoints"][str(seed)] = {"path": str(checkpoint), "sha256": test_sha(checkpoint)}
                pins["evaluations"][str(seed)] = copy.deepcopy(receipt["evaluations"])
        test_write_json(root / "result.json", result)
        pins.update({f"{prefix}_result_path": str(root / "result.json"), f"{prefix}_result_sha256": test_sha(root / "result.json")})
        protocol["arms"][name] = {"updates": updates, "pins": pins}
    for seed in evidence.recipe.SEEDS:
        checkpoint = output / f"seed{seed}/training/candidate/update_0000200.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"fixture candidate checkpoint seed{seed}".encode())
        receipt = {"seed": seed, "training_dir": str(checkpoint.parent), "checkpoint": str(checkpoint),
            "checkpoint_sha256": test_sha(checkpoint), "imported_initial_weights_sha256": "f" * 64,
            "parent_checkpoint_sha256": protocol["arms"]["parent"]["pins"]["checkpoints"][str(seed)]["sha256"], "evaluations": {}}
        for lead in evidence.recipe.EVALUATION_LEADS:
            receipt["evaluations"][str(lead)] = test_make_evaluation(output / f"seed{seed}/evaluation/candidate/lead_{lead:03d}h",
                protocol, receipt["checkpoint_sha256"], lead, 1.0)
        test_write_json(output / f"seed{seed}_receipt.json", receipt)
    test_freeze(output, protocol)
    return output, protocol


def test_candidate(output, seed=41):
    path = output / f"seed{seed}_receipt.json"
    return path, json.loads(path.read_text())


def test_mutate_provenance(output, field, value):
    path, receipt = test_candidate(output)
    record = receipt["evaluations"]["6"]
    provenance_path = Path(record["dir"]) / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance[field] = value
    test_write_json(provenance_path, provenance)
    test_refresh_record(record)
    test_write_json(path, receipt)


def test_collect_reads_all_seeds_leads_without_writes(test_evidence_bundle):
    output, protocol = test_evidence_bundle
    before = {p: test_sha(p) for p in output.parent.rglob("*") if p.is_file()}
    result = evidence.collect_readings(output, protocol)
    assert set(result) == {"paired_cells", "parent_relative_cells", "primary_verdict", "gate_pre_screen",
                           "parent_relative_gate", "decision", "rollout_response"}
    assert set(result["paired_cells"]) == {"41", "42", "43"}
    assert all(len(cells) == 85 for cells in result["paired_cells"].values())
    assert result["primary_verdict"]["overall"] == "supported"
    assert result["gate_pre_screen"]["passed"] is True
    assert result["decision"] == "advance-to-S4-freeze"
    assert result["rollout_response"]["parent_gate_failures_vs_control"] == 17
    assert {p: test_sha(p) for p in output.parent.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("failure", ["missing_seed", "extra_seed", "wrong_seed", "missing_lead", "extra_lead"])
def test_reject_incomplete_or_wrong_inventory(test_evidence_bundle, failure):
    output, protocol = test_evidence_bundle
    path, receipt = test_candidate(output)
    if failure == "missing_seed":
        (output / "seed43_receipt.json").unlink()
    elif failure == "extra_seed":
        test_write_json(output / "seed44_receipt.json", receipt)
    else:
        if failure == "wrong_seed":
            receipt["seed"] = 42
        elif failure == "missing_lead":
            receipt["evaluations"].pop("72")
        else:
            receipt["evaluations"]["18"] = receipt["evaluations"]["6"]
        test_write_json(path, receipt)
    with pytest.raises(ValueError, match="seed|lead"):
        evidence.collect_readings(output, protocol)


@pytest.mark.parametrize("artifact", ["checkpoint", "training_report", "rmse", "skill", "provenance",
                                      "control_result", "parent_result", "parent_checkpoint", "control_csv", "parent_csv"])
def test_reject_hash_drift(test_evidence_bundle, artifact):
    output, protocol = test_evidence_bundle
    _, receipt = test_candidate(output)
    paths = {"checkpoint": Path(receipt["checkpoint"]), "training_report": Path(receipt["training_dir"]) / "training_report.json",
             "rmse": Path(receipt["evaluations"]["6"]["dir"]) / "rmse.csv",
             "skill": Path(receipt["evaluations"]["6"]["dir"]) / "climatology_skill.csv",
             "provenance": Path(receipt["evaluations"]["6"]["dir"]) / "provenance.json"}
    for name, prefix in (("control", "d3"), ("parent", "parent")):
        paths[f"{name}_result"] = Path(protocol["arms"][name]["pins"][f"{prefix}_result_path"])
        arm = "process" if name == "control" else "candidate"
        paths[f"{name}_csv"] = paths[f"{name}_result"].parent / f"seed41/evaluation/{arm}/lead_006h/rmse.csv"
    paths["parent_checkpoint"] = Path(protocol["arms"]["parent"]["pins"]["checkpoints"]["41"]["path"])
    with paths[artifact].open("ab") as handle:
        handle.write(b" drift")
    with pytest.raises(ValueError, match="SHA256"):
        evidence.collect_readings(output, protocol)


@pytest.mark.parametrize("field,value,match", [("n_evaluated", 471, "cohort|n_evaluated"),
    ("n_available_windows", 471, "n_available_windows"), ("split", "test", "split"),
    ("checkpoint_sha256", "0" * 64, "checkpoint_sha256"), ("training_identity", "0" * 64, "training_identity"),
    ("evaluation_manifest_sha256", "0" * 64, "manifest_sha256"), ("lead_hours", [12], "lead_hours"),
    ("test_read", True, "test_read")])
def test_reject_wrong_provenance(test_evidence_bundle, field, value, match):
    output, protocol = test_evidence_bundle
    test_mutate_provenance(output, field, value)
    with pytest.raises(ValueError, match=match):
        evidence.collect_readings(output, protocol)


@pytest.mark.parametrize("case_error", ["shift", "valid_time", "year", "duplicate", "order", "negative_mse"])
def test_reject_case_alignment_or_test_read(test_evidence_bundle, case_error):
    output, protocol = test_evidence_bundle
    _, receipt = test_candidate(output)
    p = Path(receipt["evaluations"]["6"]["dir"]) / "provenance.json"
    cases = json.loads(p.read_text())["initializations"]
    if case_error in ("shift", "year"):
        delta = timedelta(minutes=1) if case_error == "shift" else timedelta(days=365)
        for case in cases:
            case["init_time"] = (datetime.fromisoformat(case["init_time"]) + delta).isoformat()
            case["valid_times"] = [(datetime.fromisoformat(case["valid_times"][0]) + delta).isoformat()]
    elif case_error == "valid_time":
        cases[0]["valid_times"] = [cases[0]["init_time"]]
    elif case_error == "duplicate":
        cases[1] = copy.deepcopy(cases[0])
    elif case_error == "order":
        cases[0], cases[1] = cases[1], cases[0]
    else:
        cases[0]["mse"][0][0] = -1.0
    test_mutate_provenance(output, "initializations", cases)
    with pytest.raises(ValueError, match="cases|timestamp|initialization|nonnegative"):
        evidence.collect_readings(output, protocol)


@pytest.mark.parametrize("mismatch", ["unit", "order", "variable", "climatology"])
def test_reject_units_variable_order_and_climatology(test_evidence_bundle, mismatch):
    output, protocol = test_evidence_bundle
    if mismatch == "climatology":
        test_mutate_provenance(output, "climatology", {"training_years": [2017, 2018, 2019, 2020, 2021, 2022]})
    elif mismatch == "unit":
        test_mutate_provenance(output, "units", ["degC", *UNITS[1:]])
        receipt_path, receipt = test_candidate(output)
        record = receipt["evaluations"]["6"]
        for filename in ("rmse.csv", "climatology_skill.csv"):
            path = Path(record["dir"]) / filename
            path.write_text(path.read_text().replace(",K,", ",degC,", 1))
        test_refresh_record(record)
        test_write_json(receipt_path, receipt)
    elif mismatch == "order":
        test_mutate_provenance(output, "channels", [CHANNELS[1], CHANNELS[0], *CHANNELS[2:]])
    else:
        test_mutate_provenance(output, "channels", ["not_t2m", *CHANNELS[1:]])
    with pytest.raises(ValueError, match="unit|variables|climatology"):
        evidence.collect_readings(output, protocol)


@pytest.mark.parametrize("field", ["selected_update", "updates_this_run", "data_identity", "signature", "model_code_sha256",
                                   "source_sha256", "protocol_sha256", "parent_optimizer_imported", "imported_weights", "contract_root"])
def test_reject_training_identity_or_endpoint(test_evidence_bundle, field):
    output, protocol = test_evidence_bundle
    receipt_path, receipt = test_candidate(output)
    p = Path(receipt["training_dir"]) / "training_report.json"
    report = json.loads(p.read_text())
    if field == "imported_weights":
        receipt["imported_initial_weights_sha256"] = "0" * 64
    elif field == "contract_root":
        report["contract"]["output_dir"] = str(output.parent / "unrelated")
        report["signature"] = evidence._digest(report["contract"])
    else:
        report[field] = 199 if field in ("selected_update", "updates_this_run") else (
            True if field == "parent_optimizer_imported" else "0" * 64)
    test_write_json(p, report)
    receipt["training_report_sha256"] = test_sha(p)
    test_write_json(receipt_path, receipt)
    with pytest.raises(ValueError):
        evidence.collect_readings(output, protocol)


@pytest.mark.parametrize("target", ["training", "evaluation", "receipt", "root"])
def test_reject_symlink_or_other_case_paths(test_evidence_bundle, target):
    output, protocol = test_evidence_bundle
    path, receipt = test_candidate(output)
    if target == "root":
        link = output.parent / "linked_attempt"
        link.symlink_to(output, target_is_directory=True)
        with pytest.raises(ValueError, match="symlink"):
            evidence.collect_readings(link, protocol)
        return
    if target == "receipt":
        relocated = output.parent / "seed41_saved.json"
        path.rename(relocated)
        path.symlink_to(relocated)
    else:
        receipt["training_dir" if target == "training" else "checkpoint"] = str(output / "seed42/training/candidate")
        if target == "evaluation":
            receipt["checkpoint"] = str(output / "seed41/training/candidate/update_0000200.pt")
            receipt["evaluations"]["6"]["dir"] = str(output / "seed42/evaluation/candidate/lead_006h")
        test_write_json(path, receipt)
    with pytest.raises(ValueError, match="symlink|exact evidence path"):
        evidence.collect_readings(output, protocol)


@pytest.mark.parametrize("bad_value", ["nan", "inf", "-1", "0", "1e200"])
def test_reject_nonfinite_negative_or_zero_reference_rmse(tmp_path, bad_value):
    protocol = {"train_data_identity": "a" * 64, "instance": str(tmp_path), "val_manifest": str(tmp_path / "val.jsonl")}
    Path(protocol["val_manifest"]).write_text("fixture")
    folder = tmp_path / "lead_006h"
    record = test_make_evaluation(folder, protocol, "b" * 64, 6, 0.0)
    path = folder / "rmse.csv"
    path.write_text(path.read_text().replace("6,t2m,0.0,", f"6,t2m,{bad_value},"))
    test_refresh_record(record)
    with pytest.raises(ValueError, match="finite|positive|nonnegative"):
        evidence.check_evaluation_receipt(record, expected_dir=folder, checkpoint_sha256="b" * 64,
            protocol=protocol, lead=6, manifest_sha256=test_sha(Path(protocol["val_manifest"])), denominator=True)


def test_shared_verdict_never_averages_disagreeing_seeds(test_evidence_bundle):
    output, protocol = test_evidence_bundle
    receipt_path, receipt = test_candidate(output, 43)
    for lead in (6, 12):
        record = receipt["evaluations"][str(lead)]
        folder = Path(record["dir"])
        p = folder / "provenance.json"
        provenance = json.loads(p.read_text())
        for case in provenance["initializations"]:
            case["mse"][0][0] = 2.1 ** 2
        test_write_json(p, provenance)
        p = folder / "rmse.csv"
        p.write_text(p.read_text().replace(f"{lead},t2m,1.0,", f"{lead},t2m,2.1,"))
        p = folder / "climatology_skill.csv"
        p.write_text(p.read_text().replace(f"{lead},t2m,1.0,3.0,{1 - (1 / 3) ** 2}",
                                         f"{lead},t2m,2.1,3.0,{1 - (2.1 / 3) ** 2}"))
        test_refresh_record(record)
    test_write_json(receipt_path, receipt)
    result = evidence.collect_readings(output, protocol)
    assert result["primary_verdict"]["overall"] == "unresolved"
    assert result["gate_pre_screen"]["passed"] is True
    assert result["decision"] == "registered-negative"


@pytest.mark.parametrize("reference,key", [("control", "rmse_csv_sha256"), ("parent", "provenance_sha256")])
def test_reject_reference_pin_disagreement(test_evidence_bundle, reference, key):
    output, protocol = test_evidence_bundle
    pins = protocol["arms"][reference]["pins"]
    record = pins["seeds"]["41"]["leads"]["6"] if reference == "control" else pins["evaluations"]["41"]["6"]
    record[key] = "0" * 64
    test_freeze(output, protocol)
    with pytest.raises(ValueError, match="pins"):
        evidence.collect_readings(output, protocol)


@pytest.mark.parametrize("key", ["seed", "training_report_sha256", "checkpoint_sha256", "evaluations"])
def test_receipt_required_keys_fail_closed(test_evidence_bundle, key):
    output, protocol = test_evidence_bundle
    path, receipt = test_candidate(output)
    receipt.pop(key)
    test_write_json(path, receipt)
    with pytest.raises(KeyError, match=key):
        evidence.collect_readings(output, protocol)


def test_protocol_and_decision_cannot_drift(test_evidence_bundle):
    output, protocol = test_evidence_bundle
    protocol["decision"]["gate_pre_screen"]["relative_mse_change_max"] = 1.0
    test_freeze(output, protocol)
    with pytest.raises(ValueError, match="relative_mse_change_max"):
        evidence.collect_readings(output, protocol)
    protocol["decision"]["gate_pre_screen"]["relative_mse_change_max"] = 0.0
    with pytest.raises(ValueError, match="root protocol"):
        evidence.collect_readings(output, protocol)


# Helper names follow the repository test namespace but are not standalone cases.
for TEST_HELPER in (test_sha, test_write_json, test_refresh_record, test_make_evaluation,
                    test_training_report, test_freeze, test_candidate, test_mutate_provenance):
    TEST_HELPER.__test__ = False
