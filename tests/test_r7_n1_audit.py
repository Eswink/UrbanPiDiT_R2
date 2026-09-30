"""Synthetic table-only CPU counterproofs; every writable fixture is in tmp_path."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tools import recompute_r7_n1_audit as audit

SCRIPT = Path(audit.__file__)
METRIC_FIELDS = ("seed", "arm", "lead_hours", "variable", "unit", "rmse", "n_initializations")
CASE_FIELDS = ("seed", "arm", "lead_hours", "split", "n_available_windows", "n_evaluated", "test_read")
COUNTS = {6: 22, 12: 21, 24: 19, 48: 15, 72: 11}
OFFSETS = {audit.RW_A: (0, 0), audit.RW_B: (1, 3), audit.NO_RECURRENCE: (2, 1),
           "process_local_solver_no_gate_proposal": (0, 0)}


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")


def _seal(protocol):
    content = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    protocol["protocol_sha256"] = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_manifests(data):
    for split, rows in data["manifest_rows"].items():
        (data["manifests"] / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _publish(data):
    archive = data["archive"]
    _write_csv(archive / "rmse_table.csv", data["metrics"], METRIC_FIELDS)
    _write_csv(archive / "case_table.csv", data["cases"], CASE_FIELDS)
    _write_json(archive / "paired_comparison.json", data["paired"])
    _write_json(archive / "seed41/protocol.json", data["protocol"])
    _write_manifests(data)


def _synchronize_pairs(data):
    """Fixture arithmetic only, independent of the implementation's helpers."""
    table = []
    for arm in audit.ARMS:
        for lead in audit.LEADS:
            selected = [row for row in data["metrics"] if row["arm"] == arm and row["lead_hours"] == lead]
            values = {str(row["seed"]): row["rmse"] for row in selected}
            digest = hashlib.sha256(f"synthetic cases {lead}".encode()).hexdigest()
            table.append({"arm": arm, "depth": 0, "variable": "t2m", "unit": "K", "lead_hours": lead,
                          "seeds": [41, 42], "seed_rmse": values,
                          "rmse_seed_mean": (values["41"] + values["42"]) / 2,
                          "per_seed_n_initializations": {"41": COUNTS[lead], "42": COUNTS[lead]},
                          "case_set_digests": {"41": digest, "42": digest}, "case_identity": "exact"})
    pairs, primary = {}, {}
    for focus in audit.FOCUS_ARMS:
        cells, by_lead = {}, {}
        for lead in audit.LEADS:
            focus_row = next(row for row in table if row["arm"] == focus and row["lead_hours"] == lead)
            baseline = next(row for row in table if row["arm"] == audit.RW_A and row["lead_hours"] == lead)
            delta = {str(seed): focus_row["seed_rmse"][str(seed)] - baseline["seed_rmse"][str(seed)]
                     for seed in (41, 42)}
            # Intentionally not a real verdict: audit must never interpret these fields.
            cells[f"{lead}h|t2m"] = {"unit": "K", "seed_deltas": delta, "outcome": "do-not-interpret"}
            by_lead[str(lead)] = {"seed_deltas": dict(delta), "delta_seed_mean": sum(delta.values()) / 2,
                                 "outcome": "do-not-interpret", "sign_consistent": "not-a-criterion"}
        pairs[f"{focus} - {audit.RW_A}"] = {"focus_arm": focus, "baseline_arm": audit.RW_A, "cells": cells}
        primary[audit.PRIMARY_KEYS[focus]] = {"per_lead": by_lead}
    data["paired"] = {"scientific_claim": False,
                      "identity": {"dataset_identity": "synthetic-table-identity", "evaluation_split": "val"},
                      "table": table, "pairs": pairs, "primary": primary}


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("network forbidden in CPU audit tests")
    monkeypatch.setattr(socket.socket, "connect", reject)
    monkeypatch.setattr(socket.socket, "connect_ex", reject)
    monkeypatch.setattr(socket, "create_connection", reject)


@pytest.fixture
def tables(tmp_path):
    archive, manifests = tmp_path / "archive", tmp_path / "manifests"
    (archive / "seed41").mkdir(parents=True)
    manifests.mkdir()
    protocol = {"frozen_before_any_step": True, "scientific_claim": False, "seeds": [41, 42],
                "data": {"test_read": False, "data_identity": "synthetic-table-identity"},
                "shared_controls": {"reasoning_steps": 3, "process_weight": 0.0,
                                    "validation_lead_hours": [6], "evaluation_leads_hours": [6, 12, 24, 48, 72],
                                    "selection_split": "val only; test is sealed and never read"}}
    _seal(protocol)
    metrics, cases = [], []
    for seed_index, seed in enumerate((41, 42)):
        for arm in audit.ARMS:
            for lead in (6, 12, 24, 48, 72):
                metrics.append({"seed": seed, "arm": arm, "lead_hours": lead, "variable": "t2m", "unit": "K",
                                "rmse": 10 + lead / 6 + 2 * seed_index + OFFSETS[arm][seed_index],
                                "n_initializations": COUNTS[lead]})
                cases.append({"seed": seed, "arm": arm, "lead_hours": lead, "split": "val",
                              "n_available_windows": COUNTS[lead], "n_evaluated": COUNTS[lead], "test_read": False})
    manifest_rows = {}
    for split, count, start in (("train", 186, datetime(2016, 1, 1)), ("val", 22, datetime(2016, 5, 1))):
        manifest_rows[split] = [{"sample_id": f"synthetic_{split}_{index}", "split": split,
                                 "init_time": (start + timedelta(hours=6 * index)).isoformat(),
                                 "target_time": (start + timedelta(hours=6 * (index + 1))).isoformat(),
                                 "lead_time_hours": 6, "store_path": "../cache.zarr"} for index in range(count)]
    data = {"archive": archive, "manifests": manifests, "metrics": metrics, "cases": cases,
            "protocol": protocol, "manifest_rows": manifest_rows}
    _synchronize_pairs(data)
    _publish(data)
    return data


def _recompute(data):
    return audit.recompute(data["archive"], data["manifests"])


def test_exact_deltas_spreads_ratios_weights_and_counts(tables):
    report = _recompute(tables)
    assert audit.SEEDS == (41, 42)
    assert len(report["comparisons"]) == 4
    for row in report["comparisons"]:
        assert row["lead_hours"] in (48, 72) and row["unit"] == "K"
        if row["focus_arm"] == audit.RW_B:
            assert row["seed_deltas_K"] == {"41": 1.0, "42": 3.0}
            assert row["mean_delta_K"] == 2.0
            assert row["focus_seed_spread_K"] == 4.0
            assert row["mean_delta_over_focus_spread"]["value"] == 0.5
            assert row["mean_delta_over_baseline_spread"]["value"] == 1.0
        else:
            assert row["seed_deltas_K"] == {"41": 2.0, "42": 1.0}
            assert row["mean_delta_K"] == 1.5
            assert row["focus_seed_spread_K"] == 1.0
            assert row["mean_delta_over_focus_spread"]["value"] == 1.5
            assert row["mean_delta_over_baseline_spread"]["value"] == 0.75
        assert row["baseline_seed_spread_K"] == 2.0
        assert "outcome" not in row and "sign_consistent" not in row
    assert "range, not CI" in report["seed_spread_definition"]
    assert report["manifests"]["train"] == {"split": "train", "count": 186,
        "lead_distribution_hours": {"6": 186}, "plus_6h_only": True}
    assert report["manifests"]["val"]["count"] == 22
    assert {int(lead): row["n_available_windows"] for lead, row in report["val_cases_by_lead"].items()} == COUNTS
    weights = report["training_objective"]
    assert weights["normalized_weights_exact"] == ["1/6", "2/9", "5/18", "1/3"]
    assert weights["normalized_sum_exact"] == "1"
    assert weights["reasoning_steps"] == 3 and weights["process_weight_from_protocol"] == 0.0
    assert weights["final_weight_frozen_input"] == 2
    assert weights["protocol_final_weight_field_present"] is False
    assert weights["protocol_final_weight_field_value"] is None
    assert report["scientific_claim"] is False and report["test_read"] is False
    assert report["gpu_hours"] == 0 and report["network_requests"] == 0
    assert report["limitations"] and report["cross_checks"]["t2m_csv_rows_checked"] == 40


@pytest.mark.parametrize("zero_arm", [audit.RW_A, audit.RW_B, "both"])
def test_zero_spread_is_explicit_null_not_epsilon(tables, zero_arm):
    for arm in (audit.RW_A, audit.RW_B):
        if zero_arm not in (arm, "both"):
            continue
        for lead in (48, 72):
            first = next(row["rmse"] for row in tables["metrics"]
                         if row["arm"] == arm and row["lead_hours"] == lead and row["seed"] == 41)
            for row in tables["metrics"]:
                if row["arm"] == arm and row["lead_hours"] == lead:
                    row["rmse"] = first
    _synchronize_pairs(tables)
    _publish(tables)
    for row in _recompute(tables)["comparisons"][:2]:
        for role, arm in (("focus", audit.RW_B), ("baseline", audit.RW_A)):
            ratio = row[f"mean_delta_over_{role}_spread"]
            if zero_arm in (arm, "both"):
                assert row[f"{role}_seed_spread_K"] == 0
                assert ratio == {"value": None, "defined": False, "undefined_reason": "zero seed spread (0 K)"}
            else:
                assert ratio["defined"] is True and ratio["value"] is not None


def _json_keys(value):
    if isinstance(value, dict):
        return set(value) | set().union(*(_json_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_json_keys(item) for item in value))
    return set()


def test_opposite_seed_signs_are_reported_without_classification(tables):
    for row in tables["metrics"]:
        if row["arm"] == audit.RW_B and row["seed"] == 42:
            row["rmse"] -= 6
    _synchronize_pairs(tables)
    _publish(tables)
    report = _recompute(tables)
    assert report["comparisons"][0]["seed_deltas_K"] == {"41": 1.0, "42": -3.0}
    assert report["comparisons"][0]["mean_delta_K"] == -1.0
    assert report["comparisons"][0]["mean_delta_over_baseline_spread"]["value"] == -0.5
    assert not {"classification", "outcome", "sign_consistent", "threshold"} & _json_keys(report)


@pytest.mark.parametrize("damage, message", [
    ("metric_missing", "metrics missing"), ("metric_duplicate", "duplicate t2m metric"),
    ("metric_seed", "unexpected seed"), ("metric_unit", "unit must be K"),
    ("metric_nan", "finite number"), ("metric_inf", "finite number"), ("metric_negative", "rmse/count"),
    ("metric_count", "rmse/count"), ("metric_lead", "unexpected seed"), ("metric_fractional_seed", "integer"),
    ("case_missing", "case_table missing"), ("case_duplicate", "duplicate case_table"),
    ("case_seed", "unexpected seed"), ("case_split", "must be val"), ("case_test", "test_read False"),
    ("case_count", "count disagrees"), ("case_available", "count disagrees"), ("case_both_counts", "count disagrees"),
])
def test_csv_counterproofs(tables, damage, message):
    metric, case = tables["metrics"][0], tables["cases"][0]
    changes = {"metric_seed": (metric, "seed", 43), "metric_unit": (metric, "unit", "normalized"),
               "metric_nan": (metric, "rmse", "nan"), "metric_inf": (metric, "rmse", "inf"),
               "metric_negative": (metric, "rmse", -1), "metric_count": (metric, "n_initializations", 1),
               "metric_lead": (metric, "lead_hours", 51), "metric_fractional_seed": (metric, "seed", 41.5),
               "case_seed": (case, "seed", 43), "case_split": (case, "split", "train"),
               "case_test": (case, "test_read", True), "case_count": (case, "n_evaluated", 1),
               "case_available": (case, "n_available_windows", 1)}
    if damage in changes:
        row, key, value = changes[damage]
        row[key] = value
    elif damage.endswith("missing"):
        tables["metrics" if damage.startswith("metric") else "cases"].pop(0)
    elif damage.endswith("duplicate"):
        tables["metrics" if damage.startswith("metric") else "cases"].append(dict(metric if damage.startswith("metric") else case))
    else:
        case["n_evaluated"] = case["n_available_windows"] = 1
    _publish(tables)
    with pytest.raises(ValueError, match=message):
        _recompute(tables)


@pytest.mark.parametrize("damage, message", [
    ("table_missing", "paired table missing"), ("table_duplicate", "duplicate paired"),
    ("rmse", "paired seed_rmse"), ("mean", "paired rmse_seed_mean"), ("counts", "paired count mismatch"),
    ("unit", "paired t2m unit"), ("depth", "arm/lead/depth"), ("seeds_missing", "missing, duplicate"),
    ("seeds_duplicate", "missing, duplicate"), ("seed_map", "seed keys"), ("digest", "case digest mismatch"),
    ("digest_invalid", "invalid paired case digest"), ("case_identity", "case_identity must be exact"),
    ("cell_delta", "paired cell delta"), ("cell_unit", "cell unit"), ("cell_seeds", "seed keys"),
    ("primary_delta", "paired primary delta"), ("primary_mean", "paired primary mean delta"),
    ("identity", "dataset identity mismatch"), ("split", "must be val"), ("claim", "scientific_claim false"),
    ("baseline", "focus/baseline identity"),
])
def test_paired_json_counterproofs(tables, damage, message):
    paired = tables["paired"]
    table = paired["table"][0]
    pair = paired["pairs"][f"{audit.RW_B} - {audit.RW_A}"]
    cell = pair["cells"]["48h|t2m"]
    primary = paired["primary"]["round_reference"]["per_lead"]["48"]
    changes = {"rmse": (table["seed_rmse"], "41", 0), "mean": (table, "rmse_seed_mean", 0),
               "counts": (table["per_seed_n_initializations"], "41", 1), "unit": (table, "unit", "C"),
               "depth": (table, "depth", 1), "seeds_missing": (table, "seeds", [41]),
               "seeds_duplicate": (table, "seeds", [41, 41]), "seed_map": (table, "seed_rmse", {"41": 1}),
               "digest": (table["case_set_digests"], "42", "a" * 64),
               "digest_invalid": (table["case_set_digests"], "41", "invalid"),
               "case_identity": (table, "case_identity", "count-only"),
               "cell_delta": (cell["seed_deltas"], "41", 0), "cell_unit": (cell, "unit", "normalized"),
               "cell_seeds": (cell, "seed_deltas", {"41": 1}),
               "primary_delta": (primary["seed_deltas"], "41", 0), "primary_mean": (primary, "delta_seed_mean", 0),
               "identity": (paired["identity"], "dataset_identity", "other"),
               "split": (paired["identity"], "evaluation_split", "test"), "claim": (paired, "scientific_claim", True),
               "baseline": (pair, "baseline_arm", audit.NO_RECURRENCE)}
    if damage in changes:
        target, key, value = changes[damage]
        target[key] = value
    elif damage == "table_missing":
        paired["table"].pop(0)
    else:
        paired["table"].append(dict(table))
    _publish(tables)
    with pytest.raises(ValueError, match=message):
        _recompute(tables)


@pytest.mark.parametrize("damage, message", [
    ("digest", "protocol_sha256 mismatch"), ("missing_seed", "missing, duplicate"),
    ("duplicate_seed", "missing, duplicate"), ("extra_seed", "missing, duplicate"),
    ("frozen", "must be frozen"), ("claim", "scientific_claim false"), ("test_read", "test_read must be false"),
    ("steps", "reasoning_steps"), ("process", "process_weight"), ("final_weight", "final_weight"),
    ("validation_lead", "lead declarations"), ("evaluation_leads", "lead declarations"),
    ("selection_split", "selection_split"),
])
def test_protocol_counterproofs(tables, damage, message):
    protocol, controls = tables["protocol"], tables["protocol"]["shared_controls"]
    changes = {"missing_seed": (protocol, "seeds", [41]), "duplicate_seed": (protocol, "seeds", [41, 41]),
               "extra_seed": (protocol, "seeds", [41, 42, 43]), "frozen": (protocol, "frozen_before_any_step", False),
               "claim": (protocol, "scientific_claim", True), "test_read": (protocol["data"], "test_read", True),
               "steps": (controls, "reasoning_steps", 4), "process": (controls, "process_weight", 0.1),
               "final_weight": (controls, "final_weight", 3), "validation_lead": (controls, "validation_lead_hours", [12]),
               "evaluation_leads": (controls, "evaluation_leads_hours", [48, 72]),
               "selection_split": (controls, "selection_split", "train")}
    if damage == "digest":
        protocol["protocol_sha256"] = "0" * 64
    else:
        target, key, value = changes[damage]
        target[key] = value
        _seal(protocol)
    _publish(tables)
    with pytest.raises(ValueError, match=message):
        _recompute(tables)


@pytest.mark.parametrize("damage, message", [
    ("split", "manifest split mismatch"), ("id", "duplicate sample_id"), ("init", "duplicate initialization"),
    ("lead", "lead/time mismatch"), ("fractional_lead", "integer"), ("nonpositive_lead", "must be positive"),
    ("empty", "manifest is empty"), ("overlap_id", "sample_id overlap"), ("overlap_time", "time overlap"),
])
def test_manifest_counterproofs(tables, damage, message):
    train = tables["manifest_rows"]["train"]
    first = train[0]
    if damage == "split":
        first["split"] = "val"
    elif damage == "id":
        train[1]["sample_id"] = first["sample_id"]
    elif damage == "init":
        train.append({**first, "sample_id": "distinct_id"})
    elif damage == "lead":
        first["lead_time_hours"] = 12
    elif damage == "fractional_lead":
        first["lead_time_hours"] = 6.5
    elif damage == "nonpositive_lead":
        first["lead_time_hours"] = 0
    elif damage == "empty":
        tables["manifest_rows"]["train"] = []
    elif damage == "overlap_id":
        first["sample_id"] = tables["manifest_rows"]["val"][0]["sample_id"]
    else:
        val = tables["manifest_rows"]["val"][0]
        first.update(init_time=val["init_time"], target_time=val["target_time"])
    _write_manifests(tables)
    with pytest.raises(ValueError, match=message):
        _recompute(tables)


def test_distribution_is_observed_not_a_new_threshold(tables):
    first = tables["manifest_rows"]["train"][0]
    first["lead_time_hours"] = 12
    first["target_time"] = (datetime.fromisoformat(first["init_time"]) + timedelta(hours=12)).isoformat()
    _write_manifests(tables)
    summary = _recompute(tables)["manifests"]["train"]
    assert summary["lead_distribution_hours"] == {"12": 1, "6": 185}
    assert summary["plus_6h_only"] is False and summary["count"] == 186


@pytest.mark.parametrize("text, message", [
    ('{"identity":{},"identity":{}}', "duplicate JSON key"),
    ('{"seed_rmse":{"41":1,"41":2}}', "duplicate JSON key"),
    ('{"rmse":NaN}', "non-finite JSON"), ('{"rmse":Infinity}', "non-finite JSON"),
    ('[]', "must be an object"),
])
def test_invalid_json_fails_closed(tables, text, message):
    (tables["archive"] / "paired_comparison.json").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _recompute(tables)


@pytest.mark.parametrize("text", ["seed,seed\n41,41\n", "seed,arm\n41,too,many\n"])
def test_invalid_csv_headers_are_rejected(tables, text):
    (tables["archive"] / "rmse_table.csv").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="CSV columns"):
        _recompute(tables)


def test_existing_final_weight_is_recorded_only_if_present(tables):
    tables["protocol"]["shared_controls"]["final_weight"] = 2
    _seal(tables["protocol"])
    _publish(tables)
    weights = _recompute(tables)["training_objective"]
    assert weights["protocol_final_weight_field_present"] is True
    assert weights["protocol_final_weight_field_value"] == 2


def test_stdout_json_source_and_script_digests(tables, capsys):
    assert audit.main(["--archive", str(tables["archive"]), "--manifests", str(tables["manifests"])]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == _recompute(tables)
    assert len(report["sources"]) == 6
    for record in report["sources"].values():
        raw = Path(record["path"]).read_bytes()
        assert record["sha256"] == hashlib.sha256(raw).hexdigest()
        assert record["bytes"] == len(raw)
    assert report["script"]["sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert audit.DEFAULT_ARCHIVE == SCRIPT.resolve().parents[1] / "outputs/r7_72_rw_b_subtraction"
    assert audit.DEFAULT_MANIFESTS == SCRIPT.resolve().parents[1] / "outputs/r7_m2_segment/store/manifests"


def test_read_allowlist_including_no_sealed_metadata(tables, monkeypatch):
    allowed = {tables["archive"] / name for name in ("rmse_table.csv", "case_table.csv",
        "paired_comparison.json", "seed41/protocol.json")} | {
        tables["manifests"] / "train.jsonl", tables["manifests"] / "val.jsonl", SCRIPT}
    original_read, original_open = Path.read_bytes, Path.open
    seen = []
    def read(path):
        assert path in allowed, f"unexpected read: {path}"
        seen.append(path)
        return original_read(path)
    def open_read_only(path, mode="r", *args, **kwargs):
        assert path in allowed and mode == "rb", f"unexpected open: {path} {mode}"
        return original_open(path, mode, *args, **kwargs)
    def guarded_metadata(original):
        def inspect(path, *args, **kwargs):
            assert "test.jsonl" not in path.parts and "cache.zarr" not in path.parts
            assert path.suffix not in (".pt", ".npy", ".npz")
            return original(path, *args, **kwargs)
        return inspect
    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(Path, "open", open_read_only)
    monkeypatch.setattr(Path, "stat", guarded_metadata(Path.stat))
    monkeypatch.setattr(Path, "lstat", guarded_metadata(Path.lstat))
    report = _recompute(tables)
    assert set(seen) == allowed and len(seen) == 7
    assert report["test_read"] is False


def test_output_is_exclusive_and_never_modifies_sources(tables, tmp_path):
    before = {path: path.read_bytes() for path in (*tables["archive"].rglob("*"), *tables["manifests"].iterdir()) if path.is_file()}
    output = tmp_path / "audit.json"
    args = ["--archive", str(tables["archive"]), "--manifests", str(tables["manifests"]), "--output", str(output)]
    assert audit.main(args) == 0
    first = output.read_bytes()
    assert json.loads(first)["scientific_claim"] is False
    assert audit.main(args) == 2
    assert output.read_bytes() == first
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("destination", ["source", "archive_new", "manifests_new", "no_parent", "script", "test"])
def test_output_rejection_counterproofs(tables, tmp_path, capsys, destination):
    destinations = {"source": tables["archive"] / "rmse_table.csv", "archive_new": tables["archive"] / "audit.json",
                    "manifests_new": tables["manifests"] / "audit.json", "no_parent": tmp_path / "absent/audit.json",
                    "script": SCRIPT, "test": tmp_path / "test.jsonl"}
    output = destinations[destination]
    before = output.read_bytes() if output.is_file() else None
    assert audit.main(["--archive", str(tables["archive"]), "--manifests", str(tables["manifests"]), "--output", str(output)]) == 2
    assert "N1 audit error" in capsys.readouterr().err
    if before is None:
        assert not output.exists()
    else:
        assert output.read_bytes() == before


def test_symlink_alias_is_rejected_before_target_read(tables, tmp_path, monkeypatch):
    alias = tmp_path / "archive_alias"
    alias.symlink_to(tables["archive"], target_is_directory=True)
    def no_read(path):
        raise AssertionError(f"symlink must fail before reading {path}")
    monkeypatch.setattr(Path, "read_bytes", no_read)
    with pytest.raises(ValueError, match="symlink"):
        audit.recompute(alias, tables["manifests"])


def test_missing_archive_is_a_real_failure_not_skip(tables, tmp_path, capsys):
    assert audit.main(["--archive", str(tmp_path / "missing"), "--manifests", str(tables["manifests"])]) == 2
    assert "N1 audit error" in capsys.readouterr().err


def test_standalone_cli_uses_only_stdlib_and_no_network(tables, tmp_path):
    bootstrap = """import runpy, sys
script = sys.argv.pop(1)
def reject(event, args):
    if event.startswith('socket.') or (event == 'import' and args[0].split('.')[0] in ('torch', 'model', 'training', 'numpy', 'zarr')):
        raise AssertionError('forbidden dependency/network')
sys.addaudithook(reject)
runpy.run_path(script, run_name='__main__')
"""
    completed = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", bootstrap, str(SCRIPT),
        "--archive", str(tables["archive"]), "--manifests", str(tables["manifests"])],
        cwd=tmp_path, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=15, check=False)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == _recompute(tables)
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imported <= sys.stdlib_module_names
    assert "torch" not in imported and "model" not in imported
    assert len(SCRIPT.read_text(encoding="utf-8").splitlines()) <= 600
    assert all(node.end_lineno - node.lineno + 1 <= 200 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
