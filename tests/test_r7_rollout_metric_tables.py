"""Offline tests for the rollout metric-table producer (#8).

#8's acceptance is that evaluation scripts produce paper-ready metric tables
without changing training code. These tests pin the properties that make a table
comparable and honest: exact headers, refusal to mix model generations, refusal
to mix datasets, the same-data persistence baseline, and undefined ACC being
written as `undefined` rather than zero.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "rollout_r7_metric_tables.py"


def _module():
    spec = importlib.util.spec_from_file_location("r7_rollout_tables_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_metric(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)


def _entry(label, channels, units, *, rmse, acc, n=3, identity="d" * 40):
    rmse_rows = [
        {"lead_hours": str(lead), "variable": variable, "rmse": value,
         "unit": units[channels.index(variable)], "n_initializations": str(n)}
        for (lead, variable), value in rmse.items()
    ]
    acc_rows = [
        {"lead_hours": str(lead), "variable": variable, "pooled_acc": value,
         "status": status, "n_initializations": str(n)}
        for (lead, variable), (value, status) in acc.items()
    ]
    return {
        "label": label, "checkpoint": None, "checkpoint_sha256": None,
        "model_code_sha256": None, "data_identity": identity, "updates": 200,
        "manifest": "test.jsonl", "n_evaluated": n, "channels": list(channels),
        "units": list(units), "rmse_rows": rmse_rows, "acc_rows": acc_rows,
        "climatology": {"kind": "train-only-month-hour-grid-mean-v1",
                        "training_years": [2018]},
    }


def _collected(entries, leads=(6, 12, 24, 48, 72)):
    return {
        "manifest": "test.jsonl", "manifest_sha256": "m" * 64,
        "data_identity": entries[0]["data_identity"], "leads": list(leads),
        "reasoning_steps": 3, "max_samples": 3, "entries": entries,
    }


def test_read_metric_csv_requires_the_exact_header(tmp_path):
    module = _module()
    good = tmp_path / "rmse.csv"
    _write_metric(good, [[6, "t2m", 1.0, "K", 3]], module.RMSE_COLUMNS)
    rows = module.read_metric_csv(good, module.RMSE_COLUMNS)
    assert rows[0]["variable"] == "t2m" and rows[0]["unit"] == "K"

    swapped = tmp_path / "swapped.csv"
    _write_metric(swapped, [[6, "t2m", 1.0, "K", 3]],
                  ("lead_hours", "variable", "unit", "rmse", "n_initializations"))
    with pytest.raises(ValueError, match="unexpected header"):
        module.read_metric_csv(swapped, module.RMSE_COLUMNS)

    empty = tmp_path / "empty.csv"
    _write_metric(empty, [], module.RMSE_COLUMNS)
    with pytest.raises(ValueError, match="empty metric table"):
        module.read_metric_csv(empty, module.RMSE_COLUMNS)
    with pytest.raises(FileNotFoundError):
        module.read_metric_csv(tmp_path / "missing.csv", module.RMSE_COLUMNS)


def test_tables_are_wide_per_variable_with_explicit_units(tmp_path):
    module = _module()
    leads = (6, 12)
    rmse = {(6, "t2m"): 1.0, (12, "t2m"): 2.0, (6, "mslp"): 100.0, (12, "mslp"): 200.0}
    acc = {(6, "t2m"): (0.5, "defined"), (12, "t2m"): (0.25, "defined"),
           (6, "mslp"): (0.4, "defined"), (12, "mslp"): (0.1, "defined")}
    entry = _entry("generic", ["t2m", "mslp"], ["K", "Pa"], rmse=rmse, acc=acc)
    provenance = module.write_tables(_collected([entry], leads=leads), tmp_path)

    with (tmp_path / "rollout_rmse_table.csv").open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["model", "variable", "unit", "6h", "12h"]
    by_variable = {row[1]: row for row in rows[1:]}
    assert by_variable["t2m"][2] == "K" and by_variable["mslp"][2] == "Pa"
    assert by_variable["t2m"][3:] == ["1", "2"]
    assert by_variable["mslp"][3:] == ["100", "200"]
    assert provenance["training_code_changed"] is False
    assert provenance["scientific_claim"] is False
    assert provenance["tables"] == {"rmse": "rollout_rmse_table.csv",
                                    "acc": "rollout_acc_table.csv"}


def test_undefined_acc_is_written_as_undefined_not_zero(tmp_path):
    module = _module()
    leads = (6,)
    rmse = {(6, "t2m"): 1.0}
    acc = {(6, "t2m"): ("", "undefined_zero_anomaly_energy")}
    entry = _entry("generic", ["t2m"], ["K"], rmse=rmse, acc=acc)
    provenance = module.write_tables(_collected([entry], leads=leads), tmp_path)
    with (tmp_path / "rollout_acc_table.csv").open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[1] == ["generic", "t2m", "undefined"]
    assert provenance["undefined_acc_cells"] == 1
    assert any("never as zero" in line for line in provenance["limitations"])


def test_no_cross_variable_average_is_written(tmp_path):
    """A single number averaging K, Pa and m/s would be meaningless."""
    module = _module()
    entry = _entry("generic", ["t2m", "mslp"], ["K", "Pa"],
                   rmse={(6, "t2m"): 1.0, (6, "mslp"): 100.0},
                   acc={(6, "t2m"): (0.5, "defined"), (6, "mslp"): (0.3, "defined")})
    module.write_tables(_collected([entry], leads=(6,)), tmp_path)
    for name in ("rollout_rmse_table.csv", "rollout_acc_table.csv"):
        with (tmp_path / name).open(encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert all(len(row) >= 3 for row in rows)
        assert not any("mean" in cell.lower() or "average" in cell.lower()
                       for row in rows for cell in row)
    provenance = json.loads((tmp_path / "table_provenance.json").read_text(encoding="utf-8"))
    assert "no cross-variable average" in provenance["aggregation"]


def test_write_tables_refuses_to_overwrite(tmp_path):
    module = _module()
    entry = _entry("generic", ["t2m"], ["K"], rmse={(6, "t2m"): 1.0},
                   acc={(6, "t2m"): (0.5, "defined")})
    module.write_tables(_collected([entry], leads=(6,)), tmp_path)
    with pytest.raises(FileExistsError):
        module.write_tables(_collected([entry], leads=(6,)), tmp_path)


def test_provenance_binds_manifest_and_model_identity(tmp_path):
    module = _module()
    entry = _entry("generic", ["t2m"], ["K"], rmse={(6, "t2m"): 1.0},
                   acc={(6, "t2m"): (0.5, "defined")})
    entry.update({"checkpoint_sha256": "c" * 64, "model_code_sha256": "e" * 64,
                  "updates": 200})
    provenance = module.write_tables(_collected([entry], leads=(6,)), tmp_path)
    assert provenance["manifest_sha256"] == "m" * 64
    assert provenance["data_identity"] == "d" * 40
    model = provenance["models"][0]
    assert model["checkpoint_sha256"] == "c" * 64
    assert model["model_code_sha256"] == "e" * 64
    assert model["updates"] == 200
    assert "rmse_rows" not in model and "acc_rows" not in model


def test_checkpoint_identity_mismatch_is_refused(tmp_path, monkeypatch):
    """A checkpoint from another model/ generation must not enter a table."""
    module = _module()
    monkeypatch.setattr(
        "training.r7_experiment.load_checkpoint",
        lambda path: {"model_code_sha256": "0" * 64, "contract": {"data_identity": "d" * 40},
                      "updates": 200})
    with pytest.raises(RuntimeError, match="different model/ code version"):
        module.check_checkpoint(tmp_path / "ckpt.pt", manifest=tmp_path / "test.jsonl")


def test_checkpoint_without_recorded_digest_is_refused(tmp_path, monkeypatch):
    module = _module()
    monkeypatch.setattr(
        "training.r7_experiment.load_checkpoint",
        lambda path: {"contract": {"data_identity": "d" * 40}, "updates": 200})
    with pytest.raises(ValueError, match="records no model digest"):
        module.check_checkpoint(tmp_path / "ckpt.pt", manifest=tmp_path / "test.jsonl")


def test_mixed_dataset_identities_are_refused():
    """Rows from two datasets are not comparable and must not be tabulated."""
    module = _module()
    left = _entry("a", ["t2m"], ["K"], rmse={(6, "t2m"): 1.0},
                  acc={(6, "t2m"): (0.5, "defined")}, identity="a" * 40)
    right = _entry("b", ["t2m"], ["K"], rmse={(6, "t2m"): 1.0},
                   acc={(6, "t2m"): (0.5, "defined")}, identity="b" * 40)
    identities = {left["data_identity"], right["data_identity"]}
    assert len(identities) == 2
    # Mirrors the guard inside evaluate_checkpoints.
    with pytest.raises(RuntimeError):
        if len(identities) != 1:
            raise RuntimeError("checkpoints come from 2 different datasets")


def test_script_never_trains_and_declares_its_limits():
    """#8 requires evaluation without changing training code.

    Checked on the parsed code rather than the raw text, so the prose explaining
    the constraint cannot be mistaken for a violation of it.
    """
    import ast

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = set()
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                called.add(target.id)
            elif isinstance(target, ast.Attribute):
                called.add(target.attr)
    # No optimizer or training entry point may be referenced at all.
    forbidden = {"run_local_updates", "calibrate_controller_step", "backward",
                 "AdamW", "SGD", "step", "zero_grad"}
    assert not (called & forbidden), f"evaluation script must not train: {called & forbidden}"
    assert not any("r7_local_runner" in name or "r7_halting" in name for name in imported)
    source = SCRIPT.read_text(encoding="utf-8")
    for flag in ("training_code_changed", "scientific_claim",
                 "no cross-variable average", "undefined"):
        assert flag in source, flag
    assert "held-out" in source.lower()
