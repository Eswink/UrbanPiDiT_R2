"""Synthetic CPU supervisor contracts only, not weather or GPU evidence."""
from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
import time
import zipfile

import pytest

from scripts import diagnose_r7_s3_same_case_gap as driver
from training.r7_experiment import canonical_digest


def _fake_round(tmp_path, monkeypatch, *, fail_phase=None, reject_gate=False):
    calls, phases = [], []
    monkeypatch.setattr(driver, "PLANNED_SECONDS", 5.)
    monkeypatch.setattr(driver, "HARD_SECONDS", 10.)
    output = tmp_path / "round"
    def _gate(uuid, **kwargs):
        calls.append((uuid, kwargs))
        if reject_gate:
            raise RuntimeError("synthetic headroom rejection")
        return {"passed": True, "required_bytes": max(kwargs["estimated_peak_bytes"],
                kwargs["owned_reserved_peak_bytes"]) + kwargs["margin_bytes"]}
    monkeypatch.setattr(driver.recipe._shared(), "gpu_gate", _gate)
    def _process(arguments, **kwargs):
        phase = arguments[arguments.index("--worker") + 1]
        phases.append(phase)
        if fail_phase != phase:
            if phase == "prepare":
                driver.write_exclusive(output / "prepared_protocol.json", {
                    "protocol_sha256": "a" * 64, "limitations": ["Declared synthetic fixture"]})
            elif phase == "measurement":
                driver.write_exclusive(output / "measurement_receipt.json", {
                    "scientific_claim": False, "test_read": False, "owned_cuda_reserved_peak_bytes": 3 * 2**30})
        return {"status": "failed" if fail_phase == phase else "success", "returncode": 1 if fail_phase == phase else 0,
                "reaped": True, "signals": [], "elapsed_seconds": .01}
    monkeypatch.setattr(driver, "bounded_process", _process)
    return output, calls, phases


def _run(output, tmp_path):
    return driver.run_attempt(output, code_archive=tmp_path / "archive.zip", code_commit="a" * 40,
                              code_sha256="b" * 64)


def test_synthetic_complete_round_claims_cost_and_four_phases(tmp_path, monkeypatch):
    output, calls, phases = _fake_round(tmp_path, monkeypatch)
    result = _run(output, tmp_path)
    assert phases == ["prepare", "archive", "measurement", "reading"]
    assert result["scientific_claim"] is False and result["test_read"] is False
    assert result["decision"] == "diagnostic-only" and len(result["processes"]) == 4
    assert result["whole_round_conservative_gpu_hours"] == result["elapsed_seconds_total"] / 3600
    assert json.loads((output / "attempt.json").read_text(encoding="utf-8"))["status"] == "complete"
    assert len(calls) == 4
    for uuid, gate in calls:
        assert uuid == driver.GPU_UUID and gate["estimated_peak_bytes"] == gate["margin_bytes"] == 2**31
        assert gate["owned_reserved_peak_bytes"] >= driver.KNOWN_RESERVED_PEAK
    assert calls[-1][1]["owned_reserved_peak_bytes"] == 3 * 2**30
    with pytest.raises(FileExistsError):
        _run(output, tmp_path)


@pytest.mark.parametrize("phase", ["prepare", "archive", "measurement", "reading"])
def test_true_failure_stops_all_future_workers_and_keeps_cost(tmp_path, monkeypatch, phase):
    output, _, phases = _fake_round(tmp_path, monkeypatch, fail_phase=phase)
    with pytest.raises(RuntimeError, match=phase):
        _run(output, tmp_path)
    expected = ["prepare", "archive", "measurement", "reading"]
    assert phases == expected[:expected.index(phase) + 1]
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["scientific_claim"] is False and failure["test_read"] is False and failure["limitations"]
    assert failure["whole_round_conservative_gpu_hours"] == failure["elapsed_seconds_total"] / 3600
    assert not (output / "attempt.json").exists()


def test_headroom_rejects_before_any_child_without_signal(tmp_path, monkeypatch):
    output, calls, phases = _fake_round(tmp_path, monkeypatch, reject_gate=True)
    with pytest.raises(RuntimeError, match="headroom"):
        _run(output, tmp_path)
    assert len(calls) == 1 and not phases
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["processes"] == []


def test_external_watchdog_reaps_only_owned_blocked_cpu_child(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "PLANNED_SECONDS", .1)
    monkeypatch.setattr(driver, "HARD_SECONDS", .3)
    monkeypatch.setattr(driver.recipe._shared(), "gpu_gate", lambda *args, **kwargs: {"passed": True})
    actual = driver.bounded_process
    def _blocked(arguments, **kwargs):
        return actual([driver.sys.executable, "-c", "import time; time.sleep(20)"], **kwargs)
    monkeypatch.setattr(driver, "bounded_process", _blocked)
    output = tmp_path / "blocked"
    with pytest.raises(RuntimeError, match="prepare"):
        _run(output, tmp_path)
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    process = failure["processes"][0]
    assert process["status"] == "deadline-exceeded" and process["reaped"]
    assert all(signal["pid"] == process["pid"] for signal in process["signals"])
    assert .3 <= failure["elapsed_seconds_total"] < 8


def _protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "execution_files", lambda: [])
    monkeypatch.setattr(driver, "checkpoint_pins", lambda: {"synthetic": "fixed"})
    monkeypatch.setattr(driver, "model_code_digest", lambda: "b" * 64)
    started = time.perf_counter()
    driver.freeze(tmp_path, started=started, deadline=started + driver.HARD_SECONDS,
                  code_archive=tmp_path / "archive.zip", code_commit="a" * 40, code_sha256="c" * 64)
    body = driver.read_json(tmp_path / "preparation_protocol.json")
    body.update({"checkpoints": {"synthetic": "fixed"}, "execution_files_sha256": {},
                 "train_data_identity": driver.recipe._shared().V3_DATA_IDENTITY,
                 "val_data_identity": driver.VAL_DATA_IDENTITY, "model_code_sha256": "b" * 64})
    body["protocol_sha256"] = canonical_digest({k: v for k, v in body.items() if k != "protocol_sha256"})
    driver.write_exclusive(tmp_path / "protocol.json", body)
    return body


@pytest.mark.parametrize("key,value", [("bf16", True), ("scientific_claim", True), ("test_read", True),
    ("seed", 42), ("lead_hours", [12, 24, 36, 48, 72]), ("physical_steps", 6),
    ("train_years", [2017]), ("val_years", [2023]), ("reasoning_steps", 3), ("hard_cap_seconds", 7200),
    ("known_owned_reserved_peak_bytes", 0), ("margin_bytes", 0), ("limitations", []),
    ("selection", "metric-selected cases"), ("baseline", "weaker lead-specific mean"),
    ("train_data_identity", "0" * 64), ("val_data_identity", "0" * 64), ("source_bytes", 0),
    ("expected_climatology", {"n_selected_steps": 16, "bucket_counts": {"01-00": 1}}),
    ("expected_contracts", {"candidate": {"lr": .1}})])
def test_refreeze_cannot_change_diagnostic_scope(tmp_path, monkeypatch, key, value):
    body = _protocol(tmp_path, monkeypatch)
    assert driver.validate_protocol(tmp_path, body["deadline_perf_counter"]) == body
    body[key] = value
    body["protocol_sha256"] = canonical_digest({k: v for k, v in body.items() if k != "protocol_sha256"})
    (tmp_path / "protocol.json").write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(RuntimeError):
        driver.validate_protocol(tmp_path, body["deadline_perf_counter"])


def test_deadline_outside_frozen_round_refused(tmp_path, monkeypatch):
    body = _protocol(tmp_path, monkeypatch)
    for deadline in (time.perf_counter() - 1, body["deadline_perf_counter"] + 1, float("inf"), True):
        with pytest.raises(RuntimeError, match="deadline"):
            driver.validate_protocol(tmp_path, deadline)


def _archive(tmp_path, monkeypatch):
    payload = BytesIO()
    with zipfile.ZipFile(payload, "w") as zipped:
        zipped.comment = b"a" * 40
        zipped.writestr("training/synthetic.py", "x = 1\n")
    path = tmp_path / "source.zip"
    path.write_bytes(payload.getvalue())
    output = tmp_path / "copied"
    output.mkdir()
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    body = {"code_archive_input": str(path), "code_commit": "a" * 40,
        "code_zip_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "execution_files_sha256": {"training/synthetic.py": hashlib.sha256(b"x = 1\n").hexdigest()}}
    return output, path, body


def test_exact_archive_copy_is_exclusive(tmp_path, monkeypatch):
    output, source, body = _archive(tmp_path, monkeypatch)
    driver.archive(output, body)
    assert (output / "code.zip").read_bytes() == source.read_bytes()
    assert (output / "code_commit.txt").read_text(encoding="utf-8").strip() == "a" * 40
    with pytest.raises(FileExistsError):
        driver.archive(output, body)


@pytest.mark.parametrize("key,value", [("code_commit", "b" * 40), ("code_zip_sha256", "0" * 64)])
def test_archive_commit_or_digest_drift_refused(tmp_path, monkeypatch, key, value):
    output, _, body = _archive(tmp_path, monkeypatch)
    body[key] = value
    with pytest.raises(RuntimeError, match="archive"):
        driver.archive(output, body)
    assert not (output / "code.zip").exists()


def test_linked_archive_input_refused(tmp_path, monkeypatch):
    output, source, body = _archive(tmp_path, monkeypatch)
    link = tmp_path / "linked.zip"
    link.symlink_to(source)
    body["code_archive_input"] = str(link)
    with pytest.raises(ValueError, match="nonlinked"):
        driver.archive(output, body)


def _measurement(tmp_path):
    from training.r7_gap_diagnostic import aggregate_cases, metric_summary, _gap
    frozen = [{"sample_id": f"synthetic-{split}-{year}-{month}", "split": split, "year": year,
        "month": month, "manifest_index": i, "init_time": f"{year}-{month:02d}-02T06:00:00"}
        for i, (split, year, month) in enumerate(
            [("train", year, month) for year in range(2017, 2022) for month in (1, 4, 7, 10)]
            + [("val", 2022, month) for month in (1, 4, 7, 10)])]
    climate_meta = {"training_years": list(range(2017, 2022)), "n_selected_steps": 2400,
        "bucket_counts": {f"{month:02d}-{hour:02d}": 150
                          for month in (1, 4, 7, 10) for hour in (0, 6, 12, 18)},
        "selection": "declared_train_years"}
    plan = {"cases": frozen, "channels": [f"synthetic-{i}" for i in range(17)], "units": ["K"] * 17,
            "normalization_mean": [0.] * 17, "normalization_std": [2.] * 17,
            "shape": [3360, 17, 3, 4], "climatology_metadata": climate_meta,
            "train_manifest": str(tmp_path / "train.jsonl"), "source_declaration": "synthetic-source.bin"}
    plan["selection_sha256"] = canonical_digest(plan)
    pins = {role: {"sha256": sha * 64, "updates": updates, "mode": mode}
            for role, sha, updates, mode in (("parent", "b", 1600, "l6"), ("candidate", "c", 200, "long_rollout"))}
    body = {"protocol_sha256": "a" * 64, "case_plan": plan, "train_data_identity": "d" * 64,
            "val_data_identity": "e" * 64, "source_sha256": "f" * 64,
            "model_code_sha256": "0" * 64, "checkpoints": pins, "expected_climatology": climate_meta,
            "source_bytes": driver.SOURCE_BYTES}
    means = {bucket: {"sha256": "1" * 64, "shape": [17, 3, 4], "dtype": "<f8"}
             for bucket in climate_meta["bucket_counts"]}
    climate = {**climate_meta, "kind": "train-only-month-hour-grid-mean-v1", "channels": plan["channels"],
               "means": means, "mean_identity_sha256": canonical_digest(means)}
    cases = [{**case, **metric_summary({role: [[mse] * 17 for _ in range(5)]
             for role, mse in (("parent", 1.), ("candidate", 2.), ("climatology", 4.))})} for case in frozen]
    aggregates = aggregate_cases(cases)
    value = {"scientific_claim": False, "test_read": False, "limitations": ["Synthetic only"],
        "protocol_sha256": body["protocol_sha256"], "case_plan": plan, "selection_sha256": plan["selection_sha256"],
        "n_evaluated": 24, "cases": cases, "aggregates": aggregates, "train_val_gap": _gap(aggregates),
        "climatology": climate,
        **{key: body[key] for key in ("train_data_identity", "val_data_identity", "source_sha256", "model_code_sha256")},
        "source_identity": {"sha256": body["source_sha256"], "bytes": body["source_bytes"],
                            "scope": "full-local-file", "path": str(tmp_path / "synthetic-source.bin")},
        "checkpoints": {role: {**pin, "optimizer_imported": False, "seed": 41,
                       "model_code_sha256": body["model_code_sha256"]} for role, pin in pins.items()},
        "channels": plan["channels"], "units": plan["units"], "lead_hours": list(driver.LEADS),
        "training_normalization": {"mean": plan["normalization_mean"], "std": plan["normalization_std"]},
        "optimizer_imported": False, "optimizer_updates": 0, "diagnostic_training_mode": False,
        "bf16": False, "reasoning_steps": 4, "physical_transitions_per_model_case": 12}
    driver.write_exclusive(tmp_path / "measurements.json", value)
    driver.write_exclusive(tmp_path / "measurement_receipt.json", {
        "protocol_sha256": "a" * 64, "measurements_sha256": driver.sha256_file(tmp_path / "measurements.json")})
    return value, body


def test_reading_requires_all_finite_paired_case_metrics(tmp_path):
    _, body = _measurement(tmp_path)
    driver.reading(tmp_path, body)
    receipt = driver.read_json(tmp_path / "reading_receipt.json")
    assert receipt["status"] == "recorded-diagnostic-only"


@pytest.mark.parametrize("mutation", ["missing_case", "claim", "test", "axis", "nan", "negative", "arm",
    "duplicate", "reordered", "selection", "data", "source", "endpoint", "unit", "lead", "optimizer",
    "rmse", "skill", "aggregate", "gap", "climate_count", "climate_kind", "climate_mean",
    "climate_dtype", "climate_shape", "source_scope", "source_path", "source_bytes", "endpoint_code"])
def test_reading_rejects_corrupted_measurement(tmp_path, mutation):
    value, body = _measurement(tmp_path)
    if mutation == "missing_case":
        value["cases"].pop()
    elif mutation == "claim":
        value["scientific_claim"] = True
    elif mutation == "test":
        value["test_read"] = True
    elif mutation == "axis":
        value["cases"][0]["metrics"]["candidate"]["mse"] = [[1.]]
    elif mutation in ("nan", "negative"):
        value["cases"][0]["metrics"]["candidate"]["mse"][0][0] = float("nan") if mutation == "nan" else -1.
    elif mutation == "arm":
        value["cases"][0]["metrics"].pop("climatology")
    elif mutation == "duplicate":
        value["cases"][1] = value["cases"][0]
    elif mutation == "reordered":
        value["cases"][0], value["cases"][1] = value["cases"][1], value["cases"][0]
    elif mutation in ("selection", "data", "source"):
        key = {"selection": "selection_sha256", "data": "val_data_identity", "source": "source_sha256"}[mutation]
        value[key] = "9" * 64
    elif mutation == "endpoint":
        value["checkpoints"]["candidate"]["sha256"] = "9" * 64
    elif mutation in ("unit", "lead"):
        key = "units" if mutation == "unit" else "lead_hours"
        value[key] = list(value[key])
        value[key][0] = "normalized" if mutation == "unit" else 18
    elif mutation == "optimizer":
        value["optimizer_updates"] = 1
    elif mutation in ("rmse", "skill"):
        key = "rmse" if mutation == "rmse" else "mse_skill"
        value["cases"][0]["metrics"]["candidate"][key][0][0] += .1
    elif mutation == "aggregate":
        value["aggregates"]["split"]["val"]["metrics"]["candidate"]["mse"][0][0] += .1
    elif mutation == "gap":
        value["train_val_gap"]["candidate"]["val_minus_train_mse"][0][0] += .1
    elif mutation == "climate_count":
        value["climatology"]["n_selected_steps"] = 16
    elif mutation == "climate_kind":
        value["climatology"]["kind"] = "weaker lead-specific baseline"
    elif mutation == "climate_mean":
        value["climatology"]["mean_identity_sha256"] = "9" * 64
    elif mutation in ("source_scope", "source_path", "source_bytes"):
        key = {"source_scope": "scope", "source_path": "path", "source_bytes": "bytes"}[mutation]
        value["source_identity"][key] = {"scope": "partial", "path": "/tmp/not-declared-source", "bytes": 0}[key]
    elif mutation == "endpoint_code":
        value["checkpoints"]["candidate"]["model_code_sha256"] = "9" * 64
    else:
        mean = next(iter(value["climatology"]["means"].values()))
        mean["dtype" if mutation == "climate_dtype" else "shape"] = "<f4" if mutation == "climate_dtype" else [17, 1, 1]
        value["climatology"]["mean_identity_sha256"] = canonical_digest(value["climatology"]["means"])
    (tmp_path / "measurements.json").write_text(json.dumps(value), encoding="utf-8")
    receipt = driver.read_json(tmp_path / "measurement_receipt.json")
    receipt["measurements_sha256"] = driver.sha256_file(tmp_path / "measurements.json")
    (tmp_path / "measurement_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError):
        driver.reading(tmp_path, body)
