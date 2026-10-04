"""Synthetic-archive CPU tests for the S0 gap audit; all writes are in tmp_path.

The fixture rebuilds a miniature actual-C archive (protocol, manifest, worker
receipts, checkpoints, pooled and per-case CSVs) so the audit's identity checks,
per-case recomputation and gap arithmetic are exercised without reading the
real 7.9 GiB archive.  Counterproofs tamper with exactly one link of the
identity chain at a time and require the audit to refuse it.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from tools import recompute_r7_s0_gap_audit as audit

CASE_OFFSETS = {(41, "old_ours"): 0.0, (41, "process"): -0.2, (41, "matched_generic"): -0.2,
                (42, "old_ours"): 0.1, (42, "process"): -0.1, (42, "matched_generic"): -0.1,
                (43, "old_ours"): 0.2, (43, "process"): -0.3, (43, "matched_generic"): -0.3}
LEADS = audit.LEADS
REGIONS = audit.REGIONS
VARIABLES = audit.VARIABLES
DEPTHS = audit.DEPTHS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, rows: list, fields: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _model_mse(seed: int, arm: str, variable: str, lead: int, region: str, depth: int) -> float:
    """Deterministic synthetic MSE; t2m beats climatology at 6h only, as in the real audit."""
    base = 4.0 + 0.05 * (lead // 6) + 0.001 * VARIABLES.index(variable)
    if variable == "t2m" and lead == 6:
        base = 2.0
    if region == "interior":
        base *= 0.9
    if region == "edge_2":
        base *= 1.2
    return base + CASE_OFFSETS[(seed, arm)] + 0.01 * depth


def _climatology_mse(variable: str, lead: int, region: str) -> float:
    base = 2.6 + 0.02 * (lead // 6) + 0.001 * VARIABLES.index(variable)
    if variable == "t2m":
        base = 2.5 + 0.05 * (lead // 6)
    if region == "interior":
        base *= 0.9
    if region == "edge_2":
        base *= 1.2
    return base


def _persistence_mse(variable: str, lead: int, region: str) -> float:
    return _climatology_mse(variable, lead, region) * (1.5 + 0.2 * (lead // 6))


def _per_case_rows(seed: int, arm: str, lead: int, depth: int, kind: str | None) -> tuple:
    """Split the pooled target over cases without changing its mean."""
    counts = {6: 22, 12: 21, 24: 19, 48: 15, 72: 11}
    count = counts[lead]
    rows = []
    for region in REGIONS:
        for variable in VARIABLES:
            if kind is None:
                pooled = _model_mse(seed, arm, variable, lead, region, depth)
            elif kind == "climatology":
                pooled = _climatology_mse(variable, lead, region)
            else:
                pooled = _persistence_mse(variable, lead, region)
            # A ramp whose arithmetic mean equals the pooled value exactly enough
            # for the 1e-9 bound; values are spread by +/- one percent around it.
            values = [pooled * (0.99 + 0.02 * i / max(count - 1, 1)) for i in range(count)]
            correction = pooled - sum(values) / count
            values = [value + correction for value in values]
            for index, value in enumerate(values):
                row = {"sample_id": f"s{seed}_{arm}_{region}_{variable}_{lead}_{index}",
                       "init_time": "2016-02-17T06:00:00", "region": region, "variable": variable,
                       "lead_hours": lead, "mse": repr(value), "n_initializations": count}
                if kind is not None:
                    row["baseline_kind"] = kind
                rows.append(row)
    return rows, count


def _region_rows(seed: int, arm: str, lead: int, depth: int) -> list:
    counts = {6: 22, 12: 21, 24: 19, 48: 15, 72: 11}
    rows = []
    for region in REGIONS:
        for variable in VARIABLES:
            mse = _model_mse(seed, arm, variable, lead, region, depth)
            rows.append({"region": region, "variable": variable, "lead_hours": lead,
                         "mse": repr(mse), "rmse": repr(math.sqrt(mse)),
                         "climatology_mse": repr(_climatology_mse(variable, lead, region)),
                         "n_initializations": counts[lead]})
    return rows


def _baseline_rows(seed: int, arm: str, lead: int, depth: int) -> list:
    rows = []
    for region in REGIONS:
        for variable in VARIABLES:
            for kind, value in (("climatology", _climatology_mse(variable, lead, region)),
                                ("persistence", _persistence_mse(variable, lead, region))):
                rows.append({"baseline_kind": kind, "region": region, "variable": variable,
                             "lead_hours": lead, "mse": repr(value)})
    return rows


def _publish_archive(root: Path, monkeypatch) -> Path:
    archive = root / "archive"
    source = root / "source.nc"
    source.write_bytes(b"synthetic source bytes for identity")
    # A synthetic model/ package so the identity check is self-contained; the
    # real digest is separately asserted by test_real_archive_identity_is_stable.
    model_dir = root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "forecast.py").write_text("DIM = 192\n", encoding="utf-8")
    model_code = audit._model_code_digest(root)
    protocol = {
        "format": "r7-v2-remaining-protocol-v1", "stage": "C", "scientific_claim": False,
        "test_read": False,
        "data": {"store": "/nonexistent/cache.zarr", "data_identity": "a" * 64,
                 "val_data_identity": "b" * 64, "channels": list(VARIABLES),
                 "units": ["K"] * len(VARIABLES), "train_windows": 186, "val_windows": 22,
                 "test_read": False},
        "sources": {"source_path": str(source), "source_sha256": _sha256(source)},
        "sidecar": {"identity": "c" * 64},
        "arm_configs": {
            "old_ours": {"kind": "process", "mode": "l6", "updates": 400, "model_spec": {"dim": 192}},
            "process": {"kind": "process", "mode": "l6", "updates": 400, "model_spec": {"dim": 192}},
            "matched_generic": {"kind": "generic", "mode": "l6", "updates": 400,
                                "model_spec": {"dim": 192}},
        },
        "baseline_reporting": {"required": ["persistence", "climatology"]},
    }
    protocol["protocol_sha256"] = hashlib.sha256(_canonical(protocol).encode()).hexdigest()
    _write_json(archive / "protocol.json", protocol)
    checkpoints = {}
    for seed in audit.SEEDS:
        for arm in audit.ARMS:
            path = archive / f"seed{seed}" / "training" / arm / "update_0000400.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"weights {seed} {arm}".encode())
            checkpoints[(seed, arm)] = _sha256(path)
            _write_json(archive / "workers" / f"train_seed{seed}_{arm}_k4.json",
                        {"checkpoint_sha256": checkpoints[(seed, arm)]})
            for lead in LEADS:
                for depth in DEPTHS:
                    directory = archive / f"seed{seed}" / "evaluation" / arm / f"lead_{lead:03d}h" / f"k{depth}"
                    model_rows, _ = _per_case_rows(seed, arm, lead, depth, None)
                    _write_csv(directory / "per_case_metrics.csv", model_rows,
                               ["sample_id", "init_time", "region", "variable", "lead_hours",
                                "mse", "n_initializations"])
                    fields = ["sample_id", "init_time", "region", "variable", "lead_hours",
                              "mse", "n_initializations", "baseline_kind"]
                    base_rows = []
                    for kind in ("climatology", "persistence"):
                        kind_rows, _ = _per_case_rows(seed, arm, lead, depth, kind)
                        base_rows.extend(kind_rows)
                    _write_csv(directory / "baseline_per_case_metrics.csv", base_rows, fields)
                    _write_csv(directory / "region_metrics.csv",
                               _region_rows(seed, arm, lead, depth),
                               ["region", "variable", "lead_hours", "mse", "rmse",
                                "climatology_mse", "n_initializations"])
                    _write_csv(directory / "baseline_region_metrics.csv",
                               _baseline_rows(seed, arm, lead, depth),
                               ["baseline_kind", "region", "variable", "lead_hours", "mse"])
    pins = {}
    for path in sorted(archive.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.json"}:
            pins[str(path.relative_to(archive))] = _sha256(path)
    manifest = {"status": "stage-sealed", "scientific_claim": False,
                "identity": {"model_code_sha256": model_code},
                "files_sha256": pins, "files_digest": hashlib.sha256(_canonical(pins).encode()).hexdigest()}
    _write_json(archive / "artifact_manifest.json", manifest)
    (archive / "code_commit.txt").write_text("a" * 40 + "\n", encoding="utf-8")
    monkeypatch.setattr(audit, "REPO", root)
    return archive


@pytest.fixture()
def archive(tmp_path, monkeypatch):
    return _publish_archive(tmp_path, monkeypatch)


def test_report_is_read_only_and_identity_verifies(archive):
    report = audit.recompute(archive)
    assert report["format"] == "r7-s0-gap-audit-v1"
    assert report["scientific_claim"] is False and report["test_read"] is False
    assert report["gpu_hours"] == 0 and report["network_requests"] == 0
    assert report["identity"]["model_code_sha256_matches_manifest"] is True
    assert len(report["checkpoints_sha256"]) == 9
    assert report["recomputation"]["cells_checked"] == 3 * 3 * 3 * 5 * 3 * 17
    assert report["recomputation"]["all_within_bound"] is True


def test_primary_and_guard_arithmetic(archive):
    report = audit.recompute(archive)
    primary = report["summary"]["primary_seed_means"]
    assert primary["6"]["skill_positive_all_seeds"] is True
    assert primary["12"]["skill_positive_all_seeds"] is False
    assert primary["6"]["model_rmse_seed_mean"] < primary["6"]["climatology_rmse_seed_mean"]
    counts = report["summary"]["full_region_counts_K4_process"]
    total = (len(audit.SEEDS) * len(LEADS) * len(VARIABLES))
    assert (counts["full"]["positive"] + counts["full"]["nonpositive"]
            + counts["full"]["undefined"]) == total
    # The synthetic fixture only beats climatology at t2m/6h, like the real incumbent.
    assert counts["full"]["positive"] == len(audit.SEEDS)
    for guard in ("u10", "v10", "mslp"):
        values = report["summary"]["guard_variable_seed_means_K4_process"][guard]
        assert set(values) == {"6", "12", "24", "48", "72"}


def test_counterproof_protocol_digest(archive):
    payload = json.loads((archive / "protocol.json").read_text())
    payload["data"]["data_identity"] = "d" * 64
    _write_json(archive / "protocol.json", payload)
    with pytest.raises(ValueError, match="protocol_sha256 mismatch"):
        audit.recompute(archive)


def test_counterproof_manifest_digest(archive):
    payload = json.loads((archive / "artifact_manifest.json").read_text())
    payload["files_sha256"]["protocol.json"] = "e" * 64
    _write_json(archive / "artifact_manifest.json", payload)
    with pytest.raises(ValueError, match="files_digest mismatch"):
        audit.recompute(archive)


def test_counterproof_checkpoint_digest(archive):
    (archive / "seed41" / "training" / "process" / "update_0000400.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        audit.recompute(archive)


def test_counterproof_source_bytes(archive, tmp_path):
    (tmp_path / "source.nc").write_bytes(b"other bytes")
    with pytest.raises(ValueError, match="source_sha256"):
        audit.recompute(archive)


def test_counterproof_model_code_digest(archive, tmp_path):
    (tmp_path / "model" / "forecast.py").write_text("DIM = 200\n", encoding="utf-8")
    with pytest.raises(ValueError, match="model_code_sha256"):
        audit.recompute(archive)


def test_counterproof_case_count_mismatch(archive):
    path = archive / "seed41" / "evaluation" / "old_ours" / "lead_006h" / "k1" / "per_case_metrics.csv"
    rows = list(csv.DictReader(path.open()))
    _write_csv(path, rows[:-1], list(rows[0].keys()))
    with pytest.raises(ValueError, match="count disagrees"):
        audit.recompute(archive)


def test_counterproof_missing_baseline_kind(archive):
    path = (archive / "seed41" / "evaluation" / "old_ours" / "lead_006h" / "k1"
            / "baseline_per_case_metrics.csv")
    rows = [row for row in csv.DictReader(path.open()) if row["baseline_kind"] != "persistence"]
    _write_csv(path, rows, list(rows[0].keys()))
    with pytest.raises(ValueError, match="missing baseline kind"):
        audit.recompute(archive)


def test_sealed_test_manifest_is_refused(tmp_path):
    with pytest.raises(ValueError, match="sealed test.jsonl"):
        audit._plain_path(tmp_path / "manifests" / "test.jsonl")


def test_exclusive_write_refuses_existing_and_sources(archive, tmp_path):
    protected = {str(archive / "protocol.json")}
    with pytest.raises(ValueError, match="must not overwrite"):
        audit._write_exclusive(archive / "protocol.json", "x", protected)
    target = tmp_path / "new.csv"
    audit._write_exclusive(target, "a,b\n", protected)
    with pytest.raises(FileExistsError):
        audit._write_exclusive(target, "a,b\n", protected)


def test_cli_writes_table_and_report(archive, tmp_path):
    table, report_path = tmp_path / "gap.csv", tmp_path / "audit.json"
    code = audit.main(["--archive", str(archive), "--table", str(table), "--report", str(report_path)])
    assert code == 0
    rows = list(csv.DictReader(table.open()))
    assert len(rows) == 6885
    report = json.loads(report_path.read_text())
    assert report["gap_table"]["sha256"] == _sha256(table)
    assert report["gap_table"]["rows"] == 6885
    assert all(row["variable"] in VARIABLES for row in rows)


def test_cli_error_is_nonzero_without_partial_output(archive, tmp_path):
    (archive / "code_commit.txt").write_text("not a commit\n", encoding="utf-8")
    assert audit.main(["--archive", str(archive), "--report", str(tmp_path / "r.json")]) == 2
    assert not (tmp_path / "r.json").exists()


def test_tool_has_no_network_or_write_imports():
    """Counterproof: only the standard library may be imported by the audit tool."""
    tree = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    allowed = {"__future__", "argparse", "csv", "hashlib", "io", "json", "math", "os",
               "statistics", "sys", "pathlib"}
    assert modules <= allowed, f"unexpected imports: {sorted(modules - allowed)}"
    assert "torch" not in " ".join(sorted(modules))
