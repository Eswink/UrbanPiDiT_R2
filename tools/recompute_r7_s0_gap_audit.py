"""S0 read-only gap audit for the main-model climatology campaign.

Verifies the actual-C identity chain from the frozen archive alone (protocol
digest, artifact-manifest digest, model-code digest, checkpoint digests, data
identity, source bytes), independently recomputes every pooled region cell
from the per-case CSVs, and tabulates the D1 gap table: the trained arms
against the zero-training train-only climatology and persistence baselines
over all 17 variables, five leads, three regions, three seeds and the three
inference depths.

Read-only: no training, no GPU, no network, and no modification of any archive
byte; the only writes are the exclusive --table/--report paths.  The tool is a
mechanical audit: it computes gap arithmetic and identity agreement, and never
declares a scientific verdict.

Recomputation tolerance.  The per-case mean and the archived pooled value sum
the same float64 quantities in a different order, so exact bit equality is not
expected; ``AGREEMENT_TOL`` is a float64 summation-order bound for that
cross-check, not a scientific criterion, and the observed maximum deviation is
reported.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = REPO / "outputs/r7_v2_comparison_20261004_attempt01"
LEADS = (6, 12, 24, 48, 72)
REGIONS = ("full", "interior", "edge_2")
DEPTHS = (1, 2, 4)
SEEDS = (41, 42, 43)
ARMS = ("old_ours", "process", "matched_generic")
VARIABLES = (
    "t2m", "u10", "v10", "mslp", "z850", "t850", "q850", "u850", "v850",
    "z500", "t500", "q500", "u500", "v500", "z250", "u250", "v250",
)
BASELINE_KINDS = ("climatology", "persistence")
INCUMBENT_ARM = "process"
INCUMBENT_DEPTH = 4
PRIMARY_VARIABLE = "t2m"
AGREEMENT_TOL = 1e-9
LIMITATIONS = (
    "Mechanical source-agreement, identity and gap arithmetic only; no scientific",
    "significance, SOTA or verdict field is produced by this tool.",
    "The archive is validation-only on a two-month 2016 winter segment; one season",
    "cannot support annual or cross-year claims.",
    "The incumbent process/matched_generic separation is unresolved by actual C and",
    "is not re-adjudicated here.",
    "K1/K2 are inference-depth probes of the same K4-trained checkpoints, not",
    "independently trained models; per-K gaps are not equal-compute comparisons.",
    "The per-case recomputation cross-check uses a float64 summation-order bound",
    "(1e-9 relative); the archived pooled rows remain the authoritative values.",
    "Source hashes identify the bytes read in THIS archive, not an external trusted",
    "pin; the sealed test manifest is never opened or inspected.",
    "Climatology is the train-only month-hour grid mean of this two-month segment,",
    "not a WeatherBench2 reproduction and not a full annual-cycle climatology.",
)


def _plain_path(path: Path) -> Path:
    """Reject symlinks and the sealed test manifest, without following either."""
    path = Path(os.path.abspath(path))
    if "test.jsonl" in path.parts:
        raise ValueError("sealed test.jsonl paths are forbidden, even for metadata")
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"symlink source/output paths are forbidden: {part}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _json(text: str) -> dict:
    result = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(result, dict):
        raise ValueError("JSON document must be an object")
    return result


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{label}: expected a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label}: expected a finite number")
    return number


def _read_csv(path: Path, required: set) -> list:
    text = path.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fields = reader.fieldnames or []
    if len(set(fields)) != len(fields) or not required.issubset(fields):
        raise ValueError(f"missing or duplicate CSV columns in {path.name}: {sorted(required)}")
    rows = list(reader)
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"empty or malformed CSV rows in {path.name}")
    return rows


def _model_code_digest(root: Path) -> str:
    """Mirror of training/r7_experiment.py:model_code_digest (cross-checked by test)."""
    directory = (root / "model").resolve()
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*.py")):
        if "legacy" in str(path.relative_to(directory)):
            continue
        digest.update(path.relative_to(directory).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _identity(archive: Path, root: Path) -> dict:
    protocol_raw = _plain_path(archive / "protocol.json").read_bytes()
    protocol = _json(protocol_raw.decode("utf-8"))
    body = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if hashlib.sha256(_canonical(body).encode()).hexdigest() != protocol["protocol_sha256"]:
        raise ValueError("protocol_sha256 mismatch")
    if protocol.get("scientific_claim") is not False or protocol["data"].get("test_read") is not False:
        raise ValueError("protocol must be scientific_claim false and test_read false")
    manifest_raw = _plain_path(archive / "artifact_manifest.json").read_bytes()
    manifest = _json(manifest_raw.decode("utf-8"))
    if hashlib.sha256(_canonical(manifest["files_sha256"]).encode()).hexdigest() != manifest["files_digest"]:
        raise ValueError("artifact manifest files_digest mismatch")
    if manifest.get("status") != "stage-sealed" or manifest.get("scientific_claim") is not False:
        raise ValueError("artifact manifest must be stage-sealed and non-scientific")
    recorded = manifest.get("identity", {})
    recomputed_model_code = _model_code_digest(root)
    if recorded.get("model_code_sha256") != recomputed_model_code:
        raise ValueError("model_code_sha256 does not match the working tree model/ package")
    code_commit = _plain_path(archive / "code_commit.txt").read_text(encoding="utf-8").strip()
    if len(code_commit) != 40 or any(c not in "0123456789abcdef" for c in code_commit):
        raise ValueError("code_commit.txt must be a full lowercase commit SHA")
    source_path = Path(protocol["sources"]["source_path"])
    if not source_path.is_file():
        raise ValueError(f"declared source bytes missing: {source_path}")
    recorded_source = protocol["sources"]["source_sha256"]
    if _sha256(source_path) != recorded_source:
        raise ValueError("source_sha256 does not match the declared source bytes")
    return {
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_source_sha256": protocol["sources"]["source_sha256"],
        "artifact_files_digest": manifest["files_digest"],
        "artifact_pinned_files": len(manifest["files_sha256"]),
        "manifest_identity": dict(recorded),
        "model_code_sha256_recomputed": recomputed_model_code,
        "model_code_sha256_matches_manifest": True,
        "code_commit": code_commit,
        "data_identity": protocol["data"]["data_identity"],
        "val_data_identity": protocol["data"]["val_data_identity"],
        "store": protocol["data"]["store"],
        "channels": list(protocol["data"]["channels"]),
        "units": list(protocol["data"]["units"]),
        "train_windows": protocol["data"]["train_windows"],
        "val_windows": protocol["data"]["val_windows"],
        "sidecar_identity": protocol["sidecar"]["identity"],
        "arm_configs": {
            arm: {"kind": config["kind"], "updates": config["updates"],
                  "mode": config["mode"], "model_spec": config["model_spec"]}
            for arm, config in protocol["arm_configs"].items()
        },
        "baseline_reporting": dict(protocol["baseline_reporting"]),
    }


def _checkpoint_digests(archive: Path) -> dict:
    """Independently hash the nine endpoints and compare with worker receipts."""
    result = {}
    for seed in SEEDS:
        for arm in ARMS:
            receipt = _plain_path(archive / "workers" / f"train_seed{seed}_{arm}_k4.json")
            payload = _json(receipt.read_text(encoding="utf-8"))
            recorded = payload.get("checkpoint_sha256")
            if not isinstance(recorded, str) or len(recorded) != 64:
                raise ValueError(f"worker receipt missing checkpoint_sha256: {receipt.name}")
            checkpoint = _plain_path(archive / f"seed{seed}" / "training" / arm / "update_0000400.pt")
            actual = _sha256(checkpoint)
            if actual != recorded:
                raise ValueError(f"checkpoint digest mismatch: {seed}/{arm}")
            result[f"seed{seed}/{arm}"] = {"sha256": actual, "bytes": checkpoint.stat().st_size}
    return result


def _pool_cells(rows: list, region_field: str) -> dict:
    grouped: dict = {}
    for row in rows:
        key = (row[region_field], row["variable"])
        grouped.setdefault(key, []).append(_number(row["mse"], "per-case mse"))
    return {key: statistics.fmean(values) for key, values in grouped.items()}


def _recompute_cell(archive: Path, seed: int, arm: str, lead: int, depth: int) -> dict:
    directory = archive / f"seed{seed}" / "evaluation" / arm / f"lead_{lead:03d}h" / f"k{depth}"
    model_rows = _read_csv(_plain_path(directory / "per_case_metrics.csv"), {"region", "variable", "mse"})
    base_rows = _read_csv(_plain_path(directory / "baseline_per_case_metrics.csv"),
                          {"baseline_kind", "region", "variable", "mse"})
    model_pooled = _pool_cells(model_rows, "region")
    reference = _read_csv(_plain_path(directory / "region_metrics.csv"),
                          {"region", "variable", "rmse", "mse", "climatology_mse", "n_initializations"})
    record = {}
    for row in reference:
        key = (row["region"], row["variable"])
        if key not in model_pooled:
            raise ValueError(f"region row without per-case rows: {directory} {key}")
        recorded = _number(row["mse"], "region mse")
        recomputed = model_pooled[key]
        deviation = abs(recomputed - recorded) / recorded if recorded else abs(recomputed)
        cases = len([r for r in model_rows if (r["region"], r["variable"]) == key])
        if cases != int(row["n_initializations"]):
            raise ValueError(f"per-case count disagrees with n_initializations: {directory} {key}")
        record[key] = {
            "model_mse": recorded,
            "recomputed_model_mse": recomputed,
            "model_rmse": _number(row["rmse"], "region rmse"),
            "climatology_mse": _number(row["climatology_mse"], "region climatology_mse"),
            "n_initializations": cases,
            "deviation": deviation,
        }
    baseline = {}
    for kind in BASELINE_KINDS:
        kind_rows = [row for row in base_rows if row["baseline_kind"] == kind]
        if not kind_rows:
            raise ValueError(f"missing baseline kind {kind}: {directory}")
        baseline[kind] = _pool_cells(kind_rows, "region")
    base_reference = _read_csv(_plain_path(directory / "baseline_region_metrics.csv"),
                               {"baseline_kind", "region", "variable", "mse"})
    for row in base_reference:
        key, kind = (row["region"], row["variable"]), row["baseline_kind"]
        if kind not in baseline or key not in baseline[kind]:
            raise ValueError(f"baseline row without per-case rows: {directory} {kind} {key}")
        recorded = _number(row["mse"], "baseline region mse")
        recomputed = baseline[kind][key]
        deviation = abs(recomputed - recorded) / recorded if recorded else abs(recomputed)
        target = record.get(key)
        if target is None:
            raise ValueError(f"baseline key without model row: {directory} {key}")
        target.setdefault("baselines", {})[kind] = {"mse": recorded, "recomputed_mse": recomputed,
                                                    "rmse": math.sqrt(recorded)}
        target["deviation"] = max(target["deviation"], deviation)
    return record


def _skill(model_mse: float, reference_mse: float):
    if not reference_mse > 0:
        return None, "undefined_zero_reference_energy"
    return 1.0 - model_mse / reference_mse, "defined"


def _gap_rows(archive: Path) -> tuple:
    rows, deviations = [], []
    for seed in SEEDS:
        for arm in ARMS:
            for lead in LEADS:
                for depth in DEPTHS:
                    record = _recompute_cell(archive, seed, arm, lead, depth)
                    for (region, variable), cell in sorted(record.items()):
                        climatology = cell["baselines"]["climatology"]["mse"]
                        persistence = cell["baselines"]["persistence"]["mse"]
                        climatology_skill, climatology_status = _skill(cell["model_mse"], climatology)
                        persistence_skill, persistence_status = _skill(cell["model_mse"], persistence)
                        rows.append({
                            "seed": seed, "arm": arm, "K": depth, "lead_hours": lead,
                            "region": region, "variable": variable,
                            "model_rmse": cell["model_rmse"], "model_mse": cell["model_mse"],
                            "climatology_rmse": math.sqrt(climatology),
                            "climatology_mse": climatology,
                            "persistence_rmse": math.sqrt(persistence),
                            "persistence_mse": persistence,
                            "mse_skill_climatology": climatology_skill,
                            "mse_skill_climatology_status": climatology_status,
                            "mse_skill_persistence": persistence_skill,
                            "mse_skill_persistence_status": persistence_status,
                            "n_initializations": cell["n_initializations"],
                        })
                        deviations.append(cell["deviation"])
    return rows, deviations


def _seed_means(rows: list, arm: str, depth: int, region: str, variable: str) -> dict:
    selected = [row for row in rows if row["arm"] == arm and row["K"] == depth
                and row["region"] == region and row["variable"] == variable]
    result = {}
    for lead in LEADS:
        cells = [row for row in selected if row["lead_hours"] == lead]
        if len(cells) != len(SEEDS):
            raise ValueError(f"gap rows missing seeds for {arm}/K{depth}/{region}/{variable}/{lead}h")
        result[str(lead)] = {
            "model_rmse_seed_mean": statistics.fmean(cell["model_rmse"] for cell in cells),
            "climatology_rmse_seed_mean": statistics.fmean(cell["climatology_rmse"] for cell in cells),
            "persistence_rmse_seed_mean": statistics.fmean(cell["persistence_rmse"] for cell in cells),
            "skill_climatology_seed_mean": statistics.fmean(cell["mse_skill_climatology"] for cell in cells),
            "skill_positive_all_seeds": all((cell["mse_skill_climatology"] or 0.0) > 0 for cell in cells),
            "seeds": list(SEEDS),
        }
    return result


def _summary(rows: list) -> dict:
    primary = _seed_means(rows, INCUMBENT_ARM, INCUMBENT_DEPTH, "full", PRIMARY_VARIABLE)
    incumbent = [row for row in rows if row["arm"] == INCUMBENT_ARM and row["K"] == INCUMBENT_DEPTH]
    count = {"cells": len(incumbent)}
    for region in REGIONS:
        subset = [row for row in incumbent if row["region"] == region]
        count[region] = {
            "positive": sum(1 for row in subset if (row["mse_skill_climatology"] or -1) > 0),
            "nonpositive": sum(1 for row in subset if row["mse_skill_climatology"] is not None
                               and row["mse_skill_climatology"] <= 0),
            "undefined": sum(1 for row in subset if row["mse_skill_climatology"] is None),
        }
    per_variable = {}
    for variable in VARIABLES:
        cells = [row for row in incumbent if row["region"] == "full" and row["variable"] == variable]
        per_variable[variable] = {
            "positive_cells": sum(1 for row in cells if (row["mse_skill_climatology"] or -1) > 0),
            "nonpositive_cells": sum(1 for row in cells if row["mse_skill_climatology"] is not None
                                     and row["mse_skill_climatology"] <= 0),
            "skill_by_lead_seed_mean": {
                str(lead): statistics.fmean(
                    row["mse_skill_climatology"] for row in cells if row["lead_hours"] == lead)
                for lead in LEADS},
        }
    guard = {variable: _seed_means(rows, INCUMBENT_ARM, INCUMBENT_DEPTH, "full", variable)
             for variable in ("u10", "v10", "mslp", "t2m")}
    return {
        "incumbent_arm": INCUMBENT_ARM, "incumbent_depth": INCUMBENT_DEPTH,
        "primary_variable": PRIMARY_VARIABLE,
        "primary_seed_means": primary,
        "full_region_counts_K4_process": count,
        "full_region_skill_by_variable_K4_process": per_variable,
        "guard_variable_seed_means_K4_process": guard,
        "arms_note": ("old_ours is the predecessor; process is the V2 package designated "
                      "incumbent for this campaign; matched_generic is the same-information "
                      "equivalence control and not a competitor to defeat"),
    }


def _write_exclusive(path: Path, text: str, protected: set) -> None:
    resolved = _plain_path(path)
    if str(resolved) in protected:
        raise ValueError("output must not overwrite a read-only source")
    with resolved.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _audit(archive: Path) -> tuple:
    """Return (report, gap_table_rows); the single I/O pass over the archive."""
    archive = _plain_path(Path(archive))
    if not archive.is_dir():
        raise ValueError(f"archive directory missing: {archive}")
    identity = _identity(archive, REPO)
    checkpoints = _checkpoint_digests(archive)
    rows, deviations = _gap_rows(archive)
    if not all(row["variable"] in VARIABLES for row in rows):
        raise ValueError("gap table contains an unexpected variable")
    expected = len(SEEDS) * len(ARMS) * len(DEPTHS) * len(LEADS) * len(REGIONS) * len(VARIABLES)
    if len(rows) != expected:
        raise ValueError(f"gap table row count {len(rows)} != expected {expected}")
    script = _plain_path(Path(__file__))
    return {
        "format": "r7-s0-gap-audit-v1", "scientific_claim": False, "test_read": False,
        "gpu_hours": 0, "network_requests": 0,
        "archive": str(archive), "script": {"path": str(script), "sha256": _sha256(script)},
        "inputs": {
            "protocol_sha256": identity["protocol_sha256"],
            "artifact_files_digest": identity["artifact_files_digest"],
            "artifact_pinned_files": identity["artifact_pinned_files"],
            "code_commit": identity["code_commit"],
            "model_code_sha256": identity["model_code_sha256_recomputed"],
            "data_identity": identity["data_identity"],
            "val_data_identity": identity["val_data_identity"],
            "source_sha256": identity["protocol_source_sha256"],
        },
        "identity": identity, "checkpoints_sha256": checkpoints,
        "recomputation": {
            "cells_checked": len(rows),
            "per_case_files_read": len(SEEDS) * len(ARMS) * len(LEADS) * len(DEPTHS),
            "agreement_bound_relative": AGREEMENT_TOL,
            "max_relative_deviation": max(deviations),
            "all_within_bound": max(deviations) <= AGREEMENT_TOL,
            "scope": ("archived pooled MSE vs independent mean of the archived per-case MSE "
                      "over identical cases; float64 summation-order bound only"),
        },
        "summary": _summary(rows),
        "limitations": list(LIMITATIONS),
    }, rows


def recompute(archive: Path = DEFAULT_ARCHIVE) -> dict:
    """Report-only entry point used by tests and callers that need no CSV table."""
    return _audit(archive)[0]


def _format_gap_table(rows: list) -> str:
    fields = ["seed", "arm", "K", "lead_hours", "region", "variable", "model_rmse", "model_mse",
              "climatology_rmse", "climatology_mse", "persistence_rmse", "persistence_mse",
              "mse_skill_climatology", "mse_skill_climatology_status", "mse_skill_persistence",
              "mse_skill_persistence_status", "n_initializations"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--table", type=Path, help="exclusive new CSV path for the full gap table")
    parser.add_argument("--report", type=Path, help="exclusive new JSON path for the audit report")
    args = parser.parse_args(argv)
    try:
        report, rows = _audit(args.archive)
        protected = {report["script"]["path"]}
        if args.table is not None:
            _write_exclusive(args.table, _format_gap_table(rows), protected)
            report["gap_table"] = {"path": str(_plain_path(args.table)),
                                   "sha256": _sha256(_plain_path(args.table)), "rows": len(rows)}
        if args.report is not None:
            blocked = protected | ({str(_plain_path(args.table))} if args.table else set())
            _write_exclusive(args.report, json.dumps(report, indent=2, sort_keys=True) + "\n", blocked)
        if args.table is None and args.report is None:
            sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        print(f"S0 gap audit error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
