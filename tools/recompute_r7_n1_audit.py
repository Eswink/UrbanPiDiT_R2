"""Read-only N1/D3 arithmetic audit; never load tensors, checkpoints or test.

Only six named archive/manifest files and this script are read. Seed spread is
an observed within-arm range, not uncertainty or a confidence interval. No
scientific classification, comparator sign rule, or new endpoint is evaluated.
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
from collections import Counter
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = REPO / "outputs/r7_72_rw_b_subtraction"
DEFAULT_MANIFESTS = REPO / "outputs/r7_m2_segment/store/manifests"
SEEDS = (41, 42)
LEADS = (6, 12, 24, 48, 72)
AUDIT_LEADS = (48, 72)
RW_A = "process_spacetime_rwa"
RW_B = "process_local_solver"
NO_RECURRENCE = "process_local_solver_no_recurrence"
ARMS = (RW_A, RW_B, "process_local_solver_no_gate_proposal", NO_RECURRENCE)
FOCUS_ARMS = (RW_B, NO_RECURRENCE)
PRIMARY_KEYS = {RW_B: "round_reference", NO_RECURRENCE: "primary_question"}
REASONING_STEPS = 3
FINAL_WEIGHT = Fraction(2)
LIMITATIONS = (
    "Mechanical source-agreement and arithmetic checks only; no scientific classification, "
    "new criterion, sign-rule verdict, training, evaluation or replay.",
    "Two fixed seeds (41, 42); max-minus-min seed spread is a range, not a confidence "
    "interval, significance test or uncertainty estimate.",
    "Only t2m is numerically cross-checked; comparison outputs are restricted to the "
    "already registered 48/72 h leads. Other leads only check archive/case consistency.",
    "Source hashes identify the bytes read, not an external trusted pin. Archived dataset "
    "identity and case digests are cross-checked, not rebuilt from weather arrays.",
    "Validation windows are inferred from manifest target-time membership, not cache "
    "arrays. The sealed test file is never opened or inspected, including metadata.",
    "No cache metadata is read: train-only normalization and eight climatology buckets "
    "are not independently quantified by this tool.",
    "final_weight=2 is a frozen audit input, not an inferred or invented protocol field. "
    "Fraction weights are exact formal weights, not a bitwise reconstruction of FP32 tensors.",
    "Reproducibility covers deterministic table arithmetic/JSON for identical source and "
    "script bytes in the same Python environment; no training bit-reproducibility claim.",
)


def _plain_path(path: Path) -> Path:
    """Reject symlinks without following them, including aliases of sealed input."""
    path = Path(os.path.abspath(path))
    if "test.jsonl" in path.parts:
        raise ValueError("sealed test.jsonl paths are forbidden, even for metadata")
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"symlink source/output paths are forbidden: {part}")
    return path


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


def _csv(text: str, required: set[str]) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fields = reader.fieldnames or []
    if len(set(fields)) != len(fields) or not required.issubset(fields):
        raise ValueError(f"missing or duplicate CSV columns; required: {sorted(required)}")
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("malformed CSV row")
    return rows


def _load_sources(archive: Path, manifests: Path) -> tuple[dict, dict]:
    paths = {
        "rmse_table": archive / "rmse_table.csv",
        "case_table": archive / "case_table.csv",
        "paired_comparison": archive / "paired_comparison.json",
        "protocol": archive / "seed41/protocol.json",
        "train_manifest": manifests / "train.jsonl",
        "val_manifest": manifests / "val.jsonl",
    }
    texts, sources = {}, {}
    for role, supplied in paths.items():
        path = _plain_path(supplied)
        raw = path.read_bytes()
        sources[role] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                         "bytes": len(raw)}
        texts[role] = raw.decode("utf-8")
    return texts, sources


def _number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{label}: expected a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label}: expected a finite number")
    return number


def _integer(value, label: str) -> int:
    number = _number(value, label)
    if not number.is_integer():
        raise ValueError(f"{label}: expected an integer")
    return int(number)


def _key(row: dict) -> tuple[int, str, int]:
    seed = _integer(row["seed"], "seed")
    lead = _integer(row["lead_hours"], "lead_hours")
    arm = row["arm"]
    if seed not in SEEDS or lead not in LEADS or arm not in ARMS:
        raise ValueError(f"unexpected seed/arm/lead: {(seed, arm, lead)}")
    return seed, arm, lead


def _seeds(value, label: str) -> None:
    if not isinstance(value, list) or any(type(seed) is not int for seed in value):
        raise ValueError(f"{label}: seeds must be integer ids 41, 42")
    if sorted(value) != list(SEEDS):
        raise ValueError(f"{label}: missing, duplicate or unexpected seeds; require 41, 42")


def _seed_map(value, label: str) -> dict:
    if not isinstance(value, dict) or set(value) != {str(seed) for seed in SEEDS}:
        raise ValueError(f"{label}: missing or unexpected seed keys; require 41, 42")
    return value


def _equal(actual, expected: float, label: str) -> None:
    # Exact float round-trip/arithmetic agreement, not a scientific tolerance.
    if _number(actual, label) != expected:
        raise ValueError(f"{label}: arithmetic/source mismatch ({actual!r} != {expected!r})")


def _protocol(payload: dict) -> dict:
    content = {key: value for key, value in payload.items() if key != "protocol_sha256"}
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != payload["protocol_sha256"]:
        raise ValueError("protocol_sha256 mismatch")
    _seeds(payload["seeds"], "protocol")
    if payload["frozen_before_any_step"] is not True or payload["scientific_claim"] is not False:
        raise ValueError("protocol must be frozen and scientific_claim false")
    if payload["data"]["test_read"] is not False:
        raise ValueError("protocol test_read must be false")
    controls = payload["shared_controls"]
    if type(controls["reasoning_steps"]) is not int or controls["reasoning_steps"] != REASONING_STEPS:
        raise ValueError("protocol reasoning_steps must equal frozen input 3")
    _equal(controls["process_weight"], 0.0, "protocol process_weight")
    if controls["validation_lead_hours"] != [6] or controls["evaluation_leads_hours"] != list(LEADS):
        raise ValueError("protocol lead declarations disagree with the archived registration")
    if controls["selection_split"] != "val only; test is sealed and never read":
        raise ValueError("protocol selection_split must be val only with test sealed")
    if "final_weight" in controls:
        _equal(controls["final_weight"], float(FINAL_WEIGHT), "protocol final_weight")
    return {"protocol_sha256": payload["protocol_sha256"],
            "dataset_identity": payload["data"]["data_identity"],
            "reasoning_steps": controls["reasoning_steps"],
            "process_weight": controls["process_weight"],
            "validation_lead_hours": controls["validation_lead_hours"],
            "evaluation_leads_hours": controls["evaluation_leads_hours"],
            "final_weight_field_present": "final_weight" in controls,
            "final_weight_field_value": controls.get("final_weight")}


def _manifest(text: str, split: str) -> tuple[dict, list[dict]]:
    records, ids, distribution = [], set(), Counter()
    for line in text.splitlines():
        if not line.strip():
            continue
        row = _json(line)
        if row["split"] != split:
            raise ValueError(f"{split} manifest split mismatch")
        sample_id = row["sample_id"]
        if not isinstance(sample_id, str) or not sample_id or sample_id in ids:
            raise ValueError(f"{split} manifest missing/duplicate sample_id")
        ids.add(sample_id)
        lead = _integer(row["lead_time_hours"], f"{split} manifest lead_time_hours")
        if lead <= 0:
            raise ValueError("manifest lead_time_hours must be positive")
        init = datetime.fromisoformat(row["init_time"])
        target = datetime.fromisoformat(row["target_time"])
        if target - init != timedelta(hours=lead):
            raise ValueError(f"{split} manifest lead/time mismatch")
        distribution[str(lead)] += 1
        records.append({"sample_id": sample_id, "init": init, "target": target})
    if not records:
        raise ValueError(f"{split} manifest is empty")
    if len({row["init"] for row in records}) != len(records):
        raise ValueError(f"{split} manifest duplicate initialization time")
    summary = {"split": split, "count": len(records), "lead_distribution_hours": dict(distribution),
               "plus_6h_only": set(distribution) == {"6"}}
    return summary, records


def _cases(rows: list[dict], val: list[dict]) -> tuple[dict, dict]:
    targets = {row["target"] for row in val}
    available = {lead: sum(row["init"] + timedelta(hours=lead) in targets for row in val)
                 for lead in LEADS}
    cases = {}
    for row in rows:
        key = _key(row)
        if key in cases:
            raise ValueError(f"duplicate case_table seed/arm/lead: {key}")
        if row["split"] != "val" or row["test_read"] != "False":
            raise ValueError("case_table must be val and test_read False")
        count = _integer(row["n_evaluated"], "case_table n_evaluated")
        possible = _integer(row["n_available_windows"], "case_table n_available_windows")
        if count <= 0 or count != possible or possible != available[key[2]]:
            raise ValueError(f"case_table count disagrees with available val manifest windows: {key}")
        cases[key] = count
    expected = {(seed, arm, lead) for seed in SEEDS for arm in ARMS for lead in LEADS}
    if set(cases) != expected:
        raise ValueError("case_table missing seed/arm/lead rows")
    report = {str(lead): {"split": "val", "test_read": False,
                         "n_available_windows": available[lead],
                         "n_evaluated_per_arm_per_seed": cases[(SEEDS[0], RW_A, lead)]}
              for lead in LEADS}
    return cases, report


def _metrics(rows: list[dict], cases: dict) -> dict:
    values = {}
    for row in rows:
        if row["variable"] != "t2m":
            continue
        key = _key(row)
        if row["unit"] != "K":
            raise ValueError("t2m unit must be K")
        if key in values:
            raise ValueError(f"duplicate t2m metric seed/arm/lead: {key}")
        rmse = _number(row["rmse"], "t2m rmse")
        if rmse < 0 or _integer(row["n_initializations"], "n_initializations") != cases[key]:
            raise ValueError("t2m rmse/count disagrees with case_table")
        values[key] = rmse
    if set(values) != set(cases):
        raise ValueError("t2m metrics missing seed/arm/lead rows")
    return values


def _paired_table(paired: dict, values: dict, cases: dict) -> int:
    seen, digests = set(), {}
    for row in paired["table"]:
        if row["variable"] != "t2m":
            continue
        arm, lead = row["arm"], _integer(row["lead_hours"], "paired lead_hours")
        if arm not in ARMS or lead not in LEADS or row["depth"] != 0:
            raise ValueError("unexpected paired t2m arm/lead/depth")
        if (arm, lead) in seen:
            raise ValueError("duplicate paired t2m table row")
        seen.add((arm, lead))
        if row["unit"] != "K":
            raise ValueError("paired t2m unit must be K")
        _seeds(row["seeds"], "paired table")
        rmse = _seed_map(row["seed_rmse"], "paired seed_rmse")
        counts = _seed_map(row["per_seed_n_initializations"], "paired counts")
        identities = _seed_map(row["case_set_digests"], "paired case digests")
        if row["case_identity"] != "exact":
            raise ValueError("paired case_identity must be exact")
        for seed in SEEDS:
            key, seed_id = (seed, arm, lead), str(seed)
            _equal(rmse[seed_id], values[key], "paired seed_rmse")
            if _integer(counts[seed_id], "paired count") != cases[key]:
                raise ValueError("paired count mismatch")
            digest = identities[seed_id]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid paired case digest")
            if digest != digests.setdefault(lead, digest):
                raise ValueError("paired case digest mismatch between arms/seeds")
        mean = statistics.fmean(values[(seed, arm, lead)] for seed in SEEDS)
        _equal(row["rmse_seed_mean"], mean, "paired rmse_seed_mean")
    if seen != {(arm, lead) for arm in ARMS for lead in LEADS}:
        raise ValueError("paired table missing seed/arm/lead rows")
    return len(seen)


def _paired_checks(paired: dict, protocol: dict, values: dict, cases: dict) -> dict:
    identity = paired["identity"]
    if identity["evaluation_split"] != "val" or paired["scientific_claim"] is not False:
        raise ValueError("paired comparison must be val and scientific_claim false")
    if not protocol["dataset_identity"] or identity["dataset_identity"] != protocol["dataset_identity"]:
        raise ValueError("paired/protocol dataset identity mismatch")
    table_count = _paired_table(paired, values, cases)
    for focus in FOCUS_ARMS:
        pair = paired["pairs"][f"{focus} - {RW_A}"]
        if pair["focus_arm"] != focus or pair["baseline_arm"] != RW_A:
            raise ValueError("paired focus/baseline identity mismatch")
        for lead in LEADS:
            cell = pair["cells"][f"{lead}h|t2m"]
            primary = paired["primary"][PRIMARY_KEYS[focus]]["per_lead"][str(lead)]
            if cell["unit"] != "K":
                raise ValueError("paired cell unit must be K")
            delta = _seed_map(cell["seed_deltas"], "paired cell seed_deltas")
            primary_delta = _seed_map(primary["seed_deltas"], "paired primary seed_deltas")
            computed = [values[(seed, focus, lead)] - values[(seed, RW_A, lead)] for seed in SEEDS]
            for seed, difference in zip(SEEDS, computed):
                _equal(delta[str(seed)], difference, "paired cell delta")
                _equal(primary_delta[str(seed)], difference, "paired primary delta")
            _equal(primary["delta_seed_mean"], statistics.fmean(computed), "paired primary mean delta")
    return {"dataset_identity_agrees": True, "t2m_csv_rows_checked": len(values),
            "t2m_paired_table_rows_checked": table_count,
            "paired_cells_and_primary_means_checked": len(FOCUS_ARMS) * len(LEADS),
            "scope": "data identity/source agreement and arithmetic only; no outcome fields interpreted"}


def _ratio(delta: float, spread: float) -> dict:
    if spread == 0:
        return {"value": None, "defined": False, "undefined_reason": "zero seed spread (0 K)"}
    return {"value": _number(delta / spread, "delta/spread ratio"),
            "defined": True, "undefined_reason": None}


def _comparison(values: dict, focus: str, lead: int) -> dict:
    focus_values = {str(seed): values[(seed, focus, lead)] for seed in SEEDS}
    baseline_values = {str(seed): values[(seed, RW_A, lead)] for seed in SEEDS}
    delta = {str(seed): focus_values[str(seed)] - baseline_values[str(seed)] for seed in SEEDS}
    mean = statistics.fmean(delta.values())
    focus_spread = max(focus_values.values()) - min(focus_values.values())
    baseline_spread = max(baseline_values.values()) - min(baseline_values.values())
    return {"focus_arm": focus, "baseline_arm": RW_A, "variable": "t2m", "lead_hours": lead,
            "unit": "K", "focus_seed_rmse_K": focus_values, "baseline_seed_rmse_K": baseline_values,
            "focus_seed_mean_rmse_K": statistics.fmean(focus_values.values()),
            "baseline_seed_mean_rmse_K": statistics.fmean(baseline_values.values()),
            "seed_deltas_K": delta, "mean_delta_K": mean,
            "focus_seed_spread_K": focus_spread, "baseline_seed_spread_K": baseline_spread,
            "mean_delta_over_focus_spread": _ratio(mean, focus_spread),
            "mean_delta_over_baseline_spread": _ratio(mean, baseline_spread)}


def _weights(protocol: dict) -> dict:
    raw = [Fraction(1) + (FINAL_WEIGHT - 1) * Fraction(step, REASONING_STEPS)
           for step in range(REASONING_STEPS + 1)]
    normalized = [weight / sum(raw) for weight in raw]
    return {"axis": "internal reasoning K=0..3, not physical forecast lead hours",
            "reasoning_steps": REASONING_STEPS, "final_weight_frozen_input": int(FINAL_WEIGHT),
            "final_weight_source": "frozen N1 audit input; not claimed as a protocol field",
            "protocol_final_weight_field_present": protocol["final_weight_field_present"],
            "protocol_final_weight_field_value": protocol["final_weight_field_value"],
            "unnormalized_weights_exact": [str(weight) for weight in raw],
            "normalized_weights_exact": [str(weight) for weight in normalized],
            "normalized_sum_exact": str(sum(normalized)),
            "process_weight_from_protocol": protocol["process_weight"]}


def recompute(archive: Path = DEFAULT_ARCHIVE, manifests: Path = DEFAULT_MANIFESTS) -> dict:
    """Compute a report solely from the explicitly named CSV/JSON/JSONL inputs."""
    texts, sources = _load_sources(Path(archive), Path(manifests))
    protocol = _protocol(_json(texts["protocol"]))
    train_summary, train = _manifest(texts["train_manifest"], "train")
    val_summary, val = _manifest(texts["val_manifest"], "val")
    if {row["sample_id"] for row in train} & {row["sample_id"] for row in val}:
        raise ValueError("train/val manifest sample_id overlap")
    if ({row["init"] for row in train} | {row["target"] for row in train}) & (
            {row["init"] for row in val} | {row["target"] for row in val}):
        raise ValueError("train/val manifest time overlap")
    cases, case_report = _cases(_csv(texts["case_table"],
        {"seed", "arm", "lead_hours", "split", "n_available_windows", "n_evaluated", "test_read"}), val)
    values = _metrics(_csv(texts["rmse_table"],
        {"seed", "arm", "lead_hours", "variable", "unit", "rmse", "n_initializations"}), cases)
    checks = _paired_checks(_json(texts["paired_comparison"]), protocol, values, cases)
    script_path = _plain_path(Path(__file__))
    script_raw = script_path.read_bytes()
    return {"format": "r7-n1-readonly-arithmetic-audit-v1", "scientific_claim": False,
            "test_read": False, "gpu_hours": 0, "network_requests": 0, "seeds": list(SEEDS),
            "sources": sources,
            "script": {"path": str(script_path), "sha256": hashlib.sha256(script_raw).hexdigest()},
            "archive_protocol": protocol, "cross_checks": checks,
            "manifests": {"train": train_summary, "val": val_summary}, "val_cases_by_lead": case_report,
            "seed_spread_definition": "within-arm max(seed RMSE)-min(seed RMSE), in K; range, not CI",
            "comparisons": [_comparison(values, focus, lead) for focus in FOCUS_ARMS for lead in AUDIT_LEADS],
            "training_objective": _weights(protocol), "limitations": list(LIMITATIONS)}


def _write_exclusive(output: Path, text: str, report: dict, archive: Path, manifests: Path) -> None:
    path = _plain_path(output)
    originals = {record["path"] for record in report["sources"].values()} | {report["script"]["path"]}
    if str(path) in originals or any(path.is_relative_to(_plain_path(root)) for root in (archive, manifests)):
        raise ValueError("output must not overwrite a source or enter a read-only source directory")
    # Parent must already exist. Never create directories or replace existing files.
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--manifests", type=Path, default=DEFAULT_MANIFESTS)
    parser.add_argument("--output", type=Path, help="exclusive new JSON file; existing parent required (default: stdout)")
    args = parser.parse_args(argv)
    try:
        report = recompute(args.archive, args.manifests)
        text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is None:
            sys.stdout.write(text)
        else:
            _write_exclusive(args.output, text, report, args.archive, args.manifests)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        print(f"N1 audit error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
