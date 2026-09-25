"""Seed-identity-keyed co-reasoning comparison (#60), on real regional ERA5.

Issue #6's research gate is that process-aware co-reasoning must beat the generic
recursive baseline **under a comparable budget**, and its fairness contract
requires parameter counts and measured cost alongside accuracy:

1. `generic` recursion — #5's baseline; K=0 removes recursion entirely,
2. `process_no_feedback` — process tokens without forecast feedback,
3. `process_feedback` — process + forecast co-reasoning.

The adaptive arm is evaluated separately (`docs/R7_ADAPTIVE_HALTING.md`), not here.

#60 tightened how multi-seed results are joined. Earlier versions dropped the
seed id during aggregation and zipped per-seed values by list position, so
reordering the input records flipped a stable improvement into "unresolved".
The join is now by the full ``arm/depth/variable/unit/lead/seed`` key:

- per-seed values are kept under explicit seed ids; deltas are paired by seed,
  never by position;
- missing or duplicate seeds, duplicate table entries, a variable reported in
  two units, mismatched case sets (equal counts with different
  initializations), and non-finite or negative RMSE all fail closed;
- ``rmse_seed_mean`` (mean over seeds of the per-seed cross-case pooled RMSE)
  and ``rmse_pooled_cases`` (pooled over every seed's cases) are named and
  reported separately; neither mixes physical units;
- ``sign_consistent`` and the win/loss counts stay **descriptive statistics**.
  The research gate is a separate function (`evaluate_gate`) that reads
  criteria frozen before the experiment; many unresolved variables can never
  be overruled by one improved variable.

An aggregate score is deliberately not produced: averaging RMSE across K, Pa and
m/s would be meaningless. Every seed is retained and a mixed outcome is reported
as mixed. No test split is read.

Historical results are re-audited by `reaggregate_historical`: it re-reads the
original per-evaluation ``rmse.csv``/``provenance.json`` artifacts, rebuilds the
table with this comparator, and writes a *new* audit file next to a diff against
the original report. Originals are never modified; artifacts that cannot be
read are reported as blocked entries — no seed or case ids are invented.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEEDS = (41, 42, 43)
# (arm name, model kind, construction options)
ARMS = (
    ("generic", "generic", {"latent_tokens": 16}),
    ("process_no_feedback", "process",
     {"anchored_processes": 8, "free_processes": 8, "use_forecast_feedback": False}),
    ("process_feedback", "process",
     {"anchored_processes": 8, "free_processes": 8, "use_forecast_feedback": True}),
)
ABLATION_DEPTHS = (0, 1, 3)
CONFIG_BASE = {
    "in_channels": 11, "out_channels": 11, "history_steps": 2,
    "dim": 32, "depth": 2, "heads": 4, "window_size": 4, "patch_size": 2,
    "dropout": 0.0, "default_reasoning_steps": 3,
}
DEFAULT_UPDATES = 200
SPLIT_YEARS = {"train": [2018], "val": [2019], "test": [2020]}
# rmse.csv values come from the same accumulator as the per-case MSE stored in
# provenance.json; sqrt of the per-case mean must reproduce them to float noise.
POOLED_CONSISTENCY_RTOL = 1e-9


def arm_config(kind, options, channels):
    config = dict(CONFIG_BASE)
    config["in_channels"] = channels
    config["out_channels"] = channels
    config.update(options)
    return config


def protocol_payload(source_sha256, channels, updates, depths, seeds):
    """Frozen before any optimizer step, including every fixed choice."""
    from training.r7_experiment import canonical_digest

    body = {
        "format": "r7-coreasoning-fair-budget-v1",
        "frozen_before_any_step": True,
        "source_sha256": source_sha256,
        "channels": channels,
        "arms": [
            {"name": name, "kind": kind, "model_config": arm_config(kind, options, channels)}
            for name, kind, options in ARMS
        ],
        "seeds": list(seeds),
        "optimizer_updates": updates,
        "batch_size": 2, "lr": 2e-4, "clip": 1.0, "accumulation": 1,
        "training_reasoning_steps": 3, "process_weight": 0.1,
        "ablation_depths": list(depths),
        "evaluation_split": "val", "test_evaluated": False,
        "budget_parity": (
            "arms differ by +0.030% parameters and +0.0001-0.0003% forward FLOPs; "
            "see docs/R7_BUDGET_PARITY.md"
        ),
        "all_seeds_retained": True,
        "limitations": [
            "one small tile over ten January days per year; not full seasons",
            "fixed optimizer endpoint, not extension until a metric improves",
            "validation cases were inspected during earlier iterations",
            "three seeds: no significance test, no convergence or SOTA claim",
            "K=0 is the no-recursion ablation on the same architecture",
        ],
        "scientific_claim": False,
    }
    return dict(body, protocol_sha256=canonical_digest(body))


def read_rmse_rows(evaluation_dir):
    """Per-variable RMSE from one evaluation, validated before any aggregation.

    Every row must carry a finite non-negative RMSE, a non-empty variable name
    and unit, and a positive initialization count; the same (variable, unit,
    lead) may appear only once per table.
    """
    path = Path(evaluation_dir) / "rmse.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    seen = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            try:
                lead = float(record["lead_hours"])
                variable = record["variable"]
                unit = record["unit"]
                rmse = float(record["rmse"])
                n_initializations = int(record["n_initializations"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"malformed RMSE row in {path}: {error}") from error
            if not variable or not unit:
                raise ValueError(f"empty variable or unit in {path}")
            if not math.isfinite(rmse) or rmse < 0:
                raise ValueError(f"non-finite or negative RMSE {rmse} in {path}")
            if n_initializations < 1:
                raise ValueError(f"non-positive n_initializations in {path}")
            key = (variable, unit, lead)
            if key in seen:
                raise ValueError(f"duplicate table entry {key} in {path}")
            seen.add(key)
            rows.append({
                "lead_hours": lead,
                "variable": variable,
                "unit": unit,
                "rmse": rmse,
                "n_initializations": n_initializations,
            })
    if not rows:
        raise ValueError(f"empty RMSE table: {path}")
    return rows


def read_case_identity(evaluation_dir):
    """Exact case identity from provenance.json, or None when it was not recorded.

    Returns the sorted ``[init_time, valid_times]`` list plus the evaluation
    metadata needed to tie ``rmse.csv`` rows to per-case MSE, or None when the
    evaluation dir predates provenance capture (identity is then count-only and
    must never be claimed as exact).
    """
    path = Path(evaluation_dir) / "provenance.json"
    if not path.is_file():
        return None
    provenance = json.loads(path.read_text(encoding="utf-8"))
    ordered = sorted(provenance["initializations"], key=lambda case: case["init_time"])
    init_times = [case["init_time"] for case in ordered]
    if len(set(init_times)) != len(init_times):
        raise ValueError(f"duplicate initialization time in {path}")
    leads = [float(lead) for lead in provenance["lead_hours"]]
    channels = list(provenance["channels"])
    units = list(provenance["units"])
    per_case_mse = []
    for case in ordered:
        mse = case["mse"]
        if (len(mse) != len(leads)
                or any(len(row) != len(channels) for row in mse)
                or not all(math.isfinite(value) and value >= 0
                           for row in mse for value in row)):
            raise ValueError(f"per-case MSE does not match the [lead,variable] grid in {path}")
        per_case_mse.append(mse)
    if int(provenance["n_evaluated"]) != len(ordered):
        raise ValueError(f"n_evaluated does not match the initialization list in {path}")
    return {
        "n_evaluated": len(ordered),
        "cases": [[case["init_time"], list(case["valid_times"])] for case in ordered],
        "per_case_mse": per_case_mse,
        "channels": channels,
        "units": units,
        "lead_hours": leads,
        "evaluation_manifest_sha256": provenance.get("evaluation_manifest_sha256"),
        "split": provenance.get("split"),
    }


def case_set_digest(case_identity):
    """Stable digest of the exact case list, for cross-arm set comparison.

    Covers only the init/valid-time identity, never the error data: two
    evaluations describe the same cases when this digest matches.
    """
    canonical = json.dumps(case_identity["cases"], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def summarize(records):
    """Per-arm, per-depth, per-variable table keyed with explicit seed ids.

    Every variable and depth is kept: reporting only a favoured variable would
    hide the mixed outcome these arms actually produce. Duplicate seeds for the
    same key fail closed. Two statistics are reported separately:

    - ``rmse_seed_mean``: mean over seeds of each seed's cross-case pooled
      RMSE (the rmse.csv value), paired by seed id;
    - ``rmse_pooled_cases``: pooled over every evaluated case of every seed,
      derived from the per-case MSE in provenance.json (None when provenance
      was not captured).

    Neither is ever averaged across physical units.
    """
    grouped = {}
    identities = []
    for record in records:
        arm, seed, depth = record["arm"], record["seed"], record["depth"]
        rows = read_rmse_rows(record["evaluation_dir"])
        case_identity = read_case_identity(record["evaluation_dir"])
        for row in rows:
            key = (arm, depth, row["variable"], row["unit"], row["lead_hours"])
            bucket = grouped.setdefault(key, {})
            if seed in bucket:
                raise ValueError(f"duplicate seed {seed} for {key}")
            bucket[seed] = {"rmse": row["rmse"],
                            "n_initializations": row["n_initializations"],
                            "case_identity": case_identity}
        identity = record.get("identity")
        if identity is not None:
            identities.append(identity)
    if identities and len(identities) != len(records):
        raise ValueError("identity recorded for some records but not all")
    if identities and any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("records do not share one dataset/model-code/updates identity")
    table = []
    for (arm, depth, variable, unit, lead), bucket in sorted(grouped.items()):
        seeds = sorted(bucket)
        values = [bucket[seed]["rmse"] for seed in seeds]
        pooled = _pooled_cases(bucket, (arm, depth, variable, unit, lead))
        digests = {}
        for seed in seeds:
            case_identity = bucket[seed]["case_identity"]
            digests[str(seed)] = None if case_identity is None else case_set_digest(case_identity)
        exact = None if all(digest is None for digest in digests.values()) else \
            all(digest is not None and digest == digests[str(seeds[0])]
                for digest in digests.values())
        if exact is False:
            if any(digest is None for digest in digests.values()):
                raise ValueError(
                    f"case identity captured for some seeds but not others for "
                    f"{(arm, depth, variable, unit, lead)}")
            raise ValueError(
                f"case sets differ between seeds for {(arm, depth, variable, unit, lead)}")
        table.append({
            "arm": arm, "depth": depth, "variable": variable, "unit": unit,
            "lead_hours": lead, "seeds": seeds,
            "seed_rmse": {str(seed): bucket[seed]["rmse"] for seed in seeds},
            "rmse_seed_mean": statistics.fmean(values),
            "rmse_seed_sd": statistics.stdev(values) if len(values) > 1 else None,
            "per_seed_n_initializations": {
                str(seed): bucket[seed]["n_initializations"] for seed in seeds},
            "case_set_digests": digests,
            "case_identity": "exact" if exact else ("count-only" if exact is None else "differ"),
            "rmse_pooled_cases": pooled,
        })
    return table


def _pooled_cases(bucket, key):
    """RMSE pooled over every case of every seed, from provenance per-case MSE.

    ``rmse.csv`` and the per-case MSE come from the same accumulator, so each
    seed's table value must equal sqrt(mean over its cases of per-case MSE) up
    to float noise; a mismatch fails closed because the two artifacts then do
    not describe the same evaluation. Returns None when any seed lacks
    provenance capture.
    """
    squared = []
    for seed in sorted(bucket):
        entry = bucket[seed]
        case_identity = entry["case_identity"]
        if case_identity is None:
            return None
        try:
            lead_index = case_identity["lead_hours"].index(key[4])
            variable_index = case_identity["channels"].index(key[2])
        except ValueError:
            raise ValueError(f"row {key} is absent from the provenance metadata") from None
        if case_identity["units"][variable_index] != key[3]:
            raise ValueError(
                f"unit mismatch between rmse.csv and provenance for {key}: "
                f"{key[3]} vs {case_identity['units'][variable_index]}")
        per_case = [case[lead_index][variable_index]
                    for case in case_identity["per_case_mse"]]
        expected = math.sqrt(statistics.fmean(per_case))
        if not math.isclose(expected, entry["rmse"], rel_tol=POOLED_CONSISTENCY_RTOL):
            raise ValueError(
                f"rmse.csv value {entry['rmse']} does not match the pooled per-case "
                f"MSE {expected} for {key} seed {seed}")
        squared.extend(per_case)
    return math.sqrt(statistics.fmean(squared))


def compare(table, *, baseline="generic", depth=None, on_incomplete="raise"):
    """Per-variable paired comparison of each arm against the baseline.

    Rows are joined on the full ``arm/depth/variable/unit/lead`` key and the
    per-seed deltas are paired by explicit seed id, so reordering the input
    records cannot change the output. A negative ``delta`` means the arm has
    the lower RMSE for that variable. The win/loss count is reported instead of
    an aggregate, because a single number would average across physical units.

    Fail-closed contract: a missing baseline row, differing seed sets, differing
    case sets (including equal counts with different initializations), or a
    variable reported under two units is an error. With
    ``on_incomplete="blocked"`` (historical re-aggregation only) incomplete
    keys are listed per block as blocked entries instead of raising; data
    corruption (duplicate seeds, non-finite values, unit conflicts) still
    raises.
    """
    if on_incomplete not in ("raise", "blocked"):
        raise ValueError(f"unknown on_incomplete mode: {on_incomplete}")
    depths = sorted({row["depth"] for row in table})
    if not depths:
        raise ValueError("no rows to compare")
    if depth is None:
        depth = max(depths)
    if depth not in depths:
        raise ValueError(f"depth {depth} was not measured; available={depths}")
    at_depth = [row for row in table if row["depth"] == depth]
    unit_by_key = {}
    for row in at_depth:
        key = (row["variable"], row["lead_hours"])
        unit = unit_by_key.setdefault(key, row["unit"])
        if unit != row["unit"]:
            raise ValueError(
                f"unit mismatch for {key}: {unit} vs {row['unit']}; "
                "a variable must keep one unit across arms")
    by_key = {}
    for row in at_depth:
        key = (row["arm"], row["variable"], row["unit"], row["lead_hours"])
        if key in by_key:
            raise ValueError(f"duplicate comparison row {key}")
        by_key[key] = row
    results = []
    for arm in sorted({key[0] for key in by_key} - {baseline}):
        deltas, wins, losses = [], 0, 0
        paired, blocked = [], []
        case_exact = True
        for key in sorted(k for k in by_key if k[0] == arm):
            _, variable, unit, lead = key
            row = by_key[key]
            reference = by_key.get((baseline, variable, unit, lead))
            if reference is None:
                message = f"missing baseline row for {key} at depth {depth}"
                if on_incomplete == "raise":
                    raise ValueError(message)
                blocked.append({"arm": arm, "variable": variable, "unit": unit,
                                "lead_hours": lead, "reason": message})
                continue
            if row["seeds"] != reference["seeds"]:
                missing = sorted(set(row["seeds"]) ^ set(reference["seeds"]))
                message = (f"seed sets differ for {key}: "
                           f"arm={row['seeds']} baseline={reference['seeds']} "
                           f"unpaired={missing}")
                if on_incomplete == "raise":
                    raise ValueError(message)
                blocked.append({"arm": arm, "variable": variable, "unit": unit,
                                "lead_hours": lead, "reason": message})
                continue
            for seed in row["seeds"]:
                seed_key = str(seed)
                if (row["per_seed_n_initializations"][seed_key]
                        != reference["per_seed_n_initializations"][seed_key]):
                    raise ValueError(
                        f"case-set mismatch for {key} seed {seed}: "
                        f"{row['per_seed_n_initializations'][seed_key]} vs "
                        f"{reference['per_seed_n_initializations'][seed_key]} initializations")
                arm_digest = row["case_set_digests"][seed_key]
                reference_digest = reference["case_set_digests"][seed_key]
                if (arm_digest is None) != (reference_digest is None):
                    raise ValueError(
                        f"case identity recorded for one side only for {key} seed {seed}")
                if arm_digest is not None and arm_digest != reference_digest:
                    raise ValueError(
                        f"initialization sets differ for {key} seed {seed} "
                        "despite equal counts; equal n_initializations never "
                        "implies the same cases")
                if arm_digest is None:
                    case_exact = False
            seed_deltas = [
                {"seed": seed,
                 "delta": row["seed_rmse"][str(seed)] - reference["seed_rmse"][str(seed)]}
                for seed in row["seeds"]]
            diffs = [entry["delta"] for entry in seed_deltas]
            mean_delta = statistics.fmean(diffs)
            deltas.append({"variable": variable, "unit": unit,
                           "lead_hours": lead, "delta": mean_delta})
            if mean_delta < 0:
                wins += 1
            elif mean_delta > 0:
                losses += 1
            # A win/loss split is only meaningful if the per-seed deltas agree on
            # their sign. Seed-paired deltas that flip sign mean the effect is below
            # seed noise, so those variables must not be counted as established.
            paired.append({
                "variable": variable, "unit": unit, "lead_hours": lead,
                "seed_deltas": seed_deltas,
                "sign_consistent": all(d < 0 for d in diffs) or all(d > 0 for d in diffs),
                "direction": "improved" if all(d < 0 for d in diffs) else
                             "worsened" if all(d > 0 for d in diffs) else "unresolved",
            })
        established = [row for row in paired if row["sign_consistent"]]
        results.append({
            "arm": arm, "baseline": baseline, "depth": depth,
            "case_identity": "exact" if case_exact and paired else
                             ("count-only" if paired else "none"),
            "variables_improved": wins, "variables_worsened": losses,
            "variables_compared": len(paired),
            "variables_unresolved": sum(1 for r in paired if r["direction"] == "unresolved"),
            "deltas": deltas, "seed_paired": paired,
            "blocked_keys": blocked,
            "variables_sign_consistent": len(established),
            "established_improved": sum(1 for r in established if r["direction"] == "improved"),
            "established_worsened": sum(1 for r in established if r["direction"] == "worsened"),
            "beats_baseline_everywhere": losses == 0 and wins > 0 and not blocked,
        })
    return results


def evaluate_gate(comparison, criteria):
    """Evaluate the pre-frozen research gate; never a descriptive side product.

    ``criteria`` is the full judgement written **before** the experiment, e.g.::

        {"arm": "process_feedback", "baseline": "generic", "depth": 3,
         "required_variables": [{"variable": "t2m", "unit": "K", "lead_hours": 6.0}],
         "required_direction": "improved", "allow_unresolved_required": 0}

    The gate passes only when every required variable is sign-consistent in the
    required direction. Unresolved required variables consume the frozen
    allowance and the gate fails once it is exceeded; a worsened required
    variable always fails. Absent or blocked required variables fail. This is a
    descriptive protocol check, not a significance test.
    """
    for field in ("arm", "required_variables", "required_direction",
                  "allow_unresolved_required"):
        if field not in criteria:
            raise ValueError(f"gate criteria missing {field}")
    arm = criteria["arm"]
    direction = criteria["required_direction"]
    if direction not in ("improved", "worsened"):
        raise ValueError(f"unknown required_direction: {direction}")
    blocks = [block for block in comparison if block["arm"] == arm]
    if not blocks:
        return {"gate_met": False, "evaluated": True, "failures": [f"arm {arm} absent"],
                "warnings": [], "criteria": criteria, "scientific_claim": False}
    block = blocks[0]
    paired = {(entry["variable"], entry["unit"], entry["lead_hours"]): entry
              for entry in block["seed_paired"]}
    blocked_keys = {(entry["variable"], entry["unit"], entry["lead_hours"]): entry
                    for entry in block["blocked_keys"]}
    failures, warnings = [], []
    unresolved_used = 0
    for required in criteria["required_variables"]:
        key = (required["variable"], required["unit"], float(required["lead_hours"]))
        if key in blocked_keys:
            failures.append(f"{key}: blocked ({blocked_keys[key]['reason']})")
            continue
        entry = paired.get(key)
        if entry is None:
            failures.append(f"{key}: absent from comparison")
            continue
        if entry["direction"] == direction:
            continue
        if entry["direction"] == "unresolved":
            unresolved_used += 1
            if unresolved_used > criteria["allow_unresolved_required"]:
                failures.append(f"{key}: unresolved beyond the frozen allowance "
                                f"({criteria['allow_unresolved_required']})")
            else:
                warnings.append(f"{key}: unresolved within the frozen allowance")
        else:
            failures.append(f"{key}: {entry['direction']} but {direction} required")
    return {"gate_met": not failures, "evaluated": True, "failures": failures,
            "warnings": warnings, "criteria": criteria, "arm_block": block,
            "scientific_claim": False,
            "note": "protocol check against pre-frozen criteria; no significance test"}


def reaggregate_historical(result_path, output_path, *, root=None):
    """Re-aggregate a historical result from its original evaluation artifacts.

    Reads the recorded ``records`` (arm/seed/depth/evaluation_dir), re-reads
    each ``rmse.csv`` and ``provenance.json``, rebuilds the summary and
    comparison with the seed-keyed comparator, and writes a **new** audit file
    containing the old-vs-new diff. The original file and artifacts are never
    modified; the output path must not exist. Artifacts that cannot be read are
    listed as blocked records — no seed or case ids are invented for them.
    """
    result_path = Path(result_path)
    original_bytes = result_path.read_bytes()
    original = json.loads(original_bytes)
    root = Path(root) if root is not None else Path.cwd()
    records, blocked_records = [], []
    for record in original.get("records", []):
        evaluation_dir = Path(record["evaluation_dir"])
        if not evaluation_dir.is_absolute():
            evaluation_dir = root / evaluation_dir
        try:
            read_rmse_rows(evaluation_dir)
            read_case_identity(evaluation_dir)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            blocked_records.append({"arm": record["arm"], "seed": record["seed"],
                                    "depth": record["depth"],
                                    "path": str(evaluation_dir), "reason": str(error)})
            continue
        records.append(dict(record, evaluation_dir=str(evaluation_dir)))
    table = summarize(records) if records else []
    comparison = compare(table, on_incomplete="blocked") if table else []
    summary_diff = []
    old_summary = {(row["arm"], row["depth"], row["variable"], row["unit"],
                    row["lead_hours"]): row for row in original.get("summary", [])}
    new_summary = {(row["arm"], row["depth"], row["variable"], row["unit"],
                    row["lead_hours"]): row for row in table}
    for key in sorted(set(old_summary) | set(new_summary)):
        old, new = old_summary.get(key), new_summary.get(key)
        entry = {"arm": key[0], "depth": key[1], "variable": key[2],
                 "unit": key[3], "lead_hours": key[4]}
        if old is None:
            entry.update(status="only-in-new-aggregation")
        elif new is None:
            entry.update(status="blocked-or-missing-in-new-aggregation")
        else:
            entry.update(
                status="reaggregated",
                rmse_seed_mean_old=old["rmse_mean"],
                rmse_seed_mean_new=new["rmse_seed_mean"],
                mean_difference=new["rmse_seed_mean"] - old["rmse_mean"],
                seeds_old=old["seeds"], seeds_new=new["seeds"])
        summary_diff.append(entry)
    comparison_diff = []
    old_blocks = {(block["arm"], block["depth"]): block
                  for block in original.get("comparison", [])}
    for block in comparison:
        old_block = old_blocks.get((block["arm"], block["depth"]))
        old_paired = {(entry["variable"], entry["lead_hours"]): entry
                      for entry in (old_block or {}).get("seed_paired", [])}
        for entry in block["seed_paired"]:
            old_entry = old_paired.get((entry["variable"], entry["lead_hours"]))
            comparison_diff.append({
                "arm": block["arm"], "depth": block["depth"],
                "variable": entry["variable"], "lead_hours": entry["lead_hours"],
                "direction_old": None if old_entry is None else old_entry["direction"],
                "direction_new": entry["direction"],
                "seed_deltas_old_orderless": None if old_entry is None
                else old_entry["seed_deltas"],
                "seed_deltas_new_by_seed": {str(d["seed"]): d["delta"]
                                            for d in entry["seed_deltas"]},
                "changed": None if old_entry is None
                else old_entry["direction"] != entry["direction"]})
    audit = {
        "format": "r7-coreasoning-reaggregation-audit-v1",
        "scientific_claim": False,
        "source_result": {"path": str(result_path),
                          "sha256": hashlib.sha256(original_bytes).hexdigest()},
        "original_format": original.get("format"),
        "records_total": len(original.get("records", [])),
        "records_reaggregated": len(records),
        "blocked_records": blocked_records,
        "summary_diff": summary_diff,
        "comparison_diff": comparison_diff,
        "comparison": comparison,
        "note": ("original artifacts untouched; unreadable artifacts are blocked, "
                 "never fabricated; seed deltas are now paired by explicit seed id"),
    }
    output_path = Path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return audit


def run_comparison(source, receipt, output_dir, *, updates=DEFAULT_UPDATES,
                   max_samples=8, seeds=SEEDS, depths=ABLATION_DEPTHS):
    """Publish a fresh cache, train every arm at every seed, then evaluate."""
    import torch

    from data.preprocess.r7_preflight import prepare_local
    from training.r7_evaluate import evaluate_local
    from training.r7_experiment import (
        dataset_identity, load_checkpoint, make_model, model_code_digest)
    from training.r7_local_runner import run_local_updates

    source, receipt = Path(source), Path(receipt)
    out = Path(output_dir)
    # Validate the receipt before creating anything, so a rejected run leaves no
    # half-built output directory behind.
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    if payload.get("status") != "downloaded-real-source":
        raise ValueError("a real acquisition receipt is required; no synthetic fallback")
    if payload.get("synthetic_fallback") is not False:
        raise ValueError("receipt must declare synthetic_fallback=False")
    source_sha256 = payload["local_artifact"]["sha256"]
    if not source.is_file():
        raise FileNotFoundError(source)
    # Re-derive the bytes rather than trusting the receipt's own field: a receipt
    # is caller-editable, so this is an integrity check on the input, not proof
    # of provenance (the acquisition receipt carries that separately).
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_sha256:
        raise ValueError("source file does not match the receipt's recorded SHA256")
    out.mkdir(parents=True, exist_ok=False)

    prep_config = {
        "split_years": dict(SPLIT_YEARS),
        "history_steps": 2, "history_interval_hours": 6, "lead_time_hours": 6,
        "sample_stride_hours": 6, "time_chunk": 16,
        "compute_process_targets": True,
    }
    dataset_dir = out / "dataset"
    report = prepare_local(source, prep_config, write=True,
                           store_path=dataset_dir / "cache.zarr",
                           manifest_dir=dataset_dir / "manifests", max_raw_gib=0.5)
    train_manifest = dataset_dir / "manifests" / "train.jsonl"
    val_manifest = dataset_dir / "manifests" / "val.jsonl"
    channels = int(report["shape"][1])

    protocol = protocol_payload(source_sha256, channels, updates, depths, seeds)
    with (out / "protocol.json").open("x", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=False, allow_nan=False)

    identity, dataset = dataset_identity(train_manifest)
    digest = model_code_digest()
    torch.set_num_threads(2)
    records, resources = [], []
    for seed in seeds:
        for name, kind, options in ARMS:
            model_config = arm_config(kind, options, channels)
            checkpoint, training = run_local_updates(
                dataset, kind=kind, model_config=model_config, data_identity=identity,
                output_dir=out / "training" / f"{name}_{seed}", total_updates=updates,
                batch_size=2, steps=3, seed=seed, lr=2e-4,
                process_weight=0.1, device_name="cpu")
            saved = load_checkpoint(checkpoint)
            if saved["updates"] != updates:
                raise ValueError("fixed optimizer endpoint not reached")
            resources.append({
                "arm": name, "seed": seed,
                "params": sum(p.numel() for p in make_model(kind, model_config).parameters()),
                "training": training,
            })
            for depth in depths:
                evaluation_dir = out / "evaluation" / f"{name}_{seed}_K{depth}"
                evaluate_local(val_manifest, output_dir=evaluation_dir,
                               checkpoint=checkpoint, lead_hours=(6,),
                               max_samples=max_samples, reasoning_steps=depth,
                               device_name="cpu")
                records.append({"arm": name, "seed": seed, "depth": depth,
                                "evaluation_dir": str(evaluation_dir),
                                "identity": {"dataset_identity": identity,
                                             "model_code_sha256": digest,
                                             "optimizer_updates": updates}})
        print(json.dumps({"seed": seed, "arms_complete": len(ARMS)}), flush=True)

    if model_code_digest() != digest:
        raise RuntimeError("model code changed during the comparison")
    table = summarize(records)
    result = {
        "format": "r7-coreasoning-fair-budget-result-v2",
        "complete": len(records) == len(seeds) * len(ARMS) * len(depths),
        "scientific_claim": False, "gpu_used": False, "test_evaluated": False,
        "source_cloud_requests": 0,
        "protocol": protocol, "protocol_sha256": protocol["protocol_sha256"],
        "preflight": report, "model_code_sha256": digest,
        "resources": resources, "records": records,
        "summary": table, "comparison": compare(table),
        "research_gate": "not_evaluated; run evaluate_gate with pre-frozen criteria",
        "limitations": protocol["limitations"],
    }
    with (out / "coreasoning_result.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Seed-identity-keyed co-reasoning ablation comparison (#6/#60).")
    parser.add_argument("--source", help="real acquired source archive (run mode)")
    parser.add_argument("--receipt", help="acquisition receipt JSON (run mode)")
    parser.add_argument("--out", required=True,
                        help="output result path (run) or audit path (reaggregate)")
    parser.add_argument("--reaggregate", dest="reaggregate", metavar="RESULT_JSON",
                        help="re-aggregate a historical result from its original artifacts")
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--max-samples", type=int, default=8)
    args = parser.parse_args()
    if args.reaggregate:
        if args.source or args.receipt:
            parser.error("--reaggregate takes only --out")
        audit = reaggregate_historical(args.reaggregate, args.out)
        print(json.dumps({
            "records_reaggregated": audit["records_reaggregated"],
            "blocked_records": len(audit["blocked_records"]),
            "changed_directions": sum(1 for entry in audit["comparison_diff"]
                                      if entry["changed"]),
        }, ensure_ascii=False))
        return
    if not args.source or not args.receipt:
        parser.error("run mode requires --source and --receipt")
    result = run_comparison(args.source, args.receipt, args.out,
                            updates=args.updates, max_samples=args.max_samples)
    print(json.dumps({
        "complete": result["complete"],
        "protocol_sha256": result["protocol_sha256"],
        "comparison": [{k: v for k, v in block.items() if k != "deltas"}
                       for block in result["comparison"]],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
