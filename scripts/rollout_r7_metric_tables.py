"""Paper-ready 6-72 h rollout metric tables for #8, from frozen checkpoints.

Issue #8's acceptance is that "evaluation scripts produce paper-ready metric
tables without changing training code". The rollout machinery, RMSE and ACC
accumulators already existed; what did not exist was a script that drives several
already-trained checkpoints through a 6/12/24/48/72 h rollout on the held-out
year and emits a single comparable table.

This module **only evaluates**. It imports the training-free evaluation entry
point, never calls an optimizer, and refuses to proceed if any checkpoint's
recorded model digest disagrees with the live `model/` tree — so a table cannot
silently mix two model generations.

Comparability is enforced rather than assumed: every checkpoint must have been
trained on the same dataset identity, and every row records the checkpoint hash,
the model code digest and the manifest hash. Columns are per variable and lead;
no cross-variable average is produced, because K, Pa and m/s are not summable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_LEADS = (6, 12, 24, 48, 72)
# Columns written for each (variable, lead) cell. `acc_status` matters because an
# undefined ACC is a real outcome, not a zero.
RMSE_COLUMNS = ("lead_hours", "variable", "rmse", "unit", "n_initializations")
ACC_COLUMNS = ("lead_hours", "variable", "pooled_acc", "status", "n_initializations")
# The climatology baseline rides in the same table as the forecast RMSE (#64 D-3),
# so `rmse` and `rmse_climatology` are always read off the same cases.
SKILL_COLUMNS = ("lead_hours", "variable", "rmse_forecast", "rmse_climatology",
                 "mse_skill", "unit", "n_initializations")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metric_csv(path, columns):
    """Read one metric CSV, requiring the exact expected header."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise ValueError(f"unexpected header in {path.name}: {reader.fieldnames}")
        rows = []
        for record in reader:
            rows.append({key: record[key] for key in columns})
    if not rows:
        raise ValueError(f"empty metric table: {path}")
    return rows


def check_checkpoint(checkpoint, *, manifest):
    """Load a checkpoint and refuse to mix model generations or datasets."""
    from training.r7_experiment import load_checkpoint, model_code_digest
    from training.r7_calibration_runner import file_sha256 as _sha

    saved = load_checkpoint(checkpoint)
    live = model_code_digest()
    recorded = saved.get("model_code_sha256") or (saved.get("contract") or {}).get("model_code_sha256")
    if recorded is None:
        raise ValueError(f"checkpoint {checkpoint} records no model digest")
    if recorded != live:
        raise RuntimeError(
            "checkpoint was produced by a different model/ code version; "
            "replay it with its archived code.zip instead of bypassing the identity check"
        )
    identity = (saved.get("contract") or {}).get("data_identity")
    if identity is None:
        raise ValueError(f"checkpoint {checkpoint} records no data identity")
    return {
        "checkpoint": str(Path(checkpoint).name),
        "checkpoint_sha256": _sha(checkpoint),
        "model_code_sha256": recorded,
        "data_identity": str(identity),
        "updates": saved.get("updates"),
        "manifest": str(manifest),
    }


def _collect_run(report, run_dir):
    """The per-run metric rows every entry (neural or baseline) must carry."""
    return {
        "n_evaluated": report["n_evaluated"],
        "channels": list(report["channels"]),
        "units": list(report["units"]),
        "rmse_rows": read_metric_csv(run_dir / "rmse.csv", RMSE_COLUMNS),
        "acc_rows": read_metric_csv(run_dir / "acc.csv", ACC_COLUMNS),
        "skill_rows": read_metric_csv(run_dir / "climatology_skill.csv", SKILL_COLUMNS),
        "climatology": report["climatology"],
    }


def _evaluate_persistence(manifest, output_dir, *, leads, max_samples):
    """Checkpoint-free persistence baseline on the same manifest and cases."""
    from training.r7_evaluate import evaluate_local

    run_dir = Path(output_dir) / "persistence"
    report = evaluate_local(manifest, output_dir=run_dir, lead_hours=tuple(leads),
                            max_samples=max_samples, device_name="cpu")
    if report["split"] != "test":
        raise ValueError("persistence was evaluated on a non-test split")
    entry = {
        "label": "persistence", "checkpoint": None, "checkpoint_sha256": None,
        "model_code_sha256": None, "data_identity": None, "updates": None,
        "manifest": str(manifest),
    }
    entry.update(_collect_run(report, run_dir))
    return entry


def evaluate_checkpoints(manifest, checkpoints, output_dir, *, leads=DEFAULT_LEADS,
                         reasoning_steps=None, max_samples=8, labels=None,
                         with_persistence=True):
    """Run each checkpoint over the same held-out manifest and collect tables."""
    from training.r7_evaluate import evaluate_local

    manifest = Path(manifest)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_sha = file_sha256(manifest)
    entries = []
    identities = set()
    for index, checkpoint in enumerate(checkpoints):
        label = (labels[index] if labels else None) or Path(checkpoint).parent.name
        info = check_checkpoint(checkpoint, manifest=manifest)
        identities.add(info["data_identity"])
        run_dir = output_dir / label
        report = evaluate_local(manifest, output_dir=run_dir, checkpoint=checkpoint,
                                lead_hours=tuple(leads), max_samples=max_samples,
                                reasoning_steps=reasoning_steps, device_name="cpu")
        if report["split"] != "test":
            raise ValueError(f"{label} was evaluated on split {report['split']}, not the held-out test split")
        if tuple(report["lead_hours"]) != tuple(leads):
            raise ValueError(f"{label} evaluated leads {report['lead_hours']} != {tuple(leads)}")
        info.update({"label": label})
        info.update(_collect_run(report, run_dir))
        entries.append(info)
    if len(identities) != 1:
        raise RuntimeError(
            f"checkpoints come from {len(identities)} different datasets; the table would not be comparable")
    if len({tuple(entry["channels"]) for entry in entries}) != 1:
        raise RuntimeError("checkpoints disagree on the channel order")
    if len({tuple(entry["units"]) for entry in entries}) != 1:
        raise RuntimeError("checkpoints disagree on channel units")
    # Every model must also carry the climatology baseline on its own cases, or the
    # skill column would be read off a different case set than the RMSE column.
    if len({entry["climatology"]["selection"] for entry in entries}) != 1:
        raise RuntimeError("entries disagree on how the training climatology was selected")
    if with_persistence:
        # Same manifest, same case cap: persistence must share the exact case set
        # or it is not a same-data baseline.
        baseline = _evaluate_persistence(manifest, output_dir, leads=leads,
                                        max_samples=max_samples)
        if tuple(baseline["channels"]) != tuple(entries[0]["channels"]):
            raise RuntimeError("persistence disagrees on the channel order")
        first = entries[0]["rmse_rows"][0]
        base_first = baseline["rmse_rows"][0]
        if first["n_initializations"] != base_first["n_initializations"]:
            raise RuntimeError("persistence was not scored on the same number of cases")
        entries = entries + [baseline]
    for entry in entries:
        case_counts = {row["n_initializations"] for row in entry["skill_rows"]}
        if case_counts != {str(entry["n_evaluated"])}:
            raise RuntimeError(
                f"{entry['label']} climatology rows cover {case_counts} cases but "
                f"n_evaluated is {entry['n_evaluated']}")
    return {
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "data_identity": next(iter(identities)),
        "leads": list(leads),
        "reasoning_steps": reasoning_steps,
        "max_samples": max_samples,
        "entries": entries,
    }


def _index(rows, value_key):
    return {(int(float(row["lead_hours"])), row["variable"]): row for row in rows}


def _fmt(value):
    return "" if value is None or value == "" else f"{float(value):.6g}"


def write_tables(collected, out_dir):
    """Emit the paper-ready wide tables and a coverage summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = collected["entries"]
    leads = list(collected["leads"])
    variables = list(entries[0]["channels"])
    units = list(entries[0]["units"])

    rmse_path = out_dir / "rollout_rmse_table.csv"
    with rmse_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "variable", "unit"] + [f"{h}h" for h in leads])
        for entry in entries:
            index = _index(entry["rmse_rows"], "rmse")
            for variable, unit in zip(variables, units):
                writer.writerow([entry["label"], variable, unit] + [
                    _fmt(index.get((lead, variable), {}).get("rmse")) for lead in leads])

    acc_path = out_dir / "rollout_acc_table.csv"
    undefined = 0
    with acc_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "variable"] + [f"{h}h" for h in leads])
        for entry in entries:
            index = _index(entry["acc_rows"], "pooled_acc")
            for variable in variables:
                cells = []
                for lead in leads:
                    row = index.get((lead, variable))
                    if row is None or row["status"] != "defined":
                        undefined += 1
                        cells.append("undefined")
                    else:
                        cells.append(_fmt(row["pooled_acc"]))
                writer.writerow([entry["label"], variable] + cells)

    skill_path = out_dir / "rollout_climatology_skill_table.csv"
    undefined_skill = 0
    with skill_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "metric", "variable", "unit"] + [f"{h}h" for h in leads])
        for entry in entries:
            index = _index(entry["skill_rows"], "mse_skill")
            for metric in ("rmse_forecast", "rmse_climatology", "mse_skill"):
                for variable, unit in zip(variables, units):
                    cells = []
                    for lead in leads:
                        row = index.get((lead, variable))
                        value = "" if row is None else row[metric]
                        if value == "":
                            undefined_skill += 1
                            cells.append("undefined")
                        else:
                            cells.append(_fmt(value))
                    writer.writerow([entry["label"], metric, variable, unit] + cells)

    provenance = {
        "format": "r7-rollout-metric-tables-v1",
        "scientific_claim": False,
        "training_code_changed": False,
        "manifest": collected["manifest"],
        "manifest_sha256": collected["manifest_sha256"],
        "data_identity": collected["data_identity"],
        "leads": leads,
        "reasoning_steps": collected["reasoning_steps"],
        "max_samples": collected["max_samples"],
        "models": [{k: v for k, v in entry.items()
                    if k not in ("rmse_rows", "acc_rows", "skill_rows")} for entry in entries],
        "tables": {"rmse": rmse_path.name, "acc": acc_path.name,
                   "climatology_skill": skill_path.name},
        "undefined_acc_cells": undefined,
        "undefined_skill_cells": undefined_skill,
        "aggregation": (
            "per variable and lead only; no cross-variable average, because K, Pa and "
            "m/s are not summable"
        ),
        "climatology_baseline": {
            "columns": ["rmse_forecast", "rmse_climatology", "mse_skill"],
            "definition": "mse_skill = 1 - MSE_forecast / MSE_climatology",
            "scope": ("scored on the same cases as the row's rmse, one frozen "
                      "train-only climatology shared by all models in this table"),
            "caveat": ("rmse_climatology is not a WeatherBench2 climatology and is not a "
                       "held-out-year climatology; it is the declared train-only "
                       "month-hour grid mean"),
        },
        "limitations": [
            "held-out year is small; n_evaluated is recorded per row",
            "ACC is undefined where the climatology anomaly energy is zero",
            "MSE skill is undefined where the climatology error energy is zero",
            "identical initialization sets are enforced by the shared manifest",
            "single seed per checkpoint unless several are supplied",
            "not a converged benchmark; bounded CPU checkpoints",
        ],
    }
    if undefined:
        provenance["limitations"].append(
            f"{undefined} ACC cells are undefined and written as 'undefined', never as zero")
    if undefined_skill:
        provenance["limitations"].append(
            f"{undefined_skill} MSE-skill cells are undefined and written as 'undefined', "
            "never as zero")
    with (out_dir / "table_provenance.json").open("x", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return provenance


def main():
    parser = argparse.ArgumentParser(
        description="Produce paper-ready 6-72h rollout metric tables from checkpoints.")
    parser.add_argument("--manifest", required=True, help="held-out (test) manifest")
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--label", nargs="+",
                        help="one label per checkpoint; defaults to the training directory name")
    parser.add_argument("--out", required=True, help="new directory for the tables")
    parser.add_argument("--leads", type=int, nargs="+", default=list(DEFAULT_LEADS))
    parser.add_argument("--reasoning-steps", type=int)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--no-persistence", action="store_true",
                        help="skip the checkpoint-free persistence baseline")
    args = parser.parse_args()
    if args.label and len(args.label) != len(args.checkpoint):
        parser.error("--label must provide one label per checkpoint")
    import torch

    torch.set_num_threads(2)
    collected = evaluate_checkpoints(
        args.manifest, args.checkpoint, args.out, leads=args.leads,
        reasoning_steps=args.reasoning_steps, max_samples=args.max_samples,
        labels=args.label, with_persistence=not args.no_persistence)
    provenance = write_tables(collected, args.out)
    print(json.dumps({
        "models": [entry["label"] for entry in collected["entries"]],
        "leads": collected["leads"],
        "n_evaluated": {entry["label"]: entry["n_evaluated"] for entry in collected["entries"]},
        "undefined_acc_cells": provenance["undefined_acc_cells"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
