"""Read-only, fail-closed evidence for the isolated v3 rollout continuation."""
from __future__ import annotations

import csv
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import re

from data.preprocess.r7_era5 import DEFAULT_R7_ERA5_CHANNELS
from scripts import study_r7_s3_v3_rollout_ft as recipe
from training import r7_s3_v3_screen as screen


def _require(condition, label):
    if not condition:
        raise ValueError(label)


def _equal(actual, expected, label):
    _require(actual == expected and (type(actual) is type(expected) if type(expected) in (bool, int) else True),
             f"{label} differs from frozen evidence")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _sha(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
             "SHA256 must be 64 lowercase hexadecimal characters")
    return value


def _local(value, expected=None):
    path = Path(value)
    _require(path.is_absolute() and ".." not in path.parts and "://" not in str(value),
             "absolute local path without traversal required")
    _require(not any(p.is_symlink() for p in (path, *path.parents)),
             f"symlink in evidence path: {path}")
    if expected is not None:
        _equal(str(value), str(expected), "exact evidence path")
    return path


def _hashed(path, expected):
    path = _local(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    _equal(digest.hexdigest(), _sha(expected), f"SHA256 for {path}")
    return path


def _unique(pairs):
    value = {}
    for key, item in pairs:
        _require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _json(path):
    value = json.loads(_local(path).read_text(encoding="utf-8"), object_pairs_hook=_unique,
                       parse_constant=lambda value: _require(False, f"nonfinite JSON: {value}"))
    _require(isinstance(value, dict), "JSON evidence object required")
    return value


def _number(value, *, positive=False):
    _require(not isinstance(value, bool) and isinstance(value, (float, int)),
             "numeric evidence required")
    _require(math.isfinite(value) and (value > 0 if positive else value >= 0),
             "finite positive denominator required" if positive else "finite nonnegative evidence required")
    return value


def _flags(value):
    _require(value.get("scientific_claim") is False and value.get("test_read", False) is False,
             "scientific_claim:false and no test_read required")


def _options():
    return {"mode": recipe.FT_MODE, "updates": recipe.FT_UPDATES, "lambda12": recipe.FT_LAMBDA12,
            "steps": recipe.STEPS, "lr": recipe.FT_LR, "warmup": recipe.FT_WARMUP,
            "weight_decay": recipe.WEIGHT_DECAY, "batch_size": recipe.BATCH_SIZE,
            "clip": recipe.CLIP, "checkpoint_every": recipe.CHECKPOINT_EVERY, "bf16": recipe.BF16}


def _protocol(output, protocol):
    _flags(protocol)
    _equal(_json(output / "protocol.json"), protocol, "root protocol")
    _equal(protocol["protocol_sha256"], _digest({k: v for k, v in protocol.items()
                                              if k != "protocol_sha256"}), "protocol digest")
    for key in ("model_code_sha256", "training_code_sha256"):
        _sha(protocol[key])
    for key, value in _options().items():
        _equal(protocol["arms"]["candidate"][key], value, f"candidate {key}")
    _equal(protocol["arms"]["control"]["updates"], 400, "control endpoint")
    _equal(protocol["arms"]["parent"]["updates"], recipe.PARENT_ENDPOINT_UPDATES, "parent endpoint")
    _equal(protocol["train_data_identity"], screen.V3_DATA_IDENTITY, "train data identity")
    _equal(protocol["source_sha256"], screen.V3_SOURCE_SHA256, "source identity")
    _require(Path(protocol["train_manifest"]).name == "train.jsonl"
             and Path(protocol["val_manifest"]).name == "val.jsonl", "train/val manifests only; test sealed")
    primary, gate = protocol["decision"]["primary"], protocol["decision"]["gate_pre_screen"]
    for key, expected in {"variable": recipe.PRIMARY_VARIABLE, "region": "full",
                          "leads_hours": list(recipe.PRIMARY_LEADS)}.items():
        _equal(primary[key], expected, f"primary {key}")
    for key, expected in {"variables": list(recipe.GATE_VARIABLES),
                          "leads_hours": list(recipe.EVALUATION_LEADS),
                          "relative_mse_change_max": recipe.GATE_TOLERANCE}.items():
        _equal(gate[key], expected, f"gate {key}")
    manifest = _local(protocol["val_manifest"])
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def _references(output, protocol):
    seeds, leads = {str(s) for s in recipe.SEEDS}, {str(l) for l in recipe.EVALUATION_LEADS}
    references = {}
    for name, prefix, arm, updates in (("control", "d3", "process", 400),
                                        ("parent", "parent", "candidate", recipe.PARENT_ENDPOINT_UPDATES)):
        pins = protocol["arms"][name]["pins"]
        path = _local(pins[f"{prefix}_result_path"])
        _equal(path.name, "result.json", "reference result filename")
        root = path.parent
        _require(not output.is_relative_to(root) and not root.is_relative_to(output),
                 "candidate and reference roots must be disjoint")
        result = _json(_hashed(path, pins[f"{prefix}_result_sha256"]))
        _flags(result)
        _equal(set(result["seeds"]), seeds, "reference seeds")
        for key in ("train_data_identity", "val_data_identity"):
            _equal(result[key], protocol[key], f"reference {key}")
        _equal(pins[f"{name}_updates"], updates, f"{name} updates")
        _equal(set(pins["seeds" if name == "control" else "evaluations"]), seeds, "pinned seeds")
        if name == "parent":
            _equal(set(pins["checkpoints"]), seeds, "parent checkpoint seeds")
            _equal(pins["parent_gate_failures_vs_control"], recipe.PARENT_GATE_FAILURES_VS_CONTROL,
                   "parent gate count")
            _equal(len(result["gate_pre_screen"]["failures"]), recipe.PARENT_GATE_FAILURES_VS_CONTROL,
                   "registered parent gate count")
        for seed in recipe.SEEDS:
            receipt = result["seeds"][str(seed)]
            _equal(receipt["seed"], seed, "reference seed")
            _equal(set(receipt["evaluations"]), leads, "reference leads")
            _local(receipt["checkpoint"], root / f"seed{seed}/training/{arm}/update_{updates:07d}.pt")
            if name == "parent":
                _hashed(receipt["checkpoint"], receipt["checkpoint_sha256"])
                _equal(pins["evaluations"][str(seed)], receipt["evaluations"], "parent evaluation pins")
                _equal(pins["checkpoints"][str(seed)],
                       {"path": receipt["checkpoint"], "sha256": receipt["checkpoint_sha256"]},
                       "parent checkpoint pins")
            else:
                entry = pins["seeds"][str(seed)]
                _equal(set(entry["leads"]), leads, "control pinned leads")
                _equal(entry["initial_state_sha256"], receipt["initial_state_sha256"], "control initial state")
                for lead in recipe.EVALUATION_LEADS:
                    record = receipt["evaluations"][str(lead)]
                    _equal(entry["leads"][str(lead)], {"rmse_csv_sha256": record["rmse_csv_sha256"],
                                                       "evaluation_dir": record["dir"]}, "control CSV pins")
        references[name] = (root, result)
    return references


def _training(output, protocol, receipt, seed):
    _equal(receipt["seed"], seed, "candidate seed")
    _equal(set(receipt["evaluations"]), {str(l) for l in recipe.EVALUATION_LEADS}, "candidate leads")
    folder = _local(receipt["training_dir"], output / f"seed{seed}/training/candidate")
    checkpoint = _local(receipt["checkpoint"], folder / f"update_{recipe.FT_UPDATES:07d}.pt")
    _hashed(checkpoint, receipt["checkpoint_sha256"])
    report = _json(_hashed(folder / "training_report.json", receipt["training_report_sha256"]))
    _flags(report)
    contract = report["contract"]
    _flags(contract)
    _equal(report["signature"], _digest(contract), "training contract signature")
    for key in ("protocol_sha256", "source_sha256", "model_code_sha256"):
        _equal(report[key], protocol[key], f"report {key}")
        _equal(contract[key], protocol[key], f"contract {key}")
    _equal(contract["training_code_sha256"], protocol["training_code_sha256"], "training code digest")
    _equal(report["data_identity"], protocol["train_data_identity"], "report data identity")
    _equal(contract["data_identity"], protocol["train_data_identity"], "contract data identity")
    model = protocol["arms"]["candidate"]["model"]
    _equal(contract["kind"], model["kind"], "model kind")
    _equal(contract["model"], model["spec"], "model spec")
    _equal(_digest(contract["model"]), model["spec_canonical_digest"], "model spec digest")
    _sha(contract["model_semantics_sha256"])
    _equal(contract["seed"], seed, "contract seed")
    _equal(contract["output_dir"], str(folder), "contract output root")
    for key, value in _options().items():
        if key != "lambda12":
            _equal(contract[{"updates": "total_updates", "warmup": "warmup_updates"}.get(key, key)],
                   value, f"training option {key}")
    for key in ("selected_update", "total_updates", "updates_this_run"):
        _equal(report[key], recipe.FT_UPDATES, f"report {key}")
    _equal(report["selected_checkpoint"], str(checkpoint), "selected checkpoint")
    _require(report["resumed_from_updates"] == 0 and report["selection_split"] is None
             and report["parent_optimizer_imported"] is False, "fresh optimizer and frozen endpoint required")
    pins = protocol["arms"]["parent"]["pins"]["checkpoints"][str(seed)]
    _equal(receipt["parent_checkpoint_sha256"], _sha(pins["sha256"]), "parent checkpoint SHA256")
    _require(contract.get("process_supervision") is None and contract.get("intervention") is None,
             "continuation must not inherit process supervision/intervention")
    initialization = contract["initialization"]
    _equal(report["initialization"], initialization, "report initialization")
    for key, value in {"parent_checkpoint": pins["path"], "parent_checkpoint_sha256": pins["sha256"],
                       "parent_endpoint_updates": recipe.PARENT_ENDPOINT_UPDATES}.items():
        _equal(initialization[key], value, f"initialization {key}")
    _equal(_sha(receipt["imported_initial_weights_sha256"]), _sha(contract["initial_weights_sha256"]),
           "imported initial weights")
    autoregression = contract["autoregression"]
    for key, value in {"mode": recipe.FT_MODE, "lambda12": recipe.FT_LAMBDA12, "physical_steps": 2,
                       "step_hours": 6, "detach_physical_steps": False, "detach_reasoning_steps": False,
                       "internal_deep_supervision": True}.items():
        _equal(autoregression[key], value, f"autoregression {key}")
    _equal(report["windows"], autoregression["windows"], "training windows")
    for key, value in screen.V3_TRAIN_WINDOWS.items():
        _equal(report["windows"][key], value, f"train windows {key}")
    losses = report["losses"]
    _equal([row["update"] for row in losses], list(range(1, recipe.FT_UPDATES + 1)), "loss updates")
    for row in losses:
        for key in ("loss", "l6", "l12"):
            _number(row[key])


def _table(path, channels, units, lead, count, skill=False):
    columns = ["lead_hours", "variable", *( ["rmse_forecast", "rmse_climatology", "mse_skill"]
                                           if skill else ["rmse"]), "unit", "n_initializations"]
    with _local(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _equal(reader.fieldnames, columns, "CSV schema")
        rows = list(reader)
    _equal([row["variable"] for row in rows], channels, "CSV variables/order")
    _equal([row["unit"] for row in rows], units, "CSV units")
    for row in rows:
        _require(None not in row and all(value is not None for value in row.values()), "malformed CSV row")
        _equal(row["lead_hours"], str(lead), "CSV lead")
        _equal(row["n_initializations"], str(count), "CSV cohort count")
        for key in ("rmse_forecast", "rmse_climatology") if skill else ("rmse",):
            _number(float(row[key]), positive=key == "rmse_climatology")
        if skill:
            value = float(row["mse_skill"])
            expected = 1 - (float(row["rmse_forecast"]) / float(row["rmse_climatology"])) ** 2
            _require(math.isfinite(value) and math.isclose(value, expected, rel_tol=1e-8, abs_tol=1e-9),
                     "skill CSV inconsistent/nonfinite")
    return rows


def _cases(provenance, lead, count):
    cases = provenance["initializations"]
    _equal(len(cases), count, "initialization cohort length")
    metadata, previous = [], None
    for case in cases:
        initial = datetime.fromisoformat(case["init_time"])
        _require(initial.year == 2022 and (previous is None or initial > previous),
                 "unique chronological 2022 val initializations required")
        _equal(len(case["valid_times"]), 1, "valid timestamp count")
        valid = datetime.fromisoformat(case["valid_times"][0])
        _require(valid.year == 2022 and valid - initial == timedelta(hours=lead),
                 "2022 val valid timestamp/lead mismatch; test sealed")
        _equal(case["cumulative_reasoning_steps"], [recipe.REASONING_STEPS * (lead // 6)], "case reasoning steps")
        _require(len(case["mse"]) == 1 and len(case["mse"][0]) == 17, "case MSE shape must be 1 x 17")
        for value in case["mse"][0]:
            _number(value)
        metadata.append((case["init_time"], tuple(case["valid_times"])))
        previous = initial
    return metadata


def check_evaluation_receipt(record, *, expected_dir, checkpoint_sha256, protocol, lead,
                             manifest_sha256, denominator=False):
    """Verify real evaluate_local schema and return only paired cohort metadata."""
    folder = _local(record["dir"], expected_dir)
    for filename, key in (("rmse.csv", "rmse_csv_sha256"), ("climatology_skill.csv", "skill_csv_sha256"),
                          ("provenance.json", "provenance_sha256")):
        _hashed(folder / filename, record[key])
    provenance = _json(folder / "provenance.json")
    _flags(provenance)
    count = screen.V3_VAL_COHORTS[str(lead)]
    _equal(record["n_evaluated"], count, "receipt cohort")
    _number(record["elapsed_seconds"])
    expected = {"checkpoint_sha256": _sha(checkpoint_sha256), "training_identity": protocol["train_data_identity"],
                "evaluation_manifest_sha256": manifest_sha256, "split": "val", "lead_hours": [lead],
                "n_evaluated": count, "n_available_windows": count, "step_hours": 6,
                "source_declaration": str(Path(protocol["instance"]) / "source.nc"),
                "inference_options": {"reasoning_steps": recipe.REASONING_STEPS}}
    for key, value in expected.items():
        _equal(provenance[key], value, f"provenance {key}")
    _equal(provenance["elapsed_seconds"], record["elapsed_seconds"], "evaluation elapsed seconds")
    for key in ("process_scale_sidecar_identity", "controller_sha256", "halting_policy",
                "parameter_free_baseline", "boundary_scoring", "intervention"):
        _require(provenance.get(key) is None, f"unexpected evaluation intervention: {key}")
    channels, units = provenance["channels"], provenance["units"]
    _equal(channels, [spec.channel_name for spec in DEFAULT_R7_ERA5_CHANNELS], "exact 17 variables/order")
    _require(len(units) == 17 and all(isinstance(u, str) and u not in ("", "unknown", "normalized") for u in units),
             "17 physical units required")
    _equal(provenance["climatology"]["training_years"], screen.V3_CLIMATOLOGY_YEARS, "train-only climatology years")
    cases = _cases(provenance, lead, count)
    rmse = _table(folder / "rmse.csv", channels, units, lead, count)
    skill = _table(folder / "climatology_skill.csv", channels, units, lead, count, skill=True)
    for index, (forecast, climate) in enumerate(zip(rmse, skill)):
        value = _number(float(forecast["rmse"]), positive=denominator)
        _number(value * value, positive=denominator)
        pooled = math.sqrt(math.fsum(c["mse"][0][index] for c in provenance["initializations"]) / count)
        _require(math.isclose(value, pooled, rel_tol=1e-9, abs_tol=1e-12), "RMSE/case MSE mismatch")
        _require(math.isclose(value, float(climate["rmse_forecast"]), rel_tol=1e-9, abs_tol=1e-12),
                 "RMSE/skill forecast mismatch")
    return {**{key: provenance[key] for key in ("channels", "units", "evaluation_manifest_sha256",
                                               "source_declaration", "selection", "climatology")}, "cases": cases}


def collect_readings(output, protocol):
    """Validate all declared evidence before applying the unchanged shared readings."""
    output = _local(output)
    manifest_sha = _protocol(output, protocol)
    references = _references(output, protocol)
    expected = {output / f"seed{seed}_receipt.json" for seed in recipe.SEEDS}
    _equal(set(output.glob("seed*_receipt.json")), expected, "all three candidate seed receipts")
    receipts = {seed: _json(output / f"seed{seed}_receipt.json") for seed in recipe.SEEDS}
    for seed, receipt in receipts.items():
        _training(output, protocol, receipt, seed)
        for lead in recipe.EVALUATION_LEADS:
            suffix = f"seed{seed}/evaluation/candidate/lead_{lead:03d}h"
            candidate = check_evaluation_receipt(receipt["evaluations"][str(lead)], expected_dir=output / suffix,
                checkpoint_sha256=receipt["checkpoint_sha256"], protocol=protocol, lead=lead, manifest_sha256=manifest_sha)
            for name, (root, result) in references.items():
                reference = result["seeds"][str(seed)]
                arm = "process" if name == "control" else "candidate"
                metadata = check_evaluation_receipt(reference["evaluations"][str(lead)],
                    expected_dir=root / f"seed{seed}/evaluation/{arm}/lead_{lead:03d}h",
                    checkpoint_sha256=reference["checkpoint_sha256"], protocol=protocol,
                    lead=lead, manifest_sha256=manifest_sha, denominator=True)
                for key in candidate:
                    _equal(candidate[key], metadata[key], f"paired {name} {key}")
    all_cells, parent_cells = {}, {}
    for seed in recipe.SEEDS:
        candidate = output / f"seed{seed}/evaluation/candidate"
        all_cells[str(seed)] = screen.paired_cells(candidate, references["control"][0] / f"seed{seed}/evaluation/process",
                                                  recipe.EVALUATION_LEADS)
        parent_cells[str(seed)] = screen.paired_cells(candidate, references["parent"][0] / f"seed{seed}/evaluation/candidate",
                                                     recipe.EVALUATION_LEADS)
    primary, gate = recipe.primary_verdict(all_cells), recipe.gate_verdict(all_cells)
    parent_gate = recipe.gate_verdict(parent_cells)
    return {"paired_cells": all_cells, "parent_relative_cells": parent_cells, "primary_verdict": primary,
            "gate_pre_screen": gate, "parent_relative_gate": parent_gate,
            "decision": recipe.advance_decision(primary, gate),
            "rollout_response": {"gate_failures_vs_control": len(gate["failures"]),
                                 "parent_gate_failures_vs_control": recipe.PARENT_GATE_FAILURES_VS_CONTROL,
                                 "gate_failures_vs_parent": len(parent_gate["failures"]),
                                 "text": recipe.ROLLOUT_RESPONSE_TEXT}}
