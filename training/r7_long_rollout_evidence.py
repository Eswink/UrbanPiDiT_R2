"""Strict artifact consumption for an explicitly frozen long-rollout screen."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import study_r7_s3_v3_rollout_ft as recipe
from training import r7_s3_v3_screen as screen
from .r7_experiment import canonical_digest, load_checkpoint
from .r7_autoregressive_runner import _state_digest, _module_semantics
from .r7_long_rollout_runner import training_code_digest
from .r7_rollout_evidence import (
    _equal, _flags, _hashed, _json, _local, _number, _references, check_evaluation_receipt,
)


def _validate_saved_model(saved, contract):
    import torch
    from .r7_experiment import make_model
    with torch.random.fork_rng(devices=[]):
        model = make_model(contract["kind"], contract["model"])
    expected, state = model.state_dict(), saved["model"]
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("candidate model state keys differ from its declared template")
    for name, reference in expected.items():
        value = state[name]
        if (not torch.is_tensor(value) or value.shape != reference.shape or value.dtype != reference.dtype
                or value.is_floating_point() and (value.dtype != torch.float32 or not torch.isfinite(value).all())):
            raise ValueError("candidate model state shape/dtype/finite contract differs: " + name)
    model.load_state_dict(state, strict=True)
    _equal(canonical_digest(_module_semantics(model)), contract["model_semantics_sha256"], "model semantics")


def _training(output, protocol, receipt, seed):
    _equal(receipt["seed"], seed, "candidate seed")
    _equal(set(receipt["evaluations"]), {str(l) for l in recipe.EVALUATION_LEADS}, "candidate leads")
    folder = _local(receipt["training_dir"], output / f"seed{seed}/training/candidate")
    updates = protocol["arms"]["candidate"]["updates"]
    path = _local(receipt["checkpoint"], folder / f"update_{updates:07d}.pt")
    _hashed(path, receipt["checkpoint_sha256"])
    report = _json(_hashed(folder / "training_report.json", receipt["training_report_sha256"]))
    _flags(report)
    contract = report["contract"]
    _flags(contract)
    for value in (report, contract):
        if (not isinstance(value.get("limitations"), list) or not value["limitations"]
                or any(not isinstance(item, str) or not item.strip() for item in value["limitations"])):
            raise ValueError("training contract/report nonempty limitations required")
    for key, value in {"optimization": "full-bptt", "internal_k_detach": False,
                       "physical_step_detach": False, "internal_deep_supervision": True,
                       "objective": "deep_supervised_latitude_area_mse"}.items():
        _equal(report[key], value, f"report training semantics {key}")
    _equal(report["signature"], canonical_digest(contract), "training signature")
    saved = load_checkpoint(path, expected=report["signature"])
    _equal(saved["contract"], contract, "checkpoint/report contract")
    _equal(saved["updates"], updates, "checkpoint endpoint")
    _validate_saved_model(saved, contract)
    candidate = protocol["arms"]["candidate"]
    for key in ("protocol_sha256", "source_sha256", "model_code_sha256", "training_code_sha256"):
        _equal(contract[key], protocol[key], f"contract {key}")
    for key in ("protocol_sha256", "source_sha256", "model_code_sha256"):
        _equal(report[key], protocol[key], f"report {key}")
    _equal(contract["training_code_sha256"], training_code_digest(), "current training implementation")
    _equal(contract["data_identity"], protocol["train_data_identity"], "training data identity")
    _equal(report["data_identity"], protocol["train_data_identity"], "report data identity")
    _equal(contract["kind"], candidate["model"]["kind"], "candidate kind")
    _equal(contract["model"], candidate["model"]["spec"], "candidate spec")
    _equal(canonical_digest(contract["model"]), candidate["model"]["spec_canonical_digest"], "spec digest")
    _equal(contract["seed"], seed, "training seed")
    _equal(contract["output_dir"], str(folder), "training output")
    for key in ("mode", "lr", "weight_decay", "batch_size", "clip", "checkpoint_every", "steps", "bf16"):
        _equal(contract[key], candidate[key], f"training option {key}")
    _equal(contract["warmup_updates"], candidate["warmup"], "warmup updates")
    for key in ("selected_update", "total_updates", "updates_this_run"):
        _equal(report[key], updates, f"report {key}")
    _equal(contract["total_updates"], updates, "contract endpoint")
    _equal(report["selected_checkpoint"], str(path), "selected checkpoint")
    _equal(report["selection_split"], None, "selection split")
    _equal(report["resumed_from_updates"], 0, "resume updates")
    _equal(report["parent_optimizer_imported"], False, "parent optimizer import")
    parent = protocol["arms"]["parent"]["pins"]["checkpoints"][str(seed)]
    _hashed(parent["path"], parent["sha256"])
    parent_checkpoint = load_checkpoint(parent["path"])
    initial_state = _state_digest(parent_checkpoint["model"])
    _equal(contract["initial_weights_sha256"], initial_state, "imported parent state")
    _equal(receipt["parent_checkpoint_sha256"], parent["sha256"], "receipt parent pin")
    for key, value in {"parent_checkpoint": parent["path"], "parent_checkpoint_sha256": parent["sha256"],
                       "parent_endpoint_updates": 1600}.items():
        _equal(contract["initialization"][key], value, f"parent initialization {key}")
    _equal(report["initialization"], contract["initialization"], "report initialization")
    for key, value in {"mode": "long_rollout", "physical_steps": len(candidate["physical_weights"]),
                       "physical_weights": candidate["physical_weights"], "step_hours": 6,
                       "fixed_transition_lead_hours": 6, "detach_physical_steps": False,
                       "detach_reasoning_steps": False, "internal_deep_supervision": True,
                       "window_sha256": protocol["train_windows"]["window_sha256"],
                       "windows": protocol["train_windows"]}.items():
        _equal(contract["autoregression"][key], value, f"autoregression {key}")
    _equal(report["windows"], protocol["train_windows"], "report training windows")
    _equal([r["update"] for r in report["losses"]], list(range(1, updates + 1)), "loss updates")
    for row in report["losses"]:
        _equal(len(row["per_step_losses"]), len(candidate["physical_weights"]), "physical loss axes")
        for value in row["per_step_losses"]:
            _number(value)
        _number(row["loss"])
        expected = sum(w * loss for w, loss in zip(candidate["physical_weights"], row["per_step_losses"]))
        if abs(expected - row["loss"]) > max(1e-6, abs(expected) * 1e-6):
            raise ValueError("training loss does not match the frozen weighted objective")


def collect_long_readings(output, protocol):
    output = _local(output)
    _flags(protocol)
    _equal(_json(output / "protocol.json"), protocol, "root protocol")
    _equal(protocol["protocol_sha256"], canonical_digest({k: v for k, v in protocol.items()
                                                        if k != "protocol_sha256"}), "protocol digest")
    _equal(protocol["seeds"], list(recipe.SEEDS), "frozen seed inventory")
    _equal(protocol["evaluation_leads_hours"], list(recipe.EVALUATION_LEADS), "frozen lead inventory")
    _equal(protocol["source_sha256"], screen.V3_SOURCE_SHA256, "source")
    _equal(protocol["train_data_identity"], screen.V3_DATA_IDENTITY, "train data identity")
    primary, gate = protocol["decision"]["primary"], protocol["decision"]["gate_pre_screen"]
    for key, value in {"text": recipe.PRIMARY_DECISION_TEXT, "variable": "t2m", "region": "full",
                       "leads_hours": [6, 12]}.items():
        _equal(primary[key], value, f"primary reading {key}")
    for key, value in {"text": recipe.GATE_DECISION_TEXT, "relative_mse_change_max": 0.0,
                       "variables": ["u10", "v10", "mslp"],
                       "leads_hours": list(recipe.EVALUATION_LEADS)}.items():
        _equal(gate[key], value, f"gate reading {key}")
    manifest_sha = hashlib.sha256(Path(protocol["val_manifest"]).read_bytes()).hexdigest()
    references = _references(output, protocol)
    expected = {output / f"seed{seed}_receipt.json" for seed in recipe.SEEDS}
    _equal(set(output.glob("seed*_receipt.json")), expected, "three candidate seed receipts")
    for seed in recipe.SEEDS:
        receipt = _json(output / f"seed{seed}_receipt.json")
        _training(output, protocol, receipt, seed)
        for lead in recipe.EVALUATION_LEADS:
            candidate = check_evaluation_receipt(receipt["evaluations"][str(lead)],
                expected_dir=output / f"seed{seed}/evaluation/candidate/lead_{lead:03d}h",
                checkpoint_sha256=receipt["checkpoint_sha256"], protocol=protocol,
                lead=lead, manifest_sha256=manifest_sha)
            for name, (root, result) in references.items():
                reference = result["seeds"][str(seed)]
                arm = "process" if name == "control" else "candidate"
                metadata = check_evaluation_receipt(reference["evaluations"][str(lead)],
                    expected_dir=root / f"seed{seed}/evaluation/{arm}/lead_{lead:03d}h",
                    checkpoint_sha256=reference["checkpoint_sha256"], protocol=protocol,
                    lead=lead, manifest_sha256=manifest_sha, denominator=True)
                _equal(candidate, metadata, f"paired {name} metadata")
    cells, parents = {}, {}
    for seed in recipe.SEEDS:
        candidate = output / f"seed{seed}/evaluation/candidate"
        cells[str(seed)] = screen.paired_cells(candidate, references["control"][0] / f"seed{seed}/evaluation/process",
                                               recipe.EVALUATION_LEADS)
        parents[str(seed)] = screen.paired_cells(candidate, references["parent"][0] / f"seed{seed}/evaluation/candidate",
                                                 recipe.EVALUATION_LEADS)
    primary, gate = recipe.primary_verdict(cells), recipe.gate_verdict(cells)
    return {"paired_cells": cells, "parent_relative_cells": parents, "primary_verdict": primary,
            "gate_pre_screen": gate, "parent_relative_gate": recipe.gate_verdict(parents),
            "decision": recipe.advance_decision(primary, gate)}
