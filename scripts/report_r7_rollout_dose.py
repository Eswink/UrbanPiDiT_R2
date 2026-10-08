"""Read-only readings for the rollout-dose screen whose frozen reading phase failed.

The frozen screen pipeline computes its verdicts inside a "reading" worker that calls
``training.r7_long_rollout_evidence.collect_long_readings``. That collector re-derives
the *parent* reference pins from the registered v3-BD run and requires the protocol's
parent pins to be the original v3-BD checkpoint paths and SHA256 values, then loads that
checkpoint with ``load_checkpoint``. After the climatology-anchor revision changed the
model digest those two requirements cannot both hold for a model-only migrated parent:
pinning the originals makes the load refuse them, and pinning the migrated copies makes
the reference comparison fail. Attempt02 therefore trained and evaluated all three seeds
and then failed exactly there, with the failure preserved.

This tool does not replace that failure and does not touch the frozen protocol. It
re-derives the same readings from the same artifacts, read-only, and delegates every
verdict to the *same* frozen functions the screen uses
(``scripts.study_r7_s3_v3_rollout_ft.primary_verdict`` / ``gate_verdict`` /
``advance_decision`` and ``training.r7_s3_v3_screen.paired_cells``). The one difference
from the frozen collector is the parent identity path: the registered v3-BD result is
still pinned by SHA256 as provenance, while the checkpoint identity check follows the
recorded model-only migration instead of the original file.

Scientific value: none beyond reporting what the completed run already produced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import study_r7_s3_rollout_dose as study  # noqa: E402
from scripts import study_r7_s3_v3_rollout_ft as recipe  # noqa: E402
from training import r7_s3_v3_screen as screen  # noqa: E402
from training.r7_rollout_evidence import (  # noqa: E402
    _equal, _flags, _hashed, _json, _local, _number, _require, _sha, check_evaluation_receipt,
)


def _protocol(output: Path) -> dict:
    from training.r7_experiment import canonical_digest
    body = _json(output / "protocol.json")
    _flags(body)
    _equal(body["protocol_sha256"], canonical_digest({k: v for k, v in body.items()
                                                      if k != "protocol_sha256"}), "protocol digest")
    _equal(body["kind"], "screen", "frozen execution kind")
    _equal(body["seeds"], list(recipe.SEEDS), "frozen seed inventory")
    _equal(body["evaluation_leads_hours"], list(recipe.EVALUATION_LEADS), "frozen lead inventory")
    _equal(body["source_sha256"], screen.V3_SOURCE_SHA256, "source identity")
    _equal(body["train_data_identity"], screen.V3_DATA_IDENTITY, "train data identity")
    _equal(body["test_read"], False, "test must stay sealed")
    for key, value in {"text": recipe.PRIMARY_DECISION_TEXT, "variable": "t2m", "region": "full",
                       "leads_hours": [6, 12]}.items():
        _equal(body["decision"]["primary"][key], value, f"primary reading {key}")
    for key, value in {"text": recipe.GATE_DECISION_TEXT, "relative_mse_change_max": 0.0,
                       "variables": ["u10", "v10", "mslp"],
                       "leads_hours": list(recipe.EVALUATION_LEADS)}.items():
        _equal(body["decision"]["gate_pre_screen"][key], value, f"gate reading {key}")
    return body


def _control(output: Path, protocol: dict):
    """Registered v3-D3 readings: result, cohort sizes and per-lead CSV pins."""
    from training.r7_arm_harness import sha256_file
    pins = protocol["arms"]["control"]["pins"]
    path = _local(pins["d3_result_path"])
    _equal(path.name, "result.json", "control result filename")
    result = _json(_hashed(path, pins["d3_result_sha256"]))
    _flags(result)
    root = path.parent
    _require(not output.is_relative_to(root) and not root.is_relative_to(output),
             "candidate and control roots must be disjoint")
    _equal(set(result["seeds"]), {str(s) for s in recipe.SEEDS}, "control seeds")
    _equal(result["train_data_identity"], protocol["train_data_identity"], "control train identity")
    _equal(result["val_data_identity"], protocol["val_data_identity"], "control val identity")
    for seed in recipe.SEEDS:
        receipt = result["seeds"][str(seed)]
        entry = pins["seeds"][str(seed)]
        _equal(entry["initial_state_sha256"], receipt["initial_state_sha256"], "control initial state")
        for lead in recipe.EVALUATION_LEADS:
            record = receipt["evaluations"][str(lead)]
            _equal(record["n_evaluated"], screen.V3_VAL_COHORTS[str(lead)], "control cohort")
            folder = _local(record["dir"], root / f"seed{seed}/evaluation/process/lead_{lead:03d}h")
            _hashed(folder / "rmse.csv", record["rmse_csv_sha256"])
            _equal(entry["leads"][str(lead)], {"rmse_csv_sha256": record["rmse_csv_sha256"],
                                               "evaluation_dir": record["dir"]}, "control CSV pins")
    return root, result, sha256_file(path)


def _parent(output: Path, protocol: dict):
    """Registered v3-BD provenance plus the recorded model-only migrated checkpoints."""
    from training.r7_arm_harness import sha256_file
    from training.r7_autoregressive_runner import _state_digest
    from training.r7_experiment import canonical_digest, load_checkpoint
    pins = protocol["arms"]["parent"]["pins"]
    registered = _json(_hashed(_local(pins["parent_result_path"]), pins["parent_result_sha256"]))
    _equal(len(registered["gate_pre_screen"]["failures"]), recipe.PARENT_GATE_FAILURES_VS_CONTROL,
           "registered parent gate count")
    migration = _json(_hashed(_local(pins["model_only_migration"]["receipt_path"]),
                              pins["model_only_migration"]["receipt_sha256"]))
    _equal(migration["archived_code_commit"], pins["model_only_migration"]["archived_code_commit"],
           "migration archived commit")
    _equal(migration["export_sha256"], pins["model_only_migration"]["export_sha256"],
           "migration export digest")
    for seed in recipe.SEEDS:
        record = migration["seeds"][str(seed)]
        _equal(record["source_checkpoint_sha256"], recipe.PARENT_CHECKPOINT_SHA256[seed],
               "migration source pin")
        pin = pins["checkpoints"][str(seed)]
        _hashed(_local(pin["path"]), pin["sha256"])
        _equal(record["migrated_path"], pin["path"], "migration output path")
        _equal(record["migrated_sha256"], pin["sha256"], "migration output pin")
        saved = load_checkpoint(pin["path"])
        _equal(saved["updates"], recipe.PARENT_ENDPOINT_UPDATES, "migrated parent endpoint")
        _equal(_state_digest(saved["model"]), record["state_digest"], "migrated parent state")
    return registered, migration, sha256_file(_local(pins["model_only_migration"]["receipt_path"]))


def _seed(output: Path, protocol: dict, seed: int, control_root: Path, parent_root: Path,
          parent_state: dict, manifest_sha: str):
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest, load_checkpoint
    from training.r7_autoregressive_runner import _state_digest
    candidate = protocol["arms"]["candidate"]
    receipt = _json(output / f"seed{seed}_receipt.json")
    _equal(receipt["seed"], seed, "candidate seed")
    folder = _local(receipt["training_dir"], output / f"seed{seed}/training/candidate")
    checkpoint = _local(receipt["checkpoint"], folder / f"update_{candidate['updates']:07d}.pt")
    _hashed(checkpoint, receipt["checkpoint_sha256"])
    report = _json(_hashed(folder / "training_report.json", receipt["training_report_sha256"]))
    _flags(report)
    _equal(report["signature"], canonical_digest(report["contract"]), "training signature")
    saved = load_checkpoint(checkpoint, expected=report["signature"])
    _equal(saved["updates"], candidate["updates"], "candidate endpoint")
    _equal(saved["model_code_sha256"], protocol["model_code_sha256"], "candidate model digest")
    _equal(report["data_identity"], protocol["train_data_identity"], "report data identity")
    for key in ("selected_update", "total_updates", "updates_this_run"):
        _equal(report[key], candidate["updates"], f"report {key}")
    _equal(len(report["losses"]), candidate["updates"], "loss rows")
    _equal([row["update"] for row in report["losses"]], list(range(1, candidate["updates"] + 1)),
           "loss update indices")
    weights = candidate["physical_weights"]
    for row in report["losses"]:
        _equal(len(row["per_step_losses"]), len(weights), "physical loss axes")
        for value in row["per_step_losses"]:
            _number(value)
        _number(row["loss"])
        expected = sum(w * loss for w, loss in zip(weights, row["per_step_losses"]))
        _require(abs(expected - row["loss"]) <= max(1e-6, abs(expected) * 1e-6),
                 "training loss does not match the frozen weighted objective")
    contract = report["contract"]
    _equal(contract["initialization"]["parent_checkpoint_sha256"], receipt["parent_checkpoint_sha256"],
           "imported parent pin")
    _equal(receipt["parent_checkpoint_sha256"],
           protocol["arms"]["parent"]["pins"]["checkpoints"][str(seed)]["sha256"], "protocol parent pin")
    _equal(contract["initial_weights_sha256"], parent_state[str(seed)], "imported parent state digest")
    _equal(contract["model"], candidate["model"]["spec"], "candidate spec")
    evaluations = {}
    for lead in recipe.EVALUATION_LEADS:
        record = receipt["evaluations"][str(lead)]
        evaluations[str(lead)] = check_evaluation_receipt(
            record, expected_dir=output / f"seed{seed}/evaluation/candidate/lead_{lead:03d}h",
            checkpoint_sha256=receipt["checkpoint_sha256"], protocol=protocol, lead=lead,
            manifest_sha256=manifest_sha)
        _hashed(folder / "training_report.json", receipt["training_report_sha256"])
    cells = screen.paired_cells(output / f"seed{seed}/evaluation/candidate",
                               control_root / f"seed{seed}/evaluation/process", recipe.EVALUATION_LEADS)
    parents = screen.paired_cells(output / f"seed{seed}/evaluation/candidate",
                                 parent_root / f"seed{seed}/evaluation/candidate", recipe.EVALUATION_LEADS)
    _equal(receipt["checkpoint_sha256"], sha256_file(checkpoint), "checkpoint pin")
    return cells, parents, evaluations


def report(output: Path, out_root: Path) -> dict:
    output, out_root = _local(Path(output).resolve()), Path(out_root).resolve()
    if out_root.exists():
        raise FileExistsError(f"fresh reporting output only: {out_root}")
    protocol = _protocol(output)
    manifest_sha = hashlib.sha256(_local(protocol["val_manifest"]).read_bytes()).hexdigest()
    control_root, _control_result, control_sha = _control(output, protocol)
    _parent_result, migration, migration_sha = _parent(output, protocol)
    parent_state = {str(seed): migration["seeds"][str(seed)]["state_digest"] for seed in recipe.SEEDS}
    parent_root = recipe.PARENT_RUN
    cells, parents, evaluations = {}, {}, {}
    for seed in recipe.SEEDS:
        cells[str(seed)], parents[str(seed)], evaluations[str(seed)] = _seed(
            output, protocol, seed, control_root, parent_root, parent_state, manifest_sha)
    primary, gate = recipe.primary_verdict(cells), recipe.gate_verdict(cells)
    body = {"format": "r7-rollout-dose-readings-v1", "scientific_claim": False, "test_read": False,
            "protocol_sha256": protocol["protocol_sha256"], "source_attempt": str(output),
            "candidate_updates": protocol["arms"]["candidate"]["updates"],
            "control_result_sha256": control_sha, "migration_receipt_sha256": migration_sha,
            "paired_cells": cells, "parent_relative_cells": parents, "primary_verdict": primary,
            "gate_pre_screen": gate, "parent_relative_gate": recipe.gate_verdict(parents),
            "decision": recipe.advance_decision(primary, gate), "evaluation_receipts": evaluations,
            "deviation": ("The frozen reading worker failed on the shared parent-pin check after the "
                          "model identity change; these readings come from the same artifacts, the same "
                          "frozen verdict functions and the recorded model-only migration."),
            "limitations": ["Development full-region screening on one instance; not independent scientific "
                            "confirmation", "The frozen reading phase of this attempt failed and is "
                            "preserved; this report is a read-only re-derivation, not a rerun",
                            "No test access; three seeds are consistency evidence, not significance"]}
    out_root.mkdir(parents=True, exist_ok=False)
    with (out_root / "readings.json").open("x", encoding="utf-8") as handle:
        json.dump(body, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    body = report(args.attempt, args.out)
    print(json.dumps({"primary_verdict": body["primary_verdict"], "gate_pre_screen": body["gate_pre_screen"],
                      "decision": body["decision"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
