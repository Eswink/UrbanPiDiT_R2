"""Identity witness for G1-B (decision 0119 exit 3).

Split out of ``study_r7_s3_visit_gain.py`` when that driver crossed the R-021 400-line soft
cap. This module owns exactly one question — is the ``visit_gain`` switch numerically inert
when it is off — and it answers it on a fixed synthetic batch in the calling process, before
any optimizer step, with ``torch.set_num_threads(1)`` pinned because reductions above one
intra-op thread are order-dependent and "bitwise" would otherwise not be a well-formed claim.
"""
from __future__ import annotations

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hashlib  # noqa: E402
import json  # noqa: E402

from scripts import r7_s3_visit_gain_parent as parent_module  # noqa: E402
from scripts import r7_s3_visit_gain_protocol as proto  # noqa: E402
from scripts.r7_s3_visit_gain_rate import marginal_rate, quiet_machine  # noqa: E402


def identity_witness(body: dict) -> dict:
    """Exit 3, measured in this process before any optimizer step.

    Two claims, both bitwise:

    * the off path reproduces the pre-switch expression on a fixed synthetic batch, with the
      same key set, no new buffers and the same parameter count;
    * ``visit_damped`` at visit 0 equals the off path on that batch, because ``g_0`` is
      exactly 1.0.

    ``torch.set_num_threads(1)`` is pinned and recorded: reductions above one intra-op thread
    are order-dependent, so "bitwise" is only a well-formed claim at a fixed thread count.
    """
    import torch

    from training.r7_autoregressive_runner import _state_digest
    from training.r7_experiment import make_model

    torch.set_num_threads(1)
    spec = dict(body["registry"][0]["model"]["spec"])
    parent_path = Path(body["registry"][0]["parent"]["checkpoint"])
    with torch.random.fork_rng(devices=[]):
        off = make_model("process", {**spec, "visit_gain": None}).eval()
        on = make_model("process", {**spec, "visit_gain": proto.VISIT_DAMPED}).eval()
    archived = torch.load(parent_path, map_location="cpu", weights_only=True)["model"]
    off.load_state_dict(archived, strict=True)
    on.load_state_dict(archived, strict=True)
    off_state, on_state = off.state_dict(), on.state_dict()
    generator = torch.Generator().manual_seed(20261125)
    tokens, grid, channels = 16, 33, 192
    latent = torch.randn(2, tokens, channels, generator=generator) * 0.5
    context = torch.randn(2, grid * grid, channels, generator=generator)
    with torch.no_grad():
        reference = _pre_switch_cell(off.reasoning_cell, latent, context)
        off_out = off.reasoning_cell(latent, context)
        off_out_other_visit = off.reasoning_cell(latent, context, visit_index=3, reasoning_steps=4)
        a_zero = on.reasoning_cell(latent, context, visit_index=0, reasoning_steps=4)
        a_one = on.reasoning_cell(latent, context, visit_index=1, reasoning_steps=4)
    witness = {
        "scientific_claim": False,
        "test_read": False,
        "r": 0,
        "torch_num_threads": int(torch.get_num_threads()),
        "synthetic_batch": {"batch": 2, "tokens": tokens, "context_tokens": grid * grid,
                            "dim": channels, "generator_seed": 20261125},
        "parent_checkpoint": str(parent_path),
        "parent_state_digest": _state_digest(archived),
        "off_path": {
            "bitwise_equal_to_pre_switch_expression": bool(torch.equal(off_out, reference)),
            "bitwise_equal_across_visit_indices": bool(torch.equal(off_out, off_out_other_visit)),
            "key_set_equal": set(off_state) == set(archived),
            "tensors_equal": bool(all(torch.equal(off_state[name], archived[name])
                                      for name in archived)),
            "parameter_count_equal": (sum(p.numel() for p in off.parameters())
                                       == sum(v.numel() for v in archived.values())),
            "no_new_buffers": not any(True for _ in off.reasoning_cell.buffers()),
            "missing_from_parent": sorted(set(off_state) - set(archived))[:8],
            "extra_in_parent": sorted(set(archived) - set(off_state))[:8],
        },
        "visit_damped_k0": {
            "bitwise_equal": bool(torch.equal(a_zero, off_out)),
            "gain_at_zero": proto.visit_gains(proto.VISIT_DAMPED)[0],
            "diverges_at_visit_1": bool(not torch.equal(a_one, off_out)),
        },
        "arms_share_post_parent_state": set(off_state) == set(on_state),
    }
    for field, value in witness["off_path"].items():
        if field.endswith("_equal") or field in ("no_new_buffers",) or field == "tensors_equal":
            if not value:
                raise RuntimeError(f"exit 3a failed at {field}: {witness['off_path']}")
    if not witness["visit_damped_k0"]["bitwise_equal"]:
        raise RuntimeError("exit 3b failed: visit_damped at visit 0 is not the off path")
    return witness


def _pre_switch_cell(cell, z, context):
    """The cell body as it stood before decision 0119, written here rather than imported."""
    z = z + cell.self_attn(cell.n1(z))
    z = cell.cross(z, context)
    return z + cell.ff(cell.n2(z))




def partial_receipt(output: Path, body: dict, seed: int, arm: str, *, reason: str,
             detail: dict) -> dict:
    """The receipt a stopped arm writes, recovered from its own progress log."""
    folder = output / f"seed{seed}/training/{arm}"
    progress = folder / "progress.jsonl"
    rows = []
    if progress.is_file():
        for line in progress.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    usable = [row for row in rows if isinstance(row.get("update"), int)
              and isinstance(row.get("elapsed_seconds"), (int, float))]
    reached = int(usable[-1]["update"]) if usable else 0
    elapsed = float(usable[-1]["elapsed_seconds"]) if usable else 0.0
    record = {
        "scientific_claim": False, "test_read": False, "r": 0,
        "seed": int(seed), "arm": arm, "status": "partial", "reason": reason,
        "updates_reached": reached, "updates_planned": int(body["updates"]),
        "elapsed_seconds": elapsed,
        "seconds_per_update": (round(elapsed / reached, 4) if reached else None),
        "gpu_hours_in_failed_attempt": round(elapsed / 3_600.0, 4),
        "progress_file": str(progress), "progress_rows": len(usable), "detail": detail,
        "limitations": ["a partial arm scored nothing; it is recorded so the ledger counts the "
                        "compute it consumed, and it contributes no reading to any clause"],
    }
    _write_exclusive(output / f"seed{seed}_{arm}_partial_receipt.json", record)
    return record


def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    return proto.freeze(output, started=started, deadline=deadline, code_archive=code_archive,
                        code_commit=code_commit, code_sha256=code_sha256)


EXECUTION_SOURCES = (
    "model/recursive_weather_r7.py", "model/process_forecast_r7.py",
    "model/process_step_r7.py", "model/layers/sdpa.py",
    "model/weather_forecaster_r7.py", "model/coarse_forecast.py",
    "model/coarse_encoder.py", "model/local_solver_state_r7.py",
    "model/process_readout_r7.py", "model/spacetime_conditioning_r7.py",
    "data/r7_long_rollout_dataset.py", "data/r7_zarr_dataset.py", "data/r7_store.py",
    "data/r7_evaluation.py",
    "training/r7_long_rollout_runner.py", "training/r7_recursive_losses.py",
    "training/r7_losses.py", "training/r7_experiment.py", "training/r7_arm_harness.py",
    "training/r7_evaluate.py", "training/r7_boundary_metrics.py",
    "training/r7_climatology_skill.py", "training/r7_rollout_metrics.py",
    "training/r7_simultaneous_stats.py", "training/r7_device_selection.py",
    "scripts/r7_s3_visit_gain_protocol.py", "scripts/r7_s3_visit_gain_parent.py",
    "scripts/r7_s3_visit_gain_verdict.py", "scripts/study_r7_s3_visit_gain.py",
    "scripts/r7_s3_visit_gain_constants.py", "scripts/r7_s3_visit_gain_identity.py",
    "scripts/r7_s3_visit_gain_rate.py",
)


def execution_files() -> list[str]:
    missing = [name for name in EXECUTION_SOURCES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"declared execution sources are missing: {missing}")
    return list(EXECUTION_SOURCES)


def _sha256(path: Path) -> str:
    from training.r7_arm_harness import sha256_file

    return sha256_file(Path(path))



def _identity() -> dict:
    """The cheap identities prepare binds, plus the expensive ones it only records once."""
    from training.r7_experiment import dataset_identity, model_code_digest
    from training.r7_long_rollout_runner import training_code_digest

    train_identity, _ = dataset_identity(proto.TRAIN_MANIFEST)
    val_identity, _ = dataset_identity(proto.VAL_MANIFEST)
    return {
        "execution_files_sha256": {name: _sha256(ROOT / name) for name in execution_files()},
        "model_code_sha256": model_code_digest(),
        "training_code_sha256": training_code_digest(),
        "train_data_identity": train_identity,
        "val_data_identity": val_identity,
        "train_manifest_sha256": _sha256(proto.TRAIN_MANIFEST),
        "val_manifest_sha256": _sha256(proto.VAL_MANIFEST),
        "source_sha256": proto.SOURCE_SHA256,
        "train_windows": proto.train_windows(),
        "parent_migration": {
            "run": str(parent_module.DEFAULT_OUT.relative_to(ROOT)),
            "receipt_sha256": _sha256(parent_module.DEFAULT_OUT / "migration_receipt.json"),
            "state_equality": parent_module.state_equality_evidence(),
        },
    }


def prepared_protocol(output: Path) -> dict:
    """Promote ``preparation_protocol.json`` into ``prepared_protocol.json`` with identities."""
    from training.r7_experiment import canonical_digest

    output = Path(output)
    preliminary = json.loads((output / "preparation_protocol.json").read_text(encoding="utf-8"))
    if preliminary["protocol_sha256"] != canonical_digest(
            {key: value for key, value in preliminary.items() if key != "protocol_sha256"}):
        raise RuntimeError("preparation protocol digest mismatch")
    body = {**preliminary, **_identity()}
    body.pop("protocol_sha256", None)
    body["protocol_sha256"] = canonical_digest(body)
    _write_exclusive(output / "prepared_protocol.json", body)
    return body


def validate_protocol(output: Path) -> dict:
    """Re-check the frozen body against the live tree before any stage runs."""
    from training.r7_experiment import canonical_digest, model_code_digest
    from training.r7_long_rollout_runner import training_code_digest

    body = json.loads((Path(output) / "protocol.json").read_text(encoding="utf-8"))
    if body["protocol_sha256"] != canonical_digest(
            {key: value for key, value in body.items() if key != "protocol_sha256"}):
        raise RuntimeError("protocol digest mismatch")
    if set(body["execution_files_sha256"]) != set(execution_files()):
        raise RuntimeError("incomplete execution source inventory")
    for name, expected in body["execution_files_sha256"].items():
        if _sha256(ROOT / name) != expected:
            raise RuntimeError(f"execution source changed: {name}")
    if body["model_code_sha256"] != model_code_digest():
        raise RuntimeError("model source identity changed")
    if body["training_code_sha256"] != training_code_digest():
        raise RuntimeError("training source identity changed")
    if _sha256(proto.TRAIN_MANIFEST) != body["train_manifest_sha256"]:
        raise RuntimeError("the train manifest changed")
    if _sha256(proto.VAL_MANIFEST) != body["val_manifest_sha256"]:
        raise RuntimeError("the val manifest changed")
    # The registry's pins are the frozen ones: re-derive each seed and refuse a body whose
    # recorded parent no longer matches the archived receipt it was frozen against.
    recorded = {(int(item["seed"]), item["arm"]): item["parent"] for item in body["registry"]}
    for seed in proto.SEEDS:
        path, digest = proto.parent_pin(seed)
        for arm, _ in proto.ARMS:
            entry = recorded.get((int(seed), arm)) or {}
            if Path(entry.get("checkpoint", "")).name != path.name:
                raise RuntimeError(f"seed {seed} arm {arm} names a different parent than its pin")
            if entry.get("sha256") != digest:
                raise RuntimeError(f"seed {seed} arm {arm} parent digest drifted from its pin")
    return body




def _sha256(path):
    """File digest via the round harness, so this module and the driver agree on bytes."""
    from training.r7_arm_harness import sha256_file

    return sha256_file(Path(path))


def _write_exclusive(path, body):
    """Exclusive JSON write: an artifact is created once and never re-rendered."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(body, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
