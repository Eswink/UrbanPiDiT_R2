"""Model-only migration of the registered p32,132 incumbent state under the visit-gain revision.

Adding ``visit_gain`` to ``GenericRecursiveCell`` moved ``model_code_sha256`` away from
``59340710...``, and ``training.r7_experiment.load_checkpoint`` refuses a foreign
implementation by design. This tool performs the documented model-only migration instead of
weakening that guard, exactly as decisions 0052/0075/0080 did for earlier model changes.

What it proves, per seed, before a single weight is written:

1. the source checkpoint still matches its archived ``seed{N}_incumbent_receipt.json`` SHA256
   **and** this module's own frozen pin;
2. the archived contract is self-consistent (``signature == canonical_digest(contract)``);
3. the current model tree produces **exactly the same 155 tensor keys, shapes and dtypes** as
   the archived state -- which is what "``visit_gain`` adds no parameter" means at the
   checkpoint level, and the reason ``visit_gain=None`` is a true off switch;
4. the installed state's digest equals the archived one, so the weights are the same bytes and
   only the code identity moved.

It never carries the optimizer, cursor or RNG forward. Scientific value: none.

**Why the off arm is built with ``visit_gain=None`` explicitly.** The archived spec has no
``visit_gain`` key, and the constructor's default is ``None``, so both forms build the same
module. The migration writes the explicit key so that a later reader sees which arm the
weights were installed into rather than having to infer it from an absence.
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

from scripts import r7_s3_visit_gain_protocol as proto  # noqa: E402

FORMAT = "r7-s3-visit-gain-parent-migration-v1"
DEFAULT_OUT = ROOT / "outputs/r7_s3_visit_gain_parent_migrated_20261126_attempt01"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def migrate(out_root: Path) -> dict:
    """Install each seed's archived state under the current code identity."""
    import torch

    from training.r7_arm_harness import sha256_file
    from training.r7_autoregressive_runner import _state_digest
    from training.r7_experiment import (canonical_digest, make_model, model_code_digest,
                                        save_exclusive)

    out_root = Path(out_root).resolve()
    if out_root.exists():
        raise FileExistsError(f"fresh migration output only: {out_root}")
    current = model_code_digest()
    if current == proto.PARENT_MODEL_CODE_SHA:
        raise RuntimeError(
            "the live model digest still equals the archived parent's, so this migration has "
            "nothing to translate; refusing to produce a receipt that would look like an "
            "identity translation while proving nothing")
    out_root.mkdir(parents=True, exist_ok=False)
    receipt = {
        "format": FORMAT,
        "scientific_claim": False,
        "test_read": False,
        "r": 0,
        "operation": "model-only state install; no optimizer, cursor or RNG carried forward",
        "archived_model_code_sha256": proto.PARENT_MODEL_CODE_SHA,
        "current_model_code_sha256": current,
        "migration_tool_sha256": sha256_file(Path(__file__)),
        "protocol_module_sha256": sha256_file(ROOT / "scripts/r7_s3_visit_gain_protocol.py"),
        "parent_run": str(proto.PARENT_RUN.relative_to(ROOT)),
        "parent_record_id": proto.PARENT_RECORD,
        "endpoint_updates": proto.PARENT_UPDATES,
        "checks": ["source SHA256 vs archived receipt", "source SHA256 vs frozen pin",
                   "archived contract/signature self-consistency",
                   "tensor keys/shapes/dtypes/finite against the current tree",
                   "strict load_state_dict", "installed state digest vs archived digest",
                   "read-back digest and code identity"],
        "seeds": {},
    }
    for seed in proto.SEEDS:
        source, pinned = proto.parent_pin(seed)
        archived = torch.load(source, map_location="cpu", weights_only=True)
        if archived.get("format") != "r7-local-v1":
            raise ValueError(f"seed {seed} source is not an r7-local-v1 checkpoint")
        if archived.get("model_code_sha256") != proto.PARENT_MODEL_CODE_SHA:
            raise ValueError(f"seed {seed} source was not trained under the archived revision")
        if int(archived.get("updates", -1)) != proto.PARENT_UPDATES:
            raise ValueError(f"seed {seed} source is not at {proto.PARENT_UPDATES} updates")
        contract = dict(archived["contract"])
        if canonical_digest(contract) != archived.get("signature"):
            raise ValueError(f"seed {seed} archived contract/signature is not self-consistent")
        state = archived["model"]
        archived_digest = _state_digest(state)
        spec = dict(contract["model"])
        spec.pop("visit_gain", None)
        spec["visit_gain"] = None
        with torch.random.fork_rng(devices=[]):
            template = make_model("process", spec)
        expected = template.state_dict()
        if set(expected) != set(state):
            missing = sorted(set(expected) - set(state))[:5]
            extra = sorted(set(state) - set(expected))[:5]
            raise ValueError(f"seed {seed} tensor keys differ between archived and current "
                             f"model code; missing={missing} extra={extra}")
        for name, reference in expected.items():
            value = state[name]
            if (not torch.is_tensor(value) or value.shape != reference.shape
                    or value.dtype != reference.dtype or not torch.isfinite(value).all()):
                raise ValueError(f"seed {seed} tensor {name} shape/dtype/finite contract differs")
        template.load_state_dict(state, strict=True)
        installed = template.state_dict()
        if _state_digest(installed) != archived_digest:
            raise ValueError(f"seed {seed} installed state digest differs from the archived one")
        migrated_contract = dict(contract)
        migrated_contract["model"] = spec
        migrated_contract["model_only_migration"] = {
            "archived_model_code_sha256": proto.PARENT_MODEL_CODE_SHA,
            "archived_contract_signature": archived.get("signature"),
            "archived_state_digest": archived_digest,
            "source_checkpoint_path": str(source),
            "source_checkpoint_sha256": pinned,
            "operation": "model-only state install; no optimizer, cursor or RNG carried forward",
            "checks": receipt["checks"],
        }
        migrated_contract["model_code_sha256"] = current
        destination = out_root / f"seed{seed}" / f"parent_update_{proto.PARENT_UPDATES:07d}.pt"
        save_exclusive(destination, {
            "format": "r7-local-v1",
            "model": state,
            "contract": migrated_contract,
            "signature": canonical_digest(migrated_contract),
            "updates": proto.PARENT_UPDATES,
            "epoch": int(archived.get("epoch", 0) or 0),
            "cursor": 0,
        })
        reloaded = torch.load(destination, map_location="cpu", weights_only=True)
        readback_digest = _state_digest(reloaded["model"])
        if (reloaded["model_code_sha256"] != current
                or int(reloaded["updates"]) != proto.PARENT_UPDATES
                or reloaded["signature"] != canonical_digest(reloaded["contract"])
                or readback_digest != archived_digest):
            raise ValueError(f"seed {seed} migrated checkpoint failed its own read-back")
        receipt["seeds"][str(seed)] = {
            "source_checkpoint_path": str(source),
            "source_checkpoint_sha256": pinned,
            "archived_contract_signature": archived.get("signature"),
            "archived_state_digest": archived_digest,
            "readback_state_digest": readback_digest,
            "state_digest_equal": readback_digest == archived_digest,
            "migrated_path": str(destination),
            "migrated_sha256": sha256_file(destination),
            "n_tensors": len(state),
            "tensor_bytes": int(sum(value.numel() * value.element_size()
                                    for value in state.values())),
            "visit_gain": spec["visit_gain"],
        }
    with (out_root / "migration_receipt.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return receipt


def state_equality_evidence(out_root: Path | None = None) -> dict:
    """D2: prove each seed's installed digest equals the archived parent's, bit for bit.

    Reads the migration receipt back and re-derives every claim from the files on disk rather
    than trusting the receipt's own prose: the source checkpoint's bytes, its archived receipt's
    recorded SHA256, and the migrated file's tensor digest are all checked again here.
    """
    import torch

    from training.r7_arm_harness import sha256_file
    from training.r7_autoregressive_runner import _state_digest
    from training.r7_experiment import model_code_digest

    root = Path(out_root or DEFAULT_OUT)
    receipt_path = root / "migration_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"the migration receipt is missing: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    current = model_code_digest()
    evidence = {
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "archived_model_code_sha256": proto.PARENT_MODEL_CODE_SHA,
        "current_model_code_sha256": current,
        "digest_moved": current != proto.PARENT_MODEL_CODE_SHA,
        "seeds": {},
    }
    for seed in proto.SEEDS:
        record = receipt["seeds"][str(seed)]
        source = Path(record["source_checkpoint_path"])
        migrated = Path(record["migrated_path"])
        archived_receipt = proto.parent_receipt(seed)
        archived = torch.load(source, map_location="cpu", weights_only=True)
        installed = torch.load(migrated, map_location="cpu", weights_only=True)
        source_digest = _state_digest(archived["model"])
        installed_digest = _state_digest(installed["model"])
        entry = {
            "source_sha256_recheck": sha256_file(source),
            "source_matches_archived_receipt":
                sha256_file(source) == archived_receipt.get("checkpoint_sha256"),
            "source_matches_frozen_pin": sha256_file(source) == proto.PARENT_CHECKPOINT_SHA256[seed],
            "source_state_digest": source_digest,
            "installed_state_digest": installed_digest,
            "state_digest_equal": source_digest == installed_digest,
            "key_set_equal": set(archived["model"]) == set(installed["model"]),
            "tensor_count": len(archived["model"]),
            "installed_code_identity_is_current": installed["model_code_sha256"] == current,
            "installed_endpoint": int(installed["updates"]),
            "installed_visit_gain": installed["contract"]["model"].get("visit_gain"),
        }
        entry["bitwise_identical"] = bool(
            entry["state_digest_equal"] and entry["key_set_equal"]
            and entry["source_matches_archived_receipt"] and entry["source_matches_frozen_pin"]
            and entry["installed_code_identity_is_current"])
        evidence["seeds"][str(seed)] = entry
    evidence["all_seeds_bitwise_identical"] = all(
        entry["bitwise_identical"] for entry in evidence["seeds"].values())
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="fresh exclusive migration directory")
    parser.add_argument("--evidence", action="store_true",
                        help="only re-derive the state-equality evidence from an existing run")
    args = parser.parse_args()
    if args.evidence:
        print(json.dumps(state_equality_evidence(args.out), indent=2, ensure_ascii=False))
        return 0
    receipt = migrate(args.out)
    evidence = state_equality_evidence(args.out)
    print(json.dumps({
        "status": "success",
        "output": str(args.out),
        "state_digest_equal": {seed: entry["state_digest_equal"]
                               for seed, entry in receipt["seeds"].items()},
        "all_seeds_bitwise_identical": evidence["all_seeds_bitwise_identical"],
        "current_model_code_sha256": receipt["current_model_code_sha256"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
