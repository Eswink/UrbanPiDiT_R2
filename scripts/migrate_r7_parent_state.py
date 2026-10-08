"""Model-only install of the exported v3-BD parent state under the current code.

The registered v3-BD 1600-update parent checkpoints were trained at revision
``66836d2`` (``model_code_sha256 3ddab46b...``). The later climatology-anchor
revision changed that digest, so ``training.r7_experiment.load_checkpoint`` - whose
whole job is to refuse a foreign implementation - rejects them by design. This tool
performs the documented model-only migration instead of weakening that guard:

1. it reads the state the *archived* revision exported with its own loader;
2. it re-checks the source checkpoint SHA256 pins and the exported state digests;
3. it rebuilds the current ``make_model`` template and requires the tensor keys,
   shapes, dtypes and the module-semantics digest to match the archived contract
   before a single weight is installed;
4. it writes a fresh ``r7-local-v1`` checkpoint carrying the current digest, a
   self-consistent contract/signature, and the archived provenance;
5. it never carries the optimizer, cursor or RNG forward.

Scientific value: none. This is an identity-translation step whose only output is a
checkpoint the current training harness is allowed to load.
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

SEEDS = (41, 42, 43)
ENDPOINT = 1600
PARENT_PINS = {
    41: "637857c5d5cc8b3c9e33448cae4dd1b53aed5f525639d6911f03d7420492bdac",
    42: "8f7369fbbb3983b9e838d93a2e35d608871b01b580686a13581181950849007e",
    43: "08cc7db9629fca86f3634c0b102a206d292f09a5370a93d4a89d9477c6b50252",
}
ARCHIVED_MODEL_CODE = "3ddab46b1e4c2c7e66449e39ab8247c9c7e642f45023c1c9cc14bca8f74fd217"
ARCHIVED_CODE_COMMIT = "66836d29dd257a11b0c946c8af2622b35042a5ce"
DEFAULT_EXPORT = ROOT / "outputs/r7_s3_rollout_dose_parent_export_20261009_attempt02"
DEFAULT_OUT = ROOT / "outputs/r7_s3_rollout_dose_parent_migrated_20261009_attempt01"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def migrate(export_root: Path, out_root: Path) -> dict:
    import torch

    from training.r7_arm_harness import sha256_file
    from training.r7_autoregressive_runner import _check_model_semantics, _state_digest
    from training.r7_experiment import canonical_digest, make_model, model_code_digest, save_exclusive

    export_root, out_root = Path(export_root).resolve(), Path(out_root).resolve()
    if out_root.exists():
        raise FileExistsError(f"fresh migration output only: {out_root}")
    manifest = json.loads((export_root / "export_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "r7-rollout-dose-parent-export-v1":
        raise ValueError("unexpected export manifest format")
    if manifest.get("export_code_commit") != ARCHIVED_CODE_COMMIT:
        raise ValueError("export was not produced by the archived parent revision")
    if manifest.get("export_model_code_sha256") != ARCHIVED_MODEL_CODE:
        raise ValueError("export model digest is not the archived parent digest")
    if sha256_file(export_root / "parent_state_export.pt") != manifest["export_sha256"]:
        raise ValueError("exported state file drifted from its recorded SHA256")
    if sha256_file(ROOT / "scripts/export_r7_parent_state.py") != manifest["export_tool_sha256"]:
        raise ValueError("committed exporter bytes differ from the exporter that produced this export")
    payload = torch.load(export_root / "parent_state_export.pt", map_location="cpu", weights_only=True)
    out_root.mkdir(parents=True, exist_ok=False)
    current = model_code_digest()
    receipt = {"format": "r7-rollout-dose-parent-migration-v1", "scientific_claim": False, "test_read": False,
               "archived_code_commit": ARCHIVED_CODE_COMMIT, "archived_model_code_sha256": ARCHIVED_MODEL_CODE,
               "current_model_code_sha256": current, "export_sha256": manifest["export_sha256"],
               "export_manifest_sha256": sha256_file(export_root / "export_manifest.json"),
               "migration_tool_sha256": sha256_file(Path(__file__)), "seeds": {}}
    for seed in SEEDS:
        record = manifest["seeds"][str(seed)]
        source = Path(record["source_path"])
        if sha256_file(source) != record["source_sha256"] or record["source_sha256"] != PARENT_PINS[seed]:
            raise ValueError(f"seed {seed} parent checkpoint no longer matches its pin")
        entry = payload[str(seed)]
        if entry["source_sha256"] != record["source_sha256"] or entry["updates"] != ENDPOINT:
            raise ValueError(f"seed {seed} export entry does not match its manifest record")
        if entry["archived_model_code_sha256"] != ARCHIVED_MODEL_CODE:
            raise ValueError(f"seed {seed} archived digest differs from the registered parent revision")
        if canonical_digest(entry["contract"]) != entry["signature"]:
            raise ValueError(f"seed {seed} archived contract/signature is not self-consistent")
        state = entry["model"]
        if _state_digest(state) != record["state_digest"]:
            raise ValueError(f"seed {seed} exported state digest differs from the manifest")
        spec = entry["contract"]["model"]
        with torch.random.fork_rng(devices=[]):
            template = make_model("process", spec)
        expected = template.state_dict()
        if set(expected) != set(state):
            raise ValueError(f"seed {seed} tensor keys differ between archived and current model code")
        for name, reference in expected.items():
            value = state[name]
            if (not torch.is_tensor(value) or value.shape != reference.shape or value.dtype != reference.dtype
                    or not torch.isfinite(value).all()):
                raise ValueError(f"seed {seed} tensor {name} shape/dtype/finite contract differs")
        template.load_state_dict(state, strict=True)
        semantics = _check_model_semantics(template, template)
        contract = dict(entry["contract"])
        contract["model_only_migration"] = {
            "archived_code_commit": ARCHIVED_CODE_COMMIT,
            "archived_model_code_sha256": ARCHIVED_MODEL_CODE,
            "archived_contract_signature": entry["signature"],
            "source_checkpoint_path": str(source), "source_checkpoint_sha256": record["source_sha256"],
            "exported_state_digest": record["state_digest"],
            "operation": "model-only state install; no optimizer, cursor or RNG carried forward",
            "checks": ["source SHA256 pin", "archived loader export", "tensor keys/shapes/dtypes/finite",
                       "current module tree semantics", "strict load_state_dict"],
        }
        contract["model_semantics_sha256"] = semantics
        contract["model_code_sha256"] = current
        destination = out_root / f"seed{seed}" / f"parent_update_{ENDPOINT:07d}.pt"
        save_exclusive(destination, {"format": "r7-local-v1", "model": state, "contract": contract,
                                     "signature": canonical_digest(contract), "updates": ENDPOINT,
                                     "epoch": int(entry.get("epoch", 0) or 0), "cursor": 0})
        reloaded = torch.load(destination, map_location="cpu", weights_only=True)
        if (reloaded["model_code_sha256"] != current or reloaded["updates"] != ENDPOINT
                or reloaded["signature"] != canonical_digest(reloaded["contract"])
                or _state_digest(reloaded["model"]) != record["state_digest"]):
            raise ValueError(f"seed {seed} migrated checkpoint failed its own read-back")
        receipt["seeds"][str(seed)] = {
            "source_checkpoint_path": str(source), "source_checkpoint_sha256": record["source_sha256"],
            "archived_contract_signature": entry["signature"], "state_digest": record["state_digest"],
            "migrated_path": str(destination), "migrated_sha256": sha256_file(destination),
            "n_tensors": len(state), "tensor_bytes": record["tensor_bytes"],
            "archived_model_semantics_sha256": entry["contract"].get("model_semantics_sha256"),
            "current_model_semantics_sha256": semantics,
            "semantics_equal": entry["contract"].get("model_semantics_sha256") == semantics,
            "archived_training_code_sha256": entry["contract"].get("training_code_sha256"),
        }
    with (out_root / "migration_receipt.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    receipt = migrate(args.export_root, args.out)
    print(json.dumps({"status": "success",
                      "migrated": {s: v["migrated_sha256"] for s, v in receipt["seeds"].items()},
                      "semantics_equal": {s: v["semantics_equal"] for s, v in receipt["seeds"].items()}},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
