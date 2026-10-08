"""Model-only export of the registered v3-BD 1600-update parent state.

Run with cwd at the archived code revision 66836d2 whose ``model_code_sha256`` is
``3ddab46b...`` - the digest the parent checkpoints were trained under. Using that
revision's own ``load_checkpoint`` proves the exported tensors are exactly what the
training code would load; the later revision's model-only install is then checked
against the current module tree separately.

The repository root is passed in, never hard-coded, so the tool carries no host
path. No optimizer, cursor or RNG is exported, and no scientific value is produced
here: the only output is the state the migration step consumes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from training.r7_arm_harness import sha256_file  # noqa: E402
from training.r7_autoregressive_runner import _state_digest  # noqa: E402
from training.r7_experiment import load_checkpoint, model_code_digest  # noqa: E402

SEEDS = (41, 42, 43)
ENDPOINT = 1600
ARCHIVED_MODEL_CODE = "3ddab46b1e4c2c7e66449e39ab8247c9c7e642f45023c1c9cc14bca8f74fd217"
ARCHIVED_CODE_COMMIT = "66836d29dd257a11b0c946c8af2622b35042a5ce"
PARENT_RUN_RELATIVE = "outputs/r7_s3_v3_budget_dose_20261006_attempt01"
PINS = {
    41: "637857c5d5cc8b3c9e33448cae4dd1b53aed5f525639d6911f03d7420492bdac",
    42: "8f7369fbbb3983b9e838d93a2e35d608871b01b580686a13581181950849007e",
    43: "08cc7db9629fca86f3634c0b102a206d292f09a5370a93d4a89d9477c6b50252",
}
EXPECTED_DATA = "2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac"
EXPECTED_SOURCE = "bc2ff9cfadcce604fc243bb999b3c430d5164d17fcf1de201273716a5db065f8"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="repository root that owns outputs/")
    parser.add_argument("--out", type=Path, required=True, help="fresh exclusive export directory")
    args = parser.parse_args()
    repo, out = Path(args.repo).resolve(), Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"fresh export output only: {out}")
    digest = model_code_digest()
    if digest != ARCHIVED_MODEL_CODE:
        raise SystemExit("this exporter must run under the archived parent code revision")
    parent_run = repo / PARENT_RUN_RELATIVE
    out.mkdir(parents=True, exist_ok=False)
    payload, manifest = {}, {"format": "r7-rollout-dose-parent-export-v1", "scientific_claim": False,
                             "test_read": False, "export_code_commit": ARCHIVED_CODE_COMMIT,
                             "export_model_code_sha256": digest,
                             "export_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             "seeds": {}}
    for seed in SEEDS:
        path = parent_run / f"seed{seed}" / "training" / "candidate" / f"update_{ENDPOINT:07d}.pt"
        observed = sha256_file(path)
        if observed != PINS[seed]:
            raise SystemExit(f"parent checkpoint drifted from its pin: {path}")
        saved = load_checkpoint(path)
        contract = saved["contract"]
        if (saved["updates"] != ENDPOINT or contract["seed"] != seed or contract["mode"] != "l6"
                or contract["data_identity"] != EXPECTED_DATA or contract["source_sha256"] != EXPECTED_SOURCE):
            raise SystemExit(f"parent seed {seed} contract differs from the registered endpoint")
        state = saved["model"]
        payload[str(seed)] = {"model": state, "contract": contract, "signature": saved["signature"],
                              "updates": saved["updates"], "archived_model_code_sha256": saved["model_code_sha256"],
                              "source_path": str(path), "source_sha256": observed}
        manifest["seeds"][str(seed)] = {
            "source_path": str(path), "source_sha256": observed, "updates": saved["updates"],
            "contract_signature": saved["signature"],
            "archived_model_code_sha256": saved["model_code_sha256"],
            "archived_training_code_sha256": contract.get("training_code_sha256"),
            "state_digest": _state_digest(state), "n_tensors": len(state),
            "tensor_bytes": int(sum(value.numel() * value.element_size() for value in state.values())),
        }
    torch.save(payload, out / "parent_state_export.pt")
    manifest["export_sha256"] = sha256_file(out / "parent_state_export.pt")
    with (out / "export_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"status": "success", "export_sha256": manifest["export_sha256"],
                      "state_digests": {s: manifest["seeds"][s]["state_digest"] for s in manifest["seeds"]}},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
