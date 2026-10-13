"""The H-A arm's migrated parent: the v3-BD 1600 endpoint re-installed under the current model digest.

``scripts/study_r7_s3_process_init.py`` is the only S3 arm that changes ``model/process_forecast_r7.py``,
so ``model_code_digest()`` moves and the parent migrated for the earlier revision can no longer be loaded:
``training.r7_experiment.load_checkpoint`` refuses a foreign implementation by design. This module pins the
migration that the committed ``scripts/migrate_r7_parent_state.py`` produced under the current revision; it
does not weaken that guard. The weights are unchanged from the registered parent -- only the recorded digest
and the provenance differ -- and the receipt's revision is re-checked here so a stale pin fails loudly.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIGRATED_PARENT_RUN = ROOT / "outputs/r7_s3_rollout_dose_parent_migrated_20261014_attempt01"
MIGRATION_RECEIPT = MIGRATED_PARENT_RUN / "migration_receipt.json"
ENDPOINT = 1600
MIGRATED_PARENT_SHA256 = {
    41: "3bb09c1968377ece37b5cf8d48eef34e86f81d527c7e76b2f8bf8c67da1dc05d",
    42: "91667a680d10dc67ca9fa2ddaa33f445e988fb8cd3515b4cc15f0fe21e09b05d",
    43: "b1a716bc7e36b83420e410b3860d72d637874418b538de577a5359bf20ca647a",
}


def install_parent_pins(body):
    """Replace the inherited migrated-parent checkpoint pins with this revision's migration.

    ``interior._identity()`` inherits the pins recorded for the earlier revision; the frozen protocol
    must pin the checkpoint it can actually load, so the driver calls this before freezing.
    """
    body["arms"]["parent"]["pins"]["checkpoints"] = {
        str(seed): {"path": str(MIGRATED_PARENT_RUN / f"seed{seed}" / f"parent_update_{ENDPOINT:07d}.pt"),
                    "sha256": MIGRATED_PARENT_SHA256[seed]}
        for seed in MIGRATED_PARENT_SHA256}


def pinned_parent(seed):
    """Resolve and SHA-verify the migrated model-only parent checkpoint for one seed."""
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import model_code_digest
    if seed not in MIGRATED_PARENT_SHA256:
        raise ValueError(f"declared seeds are {sorted(MIGRATED_PARENT_SHA256)}; {seed} is not among them")
    receipt = json.loads(MIGRATION_RECEIPT.read_text(encoding="utf-8"))
    if receipt.get("current_model_code_sha256") != model_code_digest():
        raise RuntimeError("the pinned migration was produced under a different model revision; "
                           "re-run scripts/migrate_r7_parent_state.py")
    path = MIGRATED_PARENT_RUN / f"seed{seed}" / f"parent_update_{ENDPOINT:07d}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"migrated parent checkpoint missing: {path}")
    observed = sha256_file(path)
    if observed != MIGRATED_PARENT_SHA256[seed]:
        raise RuntimeError(f"migrated parent checkpoint drifted from its pinned SHA256: {path}")
    return path, observed
