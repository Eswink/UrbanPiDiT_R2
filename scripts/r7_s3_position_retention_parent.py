"""The R4 parent: the same v3-BD 1600 endpoint, installed under the position-retention revision.

Decision 0080 added ``typed_evidence.position_scale`` to ``model/``, which moved
``model_code_sha256`` from ``a38be16c...`` to ``59340710...``. ``training.r7_experiment.
load_checkpoint`` refuses a foreign implementation by design, so R4 -- like every round before
it -- needs the documented model-only migration rather than a weakened guard.

This module pins ``outputs/r7_s3_position_retention_parent_migrated_20261103_attempt01``,
produced by ``scripts/migrate_r7_parent_state.py`` from the same export root the R2 round used.
The witness that the *same weights* were installed is that each seed's 155-tensor
``state_digest`` equals the R2 migration's, which itself equals the FY pin; only the code
identity and the module-tree semantics digest moved. That is what every migration in this
repository records rather than fails on.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIGRATED_PARENT_RUN = ROOT / "outputs/r7_s3_position_retention_parent_migrated_20261103_attempt01"
MIGRATION_RECEIPT = MIGRATED_PARENT_RUN / "migration_receipt.json"
ENDPOINT = 1600
MODEL_CODE_SHA256 = "59340710623ca24c7ee59ebb5711f52193ebfbaa72ed4532b41bdd987f5d9324"
MIGRATION_RECEIPT_SHA256 = "393766eed2f11d183fbf1cfa79ef619d548a145b31c4dd71046f32592b359372"
MIGRATED_PARENT_SHA256 = {
    41: "e81e4904b65c65daf81d74215eb135e7d306277109c3f3544b979ecfefed3b98",
    42: "7e2cd4cc0d07565e19c660dacdf5f67180ee8e5526f81904dc0b37afcdbe95f3",
    43: "ee33f690653c625d4b041fb95276dfb239cb9a21dc140e2bb65d26754a855d0c",
}
MIGRATED_PARENT_STATE_DIGEST = {
    41: "7c416ccb4054cf7e077788a43b02917e6a2dcf2503058152a7f0ab4b0ab70707",
    42: "857d880466f76500ed74c534e757732ae9eff4c2b3f0bbc0fa74b895fcae3623",
    43: "61fdac58b7bfacf828e1eae4968817b1cbd46616442c6d16ac0fbbb63e294e1b",
}
#: The R2 migration's state digests, kept as the independence witness for the three above.
R2_MIGRATED_PARENT_RUN = ROOT / "outputs/r7_s3_anchor_feedback_65x65_parent_migrated_20261101_attempt01"
R2_MODEL_CODE_SHA256 = "a38be16c9a04f5f1704fed6747e3d8da0f8507168464a337a85975de6b8a1a0a"
ARCHIVED_MODEL_CODE = "3ddab46b1e4c2c7e66449e39ab8247c9c7e642f45023c1c9cc14bca8f74fd217"
ARCHIVED_CODE_COMMIT = "66836d29dd257a11b0c946c8af2622b35042a5ce"


def _sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receipt():
    """Read the fresh migration receipt after checking its own identity."""
    digest = _sha256(MIGRATION_RECEIPT)
    if digest != MIGRATION_RECEIPT_SHA256:
        raise RuntimeError(f"migration receipt drifted: {digest} != {MIGRATION_RECEIPT_SHA256}")
    return json.loads(MIGRATION_RECEIPT.read_text(encoding="utf-8"))


def pinned_parent(seed):
    """Return ``(path, sha256)`` for one seed, refusing a pin that moved."""
    seed = int(seed)
    if seed not in MIGRATED_PARENT_SHA256:
        raise ValueError(f"no parent is pinned for seed {seed}")
    path = MIGRATED_PARENT_RUN / f"seed{seed}" / f"parent_update_{ENDPOINT:07d}.pt"
    expected = MIGRATED_PARENT_SHA256[seed]
    if _sha256(path) != expected:
        raise RuntimeError(f"seed {seed} migrated parent drifted from its pin")
    return path, expected


def pins():
    """The parent block a frozen protocol carries."""
    return {"run": str(MIGRATED_PARENT_RUN), "endpoint_updates": ENDPOINT,
            "migration_receipt_sha256": MIGRATION_RECEIPT_SHA256,
            "checkpoints": {str(seed): {"path": str(MIGRATED_PARENT_RUN / f"seed{seed}"
                                                        / f"parent_update_{ENDPOINT:07d}.pt"),
                                        "sha256": sha}
                            for seed, sha in MIGRATED_PARENT_SHA256.items()},
            "state_digest": {str(seed): digest
                             for seed, digest in MIGRATED_PARENT_STATE_DIGEST.items()},
            "model_code_sha256": MODEL_CODE_SHA256,
            "archived_model_code_sha256": ARCHIVED_MODEL_CODE,
            "archived_code_commit": ARCHIVED_CODE_COMMIT}


def install_parent_pins(protocol):
    """Write the parent block into a protocol body and return it."""
    protocol.setdefault("arms", {})["parent"] = pins()
    return protocol


def state_equality_evidence():
    """Prove the migration carries the state the R2 round and the FY pin both froze.

    Three independent comparisons, all required: the fresh receipt's per-seed tensor digest
    equals the R2 migration's; the source checkpoint SHA256 is the same file; and the archived
    export manifest the two migrations consumed is the same bytes. The module-tree semantics
    digest is expected to *differ* and is reported, not asserted.
    """
    fresh = receipt()
    if not (ROOT / R2_MIGRATED_PARENT_RUN).is_dir():
        raise RuntimeError(f"the R2 migration witness is missing: {R2_MIGRATED_PARENT_RUN}")
    r2 = json.loads((R2_MIGRATED_PARENT_RUN / "migration_receipt.json").read_text(encoding="utf-8"))
    evidence = {"fresh_model_code_sha256": fresh["current_model_code_sha256"],
                "r2_model_code_sha256": r2["current_model_code_sha256"],
                "archived_model_code_sha256": fresh["archived_model_code_sha256"],
                "export_manifest_sha256": fresh["export_manifest_sha256"],
                "export_manifest_matches_r2": fresh["export_manifest_sha256"] == r2["export_manifest_sha256"],
                "preserved_parent_model_code_sha256": R2_MODEL_CODE_SHA256,
                "seeds": {}}
    for seed in (41, 42, 43):
        a, b = fresh["seeds"][str(seed)], r2["seeds"][str(seed)]
        entry = {"fresh_state_digest": a["state_digest"],
                 "r2_state_digest": b["state_digest"],
                 "pinned_state_digest": MIGRATED_PARENT_STATE_DIGEST[seed],
                 "state_digest_equal_to_r2": a["state_digest"] == b["state_digest"],
                 "state_digest_equal_to_pin": a["state_digest"] == MIGRATED_PARENT_STATE_DIGEST[seed],
                 "source_checkpoint_sha256": a["source_checkpoint_sha256"],
                 "source_checkpoint_equal_to_r2":
                     a["source_checkpoint_sha256"] == b["source_checkpoint_sha256"],
                 "n_tensors": a["n_tensors"],
                 "archived_model_semantics_sha256": a["archived_model_semantics_sha256"],
                 "current_model_semantics_sha256": a["current_model_semantics_sha256"],
                 "semantics_equal": a["semantics_equal"]}
        evidence["seeds"][seed] = entry
    if not evidence["export_manifest_matches_r2"]:
        raise RuntimeError("the two migrations consumed different export manifests")
    for seed, entry in evidence["seeds"].items():
        if not (entry["state_digest_equal_to_r2"] and entry["state_digest_equal_to_pin"]
                and entry["source_checkpoint_equal_to_r2"]):
            raise RuntimeError(f"seed {seed} migration does not carry the pinned state: {entry}")
    return evidence
