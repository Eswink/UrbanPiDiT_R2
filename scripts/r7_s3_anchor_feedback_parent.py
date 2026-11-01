"""The R2 anchor-feedback round's migrated parent, pinned to the current model revision.

Decision 0058 admitted the climatology-anchor anomaly-feedback reevaluation on the 65x65
multi-year store, and plan 0026 schedules it as Track 1's R2 at the matched-compute
constants (updates 16,071, batch 2). The parent this round starts from is the same
v3-BD 1600-update endpoint the FY rounds used, but the repository's ``model/`` tree moved
after R1 froze (commit ``896eabf`` wires the urban static covariates behind an off-by-default
switch; ``6f21f71`` only touches ``data/preprocess/urban_static_aggregate.py``), so
``model_code_sha256`` is now ``a38be16c...`` and the pins the FY rounds used -- produced
under ``60de3b2d...`` -- can no longer be loaded: ``training.r7_experiment.load_checkpoint``
refuses a foreign implementation by design.

This module pins the migration that ``scripts/migrate_r7_parent_state.py`` produced under
the current revision, ``outputs/r7_s3_anchor_feedback_65x65_parent_migrated_20261101_attempt01``.
It does not weaken that guard. :func:`state_equality_evidence` proves the new migration
carries the same parent state the FY rounds froze, by comparing both receipts' source
checkpoints and per-seed tensor ``state_digest`` values against the FY pins. The module-tree
semantics digest differs (the tree gained files), which every migration in this repository
has recorded rather than failed on; the tensor key set, shapes, dtypes and state digests do
not differ, which is the part that decides whether the same weights were installed.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIGRATED_PARENT_RUN = ROOT / "outputs/r7_s3_anchor_feedback_65x65_parent_migrated_20261101_attempt01"
MIGRATION_RECEIPT = MIGRATED_PARENT_RUN / "migration_receipt.json"
#: The migration the FY rounds used, kept as the equality witness for the state below.
FY_RUN = ROOT / "outputs/r7_s3_tendency_input_parent_migrated_20261021_attempt02"
FY_RECEIPT = FY_RUN / "migration_receipt.json"
ENDPOINT = 1600
MODEL_CODE_SHA256 = "a38be16c9a04f5f1704fed6747e3d8da0f8507168464a337a85975de6b8a1a0a"
MIGRATION_RECEIPT_SHA256 = "3e97d78af606ac9dfdd540f0a9cc3a8c1836eb1d64213056f67e1f35e6645550"
MIGRATED_PARENT_SHA256 = {
    41: "7cc3a4973e4acb5ed3c6aa55028bcdae2cdf391e08f2110aa346be02d66d324d",
    42: "dde3ae094a61e0f8c10e6fbed7f0b5a41e3512f3f8f55acf000b613fff2ae082",
    43: "bd968e5d0175d04c18ee45af7b6536d80c9ca2bc147cf6a2d2c46614718c00ba",
}
MIGRATED_PARENT_STATE_DIGEST = {
    41: "7c416ccb4054cf7e077788a43b02917e6a2dcf2503058152a7f0ab4b0ab70707",
    42: "857d880466f76500ed74c534e757732ae9eff4c2b3f0bbc0fa74b895fcae3623",
    43: "61fdac58b7bfacf828e1eae4968817b1cbd46616442c6d16ac0fbbb63e294e1b",
}
#: The FY-round pins, copied verbatim from ``r7_s3_fy_annual_parent`` (frozen in decisions
#: 0050/0052 and used by the matched-compute R1), kept as the equality witness.
FY_PARENT_SHA256 = {
    41: "fb1e4d6522d32db1cc078a7590dd66c41f479da49ff58f2c6c525f8dfceca387",
    42: "d97ea802a24e62d1457854aeb12cf29b83f029fdca4f9903f19117f3585f3d69",
    43: "8aa0fac68609e0502b4c5292d0b6067e59153f74090055e28d08d2f3ac964af5",
}


def _receipt(path):
    return json.loads(path.read_text(encoding="utf-8"))


def state_equality_evidence():
    """Prove the current-revision migration carries the FY rounds' parent state.

    Two independent identities must hold per seed: the migration source checkpoint (the
    v3-BD 1600 endpoint) and the tensor ``state_digest`` of the migrated file. The FY
    receipt is also checked against the pins ``r7_s3_fy_annual_parent`` freezes, so a
    re-run migration cannot pass by re-deriving the same digests from different sources.
    """
    from training.r7_arm_harness import sha256_file
    current, frozen = _receipt(MIGRATION_RECEIPT), _receipt(FY_RECEIPT)
    observed_receipt = sha256_file(MIGRATION_RECEIPT)
    if observed_receipt != MIGRATION_RECEIPT_SHA256:
        raise RuntimeError("the current-revision migration receipt drifted from its pinned SHA256")
    if current.get("current_model_code_sha256") != MODEL_CODE_SHA256:
        raise RuntimeError("the current-revision migration was produced under a different revision")
    evidence = {}
    for seed in MIGRATED_PARENT_STATE_DIGEST:
        now, then = current["seeds"][str(seed)], frozen["seeds"][str(seed)]
        if now["source_checkpoint_sha256"] != then["source_checkpoint_sha256"]:
            raise RuntimeError(f"seed {seed}: the two migrations read different source checkpoints")
        if now["state_digest"] != then["state_digest"] \
                or now["state_digest"] != MIGRATED_PARENT_STATE_DIGEST[seed]:
            raise RuntimeError(f"seed {seed}: the two migrations do not carry the same state")
        if then["migrated_sha256"] != FY_PARENT_SHA256[seed]:
            raise RuntimeError(f"seed {seed}: the FY migration drifted from its frozen pins")
        evidence[str(seed)] = {
            "source_checkpoint_sha256": now["source_checkpoint_sha256"],
            "state_digest": now["state_digest"],
            "current_migrated_sha256": now["migrated_sha256"],
            "fy_migrated_sha256": then["migrated_sha256"],
        }
    return {
        "current_receipt": str(MIGRATION_RECEIPT), "current_receipt_sha256": observed_receipt,
        "fy_receipt": str(FY_RECEIPT), "fy_receipt_sha256": sha256_file(FY_RECEIPT),
        "equal_state_per_seed": evidence,
        "claim": ("the current-revision migration carries the same v3-BD 1600 parent state the "
                  "FY rounds froze; only the embedded code identity differs, so the migrated "
                  "files' own SHA256 values differ while the tensor digests do not"),
        "scientific_claim": False,
    }


def pinned_parent(seed):
    """Resolve and SHA-verify the migrated model-only parent checkpoint for one seed."""
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import model_code_digest
    if seed not in MIGRATED_PARENT_SHA256:
        raise ValueError(f"declared seeds are {sorted(MIGRATED_PARENT_SHA256)}; {seed} is not among them")
    receipt = _receipt(MIGRATION_RECEIPT)
    if receipt.get("current_model_code_sha256") != model_code_digest():
        raise RuntimeError("the pinned migration was produced under a different model revision; "
                           "re-run scripts/migrate_r7_parent_state.py")
    path = MIGRATED_PARENT_RUN / f"seed{seed}" / f"parent_update_{ENDPOINT:07d}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"migrated parent checkpoint missing: {path}")
    observed = sha256_file(path)
    if observed != MIGRATED_PARENT_SHA256[seed]:
        raise RuntimeError(f"migrated parent checkpoint drifted from its pinned SHA256: {path}")
    record = receipt["seeds"][str(seed)]
    if record["state_digest"] != MIGRATED_PARENT_STATE_DIGEST[seed] \
            or record["migrated_sha256"] != observed:
        raise RuntimeError(f"migration receipt disagrees with the pinned parent for seed {seed}")
    return path, observed


def migration_provenance():
    """The provenance block for the receipt that produced this revision's migrated parents."""
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import model_code_digest
    receipt = _receipt(MIGRATION_RECEIPT)
    current = model_code_digest()
    if receipt.get("current_model_code_sha256") != current:
        raise RuntimeError("the pinned migration was produced under a different model revision; "
                           "re-run scripts/migrate_r7_parent_state.py")
    for seed in MIGRATED_PARENT_SHA256:
        record = receipt["seeds"][str(seed)]
        path = MIGRATED_PARENT_RUN / f"seed{seed}" / f"parent_update_{ENDPOINT:07d}.pt"
        if (record["migrated_path"] != str(path)
                or record["migrated_sha256"] != MIGRATED_PARENT_SHA256[seed]):
            raise RuntimeError(f"migration receipt disagrees with the pinned parent for seed {seed}")
    return {
        "receipt_path": str(MIGRATION_RECEIPT), "receipt_sha256": sha256_file(MIGRATION_RECEIPT),
        "export_sha256": receipt["export_sha256"],
        "archived_code_commit": receipt["archived_code_commit"],
        "archived_model_code_sha256": receipt["archived_model_code_sha256"],
        "current_model_code_sha256": current,
        "operation": "archived-revision export, then strict current-code install; no optimizer/cursor/RNG",
        "state_equality_evidence": state_equality_evidence(),
        "rebinding_note": ("plan 0026 R2: the current revision adds the urban static covariate "
                           "trunk switch behind an off-by-default default whose off path is the "
                           "pre-existing computation, so this round changes no model file"),
        "semantics_note": ("the module-tree semantics digest differs from the archived one because "
                           "the tree gained files; every migration in this repository records that "
                           "rather than failing on it, and the tensor key set, shapes, dtypes and "
                           "per-seed state digests do not differ"),
        "region_note": ("the parent was trained on the narrow 65x65 instance; the model spec is "
                        "region-agnostic, so the same state is valid on the 65x65 multi-year store")}


def pins():
    """The frozen three-seed checkpoint map, without touching the filesystem."""
    return {str(seed): {"path": str(MIGRATED_PARENT_RUN / f"seed{seed}"
                                    / f"parent_update_{ENDPOINT:07d}.pt"),
                        "sha256": MIGRATED_PARENT_SHA256[seed]}
            for seed in sorted(MIGRATED_PARENT_SHA256)}


def install_parent_pins(body):
    """Install this revision's parent pins and provenance into a protocol body."""
    body["arms"]["parent"]["pins"]["checkpoints"] = {
        str(seed): {"path": str(MIGRATED_PARENT_RUN / f"seed{seed}"
                                / f"parent_update_{ENDPOINT:07d}.pt"),
                    "sha256": MIGRATED_PARENT_SHA256[seed]}
        for seed in MIGRATED_PARENT_SHA256}
    body["arms"]["parent"]["pins"]["model_only_migration"] = migration_provenance()
