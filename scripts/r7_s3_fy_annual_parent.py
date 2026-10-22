"""The FY annual-coverage round's migrated parent, pinned to the current model revision.

Decision 0050 froze this round against the migration
``outputs/r7_s3_rollout_dose_parent_migrated_20261015_attempt04``, which was produced under
``model_code_sha256 3fe747b2...``. Decision 0052 rebinds the round to the revision at HEAD,
``60de3b2d...``: two switches were added to ``model/`` after ``3fe747b2...``
(``tendency_input`` and ``decoder_base_mode``) and both are inert at their registered
defaults, so this round still changes no model file — but the archived checkpoint can no
longer be loaded, because ``training.r7_experiment.load_checkpoint`` refuses a foreign
implementation by design.

Rather than re-run the migration, this module pins the migration the tendency round already
produced under the current revision,
``outputs/r7_s3_tendency_input_parent_migrated_20261021_attempt02``. It is the **same**
parent state: :func:`state_equality_evidence` reads both receipts and asserts that the
source checkpoints and the per-seed tensor ``state_digest`` values match the attempt04 pins
decision 0050 froze. Only the embedded code identity differs, which is why the migrated
files' own SHA256 values differ.

The guard is not weakened: :func:`pinned_parent` still compares the receipt's
``current_model_code_sha256`` with ``model_code_digest()`` and the file bytes with their
pinned SHA256, and a checkout whose ``model/`` differs from ``60de3b2d...`` fails here
rather than silently loading a foreign checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIGRATED_PARENT_RUN = ROOT / "outputs/r7_s3_tendency_input_parent_migrated_20261021_attempt02"
MIGRATION_RECEIPT = MIGRATED_PARENT_RUN / "migration_receipt.json"
#: The migration decision 0050 froze, kept as the equality witness for the state below.
ATTEMPT04_RUN = ROOT / "outputs/r7_s3_rollout_dose_parent_migrated_20261015_attempt04"
ATTEMPT04_RECEIPT = ATTEMPT04_RUN / "migration_receipt.json"
ENDPOINT = 1600
MODEL_CODE_SHA256 = "60de3b2dc5de2871f45c7a563153ef8e1cfa688be38367c89d0c718dfc06df3e"
MIGRATION_RECEIPT_SHA256 = "73c019ddb5d1c6d2bc551bcd60cd321ca29d968dc956455b17e0262074c21112"
MIGRATED_PARENT_SHA256 = {
    41: "fb1e4d6522d32db1cc078a7590dd66c41f479da49ff58f2c6c525f8dfceca387",
    42: "d97ea802a24e62d1457854aeb12cf29b83f029fdca4f9903f19117f3585f3d69",
    43: "8aa0fac68609e0502b4c5292d0b6067e59153f74090055e28d08d2f3ac964af5",
}
MIGRATED_PARENT_STATE_DIGEST = {
    41: "7c416ccb4054cf7e077788a43b02917e6a2dcf2503058152a7f0ab4b0ab70707",
    42: "857d880466f76500ed74c534e757732ae9eff4c2b3f0bbc0fa74b895fcae3623",
    43: "61fdac58b7bfacf828e1eae4968817b1cbd46616442c6d16ac0fbbb63e294e1b",
}
#: The attempt04 pins, copied verbatim from decision 0050's frozen block.
ATTEMPT04_PARENT_SHA256 = {
    41: "f9f12e063c30cb0b471def01b67d7333d5aae95ecfba1ff0feb824a2a6f9ac45",
    42: "8d4e05ed5baa285113db353eede34e883812092085b8f86716fbee93fb13770f",
    43: "5af64655402cb16f0148b692ef7a84eb8934230cf43ed6ac9306501a7756cafa",
}


def _receipt(path):
    return json.loads(path.read_text(encoding="utf-8"))


def state_equality_evidence():
    """Prove the current-revision migration carries the attempt04 parent state.

    Two independent identities must hold per seed: the migration source checkpoint (the
    v3-BD 1600 endpoint) and the tensor ``state_digest`` of the migrated file. The attempt04
    receipt is also checked against the pins decision 0050 froze, so a re-run migration
    cannot pass by re-deriving the same digests from different sources.
    """
    from training.r7_arm_harness import sha256_file
    current, frozen = _receipt(MIGRATION_RECEIPT), _receipt(ATTEMPT04_RECEIPT)
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
        if then["migrated_sha256"] != ATTEMPT04_PARENT_SHA256[seed]:
            raise RuntimeError(f"seed {seed}: attempt04 drifted from decision 0050's frozen pins")
        evidence[str(seed)] = {
            "source_checkpoint_sha256": now["source_checkpoint_sha256"],
            "state_digest": now["state_digest"],
            "current_migrated_sha256": now["migrated_sha256"],
            "attempt04_migrated_sha256": then["migrated_sha256"],
        }
    return {
        "current_receipt": str(MIGRATION_RECEIPT), "current_receipt_sha256": observed_receipt,
        "attempt04_receipt": str(ATTEMPT04_RECEIPT),
        "attempt04_receipt_sha256": sha256_file(ATTEMPT04_RECEIPT),
        "equal_state_per_seed": evidence,
        "claim": ("the current-revision migration carries the same v3-BD 1600 parent state that "
                  "decision 0050 froze; only the embedded code identity differs, so the migrated "
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
        "rebinding_note": ("decision 0052: the current revision adds only the inert "
                           "tendency_input/decoder_base_mode switches, so this round's factor "
                           "stays the training-manifest month coverage"),
        "region_note": ("the parent was trained on the narrow 65x65 instance; the model spec is "
                        "region-agnostic, so the same state is valid on the wide 129x129 store")}


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
