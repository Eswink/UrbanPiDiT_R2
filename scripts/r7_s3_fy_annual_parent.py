"""The FY annual-coverage round's migrated parent, pinned without a model change.

This round changes **no** ``model/**/*.py`` file, so the model revision that produced the
H-D migration (``outputs/r7_s3_rollout_dose_parent_migrated_20261015_attempt04``) is still
the current one. The three seed checkpoints are reused verbatim; this module re-exports
the pins rather than copying their SHA256 values, so the two rounds cannot drift.

``training.r7_experiment.load_checkpoint`` still enforces the model digest by design; a
checkout whose ``model/`` differs from ``3fe747b2...`` cannot load these parents, and that
is a precondition failure for this round, not something to work around.
"""
from __future__ import annotations

from scripts import r7_s3_anchor_isolation_parent as _migrated

MIGRATED_PARENT_RUN = _migrated.MIGRATED_PARENT_RUN
MIGRATION_RECEIPT = _migrated.MIGRATION_RECEIPT
ENDPOINT = _migrated.ENDPOINT
MIGRATED_PARENT_SHA256 = _migrated.MIGRATED_PARENT_SHA256
MODEL_CODE_SHA256 = "3fe747b28e7d0ef53febc07739f210f103514a73fd19bfba6643670db65ffeb1"


def pinned_parent(seed):
    """Resolve and SHA-verify one seed's migrated parent under the current revision."""
    return _migrated.pinned_parent(seed)


def migration_provenance():
    """The migration receipt block, checked against ``model_code_digest()``."""
    return _migrated.migration_provenance()


def pins():
    """The frozen three-seed checkpoint map, without touching the filesystem."""
    return {str(seed): {"path": str(MIGRATED_PARENT_RUN / f"seed{seed}"
                                    / f"parent_update_{ENDPOINT:07d}.pt"),
                        "sha256": MIGRATED_PARENT_SHA256[seed]}
            for seed in sorted(MIGRATED_PARENT_SHA256)}


def install_parent_pins(body):
    """Install this revision's parent pins and provenance into a protocol body."""
    _migrated.install_parent_pins(body)
