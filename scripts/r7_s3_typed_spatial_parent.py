"""The H-B arm's migrated parent and its typed-evidence expansion.

``scripts/study_r7_s3_typed_spatial.py`` is the second S3 arm that changes ``model/`` (it adds the
``typed_spatial`` routing mode to ``model/typed_evidence_r7.py``), so ``model_code_digest()`` moves
and the parent migrated for the previous revision can no longer be loaded:
``training.r7_experiment.load_checkpoint`` refuses a foreign implementation by design. This module
pins the migration that the committed ``scripts/migrate_r7_parent_state.py`` produced under the
current revision; it does not weaken that guard. The weights are unchanged from the registered
parent -- only the recorded digest and the provenance differ -- and the receipt's revision is
re-checked here so a stale pin fails loudly instead of loading a stale parent.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIGRATED_PARENT_RUN = ROOT / "outputs/r7_s3_rollout_dose_parent_migrated_20261014_attempt02"
MIGRATION_RECEIPT = MIGRATED_PARENT_RUN / "migration_receipt.json"
ENDPOINT = 1600
EQUIVALENCE_SEED = 1234
#: The one key the spatial arm adds; excluded from the shared-initialization digest.
SPATIAL_ONLY_KEY = "typed_evidence.position_query"
MIGRATED_PARENT_SHA256 = {
    41: "a376fdac3dd3b473f26ef5164f85076e9fe4ab66399757f9ca90dabb19dd94d7",
    42: "5f395bdef5509534437a8deecd00781facffe1511947bbce3280b9c5d932e99e",
    43: "4fa69a8a74f3263eec77209c833cc273547caaa2258ab72e7d3d07f1501fc190",
}


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


def install_parent_pins(body):
    """Replace the inherited migrated-parent checkpoint pins with this revision's migration.

    ``interior._identity()`` inherits the pins recorded for the earlier revision; the frozen protocol
    must pin the checkpoint it can actually load, so the driver calls this before freezing.
    """
    body["arms"]["parent"]["pins"]["checkpoints"] = {
        str(seed): {"path": str(MIGRATED_PARENT_RUN / f"seed{seed}" / f"parent_update_{ENDPOINT:07d}.pt"),
                    "sha256": MIGRATED_PARENT_SHA256[seed]}
        for seed in MIGRATED_PARENT_SHA256}


def typed_evidence_keys(mode):
    """The state-dict keys the typed-evidence pathway adds, for one routing mode."""
    from model.typed_evidence_r7 import (
        EVIDENCE_TYPES, GENERIC_FUSION, TYPED_SPATIAL,
    )
    keys = ["typed_evidence.conv.weight", "typed_evidence.conv.bias",
            "typed_evidence.norm.weight", "typed_evidence.norm.bias"]
    if mode == GENERIC_FUSION:
        keys.append("typed_evidence.fusion.weight")
    else:
        keys += [f"typed_evidence.projections.{index}.weight" for index in range(EVIDENCE_TYPES)]
        if mode == TYPED_SPATIAL:
            keys.append("typed_evidence.position_query")
    return tuple(keys)


def expand_parent_state(parent_state, spec, payload, mode, *, seed):
    """Load a parent without typed evidence into the H-B trunk, which adds the pathway.

    Returns ``(model, state, receipt)``. The registered parent carries no ``typed_evidence`` key at
    all, so every pathway tensor is new; the parent's own tensors are installed bit for bit. The
    pathway is drawn under ``seed``, so the pooled and the spatial arm of one round seed start from
    the *same* pathway initialization and the B->C delta is attributable to the position aggregation
    rather than to a fresh random draw -- the confound the #79 round guarded against with
    ``verify_arm_pairing``. The receipt records the raw positive-control output, and a failed control
    aborts the run rather than being written down and ignored.
    """
    import torch

    from training.r7_experiment import make_model
    from training.r7_autoregressive_runner import _state_digest
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        target = make_model("process", spec)
    target_state = target.state_dict()
    missing = [key for key in target_state if key not in parent_state]
    expected = typed_evidence_keys(mode)
    if set(missing) != set(expected):
        raise RuntimeError(f"the typed-evidence pathway must add exactly {sorted(expected)}, "
                           f"got {sorted(missing)}")
    extra = [key for key in parent_state if key not in target_state]
    if extra:
        raise RuntimeError(f"the parent carries keys the target does not: {sorted(extra)}")
    state = dict(parent_state)
    for key in missing:
        state[key] = target_state[key].detach().clone()
    target.load_state_dict(state, strict=True)
    control = positive_control(spec, payload)
    if not (control["shared_bit_identical"] and control["zero_query_is_pooled"]
            and control["zeroed_projection_is_identity"]):
        raise RuntimeError("H-B positive control failed: the two typed arms are not identical outside "
                           "the position aggregation, or the zero-query / zeroed-projection identities "
                           "do not hold")
    # Digest over the keys BOTH typed arms carry, so the two receipts' values must be equal:
    # that equality is the checkable evidence that the arms share one pathway initialization.
    pathway = {key: state[key] for key in sorted(set(expected) - {SPATIAL_ONLY_KEY})}
    receipt = {"parent_keys": len(parent_state), "new_keys": sorted(missing),
               "init_seed": int(seed),
               "parent_state_sha256": _state_digest(parent_state),
               "expanded_state_sha256": _state_digest(state),
               "pathway_init_sha256": _state_digest(pathway),
               "new_params": sum(int(target_state[key].numel()) for key in missing),
               "positive_control": control}
    return target, state, receipt


def positive_control(spec, payload):
    """The raw counter-proofs: the two typed arms differ only in the position aggregation.

    Three measured facts, none of them a claim about trained behaviour. (1) Under one seed the
    pooled and spatial models share every tensor except the spatial query, and those shared tensors
    are bit-identical. (2) With the query zeroed the spatial aggregation is the pooled one to within
    float summation order. (3) Zeroing the per-type projections makes the evidence identically zero
    in both modes, so the pathway can be switched off exactly.
    """
    import torch

    from model.typed_evidence_r7 import (
        EVIDENCE_TYPES, TYPED_ROUTING, TYPED_SPATIAL, TypedEvidenceRouter,
    )
    from training.r7_experiment import make_model
    fields = dict(payload)
    dim, patch_size = int(spec["dim"]), int(spec.get("patch_size", 2))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(EQUIVALENCE_SEED)
        pooled = make_model("process", dict(spec, typed_evidence_mode=TYPED_ROUTING,
                                            typed_evidence=dict(fields)))
        pooled_state = pooled.state_dict()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(EQUIVALENCE_SEED)
        spatial = make_model("process", dict(spec, typed_evidence_mode=TYPED_SPATIAL,
                                             typed_evidence=dict(fields)))
        spatial_state = spatial.state_dict()
    added = sorted(set(spatial_state) - set(pooled_state))
    lost = sorted(set(pooled_state) - set(spatial_state))
    shared = [key for key in pooled_state if key in spatial_state]
    shared_bit_identical = (added == ["typed_evidence.position_query"] and not lost
                            and all(torch.equal(pooled_state[key], spatial_state[key])
                                    for key in shared))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(EQUIVALENCE_SEED)
        router_pooled = TypedEvidenceRouter(dim, patch_size, mode=TYPED_ROUTING, **fields)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(EQUIVALENCE_SEED)
        router_spatial = TypedEvidenceRouter(dim, patch_size, mode=TYPED_SPATIAL, **fields)
    fields_tensor = torch.randn(2, EVIDENCE_TYPES, 8, 8)
    with torch.no_grad():
        pooled_evidence = router_pooled.token_evidence(fields_tensor)
        zero_query = router_spatial.token_evidence(fields_tensor)
        zero_query_max_abs = float((pooled_evidence - zero_query).abs().max())
        router_spatial.position_query.normal_()
        moved_max_abs = float((pooled_evidence - router_spatial.token_evidence(fields_tensor)).abs().max())
        router_spatial.position_query.zero_()
        for projection in router_spatial.projections:
            projection.weight.zero_()
        zeroed = router_spatial.token_evidence(fields_tensor)
        zeroed_nonzero = int(torch.count_nonzero(zeroed))
        for projection in router_pooled.projections:
            projection.weight.zero_()
        zeroed_pooled_nonzero = int(torch.count_nonzero(router_pooled.token_evidence(fields_tensor)))
    return {"seed": EQUIVALENCE_SEED, "added_keys": added, "lost_keys": lost,
            "n_shared_compared": len(shared), "shared_bit_identical": bool(shared_bit_identical),
            "zero_query_max_abs": zero_query_max_abs, "zero_query_is_pooled": zero_query_max_abs <= 1e-6,
            "moved_max_abs_after_random_query": moved_max_abs,
            "zeroed_projection_nonzero_spatial": zeroed_nonzero,
            "zeroed_projection_nonzero_pooled": zeroed_pooled_nonzero,
            "zeroed_projection_is_identity": zeroed_nonzero == 0 and zeroed_pooled_nonzero == 0,
            "spatial_extra_params": int(router_spatial.position_query.numel())}
