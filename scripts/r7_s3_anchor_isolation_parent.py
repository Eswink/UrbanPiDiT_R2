"""The H-D arm's migrated parent, its qualified climatology anchor, and the expansion.

``scripts/study_r7_s3_anchor_isolation.py`` changes ``model/`` (it adds the ``anchor_output_prior``
switch), so ``model_code_digest()`` moves and the parent migrated for the previous revision can no
longer be loaded: ``training.r7_experiment.load_checkpoint`` refuses a foreign implementation by
design. This module pins the migration that the committed ``scripts/migrate_r7_parent_state.py``
produced under the current revision; it does not weaken that guard.

Unlike the H-A/H-B/H-C expansions, the two tensors this round adds are **not** drawn randomly. They
are the train-only climatology table derived by ``scripts/r7_s3_anchor_table_build.py`` and published
with its own ``BUILD_COMPLETE`` marker, so both arms install the *same* verified bytes: the
shared-initialization requirement is satisfied by a content identity, which is stronger than a seed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIGRATED_PARENT_RUN = ROOT / "outputs/r7_s3_rollout_dose_parent_migrated_20261015_attempt04"
MIGRATION_RECEIPT = MIGRATED_PARENT_RUN / "migration_receipt.json"
ANCHOR_TABLE_RUN = ROOT / "outputs/r7_s3_anchor_table_20261015_attempt02"
ENDPOINT = 1600
EQUIVALENCE_SEED = 1234
MIGRATED_PARENT_SHA256 = {
    41: "f9f12e063c30cb0b471def01b67d7333d5aae95ecfba1ff0feb824a2a6f9ac45",
    42: "8d4e05ed5baa285113db353eede34e883812092085b8f86716fbee93fb13770f",
    43: "5af64655402cb16f0148b692ef7a84eb8934230cf43ed6ac9306501a7756cafa",
}
ANCHOR_KEYS = ("backbone.climatology_anchor.table", "backbone.climatology_anchor.ready")
#: The registered registered arm's two switches; the arms differ only in ``anomaly_feedback``.
COMMON_SWITCHES = {"anchor_output_prior": False}


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


def migration_provenance():
    """The provenance block for the receipt that produced this revision's migrated parents.

    ``interior._identity()`` inherits the dose arm's ``model_only_migration`` block, which names a
    different receipt and digest. Keeping it while the checkpoints come from this revision would be
    a provenance contradiction, so the block is rebuilt from the receipt that produced these files.
    """
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import model_code_digest
    receipt = json.loads(MIGRATION_RECEIPT.read_text(encoding="utf-8"))
    current = model_code_digest()
    if receipt.get("current_model_code_sha256") != current:
        raise RuntimeError("the pinned migration was produced under a different model revision; "
                           "re-run scripts/migrate_r7_parent_state.py")
    for seed in MIGRATED_PARENT_SHA256:
        path = MIGRATED_PARENT_RUN / f"seed{seed}" / f"parent_update_{ENDPOINT:07d}.pt"
        record = receipt["seeds"][str(seed)]
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
        "forward_equivalence": ("the control arm's forward equals the unanchored current model bit "
                                "for bit on a fixed synthetic batch; the migration installs the "
                                "registered parent state unchanged"),
        "region_note": ("the parent was trained on the narrow 65x65 instance; the model spec is "
                        "region-agnostic, so the same state is valid on the wide 129x129 store")}


def install_parent_pins(body):
    """Replace the inherited parent pins and provenance with this revision's migration."""
    body["arms"]["parent"]["pins"]["checkpoints"] = {
        str(seed): {"path": str(MIGRATED_PARENT_RUN / f"seed{seed}" / f"parent_update_{ENDPOINT:07d}.pt"),
                    "sha256": MIGRATED_PARENT_SHA256[seed]}
        for seed in MIGRATED_PARENT_SHA256}
    body["arms"]["parent"]["pins"]["model_only_migration"] = migration_provenance()


def table_digest(table):
    """Exactly the digest ``TrainClimatologyAnchor._table_digest`` computes."""
    return hashlib.sha256(table.detach().cpu().contiguous().numpy().tobytes(order="C")).hexdigest()


def anchor_payload():
    """Load and verify the published train-only climatology table.

    The marker, the spec's own table digest and the bytes read here must agree; a mismatch means the
    artifact is not the one the round froze, so it fails loudly instead of training against it.
    """
    import torch
    from training.r7_arm_harness import sha256_file
    marker = json.loads((ANCHOR_TABLE_RUN / "BUILD_COMPLETE.json").read_text(encoding="utf-8"))
    if marker.get("build_complete") is not True or marker.get("schema_version") != 1:
        raise RuntimeError("climatology anchor table is not published complete")
    spec = json.loads((ANCHOR_TABLE_RUN / "anchor_spec.json").read_text(encoding="utf-8"))
    table = torch.load(ANCHOR_TABLE_RUN / "anchor_table.pt", map_location="cpu", weights_only=True)
    observed = table_digest(table)
    if observed != spec["table_sha256"] or observed != marker["table_sha256"]:
        raise RuntimeError("climatology table bytes differ from the published spec/marker digest")
    if list(table.shape) != spec["table_shape"]:
        raise RuntimeError("climatology table shape differs from its spec")
    receipt = json.loads((ANCHOR_TABLE_RUN / "build_receipt.json").read_text(encoding="utf-8"))
    canonical = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if receipt["spec_sha256"] != canonical or marker["spec_sha256"] != canonical:
        raise RuntimeError("climatology build receipt/marker disagrees with the published spec")
    if receipt.get("table_sha256") != observed:
        raise RuntimeError("climatology build receipt's table digest differs from the published bytes")
    payload = {"spec": spec, "table": table,
               "pins": {"run": str(ANCHOR_TABLE_RUN), "table_sha256": observed,
                        "spec_sha256": marker["spec_sha256"],
                        "table_file_sha256": sha256_file(ANCHOR_TABLE_RUN / "anchor_table.pt"),
                        "n_buckets": len(spec["bucket_keys"]),
                        "n_selected_steps": spec["n_selected_steps"],
                        "bucket_counts": list(spec["bucket_counts"]),
                        "source_sha256": spec["source_sha256"],
                        "train_data_identity": spec["train_data_identity"],
                        "build_receipt_sha256": sha256_file(ANCHOR_TABLE_RUN / "build_receipt.json")}}
    return payload


def expand_parent_state(parent_state, spec, payload, *, seed):
    """Load a parent without an anchor into the H-D trunk, which installs the qualified table.

    Returns ``(model, state, receipt)``. The registered parent carries no
    ``backbone.climatology_anchor`` key at all, so both buffers are new; the parent's own tensors are
    installed bit for bit. The table is the verified published artifact rather than a fresh draw, so
    the two arms of one seed start from identical climatology bytes by content identity.
    """
    import torch

    from training.r7_experiment import make_model
    from training.r7_autoregressive_runner import _state_digest
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        target = make_model("process", spec)
    incoming = target.state_dict()
    missing = [key for key in incoming if key not in parent_state]
    if set(missing) != set(ANCHOR_KEYS):
        raise RuntimeError(f"the climatology anchor must add exactly {list(ANCHOR_KEYS)}, "
                           f"got {sorted(missing)}")
    extra = [key for key in parent_state if key not in incoming]
    if extra:
        raise RuntimeError(f"the parent carries keys the target does not: {sorted(extra)}")
    target.backbone.climatology_anchor.install(payload["table"])
    installed = target.state_dict()
    state = dict(parent_state)
    for key in missing:
        state[key] = installed[key].detach().clone()
    target.load_state_dict(state, strict=True)
    control = positive_control(_tiny_spec())
    if not (control["state_identity"]["bit_identical_outside_anchor"]
            and control["unanchored_equals_off"]
            and control["feedback_is_read"]
            and control["zeroed_table_is_identity"]):
        raise RuntimeError("H-D positive control failed: the anchor leaks into the unanchored path, "
                           "the control arm is not the registered model, anomaly feedback is not "
                           "read, or a zeroed table does not collapse the candidate onto the control")
    anchor = {key: state[key] for key in ANCHOR_KEYS}
    receipt = {"parent_keys": len(parent_state), "new_keys": sorted(missing),
               "init_seed": int(seed), "anomaly_feedback": bool(spec.get("anomaly_feedback")),
               "anchor_output_prior": bool(spec.get("anchor_output_prior")),
               "parent_state_sha256": _state_digest(parent_state),
               "expanded_state_sha256": _state_digest(state),
               # A content digest, not a random draw: both arms must produce the same value.
               "anchor_init_sha256": _state_digest(anchor),
               "anchor_table_sha256": payload["pins"]["table_sha256"],
               "positive_control": control}
    return target, state, receipt


def _tiny_spec():
    """The small synthetic spec the control probes use; never a weather instance.

    The solver-path switches mirror the archived arm's values, not the constructor defaults: without
    ``local_solver_state`` the probe never reaches ``process_step_r7``'s recurrence branch, which is
    exactly where the real arm decodes its proposal (audit finding, 2026-10-17).
    """
    return dict(architecture="window", in_channels=17, out_channels=17, history_steps=2, dim=32,
                depth=2, heads=2, window_size=2, patch_size=2, dropout=0., spacetime_inputs=True,
                default_reasoning_steps=4, detach_between_steps=False,
                anchored_processes=4, free_processes=1, use_forecast_feedback=True,
                positional_process_readout=True, local_solver_state=True,
                solver_state_recurrence=True, solver_gate_proposal=True, draft_query_feedback=True)


def _tiny_anchor():
    """A small synthetic anchor spec/table for the control probes; never weather statistics."""
    import torch
    channels, height, width = 17, 4, 5
    keys = [[month, hour] for month in (1, 4, 7, 10) for hour in (0, 6, 12, 18)]
    table = (torch.randn(len(keys), channels, height, width) * 0.1).contiguous()
    spec = {"format": "r7-train-climatology-anchor-v1",
            "kind": "train-only-month-hour-grid-mean-v1",
            "channels": [f"c{index}" for index in range(channels)],
            "units": ["u"] * channels, "source_sha256": "1" * 64,
            "train_data_identity": "2" * 64, "climatology_mean_identity_sha256": "3" * 64,
            "normalization_mean": [0.0] * channels, "normalization_std": [1.0] * channels,
            "latitude": [float(50 - 2 * index) for index in range(height)],
            "longitude": [float(90 + 2 * index) for index in range(width)],
            "bucket_keys": keys, "bucket_counts": [2] * len(keys),
            "training_years": [2017, 2018], "n_selected_steps": 2 * len(keys),
            "selection": "declared_train_years", "table_sha256": table_digest(table),
            "table_shape": list(table.shape)}
    return spec, table


def _probe_batch(anchor_spec):
    import torch
    channels = len(anchor_spec["channels"])
    height, width = len(anchor_spec["latitude"]), len(anchor_spec["longitude"])
    generator = torch.Generator().manual_seed(11)
    return {"coarse_history": torch.randn(1, 2, channels, height, width, generator=generator),
            "atmos_target": torch.randn(1, channels, height, width, generator=generator),
            "future_target": torch.randn(1, channels, height, width, generator=generator),
            "latitude": torch.tensor(anchor_spec["latitude"]),
            "longitude": torch.tensor(anchor_spec["longitude"]),
            "lead_time_hours": torch.tensor([6.]), "init_utc_hour": torch.tensor([18.]),
            "init_day_of_year": torch.tensor([366.]), "init_calendar_year": torch.tensor([2016.]),
            "history_offsets_hours": torch.tensor([[-6., 0.]]),
            "init_year": torch.tensor([1234.]), "process_targets": torch.tensor([[9999.]]),
            "untrusted_metadata": "not a model input"}


def positive_control(spec):
    """The raw counter-proofs that make "feedback representation" a single factor.

    Four measured facts on a synthetic tiny instance, none of them a claim about trained behaviour.
    (1) The control arm's state is bit-identical to an unanchored model except for the two anchor
    buffers, so installing the anchor moves no weight. (2) With ``anchor_output_prior`` off and
    ``anomaly_feedback`` off the anchor is unread, so the control arm's forward equals the
    unanchored forward bit for bit. (3) Turning ``anomaly_feedback`` on changes the forward, so the
    mechanism is actually read. (4) Zeroing the table makes ``draft - 0 = draft``, so the candidate
    collapses onto the control bit for bit -- the identity that is asserted per seed before training.
    """
    import torch

    from training.r7_experiment import make_model
    anchor_spec, table = _tiny_anchor()
    zero = torch.zeros_like(table)
    zero_spec = dict(anchor_spec, table_sha256=table_digest(zero))

    def build(**extra):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(EQUIVALENCE_SEED)
            return make_model("process", dict(spec, **extra))

    def anchored(anchor, anchor_spec_used, **extra):
        model = build(climatology_anchor_spec=anchor_spec_used, **extra)
        model.backbone.climatology_anchor.install(anchor)
        return model.eval()

    off = build().eval()
    control = anchored(table, anchor_spec, anchor_output_prior=False, anomaly_feedback=False)
    candidate = anchored(table, anchor_spec, anchor_output_prior=False, anomaly_feedback=True)
    zero_control = anchored(zero, zero_spec, anchor_output_prior=False, anomaly_feedback=False)
    zero_candidate = anchored(zero, zero_spec, anchor_output_prior=False, anomaly_feedback=True)
    prior_on = anchored(table, anchor_spec, anchor_output_prior=True, anomaly_feedback=False)

    off_state, control_state = off.state_dict(), control.state_dict()
    missing = sorted(set(control_state) - set(off_state))
    shared = [key for key in off_state if key in control_state]
    state_identity = {
        "added_keys": missing, "lost_keys": sorted(set(off_state) - set(control_state)),
        "n_shared_compared": len(shared), "new_keys_match_anchor_keys": missing == sorted(ANCHOR_KEYS),
        "bit_identical_outside_anchor": (missing == sorted(ANCHOR_KEYS)
                                        and all(torch.equal(off_state[key], control_state[key])
                                                for key in shared)),
    }
    batch = _probe_batch(anchor_spec)
    with torch.no_grad():
        reference = off(batch, reasoning_steps=4).forecast
        control_out = control(batch, reasoning_steps=4).forecast
        candidate_out = candidate(batch, reasoning_steps=4).forecast
        zero_control_out = zero_control(batch, reasoning_steps=4).forecast
        zero_candidate_out = zero_candidate(batch, reasoning_steps=4).forecast
        prior_out = prior_on(batch, reasoning_steps=4).forecast
    return {
        "seed": EQUIVALENCE_SEED, "state_identity": state_identity,
        "unanchored_max_abs": float((token_gap(reference, control_out))),
        "unanchored_equals_off": bool(torch.equal(reference, control_out)),
        "feedback_delta_max_abs": float(token_gap(control_out, candidate_out)),
        "feedback_is_read": not bool(torch.equal(control_out, candidate_out)),
        "zeroed_control_equals_off": bool(torch.equal(reference, zero_control_out)),
        "zeroed_table_delta_max_abs": float(token_gap(zero_control_out, zero_candidate_out)),
        "zeroed_table_is_identity": bool(torch.equal(zero_control_out, zero_candidate_out)),
        "output_prior_delta_max_abs": float(token_gap(reference, prior_out)),
        "anchor_table_sha256": table_digest(table),
    }


def token_gap(left, right):
    """The max absolute difference, named so the control's numbers read as what they are."""
    return (left - right).abs().max()
