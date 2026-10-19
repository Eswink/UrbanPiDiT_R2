"""The s3-decoder-base-anchor arm's migrated parent, its qualified anchor and the base controls.

``scripts/study_r7_s3_decoder_base_anchor.py`` changes ``model/`` (it adds the
``decoder_base_mode`` switch), so ``model_code_digest()`` moves and the parent migrated for
the previous revision can no longer be loaded: ``training.r7_experiment.load_checkpoint``
refuses a foreign implementation by design. This module pins the migration that the
committed ``scripts/migrate_r7_parent_state.py`` produced under the current revision; it
does not weaken that guard and it never reuses the H-D attempt04 checkpoints (their
contracts carry the old implementation digest).

The two tensors this round adds are not drawn randomly. They are the published train-only
climatology table of ``scripts/r7_s3_anchor_table_build.py``; both arms install the same
verified bytes, and the table is evaluated but its value reaches no output
(``anchor_output_prior=False`` and ``anomaly_feedback=False``), which the base-switch
control re-asserts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIGRATED_PARENT_RUN = ROOT / "outputs/r7_s3_decoder_base_anchor_parent_migrated_20261020_attempt02"
MIGRATION_RECEIPT = MIGRATED_PARENT_RUN / "migration_receipt.json"
ANCHOR_TABLE_RUN = ROOT / "outputs/r7_s3_anchor_table_20261015_attempt02"
ENDPOINT = 1600
EQUIVALENCE_SEED = 1234
#: The digest the committed migration tool recorded for the current model revision.
CURRENT_MODEL_CODE_SHA256 = "bec3859fc1386b14777300ad3e387546829c8a0fbf52e817e1cb90324b988ac1"
#: SHA256 of ``migration_receipt.json`` itself; re-running into the same directory is refused.
MIGRATION_RECEIPT_SHA256 = "002f08439b99fc3658a422e0732183f7a48c5b43d5790c133c53c4eaf004d6c3"
MIGRATED_PARENT_SHA256 = {
    41: "e94b1dd006ee970082c4792599d658617ad924c24241d44a048b5164a5d491a2",
    42: "4a9556d32c1e92fce623259380fbfa9cd9d82e33587833d62ae0b31e3702c613",
    43: "dba3048a6bf0ae48a61caf26036baa037d89fe55bb0178b88760b5be1789bec9",
}
#: Model-state digests, invariant across model revisions because only the contract moves.
MIGRATED_PARENT_STATE_DIGEST = {
    41: "7c416ccb4054cf7e077788a43b02917e6a2dcf2503058152a7f0ab4b0ab70707",
    42: "857d880466f76500ed74c534e757732ae9eff4c2b3f0bbc0fa74b895fcae3623",
    43: "61fdac58b7bfacf828e1eae4968817b1cbd46616442c6d16ac0fbbb63e294e1b",
}
#: The H-D control receipts' expansion digests; the new revision must reproduce them.
EXPANDED_STATE_SHA256 = {
    41: "00ada5e25c4e7ef16acfb24768408f06e5e6ab091d4a91bdc7505c99229de612",
    42: "baed2ff9ca898af7c6590f5ba269411d6bd660bd9601b5bcee7ae2c2a2b660a1",
    43: "4dda647f86ffee6befcab749e04153e7dbf09c6034e62d03a8f7d00bf066cce7",
}
#: The installed climatology bytes are identical for every seed and both arms.
ANCHOR_INIT_SHA256 = "de0e76beb548881b282fbd8c6f194055286ec7d0dd1d7ba88dad476fc2aa1112"
ANCHOR_KEYS = ("backbone.climatology_anchor.table", "backbone.climatology_anchor.ready")


def _receipt():
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import model_code_digest
    if not MIGRATION_RECEIPT.is_file():
        raise FileNotFoundError(f"the current-revision parent migration is missing: {MIGRATION_RECEIPT}")
    if sha256_file(MIGRATION_RECEIPT) != MIGRATION_RECEIPT_SHA256:
        raise RuntimeError("the migration receipt drifted from its pinned SHA256")
    receipt = json.loads(MIGRATION_RECEIPT.read_text(encoding="utf-8"))
    if receipt.get("current_model_code_sha256") != model_code_digest():
        raise RuntimeError("the pinned migration was produced under a different model revision; "
                           "re-run scripts/migrate_r7_parent_state.py")
    if receipt.get("current_model_code_sha256") != CURRENT_MODEL_CODE_SHA256:
        raise RuntimeError("the pinned migration did not run under the predeclared revision")
    return receipt


def pinned_parent(seed):
    """Resolve and SHA-verify the migrated model-only parent checkpoint for one seed."""
    from training.r7_arm_harness import sha256_file
    if seed not in MIGRATED_PARENT_SHA256:
        raise ValueError(f"declared seeds are {sorted(MIGRATED_PARENT_SHA256)}; {seed} is not among them")
    receipt = _receipt()
    record = receipt["seeds"][str(seed)]
    path = MIGRATED_PARENT_RUN / f"seed{seed}" / f"parent_update_{ENDPOINT:07d}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"migrated parent checkpoint missing: {path}")
    observed = sha256_file(path)
    if observed != MIGRATED_PARENT_SHA256[seed] or record["migrated_sha256"] != observed:
        raise RuntimeError(f"migrated parent checkpoint drifted from its pinned SHA256: {path}")
    if record["state_digest"] != MIGRATED_PARENT_STATE_DIGEST[seed]:
        raise RuntimeError(f"seed {seed} migrated state digest differs from the predeclared export")
    if record["migrated_path"] != str(path):
        raise RuntimeError(f"the migration receipt names a different path for seed {seed}")
    return path, observed


def migration_provenance():
    """The provenance block for the receipt that produced this revision's migrated parents."""
    from training.r7_arm_harness import sha256_file
    receipt = _receipt()
    for seed in MIGRATED_PARENT_SHA256:
        pinned_parent(seed)
    return {
        "receipt_path": str(MIGRATION_RECEIPT), "receipt_sha256": sha256_file(MIGRATION_RECEIPT),
        "export_sha256": receipt["export_sha256"],
        "archived_code_commit": receipt["archived_code_commit"],
        "archived_model_code_sha256": receipt["archived_model_code_sha256"],
        "current_model_code_sha256": receipt["current_model_code_sha256"],
        "state_digests": {str(seed): MIGRATED_PARENT_STATE_DIGEST[seed]
                          for seed in MIGRATED_PARENT_STATE_DIGEST},
        "expanded_state_digests": {str(seed): EXPANDED_STATE_SHA256[seed]
                                   for seed in EXPANDED_STATE_SHA256},
        "anchor_init_sha256": ANCHOR_INIT_SHA256,
        "operation": "archived-revision export, then strict current-code install; no optimizer/cursor/RNG",
        "not_reused": ("the H-D attempt04 parent checkpoints carry the previous implementation digest "
                       "and are not used by this round"),
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
    """Load the published train-only climatology table through the H-D verifier.

    The table, its spec and the published marker are one artifact; re-deriving that check
    here would create a second definition that could drift from the H-D builder.
    """
    from scripts import r7_s3_anchor_isolation_parent as hd_parent
    return hd_parent.anchor_payload()


def expand_parent_state(parent_state, spec, payload, *, seed):
    """Install the verified table into the current-revision parent, adding exactly two buffers.

    Returns ``(model, state, receipt)``. The registered parent carries no
    ``backbone.climatology_anchor`` key at all, so both buffers are new; every parent tensor
    is installed bit for bit. The table is the published artifact, not a fresh draw.
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
    control = base_switch_control(spec)
    if not all(control["checks"].values()):
        raise RuntimeError(f"the decoder-base positive control failed: {control['checks']}")
    anchor = {key: state[key] for key in ANCHOR_KEYS}
    expanded_digest, anchor_digest = _state_digest(state), _state_digest(anchor)
    if expanded_digest != EXPANDED_STATE_SHA256[seed]:
        raise RuntimeError(f"seed {seed}: expanded state digest differs from the H-D control receipt")
    if anchor_digest != ANCHOR_INIT_SHA256:
        raise RuntimeError(f"seed {seed}: installed climatology bytes differ from the H-D control")
    receipt = {"parent_keys": len(parent_state), "new_keys": sorted(missing),
               "init_seed": int(seed), "decoder_base_mode": spec.get("decoder_base_mode", "persistence"),
               "anchor_output_prior": bool(spec.get("anchor_output_prior")),
               "anomaly_feedback": bool(spec.get("anomaly_feedback")),
               "parent_state_sha256": _state_digest(parent_state),
               "expanded_state_sha256": expanded_digest,
               "anchor_init_sha256": anchor_digest,
               "anchor_table_sha256": payload["pins"]["table_sha256"],
               "positive_control": control}
    return target, state, receipt


def _tiny_spec(**extra):
    """The small synthetic spec the control probes use; never a weather instance."""
    spec = dict(architecture="window", in_channels=17, out_channels=17, history_steps=2, dim=32,
                depth=2, heads=2, window_size=2, patch_size=2, dropout=0., spacetime_inputs=True,
                default_reasoning_steps=2, detach_between_steps=False,
                anchored_processes=4, free_processes=1, use_forecast_feedback=True,
                positional_process_readout=True, local_solver_state=True,
                solver_state_recurrence=True, solver_gate_proposal=True, draft_query_feedback=True)
    spec.update(extra)
    return spec


def _probe_batch(channels=17, height=4, width=5):
    import torch
    generator = torch.Generator().manual_seed(11)
    return {"coarse_history": torch.randn(1, 2, channels, height, width, generator=generator),
            "latitude": torch.linspace(50., 44., height),
            "longitude": torch.linspace(90., 98., width),
            "lead_time_hours": torch.tensor([6.]),
            "init_utc_hour": torch.tensor([18.]),
            "init_day_of_year": torch.tensor([366.]),
            "init_calendar_year": torch.tensor([2016.]),
            "history_offsets_hours": torch.tensor([[-6., 0.]])}


def base_switch_control(spec):
    """The raw counter-proofs that make "decoder base semantics" a single factor.

    Four measured facts on a synthetic tiny instance, none of them a claim about trained
    behaviour. (1) Default and explicit ``persistence`` are bitwise identical, and the
    persistence base is the last history frame. (2) With ``zero`` the returned
    ``base_state`` is zero and the forecast equals the tendency. (3) Persistence minus
    zero equals the persistence base (up to float32 addition rounding, which the CI test
    pins exactly on representable operands). (4) Zeroing the installed climatology table
    changes neither arm: the forward still evaluates the table, but its value does not
    reach any output. The two modes are one tensor set.
    """
    import torch

    from training.r7_experiment import make_model
    from scripts.r7_s3_anchor_isolation_parent import _tiny_anchor

    anchor_spec, table = _tiny_anchor()
    zero_spec = dict(anchor_spec, table_sha256=table_digest(torch.zeros_like(table)))

    def build(mode, *, anchor_spec_used=None, table_bytes=None, **extra):
        if anchor_spec_used is not None:
            extra["climatology_anchor_spec"] = anchor_spec_used
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(EQUIVALENCE_SEED)
            model = make_model("process", _tiny_spec(decoder_base_mode=mode, **extra))
        if anchor_spec_used is not None:
            model.backbone.climatology_anchor.install(
                table if table_bytes is None else table_bytes)
        return model.eval()

    batch = _probe_batch()
    default = build("persistence")
    explicit = build("persistence")
    zero = build("zero")
    anchored = dict(climatology_anchor_spec=anchor_spec, anchor_output_prior=False,
                    anomaly_feedback=False)
    persistence_anchored = build("persistence", anchor_spec_used=anchor_spec, **anchored)
    zero_anchored = build("zero", anchor_spec_used=anchor_spec, **anchored)
    zero_table_anchored = build("zero", anchor_spec_used=zero_spec,
                                table_bytes=torch.zeros_like(table), **anchored)
    persistence_zero_table = build("persistence", anchor_spec_used=zero_spec,
                                   table_bytes=torch.zeros_like(table), **anchored)
    with torch.no_grad():
        d, e = default(batch), explicit(batch)
        persisted = default.backbone(batch)
        zero_backbone = zero.backbone(batch)
        pa = persistence_anchored.backbone(batch)
        za = zero_anchored.backbone(batch)
        zt = zero_table_anchored.backbone(batch)
        pt = persistence_zero_table.backbone(batch)
        z_initial = zero(batch).initial_forecast
    base = batch["coarse_history"][:, -1]
    state_keys = set(default.state_dict()) == set(zero.state_dict()) == set(explicit.state_dict())
    state_equal = all(torch.equal(default.state_dict()[key], zero.state_dict()[key])
                      for key in default.state_dict())
    shapes_equal = all(default.state_dict()[key].shape == zero.state_dict()[key].shape
                       for key in default.state_dict())
    checks = {
        "default_equals_explicit_persistence": (torch.equal(d.forecast, e.forecast)
                                                and torch.equal(d.draft_forecasts, e.draft_forecasts)),
        "persistence_base_is_last_history": torch.equal(persisted.base_state, base),
        "zero_base_is_zero": bool((zero_backbone.base_state == 0).all()),
        "zero_forecast_is_tendency": (torch.equal(zero_backbone.forecast, zero_backbone.tendency)
                                      and torch.equal(z_initial, zero_backbone.tendency)),
        "persistence_minus_zero_is_base": torch.allclose(
            pa.forecast - za.forecast, persisted.base_state, rtol=0., atol=1e-5),
        "states_share_keys_shapes_values": state_keys and shapes_equal and state_equal,
        "lazy_anchor_value_unused_zero": torch.equal(za.forecast, zt.forecast),
        "lazy_anchor_value_unused_persistence": torch.equal(pa.forecast, pt.forecast),
    }
    return {"synthetic": True, "spec": _tiny_spec(), "checks": checks,
            "parameter_count": sum(parameter.numel() for parameter in default.parameters()),
            "anchor_table_sha256": table_digest(table),
            "zero_table_sha256": table_digest(torch.zeros_like(table)),
            "note": ("synthetic tensors only; persistence-minus-zero uses float32 addition "
                     "rounding, and the CI test pins the exact identity on representable operands")}
