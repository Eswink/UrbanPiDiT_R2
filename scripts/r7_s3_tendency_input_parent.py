"""The s3-tendency-input arm's migrated parent, input expansion and CI-safe control.

``model/coarse_encoder.py`` (plus the two pass-through constructors) gains the
``tendency_input`` switch, so ``model_code_digest()`` moves and the parent migrated for
the decoder/gradient-surgery revisions can no longer be loaded: ``load_checkpoint``
refuses a foreign implementation by design. This module pins the migration that the
committed ``scripts/migrate_r7_parent_state.py`` produced under the current revision;
it does not weaken that guard.

The candidate adds exactly one tensor, ``backbone.encoder.tendency_patch.weight``
(17 x dim x patch_size^2, bias=False), initialized to zeros. ``expand_parent_state``
installs every migrated parent tensor bit for bit, installs the published train-only
climatology table, adds that one zero tensor, and verifies both facts before returning.
The zero initialization is what makes the candidate's forward at update 0 identical to
the control's under the same seed; that is asserted on synthetic tensors in
``input_representation_control`` and never claimed as a trained-behaviour result.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIGRATED_PARENT_RUN = ROOT / "outputs/r7_s3_tendency_input_parent_migrated_20261021_attempt02"
MIGRATION_RECEIPT = MIGRATED_PARENT_RUN / "migration_receipt.json"
ANCHOR_TABLE_RUN = ROOT / "outputs/r7_s3_anchor_table_20261015_attempt02"
ENDPOINT = 1600
EQUIVALENCE_SEED = 1234
#: The digest the committed migration tool recorded for the current model revision.
CURRENT_MODEL_CODE_SHA256 = "60de3b2dc5de2871f45c7a563153ef8e1cfa688be38367c89d0c718dfc06df3e"
#: SHA256 of ``migration_receipt.json`` itself; re-running into the same directory is refused.
MIGRATION_RECEIPT_SHA256 = "73c019ddb5d1c6d2bc551bcd60cd321ca29d968dc956455b17e0262074c21112"
MIGRATED_PARENT_SHA256 = {
    41: "fb1e4d6522d32db1cc078a7590dd66c41f479da49ff58f2c6c525f8dfceca387",
    42: "d97ea802a24e62d1457854aeb12cf29b83f029fdca4f9903f19117f3585f3d69",
    43: "8aa0fac68609e0502b4c5292d0b6067e59153f74090055e28d08d2f3ac964af5",
}
#: Model-state digests, invariant across model revisions because only the contract moves.
MIGRATED_PARENT_STATE_DIGEST = {
    41: "7c416ccb4054cf7e077788a43b02917e6a2dcf2503058152a7f0ab4b0ab70707",
    42: "857d880466f76500ed74c534e757732ae9eff4c2b3f0bbc0fa74b895fcae3623",
    43: "61fdac58b7bfacf828e1eae4968817b1cbd46616442c6d16ac0fbbb63e294e1b",
}
#: Expanded state (parent + anchor buffers + the zero tendency patch) per seed.
EXPANDED_STATE_SHA256 = {
    41: "7105700be5f8cd8ce9c101a809f37d6f3ce605f0f48e4ffc325f91290c2154ed",
    42: "b169534960cc4bac1d6d64815e54e5711916b28ced8237f58ab7ce7a3f3a7cf4",
    43: "691058ad8a7bb96336d8e8f768404c6379b10da8c38d55c331cffd1e3f45e2a2",
}
#: The installed climatology bytes are identical for every seed and both arms.
ANCHOR_INIT_SHA256 = "de0e76beb548881b282fbd8c6f194055286ec7d0dd1d7ba88dad476fc2aa1112"
ANCHOR_KEYS = ("backbone.climatology_anchor.table", "backbone.climatology_anchor.ready")
TENDENCY_KEY = "backbone.encoder.tendency_patch.weight"
TENDENCY_CHANNELS = 17
#: 17 * dim * patch_size^2 at the frozen dim=192, patch_size=2; the declared capacity confound.
TENDENCY_PARAMETERS = 13056


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
    for seed in MIGRATED_PARENT_SHA256:
        if receipt["seeds"][str(seed)]["migrated_sha256"] != MIGRATED_PARENT_SHA256[seed]:
            raise RuntimeError(f"the migration receipt changed for seed {seed}")
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
        "not_reused": ("the decoder/gradient-surgery revision checkpoints carry the previous "
                       "implementation digest and are not used by this round"),
        "region_note": ("the parent was trained on the narrow 65x65 instance; the model spec is "
                        "region-agnostic, so the same state is valid on the wide 129x129 store")}


def install_parent_pins(body):
    """Replace the inherited parent pins and provenance with this revision's migration."""
    body["arms"]["parent"]["pins"]["checkpoints"] = {
        str(seed): {"path": str(MIGRATED_PARENT_RUN / f"seed{seed}" / f"parent_update_{ENDPOINT:07d}.pt"),
                    "sha256": MIGRATED_PARENT_SHA256[seed]}
        for seed in MIGRATED_PARENT_SHA256}
    body["arms"]["parent"]["pins"]["model_only_migration"] = migration_provenance()


def anchor_payload():
    """Load the published train-only climatology table through the H-D verifier."""
    from scripts import r7_s3_anchor_isolation_parent as hd_parent
    return hd_parent.anchor_payload()


def expand_parent_state(parent_state, spec, payload, *, seed):
    """Install the verified table and the zero tendency patch into the current-revision parent.

    Returns ``(model, state, receipt)``. The registered parent carries neither the
    climatology anchor nor the tendency patch, so exactly three keys are new; every
    parent tensor is installed bit for bit and the tendency patch stays exactly zero.
    """
    import torch

    from training.r7_experiment import make_model
    from training.r7_autoregressive_runner import _state_digest
    if spec.get("tendency_input") is not True:
        raise RuntimeError("the candidate spec must declare tendency_input=True")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        target = make_model("process", spec)
    incoming = target.state_dict()
    missing = [key for key in incoming if key not in parent_state]
    expected = set(ANCHOR_KEYS) | {TENDENCY_KEY}
    if set(missing) != expected or len(missing) != len(expected):
        raise RuntimeError(f"the tendency input must add exactly {sorted(expected)}, "
                           f"got {sorted(missing)}")
    extra = [key for key in parent_state if key not in incoming]
    if extra:
        raise RuntimeError(f"the parent carries keys the target does not: {sorted(extra)}")
    target.backbone.climatology_anchor.install(payload["table"])
    with torch.no_grad():
        target.backbone.encoder.tendency_patch.weight.zero_()
    installed = target.state_dict()
    state = dict(parent_state)
    for key in missing:
        state[key] = installed[key].detach().clone()
    target.load_state_dict(state, strict=True)
    control = input_representation_control()
    if not all(control["checks"].values()):
        raise RuntimeError(f"the tendency-input positive control failed: {control['checks']}")
    anchored = {key: state[key] for key in ANCHOR_KEYS}
    expanded_digest, anchor_digest = _state_digest(state), _state_digest(anchored)
    if expanded_digest != EXPANDED_STATE_SHA256[seed]:
        raise RuntimeError(f"seed {seed}: expanded state digest differs from the predeclared export")
    if anchor_digest != ANCHOR_INIT_SHA256:
        raise RuntimeError(f"seed {seed}: installed climatology bytes differ from the H-D control")
    if int(torch.count_nonzero(state[TENDENCY_KEY])) != 0:
        raise RuntimeError("the tendency patch must start at exactly zero")
    parameter_delta = TENDENCY_PARAMETERS
    receipt = {"parent_keys": len(parent_state), "new_keys": sorted(missing),
               "init_seed": int(seed), "tendency_input": True,
               "tendency_key": TENDENCY_KEY,
               "tendency_shape": list(state[TENDENCY_KEY].shape),
               "parameter_delta": parameter_delta,
               "parent_state_sha256": _state_digest(parent_state),
               "expanded_state_sha256": expanded_digest,
               "anchor_init_sha256": anchor_digest,
               "anchor_table_sha256": payload["pins"]["table_sha256"],
               "positive_control": control}
    return target, state, receipt


def _tiny_spec(**extra):
    """The small synthetic spec the control probes use; never a weather instance."""
    spec = dict(architecture="window", in_channels=17, out_channels=17, history_steps=2,
                dim=32, depth=2, heads=2, window_size=2, patch_size=2, dropout=0.,
                spacetime_inputs=True, default_reasoning_steps=2, detach_between_steps=False,
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


def input_representation_control():
    """The raw counter-proofs that make the explicit tendency one input factor.

    Measured on synthetic tensors only, never a claim about trained behaviour:
    (1) same seed, the two arms share every pre-existing tensor bitwise and the only
    new key is the all-zero tendency patch; (2) at initialization the forecast, every
    draft and the initial forecast are bitwise identical; (3) with a deterministic
    non-zero patch weight the branch output is exactly ``conv(X_t - X_{t-1})`` and the
    reversed sign gives a different result; (4) the patch has no bias, so the parameter
    delta is exactly 17 * dim * patch_size^2.
    """
    import torch

    from training.r7_experiment import make_model
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(EQUIVALENCE_SEED)
        control = make_model("process", _tiny_spec()).eval()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(EQUIVALENCE_SEED)
        candidate = make_model("process", _tiny_spec(tendency_input=True)).eval()
    control_state, candidate_state = control.state_dict(), candidate.state_dict()
    added = sorted(set(candidate_state) - set(control_state))
    shared_equal = (not (set(control_state) - set(candidate_state))
                    and all(torch.equal(control_state[key], candidate_state[key])
                            for key in control_state))
    patch = candidate.backbone.encoder.tendency_patch
    patch_zero = bool((patch.weight == 0).all())
    batch = _probe_batch()
    with torch.no_grad():
        a, b = control(batch), candidate(batch)
    forward_equal = (torch.equal(a.forecast, b.forecast)
                     and torch.equal(a.initial_forecast, b.initial_forecast)
                     and torch.equal(a.draft_forecasts, b.draft_forecasts))
    # A deterministic patch weight: capture the actual branch input and output and
    # replay the encoder's own patch + blocks + norm from those tensors.
    encoder = candidate.backbone.encoder
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(encoder.dim, TENDENCY_CHANNELS, 2, 2, generator=generator)
    captured = {}
    handle = encoder.tendency_patch.register_forward_hook(
        lambda module, inputs, output: captured.update(
            input=inputs[0].detach().clone(), output=output.detach().clone()))
    try:
        with torch.no_grad():
            patch.weight.copy_(weight)
            history = _probe_batch(height=4, width=6)["coarse_history"]
            result = encoder(history)[0]
    finally:
        handle.remove()
    with torch.no_grad():
        flat = history.reshape(1, -1, *history.shape[-2:])
        z = encoder.patch(flat) + captured["output"]
        height, width = z.shape[-2:]
        manual = z.flatten(2).transpose(1, 2)
        for block in encoder.blocks:
            manual = block(manual, (height, width))
        manual = encoder.norm(manual)
    exact = (torch.equal(result, manual)
             and torch.equal(captured["input"], history[:, -1] - history[:, -2]))
    with torch.no_grad():
        reversed_difference = encoder.tendency_patch(history[:, -2] - history[:, -1])
    sign_matters = not torch.equal(captured["output"], reversed_difference)
    # The declared parameter delta is stated at the frozen dim=192 / patch_size=2 shape.
    from model.coarse_encoder import CoarseEncoder
    real = CoarseEncoder(17, 2, dim=192, patch_size=2, depth=0, heads=2, window_size=2,
                         tendency_input=True)
    parameters = int(real.tendency_patch.weight.numel())
    real_shape = tuple(real.tendency_patch.weight.shape) == (192, TENDENCY_CHANNELS, 2, 2)
    checks = {
        "only_the_tendency_key_is_added": added == [TENDENCY_KEY],
        "shared_tensors_bitwise_equal": shared_equal,
        "tendency_patch_starts_at_zero": patch_zero,
        "candidate_forward_equals_control_at_init": forward_equal,
        "branch_is_exactly_conv_x_t_minus_x_prev": exact,
        "reversed_tendency_sign_changes_the_output": sign_matters,
        "no_bias": patch.bias is None,
        "real_shape_is_192x17x2x2": real_shape,
        "parameter_delta_is_the_declared_13056": parameters == TENDENCY_PARAMETERS,
    }
    return {"synthetic": True, "spec": _tiny_spec(), "checks": checks,
            "added_keys": added, "tendency_parameters": parameters,
            "tiny_tendency_parameters": int(patch.weight.numel()),
            "parameter_formula": "17 * dim * patch_size^2",
            "note": "synthetic tensors only; the bitwise results describe initialization, "
                    "not a trained-weather outcome"}
