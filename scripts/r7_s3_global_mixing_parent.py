"""The H-C arm's migrated parent and its context-mixing expansion.

``scripts/study_r7_s3_global_mixing.py`` is the third S3 arm that changes ``model/`` (it adds
``model/global_mixing_r7.py`` and the ``global_mixing_mode`` switch on
``model/process_forecast_r7.py``), so ``model_code_digest()`` moves and the parent migrated for the
previous revision can no longer be loaded: ``training.r7_experiment.load_checkpoint`` refuses a
foreign implementation by design. This module pins the migration that the committed
``scripts/migrate_r7_parent_state.py`` produced under the current revision; it does not weaken that
guard. The weights are unchanged from the registered parent -- only the recorded digest and the
provenance differ -- and the receipt's revision is re-checked here so a stale pin fails loudly
instead of loading a stale parent.

Both H-C arms add the *same* twelve tensors (the shell is identical, only the attention range
differs), so unlike H-B there is no arm-only key: the whole added set is the shared initialization
and both receipts' ``pathway_init_sha256`` must be equal.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIGRATED_PARENT_RUN = ROOT / "outputs/r7_s3_rollout_dose_parent_migrated_20261015_attempt02"
MIGRATION_RECEIPT = MIGRATED_PARENT_RUN / "migration_receipt.json"
ENDPOINT = 1600
EQUIVALENCE_SEED = 1234
MIGRATED_PARENT_SHA256 = {
    41: "e8b3188dcef6801563f7682696919bcff9eb50bfc3ec371fdb9bfadc03641b85",
    42: "2766f6013486719d50ef47e7d0a8d65aa4cb9b63300753e158152b605e59ca21",
    43: "73d1053b59b5e4529556133bc318bc37c2e37ee7069e5d1462b4b967a508154c",
}
#: The registered geometry the FLOPs comparison is stated at (``docs/goals/s3-global-mixing.md``).
FLOPS_REFERENCE = {"batch": 1, "tokens": 1089, "dim": 192, "heads": 4, "window": 4,
                   "padded_hw": [36, 36], "windows": 81, "tokens_per_window": 16}


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
    """Replace the inherited migrated-parent checkpoint pins with this revision's migration."""
    body["arms"]["parent"]["pins"]["checkpoints"] = {
        str(seed): {"path": str(MIGRATED_PARENT_RUN / f"seed{seed}" / f"parent_update_{ENDPOINT:07d}.pt"),
                    "sha256": MIGRATED_PARENT_SHA256[seed]}
        for seed in MIGRATED_PARENT_SHA256}


def mixing_keys(mode):
    """The state-dict keys the context-mixing layer adds. Identical for both H-C modes."""
    from model.global_mixing_r7 import MIXING_MODES
    if mode not in MIXING_MODES:
        raise ValueError(f"mixing mode must be one of {MIXING_MODES}")
    prefix = "global_mixing.block."
    return tuple(prefix + leaf for leaf in (
        "n1.weight", "n1.bias", "qkv.weight", "qkv.bias", "proj.weight", "proj.bias",
        "n2.weight", "n2.bias", "ff.net.0.weight", "ff.net.0.bias",
        "ff.net.3.weight", "ff.net.3.bias"))


def expand_parent_state(parent_state, spec, mode, *, seed):
    """Load a parent without a context-mixing layer into the H-C trunk, which adds one.

    Returns ``(model, state, receipt)``. The registered parent carries no ``global_mixing`` key at
    all, so every mixing tensor is new; the parent's own tensors are installed bit for bit. The
    layer is drawn under ``seed``, so the global and the local arm of one round seed start from the
    *same* mixing initialization and the L->G delta is attributable to the attention range rather
    than to a fresh random draw -- the confound the H-B round's review flagged. The receipt records
    the raw positive-control output, and a failed control aborts the run rather than being written
    down and ignored.
    """
    import torch

    from training.r7_experiment import make_model
    from training.r7_autoregressive_runner import _state_digest
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        target = make_model("process", dict(spec, global_mixing_mode=mode))
    target_state = target.state_dict()
    missing = [key for key in target_state if key not in parent_state]
    expected = mixing_keys(mode)
    if set(missing) != set(expected):
        raise RuntimeError(f"the context-mixing layer must add exactly {sorted(expected)}, "
                           f"got {sorted(missing)}")
    extra = [key for key in parent_state if key not in target_state]
    if extra:
        raise RuntimeError(f"the parent carries keys the target does not: {sorted(extra)}")
    state = dict(parent_state)
    for key in missing:
        state[key] = target_state[key].detach().clone()
    target.load_state_dict(state, strict=True)
    control = positive_control(spec)
    if not (control["shell_parity"]["parameter_counts_equal"]
            and control["shared_bit_identical"]
            and control["zeroed_block_is_identity"]
            and control["arms_are_not_identical"]):
        raise RuntimeError("H-C positive control failed: the two arms are not parameter-matched, "
                           "do not share one initialization, the zeroed block is not the identity, "
                           "or the arms do not differ at all")
    # Digest over the added tensors; both arms add the same set, so the two receipts' values must
    # be equal -- that equality is the checkable evidence that they share one initialization.
    pathway = {key: state[key] for key in sorted(expected)}
    receipt = {"parent_keys": len(parent_state), "new_keys": sorted(missing),
               "init_seed": int(seed), "mixing_mode": mode,
               "parent_state_sha256": _state_digest(parent_state),
               "expanded_state_sha256": _state_digest(state),
               "pathway_init_sha256": _state_digest(pathway),
               "new_params": sum(int(target_state[key].numel()) for key in missing),
               "positive_control": control}
    return target, state, receipt


def _analytic_flops(reference=None):
    """Analytic per-layer multiply-accumulate counts for both arms, and the three ratios.

    ``torch.utils.flop_counter`` does not decompose the fused SDPA in this environment, so the
    attention cost is counted from the tensor shapes instead. Three conventions are reported
    because they are three different numbers and the protocol requires the convention to be named:
    the attention length per query, the SDPA sub-cost, and the whole layer including the FF.
    """
    ref = dict(FLOPS_REFERENCE if reference is None else reference)
    n, d, heads, window = ref["tokens"], ref["dim"], ref["heads"], ref["window"]
    windows, per_window = ref["windows"], ref["tokens_per_window"]

    def layer(kind):
        norms = 2 * (n * d * 2)
        qkv = n * 3 * d * d * 2
        proj = n * d * d * 2
        feed_forward = 2 * (n * d * 4 * d * 2)
        if kind == "global":
            attention = 2 * (n * n * d * 2)
        else:
            attention = 2 * (heads * windows * per_window * per_window * (d // heads) * 2)
        return norms + qkv + attention + proj + feed_forward, attention

    global_total, global_attention = layer("global")
    local_total, local_attention = layer("local")
    return {
        "reference": ref,
        "attention_length_ratio": n / per_window,
        "global_attention_flops": global_attention, "local_attention_flops": local_attention,
        "attention_ratio": global_attention / local_attention,
        "global_layer_flops": global_total, "local_layer_flops": local_total,
        "layer_ratio": global_total / local_total,
        "aligned": False,
    }


def positive_control(spec):
    """The raw counter-proofs: the two H-C arms are parameter-matched and share one init.

    Four measured facts, none of them a claim about trained behaviour. (1) The two arms' layers
    declare the same tensors with the same shapes, so both add the same parameter count. (2) Under
    one seed the two models share every tensor outside the layer, and those are bit-identical.
    (3) Zeroing the output projection *and* the FF output layer makes the layer the exact identity,
    so a forward returns its input bit for bit -- zeroing the projection alone does not, because
    the residual is followed by the FF. (4) The two arms' layers differ on a token grid larger than
    one window, so the negative control is a real manipulation and not a no-op.
    """
    import torch

    from model.global_mixing_r7 import GLOBAL, LOCAL, ContextMixLayer, GlobalAttentionBlock
    from model.layers.window_attention import WindowAttentionBlock
    from training.r7_experiment import make_model
    dim, heads = int(spec["dim"]), int(spec["heads"])
    window = int(spec.get("window_size", 4))
    dropout = float(spec.get("dropout", 0.0) or 0.0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(EQUIVALENCE_SEED)
        global_layer = ContextMixLayer(dim, heads, window, mode=GLOBAL, dropout=dropout)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(EQUIVALENCE_SEED)
        local_layer = ContextMixLayer(dim, heads, window, mode=LOCAL, dropout=dropout)
    global_params = sum(p.numel() for p in global_layer.parameters())
    local_params = sum(p.numel() for p in local_layer.parameters())
    shell = GlobalAttentionBlock(dim, heads, 4.0, dropout)
    windowed = WindowAttentionBlock(dim, heads, window, 4.0, dropout, shift=False)
    shell_parity = {
        "parameter_counts_equal": global_params == local_params,
        "global_params": int(global_params), "local_params": int(local_params),
        "shell_keys_equal": sorted(shell.state_dict()) == sorted(windowed.state_dict()),
        "global_shell_params": int(sum(p.numel() for p in shell.parameters())),
        "windowed_shell_params": int(sum(p.numel() for p in windowed.parameters())),
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(EQUIVALENCE_SEED)
        global_model = make_model("process", dict(spec, global_mixing_mode=GLOBAL))
        global_state = global_model.state_dict()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(EQUIVALENCE_SEED)
        local_state = make_model("process", dict(spec, global_mixing_mode=LOCAL)).state_dict()
    added = sorted(set(global_state) - set(local_state))
    lost = sorted(set(local_state) - set(global_state))
    shared = [key for key in local_state if key in global_state]
    shared_bit_identical = (not added and not lost
                            and all(torch.equal(local_state[key], global_state[key])
                                    for key in shared))
    # (3) and (4) are probed on the layer itself: a token grid of 12x12 is larger than one window
    # for every window size this round uses, so the two arms cannot coincide there.
    probe_tokens = torch.randn(2, 144, dim)
    probe_hw = (12, 12)
    with torch.no_grad():
        global_out = global_layer(probe_tokens, probe_hw)
        local_out = local_layer(probe_tokens, probe_hw)
        reference = global_out
        global_layer.block.proj.weight.zero_()
        global_layer.block.proj.bias.zero_()
        proj_only = global_layer(probe_tokens, probe_hw)
        global_layer.block.ff.net[3].weight.zero_()
        global_layer.block.ff.net[3].bias.zero_()
        identity = global_layer(probe_tokens, probe_hw)
    proj_only_max_abs = float((reference - proj_only).abs().max())
    identity_max_abs = float((probe_tokens - identity).abs().max())
    arms_max_abs = float((global_out - local_out).abs().max())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(EQUIVALENCE_SEED)
        off_model = make_model("process", spec)
    off_params = sum(p.numel() for p in off_model.parameters())
    on_params = sum(p.numel() for p in global_model.parameters())
    return {
        "seed": EQUIVALENCE_SEED,
        "shell_parity": shell_parity,
        "added_keys": added, "lost_keys": lost, "n_shared_compared": len(shared),
        "shared_bit_identical": bool(shared_bit_identical),
        "new_keys_match_mixing_keys": sorted(set(global_state) - set(off_model.state_dict()))
        == sorted(mixing_keys(GLOBAL)),
        "model_param_delta": int(on_params - off_params),
        "proj_only_max_abs": proj_only_max_abs,
        "proj_only_is_identity": proj_only_max_abs == 0.0,
        "zeroed_block_max_abs": identity_max_abs,
        "zeroed_block_is_identity": bool(torch.equal(probe_tokens, identity)),
        "arms_probe_max_abs": arms_max_abs,
        "arms_are_not_identical": arms_max_abs > 0.0,
        "flops": _analytic_flops(),
    }
