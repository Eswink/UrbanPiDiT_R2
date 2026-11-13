"""The placebo construction and its measurement (decision 0113; kept out of the protocol module).

Decision 0113 D1 defines the placebo arm precisely: the same published 8-layer field, with each
covariate's cells spatially permuted under one fixed seed, reusing the published projection. What
this module adds is the part that makes the arm *auditable* rather than merely constructed:

- the raw multiset equality that the round freezes (sorted, bit-identical) is measured;
- the token-level attenuation the permutation causes is measured and **reported as a confound** -
  ``static_features`` reads 2x2 patch means, so a per-cell permutation narrows the distribution the
  pathway sees even though the raw values are the same multiset;
- the spatial autocorrelation the permutation is supposed to destroy is measured;
- the alternative constructions (block permutations) are measured and recorded as *rejected*, so a
  later reader can see the trade-off was considered instead of assuming it was not.

Keeping this beside the protocol rather than inside it is also a structural rule (R-051): the
protocol module was over the 600-line cap with this material in it.
"""
from __future__ import annotations

PLACEBO_SEED = 20261113


def placebo_field(fields, *, seed: int = PLACEBO_SEED):
    """Spatially permute each covariate independently; return the field and the permutation.

    Every covariate gets its own fixed-seed permutation of the 65*65 cells, so the placebo keeps
    each covariate's **multiset exactly** (sorted, bit-identical to the treatment) and changes
    only *where* each value sits. Channels are **not** shuffled between variables: that would
    change each token's feature distribution, which is a different and uncontrolled intervention.
    """
    import torch
    count, rows, columns = fields.shape
    cells = rows * columns
    generator = torch.Generator().manual_seed(int(seed))
    permutations, permuted = [], []
    for index in range(count):
        order = torch.randperm(cells, generator=generator)
        permutations.append(order)
        permuted.append(fields[index].reshape(cells)[order].reshape(rows, columns))
    stack = torch.stack(permuted)
    if stack.shape != fields.shape or stack.dtype != fields.dtype:
        raise RuntimeError("the placebo field is not the same shape and dtype as the treatment")
    return stack, permutations


def family_argument(treatment, variables, variability=None) -> dict:
    """J4: how this input differs from the two already-closed input-side branches.

    Both closed branches fed the model a *function of the dynamic state*: the explicit tendency is
    a linear transform of it (decision 0049), and the four physical diagnostics are a non-linear
    but same-time derivable transform (``record:s3-typed-spatial-result``, |delta rmse| <= 3.2e-6 K).
    This round's input is S(x, y): one snapshot per covariate, exogenous to the atmosphere.
    """
    return {
        "input": "S(x, y): eight time-invariant urban / land-cover covariates on the 65x65 grid",
        "prior_branch_explicit_tendency": {
            "closed_by": "decision 0049",
            "relation_to_state": "linear transform of the 17 channels (X_t - X_{t-1})",
            "why_not_the_same_axis": "a linear transform carries no information the channels lack"},
        "prior_branch_typed_spatial": {
            "closed_by": "record:s3-typed-spatial-result",
            "relation_to_state": "non-linear but same-time derivable from the 17 channels",
            "measured": "B->C |delta rmse| <= 3.2e-6 K, i.e. float32 noise",
            "why_not_the_same_axis": "same-time derivable, so it adds no exogenous information"},
        "this_branch": {
            "relation_to_state": "none: the covariates describe the surface, not the atmosphere",
            "time_invariance": "one published snapshot per covariate; temporal spread is exactly 0",
            "measured_dynamic_temporal_spread": variability,
            "checkable_claim": ("a quantity that is constant in time cannot be produced by a "
                               "non-constant function of the time-varying channels; the measured "
                               "per-channel temporal spread above is the witness")},
        "the_alternative_this_round_must_beat": {
            "statement": ("a per-cell map from climatological channels to the covariates does exist "
                          "in principle - and is itself a function of position, which is exactly "
                          "what the trunk's sin/cos(lat), sin/cos(lon) harmonics already supply"),
            "how_it_is_isolated": "the placebo arm: same 8-field multiset, same projection, spatial "
                                  "structure destroyed",
            "residual_limit": ("the permutation also attenuates token amplitude, measured in "
                              "preflight.placebo_contrast, so (f) is a necessary but not "
                              "sufficient condition for attributing the effect to structure")},
        "verdict": "family-distinct from both closed branches: exogenous and time-invariant, not a "
                   "transform of the dynamic state",
    }


def token_view(fields, variables):
    """The per-token feature matrix and the projected codes the pathway actually produces."""
    from model import static_conditioning_r7 as static
    features = static.static_features(fields, token_hw=(33, 33), patch_size=2,
                                      variables=variables)
    conditioner = static.StaticConditioning(fields, dim=192, patch_size=2, periodic_width=False,
                                            variables=variables)
    codes = conditioner(token_hw=(33, 33)).reshape(-1, 192)
    return features, codes


def moran(field) -> float:
    """Join-count style spatial autocorrelation over the four nearest neighbours.

    Recorded because it is the quantity the placebo is supposed to destroy: a per-cell permutation
    drives it to about zero, so "the placebo has no spatial structure" is measured, not assumed.
    """
    values = field.to("cpu").double()
    centred = values - values.mean()
    denominator = (centred * centred).mean()
    if float(denominator) == 0.0:
        return 0.0
    horizontal = (centred[:, :-1] * centred[:, 1:]).mean()
    vertical = (centred[:-1, :] * centred[1:, :]).mean()
    return float((horizontal + vertical) / (2 * denominator))


def placebo_contrast(treatment, placebo, variables) -> dict:
    """Measure how the placebo differs from the treatment, in raw, token and code space.

    Two facts are recorded because exit (f)'s reading depends on both: the raw multisets are
    bit-identical (so the construction is the frozen one), and the *token-level* amplitude is not
    (because the pathway reads 2x2 patch means, which a per-cell permutation shrinks toward the
    global mean). The second number is the confound this round must state, not hide.
    """
    import torch
    multiset = {name: bool(torch.equal(torch.sort(treatment[index].reshape(-1)).values,
                                      torch.sort(placebo[index].reshape(-1)).values))
                for index, name in enumerate(variables)}
    treatment_features, treatment_codes = token_view(treatment, variables)
    placebo_features, placebo_codes = token_view(placebo, variables)
    feature_ratio = float(placebo_features.std(dim=0).mean() / treatment_features.std(dim=0).mean())
    code_ratio = float(placebo_codes.std(dim=0).mean() / treatment_codes.std(dim=0).mean())
    moran_treatment = [moran(treatment[index]) for index in range(len(variables))]
    moran_placebo = [moran(placebo[index]) for index in range(len(variables))]
    token_identity = {
        "treatment_tokens_equal_placebo_multiset": bool(torch.equal(
            torch.sort(treatment_features, dim=0).values,
            torch.sort(placebo_features, dim=0).values)),
        "note": ("false is expected and is the measured confound: a per-cell permutation does not "
                 "preserve patch means, so the placebo's per-token features are a narrower "
                 "distribution than the treatment's"),
    }
    return {
        "raw_multiset_bitwise_identical": multiset,
        "raw_multiset_all_identical": all(multiset.values()),
        "token_multiset": token_identity,
        "feature_std_ratio_placebo_over_treatment": feature_ratio,
        "code_std_ratio_placebo_over_treatment": code_ratio,
        "code_l2_per_token": {"treatment": float(treatment_codes.norm(dim=1).mean()),
                              "placebo": float(placebo_codes.norm(dim=1).mean())},
        "spatial_autocorrelation": {
            "treatment_per_variable": moran_treatment, "placebo_per_variable": moran_placebo,
            "treatment_mean": sum(moran_treatment) / len(moran_treatment),
            "placebo_mean": sum(moran_placebo) / len(moran_placebo)},
        "rejected_constructions": rejected_constructions(treatment, variables),
        "confound_statement": (
            "the placebo differs from the treatment in two ways - no spatial structure AND lower "
            f"token amplitude (feature spread {feature_ratio:.4f}x, code spread {code_ratio:.4f}x). "
            "A true exit (f) therefore shows that the coherent, full-amplitude static term beats a "
            "structureless attenuated one; it does not by itself separate structure from amplitude"),
    }


def rejected_constructions(treatment, variables) -> list:
    """The alternative placebo constructions, measured, so the choice is auditable.

    Block permutations keep more of the token amplitude (they move 2x2 blocks, which the patch
    mean reads as one unit) but leave residual spatial structure. They are recorded, not used:
    the round's frozen construction is the per-cell permutation, and swapping it after seeing
    these numbers would be choosing the placebo to suit an outcome.
    """
    rows = []
    for block in (2, 4):
        candidate = _block_permute(treatment, block)
        features, _ = token_view(candidate, variables)
        base, _ = token_view(treatment, variables)
        rows.append({
            "construction": f"block_{block}x{block}_core",
            "raw_multiset_bitwise_identical": _same_multiset(treatment, candidate, variables),
            "feature_std_ratio": float(features.std(dim=0).mean() / base.std(dim=0).mean()),
            "autocorrelation_mean": sum(moran(candidate[index])
                                        for index in range(len(variables))) / len(variables),
            "why_not_used": ("higher token-amplitude retention but non-zero residual spatial "
                             "structure, so it would weaken the very contrast the placebo exists "
                             "to create; recorded rather than substituted"),
        })
    return rows


def _same_multiset(left, right, variables) -> bool:
    import torch
    return all(bool(torch.equal(torch.sort(left[index].reshape(-1)).values,
                                torch.sort(right[index].reshape(-1)).values))
               for index in range(len(variables)))


def _block_permute(fields, block: int):
    """Permute whole ``block x block`` tiles of the complete core; leave the odd border in place."""
    import torch
    count, rows, columns = fields.shape
    core_rows, core_columns = (rows // block) * block, (columns // block) * block
    tiles_down, tiles_across = core_rows // block, core_columns // block
    generator = torch.Generator().manual_seed(PLACEBO_SEED)
    out = []
    for index in range(count):
        core = fields[index][:core_rows, :core_columns].reshape(
            tiles_down, block, tiles_across, block)
        order = torch.randperm(tiles_down * tiles_across, generator=generator)
        canvas = fields[index].clone()
        canvas[:core_rows, :core_columns] = core.reshape(
            tiles_down * tiles_across, block, block)[order].reshape(
            tiles_down, block, tiles_across, block).reshape(core_rows, core_columns)
        out.append(canvas)
    return torch.stack(out)


def arm_state(parent, treatment_field, placebo_state, *, arm: str,
              arms, control: str, buffer_keys):
    """The exact state dict one arm loads ``strict=True``: parent tensors plus two buffers.

    ``control`` is the bare parent - it has *no* static keys, which is what makes its pathway
    ``None`` and its parameter count identical to the registered recipe. ``treatment`` and
    ``placebo`` add the **same** projection object and differ only in ``field``.
    """
    from model import static_conditioning_r7 as static
    if arm == control:
        return dict(parent["state"])
    if arm not in arms:
        raise ValueError(f"unknown arm {arm!r}")
    fields, variables = treatment_field
    # ``treatment_field[0]`` is the published stack; the other on arm gets ``placebo_state``.
    chosen = fields if arm == arms[1] else placebo_state
    conditioner = static.StaticConditioning(chosen, dim=192, patch_size=2, periodic_width=False,
                                            variables=variables)
    return {**parent["state"], buffer_keys[0]: conditioner.field,
            buffer_keys[1]: conditioner.projection}


def build_arm(arm: str, spec_for):
    """A fresh arm module, built without moving the caller's RNG."""
    import torch
    from training.r7_experiment import make_model
    with torch.random.fork_rng(devices=[]):
        return make_model("process", spec_for(arm))


def load_strict(state, arm: str, spec_for):
    """Build the arm and load the state strictly; return the model so the caller can measure it."""
    model = build_arm(arm, spec_for)
    model.load_state_dict(state, strict=True)
    return model


def parameter_measurements(model, arm: str, buffer_keys) -> dict:
    """Parameter count, and whether the two static buffers are parameters - both measured.

    ``parameter_delta`` is the reason a skill difference here cannot be a capacity difference, so
    it is measured per arm from the built module rather than written as zero.
    """
    parameters = {name: int(tensor.numel()) for name, tensor in model.named_parameters()}
    return {"arm": arm, "parameter_tensors": len(parameters),
            "parameters": int(sum(parameters.values())),
            "static_buffers_are_parameters": [name for name in buffer_keys if name in parameters],
            "state_tensors": len(model.state_dict()),
            "has_static_pathway": model.backbone.static_conditioning_pathway is not None}


def temporal_variability(store, *, samples: int = 24, crop: int = 8) -> dict:
    """Per-channel temporal spread of the dynamic inputs, next to the static field's zero.

    The family argument (decision 0113, "受约束 0049") rests on a difference that must be
    *checkable*: the static covariates are time-invariant by construction, while the 17 dynamic
    channels are not. This reads a small crop at several times and reports each channel's temporal
    standard deviation - cheap, read-only, and enough to show the two are not inter-derivable.
    """
    import numpy as np
    import zarr
    root = zarr.open_group(str(store), mode="r")
    total = int(root["state"].shape[0])
    if samples < 2 or total < samples:
        raise ValueError("need at least two time samples to measure temporal spread")
    picked = np.linspace(0, total - 1, samples).astype(int)
    middle = int(root["state"].shape[-1]) // 2
    low, high = max(0, middle - crop // 2), min(int(root["state"].shape[-1]), middle + crop // 2)
    channels = list(root.attrs["channels"])
    windows = []
    for step in picked:
        block = np.asarray(root["state"][int(step), :, low:high, low:high], dtype=np.float64)
        windows.append(block.reshape(len(channels), -1))
    stacked = np.stack(windows)                       # [samples, channels, cells]
    spread = stacked.std(axis=0).mean(axis=1)
    return {"time_indices": [int(step) for step in picked], "crop": [low, high],
            "channels": channels,
            "temporal_std_per_channel": [float(value) for value in spread],
            "min_temporal_std": float(spread.min()), "max_temporal_std": float(spread.max()),
            "note": ("the 17 dynamic channels vary in time at every sampled channel; the published "
                     "static field is one snapshot per covariate, so its temporal spread is exactly "
                     "zero and no function of the dynamic channels at one time equals it")}
