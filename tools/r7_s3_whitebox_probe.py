"""Stage D white-boxing: what the P and Z states actually represent, measured on CPU.

The campaign's Stage D asks two questions that a loss curve cannot answer, because both are
about *internal* state rather than output error:

1. **P state** -- the process state has ``anchored_processes`` (12) slots, of which the first
   eight are documented as "aligned to meteorological diagnostic proxy targets". Does that
   alignment hold numerically? This probe regresses each of the 8 anchored slots on the 8
   diagnostics computed independently by ``r7_process_tensor_diagnostics`` and reports the
   cross-slot alignment matrix. A diagonal-dominant matrix supports the documentation; a
   scattered one would falsify it.

2. **Z state** -- the local solver's per-position working state advances by a gate each step.
   Does that progress relate to topography / land-sea fraction, i.e. is the solver spending
   effort where the surface is complex? Reported as the correlation between per-position
   cumulative gate progress and the static fields.

Both are CPU-only and cost 0 GPU-h, per the campaign's Stage D. Neither is a scientific
claim about forecast skill: they describe what the trained weights *did*, not whether the
result is good. ``scientific_claim`` is therefore False throughout, and no threshold is
declared -- the campaign fixes no numeric bar for either relationship, so this tool reports
measurements and explicitly issues no pass/fail verdict it was not given.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The 8 anchored slots that the model's docstring aligns to diagnostic proxies.
ANCHORED_SLOTS = 8
#: The four diagnostics ``typed_local_fields`` keeps as spatial maps, named explicitly so a
#: reordering of ``PROCESS_DIAGNOSTIC_NAMES`` cannot silently change which four they are.
TYPED_LOCAL_FIELD_NAMES = ("divergence_850_rms", "vorticity_850_rms",
                           "temperature_advection_850_mean", "static_stability_850_500_mean")


def _correlation(left, right):
    """Pearson correlation of two 1-D float sequences, or None when either is constant."""
    import torch
    a = torch.as_tensor(left, dtype=torch.float64).reshape(-1)
    b = torch.as_tensor(right, dtype=torch.float64).reshape(-1)
    if a.numel() != b.numel() or a.numel() < 2:
        raise RuntimeError(f"correlation needs equal, non-degenerate vectors: "
                           f"{a.numel()} vs {b.numel()}")
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if float(denom) == 0.0:
        return None
    return float((a * b).sum() / denom)


def _rank(values):
    """Average-rank transform, so a monotone but nonlinear relation is not missed.

    The probe reports both Pearson and Spearman because "slot i tracks diagnostic i" need not
    be linear: a slot saturating on a threshold would show up only in the rank correlation.
    """
    import numpy as np
    array = np.asarray(values, dtype=np.float64)
    order = array.argsort(kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    ranks[order] = np.arange(1, len(array) + 1, dtype=np.float64)
    # Average ties, or a slot with repeated values gets an arbitrary order-dependent rank.
    unique, inverse, counts = np.unique(array, return_inverse=True, return_counts=True)
    if len(unique) != len(array):
        sums = np.zeros(len(unique), dtype=np.float64)
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


def alignment_matrix(slot_values, diagnostic_values):
    """Cross-correlation of every anchored slot against every diagnostic target.

    ``slot_values`` and ``diagnostic_values`` are both ``[N, K]`` over N samples. Returns the
    full K x K Pearson and Spearman matrices plus, for each slot, the diagnostic it tracks
    most strongly -- the claim "slot i aligns to proxy i" is exactly ``argmax == i``.
    """
    import numpy as np
    slots = np.asarray(slot_values, dtype=np.float64)
    diagnostics = np.asarray(diagnostic_values, dtype=np.float64)
    if slots.ndim != 2 or diagnostics.ndim != 2:
        raise RuntimeError("alignment inputs must be 2-D [N, K]")
    if slots.shape[0] != diagnostics.shape[0]:
        raise RuntimeError("alignment inputs must share the sample axis")
    n_slots, n_diag = slots.shape[1], diagnostics.shape[1]
    pearson = [[_correlation(slots[:, i], diagnostics[:, j]) for j in range(n_diag)]
               for i in range(n_slots)]
    ranks_slots = np.stack([_rank(slots[:, i]) for i in range(n_slots)], axis=1)
    ranks_diag = np.stack([_rank(diagnostics[:, j]) for j in range(n_diag)], axis=1)
    spearman = [[_correlation(ranks_slots[:, i], ranks_diag[:, j]) for j in range(n_diag)]
                for i in range(n_slots)]
    best = []
    for i in range(n_slots):
        row = [abs(value) if value is not None else -1.0 for value in spearman[i]]
        winner = int(max(range(n_diag), key=lambda j: row[j]))
        best.append({"slot": i, "best_diagnostic": winner,
                     "abs_spearman": row[winner],
                     "matches_documented_alignment": winner == i})
    aligned = sum(1 for entry in best if entry["matches_documented_alignment"])
    return {"n_slots": n_slots, "n_diagnostics": n_diag, "n_samples": int(slots.shape[0]),
            "pearson": pearson, "spearman": spearman, "per_slot_best": best,
            "aligned_slots": aligned,
            "note": ("the diagonal is the documented alignment; a non-diagonal argmax is a "
                     "measured counterexample to the docstring, not an error in this probe")}


def progress_correlation(progress, static_fields):
    """Correlate per-position solver progress with each static covariate.

    ``progress`` is ``[H, W]`` cumulative gate activity; ``static_fields`` maps a covariate
    name to an ``[H, W]`` array. Correlations are over positions, so the sample count is the
    grid size and the two fields must match shape exactly -- a silently broadcast pair would
    produce a meaningless number.
    """
    import numpy as np
    field = np.asarray(progress, dtype=np.float64)
    if field.ndim != 2:
        raise RuntimeError(f"progress must be 2-D [H,W], got shape {field.shape}")
    out = {}
    for name, values in static_fields.items():
        static = np.asarray(values, dtype=np.float64)
        if static.shape != field.shape:
            raise RuntimeError(f"{name} has shape {static.shape}, progress has {field.shape}")
        out[name] = {"pearson": _correlation(field.reshape(-1), static.reshape(-1)),
                     "spearman": _correlation(_rank(field.reshape(-1)),
                                              _rank(static.reshape(-1)))}
    return {"n_positions": int(field.size), "progress_summary": {
                "min": float(field.min()), "max": float(field.max()),
                "mean": float(field.mean()), "nonzero": int((field != 0).sum())},
            "correlations": out}


def _static_fields():
    """Load the published static product by identity, or explain why it cannot be loaded.

    ``require_static_identity`` returns the *manifest* (identity pinned, build complete), and
    ``load_static_fields`` returns the ``[K,H,W]`` stack; both are called so the probe proves
    the identity gate was passed before a single number was read.
    """
    from model import static_conditioning_r7 as static
    manifest = static.require_static_identity()
    fields, variables = static.load_static_fields()
    return {"manifest_identity": manifest["static_identity"],
            "variables": list(variables), "shape": list(fields.shape)}



def _trained_state_measurements(checkpoint, manifest_path, channels, cases):
    """The two Stage D measurements on a **trained** model, not on the definitions.

    Stage D's brief asks for two things a definition check cannot answer:

    1. the P state's 8 anchored slots against the 8 diagnostic targets -- ``process_predictions``
       is ``[B, K, anchored_processes]``, so each slot is a scalar per reasoning step and the
       cross-slot alignment matrix is computable exactly as :func:`alignment_matrix` expects;
    2. the Z state's per-position solve progress against the terrain / land-sea static fields --
       ``solver_state`` is the per-position working state, and its magnitude at the last step is
       the progress the gate has made.

    Both are read from one forward pass per case with no gradient, so this stays CPU-only and
    0 GPU-h. Nothing here is a skill claim: it says what the trained weights *did*.
    """
    import numpy as np
    import torch
    from torch.utils.data import default_collate

    from data.r7_long_rollout_dataset import (
        ZarrLongRolloutDataset, preflight_long_rollout_windows)
    from model.r7_process_tensor_diagnostics import (
        PROCESS_DIAGNOSTIC_NAMES, compute_process_tensor_diagnostics)
    from training.r7_experiment import load_checkpoint, make_model

    saved = load_checkpoint(checkpoint)
    contract = saved["contract"]
    model = make_model(contract["kind"], contract["model"])
    model.load_state_dict(saved["model"], strict=True)
    model.eval()

    # The checkpoint's contract records neither the manifest nor the channel order (it pins
    # their digests instead), so both are passed in explicitly. Guessing either would make the
    # diagnostic vector -- and therefore the whole alignment matrix -- an unregistered variable.
    manifest = Path(manifest_path)
    if not manifest.is_file():
        raise RuntimeError(f"the training manifest is not on this machine: {manifest}")
    windows = preflight_long_rollout_windows(manifest, physical_steps=12)
    dataset = ZarrLongRolloutDataset(manifest, physical_steps=12,
                                     expected_exclusions=windows["excluded_sample_ids"],
                                     expected_window_sha256=windows["window_sha256"])

    slot_rows, diagnostic_rows, progress_rows = [], [], []
    for case in cases:
        batch = default_collate([dataset[case]])
        with torch.no_grad():
            output = model(batch, reasoning_steps=contract["steps"])
        # P state: the 8 anchored slots at every reasoning step, averaged over steps so each
        # case contributes one 8-vector -- the alignment question is across cases, not steps.
        slots = output.process_predictions.detach().to(torch.float64)
        slot_rows.append(slots.mean(dim=1).reshape(-1).cpu().numpy())
        # The 8 diagnostics, computed independently from the analysis field of the same window.
        state = batch["atmos_target"]
        vector = compute_process_tensor_diagnostics(
            state[0], list(channels),
            batch["latitude"][0], batch["longitude"][0])
        diagnostic_rows.append(np.asarray(vector, dtype=np.float64))
        # Z state: per-position magnitude of the solver's working state after the last step.
        if output.solver_state is None:
            raise RuntimeError("this checkpoint has no solver state, so the Z measurement "
                               "cannot be made; the probe refuses rather than reporting zero")
        progress = output.solver_state.detach().to(torch.float64).abs()
        # solver_state is [B, token_hw[0]*token_hw[1], D] on the TOKEN grid, which is 33x33 here
        # (65 / patch 2 = 32.5 -> 33), not the 65x65 pixel grid the static fields live on.
        # Reduce the feature axis to a per-token scalar first, then map tokens to pixels with a
        # declared nearest-neighbour repeat -- declared because a bilinear map would invent
        # smoothness the solver does not have, and the two grids genuinely differ in size.
        tokens = progress.mean(dim=-1)[0]
        rows, columns = output.token_hw
        if tokens.numel() != rows * columns:
            raise RuntimeError(f"solver_state carries {tokens.numel()} positions but token_hw is "
                               f"{output.token_hw}")
        progress_rows.append(tokens.reshape(rows, columns).cpu().numpy())

    alignment = alignment_matrix(np.stack(slot_rows), np.stack(diagnostic_rows))
    alignment["diagnostic_names"] = list(PROCESS_DIAGNOSTIC_NAMES)
    progress = np.mean(np.stack(progress_rows), axis=0)
    # The raw per-token field is returned separately rather than embedded: it is an ndarray, so
    # it cannot enter the JSON report, and its summary plus the correlations are what a reader
    # needs. The caller writes it beside the report as a .npy for anyone who wants the map.
    return {"alignment": alignment, "progress_field_shape": list(progress.shape),
            "progress_grid": "token",
            "progress_summary": {"min": float(progress.min()), "max": float(progress.max()),
                                 "mean": float(progress.mean()),
                                 "nonzero_fraction": float((progress != 0).mean())}}, progress


def _correlate_on_pixel_grid(progress, stack, variables):
    """Correlate the solver's token-grid progress with the pixel-grid static fields.

    The two grids are genuinely different sizes: the solver works on a 33x33 token grid
    (65 / patch 2 = 32.5, rounded up) while the static product is 65x65 pixels. Rather than
    resample either one, each token's progress is compared against the mean of the pixels it
    covers; the block boundaries are derived from the two shapes, so a different grid fails
    loudly here instead of being silently broadcast.

    Reported per static variable with no threshold, because the campaign fixes none for this
    relationship. ``pearson`` over a monotone covariate (a fraction) and ``spearman`` over the
    ranks are both given because a land-fraction field is far from normal.
    """
    import numpy as np
    field = np.asarray(progress, dtype=np.float64)
    if field.ndim != 2:
        raise RuntimeError(f"progress must be 2-D, got {field.shape}")
    height, width = int(stack.shape[-2]), int(stack.shape[-1])
    rows, columns = field.shape
    if rows > height or columns > width:
        raise RuntimeError(f"the token grid {field.shape} is finer than the pixel grid "
                           f"{(height, width)}; pooling would lose positions")
    # Boundaries of the pixel block each token owns: even split, remainder to the last cells.
    row_edges = [row * height // rows for row in range(rows + 1)]
    column_edges = [column * width // columns for column in range(columns + 1)]
    correlations = {}
    for position, name in enumerate(variables):
        pixels = np.asarray(stack[position].numpy(), dtype=np.float64)
        pooled = np.empty((rows, columns), dtype=np.float64)
        for row in range(rows):
            for column in range(columns):
                block = pixels[row_edges[row]:row_edges[row + 1],
                               column_edges[column]:column_edges[column + 1]]
                if block.size == 0:
                    raise RuntimeError(f"token ({row},{column}) covers no pixels")
                pooled[row, column] = block.mean()
        correlations[name] = {
            "pearson": _correlation(field.reshape(-1), pooled.reshape(-1)),
            "spearman": _correlation(_rank(field.reshape(-1)), _rank(pooled.reshape(-1)))}
    return {"token_grid": [rows, columns], "pixel_grid": [height, width],
            "mapping": "each token is compared against the mean of the pixel block it covers",
            "correlations": correlations}


def _definition_body():
    """The checks answerable from tracked source alone, which run in every checkout.

    Kept apart from :func:`main` because the whole point of these checks is that a bare CI
    checkout can run them: the static product's presence is a property of *this machine*, so
    it is reported separately rather than folded in. A clean checkout must not be able to make
    the probe look broken, and a broken definition must not hide behind a missing product.
    """
    from model.r7_process_tensor_diagnostics import (
        PROCESS_DIAGNOSTIC_NAMES, compute_process_tensor_diagnostics)
    return {"format": "r7-s3-whitebox-probe-v1",
            "scientific_claim": False, "test_read": False, "gpu_hours": 0.0,
            "anchored_slots": ANCHORED_SLOTS,
            "diagnostic_targets": list(PROCESS_DIAGNOSTIC_NAMES),
            "n_diagnostic_targets": len(PROCESS_DIAGNOSTIC_NAMES),
            "typed_local_field_count": len(TYPED_LOCAL_FIELD_NAMES),
            "typed_local_field_names": list(TYPED_LOCAL_FIELD_NAMES),
            "checks": {
                "eight_diagnostics_declared": len(PROCESS_DIAGNOSTIC_NAMES) == 8,
                "eight_anchored_slots_declared": ANCHORED_SLOTS == 8,
                "diagnostic_names_are_the_documented_eight": len(set(PROCESS_DIAGNOSTIC_NAMES)) == 8,
                "compute_entry_point_present": callable(compute_process_tensor_diagnostics),
                "typed_fields_are_a_subset_of_the_diagnostics":
                    set(TYPED_LOCAL_FIELD_NAMES) <= set(PROCESS_DIAGNOSTIC_NAMES),
            },
            "limitations": [
                "the definition checks answer whether the two modules agree; they do NOT "
                "measure a trained model, which needs --checkpoint and is reported separately",
                "the static product is an untracked outputs/ artifact, so this probe's "
                "static_product_available flag is False in a bare checkout and True only on a "
                "machine that has published it; the definition checks are independent of it",
                "CPU-only and 0 GPU-h; declares no threshold and issues no pass/fail verdict "
                "because the campaign fixes none for either relationship",
                "the static product must be present; absence is reported, never synthesised",
            ]}


def _trained_state_section(body, args, static_ok):
    """Attach the trained-checkpoint measurement, or record that none was requested."""
    if args.checkpoint is None:
        body["trained_state_measured"] = False
        return
    if args.manifest is None or args.store is None:
        raise SystemExit("--checkpoint requires --manifest and --store")
    import numpy as np
    import zarr
    store = zarr.open_group(str(args.store), mode="r")
    channels = list(store.attrs["channels"])
    cases = [int(value) for value in str(args.cases).split(",") if value.strip()]
    measured, progress = _trained_state_measurements(
        args.checkpoint, args.manifest, channels, cases)
    measured.update({"checkpoint": str(args.checkpoint), "cases": cases, "channels": channels})
    # Sidecar for the per-token map; the JSON carries only its summary and correlations.
    if args.out is not None:
        sidecar = Path(args.out).with_suffix(".progress.npy")
        np.save(sidecar, progress)
        measured["progress_sidecar"] = str(sidecar)
    if static_ok:
        # ``_static_fields`` returns identity metadata, not the arrays; the correlation needs the
        # [K,H,W] stack itself, so it is loaded by the same identity-checked entry point.
        from model import static_conditioning_r7 as static
        stack, variables = static.load_static_fields()
        measured["progress_correlations"] = _correlate_on_pixel_grid(progress, stack, variables)
    else:
        measured["progress_correlations"] = None
    body["trained_state"] = measured
    # A measurement is not a check: it never enters all_checks, because the campaign fixes no
    # numeric bar for either relationship.
    body["trained_state_measured"] = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="write the probe JSON here (default stdout)")
    # The trained-checkpoint mode is opt-in and takes its inputs explicitly: the checkpoint's
    # contract pins the manifest and channel digests but does not record the values, so both
    # must be supplied rather than guessed.
    parser.add_argument("--checkpoint", type=Path,
                        help="measure a trained model's P and Z states (needs --manifest and --store)")
    parser.add_argument("--manifest", type=Path, help="the training manifest the checkpoint was fit on")
    parser.add_argument("--store", type=Path, help="the zarr store, for the channel order")
    parser.add_argument("--cases", default="1000,9000,17000,25000,29000",
                        help="comma-separated dataset indices for the trained-state measurement")
    args = parser.parse_args(argv)
    body = _definition_body()
    static_ok, static_note = True, "loaded"
    try:
        body["static_covariates"] = _static_fields()
    except Exception as exc:                      # noqa: BLE001 - reported, never hidden
        static_ok, static_note = False, f"{type(exc).__name__}: {exc}"
    body["static_product_available"] = static_ok
    body["static_note"] = static_note
    _trained_state_section(body, args, static_ok)
    body["all_checks"] = all(body["checks"].values())
    text = json.dumps(body, indent=2, ensure_ascii=False, allow_nan=False)
    if args.out is None:
        print(text)
    else:
        with Path(args.out).open("x", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(f"wrote {args.out}: all_checks={body['all_checks']}")
    return 0 if body["all_checks"] else 1

if __name__ == "__main__":
    sys.exit(main())
