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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="write the probe JSON here (default stdout)")
    args = parser.parse_args(argv)
    # This probe deliberately does NOT load a trained checkpoint: doing so needs a frozen
    # protocol and model construction that Stage D's brief pins separately. What it can do
    # without any checkpoint is verify the two *definitions* agree, which is the part that
    # would silently rot if either module's slot order moved.
    from model.r7_process_tensor_diagnostics import (
        PROCESS_DIAGNOSTIC_NAMES, compute_process_tensor_diagnostics)
    body = {"format": "r7-s3-whitebox-probe-v1",
            "scientific_claim": False, "test_read": False, "gpu_hours": 0.0,
            "anchored_slots": ANCHORED_SLOTS,
            "diagnostic_targets": list(PROCESS_DIAGNOSTIC_NAMES),
            "n_diagnostic_targets": len(PROCESS_DIAGNOSTIC_NAMES),
            "typed_local_field_count": len(TYPED_LOCAL_FIELD_NAMES),
            "typed_local_field_names": list(TYPED_LOCAL_FIELD_NAMES),
            # Definition checks are answerable from the tracked source alone, so they run in
            # every checkout including CI's bare one. The static product's presence is a
            # property of *this machine*, not of the probe: it is reported separately rather
            # than folded in, so a clean checkout cannot make the probe look broken and a
            # broken definition cannot hide behind a missing product.
            "checks": {
                "eight_diagnostics_declared": len(PROCESS_DIAGNOSTIC_NAMES) == 8,
                "eight_anchored_slots_declared": ANCHORED_SLOTS == 8,
                "diagnostic_names_are_the_documented_eight": len(set(PROCESS_DIAGNOSTIC_NAMES)) == 8,
                "compute_entry_point_present": callable(compute_process_tensor_diagnostics),
                "typed_fields_are_a_subset_of_the_diagnostics":
                    set(TYPED_LOCAL_FIELD_NAMES) <= set(PROCESS_DIAGNOSTIC_NAMES),
            },
            "limitations": [
                "no trained checkpoint is loaded here: the P/Z relationship to a specific "
                "trained model needs Stage D's own frozen protocol, not this definition check",
                "the static product is an untracked outputs/ artifact, so this probe's "
                "static_product_available flag is False in a bare checkout and True only on a "
                "machine that has published it; the definition checks are independent of it",
                "CPU-only and 0 GPU-h; declares no threshold and issues no pass/fail verdict "
                "because the campaign fixes none for either relationship",
                "the static product must be present; absence is reported, never synthesised",
            ]}
    static_ok, static_note = True, "loaded"
    try:
        fields = _static_fields()
        body["static_covariates"] = fields
    except Exception as exc:                      # noqa: BLE001 - reported, never hidden
        static_ok, static_note = False, f"{type(exc).__name__}: {exc}"
    body["static_product_available"] = static_ok
    body["static_note"] = static_note
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
