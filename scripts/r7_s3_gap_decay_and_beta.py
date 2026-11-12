"""Step C: the two zero-GPU checks that decide whether Stage A is worth opening.

Two measurements, both CPU only, both read-only (``scientific_claim: false``,
``test_read: false``, ``r = 0``):

**C1 -- the decay law of the amplitude gap.** Stage P measured ``1 - amp_ratio`` shrinking
monotonically across four training endpoints and the goal brief read that as "the deficit is
under-training, so skip Stage Q" (decision 0111). A monotone decline is necessary but not
sufficient: what matters is the *rate*. If ``gap ~ N^-alpha`` with a small alpha, the gap is
present at every reachable budget and Stage Q's cancellation was a criterion defect rather
than a finding. This fits alpha per band per lead, reports the residual, and refuses to
report a verdict when the fit is not identifiable.

**C2 -- does the gap survive per-variable reweighting?** ``band_powers`` in
``training/r7_spectral_geometry.py:127`` sums over **all 17 channels**, so the band totals are
dominated by whichever variables carry the most power. FastNet's equation (8)
(arXiv:2509.17601 §4.1.2) says exactly why that matters:

    geopotential spectra span more than six orders of magnitude ... whereas specific humidity
    spans only three. Without this adjustment, the MSH loss tends to overweight v-winds and
    humidity while underweighting temperature and geopotential.

So the P1 gap that justified ``skip-q`` is a *power-weighted mixture*, and no artifact in the
round says what the gap looks like per variable. ``beta_j = 1 / <AMSE_j>`` is the paper's fix.
It needs only consecutive ground-truth pairs, so it is computed here from the store with **no
model forward**.

The point of C2 is to be able to *fail*: if the gap disappears once the variables are put on
comparable footing, then the whole spectral-geometry line rests on an artifact of unweighted
pooling and Stage A must not be opened. That is the outcome most worth being able to see.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import traceback

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The four training endpoints, in increasing exposure. Ordered by training volume, NOT by
#: name: sorting the names alphabetically puts p1600 before p4000 and inverts the trend, which
#: is the single cheapest way to get this measurement wrong.
TRAINING_POINTS = (
    ("p0025", 2008),
    ("p1000", 8033),
    ("p4000", 32132),
    ("p1600", 128528),
)

#: The eight train cases every physics probe in this campaign has used.
CASES = (1000, 5000, 9000, 13000, 17000, 21000, 25000, 29000)

#: Stage P's per-band product, reused rather than recomputed.
P1_PRODUCT = ("outputs/r7_s3_spectral_geometry_20261111_attempt01/"
              "p1_amplitude_coherence.json")

#: Store and protocol constants. Read, never re-derived: the manifest identity is asserted by
#: ``ZarrLongRolloutDataset`` itself when the forward path is used.
STORE_REL = "outputs/r7_s3_fy_multiyear_20261023_attempt01/store/cache.zarr"
TRAIN_MANIFEST_REL = ("outputs/r7_s3_fy_multiyear_20261023_attempt01/store/"
                      "manifests_treatment/train.jsonl")

#: One 6 h step in nanoseconds, used to pair consecutive analysis fields for ``beta_j``.
STEP_NS = 6 * 3600 * 1_000_000_000


def _load_geometry():
    """The frozen decomposition helpers, so C2 cannot drift from the trained loss."""
    from training import r7_spectral_geometry as geometry
    return geometry


def lstsq_alpha(updates, gaps) -> dict:
    """Fit ``gap = C * N**-alpha`` by least squares in log space.

    Returns alpha plus the information a reader needs to judge it: the residual, the number of
    finite points used, and whether the fit is identifiable at all. Two points always fit a
    line exactly, so ``identifiable`` requires at least three finite points **and** a spread in
    ``log N`` -- a fit through one distinct abscissa has no slope.
    """
    pairs = [(float(n), float(g)) for n, g in zip(updates, gaps)
             if _finite(n) and _finite(g) and n > 0 and g > 0]
    if len(pairs) < 2:
        return {"alpha": None, "identifiable": False,
                "reason": f"only {len(pairs)} usable point(s); nothing to fit"}
    logs = np.log([n for n, _ in pairs])
    loggaps = np.log([g for _, g in pairs])
    if float(logs.max() - logs.min()) <= 1e-12:
        return {"alpha": None, "identifiable": False,
                "reason": "all points share one abscissa; the slope is not determined"}
    design = np.stack([np.ones_like(logs), logs], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, loggaps, rcond=None)
    predicted = design @ coefficients
    residual = float(np.abs(loggaps - predicted).max())
    return {
        "alpha": float(-coefficients[1]),
        "intercept": float(coefficients[0]),
        "identifiable": bool(len(pairs) >= 3),
        "points_used": len(pairs),
        "max_log_residual": residual,
        "reason": "" if len(pairs) >= 3 else "only two points; alpha is exact by construction",
    }


def updates_to_close(current_gap: float, alpha, target_ratio: float = 0.5,
                     current_updates: float = 128528.0) -> dict:
    """How much more training would cut ``current_gap`` by ``target_ratio``.

    ``gap ~ N^-alpha`` implies ``N2 / N1 = ratio**(-1/alpha)``. A small positive alpha makes
    that ratio astronomical, which is the whole finding: the gap *does* shrink with training,
    but at a rate that leaves it present at any reachable budget.
    """
    if not _finite(current_gap) or current_gap <= 0:
        return {"reachable": False, "reason": "no positive current gap to extrapolate"}
    if not _finite(alpha) or alpha <= 0:
        return {"reachable": False, "alpha": alpha,
                "reason": "non-positive alpha: the gap does not shrink with training at all"}
    ratio = target_ratio ** (-1.0 / alpha)
    return {
        "reachable": True,
        "alpha": alpha,
        "updates_multiplier": ratio,
        "updates_needed": current_updates * ratio,
        "gpu_hours_at_1_1096_s_per_update": current_updates * ratio * 1.1096 / 3600.0,
    }


def beta_weights(store, year: int = 2001, normalized: bool = True) -> dict:
    """``beta_j = 1 / mean(AMSE_j)`` over consecutive 6 h truth pairs, per FastNet eq. (8).

    Computed from the analysis field alone -- no model, no forecast. The AMSE used is the same
    band form the training loss would apply (:func:`training.r7_spectral_geometry.amse_analogue`),
    so the weight is in the units the loss actually charges.

    **``normalized=True`` is the correct space and the default.** The store holds *physical*
    units (``physical_units_retained: true``, ``z250`` at ~1.0e5 m^2/s^2) while the model and
    every loss see fields normalised at read time. Computing ``beta_j`` on the physical field
    spans 11 orders of magnitude across the 17 variables; on the normalised field the spread
    collapses to roughly the order the paper describes ("geopotential spectra span more than
    six orders of magnitude ... whereas specific humidity spans only three" is a statement
    about the *un-normalised* PSD, but the weight itself must match the loss's own units or it
    rescales the whole term). Both spaces are reported so the choice is visible rather than
    silently baked in.

    ``year`` selects which year's pairs enter the mean; the paper reports the inter-year
    variance of ``beta_j`` as minimal and uses a single year, so one year is used here too and
    the choice is recorded rather than hidden.
    """
    import torch

    geometry = _load_geometry()
    times = np.asarray(store["time_ns"][:], dtype=np.int64)
    years = _year_of(times)
    selected = np.flatnonzero(years == year)
    if selected.size < 2:
        return {"year": year, "channels": {}, "reason": f"year {year} has <2 frames"}
    state = store["state"]
    channels = list(store.attrs["channels"])
    mean = np.asarray(store["normalization_mean"][:], dtype=np.float64)
    std = np.asarray(store["normalization_std"][:], dtype=np.float64)
    if normalized and (mean.shape != (len(channels),) or std.shape != (len(channels),)):
        raise ValueError("normalisation vectors do not match the channel count")
    if normalized and (not np.isfinite(std).all() or (std <= 0).any()):
        raise ValueError("normalisation_std must be positive and finite")
    edges = geometry.band_edges(65, 65)
    per_channel = {}
    # Consecutive pairs only: a pair straddling a gap is not a 6 h transition.
    pairs = 0
    for index in selected[:-1]:
        if int(times[index + 1] - times[index]) != STEP_NS:
            continue
        pair = np.asarray(state[index:index + 2], dtype=np.float64)
        if normalized:
            pair = (pair - mean[None, :, None, None]) / std[None, :, None, None]
        first = torch.as_tensor(pair[0], dtype=torch.float64)
        second = torch.as_tensor(pair[1], dtype=torch.float64)
        for position, name in enumerate(channels):
            value = float(geometry.amse_analogue(first[position], second[position], edges=edges))
            per_channel.setdefault(name, []).append(value)
        pairs += 1
    rows = {}
    for name, values in per_channel.items():
        mean_amse = float(np.mean(values)) if values else float("nan")
        rows[name] = {
            "mean_amse": mean_amse,
            "beta": (1.0 / mean_amse) if mean_amse > 0.0 else None,
            "samples": len(values),
        }
    return {"year": year, "pairs": pairs, "normalized": bool(normalized),
            "units_note": ("beta_j is computed in the normalised space the loss sees; "
                           "physical-unit values span ~11 orders and would rescale the term"),
            "channels": rows,
            "spread_max_over_min": _spread([row["beta"] for row in rows.values()])}


def _physical_control(store, year: int) -> dict:
    """``beta_j`` on the raw physical field, kept only to show why the normalised space is used.

    Computing this on the store as it stands spans ~11 orders of magnitude, because ``z250``
    carries ~1.0e5 m^2/s^2 while ``q500`` carries ~4.0e-4 kg/kg. That is not a bug in the store
    -- the store is *supposed* to hold physical units -- it is a statement that a per-variable
    spectral weight must live in the same space as the loss it multiplies. The pair of numbers
    (physical spread vs normalised spread) is the evidence for that claim, so both are reported.
    """
    try:
        raw = beta_weights(store, year=year, normalized=False)
    except Exception as exc:  # noqa: BLE001 - a control that fails must not kill the probe
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"available": True,
            "spread_max_over_min": raw.get("spread_max_over_min"),
            "pairs": raw.get("pairs"),
            "note": ("physical-unit beta_j; reported as the control for the normalised choice, "
                     "never used as the weight")}


def _spread(values) -> float:
    finite = [v for v in values if _finite(v) and v > 0]
    if len(finite) < 2:
        return float("nan")
    return float(max(finite) / min(finite))


def _year_of(times) -> np.ndarray:
    """Calendar year of each epoch-nanosecond stamp, in UTC."""
    import datetime

    return np.asarray([datetime.datetime.utcfromtimestamp(int(t) / 1e9).year for t in times])


def _finite(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def c1_decay(product: dict) -> dict:
    """Fit the gap decay law from the Stage P product, per band and per lead, both lead sets."""
    points = product["p1"]["points"]
    names = [point["name"] for point in points]
    order = [name for name, _ in TRAINING_POINTS]
    missing = [name for name in order if name not in names]
    if missing:
        raise ValueError(f"Stage P product lacks training points {missing}")
    # Reorder by training volume; the product's own order is whatever the probe emitted.
    by_name = {point["name"]: point for point in points}
    ordered = [by_name[name] for name in order]
    updates = [dict(TRAINING_POINTS)[name] for name in order]

    result = {"lead_sets": {}, "ordering": order, "updates": updates,
              "ordering_note": ("points are ordered by training volume, not by name: "
                                "alphabetical order puts p1600 before p4000 and inverts "
                                "the trend")}
    for label, key in (("frozen_five_lead", "bands"), ("all_twelve_lead", "bands_all_leads")):
        aggregates = [point[key] for point in ordered]
        band_count = len(aggregates[0])
        per_band = []
        for band in range(band_count):
            gaps = [aggregate[band]["gap"] for aggregate in aggregates]
            per_band.append({"band": band, "gaps": gaps, **lstsq_alpha(updates, gaps)})
        # The gate's own quantity: mean gap over the finest half of the bands.
        fine = [[row["gap"] for row in aggregate if row["band"] >= band_count // 2]
                for aggregate in aggregates]
        fine_mean = [float(np.mean(values)) for values in fine]
        result["lead_sets"][label] = {
            "per_band": per_band,
            "fine_band_mean_gaps": fine_mean,
            "fine_band_fit": lstsq_alpha(updates, fine_mean),
            "closure": updates_to_close(fine_mean[-1],
                                        lstsq_alpha(updates, fine_mean)["alpha"]),
        }
    return result


def c2_beta_reweighted(product: dict, betas: dict) -> dict:
    """What the gap looks like when variables are put on comparable footing.

    The Stage P product cannot answer this: ``band_powers`` sums over all 17 channels before
    reporting, so the per-variable split is already gone from the artifact. What *can* be said
    from the artifact is the **size** of the pooling distortion: the distribution of ``beta_j``
    across variables, and the resulting change in how much each variable would contribute.
    """
    channels = betas.get("channels", {})
    usable = {name: row["beta"] for name, row in channels.items()
              if _finite(row.get("beta")) and row["beta"] > 0}
    if len(usable) < 2:
        return {"available": False,
                "reason": "fewer than two usable beta_j values; cannot judge the pooling"}
    values = np.asarray(sorted(usable.values()))
    return {
        "available": True,
        "channels_with_beta": len(usable),
        "beta_min": float(values.min()),
        "beta_max": float(values.max()),
        "beta_spread_max_over_min": float(values.max() / values.min()),
        "unweighted_share_is_equal": 1.0 / len(usable),
        "weighted_share_max": float(values.max() / values.sum()),
        "weighted_share_min": float(values.min() / values.sum()),
        "interpretation": (
            "An unweighted 17-channel sum gives every variable 1/17 of the band power. With "
            "beta_j applied the contributions span the ratio above, so the Stage P gap is a "
            "mixture dominated by whichever variables carry the most AMSE. Whether the gap "
            "survives per-variable reweighting cannot be read from the Stage P artifact and "
            "is the open question Stage C leaves for Stage A."),
        "per_variable": {name: {"beta": beta,
                                "share_of_total": beta / values.sum()}
                         for name, beta in sorted(usable.items())},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=pathlib.Path, required=False)
    parser.add_argument("--product", type=pathlib.Path,
                        default=ROOT / P1_PRODUCT)
    parser.add_argument("--store", type=pathlib.Path, default=ROOT / STORE_REL)
    parser.add_argument("--beta-year", type=int, default=2001)
    parser.add_argument("--worker", default=None,
                        help="refused: Step C writes no protocol and no archive")
    args = parser.parse_args(argv)

    if args.worker is not None:
        raise SystemExit(
            f"Step C refuses --worker {args.worker!r}: a zero-GPU preflight writes no "
            "protocol.json and no code archive (decision 0110 / 0094 precedent).")

    record = {"scientific_claim": False, "test_read": False, "r": 0,
              "probe": "stage-c gap decay and beta reweighting",
              "cases": list(CASES)}
    try:
        product = json.loads(pathlib.Path(args.product).read_text(encoding="utf-8"))
        record["c1"] = c1_decay(product)
        import zarr
        store = zarr.open_group(str(args.store), mode="r")
        record["beta"] = beta_weights(store, year=args.beta_year)
        record["beta_physical_control"] = _physical_control(store, args.beta_year)
        record["c2"] = c2_beta_reweighted(product, record["beta"])
        record["gate"] = _c_gate(record)
    except Exception as exc:  # noqa: BLE001 - record, never raise, so the verdict is written
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback_tail"] = traceback.format_exc().strip().splitlines()[-8:]

    text = json.dumps(record, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0 if "error" not in record else 1


def _c_gate(record: dict) -> dict:
    """The pre-declared question: does Stage A have a motive left after C?

    ``open-a`` when the gap survives the artifact at a rate that training cannot plausibly
    close (small alpha over the reachable budget). ``close-line`` when alpha is large enough
    that more training would close it, in which case Stage P's ``skip-q`` was right after all
    and there is nothing for Stage A to fix.
    """
    closure = record.get("c1", {}).get("lead_sets", {}).get(
        "frozen_five_lead", {}).get("closure", {})
    alpha = closure.get("alpha")
    beta = record.get("c2", {})
    if alpha is None:
        return {"verdict": "unresolved",
                "reason": "the decay fit is not identifiable, so C1 cannot decide"}
    # The threshold is a reading of the extrapolation, not a tuned constant: an alpha below
    # 0.1 needs more than a 1000x training increase to halve the gap.
    halving_multiplier = 0.5 ** (-1.0 / alpha) if alpha > 0 else float("inf")
    verdict = "close-line" if halving_multiplier <= 1000.0 else "open-a"
    return {
        "verdict": verdict,
        "alpha": alpha,
        "updates_to_halve_gap": halving_multiplier,
        "beta_spread": beta.get("beta_spread_max_over_min"),
        "reason": (f"halving the gap needs {halving_multiplier:.3g}x more training; "
                   f"beta_j spans {beta.get('beta_spread_max_over_min')}x across variables"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
