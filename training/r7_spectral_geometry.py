"""The limited-area AMSE-analogue: a target-matched spectral-geometry loss on a 65x65 patch.

**What this is not.** It is *not* AMSE. AMSE (arXiv:2501.19374v2, eq. 6) is defined on a
spherical-harmonic decomposition; this repository's grid is a **regional patch**
(27.0-43.0 N / 107.0-123.0 E, 0.25 deg, ``[65, 65]`` -- ``docs/DATA_SPEC.md:36``), where a
spherical harmonic transform is not well defined and none of ``pyshtools``/``ducc0``/
``torch_harmonics`` is installed. Decision 0110 therefore fixes the approximation: an
``rfft2`` radial-band decomposition with a DCT-II cross-check, and the method is always
named *AMSE-analogue on a limited-area grid*.

**Why the geometry changes rather than another penalty being added.** Under a standard MSE
objective the optimal prediction amplitude is ``sigma_X = rho < 1`` (the paper's eq. 1-2):
minimising ``E[MSE] = sigma_X^2 + 1 - 2 sigma_X rho`` over ``sigma_X`` gives ``sigma_X = rho``,
so a too-flat forecast is an *optimal-solution property of the loss*, not under-training.
C1 (untargeted divergence) and C2 (a single derived-quantity match) failed because both left
that geometry intact: C1 charged the analysis field 10.2x more than the model (decision 0094),
C2 sat at dose 0.29-0.41 and fought the skill term (decision 0109). This module changes the
per-band weight instead, which is the one thing neither of those did.

**The form, per band k, on band-aggregated PSD and coherence:**

    term1_k = (sqrt(P_k(x)) - sqrt(P_k(y)))^2
    term2_k = 2 * max(P_k(x), P_k(y)) * (1 - coh_k(x, y))
    L        = mean_k (term1_k + term2_k)

The second term is AMSE's eq. (6) with the amplitude/coherence coupling broken: MSE's exact
form is ``2*sqrt(P_x P_y)*(1 - coh)``, and replacing ``sqrt(P_x P_y)`` by ``max(P_x, P_y)`` is
the whole modification. It is parameter-free, so there is no band cutoff or term weight to
fit, and it has the same Taylor expansion as MSE at ``x = y``, which is why it is not a new
family (J4).

**Two numerical decisions, both measured rather than assumed.**

1. **``|a|^2`` is not ``re^2 + im^2`` bitwise.** On this grid, for random complex spectra,
   ``torch.abs(z) ** 2`` and ``z.real ** 2 + z.imag ** 2`` disagree in about 1 sample in 6
   (measured 8291/50000). That difference propagates into ``P_x``, so a band built from
   ``abs()`` reports a nonzero "amplitude mismatch" for a field compared with itself.
2. **``2*M*(1 - coh)`` does not vanish at ``x = y``.** Even with consistent PSDs, the
   subtraction is catastrophic when ``coh`` rounds to just below 1: measured 8291/50000
   nonzero (4141/50000 with ``coh`` clamped to 1 from above). Rationalising the term as
   ``2*M*(P_x*P_y - cross^2) / (P_x*P_y*(1 + coh))`` makes it exactly zero, because
   ``cross`` and ``P_x`` are then computed by the *same* expression, so ``P_x*P_y - cross^2``
   is exactly zero when the two fields are bitwise equal. Measured 0/50000 in both fp64 and
   fp32.

Both matter because J4's family test is *"``pred = target`` scores exactly zero"*, and J2
requires a guard that can fail. A form that reports 1e-17 at the truth cannot distinguish
"the loss prefers the truth" from "the loss prefers nothing in particular".

**The seam is real and is only quantified, never thresholded.** ``rfft2`` assumes the patch
wraps; a regional field does not. Decision 0110 requires the seam energy ratio to be
*reported* and forbids a threshold on it: a limit chosen after seeing the number is a fitted
parameter. ``seam_energy_ratio`` measures the part of the spectrum that disappears when the
best-fit linear ramp is removed.

**Where the loss may be wired.** Never into ``training/r7_long_rollout_runner.py`` (597/600
lines, R-051). ``training/r7_objective_axes.py`` is the single dispatch point; a Q round adds
a ``spectral_geometry`` key there and calls ``spectral_geometry_term`` below. Nothing in this
module writes a file, opens a store or reads a target except through explicit arguments.
"""
from __future__ import annotations

import math

import torch

#: The declared band edges, in cycles per grid interval, of the radial decomposition. This is
#: a **choice** (decision 0110 consequence 2) and is frozen here so a Q round records the same
#: edges the P0/P1 probes measured; nothing scans them. ``None`` means "derive from the grid",
#: which is what the probes use; a frozen protocol should carry the explicit list.
FROZEN_BAND_EDGES = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)

#: Name used in every record. The point of the constant is that no document, log line or
#: protocol may call this "AMSE" (decision 0110).
METHOD_NAME = "AMSE-analogue on a limited-area grid"


def _frequencies(length: int) -> list[float]:
    """Absolute rfft frequency magnitudes in cycles per grid interval."""
    return [float(abs(index if index <= length // 2 else index - length))
            for index in range(length)]


def band_edges(height: int, width: int) -> list[float]:
    """Radial band edges whose last entry is the true corner Nyquist of this grid.

    The corner is computed from the frequency arrays rather than assumed, so the bands tile
    the whole spectrum exactly once. An odd-length axis has a genuine frequency in its last
    rfft column (there is no Nyquist bin), which a ``-1`` mapping would collapse.
    """
    if height < 2 or width < 2:
        raise ValueError("band_edges needs a grid of at least 2x2")
    corner = math.hypot(max(_frequencies(height)), max(_frequencies(width)))
    edges = [edge for edge in FROZEN_BAND_EDGES if edge < corner]
    edges.append(corner)
    return edges


def radial_band_index(height: int, width: int, edges: list[float]):
    """An integer ``[H, W//2+1]`` map from each rfft2 bin to its band, plus the band count."""
    ky = torch.tensor(_frequencies(height), dtype=torch.float64)[:, None]
    kx = torch.tensor(_frequencies(width), dtype=torch.float64)[None, : width // 2 + 1]
    radius = torch.sqrt(ky * ky + kx * kx)
    index = torch.zeros_like(radius, dtype=torch.long)
    for band in range(len(edges) - 1):
        low, high = edges[band], edges[band + 1]
        last = band == len(edges) - 2
        inside = (radius >= low) & ((radius <= high) if last else (radius < high))
        index[inside] = band
    return index, len(edges) - 1


def hermitian_weights(width: int):
    """Per-column multipliers that restore the conjugate partners ``rfft2`` omits.

    Columns ``1..width//2-1`` each stand for a conjugate pair and count twice; DC never
    doubles; the Nyquist column doubles only for an odd width, where it is a genuine
    frequency rather than its own partner.
    """
    weights = torch.full((width // 2 + 1,), 2.0, dtype=torch.float64)
    weights[0] = 1.0
    if width % 2 == 0:
        weights[-1] = 1.0
    return weights[None, :]


def band_powers(prediction, target, edges: list[float] | None = None):
    """Per-band PSD and cross-spectrum as **differentiable** tensors of shape ``[K]``.

    ``P`` is ``sum(re^2 + im^2)`` over the band, weights included -- deliberately *not*
    ``|a|^2``, for the bitwise reason in the module docstring. ``cross`` uses the same
    expression, so ``P_x*P_y - cross^2`` is exactly zero when the fields are equal.

    Kept separate from ``band_statistics`` because a training loss must retain its graph:
    ``float()`` on a tensor detaches it, and a detached loss never reaches the parameters.
    """
    if prediction.shape != target.shape:
        raise ValueError(f"prediction {tuple(prediction.shape)} and target "
                         f"{tuple(target.shape)} must have the same shape")
    if prediction.shape[-1] < 2 or prediction.shape[-2] < 2:
        raise ValueError("band_powers needs a grid of at least 2x2")
    height, width = prediction.shape[-2], prediction.shape[-1]
    edges = list(edges) if edges is not None else band_edges(height, width)
    index, count = radial_band_index(height, width, edges)
    dtype = torch.float64 if prediction.dtype == torch.float64 else torch.float32
    weights = hermitian_weights(width).to(dtype)
    spectrum_x = torch.fft.rfft2(prediction.to(dtype), norm="ortho")
    spectrum_y = torch.fft.rfft2(target.to(dtype), norm="ortho")
    power_x = (spectrum_x.real * spectrum_x.real
               + spectrum_x.imag * spectrum_x.imag) * weights
    power_y = (spectrum_y.real * spectrum_y.real
               + spectrum_y.imag * spectrum_y.imag) * weights
    cross = (spectrum_x.real * spectrum_y.real
             + spectrum_x.imag * spectrum_y.imag) * weights
    # ``rfft2`` broadcasts the ``[K, W//2+1]`` weights over any number of leading axes, while
    # ``index`` is only the trailing 2-D map. Flattening the leading axes away is what makes the
    # mask line up; leaving ``[..., H, W//2+1]`` flat against a 2-D index raised an
    # ``IndexError`` the moment a real ``[C, H, W]`` field (C=17) reached this function -- a
    # defect the 2-D synthetic tests could not see.
    leading = power_x.shape[:-2]
    power_x = power_x.reshape((-1,) + power_x.shape[-2:])
    power_y = power_y.reshape((-1,) + power_y.shape[-2:])
    cross = cross.reshape((-1,) + cross.shape[-2:])
    flat_index = index.reshape(1, -1).expand(power_x.reshape(-1, index.numel()).shape[0], -1)
    flat_x = power_x.reshape(-1, index.numel())
    flat_y = power_y.reshape(-1, index.numel())
    flat_cross = cross.reshape(-1, index.numel())
    zeros = torch.zeros((), dtype=dtype)
    px, py, cr = [], [], []
    for band in range(count):
        selected = flat_index == band
        # Sum over every batch position and every bin in the band: an [N, bins] mask, so the
        # per-band power is the total over the whole field, not per leading index.
        px.append(torch.where(selected, flat_x, torch.zeros_like(flat_x)).sum())
        py.append(torch.where(selected, flat_y, torch.zeros_like(flat_y)).sum())
        cr.append(torch.where(selected, flat_cross, torch.zeros_like(flat_cross)).sum())
    return torch.stack(px), torch.stack(py), torch.stack(cr), count


def band_statistics(prediction, target, edges: list[float] | None = None) -> list[dict]:
    """The same per-band numbers as JSON-safe floats, for reporting only (never training)."""
    px, py, cr, count = band_powers(prediction, target, edges)
    index, _ = radial_band_index(prediction.shape[-2], prediction.shape[-1],
                                 list(edges) if edges is not None
                                 else band_edges(prediction.shape[-2], prediction.shape[-1]))
    rows = []
    for band in range(count):
        rows.append({"band": band, "bins": int((index == band).sum()),
                     "psd_pred": float(px[band]), "psd_target": float(py[band]),
                     "cross": float(cr[band])})
    return rows


def band_spectra(prediction, target, edges: list[float] | None = None) -> dict:
    """Per-band amplitude ratio, coherence and gap, in the reporting shape P1 emits.

    ``gap = 1 - amp_ratio`` is the quantity whose trend across training exposure decides
    whether Stage Q opens. ``overshoot`` is reported separately because decision 0110
    consequence 5 flags the ``amp_ratio > 1`` branch as behaviour the paper does not discuss,
    so the two cases must be counted apart rather than pooled.
    """
    rows = band_statistics(prediction, target, edges)
    for row in rows:
        sx, sy, cross = row["psd_pred"], row["psd_target"], row["cross"]
        denominator = math.sqrt(sx * sy) if sx > 0.0 and sy > 0.0 else 0.0
        ratio = math.sqrt(sx / sy) if sy > 0.0 else float("nan")
        row["amp_ratio"] = ratio
        row["coherence"] = (cross / denominator) if denominator > 0.0 else float("nan")
        row["gap"] = 1.0 - ratio if ratio == ratio else float("nan")
        row["overshoot"] = bool(ratio == ratio and ratio > 1.0)
    used = list(edges) if edges is not None else band_edges(prediction.shape[-2],
                                                             prediction.shape[-1])
    return {"bands": rows, "band_count": len(rows), "edges": used}


def band_loss_terms(row: dict) -> dict:
    """The two AMSE-analogue terms for one already-reported band row, as floats.

    Reporting helper only: for a differentiable loss use ``amse_analogue``, which keeps the
    graph. ``float()`` here would detach anything a training step depends on.
    """
    px = torch.as_tensor(row["psd_pred"], dtype=torch.float64)
    py = torch.as_tensor(row["psd_target"], dtype=torch.float64)
    cr = torch.as_tensor(row["cross"], dtype=torch.float64)
    first, second = _terms(px, py, cr)
    return {"amplitude": float(first), "coherence": float(second),
            "total": float(first) + float(second)}


def _terms(px, py, cr):
    """``(term1, term2)`` from band powers, in the rationalised exact-zero form.

    ``term2`` is ``2*max(P_x,P_y)*(1 - coh)`` rationalised to
    ``2*M*(P_x*P_y - cross^2) / (P_x*P_y*(1 + coh))``. Algebraically identical, but exactly
    zero when the two fields are bitwise equal -- which the literal form is not, in roughly
    one sample in six (module docstring, decision 0110's J4 test).

    **The rationalised form is singular at ``coh = -1``** (perfect anti-phase), where the
    denominator ``P*Q*(1+coh)`` vanishes while the numerator ``P*Q - cross^2`` is a floating
    point residue rather than an exact zero. The single-bin DC band reaches this exactly, and
    the division then returns ``inf`` rather than the correct limit ``2*max(P,Q)*(1-coh) =
    4*max(P,Q)`` -- the *maximum* penalty, not a small one. Measured on a real 6 h pair of
    ``v10`` at store frame 1464: ``P=802.585``, ``Q=93.396``, ``cross=-273.785``, so
    ``coh=-1.0`` and the rationalised second term came back ``inf`` where the literal value is
    ``3210.34``. A ``where`` that zeroes the whole term at ``coh <= -1`` would be worse still:
    it would report perfect anti-phase as a perfect match.

    So the literal form is used wherever ``1 + coh`` is not safely positive, which is the only
    branch where it differs in value; everywhere else the rationalised form keeps the exact-zero
    property that decision 0110's family argument depends on.
    """
    amplitude = (torch.sqrt(px) - torch.sqrt(py)) ** 2
    product = px * py
    degeneracy = product - cr * cr
    # ``where`` rather than a Python branch: a scalar branch on a tensor value would either
    # detach the graph or break under vmap/compile. Empty bands get a zero second term.
    safe_product = torch.where(product > 0.0, product, torch.ones_like(product))
    coherence = cr / torch.sqrt(safe_product)
    denominator = safe_product * (1.0 + coherence)
    rationalised = 2.0 * torch.maximum(px, py) * degeneracy / torch.where(
        denominator > 0.0, denominator, torch.ones_like(denominator))
    literal = 2.0 * torch.maximum(px, py) * (1.0 - coherence)
    # Both branches are evaluated; pick per element. ``0/0`` or ``x/0`` in the discarded
    # branch produces inf/nan and would poison a gradient, so it is replaced first.
    second = torch.where(denominator > 0.0, rationalised, literal)
    second = torch.where(product > 0.0, second, torch.zeros_like(second))
    return amplitude, second


def amse_analogue(prediction, target, *, edges: list[float] | None = None):
    """The band-mean AMSE-analogue loss for a ``[..., H, W]`` pair, differentiable.

    Exactly zero when ``prediction`` and ``target`` are bitwise equal, by construction.
    Parameter-free: no band weight, no cutoff, no epsilon.
    """
    px, py, cr, count = band_powers(prediction, target, edges)
    first, second = _terms(px, py, cr)
    return (first + second).mean()


def spectral_geometry_term(result, moved, contract):
    """``training/r7_objective_axes.py``'s entry point for a ``spectral_geometry`` round.

    ``None`` when the axis is off, returned before anything is constructed, so the registered
    path keeps its arithmetic bit for bit -- the same discipline ``constraint_term`` and
    ``matching_term`` follow. When on, this is the **deployment pairing**: the rollout's
    forecast against that step's own ``physical_targets``, exactly as the skill metric pairs
    them. C2's failure mode was a term built from a *derived* quantity only; this one is the
    forecast field itself, reweighted across bands.
    """
    axis = contract["autoregression"].get("spectral_geometry")
    if axis is None:
        return None
    if axis != "amse-analogue":
        raise ValueError(f"spectral_geometry must be absent or 'amse-analogue', got {axis!r}")
    forecasts = result.forecasts
    targets = moved.get("physical_targets")
    if targets is None:
        raise ValueError("the spectral_geometry axis requires physical_targets in the batch")
    if forecasts.shape[:2] != targets.shape[:2]:
        raise ValueError(f"forecast shape {tuple(forecasts.shape)} does not pair with target "
                         f"shape {tuple(targets.shape)}; the deployment pairing is "
                         f"[B,N,...] against [B,N,...]")
    losses = [amse_analogue(forecasts[:, step], targets[:, step]) for step in range(forecasts.shape[1])]
    return torch.stack(losses).mean()


def seam_energy_ratio(field) -> float:
    """How much bigger the wrap discontinuity is than a typical interior step, squared.

    ``rfft2`` assumes the patch repeats: the right edge continues into the left one. A
    regional field does not, so the transform attributes to the field a step that is not
    there. This measures that step directly:

        seam_rms     = rms(f[:, 0] - f[:, -1], f[0, :] - f[-1, :])
        interior_rms = rms(diff along x, diff along y)
        ratio        = seam_rms^2 / (seam_rms^2 + interior_rms^2)

    **``0.5`` is the periodic baseline** -- a field whose wrap is no more of a jump than a
    typical interior step. Above 0.5 the wrap is inventing structure; below it the field is
    unusually smooth across the boundary. Reported per channel and **never thresholded**
    (decision 0110): a limit picked after seeing the number would be a fitted parameter.

    An earlier version of this function compared the field's spectrum to the spectrum of the
    field minus its best-fit *linear ramp*, which also removes the **mean**. On a channel with
    a large offset -- ``t2m`` at ~273 K, ``mslp`` at ~101635 Pa -- the ramp then carries
    essentially all the energy and the ratio reads ~0.9999 for every field including a smooth
    one. That is a measure of "how much of the field is its mean", not of the seam, so it was
    replaced rather than thresholded.
    """
    data = field.to(torch.float64)
    if data.ndim != 2:
        raise ValueError("seam_energy_ratio expects a [H, W] field")
    horizontal_wrap = (data[:, 0] - data[:, -1]) ** 2
    vertical_wrap = (data[0, :] - data[-1, :]) ** 2
    seam = 0.5 * (horizontal_wrap.mean() + vertical_wrap.mean())
    horizontal_step = (data[:, 1:] - data[:, :-1]) ** 2
    vertical_step = (data[1:, :] - data[:-1, :]) ** 2
    interior = 0.5 * (horizontal_step.mean() + vertical_step.mean())
    total = float(seam + interior)
    if total == 0.0:
        return 0.0
    return float(seam) / total


def _energy(data, weights) -> torch.Tensor:
    """Parseval energy of a real field from its ortho rfft2, with the Hermitian weights."""
    spectrum = torch.fft.rfft2(data, norm="ortho")
    return ((spectrum.real * spectrum.real + spectrum.imag * spectrum.imag) * weights).sum()


def coarsen(field, edges: list[float] | None = None):
    """Keep the two coarsest bands and drop the rest: the J2 counter-example field.

    This is the field a live guard must be able to reject (decision 0105): it carries all of
    the field's structure in the coarse bands and none in the fine ones, which is exactly the
    "smooth forecast" AMSE is supposed to penalise more. A guard that cannot see it is inert.
    """
    if field.ndim != 2:
        raise ValueError("coarsen expects a [H, W] field")
    edges = list(edges) if edges is not None else band_edges(field.shape[0], field.shape[1])
    spectrum = torch.fft.rfft2(field.to(torch.float64), norm="ortho")
    index, _ = radial_band_index(field.shape[0], field.shape[1], edges)
    keep = index <= 1
    filtered = torch.where(keep, spectrum, torch.zeros_like(spectrum))
    return torch.fft.irfft2(filtered, s=field.shape[-2:], norm="ortho")


def dct_energy(field) -> float:
    """Energy under an orthonormal DCT-II, the boundary cross-check of decision 0110.

    DCT assumes an even reflection at the edge where rfft2 assumes periodicity. The two
    transformations must agree on Parseval; if they disagreed the conclusion would be driven
    by the boundary assumption rather than by the field.
    """
    import numpy as np
    from scipy.fft import dctn

    data = np.asarray(field.detach().to("cpu", torch.float64).numpy(), dtype=np.float64)
    coefficients = dctn(data, type=2, norm="ortho")
    return float((coefficients ** 2).sum())
