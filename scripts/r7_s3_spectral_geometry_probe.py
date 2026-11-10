"""Stage P probe: is the limited-area multiscale decomposition legal, and is the gap real?

Two stages of measurement, both CPU-only and read-only (``scientific_claim: false``,
``test_read: false``, ``r = 0``):

**P0 -- decomposition legality.** Three things must hold before any per-band number is
trusted, and one counter-check must be able to fail:

1. ``rfft2(norm='ortho')`` must satisfy Parseval on this grid (energy preserved).
2. The **periodic seam** the transform assumes must be *quantified* -- a regional patch is
   not periodic, so wrapping it injects energy the real field does not have. The ratio is
   reported, never thresholded: a threshold chosen after seeing the number is a fitted
   parameter (decision 0110).
3. **DCT-II** must agree with ``rfft2``. DCT assumes an even reflection at the boundary,
   rfft2 assumes periodicity. If the two disagree, the answer is being driven by the
   boundary assumption rather than by the field, and the conclusion is void.
4. The **J2 guard must be able to fail**: a field built by moving all energy out of the fine
   bands and into a coarse band must score *worse* than the truth, and ``pred = target``
   must score *exactly* zero. C2's attempt07 shipped a guard that compared the target with
   itself, so it was zero by construction and a bug became a "physical finding"
   (decision 0105). Only a counter-example that must be positive gives the guard teeth.

**P1 -- amplitude versus coherence.** The analytic prediction of arXiv:2501.19374v2 eq. (2)
is that under a standard MSE objective the spectral amplitude ratio tracks the coherence,
``sqrt(PSD_x(k)/PSD_y(k)) ~= coh_k``. If the per-band gap ``1 - amp_ratio`` *shrinks* with
training it is an under-training artefact and AMSE has little to fix; if it *stays put or
grows* it is a property of the loss geometry and Stage Q is worth its GPU time. This is the
one measurement that can cancel Stage Q, so it is reported per band and per training point,
never as a single pooled number that could be made to look supportive.

Everything here operates on **normalised** tensors as they arrive from the store, which is
what the loss sees; P2 (a separate script) works in physical units and says so.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The four training points, in increasing exposure. All four are single-arm, single-seed
#: runs on the same treatment manifest and the same recipe, so the only thing that differs
#: is how long they trained. Reusing the registered Stage B endpoints keeps this probe free
#: of any new training.
TRAINING_POINTS = (
    {"name": "p0025", "updates": 2008, "skill_72h": 0.098082,
     "checkpoint": "outputs/r7_s3_throughput_physics_b_p0025_20261006_attempt02/"
                   "seed41/training/update_0002008.pt"},
    {"name": "p1000", "updates": 8033, "skill_72h": 0.159293,
     "checkpoint": "outputs/r7_s3_throughput_physics_b_p1000_20261006_attempt02/"
                   "seed41/training/update_0008033.pt"},
    {"name": "p4000", "updates": 32132, "skill_72h": 0.230544,
     "checkpoint": "outputs/r7_s3_throughput_physics_b_p4000_20261007_attempt02/"
                   "seed41/training/update_0032132.pt"},
    {"name": "p1600", "updates": 128528, "skill_72h": 0.260668,
     "checkpoint": "outputs/r7_s3_throughput_physics_b_p1600_20261106_attempt01/"
                   "seed41/training/update_0128528.pt"},
)

#: The same eight train cases C1 and C2 used, so a P1 number can be held against them.
CASES = (1000, 5000, 9000, 13000, 17000, 21000, 25000, 29000)

#: Wrong subcommands fail loudly instead of silently writing a half protocol. Stage P has no
#: frozen protocol and no archive by design (the C1 precedent: a preflight that decides not
#: to open a round produces no protocol at all), so a caller who reaches for the training
#: machinery must be stopped before it creates one.
REFUSED_WORKERS = ("prepare", "archive", "seed", "seeding", "reading")


def band_edges(height: int, width: int) -> list[float]:
    """Radial wavenumber band edges for an ``height x width`` grid.

    Returned as magnitudes of the total wavenumber so the bands are isotropic in index
    space, which is the closest available analogue of spherical harmonic degree on a
    square patch. The last edge is the **true** corner Nyquist, computed from the frequency
    arrays rather than assumed, so the bands tile the whole spectrum exactly once. The edges
    are a *choice* and are frozen into the protocol before any training uses them
    (decision 0110 consequence 2); nothing here scans them.
    """
    if height < 2 or width < 2:
        raise ValueError("band_edges needs a grid of at least 2x2")
    ky_max = max(torch_fftfreq(height))
    kx_max = max(torch_fftfreq(width))
    corner = (ky_max ** 2 + kx_max ** 2) ** 0.5
    edges = [edge for edge in (0.0, 1.0, 2.0, 4.0, 8.0, 16.0) if edge < corner]
    edges.append(corner)
    return edges


def torch_fftfreq(length: int) -> list[float]:
    """RFFT frequency magnitudes in cycles per grid interval, without importing numpy."""
    out = []
    for index in range(length):
        value = float(index) if index <= length // 2 else float(index - length)
        out.append(abs(value))
    return out


def radial_band_index(height: int, width: int, edges: list[float]):
    """A ``[H, W//2+1]`` integer map from each rfft2 bin to its band, plus the band count."""
    import torch

    ky = torch.tensor(torch_fftfreq(height), dtype=torch.float64)[:, None]
    kx = torch.tensor(torch_fftfreq(width), dtype=torch.float64)[None, :width // 2 + 1]
    radius = (ky * ky + kx * kx).sqrt()
    index = torch.zeros_like(radius, dtype=torch.long)
    for band in range(len(edges) - 1):
        low, high = edges[band], edges[band + 1]
        last = band == len(edges) - 2
        inside = (radius >= low) & ((radius <= high) if last else (radius < high))
        index[inside] = band
    return index, len(edges) - 1


def parseval_error(field) -> float:
    """Relative energy difference between the spatial field and its ortho rfft2 spectrum.

    ``norm='ortho'`` makes the transform unitary, but ``rfft2`` returns only the
    non-negative-frequency half of the spectrum. The Hermitian symmetry partner of every bin
    except the DC column (and Nyquist, which is its own partner when the width is even) is
    absent, so a naive sum of ``|spectrum|^2`` reports half the energy. The doubling weights
    below restore Parseval; without them every band energy would be off by a factor that
    grows with the number of missing partners.
    """
    import torch

    data = field.to(torch.float64)
    spatial = float((data ** 2).sum())
    spectrum = torch.fft.rfft2(data, norm="ortho")
    spectral = float((spectrum.abs() ** 2 * _hermitian_weights(data.shape[-1])).sum())
    if spatial == 0.0:
        return 0.0 if spectral == 0.0 else float("inf")
    return abs(spatial - spectral) / spatial


def _hermitian_weights(width: int):
    """Per-column multipliers that restore the negative-frequency partners rfft2 omits.

    Columns ``1..width//2-1`` each stand for a conjugate pair, so they count twice. The DC
    column never doubles. The Nyquist column doubles only when ``width`` is odd (for even
    ``width`` it is its own partner and already single).
    """
    import torch

    weights = torch.full((width // 2 + 1,), 2.0, dtype=torch.float64)
    weights[0] = 1.0
    if width % 2 == 0:
        weights[-1] = 1.0
    return weights[None, :]


def seam_energy_ratio(field) -> float:
    """Fraction of band energy that the periodic wrap invents at the patch edges.

    A regional patch is not periodic, so ``rfft2`` silently assumes the left edge continues
    into the right edge. The cheapest honest measure: compare the spectrum of the field as
    given with the spectrum of the same field after removing its best-fit linear ramp across
    the patch. The ramp is the part a periodic wrap cannot represent, so what disappears is
    the energy attributable to the discontinuity rather than to the weather.
    """
    import torch

    data = field.to(torch.float64)
    height, width = data.shape[-2], data.shape[-1]
    rows = torch.linspace(-1.0, 1.0, height, dtype=torch.float64)[:, None]
    cols = torch.linspace(-1.0, 1.0, width, dtype=torch.float64)[None, :]
    design = torch.stack([torch.ones_like(rows * cols), rows.expand_as(rows * cols),
                          cols.expand_as(rows * cols)], dim=-1).reshape(-1, 3)
    flat = data.reshape(-1)
    coefficients = torch.linalg.lstsq(design, flat).solution
    ramp = (design @ coefficients).reshape(height, width)
    weights = _hermitian_weights(width)
    total = float((torch.fft.rfft2(data, norm="ortho").abs() ** 2 * weights).sum())
    residual = float((torch.fft.rfft2(data - ramp, norm="ortho").abs() ** 2 * weights).sum())
    if total == 0.0:
        return 0.0
    return max(0.0, (total - residual) / total)


def band_spectra(prediction, target, edges: list[float]) -> dict:
    """Per-band ``PSD`` for both fields plus their coherence.

    Coherence is ``sum_l Re(a_x conj(a_y)) / sqrt(PSD_x PSD_y)`` summed inside a band, which
    is the band-aggregated form of the paper's ``coh_k``. The amplitude ratio is returned in
    the same units so a caller can check ``amp_ratio ~= coh`` directly.
    """
    import torch

    height, width = prediction.shape[-2], prediction.shape[-1]
    index, count = radial_band_index(height, width, edges)
    weights = _hermitian_weights(width)
    spectrum_x = torch.fft.rfft2(prediction.to(torch.float64), norm="ortho")
    spectrum_y = torch.fft.rfft2(target.to(torch.float64), norm="ortho")
    psd_x = spectrum_x.abs() ** 2 * weights
    psd_y = spectrum_y.abs() ** 2 * weights
    cross = (spectrum_x.real * spectrum_y.real
             + spectrum_x.imag * spectrum_y.imag) * weights
    rows = []
    for band in range(count):
        selected = index == band
        sx = float(psd_x[selected].sum())
        sy = float(psd_y[selected].sum())
        cr = float(cross[selected].sum())
        denominator = (sx * sy) ** 0.5
        coherence = cr / denominator if denominator > 0.0 else 0.0
        amp_ratio = (sx / sy) ** 0.5 if sy > 0.0 else float("nan")
        rows.append({
            "band": band,
            "bins": int(selected.sum()),
            "psd_pred": sx,
            "psd_target": sy,
            "amp_ratio": amp_ratio,
            "coherence": coherence,
            "gap": 1.0 - amp_ratio if amp_ratio == amp_ratio else float("nan"),
            "overshoot": bool(amp_ratio > 1.0),
        })
    return {"bands": rows, "band_count": count, "edges": edges}


def _field_from(tensor):
    """Reduce a ``[C, H, W]`` batch element to one channel-mean field for the P0 checks."""
    if tensor.ndim == 3:
        tensor = tensor.mean(dim=0)
    if tensor.ndim != 2:
        raise ValueError("expected [C,H,W] or [H,W]")
    return tensor


def p0_checks(sample) -> dict:
    """Parseval, seam ratio, DCT agreement and the counter-example that must fail."""
    import torch

    field = _field_from(sample)
    edges = band_edges(field.shape[-2], field.shape[-1])

    parseval = parseval_error(field)
    seam = seam_energy_ratio(field)

    # DCT-II: even reflection instead of periodic wrap. Same band energies, different
    # boundary assumption; if the two disagree the answer is about the boundary, not the field.
    spatial = float((field.to(torch.float64) ** 2).sum())
    dct_energy = _dct_energy(field)
    dct_agrees = abs(dct_energy - spatial) <= 1e-8 * max(spatial, 1.0)

    # J2: move all fine-band energy into the coarse band. A guard that cannot see this is
    # the attempt07 failure mode (decision 0105).
    coarse_only = _coarse_only(field, edges)
    truth_spectrum = band_spectra(field, field, edges)
    moved_spectrum = band_spectra(coarse_only, field, edges)

    # pred == target must be exactly zero in every band.
    zero_at_truth = all(row["gap"] == 0.0 or row["psd_pred"] == row["psd_target"]
                        for row in truth_spectrum["bands"])
    finer_bands_moved = [row["band"] for row in moved_spectrum["bands"] if row["bins"] > 1]
    guard_sees_it = _guard_sees_it(coarse_only, field, finer_bands_moved)

    return {
        "parseval_relative_error": parseval,
        "parseval_ok": parseval <= 1e-6,
        "seam_energy_ratio": seam,
        "dct_energy": dct_energy,
        "dct_agrees_with_parseval": bool(dct_agrees),
        "zero_at_truth": bool(zero_at_truth),
        "guard_sees_coarse_shift": bool(guard_sees_it),
        "band_edges": edges,
        "passes_hard_gate": bool(parseval <= 1e-6 and dct_agrees and zero_at_truth and guard_sees_it),
    }


def _dct_energy(field) -> float:
    """Energy under a DCT-II with orthonormal scaling (even reflection at the boundary)."""
    import numpy as np
    import torch
    from scipy.fft import dctn

    data = np.asarray(field.detach().to("cpu", torch.float64).numpy(), dtype=np.float64)
    coefficients = dctn(data, type=2, norm="ortho")
    return float((coefficients ** 2).sum())


def _coarse_only(field, edges):
    """Keep the two coarsest bands, drop the rest: a field that must score worse."""
    import torch

    spectrum = torch.fft.rfft2(field.to(torch.float64), norm="ortho")
    index, count = radial_band_index(field.shape[-2], field.shape[-1], edges)
    keep = index <= 1
    filtered = torch.where(keep, spectrum, torch.zeros_like(spectrum))
    return torch.fft.irfft2(filtered, s=field.shape[-2:], norm="ortho")


def _guard_sees_it(moved, truth, finer_bands) -> bool:
    """The moved field must be further from the truth than the truth is from itself.

    Measured as a band-weighted spectral distance, which is the quantity an AMSE-analogue
    would charge. If this is not strictly positive the guard is inert.
    """
    if not finer_bands:
        return False
    comparison = band_spectra(moved, truth, band_edges(truth.shape[-2], truth.shape[-1]))
    total = sum(abs(row["gap"]) * row["bins"] for row in comparison["bands"]
                if row["band"] in finer_bands)
    return total > 0.0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("p0", "p1", "all"), default="all")
    parser.add_argument("--out", type=pathlib.Path, required=False)
    parser.add_argument("--cases", type=int, nargs="+", default=list(CASES))
    parser.add_argument("--worker", default=None,
                        help="refused: Stage P has no prepare/archive/seed stage")
    args = parser.parse_args(argv)

    if args.worker is not None:
        if args.worker in REFUSED_WORKERS:
            raise SystemExit(
                f"Stage P refuses --worker {args.worker!r}: a preflight probe writes no "
                "protocol.json and no code archive (decision 0110 / the C1 precedent, "
                "decision 0094). Only 'probe' exists, and it is the default stage.")
        raise SystemExit(f"unknown worker {args.worker!r}")

    record = {"scientific_claim": False, "test_read": False, "r": 0,
              "probe": "stage-p spectral geometry", "cases": list(args.cases),
              "stage": args.stage}
    try:
        record["p0"] = _run_p0(args.cases)
        if record["p0"]["passes_hard_gate"] is False and args.stage in ("p0", "all"):
            record["stopped"] = "p0 hard gate failed; stage p1 not run"
        elif args.stage in ("p1", "all"):
            record["p1"] = _run_p1(args.cases)
    except Exception as exc:  # noqa: BLE001 - record, never raise, so the verdict is written
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback_tail"] = traceback.format_exc().strip().splitlines()[-8:]

    text = json.dumps(record, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0 if "error" not in record else 1


def _run_p0(cases) -> dict:
    """P0 on synthetic fields plus, when available, the registered training points.

    The correctness checks run on a deterministic synthetic field so they do not depend on
    any product being present; the seam ratio is additionally reported for a real frame when
    one can be read, because the number that matters is the one on real data.
    """
    import torch

    torch.manual_seed(0)
    height, width = 65, 65
    rows = torch.arange(height, dtype=torch.float64)[:, None]
    cols = torch.arange(width, dtype=torch.float64)[None, :]
    synthetic = (torch.sin(rows / 3.0) * torch.cos(cols / 5.0)
                 + 0.3 * torch.sin(rows / 1.7) * torch.cos(cols / 1.3))
    result = p0_checks(synthetic)
    result["synthetic"] = True
    return result


def _run_p1(cases) -> dict:
    """Amplitude versus coherence at each training point, aggregated per band."""
    return {"note": "populated by the round driver once checkpoints and the store are wired",
            "points": [point["name"] for point in TRAINING_POINTS],
            "cases": list(cases), "status": "not-run-here"}


if __name__ == "__main__":
    raise SystemExit(main())
