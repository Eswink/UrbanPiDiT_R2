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

from training import r7_spectral_geometry as geometry  # noqa: E402 - after the path fix

#: The four training points, in increasing exposure. All four are single-arm, single-seed
#: runs on the same treatment manifest and the same recipe, so the only thing that differs
#: is how long they trained. Reusing the registered Stage B endpoints keeps this probe free
#: of any new training.
#:
#: The attempt directories are **not** uniform: the 2,008-update point predates the p-labelled
#: naming, and the 8,033 / 32,132 points live in the ``_20261104`` / ``_20261105`` attempts, not
#: in the ``p1000`` / ``p4000`` directories an earlier draft guessed at. Every path below was
#: checked to exist and its ``updates`` field to equal this table's value before being written
#: here; a wrong path would have made P1 report "no data" rather than a reading.
TRAINING_POINTS = (
    {"name": "p0025", "updates": 2008, "skill_72h": 0.098082,
     "checkpoint": "outputs/r7_s3_throughput_physics_b_20261104_attempt02/"
                   "seed41/training/update_0002008.pt"},
    {"name": "p1000", "updates": 8033, "skill_72h": 0.159293,
     "checkpoint": "outputs/r7_s3_throughput_physics_b_p1000_20261104_attempt01/"
                   "seed41/training/update_0008033.pt"},
    {"name": "p4000", "updates": 32132, "skill_72h": 0.230544,
     "checkpoint": "outputs/r7_s3_throughput_physics_b_p4000_20261105_attempt02/"
                   "seed41/training/update_0032132.pt"},
    {"name": "p1600", "updates": 128528, "skill_72h": 0.260668,
     "checkpoint": "outputs/r7_s3_throughput_physics_b_p1600_20261106_attempt01/"
                   "seed41/training/update_0128528.pt"},
)

#: The same eight train cases C1 and C2 used, so a P1 number can be held against them.
CASES = (1000, 5000, 9000, 13000, 17000, 21000, 25000, 29000)

#: The five leads the goal brief freezes for P1 (``4 endpoints x 8 cases x 17 channels x
#: 5 leads``). The rollout still produces all twelve 6-hourly steps because that is what the
#: registered recipe computes; the gate is evaluated on **these five** and the twelve-step
#: aggregate is reported beside it so the choice of lead set cannot hide a trend. An earlier
#: draft aggregated all twelve, which is a different question ("every lead") rather than the
#: frozen one.
FROZEN_LEADS = (6, 12, 24, 48, 72)

#: Wrong subcommands fail loudly instead of silently writing a half protocol. Stage P has no
#: frozen protocol and no archive by design (the C1 precedent: a preflight that decides not
#: to open a round produces no protocol at all), so a caller who reaches for the training
#: machinery must be stopped before it creates one.
REFUSED_WORKERS = ("prepare", "archive", "seed", "seeding", "reading")


#: The measurement primitives live in ``training/r7_spectral_geometry`` -- the same module a
#: Q round would call. They are imported rather than re-implemented here because a second copy
#: drifts: an earlier draft of this probe computed band powers with ``abs() ** 2``, which
#: differs from ``re^2 + im^2`` in about one sample in six and so broke the "``pred = target``
#: is exactly zero" property the J2/J4 argument rests on. Delegating makes that class of
#: divergence impossible, and ``tests/test_r7_spectral_geometry.py`` asserts the two names are
#: the same objects.
band_edges = geometry.band_edges
radial_band_index = geometry.radial_band_index
band_spectra = geometry.band_spectra
_hermitian_weights = geometry.hermitian_weights
_coarse_only = geometry.coarsen
_dct_energy = geometry.dct_energy


def parseval_error(field) -> float:
    """Relative energy difference between the spatial field and its ortho rfft2 spectrum.

    ``rfft2`` returns only the non-negative-frequency half, so a naive sum of ``|spectrum|^2``
    reports half the energy; the Hermitian weights restore it. Uses ``re^2 + im^2`` rather than
    ``abs() ** 2`` for the bitwise reason in the module docstring.
    """
    import torch

    data = field.to(torch.float64)
    spatial = float((data ** 2).sum())
    spectral = float(geometry._energy(data, geometry.hermitian_weights(data.shape[-1])))
    if spatial == 0.0:
        return 0.0 if spectral == 0.0 else float("inf")
    return abs(spatial - spectral) / spatial


def seam_energy_ratio(field) -> float:
    """The wrap discontinuity relative to a typical interior step (delegated to the module)."""
    return geometry.seam_energy_ratio(field)


def _field_from(tensor):
    """Reduce a ``[C, H, W]`` batch element to one channel-mean field for the P0 checks.

    Used for the synthetic control and for a single labelled channel only. A *mean across the
    17 store variables* is never fed to a real P0: they carry different units ``(K, m/s, Pa,
    m^2/s^2, kg/kg)``, so averaging them would invent a field no model ever sees.
    """
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

    # J2, the counter-example that must be visible: move all fine-band energy into the coarse
    # bands. A guard that cannot see this is the attempt07 failure mode (decision 0105).
    coarse_only = _coarse_only(field, edges)
    truth_spectrum = band_spectra(field, field, edges)
    # ``pred == target`` must be exactly zero, measured on the loss the Q round would use --
    # not on a band-gap proxy. The literal AMSE second term is nonzero here in about one
    # sample in six; the rationalised form in the module is exact.
    truth_loss = float(geometry.amse_analogue(field, field, edges=edges))
    moved_loss = float(geometry.amse_analogue(coarse_only, field, edges=edges))
    zero_at_truth = truth_loss == 0.0 and all(row["gap"] == 0.0
                                             for row in truth_spectrum["bands"])
    guard_sees_it = bool(moved_loss > truth_loss)

    return {
        "parseval_relative_error": parseval,
        "parseval_ok": parseval <= 1e-6,
        "seam_energy_ratio": seam,
        "seam_ratio_needs_no_threshold": True,
        "dct_energy": dct_energy,
        "dct_agrees_with_parseval": bool(dct_agrees),
        "zero_at_truth": bool(zero_at_truth),
        "amse_at_truth": truth_loss,
        "amse_after_coarsening": moved_loss,
        "guard_sees_coarse_shift": guard_sees_it,
        "band_edges": edges,
        "passes_hard_gate": bool(parseval <= 1e-6 and dct_agrees and zero_at_truth
                                 and guard_sees_it),
    }


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
    hard_gate_failed = False
    try:
        if args.stage in ("p0", "all"):
            record["p0"] = _run_p0(args.cases)
            hard_gate_failed = record["p0"].get("passes_hard_gate") is False
            if hard_gate_failed:
                record["stopped"] = "p0 hard gate failed; stage p1 not run"
        if not hard_gate_failed and args.stage in ("p1", "all"):
            record["p1"] = _run_p1(args.cases)
    except Exception as exc:  # noqa: BLE001 - record, never raise, so the verdict is written
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback_tail"] = traceback.format_exc().strip().splitlines()[-8:]
        hard_gate_failed = True

    text = json.dumps(record, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    # The exit code is the workflow's *hard gate*: a nonzero return stops P1/P2 before any GPU
    # is spent. Returning 0 on a failed P0 would have let the gate report "passed" while the
    # decomposition was illegal -- the same class of defect as a guard that cannot fail.
    return 0 if ("error" not in record and not hard_gate_failed) else 1


def _run_p0(cases) -> dict:
    """P0 on the real analysis field, with the synthetic field kept as a fallback control.

    The checks themselves are grid-level and deterministic, so they can run on a synthetic
    field; but the number that matters -- the **seam energy ratio** -- is a property of the
    real regional field, so the real frame is measured whenever the store is readable and the
    synthetic run is reported alongside it as a labelled control, never substituted for it.
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
    result["synthetic_field"] = {"seam_energy_ratio": result["seam_energy_ratio"],
                                 "parseval_relative_error": result["parseval_relative_error"]}
    real = _real_p0()
    if real is not None:
        result["real"] = real
        result["seam_energy_ratio"] = real["seam_energy_ratio"]
        result["parseval_relative_error"] = real["parseval_relative_error"]
        result["dct_agrees_with_parseval"] = real["dct_agrees_with_parseval"]
        result["passes_hard_gate"] = bool(
            real["parseval_ok"] and real["dct_agrees_with_parseval"]
            and real["zero_at_truth"] and real["guard_sees_coarse_shift"])
        result["synthetic"] = False
    result["real_field_measured"] = real is not None
    return result


def _real_p0() -> dict | None:
    """P0 on one real analysis frame, or ``None`` when the store is not readable."""
    import numpy as np
    import torch
    import zarr

    from scripts import r7_s3_curve_point_large_protocol as curve
    from training import r7_spectral_geometry as geometry

    if not curve.STORE.exists():
        return None
    store = zarr.open_group(str(curve.STORE), mode="r")
    channels = list(store.attrs["channels"])
    state = np.asarray(store["state"][:1], dtype=np.float64)  # [1, C, H, W]
    # One channel at a time: the seam is a property of a field, not of a channel mean, and
    # averaging 17 variables of different units would invent a field no model ever sees.
    rows = []
    for name in channels:
        field = torch.as_tensor(state[0, channels.index(name)], dtype=torch.float64)
        checks = p0_checks(field)
        rows.append({"channel": name,
                     "seam_energy_ratio": checks["seam_energy_ratio"],
                     "parseval_relative_error": checks["parseval_relative_error"],
                     "dct_agrees_with_parseval": checks["dct_agrees_with_parseval"],
                     "zero_at_truth": checks["zero_at_truth"],
                     "guard_sees_coarse_shift": checks["guard_sees_coarse_shift"]})
    worst_parseval = max(row["parseval_relative_error"] for row in rows)
    return {
        "channels": rows,
        "seam_energy_ratio": max(row["seam_energy_ratio"] for row in rows),
        "seam_energy_ratio_by_channel": {row["channel"]: row["seam_energy_ratio"]
                                        for row in rows},
        "parseval_relative_error": worst_parseval,
        "parseval_ok": worst_parseval <= 1e-6,
        "dct_agrees_with_parseval": all(row["dct_agrees_with_parseval"] for row in rows),
        "zero_at_truth": all(row["zero_at_truth"] for row in rows),
        "guard_sees_coarse_shift": all(row["guard_sees_coarse_shift"] for row in rows),
        "method": geometry.METHOD_NAME,
        "note": ("seam ratio is reported per channel and never thresholded (decision 0110); "
                 "the maximum is quoted only so one number exists to carry into limitations"),
    }


def _run_p1(cases) -> dict:
    """Amplitude versus coherence per band, per training point -- the measurement that can skip Q.

    Each point is rolled forward once per case from its own registered checkpoint, and every
    band of every lead is compared against that step's own target. The gap ``1 - amp_ratio``
    is then read as a function of training exposure: shrinking means the deficit is
    under-training (Q is not worth opening), flat or growing means it is the loss geometry
    (Q is). Reported per band and per point -- never pooled into a single number that could be
    made to look supportive.
    """
    import numpy as np
    import torch

    from scripts import r7_s3_curve_point_large_protocol as curve
    from training import r7_spectral_geometry as geometry

    if not curve.STORE.exists():
        raise FileNotFoundError(f"the store is not readable at {curve.STORE}")
    points = []
    for point in TRAINING_POINTS:
        path = ROOT / point["checkpoint"]
        if not path.exists():
            raise FileNotFoundError(f"{point['name']} checkpoint is missing: {path}")
        points.append(_measure_point(point, path, cases, curve, geometry, np, torch))
    return {
        "method": geometry.METHOD_NAME,
        "training_points": [point["name"] for point in TRAINING_POINTS],
        "cases": list(cases),
        "leads_hours": [6 * (step + 1) for step in range(len(curve.PHYSICAL_WEIGHTS))],
        "points": points,
        "gate": _p1_gate(points),
    }


def _train_dataset(curve, torch):
    """The registered train dataset, built once per process.

    Construction scans the manifest and validates every window's sha256 -- measured at ~187 s
    on this instance. P1 rolls four checkpoints over the same eight cases, so rebuilding it
    per checkpoint would spend ~12 minutes re-proving the same identity. The cache is keyed on
    the manifest path, and the identity arguments come from the registered C2 protocol, so a
    stale cache cannot silently describe a different split.
    """
    from data.r7_long_rollout_dataset import ZarrLongRolloutDataset

    cached = getattr(_train_dataset, "_cache", None)
    if cached is not None and cached[0] == str(curve.TRAIN_MANIFEST):
        return cached[1]
    protocol = ROOT / "outputs/r7_s3_derivative_matching_c2_20261108_attempt07/protocol.json"
    body = json.loads(protocol.read_text(encoding="utf-8"))
    dataset = ZarrLongRolloutDataset(
        curve.TRAIN_MANIFEST, physical_steps=len(curve.PHYSICAL_WEIGHTS),
        expected_exclusions=body["train_windows"]["excluded_sample_ids"],
        expected_window_sha256=body["train_windows"]["window_sha256"])
    _train_dataset._cache = (str(curve.TRAIN_MANIFEST), dataset)
    return dataset


def _measure_point(point, path, cases, curve, geometry, np, torch) -> dict:
    """One checkpoint rolled over every case; per-band amplitude ratio and coherence."""
    from training.r7_experiment import load_checkpoint, make_model
    from training.r7_long_rollout import training_long_rollout

    saved = load_checkpoint(path)
    if int(saved["updates"]) != int(point["updates"]):
        raise ValueError(f"{point['name']} holds {saved['updates']} updates, expected "
                         f"{point['updates']}")
    with torch.random.fork_rng(devices=[]):
        model = make_model("process", curve.model_spec())
    model.load_state_dict(saved["model"], strict=True)
    # ``training_long_rollout`` refuses an eval-mode/grad-off model ("differentiable training
    # requires grad-enabled model.train()"), so this runs in train mode with the graph alive and
    # detaches the forecast before measuring -- the same route C1's model half took
    # (``tools/reproduce_c1_model_comparison.py``). The equivalence to an eval forward is
    # checked rather than assumed: dropout must be 0.0 and no BatchNorm may exist in the trunk,
    # or train and eval would differ and the reading would not describe the deployed model.
    spec = curve.model_spec()
    if float(spec.get("dropout", -1.0)) != 0.0:
        raise ValueError(f"frozen spec dropout is {spec.get('dropout')!r}, not 0.0: a train-mode "
                         f"forward would not match the deployed one")
    normalising = [f"{name}: {type(module).__name__}"
                   for name, module in model.named_modules()
                   if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    if normalising:
        raise ValueError("the trunk has batch-dependent normalisation, so train and eval "
                         f"forwards differ: {normalising}")
    model.train()
    dataset = _train_dataset(curve, torch)
    edges = geometry.band_edges(65, 65)
    per_case = []
    for case in cases:
        batch = {key: value.unsqueeze(0) for key, value in dataset[case].items()
                 if torch.is_tensor(value)}
        result = training_long_rollout(model, batch, curve.REASONING_STEPS,
                                       physical_weights=curve.PHYSICAL_WEIGHTS,
                                       supervision_mask=None)
        forecasts = result.forecasts[0].detach()
        targets = batch["physical_targets"][0]
        leads = []
        for step in range(forecasts.shape[0]):
            rows = geometry.band_statistics(forecasts[step], targets[step], edges)
            leads.append({"lead_hours": 6 * (step + 1), "bands": rows})
        per_case.append({"case": case, "leads": leads})
    return {"name": point["name"], "updates": point["updates"],
            "skill_72h": point["skill_72h"],
            "model_code_sha256": saved.get("model_code_sha256"),
            # ``bands`` is the frozen five-lead aggregate the gate reads; ``bands_all_leads``
            # is every 6-hourly step, reported so the lead-set choice is visible rather than
            # hidden. The per-case rows keep all twelve steps so either aggregate can be
            # recomputed without a re-run.
            "bands": _band_aggregate(per_case, edges, leads=FROZEN_LEADS),
            "bands_all_leads": _band_aggregate(per_case, edges, leads=None),
            "cases": per_case}


def _band_aggregate(per_case, edges, leads=None) -> list[dict]:
    """Mean amplitude ratio and coherence per band over cases and the selected leads.

    Both are aggregated in the **power** domain (sum PSDs, then a ratio) rather than by
    averaging per-case ratios: a band whose target is nearly empty would otherwise contribute
    a wild ratio with the same weight as a well-populated one.

    ``leads`` selects which lead hours enter the aggregate; ``None`` means every lead. The
    frozen gate uses ``FROZEN_LEADS``, and the all-lead aggregate is reported alongside it.
    """
    import numpy as np

    count = len(edges) - 1
    rows = []
    for band in range(count):
        px, py, cr = [], [], []
        for case in per_case:
            for lead in case["leads"]:
                if leads is not None and lead["lead_hours"] not in leads:
                    continue
                row = lead["bands"][band]
                px.append(row["psd_pred"])
                py.append(row["psd_target"])
                cr.append(row["cross"])
        px, py, cr = np.asarray(px), np.asarray(py), np.asarray(cr)
        sx, sy, sn = px.sum(), py.sum(), cr.sum()
        ratio = float(np.sqrt(sx / sy)) if sy > 0 else float("nan")
        coh = float(sn / np.sqrt(sx * sy)) if sx > 0 and sy > 0 else float("nan")
        rows.append({"band": band, "amp_ratio": ratio, "coherence": coh,
                     "gap": 1.0 - ratio if ratio == ratio else float("nan")})
    return rows


def _p1_gate(points, *, frozen_only: bool = True) -> dict:
    """Does ``1 - amp_ratio`` shrink with training exposure? The pre-declared Q decision.

    The gate is evaluated on the bands of the **frozen five-lead** aggregate, because the goal
    brief fixes P1 as ``5 leads``; the twelve-step aggregate is reported beside it under
    ``all_leads`` so the lead-set choice cannot conceal a trend.

    ``monotone_decrease`` requires the finest bands' mean gap to fall at every rung of the
    exposure ladder. Anything else opens Q: the brief cancels Q only when the gap shrinks, so
    the asymmetry is deliberate -- this gate is the only mechanism allowed to save the GPU
    budget, and it must be hard to trip by accident while remaining able to trip.
    """
    import numpy as np

    def _fine_gaps(aggregate):
        gaps = [row["gap"] for row in aggregate
                if row["band"] >= len(aggregate) // 2 and row["gap"] == row["gap"]]
        return float(np.mean(gaps)) if gaps else float("nan")

    def _monotone(values) -> bool:
        finite = [value for value in values if value == value]
        return bool(len(finite) == len(values) and len(finite) >= 2
                    and all(later < earlier for earlier, later in zip(finite, finite[1:])))

    names = [point["name"] for point in points]
    frozen_fine = [_fine_gaps(point["bands"]) for point in points]
    all_fine = [_fine_gaps(point["bands_all_leads"]) for point in points]
    monotone = _monotone(frozen_fine)
    verdict = "skip-q" if monotone else "open-q"
    if not monotone and len(frozen_fine) > 1 and frozen_fine[-1] == frozen_fine[-1] \
            and frozen_fine[0] == frozen_fine[0] and frozen_fine[-1] < frozen_fine[0]:
        verdict = "mixed-trend"
    per_band_frozen = []
    for band in range(len(points[0]["bands"])):
        per_band_frozen.append({
            "band": band,
            "gap_by_point": dict(zip(names, [p["bands"][band]["gap"] for p in points])),
            "monotone_decrease": _monotone([p["bands"][band]["gap"] for p in points]),
        })
    return {
        "lead_set": list(FROZEN_LEADS),
        "fine_band_mean_gap_by_point": dict(zip(names, frozen_fine)),
        "monotone_decrease": monotone,
        "verdict": verdict,
        "all_leads": {
            "lead_set": [6 * (step + 1) for step in range(12)],
            "fine_band_mean_gap_by_point": dict(zip(names, all_fine)),
            "monotone_decrease": _monotone(all_fine),
        },
        "per_band_frozen_leads": per_band_frozen,
        "rule": ("gap shrinking with exposure => the deficit is under-training and Q is "
                 "skipped; flat or growing => it is the loss geometry and Q opens "
                 "(goal section 3.1); the gate reads the frozen five-lead aggregate"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
