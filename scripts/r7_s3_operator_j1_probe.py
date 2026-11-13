"""Stage P2 probe: the J1 direction gate, measured before any round is opened.

J1 is the criterion C1 died on: **does the constraint charge the analysis field more than it
charges the model?** If it does, its descent direction points *away* from truth, and no amount
of tuning rescues it -- that is why C1 was vetoed pre-freeze at 0 GPU-h rather than run
(decision 0094). C1 lost 8/8 cases at 6.5-15.5x. Every remaining equation-type candidate owes
the same measurement before it may be opened.

Three candidates, measured identically on the same fields:

* **A -- hydrostatic thickness.** ``dZ = (R/g) * Tbar * ln(p1/p2)`` between two pressure
  levels. The coefficients are computed from the stated constants, not typed in, so a wrong
  ``R`` or ``g`` cannot hide. It reads the *relation between* z and T, which is a structure the
  model has never been asked to respect.
* **B -- column-integrated divergence.** ``|D_850 + D_500 + D_250|^2``. Expected to fail J1 for
  the same reason C1 did: the real atmosphere is not non-divergent (measured 2.5e-05 s^-1), so
  a term pushing divergence toward zero pushes away from truth. It is included to give the
  thesis its second pillar of falsification, not because it is expected to pass.
* **D -- moisture violations.** ``q >= 0`` and ``q <= q_sat(T, p)``. C1 measured ``q >= 0`` as a
  **dead term** (0.0 violations on both sides while carrying weight 0.5), so the honest prior
  is that it is dead here too; measuring it is cheap and settles the question.

**Units are checked before anything is scored.** C1's pre-freeze review found an
``s^-2``-vs-normalised-MSE collision that made its axis unable to act at all (gradient-norm
ratio down to 4.1e-10) -- a dose artefact that would have manufactured a false negative. The
``--units`` report below must be read before any penalty number is believed.

``scientific_claim: false``, ``test_read: false``, ``r = 0``. CPU only, no GPU-h.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Gas constant for dry air (J K^-1 kg^-1) and standard gravity (m s^-2). CODATA/AMS values.
R_DRY = 287.05
GRAVITY = 9.80665

#: Pressure levels in the store, in hPa. The two thickness pairs the candidate uses.
LEVEL_PAIRS = ((850.0, 500.0), (500.0, 250.0))

#: Stage P writes no protocol and no archive, so the training subcommands must be refused
#: before they can create one (the C1 precedent, decision 0094).
REFUSED_WORKERS = ("prepare", "archive", "seed", "seeding", "reading")

#: One registered train case: the same window C1 and C2 measured, so the direction
#: columns can be held against those results without a new split.
CASES = (1000, 5000, 9000, 13000, 17000, 21000, 25000, 29000)


def thickness_coefficient(lower_hpa: float, upper_hpa: float) -> float:
    """``(R/g) * ln(p_lower / p_upper)`` in metres per kelvin.

    Computed rather than quoted so a typo in either constant shows up as a wrong coefficient
    instead of a wrong conclusion. The plan quotes 15.53 and 20.29 m/K for the two pairs.
    """
    if not 0.0 < upper_hpa < lower_hpa:
        raise ValueError("thickness needs 0 < upper < lower pressure")
    return (R_DRY / GRAVITY) * math.log(lower_hpa / upper_hpa)


def hydrostatic_residual(z_lower, z_upper, t_lower, t_upper, lower_hpa, upper_hpa):
    """Difference between measured thickness and the thickness its mean temperature implies.

    ``z`` must be geopotential **height in metres** and ``t`` absolute temperature in kelvin.
    The store's ``z*`` channels are **geopotential** (``m**2 s**-2``), so every caller must
    divide by ``g`` first; ``check_units`` reports that.

    **The sign matters and was wrong in the first draft.** Thickness is
    ``z_upper - z_lower`` (the *height gained* going up), which is positive: about 4,110 m for
    850-500 hPa. Writing ``z_lower - z_upper`` gives -4,110 m, and comparing that against
    ``(R/g) ln(p1/p2) Tbar ~ 4,082 m`` produces an RMS residual of **8,193 m** instead of the
    physical **36.6 m** -- a 224x inflation that would have read as "the analysis badly
    violates hydrostatic balance" and manufactured a direction. Measured on this store,
    2026-11-11.
    """
    import torch

    coefficient = thickness_coefficient(lower_hpa, upper_hpa)
    measured = z_upper.to(torch.float64) - z_lower.to(torch.float64)
    mean_temperature = 0.5 * (t_lower.to(torch.float64) + t_upper.to(torch.float64))
    return measured - coefficient * mean_temperature


def column_divergence_residual(d850, d500, d250):
    """The vertical integral of divergence, in s^-1."""
    import torch

    return (d850.to(torch.float64) + d500.to(torch.float64) + d250.to(torch.float64))


def moisture_violations(q, temperature, pressure_hpa):
    """Negative specific humidity and supersaturation, both in kg/kg.

    Saturation is Tetens over water; the overshoot is reported in the same units so the two
    terms can be compared without a second conversion.
    """
    import torch

    q64 = q.to(torch.float64)
    t64 = temperature.to(torch.float64)
    negative = torch.clamp(-q64, min=0.0)
    # Tetens: e_s in hPa, then q_s = 0.622 * e_s / (p - 0.378 * e_s).
    e_s = 6.112 * torch.exp(17.67 * (t64 - 273.15) / (t64 - 29.65))
    q_sat = 0.622 * e_s / (pressure_hpa - 0.378 * e_s)
    supersaturated = torch.clamp(q64 - q_sat, min=0.0)
    return {"negative": negative, "supersaturated": supersaturated, "q_sat": q_sat}


def _mean_abs(tensor) -> float:
    return float(tensor.abs().mean())


def _penalty(tensor) -> float:
    """Mean square, the same reduction the training losses use."""
    return float((tensor ** 2).mean())


def check_units(channels: dict) -> dict:
    """Report the unit evidence before any penalty is believed.

    C1's ``s^-2`` collision is the reference failure: the axis was present, finite and
    plausible, and completely unable to move the parameters. These checks are about *scale*,
    not correctness of the data, and each carries the magnitude it expects so a wrong unit
    shows as an order-of-magnitude miss rather than as a silent zero.
    """
    import torch

    report = {}
    if "z850" in channels and "z500" in channels:
        thickness = (channels["z850"].to(torch.float64)
                     - channels["z500"].to(torch.float64))
        report["z_thickness_850_500_m"] = {
            "mean": float(thickness.mean()), "std": float(thickness.std()),
            "expected_order_metres": 5500.0,
            "looks_like_geopotential_height": bool(4000.0 < abs(float(thickness.mean())) < 7000.0),
        }
        # A geopotential (m^2 s^-2) field would be ~9.8x larger; catching that here is the
        # whole point of the check.
        report["z_850_absolute_m"] = {
            "mean": float(channels["z850"].to(torch.float64).mean()),
            "expected_order_metres": 1450.0,
            "looks_like_geopotential_height": bool(
                800.0 < abs(float(channels["z850"].to(torch.float64).mean())) < 2500.0),
        }
    if "t850" in channels:
        mean_t = float(channels["t850"].to(torch.float64).mean())
        report["t_850"] = {
            "mean": mean_t,
            "looks_like_kelvin": bool(180.0 < mean_t < 340.0),
            "looks_like_celsius": bool(-95.0 < mean_t < 65.0),
            "warning": ("a value in [-95, 65] would also sit inside the kelvin range's lower "
                        "tail; the store's declared unit must be checked against DATA_SPEC"),
        }
    if "q850" in channels:
        q = channels["q850"].to(torch.float64)
        report["q_850"] = {
            "min": float(q.min()), "max": float(q.max()), "mean": float(q.mean()),
            "looks_like_kg_per_kg": bool(-0.01 < float(q.min()) and float(q.max()) < 0.05),
        }
    if "u850" in channels and "v850" in channels:
        for name in ("u850", "v850"):
            value = channels[name].to(torch.float64)
            report[f"{name}_range"] = {"min": float(value.min()), "max": float(value.max()),
                                      "expected_order_ms": 50.0}
    return report


def j1_verdict(penalty_analysis: float, penalty_model: float, dose,
               dose_band=(1e-3, 1e-1)) -> dict:
    """The direction gate plus the dose gate, as one comparable row.

    ``penalty_analysis <= penalty_model`` is J1. ``dose`` in ``dose_band`` is J3. A candidate
    must pass **both** to be worth opening: C1 failed J1 by 10.2x, and C2 passed J1 but sat at
    dose 0.29-0.41, so far above the band that it fought the skill term for gradient.

    A term that is **zero on both sides** is reported as inert rather than as a pass. C1's
    humidity term was exactly that: 0.0 violations on the analysis and the model while
    carrying weight 0.5, i.e. zero gradient dressed as a constraint (decision 0094 §6.2).
    ``0.0 <= 0.0`` is arithmetically true and scientifically empty, so it must not open a
    direction.
    """
    inert = penalty_analysis == 0.0 and penalty_model == 0.0
    finite_dose = isinstance(dose, (int, float)) and dose == dose
    low, high = dose_band
    passes_j1 = (not inert) and penalty_analysis <= penalty_model
    passes_j3 = bool(finite_dose) and low <= dose <= high
    if penalty_model > 0.0:
        ratio = penalty_analysis / penalty_model
    elif penalty_analysis > 0.0:
        ratio = float("inf")
    else:
        ratio = None
    return {
        "penalty_analysis": penalty_analysis,
        "penalty_model": penalty_model,
        # Strict JSON has no ``Infinity``. An exactly-zero model penalty makes the ratio
        # unbounded; reporting the non-finite float would make the whole record unparseable
        # for any strict reader, and a probe whose output cannot be parsed is not evidence.
        # The fact is kept in ``ratio_unbounded`` so nothing is hidden by the null.
        "ratio_analysis_over_model": ratio if ratio is None or math.isfinite(ratio) else None,
        "ratio_unbounded": bool(ratio == float("inf")),
        "dose": dose if finite_dose else None,
        "inert_on_both_sides": bool(inert),
        "passes_j1": bool(passes_j1),
        "passes_j3": bool(passes_j3),
        "open_this_direction": bool(passes_j1 and passes_j3),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=pathlib.Path, required=False)
    parser.add_argument("--worker", default=None,
                        help="refused: Stage P2 has no prepare/archive/seed stage")
    args = parser.parse_args(argv)

    if args.worker is not None:
        if args.worker in REFUSED_WORKERS:
            raise SystemExit(
                f"Stage P2 refuses --worker {args.worker!r}: a pre-freeze probe writes no "
                "protocol.json and no code archive (decision 0110 / decision 0094 precedent).")
        raise SystemExit(f"unknown worker {args.worker!r}")

    record = {"scientific_claim": False, "test_read": False, "r": 0,
              "probe": "stage-p2 j1 direction gate",
              "coefficients_m_per_k": {
                  f"{int(low)}_{int(high)}": round(thickness_coefficient(low, high), 4)
                  for low, high in LEVEL_PAIRS}}
    try:
        record["units"] = _unit_report()
        cases = _real_fields()
        record["real_fields"] = {
            "measured": cases is not None,
            "analysis_side": "physical_targets[0, -1] denormalised (the +72 h analysis field)"
            if cases is not None else None,
            "model_side": "p1600 rollouts[0, -1] denormalised (the +72 h forecast)"
            if cases is not None else None,
            "cases": [entry["case"] for entry in cases] if cases is not None else None,
        }
        record["directions"] = _direction_rows(cases)
    except Exception as exc:  # noqa: BLE001 - record, never raise, so the verdict is written
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback_tail"] = traceback.format_exc().strip().splitlines()[-8:]

    text = json.dumps(record, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0 if "error" not in record else 1


def _unit_report() -> dict:
    """Unit evidence read from the **real** store, fallback to a synthetic control.

    The store's declared units are the thing being checked, so a synthetic atmosphere can only
    ever confirm that the checks are wired -- not that this instance is in the units the
    formulas assume. Two facts this instance actually contradicts must be reported as they are:

    * ``z850``/``z500``/``z250`` carry unit ``m**2 s**-2``: they are **geopotential**, not
      geopotential height. ``hydrostatic_residual`` needs metres, so every caller must divide
      by ``g``. Skipping that would scale the thickness by 9.81 and turn a balance check into
      noise -- the same class of unit slip as C1's ``s^-2`` collision (decision 0094).
    * ``t850``/``t500`` carry unit ``K`` and read ~274 K, so they are Kelvin, not Celsius.
    """
    import numpy as np
    import torch
    import zarr

    from scripts import r7_s3_curve_point_large_protocol as curve

    synthetic = check_units({
        "z850": torch.full((8, 8), 1500.0), "z500": torch.full((8, 8), 5800.0),
        "t850": torch.full((8, 8), 285.0), "q850": torch.full((8, 8), 0.008),
        "u850": torch.zeros((8, 8)), "v850": torch.zeros((8, 8)),
    })
    report = {"synthetic_control": synthetic, "source": "synthetic"}
    if not curve.STORE.exists():
        report["note"] = "the store is not readable; only the synthetic control ran"
        return report
    store = zarr.open_group(str(curve.STORE), mode="r")
    names = list(store.attrs["channels"])
    units = list(store.attrs["units"])
    frame = torch.as_tensor(np.asarray(store["state"][0], dtype=np.float64))
    fields = {name: frame[index] for index, name in enumerate(names)}
    real = check_units(fields)
    geopotential = [name for name, unit in zip(names, units) if "s**-2" in unit]
    report.update({
        "source": "store",
        "declared_units": dict(zip(names, units)),
        "checks": real,
        "z_is_geopotential_not_height": bool(geopotential),
        "geopotential_channels": geopotential,
        "height_conversion": ("z_height_m = z_geopotential / g; the store's z reads "
                              "~14649 m^2 s^-2 at 850 hPa => ~1493 m, consistent with the "
                              "standard atmosphere, whereas reading it as metres would say "
                              "1493*9.81 m"),
        "temperature_is_kelvin": bool(real.get("t_850", {}).get("looks_like_kelvin")),
        "verdict": ("units are consistent with the formulas once z is divided by g; and they "
                    "are NOT consistent if z is used directly as metres"),
    })
    return report


def _direction_rows(cases=None, edges=None) -> dict:
    """One row per candidate. Real fields when supplied, synthetic construction otherwise.

    The synthetic branch is kept as a labelled control: it constructs a truth that satisfies
    the relation and a smoother model-like field, so it proves the gate *can* report a
    direction. But a synthetic pass says nothing about this atmosphere, so ``source`` is
    always reported and the real measurement is what the verdict is read from.
    """
    import torch

    grid = (16, 16)
    rows = torch.arange(grid[0], dtype=torch.float64)[:, None]
    cols = torch.arange(grid[1], dtype=torch.float64)[None, :]
    rough = torch.sin(rows / 2.0) * torch.cos(cols / 2.0)
    smooth = torch.sin(rows / 8.0) * torch.cos(cols / 8.0)

    # A: an atmosphere in hydrostatic balance (residual identically 0) versus a smoothed one.
    # The truth is *constructed* to satisfy the relation with the same mean temperature the
    # residual uses, so its penalty is exactly zero; anything else would mean the residual and
    # the constructor disagree, which is the check this row is meant to be.
    coefficient = thickness_coefficient(850.0, 500.0)
    t850_truth = 285.0 + 4.0 * rough
    t500_truth = torch.full_like(rough, 275.0)
    z850_truth = 1500.0 + 3.0 * rough
    # Balanced: 500 hPa sits above 850 hPa by the thickness its mean temperature implies.
    z500_truth = z850_truth + coefficient * 0.5 * (t850_truth + t500_truth)
    a_truth = _penalty(hydrostatic_residual(z850_truth, z500_truth, t850_truth,
                                            t500_truth, 850.0, 500.0))
    # The model-like field smooths z and T independently, breaking the relation: this is the
    # "same information, less structure" case the gate is supposed to charge more.
    a_model = _penalty(hydrostatic_residual(
        1500.0 + 3.0 * smooth, 5800.0 + 3.0 * smooth,
        285.0 + 4.0 * smooth, torch.full_like(smooth, 275.0), 850.0, 500.0))

    # B: real atmosphere is not non-divergent, so a smoother (less divergent) model looks
    # "better" to the term while being further from truth.
    b_truth = _penalty(column_divergence_residual(6e-5 * rough, 4e-5 * rough, 2e-5 * rough))
    b_model = _penalty(column_divergence_residual(2e-5 * smooth, 1e-5 * smooth, 0.5e-5 * smooth))

    # D: moisture violations on a physical q field.
    moisture = moisture_violations(0.008 + 0.002 * rough, 285.0 + 4.0 * rough, 850.0)
    d_truth = _penalty(torch.clamp(-moisture["negative"], min=0.0))
    d_model = _penalty(moisture["negative"] * 0.0)

    synthetic = {
        "source": "synthetic",
        "A_hydrostatic": j1_verdict(a_truth, a_model, dose=None),
        "B_column_divergence": j1_verdict(b_truth, b_model, dose=None),
        "D_moisture_negative": j1_verdict(d_truth, d_model, dose=None),
        "note": ("synthetic control only: it proves the residual and the constructor agree, "
                 "which says nothing about this atmosphere"),
    }
    if cases is None:
        return synthetic
    return {"source": "store", "synthetic_control": synthetic,
            "real": _real_direction_rows(cases, edges=edges)}


def _real_direction_rows(cases, edges=None) -> dict:
    """J1 across every registered case: this store's analysis vs the p1600 model's forecast.

    Every direction is measured on **both** sides with one function and one set of units, so
    the only difference between the two columns is the field itself -- C1's comparison design,
    which is what made its 10.2x a property of the term rather than of the setup.

    Units: the store's ``z*`` channels are **geopotential** (``m**2 s**-2``), so both sides are
    divided by ``g`` before the hydrostatic residual, because that residual is in metres.
    ``t*`` is already Kelvin. Doing this on one side only would manufacture a direction.

    A direction passes only if it passes **in every case** (``all_cases``), which is J1's
    "逐例一致" requirement; the pooled means are reported alongside so a single outlier cannot
    hide behind an average.
    """
    import torch
    from model.r7_process_tensor_diagnostics import spherical_divergence

    records = []
    for entry in cases:
        analysis, prediction = entry["analysis"], entry["prediction"]
        latitude, longitude = analysis["latitude"], analysis["longitude"]

        def divergences(fields):
            out = {}
            for name, (u_channel, v_channel) in (
                    ("850", ("u850", "v850")), ("500", ("u500", "v500")),
                    ("250", ("u250", "v250"))):
                out[name] = spherical_divergence(fields[u_channel], fields[v_channel],
                                                 latitude, longitude)
            return out

        sides = {}
        for label, fields in (("analysis", analysis), ("model", prediction)):
            z_height = {name: fields[name] / GRAVITY for name in ("z850", "z500", "z250")}
            hydrostatic = _penalty(hydrostatic_residual(
                z_height["z850"], z_height["z500"], fields["t850"], fields["t500"],
                850.0, 500.0))
            div = divergences(fields)
            column = _penalty(column_divergence_residual(div["850"], div["500"], div["250"]))
            moisture = moisture_violations(fields["q850"], fields["t850"], 850.0)
            negative = _penalty(moisture["negative"])
            supersaturated = _penalty(moisture["supersaturated"])
            sides[label] = {"A_hydrostatic": hydrostatic,
                            "B_column_divergence": column,
                            "D_moisture_negative": negative + supersaturated,
                            "D_negative_only": negative,
                            "D_supersaturated_only": supersaturated}
        records.append({"case": entry["case"],
                        "analysis": sides["analysis"], "model": sides["model"]})

    rows = {}
    for name in ("A_hydrostatic", "B_column_divergence", "D_moisture_negative"):
        analysis_values = [record["analysis"][name] for record in records]
        model_values = [record["model"][name] for record in records]
        per_case = [j1_verdict(a, m, dose=None) for a, m in zip(analysis_values, model_values)]
        pooled = j1_verdict(sum(analysis_values) / len(analysis_values),
                            sum(model_values) / len(model_values), dose=None)
        pooled["passes_j1_all_cases"] = bool(all(row["passes_j1"] for row in per_case))
        pooled["cases_worse_than_model"] = int(sum(1 for row in per_case if not row["passes_j1"]))
        pooled["case_count"] = len(records)
        pooled["per_case_ratio"] = [row["ratio_analysis_over_model"] if
                                    row["ratio_analysis_over_model"] is None
                                    or math.isfinite(row["ratio_analysis_over_model"])
                                    else None for row in per_case]
        pooled["penalty_units"] = {
            "A_hydrostatic": "m^2 (residual in metres, then mean square)",
            "B_column_divergence": "s^-2 (sum of three divergences, then mean square)",
            "D_moisture_negative": "(kg/kg)^2 (both moisture violations)",
        }[name]
        # ``passes_j1`` on the pooled row is the direction at the mean; the J1 criterion is
        # "same sign in every case", so the all-cases field is what a decision should read.
        pooled["passes_j1"] = pooled["passes_j1_all_cases"]
        rows[name] = pooled
    rows["D_breakdown"] = {
        label: {"analysis": sum(r["analysis"][label] for r in records) / len(records),
                "model": sum(r["model"][label] for r in records) / len(records)}
        for label in ("D_negative_only", "D_supersaturated_only")}
    rows["dose_note"] = ("dose = penalty/L_mse needs the registered normalised MSE for these "
                         "exact cases; it is null here rather than approximated, because a "
                         "guessed denominator would make J3 unfalsifiable")
    return rows


def _real_fields() -> list[dict] | None:
    """This store's analysis and the p1600 model's +72 h forecast, for every registered case.

    Read-only. Returns ``None`` when the store or the endpoint is unavailable, so the probe
    reports a synthetic-only run instead of inventing a real measurement. One case is not
    enough for a direction: C1's 10.2x held 8/8, and a single case could be a lead-effect
    accident (see decision 0094's note on cross-lead pairing).
    """
    import numpy as np
    import torch
    import zarr

    from scripts import r7_s3_curve_point_large_protocol as curve
    from training.r7_experiment import load_checkpoint, make_model
    from training.r7_long_rollout import training_long_rollout

    checkpoint = ROOT / ("outputs/r7_s3_throughput_physics_b_p1600_20261106_attempt01/"
                         "seed41/training/update_0128528.pt")
    if not curve.STORE.exists() or not checkpoint.exists():
        # ``None``, not ``(None, None)``: the signature and the docstring both promise a single
        # value, and the caller tests ``cases is not None`` to decide whether the run measured
        # anything real. A 2-tuple is truthy, so the degradation path used to report
        # ``measured: True`` with an empty case list and then die on the list comprehension --
        # i.e. it failed to degrade *and* failed to say so. Found by a clean checkout, where the
        # store and the p1600 checkpoint are both absent (CI has been red on this since 0440993).
        return None
    store = zarr.open_group(str(curve.STORE), mode="r")
    names = list(store.attrs["channels"])
    latitude = torch.as_tensor(np.asarray(store["latitude"][:]), dtype=torch.float64)
    longitude = torch.as_tensor(np.asarray(store["longitude"][:]), dtype=torch.float64)
    saved = load_checkpoint(checkpoint)
    with torch.random.fork_rng(devices=[]):
        model = make_model("process", curve.model_spec())
    model.load_state_dict(saved["model"], strict=True)
    spec = curve.model_spec()
    if float(spec.get("dropout", -1.0)) != 0.0:
        raise ValueError("train-mode forward would differ from the deployed model: dropout != 0")
    model.train()
    import json as _json

    from data.r7_long_rollout_dataset import ZarrLongRolloutDataset
    protocol = ROOT / "outputs/r7_s3_derivative_matching_c2_20261108_attempt07/protocol.json"
    body = _json.loads(protocol.read_text(encoding="utf-8"))
    dataset = ZarrLongRolloutDataset(
        curve.TRAIN_MANIFEST, physical_steps=len(curve.PHYSICAL_WEIGHTS),
        expected_exclusions=body["train_windows"]["excluded_sample_ids"],
        expected_window_sha256=body["train_windows"]["window_sha256"])
    batches = []
    for case in CASES:
        batches.append((case, {key: value.unsqueeze(0) for key, value in dataset[case].items()
                               if torch.is_tensor(value)}))
    # The rollout gives **normalised** fields (the dataset normalises on read), while the
    # constraint formulas only make sense in physical units, so *both* sides are mapped back
    # with the store's own constants -- identically, so the comparison stays fair.
    #
    # Direction is checked, not assumed. ``store["state"]`` holds **physical** units
    # (``read_normalized`` divides by these constants at read time), so the map back is
    # ``physical = normalised * std + mean``. An earlier draft denormalised one side and not
    # the other and reported a hydrostatic penalty of 3.6e11 m^2 against a physical residual of
    # a few hundred metres -- a unit fault that would have read as a direction.
    mean = torch.as_tensor(np.asarray(store["normalization_mean"][:]), dtype=torch.float64)
    std = torch.as_tensor(np.asarray(store["normalization_std"][:]), dtype=torch.float64)
    shape = (len(names), 1, 1)

    def physical(fields):
        """``[C, H, W]`` normalised -> physical, in one call so the axes cannot drift.

        The shape has to be ``[C, H, W]``: applying the ``(C, 1, 1)`` constants to a bare
        ``[H, W]`` slice broadcasts into ``[C, H, W]``, and the resulting per-channel dict
        entry silently becomes a 17-channel blend whose mean is dominated by ``mslp`` (~101635)
        and ``z250`` (~104149). That is not hypothetical -- it produced a "t2m mean" of 16593
        and is what ``_assert_physical_units`` is there to catch.
        """
        return fields.to(torch.float64) * std.reshape(shape) + mean.reshape(shape)

    cases = []
    for case, batch in batches:
        result = training_long_rollout(model, batch, curve.REASONING_STEPS,
                                       physical_weights=curve.PHYSICAL_WEIGHTS,
                                       supervision_mask=None)
        analysis_fields = physical(batch["physical_targets"][0, -1])        # [C, H, W]
        prediction_fields = physical(result.forecasts[0, -1].detach())      # [C, H, W]
        analysis = {name: analysis_fields[index] for index, name in enumerate(names)}
        prediction = {name: prediction_fields[index] for index, name in enumerate(names)}
        _assert_physical_units(analysis)
        for fields in (analysis, prediction):
            fields["latitude"] = latitude
            fields["longitude"] = longitude
        cases.append({"case": case, "analysis": analysis, "prediction": prediction})
    return cases


def _assert_physical_units(analysis) -> None:
    """Refuse to score a field whose magnitudes contradict the store's declared units.

    Two facts a wrong normalisation direction breaks immediately:

    * ``t2m`` is Kelvin, so a 65x65 field must average a few hundred K. A double-denormalised
      field reads thousands; an un-denormalised (normalised) one reads near zero.
    * ``z850`` is geopotential (``m**2 s**-2``), so ``z/g`` must be a plausible 850 hPa height.
      Reading ~6.6e5 m means the normalisation was applied twice.

    Raises rather than warns: a warning in a probe log is how a units fault survives into a
    result page.
    """
    t2m_mean = float(analysis["t2m"].mean())
    if not 200.0 < t2m_mean < 340.0:
        raise ValueError(f"t2m averages {t2m_mean:.1f}, which is not Kelvin: the "
                         "normalisation direction is wrong or the store is not physical")
    height = float(analysis["z850"].mean()) / GRAVITY
    if not 200.0 < height < 3000.0:
        raise ValueError(f"z850/g is {height:.1f} m, not a plausible 850 hPa height: the "
                         "normalisation was applied twice, or z is not geopotential")


if __name__ == "__main__":
    raise SystemExit(main())
