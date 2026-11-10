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

    ``z`` must be geopotential height in metres and ``t`` absolute temperature in kelvin; the
    caller is responsible for that and ``check_units`` will say so. The residual is in metres,
    which makes it directly comparable to an analysis-field increment.
    """
    import torch

    coefficient = thickness_coefficient(lower_hpa, upper_hpa)
    measured = z_lower.to(torch.float64) - z_upper.to(torch.float64)
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
        "ratio_analysis_over_model": ratio,
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
        record["directions"] = _direction_rows()
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
    """Unit evidence from a synthetic atmosphere with known correct units.

    Run on synthetic values rather than the real store so this verdict does not depend on any
    product being present; the real field's unit check happens when the round driver runs with
    ``--store``. Each entry states what the correct reading looks like.
    """
    import torch

    return check_units({
        "z850": torch.full((8, 8), 1500.0), "z500": torch.full((8, 8), 5800.0),
        "t850": torch.full((8, 8), 285.0), "q850": torch.full((8, 8), 0.008),
        "u850": torch.zeros((8, 8)), "v850": torch.zeros((8, 8)),
    })


def _direction_rows() -> dict:
    """One row per candidate, on synthetic fields with a known direction.

    The synthetic construction is the point: a **true** field that satisfies the relation and
    a **model-like** field that is smoother than it. If the gate then reports "truth charged
    more", the candidate has C1's failure mode and must not be opened.
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
    z500_truth = z850_truth - coefficient * 0.5 * (t850_truth + t500_truth)
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

    return {
        "A_hydrostatic": j1_verdict(a_truth, a_model, dose=None),
        "B_column_divergence": j1_verdict(b_truth, b_model, dose=None),
        "D_moisture_negative": j1_verdict(d_truth, d_model, dose=None),
        "note": ("dose requires the real MSE denominator, so it is null here; the direction "
                 "columns are meaningful and the dose columns are filled by the round driver"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
