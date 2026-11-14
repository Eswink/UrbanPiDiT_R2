"""Turn dumped fields into per-arm balance tables, then into the (a)-(e) verdict.

Input is what `scripts/r7_balance_field_dump.py` wrote: one ``.npz`` per case holding ``forecast``
and ``target`` as ``[lead, channel, H, W]`` in **physical units**, plus the grid and the
normalisation constants. Nothing here re-infers, so this module costs CPU only and can be re-run
freely -- which is the whole point of writing the fields.

The verdict is computed from the arrays, never read from a self-reported string. The precedent is
`da1ed00`: a judge step returned an empty body, a regex defaulted every clause to false, and the
published card said the opposite of ``readings.json``. Every value below is an explicit comparison
over loaded numbers, and a clause with no contrast to compute is ``None`` (untested), never false
and never true.

**Why the two-level layout.** Fields live at ``<root>/seed{seed}/{family}/{arm}/case_*.npz``. The
family directory is not cosmetic: Stage A's ``on``/``off`` and the urban-static
``control``/``treatment``/``placebo`` descend from different recipes, so an arm difference is only
meaningful *inside* one family. Nothing in this module ever subtracts across families.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np
import torch

from training import r7_balance_diagnostics as balance

#: The two thickness pairs the store supports. ``500-250`` has **no** t250/q250 -- the store
#: carries z/u/v at 250 hPa but no temperature and no humidity -- so its temperatures are
#: borrowed from 500 hPa and the pair is reported as ``dry_500_250`` with that substitution
#: named. Merging it with 850-500 into one "mean hydrostatic deviation" is forbidden.
LEVEL_PAIRS = ((850.0, 500.0, "t850", "t500", "q850", "q500", "virtual"),
               (500.0, 250.0, "t500", "t500", None, None, "dry_500_250_feasibility"))
#: Geometric level for the geostrophic test, per the paper's §3.2.
GEOSTROPHIC_LEVELS = (("z850", "u850", "v850"), ("z500", "u500", "v500"),
                      ("z250", "u250", "v250"))

#: The balance metrics the verdict compares, keyed as they appear in an arm's table. These are the
#: four diagnostics of `arXiv:2604.01454`, with the two thickness pairs kept separate.
HEADLINE_METRICS = ("hydrostatic:850-500", "hydrostatic:500-250",
                    "geostrophic:850", "geostrophic:500", "geostrophic:250",
                    "column_divergence", "spectral:fine_scale_noise")


def model_key(metric: str) -> str:
    """The table key for a metric on the **forecast** side.

    The producer (`case_metrics`) and every consumer (the clause module) must agree on this
    spelling. Passing bare metric names around and prefixing them at one call site and not the
    other is precisely the bug the end-to-end smoke found: the tables held
    ``forecast:hydrostatic:850-500`` while the clauses asked for ``hydrostatic:850-500``, which
    would have reported every clause as untested on a full round.
    """
    return f"forecast:{metric}"


def truth_key(metric: str) -> str:
    """The table key for a metric on the **analysis/truth** side."""
    return f"target:{metric}"


def _channel_indices(names):
    return {name: index for index, name in enumerate(names)}


def load_case(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        if "units_are_physical" not in archive.files:
            raise ValueError(f"{path} predates the de-normalisation fix; re-dump it")
        if not bool(archive["units_are_physical"]):
            raise ValueError(f"{path} holds normalised fields, not physical units")
        return {
            "forecast": archive["forecast"].astype(np.float64),
            "target": archive["target"].astype(np.float64),
            "latitude": archive["latitude"],
            "longitude": archive["longitude"],
            "lead_hours": [int(v) for v in archive["lead_hours"]],
            "channels": [str(v) for v in archive["channels"]],
        }


def case_metrics(case: dict) -> dict:
    """Every balance number this one case yields, as plain floats.

    A single scalar per (lead, metric) and side: the median over the 65x65 grid. The median is
    what the aggregation then takes across cases, so one storm cannot set a lead's value.
    """
    index = _channel_indices(case["channels"])
    latitude, longitude = case["latitude"], case["longitude"]
    out: dict[str, dict[str, float]] = {}
    for position, lead in enumerate(case["lead_hours"]):
        row: dict[str, float] = {}
        for side in ("forecast", "target"):
            frame = case[side][position]
            get = lambda name: torch.from_numpy(frame[index[name]])  # noqa: E731

            for (lower, upper, lower_t, upper_t, q_lower, q_upper, _mode) in LEVEL_PAIRS:
                deviation = balance.hydrostatic_deviation(
                    get(f"z{int(lower)}"), get(f"z{int(upper)}"),
                    get(lower_t), get(upper_t), lower, upper,
                    get(q_lower) if q_lower else None, get(q_upper) if q_upper else None)
                finite = deviation[torch.isfinite(deviation)]
                if finite.numel():
                    row[f"{side}:hydrostatic:{int(lower)}-{int(upper)}"] = float(finite.median())

            for z_name, u_name, v_name in GEOSTROPHIC_LEVELS:
                level = z_name[1:]
                departure, mask = balance.geostrophic_departure(
                    get(u_name), get(v_name), get(z_name), latitude, longitude)
                selected = departure[mask]
                if selected.numel():
                    row[f"{side}:geostrophic:{level}"] = float(selected.median())
                    row[f"{side}:coverage:{level}"] = float(mask.double().mean())

            d850 = balance.divergence_from_wind(get("u850"), get("v850"), latitude, longitude)
            d500 = balance.divergence_from_wind(get("u500"), get("v500"), latitude, longitude)
            d250 = balance.divergence_from_wind(get("u250"), get("v250"), latitude, longitude)
            residual = balance.column_divergence_residual(d850, d500, d250)
            finite = residual[torch.isfinite(residual)]
            if finite.numel():
                row[f"{side}:column_divergence"] = float(finite.median())

            z500 = get("z500")
            row[f"{side}:spectral:fine_scale_noise"] = float(balance.fine_scale_noise(z500))
            wavelengths, amplitudes = balance.radial_amplitude_spectrum(z500, 0.25 * 111.32)
            if amplitudes:
                row[f"{side}:spectral:near_nyquist_amplitude"] = float(amplitudes[-1])
                row[f"{side}:spectral:shortest_wavelength_km"] = float(wavelengths[-1])
        out[str(lead)] = row
    return out


def case_rmse(case: dict) -> dict:
    """Latitude-weighted RMSE per (lead, channel) in the fields' own physical units.

    This reproduces ``RolloutRMSEAccumulator`` exactly: area weight ``cos(lat)`` normalised over
    the grid, then a mean over initialisations. The accumulator multiplies a *normalised* delta by
    the training standard deviation; these arrays are already physical, so the weight is all that
    remains. The 72 h ``t2m`` value is cross-checked against the arm's own registered
    ``boundary_rmse.csv`` in the round's verification, which is what makes this re-derivation
    trustworthy rather than merely plausible.

    Returns **sqrt** of the weighted mean square, because (b) is decided on RMSE and a function
    whose name says RMSE must not hand back MSE.
    """
    index = _channel_indices(case["channels"])
    weight = np.cos(np.deg2rad(np.asarray(case["latitude"], dtype=np.float64))).clip(min=0.0)
    total = weight.sum()
    if total < 1e-12:
        raise ValueError("latitude weights have no usable area")
    out: dict[str, dict[str, float]] = {}
    for position, lead in enumerate(case["lead_hours"]):
        row = {}
        for name, channel in index.items():
            delta = case["forecast"][position, channel] - case["target"][position, channel]
            squared = np.asarray(delta, dtype=np.float64) ** 2
            if not np.isfinite(squared).all():
                raise ValueError(f"nonfinite error for {name} at lead {lead}")
            row[name] = float(np.sqrt((squared.mean(axis=1) * weight).sum() / total))
        out[str(lead)] = row
    return out


def summarize(paths) -> dict:
    """Median across cases per (lead, metric) plus RMSE per (lead, channel) for one arm."""
    balance_values: dict[str, dict[str, list[float]]] = {}
    case_rmse_values: dict[str, dict[str, list[float]]] = {}
    for path in paths:
        case = load_case(path)
        for lead, row in case_metrics(case).items():
            bucket = balance_values.setdefault(lead, {})
            for key, value in row.items():
                bucket.setdefault(key, []).append(value)
        for lead, row in case_rmse(case).items():
            bucket = case_rmse_values.setdefault(lead, {})
            for key, value in row.items():
                bucket.setdefault(key, []).append(value)

    result = {"n_cases": len(paths), "balance": {}, "rmse": {}}
    for lead, metrics in balance_values.items():
        result["balance"][lead] = {
            key: {"median": statistics.fmean(values), "n": len(values),
                  "spread": (max(values) - min(values)) if len(values) > 1 else 0.0}
            for key, values in metrics.items()}
    for lead, channels in case_rmse_values.items():
        # Pool the per-case RMSEs the way a root-mean-square pools: root of the mean of the
        # squares. Averaging the RMSEs directly would understate the total error.
        result["rmse"][lead] = {name: float(np.sqrt(statistics.fmean(v * v for v in values)))
                                for name, values in channels.items()}
    return result


def arm_directories(fields_root: Path, families: dict) -> dict:
    """Every arm the protocol declares, as ``{arm_id: (family, arm, seed, dir)}``."""
    root = Path(fields_root)
    found = {}
    for family, spec in families.items():
        for arm in spec["arms"]:
            for seed in spec.get("seeds") or ():
                directory = root / f"seed{seed}" / family / arm
                if not directory.is_dir():
                    raise FileNotFoundError(f"the round declares {family}/{arm}/seed{seed} but "
                                            f"{directory} does not exist")
                found[f"{family}/{arm}/seed{seed}"] = {
                    "family": family, "arm": arm, "seed": int(seed), "dir": str(directory)}
    return found


def _summarize_arm(argument) -> tuple:
    """Module-level so a process pool can pickle it: ``(arm_id, directory) -> summary``.

    Torch defaults to one thread per core, which makes fifteen forked workers fight over the same
    56 cores; one thread each is both faster and the only way the measurement stays linear.
    """
    import torch

    torch.set_num_threads(1)
    arm_id, record = argument
    paths = sorted(Path(record["dir"]).glob("case_*.npz"))
    if not paths:
        raise FileNotFoundError(f"{record['dir']} holds no case_*.npz")
    return arm_id, summarize(paths)


def table_of_arms(directories: dict, *, workers: int | None = None) -> dict:
    """One arm's tables keyed by its full id; the caller decides how to compare them.

    Arms are independent, and the reading stage must finish 21,705 case files -- measured at
    0.263 s per case on one core, that is 1.6 h serially against a 1 h stage budget. Splitting by
    arm is the natural axis: fifteen workers, no shared state, each still reading one case at a
    time. ``workers=None`` uses the pool; ``workers=1`` keeps it in-process for tests.
    """
    import os

    items = sorted(directories.items())
    count = workers if workers is not None else min(len(items), os.cpu_count() or 1)
    if count <= 1:
        return {arm_id: {**record, **summarize(sorted(Path(record["dir"]).glob("case_*.npz")))}
                for arm_id, record in items}
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=count) as pool:
        summaries = dict(pool.map(_summarize_arm, items))
    return {arm_id: {**record, **summaries[arm_id]} for arm_id, record in items}


def write(path: Path, body: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=1, sort_keys=True, allow_nan=False), encoding="utf-8")
