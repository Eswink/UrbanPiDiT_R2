"""Exact, dependency-pruned copies of reviewed active statistics; never import repo/archive code."""
from __future__ import annotations
from datetime import datetime, timedelta
import json
import math
from statistics import fmean

STATISTICS = ("mse", "climatology_mse", "acc_dot", "acc_forecast_energy", "acc_target_energy")


def number(value, *, nonnegative=False):
    if isinstance(value, bool):
        raise ValueError("boolean is not a measurement")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("finite measurement required") from exc
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError("finite nonnegative measurement required" if nonnegative else "finite measurement required")
    return result


def integer(value, *, minimum=0):
    measured = number(value)
    if measured != int(measured) or measured < minimum:
        raise ValueError("exact integer count required")
    return int(measured)


def close(actual, expected, name):
    if not math.isclose(number(actual), expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"same-case metric mismatch: {name}")


def case_key(init_time, valid_times, lead):
    if not isinstance(init_time, str) or not isinstance(valid_times, list) or len(valid_times) != 1:
        raise ValueError("one exact initialization/valid-time pair per lead required")
    try:
        init, valid = datetime.fromisoformat(init_time), datetime.fromisoformat(valid_times[0])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid initialization/valid time") from exc
    if valid - init != timedelta(hours=lead):
        raise ValueError("valid time does not equal initialization plus lead")
    return (init_time, tuple(valid_times))


def pooled_statistics(rows):
    """Pool sufficient statistics before sqrt/division; never average case ACC."""
    if not rows:
        raise ValueError("no cases to pool")
    values = {key: [number(row[key], nonnegative=key != "acc_dot") for row in rows]
              for key in STATISTICS}
    for row in rows:
        mse_value, climate = number(row["mse"], nonnegative=True), number(row["climatology_mse"], nonnegative=True)
        p, t, d = (number(row[key], nonnegative=key != "acc_dot")
                   for key in ("acc_forecast_energy", "acc_target_energy", "acc_dot"))
        scale = max(p, t, abs(d), mse_value, climate, 1e-12)
        if abs(climate - t) > 1e-9 * scale or abs(mse_value - (p + t - 2 * d)) > 1e-9 * scale:
            raise ValueError("physical MSE/ACC sufficient-statistic identity mismatch")
    mse, climatology = fmean(values["mse"]), fmean(values["climatology_mse"])
    dot = fmean(values["acc_dot"])
    forecast, target = fmean(values["acc_forecast_energy"]), fmean(values["acc_target_energy"])
    denominator = math.sqrt(forecast) * math.sqrt(target)
    if abs(dot) > denominator + 1e-9 * max(denominator, 1e-12):
        raise ValueError("anomaly dot product exceeds its Cauchy bound")
    skill = None if climatology == 0 else 1 - mse / climatology
    acc = None if denominator == 0 else max(-1., min(1., dot / denominator))
    return {"mse": mse, "rmse": math.sqrt(mse), "climatology_mse": climatology,
            "rmse_climatology": math.sqrt(climatology), "mse_skill": skill,
            "skill_status": "undefined_zero_climatology_mse" if skill is None else "defined",
            "pooled_acc": acc,
            "acc_status": "undefined_zero_anomaly_energy" if acc is None else "defined",
            "acc_dot": dot, "acc_forecast_energy": forecast, "acc_target_energy": target,
            "n_initializations": len(rows),
            "bad_case_counts": {
                "worse_than_climatology": sum(m > c for m, c in zip(values["mse"], values["climatology_mse"])),
                "zero_climatology_mse": sum(c == 0 for c in values["climatology_mse"]),
                "negative_acc_dot": sum(d < 0 for d in values["acc_dot"]),
                "zero_anomaly_energy": sum(p == 0 or t == 0 for p, t in
                                           zip(values["acc_forecast_energy"], values["acc_target_energy"]))}}


def _list(value):
    result = json.loads(value) if isinstance(value, str) else value
    if not isinstance(result, list):
        raise ValueError("explicit list required in metric artifact")
    return result


def _canonical_metric(row, data, *, pooled=False):
    variable = row["variable"]
    index = data["channels"].index(variable)
    std = number(data["normalization_std"][index], nonnegative=True)
    if std <= 0 or row["unit"] != data["units"][index]:
        raise ValueError("physical metric needs pinned positive training std and exact unit")
    if row["acc_statistic_units"] != "normalized_anomaly_squared":
        raise ValueError("unknown ACC statistic scale; no guessed physical conversion")
    n = integer(row["n_initializations"], minimum=1)
    factor = std ** 2 / n if pooled else std ** 2
    if not pooled and n != 1:
        raise ValueError("each initialization must have n_initializations=1")
    close(row["rmse"], number(row["mse"], nonnegative=True) ** .5, "rmse/MSE")
    close(row["rmse_climatology"], number(row["climatology_mse"], nonnegative=True) ** .5, "climatology RMSE/MSE")
    close(row["mse_climatology"], number(row["climatology_mse"], nonnegative=True), "climatology alias")
    canonical = {key: row[key] for key in ("region", "variable", "unit", "lead_hours", "rmse", "rmse_climatology", "mse_skill")}
    canonical["lead_hours"] = integer(row["lead_hours"], minimum=1)
    canonical.update(mse=number(row["mse"], nonnegative=True),
                     climatology_mse=number(row["climatology_mse"], nonnegative=True),
                     pooled_acc=row["acc"], n_initializations=n, acc_status=row["acc_status"],
                     skill_status="undefined_zero_climatology_mse" if row["skill_status"] ==
                     "undefined_zero_climatology_energy" else row["skill_status"])
    canonical.update({key: number(row[key], nonnegative=key != "acc_dot") * factor for key in STATISTICS[2:]})
    if not pooled:
        canonical.update(init_time=row["init_time"], valid_times=_list(row["valid_times"]))
    return canonical
