"""Same-forecast archived metrics per case/region and CPU pooled sufficient statistics."""
from __future__ import annotations

import math

from r7_k3_reference_support import FORECASTS, REGIONS, finite_number, require

STAT_KEYS = ("mse", "climatology_mse", "anomaly_dot", "forecast_energy", "target_energy")


def _close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12 * max(1.0, abs(a), abs(b)))


def validate_stat(stat):
    require(set(stat) == set(STAT_KEYS) | {"normalized_acc", "std_squared"}, "exact physical/normalized statistic keys required")
    for key in STAT_KEYS:
        finite_number(stat[key], nonnegative=key != "anomaly_dot")
    scale = finite_number(stat["std_squared"], nonnegative=True)
    require(scale > 0, "positive physical std_squared required")
    normalized = stat["normalized_acc"]
    require(set(normalized) == {"dot", "forecast_energy", "target_energy"}, "normalized ACC dot/energy required")
    for source, dest in (("dot", "anomaly_dot"), ("forecast_energy", "forecast_energy"), ("target_energy", "target_energy")):
        value = finite_number(normalized[source], nonnegative=source != "dot")
        require(_close(value * scale, stat[dest]), "normalized ACC physical std-squared conversion differs")
    require(_close(stat["climatology_mse"], stat["target_energy"]), "climatology MSE/target energy identity differs")
    require(_close(stat["mse"], stat["forecast_energy"] + stat["target_energy"] - 2 * stat["anomaly_dot"]),
            "physical MSE/anomaly dot-energy identity differs")
    require(abs(stat["anomaly_dot"]) <= math.sqrt(stat["forecast_energy"]) * math.sqrt(stat["target_energy"]) +
            1e-9 * max(1.0, stat["forecast_energy"], stat["target_energy"]), "anomaly Cauchy bound violated")


def region_statistics(prediction, target, climate, latitude, *, lead, names, units, std):
    """Use old metric classes directly; flatten selected native cells to [...,N,1]."""
    import torch
    from training.r7_boundary_metrics import boundary_masks
    from training.r7_rollout_metrics import RolloutRMSEAccumulator
    from training.r7_acc import RolloutACCAccumulator
    from training.r7_climatology_skill import RolloutClimatologySkillAccumulator

    require(prediction.shape == target.shape == climate.shape and prediction.shape[:3] == (1, 1, len(names)),
            "single case/single lead/all-variable fields required")
    require(all(x.device.type == "cpu" for x in (prediction, target, climate)), "case metrics are CPU-only")
    h, w = prediction.shape[-2:]
    lat = torch.as_tensor(latitude, dtype=torch.float64)
    require(lat.shape == (h,), "one native latitude axis required")
    scales = torch.as_tensor(std, dtype=torch.float64).square()
    result = {}
    for region, margin, mask in boundary_masks(h, w, (2,)):
        def select(x):
            return x[..., mask].unsqueeze(-1)
        p, t, c = (select(x) for x in (prediction, target, climate))
        selected_lat = lat[:, None].expand(h, w)[mask]
        rmse = RolloutRMSEAccumulator((lead,), names, training_std=std, units=units)
        acc = RolloutACCAccumulator((lead,), names)
        skill = RolloutClimatologySkillAccumulator((lead,), names, training_std=std, units=units)
        rmse.update(p, t, selected_lat)
        acc.update(p, t, c, selected_lat)
        skill.update(p, t, c, selected_lat)
        require(rmse.initializations == acc.initializations == skill.initializations == 1, "single case metrics count differs")
        torch.testing.assert_close(rmse.sum_squared_error, skill.forecast_squared_error, rtol=1e-12, atol=1e-12)
        rows = []
        for j, name in enumerate(names):
            dot, pe, te = (float(x[0, j]) for x in (acc.dot, acc.forecast_energy, acc.target_energy))
            scale = float(scales[j])
            row = {"mse": float(rmse.sum_squared_error[0, j]),
                   "climatology_mse": float(skill.climatology_squared_error[0, j]),
                   "anomaly_dot": dot * scale, "forecast_energy": pe * scale, "target_energy": te * scale,
                   "std_squared": scale, "normalized_acc": {"dot": dot, "forecast_energy": pe, "target_energy": te}}
            validate_stat(row)
            rows.append(row)
        area = torch.cos(torch.deg2rad(lat)).clamp_min(0)[:, None].expand(h, w)
        result[region] = {"margin_cells": margin, "n_grid_points": int(mask.sum()),
                          "full_area_fraction": float(area[mask].sum() / area.sum()), "variables": rows}
    require(tuple(result) == REGIONS, "frozen complementary region order differs")
    return result


def score_case(sample, prediction, climate, *, lead, names, units, std, cumulative_steps, model_calls):
    target = sample["rollout_targets"].unsqueeze(0)
    persistence = sample["coarse_history"][-1].unsqueeze(0).unsqueeze(0)
    forecast_fields = dict(original_k3=prediction, persistence=persistence, climatology=climate)
    stats = {name: region_statistics(field, target, climate, sample["latitude"], lead=lead,
                                      names=names, units=units, std=std) for name, field in forecast_fields.items()}
    return {"init_time": sample["init_time"], "valid_times": list(sample["valid_times"]),
            "cumulative_reasoning_steps": cumulative_steps, "model_calls": model_calls, "statistics": stats}


def pool_cases(cases, *, seed, lead, names, units):
    require(cases and len(names) == len(units) and len(set(names)) == len(names), "nonempty exact case/variable/unit set required")
    identities = [(c["init_time"], tuple(c["valid_times"])) for c in cases]
    require(len(set(identities)) == len(cases), "duplicate cases cannot silently be pooled")
    rows = []
    for forecast in FORECASTS:
        for region in REGIONS:
            for j, name in enumerate(names):
                stats = []
                for case in cases:
                    require(set(case["statistics"]) == set(FORECASTS), "every case requires model/persistence/climatology")
                    require(tuple(case["statistics"][forecast]) == REGIONS, "every case requires all complementary regions")
                    values = case["statistics"][forecast][region]["variables"]
                    require(len(values) == len(names), "all variables required per case/region")
                    stat = values[j]
                    validate_stat(stat)
                    stats.append(stat)
                require(len({s["std_squared"] for s in stats}) == 1, "training normalization changed between cases")
                sums = {k: math.fsum(s[k] for s in stats) for k in STAT_KEYS}
                denominator = math.sqrt(sums["forecast_energy"]) * math.sqrt(sums["target_energy"])
                acc = max(-1.0, min(1.0, sums["anomaly_dot"] / denominator)) if denominator else None
                clim = sums["climatology_mse"]
                rows.append({"seed": seed, "lead_hours": lead, "forecast": forecast, "region": region,
                             "variable": name, "unit": units[j], "n_initializations": len(cases),
                             "mse": sums["mse"] / len(cases), "rmse": math.sqrt(sums["mse"] / len(cases)),
                             "climatology_mse": clim / len(cases), "rmse_climatology": math.sqrt(clim / len(cases)),
                             "mse_skill": 1.0 - sums["mse"] / clim if clim else None,
                             "pooled_acc": acc, "acc_status": "defined" if denominator else "undefined_zero_anomaly_energy",
                             "skill_status": "defined" if clim else "undefined_zero_climatology_energy",
                             "anomaly_dot_sum": sums["anomaly_dot"], "forecast_energy_sum": sums["forecast_energy"],
                             "target_energy_sum": sums["target_energy"]})
    return rows


def validate_evaluation(report, protocol, job):
    declared = protocol["data"]["evaluation_cases"][str(job["lead"])]
    cases = report["cases"]
    require([[c["init_time"], c["valid_times"]] for c in cases] == declared["cases"], "fixed single-lead cases differ")
    require(report["n_available_windows"] == declared["n_available"] and report["n_evaluated"] == len(cases), "case counts differ")
    require(report["channels"] == protocol["data"]["channels"] and report["units"] == protocol["data"]["units"], "all17 variable order/units differ")
    require(report["split"] == "val" and report["test_read"] is False and report["scientific_claim"] is False,
            "validation-only non-scientific reference required")
    require(all(c["model_calls"] == job["lead"] // 6 and c["cumulative_reasoning_steps"] == [3 * job["lead"] // 6] for c in cases),
            "exact old K3 physical/internal work required")
    expected = pool_cases(cases, seed=job["seed"], lead=job["lead"], names=report["channels"], units=report["units"])
    require(report["rows"] == expected, "pooled rows must be recomputed from every sufficient statistic")
    return expected
