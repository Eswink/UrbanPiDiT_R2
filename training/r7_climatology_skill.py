"""Explicit climatology MSE/RMSE skill and its one-sided ACC identity (#62).

The library's pooled ACC (`training/r7_acc.py`) is the *uncentered* anomaly
correlation against one frozen climatology: anomalies are taken against that
same climatology and no further centering happens. On identical cases,
identical climatology and identical weights this satisfies a one-sided
identity with MSE skill:

    ACC  = dot / (||p|| * ||t||)
    skill = 1 - MSE_forecast / MSE_climatology
         = 1 - (||p||^2 + ||t||^2 - 2 dot) / ||t||^2        (pooled, weighted)

so **ACC < 0 forces skill < 0** (a negative ACC really does mean the forecast
is worse than that climatology on those cases), while **ACC > 0 implies
nothing** about the skill sign: a large forecast error orthogonal to the truth
anomaly can keep ACC positive while the skill is negative. This module adds
the explicit climatology RMSE / MSE-skill baseline so tables can carry both
quantities without mixing them, and records the identity as a test.

Premises (part of the contract, not optional): one external frozen
climatology shared by the anomalies and the skill reference; equal weight per
initialization with latitude-area means inside a field; skill is stated only
on exactly the same cases that were scored and is never averaged across
variables, across ACC definitions, or across case sets.
"""
from __future__ import annotations
import torch
from model.r7_rollout import validate_horizons


class RolloutClimatologySkillAccumulator:
    """Pooled latitude-weighted MSE of the forecast and of the frozen climatology.

    Reports, per lead/variable: forecast RMSE, climatology RMSE (the explicit
    baseline) and the MSE skill 1 - MSE_forecast / MSE_climatology. Zero-energy
    climatology targets leave the skill undefined (NaN) instead of inventing a
    value. No cross-variable aggregate is produced.
    """

    def __init__(self, lead_hours, variables):
        self.lead_hours = validate_horizons(lead_hours)
        self.variables = tuple(variables)
        if not self.variables or len(set(self.variables)) != len(self.variables):
            raise ValueError("unique nonempty variables required")
        shape = (len(self.lead_hours), len(self.variables))
        self.forecast_squared_error = torch.zeros(shape, dtype=torch.float64)
        self.climatology_squared_error = torch.zeros(shape, dtype=torch.float64)
        self.initializations = 0

    @torch.no_grad()
    def update(self, prediction, target, climatology, latitude):
        if (prediction.ndim != 5 or prediction.shape != target.shape
                or prediction.shape != climatology.shape or min(prediction.shape) < 1):
            raise ValueError("equal nonempty [B,L,C,H,W] fields required")
        b, l, c, h, w = prediction.shape
        if (l, c) != self.forecast_squared_error.shape \
                or not (prediction.device == target.device == climatology.device):
            raise ValueError("horizon/channel/device mismatch")
        if not all(x.is_floating_point() and torch.isfinite(x).all()
                   for x in (prediction, target, climatology)):
            raise ValueError("skill needs finite floating-point fields")
        lat = torch.as_tensor(latitude, device=prediction.device, dtype=torch.float64)
        if lat.shape == (h,):
            lat = lat[None, :].expand(b, h)
        if lat.shape != (b, h) or not torch.isfinite(lat).all() or (lat.abs() > 90).any():
            raise ValueError("invalid latitude")
        weight = torch.cos(torch.deg2rad(lat)).clamp_min(0)
        sums = weight.sum(-1)
        if (sums < 1e-12).any():
            raise ValueError("zero usable area")
        weight = weight / sums[:, None]

        def reduce(field):
            return (field.mean(-1) * weight[:, None, None, :]).sum(-1).sum(0).cpu()

        forecast_error = (prediction.double() - target.double())
        climatology_error = (target.double() - climatology.double())
        self.forecast_squared_error += reduce(forecast_error.square())
        self.climatology_squared_error += reduce(climatology_error.square())
        self.initializations += b

    def compute(self):
        if not self.initializations:
            raise RuntimeError("no initialization samples")
        mse_forecast = self.forecast_squared_error / self.initializations
        mse_climatology = self.climatology_squared_error / self.initializations
        skill = torch.full_like(mse_climatology, float("nan"))
        usable = mse_climatology > 0
        skill[usable] = 1.0 - mse_forecast[usable] / mse_climatology[usable]
        return {"rmse_forecast": mse_forecast.sqrt(),
                "rmse_climatology": mse_climatology.sqrt(),
                "mse_skill": skill}


def verify_acc_skill_consistency(acc, skill, *, tolerance=1e-9):
    """Check the one-sided identity between pooled uncentered ACC and MSE skill.

    Both tensors are [lead, variable] values computed on the same cases with
    the same frozen climatology and the same weights. Returns a report dict;
    a violation is any finite pair with negative ACC but non-negative skill.
    Positive ACC with negative skill is *allowed* by the identity and is
    reported as a count, never as an error.
    """
    acc = torch.as_tensor(acc, dtype=torch.float64)
    skill = torch.as_tensor(skill, dtype=torch.float64)
    if acc.shape != skill.shape:
        raise ValueError("acc/skill shape mismatch")
    finite = torch.isfinite(acc) & torch.isfinite(skill)
    violations = int((finite & (acc < -tolerance) & (skill >= -tolerance)).sum())
    allowed = int((finite & (acc >= -tolerance) & (skill < -tolerance)).sum())
    return {"consistent": violations == 0,
            "violations": violations,
            "positive_acc_negative_skill": allowed,
            "identity": "ACC < 0 implies MSE skill < 0; ACC > 0 implies nothing",
            "tolerance": float(tolerance)}
