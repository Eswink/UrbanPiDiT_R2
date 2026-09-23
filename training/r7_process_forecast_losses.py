from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping

import torch
from torch.nn import functional as F

from .r7_recursive_losses import deep_supervised_forecast_mse


@dataclass
class ProcessForecastLossBreakdown:
    total: torch.Tensor
    forecast: torch.Tensor
    process: torch.Tensor


def process_forecast_coreasoning_loss(
    batch:Mapping[str,torch.Tensor],
    out,
    *,
    forecast_weight:float=1.0,
    process_weight:float=0.1,
    final_weight:float=2.0,
)->ProcessForecastLossBreakdown:
    forecast=deep_supervised_forecast_mse(
        out.draft_forecasts,
        batch["atmos_target"],
        batch.get("latitude"),
        final_weight=final_weight,
    )

    process=forecast.new_zeros(())
    if (
        process_weight>0
        and "process_targets" in batch
        and out.process_predictions.shape[1]>0
    ):
        target=batch["process_targets"].to(
            device=out.process_predictions.device,
            dtype=out.process_predictions.dtype,
        )
        n=min(target.shape[-1],out.process_predictions.shape[-1])
        expanded=target[...,:n].unsqueeze(1).expand(
            -1,out.process_predictions.shape[1],-1
        )
        process=F.mse_loss(
            out.process_predictions[...,:n],
            expanded,
        )

    total=float(forecast_weight)*forecast+float(process_weight)*process
    return ProcessForecastLossBreakdown(
        total=total,
        forecast=forecast,
        process=process,
    )
