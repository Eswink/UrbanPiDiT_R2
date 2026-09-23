from __future__ import annotations
from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass
class LossBreakdown:
    total: torch.Tensor
    forecast: torch.Tensor
    process: torch.Tensor
    router: torch.Tensor
    verifier: torch.Tensor
    compute: torch.Tensor
    diffusion: torch.Tensor


class R2Loss:
    """V6/R7 multi-objective loss with process deep supervision."""

    def __init__(
        self,
        forecast_weight=1.0,
        process_weight=0.1,
        router_weight=0.1,
        verifier_weight=0.1,
        compute_weight=0.01,
        diffusion_weight=0.05,
    ):
        self.fw=forecast_weight
        self.pw=process_weight
        self.rw=router_weight
        self.vw=verifier_weight
        self.cw=compute_weight
        self.dw=diffusion_weight

    def _verifier_target(self,batch,out):
        if 'verifier_target' in batch:
            return batch['verifier_target'].float().view_as(
                out.diagnostics.verifier_score
            ).clamp(0,1)
        err=(out.forecast.detach()-batch['urban_target']).abs().mean(
            dim=(1,2,3)
        )
        scale=batch['urban_target'].detach().flatten(1).std(
            dim=1
        ).clamp_min(1e-4)
        return torch.exp(-err/scale).clamp(0,1)

    def _process_loss(self,batch,out):
        if 'process_targets' not in batch:
            return out.forecast.new_zeros(())

        target=batch['process_targets']
        trace=out.diagnostics.trace
        if trace is not None and trace.process_predictions is not None:
            pred=trace.process_predictions
            n=min(target.shape[-1],pred.shape[-1])
            expanded=target[...,:n].unsqueeze(1).expand(
                -1,pred.shape[1],-1
            )
            return F.mse_loss(pred[...,:n],expanded)

        pred=out.diagnostics.process_predictions
        n=min(target.shape[-1],pred.shape[-1])
        return F.mse_loss(pred[...,:n],target[...,:n])

    def __call__(self,batch,out,diffusion_loss=None):
        lf=F.smooth_l1_loss(out.forecast,batch['urban_target'])
        lp=self._process_loss(batch,out)
        lr=out.forecast.new_zeros(())

        if 'zoom_target' in batch:
            lr=F.binary_cross_entropy(
                out.diagnostics.zoom_probability.clamp(1e-5,1-1e-5),
                batch['zoom_target'].float().view_as(
                    out.diagnostics.zoom_probability
                ),
            )

        vt=self._verifier_target(batch,out)
        lv=F.binary_cross_entropy(
            out.diagnostics.verifier_score.clamp(1e-5,1-1e-5),vt
        )
        lc=out.diagnostics.compute_cost.mean()
        ld=(
            out.forecast.new_zeros(())
            if diffusion_loss is None
            else diffusion_loss
        )
        total=(
            self.fw*lf+
            self.pw*lp+
            self.rw*lr+
            self.vw*lv+
            self.cw*lc+
            self.dw*ld
        )
        return LossBreakdown(total,lf,lp,lr,lv,lc,ld)
