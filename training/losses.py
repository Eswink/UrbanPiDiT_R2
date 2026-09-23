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
    """V6 多目标损失。

    Verifier 目标优先读取 batch['verifier_target']；若无，则用相对目标场标准差
    归一化的 sample-wise MAE 构造 soft confidence target：exp(-MAE/std)。
    该目标只用于校准“当前 forecast 是否足够可信”，不声明现实因果置信度。
    """
    def __init__(self,forecast_weight=1.0,process_weight=0.1,router_weight=0.1,verifier_weight=0.1,compute_weight=0.01,diffusion_weight=0.05):
        self.fw=forecast_weight; self.pw=process_weight; self.rw=router_weight; self.vw=verifier_weight; self.cw=compute_weight; self.dw=diffusion_weight

    def _verifier_target(self,batch,out):
        if 'verifier_target' in batch:
            return batch['verifier_target'].float().view_as(out.diagnostics.verifier_score).clamp(0,1)
        err=(out.forecast.detach()-batch['urban_target']).abs().mean(dim=(1,2,3))
        scale=batch['urban_target'].detach().flatten(1).std(dim=1).clamp_min(1e-4)
        return torch.exp(-err/scale).clamp(0,1)

    def __call__(self,batch,out,diffusion_loss=None):
        lf=F.smooth_l1_loss(out.forecast,batch['urban_target'])
        lp=out.forecast.new_zeros(()); lr=out.forecast.new_zeros(())
        if 'process_targets' in batch:
            n=min(batch['process_targets'].shape[-1],out.diagnostics.process_predictions.shape[-1])
            lp=F.mse_loss(out.diagnostics.process_predictions[...,:n],batch['process_targets'][...,:n])
        if 'zoom_target' in batch:
            lr=F.binary_cross_entropy(out.diagnostics.zoom_probability.clamp(1e-5,1-1e-5),batch['zoom_target'].float().view_as(out.diagnostics.zoom_probability))
        vt=self._verifier_target(batch,out)
        lv=F.binary_cross_entropy(out.diagnostics.verifier_score.clamp(1e-5,1-1e-5),vt)
        lc=out.diagnostics.compute_cost.mean(); ld=out.forecast.new_zeros(()) if diffusion_loss is None else diffusion_loss
        total=self.fw*lf+self.pw*lp+self.rw*lr+self.vw*lv+self.cw*lc+self.dw*ld
        return LossBreakdown(total,lf,lp,lr,lv,lc,ld)
