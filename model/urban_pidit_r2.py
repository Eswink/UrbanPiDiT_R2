from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
import torch
from torch import nn
from .coarse_encoder import CoarseEncoder
from .urban_static_encoder import UrbanStaticEncoder
from .process_state import ProcessStateInitializer
from .process_reasoner import RecurrentProcessReasoner
from .reasoning_router import ReasoningRouter
from .urban_expert import UrbanExpert
from .verifier import ForecastVerifier
from .residual_diffusion import ResidualDiffusionRefiner
from .state import R2Diagnostics, ReasoningTrace


@dataclass
class R2Output:
    forecast: torch.Tensor
    residual: torch.Tensor
    diagnostics: R2Diagnostics


class UrbanPiDiTR2(nn.Module):
    """UrbanPiDiT-R² V6/R7 transition: Reason → Route → Zoom → Verify."""

    def __init__(
        self,
        coarse_channels:int=12,
        coarse_history_steps:int=2,
        urban_channels:int=7,
        urban_history_steps:int=2,
        static_channels:int=6,
        out_channels:int=7,
        coarse_dim:int=192,
        process_dim:int=256,
        urban_dim:int=256,
        coarse_patch:int=2,
        urban_patch:int=4,
        coarse_depth:int=4,
        urban_depth:int=8,
        coarse_heads:int=6,
        process_heads:int=8,
        urban_heads:int=8,
        window_size:int=8,
        anchored_processes:int=12,
        free_processes:int=12,
        train_reasoning_steps:int=2,
        max_reasoning_steps:int=6,
        zoom_threshold:float=0.5,
        verifier_threshold:float=0.65,
        enable_diffusion:bool=False,
        diffusion_steps:int=4,
        dropout:float=0.0,
        activation_checkpointing:bool=False,
    ):
        super().__init__()
        self.train_reasoning_steps=train_reasoning_steps
        self.max_reasoning_steps=max_reasoning_steps
        self.zoom_threshold=zoom_threshold
        self.verifier_threshold=verifier_threshold
        self.enable_diffusion=enable_diffusion
        self.diffusion_steps=diffusion_steps

        self.coarse=CoarseEncoder(
            coarse_channels,coarse_history_steps,coarse_dim,coarse_patch,
            coarse_depth,coarse_heads,window_size,dropout,activation_checkpointing,
        )
        self.static_preview=UrbanStaticEncoder(
            static_channels,process_dim,patch_size=urban_patch
        )
        self.static_to_context=nn.Linear(process_dim,coarse_dim)
        self.process_init=ProcessStateInitializer(
            process_dim,anchored_processes,free_processes,process_heads,
            source_dim=coarse_dim,
        )
        self.reason_context=nn.Linear(coarse_dim,process_dim)
        self.reasoner=RecurrentProcessReasoner(
            process_dim,process_heads,3.0,dropout,activation_checkpointing
        )
        self.router=ReasoningRouter(process_dim,coarse_dim)
        self.urban=UrbanExpert(
            urban_channels,urban_history_steps,static_channels,out_channels,
            coarse_dim,process_dim,urban_dim,urban_patch,urban_depth,urban_heads,
            window_size,dropout,use_checkpointing=activation_checkpointing,
        )
        self.verifier=ForecastVerifier(process_dim,anchored_processes)
        self.diffusion=(
            ResidualDiffusionRefiner(out_channels) if enable_diffusion else None
        )
        self.anchored_processes=anchored_processes

    def _cheap_context(self,coarse_tokens,static_tokens):
        # Static morphology is only a cheap summary at this stage; full-resolution
        # urban computation remains behind the router.
        s=self.static_to_context(static_tokens.mean(1,keepdim=True))
        return torch.cat([coarse_tokens,s],1)

    def _run_urban_selected(self,batch,coarse,p,route_mask):
        B=batch['urban_baseline'].shape[0]
        residual=torch.zeros_like(batch['urban_baseline'])
        selected=torch.nonzero(route_mask,as_tuple=False).flatten()
        if selected.numel()==0:
            return residual,None
        r,tok,_=self.urban(
            batch['urban_history'][selected],
            batch['urban_static'][selected],
            coarse[selected],
            p[selected],
        )
        residual[selected]=r
        return residual,tok

    def _trace_process_predictions(self,p0,states):
        preds=[p0]
        preds.extend(self.verifier.predict_process(state) for state in states)
        return preds

    def forward(
        self,
        batch:Dict[str,torch.Tensor],
        *,
        force_zoom:Optional[bool]=None,
        adaptive_reasoning:bool=False,
        hard_route:bool=False,
    ) -> R2Output:
        coarse,chw=self.coarse(batch['coarse_history'])
        static_tok,_=self.static_preview(batch['urban_static'])
        cheap=self._cheap_context(coarse,static_tok)

        p,p0=self.process_init(cheap)
        pctx=self.reason_context(cheap)
        steps=self.train_reasoning_steps if self.training or not adaptive_reasoning else 1
        p,states=self.reasoner(p,pctx,steps=steps)
        process_pred_steps=self._trace_process_predictions(p0,states)

        logit,prob,roi=self.router(p,coarse,chw)
        B=prob.shape[0]
        if force_zoom is None:
            route_mask=(
                prob>=self.zoom_threshold
                if (hard_route and not self.training)
                else torch.ones(B,device=prob.device,dtype=torch.bool)
            )
        else:
            route_mask=torch.full(
                (B,),bool(force_zoom),device=prob.device,dtype=torch.bool
            )

        baseline=batch['urban_baseline']
        residual,urban_tok=self._run_urban_selected(batch,coarse,p,route_mask)
        forecast=baseline+residual
        score,pp,extra=self.verifier(p,baseline,forecast,residual)

        steps_per_sample=torch.full(
            (B,),int(steps),device=prob.device,dtype=torch.long
        )
        verifier_scores=[score]
        active_masks=[]

        # Adaptive inference is sample-wise. Samples that already satisfy the
        # verifier no longer consume recurrent reasoning or urban refinement.
        if adaptive_reasoning and not self.training:
            active=(score<self.verifier_threshold) & (
                steps_per_sample<self.max_reasoning_steps
            )
            while bool(active.any()):
                active_masks.append(active.detach().clone())
                selected=torch.nonzero(active,as_tuple=False).flatten()
                p_next,new_states=self.reasoner(
                    p[selected],pctx[selected],steps=1
                )
                p=p.clone()
                p[selected]=p_next
                steps_per_sample[selected]+=1
                process_pred_steps.append(self.verifier.predict_process(p))

                logit,prob,roi=self.router(p,coarse,chw)
                if force_zoom is None:
                    proposed_route=prob>=self.zoom_threshold
                    route_mask=torch.where(active,proposed_route,route_mask)

                update_mask=route_mask & active
                residual_update,urban_tok=self._run_urban_selected(
                    batch,coarse,p,update_mask
                )
                residual=residual.clone()
                residual[active]=residual_update[active]
                forecast=baseline+residual
                score,pp,extra=self.verifier(p,baseline,forecast,residual)
                verifier_scores.append(score)

                active=(score<self.verifier_threshold) & (
                    steps_per_sample<self.max_reasoning_steps
                )

        actual_steps=int(steps_per_sample.max().item())
        process_trace=torch.stack(process_pred_steps,dim=1)
        trace=ReasoningTrace(
            steps=actual_steps,
            actions=['INIT']+['REASON']*actual_steps,
            verifier_scores=verifier_scores,
            process_predictions=process_trace,
            active_masks=active_masks,
        )

        compute_cost=prob.detach() if hard_route else prob
        diag=R2Diagnostics(
            prob,
            score,
            pp,
            roi,
            actual_steps,
            compute_cost,
            route_mask,
            extra,
            trace,
            steps_per_sample,
        )
        return R2Output(forecast,residual,diag)

    def diffusion_loss(self,batch,out:R2Output):
        if self.diffusion is None:
            return out.forecast.new_zeros(())
        target_residual=batch['urban_target']-batch['urban_baseline']
        return self.diffusion.training_loss(target_residual,out.residual.detach())
