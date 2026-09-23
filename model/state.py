from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import torch


@dataclass
class ReasoningAction:
    action_type: str
    zoom_probability: torch.Tensor
    roi_scores: Optional[torch.Tensor] = None


@dataclass
class ReasoningTrace:
    """Auditable latent reasoning trajectory.

    process_predictions is shaped [B, S+1, P] when available: initializer
    prediction P0 followed by predictions after each recurrent reasoning step.
    active_masks records which samples actually consumed each extra adaptive step.
    """
    steps: int
    actions: List[str] = field(default_factory=list)
    verifier_scores: List[torch.Tensor] = field(default_factory=list)
    process_predictions: Optional[torch.Tensor] = None
    active_masks: List[torch.Tensor] = field(default_factory=list)


@dataclass
class WeatherState:
    coarse_tokens: torch.Tensor
    process_tokens: torch.Tensor
    coarse_hw: tuple[int,int]
    urban_tokens: Optional[torch.Tensor] = None
    forecast: Optional[torch.Tensor] = None
    confidence: Optional[torch.Tensor] = None


@dataclass
class R2Diagnostics:
    zoom_probability: torch.Tensor
    verifier_score: torch.Tensor
    process_predictions: torch.Tensor
    roi_scores: torch.Tensor
    reasoning_steps: int
    compute_cost: torch.Tensor
    route_mask: torch.Tensor
    extras: Dict[str, torch.Tensor] = field(default_factory=dict)
    trace: Optional[ReasoningTrace] = None
    reasoning_steps_per_sample: Optional[torch.Tensor] = None
