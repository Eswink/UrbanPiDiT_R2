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
    steps: int
    actions: List[str] = field(default_factory=list)
    verifier_scores: List[torch.Tensor] = field(default_factory=list)
    process_predictions: Optional[torch.Tensor] = None

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
