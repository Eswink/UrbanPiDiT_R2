"""Parameter exponential moving average for the S3 EMA weight-averaging round.

The frozen rule is deliberately the plain one: ``shadow <- decay * shadow +
(1 - decay) * parameter`` after every optimizer step, initialized from the
initial (parent) weights, over every trainable parameter, **no bias
correction** (the shadow is never divided by ``1 - decay**t``) and **no
warm-up** (the first step already uses ``decay``). ``decay=1.0`` therefore
keeps the parent weights forever while ``decay=0.0`` follows the latest
weights exactly; neither is the round's factor, they are the boundary
counterproofs.
"""
from __future__ import annotations

import math

import torch

FORMAT = "r7-parameter-ema-v1"
DEFAULT_DECAY = 0.999


def require_ema_decay(decay):
    """Validate an explicit EMA decay; ``None`` means the registered path (no EMA)."""
    if decay is None:
        return None
    if isinstance(decay, bool) or not isinstance(decay, (int, float)):
        raise ValueError("ema_decay must be a float in [0,1] or None")
    value = float(decay)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("ema_decay must be finite and in [0,1]")
    return value


class ParameterEMA:
    """Moving average over ``model.named_parameters()`` entries with ``requires_grad``."""

    def __init__(self, model, decay=DEFAULT_DECAY):
        self.decay = require_ema_decay(decay)
        if self.decay is None:
            raise ValueError("ema_decay=None does not construct an EMA")
        self._parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if not self._parameters:
            raise ValueError("the model has no trainable parameters to average")
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in self._parameters.items()
        }
        self.updates = 0

    @torch.no_grad()
    def update(self):
        """One update after one optimizer step; no warm-up and no bias correction."""
        for name, parameter in self._parameters.items():
            value = parameter.detach()
            if not torch.isfinite(value).all():
                raise RuntimeError("nonfinite trainable parameter before EMA update: " + name)
            shadow = self.shadow[name]
            if shadow.device != value.device:
                shadow = shadow.to(value.device)
                self.shadow[name] = shadow
            shadow.mul_(self.decay).add_(value, alpha=1.0 - self.decay)
        self.updates += 1
        return self.updates

    @torch.no_grad()
    def state_dict(self, model):
        """The full model state with every trainable parameter replaced by its average."""
        state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        for name, shadow in self.shadow.items():
            state[name] = shadow.detach().cpu().clone()
        return state
