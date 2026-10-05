"""Typed local diagnostic evidence for the #79 process forward path.

The frozen mechanism (docs/R7_S2_79_TYPED_DIAGNOSTICS_DESIGN.md): four local
physical fields - 850 hPa divergence, 850 hPa vorticity, 850 hPa temperature
advection and the 850-500 hPa static-stability difference - are computed with
the existing differentiable operators (no second differentiation is written
here) from the model's known state ``X_t`` and from its own initial draft
``Y_0``, standardized with train-only statistics published as a sidecar, then
patch-encoded by one shared projection and added into the anchored process
slots *before the first reasoning step*. The recurrence after that point is the
existing one, untouched.

Two capacity-matched modes exist, and the difference between them is the whole
point of the round:

- ``typed_routing``: each field keeps its own ``dim x dim`` projection and its
  output reaches only the anchored slot that belongs to its type. Per-type
  arrival is a measured property, not a claim (see ``inject`` and the probes).
- ``generic_fusion``: the same fields, the same parameter count and the same
  compute, but one merged projection consumes all four fields at every patch
  position and the fused vector is added to every evidence slot. It is the
  control arm against which typed routing is attributed.

Sources are legal model inputs only: ``X_t`` is the anchor the backbone already
decodes from and ``Y_0`` is the model's own initial decode. No target, no
future field and no validation label is read here, and the module is built last
under ``isolated_stream`` so that switching it off is the previous
implementation bit for bit. The train-only statistics arrive as
non-persistent buffers; nothing in this module enters a checkpoint's state
dict.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn

from .layers.patch_grid import pad_patch_grid
from .r7_process_tensor_diagnostics import (
    LOCAL_FIELD_NAMES, TYPED_EVIDENCE_CHANNELS, typed_local_fields,
)

TYPED_ROUTING = "typed_routing"
GENERIC_FUSION = "generic_fusion"
EVIDENCE_MODES = (TYPED_ROUTING, GENERIC_FUSION)

#: One anchored slot per typed field, in field order. Free slots and the other
#: anchored slots are never written by this module.
EVIDENCE_SLOTS = (0, 1, 2, 3)
EVIDENCE_TYPES = len(LOCAL_FIELD_NAMES)


def _vector(values, length, name, *, positive=False):
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if (tensor.ndim != 1 or tensor.shape[0] != length
            or not bool(torch.isfinite(tensor).all())):
        raise ValueError(f"{name} must be a finite [{length}] vector")
    if positive and bool((tensor <= 0).any()):
        raise ValueError(f"{name} must be positive")
    return tensor


class TypedEvidenceRouter(nn.Module):
    """Shared patch encoding plus per-type or fused projection into 4 slots.

    The state de-normalization vectors and the field standardization vectors
    are non-persistent buffers supplied by the caller from the published
    train-only sidecar: they are data identity, not trained artifacts, and a
    registry round-trip through a state dict would make them look like weights.
    """

    MODES = EVIDENCE_MODES
    SLOTS = EVIDENCE_SLOTS
    TYPES = EVIDENCE_TYPES

    def __init__(self, dim: int, patch_size: int, *, mode: str,
                 channels: Sequence[str], denorm_mean, denorm_std,
                 field_scale, field_mean, field_std, field_active):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        dim, patch_size = int(dim), int(patch_size)
        if dim < 1 or patch_size < 1:
            raise ValueError("dim and patch_size must be positive")
        names = tuple(channels)
        if (len(names) < 2 or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("channels must be unique nonempty names")
        missing = [name for name in TYPED_EVIDENCE_CHANNELS if name not in names]
        if missing:
            raise ValueError(f"typed evidence needs the diagnostic channels; missing {missing}")
        if len(names) < EVIDENCE_TYPES + 1:
            raise ValueError("at least one non-diagnostic channel is required")
        self.dim = dim
        self.patch_size = patch_size
        self.mode = mode
        self.channels = names
        active = torch.as_tensor(field_active, dtype=torch.bool)
        if active.shape != (EVIDENCE_TYPES,):
            raise ValueError(f"field_active must be {EVIDENCE_TYPES} booleans")
        if not bool(active.any()):
            raise ValueError("at least one typed field must be active; "
                             "an all-degenerate set is refused, not faked")
        # Shared patch projection: the same 1 -> dim patch convolution and
        # normalization are applied to every field, exactly like the draft
        # encoder's patch on the same grid. Constructed before the projections
        # so the two arms' shared tensors come from one stream position.
        self.conv = nn.Conv2d(1, dim, patch_size, patch_size)
        self.norm = nn.LayerNorm(dim)
        if mode == TYPED_ROUTING:
            self.projections = nn.ModuleList(
                [nn.Linear(dim, dim, bias=False) for _ in range(EVIDENCE_TYPES)])
        else:
            self.fusion = nn.Linear(EVIDENCE_TYPES * dim, dim, bias=False)
        self.register_buffer("denorm_mean", _vector(denorm_mean, len(names), "denorm_mean"),
                             persistent=False)
        self.register_buffer("denorm_std", _vector(denorm_std, len(names), "denorm_std",
                                                   positive=True),
                             persistent=False)
        self.register_buffer("field_scale", _vector(field_scale, EVIDENCE_TYPES,
                                                    "field_scale", positive=True),
                             persistent=False)
        self.register_buffer("field_mean", _vector(field_mean, EVIDENCE_TYPES, "field_mean"),
                             persistent=False)
        self.register_buffer("field_std", _vector(field_std, EVIDENCE_TYPES, "field_std",
                                                  positive=True),
                             persistent=False)
        self.register_buffer("field_active", active, persistent=False)

    def field_maps(self, state: torch.Tensor, batch: Mapping) -> torch.Tensor:
        """``[B,4,H,W]`` standardized train-only fields of one normalized state.

        The state is de-normalized with the store's own train-only vectors, the
        four fields are computed in physical units by the shared operators and
        standardized with the published sidecar. Masked types are exactly zero
        and are never divided, so a degenerate field cannot leak a ratio.
        """
        latitude, longitude = batch.get("latitude"), batch.get("longitude")
        if latitude is None or longitude is None:
            raise ValueError("typed evidence requires latitude and longitude in the batch")
        if (not isinstance(state, torch.Tensor) or state.ndim != 4
                or state.shape[1] != len(self.channels) or min(state.shape) < 1):
            raise ValueError("typed evidence state must be [B,C,H,W] on the declared channels")
        with torch.autocast(state.device.type, enabled=False):
            physical = (state.float() * self.denorm_std.to(state.device)[:, None, None]
                        + self.denorm_mean.to(state.device)[:, None, None])
            fields = typed_local_fields(physical, self.channels,
                                        latitude.to(state.device), longitude.to(state.device))
            active = self.field_active.to(state.device)
            result = torch.zeros_like(fields)
            scale = self.field_scale.to(state.device)[active][:, None, None]
            center = self.field_mean.to(state.device)[active][:, None, None]
            deviation = self.field_std.to(state.device)[active][:, None, None]
            result[:, active] = (fields[:, active] / scale - center) / deviation
            if not bool(torch.isfinite(result).all()):
                raise ValueError("nonfinite standardized typed evidence")
            return result

    def encode(self, fields: torch.Tensor) -> torch.Tensor:
        """``[B,4,hw,dim]`` patch tokens of the four fields, sharing one projection."""
        batch, count, height, width = fields.shape
        if count != EVIDENCE_TYPES:
            raise ValueError(f"typed fields must have exactly {EVIDENCE_TYPES} channels")
        with torch.autocast(fields.device.type, enabled=False):
            padded = pad_patch_grid(fields.reshape(batch * count, 1, height, width),
                                    self.patch_size)
            encoded = self.conv(padded)
            rows, columns = encoded.shape[-2:]
            tokens = self.norm(encoded.flatten(2).transpose(1, 2))
            return tokens.reshape(batch, count, rows * columns, self.dim)

    def token_evidence(self, fields: torch.Tensor) -> torch.Tensor:
        """The per-sample evidence added to the slots.

        Typed mode returns ``[B,4,dim]`` (one projection per type, pooled over
        patch positions); generic mode returns ``[B,dim]`` (one fused vector
        from all four types at every position). Both consume identical inputs
        and identical parameter counts; that is what makes the contrast
        attributable to the type structure rather than to capacity.

        Masked types contribute exactly zero: their fields are zeroed before
        the shared encode (so a masked type can never carry a non-finite or
        out-of-range value into the shared projection) and their tokens are
        zeroed after it (a zero field still picks up the convolution and
        normalization biases, which must not become evidence).
        """
        with torch.autocast(fields.device.type, enabled=False):
            active = self.field_active.to(fields.device)
            mask = active.reshape(1, EVIDENCE_TYPES, 1, 1)
            fields = torch.where(mask, fields, torch.zeros((), device=fields.device,
                                                           dtype=fields.dtype))
            tokens = self.encode(fields)
            tokens = torch.where(active.reshape(1, EVIDENCE_TYPES, 1, 1), tokens,
                                 torch.zeros((), device=tokens.device, dtype=tokens.dtype))
            if self.mode == TYPED_ROUTING:
                projected = [projection(tokens[:, index])
                             for index, projection in enumerate(self.projections)]
                return torch.stack([value.mean(dim=1) for value in projected], dim=1)
            positions = tokens.shape[2]
            stacked = tokens.permute(0, 2, 1, 3).reshape(
                tokens.shape[0], positions, EVIDENCE_TYPES * self.dim)
            return self.fusion(stacked).mean(dim=1)

    def evidence(self, batch: Mapping, *, anchor: torch.Tensor,
                 draft: torch.Tensor) -> torch.Tensor:
        """Evidence of ``X_t`` plus evidence of the model's own initial draft.

        The two sources are computed and encoded separately in their own
        physical time and are added only after their projections; they are
        never mixed into one field. The caller passes the exact tensors the
        first reasoning step consumes - the anchor ``X_t`` and the initial
        draft ``Y_0`` - so the evidence cannot read a different tensor than the
        recurrence does.
        """
        initial = self.token_evidence(self.field_maps(anchor, batch))
        draft = self.token_evidence(self.field_maps(draft, batch))
        return initial + draft

    def inject(self, process: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        """Add the evidence into the anchored evidence slots; nothing else moves."""
        if (process.ndim != 3 or process.shape[-1] != self.dim
                or process.shape[1] < EVIDENCE_TYPES):
            raise ValueError("process state must be [B,M>=4,dim]")
        if self.mode == TYPED_ROUTING:
            if evidence.shape != (process.shape[0], EVIDENCE_TYPES, self.dim):
                raise ValueError("typed evidence must be [B,4,dim]")
            return torch.cat((process[:, :EVIDENCE_TYPES] + evidence,
                              process[:, EVIDENCE_TYPES:]), dim=1)
        if evidence.shape != (process.shape[0], self.dim):
            raise ValueError("fused evidence must be [B,dim]")
        fused = process[:, :EVIDENCE_TYPES] + evidence[:, None, :]
        return torch.cat((fused, process[:, EVIDENCE_TYPES:]), dim=1)
