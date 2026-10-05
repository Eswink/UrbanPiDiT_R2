"""Train-fitted process scaling and explicitly timed supervision-only fields.

No model or dataset is changed here. Callers supply stored UTC nanoseconds;
missing timestamps are errors, never reconstructed from calendar phase/lead.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import torch

from data.preprocess.process_diagnostics import (
    PROCESS_DIAGNOSTIC_NAMES, REQUIRED_PROCESS_CHANNELS,
)
from model.r7_process_tensor_diagnostics import compute_process_tensor_diagnostics

HOUR_NS = 3_600_000_000_000
ATMOSPHERIC_CHANNEL_COUNT = 17


def _validate_metadata(metadata):
    # One validator owns names/units, provenance, relative floor and mask rules.
    from data.preprocess.r7_process_scale_sidecar import validate_scale_metadata
    validate_scale_metadata(metadata)


def _fixed_vector(value, name, count, *, positive=False):
    raw = torch.as_tensor(value)
    if raw.dtype == torch.bool or raw.is_complex() or raw.shape != (count,):
        raise ValueError(f"{name} must be a numeric [{count}] tensor")
    vector = raw.detach().to(device="cpu", dtype=torch.float32).clone()
    if not bool(torch.isfinite(vector).all()) or (positive and not bool((vector > 0).all())):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return vector


def _time_tensor(value, name, shape, device):
    raw = torch.as_tensor(value)
    if raw.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"{name} must be integer UTC nanoseconds")
    if raw.shape != shape or bool((raw < 0).any()):
        raise ValueError(f"{name} must be nonnegative with shape {shape}")
    return raw.detach().to(device=device, dtype=torch.int64).clone()


def _batch_times(batch, history, *, training, need_future):
    size, count = history.shape[:2]
    device = history.device
    required = ("init_time_ns", "valid_time_ns", "history_time_ns", "lead_time_hours")
    missing = [name for name in required if name not in batch]
    if missing:
        raise ValueError(f"missing explicit diagnostic timestamps/context: {missing}")
    init = _time_tensor(batch["init_time_ns"], "init_time_ns", (size,), device)
    valid = _time_tensor(batch["valid_time_ns"], "valid_time_ns", (size,), device)
    times = _time_tensor(batch["history_time_ns"], "history_time_ns", (size, count), device)
    if not torch.equal(times[:, -1], init):
        raise ValueError("input diagnostics timestamp must equal init/final history timestamp")
    if count > 1:
        intervals = times.diff(dim=1)
        if not bool((intervals > 0).all()) or not bool((intervals == intervals[:, :1]).all()):
            raise ValueError("history timestamps must have positive regular cadence")
    lead = torch.as_tensor(batch["lead_time_hours"], device=device)
    if (lead.dtype == torch.bool or lead.is_complex() or lead.shape != (size,)
            or not bool(torch.isfinite(lead).all()) or not bool((lead > 0).all())):
        raise ValueError("lead_time_hours must be finite positive [B]")
    delta = lead.double() * HOUR_NS
    if (not bool((delta == delta.round()).all()) or bool((delta >= 2 ** 63).any())
            or not bool((valid > init).all()) or not torch.equal(valid - init, delta.long())):
        raise ValueError("valid time must exactly equal init + lead, not K * lead")
    # Inference deliberately does not even read a future-label timestamp.
    if training and (need_future or "target_time_ns" in batch):
        if "target_time_ns" not in batch:
            raise ValueError("future targets require explicit target_time_ns")
        target = _time_tensor(batch["target_time_ns"], "target_time_ns", (size,), device)
        if not torch.equal(target, valid):
            raise ValueError("future target timestamp must equal valid time")
    return init, valid


def _split_labels(batch, size, *, require_train):
    labels = batch.get("split")
    if isinstance(labels, str):
        labels = (labels,) * size
    elif labels is not None:
        labels = tuple(labels)
    if labels is not None and (len(labels) != size or any(not isinstance(x, str) for x in labels)):
        raise ValueError("split must be a string or DataLoader [B] string labels")
    if require_train and (labels is None or any(x != "train" for x in labels)):
        raise ValueError("future diagnostic supervision is allowed only for explicit train split")
    return labels


def _declared_names(batch, key, expected, size):
    if key not in batch:
        return
    value = batch[key]
    direct = tuple(value)
    if direct == expected:
        return
    rows = [tuple(row) for row in direct if not isinstance(row, str)]
    if len(rows) == size and all(row == expected for row in rows):
        return
    if len(rows) == len(expected) and all(row == (name,) * size for row, name in zip(rows, expected)):
        return
    raise ValueError(f"{key} names/order mismatch")


@dataclass(frozen=True)
class ProcessDiagnosticFields:
    """Different sources stay different names, even at the same valid time."""
    input_diagnostics: torch.Tensor
    future_diagnostic_targets: torch.Tensor | None
    draft_diagnostics: torch.Tensor | None
    input_diagnostics_time_ns: torch.Tensor
    future_diagnostic_targets_time_ns: torch.Tensor | None
    draft_diagnostics_time_ns: torch.Tensor | None
    init_time_ns: torch.Tensor
    valid_time_ns: torch.Tensor
    names: tuple[str, ...]
    active_mask: torch.Tensor
    metadata: Mapping
    channel_names: tuple[str, ...]
    spatial_shape: tuple[int, int]
    training: bool
    split: tuple[str, ...] | None


def validate_diagnostic_fields(fields, *, need_input=False, need_future=False, need_draft=False):
    if not isinstance(fields, ProcessDiagnosticFields):
        raise ValueError("enabled diagnostic loss requires ProcessDiagnosticFields metadata")
    _validate_metadata(fields.metadata)
    if fields.names != tuple(PROCESS_DIAGNOSTIC_NAMES):
        raise ValueError("process diagnostic names/order mismatch")
    mask = torch.as_tensor(fields.active_mask)
    expected = torch.as_tensor(fields.metadata["active_mask"], dtype=torch.bool, device=mask.device)
    if mask.dtype != torch.bool or mask.shape != (8,) or not torch.equal(mask, expected):
        raise ValueError("diagnostic active mask/metadata mismatch")
    if not bool(mask.any()):
        raise ValueError("diagnostic active mask cannot be entirely disabled")
    if fields.init_time_ns.ndim != 1:
        raise ValueError("init_time_ns must be [B]")
    size = fields.init_time_ns.shape[0]
    init = _time_tensor(fields.init_time_ns, "init_time_ns", (size,), fields.init_time_ns.device)
    valid = _time_tensor(fields.valid_time_ns, "valid_time_ns", (size,), init.device)
    if not bool((valid > init).all()):
        raise ValueError("diagnostic valid timestamp must follow init")
    sources = (
        ("input_diagnostics", fields.input_diagnostics, fields.input_diagnostics_time_ns, need_input, init),
        ("future_diagnostic_targets", fields.future_diagnostic_targets,
         fields.future_diagnostic_targets_time_ns, need_future, valid),
        ("draft_diagnostics", fields.draft_diagnostics, fields.draft_diagnostics_time_ns, need_draft, valid),
    )
    for name, value, timestamp, required, reference in sources:
        if value is None:
            if required or timestamp is not None:
                raise ValueError(f"missing {name} or inconsistent timestamp")
            continue
        ndim = 3 if name == "draft_diagnostics" else 2
        if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
                or value.ndim != ndim or value.shape[0] != size or value.shape[-1] != 8
                or min(value.shape) < 1 or not bool(torch.isfinite(value).all())):
            raise ValueError(f"{name} must be finite exact [B,8] or [B,S,8] diagnostics")
        if timestamp is None:
            raise ValueError(f"missing {name} timestamp")
        stamp = _time_tensor(timestamp, name + "_time_ns", tuple(value.shape[:-1]), init.device)
        target = reference[:, None].expand_as(stamp) if ndim == 3 else reference
        if not torch.equal(stamp, target):
            raise ValueError(f"{name} timestamp misalignment; internal K does not add lead")
        if bool((value[..., ~mask.to(value.device)] != 0).any()):
            raise ValueError(f"{name} masked channels must be exactly zero")
    if need_future and (not fields.training or fields.split is None or any(x != "train" for x in fields.split)):
        raise ValueError("future diagnostic targets are train-only supervision")
    if fields.future_diagnostic_targets is not None and fields.future_diagnostic_targets.requires_grad:
        raise ValueError("future diagnostic targets must be detached supervision")
    return mask


class ProcessDiagnosticContext:
    """A fixed train-fitted context, not a batch-dependent normalizer or model."""
    def __init__(self, metadata: Mapping, normalization_mean, normalization_std,
                 channel_names: Sequence[str]):
        if not isinstance(metadata, Mapping):
            raise ValueError("process scale metadata is required")
        self._metadata = deepcopy(dict(metadata))
        _validate_metadata(self._metadata)
        self.channel_names = tuple(channel_names)
        if (len(self.channel_names) != ATMOSPHERIC_CHANNEL_COUNT
                or any(not isinstance(name, str) or not name for name in self.channel_names)
                or len(set(self.channel_names)) != ATMOSPHERIC_CHANNEL_COUNT):
            raise ValueError("normalization/channel names must exactly match 17 unique channels")
        if any(name not in self.channel_names for name in REQUIRED_PROCESS_CHANNELS):
            raise ValueError("missing atmospheric process diagnostic channel names")
        self.normalization_mean = _fixed_vector(normalization_mean, "normalization_mean", 17)
        self.normalization_std = _fixed_vector(normalization_std, "normalization_std", 17, positive=True)
        self._scale = _fixed_vector(self._metadata["physical_unit_scale"], "physical_unit_scale", 8, positive=True)
        self._mean = _fixed_vector(self._metadata["dimensionless_mean"], "dimensionless_mean", 8)
        self._std = _fixed_vector(self._metadata["dimensionless_std"], "dimensionless_std", 8)
        self.active_mask = torch.as_tensor(self._metadata["active_mask"], dtype=torch.bool).clone()
        if not bool((self._std[self.active_mask] > 0).all()):
            raise ValueError("unsafe FP32 normalization: active std must remain positive")

    @property
    def metadata(self) -> dict:
        """Public contract snapshot; mutating it cannot alter this fixed context."""
        return deepcopy(self._metadata)

    def diagnostics(self, normalized: torch.Tensor, latitude, longitude) -> torch.Tensor:
        if (not isinstance(normalized, torch.Tensor) or not normalized.is_floating_point()
                or normalized.ndim < 3 or normalized.shape[-3] != 17 or min(normalized.shape) < 1):
            raise ValueError("normalized atmospheric state must be [...,17,H,W]")
        if not bool(torch.isfinite(normalized).all()):
            raise ValueError("nonfinite normalized atmospheric state")
        with torch.autocast(normalized.device.type, enabled=False):
            # ALWAYS the frozen train tensors; never fit or rescale using a batch.
            mean = self.normalization_mean.to(normalized.device)[:, None, None]
            std = self.normalization_std.to(normalized.device)[:, None, None]
            physical = normalized.float() * std + mean
            raw = compute_process_tensor_diagnostics(physical, self.channel_names, latitude, longitude)
            active = self.active_mask.to(raw.device)
            scale, center, deviation = (value.to(raw.device) for value in (self._scale, self._mean, self._std))
            # Do not divide masked channels by a floor, even transiently.
            safe = (raw[..., active] / scale[active] - center[active]) / deviation[active]
            result = torch.zeros_like(raw).index_copy(-1, active.nonzero().flatten(), safe)
            if not bool(torch.isfinite(result).all()):
                raise ValueError("nonfinite or unsafe normalized process diagnostics")
            return result

    def fields_from_batch(self, batch: Mapping, *, training: bool = True,
                          need_future: bool = True) -> ProcessDiagnosticFields:
        if type(training) is not bool or type(need_future) is not bool:
            raise ValueError("training and need_future must be boolean")
        history = batch.get("coarse_history")
        if (not isinstance(history, torch.Tensor) or history.ndim != 5
                or history.shape[2] != 17 or min(history.shape) < 1):
            raise ValueError("coarse_history must be nonempty [B,T,17,H,W]")
        if "latitude" not in batch or "longitude" not in batch:
            raise ValueError("diagnostic context requires explicit latitude and longitude")
        future = training and need_future
        init, valid = _batch_times(batch, history, training=training, need_future=future)
        split = _split_labels(batch, history.shape[0], require_train=future)
        _declared_names(batch, "channel_names", self.channel_names, history.shape[0])
        _declared_names(batch, "process_diagnostic_names", tuple(PROCESS_DIAGNOSTIC_NAMES), history.shape[0])
        with torch.no_grad():
            inputs = self.diagnostics(history[:, -1].detach(), batch["latitude"], batch["longitude"])
            target = None
            if future:
                if "atmos_target" not in batch:
                    raise ValueError("future diagnostic supervision requires atmos_target")
                state = batch["atmos_target"]
                if (not isinstance(state, torch.Tensor)
                        or state.shape != (history.shape[0], 17, *history.shape[-2:])
                        or state.device != history.device):
                    raise ValueError("future atmospheric target shape/device mismatch")
                target = self.diagnostics(state.detach(), batch["latitude"], batch["longitude"])
        return ProcessDiagnosticFields(
            inputs, target, None, init, valid.clone() if future else None, None,
            init, valid, tuple(PROCESS_DIAGNOSTIC_NAMES), self.active_mask.clone(), self.metadata,
            self.channel_names, tuple(history.shape[-2:]), training, split,
        )

    def validate_prepared_fields(self, fields: ProcessDiagnosticFields, batch: Mapping, *,
                                 need_future: bool = False) -> None:
        """Validate a cached loss context against this batch, without refitting/recomputing D."""
        validate_diagnostic_fields(fields, need_future=need_future)
        if fields.metadata != self._metadata or fields.channel_names != self.channel_names:
            raise ValueError("prepared diagnostic fields/context metadata or names mismatch")
        history = batch.get("coarse_history")
        if (not isinstance(history, torch.Tensor) or history.ndim != 5
                or history.shape[:1] != fields.input_diagnostics.shape[:1]
                or history.shape[2:] != (17, *fields.spatial_shape) or min(history.shape) < 1):
            raise ValueError("prepared fields/history shape mismatch")
        init, valid = _batch_times(batch, history, training=True, need_future=need_future)
        if not torch.equal(init, fields.init_time_ns) or not torch.equal(valid, fields.valid_time_ns):
            raise ValueError("prepared fields/batch timestamp misalignment")
        labels = _split_labels(batch, history.shape[0], require_train=need_future)
        if labels != fields.split:
            raise ValueError("prepared fields/batch split mismatch")
        _declared_names(batch, "channel_names", self.channel_names, history.shape[0])
        _declared_names(batch, "process_diagnostic_names", tuple(PROCESS_DIAGNOSTIC_NAMES), history.shape[0])
        if "latitude" not in batch or "longitude" not in batch:
            raise ValueError("diagnostic context requires explicit latitude and longitude")
        if need_future:
            target = batch.get("atmos_target")
            if (not isinstance(target, torch.Tensor) or not target.is_floating_point()
                    or target.shape != (history.shape[0], 17, *fields.spatial_shape)
                    or target.device != history.device or not bool(torch.isfinite(target).all())):
                raise ValueError("future atmospheric target missing, nonfinite or shape/device mismatch")

    def with_drafts(self, fields: ProcessDiagnosticFields, drafts: torch.Tensor,
                    latitude, longitude, *, draft_time_ns=None) -> ProcessDiagnosticFields:
        validate_diagnostic_fields(fields)
        if fields.metadata != self._metadata or fields.channel_names != self.channel_names:
            raise ValueError("prepared diagnostic fields/context metadata or names mismatch")
        if drafts.ndim == 4:
            drafts = drafts.unsqueeze(1)
        expected = (fields.input_diagnostics.shape[0], 17, *fields.spatial_shape)
        if (drafts.ndim != 5 or drafts.shape[1] < 1
                or (drafts.shape[0], *drafts.shape[2:]) != expected):
            raise ValueError("drafts must exactly match prepared [B,S,17,H,W] state")
        stamp = fields.valid_time_ns[:, None].expand(-1, drafts.shape[1]).clone()
        if draft_time_ns is not None:
            actual = _time_tensor(draft_time_ns, "draft_time_ns", tuple(stamp.shape), stamp.device)
            if not torch.equal(actual, stamp):
                raise ValueError("draft timestamp mismatch; internal K does not add lead")
        values = self.diagnostics(drafts, latitude, longitude)
        return replace(fields, draft_diagnostics=values, draft_diagnostics_time_ns=stamp)

    def fields_from_output(self, batch: Mapping, out, *, training: bool = True,
                           need_future: bool = True) -> ProcessDiagnosticFields:
        fields = self.fields_from_batch(batch, training=training, need_future=need_future)
        return self.with_drafts(fields, out.draft_forecasts, batch["latitude"], batch["longitude"],
                                draft_time_ns=getattr(out, "draft_time_ns", None))


def process_gradient_ownership(model, draft: torch.Tensor | None = None) -> dict:
    """Read-only post-backward norms; missing grads are distinct from zero grads.

    These overlapping architectural groups are descriptive, not attribution of
    scientific causality. Read after the runner's streamed VJP/accumulation.
    """
    groups = {name: [] for name in ("process_queries", "readout", "shared", "draft_forecast")}
    for name, parameter in model.named_parameters():
        if name == "process_queries":
            owner = "process_queries"
        elif name.startswith(("process_readout.", "process_reader.", "process_to_context.")):
            owner = "readout"
        elif name.startswith(("correction_head.", "proposal_head.")):
            owner = "draft_forecast"
        else:
            owner = "shared"
        groups[owner].append(parameter.grad)
    result = {}
    for name, values in groups.items():
        present = [value.detach().float() for value in values if value is not None]
        if any(not bool(torch.isfinite(value).all()) for value in present):
            raise ValueError(f"nonfinite gradients in {name}")
        norm = sum(float(value.double().square().sum()) for value in present) ** .5
        result[name] = {"norm": norm, "grad_tensors": len(present), "missing_grads": len(values) - len(present)}
    if draft is not None:
        if not draft.is_leaf and not draft.retains_grad:
            raise ValueError("draft gradient logging requires retain_grad() before backward")
        gradient = draft.grad
        if gradient is not None and not bool(torch.isfinite(gradient).all()):
            raise ValueError("nonfinite draft tensor gradient")
        result["draft_tensor"] = {"norm": None if gradient is None else float(gradient.double().norm()),
                                  "has_grad": gradient is not None}
    return result
