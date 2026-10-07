"""Explicit train-only, normalized FP32 month/hour constants (no fitting or I/O).

The caller qualifies the source, selection, units and normalization outside this
module. Hashes bind declarations and installed bytes, not their scientific truth.
Construction allocates zeros only; explicit install or a validated ordinary state
load must precede querying. Missing buckets never use held-out data or a fallback.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping

import torch
from torch import nn

from .spacetime_conditioning_r7 import advance_calendar_time, require_spacetime_fields

_SPEC_KEYS = frozenset({
    "format", "kind", "channels", "units", "source_sha256", "train_data_identity",
    "climatology_mean_identity_sha256", "normalization_mean", "normalization_std",
    "latitude", "longitude", "bucket_keys", "bucket_counts", "training_years",
    "n_selected_steps", "selection", "table_sha256", "table_shape",
})
_INPUT_FIELDS = ("latitude", "longitude", "init_calendar_year", "init_day_of_year",
                 "init_utc_hour", "lead_time_hours")
_SYNOPTIC_HOURS = (0, 6, 12, 18)


def _json_list(value, name, *, size=None):
    if type(value) is not list or not value:
        raise ValueError(f"{name} must be a nonempty JSON list")
    if size is not None and len(value) != size:
        raise ValueError(f"{name} must contain exactly {size} values")
    return value


def _finite_numbers(value, name, *, size=None, floats_only=False, positive=False):
    values = _json_list(value, name, size=size)
    types = (float,) if floats_only else (int, float)
    if any(type(item) not in types or not math.isfinite(item) for item in values):
        raise ValueError(f"{name} must contain finite JSON {'floats' if floats_only else 'numbers'}")
    if positive and any(item <= 0 for item in values):
        raise ValueError(f"{name} must contain positive values")
    return values


def _validate_spec(spec):
    """Exact JSON schema, including identities and dimension/count consistency."""
    if type(spec) is not dict:
        raise TypeError("climatology spec must be a JSON dict")
    if set(spec) != _SPEC_KEYS:
        raise ValueError("climatology spec must have the exact declared key set")
    if spec["format"] != "r7-train-climatology-anchor-v1":
        raise ValueError("invalid climatology spec format")
    if spec["kind"] != "train-only-month-hour-grid-mean-v1":
        raise ValueError("invalid climatology spec kind")
    for name in ("source_sha256", "train_data_identity", "climatology_mean_identity_sha256",
                 "table_sha256"):
        digest = spec[name]
        if type(digest) is not str or len(digest) != 64 or any(
                char not in "0123456789abcdefABCDEF" for char in digest):
            raise ValueError(f"{name} must be a 64-hex SHA256 identity")
    channels = _json_list(spec["channels"], "channels")
    if any(type(item) is not str or not item.strip() for item in channels):
        raise ValueError("channels must contain nonempty strings")
    if len(set(channels)) != len(channels):
        raise ValueError("channels must be unique and ordered")
    units = _json_list(spec["units"], "units", size=len(channels))
    if any(type(item) is not str or not item.strip() for item in units):
        raise ValueError("units must contain nonempty strings, one per channel")
    for name in ("normalization_mean", "normalization_std"):
        _finite_numbers(spec[name], name, size=len(channels), floats_only=True,
                        positive=name == "normalization_std")
    latitude = _finite_numbers(spec["latitude"], "latitude")
    longitude = _finite_numbers(spec["longitude"], "longitude")
    keys = _json_list(spec["bucket_keys"], "bucket_keys")
    for key in keys:
        _json_list(key, "bucket_keys entry", size=2)
        if any(type(item) is not int for item in key) or not 1 <= key[0] <= 12 \
                or key[1] not in _SYNOPTIC_HOURS:
            raise ValueError("bucket_keys must be legal [month, synoptic UTC hour] pairs")
    if keys != sorted(keys) or len({tuple(key) for key in keys}) != len(keys):
        raise ValueError("bucket_keys must be unique and sorted")
    counts = _json_list(spec["bucket_counts"], "bucket_counts", size=len(keys))
    if any(type(item) is not int or item <= 0 for item in counts):
        raise ValueError("bucket_counts must contain positive integers")
    years = _json_list(spec["training_years"], "training_years")
    if any(type(item) is not int or not 1 <= item <= 9999 for item in years) \
            or years != sorted(set(years)):
        raise ValueError("training_years must contain unique sorted Gregorian integer years")
    steps = spec["n_selected_steps"]
    if type(steps) is not int or steps != sum(counts):
        raise ValueError("n_selected_steps must equal the exact sum of bucket_counts")
    if type(spec["selection"]) is not str or spec["selection"] not in (
            "declared_train_years", "declared_train_time_ranges"):
        raise ValueError("selection must explicitly declare the train-only selection mode")
    shape = _json_list(spec["table_shape"], "table_shape", size=4)
    expected = [len(keys), len(channels), len(latitude), len(longitude)]
    if any(type(item) is not int for item in shape) or shape != expected:
        raise ValueError(f"table_shape must match [N,C,H,W]={expected}")
    return json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _table_digest(table):
    """SHA256 of logical contiguous CPU FP32 bytes; called only at install/load."""
    return hashlib.sha256(table.detach().cpu().contiguous().numpy().tobytes(order="C")).hexdigest()


def _numeric_field(value, name, history, *, per_sample=False):
    if not torch.is_tensor(value) or value.dtype == torch.bool or value.is_complex():
        raise TypeError(f"{name} must be a real numeric tensor")
    if value.device != history.device:
        raise ValueError(f"{name} must be on the history device")
    if value.layout != torch.strided or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be dense and finite")
    if per_sample and value.ndim != 0 and tuple(value.shape) != (history.shape[0],):
        raise ValueError(f"{name} must be scalar or [{history.shape[0]}]")
    if per_sample:
        return value.detach().expand(history.shape[0]).to(torch.float64)
    return value.detach()


def _calendar_month(year, day):
    leap = (year.remainder(4) == 0) & ((year.remainder(100) != 0) | (year.remainder(400) == 0))
    ends = torch.tensor((31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365),
                        device=day.device, dtype=torch.int64)
    adjusted = ends[None, :] + leap[:, None] * (torch.arange(12, device=day.device)[None, :] >= 1)
    return (day[:, None] > adjusted).sum(dim=1) + 1


def uses_climatology_anchor(model):
    """Detect the explicit optional anchor, including PE-off wrapped backbones."""
    forecaster = getattr(model, "forecaster", model)
    backbone = getattr(forecaster, "backbone", forecaster)
    return getattr(backbone, "climatology_anchor", None) is not None


class TrainClimatologyAnchor(nn.Module):
    """Immutable-declaration table queried once per physical forecast transition.

    ``spec`` returns a fresh deep JSON copy; its canonical snapshot never aliases
    caller metadata. Persistent buffers are ordinary checkpoint tensors, not
    parameters. Public install is one-shot, and loads accept only the declared
    finite FP32 bytes with a true readiness marker. No source is read or fitted.
    """

    def __init__(self, spec: dict):
        super().__init__()
        self._spec_json = _validate_spec(spec)
        snapshot = json.loads(self._spec_json)
        self._shape = tuple(snapshot["table_shape"])
        self._sha256 = snapshot["table_sha256"].lower()
        self._latitude = tuple(snapshot["latitude"])
        self._longitude = tuple(snapshot["longitude"])
        self._bucket_keys = tuple(tuple(key) for key in snapshot["bucket_keys"])
        self._installed = False
        self.register_buffer("table", torch.zeros(self._shape, dtype=torch.float32, device="cpu"), persistent=True)
        self.register_buffer("ready", torch.tensor(False, dtype=torch.bool, device="cpu"), persistent=True)

    @property
    def spec(self):
        return json.loads(self._spec_json)

    @property
    def spec_json(self):
        return self._spec_json

    def _require_storage(self):
        if self.table.dtype != torch.float32 or tuple(self.table.shape) != self._shape \
                or self.table.layout != torch.strided or self.table.requires_grad or self.table.grad is not None:
            raise ValueError("climatology table buffer must remain dense FP32 with declared shape and no grad")
        if self.ready.dtype != torch.bool or self.ready.ndim != 0 \
                or self.ready.device != self.table.device:
            raise ValueError("climatology ready buffer must remain scalar bool on the table device")

    def _validate_table(self, table):
        if not torch.is_tensor(table):
            raise TypeError("climatology table must be a tensor")
        if table.dtype != torch.float32 or tuple(table.shape) != self._shape \
                or table.layout != torch.strided:
            raise ValueError("climatology table must be dense FP32 with the declared table_shape")
        if table.requires_grad or table.grad_fn is not None or table.grad is not None:
            raise ValueError("climatology table must have no grad or requires_grad")
        if not torch.isfinite(table).all():
            raise ValueError("climatology table must be finite")
        if _table_digest(table) != self._sha256:
            raise ValueError("climatology table SHA256 does not match the immutable spec")

    def install(self, table: torch.Tensor):
        """Explicit externally fitted normalized table; copy without dtype changes."""
        self._require_storage()
        if self._installed or bool(self.ready):
            raise RuntimeError("climatology install is one-shot; overwrite is forbidden")
        self._validate_table(table)
        with torch.no_grad():
            self.table.copy_(table)
            self.ready.fill_(True)
        self._installed = True

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Fail before accepting either buffer, even for a non-strict parent load."""
        try:
            self._require_storage()
            if prefix + "table" not in state_dict or prefix + "ready" not in state_dict:
                raise ValueError("climatology load requires both table and ready buffers")
            incoming_ready = state_dict[prefix + "ready"]
            if not torch.is_tensor(incoming_ready) or incoming_ready.dtype != torch.bool \
                    or incoming_ready.ndim != 0 or incoming_ready.layout != torch.strided \
                    or not bool(incoming_ready):
                raise ValueError("climatology checkpoint ready must be scalar bool True")
            self._validate_table(state_dict[prefix + "table"])
        except (TypeError, ValueError, RuntimeError) as error:
            error_msgs.append(f"{prefix}climatology anchor rejected state: {error}")
            return
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
        self._installed = True

    def forward(self, batch: Mapping, history: torch.Tensor, *, default_lead_hours: float = 6.0):
        """[B,C,H,W] at init + actual declared lead; no targets/baselines or fallback.

        ``default_lead_hours`` is API compatibility metadata, never a replacement
        for missing lead_time_hours. Actual leads may be positive multiples of 6h.
        """
        self._require_storage()
        if not torch.is_tensor(history) or history.ndim != 5 or min(history.shape) < 1 \
                or not history.is_floating_point() or history.layout != torch.strided:
            raise ValueError("history must be nonempty floating [B,T,C,H,W]")
        if history.device != self.table.device:
            raise ValueError("history and climatology table must be on the same device")
        if not bool(self.ready):
            raise RuntimeError("climatology anchor is unready; explicit install/load is required")
        if tuple(history.shape[2:]) != self._shape[1:]:
            raise ValueError("history channels/grid must match the climatology table_shape")
        if not torch.isfinite(history).all():
            raise ValueError("history must be finite")
        if not isinstance(batch, Mapping):
            raise TypeError("climatology batch must be a Mapping")
        if "atmos_baseline" in batch:
            raise ValueError("atmos_baseline is forbidden with a climatology anchor")
        missing = [name for name in _INPUT_FIELDS if name not in batch]
        if missing:
            raise KeyError(f"climatology anchor requires explicit {missing}; no fallback")
        if type(default_lead_hours) not in (int, float) or not math.isfinite(default_lead_hours) \
                or default_lead_hours <= 0 or default_lead_hours % 6 != 0:
            raise ValueError("default_lead_hours must declare a positive multiple of 6h")
        for name, size in (("latitude", self._shape[2]), ("longitude", self._shape[3])):
            coordinate = _numeric_field(batch[name], name, history)
            if tuple(coordinate.shape) not in ((size,), (history.shape[0], size)):
                raise ValueError(f"{name} must match the unbatched or collated history grid")
        year, day, hour, lead = [_numeric_field(batch[name], name, history, per_sample=True)
                                for name in _INPUT_FIELDS[2:]]
        latitude, longitude, _, _ = require_spacetime_fields(
            batch, history_shape=history.shape[-2:], batch_size=history.shape[0])
        for name, actual, expected in (("latitude", latitude, self._latitude),
                                       ("longitude", longitude, self._longitude)):
            declared = torch.tensor(expected, device=actual.device, dtype=actual.dtype)
            if not torch.equal(actual, declared):
                raise ValueError(f"{name} must exactly match the declared climatology grid in input dtype")
        if bool(((lead <= 0) | (lead.remainder(6) != 0)).any()):
            raise ValueError("lead_time_hours must be positive integer multiples of 6h")
        if bool(((hour < 0) | (hour >= 24) | (hour.remainder(6) != 0)).any()):
            raise ValueError("init_utc_hour must be an exact synoptic UTC hour (0,6,12,18)")
        valid_year, valid_day, valid_hour = advance_calendar_time(year, day, hour, lead)
        if bool((valid_hour.remainder(6) != 0).any()):
            raise ValueError("valid UTC hour must be exactly synoptic")
        month = _calendar_month(valid_year, valid_day)
        keys = torch.tensor(self._bucket_keys, device=history.device, dtype=torch.int64)
        matches = (month[:, None] == keys[None, :, 0]) & (valid_hour[:, None] == keys[None, :, 1])
        if not bool((matches.sum(dim=1) == 1).all()):
            raise ValueError("missing unique train-only climatology bucket; no held-out fill or fallback")
        result = self.table.index_select(0, matches.to(torch.int64).argmax(dim=1)).to(history.dtype)
        if not torch.isfinite(result).all():
            raise ValueError("climatology anchor is nonfinite in the history dtype")
        return result
