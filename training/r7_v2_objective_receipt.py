"""Reconstruct serialized FP32 training objectives, not physical forecast metrics.

The archived _draft_loss explicitly computes FP32 even with BF16 autocast.
PyTorch evaluates two distinct operations: scalar multiply, then tensor add.
JSON floats retain each FP32 component exactly; binary64 algebra is not the
recorded operation and must not be used to relax metric/scientific tolerances.
"""
from __future__ import annotations

import math
import struct


def float32(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("finite numeric FP32 training component required")
    try:
        rounded = struct.unpack("!f", struct.pack("!f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError("training component exceeds finite FP32 range") from exc
    if not math.isfinite(rounded):
        raise ValueError("finite FP32 training component required")
    return rounded


def _component(value):
    rounded = float32(value)
    if value < 0 or value != rounded:
        raise ValueError("serialized training component must be nonnegative native FP32, not inferred precision")
    return rounded


def verify_training_objective(row, *, mode, controls, contract):
    """Exact IEEE binary32 receipt check, independent of torch/GPU and metrics close()."""
    if (mode not in ("l6", "two_step") or type(controls["bf16"]) is not bool
            or contract["bf16"] is not controls["bf16"]
            or contract["mode"] != mode
            or contract["autoregression"].get("objective") != "deep_supervised_latitude_area_mse"
            or contract["autoregression"].get("loss_space") != "normalized"
            or contract["autoregression"].get("internal_deep_supervision") is not True):
        raise ValueError("frozen FP32 deep-supervision recipe and actual autocast flags required")
    actual, l6 = _component(row["loss"]), _component(row["l6"])
    if mode == "two_step":
        weight = controls["lambda12"]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight < 0:
            raise ValueError("finite nonnegative frozen lambda required")
        l12 = _component(row["l12"])
        product = float32(float32(weight) * l12)
        expected = float32(l6 + product)
    else:
        if row["l12"] is not None:
            raise ValueError("L6 training cannot consume a future L12 loss")
        product, expected = None, l6
    if struct.pack("!f", actual) != struct.pack("!f", expected):
        raise ValueError("training objective FP32 multiply/add receipt bits mismatch")
    return {"dtype": "float32", "mode": mode, "expected_loss": expected,
            "rounded_weighted_l12": product, "bits_match": True,
            "metric_tolerance_changed": False, "scientific_claim": False}
