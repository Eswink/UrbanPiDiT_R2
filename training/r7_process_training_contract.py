"""Bind diagnostic supervision and its fixed atmospheric inverse to train data."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

from .r7_experiment import canonical_digest

WEIGHT_KEYS = ("input_diagnostic_weight", "future_diagnostic_weight", "draft_diagnostic_weight")


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a SHA256 digest")
    return value


def _root_inverse(root, data_identity):
    from data.r7_store import normalization
    mean, std = normalization(root, count=17)
    names = list(root.attrs["channels"])
    if len(names) != 17 or len(set(names)) != 17:
        raise ValueError("fixed train inverse requires 17 ordered unique channels")
    return {"channel_names": names, "normalization_mean": mean.tolist(),
            "normalization_std": std.tolist(), "data_identity": data_identity}


def _actual_train_inverse(metadata, data_identity):
    from .r7_experiment import dataset_identity
    _digest(data_identity, "data_identity")
    if metadata["data_identity"] != data_identity:
        raise ValueError("sidecar data identity differs from the training contract")
    actual, reader = dataset_identity(metadata["train_manifest"])
    if actual != data_identity:
        raise ValueError("actual training data/normalization identity differs from sidecar")
    paths = {(reader.manifest.parent / record["store_path"]).resolve() for record in reader.records}
    if paths != {Path(metadata["store"]).resolve()}:
        raise ValueError("sidecar store differs from the actual train manifest")
    return _root_inverse(reader._store(reader.records[0]), data_identity)


def _context_inverse(context, data_identity):
    return {"channel_names": list(context.channel_names),
            "normalization_mean": context.normalization_mean.detach().cpu().float().tolist(),
            "normalization_std": context.normalization_std.detach().cpu().float().tolist(),
            "data_identity": data_identity}


def supervision_training_contract(context, supplied, *, kind, process_weight, data_identity=None):
    if context is None and supplied is None:
        return None, {}
    if context is None or not isinstance(supplied, dict):
        raise ValueError("diagnostic supervision requires both context and contract")
    if kind != "process" or process_weight != 0:
        raise ValueError("diagnostic supervision requires a process model and legacy weight zero")
    required = {"sidecar_path", "sidecar_identity", "protocol_sha256", *WEIGHT_KEYS}
    if set(supplied) != required:
        raise ValueError("diagnostic supervision contract has missing or unexpected fields")
    from data.preprocess.r7_process_scale_sidecar import load_process_scale_sidecar
    metadata = load_process_scale_sidecar(Path(supplied["sidecar_path"]))
    identity = _digest(supplied["sidecar_identity"], "sidecar_identity")
    if metadata["sidecar_identity"] != identity:
        raise ValueError("diagnostic sidecar identity mismatch")
    if canonical_digest(context.metadata) != canonical_digest(metadata):
        raise ValueError("diagnostic context differs from the published sidecar")
    _digest(supplied["protocol_sha256"], "protocol_sha256")
    fixed = _actual_train_inverse(metadata, data_identity)
    if _context_inverse(context, data_identity) != fixed:
        raise ValueError("fixed train normalization/channel order differs from diagnostic context")
    weights = {}
    for name in WEIGHT_KEYS:
        value = supplied[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
        weights[name] = float(value)
    result = deepcopy(supplied)
    result.update(weights, fixed_train_context=fixed,
                  fixed_train_context_sha256=canonical_digest(fixed))
    result["sidecar_path"] = str(Path(result["sidecar_path"]).resolve())
    return result, {"process_supervision_context": context, **weights}


def verify_evaluation_sidecar(contract, path, *, evaluation_store=None, evaluation_root=None):
    supervision = contract.get("process_supervision")
    if supervision is None:
        if path is not None:
            raise ValueError("a sidecar override is not accepted for a legacy checkpoint")
        return None
    if path is None:
        raise ValueError("new diagnostic checkpoint evaluation requires its explicit scale sidecar")
    from data.preprocess.r7_process_scale_sidecar import load_process_scale_sidecar
    metadata = load_process_scale_sidecar(Path(path))
    if metadata["sidecar_identity"] != supervision["sidecar_identity"]:
        raise ValueError("checkpoint diagnostic sidecar identity mismatch")
    _digest(supervision["protocol_sha256"], "protocol_sha256")
    fixed = _actual_train_inverse(metadata, contract["data_identity"])
    if (supervision.get("fixed_train_context") != fixed
            or supervision.get("fixed_train_context_sha256") != canonical_digest(fixed)):
        raise ValueError("checkpoint fixed train context identity mismatch")
    if evaluation_store is None or evaluation_root is None:
        raise ValueError("new diagnostic evaluation requires explicit held-out store identity")
    if Path(evaluation_store).resolve() != Path(metadata["store"]).resolve():
        raise ValueError("held-out store differs from the trained sidecar store")
    if _root_inverse(evaluation_root, contract["data_identity"]) != fixed:
        raise ValueError("held-out normalization/channel order differs from fixed train context")
    return metadata["sidecar_identity"]
