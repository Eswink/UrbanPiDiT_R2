"""Read-only exact 6 h train windows for directly supervised physical rollouts.

Preflight reads timestamps, records and static metadata, never state fields.
Every intermediate target must exist and belong to train; neither adjacent
indices nor an available terminal frame can substitute for a missing step.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .r7_autoregressive_dataset import _train_reader
from .r7_store import HOUR_NS, normalization, split_time_labels, validate_record, validate_store

STEP_HOURS = 6


def _require_physical_steps(physical_steps):
    if isinstance(physical_steps, bool) or not isinstance(physical_steps, int) or physical_steps < 1:
        raise ValueError("physical_steps must be a positive integer")
    return physical_steps


def _discover_long_windows(reader, physical_steps):
    root = reader._store(reader.records[0])
    raw = validate_store(root)
    lookup = {int(stamp): index for index, stamp in enumerate(raw)}
    labels = split_time_labels(root)
    years = set(root.attrs["split_years"]["train"])
    windows, exclusions, pins = [], [], []
    for position, record in enumerate(reader.records):
        indices = validate_record(root, record)
        if len(record["history_indices"]) != 2 or record["lead_time_hours"] != STEP_HOURS:
            raise ValueError("two history frames and a +6 h train target are required")
        init_ns = int(raw[indices[-2]])
        if init_ns - int(raw[indices[0]]) != STEP_HOURS * HOUR_NS or init_ns % (STEP_HOURS * HOUR_NS):
            raise ValueError("history must be exact, UTC-aligned 6 h cadence")
        targets, exclusion = [], None
        for step in range(1, physical_steps + 1):
            lead = step * STEP_HOURS
            stamp_ns = init_ns + lead * HOUR_NS
            target_index = lookup.get(stamp_ns)
            target_time = pd.Timestamp(stamp_ns).isoformat()
            reason = None
            if target_index is None:
                reason = f"missing_exact_t{lead}"
            elif ((labels is not None and labels[target_index] != "train")
                  or (labels is None and pd.Timestamp(stamp_ns).year not in years)):
                reason = f"t{lead}_outside_train_split"
            if reason is not None:
                if exclusion is None:
                    exclusion = {"sample_id": record["sample_id"], "init_time": record["init_time"],
                                 "physical_step": step, "lead_time_hours": lead,
                                 "required_target_time": target_time, "reason": reason}
                continue
            target = dict(record, target_index=target_index, target_time=target_time, lead_time_hours=lead)
            # Validate every available train target, even after an earlier gap.
            validate_record(root, target)
            targets.append(target)
        if exclusion is not None:
            exclusions.append(exclusion)
            continue
        windows.append((position, tuple(targets)))
        pins.append({"sample_id": record["sample_id"], "manifest_index": position,
                     "store_path": record["store_path"], "init_time": record["init_time"],
                     "history_indices": list(record["history_indices"]),
                     "history_times": list(record["history_times"]),
                     "target_indices": [target["target_index"] for target in targets],
                     "target_times": [target["target_time"] for target in targets]})
    identity = {"physical_steps": physical_steps, "step_hours": STEP_HOURS,
                "windows": pins, "exclusions": exclusions}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    summary = {"input_windows": len(reader), "usable_windows": len(windows),
               "excluded_sample_ids": [item["sample_id"] for item in exclusions],
               "exclusions": exclusions, "window_sha256": digest, "windows": pins,
               "physical_steps": physical_steps, "step_hours": STEP_HOURS,
               "selection": "exact t-6,t,t+6..t+6N, every frame owned by train; no index adjacency",
               "test_read": False, "state_fields_read": False}
    return windows, summary


def preflight_long_rollout_windows(manifest, *, physical_steps=12):
    """Return deterministic metadata discovery for a caller's protocol freeze."""
    steps = _require_physical_steps(physical_steps)
    _, summary = _discover_long_windows(_train_reader(manifest), steps)
    return summary


def _require_exclusions(expected_exclusions, summary):
    if isinstance(expected_exclusions, (str, bytes)):
        raise ValueError("expected_exclusions must be a collection of sample IDs")
    try:
        declared = list(expected_exclusions)
    except TypeError as error:
        raise ValueError("expected_exclusions must be a collection of sample IDs") from error
    if any(not isinstance(item, str) for item in declared):
        raise ValueError("expected_exclusions must be a collection of sample IDs")
    if len(set(declared)) != len(declared) or declared != summary["excluded_sample_ids"]:
        raise ValueError("long-rollout exclusions must be predeclared exactly in manifest order: "
                         + json.dumps(summary["exclusions"], sort_keys=True))


class ZarrLongRolloutDataset(Dataset):
    """Completed train samples plus normalized targets [N,C,H,W], strictly finite.

    The exclusion list is exact and ordered. An optional metadata digest pins the
    discovered histories, target indices/times and exclusions, not state bytes;
    source/store content identity remains the caller's separate freeze contract.
    """

    def __init__(self, manifest, *, physical_steps=12, expected_exclusions=(), expected_window_sha256=None):
        self.physical_steps = _require_physical_steps(physical_steps)
        self.reader = _train_reader(manifest)
        self.windows, self.summary = _discover_long_windows(self.reader, self.physical_steps)
        _require_exclusions(expected_exclusions, self.summary)
        if expected_window_sha256 is not None and expected_window_sha256 != self.summary["window_sha256"]:
            raise ValueError("long-rollout window_sha256 mismatch")
        if not self.windows:
            raise ValueError("no complete exact long-rollout training windows")
        self.manifest = self.reader.manifest
        self.records = [self.reader.records[position] for position, _ in self.windows]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        position, targets = self.windows[index]
        sample = self.reader[position]
        root = self.reader._store(targets[0])
        mean, std = normalization(root, count=root["state"].shape[1])
        fields = [sample["atmos_target"]]
        for record in targets[1:]:
            target_index = validate_record(root, record)[-1]
            field = np.asarray(root["state"][target_index], dtype=np.float32)
            normalized = (field - mean[:, None, None]) / std[:, None, None]
            if not np.isfinite(normalized).all():
                raise ValueError("nonfinite physical atmospheric target")
            fields.append(torch.from_numpy(normalized))
        sample["physical_targets"] = torch.stack(fields)
        if self.physical_steps >= 2:
            sample["future_target"] = sample["physical_targets"][1]
            sample["future_valid_time"] = targets[1]["target_time"]
        sample["init_time"] = targets[0]["init_time"]
        sample["physical_valid_times"] = [record["target_time"] for record in targets]
        # Explicit producer metadata; never recover this from legacy init_year.
        sample["init_calendar_year"] = torch.tensor(float(pd.Timestamp(sample["init_time"]).year))
        return sample
