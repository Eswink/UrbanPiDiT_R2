"""Read-only, exact +6/+12 h train windows with explicitly frozen exclusions.

Discovery reads timestamps/records only, never held-out fields. A missing +12 h
frame or split boundary is not an adjacent-index forecast and is not silently
skipped: call ``preflight_training_windows`` before freezing the protocol, then
pass its exact excluded sample IDs to the dataset (and use it in both arms).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .r7_store import HOUR_NS, normalization, split_time_labels, validate_record, validate_store
from .r7_zarr_dataset import ZarrAtmosWindowDataset


def _train_reader(manifest):
    path = Path(manifest)
    if path.name != "train.jsonl" or path.resolve().name != "train.jsonl":
        raise ValueError("only an explicit train.jsonl manifest is accepted; no sealed test read")
    reader = ZarrAtmosWindowDataset(path)
    if any(record.get("split") != "train" for record in reader.records):
        raise ValueError("autoregressive training requires train records only")
    stores = {(reader.manifest.parent / record["store_path"]).resolve()
              for record in reader.records}
    if len(stores) != 1:
        raise ValueError("one completed store per autoregressive train manifest required")
    return reader


def _discover_windows(reader):
    root = reader._store(reader.records[0])
    raw = validate_store(root)
    lookup = {int(stamp): index for index, stamp in enumerate(raw)}
    labels = split_time_labels(root)
    years = set(root.attrs["split_years"]["train"])
    windows, exclusions = [], []
    for position, record in enumerate(reader.records):
        indices = validate_record(root, record)
        if len(record["history_indices"]) != 2 or record["lead_time_hours"] != 6:
            raise ValueError("two history frames and a +6 h train target are required")
        init_ns = int(raw[indices[-2]])
        if init_ns - int(raw[indices[0]]) != 6 * HOUR_NS or init_ns % (6 * HOUR_NS):
            raise ValueError("history must be exact, UTC-aligned 6 h cadence")
        future_ns = init_ns + 12 * HOUR_NS
        future = lookup.get(future_ns)
        reason = None
        if future is None:
            reason = "missing_exact_t12"
        elif ((labels is not None and labels[future] != "train")
              or (labels is None and pd.Timestamp(future_ns).year not in years)):
            reason = "t12_outside_train_split"
        if reason is not None:
            exclusions.append({"sample_id": record["sample_id"], "init_time": record["init_time"],
                               "required_future_time": pd.Timestamp(future_ns).isoformat(),
                               "reason": reason})
            continue
        future_record = dict(record, target_index=future,
                             target_time=pd.Timestamp(future_ns).isoformat(), lead_time_hours=12)
        # Reuse the same index/time/split checks as the ordinary training loader.
        validate_record(root, future_record)
        windows.append((position, future_record))
    pins = [{"sample_id": reader.records[position]["sample_id"],
             "history_indices": record["history_indices"],
             "target_indices": [reader.records[position]["target_index"], record["target_index"]],
             "init_time": record["init_time"], "future_time": record["target_time"]}
            for position, record in windows]
    digest = hashlib.sha256(json.dumps(pins, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    summary = {"input_windows": len(reader), "usable_windows": len(windows),
               "excluded_sample_ids": [item["sample_id"] for item in exclusions],
               "exclusions": exclusions, "window_sha256": digest,
               "selection": "exact t-6,t,t+6,t+12, every frame owned by train; no index adjacency",
               "test_read": False, "state_fields_read": False}
    return windows, summary


def preflight_training_windows(manifest):
    """Return metadata-only discovery for protocol freeze; never authorize a skip."""
    _, summary = _discover_windows(_train_reader(manifest))
    return summary


class ZarrAutoregressiveDataset(Dataset):
    """Reuse the completed train loader; expose +12 h solely as ``future_target``.

    ``expected_exclusions`` must exactly match metadata-only discovery. The
    default is strict (no exclusions). A changed gap or split cannot silently
    change the train case set after freeze. Nonfinite fields always raise, even
    for an explicitly declared boundary exclusion elsewhere in the manifest.
    """

    def __init__(self, manifest, *, expected_exclusions=()):
        self.reader = _train_reader(manifest)
        self.windows, self.summary = _discover_windows(self.reader)
        if (isinstance(expected_exclusions, (str, bytes))
                or any(not isinstance(item, str) for item in expected_exclusions)):
            raise ValueError("expected_exclusions must be a collection of sample IDs")
        declared = list(expected_exclusions)
        if (len(set(declared)) != len(declared)
                or set(declared) != set(self.summary["excluded_sample_ids"])):
            raise ValueError("missing exact t+12/split boundary exclusions must be predeclared exactly: "
                             + json.dumps(self.summary["exclusions"], sort_keys=True))
        if not self.windows:
            raise ValueError("no complete exact two-step training windows")
        self.manifest = self.reader.manifest
        self.records = [self.reader.records[position] for position, _ in self.windows]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        position, future_record = self.windows[index]
        sample = self.reader[position]
        root = self.reader._store(future_record)
        future_index = validate_record(root, future_record)[-1]
        mean, std = normalization(root, count=root["state"].shape[1])
        field = np.asarray(root["state"][future_index], dtype=np.float32)
        future = (field - mean[:, None, None]) / std[:, None, None]
        if not np.isfinite(future).all():
            raise ValueError("nonfinite future atmospheric target")
        sample["future_target"] = torch.from_numpy(future)
        sample["init_time"] = future_record["init_time"]
        sample["future_valid_time"] = future_record["target_time"]
        # Explicit new producer metadata; never infer the year from old init_year.
        sample["init_calendar_year"] = torch.tensor(float(pd.Timestamp(future_record["init_time"]).year))
        return sample
