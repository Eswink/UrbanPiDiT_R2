"""Explicit, target-free case selection. Existing balanced defaults are unchanged."""
from __future__ import annotations
import pandas as pd
from data.preprocess.contracts import utc_time_index


def _stamp(value):
    if not isinstance(value, str):
        raise ValueError("timestamps must be explicit UTC-naive strings")
    try:
        t = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid timestamp") from exc
    if pd.isna(t) or t.tz is not None or t.minute or t.second or t.microsecond or t.nanosecond or t.hour % 6:
        raise ValueError("six-hour UTC-naive timestamp required")
    return t


def select_explicit_initializations(records, available_times, initializations):
    """Select exactly declared cases, ordered chronologically, or fail closed.

    Only record timestamps/split are read. Neither predictions nor target values
    are inspected. Every history and intermediate six-hour step through72h must
    exist within the same year, so a calendar gap cannot be silently bridged.
    """
    available = set(utc_time_index(available_times).asi8.tolist())
    if not records or len({r["split"] for r in records}) != 1 or records[0]["split"] not in ("val", "test"):
        raise ValueError("one nonempty held-out split required")
    by_id = {}
    for record in records:
        key = _stamp(record["init_time"])
        if key in by_id:
            raise ValueError("duplicate initialization in records")
        by_id[key] = record
    requested = [_stamp(v) for v in initializations]
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("explicit initializations must be nonempty and unique")
    selected = []
    for init in sorted(requested):
        if init not in by_id:
            raise ValueError("requested initialization absent from manifest")
        record = by_id[init]
        history = [_stamp(v) for v in record["history_times"]]
        if not history or history[-1] != init or any(b-a != pd.Timedelta(hours=6) for a,b in zip(history,history[1:])):
            raise ValueError("history must be consecutive and end at initialization")
        if _stamp(record["target_time"]) != init + pd.Timedelta(hours=6):
            raise ValueError("manifest must be a six-hour transition")
        required = history + [init + pd.Timedelta(hours=h) for h in range(0,73,6)]
        if not all(t.value in available and t.year == init.year for t in required):
            raise ValueError("explicit case lacks complete same-year history/72h trajectory")
        selected.append(dict(record))
    return selected
