"""Derive the 4-month control manifest with a complete 72 h horizon.

The 6 h control manifest also admits init times whose 72 h rollout targets fall on day 31
or in the next month. The long-rollout dataset reads those targets from the full-year
store, so the control arm would silently see other months for those windows. This script
writes a fresh exclusive subset whose ``t+6 .. t+72`` targets all satisfy the same
four-month, day 1-30, UTC 00/06/12/18 selection as the registered control.

No new frames are introduced: every retained record points at the already-preflighted
full-year store. The val/test files are copied byte for byte so both arms keep one split.
"""
from __future__ import annotations

import collections
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
INSTANCE = ROOT / "outputs/r7_s3_fy_annual_20261021_attempt01"
SOURCE = INSTANCE / "store/manifests_control"
TARGET = INSTANCE / "store/manifests_control_h72"
MONTHS = (1, 4, 7, 10)
HOURS = (0, 6, 12, 18)
TRAIN_YEARS = (2017, 2018, 2019, 2020, 2021)
PHYSICAL_STEPS = 12


def in_selection(stamp):
    """The registered four-month selection, applied to every rollout target."""
    return (stamp.month in MONTHS and 1 <= stamp.day <= 30 and stamp.hour in HOURS
            and stamp.year in TRAIN_YEARS)


def why_dropped(record):
    init = datetime.fromisoformat(record["init_time"])
    for step in range(1, PHYSICAL_STEPS + 1):
        target = init + timedelta(hours=6 * step)
        if not in_selection(target):
            return {"sample_id": record["sample_id"], "init_time": record["init_time"],
                    "physical_step": step, "target_time": target.isoformat(),
                    "reason": "target_outside_four_month_selection"}
    return None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if TARGET.exists() or TARGET.is_symlink():
        raise FileExistsError(f"fresh exclusive output only: {TARGET}")
    records = [json.loads(line) for line in
               (SOURCE / "train.jsonl").open(encoding="utf-8") if line.strip()]
    kept, dropped = [], []
    for record in records:
        reason = why_dropped(record)
        if reason is None:
            kept.append(record)
        else:
            dropped.append(reason)
    if not kept:
        raise RuntimeError("the 72 h control subset is empty")
    TARGET.mkdir(parents=True, exist_ok=False)
    for split in ("val", "test"):
        shutil.copyfile(SOURCE / f"{split}.jsonl", TARGET / f"{split}.jsonl")
    with (TARGET / "train.jsonl").open("x", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    marker = {"schema_version": 1, "build_complete": True, "scientific_claim": False,
              "selection": {"months": list(MONTHS), "days": [1, 30], "hours": list(HOURS),
                            "years": list(TRAIN_YEARS),
                            "horizon_hours": 6 * PHYSICAL_STEPS,
                            "rule": "history, init and every t+6..t+72 target in the selection"},
              "counts": {"input_6h_windows": len(records), "kept_72h_windows": len(kept),
                         "dropped_72h_windows": len(dropped)},
              "splits": {"train": len(kept), "val": sum(1 for _ in (TARGET / "val.jsonl").open()),
                         "test": sum(1 for _ in (TARGET / "test.jsonl").open())},
              "train_sha256": digest(TARGET / "train.jsonl"),
              "source_train_sha256": digest(SOURCE / "train.jsonl"),
              "val_sha256": digest(TARGET / "val.jsonl"),
              "test_sha256": digest(TARGET / "test.jsonl"),
              "drop_reasons": dict(collections.Counter(item["reason"] for item in dropped)),
              "source": str(SOURCE)}
    with (TARGET / "BUILD_COMPLETE.json").open("x", encoding="utf-8") as handle:
        json.dump(marker, handle, indent=2, ensure_ascii=False, allow_nan=False)
    with (TARGET / "dropped_windows.json").open("x", encoding="utf-8") as handle:
        json.dump(dropped, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(marker, ensure_ascii=False))


if __name__ == "__main__":
    main()
