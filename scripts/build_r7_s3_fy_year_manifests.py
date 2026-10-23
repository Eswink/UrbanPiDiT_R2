"""Derive both arm manifests for the training-year coverage round.

The store's own ``manifests`` directory covers every train year (2000-2021). The round
needs two read sets that differ in exactly one way -- the years they touch:

* ``manifests_treatment`` is that directory verbatim: every 2000-2021 window.
* ``manifests_control`` keeps only windows whose **entire** read set lies in 2017-2021:
  both history steps, the init time, and every ``t+6 .. t+72`` rollout target. A plain
  init-year filter would be wrong here: the first windows of 2017 read 2016 history
  frames, and the last windows of 2021 roll targets into 2022, so the control would
  silently train on years outside its declared range.

No new frames are introduced: every retained record points at the already-preflighted
store. The val/test files are copied byte for byte so both arms keep one split.
"""
from __future__ import annotations

import collections
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
INSTANCE = ROOT / "outputs/r7_s3_fy_multiyear_20261023_attempt01"
SOURCE = INSTANCE / "store/manifests"
CONTROL = INSTANCE / "store/manifests_control"
TREATMENT = INSTANCE / "store/manifests_treatment"
CONTROL_YEARS = (2017, 2018, 2019, 2020, 2021)
TREATMENT_YEARS = tuple(range(2000, 2022))
HOURS = (0, 6, 12, 18)
PHYSICAL_STEPS = 12
SPLITS = ("val", "test")


def _in_years(stamp, years):
    return stamp.year in years and stamp.hour in HOURS


def why_dropped(record, years=CONTROL_YEARS):
    """Name the first read-set member that falls outside the control's declared years."""
    init = datetime.fromisoformat(record["init_time"])
    for index, value in enumerate(record.get("history_times", ())):
        stamp = datetime.fromisoformat(value)
        if not _in_years(stamp, years):
            return {"sample_id": record["sample_id"], "init_time": record["init_time"],
                    "member": f"history_{index}", "member_time": stamp.isoformat(),
                    "reason": "history_outside_declared_years"}
    if not _in_years(init, years):
        return {"sample_id": record["sample_id"], "init_time": record["init_time"],
                "member": "init", "member_time": init.isoformat(),
                "reason": "init_outside_declared_years"}
    for step in range(1, PHYSICAL_STEPS + 1):
        target = init + timedelta(hours=6 * step)
        if not _in_years(target, years):
            return {"sample_id": record["sample_id"], "init_time": record["init_time"],
                    "member": f"t+{6 * step}h", "member_time": target.isoformat(),
                    "reason": "target_outside_declared_years"}
    return None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _records(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _write_lines(path, records):
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _marker(target, name, years, records, kept, dropped, source_sha):
    return {
        "schema_version": 1, "build_complete": True, "scientific_claim": False,
        "arm": name,
        "selection": {"years": list(years), "hours": list(HOURS),
                      "horizon_hours": 6 * PHYSICAL_STEPS,
                      "rule": "history, init and every t+6..t+72 target in the declared years"},
        "counts": {"input_windows": len(records), "kept_windows": len(kept),
                   "dropped_windows": len(dropped)},
        "splits": {split: len(_records(target / f"{split}.jsonl"))
                   for split in ("train", *SPLITS)},
        "train_sha256": digest(target / "train.jsonl"),
        "val_sha256": digest(target / "val.jsonl"),
        "test_sha256": digest(target / "test.jsonl"),
        "source_train_sha256": source_sha,
        "source": str(SOURCE),
        "drop_reasons": dict(collections.Counter(item["reason"] for item in dropped)),
    }


def _finish(target, name, years, records, kept, dropped, source_sha):
    target.mkdir(parents=True, exist_ok=False)
    for split in SPLITS:
        shutil.copyfile(SOURCE / f"{split}.jsonl", target / f"{split}.jsonl")
    _write_lines(target / "train.jsonl", kept)
    marker = _marker(target, name, years, records, kept, dropped, source_sha)
    if marker["splits"]["train"] != len(kept):
        raise RuntimeError(f"{name}: written train line count disagrees with the kept records")
    (target / "BUILD_COMPLETE.json").write_text(
        json.dumps(marker, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (target / "dropped_windows.json").write_text(
        json.dumps(dropped, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return marker


def main():
    for target in (CONTROL, TREATMENT):
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"fresh exclusive output only: {target}")
    records = _records(SOURCE / "train.jsonl")
    if not records:
        raise RuntimeError("the store's train manifest is empty")
    source_sha = digest(SOURCE / "train.jsonl")

    kept, dropped = [], []
    for record in records:
        reason = why_dropped(record)
        if reason is None:
            kept.append(record)
        else:
            dropped.append(reason)
    if not kept:
        raise RuntimeError("the control read set is empty")
    control = _finish(CONTROL, "control", CONTROL_YEARS, records, kept, dropped, source_sha)
    treatment = _finish(TREATMENT, "treatment", TREATMENT_YEARS, records, records, [], source_sha)

    if (CONTROL / "val.jsonl").read_bytes() != (TREATMENT / "val.jsonl").read_bytes():
        raise RuntimeError("the two val manifests are not byte-identical")
    if (CONTROL / "test.jsonl").read_bytes() != (TREATMENT / "test.jsonl").read_bytes():
        raise RuntimeError("the two test manifests are not byte-identical")
    print(json.dumps({"control": control["counts"], "treatment": treatment["counts"],
                      "val": control["splits"]["val"], "test": control["splits"]["test"],
                      "control_train_sha256": control["train_sha256"],
                      "treatment_train_sha256": treatment["train_sha256"]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
