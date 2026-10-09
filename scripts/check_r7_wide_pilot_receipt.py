"""Read-only check of a wide-region part receipt against its frozen criteria.

The pilot's success criteria were frozen in ``acquisition_protocol.json`` before
any download. This tool evaluates them against the receipt that the download
actually wrote, and against the registered target-box part it names as the
control. It never decides by inspection: every criterion is a comparison of
recorded numbers, a missing field is a failure rather than a skip, and any
criterion that does not hold is reported as a deviation rather than being
dropped from the list.

Read-only, no network, no CUDA device.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_criteria(record, protocol, receipt, expected_channels):
    """Identity of the download itself: status, fallback, read plan, snapshot, channels."""
    record("receipt status is a completed real download",
           receipt.get("status") == "downloaded-real-source",
           receipt.get("status"), "downloaded-real-source")
    record("no synthetic fallback was used",
           receipt.get("synthetic_fallback") is False
           and receipt.get("source_is_real_reanalysis") is True,
           {"synthetic_fallback": receipt.get("synthetic_fallback"),
            "source_is_real_reanalysis": receipt.get("source_is_real_reanalysis")},
           {"synthetic_fallback": False, "source_is_real_reanalysis": True})
    record("the frozen read plan is the one the receipt recorded",
           receipt.get("read_plan_protocol_sha256") == protocol["read_plan"]["protocol_sha256"],
           receipt.get("read_plan_protocol_sha256"), protocol["read_plan"]["protocol_sha256"])
    record("the source snapshot is the pinned one",
           receipt.get("snapshot_id") == protocol["source"]["snapshot_id"],
           receipt.get("snapshot_id"), protocol["source"]["snapshot_id"])
    payloads = receipt.get("payloads", {})
    recorded_channels = receipt.get("channels", [])
    record("the recorded channel list is the frozen 17",
           sorted(recorded_channels) == expected_channels,
           sorted(recorded_channels), expected_channels)
    record("every frozen channel has a payload entry",
           sorted(payloads) == expected_channels, sorted(payloads), expected_channels)
    record("every stored channel payload is finite",
           bool(payloads) and all(value.get("finite") is True for value in payloads.values()),
           {name: value.get("finite") for name, value in payloads.items()}, "all true")
    record("payload hashes are recorded for every channel",
           bool(payloads) and all(len(value.get("payload_sha256", "")) == 64
                                  for value in payloads.values()),
           len(payloads), "one 64-hex digest per channel")


def _scope_criteria(record, receipt, scope):
    """Scope of the read: stamps, window, init hours, shape, level axis, channel count."""
    stamps = len(receipt.get("timestamps", []))
    timestamps = receipt.get("timestamps", [])
    record("stamp count matches the frozen scope",
           stamps == scope["stamps"], stamps, scope["stamps"])
    record("the first and last stamps are the frozen window",
           bool(timestamps) and timestamps[0] == scope["first_time"]
           and timestamps[-1] == scope["last_time"],
           [timestamps[0] if timestamps else None, timestamps[-1] if timestamps else None],
           [scope["first_time"], scope["last_time"]])
    record("all four UTC initialization hours are covered",
           receipt.get("init_hour_coverage") == {"0": 30, "6": 30, "12": 30, "18": 30},
           receipt.get("init_hour_coverage"), {"0": 30, "6": 30, "12": 30, "18": 30})
    record("the stored shape is the wide grid",
           receipt.get("shape") == [scope["stamps"], len(scope["levels_hpa"]), 129, 129],
           receipt.get("shape"), [scope["stamps"], len(scope["levels_hpa"]), 129, 129])
    record("the stored pressure axis is the planned one",
           receipt.get("stored_level_axis_hpa") == scope["levels_hpa"],
           receipt.get("stored_level_axis_hpa"), scope["levels_hpa"])
    record("the channel count is the frozen 17",
           receipt.get("channel_count") == scope["channels"],
           receipt.get("channel_count"), scope["channels"])


def _cost_criteria(record, receipt, control, budgets, declared, reference_sha256):
    """Measured cost against the registered target-box control and the frozen budgets."""
    measured = receipt.get("network_body_bytes")
    limit = 1.2 * control["network_body_bytes"]
    record("measured network bytes are within 20% of the target-box part",
           isinstance(measured, int) and measured <= limit,
           measured, f"<= {int(limit)} (registered {control['network_body_bytes']})")
    decoded = receipt.get("decoded_chunk_budget", {}).get("charged_bytes")
    control_decoded = control.get("decoded_charged_bytes",
                                  control.get("decoded_chunk_budget", {}).get("charged_bytes"))
    record("decoded charged bytes are within 1% of the target-box part",
           isinstance(decoded, int) and decoded <= 1.01 * control_decoded,
           decoded, f"<= {int(1.01 * control_decoded)} (registered {control_decoded})")
    elapsed = receipt.get("elapsed_seconds")
    record("elapsed seconds are inside the frozen per-part deadline",
           isinstance(elapsed, (int, float)) and elapsed <= budgets["deadline_seconds"],
           elapsed, f"<= {budgets['deadline_seconds']}")
    record("the control receipt file is the one the protocol named",
           reference_sha256 == declared["sha256"],
           reference_sha256, declared["sha256"])


def _load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def check(protocol, receipt, reference, *, protocol_sha256=None, reference_sha256=None):
    """Evaluate every frozen criterion against the given control receipt."""
    from data.download.read_plan_frozen import channel_plan

    scope = protocol["pilot_scope"]
    budgets = protocol["budgets"]
    control = reference
    declared = protocol["reference_part"]
    expected_channels = sorted(row["channel"] for row in channel_plan())
    results = []

    def record(name, ok, observed, expected):
        results.append({"criterion": name, "passed": bool(ok),
                        "observed": observed, "expected": expected})

    _source_criteria(record, protocol, receipt, expected_channels)
    _scope_criteria(record, receipt, scope)
    _cost_criteria(record, receipt, control, budgets, declared, reference_sha256)

    report = {
        "format": "r7-wide-region-pilot-criteria-v1",
        "scientific_claim": False,
        "protocol_sha256": protocol_sha256,
        "protocol_declared_sha256": protocol.get("protocol_sha256"),
        "reference_sha256": reference_sha256,
        "reference_declared_sha256": declared["sha256"],
        "criteria": results,
        "passed": sum(1 for entry in results if entry["passed"]),
        "total": len(results),
        "verdict": "pass" if all(entry["passed"] for entry in results) else "deviation",
        "deviations": [entry for entry in results if not entry["passed"]],
        "notes": [
            "the central-block identity criterion is checked by "
            "scripts/verify_r7_wide_target_block.py, which needs the written NetCDF",
            "a criterion that cannot be evaluated because a field is missing is a failure here, "
            "not a skip",
        ],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check a wide-region part receipt against its frozen pilot criteria.")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--report", help="new JSON report path")
    args = parser.parse_args()
    protocol_path, receipt_path, reference_path = (Path(args.protocol), Path(args.receipt),
                                                   Path(args.reference))
    protocol = _load(protocol_path)
    reference = _load(reference_path)
    report = check(protocol, _load(receipt_path), reference,
                   protocol_sha256=_sha256_file(protocol_path),
                   reference_sha256=_sha256_file(reference_path))
    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.report:
        path = Path(args.report)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing report: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
