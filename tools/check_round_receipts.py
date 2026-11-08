"""Assert the round's receipts are complete, not merely present.

Why this exists. ``study_r7_s3_wide_single_factor.write_exclusive`` opens the target with
``open("x")`` and then calls ``json.dump(..., allow_nan=False)``. A non-finite value raises
*inside* the write, so the file is left **partially written** -- and for the reading stage
that half-file **is** the receipt (``r7_round_stages.STAGE_RECEIPTS["reading"]``), so the
wrapper skips the stage as already done and the round ends with no verdict at all.

Measured on this repository (2026-11-08): dumping ``{"a":1.0,"b":nan,"c":2.0}`` with those
options raises ``Out of range float values are not JSON compliant: nan`` after 21 bytes, and
the file remains on disk. The real inputs are finite -- 255 rows of a produced 17-variable x
5-lead table parse finite, and the five climatology denominators are finite and non-zero --
so this is a guard against a corrupt evaluation, not a claim that one is expected.

The two files that write the receipt are pinned in the running round's
``execution_files_sha256``, so this guard lives beside them rather than inside them: it is
read-only, takes no GPU, and cannot change any frozen identity.

    python tools/check_round_receipts.py --attempt outputs/<stem>_attemptNN
    python tools/check_round_receipts.py --attempt ... --require-seeds 41 42 43

Exit codes: 0 = every required receipt is present AND parses as complete JSON; 1 = a receipt
is missing or half-written; 2 = usage error.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

#: The stage receipts the orchestrator treats as "this stage is done".
STAGE_RECEIPTS = {
    "prepare": "prepared_protocol.json",
    "archive": "archive_receipt.json",
    "reading": "readings.json",
}


#: The keys ``readings()`` always emits. A truncated write can still land on a valid JSON
#: prefix (``{"scientific_claim": false`` parses if it happens to be closed), and a file that
#: merely parses is not a reading: it must carry the structure the exits are read from.
READINGS_REQUIRED_KEYS = ("scientific_claim", "test_read", "protocol_sha256",
                          "exit_conditions", "seeds")


def _check_one(path: Path, required_keys: tuple[str, ...] = ()) -> dict:
    """Is this file a complete JSON document, or a half-written one?"""
    record = {"path": str(path), "exists": path.is_file(), "ok": False, "reason": ""}
    if not record["exists"]:
        record["reason"] = "missing"
        return record
    text = path.read_text(encoding="utf-8", errors="replace")
    record["bytes"] = len(text.encode("utf-8"))
    try:
        body = json.loads(text)
    except json.JSONDecodeError as exc:
        # The signature of the partial-write failure: truncated at the point of the raise.
        record["reason"] = f"truncated or invalid JSON ({exc.msg} at char {exc.pos})"
        return record
    record["top_level_keys"] = sorted(body)[:12] if isinstance(body, dict) else []
    if required_keys:
        # Parseability is not completeness. Measured: a file containing only
        # {"scientific_claim": false} parsed fine and passed, even though a reading must carry
        # the exit_conditions the verdict is read from. The exits are never recomputed here --
        # this only refuses a document that cannot contain them.
        if not isinstance(body, dict):
            record["reason"] = "not a JSON object"
            return record
        missing = [key for key in required_keys if key not in body]
        if missing:
            record["reason"] = f"incomplete reading: missing keys {missing}"
            return record
        if not isinstance(body.get("exit_conditions"), dict):
            record["reason"] = "incomplete reading: exit_conditions is not an object"
            return record
    record["ok"] = True
    return record


def check(attempt: Path, seeds: list[int] | None, require_reading: bool) -> dict:
    """Every expected receipt exists and parses; a seed receipt must carry both arms."""
    attempt = Path(attempt)
    results = []
    for phase, name in STAGE_RECEIPTS.items():
        if phase == "reading" and not require_reading:
            continue
        # Only the reading has a required internal structure: the two protocol receipts are
        # validated by the driver itself, and demanding keys of them here would duplicate that.
        required = READINGS_REQUIRED_KEYS if phase == "reading" else ()
        results.append({"stage": phase, **_check_one(attempt / name, required)})
    for seed in seeds or []:
        entry = {"stage": f"seed{seed}", **_check_one(attempt / f"seed{seed}_receipt.json")}
        if entry.get("ok"):
            body = json.loads((attempt / f"seed{seed}_receipt.json").read_text(encoding="utf-8"))
            arms = body.get("arms") or {}
            entry["arms"] = sorted(arms)
            # A one-armed receipt is half a paired comparison: refuse it here rather than in
            # the reading, where the missing arm surfaces as a KeyError after hours of work.
            entry["ok"] = len(arms) == 2
            if not entry["ok"]:
                entry["reason"] = f"expected 2 arms, found {sorted(arms)}"
        results.append(entry)
    failed = [item for item in results if not item["ok"]]
    return {"attempt": str(attempt), "passed": not failed, "checked": len(results),
            "failed": failed, "results": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--require-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--require-reading", action="store_true",
                        help="also require readings.json to be a complete document")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if not args.attempt.is_dir():
        print(json.dumps({"error": f"not a directory: {args.attempt}"}))
        return 2
    report = check(args.attempt, args.require_seeds, args.require_reading)
    if args.quiet:
        print(json.dumps({k: report[k] for k in ("passed", "checked", "failed")},
                         ensure_ascii=False))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
