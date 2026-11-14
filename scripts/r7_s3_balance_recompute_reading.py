"""Recompute a balance round's `diagnostics.json` from fields it already dumped.

**Why this exists.** A round's expensive half is the re-inference that writes
``seed{N}/{family}/{arm}/case_*.npz``. Its cheap half is arithmetic over those files. When the
arithmetic half has a defect, re-running the whole stage would discard tens of GPU-hours of
verified measurement to recompute numbers that the frozen receipts already pin by digest.

This script separates the two. It takes an attempt directory, **re-verifies every arm's field
files against the digest its own seed receipt recorded**, and only then computes the reading. It
records which code identity did the computing, because that identity is not the one the frozen
``protocol.json`` pinned -- and a reader must be able to see that, not have it hidden.

**What it refuses to do.**

- It will not write a reading if any arm's fields have drifted from its receipt. A single
  mismatched digest aborts the run, because a reading over changed fields is a different
  measurement wearing the old round's name.
- It will not re-check the execution-file hashes. That check exists so a *stage* cannot run code
  the freeze did not authorise; here the fields are the authorised artefact and the code is
  named explicitly in the output. Bypassing it silently would be the defect; naming it is the
  fix.
- It never infers, never opens a checkpoint and never touches the GPU.

**The record it leaves.** ``diagnostics.json`` plus ``reading_derivation.json`` holding the source
attempt, every arm's field digest against its receipt, the code commit and per-file hashes of the
code that computed it, and a plain statement that the frozen reading stage could not produce this
file and why.

Usage::

    .venv/bin/python scripts/r7_s3_balance_recompute_reading.py \\
        --attempt outputs/r7_s3_balance_diagnostics_20261114_attempt01
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fields_digest(files) -> str:
    """Exactly `scripts/r7_balance_field_dump._digest_of_listing`: name, NUL, file digest."""
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(_sha256(path).encode())
    return digest.hexdigest()


def verify_fields(attempt: Path) -> dict:
    """Every arm's field tree against the digest its seed receipt recorded at dump time."""
    receipts = {}
    for receipt_path in sorted(Path(attempt).glob("seed*_receipt.json")):
        seed = receipt_path.name.removeprefix("seed").removesuffix("_receipt.json")
        receipts[seed] = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not receipts:
        raise SystemExit(f"no seed*_receipt.json under {attempt}; nothing pins the fields")

    rows, failures = [], []
    for seed, receipt in sorted(receipts.items()):
        for arm, record in sorted((receipt.get("arms") or {}).items()):
            family, name = arm.split("/")
            directory = Path(attempt) / f"seed{seed}" / family / name
            files = sorted(directory.glob("case_*.npz"))
            if not files:
                failures.append(f"seed{seed}/{arm}: no case_*.npz under {directory}")
                continue
            actual = _fields_digest(files)
            recorded = record.get("fields_digest")
            rows.append({"arm": f"seed{seed}/{arm}", "n_files": len(files),
                         "recorded_fields_digest": recorded, "actual_fields_digest": actual,
                         "matches": bool(recorded == actual),
                         "checkpoint_sha256": record.get("checkpoint_sha256"),
                         "cases_total": record.get("cases_total")})
            if recorded != actual:
                failures.append(f"seed{seed}/{arm}: fields drifted from the receipt")
            if int(record.get("cases_total", 0)) != len(files):
                failures.append(f"seed{seed}/{arm}: {len(files)} files but the receipt says "
                                f"{record.get('cases_total')}")
    if failures:
        raise SystemExit("refusing to compute a reading over unverified fields:\n  "
                         + "\n  ".join(failures))
    return {"arms": rows, "n_arms": len(rows),
            "n_cases_per_arm": sorted({row["n_files"] for row in rows}),
            "all_fields_match_receipts": True}


def code_identity() -> dict:
    """The identity of the code doing the arithmetic, named so a reader can judge it."""
    def git(*arguments: str) -> str:
        return subprocess.run(["git", *arguments], cwd=str(ROOT), check=True,
                              capture_output=True, text=True, timeout=60).stdout.strip()

    from scripts.r7_s3_balance_protocol import ENTRY_FILES, execution_files
    return {"commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": git("status", "--porcelain"),
            "entry_files": list(ENTRY_FILES),
            "execution_files_sha256": {name: _sha256(ROOT / name)
                                       for name in execution_files()}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--attempt", required=True, type=Path)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing diagnostics.json (a re-derivation)")
    options = parser.parse_args(argv)

    from scripts import r7_balance_verdict as verdict
    from scripts.r7_s3_balance_protocol import validate_protocol
    from scripts.study_r7_s3_balance_round import readings

    attempt = Path(options.attempt).resolve()
    protocol_path = attempt / "protocol.json"
    body = json.loads(protocol_path.read_text(encoding="utf-8"))
    # The protocol's own digest still has to hold: this proves the frozen constants (arms, seeds,
    # leads, case count) are the ones that produced the fields.
    from training.r7_experiment import canonical_digest
    if body["protocol_sha256"] != canonical_digest(
            {key: value for key, value in body.items() if key != "protocol_sha256"}):
        raise SystemExit("protocol.json fails its own digest check")

    verification = verify_fields(attempt)
    identity = code_identity()
    frozen_driver = body["execution_files_sha256"].get("scripts/study_r7_s3_balance_round.py")
    current_driver = identity["execution_files_sha256"].get("scripts/study_r7_s3_balance_round.py")

    diagnostics = readings(attempt, body)
    diagnostics["reading_derivation"] = {
        "kind": "recomputed-from-verified-fields",
        "source_attempt": str(attempt),
        "protocol_sha256": body["protocol_sha256"],
        "frozen_driver_sha256": frozen_driver,
        "computing_driver_sha256": current_driver,
        "code_identity_differs_from_freeze": bool(frozen_driver != current_driver),
        "code": identity,
        "field_verification": verification,
        "statement": (
            "The GPU round's fields were dumped under the frozen protocol and every arm's digest "
            "matches the digest its seed receipt recorded. This reading was recomputed from "
            "those fields by the code identity below, which is not the identity the frozen "
            "protocol pinned: the frozen reading stage raised NameError before writing any "
            "diagnostics, so it could not produce this file. The fields are unchanged; only the "
            "arithmetic over them was re-run."),
    }
    out = attempt / "diagnostics.json"
    if out.exists() and not options.force:
        raise SystemExit(f"{out} exists; pass --force to record a re-derivation")
    out.write_text(json.dumps(diagnostics, indent=1, sort_keys=True, allow_nan=False),
                   encoding="utf-8")
    derivation = attempt / "reading_derivation.json"
    derivation.write_text(json.dumps(diagnostics["reading_derivation"], indent=2,
                                     sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps({"attempt": str(attempt), "n_arms": diagnostics["n_arms"],
                      "n_cases": diagnostics["n_cases"],
                      "code_identity_differs_from_freeze":
                          diagnostics["reading_derivation"]["code_identity_differs_from_freeze"],
                      "verdict": diagnostics["exits"]["verdict"],
                      "values": diagnostics["exits"]["values"],
                      "out": str(out)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
