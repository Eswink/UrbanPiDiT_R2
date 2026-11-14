"""P2 smoke preview: run the real verdict modules over a small field dump, cheaply.

This is **not** the round's reading and never decides anything. It exists so the whole chain --
dump, per-arm tables, five clauses -- can be exercised on a handful of cases before GPU hours are
committed, and so a reader can reproduce the P2 numbers quoted in
`docs/plans/0033-balance-diagnostics.md` §9.

It is deliberately a thin wrapper: the tables come from `scripts/r7_balance_verdict.py` and the
clauses from `scripts/r7_balance_exits.py`, the same modules the round's `reading` stage calls.
An earlier version of this file carried its own copy of the verdict with the arm comparison
hardcoded to "not available", which meant the smoke preview and the real reading could disagree;
one implementation is the only way that stays honest.

Input is a directory tree of ``seed{seed}/{family}/{arm}/case_*.npz`` written by
`scripts/r7_balance_field_dump.py`, exactly like the round's own layout, so a preview over one arm
and a preview over fifteen run the same code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import r7_balance_exits as exits  # noqa: E402
from scripts import r7_balance_verdict as verdict  # noqa: E402
from scripts import r7_s3_balance_protocol as proto  # noqa: E402


def discover(root: Path) -> dict:
    """Every arm present under ``seed*/family/arm``, in the same shape the round builds."""
    found = {}
    for seed_dir in sorted(Path(root).glob("seed*")):
        try:
            seed = int(seed_dir.name.removeprefix("seed"))
        except ValueError:
            continue
        for family_dir in sorted(path for path in seed_dir.iterdir() if path.is_dir()):
            for arm_dir in sorted(path for path in family_dir.iterdir() if path.is_dir()):
                if not sorted(arm_dir.glob("case_*.npz")):
                    continue
                found[f"{family_dir.name}/{arm_dir.name}/seed{seed}"] = {
                    "family": family_dir.name, "arm": arm_dir.name, "seed": seed,
                    "dir": str(arm_dir)}
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", required=True, type=Path,
                        help="root holding seed*/family/arm/case_*.npz")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-cases", type=int, default=10**9,
                        help="cap cases per arm, for a fast preview")
    options = parser.parse_args(argv)

    directories = discover(options.fields)
    if not directories:
        raise SystemExit(f"no seed*/family/arm/case_*.npz under {options.fields}")
    if options.max_cases < 10**9:
        trimmed = {}
        for arm_id, record in directories.items():
            subset = Path(record["dir"]).parent.parent / f".preview_{record['arm']}"
            subset.mkdir(parents=True, exist_ok=True)
            for path in sorted(Path(record["dir"]).glob("case_*.npz"))[:options.max_cases]:
                target = subset / path.name
                if not target.exists():
                    target.symlink_to(path.resolve())
            trimmed[arm_id] = {**record, "dir": str(subset)}
        directories = trimmed
        for record in directories.values():
            print(f"[preview] {record['dir']}: "
                  f"{len(list(Path(record['dir']).glob('case_*.npz')))} cases", flush=True)

    tables = verdict.table_of_arms(directories)
    families = {name: {"arms": list(spec["arms"]), "factor": spec["factor"],
                       "seeds": list(proto.SEEDS)}
                for name, spec in proto.FAMILIES.items()}
    judgement = exits.evaluate(tables, families, proto.LEADS, preflight_passed=True)
    report = {
        "format": "r7-s3-balance-diagnostics-preview-v1",
        "fields_dir": str(options.fields),
        "max_cases_per_arm": options.max_cases,
        "arms": tables,
        "exits": judgement,
        "note": ("preview only: a handful of cases, and (e) is asserted rather than run. The "
                 "round's reading stage is the only source of a verdict."),
        "scientific_claim": False,
        "test_read": False,
        "r": 0,
    }
    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text(json.dumps(report, indent=1, sort_keys=True, allow_nan=False),
                           encoding="utf-8")
    print(json.dumps({"n_arms": len(tables),
                      "values": judgement["values"], "verdict": judgement["verdict"],
                      "out": str(options.out)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
