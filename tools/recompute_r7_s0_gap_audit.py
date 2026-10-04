"""S0 read-only gap audit: CLI and CSV emission for the campaign S0 node.

Archive parsing, identity verification and the gap arithmetic live in
``tools/r7_s0_gap_audit_support.py``; this module owns argument handling, the
exclusive-write policy and the flat CSV table. Read-only: no training, no GPU,
no network, and no modification of any archive byte. See
``docs/R7_S0_INCUMBENT_GAP_AUDIT.md`` for the frozen contract.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

from tools.r7_s0_gap_audit_support import (  # noqa: F401 - re-exported for tests
    AGREEMENT_TOL, ARMS, DEFAULT_ARCHIVE, DEPTHS, LEADS, LIMITATIONS, REGIONS, SEEDS,
    VARIABLES, _audit, _gap_rows, _plain_path, _sha256, recompute,
)


def _format_gap_table(rows: list) -> str:
    fields = ["seed", "arm", "K", "lead_hours", "region", "variable", "model_rmse", "model_mse",
              "climatology_rmse", "climatology_mse", "persistence_rmse", "persistence_mse",
              "mse_skill_climatology", "mse_skill_climatology_status", "mse_skill_persistence",
              "mse_skill_persistence_status", "n_initializations"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--table", type=Path, help="exclusive new CSV path for the full gap table")
    parser.add_argument("--report", type=Path, help="exclusive new JSON path for the audit report")
    args = parser.parse_args(argv)
    try:
        report, rows = _audit(args.archive)
        protected = {report["script"]["path"]}
        if args.table is not None:
            from tools.r7_s0_gap_audit_support import _write_exclusive

            _write_exclusive(args.table, _format_gap_table(rows), protected)
            report["gap_table"] = {"path": str(_plain_path(args.table)),
                                   "sha256": _sha256(_plain_path(args.table)), "rows": len(rows)}
        if args.report is not None:
            from tools.r7_s0_gap_audit_support import _write_exclusive

            blocked = protected | ({str(_plain_path(args.table))} if args.table else set())
            _write_exclusive(args.report, json.dumps(report, indent=2, sort_keys=True) + "\n", blocked)
        if args.table is None and args.report is None:
            import sys

            sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        import sys

        print(f"S0 gap audit error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
