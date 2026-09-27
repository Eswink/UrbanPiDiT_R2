"""Narrow an experiment run's checkpoint footprint to its selected endpoints.

Why this exists. The campaign's new-artifact cap is 16 GiB; a four-arm,
three-seed, 800-update ablation writes a checkpoint every 100 updates, and the
intermediate ones are the bulk of it (5.7 GiB of checkpoints, of which 0.94 GiB
is the endpoints). The cap is a hard boundary and the authorization resolves a
breach by narrowing scope, so this tool narrows *storage* without touching the
evidence:

- only files matching ``update_%07d.pt`` **that are not** the ``selected_checkpoint``
  recorded in that arm's own ``training_report.json`` are removed;
- every evaluation directory, every ``rmse.csv``, every report, every protocol
  and every result JSON is left alone - the documented numbers are produced from
  the selected checkpoint, and its SHA256 is already recorded in the run result;
- the selected checkpoint is never removed, so every published number stays
  reproducible from the artifacts that remain.

It refuses to run without ``--apply``, prints every file it would remove with its
size, and writes a receipt so the removal is auditable after the fact rather than
being an unexplained drop in ``du``.

    python tools/prune_r7_run_checkpoints.py --run outputs/r7_65_c1
    python tools/prune_r7_run_checkpoints.py --run outputs/r7_65_c1 --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHECKPOINT_GLOB = "update_*.pt"


def plan_prune(run: Path):
    """Return (keep, remove) checkpoint paths for one run directory."""
    run = Path(run)
    if not run.is_dir():
        raise FileNotFoundError(run)
    keep, remove = [], []
    reports = sorted(run.rglob("training_report.json"))
    if not reports:
        raise ValueError(f"no training_report.json under {run}; refusing to guess")
    for report_path in reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        selected = report.get("selected_checkpoint")
        if not selected:
            raise ValueError(f"{report_path} records no selected_checkpoint")
        selected = Path(selected).resolve()
        arm_dir = report_path.parent
        checkpoints = sorted(arm_dir.glob(CHECKPOINT_GLOB))
        if not checkpoints:
            raise ValueError(f"{arm_dir} has no checkpoints to account for")
        if selected not in {path.resolve() for path in checkpoints}:
            raise ValueError(
                f"the selected checkpoint {selected} is not among the checkpoints in "
                f"{arm_dir}; refusing to prune an arm whose endpoint is missing")
        for path in checkpoints:
            (keep if path.resolve() == selected else remove).append(path)
    return keep, remove


def main():
    parser = argparse.ArgumentParser(
        description="Prune non-selected checkpoints from an R7 run (evidence-preserving).")
    parser.add_argument("--run", required=True, help="run directory to prune")
    parser.add_argument("--apply", action="store_true",
                        help="actually delete; without it this is a dry run")
    parser.add_argument("--receipt", default=None,
                        help="receipt path (default: <run>/checkpoint_prune_receipt.json)")
    args = parser.parse_args()

    run = Path(args.run)
    keep, remove = plan_prune(run)
    keep_bytes = sum(path.stat().st_size for path in keep)
    remove_bytes = sum(path.stat().st_size for path in remove)
    print(f"run: {run}")
    print(f"  keep   {len(keep):3d} checkpoints ({keep_bytes/2**20:9.1f} MiB)")
    print(f"  remove {len(remove):3d} checkpoints ({remove_bytes/2**20:9.1f} MiB)")
    for path in remove[:5]:
        print(f"    - {path.relative_to(run)} ({path.stat().st_size/2**20:.1f} MiB)")
    if len(remove) > 5:
        print(f"    ... and {len(remove) - 5} more")

    if not args.apply:
        print("dry run: nothing removed (pass --apply to delete)")
        return

    import hashlib

    def _sha256(path):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    receipt = {
        "format": "r7-checkpoint-prune-receipt-v1",
        "reason": ("narrow the campaign's new-artifact footprint to its selected "
                   "endpoints; every evaluation artifact, report, protocol and result "
                   "file is preserved, and the selected checkpoint of each arm is kept "
                   "so the published numbers stay reproducible"),
        "run": str(run),
        "kept": [{"path": str(path.relative_to(run)), "bytes": path.stat().st_size,
                  "sha256": _sha256(path)} for path in keep],
        "removed": [{"path": str(path.relative_to(run)), "bytes": path.stat().st_size,
                     "sha256": _sha256(path)} for path in remove],
        "removed_bytes": remove_bytes,
    }
    for path in remove:
        path.unlink()
    receipt_path = Path(args.receipt) if args.receipt \
        else run / "checkpoint_prune_receipt.json"
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"removed {len(remove)} checkpoints ({remove_bytes/2**20:.1f} MiB); "
          f"receipt at {receipt_path}")


if __name__ == "__main__":
    main()
