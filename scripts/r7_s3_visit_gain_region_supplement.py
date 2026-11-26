"""Score the objective's literal region, ``interior_32``, on the frozen G1-B checkpoints.

Why this exists: the objective and decisions 0118/0036 declare exit 1 on **72 h
``interior_32`` t2m**, but the round froze ``boundary_margins=(1,)`` because on this 65x65
instance a 32-cell margin leaves a single grid point. ADR 0119 argued that
``full(65x65)`` is the wide instance's ``interior_32`` and froze ``full`` instead. That
substitution was recorded before any reading existed, but it is a change to the objective's
literal text, so the round's verdict currently rests on a revised criterion.

This script removes the dependence. It re-scores the six checkpoints the round already
produced, at the one lead the exit names, with ``boundary_margins=(1, 32)`` so the CSV
carries ``interior_32`` (1 point) beside ``full`` (4,225 points) and ``interior_1``
(3,969 points). If the literal region and the frozen region agree in sign, the exit 1
verdict holds under both readings and no criterion revision is load-bearing.

It trains nothing, evaluates one lead, and writes only to a new exclusive directory. The
frozen ``protocol.json`` is read, never rewritten: this is a supplementary measurement of
already-frozen checkpoints, not a new round.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ATTEMPT = ROOT / "outputs/r7_s3_visit_gain_20261126_attempt01"
PROTOCOL = ATTEMPT / "protocol.json"
LEAD = 72
MARGINS = (1, 32)


def _sha256(path: Path) -> str:
    from training.r7_arm_harness import sha256_file

    return sha256_file(Path(path))


def supplement(seed: int, arm: str, destination: Path, *, device: str, max_samples: int,
               deadline: float) -> dict:
    """Re-score one frozen checkpoint at 72 h with the literal and frozen regions together."""
    from scripts.r7_m3_offline import deny_network
    from training.r7_evaluate import EvaluationPreparation, evaluate_local

    deny_network()
    body = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt_path = ATTEMPT / f"seed{int(seed)}_{arm}_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    checkpoint = Path(receipt["checkpoint"])
    if _sha256(checkpoint) != receipt["checkpoint_sha256"]:
        raise RuntimeError(f"seed {seed} arm {arm}: checkpoint moved since its receipt")
    if int(receipt["updates"]) != int(body["updates"]):
        raise RuntimeError(f"seed {seed} arm {arm}: receipt is not at the frozen endpoint")
    prepared = EvaluationPreparation(Path(body["val_manifest"]), deadline=deadline)
    started = time.perf_counter()
    provenance = evaluate_local(
        str(body["val_manifest"]), output_dir=destination, checkpoint=str(checkpoint),
        lead_hours=(LEAD,), max_samples=int(max_samples), device_name=device,
        reasoning_steps=int(body["steps"]), boundary_margins=MARGINS, deadline=deadline,
        prepared=prepared)
    return {
        "scientific_claim": False,
        "test_read": False,
        "r": 0,
        "kind": "supplementary-region-reading",
        "why": ("the objective names interior_32 for exit 1; the round froze full(65x65) under "
                "ADR 0119. This re-scores the same checkpoint at the same lead with both region "
                "families so the exit can be read under either wording."),
        "seed": int(seed),
        "arm": arm,
        "visit_gain": receipt["visit_gain"],
        "lead_hours": LEAD,
        "margins_cells": list(MARGINS),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "parent_checkpoint_sha256": receipt["parent_checkpoint_sha256"],
        "protocol_sha256": body["protocol_sha256"],
        "train_data_identity": body["train_data_identity"],
        "val_data_identity": body["val_data_identity"],
        "val_manifest_sha256": body["val_manifest_sha256"],
        "n_evaluated": provenance["n_evaluated"],
        "n_available_windows": provenance["n_available_windows"],
        "evaluation_seconds": provenance["elapsed_seconds"],
        "wall_seconds": round(time.perf_counter() - started, 1),
        "destination": str(destination),
        "boundary_rmse_csv_sha256": _sha256(Path(destination) / "boundary_rmse.csv"),
        "provenance_sha256": _sha256(Path(destination) / "provenance.json"),
        "limitations": [
            "interior_32 on a 65x65 grid is one grid point (the centre cell), so its RMSE has a "
            "single spatial sample and is far noisier than the 4,225-point full region",
            "this scores an existing checkpoint; it trains nothing and cannot change any arm",
            "not a revision of any frozen criterion: both regions are reported side by side",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", required=True, choices=("visit_damped", "uniform_damped"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=10 ** 9)
    parser.add_argument("--deadline", type=float, default=None)
    arguments = parser.parse_args(argv)

    out = Path(arguments.out)
    destination = out / f"seed{arguments.seed}_{arguments.arm}_lead{LEAD:03d}h"
    receipt_out = out / f"seed{arguments.seed}_{arguments.arm}_supplement.json"
    if destination.exists() or receipt_out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {destination}")
    deadline = arguments.deadline or (time.perf_counter() + 6 * 3600.0)
    record = supplement(arguments.seed, arguments.arm, destination, device=arguments.device,
                        max_samples=arguments.max_samples, deadline=deadline)
    out.mkdir(parents=True, exist_ok=True)
    with receipt_out.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(f"[supplement] seed {arguments.seed} {arguments.arm} -> {receipt_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
