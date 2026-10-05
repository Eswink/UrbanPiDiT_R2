"""#79 registered reading and finalize driver (split from the seed runner).

Kept separate so both files stay inside the R-051 line cap; the protocol, arm
construction and per-seed runner live in scripts/study_r7_79_typed_evidence.py
and this module imports them rather than restating any of them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.study_r7_79_typed_evidence import (
    ARM_A, ARM_NAMES, COMPARATOR_DEPTH, DEADLINE_SECONDS_PER_SEED, EVALUATION_LEADS,
    HARD_CAP_SECONDS_ROUND, PLANNED_SECONDS_ROUND, PRIMARY_LEADS,
    PRIMARY_PAIR, PRIMARY_VARIABLE, SECONDARY_PAIR, SEEDS, UPDATES,
    run_seed,
)


def primary_reading(pairs):
    """The preregistered primary cells, read by the registered decision text."""
    def cell_outcomes(pair, variable, leads):
        block = pairs.get(f"{pair[0]} - {pair[1]}", {}).get("cells", {})
        cells, verdicts = {}, []
        for lead in leads:
            entry = block.get(f"{lead}h|{variable}")
            if entry is None or not entry["sign_consistent"]:
                cells[str(lead)] = {"delta_seed_mean": None, "outcome": "unresolved",
                                    "seed_deltas": ({} if entry is None
                                                    else dict(entry["seed_deltas"])),
                                    "reading": ("per-seed deltas disagree or the comparator "
                                                "returned no cell; no verdict and no seed "
                                                "mean under the frozen rule")}
                continue
            deltas = list(entry["seed_deltas"].values())
            delta = sum(deltas) / len(deltas)
            outcome = "supported" if delta < 0 else "worsened"
            cells[str(lead)] = {"delta_seed_mean": delta, "outcome": outcome,
                                "sign_consistent": True,
                                "seed_deltas": dict(entry["seed_deltas"]),
                                "reading": f"{outcome}: every seed agrees, delta {delta:+.6f}"}
            verdicts.append(outcome)
        return cells, verdicts

    typed_cells, typed_verdicts = cell_outcomes(PRIMARY_PAIR, PRIMARY_VARIABLE,
                                                PRIMARY_LEADS)
    overall_cells, overall_verdicts = cell_outcomes(SECONDARY_PAIR, PRIMARY_VARIABLE,
                                                    PRIMARY_LEADS)
    typed_supported = bool(typed_verdicts) and all(v == "supported" for v in typed_verdicts)
    overall_worsened = any(v == "worsened" for v in overall_verdicts)
    typed_worsened = any(v == "worsened" for v in typed_verdicts)
    if typed_supported and not overall_worsened:
        headline = "supported: the typed pair is supported on both leads and the overall pair does not worsen"
    elif typed_worsened:
        headline = "falsified: a registered typed-attribution lead moved the wrong way"
    elif not typed_verdicts:
        headline = "no sign-consistent typed-attribution cell"
    else:
        headline = "mixed/unresolved: the typed pair did not support on both leads"
    return {"primary_pair": list(PRIMARY_PAIR), "secondary_pair": list(SECONDARY_PAIR),
            "variable": PRIMARY_VARIABLE, "leads_hours": list(PRIMARY_LEADS),
            "typed_attribution": {"cells": typed_cells, "supported": typed_supported},
            "overall_pathway": {"cells": overall_cells,
                                "worsened": bool(overall_worsened)},
            "headline": headline,
            "reading": ("typed attribution requires both leads of typed_routing - "
                        "generic_fusion sign-consistent and negative with no worsened "
                        "lead on typed_routing - process_v2; a gain on the overall pair "
                        "alone is reported as an overall-pathway effect only")}


def main():
    """Seed mode runs one seed; finalize mode merges the declared seeds."""
    from scripts.r7_m3_offline import deny_network

    deny_network()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("seed", "finalize"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--manifests", type=Path,
                        default=ROOT / "outputs/r7_s1_seasons_2017/store/manifests")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/r7_79_typed_evidence_pilot")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--deadline-seconds", type=float, default=DEADLINE_SECONDS_PER_SEED)
    args = parser.parse_args()

    if args.mode == "seed":
        if args.seed is None or args.seed not in SEEDS:
            parser.error(f"--mode seed requires --seed in {SEEDS}")
        started = time.perf_counter()
        seed_dir = args.out / f"seed{args.seed}"
        attempt = {"format": "r7-79-typed-evidence-seed-attempt-v1", "seed": args.seed,
                   "scientific_claim": False, "test_read": False,
                   "planned_seconds_round": PLANNED_SECONDS_ROUND,
                   "hard_cap_seconds_round": HARD_CAP_SECONDS_ROUND,
                   "deadline_seconds_per_seed": args.deadline_seconds,
                   "started_perf_counter": started, "status": "running"}
        try:
            result = run_seed(args.manifests, seed_dir, seed=args.seed,
                              updates=args.updates, device_name=args.device,
                              deadline_seconds=args.deadline_seconds)
        except BaseException as exc:
            attempt.update({"status": "failed", "finalized": False,
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                            "elapsed_seconds": time.perf_counter() - started,
                            "no_retry_or_resurrection": True})
            if seed_dir.exists():
                (seed_dir / "attempt.json").write_text(
                    json.dumps(attempt, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")
            raise
        attempt.update({"status": "success", "finalized": True,
                        "elapsed_seconds": time.perf_counter() - started,
                        "protocol_sha256": result["protocol_sha256"],
                        "soft_overrun_seconds": max(0.0, time.perf_counter() - started
                                                    - PLANNED_SECONDS_ROUND / len(SEEDS))})
        (seed_dir / "attempt.json").write_text(
            json.dumps(attempt, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8")
        return 0

    from training.r7_arm_harness import (comparator_blocks, merge_seed_results,
                                         pair_cells, write_study_tables)

    merged = merge_seed_results(args.out, seeds=SEEDS, arms=ARM_NAMES,
                                fmt="r7-79-typed-evidence-three-arm-result-v1")
    identity = {"dataset_identity": merged["protocol"]["data"]["data_identity"],
                "typed_evidence_identity":
                    merged["protocol"]["data"]["typed_evidence_identity"],
                "model_code_sha256": merged["model_code_sha256"],
                "declared_update_budget": UPDATES, "evaluation_split": "val"}
    pairs_wanted = {PRIMARY_PAIR, SECONDARY_PAIR}
    table, blocks = comparator_blocks(merged, pairs=pairs_wanted, depth=COMPARATOR_DEPTH,
                                      identity=identity)
    pairs = pair_cells(blocks, leads=EVALUATION_LEADS)
    payload = {"format": "r7-79-typed-evidence-three-arm-comparison-v1",
               "scientific_claim": False, "identity": identity, "pairs": pairs,
               "primary": primary_reading(pairs),
               "rule": ("seed-paired: a cell is improved/worsened only when every "
                        "declared seed's delta agrees in sign; disagreements are "
                        "unresolved and are never averaged into a win"),
               "evidence_probe": {}, "table": table}
    probe_by_seed = {seed: json.loads((args.out / f"seed{seed}" / "seed_result.json")
                                      .read_text(encoding="utf-8"))["evidence_probe"]
                     for seed in merged["seeds"]}
    payload["evidence_probe"] = probe_by_seed
    with (args.out / "paired_comparison.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    write_study_tables(merged, args.out)
    print(json.dumps({
        "complete": True, "seeds": merged["seeds"], "scientific_claim": False,
        "protocol_sha256": merged["protocol_sha256"],
        "model_code_sha256": merged["model_code_sha256"],
        "gpu_hours_training": merged["budget"]["training_seconds_total"] / 3600.0,
        "gpu_hours_evaluation": merged["budget"]["evaluation_seconds_total"] / 3600.0,
        "primary": payload["primary"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
