"""#78 R-B registered reading and finalize driver (split from the seed runner).

Kept separate so both files stay inside the R-051 line cap; the protocol, arm
construction and per-seed runner live in scripts/study_r7_78_rb_loss.py and
this module imports them rather than restating any of them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.study_r7_78_rb_loss import (
    ARM_NAMES, COMPARATOR_DEPTH, EVALUATION_LEADS, HARD_CAP_SECONDS_ROUND,
    PLANNED_SECONDS_ROUND, PRIMARY_LEADS, PRIMARY_PAIR, PRIMARY_VARIABLE, SEEDS,
    UPDATES, primary_reading,
)


def main():
    from scripts.r7_m3_offline import deny_network
    from training.r7_arm_harness import (comparator_blocks, merge_seed_results,
                                         pair_cells, write_study_tables)

    deny_network()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/r7_78_rb_loss_pilot")
    args = parser.parse_args()

    merged = merge_seed_results(args.out, seeds=SEEDS, arms=ARM_NAMES,
                                fmt="r7-78-rb-loss-single-factor-result-v1")
    identity = {"dataset_identity": merged["protocol"]["data"]["data_identity"],
                "change_scale_identity":
                    merged["protocol"]["data"]["change_scale_identity"],
                "model_code_sha256": merged["model_code_sha256"],
                "declared_update_budget": UPDATES, "evaluation_split": "val"}
    table, blocks = comparator_blocks(merged, pairs={PRIMARY_PAIR}, depth=COMPARATOR_DEPTH,
                                      identity=identity)
    pairs = pair_cells(blocks, leads=EVALUATION_LEADS)
    probe_by_seed = {seed: json.loads((args.out / f"seed{seed}" / "seed_result.json")
                                      .read_text(encoding="utf-8"))["loss_weight_probe"]
                     for seed in merged["seeds"]}
    payload = {"format": "r7-78-rb-loss-single-factor-comparison-v1",
               "scientific_claim": False, "identity": identity, "pairs": pairs,
               "primary": primary_reading(pairs),
               "rule": ("seed-paired: a cell is improved/worsened only when every "
                        "declared seed's delta agrees in sign; disagreements are "
                        "unresolved and are never averaged into a win"),
               "loss_weight_probe": probe_by_seed,
               "mechanism_difference_from_ra": (
                   "R-A changed the decode with the objective fixed; R-B changes the "
                   "objective with the decode fixed at identity, so this round tests "
                   "the reweighting hypothesis and carries no claim from R-A."),
               "budgets": {"planned_seconds_round": PLANNED_SECONDS_ROUND,
                           "hard_cap_seconds_round": HARD_CAP_SECONDS_ROUND},
               "table": table}
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
