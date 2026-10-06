"""D2-analog of S3 on the v3 instance: same-data climatology and persistence.

This is the v3 rerun of the S3-D2 round: a CPU-only, zero-GPU bounded run that
freezes its protocol BEFORE any evaluation, then scores the two parameter-free
references the S3 gate needs on exactly the same cases:

- ``climatology``: train-only (month, hour) grid-cell mean fitted on the five
  v3 train years (2017-2021) only, fail-closed on missing buckets;
- ``persistence``: the last legal history frame, no training.

The evaluation reuses exactly the path the model comparisons use
(``evaluate_local``), so the reference RMSEs are on the identical val cases,
variables, units and spatial weighting as the v3 incumbent/candidate runs. No
test manifest is opened. No GPU device is touched.
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

INSTANCE = ROOT / "outputs/r7_s3_confirmation_train2017_2021_v3"
STORE = INSTANCE / "store/cache.zarr"
TRAIN_MANIFEST = INSTANCE / "store/manifests/train.jsonl"
VAL_MANIFEST = INSTANCE / "store/manifests/val.jsonl"
EVALUATION_LEADS = (6, 12, 24, 48, 72)
PLANNED_SECONDS_ROUND = 1800.0
HARD_CAP_SECONDS_ROUND = 3600.0
DEFAULT_OUT = ROOT / "outputs/r7_s3_v3_d2_baselines_20261006_attempt01"


def protocol_payload():
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest, dataset_identity
    from training.r7_s3_v3_screen import V3_DATA_IDENTITY, V3_SOURCE_SHA256
    data_identity, _ = dataset_identity(TRAIN_MANIFEST)
    if data_identity != V3_DATA_IDENTITY:
        raise RuntimeError("v3 train data identity differs from the frozen instance")
    return {
        "format": "r7-s3-v3-d2-baselines-protocol-v1",
        "stage": "S3-V3-D2",
        "objective": ("score the train-only climatology and persistence references on the "
                      "v3 confirmation instance's val split (2022) at the five leads and all "
                      "17 variables, on exactly the cases the v3 incumbent/candidate runs will "
                      "be scored on, with the store/train identities pinned before any "
                      "evaluation; the climatology is now fitted on five train years "
                      "(2017-2021) instead of the single S3-D2 year"),
        "store": str(STORE.resolve()),
        "train_manifest": str(TRAIN_MANIFEST.resolve()),
        "val_manifest": str(VAL_MANIFEST.resolve()),
        "data_identity": data_identity,
        "source_sha256": V3_SOURCE_SHA256,
        "baselines": ["climatology", "persistence"],
        "evaluation": {
            "split": "val",
            "lead_hours": list(EVALUATION_LEADS),
            "history_steps": 2,
            "step_hours": 6,
            "max_samples": None,  # full val coverage: every complete window is scored
            "device": "cpu",
            "case_selection": ("per-lead cohorts: each lead is scored on every complete val "
                               "rollout window for that lead (the same per-lead cohort rule "
                               "the confirmation gate uses), never an all-lead intersection"),
        },
        "budgets": {"planned_seconds": PLANNED_SECONDS_ROUND,
                    "hard_cap_seconds": HARD_CAP_SECONDS_ROUND},
        "test_read_policy": ("the test split (2023) is never opened by this study; only the "
                            "train manifest (for the climatology fit) and the val manifest "
                            "(for scoring) are read"),
        "scientific_claim": False,
        "limitations": [
            "parameter-free reference scoring only; it produces no model result and no skill claim",
            "the climatology is a train-only (month, hour) grid-cell mean, not a WeatherBench2 reproduction",
            "one ROI, five train years plus one eval year, 17 channels; val is 2022 only",
        ],
    }


def main(argv=None):
    from scripts.r7_m3_offline import deny_network
    from training.r7_arm_harness import sha256_file
    from training.r7_evaluate import evaluate_local
    from training.r7_experiment import canonical_digest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    deny_network()

    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError(f"refusing existing output: {args.out}")
    started = time.perf_counter()

    protocol = protocol_payload()
    protocol["protocol_sha256"] = canonical_digest(protocol)
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "protocol.json").open("x", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=False, allow_nan=False)
    frozen = json.loads((args.out / "protocol.json").read_text(encoding="utf-8"))
    if frozen["protocol_sha256"] != canonical_digest(
            {key: value for key, value in frozen.items() if key != "protocol_sha256"}):
        raise RuntimeError("the protocol on disk is not the one this run measured")
    print(json.dumps({"frozen_protocol_sha256": protocol["protocol_sha256"]}), flush=True)

    results = {"format": "r7-s3-v3-d2-baselines-result-v1", "scientific_claim": False,
               "protocol_sha256": protocol["protocol_sha256"],
               "data_identity": protocol["data_identity"], "evaluation": {}}
    fitted_climatology = None
    for lead in EVALUATION_LEADS:
        for baseline in ("climatology", "persistence"):
            run_dir = args.out / baseline / f"lead_{lead:03d}h"
            report = evaluate_local(
                str(VAL_MANIFEST), output_dir=run_dir, baseline=baseline,
                lead_hours=(lead,), max_samples=10**9,
                device_name="cpu")
            if report["split"] != "val":
                raise ValueError(f"{baseline} was scored on {report['split']}, not val")
            if report["n_evaluated"] != report["n_available_windows"]:
                raise ValueError(f"{baseline}@{lead}h covered {report['n_evaluated']} of "
                                 f"{report['n_available_windows']} val windows")
            climatology = {key: value for key, value in report["climatology"].items()
                           if key != "bucket_counts"}
            if report["climatology"]["training_years"] != [2017, 2018, 2019, 2020, 2021]:
                raise ValueError("v3 climatology is not fitted on the five train years")
            if fitted_climatology is None:
                fitted_climatology = json.dumps(climatology, sort_keys=True)
            elif json.dumps(climatology, sort_keys=True) != fitted_climatology:
                raise ValueError("a baseline run fitted a different climatology")
            results["evaluation"][f"{baseline}@{lead}h"] = {
                "baseline": baseline, "lead_hours": lead,
                "split": report["split"],
                "n_evaluated": report["n_evaluated"],
                "n_available_windows": report["n_available_windows"],
                "channels": list(report["channels"]), "units": list(report["units"]),
                "elapsed_seconds": report["elapsed_seconds"],
                "climatology": climatology,
                "bucket_counts": report["climatology"]["bucket_counts"],
                "rmse_csv": str(run_dir / "rmse.csv"),
                "skill_csv": str(run_dir / "climatology_skill.csv"),
                "rmse_sha256": sha256_file(run_dir / "rmse.csv"),
                "skill_sha256": sha256_file(run_dir / "climatology_skill.csv"),
            }
            if time.perf_counter() - started > HARD_CAP_SECONDS_ROUND:
                raise TimeoutError("v3-D2 exceeded the frozen hard cap; aborting without a claim")
        print(json.dumps({"scored_lead": lead}), flush=True)
    results["elapsed_seconds_total"] = time.perf_counter() - started
    results["soft_overrun_seconds"] = max(
        0.0, results["elapsed_seconds_total"] - PLANNED_SECONDS_ROUND)
    with (args.out / "result.json").open("x", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"baselines_done": list(results["evaluation"]),
                      "elapsed_seconds": round(results["elapsed_seconds_total"], 1),
                      "soft_overrun_seconds": round(results["soft_overrun_seconds"], 1)}),
          flush=True)
    return results


if __name__ == "__main__":
    main()
