"""D2-analog on the wide-region (129x129) instance: same-data climatology and persistence.

The wide-region single-factor round needs the parameter-free references on the wide
val split, on exactly the cases a wide candidate will be scored on, and — because
the frozen target box is the central 65x65 of the 129 grid — the same references
restricted to that interior. ``evaluate_local`` already does both: with
``boundary_margins=(32,)`` it writes ``boundary_rmse.csv`` whose ``interior_32``
region is the central 65x65 block, for the model *and* for the climatology baseline,
so ``skill = 1 - (rmse_model / rmse_climatology)**2`` on the interior is a valid
paired MSE skill (identical region, cases and cosine-latitude weighting).

CPU-only, zero GPU. The protocol is frozen before any evaluation. The sealed test
split (2023) is never opened: only the train manifest (climatology fit) and the val
manifest (scoring) are read.
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

INSTANCE = ROOT / "outputs/r7_s3_wide_instance_v4_20261009_attempt01"
STORE = INSTANCE / "store/cache.zarr"
TRAIN_MANIFEST = INSTANCE / "store/manifests/train.jsonl"
VAL_MANIFEST = INSTANCE / "store/manifests/val.jsonl"
TEST_MANIFEST = INSTANCE / "store/manifests/test.jsonl"
EVALUATION_LEADS = (6, 12, 24, 48, 72)
BOUNDARY_MARGINS = (32,)  # 129x129 interior_32 == the frozen central 65x65 box
PLANNED_SECONDS_ROUND = 5400.0
HARD_CAP_SECONDS_ROUND = 14400.0
DEFAULT_OUT = ROOT / "outputs/r7_s3_wide_d2_baselines_20261009_attempt01"


def source_sha256():
    """The source the store was actually built from, read from the build's own record.

    The store records a full-local-file fingerprint of its source, so this pins the
    exact bytes without keeping a second copy of the 2 GB NetCDF in the instance dir.
    """
    report = json.loads((INSTANCE / "store/manifests/source_preflight.json").read_text(encoding="utf-8"))
    fingerprint = report["fingerprint"]
    if fingerprint.get("scope") != "full-local-file":
        raise ValueError("the store source fingerprint is not the full local file")
    return fingerprint["sha256"]


def protocol_constants():
    """The frozen fields that do not depend on the built store (unit-testable offline)."""
    return {
        "format": "r7-s3-wide-d2-baselines-protocol-v1",
        "stage": "S3-wide-D2",
        "objective": ("score the train-only climatology and persistence references on the "
                      "wide-region instance's val split (2022) at the five leads and all 17 "
                      "variables, on the full 129x129 grid and on the interior_32 region that "
                      "is the frozen central 65x65 target box, with the store/train identities "
                      "pinned before any evaluation"),
        "store": str(STORE.resolve()),
        "train_manifest": str(TRAIN_MANIFEST.resolve()),
        "val_manifest": str(VAL_MANIFEST.resolve()),
        "test_manifest_never_read": str(TEST_MANIFEST.resolve()),
        "region": {"points": [129, 129], "interior": "boundary_masks(129,129,(32,))['interior_32']"},
        "baselines": ["climatology", "persistence"],
        "evaluation": {
            "split": "val",
            "lead_hours": list(EVALUATION_LEADS),
            "history_steps": 2,
            "step_hours": 6,
            "boundary_margins": list(BOUNDARY_MARGINS),
            "max_samples": None,
            "device": "cpu",
            "case_selection": ("per-lead cohorts: each lead is scored on every complete val "
                               "rollout window for that lead, never an all-lead intersection"),
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
            "the wide region is a development instance; the frozen target box is unchanged",
        ],
    }


def protocol_payload():
    from training.r7_experiment import dataset_identity
    data_identity, _ = dataset_identity(TRAIN_MANIFEST)
    body = protocol_constants()
    body["source_sha256"] = source_sha256()
    body["data_identity"] = data_identity
    return body


def _score_one(lead, baseline, out):
    from training.r7_evaluate import evaluate_local
    from training.r7_arm_harness import sha256_file

    run_dir = out / baseline / f"lead_{lead:03d}h"
    report = evaluate_local(
        str(VAL_MANIFEST), output_dir=run_dir, baseline=baseline,
        lead_hours=(lead,), max_samples=10**9, device_name="cpu",
        boundary_margins=BOUNDARY_MARGINS)
    if report["split"] != "val":
        raise ValueError(f"{baseline} was scored on {report['split']}, not val")
    if report["n_evaluated"] != report["n_available_windows"]:
        raise ValueError(f"{baseline}@{lead}h covered {report['n_evaluated']} of "
                         f"{report['n_available_windows']} val windows")
    if report["climatology"]["training_years"] != [2017, 2018, 2019, 2020, 2021]:
        raise ValueError("the wide climatology is not fitted on the five train years")
    if report["boundary_scoring"] is None:
        raise ValueError("boundary scoring is absent; the interior_32 readout would be missing")
    return {
        "baseline": baseline, "lead_hours": lead, "split": report["split"],
        "n_evaluated": report["n_evaluated"], "n_available_windows": report["n_available_windows"],
        "channels": list(report["channels"]), "units": list(report["units"]),
        "elapsed_seconds": report["elapsed_seconds"],
        "climatology": {k: v for k, v in report["climatology"].items() if k != "bucket_counts"},
        "bucket_counts": report["climatology"]["bucket_counts"],
        "boundary_regions": report["boundary_scoring"]["regions"],
        "rmse_csv": str(run_dir / "rmse.csv"),
        "rmse_sha256": sha256_file(run_dir / "rmse.csv"),
        "boundary_rmse_csv": str(run_dir / "boundary_rmse.csv"),
        "boundary_rmse_sha256": sha256_file(run_dir / "boundary_rmse.csv"),
        "skill_csv": str(run_dir / "climatology_skill.csv"),
        "skill_sha256": sha256_file(run_dir / "climatology_skill.csv"),
    }


def main(argv=None):
    from scripts.r7_m3_offline import deny_network
    from training.r7_experiment import canonical_digest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    deny_network()

    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError(f"refusing existing output: {args.out}")
    if not STORE.is_dir():
        raise FileNotFoundError(f"the wide store is not built yet: {STORE}")
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

    results = {"format": "r7-s3-wide-d2-baselines-result-v1", "scientific_claim": False,
               "protocol_sha256": protocol["protocol_sha256"],
               "data_identity": protocol["data_identity"], "evaluation": {}}
    fitted = None
    for lead in EVALUATION_LEADS:
        for baseline in ("climatology", "persistence"):
            entry = _score_one(lead, baseline, args.out)
            fingerprint = json.dumps(entry["climatology"], sort_keys=True)
            if fitted is None:
                fitted = fingerprint
            elif fingerprint != fitted:
                raise ValueError("a baseline run fitted a different climatology")
            results["evaluation"][f"{baseline}@{lead}h"] = entry
            if time.perf_counter() - started > HARD_CAP_SECONDS_ROUND:
                raise TimeoutError("wide-D2 exceeded the frozen hard cap; aborting without a claim")
        print(json.dumps({"scored_lead": lead}), flush=True)
    results["elapsed_seconds_total"] = time.perf_counter() - started
    results["soft_overrun_seconds"] = max(0.0, results["elapsed_seconds_total"] - PLANNED_SECONDS_ROUND)
    with (args.out / "result.json").open("x", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"baselines_done": list(results["evaluation"]),
                      "elapsed_seconds": round(results["elapsed_seconds_total"], 1),
                      "soft_overrun_seconds": round(results["soft_overrun_seconds"], 1)}),
          flush=True)
    return results


if __name__ == "__main__":
    main()
