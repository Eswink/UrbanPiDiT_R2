"""D3-analog of S3 on the v3 instance: same-data actual-C incumbent control.

The registered claim of this study is *screening*, not confirmation: retrain the
incumbent arm (process, mode='l6', 400 updates, three pre-declared seeds) on the
five-year train split (2017-2021) of the v3 instance and score it on 2022 val,
so the v3 budget-dose screen and any later candidate screen have a same-data,
same-budget control instead of the v2 single-year numbers. It decides nothing
about climatology superiority.

Fidelity to actual C is enforced on the CPU before any optimizer step:

- the initialized state of every seed must reproduce the archived actual-C
  pairing digest bitwise (``full_initial_state_sha256`` and the initialization
  report digest), and the model spec is the archived process spec verbatim;
- the training/evaluation path mirrors the archived worker: ``fine_tune`` with a
  frozen endpoint (no validation selection), then ``evaluate_local`` per lead
  without a scale sidecar. The instance's process-scale sidecar is recorded as
  instance context only: the actual-C process arm never trained or evaluated
  with process supervision.

Evaluation reuses ``evaluate_local`` per-lead so the incumbent is scored on
exactly the per-lead val cohorts the v3 baselines are scored on. The test split
(2023) is never opened by this study; GPU execution is co-resident, read-only
gated on the declared UUID with the frozen 2048+2048 MiB margin before every
spawn, and the whole round is deadline-bounded by frozen soft/hard seconds.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEEDS = (41, 42, 43)
UPDATES = 400
STEPS = 4
MODE = "l6"
LAMBDA12 = 0.5
LR = 1e-4
WARMUP = 20
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 1
CLIP = 1.0
CHECKPOINT_EVERY = 20
BF16 = False
EVALUATION_LEADS = (6, 12, 24, 48, 72)
REASONING_STEPS = 4
DEADLINE_SECONDS_PER_SEED = 3600.0
PLANNED_SECONDS_ROUND = 5400.0
HARD_CAP_SECONDS_ROUND = 10800.0

INSTANCE = ROOT / "outputs/r7_s3_confirmation_train2017_2021_v3"
STORE = INSTANCE / "store/cache.zarr"
TRAIN_MANIFEST = INSTANCE / "store/manifests/train.jsonl"
VAL_MANIFEST = INSTANCE / "store/manifests/val.jsonl"
PROCESS_SIDECAR = ROOT / "outputs/r7_s3_confirmation_train2017_2021_v3_process_scale"
D2_RESULT = ROOT / "outputs/r7_s3_v3_d2_baselines_20261006_attempt01/result.json"
DEFAULT_OUT = ROOT / "outputs/r7_s3_v3_d3_incumbent_20261006_attempt01"
GPU_UUID = "GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b"  # co-resident GPU0 at freeze time
HEADROOM_MARGIN_BYTES = 2048 * 2**20
ESTIMATED_PEAK_BYTES = 2048 * 2**20


def _shared():
    from training import r7_s3_v3_screen
    return r7_s3_v3_screen


def process_scale_sidecar_identity():
    metadata = json.loads((PROCESS_SIDECAR / "scale_metadata.json").read_text(encoding="utf-8"))
    return metadata["sidecar_identity"]


def protocol_payload():
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest, dataset_identity
    shared = _shared()
    spec, initialization, _ = shared.archived_process_spec()
    windows = shared.train_windows(TRAIN_MANIFEST)
    train_identity, _ = dataset_identity(TRAIN_MANIFEST)
    val_identity, _ = dataset_identity(VAL_MANIFEST)
    if train_identity != shared.V3_DATA_IDENTITY:
        raise RuntimeError("v3 train data identity differs from the frozen instance")
    return {
        "format": "r7-s3-v3-d3-incumbent-protocol-v1",
        "stage": "S3-V3-D3",
        "objective": ("retrain the actual-C process incumbent (mode l6, 400 updates, seeds "
                      "41/42/43) on the v3 confirmation instance's five-year train split "
                      "(2017-2021) and score it on the 2022 val split at all five leads, "
                      "providing the same-data control for the v3 budget-dose screen; frozen "
                      "endpoint, no validation selection, test never read"),
        "instance": str(INSTANCE.resolve()),
        "store": str(STORE.resolve()),
        "train_manifest": str(TRAIN_MANIFEST.resolve()),
        "val_manifest": str(VAL_MANIFEST.resolve()),
        "test_manifest": str((INSTANCE / "store/manifests/test.jsonl").resolve()),
        "source_sha256": shared.V3_SOURCE_SHA256,
        "code": shared.code_state(),
        "train_data_identity": train_identity,
        "val_data_identity": val_identity,
        "train_windows": windows,
        "climatology_training_years": list(shared.V3_CLIMATOLOGY_YEARS),
        "instance_sidecars": {
            "process_scale": {"path": str(PROCESS_SIDECAR.resolve()),
                              "identity": process_scale_sidecar_identity(),
                              "metadata_sha256": sha256_file(PROCESS_SIDECAR / "scale_metadata.json"),
                              "used_in_training_or_evaluation": False,
                              "note": ("actual-C process arm carries no process supervision; the "
                                       "sidecar is recorded for instance identity context only")}},
        "reference_readings": {
            "climatology_and_persistence_result_sha256": sha256_file(D2_RESULT),
            "note": ("the v3 baselines scored the same per-lead val cohorts with the same "
                     "evaluate_local path; this study records their identity only"),
        },
        "fidelity": {
            "archived_protocol_file_sha256":
                "57e659de40d9f0fcb6e28102e693e236692459f187175e5ab466f827857ca231",
            "check": ("before any optimizer step, every seed's (full_initial_state_sha256, "
                      "initialization_report_sha256) must equal the archived actual-C pairing "
                      "digest; a mismatch aborts the round"),
            "archived_pairing": {str(seed): shared.ARCHIVED_PAIRING[seed] for seed in SEEDS},
            "code_identity": ("current model/ training code; the v3 dataset and normalization "
                              "differ from the archived actual-C run by data only"),
        },
        "model": {"kind": "process", "spec": spec,
                  "spec_canonical_digest": canonical_digest(spec)},
        "initialization_canonical_digest": canonical_digest(initialization),
        "inference": {"reasoning_steps": REASONING_STEPS,
                      "note": "K4 probe on the same K4-trained checkpoint; not independent K training"},
        "seeds": list(SEEDS),
        "shared_controls": {"optimizer": f"AdamW lr {LR} weight_decay {WEIGHT_DECAY}",
                            "lr_schedule": (f"linear warmup {WARMUP} then cosine to 0.1 of peak at "
                                            f"{UPDATES} updates"),
                            "updates": UPDATES, "steps": STEPS, "mode": MODE, "lambda12": LAMBDA12,
                            "batch_size": BATCH_SIZE, "clip": CLIP, "bf16": BF16,
                            "checkpoint_every": CHECKPOINT_EVERY,
                            "objective": ("deep-supervised latitude-area normalized MSE over initial/all "
                                          "K drafts, final weight 2; full BPTT"),
                            "selection": "frozen endpoint; no validation selection"},
        "evaluation": {"split": "val", "lead_hours": list(EVALUATION_LEADS),
                       "expected_cohorts": dict(shared.V3_VAL_COHORTS),
                       "case_selection": "per-lead complete val cohorts; never an all-lead intersection",
                       "device": "cuda via evaluate_local on the declared GPU UUID"},
        "decision_rule": ("v3-D3 is a descriptive same-data control readout: report per-lead RMSE "
                          "and skill tables for all 17 variables; no pass/fail verdict, no "
                          "significance claim. The v3 budget-dose screen pairs its arms against "
                          "this same-data incumbent with that round's own pre-registered rule."),
        "test_read": False,
        "gpu": {"policy": "shared", "uuid": GPU_UUID, "estimated_peak_mib": 2048,
                "headroom_margin_mib": 2048,
                "gate": ("read-only free check before every seed: free >= max(estimated 2048 MiB, "
                         "owned reserved peak) + 2048 MiB; no neighbour signals or eviction"),
                "neighbor_policy": "nvidia-smi/torch read-only; no neighbor signals or eviction",
                "billing": "continuous first spawn through last owned reap; failures and cleanup charged"},
        "budgets": {"planned_seconds_round": PLANNED_SECONDS_ROUND,
                    "hard_cap_seconds_round": HARD_CAP_SECONDS_ROUND,
                    "deadline_seconds_per_seed": DEADLINE_SECONDS_PER_SEED,
                    "whole_round_scope": ("CPU fidelity checks, training, per-lead evaluation, "
                                          "aggregation; soft overrun continues and is recorded, only "
                                          "the hard cap truncates")},
        "test_read_policy": "the 2023 test split is never opened; only train (fit) and val (scoring)",
        "scientific_claim": False,
        "limitations": [
            "screening control only: one instance (2017-2021 train / 2022 val, one ROI, 17 "
            "channels); no significance, convergence or SOTA claim",
            "three seeds are consistency evidence, not a significance test",
            "K4 is an inference-depth probe on the same K4-trained checkpoint, not an independent model",
            "the incumbent is retrained, not imported: v2-instance numbers are a different instance "
            "and are not reused as this instance's control",
            "GPU runs are co-resident; latency/memory observations include neighbor load and are not "
            "a weather-distribution benchmark",
            "validation split only; the test split stays sealed until the S4 preregistered read",
        ],
    }


def run_seed(seed, output_dir, *, deadline, device_name):
    from training.r7_autoregressive_runner import fine_tune
    from training.r7_evaluate import evaluate_local
    from training.r7_experiment import dataset_identity
    from training.r7_arm_harness import sha256_file
    shared = _shared()
    if seed not in SEEDS:
        raise ValueError(f"declared seeds are {SEEDS}; {seed} is not among them")
    _, _, configuration = shared.archived_process_spec()
    model, initialization, resolved_spec, state_digest, report_digest = \
        shared.cpu_fidelity(seed, configuration)

    windows = shared.train_windows(TRAIN_MANIFEST)
    train_identity, _ = dataset_identity(TRAIN_MANIFEST)
    if train_identity != shared.V3_DATA_IDENTITY:
        raise RuntimeError("v3 train data identity differs from the frozen instance")
    contract = shared.contract_for(seed, resolved_spec, initialization,
                                   shared.seed_protocol_stamp(output_dir), train_identity,
                                   windows, source_sha256=shared.V3_SOURCE_SHA256,
                                   mode=MODE, lambda12=LAMBDA12, arm="process")
    train_dir = output_dir / f"seed{seed}" / "training" / "process"
    if train_dir.exists() or train_dir.is_symlink():
        raise FileExistsError(f"fresh training output only: {train_dir}")
    checkpoint, report = fine_tune(
        TRAIN_MANIFEST, train_dir, model=model, contract=contract, parent_weights=None,
        steps=STEPS, updates=UPDATES, seed=seed, mode=MODE, lr=LR, warmup=WARMUP,
        weight_decay=WEIGHT_DECAY, bf16=BF16, deadline=deadline, device_name=device_name,
        resume=None, checkpoint_every=CHECKPOINT_EVERY, batch_size=BATCH_SIZE, clip=CLIP,
        lambda12=LAMBDA12)
    if report["updates_this_run"] != UPDATES or report["selected_update"] != UPDATES:
        raise RuntimeError(f"seed {seed} did not reach the frozen endpoint")
    if report["data_identity"] != train_identity:
        raise RuntimeError(f"seed {seed} trained on a different data identity")
    receipts = {"seed": seed, "training_dir": str(train_dir), "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "training_report_sha256": sha256_file(train_dir / "training_report.json"),
                "initial_state_sha256": state_digest, "initialization_report_sha256": report_digest,
                "final_loss": report["losses"][-1]["loss"],
                "elapsed_seconds": report["elapsed_seconds"], "evaluations": {}}
    for lead in EVALUATION_LEADS:
        run_dir = output_dir / f"seed{seed}" / "evaluation" / "process" / f"lead_{lead:03d}h"
        if run_dir.exists() or run_dir.is_symlink():
            raise FileExistsError(f"fresh evaluation output only: {run_dir}")
        provenance = evaluate_local(
            str(VAL_MANIFEST), output_dir=run_dir, checkpoint=checkpoint,
            lead_hours=(lead,), max_samples=10**9, device_name=device_name,
            reasoning_steps=REASONING_STEPS, deadline=deadline)
        expected = shared.V3_VAL_COHORTS[str(lead)]
        if (provenance["split"] != "val"
                or "test_read" in provenance  # the legacy evaluator never knows about test at all
                or provenance["n_evaluated"] != expected
                or provenance["n_available_windows"] != expected
                or provenance["lead_hours"] != [lead]
                or provenance.get("process_scale_sidecar_identity") is not None):
            raise RuntimeError(f"seed {seed} lead {lead} provenance differs from the frozen expectation")
        if provenance["climatology"]["training_years"] != list(shared.V3_CLIMATOLOGY_YEARS):
            raise RuntimeError(f"seed {seed} lead {lead} climatology is not train-only 2017-2021")
        receipts["evaluations"][str(lead)] = {
            "dir": str(run_dir), "n_evaluated": provenance["n_evaluated"],
            "elapsed_seconds": provenance["elapsed_seconds"],
            "rmse_csv_sha256": sha256_file(run_dir / "rmse.csv"),
            "skill_csv_sha256": sha256_file(run_dir / "climatology_skill.csv"),
            "provenance_sha256": sha256_file(run_dir / "provenance.json")}
    return receipts


def main(argv=None):
    from scripts.r7_m3_offline import deny_network
    from training.r7_experiment import canonical_digest
    shared = _shared()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    deny_network()

    for manifest in (TRAIN_MANIFEST, VAL_MANIFEST):
        shared.refused_test_manifest(manifest)
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
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
    print(json.dumps({"frozen_protocol_sha256": protocol["protocol_sha256"],
                      "env": platform.platform()}), flush=True)

    results = {"format": "r7-s3-v3-d3-incumbent-result-v1", "scientific_claim": False,
               "test_read": False, "limitations": list(protocol["limitations"]),
               "protocol_sha256": protocol["protocol_sha256"],
               "train_data_identity": protocol["train_data_identity"],
               "val_data_identity": protocol["val_data_identity"],
               "gpu_gates": [], "seeds": {}}
    hard_deadline = started + HARD_CAP_SECONDS_ROUND
    shared.pin_declared_gpu(GPU_UUID)
    import torch
    torch.set_num_threads(4)
    owned_peak = 0
    first_spawn = last_reap = None
    try:
        for seed in SEEDS:
            gate = shared.gpu_gate(GPU_UUID, owned_reserved_peak_bytes=owned_peak,
                                   estimated_peak_bytes=ESTIMATED_PEAK_BYTES,
                                   margin_bytes=HEADROOM_MARGIN_BYTES)
            results["gpu_gates"].append({"seed": seed, **gate})
            seed_deadline = min(time.perf_counter() + DEADLINE_SECONDS_PER_SEED, hard_deadline)
            if first_spawn is None:
                first_spawn = time.perf_counter()
            results["seeds"][str(seed)] = run_seed(seed, args.out, deadline=seed_deadline,
                                                   device_name="cuda:0")
            last_reap = time.perf_counter()
            owned_peak = max(owned_peak, int(torch.cuda.max_memory_reserved(0)))
            with (args.out / "result.json").open("w", encoding="utf-8") as handle:
                json.dump(results, handle, indent=2, ensure_ascii=False, allow_nan=False)
            print(json.dumps({"seed_done": seed,
                              "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
            if time.perf_counter() > hard_deadline:
                raise TimeoutError("v3-D3 exceeded the frozen hard cap; aborting without a claim")
    except BaseException as exc:
        import traceback
        failure = {"status": "failed", "scientific_claim": False, "test_read": False,
                   "protocol_sha256": protocol["protocol_sha256"],
                   "elapsed_seconds": time.perf_counter() - started,
                   "failure_reason": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc(),
                   "seeds_completed": sorted(results["seeds"]),
                   "note": "partial evidence retained; this attempt is not resumed in place"}
        with (args.out / "failure.json").open("x", encoding="utf-8") as handle:
            json.dump(failure, handle, indent=2, ensure_ascii=False, allow_nan=False)
        raise
    results["elapsed_seconds_total"] = time.perf_counter() - started
    results["soft_overrun_seconds"] = max(0.0, results["elapsed_seconds_total"] - PLANNED_SECONDS_ROUND)
    results["first_spawn_to_last_reap_seconds"] = None if first_spawn is None else last_reap - first_spawn
    results["gpu_hours_conservative"] = results["elapsed_seconds_total"] / 3600.0
    with (args.out / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False, allow_nan=False)
    with (args.out / "attempt.json").open("x", encoding="utf-8") as handle:
        json.dump({"format": "r7-s3-v3-d3-attempt-v1", "status": "complete",
                   "scientific_claim": False, "protocol_sha256": protocol["protocol_sha256"],
                   "elapsed_seconds_total": results["elapsed_seconds_total"]}, handle,
                  indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"seeds_done": sorted(results["seeds"]),
                      "elapsed_seconds": round(results["elapsed_seconds_total"], 1),
                      "soft_overrun_seconds": round(results["soft_overrun_seconds"], 1)}), flush=True)
    return results


if __name__ == "__main__":
    main()
