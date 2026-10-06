"""S3 v3 budget-dose screen: l6 at 1600 updates against the v3 400-update incumbent.

The v3 rerun of the registered budget-curve question on the expanded instance
(train 2017-2021, 5x the single 2017 year). The v2 readings resolved toward
data: t2m improved monotonically through 1600 updates while the 48-72 h
u10/v10/mslp gate cells grew 13 -> 17, and the pre-declared budget-response
reading concluded the single-year budget was saturating. This round holds the
dose fixed at the final v2 dose (1600 l6 updates, 5% warmup) and changes only
the train data (five years), against a same-instance 400-update control
retrained on v3, so the primary and gate readings directly test whether the
data expansion clears the long-lead gate the budget extension could not.

The candidate is the same recipe as the v3-D3 incumbent (process, l6, same
spec, seeded initialization, v3 train split and per-lead val cohorts) run to
1600 updates. The control is the registered v3-D3 400-update run, pinned by
SHA256 and re-verified byte-for-byte at read time. The candidate spends 4x the
control's training updates; a positive verdict prices that training and is not
a compute-matched comparison.

Validation-only, offline, co-resident on the declared GPU UUID with a read-only
headroom gate before every spawn, deadline-bounded by frozen soft/hard seconds.
The 2023 test split is never opened.
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
CANDIDATE_MODE = "l6"
CANDIDATE_UPDATES = 1600
CONTROL_UPDATES = 400
CANDIDATE_LAMBDA12 = 0.5
STEPS = 4
LR = 1e-4
WARMUP = 80  # 5% of the 1600-update endpoint, matching the control's 20/400 convention
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 1
CLIP = 1.0
CHECKPOINT_EVERY = 20
BF16 = False
EVALUATION_LEADS = (6, 12, 24, 48, 72)
REASONING_STEPS = 4
PRIMARY_VARIABLE = "t2m"
PRIMARY_LEADS = (6, 12)
GATE_VARIABLES = ("u10", "v10", "mslp")
GATE_TOLERANCE = 0.0
DEADLINE_SECONDS_PER_SEED = 3600.0
PLANNED_SECONDS_ROUND = 6300.0
HARD_CAP_SECONDS_ROUND = 12600.0

INSTANCE = ROOT / "outputs/r7_s3_confirmation_train2017_2021_v3"
TRAIN_MANIFEST = INSTANCE / "store/manifests/train.jsonl"
VAL_MANIFEST = INSTANCE / "store/manifests/val.jsonl"
D3 = ROOT / "outputs/r7_s3_v3_d3_incumbent_20261006_attempt01"
D3_RESULT = D3 / "result.json"
DEFAULT_OUT = ROOT / "outputs/r7_s3_v3_budget_dose_20261006_attempt01"
GPU_UUID = "GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b"  # co-resident GPU0 at freeze time
HEADROOM_MARGIN_BYTES = 2048 * 2**20
ESTIMATED_PEAK_BYTES = 2048 * 2**20

PRIMARY_DECISION_TEXT = (
    "Read the per-seed paired RMSE delta of t2m/full at 6 h and 12 h for "
    "'l6_1600 - incumbent_l6_400' on the v3 instance's 2022 val cohorts. A lead reads "
    "'supported' when every declared seed agrees in sign and the candidate delta is "
    "negative (lower RMSE); 'worsened' when every seed agrees and the delta is positive; "
    "'unresolved' otherwise (disagreement is never averaged). The dose is supported only "
    "when BOTH 6 h and 12 h read supported. A supported primary with a passing gate "
    "pre-screen advances the 1600-update endpoint to the S4 freeze package; any other "
    "combination registers this round's result and the endpoint stops."
)
DATA_RESPONSE_TEXT = (
    "Data-response reading (pre-declared, reported alongside the verdict, never a "
    "pass/fail): compare this round's gate pre-screen with the registered v2-BC reading "
    "at the same 1600-update dose (17 positive cells at 48/72 h on the single 2017 train "
    "year). If the dose on five train years clears or strongly reduces the long-lead gate "
    "cells while the primary stays supported, the v2 budget-response conclusion is "
    "confirmed: the long-lead degradation was a data-scarcity artifact and the data "
    "expansion is the right capability investment. If the gate cells persist or worsen, "
    "the long-lead degradation is not fixed by train volume alone at this dose and the "
    "next question moves to what the long leads need (loss weighting, longer-lead "
    "supervision, or a dose still larger than 1600)."
)
GATE_DECISION_TEXT = (
    "Pre-screen (not part of the primary verdict): for u10/v10/mslp at every lead "
    "{6,12,24,48,72} h, the per-seed relative MSE change "
    "(candidate_mse - control_mse)/control_mse must be <= 0.0 on full-region "
    "cohorts. Any positive cell fails the pre-screen; the endpoint then does not "
    "advance to the S4 freeze even if the primary reads supported."
)
LIMITATIONS = [
    "screening only: one instance (2017-2021 train / 2022 val, one ROI, 17 channels), no "
    "significance, convergence or SOTA claim",
    "the control readings come from the registered v3-D3 incumbent run (400 l6 updates) "
    "on the same store and are pinned by SHA256, not retrained inside this protocol",
    "the candidate deliberately spends 4x the control's training updates and FLOPs; the "
    "verdict prices that extra training and is not a compute-matched comparison",
    "the 1600-update dose on five train years is about 0.7 epochs, against about 3.4 "
    "epochs on the single v2 train year; the round prices the declared dose, not an "
    "epoch-matched dose",
    "three seeds are consistency evidence, not a significance test",
    "K4 is an inference-depth probe on the same K4-trained checkpoint, not an "
    "independent model",
    "GPU runs are co-resident; latency/memory observations include neighbor load",
    "validation split only; the test split stays sealed until the S4 preregistered read",
]


def _shared():
    from training import r7_s3_v3_screen
    return r7_s3_v3_screen


def control_pins():
    """Pin the v3-D3 400-update control readings that this round compares against."""
    from training.r7_arm_harness import sha256_file
    shared = _shared()
    if not D3_RESULT.is_file():
        raise FileNotFoundError(f"v3-D3 control result missing: {D3_RESULT}")
    result = json.loads(D3_RESULT.read_text(encoding="utf-8"))
    if sorted(result["seeds"]) != [str(seed) for seed in SEEDS]:
        raise RuntimeError("v3-D3 control does not cover the declared seeds")
    pins = {"d3_result_sha256": sha256_file(D3_RESULT), "control_updates": CONTROL_UPDATES,
            "seeds": {}}
    for seed in SEEDS:
        receipts = result["seeds"][str(seed)]
        entry = {"checkpoint_sha256": receipts["checkpoint_sha256"],
                 "training_report_sha256": receipts["training_report_sha256"],
                 "initial_state_sha256": receipts["initial_state_sha256"], "leads": {}}
        for lead in EVALUATION_LEADS:
            record = receipts["evaluations"][str(lead)]
            if record["n_evaluated"] != shared.V3_VAL_COHORTS[str(lead)]:
                raise RuntimeError(f"v3-D3 seed {seed} lead {lead} cohort differs from the "
                                   f"frozen expectation")
            entry["leads"][str(lead)] = {"rmse_csv_sha256": record["rmse_csv_sha256"],
                                         "evaluation_dir": record["dir"]}
        pins["seeds"][str(seed)] = entry
    return pins


def protocol_payload(parity):
    from training.r7_experiment import canonical_digest, dataset_identity
    shared = _shared()
    spec, initialization, _ = shared.archived_process_spec()
    windows = shared.train_windows(TRAIN_MANIFEST)
    train_identity, _ = dataset_identity(TRAIN_MANIFEST)
    val_identity, _ = dataset_identity(VAL_MANIFEST)
    if train_identity != shared.V3_DATA_IDENTITY:
        raise RuntimeError("v3 train data identity differs from the frozen instance")
    return {
        "format": "r7-s3-v3-budget-dose-protocol-v1",
        "stage": "S3-V3-BD",
        "objective": ("budget-dose screen on the expanded v3 instance: the same process "
                      "model, initialization and v3 data as the v3-D3 incumbent, run to "
                      "1600 l6 updates, val-only three seeds; reads whether the data "
                      "expansion clears the long-lead gate that the same dose failed on the "
                      "single-year v2 instance"),
        "instance": str(INSTANCE.resolve()),
        "train_manifest": str(TRAIN_MANIFEST.resolve()),
        "val_manifest": str(VAL_MANIFEST.resolve()),
        "test_manifest": str((INSTANCE / "store/manifests/test.jsonl").resolve()),
        "source_sha256": shared.V3_SOURCE_SHA256,
        "code": shared.code_state(),
        "train_data_identity": train_identity,
        "val_data_identity": val_identity,
        "train_windows": windows,
        "arms": {
            "candidate": {"name": "l6_1600", "mode": CANDIDATE_MODE,
                          "updates": CANDIDATE_UPDATES, "lambda12": CANDIDATE_LAMBDA12,
                          "steps": STEPS, "lr": LR, "warmup": WARMUP,
                          "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
                          "clip": CLIP, "checkpoint_every": CHECKPOINT_EVERY, "bf16": BF16,
                          "selection": "frozen endpoint; no validation selection",
                          "model": {"kind": "process", "spec": spec,
                                    "spec_canonical_digest": canonical_digest(spec)}},
            "control": {"name": "incumbent_l6_400", "mode": "l6", "updates": CONTROL_UPDATES,
                        "source": "v3-D3 registered run on this same instance",
                        "pins": control_pins()}},
        "factor": {"name": "training update budget on the expanded instance",
                   "dose": 2,
                   "control_updates": CONTROL_UPDATES,
                   "prior_readings": [
                       {"updates": 800, "round": "S3-UB (v2)", "instance": "v2 single-year train",
                        "registered_verdict": "primary supported / gate failed 13 cells at 48-72 h"},
                       {"updates": 1600, "round": "S3-BC (v2)", "instance": "v2 single-year train",
                        "registered_verdict": ("primary supported / gate worsened to 17 cells; "
                                               "pre-declared reading: single-year budget saturating, "
                                               "next investment is data")}],
                   "candidate_updates": CANDIDATE_UPDATES,
                   "schedule": ("linear warmup of 5% of the frozen endpoint, then cosine to 0.1 "
                                "of peak at the endpoint; the convention is held fixed while the "
                                "instance data expands"),
                   "known_confound": ("the candidate trains 4x longer and spends 4x the training "
                                      "FLOPs; the verdict is about the recipe run to that budget "
                                      "on five train years, not a compute-matched arm"),
                   "data_response_text": DATA_RESPONSE_TEXT},
        "flop_probe": parity,
        "decision": {"primary": {"variable": PRIMARY_VARIABLE, "region": "full",
                                 "leads_hours": list(PRIMARY_LEADS),
                                 "reading": "l6_1600 - incumbent_l6_400",
                                 "text": PRIMARY_DECISION_TEXT},
                     "data_response": {"text": DATA_RESPONSE_TEXT},
                     "gate_pre_screen": {"variables": list(GATE_VARIABLES),
                                         "leads_hours": list(EVALUATION_LEADS),
                                         "relative_mse_change_max": GATE_TOLERANCE,
                                         "text": GATE_DECISION_TEXT},
                     "secondary": {"leads_hours": [24, 48, 72],
                                   "note": "reported for all 17 variables, never merged into the verdict"}},
        "gpu": {"policy": "shared", "uuid": GPU_UUID, "estimated_peak_mib": 2048,
                "headroom_margin_mib": 2048,
                "gate": ("read-only free check before every seed: free >= max(estimated 2048 MiB, "
                         "owned reserved peak) + 2048 MiB; no neighbour signals or eviction"),
                "neighbor_policy": "nvidia-smi/torch read-only; no neighbor signals or eviction",
                "billing": "continuous first spawn through last owned reap; failures and cleanup charged"},
        "budgets": {"planned_seconds_round": PLANNED_SECONDS_ROUND,
                    "hard_cap_seconds_round": HARD_CAP_SECONDS_ROUND,
                    "deadline_seconds_per_seed": DEADLINE_SECONDS_PER_SEED,
                    "whole_round_scope": ("CPU fidelity/FLOP checks, candidate training, per-lead "
                                          "evaluation, paired reading; soft overrun continues and is "
                                          "recorded, only the hard cap truncates")},
        "test_read": False,
        "test_read_policy": "the 2023 test split is never opened; only train (fit) and val (scoring)",
        "scientific_claim": False,
        "limitations": LIMITATIONS,
    }


def primary_verdict(all_cells):
    from training.r7_verdict_readings import per_seed_sign_verdict
    return per_seed_sign_verdict(all_cells, seeds=SEEDS, primary_leads=PRIMARY_LEADS,
                                 primary_variable=PRIMARY_VARIABLE)


def gate_verdict(all_cells):
    from training.r7_verdict_readings import gate_relative_mse_verdict
    return gate_relative_mse_verdict(all_cells, seeds=SEEDS,
                                     evaluation_leads=EVALUATION_LEADS,
                                     gate_variables=GATE_VARIABLES,
                                     tolerance=GATE_TOLERANCE)


def advance_decision(primary, gate):
    from training.r7_verdict_readings import advance_decision as shared_advance
    return shared_advance(primary, gate)


def run_seed(seed, output_dir, *, deadline, device_name):
    from training.r7_autoregressive_runner import fine_tune
    from training.r7_evaluate import evaluate_local
    from training.r7_experiment import dataset_identity
    from training.r7_arm_harness import sha256_file
    shared = _shared()
    if seed not in SEEDS:
        raise ValueError(f"declared seeds are {SEEDS}; {seed} is not among them")
    _, _, configuration = shared.archived_process_spec()
    model, initialization, spec, state_digest, report_digest = \
        shared.cpu_fidelity(seed, configuration)
    train_identity, _ = dataset_identity(TRAIN_MANIFEST)
    if train_identity != shared.V3_DATA_IDENTITY:
        raise RuntimeError("v3 train data identity differs from the frozen instance")
    contract = shared.contract_for(seed, spec, initialization,
                                   shared.seed_protocol_stamp(output_dir), train_identity,
                                   shared.train_windows(TRAIN_MANIFEST),
                                   source_sha256=shared.V3_SOURCE_SHA256,
                                   mode=CANDIDATE_MODE, lambda12=CANDIDATE_LAMBDA12,
                                   arm="candidate")
    train_dir = output_dir / f"seed{seed}" / "training" / "candidate"
    if train_dir.exists() or train_dir.is_symlink():
        raise FileExistsError(f"fresh training output only: {train_dir}")
    checkpoint, report = fine_tune(
        TRAIN_MANIFEST, train_dir, model=model, contract=contract, parent_weights=None,
        steps=STEPS, updates=CANDIDATE_UPDATES, seed=seed, mode=CANDIDATE_MODE, lr=LR,
        warmup=WARMUP, weight_decay=WEIGHT_DECAY, bf16=BF16, deadline=deadline,
        device_name=device_name, resume=None, checkpoint_every=CHECKPOINT_EVERY,
        batch_size=BATCH_SIZE, clip=CLIP, lambda12=CANDIDATE_LAMBDA12)
    if report["updates_this_run"] != CANDIDATE_UPDATES \
            or report["selected_update"] != CANDIDATE_UPDATES:
        raise RuntimeError(f"seed {seed} did not reach the frozen candidate endpoint")
    if not all(record["l12"] is None for record in report["losses"]):
        raise RuntimeError(f"seed {seed} trained with l12 supervision although mode is l6")
    if report["data_identity"] != train_identity:
        raise RuntimeError(f"seed {seed} trained on a different data identity")
    receipts = {"seed": seed, "training_dir": str(train_dir), "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "training_report_sha256": sha256_file(train_dir / "training_report.json"),
                "initial_state_sha256": state_digest, "initialization_report_sha256": report_digest,
                "final_loss": report["losses"][-1]["loss"],
                "loss_windows": shared.loss_windows(report),
                "elapsed_seconds": report["elapsed_seconds"], "evaluations": {}}
    for lead in EVALUATION_LEADS:
        run_dir = output_dir / f"seed{seed}" / "evaluation" / "candidate" / f"lead_{lead:03d}h"
        if run_dir.exists() or run_dir.is_symlink():
            raise FileExistsError(f"fresh evaluation output only: {run_dir}")
        provenance = evaluate_local(
            str(VAL_MANIFEST), output_dir=run_dir, checkpoint=checkpoint,
            lead_hours=(lead,), max_samples=10**9, device_name=device_name,
            reasoning_steps=REASONING_STEPS, deadline=deadline)
        expected = shared.V3_VAL_COHORTS[str(lead)]
        if (provenance["split"] != "val" or "test_read" in provenance
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
    from training.r7_arm_harness import sha256_file
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
    parity = shared.flop_probe(TRAIN_MANIFEST, candidate_updates=CANDIDATE_UPDATES,
                               control_updates=CONTROL_UPDATES, probe_seed=41)
    print(json.dumps({"flop_probe": parity}), flush=True)
    protocol = protocol_payload(parity)
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

    results = {"format": "r7-s3-v3-budget-dose-result-v1", "scientific_claim": False,
               "test_read": False, "limitations": LIMITATIONS,
               "protocol_sha256": protocol["protocol_sha256"],
               "train_data_identity": protocol["train_data_identity"],
               "val_data_identity": protocol["val_data_identity"],
               "flop_probe": parity, "gpu_gates": [], "seeds": {}}
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
                raise TimeoutError("v3-BD exceeded the frozen hard cap; aborting without a claim")
        pins = json.loads((args.out / "protocol.json").read_text(
            encoding="utf-8"))["arms"]["control"]["pins"]
        all_cells = {}
        for seed in SEEDS:
            control_dir = D3 / f"seed{seed}" / "evaluation" / "process"
            candidate_dir = args.out / f"seed{seed}" / "evaluation" / "candidate"
            for lead in EVALUATION_LEADS:
                expected_sha = pins["seeds"][str(seed)]["leads"][str(lead)]["rmse_csv_sha256"]
                actual_sha = sha256_file(control_dir / f"lead_{lead:03d}h" / "rmse.csv")
                if actual_sha != expected_sha:
                    raise RuntimeError(f"control CSV drifted from its pinned SHA256: "
                                       f"seed {seed} lead {lead}")
            all_cells[str(seed)] = shared.paired_cells(candidate_dir, control_dir,
                                                       EVALUATION_LEADS)
        results["paired_cells"] = all_cells
        results["primary_verdict"] = primary_verdict(all_cells)
        results["gate_pre_screen"] = gate_verdict(all_cells)
        results["decision"] = advance_decision(results["primary_verdict"],
                                               results["gate_pre_screen"])
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
    with (args.out / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False, allow_nan=False)
    with (args.out / "attempt.json").open("x", encoding="utf-8") as handle:
        json.dump({"format": "r7-s3-v3-budget-dose-attempt-v1", "status": "complete",
                   "scientific_claim": False, "protocol_sha256": protocol["protocol_sha256"],
                   "decision": results["decision"],
                   "elapsed_seconds_total": results["elapsed_seconds_total"]}, handle,
                  indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"verdict": results["primary_verdict"]["overall"],
                      "gate": results["gate_pre_screen"]["passed"],
                      "decision": results["decision"],
                      "elapsed_seconds": round(results["elapsed_seconds_total"], 1)}), flush=True)
    return results


if __name__ == "__main__":
    main()
