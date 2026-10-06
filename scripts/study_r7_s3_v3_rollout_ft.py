"""S3 v3 rollout fine-tune: a short low-LR two-step phase on the registered 1600 endpoint.

The v3 budget-dose round resolved negative: 1600 single-step (l6) updates on the
five-train-year instance kept the long-lead u10/v10/mslp gate at 17 positive
cells against the v3-D3 400-update control, exactly the v2-BC count at the same
dose. Published recursive weather models do not spend updates on rollout from
the start (the archived S3-D4 two-step arm did, at FLOP parity, and damaged the
short leads); the recipe attested in the literature is a long single-step phase
followed by a short, low-learning-rate multi-step fine-tune (GraphCast 11k
rollout updates on an LR floor, AIFS/FuXi curriculum, Stormer multi-step
fine-tuning credited with the long-lead gains). This round prices that shape at
the affordable dose: 200 two-step updates at one-fifth of the peak LR,
continued from the registered 1600-update endpoint, val-only, three seeds.

The verdict form is unchanged from every prior screening round of this node:
primary = per-seed paired sign of t2m/full at 6 h and 12 h against the pinned
v3-D3 400-update control; gate = u10/v10/mslp relative MSE change <= 0.0 at all
five leads; the conjunction is the advance rule. Reported alongside, never a
pass/fail, is the pre-declared rollout-response reading: a parent-relative
comparison against the registered v3-BD endpoint (which the candidate continues
from) on the same gate cells.

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
FT_MODE = "two_step"
FT_UPDATES = 200
FT_LAMBDA12 = 0.5
FT_LR = 2e-5  # one-fifth of the training peak; a short continuation, not a re-train
FT_WARMUP = 10  # 5% of the 200-update endpoint, matching the control's 20/400 convention
STEPS = 4
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
PLANNED_SECONDS_ROUND = 4200.0
# The round must fit the remaining campaign cap: 9.9238 GPU-h used at freeze,
# 12.0 cumulative, so the hard cap is set to 7200 s (2.0 h) and the worst-case
# cumulative stays below 12.0 (11.92). The hard cap truncates without a claim.
HARD_CAP_SECONDS_ROUND = 7200.0

INSTANCE = ROOT / "outputs/r7_s3_confirmation_train2017_2021_v3"
TRAIN_MANIFEST = INSTANCE / "store/manifests/train.jsonl"
VAL_MANIFEST = INSTANCE / "store/manifests/val.jsonl"
PARENT_RUN = ROOT / "outputs/r7_s3_v3_budget_dose_20261006_attempt01"
PARENT_RESULT_SHA256 = "61d3f1d3eedb5208da616457d4bff1a26a57a4c24a6470f595ebb5bb6cb060e4"
PARENT_ENDPOINT_UPDATES = 1600
PARENT_GATE_FAILURES_VS_CONTROL = 17  # registered v3-BD reading, bound before any training
PARENT_CHECKPOINT_SHA256 = {
    41: "637857c5d5cc8b3c9e33448cae4dd1b53aed5f525639d6911f03d7420492bdac",
    42: "8f7369fbbb3983b9e838d93a2e35d608871b01b580686a13581181950849007e",
    43: "08cc7db9629fca86f3634c0b102a206d292f09a5370a93d4a89d9477c6b50252",
}
D3 = ROOT / "outputs/r7_s3_v3_d3_incumbent_20261006_attempt01"
D3_RESULT = D3 / "result.json"
DEFAULT_OUT = ROOT / "outputs/r7_s3_v3_rollout_ft_20261006_attempt01"
GPU_UUID = "GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b"  # co-resident GPU0 at freeze time
HEADROOM_MARGIN_BYTES = 2048 * 2**20
ESTIMATED_PEAK_BYTES = 2048 * 2**20

PRIMARY_DECISION_TEXT = (
    "Read the per-seed paired RMSE delta of t2m/full at 6 h and 12 h for "
    "'rollout_ft - incumbent_l6_400' on the v3 instance's 2022 val cohorts. A lead reads "
    "'supported' when every declared seed agrees in sign and the candidate delta is "
    "negative (lower RMSE); 'worsened' when every seed agrees and the delta is positive; "
    "'unresolved' otherwise (disagreement is never averaged). The candidate is supported "
    "only when BOTH 6 h and 12 h read supported. A supported primary with a passing gate "
    "pre-screen advances the endpoint to the S4 freeze package; any other combination "
    "registers this round's result and the endpoint stops."
)
GATE_DECISION_TEXT = (
    "Pre-screen (not part of the primary verdict): for u10/v10/mslp at every lead "
    "{6,12,24,48,72} h, the per-seed relative MSE change "
    "(candidate_mse - control_mse)/control_mse must be <= 0.0 on full-region "
    "cohorts. Any positive cell fails the pre-screen; the endpoint then does not "
    "advance to the S4 freeze even if the primary reads supported."
)
ROLLOUT_RESPONSE_TEXT = (
    "Rollout-response reading (pre-declared, reported alongside the verdict, never a "
    "pass/fail): the candidate continues the registered v3-BD 1600-update endpoint, whose "
    "gate pre-screen against the same control failed with 17 positive cells (24 h 1, "
    "48 h 7, 72 h 9). Compare this round's gate pre-screen and parent-relative cells "
    "(candidate vs the registered v3-BD endpoints, same seeds/leads) with that reading. "
    "If the short low-LR two-step fine-tune clears or strongly reduces the long-lead gate "
    "cells while the primary stays supported, first-hand literature's 'rollout fine-tune "
    "after convergence' recipe transfers to this model class at this scale. If the gate "
    "cells persist or worsen, or short leads degrade, the negative is registered: at this "
    "dose and scale a short rollout phase does not fix the long-lead degradation, and the "
    "next question is a genuinely larger rollout budget or a different long-lead mechanism."
)

LIMITATIONS = [
    "screening only: one instance (2017-2021 train / 2022 val, one ROI, 17 channels), no "
    "significance, convergence or SOTA claim",
    "the control readings come from the registered v3-D3 incumbent run (400 l6 updates) "
    "on the same store and are pinned by SHA256, not retrained inside this protocol",
    "the candidate inherits the registered v3-BD 1600-update endpoint; the round prices "
    "the 200 two-step continuation updates on top of it, not the endpoint itself",
    "the rollout fine-tune dose (200 updates, LR 2e-5) is far below the published "
    "fine-tune budgets (thousands of updates on TPU/GPU fleets); a null result bounds "
    "this dose only",
    "three seeds are consistency evidence, not a significance test",
    "K4 is an inference-depth probe on the same K4-trained checkpoint, not an "
    "independent model",
    "GPU runs are co-resident; latency/memory observations include neighbor load",
    "validation split only; the test split stays sealed until the S4 preregistered read",
]


def _shared():
    from training import r7_s3_v3_screen
    return r7_s3_v3_screen


def _key_windows(report, key, size=50):
    values = []
    for row in report["losses"]:
        if key not in row or row[key] is None:
            raise RuntimeError(f"loss key {key} missing from the training report")
        values.append(row[key])
    windows = {}
    for start in range(0, len(values), size):
        segment = values[start:start + size]
        windows[f"{start + 1}-{start + len(segment)}"] = round(sum(segment) / len(segment), 6)
    return windows


def pinned_parent(seed):
    """Resolve and SHA-verify the registered parent checkpoint for one seed."""
    from training.r7_arm_harness import sha256_file
    if seed not in SEEDS:
        raise ValueError(f"declared seeds are {SEEDS}; {seed} is not among them")
    path = PARENT_RUN / f"seed{seed}" / "training" / "candidate" / \
        f"update_{PARENT_ENDPOINT_UPDATES:07d}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"registered v3-BD parent checkpoint missing: {path}")
    observed = sha256_file(path)
    if observed != PARENT_CHECKPOINT_SHA256[seed]:
        raise RuntimeError(f"v3-BD parent checkpoint drifted from its pinned SHA256: {path}")
    return path, observed


def parent_pins():
    """Freeze-time evidence that the parent run and endpoints are the registered ones."""
    from training.r7_arm_harness import sha256_file
    if not (PARENT_RUN / "result.json").is_file():
        raise FileNotFoundError(f"registered v3-BD result missing: {PARENT_RUN / 'result.json'}")
    result_sha = sha256_file(PARENT_RUN / "result.json")
    if result_sha != PARENT_RESULT_SHA256:
        raise RuntimeError("v3-BD result.json drifted from its pinned SHA256")
    result = json.loads((PARENT_RUN / "result.json").read_text(encoding="utf-8"))
    if len(result["gate_pre_screen"]["failures"]) != PARENT_GATE_FAILURES_VS_CONTROL:
        raise RuntimeError("v3-BD gate reading differs from the registered 17 cells")
    pins = {"parent_result_sha256": result_sha,
            "parent_updates": PARENT_ENDPOINT_UPDATES,
            "parent_gate_failures_vs_control": PARENT_GATE_FAILURES_VS_CONTROL,
            "checkpoints": {}}
    for seed in SEEDS:
        path, observed = pinned_parent(seed)
        pins["checkpoints"][str(seed)] = {"path": str(path), "sha256": observed}
    return pins


def control_pins():
    """Pin the v3-D3 400-update control readings that this round compares against."""
    from training.r7_arm_harness import sha256_file
    shared = _shared()
    if not D3_RESULT.is_file():
        raise FileNotFoundError(f"v3-D3 control result missing: {D3_RESULT}")
    result = json.loads(D3_RESULT.read_text(encoding="utf-8"))
    if sorted(result["seeds"]) != [str(seed) for seed in SEEDS]:
        raise RuntimeError("v3-D3 control does not cover the declared seeds")
    pins = {"d3_result_sha256": sha256_file(D3_RESULT), "control_updates": 400, "seeds": {}}
    for seed in SEEDS:
        receipts = result["seeds"][str(seed)]
        entry = {"initial_state_sha256": receipts["initial_state_sha256"], "leads": {}}
        for lead in EVALUATION_LEADS:
            record = receipts["evaluations"][str(lead)]
            if record["n_evaluated"] != shared.V3_VAL_COHORTS[str(lead)]:
                raise RuntimeError(f"v3-D3 seed {seed} lead {lead} cohort differs from the "
                                   f"frozen expectation")
            entry["leads"][str(lead)] = {"rmse_csv_sha256": record["rmse_csv_sha256"],
                                         "evaluation_dir": record["dir"]}
        pins["seeds"][str(seed)] = entry
    return pins


def flop_probe(train_manifest, *, probe_seed=41):
    """Per-update objective cost for the one-step control and the two-step arm."""
    import torch
    from torch.utils.data import default_collate
    from torch.utils.flop_counter import FlopCounterMode
    from data.r7_autoregressive_dataset import ZarrAutoregressiveDataset
    from training.r7_autoregressive_rollout import training_one_step, training_two_step
    from training.r7_v2_profile import seeded_mapped_model
    torch.set_num_threads(4)
    shared = _shared()
    windows = shared.train_windows(train_manifest)
    dataset = ZarrAutoregressiveDataset(train_manifest,
                                        expected_exclusions=windows["excluded_sample_ids"])
    probe = default_collate([dataset[0]])
    _, _, configuration = shared.archived_process_spec()
    model, _, _ = seeded_mapped_model(configuration, probe_seed, "process")
    model.train()
    measured = {}
    for name, objective in (("one_step", training_one_step), ("two_step", training_two_step)):
        model.zero_grad(set_to_none=True)
        with torch.enable_grad(), FlopCounterMode(display=False) as counter:
            output = objective(model, probe, reasoning_steps=REASONING_STEPS)
            output.loss.backward()
            measured[name] = int(counter.get_total_flops())
        model.zero_grad(set_to_none=True)
    return {"probe_sample_id": probe["sample_id"][0],
            "one_step_forward_backward_flops": measured["one_step"],
            "two_step_forward_backward_flops": measured["two_step"],
            "fine_tune_updates": FT_UPDATES, "control_updates": 400,
            "fine_tune_training_flops": FT_UPDATES * measured["two_step"],
            "control_training_flops": 400 * measured["one_step"],
            "flop_convention": ("supported aten operations under enable_grad; "
                                "elementwise/normalization omitted")}


def protocol_payload(parity):
    from training.r7_experiment import canonical_digest, dataset_identity
    shared = _shared()
    spec, _, _ = shared.archived_process_spec()
    train_identity, _ = dataset_identity(TRAIN_MANIFEST)
    val_identity, _ = dataset_identity(VAL_MANIFEST)
    if train_identity != shared.V3_DATA_IDENTITY:
        raise RuntimeError("v3 train data identity differs from the frozen instance")
    return {
        "format": "r7-s3-v3-rollout-ft-protocol-v1",
        "stage": "S3-V3-RFT",
        "objective": ("short low-LR two-step rollout fine-tune continued from the registered "
                      "v3-BD 1600-update endpoint: does the published 'rollout fine-tune "
                      "after convergence' shape clear the 48-72 h u10/v10/mslp gate cells "
                      "that 1600 single-step updates on five train years left at 17"),
        "instance": str(INSTANCE.resolve()),
        "train_manifest": str(TRAIN_MANIFEST.resolve()),
        "val_manifest": str(VAL_MANIFEST.resolve()),
        "test_manifest": str((INSTANCE / "store/manifests/test.jsonl").resolve()),
        "source_sha256": shared.V3_SOURCE_SHA256,
        "code": shared.code_state(),
        "train_data_identity": train_identity,
        "val_data_identity": val_identity,
        "train_windows": shared.train_windows(TRAIN_MANIFEST),
        "arms": {
            "candidate": {"name": "rollout_ft_200", "mode": FT_MODE, "updates": FT_UPDATES,
                          "lambda12": FT_LAMBDA12, "steps": STEPS, "lr": FT_LR,
                          "warmup": FT_WARMUP, "weight_decay": WEIGHT_DECAY,
                          "batch_size": BATCH_SIZE, "clip": CLIP,
                          "checkpoint_every": CHECKPOINT_EVERY, "bf16": BF16,
                          "selection": "frozen endpoint; no validation selection",
                          "imports": "registered v3-BD 1600-update endpoint per seed",
                          "model": {"kind": "process", "spec": spec,
                                    "spec_canonical_digest": canonical_digest(spec)}},
            "control": {"name": "incumbent_l6_400", "mode": "l6", "updates": 400,
                        "source": "v3-D3 registered run on this same instance",
                        "pins": control_pins()},
            "parent": {"name": "l6_1600", "mode": "l6", "updates": PARENT_ENDPOINT_UPDATES,
                       "source": "v3-BD registered run on this same instance",
                       "pins": parent_pins()}},
        "factor": {"name": "short low-LR two-step rollout fine-tune on the converged endpoint",
                   "literature_basis": [
                       {"source": "GraphCast (arXiv 2212.12794, accessed 2026-10-06)",
                        "note": "299k single-step updates then ~11k rollout updates N=2..12 on an LR floor"},
                       {"source": "Stormer (arXiv 2312.03876, accessed 2026-10-06)",
                        "note": "single-step pretraining then multi-step fine-tune; long-lead gain credited to it"},
                       {"source": "Keisler (arXiv 2202.07575, accessed 2026-10-06)",
                        "note": "small-budget 4/8/12-step curriculum rollout; rollout-only loss slightly worse"}],
                   "known_confound": ("the candidate inherits the registered 1600-update endpoint; the "
                                      "verdict prices the fielded recipe (1600 l6 + 200 two-step) vs the "
                                      "400-update control, not the fine-tune in isolation"),
                   "dose": ("200 two-step updates, LR 2e-5 (one-fifth of the training peak), 5% warmup; "
                            "far below the published fine-tune budgets"),
                   "rollout_response_text": ROLLOUT_RESPONSE_TEXT},
        "flop_probe": parity,
        "decision": {"primary": {"variable": PRIMARY_VARIABLE, "region": "full",
                                 "leads_hours": list(PRIMARY_LEADS),
                                 "reading": "rollout_ft - incumbent_l6_400",
                                 "text": PRIMARY_DECISION_TEXT},
                     "rollout_response": {"text": ROLLOUT_RESPONSE_TEXT},
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
                    "campaign_cap_note": ("campaign used 9.9238 GPU-h at freeze, cumulative cap 12.0; "
                                          "the 7200 s hard cap keeps the worst case at 11.92"),
                    "whole_round_scope": ("CPU FLOP probe, parent import, two-step fine-tune, per-lead "
                                          "evaluation, paired readings; soft overrun continues and is "
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
    from training.r7_experiment import dataset_identity, load_checkpoint
    from training.r7_arm_harness import sha256_file
    from training.r7_v2_profile import seeded_mapped_model
    shared = _shared()
    parent_path, parent_sha = pinned_parent(seed)
    saved = load_checkpoint(parent_path)
    parent_state = saved["model"]
    _, _, configuration = shared.archived_process_spec()
    model, _, spec = seeded_mapped_model(configuration, seed, "process")
    train_identity, _ = dataset_identity(TRAIN_MANIFEST)
    if train_identity != shared.V3_DATA_IDENTITY:
        raise RuntimeError("v3 train data identity differs from the frozen instance")
    initialization = {"parent_run": str(PARENT_RUN), "parent_checkpoint": str(parent_path),
                      "parent_checkpoint_sha256": parent_sha,
                      "parent_endpoint_updates": PARENT_ENDPOINT_UPDATES,
                      "description": ("registered v3-BD l6 endpoint imported as the strict "
                                      "model state dict; fresh optimizer, no resume")}
    contract = shared.contract_for(seed, spec, initialization,
                                   shared.seed_protocol_stamp(output_dir), train_identity,
                                   shared.train_windows(TRAIN_MANIFEST),
                                   source_sha256=shared.V3_SOURCE_SHA256,
                                   mode=FT_MODE, lambda12=FT_LAMBDA12, arm="candidate")
    train_dir = output_dir / f"seed{seed}" / "training" / "candidate"
    if train_dir.exists() or train_dir.is_symlink():
        raise FileExistsError(f"fresh training output only: {train_dir}")
    checkpoint, report = fine_tune(
        TRAIN_MANIFEST, train_dir, model=model, contract=contract, parent_weights=parent_state,
        steps=STEPS, updates=FT_UPDATES, seed=seed, mode=FT_MODE, lr=FT_LR, warmup=FT_WARMUP,
        weight_decay=WEIGHT_DECAY, bf16=BF16, deadline=deadline, device_name=device_name,
        resume=None, checkpoint_every=CHECKPOINT_EVERY, batch_size=BATCH_SIZE, clip=CLIP,
        lambda12=FT_LAMBDA12)
    if report["updates_this_run"] != FT_UPDATES or report["selected_update"] != FT_UPDATES:
        raise RuntimeError(f"seed {seed} did not reach the frozen fine-tune endpoint")
    if any(record["l12"] is None for record in report["losses"]):
        raise RuntimeError(f"seed {seed} did not supervise l12 although mode is two_step")
    if report["data_identity"] != train_identity:
        raise RuntimeError(f"seed {seed} trained on a different data identity")
    receipts = {"seed": seed, "training_dir": str(train_dir), "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "training_report_sha256": sha256_file(train_dir / "training_report.json"),
                "parent_checkpoint_sha256": parent_sha,
                "imported_initial_weights_sha256": report["contract"]["initial_weights_sha256"],
                "final_loss": report["losses"][-1]["loss"],
                "final_l6": report["losses"][-1]["l6"], "final_l12": report["losses"][-1]["l12"],
                "loss_windows": shared.loss_windows(report, size=50),
                "l12_windows": _key_windows(report, "l12", size=50),
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
            "rmse_csv_sha256": sha256_file(run_dir / "rmse.csv")}
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
    parity = flop_probe(TRAIN_MANIFEST)
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

    results = {"format": "r7-s3-v3-rollout-ft-result-v1", "scientific_claim": False,
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
                raise TimeoutError("v3-RFT exceeded the frozen hard cap; aborting without a claim")
        pins = json.loads((args.out / "protocol.json").read_text(
            encoding="utf-8"))["arms"]["control"]["pins"]
        all_cells = {}
        parent_cells = {}
        for seed in SEEDS:
            control_dir = D3 / f"seed{seed}" / "evaluation" / "process"
            parent_dir = PARENT_RUN / f"seed{seed}" / "evaluation" / "candidate"
            candidate_dir = args.out / f"seed{seed}" / "evaluation" / "candidate"
            for lead in EVALUATION_LEADS:
                expected_sha = pins["seeds"][str(seed)]["leads"][str(lead)]["rmse_csv_sha256"]
                actual_sha = sha256_file(control_dir / f"lead_{lead:03d}h" / "rmse.csv")
                if actual_sha != expected_sha:
                    raise RuntimeError(f"control CSV drifted from its pinned SHA256: "
                                       f"seed {seed} lead {lead}")
            all_cells[str(seed)] = shared.paired_cells(candidate_dir, control_dir,
                                                       EVALUATION_LEADS)
            parent_cells[str(seed)] = shared.paired_cells(candidate_dir, parent_dir,
                                                          EVALUATION_LEADS)
        results["paired_cells"] = all_cells
        results["parent_relative_cells"] = parent_cells
        results["primary_verdict"] = primary_verdict(all_cells)
        results["gate_pre_screen"] = gate_verdict(all_cells)
        parent_gate = gate_verdict(parent_cells)
        results["parent_relative_gate"] = parent_gate
        results["rollout_response"] = {
            "gate_failures_vs_control": len(results["gate_pre_screen"]["failures"]),
            "parent_gate_failures_vs_control": PARENT_GATE_FAILURES_VS_CONTROL,
            "gate_failures_vs_parent": len(parent_gate["failures"]),
            "text": ROLLOUT_RESPONSE_TEXT}
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
        json.dump({"format": "r7-s3-v3-rollout-ft-attempt-v1", "status": "complete",
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
