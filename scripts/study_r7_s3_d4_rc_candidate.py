"""D4 of S3: the R-C lead-coverage candidate screen against the same-data incumbent.

One factor, pre-registered: the training objective recipe moves from the
incumbent's ``l6`` deep supervision (400 updates) to the already-qualified
``two_step`` objective, which adds a 0.5-weighted +12 h supervised second step
at the same parameter count. Because the measured training forward+backward
cost ratio ``two_step / l6`` is 2.0018 on the v2 probe, the candidate runs 200
updates: the two arms are paired on parameter count, training FLOPs (within
0.1%) and the same per-seed seeded initialization.

Control readings are the S3-D3 incumbent run on this same store (pinned by
protocol/result/checkpoint/CSV SHA256 in this round's frozen protocol); the
control is not retrained here. The registered verdict reads only t2m/full at
6 h and 12 h by the per-seed sign rule; u10/v10/mslp relative MSE change over
all five leads is a pre-screen, not part of the verdict. A supported verdict
with a passing pre-screen advances the candidate to the S4 freeze; anything
else is registered as this round's result and the candidate stops.

The test split (2023) is never opened; the round is validation-only, offline,
co-resident on the declared GPU UUID with a read-only headroom gate, and
deadline-bounded by frozen soft/hard seconds.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEEDS = (41, 42, 43)
CANDIDATE_MODE = "two_step"
CANDIDATE_UPDATES = 200
CANDIDATE_LAMBDA12 = 0.5
STEPS = 4
LR = 1e-4
WARMUP = 10  # 5% of the 200-update candidate, matching the control's 20/400 fraction
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
PLANNED_SECONDS_ROUND = 5400.0
HARD_CAP_SECONDS_ROUND = 10800.0

INSTANCE = ROOT / "outputs/r7_s3_confirmation_2017_2022_2023_v2"
TRAIN_MANIFEST = INSTANCE / "store/manifests/train.jsonl"
VAL_MANIFEST = INSTANCE / "store/manifests/val.jsonl"
D3 = ROOT / "outputs/r7_s3_d3_incumbent_20261005_attempt01"
D3_RESULT = D3 / "result.json"
DEFAULT_OUT = ROOT / "outputs/r7_s3_d4_rc_candidate_20261005_attempt01"
GPU_UUID = "GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b"  # idle GPU0 at freeze time (GPU1 co-resident neighbour)
HEADROOM_MARGIN_BYTES = 2048 * 2**20
ESTIMATED_PEAK_BYTES = 2048 * 2**20
SOURCE_SHA256 = "e0b51616a7c31f29b9832221e74522f8b5cad4b36b72b0d7586d140787e3b42d"
EXPECTED_COHORTS = {"6": 472, "12": 468, "24": 460, "48": 444, "72": 428}
EXPECTED_TRAIN_WINDOWS_SHA = "f7f0c74c7cd8b055f4b8c9aa828d925c1d853245c7471699cc7f671043d59503"
ARCHIVED = ROOT / "outputs/r7_v2_comparison_20261004_attempt01"
ARCHIVED_PAIRING = {
    41: {"state": "a79ea47fa098bfc4dce651e2ea6c8fd18d996e4ff88a0e8503832645a29f1c47",
         "report": "b33c756723575681e0f7c57eb38ee8b2ba4715af708d00381c0914d803deb217"},
    42: {"state": "ec5bb5ef581e1cfe4923b938f60635fd5c79aaebd0f3f1fb37f01524ea3a26d0",
         "report": "8070b6d437d06dc94d21465fe3bed870b19ca7c773eaa882fbcdf976259e85f4"},
    43: {"state": "844bd23402cabb3f0cfb961efa4ba2e854207a06c51b51d32fba89e6e358efdf",
         "report": "8e2de147b25671b810a94d4c400f0ef567af189d1e634fb87734656556762c25"},
}
PRIMARY_DECISION_TEXT = (
    "Read the per-seed paired RMSE delta of t2m/full at 6 h and 12 h for "
    "'two_step_200 - incumbent_l6_400' on the 2022 val cohorts. A lead reads "
    "'supported' when every declared seed agrees in sign and the candidate delta is "
    "negative (lower RMSE); 'worsened' when every seed agrees and the delta is "
    "positive; 'unresolved' otherwise (disagreement is never averaged). The candidate "
    "is supported only when BOTH 6 h and 12 h read supported. A worsened lead or two "
    "unresolved leads falsify the lead-coverage candidate at this budget and the "
    "candidate stops; the round result is registered either way. The 24 h reading is "
    "reported as secondary and cannot change the verdict."
)
GATE_DECISION_TEXT = (
    "Pre-screen (not part of the primary verdict): for u10/v10/mslp at every lead "
    "{6,12,24,48,72} h, the per-seed relative MSE change "
    "(candidate_mse - control_mse)/control_mse must be <= 0.0 on full-region "
    "cohorts. Any positive cell fails the pre-screen; the candidate then does not "
    "advance to the S4 freeze even if the primary reads supported."
)
LIMITATIONS = [
    "screening only: one instance (2017 train / 2022 val, one ROI, 17 channels), no "
    "significance, convergence or SOTA claim",
    "the control readings come from the registered S3-D3 incumbent run on the same "
    "store and are pinned by SHA256, not retrained inside this protocol",
    "FLOP-matching follows the measured two_step/l6 objective ratio 2.0018; equal "
    "updates is deliberately NOT the comparison, so the update allocator differs",
    "three seeds are consistency evidence, not a significance test",
    "K4 is an inference-depth probe on the same K4-trained checkpoint, not an "
    "independent model",
    "GPU runs are co-resident; latency/memory observations include neighbor load",
    "validation split only; the test split stays sealed until the S4 preregistered read",
]


def refused_test_manifest(manifest):
    if Path(manifest).name == "test.jsonl":
        raise ValueError("this study is validation-only: the test split stays sealed")


def archived_process_spec():
    protocol = json.loads((ARCHIVED / "protocol.json").read_text(encoding="utf-8"))
    configuration = protocol["configuration"]
    return (configuration["model_specs"]["process"]["model"],
            configuration["initialization"], configuration)


def cpu_fidelity(seed, configuration):
    from training.r7_v2_profile import seeded_mapped_model, state_hash
    from training.r7_v2_protocol import digest
    model, report, spec = seeded_mapped_model(configuration, seed, "process")
    expected = ARCHIVED_PAIRING[seed]
    actual_state, actual_report = state_hash(model.state_dict()), digest(report)
    if actual_state != expected["state"] or actual_report != expected["report"]:
        raise RuntimeError(f"seed {seed} initialized state differs from archived actual C")
    return model, report, spec, actual_state, actual_report


def flop_parity_probe():
    """Measure the two objectives' actual forward+backward cost on the v2 probe."""
    import torch
    from torch.utils.data import default_collate
    from torch.utils.flop_counter import FlopCounterMode
    from data.r7_autoregressive_dataset import ZarrAutoregressiveDataset
    from training.r7_autoregressive_rollout import training_one_step, training_two_step
    from training.r7_v2_profile import seeded_mapped_model
    torch.set_num_threads(4)
    dataset = ZarrAutoregressiveDataset(TRAIN_MANIFEST, expected_exclusions=d4_train_exclusions())
    probe = default_collate([dataset[0]])
    spec, initialization, configuration = archived_process_spec()

    def measure(objective):
        model, _, _ = seeded_mapped_model(configuration, 41, "process")
        model.train()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad(), FlopCounterMode(display=False) as counter:
            output = objective(model, probe, 4)
            forward = int(counter.get_total_flops())
        model.zero_grad(set_to_none=True)
        with torch.enable_grad(), FlopCounterMode(display=False) as counter:
            output = objective(model, probe, 4)
            output.loss.backward()
            forward_backward = int(counter.get_total_flops())
        model.zero_grad(set_to_none=True)
        return forward, forward_backward

    f6, fb6 = measure(lambda model, batch, k: training_one_step(model, batch, reasoning_steps=k))
    f12, fb12 = measure(lambda model, batch, k: training_two_step(model, batch, reasoning_steps=k,
                                                                  lambda12=CANDIDATE_LAMBDA12))
    ratio = fb12 / fb6
    return {"probe_sample_id": probe["sample_id"][0],
            "l6_forward_flops": f6, "l6_forward_backward_flops": fb6,
            "two_step_forward_flops": f12, "two_step_forward_backward_flops": fb12,
            "ratio_two_step_over_l6": ratio,
            "matched_updates": {"candidate_two_step": CANDIDATE_UPDATES,
                                "control_l6": round(CANDIDATE_UPDATES * ratio)},
            "candidate_total_flops": CANDIDATE_UPDATES * fb12,
            "control_total_flops": round(CANDIDATE_UPDATES * ratio) * fb6,
            "parity_relative_error": abs(CANDIDATE_UPDATES * fb12 - round(CANDIDATE_UPDATES * ratio) * fb6)
                                     / (round(CANDIDATE_UPDATES * ratio) * fb6),
            "flop_convention": "supported aten operations under enable_grad; elementwise/normalization omitted"}


def d4_train_exclusions():
    from data.r7_autoregressive_dataset import preflight_training_windows
    windows = preflight_training_windows(TRAIN_MANIFEST)
    if windows["window_sha256"] != EXPECTED_TRAIN_WINDOWS_SHA:
        raise RuntimeError("v2 train windows differ from the frozen expectation")
    if windows["usable_windows"] != 468:
        raise RuntimeError("v2 train usable windows differ from the frozen expectation")
    return windows["excluded_sample_ids"]


def control_pins():
    """Pin the D3 control readings that this round compares against."""
    from training.r7_arm_harness import sha256_file
    if not D3_RESULT.is_file():
        raise FileNotFoundError(f"D3 control result missing: {D3_RESULT}")
    result = json.loads(D3_RESULT.read_text(encoding="utf-8"))
    if sorted(result["seeds"]) != [str(seed) for seed in SEEDS]:
        raise RuntimeError("D3 control does not cover the declared seeds")
    pins = {"d3_result_sha256": sha256_file(D3_RESULT), "seeds": {}}
    for seed in SEEDS:
        receipts = result["seeds"][str(seed)]
        entry = {"checkpoint_sha256": receipts["checkpoint_sha256"],
                 "training_report_sha256": receipts["training_report_sha256"],
                 "initial_state_sha256": receipts["initial_state_sha256"], "leads": {}}
        for lead in EVALUATION_LEADS:
            record = receipts["evaluations"][str(lead)]
            if record["n_evaluated"] != EXPECTED_COHORTS[str(lead)]:
                raise RuntimeError(f"D3 seed {seed} lead {lead} cohort differs from the frozen expectation")
            entry["leads"][str(lead)] = {"rmse_csv_sha256": record["rmse_csv_sha256"],
                                         "evaluation_dir": record["dir"]}
        pins["seeds"][str(seed)] = entry
    return pins


def contract_for(seed, spec, initialization, protocol_sha256, data_identity, windows):
    from training.r7_v2_protocol import child_contract
    protocol = {"arm_configs": {"candidate": {"kind": "process", "mode": CANDIDATE_MODE}},
                "data": {"data_identity": data_identity},
                "sources": {"source_sha256": SOURCE_SHA256},
                "protocol_sha256": protocol_sha256, "windows": windows,
                "parents": {},
                "cpu_profile": {"pairing": {str(seed): {"candidate": {"seed": seed}}}},
                "shared_controls": {"lambda12": CANDIDATE_LAMBDA12},
                "limitations": []}
    contract = child_contract(protocol, {"arm": "candidate", "seed": seed}, spec)
    contract["initialization"] = initialization
    contract["parent_provenance"] = None
    contract.pop("process_supervision", None)
    return contract


def protocol_payload(parity):
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest, dataset_identity
    from data.r7_autoregressive_dataset import preflight_training_windows
    spec, initialization, _ = archived_process_spec()
    windows = preflight_training_windows(TRAIN_MANIFEST)
    train_identity, _ = dataset_identity(TRAIN_MANIFEST)
    val_identity, _ = dataset_identity(VAL_MANIFEST)
    return {
        "format": "r7-s3-d4-rc-candidate-protocol-v1",
        "stage": "S3-D4",
        "objective": ("single-factor screen of the R-C lead-coverage candidate: the same "
                      "process model, initialization and data as the S3-D3 incumbent, with the "
                      "training objective switched from l6-only deep supervision to the "
                      "two_step objective (l6 + 0.5*l12); FLOP-matched at 200 candidate updates "
                      "against the control's 400; val-only three seeds"),
        "instance": str(INSTANCE.resolve()),
        "train_manifest": str(TRAIN_MANIFEST.resolve()),
        "val_manifest": str(VAL_MANIFEST.resolve()),
        "test_manifest": str((INSTANCE / "store/manifests/test.jsonl").resolve()),
        "source_sha256": SOURCE_SHA256,
        "code": None,  # filled by main() at freeze time
        "train_data_identity": train_identity,
        "val_data_identity": val_identity,
        "train_windows": windows,
        "arms": {
            "candidate": {"name": "two_step_200", "mode": CANDIDATE_MODE,
                          "updates": CANDIDATE_UPDATES, "lambda12": CANDIDATE_LAMBDA12,
                          "steps": STEPS, "lr": LR, "warmup": WARMUP,
                          "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
                          "clip": CLIP, "checkpoint_every": CHECKPOINT_EVERY, "bf16": BF16,
                          "selection": "frozen endpoint; no validation selection",
                          "model": {"kind": "process", "spec": spec,
                                    "spec_canonical_digest": canonical_digest(spec)}},
            "control": {"name": "incumbent_l6_400", "mode": "l6", "updates": 400,
                        "source": "S3-D3 registered run on this same instance",
                        "pins": control_pins()}},
        "flop_parity": parity,
        "decision": {"primary": {"variable": PRIMARY_VARIABLE, "region": "full",
                                 "leads_hours": list(PRIMARY_LEADS),
                                 "reading": "two_step_200 - incumbent_l6_400",
                                 "text": PRIMARY_DECISION_TEXT},
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


def read_rmse_table(path):
    """{variable: rmse} for one single-lead rmse.csv (full-region pooled row set)."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 17 or len({row["variable"] for row in rows}) != 17:
        raise RuntimeError(f"rmse table must carry all 17 variables: {path}")
    return {row["variable"]: float(row["rmse"]) for row in rows}


def paired_cells(candidate_dir, control_dir):
    """Per (seed, lead, variable) RMSE and relative-MSE-change cells."""
    cells = {}
    for lead in EVALUATION_LEADS:
        candidate = read_rmse_table(Path(candidate_dir) / f"lead_{lead:03d}h" / "rmse.csv")
        control = read_rmse_table(Path(control_dir) / f"lead_{lead:03d}h" / "rmse.csv")
        if set(candidate) != set(control):
            raise RuntimeError(f"candidate/control variables differ at lead {lead}")
        for variable in sorted(candidate):
            delta = candidate[variable] - control[variable]
            relative = (candidate[variable] ** 2 - control[variable] ** 2) / (control[variable] ** 2)
            if not math.isfinite(delta) or not math.isfinite(relative):
                raise RuntimeError(f"nonfinite paired cell at lead {lead} variable {variable}")
            cells[f"{lead}|{variable}"] = {"lead_hours": lead, "variable": variable,
                                           "candidate_rmse": candidate[variable],
                                           "control_rmse": control[variable],
                                           "rmse_delta": delta, "relative_mse_change": relative}
    return cells


def seed_verdict(seed_cells):
    """The registered primary verdict for one seed: t2m at 6 and 12 h."""
    verdict = {}
    for lead in PRIMARY_LEADS:
        delta = seed_cells[f"{lead}|{PRIMARY_VARIABLE}"]["rmse_delta"]
        verdict[lead] = "candidate_lower" if delta < 0 else ("candidate_higher" if delta > 0 else "tie")
    return verdict


def primary_verdict(all_cells):
    """Per-seed sign agreement for t2m 6/12 h; disagreement is unresolved, never averaged."""
    leads = {}
    for lead in PRIMARY_LEADS:
        signs = {}
        for seed in SEEDS:
            delta = all_cells[str(seed)][f"{lead}|{PRIMARY_VARIABLE}"]["rmse_delta"]
            signs[seed] = 0 if abs(delta) < 1e-12 else (1 if delta > 0 else -1)
        distinct = set(signs.values())
        if distinct == {-1}:
            leads[lead] = "supported"
        elif distinct == {1}:
            leads[lead] = "worsened"
        else:
            leads[lead] = "unresolved"
    if leads[6] == "supported" and leads[12] == "supported":
        overall = "supported"
    elif "worsened" in leads.values():
        overall = "worsened"
    elif all(value == "unresolved" for value in leads.values()):
        overall = "unresolved"
    else:
        overall = "unsupported"
    return {"per_lead": leads, "overall": overall,
            "supported_rule": "both 6 h and 12 h supported (all seeds same sign, candidate lower)",
            "no_averaging": "sign disagreement is unresolved and never averaged into a verdict"}


def gate_verdict(all_cells):
    """u10/v10/mslp relative MSE change <= 0 at every lead and seed."""
    failures = []
    for seed in SEEDS:
        for lead in EVALUATION_LEADS:
            for variable in GATE_VARIABLES:
                value = all_cells[str(seed)][f"{lead}|{variable}"]["relative_mse_change"]
                if value > GATE_TOLERANCE:
                    failures.append({"seed": seed, "lead_hours": lead, "variable": variable,
                                     "relative_mse_change": value})
    return {"passed": not failures, "tolerance": GATE_TOLERANCE,
            "failures": failures,
            "rule": "u10/v10/mslp relative MSE change <= 0.0 at every lead/seed/full"}


def gpu_gate(owned_reserved_peak_bytes=0):
    import subprocess
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,memory.free,memory.total", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=60)
    rows = [line.split(", ") for line in query.stdout.strip().splitlines()]
    match = next((row for row in rows if row[0].strip() == GPU_UUID), None)
    if match is None:
        raise RuntimeError(f"declared GPU UUID {GPU_UUID} not visible to nvidia-smi")
    free_mib, total_mib = int(match[1]), int(match[2])
    required_mib = max(ESTIMATED_PEAK_BYTES, owned_reserved_peak_bytes) // 2**20 + HEADROOM_MARGIN_BYTES // 2**20
    gate = {"uuid": GPU_UUID, "free_bytes": free_mib * 2**20, "total_bytes": total_mib * 2**20,
            "required_bytes": required_mib * 2**20,
            "owned_reserved_peak_bytes": owned_reserved_peak_bytes,
            "passed": bool(free_mib >= required_mib), "source": "nvidia-smi read-only"}
    if not gate["passed"]:
        raise RuntimeError(f"co-residency headroom gate failed: {gate}")
    return gate


def _pin_declared_gpu():
    import os
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", GPU_UUID):
        raise RuntimeError("CUDA_VISIBLE_DEVICES already set to a different device")
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID


def code_state():
    import subprocess
    def run(*arguments):
        return subprocess.run(["git", *arguments], cwd=ROOT, check=True,
                              capture_output=True, text=True, timeout=60).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": run("status", "--porcelain")}


def seed_protocol_stamp(output_dir):
    frozen = json.loads((output_dir / "protocol.json").read_text(encoding="utf-8"))
    return frozen["protocol_sha256"]


def run_seed(seed, output_dir, *, deadline, device_name):
    from training.r7_autoregressive_runner import fine_tune
    from training.r7_evaluate import evaluate_local
    from training.r7_experiment import dataset_identity
    from training.r7_arm_harness import sha256_file
    if seed not in SEEDS:
        raise ValueError(f"declared seeds are {SEEDS}; {seed} is not among them")
    _, _, configuration = archived_process_spec()
    model, initialization, spec, state_digest, report_digest = cpu_fidelity(seed, configuration)
    train_identity, _ = dataset_identity(TRAIN_MANIFEST)
    contract = contract_for(seed, spec, initialization, seed_protocol_stamp(output_dir),
                            train_identity, json.loads((output_dir / "protocol.json").read_text(
                                encoding="utf-8"))["train_windows"])
    train_dir = output_dir / f"seed{seed}" / "training" / "candidate"
    if train_dir.exists() or train_dir.is_symlink():
        raise FileExistsError(f"fresh training output only: {train_dir}")
    checkpoint, report = fine_tune(
        TRAIN_MANIFEST, train_dir, model=model, contract=contract, parent_weights=None,
        steps=STEPS, updates=CANDIDATE_UPDATES, seed=seed, mode=CANDIDATE_MODE, lr=LR,
        warmup=WARMUP, weight_decay=WEIGHT_DECAY, bf16=BF16, deadline=deadline,
        device_name=device_name, resume=None, checkpoint_every=CHECKPOINT_EVERY,
        batch_size=BATCH_SIZE, clip=CLIP, lambda12=CANDIDATE_LAMBDA12)
    if report["updates_this_run"] != CANDIDATE_UPDATES or report["selected_update"] != CANDIDATE_UPDATES:
        raise RuntimeError(f"seed {seed} did not reach the frozen candidate endpoint")
    if not all(record["l12"] is not None for record in report["losses"][:3]):
        raise RuntimeError(f"seed {seed} candidate objective did not supervise the +12 h step")
    receipts = {"seed": seed, "training_dir": str(train_dir), "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "training_report_sha256": sha256_file(train_dir / "training_report.json"),
                "initial_state_sha256": state_digest, "initialization_report_sha256": report_digest,
                "final_loss": report["losses"][-1]["loss"], "elapsed_seconds": report["elapsed_seconds"],
                "evaluations": {}}
    for lead in EVALUATION_LEADS:
        run_dir = output_dir / f"seed{seed}" / "evaluation" / "candidate" / f"lead_{lead:03d}h"
        if run_dir.exists() or run_dir.is_symlink():
            raise FileExistsError(f"fresh evaluation output only: {run_dir}")
        provenance = evaluate_local(
            str(VAL_MANIFEST), output_dir=run_dir, checkpoint=checkpoint,
            lead_hours=(lead,), max_samples=10**9, device_name=device_name,
            reasoning_steps=REASONING_STEPS, deadline=deadline)
        expected = EXPECTED_COHORTS[str(lead)]
        if (provenance["split"] != "val" or "test_read" in provenance
                or provenance["n_evaluated"] != expected
                or provenance["n_available_windows"] != expected
                or provenance["lead_hours"] != [lead]
                or provenance.get("process_scale_sidecar_identity") is not None):
            raise RuntimeError(f"seed {seed} lead {lead} provenance differs from the frozen expectation")
        if provenance["climatology"]["training_years"] != [2017]:
            raise RuntimeError(f"seed {seed} lead {lead} climatology is not train-only 2017")
        receipts["evaluations"][str(lead)] = {
            "dir": str(run_dir), "n_evaluated": provenance["n_evaluated"],
            "elapsed_seconds": provenance["elapsed_seconds"],
            "rmse_csv_sha256": sha256_file(run_dir / "rmse.csv"),
            "provenance_sha256": sha256_file(run_dir / "provenance.json")}
    return receipts


def main(argv=None):
    from scripts.r7_m3_offline import deny_network
    from training.r7_experiment import canonical_digest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    deny_network()
    for manifest in (TRAIN_MANIFEST, VAL_MANIFEST):
        refused_test_manifest(manifest)
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError(f"refusing existing output: {args.out}")

    started = time.perf_counter()
    parity = flop_parity_probe()
    print(json.dumps({"flop_parity": parity}), flush=True)
    protocol = protocol_payload(parity)
    protocol["code"] = code_state()
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

    results = {"format": "r7-s3-d4-rc-candidate-result-v1", "scientific_claim": False,
               "test_read": False, "limitations": LIMITATIONS,
               "protocol_sha256": protocol["protocol_sha256"],
               "train_data_identity": protocol["train_data_identity"],
               "val_data_identity": protocol["val_data_identity"],
               "flop_parity": parity, "gpu_gates": [], "seeds": {}}
    hard_deadline = started + HARD_CAP_SECONDS_ROUND
    _pin_declared_gpu()
    import torch
    torch.set_num_threads(4)
    owned_peak = 0
    first_spawn = last_reap = None
    try:
        for seed in SEEDS:
            gate = gpu_gate(owned_reserved_peak_bytes=owned_peak)
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
                raise TimeoutError("D4 exceeded the frozen hard cap; aborting without a claim")
        # Paired reading against the pinned D3 control; the control CSVs consumed
        # must be byte-identical to the SHA256 pinned in the frozen protocol.
        from training.r7_arm_harness import sha256_file
        pins = json.loads((args.out / "protocol.json").read_text(encoding="utf-8"))["arms"]["control"]["pins"]
        all_cells = {}
        for seed in SEEDS:
            control_dir = D3 / f"seed{seed}" / "evaluation" / "process"
            candidate_dir = args.out / f"seed{seed}" / "evaluation" / "candidate"
            for lead in EVALUATION_LEADS:
                expected_sha = pins["seeds"][str(seed)]["leads"][str(lead)]["rmse_csv_sha256"]
                actual_sha = sha256_file(control_dir / f"lead_{lead:03d}h" / "rmse.csv")
                if actual_sha != expected_sha:
                    raise RuntimeError(f"control CSV drifted from its pinned SHA256: seed {seed} lead {lead}")
            all_cells[str(seed)] = paired_cells(candidate_dir, control_dir)
        results["paired_cells"] = all_cells
        results["primary_verdict"] = primary_verdict(all_cells)
        results["gate_pre_screen"] = gate_verdict(all_cells)
        results["decision"] = ("advance-to-D5-freeze" if (results["primary_verdict"]["overall"] == "supported"
                                and results["gate_pre_screen"]["passed"])
                               else "candidate-stops-registered-negative")
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
        json.dump({"format": "r7-s3-d4-attempt-v1", "status": "complete",
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
