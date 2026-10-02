"""Frozen, validation-only M3 process-supervision controls; no experiment runs here."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .r7_rw_b_subtraction_protocol import (
    BATCH_SIZE, CLIP, EARLY_STOPPING_PATIENCE, EVALUATION_LEADS,
    EVALUATION_MAX_SAMPLES, LR, MINIMUM_IMPROVEMENT, MINIMUM_LR_RATIO,
    RW_A_CONFIG, SEEDS, SWITCH_KEYS, UPDATES, VALIDATION_EVERY,
    VALIDATION_LEADS, WARMUP_UPDATES, WEIGHT_DECAY,
)

REASONING_STEPS = 4
ARM_NAMES = ("aux_off", "input_aux", "future_draft_aux")
ARMS = tuple((name, "process", dict(RW_A_CONFIG, default_reasoning_steps=4))
             for name in ARM_NAMES)
WEIGHTS = {
    "aux_off": {"input_diagnostic_weight": 0.0, "future_diagnostic_weight": 0.0,
                "draft_diagnostic_weight": 0.0},
    "input_aux": {"input_diagnostic_weight": 0.1, "future_diagnostic_weight": 0.0,
                  "draft_diagnostic_weight": 0.0},
    "future_draft_aux": {"input_diagnostic_weight": 0.0, "future_diagnostic_weight": 0.05,
                         "draft_diagnostic_weight": 0.05},
}
PAIRS = (("input_aux", "aux_off"), ("future_draft_aux", "aux_off"),
         ("future_draft_aux", "input_aux"))
LEADS = EVALUATION_LEADS
GLOBAL_SECONDS = 3600.0
ROUND_SECONDS = 1800.0
WORKER_SECONDS = 1800.0
CLEANUP_SECONDS = 10.0
HEADROOM_MARGIN_MIB = 2048
PROTOCOL_FORMAT = "r7-73-process-supervision-protocol-v1"
RESULT_FORMAT = "r7-m3-process-supervision-result-v1"
AUTHORIZATION_SCOPE = "r7-73-process-supervision-3arm-2seed-400updates-k4"
LIMITATIONS = [
    "scientific_claim:false: one bounded winter segment, two seeds, no significance or SOTA claim",
    "validation-only selection and evaluation; sealed test manifests and fields are never opened",
    "400 updates and Ktrain=4 do not establish convergence or generalization",
    "0.1 total auxiliary budget is frozen from the historical legacy default (process_forecast_coreasoning_loss and #65); current N1 legacy weight was zero; no lambda sweep",
    "FLOPs count full forward plus actual new losses and measured backward, not the streamed training graph",
    "FlopCounterMode excludes elementwise/normalization arithmetic, including future/draft diagnostic arithmetic",
    "one train-case CPU gradient ownership/norm probe is descriptive, not causal or population evidence",
    "shared GPU neighbors may affect wall time; their measurements are neither removed nor adjusted",
    "dataset_identity binds manifests/store metadata and norms; separately pinned source bytes establish provenance",
    "reproducibility is code/data/protocol pinned and initialization bitwise paired; GPU results are not asserted bitwise reproducible",
    "monthly-hour train climatology is not a WeatherBench2 or strong seasonal reference",
    "historical K3 N1 training subtotal about1014.85s and old whole round1292.54s are not remeasured M3 costs; K4 plus36fresh startups may exhaust the1800s cap; failure stops without expanding scope",
]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    """All public artifacts are exclusive; no repair-in-place or resume overwrite."""
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def preserve_artifact_error(original, additional, *, label):
    """Keep the triggering exception; append explicit cleanup/publication diagnostics."""
    message = f"{label}: {type(additional).__name__}: {additional}"
    if original is not None:
        original.add_note(message)
        import sys
        try:
            print(message, file=sys.stderr, flush=True)
        except OSError as exc:
            original.add_note(f"secondary diagnostic output failed: {type(exc).__name__}: {exc}")
        return
    raise additional


def read_json(path):
    path = Path(path)
    if path.name == "test.jsonl" or path.is_symlink():
        raise ValueError("sealed test manifests and symlink artifacts are forbidden")
    return json.loads(path.read_text(encoding="utf-8"))


def model_config(channels):
    return {"in_channels": channels, "out_channels": channels, "history_steps": 2,
            **dict(RW_A_CONFIG, default_reasoning_steps=REASONING_STEPS)}


def supervision_contract(protocol, arm):
    if arm not in ARM_NAMES:
        raise ValueError("undeclared arm")
    return {"sidecar_path": protocol["sidecar"]["path"],
            "sidecar_identity": protocol["sidecar"]["identity"],
            **WEIGHTS[arm], "protocol_sha256": protocol["protocol_sha256"]}


def expected_training_contract(protocol, arm):
    """Expected derived fixed inverse matches runner's explicitly supplied six fields."""
    fixed = {"channel_names": protocol["data"]["channels"],
             "normalization_mean": protocol["data"]["normalization_mean"],
             "normalization_std": protocol["data"]["normalization_std"],
             "data_identity": protocol["data"]["data_identity"]}
    return {**supervision_contract(protocol, arm), "fixed_train_context": fixed,
            "fixed_train_context_sha256": digest(fixed)}


def planned_jobs():
    """Six fresh training processes, then thirty independent single-lead evaluations."""
    trains = [{"phase": "train", "seed": seed, "arm": arm, "lead": None}
              for seed in SEEDS for arm in ARM_NAMES]
    evaluations = [{"phase": "evaluate", "seed": seed, "arm": arm, "lead": lead}
                   for seed in SEEDS for arm in ARM_NAMES for lead in LEADS]
    return trains + evaluations


def job_key(job):
    tail = "" if job["lead"] is None else f"_lead{job['lead']:03d}h"
    return f"{job['phase']}_seed{job['seed']}_{job['arm']}{tail}"


def shared_controls():
    return {
        "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY,
        "updates": UPDATES, "batch_size": BATCH_SIZE, "clip": CLIP,
        "warmup_updates": WARMUP_UPDATES, "minimum_lr_ratio": MINIMUM_LR_RATIO,
        "validation_every": VALIDATION_EVERY, "patience": EARLY_STOPPING_PATIENCE,
        "minimum_improvement": MINIMUM_IMPROVEMENT,
        "validation_leads": list(VALIDATION_LEADS), "evaluation_leads": list(LEADS),
        "evaluation_max_samples": EVALUATION_MAX_SAMPLES, "reasoning_steps": REASONING_STEPS,
        "legacy_process_weight": 0.0, "bf16": False, "step_hours": 6,
        "checkpoint_selection": "lowest mean latitude-weighted normalized validation MSE; ties keep earlier",
        "loss": "latitude-weighted deep-supervised forecast plus frozen diagnostic weights; streamed truncated BPTT",
        "sample_order": "torch.randperm(len(dataset), manual_seed(seed+epoch))",
    }


def verify_authorization(path):
    """A named written authorization must match, never an inferred permission."""
    authorization = read_json(path)
    expected = {"seeds": list(SEEDS), "arms": list(ARM_NAMES), "updates": UPDATES,
                "reasoning_steps": REASONING_STEPS, "max_gpu_seconds": GLOBAL_SECONDS,
                "max_round_seconds": ROUND_SECONDS, "max_worker_seconds": WORKER_SECONDS,
                "gpu_policy": "shared"}
    if (authorization.get("status") != "authorized"
            or authorization.get("scope") != AUTHORIZATION_SCOPE
            or not isinstance(authorization.get("user_response"), str)
            or not authorization["user_response"].strip()
            or authorization.get("bounds") != expected):
        raise ValueError("actual recorded named M3 user authorization with exact bounds is required")
    return authorization


def build_protocol(*, manifests, output, dataset, sources, sidecar, code, profile,
                   authorization, authorization_sha256, gpu_uuid, estimated_peak_mib):
    if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-") or "," in gpu_uuid:
        raise ValueError("one physical GPU UUID required")
    if (isinstance(estimated_peak_mib, bool) or not isinstance(estimated_peak_mib, int)
            or estimated_peak_mib < 1):
        raise ValueError("positive existing peak estimate in MiB required; no zero-memory guess")
    body = {
        "format": PROTOCOL_FORMAT, "issue": "#73 M3 process supervision",
        "scientific_claim": False, "limitations": LIMITATIONS,
        "frozen_before_any_step": True, "test_read": False,
        "output": str(Path(output).resolve()), "manifests": str(Path(manifests).resolve()),
        "data": dataset, "sources": sources, "sidecar": sidecar, "code": code,
        "authorization": authorization, "authorization_sha256": authorization_sha256,
        "seeds": list(SEEDS), "arms": [
            {"name": arm, "kind": "process", "model_config": model_config(len(dataset["channels"])),
             "switches": {key: ARMS[0][2].get(key, default) for key, default in SWITCH_KEYS},
             "legacy_process_weight": 0.0, "supervision_weights": WEIGHTS[arm],
             **profile["measurements"][arm]} for arm in ARM_NAMES],
        "cpu_profile": profile, "shared_controls": shared_controls(),
        "gpu": {"policy": "shared", "uuid": gpu_uuid, "estimated_peak_mib": estimated_peak_mib,
                "headroom_margin_mib": HEADROOM_MARGIN_MIB,
                "gate": "memory.free >= max(frozen estimate, observed owned reserved peak) + margin before EVERY spawn",
                "max_gpu_seconds": GLOBAL_SECONDS, "max_round_seconds": ROUND_SECONDS,
                "max_worker_seconds": WORKER_SECONDS, "cleanup_reserve_seconds": CLEANUP_SECONDS, "threads": 4,
                "billing": "one continuous monotonic clock, before first spawn/startup through last owned reap; all gaps/evals/cleanup included",
                "neighbor_policy": "read-only metadata; never signal non-owned PIDs; no retries or alternate-device fallback"},
        "jobs": planned_jobs(),
        "comparisons": [{"focus": focus, "baseline": baseline, "depth": 0}
                        for focus, baseline in PAIRS],
        "reporting": {
            "endpoints": "all 17 physical-unit variables x all five frozen leads; no new primary threshold",
            "rule": "#60 seed-paired signs only: all seeds negative=improved, all positive=worsened, otherwise unresolved",
            "unresolved": "pause, do not advance to next campaign node; negative/mixed results are retained",
            "tables": ["arm_table.csv", "training_table.csv", "memory_table.csv",
                       "rmse_table.csv", "case_table.csv", "acc_table.csv"],
            "required_training_runs": 6, "required_evaluations": 30,
            "required_rmse_cells": 510, "required_pair_cells_per_pair": 85,
        },
    }
    result = {**body, "protocol_sha256": digest(body)}
    validate_protocol(result)
    return result


def validate_cpu_profile(profile):
    """No omitted arms, seed pairing or component ownership in the frozen evidence."""
    if (profile.get("scientific_claim") is not False or not profile.get("limitations")
            or profile.get("device") != "cpu" or profile.get("reasoning_steps") != REASONING_STEPS
            or profile.get("probe_batch_size") != BATCH_SIZE or profile.get("gradient_probe_seed") != SEEDS[0]
            or set(profile.get("measurements", {})) != set(ARM_NAMES)
            or set(profile.get("gradient_ownership", {})) != set(ARM_NAMES)
            or set(profile.get("pairing", {})) != {str(seed) for seed in SEEDS}
            or "uncounted" not in profile.get("flop_convention", "")):
        raise ValueError("complete actual CPU profile and FLOP arithmetic limitations required")
    for entry in profile["pairing"].values():
        hashes = entry["full_initial_state_sha256"]
        if (entry.get("all_shared_pairs_identical") is not True or entry.get("same_tensor_set") is not True
                or set(hashes) != set(ARM_NAMES) or len(set(hashes.values())) != 1
                or any(not isinstance(value, str) or len(value) != 64 for value in hashes.values())):
            raise ValueError("both exact seeds must have bitwise identical full state hashes")
    for arm in ARM_NAMES:
        ownership = profile["gradient_ownership"][arm]
        if (ownership.get("scientific_claim") is not False or not ownership.get("limitations")
                or not ownership.get("sample_id")
                or set(ownership.get("components", {})) != {"forecast", "input", "future", "draft", "total"}):
            raise ValueError("all CPU first train-case gradient components must be recorded")
        for component in ownership["components"].values():
            if not math.isfinite(component["loss"]):
                raise ValueError("finite CPU component loss required")
            groups = component["groups"]
            if set(groups) != {"shared", "query", "readout", "forecast_history", "forecast_drafts"}:
                raise ValueError("all gradient ownership groups must be recorded")
            if any(not math.isfinite(group["norm"]) or group["norm"] < 0 for group in groups.values()):
                raise ValueError("finite gradient ownership norms required")


def validate_protocol(protocol):
    body = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if protocol.get("protocol_sha256") != digest(body):
        raise ValueError("frozen protocol digest mismatch")
    if (protocol.get("format") != PROTOCOL_FORMAT or protocol.get("scientific_claim") is not False
            or not protocol.get("limitations") or protocol.get("test_read") is not False
            or protocol.get("frozen_before_any_step") is not True
            or protocol.get("seeds") != list(SEEDS)
            or protocol.get("shared_controls") != shared_controls()
            or protocol.get("jobs") != planned_jobs()):
        raise ValueError("frozen M3 controls/job set changed")
    validate_cpu_profile(protocol["cpu_profile"])
    channels = protocol["data"]["channels"]
    units = protocol["data"]["units"]
    if (len(channels) != 17 or len(set(channels)) != 17 or len(units) != 17
            or any(unit in ("", "unknown", "normalized") for unit in units)):
        raise ValueError("all 17 physical channels with units are required")
    if [entry["name"] for entry in protocol["arms"]] != list(ARM_NAMES):
        raise ValueError("frozen arm set changed")
    for entry in protocol["arms"]:
        if (entry["kind"] != "process" or entry["model_config"] != model_config(17)
                or entry["legacy_process_weight"] != 0
                or entry["supervision_weights"] != WEIGHTS[entry["name"]]
                or entry["model_config"].get("local_solver_state", False)):
            raise ValueError("M3 may only change the frozen diagnostic weights")
        for key in ("parameters", "trainable_parameters", "forward_flops", "forward_backward_flops"):
            if (isinstance(entry[key], bool) or not isinstance(entry[key], int) or entry[key] <= 0
                    or entry[key] != protocol["cpu_profile"]["measurements"][entry["name"]][key]):
                raise ValueError("actual positive per-arm CPU counts required")
    gpu = protocol["gpu"]
    if (gpu["policy"] != "shared" or gpu["max_gpu_seconds"] != GLOBAL_SECONDS
            or gpu["max_worker_seconds"] != WORKER_SECONDS or gpu["max_round_seconds"] != ROUND_SECONDS
            or gpu["cleanup_reserve_seconds"] != CLEANUP_SECONDS
            or gpu["headroom_margin_mib"] != HEADROOM_MARGIN_MIB
            or gpu["threads"] != 4 or not gpu["uuid"].startswith("GPU-")
            or not math.isfinite(gpu["estimated_peak_mib"]) or gpu["estimated_peak_mib"] < 1):
        raise ValueError("frozen shared-GPU budget/headroom contract changed")
    expected = [{"focus": first, "baseline": second, "depth": 0} for first, second in PAIRS]
    if protocol["comparisons"] != expected:
        raise ValueError("frozen descriptive comparisons changed")
    return protocol


def verify_protocol(path):
    return validate_protocol(read_json(path))
