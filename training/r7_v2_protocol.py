"""Independent v2 attempt protocol, safe exclusive publication and frozen job inventory."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "r7-v2-remaining-protocol-v1"
DEFAULT_OUTPUT = ROOT / "outputs/r7_74_autoregressive_20261003_attempt01"
B_SEEDS = (41, 42)
C_SEEDS = (41, 42, 43)
LEADS = (6, 12, 24, 48, 72)
REGIONS = ("full", "interior", "edge_2")
B_ARMS = ("continue_l6", "rollout_l6_l12", "equal_compute_l6")
C_ARMS = ("old_ours", "process", "matched_generic")
BUDGETS = {"B": (5400.0, 10800.0), "C": (10800.0, 21600.0)}
BASELINE_REPORTING = {
    "required": ["persistence", "climatology"], "training_updates": 0,
    "persistence": "known last history frame repeated at the requested lead; no future observation input",
    "climatology": "train-only monthly-hour climatology, with the existing physical-unit inverse",
    "cases": "same exact requested-lead validation cases as the model; no cohort narrowing or exclusions",
    "regions": list(REGIONS), "variables": "all 17 frozen channels",
    "metrics": ["rmse", "acc", "mse_skill"],
    "tables": {"region_metrics": "baseline_region_metrics.csv", "per_case_metrics": "baseline_per_case_metrics.csv"},
    "execution_scope": "mandatory within every existing B/C evaluation worker, included in its measured cost; no added jobs",
    "scientific_gate_change": False,
}
CLEANUP_SECONDS = 10.0
CONTROLS = {"steps": 4, "batch_size": 1, "clip": 1.0, "lr": 1e-4,
            "warmup": 20, "weight_decay": 1e-4, "bf16": False,
            "checkpoint_every": 20, "lambda12": 0.5, "resume": None,
            "sample_order": "same seed and exact declared valid training windows; new optimizer per arm",
            "gradient_recipe": "deep supervision per physical step: existing normalized K-weighted area MSE of initial/all drafts, final weight2; full BPTT across internal K and physical rollout"}
LIMITATIONS = [
    "scientific_claim:false; validation-only bounded local study, no significance or SOTA claim",
    "sealed test manifests and fields are never read; existing outputs remain read-only",
    "B children use full BPTT, unlike the parent's streamed/detached training; not strict resume",
    "B arms share gradient/optimizer/sample recipes; comparisons address only the rollout objective",
    "equal-compute is an update-count control, not assumed equal FLOPs or wall time; actual costs are reported",
    "CPU FlopCounter counts supported operations; elementwise/normalization arithmetic is not counted",
    "two B seeds and one winter segment do not establish convergence or generalization",
    "shared GPU neighbor metadata are read-only; no neighbor is signaled or removed from evidence",
    "code/data/protocol-pinned reproducibility; bitwise GPU reproducibility is not asserted",
    "soft overrun continues; hard whole-round truncation or any failure stops this attempt without retry",
]
B_REPORTING = {
    "primary": {"variable": "t2m", "lead_hours": [6, 12], "degradation_tolerance": 0.0,
                "unit": "K", "rule": "strict seed-paired same-sign; no significance"},
    "candidate_selection": {
        "focus": "rollout_l6_l12", "baseline": "continue_l6", "equal_compute": "equal_compute_l6",
        "rule": "both primary leads all seeds RMSE lower than continue_l6 and no seed worsening vs equal_compute; otherwise retain L6 for independent C; no posthoc selection"},
    "full_endpoints": {"variables": "all 17 frozen channels", "lead_hours": list(LEADS),
                       "regions": list(REGIONS), "cases": "all exact per-lead validation cases, max_samples=32"},
    "comparison_seed_rule": "paired within seed and lead; no cross-lead averaging of different cohorts",
    "adaptive": {"start": False, "reason": "C requires an independent configuration after B"},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def local_path(path):
    requested = Path(path).absolute()
    if ("://" in str(path) or ".." in requested.parts
            or any(p.is_symlink() for p in (requested, *requested.parents))):
        raise ValueError("local paths without symlink ancestors or traversal required")
    if requested.name == "test.jsonl":
        raise ValueError("sealed test manifest forbidden")
    return requested.resolve()


def safe_output(path):
    """Only a named new attempt root, never a protected tree or nested old output."""
    output = local_path(path)
    parts = output.parts
    if (any("legacy" in p.lower() for p in parts)
            or any(parts[i:i + 2] in (("data", "raw"), ("data", "processed"), ("data", "interim"))
                   for i in range(len(parts) - 1))):
        raise ValueError("protected data/legacy output path forbidden")
    if not re.fullmatch(r"r7_(?:74_autoregressive|v2_comparison)_\d{8}_attempt\d{2,}", output.name):
        raise ValueError("new independent v2 attempt output name required")
    if "outputs" in parts and output.parent.name != "outputs":
        raise ValueError("nested or original output path forbidden")
    if output.is_relative_to(ROOT) and output.parent != ROOT / "outputs":
        raise ValueError("repository writes are confined to a new outputs attempt root")
    if output.exists() and not output.is_dir():
        raise ValueError("attempt output must be a directory")
    return output


def write_path(path, output):
    root, requested = safe_output(output), local_path(path)
    if requested == root or not requested.is_relative_to(root):
        raise ValueError("write must remain within the new exclusive attempt output")
    if any("legacy" in p.lower() or p == "test.jsonl" for p in requested.relative_to(root).parts):
        raise ValueError("protected child output name forbidden")
    return requested


def make_directory(path, output, *, exist_ok=False):
    path = write_path(path, output)
    path.mkdir(parents=True, exist_ok=exist_ok)
    return path


def sha256_file(path):
    value = hashlib.sha256()
    with local_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    return json.loads(local_path(path).read_text(encoding="utf-8"))


def write_json(path, value, *, output):
    with write_path(path, output).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def preserve_error(original, additional, label):
    if original is None:
        raise additional
    original.add_note(f"{label}: {type(additional).__name__}: {additional}")


def monotonic_boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()


def arm_configs(stage, configuration=None):
    if stage == "B":
        return {arm: {"kind": "process", "mode": "two_step" if arm == "rollout_l6_l12" else "l6",
                      "updates": 400 if arm == "equal_compute_l6" else 200}
                for arm in B_ARMS}
    if stage != "C" or not isinstance(configuration, dict):
        raise ValueError("C requires an explicit independently frozen configuration")
    required = ("mode", "model_specs", "initialization", "primary", "tolerances",
                "case_unit_selection", "adaptive_gate", "reference", "selection_evidence")
    if any(not configuration.get(key) for key in required) or configuration["mode"] not in ("l6", "two_step"):
        raise ValueError("nonempty C method, primary/tolerances/case units/reference/adaptive gate required")
    specs = configuration["model_specs"]
    initialization = configuration["initialization"]
    if (set(specs) != set(C_ARMS) or not initialization.get("anchor")
            or set(initialization.get("mapping", {})) != set(C_ARMS)):
        raise ValueError("C needs three explicit current model specs and mapped same-anchor initialization")
    for arm, spec in specs.items():
        if (spec.get("kind") != ("generic" if arm == "matched_generic" else "process")
                or not spec.get("model") or spec["model"].get("detach_between_steps") is not False
                or not initialization["mapping"][arm]):
            raise ValueError("C kind/full-BPTT model spec and explicit mapping required")
    return {arm: {"kind": specs[arm]["kind"], "model_spec": specs[arm]["model"],
                  "mode": configuration["mode"], "updates": 400} for arm in C_ARMS}


def planned_jobs(stage="B", configuration=None):
    arms = arm_configs(stage, configuration)
    seeds, ks = (B_SEEDS, (4,)) if stage == "B" else (C_SEEDS, (1, 2, 4))
    training = [{"phase": "train", "seed": seed, "arm": arm, "lead": None, "reasoning_steps": 4}
                for seed in seeds for arm in arms]
    scoring = [{"phase": "evaluate", "seed": seed, "arm": arm, "lead": lead, "reasoning_steps": k}
               for seed in seeds for arm in arms for lead in LEADS for k in ks]
    return training + scoring


def job_key(job):
    lead = "" if job["lead"] is None else f"_lead{job['lead']:03d}h"
    return f"{job['phase']}_seed{job['seed']}_{job['arm']}{lead}_k{job['reasoning_steps']}"


def worker_result_path(output, job):
    return write_path(Path(output) / "workers" / (job_key(job) + ".json"), output)


def train_output_dir(output, seed, arm):
    return write_path(Path(output) / f"seed{seed}/training/{arm}", output)


def evaluation_dir(output, job):
    return write_path(Path(output) / f"seed{job['seed']}/evaluation/{job['arm']}/lead_{job['lead']:03d}h/k{job['reasoning_steps']}", output)


def child_contract(protocol, job, model_spec):
    return {"kind": protocol["arm_configs"][job["arm"]]["kind"], "model": dict(model_spec),
            "data_identity": protocol["data"]["data_identity"],
            "source_sha256": protocol["sources"]["source_sha256"],
            "protocol_sha256": protocol["protocol_sha256"],
            "autoregression": {"excluded_sample_ids": list(protocol["windows"]["excluded_sample_ids"]),
                               "windows": protocol["windows"], "window_sha256": protocol["windows"]["window_sha256"],
                               "windows_sha256": digest(protocol["windows"]),
                               "mode": protocol["arm_configs"][job["arm"]]["mode"],
                               "lambda12": protocol["shared_controls"]["lambda12"] if protocol["arm_configs"][job["arm"]]["mode"] == "two_step" else 0.0,
                               "physical_rollout_bptt": True, "internal_reasoning_bptt": True},
            "parent_provenance": protocol["parents"].get(str(job["seed"])),
            "initialization": protocol["cpu_profile"]["pairing"][str(job["seed"])][job["arm"]],
            "scientific_claim": False, "limitations": list(protocol["limitations"])}


def build_protocol(*, stage, output, manifests, data, sources, sidecar, windows, parents,
                   profile, code, gpu_uuid, round_started_perf_counter, boot_id, configuration=None):
    planned, hard = BUDGETS[stage]
    body = {"format": FORMAT, "stage": stage, "output": str(safe_output(output)),
            "manifests": str(local_path(manifests)), "data": data, "sources": sources,
            "sidecar": sidecar, "windows": windows, "parents": parents, "cpu_profile": profile,
            "code": code, "configuration": configuration, "arm_configs": arm_configs(stage, configuration),
            "jobs": planned_jobs(stage, configuration), "shared_controls": dict(CONTROLS),
            "baseline_reporting": BASELINE_REPORTING,
            "reporting": B_REPORTING if stage == "B" else {key: configuration[key] for key in
                        ("primary", "tolerances", "case_unit_selection", "adaptive_gate", "reference", "selection_evidence")},
            "gpu": {"policy": "shared", "uuid": gpu_uuid, "estimated_peak_mib": 2048,
                    "headroom_margin_mib": 2048, "threads": 4,
                    "estimate_source": "conservative engineering estimate; owned peak anticipated below 2GiB, not yet measured",
                    "gate": "memory.free >= max(estimated2048MiB, actual owned reserved peak) + 2048MiB before every spawn",
                    "billing": "continuous first GPU spawn through last owned reap; startup, gaps, failures and owned cleanup fully charged",
                    "neighbor_policy": "nvidia-smi read-only; direct owned handles only; no neighbor signals"},
            "round_started_perf_counter": round_started_perf_counter, "monotonic_boot_id": boot_id,
            "planned_seconds": planned, "hard_cap_seconds": hard, "cleanup_reserve_seconds": CLEANUP_SECONDS,
            "whole_clock_scope": "earliest CPU CLI entry including imports/prepare/freeze, prepare-run gap, training, evaluation, aggregation and owned cleanup; same boot",
            "whole_round_cost_reference": str(Path(output) / "attempt.json"),
            "scientific_claim": False, "limitations": list(LIMITATIONS), "test_read": False,
            "frozen_before_any_step": True, "no_total_gpu_cap": True, "no_retry_or_resurrection": True}
    return validate_protocol({**body, "protocol_sha256": digest(body)})


def validate_protocol(protocol):
    body = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if protocol.get("protocol_sha256") != digest(body):
        raise ValueError("v2 frozen protocol digest mismatch")
    stage = protocol.get("stage")
    if stage not in BUDGETS:
        raise ValueError("independent B/C stage required")
    planned, hard = BUDGETS[stage]
    if (protocol.get("format") != FORMAT or protocol.get("scientific_claim") is not False
            or protocol.get("test_read") is not False or not protocol.get("limitations")
            or protocol.get("frozen_before_any_step") is not True or protocol.get("no_total_gpu_cap") is not True
            or protocol.get("no_retry_or_resurrection") is not True
            or protocol.get("planned_seconds") != planned or protocol.get("hard_cap_seconds") != hard
            or protocol.get("cleanup_reserve_seconds") != CLEANUP_SECONDS
            or protocol.get("shared_controls") != CONTROLS
            or protocol.get("arm_configs") != arm_configs(stage, protocol.get("configuration"))
            or protocol.get("jobs") != planned_jobs(stage, protocol.get("configuration"))):
        raise ValueError("frozen v2 jobs/recipe/budget/flags changed")
    if protocol.get("baseline_reporting") != BASELINE_REPORTING:
        raise ValueError("mandatory B/C persistence and climatology reporting changed")
    if stage == "B" and protocol.get("reporting") != B_REPORTING:
        raise ValueError("predeclared B reporting/selection gate changed")
    anchor, boot = protocol.get("round_started_perf_counter"), protocol.get("monotonic_boot_id")
    if (isinstance(anchor, bool) or not isinstance(anchor, (float, int)) or not math.isfinite(anchor)
            or anchor < 0 or not isinstance(boot, str) or not boot):
        raise ValueError("frozen finite same-boot prepare anchor required")
    output = safe_output(protocol["output"])
    if (str(output) != protocol["output"] or protocol.get("whole_round_cost_reference") != str(output / "attempt.json")
            or str(local_path(protocol["manifests"])) != protocol["manifests"]):
        raise ValueError("absolute frozen output/manifest paths required")
    gpu = protocol["gpu"]
    if (gpu.get("policy") != "shared" or gpu.get("estimated_peak_mib") != 2048
            or gpu.get("headroom_margin_mib") != 2048 or gpu.get("threads") != 4
            or not isinstance(gpu.get("uuid"), str) or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", gpu["uuid"])
            or any(key in gpu for key in ("max_gpu_seconds", "max_round_seconds", "worker_seconds"))):
        raise ValueError("fixed shared UUID/2048+2048 gate without old GPU/worker caps required")
    windows = protocol["windows"]
    if (not isinstance(windows.get("excluded_sample_ids"), list)
            or len(set(windows["excluded_sample_ids"])) != len(windows["excluded_sample_ids"])):
        raise ValueError("explicit exact window exclusions required")
    if (len(protocol["data"].get("channels", [])) != 17
            or len(protocol["data"].get("units", [])) != 17
            or set(protocol["data"].get("evaluation_cases", {})) != {str(lead) for lead in LEADS}
            or protocol["data"].get("test_read") is not False):
        raise ValueError("17 variables and full exact per-lead validation cohorts required")
    if stage == "B" and set(protocol["parents"]) != {str(seed) for seed in B_SEEDS}:
        raise ValueError("both exact seed41/42 selected400 parents required")
    for parent in protocol["parents"].values():
        parent_root = local_path(parent["original_output"])
        if (output.is_relative_to(parent_root) or parent_root.is_relative_to(output)
                or parent["model_spec"].get("detach_between_steps") is not False):
            raise ValueError("read-only parent output and new full-BPTT child model spec required")
    return protocol


def verify_protocol(path):
    value = validate_protocol(read_json(path))
    if local_path(path) != Path(value["output"]) / "protocol.json":
        raise ValueError("protocol path/output mismatch")
    return value
