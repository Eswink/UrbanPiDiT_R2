"""N1 D2 preregistration and primary reader; standard library only, no data reads.

The driver writes the returned JSON exclusively before any optimizer step, supplies
its own authorization/code records, and enforces the registered resource limits.
Neither registration nor a nonempty authorization record grants permission to run.
The frozen brief, not the previous subtraction round's four-arm rule, is the source
of this round's reading. Tensor/parameter checks belong to r7_frozen_z_intervention.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from training.r7_rw_b_subtraction_protocol import (
    BATCH_SIZE, CLIP, COMPARATOR_DEPTH, DEADLINE_SECONDS,
    EARLY_STOPPING_PATIENCE, EVALUATION_LEADS, EVALUATION_MAX_SAMPLES,
    FLOP_CONVENTION, LR, MINIMUM_IMPROVEMENT, MINIMUM_LR_RATIO,
    PROCESS_WEIGHT, REASONING_STEPS, RW_A_CONFIG, RW_B_CONFIG, SEEDS,
    SEGMENT_NOTE, SWITCH_KEYS, UPDATES, VALIDATION_EVERY, VALIDATION_LEADS,
    WARMUP_UPDATES, WEIGHT_DECAY,
)

RW_A_ARM = "process_spacetime_rwa"
RW_B_ARM = "process_local_solver"
FROZEN_Z_ARM = "process_local_solver_frozen_z"
ARMS = ((RW_A_ARM, "process", dict(RW_A_CONFIG)),
        (RW_B_ARM, "process", dict(RW_B_CONFIG)),
        (FROZEN_Z_ARM, "process", dict(RW_B_CONFIG)))
ARM_NAMES = tuple(entry[0] for entry in ARMS)
PAIRS = ((FROZEN_Z_ARM, RW_A_ARM), (RW_B_ARM, RW_A_ARM), (FROZEN_Z_ARM, RW_B_ARM))
PAIR_ROLES = ("primary question", "contemporaneous reference", "paired carrier contrast")
PRIMARY_VARIABLE = "t2m"
PRIMARY_LEADS = EVALUATION_LEADS
FINAL_WEIGHT = 2.0  # r7_streaming.streaming_forecast_backward's existing runner default.
GPU_CAP_HOURS = 0.45
GPU_CAP_SECONDS = 1620.0
BRIEF_POINTER = "docs/goals/main-model-v2-pivot-audit.md §3 (D2), §5"
COMPARATOR_POINTER = "training/r7_coreasoning_compare.py (#60), depth=0"
# Construction constants from the control's canonical make_spec, not outcome thresholds.
FROZEN_Z_STD = 0.02
FROZEN_Z_SEED_OFFSET = 1_000_003
SPEC_FIELDS = frozenset(("format", "seed", "random_seed", "shape", "token_hw",
                         "patch_size", "std", "tensor_sha256"))
PRIMARY_DECISION_TEXT = (
    "来源：docs/goals/main-model-v2-pivot-audit.md §3，D2 臂的预声明读法。原文：\n"
    "1. 冻结随机 Z 臂的 48h 与 72h 仍然 worsened（逐 seed 同号）⇒ 排除「学到的 Z」为主因，"
    "恶化落在「锚定提案 + 门控被施加」这条通路或更外层；\n"
    "2. 48h/72h 不再 worsened ⇒ 学到的 Z 是载体——与 D1 探针的 (a) 倾向冲突，"
    "必须在证据页里明确记录这一冲突并重审 D1 的相关读数；\n"
    "3. 两 seed 反号（unresolved）⇒ 读作「在 2 seed × 400 updates 下不可分辨」，"
    "按停止条件 3 触发 N2d 提议。\n"
    "本轮限域读取：第一条只排除 learned Z 对该恶化的必要性（在此预算范围），不做普遍主因断言；"
    "第二条只登记 learned Z 为载体候选并记录/重审与 D1 的冲突。"
    "任一 primary 长 lead unresolved（含 #60 的零 delta、缺失 cell/seed）即不可分辨、提议 N2d 并停。"
    "48/72h 两 lead 不同方向仅写读数 mixed，不擅自归到支持 carrier；不新增门槛或臂。"
    "t2m 五时效均保留逐 seed 读数，只在 exact seeds 41/42 同号时报告均值。"
    "三对均为本轮配对，RW-B−RW-A 是 contemporaneous reference，frozen−RW-B 是单独的 paired "
    "carrier contrast，不池化历史，不作显著性判断。"
)
BRANCH_RULE = {
    "exclude-learned-z-necessity": {
        "if": "primary 48h and 72h both worsened by the existing per-seed sign rule",
        "reading": (f"{BRIEF_POINTER} 第1条：冻结随机 Z 的48/72h仍 worsened，"
                    "在2 seed ×400 updates与本轮预算范围排除 learned Z 的必要性；"
                    "恶化在锚定提案+门控被施加的通路或更外层，不做普遍主因断言。")},
    "learned-z-carrier-candidate": {
        "if": "primary 48h and 72h both no longer worsened (supported; zero is unresolved)",
        "reading": (f"{BRIEF_POINTER} 第2条：48/72h不再 worsened，learned Z 为载体候选；"
                    "明确记录与D1探针(a)倾向的冲突并重审相关读数，不宣称机制已证实。")},
    "cannot-distinguish": {
        "if": "either primary long lead unresolved, including missing cells/seeds or zero",
        "reading": (f"{BRIEF_POINTER} 第3条与停止条件3：在2 seed ×400 updates下不可分辨；"
                    "提议N2d并停，不追加臂数，不自动执行下一节点。")},
    "mixed-long-leads": {
        "if": "the two settled primary long leads have different directions",
        "reading": (f"{BRIEF_POINTER} 的逐lead读数：48/72h读数 mixed，不能合称仍 worsened"
                    "或不再 worsened；不归为支持carrier，不新增门槛或臂。")},
}
LIMITATIONS = [
    "two seeds at 400 updates under a bounded N1 D2 budget, not convergence or SOTA",
    "seed sign agreement is descriptive consistency, not significance; no new threshold",
    "learned-Z necessity is read only within this budget, not a universal main-cause claim",
    "a carrier candidate conflicts with D1 probe (a) and requires explicit recording/review",
    "total and trainable parameters/FLOPs are not capacity matched; frozen solver parameters "
    "reduce trainable capacity and the measured difference must be reported",
    "same-round RW-A/RW-B are retrained; historical rounds are never pooled",
    "one winter ERA5 segment in one region; train/val only, test remains sealed",
    "supplied identities, counts and authorization records are recorded, not authenticated; "
    "the driver must verify tensor identity/freezing and enforce resource caps",
]
DATA_POLICY = {
    "train_read": True, "val_read": True, "test_read": False,
    "test_policy": "sealed; test is never read, including for selection or cost measurement",
    "split_mode": "time_ranges (docs/decisions/0005-*.md, 0008-*.md)",
    "normalization": "store train-only centered mean/std, applied at read time",
    "segment": SEGMENT_NOTE,
    "identity_policy": "driver-supplied source/data identity; driver checks preflight receipt, "
                       "source SHA256 and BUILD_COMPLETE; this module does not read data",
}


def canonical_digest(value):
    """Byte-compatible with r7_experiment.canonical_digest without importing torch."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _record(value, name):
    if (not isinstance(value, (dict, list, str)) or not value
            or (isinstance(value, str) and not value.strip())):
        raise ValueError(f"{name} requires a nonempty JSON record; none is fabricated")
    return _json_copy(value)


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value, name):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite number")
    return value


def _specifications(specs):
    """Check make_spec's JSON contract, not the tensor bytes (a driver check)."""
    if not isinstance(specs, dict) or set(specs) != {str(seed) for seed in SEEDS}:
        raise ValueError("intervention_specs must contain exactly string seeds 41 and 42")
    specs = _json_copy(specs)
    grids = []
    for seed in SEEDS:
        spec = specs[str(seed)]
        if not isinstance(spec, dict) or set(spec) != SPEC_FIELDS:
            raise ValueError("noncanonical frozen-Z spec; no thaw/freeze model switch is allowed")
        grid = spec["token_hw"]
        if not isinstance(grid, list) or len(grid) != 2:
            raise ValueError("frozen-Z token_hw must be [rows, cols]")
        for side in grid:
            _integer(side, "token_hw")
        expected = {"format": "r7-frozen-z-v1", "seed": seed,
                    "random_seed": seed + FROZEN_Z_SEED_OFFSET,
                    "shape": [1, grid[0] * grid[1], RW_B_CONFIG["dim"]],
                    "token_hw": grid, "patch_size": RW_B_CONFIG["patch_size"],
                    "std": FROZEN_Z_STD}
        actual = {key: spec[key] for key in expected}
        if canonical_digest(actual) != canonical_digest(expected):
            raise ValueError("frozen-Z spec differs from make_spec's frozen construction")
        digest = spec["tensor_sha256"]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("frozen-Z tensor_sha256 must be a CPU FP32 byte SHA256")
        grids.append(grid)
    if grids[0] != grids[1]:
        raise ValueError("every declared seed must use the same frozen-Z geometry")
    return specs


def _comparisons():
    return [{"focus": focus, "baseline": baseline, "depth": COMPARATOR_DEPTH,
             "role": role, "reading": f"{focus} - {baseline}"}
            for (focus, baseline), role in zip(PAIRS, PAIR_ROLES)]


def _fixed_fields(channels):
    """All immutable registration/control/cap fields, checked even after a rehash."""
    _integer(channels, "channels")
    comparisons = _comparisons()
    return {
        "format": "r7-n1-frozen-z-protocol-v1", "version": 1,
        "frozen_before_any_step": True, "scientific_claim": False, "test_read": False,
        "brief_pointer": BRIEF_POINTER, "limitations": list(LIMITATIONS),
        "channels": channels, "seeds": list(SEEDS),
        "seed_policy": "all declared seeds share one complete protocol digest; report every seed",
        "historical_pooling": False,
        "primary_registration": {
            "registered_before_first_optimizer_step": True, "frozen_before_any_step": True,
            "variable": PRIMARY_VARIABLE, "leads_hours": list(PRIMARY_LEADS),
            "pairs": comparisons, "decision_text": PRIMARY_DECISION_TEXT,
            "branch_rule": _json_copy(BRANCH_RULE), "brief_pointer": BRIEF_POINTER,
            "aggregate_rule": (f"{COMPARATOR_POINTER}: exact declared seeds, finite deltas; "
                               "improved/worsened only if all deltas are strictly same-sign; "
                               "zero/missing/disagreement is unresolved, never averaged away"),
            "secondary_reporting": "all variables x five leads for all three pairs, separately "
                                   "from primary; no historical pooling or significance test"},
        "comparisons": comparisons,
        "shared_controls": {
            "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY,
            "lr_schedule": {"kind": "linear_warmup_cosine", "warmup_updates": WARMUP_UPDATES,
                            "minimum_lr_ratio": MINIMUM_LR_RATIO, "end_update": UPDATES},
            "max_updates": UPDATES, "batch_size": BATCH_SIZE, "clip": CLIP,
            "validation_every": VALIDATION_EVERY,
            "validation_lead_hours": list(VALIDATION_LEADS),
            "early_stopping": {"patience": EARLY_STOPPING_PATIENCE,
                               "minimum_relative_improvement": MINIMUM_IMPROVEMENT},
            "checkpoint_selection_rule": "lowest mean latitude-weighted normalized validation "
                                         "MSE; ties keep the earlier checkpoint",
            "loss": "latitude-weighted normalized MSE, streamed truncated BPTT",
            "reasoning_steps": REASONING_STEPS, "process_weight": PROCESS_WEIGHT,
            "final_weight": FINAL_WEIGHT,
            "final_weight_axis": "linear 1..2 over K+1 internal drafts, not physical lead; "
                                 "all drafts use the same +6h target (runner default)",
            "sample_order": "torch.randperm(len(dataset), generator=manual_seed(seed+epoch))",
            "evaluation_leads_hours": list(EVALUATION_LEADS),
            "evaluation_max_samples": EVALUATION_MAX_SAMPLES,
            "evaluation_policy": "free autoregressive rollout on val for every arm/seed/lead",
            "selection_split": "val only; test stays sealed", "comparator_depth": COMPARATOR_DEPTH,
            "comparison": COMPARATOR_POINTER, "wall_clock_limit_seconds": DEADLINE_SECONDS},
        "intervention": {
            "arm": FROZEN_Z_ARM, "implementation": "driver/control only; no model_config key",
            "construction": "make_spec(seed, token_hw, dim, patch_size); CPU FP32 contiguous "
                            "randn at seed+1000003, std .02; same-shape fixed random Z",
            "persistent_buffer": "_frozen_z_value", "requires_grad": False,
            "freeze_parameters": ["solver_init", "solver_cell.*"],
            "cell_policy": "original solver_cell executes; discard its output and return the "
                           "same frozen-Z buffer at every internal step, no learned recurrence",
            "other_paths": "RW-B model_config unchanged; all other parameters/pathways retained",
            "verification_required": ["CPU FP32 tensor SHA256 and persistent buffer identity",
                                      "parameter identity and requires_grad flags",
                                      "non-intervened RW-B pathways bitwise identical",
                                      "checkpoint restore retains the intervention"],
            "verification_status": "required driver checks; not performed by this pure reader"},
        "arm_pairing": {
            "shared_initialization_anchor": RW_A_ARM,
            "rule": "same-seed construction and reference transfer before the first step; "
                    "measure shared tensor equality and applied/ignored names; frozen arm "
                    "retains RW-B parameters and differs only by the registered intervention",
            "explicitly_not_matched": ["total/trainable parameter counts", "forward/backward FLOPs",
                                       "wall time", "peak memory"]},
        "budget": {
            "gpu_hours_cap": GPU_CAP_HOURS, "gpu_seconds_cap": GPU_CAP_SECONDS,
            "global_deadline_seconds": DEADLINE_SECONDS, "max_devices": 1,
            "single_device": True, "deadline_reset_per_seed": False,
            "scope": "one whole N1 D2 round: three arms x two seeds; one clock begins before "
                     "select_device/first CUDA allocation and covers GPU setup, training, "
                     "validation, evaluation, sync and interstage time through final GPU sync; "
                     "elapsed seconds x one exclusive device <=1620s; CPU-only finalize is "
                     "reported in global wall time, not charged again as GPU time",
            "global_deadline_scope": "one monotonic <=1800s deadline from driver start, including "
                                     "preflight, CPU cost measurement, registration and all seeds",
            "exclusivity": "one exclusive device; no concurrent GPU workloads, multi-GPU or "
                           "overlap that hides cost; stop if exclusivity cannot be established",
            "stop_policy": "stop at either cap or insufficient ledger; partial/queued/cancelled/"
                           "skipped is not PASS; no automatic extension or next-node run"},
        "authorization_policy": "require an explicit nonempty caller record; presence alone "
                                "is not proof of human authorization (decision 0021)",
        "code_identity_policy": "opaque driver-supplied SHA/code.zip/tree identity; no network lookup",
        "cost_reporting": {
            "measurement_device": "cpu", "measured_before_registration": True,
            "measurement_policy": "driver measures each arm after its intervention on CPU; "
                                  "record actual total/trainable parameters, forward and "
                                  "forward+backward FLOPs, including frozen arm's lower trainable count",
            "tables": ["arm_table.csv: total/trainable parameters and forward/forward+backward FLOPs",
                       "training_table.csv: elapsed time and seconds/update",
                       "memory_table.csv: peak allocated/reserved bytes",
                       "rmse_table.csv: every variable x five leads x three arms x both seeds",
                       "case_table.csv: scored cases per arm/seed/lead, test_read=False"],
            "note": "CPU FLOP measurements are not GPU timing; backward is measured, never "
                    "assumed 2x forward; all four cost views and actual GPU-h are required"},
        "flop_convention": FLOP_CONVENTION,
    }


def _arm_entries(channels, measured):
    if not isinstance(measured, dict) or set(measured) != set(ARM_NAMES):
        raise ValueError("cost measurements must cover exactly the three declared arms")
    entries = []
    for name, kind, config in ARMS:
        cost = measured[name]
        values = {key: _integer(cost[key], f"{name}.{key}", 0 if key == "trainable_parameters" else 1)
                  for key in ("parameters", "trainable_parameters", "forward_flops",
                              "forward_backward_flops")}
        if values["trainable_parameters"] > values["parameters"]:
            raise ValueError("trainable parameters cannot exceed total parameters")
        entries.append({"name": name, "kind": kind,
                        "model_config": {"in_channels": channels, "out_channels": channels,
                                         "history_steps": 2, **config},
                        "switches": {key: config.get(key, default) for key, default in SWITCH_KEYS},
                        **values})
    baseline, frozen = entries[1], entries[2]
    if (frozen["parameters"] != baseline["parameters"]
            or frozen["trainable_parameters"] >= baseline["trainable_parameters"]):
        raise ValueError("frozen-Z keeps RW-B total parameters but must freeze solver parameters")
    return entries


def refused_test_manifest(manifest):
    """Validation-only: never pass a sealed test manifest to a dataset reader."""
    if Path(manifest).name == "test.jsonl":
        raise ValueError("N1 D2 is train/validation-only; test stays sealed")


def protocol_payload(manifests_dir, identity, channels, measured, *,
                     intervention_specs, authorization, code_identity, source_identity=None):
    """Return detached JSON with one digest covering every seed; write nothing."""
    body = _fixed_fields(channels)
    manifests = Path(manifests_dir)
    if identity is None or identity == "" or identity == {}:
        raise ValueError("driver-supplied source/data identity is required")
    body.update(
        data={**DATA_POLICY, "manifests_dir": str(manifests),
              "store": str(manifests.parent / "cache.zarr"),
              "train_manifest": str(manifests / "train.jsonl"),
              "val_manifest": str(manifests / "val.jsonl"),
              "test_manifest": str(manifests / "test.jsonl"),
              "data_identity": identity, "source_identity": source_identity,
              "source_identity_status": ("not supplied; not independently verified"
                                         if source_identity is None else
                                         "driver-supplied; not independently verified"),
              "source_identity_scope": "source/receipt/preflight/BUILD_COMPLETE identity only; "
                                       "a source byte digest does not read test fields for "
                                       "evaluation or selection; test remains sealed"},
        arms=_arm_entries(channels, measured),
        intervention_specs=_specifications(intervention_specs),
        authorization=_record(authorization, "authorization"),
        code_identity=_record(code_identity, "code_identity"))
    body = _json_copy(body)
    return dict(body, protocol_sha256=canonical_digest(body))


def verified_registration(protocol_path):
    """Read only the registration JSON; reject rehashed changes to frozen fields."""
    frozen = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    try:
        if not isinstance(frozen, dict):
            raise ValueError("registration must be a JSON object")
        body = {key: value for key, value in frozen.items() if key != "protocol_sha256"}
        if frozen.get("protocol_sha256") != canonical_digest(body):
            raise ValueError("protocol digest mismatch")
        for key, expected in _fixed_fields(frozen["channels"]).items():
            if canonical_digest(frozen.get(key)) != canonical_digest(expected):
                raise ValueError(f"frozen registration differs at {key}")
        data = frozen["data"]
        expected = protocol_payload(
            data["manifests_dir"], data["data_identity"], frozen["channels"],
            {entry["name"]: entry for entry in frozen["arms"]},
            intervention_specs=frozen["intervention_specs"],
            authorization=frozen["authorization"], code_identity=frozen["code_identity"],
            source_identity=data["source_identity"])
        if (canonical_digest(frozen["arms"]) != canonical_digest(expected["arms"])
                or len(frozen["arms"]) != len(ARMS)):
            raise ValueError("frozen arm configurations or cost fields differ")
        if canonical_digest(data) != canonical_digest(expected["data"]):
            raise ValueError("frozen data read policies or manifest paths differ")
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"invalid N1 D2 registration: {error}") from error
    return frozen


def _read_cell(entry):
    result = {"seed_deltas": {}, "delta_seed_mean": None, "sign_consistent": False,
              "outcome": "unresolved", "unit": "K", "reading": "missing comparator cell"}
    if entry is None:
        return result
    deltas = entry.get("seed_deltas")
    if not isinstance(deltas, dict) or any(type(seed) is not str for seed in deltas):
        raise ValueError("pair_cells seed_deltas must be keyed by string seed IDs")
    if set(deltas) - {str(seed) for seed in SEEDS}:
        raise ValueError("unexpected seed; historical or extra seeds must not be pooled")
    deltas = {seed: _finite(value, f"seed {seed} delta") for seed, value in deltas.items()}
    result["seed_deltas"] = deltas
    if entry.get("unit") != "K":
        raise ValueError("primary t2m deltas must be in physical K")
    if (type(entry.get("sign_consistent")) is not bool
            or entry.get("outcome") not in ("improved", "worsened", "unresolved")):
        raise ValueError("cell lacks the comparator's sign_consistent/outcome")
    if set(deltas) != {str(seed) for seed in SEEDS}:
        result["reading"] = "missing declared seed; no mean or verdict"
        return result
    negative = all(delta < 0 for delta in deltas.values())
    positive = all(delta > 0 for delta in deltas.values())
    if not (negative or positive):
        result["reading"] = f"{COMPARATOR_POINTER}: zero or disagreeing signs is unresolved"
        return result
    if not entry["sign_consistent"] or entry["outcome"] == "unresolved":
        result["reading"] = "comparator cell unresolved; no seed mean or verdict"
        return result
    expected = "improved" if negative else "worsened"
    if entry["outcome"] != expected:
        raise ValueError("comparator outcome contradicts the signed seed deltas")
    result.update(delta_seed_mean=mean(list(deltas.values())), sign_consistent=True,
                  outcome="supported" if expected == "improved" else "worsened",
                  reading=f"{COMPARATOR_POINTER}: every declared seed has "
                          f"{'lower' if negative else 'higher'} t2m RMSE")
    return result


def _read_pair(pairs, focus, baseline, role):
    pair = f"{focus} - {baseline}"
    block = pairs.get(pair, {})
    if block and (block.get("focus_arm", focus) != focus
                  or block.get("baseline_arm", baseline) != baseline):
        raise ValueError("pair_cells arm identity contradicts the declared pair")
    cells = block.get("cells", {})
    per_lead = {str(lead): _read_cell(cells.get(f"{lead}h|{PRIMARY_VARIABLE}"))
                for lead in PRIMARY_LEADS}
    outcomes = [cell["outcome"] for cell in per_lead.values()]
    long_leads = {str(lead): per_lead[str(lead)]["outcome"] for lead in (48, 72)}
    return {"pair": pair, "role": role, "reading": pair, "variable": PRIMARY_VARIABLE,
            "leads_hours": list(PRIMARY_LEADS), "per_lead": per_lead,
            "verdict_counts": {name: outcomes.count(name)
                               for name in ("supported", "worsened", "unresolved")},
            "long_lead_outcomes": long_leads,
            "long_leads_worsened": all(value == "worsened" for value in long_leads.values()),
            "scientific_claim": False, "test_read": False, "limitations": list(LIMITATIONS)}


def primary_reading(pairs):
    """Read pair_cells output, preserving all leads and all three separate roles."""
    views = [_read_pair(pairs, focus, baseline, role)
             for (focus, baseline), role in zip(PAIRS, PAIR_ROLES)]
    long_leads = list(views[0]["long_lead_outcomes"].values())
    if "unresolved" in long_leads:
        branch = "cannot-distinguish"
    elif all(outcome == "worsened" for outcome in long_leads):
        branch = "exclude-learned-z-necessity"
    elif all(outcome == "supported" for outcome in long_leads):
        branch = "learned-z-carrier-candidate"
    else:
        branch = "mixed-long-leads"
    return {"format": "r7-n1-frozen-z-primary-reading-v1", "version": 1,
            "scientific_claim": False, "test_read": False, "limitations": list(LIMITATIONS),
            "brief_pointer": BRIEF_POINTER, "decision_text": PRIMARY_DECISION_TEXT,
            "branch_rule": _json_copy(BRANCH_RULE), "branch": branch,
            "branch_reading": BRANCH_RULE[branch]["reading"],
            "stop_required": branch == "cannot-distinguish",
            "next_node_proposal": "N2d" if branch == "cannot-distinguish" else None,
            "primary_question": views[0], "round_reference": views[1],
            "paired_carrier_contrast": views[2],
            "reporting_order": [view["pair"] for view in views], "historical_pooling": False,
            "note": "primary reads only frozen−RW-A; contemporaneous reference and paired "
                    "carrier contrast are reported separately; no significance test"}
