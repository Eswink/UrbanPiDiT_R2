"""Round-three capacity attribution for the space-time inputs (#71 M1).

Round two measured a large effect and left one hole open: ``B - A`` (space-time
inputs on, real fields) is -1.0..-2.0 K on t2m at 24-72 h, but B carries 19,488
more parameters than A and nothing in that run separated *what the module is
shown* from *what the module can represent*. Round three pays for that
separation with two control arms and preregisters what it is allowed to conclude.

Four arms, one protocol, all trained from scratch:

- ``process_pooled``                    (A) neither pathway - the reference;
- ``process_spacetime_only``            (B) the pathway with the real fields;
- ``process_spacetime_constant``        (E) the pathway, parameters and compute,
  with the fields replaced by same-shape zeros: no information in the input, so
  the module can only contribute one constant vector;
- ``process_spacetime_shuffled``        (P) the pathway with real fields rolled
  by one along the *sample* axis: real values, wrong pairing.

So ``B - A`` = information + capacity/bias, ``B - E`` = information at fixed
capacity, ``E - A`` = capacity/bias alone, and ``P - A`` = capacity with real
values but a broken correspondence. E and B are bitwise identical in every
parameter tensor and identical in measured FLOPs, which is what makes ``B - E``
a contrast of inputs rather than of capacity.

What is held fixed: the M2 two-month real-ERA5 segment (train and val manifests
only; the test manifest is never opened), the optimizer, schedule, update
budget, batch size, gradient clip, validation frequency, checkpoint selection and
early stopping of the round-two/B2 protocol, and the seed set declared in
``SEEDS`` before anything runs. All four arms are built from one seed with the
added module constructed last under a rewound stream, and ``arm_pairing``
*measures* that the shared tensors are bitwise equal, that the anchor loads into
every other arm through the trainer's own transfer rule, and that E and B are
identical tensor for tensor.

The primary is registered in ``protocol.json`` and frozen before the first
optimizer step: variable t2m, leads 6/12/24/48/72 h, pairs B-A, B-E, E-A and
P-A, with the reading rules written out in ``PRIMARY_DECISION_TEXT``. The
per-variable, per-lead table is reported in full for all 17 variables and 5
leads; the primary verdict is separate from the aggregate counts, and no
threshold is introduced beyond the comparator's own per-seed sign rule.

    python scripts/study_r7_71_72_round_three.py --mode seed --seed 41 \\
        --device-index 0 --manifests outputs/r7_m2_segment/store/manifests \\
        --out outputs/r7_71_72_round_three
    python scripts/study_r7_71_72_round_three.py --mode finalize \\
        --manifests outputs/r7_m2_segment/store/manifests --out outputs/r7_71_72_round_three
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEEDS = (41, 42, 43)
UPDATES = 400
BATCH_SIZE = 2
LR = 2e-4
CLIP = 1.0
WEIGHT_DECAY = 1e-4
WARMUP_UPDATES = 80
MINIMUM_LR_RATIO = 0.1
VALIDATION_EVERY = 100
EARLY_STOPPING_PATIENCE = 4
MINIMUM_IMPROVEMENT = 0.001
VALIDATION_LEADS = (6,)
EVALUATION_LEADS = (6, 12, 24, 48, 72)
REASONING_STEPS = 3
PROCESS_WEIGHT = 0.0  # the process arm is generic+process tokens; not the process loss
EVALUATION_MAX_SAMPLES = 64
COMPARATOR_DEPTH = 0
# The round's own wall-clock cap, checked inside the run: a seed that has already
# spent its budget does not get to start another arm and quietly overrun it.
DEADLINE_SECONDS = 1800.0

BASE_ARM_CONFIG = {"architecture": "window", "dim": 192, "depth": 4, "heads": 4,
                   "window_size": 4, "patch_size": 2, "dropout": 0.0,
                   "anchored_processes": 8, "free_processes": 8,
                   "use_forecast_feedback": True,
                   "default_reasoning_steps": REASONING_STEPS}
FIELDS = "fields"
ARMS = (
    ("process_pooled", "process", dict(BASE_ARM_CONFIG)),
    ("process_spacetime_only", "process",
     dict(BASE_ARM_CONFIG, spacetime_inputs=True, spacetime_field_mode=FIELDS)),
    ("process_spacetime_constant", "process",
     dict(BASE_ARM_CONFIG, spacetime_inputs=True, spacetime_field_mode="constant")),
    ("process_spacetime_shuffled", "process",
     dict(BASE_ARM_CONFIG, spacetime_inputs=True, spacetime_field_mode="shuffled")),
)
ARM_NAMES = tuple(entry[0] for entry in ARMS)
BASELINE_ARM = "process_pooled"
# The five pairs the round reports, in the notation of the study: focus - baseline.
# P - B is carried along because the table already holds both sides of it and
# dropping it would be a choice about the report rather than about the data.
PAIRS = (
    ("process_spacetime_only", "process_pooled"),          # B - A
    ("process_spacetime_only", "process_spacetime_constant"),   # B - E
    ("process_spacetime_constant", "process_pooled"),      # E - A
    ("process_spacetime_shuffled", "process_pooled"),      # P - A
    ("process_spacetime_constant", "process_spacetime_shuffled"),  # E - P
    ("process_spacetime_shuffled", "process_spacetime_only"),      # P - B
)
REQUIRED_PAIRS = (
    ("process_spacetime_only", "process_pooled"),
    ("process_spacetime_only", "process_spacetime_constant"),
    ("process_spacetime_constant", "process_pooled"),
    ("process_spacetime_shuffled", "process_pooled"),
    ("process_spacetime_constant", "process_spacetime_shuffled"),
)

PRIMARY_VARIABLE = "t2m"
PRIMARY_LEADS = (6, 12, 24, 48, 72)
PRIMARY_PAIRS = (
    ("process_spacetime_only", "process_pooled"),
    ("process_spacetime_only", "process_spacetime_constant"),
    ("process_spacetime_constant", "process_pooled"),
    ("process_spacetime_shuffled", "process_pooled"),
)
# Frozen before the first optimizer step; written here, never edited after the run.
PRIMARY_DECISION_TEXT = (
    "Read on the three-seed paired mean delta of t2m, per lead, only where the "
    "comparator's per-seed sign rule makes the cell sign-consistent (disagreement "
    "is unresolved and is never averaged into a verdict). Write s_E = (E-A)/(B-A), "
    "s_P = (P-A)/(B-A) and s_I = (B-E)/(B-A) for that cell. Then: (i) if s_E >= 0.5 "
    "the cell reads 'capacity/bias dominated - the module without information "
    "reproduces at least half of B-A'; (ii) if s_E <= 0.25 and s_I >= 0.75 the cell "
    "reads 'information dominated'; (iii) between 0.25 and 0.5 the cell reads "
    "'mixed, attribution unresolved' and no attribution is claimed; (iv) if E-A is "
    "opposite in sign to B-A the cell reads 'capacity/bias does not reproduce the "
    "gain at this lead'. The headline of the round is the modal reading over the "
    "sign-consistent leads with the full per-lead table reported beside it; if the "
    "leads disagree, that disagreement is the result and is reported as such. P is "
    "reported next to every cell: if s_P >= 0.5 as well, the round states that "
    "real-valued but mispaired fields suffice for at least half of B-A. The "
    "variable and the leads above are fixed by this text and are never re-picked "
    "after the numbers exist; no threshold beyond these fractions is introduced, "
    "and no existing threshold is relaxed."
)
FIELD_MODE_NOTE = (
    "field_mode is applied inside SpacetimeConditioning.forward, after "
    "require_spacetime_fields: the dataset, the rollout and every whitelisted path "
    "still carry the real fields and a missing or out-of-range field still raises, "
    "so a control arm cannot quietly read a different input path. 'constant' "
    "substitutes same-shape zeros (the module's contribution collapses to one "
    "vector for every position and sample); 'shuffled' rolls init_utc_hour and "
    "init_day_of_year by one along the sample axis, deterministically, leaving the "
    "values themselves a permutation of the real ones. A batch of one has no other "
    "sample to borrow from, so the roll is the identity there: a within-batch "
    "mismatch needs a partner, and every within-batch permutation has that "
    "property. That is declared rather than hidden because the validation and "
    "published-evaluation paths score one window per forward pass (batch 1), so "
    "the P arm's mismatch acts on its training forwards (batch 2) while its "
    "validation and evaluation forwards carry the correct pairing; an arm that "
    "raised at batch 1 could not be scored at all. Latitude/longitude are "
    "batch-invariant by construction (a batch whose samples disagree about the "
    "grid is rejected), so no within-batch roll is possible or invented for them."
)

FLOP_CONVENTION = (
    "FlopCounterMode over one forward pass under torch.enable_grad(); counts "
    "linear/conv/matmul and aten-dispatched attention matmuls; elementwise and "
    "normalization ops are not counted; no parameter hooks (SDPA is parameterless "
    "and a parameter hook would silently undercount attention). Backward is "
    "measured separately, never assumed to be 2x forward."
)

SEGMENT_NOTE = ("M2 two-month segment, 2016-01-01..2016-02-29, 240 consecutive "
                "6-hourly steps of real ERA5, 17 channels on a 65x65 grid; train "
                "[01-01, 02-17), val [02-17, 02-23); the test block is never read "
                "in this round")
LIMITATIONS = [
    "one bounded four-arm run at 400 updates, not a convergence or SOTA comparison",
    "three seeds: sign agreement across three seeds is consistency, not significance, "
    "and no significance threshold is introduced or relaxed",
    "A differs from B/E/P in parameters and FLOPs by construction, so B-A, E-A and "
    "P-A remain capacity-confounded in the sense that the module is absent from A; "
    "B-E is the pair that holds parameters, module and measured FLOPs fixed and "
    "varies only what the conditioning module is shown",
    "the control arms measure capacity and field pairing only: neither says "
    "anything about position dependence, which round two already answered",
    "P's mismatch acts on its training forwards (batch 2); the validation and "
    "published-evaluation paths score one window per forward pass, where a "
    "within-batch roll has no partner and is the identity, so P is trained with a "
    "broken field/sample correspondence and scored with the correct one",
    "the primary is registered for t2m at 6/12/24/48/72 h over the three declared "
    "seeds; every other variable and the aggregate counts are reported as "
    "secondary and are not a substitute for it",
    "one winter segment of one year in one region: no seasonal, cross-year or "
    "cross-region conclusion is testable",
    "validation-split only; the test split stays sealed in this round and was not "
    "opened for method selection at any point",
    "the val climatology is the store's 8-bucket (month, hour) train-only mean, which "
    "is not a strong seasonal climatology",
    "the round-two numbers are a different protocol digest and a different model "
    "code digest; they are cited as a stability observation only and never pooled",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arm_config(kind, channels, config):
    return {"in_channels": channels, "out_channels": channels, "history_steps": 2, **config}


def count_flops(model, batch, *, reasoning_steps):
    """Forward-only and forward+backward FLOPs, same convention as the audit."""
    from model.r7_halting import forecast_inputs
    from torch.utils.flop_counter import FlopCounterMode

    inputs = forecast_inputs(batch)
    model.zero_grad()
    with torch.enable_grad():
        with FlopCounterMode(display=False) as counter:
            model(inputs, reasoning_steps=reasoning_steps)
        forward = int(counter.get_total_flops())
    model.zero_grad()
    with torch.enable_grad():
        with FlopCounterMode(display=False) as counter:
            output = model(inputs, reasoning_steps=reasoning_steps)
            output.forecast.square().mean().backward()
        forward_backward = int(counter.get_total_flops())
    model.zero_grad()
    return forward, forward_backward


def refused_test_manifest(manifest: Path) -> None:
    """This round is validation-only; a test manifest must never reach the reader."""
    if Path(manifest).name == "test.jsonl":
        raise ValueError("this round is validation-only: the test split stays sealed")


def arm_switches(config):
    return {"spacetime_inputs": bool(config.get("spacetime_inputs", False)),
            "spacetime_field_mode": config.get("spacetime_field_mode", FIELDS)}


def protocol_payload(manifests_dir, identity, channels, measured):
    """The frozen protocol. Identical for every seed; its digest is the pin."""
    from training.r7_experiment import canonical_digest

    body = {
        "format": "r7-71-72-round-three-four-arm-protocol-v1",
        "frozen_before_any_step": True,
        "issue": "#71 M1 (round three: capacity attribution for the space-time inputs)",
        "objective": ("separate the round-two B-A effect on t2m into what the "
                      "conditioning module is shown (information) and what it can "
                      "represent (capacity/bias), with preregistered primary cells; "
                      "never a skill or superiority claim"),
        "attribution_rule": ("a difference between two arms is only attributable to "
                             "the mechanism they differ in when they are matched in "
                             "parameters and compute; B-E is matched here, A is not"),
        "data": {
            "store": str(Path(manifests_dir).parent / "cache.zarr"),
            "train_manifest": str(Path(manifests_dir) / "train.jsonl"),
            "val_manifest": str(Path(manifests_dir) / "val.jsonl"),
            "test_manifest": str(Path(manifests_dir) / "test.jsonl"),
            "test_read": False,
            "test_policy": "sealed this round; only the val split is scored",
            "data_identity": str(identity),
            "split_mode": "time_ranges (docs/decisions/0005-*.md, 0008-*.md)",
            "normalization": "store train-only centered mean/std, applied at read time",
            "segment": SEGMENT_NOTE,
        },
        "seeds": list(SEEDS),
        "seed_policy": ("declared before running; every declared seed is reported "
                        "whether or not it favours any arm"),
        "arms": [{"name": name, "kind": kind,
                  "model_config": arm_config(kind, channels, config),
                  "switches": arm_switches(config),
                  "parameters": measured[name]["parameters"],
                  "forward_flops": measured[name]["forward_flops"],
                  "forward_backward_flops": measured[name]["forward_backward_flops"]}
                 for name, kind, config in ARMS],
        "primary_registration": {
            "registered_before_first_optimizer_step": True,
            "frozen_before_any_step": True,
            "variable": PRIMARY_VARIABLE,
            "leads_hours": list(PRIMARY_LEADS),
            "pairs": [{"focus": focus, "baseline": baseline,
                       "reading": f"{focus} - {baseline}"}
                      for focus, baseline in PRIMARY_PAIRS],
            "decision_text": PRIMARY_DECISION_TEXT,
            "secondary_reporting": ("all 17 variables x 5 leads are reported for "
                                    "every pair, and the aggregate improved/"
                                    "worsened/unresolved counts are reported "
                                    "separately from the primary verdict; the "
                                    "primary is never re-picked after the run"),
            "aggregate_rule": ("the comparator's per-seed sign rule at depth "
                               f"{COMPARATOR_DEPTH}; a cell is improved/worsened "
                               "only when every declared seed agrees in sign, "
                               "otherwise it is unresolved"),
        },
        "field_mode_semantics": FIELD_MODE_NOTE,
        "arm_pairing": {
            "shared_initialization_anchor": BASELINE_ARM,
            "rule": ("all four arms are constructed from the same seed with the added "
                     "module constructed last under a rewound stream, so every tensor "
                     "two arms share starts bitwise identical; the anchor's "
                     "state_dict is loaded into each other arm before its first step "
                     "through the trainer's transfer rule, and the applied/ignored "
                     "names are recorded in the result"),
            "matched_pair": ["process_spacetime_only", "process_spacetime_constant"],
            "matched_pair_rule": ("E and B hold the same parameter tensors, the same "
                                  "seeded values and the same measured FLOPs; only "
                                  "what the module is shown differs, and both are "
                                  "checked in-run rather than asserted here"),
            "explicitly_not_matched": ["parameter count", "forward/backward FLOPs",
                                       "wall time", "peak memory"],
        },
        "comparisons": [{"focus": focus, "baseline": baseline, "depth": COMPARATOR_DEPTH}
                        for focus, baseline in PAIRS],
        "shared_controls": {
            "optimizer": f"AdamW lr {LR} weight_decay {WEIGHT_DECAY}",
            "lr_schedule": (f"linear warmup over {WARMUP_UPDATES} updates to the peak, "
                            f"then cosine decay to {MINIMUM_LR_RATIO} of peak at update "
                            f"{UPDATES}"),
            "max_updates": UPDATES,
            "batch_size": BATCH_SIZE,
            "clip": CLIP,
            "validation_every": VALIDATION_EVERY,
            "validation_lead_hours": list(VALIDATION_LEADS),
            "early_stopping": (f"stop when {EARLY_STOPPING_PATIENCE} consecutive "
                               f"validation checks fail to improve by "
                               f"{MINIMUM_IMPROVEMENT:.1%} relative; reads validation "
                               "only"),
            "checkpoint_selection_rule": ("lowest mean latitude-weighted normalized "
                                          "validation MSE; ties keep the earlier one"),
            "loss": "latitude-weighted MSE, streamed truncated BPTT",
            "reasoning_steps": REASONING_STEPS,
            "process_weight": PROCESS_WEIGHT,
            "sample_order": "torch.randperm(len(dataset), generator=manual_seed(seed+epoch))",
            "evaluation_leads_hours": list(EVALUATION_LEADS),
            "evaluation_policy": ("free autoregressive rollout on val, one run per "
                                  "(arm, seed, lead) holding the other leads fixed"),
            "evaluation_max_samples": EVALUATION_MAX_SAMPLES,
            "selection_split": "val only; test is sealed and never read",
            "comparison": ("training/r7_coreasoning_compare.summarize/compare (the "
                           "#60-fixed comparator), paired per seed at depth "
                           f"{COMPARATOR_DEPTH}"),
            "comparison_rule": ("a cell counts as improved/worsened only when every "
                               "declared seed's delta agrees in sign; disagreement is "
                               "reported as unresolved, never averaged away"),
            "wall_clock_limit_seconds": DEADLINE_SECONDS,
        },
        "flop_convention": FLOP_CONVENTION,
        "compute_alignment": {
            "claimed_aligned": ["data", "seed", "sample order", "normalization",
                                "target variable", "optimizer updates", "batch size",
                                "lr schedule", "checkpoint rule", "early-stopping rule",
                                "shared parameters at initialisation",
                                "parameters and FLOPs between B and E"],
            "explicitly_not_aligned": ["parameter count", "forward/backward FLOPs",
                                       "wall time", "peak memory"],
            "measurement": ("parameter, FLOPs, wall-time and case-count tables are all "
                            "reported; forward-only FLOPs never stand in for training "
                            "cost"),
        },
        "scientific_claim": False,
        "limitations": LIMITATIONS,
    }
    return dict(body, protocol_sha256=canonical_digest(body))


def measure_arms(channels, probe_batch):
    """Parameters and FLOPs for all four arms, measured before the protocol is frozen.

    Parameters come from ``training/r7_budget_audit.count_parameters``; forward FLOPs
    from its ``count_forward_flops`` at the same reasoning depth. Forward+backward is
    measured separately with the same counter convention rather than assumed to be 2x.
    """
    from training.r7_budget_audit import count_forward_flops, count_parameters
    from training.r7_experiment import make_model, seed_everything

    measured = {}
    for name, kind, config in ARMS:
        seed_everything(SEEDS[0])
        model = make_model(kind, arm_config(kind, channels, config))
        counted = count_forward_flops(model, probe_batch, reasoning_steps=REASONING_STEPS)
        forward, forward_backward = count_flops(model, probe_batch,
                                                reasoning_steps=REASONING_STEPS)
        if counted != forward:
            raise RuntimeError(f"{name}: the two forward FLOP counters disagree")
        measured[name] = {"parameters": int(count_parameters(model)),
                          "forward_flops": forward,
                          "forward_backward_flops": forward_backward}
        del model
    return measured


def anchor_transfer_rule(anchor, target):
    """The transfer mapping ``run_scheduled_runner`` applies, mirrored name by name."""
    applied = sorted(name for name, tensor in anchor.items()
                     if name in target and tuple(target[name].shape) == tuple(tensor.shape))
    ignored = sorted(name for name in anchor if name not in applied)
    return applied, ignored


def verify_arm_pairing(channels, seed):
    """Measured, not assumed: the four arms share one seeded initialization.

    Two checks, both real:

    - the four state dicts are built from the same seed and compared name by name, so
      "shared" means same name *and* bitwise equal value, for every one of the six
      arm pairs;
    - the anchor's tensors are actually loaded into a freshly constructed copy of each
      other arm through the trainer's transfer rule, and every applied tensor is then
      compared against the anchor bitwise. The counts recorded here are predictions of
      what the training report will contain, and ``run_seed`` fails if the report
      disagrees.

    The E/B pair is also compared tensor for tensor without the "shared" filter: the
    two arms must have the same state_dict keys and the same values, because the
    round's information contrast is only a contrast of inputs if they do.
    """
    from training.r7_experiment import make_model, seed_everything

    states = {}
    for name, kind, config in ARMS:
        seed_everything(seed)
        model = make_model(kind, arm_config(kind, channels, config))
        states[name] = {key: value.clone() for key, value in model.state_dict().items()}

    pairs = {}
    for first in ARM_NAMES:
        for second in ARM_NAMES:
            if first >= second:
                continue
            shared = sorted(set(states[first]) & set(states[second]))
            pairs[f"{first}|{second}"] = {
                "shared_tensors": len(shared),
                "shared_parameters": int(sum(states[first][key].numel() for key in shared)),
                "shared_tensors_identical": bool(
                    all(torch.equal(states[first][key], states[second][key])
                        for key in shared)),
                "added_in_second": sorted(set(states[second]) - set(states[first])),
                "added_in_first": sorted(set(states[first]) - set(states[second])),
                "same_tensor_set": set(states[first]) == set(states[second]),
                "all_tensors_identical": bool(
                    set(states[first]) == set(states[second])
                    and all(torch.equal(states[first][key], states[second][key])
                            for key in states[first])),
            }
    anchor = states[BASELINE_ARM]
    transfers = {}
    for name, kind, config in ARMS[1:]:
        seed_everything(seed)
        model = make_model(kind, arm_config(kind, channels, config))
        target = model.state_dict()
        applied, ignored = anchor_transfer_rule(anchor, target)
        if not applied:
            raise RuntimeError(f"the anchor loaded nothing into {name}")
        model.load_state_dict({key: anchor[key] for key in applied}, strict=False)
        loaded = model.state_dict()
        transfers[name] = {
            "applied_count": len(applied), "ignored_count": len(ignored),
            "applied_parameters": applied, "ignored_parameters": ignored,
            "post_load_all_applied_bitwise_equal": bool(
                all(torch.equal(loaded[key], anchor[key]) for key in applied)),
        }
        del model
    return {
        "anchor": BASELINE_ARM,
        "pairwise_shared_tensors": pairs,
        "all_shared_pairs_identical": bool(
            all(entry["shared_tensors_identical"] for entry in pairs.values())),
        "anchor_transfer": transfers,
        "tensor_counts": {name: len(state) for name, state in states.items()},
        "parameter_tensors": {name: int(sum(tensor.numel() for tensor in state.values()))
                              for name, state in states.items()},
        "rule": ("measured on same-seed state dicts; the anchor is loaded into each "
                 "other arm with the trainer's own transfer rule and every applied "
                 "tensor is compared bitwise afterwards"),
    }


def run_seed(manifests_dir, output_dir, *, seed, updates=UPDATES, device_name="cuda"):
    from data.r7_evaluation import ZarrRolloutDataset
    from data.r7_zarr_dataset import ZarrAtmosWindowDataset
    from torch.utils.data import default_collate
    from training.r7_evaluate import evaluate_local
    from training.r7_experiment import (dataset_identity, load_checkpoint, make_model,
                                        model_code_digest, seed_everything, select_device)
    from training.r7_scheduled_runner import run_scheduled_updates

    if seed not in SEEDS:
        raise ValueError(f"declared seeds are {SEEDS}; {seed} is not among them")
    manifests_dir, output_dir = Path(manifests_dir), Path(output_dir)
    train_manifest, val_manifest = manifests_dir / "train.jsonl", manifests_dir / "val.jsonl"
    for manifest in (train_manifest, val_manifest):
        refused_test_manifest(manifest)
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing existing output: {output_dir}")

    dataset = ZarrAtmosWindowDataset(train_manifest)
    identity, _ = dataset_identity(train_manifest)
    channels = int(dataset[0]["coarse_history"].shape[1])
    device = select_device(device_name)
    probe_batch = default_collate([dataset[0], dataset[1]])
    validation_dataset = ZarrRolloutDataset(manifests_dir.parent / "cache.zarr", split="val",
                                            lead_hours=VALIDATION_LEADS, history_steps=2,
                                            step_hours=6)

    measured = measure_arms(channels, probe_batch)
    protocol = protocol_payload(manifests_dir, identity, channels, measured)
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol_path = output_dir / "protocol.json"
    with protocol_path.open("x", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=False, allow_nan=False)
    # Re-read before the first optimizer step: the freeze is checked here, not
    # asserted about the run afterwards. The primary registration is part of what
    # is re-read, so "the verdict was written down before any step" is a check.
    frozen = json.loads(protocol_path.read_text(encoding="utf-8"))
    if frozen["protocol_sha256"] != protocol["protocol_sha256"]:
        raise RuntimeError("the protocol on disk is not the one this run measured")
    registration = frozen["primary_registration"]
    if (registration["variable"] != PRIMARY_VARIABLE
            or tuple(registration["leads_hours"]) != PRIMARY_LEADS
            or tuple((entry["focus"], entry["baseline"])
                     for entry in registration["pairs"]) != PRIMARY_PAIRS):
        raise RuntimeError("the primary registration on disk is not the declared one")
    if not registration["decision_text"].strip():
        raise RuntimeError("the primary registration carries no decision text")

    pairing = verify_arm_pairing(channels, seed)
    if not pairing["all_shared_pairs_identical"]:
        raise RuntimeError("the four arms do not share bitwise identical seeded weights; "
                           "the comparison would not be a comparison of the inputs")
    matched = "|".join(sorted(("process_spacetime_only", "process_spacetime_constant")))
    if not pairing["pairwise_shared_tensors"][matched]["all_tensors_identical"]:
        raise RuntimeError("E and B are not the same tensors with the same values; "
                           "B-E would not be an input contrast")
    for name, entry in pairing["anchor_transfer"].items():
        if entry["ignored_count"] or not entry["post_load_all_applied_bitwise_equal"]:
            raise RuntimeError(f"the anchor did not transfer cleanly into {name}")
    for field in ("parameters", "forward_flops", "forward_backward_flops"):
        if measured["process_spacetime_only"][field] != \
                measured["process_spacetime_constant"][field]:
            raise RuntimeError(
                f"B and E are not compute-matched on {field}: "
                f"{measured['process_spacetime_only'][field]} vs "
                f"{measured['process_spacetime_constant'][field]}")

    results = {
        "format": "r7-71-72-round-three-four-arm-result-v1",
        "scientific_claim": False, "test_read": False, "seed": seed, "seeds": list(SEEDS),
        "device": device_name,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__), "platform": platform.platform(),
        "model_code_sha256": model_code_digest(),
        "protocol_sha256": protocol["protocol_sha256"], "protocol": protocol,
        "flop_measurements": measured, "arm_pairing": pairing,
        "training": {}, "evaluation": {}, "budget": {},
    }

    started = time.perf_counter()
    anchors = {}
    for name, kind, config in ARMS:
        if time.perf_counter() - started > DEADLINE_SECONDS:
            raise RuntimeError(
                f"the {DEADLINE_SECONDS:.0f}s wall-clock limit for this seed was reached "
                f"before {name} started; stopping rather than overrunning the budget")
        resolved = arm_config(kind, channels, config)
        shared = None
        if name == BASELINE_ARM:
            seed_everything(seed)
            anchors[name] = {key: value.clone()
                             for key, value in make_model(kind, resolved).state_dict().items()}
        else:
            shared = anchors[BASELINE_ARM]
        run_dir = output_dir / "training" / name
        checkpoint, report = run_scheduled_updates(
            dataset, kind=kind, model_config=resolved, data_identity=identity,
            output_dir=run_dir, total_updates=updates, batch_size=BATCH_SIZE,
            steps=REASONING_STEPS, seed=seed, lr=LR, clip=CLIP,
            process_weight=PROCESS_WEIGHT, warmup_updates=WARMUP_UPDATES,
            minimum_lr_ratio=MINIMUM_LR_RATIO, validation_every=VALIDATION_EVERY,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            minimum_improvement=MINIMUM_IMPROVEMENT, validation_lead_hours=VALIDATION_LEADS,
            device_name=device_name, validation_dataset=validation_dataset,
            shared_initial_state=shared)
        saved = load_checkpoint(checkpoint)
        if saved["updates"] != report["selected_update"]:
            raise ValueError(f"{name} selected checkpoint/update mismatch")
        applied = report["shared_initial_state"]
        if name != BASELINE_ARM:
            predicted = pairing["anchor_transfer"][name]
            if applied["applied_count"] != predicted["applied_count"] or \
                    applied["ignored_count"] != predicted["ignored_count"]:
                raise RuntimeError(
                    f"{name}: the training report's transfer counts "
                    f"({applied['applied_count']}/{applied['ignored_count']}) differ from "
                    f"the measured ones ({predicted['applied_count']}/"
                    f"{predicted['ignored_count']})")
        results["training"][name] = {
            "protocol_sha256": protocol["protocol_sha256"],
            "spacetime_field_mode": config.get("spacetime_field_mode", FIELDS),
            "spacetime_inputs": bool(config.get("spacetime_inputs", False)),
            "updates_run": report["updates_this_run"],
            "selected_update": report["selected_update"],
            "selected_validation_mse": report["selected_validation_mse"],
            "early_stopped": report["early_stopped"], "stopped_reason": report["stopped_reason"],
            "elapsed_seconds": report["elapsed_seconds"],
            "seconds_per_update": report["seconds_per_update"],
            "validation_checks": report["validations"],
            "peak_reserved_bytes": report["peak_reserved_bytes"],
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "parameters": measured[name]["parameters"],
            "shared_initial_state": applied,
            "first_epoch_mean_loss": (report["losses"][0]["loss"] if report["losses"] else None),
            "last_epoch_mean_loss": (report["losses"][-1]["loss"] if report["losses"] else None),
        }
        print(json.dumps({"trained": name, "seed": seed,
                          "selected_update": report["selected_update"],
                          "seconds": round(report["elapsed_seconds"], 1)}), flush=True)

    results["budget"]["training_seconds_total"] = time.perf_counter() - started

    for name, kind, config in ARMS:
        for lead in EVALUATION_LEADS:
            run_dir = output_dir / "evaluation" / name / f"lead_{lead:03d}h"
            report = evaluate_local(
                val_manifest, output_dir=run_dir,
                checkpoint=(output_dir / "training" / name
                            / f"update_{results['training'][name]['selected_update']:07d}.pt"),
                lead_hours=(lead,), max_samples=EVALUATION_MAX_SAMPLES,
                device_name=device_name, reasoning_steps=REASONING_STEPS)
            if report["split"] != "val":
                raise ValueError(f"{name} was scored on {report['split']}, not val")
            results["evaluation"][f"{name}@{lead}h"] = {
                "arm": name, "lead_hours": lead, "split": report["split"],
                "n_evaluated": report["n_evaluated"],
                "n_available_windows": report["n_available_windows"],
                "channels": list(report["channels"]), "units": list(report["units"]),
                "elapsed_seconds": report["elapsed_seconds"],
                "trainable_parameters": report["trainable_parameters"],
                "climatology": {key: value for key, value in report["climatology"].items()
                                if key != "bucket_counts"},
                "bucket_counts": report["climatology"]["bucket_counts"],
                "evaluation_dir": str(run_dir), "rmse_csv": str(run_dir / "rmse.csv"),
                "skill_csv": str(run_dir / "climatology_skill.csv"),
            }
    print(json.dumps({"evaluated_seed": seed, "leads": list(EVALUATION_LEADS)}), flush=True)

    for lead in EVALUATION_LEADS:
        counts = {name: results["evaluation"][f"{name}@{lead}h"]["n_evaluated"]
                  for name in ARM_NAMES}
        if len(set(counts.values())) != 1:
            raise RuntimeError(f"arms scored on different case counts at {lead}h: {counts}")
        results.setdefault("case_counts_by_lead", {})[str(lead)] = counts[ARM_NAMES[0]]

    results["mode_probe"] = mode_probe(manifests_dir, results, device_name)
    with (output_dir / "seed_result.json").open("x", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return results


# A *numerical* tolerance, not a scientific one: a float32 matmul does not promise
# the same rounding for every row of a large M, so an input that is bitwise constant
# comes out of the conditioning MLP with row-to-row differences of about one ULP.
# The probe reports the deviations themselves as well, so this flag can be checked
# rather than trusted, and the information contrast the arms are for lives orders of
# magnitude above it.
FLOAT32_CONSTANCY_TOLERANCE = 1e-6


def spread(term: torch.Tensor) -> dict:
    """How far a [B,N,D] conditioning term is from one vector for every position."""
    position = float((term - term[:, :1, :]).abs().max())
    sample = float((term - term[:1]).abs().max())
    scale = float(term.norm(dim=-1).mean())
    return {"term_norm_mean": scale,
            "position_deviation_abs": position, "sample_deviation_abs": sample,
            "position_deviation_relative": (position / scale if scale else 0.0),
            "sample_deviation_relative": (sample / scale if scale else 0.0),
            "constant_up_to_float32_rounding": bool(
                (position / scale if scale else 0.0) <= FLOAT32_CONSTANCY_TOLERANCE)}


def mode_probe(manifests_dir, results, device_name):
    """On the *trained* checkpoints: is each arm's conditioning what it claims?

    The probe runs the trained conditioning module on two real validation windows
    at once, so a within-batch roll is active, and records how far its contribution
    is from one vector for every position and every sample. The expected pattern is
    read off the declared mode, not fitted: the constant arm has to come out
    constant, the fields and shuffled arms have to vary, and the pooled arm has no
    module at all. The deviations are reported next to the flag, because "constant"
    here can only mean constant to within float32 rounding of the matmul.
    """
    from data.r7_evaluation import ZarrRolloutDataset
    from torch.utils.data import default_collate
    from training.r7_experiment import load_checkpoint, make_model, select_device

    device = select_device(device_name)
    dataset = ZarrRolloutDataset(Path(manifests_dir).parent / "cache.zarr", split="val",
                                lead_hours=(6,), history_steps=2, step_hours=6)
    pair = {name: value.to(device) if torch.is_tensor(value) else value
            for name, value in default_collate([dataset[0], dataset[1]]).items()}
    probe = {}
    for name, kind, config in ARMS:
        saved = load_checkpoint(Path(results["training"][name]["checkpoint"]))
        model = make_model(kind, saved["contract"]["model"]).to(device).eval()
        model.load_state_dict(saved["model"], strict=True)
        module = getattr(model.backbone, "spacetime", None)
        declared = saved["contract"]["model"].get("spacetime_field_mode", FIELDS)
        entry = {"trained": True,
                 "checkpoint_sha256": results["training"][name]["checkpoint_sha256"],
                 "spacetime_inputs": bool(saved["contract"]["model"].get("spacetime_inputs",
                                                                        False)),
                 "field_mode": declared,
                 "module_present": module is not None,
                 "rule": ("the term is the module's own output for two real validation "
                          "windows in one batch (the roll of the shuffled arm is active "
                          "at batch 2); deviations say how far the [B,N,D] term is from "
                          "being one vector for every position and sample, and the "
                          f"constancy flag uses a float32 rounding tolerance of "
                          f"{FLOAT32_CONSTANCY_TOLERANCE:g} relative, not a criterion "
                          "about the science")}
        if module is None:
            entry.update({"positions": 0, "term_norm_mean": 0.0,
                          "position_deviation_abs": 0.0, "sample_deviation_abs": 0.0,
                          "position_deviation_relative": 0.0,
                          "sample_deviation_relative": 0.0,
                          "constant_up_to_float32_rounding": None,
                          "consistent_with_declared_mode": True})
        else:
            with torch.no_grad():
                base = model.backbone(pair)
                term = module(pair, history=pair["coarse_history"], token_hw=base.token_hw)
            entry.update({"positions": int(term.shape[-2]), **spread(term)})
            constant = entry["constant_up_to_float32_rounding"]
            entry["consistent_with_declared_mode"] = (
                constant if declared == "constant" else not constant)
        probe[name] = entry
        del model
    return probe


def merge_seeds(output_dir, seeds=SEEDS):
    """Merge per-seed runs, refusing an incomplete or inconsistent set."""
    loaded = {}
    for seed in seeds:
        path = Path(output_dir) / f"seed{seed}" / "seed_result.json"
        if not path.is_file():
            raise FileNotFoundError(f"declared seed {seed} has no result at {path}; an "
                                    "incomplete run is reported as incomplete")
        loaded[seed] = json.loads(path.read_text(encoding="utf-8"))
    digests = {result["protocol_sha256"] for result in loaded.values()}
    if len(digests) != 1:
        raise RuntimeError(f"seeds ran under different protocols: {sorted(digests)}")
    codes = {result["model_code_sha256"] for result in loaded.values()}
    if len(codes) != 1:
        raise RuntimeError(f"seeds ran on different model code: {sorted(codes)}")
    # Every one of the twelve (seed, arm) training records has to carry that one
    # digest, so "one protocol for twelve runs" is read off the runs themselves.
    per_run = {f"{seed}/{arm}": entry["protocol_sha256"]
               for seed, result in loaded.items() for arm, entry in result["training"].items()}
    arms_seen = {arm for result in loaded.values() for arm in result["training"]}
    if arms_seen != set(ARM_NAMES) or len(per_run) != len(seeds) * len(ARM_NAMES):
        raise RuntimeError(f"expected {len(seeds) * len(ARM_NAMES)} arm runs, found "
                           f"{len(per_run)}: {sorted(per_run)}")
    if len(set(per_run.values())) != 1:
        raise RuntimeError(f"the twelve runs do not share one protocol digest: {per_run}")
    for seed, result in loaded.items():
        if not result["arm_pairing"]["all_shared_pairs_identical"]:
            raise RuntimeError(f"seed {seed}: the arms did not share one initialization")
    first = loaded[seeds[0]]
    merged = {"format": "r7-71-72-round-three-four-arm-result-v1", "scientific_claim": False,
              "test_read": False, "seeds": list(seeds),
              "protocol_sha256": first["protocol_sha256"], "protocol": first["protocol"],
              "model_code_sha256": first["model_code_sha256"],
              "run_protocol_sha256": per_run,
              "flop_measurements": first["flop_measurements"],
              "arm_pairing": {str(seed): result["arm_pairing"]
                              for seed, result in loaded.items()},
              "segments": {str(seed): {"device": result["device"], "gpu": result["gpu"],
                                       "torch_version": result["torch_version"]}
                           for seed, result in loaded.items()},
              "training": {}, "evaluation": {}, "case_counts_by_lead": {},
              "mode_probe": {}, "budget": {"training_seconds_total": 0.0}}
    for seed, result in loaded.items():
        merged["training"][str(seed)] = result["training"]
        for key, entry in result["evaluation"].items():
            merged["evaluation"][f"{seed}/{key}"] = {**entry, "seed": seed}
        merged["case_counts_by_lead"].update(result["case_counts_by_lead"])
        merged["mode_probe"][str(seed)] = result["mode_probe"]
        merged["budget"]["training_seconds_total"] += result["budget"]["training_seconds_total"]
    if not merged["training"] or not merged["evaluation"]:
        raise RuntimeError("merged result is empty; refusing to score nothing")
    return merged


def paired_comparison(merged, output_dir):
    """The #60 comparator, paired per seed, for every declared pair."""
    from training.r7_coreasoning_compare import compare, summarize

    identity = {"dataset_identity": merged["protocol"]["data"]["data_identity"],
                "model_code_sha256": merged["model_code_sha256"],
                "declared_update_budget": UPDATES, "evaluation_split": "val"}
    records = [{"arm": entry["arm"], "seed": entry["seed"], "depth": COMPARATOR_DEPTH,
                "evaluation_dir": entry["evaluation_dir"], "identity": identity}
               for entry in merged["evaluation"].values()]
    table = summarize(records)
    blocks = {}
    for baseline in sorted({baseline for _, baseline in PAIRS}):
        for block in compare(table, baseline=baseline, depth=COMPARATOR_DEPTH):
            if (block["arm"], block["baseline"]) not in PAIRS:
                continue
            blocks[(block["arm"], block["baseline"])] = block
    missing = [pair for pair in PAIRS if pair not in blocks]
    if missing:
        raise RuntimeError(f"the comparator returned no block for {missing}")
    pairs = {}
    for (focus, baseline), block in blocks.items():
        cells = {}
        for entry in block["seed_paired"]:
            key = f"{int(entry['lead_hours'])}h|{entry['variable']}"
            cells[key] = {
                "outcome": entry["direction"], "unit": entry["unit"],
                "sign_consistent": entry["sign_consistent"],
                "seed_deltas": {str(item["seed"]): item["delta"]
                                for item in entry["seed_deltas"]},
                "direction_rule": ("the comparator labels a cell improved/worsened only "
                                   "when every declared seed's delta agrees in sign; "
                                   "otherwise it is unresolved"),
            }
        totals = {name: sum(1 for value in cells.values() if value["outcome"] == name)
                  for name in ("improved", "worsened", "unresolved")}
        per_lead = {}
        for key, value in cells.items():
            per_lead.setdefault(key.split("|")[0], {"improved": 0, "worsened": 0,
                                                   "unresolved": 0})[value["outcome"]] += 1
        pairs[f"{focus} - {baseline}"] = {
            "focus_arm": focus, "baseline_arm": baseline,
            "required": (focus, baseline) in REQUIRED_PAIRS,
            "footnote": ("negative delta = the focus arm has the lower RMSE; the "
                         f"baseline is {baseline}"),
            "totals": totals, "per_lead": per_lead, "cells": cells,
            "block": {key: block[key] for key in
                      ("arm", "baseline", "depth", "case_identity", "variables_improved",
                       "variables_worsened", "variables_compared", "variables_unresolved",
                       "variables_sign_consistent", "established_improved",
                       "established_worsened", "beats_baseline_everywhere")},
        }
    payload = {
        "format": "r7-71-72-round-three-four-arm-comparison-v1",
        "scientific_claim": False,
        "identity": identity, "pairs": pairs,
        "reporting_order": [f"{focus} - {baseline}" for focus, baseline in PAIRS],
        "required": [f"{focus} - {baseline}" for focus, baseline in REQUIRED_PAIRS],
        "primary": primary_reading(pairs),
        "rule": ("seed-paired: a cell is improved/worsened only when every declared "
                 "seed's delta agrees in sign; disagreements are unresolved and are "
                 "never averaged into a win"),
        "table": table,
    }
    with (Path(output_dir) / "paired_comparison.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    write_tables(merged, output_dir)
    return payload


def primary_reading(pairs):
    """The preregistered primary cell, read by the registered decision text.

    This reads the *pair* blocks the comparator produced; it introduces no
    threshold of its own beyond the fractions frozen in ``PRIMARY_DECISION_TEXT``
    and it reports the raw numbers next to every verdict so the arithmetic can be
    checked by hand.

    The three pairs the reading is written on are B-A (the effect being
    attributed), B-E (the information part) and E-A (the capacity/bias part); the
    frozen text reads a cell only where the comparator's own per-seed sign rule
    makes it sign-consistent, and reports P-A *next to* the cell rather than as a
    precondition. ``share_information_of_B_A`` is 1 - ``share_E_of_B_A`` by
    construction (E-A + B-E = B-A), which is why the text's two thresholds on those
    two shares describe one partition of the interval.
    """
    def cell(pair, lead, variable=PRIMARY_VARIABLE):
        entry = pairs.get(pair, {}).get("cells", {}).get(f"{lead}h|{variable}")
        return entry

    def seed_mean(entry):
        deltas = list(entry["seed_deltas"].values())
        return sum(deltas) / len(deltas) if deltas else None

    leads = {}
    verdicts = []
    for lead in PRIMARY_LEADS:
        b_a = cell("process_spacetime_only - process_pooled", lead)
        b_e = cell("process_spacetime_only - process_spacetime_constant", lead)
        e_a = cell("process_spacetime_constant - process_pooled", lead)
        p_a = cell("process_spacetime_shuffled - process_pooled", lead)
        record = {"delta_B_minus_A": seed_mean(b_a) if b_a else None,
                  "delta_B_minus_E": seed_mean(b_e) if b_e else None,
                  "delta_E_minus_A": seed_mean(e_a) if e_a else None,
                  "delta_P_minus_A": seed_mean(p_a) if p_a else None,
                  "sign_consistent": {name: (entry["sign_consistent"] if entry else None)
                                      for name, entry in (("B-A", b_a), ("B-E", b_e),
                                                          ("E-A", e_a), ("P-A", p_a))}}
        base = record["delta_B_minus_A"]
        core = (b_a, b_e, e_a)
        if base in (None, 0.0) or not all(entry is not None and entry["sign_consistent"]
                                          for entry in core):
            record["verdict"] = "unresolved"
            record["reading"] = ("unresolved: the effect (B-A) or one of the two parts "
                                 "(B-E, E-A) is not sign-consistent across seeds, so the "
                                 "frozen rule writes no attribution for this lead")
        else:
            share_e = record["delta_E_minus_A"] / base
            share_i = record["delta_B_minus_E"] / base
            record.update({"share_E_of_B_A": share_e, "share_information_of_B_A": share_i})
            if share_e < 0:
                record["verdict"] = "capacity/bias does not reproduce the gain"
                record["reading"] = ("capacity/bias does not reproduce the gain at this "
                                     "lead: E-A has the opposite sign to B-A")
            elif share_e >= 0.5:
                record["verdict"] = "capacity/bias dominated"
                record["reading"] = ("capacity/bias dominated: E-A reproduces at least "
                                     "half of B-A")
            elif share_e <= 0.25:
                record["verdict"] = "information dominated"
                record["reading"] = ("information dominated: E-A reproduces at most a "
                                     "quarter of B-A and B-E carries at least three "
                                     "quarters of it")
            else:
                record["verdict"] = "mixed"
                record["reading"] = ("mixed: E-A reproduces between a quarter and a half "
                                     "of B-A; attribution unresolved between the two")
            verdicts.append(record["verdict"])
        if p_a is not None and p_a["sign_consistent"]:
            share_p = record["delta_P_minus_A"] / base if base not in (None, 0.0) else None
            record.update({"P_sign_consistent": True, "share_P_of_B_A": share_p})
            if share_p is not None and share_p >= 0.5:
                record["P_note"] = ("real-valued but mispaired fields (P) reproduce at "
                                    "least half of B-A at this lead")
            else:
                record["P_note"] = ("real-valued but mispaired fields (P) reproduce less "
                                    "than half of B-A at this lead")
        else:
            record.update({"P_sign_consistent": False, "share_P_of_B_A": None,
                           "P_note": ("P-A is not sign-consistent at this lead: the "
                                      "shuffled arm's difference from A changes sign "
                                      "across seeds, and no share is read off it")})
        leads[str(lead)] = record
    counts = {verdict: verdicts.count(verdict) for verdict in sorted(set(verdicts))}
    if not verdicts:
        headline = "no sign-consistent primary cell"
    elif len(counts) == 1:
        headline = f"all {len(verdicts)} sign-consistent leads read the same way: {next(iter(counts))}"
    else:
        headline = ("the sign-consistent leads disagree: " + ", ".join(
            f"{verdict} x{count}" for verdict, count in counts.items())
            + " - the per-lead table is the result")
    return {"variable": PRIMARY_VARIABLE, "leads_hours": list(PRIMARY_LEADS),
            "decision_text": PRIMARY_DECISION_TEXT,
            "per_lead": leads, "verdict_counts": counts, "headline": headline,
            "note": ("read from the comparator's seed-paired deltas; the aggregate "
                     "counts of the full 17-variable table are reported separately and "
                     "are not the primary")}


def write_tables(merged, output_dir):
    output_dir = Path(output_dir)
    with (output_dir / "arm_table.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "parameters", "forward_flops", "forward_backward_flops",
                         "spacetime_inputs", "spacetime_field_mode", "gpu_seconds_total"])
        for entry in merged["protocol"]["arms"]:
            seconds = sum(per_arm[entry["name"]]["elapsed_seconds"]
                          for per_arm in merged["training"].values())
            writer.writerow([entry["name"], entry["parameters"], entry["forward_flops"],
                             entry["forward_backward_flops"],
                             entry["switches"]["spacetime_inputs"],
                             entry["switches"]["spacetime_field_mode"],
                             f"{seconds:.1f}"])
    with (output_dir / "training_table.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "arm", "protocol_sha256", "spacetime_field_mode",
                         "selected_update", "updates_run", "early_stopped",
                         "seconds_per_update", "elapsed_seconds", "peak_reserved_mib",
                         "shared_applied", "shared_ignored"])
        for seed in sorted(merged["training"], key=int):
            for arm, entry in sorted(merged["training"][seed].items()):
                shared = entry["shared_initial_state"]
                writer.writerow([seed, arm, entry["protocol_sha256"],
                                 entry["spacetime_field_mode"], entry["selected_update"],
                                 entry["updates_run"], entry["early_stopped"],
                                 f"{entry['seconds_per_update']:.6f}",
                                 f"{entry['elapsed_seconds']:.2f}",
                                 f"{(entry['peak_reserved_bytes'] or 0) / 2**20:.1f}",
                                 shared["applied_count"], shared["ignored_count"]])
    with (output_dir / "rmse_table.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "arm", "lead_hours", "variable", "unit", "rmse",
                         "rmse_climatology", "mse_skill", "n_initializations"])
        for key, entry in sorted(merged["evaluation"].items(),
                                 key=lambda item: (item[1]["seed"], item[1]["arm"],
                                                   item[1]["lead_hours"])):
            rmse = {row["variable"]: row for row in _csv_rows(entry["rmse_csv"])}
            skill = {row["variable"]: row for row in _csv_rows(entry["skill_csv"])}
            for variable, row in rmse.items():
                skill_row = skill.get(variable, {})
                writer.writerow([entry["seed"], entry["arm"], entry["lead_hours"], variable,
                                 row["unit"], row["rmse"],
                                 skill_row.get("rmse_climatology", ""),
                                 skill_row.get("mse_skill", "") or "undefined",
                                 row.get("n_initializations", entry["n_evaluated"])])
    with (output_dir / "case_table.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "arm", "lead_hours", "split", "n_available_windows",
                         "n_evaluated", "test_read"])
        for key, entry in sorted(merged["evaluation"].items(),
                                 key=lambda item: (item[1]["seed"], item[1]["arm"],
                                                   item[1]["lead_hours"])):
            writer.writerow([entry["seed"], entry["arm"], entry["lead_hours"], entry["split"],
                             entry["n_available_windows"], entry["n_evaluated"], "False"])


def _csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_rmse(merged):
    """Seed-mean physical RMSE per (arm, lead, variable), with the climatology column."""
    import statistics

    grouped = {}
    for entry in merged["evaluation"].values():
        rmse = {row["variable"]: row for row in _csv_rows(entry["rmse_csv"])}
        skill = {row["variable"]: row for row in _csv_rows(entry["skill_csv"])}
        for variable, row in rmse.items():
            key = (entry["arm"], entry["lead_hours"], variable, row["unit"])
            bucket = grouped.setdefault(key, {"rmse": [], "climatology": []})
            bucket["rmse"].append(float(row["rmse"]))
            if variable in skill:
                bucket["climatology"].append(float(skill[variable]["rmse_climatology"]))
    summary = {}
    for (arm, lead, variable, unit), bucket in grouped.items():
        summary[f"{arm}|{lead}h|{variable}"] = {
            "arm": arm, "lead_hours": lead, "variable": variable, "unit": unit,
            "rmse_seed_mean": statistics.fmean(bucket["rmse"]), "n_seeds": len(bucket["rmse"]),
            "rmse_climatology": (statistics.fmean(bucket["climatology"])
                                 if bucket["climatology"] else None)}
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Bounded four-arm round-three capacity attribution for #71 M1.")
    parser.add_argument("--mode", required=True, choices=("seed", "finalize"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--manifests", default="outputs/r7_m2_segment/store/manifests")
    parser.add_argument("--out", default="outputs/r7_71_72_round_three")
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-index", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "seed":
        if args.seed is None or args.seed not in SEEDS:
            parser.error(f"--mode seed requires --seed in {SEEDS}")
        device = args.device
        if args.device_index is not None and args.device.startswith("cuda"):
            torch.cuda.set_device(args.device_index)
            device = f"cuda:{args.device_index}"
        run_seed(Path(args.manifests), Path(args.out) / f"seed{args.seed}", seed=args.seed,
                 updates=args.updates, device_name=device)
        return

    output_dir = Path(args.out)
    merged = merge_seeds(output_dir)
    with (output_dir / "merged_result.json").open("x", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2, ensure_ascii=False, allow_nan=False)
    comparison = paired_comparison(merged, output_dir)
    summary = summarize_rmse(merged)
    with (output_dir / "rmse_seed_mean.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    arms = merged["protocol"]["arms"]
    probe = {seed: {arm: {key: value for key, value in entry.items()
                          if key in ("spacetime_inputs", "field_mode", "module_present",
                                     "positions", "position_deviation_relative",
                                     "sample_deviation_relative",
                                     "constant_up_to_float32_rounding",
                                     "consistent_with_declared_mode")}
                    for arm, entry in per_arm.items()}
             for seed, per_arm in merged["mode_probe"].items()}
    print(json.dumps({
        "complete": True, "seeds": merged["seeds"],
        "protocol_sha256": merged["protocol_sha256"],
        "model_code_sha256": merged["model_code_sha256"],
        "gpu_hours": merged["budget"]["training_seconds_total"] / 3600.0,
        "parameters": {entry["name"]: entry["parameters"] for entry in arms},
        "forward_flops": {entry["name"]: entry["forward_flops"] for entry in arms},
        "paired_totals": {key: value["totals"] for key, value in comparison["pairs"].items()},
        "paired_per_lead": {key: value["per_lead"] for key, value in comparison["pairs"].items()},
        "primary": comparison["primary"],
        "mode_probe": probe,
        "scientific_claim": False,
    }, indent=1))


if __name__ == "__main__":
    main()
