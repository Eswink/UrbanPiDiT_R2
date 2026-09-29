"""#72 M2-B (RW-B): bounded four-arm comparison of the local gated solver state.

RW-B adds a per-patch working state ``Z`` updated by a local gated cell, an absolute
proposal anchored on the known state ``X_t`` instead of a tendency accumulated on the
draft, per-position gates that mix the two, and -- a weaker candidate from the same
contract (``docs/R7_MAIN_MODEL_V2_DESIGN.md`` section 3) -- learnable source-role
markers on the concatenated recurrent key. The contract asks a round to either support
the role markers or keep the on/off contrast as a negative result, so this round pays
for that contrast rather than bundling it in and leaving a negative unreadable.

Four arms, one protocol, all trained from scratch, all with the space-time conditioning
pathway *on* so the read/solve structure is the only factor that varies: the pooled
reference read (``process_spacetime_only``), the additive per-position read of round
two's C arm (``process_spacetime_rwa``, RW-A), that read plus the gated state
(``process_local_solver``, RW-B), and RW-B plus the role markers
(``process_local_solver_roles``). The registered primary is RW-B - RW-A on t2m at
6/12/24/48/72 h; the role contrast is reported next to it and never merged into it, and
both are reported next to the parameter and FLOP counts, because these arms are *not*
capacity-matched and no attribution is claimed from a pair that is not. Held fixed: the
M2 segment (train and val manifests only; the test manifest is never opened), and the
optimizer, schedule, budget, batch size, clip, validation frequency, checkpoint
selection and early stopping of the round-two/three protocol family.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.r7_arm_harness import (comparator_blocks, measure_arms, merge_seed_results,
                                   pair_cells, sha256_file, verify_arm_pairing,
                                   write_study_tables)
from training.r7_solver_probes import write_depth_probe_table

SEEDS = (41, 42)
UPDATES = 400
BATCH_SIZE = 2
LR = 2e-4
WEIGHT_DECAY = 1e-4
CLIP = 1.0
WARMUP_UPDATES = 80
MINIMUM_LR_RATIO = 0.1
VALIDATION_EVERY = 100
EARLY_STOPPING_PATIENCE = 4
MINIMUM_IMPROVEMENT = 0.001
VALIDATION_LEADS = (6,)
EVALUATION_LEADS = (6, 12, 24, 48, 72)
REASONING_STEPS = 3
PROCESS_WEIGHT = 0.0
EVALUATION_MAX_SAMPLES = 64
COMPARATOR_DEPTH = 0
DEPTH_PROBE_STEPS = (1, 2, 4)
DEPTH_PROBE_LEADS = (6, 24)
CORRECTION_PROBE_SAMPLES = 8
DEADLINE_SECONDS = 1800.0

BASE_ARM_CONFIG = {"architecture": "window", "dim": 192, "depth": 4, "heads": 4,
                   "window_size": 4, "patch_size": 2, "dropout": 0.0,
                   "anchored_processes": 8, "free_processes": 8,
                   "use_forecast_feedback": True, "spacetime_inputs": True,
                   "default_reasoning_steps": REASONING_STEPS}
ARMS = (
    ("process_spacetime_only", "process", dict(BASE_ARM_CONFIG)),
    ("process_spacetime_rwa", "process",
     dict(BASE_ARM_CONFIG, positional_process_readout=True)),
    ("process_local_solver", "process",
     dict(BASE_ARM_CONFIG, positional_process_readout=True, local_solver_state=True)),
    ("process_local_solver_roles", "process",
     dict(BASE_ARM_CONFIG, positional_process_readout=True, local_solver_state=True,
          source_role_markers=True)),
)
ARM_NAMES = tuple(entry[0] for entry in ARMS)
SWITCH_KEYS = (("spacetime_inputs", False), ("spacetime_field_mode", "fields"),
               ("positional_process_readout", False),
               ("local_solver_state", False), ("source_role_markers", False))
BASELINE_ARM = "process_spacetime_only"
RW_A_ARM = "process_spacetime_rwa"
RW_B_ARM = "process_local_solver"
RW_B_ROLES_ARM = "process_local_solver_roles"
PAIRS = ((RW_B_ARM, RW_A_ARM), (RW_B_ROLES_ARM, RW_B_ARM), (RW_A_ARM, BASELINE_ARM),
         (RW_B_ARM, BASELINE_ARM), (RW_B_ROLES_ARM, BASELINE_ARM))
PRIMARY_PAIR = (RW_B_ARM, RW_A_ARM)
PRIMARY_VARIABLE = "t2m"
PRIMARY_LEADS = (6, 12, 24, 48, 72)
PRIMARY_DECISION_TEXT = (
    "Read the seed-paired mean delta of t2m RMSE per lead for 'process_local_solver - "
    "process_spacetime_rwa', only where the comparator's per-seed sign rule makes the "
    "cell sign-consistent (disagreement is unresolved and is never averaged into a "
    "verdict). A negative delta means RW-B has the lower RMSE. A lead reads 'supported' "
    "when every declared seed agrees and the delta is negative, 'worsened' when every "
    "seed agrees and the delta is positive, and 'unresolved' otherwise. The headline is "
    "the modal reading over the sign-consistent leads with the full per-lead table "
    "beside it; if the leads disagree, that disagreement is the result. Degradations "
    "written down before the run, which do not by themselves falsify the mechanism: "
    "(a) one or more leads unresolved or worsened while t2m 6 h and 12 h are supported; "
    "(b) the aggregate improved/worsened counts over all 17 variables disagreeing with "
    "the t2m reading; (c) no lead supported while no lead is sign-consistently worsened. "
    "What falsifies it: t2m 6 h and 12 h both worsened, or every lead unresolved. The "
    "role-marker contrast is read the same way on 'process_local_solver_roles - "
    "process_local_solver' and reported as a second cell, never merged into the first. "
    "The variable and the leads above are fixed by this text and are never re-picked "
    "after the numbers exist; no threshold beyond the comparator's own sign rule is "
    "introduced and no existing threshold is relaxed."
)
SEGMENT_NOTE = ("M2 two-month segment, 2016-01-01..2016-02-29, 240 consecutive 6-hourly "
                "steps of real ERA5, 17 channels on a 65x65 grid; train [01-01, 02-17), "
                "val [02-17, 02-23); the test block is never read")
FLOP_CONVENTION = ("FlopCounterMode over one forward pass under torch.enable_grad(): "
                   "linear/conv/matmul and aten-dispatched attention matmuls are counted, "
                   "elementwise and normalization ops are not, and no parameter hooks are "
                   "used (SDPA is parameterless and a hook would undercount attention). "
                   "Backward is measured, never assumed to be 2x forward.")
LIMITATIONS = [
    "one bounded four-arm run at 400 updates, not a convergence or SOTA comparison",
    "two seeds: sign agreement across two seeds is consistency, not significance, and no "
    "significance threshold is introduced or relaxed; two seeds are weaker than the three "
    "the previous rounds used",
    "the arms are NOT capacity-matched: RW-A and RW-B each add parameters and FLOPs, so "
    "every pair here is capacity-confounded and only the measured counts say by how much",
    "the K=1/2/4 depth probe runs on a checkpoint trained at K=3: not an independently "
    "trained K=1/2/4 model",
    "the correction probe reports error/update geometry on validation windows only, and "
    "its retrospective damping uses future truth: a diagnostic, never a deployable rule",
    "one winter segment of one year in one region: no seasonal, cross-year or cross-region "
    "conclusion is testable",
    "the space-time conditioning pathway is on in every arm, so this round says nothing "
    "about the pathway itself; round three answered that under its own digest",
    "the role markers add no measured FLOPs (an additive constant on each half of the key), "
    "so that contrast is a parameter contrast, not a compute one",
    "validation-split only; the test split stays sealed and was never opened for selection",
    "the val climatology is the store's 8-bucket (month, hour) train-only mean, not a "
    "strong seasonal climatology",
    "the round-two and round-three numbers are different protocol digests and different "
    "model code digests: a stability observation only, never pooled with this round's",
]


def refused_test_manifest(manifest: Path) -> None:
    """This round is validation-only; a test manifest must never reach the reader."""
    if Path(manifest).name == "test.jsonl":
        raise ValueError("this round is validation-only: the test split stays sealed")


def protocol_payload(manifests_dir, identity, channels, measured):
    """The frozen protocol. Identical for every seed; its digest is the pin."""
    from training.r7_experiment import canonical_digest
    from training.r7_arm_harness import arm_config

    body = {
        "format": "r7-72-rw-b-four-arm-protocol-v1",
        "frozen_before_any_step": True,
        "issue": "#72 M2-B (RW-B: local gated solver state, anchored proposal, "
                 "per-position gate, source-role markers)",
        "objective": ("measure whether the local gated solver state improves on the "
                      "additive per-position read at the same inputs and loss, with the "
                      "role-marker candidate separated rather than bundled"),
        "data": {
            "store": str(Path(manifests_dir).parent / "cache.zarr"),
            "train_manifest": str(Path(manifests_dir) / "train.jsonl"),
            "val_manifest": str(Path(manifests_dir) / "val.jsonl"),
            "test_manifest": str(Path(manifests_dir) / "test.jsonl"),
            "test_read": False, "test_policy": "sealed; only the val split is scored",
            "data_identity": str(identity),
            "split_mode": "time_ranges (docs/decisions/0005-*.md, 0008-*.md)",
            "normalization": "store train-only centered mean/std, applied at read time",
            "segment": SEGMENT_NOTE},
        "seeds": list(SEEDS),
        "seed_policy": ("declared before running; every declared seed is reported "
                        "whether or not it favours any arm"),
        "arms": [{"name": name, "kind": kind,
                  "model_config": arm_config(kind, channels, config),
                  "switches": {key: config.get(key, default) for key, default in SWITCH_KEYS},
                  "parameters": measured[name]["parameters"],
                  "forward_flops": measured[name]["forward_flops"],
                  "forward_backward_flops": measured[name]["forward_backward_flops"]}
                 for name, kind, config in ARMS],
        "primary_registration": {
            "registered_before_first_optimizer_step": True,
            "frozen_before_any_step": True,
            "variable": PRIMARY_VARIABLE,
            "leads_hours": list(PRIMARY_LEADS),
            "pairs": [{"focus": PRIMARY_PAIR[0], "baseline": PRIMARY_PAIR[1],
                       "reading": f"{PRIMARY_PAIR[0]} - {PRIMARY_PAIR[1]}"}],
            "decision_text": PRIMARY_DECISION_TEXT,
            "secondary_reporting": ("all 17 variables x 5 leads are reported for every "
                                    "pair, and the aggregate improved/worsened/unresolved "
                                    "counts are reported separately from the primary "
                                    "verdict; the primary is never re-picked after the "
                                    "run"),
            "aggregate_rule": ("the comparator's per-seed sign rule at depth "
                               f"{COMPARATOR_DEPTH}; a cell is improved/worsened only "
                               "when every declared seed agrees in sign, otherwise it "
                               "is unresolved"),
        },
        "arm_pairing": {
            "shared_initialization_anchor": BASELINE_ARM,
            "rule": ("every arm is constructed from the same seed with the added modules "
                     "built last under a rewound stream, so every tensor two arms share "
                     "starts bitwise identical; the reference arm's state_dict is loaded "
                     "into each other arm before its first step through the trainer's "
                     "transfer rule, and the applied/ignored names are recorded"),
            "explicitly_not_matched": ["parameter count", "forward/backward FLOPs",
                                       "wall time", "peak memory"]},
        "comparisons": [{"focus": focus, "baseline": baseline, "depth": COMPARATOR_DEPTH}
                        for focus, baseline in PAIRS],
        "shared_controls": {
            "optimizer": f"AdamW lr {LR} weight_decay {WEIGHT_DECAY} (fixed in the runner)",
            "lr_schedule": (f"linear warmup over {WARMUP_UPDATES} updates to the peak, then "
                            f"cosine decay to {MINIMUM_LR_RATIO} of peak at {UPDATES}"),
            "max_updates": UPDATES,
            "batch_size": BATCH_SIZE,
            "clip": CLIP,
            "validation_every": VALIDATION_EVERY,
            "validation_lead_hours": list(VALIDATION_LEADS),
            "early_stopping": (f"stop when {EARLY_STOPPING_PATIENCE} consecutive validation "
                               f"checks fail to improve by {MINIMUM_IMPROVEMENT:.1%} "
                               "relative; reads validation only"),
            "checkpoint_selection_rule": ("lowest mean latitude-weighted normalized "
                                          "validation MSE; ties keep the earlier one"),

            "loss": "latitude-weighted MSE, streamed truncated BPTT",
            "reasoning_steps": REASONING_STEPS,
            "process_weight": PROCESS_WEIGHT,
            "sample_order": "torch.randperm(len(dataset), generator=manual_seed(seed+epoch))",
            "evaluation_leads_hours": list(EVALUATION_LEADS),
            "evaluation_policy": ("free autoregressive rollout on val, one run per (arm, "
                                  "seed, lead) with the other leads held fixed"),
            "evaluation_max_samples": EVALUATION_MAX_SAMPLES,
            "selection_split": "val only; test is sealed and never read",
            "comparison": ("training/r7_coreasoning_compare.summarize/compare (the "
                           f"#60-fixed comparator), paired per seed at depth {COMPARATOR_DEPTH}"),
            "comparison_rule": ("a cell counts as improved/worsened only when every "
                               "declared seed's delta agrees in sign; disagreement is "
                               "unresolved and never averaged away"),
            "depth_probe": (f"K in {list(DEPTH_PROBE_STEPS)} on the K={REASONING_STEPS} "
                            f"checkpoint of {RW_A_ARM} and {RW_B_ARM} at leads "
                            f"{list(DEPTH_PROBE_LEADS)} h"),
            "correction_probe": (f"per-step error/update geometry over {CORRECTION_PROBE_SAMPLES} "
                                 "validation windows, "
                                 "training/r7_correction_diagnostic.collect_correction_terms"),
            "wall_clock_limit_seconds": DEADLINE_SECONDS,
        },
        "cost_reporting": {
            "tables": ["arm_table.csv: parameters, forward and forward+backward FLOPs",
                       "training_table.csv: wall clock and seconds per update",
                       "memory_table.csv: peak allocated and reserved bytes",
                       "rmse_table.csv: physical RMSE and climatology skill, all 17 "
                       "variables x 5 leads x 4 arms x every seed",
                       "case_table.csv: cases scored per arm and lead",
                       "depth_probe_table.csv: the K=1/2/4 probe rows"],
            "note": ("forward-only FLOPs never stand in for training cost; the four cost "
                     "views are reported together"),
        },
        "flop_convention": FLOP_CONVENTION,
        "scientific_claim": False,
        "limitations": LIMITATIONS,
    }
    return dict(body, protocol_sha256=canonical_digest(body))


def verified_registration(protocol_path):
    """Re-read the protocol from disk and check the frozen registration matches.

    This runs *before the first optimizer step*, so "the verdict was written down
    before any step" is a check rather than a claim about the run afterwards. A
    protocol whose digest does not survive the JSON round trip, or whose primary
    is not the declared one, stops the run.
    """
    frozen = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    from training.r7_experiment import canonical_digest

    if frozen["protocol_sha256"] != canonical_digest(
            {key: value for key, value in frozen.items() if key != "protocol_sha256"}):
        raise RuntimeError("the protocol on disk is not the one this run measured")
    registration = frozen["primary_registration"]
    if (registration["variable"] != PRIMARY_VARIABLE
            or tuple(registration["leads_hours"]) != PRIMARY_LEADS
            or tuple((entry["focus"], entry["baseline"])
                     for entry in registration["pairs"]) != (PRIMARY_PAIR,)):
        raise RuntimeError("the primary registration on disk is not the declared one")
    if not registration["decision_text"].strip():
        raise RuntimeError("the primary registration carries no decision text")
    return frozen


def run_seed(manifests_dir, output_dir, *, seed, updates=UPDATES, device_name="cuda"):
    from data.r7_evaluation import ZarrRolloutDataset
    from data.r7_zarr_dataset import ZarrAtmosWindowDataset
    from torch.utils.data import default_collate
    from training.r7_arm_harness import arm_config
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

    measured = measure_arms(ARMS, channels=channels, probe_batch=probe_batch,
                            reasoning_steps=REASONING_STEPS, seed=seed)
    protocol = protocol_payload(manifests_dir, identity, channels, measured)
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol_path = output_dir / "protocol.json"
    with protocol_path.open("x", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=False, allow_nan=False)
    # Re-read before the first optimizer step: the freeze is checked here, not
    # asserted about the run afterwards.
    verified_registration(protocol_path)

    pairing = verify_arm_pairing(ARMS, baseline=BASELINE_ARM, channels=channels, seed=seed)
    if not pairing["all_shared_pairs_identical"]:
        raise RuntimeError("the four arms do not share bitwise identical seeded weights; "
                           "every comparison would be confounded by the initialization")
    for name, entry in pairing["anchor_transfer"].items():
        if entry["ignored_count"] or not entry["post_load_all_applied_bitwise_equal"]:
            raise RuntimeError(f"the reference arm did not transfer cleanly into {name}")
    for focus, baseline in ((RW_A_ARM, BASELINE_ARM), (RW_B_ARM, RW_A_ARM),
                            (RW_B_ROLES_ARM, RW_B_ARM)):
        shared = pairing["pairwise_shared_tensors"]["|".join(sorted((focus, baseline)))]
        if not (shared["added_in_first"] or shared["added_in_second"]):
            raise RuntimeError(f"{focus} and {baseline} have the same tensor set: the "
                               "switch between them is not doing anything and the "
                               "comparison would be between two identical models")
        if pairing["tensor_counts"][focus] <= pairing["tensor_counts"][baseline]:
            raise RuntimeError(f"{focus} does not add a tensor over {baseline} "
                               f"({pairing['tensor_counts'][focus]} vs "
                               f"{pairing['tensor_counts'][baseline]}): the switch it "
                               "carries is silently ignored")

    results = {
        "format": "r7-72-rw-b-four-arm-result-v1", "scientific_claim": False,
        "test_read": False, "seed": seed, "seeds": list(SEEDS), "device": device_name,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__), "platform": platform.platform(),
        "model_code_sha256": model_code_digest(), "protocol": protocol,
        "protocol_sha256": protocol["protocol_sha256"], "arm_pairing": pairing,
        "flop_measurements": measured,
        "training": {}, "evaluation": {}, "probes": {}, "budget": {}}

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
        checkpoint, report = run_scheduled_updates(
            dataset, kind=kind, model_config=resolved, data_identity=identity,
            output_dir=output_dir / "training" / name, total_updates=updates,
            batch_size=BATCH_SIZE, steps=REASONING_STEPS, seed=seed, lr=LR, clip=CLIP,
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
            if (applied["applied_count"] != predicted["applied_count"]
                    or applied["ignored_count"] != predicted["ignored_count"]):
                raise RuntimeError(
                    f"{name}: the training report's transfer counts "
                    f"({applied['applied_count']}/{applied['ignored_count']}) differ from "
                    f"the measured ones ({predicted['applied_count']}/"
                    f"{predicted['ignored_count']})")
        results["training"][name] = {
            "protocol_sha256": protocol["protocol_sha256"],
            "switches": {key: config.get(key, default) for key, default
                                  in SWITCH_KEYS},
            "updates_run": report["updates_this_run"],
            "selected_update": report["selected_update"],
            "selected_validation_mse": report["selected_validation_mse"],
            "early_stopped": report["early_stopped"],
            "stopped_reason": report["stopped_reason"],
            "elapsed_seconds": report["elapsed_seconds"],
            "seconds_per_update": report["seconds_per_update"],
            "validation_checks": report["validations"],
            "peak_allocated_bytes": report["peak_allocated_bytes"],
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
                "peak_allocated_bytes": (torch.cuda.max_memory_allocated(device)
                                         if device.type == "cuda" else None),
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

    from training.r7_solver_probes import solver_probes

    results["probes"] = solver_probes(
        manifests_dir, output_dir, results, arms=(RW_A_ARM, RW_B_ARM),
        steps_list=DEPTH_PROBE_STEPS, leads=DEPTH_PROBE_LEADS, device_name=device_name,
        max_samples=EVALUATION_MAX_SAMPLES, trained_steps=REASONING_STEPS,
        windows=CORRECTION_PROBE_SAMPLES)
    with (output_dir / "seed_result.json").open("x", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return results


def primary_reading(pairs):
    """The preregistered primary cell, read by the registered decision text.

    ``pairs`` is keyed by ``"focus - baseline"``. A lead the comparator never
    reported reads as unresolved, not as a missing row.
    """
    def cell(pair):
        block = pairs.get(f"{pair[0]} - {pair[1]}", {}).get("cells", {})
        return {lead: block.get(f"{lead}h|{PRIMARY_VARIABLE}") for lead in PRIMARY_LEADS}

    def seed_mean(entry):
        deltas = list(entry["seed_deltas"].values())
        return sum(deltas) / len(deltas) if deltas else None

    def read(pair):
        cells, verdicts = {}, []
        for lead, entry in cell(pair).items():
            if entry is None or not entry["sign_consistent"]:
                # No seed mean here on purpose: the rule forbids averaging a
                # disagreement, and a mean next to "unresolved" is the number people
                # would quote.
                cells[str(lead)] = {
                    "delta_seed_mean": None, "outcome": "unresolved",
                    "sign_consistent": False,
                    "seed_deltas": ({} if entry is None else dict(entry["seed_deltas"])),
                    "reading": ("the per-seed deltas disagree in sign, or the comparator "
                                "returned no cell; the frozen rule writes no verdict and "
                                "no seed mean here")}
                continue
            delta = seed_mean(entry)
            outcome = "supported" if delta < 0 else "worsened"
            cells[str(lead)] = {"delta_seed_mean": delta, "sign_consistent": True,
                                "outcome": outcome,
                                "seed_deltas": dict(entry["seed_deltas"]),
                                "reading": (f"{outcome}: every declared seed agrees and "
                                            "RW-B has the lower RMSE" if delta < 0 else
                                            f"{outcome}: every declared seed agrees and "
                                            "RW-B has the higher RMSE")}
            verdicts.append(outcome)
        counts = {name: verdicts.count(name) for name in sorted(set(verdicts))}
        if not verdicts:
            headline = "no sign-consistent primary cell"
        elif len(counts) == 1:
            headline = (f"all {len(verdicts)} sign-consistent leads read the same way: "
                        f"{next(iter(counts))}")
        else:
            headline = ("the sign-consistent leads disagree: "
                        + ", ".join(f"{name} x{count}" for name, count in counts.items())
                        + " - the per-lead table is the result")
        return {"variable": PRIMARY_VARIABLE, "leads_hours": list(PRIMARY_LEADS),
                "pair": f"{pair[0]} - {pair[1]}", "per_lead": cells,
                "verdict_counts": counts, "headline": headline}
    return {"decision_text": PRIMARY_DECISION_TEXT,
            "primary": read(PRIMARY_PAIR),
            "role_marker_cell": {"pair": f"{RW_B_ROLES_ARM} - {RW_B_ARM}",
                                 "reported_exactly_the_same_way": read((RW_B_ROLES_ARM,
                                                                        RW_B_ARM))},
            "note": ("read from the comparator's seed-paired deltas; the aggregate counts "
                     "of the full 17-variable table are reported separately and are not "
                     "the primary")}


def main():
    parser = argparse.ArgumentParser(description="Bounded four-arm RW-B round for #72.")
    parser.add_argument("--mode", required=True, choices=("seed", "finalize"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--manifests", default="outputs/r7_m2_segment/store/manifests")
    parser.add_argument("--out", default="outputs/r7_72_rw_b_pilot")
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
    merged = merge_seed_results(output_dir, seeds=SEEDS, arms=ARM_NAMES,
                                fmt="r7-72-rw-b-four-arm-result-v1")
    with (output_dir / "merged_result.json").open("x", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2, ensure_ascii=False, allow_nan=False)
    identity = {"dataset_identity": merged["protocol"]["data"]["data_identity"],
                "model_code_sha256": merged["model_code_sha256"],
                "declared_update_budget": UPDATES, "evaluation_split": "val"}
    table, blocks = comparator_blocks(merged, pairs=PAIRS, depth=COMPARATOR_DEPTH,
                                      identity=identity)
    pairs = pair_cells(blocks, leads=EVALUATION_LEADS)
    payload = {"format": "r7-72-rw-b-four-arm-comparison-v1", "scientific_claim": False,
               "identity": identity, "pairs": pairs,
               "reporting_order": [f"{focus} - {baseline}" for focus, baseline in PAIRS],
               "primary": primary_reading(pairs),
               "rule": ("seed-paired: a cell is improved/worsened only when every "
                        "declared seed's delta agrees in sign; disagreements are "
                        "unresolved and are never averaged into a win"),
               "table": table}
    with (output_dir / "paired_comparison.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    write_study_tables(merged, output_dir)
    write_depth_probe_table(merged, output_dir)
    print(json.dumps({
        "complete": True, "seeds": merged["seeds"], "scientific_claim": False,
        "protocol_sha256": merged["protocol_sha256"],
        "model_code_sha256": merged["model_code_sha256"],
        "gpu_hours_training": merged["budget"]["training_seconds_total"] / 3600.0,
        "gpu_hours_evaluation": merged["budget"]["evaluation_seconds_total"] / 3600.0,
        "parameters": {entry["name"]: entry["parameters"]
                       for entry in merged["protocol"]["arms"]},
        "forward_flops": {entry["name"]: entry["forward_flops"]
                          for entry in merged["protocol"]["arms"]},
        "paired_totals": {key: value["totals"] for key, value in pairs.items()},
        "primary": payload["primary"]}, indent=1))


if __name__ == "__main__":
    main()
