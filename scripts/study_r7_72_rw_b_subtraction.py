"""D4: the bounded leave-one-out contrast for the RW-B subtraction round.

The bounded round registered ``RW-B - RW-A`` as negative and could not attribute it
(``docs/R7_72_RW_B_PILOT.md``). This round freezes one pre-declared comparison and four
arms:

    RW-A               additive per-position read, the baseline of the comparison
    RW-B               local_solver_state=True, both sub-switches at their default
    RW-B minus (a)     solver_gate_proposal=False
    RW-B minus (b)     solver_state_recurrence=False

Same store, same protocol family, same 400-update budget, two seeds, validation only,
test sealed. The protocol is written with ``'x'`` and re-read before the first optimizer
step, and the registration is checked against the declared constants, so "the single
comparison was written down before the numbers" is a check rather than a claim about the
run afterwards.

The reading is the one the round pre-declared: for each leave-one-out arm, the existing
#60 comparator's per-seed same-sign rule on ``arm - RW-A`` at t2m x 6/12/24/48/72 h x the
two declared seeds. No threshold, endpoint or case set is introduced here.

Run ``--mode seed --seed N`` per seed (one GPU each) and ``--mode finalize`` once
afterwards; the finalize step refuses an incomplete seed set.
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
from training.r7_rw_b_subtraction_protocol import (ARMS, ARM_NAMES, BATCH_SIZE, CLIP,
                                                 COMPARATOR_DEPTH, CORRECTION_PROBE_SAMPLES,
                                                 DEADLINE_SECONDS, EARLY_STOPPING_PATIENCE,
                                                 EVALUATION_LEADS,
                                                 EVALUATION_MAX_SAMPLES, LR,
                                                 MINIMUM_IMPROVEMENT, MINIMUM_LR_RATIO,
                                                 NO_GATE_ARM, NO_RECURRENCE_ARM, PAIRS,
                                                 PRIMARY_DECISION_TEXT, PRIMARY_LEADS,
                                                 PRIMARY_VARIABLE, PROCESS_WEIGHT,
                                                 REASONING_STEPS, RW_A_ARM, RW_B_ARM,
                                                 SEEDS, SWITCH_KEYS, UPDATES,
                                                 VALIDATION_EVERY, VALIDATION_LEADS,
                                                 WARMUP_UPDATES, refused_test_manifest,
                                                 protocol_payload, verified_registration)


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

    pairing = verify_arm_pairing(ARMS, baseline=RW_A_ARM, channels=channels, seed=seed)
    if not pairing["all_shared_pairs_identical"]:
        raise RuntimeError("the four arms do not share bitwise identical seeded weights; "
                           "every comparison would be confounded by the initialization")
    for name, entry in pairing["anchor_transfer"].items():
        if entry["ignored_count"] or not entry["post_load_all_applied_bitwise_equal"]:
            raise RuntimeError(f"the reference arm did not transfer cleanly into {name}")
    for focus, baseline in PAIRS:
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
        "format": "r7-72-rw-b-subtraction-result-v1", "scientific_claim": False,
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
        if name == RW_A_ARM:
            seed_everything(seed)
            anchors[name] = {key: value.clone()
                             for key, value in make_model(kind, resolved).state_dict().items()}
        else:
            shared = anchors[RW_A_ARM]
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
        if name != RW_A_ARM:
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
            "switches": {key: config.get(key, default) for key, default in SWITCH_KEYS},
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

    from training.r7_solver_probes import correction_probe

    results["probes"]["correction"] = correction_probe(
        manifests_dir, results, arms=(RW_A_ARM, RW_B_ARM, NO_GATE_ARM, NO_RECURRENCE_ARM),
        device_name=device_name, windows=CORRECTION_PROBE_SAMPLES, max_steps=REASONING_STEPS)
    with (output_dir / "seed_result.json").open("x", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return results


def read_arm(pairs, focus, role):
    """One arm read by the registered decision text.

    ``pairs`` is keyed by ``"focus - baseline"``. A lead the comparator never reported
    reads as unresolved, not as a missing row.
    """
    block = pairs.get(f"{focus} - {RW_A_ARM}", {}).get("cells", {})
    cells, verdicts = {}, []
    for lead in PRIMARY_LEADS:
        entry = block.get(f"{lead}h|{PRIMARY_VARIABLE}")
        if entry is None or not entry["sign_consistent"]:
            cells[str(lead)] = {
                "delta_seed_mean": None, "outcome": "unresolved",
                "sign_consistent": False,
                "seed_deltas": ({} if entry is None else dict(entry["seed_deltas"])),
                "reading": ("the per-seed deltas disagree in sign, or the comparator "
                            "returned no cell; the frozen rule writes no verdict and "
                            "no seed mean here")}
            continue
        deltas = list(entry["seed_deltas"].values())
        delta = sum(deltas) / len(deltas) if deltas else None
        outcome = "supported" if (delta is not None and delta < 0) else "worsened"
        cells[str(lead)] = {"delta_seed_mean": delta, "sign_consistent": True,
                            "outcome": outcome,
                            "seed_deltas": dict(entry["seed_deltas"]),
                            "reading": (
                                f"{outcome}: every declared seed agrees and this arm "
                                "has the lower RMSE" if outcome == "supported" else
                                f"{outcome}: every declared seed agrees and this arm "
                                "has the higher RMSE")}
        verdicts.append(outcome)
    counts = {name: verdicts.count(name) for name in sorted(set(verdicts))}
    if not verdicts:
        headline = "no sign-consistent cell"
    elif len(counts) == 1:
        headline = (f"all {len(verdicts)} sign-consistent leads read the same way: "
                    f"{next(iter(counts))}")
    else:
        headline = ("the sign-consistent leads disagree: "
                    + ", ".join(f"{name} x{count}" for name, count in counts.items())
                    + " - the per-lead table is the result")
    long_leads = {str(lead): cells[str(lead)]["outcome"] for lead in (48, 72)}
    return {"variable": PRIMARY_VARIABLE, "leads_hours": list(PRIMARY_LEADS),
            "pair": f"{focus} - {RW_A_ARM}", "role": role, "per_lead": cells,
            "verdict_counts": counts, "headline": headline,
            "long_lead_outcomes": long_leads,
            "long_leads_worsened": all(value == "worsened"
                                       for value in long_leads.values())}


def primary_reading(pairs):
    """The preregistered single comparison, read by the registered branch rule.

    The rule is applied in the order the protocol froze: the negative control first,
    because a confounded protocol makes the primary uninterpretable.
    """
    negative_control = read_arm(pairs, NO_GATE_ARM, "negative control")
    primary = read_arm(pairs, NO_RECURRENCE_ARM,
                       "primary question (which piece carries the 48/72 h degradation)")
    control_settled = (negative_control["per_lead"]["48"]["sign_consistent"]
                       and negative_control["per_lead"]["72"]["sign_consistent"])
    primary_settled = (primary["per_lead"]["48"]["sign_consistent"]
                       and primary["per_lead"]["72"]["sign_consistent"])
    if not control_settled:
        branch = "control-unresolved"
    elif negative_control["long_leads_worsened"]:
        branch = "stop-confounded-control"
    elif not primary_settled:
        branch = "cannot-attribute"
    elif primary["long_leads_worsened"]:
        branch = "attribute-to-gate-proposal"
    else:
        branch = "attribute-to-recurrence"
    return {"decision_text": PRIMARY_DECISION_TEXT,
            "branch_rule": ("frozen in protocol.json before the first optimizer step; "
                            "read here in the order: negative control, then primary"),
            "branch": branch, "negative_control": negative_control,
            "primary_question": primary,
            "round_reference": read_arm(pairs, RW_B_ARM, "round reference"),
            "note": ("read from the comparator's seed-paired deltas; the aggregate counts "
                     "of the full 17-variable table are reported separately and are not "
                     "the primary")}


def finalize(output_dir, manifests_dir):
    output_dir = Path(output_dir)
    merged = merge_seed_results(output_dir, seeds=SEEDS, arms=ARM_NAMES,
                                fmt="r7-72-rw-b-subtraction-result-v1")
    with (output_dir / "merged_result.json").open("x", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2, ensure_ascii=False, allow_nan=False)
    identity = {"dataset_identity": merged["protocol"]["data"]["data_identity"],
                "model_code_sha256": merged["model_code_sha256"],
                "declared_update_budget": UPDATES, "evaluation_split": "val"}
    table, blocks = comparator_blocks(merged, pairs=PAIRS, depth=COMPARATOR_DEPTH,
                                      identity=identity)
    pairs = pair_cells(blocks, leads=EVALUATION_LEADS)
    payload = {"format": "r7-72-rw-b-subtraction-comparison-v1", "scientific_claim": False,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=("seed", "finalize"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--manifests", default="outputs/r7_m2_segment/store/manifests")
    parser.add_argument("--out", default="outputs/r7_72_rw_b_subtraction")
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
        return 0
    finalize(Path(args.out), Path(args.manifests))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
