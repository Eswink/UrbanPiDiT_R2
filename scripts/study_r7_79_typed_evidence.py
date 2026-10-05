"""#79 S2 three-arm run: typed local diagnostic evidence in the process forward.

The registered contrast is the pathway that turns four local physical fields
(850 hPa divergence, 850 hPa vorticity, 850 hPa temperature advection and the
850-500 hPa static-stability difference), extracted with the existing
differentiable operators from the known state X_t and the model's own initial
draft Y_0, into evidence at the anchored process slots before the first
reasoning step. Three arms share one seed, one trunk initialization, one
dataset and one update budget:

- ``process_v2``: the incumbent configuration, no evidence pathway (aux=0).
- ``generic_fusion``: the same four fields, same parameter count, same compute,
  but one merged projection at every patch position; the capacity control.
- ``typed_routing``: one projection per type, each reaching only its own slot;
  the candidate.

Attribution rule (frozen before the run): A -> C can only show an overall gain;
only B -> C can attribute anything to the *typed* structure, because B and C
consume identical information through identical capacity. The registered
primary reads t2m at 6 h and 12 h on the val split by the #60-fixed per-seed
sign rule for both pairs; supported requires the C - B pair supported (or C - A
not worsened while C - B is supported). The test split is never opened, aux=0
throughout, and all 17 variables x 5 leads are reported without entering the
verdict.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.r7_arm_harness import measure_arms, sha256_file, write_study_tables

SEEDS = (41, 42, 43)
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
REASONING_STEPS = 4
PROCESS_WEIGHT = 0.0
EVALUATION_MAX_SAMPLES = 64
DEADLINE_SECONDS_PER_SEED = 5400.0  # per-seed hard stop, frozen before the run
PLANNED_SECONDS_ROUND = 9000.0      # soft budget for all three seeds
HARD_CAP_SECONDS_ROUND = 18000.0    # loose hard cap for all three seeds
COMPARATOR_DEPTH = 0
PROBE_RELATIVE_TOLERANCE = 1e-3

BASE = {"architecture": "window", "dim": 192, "depth": 4, "heads": 4, "window_size": 4,
        "patch_size": 2, "dropout": 0.0, "anchored_processes": 8, "free_processes": 8,
        "use_forecast_feedback": True, "positional_process_readout": True,
        "default_reasoning_steps": REASONING_STEPS}
ARM_A = "process_v2"
ARM_B = "generic_fusion"
ARM_C = "typed_routing"
ARM_NAMES = (ARM_A, ARM_B, ARM_C)
PRIMARY_PAIR = (ARM_C, ARM_B)
SECONDARY_PAIR = (ARM_C, ARM_A)
PRIMARY_VARIABLE = "t2m"
PRIMARY_LEADS = (6, 12)
SIDECAR_DIR = ROOT / "outputs/r7_s1_seasons_2017/typed_evidence"
PRIMARY_DECISION_TEXT = (
    "Read the seed-paired mean delta of t2m validation RMSE per lead for "
    "'typed_routing - generic_fusion' AND for 'typed_routing - process_v2', only where "
    "the per-seed deltas agree in sign (disagreement is unresolved and is never averaged "
    "into a verdict). A negative delta means the focus arm has the lower RMSE. A lead "
    "reads 'supported' when every declared seed agrees and the delta is negative, "
    "'worsened' when every seed agrees and the delta is positive, and 'unresolved' "
    "otherwise. The typed mechanism is supported only when BOTH 6 h and 12 h are "
    "supported on typed_routing - generic_fusion (the capacity-matched attribution pair) "
    "and neither lead of typed_routing - process_v2 is worsened; a gain on C - A alone is "
    "an overall-pathway effect and is reported as such, never as typed attribution. "
    "Falsified if any C - B lead is worsened or both are unresolved. What was written "
    "down before the run: this pilot screens one front-path mechanism on one dev store at "
    "400 updates and cannot support a scientific claim in either direction; it decides "
    "whether typed evidence earns a place in the next confirmation instance."
)
LIMITATIONS = [
    "one bounded three-arm run at 400 updates per seed on the 2017 dev store; no "
    "convergence, significance or SOTA claim",
    "the dev store's val/test are short windows inside one year and one region; the "
    "final confirmation instance is a separate, larger acquisition with unseen years",
    "the evidence pathway reads only initialization-time inputs (X_t and the model's own "
    "initial draft); no target, future field or validation label enters it",
    "aux=0 throughout: no future/draft diagnostic loss, no PCGrad and no controller "
    "enter this round, by frozen design",
    "validation-split only; the test split was never opened by this study",
    "the standardization vectors are fitted on this store's train split; a different "
    "store requires its own sidecar",
    "three seeds are consistency evidence, not a significance test; no significance "
    "threshold is introduced",
]


def refused_test_manifest(manifest: Path) -> None:
    if Path(manifest).name == "test.jsonl":
        raise ValueError("this study is validation-only: the test split stays sealed")


def load_evidence_payload():
    """Read the published sidecar plus the store's own train-only state standards.

    Returns ``(sidecar_meta, payload)``. The payload's de-normalization vectors
    come from the store's declared ``normalization_mean``/``normalization_std``
    arrays (the same ones the reader applies), and the four field statistics
    come from the sidecar; both are train-only by construction and the sidecar's
    data identity is checked against the manifest's by the caller.
    """
    import numpy as np
    import zarr

    from data.preprocess.r7_typed_evidence_scale import load_typed_evidence_scale

    meta = load_typed_evidence_scale(SIDECAR_DIR)
    root = zarr.open_group(str(meta["store"]), mode="r")
    channels = list(root.attrs["channels"])
    if channels != list(meta["channels"]):
        raise ValueError("the sidecar channel order does not match the store")
    denorm_mean = [float(value) for value in np.asarray(root["normalization_mean"][:])]
    denorm_std = [float(value) for value in np.asarray(root["normalization_std"][:])]
    payload = {"channels": channels,
               "denorm_mean": denorm_mean, "denorm_std": denorm_std,
               "field_scale": [float(v) for v in meta["runtime_scale"]],
               "field_mean": [float(v) for v in meta["runtime_mean"]],
               "field_std": [float(v) for v in meta["runtime_std"]],
               "field_active": [bool(v) for v in meta["active_mask"]]}
    return meta, payload


def build_arms(payload):
    """The three arms: one config each, differing only in the switch and payload."""
    return (
        (ARM_A, "process", dict(BASE)),
        (ARM_B, "process", dict(BASE, typed_evidence_mode="generic_fusion",
                                typed_evidence=dict(payload))),
        (ARM_C, "process", dict(BASE, typed_evidence_mode="typed_routing",
                                typed_evidence=dict(payload))),
    )


def arm_switches(config):
    return {"typed_evidence_mode": config.get("typed_evidence_mode", "off")}


def evidence_probe(arm_models, batch):
    """The wiring metric: evidence must reach the slots and change the forecast.

    A, B and C hold bitwise identical trunk weights (same seed, pathway built
    under a rewound stream), so any forward difference is the pathway. The
    probe measures three things: that the off arm's first-step state is exactly
    its queries expansion and the evidence arms' is not; that C's per-type
    arrival holds (perturbing one field moves exactly one slot); and that the
    three forecasts are not all equal.
    """
    from model.typed_evidence_r7 import EVIDENCE_TYPES, TYPED_ROUTING

    model_a, model_b, model_c = arm_models
    with torch.no_grad():
        forecast_a = model_a(batch).forecast
        forecast_b = model_b(batch).forecast
        forecast_c = model_c(batch).forecast
        queries = model_a.process_queries.expand(forecast_a.shape[0], -1, -1)
        anchor = batch["coarse_history"][:, -1, :forecast_a.shape[1]]
        seeded_a = model_a.initial_process_state(queries, batch, anchor=anchor,
                                                 draft=forecast_a)
        seeded_c = model_c.initial_process_state(queries, batch, anchor=anchor,
                                                 draft=forecast_c)
    difference_a = (seeded_a - queries).abs().max().item()
    difference_c = (seeded_c - queries).abs().max().item()
    # per-type arrival: poison one field of the standardized maps
    router = model_c.typed_evidence
    fields = router.field_maps(anchor, batch)
    with torch.no_grad():
        base = router.token_evidence(fields)
        arrivals = []
        for index in range(EVIDENCE_TYPES):
            poisoned = fields.clone()
            poisoned[:, index] += 0.5
            moved = (router.token_evidence(poisoned) - base).abs().amax(dim=(0, 2))
            arrivals.append([float(value) for value in moved.detach()])
    off_ok = difference_a == 0.0 and difference_c > 0.0
    typed_ok = all(arrivals[index][index] > 0.0
                   and all(value == 0.0 for other, value in enumerate(arrivals[index])
                           if other != index)
                   for index in range(EVIDENCE_TYPES))
    changed = not torch.equal(forecast_a, forecast_c)
    return {"off_path_state_unchanged": bool(difference_a == 0.0),
            "evidence_state_moved": float(difference_c),
            "typed_arrival_matrix": arrivals,
            "typed_arrival_ok": bool(typed_ok),
            "forecast_changed_vs_incumbent": bool(changed),
            "within_tolerance": bool(off_ok and typed_ok and changed)}


def protocol_payload(manifests_dir, identity, channels, measured, payload,
                     sidecar_identity):
    from training.r7_experiment import canonical_digest

    arms = build_arms(payload)
    body = {
        "format": "r7-79-typed-evidence-three-arm-protocol-v1",
        "frozen_before_any_step": True,
        "issue": "#79 typed local diagnostic evidence in the process forward (S2)",
        "objective": ("measure whether adding typed local diagnostic evidence at the "
                      "anchored process slots changes validation skill relative to a "
                      "capacity-matched untyped fusion arm and to the incumbent, under "
                      "an otherwise identical model, loss (aux=0) and update budget"),
        "data": {
            "store": str(Path(manifests_dir).parent / "cache.zarr"),
            "train_manifest": str(Path(manifests_dir) / "train.jsonl"),
            "val_manifest": str(Path(manifests_dir) / "val.jsonl"),
            "test_manifest": str(Path(manifests_dir) / "test.jsonl"),
            "test_read": False, "test_policy": "sealed; only the val split is scored",
            "data_identity": str(identity),
            "split_mode": "time_ranges; four-season 2017 dev store",
            "normalization": "store train-only centered mean/std, applied at read time",
            "typed_evidence_sidecar": str(SIDECAR_DIR),
            "typed_evidence_identity": str(sidecar_identity),
            "typed_evidence_fields": ["divergence_850", "vorticity_850",
                                      "temperature_advection_850",
                                      "static_stability_850_500"],
            "typed_evidence_active": [bool(value) for value in payload["field_active"]],
            "typed_evidence_rule": ("train-only field standardization from the published "
                                    "sidecar; fields extracted with the existing "
                                    "operators from X_t and the model's own initial draft"),
        },
        "seeds": list(SEEDS),
        "arms": [{"name": name, "kind": kind, "config": config,
                  "switches": arm_switches(config),
                  "parameters": measured[name]["parameters"],
                  "forward_flops": measured[name]["forward_flops"],
                  "forward_backward_flops": measured[name]["forward_backward_flops"]}
                 for name, kind, config in arms],
        "primary_registration": {
            "registered_before_first_optimizer_step": True,
            "variable": PRIMARY_VARIABLE, "leads_hours": list(PRIMARY_LEADS),
            "pair": {"focus": PRIMARY_PAIR[0], "baseline": PRIMARY_PAIR[1],
                     "reading": f"{PRIMARY_PAIR[0]} - {PRIMARY_PAIR[1]} "
                                "(capacity-matched typed attribution)"},
            "secondary_pair": {"focus": SECONDARY_PAIR[0], "baseline": SECONDARY_PAIR[1],
                               "reading": f"{SECONDARY_PAIR[0]} - {SECONDARY_PAIR[1]} "
                                          "(overall pathway effect; never typed attribution)"},
            "decision_text": PRIMARY_DECISION_TEXT,
            "secondary_reporting": ("all 17 variables x 5 leads are reported for both "
                                    "pairs; the primary is never re-picked after the run"),
            "aggregate_rule": (f"per-seed sign agreement at depth {COMPARATOR_DEPTH}; a "
                               "cell is improved/worsened only when every declared seed "
                               "agrees in sign, otherwise unresolved"),
        },
        "shared_controls": {
            "optimizer": f"AdamW lr {LR} weight_decay {WEIGHT_DECAY}",
            "lr_schedule": (f"linear warmup over {WARMUP_UPDATES} updates then cosine "
                            f"decay to {MINIMUM_LR_RATIO} of peak at {UPDATES}"),
            "max_updates": UPDATES, "batch_size": BATCH_SIZE, "clip": CLIP,
            "validation_every": VALIDATION_EVERY,
            "validation_lead_hours": list(VALIDATION_LEADS),
            "early_stopping": (f"{EARLY_STOPPING_PATIENCE} consecutive non-improving "
                               f"checks at {MINIMUM_IMPROVEMENT:.3%} relative"),
            "loss": "latitude-weighted MSE, streamed truncated BPTT (unchanged)",
            "reasoning_steps": REASONING_STEPS, "process_weight": PROCESS_WEIGHT,
            "aux": "0; no future/draft diagnostic loss and no controller in this round",
            "evaluation_leads_hours": list(EVALUATION_LEADS),
            "evaluation_max_samples": EVALUATION_MAX_SAMPLES,
            "selection_split": "val only; test sealed",
        },
        "mechanism_prediction": {
            "text": ("typed evidence at the anchored slots should let the same parameter "
                     "count express the local process structure more directly than the "
                     "untyped fusion of the identical fields, showing as a negative "
                     "typed_routing - generic_fusion delta; the run records the per-type "
                     "arrival probe and the off-path identity probe before training so "
                     "the wiring is checked, not assumed"),
            "falsifier": ("a worsened or unresolved typed_routing - generic_fusion pair "
                          "reads as the type structure not being a useful lever at this "
                          "budget and instance; that is a clean negative, not a reason "
                          "to add more diagnostic types"),
        },
        "budgets": {
            "planned_seconds_round": PLANNED_SECONDS_ROUND,
            "hard_cap_seconds_round": HARD_CAP_SECONDS_ROUND,
            "deadline_seconds_per_seed": DEADLINE_SECONDS_PER_SEED,
            "whole_round_scope": ("all declared seeds: CPU preflight/probe, training, "
                                  "evaluation, aggregation; soft overrun is recorded and "
                                  "continues, only the hard cap truncates"),
            "frozen_before_any_step": True,
        },
        "scientific_claim": False,
        "limitations": LIMITATIONS,
    }
    return dict(body, protocol_sha256=canonical_digest(body))


def verified_registration(protocol_path):
    from training.r7_experiment import canonical_digest

    frozen = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if frozen["protocol_sha256"] != canonical_digest(
            {key: value for key, value in frozen.items() if key != "protocol_sha256"}):
        raise RuntimeError("the protocol on disk is not the one this run measured")
    registration = frozen["primary_registration"]
    if (registration["variable"] != PRIMARY_VARIABLE
            or tuple(registration["leads_hours"]) != PRIMARY_LEADS
            or (registration["pair"]["focus"], registration["pair"]["baseline"])
            != PRIMARY_PAIR
            or (registration["secondary_pair"]["focus"],
                registration["secondary_pair"]["baseline"]) != SECONDARY_PAIR
            or not registration["decision_text"].strip()):
        raise RuntimeError("the primary registration on disk is not the declared one")
    return frozen


def run_seed(manifests_dir, output_dir, *, seed, updates=UPDATES, device_name="cuda",
             deadline_seconds=DEADLINE_SECONDS_PER_SEED):
    from data.r7_evaluation import ZarrRolloutDataset
    from data.r7_zarr_dataset import ZarrAtmosWindowDataset
    from torch.utils.data import default_collate
    from training.r7_arm_harness import arm_config, verify_arm_pairing
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

    meta, payload = load_evidence_payload()
    dataset = ZarrAtmosWindowDataset(train_manifest)
    identity, _ = dataset_identity(train_manifest)
    if meta["data_identity"] != identity:
        raise ValueError("the typed-evidence sidecar was fitted on different data: "
                         f"{meta['data_identity']} != {identity}")
    channels = int(dataset[0]["coarse_history"].shape[1])
    if channels != len(payload["channels"]):
        raise ValueError(f"the sidecar must match {channels} channels")
    device = select_device(device_name)
    probe_batch = default_collate([dataset[0], dataset[1]])
    validation_dataset = ZarrRolloutDataset(manifests_dir.parent / "cache.zarr", split="val",
                                            lead_hours=VALIDATION_LEADS, history_steps=2,
                                            step_hours=6)

    arms = build_arms(payload)
    resolved_arms = tuple((name, kind, arm_config(kind, channels, config))
                          for name, kind, config in arms)
    measured = measure_arms(arms, channels=channels, probe_batch=probe_batch,
                            reasoning_steps=REASONING_STEPS, seed=seed)
    if (measured[ARM_B]["parameters"] != measured[ARM_C]["parameters"]
            or measured[ARM_B]["forward_flops"] != measured[ARM_C]["forward_flops"]
            or (measured[ARM_B]["forward_backward_flops"]
                != measured[ARM_C]["forward_backward_flops"])):
        raise RuntimeError("the B/C arms must not differ in parameters or FLOPs: the "
                           "attribution pair must be capacity-matched")
    if measured[ARM_A]["parameters"] >= measured[ARM_B]["parameters"]:
        raise RuntimeError("the evidence arms must carry strictly more parameters than "
                           "the incumbent; a non-increase means the pathway is not wired")
    protocol = protocol_payload(manifests_dir, identity, channels, measured, payload,
                                meta["typed_evidence_identity"])
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol_path = output_dir / "protocol.json"
    with protocol_path.open("x", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=False, allow_nan=False)
    verified_registration(protocol_path)

    # Wiring probes before any step. Each probe model is seeded immediately
    # before its own construction, so all three share one trunk initialization.
    probe_models = []
    for (name, kind, config), (_, _, resolved) in zip(arms, resolved_arms):
        seed_everything(seed)
        probe_models.append(make_model("process", resolved))
    first_state = probe_models[0].state_dict()
    for model in probe_models[1:]:
        other = model.state_dict()
        if not all(torch.equal(first_state[key], other[key]) for key in first_state):
            raise RuntimeError("probe precondition failed: the three probe models must "
                               "start from bitwise equal same-seed trunk weights")
    probe = evidence_probe(probe_models, probe_batch)
    for model in probe_models:
        del model
    if not probe["within_tolerance"]:
        raise RuntimeError(f"the typed-evidence wiring probe failed: {probe}")

    results = {
        "format": "r7-79-typed-evidence-three-arm-result-v1", "scientific_claim": False,
        "test_read": False, "seed": seed, "seeds": list(SEEDS), "device": device_name,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__), "platform": platform.platform(),
        "model_code_sha256": model_code_digest(), "protocol_sha256": protocol["protocol_sha256"],
        "protocol": protocol, "evidence_probe": probe, "flop_measurements": measured,
        "training": {}, "evaluation": {}, "probes": {}, "budget": {}}

    started = time.perf_counter()
    pairing = verify_arm_pairing(arms, baseline=ARM_A, channels=channels, seed=seed)
    if not pairing["all_shared_pairs_identical"]:
        raise RuntimeError("the three arms do not share bitwise identical seeded trunk "
                           "weights; the comparison would be confounded by initialization")
    results["arm_pairing"] = pairing

    anchors = {}
    for (name, kind, config), (_, _, resolved) in zip(arms, resolved_arms):
        if time.perf_counter() - started > deadline_seconds:
            raise RuntimeError(f"{deadline_seconds}s wall limit reached before {name} "
                               "started; stopping rather than overrunning")
        shared = None
        if name == ARM_A:
            seed_everything(seed)
            anchors[name] = {key: value.clone()
                             for key, value in make_model(kind, resolved).state_dict().items()}
        else:
            shared = anchors[ARM_A]
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
        if name != ARM_A and applied["ignored_count"]:
            raise RuntimeError(f"{name}: shared-state transfer ignored "
                               f"{applied['ignored_count']} tensors")
        loaded_names = {key for key in saved["model"]}
        added = sorted(loaded_names - set(anchors[ARM_A]))
        if (name == ARM_A and added) or (name != ARM_A and not added):
            raise RuntimeError(f"{name}: the evidence pathway is not visible in the "
                               f"checkpoint keys; added={added}")
        results["training"][name] = {
            "protocol_sha256": protocol["protocol_sha256"],
            "switches": arm_switches(config),
            "added_state_keys_vs_incumbent": added,
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
            "last_epoch_mean_loss": (report["losses"][-1]["loss"] if report["losses"] else None)}
        print(json.dumps({"trained": name, "seed": seed,
                          "selected_update": report["selected_update"],
                          "seconds": round(report["elapsed_seconds"], 1)}), flush=True)

    results["budget"]["training_seconds_total"] = time.perf_counter() - started

    for name, kind, config in arms:
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
                "skill_csv": str(run_dir / "climatology_skill.csv")}
    print(json.dumps({"evaluated_seed": seed, "leads": list(EVALUATION_LEADS)}), flush=True)

    for lead in EVALUATION_LEADS:
        counts = {name: results["evaluation"][f"{name}@{lead}h"]["n_evaluated"]
                  for name in ARM_NAMES}
        if len(set(counts.values())) != 1:
            raise RuntimeError(f"arms scored on different case counts at {lead}h: {counts}")
        results.setdefault("case_counts_by_lead", {})[str(lead)] = counts[ARM_A]

    with (output_dir / "seed_result.json").open("x", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return results
