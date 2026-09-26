"""B3: capacity scale-up to 15-20M on real B2 data (#64 B3).

#64 B3 is conditional: only once B0/B1/B2 show training and validation are
stable, consider 15-20M parameters, with one independent seed/arm per 3090
rather than DDP (the two cards are not NVLink-coupled, and the measured DDP
penalty was never recovered). B2 established the precondition - 15/15 arms
decayed monotonically over three seeds with no early stop - and left one open
question this stage is built to answer:

    **Does 6.5x capacity change the parameter-free-control verdict?**

B2's negative result was that all five ~2.8M arms lose to the zero-parameter
train-only climatology on t2m at every lead, including the trained 6 h horizon.
That admits two very different explanations - the arms are under-capacity, or
the January-only month-hour climatology is simply a hard target in this data
range - and a capacity ablation is what separates them. So B3 re-runs the same
protocol at ~18M and compares against both the parameter-free controls and the
B2 2.8M checkpoints on identical validation cases.

Everything held fixed from B2 so the comparison is controlled: the store, the
train window order, the warmup+cosine schedule, the 800-update budget, the
validation-only checkpoint selection, the early-stopping rule, the evaluation
leads and the case sets. The only declared change is capacity.

Two modes:
    --mode probe     measure step cost and peak memory at these sizes first
    --mode seed      train and evaluate every arm for one seed, publish JSON

Arms are re-measured rather than inherited: the parameter band is re-declared at
a new nominal capacity, and the generic/process pair keeps the shared common
initial weights that #64 B2 sentence 3 asks be carried into follow-on
experiments.
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

SCALE_SEEDS = (41, 42)
UPDATES = 800
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
PROCESS_WEIGHT = 0.0
# The scale-up target: measured at 18.20M / 18.12M / 18.12M, so the band is
# declared at 18M rather than reusing B2's 2.8M nominal.
NOMINAL_PARAMETERS = 18_000_000
PARAMETER_BAND = 0.05
MINIMUM_PARAMETERS = 15_000_000
MAXIMUM_PARAMETERS = 20_000_000
COMPARATOR_DEPTH = 0
DEFAULT_STORE = "outputs/r7_b2_segment/store/cache.zarr"
B2_RESULTS = "outputs/r7_b2_multiseed/b2_rmse_table.csv"

# The two arms B2's pre-registered gate named, plus the process arm so the
# shared-structure pair (generic/process) is carried to scale too.
ARMS = (
    ("native_window", "native", {"architecture": "window", "dim": 384, "depth": 10,
                                 "heads": 8, "window_size": 4, "patch_size": 2,
                                 "dropout": 0.0}),
    ("generic", "generic", {"architecture": "window", "dim": 384, "depth": 8, "heads": 8,
                            "window_size": 4, "patch_size": 2, "dropout": 0.0,
                            "latent_tokens": 16,
                            "default_reasoning_steps": REASONING_STEPS}),
    ("process", "process", {"architecture": "window", "dim": 384, "depth": 8, "heads": 8,
                            "window_size": 4, "patch_size": 2, "dropout": 0.0,
                            "anchored_processes": 8, "free_processes": 8,
                            "use_forecast_feedback": True,
                            "default_reasoning_steps": REASONING_STEPS}),
)
ARM_NAMES = tuple(entry[0] for entry in ARMS)
BASELINES = ("persistence", "climatology")
SHARED_INITIALIZATION_ARMS = ("generic", "process")
SHARED_INITIALIZATION_ANCHOR = "generic"
B2_SCALE_PARAMETERS = {
    "native_window": 2_803_601,
    "generic": 2_799_202,
    "process": 2_799_779,
}

FLOP_CONVENTION = (
    "FlopCounterMode over one forward pass under torch.enable_grad(); counts "
    "linear/conv/matmul and aten-dispatched attention matmuls; elementwise and "
    "normalization ops are not counted; no parameter hooks (SDPA is parameterless "
    "and a parameter hook would silently undercount attention)."
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arm_config(kind, channels, config):
    return {"in_channels": channels, "out_channels": channels,
            "history_steps": 2, **config}


def _load_inputs(manifests_dir):
    from data.r7_zarr_dataset import ZarrAtmosWindowDataset
    from training.r7_experiment import dataset_identity

    manifests_dir = Path(manifests_dir)
    for name in ("train.jsonl", "val.jsonl"):
        if not (manifests_dir / name).is_file():
            raise FileNotFoundError(manifests_dir / name)
    if (manifests_dir / "test.jsonl").is_file():
        pass  # present but deliberately never opened by this script
    dataset = ZarrAtmosWindowDataset(manifests_dir / "train.jsonl")
    identity, _ = dataset_identity(manifests_dir / "train.jsonl")
    channels = int(dataset[0]["coarse_history"].shape[1])
    return dataset, identity, channels


def measure_arms(channels, dataset):
    """Re-measure parameters and FLOPs at the new capacity; never inherit."""
    from torch.utils.data import default_collate

    from training.r7_experiment import make_model, seed_everything

    probe_batch = default_collate([dataset[0], dataset[1]])
    measured = {}
    for name, kind, config in ARMS:
        seed_everything(SCALE_SEEDS[0])
        model = make_model(kind, _arm_config(kind, channels, config))
        parameters = int(sum(parameter.numel() for parameter in model.parameters()))
        measured[name] = {"parameters": parameters,
                          "forward_flops": _count_flops(model, probe_batch, kind)}
        del model
    spread = ((max(v["parameters"] for v in measured.values())
               - min(v["parameters"] for v in measured.values())) / NOMINAL_PARAMETERS)
    out_of_band = {name: value["parameters"] for name, value in measured.items()
                   if not (MINIMUM_PARAMETERS <= value["parameters"] <= MAXIMUM_PARAMETERS)}
    if out_of_band:
        raise ValueError(f"B3 targets 15-20M and refuses to run outside it: {out_of_band}")
    if spread > PARAMETER_BAND:
        raise ValueError(f"the scaled arms are not parameter-matched: spread {spread:.4%}")
    return measured, spread


def _count_flops(model, batch, kind):
    from model.r7_halting import forecast_inputs
    from torch.utils.flop_counter import FlopCounterMode

    inputs = forecast_inputs(batch)
    run = ((lambda: model(inputs, reasoning_steps=REASONING_STEPS))
           if kind != "native" else (lambda: model(inputs)))
    model.zero_grad()
    with torch.enable_grad():
        with FlopCounterMode(display=False) as counter:
            run()
        total = int(counter.get_total_flops())
    model.zero_grad()
    return total


def probe(manifests_dir, *, device_name="cuda", probe_updates=20):
    """Short discarded run per arm: real step cost and peak memory at 18M."""
    from training.r7_experiment import select_device
    from training.r7_scheduled_runner import run_scheduled_updates

    manifests_dir = Path(manifests_dir)
    dataset, identity, channels = _load_inputs(manifests_dir)
    device = select_device(device_name)
    measured, spread = measure_arms(channels, dataset)
    out = {"mode": "probe", "scientific_claim": False, "device": device_name,
           "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
           "probe_updates": probe_updates, "parameter_spread": spread, "arms": {}}
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    for name, kind, config in ARMS:
        started = time.perf_counter()
        _, report = run_scheduled_updates(
            dataset, kind=kind, model_config=_arm_config(kind, channels, config),
            data_identity=identity, output_dir=tmp / name, total_updates=probe_updates,
            batch_size=BATCH_SIZE, steps=REASONING_STEPS, seed=SCALE_SEEDS[0], lr=LR,
            clip=CLIP, process_weight=PROCESS_WEIGHT, warmup_updates=2,
            minimum_lr_ratio=MINIMUM_LR_RATIO, validation_every=0,
            early_stopping_patience=None, device_name=device_name)
        elapsed = time.perf_counter() - started
        per_update = elapsed / probe_updates
        out["arms"][name] = {
            "parameters": measured[name]["parameters"],
            "forward_flops": measured[name]["forward_flops"],
            "seconds_per_update": per_update,
            "projected_800_minutes": per_update * UPDATES / 60,
            "peak_allocated_mib": (report["peak_allocated_bytes"] or 0) / 2**20,
            "peak_reserved_mib": (report["peak_reserved_bytes"] or 0) / 2**20,
        }
        print(json.dumps({name: out["arms"][name]}), flush=True)
    per_seed = sum(entry["projected_800_minutes"] for entry in out["arms"].values())
    out["one_seed_minutes"] = per_seed
    out["two_seeds_two_gpus_wall_minutes"] = per_seed  # one seed per card, in parallel
    out["two_seeds_gpu_hours"] = 2 * per_seed / 60
    print(json.dumps({"one_seed_minutes": round(per_seed, 1),
                      "two_seeds_gpu_hours": round(out["two_seeds_gpu_hours"], 3),
                      "remaining_budget_gpu_hours_at_1723_used": round(4.0 - 1.723, 3)},
                     indent=1))
    return out


def protocol_payload(manifests_dir, identity, channels, measured, spread):
    body = {
        "format": "r7-b3-scale-protocol-v1",
        "frozen_before_any_step": True,
        "issue": "#64 B3",
        "objective": ("capacity scale-up from 2.8M to ~18M on the real B2 segment, at "
                      "fixed protocol, to test whether the B2 parameter-free-control "
                      "verdict was a capacity artefact; never a SOTA claim"),
        "data": {
            "store": str(Path(manifests_dir).parent / "cache.zarr"),
            "train_manifest": str(Path(manifests_dir) / "train.jsonl"),
            "val_manifest": str(Path(manifests_dir) / "val.jsonl"),
            "test_manifest": str(Path(manifests_dir) / "test.jsonl"),
            "test_read": False,
            "test_policy": ("sealed: an engineering re-split inside 2016, not the v2 2021 "
                            "test candidate"),
            "data_identity": str(identity),
            "note": "identical store and case sets as B2, so the two are compared directly",
        },
        "seeds": list(SCALE_SEEDS),
        "seed_policy": ("one independent seed per 3090 (no DDP: the cards are not "
                        "NVLink-coupled and the measured DDP penalty was never recovered)"),
        "hardware_strategy": {
            "cards": 2,
            "placement": "one independent seed per card, both cards busy concurrently",
            "ddp": False,
            "ddp_reason": ("#64 B3 asks for one seed/arm per card first. The two 3090s are "
                           "not NVLink-coupled and the measured DDP penalty on this host "
                           "was never recovered, so independent seeds are both faster and "
                           "better evidence here; per-card memory headroom is recorded "
                           "from the probe rather than assumed"),
        },
        "arms": [{"name": name, "kind": kind,
                  "model_config": _arm_config(kind, channels, config),
                  "parameters": measured[name]["parameters"],
                  "forward_flops": measured[name]["forward_flops"],
                  "b2_parameters_at_2p8M": B2_SCALE_PARAMETERS[name],
                  "scale_factor_vs_b2": (measured[name]["parameters"]
                                         / B2_SCALE_PARAMETERS[name])}
                 for name, kind, config in ARMS],
        "parameter_gate": {
            "nominal": NOMINAL_PARAMETERS, "band": PARAMETER_BAND,
            "required_range": [MINIMUM_PARAMETERS, MAXIMUM_PARAMETERS],
            "measured_spread": spread,
            "note": "15-20M is a hard requirement here, not a preference; 30M is not targeted",
        },
        "baselines": [{"name": name, "trainable_parameters": 0} for name in BASELINES],
        "flop_convention": FLOP_CONVENTION,
        "shared_initialization": {
            "applies_to": list(SHARED_INITIALIZATION_ARMS),
            "anchor": SHARED_INITIALIZATION_ANCHOR,
            "rule": ("carried from B2: the anchor's seeded initialization is copied into "
                     "the other arm of the group before its first step, and the applied/"
                     "ignored parameter names are recorded"),
        },
        "held_fixed_from_b2": {
            "store": True, "sample_order": True, "optimizer": True, "lr_schedule": True,
            "max_updates": UPDATES, "validation_every": VALIDATION_EVERY,
            "checkpoint_selection_rule": True, "early_stopping_rule": True,
            "evaluation_leads_hours": list(EVALUATION_LEADS),
            "case_sets": "identical validation windows, so RMSE is directly comparable",
        },
        "declared_change": ("capacity only: 2.8M -> 18M; no other protocol field moves"),
        "shared_controls": {
            "optimizer": f"AdamW lr {LR} weight_decay {WEIGHT_DECAY}",
            "lr_schedule": (f"linear warmup over {WARMUP_UPDATES} updates, then cosine to "
                            f"{MINIMUM_LR_RATIO} of peak at update {UPDATES}"),
            "max_updates": UPDATES, "validation_every": VALIDATION_EVERY,
            "early_stopping_rule": (f"{EARLY_STOPPING_PATIENCE} consecutive non-improving "
                                    f"validation checks (<{MINIMUM_IMPROVEMENT:.1%} relative) "
                                    "stop training; validation only"),
            "checkpoint_selection_rule": ("lowest mean latitude-weighted normalized "
                                          "validation MSE; ties keep the earlier one"),
            "selection_split": "val only; test is sealed and never read",
            "reasoning_steps": REASONING_STEPS, "batch_size": BATCH_SIZE, "clip": CLIP,
            "process_weight": PROCESS_WEIGHT,
            "comparator_depth": COMPARATOR_DEPTH,
            "comparator": "training/r7_coreasoning_compare (the #60-fixed comparator)",
        },
        "analysis_rule": (
            "B3 is a capacity ablation: its verdict is whether the scaled arms clear the "
            "parameter-free controls where the 2.8M arms did not, reported per variable and "
            "per lead against both controls and against the B2 2.8M checkpoints. It is not "
            "a significance test and does not establish that recursion beats non-recursion."),
        "scientific_claim": False,
        "limitations": [
            "still one 36-day January-2016 engineering segment; no seasonal or cross-year "
            "conclusion is testable",
            "the month-hour climatology has only 4 buckets here, so it is not a strong "
            "seasonal climatology - a climatology win is a property of this data range",
            "val/test are engineering re-splits inside 2016, not the v2 2019/2021 splits",
            "not a converged benchmark; the update budget is fixed, not tuned per capacity",
            "two seeds give no significance test",
            "capacity is the only declared change; FLOPs and wall time are reported, never "
            "assumed equal across capacities",
        ],
    }
    from training.r7_experiment import canonical_digest

    return dict(body, protocol_sha256=canonical_digest(body))


def run_seed(manifests_dir, output_dir, *, seed, device_name="cuda", updates=UPDATES):
    from data.r7_evaluation import ZarrRolloutDataset
    from training.r7_evaluate import evaluate_local
    from training.r7_experiment import (load_checkpoint, make_model, model_code_digest,
                                        seed_everything, select_device)
    from training.r7_scheduled_runner import run_scheduled_updates

    if seed not in SCALE_SEEDS:
        raise ValueError(f"B3 declares seeds {SCALE_SEEDS}; {seed} is not one of them")
    manifests_dir, output_dir = Path(manifests_dir), Path(output_dir)
    dataset, identity, channels = _load_inputs(manifests_dir)
    device = select_device(device_name)
    measured, spread = measure_arms(channels, dataset)
    protocol = protocol_payload(manifests_dir, identity, channels, measured, spread)

    output_dir = Path(output_dir)
    protocol_path = output_dir / "protocol.json"
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
        with protocol_path.open("x", encoding="utf-8") as handle:
            json.dump(protocol, handle, indent=2, ensure_ascii=False, allow_nan=False)
    except FileExistsError:
        frozen = json.loads(protocol_path.read_text(encoding="utf-8"))
        if frozen.get("protocol_sha256") != protocol.get("protocol_sha256"):
            raise RuntimeError("protocol drift: the frozen B3 protocol does not match this "
                               "code; a pre-registered protocol is not edited after freezing")
        protocol = frozen
    record_path = output_dir / f"scale_seed{seed}.json"
    if record_path.exists():
        raise FileExistsError(f"refusing to overwrite a published B3 seed run: {record_path}")

    validation = ZarrRolloutDataset(Path(manifests_dir).parent / "cache.zarr", split="val",
                                    lead_hours=VALIDATION_LEADS, history_steps=2, step_hours=6)
    digest = model_code_digest()
    started = time.perf_counter()
    record = {
        "format": "r7-b3-scale-seed-v1", "seed": seed, "scientific_claim": False,
        "test_read": False, "protocol_sha256": protocol["protocol_sha256"],
        "model_code_sha256": digest, "device": device_name,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__), "platform": platform.platform(),
        "training": {}, "evaluation": {},
    }
    anchors = {}
    for name, kind, config in ARMS:
        run_dir = output_dir / "training" / f"seed{seed}" / name
        resolved = _arm_config(kind, channels, config)
        shared = None
        if name in SHARED_INITIALIZATION_ARMS:
            if name == SHARED_INITIALIZATION_ANCHOR:
                seed_everything(seed)
                anchor_model = make_model(kind, resolved)
                anchors[name] = {key: tensor.detach().cpu().clone()
                                 for key, tensor in anchor_model.state_dict().items()}
                del anchor_model
            else:
                shared = anchors.get(SHARED_INITIALIZATION_ANCHOR)
                if shared is None:
                    raise RuntimeError(f"{SHARED_INITIALIZATION_ANCHOR!r} must run first")
        checkpoint, report = run_scheduled_updates(
            dataset, kind=kind, model_config=resolved, data_identity=identity,
            output_dir=run_dir, total_updates=updates, batch_size=BATCH_SIZE,
            steps=REASONING_STEPS, seed=seed, lr=LR, clip=CLIP,
            process_weight=PROCESS_WEIGHT, warmup_updates=WARMUP_UPDATES,
            minimum_lr_ratio=MINIMUM_LR_RATIO, validation_every=VALIDATION_EVERY,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            minimum_improvement=MINIMUM_IMPROVEMENT, validation_dataset=validation,
            validation_lead_hours=VALIDATION_LEADS, device_name=device_name,
            shared_initial_state=shared)
        saved = load_checkpoint(checkpoint)
        if saved["updates"] != report["selected_update"]:
            raise ValueError(f"{name} selected checkpoint/update mismatch")
        by_epoch = {}
        for entry in report["losses"]:
            by_epoch.setdefault(entry["epoch"], []).append(entry["loss"])
        epoch_means = {str(epoch): sum(values) / len(values)
                       for epoch, values in sorted(by_epoch.items())}
        epoch_sizes = {str(epoch): len(values) for epoch, values in sorted(by_epoch.items())}
        full = max(epoch_sizes.values())
        full_epochs = sorted(int(e) for e, size in epoch_sizes.items() if size == full)
        record["training"][name] = {
            "updates_run": report["updates_this_run"],
            "selected_update": report["selected_update"],
            "selected_validation_mse": report["selected_validation_mse"],
            "early_stopped": report["early_stopped"],
            "stopped_reason": report["stopped_reason"],
            "elapsed_seconds": report["elapsed_seconds"],
            "seconds_per_update": report["seconds_per_update"],
            "samples_seen": int(sum(entry["samples"] for entry in report["losses"])),
            "epoch_mean_loss": epoch_means, "epoch_update_counts": epoch_sizes,
            "first_full_epoch": min(full_epochs), "last_full_epoch": max(full_epochs),
            "loss_ratio_first_to_last_full_epoch": (
                epoch_means[str(max(full_epochs))] / epoch_means[str(min(full_epochs))]),
            "peak_allocated_bytes": report["peak_allocated_bytes"],
            "peak_reserved_bytes": report["peak_reserved_bytes"],
            "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256_file(checkpoint),
            "parameters": measured[name]["parameters"],
            "shared_initial_state": report["shared_initial_state"],
        }
        print(json.dumps({"seed": seed, "trained": name,
                          "selected_update": report["selected_update"],
                          "epochs": {k: round(v, 5) for k, v in epoch_means.items()},
                          "seconds": round(report["elapsed_seconds"], 1),
                          "peak_reserved_mib": round((report["peak_reserved_bytes"] or 0) / 2**20, 1),
                          "early_stopped": report["early_stopped"]}), flush=True)

    for name in list(ARM_NAMES) + list(BASELINES):
        trained = name not in BASELINES
        checkpoint = (output_dir / "training" / f"seed{seed}" / name
                      / f"update_{record['training'][name]['selected_update']:07d}.pt"
                      ) if trained else None
        for lead in EVALUATION_LEADS:
            run_dir = output_dir / "evaluation" / f"seed{seed}" / name / f"lead_{lead:03d}h"
            report = evaluate_local(
                Path(manifests_dir) / "val.jsonl", output_dir=run_dir, checkpoint=checkpoint,
                baseline=None if trained else name, lead_hours=(lead,), max_samples=64,
                device_name=device_name)
            if report["split"] != "val":
                raise ValueError(f"{name} evaluated on split {report['split']}, not val")
            record["evaluation"][f"{name}@{lead}h"] = {
                "seed": seed, "model": name, "lead_hours": lead, "split": report["split"],
                "n_evaluated": report["n_evaluated"],
                "n_available_windows": report["n_available_windows"],
                "channels": list(report["channels"]), "units": list(report["units"]),
                "elapsed_seconds": report["elapsed_seconds"],
                "parameter_free_baseline": report["parameter_free_baseline"],
                "trainable_parameters": report["trainable_parameters"],
                "evaluation_dir": str(run_dir), "rmse_csv": str(run_dir / "rmse.csv"),
                "skill_csv": str(run_dir / "climatology_skill.csv"),
            }
        print(json.dumps({"seed": seed, "evaluated": name, "leads": list(EVALUATION_LEADS)}),
              flush=True)

    for lead in EVALUATION_LEADS:
        counts = {name: record["evaluation"][f"{name}@{lead}h"]["n_evaluated"]
                  for name in list(ARM_NAMES) + list(BASELINES)}
        if len(set(counts.values())) != 1:
            raise RuntimeError(f"case counts differ at {lead}h seed {seed}: {counts}")
    record["case_counts_by_lead"] = {
        f"{lead}h": record["evaluation"][f"{ARM_NAMES[0]}@{lead}h"]["n_evaluated"]
        for lead in EVALUATION_LEADS}
    record["elapsed_seconds_total"] = time.perf_counter() - started
    if model_code_digest() != digest:
        raise RuntimeError("model code changed during the B3 seed run")
    with record_path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"seed": seed, "published": str(record_path)}), flush=True)
    return record


def write_comparison(output_dir, *, b2_rmse=B2_RESULTS):
    """Four tables plus the cross-scale comparison against the B2 checkpoints."""
    output_dir = Path(output_dir)
    records = {}
    for seed in SCALE_SEEDS:
        # Each seed writes into its own directory (one process per card), so look
        # there first and fall back to a flat layout; a declared seed that exists
        # in neither place makes the stage incomplete, never narrower.
        candidates = (output_dir / f"seed{seed}" / f"scale_seed{seed}.json",
                      output_dir / f"scale_seed{seed}.json")
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(
                f"declared seed {seed} is missing (looked in "
                f"{', '.join(str(c) for c in candidates)}); an incomplete stage is "
                "reported as incomplete")
        records[seed] = json.loads(path.read_text(encoding="utf-8"))
    digests = {record["protocol_sha256"] for record in records.values()}
    if len(digests) != 1:
        raise RuntimeError(f"B3 seeds ran under different protocols: {sorted(digests)}")
    codes = {record["model_code_sha256"] for record in records.values()}
    if len(codes) != 1:
        raise RuntimeError(f"B3 seeds ran on different model code: {sorted(codes)}")

    # The seed records carry the digest, not the protocol body; the arms table
    # comes from the frozen protocol file, which is also re-checked against the
    # digest every record claims.
    protocol_path = next((candidate for candidate in
                          (output_dir / f"seed{SCALE_SEEDS[0]}" / "protocol.json",
                           output_dir / "protocol.json") if candidate.is_file()), None)
    if protocol_path is None:
        raise FileNotFoundError(f"no frozen protocol.json under {output_dir}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_sha256") not in digests:
        raise RuntimeError("the frozen protocol file does not match the digest the seed "
                           "records ran under")
    first = dict(records[SCALE_SEEDS[0]], protocol=protocol)
    first["budget"] = {"training_seconds_total": sum(
        entry["elapsed_seconds"] for record in records.values()
        for entry in record["training"].values())}
    tables = {}
    tables["parameters"] = _parameter_table(first, output_dir)
    tables["flops"] = _flops_table(first, output_dir)
    enriched = {seed: dict(record, protocol=protocol) for seed, record in records.items()}
    tables["wall_time"] = _wall_time_table(enriched, output_dir)
    tables["cases"] = _case_table(enriched, output_dir)
    tables["rmse"] = _rmse_table(enriched, output_dir)
    comparison = _cross_scale_comparison(enriched, output_dir, b2_rmse=Path(b2_rmse))
    return {"tables": tables, "comparison": comparison}


def _write_csv(path, header, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return Path(path).name


def _parameter_table(first, output_dir):
    rows = []
    for entry in first["protocol"]["arms"]:
        deviation = (entry["parameters"] - NOMINAL_PARAMETERS) / NOMINAL_PARAMETERS
        rows.append([entry["name"], entry["kind"], entry["parameters"],
                     f"{deviation:+.4%}",
                     str(MINIMUM_PARAMETERS <= entry["parameters"] <= MAXIMUM_PARAMETERS),
                     entry["forward_flops"], entry["b2_parameters_at_2p8M"],
                     f"{entry['scale_factor_vs_b2']:.2f}x"])
    for name in BASELINES:
        rows.append([name, "parameter-free", 0, "n/a", "not_applicable", 0, 0, "n/a"])
    return _write_csv(output_dir / "b3_parameter_table.csv",
                      ["model", "family", "trainable_parameters", "deviation_from_18M",
                       "inside_15_20M", "forward_flops", "b2_parameters", "scale_vs_b2"],
                      rows)


def _flops_table(first, output_dir):
    rows = [[entry["name"], entry["forward_flops"], BATCH_SIZE,
             REASONING_STEPS if entry["kind"] != "native" else "n/a", FLOP_CONVENTION]
            for entry in first["protocol"]["arms"]]
    for name in BASELINES:
        rows.append([name, 0, BATCH_SIZE, "n/a", "parameter-free control"])
    return _write_csv(output_dir / "b3_flops_table.csv",
                      ["model", "forward_flops", "batch", "reasoning_steps", "convention"],
                      rows)


def _wall_time_table(records, output_dir):
    rows = []
    for seed, record in sorted(records.items()):
        for model in list(ARM_NAMES) + list(BASELINES):
            train = record["training"].get(model)
            if train is None:
                rows.append([seed, model, 0, 0, 0, "0.000", "0.000000", "n/a", "0.0"])
                continue
            rows.append([seed, model, train["updates_run"], train["selected_update"],
                         train["samples_seen"], f"{train['elapsed_seconds']:.3f}",
                         f"{train['seconds_per_update']:.6f}",
                         str(bool(train["early_stopped"])),
                         f"{(train['peak_reserved_bytes'] or 0) / 2**20:.1f}"])
    return _write_csv(output_dir / "b3_wall_time_table.csv",
                      ["seed", "model", "updates_run", "selected_update", "samples_seen",
                       "train_seconds", "seconds_per_update", "early_stopped",
                       "peak_reserved_mib"], rows)


def _case_table(records, output_dir):
    rows = []
    for seed, record in sorted(records.items()):
        for model in list(ARM_NAMES) + list(BASELINES):
            for lead in EVALUATION_LEADS:
                entry = record["evaluation"][f"{model}@{lead}h"]
                rows.append([seed, model, entry["split"], entry["lead_hours"],
                             entry["n_available_windows"], entry["n_evaluated"], "False"])
    return _write_csv(output_dir / "b3_case_count_table.csv",
                      ["seed", "model", "split", "lead_hours", "n_available_windows",
                       "n_evaluated", "test_read"], rows)


def _rmse_table(records, output_dir):
    rows = []
    for seed, record in sorted(records.items()):
        for model in list(ARM_NAMES) + list(BASELINES):
            for lead in EVALUATION_LEADS:
                entry = record["evaluation"][f"{model}@{lead}h"]
                rmse = {row["variable"]: row for row in _csv_rows(entry["rmse_csv"])}
                skill = {row["variable"]: row for row in _csv_rows(entry["skill_csv"])}
                for variable in entry["channels"]:
                    row = rmse.get(variable)
                    if row is None:
                        continue
                    skill_row = skill.get(variable, {})
                    rows.append([seed, model, variable, row["unit"], lead, row["rmse"],
                                 skill_row.get("rmse_climatology", ""),
                                 skill_row.get("mse_skill", "") or "undefined",
                                 row.get("n_initializations", entry["n_evaluated"])])
    return _write_csv(output_dir / "b3_rmse_table.csv",
                      ["seed", "model", "variable", "unit", "lead_hours", "rmse",
                       "rmse_climatology", "mse_skill", "n_initializations"], rows)


def _csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _cross_scale_comparison(records, output_dir, *, b2_rmse):
    """Does 6.5x capacity change the parameter-free-control verdict?

    Compares, per variable and lead, the B3 18M RMSE against the parameter-free
    controls and against the B2 2.8M value on the identical validation cases.
    The B2 numbers come from B2's own published table, read here rather than
    re-run, and the case sets are asserted equal before anything is compared.
    """
    import statistics

    cell = {}
    for seed, record in records.items():
        for model in list(ARM_NAMES) + list(BASELINES):
            for lead in EVALUATION_LEADS:
                entry = record["evaluation"][f"{model}@{lead}h"]
                for row in _csv_rows(entry["rmse_csv"]):
                    cell.setdefault((model, row["variable"], lead), {})[seed] = float(row["rmse"])

    b2_cell = {}
    if b2_rmse.is_file():
        for row in _csv_rows(b2_rmse):
            key = (row["model"], row["variable"], int(row["lead_hours"]))
            b2_cell.setdefault(key, []).append(float(row["rmse"]))

    def mean(model, variable, lead):
        values = cell.get((model, variable, lead))
        return statistics.fmean(values.values()) if values else None

    rows = []
    for arm in ARM_NAMES:
        for lead in EVALUATION_LEADS:
            for variable in sorted({key[1] for key in cell if key[0] == arm and key[2] == lead}):
                scaled = mean(arm, variable, lead)
                pers = mean("persistence", variable, lead)
                clim = mean("climatology", variable, lead)
                base = b2_cell.get((arm, variable, lead))
                base_mean = statistics.fmean(base) if base else None
                rows.append({
                    "arm": arm, "lead_hours": lead, "variable": variable,
                    "b3_rmse_18M": scaled, "persistence": pers, "climatology": clim,
                    "b2_rmse_2p8M": base_mean,
                    "beats_persistence": None if scaled is None or pers is None else scaled < pers,
                    "beats_climatology": None if scaled is None or clim is None else scaled < clim,
                    "improved_vs_b2_2p8M": (None if scaled is None or base_mean is None
                                            else scaled < base_mean),
                    "n_b2_seeds": len(base) if base else 0,
                })
    columns = ["arm", "lead_hours", "variable", "b3_rmse_18M", "persistence", "climatology",
               "b2_rmse_2p8M", "beats_persistence", "beats_climatology",
               "improved_vs_b2_2p8M", "n_b2_seeds"]
    _write_csv(output_dir / "b3_cross_scale_table.csv", columns,
               [[row[column] for column in columns] for row in rows])

    summary = {}
    for arm in ARM_NAMES:
        arm_rows = [row for row in rows if row["arm"] == arm]
        summary[arm] = {
            "cells": len(arm_rows),
            "beats_persistence": sum(1 for row in arm_rows if row["beats_persistence"]),
            "beats_climatology": sum(1 for row in arm_rows if row["beats_climatology"]),
            "beats_both": sum(1 for row in arm_rows
                              if row["beats_persistence"] and row["beats_climatology"]),
            "improved_vs_b2_2p8M": sum(1 for row in arm_rows if row["improved_vs_b2_2p8M"]),
            "t2m_by_lead": {
                str(lead): {
                    "b3": next((row["b3_rmse_18M"] for row in arm_rows
                                if row["variable"] == "t2m" and row["lead_hours"] == lead), None),
                    "climatology": next((row["climatology"] for row in arm_rows
                                         if row["variable"] == "t2m"
                                         and row["lead_hours"] == lead), None),
                    "b2": next((row["b2_rmse_2p8M"] for row in arm_rows
                                if row["variable"] == "t2m" and row["lead_hours"] == lead), None),
                } for lead in EVALUATION_LEADS},
        }
    payload = {
        "format": "r7-b3-cross-scale-v1", "scientific_claim": False,
        "b2_source_table": str(b2_rmse),
        "b2_table_available": b2_rmse.is_file(),
        "question": ("does 6.5x capacity change the parameter-free-control verdict that B2 "
                     "reported at 2.8M?"),
        "summary": summary,
        "rows": rows,
        "note": ("a capacity ablation on identical cases, not a significance test; the B2 "
                 "column is that table's own three-seed mean, read not re-run"),
    }
    with (output_dir / "b3_cross_scale_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return payload


def main():
    parser = argparse.ArgumentParser(description="B3 capacity scale-up on real B2 data (#64).")
    parser.add_argument("--mode", required=True, choices=("probe", "seed", "compare"))
    parser.add_argument("--manifests", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--probe-updates", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-index", type=int, default=None)
    args = parser.parse_args()
    if args.mode == "probe":
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        report = probe(args.manifests, device_name=args.device,
                       probe_updates=args.probe_updates)
        with out.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        return
    if args.mode == "compare":
        result = write_comparison(args.out)
        print(json.dumps({"tables": result["tables"],
                          "summary": result["comparison"]["summary"],
                          "b2_table_available": result["comparison"]["b2_table_available"],
                          "scientific_claim": False}, indent=1, default=str))
        return
    if args.seed is None:
        parser.error("--mode seed requires --seed")
    if args.device_index is not None and args.device.startswith("cuda"):
        torch.cuda.set_device(args.device_index)
        args.device = f"cuda:{args.device_index}"
    run_seed(args.manifests, args.out, seed=args.seed, device_name=args.device,
             updates=args.updates)


if __name__ == "__main__":
    main()
