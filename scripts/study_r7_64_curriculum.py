"""#64 remainder: a bounded 1->2->4 lead-time training curriculum.

What this tests, and what it deliberately does not. #64 asks for a curriculum
over **prediction lead times** (train at +6h, then +12h, then +24h) as a separate
bounded experiment. That is a different axis from the model's **internal
reasoning depth K**: K is extra recurrence inside one forecast step and never
advances physical time, while a lead-time curriculum changes which future hour
the target is. The two are never mixed here, and the protocol says so explicitly
so a later reader cannot read one as the other.

Design, frozen before any step:

- the curriculum arm is trained on +6h, then +12h, then +24h, each stage getting
  its own share of updates and a warm restart of the schedule;
- the control is the same architecture trained on +6h for the **whole** update
  budget, so the comparison is "same total updates, same data, different
  distribution over targets";
- because the curriculum arm's later stages train on longer leads, the two arms
  see different target sets by construction. That is the intervention, not a
  confound, but it means the arms are **not** compared on "same data" - only on
  same store, same window init times and same total updates. The protocol records
  this rather than claiming an alignment it does not have;
- evaluation is free 6/12/24/48/72 h rollout on val, so both arms are scored on
  the leads the curriculum trained and the leads it did not.

    python scripts/study_r7_64_curriculum.py --device cuda --out outputs/r7_64_curriculum
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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_STORE = "outputs/r7_b2_segment/store/cache.zarr"
DEFAULT_MANIFESTS = "outputs/r7_b2_segment/store/manifests"

SEEDS = (41, 42, 43)
TOTAL_UPDATES = 800
# The curriculum's three stages. Weights are the share of TOTAL_UPDATES, chosen
# so the stages are not equal: the +6h stage keeps the largest share because it
# is the lead every evaluation includes, and the later stages exist to shape the
# multi-lead behaviour rather than to replace it.
CURRICULUM_STAGES = ((6, 0.5), (12, 0.25), (24, 0.25))
CONTROL_LEAD_HOURS = 6
BATCH_SIZE = 2
LR = 2e-4
CLIP = 1.0
WEIGHT_DECAY = 1e-4
WARMUP_UPDATES = 40
MINIMUM_LR_RATIO = 0.1
VALIDATION_EVERY = 100
EARLY_STOPPING_PATIENCE = 4
MINIMUM_IMPROVEMENT = 0.001
VALIDATION_LEADS = (6,)
EVALUATION_LEADS = (6, 12, 24, 48, 72)
REASONING_STEPS = 4
# #65 C1 found aux=0.1 the only setting with a seed-consistent (negative) effect
# on t2m; the curriculum question is orthogonal, so the auxiliary loss stays off
# here and the arm is described as an architecture comparison only.
PROCESS_WEIGHT = 0.0
ARCHITECTURE = {"architecture": "window", "dim": 192, "depth": 4, "heads": 4,
                "window_size": 4, "patch_size": 2, "dropout": 0.0,
                "anchored_processes": 8, "free_processes": 8,
                "use_forecast_feedback": True,
                "default_reasoning_steps": REASONING_STEPS}

FLOP_CONVENTION = (
    "FlopCounterMode over one forward pass under torch.enable_grad(); counts "
    "linear/conv/matmul and aten-dispatched attention matmuls; elementwise and "
    "normalization ops are not counted; no parameter hooks."
)
LIMITATIONS = [
    "one 36-day engineering segment (January 2016) of real ERA5; single season",
    "val/test are engineering re-splits inside 2016, not a sealed test set; "
    "test is never read",
    "the curriculum arm trains on a different multi-lead target set than the "
    "control by construction - that is the intervention, and the two are NOT "
    "claimed to be trained on identical data",
    "three seeds are not a significance test",
    "bounded checkpoints, not converged training runs",
    "this curriculum is over prediction LEAD TIMES and is unrelated to the "
    "model's internal reasoning depth K",
    "no precipitation target exists in this store, so no precipitation skill is claimed",
]
SCIENTIFIC_CLAIM = False


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_updates(total_updates=TOTAL_UPDATES):
    """The per-stage update counts, with the remainder given to the last stage so
    the stages always sum exactly to the declared budget."""
    counts = {}
    assigned = 0
    for index, (lead, share) in enumerate(CURRICULUM_STAGES):
        if index == len(CURRICULUM_STAGES) - 1:
            counts[lead] = total_updates - assigned
        else:
            counts[lead] = int(total_updates * share)
            assigned += counts[lead]
    return counts


def build_lead_manifest(store, split, lead_hours, *, history_steps=2, step_hours=6):
    """Build one manifest's records for an explicit lead, from the store itself.

    This mirrors the store's own time-range split ownership (decision 0005)
    instead of re-deriving it: a window is kept only when every history stamp,
    the init stamp and the target stamp all belong to the requested split.
    """
    import zarr

    from data.preprocess.contracts import utc_time_index
    from data.r7_store import split_time_labels, validate_record, validate_store

    root = zarr.open_group(str(store), mode="r")
    validate_store(root)
    labels = split_time_labels(root)
    times = utc_time_index(np.asarray(root["time_ns"][:]).astype("datetime64[ns]"))
    steps = lead_hours // step_hours
    if lead_hours % step_hours:
        raise ValueError("lead hours must be a multiple of the transition step")
    records = []
    for position in range(history_steps - 1, len(times)):
        if labels is not None and labels[position] != split:
            continue
        history = [position - (history_steps - 1 - j) for j in range(history_steps)]
        target = position + steps
        required = history + [target]
        if any(index >= len(times) or (labels is not None and labels[index] != split)
               for index in required):
            continue
        record = {
            "sample_id": f"era5z_{split}_{times[position]:%Y%m%d%H}_p{lead_hours:03d}h",
            "store_path": str(Path(store).resolve()),
            "split": split,
            "history_indices": history,
            "target_index": target,
            "history_times": [times[index].isoformat() for index in history],
            "init_time": times[position].isoformat(),
            "target_time": times[target].isoformat(),
            "lead_time_hours": lead_hours,
        }
        # the window must satisfy the store's own contract before it is written
        validate_record(root, record)
        records.append(record)
    if not records:
        raise ValueError(f"no {split} windows at lead {lead_hours}h")
    ids = [record["sample_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample ids in the generated manifest")
    return records


def write_manifest(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def publish_manifest_build(marker_dir, manifests, window_counts):
    """Write the build-complete marker the readers require.

    Every manifest consumer calls ``require_complete_manifest``, which refuses a
    manifest without this marker - a deliberate guard so a half-written window
    list cannot be read as a dataset. These manifests are derived from an
    already-published store, so the marker records which store and which lead
    each file came from rather than implying a new data build.
    """
    marker_dir = Path(marker_dir)
    marker_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "build_complete": True,
        "format": "r7-curriculum-manifest-build-v1",
        "derived_from_store": str(Path(DEFAULT_STORE).resolve()),
        "manifests": {str(lead): Path(path).name for lead, path in manifests.items()},
        "window_counts": {str(lead): int(count) for lead, count in window_counts.items()},
        "note": ("each manifest lists windows built from the published store's own "
                 "time-range split; this marker records the derivation, it does not "
                 "claim a new data build"),
    }
    marker = marker_dir / "BUILD_COMPLETE.json"
    if marker.exists():
        raise FileExistsError(marker)
    with marker.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return marker


class LeadManifestDataset:
    """A dataset over explicit manifest records, with a fixed lead."""

    def __init__(self, store, records):
        import zarr

        from data.r7_store import normalization, validate_store

        self.store = str(Path(store).resolve())
        self.root = zarr.open_group(self.store, mode="r")
        validate_store(self.root)
        self.records = list(records)
        if not self.records:
            raise ValueError("nonempty records required")
        state = self.root["state"]
        mean, std = normalization(self.root, count=state.shape[1])
        self.mean, self.std = np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
        self.latitude = np.asarray(self.root["latitude"][:], dtype=np.float32)
        self.longitude = np.asarray(self.root["longitude"][:], dtype=np.float32)
        self.channels = list(self.root.attrs["channels"])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        state = self.root["state"]
        indices = list(record["history_indices"]) + [record["target_index"]]
        frames = np.stack([np.asarray(state[i], dtype=np.float32) for i in indices])
        frames = (frames - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        if not np.isfinite(frames).all():
            raise ValueError("nonfinite atmospheric sample")
        return {
            "coarse_history": torch.from_numpy(frames[:-1]),
            "atmos_target": torch.from_numpy(frames[-1]),
            "lead_time_hours": torch.tensor(float(record["lead_time_hours"])),
            "latitude": torch.from_numpy(self.latitude),
            "longitude": torch.from_numpy(self.longitude),
            "sample_id": record["sample_id"],
        }


def _common_config(channels):
    return {"in_channels": channels, "out_channels": channels, "history_steps": 2,
            **ARCHITECTURE}


def protocol_payload(manifests, identity, channels, measured, stage_counts,
                     lead_manifest_sha256=None):
    body = {
        "format": "r7-64-curriculum-protocol-v1",
        "frozen_before_any_step": True,
        "issue": "#64 remainder (lead-time curriculum)",
        "objective": ("a bounded 1->2->4 prediction-lead curriculum against a fixed +6h "
                      "control at the same total update budget; never a SOTA claim, and "
                      "never a statement about the model's internal reasoning depth K"),
        "data": {
            "store": str(Path(DEFAULT_STORE).resolve()),
            "manifests": {str(lead): str(path) for lead, path in manifests.items()},
            "lead_manifest_sha256": dict(lead_manifest_sha256 or {}),
            "data_identity": str(identity),
            "test_read": False,
            "normalization": "store train-only centered mean/std, applied at read time",
        },
        "curriculum": {
            "stages": [{"lead_hours": lead, "updates": stage_counts[lead],
                        "share": share} for lead, share in CURRICULUM_STAGES],
            "control_lead_hours": CONTROL_LEAD_HOURS,
            "control_updates": TOTAL_UPDATES,
            "alignment_claim": ("same store, same init times where the lead allows, same "
                                "total updates, same schedule shape, same model config"),
            "explicitly_not_aligned": [
                "the target set: the curriculum arm trains on +6/+12/+24h targets while "
                "the control trains on +6h only",
                "the number of available windows per stage (94/93/91 at 6/12/24h)",
            ],
            "not_the_reasoning_depth": ("the curriculum changes which future hour is the "
                                        "target; the internal recurrence K never advances "
                                        "physical time and is held at the declared value"),
        },
        "seeds": list(SEEDS),
        "seed_policy": "declared before running; every declared seed is reported",
        "arms": ["curriculum_1_2_4", "control_lead6"],
        "model": _common_config(channels),
        "parameters": measured["parameters"],
        "forward_flops": measured["forward_flops"],
        "forward_backward_flops": measured["forward_backward_flops"],
        "shared_controls": {
            "optimizer": f"AdamW lr {LR} weight_decay {WEIGHT_DECAY}",
            "lr_schedule": (f"linear warmup over {WARMUP_UPDATES} updates then cosine to "
                            f"{MINIMUM_LR_RATIO} of peak, restarted per curriculum stage"),
            "max_updates": TOTAL_UPDATES,
            "batch_size": BATCH_SIZE,
            "clip": CLIP,
            "reasoning_steps": REASONING_STEPS,
            "process_weight": PROCESS_WEIGHT,
            "validation_every": VALIDATION_EVERY,
            "validation_lead_hours": list(VALIDATION_LEADS),
            "selection_split": "val only; test is never read",
            "checkpoint_selection_rule": ("lowest mean latitude-weighted normalized "
                                          "validation MSE over the whole run"),
            "evaluation_leads_hours": list(EVALUATION_LEADS),
            "sample_order": "torch.randperm(len(dataset), generator=manual_seed(seed+epoch))",
        },
        "decision_rule": {
            "primary_variable": "t2m",
            "statement": ("the curriculum is described as better only where it improves the "
                          "primary variable with the same sign in every declared seed; "
                          "every variable and lead is still reported"),
        },
        "flop_convention": FLOP_CONVENTION,
        "scientific_claim": SCIENTIFIC_CLAIM,
        "limitations": LIMITATIONS,
    }
    from training.r7_experiment import canonical_digest

    return dict(body, protocol_sha256=canonical_digest(body))


def _write_json_exclusive(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)


def run_curriculum(store, output_dir, *, seeds=SEEDS, device_name="cuda",
                   total_updates=TOTAL_UPDATES):
    from data.r7_evaluation import ZarrRolloutDataset
    from torch.utils.data import default_collate
    from training.r7_evaluate import evaluate_local
    from training.r7_experiment import (make_model, model_code_digest, seed_everything,
                                        select_device)
    from training.r7_scheduled_runner import run_scheduled_updates

    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing existing output: {output_dir}")
    # The output directory is created before any manifest is written so that a
    # partial run leaves its own trail rather than scattering files next to an
    # absent run directory.
    output_dir.mkdir(parents=True, exist_ok=False)
    device = select_device(device_name)

    # ---- build the per-lead train manifests from the store, read-only --------
    manifest_dir = output_dir / "manifests"
    manifests = {}
    window_counts = {}
    datasets = {}
    control_records = None
    for lead in sorted({lead for lead, _ in CURRICULUM_STAGES} | {CONTROL_LEAD_HOURS}):
        records = build_lead_manifest(store, "train", lead)
        path = write_manifest(records, manifest_dir / f"train_p{lead:03d}h.jsonl")
        manifests[lead] = path
        window_counts[lead] = len(records)
        datasets[lead] = LeadManifestDataset(store, records)
        if lead == CONTROL_LEAD_HOURS:
            control_records = records
    publish_manifest_build(manifest_dir, manifests, window_counts)
    # ``evaluate_local`` derives the training identity from ``train.jsonl`` beside
    # the val manifest, so the control's +6h window list is also published under
    # that canonical name. The content is identical to train_p006h.jsonl; the
    # duplicate exists because that reader convention is fixed and the
    # curriculum's per-lead names are not.
    val_manifest = manifest_dir / "val_p006h.jsonl"
    write_manifest(build_lead_manifest(store, "val", CONTROL_LEAD_HOURS), val_manifest)
    write_manifest(control_records, manifest_dir / "train.jsonl")
    channels = datasets[CONTROL_LEAD_HOURS].channels
    # ``identity`` must be the value the evaluator will independently derive from
    # the canonical manifest, or its checkpoint/identity check refuses the pairing.
    # The per-lead manifests are therefore recorded as separate provenance rather
    # than folded into the identity: two different digests for one fact would make
    # the readers disagree.
    from training.r7_experiment import dataset_identity

    identity, _ = dataset_identity(manifest_dir / "train.jsonl")
    lead_manifest_sha256 = {str(lead): _sha256_file(path)
                            for lead, path in sorted(manifests.items())}

    validation_dataset = ZarrRolloutDataset(
        store, split="val", lead_hours=VALIDATION_LEADS, history_steps=2, step_hours=6)

    seed_everything(SEEDS[0])
    model = make_model("process", _common_config(len(channels)))
    probe_batch = default_collate([datasets[CONTROL_LEAD_HOURS][0],
                                   datasets[CONTROL_LEAD_HOURS][1]])
    measured = {"parameters": int(sum(p.numel() for p in model.parameters()))}
    fwd, fwd_bwd = _count_flops(model, probe_batch, REASONING_STEPS)
    measured.update(forward_flops=fwd, forward_backward_flops=fwd_bwd,
                    model_code_sha256=model_code_digest())
    del model

    stage_counts = stage_updates(total_updates)
    # The frozen warmup and validation cadence apply to the declared budget. A
    # shortened smoke run keeps the same shape with values that fit inside a
    # single stage, so it exercises the plumbing instead of silently running a
    # different protocol. The cadence must allow at least two checks, or the
    # runner's "validation ran but selected nothing" guard fires correctly and
    # the smoke run would be testing the guard rather than the harness.
    if total_updates != TOTAL_UPDATES:
        smallest_stage = min(count for count in stage_counts.values() if count > 0)
        smoke_warmup = max(1, min(WARMUP_UPDATES, smallest_stage // 10))
        smoke_validation = max(1, min(VALIDATION_EVERY, smallest_stage // 2))
        # A smoke run must reach its declared endpoint; early stopping is part of
        # the declared protocol, not part of checking that the plumbing works,
        # and leaving it on would make the run's short stage an expected outcome
        # that the budget guard below then reports as a failure.
        smoke_patience = None
    else:
        smoke_warmup = WARMUP_UPDATES
        smoke_validation = VALIDATION_EVERY
        smoke_patience = EARLY_STOPPING_PATIENCE
    protocol = protocol_payload(manifests, identity, len(channels), measured,
                                stage_counts, lead_manifest_sha256)
    _write_json_exclusive(output_dir / "protocol.json", protocol)

    results = {
        "format": "r7-64-curriculum-result-v1",
        "scientific_claim": SCIENTIFIC_CLAIM, "test_read": False,
        "gpu_used": device.type == "cuda",
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__), "platform": platform.platform(),
        "device": device_name, "seeds": list(seeds),
        "data_identity": identity, "model_code_sha256": model_code_digest(),
        "protocol_sha256": protocol["protocol_sha256"],
        "stage_updates": stage_counts, "flop_measurements": measured,
        "training": {}, "evaluation": {},
    }

    started = time.perf_counter()
    for seed in seeds:
        per_seed = {}
        for arm in ("control_lead6", "curriculum_1_2_4"):
            run_dir = output_dir / "training" / f"seed{seed}" / arm
            carried = None
            stages = (((CONTROL_LEAD_HOURS, total_updates),) if arm == "control_lead6"
                      else tuple((lead, stage_counts[lead])
                                 for lead, _ in CURRICULUM_STAGES))
            stage_reports = []
            for stage_index, (lead, count) in enumerate(stages):
                if count <= 0:
                    continue
                stage_dir = run_dir / f"stage{stage_index}_p{lead:03d}h"
                # Each stage is a fresh run of ``count`` updates. ``resume`` is the
                # wrong mechanism across stages: the runner binds ``dataset_length``
                # into the checkpoint signature, and the per-lead window counts
                # differ (94/93/91), so a resume would be refused - correctly, since
                # the resumed run would not be the same data contract. The learned
                # weights are carried over instead, which is what a curriculum
                # actually transfers, and the per-stage schedule restarts by design.
                checkpoint, report = run_scheduled_updates(
                    datasets[lead], kind="process",
                    model_config=_common_config(len(channels)),
                    data_identity=identity, output_dir=stage_dir,
                    total_updates=count, batch_size=BATCH_SIZE,
                    steps=REASONING_STEPS, seed=seed, lr=LR, clip=CLIP,
                    process_weight=PROCESS_WEIGHT, warmup_updates=smoke_warmup,
                    minimum_lr_ratio=MINIMUM_LR_RATIO,
                    validation_every=smoke_validation,
                    early_stopping_patience=smoke_patience,
                    minimum_improvement=MINIMUM_IMPROVEMENT,
                    validation_lead_hours=VALIDATION_LEADS, device_name=device_name,
                    validation_dataset=validation_dataset,
                    shared_initial_state=carried)
                if report["updates_this_run"] != count:
                    raise RuntimeError(
                        f"{arm} stage {stage_index} ran {report['updates_this_run']} "
                        f"updates, expected {count}; a short stage would silently "
                        "change the declared budget")
                carried = _trainable_state(checkpoint)
                stage_reports.append({
                    "stage": stage_index, "lead_hours": lead,
                    "updates": count, "selected_update": report["selected_update"],
                    "selected_validation_mse": report["selected_validation_mse"],
                    "early_stopped": report["early_stopped"],
                    "stopped_reason": report["stopped_reason"],
                    "elapsed_seconds": report["elapsed_seconds"],
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": _sha256_file(checkpoint),
                    "shared_initial_state": {
                        "provided": report["shared_initial_state"]["provided"],
                        "applied_count": report["shared_initial_state"]["applied_count"],
                        "ignored_count": report["shared_initial_state"]["ignored_count"],
                    },
                    "final_update_loss": report["losses"][-1]["loss"] if report["losses"] else None,
                })
                print(json.dumps({"phase": "curriculum", "arm": arm, "seed": seed,
                                  "stage": stage_index, "lead_hours": lead,
                                  "updates": count,
                                  "seconds": round(report["elapsed_seconds"], 1)}), flush=True)
            if not stage_reports:
                raise RuntimeError(f"{arm} produced no checkpoint")
            final = stage_reports[-1]
            # Per-stage last-update loss, kept separate from a whole-run curve:
            # the stages optimise different target sets, so one epoch series
            # across them would not be a single learning curve.
            stage_losses = {str(index): stage["final_update_loss"]
                            for index, stage in enumerate(stage_reports)}
            per_seed[arm] = {
                "stages": stage_reports,
                "final_checkpoint": final["checkpoint"],
                "final_checkpoint_sha256": final["checkpoint_sha256"],
                "total_elapsed_seconds": sum(s["elapsed_seconds"] for s in stage_reports),
                "updates_total": sum(s["updates"] for s in stage_reports),
                "parameters": measured["parameters"],
                "forward_flops": measured["forward_flops"],
                "forward_backward_flops": measured["forward_backward_flops"],
                "stage_losses": stage_losses,
            }
        results["training"][str(seed)] = per_seed
    results["budget"] = {"training_seconds_total": time.perf_counter() - started}

    # ---- evaluation: val only, free rollout at every lead -------------------
    for seed in seeds:
        for arm in ("control_lead6", "curriculum_1_2_4"):
            for lead in EVALUATION_LEADS:
                run_dir = output_dir / "evaluation" / f"seed{seed}" / arm / f"lead_{lead:03d}h"
                source = results["training"][str(seed)][arm]["final_checkpoint"]
                report = evaluate_local(
                    val_manifest, output_dir=run_dir,
                    checkpoint=source, lead_hours=(lead,), max_samples=64,
                    device_name=device_name, reasoning_steps=REASONING_STEPS)
                if report["split"] != "val":
                    raise ValueError(f"{arm} was evaluated on split {report['split']}")
                results["evaluation"][f"seed{seed}/{arm}@{lead}h"] = {
                    "seed": seed, "arm": arm, "lead_hours": lead,
                    "n_evaluated": report["n_evaluated"],
                    "channels": list(report["channels"]), "units": list(report["units"]),
                    "elapsed_seconds": report["elapsed_seconds"],
                    "timing_scope": report["timing_scope"],
                    "rmse_csv": str(run_dir / "rmse.csv"),
                    "evaluation_dir": str(run_dir),
                }
        print(json.dumps({"phase": "curriculum", "evaluated_seed": seed}), flush=True)

    for lead in EVALUATION_LEADS:
        for seed in seeds:
            counts = {arm: results["evaluation"][f"seed{seed}/{arm}@{lead}h"]["n_evaluated"]
                      for arm in ("control_lead6", "curriculum_1_2_4")}
            if len(set(counts.values())) != 1:
                raise RuntimeError(f"arms scored on different case counts at {lead}h "
                                   f"seed {seed}: {counts}")
            results.setdefault("case_counts_by_lead", {}).setdefault(
                f"{lead}h", {})[str(seed)] = counts["control_lead6"]

    _write_json_exclusive(output_dir / "curriculum_result.json", results)
    _write_tables(results, output_dir)
    return results


def _trainable_state(checkpoint):
    """Load a published checkpoint's weights for carrying into the next stage."""
    from training.r7_experiment import load_checkpoint

    saved = load_checkpoint(checkpoint)
    return {name: tensor.detach().cpu().clone()
            for name, tensor in saved["model"].items()}


def _count_flops(model, batch, reasoning_steps):
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
            out = model(inputs, reasoning_steps=reasoning_steps)
            loss = out.forecast.square().mean()
            if out.process_predictions.shape[1] > 0:
                loss = loss + out.process_predictions.square().mean()
            loss.backward()
        forward_backward = int(counter.get_total_flops())
    model.zero_grad()
    return forward, forward_backward


def _write_tables(results, output_dir):
    parameter_rows = [{"arm": arm, "parameters": results["training"]["41"][arm]["parameters"],
                       "forward_flops": results["training"]["41"][arm]["forward_flops"],
                       "forward_backward_flops":
                           results["training"]["41"][arm]["forward_backward_flops"]}
                      for arm in ("control_lead6", "curriculum_1_2_4")]
    with (output_dir / "cost_table.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(parameter_rows[0]))
        writer.writeheader()
        writer.writerows(parameter_rows)

    wall_rows = []
    for seed, per_arm in results["training"].items():
        for arm, entry in per_arm.items():
            wall_rows.append({"seed": seed, "arm": arm,
                              "wall_time_seconds": entry["total_elapsed_seconds"],
                              "updates_total": entry["updates_total"],
                              "stages": len(entry["stages"])})
    with (output_dir / "wall_time_table.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(wall_rows[0]))
        writer.writeheader()
        writer.writerows(wall_rows)

    rmse_rows = []
    for entry in results["evaluation"].values():
        with Path(entry["rmse_csv"]).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rmse_rows.append({"seed": entry["seed"], "arm": entry["arm"],
                                  "evaluation_lead_hours": entry["lead_hours"],
                                  "variable": row["variable"], "unit": row["unit"],
                                  "rmse": row["rmse"]})
    with (output_dir / "rmse_table.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rmse_rows[0]))
        writer.writeheader()
        writer.writerows(rmse_rows)


def main():
    parser = argparse.ArgumentParser(description="#64 lead-time curriculum (bounded).")
    parser.add_argument("--store", default=DEFAULT_STORE)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--updates", type=int, default=TOTAL_UPDATES)
    args = parser.parse_args()
    results = run_curriculum(args.store, args.out, seeds=tuple(args.seeds),
                             device_name=args.device, total_updates=args.updates)
    print(json.dumps({"complete": True,
                      "protocol_sha256": results["protocol_sha256"],
                      "training_seconds": results["budget"]["training_seconds_total"]}))


if __name__ == "__main__":
    main()
