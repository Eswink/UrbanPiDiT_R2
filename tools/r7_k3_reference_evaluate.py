"""Companion evaluator: unchanged archived checkpoint/model/input/rollout, one trajectory per case."""
from __future__ import annotations

from pathlib import Path
import time

from r7_k3_reference_cases import pool_cases, score_case, validate_evaluation
from r7_k3_reference_support import DATA_SHA, MODEL_SHA, check_deadline, require


def verify_saved(saved, parent):
    require(saved["contract"] == parent["contract"] and saved["signature"] == parent["signature"]
            and saved["model_code_sha256"] == MODEL_SHA and saved["updates"] == 400,
            "original strict checkpoint body must match qualified genuine K3 signature/model/update")
    state = saved["model"]
    require(len(state) == 131 and sum(t.numel() for t in state.values()) == 2968259,
            "genuine original131 tensors/2968259 entries required")
    import torch
    require(all(torch.isfinite(t).all() for t in state.values()), "nonfinite original checkpoint tensors")


def metadata_preflight(protocol, lead):
    """Only old dataset metadata and train/val manifests; no weather __getitem__ here."""
    from data.r7_store import validate_record
    from data.r7_evaluation import ZarrRolloutDataset
    from training.r7_experiment import dataset_identity
    manifest = Path(protocol["repo"]) / "outputs/r7_m2_segment/store/manifests/val.jsonl"
    train_identity, _ = dataset_identity(manifest.parent / "train.jsonl")
    val_identity, reader = dataset_identity(manifest)
    require(train_identity == DATA_SHA == protocol["data"]["data_identity"] and val_identity == protocol["data"]["val_data_identity"],
            "old dataset/normalization train-val identity differs")
    require(all(r["split"] == "val" and len(r["history_indices"]) == 2 and r["lead_time_hours"] == 6 for r in reader.records),
            "two-history+6h validation-only manifest required")
    stores = set()
    for rec in reader.records:
        root = reader._store(rec)
        validate_record(root, rec)
        path = Path(rec["store_path"])
        stores.add(str(path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()))
    require(stores == {protocol["data"]["store"]}, "old manifest must refer to frozen existing M2 store")
    ds = ZarrRolloutDataset(protocol["data"]["store"], split="val", lead_hours=(lead,), history_steps=2, step_hours=6)
    allowed = {r["init_time"] for r in reader.records}
    ds.windows = [w for w in ds.windows if ds.times[w[0][-1]].isoformat() in allowed]
    declared = protocol["data"]["evaluation_cases"][str(lead)]
    actual = [[ds.times[history[-1]].isoformat(), [ds.times[i].isoformat() for i in targets]] for history, targets in ds.windows]
    require(actual == declared["cases"] and len(ds) == declared["n_available"] <= protocol["max_samples"], "single-lead fixed case metadata differs")
    require(list(ds.names) == protocol["data"]["channels"] and list(ds.units) == protocol["data"]["units"]
            and ds.mean.tolist() == protocol["data"]["normalization_mean"] and ds.std.tolist() == protocol["data"]["normalization_std"],
            "unchanged17-channel train-only normalization/units required")
    return ds


def evaluate_cases(model, ds, climatology, *, seed, lead, device, deadline):
    """No intermediate truth input, no new Gregorian/calendar producers, no region refowards."""
    import torch
    from data.r7_evaluation import normalized_climatology
    from model.r7_rollout import autoregressive_rollout, rollout_model_input
    cases = []
    with torch.no_grad():
        for index in range(len(ds)):
            check_deadline(deadline)
            sample = ds[index]
            climate = normalized_climatology(climatology, sample["valid_times"], ds.mean, ds.std).unsqueeze(0)
            initial = rollout_model_input(sample, lead_hours=float(sample["lead_time_hours"]), device=device)
            trajectory = autoregressive_rollout(model, initial, lead_hours=(lead,), step_hours=6, history_interval_hours=6,
                                                inference_kwargs={"reasoning_steps": 3})
            prediction = trajectory.forecasts.cpu()
            steps = trajectory.cumulative_reasoning_steps.cpu().tolist()[0]
            require(trajectory.model_calls == lead // 6 and steps == [3 * lead // 6], "oldK3 full rollout work required")
            cases.append(score_case(sample, prediction, climate, lead=lead, names=ds.names, units=ds.units, std=ds.std,
                                    cumulative_steps=steps, model_calls=trajectory.model_calls))
            check_deadline(deadline)
    return cases


def evaluate_job(protocol, job, deadline):
    import torch
    from data.r7_evaluation import fit_training_climatology
    from training.r7_experiment import load_checkpoint, make_model, model_code_digest
    check_deadline(deadline)
    started = time.perf_counter()
    parent = protocol["parents"][str(job["seed"])]
    # ORIGINAL loader, unmodified. No wrapper, patched digest, M3 hardpin or unsafe deserialization.
    saved = load_checkpoint(parent["checkpoint"], expected=parent["signature"])
    verify_saved(saved, parent)
    require(model_code_digest() == MODEL_SHA, "runtime archive model digest differs")
    model = make_model(saved["contract"]["kind"], saved["contract"]["model"])
    model.load_state_dict(saved["model"], strict=True)
    del saved
    check_deadline(deadline)
    ds = metadata_preflight(protocol, job["lead"])
    check_deadline(deadline)
    climatology = fit_training_climatology(protocol["data"]["store"])
    require(climatology["selection"] == "declared_train_time_ranges" and climatology["n_selected_steps"] == 188
            and climatology["channels"] == protocol["data"]["channels"] and len(climatology["counts"]) == 8,
            "old train-only8-bucket climatology required; no held-out fallback")
    device = torch.device("cuda:0")
    model = model.to(device).eval()
    cases = evaluate_cases(model, ds, climatology, seed=job["seed"], lead=job["lead"], device=device, deadline=deadline)
    torch.cuda.synchronize(device)
    check_deadline(deadline)
    report = {"scientific_claim": False, "limitations": protocol["limitations"], "test_read": False, "split": "val",
              "seed": job["seed"], "lead_hours": [job["lead"]], "channels": list(ds.names), "units": list(ds.units),
              "n_evaluated": len(cases), "n_available_windows": len(ds), "cases": cases,
              "checkpoint": parent["checkpoint"], "checkpoint_sha256": parent["checkpoint_sha256"],
              "signature": parent["signature"], "model_code_sha256": MODEL_SHA,
              "training_identity": DATA_SHA, "calendar": protocol["calendar"],
              "climatology": {k: v for k, v in climatology.items() if k not in ("means", "counts")},
              "climatology_bucket_counts": {f"{m:02d}-{h:02d}": n for (m, h), n in climatology["counts"].items()},
              "elapsed_seconds": time.perf_counter() - started,
              "timing_scope": "whole archived loader/model/data/train-climatology/cases/CPU-statistics loop; not isolated forward latency",
              "extra_model_forward_for_regions_or_baselines": 0,
              "rows": pool_cases(cases, seed=job["seed"], lead=job["lead"], names=ds.names, units=ds.units)}
    validate_evaluation(report, protocol, job)
    return report
