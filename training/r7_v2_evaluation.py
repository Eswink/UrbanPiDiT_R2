"""Validation-only, single-lead/K4-checkpoint probes for the independent V2 route."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import pandas as pd
import torch

from data.r7_evaluation import ZarrRolloutDataset, fit_training_climatology, normalized_climatology
from data.r7_store import validate_record
from data.r7_zarr_dataset import ZarrAtmosWindowDataset
from model.r7_rollout import autoregressive_rollout, rollout_model_input, validate_horizons
from .r7_acc import RolloutACCAccumulator
from .r7_boundary_metrics import boundary_masks
from .r7_climatology_skill import RolloutClimatologySkillAccumulator, verify_acc_skill_consistency
from .r7_experiment import canonical_digest, dataset_identity, load_checkpoint, make_model, select_device
from .r7_process_training_contract import verify_evaluation_sidecar
from .r7_rollout_metrics import RolloutRMSEAccumulator

ALL_LEADS = (6, 12, 24, 48, 72)
PROBE_STEPS = (1, 2, 4)
STEP_HOURS = 6
LIMITATIONS = [
    "Validation only; no test observations or test-based selection; not a scientific success verdict.",
    "Monthly/hourly train-only climatology is not a WeatherBench2 reproduction.",
    "K1/K2 are inference-depth probes of the SAME K4-trained checkpoint, not independently trained models.",
    "Each lead retains its own complete validation cases; pair same-lead arms/K only, never average across leads.",
    "Boundary scoring stratifies the same forecasts; no boundary forcing or causal mechanism claim.",
    "No cross-variable aggregate; undefined anomaly energy is null with a status, never fabricated zero skill.",
    "One resident input timing is not a weather-distribution benchmark; CUDA co-residency can affect latency.",
    "Training-loss forward/backward FLOPs belong to separately pinned CPU preparation; not measured here.",
    "No bitwise cross-hardware reproducibility claim; identities/options/cases support numerical replay.",
]


def _check_deadline(deadline):
    if deadline is not None:
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
            raise ValueError("deadline must be a finite absolute perf_counter time")
        if time.perf_counter() >= deadline:
            raise TimeoutError("V2 evaluation deadline exceeded")


def _store_path(manifest, record):
    path = Path(record["store_path"])
    return (manifest.parent / path).resolve() if not path.is_absolute() else path.resolve()


def _validation_data(manifest, lead):
    reader = ZarrAtmosWindowDataset(manifest)
    stores = {_store_path(manifest, r) for r in reader.records}
    if {r["split"] for r in reader.records} != {"val"} or len(stores) != 1:
        raise ValueError("V2 requires one held-out val split and one store; test is forbidden")
    root = reader._store(reader.records[0])
    for record in reader.records:
        validate_record(root, record)
    history_lengths = {len(r["history_indices"]) for r in reader.records}
    if len(history_lengths) != 1 or any(r["lead_time_hours"] != STEP_HOURS for r in reader.records):
        raise ValueError("V2 manifest requires uniform history lengths and six-hour transition records")
    for r in reader.records:
        times = pd.DatetimeIndex(r["history_times"])
        if len(times) > 1 and not (times[1:] - times[:-1] == pd.Timedelta(hours=STEP_HOURS)).all():
            raise ValueError("V2 history cadence must be six hours")
    records = {pd.Timestamp(r["init_time"]).isoformat(): r for r in reader.records}
    if len(records) != len(reader.records):
        raise ValueError("duplicate initialization cases; one manifest record per initialization required")
    store = next(iter(stores))
    ds = ZarrRolloutDataset(store, split="val", lead_hours=(lead,),
                            history_steps=next(iter(history_lengths)), step_hours=STEP_HOURS)
    ds.windows = [window for window in ds.windows if ds.times[window[0][-1]].isoformat() in records]
    if not ds.windows:
        raise ValueError("manifest has no complete requested-lead validation cases")
    if (len(ds.names) != 17 or len(set(ds.names)) != 17
            or any(not isinstance(v, str) or not v for v in ds.names)):
        raise ValueError("V2 requires all 17 ordered unique variable names")
    if len(ds.units) != 17 or any(not isinstance(u, str) or not u.strip()
                                  or u.strip().lower() in ("unknown", "normalized") for u in ds.units):
        raise ValueError("V2 requires explicit physical units for all 17 variables")
    # Same-lead arm/K pairing is the contract; never narrow short leads to a 72h cohort.
    boundary_masks(int(root["state"].shape[-2]), int(root["state"].shape[-1]), (2,))
    return ds, root, store, records


def _checkpoint_model(manifest, checkpoint, root, store, sidecar):
    saved = load_checkpoint(checkpoint)  # Never weaken current-code/contract digest verification.
    contract = saved["contract"]
    if contract["kind"] not in ("generic", "process") or type(contract.get("steps")) is not int or contract["steps"] != 4:
        raise ValueError("V2 K1/2/4 probes require a recursive K4-trained checkpoint, not an old K3/native parent")
    training_identity, reader = dataset_identity(manifest.parent / "train.jsonl")
    if training_identity != contract["data_identity"]:
        raise ValueError("checkpoint training data/normalization identity mismatch")
    if ({r["split"] for r in reader.records} != {"train"}
            or {_store_path(reader.manifest, r) for r in reader.records} != {store}):
        raise ValueError("evaluation must use the same store as the train-only checkpoint manifest")
    sidecar_identity = verify_evaluation_sidecar(
        contract, sidecar, evaluation_store=store, evaluation_root=root)
    model = make_model(contract["kind"], contract["model"])
    intervention = contract.get("intervention")
    if intervention is not None:
        from .r7_frozen_z_intervention import install_frozen_z, validate_frozen_z
        install_frozen_z(model, intervention)
    model.load_state_dict(saved["model"], strict=True)
    if intervention is not None:
        validate_frozen_z(model, intervention)
    return model, saved, training_identity, sidecar_identity


class RegionMetrics:
    """Existing accumulators on genuinely selected cells, with identical area/case weights.

    Selected fields become [B,L,C,N,1] and their original latitudes become [N].
    This represents an irregular edge exactly; excluded cells never contribute
    fictitious zero anomalies/energy or dilute the area-normalized MSE.
    """
    def __init__(self, lead_hours, variables, *, training_std, units):
        self.lead_hours, self.variables = tuple(lead_hours), tuple(variables)
        self.metrics = {}
        for region in ("full", "interior", "edge_2"):
            self.metrics[region] = (
                RolloutRMSEAccumulator(lead_hours, variables, training_std=training_std, units=units),
                RolloutACCAccumulator(lead_hours, variables),
                RolloutClimatologySkillAccumulator(lead_hours, variables, training_std=training_std, units=units),
            )
        self.latitude = self.width = self.regions = None

    def update(self, prediction, target, climatology, latitude):
        if (prediction.ndim != 5 or prediction.shape != target.shape or prediction.shape != climatology.shape
                or min(prediction.shape) < 1):
            raise ValueError("equal nonempty [B,L,C,H,W] region fields required")
        b, l, c, h, w = prediction.shape
        if (l, c) != (len(self.lead_hours), len(self.variables)) or not (prediction.device == target.device == climatology.device):
            raise ValueError("region horizon/channel/device mismatch")
        if not all(x.is_floating_point() and torch.isfinite(x).all() for x in (prediction, target, climatology)):
            raise ValueError("finite floating region fields required; no bad variables/cases dropped")
        lat = torch.as_tensor(latitude, dtype=torch.float64).detach().cpu()
        if lat.shape == (h,):
            lat = lat[None, :].expand(b, h)
        if (lat.shape != (b, h) or not torch.isfinite(lat).all() or (lat.abs() > 90).any()
                or not torch.equal(lat, lat[0:1].expand(b, h))):
            raise ValueError("one finite immutable latitude grid [H] or [B,H] required")
        axis = lat[0]
        if h > 1 and not ((axis.diff() > 0).all() or (axis.diff() < 0).all()):
            raise ValueError("region latitude must be strictly monotone")
        if self.latitude is not None and (self.width != w or not torch.equal(self.latitude, axis)):
            raise ValueError("region grid changed between initialization cases")
        regions = boundary_masks(h, w, (2,))
        area_weights = torch.cos(torch.deg2rad(axis)).clamp_min(0)[:, None].expand(h, w)
        areas = [area_weights[mask].sum() for _, _, mask in regions]
        if any(area < 1e-12 for area in areas):
            raise ValueError("empty or zero usable area in scoring region")
        definitions = []
        for (original, margin, mask), area in zip(regions, areas):
            name = "interior" if original == "interior_2" else original
            select = mask.flatten().to(prediction.device)
            fields = [x.flatten(-2)[..., select].unsqueeze(-1) for x in (prediction, target, climatology)]
            selected_latitude = axis[:, None].expand(h, w)[mask]
            rmse, acc, skill = self.metrics[name]
            rmse.update(fields[0], fields[1], selected_latitude)
            acc.update(*fields, selected_latitude)
            skill.update(*fields, selected_latitude)
            definitions.append(dict(region=name, margin_cells=margin, n_grid_points=int(mask.sum()),
                                    full_area_fraction=float(area / areas[0])))
        self.latitude, self.width, self.regions = axis.clone(), w, definitions

    def rows(self):
        result = []
        for definition in self.regions or []:
            rmse, acc, skill = self.metrics[definition["region"]]
            if rmse.initializations != acc.initializations or rmse.initializations != skill.initializations:
                raise ValueError("region metrics must score exactly the same initialization cases")
            if tuple(rmse.units) != tuple(skill.units):
                raise ValueError("region forecast/climatology metric units differ")
            errors, correlations, skills = rmse.compute(), acc.compute(), skill.compute()
            if not all(torch.isfinite(x).all() for x in (errors, skills["rmse_climatology"],
                                                         acc.dot, acc.forecast_energy, acc.target_energy)):
                raise ValueError("region sufficient-statistic overflow")
            if not torch.allclose(errors, skills["rmse_forecast"], rtol=1e-12, atol=1e-12):
                raise ValueError("region forecast and skill MSE differ")
            usable_acc = (acc.forecast_energy > 0) & (acc.target_energy > 0)
            usable_skill = skill.climatology_squared_error > 0
            if ((usable_acc & ~torch.isfinite(correlations)).any()
                    or (usable_skill & ~torch.isfinite(skills["mse_skill"])).any()):
                raise ValueError("nonfinite metric with nonzero anomaly energy; not an undefined-zero case")
            identity = verify_acc_skill_consistency(correlations, skills["mse_skill"])
            if not identity["consistent"]:
                raise ValueError("region ACC/MSE-skill same-mask identity violated")
            for i, lead in enumerate(self.lead_hours):
                for j, variable in enumerate(self.variables):
                    corr, value = correlations[i, j], skills["mse_skill"][i, j]
                    corr_defined, skill_defined = bool(torch.isfinite(corr)), bool(torch.isfinite(value))
                    row = dict(**definition, lead_hours=lead, variable=variable, unit=rmse.units[j],
                               rmse=float(errors[i, j]), mse=float(errors[i, j].square()),
                               rmse_climatology=float(skills["rmse_climatology"][i, j]),
                               climatology_mse=float(skills["rmse_climatology"][i, j].square()),
                               mse_climatology=float(skills["rmse_climatology"][i, j].square()),
                               acc_dot=float(acc.dot[i, j]),
                               acc_forecast_energy=float(acc.forecast_energy[i, j]),
                               acc_target_energy=float(acc.target_energy[i, j]),
                               acc_statistic_units="normalized_anomaly_squared",
                               mse_skill=float(value) if skill_defined else None,
                               acc=float(corr) if corr_defined else None,
                               acc_status="defined" if corr_defined else "undefined_zero_anomaly_energy",
                               skill_status="defined" if skill_defined else "undefined_zero_climatology_energy",
                               n_initializations=rmse.initializations)
                    row["bad_reasons"] = _bad_reasons(row)
                    result.append(row)
        if not result:
            raise RuntimeError("no initialization samples in region metrics")
        return result


def _bad_reasons(row):
    return [name for name, bad in (
        ("negative_mse_skill", row["mse_skill"] is not None and row["mse_skill"] < 0),
        ("negative_acc", row["acc"] is not None and row["acc"] < 0),
        ("undefined_mse_skill", row["mse_skill"] is None),
        ("undefined_acc", row["acc"] is None),
    ) if bad]


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory_snapshot(device):
    if device.type != "cuda":
        return dict(allocated_bytes=None, reserved_bytes=None, peak_allocated_bytes=None, peak_reserved_bytes=None)
    return dict(allocated_bytes=torch.cuda.memory_allocated(device), reserved_bytes=torch.cuda.memory_reserved(device),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_reserved_bytes=torch.cuda.max_memory_reserved(device))


def _tensor_digest(items):
    digest = hashlib.sha256()
    for name, tensor in sorted(items):
        value = tensor.detach().cpu().contiguous()
        digest.update(str((name, tuple(value.shape), str(value.dtype))).encode() + b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _profile_forward(model, inputs, reasoning_steps, deadline):
    """Actual resident model forward, not rollout/metrics; no allocator reset/clearing."""
    if any(module.training for module in model.modules()):
        raise ValueError("isolated profiling requires model.eval()")
    history = inputs["coarse_history"]
    device = history.device
    expected = (history.shape[0], history.shape[2], history.shape[3], history.shape[4])
    model_before, inputs_before = _tensor_digest(model.state_dict().items()), _tensor_digest(inputs.items())

    def checked(output):
        if output.forecast.shape != expected or output.forecast.device != device or not torch.isfinite(output.forecast).all():
            raise ValueError("isolated model forward produced invalid/nonfinite full-state forecast")
        counts = getattr(output, "reasoning_steps_per_sample", None)
        if counts is None:
            depth = getattr(output, "reasoning_steps", None)
            if type(depth) is not int or depth != reasoning_steps:
                raise ValueError("isolated forward actual K differs from requested probe")
            return [depth] * expected[0]
        if counts.shape != (expected[0],) or counts.dtype != torch.long or not (counts == reasoning_steps).all():
            raise ValueError("isolated forward actual K differs from requested probe")
        return counts.detach().cpu().tolist()

    started = time.perf_counter()
    for _ in range(3):
        _check_deadline(deadline)
        output = model(inputs, reasoning_steps=reasoning_steps)
        _synchronize(device)
        checked(output)
        del output
    durations, counts = [], []
    for _ in range(10):
        _check_deadline(deadline)
        _synchronize(device)
        stamp = time.perf_counter()
        output = model(inputs, reasoning_steps=reasoning_steps)
        _synchronize(device)
        durations.append(time.perf_counter() - stamp)
        counts.append(checked(output))  # Validation/CPU transfer excluded from timed interval.
        del output
    _check_deadline(deadline)
    if model_before != _tensor_digest(model.state_dict().items()) or inputs_before != _tensor_digest(inputs.items()):
        raise ValueError("model or resident inputs mutated during isolated profiling")
    return dict(device=str(device), warmup_excluded=3, repetitions=10, reasoning_steps=reasoning_steps,
                seconds_per_batch=durations, median_seconds_per_batch=statistics.median(durations),
                mean_seconds_per_batch=statistics.mean(durations), actual_reasoning_steps_per_sample=counts,
                input_shape=list(history.shape), input_sha256=inputs_before, model_state_sha256=model_before,
                elapsed_seconds=time.perf_counter() - started,
                scope="resident-batch model K forward plus synchronization; excludes IO/transfers/metrics/validation",
                cuda_memory=_memory_snapshot(device), gpu_latency_measured=device.type == "cuda")


def _reference_forecasts(sample, climate):
    """Zero-trained controls; only known last history and frozen train-only climate.

    Baseline construction never inspects targets or other future-bearing sample
    keys, runs no model, and reuses the exact fields subsequently scored below.
    """
    history = sample["coarse_history"]
    if (history.ndim != 4 or min(history.shape) < 1 or climate.ndim != 5
            or tuple(history.shape[1:]) != tuple(climate.shape[2:])
            or history.device != climate.device or not history.is_floating_point()
            or not torch.isfinite(history).all()):
        raise ValueError("baseline requires finite known [T,C,H,W] history matching the climate grid/device")
    return {"persistence": history[-1][None, None].expand_as(climate), "climatology": climate}


def _reference_rows(metrics):
    return [dict(baseline_kind=kind, zero_train_updates=0, parameters=0, trainable_parameters=0, **row)
            for kind, accumulator in metrics.items() for row in accumulator.rows()]


def _csv_row(row):
    return {key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
            for key, value in row.items()}


def _write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(_csv_row(row) for row in rows)


@torch.no_grad()
def evaluate_v2(manifest, checkpoint, output_dir, *, lead_hours=(6,), reasoning_steps=4,
                max_samples=32, device_name="cuda:0", deadline=None, process_scale_sidecar=None):
    """One fresh worker's single lead/K probe; returns provenance and complete metric rows.

    The worker owns pre-init 0/0 checks and peak reset before any CUDA allocations.
    This helper observes actual allocator state only; errors propagate to the driver
    and already-written per-case CSV evidence is retained without accepting a result.
    """
    whole_started = time.perf_counter()
    _check_deadline(deadline)
    leads = validate_horizons(lead_hours, STEP_HOURS)
    if len(leads) != 1 or leads[0] not in ALL_LEADS:
        raise ValueError("V2 worker requires one lead from 6/12/24/48/72")
    if type(reasoning_steps) is not int or reasoning_steps not in PROBE_STEPS:
        raise ValueError("one reasoning_steps probe from K1/2/4 required")
    if type(max_samples) is not int or max_samples < 1:
        raise ValueError("max_samples must be a positive explicit cap")
    manifest, checkpoint, out = Path(manifest), Path(checkpoint), Path(output_dir)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    ds, root, store, records = _validation_data(manifest, leads[0])
    _check_deadline(deadline)
    model, saved, training_identity, sidecar_identity = _checkpoint_model(
        manifest, checkpoint, root, store, process_scale_sidecar)
    _check_deadline(deadline)
    climatology = fit_training_climatology(store)
    if tuple(climatology["channels"]) != ds.names:
        raise ValueError("climatology variable order differs from evaluation")
    _check_deadline(deadline)
    device = select_device(device_name)
    entry_memory = _memory_snapshot(device)
    model = model.to(device).eval()
    pooled = RegionMetrics(leads, ds.names, training_std=ds.std, units=ds.units)
    references = {kind: RegionMetrics(leads, ds.names, training_std=ds.std, units=ds.units)
                  for kind in ("persistence", "climatology")}
    out.mkdir(parents=True, exist_ok=False)
    _synchronize(device)
    loop_baseline = _memory_snapshot(device)
    loop_started = time.perf_counter()
    initializations, bad_cases, baseline_initializations = [], [], []
    profile_inputs = None
    with ((out / "per_case_metrics.csv").open("x", encoding="utf-8", newline="") as stream,
          (out / "baseline_per_case_metrics.csv").open("x", encoding="utf-8", newline="") as baseline_stream):
        writer = baseline_writer = None
        for i in range(min(max_samples, len(ds))):
            _check_deadline(deadline)
            sample = ds[i]
            inputs = rollout_model_input(sample, lead_hours=STEP_HOURS, device=device)
            if profile_inputs is None:
                profile_inputs = inputs
            trajectory = autoregressive_rollout(model, inputs, lead_hours=leads, step_hours=STEP_HOURS,
                                                history_interval_hours=STEP_HOURS,
                                                inference_kwargs={"reasoning_steps": reasoning_steps})
            expected_work = reasoning_steps * leads[0] // STEP_HOURS
            if not (trajectory.cumulative_reasoning_steps == expected_work).all():
                raise ValueError("rollout actual cumulative K differs from fixed-depth physical transitions")
            prediction = trajectory.forecasts.cpu()
            work = trajectory.cumulative_reasoning_steps.cpu().tolist()[0]
            del trajectory
            target = sample["rollout_targets"].unsqueeze(0)
            climate = normalized_climatology(climatology, sample["valid_times"], ds.mean, ds.std).unsqueeze(0)
            case = RegionMetrics(leads, ds.names, training_std=ds.std, units=ds.units)
            case.update(prediction, target, climate, sample["latitude"])
            pooled.update(prediction, target, climate, sample["latitude"])
            rows = case.rows()
            record = records[sample["init_time"]]
            flat = [dict(sample_id=record["sample_id"], init_time=sample["init_time"],
                         valid_time=sample["valid_times"][0], valid_times=sample["valid_times"],
                         reasoning_steps=reasoning_steps, **row) for row in rows]
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
                writer.writeheader()
            writer.writerows(_csv_row(row) for row in flat)
            stream.flush()  # A later real error/deadline must not erase completed cases.
            bad_cases.extend(row for row in flat if row["bad_reasons"])
            initializations.append(dict(sample_id=record["sample_id"], init_time=sample["init_time"],
                                        valid_times=sample["valid_times"], cumulative_reasoning_steps=work,
                                        region_metrics=rows))
            reference_cases = {}
            for kind, forecast in _reference_forecasts(sample, climate).items():
                reference_cases[kind] = RegionMetrics(leads, ds.names, training_std=ds.std, units=ds.units)
                reference_cases[kind].update(forecast, target, climate, sample["latitude"])
                references[kind].update(forecast, target, climate, sample["latitude"])
            baseline_rows = _reference_rows(reference_cases)
            baseline_flat = [dict(sample_id=record["sample_id"], init_time=sample["init_time"],
                                  valid_time=sample["valid_times"][0], valid_times=sample["valid_times"],
                                  reasoning_steps=None, **row) for row in baseline_rows]
            if baseline_writer is None:
                baseline_writer = csv.DictWriter(baseline_stream, fieldnames=list(baseline_flat[0]))
                baseline_writer.writeheader()
            baseline_writer.writerows(_csv_row(row) for row in baseline_flat)
            baseline_stream.flush()
            baseline_initializations.append(dict(sample_id=record["sample_id"], init_time=sample["init_time"],
                                                 valid_times=sample["valid_times"], region_metrics=baseline_rows))
    _synchronize(device)
    loop_elapsed = time.perf_counter() - loop_started
    loop_memory = _memory_snapshot(device)
    _check_deadline(deadline)
    rows = pooled.rows()
    _write_csv(out / "region_metrics.csv", rows)
    baseline_rows = _reference_rows(references)
    _write_csv(out / "baseline_region_metrics.csv", baseline_rows)
    isolated = _profile_forward(model, profile_inputs, reasoning_steps, deadline)
    _synchronize(device)
    final_memory = _memory_snapshot(device)
    contract = saved["contract"]
    provenance = dict(
        format="r7-v2-evaluation-v1", scientific_claim=False, limitations=list(LIMITATIONS),
        checkpoint=str(checkpoint.resolve()), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        model_code_sha256=saved["model_code_sha256"], checkpoint_contract_sha256=saved["signature"],
        checkpoint_training_steps=4, reasoning_steps=reasoning_steps, independently_trained_k1=False,
        depth_probe="same K4-trained checkpoint; inference-depth probe, not independent K1/K2 training",
        inference_options={"reasoning_steps": reasoning_steps}, model_kind=contract["kind"], model_config=contract["model"],
        known_input_budget=dict(spacetime_inputs=bool(contract["model"].get("spacetime_inputs", False)),
                               producer_calendar_metadata=True, cross_arm_equality="must be checked by frozen protocol, not assumed"),
        training_identity=training_identity, process_scale_sidecar_identity=sidecar_identity,
        evaluation_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        source_declaration=root.attrs["source"], channels=list(ds.names), units=list(ds.units),
        split="val", test_read=False, lead_hours=list(leads), step_hours=STEP_HOURS,
        n_manifest_records=len(records), n_available_windows=len(ds), n_evaluated=len(initializations),
        max_samples=max_samples, selection="first N chronological manifest initializations complete at this requested lead; no 72h cohort narrowing",
        case_pairing_scope="same seed/lead across arms and K; different leads retain all their own legal cases",
        case_selection_sha256=canonical_digest([dict(sample_id=c["sample_id"], init_time=c["init_time"]) for c in initializations]),
        region_definitions=pooled.regions, region_metrics=rows, initializations=initializations,
        baseline_region_metrics=baseline_rows, baseline_initializations=baseline_initializations,
        baseline_definition={"persistence": {"kind": "known-last-history-frame", "training_updates": 0},
                             "climatology": {"kind": "train-only-month-hour", "training_updates": 0},
                             "case_pairing": "same exact requested-lead initializations and full/interior/edge_2"},
        bad_variable_metrics=[row for row in rows if row["bad_reasons"]], bad_case_metrics=bad_cases,
        bad_definition="negative skill/ACC or undefined anomaly energy; diagnostic flags only, no rows excluded",
        climatology=dict(kind=climatology["kind"], training_years=climatology["training_years"],
                         selection=climatology["selection"], n_selected_steps=climatology["n_selected_steps"],
                         bucket_counts={f"{m:02d}-{h:02d}": n for (m, h), n in climatology["counts"].items()},
                         scope="same case/variable/region/area weights for RMSE, pooled ACC and climatology MSE skill"),
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        forward_backward_flops=None, flop_measurement_owner="separate CPU preparation with actual training loss and backward",
        elapsed_seconds=loop_elapsed, loop_elapsed_seconds=loop_elapsed,
        timing_scope="actual evaluation loop including IO/transfers/rollout/model and zero-trained baseline CPU metrics/per-case CSV, not isolated latency",
        isolated_forward=isolated,
        cuda_memory=dict(entry=entry_memory, loop_baseline=loop_baseline, after_loop=loop_memory, after_profile=final_memory,
                         allocator_reset_performed=False, preinit_baseline_measurement_owner="fresh worker",
                         scope="actual process allocator; peak since worker reset, including model/inputs/loop/profile; never cleared here"),
        tables=dict(region_metrics="region_metrics.csv", per_case_metrics="per_case_metrics.csv",
                    baseline_region_metrics="baseline_region_metrics.csv", baseline_per_case_metrics="baseline_per_case_metrics.csv"),
        reproducibility_level="identity-bound numerical replay; no cross-device bitwise claim",
        whole_elapsed_seconds=time.perf_counter() - whole_started,
        whole_timing_scope="helper preparation, loop, region CSV and profiling; excludes final provenance JSON publication; driver owns whole worker time",
    )
    if contract.get("intervention") is not None:
        provenance["intervention"] = contract["intervention"]
    if contract.get("process_supervision") is not None:
        provenance["training_protocol_sha256"] = contract["process_supervision"]["protocol_sha256"]
    _check_deadline(deadline)
    with (out / "provenance.json").open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return provenance
