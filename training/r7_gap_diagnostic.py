"""Read-only same-case train/development gap diagnostic, never scientific acceptance.

The owned caller freezes/publishes the protocol and enforces the blocking-call
wall deadline. This core returns JSON only; it never spawns, trains or publishes.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from data.r7_autoregressive_dataset import preflight_training_windows
from data.r7_evaluation import ZarrRolloutDataset, fit_training_climatology, normalized_climatology
from data.r7_long_rollout_dataset import ZarrLongRolloutDataset, preflight_long_rollout_windows
from data.r7_store import HOUR_NS, init_time_fields, split_time_labels, validate_record, validate_store
from data.r7_zarr_dataset import ZarrAtmosWindowDataset
from model.r7_rollout import autoregressive_rollout, rollout_model_input
from .r7_autoregressive_runner import _module_semantics
from .r7_experiment import canonical_digest, dataset_identity, load_checkpoint, make_model, select_device
from .r7_rollout_metrics import RolloutRMSEAccumulator

LEADS = (6, 12, 24, 48, 72)
TARGET_POSITIONS = (0, 1, 3, 7, 11)
TRAIN_YEARS = list(range(2017, 2022))
MONTHS = (1, 4, 7, 10)
LABELS = {"train": "in_sample_train", "val": "development_val"}
LIMITATIONS = [
    "Twenty in-sample training cases and four development cases are sparse, not independent confirmation.",
    "Training cases and the train-fitted climatology are optimistic; neither is an independent baseline sample.",
    "The train/validation gap and paired changes are descriptive, not causal or generalization proof.",
    "Monthly/hourly train grid means are unchanged local climatology, not a WeatherBench2 reproduction.",
    "No test access, optimizer update, threshold selection or scientific verdict; no cross-variable unit average.",
    "Config-reproducible only: exact pins do not establish bitwise agreement across devices/software.",
    "Cooperative read/transition deadlines cannot interrupt blocking calls; the owned external watchdog must.",
]


def _deadline(deadline):
    if deadline is not None:
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
            raise ValueError("deadline must be a finite absolute perf_counter limit")
        if time.perf_counter() >= deadline:
            raise RuntimeError("gap diagnostic deadline exceeded")


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a SHA256 digest")
    return value


def _local(path, *, name=None):
    requested = Path(path)
    if "://" in str(path) or any(p.is_symlink() for p in (requested, *requested.parents)):
        raise ValueError("local nonsymlink paths required")
    result = requested.resolve()
    if result.name == "test.jsonl" or (name is not None and result.name != name):
        raise ValueError(f"only explicit {name or 'non-test'} paths accepted; test stays sealed")
    return result


def _sha256_file(path, deadline=None):
    value = hashlib.sha256()
    with _local(path).open("rb") as handle:
        while True:
            _deadline(deadline)
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            value.update(chunk)
    return value.hexdigest()


def _matches(actual, expected, label):
    """Typed recursive subset for optional, already digest-bound protocol pins."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"{label} differs")
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f"{label}.{key} missing")
            _matches(actual[key], value, f"{label}.{key}")
    elif canonical_digest(actual) != canonical_digest(expected):
        raise ValueError(f"{label} differs")


def _records(reader, split, store, root):
    previous = None
    for record in reader.records:
        if record.get("split") != split or _local(reader.manifest.parent / record["store_path"]) != store:
            raise ValueError("one same completed store and the declared split required")
        validate_record(root, record)
        history = record["history_times"]
        if len(history) != 2 or record["lead_time_hours"] != 6:
            raise ValueError("exact two-history +6h records required")
        stamps = [pd.Timestamp(t) for t in history + [record["target_time"]]]
        stamp = pd.Timestamp(record["init_time"])
        years = TRAIN_YEARS if split == "train" else [2022]
        if any(t.year not in years or t.month not in MONTHS for t in stamps):
            raise ValueError("wrong diagnostic year/month")
        if stamp.value % (6 * HOUR_NS) or stamps[1].value - stamps[0].value != 6 * HOUR_NS:
            raise ValueError("exact UTC-aligned 6h history required")
        if previous is not None and stamp <= previous:
            raise ValueError("manifest initializations must be unique and chronological")
        previous = stamp


def _pin(record, position, history, targets, times):
    return {"sample_id": record["sample_id"], "manifest_index": position,
            "record_sha256": canonical_digest(record), "store_path": record["store_path"],
            "init_time": record["init_time"], "history_indices": list(history),
            "history_times": [times[i].isoformat() for i in history],
            "target_indices": list(targets), "target_times": [times[i].isoformat() for i in targets]}


def _climate_metadata(root, raw):
    times = pd.DatetimeIndex(raw.astype("datetime64[ns]"))
    labels = split_time_labels(root)
    chosen = [t for i, t in enumerate(times) if (t.year in TRAIN_YEARS if labels is None else labels[i] == "train")]
    counts = {}
    for stamp in chosen:
        if stamp.year not in TRAIN_YEARS or stamp.month not in MONTHS or stamp.hour not in (0, 6, 12, 18):
            raise ValueError("climatology train stamp outside declared season/cadence")
        key = f"{stamp.month:02d}-{stamp.hour:02d}"
        counts[key] = counts.get(key, 0) + 1
    if set(counts) != {f"{m:02d}-{h:02d}" for m in MONTHS for h in (0, 6, 12, 18)}:
        raise ValueError("all sixteen train climatology buckets required")
    return {"training_years": list(TRAIN_YEARS), "n_selected_steps": len(chosen),
            "bucket_counts": counts, "selection": "declared_train_years" if labels is None else "declared_train_time_ranges"}


def metadata_case_plan(train_manifest, val_manifest):
    """Metadata only: lower chronological median per year/month; exactly 20+4 cases.

    All twelve physical targets are bound even though only five leads are scored.
    Boundary/gap exclusions remain frozen, never silently changed after selection.
    """
    train, val = _local(train_manifest, name="train.jsonl"), _local(val_manifest, name="val.jsonl")
    if train.parent != val.parent:
        raise ValueError("same completed manifest publication required")
    readers = {s: ZarrAtmosWindowDataset(p) for s, p in (("train", train), ("val", val))}
    store = _local(train.parent / readers["train"].records[0]["store_path"])
    root = readers["train"]._store(readers["train"].records[0])
    raw = validate_store(root)
    if root.attrs["split_years"]["train"] != TRAIN_YEARS or root.attrs["split_years"]["val"] != [2022]:
        raise ValueError("train2017-2021 / val2022 declaration required")
    for split, reader in readers.items():
        _records(reader, split, store, root)
    names, units = list(root.attrs["channels"]), list(root.attrs.get("units", []))
    if len(names) != 17 or len(units) != 17 or any(not isinstance(u, str) or u in ("", "unknown", "normalized") for u in units):
        raise ValueError("all 17 original variables and explicit physical units required")
    long = preflight_long_rollout_windows(train, physical_steps=12)
    times = pd.DatetimeIndex(raw.astype("datetime64[ns]"))
    train_pins = []
    for pin in long["windows"]:
        record = readers["train"].records[pin["manifest_index"]]
        train_pins.append(_pin(record, pin["manifest_index"], pin["history_indices"], pin["target_indices"], times))
    vd = ZarrRolloutDataset(store, split="val", lead_hours=tuple(range(6, 73, 6)), history_steps=2, step_hours=6)
    windows = {tuple(h): t for h, t in vd.windows}
    val_pins, val_excluded = [], []
    for position, record in enumerate(readers["val"].records):
        history = tuple(record["history_indices"])
        if history not in windows:
            val_excluded.append(record["sample_id"])
            continue
        val_pins.append(_pin(record, position, history, windows[history], times))
    cases = []
    for split, pins, years in (("train", train_pins, TRAIN_YEARS), ("val", val_pins, [2022])):
        for year in years:
            for month in MONTHS:
                group = sorted([p for p in pins if pd.Timestamp(p["init_time"]).year == year
                                and pd.Timestamp(p["init_time"]).month == month], key=lambda p: p["init_time"])
                if not group:
                    raise ValueError(f"no complete 12-transition {split} window for {year}-{month:02d}")
                rank = (len(group) - 1) // 2
                pin = group[rank]
                if any(pd.Timestamp(t).year not in years or pd.Timestamp(t).month not in MONTHS
                       for t in pin["history_times"] + pin["target_times"]):
                    raise ValueError("complete window crosses declared diagnostic years/months")
                stamp = pd.Timestamp(pin["init_time"])
                cases.append({**pin, "split": split, "group_label": LABELS[split], "year": year, "month": month,
                              "group": f"{split}-{year}-{month:02d}", "group_complete_windows": len(group),
                              "selection_rank": rank, "history_offsets_hours": [-6., 0.],
                              **init_time_fields(stamp.value), "lead_hours": list(LEADS),
                              "scored_target_indices": [pin["target_indices"][i] for i in TARGET_POSITIONS],
                              "valid_times": [pin["target_times"][i] for i in TARGET_POSITIONS]})
    body = {"format": "r7-same-case-gap-selection-v1", "scientific_claim": False, "test_read": False,
            "state_fields_read": False, "train_manifest": str(train), "val_manifest": str(val), "store": str(store),
            "source_declaration": root.attrs["source"], "channels": names, "units": units,
            "normalization_mean": np.asarray(root["normalization_mean"][:]).tolist(),
            "normalization_std": np.asarray(root["normalization_std"][:]).tolist(), "shape": list(root["state"].shape),
            "train_windows": long, "val_windows": {"windows": val_pins, "excluded_sample_ids": val_excluded},
            "climatology_metadata": _climate_metadata(root, raw), "n_cases": len(cases), "cases": cases,
            "selection": "lower chronological median complete window per train year/month and val month; no adjacency lookup"}
    return {**body, "selection_sha256": canonical_digest(body)}


def _source_identity(manifest, root, expected, deadline):
    source = _local(manifest.parent / root.attrs["source"])
    if not source.is_file() or _sha256_file(source, deadline) != expected:
        raise ValueError("full source SHA256 mismatch or missing local source file")
    result = {"path": str(source), "sha256": expected, "bytes": source.stat().st_size, "scope": "full-local-file"}
    preflight = manifest.parent / "source_preflight.json"
    if preflight.exists():
        _deadline(deadline)
        report = json.loads(preflight.read_text(encoding="utf-8"))
        if (_local(report["source_path"]) != source or report["fingerprint"] !=
                {"scope": "full-local-file", "sha256": expected, "bytes": result["bytes"]}):
            raise ValueError("source root/preflight fingerprint mismatch")
        result["preflight_sha256"] = _sha256_file(preflight, deadline)
    return result


def _restore(path, expected_hash, role, train_identity, source_hash, windows, extra, deadline):
    _deadline(deadline)
    if _sha256_file(path, deadline) != expected_hash:
        raise ValueError(f"{role} checkpoint SHA256 mismatch")
    saved = load_checkpoint(_local(path))
    contract = saved["contract"]
    endpoint, mode = (1600, "l6") if role == "parent" else (200, "long_rollout")
    _matches(contract, {"kind": "process", "seed": 41, "steps": 4, "mode": mode, "total_updates": endpoint,
                        "bf16": False, "data_identity": train_identity, "source_sha256": source_hash,
                        "model_code_sha256": saved["model_code_sha256"]}, role)
    _digest(contract.get("protocol_sha256"), f"{role} training protocol")
    if type(saved.get("updates")) is not int or saved["updates"] != endpoint:
        raise ValueError(f"{role} must be the frozen endpoint")
    if contract.get("process_supervision") is not None or contract.get("intervention") is not None:
        raise ValueError("atmospheric-only ordinary checkpoints required")
    if contract.get("scientific_claim") is not False or contract.get("test_read", False) is not False:
        raise ValueError("checkpoint claims/test scope invalid")
    _matches(contract, extra, f"{role} expected_contract")
    _matches(contract["autoregression"], {"mode": mode, "step_hours": 6, "fixed_transition_lead_hours": 6,
        "window_sha256": windows["window_sha256"], "windows": windows,
        "excluded_sample_ids": windows["excluded_sample_ids"]}, f"{role} windows")
    if role == "candidate":
        _matches(contract, {"lr": 2e-5, "warmup_updates": 10}, "candidate recipe")
        _matches(contract["autoregression"], {"physical_steps": 12,
            "physical_weights": [1., .5, 0., .5, 0., 0., 0., .5, 0., 0., 0., .5],
            "detach_physical_steps": False, "detach_reasoning_steps": False}, "candidate physical recipe")
    _deadline(deadline)
    with torch.random.fork_rng(devices=[]):
        model = make_model("process", contract["model"])
    state, template = saved["model"], model.state_dict()
    if not isinstance(state, dict) or set(state) != set(template):
        raise ValueError("strict model state keys differ")
    for key, value in state.items():
        if (not torch.is_tensor(value) or value.device.type != "cpu" or value.shape != template[key].shape
                or value.dtype != template[key].dtype):
            raise ValueError(f"strict model state shape/dtype differs: {key}")
        if value.is_floating_point() and (value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError(f"finite FP32 model state required: {key}")
    semantics = canonical_digest(_module_semantics(model))
    if contract.get("model_semantics_sha256") != semantics:
        raise ValueError("checkpoint model semantics differ")
    if (model.default_reasoning_steps != 4 or model.backbone.in_channels != 17
            or model.backbone.history_steps != 2 or model.backbone.out_channels != 17):
        raise ValueError("K4, two-history, all17-channel model required")
    model.load_state_dict(state, strict=True)
    model.eval()
    _deadline(deadline)
    return model, {"sha256": expected_hash, "contract_sha256": saved["signature"], "updates": endpoint,
                   "training_protocol_sha256": contract["protocol_sha256"], "mode": mode, "seed": 41,
                   "model_code_sha256": saved["model_code_sha256"], "model_semantics_sha256": semantics,
                   "model_spec_sha256": canonical_digest(contract["model"]), "optimizer_imported": False}


class _ReadGuard:
    def __init__(self, array, deadline):
        self.array, self.deadline = array, deadline

    def __getattr__(self, name):
        return getattr(self.array, name)

    def __getitem__(self, index):
        _deadline(self.deadline)
        return self.array[index]


class _RootGuard:
    def __init__(self, root, deadline):
        self.root, self.attrs, self.deadline = root, root.attrs, deadline

    def __contains__(self, name):
        return name in self.root

    def __getitem__(self, name):
        array = self.root[name]
        return _ReadGuard(array, self.deadline) if name in ("state", "process_diagnostics_raw") else array


@contextmanager
def _guard_reads(store, reader, deadline):
    """Narrow owned-worker-only guard; restore library binding and cached roots."""
    import zarr
    original, cached = zarr.open_group, reader._stores
    def guarded(path, *args, **kwargs):
        root = original(path, *args, **kwargs)
        if Path(path).resolve() == store and kwargs.get("mode") == "r":
            return _RootGuard(root, deadline)
        return root
    try:
        zarr.open_group = guarded
        reader._stores = {key: _RootGuard(root, deadline) for key, root in cached.items()}
        yield
    finally:
        zarr.open_group = original
        reader._stores = cached


def climatology_identity(climatology):
    """Opaque identity of the unchanged estimator's per-bucket FP64 mean arrays."""
    means = {}
    for key, mean in sorted(climatology["means"].items()):
        array = np.ascontiguousarray(mean)
        if array.dtype != np.float64 or not np.isfinite(array).all():
            raise ValueError("unchanged finite FP64 climatology means required")
        means[f"{key[0]:02d}-{key[1]:02d}"] = {"dtype": array.dtype.str, "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    counts = {f"{m:02d}-{h:02d}": n for (m, h), n in climatology["counts"].items()}
    if set(means) != set(counts) or sum(counts.values()) != climatology["n_selected_steps"]:
        raise ValueError("climatology bucket/step identity differs")
    return {"kind": climatology["kind"], "training_years": climatology["training_years"],
            "channels": climatology["channels"], "selection": climatology["selection"],
            "n_selected_steps": climatology["n_selected_steps"], "bucket_counts": counts,
            "means": means, "mean_identity_sha256": canonical_digest(means)}


def physical_mse(prediction, target, latitude, *, training_std, variables, units):
    """[5,17] physical MSE, equal case weight and separate latitude-area weights."""
    metric = RolloutRMSEAccumulator(LEADS, variables, training_std=training_std, units=units)
    metric.update(prediction, target, latitude)
    return metric.sum_squared_error / metric.initializations


def _ratio(numerator, denominator, undefined):
    denominator = np.broadcast_to(denominator, numerator.shape)
    values, statuses = [], []
    for top, bottom in zip(numerator, denominator):
        row = [float(n / d) if d > 0 else None for n, d in zip(top, bottom)]
        if any(v is not None and not math.isfinite(v) for v in row):
            raise ValueError("nonfinite descriptive ratio")
        values.append(row)
        statuses.append(["defined" if d > 0 else undefined for d in bottom])
    return values, statuses


def metric_summary(mse_by_model):
    """Pure per-variable summaries; undefined zero denominators are JSON null."""
    if set(mse_by_model) != {"parent", "candidate", "climatology"}:
        raise ValueError("all two models and identical-case climatology required")
    arrays = {key: np.asarray(value, dtype=np.float64) for key, value in mse_by_model.items()}
    if any(x.shape != (5, 17) or not np.isfinite(x).all() or (x < 0).any() for x in arrays.values()):
        raise ValueError("finite nonnegative [5,17] per-variable MSE required")
    metrics = {}
    for key, mse in arrays.items():
        rmse = np.sqrt(mse)
        relative, status = _ratio(mse - mse[:1], mse[:1], "undefined_zero_6h_mse")
        metrics[key] = {"mse": mse.tolist(), "rmse": rmse.tolist(),
                        "mse_growth_from_6h": (mse - mse[:1]).tolist(),
                        "relative_mse_growth": relative, "relative_mse_growth_status": status}
        if key != "climatology":
            skill, status = _ratio(arrays["climatology"] - mse, arrays["climatology"], "undefined_zero_climatology_mse")
            metrics[key].update(mse_skill=skill, mse_skill_status=status)
    delta = arrays["candidate"] - arrays["parent"]
    relative, status = _ratio(delta, arrays["parent"], "undefined_zero_parent_mse")
    return {"metrics": metrics, "pair_delta": {"candidate_minus_parent_mse": delta.tolist(),
        "candidate_minus_parent_rmse": (np.sqrt(arrays["candidate"]) - np.sqrt(arrays["parent"])).tolist(),
        "relative_mse_change": relative, "relative_mse_change_status": status}}


def aggregate_cases(cases):
    """Simple equal-case mean MSE before RMSE/skill; never average unlike units."""
    if not cases or any(case["split"] not in LABELS for case in cases):
        raise ValueError("train/development cases required")
    groups = {"split": {}, "train_year": {}, "train_month": {}}
    for case in cases:
        groups["split"].setdefault(case["split"], []).append(case)
        if case["split"] == "train":
            for axis, key in (("train_year", "year"), ("train_month", "month")):
                groups[axis].setdefault(str(case[key]), []).append(case)
    result = {}
    for axis, rows in groups.items():
        result[axis] = {}
        for key, selected in rows.items():
            mse = {model: np.mean([case["metrics"][model]["mse"] for case in selected], axis=0)
                   for model in ("parent", "candidate", "climatology")}
            result[axis][key] = {"n_cases": len(selected), "sample_ids": [c["sample_id"] for c in selected],
                                **metric_summary(mse)}
    return result


def _gap(aggregates):
    result = {}
    for model in ("parent", "candidate", "climatology"):
        train, val = [aggregates["split"][s]["metrics"][model] for s in ("train", "val")]
        delta = np.asarray(val["mse"]) - np.asarray(train["mse"])
        relative, status = _ratio(delta, np.asarray(train["mse"]), "undefined_zero_train_mse")
        result[model] = {"val_minus_train_mse": delta.tolist(),
                         "val_minus_train_rmse": (np.asarray(val["rmse"]) - np.asarray(train["rmse"])).tolist(),
                         "relative_mse_gap": relative, "relative_mse_gap_status": status}
    return result


@torch.no_grad()
def predict_case(model, sample, *, device=torch.device("cpu"), deadline=None):
    """One target-free 12-transition K4 rollout; deadline checked at every call."""
    _deadline(deadline)
    if any(module.training for module in model.modules()) or torch.is_autocast_enabled(device.type):
        raise ValueError("eval FP32 inference without autocast required")
    initial = rollout_model_input(sample, lead_hours=6., device=device)
    if initial["coarse_history"].dtype != torch.float32:
        raise ValueError("FP32 history required")
    hook = model.register_forward_pre_hook(lambda *_args: _deadline(deadline))
    try:
        trajectory = autoregressive_rollout(model, initial, lead_hours=LEADS, step_hours=6,
            history_interval_hours=6, inference_kwargs={"reasoning_steps": 4})
    finally:
        hook.remove()
    _deadline(deadline)
    if (trajectory.model_calls != 12 or trajectory.forecasts.dtype != torch.float32
            or trajectory.cumulative_reasoning_steps.cpu().tolist() != [[4, 8, 16, 32, 48]]):
        raise ValueError("exact twelve-transition FP32 K4 rollout required")
    return trajectory.forecasts.cpu()


def _case_datasets(plan, train, val):
    train_ds = ZarrLongRolloutDataset(train, physical_steps=12,
        expected_exclusions=plan["train_windows"]["excluded_sample_ids"],
        expected_window_sha256=plan["train_windows"]["window_sha256"])
    val_ds = ZarrRolloutDataset(plan["store"], split="val", lead_hours=LEADS, history_steps=2, step_hours=6)
    lookup = {tuple(h): (h, t) for h, t in val_ds.windows}
    selected = []
    for case in plan["cases"]:
        if case["split"] == "val":
            window = lookup.get(tuple(case["history_indices"]))
            if window is None or window[1] != case["scored_target_indices"]:
                raise ValueError("exact validation history/target whitelist differs")
            selected.append(window)
    val_ds.windows = selected
    return train_ds, val_ds


@torch.no_grad()
def run_gap_diagnostic(train_manifest, val_manifest, *, parent_checkpoint, candidate_checkpoint,
                       parent_sha256, candidate_sha256, expected_case_plan, expected_train_identity,
                       expected_val_identity, expected_source_sha256, protocol_sha256,
                       device_name="cpu", deadline=None, protocol=None):
    """Return JSON body without publication. Optional protocol pins cannot relax scope.

    ``protocol`` is digest-bound and can add nested ``expected_contracts`` subsets
    for parent/candidate, plus ``expected_climatology`` counts/mean identity pins.
    Without count overrides the registered v3 estimator must fit 2400 steps,
    sixteen month/hour buckets of 150, in 2017-2021. Smaller CPU fixture counts
    must be explicitly frozen by their synthetic engineering protocol.
    """
    _deadline(deadline)
    if torch.device(device_name).type not in ("cpu", "cuda") or any(
            torch.is_autocast_enabled(kind) for kind in ("cpu", "cuda")):
        raise ValueError("CPU/CUDA FP32 inference without autocast required")
    for name, value in (("parent_sha256", parent_sha256), ("candidate_sha256", candidate_sha256),
                        ("train identity", expected_train_identity), ("val identity", expected_val_identity),
                        ("source SHA256", expected_source_sha256), ("protocol SHA256", protocol_sha256)):
        _digest(value, name)
    options = {} if protocol is None else protocol
    if protocol is not None:
        if (protocol.get("protocol_sha256") != protocol_sha256 or canonical_digest(
                {k: v for k, v in protocol.items() if k != "protocol_sha256"}) != protocol_sha256
                or protocol.get("scientific_claim") is not False or protocol.get("test_read") is not False):
            raise ValueError("frozen diagnostic protocol digest/claims differ")
    plan = metadata_case_plan(train_manifest, val_manifest)
    if canonical_digest(plan) != canonical_digest(expected_case_plan):
        raise ValueError("exact metadata case plan differs from freeze")
    _deadline(deadline)
    train, val = Path(plan["train_manifest"]), Path(plan["val_manifest"])
    train_identity, reader = dataset_identity(train)
    val_identity, _ = dataset_identity(val)
    if (train_identity, val_identity) != (expected_train_identity, expected_val_identity):
        raise ValueError("training/validation data and normalization identity mismatch")
    root = reader._store(reader.records[0])
    source = _source_identity(train, root, expected_source_sha256, deadline)
    climate_pins = options.get("expected_climatology", {"n_selected_steps": 2400,
        "bucket_counts": {f"{m:02d}-{h:02d}": 150 for m in MONTHS for h in (0, 6, 12, 18)}})
    if not isinstance(climate_pins, dict) or not {"n_selected_steps", "bucket_counts"} <= set(climate_pins):
        raise ValueError("explicit climatology count/bucket pins required")
    _matches(plan["climatology_metadata"], {k: v for k, v in climate_pins.items()
        if k in plan["climatology_metadata"]}, "climatology metadata")
    extra = options.get("expected_contracts", {})
    models, checkpoints = {}, {}
    for role, path, sha, windows in (("parent", parent_checkpoint, parent_sha256, preflight_training_windows(train)),
                                   ("candidate", candidate_checkpoint, candidate_sha256, plan["train_windows"])):
        models[role], checkpoints[role] = _restore(path, sha, role, train_identity, expected_source_sha256,
                                                  windows, extra.get(role, {}), deadline)
    if checkpoints["parent"]["model_spec_sha256"] != checkpoints["candidate"]["model_spec_sha256"]:
        raise ValueError("parent/candidate model configuration differs")
    train_ds, val_ds = _case_datasets(plan, train, val)
    _deadline(deadline)
    device = select_device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for model in models.values():
        model.to(device).eval()
    cases, train_indices = [], {position: i for i, (position, _) in enumerate(train_ds.windows)}
    val_index, started = 0, time.perf_counter()
    with _guard_reads(Path(plan["store"]), train_ds.reader, deadline):
        _deadline(deadline)
        climatology = fit_training_climatology(plan["store"])
        _deadline(deadline)
        climate_identity = climatology_identity(climatology)
        _matches(climate_identity, {**plan["climatology_metadata"], "channels": plan["channels"]}, "climatology actual fit")
        _matches(climate_identity, climate_pins, "climatology frozen pins")
        for case in plan["cases"]:
            _deadline(deadline)
            if case["split"] == "train":
                sample = train_ds[train_indices[case["manifest_index"]]]
                if sample["physical_valid_times"] != case["target_times"]:
                    raise ValueError("train sample exact physical times differ")
                sample["rollout_targets"] = sample["physical_targets"][list(TARGET_POSITIONS)]
                sample["valid_times"] = case["valid_times"]
            else:
                sample, val_index = val_ds[val_index], val_index + 1
            if sample["init_time"] != case["init_time"] or sample["valid_times"] != case["valid_times"]:
                raise ValueError("sample initialization/scored times differ")
            target = sample["rollout_targets"].unsqueeze(0)
            climate = normalized_climatology(climatology, case["valid_times"], val_ds.mean, val_ds.std).unsqueeze(0)
            predictions = {"climatology": climate}
            for role, model in models.items():
                predictions[role] = predict_case(model, sample, device=device, deadline=deadline)
            mse = {key: physical_mse(value, target, sample["latitude"], training_std=val_ds.std,
                                    variables=val_ds.names, units=val_ds.units).tolist()
                   for key, value in predictions.items()}
            cases.append({**case, **metric_summary(mse)})
            _deadline(deadline)
    if len(cases) != 24 or val_index != 4:
        raise ValueError("all24 same cases required; no silent dropping")
    aggregates = aggregate_cases(cases)
    peaks = {"owned_cuda_allocated_peak_bytes": 0, "owned_cuda_reserved_peak_bytes": 0}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peaks = {"owned_cuda_allocated_peak_bytes": int(torch.cuda.max_memory_allocated(device)),
                 "owned_cuda_reserved_peak_bytes": int(torch.cuda.max_memory_reserved(device))}
        if any(value <= 0 for value in peaks.values()):
            raise ValueError("positive owned CUDA peaks required")
    _deadline(deadline)
    body = {"format": "r7-same-case-gap-diagnostic-v1", "scientific_claim": False, "test_read": False,
            "optimizer_imported": False, "optimizer_updates": 0, "diagnostic_training_mode": False,
            "limitations": list(LIMITATIONS), "protocol_sha256": protocol_sha256,
            "selection_sha256": plan["selection_sha256"], "case_plan": plan, "n_evaluated": len(cases),
            "train_data_identity": train_identity, "val_data_identity": val_identity, "source_identity": source,
            "source_sha256": expected_source_sha256, "checkpoints": checkpoints,
            "model_code_sha256": checkpoints["parent"]["model_code_sha256"],
            "training_normalization": {"mean": plan["normalization_mean"], "std": plan["normalization_std"]},
            "channels": plan["channels"], "units": plan["units"], "lead_hours": list(LEADS),
            "climatology": climate_identity, "cases": cases, "aggregates": aggregates,
            "train_val_gap": _gap(aggregates), "reading_scope": "descriptive; no independent/causal interpretation",
            "reproducibility": "config-reproducible; bitwise cross-device identity not established",
            "device_type": device.type, "bf16": False, "reasoning_steps": 4, "physical_transitions_per_model_case": 12,
            "elapsed_seconds": time.perf_counter() - started, **peaks}
    json.dumps(body, allow_nan=False)
    return body
