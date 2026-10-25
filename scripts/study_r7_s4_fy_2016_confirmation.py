"""S4 first confirmation of the FY candidate on the complete unseen year 2016.

Scores the three frozen FY candidate (treatment) checkpoints and the three
same-data incumbent (control) checkpoints on 2016 and reports the D4
composition. Per decision 0059 this does **not** build a 2016 store: the frames
come from the pinned multiyear store, the FY 2017-2021 normalization is applied
explicitly, and the checkpoint identity is asserted explicitly against the arm's
FY ``train.jsonl``.

Read-only with respect to every frozen artifact: it opens stores and checkpoints,
writes only into its own ``--out`` directory, and never touches ``data/`` or the
archive snapshots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HOUR_NS = 3_600_000_000_000


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path):
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if body.get("format") != "r7-s4-fy-2016-confirmation-protocol-v1":
        raise ValueError(f"unexpected protocol format: {body.get('format')!r}")
    return body


def make_scan(store_path, *, history_steps, step_hours, mean, std, allowed_year):
    """A store scan restricted to the declared instance year with explicit normalization.

    ``RolloutStoreScan`` takes its normalization from the store it opens and its
    held-out membership from the store's ``split_years``. This instance needs
    neither: the frames are 2016 (a year the multiyear store does not declare as
    held-out) and the normalization is the FY 2017-2021 fit. Both are replaced
    here, so the frozen reader machinery is reused unchanged (decision 0059).
    """
    from data.r7_evaluation import RolloutStoreScan

    scan = RolloutStoreScan(store_path, split="test", history_steps=history_steps,
                            step_hours=step_hours)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape != (len(scan.names),) or std.shape != mean.shape:
        raise ValueError("normalization arrays must match the channel count")
    if not np.isfinite(mean).all() or not (std > 0).all():
        raise ValueError("normalization must be finite with positive std")
    scan.mean, scan.std = mean, std
    scan.allowed_years = frozenset({int(allowed_year)})
    return scan


def read_normalization(store_path, protocol):
    import zarr

    root = zarr.open_group(str(store_path), mode="r")
    arrays = {}
    for name in ("normalization_mean", "normalization_std"):
        raw = np.asarray(root[name][:], dtype=np.float32)
        digest = hashlib.sha256(raw.tobytes()).hexdigest()
        expected = protocol["normalization"]["mean_sha256" if name.endswith("mean") else "std_sha256"]
        if digest != expected:
            raise ValueError(f"{name} has {digest}, expected {expected}")
        arrays[name] = raw
    return arrays["normalization_mean"], arrays["normalization_std"]


def read_climatology(protocol):
    """The FY 2017-2021 train-only climatology, refit and checked against the reference.

    ``reference.json`` records the fit's provenance and the resulting per-lead
    climatology RMSE, but not the 48-bucket mean table, so the table is refit
    from the FY store's own declared train years and every declared field is
    checked against the frozen reference before it is used.
    """
    from data.r7_evaluation import fit_training_climatology

    path = ROOT / protocol["climatology"]["reference_path"]
    digest = sha256_file(path)
    if digest != protocol["climatology"]["reference_sha256"]:
        raise ValueError(f"climatology reference has {digest}, expected "
                         f"{protocol['climatology']['reference_sha256']}")
    body = json.loads(path.read_text(encoding="utf-8"))
    declared = body["leads"][str(min(protocol["leads"]))]["climatology"]
    table = fit_training_climatology(ROOT / protocol["normalization"]["source_store"])
    if table["kind"] != declared["kind"]:
        raise ValueError(f"climatology kind {table['kind']!r} != declared {declared['kind']!r}")
    if sorted(table["training_years"]) != sorted(declared["training_years"]):
        raise ValueError(f"climatology training years {table['training_years']} != "
                         f"declared {declared['training_years']}")
    if int(table["n_selected_steps"]) != int(declared["n_selected_steps"]):
        raise ValueError(f"climatology selected {table['n_selected_steps']} steps, "
                         f"declared {declared['n_selected_steps']}")
    return table


def assert_checkpoint_identity(checkpoint, protocol, arm):
    """The identity gate, run explicitly and for real (decision 0059).

    ``training/r7_evaluate.py`` compares a sibling ``train.jsonl``'s identity to
    the checkpoint contract. This instance has no sibling manifest, so the same
    comparison is made here against the FY arm's own ``train.jsonl``, computed in
    the FY directory where ``../cache.zarr`` is the FY store.
    """
    from training.r7_experiment import dataset_identity, load_checkpoint

    saved = load_checkpoint(checkpoint)
    contract = saved["contract"]
    manifest = ROOT / protocol["arms"][arm]["train_manifest"]
    identity, _ = dataset_identity(manifest)
    expected = protocol["arms"][arm]["data_identity"]
    if identity != expected:
        raise ValueError(f"{arm}: train.jsonl identity {identity} != protocol {expected}")
    if contract["data_identity"] != identity:
        raise ValueError(f"{arm}: checkpoint data_identity {contract['data_identity']} "
                         f"!= {identity} (checkpoint training data/normalization identity mismatch)")
    return saved, contract


def region_masks(height, width, margins):
    from training.r7_boundary_metrics import boundary_masks

    rename = {"full": "full", "interior_1": "interior", "edge_1": "edge"}
    regions = []
    for name, margin, mask in boundary_masks(height, width, tuple(margins)):
        if name not in rename:
            raise ValueError(f"unexpected boundary region {name!r} for margins {tuple(margins)}")
        regions.append((rename[name], margin, mask))
    return regions


def case_mse(prediction, target, std, latitude, masks):
    """Per-case latitude-weighted MSE per (lead, variable) for every region.

    ``prediction``/``target`` are normalized [1,L,C,H,W]; ``std`` restores the
    physical units. Returns ``{region: ndarray[L,C]}``.
    """
    import torch

    std_t = torch.as_tensor(std, dtype=torch.float64)
    delta = (prediction.detach().double() - target.detach().double())
    delta = delta * std_t[None, None, :, None, None]
    squared = delta.square()
    lat = torch.as_tensor(latitude, dtype=torch.float64)
    weight = torch.cos(torch.deg2rad(lat)).clamp_min(0)
    height, width = squared.shape[-2], squared.shape[-1]
    weight2d = weight[:, None].expand(height, width)
    out = {}
    for name, margin, mask in masks:
        if name == "full":
            selector = torch.ones(height, width, dtype=torch.bool)
        else:
            selector = torch.as_tensor(np.asarray(mask), dtype=torch.bool)
        values = squared[:, :, :, selector]
        weights = weight2d[selector]
        total = weights.sum()
        if float(total) < 1e-12:
            raise ValueError(f"region {name} has no usable area")
        out[name] = (values * weights[None, None, None, :]).sum(-1).div(total).squeeze(0).numpy()
    return out


def score_checkpoint(checkpoint, contract, dataset, climatology, std, masks, device_name):
    """Per-case model and climatology MSE for one checkpoint over the shared case set."""
    import torch
    from data.r7_evaluation import normalized_climatology
    from model.r7_rollout import autoregressive_rollout, rollout_model_input
    from training.r7_experiment import load_checkpoint, make_model, select_device

    device = select_device(device_name)
    model = make_model(contract["kind"], contract["model"])
    model.load_state_dict(load_checkpoint(checkpoint)["model"], strict=True)
    model = model.to(device).eval()
    inference = {} if contract["kind"] == "native" else {"reasoning_steps": contract["steps"]}
    model_mse = {name: [] for name in ("full", "interior", "edge")}
    clim_mse = {name: [] for name in ("full", "interior", "edge")}
    init_times = []
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            climate = normalized_climatology(
                climatology, sample["valid_times"], dataset.mean, dataset.std).unsqueeze(0)
            initial = rollout_model_input(sample, lead_hours=float(sample["lead_time_hours"]),
                                          device=device)
            trajectory = autoregressive_rollout(model, initial, lead_hours=dataset.lead_hours,
                                                step_hours=dataset.step_hours,
                                                history_interval_hours=dataset.step_hours,
                                                inference_kwargs=inference)
            prediction = trajectory.forecasts.cpu()
            target = sample["rollout_targets"].unsqueeze(0)
            model_regions = case_mse(prediction, target, std, sample["latitude"], masks)
            clim_regions = case_mse(climate, target, std, sample["latitude"], masks)
            for name in model_mse:
                model_mse[name].append(model_regions[name])
                clim_mse[name].append(clim_regions[name])
            init_times.append(sample["init_time"])
    return {
        "init_times": init_times,
        "model_mse": {name: np.stack(rows) for name, rows in model_mse.items()},
        "climatology_mse": {name: np.stack(rows) for name, rows in clim_mse.items()},
    }


def group_indices(init_times):
    """The declared groups: the whole set, the declared year, and the four seasons."""
    import pandas as pd

    stamps = pd.DatetimeIndex(init_times)
    seasons = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
    groups = {"total set": np.arange(len(stamps)), "declared year 2016": np.arange(len(stamps))}
    for name, months in seasons.items():
        groups[name] = np.nonzero(np.isin(stamps.month, months))[0]
    return groups


def aggregate(values, climatology, group):
    """Group-level RMSE and skill from the ratio of mean MSEs (protocol section 4)."""
    if group.size == 0:
        return {"mse_model": None, "mse_climatology": None, "rmse_model": None,
                "rmse_climatology": None, "skill": None, "status": "undefined_empty_group"}
    model_mean = float(values[group].mean())
    clim_mean = float(climatology[group].mean())
    if clim_mean <= 0:
        return {"mse_model": model_mean, "mse_climatology": clim_mean,
                "rmse_model": None, "skill": None, "status": "undefined_nonpositive_climatology"}
    return {"mse_model": model_mean, "mse_climatology": clim_mean,
            "rmse_model": model_mean ** 0.5, "rmse_climatology": clim_mean ** 0.5,
            "skill": 1.0 - model_mean / clim_mean, "status": "defined"}


def paired_interval(cells, block_length, resamples, seed, alpha):
    from training.r7_simultaneous_stats import simultaneous_paired_interval

    if not cells:
        return {"status": "insufficient-evidence", "reason": "no cells"}
    n_cases = len(next(iter(cells.values()))[0])
    if n_cases < int(block_length):
        return {"status": "insufficient-evidence", "n_cases": n_cases,
                "block_length": int(block_length),
                "reason": f"fewer cases ({n_cases}) than the declared block length"}
    return simultaneous_paired_interval(cells, block_length=block_length,
                                        resamples=resamples, seed=seed, alpha=alpha)


def build_readings(scored, climatology_by_lead, protocol):
    """All D4 readings: aggregates per group/variable/lead/region plus the intervals."""
    names = list(scored["treatment"]["names"])
    leads = list(protocol["leads"])
    groups = group_indices(scored["treatment"]["init_times"])
    arms = list(scored)
    readings = {"leads": leads, "variables": names, "groups": {k: int(v.size) for k, v in groups.items()},
                "arms": arms, "aggregates": {}, "guard": {}, "primary_family": {}}
    for arm in arms:
        readings["aggregates"][arm] = {}
        for region in ("full", "interior", "edge"):
            readings["aggregates"][arm][region] = {}
            for group, indices in groups.items():
                readings["aggregates"][arm][region][group] = {}
                for position, lead in enumerate(leads):
                    row = {}
                    for channel, variable in enumerate(names):
                        model = scored[arm]["model_mse"][region][:, :, position, channel].mean(0)
                        reference = scored[arm]["climatology_mse"][region][:, :, position, channel].mean(0)
                        row[variable] = aggregate(model, reference, indices)
                    readings["aggregates"][arm][region][group][str(lead)] = row
    return readings, groups, names, leads


def seed_mean(scored, arm, region):
    """Seed-averaged per-case MSE [cases, L, C] for one arm."""
    return scored[arm]["model_mse"][region].mean(0)


def primary_intervals(scored, groups, names, leads, protocol, region="full"):
    """Simultaneous intervals for the t2m primary family, one family per group.

    ``training/r7_simultaneous_stats.simultaneous_paired_interval`` requires every
    cell of a family to share one identical case-key set, so a family that mixes
    the total set with the four season groups is not expressible: the groups
    partition the cases. The protocol's family is therefore realised as one
    family per group (its five leads), which is the finest split that keeps a
    shared case set, and this is recorded in the readings.
    """
    candidate = seed_mean(scored, "treatment", region)
    channel = names.index("t2m")
    families = {}
    for group, indices in groups.items():
        keys = [scored["treatment"]["init_times"][i] for i in indices]
        reference = scored["treatment"]["climatology_mse"][region][0][indices, :, channel]
        cells = {}
        for position, lead in enumerate(leads):
            model = candidate[indices, position, channel]
            denom = reference[:, position]
            if np.any(denom <= 0):
                cells[f"{lead}h|t2m"] = (keys, np.zeros(len(indices)))
                continue
            cells[f"{lead}h|t2m"] = (keys, 1.0 - model / denom)
        families[group] = paired_interval(
            cells, protocol["block_length"], protocol["resamples"],
            protocol["resample_seed"], protocol["alpha_1"])
    return families


def guard_intervals(scored, groups, names, leads, protocol, region="full"):
    """Simultaneous upper bounds for the u10/v10/mslp incumbent comparison, per group."""
    candidate = seed_mean(scored, "treatment", region)
    incumbent = seed_mean(scored, "control", region)
    families = {}
    for group, indices in groups.items():
        keys = [scored["treatment"]["init_times"][i] for i in indices]
        cells = {}
        for position, lead in enumerate(leads):
            for variable in ("u10", "v10", "mslp"):
                channel = names.index(variable)
                focus = candidate[indices, position, channel]
                base = incumbent[indices, position, channel]
                if np.any(base <= 0):
                    continue
                cells[f"{lead}h|{variable}"] = (keys, focus / base - 1.0)
        families[group] = paired_interval(
            cells, protocol["block_length"], protocol["resamples"],
            protocol["resample_seed"], protocol["alpha_1"])
    return families


def sanitize(value, path="", report=None):
    """Replace non-finite floats with null and record where they were.

    Strict JSON has no NaN, and a non-finite reading is a fact about the run
    rather than something to round away, so the paths are kept in the output.
    """
    import math

    if report is None:
        report = []
    if isinstance(value, dict):
        return {key: sanitize(item, f"{path}.{key}", report) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item, f"{path}[{index}]", report) for index, item in enumerate(value)]
    if isinstance(value, float) and not math.isfinite(value):
        report.append(path)
        return None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="0 scores the whole declared case set")
    args = parser.parse_args(argv)
    from data.r7_evaluation import ZarrRolloutDataset

    protocol = load_protocol(args.protocol)
    if args.out.exists():
        raise FileExistsError(f"refusing existing output: {args.out}")
    store = ROOT / protocol["instance"]["frame_source_store"]
    mean, std = read_normalization(ROOT / protocol["normalization"]["source_store"], protocol)
    climatology = read_climatology(protocol)
    scan = make_scan(store, history_steps=protocol["case_set"]["history_steps"],
                     step_hours=protocol["case_set"]["step_hours"], mean=mean, std=std,
                     allowed_year=protocol["instance"]["target_year"])
    dataset = ZarrRolloutDataset(store, split="test", lead_hours=tuple(protocol["leads"]),
                                 history_steps=protocol["case_set"]["history_steps"],
                                 step_hours=protocol["case_set"]["step_hours"], scan=scan)
    if args.max_samples:
        dataset.windows = dataset.windows[:args.max_samples]
    masks = region_masks(int(np.asarray(scan.latitude).size), int(np.asarray(scan.longitude).size),
                         protocol["boundary_margins"])
    scored = {}
    for arm in ("treatment", "control"):
        scored[arm] = {"names": list(scan.names), "model_mse": {}, "climatology_mse": {}}
        per_seed = []
        for seed in protocol["arms"][arm]["seeds"]:
            checkpoint = ROOT / protocol["arms"][arm]["checkpoints"][str(seed)]
            _, contract = assert_checkpoint_identity(checkpoint, protocol, arm)
            result = score_checkpoint(checkpoint, contract, dataset, climatology, std, masks,
                                      args.device)
            result["seed"] = seed
            per_seed.append(result)
        scored[arm]["init_times"] = per_seed[0]["init_times"]
        for region in ("full", "interior", "edge"):
            scored[arm]["model_mse"][region] = np.stack(
                [r["model_mse"][region] for r in per_seed])
            scored[arm]["climatology_mse"][region] = np.stack(
                [r["climatology_mse"][region] for r in per_seed])
    readings, groups, names, leads = build_readings(scored, climatology, protocol)
    readings["primary_family"] = primary_intervals(scored, groups, names, leads, protocol)
    readings["guard"] = guard_intervals(scored, groups, names, leads, protocol)
    readings["family_note"] = ("one family per group: simultaneous_paired_interval requires every "
                               "cell of a family to share one identical case-key set, and the "
                               "groups partition the 2016 cases")
    readings["protocol_sha256"] = sha256_file(args.protocol)
    readings["scientific_claim"] = False
    readings["limitations"] = protocol["limitations"]
    non_finite = []
    readings = sanitize(readings, "", non_finite)
    readings["non_finite_paths"] = non_finite
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "readings.json").write_text(
        json.dumps(readings, ensure_ascii=False, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"readings": str(args.out / "readings.json"),
                      "cases": len(scored["treatment"]["init_times"]),
                      "groups": readings["groups"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
