from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TimeElapsedColumn, TimeRemainingColumn

from .common import build_dataloaders, lead_steps_to_tag, load_config, parse_data_config, save_json
from .external_models.urban_loader_adapter import move_batch_to_device
from .forecast_base import ForecastModelBase
from .forecast_models import build_baselines_from_config, build_forecast_baseline
from .metrics import BatchTensors, compute_batch_metrics, compute_batch_metrics_per_channel, default_clim, extract_lat

try:
    from ..model_complexity import forecast_model_complexity_summary
except ImportError:
    from model_complexity import forecast_model_complexity_summary


def _target_for_lead(batch: Dict, lead_index: int) -> torch.Tensor:
    if "y" in batch:
        return batch["y"][:, lead_index].float()
    return batch["x0"].float()


def _model_device(model: ForecastModelBase) -> torch.device:
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


def model_complexity_summary(
    model: ForecastModelBase,
    batch: Dict | None = None,
    lead_times: Sequence[int] | None = None,
) -> Dict[str, Any]:
    return forecast_model_complexity_summary(model, batch=batch, lead_times=lead_times)


@torch.no_grad()
def evaluate_forecast_baseline(
    model: ForecastModelBase,
    dl,
    *,
    lead_times: Sequence[int],
    time_step_hours: float,
    var_names: Sequence[str],
    progress_desc: str,
) -> Dict:
    model.eval()
    device = _model_device(model)
    complexity: Dict[str, float | int | str] | None = None
    sums: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, Dict[str, int]] = {}
    channel_sums: Dict[str, Dict[str, torch.Tensor]] = {}

    try:
        total = len(dl)
    except Exception:
        total = None

    with Progress(BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), TimeRemainingColumn(), transient=False) as progress:
        task = progress.add_task(progress_desc, total=total)
        for batch in dl:
            batch = move_batch_to_device(batch, device)
            if complexity is None:
                complexity = model_complexity_summary(model, batch, lead_times=lead_times)
            pred = model.predict(batch, lead_times=lead_times)
            pred_ensemble = None
            if isinstance(pred, torch.Tensor) and pred.ndim == 6:
                pred_ensemble = pred.float()
                pred_mean = pred_ensemble.mean(dim=1)
            elif isinstance(pred, torch.Tensor) and pred.ndim == 5:
                pred_mean = pred.float()
            else:
                raise ValueError(
                    "model.predict must return [B,L,C,H,W] or [B,E,L,C,H,W], "
                    f"got {getattr(pred, 'shape', None)}"
                )
            mean = batch["norm"]["mean"].float()
            std = batch["norm"]["std"].float()

            for li, lt in enumerate(lead_times):
                tag = lead_steps_to_tag(int(lt), time_step_hours=time_step_hours)
                target = _target_for_lead(batch, li)
                pred_ensemble_li = pred_ensemble[:, :, li].float() if pred_ensemble is not None else None
                bt = BatchTensors(
                    pred_norm=pred_mean[:, li].float(),
                    pred_ensemble_norm=pred_ensemble_li,
                    target_norm=target,
                    mean=mean,
                    std=std,
                    clim_denorm=default_clim(batch),
                    lat=extract_lat(batch),
                )
                metric = compute_batch_metrics(bt)
                sums.setdefault(tag, {"rmse": 0.0, "mae": 0.0, "bias": 0.0, "crps": 0.0, "acc": 0.0})
                counts.setdefault(tag, {"rmse": 0, "mae": 0, "bias": 0, "crps": 0, "acc": 0})
                for key, value in metric.items():
                    if torch.isfinite(value):
                        sums[tag][key] += float(value)
                        counts[tag][key] += 1

                per_ch = compute_batch_metrics_per_channel(bt)
                channel_sums.setdefault(tag, {})
                for key in ("rmse_ch", "mae_ch", "bias_ch", "crps_ch", "acc_ch"):
                    if key not in per_ch:
                        continue
                    value = per_ch[key].detach().cpu()
                    if key not in channel_sums[tag]:
                        channel_sums[tag][key] = value
                    else:
                        channel_sums[tag][key] += value
            progress.advance(task)

    out: Dict = {}
    denom_batches = max(len(dl), 1)
    for lt in lead_times:
        tag = lead_steps_to_tag(int(lt), time_step_hours=time_step_hours)
        item = {
            "lead_steps": int(lt),
            "lead_tag": tag,
            "RMSE": sums[tag]["rmse"] / max(counts[tag]["rmse"], 1),
            "MAE": sums[tag]["mae"] / max(counts[tag]["mae"], 1),
            "Bias": sums[tag]["bias"] / max(counts[tag]["bias"], 1),
            "CRPS": sums[tag]["crps"] / max(counts[tag]["crps"], 1),
        }
        if counts[tag]["acc"] > 0:
            item["ACC"] = sums[tag]["acc"] / max(counts[tag]["acc"], 1)
        ch = channel_sums.get(tag, {})
        if "rmse_ch" in ch:
            rmse = (ch["rmse_ch"] / denom_batches).numpy().tolist()
            mae = (ch["mae_ch"] / denom_batches).numpy().tolist()
            bias = (ch["bias_ch"] / denom_batches).numpy().tolist()
            crps = (ch["crps_ch"] / denom_batches).numpy().tolist()
            item["RMSE_per_var"] = {name: rmse[i] for i, name in enumerate(var_names)}
            item["MAE_per_var"] = {name: mae[i] for i, name in enumerate(var_names)}
            item["Bias_per_var"] = {name: bias[i] for i, name in enumerate(var_names)}
            item["CRPS_per_var"] = {name: crps[i] for i, name in enumerate(var_names)}
        if "acc_ch" in ch:
            acc = (ch["acc_ch"] / denom_batches).numpy().tolist()
            item["ACC_per_var"] = {name: acc[i] for i, name in enumerate(var_names)}
        out[tag] = item
    out["__model_complexity__"] = complexity or model_complexity_summary(model)
    return out


def _all_model_specs(cfg: Dict) -> Dict[str, ForecastModelBase]:
    models = build_baselines_from_config(cfg)
    baseline_cfg = dict(cfg.get("baselines", {}) or {})
    perturb = baseline_cfg.get("perturbation_models", []) or []
    if not perturb:
        return models

    data = dict(cfg.get("data", {}) or {})
    dyn = list(data.get("dynamic_vars", cfg.get("dynamic_vars", [])) or [])
    stat = list(data.get("static_vars", cfg.get("static_vars", [])) or [])
    k = int(data.get("k", cfg.get("k", 1)))
    forecast = dict(cfg.get("forecast", {}) or {})
    lead_times = forecast.get("lead_times", cfg.get("lead_times", None))
    default_policy = str(baseline_cfg.get("static_policy", "dynamic_only"))

    for item in perturb:
        name = str(item["name"])
        alias = str(item.get("alias", name))
        policy = str(item.get("static_policy", default_policy))
        params = dict(item.get("params", {}) or {})
        models[alias] = build_forecast_baseline(
            name,
            dynamic_vars=dyn,
            static_vars=stat,
            k=k,
            lead_times=lead_times,
            static_policy=policy,
            params=params,
        )
    return models


def run_forecast_baselines(*, config_path: str | Path, out_dir: str | Path = "outputs/baselines/fair") -> Dict:
    cfg = load_config(config_path)
    data_cfg = parse_data_config(cfg)
    _, val_dl, test_dl = build_dataloaders(data_cfg, shuffle_train=False)
    lead_times = data_cfg.lead_times or [data_cfg.delta_t]
    models = _all_model_specs(cfg)

    results: Dict = {
        "config": str(config_path),
        "protocol": dict(cfg.get("baselines", {}) or {}).get("protocol", "forecast_baselines"),
        "lead_times": lead_times,
        "static_schema": data_cfg.static_schema,
        "models": {},
    }

    for alias, model in models.items():
        results["models"][alias] = {
            "name": model.spec.name,
            "family": model.spec.family,
            "uses_static": bool(model.spec.uses_static),
            "static_policy": model.static_policy,
            "val": evaluate_forecast_baseline(
                model,
                val_dl,
                lead_times=lead_times,
                time_step_hours=data_cfg.time_step_hours,
                var_names=data_cfg.dynamic_vars,
                progress_desc=f"val {alias}",
            ),
            "test": evaluate_forecast_baseline(
                model,
                test_dl,
                lead_times=lead_times,
                time_step_hours=data_cfg.time_step_hours,
                var_names=data_cfg.dynamic_vars,
                progress_desc=f"test {alias}",
            ),
        }

    out_dir = Path(out_dir)
    save_json(results, out_dir / "results.json")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fair static-information controlled forecast baselines")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--out_dir", type=str, default="outputs/baselines/fair")
    args = parser.parse_args()
    results = run_forecast_baselines(config_path=args.config, out_dir=args.out_dir)
    import json

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()