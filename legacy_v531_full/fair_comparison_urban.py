r"""
UrbanPiDiT 统一评估与 schema 导出。

本模块负责把 UrbanPiDiT 的推理结果转换为与基线一致的指标结构，并在首个
batch 上测量实际参数量与 Conv2d/Linear FLOPs。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import yaml
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TimeElapsedColumn, TimeRemainingColumn

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    import pytorch_lightning as pl
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError("fair_comparison_urban.py 需要安装 pytorch_lightning。") from exc

try:
    from .baselines.common import lead_steps_to_tag
    from .baselines.external_models.urban_loader_adapter import move_batch_to_device
    from .baselines.metrics import (
        BatchTensors,
        compute_batch_metrics,
        compute_batch_metrics_per_channel,
        default_clim,
        extract_lat,
    )
    from .data.loader import MetroWeatherDataModule
    from .model_complexity import urban_pidit_complexity_summary
    from .pidit_lit import UrbanPiDiTLitModule
    from .utils.config_builder import build_datamodule_kwargs, build_litmodule_kwargs
except ImportError:
    from baselines.common import lead_steps_to_tag
    from baselines.external_models.urban_loader_adapter import move_batch_to_device
    from baselines.metrics import (
        BatchTensors,
        compute_batch_metrics,
        compute_batch_metrics_per_channel,
        default_clim,
        extract_lat,
    )
    from data.loader import MetroWeatherDataModule
    from model_complexity import urban_pidit_complexity_summary
    from pidit_lit import UrbanPiDiTLitModule
    from utils.config_builder import build_datamodule_kwargs, build_litmodule_kwargs


METRICS = ("RMSE", "MAE", "Bias", "CRPS", "ACC")


@dataclass(frozen=True)
class UrbanEvalOptions:
    lead_times: Sequence[int]
    time_step_hours: float
    var_names: Sequence[str]
    split: str = "test"
    compute_complexity: bool = True


@dataclass(frozen=True)
class UrbanPredictContext:
    lit: UrbanPiDiTLitModule
    x_ctx: torch.Tensor
    leads: Sequence[int]
    static_kwargs: Dict[str, Optional[torch.Tensor]]
    infer: Dict[str, Any]
    batch_idx: int
    steps: int
    t_start: float
    t_end: float
    is_ensemble: bool
    ensemble_size: int


def load_yaml(path: str | Path) -> Dict[str, Any]:
    r"""
    读取 YAML 配置。

    Parameters
    ----
    path : str or Path
        YAML 文件路径。

    Returns
    ----
    Dict[str, Any]
        配置字典。
    """

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _lead_list(batch: Dict[str, Any]) -> Optional[List[int]]:
    raw = batch.get("lead_times", None)
    if raw is None:
        return None
    if isinstance(raw, torch.Tensor):
        if raw.ndim == 2:
            raw = raw[0]
        return [int(x) for x in raw.detach().cpu().tolist()]
    try:
        return [int(x) for x in list(raw)]
    except Exception:
        return None


def _target_for_lead(batch: Dict[str, Any], lead: int, fallback_index: int) -> torch.Tensor:
    leads = _lead_list(batch)
    if "y" in batch and leads is not None and int(lead) in leads:
        return batch["y"][:, int(leads.index(int(lead)))].float()
    if "y" in batch:
        return batch["y"][:, int(fallback_index)].float()
    return batch["x0"].float()


def _eval_leads(requested: Sequence[int], batch: Dict[str, Any]) -> List[int]:
    leads = [int(x) for x in requested]
    available = _lead_list(batch)
    if available:
        leads = [lead for lead in leads if lead in available]
        if not leads:
            leads = [int(available[0])]
    return sorted(list(dict.fromkeys(leads)))


def _inference_cfg(lit: UrbanPiDiTLitModule, split: str) -> Dict[str, Any]:
    infer = dict(lit._get_infer_cfg())
    key = "val_mode" if split == "val" else "test_mode"
    infer["active_mode"] = str(infer.get(key, infer.get("mode", "deterministic"))).lower()
    return infer


def _static_kwargs(lit: UrbanPiDiTLitModule, batch: Dict[str, Any]) -> Dict[str, Optional[torch.Tensor]]:
    return dict(lit._model_static_kwargs(lit._move_static_batch(batch)))


def _lead_condition(lit: UrbanPiDiTLitModule, x_ctx: torch.Tensor, lead: int) -> Optional[torch.Tensor]:
    if not bool(getattr(lit, "use_lead_time_conditioning", False)):
        return None
    raw = torch.full((x_ctx.size(0),), float(int(lead)), device=x_ctx.device, dtype=torch.float32)
    return lit._normalize_lead_time(raw)


def _diffusion_params(lit: UrbanPiDiTLitModule) -> Tuple[int, float, float]:
    cfg = dict(lit.hparams.get("diffusion_cfg", {}) or {})
    return int(cfg.get("sample_steps", 8)), float(cfg.get("t_start", 0.0)), float(cfg.get("t_end", 1.0))


def _make_predict_context(
    lit: UrbanPiDiTLitModule,
    batch: Dict[str, Any],
    leads: Sequence[int],
    batch_idx: int,
    split: str,
) -> UrbanPredictContext:
    infer = _inference_cfg(lit, split)
    mode = str(infer.get("active_mode", "deterministic"))
    steps, t_start, t_end = _diffusion_params(lit)
    return UrbanPredictContext(
        lit=lit,
        x_ctx=batch["x_ctx"],
        leads=leads,
        static_kwargs=_static_kwargs(lit, batch),
        infer=infer,
        batch_idx=int(batch_idx),
        steps=steps,
        t_start=t_start,
        t_end=t_end,
        is_ensemble=mode in {"ensemble", "ens", "probabilistic"},
        ensemble_size=int(infer.get("ensemble_size", 20)),
    )


def _seed_base(ctx: UrbanPredictContext) -> Tuple[int, int, int]:
    fixed_seed = int(ctx.infer.get("fixed_seed", 42))
    base_seed = int(ctx.infer.get("base_seed", fixed_seed))
    offset = int(ctx.infer.get("seed_offset_per_batch", 100000)) * int(ctx.batch_idx)
    return fixed_seed, base_seed, offset


def _predict_single(ctx: UrbanPredictContext, x_ctx: torch.Tensor, lead: int, seed: int) -> torch.Tensor:
    return ctx.lit.predict_one(
        x_ctx,
        steps=ctx.steps,
        t_start=ctx.t_start,
        t_end=ctx.t_end,
        seed=int(seed),
        lead_time=_lead_condition(ctx.lit, x_ctx, int(lead)),
        **ctx.static_kwargs,
    )


def _predict_members(ctx: UrbanPredictContext, x_ctx: torch.Tensor, lead: int, seed_offset: int) -> torch.Tensor:
    _, base_seed, _ = _seed_base(ctx)
    return ctx.lit.predict_ensemble(
        x_ctx,
        ensemble_size=ctx.ensemble_size,
        base_seed=int(base_seed),
        steps=ctx.steps,
        t_start=ctx.t_start,
        t_end=ctx.t_end,
        seed_offset=int(seed_offset),
        lead_time=_lead_condition(ctx.lit, x_ctx, int(lead)),
        **ctx.static_kwargs,
    )


def _predict_direct(ctx: UrbanPredictContext) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    fixed_seed, _, batch_offset = _seed_base(ctx)
    preds: Dict[int, torch.Tensor] = {}
    ensembles: Dict[int, torch.Tensor] = {}
    for lead in ctx.leads:
        lead_offset = 0 if bool(getattr(ctx.lit, "correlate_noise_across_leads", False)) else int(lead) * 1000
        if ctx.is_ensemble:
            ens = _predict_members(ctx, ctx.x_ctx, int(lead), batch_offset + lead_offset)
            preds[int(lead)] = ens.mean(dim=1)
            ensembles[int(lead)] = ens
            continue
        preds[int(lead)] = _predict_single(ctx, ctx.x_ctx, int(lead), fixed_seed + batch_offset + lead_offset)
    return preds, ensembles


def _predict_autoregressive(ctx: UrbanPredictContext) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    fixed_seed, _, batch_offset = _seed_base(ctx)
    preds: Dict[int, torch.Tensor] = {}
    ensembles: Dict[int, torch.Tensor] = {}
    x_ctx = ctx.x_ctx
    for step in range(1, max(int(x) for x in ctx.leads) + 1):
        step_offset = batch_offset + step * 1000
        if ctx.is_ensemble:
            ens = _predict_members(ctx, x_ctx, 1, step_offset)
            x_pred = ens.mean(dim=1)
        else:
            ens = None
            x_pred = _predict_single(ctx, x_ctx, 1, fixed_seed + step_offset)
        if step in ctx.leads:
            preds[int(step)] = x_pred
            if ens is not None:
                ensembles[int(step)] = ens
        x_ctx = ctx.lit._update_ctx_autoregressive(x_ctx, x_pred)
    return preds, ensembles


def _predict_by_lead(ctx: UrbanPredictContext) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    mode = str(getattr(ctx.lit, "multi_horizon_inference", "direct")).lower()
    if mode == "direct" and not bool(getattr(ctx.lit, "use_lead_time_conditioning", False)):
        mode = "autoregressive"
    if mode == "autoregressive":
        return _predict_autoregressive(ctx)
    return _predict_direct(ctx)


def _accumulate_scalar(sums: Dict[str, Dict[str, float]], counts: Dict[str, Dict[str, int]], tag: str, metric: Dict[str, torch.Tensor]) -> None:
    sums.setdefault(tag, {key.lower(): 0.0 for key in METRICS})
    counts.setdefault(tag, {key.lower(): 0 for key in METRICS})
    for key in ("rmse", "mae", "bias", "crps", "acc"):
        value = metric.get(key, None)
        if value is not None and torch.isfinite(value):
            sums[tag][key] += float(value.detach().cpu())
            counts[tag][key] += 1


def _accumulate_channel(channel_sums: Dict[str, Dict[str, torch.Tensor]], tag: str, per_ch: Dict[str, torch.Tensor]) -> None:
    channel_sums.setdefault(tag, {})
    for key in ("rmse_ch", "mae_ch", "bias_ch", "crps_ch", "acc_ch"):
        value = per_ch.get(key, None)
        if value is None:
            continue
        value = value.detach().cpu()
        if key not in channel_sums[tag]:
            channel_sums[tag][key] = value
        else:
            channel_sums[tag][key] += value


def _attach_channel_metrics(item: Dict[str, Any], ch: Dict[str, torch.Tensor], denom: int, var_names: Sequence[str]) -> None:
    mapping = {
        "rmse_ch": "RMSE_per_var",
        "mae_ch": "MAE_per_var",
        "bias_ch": "Bias_per_var",
        "crps_ch": "CRPS_per_var",
        "acc_ch": "ACC_per_var",
    }
    for source, target in mapping.items():
        if source not in ch:
            continue
        values = (ch[source] / max(int(denom), 1)).numpy().tolist()
        item[target] = {name: values[i] for i, name in enumerate(var_names) if i < len(values)}


def _finalize_metrics(state: Dict[str, Any], options: UrbanEvalOptions) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for lead in options.lead_times:
        tag = lead_steps_to_tag(int(lead), time_step_hours=options.time_step_hours)
        if tag not in state["sums"]:
            continue
        item = {"lead_steps": int(lead), "lead_tag": tag}
        for label, key in (("RMSE", "rmse"), ("MAE", "mae"), ("Bias", "bias"), ("CRPS", "crps"), ("ACC", "acc")):
            count = max(int(state["counts"][tag].get(key, 0)), 1)
            item[label] = state["sums"][tag][key] / count
        denom = int(state.get("channel_counts", {}).get(tag, state.get("num_batches", 1)))
        _attach_channel_metrics(item, state["channel_sums"].get(tag, {}), denom, options.var_names)
        out[tag] = item
    return out


def _accumulate_urban_batch(
    state: Dict[str, Any],
    batch: Dict[str, Any],
    preds: Dict[int, torch.Tensor],
    ensembles: Dict[int, torch.Tensor],
    leads: List[int],
    options: UrbanEvalOptions,
) -> None:
    mean = batch["norm"]["mean"].float()
    std = batch["norm"]["std"].float()
    for li, lead in enumerate(leads):
        tag = lead_steps_to_tag(int(lead), time_step_hours=options.time_step_hours)
        bt = BatchTensors(
            pred_norm=preds[int(lead)].float(),
            pred_ensemble_norm=ensembles.get(int(lead), None),
            target_norm=_target_for_lead(batch, int(lead), li),
            mean=mean,
            std=std,
            clim_denorm=default_clim(batch),
            lat=extract_lat(batch),
        )
        _accumulate_scalar(state["sums"], state["counts"], tag, compute_batch_metrics(bt))
        _accumulate_channel(state["channel_sums"], tag, compute_batch_metrics_per_channel(bt))
        state["channel_counts"][tag] = int(state["channel_counts"].get(tag, 0)) + 1
    state["num_batches"] += 1


@torch.no_grad()
def evaluate_urban_pidit(lit: UrbanPiDiTLitModule, dl: Any, options: UrbanEvalOptions) -> Dict[str, Any]:
    r"""
    评估 UrbanPiDiT 并导出基线兼容 schema。

    Parameters
    ----
    lit : UrbanPiDiTLitModule
        已加载 checkpoint 的 LightningModule。
    dl : Any
        DataLoader。
    options : UrbanEvalOptions
        评估配置。

    Returns
    ----
    Dict[str, Any]
        统一 schema 的 UrbanPiDiT 结果。
    """

    lit.eval()
    state: Dict[str, Any] = {"sums": {}, "counts": {}, "channel_sums": {}, "channel_counts": {}, "num_batches": 0}
    complexity = None
    total = len(dl) if hasattr(dl, "__len__") else None
    with Progress(BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), TimeRemainingColumn(), transient=False) as progress:
        task = progress.add_task(f"{options.split} UrbanPiDiT", total=total)
        for batch_idx, raw_batch in enumerate(dl):
            batch = move_batch_to_device(raw_batch, lit.device)
            leads = _eval_leads(options.lead_times, batch)
            if complexity is None and options.compute_complexity:
                complexity = urban_pidit_complexity_summary(lit, batch, lead_times=leads)
            preds, ensembles = _predict_by_lead(_make_predict_context(lit, batch, leads, batch_idx, options.split))
            _accumulate_urban_batch(state, batch, preds, ensembles, leads, options)
            progress.advance(task)
    return {
        "model_name": "UrbanPiDiT",
        "alias": "UrbanPiDiT",
        "family": "diffusion_transformer",
        "static_policy": "full_static",
        "split": options.split,
        "complexity": complexity or {},
        "metrics": _finalize_metrics(state, options),
    }


def build_urban_result(config_path: str | Path, ckpt_path: str | Path, device: torch.device) -> Dict[str, Any]:
    r"""
    构建 UrbanPiDiT 的统一结果。

    Parameters
    ----
    config_path : str or Path
        UrbanPiDiT 配置文件。
    ckpt_path : str or Path
        Lightning checkpoint。
    device : torch.device
        运行设备。

    Returns
    ----
    Dict[str, Any]
        UrbanPiDiT 统一结果。
    """

    cfg = load_yaml(config_path)
    pl.seed_everything(int(cfg.get("train", {}).get("seed", 42)), workers=True)
    eval_cfg = dict(cfg.get("eval", {}) or {})
    train_cfg = dict(cfg.get("train", {}) or {})
    batch_size = int(eval_cfg.get("batch_size", train_cfg.get("batch_size", 1)))
    num_workers = int(eval_cfg.get("num_workers", train_cfg.get("num_workers", 8)))
    dm = MetroWeatherDataModule(**build_datamodule_kwargs(cfg, batch_size=batch_size, num_workers=num_workers))
    dm.prepare_data()
    dm.setup(stage="test")
    lit = UrbanPiDiTLitModule.load_from_checkpoint(str(ckpt_path), **build_litmodule_kwargs(cfg)).to(device)
    forecast = dict(cfg.get("forecast", {}) or {})
    data_cfg = dict(cfg.get("data", {}) or {})
    default_delta_t = data_cfg.get("delta_t", cfg.get("delta_t", 1))
    leads = [int(x) for x in forecast.get("eval_lead_times", forecast.get("lead_times", [default_delta_t]))]
    var_names = list(data_cfg.get("dynamic_vars", cfg.get("dynamic_vars", [])) or [])
    options = UrbanEvalOptions(leads, float(forecast.get("time_step_hours", 6.0)), var_names)
    return evaluate_urban_pidit(lit, dm.test_dataloader(), options)