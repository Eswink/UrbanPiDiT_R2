#!/usr/bin/env python3
"""UrbanPiDiT V5.3.1 morphology counterfactual intervention suite.

该脚本执行 reviewer-facing 的反事实干预诊断：构建原始 dummy batch，应用形态与风场干预，
实际运行模型 forward，并输出 prediction/proxy/process-graph delta。输出只作为 sensitivity
intervention diagnostic，不声明因果证明。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_v531_reviewer_evidence_checks import load_config, make_dummy_batch, to_jsonable, write_report
from models.model_registry import build_model_v531_from_config, dropped_model_kwargs, filter_model_kwargs
from urbanpidit_version import CLAIM_BOUNDARY, RELEASE_NAME, VERSION, VERSION_NAME
from utils.config_builder import build_model_cfg


TensorBatch = Dict[str, torch.Tensor]
InterventionFn = Callable[[TensorBatch, Mapping[str, Any]], TensorBatch]


@dataclass(frozen=True)
class CounterfactualCase:
    """一次反事实干预定义。"""

    name: str
    description: str
    apply: InterventionFn


def clone_batch(batch: Mapping[str, torch.Tensor]) -> TensorBatch:
    return {key: value.clone() for key, value in batch.items()}


def history_steps(model_cfg: Mapping[str, Any]) -> int:
    dyn_count = max(len(model_cfg.get("dynamic_vars", []) or []), 1)
    dynamic_ctx = int(model_cfg.get("ctx_channels", 0)) - int(model_cfg.get("static_channels", 0))
    return max(dynamic_ctx // dyn_count, 1)


def sync_static_raw(batch: TensorBatch, model_cfg: Mapping[str, Any]) -> TensorBatch:
    static_channels = int(model_cfg.get("static_channels", 0))
    if static_channels <= 0 or "static_raw" not in batch or "x_ctx" not in batch:
        return batch
    static_raw = batch["static_raw"].to(device=batch["x_ctx"].device, dtype=batch["x_ctx"].dtype)
    if static_raw.size(1) < static_channels:
        pad = torch.zeros(
            static_raw.size(0),
            static_channels - static_raw.size(1),
            static_raw.size(2),
            static_raw.size(3),
            device=static_raw.device,
            dtype=static_raw.dtype,
        )
        static_raw = torch.cat([static_raw, pad], dim=1)
    static_raw = static_raw[:, :static_channels]
    x_ctx = batch["x_ctx"].clone()
    x_ctx[:, -static_channels:] = static_raw
    batch["static_raw"] = static_raw
    batch["x_ctx"] = x_ctx
    return batch


def rebuild_static_raw(batch: TensorBatch, model_cfg: Mapping[str, Any]) -> TensorBatch:
    parts = []
    if batch.get("static_cat") is not None and batch["static_cat"].numel() > 0:
        parts.append(batch["static_cat"].float())
    if batch.get("static_cont") is not None and batch["static_cont"].numel() > 0:
        parts.append(batch["static_cont"].float())
    if parts:
        batch["static_raw"] = torch.cat(parts, dim=1)
    return sync_static_raw(batch, model_cfg)


def zero_static(batch: TensorBatch, model_cfg: Mapping[str, Any]) -> TensorBatch:
    out = clone_batch(batch)
    for key in ("static_raw", "static_cont", "static_cat"):
        if key in out and out[key].numel() > 0:
            out[key] = torch.zeros_like(out[key])
    return sync_static_raw(out, model_cfg)


def shuffle_static(batch: TensorBatch, model_cfg: Mapping[str, Any]) -> TensorBatch:
    out = clone_batch(batch)
    batch_size = int(out["x_t"].size(0))
    if batch_size > 1:
        perm = torch.arange(batch_size - 1, -1, -1, device=out["x_t"].device)
        for key in ("static_raw", "static_cont", "static_cat"):
            if key in out and out[key].numel() > 0:
                out[key] = out[key][perm]
    else:
        for key in ("static_raw", "static_cont", "static_cat"):
            if key in out and out[key].numel() > 0:
                out[key] = torch.roll(out[key], shifts=1, dims=-1)
    return sync_static_raw(out, model_cfg)


def permute_landcover(batch: TensorBatch, model_cfg: Mapping[str, Any]) -> TensorBatch:
    out = clone_batch(batch)
    schema = dict(model_cfg.get("static_schema", {}) or {})
    cardinality = int(dict(schema.get("categorical_cardinality", {}) or {}).get("landcover", 20))
    if out.get("static_cat") is not None and out["static_cat"].numel() > 0:
        out["static_cat"] = (out["static_cat"].long() + 1) % max(cardinality, 1)
    if out.get("static_raw") is not None and out["static_raw"].size(1) > 0:
        out["static_raw"][:, 0:1] = out.get("static_cat", out["static_raw"][:, 0:1]).float()
    return sync_static_raw(out, model_cfg)


def remove_static_cont_channel(batch: TensorBatch, model_cfg: Mapping[str, Any], cont_index: int) -> TensorBatch:
    out = clone_batch(batch)
    if out.get("static_cont") is not None and out["static_cont"].size(1) > cont_index:
        out["static_cont"][:, cont_index : cont_index + 1] = 0.0
    raw_index = cont_index + 1
    if out.get("static_raw") is not None and out["static_raw"].size(1) > raw_index:
        out["static_raw"][:, raw_index : raw_index + 1] = 0.0
    return sync_static_raw(out, model_cfg)


def remove_building_volume(batch: TensorBatch, model_cfg: Mapping[str, Any]) -> TensorBatch:
    return remove_static_cont_channel(batch, model_cfg, cont_index=2)


def remove_population(batch: TensorBatch, model_cfg: Mapping[str, Any]) -> TensorBatch:
    return remove_static_cont_channel(batch, model_cfg, cont_index=3)


def set_roughness_channels(batch: TensorBatch, model_cfg: Mapping[str, Any], *, high: bool) -> TensorBatch:
    out = clone_batch(batch)
    value = 2.0 if high else 0.0
    if out.get("static_cont") is not None and out["static_cont"].numel() > 0:
        channels = min(3, out["static_cont"].size(1))
        if high:
            out["static_cont"][:, :channels] = out["static_cont"][:, :channels].abs() + value
        else:
            out["static_cont"][:, :channels] = value
    if out.get("static_raw") is not None and out["static_raw"].size(1) > 1:
        channels = min(3, out["static_raw"].size(1) - 1)
        if high:
            out["static_raw"][:, 1 : 1 + channels] = out["static_raw"][:, 1 : 1 + channels].abs() + value
        else:
            out["static_raw"][:, 1 : 1 + channels] = value
    return sync_static_raw(out, model_cfg)


def high_roughness_intervention(batch: TensorBatch, model_cfg: Mapping[str, Any]) -> TensorBatch:
    return set_roughness_channels(batch, model_cfg, high=True)


def low_roughness_intervention(batch: TensorBatch, model_cfg: Mapping[str, Any]) -> TensorBatch:
    return set_roughness_channels(batch, model_cfg, high=False)


def rotate_wind_90(batch: TensorBatch, model_cfg: Mapping[str, Any]) -> TensorBatch:
    out = clone_batch(batch)
    dynamic_vars = list(model_cfg.get("dynamic_vars", []) or [])
    if "u10" not in dynamic_vars or "v10" not in dynamic_vars:
        return out
    steps = history_steps(model_cfg)
    u_base = int(dynamic_vars.index("u10")) * steps
    v_base = int(dynamic_vars.index("v10")) * steps
    x_ctx = out["x_ctx"].clone()
    for lag in range(steps):
        u_idx = u_base + lag
        v_idx = v_base + lag
        if max(u_idx, v_idx) >= x_ctx.size(1):
            continue
        u = x_ctx[:, u_idx].clone()
        v = x_ctx[:, v_idx].clone()
        x_ctx[:, u_idx] = -v
        x_ctx[:, v_idx] = u
    out["x_ctx"] = x_ctx
    return out


def intervention_cases() -> list[CounterfactualCase]:
    return [
        CounterfactualCase("zero_static", "Set all static morphology channels to zero.", zero_static),
        CounterfactualCase("shuffle_static", "Shuffle static morphology fields across batch or spatially roll single-sample fields.", shuffle_static),
        CounterfactualCase("permute_landcover", "Shift categorical landcover ids while keeping dynamic history fixed.", permute_landcover),
        CounterfactualCase("remove_building_volume", "Zero the building_volume continuous morphology channel.", remove_building_volume),
        CounterfactualCase("remove_population", "Zero the population continuous morphology channel.", remove_population),
        CounterfactualCase("rotate_wind_90", "Rotate historical u10/v10 context by 90 degrees.", rotate_wind_90),
        CounterfactualCase("high_roughness_intervention", "Increase building-related roughness proxy source channels.", high_roughness_intervention),
        CounterfactualCase("low_roughness_intervention", "Suppress building-related roughness proxy source channels.", low_roughness_intervention),
    ]


def selected_cases(names: Optional[list[str]]) -> list[CounterfactualCase]:
    cases = intervention_cases()
    if not names:
        return cases
    wanted = set(names)
    selected = [case for case in cases if case.name in wanted]
    missing = sorted(wanted.difference({case.name for case in selected}))
    if missing:
        raise ValueError(f"未知 counterfactual intervention: {', '.join(missing)}")
    return selected


def concat_batches(batches: list[TensorBatch]) -> TensorBatch:
    keys = batches[0].keys()
    return {key: torch.cat([batch[key] for batch in batches], dim=0) for key in keys}


def tensor_delta(left: torch.Tensor, right: torch.Tensor) -> Dict[str, Any]:
    diff = (right.detach().float() - left.detach().float()).cpu()
    base_norm = left.detach().float().cpu().norm().clamp_min(1e-12)
    return {
        "norm": diff.norm().item(),
        "mean_abs": diff.abs().mean().item(),
        "max_abs": diff.abs().max().item(),
        "relative_norm": (diff.norm() / base_norm).item(),
    }


def proxy_fields(model: torch.nn.Module, batch: TensorBatch) -> Optional[Dict[str, torch.Tensor]]:
    encoder = getattr(model, "process_proxy_encoder", None)
    if encoder is None:
        return None
    out = encoder(
        batch["x_t"],
        static_raw=batch.get("static_raw"),
        static_cont=batch.get("static_cont"),
        static_cat=batch.get("static_cat"),
        hour_of_day=batch.get("hour_of_day"),
    )
    return out.proxy_fields


def fields_delta(left: Optional[Mapping[str, torch.Tensor]], right: Optional[Mapping[str, torch.Tensor]]) -> Dict[str, Any]:
    if not left or not right:
        return {"available": False, "norm": 0.0, "mean_abs": 0.0, "max_abs": 0.0, "per_field_norm": {}}
    keys = sorted(set(left).intersection(right))
    if not keys:
        return {"available": False, "norm": 0.0, "mean_abs": 0.0, "max_abs": 0.0, "per_field_norm": {}}
    diffs = [(right[key].detach().float() - left[key].detach().float()).flatten() for key in keys]
    joined = torch.cat([diff.cpu() for diff in diffs])
    return {
        "available": True,
        "norm": joined.norm().item(),
        "mean_abs": joined.abs().mean().item(),
        "max_abs": joined.abs().max().item(),
        "per_field_norm": {key: diffs[idx].cpu().norm().item() for idx, key in enumerate(keys)},
    }


def process_graph_adjacency(
    model: torch.nn.Module,
    batch: TensorBatch,
    model_cfg: Mapping[str, Any],
    proxies: Optional[Mapping[str, torch.Tensor]],
) -> tuple[Optional[torch.Tensor], Dict[str, Any]]:
    graph = getattr(model, "anisotropic_process_graph", None)
    if graph is None:
        return None, {"available": False}
    height = int(model_cfg["H"])
    width = int(model_cfg["W"])
    wind_uv = None
    if hasattr(model, "_extract_context_wind"):
        wind_uv = model._extract_context_wind(batch["x_ctx"], (height, width))
    roughness = None if proxies is None else proxies.get("roughness_proxy")
    adj, diagnostics = graph.build_adjacency(
        static_feat=batch.get("static_raw"),
        spatial_hw=(height, width),
        wind_uv=wind_uv,
        static_cont=batch.get("static_cont"),
        static_cat=batch.get("static_cat"),
        roughness_proxy=roughness,
        diffusion_t=batch.get("t"),
    )
    return adj, to_jsonable(diagnostics)


def graph_delta(left: Optional[torch.Tensor], right: Optional[torch.Tensor]) -> Dict[str, Any]:
    if left is None or right is None:
        return {"available": False, "norm": 0.0, "mean_abs": 0.0, "max_abs": 0.0, "relative_norm": 0.0}
    out = tensor_delta(left, right)
    out["available"] = True
    return out


def run_forward(model: torch.nn.Module, batch: TensorBatch) -> tuple[torch.Tensor, Dict[str, Any]]:
    return model(
        batch["x_t"],
        batch["x_ctx"],
        batch["t"],
        lead_time=batch.get("lead_time"),
        static_raw=batch.get("static_raw"),
        static_cont=batch.get("static_cont"),
        static_cat=batch.get("static_cat"),
        hour_of_day=batch.get("hour_of_day"),
        return_diagnostics=True,
    )


def case_report(
    name: str,
    description: str,
    base_batch: TensorBatch,
    case_batch: TensorBatch,
    base_pred: torch.Tensor,
    case_pred: torch.Tensor,
    model: torch.nn.Module,
    model_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    base_proxies = proxy_fields(model, base_batch)
    case_proxies = proxy_fields(model, case_batch)
    base_adj, base_graph_diag = process_graph_adjacency(model, base_batch, model_cfg, base_proxies)
    case_adj, case_graph_diag = process_graph_adjacency(model, case_batch, model_cfg, case_proxies)
    pred_delta = tensor_delta(base_pred, case_pred)
    proxy_delta = fields_delta(base_proxies, case_proxies)
    proc_graph_delta = graph_delta(base_adj, case_adj)
    return {
        "name": name,
        "description": description,
        "prediction_delta_norm": pred_delta["norm"],
        "proxy_delta_norm": proxy_delta["norm"],
        "process_graph_delta_norm": proc_graph_delta["norm"],
        "prediction_delta": pred_delta,
        "proxy_delta": proxy_delta,
        "process_graph_delta": proc_graph_delta,
        "static_raw_delta": tensor_delta(base_batch["static_raw"], case_batch["static_raw"]),
        "x_ctx_delta": tensor_delta(base_batch["x_ctx"], case_batch["x_ctx"]),
        "base_process_graph_diagnostics": base_graph_diag,
        "case_process_graph_diagnostics": case_graph_diag,
    }


def build_counterfactual_report(
    cfg: Mapping[str, Any],
    *,
    config_path: str | Path,
    batch_size: int = 1,
    case_names: Optional[list[str]] = None,
) -> Dict[str, Any]:
    model_cfg = build_model_cfg(dict(cfg))
    model = build_model_v531_from_config(model_cfg)
    model.eval()

    base_batch = make_dummy_batch(cfg, model_cfg, batch_size=batch_size)
    cases = selected_cases(case_names)
    batches = [clone_batch(base_batch)] + [case.apply(base_batch, model_cfg) for case in cases]
    names = ["original_static"] + [case.name for case in cases]
    descriptions = ["Unmodified static morphology baseline."] + [case.description for case in cases]

    with torch.no_grad():
        combined_pred, combined_diag = run_forward(model, concat_batches(batches))

    chunks = list(combined_pred.chunk(len(batches), dim=0))
    base_pred = chunks[0]
    reports = []
    for index, name in enumerate(names):
        reports.append(
            case_report(
                name,
                descriptions[index],
                batches[0],
                batches[index],
                base_pred,
                chunks[index],
                model,
                model_cfg,
            )
        )

    return {
        "release_name": RELEASE_NAME,
        "version": VERSION,
        "version_name": VERSION_NAME,
        "claim_boundary": CLAIM_BOUNDARY,
        "diagnostic_scope": "counterfactual intervention diagnostic, not causal proof",
        "config_path": str(Path(config_path).expanduser().resolve()),
        "trained_checkpoint": False,
        "case_count": len(reports),
        "intervention_names": names,
        "model_cfg": to_jsonable(filter_model_kwargs(model_cfg)),
        "dropped_model_cfg_keys": sorted(dropped_model_kwargs(model_cfg).keys()),
        "batch_report": {
            "base_batch_size": int(batch_size),
            "combined_batch_size": int(combined_pred.size(0)),
            "x_t_shape": list(base_batch["x_t"].shape),
            "x_ctx_shape": list(base_batch["x_ctx"].shape),
            "target_tensor_present": False,
        },
        "forward_report": {
            "prediction_shape": list(combined_pred.shape),
            "prediction_finite": bool(torch.isfinite(combined_pred).all().cpu().item()),
            "diagnostic_key_count": len(combined_diag),
        },
        "cases": to_jsonable(reports),
        "combined_forward_diagnostics": to_jsonable(combined_diag),
    }


def dry_run_report(case_names: Optional[list[str]]) -> Dict[str, Any]:
    cases = selected_cases(case_names)
    return {
        "release_name": RELEASE_NAME,
        "version": VERSION,
        "diagnostic_scope": "registry only; no model forward executed",
        "intervention_names": ["original_static"] + [case.name for case in cases],
    }


def main(argv: Optional[list[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Run UrbanPiDiT V5.3.1 morphology counterfactual diagnostics.")
    parser.add_argument("--config", type=str, default="configs/urbanpidit_v53_morpho_process.yaml")
    parser.add_argument("--output", type=str, default="outputs/v53_morphology_counterfactual.json")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args(argv)

    if args.dry_run:
        report = dry_run_report(args.cases)
    else:
        cfg = load_config(args.config)
        report = build_counterfactual_report(
            cfg,
            config_path=args.config,
            batch_size=args.batch_size,
            case_names=args.cases,
        )
    write_report(report, args.output)
    print(f"Wrote morphology counterfactual JSON: {Path(args.output).expanduser().resolve()}")
    print(f"Counterfactual cases: {', '.join(report['intervention_names'])}")
    return report


if __name__ == "__main__":
    main()