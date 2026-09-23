r"""
公平对比记录合并与 Markdown 表格输出。

该模块只处理已经评估完成的结果：UrbanPiDiT、fair forecast baselines 与 external
baselines 都会被规整为统一 records，再生成 JSON 与 Markdown 产物。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import yaml

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from .baselines.forecast_runner import run_forecast_baselines
    from .fair_comparison_urban import build_urban_result
except ImportError:
    from baselines.forecast_runner import run_forecast_baselines
    from fair_comparison_urban import build_urban_result


METRICS = ("RMSE", "MAE", "Bias", "CRPS", "ACC")


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


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    r"""
    保存 JSON 文件。

    Parameters
    ----
    data : Dict[str, Any]
        待保存数据。
    path : str or Path
        输出路径。

    Returns
    ----
    None
    """

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_device(name: str) -> torch.device:
    r"""
    解析运行设备。

    Parameters
    ----
    name : str
        ``auto``、``cpu`` 或 ``cuda``。

    Returns
    ----
    torch.device
        运行设备。
    """

    if str(name).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def records_from_urban(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    r"""
    将 UrbanPiDiT 统一结果展开为记录表。

    Parameters
    ----
    result : Dict[str, Any]
        UrbanPiDiT 统一结果。

    Returns
    ----
    List[Dict[str, Any]]
        记录列表。
    """

    return [_record_from_metric(result, tag, metric) for tag, metric in dict(result.get("metrics", {})).items()]


def records_from_forecast_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    r"""
    将 forecast baseline 结果展开为记录表。

    Parameters
    ----
    results : Dict[str, Any]
        ``run_forecast_baselines`` 输出。

    Returns
    ----
    List[Dict[str, Any]]
        记录列表。
    """

    records: List[Dict[str, Any]] = []
    for alias, item in dict(results.get("models", {}) or {}).items():
        base = {
            "alias": alias,
            "model_name": item.get("name"),
            "family": item.get("family"),
            "static_policy": item.get("static_policy"),
        }
        for split in ("val", "test"):
            split_metrics = dict(item.get(split, {}) or {})
            complexity = split_metrics.get("__model_complexity__")
            for tag, metric in split_metrics.items():
                if str(tag).startswith("__"):
                    continue
                records.append(_record_from_metric({**base, "split": split, "complexity": complexity}, tag, metric))
    return records


def records_from_external_results(
    results: Dict[str, Any],
    metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    r"""
    将 external baseline 结果展开为记录表。

    Parameters
    ----
    results : Dict[str, Any]
        ``external_baselines_results.json`` 或 ``baseline_comparison.json``。
    metadata : Dict[str, Dict[str, Any]], optional, default=None
        从 external config 读取的 alias 元数据。

    Returns
    ----
    List[Dict[str, Any]]
        记录列表。
    """

    meta = metadata or {}
    if "records" in results:
        return [_normalise_external_record(record, meta) for record in results.get("records", [])]
    records: List[Dict[str, Any]] = []
    for alias, item in dict(results.get("models", {}) or {}).items():
        records.extend(_records_from_external_model(str(alias), item, meta))
    return records


def _records_from_external_model(
    alias: str,
    item: Dict[str, Any],
    metadata: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    item_meta = dict(metadata.get(alias, {}) or {})
    base = {
        "alias": alias,
        "model_name": item.get("name", item_meta.get("model_name")),
        "family": item_meta.get("family", "external_weather"),
        "static_policy": item.get("static_policy", item_meta.get("static_policy")),
    }
    records: List[Dict[str, Any]] = []
    for split in ("val", "test"):
        for tag, metric in dict(item.get("metrics", {}).get(split, {}) or {}).items():
            if str(tag).startswith("__"):
                continue
            payload = {**base, "split": split, "complexity": item.get("model_complexity")}
            records.append(_record_from_metric(payload, tag, metric))
    return records


def _normalise_external_record(record: Dict[str, Any], metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    alias = str(record.get("alias", record.get("model_name", record.get("name", "external_baseline"))))
    meta = dict(metadata.get(alias, {}) or {})
    out = dict(record)
    out["alias"] = alias
    out["model_name"] = out.get("model_name", out.get("name", meta.get("model_name", alias)))
    out["family"] = out.get("family", meta.get("family", "external_weather"))
    out["static_policy"] = out.get("static_policy", meta.get("static_policy"))
    return out


def _record_from_metric(base: Dict[str, Any], tag: str, metric: Dict[str, Any]) -> Dict[str, Any]:
    complexity = dict(base.get("complexity", {}) or {})
    record = {
        "alias": base.get("alias", base.get("model_name")),
        "model_name": base.get("model_name", base.get("alias")),
        "family": base.get("family"),
        "static_policy": base.get("static_policy"),
        "split": base.get("split", "test"),
        "lead_tag": tag,
        "lead_steps": metric.get("lead_steps"),
        "num_parameters": complexity.get("num_parameters"),
        "num_trainable": complexity.get("num_trainable"),
        "estimated_gflops": complexity.get("estimated_gflops"),
        "estimated_gflops_per_sample": complexity.get("estimated_gflops_per_sample"),
        "one_step_gflops": complexity.get("one_step_gflops"),
        "one_step_gflops_per_sample": complexity.get("one_step_gflops_per_sample"),
        "rollout_gflops": complexity.get("rollout_gflops"),
        "rollout_gflops_per_sample": complexity.get("rollout_gflops_per_sample"),
        "rollout_steps": complexity.get("rollout_steps"),
        "num_diffusion_steps": complexity.get("num_diffusion_steps"),
    }
    for key in METRICS:
        record[key] = metric.get(key)
    return record


def make_markdown_tables(records: Sequence[Dict[str, Any]]) -> str:
    r"""
    生成公平对比 Markdown 表。

    Parameters
    ----
    records : Sequence[Dict[str, Any]]
        展平后的记录表。

    Returns
    ----
    str
        Markdown 内容。
    """

    lines = ["# Fair comparison tables\n"]
    test_records = [r for r in records if str(r.get("split", "test")) == "test"]
    for lead in sorted({str(r.get("lead_tag")) for r in test_records}, key=_lead_sort_key):
        rows = [r for r in test_records if str(r.get("lead_tag")) == lead]
        lines.append(f"\n## Test metrics @{lead}\n")
        lines.append(_metrics_table(rows))
    lines.append("\n## Static information and capacity overview\n")
    lines.append(_capacity_table(test_records))
    return "".join(lines)


def _metrics_table(rows: Sequence[Dict[str, Any]]) -> str:
    header = (
        "| model | family | static | params(M) | GFLOPs/sample | RMSE | MAE | "
        "Bias | CRPS | ACC | RMSE/MParam | RMSE/GFLOP |\n"
    )
    sep = "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    body = [_metrics_row(row) for row in sorted(rows, key=lambda r: str(r.get("alias")))]
    return header + sep + "".join(body)


def _capacity_table(rows: Sequence[Dict[str, Any]]) -> str:
    latest: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        latest[str(row.get("alias"))] = row
    header = "| model | family | static_policy | params | GFLOPs/sample | rollout_steps | diffusion_steps |\n"
    sep = "|---|---|---|---:|---:|---:|---:|\n"
    return header + sep + "".join(_capacity_row(alias, row) for alias, row in sorted(latest.items()))


def _capacity_row(alias: str, row: Dict[str, Any]) -> str:
    return (
        f"| {alias} | {row.get('family', '')} | {row.get('static_policy', '')} | "
        f"{_fmt(row.get('num_parameters'), 0)} | {_fmt(row.get('estimated_gflops_per_sample'), 4)} | "
        f"{_fmt(row.get('rollout_steps'), 0)} | {_fmt(row.get('num_diffusion_steps'), 0)} |\n"
    )


def _metrics_row(row: Dict[str, Any]) -> str:
    params_m = _ratio(row.get("num_parameters"), 1_000_000.0)
    gflops = _to_float(row.get("estimated_gflops_per_sample"))
    rmse = _to_float(row.get("RMSE"))
    return (
        f"| {row.get('alias', '')} | {row.get('family', '')} | {row.get('static_policy', '')} | "
        f"{_fmt(params_m, 3)} | {_fmt(gflops, 4)} | {_fmt(row.get('RMSE'), 6)} | "
        f"{_fmt(row.get('MAE'), 6)} | {_fmt(row.get('Bias'), 6)} | {_fmt(row.get('CRPS'), 6)} | "
        f"{_fmt(row.get('ACC'), 6)} | {_fmt(_safe_div(rmse, params_m), 6)} | "
        f"{_fmt(_safe_div(rmse, gflops), 6)} |\n"
    )


def _lead_sort_key(lead: str) -> tuple[int, str]:
    digits = "".join(ch for ch in lead if ch.isdigit())
    return (int(digits) if digits else 10**9, lead)


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _ratio(value: Any, denom: float) -> Optional[float]:
    value_f = _to_float(value)
    if value_f is None:
        return None
    return value_f / float(denom)


def _safe_div(num: Optional[float], denom: Optional[float]) -> Optional[float]:
    if num is None or denom is None or denom == 0.0:
        return None
    return num / denom


def _fmt(value: Any, digits: int) -> str:
    value_f = _to_float(value)
    if value_f is None:
        return "NA"
    return f"{value_f:.{int(digits)}f}"


def load_external_results(path: Optional[str]) -> Optional[Dict[str, Any]]:
    r"""
    读取已存在的 external baseline 结果。

    Parameters
    ----
    path : str, optional
        JSON 文件路径。

    Returns
    ----
    Dict[str, Any], optional
        结果字典。路径为空时返回 None。
    """

    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"External results not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def external_metadata_by_alias(config_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    r"""
    从 external baseline 配置中提取 alias 元数据。

    Parameters
    ----
    config_path : str, optional
        external baseline 配置路径。

    Returns
    ----
    Dict[str, Dict[str, Any]]
        alias 到元数据的映射。
    """

    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    cfg = load_yaml(path)
    items = dict(cfg.get("baselines", {}) or {}).get("models", []) or []
    metadata: Dict[str, Dict[str, Any]] = {}
    for item in items:
        if isinstance(item, str):
            item = {"name": item, "alias": item}
        alias = str(item.get("alias", item.get("name", "")))
        if not alias:
            continue
        metadata[alias] = {
            "model_name": item.get("name", alias),
            "family": item.get("family", "external_weather"),
            "static_policy": item.get("static_policy"),
        }
    return metadata


def run_fair_comparison(args: Any) -> Dict[str, Any]:
    r"""
    执行公平对比流程。

    Parameters
    ----
    args : Any
        CLI 参数对象。

    Returns
    ----
    Dict[str, Any]
        完整结果。
    """

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    urban = build_urban_result(args.urbanpidit_config, args.urbanpidit_ckpt, resolve_device(args.device))
    records = records_from_urban(urban)
    forecast = _maybe_run_forecast_baselines(args, out_dir)
    if forecast is not None:
        records.extend(records_from_forecast_results(forecast))
    external = load_external_results(args.external_results)
    if external is not None:
        records.extend(records_from_external_results(external, external_metadata_by_alias(args.external_config)))
    result = {"urbanpidit": urban, "forecast_baselines": forecast, "external_baselines": external, "records": records}
    save_json(result, out_dir / "fair_comparison.json")
    save_json({"records": records}, out_dir / "combined_records.json")
    (out_dir / "fair_comparison.md").write_text(make_markdown_tables(records), encoding="utf-8")
    return result


def _maybe_run_forecast_baselines(args: Any, out_dir: Path) -> Optional[Dict[str, Any]]:
    if bool(getattr(args, "skip_forecast_baselines", False)):
        return None
    return run_forecast_baselines(config_path=args.baseline_config, out_dir=out_dir / "forecast_baselines")