"""实验指标产物解析。"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

LEAD_RE = re.compile(r"@(\d+(?:\.\d+)?h)$")
METRIC_VAR_RE = re.compile(r"^(?:test|val)/([^_/]+)_(.+)@(\d+(?:\.\d+)?h)$")


def load_json(path: str | Path) -> dict[str, Any]:
    """读取 JSON。"""

    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return data


def find_result_files(root: str | Path) -> list[Path]:
    """扫描常见指标 JSON 文件。"""

    root_path = Path(root)
    if not root_path.exists():
        return []
    names = {"results.json", "summary.json", "metrics.json"}
    return sorted(path for path in root_path.rglob("*.json") if path.name in names)


def flatten_baseline_results(data: dict[str, Any]) -> list[dict[str, Any]]:
    """把基线结果展开成表格记录。"""

    rows: list[dict[str, Any]] = []
    if "models" in data and isinstance(data["models"], dict):
        for model_name, model_data in data["models"].items():
            rows.extend(_flatten_split_tree(model_data, model_name=str(model_name)))
        return rows

    for maybe_model, model_data in data.items():
        if isinstance(model_data, dict) and any(split in model_data for split in ("val", "test")):
            rows.extend(_flatten_split_tree(model_data, model_name=str(maybe_model)))
    if rows:
        return rows

    rows.extend(_flatten_split_tree(data, model_name=str(data.get("name", "experiment"))))
    return rows


def _flatten_split_tree(data: dict[str, Any], *, model_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("val", "test"):
        split_data = data.get(split)
        if not isinstance(split_data, dict):
            continue
        for lead_time, payload in split_data.items():
            if isinstance(payload, dict):
                rows.extend(_rows_from_metric_payload(model_name, split, str(lead_time), payload))
    return rows


def _rows_from_metric_payload(
    model_name: str,
    split: str,
    lead_time: str,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_name, value in payload.items():
        if isinstance(value, dict) and metric_name.endswith("_per_var"):
            base_metric = metric_name.replace("_per_var", "")
            for variable, metric_value in value.items():
                if _is_number(metric_value):
                    rows.append(
                        {
                            "model": model_name,
                            "split": split,
                            "lead_time": lead_time,
                            "variable": str(variable),
                            "metric_name": base_metric,
                            "metric_value": float(metric_value),
                        }
                    )
        elif _is_number(value):
            rows.append(
                {
                    "model": model_name,
                    "split": split,
                    "lead_time": lead_time,
                    "variable": "mean",
                    "metric_name": str(metric_name),
                    "metric_value": float(value),
                }
            )
    return rows


def flatten_lightning_metrics(data: dict[str, Any]) -> list[dict[str, Any]]:
    """展开 Lightning 风格 test/RMSE_var@6h 指标。"""

    rows: list[dict[str, Any]] = []
    for key, value in data.items():
        if not _is_number(value):
            continue
        split = "test" if str(key).startswith("test/") else "val" if str(key).startswith("val/") else None
        if split is None:
            continue
        metric_key = str(key).split("/", 1)[1]
        match = METRIC_VAR_RE.match(str(key))
        if match:
            metric_name, variable, lead_time = match.groups()
            rows.append(
                {
                    "model": "UrbanPiDiT",
                    "split": split,
                    "lead_time": lead_time,
                    "variable": variable,
                    "metric_name": metric_name,
                    "metric_value": float(value),
                }
            )
            continue
        lead_match = LEAD_RE.search(metric_key)
        lead_time = lead_match.group(1) if lead_match else ""
        metric_name = metric_key.replace(f"@{lead_time}", "") if lead_time else metric_key
        rows.append(
            {
                "model": "UrbanPiDiT",
                "split": split,
                "lead_time": lead_time,
                "variable": "mean",
                "metric_name": metric_name,
                "metric_value": float(value),
            }
        )
    return rows


def rows_for_db(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """转换为数据库指标记录。"""

    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "model_name": row.get("model", row.get("model_name", "experiment")),
                "metric_name": row.get("metric_name"),
                "metric_value": row.get("metric_value"),
                "lead_time": row.get("lead_time"),
                "variable": row.get("variable"),
                "split": row.get("split"),
            }
        )
    return out


def parse_result_json(path: str | Path) -> list[dict[str, Any]]:
    """根据 JSON 结构自动解析指标。"""

    data = load_json(path)
    rows = flatten_lightning_metrics(data)
    if rows:
        return rows
    return flatten_baseline_results(data)


def _is_number(value: Any) -> bool:
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))