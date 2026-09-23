from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


METRICS = ["RMSE", "MAE", "Bias", "CRPS", "ACC"]
DEFAULT_LEADS = ["6h", "12h", "18h", "24h"]
DEFAULT_INPUT = "outputs/fair_comparison/combined_records.json"
DEFAULT_OUTPUT = "all_tables.md"


def load_json(path: str | Path) -> Dict[str, Any]:
    r"""
    读取 JSON 文件。

    Parameters
    ----
    path : str or Path
        输入 JSON 路径。

    Returns
    ----
    Dict[str, Any]
        JSON 内容。
    """

    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_records(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    r"""
    提取统一记录表，兼容新旧结果结构。

    Parameters
    ----
    data : Dict[str, Any]
        结果 JSON。

    Returns
    ----
    List[Dict[str, Any]]
        统一记录列表。
    """

    if "records" in data:
        return [dict(record) for record in data.get("records", [])]
    return legacy_static_records(data)


def legacy_static_records(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    r"""
    将旧版静态消融 flat logger JSON 转成记录表。

    Parameters
    ----
    data : Dict[str, Any]
        旧版结果 JSON。

    Returns
    ----
    List[Dict[str, Any]]
        记录列表。
    """

    rows: List[Dict[str, Any]] = []
    for alias, values in data.items():
        for lead in DEFAULT_LEADS:
            row = {"alias": alias, "model_name": alias, "split": "test", "lead_tag": lead}
            for metric in METRICS:
                row[metric] = values.get(f"test/{metric}@{lead}")
            rows.append(row)
    return rows


def make_markdown_tables(records: Sequence[Dict[str, Any]]) -> str:
    r"""
    生成多模型族公平对比表。

    Parameters
    ----
    records : Sequence[Dict[str, Any]]
        统一记录列表。

    Returns
    ----
    str
        Markdown 表格内容。
    """

    test_records = [row for row in records if str(row.get("split", "test")) == "test"]
    leads = sorted({str(row.get("lead_tag")) for row in test_records}, key=lead_sort_key)
    parts = ["# Fair comparison tables\n"]
    for lead in leads:
        rows = [row for row in test_records if str(row.get("lead_tag")) == lead]
        parts.append(f"\n## Test metrics @{lead}\n\n")
        parts.append(metric_table(rows))
    parts.append("\n## Capacity-normalized overview\n\n")
    parts.append(capacity_table(test_records))
    parts.append("\n## Static information access\n\n")
    parts.append(static_policy_table(test_records))
    return "".join(parts)


def metric_table(rows: Sequence[Dict[str, Any]]) -> str:
    r"""
    生成单 lead 指标表。

    Parameters
    ----
    rows : Sequence[Dict[str, Any]]
        同一提前期的记录。

    Returns
    ----
    str
        Markdown 表格。
    """

    header = (
        "| model | family | static | params(M) | GFLOPs/sample | RMSE | MAE | "
        "Bias | CRPS | ACC | RMSE/MParam | RMSE/GFLOP |\n"
    )
    sep = "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    body = "".join(metric_row(row) for row in sorted(rows, key=model_sort_key))
    return header + sep + body


def metric_row(row: Dict[str, Any]) -> str:
    r"""
    生成单模型指标行。

    Parameters
    ----
    row : Dict[str, Any]
        模型记录。

    Returns
    ----
    str
        Markdown 行。
    """

    params_m = safe_ratio(row.get("num_parameters"), 1_000_000.0)
    gflops = to_float(row.get("estimated_gflops_per_sample"))
    rmse = to_float(row.get("RMSE"))
    return (
        f"| {cell(row.get('alias'))} | {cell(row.get('family'))} | "
        f"{cell(row.get('static_policy'))} | {fmt(params_m, 3)} | {fmt(gflops, 4)} | "
        f"{fmt(row.get('RMSE'), 6)} | {fmt(row.get('MAE'), 6)} | "
        f"{fmt(row.get('Bias'), 6)} | {fmt(row.get('CRPS'), 6)} | "
        f"{fmt(row.get('ACC'), 6)} | {fmt(safe_div(rmse, params_m), 6)} | "
        f"{fmt(safe_div(rmse, gflops), 6)} |\n"
    )


def capacity_table(records: Sequence[Dict[str, Any]]) -> str:
    r"""
    生成容量归一化总览表。

    Parameters
    ----
    records : Sequence[Dict[str, Any]]
        test split 记录。

    Returns
    ----
    str
        Markdown 表格。
    """

    rows = latest_by_model(records)
    header = "| model | family | params | GFLOPs/sample | rollout_steps | diffusion_steps |\n"
    sep = "|---|---|---:|---:|---:|---:|\n"
    body = "".join(capacity_row(row) for row in sorted(rows, key=model_sort_key))
    return header + sep + body


def capacity_row(row: Dict[str, Any]) -> str:
    r"""
    生成容量表单行。

    Parameters
    ----
    row : Dict[str, Any]
        模型记录。

    Returns
    ----
    str
        Markdown 行。
    """

    return (
        f"| {cell(row.get('alias'))} | {cell(row.get('family'))} | "
        f"{fmt(row.get('num_parameters'), 0)} | {fmt(row.get('estimated_gflops_per_sample'), 4)} | "
        f"{fmt(row.get('rollout_steps'), 0)} | {fmt(row.get('num_diffusion_steps'), 0)} |\n"
    )


def static_policy_table(records: Sequence[Dict[str, Any]]) -> str:
    r"""
    生成静态信息访问策略表。

    Parameters
    ----
    records : Sequence[Dict[str, Any]]
        test split 记录。

    Returns
    ----
    str
        Markdown 表格。
    """

    rows = latest_by_model(records)
    header = "| model | family | static_policy |\n"
    sep = "|---|---|---|\n"
    body = "".join(
        f"| {cell(row.get('alias'))} | {cell(row.get('family'))} | "
        f"{cell(row.get('static_policy'))} |\n"
        for row in sorted(rows, key=model_sort_key)
    )
    return header + sep + body


def latest_by_model(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    r"""
    每个模型保留一条代表性记录。

    Parameters
    ----
    records : Sequence[Dict[str, Any]]
        记录列表。

    Returns
    ----
    List[Dict[str, Any]]
        去重后的记录列表。
    """

    latest: Dict[str, Dict[str, Any]] = {}
    for row in records:
        latest[str(row.get("alias", row.get("model_name", "")))] = dict(row)
    return list(latest.values())


def lead_sort_key(lead: str) -> tuple[int, str]:
    r"""
    生成 lead 排序键。

    Parameters
    ----
    lead : str
        lead 标签。

    Returns
    ----
    tuple[int, str]
        排序键。
    """

    digits = "".join(ch for ch in lead if ch.isdigit())
    return (int(digits) if digits else 10**9, lead)


def model_sort_key(row: Dict[str, Any]) -> tuple[str, str]:
    r"""
    生成模型排序键。

    Parameters
    ----
    row : Dict[str, Any]
        模型记录。

    Returns
    ----
    tuple[str, str]
        排序键。
    """

    return (str(row.get("family", "")), str(row.get("alias", row.get("model_name", ""))))


def to_float(value: Any) -> Optional[float]:
    r"""
    转换为有限浮点数。

    Parameters
    ----
    value : Any
        输入值。

    Returns
    ----
    float, optional
        有限浮点数。
    """

    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def safe_ratio(value: Any, denom: float) -> Optional[float]:
    r"""
    安全比例计算。

    Parameters
    ----
    value : Any
        分子。
    denom : float
        分母。

    Returns
    ----
    float, optional
        比值。
    """

    value_f = to_float(value)
    if value_f is None or denom == 0.0:
        return None
    return value_f / float(denom)


def safe_div(num: Optional[float], denom: Optional[float]) -> Optional[float]:
    r"""
    安全除法。

    Parameters
    ----
    num : float, optional
        分子。
    denom : float, optional
        分母。

    Returns
    ----
    float, optional
        商。
    """

    if num is None or denom is None or denom == 0.0:
        return None
    return num / denom


def fmt(value: Any, digits: int) -> str:
    r"""
    格式化浮点数。

    Parameters
    ----
    value : Any
        输入值。
    digits : int
        小数位。

    Returns
    ----
    str
        表格文本。
    """

    value_f = to_float(value)
    if value_f is None:
        return "NA"
    return f"{value_f:.{int(digits)}f}"


def cell(value: Any) -> str:
    r"""
    格式化表格单元格。

    Parameters
    ----
    value : Any
        输入值。

    Returns
    ----
    str
        单元格文本。
    """

    if value is None:
        return "NA"
    return str(value).replace("|", "\\|")


def build_parser() -> argparse.ArgumentParser:
    r"""
    构建命令行参数。

    Returns
    ----
    argparse.ArgumentParser
        参数解析器。
    """

    parser = argparse.ArgumentParser(description="Generate markdown tables from fair comparison records")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = extract_records(load_json(args.input))
    Path(args.output).write_text(make_markdown_tables(records), encoding="utf-8")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
