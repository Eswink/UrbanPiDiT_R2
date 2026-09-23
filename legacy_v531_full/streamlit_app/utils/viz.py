"""Streamlit 图表辅助函数。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def pivot_metric_matrix(
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
    split: str = "test",
    model_name: str | None = None,
) -> tuple[list[str], list[str], np.ndarray]:
    """构建变量 × lead_time 指标矩阵。"""

    filtered = [
        row
        for row in rows
        if str(row.get("metric_name")) == metric_name
        and str(row.get("split")) == split
        and (model_name is None or str(row.get("model_name", row.get("model", ""))) == model_name)
    ]
    variables = sorted({str(row.get("variable", "")) for row in filtered if row.get("variable")})
    leads = sorted({str(row.get("lead_time", "")) for row in filtered if row.get("lead_time")}, key=_lead_key)
    mat = np.full((len(variables), len(leads)), np.nan, dtype=float)
    var_idx = {name: i for i, name in enumerate(variables)}
    lead_idx = {name: i for i, name in enumerate(leads)}
    for row in filtered:
        variable = str(row.get("variable", ""))
        lead = str(row.get("lead_time", ""))
        if variable not in var_idx or lead not in lead_idx:
            continue
        mat[var_idx[variable], lead_idx[lead]] = float(row.get("metric_value"))
    return variables, leads, mat


def plot_metric_heatmap(
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
    split: str = "test",
    model_name: str | None = None,
    title: str | None = None,
):
    """绘制指标热力图。"""

    variables, leads, mat = pivot_metric_matrix(rows, metric_name=metric_name, split=split, model_name=model_name)
    fig, ax = plt.subplots(figsize=(max(6.0, len(leads) * 1.0), max(3.5, len(variables) * 0.4)))
    if len(variables) == 0 or len(leads) == 0:
        ax.text(0.5, 0.5, "没有可用指标", ha="center", va="center")
        ax.axis("off")
        return fig

    masked = np.ma.array(mat, mask=~np.isfinite(mat))
    im = ax.imshow(masked, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(leads)))
    ax.set_xticklabels(leads)
    ax.set_yticks(np.arange(len(variables)))
    ax.set_yticklabels(variables)
    ax.set_xlabel("Lead time")
    ax.set_ylabel("Variable")
    ax.set_title(title or f"{split} {metric_name}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.3g}", ha="center", va="center", color="white", fontsize=8)
    fig.tight_layout()
    return fig


def plot_metric_curve(
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
    split: str = "test",
    variable: str = "mean",
    model_name: str | None = None,
):
    """绘制指标随 lead_time 变化曲线。"""

    filtered = [
        row
        for row in rows
        if str(row.get("metric_name")) == metric_name
        and str(row.get("split")) == split
        and str(row.get("variable")) == variable
        and (model_name is None or str(row.get("model_name", row.get("model", ""))) == model_name)
    ]
    filtered.sort(key=lambda row: _lead_key(str(row.get("lead_time", ""))))
    fig, ax = plt.subplots(figsize=(7, 4))
    if not filtered:
        ax.text(0.5, 0.5, "没有可用指标", ha="center", va="center")
        ax.axis("off")
        return fig
    x = [str(row.get("lead_time", "")) for row in filtered]
    y = [float(row.get("metric_value")) for row in filtered]
    ax.plot(x, y, marker="o", linewidth=2)
    ax.set_xlabel("Lead time")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{split} {metric_name} / {variable}")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def save_figure(fig, path: str | Path) -> None:
    """保存图表。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=200, bbox_inches="tight")


def _lead_key(value: str) -> float:
    text = str(value).replace("h", "")
    try:
        return float(text)
    except ValueError:
        return float("inf")