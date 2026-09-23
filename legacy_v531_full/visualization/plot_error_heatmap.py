## 误差热力图


import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_json(fp: str) -> Dict:
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_csv_str(s: str) -> List[str]:
    s = str(s).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_csv_ints(s: str) -> List[int]:
    return [int(x) for x in _parse_csv_str(s)]


def _parse_hours_tag(tag: str) -> Optional[float]:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)h", str(tag).strip())
    if not m:
        return None
    return float(m.group(1))


def _infer_lead_tags(metrics: Dict, metric: str) -> List[str]:
    pat = re.compile(rf"^test/{re.escape(metric)}_[^@]+@(.+)$")
    tags = []
    for k in metrics.keys():
        m = pat.match(k)
        if m:
            tags.append(str(m.group(1)))
    tags = sorted(list(dict.fromkeys(tags)))
    tags_sorted = []
    parsed = []
    for t in tags:
        h = _parse_hours_tag(t)
        if h is None:
            tags_sorted.append(t)
        else:
            parsed.append((h, t))
    parsed.sort(key=lambda x: x[0])
    tags_sorted = [t for _, t in parsed] + [t for t in tags_sorted if _parse_hours_tag(t) is None]
    return tags_sorted


def _infer_vars(metrics: Dict, metric: str, lead_tag: str) -> List[str]:
    pat = re.compile(rf"^test/{re.escape(metric)}_(.+)@{re.escape(lead_tag)}$")
    vars_ = []
    for k in metrics.keys():
        m = pat.match(k)
        if m:
            vars_.append(str(m.group(1)))
    return sorted(list(dict.fromkeys(vars_)))


def _metric_key(metric: str, var: str, lead_tag: str) -> str:
    return f"test/{metric}_{var}@{lead_tag}"


def _build_matrix(
    metrics: Dict,
    *,
    metric: str,
    vars_: Sequence[str],
    lead_tags: Sequence[str],
) -> np.ndarray:
    mat = np.full((len(vars_), len(lead_tags)), np.nan, dtype=np.float64)
    for i, v in enumerate(vars_):
        for j, lt in enumerate(lead_tags):
            val = metrics.get(_metric_key(metric, v, lt), None)
            if val is None:
                continue
            try:
                mat[i, j] = float(val)
            except Exception:
                mat[i, j] = np.nan
    return mat


def _plot_heatmap(
    mat: np.ndarray,
    *,
    vars_: Sequence[str],
    lead_tags: Sequence[str],
    title: str,
    cmap: str,
    annot: bool,
    fmt: str,
    cbar_label: str,
    scale: str,
    vmax_percentile: float,
    save_path: str,
):
    mask = ~np.isfinite(mat)

    scale = str(scale).lower()
    if scale not in ("linear", "log10"):
        raise ValueError("scale 仅支持 linear/log10")

    mat_plot = np.array(mat, dtype=np.float64, copy=True)
    if scale == "log10":
        with np.errstate(invalid="ignore"):
            min_pos = np.nanmin(mat_plot[(~mask) & (mat_plot > 0)]) if np.any((~mask) & (mat_plot > 0)) else np.nan
        if not np.isfinite(min_pos):
            raise ValueError("log10 缩放要求矩阵存在正数值")
        with np.errstate(divide="ignore", invalid="ignore"):
            mat_plot = np.log10(np.maximum(mat_plot, min_pos))

    m = np.ma.array(mat_plot, mask=mask)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#e6e6e6")

    fig, ax = plt.subplots(figsize=(max(6.0, 0.8 * len(lead_tags)), max(3.0, 0.4 * len(vars_))))
    vmin = None
    vmax = None
    if float(vmax_percentile) < 100.0:
        with np.errstate(invalid="ignore"):
            vmax = float(np.nanpercentile(mat_plot[~mask], float(vmax_percentile)))
            if np.isfinite(vmax):
                vmin = float(np.nanmin(mat_plot[~mask]))
    im = ax.imshow(m, aspect="auto", cmap=cmap_obj, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(lead_tags)))
    ax.set_xticklabels(list(lead_tags))
    ax.set_yticks(np.arange(len(vars_)))
    ax.set_yticklabels(list(vars_))
    ax.set_xlabel("Lead Time")
    ax.set_ylabel("Variable")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    if annot:
        norm = im.norm
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isfinite(mat[i, j]):
                    continue
                v = mat_plot[i, j]
                rgba = cmap_obj(norm(v))
                lum = 0.299 * float(rgba[0]) + 0.587 * float(rgba[1]) + 0.114 * float(rgba[2])
                txt_color = "white" if lum < 0.5 else "black"
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", fontsize=8, color=txt_color)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_json", type=str, required=True)
    parser.add_argument("--experiment", type=str, default="baseline")
    parser.add_argument("--metric", type=str, default="RMSE")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--leads", type=str, default="")
    parser.add_argument("--vars", type=str, default="")
    parser.add_argument("--relative_to", type=str, default="none")
    parser.add_argument("--annot", action="store_true")
    parser.add_argument("--fmt", type=str, default=".2f")
    parser.add_argument("--cmap", type=str, default="magma")
    parser.add_argument("--scale", type=str, default="linear")
    parser.add_argument("--vmax_percentile", type=float, default=100.0)
    parser.add_argument("--out_path", type=str, default="")
    args = parser.parse_args()

    results = _load_json(args.results_json)
    exp = str(args.experiment)
    if exp not in results:
        raise KeyError(f"experiment 不在 results_json: {exp}, available={list(results.keys())[:20]}")
    metrics = dict(results.get(exp, {}) or {})

    metric = str(args.metric).upper()
    if metric not in ("RMSE", "MAE"):
        raise ValueError("metric 仅支持 RMSE/MAE")

    dyn_vars = None
    time_step_hours = None
    if str(args.config).strip():
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        dyn_vars = list(cfg.get("dynamic_vars", []))
        forecast_cfg = dict(cfg.get("forecast", {}) or {})
        time_step_hours = float(forecast_cfg.get("time_step_hours", 6.0))

    if str(args.leads).strip():
        leads_hours = _parse_csv_ints(args.leads)
        lead_tags = [f"{int(h)}h" if float(h).is_integer() else f"{float(h):.1f}h" for h in leads_hours]
    else:
        lead_tags = _infer_lead_tags(metrics, metric)
        if not lead_tags and dyn_vars and time_step_hours is not None:
            lead_steps = list((cfg.get("forecast", {}) or {}).get("eval_lead_times", (cfg.get("forecast", {}) or {}).get("lead_times", [cfg.get("delta_t", 1)])))
            lead_tags = []
            for s in lead_steps:
                h = float(time_step_hours) * float(int(s))
                lead_tags.append(f"{int(h)}h" if abs(h - round(h)) < 1e-6 else f"{h:.1f}h")

    if not lead_tags:
        raise ValueError("无法从 results_json 推断 lead_tags，请用 --leads 显式指定，例如 6,12,18,24")

    if str(args.vars).strip():
        vars_ = _parse_csv_str(args.vars)
    elif dyn_vars:
        vars_ = dyn_vars
    else:
        vars_ = _infer_vars(metrics, metric, lead_tags[0])

    if not vars_:
        raise ValueError("无法确定变量列表，请用 --vars 显式指定")

    mat = _build_matrix(metrics, metric=metric, vars_=vars_, lead_tags=lead_tags)

    rel = str(args.relative_to).lower()
    title_suffix = exp
    cbar_label = metric
    if rel != "none":
        if rel != "baseline":
            raise ValueError("relative_to 仅支持 none/baseline")
        base = dict(results.get("baseline", {}) or {})
        mat_base = _build_matrix(base, metric=metric, vars_=vars_, lead_tags=lead_tags)
        with np.errstate(divide="ignore", invalid="ignore"):
            mat = (mat - mat_base) / mat_base * 100.0
        title_suffix = f"{exp} vs baseline"
        cbar_label = f"{metric} Relative Change (%)"

    title = f"{metric} Heatmap ({title_suffix})"
    if not args.out_path:
        out_path = str(Path(args.results_json).with_suffix("").name)
        out_file = f"heatmap_{metric.lower()}_{exp}"
        if rel != "none":
            out_file += "_vs_baseline"
        args.out_path = str(Path("visualization") / "out" / f"{out_file}.png")

    _plot_heatmap(
        mat,
        vars_=vars_,
        lead_tags=lead_tags,
        title=title,
        cmap=str(args.cmap),
        annot=bool(args.annot),
        fmt=str(args.fmt),
        cbar_label=cbar_label,
        scale=str(args.scale),
        vmax_percentile=float(args.vmax_percentile),
        save_path=str(args.out_path),
    )
    print(f"Saved heatmap to: {args.out_path}")


if __name__ == "__main__":
    main()
