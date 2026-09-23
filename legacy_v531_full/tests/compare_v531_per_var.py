#!/usr/bin/env python3
"""V5.3.1 各变量（d2m, sp, t2m, tcc, tp, u10, v10）详细对比。"""

import json
from pathlib import Path

BASE = Path(
    "/3240608030/weather-q/UrbanPiDiT_V4_with_baselines/outputs/experiments/v531"
)

COMPLETED = [
    "cumulative_v4_base",
    "cumulative_v5_static_enc",
    "cumulative_v5_morph_graph",
    "cumulative_v5_wind_graph",
    "cumulative_v51_micromet",
]
VARS = ["d2m", "t2m", "sp", "tcc", "tp", "u10", "v10"]
VAR_CN = {
    "d2m": "露点温度", "t2m": "2m温度", "sp": "地表气压",
    "tcc": "总云量", "tp": "降水量", "u10": "10m纬向风", "v10": "10m经向风",
}
METRICS = ["RMSE", "MAE", "ACC", "CRPS", "Bias"]

all_data: dict[str, dict] = {}
for exp in COMPLETED:
    with open(BASE / exp / "summary.json") as f:
        s = json.load(f)
    m = s["metrics"]
    exp_data: dict[str, dict[str, float]] = {}
    for v in VARS:
        exp_data[v] = {}
        for metric in METRICS:
            key = f"test/{metric}_{v}@6h"
            if key not in m:
                key = f"test/{metric}_{v}"
            if key in m:
                exp_data[v][metric] = m[key]
    exp_data["_meta"] = {
        "desc": s["desc"],
        "elapsed_h": round(s["elapsed_sec"] / 3600, 1),
    }
    all_data[exp] = exp_data

# Per-variable: RMSE@6h, 12h, 18h, 24h
LEAD_LABELS = {"6h": "6h", "12h": "12h", "18h": "18h", "24h": "24h"}

for metric in ["RMSE", "MAE", "ACC", "CRPS"]:
    print(f"\n{'='*90}")
    print(f"  test/{metric} per variable per lead time")
    print(f"{'='*90}")

    for v in VARS:
        print(f"\n  --- {v} ({VAR_CN.get(v, v)}) ---")
        header = f"{'实验':<28s}"
        for lt in ["6h", "12h", "18h", "24h"]:
            header += f" {lt:>9s}"
        print(f"  {header}")
        print(f"  {'-'*65}")

        # Collect & sort by mean across leads
        exp_scores = []
        for exp in COMPLETED:
            with open(BASE / exp / "summary.json") as f:
                s = json.load(f)
            m = s["metrics"]
            scores = {}
            for lt in ["6h", "12h", "18h", "24h"]:
                key = f"test/{metric}_{v}@{lt}"
                if key not in m:
                    key = f"test/{metric}_{v}"
                if key in m:
                    scores[lt] = m[key]
            if scores:
                avg = sum(scores.values()) / len(scores)
                exp_scores.append((exp, scores, avg))

        # Sort: RMSE/MAE/CRPS lower=better, ACC higher=better
        reverse = (metric == "ACC")
        exp_scores.sort(key=lambda x: x[2], reverse=reverse)
        best_key = "MIN" if not reverse else "MAX"
        best_exp = exp_scores[0][1]

        for exp, scores, avg in exp_scores:
            short = exp.replace("cumulative_", "")
            parts = [f"{short:<28s}"]
            for lt in ["6h", "12h", "18h", "24h"]:
                val = scores.get(lt)
                if val is not None:
                    # Highlight best in each column
                    best_val = best_exp.get(lt)
                    if best_val is not None and val == best_val:
                        parts.append(f"*{val:>8.4f}*")
                    else:
                        parts.append(f" {val:>9.4f}")
                else:
                    parts.append(f" {'N/A':>9s}")
            print(f"  {' '.join(parts)}")
        print(f"  (* = 该列最优 {best_key})")

# Bias comparison at 6h
print(f"\n{'='*90}")
print(f"  test/Bias @6h — 系统性偏差诊断")
print(f"{'='*90}")
header = f"{'实验':<28s} " + " ".join(f"{v:>8s}" for v in VARS)
print(f"  {header}")
print(f"  {'-'*70}")
for exp in COMPLETED:
    with open(BASE / exp / "summary.json") as f:
        s = json.load(f)
    m = s["metrics"]
    short = exp.replace("cumulative_", "")
    parts = [f"{short:<28s}"]
    for v in VARS:
        key = f"test/Bias_{v}@6h"
        val = m.get(key, 0.0)
        parts.append(f"{val:>+8.4f}")
    print(f"  {' '.join(parts)}")
print(f"  (+ 正向偏大, - 负向偏小, 接近 0 最好)")

# RMSE degradation at long leads (24h vs 6h ratio)
print(f"\n{'='*90}")
print(f"  test/RMSE 24h/6h 衰减比 — 长提前期退化程度诊断")
print(f"{'='*90}")
header = f"{'实验':<28s} " + " ".join(f"{v:>7s}" for v in VARS)
print(f"  {header}")
print(f"  {'-'*65}")
for exp in COMPLETED:
    with open(BASE / exp / "summary.json") as f:
        s = json.load(f)
    m = s["metrics"]
    short = exp.replace("cumulative_", "")
    parts = [f"{short:<28s}"]
    for v in VARS:
        k6 = f"test/RMSE_{v}@6h"
        k24 = f"test/RMSE_{v}@24h"
        if k6 in m and k24 in m and m[k6] > 0:
            ratio = m[k24] / m[k6]
            parts.append(f"{ratio:>7.2f}")
        else:
            parts.append(f"  {'N/A':>5s}")
    print(f"  {' '.join(parts)}")
print(f"  (比值越小越好: c=V4base, s=static_enc, m=morph_graph, w=wind_graph, u=micromet)")
print(f"  V4_best=CA_UG_LossPack (参考): sp=2.44  t2m=1.78  d2m=2.23  u10=1.73  v10=1.89")

# Summary: which variable is most improved by V5
print(f"\n{'='*90}")
print(f"  V5 vs V4: 各变量 24h RMSE 改善幅度 (%)")
print(f"{'='*90}")
base_name = "cumulative_v4_base"
base_m = json.loads((BASE / base_name / "summary.json").read_text())["metrics"]
for exp in COMPLETED[1:]:
    short = exp.replace("cumulative_", "")
    with open(BASE / exp / "summary.json") as f:
        m = json.load(f)["metrics"]
    parts = [f"{short:<28s}"]
    for v in VARS:
        k24 = f"test/RMSE_{v}@24h"
        if k24 in base_m and k24 in m:
            delta = (m[k24] - base_m[k24]) / base_m[k24] * 100
            parts.append(f"{delta:>+6.1f}%")
        else:
            parts.append(f"  {'N/A':>4s}")
    print(f"  {' '.join(parts)}")