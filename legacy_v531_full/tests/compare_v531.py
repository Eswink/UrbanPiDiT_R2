#!/usr/bin/env python3
"""V5.3.1 Cumulative Stack 实验结果横向对比。"""

import json
import os
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

data: dict[str, dict] = {}
for exp in COMPLETED:
    with open(BASE / exp / "summary.json") as f:
        s = json.load(f)
    m = s["metrics"]
    data[exp] = {
        "desc": s["desc"],
        "elapsed_h": round(s["elapsed_sec"] / 3600, 1),
        "RMSE@6h": round(m["test/RMSE@6h"], 2),
        "RMSE@12h": round(m["test/RMSE@12h"], 2),
        "RMSE@18h": round(m["test/RMSE@18h"], 2),
        "RMSE@24h": round(m["test/RMSE@24h"], 2),
        "RMSE_mean_leads": round(m["test/RMSE_mean_leads"], 2),
        "MAE_mean_leads": round(m["test/MAE_mean_leads"], 2),
        "CRPS_mean_leads": round(m["test/CRPS_mean_leads"], 2),
        "ACC@6h": round(m["test/ACC@6h"], 3),
        "ACC@12h": round(m["test/ACC@12h"], 3),
        "ACC@18h": round(m["test/ACC@18h"], 3),
        "ACC@24h": round(m["test/ACC@24h"], 3),
        "ACC_mean_leads": round(m["test/ACC_mean_leads"], 3),
        "loss_recon": round(m["test/loss_recon"], 4),
    }

with open(BASE / "cumulative_v52_process_loss" / "status.json") as f:
    v52_status = json.load(f)

# RMSE table
print("=" * 100)
print("V5.3.1 Cumulative Stack — test/RMSE per Lead (越低越好)")
print("=" * 100)
header = f"{'实验':<28s} {'耗时(h)':>7s} {'6h':>8s} {'12h':>8s} {'18h':>8s} {'24h':>8s} {'mean':>8s}"
print(header)
print("-" * len(header))
for exp in COMPLETED:
    d = data[exp]
    short = exp.replace("cumulative_", "")
    print(
        f"{short:<28s} {d['elapsed_h']:>7.1f} "
        f"{d['RMSE@6h']:>8.2f} {d['RMSE@12h']:>8.2f} "
        f"{d['RMSE@18h']:>8.2f} {d['RMSE@24h']:>8.2f} "
        f"{d['RMSE_mean_leads']:>8.2f}"
    )

# Delta vs v4_base
print("\n--- 相对于 v4_base 的 RMSE 变化 (%) ---")
base_r = data["cumulative_v4_base"]
for exp in COMPLETED[1:]:
    d = data[exp]
    short = exp.replace("cumulative_", "")
    parts = []
    for k in ["RMSE@6h", "RMSE@12h", "RMSE@18h", "RMSE@24h", "RMSE_mean_leads"]:
        delta = (d[k] - base_r[k]) / base_r[k] * 100
        parts.append(f"{delta:+.1f}%")
    print(f"  {short:<28s} {' | '.join(parts)}")

# ACC table
print(f"\n{'实验':<28s} {'ACC@6h':>8s} {'ACC@12h':>8s} {'ACC@18h':>8s} {'ACC@24h':>8s} {'ACC_mean':>8s}")
print("-" * 75)
for exp in COMPLETED:
    d = data[exp]
    short = exp.replace("cumulative_", "")
    print(
        f"{short:<28s} {d['ACC@6h']:>8.3f} {d['ACC@12h']:>8.3f} "
        f"{d['ACC@18h']:>8.3f} {d['ACC@24h']:>8.3f} "
        f"{d['ACC_mean_leads']:>8.3f}"
    )

# Per-variable RMSE@6h
print("\n--- Per-variable test/RMSE@6h ---")
vars_order = ["d2m", "t2m", "sp", "tcc", "tp", "u10", "v10"]
v_header = f"{'实验':<28s} " + " ".join(f"{v:>8s}" for v in vars_order)
print(v_header)
print("-" * len(v_header))
for exp in COMPLETED:
    with open(BASE / exp / "summary.json") as f:
        s = json.load(f)
    m = s["metrics"]
    short = exp.replace("cumulative_", "")
    vals = []
    for v in vars_order:
        vals.append(f"{m[f'test/RMSE_{v}@6h']:>8.4f}" if f"test/RMSE_{v}@6h" in m else f"{m.get(f'test/RMSE_{v}', 0):>8.4f}")
    print(f"{short:<28s} " + " ".join(vals))

# Best / summary
best_rmse_exp = min(COMPLETED, key=lambda e: data[e]["RMSE_mean_leads"])
best_acc_exp = max(COMPLETED, key=lambda e: data[e]["ACC_mean_leads"])
best_6h_exp = min(COMPLETED, key=lambda e: data[e]["RMSE@6h"])

print("\n=== 汇总 ===")
print(f"  已完成实验: {len(COMPLETED)}/29")
print(f"  训练中:     cumulative_v52_process_loss")
print(f"  待启动:     23 个")
print(f"  最佳 RMSE_mean_leads:  {best_rmse_exp} = {data[best_rmse_exp]['RMSE_mean_leads']:.2f}")
print(f"  最佳 RMSE@6h:         {best_6h_exp} = {data[best_6h_exp]['RMSE@6h']:.2f}")
print(f"  最佳 ACC_mean_leads:   {best_acc_exp} = {data[best_acc_exp]['ACC_mean_leads']:.3f}")

# Save
out_path = BASE / "v531_comparison.json"
out_path.write_text(
    json.dumps(
        {
            "completed": data,
            "in_progress": {"cumulative_v52_process_loss": v52_status},
            "pending_count": 23,
            "best_rmse_mean_leads_exp": best_rmse_exp,
            "best_rmse_mean_leads_val": data[best_rmse_exp]["RMSE_mean_leads"],
            "best_acc_mean_leads_exp": best_acc_exp,
            "best_acc_mean_leads_val": data[best_acc_exp]["ACC_mean_leads"],
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)
print(f"\n比对文件: {out_path}")