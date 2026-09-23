#!/usr/bin/env python3
"""汇总 V4 实验的 TFEvent 日志关键指标到 JSON。"""

import json
import sys
from pathlib import Path

COMPACT_PATH = Path(
    "/3240608030/weather-q/UrbanPiDiT_V4/logs/v4_metrics_compact.json"
)
OUTPUT_PATH = Path(
    "/3240608030/weather-q/UrbanPiDiT_V4/logs/v4_summary.json"
)

compact = json.loads(COMPACT_PATH.read_text(encoding="utf-8"))

summary: dict[str, dict[str, float]] = {}

for exp_name in sorted(compact.keys()):
    exp = compact[exp_name]
    for stage_name in ["stage1_6h", "stage2_multi_lead"]:
        if stage_name not in exp:
            continue
        stage = exp[stage_name]
        best = stage.get("best", {})
        key_label = f"{exp_name}/{stage_name}"
        entry: dict[str, float] = {}

        val_keys = [
            "val/RMSE/mean@6", "val/MAE/mean@6",
            "val/RMSE_mean_leads", "val/MAE_mean_leads", "val/loss",
        ]
        test_keys = [
            "test/RMSE@6h", "test/MAE@6h",
            "test/RMSE_mean_leads", "test/MAE_mean_leads", "test/CRPS@6h",
        ]

        for m in val_keys + test_keys:
            if m in best:
                entry[m] = round(best[m]["value"], 4)

        for v in ["d2m", "sp", "t2m", "tcc", "tp", "u10", "v10"]:
            for suffix in ["@6h", ""]:
                k = f"test/RMSE_{v}{suffix}"
                if k in best:
                    entry[f"test/RMSE/{v}"] = round(best[k]["value"], 6)
                    break

        if entry:
            summary[key_label] = entry

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(f"wrote {OUTPUT_PATH}")
print(f"experiments: {len(compact)}, entries: {len(summary)}")

# Stage1 test/RMSE@6h 排名
stage1 = [
    (k, v) for k, v in summary.items()
    if k.endswith("/stage1_6h") and "test/RMSE@6h" in v
]
stage1.sort(key=lambda x: x[1]["test/RMSE@6h"])
print("\n=== Stage1 test/RMSE@6h (越低越好) ===")
for i, (name, entry) in enumerate(stage1):
    exp = name.split("/")[0]
    print(f"  {i+1:2d}. {exp:<25s} RMSE={entry['test/RMSE@6h']:.2f}")

# Stage2 test/RMSE_mean_leads 排名
stage2 = [
    (k, v) for k, v in summary.items()
    if k.endswith("/stage2_multi_lead") and "test/RMSE_mean_leads" in v
]
stage2.sort(key=lambda x: x[1]["test/RMSE_mean_leads"])
print("\n=== Stage2 test/RMSE_mean_leads (multi-lead, 越低越好) ===")
for i, (name, entry) in enumerate(stage2):
    exp = name.split("/")[0]
    print(f"  {i+1:2d}. {exp:<25s} RMSE={entry['test/RMSE_mean_leads']:.2f}")