#!/usr/bin/env python3
"""从完整提取的 V4 metrics JSON 生成精简汇总表。
只保留 best 值 + 每条曲线的首、尾、最优三个点。"""
from __future__ import annotations

import json
import sys
from pathlib import Path


FULL = Path("/3240608030/weather-q/UrbanPiDiT_V4/logs/v4_metrics_extracted.json")
COMPACT = Path("/3240608030/weather-q/UrbanPiDiT_V4/logs/v4_metrics_compact.json")


def compact_curve(curve: list[dict]) -> list[dict]:
    """从完整曲线中提取 epoch 0, epoch 末, 最优点 三个关键帧。"""
    if not curve:
        return []
    first = curve[0]
    last = curve[-1]
    best = min(curve, key=lambda x: x["value"])
    result = [first]
    if best not in (first, last):
        result.append(best)
    if last not in (first, best):
        result.append(last)
    return result


def main() -> int:
    with open(FULL, encoding="utf-8") as f:
        full = json.load(f)

    compact: dict[str, Any] = {}
    for exp_name, exp_data in full.items():
        compact[exp_name] = {}
        for stage, stage_data in exp_data.items():
            if "error" in stage_data:
                compact[exp_name][stage] = stage_data
                continue
            best = stage_data.get("best", {})
            curves = stage_data.get("curves", {})
            compact_curves = {
                k: compact_curve(v) for k, v in curves.items()
            }
            compact[exp_name][stage] = {
                "best": best,
                "curves_compact": compact_curves,
                "num_epochs": stage_data.get("num_epochs", 0),
                "train_best": stage_data.get("train_best", {}),
            }

    with open(COMPACT, "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2, ensure_ascii=False)

    mb = COMPACT.stat().st_size / (1024 * 1024)
    print(f"精简版已写入: {COMPACT} ({mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())