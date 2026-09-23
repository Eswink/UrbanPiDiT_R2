#!/usr/bin/env python3
"""从 UrbanPiDiT V4 logs/ 中提取 TensorBoard 标量指标到 JSON。"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 抑制 TF 告警

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


LOG_ROOT = Path("/3240608030/weather-q/UrbanPiDiT_V4/logs")
OUTPUT = Path("/3240608030/weather-q/UrbanPiDiT_V4/logs/v4_metrics_extracted.json")

STAGES = ["stage1_6h", "stage2_multi_lead"]

# 指标分层：
#   epoch: epoch number
#   val/*: validation (epoch-level)
#   test/*: test (epoch-level, 通常只在 stage1 有)
#   train/*_epoch: training (epoch-level)
#   train/*_step: per-step (跳过)
#   特殊: hp_metric, lr-AdamW

LEAD_TAGS_PATTERN = re.compile(
    r"(val|test)/(RMSE|MAE|CRPS)(?:_([a-z0-9]+))?(@(\d+)h)?(#\d+)?"
)
NON_LEAD_TAGS = {
    "val/loss", "val/loss_recon", "val/fft", "val/grad",
    "val/phys_spatial", "val/phys_tp_nonneg", "val/phys_tcc_01",
    "val/phys_dew_leq_t", "val/phys_sp_nonneg", "val/phys_wind_div",
    "val/main_lead_steps", "val/MAE", "val/RMSE", "val/MAE_d2m",
    "val/MAE_sp", "val/MAE_t2m", "val/MAE_tcc", "val/MAE_tp",
    "val/MAE_u10", "val/MAE_v10",
    "val/RMSE_d2m", "val/RMSE_sp", "val/RMSE_t2m", "val/RMSE_tcc",
    "val/RMSE_tp", "val/RMSE_u10", "val/RMSE_v10",
    "val/MAE_mean_leads", "val/RMSE_mean_leads",
    "train/loss_epoch", "train/loss_recon_epoch",
    "train/RMSE@6h_epoch", "train/MAE@6h_epoch",
    "test/loss", "test/loss_recon",
    "epoch",
}


def parse_metric_tag(tag: str) -> dict[str, Any] | None:
    """解析 val/RMSE_d2m@6h#1 类指标名。"""
    m = LEAD_TAGS_PATTERN.match(tag)
    if m is None:
        return None
    return {
        "split": m.group(1),   # val / test
        "metric": m.group(2),  # RMSE / MAE / CRPS
        "variable": m.group(3) or "mean",
        "lead_hours": m.group(5) or "mean",
    }


def load_events(logdir: Path) -> dict[str, list[dict[str, Any]]]:
    """加载一个 stage 目录下所有 tfevents 标量，聚合后按 tag 返回 (step, wall_time, value) 列表。"""
    ea = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    result: dict[str, list[dict[str, Any]]] = {}
    for tag in tags:
        events = ea.Scalars(tag)
        result[tag] = [{"step": e.step, "wall_time": e.wall_time, "value": e.value} for e in events]
    return result


def best_value(series: list[dict[str, Any]], mode: str = "min") -> dict[str, Any] | None:
    """从时序中取最优值（RMSE/MAE/loss 越小越好）。"""
    if not series:
        return None
    if mode == "min":
        return min(series, key=lambda x: x["value"])
    return max(series, key=lambda x: x["value"])


def extract_summary(events: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """从聚合 events 中提取每个 lead/metic 组合的最优值和对应 epoch。"""
    # 先找 epoch 映射
    epoch_map: dict[int, int] = {}
    if "epoch" in events:
        for pt in events["epoch"]:
            epoch_map[pt["step"]] = int(pt["value"])
    elif "hp_metric" not in events:
        # fallback: 用 step 当作 epoch（某些日志无 epoch tag）
        pass

    def epoch_of(pt: dict[str, Any]) -> int:
        return epoch_map.get(pt["step"], pt["step"])

    lead_best: dict[str, Any] = {}
    curves: dict[str, list[dict[str, Any]]] = {}

    for tag, series in events.items():
        parsed = parse_metric_tag(tag)
        if parsed is None:
            continue

        split, metric, var, lead = parsed["split"], parsed["metric"], parsed["variable"], parsed["lead_hours"]

        # 只取 val/test（epoch 级）；跳过 step 级
        if split not in ("val", "test"):
            continue

        best = best_value(series, mode="min")
        key = f"{split}/{metric}/{var}@{lead}"
        if best is not None:
            lead_best[key] = {
                "value": best["value"],
                "step": best["step"],
                "epoch": epoch_of(best),
                "wall_time": best["wall_time"],
            }

        # 曲线数据（全量 epoch 序列）
        curve_key = f"{key}_curve"
        curves[curve_key] = [
            {"step": pt["step"], "epoch": epoch_of(pt), "value": pt["value"]} for pt in series
        ]

    # 直接 val/RMSE_mean_leads, val/MAE_mean_leads, val/loss 等
    for tag in NON_LEAD_TAGS:
        if tag not in events:
            continue
        series = events[tag]
        best = best_value(series, mode="min")
        if best is not None:
            lead_best[tag] = {
                "value": best["value"],
                "step": best["step"],
                "epoch": epoch_of(best),
                "wall_time": best["wall_time"],
            }
        curves[f"{tag}_curve"] = [
            {"step": pt["step"], "epoch": epoch_of(pt), "value": pt["value"]} for pt in series
        ]

    # test metrics in stage1 have CRPS, but stage2 may lack them
    test_tags = [t for t in events if t.startswith("test/") and not t.endswith("_step")]
    for tag in test_tags:
        if tag in lead_best or tag in curves:
            continue
        series = events[tag]
        best = best_value(series, mode="min")
        if best is not None:
            lead_best[tag] = {
                "value": best["value"],
                "step": best["step"],
                "epoch": epoch_of(best),
                "wall_time": best["wall_time"],
            }
        curves[f"{tag}_curve"] = [
            {"step": pt["step"], "epoch": epoch_of(pt), "value": pt["value"]} for pt in series
        ]

    # train epoch-level metrics (loss, RMSE@6h_epoch etc.)
    train_summary: dict[str, Any] = {}
    train_tags = [t for t in events if t.startswith("train/") and t.endswith("_epoch")]
    for tag in train_tags:
        series = events[tag]
        best = best_value(series, mode="min")
        if best is not None:
            train_summary[tag] = {
                "value": best["value"],
                "step": best["step"],
                "epoch": epoch_of(best),
                "wall_time": best["wall_time"],
            }

    return {
        "best": lead_best,
        "curves": curves,
        "train_best": train_summary,
        "num_epochs": max(epoch_map.values()) if epoch_map else 0,
    }


def main() -> int:
    log_root = LOG_ROOT
    experiment_names = sorted(
        d.name for d in log_root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )

    result: dict[str, Any] = {}
    errors: list[str] = []

    for exp in experiment_names:
        exp_dir = log_root / exp
        result[exp] = {}
        print(f"\n[processing] {exp}")

        for stage in STAGES:
            stage_dir = exp_dir / stage
            if not stage_dir.is_dir():
                print(f"  {stage}: 目录不存在，跳过")
                continue

            tf_files = sorted(stage_dir.glob("events.out.tfevents.*"))
            if not tf_files:
                print(f"  {stage}: 无 tfevents 文件，跳过")
                continue

            print(f"  {stage}: {len(tf_files)} 个 event 文件")
            try:
                events = load_events(stage_dir)
                summary = extract_summary(events)
                result[exp][stage] = summary
                n_metrics = len(summary.get("best", {}))
                print(f"  {stage}: 提取 {n_metrics} 个最优指标, {len(summary.get('curves', {}))} 条曲线")
            except Exception as exc:
                msg = f"{exp}/{stage}: {exc}"
                errors.append(msg)
                print(f"  {stage}: 加载失败 - {exc}")
                result[exp][stage] = {"error": str(exc)}

    # 写 JSON（缩进 2，中文不转义）
    output_path = OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n=== 完成 ===")
    print(f"输出: {output_path}")
    print(f"实验组: {len(result)}")
    total_metrics = sum(
        len(exp_v.get(stage, {}).get("best", {}))
        for exp_v in result.values()
        for stage in STAGES
        if stage in exp_v
    )
    print(f"指标组合数: {total_metrics}")
    if errors:
        print(f"错误: {len(errors)}")
        for e in errors:
            print(f"  - {e}")

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())