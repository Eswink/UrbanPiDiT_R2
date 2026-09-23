from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

UCI_DOI = "10.24432/C5JS49"
UCI_CANONICAL_URL = "https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data"
BEIJING_BOUNDARY_MIRROR = "https://github.com/VIP233333/Coordinate-data-of-the-six-urban-districts-of-Beijing"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_rows(csv_path: Path) -> list[dict[str, object]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"year", "month", "day", "hour", "DEWP", "TEMP", "PRES", "Iws"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"真实站点 CSV 缺少字段: {sorted(missing)}")

    out: list[dict[str, object]] = []
    for source_index, row in enumerate(rows):
        r: dict[str, object] = dict(row)
        try:
            r["_dt"] = datetime(int(str(r["year"])), int(str(r["month"])), int(str(r["day"])), int(str(r["hour"])))
            for key in ["DEWP", "TEMP", "PRES", "Iws"]:
                float(str(r[key]))
        except (ValueError, TypeError):
            continue
        r["_source_index"] = source_index
        out.append(r)

    out.sort(key=lambda x: x["_dt"])
    if len(out) < 16:
        raise ValueError("有效真实记录不足 16 条，无法构造无泄漏 train/val/test smoke 样本")
    return out


def _norm_weather(rows: list[dict[str, object]]) -> np.ndarray:
    """固定物理尺度归一化，避免用未来数据拟合统计量造成 smoke 数据泄漏。"""
    a = np.asarray(
        [[float(str(r["TEMP"])), float(str(r["DEWP"])), float(str(r["PRES"])), float(str(r["Iws"]))] for r in rows],
        dtype=np.float32,
    )
    a[:, 0] /= 40.0
    a[:, 1] /= 40.0
    a[:, 2] = (a[:, 2] - 1000.0) / 50.0
    a[:, 3] /= 100.0
    return a


def _broadcast_history(x: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """把真实点观测广播到网格，仅验证张量契约；绝不代表空间天气真值。"""
    return np.broadcast_to(x[..., None, None], (*x.shape, *hw)).copy().astype(np.float32)


def _point_in_polygon(x: np.ndarray, y: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """向量化 ray-casting；避免为 smoke fixture 强制引入 GIS 依赖。"""
    inside = np.zeros_like(x, dtype=bool)
    xj, yj = polygon[-1]
    for xi, yi in polygon:
        crossing = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        )
        inside ^= crossing
        xj, yj = xi, yi
    return inside


def _load_geojson_outer_ring(path: Path) -> tuple[np.ndarray, str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("type") != "FeatureCollection" or not obj.get("features"):
        raise ValueError("静态 GeoJSON 必须是非空 FeatureCollection")
    feature = obj["features"][0]
    geometry = feature.get("geometry", {})
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if gtype == "Polygon":
        ring = coords[0]
    elif gtype == "MultiPolygon":
        ring = coords[0][0]
    else:
        raise ValueError(f"暂不支持静态 GeoJSON geometry type: {gtype}")
    polygon = np.asarray(ring, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 4:
        raise ValueError("GeoJSON 外环坐标非法")
    name = str(feature.get("properties", {}).get("name", "unknown"))
    return polygon, name


def _real_geo_static(geojson_path: Path, hw: tuple[int, int]) -> tuple[np.ndarray, dict[str, object]]:
    """把真实行政边界栅格化为 smoke static tensor。

    通道:
    0: x_norm（位置编码，不是观测）
    1: y_norm（位置编码，不是观测）
    2: district_mask（来自真实 GeoJSON）
    3: district_edge（由真实 mask 派生）
    """
    polygon, feature_name = _load_geojson_outer_ring(geojson_path)
    lon_min, lat_min = polygon.min(axis=0)
    lon_max, lat_max = polygon.max(axis=0)
    pad_lon = max((lon_max - lon_min) * 0.08, 1e-4)
    pad_lat = max((lat_max - lat_min) * 0.08, 1e-4)
    bbox = (lon_min - pad_lon, lat_min - pad_lat, lon_max + pad_lon, lat_max + pad_lat)

    h, w = hw
    lon = np.linspace(bbox[0], bbox[2], w, dtype=np.float64)
    lat = np.linspace(bbox[3], bbox[1], h, dtype=np.float64)  # raster row 0 = north
    lon_grid, lat_grid = np.meshgrid(lon, lat, indexing="xy")
    mask = _point_in_polygon(lon_grid, lat_grid, polygon).astype(np.float32)

    # 4-neighbour morphological edge，完全由真实边界 mask 派生。
    edge = np.zeros_like(mask)
    edge[1:, :] = np.maximum(edge[1:, :], np.abs(mask[1:, :] - mask[:-1, :]))
    edge[:-1, :] = np.maximum(edge[:-1, :], np.abs(mask[:-1, :] - mask[1:, :]))
    edge[:, 1:] = np.maximum(edge[:, 1:], np.abs(mask[:, 1:] - mask[:, :-1]))
    edge[:, :-1] = np.maximum(edge[:, :-1], np.abs(mask[:, :-1] - mask[:, 1:]))

    yy, xx = np.meshgrid(
        np.linspace(-1, 1, h, dtype=np.float32),
        np.linspace(-1, 1, w, dtype=np.float32),
        indexing="ij",
    )
    static = np.stack([xx, yy, mask, edge], axis=0).astype(np.float32)
    metadata = {
        "feature_name": feature_name,
        "bbox_crs84": [float(v) for v in bbox],
        "mask_fraction": float(mask.mean()),
        "channels": ["x_norm", "y_norm", "real_district_mask", "derived_district_edge"],
    }
    return static, metadata


def _portable_source_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(project_root))
    except ValueError:
        # 测试/外部源可能不在 project_root 内；保留绝对路径仅用于 provenance，
        # 不影响 manifest data artifact 的可移植相对路径。
        return str(resolved)


def _write_manifest(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_split_samples(
    *,
    split: str,
    rows: list[dict[str, object]],
    out_dir: Path,
    static: np.ndarray,
    static_geojson_path: Path,
    manifest_dir: Path,
    project_root: Path,
    dynamic_sha: str,
    static_sha: str,
    history_steps: int,
    coarse_hw: tuple[int, int],
    urban_hw: tuple[int, int],
    stride: int,
) -> list[dict[str, object]]:
    vals = _norm_weather(rows)
    records: list[dict[str, object]] = []
    for start in range(0, len(rows) - history_steps, stride):
        end = start + history_steps
        hist = vals[start:end]
        target = vals[end, 0:1]
        coarse_history = _broadcast_history(hist, coarse_hw)
        urban_history = _broadcast_history(hist[:, 0:1], urban_hw)
        baseline = np.full((1, *urban_hw), hist[-1, 0], dtype=np.float32)
        urban_target = np.full((1, *urban_hw), target[0], dtype=np.float32)

        start_dt = rows[start]["_dt"]
        target_dt = rows[end]["_dt"]
        assert isinstance(start_dt, datetime) and isinstance(target_dt, datetime)
        sid = f"uci_bj_{split}_{start_dt:%Y%m%d%H}_{target_dt:%Y%m%d%H}"
        npz_path = out_dir / f"{sid}.npz"
        np.savez_compressed(
            npz_path,
            coarse_history=coarse_history,
            urban_history=urban_history,
            urban_static=static,
            urban_baseline=baseline,
            urban_target=urban_target,
        )
        records.append(
            {
                "sample_id": sid,
                "path": os.path.relpath(npz_path.resolve(), manifest_dir.resolve()),
                "split": split,
                "start_time": start_dt.isoformat(),
                "target_time": target_dt.isoformat(),
                "source_row_start": int(rows[start]["_source_index"]),
                "source_row_target": int(rows[end]["_source_index"]),
                "dynamic_source": "UCI Beijing PM2.5 dataset / meteorological observations",
                "dynamic_source_doi": UCI_DOI,
                "dynamic_source_file_sha256": dynamic_sha,
                "static_source": "Beijing Dongcheng administrative boundary GeoJSON smoke subset",
                "static_source_file": _portable_source_path(static_geojson_path, project_root),
                "static_source_file_sha256": static_sha,
                "dynamic_is_real_observation": True,
                "static_contains_real_geodata": True,
                "dynamic_spatial_semantics": "single-station observation broadcast for contract smoke only",
                "static_semantics": "real administrative boundary raster + positional channels; NOT urban morphology truth",
                "spatiotemporal_colocation_valid": False,
                "scientific_training_ready": False,
            }
        )
    return records


def build_real_station_smoke(
    csv_path: str | Path,
    out_dir: str | Path,
    manifest_dir: str | Path,
    *,
    static_geojson_path: str | Path = "data/raw/real_smoke/beijing_dongcheng_boundary_smoke.geojson",
    history_steps: int = 4,
    coarse_hw: tuple[int, int] = (8, 8),
    urban_hw: tuple[int, int] = (32, 32),
    stride: int = 2,
) -> dict[str, Path]:
    csv_path = Path(csv_path)
    static_geojson_path = Path(static_geojson_path)
    out_dir = Path(out_dir)
    manifest_dir = Path(manifest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(csv_path)
    project_root = manifest_dir.resolve().parents[2]
    static, static_meta = _real_geo_static(static_geojson_path, urban_hw)
    dynamic_sha = _sha256(csv_path)
    static_sha = _sha256(static_geojson_path)

    # 清理旧 smoke 样本，避免前一版切分残留在目录中被误认为当前产物。
    for old in out_dir.glob("uci_bj_*.npz"):
        old.unlink()

    # 关键：先按原始小时记录切分，再在每个 split 内独立构窗。
    # 因此 train/val/test 不会共享任何原始小时，消除滑窗穿越 split 的时间泄漏。
    n = len(rows)
    i_train = max(history_steps + 1, int(n * 0.70))
    i_val = max(i_train + history_steps + 1, int(n * 0.85))
    i_val = min(i_val, n - (history_steps + 1))
    raw_splits = {
        "train": rows[:i_train],
        "val": rows[i_train:i_val],
        "test": rows[i_val:],
    }

    split_records: dict[str, list[dict[str, object]]] = {}
    for split, split_rows in raw_splits.items():
        recs = _build_split_samples(
            split=split,
            rows=split_rows,
            out_dir=out_dir,
            static=static,
            static_geojson_path=static_geojson_path,
            manifest_dir=manifest_dir,
            project_root=project_root,
            dynamic_sha=dynamic_sha,
            static_sha=static_sha,
            history_steps=history_steps,
            coarse_hw=coarse_hw,
            urban_hw=urban_hw,
            stride=stride,
        )
        if not recs:
            raise RuntimeError(f"{split} split 没有可用窗口；请增加 fixture 时长或降低 history_steps")
        split_records[split] = recs

    paths: dict[str, Path] = {}
    for split, recs in split_records.items():
        path = manifest_dir / f"{split}.jsonl"
        _write_manifest(path, recs)
        paths[split] = path

    # 用 source_row 区间进行机器可审计的不相交证明。
    split_row_ranges = {
        split: [int(rows_split[0]["_source_index"]), int(rows_split[-1]["_source_index"])]
        for split, rows_split in raw_splits.items()
    }
    provenance = {
        "dynamic": {
            "source_csv": _portable_source_path(csv_path, project_root),
            "source_sha256": dynamic_sha,
            "canonical_url": UCI_CANONICAL_URL,
            "doi": UCI_DOI,
            "is_real_observation": True,
            "is_spatial_grid": False,
        },
        "static": {
            "source_geojson": _portable_source_path(static_geojson_path, project_root),
            "source_sha256": static_sha,
            "mirror_url": BEIJING_BOUNDARY_MIRROR,
            "is_real_geodata": True,
            "is_urban_morphology": False,
            **static_meta,
        },
        "records": len(rows),
        "samples_by_split": {k: len(v) for k, v in split_records.items()},
        "history_steps": history_steps,
        "coarse_hw": list(coarse_hw),
        "urban_hw": list(urban_hw),
        "normalization": {"TEMP": "x/40", "DEWP": "x/40", "PRES": "(x-1000)/50", "Iws": "x/100"},
        "split_policy": "raw-hour chronological split first; windows built independently inside each split",
        "split_source_row_ranges": split_row_ranges,
        "scientific_training_ready": False,
        "limitations": [
            "UCI fixture 是北京机场附近单站小时记录，不是 ERA5/HRCLDAS/SMBFD 网格。",
            "把单站动态观测广播到网格只验证数据契约与训练闭环，不代表空间超分真值。",
            "静态输入含真实东城区行政边界，但不是 WorldCover/CNBH/建筑/DEM 等城市形态真值。",
            "动态站点与东城区边界并非严格时空共址，因此本 fixture 仅用于工程 smoke，不用于科学指标。",
        ],
    }
    (manifest_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/real_smoke/beijing_uci_pm25_smoke.csv")
    ap.add_argument("--static-geojson", default="data/raw/real_smoke/beijing_dongcheng_boundary_smoke.geojson")
    ap.add_argument("--out-dir", default="data/processed/real_smoke")
    ap.add_argument("--manifest-dir", default="data/manifests/real_smoke")
    ap.add_argument("--history-steps", type=int, default=4)
    ap.add_argument("--coarse-hw", type=int, nargs=2, default=[8, 8])
    ap.add_argument("--urban-hw", type=int, nargs=2, default=[32, 32])
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()
    paths = build_real_station_smoke(
        args.csv,
        args.out_dir,
        args.manifest_dir,
        static_geojson_path=args.static_geojson,
        history_steps=args.history_steps,
        coarse_hw=tuple(args.coarse_hw),
        urban_hw=tuple(args.urban_hw),
        stride=args.stride,
    )
    print("真实数据 smoke 已转换为 R² data contract:")
    for split, path in paths.items():
        print(f"  {split}: {path}")


if __name__ == "__main__":
    main()
