from __future__ import annotations

import argparse
import math
from pathlib import Path

S3_PREFIX = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
VERSIONS = {2020: "v100", 2021: "v200"}


def _format_tile_coord(value: int, positive: str, negative: str, digits: int) -> str:
    prefix = positive if value >= 0 else negative
    return f"{prefix}{abs(value):0{digits}d}"


def worldcover_tile_id(lon: float, lat: float) -> str:
    """返回包含给定点的 3°×3° WorldCover lower-left tile ID，例如 N39E114。"""
    lon0 = math.floor(lon / 3.0) * 3
    lat0 = math.floor(lat / 3.0) * 3
    return _format_tile_coord(lat0, "N", "S", 2) + _format_tile_coord(lon0, "E", "W", 3)


def worldcover_cog_url(tile: str, *, year: int = 2021) -> str:
    version = VERSIONS[year]
    return (
        f"{S3_PREFIX}/{version}/{year}/map/"
        f"ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"
    )


def extract_worldcover_roi(
    out: str | Path,
    *,
    bbox: tuple[float, float, float, float] = (116.37, 39.85, 116.46, 39.98),
    year: int = 2021,
) -> Path:
    """从公开 COG 通过 HTTP range 读取极小 ROI，并保留原始分类值。

    bbox: lon_min, lat_min, lon_max, lat_max (EPSG:4326)。
    当前 smoke 实现要求 ROI 位于单个 3° tile；正式大 ROI 可在上层做多 tile mosaic。
    """
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError as exc:
        raise RuntimeError("WorldCover COG 需要 rasterio；请执行 `pip install -r requirements-data.txt`") from exc

    lon_min, lat_min, lon_max, lat_max = map(float, bbox)
    corners = {
        worldcover_tile_id(lon_min + 1e-9, lat_min + 1e-9),
        worldcover_tile_id(lon_max - 1e-9, lat_min + 1e-9),
        worldcover_tile_id(lon_min + 1e-9, lat_max - 1e-9),
        worldcover_tile_id(lon_max - 1e-9, lat_max - 1e-9),
    }
    if len(corners) != 1:
        raise ValueError(f"smoke ROI 跨越多个 WorldCover tile: {sorted(corners)}；请拆分后 mosaic")
    tile = next(iter(corners))
    url = worldcover_cog_url(tile, year=year)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
            with rasterio.open(url) as src:
                if src.crs is None or src.crs.to_epsg() != 4326:
                    raise RuntimeError(f"WorldCover COG CRS 非预期 EPSG:4326: {src.crs}")
                window = from_bounds(lon_min, lat_min, lon_max, lat_max, transform=src.transform)
                window = window.round_offsets().round_lengths()
                data = src.read(1, window=window)
                if data.size == 0:
                    raise RuntimeError("WorldCover ROI 读取为空")
                transform = src.window_transform(window)
                profile = src.profile.copy()
                profile.update(
                    driver="GTiff",
                    height=data.shape[0],
                    width=data.shape[1],
                    count=1,
                    transform=transform,
                    compress="deflate",
                )
                with rasterio.open(out, "w", **profile) as dst:
                    dst.write(data, 1)
    except Exception as exc:
        if isinstance(exc, (ValueError, RuntimeError)):
            raise
        raise RuntimeError(
            "无法通过 HTTP range 读取 ESA WorldCover COG。"
            "当前环境可能无 DNS/外网；未使用 WMS RGB 或合成分类值替代。"
        ) from exc
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="提取 ESA WorldCover 10m 原始分类值的极小 ROI。")
    ap.add_argument("--out", default="data/raw/worldcover/worldcover_beijing_cog_smoke.tif")
    ap.add_argument("--bbox", nargs=4, type=float, default=[116.37, 39.85, 116.46, 39.98])
    ap.add_argument("--year", type=int, choices=[2020, 2021], default=2021)
    args = ap.parse_args()
    try:
        path = extract_worldcover_roi(args.out, bbox=tuple(args.bbox), year=args.year)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"WORLDCOVER_COG_DOWNLOAD_FAILED: {exc}") from exc
    print(f"WorldCover categorical COG ROI: {path}")
    print(f"tile={worldcover_tile_id(args.bbox[0], args.bbox[1])}")


if __name__ == "__main__":
    main()
