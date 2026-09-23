from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

ARCO_025 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
OPTIONAL_DEPS = ("xarray", "pandas", "zarr", "gcsfs")


def _require_optional_dependencies() -> None:
    missing = [name for name in OPTIONAL_DEPS if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "ARCO ERA5 下载缺少可选依赖: "
            + ", ".join(missing)
            + "。请执行 `pip install -r requirements-data.txt` 后重试。"
        )


def extract_arco_era5(
    out: str | Path,
    *,
    time: str = "2021-07-01T00:00:00",
    hours: int = 6,
    lat_min: float = 39.0,
    lat_max: float = 41.0,
    lon_min: float = 115.0,
    lon_max: float = 118.0,
) -> Path:
    _require_optional_dependencies()
    import pandas as pd
    import xarray as xr

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        ds = xr.open_zarr(ARCO_025, chunks=None, storage_options={"token": "anon"})
    except Exception as exc:  # 网络、GCS 或上游 Zarr 变化都必须显式失败。
        raise RuntimeError(
            "无法匿名打开 Google ARCO ERA5 公共 Zarr。"
            "当前环境可能无外网/DNS，或上游 schema/依赖发生变化；未使用任何合成数据回退。"
        ) from exc

    t0 = pd.Timestamp(time)
    t1 = t0 + pd.Timedelta(hours=hours - 1)
    expected = [
        "2m_temperature",
        "2m_dewpoint_temperature",
        "mean_sea_level_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ]
    needed = [v for v in expected if v in ds]
    if not needed:
        raise RuntimeError("ARCO schema 未发现预期 surface variables，请检查上游版本。")

    # ERA5 latitude 常为北到南递减。若上游排序改变则按坐标方向自适应。
    lat_values = ds["latitude"].values
    lat_slice = slice(lat_max, lat_min) if lat_values[0] > lat_values[-1] else slice(lat_min, lat_max)
    sub = ds[needed].sel(
        time=slice(t0, t1),
        latitude=lat_slice,
        longitude=slice(lon_min, lon_max),
    )
    if int(sub.sizes.get("time", 0)) == 0:
        raise RuntimeError("ARCO ERA5 子集为空；请检查时间/ROI 与上游坐标。")
    sub.to_netcdf(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="从 Google ARCO ERA5 匿名公共 Zarr 提取极小北京 ROI。")
    ap.add_argument("--out", default="data/raw/arco_era5/beijing_smoke.nc")
    ap.add_argument("--time", default="2021-07-01T00:00:00")
    ap.add_argument("--hours", type=int, default=6)
    ap.add_argument("--lat-min", type=float, default=39.0)
    ap.add_argument("--lat-max", type=float, default=41.0)
    ap.add_argument("--lon-min", type=float, default=115.0)
    ap.add_argument("--lon-max", type=float, default=118.0)
    args = ap.parse_args()
    try:
        path = extract_arco_era5(
            args.out,
            time=args.time,
            hours=args.hours,
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            lon_min=args.lon_min,
            lon_max=args.lon_max,
        )
    except RuntimeError as exc:
        raise SystemExit(f"ARCO_ERA5_DOWNLOAD_FAILED: {exc}") from exc
    print(f"ERA5 smoke subset: {path}")


if __name__ == "__main__":
    main()
