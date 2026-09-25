from __future__ import annotations

import argparse
import hashlib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .http_public import open_public

WMS_ENDPOINT = "https://titiler.terrascope.be/wms"
LAYER_2021 = "esa-worldcover-map-10m-2021-v2_map"


def worldcover_wms_url(bbox_lonlat: tuple[float, float, float, float], width: int, height: int) -> str:
    """构造 WorldCover WMS 预览 URL。

    WMS 1.1.1 + EPSG:4326 的 bbox 顺序是 lon_min,lat_min,lon_max,lat_max。
    返回 PNG 是真实服务渲染结果，但只是可视化/接入 smoke；正式训练必须读取 COG 分类值。
    """
    params = {
        "service": "WMS",
        "request": "GetMap",
        "version": "1.1.1",
        "layers": LAYER_2021,
        "styles": "",
        "srs": "EPSG:4326",
        "bbox": ",".join(str(x) for x in bbox_lonlat),
        "width": str(width),
        "height": str(height),
        "format": "image/png",
        "transparent": "false",
    }
    return WMS_ENDPOINT + "?" + urllib.parse.urlencode(params)


def download_worldcover_preview(
    out: str | Path,
    *,
    bbox: tuple[float, float, float, float] = (116.20, 39.80, 116.50, 40.05),
    size: int = 256,
) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    url = worldcover_wms_url(tuple(float(x) for x in bbox), size, size)
    req = urllib.request.Request(url, headers={"User-Agent": "UrbanPiDiT-R2/6 real-data-smoke"})
    try:
        with open_public(req, timeout=60) as response:
            content_type = response.headers.get("Content-Type", "")
            payload = response.read()
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise RuntimeError(
            "WorldCover WMS 下载被拒或失败（含 SSRF 防护拒绝非公网地址）；"
            "当前环境可能无外网/DNS；未使用任何伪造静态栅格替代该下载结果。"
        ) from exc
    if not payload.startswith(b"\x89PNG"):
        raise RuntimeError(f"WorldCover WMS 未返回 PNG，Content-Type={content_type!r}")
    out.write_bytes(payload)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/worldcover/worldcover_beijing_smoke.png")
    ap.add_argument("--bbox", nargs=4, type=float, default=[116.20, 39.80, 116.50, 40.05])
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()
    try:
        path = download_worldcover_preview(args.out, bbox=tuple(args.bbox), size=args.size)
    except RuntimeError as exc:
        raise SystemExit(f"WORLDCOVER_DOWNLOAD_FAILED: {exc}") from exc
    print(f"WorldCover WMS smoke preview: {path}")
    print("注意：WMS RGB 只验证真实静态数据接入，不作为正式 land-cover 分类训练真值。")
    print("SHA256:", hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
