from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.download.uci_beijing import download_uci_beijing, sha256
from data.preprocess.real_station_smoke import build_real_station_smoke


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--network",
        action="store_true",
        help="尝试从 UCI/镜像联网下载；默认使用已捕获的真实 fixture",
    )
    args = ap.parse_args()
    raw_dir = ROOT / "data/raw/real_smoke"
    bundled = raw_dir / "beijing_uci_pm25_smoke.csv"
    try:
        if args.network:
            csv_path = download_uci_beijing(raw_dir / "downloaded")
        else:
            # 复制到独立 captured 目录，确保准备链路不会直接读取原 fixture 路径。
            csv_path = download_uci_beijing(raw_dir / "captured", fixture=bundled)
    except RuntimeError as exc:
        raise SystemExit(f"REAL_DYNAMIC_DOWNLOAD_FAILED: {exc}") from exc

    print("真实数据 SHA256:", sha256(csv_path))
    paths = build_real_station_smoke(
        csv_path,
        ROOT / "data/processed/real_smoke",
        ROOT / "data/manifests/real_smoke",
        static_geojson_path=raw_dir / "beijing_dongcheng_boundary_smoke.geojson",
    )
    for split, path in paths.items():
        print(split, path)


if __name__ == "__main__":
    main()
