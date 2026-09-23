from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.download.arco_era5 import extract_arco_era5
from data.download.uci_beijing import download_uci_beijing
from data.download.worldcover_cog import extract_worldcover_roi
from data.download.worldcover_smoke import download_worldcover_preview


def _attempt(name: str, fn) -> dict:
    try:
        path = Path(fn())
        return {"source": name, "status": "success", "path": str(path.resolve()), "bytes": path.stat().st_size}
    except Exception as exc:  # noqa: BLE001 - 此脚本的目标就是形成机器可读的失败审计。
        return {"source": name, "status": "blocked_or_failed", "error_type": type(exc).__name__, "error": str(exc)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default="audit/real_data_download_status.json")
    args = ap.parse_args()
    status = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "No synthetic fallback. Failure is recorded verbatim.",
        "attempts": [
            _attempt("UCI_Beijing_PM25", lambda: download_uci_beijing(ROOT / "data/raw/uci_beijing_network")),
            _attempt(
                "WorldCover_2021_COG",
                lambda: extract_worldcover_roi(ROOT / "data/raw/worldcover/worldcover_beijing_cog_smoke.tif"),
            ),
            _attempt(
                "WorldCover_2021_WMS_preview",
                lambda: download_worldcover_preview(ROOT / "data/raw/worldcover/worldcover_beijing_wms_smoke.png", size=64),
            ),
            _attempt(
                "Google_ARCO_ERA5",
                lambda: extract_arco_era5(ROOT / "data/raw/arco_era5/beijing_smoke.nc", hours=2),
            ),
        ],
    }
    out = ROOT / args.out_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(f"download audit: {out}")


if __name__ == "__main__":
    main()
