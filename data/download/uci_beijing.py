from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

UCI_ZIP = "https://archive.ics.uci.edu/static/public/381/beijing+pm2.5.zip"
MIRROR_CSV = (
    "https://raw.githubusercontent.com/634671436/Air_Pollution_Forcast_Beijing/"
    "master/Air_Pollution_Forcast_Beijing/resource/PRSA_data_2010.1.1-2014.12.31.csv"
)
CSV_NAME = "PRSA_data_2010.1.1-2014.12.31.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dst: Path, timeout: int = 60) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "UrbanPiDiT-R2/6 real-data-smoke"})
    with urllib.request.urlopen(req, timeout=timeout) as r, dst.open("wb") as f:
        shutil.copyfileobj(r, f)


def download_uci_beijing(out_dir: str | Path, *, fixture: str | Path | None = None) -> Path:
    """下载北京真实逐小时气象记录。

    顺序：UCI 官方 ZIP -> GitHub 文本镜像。若传入 fixture，仅复制已有真实 fixture，
    主要用于无外网 CI/sandbox；不会生成任何合成观测。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / CSV_NAME

    if fixture is not None:
        src = Path(fixture)
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, out_csv)
        return out_csv

    errors: list[str] = []
    zip_path = out_dir / "beijing_pm25.zip"
    try:
        _download(UCI_ZIP, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            candidates = [n for n in zf.namelist() if n.endswith(CSV_NAME)]
            if not candidates:
                raise RuntimeError(f"UCI ZIP 中未找到 {CSV_NAME}")
            with zf.open(candidates[0]) as src, out_csv.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        return out_csv
    except Exception as exc:  # noqa: BLE001 - 下载器需要记录完整 fallback 原因
        errors.append(f"UCI: {exc}")
        zip_path.unlink(missing_ok=True)

    try:
        _download(MIRROR_CSV, out_csv)
        return out_csv
    except Exception as exc:  # noqa: BLE001
        errors.append(f"mirror: {exc}")
        out_csv.unlink(missing_ok=True)

    raise RuntimeError("真实数据下载失败；未使用合成数据回退。\n" + "\n".join(errors))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/raw/uci_beijing")
    ap.add_argument("--fixture", default=None, help="无网环境下复制已捕获的真实数据 fixture")
    args = ap.parse_args()
    p = download_uci_beijing(args.out_dir, fixture=args.fixture)
    print(f"已准备真实北京气象数据: {p}")
    print(f"SHA256: {sha256(p)}")


if __name__ == "__main__":
    main()
