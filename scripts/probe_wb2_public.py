from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.download.wb2_public_probe import probe_weatherbench2


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args=ap.parse_args()
    path=Path(args.out)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    report=probe_weatherbench2()
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    print(json.dumps({
        "source": report["source"],
        "variables": sorted(report["variables"]),
        "missing": report["missing_variables"],
        "field_values_loaded": report["field_values_loaded"],
    }))


if __name__=="__main__":
    main()
