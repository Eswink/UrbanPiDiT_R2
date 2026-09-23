from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.download.wb2_public_probe import probe_weatherbench2, probe_arco


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args=ap.parse_args()
    path=Path(args.out)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    report={
        "weatherbench2": probe_weatherbench2(),
        "arco": probe_arco(),
    }
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    print(json.dumps({
        name: {
            "source": item["source"],
            "variables": sorted(item["variables"]),
            "missing": item["missing_variables"],
            "field_values_loaded": item["field_values_loaded"],
        }
        for name,item in report.items()
    }))


if __name__=="__main__":
    main()
