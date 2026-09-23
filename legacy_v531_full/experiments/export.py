from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _flatten(prefix: str, obj: Any, out: Dict[str, Any]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), value, out)
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        out[prefix] = obj


def json_to_rows(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, dict) and "files" in obj and isinstance(obj["files"], list):
        rows = []
        for item in obj["files"]:
            row = {"path": item.get("path", "")}
            _flatten("data", item.get("data"), row)
            rows.append(row)
        return rows
    row: Dict[str, Any] = {}
    _flatten("", obj, row)
    return [row]


def export_json_to_csv(input_path: str | Path, output_path: str | Path) -> int:
    obj = json.loads(Path(input_path).read_text(encoding="utf-8"))
    rows = json_to_rows(obj)
    keys = sorted({key for row in rows for key in row.keys()})
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export collected JSON results to CSV")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--out", type=str, default="outputs/experiments/results.csv")
    args = parser.parse_args()
    n = export_json_to_csv(args.input, args.out)
    print(json.dumps({"rows": n, "out": args.out}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()