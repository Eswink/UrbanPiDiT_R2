from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_results(root: str | Path, *, pattern: str = "*.json") -> Dict[str, Any]:
    root = Path(root)
    files = sorted(root.rglob(pattern))
    return {"root": str(root), "files": [{"path": str(path), "data": _load_json(path)} for path in files]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect experiment JSON outputs")
    parser.add_argument("--root", type=str, default="outputs")
    parser.add_argument("--pattern", type=str, default="*.json")
    parser.add_argument("--out", type=str, default="outputs/experiments/collected_results.json")
    args = parser.parse_args()

    result = collect_results(args.root, pattern=args.pattern)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"num_files": len(result["files"]), "out": str(out)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()