from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.download.ncar_public_probe import probe_ncar_examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    report = probe_ncar_examples()
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    with out.open('x', encoding='utf-8') as stream:
        stream.write(text)
    print(text)


if __name__ == '__main__':
    main()
