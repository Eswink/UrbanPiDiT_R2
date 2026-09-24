"""Recursive R7 training entry point (bounded smoke config by default)."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.r7_recursive_trainer import (
    load_recursive_config,
    run_recursive,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--config',
        default='configs/r7_recursive_smoke.yaml',
    )
    args = ap.parse_args()
    path = run_recursive(load_recursive_config(args.config))
    print(f'R7 recursive checkpoint: {path}')


if __name__ == '__main__':
    main()
