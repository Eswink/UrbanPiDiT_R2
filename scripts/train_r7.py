"""Native R7 training entry point (bounded smoke config by default)."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.r7_trainer import load_r7_config, run_r7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--config',
        default='configs/r7_native_smoke.yaml',
    )
    args = ap.parse_args()
    cfg = load_r7_config(args.config)
    path = run_r7(cfg)
    print(f'R7 checkpoint: {path}')


if __name__ == '__main__':
    main()
