from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

try:
    from .smoke_test_v51 import run_smoke
except ImportError:  # pragma: no cover - 支持 python experiments/*.py 直接运行
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.smoke_test_v51 import run_smoke


def run_v52_smoke(config: str) -> Dict[str, Any]:
    result = run_smoke(config)
    micromet_keys = list(result.get("diagnostics", {}).get("micromet", []) or [])
    result["protocol"] = "v52_smoke_forward"
    result["interleaved_diagnostics"] = {
        "has_block_residual_norm": any("micromet_block_" in key and "residual_norm" in key for key in micromet_keys),
        "has_token_residual_norm": any("micromet_block_" in key and "token_residual_norm" in key for key in micromet_keys),
        "keys": micromet_keys,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="V5.2 MicroMetRefine lightweight dummy forward smoke test")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--dry_run", "--dry-run", action="store_true", help="accepted for manifest-style invocation; smoke still runs a lightweight forward")
    args = parser.parse_args()

    result = run_v52_smoke(args.config)
    result["dry_run"] = bool(args.dry_run)
    print(result)
    if not result["finite"]:
        raise SystemExit(1)
    if not result["interleaved_diagnostics"]["has_block_residual_norm"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()