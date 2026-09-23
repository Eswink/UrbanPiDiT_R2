r"""
UrbanPiDiT 与基线模型的公平对比入口。

核心实现拆分在：
- ``fair_comparison_urban.py``：UrbanPiDiT 评估与统一 schema；
- ``fair_comparison_records.py``：结果合并、外部 JSON 兼容与 Markdown 表格。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from .fair_comparison_records import run_fair_comparison
except ImportError:
    from fair_comparison_records import run_fair_comparison


def build_parser() -> argparse.ArgumentParser:
    r"""
    构建 CLI parser。

    Returns
    ----
    argparse.ArgumentParser
        命令行 parser。
    """

    parser = argparse.ArgumentParser(description="Run UrbanPiDiT fair comparison against baselines")
    parser.add_argument("--urbanpidit_config", required=True)
    parser.add_argument("--urbanpidit_ckpt", required=True)
    parser.add_argument("--baseline_config", default="configs/fair_baselines.yaml")
    parser.add_argument("--external_config", default="configs/baselines_external_weather_suite.yaml")
    parser.add_argument("--external_results", default=None)
    parser.add_argument("--out_dir", default="outputs/fair_comparison")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip_forecast_baselines", action="store_true")
    return parser


def main() -> None:
    r"""
    命令行入口。

    Parameters
    ----
    None

    Returns
    ----
    None
    """

    result = run_fair_comparison(build_parser().parse_args())
    print(json.dumps({"out": "fair_comparison", "num_records": len(result["records"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()