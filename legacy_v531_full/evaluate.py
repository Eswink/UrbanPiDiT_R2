"""UrbanPiDiT V5.3.1 MorphoProcessDiT 评估/测试入口脚本。

用法示例：
    python evaluate.py --config configs/beijing.yaml --ckpt path/to/ckpt.ckpt
"""

from __future__ import annotations

import argparse

from training.evaluator import evaluate_checkpoint
from training.trainer import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="配置文件路径 (yaml)")
    parser.add_argument("--ckpt", type=str, required=True, help="Lightning checkpoint 路径")
    parser.add_argument("--output", type=str, default=None, help="可选：保存指标 JSON 的路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate_checkpoint(cfg, args.ckpt, output_path=args.output)


if __name__ == "__main__":
    main()