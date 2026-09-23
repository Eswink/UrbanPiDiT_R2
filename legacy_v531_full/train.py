"""UrbanPiDiT V5.3.1 MorphoProcessDiT 训练入口脚本。

用法示例：
    python train.py --config configs/beijing.yaml
    python train.py --config configs/beijing_v4_two_stage.yaml
    python train.py --config configs/urbanpidit_v531_reviewer_evidence.yaml
"""

from __future__ import annotations

import argparse

from training.trainer import load_config, run_training_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="配置文件路径 (yaml)")
    parser.add_argument("--resume", action="store_true", help="启用断点训练，自动识别阶段并继续")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="指定断点 checkpoint 路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_training_from_config(cfg, resume=args.resume, resume_ckpt=args.resume_ckpt)


if __name__ == "__main__":
    main()