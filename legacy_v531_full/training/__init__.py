"""UrbanPiDiT 训练与评估共享入口。"""

from training.trainer import load_config, run_training_from_config

__all__ = ["load_config", "run_training_from_config"]