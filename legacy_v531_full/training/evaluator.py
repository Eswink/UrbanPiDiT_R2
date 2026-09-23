"""UrbanPiDiT 评估工具。

本模块复用训练/评估配置构造，返回 Lightning test 阶段记录的全部指标。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

try:
    import pytorch_lightning as pl
except Exception as e:
    raise ImportError("评估脚本需要安装 pytorch_lightning。请先安装后再运行。") from e

from data.loader import MetroWeatherDataModule
from pidit_lit import UrbanPiDiTLitModule
from utils.config_builder import build_datamodule_kwargs, build_litmodule_kwargs


def evaluate_checkpoint(
    cfg: Dict[str, Any],
    ckpt_path: str,
    *,
    output_path: Optional[str] = None,
) -> Dict[str, float]:
    """评估 checkpoint 并返回指标字典。"""

    if not ckpt_path:
        raise ValueError("ckpt_path 为空，无法评估")
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"找不到 checkpoint：{ckpt_path}")

    seed = int(cfg.get("train", {}).get("seed", 42))
    pl.seed_everything(seed, workers=True)

    dm = _build_eval_datamodule(cfg)
    dm.prepare_data()
    dm.setup(stage="test")

    lit = UrbanPiDiTLitModule.load_from_checkpoint(ckpt_path, **build_litmodule_kwargs(cfg))
    eval_cfg = dict(cfg.get("eval", {}) or {})
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        deterministic=True,
        limit_test_batches=eval_cfg.get("limit_test_batches", 1.0),
    )

    results = trainer.test(lit, datamodule=dm)
    metrics = dict(results[0] if results else {})
    metrics = {str(key): _to_float(value) for key, value in metrics.items()}

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    return metrics


def _build_eval_datamodule(cfg: Dict[str, Any]) -> MetroWeatherDataModule:
    eval_cfg = dict(cfg.get("eval", {}) or {})
    train_cfg = dict(cfg.get("train", {}) or {})
    batch_size = int(eval_cfg.get("batch_size", train_cfg.get("batch_size", 1)))
    num_workers = int(eval_cfg.get("num_workers", train_cfg.get("num_workers", 8)))
    return MetroWeatherDataModule(
        **build_datamodule_kwargs(cfg, batch_size=batch_size, num_workers=num_workers)
    )


def _to_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    return float(value)