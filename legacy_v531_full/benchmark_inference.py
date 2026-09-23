import argparse
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml

try:
    import pytorch_lightning as pl
except Exception as e:
    raise ImportError("该脚本需要安装 pytorch_lightning。请先安装后再运行。") from e

from data.loader import MetroWeatherDataModule
from pidit_lit import UrbanPiDiTLitModule
from utils.config_builder import build_datamodule_kwargs, build_litmodule_kwargs


def _build_lit(cfg: Dict, ckpt_path: str) -> UrbanPiDiTLitModule:
    return UrbanPiDiTLitModule.load_from_checkpoint(ckpt_path, **build_litmodule_kwargs(cfg))


def _build_dm(cfg: Dict, *, batch_size: int, num_workers: int) -> MetroWeatherDataModule:
    kwargs = build_datamodule_kwargs(cfg, batch_size=batch_size, num_workers=num_workers)
    kwargs["static_perturb"] = {"mode": "none"}

    dm = MetroWeatherDataModule(**kwargs)
    dm.prepare_data()
    dm.setup(stage="test")
    return dm


def _resolve_lead_steps(cfg: Dict, lead_steps: Optional[int]) -> int:
    if lead_steps is not None:
        return int(lead_steps)
    forecast_cfg = dict(cfg.get("forecast", {}) or {})
    leads = forecast_cfg.get("eval_lead_times", forecast_cfg.get("lead_times", [cfg.get("delta_t", 1)]))
    if not leads:
        return int(cfg.get("delta_t", 1))
    return int(list(leads)[0])


def _get_batch(dm: MetroWeatherDataModule, split: str):
    split = str(split).lower()
    if split == "train":
        loader = dm.train_dataloader()
    elif split == "val":
        loader = dm.val_dataloader()
    else:
        loader = dm.test_dataloader()
    return next(iter(loader))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--lead_steps", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--t_start", type=float, default=None)
    parser.add_argument("--t_end", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(args.seed) if args.seed is not None else int(cfg.get("train", {}).get("seed", 42))
    pl.seed_everything(seed, workers=True)

    diff_cfg = dict(cfg.get("diffusion", {}) or {})
    steps = int(args.steps) if args.steps is not None else int(diff_cfg.get("sample_steps", 8))
    t_start = float(args.t_start) if args.t_start is not None else float(diff_cfg.get("t_start", 0.0))
    t_end = float(args.t_end) if args.t_end is not None else float(diff_cfg.get("t_end", 1.0))

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    batch_size = 256
    dm = _build_dm(cfg, batch_size=batch_size, num_workers=int(args.num_workers))
    lit = _build_lit(cfg, args.ckpt).to(device)
    lit.eval()

    batch = _get_batch(dm, args.split)
    x_ctx = batch["x_ctx"].to(device)

    lead_steps = _resolve_lead_steps(cfg, args.lead_steps)
    lead_time = None
    if getattr(lit, "use_lead_time_conditioning", False):
        lt_steps_t = torch.full((x_ctx.size(0),), float(int(lead_steps)), device=device, dtype=torch.float32)
        lead_time = lit._normalize_lead_time(lt_steps_t)

    total_params = sum(p.numel() for p in lit.parameters())
    trainable_params = sum(p.numel() for p in lit.parameters() if p.requires_grad)

    with torch.no_grad():
        for _ in range(int(args.warmup)):
            _ = lit.predict_one(x_ctx, steps=steps, t_start=t_start, t_end=t_end, seed=None, lead_time=lead_time)
        _sync(device)

        start = time.perf_counter()
        for _ in range(int(args.iters)):
            _ = lit.predict_one(x_ctx, steps=steps, t_start=t_start, t_end=t_end, seed=None, lead_time=lead_time)
        _sync(device)
        elapsed = time.perf_counter() - start

    batch_size = int(x_ctx.size(0))
    denom = max(int(args.iters) * batch_size, 1)
    ms_per_sample = float(elapsed) * 1000.0 / float(denom)

    print(f"params_total: {total_params}")
    print(f"params_trainable: {trainable_params}")
    print(f"inference_ms_per_sample: {ms_per_sample:.6f}")
    print(f"batch_size: {batch_size}")
    print(f"device: {device.type}")
    print(f"steps: {steps}")
    print(f"t_start: {t_start}")
    print(f"t_end: {t_end}")
    print(f"lead_steps: {lead_steps}")


if __name__ == "__main__":
    main()
