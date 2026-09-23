from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .common import build_dataloaders, load_config, parse_data_config
from .mlp import SimpleMLP
from .tree import _flatten_per_pixel_features, _to_numpy


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parse_lead_steps(name: str) -> Optional[int]:
    m = re.search(r"lead(\d+)", name)
    if not m:
        return None
    return int(m.group(1))


def _linear_params(model: Any) -> Optional[int]:
    try:
        coef = getattr(model, "coef_", None)
        if coef is None:
            return None
        n = int(np.asarray(coef).size)
        intercept = getattr(model, "intercept_", None)
        if intercept is not None:
            n += int(np.asarray(intercept).size)
        return n
    except Exception:
        return None


def _mlp_from_state(state: Dict[str, torch.Tensor]) -> SimpleMLP:
    weights = [k for k in state.keys() if k.endswith(".weight")]
    if "net.0.weight" not in state:
        raise KeyError("state_dict missing net.0.weight")
    in_dim = int(state["net.0.weight"].shape[1])
    hidden = int(state["net.0.weight"].shape[0])
    last_w = max(weights, key=lambda k: int(k.split(".")[1]))
    out_dim = int(state[last_w].shape[0])
    model = SimpleMLP(in_dim=in_dim, out_dim=out_dim, hidden=hidden, dropout=0.0)
    model.load_state_dict(state, strict=True)
    return model


def _get_batch(split: str, data_cfg, batch_size: int, num_workers: int):
    data_cfg.eval_batch_size = int(batch_size)
    data_cfg.num_workers = int(num_workers)
    _, val_dl, test_dl = build_dataloaders(data_cfg, shuffle_train=False)
    split = str(split).lower()
    if split == "val":
        return next(iter(val_dl))
    return next(iter(test_dl))


def _time_inference(fn, *, iters: int, warmup: int, device: Optional[torch.device]) -> float:
    for _ in range(int(warmup)):
        fn()
    if device is not None:
        _sync(device)
    start = time.perf_counter()
    for _ in range(int(iters)):
        fn()
    if device is not None:
        _sync(device)
    return time.perf_counter() - start


def _predict_linear(model, batch) -> None:
    x_ctx = batch["x_ctx"].float()
    b, c, h, w = x_ctx.shape
    x = x_ctx.reshape(b, c * h * w).detach().cpu().numpy()
    _ = model.predict(x)


def _predict_tree(model, batch, *, add_xy: bool) -> None:
    x_ctx = batch["x_ctx"].float()
    x, _, _ = _flatten_per_pixel_features(x_ctx, add_xy=add_xy)
    if isinstance(model, (list, tuple)):
        preds = []
        for m in model:
            preds.append(_to_numpy(m.predict(x)).astype(np.float32).reshape(-1, 1))
        _ = np.concatenate(preds, axis=1)
    else:
        _ = _to_numpy(model.predict(x)).astype(np.float32)


def _predict_mlp(model: SimpleMLP, batch, device: torch.device) -> None:
    x_ctx = batch["x_ctx"].float()
    b, c, h, w = x_ctx.shape
    x = x_ctx.reshape(b, c * h * w).to(device)
    _ = model(x)


def _load_tree_add_xy(results_path: Path) -> bool:
    if not results_path.exists():
        return True
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
        return bool(data.get("add_xy", True))
    except Exception:
        return True


def _bench_linear(models_dir: Path, batch, iters: int, warmup: int) -> List[Dict[str, Any]]:
    out = []
    import joblib

    for fp in sorted(models_dir.glob("*.joblib")):
        name = fp.stem
        lead_steps = _parse_lead_steps(name)
        kind = name.split("_")[0]
        item: Dict[str, Any] = {
            "baseline": "linear",
            "model": kind,
            "lead_steps": lead_steps,
            "file": str(fp),
            "params": None,
            "inference_ms_per_sample": None,
        }
        try:
            model = joblib.load(fp)
            item["params"] = _linear_params(model)
            elapsed = _time_inference(lambda: _predict_linear(model, batch), iters=iters, warmup=warmup, device=None)
            bsz = int(batch["x_ctx"].shape[0])
            item["inference_ms_per_sample"] = float(elapsed) * 1000.0 / float(max(iters * bsz, 1))
        except Exception as e:
            item["error"] = str(e)
        out.append(item)
    return out


def _bench_tree(models_dir: Path, batch, iters: int, warmup: int, model_name: str) -> List[Dict[str, Any]]:
    out = []
    import joblib

    add_xy = _load_tree_add_xy(models_dir / f"results_{model_name}.json")
    for fp in sorted(models_dir.glob(f"{model_name}_lead*.joblib")):
        name = fp.stem
        lead_steps = _parse_lead_steps(name)
        item: Dict[str, Any] = {
            "baseline": "tree",
            "model": model_name,
            "lead_steps": lead_steps,
            "file": str(fp),
            "params": None,
            "inference_ms_per_sample": None,
            "add_xy": add_xy,
        }
        try:
            model = joblib.load(fp)
            elapsed = _time_inference(
                lambda: _predict_tree(model, batch, add_xy=add_xy), iters=iters, warmup=warmup, device=None
            )
            bsz = int(batch["x_ctx"].shape[0])
            item["inference_ms_per_sample"] = float(elapsed) * 1000.0 / float(max(iters * bsz, 1))
        except Exception as e:
            item["error"] = str(e)
        out.append(item)
    return out


def _bench_mlp(models_dir: Path, batch, iters: int, warmup: int, device: torch.device) -> List[Dict[str, Any]]:
    out = []
    for fp in sorted(models_dir.glob("*.pt")):
        name = fp.stem
        lead_steps = _parse_lead_steps(name)
        item: Dict[str, Any] = {
            "baseline": "mlp_3layer",
            "model": "mlp",
            "lead_steps": lead_steps,
            "file": str(fp),
            "params": None,
            "inference_ms_per_sample": None,
            "device": str(device),
        }
        try:
            state = torch.load(fp, map_location="cpu")
            model = _mlp_from_state(state).to(device)
            model.eval()
            item["params"] = int(sum(p.numel() for p in model.parameters()))
            with torch.no_grad():
                elapsed = _time_inference(
                    lambda: _predict_mlp(model, batch, device), iters=iters, warmup=warmup, device=device
                )
            bsz = int(batch["x_ctx"].shape[0])
            item["inference_ms_per_sample"] = float(elapsed) * 1000.0 / float(max(iters * bsz, 1))
        except Exception as e:
            item["error"] = str(e)
        out.append(item)
    return out


def main():
    p = argparse.ArgumentParser(description="Benchmark baselines: params and inference ms/sample")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--models_root", type=str, default="outputs/baselines_models")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--only", type=str, default="")
    args = p.parse_args()

    cfg = load_config(args.config)
    data_cfg = parse_data_config(cfg)
    batch = _get_batch(args.split, data_cfg, args.batch_size, args.num_workers)

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    root = Path(args.models_root)
    only = {x.strip().lower() for x in str(args.only).split(",") if x.strip()}

    results: List[Dict[str, Any]] = []

    if (not only) or ("linear" in only):
        linear_dir = root / "linear"
        if linear_dir.exists():
            results.extend(_bench_linear(linear_dir, batch, args.iters, args.warmup))

    if (not only) or ("mlp" in only):
        mlp_dir = root / "mlp"
        if mlp_dir.exists():
            results.extend(_bench_mlp(mlp_dir, batch, args.iters, args.warmup, device))

    if (not only) or ("tree" in only) or ("xgb" in only) or ("rf" in only) or ("rf_gpu" in only):
        tree_dir = root / "tree"
        if tree_dir.exists():
            if (not only) or ("xgb" in only) or ("tree" in only):
                results.extend(_bench_tree(tree_dir, batch, args.iters, args.warmup, "xgb"))
            if (not only) or ("rf_gpu" in only) or ("tree" in only):
                results.extend(_bench_tree(tree_dir, batch, args.iters, args.warmup, "rf_gpu"))
            if (not only) or ("rf" in only) or ("tree" in only):
                results.extend(_bench_tree(tree_dir, batch, args.iters, args.warmup, "rf"))

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
