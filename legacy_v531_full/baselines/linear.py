from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.linear_model import MultiTaskLasso, Ridge
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .common import (
    as_numpy,
    build_dataloaders,
    lead_steps_to_tag,
    load_config,
    maybe_limit_rows,
    parse_data_config,
    save_json,
)
from .metrics import BatchTensors, compute_batch_metrics, compute_batch_metrics_per_channel, default_clim, extract_lat


def _collect_flattened_xy(
    dl,
    *,
    lead_index: int,
    max_samples: Optional[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Collect X=[N,F] and Y=[N,O] by flattening full 8x8 fields."""

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    n = 0
    it = iter(dl)
    try:
        first = next(it)
    except StopIteration:
        return np.empty((0, 0)), np.empty((0, 0))

    x_ctx = first["x_ctx"].float()  # [B, ctx_C, H, W]
    B, ctx_C, H, W = x_ctx.shape
    x = x_ctx.reshape(B, ctx_C * H * W)

    if "y" in first:
        y0 = first["y"][:, lead_index].float()  # [B, C, H, W]
    else:
        y0 = first["x0"].float()
    y = y0.reshape(B, -1)

    xs.append(as_numpy(x))
    ys.append(as_numpy(y))
    n += B

    try:
        total_batches = len(dl)
    except Exception:
        total_batches = None
    if max_samples is not None and max_samples > 0 and B > 0:
        needed = int(math.ceil(float(max_samples) / float(B)))
        total_batches = min(total_batches, needed) if total_batches is not None else needed

    with Progress(
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
    ) as progress:
        task = progress.add_task("collect train samples", total=total_batches)
        progress.advance(task)
        for batch in it:
            x_ctx = batch["x_ctx"].float()  # [B, ctx_C, H, W]
            B, ctx_C, H, W = x_ctx.shape
            x = x_ctx.reshape(B, ctx_C * H * W)

            if "y" in batch:
                y0 = batch["y"][:, lead_index].float()  # [B, C, H, W]
            else:
                y0 = batch["x0"].float()
            y = y0.reshape(B, -1)

            xs.append(as_numpy(x))
            ys.append(as_numpy(y))
            n += B
            progress.advance(task)
            if max_samples is not None and max_samples > 0 and n >= max_samples:
                break

    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)
    X = maybe_limit_rows(X, max_samples)
    Y = maybe_limit_rows(Y, max_samples)
    return X, Y


@torch.no_grad()
def _evaluate_model(
    model,
    dl,
    *,
    lead_index: int,
    lead_steps: int,
    time_step_hours: float,
    var_names: List[str],
    progress_desc: str = "eval",
) -> Dict:
    metrics_sum = {"rmse": 0.0, "mae": 0.0, "acc": 0.0}
    metrics_count = {"rmse": 0, "mae": 0, "acc": 0}
    rmse_ch_sum = None
    mae_ch_sum = None
    acc_ch_sum = None

    try:
        total = len(dl)
    except Exception:
        total = None
    with Progress(
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
    ) as progress:
        task = progress.add_task(progress_desc, total=total)
        for batch in dl:
            x_ctx = batch["x_ctx"].float()
            B, ctx_C, H, W = x_ctx.shape
            x = as_numpy(x_ctx.reshape(B, ctx_C * H * W))

            y_pred = model.predict(x).astype(np.float32)
            pred_norm = torch.from_numpy(y_pred).reshape(B, -1, H, W)

            if "y" in batch:
                target_norm = batch["y"][:, lead_index].float().reshape(B, -1, H, W)
            else:
                target_norm = batch["x0"].float().reshape(B, -1, H, W)

            mean = batch["norm"]["mean"].float()
            std = batch["norm"]["std"].float()

            bt = BatchTensors(
                pred_norm=pred_norm,
                target_norm=target_norm,
                mean=mean,
                std=std,
                clim_denorm=default_clim(batch),
                lat=extract_lat(batch),
            )
            m = compute_batch_metrics(bt)
            for k, v in m.items():
                if torch.isfinite(v):
                    metrics_sum[k] += float(v)
                    metrics_count[k] += 1

            mc = compute_batch_metrics_per_channel(bt)
            if rmse_ch_sum is None:
                rmse_ch_sum = mc["rmse_ch"].detach().cpu()
                mae_ch_sum = mc["mae_ch"].detach().cpu()
                if "acc_ch" in mc:
                    acc_ch_sum = mc["acc_ch"].detach().cpu()
            else:
                rmse_ch_sum += mc["rmse_ch"].detach().cpu()
                mae_ch_sum += mc["mae_ch"].detach().cpu()
                if "acc_ch" in mc and acc_ch_sum is not None:
                    acc_ch_sum += mc["acc_ch"].detach().cpu()
            progress.advance(task)

    tag = lead_steps_to_tag(int(lead_steps), time_step_hours=time_step_hours)
    out: Dict = {
        "lead_steps": int(lead_steps),
        "lead_tag": tag,
        "RMSE": metrics_sum["rmse"] / max(metrics_count["rmse"], 1),
        "MAE": metrics_sum["mae"] / max(metrics_count["mae"], 1),
    }

    if metrics_count["acc"] > 0:
        out["ACC"] = metrics_sum["acc"] / float(metrics_count["acc"])

    if rmse_ch_sum is not None:
        denom = len(dl)
        rmse_ch = (rmse_ch_sum / max(denom, 1)).numpy().tolist()
        mae_ch = (mae_ch_sum / max(denom, 1)).numpy().tolist()
        out["RMSE_per_var"] = {n: rmse_ch[i] for i, n in enumerate(var_names)}
        out["MAE_per_var"] = {n: mae_ch[i] for i, n in enumerate(var_names)}
        if acc_ch_sum is not None:
            acc_ch = (acc_ch_sum / max(denom, 1)).numpy().tolist()
            out["ACC_per_var"] = {n: acc_ch[i] for i, n in enumerate(var_names)}

    return out


def _count_params(model) -> Optional[int]:
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


def _bench_inference(model, batch, *, iters: int, warmup: int) -> Optional[float]:
    try:
        x_ctx = batch["x_ctx"].float()
        B, ctx_C, H, W = x_ctx.shape
        x = x_ctx.reshape(B, ctx_C * H * W).detach().cpu().numpy()
    except Exception:
        return None

    for _ in range(int(warmup)):
        _ = model.predict(x)
    start = time.perf_counter()
    for _ in range(int(iters)):
        _ = model.predict(x)
    elapsed = time.perf_counter() - start
    denom = max(int(iters) * int(B), 1)
    return float(elapsed) * 1000.0 / float(denom)


def run_linear_baseline(
    *,
    config_path: str | Path,
    kind: str = "ridge",
    alpha: float = 1.0,
    max_train_samples: Optional[int] = None,
    out_dir: str | Path = "outputs/baselines/linear",
    load_dir: str | Path | None = None,
    skip_fit: bool = False,
    bench_iters: int = 50,
    bench_warmup: int = 10,
    bench_split: str = "test",
) -> Dict:
    """Train and evaluate a linear regression baseline.

    Notes:
        - Input: flatten of all ctx frames/vars/pixels.
        - Output: flatten of all target vars/pixels.
    """

    cfg = load_config(config_path)
    data_cfg = parse_data_config(cfg)
    train_dl, val_dl, test_dl = build_dataloaders(data_cfg, shuffle_train=False)

    lead_times = data_cfg.lead_times or [data_cfg.delta_t]

    results: Dict = {
        "baseline": f"linear_{kind}",
        "alpha": float(alpha),
        "lead_times": lead_times,
        "max_train_samples": max_train_samples,
        "bench": {},
        "val": {},
        "test": {},
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    load_dir = Path(load_dir) if load_dir is not None else None
    if load_dir is not None:
        load_dir.mkdir(parents=True, exist_ok=True)

    import joblib

    bench_dl = val_dl if str(bench_split).lower() == "val" else test_dl
    try:
        bench_batch = next(iter(bench_dl))
    except Exception:
        bench_batch = None

    for li, lt in enumerate(lead_times):
        tag = lead_steps_to_tag(int(lt), time_step_hours=data_cfg.time_step_hours)
        model_path = out_dir / f"{kind}_lead{lt}_{tag}.joblib"
        if load_dir is not None:
            model_path = load_dir / f"{kind}_lead{lt}_{tag}.joblib"

        model = None
        if skip_fit and model_path.exists():
            model = joblib.load(model_path)

        if model is None:
            # Collect training data
            X_train, Y_train = _collect_flattened_xy(train_dl, lead_index=li, max_samples=max_train_samples)

            if kind.lower() in ("ridge", "linreg", "linear"):
                model = Ridge(alpha=float(alpha), random_state=0)
            elif kind.lower() in ("lasso", "multitasklasso"):
                model = MultiTaskLasso(alpha=float(alpha), random_state=0, max_iter=10_000)
            else:
                raise ValueError(f"Unknown linear kind: {kind}")

            with Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                TimeElapsedColumn(),
                transient=False,
            ) as progress:
                task = progress.add_task(f"fit {kind} lead {tag}", total=1)
                model.fit(X_train, Y_train)
                progress.advance(task)

            model_path = out_dir / f"{kind}_lead{lt}_{tag}.joblib"
            joblib.dump(model, model_path)

        params = _count_params(model)
        ms_per_sample = None
        if bench_batch is not None:
            ms_per_sample = _bench_inference(model, bench_batch, iters=bench_iters, warmup=bench_warmup)
        results["bench"][tag] = {
            "params": params,
            "inference_ms_per_sample": ms_per_sample,
            "batch_size": int(bench_batch["x_ctx"].shape[0]) if bench_batch is not None else None,
            "split": str(bench_split),
        }
        print(
            f"[BENCH] linear {kind} lead={tag} params={params} ms/sample={ms_per_sample} batch={results['bench'][tag]['batch_size']} split={bench_split}"
        )

        # Evaluate
        res_val = _evaluate_model(
            model,
            val_dl,
            lead_index=li,
            lead_steps=int(lt),
            time_step_hours=data_cfg.time_step_hours,
            var_names=data_cfg.dynamic_vars,
            progress_desc=f"val lead {tag}",
        )
        res_test = _evaluate_model(
            model,
            test_dl,
            lead_index=li,
            lead_steps=int(lt),
            time_step_hours=data_cfg.time_step_hours,
            var_names=data_cfg.dynamic_vars,
            progress_desc=f"test lead {tag}",
        )

        results["val"][tag] = res_val
        results["test"][tag] = res_test

    save_json(results, out_dir / "results.json")
    return results


def main():
    p = argparse.ArgumentParser(description="UrbanPiDiT baseline: Linear Regression")
    p.add_argument("--config", required=True, type=str, help="yaml config path")
    p.add_argument("--kind", default="ridge", choices=["ridge", "lasso"], help="linear reg type")
    p.add_argument("--alpha", type=float, default=1.0, help="regularization strength")
    p.add_argument("--max_train_samples", type=int, default=0, help="optional cap for training samples")
    p.add_argument("--out_dir", type=str, default="outputs/baselines/linear")
    p.add_argument("--load_dir", type=str, default=None)
    p.add_argument("--skip_fit", action="store_true")
    p.add_argument("--bench_iters", type=int, default=50)
    p.add_argument("--bench_warmup", type=int, default=10)
    p.add_argument("--bench_split", type=str, default="test")
    args = p.parse_args()

    max_samples = args.max_train_samples if args.max_train_samples and args.max_train_samples > 0 else None
    results = run_linear_baseline(
        config_path=args.config,
        kind=args.kind,
        alpha=args.alpha,
        max_train_samples=max_samples,
        out_dir=args.out_dir,
        load_dir=args.load_dir,
        skip_fit=bool(args.skip_fit),
        bench_iters=int(args.bench_iters),
        bench_warmup=int(args.bench_warmup),
        bench_split=str(args.bench_split),
    )
    print(json_dumps(results))


def json_dumps(obj: Dict) -> str:
    import json

    return json.dumps(obj, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
