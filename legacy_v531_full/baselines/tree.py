from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
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


def _flatten_per_pixel_features(
    x_ctx: torch.Tensor,
    *,
    add_xy: bool,
) -> Tuple[np.ndarray, int, int]:
    """Convert x_ctx=[B,ctx_C,H,W] to X=[B*H*W, ctx_C(+2)]."""

    B, ctx_C, H, W = x_ctx.shape
    # [B,ctx_C,H,W] -> [B,H,W,ctx_C] -> [B*H*W, ctx_C]
    X = x_ctx.permute(0, 2, 3, 1).reshape(B * H * W, ctx_C)
    if add_xy:
        ys, xs = torch.meshgrid(
            torch.linspace(0, 1, H, device=x_ctx.device),
            torch.linspace(0, 1, W, device=x_ctx.device),
            indexing="ij",
        )
        xy = torch.stack([ys, xs], dim=-1).reshape(H * W, 2)
        xy = xy.unsqueeze(0).expand(B, -1, -1).reshape(B * H * W, 2)
        X = torch.cat([X, xy], dim=1)
    return as_numpy(X.float()), H, W


def _flatten_per_pixel_targets(y: torch.Tensor) -> np.ndarray:
    """Convert y=[B,C,H,W] to Y=[B*H*W, C]."""

    B, C, H, W = y.shape
    Y = y.permute(0, 2, 3, 1).reshape(B * H * W, C)
    return as_numpy(Y.float())


def _to_numpy(x) -> np.ndarray:
    try:
        import cupy as cp

        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception:
        pass
    try:
        return x.to_numpy()
    except Exception:
        return np.asarray(x)


def _collect_tabular_xy(
    dl,
    *,
    lead_index: int,
    add_xy: bool,
    max_rows: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    H = W = -1
    n = 0
    it = iter(dl)
    try:
        first = next(it)
    except StopIteration:
        return np.empty((0, 0)), np.empty((0, 0)), H, W

    x_ctx = first["x_ctx"].float()
    X, H, W = _flatten_per_pixel_features(x_ctx, add_xy=add_xy)
    if "y" in first:
        y0 = first["y"][:, lead_index].float()
    else:
        y0 = first["x0"].float()
    Y = _flatten_per_pixel_targets(y0)
    xs.append(X)
    ys.append(Y)
    n += X.shape[0]

    try:
        total_batches = len(dl)
    except Exception:
        total_batches = None
    if max_rows is not None and max_rows > 0 and X.shape[0] > 0:
        needed = int(math.ceil(float(max_rows) / float(X.shape[0])))
        total_batches = min(total_batches, needed) if total_batches is not None else needed

    with Progress(
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
    ) as progress:
        task = progress.add_task("collect train rows", total=total_batches)
        progress.advance(task)
        for batch in it:
            x_ctx = batch["x_ctx"].float()
            X, H, W = _flatten_per_pixel_features(x_ctx, add_xy=add_xy)

            if "y" in batch:
                y0 = batch["y"][:, lead_index].float()
            else:
                y0 = batch["x0"].float()
            Y = _flatten_per_pixel_targets(y0)

            xs.append(X)
            ys.append(Y)
            n += X.shape[0]
            progress.advance(task)
            if max_rows is not None and max_rows > 0 and n >= max_rows:
                break

    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)
    X = maybe_limit_rows(X, max_rows)
    Y = maybe_limit_rows(Y, max_rows)
    return X, Y, H, W


@torch.no_grad()
def _evaluate_tabular_model(
    model,
    dl,
    *,
    lead_index: int,
    lead_steps: int,
    time_step_hours: float,
    var_names: List[str],
    add_xy: bool,
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
            X, _, _ = _flatten_per_pixel_features(x_ctx, add_xy=add_xy)
            if isinstance(model, (list, tuple)):
                preds = []
                for m in model:
                    preds.append(_to_numpy(m.predict(X)).astype(np.float32).reshape(-1, 1))
                y_pred = np.concatenate(preds, axis=1)
            else:
                y_pred = _to_numpy(model.predict(X)).astype(np.float32)
            pred_norm = torch.from_numpy(y_pred).reshape(B, H, W, -1).permute(0, 3, 1, 2)

            if "y" in batch:
                target_norm = batch["y"][:, lead_index].float()
            else:
                target_norm = batch["x0"].float()

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


def run_tree_baseline(
    *,
    config_path: str | Path,
    model: str = "xgb",
    max_train_rows: Optional[int] = None,
    add_xy: bool = True,
    out_dir: str | Path = "outputs/baselines/tree",
    load_dir: str | Path | None = None,
    skip_fit: bool = False,
    bench_iters: int = 50,
    bench_warmup: int = 10,
    bench_split: str = "test",
    # RF
    rf_n_estimators: int = 400,
    rf_max_depth: Optional[int] = None,
    # XGB
    xgb_n_estimators: int = 600,
    xgb_max_depth: int = 8,
    xgb_learning_rate: float = 0.05,
    xgb_subsample: float = 0.9,
    xgb_colsample_bytree: float = 0.9,
) -> Dict:
    """Train and evaluate a tabular tree baseline (per-pixel).

    This treats each grid cell as a separate tabular row:
      - Features: ctx channels at that pixel (k*vars + static), optional (y,x) coords
      - Target: future vars at that pixel

    In the 8x8 setting, this effectively multiplies sample size by 64,
    which often helps trees.
    """

    cfg = load_config(config_path)
    data_cfg = parse_data_config(cfg)
    train_dl, val_dl, test_dl = build_dataloaders(data_cfg, shuffle_train=False)

    lead_times = data_cfg.lead_times or [data_cfg.delta_t]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    load_dir = Path(load_dir) if load_dir is not None else None
    if load_dir is not None:
        load_dir.mkdir(parents=True, exist_ok=True)

    results: Dict = {
        "baseline": f"tree_{model}",
        "lead_times": lead_times,
        "max_train_rows": max_train_rows,
        "add_xy": bool(add_xy),
        "bench": {},
        "val": {},
        "test": {},
    }

    # Lazy import to avoid hard dependency
    XGBRegressor = None
    if model.lower() in ("xgb", "xgboost"):
        from xgboost import XGBRegressor as _XGBRegressor

        XGBRegressor = _XGBRegressor

    CumlRandomForestRegressor = None
    if model.lower() in ("rf_gpu", "cuml", "cuml_rf"):
        try:
            from cuml.ensemble import RandomForestRegressor as _CumlRF

            CumlRandomForestRegressor = _CumlRF
        except Exception as e:
            raise ImportError("cuml is not installed or not available") from e

    import joblib
    bench_dl = val_dl if str(bench_split).lower() == "val" else test_dl
    try:
        bench_batch = next(iter(bench_dl))
    except Exception:
        bench_batch = None

    for li, lt in enumerate(lead_times):
        tag = lead_steps_to_tag(int(lt), time_step_hours=data_cfg.time_step_hours)
        model_path = out_dir / f"{model}_lead{lt}_{tag}.joblib"
        if load_dir is not None:
            model_path = load_dir / f"{model}_lead{lt}_{tag}.joblib"

        reg = None
        if skip_fit and model_path.exists():
            reg = joblib.load(model_path)

        if reg is None:
            X_train, Y_train, H, W = _collect_tabular_xy(
                train_dl,
                lead_index=li,
                add_xy=add_xy,
                max_rows=max_train_rows,
            )

            if model.lower() in ("rf", "random_forest", "randomforest"):
                estimator = RandomForestRegressor(
                    n_estimators=1,
                    max_depth=None if rf_max_depth is None else int(rf_max_depth),
                    random_state=0,
                    n_jobs=-1,
                    warm_start=True,
                )
                reg = estimator
            elif model.lower() in ("rf_gpu", "cuml", "cuml_rf"):
                reg = []
                for _ in range(Y_train.shape[1]):
                    params = {
                        "n_estimators": int(rf_n_estimators),
                        "random_state": 0,
                    }
                    if rf_max_depth is not None and int(rf_max_depth) > 0:
                        params["max_depth"] = int(rf_max_depth)
                    reg.append(CumlRandomForestRegressor(**params))
            elif model.lower() in ("xgb", "xgboost"):
                if XGBRegressor is None:
                    raise ImportError("xgboost is not installed")
                base = XGBRegressor(
                    n_estimators=int(xgb_n_estimators),
                    max_depth=int(xgb_max_depth),
                    learning_rate=float(xgb_learning_rate),
                    subsample=float(xgb_subsample),
                    colsample_bytree=float(xgb_colsample_bytree),
                    objective="reg:squarederror",
                    tree_method="hist",
                    random_state=0,
                    n_jobs=-1,
                )
                reg = MultiOutputRegressor(base, n_jobs=-1)
            else:
                raise ValueError(f"Unknown tree model: {model}")

            if model.lower() in ("rf", "random_forest", "randomforest"):
                total_trees = int(rf_n_estimators)
                step = 1
                with Progress(
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    transient=False,
                ) as progress:
                    task = progress.add_task(f"fit rf lead {tag}", total=total_trees)
                    current = 0
                    while current < total_trees:
                        next_n = min(current + step, total_trees)
                        reg.n_estimators = next_n
                        reg.fit(X_train, Y_train)
                        progress.advance(task, next_n - current)
                        current = next_n
            elif model.lower() in ("rf_gpu", "cuml", "cuml_rf"):
                with Progress(
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    transient=False,
                ) as progress:
                    total = len(reg)
                    task = progress.add_task(f"fit rf_gpu lead {tag}", total=total)
                    for ci, est in enumerate(reg):
                        est.fit(X_train, Y_train[:, ci])
                        progress.advance(task)
            else:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("{task.description}"),
                    TimeElapsedColumn(),
                    transient=False,
                ) as progress:
                    task = progress.add_task(f"fit {model} lead {tag}", total=1)
                    reg.fit(X_train, Y_train)
                    progress.advance(task)

            model_path = out_dir / f"{model}_lead{lt}_{tag}.joblib"
            joblib.dump(reg, model_path)

        ms_per_sample = None
        if bench_batch is not None:
            x_ctx = bench_batch["x_ctx"].float()
            B, _, _, _ = x_ctx.shape
            X, _, _ = _flatten_per_pixel_features(x_ctx, add_xy=add_xy)
            for _ in range(int(bench_warmup)):
                if isinstance(reg, (list, tuple)):
                    preds = []
                    for m in reg:
                        preds.append(_to_numpy(m.predict(X)).astype(np.float32).reshape(-1, 1))
                    _ = np.concatenate(preds, axis=1)
                else:
                    _ = _to_numpy(reg.predict(X)).astype(np.float32)
            start = time.perf_counter()
            for _ in range(int(bench_iters)):
                if isinstance(reg, (list, tuple)):
                    preds = []
                    for m in reg:
                        preds.append(_to_numpy(m.predict(X)).astype(np.float32).reshape(-1, 1))
                    _ = np.concatenate(preds, axis=1)
                else:
                    _ = _to_numpy(reg.predict(X)).astype(np.float32)
            elapsed = time.perf_counter() - start
            denom = max(int(bench_iters) * int(B), 1)
            ms_per_sample = float(elapsed) * 1000.0 / float(denom)

        results["bench"][tag] = {
            "params": None,
            "inference_ms_per_sample": ms_per_sample,
            "batch_size": int(bench_batch["x_ctx"].shape[0]) if bench_batch is not None else None,
            "split": str(bench_split),
        }
        print(
            f"[BENCH] tree {model} lead={tag} params=None ms/sample={ms_per_sample} batch={results['bench'][tag]['batch_size']} split={bench_split}"
        )

        # Evaluate
        res_val = _evaluate_tabular_model(
            reg,
            val_dl,
            lead_index=li,
            lead_steps=int(lt),
            time_step_hours=data_cfg.time_step_hours,
            var_names=data_cfg.dynamic_vars,
            add_xy=add_xy,
            progress_desc=f"val lead {tag}",
        )
        res_test = _evaluate_tabular_model(
            reg,
            test_dl,
            lead_index=li,
            lead_steps=int(lt),
            time_step_hours=data_cfg.time_step_hours,
            var_names=data_cfg.dynamic_vars,
            add_xy=add_xy,
            progress_desc=f"test lead {tag}",
        )
        results["val"][tag] = res_val
        results["test"][tag] = res_test

    save_json(results, out_dir / "results.json")
    return results


def main():
    p = argparse.ArgumentParser(description="UrbanPiDiT baseline: Tree models (tabular per-pixel)")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--model", default="xgb", choices=["xgb", "rf", "rf_gpu"])
    p.add_argument("--max_train_rows", type=int, default=0)
    p.add_argument("--no_xy", action="store_true", help="disable coordinate features")
    p.add_argument("--out_dir", type=str, default="outputs/baselines/tree")
    p.add_argument("--load_dir", type=str, default=None)
    p.add_argument("--skip_fit", action="store_true")
    p.add_argument("--bench_iters", type=int, default=50)
    p.add_argument("--bench_warmup", type=int, default=10)
    p.add_argument("--bench_split", type=str, default="test")

    # RF
    p.add_argument("--rf_n_estimators", type=int, default=400)
    p.add_argument("--rf_max_depth", type=int, default=0)

    # XGB
    p.add_argument("--xgb_n_estimators", type=int, default=600)
    p.add_argument("--xgb_max_depth", type=int, default=8)
    p.add_argument("--xgb_learning_rate", type=float, default=0.05)
    p.add_argument("--xgb_subsample", type=float, default=0.9)
    p.add_argument("--xgb_colsample_bytree", type=float, default=0.9)

    args = p.parse_args()
    max_rows = args.max_train_rows if args.max_train_rows and args.max_train_rows > 0 else None
    rf_md = args.rf_max_depth if args.rf_max_depth and args.rf_max_depth > 0 else None

    results = run_tree_baseline(
        config_path=args.config,
        model=args.model,
        max_train_rows=max_rows,
        add_xy=(not bool(args.no_xy)),
        out_dir=args.out_dir,
        load_dir=args.load_dir,
        skip_fit=bool(args.skip_fit),
        bench_iters=int(args.bench_iters),
        bench_warmup=int(args.bench_warmup),
        bench_split=str(args.bench_split),
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=rf_md,
        xgb_n_estimators=args.xgb_n_estimators,
        xgb_max_depth=args.xgb_max_depth,
        xgb_learning_rate=args.xgb_learning_rate,
        xgb_subsample=args.xgb_subsample,
        xgb_colsample_bytree=args.xgb_colsample_bytree,
    )
    import json

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
