from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TimeElapsedColumn, TimeRemainingColumn

from .common import (
    build_dataloaders,
    lead_steps_to_tag,
    load_config,
    parse_data_config,
    save_json,
)
from .metrics import BatchTensors, compute_batch_metrics, compute_batch_metrics_per_channel, default_clim, extract_lat


class SimpleMLP(nn.Module):
    """A minimal 3-layer MLP: Linear->ReLU->Linear->ReLU->Linear."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainCfg:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden: int = 1024
    dropout: float = 0.0
    grad_clip: float = 1.0
    early_stop_patience: int = 10


def _batch_to_xy(batch: Dict, *, lead_index: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    x_ctx = batch["x_ctx"].float()  # [B,ctx_C,H,W]
    B, ctx_C, H, W = x_ctx.shape
    X = x_ctx.reshape(B, ctx_C * H * W)

    if "y" in batch:
        y0 = batch["y"][:, lead_index].float()  # [B,C,H,W]
    else:
        y0 = batch["x0"].float()  # [B,C,H,W]
    Y = y0.reshape(B, -1)
    return X, Y, H, W


@torch.no_grad()
def _eval_mlp(
    model: nn.Module,
    dl: DataLoader,
    *,
    lead_index: int,
    lead_steps: int,
    time_step_hours: float,
    var_names: List[str],
    device: torch.device,
    progress_desc: str = "eval",
) -> Dict:
    model.eval()
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
        transient=True,
    ) as progress:
        task = progress.add_task(progress_desc, total=total)
        for batch in dl:
            X, Y, H, W = _batch_to_xy(batch, lead_index=lead_index)
            X = X.to(device)
            pred = model(X).reshape(-1, len(var_names), H, W).cpu()
            target_norm = Y.reshape(-1, len(var_names), H, W)

            mean = batch["norm"]["mean"].float()
            std = batch["norm"]["std"].float()

            bt = BatchTensors(
                pred_norm=pred,
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


def run_mlp_baseline(
    *,
    config_path: str | Path,
    train_cfg: TrainCfg | None = None,
    out_dir: str | Path = "outputs/baselines/mlp",
    device: str | None = None,
    load_dir: str | Path | None = None,
    skip_fit: bool = False,
    bench_iters: int = 50,
    bench_warmup: int = 10,
    bench_split: str = "test",
) -> Dict:
    """Train and evaluate a simple 3-layer MLP baseline."""

    cfg = load_config(config_path)
    data_cfg = parse_data_config(cfg)
    train_dl, val_dl, test_dl = build_dataloaders(data_cfg, shuffle_train=True)

    lead_times = data_cfg.lead_times or [data_cfg.delta_t]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    load_dir = Path(load_dir) if load_dir is not None else None
    if load_dir is not None:
        load_dir.mkdir(parents=True, exist_ok=True)

    if train_cfg is None:
        train_cfg = TrainCfg()

    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    results: Dict = {
        "baseline": "mlp_3layer",
        "lead_times": lead_times,
        "train_cfg": train_cfg.__dict__,
        "device": str(dev),
        "bench": {},
        "val": {},
        "test": {},
    }

    # Infer dimensions from one batch
    first = next(iter(train_dl))
    X0, Y0, H, W = _batch_to_xy(first, lead_index=0)
    in_dim = int(X0.shape[1])
    out_dim = int(Y0.shape[1])
    C = len(data_cfg.dynamic_vars)
    assert out_dim == C * H * W, f"Unexpected out_dim={out_dim} (C={C}, H={H}, W={W})"

    loss_fn = nn.MSELoss()
    bench_dl = val_dl if str(bench_split).lower() == "val" else test_dl
    try:
        bench_batch = next(iter(bench_dl))
    except Exception:
        bench_batch = None

    for li, lt in enumerate(lead_times):
        tag = lead_steps_to_tag(int(lt), time_step_hours=data_cfg.time_step_hours)
        model_path = out_dir / f"mlp_lead{lt}_{tag}.pt"
        if load_dir is not None:
            model_path = load_dir / f"mlp_lead{lt}_{tag}.pt"

        model = SimpleMLP(in_dim=in_dim, out_dim=out_dim, hidden=train_cfg.hidden, dropout=train_cfg.dropout).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

        if skip_fit and model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=dev))
        else:
            best_val = float("inf")
            best_state = None
            bad_epochs = 0

            with Progress(
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=False,
            ) as progress:
                epoch_task = progress.add_task(f"lead {tag} epochs", total=int(train_cfg.epochs))
                try:
                    train_total = len(train_dl)
                except Exception:
                    train_total = None
                batch_task = progress.add_task("batches", total=train_total)
                for epoch in range(int(train_cfg.epochs)):
                    model.train()
                    progress.reset(batch_task, total=train_total)
                    for batch in train_dl:
                        X, Y, _, _ = _batch_to_xy(batch, lead_index=li)
                        X = X.to(dev)
                        Y = Y.to(dev)

                        pred = model(X)
                        loss = loss_fn(pred, Y)

                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        if train_cfg.grad_clip and train_cfg.grad_clip > 0:
                            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(train_cfg.grad_clip))
                        opt.step()
                        progress.advance(batch_task)

                    val_res = _eval_mlp(
                        model,
                        val_dl,
                        lead_index=li,
                        lead_steps=int(lt),
                        time_step_hours=data_cfg.time_step_hours,
                        var_names=data_cfg.dynamic_vars,
                        device=dev,
                        progress_desc=f"val lead {tag} epoch {epoch + 1}",
                    )
                    val_rmse = float(val_res["RMSE"])
                    if val_rmse < best_val:
                        best_val = val_rmse
                        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                        bad_epochs = 0
                    else:
                        bad_epochs += 1

                    progress.advance(epoch_task)

                    if train_cfg.early_stop_patience and bad_epochs >= int(train_cfg.early_stop_patience):
                        progress.update(epoch_task, total=epoch + 1, completed=epoch + 1)
                        break

            if best_state is not None:
                model.load_state_dict(best_state)

            model_path = out_dir / f"mlp_lead{lt}_{tag}.pt"
            torch.save(model.state_dict(), model_path)

        params = int(sum(p.numel() for p in model.parameters()))
        ms_per_sample = None
        if bench_batch is not None:
            x_ctx = bench_batch["x_ctx"].float()
            B, ctx_C, H, W = x_ctx.shape
            x = x_ctx.reshape(B, ctx_C * H * W).to(dev)
            with torch.no_grad():
                for _ in range(int(bench_warmup)):
                    _ = model(x)
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                start = time.perf_counter()
                for _ in range(int(bench_iters)):
                    _ = model(x)
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                elapsed = time.perf_counter() - start
            denom = max(int(bench_iters) * int(B), 1)
            ms_per_sample = float(elapsed) * 1000.0 / float(denom)

        results["bench"][tag] = {
            "params": params,
            "inference_ms_per_sample": ms_per_sample,
            "batch_size": int(bench_batch["x_ctx"].shape[0]) if bench_batch is not None else None,
            "split": str(bench_split),
        }
        print(
            f"[BENCH] mlp lead={tag} params={params} ms/sample={ms_per_sample} batch={results['bench'][tag]['batch_size']} split={bench_split}"
        )

        # Final eval
        res_val = _eval_mlp(
            model,
            val_dl,
            lead_index=li,
            lead_steps=int(lt),
            time_step_hours=data_cfg.time_step_hours,
            var_names=data_cfg.dynamic_vars,
            device=dev,
            progress_desc=f"final val lead {tag}",
        )
        res_test = _eval_mlp(
            model,
            test_dl,
            lead_index=li,
            lead_steps=int(lt),
            time_step_hours=data_cfg.time_step_hours,
            var_names=data_cfg.dynamic_vars,
            device=dev,
            progress_desc=f"final test lead {tag}",
        )
        results["val"][tag] = res_val
        results["test"][tag] = res_test

    save_json(results, out_dir / "results.json")
    return results


def main():
    p = argparse.ArgumentParser(description="UrbanPiDiT baseline: Simple 3-layer MLP")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--out_dir", type=str, default="outputs/baselines/mlp")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--load_dir", type=str, default=None)
    p.add_argument("--skip_fit", action="store_true")
    p.add_argument("--bench_iters", type=int, default=50)
    p.add_argument("--bench_warmup", type=int, default=10)
    p.add_argument("--bench_split", type=str, default="test")

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--early_stop_patience", type=int, default=10)

    args = p.parse_args()
    train_cfg = TrainCfg(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden=args.hidden,
        dropout=args.dropout,
        grad_clip=args.grad_clip,
        early_stop_patience=args.early_stop_patience,
    )

    res = run_mlp_baseline(
        config_path=args.config,
        train_cfg=train_cfg,
        out_dir=args.out_dir,
        device=args.device,
        load_dir=args.load_dir,
        skip_fit=bool(args.skip_fit),
        bench_iters=int(args.bench_iters),
        bench_warmup=int(args.bench_warmup),
        bench_split=str(args.bench_split),
    )
    import json

    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
