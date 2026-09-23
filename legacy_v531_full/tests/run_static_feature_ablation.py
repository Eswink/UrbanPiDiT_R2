import argparse
import csv
import json
import os
import sys
from copy import deepcopy

import torch
import yaml

try:
    import pytorch_lightning as pl
except Exception as e:
    raise ImportError("该脚本需要安装 pytorch_lightning。请先安装后再运行。") from e

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import MetroWeatherDataModule
from pidit_lit import UrbanPiDiTLitModule


def _hours_tag(time_step_hours: float, lead_steps: int) -> str:
    hours = float(time_step_hours) * float(int(lead_steps))
    if abs(hours - round(hours)) < 1e-6:
        return f"{int(round(hours))}h"
    return f"{hours:.1f}h"


def _to_float(v):
    if v is None:
        return None
    if isinstance(v, (float, int)):
        return float(v)
    try:
        if hasattr(v, "detach"):
            v = v.detach()
        if hasattr(v, "item"):
            return float(v.item())
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return None


def _build_dm(cfg, *, batch_size=None, num_workers=None, static_perturb=None):
    data_root = cfg["data_root"]
    dyn = cfg["dynamic_vars"]
    stat = cfg.get("static_vars", [])
    include_static = bool(cfg.get("include_static", True))
    broadcast_static = bool(cfg.get("broadcast_static", False))
    k = int(cfg["k"])
    delta_t = int(cfg["delta_t"])

    forecast_cfg = dict(cfg.get("forecast", {}) or {})
    lead_times = forecast_cfg.get("lead_times", None)

    if batch_size is None:
        batch_size = int(cfg.get("eval", {}).get("batch_size", cfg.get("train", {}).get("batch_size", 1)))
    if num_workers is None:
        num_workers = int(cfg.get("eval", {}).get("num_workers", cfg.get("train", {}).get("num_workers", 8)))

    dm = MetroWeatherDataModule(
        data_root=data_root,
        k=k,
        delta_t=delta_t,
        lead_times=lead_times,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        dynamic_vars=dyn,
        static_vars=stat,
        include_static=include_static,
        broadcast_static=broadcast_static,
        static_perturb=static_perturb,
    )
    dm.prepare_data()
    dm.setup(stage="test")
    return dm


def _build_lit(cfg, ckpt_path: str):
    dyn = cfg["dynamic_vars"]
    stat = cfg.get("static_vars", [])
    include_static = bool(cfg.get("include_static", True))
    k = int(cfg["k"])

    forecast_cfg = dict(cfg.get("forecast", {}) or {})
    ablation = cfg.get("ablation", {})
    urban_graph_cfg = cfg.get("urban_graph", {})

    drop_path_rate = float(cfg["model"].get("drop_path_rate", 0.0))
    if not bool(ablation.get("use_drop_path", True)):
        drop_path_rate = 0.0

    model_cfg = {
        "H": int(cfg.get("H", 8)),
        "W": int(cfg.get("W", 8)),
        "D": int(cfg["model"]["D"]),
        "depth": int(cfg["model"]["depth"]),
        "heads": int(cfg["model"]["heads"]),
        "mlp_ratio": float(cfg["model"]["mlp_ratio"]),
        "drop_path_rate": drop_path_rate,
        "dropout": float(cfg["model"].get("dropout", 0.0)),
        "in_channels": len(dyn),
        "ctx_channels": len(dyn) * k + (len(stat) if include_static else 0),
        "static_channels": (len(stat) if include_static else 0),
        "out_channels": len(dyn),
        "use_cross_attention": bool(ablation.get("use_cross_attention", True)),
        "use_position_encoding": bool(ablation.get("use_position_encoding", True)),
        "use_timestep_conditioning": bool(ablation.get("use_timestep_conditioning", True)),
        "use_lead_time_conditioning": bool(ablation.get("use_lead_time_conditioning", False)),
        "use_hybrid_attention": bool(ablation.get("use_hybrid_attention", False)),
        "use_multi_scale": bool(ablation.get("use_multi_scale", False)),
        "use_variable_graph": bool(ablation.get("use_variable_graph", False)),
        "use_urban_graph": bool(ablation.get("use_urban_graph", False)),
        "urban_graph_k": int(urban_graph_cfg.get("k", 8)),
        "urban_graph_sigma": float(urban_graph_cfg.get("sigma", 1.0)),
    }

    optim_cfg = {
        "lr": float(cfg["train"]["lr"]),
        "weight_decay": float(cfg["train"].get("weight_decay", 1e-4)),
        "max_epochs": int(cfg["train"].get("max_epochs", 200)),
        "loss_weights": cfg["train"].get("loss_weights", None),
        "use_channel_weights": bool(ablation.get("use_channel_weights", True)),
        "use_adaptive_weights": bool(ablation.get("use_adaptive_weights", False)),
    }
    if optim_cfg["loss_weights"] is None:
        optim_cfg.pop("loss_weights")

    diffusion_cfg = cfg["diffusion"]
    physics_cfg = dict(cfg.get("physics", {}))
    physics_cfg["use_physics_loss"] = bool(ablation.get("use_physics_loss", True))
    physics_cfg["use_fft_loss"] = bool(ablation.get("use_fft_loss", True))
    physics_cfg["use_gradient_loss"] = bool(ablation.get("use_gradient_loss", True))
    physics_cfg["use_hard_physics"] = bool(ablation.get("use_hard_physics", False))
    physics_cfg["use_adversarial"] = bool(ablation.get("use_adversarial", False))
    physics_cfg["use_distillation"] = bool(ablation.get("use_distillation", False))

    metrics_cfg = dict(cfg.get("metrics", {}) or {})
    metrics_cfg["var_names"] = dyn
    inference_cfg = cfg.get("inference", {})

    return UrbanPiDiTLitModule.load_from_checkpoint(
        ckpt_path,
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        diffusion_cfg=diffusion_cfg,
        physics_cfg=physics_cfg,
        metrics_cfg=metrics_cfg,
        inference_cfg=inference_cfg,
        forecast_cfg=forecast_cfg,
    )


def _run_one(trainer, cfg, ckpt_path: str, *, static_perturb, batch_size=None, num_workers=None):
    dm = _build_dm(cfg, batch_size=batch_size, num_workers=num_workers, static_perturb=static_perturb)
    lit = _build_lit(cfg, ckpt_path)
    outs = trainer.test(lit, datamodule=dm, verbose=False)
    if not outs:
        return {}
    out0 = outs[0] if isinstance(outs, list) else outs
    return {k: _to_float(v) for k, v in dict(out0).items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--limit_test_batches", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("train", {}).get("seed", 42))
    pl.seed_everything(seed, workers=True)

    os.makedirs(args.out_dir, exist_ok=True)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = "auto"
    if args.device is not None:
        d = str(args.device).lower()
        if d in ("cpu", "gpu"):
            accelerator = d
            devices = "auto"

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        limit_test_batches=float(args.limit_test_batches),
    )

    experiments = [
        ("baseline", {"mode": "none"}),
        ("spatial_shuffle", {"mode": "shuffle_hw", "seed": seed, "deterministic": True}),
        ("mean_out", {"mode": "fill_mean"}),
        ("zero_out", {"mode": "fill_zero"}),
        ("rot90", {"mode": "rot90", "k": 1}),
        ("flip_ud", {"mode": "flip_ud"}),
    ]

    results_by_exp = {}
    for name, sp in experiments:
        cfg_i = deepcopy(cfg)
        cfg_i["static_perturb"] = dict(sp)
        results_by_exp[name] = _run_one(
            trainer,
            cfg_i,
            args.ckpt,
            static_perturb=cfg_i.get("static_perturb", None),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results_by_exp, f, indent=2, sort_keys=True, ensure_ascii=False)

    forecast_cfg = dict(cfg.get("forecast", {}) or {})
    time_step_hours = float(forecast_cfg.get("time_step_hours", 6.0))
    eval_leads = [int(x) for x in forecast_cfg.get("eval_lead_times", forecast_cfg.get("lead_times", [1]))]
    eval_leads = sorted(list(dict.fromkeys(eval_leads)))
    lead_tags = {l: _hours_tag(time_step_hours, l) for l in eval_leads}

    def get_metric(exp_name: str, key: str):
        return results_by_exp.get(exp_name, {}).get(key, None)

    baseline = results_by_exp.get("baseline", {})
    rows = []
    mode_by_exp = {name: dict(sp).get("mode", "none") for name, sp in experiments}
    for exp_name, _ in experiments:
        for l in eval_leads:
            tag = lead_tags[l]
            row = {
                "experiment": exp_name,
                "static_mode": mode_by_exp.get(exp_name, "none"),
                "lead_steps": l,
                "lead_hours": float(time_step_hours) * float(l),
                "MAE": get_metric(exp_name, f"test/MAE@{tag}"),
                "RMSE": get_metric(exp_name, f"test/RMSE@{tag}"),
                "CRPS": get_metric(exp_name, f"test/CRPS@{tag}"),
                "ACC": get_metric(exp_name, f"test/ACC@{tag}"),
                "dMAE": None,
                "dRMSE": None,
                "dCRPS": None,
                "dACC": None,
            }
            if baseline:
                b_mae = baseline.get(f"test/MAE@{tag}", None)
                b_rmse = baseline.get(f"test/RMSE@{tag}", None)
                b_crps = baseline.get(f"test/CRPS@{tag}", None)
                b_acc = baseline.get(f"test/ACC@{tag}", None)
                if row["MAE"] is not None and b_mae is not None:
                    row["dMAE"] = row["MAE"] - b_mae
                if row["RMSE"] is not None and b_rmse is not None:
                    row["dRMSE"] = row["RMSE"] - b_rmse
                if row["CRPS"] is not None and b_crps is not None:
                    row["dCRPS"] = row["CRPS"] - b_crps
                if row["ACC"] is not None and b_acc is not None:
                    row["dACC"] = row["ACC"] - b_acc
            rows.append(row)

        mean_row = {
            "experiment": exp_name,
            "static_mode": mode_by_exp.get(exp_name, "none"),
            "lead_steps": "mean_leads",
            "lead_hours": "",
            "MAE": get_metric(exp_name, "test/MAE_mean_leads"),
            "RMSE": get_metric(exp_name, "test/RMSE_mean_leads"),
            "CRPS": "",
            "ACC": get_metric(exp_name, "test/ACC_mean_leads"),
            "dMAE": None,
            "dRMSE": None,
            "dCRPS": "",
            "dACC": None,
        }
        if baseline:
            b_mae = baseline.get("test/MAE_mean_leads", None)
            b_rmse = baseline.get("test/RMSE_mean_leads", None)
            b_acc = baseline.get("test/ACC_mean_leads", None)
            if mean_row["MAE"] is not None and b_mae is not None:
                mean_row["dMAE"] = mean_row["MAE"] - b_mae
            if mean_row["RMSE"] is not None and b_rmse is not None:
                mean_row["dRMSE"] = mean_row["RMSE"] - b_rmse
            if mean_row["ACC"] is not None and b_acc is not None:
                mean_row["dACC"] = mean_row["ACC"] - b_acc
        rows.append(mean_row)

    csv_path = os.path.join(args.out_dir, "report.csv")
    fieldnames = ["experiment", "static_mode", "lead_steps", "lead_hours", "MAE", "RMSE", "CRPS", "ACC", "dMAE", "dRMSE", "dCRPS", "dACC"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
