## 静态特征图相关可视化

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

try:
    import pytorch_lightning as pl
except Exception as e:
    raise ImportError("该脚本需要安装 pytorch_lightning。请先安装后再运行。") from e

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loader import MetroWeatherDataModule, BeijingWeatherDataset
from pidit_lit import UrbanPiDiTLitModule
PERTURBS: List[Tuple[str, str]] = [
    ("spatial_shuffle", "shuffle_hw"),
    ("rot90", "rot90"),
    ("zero_out", "fill_zero"),
]


def load_results(fp: str) -> Dict:
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_rmse_increase_by_var(results: Dict, lead_tag: str, vars_: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    base = results.get("baseline", {})
    for var in vars_:
        base_key = f"test/RMSE_{var}@{lead_tag}"
        b = base.get(base_key, None)
        out[var] = {}
        for exp, _ in PERTURBS:
            r = results.get(exp, {})
            k = f"test/RMSE_{var}@{lead_tag}"
            v = r.get(k, None)
            if b is None or v is None or b == 0:
                out[var][exp] = np.nan
            else:
                out[var][exp] = (float(v) - float(b)) / float(b) * 100.0
    return out


def plot_bar_rmse_increase(rmse_inc: Dict[str, Dict[str, float]], vars_: List[str], *, title: str, save_path: str):
    x = np.arange(len(vars_))
    width = 0.28

    def vals_of(exp: str) -> List[float]:
        return [rmse_inc[v].get(exp, np.nan) for v in vars_]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - width, vals_of("spatial_shuffle"), width, label="Shuffle")
    ax.bar(x, vals_of("rot90"), width, label="Rotate")
    ax.bar(x + width, vals_of("zero_out"), width, label="Zero-out")

    ax.set_xticks(x)
    ax.set_xticklabels(vars_, rotation=0)
    ax.set_ylabel("RMSE Relative Increase (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def _build_lit(cfg: Dict, ckpt_path: str) -> UrbanPiDiTLitModule:
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


def _build_dataset(cfg: Dict, *, static_perturb: Dict) -> BeijingWeatherDataset:
    data_root = cfg["data_root"]
    dyn = cfg["dynamic_vars"]
    stat = cfg.get("static_vars", [])
    include_static = bool(cfg.get("include_static", True))
    broadcast_static = bool(cfg.get("broadcast_static", False))
    k = int(cfg["k"])
    delta_t = int(cfg["delta_t"])
    forecast_cfg = dict(cfg.get("forecast", {}) or {})
    lead_times = forecast_cfg.get("lead_times", None)

    dm = MetroWeatherDataModule(
        data_root=data_root,
        k=k,
        delta_t=delta_t,
        lead_times=lead_times,
        batch_size=1,
        num_workers=0,
        dynamic_vars=dyn,
        static_vars=stat,
        include_static=include_static,
        broadcast_static=broadcast_static,
        static_perturb=static_perturb,
    )
    dm.prepare_data()
    dm.setup(stage="test")
    return dm.test_ds


def _get_building_map(ds: BeijingWeatherDataset, idx: int, var_name: str) -> np.ndarray:
    mode, payload, t_idx = ds.index[idx]
    if mode == "npz":
        data = np.load(payload)
        if var_name not in data:
            raise KeyError(f"{var_name} 不在 {payload}")
        x = data[var_name][t_idx] if ds.broadcast_static else data[var_name][0]
        return np.asarray(x, dtype=np.float32).squeeze()
    flist = payload
    f_static = flist[t_idx] if ds.broadcast_static else flist[0]
    arr_s = np.load(f_static)
    from data.loader import CHANNEL_MAPPING

    chs = CHANNEL_MAPPING.get(var_name, None)
    if chs is None:
        raise KeyError(f"{var_name} 不在 CHANNEL_MAPPING")
    return np.asarray(arr_s[chs], dtype=np.float32).squeeze()


def _predict_one(lit: UrbanPiDiTLitModule, x_ctx: torch.Tensor, *, lead_steps: int, cfg: Dict, seed: int):
    diff_cfg = cfg["diffusion"]
    steps = int(diff_cfg.get("sample_steps", 8))
    t_start = float(diff_cfg.get("t_start", 0.0))
    t_end = float(diff_cfg.get("t_end", 1.0))
    lt_cond = None
    if lit.use_lead_time_conditioning:
        lt_steps_t = torch.full((x_ctx.size(0),), float(lead_steps), device=x_ctx.device, dtype=torch.float32)
        lt_cond = lit._normalize_lead_time(lt_steps_t)
    return lit.predict_one(x_ctx, steps=steps, t_start=t_start, t_end=t_end, seed=seed, lead_time=lt_cond)


def _lead_tag(time_step_hours: float, lead_steps: int) -> str:
    hours = float(time_step_hours) * float(int(lead_steps))
    if abs(hours - round(hours)) < 1e-6:
        return f"{int(round(hours))}h"
    return f"{hours:.1f}h"


def _predict_pair(
    lit: UrbanPiDiTLitModule,
    sample_base: Dict,
    sample_rot: Dict,
    *,
    lead_steps: int,
    cfg: Dict,
    seed: int,
    device: torch.device,
):
    x_ctx_base = sample_base["x_ctx"].unsqueeze(0).to(device)
    x_ctx_rot = sample_rot["x_ctx"].unsqueeze(0).to(device)
    norm = sample_base["norm"]
    mean = norm["mean"].unsqueeze(0).to(device)
    std = norm["std"].unsqueeze(0).to(device)

    pred_base = _predict_one(lit, x_ctx_base, lead_steps=lead_steps, cfg=cfg, seed=seed)
    pred_rot = _predict_one(lit, x_ctx_rot, lead_steps=lead_steps, cfg=cfg, seed=seed)

    pred_base = pred_base * std + mean
    pred_rot = pred_rot * std + mean
    return pred_base, pred_rot


def _select_typical_sample(
    ds_base: BeijingWeatherDataset,
    ds_rot: BeijingWeatherDataset,
    lit: UrbanPiDiTLitModule,
    *,
    lead_steps: int,
    cfg: Dict,
    seed: int,
    var_idx: int,
    device: torch.device,
    max_samples: int,
):
    total = min(len(ds_base), len(ds_rot), max_samples)
    best_idx = 0
    best_score = -1.0
    for i in range(total):
        sample_base = ds_base[i]
        sample_rot = ds_rot[i]
        pred_base, pred_rot = _predict_pair(
            lit,
            sample_base,
            sample_rot,
            lead_steps=lead_steps,
            cfg=cfg,
            seed=seed,
            device=device,
        )
        diff = pred_base - pred_rot
        diff_var = diff[0, var_idx]
        score = torch.max(torch.abs(diff_var)).item()
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx, best_score


def plot_diff_with_building(diff_arr: np.ndarray, building_arr: np.ndarray, *, save_path: str, title: str, cmap: str = "RdBu_r"):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    im0 = axes[0].imshow(diff_arr, cmap=cmap)
    axes[0].set_title("Prediction_Original - Prediction_Rotated")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(building_arr, cmap="inferno", vmin=0.0, vmax=1.0)
    axes[1].set_title("Building Density")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_json", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--lead_steps", type=int, default=2)
    parser.add_argument("--var", type=str, default="t2m")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--auto_sample", action="store_true")
    parser.add_argument("--max_auto_samples", type=int, default=200)
    parser.add_argument("--building_var", type=str, default="buildings")
    args = parser.parse_args()

    results = load_results(args.results_json)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    seed = int(cfg.get("train", {}).get("seed", 42))
    pl.seed_everything(seed, workers=True)

    forecast_cfg = dict(cfg.get("forecast", {}) or {})
    time_step_hours = float(forecast_cfg.get("time_step_hours", 6.0))
    lead_tag = _lead_tag(time_step_hours, args.lead_steps)
    vars_ = list(cfg.get("dynamic_vars", []))
    rmse_inc = compute_rmse_increase_by_var(results, lead_tag=lead_tag, vars_=vars_)
    plot_bar_rmse_increase(
        rmse_inc,
        vars_,
        title=f"RMSE Relative Increase @ {lead_tag} (by Variable)",
        save_path=str(Path(args.out_dir) / "rmse_increase.png"),
    )

    ds_base = _build_dataset(cfg, static_perturb={"mode": "none"})
    ds_rot = _build_dataset(cfg, static_perturb={"mode": "rot90", "k": 1})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit = _build_lit(cfg, args.ckpt).to(device)
    lit.eval()

    var_idx = cfg["dynamic_vars"].index(args.var)
    sample_idx = args.sample_idx
    if args.auto_sample:
        with torch.no_grad():
            sample_idx, score = _select_typical_sample(
                ds_base,
                ds_rot,
                lit,
                lead_steps=args.lead_steps,
                cfg=cfg,
                seed=seed,
                var_idx=var_idx,
                device=device,
                max_samples=args.max_auto_samples,
            )
        print(f"Auto-selected sample_idx={sample_idx} score={score:.4f}")

    sample_base = ds_base[sample_idx]
    sample_rot = ds_rot[sample_idx]

    with torch.no_grad():
        pred_base, pred_rot = _predict_pair(
            lit,
            sample_base,
            sample_rot,
            lead_steps=args.lead_steps,
            cfg=cfg,
            seed=seed,
            device=device,
        )
    diff = (pred_base - pred_rot)[0, var_idx].detach().cpu().numpy()

    building = _get_building_map(ds_base, sample_idx, args.building_var)
    plot_diff_with_building(
        diff,
        building,
        save_path=str(Path(args.out_dir) / "diff_vs_building.png"),
        title=f"{args.var} @ {lead_tag}",
    )

    print(f"Saved figures to: {args.out_dir}")


if __name__ == "__main__":
    main()
