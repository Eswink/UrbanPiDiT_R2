# -*- coding: utf-8 -*-
"""
UrbanPiDiT qualitative case plot (final publication version)

特性：
1. 默认横向布局：行 = GT / Pred / Error，列 = 多个变量
2. 顶部列标题带单位，例如 t2m [K], u10 [m s^-1]
3. 左侧只保留 GT / Pred / Error，不再重复单位
4. 适合论文排版的更宽松列间距与色条间距
5. 保留 auto_select、24h 单 lead 绘图、publication colormap 等能力
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    import pytorch_lightning as pl
except Exception as e:
    raise ImportError("该脚本需要安装 pytorch_lightning。请先安装后再运行。") from e

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loader import MetroWeatherDataModule
from pidit_lit import UrbanPiDiTLitModule


# 变量元信息（与论文中变量单位保持一致）
VAR_META = {
    "t2m": {"label": "t2m", "unit": "K"},
    "d2m": {"label": "d2m", "unit": "K"},
    "u10": {"label": "u10", "unit": r"m s$^{-1}$"},
    "v10": {"label": "v10", "unit": r"m s$^{-1}$"},
    "sp": {"label": "sp", "unit": "Pa"},
    "tp": {"label": "tp", "unit": "m"},
    "tcc": {"label": "tcc", "unit": "1"},
}


def _parse_csv_str(s: str) -> List[str]:
    s = str(s).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_csv_ints(s: str) -> List[int]:
    return [int(x) for x in _parse_csv_str(s)]


def _lead_tag(time_step_hours: float, lead_steps: int) -> str:
    hours = float(time_step_hours) * float(int(lead_steps))
    if abs(hours - round(hours)) < 1e-6:
        return f"{int(round(hours))}h"
    return f"{hours:.1f}h"


def _var_title(var_name: str, title_style: str = "compact") -> str:
    meta = VAR_META.get(var_name, {"label": var_name, "unit": ""})
    label = meta.get("label", var_name)
    unit = meta.get("unit", "")
    if not unit or title_style == "plain":
        return label
    if title_style == "newline":
        return f"{label}\n[{unit}]"
    return f"{label} [{unit}]"


def _resolve_cmap(name: str):
    name = str(name)
    if name.startswith("cmc."):
        try:
            import cmcrameri.cm as cmc
            return getattr(cmc, name.split(".", 1)[1])
        except Exception:
            fallback = {"cmc.batlow": "cividis", "cmc.vik": "RdBu_r"}.get(name, "viridis")
            return plt.get_cmap(fallback)
    return plt.get_cmap(name)


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


def _build_dm(cfg: Dict, *, batch_size: int, num_workers: int) -> MetroWeatherDataModule:
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
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        dynamic_vars=dyn,
        static_vars=stat,
        include_static=include_static,
        broadcast_static=broadcast_static,
        static_perturb={"mode": "none"},
    )
    dm.prepare_data()
    dm.setup(stage="test")
    return dm


def _denorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std + mean


def _pick_gt(sample: Dict, *, lead_steps: int) -> torch.Tensor:
    if "y" not in sample:
        return sample["x0"]
    lead_list = sample["lead_times"].detach().cpu().tolist()
    try:
        li = lead_list.index(int(lead_steps))
    except Exception as e:
        raise KeyError(f"lead_steps={lead_steps} 不在 sample.lead_times={lead_list}") from e
    return sample["y"][li]


def _predict_one(
    lit: UrbanPiDiTLitModule,
    x_ctx: torch.Tensor,
    *,
    lead_steps: int,
    steps: int,
    t_start: float,
    t_end: float,
    seed: Optional[int],
) -> torch.Tensor:
    lt_cond = None
    if getattr(lit, "use_lead_time_conditioning", False):
        lt_steps_t = torch.full((x_ctx.size(0),), float(int(lead_steps)), device=x_ctx.device, dtype=torch.float32)
        lt_cond = lit._normalize_lead_time(lt_steps_t)
    return lit.predict_one(
        x_ctx,
        steps=int(steps),
        t_start=float(t_start),
        t_end=float(t_end),
        seed=seed,
        lead_time=lt_cond,
    )


def _to_numpy2(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def _auto_select_sample(
    *,
    ds,
    lit: UrbanPiDiTLitModule,
    device: torch.device,
    dyn_vars: List[str],
    vars_: Sequence[str],
    mode: str,
    max_samples: int,
    seed: int,
    steps: int,
    t_start: float,
    t_end: float,
    lead_steps: int,
) -> int:
    total = min(int(max_samples), len(ds))
    if total <= 0:
        raise ValueError("max_samples 必须大于 0")

    if mode == "random":
        rng = np.random.default_rng(int(seed))
        return int(rng.integers(0, total))

    var_indices = [dyn_vars.index(v) for v in vars_]
    best_idx = 0
    best_score = -1.0

    for i in range(total):
        sample = ds[i]
        x_ctx = sample["x_ctx"].unsqueeze(0).to(device)
        norm = sample["norm"]
        mean = norm["mean"].unsqueeze(0).to(device)
        std = norm["std"].unsqueeze(0).to(device)

        gt_norm = _pick_gt(sample, lead_steps=lead_steps).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_norm = _predict_one(
                lit,
                x_ctx,
                lead_steps=lead_steps,
                steps=steps,
                t_start=t_start,
                t_end=t_end,
                seed=seed,
            )

        gt = _denorm(gt_norm, mean, std)[0, var_indices]
        pred = _denorm(pred_norm, mean, std)[0, var_indices]
        diff = pred - gt

        if mode == "max_error":
            score = float(torch.max(torch.abs(diff)).item())
        elif mode == "max_diff":
            score = float(torch.max(diff).item())
        else:
            raise ValueError(f"Unknown auto_select mode: {mode}")

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def _style_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)
        spine.set_color("0.65")


def _plot_one_lead(
    *,
    out_path: str,
    title: str,
    vars_: Sequence[str],
    gt: torch.Tensor,
    pred: torch.Tensor,
    cmap,
    err_cmap,
    tcc_fixed_01: bool,
    dpi: int,
    fig_w: float,
    row_h: float,
    wspace: float,
    hspace: float,
    cbar_size: str,
    cbar_pad: float,
    show_suptitle: bool,
    layout: str,
    col_w: float,
    title_style: str,
    label_fontsize: float,
    title_fontsize: float,
    row_label_fontsize: float,
):
    gt_np = _to_numpy2(gt)
    pred_np = _to_numpy2(pred)
    err_np = pred_np - gt_np

    layout = str(layout).lower()
    nvars = len(vars_)

    if layout == "horizontal":
        nrows, ncols = 3, nvars
        fig_h = max(float(row_h) * nrows, 3.4)
        fig_w_eff = max(float(col_w) * ncols, float(fig_w))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(fig_w_eff, fig_h),
            squeeze=False,
            gridspec_kw={"wspace": float(wspace), "hspace": float(hspace)},
        )
        fig.patch.set_facecolor("white")

        row_labels = ["GT", "Pred", "Error"]
        for j, v in enumerate(vars_):
            axes[0][j].set_title(_var_title(v, title_style=title_style), fontsize=title_fontsize, pad=10)

            g = gt_np[j]
            p = pred_np[j]
            e = err_np[j]

            if tcc_fixed_01 and v == "tcc":
                vmin, vmax = 0.0, 1.0
            else:
                vmin = float(np.min([g.min(), p.min()]))
                vmax = float(np.max([g.max(), p.max()]))
                if not np.isfinite(vmin) or not np.isfinite(vmax):
                    vmin, vmax = 0.0, 1.0
                if abs(vmax - vmin) < 1e-12:
                    vmax = vmin + 1e-6

            emax = float(np.max(np.abs(e))) if np.isfinite(e).any() else 1.0
            if emax < 1e-12:
                emax = 1e-6

            axes[0][j].imshow(g, cmap=cmap, vmin=vmin, vmax=vmax)
            im1 = axes[1][j].imshow(p, cmap=cmap, vmin=vmin, vmax=vmax)
            im2 = axes[2][j].imshow(e, cmap=err_cmap, vmin=-emax, vmax=emax)

            for i in range(3):
                _style_axis(axes[i][j])

            divider_main = make_axes_locatable(axes[1][j])
            cax_main = divider_main.append_axes("right", size=str(cbar_size), pad=float(cbar_pad))
            cb1 = fig.colorbar(im1, cax=cax_main)
            cb1.ax.tick_params(labelsize=label_fontsize, length=2)

            divider_err = make_axes_locatable(axes[2][j])
            cax_err = divider_err.append_axes("right", size=str(cbar_size), pad=float(cbar_pad))
            cb2 = fig.colorbar(im2, cax=cax_err)
            cb2.ax.tick_params(labelsize=label_fontsize, length=2)

        for i, lab in enumerate(row_labels):
            axes[i][0].set_ylabel(lab, fontsize=row_label_fontsize)

    else:
        nrows, ncols = nvars, 3
        fig_h = max(float(row_h) * nrows, 2.2)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(float(fig_w), fig_h),
            squeeze=False,
            gridspec_kw={"wspace": float(wspace), "hspace": float(hspace)},
        )
        fig.patch.set_facecolor("white")
        axes[0][0].set_title("GT", fontsize=title_fontsize, pad=8)
        axes[0][1].set_title("Pred", fontsize=title_fontsize, pad=8)
        axes[0][2].set_title("Error", fontsize=title_fontsize, pad=8)

        for i, v in enumerate(vars_):
            g = gt_np[i]
            p = pred_np[i]
            e = err_np[i]

            if tcc_fixed_01 and v == "tcc":
                vmin, vmax = 0.0, 1.0
            else:
                vmin = float(np.min([g.min(), p.min()]))
                vmax = float(np.max([g.max(), p.max()]))
                if not np.isfinite(vmin) or not np.isfinite(vmax):
                    vmin, vmax = 0.0, 1.0
                if abs(vmax - vmin) < 1e-12:
                    vmax = vmin + 1e-6

            axes[i][0].imshow(g, cmap=cmap, vmin=vmin, vmax=vmax)
            im1 = axes[i][1].imshow(p, cmap=cmap, vmin=vmin, vmax=vmax)

            emax = float(np.max(np.abs(e))) if np.isfinite(e).any() else 1.0
            if emax < 1e-12:
                emax = 1e-6
            im2 = axes[i][2].imshow(e, cmap=err_cmap, vmin=-emax, vmax=emax)

            axes[i][0].set_ylabel(_var_title(v, title_style=title_style), fontsize=row_label_fontsize)
            for j in range(3):
                _style_axis(axes[i][j])

            divider_main = make_axes_locatable(axes[i][1])
            cax_main = divider_main.append_axes("right", size=str(cbar_size), pad=float(cbar_pad))
            cb1 = fig.colorbar(im1, cax=cax_main)
            cb1.ax.tick_params(labelsize=label_fontsize, length=2)

            divider_err = make_axes_locatable(axes[i][2])
            cax_err = divider_err.append_axes("right", size=str(cbar_size), pad=float(cbar_pad))
            cb2 = fig.colorbar(im2, cax=cax_err)
            cb2.ax.tick_params(labelsize=label_fontsize, length=2)

    if show_suptitle and title:
        fig.suptitle(title, fontsize=max(title_fontsize + 1, 12), y=0.995)
        fig.subplots_adjust(left=0.055, right=0.995, bottom=0.05, top=0.90)
    else:
        fig.subplots_adjust(left=0.055, right=0.995, bottom=0.05, top=0.95)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=int(dpi), bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--auto_select", type=str, default="none")
    parser.add_argument("--auto_max_samples", type=int, default=200)
    parser.add_argument("--vars", type=str, default="t2m,u10,v10,tp,tcc")
    parser.add_argument("--leads", type=str, default="")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--t_start", type=float, default=None)
    parser.add_argument("--t_end", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cmap", type=str, default="cmc.batlow")
    parser.add_argument("--err_cmap", type=str, default="cmc.vik")
    parser.add_argument("--tcc_fixed_01", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--fig_w", type=float, default=10.8)
    parser.add_argument("--row_h", type=float, default=1.62)
    parser.add_argument("--wspace", type=float, default=0.14)
    parser.add_argument("--hspace", type=float, default=0.12)
    parser.add_argument("--cbar_size", type=str, default="2.2%")
    parser.add_argument("--cbar_pad", type=float, default=0.05)
    parser.add_argument("--show_suptitle", action="store_true")
    parser.add_argument("--layout", type=str, default="horizontal", choices=["horizontal", "vertical"])
    parser.add_argument("--col_w", type=float, default=1.78)
    parser.add_argument("--title_style", type=str, default="compact", choices=["compact", "newline", "plain"])
    parser.add_argument("--label_fontsize", type=float, default=8.5)
    parser.add_argument("--title_fontsize", type=float, default=11.5)
    parser.add_argument("--row_label_fontsize", type=float, default=12)
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": args.title_fontsize,
            "axes.labelsize": args.row_label_fontsize,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
        }
    )

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(args.seed) if args.seed is not None else int(cfg.get("train", {}).get("seed", 42))
    pl.seed_everything(seed, workers=True)

    forecast_cfg = dict(cfg.get("forecast", {}) or {})
    time_step_hours = float(forecast_cfg.get("time_step_hours", 6.0))
    dyn_vars = list(cfg.get("dynamic_vars", []))

    vars_in = _parse_csv_str(args.vars)
    if not vars_in:
        vars_in = dyn_vars
    vars_ = [v for v in vars_in if v in dyn_vars]
    if not vars_:
        raise ValueError(f"vars 为空或不在 dynamic_vars 中: vars={vars_in} dynamic_vars={dyn_vars}")

    if str(args.leads).strip():
        leads = _parse_csv_ints(args.leads)
    else:
        leads = list(forecast_cfg.get("eval_lead_times", forecast_cfg.get("lead_times", [cfg.get("delta_t", 1)])))
    leads = sorted(list(dict.fromkeys([int(x) for x in leads])))

    diff_cfg = dict(cfg.get("diffusion", {}) or {})
    steps = int(args.steps) if args.steps is not None else int(diff_cfg.get("sample_steps", 8))
    t_start = float(args.t_start) if args.t_start is not None else float(diff_cfg.get("t_start", 0.0))
    t_end = float(args.t_end) if args.t_end is not None else float(diff_cfg.get("t_end", 1.0))

    dm = _build_dm(cfg, batch_size=1, num_workers=0)
    split = str(args.split).lower()
    if split == "train":
        ds = dm.train_ds
    elif split == "val":
        ds = dm.val_ds
    else:
        ds = dm.test_ds

    auto_mode = str(args.auto_select).lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit = _build_lit(cfg, args.ckpt).to(device)
    lit.eval()

    sample_idx = int(args.sample_idx)
    if auto_mode != "none":
        if auto_mode not in ("max_error", "max_diff", "random"):
            raise ValueError("auto_select 只支持 none/max_error/max_diff/random")
        lead_for_select = int(leads[0])
        sample_idx = _auto_select_sample(
            ds=ds,
            lit=lit,
            device=device,
            dyn_vars=dyn_vars,
            vars_=vars_,
            mode=auto_mode,
            max_samples=int(args.auto_max_samples),
            seed=seed,
            steps=steps,
            t_start=t_start,
            t_end=t_end,
            lead_steps=lead_for_select,
        )
        print(f"Auto-selected sample_idx={sample_idx} mode={auto_mode} lead_steps={lead_for_select}")

    sample = ds[sample_idx]
    x_ctx = sample["x_ctx"].unsqueeze(0).to(device)
    norm = sample["norm"]
    mean = norm["mean"].unsqueeze(0).to(device)
    std = norm["std"].unsqueeze(0).to(device)

    var_indices = [dyn_vars.index(v) for v in vars_]
    cmap = _resolve_cmap(args.cmap)
    err_cmap = _resolve_cmap(args.err_cmap)

    for lead_steps in leads:
        lead_h = _lead_tag(time_step_hours, lead_steps)
        gt_norm = _pick_gt(sample, lead_steps=lead_steps).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_norm = _predict_one(
                lit,
                x_ctx,
                lead_steps=lead_steps,
                steps=steps,
                t_start=t_start,
                t_end=t_end,
                seed=seed,
            )

        gt = _denorm(gt_norm, mean, std)[0, var_indices]
        pred = _denorm(pred_norm, mean, std)[0, var_indices]

        out_path = str(Path(args.out_dir) / f"case_idx{int(sample_idx):04d}_{lead_h}.png")
        title = f"Sample {int(sample_idx)} | Lead {lead_h}"
        _plot_one_lead(
            out_path=out_path,
            title=title,
            vars_=vars_,
            gt=gt,
            pred=pred,
            cmap=cmap,
            err_cmap=err_cmap,
            tcc_fixed_01=bool(args.tcc_fixed_01),
            dpi=int(args.dpi),
            fig_w=float(args.fig_w),
            row_h=float(args.row_h),
            wspace=float(args.wspace),
            hspace=float(args.hspace),
            cbar_size=str(args.cbar_size),
            cbar_pad=float(args.cbar_pad),
            show_suptitle=bool(args.show_suptitle),
            layout=str(args.layout),
            col_w=float(args.col_w),
            title_style=str(args.title_style),
            label_fontsize=float(args.label_fontsize),
            title_fontsize=float(args.title_fontsize),
            row_label_fontsize=float(args.row_label_fontsize),
        )

    print(f"Saved figures to: {args.out_dir}")


if __name__ == "__main__":
    main()
