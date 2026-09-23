## 频谱诊断图

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

try:
    import pytorch_lightning as pl
except Exception as e:
    raise ImportError("该脚本需要安装 pytorch_lightning。请先安装后再运行。") from e

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loader import MetroWeatherDataModule
from pidit_lit import UrbanPiDiTLitModule


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


def _pick_gt(sample: Dict, *, lead_steps: int) -> torch.Tensor:
    if "y" not in sample:
        return sample["x0"]
    lead_list = sample["lead_times"].detach().cpu().tolist()
    try:
        li = lead_list.index(int(lead_steps))
    except Exception as e:
        raise KeyError(f"lead_steps={lead_steps} 不在 sample.lead_times={lead_list}") from e
    return sample["y"][li]


def _denorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std + mean


def _predict_one(
    lit: UrbanPiDiTLitModule,
    x_ctx: torch.Tensor,
    *,
    lead_steps: int,
    cfg: Dict,
    seed: int,
) -> torch.Tensor:
    diff_cfg = dict(cfg.get("diffusion", {}) or {})
    steps = int(diff_cfg.get("sample_steps", 8))
    t_start = float(diff_cfg.get("t_start", 0.0))
    t_end = float(diff_cfg.get("t_end", 1.0))

    lt_cond = None
    if getattr(lit, "use_lead_time_conditioning", False):
        lt_steps_t = torch.full((x_ctx.size(0),), float(int(lead_steps)), device=x_ctx.device, dtype=torch.float32)
        lt_cond = lit._normalize_lead_time(lt_steps_t)

    return lit.predict_one(x_ctx, steps=steps, t_start=t_start, t_end=t_end, seed=int(seed), lead_time=lt_cond)


def _choose_indices(ds, *, n_samples: int, mode: str, seed: int, sample_idx: int) -> List[int]:
    mode = str(mode).lower()
    n = len(ds)
    if mode == "fixed":
        return [int(sample_idx)]
    if mode != "random":
        raise ValueError("sample_mode 仅支持 random/fixed")
    m = min(int(n_samples), n)
    rng = np.random.default_rng(int(seed))
    if m <= 0:
        return []
    if m == n:
        return list(range(n))
    return list(rng.choice(n, size=m, replace=False).tolist())


def _radial_bins(H: int, W: int) -> Tuple[np.ndarray, int]:
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rbin = np.rint(r).astype(np.int32)
    rmax = int(rbin.max())
    return rbin, rmax


def _radial_psd2d(x: np.ndarray, *, rbin: np.ndarray, rmax: int, normalize: bool) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - float(np.mean(x))
    F = np.fft.fftshift(np.fft.fft2(x))
    P = np.abs(F) ** 2

    out = np.zeros((rmax + 1,), dtype=np.float64)
    cnt = np.zeros((rmax + 1,), dtype=np.float64)
    for k in range(rmax + 1):
        m = rbin == k
        if not np.any(m):
            continue
        out[k] = float(P[m].mean())
        cnt[k] = float(np.sum(m))

    if normalize:
        s = float(out.sum())
        if s > 0:
            out = out / s
    return out


def _default_bands(rmax: int) -> List[Tuple[str, Tuple[int, int]]]:
    bands = [("low", (0, 1)), ("mid", (2, 3)), ("high", (4, rmax))]
    out = []
    for name, (a, b) in bands:
        if a > rmax:
            continue
        out.append((name, (a, min(b, rmax))))
    return out


def _band_energy(psd: np.ndarray, *, band: Tuple[int, int]) -> float:
    a, b = int(band[0]), int(band[1])
    a = max(a, 0)
    b = min(b, psd.shape[0] - 1)
    if a > b:
        return float("nan")
    return float(np.sum(psd[a : b + 1]))


def _accumulate_radial_psd(
    ds,
    idxs: Sequence[int],
    *,
    lit_a: UrbanPiDiTLitModule,
    lit_b: Optional[UrbanPiDiTLitModule],
    device: torch.device,
    cfg: Dict,
    seed: int,
    vars_: Sequence[str],
    leads: Sequence[int],
    normalize_psd: bool,
) -> Tuple[List[str], List[int], np.ndarray, np.ndarray, Optional[np.ndarray], int]:
    dyn_vars = list(cfg.get("dynamic_vars", []))
    var_indices = [dyn_vars.index(v) for v in vars_]

    sample0 = ds[int(idxs[0])]
    gt0 = _pick_gt(sample0, lead_steps=int(leads[0]))
    H, W = int(gt0.shape[-2]), int(gt0.shape[-1])
    rbin, rmax = _radial_bins(H, W)
    nb = rmax + 1

    gt_sum = np.zeros((len(vars_), len(leads), nb), dtype=np.float64)
    a_sum = np.zeros((len(vars_), len(leads), nb), dtype=np.float64)
    b_sum = np.zeros((len(vars_), len(leads), nb), dtype=np.float64) if lit_b is not None else None
    count = 0

    for idx in idxs:
        sample = ds[int(idx)]
        x_ctx = sample["x_ctx"].unsqueeze(0).to(device)
        norm = sample["norm"]
        mean = norm["mean"].unsqueeze(0).to(device)
        std = norm["std"].unsqueeze(0).to(device)

        for li, lead_steps in enumerate(leads):
            gt_norm = _pick_gt(sample, lead_steps=int(lead_steps)).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_a_norm = _predict_one(lit_a, x_ctx, lead_steps=int(lead_steps), cfg=cfg, seed=seed)
                pred_b_norm = None
                if lit_b is not None:
                    pred_b_norm = _predict_one(lit_b, x_ctx, lead_steps=int(lead_steps), cfg=cfg, seed=seed)

            gt = _denorm(gt_norm, mean, std)[0, var_indices].detach().cpu().numpy()
            pa = _denorm(pred_a_norm, mean, std)[0, var_indices].detach().cpu().numpy()
            pb = _denorm(pred_b_norm, mean, std)[0, var_indices].detach().cpu().numpy() if pred_b_norm is not None else None

            for vi in range(len(vars_)):
                gt_sum[vi, li] += _radial_psd2d(gt[vi], rbin=rbin, rmax=rmax, normalize=normalize_psd)
                a_sum[vi, li] += _radial_psd2d(pa[vi], rbin=rbin, rmax=rmax, normalize=normalize_psd)
                if b_sum is not None and pb is not None:
                    b_sum[vi, li] += _radial_psd2d(pb[vi], rbin=rbin, rmax=rmax, normalize=normalize_psd)

        count += 1

    if count > 0:
        gt_sum /= float(count)
        a_sum /= float(count)
        if b_sum is not None:
            b_sum /= float(count)

    return list(vars_), list(leads), gt_sum, a_sum, b_sum, rmax


def _plot_radial_psd_grid(
    *,
    out_path: str,
    title: str,
    vars_: Sequence[str],
    lead_labels: Sequence[str],
    gt_psd: np.ndarray,
    a_psd: np.ndarray,
    b_psd: Optional[np.ndarray],
    label_a: str,
    label_b: str,
    rmax: int,
):
    nrows = len(vars_)
    ncols = len(lead_labels)
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(9.0, 2.8 * ncols), max(3.0, 2.0 * nrows)), squeeze=False)
    xs = np.arange(rmax + 1)

    for i, v in enumerate(vars_):
        for j, lead_lab in enumerate(lead_labels):
            ax = axes[i][j]
            ax.plot(xs, gt_psd[i, j], label="GT", linewidth=1.5)
            ax.plot(xs, a_psd[i, j], label=label_a, linewidth=1.5)
            if b_psd is not None:
                ax.plot(xs, b_psd[i, j], label=label_b, linewidth=1.5)
            if i == 0:
                ax.set_title(str(lead_lab))
            if j == 0:
                ax.set_ylabel(str(v))
            if i == nrows - 1:
                ax.set_xlabel("Radial Bin")
            ax.grid(True, linestyle="--", alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(2, len(labels)), frameon=False)
    fig.suptitle(title, y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_band_error(
    *,
    out_path: str,
    title: str,
    vars_: Sequence[str],
    lead_labels: Sequence[str],
    gt_psd: np.ndarray,
    a_psd: np.ndarray,
    b_psd: Optional[np.ndarray],
    label_a: str,
    label_b: str,
    bands: List[Tuple[str, Tuple[int, int]]],
):
    ncols = len(lead_labels)
    fig, axes = plt.subplots(1, ncols, figsize=(max(8.0, 3.2 * ncols), 4.0), squeeze=False)

    for j, lead_lab in enumerate(lead_labels):
        ax = axes[0][j]
        xs = np.arange(len(bands))
        width = 0.35 if b_psd is not None else 0.5

        gt_e = np.array([_band_energy(gt_psd[:, j].mean(axis=0), band=b) for _, b in bands], dtype=np.float64)
        a_e = np.array([_band_energy(a_psd[:, j].mean(axis=0), band=b) for _, b in bands], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            a_err = (a_e - gt_e) / gt_e * 100.0

        ax.bar(xs - width / 2.0 if b_psd is not None else xs, a_err, width, label=label_a)

        if b_psd is not None:
            b_e = np.array([_band_energy(b_psd[:, j].mean(axis=0), band=b) for _, b in bands], dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                b_err = (b_e - gt_e) / gt_e * 100.0
            ax.bar(xs + width / 2.0, b_err, width, label=label_b)

        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels([n for n, _ in bands])
        ax.set_title(str(lead_lab))
        ax.set_ylabel("Band Energy Relative Error (%)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(2, len(labels)), frameon=False)
    fig.suptitle(title, y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_a", type=str, required=True)
    parser.add_argument("--ckpt_b", type=str, default="")
    parser.add_argument("--label_a", type=str, default="A")
    parser.add_argument("--label_b", type=str, default="B")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--vars", type=str, default="t2m,u10,v10,tp,tcc")
    parser.add_argument("--leads", type=str, default="")
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--sample_mode", type=str, default="random")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mode", type=str, default="radial_psd")
    parser.add_argument("--normalize_psd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(args.seed) if args.seed is not None else int(cfg.get("train", {}).get("seed", 42))
    pl.seed_everything(seed, workers=True)

    forecast_cfg = dict(cfg.get("forecast", {}) or {})
    time_step_hours = float(forecast_cfg.get("time_step_hours", 6.0))

    dyn_vars = list(cfg.get("dynamic_vars", []))
    vars_in = _parse_csv_str(args.vars)
    vars_ = [v for v in vars_in if v in dyn_vars] if vars_in else dyn_vars
    if not vars_:
        raise ValueError("vars 为空或不在 dynamic_vars 中")

    if str(args.leads).strip():
        leads = _parse_csv_ints(args.leads)
    else:
        leads = list(forecast_cfg.get("eval_lead_times", forecast_cfg.get("lead_times", [cfg.get("delta_t", 1)])))
    leads = [int(x) for x in leads]
    leads = sorted(list(dict.fromkeys(leads)))

    dm = _build_dm(cfg, batch_size=1, num_workers=0)
    split = str(args.split).lower()
    if split == "train":
        ds = dm.train_ds
    elif split == "val":
        ds = dm.val_ds
    else:
        ds = dm.test_ds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_a = _build_lit(cfg, str(args.ckpt_a)).to(device)
    lit_a.eval()
    lit_b = None
    if str(args.ckpt_b).strip():
        lit_b = _build_lit(cfg, str(args.ckpt_b)).to(device)
        lit_b.eval()

    idxs = _choose_indices(ds, n_samples=int(args.n_samples), mode=str(args.sample_mode), seed=seed, sample_idx=int(args.sample_idx))
    if not idxs:
        raise ValueError("未能选到样本")

    vars_, leads, gt_psd, a_psd, b_psd, rmax = _accumulate_radial_psd(
        ds,
        idxs,
        lit_a=lit_a,
        lit_b=lit_b,
        device=device,
        cfg=cfg,
        seed=seed,
        vars_=vars_,
        leads=leads,
        normalize_psd=bool(args.normalize_psd),
    )

    mode = str(args.mode).lower()
    lead_tags = ",".join([_lead_tag(time_step_hours, l) for l in leads])
    title = f"Spectrum Diagnostics | vars={len(vars_)} leads={lead_tags} | n={len(idxs)}"
    lead_labels = [_lead_tag(time_step_hours, l) for l in leads]

    if mode == "radial_psd":
        _plot_radial_psd_grid(
            out_path=str(args.out_path),
            title=title,
            vars_=vars_,
            lead_labels=lead_labels,
            gt_psd=gt_psd,
            a_psd=a_psd,
            b_psd=b_psd,
            label_a=str(args.label_a),
            label_b=str(args.label_b),
            rmax=rmax,
        )
    elif mode == "band_error":
        bands = _default_bands(rmax)
        _plot_band_error(
            out_path=str(args.out_path),
            title=title,
            vars_=vars_,
            lead_labels=lead_labels,
            gt_psd=gt_psd,
            a_psd=a_psd,
            b_psd=b_psd,
            label_a=str(args.label_a),
            label_b=str(args.label_b),
            bands=bands,
        )
    else:
        raise ValueError("mode 仅支持 radial_psd/band_error")

    print(f"Saved figure to: {args.out_path}")


if __name__ == "__main__":
    main()
