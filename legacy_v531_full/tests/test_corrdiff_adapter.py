from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

# Keep tiny CorrDiff CPU tests deterministic and avoid backend thread oversubscription stalls.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from baselines.forecast_models import available_baselines, build_forecast_baseline
from baselines.forecast_runner import evaluate_forecast_baseline


DYNAMIC_VARS = ["d2m", "sp", "t2m", "tcc", "tp", "u10", "v10"]
STATIC_VARS = ["landcover", "building_surface", "buildings", "building_volume", "population"]


def _dummy_batch(batch_size=2, c=7, k=4, s=5, h=8, w=8, leads=(1, 2, 3)):
    x_ctx = torch.randn(batch_size, c * k + s, h, w)
    static_raw = x_ctx[:, c * k :]
    return {
        "x_ctx": x_ctx,
        "static_raw": static_raw,
        "static_cont": static_raw[:, 1:],
        "static_cat": static_raw[:, :1].round().long().clamp(min=0, max=19),
        "y": torch.randn(batch_size, len(leads), c, h, w),
        "x0": torch.randn(batch_size, c, h, w),
        "lead_times": torch.tensor(leads, dtype=torch.long),
        "hour_of_day": torch.zeros(batch_size, dtype=torch.long),
        "norm": {
            "mean": torch.zeros(c, 1, 1),
            "std": torch.ones(c, 1, 1),
        },
        "clim": torch.zeros(batch_size, c, h, w),
    }


class _TinyForecastDataset(Dataset):
    def __init__(self, n=3, leads=(1, 2)):
        self.n = n
        self.leads = tuple(leads)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        batch = _dummy_batch(batch_size=1, leads=self.leads)
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v[0]
            elif isinstance(v, dict):
                out[k] = {kk: vv for kk, vv in v.items()}
            else:
                out[k] = v
        return out


def _build_corrdiff(**extra):
    params = {
        "hidden_channels": 16,
        "mean_depth": 1,
        "correction_steps": 1,
        "include_coords": True,
        "include_hour": True,
        "eval_ensemble_size": 3,
        "return_ensemble_eval": True,
        "noise_scale": 0.05,
    }
    params.update(extra)
    return build_forecast_baseline(
        "corrdiff",
        dynamic_vars=DYNAMIC_VARS,
        static_vars=STATIC_VARS,
        k=4,
        lead_times=[1, 2, 3],
        static_policy="same_static",
        params=params,
    )


def test_corrdiff_is_registered():
    assert "corrdiff" in set(available_baselines())


@torch.no_grad()
def test_corrdiff_train_forward_and_eval_ensemble_shapes():
    batch = _dummy_batch(leads=(1, 2, 3))
    model = _build_corrdiff()
    model.train()
    pred_train = model(batch, lead_times=[1, 2, 3])
    assert pred_train.shape == (2, 3, 7, 8, 8)
    assert torch.isfinite(pred_train).all()

    pred_eval = model.predict(batch, lead_times=[1, 2, 3])
    assert pred_eval.shape == (2, 3, 3, 7, 8, 8)  # [B,E,L,C,H,W]
    assert torch.isfinite(pred_eval).all()


@torch.no_grad()
def test_corrdiff_static_information_protocol_variants_run():
    batch = _dummy_batch(leads=(1, 2))
    for policy in ["same_static", "dynamic_only", "static_zero", "static_shuffle"]:
        model = build_forecast_baseline(
            "corrdiff",
            dynamic_vars=DYNAMIC_VARS,
            static_vars=STATIC_VARS,
            k=4,
            lead_times=[1, 2],
            static_policy=policy,
            params={
                "hidden_channels": 16,
                "mean_depth": 1,
                "correction_steps": 1,
                "include_coords": True,
                "include_hour": False,
                "eval_ensemble_size": 2,
            },
        )
        out = model.predict(batch, lead_times=[1, 2])
        assert out.shape == (2, 2, 2, 7, 8, 8), policy


@torch.no_grad()
def test_forecast_runner_accepts_corrdiff_ensemble_outputs():
    model = build_forecast_baseline(
        "corrdiff",
        dynamic_vars=DYNAMIC_VARS,
        static_vars=STATIC_VARS,
        k=4,
        lead_times=[1, 2],
        static_policy="same_static",
        params={
            "hidden_channels": 16,
            "mean_depth": 1,
            "correction_steps": 1,
            "include_coords": True,
            "include_hour": True,
            "eval_ensemble_size": 3,
            "return_ensemble_eval": True,
        },
    )
    dl = DataLoader(_TinyForecastDataset(n=2, leads=(1, 2)), batch_size=1)
    results = evaluate_forecast_baseline(
        model,
        dl,
        lead_times=[1, 2],
        time_step_hours=6,
        var_names=DYNAMIC_VARS,
        progress_desc="test corrdiff",
    )
    assert "6h" in results and "12h" in results
    lead_results = {key: value for key, value in results.items() if not key.startswith("__")}
    for lead_item in lead_results.values():
        assert {"RMSE", "MAE", "Bias", "CRPS", "ACC"}.issubset(lead_item)
        assert set(lead_item["CRPS_per_var"]) == set(DYNAMIC_VARS)