from __future__ import annotations

import inspect

import pytest
import torch

from losses import (
    FeasibilityProxyLoss,
    PhysicalConsistencyConfig,
    PhysicalConsistencyLoss,
    ProcessConsistencyLoss,
    StructureProxyLoss,
)


def _calculator(config: PhysicalConsistencyConfig) -> PhysicalConsistencyLoss:
    weights = torch.ones(1, 7, 1, 1)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    return PhysicalConsistencyLoss(
        config,
        fft_channel_weights=weights,
        grad_channel_weights=weights,
        sobel_x=sobel_x,
        sobel_y=sobel_y,
    )


def _physical_field(batch: int = 1, height: int = 4, width: int = 4) -> torch.Tensor:
    pred = torch.zeros(batch, 7, height, width)
    pred[:, 0] = 10.0
    pred[:, 1] = 100000.0
    pred[:, 2] = 20.0
    pred[:, 3] = 0.5
    pred[:, 4] = 1.0
    pred[:, 5] = 2.0
    pred[:, 6] = 1.0
    return pred


def test_facade_keeps_legacy_keys_and_adds_layered_diagnostics() -> None:
    pred = _physical_field()
    target = pred.clone()
    config = PhysicalConsistencyConfig(
        use_process_consistency=True,
        lambda_fft=0.0,
        lambda_grad=0.0,
        lambda_process_rh=0.2,
        lambda_process_drag=0.0,
        lambda_process_diurnal=0.0,
    )
    out = _calculator(config)(pred, target, std=torch.ones_like(pred), mean=torch.zeros_like(pred))

    for key in (
        "x0_pred_denorm",
        "loss_phys_spatial",
        "loss_feasibility",
        "loss_process_consistency",
        "loss_structure",
        "loss_fft",
        "loss_grad",
        "logs",
        "feasibility",
        "structure",
        "process_consistency",
    ):
        assert key in out

    logs = out["logs"]
    for key in (
        "phys_tp_nonneg",
        "phys_tcc_01",
        "phys_dew_leq_t",
        "phys_sp_nonneg",
        "phys_wind_div",
        "loss/feasibility_total",
        "loss/structure_total",
        "loss/process_total",
        "physical_feasibility_score",
        "structure_score",
        "process_consistency_score",
    ):
        assert key in logs
        assert torch.isfinite(logs[key])

    assert torch.allclose(out["loss_fft"], torch.tensor(0.0))
    assert torch.allclose(out["loss_grad"], torch.tensor(0.0))


def test_legacy_total_stays_equivalent_when_process_terms_disabled() -> None:
    pred = _physical_field()
    pred[:, 0] = 25.0
    pred[:, 2] = 20.0
    pred[:, 3] = 1.5
    pred[:, 4] = -2.0
    pred[:, 5, :, 2:] = 4.0
    target = pred.clone()
    config = PhysicalConsistencyConfig(
        use_process_consistency=False,
        lambda_fft=0.0,
        lambda_grad=0.0,
        w_tp=1.0,
        w_tcc=0.5,
        w_dew=0.5,
        w_sp=0.1,
        w_div=0.1,
    )
    out = _calculator(config)(pred, target, std=torch.ones_like(pred), mean=torch.zeros_like(pred))

    feasibility = FeasibilityProxyLoss(config)(pred)["total"]
    wind_div = StructureProxyLoss.wind_divergence(pred)
    expected = feasibility + config.w_div * wind_div

    assert torch.allclose(out["loss_phys_spatial"], expected)
    assert torch.allclose(out["loss_process_consistency"], config.w_div * wind_div)
    assert torch.allclose(out["process_consistency"]["total"], torch.tensor(0.0))


def test_relative_humidity_proxy_penalizes_supersaturation() -> None:
    good = _physical_field()
    bad = good.clone()
    bad[:, 0] = 25.0
    bad[:, 2] = 20.0

    good_loss = ProcessConsistencyLoss.relative_humidity_consistency(good)
    bad_loss = ProcessConsistencyLoss.relative_humidity_consistency(bad)

    assert torch.allclose(good_loss, torch.tensor(0.0), atol=1e-6)
    assert bad_loss > good_loss


def test_drag_and_diurnal_proxies_use_only_static_and_current_hour() -> None:
    latest = _physical_field(height=2, width=2)
    pred = latest.clone()
    static_cont = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]])

    pred[:, 5] = torch.tensor([[[1.0, 1.0], [5.0, 5.0]]])
    latest[:, 5] = 1.0
    drag, drag_diag = ProcessConsistencyLoss.drag_consistency(pred, latest_state=latest, static_cont=static_cont)
    skipped_drag, skipped_drag_diag = ProcessConsistencyLoss.drag_consistency(pred, static_cont=static_cont)

    assert drag > 0.0
    assert torch.allclose(drag_diag["drag_available"], torch.tensor(1.0))
    assert torch.allclose(skipped_drag, torch.tensor(0.0))
    assert torch.allclose(skipped_drag_diag["drag_available"], torch.tensor(0.0))

    pred[:, 2] = torch.tensor([[[30.0, 30.0], [20.0, 20.0]]])
    latest[:, 2] = 20.0
    skipped, skipped_diag = ProcessConsistencyLoss.diurnal_thermal_response(
        pred,
        latest_state=latest,
        static_cont=static_cont,
        hour_of_day=None,
    )
    active, active_diag = ProcessConsistencyLoss.diurnal_thermal_response(
        pred,
        latest_state=latest,
        static_cont=static_cont,
        hour_of_day=torch.tensor([12.0]),
    )

    assert torch.allclose(skipped, torch.tensor(0.0))
    assert torch.allclose(skipped_diag["diurnal_available"], torch.tensor(0.0))
    assert active > 0.0
    assert torch.allclose(active_diag["diurnal_available"], torch.tensor(1.0))


def test_daytime_and_nighttime_diurnal_gaps_are_reported() -> None:
    latest = _physical_field(batch=2, height=2, width=2)
    pred = latest.clone()
    static_cont = torch.tensor(
        [
            [[[0.0, 0.0], [1.0, 1.0]]],
            [[[0.0, 0.0], [1.0, 1.0]]],
        ]
    )
    pred[:, 2] = torch.tensor(
        [
            [[20.0, 20.0], [30.0, 30.0]],
            [[20.0, 20.0], [30.0, 30.0]],
        ]
    )
    latest[:, 2] = 20.0

    loss, diag = ProcessConsistencyLoss.diurnal_thermal_response(
        pred,
        latest_state=latest,
        static_cont=static_cont,
        hour_of_day=torch.tensor([12.0, 22.0]),
    )

    assert torch.isfinite(loss)
    assert torch.allclose(diag["diurnal_available"], torch.tensor(1.0))
    assert "daytime_heat_response_gap" in diag
    assert "nighttime_heat_release_gap" in diag


def test_process_consistency_interface_has_no_future_inputs() -> None:
    facade_params = set(inspect.signature(PhysicalConsistencyLoss.__call__).parameters)
    process_params = set(inspect.signature(ProcessConsistencyLoss.__call__).parameters)
    forbidden = {"future", "y_future", "x0_future", "future_target"}

    assert forbidden.isdisjoint(facade_params)
    assert forbidden.isdisjoint(process_params)


def test_dtype_and_device_are_preserved_for_diagnostics() -> None:
    pred = _physical_field().double()
    target = pred.clone()
    config = PhysicalConsistencyConfig(use_process_consistency=True, lambda_process_rh=0.1)
    out = _calculator(config)(pred, target, std=torch.ones_like(pred), mean=torch.zeros_like(pred))

    assert out["loss_phys_spatial"].dtype == pred.dtype
    assert out["loss_phys_spatial"].device == pred.device
    assert out["process_consistency"]["total"].dtype == pred.dtype


def test_rollout_process_context_ignores_teacher_forced_future_gt(monkeypatch: pytest.MonkeyPatch) -> None:
    pl = pytest.importorskip("pytorch_lightning")
    del pl
    from pidit_lit import UrbanPiDiTLitModule

    model_cfg = {
        "H": 2,
        "W": 2,
        "D": 16,
        "depth": 1,
        "heads": 2,
        "mlp_ratio": 2.0,
        "in_channels": 7,
        "ctx_channels": 14,
        "out_channels": 7,
        "static_channels": 0,
        "dropout": 0.0,
        "drop_path_rate": 0.0,
    }
    optim_cfg = {"lr": 1e-3, "max_epochs": 1, "loss_weights": [1.0] * 7, "use_channel_weights": False}
    diffusion_cfg = {"P_mean": 0.0, "P_std": 0.0}
    physics_cfg = {
        "use_physics_loss": True,
        "use_fft_loss": False,
        "use_gradient_loss": False,
        "use_process_consistency": True,
        "lambda": 0.1,
        "lambda_process_drag": 0.1,
        "lambda_process_diurnal": 0.1,
    }
    forecast_cfg = {
        "eval_lead_times": [1, 2],
        "rollout_finetune": {
            "enabled": True,
            "max_steps": 2,
            "loss_on": "all",
            "teacher_forcing": {"start_ratio": 1.0, "end_ratio": 1.0, "decay_epochs": 0},
        },
        "apply_aux_losses_to_all_leads": True,
    }
    module = UrbanPiDiTLitModule(model_cfg, optim_cfg, diffusion_cfg, physics_cfg, {}, forecast_cfg=forecast_cfg)
    module.trainer = type("TrainerStub", (), {"barebones": True})()
    module._current_fx_name = "training_step"

    pred_step1 = torch.full((1, 7, 2, 2), 1.0)
    pred_step2 = torch.full((1, 7, 2, 2), 2.0)

    class FixedNet(torch.nn.Module):
        in_channels = 7
        out_channels = 7
        real_k = 2

        def __init__(self) -> None:
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(()))
            self.calls = 0
            self.seen_latest = []

        def forward(self, x_t: torch.Tensor, x_ctx: torch.Tensor, t: torch.Tensor, **kwargs: object) -> torch.Tensor:
            del x_t, t, kwargs
            latest_idx = torch.tensor([1, 3, 5, 7, 9, 11, 13], device=x_ctx.device, dtype=torch.long)
            self.seen_latest.append(x_ctx.index_select(1, latest_idx).detach().clone())
            self.calls += 1
            return (pred_step1 if self.calls == 1 else pred_step2).to(self.param.device) + self.param * 0.0

    module.net = FixedNet()
    module.loss_weights = torch.ones(1, 7, 1, 1)
    captured_latest = []
    original = module._calculate_physics_losses

    def capture_physics(x0_pred, x0_gt, std, mean, static_batch=None, x_ctx=None):
        out = module._attach_latest_state(static_batch, x_ctx, mean=mean, std=std)
        captured_latest.append(out["latest_state"].detach().clone())
        return original(x0_pred, x0_gt, std, mean, static_batch, x_ctx=x_ctx)

    monkeypatch.setattr(module, "_calculate_physics_losses", capture_physics)

    x_ctx = torch.zeros(1, 14, 2, 2)
    y_futures = torch.stack(
        [
            torch.full((1, 7, 2, 2), 100.0),
            torch.full((1, 7, 2, 2), 200.0),
        ],
        dim=1,
    )
    static_cont = torch.ones(1, 1, 2, 2)
    static_batch = {"static_raw": static_cont, "static_cont": static_cont, "static_cat": None, "hour_of_day": torch.tensor([12.0])}

    module._training_step_rollout_finetune(
        x_ctx=x_ctx,
        y_futures=y_futures,
        lead_list=[1, 2],
        mean=torch.zeros(1, 7, 1, 1),
        std=torch.ones(1, 7, 1, 1),
        static_batch=static_batch,
    )

    assert len(captured_latest) == 2
    assert torch.allclose(module.net.seen_latest[0], torch.zeros_like(module.net.seen_latest[0]))
    assert torch.allclose(module.net.seen_latest[1], y_futures[:, 0])
    assert torch.allclose(captured_latest[0], torch.zeros_like(captured_latest[0]))
    assert torch.allclose(captured_latest[1], pred_step1)
    assert not torch.allclose(captured_latest[1], y_futures[:, 0])
