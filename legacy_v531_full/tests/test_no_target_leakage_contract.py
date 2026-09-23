"""验证/测试阶段的目标场泄漏契约测试。

该测试用轻量模型替身拦截评估阶段输入，确保目标场只参与指标计算，
不会被送入扩散生成输入或上下文输入。
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
import torch
import torch.nn as nn

import pidit_lit
from utils.diagnostics import collect_model_diagnostics


class SpyDenoiser(nn.Module):
    """记录评估阶段传入模型的扩散输入和上下文。"""

    def __init__(self, out_channels: int = 2, in_channels: int = 2) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.real_k = 2
        self.recorded_x_t: List[torch.Tensor] = []
        self.recorded_x_ctx: List[torch.Tensor] = []
        self.return_diagnostics_flags: List[bool] = []
        self.last_forward_diagnostics: Dict[str, Any] = {}

    def forward(
        self,
        x_t: torch.Tensor,
        x_ctx: torch.Tensor,
        t: torch.Tensor,
        *,
        lead_time: torch.Tensor | None = None,
        static_raw: torch.Tensor | None = None,
        static_cont: torch.Tensor | None = None,
        static_cat: torch.Tensor | None = None,
        hour_of_day: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, Any]]:
        self.recorded_x_t.append(x_t.detach().cpu().clone())
        self.recorded_x_ctx.append(x_ctx.detach().cpu().clone())
        self.return_diagnostics_flags.append(bool(return_diagnostics))
        diagnostics = {
            "leakage/target_tensor_provided": torch.tensor(0.0, device=x_t.device),
            "proxy/roughness_proxy_mean": torch.tensor(1.0, device=x_t.device),
            "model/use_process_proxy_encoder": False,
            "model/name": "not-a-scalar-log",
            "proxy/vector_should_not_log": torch.ones(2, device=x_t.device),
        }
        self.last_forward_diagnostics = diagnostics
        pred = torch.zeros(
            (x_t.size(0), self.out_channels, x_t.size(2), x_t.size(3)),
            dtype=x_t.dtype,
            device=x_t.device,
        )
        if return_diagnostics:
            return pred, diagnostics
        return pred


def _lit_configs() -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    model_cfg = {
        "H": 2,
        "W": 2,
        "in_channels": 2,
        "ctx_channels": 4,
        "out_channels": 2,
        "static_channels": 0,
        "D": 8,
        "depth": 1,
        "heads": 1,
        "use_lead_time_conditioning": True,
    }
    optim_cfg = {
        "lr": 1e-3,
        "max_epochs": 1,
        "weight_decay": 0.0,
        "use_channel_weights": False,
        "loss_weights": [1.0, 1.0],
    }
    diffusion_cfg = {"P_mean": -0.8, "P_std": 0.8, "sample_steps": 1, "t_start": 0.0, "t_end": 1.0}
    physics_cfg = {
        "use_physics_loss": False,
        "use_fft_loss": False,
        "use_gradient_loss": False,
        "lambda": 0.0,
        "lambda_fft": 0.0,
        "lambda_grad": 0.0,
    }
    metrics_cfg = {"var_names": ["a", "b"], "acc": {"enabled": False}}
    inference_cfg = {"mode": "deterministic", "fixed_seed": 7}
    forecast_cfg = {
        "time_step_hours": 6.0,
        "eval_lead_times": [1],
        "max_lead_time_steps": 1,
        "multi_horizon_inference": "direct",
        "correlate_noise_across_leads": True,
    }
    return model_cfg, optim_cfg, diffusion_cfg, physics_cfg, metrics_cfg, inference_cfg, forecast_cfg


def _build_lit_module(monkeypatch: pytest.MonkeyPatch, spy: SpyDenoiser) -> pidit_lit.UrbanPiDiTLitModule:
    monkeypatch.setattr(pidit_lit, "build_model_from_config", lambda cfg: spy)
    lit = pidit_lit.UrbanPiDiTLitModule(*_lit_configs())
    logged: Dict[str, torch.Tensor] = {}

    def fake_log(name: str, value: Any, *args: Any, **kwargs: Any) -> None:
        if isinstance(value, torch.Tensor):
            logged[name] = value.detach().cpu()

    def fake_physics_losses(
        x0_pred: torch.Tensor,
        x0_gt: torch.Tensor,
        std: torch.Tensor,
        mean: torch.Tensor,
        static_batch: Dict[str, torch.Tensor | None] | None = None,
        x_ctx: torch.Tensor | None = None,
    ) -> Dict[str, Any]:
        zero = x0_pred.new_tensor(0.0)
        return {
            "x0_pred_denorm": x0_pred * std + mean,
            "loss_phys_spatial": zero,
            "loss_fft": zero,
            "loss_grad": zero,
            "logs": {},
        }

    monkeypatch.setattr(lit, "log", fake_log)
    monkeypatch.setattr(lit, "_calculate_physics_losses", fake_physics_losses)
    lit.logged_scalars = logged
    return lit


def _eval_batch() -> Dict[str, Any]:
    batch_size, channels, height, width = 2, 2, 2, 2
    x_ctx = torch.full((batch_size, channels * 2, height, width), 0.25)
    x0 = torch.full((batch_size, channels, height, width), 12345.0)
    y = torch.full((batch_size, 1, channels, height, width), 67890.0)
    return {
        "x_ctx": x_ctx,
        "x0": x0,
        "y": y,
        "lead_times": torch.tensor([[1], [1]]),
        "norm": {
            "mean": torch.zeros(1, channels, 1, 1),
            "std": torch.ones(1, channels, 1, 1),
        },
    }


@pytest.mark.parametrize("step_name", ["validation_step", "test_step"])
def test_eval_step_does_not_feed_target_into_model(monkeypatch: pytest.MonkeyPatch, step_name: str) -> None:
    spy = SpyDenoiser()
    lit = _build_lit_module(monkeypatch, spy)
    batch = _eval_batch()

    result = getattr(lit, step_name)(batch, 0)

    assert result
    assert spy.recorded_x_t
    assert spy.recorded_x_ctx
    assert any(spy.return_diagnostics_flags)
    target_values = {12345.0, 67890.0}
    for x_t in spy.recorded_x_t:
        for value in target_values:
            assert not torch.any(torch.isclose(x_t, torch.tensor(value)))
    for x_ctx in spy.recorded_x_ctx:
        assert torch.equal(x_ctx, batch["x_ctx"])
        for value in target_values:
            assert not torch.any(torch.isclose(x_ctx, torch.tensor(value)))


def test_eval_diagnostics_logs_only_scalar_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = SpyDenoiser()
    lit = _build_lit_module(monkeypatch, spy)
    lit.validation_step(_eval_batch(), 0)

    assert "val/diagnostics/leakage/target_tensor_provided" in lit.logged_scalars
    assert "val/diagnostics/proxy/roughness_proxy_mean" in lit.logged_scalars
    assert "val/diagnostics/model/use_process_proxy_encoder" in lit.logged_scalars
    assert "val/diagnostics/model/name" not in lit.logged_scalars
    assert "val/diagnostics/proxy/vector_should_not_log" not in lit.logged_scalars


def test_collect_model_diagnostics_filters_non_scalar_values() -> None:
    logs = collect_model_diagnostics(
        {
            "proxy/scalar_float": 1.5,
            "proxy/scalar_tensor": torch.tensor(2.5),
            "process_graph/finite_bool": True,
            "proxy/vector": torch.tensor([1.0, 2.0]),
            "proxy/list": [1.0, 2.0],
            "model/name": "UrbanPiDiT",
            "other/scalar": 3.0,
        },
        log_prefix="test/diagnostics",
    )

    assert set(logs) == {
        "test/diagnostics/process_graph/finite_bool",
        "test/diagnostics/proxy/scalar_float",
        "test/diagnostics/proxy/scalar_tensor",
    }
    assert float(logs["test/diagnostics/process_graph/finite_bool"]) == 1.0