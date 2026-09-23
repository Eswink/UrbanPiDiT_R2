r"""
CorrDiff 论文结构的 8x8 城市域适配版本。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from baselines.faithful.common import (
    FourierScalarEmbedding,
    edm_preconditioning,
    edm_weighted_mse,
    karras_schedule,
    sample_log_normal_sigma,
)
from baselines.faithful.corrdiff.unet import SongUNetSmall
from baselines.forecast_base import ForecastBatchView, ForecastModelBase, ForecastModelSpec, TensorDict


class FaithfulCorrDiff(ForecastModelBase):
    r"""
    适配 UrbanPiDiT 数据集的 CorrDiff 两网络基线。

    Parameters
    ----
    dynamic_vars : Sequence[str]
        动态变量名。
    static_vars : Sequence[str], optional
        静态变量名。
    k : int, optional, default=1
        历史帧数量。
    lead_times : Sequence[int], optional
        提前期步数。
    static_policy : str, optional, default="same_static"
        静态信息策略。
    hidden_channels : int, optional, default=128
        UNet 基础通道数。
    cond_channels : int, optional, default=128
        条件向量维度。
    channel_mult : Sequence[int], optional, default=(1, 2)
        UNet 通道倍率。
    dropout : float, optional, default=0.0
        dropout 概率。
    use_attention : bool, optional, default=True
        是否在 UNet 瓶颈层启用空间自注意力。
    num_attention_heads : int, optional, default=4
        UNet 空间自注意力头数量。
    training_stage : str, optional, default="regression"
        ``regression`` 或 ``diffusion``。
    regression_checkpoint : str, optional
        diffusion 阶段加载的 regression checkpoint。
    freeze_regression : bool, optional, default=True
        diffusion 阶段是否冻结 regression UNet。
    num_samples : int, optional, default=1
        评估 ensemble 数量。
    """

    spec = ForecastModelSpec(
        name="faithful_corrdiff",
        description="Faithful CorrDiff-style regression UNet plus EDM residual diffusion baseline",
        uses_static=True,
        trainable=True,
        family="faithful_weather_baseline",
    )

    def __init__(
        self,
        *,
        dynamic_vars: Sequence[str],
        static_vars: Sequence[str] | None = None,
        k: int = 1,
        lead_times: Sequence[int] | None = None,
        static_policy: str = "same_static",
        hidden_channels: int = 128,
        cond_channels: int = 128,
        channel_mult: Sequence[int] = (1, 2),
        dropout: float = 0.0,
        use_attention: bool = True,
        num_attention_heads: int = 4,
        sigma_min: float = 0.002,
        sigma_max: float = 0.8,
        sigma_data: float = 0.5,
        sigma_sample_mean: float = -1.2,
        sigma_sample_std: float = 1.2,
        rho: float = 7.0,
        num_noise_levels: int = 18,
        num_samples: int = 1,
        training_stage: str = "regression",
        regression_checkpoint: str | None = None,
        freeze_regression: bool = True,
        residual: bool = True,
        forecast_protocol: str = "direct",
        one_step_lead: int = 1,
        time_step_hours: float = 6.0,
    ) -> None:
        super().__init__(
            dynamic_vars=dynamic_vars,
            static_vars=static_vars,
            k=k,
            lead_times=lead_times,
            static_policy=static_policy,
            forecast_protocol=forecast_protocol,
            one_step_lead=one_step_lead,
            time_step_hours=time_step_hours,
        )
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_data = float(sigma_data)
        self.sigma_sample_mean = float(sigma_sample_mean)
        self.sigma_sample_std = float(sigma_sample_std)
        self.rho = float(rho)
        self.num_noise_levels = int(num_noise_levels)
        self.num_samples = max(1, int(num_samples))
        self.training_stage = str(training_stage).lower()
        self.residual = bool(residual)
        if self.training_stage not in {"regression", "diffusion"}:
            raise ValueError("training_stage must be 'regression' or 'diffusion'")
        cond_input = self.feature_channels + 1
        self.condition_embedding = FourierScalarEmbedding(cond_channels, num_frequencies=16, hidden_dim=cond_channels)
        self.condition_proj = torch.nn.Sequential(
            torch.nn.Conv2d(cond_input, cond_channels, kernel_size=1),
            torch.nn.SiLU(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.regression_unet = SongUNetSmall(
            in_channels=cond_input,
            out_channels=self.out_channels,
            hidden_channels=hidden_channels,
            cond_channels=cond_channels,
            channel_mult=tuple(int(x) for x in channel_mult),
            dropout=dropout,
            use_attention=use_attention,
            num_attention_heads=num_attention_heads,
        )
        diffusion_input = cond_input + self.out_channels * 2 + 1
        self.diffusion_unet = SongUNetSmall(
            in_channels=diffusion_input,
            out_channels=self.out_channels,
            hidden_channels=hidden_channels,
            cond_channels=cond_channels,
            channel_mult=tuple(int(x) for x in channel_mult),
            dropout=dropout,
            use_attention=use_attention,
            num_attention_heads=num_attention_heads,
        )
        self._regression_ckpt_path: Optional[Path] = Path(regression_checkpoint) if regression_checkpoint else None
        self._regression_loaded: bool = self._regression_ckpt_path is None
        if self._regression_ckpt_path is not None and self._regression_ckpt_path.exists():
            self.load_regression_checkpoint(self._regression_ckpt_path)
        if self.training_stage == "diffusion" and freeze_regression:
            for param in self.regression_unet.parameters():
                param.requires_grad_(False)

    def load_regression_checkpoint(self, path: str | Path) -> None:
        r"""
        加载 regression UNet 权重。

        Parameters
        ----
        path : str or Path
            checkpoint 路径。

        Returns
        ----
        None
            原地加载权重。
        """

        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch < 2.6
            state = torch.load(path, map_location="cpu")
        raw = state.get("model_state_dict", state) if isinstance(state, dict) else state
        prefix = "regression_unet."
        reg_state = {key[len(prefix) :]: value for key, value in raw.items() if key.startswith(prefix)}
        self.regression_unet.load_state_dict(reg_state or raw, strict=False)
        self._regression_loaded = True

    def _ensure_regression_loaded(self) -> None:
        r"""
        延迟加载 regression checkpoint（如果尚未加载且路径存在）。

        在 diffusion stage 的训练/推理首次调用时触发。
        """
        if self._regression_loaded:
            return
        if self._regression_ckpt_path is None:
            self._regression_loaded = True
            return
        if not self._regression_ckpt_path.exists():
            raise FileNotFoundError(
                f"Regression checkpoint not found: {self._regression_ckpt_path}. "
                f"Train the regression stage first before running the diffusion stage."
            )
        self.load_regression_checkpoint(self._regression_ckpt_path)
        if self.training_stage == "diffusion":
            for param in self.regression_unet.parameters():
                param.requires_grad_(False)

    def _lead_map(self, view: ForecastBatchView, lead_step: int | torch.Tensor) -> torch.Tensor:
        bsz, _, height, width = view.last.shape
        if isinstance(lead_step, torch.Tensor):
            lead_value = float(lead_step.detach().flatten()[0].cpu())
        else:
            lead_value = float(lead_step)
        denom = float(max(self.lead_times) if self.lead_times else max(int(lead_value), 1))
        return torch.full((bsz, 1, height, width), lead_value / max(denom, 1.0), device=view.last.device, dtype=view.last.dtype)

    def _condition_field(self, view: ForecastBatchView, lead_step: int | torch.Tensor) -> torch.Tensor:
        return torch.cat([self.make_features(view), self._lead_map(view, lead_step)], dim=1)

    def _condition_vector(self, cond_field: torch.Tensor, lead_step: int | torch.Tensor) -> torch.Tensor:
        pooled = self.condition_proj(cond_field).flatten(1)
        if isinstance(lead_step, torch.Tensor):
            lead = lead_step.to(device=cond_field.device, dtype=cond_field.dtype).flatten()
            lead = lead.expand(cond_field.shape[0]) if lead.numel() == 1 else lead
        else:
            lead = torch.full((cond_field.shape[0],), float(lead_step), device=cond_field.device, dtype=cond_field.dtype)
        return pooled + self.condition_embedding(lead)

    def _regression_mean(self, view: ForecastBatchView, lead_step: int | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cond_field = self._condition_field(view, lead_step)
        cond_vec = self._condition_vector(cond_field, lead_step)
        delta = self.regression_unet(cond_field, cond_vec)
        mean = view.last + delta if self.residual else delta
        return mean, cond_field, cond_vec

    def _select_target(self, batch: TensorDict, view: ForecastBatchView, lead_step: int) -> torch.Tensor:
        target = batch.get("y", batch.get("x0", None))
        if target is None:
            raise KeyError("FaithfulCorrDiff.training_loss requires batch['y'] or batch['x0']")
        if not isinstance(target, torch.Tensor):
            target = torch.as_tensor(target)
        target = target.to(device=view.last.device, dtype=view.last.dtype)
        if target.ndim == 4:
            return target
        if target.ndim != 5:
            raise ValueError(f"Expected target 4D/5D, got {tuple(target.shape)}")
        raw_leads = batch.get("lead_times", None)
        if isinstance(raw_leads, torch.Tensor):
            leads = [int(x) for x in raw_leads.detach().cpu().flatten().tolist()]
        elif raw_leads is not None:
            leads = [int(x) for x in list(raw_leads)]
        else:
            leads = [int(x) for x in view.lead_steps.detach().cpu().flatten().tolist()]
        idx = leads.index(int(lead_step)) if int(lead_step) in leads else 0
        return target[:, min(idx, target.shape[1] - 1)]

    def _diffusion_denoise(
        self,
        noisy_residual: torch.Tensor,
        sigma: torch.Tensor,
        cond_field: torch.Tensor,
        cond_vec: torch.Tensor,
        mean_forecast: torch.Tensor,
    ) -> torch.Tensor:
        coeffs = edm_preconditioning(sigma, noisy_residual, sigma_data=self.sigma_data)
        sigma_map = coeffs.c_noise.view(noisy_residual.shape[0], 1, 1, 1).expand(
            -1,
            1,
            noisy_residual.shape[-2],
            noisy_residual.shape[-1],
        )
        network_input = torch.cat([cond_field, mean_forecast, coeffs.c_in * noisy_residual, sigma_map], dim=1)
        raw = self.diffusion_unet(network_input, cond_vec)
        return coeffs.c_skip * noisy_residual + coeffs.c_out * raw

    def training_loss(self, batch: TensorDict, lead_times: Sequence[int] | None = None) -> torch.Tensor:
        r"""
        根据 ``training_stage`` 计算 regression 或 diffusion 阶段损失。

        Parameters
        ----
        batch : TensorDict
            UrbanPiDiT loader batch。
        lead_times : Sequence[int], optional
            训练提前期。

        Returns
        ----
        torch.Tensor
            标量损失。
        """

        self._ensure_regression_loaded()
        requested = [int(x) for x in list(lead_times or [self.one_step_lead])]
        view = self.prepare_batch(batch, lead_times=requested)
        losses = []
        for lead in requested:
            target = self._select_target(batch, view, lead)
            mean_forecast, cond_field, cond_vec = self._regression_mean(view, lead)
            if self.training_stage == "regression":
                losses.append(F.mse_loss(mean_forecast, target))
                continue
            clean_residual = target - mean_forecast.detach()
            sigma = sample_log_normal_sigma(
                clean_residual.shape[0],
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                mean=self.sigma_sample_mean,
                std=self.sigma_sample_std,
                device=clean_residual.device,
                dtype=clean_residual.dtype,
            )
            noisy = clean_residual + torch.randn_like(clean_residual) * sigma.view(clean_residual.shape[0], 1, 1, 1)
            denoised = self._diffusion_denoise(noisy, sigma, cond_field, cond_vec, mean_forecast.detach())
            losses.append(edm_weighted_mse(denoised, clean_residual, sigma, sigma_data=self.sigma_data))
        return torch.stack(losses).mean()

    def _sample_residual(self, cond_field: torch.Tensor, cond_vec: torch.Tensor, mean_forecast: torch.Tensor) -> torch.Tensor:
        schedule = karras_schedule(
            self.num_noise_levels,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            device=mean_forecast.device,
            dtype=mean_forecast.dtype,
            append_zero=True,
        )
        current = torch.randn_like(mean_forecast) * schedule[0]
        for idx in range(schedule.numel() - 1):
            sigma_now = schedule[idx]
            sigma = sigma_now.expand(current.shape[0])
            next_sigma = schedule[idx + 1]
            denoised = self._diffusion_denoise(current, sigma, cond_field, cond_vec, mean_forecast)
            derivative = (current - denoised) / sigma.view(current.shape[0], 1, 1, 1).clamp_min(1e-12)
            proposal = current + (next_sigma - sigma_now) * derivative
            if next_sigma > 0:
                next_vec = next_sigma.expand(current.shape[0])
                denoised_next = self._diffusion_denoise(proposal, next_vec, cond_field, cond_vec, mean_forecast)
                derivative_next = (proposal - denoised_next) / next_vec.view(current.shape[0], 1, 1, 1).clamp_min(1e-12)
                current = current + (next_sigma - sigma_now) * 0.5 * (derivative + derivative_next)
            else:
                current = proposal
        return current

    def forecast_lead(self, view: ForecastBatchView, lead_step: torch.Tensor) -> torch.Tensor:
        r"""
        预测单个提前期。

        Parameters
        ----
        view : ForecastBatchView
            规范化 batch 视图。
        lead_step : torch.Tensor
            提前期步数。

        Returns
        ----
        torch.Tensor
            预测场，shape :math:`(B, C, H, W)`。
        """

        self._ensure_regression_loaded()
        mean_forecast, cond_field, cond_vec = self._regression_mean(view, lead_step)
        if self.training_stage == "regression" or self.num_noise_levels <= 0:
            return mean_forecast
        samples = [mean_forecast + self._sample_residual(cond_field, cond_vec, mean_forecast) for _ in range(self.num_samples)]
        return torch.stack(samples, dim=0).mean(dim=0)

    @torch.no_grad()
    def predict(self, batch: TensorDict, lead_times: Sequence[int] | None = None) -> torch.Tensor:
        self._ensure_regression_loaded()
        self.eval()
        view = self.prepare_batch(batch, lead_times=lead_times)
        if self.training_stage == "diffusion" and self.num_samples > 1:
            ensemble = []
            for _ in range(self.num_samples):
                outputs = []
                for lead in view.lead_steps:
                    mean_forecast, cond_field, cond_vec = self._regression_mean(view, lead)
                    outputs.append(mean_forecast + self._sample_residual(cond_field, cond_vec, mean_forecast))
                ensemble.append(torch.stack(outputs, dim=1))
            return torch.stack(ensemble, dim=1)
        outputs = [self.forecast_lead(view, lead) for lead in view.lead_steps]
        return torch.stack(outputs, dim=1)