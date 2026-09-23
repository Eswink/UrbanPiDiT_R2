r"""
GenCast 论文结构的 8x8 城市域适配版本。
"""

from __future__ import annotations

from typing import Sequence

import torch

from baselines.faithful.common import edm_weighted_mse, karras_schedule, sample_log_normal_sigma
from baselines.faithful.gencast.denoiser import GenCastDenoiser
from baselines.forecast_base import ForecastBatchView, ForecastModelBase, ForecastModelSpec, TensorDict


class FaithfulGenCast(ForecastModelBase):
    r"""
    适配 UrbanPiDiT 数据集的 GenCast EDM 三图基线。

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
    hidden_channels : int, optional, default=256
        mesh/grid 节点隐藏维度。
    num_layers : int, optional, default=8
        Transformer 层数。
    num_heads : int, optional, default=4
        注意力头数量。
    ffn_hidden : int, optional, default=1024
        FFN 隐层维度。
    dropout : float, optional, default=0.0
        dropout 概率。
    encoder_depth : int, optional, default=2
        grid2mesh GNN 的 MLP 深度。
    decoder_depth : int, optional, default=2
        mesh2grid GNN 的 MLP 深度。
    use_graph_attention : bool, optional, default=True
        是否使用 mesh 拓扑稀疏注意力掩码。
    graph_k : int, optional, default=8
        grid2mesh radius query 缺失 receiver 时的 fallback kNN 数。
    sigma_min : float, optional, default=0.002
        最小噪声水平。
    sigma_max : float, optional, default=0.8
        最大噪声水平。
    sigma_data : float, optional, default=0.5
        EDM 数据标准差。
    sigma_sample_mean : float, optional, default=-1.2
        训练噪声采样 log 均值。
    sigma_sample_std : float, optional, default=1.2
        训练噪声采样 log 标准差。
    rho : float, optional, default=7.0
        Karras schedule 形状参数。
    num_noise_levels : int, optional, default=18
        采样步数。
    num_samples : int, optional, default=1
        评估 ensemble 样本数。
    stochastic_eval : bool, optional, default=False
        评估是否随机采样。
    residual : bool, optional, default=True
        是否在 residual 空间建模。
    """

    spec = ForecastModelSpec(
        name="faithful_gencast",
        description="Faithful GenCast-style EDM transformer denoising baseline",
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
        hidden_channels: int = 256,
        num_layers: int = 8,
        num_heads: int = 4,
        ffn_hidden: int = 1024,
        dropout: float = 0.0,
        encoder_depth: int = 2,
        decoder_depth: int = 2,
        use_graph_attention: bool = True,
        graph_k: int = 8,
        mesh_size: int | None = None,
        mesh_height: int = 4,
        mesh_width: int = 4,
        k_nn: int | None = None,
        radius_query_fraction_edge_length: float = 0.6,
        sigma_min: float = 0.002,
        sigma_max: float = 0.8,
        sigma_data: float = 0.5,
        sigma_sample_mean: float = -1.2,
        sigma_sample_std: float = 1.2,
        rho: float = 7.0,
        num_noise_levels: int = 18,
        num_samples: int = 1,
        stochastic_eval: bool = False,
        residual: bool = True,
        forecast_protocol: str = "official_rollout",
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
        self.stochastic_eval = bool(stochastic_eval)
        self.residual = bool(residual)
        self.denoiser = GenCastDenoiser(
            cond_channels=self.feature_channels + 1,
            out_channels=self.out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_hidden=ffn_hidden,
            dropout=dropout,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            use_graph_attention=use_graph_attention,
            graph_k=graph_k,
            mesh_size=mesh_size,
            mesh_height=mesh_height,
            mesh_width=mesh_width,
            k_nn=k_nn,
            radius_query_fraction_edge_length=radius_query_fraction_edge_length,
            sigma_data=sigma_data,
        )

    def _lead_map(self, view: ForecastBatchView, lead_step: int | torch.Tensor) -> torch.Tensor:
        bsz, _, height, width = view.last.shape
        if isinstance(lead_step, torch.Tensor):
            lead_value = float(lead_step.detach().flatten()[0].cpu())
        else:
            lead_value = float(lead_step)
        denom = float(max(self.lead_times) if self.lead_times else max(int(lead_value), 1))
        value = lead_value / max(denom, 1.0)
        return torch.full((bsz, 1, height, width), value, device=view.last.device, dtype=view.last.dtype)

    def _condition(self, view: ForecastBatchView, lead_step: int | torch.Tensor) -> torch.Tensor:
        return torch.cat([self.make_features(view), self._lead_map(view, lead_step)], dim=1)

    def _to_model_space(self, view: ForecastBatchView, field: torch.Tensor) -> torch.Tensor:
        return field - view.last if self.residual else field

    def _from_model_space(self, view: ForecastBatchView, field: torch.Tensor) -> torch.Tensor:
        return view.last + field if self.residual else field

    def _select_target(self, batch: TensorDict, view: ForecastBatchView, lead_step: int) -> torch.Tensor:
        target = batch.get("y", batch.get("x0", None))
        if target is None:
            raise KeyError("FaithfulGenCast.training_loss requires batch['y'] or batch['x0']")
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

    def training_loss(self, batch: TensorDict, lead_times: Sequence[int] | None = None) -> torch.Tensor:
        r"""
        计算 GenCast EDM 训练损失。

        Parameters
        ----
        batch : TensorDict
            UrbanPiDiT loader batch。
        lead_times : Sequence[int], optional
            训练提前期，默认使用单步。

        Returns
        ----
        torch.Tensor
            标量损失。
        """

        requested = [int(x) for x in list(lead_times or [self.one_step_lead])]
        view = self.prepare_batch(batch, lead_times=requested)
        losses = []
        for lead in requested:
            target = self._select_target(batch, view, lead)
            clean = self._to_model_space(view, target)
            sigma = sample_log_normal_sigma(
                clean.shape[0],
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                mean=self.sigma_sample_mean,
                std=self.sigma_sample_std,
                device=clean.device,
                dtype=clean.dtype,
            )
            sigma_b = sigma.view(clean.shape[0], *([1] * (clean.ndim - 1)))
            noisy = clean + torch.randn_like(clean) * sigma_b
            denoised = self.denoiser(noisy, sigma, self._condition(view, lead))
            losses.append(edm_weighted_mse(denoised, clean, sigma, sigma_data=self.sigma_data))
        return torch.stack(losses).mean()

    def _sample_once(self, view: ForecastBatchView, lead_step: torch.Tensor) -> torch.Tensor:
        cond = self._condition(view, lead_step)
        schedule = karras_schedule(
            self.num_noise_levels,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            device=view.last.device,
            dtype=view.last.dtype,
            append_zero=True,
        )
        current = torch.zeros_like(view.last)
        if self.training or self.stochastic_eval:
            current = current + torch.randn_like(current) * schedule[0]
        for idx in range(schedule.numel() - 1):
            sigma = schedule[idx].expand(current.shape[0])
            next_sigma = schedule[idx + 1]
            denoised = self.denoiser(current, sigma, cond)
            derivative = (current - denoised) / sigma.view(current.shape[0], 1, 1, 1).clamp_min(1e-12)
            proposal = current + (next_sigma - schedule[idx]) * derivative
            if next_sigma > 0:
                next_vec = next_sigma.expand(current.shape[0])
                denoised_next = self.denoiser(proposal, next_vec, cond)
                derivative_next = (proposal - denoised_next) / next_sigma.view(1, 1, 1, 1).clamp_min(1e-12)
                current = current + (next_sigma - schedule[idx]) * 0.5 * (derivative + derivative_next)
            else:
                current = proposal
        return self._from_model_space(view, current)

    def forecast_lead(self, view: ForecastBatchView, lead_step: torch.Tensor) -> torch.Tensor:
        r"""
        预测单个提前期的 ensemble mean。

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

        samples = [self._sample_once(view, lead_step) for _ in range(self.num_samples if self.stochastic_eval else 1)]
        return torch.stack(samples, dim=0).mean(dim=0)

    def _sample_direct(self, view: ForecastBatchView) -> torch.Tensor:
        outputs = [self._sample_once(view, lead) for lead in view.lead_steps]
        return torch.stack(outputs, dim=1)

    def _sample_rollout(self, view: ForecastBatchView) -> torch.Tensor:
        requested = [int(x) for x in view.lead_steps.detach().cpu().tolist()]
        if not requested:
            raise ValueError("lead_steps must not be empty for official_rollout")
        max_lead = max(requested)
        outputs: dict[int, torch.Tensor] = {}
        current = view
        one_step = self._one_step_tensor(view)
        for step in range(1, max_lead + 1):
            pred = self._sample_once(current, one_step)
            current = self._roll_history(current, pred)
            if step in requested:
                outputs[step] = pred
        return torch.stack([outputs[step] for step in requested], dim=1)

    def _sample_forecast(self, view: ForecastBatchView) -> torch.Tensor:
        if self.forecast_protocol == "official_rollout":
            return self._sample_rollout(view)
        return self._sample_direct(view)

    @torch.no_grad()
    def predict(self, batch: TensorDict, lead_times: Sequence[int] | None = None) -> torch.Tensor:
        self.eval()
        view = self.prepare_batch(batch, lead_times=lead_times)
        if self.stochastic_eval and self.num_samples > 1:
            ensemble = [self._sample_forecast(view) for _ in range(self.num_samples)]
            return torch.stack(ensemble, dim=1)
        return self._sample_forecast(view)