r"""
EDM 预条件与加权损失。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EDMPrecondOutput:
    r"""
    EDM 预条件系数集合。

    Parameters
    ----
    c_skip : torch.Tensor
        跳连系数，shape 可广播到目标张量。
    c_out : torch.Tensor
        网络输出系数，shape 可广播到目标张量。
    c_in : torch.Tensor
        noisy input 输入缩放系数，shape 可广播到目标张量。
    c_noise : torch.Tensor
        噪声嵌入标量，shape :math:`(B,)`。
    weight : torch.Tensor
        EDM 加权损失系数，shape 可广播到目标张量。
    """

    c_skip: torch.Tensor
    c_out: torch.Tensor
    c_in: torch.Tensor
    c_noise: torch.Tensor
    weight: torch.Tensor


def _sigma_vector(sigma: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(sigma, torch.Tensor):
        sigma = torch.as_tensor(sigma, device=reference.device, dtype=reference.dtype)
    sigma = sigma.to(device=reference.device, dtype=reference.dtype).flatten()
    if sigma.numel() == 1:
        sigma = sigma.expand(reference.shape[0])
    if sigma.numel() != reference.shape[0]:
        raise ValueError(f"Expected sigma batch {reference.shape[0]}, got {sigma.numel()}")
    return sigma.clamp_min(1e-12)


def _broadcast(sigma: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    shape = [sigma.shape[0]] + [1] * (reference.ndim - 1)
    return sigma.view(*shape)


def edm_preconditioning(
    sigma: torch.Tensor | float,
    reference: torch.Tensor,
    *,
    sigma_data: float = 0.5,
) -> EDMPrecondOutput:
    r"""
    计算 Karras EDM 预条件系数。

    Parameters
    ----
    sigma : torch.Tensor or float
        噪声水平，shape :math:`(B,)` 或标量。
    reference : torch.Tensor
        用于确定 batch 和广播维度的参考张量。
    sigma_data : float, optional, default=0.5
        数据标准差超参数。

    Returns
    ----
    EDMPrecondOutput
        可直接用于 EDM denoiser 的系数集合。
    """

    sigma_vec = _sigma_vector(sigma, reference)
    sigma_b = _broadcast(sigma_vec, reference)
    sigma_data_t = torch.as_tensor(float(sigma_data), device=reference.device, dtype=reference.dtype)
    denom = sigma_b.square() + sigma_data_t.square()
    c_skip = sigma_data_t.square() / denom
    c_out = sigma_b * sigma_data_t / torch.sqrt(denom)
    c_in = 1.0 / torch.sqrt(denom)
    c_noise = torch.log(sigma_vec.clamp_min(1e-12)) / 4.0
    weight = denom / (sigma_b * sigma_data_t).square().clamp_min(1e-12)
    return EDMPrecondOutput(c_skip=c_skip, c_out=c_out, c_in=c_in, c_noise=c_noise, weight=weight)


def edm_weighted_mse(
    denoised: torch.Tensor,
    clean: torch.Tensor,
    sigma: torch.Tensor | float,
    *,
    sigma_data: float = 0.5,
) -> torch.Tensor:
    r"""
    计算 EDM 加权 MSE。

    Parameters
    ----
    denoised : torch.Tensor
        去噪结果。
    clean : torch.Tensor
        干净目标。
    sigma : torch.Tensor or float
        噪声水平。
    sigma_data : float, optional, default=0.5
        数据标准差超参数。

    Returns
    ----
    torch.Tensor
        标量损失。
    """

    coeffs = edm_preconditioning(sigma, clean, sigma_data=sigma_data)
    return (coeffs.weight * (denoised - clean).square()).mean()