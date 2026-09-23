r"""
Karras/EDM 噪声调度工具。
"""

from __future__ import annotations

import torch


def sample_log_normal_sigma(
    batch_size: int,
    *,
    sigma_min: float,
    sigma_max: float,
    mean: float = -1.2,
    std: float = 1.2,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""
    按 EDM 常用对数正态分布采样噪声水平。

    Parameters
    ----
    batch_size : int
        批大小。
    sigma_min : float
        最小噪声水平。
    sigma_max : float
        最大噪声水平。
    mean : float, optional, default=-1.2
        log sigma 均值。
    std : float, optional, default=1.2
        log sigma 标准差。
    device : torch.device or str
        输出设备。
    dtype : torch.dtype, optional, default=torch.float32
        输出类型。

    Returns
    ----
    torch.Tensor
        shape :math:`(B,)` 的噪声水平。
    """

    sigma = torch.randn(int(batch_size), device=device, dtype=dtype) * float(std) + float(mean)
    return sigma.exp().clamp(min=float(sigma_min), max=float(sigma_max))


def karras_schedule(
    num_steps: int,
    *,
    sigma_min: float,
    sigma_max: float,
    rho: float = 7.0,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    append_zero: bool = True,
) -> torch.Tensor:
    r"""
    生成 Karras EDM 采样噪声序列。

    Parameters
    ----
    num_steps : int
        非零采样步数。
    sigma_min : float
        最小噪声水平。
    sigma_max : float
        最大噪声水平。
    rho : float, optional, default=7.0
        Karras schedule 形状参数。
    device : torch.device or str
        输出设备。
    dtype : torch.dtype, optional, default=torch.float32
        输出类型。
    append_zero : bool, optional, default=True
        是否在末尾追加 0，便于 ODE 采样。

    Returns
    ----
    torch.Tensor
        从大到小排列的噪声水平。
    """

    steps = max(1, int(num_steps))
    rho = max(float(rho), 1e-6)
    ramp = torch.linspace(0.0, 1.0, steps, device=device, dtype=dtype)
    min_inv = float(sigma_min) ** (1.0 / rho)
    max_inv = float(sigma_max) ** (1.0 / rho)
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho
    if append_zero:
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)], dim=0)
    return sigmas