"""评估指标（v3）

包含：
- RMSE
- MAE
- CRPS（连续秩概率评分，ensemble 近似）

CRPS 用于概率预报/集成预报的评价：
CRPS = E|X - y| - 0.5 E|X - X'|
其中 X, X' 是从预测分布中独立采样的随机变量。
"""

import torch


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """均方根误差（RMSE）。"""

    return torch.sqrt(torch.mean((pred - target) ** 2))


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """平均绝对误差（MAE）。"""

    return torch.mean(torch.abs(pred - target))


def crps_ensemble(ensemble: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """集成预报的 CRPS 近似计算。

    Args:
        ensemble: [B, M, C, H, W]
        target:   [B, C, H, W]
        reduction:
            - "none"：返回 [B, C, H, W]
            - "mean"：返回标量
            - "channel_mean"：返回 [C]

    Returns:
        CRPS

    公式（ensemble 近似）：
        CRPS ≈ (1/M) * Σ_i |x_i - y| - (1/(2M^2)) * Σ_{i,j} |x_i - x_j|
    """

    if ensemble.ndim != 5:
        raise ValueError(f"ensemble 必须是 5D [B,M,C,H,W]，但得到 {tuple(ensemble.shape)}")
    if target.ndim != 4:
        raise ValueError(f"target 必须是 4D [B,C,H,W]，但得到 {tuple(target.shape)}")

    B, M, C, H, W = ensemble.shape
    if target.shape != (B, C, H, W):
        raise ValueError(f"target shape {tuple(target.shape)} 与 ensemble 的 (B,C,H,W)={(B,C,H,W)} 不匹配")

    # 第一项：E|X - y|
    term1 = torch.mean(torch.abs(ensemble - target.unsqueeze(1)), dim=1)  # [B,C,H,W]

    # 第二项：0.5 * E|X - X'|
    pairwise = torch.abs(ensemble.unsqueeze(2) - ensemble.unsqueeze(1))  # [B,M,M,C,H,W]
    term2 = 0.5 * torch.mean(pairwise, dim=(1, 2))  # [B,C,H,W]

    crps = term1 - term2

    if reduction == "none":
        return crps
    if reduction == "mean":
        return crps.mean()
    if reduction == "channel_mean":
        return crps.mean(dim=(0, 2, 3))  # [C]

    raise ValueError(f"未知 reduction: {reduction}")
