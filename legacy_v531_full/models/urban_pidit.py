"""UrbanPiDiT V5.3.1 MorphoProcessDiT.

Morphology-conditioned process diffusion transformer for urban microclimate forecasting.
城市形态字段在本模型中被转化为 morphology-derived process proxies，用于条件调制、
过程图传播和诊断输出；这些代理量不声明为实测物理参数或因果证明。

说明：
- 核心任务：给定噪声场 x_t、上下文 x_ctx（历史帧 + 静态特征）以及扩散时间步 t，预测干净场 x0。
- 审稿证据链：forward 可选返回统一 diagnostics，用于过程调制、图结构和反事实敏感性检查。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from urbanpidit_version import CLAIM_BOUNDARY, METHOD_FAMILY, VERSION, VERSION_NAME, VERSION_SHORT
except ImportError:  # pragma: no cover
    CLAIM_BOUNDARY = "morphology-derived process proxies are diagnostic conditioning signals."
    METHOD_FAMILY = "Morphology-conditioned Process Diffusion Transformer"
    VERSION = "5.3.1"
    VERSION_NAME = "UrbanPiDiT-V5.3.1-MorphoProcessDiT"
    VERSION_SHORT = "V5.3.1"

try:
    from .anisotropic_urban_process_graph import AnisotropicUrbanProcessGraph
    from .micromet_coupling import MicroMetCouplingOperator
    from .morpho_process_proxy import MorphologyProcessProxyEncoder, PROCESS_PROXY_NAMES
    from .morphology_graph import HeterogeneousMorphologyGraph, WindAwareMorphologyGraph
    from .process_conditioned_adaln import ProcessConditionedAdaLN
    from .static_encoder import StaticMorphologyEncoder
    from .urban_canopy import UrbanCanopyCoupling
    from .urban_morphology_control import UrbanMorphologyControlBranch
    from .variable_graph_v2 import ProxyConditionedVariableGraph
except ImportError:
    from models.anisotropic_urban_process_graph import AnisotropicUrbanProcessGraph
    from models.micromet_coupling import MicroMetCouplingOperator
    from models.morpho_process_proxy import MorphologyProcessProxyEncoder, PROCESS_PROXY_NAMES
    from models.morphology_graph import HeterogeneousMorphologyGraph, WindAwareMorphologyGraph
    from models.process_conditioned_adaln import ProcessConditionedAdaLN
    from models.static_encoder import StaticMorphologyEncoder
    from models.urban_canopy import UrbanCanopyCoupling
    from models.urban_morphology_control import UrbanMorphologyControlBranch
    from models.variable_graph_v2 import ProxyConditionedVariableGraph


# ============================
# 配置结构
# ============================


@dataclass
class ModelConfig:
    """模型配置类（用于快速构建模型）"""

    # 空间分辨率
    H: int = 8
    W: int = 8

    # 通道数
    in_channels: int = 7
    ctx_channels: int = 33
    out_channels: int = 7

    # Transformer
    D: int = 512
    depth: int = 12
    heads: int = 8
    mlp_ratio: float = 4.0

    # 静态变量数（如果为 0，则尝试从 ctx_channels 推断）
    static_channels: int = 5

    # 正则
    drop_path_rate: float = 0.1
    dropout: float = 0.1

    # 组件开关（用于消融）
    use_hybrid_attention: bool = False
    use_multi_scale: bool = False
    use_cross_attention: bool = True
    use_position_encoding: bool = True
    use_timestep_conditioning: bool = True
    # 是否启用“预测提前期(lead time)”条件（一个模型覆盖多个提前期）
    # 该开关主要用于多提前期训练/评估的消融实验。
    use_lead_time_conditioning: bool = False
    use_variable_graph: bool = False
    use_dynamic_vg: Optional[bool] = None
    use_static_vg: Optional[bool] = None
    use_proxy_conditioned_vg: bool = False
    vg_num_heads: int = 4
    use_static_morphology_encoder: bool = False
    use_morphology_graph: bool = False
    use_wind_aware_graph: bool = False
    morphology_graph_cfg: Optional[Dict[str, Any]] = None
    use_urban_canopy_coupling: bool = False
    use_urban_canopy: Optional[bool] = None
    canopy_coupling_mode: str = "pre_transformer"
    urban_canopy_cfg: Optional[Dict[str, Any]] = None
    use_micromet_coupling: bool = False
    micromet_coupling_cfg: Optional[Dict[str, Any]] = None
    use_process_proxy_encoder: bool = False
    use_process_adaln: bool = False
    use_urban_control_branch: bool = False
    use_anisotropic_process_graph: bool = False
    use_micromet_token_branches: bool = False
    use_morphology_residual_head: bool = False
    degrade_to_v52: bool = False
    morpho_process_cfg: Optional[Dict[str, Any]] = None
    use_urban_graph: bool = False
    urban_graph_k: int = 8
    urban_graph_sigma: float = 1.0
    dynamic_vars: Optional[Tuple[str, ...]] = None
    static_vars: Optional[Tuple[str, ...]] = None
    static_schema: Optional[Dict[str, Any]] = None


# ============================
# 基础组件
# ============================


def build_urban_knn_graph(static_feat: torch.Tensor, k: int = 8, sigma: float = 1.0) -> torch.Tensor:
    """根据静态特征构建 kNN 图并返回行归一化邻接 A_norm。
    Args:
        static_feat: [N, S]，N=H*W, S=静态变量数(例如5)，应为 float tensor
        k: 每个节点的近邻数（建议 6~12）
        sigma: RBF 核宽度
    Returns:
        A_norm: [N, N] 行归一化邻接矩阵（dense）
    """
    N, S = static_feat.shape

    # 标准化（防止某个静态变量尺度压制其它变量）
    feat = (static_feat - static_feat.mean(dim=0, keepdim=True)) / (static_feat.std(dim=0, keepdim=True) + 1e-6)

    # pairwise L2 distance: [N,N]
    dist2 = torch.cdist(feat, feat, p=2.0) ** 2

    # 找每行 topk（包含自己），取 k+1 再去掉自己
    # 如果 N <= k，取 N 作为 k
    curr_k = min(k + 1, N)
    knn = torch.topk(dist2, k=curr_k, largest=False).indices  # [N, curr_k]

    A = torch.zeros((N, N), device=static_feat.device, dtype=static_feat.dtype)

    for i in range(N):
        nbrs = knn[i]
        nbrs = nbrs[nbrs != i][:k]  # 去掉自己
        w = torch.exp(-dist2[i, nbrs] / (2.0 * sigma * sigma))
        A[i, nbrs] = w

    # 可选：加自环，稳定
    A.fill_diagonal_(1.0)

    # 行归一化
    A_norm = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
    return A_norm



def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """扩散时间步嵌入（正弦/余弦位置编码）。

    Args:
        t: 时间步张量 [B]（通常在 [0, 1] 或离散步）
        dim: 嵌入维度
        max_period: 最大周期

    Returns:
        时间步嵌入 [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepMLP(nn.Module):
    """将时间步 t 映射为条件向量 c。"""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out),
        )
        self.dim_in = dim_in
        self.dim_out = dim_out

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        e = timestep_embedding(t, self.dim_in)
        return self.net(e)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """自适应归一化（AdaLN）调制：x' = x * (1 + scale) + shift。"""

    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Stochastic Depth (DropPath) 实现。

    随机丢弃整个残差分支，增强模型鲁棒性。
    """

    if drop_prob == 0.0 or (not training):
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """DropPath 模块封装。"""

    def __init__(self, drop_prob: Optional[float] = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob or 0.0, self.training)


def _resize_pos_embed(pos_embed: torch.Tensor, old_hw: Tuple[int, int], new_hw: Tuple[int, int]) -> torch.Tensor:
    """对可学习位置编码做双线性插值。

    pos_embed: [1, old_H*old_W, D]

    Returns:
        [1, new_H*new_W, D]
    """

    old_h, old_w = int(old_hw[0]), int(old_hw[1])
    new_h, new_w = int(new_hw[0]), int(new_hw[1])

    if old_h == new_h and old_w == new_w:
        return pos_embed

    if pos_embed.ndim != 3:
        raise ValueError(f"pos_embed 必须是 [1, HW, D]，但得到 {tuple(pos_embed.shape)}")

    _, hw, d = pos_embed.shape
    if hw != old_h * old_w:
        raise ValueError(
            f"pos_embed 的 HW={hw} 与 old_hw={old_hw} 不匹配（应为 {old_h*old_w}）"
        )

    # [1, HW, D] -> [1, D, H, W]
    pos = pos_embed.reshape(1, old_h, old_w, d).permute(0, 3, 1, 2)
    pos = F.interpolate(pos, size=(new_h, new_w), mode="bilinear", align_corners=False)
    # [1, D, H, W] -> [1, HW, D]
    pos = pos.permute(0, 2, 3, 1).reshape(1, new_h * new_w, d)
    return pos


# ============================
# 注意力模块
# ============================


class SpatialAttention(nn.Module):
    """空间注意力模块（类似 CBAM 的 spatial attention）。"""

    def __init__(self, dim: int):
        super().__init__()
        # 输入为 (avg, max) 两个通道
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
        x: [B, C, H, W]

        Returns:
        attention: [B, 1, H, W]
        """

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x_cat))


class ChannelAttention(nn.Module):
    """通道注意力模块（类似 CBAM 的 channel attention）。

    注意：原先若使用 nn.AdaptiveMaxPool2d(1)，在开启 deterministic=True 时，
    CUDA backward 可能触发非确定性算子报错。这里用 torch.amax 替代全局最大池化。
    """

    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        hidden = max(1, dim // reduction)  # 防止 dim 太小导致 hidden=0
        self.fc = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            attention: [B, C, 1, 1]
        """
        b, c, _, _ = x.shape

        # 平均池化分支（确定性）
        avg_out = self.fc(self.avg_pool(x).view(b, c))

        # 最大池化分支（确定性替代）
        # 全局最大：对空间维 (H, W) 取最大，等价于 AdaptiveMaxPool2d(1)
        max_vals = torch.amax(x, dim=(-2, -1)).view(b, c)
        max_out = self.fc(max_vals)

        return self.sigmoid(avg_out + max_out).view(b, c, 1, 1)

class HybridAttentionBlock(nn.Module):
    """混合注意力块：多头自注意力 + (可选)空间/通道注意力 + MLP。

    说明：
    - v2 版本中在 reshape 时使用 int(sqrt(N)) 推断 H/W，若 H!=W 会产生逻辑错误。
    - v3 版本在构造时显式传入 H/W（或在 forward 时显式给出），以避免错误。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_hybrid: bool = True,
        spatial_hw: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.use_hybrid = bool(use_hybrid)
        self.spatial_hw = spatial_hw

        # 标准多头自注意力
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=drop, batch_first=True)

        # MLP
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
        )

        # 条件调制（输出 6 组 shift/scale/gate）
        self.cond_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # 混合注意力：空间/通道
        if self.use_hybrid:
            self.spatial_attn = SpatialAttention(dim)
            self.channel_attn = ChannelAttention(dim)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        spatial_hw: Optional[Tuple[int, int]] = None,
        process_adaln: Optional[ProcessConditionedAdaLN] = None,
        proxy_fields: Optional[Dict[str, torch.Tensor]] = None,
        block_idx: int = 0,
    ) -> torch.Tensor:
        """Args:
        x: [B, N, D]
        c: [B, D]
        spatial_hw: (H, W)，若为 None 则使用构造时的 spatial_hw

        Returns:
        [B, N, D]
        """

        # 条件调制参数
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.cond_mlp(c).chunk(6, dim=-1)

        # 自注意力分支
        h = self.norm1(x)
        if process_adaln is None:
            h = modulate(h, shift_msa, scale_msa)
        else:
            h = process_adaln(h, shift_msa, scale_msa, proxy_fields=proxy_fields, block_idx=block_idx)
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + self.drop_path(gate_msa.unsqueeze(1) * h)

        # 额外空间/通道注意力（可选）
        if self.use_hybrid:
            b, n, d = x.shape

            hw = spatial_hw or self.spatial_hw
            if hw is None:
                # 兜底：仅当 N 是完全平方数时才允许推断
                side = int(math.sqrt(n))
                if side * side != n:
                    raise ValueError(
                        f"HybridAttentionBlock 无法从 N={n} 推断空间尺寸，请在构造或 forward 时提供 spatial_hw=(H,W)"
                    )
                h_, w_ = side, side
            else:
                h_, w_ = int(hw[0]), int(hw[1])
                if h_ * w_ != n:
                    raise ValueError(f"HybridAttentionBlock: 提供的 spatial_hw={hw} 与 N={n} 不匹配")

            x_2d = x.transpose(1, 2).reshape(b, d, h_, w_)

            # 空间注意力
            spatial_weight = self.spatial_attn(x_2d)
            x_2d = x_2d * spatial_weight

            # 通道注意力
            channel_weight = self.channel_attn(x_2d)
            x_2d = x_2d * channel_weight

            x = x_2d.reshape(b, d, n).transpose(1, 2)

        # MLP 分支
        h2 = self.norm2(x)
        if process_adaln is None:
            h2 = modulate(h2, shift_mlp, scale_mlp)
        else:
            h2 = process_adaln(h2, shift_mlp, scale_mlp, proxy_fields=proxy_fields, block_idx=block_idx)
        h2 = self.mlp(h2)
        x = x + self.drop_path(gate_mlp.unsqueeze(1) * h2)

        return x


class CrossAttentionBlock(nn.Module):
    """交叉注意力块：先自注意力，再用上下文做 cross-attn，然后 MLP。"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()

        # 自注意力
        self.norm_q_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=drop, batch_first=True)

        # 交叉注意力
        self.norm_q_cross = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=drop, batch_first=True)

        # MLP
        self.norm_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
        )

        # 条件调制（9 组 shift/scale/gate）
        self.cond_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        c: torch.Tensor,
        process_adaln: Optional[ProcessConditionedAdaLN] = None,
        proxy_fields: Optional[Dict[str, torch.Tensor]] = None,
        block_idx: int = 0,
    ) -> torch.Tensor:
        """Args:
        x_q: [B, N, D]
        x_kv: [B, M, D]
        c: [B, D]

        Returns:
        [B, N, D]
        """

        params = self.cond_mlp(c).chunk(9, dim=-1)
        shift_self, scale_self, gate_self = params[0], params[1], params[2]
        shift_cross, scale_cross, gate_cross = params[3], params[4], params[5]
        shift_mlp, scale_mlp, gate_mlp = params[6], params[7], params[8]

        # 自注意力
        h_self = self.norm_q_self(x_q)
        if process_adaln is None:
            h_self = modulate(h_self, shift_self, scale_self)
        else:
            h_self = process_adaln(h_self, shift_self, scale_self, proxy_fields=proxy_fields, block_idx=block_idx)
        h_self = self.self_attn(h_self, h_self, h_self, need_weights=False)[0]
        x_q = x_q + self.drop_path(gate_self.unsqueeze(1) * h_self)

        # 交叉注意力
        h_cross = self.norm_q_cross(x_q)
        if process_adaln is None:
            h_cross = modulate(h_cross, shift_cross, scale_cross)
        else:
            h_cross = process_adaln(h_cross, shift_cross, scale_cross, proxy_fields=proxy_fields, block_idx=block_idx)
        h_cross = self.cross_attn(query=h_cross, key=x_kv, value=x_kv, need_weights=False)[0]
        x_q = x_q + self.drop_path(gate_cross.unsqueeze(1) * h_cross)

        # MLP
        h_mlp = self.norm_mlp(x_q)
        if process_adaln is None:
            h_mlp = modulate(h_mlp, shift_mlp, scale_mlp)
        else:
            h_mlp = process_adaln(h_mlp, shift_mlp, scale_mlp, proxy_fields=proxy_fields, block_idx=block_idx)
        h_mlp = self.mlp(h_mlp)
        x_q = x_q + self.drop_path(gate_mlp.unsqueeze(1) * h_mlp)

        return x_q


# ============================
# 多尺度融合
# ============================


class MultiScaleFusion(nn.Module):
    """多尺度特征金字塔融合（类似 FPN）。"""

    def __init__(self, dim: int, num_scales: int = 3):
        super().__init__()
        if num_scales < 1:
            raise ValueError("num_scales 必须 >= 1")
        self.num_scales = num_scales

        self.downsample_layers = nn.ModuleList(
            [nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1) for _ in range(num_scales - 1)]
        )
        self.upsample_layers = nn.ModuleList(
            [nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2) for _ in range(num_scales - 1)]
        )
        self.fusion_convs = nn.ModuleList(
            [nn.Conv2d(dim * 2, dim, kernel_size=1) for _ in range(num_scales - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
        x: [B, D, H, W]

        Returns:
        fused: [B, D, H, W]
        """

        # 构建金字塔（自顶向下）
        pyramid = [x]
        for down in self.downsample_layers:
            pyramid.append(down(pyramid[-1]))

        # 融合金字塔（自底向上）
        fused = pyramid[-1]
        for i in range(len(self.upsample_layers) - 1, -1, -1):
            fused = self.upsample_layers[i](fused)
            # 由于 stride=2 的上下采样可能带来 1 像素差异，这里做一次对齐裁剪
            if fused.shape[-2:] != pyramid[i].shape[-2:]:
                fused = F.interpolate(fused, size=pyramid[i].shape[-2:], mode="bilinear", align_corners=False)
            fused = torch.cat([fused, pyramid[i]], dim=1)
            fused = self.fusion_convs[i](fused)

        return fused


class VariableChannelAttention(nn.Module):
    """变量间通道交互模块（更稳版本）：
    - 使用可学习邻接 logits，通过 softmax 得到行归一化的邻接矩阵（像注意力权重）
    - 使用 alpha 控制静态图强度，避免 adj 放大导致训练不稳
    - 使用 gamma 门控残差增量（初始化为 0），保证模块初始为恒等映射
    """

    def __init__(self, in_channels: int, dropout: float = 0.0, symmetric: bool = False):
        super().__init__()
        self.c = in_channels
        self.symmetric = symmetric

        # 邻接 logits（不直接用 adj，避免无约束爆炸）
        self.adj_logits = nn.Parameter(torch.zeros(in_channels, in_channels))
        nn.init.normal_(self.adj_logits, mean=0.0, std=0.02)

        # 控制静态图强度：adj = I + alpha * A
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self.norm = nn.LayerNorm(in_channels)

        hidden = max(8, in_channels * 4)
        self.mixer = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, in_channels),
        )

        # 门控系数，初始化为 0，让模块刚开始等同于恒等映射（训练更稳）
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        """
        b, c, h, w = x.shape
        assert c == self.c

        x_in = x.permute(0, 2, 3, 1)  # [B,H,W,C]

        # 计算邻接矩阵 A（行归一化）
        logits = self.adj_logits
        if self.symmetric:
            logits = 0.5 * (logits + logits.t())  # 可选：对称图更可解释

        A = torch.softmax(logits, dim=-1)  # [C,C] 每行和为 1，像“注意力权重”

        # 静态图传播：x @ (I + alpha*A)
        I = torch.eye(c, device=x.device, dtype=x.dtype)
        adj = I + torch.clamp(self.alpha, 0.0, 1.0) * A

        x_adj = torch.matmul(x_in, adj)  # [B,H,W,C]

        # 动态交互（MLP），加门控残差
        delta = self.mixer(self.norm(x_adj))
        x_out = x_in + self.gamma * delta

        return x_out.permute(0, 3, 1, 2)  # [B,C,H,W]


class UrbanSpatialGraph(nn.Module):
    """Urban Graph 空间传播层（基于静态城市特征的 kNN 图）。

    输入 token: [B, N, D]，N=H*W
    使用预计算的 A_norm: [N, N]（行归一化邻接）
    输出: [B, N, D]
    """

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        # 门控系数，初始化为 0，初始等价恒等映射，训练更稳
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
            A_norm: [N, N] 行归一化邻接矩阵（buffer）
        """
        # (N,N) @ (B,N,D) -> (B,N,D)
        x_msg = torch.einsum("nm,bmd->bnd", A_norm, x)
        delta = self.mlp(self.norm(x_msg))
        return x + self.gamma * delta


class MorphologyResidualPredictionHead(nn.Module):
    """形态代理调制的预测残差头。"""

    def __init__(self, dim: int, out_channels: int, proxy_channels: int = 7) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.base = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out_channels))
        self.gate = nn.Sequential(
            nn.Conv2d(proxy_channels, max(out_channels * 2, 8), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(max(out_channels * 2, 8), out_channels, kernel_size=1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.last_diagnostics: Dict[str, torch.Tensor] = {}

    def forward(self, tokens: torch.Tensor, proxy_fields: Dict[str, torch.Tensor], spatial_hw: Tuple[int, int]) -> torch.Tensor:
        batch, nodes, _ = tokens.shape
        height, width = int(spatial_hw[0]), int(spatial_hw[1])
        if nodes != height * width:
            raise ValueError(f"token 数 N={nodes} 与 spatial_hw={spatial_hw} 不匹配")
        residual = self.base(tokens).reshape(batch, height, width, self.out_channels).permute(0, 3, 1, 2)
        proxy = torch.cat([proxy_fields[name] for name in PROCESS_PROXY_NAMES if name in proxy_fields], dim=1).to(
            device=tokens.device,
            dtype=tokens.dtype,
        )
        if proxy.size(1) != self.gate[0].in_channels:
            if proxy.size(1) > self.gate[0].in_channels:
                proxy = proxy[:, : self.gate[0].in_channels]
            else:
                pad = proxy.new_zeros(
                    proxy.size(0),
                    self.gate[0].in_channels - proxy.size(1),
                    proxy.size(2),
                    proxy.size(3),
                )
                proxy = torch.cat([proxy, pad], dim=1)
        if proxy.shape[-2:] != (height, width):
            proxy = F.interpolate(proxy, size=(height, width), mode="bilinear", align_corners=False)
        gate = torch.tanh(self.gate(proxy))
        out = gate * residual
        self.last_diagnostics = {
            "residual_prediction/residual_norm": out.detach().norm(),
            "residual_prediction/spatial_gate_mean": gate.detach().mean(),
        }
        return out


# ============================
# 主模型
# ============================


class UrbanPiDiT(nn.Module):
    """Morphology-conditioned process diffusion transformer（UrbanPiDiT）。"""

    def __init__(
        self,
        H: int = 8,
        W: int = 8,
        in_channels: int = 7,
        ctx_channels: int = 33,
        out_channels: int = 7,
        D: int = 512,
        depth: int = 12,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        static_channels: int = 5,
        drop_path_rate: float = 0.0,
        dropout: float = 0.0,
        use_hybrid_attention: bool = False,
        use_multi_scale: bool = False,
        use_cross_attention: bool = True,
        use_position_encoding: bool = True,
        use_timestep_conditioning: bool = True,
        use_lead_time_conditioning: bool = False,
        use_variable_graph: bool = False,
        use_dynamic_vg: Optional[bool] = None,
        use_static_vg: Optional[bool] = None,
        use_proxy_conditioned_vg: bool = False,
        vg_num_heads: int = 4,
        use_static_morphology_encoder: bool = False,
        use_morphology_graph: bool = False,
        use_wind_aware_graph: bool = False,
        morphology_graph_cfg: Optional[Dict[str, Any]] = None,
        use_urban_canopy_coupling: bool = False,
        use_urban_canopy: Optional[bool] = None,
        canopy_coupling_mode: str = "pre_transformer",
        urban_canopy_cfg: Optional[Dict[str, Any]] = None,
        use_micromet_coupling: bool = False,
        micromet_coupling_cfg: Optional[Dict[str, Any]] = None,
        use_process_proxy_encoder: bool = False,
        use_process_adaln: bool = False,
        use_urban_control_branch: bool = False,
        use_anisotropic_process_graph: bool = False,
        use_micromet_token_branches: bool = False,
        use_morphology_residual_head: bool = False,
        degrade_to_v52: bool = False,
        morpho_process_cfg: Optional[Dict[str, Any]] = None,
        use_urban_graph: bool = False,
        urban_graph_k: int = 8,
        urban_graph_sigma: float = 1.0,
        dynamic_vars: Optional[Tuple[str, ...]] = None,
        static_vars: Optional[Tuple[str, ...]] = None,
        static_schema: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.base_H = int(H)
        self.base_W = int(W)
        self.in_channels = int(in_channels)
        self.ctx_channels = int(ctx_channels)
        self.out_channels = int(out_channels)
        self.D = int(D)
        self.use_multi_scale = bool(use_multi_scale)
        self.use_cross_attention = bool(use_cross_attention)
        self.use_position_encoding = bool(use_position_encoding)
        self.use_timestep_conditioning = bool(use_timestep_conditioning)
        self.use_lead_time_conditioning = bool(use_lead_time_conditioning)
        self.use_variable_graph = bool(use_variable_graph)
        self.use_proxy_conditioned_vg = bool(use_proxy_conditioned_vg)
        self.vg_num_heads = max(int(vg_num_heads), 1)
        self.use_dynamic_vg = bool(self.use_variable_graph if use_dynamic_vg is None else use_dynamic_vg)
        self.use_static_vg = bool(self.use_variable_graph if use_static_vg is None else use_static_vg)
        self.use_static_morphology_encoder = bool(use_static_morphology_encoder)
        self.use_morphology_graph = bool(use_morphology_graph)
        self.use_wind_aware_graph = bool(use_wind_aware_graph)
        self.morphology_graph_cfg = dict(morphology_graph_cfg or {})
        self.use_urban_canopy_coupling = bool(use_urban_canopy_coupling if use_urban_canopy is None else use_urban_canopy)
        self.canopy_coupling_mode = str(canopy_coupling_mode)
        self.urban_canopy_cfg = dict(urban_canopy_cfg or {})
        self.use_micromet_coupling = bool(use_micromet_coupling)
        self.micromet_coupling_cfg = dict(micromet_coupling_cfg or {})
        self.degrade_to_v52 = bool(degrade_to_v52)
        self.morpho_process_cfg = dict(morpho_process_cfg or {})
        self.use_process_proxy_encoder = bool(use_process_proxy_encoder and not self.degrade_to_v52)
        self.use_process_adaln = bool(use_process_adaln and not self.degrade_to_v52)
        self.use_urban_control_branch = bool(use_urban_control_branch and not self.degrade_to_v52)
        self.use_anisotropic_process_graph = bool(use_anisotropic_process_graph and not self.degrade_to_v52)
        self.use_micromet_token_branches = bool(use_micromet_token_branches and not self.degrade_to_v52)
        self.use_morphology_residual_head = bool(use_morphology_residual_head and not self.degrade_to_v52)
        self.micromet_coupling_mode = str(self.micromet_coupling_cfg.get("mode", "pre"))
        self.micromet_coupling_interval = max(int(self.micromet_coupling_cfg.get("interleaved_interval", 1)), 1)
        legacy_micromet_token_modes = {"interleaved", "inside", "mid"}
        v53_micromet_token_modes = {"token", "token_interleaved"}
        self.use_micromet_token_adapter = bool(
            self.micromet_coupling_mode in legacy_micromet_token_modes
            or (self.use_micromet_token_branches and self.micromet_coupling_mode in v53_micromet_token_modes)
        )
        self.micromet_token_to_state: Optional[nn.Module] = None
        self.micromet_state_to_token: Optional[nn.Module] = None
        self.last_micromet_diagnostics: Dict[str, torch.Tensor] = {}
        self.last_urban_canopy_diagnostics: Dict[str, torch.Tensor] = {}
        self.last_morphology_graph_diagnostics: Dict[str, torch.Tensor] = {}
        self.use_urban_graph = bool(use_urban_graph)
        self.urban_graph_k = int(urban_graph_k)
        self.urban_graph_sigma = float(urban_graph_sigma)
        self.dynamic_vars = tuple(dynamic_vars or ())
        self.static_vars = tuple(static_vars or ())
        self.static_schema = dict(static_schema or {})

        # ---------- 推断历史帧数 k 与静态变量数 ----------
        # 约定：ctx = [每个变量的 k 帧] + [静态变量(可选)]
        # 因而：ctx_channels = in_channels * k + static_channels
        if static_channels < 0:
            raise ValueError("static_channels 不能为负数")

        if static_channels == 0:
            # 允许自动推断：当 include_static=True 且 static_channels 未显式给出时，
            # 仅在 static_channels < in_channels 的常见设置下，ctx_channels % in_channels 才能正确恢复静态通道数。
            inferred_static = int(ctx_channels % in_channels)
            inferred_k_num = ctx_channels - inferred_static
            if inferred_k_num < 0 or (inferred_k_num % in_channels != 0):
                raise ValueError(
                    f"无法从 ctx_channels={ctx_channels}, in_channels={in_channels} 推断 k/static_channels；"
                    "请在模型配置中显式提供 static_channels。"
                )
            self.real_static = inferred_static
            self.real_k = int(inferred_k_num // in_channels)
        else:
            if ctx_channels < static_channels:
                raise ValueError(f"ctx_channels({ctx_channels}) 不能小于 static_channels({static_channels})")
            k_num = ctx_channels - static_channels
            if k_num % in_channels != 0:
                raise ValueError(
                    f"ctx_channels({ctx_channels}) - static_channels({static_channels}) 必须能被 in_channels({in_channels}) 整除"
                )
            self.real_static = int(static_channels)
            self.real_k = int(k_num // in_channels)

        if self.real_k <= 0:
            raise ValueError(f"历史帧数 k 推断结果异常：k={self.real_k}（请检查 ctx_channels/in_channels/static_channels）")

        print(
            f"[UrbanPiDiT {VERSION_SHORT}] 配置: HxW={self.base_H}x{self.base_W}, "
            f"K={self.real_k}, 静态变量={self.real_static}, "
            f"DropPath={drop_path_rate}, Dropout={dropout}, "
            f"混合注意力={use_hybrid_attention}, 多尺度={use_multi_scale}, "
            f"交叉注意力={use_cross_attention}, 位置编码={use_position_encoding}, 时间步条件={use_timestep_conditioning}, "
            f"提前期条件={use_lead_time_conditioning}, "
            f"变量关系图={self.use_variable_graph}, 动态VG={self.use_dynamic_vg}, 静态VG={self.use_static_vg}, "
            f"代理条件VG={self.use_proxy_conditioned_vg}, VG heads={self.vg_num_heads}, "
            f"旧城市空间图={use_urban_graph}, 静态形态编码={use_static_morphology_encoder}, "
            f"形态图={use_morphology_graph}, 风向图={use_wind_aware_graph}, "
            f"冠层耦合={self.use_urban_canopy_coupling}, 微气象耦合={self.use_micromet_coupling}, "
            f"过程代理={self.use_process_proxy_encoder}, 过程AdaLN={self.use_process_adaln}, "
            f"控制分支={self.use_urban_control_branch}, 自适应过程图={self.use_anisotropic_process_graph}"
        )

        # ---------- 上一帧索引（用于构造噪声输入） ----------
        # 约定上下文通道顺序为「变量优先」：
        #   [var0_t0, var0_t1, ..., var0_t{k-1}, var1_t0, ..., var{C-1}_t{k-1}, (static...)]
        indices = [i * self.real_k + (self.real_k - 1) for i in range(in_channels)]
        self.register_buffer("last_indices", torch.tensor(indices, dtype=torch.long), persistent=False)

        # ---------- 变量关系图（可选） ----------
        # use_variable_graph 维持旧配置兼容；use_dynamic_vg/use_static_vg 用于精确消融。
        if self.use_dynamic_vg:
            self.var_graph_t = VariableChannelAttention(in_channels * 2, dropout=dropout, symmetric=True)
        else:
            self.var_graph_t = None

        if self.use_static_vg:
            self.var_graph_ctx = VariableChannelAttention(ctx_channels, dropout=dropout, symmetric=True)
        else:
            self.var_graph_ctx = None

        proxy_vg_cfg = dict(self.morpho_process_cfg.get("proxy_conditioned_variable_graph", {}) or {})
        if self.use_proxy_conditioned_vg:
            self.proxy_cond_vg = ProxyConditionedVariableGraph(
                in_channels * 2,
                num_heads=int(proxy_vg_cfg.get("num_heads", self.vg_num_heads)),
                hidden_dim=proxy_vg_cfg.get("hidden_dim", None),
                dropout=float(proxy_vg_cfg.get("dropout", dropout)),
                symmetric=bool(proxy_vg_cfg.get("symmetric", True)),
                use_proxy_context=bool(proxy_vg_cfg.get("use_proxy_context", True)),
            )
        else:
            self.proxy_cond_vg = None
        self.last_variable_graph_diagnostics: Dict[str, torch.Tensor] = {}

        # ---------- 城市冠层耦合（可选） ----------
        if self.use_urban_canopy_coupling:
            self.urban_canopy = UrbanCanopyCoupling(
                in_channels=in_channels,
                static_channels=int(self.urban_canopy_cfg.get("static_channels", self.real_static)),
                dynamic_vars=self.dynamic_vars,
                hidden_channels=int(self.urban_canopy_cfg.get("hidden_channels", max(D // 8, 16))),
                dropout=float(self.urban_canopy_cfg.get("dropout", dropout)),
                use_wind_drag=bool(self.urban_canopy_cfg.get("use_wind_drag", True)),
                use_thermal=bool(self.urban_canopy_cfg.get("use_thermal", True)),
                use_moisture=bool(self.urban_canopy_cfg.get("use_moisture", True)),
                use_cross_var=bool(self.urban_canopy_cfg.get("use_cross_var", True)),
                max_wind_drag=float(self.urban_canopy_cfg.get("max_wind_drag", self.urban_canopy_cfg.get("drag_strength", 0.20))),
                max_branch_scale=float(self.urban_canopy_cfg.get("max_branch_scale", 0.10)),
                graph_cfg=self.urban_canopy_cfg.get("graph_cfg", self.morphology_graph_cfg),
            )
        else:
            self.urban_canopy = None

        # ---------- 微气象过程耦合（可选） ----------
        micromet_token_cfg = dict(self.morpho_process_cfg.get("micromet_token_branches", {}) or {})
        micromet_init_gate = float(
            self.micromet_coupling_cfg.get("init_gate", micromet_token_cfg.get("init_gate", 0.0))
        )
        if self.use_micromet_coupling:
            self.micromet_coupling = MicroMetCouplingOperator(
                in_channels=in_channels,
                static_channels=int(self.micromet_coupling_cfg.get("static_channels", self.real_static)),
                dynamic_vars=self.dynamic_vars,
                hidden_channels=int(self.micromet_coupling_cfg.get("hidden_channels", max(D // 8, 16))),
                dropout=float(self.micromet_coupling_cfg.get("dropout", dropout)),
                enabled=bool(self.micromet_coupling_cfg.get("enabled", True)),
                mode=self.micromet_coupling_mode,
                enable_momentum_drag=bool(self.micromet_coupling_cfg.get("enable_momentum_drag", True)),
                enable_thermal_storage=bool(self.micromet_coupling_cfg.get("enable_thermal_storage", True)),
                enable_moisture_evaporation=bool(self.micromet_coupling_cfg.get("enable_moisture_evaporation", True)),
                enable_ventilation_mixing=bool(self.micromet_coupling_cfg.get("enable_ventilation_mixing", True)),
                max_drag=float(self.micromet_coupling_cfg.get("max_drag", 0.20)),
                max_branch_scale=float(self.micromet_coupling_cfg.get("max_branch_scale", 0.10)),
                init_gate=micromet_init_gate,
                graph_cfg=self.micromet_coupling_cfg.get("graph_cfg", self.morphology_graph_cfg),
            )
        else:
            self.micromet_coupling = None

        # ---------- 城市空间图（可选） ----------
        if self.use_urban_graph:
            self.urban_graph = UrbanSpatialGraph(D, dropout=dropout)
            # buffer: 存储预计算的 A_norm，或者在 forward 中动态计算
            # 这里的 N=H*W，实际使用时需要 reshape
            self.register_buffer("A_norm", None)
        else:
            self.urban_graph = None

        # ---------- 城市形态物理耦合图（可选） ----------
        if self.use_morphology_graph or self.use_wind_aware_graph:
            graph_cls = WindAwareMorphologyGraph if self.use_wind_aware_graph else HeterogeneousMorphologyGraph
            graph_kwargs = {
                "dim": D,
                "k": int(self.morphology_graph_cfg.get("k", self.urban_graph_k)),
                "morphology_sigma": float(
                    self.morphology_graph_cfg.get(
                        "morphology_sigma",
                        self.morphology_graph_cfg.get("sigma", self.urban_graph_sigma),
                    )
                ),
                "spatial_sigma": float(
                    self.morphology_graph_cfg.get(
                        "spatial_sigma",
                        self.morphology_graph_cfg.get("sigma", self.urban_graph_sigma),
                    )
                ),
                "dropout": float(self.morphology_graph_cfg.get("dropout", dropout)),
                "use_self_loop": bool(self.morphology_graph_cfg.get("use_self_loop", True)),
            }
            if graph_cls is WindAwareMorphologyGraph:
                graph_kwargs["wind_temperature"] = float(self.morphology_graph_cfg.get("wind_temperature", 0.25))
            else:
                graph_kwargs["relation_count"] = int(self.morphology_graph_cfg.get("relation_count", 2))
            self.morphology_graph = graph_cls(**graph_kwargs)
        else:
            self.morphology_graph = None

        # ---------- 静态形态编码器（可选） ----------
        if self.use_static_morphology_encoder and self.real_static > 0:
            self.static_morphology_encoder = StaticMorphologyEncoder(
                static_vars=self.static_vars or tuple(f"static_{i}" for i in range(self.real_static)),
                static_schema=self.static_schema,
                out_channels=self.real_static,
                hidden_channels=max(self.real_static * 4, 16),
                categorical_embed_dim=int(self.static_schema.get("categorical_embed_dim", 4)) if self.static_schema else 4,
                dropout=float(dropout),
            )
        else:
            self.static_morphology_encoder = None

        # ---------- 输入投影 ----------
        # 目标场投影：将 [x_t, x_last] 拼接后映射到 D
        self.proj_t = nn.Linear(in_channels * 2, D)

        # 上下文投影：将 ctx_channels 映射到 D
        self.proj_ctx = nn.Linear(ctx_channels, D)

        # ---------- 位置编码（可选） ----------
        if self.use_position_encoding:
            self.pos_embed_t = nn.Parameter(torch.zeros(1, self.base_H * self.base_W, D))
            self.pos_embed_ctx = nn.Parameter(torch.zeros(1, self.base_H * self.base_W, D))
            nn.init.normal_(self.pos_embed_t, std=0.02)
            nn.init.normal_(self.pos_embed_ctx, std=0.02)
        else:
            self.pos_embed_t = None
            self.pos_embed_ctx = None

        # ---------- 时间步条件（可选） ----------
        self.t_embed = TimestepMLP(256, D) if self.use_timestep_conditioning else None

        # ---------- 提前期(lead time)条件（可选） ----------
        # 业界常见做法：将预测提前期作为额外条件输入网络（例如连续提前期/多提前期训练）。
        # 这里与扩散噪声时间步 t 的处理一致：使用 Fourier embedding + MLP 投影到 [B,D]。
        self.lt_embed = TimestepMLP(256, D) if self.use_lead_time_conditioning else None

        # ---------- DropPath 概率（线性递增） ----------
        dpr = [x.item() for x in torch.linspace(0, float(drop_path_rate), int(depth) + 2)]

        # ---------- 上下文编码器（2 层） ----------
        self.ctx_encoder = nn.ModuleList(
            [
                HybridAttentionBlock(
                    D,
                    heads,
                    mlp_ratio=mlp_ratio,
                    drop=float(dropout),
                    drop_path=dpr[i],
                    use_hybrid=use_hybrid_attention,
                    spatial_hw=(self.base_H, self.base_W),
                )
                for i in range(2)
            ]
        )

        # ---------- 主干 Transformer ----------
        if self.use_cross_attention:
            self.blocks = nn.ModuleList(
                [
                    CrossAttentionBlock(
                        D,
                        heads,
                        mlp_ratio=mlp_ratio,
                        drop=float(dropout),
                        drop_path=dpr[i + 2],
                    )
                    for i in range(int(depth))
                ]
            )
        else:
            # 仅使用自注意力块（无 cross-attn）
            self.blocks = nn.ModuleList(
                [
                    HybridAttentionBlock(
                        D,
                        heads,
                        mlp_ratio=mlp_ratio,
                        drop=float(dropout),
                        drop_path=dpr[i + 2],
                        use_hybrid=use_hybrid_attention,
                        spatial_hw=(self.base_H, self.base_W),
                    )
                    for i in range(int(depth))
                ]
            )

        # ---------- 多尺度融合（可选） ----------
        self.multi_scale_fusion = MultiScaleFusion(D, num_scales=3) if self.use_multi_scale else None

        if self.micromet_coupling is not None and self.use_micromet_token_adapter:
            self.micromet_token_to_state = nn.Sequential(
                nn.LayerNorm(D),
                nn.Linear(D, in_channels),
            )
            self.micromet_state_to_token = nn.Sequential(
                nn.LayerNorm(in_channels),
                nn.Linear(in_channels, D),
            )
            nn.init.zeros_(self.micromet_state_to_token[-1].weight)
            nn.init.zeros_(self.micromet_state_to_token[-1].bias)

        # ---------- V5.3 morphology-derived process proxy modules ----------
        proxy_cfg = dict(self.morpho_process_cfg.get("proxy_encoder", {}) or {})
        if self.use_process_proxy_encoder and self.real_static > 0:
            self.process_proxy_encoder = MorphologyProcessProxyEncoder(
                static_channels=int(proxy_cfg.get("static_channels", self.real_static)),
                hidden_channels=tuple(proxy_cfg.get("hidden_channels", (64, 128, 256))),
                categorical_scale=float(proxy_cfg.get("categorical_scale", 20.0)),
                dropout=float(proxy_cfg.get("dropout", dropout)),
                enabled=bool(proxy_cfg.get("enabled", True)),
                use_diurnal_modulation=bool(proxy_cfg.get("use_diurnal_modulation", True)),
                proxy_init=str(proxy_cfg.get("proxy_init", "zero")),
                init_gate=float(proxy_cfg.get("init_gate", 0.0)),
            )
        else:
            self.process_proxy_encoder = None
        self.last_process_proxy_diagnostics: Dict[str, torch.Tensor] = {}
        self.last_process_proxy_fields: Optional[Dict[str, torch.Tensor]] = None

        adaln_cfg = dict(self.morpho_process_cfg.get("process_adaln", {}) or {})
        self.process_adaln = (
            ProcessConditionedAdaLN(
                dim=D,
                depth=int(depth),
                hidden_dim=int(adaln_cfg.get("proxy_proj_dim", max(D // 2, 32))),
                init_alpha=float(adaln_cfg.get("init_alpha", 0.0)),
                enabled=True,
            )
            if self.use_process_adaln
            else None
        )
        self.last_process_adaln_diagnostics: Dict[str, torch.Tensor] = {}

        control_cfg = dict(self.morpho_process_cfg.get("urban_control_branch", {}) or {})
        feature_channels = tuple(proxy_cfg.get("hidden_channels", (64, 128, 256)))
        self.urban_control_branch = (
            UrbanMorphologyControlBranch(
                feature_channels=feature_channels,
                dim=D,
                depth=int(depth),
                time_dim=D,
                enabled=bool(control_cfg.get("enabled", True)),
                init_gate=float(control_cfg.get("init_gate", 0.0)),
            )
            if self.use_urban_control_branch
            else None
        )
        self.last_urban_control_diagnostics: Dict[str, torch.Tensor] = {}

        process_graph_cfg = dict(self.morpho_process_cfg.get("anisotropic_process_graph", {}) or {})
        self.anisotropic_process_graph = (
            AnisotropicUrbanProcessGraph(
                dim=D,
                k=int(process_graph_cfg.get("k", self.morphology_graph_cfg.get("k", self.urban_graph_k))),
                morphology_sigma=float(process_graph_cfg.get("morphology_sigma", 1.0)),
                spatial_sigma=float(process_graph_cfg.get("spatial_sigma", 1.0)),
                wind_temperature=float(process_graph_cfg.get("wind_temperature", 0.25)),
                wind_strength=float(process_graph_cfg.get("wind_strength_base", 1.0)),
                roughness_blocking_strength=float(process_graph_cfg.get("roughness_blocking_base", 0.5)),
                enable_cache=bool(process_graph_cfg.get("enable_cache", False)),
            )
            if self.use_anisotropic_process_graph
            else None
        )
        self.last_process_graph_diagnostics: Dict[str, torch.Tensor] = {}

        # ---------- 输出头 ----------
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, out_channels)
        self.morphology_residual_head = (
            MorphologyResidualPredictionHead(D, out_channels, proxy_channels=7)
            if self.use_morphology_residual_head
            else None
        )
        self.last_forward_diagnostics: Dict[str, Any] = {}

    def _get_pos_t(self, H: int, W: int) -> Optional[torch.Tensor]:
        if not self.use_position_encoding or self.pos_embed_t is None:
            return None
        return _resize_pos_embed(self.pos_embed_t, (self.base_H, self.base_W), (H, W))

    def _get_pos_ctx(self, H: int, W: int) -> Optional[torch.Tensor]:
        if not self.use_position_encoding or self.pos_embed_ctx is None:
            return None
        return _resize_pos_embed(self.pos_embed_ctx, (self.base_H, self.base_W), (H, W))

    def _extract_graph_static_features(
        self,
        x_ctx: torch.Tensor,
        static_raw: Optional[torch.Tensor],
        static_cont: Optional[torch.Tensor],
        static_cat: Optional[torch.Tensor],
        spatial_hw: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        height, width = int(spatial_hw[0]), int(spatial_hw[1])
        if static_raw is not None and static_raw.numel() > 0 and static_raw.ndim == 4:
            feat = static_raw.to(device=x_ctx.device, dtype=x_ctx.dtype)
            if feat.shape[-2:] != (height, width):
                feat = F.interpolate(feat, size=(height, width), mode="nearest")
            return feat

        candidates = [static_cont]
        if static_cat is not None and static_cat.numel() > 0:
            candidates.append(static_cat.float())

        valid = []
        for item in candidates:
            if item is None or item.numel() == 0:
                continue
            if item.ndim != 4:
                continue
            feat = item.to(device=x_ctx.device, dtype=x_ctx.dtype)
            if feat.shape[-2:] != (height, width):
                feat = F.interpolate(feat, size=(height, width), mode="nearest")
            valid.append(feat)

        if valid:
            return torch.cat(valid, dim=1)
        if self.real_static > 0 and x_ctx.size(1) >= self.real_static:
            return x_ctx[:, -self.real_static :, :, :]
        return None

    def _extract_context_wind(self, x_ctx: torch.Tensor, spatial_hw: Tuple[int, int]) -> Optional[torch.Tensor]:
        if not (
            self.use_wind_aware_graph
            or self.use_micromet_coupling
            or self.use_urban_canopy_coupling
            or self.use_anisotropic_process_graph
        ):
            return None
        if self.dynamic_vars:
            try:
                u_idx = self.dynamic_vars.index("u10")
                v_idx = self.dynamic_vars.index("v10")
            except ValueError:
                return None
        else:
            if self.in_channels < 7:
                return None
            u_idx, v_idx = 5, 6

        u_channel = u_idx * self.real_k + (self.real_k - 1)
        v_channel = v_idx * self.real_k + (self.real_k - 1)
        if max(u_channel, v_channel) >= x_ctx.size(1):
            return None
        wind = x_ctx[:, [u_channel, v_channel], :, :]
        height, width = int(spatial_hw[0]), int(spatial_hw[1])
        if wind.shape[-2:] != (height, width):
            wind = F.interpolate(wind, size=(height, width), mode="bilinear", align_corners=False)
        return wind

    def _apply_micromet_coupling(
        self,
        x: torch.Tensor,
        *,
        static_raw: Optional[torch.Tensor] = None,
        static_cont: Optional[torch.Tensor] = None,
        static_cat: Optional[torch.Tensor] = None,
        hour_of_day: Optional[torch.Tensor] = None,
        wind_uv: Optional[torch.Tensor] = None,
        diagnostic_prefix: str = "",
    ) -> torch.Tensor:
        if self.micromet_coupling is None:
            return x
        out, diagnostics = self.micromet_coupling(
            x,
            static_raw=static_raw,
            static_cont=static_cont,
            static_cat=static_cat,
            hour_of_day=hour_of_day,
            wind_uv=wind_uv,
        )
        if diagnostic_prefix:
            residual_norm = (out - x).detach().norm()
            diagnostics = dict(diagnostics)
            diagnostics[f"{diagnostic_prefix}_residual_norm"] = residual_norm
            prefixed = {f"{diagnostic_prefix}/{key}": value for key, value in diagnostics.items()}
            self.last_micromet_diagnostics.update(prefixed)
            self.last_micromet_diagnostics.update(diagnostics)
        else:
            self.last_micromet_diagnostics = diagnostics
        return out

    def _apply_token_micromet_coupling(
        self,
        tokens: torch.Tensor,
        *,
        spatial_hw: Tuple[int, int],
        static_raw: Optional[torch.Tensor] = None,
        static_cont: Optional[torch.Tensor] = None,
        static_cat: Optional[torch.Tensor] = None,
        hour_of_day: Optional[torch.Tensor] = None,
        wind_uv: Optional[torch.Tensor] = None,
        block_idx: Optional[int] = None,
    ) -> torch.Tensor:
        if self.micromet_coupling is None or self.micromet_token_to_state is None or self.micromet_state_to_token is None:
            return tokens
        batch, nodes, channels = tokens.shape
        height, width = int(spatial_hw[0]), int(spatial_hw[1])
        if nodes != height * width:
            raise ValueError(f"token 数 N={nodes} 与 spatial_hw={spatial_hw} 不匹配")
        state = self.micromet_token_to_state(tokens).transpose(1, 2).reshape(batch, self.in_channels, height, width)
        prefix = f"micromet_block_{int(block_idx):02d}" if block_idx is not None else "micromet_block"
        coupled_state = self._apply_micromet_coupling(
            state,
            static_raw=static_raw,
            static_cont=static_cont,
            static_cat=static_cat,
            hour_of_day=hour_of_day,
            wind_uv=wind_uv,
            diagnostic_prefix=prefix,
        )
        state_delta = (coupled_state - state).flatten(2).transpose(1, 2)
        token_delta = self.micromet_state_to_token(state_delta)
        out = tokens + token_delta
        self.last_micromet_diagnostics[f"{prefix}_token_residual_norm"] = token_delta.detach().norm()
        return out

    def _diagnostic_value(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            detached = value.detach()
            if detached.numel() == 1:
                return detached.to(device="cpu").item()
            return {
                "shape": list(detached.shape),
                "mean": detached.float().mean().to(device="cpu").item(),
                "std": detached.float().std(unbiased=False).to(device="cpu").item(),
                "norm": detached.float().norm().to(device="cpu").item(),
            }
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [self._diagnostic_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._diagnostic_value(item) for key, item in value.items()}
        return str(value)

    def _extend_diagnostics(
        self,
        target: Dict[str, Any],
        source: Optional[Dict[str, Any]],
        *,
        prefix: str = "",
    ) -> None:
        if not source:
            return
        for key, value in source.items():
            name = str(key)
            if prefix and not name.startswith(prefix):
                name = f"{prefix}{name}"
            target[name] = self._diagnostic_value(value)

    def _collect_diagnostics(
        self,
        y: torch.Tensor,
        *,
        x_t: torch.Tensor,
        x_ctx: torch.Tensor,
        static_raw: Optional[torch.Tensor],
        static_cont: Optional[torch.Tensor],
        static_cat: Optional[torch.Tensor],
        hour_of_day: Optional[torch.Tensor],
        lead_time: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {
            "version/version": VERSION,
            "version/version_name": VERSION_NAME,
            "version/method_family": METHOD_FAMILY,
            "version/claim_boundary": CLAIM_BOUNDARY,
            "model/use_process_proxy_encoder": self.use_process_proxy_encoder,
            "model/use_process_adaln": self.use_process_adaln,
            "model/use_urban_control_branch": self.use_urban_control_branch,
            "model/use_anisotropic_process_graph": self.use_anisotropic_process_graph,
            "model/use_micromet_coupling": self.use_micromet_coupling,
            "model/use_micromet_token_branches": self.use_micromet_token_branches,
            "model/use_morphology_residual_head": self.use_morphology_residual_head,
            "model/use_proxy_conditioned_vg": self.use_proxy_conditioned_vg,
            "model/vg_num_heads": self.vg_num_heads,
            "model/degrade_to_v52": self.degrade_to_v52,
            "input/x_t_shape": list(x_t.shape),
            "input/x_ctx_shape": list(x_ctx.shape),
            "input/static_raw_available": bool(static_raw is not None and static_raw.numel() > 0),
            "input/static_cont_available": bool(static_cont is not None and static_cont.numel() > 0),
            "input/static_cat_available": bool(static_cat is not None and static_cat.numel() > 0),
            "input/hour_of_day_available": bool(hour_of_day is not None),
            "input/lead_time_available": bool(lead_time is not None),
            "output/shape": list(y.shape),
            "output/mean": y.detach().float().mean().to(device="cpu").item(),
            "output/std": y.detach().float().std(unbiased=False).to(device="cpu").item(),
            "output/norm": y.detach().float().norm().to(device="cpu").item(),
            "output/finite": bool(torch.isfinite(y.detach()).all().to(device="cpu").item()),
            "leakage/target_tensor_provided": False,
            "leakage/future_frame_access_declared": False,
        }
        self._extend_diagnostics(diagnostics, self.last_process_proxy_diagnostics)
        self._extend_diagnostics(diagnostics, self.last_process_adaln_diagnostics)
        self._extend_diagnostics(diagnostics, self.last_urban_control_diagnostics)
        self._extend_diagnostics(diagnostics, self.last_process_graph_diagnostics, prefix="process_graph/")
        self._extend_diagnostics(diagnostics, self.last_micromet_diagnostics, prefix="micromet/")
        self._extend_diagnostics(diagnostics, self.last_urban_canopy_diagnostics, prefix="urban_canopy/")
        self._extend_diagnostics(diagnostics, self.last_morphology_graph_diagnostics, prefix="morphology_graph/")
        self._extend_diagnostics(diagnostics, self.last_variable_graph_diagnostics)
        if self.morphology_residual_head is not None:
            self._extend_diagnostics(diagnostics, self.morphology_residual_head.last_diagnostics)
        diagnostics["model/diagnostic_key_count"] = len(diagnostics)
        diagnostics["model/diagnostic_prefixes_observed"] = sorted({key.split("/", 1)[0] for key in diagnostics})
        self.last_forward_diagnostics = diagnostics
        return diagnostics

    def _build_process_proxy_context(
        self,
        ref: torch.Tensor,
        *,
        static_raw: Optional[torch.Tensor] = None,
        static_cont: Optional[torch.Tensor] = None,
        static_cat: Optional[torch.Tensor] = None,
        hour_of_day: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[Dict[str, torch.Tensor]], Optional[list[torch.Tensor]]]:
        if self.process_proxy_encoder is None:
            self.last_process_proxy_diagnostics = {}
            self.last_process_proxy_fields = None
            return None, None
        out = self.process_proxy_encoder(
            ref,
            static_raw=static_raw,
            static_cont=static_cont,
            static_cat=static_cat,
            hour_of_day=hour_of_day,
        )
        self.last_process_proxy_diagnostics = out.diagnostics
        self.last_process_proxy_fields = out.proxy_fields
        return out.proxy_fields, out.level_features

    def _apply_process_graph(
        self,
        tokens: torch.Tensor,
        *,
        proxy_fields: Optional[Dict[str, torch.Tensor]],
        static_raw: Optional[torch.Tensor],
        static_cont: Optional[torch.Tensor],
        static_cat: Optional[torch.Tensor],
        wind_uv: Optional[torch.Tensor],
        diffusion_t: torch.Tensor,
        spatial_hw: Tuple[int, int],
    ) -> torch.Tensor:
        if self.anisotropic_process_graph is None:
            return tokens
        roughness = None if proxy_fields is None else proxy_fields.get("roughness_proxy")
        out, diagnostics = self.anisotropic_process_graph(
            tokens,
            static_feat=static_raw,
            static_cont=static_cont,
            static_cat=static_cat,
            wind_uv=wind_uv,
            roughness_proxy=roughness,
            diffusion_t=diffusion_t,
            spatial_hw=spatial_hw,
            return_diagnostics=True,
        )
        self.last_process_graph_diagnostics = diagnostics
        return out

    def _apply_morphology_residual_head(
        self,
        y: torch.Tensor,
        tokens: torch.Tensor,
        proxy_fields: Optional[Dict[str, torch.Tensor]],
        spatial_hw: Tuple[int, int],
    ) -> torch.Tensor:
        if self.morphology_residual_head is None or proxy_fields is None:
            return y
        residual = self.morphology_residual_head(tokens, proxy_fields, spatial_hw)
        return y + residual

    def forward(
        self,
        x_t: torch.Tensor,
        x_ctx: torch.Tensor,
        t: torch.Tensor,
        lead_time: Optional[torch.Tensor] = None,
        static_raw: Optional[torch.Tensor] = None,
        static_cont: Optional[torch.Tensor] = None,
        static_cat: Optional[torch.Tensor] = None,
        hour_of_day: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """前向传播。

        默认返回预测张量；显式开启 return_diagnostics 时返回审稿证据诊断。

        Args:
            x_t: 噪声输入 [B, C, H, W]
            x_ctx: 上下文（历史帧 + 静态变量）[B, ctx_C, H, W]
            t: 扩散时间步 [B]

        Returns:
            x0_pred 或 (x0_pred, diagnostics)
        """

        b = x_t.size(0)
        H = int(x_t.size(2))
        W = int(x_t.size(3))
        self.last_process_adaln_diagnostics = {}
        self.last_urban_control_diagnostics = {}
        self.last_process_graph_diagnostics = {}
        self.last_micromet_diagnostics = {}
        self.last_urban_canopy_diagnostics = {}
        self.last_morphology_graph_diagnostics = {}
        self.last_variable_graph_diagnostics = {}
        self.last_forward_diagnostics = {}
        if self.morphology_residual_head is not None:
            self.morphology_residual_head.last_diagnostics = {}

        if self.static_morphology_encoder is not None and self.real_static > 0:
            has_static_raw = static_raw is not None and static_raw.numel() > 0
            has_static_cont = static_cont is not None and static_cont.numel() > 0
            has_static_cat = static_cat is not None and static_cat.numel() > 0
            if not (has_static_raw or has_static_cont or has_static_cat):
                static_raw = x_ctx[:, -self.real_static :, :, :]

            static_encoded = self.static_morphology_encoder(
                static_raw=static_raw,
                static_cont=static_cont,
                static_cat=static_cat,
                spatial_hw=(H, W),
            )
            if static_encoded.shape[-2:] != (H, W):
                static_encoded = F.interpolate(static_encoded, size=(H, W), mode="bilinear", align_corners=False)
            dyn_ctx = x_ctx[:, : self.in_channels * self.real_k, :, :]
            x_ctx = torch.cat([dyn_ctx, static_encoded.to(dtype=x_ctx.dtype, device=x_ctx.device)], dim=1)

        # 提取每个变量的上一帧（last frame）用于条件输入
        # 注意：要求 x_ctx 的动态通道顺序为「变量优先」。
        x_last = x_ctx[:, self.last_indices, :, :]
        static_for_coupling = self._extract_graph_static_features(
            x_ctx=x_ctx,
            static_raw=static_raw,
            static_cont=static_cont,
            static_cat=static_cat,
            spatial_hw=(H, W),
        )
        proxy_fields, proxy_level_features = self._build_process_proxy_context(
            x_last,
            static_raw=static_for_coupling,
            static_cont=static_cont,
            static_cat=static_cat,
            hour_of_day=hour_of_day,
        )
        context_wind = self._extract_context_wind(x_ctx, spatial_hw=(H, W))
        if self.urban_canopy is not None:
            x_last, diagnostics = self.urban_canopy(
                x_last,
                static_for_coupling,
                static_raw=static_raw,
                static_cont=static_cont,
                static_cat=static_cat,
                hour_of_day=hour_of_day,
                wind_uv=context_wind,
                return_diagnostics=True,
            )
            self.last_urban_canopy_diagnostics = diagnostics
        if self.micromet_coupling is not None and self.micromet_coupling_mode in {"pre", "pre_transformer"}:
            x_last = self._apply_micromet_coupling(
                x_last,
                static_raw=static_for_coupling,
                static_cont=static_cont,
                static_cat=static_cat,
                hour_of_day=hour_of_day,
                wind_uv=context_wind,
            )
        qt_input = torch.cat([x_t, x_last], dim=1)  # [B, 2C, H, W]

        # 变量关系交互（可选）
        if self.use_dynamic_vg and (self.var_graph_t is not None):
            qt_input = self.var_graph_t(qt_input)

        if self.use_proxy_conditioned_vg and (self.proxy_cond_vg is not None):
            qt_input = self.proxy_cond_vg(qt_input, proxy_fields=proxy_fields)
            self.last_variable_graph_diagnostics = dict(self.proxy_cond_vg.last_diagnostics)

        # 目标场投影
        qt = qt_input.permute(0, 2, 3, 1).reshape(b, H * W, self.in_channels * 2)
        qt = self.proj_t(qt)  # [B, HW, D]

        # 位置编码（可选，支持插值）
        pos_t = self._get_pos_t(H, W)
        if pos_t is not None:
            qt = qt + pos_t.to(dtype=qt.dtype, device=qt.device)

        # -------------------------
        # 条件向量 c：可由“扩散时间步 t”与“预测提前期 lead_time”共同构成
        # -------------------------
        # 1) 扩散时间步条件（可选）
        if self.use_timestep_conditioning and (self.t_embed is not None):
            c = self.t_embed(t)  # [B, D]
        else:
            # 没有时间步条件时，用 0 向量作为条件（等价于不调制）
            c = torch.zeros(b, self.D, device=x_t.device, dtype=x_t.dtype)

        # 2) 预测提前期条件（可选）
        # lead_time 推荐传入归一化后的连续数值（例如：delta_step / max_step），也支持离散步长。
        if self.use_lead_time_conditioning and (self.lt_embed is not None) and (lead_time is not None):
            # 保证 shape=[B]
            if lead_time.ndim != 1 or lead_time.shape[0] != b:
                raise ValueError(
                    f"lead_time 必须是 shape=[B] 的 1D 张量，但得到 {tuple(lead_time.shape)} (B={b})"
                )
            c = c + self.lt_embed(lead_time.to(device=x_t.device))

        # 城市形态图传播（可选）
        if self.morphology_graph is not None:
            has_schema_static = (static_cont is not None and static_cont.numel() > 0) or (
                static_cat is not None and static_cat.numel() > 0
            )
            static_feat = None if has_schema_static else static_for_coupling
            if has_schema_static or (static_feat is not None and static_feat.numel() > 0):
                qt, graph_diag = self.morphology_graph(
                    qt,
                    static_feat=static_feat,
                    static_cont=static_cont if static_cont is not None and static_cont.numel() > 0 else None,
                    static_cat=static_cat if static_cat is not None and static_cat.numel() > 0 else None,
                    spatial_hw=(H, W),
                    wind_uv=context_wind,
                    return_diagnostics=True,
                )
                graph_diag["urbanpidit_schema_static_used"] = qt.new_tensor(float(has_schema_static))
                graph_diag["urbanpidit_static_feat_fallback"] = qt.new_tensor(float(not has_schema_static))
                self.last_morphology_graph_diagnostics = graph_diag

        qt = self._apply_process_graph(
            qt,
            proxy_fields=proxy_fields,
            static_raw=static_for_coupling,
            static_cont=static_cont,
            static_cat=static_cat,
            wind_uv=context_wind,
            diffusion_t=t,
            spatial_hw=(H, W),
        )

        control_residuals = []
        if self.urban_control_branch is not None:
            control_residuals = self.urban_control_branch(proxy_level_features, c)
            self.last_urban_control_diagnostics = self.urban_control_branch.last_diagnostics
        # 推荐位置：patch embedding (proj_t) 之后，Transformer 之前
        if self.use_urban_graph and (self.urban_graph is not None):
            # 1. 尝试从 x_ctx 中提取静态特征（假设位于最后 static_channels 个通道）
            # x_ctx: [B, ctx_C, H, W]
            # 静态变量数 = self.real_static
            if self.real_static > 0:
                # 提取静态特征：[B, S, H, W]
                static_feat = x_ctx[:, -self.real_static :, :, :]
                # 取 batch 中第一个样本构建图（假设静态特征对所有样本一致，或者广播）
                # [S, H, W] -> [H, W, S] -> [N, S]
                s_feat = static_feat[0].permute(1, 2, 0).reshape(H * W, -1)

                # 构建/更新 A_norm
                # 如果是训练阶段且 batch 较大，其实也可以只算一次；但为了支持 batch 间静态特征可能不同（虽然极少见），
                # 或者简单起见，这里实时计算（N=64很快）。
                # 也可以用 self.A_norm 缓存，只计算一次。
                if self.A_norm is None or self.A_norm.shape[0] != H * W:
                    A_norm = build_urban_knn_graph(s_feat, k=self.urban_graph_k, sigma=self.urban_graph_sigma)
                    self.register_buffer("A_norm", A_norm, persistent=False)

                # 执行传播
                # qt: [B, HW, D] -> [B, N, D]
                # A_norm: [N, N]
                qt = self.urban_graph(qt, self.A_norm)

        # 变量关系交互（可选，上下文）
        if self.use_static_vg and (self.var_graph_ctx is not None):
            x_ctx = self.var_graph_ctx(x_ctx)

        # 上下文投影
        kv = x_ctx.permute(0, 2, 3, 1).reshape(b, H * W, x_ctx.size(1))
        kv = self.proj_ctx(kv)  # [B, HW, D]

        # 位置编码（可选，支持插值）
        pos_ctx = self._get_pos_ctx(H, W)
        if pos_ctx is not None:
            kv = kv + pos_ctx.to(dtype=kv.dtype, device=kv.device)

        # 上下文编码
        for blk in self.ctx_encoder:
            kv = blk(kv, c, spatial_hw=(H, W))

        # 主干网络
        if self.use_cross_attention:
            for block_idx, blk in enumerate(self.blocks):
                qt = blk(
                    qt,
                    kv,
                    c,
                    process_adaln=self.process_adaln,
                    proxy_fields=proxy_fields,
                    block_idx=block_idx,
                )
                if block_idx < len(control_residuals):
                    qt = qt + control_residuals[block_idx].to(device=qt.device, dtype=qt.dtype)
                self.last_process_adaln_diagnostics.update(
                    getattr(self.process_adaln, "last_diagnostics", {}) if self.process_adaln is not None else {}
                )
                current_block = block_idx + 1
                if (
                    self.micromet_coupling is not None
                    and self.use_micromet_token_adapter
                    and current_block % self.micromet_coupling_interval == 0
                ):
                    qt = self._apply_token_micromet_coupling(
                        qt,
                        spatial_hw=(H, W),
                        static_raw=static_for_coupling,
                        static_cont=static_cont,
                        static_cat=static_cat,
                        hour_of_day=hour_of_day,
                        wind_uv=context_wind,
                        block_idx=current_block,
                    )
        else:
            for block_idx, blk in enumerate(self.blocks):
                qt = blk(
                    qt,
                    c,
                    spatial_hw=(H, W),
                    process_adaln=self.process_adaln,
                    proxy_fields=proxy_fields,
                    block_idx=block_idx,
                )
                if block_idx < len(control_residuals):
                    qt = qt + control_residuals[block_idx].to(device=qt.device, dtype=qt.dtype)
                self.last_process_adaln_diagnostics.update(
                    getattr(self.process_adaln, "last_diagnostics", {}) if self.process_adaln is not None else {}
                )
                current_block = block_idx + 1
                if (
                    self.micromet_coupling is not None
                    and self.use_micromet_token_adapter
                    and current_block % self.micromet_coupling_interval == 0
                ):
                    qt = self._apply_token_micromet_coupling(
                        qt,
                        spatial_hw=(H, W),
                        static_raw=static_for_coupling,
                        static_cont=static_cont,
                        static_cat=static_cat,
                        hour_of_day=hour_of_day,
                        wind_uv=context_wind,
                        block_idx=current_block,
                    )

        # 多尺度融合（可选）
        if self.use_multi_scale and (self.multi_scale_fusion is not None):
            qt_2d = qt.transpose(1, 2).reshape(b, self.D, H, W)
            qt_2d = self.multi_scale_fusion(qt_2d)
            qt = qt_2d.reshape(b, self.D, H * W).transpose(1, 2)

        # 输出头
        qt = self.norm(qt)
        y = self.head(qt)  # [B, HW, C]
        y = y.reshape(b, H, W, self.out_channels).permute(0, 3, 1, 2)
        y = self._apply_morphology_residual_head(y, qt, proxy_fields, (H, W))
        if self.micromet_coupling is not None and self.micromet_coupling_mode in {"post", "post_transformer"}:
            y = self._apply_micromet_coupling(
                y,
                static_raw=static_for_coupling,
                static_cont=static_cont,
                static_cat=static_cat,
                hour_of_day=hour_of_day,
                wind_uv=context_wind,
            )
        if return_diagnostics:
            diagnostics = self._collect_diagnostics(
                y,
                x_t=x_t,
                x_ctx=x_ctx,
                static_raw=static_raw,
                static_cont=static_cont,
                static_cat=static_cat,
                hour_of_day=hour_of_day,
                lead_time=lead_time,
            )
            return y, diagnostics
        return y


# ============================
# 自适应权重网络
# ============================


class AdaptiveWeightNet(nn.Module):
    """自适应权重学习网络。

    使用一个小型网络，根据各项损失的历史统计自动给出权重，避免手工调参。

    注意：这里输出 Softmax 权重，天然满足和为 1。
    """

    def __init__(self, num_weights: int, hidden_dim: int = 128):
        super().__init__()
        if num_weights <= 0:
            raise ValueError("num_weights 必须为正数")

        self.num_weights = int(num_weights)
        self.net = nn.Sequential(
            nn.Linear(self.num_weights, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_weights),
            nn.Softmax(dim=-1),
        )

        # 初始化为均匀权重（用 buffer 存 EMA 的损失历史）
        self.register_buffer("loss_history", torch.ones(self.num_weights) / self.num_weights)

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        """根据当前损失向量更新 EMA，并输出权重。"""

        if losses.ndim != 1 or losses.numel() != self.num_weights:
            raise ValueError(f"losses 必须是 shape=[{self.num_weights}]，但得到 {tuple(losses.shape)}")

        # EMA 更新（不参与梯度）
        self.loss_history = 0.9 * self.loss_history + 0.1 * losses.detach()

        # 归一化后送入权重网络
        normalized = self.loss_history / (self.loss_history.sum() + 1e-8)
        return self.net(normalized)


def build_model(config: ModelConfig) -> UrbanPiDiT:
    """根据 ModelConfig 构建 UrbanPiDiT。"""

    return UrbanPiDiT(
        H=config.H,
        W=config.W,
        in_channels=config.in_channels,
        ctx_channels=config.ctx_channels,
        out_channels=config.out_channels,
        D=config.D,
        depth=config.depth,
        heads=config.heads,
        mlp_ratio=config.mlp_ratio,
        static_channels=config.static_channels,
        drop_path_rate=config.drop_path_rate,
        dropout=config.dropout,
        use_hybrid_attention=config.use_hybrid_attention,
        use_multi_scale=config.use_multi_scale,
        use_cross_attention=config.use_cross_attention,
        use_position_encoding=config.use_position_encoding,
        use_timestep_conditioning=config.use_timestep_conditioning,
        use_lead_time_conditioning=config.use_lead_time_conditioning,
        use_variable_graph=config.use_variable_graph,
        use_dynamic_vg=config.use_dynamic_vg,
        use_static_vg=config.use_static_vg,
        use_proxy_conditioned_vg=config.use_proxy_conditioned_vg,
        vg_num_heads=config.vg_num_heads,
        use_static_morphology_encoder=config.use_static_morphology_encoder,
        use_morphology_graph=config.use_morphology_graph,
        use_wind_aware_graph=config.use_wind_aware_graph,
        morphology_graph_cfg=config.morphology_graph_cfg,
        use_urban_canopy_coupling=config.use_urban_canopy_coupling,
        use_urban_canopy=config.use_urban_canopy,
        canopy_coupling_mode=config.canopy_coupling_mode,
        urban_canopy_cfg=config.urban_canopy_cfg,
        use_micromet_coupling=config.use_micromet_coupling,
        micromet_coupling_cfg=config.micromet_coupling_cfg,
        use_process_proxy_encoder=config.use_process_proxy_encoder,
        use_process_adaln=config.use_process_adaln,
        use_urban_control_branch=config.use_urban_control_branch,
        use_anisotropic_process_graph=config.use_anisotropic_process_graph,
        use_micromet_token_branches=config.use_micromet_token_branches,
        use_morphology_residual_head=config.use_morphology_residual_head,
        degrade_to_v52=config.degrade_to_v52,
        morpho_process_cfg=config.morpho_process_cfg,
        use_urban_graph=config.use_urban_graph,
        urban_graph_k=config.urban_graph_k,
        urban_graph_sigma=config.urban_graph_sigma,
        dynamic_vars=config.dynamic_vars,
        static_vars=config.static_vars,
        static_schema=config.static_schema,
    )


if __name__ == "__main__":
    # 简单自测（不依赖真实数据）
    cfg = ModelConfig(
        H=8,
        W=8,
        in_channels=7,
        ctx_channels=33,
        out_channels=7,
        D=256,
        depth=6,
        heads=4,
        use_hybrid_attention=True,
        use_multi_scale=True,
        use_cross_attention=True,
        use_position_encoding=True,
        use_timestep_conditioning=True,
        use_urban_graph=True,
    )

    model = build_model(cfg)

    B = 2
    x_t = torch.randn(B, 7, 8, 8)
    x_ctx = torch.randn(B, 33, 8, 8)
    t = torch.rand(B)

    # 示例：lead_time 取 [0,1] 区间的连续值（例如 6h/24h=0.25）
    lead_time = torch.rand(B)

    y = model(x_t, x_ctx, t, lead_time=lead_time)

    print(f"输入形状: x_t={tuple(x_t.shape)}, x_ctx={tuple(x_ctx.shape)}, t={tuple(t.shape)}")
    print(f"输出形状: y={tuple(y.shape)}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
