"""静态城市形态特征编码模块。

该模块把连续静态变量与类别静态变量分离编码，并保持输出通道数与旧版
`x_ctx` 静态尾部一致，便于在不改变训练/评估主流程的情况下接入形态编码。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_CATEGORICAL_STATIC_VARS = ("landcover",)


def _parse_static_schema(static_vars: Iterable[str], static_schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    ordered_vars = [str(v) for v in static_vars]
    schema = dict(static_schema or {})
    known = set(ordered_vars)

    categorical_cfg = schema.get("categorical", None)
    if categorical_cfg is None:
        categorical = []
        for name in ordered_vars:
            spec = schema.get(name, None)
            if isinstance(spec, dict) and str(spec.get("type", "")).lower() in {"cat", "categorical", "category"}:
                categorical.append(name)
        if not categorical:
            categorical = [name for name in ordered_vars if name in DEFAULT_CATEGORICAL_STATIC_VARS]
    else:
        categorical = [str(v) for v in categorical_cfg if str(v) in known]

    continuous_cfg = schema.get("continuous", None)
    if continuous_cfg is None:
        categorical_set = set(categorical)
        continuous = []
        for name in ordered_vars:
            spec = schema.get(name, None)
            if isinstance(spec, dict) and str(spec.get("type", "")).lower() in {"cont", "continuous", "numeric"}:
                continuous.append(name)
            elif name not in categorical_set:
                continuous.append(name)
    else:
        continuous = [str(v) for v in continuous_cfg if str(v) in known]

    categorical_set = set(categorical)
    continuous = [name for name in ordered_vars if name in set(continuous) and name not in categorical_set]
    categorical = [name for name in ordered_vars if name in categorical_set]

    raw_cardinality = schema.get("categorical_cardinality", schema.get("num_categories", schema.get("cardinality", {})))
    cardinality = dict(raw_cardinality) if isinstance(raw_cardinality, dict) else {}
    for name in categorical:
        spec = schema.get(name, None)
        if isinstance(spec, dict):
            if "num_classes" in spec:
                cardinality[name] = int(spec["num_classes"])
            elif "num_categories" in spec:
                cardinality[name] = int(spec["num_categories"])
            elif "cardinality" in spec:
                cardinality[name] = int(spec["cardinality"])
        cardinality.setdefault(name, 20 if name == "landcover" else 256)

    return {
        "continuous": continuous,
        "categorical": categorical,
        "categorical_cardinality": {name: int(cardinality[name]) for name in categorical if name in cardinality},
    }


class StaticFeatureEncoder(nn.Module):
    """连续/类别静态变量的空间编码器。"""

    def __init__(
        self,
        continuous_vars: Iterable[str],
        categorical_cardinality: Dict[str, int],
        out_channels: int,
        hidden_channels: Optional[int] = None,
        categorical_embed_dim: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.continuous_vars = [str(v) for v in continuous_vars]
        self.categorical_vars = [str(v) for v in categorical_cardinality.keys()]
        self.out_channels = int(out_channels)
        self.categorical_embed_dim = max(int(categorical_embed_dim), 1)

        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(max(int(num_classes), 1), self.categorical_embed_dim)
                for name, num_classes in categorical_cardinality.items()
            }
        )

        in_channels = len(self.continuous_vars) + len(self.categorical_vars) * self.categorical_embed_dim
        self.in_channels = int(in_channels)
        hidden = int(hidden_channels or max(self.out_channels * 2, 8))

        if self.in_channels > 0 and self.out_channels > 0:
            self.net = nn.Sequential(
                nn.Conv2d(self.in_channels, hidden, kernel_size=1),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Conv2d(hidden, self.out_channels, kernel_size=1),
            )
        else:
            self.net = None

    def forward(
        self,
        static_cont: Optional[torch.Tensor] = None,
        static_cat: Optional[torch.Tensor] = None,
        spatial_hw: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        parts = []
        device = None
        dtype = torch.float32
        batch_size = None
        height = None
        width = None

        if static_cont is not None and static_cont.numel() > 0:
            static_cont = static_cont.float()
            parts.append(static_cont)
            device = static_cont.device
            dtype = static_cont.dtype
            batch_size, _, height, width = static_cont.shape

        if static_cat is not None and static_cat.numel() > 0 and len(self.categorical_vars) > 0:
            static_cat = static_cat.long()
            device = static_cat.device
            batch_size, _, height, width = static_cat.shape
            for idx, name in enumerate(self.categorical_vars):
                if idx >= static_cat.size(1):
                    break
                emb = self.embeddings[name]
                x = static_cat[:, idx].clamp(min=0, max=emb.num_embeddings - 1)
                x = emb(x).permute(0, 3, 1, 2).contiguous()
                parts.append(x.to(dtype=dtype if parts else torch.float32))

        if not parts:
            if static_cont is not None:
                device = static_cont.device
                batch_size = int(static_cont.shape[0])
                if static_cont.ndim >= 4:
                    height, width = int(static_cont.shape[-2]), int(static_cont.shape[-1])
            if static_cat is not None:
                device = static_cat.device
                batch_size = int(static_cat.shape[0])
                if static_cat.ndim >= 4:
                    height, width = int(static_cat.shape[-2]), int(static_cat.shape[-1])
            if spatial_hw is None or batch_size is None or device is None:
                raise ValueError("StaticFeatureEncoder 缺少可推断空间尺寸的静态输入")
            height, width = int(spatial_hw[0]), int(spatial_hw[1])
            return torch.zeros((batch_size, self.out_channels, height, width), device=device, dtype=dtype)

        x = torch.cat(parts, dim=1)
        if self.net is None:
            return torch.zeros((x.size(0), self.out_channels, x.size(2), x.size(3)), device=x.device, dtype=x.dtype)
        return self.net(x)


class StaticMorphologyEncoder(nn.Module):
    """保持旧接口形状的城市静态形态编码器。"""

    def __init__(
        self,
        static_vars: Iterable[str],
        static_schema: Optional[Dict[str, Any]],
        out_channels: int,
        hidden_channels: Optional[int] = None,
        categorical_embed_dim: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.static_vars = [str(v) for v in static_vars]
        self.schema = _parse_static_schema(self.static_vars, static_schema)
        self.out_channels = int(out_channels)

        self.feature_encoder = StaticFeatureEncoder(
            continuous_vars=self.schema["continuous"],
            categorical_cardinality=self.schema["categorical_cardinality"],
            out_channels=self.out_channels,
            hidden_channels=hidden_channels,
            categorical_embed_dim=categorical_embed_dim,
            dropout=dropout,
        )
        self.residual_gate = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        static_raw: Optional[torch.Tensor] = None,
        static_cont: Optional[torch.Tensor] = None,
        static_cat: Optional[torch.Tensor] = None,
        spatial_hw: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        if static_raw is not None and static_raw.numel() > 0:
            static_raw = static_raw.float()
            spatial_hw = (int(static_raw.size(-2)), int(static_raw.size(-1)))

        if static_cont is None and static_cat is None:
            if static_raw is None:
                raise ValueError("StaticMorphologyEncoder 至少需要 static_raw 或 static_cont/static_cat")
            return static_raw

        encoded = self.feature_encoder(static_cont=static_cont, static_cat=static_cat, spatial_hw=spatial_hw)
        if static_raw is None or static_raw.numel() == 0:
            return encoded

        if encoded.shape[-2:] != static_raw.shape[-2:]:
            encoded = F.interpolate(encoded, size=static_raw.shape[-2:], mode="bilinear", align_corners=False)

        if static_raw.size(1) == self.out_channels:
            return static_raw + self.residual_gate * encoded
        return encoded