"""模型配置注册与安全构造入口。

该模块只做一件事：把统一配置构造器输出的模型配置，过滤成当前
`UrbanPiDiT` 构造器实际支持的参数，避免 legacy / 新实验段的兼容键
导致模型实例化失败。
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Mapping

try:
    from .urban_pidit import UrbanPiDiT
except ImportError:  # pragma: no cover
    from models.urban_pidit import UrbanPiDiT


def urban_pidit_allowed_kwargs() -> set[str]:
    """返回 `UrbanPiDiT.__init__` 当前接受的参数名。"""

    sig = inspect.signature(UrbanPiDiT.__init__)
    return {name for name in sig.parameters if name != "self"}


def filter_model_kwargs(model_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """过滤模型配置中的非构造参数。

    Args:
        model_cfg: 来自 `build_model_cfg` 或 YAML `model` 段合并后的配置。

    Returns:
        可安全传给 `UrbanPiDiT(**kwargs)` 的字典。
    """

    allowed = urban_pidit_allowed_kwargs()
    return {str(key): value for key, value in dict(model_cfg or {}).items() if str(key) in allowed}


def dropped_model_kwargs(model_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """返回被过滤掉的兼容/实验键，便于测试和诊断。"""

    allowed = urban_pidit_allowed_kwargs()
    return {str(key): value for key, value in dict(model_cfg or {}).items() if str(key) not in allowed}


def build_model_from_config(model_cfg: Mapping[str, Any]) -> UrbanPiDiT:
    """从配置安全构建 UrbanPiDiT。

    支持 V4 legacy、V5 canopy、V5.1 micromet 配置共用同一入口。
    未被当前模型构造器接收的键会被白名单过滤，不参与构造。
    """

    return UrbanPiDiT(**filter_model_kwargs(model_cfg))


def build_model_v53_from_config(model_cfg: Mapping[str, Any]) -> UrbanPiDiT:
    """从 V5.3 配置安全构建 UrbanPiDiT。

    该入口与通用入口等价，保留独立名称便于历史脚本显式引用。
    """

    return build_model_from_config(model_cfg)


def build_model_v531_from_config(model_cfg: Mapping[str, Any]) -> UrbanPiDiT:
    """从 V5.3.1 reviewer-evidence 配置安全构建 UrbanPiDiT。"""

    return build_model_from_config(model_cfg)


__all__ = [
    "build_model_from_config",
    "build_model_v53_from_config",
    "build_model_v531_from_config",
    "dropped_model_kwargs",
    "filter_model_kwargs",
    "urban_pidit_allowed_kwargs",
]