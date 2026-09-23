"""UrbanPiDiT V5.3.1 MorphoProcessDiT reviewer-evidence package.

该包提供城市微尺度气象预测模型、训练评估组件和审稿证据脚本。
V5.3.1 的重点是统一版本元数据、暴露 morphology-conditioned process diagnostics，
并提供可运行的 morphology counterfactual 诊断链路。
"""

try:
    from .urbanpidit_version import VERSION, VERSION_INFO
except ImportError:  # pragma: no cover
    from urbanpidit_version import VERSION, VERSION_INFO

from .models.urban_pidit import UrbanPiDiT, ModelConfig, AdaptiveWeightNet, build_model
from .pidit_lit import UrbanPiDiTLitModule
from .data.loader import BeijingWeatherDataset, MetroWeatherDataModule

__all__ = [
    "UrbanPiDiT",
    "ModelConfig",
    "AdaptiveWeightNet",
    "build_model",
    "UrbanPiDiTLitModule",
    "BeijingWeatherDataset",
    "MetroWeatherDataModule",
    "VERSION_INFO",
]

__version__ = VERSION
