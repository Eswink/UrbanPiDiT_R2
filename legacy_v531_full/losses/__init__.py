"""损失模块导出。"""

from __future__ import annotations

try:
    from .physical_consistency import (
        FeasibilityProxyLoss,
        PhysicalConsistencyConfig,
        PhysicalConsistencyLoss,
        ProcessConsistencyLoss,
        ProcessProxyConsistencyLoss,
        StructureProxyLoss,
    )
except ImportError:
    from physical_consistency import (
        FeasibilityProxyLoss,
        PhysicalConsistencyConfig,
        PhysicalConsistencyLoss,
        ProcessConsistencyLoss,
        ProcessProxyConsistencyLoss,
        StructureProxyLoss,
    )

__all__ = [
    "FeasibilityProxyLoss",
    "PhysicalConsistencyConfig",
    "PhysicalConsistencyLoss",
    "ProcessConsistencyLoss",
    "ProcessProxyConsistencyLoss",
    "StructureProxyLoss",
]