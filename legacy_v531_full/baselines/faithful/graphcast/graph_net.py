r"""
GraphCast 风格图网络模块。
"""

from __future__ import annotations

from baselines.faithful.common.typed_graph import (
    BipartiteGraphNet,
    BipartiteInteractionBlock,
    GraphMLP,
    MeshGraphNet,
)

# 向后兼容旧 import 名称。
InteractionNetwork = BipartiteInteractionBlock

__all__ = [
    "BipartiteGraphNet",
    "BipartiteInteractionBlock",
    "GraphMLP",
    "InteractionNetwork",
    "MeshGraphNet",
]