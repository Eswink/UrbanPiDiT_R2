r"""
论文忠实复现版本强基线。
"""

from __future__ import annotations

try:
    from .corrdiff import FaithfulCorrDiff
    from .fourcastnet import FaithfulFourCastNet
    from .gencast import FaithfulGenCast
    from .graphcast import FaithfulGraphCast
except ImportError:  # pragma: no cover - minimal install fallback
    FaithfulCorrDiff = None  # type: ignore
    FaithfulFourCastNet = None  # type: ignore
    FaithfulGenCast = None  # type: ignore
    FaithfulGraphCast = None  # type: ignore

__all__ = [
    "FaithfulCorrDiff",
    "FaithfulFourCastNet",
    "FaithfulGenCast",
    "FaithfulGraphCast",
]