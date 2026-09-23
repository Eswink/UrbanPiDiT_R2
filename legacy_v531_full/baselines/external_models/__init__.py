"""Loader-compatible adapters for strong weather baselines.

The official FourCastNet and GraphCast/GenCast source trees are vendored under
``baselines/external_sources`` for provenance.  The adapter classes in this
package expose UrbanPiDiT's ``BeijingWeatherDataset`` batch format through the
same ``ForecastModelBase`` interface used by the existing fair-baseline runner.
"""

from .fourcastnet_adapter import UrbanFourCastNetAdapter
from .graphcast_adapter import UrbanGraphCastAdapter
from .gencast_adapter import UrbanGenCastAdapter
from .corrdiff_adapter import UrbanCorrDiffAdapter
from .urban_loader_adapter import UrbanBatchAdapter, UrbanForecastBatch

__all__ = [
    "UrbanBatchAdapter",
    "UrbanForecastBatch",
    "UrbanFourCastNetAdapter",
    "UrbanGraphCastAdapter",
    "UrbanGenCastAdapter",
    "UrbanCorrDiffAdapter",
]
