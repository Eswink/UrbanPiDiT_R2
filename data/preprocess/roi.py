from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class GeoROI:
    west: float; south: float; east: float; north: float
    def validate(self):
        if not (-180 <= self.west < self.east <= 180): raise ValueError('经度范围非法')
        if not (-90 <= self.south < self.north <= 90): raise ValueError('纬度范围非法')
        return self

JINGJINJI_COARSE = GeoROI(113.0, 38.0, 120.0, 42.0)
BEIJING_CACHE = GeoROI(115.2, 39.2, 117.8, 41.2)
