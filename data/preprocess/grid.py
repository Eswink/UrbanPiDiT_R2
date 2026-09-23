"""Coordinate-only guards: no sorting, interpolation or data mutation."""
from __future__ import annotations
import numpy as np


def _axis(values, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size < 2 or raw.dtype.kind not in 'iuf':
        raise ValueError(f'{name} must be a numeric 1-D axis with at least two points')
    axis = raw.astype(np.float64)
    if not np.isfinite(axis).all():
        raise ValueError(f'{name} contains nonfinite coordinates')
    delta = np.diff(axis)
    if not (np.all(delta > 0) or np.all(delta < 0)):
        raise ValueError(f'{name} must be strictly monotone; folded/duplicate axes are invalid')
    return axis


def regular_latlon_spacing(latitude, longitude) -> float:
    """Validate a regular, non-wrapped lat/lon grid without changing its order.

    Either direction is allowed. Longitude must lie entirely in [-180, 180]
    or [0, 360), have span < 360, and not wrap across its convention's seam.
    A dateline-crossing ROI can use, for example, [179.75, 180, 180.25].
    [359.75, 0, 0.25] must first be converted *together with its data* by the
    caller. This function never guesses a convention or reorders fields.
    """
    lat = _axis(latitude, 'latitude')
    lon = _axis(longitude, 'longitude')
    if np.any(np.abs(lat) > 90):
        raise ValueError('latitude outside [-90, 90]')
    signed = np.all((lon >= -180) & (lon <= 180))
    positive = np.all((lon >= 0) & (lon < 360))
    if not (signed or positive) or np.ptp(lon) >= 360:
        raise ValueError('longitude must use one non-wrapped convention without a duplicate seam')
    lat_diff, lon_diff = np.abs(np.diff(lat)), np.abs(np.diff(lon))
    if not np.allclose(lat_diff, lat_diff[0], rtol=0, atol=1e-5):
        raise ValueError('latitude 非规则网格')
    if not np.allclose(lon_diff, lon_diff[0], rtol=0, atol=1e-5):
        raise ValueError('longitude 非规则网格')
    if not np.isclose(lat_diff[0], lon_diff[0], rtol=0, atol=1e-5):
        raise ValueError('非等经纬网格')
    return float((lat_diff[0] + lon_diff[0]) / 2)
