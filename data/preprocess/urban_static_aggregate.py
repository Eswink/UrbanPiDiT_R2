"""Area-weighted aggregation of native urban static rasters to the R7 0.25 deg grid.

Decision 0063 / plan 0027 / ``docs/DATA_SPEC.md`` L2. The R7 frozen grid is 65x65
at 0.25 deg over 27-43N / 107-123E, i.e. cell **edges** 26.875-43.125N /
106.875-123.125E. Row 0 is the northernmost row, matching the weather instance's
descending ``latitude``.

Both native grids used here align **exactly** with those edges, so aggregation
needs no resampling to align: GHSL ``4326_3ss`` cells are 3 arcsec (0.25 deg = 300
native cells) and ESA WorldCover cells are 1/12000 deg (0.25 deg = 3000 native
cells). Each native cell therefore falls wholly inside exactly one target cell.

The output is a **derived statistic of real observations**, not a truth field. It
is not interpolation, and it is not "upsampling a coarse field into a fine truth".
Area weights come from the WGS84 authalic sphere (``R = 6371007.181 m``), which is
the equal-area basis: a native lat/lon cell's weight is
``R^2 * dlon_rad * (sin(lat_top) - sin(lat_bottom))``. That is the same as placing
the grid on a cylindrical equal-area projection and weighting by projected area,
which is what ``spatial_resampling = area_weighted_coarsen`` means here.

Per role:
- ``land_cover``     : class area fraction per target cell.
- ``built_surface``  : sum of built m^2 divided by the target cell area in m^2.
- ``building_height``: area-weighted mean over native cells that carry a value.
- ``population``     : sum of population counts divided by the cell area in km^2.
"""
from __future__ import annotations

import math

import numpy as np

from data.download.urban_static_sources import (
    EDGE_EAST,
    EDGE_NORTH,
    EDGE_SOUTH,
    EDGE_WEST,
    GRID_HEIGHT,
    GRID_SPACING_DEG,
    GRID_WIDTH,
)

AUTHALIC_RADIUS_M = 6371007.181          # WGS84 authalic sphere radius
SPATIAL_RESAMPLING = "area_weighted_coarsen"

def target_lat_bounds(row: int) -> tuple[float, float]:
    """(south, north) edges of target row ``row``; row 0 is northernmost."""
    north = EDGE_NORTH - GRID_SPACING_DEG * row
    return north - GRID_SPACING_DEG, north


def target_lon_bounds(col: int) -> tuple[float, float]:
    """(west, east) edges of target column ``col``."""
    west = EDGE_WEST + GRID_SPACING_DEG * col
    return west, west + GRID_SPACING_DEG


def sin_band(lat_south: float, lat_north: float) -> float:
    """Exact equal-area weight of a lat/lon band (dimensionless, in sin units)."""
    return math.sin(math.radians(lat_north)) - math.sin(math.radians(lat_south))


def cell_area_m2(row: int) -> float:
    """Target cell area in m^2 on the WGS84 authalic sphere."""
    lat_south, lat_north = target_lat_bounds(row)
    lon_span = math.radians(GRID_SPACING_DEG)
    return AUTHALIC_RADIUS_M ** 2 * lon_span * sin_band(lat_south, lat_north)


def coverage_fraction(tile_bounds) -> np.ndarray:
    """Fraction of each target cell covered by the union of ``tile_bounds``.

    A published raster product may legitimately omit a tile (ESA WorldCover does
    not publish all-ocean 3 deg tiles), so the cells that tile would have covered
    are only partially mapped. This array exposes that instead of hiding it: the
    affected cells' statistics are computed over the covered part only, and the
    consumer can see the coverage. Nothing is synthesised for the gap.
    """
    out = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float64)
    for row in range(GRID_HEIGHT):
        lat_south, lat_north = target_lat_bounds(row)
        for col in range(GRID_WIDTH):
            lon_west, lon_east = target_lon_bounds(col)
            area = sin_band(lat_south, lat_north) * (lon_east - lon_west)
            covered = 0.0
            for west, south, east, north in tile_bounds:
                dlon = max(0.0, min(lon_east, east) - max(lon_west, west))
                dsin = max(0.0, sin_band(max(lat_south, south), min(lat_north, north)))
                covered += dlon * dsin
            out[row, col] = covered / area if area > 0 else 0.0
    return out


def _target_row_of_lat(lat: float) -> int:
    return int(math.floor((EDGE_NORTH - lat) / GRID_SPACING_DEG))


def _target_col_of_lon(lon: float) -> int:
    return int(math.floor((lon - EDGE_WEST) / GRID_SPACING_DEG))


def _native_row_weights(dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(sin-band weight per native row, lat_top, lat_bottom) for one raster.

    Row *assignment* is not decided here: a native row can straddle a target edge
    and must contribute a proportional share to both neighbouring target rows
    (decision 0071). Callers use ``_row_spans`` for that.
    """
    top = dataset.bounds.top
    dy = dataset.transform.e                 # negative
    rows = np.arange(dataset.height)
    lat_top = top + rows * dy
    lat_bottom = top + (rows + 1) * dy
    weights = np.sin(np.radians(lat_top)) - np.sin(np.radians(lat_bottom))
    return weights, lat_top, lat_bottom


def _sin_band_of_row(dataset, native_row: int) -> float:
    """Spherical sin-band weight of one native row (dimensionless)."""
    dy = dataset.transform.e
    top = dataset.bounds.top + native_row * dy
    return math.sin(math.radians(top)) - math.sin(math.radians(top + dy))


def _row_spans(lat_top: np.ndarray, lat_bottom: np.ndarray) -> list[tuple[int, int, float]]:
    """(target_row, native_row, overlap_fraction) triples for every native row.

    Decision 0071: assignment is by **cell-edge area overlap**, not by which target
    cell contains the native cell's centre. A native row that straddles a target
    edge appears twice, once per neighbour, with the fraction of its own spherical
    band that falls inside each. This is what ``DATA_SPEC.md`` calls
    ``area_weighted_coarsen``; the centre rule it replaces was measurably wrong
    because the GHSL grid sits half a native cell off the target edges.
    """
    spans: list[tuple[int, int, float]] = []
    for index in range(lat_top.size):
        top = float(lat_top[index])
        bottom = float(lat_bottom[index])
        band = math.sin(math.radians(top)) - math.sin(math.radians(bottom))
        if band <= 0.0:
            continue
        first = _target_row_of_lat(top - 1e-12)
        last = _target_row_of_lat(bottom + 1e-12)
        for row in range(min(first, last), max(first, last) + 1):
            if not 0 <= row < GRID_HEIGHT:
                continue
            south, north = target_lat_bounds(row)
            overlap = (math.sin(math.radians(min(top, north)))
                       - math.sin(math.radians(max(bottom, south))))
            if overlap <= 0.0:
                continue
            spans.append((row, index, overlap / band))
    return spans


def _col_spans(lon_left: np.ndarray, lon_right: np.ndarray) -> list[tuple[int, int, float]]:
    """(target_col, native_col, overlap_fraction) triples, lon axis of ``_row_spans``."""
    spans: list[tuple[int, int, float]] = []
    for index in range(lon_left.size):
        left = float(lon_left[index])
        right = float(lon_right[index])
        width = right - left
        if width <= 0.0:
            continue
        first = _target_col_of_lon(left + 1e-12)
        last = _target_col_of_lon(right - 1e-12)
        for col in range(min(first, last), max(first, last) + 1):
            if not 0 <= col < GRID_WIDTH:
                continue
            west, east = target_lon_bounds(col)
            overlap = min(right, east) - max(left, west)
            if overlap <= 0.0:
                continue
            spans.append((col, index, overlap / width))
    return spans


def _rasterio():
    """Import rasterio lazily.

    It is an optional data dependency (``requirements-data.txt``) that CI does not
    install, so importing this module must not require it: the pure geometry
    helpers below stay importable everywhere, and only the raster readers pull it in.
    """
    import rasterio
    return rasterio


def _native_col_edges(dataset) -> tuple[np.ndarray, np.ndarray]:
    """(lon_left, lon_right) per native column, in degrees."""
    left = dataset.bounds.left
    dx = dataset.transform.a                 # positive
    cols = np.arange(dataset.width)
    return left + cols * dx, left + (cols + 1) * dx


def _open(path):
    return _rasterio().open(path)


def _read(dataset, rows: slice, cols: slice) -> np.ndarray:
    from rasterio.windows import Window
    window = Window(cols.start, rows.start, cols.stop - cols.start, rows.stop - rows.start)
    return dataset.read(1, window=window)


def _accumulate_counts(dataset, row_spans, col_spans, *, classes: tuple[int, ...]) -> np.ndarray:
    """Exact area-weighted per-class counts, shape (65, 65, len(classes)).

    Every native cell contributes to each target cell it overlaps, scaled by the
    fraction of the native cell that falls inside that target cell (decision 0071).
    The latitude fraction is a ratio of spherical sin bands, so a native row that
    straddles a target edge is split in true area proportion rather than assigned
    whole to whichever cell happens to contain its centre.
    """
    counts = np.zeros((GRID_HEIGHT, GRID_WIDTH, len(classes)), dtype=np.float64)
    block = _read(dataset, slice(0, dataset.height), slice(0, dataset.width))
    for row, native_row, row_fraction in row_spans:
        # ``row_fraction`` is a proportion *of this native row's sin band*, so
        # multiplying by that band yields the exact spherical band this row
        # contributes to the target cell. Without the band factor the count would be
        # in pixel units and the fraction would be off by the cell area.
        band_share = _sin_band_of_row(dataset, native_row) * row_fraction
        line = block[native_row]
        for col, native_col, col_fraction in col_spans:
            value = line[native_col]
            share = band_share * col_fraction
            for position, code in enumerate(classes):
                if value == code:
                    counts[row, col, position] += share
    return counts


def _accumulate_sums(dataset, row_spans, col_spans, *, valid=None,
                     weight_by_area: bool) -> np.ndarray:
    """Value sums and weight totals, shape (65, 65, 2), split by edge overlap.

    ``weight_by_area`` distinguishes an **intensive** quantity (a mean, which must
    be area-weighted because a native cell's area depends on its latitude) from an
    **extensive** one (an absolute area or count, which is simply summed). Applying
    the area weight to an extensive quantity would scale it by the sin band and is
    exactly the bug this flag exists to prevent.
    """
    totals = np.zeros((GRID_HEIGHT, GRID_WIDTH, 2), dtype=np.float64)
    block = _read(dataset, slice(0, dataset.height), slice(0, dataset.width)).astype(np.float64)
    mask = None if valid is None else valid(block)
    values = block if mask is None else np.where(mask, block, 0.0)
    carried = np.ones(block.shape, dtype=np.float64) if mask is None else mask.astype(np.float64)
    for row, native_row, row_fraction in row_spans:
        if weight_by_area:
            # The row fraction already carries the lat proportion of this native row;
            # multiplying by its sin band converts a mean into an area-weighted mean.
            row_weight = _sin_band_of_row(dataset, native_row)
        else:
            row_weight = 1.0
        line = values[native_row]
        line_carried = carried[native_row]
        for col, native_col, col_fraction in col_spans:
            share = row_weight * row_fraction * col_fraction
            totals[row, col, 0] += share * float(line[native_col])
            totals[row, col, 1] += share * float(line_carried[native_col])
    return totals


def aggregate_land_cover(paths, *, classes=(10, 40, 50, 60, 80)) -> dict:
    """Class area fraction **of the target cell**, per class code.

    The denominator is the whole cell area, not the sum of the counted classes: a
    class that is not in ``classes`` (grassland, shrubland, wetland, ...) or a
    nodata pixel must not inflate the share of the counted ones. The share of the
    cell that is mapped at all is therefore the sum of the returned fractions, and
    ``land_cover_coverage_frac_*`` separately reports the share covered by the
    downloaded tiles.
    """
    counts = np.zeros((GRID_HEIGHT, GRID_WIDTH, len(classes)), dtype=np.float64)
    cols_per_cell = None
    for path in paths:
        with _open(path) as dataset:
            if cols_per_cell is None:
                cols_per_cell = int(round(GRID_SPACING_DEG / dataset.transform.a))
            _, lat_top, lat_bottom = _native_row_weights(dataset)
            counts += _accumulate_counts(dataset, _row_spans(lat_top, lat_bottom),
                                         _col_spans(*_native_col_edges(dataset)),
                                         classes=classes)
    if cols_per_cell is None:
        raise ValueError("aggregate_land_cover needs at least one raster")
    denominator = np.array(
        [[sin_band(*target_lat_bounds(row)) * cols_per_cell for _ in range(GRID_WIDTH)]
         for row in range(GRID_HEIGHT)])
    fraction = np.divide(counts, denominator[:, :, None],
                         out=np.zeros_like(counts), where=denominator[:, :, None] > 0)
    mapped = counts.sum(axis=2) / np.where(denominator > 0, denominator, np.nan)
    return {"counts": counts, "cell_area_weight": denominator, "mapped_fraction": mapped,
            "fractions": {code: fraction[:, :, i] for i, code in enumerate(classes)}}


def aggregate_built_surface(paths) -> dict:
    """Sum of built m^2 divided by the target cell area in m^2."""
    totals = np.zeros((GRID_HEIGHT, GRID_WIDTH, 2), dtype=np.float64)
    for path in paths:
        with _open(path) as dataset:
            _, lat_top, lat_bottom = _native_row_weights(dataset)
            totals += _accumulate_sums(dataset, _row_spans(lat_top, lat_bottom),
                                       _col_spans(*_native_col_edges(dataset)),
                                       weight_by_area=False)
    area = np.array([[cell_area_m2(row) for _ in range(GRID_WIDTH)]
                     for row in range(GRID_HEIGHT)])
    fraction = np.divide(totals[:, :, 0], area, out=np.zeros_like(area), where=area > 0)
    return {"built_m2": totals[:, :, 0], "cell_area_m2": area, "fraction": fraction}


def aggregate_building_height(paths) -> dict:
    """Area-weighted mean of ANBH over native cells carrying a value."""
    totals = np.zeros((GRID_HEIGHT, GRID_WIDTH, 2), dtype=np.float64)
    for path in paths:
        with _open(path) as dataset:
            _, lat_top, lat_bottom = _native_row_weights(dataset)
            nodata = dataset.nodata
            if nodata is None:
                # The 3 arcsec product documents an empty NoData: zero means "no
                # built surface", so it carries no height and gets no weight.
                valid = lambda block: block > 0.0
            else:
                valid = lambda block: (block != nodata) & np.isfinite(block) & (block > 0.0)
            totals += _accumulate_sums(dataset, _row_spans(lat_top, lat_bottom),
                                       _col_spans(*_native_col_edges(dataset)),
                                       valid=valid, weight_by_area=True)
    weight = totals[:, :, 1]
    mean = np.divide(totals[:, :, 0], weight, out=np.zeros_like(weight), where=weight > 0)
    return {"weighted_height": totals[:, :, 0], "weight": weight, "mean": mean}


def aggregate_population(paths) -> dict:
    """Sum of population counts divided by the cell area in km^2."""
    totals = np.zeros((GRID_HEIGHT, GRID_WIDTH, 2), dtype=np.float64)
    for path in paths:
        with _open(path) as dataset:
            _, lat_top, lat_bottom = _native_row_weights(dataset)
            nodata = dataset.nodata
            valid = None if nodata is None else (
                lambda block: (block != nodata) & np.isfinite(block) & (block >= 0.0))
            totals += _accumulate_sums(dataset, _row_spans(lat_top, lat_bottom),
                                       _col_spans(*_native_col_edges(dataset)),
                                       valid=valid, weight_by_area=False)
    area_km2 = np.array([[cell_area_m2(row) / 1e6 for _ in range(GRID_WIDTH)]
                         for row in range(GRID_HEIGHT)])
    density = np.divide(totals[:, :, 0], area_km2, out=np.zeros_like(area_km2),
                        where=area_km2 > 0)
    return {"counts": totals[:, :, 0], "cell_area_km2": area_km2, "density": density}
