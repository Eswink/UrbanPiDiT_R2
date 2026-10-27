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
# Native rows read per chunk. The area weight is applied per native row regardless,
# so this only bounds memory; it never coarsens the weighting.
READ_CHUNK_ROWS = 2048


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


def _native_row_weights(dataset) -> tuple[np.ndarray, np.ndarray]:
    """(target_row per native row, sin weight per native row) for one raster."""
    top = dataset.bounds.top
    dy = dataset.transform.e                 # negative
    rows = np.arange(dataset.height)
    lat_top = top + rows * dy
    lat_bottom = top + (rows + 1) * dy
    weights = np.sin(np.radians(lat_top)) - np.sin(np.radians(lat_bottom))
    centers = (lat_top + lat_bottom) / 2.0
    target = np.floor((EDGE_NORTH - centers) / GRID_SPACING_DEG).astype(np.int64)
    return target, weights


def _native_col_targets(dataset) -> np.ndarray:
    left = dataset.bounds.left
    dx = dataset.transform.a                 # positive
    cols = np.arange(dataset.width)
    centers = left + (cols + 0.5) * dx
    return np.floor((centers - EDGE_WEST) / GRID_SPACING_DEG).astype(np.int64)


def _overlapping(dataset) -> tuple[list[int], list[int]]:
    """Target rows and columns this raster intersects, ascending."""
    lat_south, lat_north = sorted((dataset.bounds.bottom, dataset.bounds.top))
    lon_west, lon_east = sorted((dataset.bounds.left, dataset.bounds.right))
    first_row = max(0, _target_row_of_lat(lat_north - 1e-9))
    last_row = min(GRID_HEIGHT - 1, _target_row_of_lat(lat_south + 1e-9))
    first_col = max(0, _target_col_of_lon(lon_west + 1e-9))
    last_col = min(GRID_WIDTH - 1, _target_col_of_lon(lon_east - 1e-9))
    return list(range(first_row, last_row + 1)), list(range(first_col, last_col + 1))


def _rasterio():
    """Import rasterio lazily.

    It is an optional data dependency (``requirements-data.txt``) that CI does not
    install, so importing this module must not require it: the pure geometry
    helpers below stay importable everywhere, and only the raster readers pull it in.
    """
    import rasterio
    return rasterio


def _open(path):
    return _rasterio().open(path)


def _read(dataset, rows: slice, cols: slice) -> np.ndarray:
    from rasterio.windows import Window
    window = Window(cols.start, rows.start, cols.stop - cols.start, rows.stop - rows.start)
    return dataset.read(1, window=window)


def _blocked_runs(index: np.ndarray) -> list[tuple[int, int, int]]:
    """Runs of equal value: (value, start, stop) over a 1-D integer array."""
    if index.size == 0:
        return []
    change = np.flatnonzero(index[1:] != index[:-1]) + 1
    bounds = np.concatenate(([0], change, [index.size]))
    return [(int(index[bounds[i]]), int(bounds[i]), int(bounds[i + 1]))
            for i in range(len(bounds) - 1)]


def _row_bands(target_rows: np.ndarray, row: int) -> list[tuple[int, int]]:
    """(start, stop) native-row runs whose target row is ``row``, chunked."""
    mask = np.flatnonzero(target_rows == row)
    if mask.size == 0:
        return []
    start, stop = int(mask[0]), int(mask[-1]) + 1
    return [(band_start, min(band_start + READ_CHUNK_ROWS, stop))
            for band_start in range(start, stop, READ_CHUNK_ROWS)]


def _accumulate_counts(dataset, col_targets: np.ndarray, target_rows: np.ndarray,
                       weights: np.ndarray, *, classes: tuple[int, ...]) -> np.ndarray:
    """Exact area-weighted per-class counts, shape (65, 65, len(classes)).

    The weight is applied **per native row** (``sin(lat_top) - sin(lat_bottom)`` of
    that row), not per read chunk: a chunk can straddle a sharp urban/rural
    boundary, and collapsing it to one weight would bias the cell. Only the
    requested class codes are counted, so a nodata code (WorldCover uses 0) is
    excluded from the numerator and from the cell-area denominator alike.
    """
    counts = np.zeros((GRID_HEIGHT, GRID_WIDTH, len(classes)), dtype=np.float64)
    for row in sorted(set(int(v) for v in target_rows)):
        if not 0 <= row < GRID_HEIGHT:
            continue
        for start, stop in _row_bands(target_rows, row):
            block = _read(dataset, slice(start, stop), slice(0, dataset.width))
            row_weights = weights[start:stop]
            for col, c0, c1 in _blocked_runs(col_targets):
                if not 0 <= col < GRID_WIDTH:
                    continue
                segment = block[:, c0:c1]
                for position, code in enumerate(classes):
                    per_row = (segment == code).sum(axis=1)
                    if per_row.any():
                        counts[row, col, position] += float((row_weights * per_row).sum())
    return counts


def _accumulate_sums(dataset, col_targets: np.ndarray, target_rows: np.ndarray,
                     weights: np.ndarray, *, valid=None, weight_by_area: bool) -> np.ndarray:
    """Value sums and weight totals, shape (65, 65, 2), weighted per native row.

    ``weight_by_area`` distinguishes an **intensive** quantity (a mean, which must
    be area-weighted because a native cell's area depends on its latitude) from an
    **extensive** one (an absolute area or count, which is simply summed). Applying
    the area weight to an extensive quantity would scale it by the sin band and is
    exactly the bug this flag exists to prevent.
    """
    totals = np.zeros((GRID_HEIGHT, GRID_WIDTH, 2), dtype=np.float64)
    for row in sorted(set(int(v) for v in target_rows)):
        if not 0 <= row < GRID_HEIGHT:
            continue
        for start, stop in _row_bands(target_rows, row):
            block = _read(dataset, slice(start, stop), slice(0, dataset.width)).astype(np.float64)
            row_weights = weights[start:stop] if weight_by_area else np.ones(stop - start)
            if valid is None:
                values = block
                carried = np.ones(block.shape, dtype=np.float64)
            else:
                mask = valid(block)
                values = np.where(mask, block, 0.0)
                carried = mask.astype(np.float64)
            for col, c0, c1 in _blocked_runs(col_targets):
                if not 0 <= col < GRID_WIDTH:
                    continue
                totals[row, col, 0] += float((row_weights * values[:, c0:c1].sum(axis=1)).sum())
                totals[row, col, 1] += float((row_weights * carried[:, c0:c1].sum(axis=1)).sum())
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
            target_rows, weights = _native_row_weights(dataset)
            counts += _accumulate_counts(dataset, _native_col_targets(dataset),
                                         target_rows, weights, classes=classes)
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
            target_rows, weights = _native_row_weights(dataset)
            totals += _accumulate_sums(dataset, _native_col_targets(dataset),
                                       target_rows, weights, weight_by_area=False)
    area = np.array([[cell_area_m2(row) for _ in range(GRID_WIDTH)]
                     for row in range(GRID_HEIGHT)])
    fraction = np.divide(totals[:, :, 0], area, out=np.zeros_like(area), where=area > 0)
    return {"built_m2": totals[:, :, 0], "cell_area_m2": area, "fraction": fraction}


def aggregate_building_height(paths) -> dict:
    """Area-weighted mean of ANBH over native cells carrying a value."""
    totals = np.zeros((GRID_HEIGHT, GRID_WIDTH, 2), dtype=np.float64)
    for path in paths:
        with _open(path) as dataset:
            target_rows, weights = _native_row_weights(dataset)
            nodata = dataset.nodata
            if nodata is None:
                # The 3 arcsec product documents an empty NoData: zero means "no
                # built surface", so it carries no height and gets no weight.
                valid = lambda block: block > 0.0
            else:
                valid = lambda block: (block != nodata) & np.isfinite(block) & (block > 0.0)
            totals += _accumulate_sums(dataset, _native_col_targets(dataset),
                                       target_rows, weights, valid=valid,
                                       weight_by_area=True)
    weight = totals[:, :, 1]
    mean = np.divide(totals[:, :, 0], weight, out=np.zeros_like(weight), where=weight > 0)
    return {"weighted_height": totals[:, :, 0], "weight": weight, "mean": mean}


def aggregate_population(paths) -> dict:
    """Sum of population counts divided by the cell area in km^2."""
    totals = np.zeros((GRID_HEIGHT, GRID_WIDTH, 2), dtype=np.float64)
    for path in paths:
        with _open(path) as dataset:
            target_rows, weights = _native_row_weights(dataset)
            nodata = dataset.nodata
            valid = None if nodata is None else (
                lambda block: (block != nodata) & np.isfinite(block) & (block >= 0.0))
            totals += _accumulate_sums(dataset, _native_col_targets(dataset),
                                       target_rows, weights, valid=valid,
                                       weight_by_area=False)
    area_km2 = np.array([[cell_area_m2(row) / 1e6 for _ in range(GRID_WIDTH)]
                         for row in range(GRID_HEIGHT)])
    density = np.divide(totals[:, :, 0], area_km2, out=np.zeros_like(area_km2),
                        where=area_km2 > 0)
    return {"counts": totals[:, :, 0], "cell_area_km2": area_km2, "density": density}
