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


def _column_attribution(dataset) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Split every native column between the target columns it overlaps.

    Returns ``(target_a, fraction_a, target_b, fraction_b)`` where the ``b`` arrays are
    all zero when a native column lies wholly inside one target column. Decision 0071:
    a native column that straddles a target edge contributes to **both** neighbours in
    proportion to the width it puts in each. Assigning it to whichever target column
    contains its centre is exactly the rule that made the published values wrong.
    """
    transform = dataset.transform
    lon_left = transform.c + transform.a * np.arange(dataset.width)
    lon_right = lon_left + transform.a
    width = lon_right - lon_left
    first = np.floor((lon_left - EDGE_WEST) / GRID_SPACING_DEG).astype(np.int64)
    last = np.floor((lon_right - EDGE_WEST) / GRID_SPACING_DEG - 1e-12).astype(np.int64)
    # fraction of this native column inside target column `first` / `last`
    f_first = np.clip((EDGE_WEST + (first + 1) * GRID_SPACING_DEG - lon_left) / width, 0.0, 1.0)
    f_last = np.clip((lon_right - (EDGE_WEST + last * GRID_SPACING_DEG)) / width, 0.0, 1.0)
    single = first == last
    fraction_b = np.where(single, 0.0, f_last)
    fraction_a = np.where(single, f_first, f_first)
    ok_a = (first >= 0) & (first < GRID_WIDTH)
    ok_b = (~single) & (last >= 0) & (last < GRID_WIDTH)
    return [(first, fraction_a, ok_a), (last, fraction_b, ok_b)]


def _native_rows_for_target(dataset, lat_south: float, lat_north: float):
    """(start, stop, per-native-row overlap fraction) for the rows overlapping a target row."""
    transform = dataset.transform
    lat_top = transform.f + transform.e * np.arange(dataset.height)
    lat_bottom = lat_top + transform.e
    over_bottom = np.maximum(lat_bottom, lat_south)
    over_top = np.minimum(lat_top, lat_north)
    index = np.flatnonzero(over_top > over_bottom)
    if index.size == 0:
        return []
    band = np.sin(np.radians(lat_top[index])) - np.sin(np.radians(lat_bottom[index]))
    overlap = np.sin(np.radians(over_top[index])) - np.sin(np.radians(over_bottom[index]))
    start, stop = int(index[0]), int(index[-1]) + 1
    fractions = np.zeros(stop - start, dtype=np.float64)
    fractions[index - start] = overlap / band
    return [(start, stop, fractions)]


def _binned(values: np.ndarray, targets: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Sum ``values`` (per native column, already weighted) into target column bins."""
    return np.bincount(targets[ok], weights=values[ok], minlength=GRID_WIDTH)[:GRID_WIDTH]


def _accumulate_counts(dataset, *, classes: tuple[int, ...]) -> np.ndarray:
    """Area-weighted per-class counts, shape (65, 65, len(classes)).

    Attribution is exact spherical area overlap (decision 0071). The reduction is
    vectorised per target row and per native-row block: a WorldCover tile is
    36,000 x 36,000, so a per-native-cell Python loop costs about an hour per tile and
    is what made the first edge-binned run exceed its six-hour stage timeout.
    """
    counts = np.zeros((GRID_HEIGHT, GRID_WIDTH, len(classes)), dtype=np.float64)
    attribution = _column_attribution(dataset)
    for row in range(GRID_HEIGHT):
        lat_south, lat_north = target_lat_bounds(row)
        for start, stop, row_fraction in _native_rows_for_target(dataset, lat_south, lat_north):
            block = _read(dataset, slice(start, stop), slice(0, dataset.width))
            if block.size == 0:
                continue
            # Scale each native row by its own sin band: row_fraction is a proportion
            # of that band, so band * fraction is the exact spherical area the row puts
            # inside this target row. Without the band the numerator would be in pixel
            # units while the denominator is dimensionless, inflating shares by the cell
            # area (about 8.7e5 pixels for a 0.25 deg cell).
            lat_top = dataset.transform.f + dataset.transform.e * np.arange(start, stop)
            band = np.abs(np.sin(np.radians(lat_top))
                          - np.sin(np.radians(lat_top + dataset.transform.e)))
            weight = (row_fraction * band)[:, None]
            for position, code in enumerate(classes):
                hit = (block == code).astype(np.float64) * weight
                column_totals = hit.sum(axis=0)
                for targets, fraction, ok in attribution:
                    counts[row, :, position] += _binned(column_totals * fraction, targets, ok)
    return counts


def _accumulate_sums(dataset, *, valid=None, weight_by_area: bool) -> np.ndarray:
    """Value sums and weight totals, shape (65, 65, 2), split by exact edge overlap.

    ``weight_by_area`` distinguishes an **intensive** quantity (a mean, which must be
    area-weighted because a native cell's area depends on its latitude) from an
    **extensive** one (an absolute area or count, which is simply summed). Applying the
    area weight to an extensive quantity would scale it by the sin band and is exactly
    the bug this flag exists to prevent.
    """
    totals = np.zeros((GRID_HEIGHT, GRID_WIDTH, 2), dtype=np.float64)
    attribution = _column_attribution(dataset)
    for row in range(GRID_HEIGHT):
        lat_south, lat_north = target_lat_bounds(row)
        for start, stop, row_fraction in _native_rows_for_target(dataset, lat_south, lat_north):
            block = _read(dataset, slice(start, stop), slice(0, dataset.width)).astype(np.float64)
            if block.size == 0:
                continue
            if valid is not None:
                mask = valid(block)
                values = np.where(mask, block, 0.0)
                carried = mask.astype(np.float64)
            else:
                values = block
                carried = np.ones(block.shape, dtype=np.float64)
            if weight_by_area:
                # row_fraction is a proportion of the native row's sin band, so scaling
                # by the band turns a mean into an area-weighted mean.
                lat_top = dataset.transform.f + dataset.transform.e * np.arange(start, stop)
                band = np.abs(np.sin(np.radians(lat_top))
                              - np.sin(np.radians(lat_top + dataset.transform.e)))
                row_weight = row_fraction * band
            else:
                row_weight = row_fraction
            value_totals = (values * row_weight[:, None]).sum(axis=0)
            carried_totals = (carried * row_weight[:, None]).sum(axis=0)
            for targets, fraction, ok in attribution:
                totals[row, :, 0] += _binned(value_totals * fraction, targets, ok)
                totals[row, :, 1] += _binned(carried_totals * fraction, targets, ok)
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
            counts += _accumulate_counts(dataset, classes=classes)
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


def _tile_bounds(paths) -> list[tuple[float, float, float, float]]:
    """``(west, south, east, north)`` for every raster, for the coverage mask."""
    bounds = []
    for path in paths:
        with _open(path) as dataset:
            box = dataset.bounds
            bounds.append((box.left, box.bottom, box.right, box.top))
    return bounds


def aggregate_built_surface(paths) -> dict:
    """Sum of built m^2 divided by the target cell area in m^2.

    ``coverage`` reports how much of each cell the downloaded tiles actually span. A
    missing tile contributes nothing, so without this mask a hole would be published
    as ``0`` built-up area — indistinguishable from a cell that genuinely has none
    (decision 0073 §②). The caller must refuse to publish when coverage < 1.
    """
    totals = np.zeros((GRID_HEIGHT, GRID_WIDTH, 2), dtype=np.float64)
    for path in paths:
        with _open(path) as dataset:
            totals += _accumulate_sums(dataset, weight_by_area=False)
    area = np.array([[cell_area_m2(row) for _ in range(GRID_WIDTH)]
                     for row in range(GRID_HEIGHT)])
    fraction = np.divide(totals[:, :, 0], area, out=np.zeros_like(area), where=area > 0)
    return {"built_m2": totals[:, :, 0], "cell_area_m2": area, "fraction": fraction,
            "coverage": coverage_fraction(_tile_bounds(paths))}


def aggregate_building_height(paths) -> dict:
    """Area-weighted mean of ANBH over native cells carrying a value.

    ``coverage`` carries the same meaning as in ``aggregate_built_surface``: a tile
    that was never downloaded leaves zero weight, which would otherwise read as
    "no buildings here" rather than "no data here".
    """
    totals = np.zeros((GRID_HEIGHT, GRID_WIDTH, 2), dtype=np.float64)
    for path in paths:
        with _open(path) as dataset:
            nodata = dataset.nodata
            if nodata is None:
                # The 3 arcsec product documents an empty NoData: zero means "no
                # built surface", so it carries no height and gets no weight.
                valid = lambda block: block > 0.0
            else:
                valid = lambda block: (block != nodata) & np.isfinite(block) & (block > 0.0)
            totals += _accumulate_sums(dataset, valid=valid, weight_by_area=True)
    weight = totals[:, :, 1]
    mean = np.divide(totals[:, :, 0], weight, out=np.zeros_like(weight), where=weight > 0)
    return {"weighted_height": totals[:, :, 0], "weight": weight, "mean": mean,
            "coverage": coverage_fraction(_tile_bounds(paths))}


def aggregate_population(paths) -> dict:
    """Sum of population counts divided by the cell area in km^2.

    ``coverage`` has the same meaning as the other roles (decision 0073 §②): a
    missing tile would otherwise publish a population of zero for a cell that was
    simply never measured.
    """
    totals = np.zeros((GRID_HEIGHT, GRID_WIDTH, 2), dtype=np.float64)
    for path in paths:
        with _open(path) as dataset:
            nodata = dataset.nodata
            valid = None if nodata is None else (
                lambda block: (block != nodata) & np.isfinite(block) & (block >= 0.0))
            totals += _accumulate_sums(dataset, valid=valid, weight_by_area=False)
    area_km2 = np.array([[cell_area_m2(row) / 1e6 for _ in range(GRID_WIDTH)]
                         for row in range(GRID_HEIGHT)])
    density = np.divide(totals[:, :, 0], area_km2, out=np.zeros_like(area_km2),
                        where=area_km2 > 0)
    return {"counts": totals[:, :, 0], "cell_area_km2": area_km2, "density": density,
            "coverage": coverage_fraction(_tile_bounds(paths))}
