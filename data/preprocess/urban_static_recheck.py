"""Independent re-derivation of one aggregated cell (decision 0071).

This module exists to be the **third implementation**. It deliberately imports nothing
from ``urban_static_aggregate``: the recheck it serves must not share an attribution
rule with the code it checks. The previous arrangement reimplemented the aggregator's
cell-centre rule inside the driver, so its 5.15e-08 agreement proved only that two
copies of one rule agreed.

Attribution here is by **exact spherical area overlap**: a native cell contributes the
fraction of its own area that falls inside the target box, with latitude as a ratio of
sin bands and longitude as a linear ratio. The output is a derived statistic of real
observations on the frozen 0.25 deg cell edges, not an interpolation and not a truth
field.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data.download import urban_static_sources as sources

AUTHALIC_RADIUS_M = 6371007.181          # WGS84 authalic sphere radius


def independent_cell(attempt: Path, layer_name: str, row: int, col: int, *,
                      role: str, classes=(10, 40, 50, 60, 80)) -> dict:
    """Recompute one cell from the raw tiles, sharing no attribution rule with the aggregator.

    This is the third implementation decision 0071 requires. It reads each native cell
    that touches the target box and scores it by the **exact fraction of its own
    spherical area** that falls inside that box (latitude as a ratio of sin bands,
    longitude as a linear ratio), then reduces. It deliberately does not call
    ``aggregate._row_spans`` / ``aggregate._col_spans`` / ``aggregate._native_*``: the
    previous version of this function reimplemented the *same* cell-centre rule the
    aggregator used, so a 5.15e-08 agreement proved only that two copies of one rule
    agreed, not that the rule was right.
    """
    import rasterio
    from rasterio.windows import Window

    # Cell geometry from the frozen constants, not from an aggregate helper, so a bug
    # in the aggregator's bounds cannot hide here too.
    lat_north = sources.EDGE_NORTH - row * sources.GRID_SPACING_DEG
    lat_south = lat_north - sources.GRID_SPACING_DEG
    lon_west = sources.EDGE_WEST + col * sources.GRID_SPACING_DEG
    lon_east = lon_west + sources.GRID_SPACING_DEG
    # The WGS84 authalic radius is a physical constant, not an attribution rule;
    # restating it here keeps this function free of the aggregator's module surface.
    authalic_radius_m = 6371007.181
    cell_area_m2 = (authalic_radius_m ** 2
                    * np.radians(sources.GRID_SPACING_DEG)
                    * (np.sin(np.radians(lat_north)) - np.sin(np.radians(lat_south))))

    per_class = np.zeros(len(classes))
    value_sum = 0.0
    area_weight = 0.0
    for path in tile_paths(attempt, layer_name):
        with rasterio.open(path) as dataset:
            bounds = dataset.bounds
            if not (bounds.left < lon_east and bounds.right > lon_west
                    and bounds.bottom < lat_north and bounds.top > lat_south):
                continue
            transform = dataset.transform
            lat_top = transform.f + transform.e * np.arange(dataset.height)
            lat_bottom = lat_top + transform.e
            lon_left = transform.c + transform.a * np.arange(dataset.width)
            lon_right = lon_left + transform.a
            over_bottom = np.maximum(lat_bottom, lat_south)
            over_top = np.minimum(lat_top, lat_north)
            over_left = np.maximum(lon_left, lon_west)
            over_right = np.minimum(lon_right, lon_east)
            has_row = over_top > over_bottom
            has_col = over_right > over_left
            if not has_row.any() or not has_col.any():
                continue
            rows = np.flatnonzero(has_row)
            cols = np.flatnonzero(has_col)
            window = Window(int(cols[0]), int(rows[0]),
                            int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1))
            block = dataset.read(1, window=window).astype(np.float64)
            # Fraction of each native cell inside the target box, as true spherical area.
            lat_fraction = ((np.sin(np.radians(over_top[rows]))
                             - np.sin(np.radians(over_bottom[rows])))
                            / (np.sin(np.radians(lat_top[rows]))
                               - np.sin(np.radians(lat_bottom[rows]))))
            lon_fraction = ((over_right[cols] - over_left[cols])
                            / (lon_right[cols] - lon_left[cols]))
            share = lat_fraction[:, None] * lon_fraction[None, :]
            if role == "land_cover":
                for position, code in enumerate(classes):
                    per_class[position] += float(np.nansum((block == code) * share))
            elif role == "building_height":
                mask = block > 0
                value_sum += float(np.nansum(np.where(mask, block, 0.0) * share))
                area_weight += float(np.nansum(mask * share))
            else:
                value_sum += float(np.nansum(block * share))

    if role == "land_cover":
        # The denominator is the whole cell, in the same dimensionless units the shares
        # are expressed in: the sin band of the target cell times the number of native
        # columns spanning it.
        if not tile_paths(attempt, layer_name):
            return {"fraction": {code: 0.0 for code in classes}, "total": 0.0}
        with rasterio.open(tile_paths(attempt, layer_name)[0]) as dataset:
            native_cols = sources.GRID_SPACING_DEG / dataset.transform.a
        total = (np.sin(np.radians(lat_north)) - np.sin(np.radians(lat_south))) * native_cols
        return {"fraction": {code: per_class[i] / total if total else 0.0
                             for i, code in enumerate(classes)}, "total": total}
    if role == "building_height":
        return {"mean": value_sum / area_weight if area_weight else 0.0,
                "weight": area_weight}
    if role == "built_surface":
        return {"fraction": value_sum / cell_area_m2}
    return {"density": value_sum / (cell_area_m2 / 1e6)}



def tile_paths(attempt: Path, layer_name: str) -> list[Path]:
    """Native tile paths for one layer inside an attempt directory."""
    directory = Path(attempt) / "native" / layer_name
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.suffix == ".tif")
