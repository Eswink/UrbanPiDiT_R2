"""CI-safe tests for the urban static-covariate round (decision 0063 / plan 0027).

These tests never touch the network and never depend on ``outputs/``: the tile
grids are recomputed from the frozen ROI, the aggregation is exercised on a
synthetic raster written into ``tmp_path``, and the failure semantics (an
unreachable layer must be recorded as ``failed-no-fallback`` and must not be
substituted) are asserted by counter-example.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.download import urban_static_sources as sources          # noqa: E402
from data.preprocess import urban_static_aggregate as aggregate    # noqa: E402


def load_driver():
    spec = importlib.util.spec_from_file_location(
        "r7_urban_static_round", ROOT / "scripts" / "r7_urban_static_round.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- ROI and tile grids -------------------------------------------------------

def test_roi_edges_match_the_frozen_read_plan():
    assert (sources.ROI_SOUTH, sources.ROI_NORTH) == (27.0, 43.0)
    assert (sources.ROI_WEST, sources.ROI_EAST) == (107.0, 123.0)
    assert (sources.EDGE_SOUTH, sources.EDGE_NORTH) == (26.875, 43.125)
    assert (sources.EDGE_WEST, sources.EDGE_EAST) == (106.875, 123.125)
    assert (sources.GRID_HEIGHT, sources.GRID_WIDTH) == (65, 65)
    assert sources.GRID_SPACING_DEG == 0.25


def test_worldcover_tiles_cover_the_roi_and_are_3_degree_tiles():
    tiles = sources.worldcover_tiles()
    assert len(tiles) == 49 and len(set(tiles)) == 49
    assert "N24E105" in tiles and "N42E123" in tiles
    corners = [(int(t[1:3]) * (1 if t[0] == "N" else -1),
                int(t[4:7]) * (1 if t[3] == "E" else -1)) for t in tiles]
    for lat, lon in ((sources.EDGE_SOUTH, sources.EDGE_WEST),
                     (sources.EDGE_SOUTH, sources.EDGE_EAST),
                     (sources.EDGE_NORTH, sources.EDGE_WEST),
                     (sources.EDGE_NORTH, sources.EDGE_EAST)):
        assert any(tile_lat <= lat <= tile_lat + 3 and tile_lon <= lon <= tile_lon + 3
                   for tile_lat, tile_lon in corners), (lat, lon)


def test_ghsl_tiles_are_the_nine_covering_the_roi():
    assert sources.ghsl_tiles() == [(r, c) for r in (5, 6, 7) for c in (29, 30, 31)]


def test_ghsl_tile_origin_reproduces_the_measured_bounds():
    # The measured tile R6_C28 spans lon 89.992083..99.992083 and lat
    # 29.099583..39.099583; the grid must reproduce that exactly.
    row, col = 6, 28
    left = sources.GHSL_TILE_ORIGIN_LON + (col - sources.GHSL_TILE_ORIGIN_COL) * 10.0
    bottom = sources.GHSL_TILE_ORIGIN_LAT - (row - sources.GHSL_TILE_ORIGIN_ROW) * 10.0
    assert left == pytest.approx(89.99208295968086)
    assert bottom == pytest.approx(29.099583419121082)


def test_every_layer_url_has_the_expected_extension():
    for layer in sources.layer_catalog():
        assert layer["urls"], layer["name"]
        for url in layer["urls"]:
            assert url.endswith((".tif", ".zip")), url


# --- failure semantics (R-008 counter-example) --------------------------------

def test_an_unreachable_tile_makes_the_whole_layer_failed_no_fallback(monkeypatch):
    driver = load_driver()
    calls = {"n": 0}

    def fake_probe(url, **kwargs):
        calls["n"] += 1
        return {"url": url, "method": "HEAD", "http_status": 404, "content_length": None,
                "content_type": None, "reachable": False, "error": "HTTP 404"}

    monkeypatch.setattr(driver.sources, "probe", fake_probe)
    monkeypatch.setattr(driver, "new_attempt", lambda: Path("/tmp/r7-urban-static-test"))
    report = driver.verify_endpoints()
    assert report["layers_reachable"] == []
    assert set(report["layers_dropped"]) >= {"worldcover_v200_2021", "ghsl_pop_2020"}
    assert calls["n"] > 0


def test_landscan_is_recorded_as_dropped_without_a_substitute():
    names = {row["name"] for row in json.loads(json.dumps(
        load_driver().DROPPED_LAYERS))}
    assert "landscan_global" in names
    entry = next(row for row in load_driver().DROPPED_LAYERS
                 if row["name"] == "landscan_global")
    assert entry["failed_no_fallback"] is True
    # No account-gated layer may appear in the fetchable catalog.
    catalog = {layer["name"] for layer in sources.layer_catalog()}
    assert not any("landscan" in name for name in catalog)


# --- aggregation is area-weighted, not interpolation --------------------------

def _synthetic_raster(path: Path, *, value, lat_south, lat_north, lon_west, lon_east):
    import rasterio
    from rasterio.transform import from_bounds

    height, width = 120, 120
    transform = from_bounds(lon_west, lat_south, lon_east, lat_north, width, height)
    with rasterio.open(path, "w", driver="GTiff", height=height, width=width, count=1,
                       dtype="float32", crs="EPSG:4326", transform=transform) as dataset:
        dataset.write(np.full((height, width), value, dtype="float32"), 1)
    return path


def test_built_surface_fraction_is_a_plain_sum_over_cell_area(tmp_path):
    # A constant built-surface raster of 1 m^2 per native cell over one whole
    # target cell must give fraction = (native cells) / (cell area in m^2).
    row, col = 30, 30
    lat_south, lat_north = aggregate.target_lat_bounds(row)
    lon_west, lon_east = aggregate.target_lon_bounds(col)
    path = _synthetic_raster(tmp_path / "flat.tif", value=1.0, lat_south=lat_south,
                             lat_north=lat_north, lon_west=lon_west, lon_east=lon_east)
    result = aggregate.aggregate_built_surface([path])
    # 120x120 native cells each 1 m^2 over a cell of ~5.7e8 m^2.
    assert result["fraction"][row, col] == pytest.approx(120 * 120 / aggregate.cell_area_m2(row), rel=1e-6)
    # And it is a *sum*, so the area weight must not have been applied twice.
    assert result["built_m2"][row, col] == pytest.approx(120 * 120, rel=1e-6)


def test_land_cover_fraction_is_area_weighted_within_a_cell(tmp_path):
    import rasterio
    from rasterio.transform import from_bounds

    row, col = 20, 40
    lat_south, lat_north = aggregate.target_lat_bounds(row)
    lon_west, lon_east = aggregate.target_lon_bounds(col)
    # 3000 native rows is one real WorldCover cell, so the sub-band granularity
    # actually exercises the intra-cell equal-area weighting.
    height, width = 3000, 40
    transform = from_bounds(lon_west, lat_south, lon_east, lat_north, width, height)
    data = np.full((height, width), 10, dtype="uint8")
    data[:height // 2, :] = 50     # the northern half is built-up
    path = tmp_path / "classes.tif"
    with rasterio.open(path, "w", driver="GTiff", height=height, width=width, count=1,
                       dtype="uint8", crs="EPSG:4326", transform=transform) as dataset:
        dataset.write(data, 1)
    result = aggregate.aggregate_land_cover([path])
    built = result["fractions"][50][row, col]
    tree = result["fractions"][10][row, col]
    assert built + tree == pytest.approx(1.0, abs=1e-9)
    # Northern native rows cover less area at the same angular span, so a raw
    # count share of 0.5 must come out strictly below 0.5 once weighted.
    assert 0.49 < built < 0.5
    assert built + tree == pytest.approx(1.0, abs=1e-9)


def test_building_height_is_an_area_weighted_mean(tmp_path):
    import rasterio
    from rasterio.transform import from_bounds

    row, col = 10, 10
    lat_south, lat_north = aggregate.target_lat_bounds(row)
    lon_west, lon_east = aggregate.target_lon_bounds(col)
    height, width = 60, 60
    transform = from_bounds(lon_west, lat_south, lon_east, lat_north, width, height)
    data = np.zeros((height, width), dtype="float32")
    data[:30, :] = 10.0        # northern half built at 10 m
    path = tmp_path / "height.tif"
    with rasterio.open(path, "w", driver="GTiff", height=height, width=width, count=1,
                       dtype="float32", crs="EPSG:4326", transform=transform) as dataset:
        dataset.write(data, 1)
    result = aggregate.aggregate_building_height([path])
    # Only the northern half carries a value; the mean over those cells is 10 m,
    # and the zero rows contribute no weight.
    assert result["mean"][row, col] == pytest.approx(10.0, rel=1e-6)
    assert result["weight"][row, col] > 0


def test_area_basis_is_the_authalic_sphere_not_a_constant():
    south, north = 27.0, 27.25
    assert aggregate.sin_band(south, north) == pytest.approx(
        np.sin(np.radians(north)) - np.sin(np.radians(south)), rel=1e-12)
    # A northern cell is smaller than a southern one at the same 0.25 deg span.
    assert aggregate.cell_area_m2(0) < aggregate.cell_area_m2(64)
