"""Tests for the third-implementation re-derivation (decision 0071) and the
fail-closed guards added by decision 0072.

The defect these tests exist for: ``urban_static_recheck.py`` accumulated a raw
*pixel share* while dividing by ``sin_band * native_cols`` — neither an area nor a
count — which inflated every land-cover fraction by ~9e5 and made the
`building_height` mean un-area-weighted. Nothing covered the module, so the bug
survived until a real run reported ``passed: false``.

Every assertion below is written so that the *old* formulation fails it. The
counter-proof tests state the old formula explicitly and assert it *disagrees*, so
a future "simplification" back to the wrong denominator cannot pass silently.

These tests never touch the network and never read ``outputs/``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.download import urban_static_sources as sources              # noqa: E402
from data.preprocess import urban_static_aggregate as aggregate         # noqa: E402
from data.preprocess import urban_static_recheck as recheck             # noqa: E402


def load_driver():
    spec = importlib.util.spec_from_file_location(
        "r7_urban_static_round_recheck_test",
        ROOT / "scripts" / "r7_urban_static_round.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_tile(attempt: Path, layer: str, name: str, data, *,
                lat_south, lat_north, lon_west, lon_east) -> Path:
    """Write one native raster into ``<attempt>/native/<layer>/``."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds

    directory = Path(attempt) / "native" / layer
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    height, width = data.shape
    transform = from_bounds(lon_west, lat_south, lon_east, lat_north, width, height)
    dtype = "uint8" if data.dtype == np.uint8 else "float32"
    with rasterio.open(path, "w", driver="GTiff", height=height, width=width, count=1,
                       dtype=dtype, crs="EPSG:4326", transform=transform) as dataset:
        dataset.write(data.astype(dtype), 1)
    return path


# --- the decisive assertion: a one-class cell is exactly 1.0 ------------------

def test_a_cell_covered_by_one_class_has_fraction_exactly_one(tmp_path):
    """A cell entirely of code 50 must be 1.0, not ~9e5.

    This is the assertion that catches the sin-band defect. The old code summed a
    per-pixel dimensionless share (``sum(share) == n_pixels == 9e6`` for a real
    3000x3000 WorldCover tile) and divided by ``sin_band * native_cols`` (~9.84), so
    it returned ~9.1e5 for a cell that is 100% one class.
    """
    row, col = 20, 40
    lat_south, lat_north = aggregate.target_lat_bounds(row)
    lon_west, lon_east = aggregate.target_lon_bounds(col)
    attempt = tmp_path / "attempt"
    # 300 native rows/cols is enough to exercise the geometry; the defect is in the
    # denominator, so the pixel count only has to be >> 1.
    data = np.full((300, 300), 50, dtype=np.uint8)
    _write_tile(attempt, "worldcover_x", "t.tif", data, lat_south=lat_south,
                lat_north=lat_north, lon_west=lon_west, lon_east=lon_east)

    result = recheck.independent_cell(attempt, "worldcover_x", row, col, role="land_cover")
    assert result["fraction"][50] == pytest.approx(1.0, abs=1e-9)
    # The denominator is a cell *area* in m^2, the same quantity every other role uses.
    assert result["total"] == pytest.approx(aggregate.cell_area_m2(row), rel=1e-9)

    # Counter-proof: the superseded denominator gives a wildly different answer.
    # It divided the pixel count (300*300) by `sin_band * native_cols`, so the same
    # fully-covered cell came out at n_pixels / (sin_band * 3000) ~ 8.7e3, not 1.0.
    sin_band = (np.sin(np.radians(lat_north)) - np.sin(np.radians(lat_south)))
    old_total = sin_band * 3000
    old_fraction = (300 * 300) / old_total
    assert old_fraction > 1e3
    assert abs(old_fraction - result["fraction"][50]) > 1e2


def test_land_cover_fractions_are_dimensionless_shares_that_sum_to_at_most_one(tmp_path):
    """Two classes must split the cell, summing to 1.0 — never to a pixel count."""
    row, col = 12, 33
    lat_south, lat_north = aggregate.target_lat_bounds(row)
    lon_west, lon_east = aggregate.target_lon_bounds(col)
    attempt = tmp_path / "attempt"
    data = np.full((300, 300), 10, dtype=np.uint8)
    data[:, :150] = 50                     # western half built-up
    _write_tile(attempt, "worldcover_x", "t.tif", data, lat_south=lat_south,
                lat_north=lat_north, lon_west=lon_west, lon_east=lon_east)

    result = recheck.independent_cell(attempt, "worldcover_x", row, col, role="land_cover")
    built, tree = result["fraction"][50], result["fraction"][10]
    assert built == pytest.approx(0.5, rel=1e-6)      # longitude split is area-linear
    assert built + tree == pytest.approx(1.0, abs=1e-9)
    assert 0.0 <= built <= 1.0


# --- building_height must be area-weighted ------------------------------------

def test_building_height_is_area_weighted_not_a_plain_mean(tmp_path):
    """North half 100 m, south half 10 m: the mean must fall *below* 55.

    Native rows nearer the pole cover less area, so the high northern half carries
    less weight than a plain pixel mean gives it. Both halves must be non-zero: the
    product documents an empty NoData where zero means "no built surface", so a zero
    row carries no weight at all (``mask = block > 0``) and is not an observation.
    """
    row, col = 18, 22
    lat_south, lat_north = aggregate.target_lat_bounds(row)
    lon_west, lon_east = aggregate.target_lon_bounds(col)
    attempt = tmp_path / "attempt"
    height = 3000                            # one real 3 arcsec cell vertically
    data = np.full((height, 40), 10.0, dtype=np.float32)
    data[: height // 2, :] = 100.0           # northern half is tall, southern low
    _write_tile(attempt, "ghsl_built_h_x", "t.tif", data, lat_south=lat_south,
                lat_north=lat_north, lon_west=lon_west, lon_east=lon_east)

    result = recheck.independent_cell(attempt, "ghsl_built_h_x", row, col,
                                      role="building_height")
    mean = result["mean"]
    # Analytic expectation: the mean is weighted by each half's *spherical* area, so
    # the sin bands of the two halves (not their pixel counts) set the weights.
    mid = lat_north - sources.GRID_SPACING_DEG / 2.0
    north_band = np.sin(np.radians(lat_north)) - np.sin(np.radians(mid))
    south_band = np.sin(np.radians(mid)) - np.sin(np.radians(lat_south))
    expected = (100.0 * north_band + 10.0 * south_band) / (north_band + south_band)
    assert mean == pytest.approx(expected, rel=1e-9)
    # The weight is an absolute area in m^2 (a fraction of the cell at most).
    assert 0.0 < result["weight"] <= aggregate.cell_area_m2(row) * (1 + 1e-9)
    # Counter-proof: the old code used the dimensionless share for numerator and
    # weight, which makes every interior pixel weigh the same and returns exactly the
    # unweighted 55.0. The two are distinguishable at ~7e-4 relative.
    assert abs(mean - 55.0) > 1e-3
    assert abs(mean - 55.0) / 55.0 > 1e-4


def test_building_height_of_a_flat_cell_is_that_height(tmp_path):
    """A uniform ANBH field must come back unchanged, whatever the latitude."""
    row, col = 5, 60
    lat_south, lat_north = aggregate.target_lat_bounds(row)
    lon_west, lon_east = aggregate.target_lon_bounds(col)
    attempt = tmp_path / "attempt"
    data = np.full((300, 300), 12.5, dtype=np.float32)
    _write_tile(attempt, "ghsl_built_h_x", "t.tif", data, lat_south=lat_south,
                lat_north=lat_north, lon_west=lon_west, lon_east=lon_east)
    result = recheck.independent_cell(attempt, "ghsl_built_h_x", row, col,
                                      role="building_height")
    assert result["mean"] == pytest.approx(12.5, rel=1e-9)


# --- extensive roles must NOT be weighted by area a second time ---------------

def test_built_surface_uses_the_overlap_fraction_not_the_overlap_area(tmp_path):
    """A full cell of V m^2 per native pixel gives V * n_pixels / cell_area.

    ``built_surface`` values already carry their own area. Weighting them by an
    absolute overlap area as well would make the sum equal ``V * cell_area`` and the
    fraction collapse to ``V`` — measured at 6.5e+03-7.5e+03 relative error on the
    real tiles (decision 0072 §③).
    """
    row, col = 25, 25
    lat_south, lat_north = aggregate.target_lat_bounds(row)
    lon_west, lon_east = aggregate.target_lon_bounds(col)
    attempt = tmp_path / "attempt"
    value, n_rows, n_cols = 2.0, 300, 300
    data = np.full((n_rows, n_cols), value, dtype=np.float32)
    _write_tile(attempt, "ghsl_built_s_x", "t.tif", data, lat_south=lat_south,
                lat_north=lat_north, lon_west=lon_west, lon_east=lon_east)

    result = recheck.independent_cell(attempt, "ghsl_built_s_x", row, col,
                                      role="built_surface")
    expected = value * n_rows * n_cols / aggregate.cell_area_m2(row)
    assert result["fraction"] == pytest.approx(expected, rel=1e-9)
    # Counter-proof: double-weighting by area would return the value itself.
    assert abs(result["fraction"] - value) > 1e-3


def test_population_density_is_a_plain_sum_over_cell_area(tmp_path):
    """Population is extensive too: doubling every pixel doubles the density."""
    row, col = 25, 25
    lat_south, lat_north = aggregate.target_lat_bounds(row)
    lon_west, lon_east = aggregate.target_lon_bounds(col)
    densities = []
    for factor in (1.0, 2.0):
        attempt = tmp_path / f"attempt_{factor}"
        data = np.full((300, 300), 4.0 * factor, dtype=np.float32)
        _write_tile(attempt, "ghsl_pop_x", "t.tif", data, lat_south=lat_south,
                    lat_north=lat_north, lon_west=lon_west, lon_east=lon_east)
        densities.append(recheck.independent_cell(
            attempt, "ghsl_pop_x", row, col, role="population")["density"])
    assert densities[1] == pytest.approx(2.0 * densities[0], rel=1e-9)
    expected = 4.0 * 300 * 300 / (aggregate.cell_area_m2(row) / 1e6)
    assert densities[0] == pytest.approx(expected, rel=1e-9)


# --- the recheck must disagree with a *wrong* aggregator ----------------------

def test_recheck_uses_exact_boundary_overlap_not_the_native_centre(tmp_path):
    """A native column straddling the cell edge must count *by how much it overlaps*.

    This is the discriminator decision 0071 exists for. Place a 3-column tile whose
    middle column is centred exactly on the target cell's western edge and is the only
    built column:

    * exact-overlap attribution → that column contributes the half of itself that is
      inside the cell, i.e. a built share of ``0.5 / 3``;
    * a cell-centre rule → the column's centre sits on the boundary, so it is either
      fully in (``1 / 3``) or fully out (``0``), never ``0.5 / 3``.

    A recheck that shared the aggregator's centre rule would return one of the two
    all-or-nothing answers and could not detect the wrong rule.
    """
    row, col = 30, 30
    lat_south, lat_north = aggregate.target_lat_bounds(row)
    lon_west, lon_east = aggregate.target_lon_bounds(col)
    step = (lon_east - lon_west) / 3.0
    attempt = tmp_path / "attempt"
    # Columns 0,1,2 span [lon_west - 1.5*step, lon_west + 1.5*step]; column 1's centre
    # lands exactly on lon_west and its body straddles the edge.
    data = np.full((300, 3), 10, dtype=np.uint8)
    data[:, 1] = 50                            # only the straddling column is built
    _write_tile(attempt, "straddle", "t.tif", data, lat_south=lat_south,
                lat_north=lat_north, lon_west=lon_west - 1.5 * step,
                lon_east=lon_west + 1.5 * step)

    result = recheck.independent_cell(attempt, "straddle", row, col, role="land_cover")
    built = result["fraction"][50]
    assert built == pytest.approx(0.5 / 3.0, rel=1e-6)
    # The two centre-rule answers are both far away and must not coincide with ours.
    assert abs(built - 1.0 / 3.0) > 0.1
    assert built > 0.01            # definitively not "fully out"


# --- publish must fail closed on a failed recheck ----------------------------

def test_publish_refuses_when_the_recheck_did_not_pass(tmp_path, monkeypatch):
    """A recorded ``passed: false`` must stop publication, not annotate it."""
    import json

    driver = load_driver()
    attempt = tmp_path / "attempt99"
    attempt.mkdir()
    (attempt / "protocol.json").write_text(json.dumps({"dropped_layers": []}), "utf-8")
    (attempt / "aggregate_receipt.json").write_text(json.dumps({"fields_sha256": "x"}), "utf-8")
    (attempt / "recheck_receipt.json").write_text(json.dumps({
        "passed": False, "max_relative_delta": 913872.595, "tolerance": 1e-5,
        "comparisons": [{"row": 7, "col": 64, "field": "lc_built_frac",
                         "relative_delta": 9.1e5}],
    }), "utf-8")
    monkeypatch.setattr(driver, "latest_attempt", lambda *a, **k: attempt)
    published = []
    monkeypatch.setattr(driver, "_publish", lambda **kw: published.append(kw) or {})

    with pytest.raises(SystemExit) as error:
        driver.publish()
    assert "did not pass" in str(error.value)
    assert "lc_built_frac" in str(error.value)
    assert published == [], "publish must not reach the writer"


def test_publish_proceeds_when_the_recheck_passed(tmp_path, monkeypatch):
    """The mirror case: ``passed: true`` must not be blocked by the new guard."""
    import json

    driver = load_driver()
    attempt = tmp_path / "attempt98"
    attempt.mkdir()
    for name, body in (("protocol.json", {"dropped_layers": []}),
                       ("aggregate_receipt.json", {"fields_sha256": "x"}),
                       ("recheck_receipt.json",
                        {"passed": True, "max_relative_delta": 1e-9, "tolerance": 1e-5,
                         "comparisons": []})):
        (attempt / name).write_text(json.dumps(body), "utf-8")
    monkeypatch.setattr(driver, "latest_attempt", lambda *a, **k: attempt)
    monkeypatch.setattr(driver, "_publish",
                        lambda **kw: {"static_identity": "abc"})
    assert driver.publish()["static_identity"] == "abc"


# --- a dropped named layer must say why (R-008) ------------------------------

def test_a_dropped_layer_without_a_reason_is_refused():
    """``reason: None`` is what attempt03 recorded for three real layers."""
    driver = load_driver()
    with pytest.raises(SystemExit) as error:
        driver.require_dropped_reasons([{"name": "ghsl_built_s_2000", "reason": None}])
    assert "ghsl_built_s_2000" in str(error.value)


def test_a_dropped_layer_with_a_blank_reason_is_refused():
    driver = load_driver()
    with pytest.raises(SystemExit):
        driver.require_dropped_reasons([{"name": "x", "reason": "   "}])


def test_a_missing_reason_key_is_refused():
    driver = load_driver()
    with pytest.raises(SystemExit):
        driver.require_dropped_reasons([{"name": "x"}])


def test_every_shipped_dropped_layer_has_a_reason():
    """The two hand-written drops must keep their evidence-backed reasons."""
    driver = load_driver()
    driver.require_dropped_reasons(driver.DROPPED_LAYERS)
    for entry in driver.DROPPED_LAYERS:
        assert isinstance(entry["reason"], str) and entry["reason"].strip()
        assert entry.get("evidence")


def test_an_unreachable_layer_is_dropped_with_a_nonempty_reason(tmp_path, monkeypatch):
    """A layer whose every probe fails must carry a reason naming what was observed.

    ``OUTPUTS`` is redirected into ``tmp_path`` so this never writes an attempt
    directory into the real publication tree, and ``probe`` is restored by monkeypatch
    rather than left mutated on the shared sources module.
    """
    import json

    driver = load_driver()
    monkeypatch.setattr(driver, "OUTPUTS", tmp_path, raising=False)
    monkeypatch.setattr(driver.sources, "probe", lambda url, **kw: {
        "url": url, "method": "HEAD", "http_status": 404, "content_length": None,
        "content_type": None, "content_range": None, "reachable": False,
        "error": "HTTP 404"})

    report = driver.verify_endpoints()
    body = json.loads((Path(report["attempt"]) / "endpoint_report.json").read_text("utf-8"))
    assert body["dropped"], "an all-unreachable catalog must produce drops"
    driver.require_dropped_reasons(body["dropped"])       # must not raise
    # Only the probe-driven drops carry the observed status; the two hand-written
    # entries in DROPPED_LAYERS state their own reason and are asserted separately.
    probed = [entry for entry in body["dropped"] if "404" in (entry.get("reason") or "")]
    assert probed, "no probe-driven drop carries its observed HTTP status"
    for entry in probed:
        assert entry.get("failed_no_fallback") is True
        assert entry["reason"].strip()
    # Nothing may have been written outside the redirected tree.
    assert Path(report["attempt"]).parent == tmp_path
