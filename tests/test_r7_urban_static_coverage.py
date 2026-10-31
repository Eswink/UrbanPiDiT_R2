"""Tests for decision 0073: transient-vs-absent probing and coverage masks.

The two defects these tests exist for:

1. ``attempt03`` recorded three real layers (``ghsl_built_s_2000/2005/2010``) as
   ``failed-no-fallback`` on nothing but ``read timeout``; a live re-probe found
   45/45 tiles answering HTTP 200. A single timeout must not be able to drop a
   layer, and a layer whose probes all timed out must be *unreliable*, not absent.
2. ``aggregate_built_surface`` / ``_building_height`` / ``_population`` had no
   coverage mask, so a missing tile was published as ``0`` — indistinguishable from
   "there genuinely are no buildings here". ``attempt03`` shipped 288 such cells.

Every guard is asserted by counter-example: a case that must be refused, and a case
that must be allowed. These tests never touch the network and never read ``outputs/``.
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
from data.preprocess import urban_static_aggregate as aggregate        # noqa: E402
from data.preprocess import urban_static_endpoints as endpoints        # noqa: E402


def load_driver():
    spec = importlib.util.spec_from_file_location(
        "r7_urban_static_round_cov", ROOT / "scripts" / "r7_urban_static_round.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _probe(*, status=None, reachable=False, error=None, failure_class=None):
    record = {"url": "https://example.invalid/t.tif", "method": "HEAD",
              "http_status": status, "content_length": None, "content_type": None,
              "content_range": None, "reachable": reachable, "error": error}
    if failure_class is not None:
        record["failure_class"] = failure_class
    return record


# --- a timeout is not evidence of absence (0073 §①) --------------------------

def test_only_a_4xx_counts_as_absent():
    """404/403 say "not there". A timeout and a 502 do not."""
    assert sources._classify_failure(_probe(status=404)) == "absent"
    assert sources._classify_failure(_probe(status=403)) == "absent"
    assert sources._classify_failure(_probe(status=None,
                                            error="TimeoutError: read timed out")) == "transient"
    assert sources._classify_failure(_probe(status=502)) == "transient"
    assert sources._classify_failure(_probe(status=503)) == "transient"


def test_a_4xx_is_not_retried_but_a_timeout_is(monkeypatch):
    """Retrying an answer is waste; not retrying a throttle is the attempt03 bug."""
    calls = []

    def fake_probe(url, *, range_bytes=None, timeout=60.0):
        calls.append(timeout)
        return _probe(status=404, error="HTTP 404")

    monkeypatch.setattr(sources, "probe", fake_probe)
    result = sources.probe_with_retry("https://example.invalid/t.tif", sleep=lambda s: None)
    assert len(calls) == 1, "a definitive 4xx must stop the retry loop"
    assert result["failure_class"] == "absent"

    calls.clear()

    def fake_timeout(url, *, range_bytes=None, timeout=60.0):
        calls.append(timeout)
        return _probe(status=None, error="TimeoutError: The read operation timed out")

    monkeypatch.setattr(sources, "probe", fake_timeout)
    result = sources.probe_with_retry("https://example.invalid/t.tif", sleep=lambda s: None)
    assert calls == list(sources.PROBE_TIMEOUTS), "every timeout must be tried"
    assert result["failure_class"] == "transient"
    assert result["attempts"] == len(sources.PROBE_TIMEOUTS)


def test_a_later_success_clears_a_transient_failure(monkeypatch):
    """The attempt03 scenario: first dial times out, second answers 200."""
    sequence = iter([
        _probe(status=None, error="TimeoutError: read timed out"),
        _probe(status=200, reachable=True),
    ])
    monkeypatch.setattr(sources, "probe", lambda url, **kw: next(sequence))
    result = sources.probe_with_retry("https://example.invalid/t.tif", sleep=lambda s: None)
    assert result["reachable"] is True
    assert result["failure_class"] == "none"
    assert result["attempts"] == 2


# --- verdict classification (0073 §①) ----------------------------------------

def test_all_transient_probes_make_a_layer_unreliable_not_dropped():
    probes = [_probe(status=None, error="TimeoutError", failure_class="transient")] * 9
    assert endpoints._endpoint_verdict(probes) == "unreliable"


def test_all_4xx_probes_make_a_layer_failed_no_fallback():
    probes = [_probe(status=404, failure_class="absent")] * 9
    assert endpoints._endpoint_verdict(probes) == "failed-no-fallback"


def test_one_reachable_tile_keeps_the_layer():
    probes = [_probe(status=200, reachable=True, failure_class="none"),
              _probe(status=404, failure_class="absent")]
    assert endpoints._endpoint_verdict(probes) == "reachable-with-absent-tiles"


def test_a_mixed_transient_and_absent_without_success_is_not_a_drop():
    """Only *every* tile being a 4xx is positive evidence of absence."""
    probes = [_probe(status=404, failure_class="absent"),
              _probe(status=None, error="TimeoutError", failure_class="transient")]
    assert endpoints._endpoint_verdict(probes) == "unreliable"


# --- freeze refuses an unmeasurable layer (0073 §①) ---------------------------

def test_freeze_refuses_when_a_layer_could_not_be_measured():
    with pytest.raises(SystemExit) as error:
        endpoints.require_no_unreliable_layers(
            [{"name": "ghsl_built_s_2000", "reason": "every probe was transient"}])
    assert "ghsl_built_s_2000" in str(error.value)
    assert "retry" in str(error.value).lower() or "Retry" in str(error.value)


def test_freeze_allows_a_clean_run():
    endpoints.require_no_unreliable_layers([])      # must not raise


# --- coverage masks (0073 §②) -------------------------------------------------

def _write_tile(attempt: Path, layer: str, name: str, data, *,
                lat_south, lat_north, lon_west, lon_east):
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


def test_a_complete_layer_reports_full_coverage(tmp_path):
    pytest.importorskip("rasterio")
    row, col = 25, 25
    lat_south, lat_north = aggregate.target_lat_bounds(row)
    lon_west, lon_east = aggregate.target_lon_bounds(col)
    path = _write_tile(tmp_path, "s", "full.tif", np.full((20, 20), 1.0, np.float32),
                       lat_south=lat_south, lat_north=lat_north,
                       lon_west=lon_west, lon_east=lon_east)
    result = aggregate.aggregate_built_surface([path])
    assert result["coverage"][row, col] == pytest.approx(1.0, abs=1e-6)
    assert result["fraction"][row, col] > 0


def test_a_missing_tile_leaves_a_coverage_hole(tmp_path):
    """The attempt03 signature: half the cells are covered, the rest are not."""
    pytest.importorskip("rasterio")
    row, col = 25, 25
    lat_south, lat_north = aggregate.target_lat_bounds(row)
    lon_west, lon_east = aggregate.target_lon_bounds(col)
    # Cover only the western half of the cell.
    mid = (lon_west + lon_east) / 2.0
    path = _write_tile(tmp_path, "s", "half.tif", np.full((20, 20), 1.0, np.float32),
                       lat_south=lat_south, lat_north=lat_north,
                       lon_west=lon_west, lon_east=mid)
    result = aggregate.aggregate_built_surface([path])
    assert result["coverage"][row, col] == pytest.approx(0.5, abs=0.02)
    # And the published fraction is only half of what a complete tile would give.
    assert result["fraction"][row, col] > 0


def test_population_and_height_also_report_coverage(tmp_path):
    pytest.importorskip("rasterio")
    row, col = 25, 25
    lat_south, lat_north = aggregate.target_lat_bounds(row)
    lon_west, lon_east = aggregate.target_lon_bounds(col)
    mid = (lon_west + lon_east) / 2.0
    path = _write_tile(tmp_path, "h", "half.tif", np.full((20, 20), 5.0, np.float32),
                       lat_south=lat_south, lat_north=lat_north,
                       lon_west=lon_west, lon_east=mid)
    for role in (aggregate.aggregate_building_height, aggregate.aggregate_population):
        result = role([path])
        assert result["coverage"][row, col] == pytest.approx(0.5, abs=0.02)
        assert result["coverage"].max() <= 0.51


# --- the aggregate phase refuses a hole (0073 §②) ----------------------------

def test_aggregate_refuses_an_incomplete_layer():
    """This is the guard that would have stopped attempt03 from publishing."""
    driver = load_driver()
    coverage = np.ones((aggregate.GRID_HEIGHT, aggregate.GRID_WIDTH))
    coverage[:16, :12] = 0.0                     # the NW block attempt03 lost
    with pytest.raises(SystemExit) as error:
        driver.require_full_coverage("ghsl_built_s_2015", coverage)
    message = str(error.value)
    assert "ghsl_built_s_2015" in message
    assert "192" in message                      # 16*12 cells, named precisely
    assert "zeros" in message


def test_aggregate_allows_full_coverage():
    driver = load_driver()
    coverage = np.ones((aggregate.GRID_HEIGHT, aggregate.GRID_WIDTH))
    driver.require_full_coverage("ghsl_built_s_2020", coverage)   # must not raise


def test_partial_coverage_is_only_allowed_when_declared():
    driver = load_driver()
    coverage = np.ones((aggregate.GRID_HEIGHT, aggregate.GRID_WIDTH))
    coverage[0, 0] = 0.25
    with pytest.raises(SystemExit):
        driver.require_full_coverage("x", coverage)
    # An explicit, reason-bearing opt-out is the only way through.
    driver.require_full_coverage("x", coverage, allows_partial=True)


def test_floating_point_edge_noise_is_not_a_coverage_hole():
    """Complete tiles leave a ~3.3e-14 deficit; that must not refuse a good layer.

    Measured on the nine complete GHSL tiles: 65 cells come out at
    0.9999999999999672 rather than 1.0. A strict ``< 1.0`` reads those as uncovered
    and would reject a layer that is in fact complete — the mirror image of the
    attempt03 bug, and just as wrong.
    """
    driver = load_driver()
    coverage = np.ones((aggregate.GRID_HEIGHT, aggregate.GRID_WIDTH))
    coverage[:5, :] = 0.9999999999999672
    driver.require_full_coverage("ghsl_built_s_2020", coverage)      # must not raise
    assert driver.COVERAGE_TOLERANCE < 0.5, "tolerance must stay far below a real hole"


def test_a_real_half_cell_hole_is_still_refused():
    driver = load_driver()
    coverage = np.ones((aggregate.GRID_HEIGHT, aggregate.GRID_WIDTH))
    coverage[10, 10] = 0.5
    with pytest.raises(SystemExit):
        driver.require_full_coverage("x", coverage)
