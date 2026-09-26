"""Offline tests for the B2 segment read path (#64 B2, decision 0008).

The fixtures shrink the global axes but keep the guard semantics; the pinned
chunk geometry is asserted separately, so a silent upstream rechunk cannot pass.
The property this module exists for is structural: the B2 request must strictly
contain the frozen D1 request, or the train range would stop being byte-identical
and B1's train-only baselines would no longer transfer.
"""
from __future__ import annotations

import contextlib
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import data.download.earthmover_spatial_b2 as module
import data.download.earthmover_spatial_d1 as d1_module
from data.download.earthmover_spatial_b2 import (
    B2_LAST_STAMP,
    B2_STAMPS,
    b2_request,
    extract_b2,
    preflight,
    _requested_times,
)
from data.download.earthmover_spatial_d1 import (
    SPATIAL_PRESSURE_CHUNKS,
    SPATIAL_SURFACE_CHUNKS,
)
from tests.test_r7_earthmover_spatial_d1 import (
    SNAPSHOT,
    FakeIc,
    FakeSession,
    FakeZarr,
    spatial_root,
)


def test_frozen_chunk_geometry_is_still_pinned():
    # The B2 path reuses D1's audited chunk addressing, so the measured layout
    # must still hold or the cost model no longer applies.
    assert SPATIAL_SURFACE_CHUNKS == (1, 721, 1440)
    assert SPATIAL_PRESSURE_CHUNKS == (1, 1, 720, 1440) or \
        SPATIAL_PRESSURE_CHUNKS == (1, 1, 721, 1440)


def test_request_is_the_frozen_36_day_segment():
    stamps, request = _requested_times()
    assert len(stamps) == B2_STAMPS == 144
    assert stamps[0] == pd.Timestamp("2016-01-01T00:00")
    assert stamps[-1] == pd.Timestamp(B2_LAST_STAMP)
    assert request["days"] == 36 and request["steps"] == 144
    assert sorted({stamp.hour for stamp in stamps}) == [0, 6, 12, 18]


def test_request_contains_the_frozen_d1_prefix():
    """The whole point of the extension: D1's 120 stamps are the B2 prefix.

    If this broke, B2's train range would differ from D1's, its train-only
    normalization would differ, and the B1 numbers could not be compared with B2.
    """
    stamps, _ = _requested_times()
    assert stamps.index(pd.Timestamp("2016-01-30T18:00")) == 119
    assert len(stamps) - 120 == 24  # the extension is exactly the 6 extra days


def test_request_ends_where_the_extension_says():
    request = b2_request()
    assert request["last_time"] == "2016-02-05T18:00:00"
    assert request["first_time"] == "2016-01-01T00:00:00"
    # 36 days at a 6-hour cadence
    assert (pd.Timestamp(request["last_time"]) - pd.Timestamp(request["first_time"])) \
        == pd.Timedelta(days=35, hours=18)


def test_b2_request_rejects_a_wrong_length():
    with pytest.raises(ValueError, match="144 stamps"):
        b2_request(days=30)


@pytest.fixture
def shrunk_geometry(monkeypatch):
    # The chunk constants live in the D1 module, whose validated read plan B2
    # reuses unmodified; patch them there so the guard semantics are exercised.
    monkeypatch.setattr(d1_module, "SPATIAL_SURFACE_CHUNKS", (1, 65, 65))
    monkeypatch.setattr(d1_module, "SPATIAL_PRESSURE_CHUNKS", (1, 1, 65, 65))


@pytest.fixture
def short_request(monkeypatch):
    """Replace the real 144-stamp request with the shrunk fixture's stamps.

    The B2-specific prefix guard is asserted separately by
    ``test_request_contains_the_frozen_d1_prefix`` against the real request; here
    the point is the read/publish contract, so the request is shrunk to whatever
    the synthetic source holds.
    """
    times = pd.date_range("2016-01-01", periods=6, freq="6h")
    request = {"days": 1, "start": "2016-01-01T00:00", "steps": 6,
               "first_time": times[0].isoformat(), "last_time": times[-1].isoformat(),
               "init_times": 6, "season": "Jan"}
    monkeypatch.setattr(module, "_requested_times", lambda: (list(times), request))
    return times


def test_extract_publishes_artifact_and_honest_receipt(shrunk_geometry, short_request,
                                                       tmp_path, monkeypatch):
    root, times = spatial_root(stamps=6)
    readings = iter([10**9, 10**9 + 500 * 2**20])
    monkeypatch.setattr(module, "_open_pinned_session",
                        lambda: (FakeSession(root), FakeIc(), FakeZarr()))
    monkeypatch.setattr(module, "_net_recv_bytes", lambda: next(readings))
    out_nc = tmp_path / "source.nc"
    receipt_path = tmp_path / "receipt.json"
    receipt = extract_b2(out_nc, receipt_path)
    assert receipt["status"] == "downloaded-real-source"
    assert receipt["source_is_real_reanalysis"] is True
    assert receipt["synthetic_fallback"] is False
    assert receipt["scientific_claim"] is False
    assert receipt["contains_d1_prefix_stamps"] == 120  # the real prefix, declared
    assert len(receipt["timestamps"]) == 6
    assert receipt["network_body_bytes"] == 500 * 2**20
    assert receipt["local_artifact"]["sha256"] == hashlib.sha256(out_nc.read_bytes()).hexdigest()
    assert len(receipt["payloads"]) == 17
    assert all(payload["finite"] for payload in receipt["payloads"].values())
    assert any("36-day" in note for note in receipt["limitations"])
    assert receipt["protocol"]["b2_request"]["days"] == 1  # shrunk fixture request echoed
    assert receipt["decoded_chunk_budget"]["charged_bytes"] > 0


def test_extract_b2_failure_leaves_failed_no_fallback(shrunk_geometry, short_request,
                                                      tmp_path, monkeypatch):
    root, _ = spatial_root(stamps=6)
    readings = iter([10**9, 10**9 + 1024])
    monkeypatch.setattr(module, "_open_pinned_session",
                        lambda: (FakeSession(root), FakeIc(), FakeZarr()))
    monkeypatch.setattr(module, "_net_recv_bytes", lambda: next(readings))

    def explode(plan, budget, deadline):
        raise ValueError("source turned nonfinite mid-stream")
    monkeypatch.setattr(module, "_collect_frames", explode)
    out_nc = tmp_path / "source.nc"
    receipt_path = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="nonfinite"):
        extract_b2(out_nc, receipt_path)
    stored = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert stored["status"] == "failed-no-fallback"
    assert stored["synthetic_fallback"] is False
    assert not out_nc.exists()


def test_extract_b2_refuses_existing_outputs(shrunk_geometry, short_request, tmp_path,
                                             monkeypatch):
    root, _ = spatial_root(stamps=2)
    monkeypatch.setattr(module, "_open_pinned_session",
                        lambda: (FakeSession(root), FakeIc(), FakeZarr()))
    out_nc = tmp_path / "source.nc"
    out_nc.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="refusing existing output"):
        extract_b2(out_nc, tmp_path / "receipt.json")


def test_preflight_writes_nothing_without_a_report_path(shrunk_geometry, short_request,
                                                        tmp_path, monkeypatch, capsys):
    root, _ = spatial_root(stamps=6)
    monkeypatch.setattr(module, "_open_pinned_session",
                        lambda: (FakeSession(root), FakeIc(), FakeZarr()))
    report = preflight()
    assert report["mode"] == "read-only-preflight"
    assert report["scientific_training_certified"] is False
    assert capsys.readouterr().out
    assert list(tmp_path.glob("*")) == []


def test_preflight_report_path_is_exclusive(shrunk_geometry, short_request, tmp_path,
                                            monkeypatch):
    root, _ = spatial_root(stamps=6)
    monkeypatch.setattr(module, "_open_pinned_session",
                        lambda: (FakeSession(root), FakeIc(), FakeZarr()))
    target = tmp_path / "report.json"
    preflight(target)
    assert target.is_file()
    with pytest.raises(FileExistsError):
        preflight(target)
