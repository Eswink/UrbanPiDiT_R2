"""Offline tests for the M2 two-month segment read path (#69).

The fixtures shrink the global axes but keep the guard semantics; the pinned
chunk geometry is asserted separately, so a silent upstream rechunk cannot pass.

The property this module exists for is the one the earlier segments could not
provide: the train block must span **two** months so the train-only
``(month, hour)`` climatology has eight buckets instead of four, and both
held-out blocks must be wide enough to carry a 48 h and a 72 h rollout. The M2
request must also still contain the frozen D1 and B2 requests as prefixes, or the
earlier segments stop being prefixes and no earlier baseline transfers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import data.download.earthmover_spatial_m2 as module
from data.download.earthmover_spatial_b2 import B2_LAST_STAMP, B2_STAMPS
from data.download.earthmover_spatial_d1 import (
    FIRST_STAGE_DECODED_BYTES_CAP,
    SPATIAL_PRESSURE_CHUNKS,
    SPATIAL_SURFACE_CHUNKS,
)
from data.download.earthmover_spatial_m2 import (
    M2_LAST_STAMP,
    M2_PART_STAMPS,
    M2_STAMPS,
    extract_m2_part,
    merge_m2_parts,
    _requested_times,
    _second_stage_protocol,
)
from data.download.read_plan_frozen import (
    SECOND_STAGE_DECODED_BYTES_CAP,
    SECOND_STAGE_NEW_ARTIFACT_BYTES_CAP,
)
from tests.test_r7_earthmover_spatial_d1 import (
    SNAPSHOT,
    FakeIc,
    FakeSession,
    FakeZarr,
    spatial_root,
)


def test_frozen_chunk_geometry_is_still_pinned():
    """M2 reuses D1's audited chunk addressing, so the measured layout must hold."""
    assert SPATIAL_SURFACE_CHUNKS == (1, 721, 1440)
    assert SPATIAL_PRESSURE_CHUNKS == (1, 1, 721, 1440)


def test_request_is_the_frozen_60_day_segment():
    stamps, request = _requested_times()
    assert len(stamps) == M2_STAMPS == 240
    assert stamps[0] == pd.Timestamp("2016-01-01T00:00")
    assert stamps[-1] == pd.Timestamp(M2_LAST_STAMP) == pd.Timestamp("2016-02-29T18:00")
    assert request["days"] == 60 and request["steps"] == 240
    assert sorted({stamp.hour for stamp in stamps}) == [0, 6, 12, 18]


def test_request_contains_the_frozen_d1_and_b2_prefixes():
    """The extension must strictly contain both earlier requests, in order."""
    stamps, _ = _requested_times()
    assert stamps.index(pd.Timestamp("2016-01-30T18:00")) == 119
    assert stamps.index(pd.Timestamp(B2_LAST_STAMP)) == B2_STAMPS - 1 == 143
    assert stamps[:B2_STAMPS] == list(pd.date_range("2016-01-01T00:00", B2_LAST_STAMP,
                                                    freq="6h"))


def test_two_month_segment_yields_eight_train_buckets():
    """The whole point of M2: 8 buckets instead of the January segments' 4."""
    stamps, _ = _requested_times()
    train = [t for t in stamps if t < pd.Timestamp("2016-02-17T00:00")]
    buckets = sorted({(t.month, t.hour) for t in train})
    assert len(buckets) == 8, buckets
    assert {month for month, _ in buckets} == {1, 2}
    # Every February bucket is filled by real February days, not a fallback.
    counts = {bucket: sum(1 for t in train if (t.month, t.hour) == bucket)
              for bucket in buckets}
    assert all(count >= 16 for count in counts.values()), counts


def test_held_out_blocks_carry_a_72h_rollout():
    """A 72 h free rollout needs one history frame plus twelve lead frames.

    The January re-cut's 8-stamp val block carried **none**, which is why model
    selection could not read 48/72 h there; both M2 held-out blocks must clear
    that bar, and the test split is the one the sealed evaluation reads.
    """
    stamps, _ = _requested_times()
    ranges = {"val": ("2016-02-17T00:00", "2016-02-23T00:00"),
              "test": ("2016-02-23T00:00", "2016-03-01T00:00")}
    for name, (start, stop) in ranges.items():
        block = [t for t in stamps if pd.Timestamp(start) <= t < pd.Timestamp(stop)]
        assert len(block) == 24 if name == "val" else len(block) == 28
        for lead in (6, 12, 24, 48, 72):
            steps = lead // 6
            windows = sum(1 for i in range(1, len(block)) if i + steps <= len(block) - 1)
            assert windows > 0, f"{name} has no {lead}h window"
        steps = 12
        windows_72 = sum(1 for i in range(1, len(block)) if i + steps <= len(block) - 1)
        assert windows_72 == (11 if name == "val" else 15), (name, windows_72)
        # Every scored valid time stays inside February, so the eight trained
        # buckets all apply and normalized_climatology's fail-closed refusal is
        # never triggered.
        assert {t.month for t in block} == {2}


def test_second_stage_protocol_is_independent_of_the_first_stage_identity():
    """M2 spends the second-stage caps and must not rewrite the first-stage digest.

    The D1/B2 receipts archived ``d3161af5...``; editing the first-stage caps to
    accommodate M2 would make a re-run derive a digest that no longer matches its
    own receipt. M2 therefore has its own caps and its own digest.
    """
    protocol = _second_stage_protocol()
    from data.download.read_plan_frozen import frozen_protocol

    assert protocol["protocol_sha256"] != frozen_protocol()["protocol_sha256"]
    assert protocol["budget_caps"] == {
        "new_artifacts_bytes": SECOND_STAGE_NEW_ARTIFACT_BYTES_CAP,
        "decoded_source_bytes": SECOND_STAGE_DECODED_BYTES_CAP}
    assert SECOND_STAGE_NEW_ARTIFACT_BYTES_CAP > FIRST_STAGE_DECODED_BYTES_CAP
    assert protocol["stage"] == "second"


def test_part_layout_covers_the_segment_without_overlap():
    assert M2_STAMPS % M2_PART_STAMPS == 0
    covered = []
    for offset in range(0, M2_STAMPS, M2_PART_STAMPS):
        covered.extend(range(offset, offset + M2_PART_STAMPS))
    assert covered == list(range(M2_STAMPS))


@pytest.mark.parametrize("offset,stamps", [(0, 0), (-1, 10), (200, 120), (240, 1)])
def test_write_refuses_a_part_outside_the_segment(tmp_path, offset, stamps):
    with pytest.raises(ValueError, match="outside"):
        extract_m2_part(tmp_path / "part.nc", tmp_path / "receipt.json",
                        offset=offset, stamps=stamps)


def test_part_receipt_is_second_stage_and_fail_closed(shrunk_geometry, frozen_request,
                                                     tmp_path, monkeypatch):
    """A budget failure must leave a failed-no-fallback receipt, never data."""
    root, _ = spatial_root(stamps=6)
    monkeypatch.setattr(module, "_open_pinned_session",
                        lambda: (FakeSession(root), FakeIc(), FakeZarr()))
    monkeypatch.setattr(module, "_net_recv_bytes", lambda: 10**9)
    with pytest.raises(RuntimeError, match="BEFORE"):
        extract_m2_part(tmp_path / "part.nc", tmp_path / "receipt.json",
                        offset=0, stamps=6, decoded_budget_bytes=2**20)
    stored = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert stored["status"] == "failed-no-fallback"
    assert stored["synthetic_fallback"] is False
    assert stored["scientific_claim"] is False
    assert stored["read_plan_stage"] == "second"
    assert stored["budget_caps"]["new_artifacts_bytes"] == SECOND_STAGE_NEW_ARTIFACT_BYTES_CAP
    assert not (tmp_path / "part.nc").exists()


def test_merge_refuses_a_part_that_does_not_match_its_receipt(tmp_path):
    """An unverified or incomplete part must never be silently concatenated."""
    nc = tmp_path / "part.nc"
    nc.write_bytes(b"not a netcdf")
    receipt = tmp_path / "part.json"
    receipt.write_text(json.dumps({
        "status": "downloaded-real-source",
        "local_artifact": {"sha256": "0" * 64, "path": "part.nc", "bytes": 12},
        "network_body_bytes": 1,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match its receipt hash"):
        merge_m2_parts([(nc, receipt)], tmp_path / "merged.nc", tmp_path / "merged.json")


def test_merge_refuses_a_failed_part(tmp_path):
    nc = tmp_path / "part.nc"
    nc.write_bytes(b"payload")
    digest = hashlib.sha256(b"payload").hexdigest()
    receipt = tmp_path / "part.json"
    receipt.write_text(json.dumps({
        "status": "failed-no-fallback",
        "local_artifact": {"sha256": digest, "path": "part.nc", "bytes": 7},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="not a completed download"):
        merge_m2_parts([(nc, receipt)], tmp_path / "merged.nc", tmp_path / "merged.json")


def test_merge_refuses_an_existing_output(tmp_path):
    for name in ("merged.nc", "merged.json"):
        (tmp_path / name).write_bytes(b"x")
    with pytest.raises(FileExistsError):
        merge_m2_parts([(tmp_path / "a.nc", tmp_path / "a.json")],
                       tmp_path / "merged.nc", tmp_path / "merged.json")


@pytest.fixture
def shrunk_geometry(monkeypatch):
    """Shrink the pinned chunk geometry to the fixture's 65-point axes.

    ``validate_spatial_namespace`` lives in the D1 module and reads the chunk
    constants from *its own* globals, so the patch has to target that module
    rather than the M2 wrapper that only imports the function.
    """
    import data.download.earthmover_spatial_d1 as d1_module

    monkeypatch.setattr(d1_module, "SPATIAL_SURFACE_CHUNKS", (1, 65, 65))
    monkeypatch.setattr(d1_module, "SPATIAL_PRESSURE_CHUNKS", (1, 1, 65, 65))


@pytest.fixture
def frozen_request(monkeypatch):
    """Replace the real 240-stamp M2 request with the fixture's stamps."""
    times = pd.date_range("2016-01-01", periods=6, freq="6h")
    request = {"days": 1, "start": "2016-01-01T00:00", "steps": 6,
               "first_time": times[0].isoformat(), "last_time": times[-1].isoformat(),
               "init_times": 6, "season": "Jan-Feb"}
    monkeypatch.setattr(module, "_requested_times", lambda: (list(times), request))
    return times
