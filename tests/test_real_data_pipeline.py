from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from data.download.uci_beijing import download_uci_beijing
from data.preprocess.real_station_smoke import build_real_station_smoke

ROOT = Path(__file__).resolve().parents[1]
DYNAMIC = ROOT / "data/raw/real_smoke/beijing_uci_pm25_smoke.csv"
STATIC = ROOT / "data/raw/real_smoke/beijing_dongcheng_boundary_smoke.geojson"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_real_fixture_checksums_are_stable():
    assert _sha(DYNAMIC) == "277a2b00094398a9959aa8d4060b9035d581c89fbc009309097fd3bc38241129"
    assert _sha(STATIC) == "ac90e8136273f7b41c4c03b673feb274bf51438b30b0f59976d0e93d43db90d6"


def test_preprocess_uses_real_static_and_no_temporal_split_overlap(tmp_path: Path):
    processed = tmp_path / "processed"
    manifests = tmp_path / "manifests"
    paths = build_real_station_smoke(
        DYNAMIC,
        processed,
        manifests,
        static_geojson_path=STATIC,
        history_steps=4,
        coarse_hw=(8, 8),
        urban_hw=(32, 32),
        stride=2,
    )

    splits = {k: _read_jsonl(v) for k, v in paths.items()}
    assert all(splits.values())
    row_ranges = {
        split: (
            min(r["source_row_start"] for r in rows),
            max(r["source_row_target"] for r in rows),
        )
        for split, rows in splits.items()
    }
    assert row_ranges["train"][1] < row_ranges["val"][0]
    assert row_ranges["val"][1] < row_ranges["test"][0]

    for prev, nxt in [("train", "val"), ("val", "test")]:
        prev_end = max(datetime.fromisoformat(r["target_time"]) for r in splits[prev])
        next_start = min(datetime.fromisoformat(r["start_time"]) for r in splits[nxt])
        assert prev_end < next_start

    first = splits["train"][0]
    assert first["dynamic_is_real_observation"] is True
    assert first["static_contains_real_geodata"] is True
    assert first["scientific_training_ready"] is False
    assert first["spatiotemporal_colocation_valid"] is False

    artifact_path = (paths["train"].parent / first["path"]).resolve()
    with np.load(artifact_path, allow_pickle=False) as z:
        assert z["coarse_history"].shape == (4, 4, 8, 8)
        assert z["urban_history"].shape == (4, 1, 32, 32)
        assert z["urban_static"].shape == (4, 32, 32)
        assert z["urban_target"].shape == (1, 32, 32)
        assert np.isfinite(z["coarse_history"]).all()
        assert np.isfinite(z["urban_static"]).all()
        # 真实东城区 mask / 派生 edge 必须非空、非全满，否则 GeoJSON 栅格化失效。
        district_mask = z["urban_static"][2]
        district_edge = z["urban_static"][3]
        assert 0.05 < float(district_mask.mean()) < 0.95
        assert float(district_edge.sum()) > 0
        # 第一条原始记录：TEMP=-11, DEWP=-21, PRES=1021, Iws=1.79。
        np.testing.assert_allclose(
            z["coarse_history"][0, :, 0, 0],
            np.array([-11 / 40, -21 / 40, 21 / 50, 1.79 / 100], dtype=np.float32),
            rtol=0,
            atol=1e-6,
        )

    provenance = json.loads((manifests / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["dynamic"]["source_sha256"] == _sha(DYNAMIC)
    assert provenance["static"]["source_sha256"] == _sha(STATIC)
    assert provenance["static"]["is_real_geodata"] is True
    assert provenance["static"]["is_urban_morphology"] is False
    assert provenance["scientific_training_ready"] is False


def test_uci_fixture_copy_is_byte_identical(tmp_path: Path):
    copied = download_uci_beijing(tmp_path, fixture=DYNAMIC)
    assert copied.read_bytes() == DYNAMIC.read_bytes()


def test_uci_network_failure_never_silently_synthesizes(monkeypatch, tmp_path: Path):
    import data.download.uci_beijing as mod

    def fail(*args, **kwargs):
        raise OSError("offline-by-test")

    monkeypatch.setattr(mod, "_download", fail)
    with pytest.raises(RuntimeError, match="未使用合成数据回退"):
        download_uci_beijing(tmp_path)
    assert not (tmp_path / mod.CSV_NAME).exists()


def test_worldcover_beijing_tile_and_url_are_deterministic():
    from data.download.worldcover_cog import worldcover_cog_url, worldcover_tile_id

    tile = worldcover_tile_id(116.40, 39.90)
    assert tile == "N39E114"
    assert worldcover_cog_url(tile, year=2021) == (
        "https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
        "v200/2021/map/ESA_WorldCover_10m_2021_v200_N39E114_Map.tif"
    )
