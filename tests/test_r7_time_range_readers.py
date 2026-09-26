"""Read-side agreement with a store's declared ``split_time_ranges`` (#64 D-2).

The builder gained the time-range split mode in decision 0005, but every
*reader* still decided window ownership by calendar year. On the frozen D1
segment - one continuous January of a single year - the year filter cannot
discriminate the declared train window from the declared val/test windows, so

- ``r7_store.validate_record`` rejected every val/test record (0/10 each),
- ``ZarrRolloutDataset`` produced no val/test windows at all, and
- ``fit_training_climatology`` silently fitted on all 120 steps, including the
  24 held-out val/test steps (train-only is 96).

These tests pin the reader side to the store's own declaration. The default
year mode must stay bit-for-bit unchanged, so its behaviour is re-asserted here
on the same fixtures.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from data.preprocess.r7_era5_zarr import build_r7_era5_zarr_from_dataset
from data.r7_evaluation import ZarrRolloutDataset, fit_training_climatology
from data.r7_store import split_time_labels, validate_record, validate_store
from test_r7_time_range_splits import RANGES, SPECS, build, regional_dataset

# fixture RANGES subdivides 2016-01-01 .. 2016-01-20 (80 six-hourly steps):
# train [01-01, 01-10) = 36 steps, val [01-10, 01-15) = 20, test [01-15, 01-20) = 20.
# The final day sits outside every range, which is what makes the fallback to
# "all 2016 steps" observable.
TRAIN_STEPS, VAL_STEPS, TEST_STEPS, OUTSIDE_STEPS = 36, 20, 20, 4


def _records(manifest_dir, split):
    return [json.loads(line) for line in
            (Path(manifest_dir) / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _timestamps(root):
    return pd.DatetimeIndex(np.asarray(root["time_ns"][:]).astype("datetime64[ns]"))


def _year_mode_store(tmp_path, stamps=40):
    """A three-year store built through the default, year-based path."""
    blocks = []
    for year in (2018, 2019, 2020):
        block = regional_dataset(stamps)
        block = block.assign_coords(time=pd.date_range(f"{year}-01-01", periods=stamps, freq="6h"))
        blocks.append(block)
    return build_r7_era5_zarr_from_dataset(
        xr.concat(blocks, dim="time"), store_path=tmp_path / "store.zarr",
        manifest_dir=tmp_path / "manifests", specs=SPECS,
        split_years={"train": [2018], "val": [2019], "test": [2020]},
        time_chunk=16, spatial_chunk=(64, 64), compute_process_targets=False)


def test_split_time_labels_are_none_in_year_mode(tmp_path):
    """Year mode reports no per-step labels, so readers keep the year path."""
    _year_mode_store(tmp_path, stamps=8)
    root = zarr.open_group(str(tmp_path / "store.zarr"), mode="r")
    assert "split_mode" not in root.attrs
    assert split_time_labels(root) is None


def test_range_store_labels_match_declared_ranges(tmp_path):
    build(tmp_path, stamps=80)
    root = zarr.open_group(str(tmp_path / "store.zarr"), mode="r")
    validate_store(root)
    labels = split_time_labels(root)
    assert labels is not None
    counts = {split: int((labels == split).sum()) for split in ("train", "val", "test")}
    assert counts == {"train": TRAIN_STEPS, "val": VAL_STEPS, "test": TEST_STEPS}
    assert int((labels == "").sum()) == OUTSIDE_STEPS


def test_validate_record_accepts_every_published_range_record(tmp_path):
    """The regression: val/test records must validate against their own ranges."""
    paths = build(tmp_path, stamps=80)
    root = zarr.open_group(str(tmp_path / "store.zarr"), mode="r")
    observed = {}
    for split in ("train", "val", "test"):
        records = _records(paths[split].parent, split)
        for record in records:
            try:
                validate_record(root, record)
            except ValueError as error:  # pragma: no cover - failure path
                pytest.fail(f"{split} record {record['sample_id']} rejected: {error}")
        observed[split] = (len(records), len(records))
    assert observed == {"train": (34, 34), "val": (18, 18), "test": (18, 18)}
    # Under the old year-only check every held-out record failed, because all
    # steps are 2016 while val/test declare 2017/2018.
    assert root.attrs["split_years"] == {"train": [2016], "val": [2017], "test": [2018]}


def test_validate_record_still_rejects_a_crossing_record(tmp_path):
    """Ownership by range must not become a blanket amnesty."""
    paths = build(tmp_path, stamps=80)
    root = zarr.open_group(str(tmp_path / "store.zarr"), mode="r")
    record = _records(paths["train"].parent, "train")[0]
    with pytest.raises(ValueError, match="crosses split boundaries"):
        validate_record(root, dict(record, split="val"))


def test_range_store_declaring_unparseable_ranges_fails_closed(tmp_path):
    """A range-mode store with a broken declaration must not fall back to years."""
    build(tmp_path, stamps=80)
    root = zarr.open_group(str(tmp_path / "store.zarr"), mode="r")
    stamped = root.attrs["split_time_ranges"]
    broken = dict(stamped)
    broken["val"] = [["2016-01-20", "2016-01-10"]]  # start >= stop
    fake = type("_Attrs", (), {"attrs": dict(root.attrs, split_time_ranges=broken),
                               "__getitem__": lambda self, key: root[key]})()
    with pytest.raises(ValueError, match="start < stop"):
        split_time_labels(fake)
    # the real store is untouched and still parses
    assert root.attrs["split_time_ranges"] == stamped
    assert split_time_labels(root) is not None


def test_rollout_dataset_finds_held_out_windows(tmp_path):
    build(tmp_path, stamps=80)
    store = tmp_path / "store.zarr"
    root = zarr.open_group(str(store), mode="r")
    labels = split_time_labels(root)
    for split in ("val", "test"):
        dataset = ZarrRolloutDataset(str(store), split=split, lead_hours=(6, 12))
        assert len(dataset.windows) == 17
        for history, targets in dataset.windows:
            for position in history + targets:
                assert labels[position] == split
    with pytest.raises(ValueError, match="held-out"):
        ZarrRolloutDataset(str(store), split="train", lead_hours=(6,))


def test_climatology_uses_only_the_declared_train_window(tmp_path):
    """80 steps with 36 train, not all 80: the D1 120 -> 96 defect in miniature."""
    build(tmp_path, stamps=80)
    climatology = fit_training_climatology(tmp_path / "store.zarr")
    assert climatology["selection"] == "declared_train_time_ranges"
    assert climatology["n_selected_steps"] == TRAIN_STEPS
    assert sum(climatology["counts"].values()) == TRAIN_STEPS
    # one January bucket per synoptic hour, nine train steps each
    assert climatology["counts"] == {(1, hour): 9 for hour in (0, 6, 12, 18)}
    assert climatology["train_time_ranges"] == [["2016-01-01T00:00:00", "2016-01-10T00:00:00"]]
    assert climatology["split_mode"] == "time_ranges"
    # the (37th) first excluded step is exactly the declared val-range start
    root = zarr.open_group(str(tmp_path / "store.zarr"), mode="r")
    assert _timestamps(root)[TRAIN_STEPS].isoformat() == "2016-01-10T00:00:00"


def test_year_mode_reader_behaviour_is_unchanged(tmp_path):
    """The year path must still select by declared years only."""
    paths = _year_mode_store(tmp_path, stamps=40)
    root = zarr.open_group(str(tmp_path / "store.zarr"), mode="r")
    assert split_time_labels(root) is None
    for split in ("train", "val", "test"):
        for record in _records(paths[split].parent, split):
            validate_record(root, record)
    climatology = fit_training_climatology(tmp_path / "store.zarr")
    assert climatology["selection"] == "declared_train_years"
    assert climatology["n_selected_steps"] == 40  # the whole declared train year
    assert climatology["training_years"] == [2018]
    assert "split_mode" not in climatology
    times = _timestamps(root)
    dataset = ZarrRolloutDataset(str(tmp_path / "store.zarr"), split="val", lead_hours=(6, 12))
    expected = [i for i, stamp in enumerate(times)
                if stamp.year == 2019 and i >= 1 and i + 2 < len(times)
                and all(times[j].year == 2019 for j in (i - 1, i, i + 1, i + 2))]
    assert dataset.windows == [([i - 1, i], [i + 1, i + 2]) for i in expected]
    assert len(dataset.windows) == 37


def test_publication_replay_delegates_ownership_to_the_readers():
    """The D1 replay must call the readers, not recompute the rule itself.

    `scripts/verify_r7_d1_store.py` once derived split ownership locally, so it
    stayed green while `validate_record` rejected all 20 held-out records. The
    check here is structural: the replay must import the shared helper and the
    reader-side validator, and must not rebuild a local range mask.
    """
    import ast
    source = (Path(__file__).resolve().parents[1] / "scripts" / "verify_r7_d1_store.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    imported = {alias.name for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names}
    assert "split_time_labels" in imported
    assert "validate_record" in imported
    assert "fit_training_climatology" in imported
    assert "validate_record(root, record)" in source
    # a local re-derivation of ownership would reintroduce the blind spot
    assert "split_of[(stamps_ns >= begin)" not in source
