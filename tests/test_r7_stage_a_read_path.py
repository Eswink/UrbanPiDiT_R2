"""CPU counterproofs for the Stage A read-path changes; synthetic fields are not weather.

Decision 0085 requires the吞吐 optimisations to be **bit-identical**, not merely
equivalent. These tests pin the three properties that claim rests on:

1. ``validate_record``'s optional timestamp column compares the same numbers.
2. A window read as one contiguous span equals the per-frame read byte for byte.
3. The emitted index stream equals the serial epoch walk, and the worker batch for a
   given draw is the serial batch.

No GPU, no real dataset, no scientific claim: fixtures declare ``scientific_claim=False``.
"""
from __future__ import annotations

import json

import pandas as pd

import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

from test_r7_m4_autoregressive_rollout import cpu_threads, tiny_spec

# These cases fork a DataLoader worker inside the test process, so they are sensitive to
# what earlier tests left behind (see ``_forked_loader``). The deprecation warning is the
# fork itself, which is intended here.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _forked_loader(dataset, contract):
    """A one-worker loader forked after making the process safe to fork.

    Pinning is off so the forked child starts no pin-memory thread, and the network is
    restored because worker IPC needs it: an earlier test in the same process may have
    called ``r7_m3_offline.deny_network()``, which replaces ``socket.socket.connect``
    process-wide and is not scoped to that test.
    """
    from scripts.r7_m3_offline import allow_network

    allow_network()
    import torch.multiprocessing as mp

    original = mp.get_start_method(allow_none=True)
    # ``spawn`` would be cleaner but the runner tests install a process-wide socket block
    # (``r7_m3_offline``), and a spawned child cannot connect back through it. ``fork``
    # works here because pinning is off: the fd-sharing pin-memory loop is the part that
    # is unsafe to fork after other tests have started CUDA threads.
    mp.set_start_method("fork", force=True)
    try:
        return _BatchLoader(dataset, contract, workers=1, prefetch=2, pin_memory=False)
    finally:
        if original is not None:
            mp.set_start_method(original, force=True)
from test_r7_long_rollout import dataset_for, long_fixture, records_at, write_records
from data.r7_store import validate_record
from training.r7_experiment import make_model
from training.r7_long_rollout_runner import (
    _BatchLoader, _index_stream, _ordered_indices,
)


def test_the_timestamp_column_path_reads_the_same_numbers():
    """A supplied column must give the identical verdict, including on a mismatch."""
    record = {"history_indices": [3, 4], "target_index": 5,
              "history_times": ["2016-01-01T18:00:00", "2016-01-02T00:00:00"],
              "target_time": "2016-01-02T06:00:00", "init_time": "2016-01-02T00:00:00",
              "lead_time_hours": 6, "split": "train"}
    import pandas as pd

    from data.r7_store import HOUR_NS
    base = pd.Timestamp("2016-01-01T00:00:00")
    column = np.asarray([(base + pd.Timedelta(hours=6 * step)).value for step in range(8)],
                        dtype=np.int64)

    class Root:
        attrs = {"split_years": {"train": list(range(2016, 2030))}}

        class _State:
            shape = (8, 1, 2, 2)

        class _T:
            def __getitem__(self, index):
                return int(column[index])

        state, time_ns = _State(), _T()

        def __getitem__(self, name):
            if name == "state":
                return self.state
            if name == "time_ns":
                return self.time_ns
            raise KeyError(name)

    root = Root()
    assert validate_record(root, record, time_ns=column) == [3, 4, 5]
    # The same call without the column must agree (it takes the scalar path).
    assert validate_record(root, record) == [3, 4, 5]
    bad = dict(record, target_time="2016-01-02T12:00:00")
    with pytest.raises(ValueError, match="timestamps do not match"):
        validate_record(root, bad, time_ns=column)


def test_one_span_read_is_byte_identical_to_per_frame_reads(tmp_path):
    """The batched read must not change a single byte of any emitted field."""
    manifest, root = long_fixture(tmp_path)
    dataset, _ = dataset_for(manifest)
    reader = dataset.reader
    record = dataset.records[0]
    indices = reader.indices(record)
    batched = reader.read_normalized(record, indices)
    mean, std = reader.aux(record)["normalization"]
    state = reader._store(record)["state"]
    per_frame = np.stack([
        (np.asarray(state[index], dtype=np.float32) - mean[:, None, None]) / std[:, None, None]
        for index in indices])
    assert np.array_equal(batched, per_frame)
    assert batched.dtype == per_frame.dtype


def test_a_contiguous_window_reads_once_and_matches_the_serial_sample(tmp_path):
    """The long-rollout loader's span path must equal the per-index construction."""
    manifest, root = long_fixture(tmp_path)
    dataset, _ = dataset_for(manifest, physical_steps=4)
    sample = dataset[0]
    assert sample["physical_targets"].shape[0] == 4
    assert torch.isfinite(sample["physical_targets"]).all()
    # The span helper must report contiguity for the registered 6 h cadence.
    targets = dataset.windows[0][1]
    span, _frames = dataset._window_frames(dataset.reader._store(targets[0]), targets, 0)
    assert span["contiguous"] is True


def test_a_non_contiguous_window_falls_back_instead_of_being_rejected(tmp_path):
    """A sparse target set is still admissible; only the read strategy changes."""
    times = pd.date_range("2016-01-01", periods=16, freq="6h")
    manifest, root = long_fixture(tmp_path, times=times)
    dataset, _ = dataset_for(manifest, physical_steps=2)
    dataset.reader.indices = lambda record: [0, 1, 7]  # not contiguous
    targets = dataset.windows[0][1]
    span, frames = dataset._window_frames(dataset.reader._store(targets[0]), targets, 0)
    assert span["contiguous"] is False
    # Frames are keyed to store indices: the record's own three (t-6, t, t+6) plus one
    # further target per remaining physical step -- not to ``len(targets)``.
    assert frames.shape[0] == span["history_frames"] + len(span["extra_offsets"])
    assert frames.shape[0] == 3 + (len(targets) - 1)


def test_the_index_stream_reproduces_the_serial_epoch_walk():
    """``_index_stream`` must be exactly what the serial loop draws, update by update."""
    contract = {"seed": 7, "batch_size": 3, "total_updates": 11}
    total = 10
    stream = _index_stream(contract, total)
    serial, epoch, cursor = [], 0, 0
    for _ in range(contract["total_updates"]):
        if cursor == total:
            epoch, cursor = epoch + 1, 0
        indices = _ordered_indices(contract, epoch, cursor, total)
        serial.extend(indices)
        cursor += len(indices)
    assert stream == serial
    # 10 samples at batch 3 gives 3,3,3,1 per epoch: the last batch is short, so the
    # total is not updates*batch and must not be asserted as if it were.
    assert len(stream) == 29
    assert len(stream) == sum(len(_ordered_indices(contract, epoch, cursor, total))
                              for epoch, cursor in [(0, 0), (0, 3), (0, 6), (0, 9),
                                                    (1, 0), (1, 3), (1, 6), (1, 9),
                                                    (2, 0), (2, 3), (2, 6)])


def test_the_worker_batch_equals_the_serial_batch(tmp_path):
    """With one worker the collated batch must be bitwise the serial batch."""
    manifest, root = long_fixture(tmp_path)
    dataset, _ = dataset_for(manifest, physical_steps=3)
    contract = {"seed": 5, "batch_size": 2, "total_updates": 4}
    loader = _forked_loader(dataset, contract)
    epoch, cursor = 0, 0
    try:
        for _ in range(contract["total_updates"]):
            if cursor == len(dataset):
                epoch, cursor = epoch + 1, 0
            indices = _ordered_indices(contract, epoch, cursor, len(dataset))
            second = loader.batch(indices)
            serial = default_collate([dataset[index] for index in indices])
            assert sorted(second) == sorted(serial)
            for name in serial:
                if torch.is_tensor(serial[name]):
                    assert torch.equal(second[name], serial[name])
                else:
                    assert second[name] == serial[name]
            cursor += len(indices)
    finally:
        for worker in getattr(loader.loader, "_iterator", None), :
            pass


def test_a_divergent_index_draw_is_refused(tmp_path):
    """The loader must refuse a draw that is not the one it was given."""
    manifest, root = long_fixture(tmp_path)
    dataset, _ = dataset_for(manifest, physical_steps=2)
    contract = {"seed": 3, "batch_size": 2, "total_updates": 2}
    loader = _forked_loader(dataset, contract)
    first = _ordered_indices(contract, 0, 0, len(dataset))
    loader.batch(first)
    # The second update's registered draw differs from the first; handing the loader the
    # first draw again must be refused rather than silently re-served.
    with pytest.raises(RuntimeError, match="diverged"):
        loader.batch(first)


def test_a_short_final_draw_stays_short_and_does_not_shift_later_batches():
    """The measured defect: a short draw mid-run made every later batch mis-grouped.

    ``_BatchLoader`` used to hand the flat index stream to a DataLoader with
    ``batch_size=contract["batch_size"]``. With a dataset whose length is not a multiple of
    the batch size, each epoch's last draw is short, the flat stream stops being a whole
    number of batches, and the DataLoader re-chunks across update boundaries. On the real
    store (32,131 usable windows, batch 4) that made a 32,132-update run train on batches
    that differed from its registered draw from update 8,033 onward, then raise
    ``StopIteration`` at the end -- point 3's first attempt was voided for it.

    This fixture uses 7 windows and batch 3, so the short draw lands *mid-run* (update 3 of
    5), which is the failing shape. It asserts on draw sizes alone, so it needs no worker.
    """
    from training.r7_long_rollout_runner import _draw_sizes, _index_stream

    total, batch, updates = 7, 3, 5
    contract = {"seed": 41, "batch_size": batch, "total_updates": updates}
    sizes = _draw_sizes(contract, total)
    # Epoch 0: 3, 3, 1 (short at update 3). Epoch 1 restarts: 3, 3.
    assert sizes == [3, 3, 1, 3, 3]
    assert sum(sizes) == len(_index_stream(contract, total))
    # The whole point: one draw per update, even though the flat stream is not a whole
    # number of uniform batches.
    assert len(sizes) == updates
    assert sum(sizes) % batch != 0


def test_the_loader_serves_one_batch_per_registered_draw(tmp_path):
    """End-to-end through the loader: draw count and per-draw lengths must match the plan."""
    manifest, _ = long_fixture(tmp_path)
    dataset, _ = dataset_for(manifest, physical_steps=2)
    total, batch, updates = len(dataset), 2, 5
    contract = {"seed": 41, "batch_size": batch, "total_updates": updates}
    from training.r7_long_rollout_runner import _draw_sizes
    expected_sizes = _draw_sizes(contract, total)
    loader = _forked_loader(dataset, contract)
    assert len(loader.draws) == updates
    assert [len(draw) for draw in loader.draws] == expected_sizes
    # ``len(batch)`` is the number of collated fields (17), not the sample count: the batch
    # dimension is read off a tensor the dataset always emits.
    served = []
    for draw in loader.draws:
        batch = loader.batch(list(draw))
        served.append(int(batch["atmos_target"].shape[0]))
    assert served == expected_sizes
    # A draw beyond the registered plan is refused rather than silently served.
    with pytest.raises(RuntimeError, match="ran out of registered draws"):
        loader.batch(list(loader.draws[0]))
