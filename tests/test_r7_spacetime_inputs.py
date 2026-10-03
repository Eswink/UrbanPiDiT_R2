"""D1/D5: the known initialization-time conditions really enter the forecaster.

Three claims, each with the failure it guards against:

- the readers export the window's initialization time, derived from the store's
  own ``time_ns`` entry, checked against an independent parse of the manifest
  timestamp in the test (not against the helper the reader calls);
- the backbone consumes it: the output moves when the phase or the lead moves and
  stays bitwise unchanged when a field the model does not declare is changed;
  geography enters *per token*, which a domain-mean scalar could not do - the
  latitude axis is reversed while its mean is held fixed;
- a declared field that is missing raises instead of falling back to the server
  clock or to a placeholder, and the module contains no clock source at all.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest
import torch

from data.r7_evaluation import ZarrRolloutDataset
from data.r7_zarr_dataset import ZarrAtmosWindowDataset
from model.process_forecast_r7 import ProcessForecastCoReasoner
from model.r7_halting import forecast_inputs
from model.spacetime_conditioning_r7 import (CALENDAR_INPUT_FIELDS, SPACETIME_INPUT_FIELDS, SpacetimeConditioning,
    phase_features, position_features)
from training.r7_experiment import make_model, seed_everything
from test_r7_storage_safety import build

MODULE_PATH = Path(__file__).resolve().parents[1] / "model" / "spacetime_conditioning_r7.py"
TINY = {"in_channels": 1, "out_channels": 1, "history_steps": 2, "dim": 16, "depth": 1,
        "heads": 2, "window_size": 2, "patch_size": 2, "dropout": 0.0,
        "anchored_processes": 1, "free_processes": 1, "default_reasoning_steps": 1,
        "spacetime_inputs": True}


def _sample(batch_size: int = 2, hw: tuple[int, int] = (4, 6), seed: int = 5):
    generator = torch.Generator().manual_seed(seed)
    return {
        "coarse_history": torch.randn(batch_size, 2, 1, *hw, generator=generator),
        "atmos_target": torch.randn(batch_size, 1, *hw, generator=generator),
        "lead_time_hours": torch.full((batch_size,), 6.0),
        "latitude": torch.linspace(60.0, 25.0, hw[0]),
        "longitude": torch.linspace(100.0, 112.0, hw[1]),
        "init_utc_hour": torch.tensor(3.0),
        "init_day_of_year": torch.tensor(15.0),
        "init_year": torch.tensor(2016.0),
        "grid_spacing_deg": torch.tensor(0.25),
        "sample_id": "synthetic_00000",
        "process_targets": torch.zeros(batch_size, 1),
    }


def _model():
    seed_everything(7)
    return make_model("process", dict(TINY)).eval()


def _forecast(model, batch):
    with torch.no_grad():
        return model(forecast_inputs(batch), reasoning_steps=1).forecast


def test_the_training_reader_exports_the_init_time_of_the_window(tmp_path):
    """The exported phase fields are the window's own initialization time."""
    paths = build(tmp_path)
    dataset = ZarrAtmosWindowDataset(paths["train"])
    sample = dataset[0]
    record = dataset.records[0]
    stamp = pd.Timestamp(record["init_time"])
    assert record["init_time"] == record["history_times"][-1]  # init == last history step
    assert float(sample["init_utc_hour"]) == stamp.hour
    assert float(sample["init_day_of_year"]) == stamp.dayofyear
    assert float(sample["init_year"]) == stamp.year
    assert float(sample["init_calendar_year"]) == stamp.year
    history = sample["coarse_history"]
    assert sample["latitude"].shape == (history.shape[-2],)
    assert sample["longitude"].shape == (history.shape[-1],)


def test_the_rollout_reader_agrees_with_the_training_reader(tmp_path):
    """Both producers define "initialization time" the same way, or not at all."""
    build(tmp_path)
    training = ZarrAtmosWindowDataset(tmp_path / "manifest" / "train.jsonl")[0]
    rollout = ZarrRolloutDataset(tmp_path / "output", split="val", lead_hours=(6,))[0]
    stamp = pd.Timestamp(rollout["init_time"])
    assert float(rollout["init_utc_hour"]) == stamp.hour
    assert float(rollout["init_day_of_year"]) == stamp.dayofyear
    assert float(rollout["init_year"]) == stamp.year
    assert float(rollout["init_calendar_year"]) == stamp.year
    expected = {"coarse_history", "lead_time_hours"} | set(SPACETIME_INPUT_FIELDS) | set(CALENDAR_INPUT_FIELDS)
    assert set(forecast_inputs(training)) == expected
    assert set(forecast_inputs(rollout)) == expected


@pytest.mark.parametrize("field", SPACETIME_INPUT_FIELDS)
def test_a_missing_declared_field_fails_instead_of_falling_back(field):
    """Counterproof for the guard: remove one field and the forward must refuse.

    With the switch off the same batch is accepted, which is what makes this a
    statement about the pathway rather than about the batch.
    """
    batch = _sample()
    del batch[field]
    model = _model()
    with pytest.raises((KeyError, ValueError, TypeError), match=field):
        _forecast(model, batch)
    seed_everything(7)
    without = make_model("process", {**TINY, "spacetime_inputs": False}).eval()
    assert _forecast(without, batch).shape == (2, 1, 4, 6)


def test_the_output_follows_the_declared_fields_and_only_those():
    """Field-level causality: phase and lead move it, undeclared fields do not."""
    model = _model()
    base = _sample()
    reference = _forecast(model, base)

    moved = {
        "lead_time_hours": torch.full((2,), 12.0),
        "init_utc_hour": torch.tensor(9.0),
        "init_day_of_year": torch.tensor(200.0),
        "latitude": torch.linspace(25.0, 60.0, 4),
    }
    for field, value in moved.items():
        batch = dict(base, **{field: value})
        assert not torch.equal(_forecast(model, batch), reference), field

    unread = {
        "atmos_target": torch.full_like(base["atmos_target"], 1e6),
        "process_targets": torch.full_like(base["process_targets"], 1e6),
        "sample_id": "a_different_id",
        "grid_spacing_deg": torch.tensor(1.0),
        "init_year": torch.tensor(1999.0),
    }
    for field, value in unread.items():
        batch = dict(base, **{field: value})
        assert torch.equal(_forecast(model, batch), reference), (
            f"{field} is not declared as a model input but changed the output")


def test_geography_enters_per_token_not_as_a_domain_mean():
    """Reversing the latitude axis keeps its mean and must still change the output.

    A per-sample scalar built from the latitudes would be invariant under this
    reversal; a per-token position term is not.
    """
    model = _model()
    batch = _sample()
    reversed_latitude = batch["latitude"].flip(0)
    assert float(batch["latitude"].mean()) == float(reversed_latitude.mean())
    assert not torch.equal(_forecast(model, dict(batch, latitude=reversed_latitude)),
                           _forecast(model, batch))

    features = position_features(batch["latitude"], batch["longitude"],
                                token_hw=(2, 3), patch_size=2)
    assert features.shape == (6, 4)
    assert not torch.equal(features[0], features[1])          # rows differ
    assert not torch.equal(features[0], features[3])          # columns differ
    reversed_features = position_features(reversed_latitude, batch["longitude"],
                                         token_hw=(2, 3), patch_size=2)
    assert not torch.equal(features, reversed_features)


def test_the_phase_is_the_valid_time_and_wraps():
    """``init + lead`` for both terms, with the periodicity coming from the harmonics."""
    midnight_crossing = phase_features(torch.tensor([23.0]), torch.tensor([15.0]),
                                       torch.tensor([12.0]))
    next_morning = phase_features(torch.tensor([11.0]), torch.tensor([16.0]),
                                  torch.tensor([0.0]))
    # 12 h after 23:00 on day 15 is 11:00 on day 16: the same valid time, reached
    # two different ways.
    assert torch.allclose(midnight_crossing, next_morning, rtol=0, atol=1e-6)
    # The diurnal pair alone repeats across midnight; the annual pair must not,
    # because the date has advanced by a day.
    same_day = phase_features(torch.tensor([11.0]), torch.tensor([15.0]),
                              torch.tensor([0.0]))
    assert torch.allclose(midnight_crossing[0, 2:], same_day[0, 2:], rtol=0, atol=1e-6)
    assert not torch.allclose(midnight_crossing[0, :2], same_day[0, :2], rtol=0, atol=1e-6)
    # A 24 h lead from day 1 00:00 is day 2 00:00, and the annual phase therefore
    # advances with the lead rather than only with the day of year.
    midnight = phase_features(torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([0.0]))
    day_later = phase_features(torch.tensor([0.0]), torch.tensor([2.0]), torch.tensor([0.0]))
    lead_of_a_day = phase_features(torch.tensor([0.0]), torch.tensor([1.0]),
                                   torch.tensor([24.0]))
    assert torch.allclose(lead_of_a_day, day_later, rtol=0, atol=1e-6)
    assert not torch.equal(midnight, lead_of_a_day)


def test_the_conditioning_module_reads_no_clock():
    """Counterproof for "no server clock": the module cannot reach one.

    The phase has to be a function of the declared fields and the requested lead.
    A `datetime.now()` (or a `time.time()`) added later would break that, and this
    scan is what notices.
    """
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"time", "datetime", "calendar", "dateutil", "arrow", "pendulum"}
    forbidden = {"now", "today", "utcnow", "time", "time_ns", "monotonic"}
    used = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not used & forbidden, f"clock-like attribute access: {used & forbidden}"


def test_the_conditioning_term_is_added_to_the_token_grid():
    """The term has one vector per token and is added, not substituted."""
    module = SpacetimeConditioning(dim=16, patch_size=2)
    batch = _sample()
    history = batch["coarse_history"]
    tokens = torch.zeros(2, 6, 16)
    term = module(batch, history=history, token_hw=(2, 3))
    assert term.shape == tokens.shape
    assert not torch.equal(term[:, 0], term[:, 1])
    assert float(term.detach().abs().sum()) > 0.0


def test_an_out_of_range_or_misaligned_phase_input_is_rejected():
    """Declared is not the same as plausible: ranges and shapes are checked."""
    model = _model()
    for field, value in (("init_utc_hour", torch.tensor(24.0)),
                         ("init_utc_hour", torch.tensor(-1.0)),
                         ("init_day_of_year", torch.tensor(0.0)),
                         ("init_day_of_year", torch.tensor(367.0)),
                         ("init_utc_hour", torch.tensor([1.0, 2.0, 3.0])),
                         ("init_utc_hour", torch.tensor(float("nan"))),
                         ("latitude", torch.tensor([30.0]))):
        with pytest.raises(ValueError, match="init_utc_hour|init_day_of_year|latitude"):
            _forecast(model, dict(_sample(), **{field: value}))
