"""CPU-only synthetic counterproofs; fixture identities/units are NOT real weather.

The external train fit and source qualification are intentionally absent. These
unit tests establish only structural validation, byte identity and time lookup.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import model.climatology_anchor_r7 as anchor_module
from model.climatology_anchor_r7 import TrainClimatologyAnchor, uses_climatology_anchor


def tiny_spec_table():
    """Declared fake metadata and normalized toy bytes, never a weather fixture."""
    keys = [[month, hour] for month in (1, 2, 3, 12) for hour in (0, 6, 12, 18)]
    table = torch.arange(16 * 2 * 2 * 3, dtype=torch.float32).reshape(16, 2, 2, 3) / 100
    spec = {
        "format": "r7-train-climatology-anchor-v1", "kind": "train-only-month-hour-grid-mean-v1",
        "channels": ["toy_a", "toy_b"], "units": ["toy_unit_a", "toy_unit_b"],
        "source_sha256": "1" * 64, "train_data_identity": "2" * 64,
        "climatology_mean_identity_sha256": "3" * 64,
        "normalization_mean": [0.0, 1.0], "normalization_std": [1.0, 2.0],
        "latitude": [60.125, 30.375], "longitude": [100.125, 110.375, 120.625],
        "bucket_keys": keys, "bucket_counts": [2] * 16, "training_years": [2016, 2017],
        "n_selected_steps": 32, "selection": "declared_train_years",
        "table_sha256": hashlib.sha256(table.numpy().tobytes()).hexdigest(),
        "table_shape": [16, 2, 2, 3],
    }
    return spec, table


def _installed():
    spec, table = tiny_spec_table()
    anchor = TrainClimatologyAnchor(spec)
    anchor.install(table)
    return anchor, spec, table


def _inputs(batch_size=1, *, dtype=torch.float32):
    history = torch.zeros(batch_size, 2, 2, 2, 3, dtype=dtype)
    batch = {
        "latitude": torch.tensor([60.125, 30.375], dtype=dtype),
        "longitude": torch.tensor([100.125, 110.375, 120.625], dtype=dtype),
        "init_calendar_year": torch.tensor(2017), "init_day_of_year": torch.tensor(1),
        "init_utc_hour": torch.tensor(0.0), "lead_time_hours": torch.tensor(6.0),
    }
    return batch, history


def test_constructor_is_unready_fp32_persistent_and_consumes_no_rng():
    spec, _ = tiny_spec_table()
    before = torch.random.get_rng_state().clone()
    anchor = TrainClimatologyAnchor(spec)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert not bool(anchor.ready)
    assert anchor.table.dtype == torch.float32 and torch.count_nonzero(anchor.table) == 0
    assert anchor.ready.dtype == torch.bool and anchor.ready.shape == ()
    assert dict(anchor.named_parameters()) == {}
    assert set(dict(anchor.named_buffers())) == {"table", "ready"}
    assert set(anchor.state_dict()) == {"table", "ready"}
    batch, history = _inputs()
    with pytest.raises(RuntimeError, match="unready"):
        anchor(batch, history)


def test_canonical_spec_is_immutable_deep_snapshot_not_live_caller_metadata():
    spec, table = tiny_spec_table()
    anchor = TrainClimatologyAnchor(spec)
    expected_json = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False)
    spec["latitude"][0] = -80.0
    spec["bucket_keys"][0][0] = 11
    spec["normalization_mean"][0] = 987.0
    spec["table_sha256"] = "0" * 64
    exposed = anchor.spec
    exposed["units"][0] = "mutated"
    exposed["channels"].append("mutated")
    assert anchor.spec_json == expected_json
    assert anchor.spec == json.loads(expected_json)
    with pytest.raises(AttributeError):
        anchor.spec = exposed
    with pytest.raises(AttributeError):
        anchor.spec_json = "mutated"
    anchor.install(table)
    batch, history = _inputs()
    assert torch.equal(anchor(batch, history), table[1:2])


@pytest.mark.parametrize("field,value", [
    ("format", "other-v1"), ("kind", "held-out-month-hour-grid-mean-v1"),
    ("channels", []), ("channels", ["toy_a", "toy_a"]), ("channels", ["toy_a", 2]),
    ("units", ["toy_unit_a"]), ("units", ["toy_unit_a", ""]),
    ("units", ["toy_unit_a", 123]), ("units", ("a", "b")),
    ("normalization_mean", [0, 1]), ("normalization_mean", [0.0, float("nan")]),
    ("normalization_mean", [0.0]), ("normalization_std", [1.0, 0.0]),
    ("normalization_std", [1.0, -1.0]), ("normalization_std", [1.0, float("inf")]),
    ("latitude", [float("nan"), 30.375]), ("longitude", [True, 110.375, 120.625]),
    ("bucket_keys", [[0, 0]]), ("bucket_keys", [[13, 0]]),
    ("bucket_keys", [[1, 3]]), ("bucket_keys", [[True, 0]]),
    ("bucket_keys", [[1, 0], [1, 0]]), ("bucket_keys", [[2, 0], [1, 0]]),
    ("bucket_keys", [[1, 0, 6]]), ("bucket_counts", [1]),
    ("bucket_counts", [0] * 16), ("bucket_counts", [True] * 16),
    ("training_years", [2017, 2016]), ("training_years", [2016, 2016]),
    ("training_years", [2016.0]), ("training_years", [0]),
    ("training_years", [10000]), ("n_selected_steps", 31), ("n_selected_steps", 32.0),
    ("selection", "held_out_years"), ("table_shape", [16, 1, 2, 3]),
    ("table_shape", [16.0, 2, 2, 3]),
])
def test_rejects_invalid_declared_spec(field, value):
    spec, _ = tiny_spec_table()
    spec[field] = value
    with pytest.raises((ValueError, TypeError), match=field):
        TrainClimatologyAnchor(spec)


@pytest.mark.parametrize("field", [
    "source_sha256", "train_data_identity", "climatology_mean_identity_sha256", "table_sha256",
])
@pytest.mark.parametrize("bad_identity", ["f" * 63, "g" * 64, "f" * 65, 123])
def test_source_and_table_identities_must_be_exact_64_hex(field, bad_identity):
    spec, _ = tiny_spec_table()
    spec[field] = bad_identity
    with pytest.raises(ValueError, match=field):
        TrainClimatologyAnchor(spec)


def test_exact_json_key_set_never_accepts_table_values_or_range_details():
    spec, _ = tiny_spec_table()
    for extra in ("table", "train_time_ranges", "unknown"):
        with pytest.raises(ValueError, match="exact declared key set"):
            TrainClimatologyAnchor({**spec, extra: []})
    del spec["units"]
    with pytest.raises(ValueError, match="exact declared key set"):
        TrainClimatologyAnchor(spec)
    with pytest.raises(TypeError, match="JSON dict"):
        TrainClimatologyAnchor([])


def test_structural_identity_and_units_validation_is_not_external_source_truth():
    spec, table = tiny_spec_table()
    spec["selection"] = "declared_train_time_ranges"
    spec["source_sha256"] = "A" * 64
    spec["train_data_identity"] = "B" * 64
    spec["units"] = ["explicit_fake_a", "explicit_fake_b"]
    spec["table_sha256"] = spec["table_sha256"].upper()
    anchor = TrainClimatologyAnchor(spec)
    anchor.install(table)
    assert anchor.spec["source_sha256"] == "A" * 64
    assert bool(anchor.ready)  # No claim that the declared fake source or units is real.


def test_install_copies_once_without_alias_gradient_or_rng_and_returns_history_dtype():
    anchor, _, table = _installed()
    expected = table.clone()
    table.add_(900)
    assert torch.equal(anchor.table, expected)
    assert anchor.table.dtype == torch.float32 and not anchor.table.requires_grad
    with pytest.raises(RuntimeError, match="one-shot"):
        anchor.install(expected)
    anchor.ready.fill_(False)
    with pytest.raises(RuntimeError, match="one-shot"):
        anchor.install(expected)
    anchor.ready.fill_(True)
    batch, history = _inputs(dtype=torch.float64)
    history.requires_grad_(True)
    before = torch.random.get_rng_state().clone()
    result = anchor(batch, history)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert result.dtype == history.dtype and result.shape == (1, 2, 2, 3)
    assert result.device == history.device and not result.requires_grad and result.grad_fn is None
    result.zero_()
    assert torch.equal(anchor.table, expected)


@pytest.mark.parametrize("mutation", ["shape", "float64", "integer", "nan", "grad", "hash"])
def test_install_rejects_bad_table_before_any_acceptance(mutation):
    spec, table = tiny_spec_table()
    anchor = TrainClimatologyAnchor(spec)
    bad = table.clone()
    if mutation == "shape":
        bad = bad[:1]
    elif mutation == "float64":
        bad = bad.double()
    elif mutation == "integer":
        bad = bad.long()
    elif mutation == "nan":
        bad[0, 0, 0, 0] = float("nan")
    elif mutation == "grad":
        bad.requires_grad_(True)
    else:
        bad[0, 0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="table"):
        anchor.install(bad)
    assert not bool(anchor.ready) and torch.count_nonzero(anchor.table) == 0
    anchor.install(table)
    assert bool(anchor.ready)


def test_install_non_tensor_and_storage_cast_are_explicit_failures():
    spec, table = tiny_spec_table()
    anchor = TrainClimatologyAnchor(spec)
    with pytest.raises(TypeError, match="tensor"):
        anchor.install([])
    cast = TrainClimatologyAnchor(spec).to(dtype=torch.float64)
    with pytest.raises(ValueError, match="FP32"):
        cast.install(table)
    anchor.install(table)
    anchor.to(dtype=torch.float16)
    batch, history = _inputs(dtype=torch.float16)
    with pytest.raises(ValueError, match="FP32"):
        anchor(batch, history)


def test_ordinary_nested_strict_load_validates_then_restores_no_fit_or_rng():
    anchor, spec, table = _installed()
    parent = nn.Module()
    parent.anchor = anchor
    state = {key: value.clone() for key, value in parent.state_dict().items()}
    fresh = nn.Module()
    fresh.anchor = TrainClimatologyAnchor(json.loads(json.dumps(spec)))
    before = torch.random.get_rng_state().clone()
    fresh.load_state_dict(state, strict=True)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert fresh.anchor.ready and torch.equal(fresh.anchor.table, table)
    batch, history = _inputs()
    assert torch.equal(fresh.anchor(batch, history), anchor(batch, history))
    with pytest.raises(RuntimeError, match="one-shot"):
        fresh.anchor.install(table)


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("mutation", [
    "unready", "ready_shape", "ready_dtype", "missing_ready", "missing_table",
    "hash", "shape", "dtype", "nan", "grad",
])
def test_load_rejects_unready_or_tampered_constant_before_acceptance(strict, mutation):
    installed, spec, _ = _installed()
    fresh = TrainClimatologyAnchor(spec)
    state = {key: value.clone() for key, value in installed.state_dict().items()}
    if mutation == "unready":
        state["ready"].fill_(False)
    elif mutation == "ready_shape":
        state["ready"] = torch.tensor([True])
    elif mutation == "ready_dtype":
        state["ready"] = torch.tensor(1)
    elif mutation.startswith("missing_"):
        del state[mutation.removeprefix("missing_")]
    elif mutation == "hash":
        state["table"][0, 0, 0, 0] += 1
    elif mutation == "shape":
        state["table"] = state["table"][:1]
    elif mutation == "dtype":
        state["table"] = state["table"].double()
    elif mutation == "nan":
        state["table"][0, 0, 0, 0] = float("nan")
    else:
        state["table"].requires_grad_(True)
    with pytest.raises(RuntimeError, match="climatology anchor rejected state"):
        fresh.load_state_dict(state, strict=strict)
    assert not bool(fresh.ready) and torch.count_nonzero(fresh.table) == 0


def test_unready_constructor_state_and_coherent_wrong_digest_never_load():
    spec, _ = tiny_spec_table()
    fresh = TrainClimatologyAnchor(spec)
    with pytest.raises(RuntimeError, match="ready"):
        TrainClimatologyAnchor(spec).load_state_dict(fresh.state_dict())
    installed, _, _ = _installed()
    spec["table_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="SHA256"):
        TrainClimatologyAnchor(spec).load_state_dict(installed.state_dict())
    cast = TrainClimatologyAnchor(installed.spec).double()
    with pytest.raises(RuntimeError, match="FP32"):
        cast.load_state_dict(installed.state_dict())


@pytest.mark.parametrize("year,day,hour,lead,month,valid_hour", [
    (2017, 31, 18, 6, 2, 0), (2016, 59, 18, 6, 2, 0),
    (2016, 60, 18, 6, 3, 0), (2017, 59, 18, 6, 3, 0),
    (2000, 59, 18, 6, 2, 0), (1900, 59, 18, 6, 3, 0),
    (2016, 366, 18, 6, 1, 0), (2017, 365, 18, 6, 1, 0),
    (2017, 1, 0, 12, 1, 12), (2017, 1, 18, 30, 1, 0),
])
def test_queries_exact_gregorian_valid_month_hour_not_initialization_bucket(
        year, day, hour, lead, month, valid_hour):
    anchor, spec, table = _installed()
    batch, history = _inputs()
    batch.update(init_calendar_year=torch.tensor(year), init_day_of_year=torch.tensor(day),
                 init_utc_hour=torch.tensor(hour), lead_time_hours=torch.tensor(lead))
    expected = spec["bucket_keys"].index([month, valid_hour])
    assert torch.equal(anchor(batch, history), table[expected:expected + 1])


def test_batched_samples_select_different_buckets_with_collated_exact_grid():
    anchor, _, table = _installed()
    batch, history = _inputs(3)
    batch.update(init_calendar_year=torch.tensor([2017, 2016, 2016]),
                 init_day_of_year=torch.tensor([31, 60, 366]),
                 init_utc_hour=torch.tensor([18, 6, 18]),
                 lead_time_hours=torch.tensor([6, 6, 6]))
    for name in ("latitude", "longitude"):
        batch[name] = batch[name].expand(3, -1).clone()
    assert torch.equal(anchor(batch, history), table[[4, 6, 0]])  # Leap Feb 29 remains February at 12:00.
    batch["latitude"][1, 0] += 0.01
    with pytest.raises(ValueError, match="differs"):
        anchor(batch, history)


def test_missing_month_and_missing_hour_buckets_have_no_heldout_fill():
    anchor, spec, table = _installed()
    batch, history = _inputs()
    batch["init_day_of_year"] = torch.tensor(91)  # April 1 (non-leap).
    with pytest.raises(ValueError, match="no held-out fill"):
        anchor(batch, history)
    for name in ("bucket_keys", "bucket_counts"):
        del spec[name][1]  # Delete Jan 06:00, not replace it with a default.
    spec["n_selected_steps"] -= 2
    table = torch.cat([table[:1], table[2:]], dim=0)
    spec["table_shape"][0] -= 1
    spec["table_sha256"] = hashlib.sha256(table.numpy().tobytes()).hexdigest()
    anchor = TrainClimatologyAnchor(spec)
    anchor.install(table)
    batch, history = _inputs()
    with pytest.raises(ValueError, match="missing unique"):
        anchor(batch, history)


@pytest.mark.parametrize("field,value", [
    ("init_calendar_year", 0), ("init_calendar_year", 10000), ("init_calendar_year", 2016.5),
    ("init_calendar_year", float("nan")), ("init_day_of_year", 0), ("init_day_of_year", 366),
    ("init_day_of_year", 1.5), ("init_utc_hour", 24), ("init_utc_hour", -6),
    ("init_utc_hour", 3), ("init_utc_hour", 6.5), ("lead_time_hours", 0),
    ("lead_time_hours", -6), ("lead_time_hours", 6.5), ("lead_time_hours", 7),
    ("lead_time_hours", float("nan")), ("lead_time_hours", float("inf")),
    ("init_day_of_year", float("nan")), ("init_utc_hour", float("nan")),
])
def test_rejects_invalid_year_day_hour_and_actual_lead(field, value):
    anchor, _, _ = _installed()
    batch, history = _inputs()
    batch[field] = torch.tensor(value)
    with pytest.raises(ValueError, match=field):
        anchor(batch, history)


@pytest.mark.parametrize("field", [
    "latitude", "longitude", "init_calendar_year", "init_day_of_year", "init_utc_hour",
    "lead_time_hours",
])
def test_missing_required_field_has_no_clock_legacy_year_or_default_lead_fallback(field):
    anchor, _, _ = _installed()
    batch, history = _inputs()
    del batch[field]
    batch["init_year"] = torch.tensor(2017)
    with pytest.raises(KeyError, match=field):
        anchor(batch, history, default_lead_hours=6.0)


@pytest.mark.parametrize("field", [
    "init_calendar_year", "init_day_of_year", "init_utc_hour", "lead_time_hours",
])
@pytest.mark.parametrize("shape", [(2, 1), (1, 2), (3,)])
def test_calendar_and_lead_accept_only_scalar_or_exact_batch_vector(field, shape):
    anchor, _, _ = _installed()
    batch, history = _inputs(2)
    batch[field] = torch.ones(shape)
    with pytest.raises(ValueError, match="scalar or"):
        anchor(batch, history)


@pytest.mark.parametrize("field", ["latitude", "longitude"])
def test_coordinate_values_shape_finiteness_and_input_dtype_projection_are_exact(field):
    anchor, _, _ = _installed()
    batch, history = _inputs(dtype=torch.float64)
    assert anchor(batch, history).dtype == torch.float64
    changed = batch[field].clone()
    changed[0] += 1e-10  # No tolerance even though a float32 cast would erase it.
    with pytest.raises(ValueError, match="exactly match"):
        anchor(dict(batch, **{field: changed}), history)
    changed[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        anchor(dict(batch, **{field: changed}), history)
    with pytest.raises(ValueError, match="grid"):
        anchor(dict(batch, **{field: batch[field].expand(2, -1)}), history)


def test_rejects_channel_grid_history_dtype_device_and_numeric_field_dtype():
    anchor, _, _ = _installed()
    batch, history = _inputs()
    for bad in (history[:, :, :1], history[..., :1, :], history[:, 0], history[:, :0], history.long()):
        with pytest.raises(ValueError, match="history"):
            anchor(batch, bad)
    bad = history.clone()
    bad[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        anchor(batch, bad)
    with pytest.raises(ValueError, match="device"):
        anchor(batch, torch.empty(history.shape, device="meta"))
    for bad in (torch.tensor(True), torch.tensor(6 + 0j), 6.0):
        with pytest.raises(TypeError, match="real numeric tensor"):
            anchor(dict(batch, lead_time_hours=bad), history)
    with pytest.raises(ValueError, match="history device"):
        anchor(dict(batch, latitude=torch.empty(2, device="meta")), history)
    batch.update(init_calendar_year=torch.tensor(9999), init_day_of_year=torch.tensor(365),
                 init_utc_hour=torch.tensor(18))
    with pytest.raises(ValueError, match="calendar valid time"):
        anchor(batch, history)


def test_forward_consumes_no_targets_legacy_metadata_or_baseline_values(monkeypatch):
    anchor, _, table = _installed()
    batch, history = _inputs()

    class KnownOnly(dict):
        def __getitem__(self, key):
            assert key in batch, f"undeclared future/target access: {key}"
            return super().__getitem__(key)

    unread = dict(atmos_target=object(), future_calendar_year=object(), init_year=object(),
                  process_targets=object(), climatology_mean=object())
    supplied = KnownOnly({**batch, **unread})
    monkeypatch.setattr(anchor_module, "_table_digest", lambda _: pytest.fail("per-forward hashing"))
    assert torch.equal(anchor(supplied, history), table[1:2])
    with pytest.raises(ValueError, match="atmos_baseline"):
        anchor(KnownOnly({**supplied, "atmos_baseline": object()}), history)


def test_logical_noncontiguous_fp32_bytes_install_but_manual_grad_rejects():
    spec, table = tiny_spec_table()
    view = table.transpose(-1, -2).contiguous().transpose(-1, -2)
    assert not view.is_contiguous() and torch.equal(view, table)
    anchor = TrainClimatologyAnchor(spec)
    anchor.install(view)
    assert torch.equal(anchor.table, table)
    bad = table.clone()
    bad.grad = torch.ones_like(bad)
    with pytest.raises(ValueError, match="no grad"):
        TrainClimatologyAnchor(spec).install(bad)


def test_default_off_native_omitted_and_none_preserve_weights_rng_and_outputs():
    from model.weather_forecaster_r7 import NativeAtmosForecaster

    options = dict(in_channels=2, history_steps=2, out_channels=2, dim=8, patch_size=1,
                   depth=1, heads=2, window_size=2, dropout=0.0, spacetime_inputs=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(83)
        omitted = NativeAtmosForecaster(**options).eval()
        stream = torch.random.get_rng_state().clone()
        torch.manual_seed(83)
        explicit = NativeAtmosForecaster(**options, climatology_anchor_spec=None).eval()
        assert torch.equal(torch.random.get_rng_state(), stream)
    assert not uses_climatology_anchor(omitted) and not uses_climatology_anchor(explicit)
    assert set(omitted.state_dict()) == set(explicit.state_dict())
    assert not any("climatology_anchor" in key for key in explicit.state_dict())
    assert all(torch.equal(value, explicit.state_dict()[key]) for key, value in omitted.state_dict().items())
    _, history = _inputs()
    with torch.no_grad():
        left = omitted({"coarse_history": history})
        right = explicit({"coarse_history": history})
    assert torch.equal(left.forecast, right.forecast) and torch.equal(left.tendency, right.tendency)


def test_capability_query_unwraps_forecaster_then_backbone_without_requiring_pe():
    anchor, _, _ = _installed()
    core = SimpleNamespace(climatology_anchor=anchor, spacetime_inputs=False)
    assert uses_climatology_anchor(core)
    assert uses_climatology_anchor(SimpleNamespace(backbone=core))
    assert uses_climatology_anchor(SimpleNamespace(forecaster=SimpleNamespace(backbone=core)))
    assert not uses_climatology_anchor(SimpleNamespace(climatology_anchor=None))
    assert not uses_climatology_anchor(SimpleNamespace(forecaster=SimpleNamespace()))


def test_module_import_has_no_clock_io_fitting_randomness_or_unsafe_runtime_dependency():
    path = Path(anchor_module.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert imports <= {"__future__", "hashlib", "json", "math", "typing", "torch", "spacetime_conditioning_r7"}
    calls = {node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not calls & {"open", "eval", "exec"}
    assert not attributes & {"now", "today", "utcnow", "time", "monotonic", "fit", "save", "load", "rand", "randn"}
