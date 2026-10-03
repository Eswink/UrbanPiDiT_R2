"""D2: every path hands the model the same declared field set.

Seven paths reach the forecaster with a batch dict: fixed training, truncated
(stationary truncated BPTT), streaming/validation rollouts, published evaluation,
adaptive inference, the free rollout, and the inference profiler. #71's whole risk
is drift between them - one path quietly dropping the initialization-time fields
while another carries them - so the pin here is an equality between the sets the
paths actually pass, not a hand-written list of expected keys (a parallel list
would drift exactly like the paths would).

`test_the_declared_inputs_are_exactly_what_the_model_reads` closes the other half:
the whitelist must not advertise a field the model never reads (dead weight that a
future reader would take for conditioning), and the model must not read a field the
whitelist strips (the historical failure mode, where ``latitude`` was available in
every sample and reached no model).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import default_collate

from data.r7_evaluation import ZarrRolloutDataset
from model.r7_halting import (DECLARED_MODEL_INPUTS, AdaptiveProcessForecaster,
    forecast_inputs)
from model.r7_rollout import autoregressive_rollout, rollout_model_input
from model.spacetime_conditioning_r7 import CALENDAR_INPUT_FIELDS, SPACETIME_INPUT_FIELDS
from model.known_context_r7 import HISTORY_CONTEXT_FIELDS
from training.r7_evaluate import evaluate_local
from training.r7_experiment import dataset_identity, make_model, seed_everything
from training.r7_inference_profile import profile_forward
from training.r7_local_runner import run_local_updates, update_group
from training.r7_scheduled_runner import score_validation
from training.r7_streaming import backward_streamed_truncated
from test_r7_storage_safety import build

EXPECTED = frozenset({"coarse_history", "lead_time_hours", *SPACETIME_INPUT_FIELDS})
WITH_CALENDAR = EXPECTED | frozenset(CALENDAR_INPUT_FIELDS) | frozenset(HISTORY_CONTEXT_FIELDS)
LEAKAGE = frozenset({"atmos_target", "atmos_baseline", "rollout_targets", "process_targets",
                     "sample_id", "init_time", "valid_times", "init_year", "grid_spacing_deg"})
PROCESS_CONFIG = {"in_channels": 1, "out_channels": 1, "history_steps": 2, "dim": 16,
                  "depth": 1, "heads": 2, "window_size": 2, "patch_size": 2, "dropout": 0.0,
                  "anchored_processes": 2, "free_processes": 1, "default_reasoning_steps": 1,
                  "spacetime_inputs": True, "positional_process_readout": True}
NATIVE_CONFIG = {name: value for name, value in PROCESS_CONFIG.items()
                 if name not in ("anchored_processes", "free_processes",
                                 "default_reasoning_steps", "positional_process_readout")}


def capturing(*modules) -> set[frozenset]:
    """Collect the key set of every dict handed to any of ``modules``."""
    seen: set[frozenset] = set()

    def hook(_module, inputs):
        if inputs and isinstance(inputs[0], dict):
            seen.add(frozenset(inputs[0]))

    for module in modules:
        module.register_forward_pre_hook(hook)
    return seen


def watch(model):
    """Watch a model and its backbone; both receive the batch on these paths."""
    modules = [model]
    backbone = getattr(getattr(model, "forecaster", model), "backbone", None)
    if backbone is not None:
        modules.append(backbone)
    return capturing(*modules)


def test_the_whitelist_is_the_declared_set_and_carries_no_target():
    full = {"coarse_history": torch.zeros(1), "lead_time_hours": torch.zeros(1),
            **{name: torch.zeros(1) for name in SPACETIME_INPUT_FIELDS},
            **{name: torch.zeros(1) for name in LEAKAGE}}
    whitelisted = forecast_inputs(full)
    assert set(whitelisted) == EXPECTED
    assert set(whitelisted) | set(CALENDAR_INPUT_FIELDS) | set(HISTORY_CONTEXT_FIELDS) == set(DECLARED_MODEL_INPUTS)
    with_calendar = forecast_inputs(dict(full, init_calendar_year=torch.tensor(2016.0),
                                        history_offsets_hours=torch.tensor([[-6., 0.]])))
    assert set(with_calendar) == set(DECLARED_MODEL_INPUTS) == WITH_CALENDAR
    assert not set(with_calendar) & LEAKAGE
    assert not set(whitelisted) & LEAKAGE
    with pytest.raises(KeyError):
        forecast_inputs({"lead_time_hours": torch.zeros(1)})  # history is not optional


def test_every_path_hands_the_model_the_same_field_set(tmp_path):
    """Fixed, truncated, validation, adaptive, rollout and profile - one field set."""
    data = build(tmp_path / "data")
    identity, dataset = dataset_identity(data["train"])
    validation = ZarrRolloutDataset(tmp_path / "data" / "output", split="val", lead_hours=(6,),
                                    history_steps=2, step_hours=6)
    sample = validation[0]
    collated = default_collate([dataset[0], dataset[1]])
    observed = {}

    seed_everything(3)
    native = make_model("native", dict(NATIVE_CONFIG)).train()
    seen = watch(native)
    update_group(native, torch.optim.AdamW(native.parameters()), [collated], kind="native",
                 device=torch.device("cpu"), steps=0, bf16=False, process_weight=0.0, clip=1.0)
    observed["fixed"] = set(seen)

    process = make_model("process", dict(PROCESS_CONFIG)).train()
    seen = watch(process)
    backward_streamed_truncated(process, collated, reasoning_steps=1, process_weight=0.0)
    observed["truncated"] = set(seen)

    seed_everything(4)
    native = make_model("native", dict(NATIVE_CONFIG)).eval()
    seen = watch(native)
    score_validation(native, validation, kind="native", device=torch.device("cpu"),
                     steps=1, lead_hours=(6,))
    observed["validation"] = set(seen)

    process = make_model("process", dict(PROCESS_CONFIG)).eval()
    adapter = AdaptiveProcessForecaster(process).eval()
    seen = watch(adapter)
    adapter.initial_state(rollout_model_input(sample))
    observed["adaptive"] = set(seen)

    seen = watch(adapter)
    autoregressive_rollout(adapter, rollout_model_input(sample), lead_hours=(6, 12),
                           step_hours=6, history_interval_hours=6,
                           inference_kwargs={"max_steps": 1, "min_steps": 1,
                                             "force_full_depth": True})
    observed["rollout"] = set(seen)

    seed_everything(6)
    native = make_model("native", dict(NATIVE_CONFIG)).eval()
    seen = watch(native)
    profile_forward(native, rollout_model_input(sample), warmup=1, repetitions=1,
                    precision="fp32")
    observed["profile"] = set(seen)

    assert identity  # the run above is bound to the store's identity
    for name, sets in observed.items():
        assert sets == {WITH_CALENDAR}, f"{name} handed the model {sets}"


def test_the_evaluation_path_carries_the_declared_fields(tmp_path, monkeypatch):
    """The published evaluation is checked on its own, so a failure names the path."""
    data = build(tmp_path / "data")
    identity, dataset = dataset_identity(data["train"])
    checkpoint, _ = run_local_updates(
        dataset, kind="native", model_config=dict(NATIVE_CONFIG), data_identity=identity,
        output_dir=tmp_path / "training", total_updates=1, batch_size=1, steps=0, seed=5,
        process_weight=0.0, device_name="cpu")
    import training.r7_evaluate as evaluate_module
    original = evaluate_module.make_model
    observed: set[frozenset] = set()

    def watched(kind, config):
        model = original(kind, config)
        modules = [model] + ([model.backbone] if hasattr(model, "backbone") else [])
        for module in modules:
            module.register_forward_pre_hook(
                lambda _module, inputs: observed.add(frozenset(inputs[0]))
                if inputs and isinstance(inputs[0], dict) else None)
        return model

    monkeypatch.setattr(evaluate_module, "make_model", watched)
    evaluate_local(data["val"], output_dir=tmp_path / "eval", checkpoint=checkpoint,
                   lead_hours=(6,), max_samples=1, device_name="cpu")
    assert observed == {WITH_CALENDAR}, f"evaluation handed the model {observed}"


class Recording(dict):
    """A batch dict that records which of its keys are looked up, and misses."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read: set[str] = set()
        self.missing: set[str] = set()

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        (self.read if key in self else self.missing).add(key)
        return super().get(key, default)


def _batch(batch_size: int = 2, hw: tuple[int, int] = (4, 6)):
    generator = torch.Generator().manual_seed(17)
    return {
        "coarse_history": torch.randn(batch_size, 2, 1, *hw, generator=generator),
        "atmos_target": torch.randn(batch_size, 1, *hw, generator=generator),
        "atmos_baseline": torch.randn(batch_size, 1, *hw, generator=generator),
        "lead_time_hours": torch.full((batch_size,), 6.0),
        "latitude": torch.linspace(60.0, 25.0, hw[0]),
        "longitude": torch.linspace(100.0, 112.0, hw[1]),
        "init_utc_hour": torch.tensor(3.0),
        "init_day_of_year": torch.tensor(15.0),
        "init_year": torch.tensor(2016.0),
        "grid_spacing_deg": torch.tensor(0.25),
    }


def _read_set(model, batch):
    recorded = Recording(forecast_inputs(batch))
    with torch.no_grad():
        model(recorded, reasoning_steps=1)
    return recorded


def test_the_declared_inputs_are_exactly_what_the_model_reads():
    """A declared field nobody reads, or a read the whitelist strips, both fail."""
    seed_everything(8)
    switched_on = make_model("process", dict(PROCESS_CONFIG)).eval()
    recorded = _read_set(switched_on, _batch())
    assert recorded.read == EXPECTED
    # ``atmos_baseline`` is an optional override that reaches the backbone only when
    # a caller passes it directly; the whitelist deliberately never forwards it, and
    # this is the pin that keeps that the *only* stripped lookup.
    assert recorded.missing == {"atmos_baseline"}

    seed_everything(8)
    switched_off = make_model("process", {**PROCESS_CONFIG, "spacetime_inputs": False,
                                          "positional_process_readout": False}).eval()
    recorded = _read_set(switched_off, _batch())
    assert recorded.read == {"coarse_history", "lead_time_hours"}
    assert recorded.read < set(DECLARED_MODEL_INPUTS)


def test_the_rollout_advances_the_lead_only_for_the_phase_model():
    """The lead convention is part of the model's declared inputs, not a guess."""
    sample = {"coarse_history": torch.randn(2, 1, 4, 6),
              "lead_time_hours": torch.tensor(6.0), "latitude": torch.linspace(60, 25, 4),
              "longitude": torch.linspace(100, 112, 6), "init_utc_hour": torch.tensor(0.0),
              "init_day_of_year": torch.tensor(1.0)}
    leads = {}

    def leads_seen(model, name):
        seen = []

        def hook(_module, inputs):
            seen.append(float(inputs[0]["lead_time_hours"][0]))

        getattr(getattr(model, "forecaster", model), "backbone").register_forward_pre_hook(hook)
        leads[name] = seen

    seed_everything(9)
    off = make_model("process", {**PROCESS_CONFIG, "spacetime_inputs": False,
                                 "positional_process_readout": False}).eval()
    leads_seen(off, "off")
    autoregressive_rollout(off, rollout_model_input(sample), lead_hours=(6, 12, 18),
                           step_hours=6, history_interval_hours=6,
                           inference_kwargs={"reasoning_steps": 1})
    seed_everything(9)
    on = make_model("process", dict(PROCESS_CONFIG)).eval()
    leads_seen(on, "on")
    autoregressive_rollout(on, rollout_model_input(sample), lead_hours=(6, 12, 18),
                           step_hours=6, history_interval_hours=6,
                           inference_kwargs={"reasoning_steps": 1})
    assert leads["off"] == [6.0, 6.0, 6.0]
    assert leads["on"] == [6.0, 12.0, 18.0]


def test_the_rollout_never_invents_a_missing_declared_field():
    """Counterproof: without the fields the phase model fails instead of guessing."""
    sample = {"coarse_history": torch.randn(2, 1, 4, 6)}
    seed_everything(9)
    model = make_model("process", dict(PROCESS_CONFIG)).eval()
    with pytest.raises((KeyError, ValueError), match="latitude|longitude|init_utc_hour|"
                                                    "init_day_of_year"):
        autoregressive_rollout(model, rollout_model_input(sample), lead_hours=(6,),
                              step_hours=6, history_interval_hours=6,
                              inference_kwargs={"reasoning_steps": 1})
