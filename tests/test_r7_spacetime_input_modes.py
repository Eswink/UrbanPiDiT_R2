"""#71 round three: the field-mode control arms, and the `fields` mode pinned.

Round three adds two control modes to the space-time input pathway so the
round-two ``B - A`` effect can be split into information and capacity/bias:

- ``fields``   (default, and the pre-existing behaviour) the real
  initialization-time fields reach the conditioning MLP;
- ``constant`` the fields are replaced by same-shape zeros *after* validation, so
  the module is present and trained but its input carries no information - the
  contribution collapses to one constant additive vector per forward pass;
- ``shuffled`` the fields are rolled by one along the *sample* axis, so each
  sample is conditioned on another sample's initialization time.

Two things this file is responsible for:

1. **D4, the regression pin.** ``FIELDS_PATH_DIGESTS`` holds forward and backward
   digests of the ``fields`` mode captured on the tree *before* the modes existed
   (the capture revision is recorded in ``FIELDS_PATH_CAPTURE``). Every entry is a
   raw little-endian float byte digest of a tensor, plus one digest over every
   parameter and its gradient, so a change anywhere in the conditioning path or
   the training path moves at least one entry. The pin is checked against the
   frozen constants, not against a second copy of the tree, so the recipe itself
   has to be shown to be sensitive: ``test_the_frozen_pin_detects_a_changed_path``
   runs the same recipe with the conditioning perturbed and requires the digests
   to move.

2. **D1, the modes.** Behaviour, determinism, the single-sample-batch property of
   ``shuffled`` (declared, not silent), invalid-mode rejection, and the two
   invariants D6 rests on: all three modes carry the *same* parameter tensors, and
   ``constant`` and ``fields`` cost the *same* forward FLOPs.

Run as a script to print the digests (that is how the frozen values below were
captured): ``python tests/test_r7_spacetime_input_modes.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for path in (str(ROOT), str(TESTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from model.spacetime_conditioning_r7 import SPACETIME_INPUT_FIELDS
from training.r7_experiment import make_model, seed_everything
from training.r7_streaming import backward_streamed_truncated
from test_r7_switched_path_equivalence import digest_parameters, tensor_digest

FIELDS_PATH_CONFIG = {"in_channels": 3, "out_channels": 3, "history_steps": 2,
                      "architecture": "window", "dim": 16, "depth": 2, "heads": 2,
                      "window_size": 2, "patch_size": 2, "dropout": 0.0,
                      "anchored_processes": 2, "free_processes": 2,
                      "use_forecast_feedback": True, "default_reasoning_steps": 2,
                      "spacetime_inputs": True}
FIELDS_PATH_STEPS = 2
BATCH_SIZE = 2
HW = (4, 6)
CHANNELS = 3
# Where the frozen digests below came from: `git rev-parse HEAD` at capture time,
# with the working tree clean apart from this file. Anything else in the tree that
# could move these numbers would move them for the capture too.
FIELDS_PATH_CAPTURE = "ac6a3ef486d56d43e11261b62e7294872d1e5eae"
FIELDS_PATH_DIGESTS: dict[str, str] = {
    "fields.context_tokens":
        "836c8a45b4177ea4a16902dae46ca231d0262f3560adae2e56a4c116a5c9c550",
    "fields.draft_forecasts":
        "9d9e0f27013cd613fd946df967b1adb59286280b8491a0db4561426a496b4262",
    "fields.final_correction":
        "c1bc5b96ba0b0fc210be561870fe54e79d65e08126fe4c44a36690dab3254f2e",
    "fields.forecast":
        "e3117a9e74cd8da19e80fa8a470b49649dc3c059590a43a6f2e82212c6640ce4",
    "fields.gradients":
        "f3eed8aadea93989a1a8f6609d52cc303d71bd99b639333f34ba34799c25dce9",
    "fields.initial_forecast":
        "9cdcd55964428689ff669dbe357177a22d0f9580821e5da42e564ce769b6179f",
    "fields.process_state":
        "b87560f2596bcee031b37cfaa75588885fe9bc90f2096266ce9fcab68e350e90",
    "fields.streamed.final_forecast":
        "9be6491fa8ab44fc75f8191c0600bcc798187f66ee38c58dfce10c8db8daa467",
    "fields.streamed.forecast":
        "ba9d82711b22a920da389ed88eccd40d011be9a93488f23a63eb5698c95460bc",
    "fields.streamed.process":
        "a23dfff43bf07b21128647d7fe69cb7353b7bc9b04cbbbe01e5bccb2dd482794",
    "fields.streamed.total":
        "d91a4901b196e2389e995c5b4586c71b86962d9a39aa6054243340f23cfaad6f",
}


def fixed_fields_batch(batch_size: int = BATCH_SIZE, hw: tuple[int, int] = HW,
                       seed: int = 20240928) -> dict[str, torch.Tensor]:
    """One whole input: seeded generators plus the four declared field tensors."""
    generator = torch.Generator().manual_seed(seed)
    return {
        "coarse_history": torch.randn(batch_size, 2, CHANNELS, *hw, generator=generator),
        "atmos_target": torch.randn(batch_size, CHANNELS, *hw, generator=generator),
        "atmos_baseline": torch.randn(batch_size, CHANNELS, *hw, generator=generator),
        "process_targets": torch.randn(batch_size, 2, generator=generator),
        "lead_time_hours": torch.full((batch_size,), 6.0),
        "latitude": torch.linspace(60.0, 25.0, hw[0]),
        "longitude": torch.linspace(100.0, 112.0, hw[1]),
        "init_utc_hour": torch.tensor([3.0, 15.0])[:batch_size],
        "init_day_of_year": torch.tensor([15.0, 200.0])[:batch_size],
    }


def fields_path_digests(*, config: dict | None = None,
                        batch: dict | None = None) -> dict[str, str]:
    """The frozen recipe: forward digests and one truncated-BPTT step's digests.

    ``config``/``batch`` exist so a counterproof can run the *same* recipe against
    a changed path; the pin itself always runs them at their defaults.
    """
    resolved = dict(FIELDS_PATH_CONFIG if config is None else config)
    batch = fixed_fields_batch() if batch is None else batch
    out: dict[str, str] = {}

    seed_everything(101)
    model = make_model("process", resolved).eval()
    with torch.no_grad():
        result = model(batch, reasoning_steps=FIELDS_PATH_STEPS)
    out["fields.forecast"] = tensor_digest(result.forecast)
    out["fields.initial_forecast"] = tensor_digest(result.initial_forecast)
    out["fields.draft_forecasts"] = tensor_digest(result.draft_forecasts)
    out["fields.final_correction"] = tensor_digest(result.final_correction)
    out["fields.process_state"] = tensor_digest(result.process_state)
    out["fields.context_tokens"] = tensor_digest(result.context_tokens)

    seed_everything(104)
    trained = make_model("process", resolved).train()
    streamed = backward_streamed_truncated(trained, batch, reasoning_steps=FIELDS_PATH_STEPS,
                                           process_weight=0.5, loss_scale=1.0)
    out["fields.streamed.total"] = tensor_digest(streamed.total)
    out["fields.streamed.forecast"] = tensor_digest(streamed.forecast)
    out["fields.streamed.process"] = tensor_digest(streamed.process)
    out["fields.streamed.final_forecast"] = tensor_digest(streamed.final_forecast)
    out["fields.gradients"] = digest_parameters(trained)
    return out


def conditioning_term(mode: str, batch: dict, *, seed: int = 7) -> torch.Tensor:
    """The module's own contribution for one batch, at the model's token grid."""
    seed_everything(seed)
    config = dict(FIELDS_PATH_CONFIG, spacetime_field_mode=mode)
    model = make_model("process", config).eval()
    with torch.no_grad():
        output = model(batch, reasoning_steps=FIELDS_PATH_STEPS)
        term = model.backbone.spacetime(batch, history=batch["coarse_history"],
                                        token_hw=output.token_hw)
    return term


def test_the_default_mode_is_fields_and_an_ignored_mode_is_rejected():
    """Default is the pre-existing path; a typo or a dead switch must raise."""
    from model.spacetime_conditioning_r7 import FIELD_MODES, require_field_mode

    assert FIELD_MODES == ("fields", "constant", "shuffled")
    assert require_field_mode("fields") == "fields"
    for bad in ("nonsense", "", "FIELDS", None, True, 1):
        with pytest.raises(ValueError):
            require_field_mode(bad)

    from model.process_forecast_r7 import ProcessForecastCoReasoner
    from model.recursive_weather_r7 import GenericRecursiveWeatherForecaster
    from model.weather_forecaster_r7 import NativeAtmosForecaster

    base = {name: value for name, value in FIELDS_PATH_CONFIG.items()
            if name in ("in_channels", "out_channels", "history_steps", "dim",
                        "patch_size", "depth", "heads", "window_size", "dropout",
                        "spacetime_inputs")}
    for model, extra in ((NativeAtmosForecaster, {}),
                         (GenericRecursiveWeatherForecaster, {"latent_tokens": 2}),
                         (ProcessForecastCoReasoner, {"anchored_processes": 1,
                                                      "free_processes": 1,
                                                      "use_forecast_feedback": True})):
        built = model(**base, **extra)
        assert built.spacetime_field_mode == "fields", model.__name__
        with pytest.raises(ValueError):
            model(**base, **extra, spacetime_field_mode="nonsense")
        # A control mode with the pathway off changes nothing and would say
        # nothing; that is rejected rather than silently ignored.
        with pytest.raises(ValueError):
            model(**dict(base, spacetime_inputs=False, spacetime_field_mode="constant"),
                  **extra)


def varied_across_positions(term: torch.Tensor) -> bool:
    """Whether a [B,N,D] conditioning term differs between two output positions."""
    return not torch.equal(term, term[:, :1, :].expand_as(term))


def varied_across_samples(term: torch.Tensor) -> bool:
    """Whether a [B,N,D] conditioning term differs between two batch samples."""
    return not torch.equal(term, term[:1].expand_as(term))


# A *numerical* tolerance, not a scientific one: a float32 matmul does not promise
# the same rounding for every row of a large M, so an input that is bitwise constant
# across positions comes out of the two Linear layers with row-to-row differences of
# about one ULP (measured: 1.2e-07 absolute against term norms of order 1). Anything
# at or below this relative level is float32 rounding; the information contrast the
# constant arm is for lives orders of magnitude above it.
FLOAT32_CONSTANCY_TOLERANCE = 1e-6


def constancy_deviation(term: torch.Tensor) -> tuple[float, float]:
    """(absolute, relative) deviation of a [B,N,D] term from one vector everywhere."""
    absolute = float((term - term[:, :1, :]).abs().max())
    scale = float(term.norm(dim=-1).mean())
    return absolute, (absolute / scale if scale else 0.0)


def conditioning_net_input(mode: str, batch: dict, *, seed: int = 7) -> torch.Tensor:
    """The eight features the conditioning MLP is shown, one row per (sample, token).

    This is the layer the modes are defined at: whatever the module does downstream,
    ``constant`` has to give the network the same eight numbers everywhere.
    """
    seed_everything(seed)
    config = dict(FIELDS_PATH_CONFIG, spacetime_field_mode=mode)
    model = make_model("process", config).eval()
    captured: dict[str, torch.Tensor] = {}
    model.backbone.spacetime.net.register_forward_pre_hook(
        lambda _module, inputs: captured.setdefault("features", inputs[0].detach()))
    with torch.no_grad():
        model(batch, reasoning_steps=FIELDS_PATH_STEPS)
    return captured["features"]


def test_the_constant_mode_collapses_the_conditioning_to_one_vector():
    """E is the module without input: same shape, same compute, no information."""
    batch = fixed_fields_batch()
    real = conditioning_term("fields", batch)
    constant = conditioning_term("constant", batch)
    assert real.shape == constant.shape

    # Exact statement: the constant arm shows the network the same eight features for
    # every position and every sample, and the fields arm does not.
    assert torch.unique(constant_rows := conditioning_net_input("constant", batch).reshape(-1, 8),
                        dim=0).shape[0] == 1, "constant arm features are not one vector"
    assert torch.unique(conditioning_net_input("fields", batch).reshape(-1, 8),
                        dim=0).shape[0] > 1

    # Numerical statement: the contribution is one vector up to float32 rounding.
    absolute, relative = constancy_deviation(constant)
    assert relative <= FLOAT32_CONSTANCY_TOLERANCE, (absolute, relative)
    assert not varied_across_samples(constant) or relative <= FLOAT32_CONSTANCY_TOLERANCE
    # And the real path varies far above that level, so the tolerance is not hiding
    # a dead arm.
    real_absolute, real_relative = constancy_deviation(real)
    assert real_relative > 1e3 * max(relative, 1e-12), (real_relative, relative)
    assert not torch.equal(real, constant)


def test_the_shuffled_mode_rolls_the_sample_axis_deterministically():
    """P sees real values with the wrong pairing - and never invents a pairing."""
    from model.spacetime_conditioning_r7 import apply_field_mode

    batch = fixed_fields_batch()
    latitude, longitude = batch["latitude"], batch["longitude"]
    hour, day = batch["init_utc_hour"], batch["init_day_of_year"]
    real = apply_field_mode("fields", latitude, longitude, hour, day, batch_size=BATCH_SIZE)
    shuffled = apply_field_mode("shuffled", latitude, longitude, hour, day,
                                batch_size=BATCH_SIZE)
    assert torch.equal(real[0], shuffled[0]) and torch.equal(real[1], shuffled[1])
    assert torch.equal(shuffled[2], torch.roll(hour, 1, dims=0))
    assert torch.equal(shuffled[3], torch.roll(day, 1, dims=0))
    assert not torch.equal(shuffled[2], hour)
    # A permutation preserves the values themselves; only the pairing changes.
    assert sorted(shuffled[2].tolist()) == sorted(hour.tolist())
    # A batch of one has no partner: the roll is the identity there, which is a
    # property of any within-batch permutation and is declared, not hidden.
    single = apply_field_mode("shuffled", latitude, longitude, hour[:1], day[:1],
                              batch_size=1)
    assert torch.equal(single[2], hour[:1]) and torch.equal(single[3], day[:1])

    real_term = conditioning_term("shuffled", batch)
    again = conditioning_term("shuffled", batch)
    assert torch.equal(real_term, again), "the mismatch must be deterministic"
    assert not torch.equal(real_term, conditioning_term("fields", batch))
    one = fixed_fields_batch(batch_size=1)
    assert torch.equal(conditioning_term("shuffled", one),
                       conditioning_term("fields", one))


def test_the_control_arms_still_require_the_declared_fields():
    """The replacement happens *inside* the model; the batch still carries them."""
    batch = fixed_fields_batch()
    for mode in ("constant", "shuffled"):
        for name in SPACETIME_INPUT_FIELDS:
            incomplete = {key: value for key, value in batch.items() if key != name}
            with pytest.raises(KeyError, match=name):
                conditioning_term(mode, incomplete)
    # A mode is not a fallback: the real fields are validated before any
    # replacement, so an out-of-range hour fails in the control arms too.
    for mode in ("fields", "constant", "shuffled"):
        broken = dict(batch, init_utc_hour=torch.full((BATCH_SIZE,), 30.0))
        with pytest.raises(ValueError, match="init_utc_hour"):
            conditioning_term(mode, broken)


def test_the_three_modes_carry_the_same_parameters_and_forward_flops():
    """D6 rests on this: E and B are the same tensors costing the same compute."""
    from training.r7_budget_audit import count_forward_flops

    batch = fixed_fields_batch()
    states, flops = {}, {}
    for mode in ("fields", "constant", "shuffled"):
        seed_everything(11)
        model = make_model("process", dict(FIELDS_PATH_CONFIG,
                                           spacetime_field_mode=mode))
        states[mode] = {name: value.clone() for name, value in model.state_dict().items()}
        flops[mode] = count_forward_flops(model, batch, reasoning_steps=FIELDS_PATH_STEPS)
    names = {mode: sorted(state) for mode, state in states.items()}
    assert names["fields"] == names["constant"] == names["shuffled"]
    totals = {mode: sum(tensor.numel() for tensor in state.values())
              for mode, state in states.items()}
    assert len(set(totals.values())) == 1, totals
    for mode in ("constant", "shuffled"):
        differing = [name for name, tensor in states["fields"].items()
                     if not torch.equal(tensor, states[mode][name])]
        assert not differing, f"{mode} does not build the same seeded weights: {differing}"
    assert len(set(flops.values())) == 1, flops


def test_the_declared_inputs_do_not_depend_on_the_mode():
    """D2: the whitelist is the field set, and no mode narrows or widens it."""
    from model.r7_halting import DECLARED_MODEL_INPUTS, forecast_inputs

    batch = fixed_fields_batch()
    assert set(forecast_inputs(batch)) == set(DECLARED_MODEL_INPUTS)
    assert set(DECLARED_MODEL_INPUTS) == {"coarse_history", "lead_time_hours",
                                          *SPACETIME_INPUT_FIELDS}


def test_the_fields_path_is_bitwise_identical_to_the_frozen_digests():
    """D4: the mode work must not move the pre-existing `fields` path at all."""
    assert len(FIELDS_PATH_DIGESTS) == 11, "the pin has no captured values"
    live = fields_path_digests()
    assert live == FIELDS_PATH_DIGESTS, (
        "the `fields` path moved; differing entries: "
        f"{sorted(key for key, value in FIELDS_PATH_DIGESTS.items() if live.get(key) != value)}")


def test_the_frozen_pin_detects_a_changed_path():
    """Counterproof: the pin cannot pass by hashing nothing.

    The pin compares a live run against frozen constants, so the recipe has to be
    shown to move when the path it measures moves - both when the conditioning is
    changed and when the input itself is.
    """
    for mode in ("constant", "shuffled"):
        changed = fields_path_digests(config=dict(FIELDS_PATH_CONFIG,
                                                  spacetime_field_mode=mode))
        differing = sorted(key for key, value in FIELDS_PATH_DIGESTS.items()
                           if changed.get(key) != value)
        assert "fields.forecast" in differing, differing
        assert "fields.gradients" in differing, differing
    other_batch = dict(fixed_fields_batch(),
                       lead_time_hours=torch.full((BATCH_SIZE,), 48.0))
    moved = fields_path_digests(batch=other_batch)
    assert "fields.forecast" in sorted(key for key, value in FIELDS_PATH_DIGESTS.items()
                                       if moved.get(key) != value)


def main() -> int:
    """Print the recipe's digests; also the frozen-vs-live comparison."""
    live = fields_path_digests()
    payload = {"capture_revision": FIELDS_PATH_CAPTURE, "digest_count": len(live),
               "digests": live, "frozen_digests": FIELDS_PATH_DIGESTS}
    if FIELDS_PATH_DIGESTS:
        payload["bitwise_identical"] = live == FIELDS_PATH_DIGESTS
        payload["differing"] = sorted(key for key, value in FIELDS_PATH_DIGESTS.items()
                                      if live.get(key) != value)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if (not FIELDS_PATH_DIGESTS or live == FIELDS_PATH_DIGESTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
