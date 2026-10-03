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

1. **D4, the regression pin.** The ``fields`` path is pinned by running one fixed
   recipe twice *in the same process*: against the pre-change revision read out of
   the git object store (``git archive ac6a3ef``, round three's start SHA) and
   against the live working tree. Same process, same interpreter, same threading
   configuration: the comparison is of raw little-endian float bytes and no
   tolerance is applied. The digests that recipe produced on the development
   machine *before* the modes existed are recorded in ``FIELDS_PATH_CAPTURE`` as
   provenance for the evidence document - but they are **not** asserted as
   constants, because they are not portable: measured on the capturing machine,
   turning oneDNN off moves 8 of the 11 and running one thread moves
   ``fields.gradients``. A digest frozen on one machine cannot be asserted on
   another (CI is another machine), so the pin is the two-tree comparison and the
   record is the record. ``test_the_frozen_pin_detects_a_changed_path`` perturbs
   the frozen copy and requires the digests to move, so a recipe that hashed
   nothing cannot look like a pass.

2. **D1, the modes.** Behaviour, determinism, the single-sample-batch property of
   ``shuffled`` (declared, not silent), invalid-mode rejection, and the two
   invariants D6 rests on: all three modes carry the *same* parameter tensors, and
   ``constant`` and ``fields`` cost the *same* forward FLOPs.

Run as a script to print the digests (that is how the recorded values below were
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
from test_r7_switched_path_equivalence import (SNAPSHOT_FILES, archived_sources,
    assert_import_rewrite_is_the_only_edit, digest_parameters, import_frozen,
    live_package, rewrite_model_imports, tensor_digest)

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
# The revision the pin is taken against: round three's start SHA, i.e. the tree as it
# stood before the field modes existed. Reachable in CI because ci.yml checks out with
# fetch-depth: 0, and the test fails rather than skipping if it is not.
FIELDS_PATH_PRE_CHANGE_SHA = "ac6a3ef486d56d43e11261b62e7294872d1e5eae"
# The frozen copy needs two modules the round-one/round-two snapshot list predates:
# the conditioning module itself, and the positional process readout that the
# archived process arm imports. Listing them keeps the snapshot to the modules the
# recipe actually executes.
FIELDS_PATH_FILES = tuple(sorted(set(SNAPSHOT_FILES)
                                 | {"model/spacetime_conditioning_r7.py",
                                    "model/process_readout_r7.py"}))
# What the recipe produced on the development machine *before* the modes existed
# (`git rev-parse HEAD` = FIELDS_PATH_PRE_CHANGE_SHA at capture time, with the working
# tree clean apart from this file). Recorded for the evidence document; deliberately
# NOT asserted, because these numbers are machine-local: with oneDNN disabled 8 of the
# 11 move, and under a single thread `fields.gradients` moves. The assertion that runs
# in CI is the in-process comparison against the frozen revision below.
FIELDS_PATH_CAPTURE = FIELDS_PATH_PRE_CHANGE_SHA
FIELDS_PATH_CAPTURE_DIGESTS: dict[str, str] = {
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


def fields_path_digests(package, *, config: dict | None = None,
                        batch: dict | None = None) -> dict[str, str]:
    """The frozen recipe: forward digests and one truncated-BPTT step's digests.

    ``package`` supplies the implementation under test (``make_model``,
    ``seed_everything``, ``backward_streamed_truncated``), so the same text runs
    against the live tree and against the archived pre-change revision in one
    process. ``config``/``batch`` exist so a counterproof can run the recipe
    against a changed path; the pin always runs them at their defaults.
    """
    resolved = dict(FIELDS_PATH_CONFIG if config is None else config)
    batch = fixed_fields_batch() if batch is None else batch
    out: dict[str, str] = {}

    package.seed_everything(101)
    model = package.make_model("process", resolved).eval()
    with torch.no_grad():
        result = model(batch, reasoning_steps=FIELDS_PATH_STEPS)
    out["fields.forecast"] = tensor_digest(result.forecast)
    out["fields.initial_forecast"] = tensor_digest(result.initial_forecast)
    out["fields.draft_forecasts"] = tensor_digest(result.draft_forecasts)
    out["fields.final_correction"] = tensor_digest(result.final_correction)
    out["fields.process_state"] = tensor_digest(result.process_state)
    out["fields.context_tokens"] = tensor_digest(result.context_tokens)

    package.seed_everything(104)
    trained = package.make_model("process", resolved).train()
    streamed = package.backward_streamed_truncated(
        trained, batch, reasoning_steps=FIELDS_PATH_STEPS, process_weight=0.5, loss_scale=1.0)
    out["fields.streamed.total"] = tensor_digest(streamed.total)
    out["fields.streamed.forecast"] = tensor_digest(streamed.forecast)
    out["fields.streamed.process"] = tensor_digest(streamed.process)
    out["fields.streamed.final_forecast"] = tensor_digest(streamed.final_forecast)
    out["fields.gradients"] = digest_parameters(trained)
    return out


def materialize_frozen(files: dict[str, bytes], root: Path, *, model_package: str,
                       training_package: str,
                       edits: dict[str, tuple[str, str]] | None = None):
    """Write an archived tree under new package names and import it.

    The same construction as the equivalence test's ``materialize``, with one
    difference: the file list is a parameter, because this pin archives a later
    revision that also contains ``model/spacetime_conditioning_r7.py``.
    """
    root.mkdir(parents=True, exist_ok=True)
    names = {"model": model_package, "training": training_package}
    missing = [name for name in FIELDS_PATH_FILES if name not in files]
    if missing:
        raise AssertionError(f"the archived revision does not contain {missing}")
    for archived_name in FIELDS_PATH_FILES:
        top, _, relative = archived_name.partition("/")
        original = files[archived_name].decode("utf-8")
        rewritten = (rewrite_model_imports(original, model_package) if top == "training"
                     else original)
        for target, (before, after) in (edits or {}).items():
            if target == archived_name:
                if before not in rewritten:
                    raise AssertionError(f"perturbation target {before!r} absent from "
                                         f"{archived_name}")
                rewritten = rewritten.replace(before, after, 1)
        destination = root / names[top] / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rewritten, encoding="utf-8")
        if top == "training":
            assert_import_rewrite_is_the_only_edit(original, rewritten, model_package)
    for package in names.values():
        (root / package / "__init__.py").write_text("", encoding="utf-8")
    return import_frozen(root, model_package=model_package, training_package=training_package)


def frozen_fields_package(tmp_path: Path, *, name: str = "pre_change_fields",
                          edits: dict[str, tuple[str, str]] | None = None):
    root = tmp_path / name
    return materialize_frozen(archived_sources(FIELDS_PATH_PRE_CHANGE_SHA), root,
                              model_package=f"{name}_model",
                              training_package=f"{name}_training", edits=edits)


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
# about one ULP (measured: 1.2e-07 absolute against term norms of order 1, and 0.0 on
# the CUDA path used for the run). Two orders of margin are left for another BLAS,
# which still leaves this three orders below the varying arms: the fields mode's own
# deviation is ~9e-2 relative on this recipe.
FLOAT32_CONSTANCY_TOLERANCE = 1e-5


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
    assert real_relative > 100 * max(relative, 1e-9), (real_relative, relative)
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
    from model.spacetime_conditioning_r7 import CALENDAR_INPUT_FIELDS
    from model.known_context_r7 import HISTORY_CONTEXT_FIELDS

    batch = fixed_fields_batch()
    expected = {"coarse_history", "lead_time_hours", *SPACETIME_INPUT_FIELDS}
    assert set(forecast_inputs(batch)) == expected
    assert set(DECLARED_MODEL_INPUTS) == expected | set(CALENDAR_INPUT_FIELDS) | set(HISTORY_CONTEXT_FIELDS)
    with_calendar = dict(batch, init_calendar_year=torch.full_like(batch["init_day_of_year"], 2016),
                         history_offsets_hours=torch.tensor([[-6., 0.]]).expand(BATCH_SIZE, -1))
    assert set(forecast_inputs(with_calendar)) == set(DECLARED_MODEL_INPUTS)
    assert "init_year" not in forecast_inputs(dict(with_calendar, init_year=torch.tensor(2001)))


def test_the_fields_path_is_bitwise_identical_to_the_frozen_revision(tmp_path):
    """D4: the mode work must not move the pre-existing `fields` path at all.

    The comparison is between two implementations in one process - the archived
    pre-change revision and the live tree - so it measures the change and not the
    machine: no digest is frozen across environments, and no tolerance is applied.
    """
    frozen = frozen_fields_package(tmp_path)
    before = fields_path_digests(frozen)
    if not str(Path(frozen.model_file)).startswith(str((tmp_path / "pre_change_fields").resolve())):
        raise AssertionError("the frozen run did not import the archived tree; the "
                             "comparison would be against the live tree itself")
    live = fields_path_digests(live_package())
    assert len(before) == 11, before
    assert before == live, (
        "the `fields` path moved relative to the pre-change revision; differing "
        f"entries: {sorted(k for k, v in before.items() if live.get(k) != v)}")


def test_the_frozen_pin_detects_a_changed_path(tmp_path):
    """Counterproof: the pin cannot pass by hashing nothing.

    Two independent ways for the pins to move, both real: a perturbed copy of the
    frozen implementation must produce different digests from the live one, and the
    live tree's own control modes must produce different digests from its `fields`
    mode.
    """
    perturbed = frozen_fields_package(
        tmp_path, name="perturbed_fields",
        edits={"model/coarse_forecast.py": ("hidden=hidden or max(32,dim//2)",
                                           "hidden=hidden or max(64,dim//2)")})
    changed = fields_path_digests(perturbed)
    live = fields_path_digests(live_package())
    differing = sorted(key for key, value in changed.items() if live.get(key) != value)
    assert "fields.forecast" in differing, differing
    assert "fields.gradients" in differing, differing

    package = live_package()
    for mode in ("constant", "shuffled"):
        mode_changed = fields_path_digests(
            package, config=dict(FIELDS_PATH_CONFIG, spacetime_field_mode=mode))
        moved = sorted(key for key, value in mode_changed.items() if live.get(key) != value)
        assert "fields.forecast" in moved, moved
        assert "fields.gradients" in moved, moved
    other_batch = dict(fixed_fields_batch(),
                       lead_time_hours=torch.full((BATCH_SIZE,), 48.0))
    moved = fields_path_digests(package, batch=other_batch)
    assert "fields.forecast" in sorted(key for key, value in live.items()
                                       if moved.get(key) != value)


def main() -> int:
    """Print the recipe's digests, the recorded capture and the live comparison."""
    live = fields_path_digests(live_package())
    payload = {"capture_revision": FIELDS_PATH_PRE_CHANGE_SHA, "digest_count": len(live),
               "digests": live, "captured_digests": FIELDS_PATH_CAPTURE_DIGESTS,
               "capture_matches_this_machine": live == FIELDS_PATH_CAPTURE_DIGESTS,
               "capture_differing": sorted(
                   key for key, value in FIELDS_PATH_CAPTURE_DIGESTS.items()
                   if live.get(key) != value)}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
